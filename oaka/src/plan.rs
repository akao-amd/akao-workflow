//! `plan.toml` in a Work Directory, its validation, and `plan.lock.toml`.

use crate::profile::{targets_text, Arg, Library, Profile, Target};
use crate::stack::{self, Stacks};
use crate::sys;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use toml::Table;

pub const PLAN: &str = "plan.toml";
pub const LOCK: &str = "plan.lock.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    #[serde(default, rename = "server")]
    pub servers: Vec<ServerSpec>,
    #[serde(default, rename = "client")]
    pub clients: Vec<ClientSpec>,
    /// Packages from the library's stacks.toml to install from a tree before the servers.
    #[serde(default)]
    pub stack: BTreeMap<String, StackSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackSpec {
    /// A git worktree of the package's repo; created from it when missing.
    pub tree: String,
    /// One revision.  None with no commits/bisect: install the tree as it is.
    pub commit: Option<String>,
    /// The whole plan runs once per revision, in order (A-B-A: [A, B, A]).
    pub commits: Option<Vec<String>>,
    /// `git bisect run` over good..bad, with the plan's gates as the verdict.
    pub bisect: Option<BisectSpec>,
    pub clean: Option<CleanWhen>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BisectSpec {
    pub good: String,
    pub bad: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CleanWhen {
    /// Keep caches: what A-B-A needs, since a cache reused across commits must show up.
    Never,
    /// Delete the package's clean paths before every install.
    BeforeInstall,
}

impl CleanWhen {
    pub fn as_str(self) -> &'static str {
        match self {
            CleanWhen::Never => "never",
            CleanWhen::BeforeInstall => "before-install",
        }
    }
}

/// One package of the plan's stack, resolved against stacks.toml.
#[derive(Debug)]
pub struct StackPkg {
    pub name: String,
    pub tree: String,
    /// The install recipe for the plan's GPU arch.
    pub install: String,
    /// The fixed revision, or None for "the tree as it is" (and for the varying package).
    pub commit: Option<String>,
    pub clean: CleanWhen,
}

/// How the stack varies over the run.
#[derive(Debug, PartialEq)]
pub enum Vary {
    /// One stack (or none): the plan runs once.
    No,
    /// The plan runs once per revision of `package`.
    Commits { package: String, commits: Vec<String> },
    /// `git bisect run` over good..bad of `package`.
    Bisect { package: String, good: String, bad: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerSpec {
    pub name: String,
    pub profile: String,
    #[serde(default)]
    pub gpus: Vec<u32>,
    pub model: Option<String>,
    pub port: Option<u16>,
    #[serde(default)]
    pub env: Table,
    #[serde(default)]
    pub args: Table,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum ClientSpec {
    /// sgl-eval GSM8K; the run stops when the score is below min_score.
    #[serde(rename = "gsm8k")]
    Gsm8k {
        server: String,
        #[serde(default)]
        thinking: bool,
        #[serde(default = "default_min_score")]
        min_score: f64,
    },
    /// InferenceX fixed-sequence-length throughput points.
    #[serde(rename = "fixed-seq")]
    FixedSeq {
        server: String,
        isl_osl: Vec<[u32; 2]>,
        conc: Vec<u32>,
        #[serde(default = "default_range_ratio")]
        range_ratio: f64,
        #[serde(default = "default_repeats")]
        repeats: u32,
        off_spec: Option<OffSpec>,
        /// Gate: every point's median output tok/s over its repeats must reach this.
        min_output_tok_s: Option<f64>,
    },
}

/// Deviations from the InferenceX client policy.  Results are marked OFFSPEC.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OffSpec {
    /// InferenceX uses 10 prompts per unit of concurrency.
    pub prompts_per_conc: Option<u32>,
}

fn default_min_score() -> f64 {
    0.90
}
fn default_range_ratio() -> f64 {
    0.8
}
fn default_repeats() -> u32 {
    1
}

pub const PROMPTS_PER_CONC: u32 = 10;

impl ClientSpec {
    pub fn server(&self) -> &str {
        match self {
            ClientSpec::Gsm8k { server, .. } | ClientSpec::FixedSeq { server, .. } => server,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            ClientSpec::Gsm8k { .. } => "gsm8k",
            ClientSpec::FixedSeq { .. } => "fixed-seq",
        }
    }
}

/// A server with everything resolved except its port.
#[derive(Debug)]
pub struct Server {
    pub name: String,
    /// The profile the plan names, before the plan's overrides.
    pub base: Profile,
    /// base + the plan's env/args overrides.
    pub effective: Profile,
    pub model: String,
    pub gpus: Vec<u32>,
    pub tp: u32,
    pub port: Option<u16>,
}

impl Server {
    /// The name clients must send as `model`: --served-model-name if set, else the path.
    pub fn served_name(&self) -> String {
        match self.effective.arg("served-model-name") {
            Some(Arg::Value(s)) => s.clone(),
            _ => self.model.clone(),
        }
    }
}

/// The result of `oaka check`: the plan resolved against the library and the machine.
pub struct Checked {
    pub dir: PathBuf,
    pub plan: Plan,
    pub servers: Vec<Server>,
    /// The plan's packages, in stacks.toml (= install) order.
    pub stack: Vec<StackPkg>,
    /// The GPU arch of the plan's GPUs, which picks per-arch recipes (None: unknown/mixed).
    pub arch: Option<String>,
    pub vary: Vary,
    /// The library's stacks.toml.
    pub stacks: Stacks,
    /// Non-fatal findings, e.g. a model directory that does not exist here.
    pub warnings: Vec<String>,
    /// The machine the plan was checked against.
    pub machine: Machine,
}

pub fn valid_server_name(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_alphanumeric())
        && c.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

pub fn load(dir: &Path) -> Result<Plan> {
    let path = dir.join(PLAN);
    if !path.exists() {
        bail!("no {} in {}; write one with `oaka draft`", PLAN, dir.display());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// What a plan is checked against: this container's GPUs and ROCm.
#[derive(Debug, Default)]
pub struct Machine {
    /// GPU archs in HIP device order; None when they cannot be seen.
    pub gpus: Option<Vec<String>>,
    /// ROCm version, e.g. "10.0.0"; None when it cannot be found.
    pub rocm: Option<String>,
}

impl Machine {
    pub fn probe() -> Machine {
        Machine {
            gpus: sys::gpus(),
            rocm: sys::rocm().map(|(v, _)| v),
        }
    }
}

/// Validate the plan in `dir`.  Every error names the field to fix.
pub fn check(dir: &Path, lib: &Library) -> Result<Checked> {
    let plan = load(dir)?;
    check_plan(dir, plan, lib, &Machine::probe())
}

pub fn check_plan(dir: &Path, plan: Plan, lib: &Library, machine: &Machine) -> Result<Checked> {
    let mut warnings = Vec::new();
    if plan.servers.is_empty() {
        bail!("the plan has no [[server]]");
    }
    let mut servers: Vec<Server> = Vec::new();
    for (i, s) in plan.servers.iter().enumerate() {
        let at = format!("server #{} ({:?})", i + 1, s.name);
        if !valid_server_name(&s.name) {
            bail!("{at}: name must match [A-Za-z0-9][A-Za-z0-9_-]*");
        }
        if servers.iter().any(|o| o.name == s.name) {
            bail!("{at}: duplicate server name");
        }
        if s.profile.is_empty() {
            let names = lib.list()?;
            bail!(
                "{at}: profile is empty; pick one of: {}",
                if names.is_empty() {
                    "<none in library>".into()
                } else {
                    names.join(" ")
                }
            );
        }
        let base = lib.resolve(&s.profile).with_context(|| at.clone())?;
        let mut effective = base.clone();
        effective
            .overlay(&s.env, &s.args)
            .with_context(|| format!("{at}: env/args overrides"))?;
        let model = s
            .model
            .clone()
            .or_else(|| base.model.clone())
            .with_context(|| format!("{at}: profile {} has no model; set `model` in the server", s.profile))?;
        if !Path::new(&model).exists() {
            warnings.push(format!("{at}: model {model} does not exist on this machine"));
        }
        if s.gpus.is_empty() {
            bail!("{at}: gpus is empty; list the GPU indices this server may use, e.g. gpus = [0]");
        }
        let mut sorted = s.gpus.clone();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != s.gpus.len() {
            bail!("{at}: gpus lists a GPU twice");
        }
        let tp = match base.tp {
            Some(tp) if tp as usize != s.gpus.len() => {
                bail!(
                    "{at}: profile {} pins tp = {tp} but gpus lists {}",
                    s.profile,
                    s.gpus.len()
                )
            }
            Some(tp) => tp,
            None => s.gpus.len() as u32,
        };
        if let Some(gpus) = &machine.gpus {
            if let Some(g) = s.gpus.iter().find(|g| **g as usize >= gpus.len()) {
                match gpus.len() {
                    0 => bail!("{at}: GPU {g} does not exist; this machine has no GPUs"),
                    n => bail!("{at}: GPU {g} does not exist; this machine has {n} GPUs (0-{})", n - 1),
                }
            }
        }
        check_targets(&at, &s.profile, &base, &s.gpus, machine)?;
        for o in &servers {
            if let Some(g) = s.gpus.iter().find(|g| o.gpus.contains(g)) {
                bail!("{at}: GPU {g} is already used by server {}", o.name);
            }
        }
        if let Some(p) = s.port {
            if p < 1024 {
                bail!("{at}: port {p} is privileged");
            }
            if servers.iter().any(|o| o.port == Some(p)) {
                bail!("{at}: port {p} is already used by another server");
            }
        }
        servers.push(Server {
            name: s.name.clone(),
            base,
            effective,
            model,
            gpus: s.gpus.clone(),
            tp,
            port: s.port,
        });
    }
    for (i, c) in plan.clients.iter().enumerate() {
        let at = format!("client #{} ({})", i + 1, c.kind());
        if !servers.iter().any(|s| s.name == c.server()) {
            bail!("{at}: server {:?} is not in the plan", c.server());
        }
        match c {
            ClientSpec::Gsm8k { min_score, .. } => {
                if !(0.0..=1.0).contains(min_score) {
                    bail!("{at}: min_score must be within 0..1");
                }
            }
            ClientSpec::FixedSeq {
                isl_osl,
                conc,
                range_ratio,
                repeats,
                off_spec,
                min_output_tok_s,
                ..
            } => {
                if isl_osl.is_empty() || isl_osl.iter().flatten().any(|n| *n == 0) {
                    bail!("{at}: isl_osl must list positive [isl, osl] pairs");
                }
                if conc.is_empty() || conc.contains(&0) {
                    bail!("{at}: conc must list positive concurrencies");
                }
                if !(*range_ratio > 0.0 && *range_ratio <= 1.0) {
                    bail!("{at}: range_ratio must be within (0, 1]");
                }
                if *repeats == 0 {
                    bail!("{at}: repeats must be at least 1");
                }
                if let Some(OffSpec {
                    prompts_per_conc: Some(0),
                }) = off_spec
                {
                    bail!("{at}: off_spec.prompts_per_conc must be positive");
                }
                if min_output_tok_s.is_some_and(|m| !m.is_finite() || m <= 0.0) {
                    bail!("{at}: min_output_tok_s must be positive");
                }
            }
        }
    }
    // The arch of the GPUs the plan uses picks per-arch recipes.
    let arch = machine.gpus.as_ref().and_then(|gpus| {
        let mut used: Vec<&String> = servers
            .iter()
            .flat_map(|s| &s.gpus)
            .map(|g| &gpus[*g as usize])
            .collect();
        used.dedup();
        (used.len() == 1).then(|| used[0].clone())
    });
    let stacks = stack::load(lib)?;
    let (stack, vary) = check_stack(&plan, &stacks, arch.as_deref(), &mut warnings)?;
    Ok(Checked {
        dir: dir.to_path_buf(),
        plan,
        servers,
        stack,
        arch,
        vary,
        stacks,
        warnings,
        machine: Machine {
            gpus: machine.gpus.clone(),
            rocm: machine.rocm.clone(),
        },
    })
}

/// The profile's `arch` targets against this container: for every GPU of the server, one
/// target must fit its arch and the container's ROCm version.  Unknown GPUs leave only
/// the ROCm part to judge; an unknown ROCm version fails a target that names one.
fn check_targets(at: &str, profile: &str, base: &Profile, gpus: &[u32], machine: &Machine) -> Result<()> {
    if base.arch.is_empty() {
        return Ok(());
    }
    let wanted = targets_text(&base.arch);
    let on: Vec<(Option<u32>, Option<&str>)> = match &machine.gpus {
        Some(archs) => gpus
            .iter()
            .map(|g| (Some(*g), Some(archs[*g as usize].as_str())))
            .collect(),
        None => vec![(None, None)],
    };
    for (g, arch) in on {
        let by_arch: Vec<&Target> = base
            .arch
            .iter()
            .filter(|t| arch.is_none_or(|a| t.fits_arch(a)))
            .collect();
        if by_arch.is_empty() {
            bail!(
                "{at}: GPU {} is {}, but profile {profile} is for {wanted}",
                g.unwrap(),
                arch.unwrap()
            );
        }
        let fits: Vec<Option<bool>> = by_arch.iter().map(|t| t.fits_rocm(machine.rocm.as_deref())).collect();
        if fits.contains(&Some(true)) {
            continue;
        }
        match &machine.rocm {
            None => bail!(
                "{at}: profile {profile} is for {wanted}, and this container's ROCm version is unknown \
                 (no .info/version under $ROCM_PATH, $ROCM_HOME or /opt/rocm; set OAKA_ROCM)"
            ),
            Some(v) => bail!(
                "{at}: this container has ROCm {v}{}, but profile {profile} is for {wanted}; \
                 pick a profile for it (oaka profile ls) or a matching container",
                arch.map(|a| format!(" on {a}")).unwrap_or_default()
            ),
        }
    }
    Ok(())
}

/// A revision as git takes it; quoted where it is used, but never an option.
fn valid_rev(r: &str) -> bool {
    !r.is_empty() && !r.starts_with('-') && !r.chars().any(|c| c.is_whitespace() || c.is_control())
}

fn check_stack(
    plan: &Plan,
    stacks: &Stacks,
    arch: Option<&str>,
    warnings: &mut Vec<String>,
) -> Result<(Vec<StackPkg>, Vary)> {
    for name in plan.stack.keys() {
        if stacks.get(name).is_none() {
            bail!(
                "[stack.{name}]: no package {name} in {}; it has: {}",
                stacks.path.display(),
                if stacks.packages.is_empty() {
                    "<none>".to_string()
                } else {
                    stacks.names().join(" ")
                }
            );
        }
    }
    let mut pkgs = Vec::new();
    let mut vary = Vary::No;
    for (name, pkg) in &stacks.packages {
        let Some(s) = plan.stack.get(name) else {
            continue;
        };
        let at = format!("[stack.{name}]");
        let install = pkg
            .recipe(arch)
            .with_context(|| format!("{at}: {name} in {}", stacks.path.display()))?;
        if !s.tree.starts_with('/') || s.tree.contains(['\n', '\t']) {
            bail!("{at}: tree must be an absolute path, e.g. /<year>/nocopy/{name}-<task>");
        }
        let tree = s.tree.trim_end_matches('/').to_string();
        if tree == pkg.repo.trim_end_matches('/') {
            bail!(
                "{at}: tree is the package's repo {}; use a worktree of it \
                 (oaka creates one when the tree does not exist)",
                pkg.repo
            );
        }
        let given = [s.commit.is_some(), s.commits.is_some(), s.bisect.is_some()];
        if given.iter().filter(|g| **g).count() > 1 {
            bail!("{at}: set one of commit, commits, bisect");
        }
        let mut revs: Vec<&String> = s.commit.iter().collect();
        let mut default_clean = CleanWhen::Never;
        if let Some(commits) = &s.commits {
            if commits.len() < 2 {
                bail!("{at}: commits needs at least two revisions, e.g. [\"<A>\", \"<B>\", \"<A>\"]; one is `commit`");
            }
            revs.extend(commits);
        }
        if let Some(b) = &s.bisect {
            if b.good == b.bad {
                bail!("{at}: bisect.good and bisect.bad are the same revision");
            }
            revs.extend([&b.good, &b.bad]);
            default_clean = CleanWhen::BeforeInstall;
        }
        if revs.iter().any(|r| r.is_empty()) {
            bail!("{at}: a revision is empty; fill it in, or remove commit to install the tree as it is");
        }
        if let Some(r) = revs.iter().find(|r| !valid_rev(r)) {
            bail!("{at}: {r:?} is not a git revision");
        }
        let exists = Path::new(&tree).exists();
        if !exists && revs.is_empty() {
            bail!(
                "{at}: tree {tree} does not exist; set commit to create it as a worktree of {}",
                pkg.repo
            );
        }
        if !exists && !Path::new(&pkg.repo).is_dir() {
            warnings.push(format!(
                "{at}: neither the tree {tree} nor the repo {} exists on this machine",
                pkg.repo
            ));
        }
        let varying = s.commits.is_some() || s.bisect.is_some();
        if varying && vary != Vary::No {
            bail!("{at}: only one package may vary (commits or bisect) in a plan");
        }
        if let Some(commits) = &s.commits {
            vary = Vary::Commits {
                package: name.clone(),
                commits: commits.clone(),
            };
        }
        if let Some(b) = &s.bisect {
            vary = Vary::Bisect {
                package: name.clone(),
                good: b.good.clone(),
                bad: b.bad.clone(),
            };
        }
        pkgs.push(StackPkg {
            name: name.clone(),
            tree,
            install: install.to_string(),
            commit: if varying { None } else { s.commit.clone() },
            clean: s.clean.unwrap_or(default_clean),
        });
    }
    if vary != Vary::No && plan.clients.is_empty() {
        bail!("[stack]: a plan that varies the stack needs at least one [[client]] to measure with");
    }
    if let Vary::Bisect { package, .. } = &vary {
        let mut gated = false;
        for c in &plan.clients {
            match c {
                ClientSpec::Gsm8k { .. } => gated = true,
                ClientSpec::FixedSeq {
                    min_output_tok_s,
                    isl_osl,
                    conc,
                    ..
                } => {
                    gated |= min_output_tok_s.is_some();
                    if isl_osl.len() * conc.len() > 1 {
                        warnings.push(format!(
                            "[stack.{package}]: bisect measures every point of every client at each step; \
                             one cheap, high-contrast point is usually enough"
                        ));
                    }
                }
            }
        }
        if !gated {
            bail!(
                "[stack.{package}]: bisect needs a gate to decide good/bad: a gsm8k client (min_score) \
                 or a fixed-seq client with min_output_tok_s"
            );
        }
    }
    Ok((pkgs, vary))
}

/// Values compile chose automatically; kept so a recompile does not move them.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Lock {
    #[serde(default)]
    pub port: BTreeMap<String, u16>,
}

pub fn load_lock(dir: &Path) -> Result<Lock> {
    let path = dir.join(LOCK);
    if !path.exists() {
        return Ok(Lock::default());
    }
    let text = fs::read_to_string(&path)?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn save_lock(dir: &Path, lock: &Lock) -> Result<()> {
    let text = format!(
        "# Written by `oaka compile`: values it chose for the plan.  Delete to choose again.\n{}",
        toml::to_string(lock)?
    );
    fs::write(dir.join(LOCK), text).context("writing plan.lock.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> (Library, PathBuf) {
        let dir = std::env::temp_dir().join(format!("oaka-plan-{}-{}", std::process::id(), fastrand::u32(..)));
        let lib = Library { root: dir.clone() };
        fs::create_dir_all(lib.profiles_dir().join("m")).unwrap();
        fs::write(
            lib.profile_path("m/base"),
            "model = '/model/m'\narch = ['gfx950']\n[args]\npage-size = 1\n",
        )
        .unwrap();
        fs::write(lib.profile_path("m/tp2"), "extends = 'm/base'\ntp = 2\n").unwrap();
        fs::write(
            lib.profile_path("m/new"),
            "extends = 'm/base'\narch = [['gfx950', '10.1']]\n",
        )
        .unwrap();
        (lib, dir)
    }

    fn checked(lib: &Library, text: &str, machine: Machine) -> Result<Checked> {
        check_plan(Path::new("/w"), toml::from_str(text)?, lib, &machine)
    }

    fn eight(arch: &str) -> Machine {
        Machine {
            gpus: Some(vec![arch.to_string(); 8]),
            rocm: None,
        }
    }

    fn rocm(v: &str) -> Machine {
        Machine {
            gpus: None,
            rocm: Some(v.to_string()),
        }
    }

    const OK: &str = r#"
[[server]]
name = "a"
profile = "m/base"
gpus = [3]
[server.args]
page-size = 64
[[client]]
kind = "fixed-seq"
server = "a"
isl_osl = [[1024, 1024]]
conc = [4, 8]
[[client]]
kind = "gsm8k"
server = "a"
"#;

    #[test]
    fn valid_plan() {
        let (lib, dir) = lib();
        let c = checked(&lib, OK, eight("gfx950")).unwrap();
        let s = &c.servers[0];
        assert_eq!(s.tp, 1);
        assert_eq!(s.effective.launch_args(), vec![vec!["--page-size", "64"]]);
        assert_eq!(s.base.launch_args(), vec![vec!["--page-size", "1"]]);
        assert_eq!(s.served_name(), "/model/m");
        match &c.plan.clients[0] {
            ClientSpec::FixedSeq {
                range_ratio, repeats, ..
            } => assert_eq!((*range_ratio, *repeats), (0.8, 1)),
            _ => panic!(),
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects() {
        let (lib, dir) = lib();
        let bad = |text: &str, machine, needle: &str| {
            let e = format!("{:#}", checked(&lib, text, machine).err().expect(needle));
            assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        };
        bad(OK, eight("gfx1250"), "is for gfx950");
        bad(&OK.replace("[3]", "[9]"), eight("gfx950"), "does not exist");
        bad(&OK.replace("[3]", "[]"), Machine::default(), "gpus is empty");
        bad(&OK.replace("m/base", "m/tp2"), Machine::default(), "pins tp = 2");
        bad(
            &OK.replace("page-size = 64", "port = 1"),
            Machine::default(),
            "set by oaka",
        );
        bad(
            &OK.replace("server = \"a\"", "server = \"b\""),
            Machine::default(),
            "not in the plan",
        );
        bad(
            &OK.replace("conc = [4, 8]", "conc = [4, 8]\nwarmups = 3"),
            Machine::default(),
            "unknown field",
        );
        bad(
            &OK.replace("conc = [4, 8]", "conc = []"),
            Machine::default(),
            "positive concurrencies",
        );
        bad(
            &format!("{OK}\n[[server]]\nname = 'b'\nprofile = 'm/base'\ngpus = [3]\n"),
            Machine::default(),
            "already used by server a",
        );
        bad(
            &OK.replace("m/base", ""),
            Machine::default(),
            "pick one of: m/base m/new m/tp2",
        );
        // ROCm: a profile for another version, or a container whose version is unknown.
        let new = OK.replace("m/base", "m/new");
        bad(
            &new,
            rocm("10.0.0"),
            "this container has ROCm 10.0.0, but profile m/new is for gfx950:10.1",
        );
        let mut both = eight("gfx950");
        both.rocm = Some("10.0.1".into());
        bad(
            &new,
            both,
            "this container has ROCm 10.0.1 on gfx950, but profile m/new is for gfx950:10.1",
        );
        bad(&new, Machine::default(), "this container's ROCm version is unknown");
        assert!(checked(&lib, &new, rocm("10.1.2")).is_ok());
        let mut fits = eight("gfx950");
        fits.rocm = Some("10.1.0".into());
        assert!(checked(&lib, &new, fits).is_ok());
        assert!(checked(&lib, OK, rocm("10.0.0")).is_ok(), "no rocm in the profile: any");
        fs::remove_dir_all(dir).unwrap();
    }
}
