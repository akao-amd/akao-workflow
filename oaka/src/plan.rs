//! `plan.toml` in a Work Directory, its validation, and `plan.lock.toml`.

use crate::profile::{Arg, Library, Profile};
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
    /// Non-fatal findings, e.g. a model directory that does not exist here.
    pub warnings: Vec<String>,
}

pub fn valid_server_name(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_alphanumeric()) && c.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

pub fn load(dir: &Path) -> Result<Plan> {
    let path = dir.join(PLAN);
    if !path.exists() {
        bail!("no {} in {}; write one with `oaka draft`", PLAN, dir.display());
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Validate the plan in `dir`.  Every error names the field to fix.
pub fn check(dir: &Path, lib: &Library) -> Result<Checked> {
    let plan = load(dir)?;
    check_plan(dir, plan, lib, sys::gpus())
}

pub fn check_plan(dir: &Path, plan: Plan, lib: &Library, machine: Option<Vec<String>>) -> Result<Checked> {
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
            bail!("{at}: profile is empty; pick one of: {}", if names.is_empty() { "<none in library>".into() } else { names.join(" ") });
        }
        let base = lib.resolve(&s.profile).with_context(|| at.clone())?;
        let mut effective = base.clone();
        effective.overlay(&s.env, &s.args).with_context(|| format!("{at}: env/args overrides"))?;
        let model = s.model.clone().or_else(|| base.model.clone()).with_context(|| {
            format!("{at}: profile {} has no model; set `model` in the server", s.profile)
        })?;
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
                bail!("{at}: profile {} pins tp = {tp} but gpus lists {}", s.profile, s.gpus.len())
            }
            Some(tp) => tp,
            None => s.gpus.len() as u32,
        };
        if let Some(gpus) = &machine {
            if let Some(g) = s.gpus.iter().find(|g| **g as usize >= gpus.len()) {
                match gpus.len() {
                    0 => bail!("{at}: GPU {g} does not exist; this machine has no GPUs"),
                    n => bail!("{at}: GPU {g} does not exist; this machine has {n} GPUs (0-{})", n - 1),
                }
            }
            if !base.arch.is_empty() {
                for g in &s.gpus {
                    let a = &gpus[*g as usize];
                    if !base.arch.contains(a) {
                        bail!("{at}: GPU {g} is {a}, but profile {} is for {}", s.profile, base.arch.join(" "));
                    }
                }
            }
        }
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
        servers.push(Server { name: s.name.clone(), base, effective, model, gpus: s.gpus.clone(), tp, port: s.port });
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
            ClientSpec::FixedSeq { isl_osl, conc, range_ratio, repeats, off_spec, .. } => {
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
                if let Some(OffSpec { prompts_per_conc: Some(0) }) = off_spec {
                    bail!("{at}: off_spec.prompts_per_conc must be positive");
                }
            }
        }
    }
    Ok(Checked { dir: dir.to_path_buf(), plan, servers, warnings })
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
        fs::write(lib.profile_path("m/base"), "model = '/model/m'\narch = ['gfx950']\n[args]\npage-size = 1\n").unwrap();
        fs::write(lib.profile_path("m/tp2"), "extends = 'm/base'\ntp = 2\n").unwrap();
        (lib, dir)
    }

    fn checked(lib: &Library, text: &str, machine: Option<Vec<String>>) -> Result<Checked> {
        check_plan(Path::new("/w"), toml::from_str(text)?, lib, machine)
    }

    fn eight(arch: &str) -> Option<Vec<String>> {
        Some(vec![arch.to_string(); 8])
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
            ClientSpec::FixedSeq { range_ratio, repeats, .. } => assert_eq!((*range_ratio, *repeats), (0.8, 1)),
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
        bad(&OK.replace("[3]", "[]"), None, "gpus is empty");
        bad(&OK.replace("m/base", "m/tp2"), None, "pins tp = 2");
        bad(&OK.replace("page-size = 64", "port = 1"), None, "set by oaka");
        bad(&OK.replace("server = \"a\"", "server = \"b\""), None, "not in the plan");
        bad(&OK.replace("conc = [4, 8]", "conc = [4, 8]\nwarmups = 3"), None, "unknown field");
        bad(&OK.replace("conc = [4, 8]", "conc = []"), None, "positive concurrencies");
        bad(&format!("{OK}\n[[server]]\nname = 'b'\nprofile = 'm/base'\ngpus = [3]\n"), None, "already used by server a");
        bad(&OK.replace("m/base", ""), None, "pick one of: m/base m/tp2");
        fs::remove_dir_all(dir).unwrap();
    }
}
