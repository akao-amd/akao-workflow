//! Suites: `<lib>/suites/<name>.toml`, a named benchmark matrix applied to any profile.
//!
//! A suite fixes what is measured (layouts x ISL/OSL pairs x concurrencies, and the gsm8k
//! gate) so that "the MI45x DPX standard suite" means the same runs on every box and for
//! every recipe; the profile says what is served.  `oaka draft --suite <name> --profile <p>
//! --gpus <a,b>` writes one plan per layout, `<layout>/plan.toml`, each an ordinary plan
//! (check, compile and run it as any other).  A layout places servers on the suite's GPUs
//! by position: `[[0], [1]]` is two servers, one on each of the GPUs given.
//!
//! Like a profile, a suite is self-contained but for `extends` (another suite): the child's
//! fields replace the parent's, its layouts are added after the parent's (one of the same
//! name replaces it in place).

use crate::draft::{stack_arg, stack_block, toml_list};
use crate::plan::valid_server_name;
use crate::profile::Library;
use crate::stack;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::fs;

/// One suite file, as written on disk.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteFile {
    pub description: Option<String>,
    /// GPU archs it is for; empty = any.  Absent: the parent's.
    pub arch: Option<Vec<String>>,
    pub extends: Option<String>,
    /// How many GPUs it takes (`--gpus` must give exactly these).
    pub gpus: Option<usize>,
    /// The GPUs must be the partitions of one physical GPU, in order: plan GPUs 2k, 2k+1
    /// (a DPX mode of gfx1250, where HIP numbers a GPU's two partitions next to each other).
    pub partitions_of_one_gpu: Option<bool>,
    pub isl_osl: Option<Vec<[u32; 2]>>,
    pub conc: Option<Vec<u32>>,
    pub range_ratio: Option<f64>,
    /// A gsm8k gate per server, before its throughput points.
    pub gsm8k: Option<Gsm8kGate>,
    #[serde(default)]
    pub layout: Vec<Layout>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gsm8kGate {
    pub min_score: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    /// Also the plan's directory and its servers' names.
    pub name: String,
    pub description: Option<String>,
    /// One entry per server: positions in the suite's GPUs.
    pub servers: Vec<Vec<usize>>,
    /// Measure the servers at the same time (their fixed-seq clients in lockstep).
    #[serde(default)]
    pub lockstep: bool,
    /// Serve `<profile><suffix>` instead of the profile, e.g. "-pd1p1d" for its prefill-decode
    /// variant; the draft fails, naming it, when the library has none.
    pub profile_suffix: Option<String>,
}

/// A suite with its extends chain applied.
#[derive(Debug)]
pub struct Suite {
    pub name: String,
    pub description: String,
    pub arch: Vec<String>,
    pub gpus: usize,
    pub partitions_of_one_gpu: bool,
    pub isl_osl: Vec<[u32; 2]>,
    pub conc: Vec<u32>,
    pub range_ratio: f64,
    pub gsm8k: Option<Gsm8kGate>,
    pub layouts: Vec<Layout>,
}

impl Library {
    pub fn suites_dir(&self) -> std::path::PathBuf {
        self.root.join("suites")
    }

    pub fn suite_path(&self, name: &str) -> std::path::PathBuf {
        self.suites_dir().join(format!("{name}.toml"))
    }

    /// All suite names, sorted.
    pub fn suites(&self) -> Result<Vec<String>> {
        let dir = self.suites_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut names: Vec<String> = fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        Ok(names)
    }

    pub fn suite(&self, name: &str) -> Result<Suite> {
        let mut chain: Vec<(String, SuiteFile)> = Vec::new();
        let mut next = Some(name.to_string());
        while let Some(n) = next {
            if !valid_suite_name(&n) {
                bail!("invalid suite name {n:?}: [A-Za-z0-9][A-Za-z0-9_.-]*, a file of the library's suites/");
            }
            if chain.iter().any(|(c, _)| *c == n) {
                bail!("suite {name}: extends loops back to {n}");
            }
            let path = self.suite_path(&n);
            if !path.exists() {
                bail!("suite {n} not found ({}); see `oaka suite ls`", path.display());
            }
            let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            let f: SuiteFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
            next = f.extends.clone();
            chain.push((n, f));
        }
        let mut s = Suite {
            name: name.to_string(),
            description: String::new(),
            arch: Vec::new(),
            gpus: 0,
            partitions_of_one_gpu: false,
            isl_osl: Vec::new(),
            conc: Vec::new(),
            range_ratio: 0.8,
            gsm8k: None,
            layouts: Vec::new(),
        };
        // Root first, so each child overrides its parent.
        for (_, f) in chain.into_iter().rev() {
            if let Some(v) = f.description {
                s.description = v;
            }
            if let Some(v) = f.arch {
                s.arch = v;
            }
            if let Some(v) = f.gpus {
                s.gpus = v;
            }
            if let Some(v) = f.partitions_of_one_gpu {
                s.partitions_of_one_gpu = v;
            }
            if let Some(v) = f.isl_osl {
                s.isl_osl = v;
            }
            if let Some(v) = f.conc {
                s.conc = v;
            }
            if let Some(v) = f.range_ratio {
                s.range_ratio = v;
            }
            if f.gsm8k.is_some() {
                s.gsm8k = f.gsm8k;
            }
            for l in f.layout {
                match s.layouts.iter_mut().find(|x| x.name == l.name) {
                    Some(x) => *x = l,
                    None => s.layouts.push(l),
                }
            }
        }
        s.validate()?;
        Ok(s)
    }
}

/// A suite is a file of the library's suites/, named [A-Za-z0-9][A-Za-z0-9_.-]*: never a path.
fn valid_suite_name(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_alphanumeric())
        && c.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

impl Suite {
    fn validate(&self) -> Result<()> {
        let at = format!("suite {}", self.name);
        if self.gpus == 0 {
            bail!("{at}: gpus is required (how many GPUs it takes)");
        }
        if self.partitions_of_one_gpu && self.gpus != 2 {
            bail!("{at}: partitions_of_one_gpu takes 2 GPUs, not {}", self.gpus);
        }
        if self.isl_osl.is_empty() || self.conc.is_empty() {
            bail!("{at}: isl_osl and conc are required");
        }
        if self.isl_osl.iter().flatten().any(|&v| v == 0) || self.conc.contains(&0) {
            bail!("{at}: isl_osl and conc must be positive");
        }
        if let Some(g) = &self.gsm8k {
            if !(0.0..=1.0).contains(&g.min_score) {
                bail!("{at}: gsm8k min_score must be within 0..1");
            }
        }
        if !(self.range_ratio > 0.0 && self.range_ratio <= 1.0) {
            bail!("{at}: range_ratio must be within (0, 1]");
        }
        if self.layouts.is_empty() {
            bail!("{at}: no [[layout]]");
        }
        for l in &self.layouts {
            if !valid_server_name(&l.name) {
                bail!("{at}: layout name {:?}: letters, digits, '-' and '_' only", l.name);
            }
            if l.servers.is_empty() || l.servers.iter().any(|g| g.is_empty()) {
                bail!("{at}, layout {}: every server needs at least one GPU position", l.name);
            }
            let mut used: Vec<usize> = l.servers.iter().flatten().copied().collect();
            if let Some(p) = used.iter().find(|&&p| p >= self.gpus) {
                bail!(
                    "{at}, layout {}: GPU position {p}, but the suite takes {} GPUs",
                    l.name,
                    self.gpus
                );
            }
            used.sort();
            if used.windows(2).any(|w| w[0] == w[1]) {
                bail!("{at}, layout {}: a GPU position is used twice", l.name);
            }
            if l.lockstep && l.servers.len() < 2 {
                bail!("{at}, layout {}: lockstep needs two servers or more", l.name);
            }
        }
        Ok(())
    }

    /// The suite's own description, as `oaka suite show` prints it.
    pub fn text(&self) -> String {
        let pairs: Vec<String> = self.isl_osl.iter().map(|[i, o]| format!("{i}/{o}")).collect();
        let mut out = format!("{}: {}\n", self.name, self.description);
        out += &format!(
            "  {} GPUs{}{}\n",
            self.gpus,
            if self.partitions_of_one_gpu {
                " (the two partitions of one physical GPU)"
            } else {
                ""
            },
            if self.arch.is_empty() {
                String::new()
            } else {
                format!(", for {}", self.arch.join(" "))
            }
        );
        out += &format!(
            "  fixed-seq {} x conc {} (range ratio {}), {}\n",
            pairs.join(" "),
            toml_list(&self.conc),
            self.range_ratio,
            match &self.gsm8k {
                Some(g) => format!("gsm8k gate {} per server", g.min_score),
                None => "no gsm8k gate".into(),
            }
        );
        for l in &self.layouts {
            out += &format!(
                "  layout {:<8} servers on {}{}{}{}\n",
                l.name,
                l.servers.iter().map(|g| toml_list(g)).collect::<Vec<_>>().join(" "),
                if l.lockstep { ", in lockstep" } else { "" },
                l.profile_suffix
                    .as_ref()
                    .map(|s| format!(", profile <profile>{s}"))
                    .unwrap_or_default(),
                l.description.as_ref().map(|d| format!(": {d}")).unwrap_or_default()
            );
        }
        out
    }
}

pub struct DraftOptions<'a> {
    pub profile: &'a str,
    pub gpus: &'a [u32],
    pub stacks: &'a [String],
    pub thinking: bool,
    pub port: Option<u16>,
    /// What the drafted plans are checked against (Machine::probe() in a worker): the suite's
    /// arch and each profile's targets.
    pub machine: &'a crate::plan::Machine,
}

/// One plan per layout: (layout name, plan.toml text).
pub fn render(lib: &Library, suite: &Suite, o: &DraftOptions) -> Result<Vec<(String, String)>> {
    let at = format!("suite {}", suite.name);
    if o.profile.is_empty() {
        bail!("--suite needs --profile: the recipe the suite measures");
    }
    // "<model>" is "<model>/base": the suffix goes after the recipe ("<model>/base-tp2").
    let base_profile = crate::profile::canonical(o.profile)?;
    lib.resolve(&base_profile)?;
    if o.gpus.len() != suite.gpus {
        bail!("{at} takes {} GPUs (--gpus), not {}", suite.gpus, o.gpus.len());
    }
    let mut distinct = o.gpus.to_vec();
    distinct.sort();
    distinct.dedup();
    if distinct.len() != o.gpus.len() {
        bail!("{at}: --gpus {} names a GPU twice", toml_list(o.gpus));
    }
    if let Some(base) = o.port {
        let most = suite.layouts.iter().map(|l| l.servers.len()).max().unwrap_or(1);
        if base < 1024 || base.checked_add(most as u16 - 1).is_none() {
            bail!(
                "--port {base}: the servers need ports {base} up to {base}+{}, within 1024-65535",
                most - 1
            );
        }
    }
    if suite.partitions_of_one_gpu && !(o.gpus[0].is_multiple_of(2) && o.gpus[1] == o.gpus[0] + 1) {
        bail!(
            "{at} measures the two partitions of one physical GPU: --gpus 2k,2k+1 (e.g. 4,5), not {}",
            toml_list(o.gpus)
        );
    }
    // The suite's arch, judged on the GPUs it is given (a box may mix archs).
    if let (Some(archs), false) = (&o.machine.gpus, suite.arch.is_empty()) {
        for &g in o.gpus {
            match archs.get(g as usize) {
                None => bail!("{at}: GPU {g} does not exist here ({} GPUs)", archs.len()),
                Some(a) if !suite.arch.contains(a) => {
                    bail!("{at} is for {}; GPU {g} is {a}", suite.arch.join(" "))
                }
                Some(_) => {}
            }
        }
    }
    let stacks = stack::load(lib)?;
    let mut plans = Vec::new();
    for l in &suite.layouts {
        let profile = format!("{base_profile}{}", l.profile_suffix.as_deref().unwrap_or(""));
        if !lib.profile_path(&profile).exists() {
            bail!(
                "{at}, layout {}: needs profile {profile} ({}); write it, extending {}",
                l.name,
                l.description.as_deref().unwrap_or("its variant of the profile"),
                base_profile
            );
        }
        let p = lib.resolve(&profile)?;
        let names: Vec<String> = if l.servers.len() == 1 {
            vec![l.name.clone()]
        } else {
            (0..l.servers.len()).map(|i| format!("{}_{i}", l.name)).collect()
        };
        let mut out = format!(
            "# oaka plan: suite {}, layout {}{}\n# profile {profile}; written by `oaka draft --suite`.  oaka check / oaka run\n",
            suite.name,
            l.name,
            l.description.as_ref().map(|d| format!(" ({d})")).unwrap_or_default()
        );
        for (i, (name, pos)) in names.iter().zip(&l.servers).enumerate() {
            let gpus: Vec<u32> = pos.iter().map(|&k| o.gpus[k]).collect();
            if let Some(tp) = p.tp {
                let want = tp as usize * p.instances() as usize;
                if want != gpus.len() {
                    bail!(
                        "{at}, layout {}: profile {profile} takes {want} GPUs (tp {tp}), the layout gives {}",
                        l.name,
                        gpus.len()
                    );
                }
            }
            out += &format!(
                "\n[[server]]\nname = \"{name}\"\nprofile = \"{profile}\"\ngpus = {}\n",
                toml_list(&gpus)
            );
            match o.port {
                Some(base) => out += &format!("port = {}\n", base as usize + i),
                None => out += "# port = 29911            # pin it when other workers share the box's network\n",
            }
        }
        for name in &names {
            if let Some(g) = &suite.gsm8k {
                out += &format!(
                    "\n[[client]]\nkind = \"gsm8k\"\nserver = \"{name}\"\nthinking = {}\nmin_score = {}\n",
                    o.thinking, g.min_score
                );
            }
        }
        let pairs: Vec<String> = suite.isl_osl.iter().map(|[i, o]| format!("[{i}, {o}]")).collect();
        for name in &names {
            out += &format!(
                "\n[[client]]\nkind = \"fixed-seq\"\nserver = \"{name}\"\nisl_osl = [{}]\nconc = {}\nrange_ratio = {}\n",
                pairs.join(", "),
                toml_list(&suite.conc),
                suite.range_ratio
            );
            if l.lockstep {
                out += &format!("lockstep = \"{}\"\n", l.name);
            }
        }
        for arg in o.stacks {
            let (s, commit) = stack_arg(arg);
            let Some(pkg) = stacks.get(s) else {
                bail!(
                    "--stack {s}: not in {}; it has: {}",
                    stacks.path.display(),
                    stacks.names().join(" ")
                );
            };
            out += "\n";
            out += &stack_block(s, &pkg.repo, &format!("/root/trees/{s}"), commit);
        }
        // The plan's own rules (GPUs per server, PD shares, ports, clients), without its stack,
        // whose commit may still be the caller's to fill in.
        let mut plan: crate::plan::Plan = toml::from_str(&out).context("a drafted plan does not parse")?;
        plan.stack.clear();
        crate::plan::check_plan(&lib.root, plan, lib, o.machine)
            .with_context(|| format!("{at}, layout {}: the drafted plan", l.name))?;
        plans.push((l.name.clone(), out));
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan;

    fn lib() -> (std::path::PathBuf, Library) {
        let root = std::env::temp_dir().join(format!("oaka-suite-{}-{}", std::process::id(), fastrand::u32(..)));
        let lib = Library { root: root.clone() };
        fs::create_dir_all(lib.profiles_dir().join("m")).unwrap();
        fs::create_dir_all(lib.suites_dir()).unwrap();
        fs::write(lib.profile_path("m/r"), "model = '/model/m'\n").unwrap();
        fs::write(lib.profile_path("m/r-pd1p1d"), "extends = 'm/r'\ntp = 1\n[pd]\n").unwrap();
        fs::write(
            lib.root.join("stacks.toml"),
            "[sglang]\nrepo = '/sgl-workspace/sglang'\nmodule = 'sglang'\ninstall = 'true'\n",
        )
        .unwrap();
        fs::write(
            lib.suite_path("std"),
            "description = 'std'\narch = ['gfx1250']\ngpus = 2\npartitions_of_one_gpu = true\n\
             isl_osl = [[1024, 1024], [70000, 300]]\nconc = [4, 64]\ngsm8k = { min_score = 0.93 }\n\
             [[layout]]\nname = 'tp1'\nservers = [[0]]\n\
             [[layout]]\nname = 'tp1x2'\nservers = [[0], [1]]\nlockstep = true\n\
             [[layout]]\nname = 'tp2'\nservers = [[0, 1]]\n",
        )
        .unwrap();
        fs::write(
            lib.suite_path("extra"),
            "extends = 'std'\ndescription = 'extra'\n\
             [[layout]]\nname = 'pd1p1d'\nservers = [[0, 1]]\nprofile_suffix = '-pd1p1d'\n",
        )
        .unwrap();
        (root, lib)
    }

    #[test]
    fn suites_extend_and_draft_plans_that_check() {
        let (root, lib) = lib();
        assert_eq!(lib.suites().unwrap(), ["extra", "std"]);
        let s = lib.suite("extra").unwrap();
        assert_eq!(s.description, "extra");
        assert_eq!(s.isl_osl, [[1024, 1024], [70000, 300]]);
        let names: Vec<&str> = s.layouts.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["tp1", "tp1x2", "tp2", "pd1p1d"]);

        let stacks = vec!["sglang@abc123".to_string()];
        let o = DraftOptions {
            profile: "m/r",
            gpus: &[4, 5],
            stacks: &stacks,
            thinking: true,
            port: Some(31341),
            machine: &plan::Machine::default(),
        };
        let plans = render(&lib, &s, &o).unwrap();
        let mut seen = Vec::new();
        for (layout, text) in &plans {
            let c = plan::check_plan(&root, toml::from_str(text).unwrap(), &lib, &plan::Machine::default())
                .unwrap_or_else(|e| panic!("{layout}: {e:#}\n{text}"));
            let servers: Vec<(String, Vec<u32>, Option<u16>)> = c
                .servers
                .iter()
                .map(|s| (s.name.clone(), s.gpus.clone(), s.port))
                .collect();
            seen.push((layout.clone(), servers, c.plan.clients.len()));
        }
        assert_eq!(
            seen,
            [
                ("tp1".into(), vec![("tp1".into(), vec![4], Some(31341))], 2),
                (
                    "tp1x2".into(),
                    vec![
                        ("tp1x2_0".into(), vec![4], Some(31341)),
                        ("tp1x2_1".into(), vec![5], Some(31342))
                    ],
                    4
                ),
                ("tp2".into(), vec![("tp2".into(), vec![4, 5], Some(31341))], 2),
                ("pd1p1d".into(), vec![("pd1p1d".into(), vec![4, 5], Some(31341))], 2),
            ]
        );
        assert!(plans[1].1.matches("lockstep = \"tp1x2\"").count() == 2);
        assert!(
            plans[0].1.contains("thinking = true")
                && plans[0].1.contains("[stack.sglang]")
                && plans[0].1.contains("commit = \"abc123\"")
        );

        // The suite's rules are the draft's errors.
        let e = |gpus: &[u32]| format!("{:#}", render(&lib, &s, &DraftOptions { gpus, ..o }).err().unwrap());
        assert!(e(&[5, 6]).contains("--gpus 2k,2k+1"));
        assert!(e(&[4]).contains("takes 2 GPUs"));
        fs::remove_file(lib.profile_path("m/r-pd1p1d")).unwrap();
        assert!(e(&[4, 5]).contains("needs profile m/r-pd1p1d"));
        assert!(e(&[4, 4]).contains("names a GPU twice"));
        // Drafted plans are checked against the machine given: a ROCm-pinned profile fits a
        // matching one, and the suite's arch is enforced.
        fs::write(
            lib.profile_path("m/rocm"),
            "model = '/model/m'\narch = [['gfx1250', '10.1']]\n",
        )
        .unwrap();
        fs::write(lib.profile_path("m/rocm-pd1p1d"), "extends = 'm/rocm'\ntp = 1\n[pd]\n").unwrap();
        let gfx1250 = plan::Machine {
            gpus: Some(vec!["gfx1250".into(); 8]),
            rocm: Some("10.1.0".into()),
        };
        render(
            &lib,
            &s,
            &DraftOptions {
                profile: "m/rocm",
                machine: &gfx1250,
                ..o
            },
        )
        .unwrap();
        let gfx950 = plan::Machine {
            gpus: Some(vec!["gfx950".into(); 8]),
            rocm: Some("10.1.0".into()),
        };
        let wrong = render(&lib, &s, &DraftOptions { machine: &gfx950, ..o }).err().unwrap();
        assert!(format!("{wrong:#}").contains("is for gfx1250"));
        let mixed = plan::Machine {
            gpus: Some(["gfx950", "gfx950", "gfx1250", "gfx1250"].map(String::from).to_vec()),
            rocm: Some("10.1.0".into()),
        };
        let on = |gpus: &[u32]| {
            render(
                &lib,
                &s,
                &DraftOptions {
                    gpus,
                    profile: "m/rocm",
                    machine: &mixed,
                    ..o
                },
            )
        };
        on(&[2, 3]).unwrap();
        assert!(format!("{:#}", on(&[0, 1]).err().unwrap()).contains("GPU 0 is gfx950"));
        assert!(format!("{:#}", on(&[4, 5]).err().unwrap()).contains("GPU 4 does not exist"));
        let port = |p| {
            format!(
                "{:#}",
                render(&lib, &s, &DraftOptions { port: Some(p), ..o }).err().unwrap()
            )
        };
        assert!(port(65535).contains("within 1024-65535") && port(80).contains("within 1024-65535"));
        fs::write(lib.profile_path("m/base"), "model = '/model/m'\n").unwrap();
        fs::write(lib.profile_path("m/base-pd1p1d"), "extends = 'm/base'\ntp = 1\n[pd]\n").unwrap();
        let plans = render(&lib, &s, &DraftOptions { profile: "m", ..o }).unwrap();
        assert!(plans[3].1.contains("profile = \"m/base-pd1p1d\""));
        assert!(format!("{:#}", lib.suite("../x").err().unwrap()).contains("invalid suite name"));
        fs::write(lib.suite_path("anyarch"), "extends = 'std'\narch = []\n").unwrap();
        assert!(lib.suite("anyarch").unwrap().arch.is_empty() && s.arch == ["gfx1250"]);
        fs::write(
            lib.suite_path("zero"),
            "gpus = 1\nisl_osl = [[1, 1]]\nconc = [0]\n[[layout]]\nname = 'x'\nservers = [[0]]\n",
        )
        .unwrap();
        assert!(format!("{:#}", lib.suite("zero").err().unwrap()).contains("must be positive"));
        fs::write(
            lib.suite_path("bad"),
            "gpus = 2\nisl_osl = [[1, 1]]\nconc = [1]\n[[layout]]\nname = 'x'\nservers = [[0], [2]]\n",
        )
        .unwrap();
        assert!(format!("{:#}", lib.suite("bad").err().unwrap()).contains("GPU position 2"));
        fs::remove_dir_all(root).unwrap();
    }
}
