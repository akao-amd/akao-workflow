//! `oaka doctor`: read-only checks of what compiled scripts need in this worker.  Each line
//! is `ok`, `warn` or `FAIL` with the fix; any FAIL makes the command fail.

use crate::compile::infx_root;
use crate::profile::{Engine, Library};
use crate::stack;
use crate::sys;
use anyhow::{bail, Result};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The checks, printed as they run, or kept for one JSON document at the end.
#[derive(Default)]
struct Report {
    json: bool,
    failed: usize,
    checks: Vec<serde_json::Value>,
}

impl Report {
    fn add(&mut self, status: &str, what: &str, detail: &str) {
        if self.json {
            self.checks
                .push(serde_json::json!({ "status": status, "check": what, "detail": detail }));
        } else {
            println!(
                "{:<5} {what:<11} {detail}",
                if status == "fail" { "FAIL" } else { status }
            );
        }
    }
    fn ok(&mut self, what: &str, detail: impl AsRef<str>) {
        self.add("ok", what, detail.as_ref());
    }
    fn warn(&mut self, what: &str, detail: impl AsRef<str>) {
        self.add("warn", what, detail.as_ref());
    }
    fn fail(&mut self, what: &str, detail: impl AsRef<str>) {
        self.add("fail", what, detail.as_ref());
        self.failed += 1;
    }
}

/// The first executable `name` on PATH.
fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(name))
        .find(|p| {
            p.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        })
}

/// stdout of a successful command, with extra environment.
fn output(argv: &[&str], env: &[(&str, &str)]) -> Option<String> {
    let out = Command::new(argv[0])
        .args(&argv[1..])
        .envs(env.iter().copied())
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn run(lib: &Library, json: bool) -> Result<()> {
    let mut r = Report {
        json,
        ..Report::default()
    };
    check_library(&mut r, lib);
    check_stacks(&mut r, lib);

    match sys::gpus() {
        Some(g) if !g.is_empty() => {
            let from = if std::env::var_os("OAKA_GPUS").is_some() {
                " (from OAKA_GPUS)"
            } else {
                ""
            };
            r.ok(
                "gpus",
                format!("{} x {}{from}", g.len(), sys::arch().unwrap_or_else(|| g.join(","))),
            )
        }
        _ => r.fail(
            "gpus",
            "no ROCm GPUs visible in /sys/class/kfd; check the container's --device flags",
        ),
    }

    match sys::rocm() {
        Some((v, from)) => r.ok("rocm", format!("{v} (from {from})")),
        None => r.warn(
            "rocm",
            "version unknown: no .info/version under $ROCM_PATH, $ROCM_HOME or /opt/rocm; \
             profiles whose arch names a ROCm version will not check (set OAKA_ROCM)",
        ),
    }

    for tool in ["bash", "python3"] {
        if which(tool).is_none() {
            r.fail(tool, "not on PATH; every compiled script needs it");
        }
    }

    let infx = infx_root();
    if !Path::new(&infx).join("infx/bench").is_dir() {
        r.fail(
            "inferencex",
            format!("{infx} missing; akao init clones it (or git clone InferenceX there)"),
        );
    } else if output(
        &["python3", "-c", "import infx.bench.fixed_seq"],
        &[("PYTHONPATH", &infx)],
    )
    .is_none()
    {
        r.fail(
            "inferencex",
            format!("python3 cannot import infx.bench.fixed_seq from {infx}"),
        );
    } else {
        let head = output(&["git", "-C", &infx, "log", "-1", "--format=%h %cs"], &[]).unwrap_or_else(|| "?".into());
        r.ok("inferencex", format!("{infx} at {head}"));
    }

    check_engines(&mut r, lib);

    match std::env::var("AKAO_ARTIFACT_ROOT") {
        Err(_) => r.warn(
            "artifacts",
            "AKAO_ARTIFACT_ROOT is not set: this container predates it, so its root is its working \
             directory (where `docker exec` starts; the controller names it), not a path derived from \
             today's week.  To pin one: on the console, docker --context <nick> rm -f akao_<name>, then \
             akao init (/root and /<year> are bind mounts and survive; installs into the image do not)",
        ),
        Ok(v) => match sys::artifact_root() {
            Some(root) if root.is_dir() => r.ok("artifacts", root.display().to_string()),
            Some(root) => r.warn(
                "artifacts",
                format!("AKAO_ARTIFACT_ROOT={} does not exist", root.display()),
            ),
            None => r.warn(
                "artifacts",
                format!("AKAO_ARTIFACT_ROOT={v} is not an absolute path; ignored"),
            ),
        },
    }

    // The worker's clone of akao-workflow: its skill, and where it fixes the tools.
    let repo = std::env::var("AKAO_REPO_ROOT")
        .ok()
        .filter(|r| !r.is_empty())
        .unwrap_or_else(|| "/root/akao-workflow".into());
    if Path::new(&repo).join("skills/worker/SKILL.md").is_file() {
        let head = output(&["git", "-C", &repo, "log", "-1", "--format=%h %cs"], &[]).unwrap_or_else(|| "?".into());
        r.ok("repo", format!("{repo} at {head}"));
    } else {
        r.warn(
            "repo",
            format!("no akao-workflow clone at {repo} (the worker skill lives there); akao init ships one"),
        );
    }

    match which("sgl-eval") {
        Some(p) => r.ok("sgl-eval", p.display().to_string()),
        None => r.warn(
            "sgl-eval",
            "not on PATH; gsm8k clients will fail (akao init installs it; else pip install sgl-eval, never sglang[test])",
        ),
    }
    match which("oaka") {
        Some(p) => r.ok("oaka", p.display().to_string()),
        None => r.warn(
            "oaka",
            "not on PATH; akao init links $AKAO_REPO_ROOT/oaka/bin/oaka to /usr/local/bin",
        ),
    }

    if json {
        let out = serde_json::json!({ "ok": r.failed == 0, "checks": r.checks });
        println!("{}", serde_json::to_string_pretty(&out)?);
    }
    if r.failed > 0 {
        bail!("{} check(s) failed", r.failed);
    }
    Ok(())
}

/// Where python3 finds an engine's package (without importing it), or None.
fn engine_origin(engine: Engine) -> Option<String> {
    let find = format!(
        "import importlib.util as u; s = u.find_spec({:?}); print(s.origin if s and s.origin else '')",
        engine.module()
    );
    let origin = output(&["python3", "-c", &find], &[]).filter(|o| !o.is_empty())?;
    match engine {
        // `atom` is a common name: it must be the one with ATOM's OpenAI server.
        Engine::Atom => Path::new(&origin)
            .with_file_name("entrypoints/openai_server.py")
            .is_file()
            .then_some(origin),
        _ => Some(origin),
    }
}

/// The serving engines python3 finds here.  One is enough; an engine that a profile for
/// this machine uses but that is missing is worth a warning.
fn check_engines(r: &mut Report, lib: &Library) {
    let (arch, rocm) = (sys::arch(), sys::rocm().map(|(v, _)| v));
    let wanted: Vec<(String, Engine)> = lib
        .list()
        .unwrap_or_default()
        .iter()
        .filter_map(|n| lib.resolve(n).ok())
        .filter(|p| p.fits(arch.as_deref(), rocm.as_deref()))
        .map(|p| (p.name, p.engine))
        .collect();
    let mut found = 0;
    for engine in Engine::ALL {
        let users: Vec<&str> = wanted
            .iter()
            .filter(|(_, e)| *e == engine)
            .map(|(n, _)| n.as_str())
            .collect();
        match engine_origin(engine) {
            Some(origin) if engine == Engine::Vllm && which("vllm").is_none() => {
                found += 1;
                r.warn("vllm", format!("{origin}, but no `vllm` on PATH to serve with"));
            }
            Some(origin) => {
                found += 1;
                r.ok(engine.as_str(), origin);
            }
            None if !users.is_empty() => r.warn(
                engine.as_str(),
                format!(
                    "python3 cannot find {}; profiles {} use it",
                    engine.module(),
                    users.join(" ")
                ),
            ),
            None => {}
        }
    }
    if found == 0 {
        r.fail(
            "engines",
            "python3 finds none of sglang, vllm, atom: no server can start in this container",
        );
    }
}

fn check_library(r: &mut Report, lib: &Library) {
    if !lib.profiles_dir().is_dir() {
        r.fail(
            "library",
            format!(
                "{} missing; the library is the oaka/ dir of the akao-workflow clone, which akao init ships",
                lib.profiles_dir().display()
            ),
        );
        return;
    }
    let names = match lib.list() {
        Ok(n) => n,
        Err(e) => return r.fail("library", format!("{e:#}")),
    };
    let broken: Vec<String> = names
        .iter()
        .filter_map(|n| lib.resolve(n).err().map(|e| format!("{e:#}")))
        .collect();
    if broken.is_empty() {
        r.ok("library", format!("{}: {} profiles", lib.root.display(), names.len()));
    } else {
        r.fail("library", broken.join("; "));
    }
}

/// stacks.toml parses; per package: its repo is a git checkout here, and where its module
/// imports from now (and what oaka last installed).
fn check_stacks(r: &mut Report, lib: &Library) {
    let path = stack::path(lib);
    if !path.exists() {
        return r.warn("stacks", format!("no {}: plans cannot use [stack]", path.display()));
    }
    let stacks = match stack::load(lib) {
        Ok(s) => s,
        Err(e) => return r.fail("stacks", format!("{e:#}")),
    };
    r.ok("stacks", format!("{}: {}", path.display(), stacks.names().join(" ")));
    let state = match std::env::var("OAKA_STACK_STATE") {
        Ok(s) if !s.is_empty() => Some(PathBuf::from(s)),
        _ => output(
            &[
                "python3",
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            &[],
        )
        .map(|sp| Path::new(&sp).join(".oaka-stack")),
    };
    for (name, p) in &stacks.packages {
        let what = format!("stack {name}");
        if output(&["git", "-C", &p.repo, "rev-parse", "--git-dir"], &[]).is_none() {
            r.warn(
                &what,
                format!(
                    "repo {} is not a git checkout here; plans need an existing tree",
                    p.repo
                ),
            );
            continue;
        }
        let find = format!(
            "import importlib.util as u; s = u.find_spec({:?}); print(s.origin if s else '')",
            p.module
        );
        let origin = output(&["python3", "-c", &find], &[]).unwrap_or_default();
        let installed = state
            .as_ref()
            .and_then(|s| std::fs::read_to_string(s.join(name).join("installed")).ok())
            .map(|l| format!("; oaka last installed {}", l.trim()))
            .unwrap_or_default();
        let archs = p.archs();
        let recipes = if archs.is_empty() {
            "any GPU arch".to_string()
        } else {
            archs.join(" ")
        };
        let line = format!(
            "{} imports from {}{installed}; recipes: {recipes}",
            p.module,
            if origin.is_empty() { "nowhere" } else { &origin }
        );
        match sys::arch() {
            Some(a) if !archs.is_empty() && !archs.contains(&a.as_str()) => {
                r.warn(&what, format!("{line}; none for this machine's {a}"))
            }
            _ => r.ok(&what, line),
        }
    }
}
