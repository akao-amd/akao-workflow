//! `oaka doctor`: read-only checks of what compiled scripts need in this worker.  Each line
//! is `ok`, `warn` or `FAIL` with the fix; any FAIL makes the command fail.

use crate::compile::infx_root;
use crate::profile::Library;
use crate::stack;
use crate::sys;
use anyhow::{bail, Result};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Default)]
struct Report {
    failed: usize,
}

impl Report {
    fn ok(&mut self, what: &str, detail: impl AsRef<str>) {
        println!("ok    {what:<11} {}", detail.as_ref());
    }
    fn warn(&mut self, what: &str, detail: impl AsRef<str>) {
        println!("warn  {what:<11} {}", detail.as_ref());
    }
    fn fail(&mut self, what: &str, detail: impl AsRef<str>) {
        println!("FAIL  {what:<11} {}", detail.as_ref());
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

pub fn run(lib: &Library) -> Result<()> {
    let mut r = Report::default();
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

    let find = "import importlib.util as u; s = u.find_spec('sglang'); print(s.origin if s else '')";
    match output(&["python3", "-c", find], &[]) {
        Some(origin) if !origin.is_empty() => r.ok("sglang", origin),
        _ => r.fail("sglang", "python3 cannot find the sglang package"),
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
            "not on PATH; akao init links /<year>/oaka/bin/oaka to /usr/local/bin",
        ),
    }

    if r.failed > 0 {
        bail!("{} check(s) failed", r.failed);
    }
    Ok(())
}

fn check_library(r: &mut Report, lib: &Library) {
    if !lib.profiles_dir().is_dir() {
        r.fail(
            "library",
            format!(
                "{} missing; akao init deploys /<year>/oaka",
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
        r.ok(
            &what,
            format!(
                "{} imports from {}{installed}",
                p.module,
                if origin.is_empty() { "nowhere" } else { &origin }
            ),
        );
    }
}
