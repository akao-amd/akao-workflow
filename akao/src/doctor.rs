//! `akao doctor`: read-only checks of what akao needs on this machine.  Each line is
//! `ok`, `warn` or `FAIL` with the fix; any FAIL makes the command fail.

use crate::state::{self, State};
use anyhow::{bail, Result};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The real ssh the PATH wrapper must hand over to.
const REAL_SSH: &str = "/usr/bin/ssh";

#[derive(Default)]
struct Report {
    failed: usize,
}

impl Report {
    fn ok(&mut self, what: &str, detail: impl AsRef<str>) {
        println!("ok    {what:<14} {}", detail.as_ref());
    }
    fn warn(&mut self, what: &str, detail: impl AsRef<str>) {
        println!("warn  {what:<14} {}", detail.as_ref());
    }
    fn fail(&mut self, what: &str, detail: impl AsRef<str>) {
        println!("FAIL  {what:<14} {}", detail.as_ref());
        self.failed += 1;
    }
}

fn executable(p: &Path) -> bool {
    p.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The first executable `name` on PATH.
fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(name))
        .find(|p| executable(p))
}

/// stdout of a successful command.
fn output(argv: &[&str]) -> Option<String> {
    let out = Command::new(argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn run() -> Result<()> {
    let mut r = Report::default();
    let state = match State::load() {
        Ok(s) => {
            r.ok("config root", s.root.display().to_string());
            s
        }
        Err(e) => {
            r.fail("config root", format!("{e:#}"));
            bail!("1 check failed; the others need AKAO_CONFIG_ROOT");
        }
    };
    check_config(&mut r, &state);
    let nicks = check_hosts(&mut r, &state);
    check_ssh(&mut r, &state, &nicks);
    check_template(&mut r, &state);
    check_deploy(&mut r, &state);
    check_repo(&mut r, &state);
    match output(&["docker", "--version"]) {
        Some(v) => r.ok("docker", v.trim()),
        None => r.fail("docker", "docker CLI not found; init drives workers through it"),
    }
    check_controller(&mut r, &state);
    if r.failed > 0 {
        bail!("{} check(s) failed", r.failed);
    }
    Ok(())
}

/// The repo checkout init ships (skills, /<year>/CLAUDE.md), and the console's own
/// /<year>/CLAUDE.md and AGENTS.md: links to the repo's year/CLAUDE.md.
fn check_repo(r: &mut Report, state: &State) {
    let repo = match state::repo_root() {
        Ok((repo, _)) => repo,
        Err(e) => return r.fail("repo", format!("{e:#}")),
    };
    let bin = Path::new(&repo).join("oaka/bin/oaka");
    match output(&[&bin.to_string_lossy(), "--version"]) {
        Some(v) => r.ok("oaka", format!("{} ({})", bin.display(), v.trim())),
        None => r.fail(
            "oaka",
            format!(
                "{} missing or broken; commit in akao-workflow (the post-commit hook builds it)",
                bin.display()
            ),
        ),
    }
    if !Path::new(&repo).join(".git").exists() || !Path::new(&repo).join("year/CLAUDE.md").is_file() {
        return r.fail(
            "repo",
            format!(
                "{repo} is not a checkout of akao-workflow with year/CLAUDE.md (init ships it to every \
                 worker); export {}=<your checkout>",
                state::REPO_ROOT_ENV
            ),
        );
    }
    let head = output(&["git", "-C", &repo, "log", "-1", "--format=%h %s"]).unwrap_or_default();
    match output(&["git", "-C", &repo, "status", "--porcelain", "--untracked-files=no"]) {
        Some(dirty) if !dirty.trim().is_empty() => r.warn(
            "repo",
            format!(
                "{repo} at {}; uncommitted changes are not shipped (init sends commits only)",
                head.trim()
            ),
        ),
        _ => r.ok("repo", format!("{repo} at {}", head.trim())),
    }
    let want = Path::new(&repo).join("year/CLAUDE.md");
    let Ok(src) = state.require("deploy_src") else { return };
    for name in ["CLAUDE.md", "AGENTS.md"] {
        let have = Path::new(&src).join(name);
        let same = have.canonicalize().ok() == want.canonicalize().ok();
        if same {
            r.ok(name, format!("{} -> {}", have.display(), want.display()));
        } else {
            r.warn(
                name,
                format!(
                    "{} is not the repo's; agents under {src} load it: ln -sfn {} {}",
                    have.display(),
                    want.display(),
                    have.display()
                ),
            );
        }
    }
}

/// What a controller agent on this console uses besides akao itself: never fatal, akao
/// runs without them.
fn check_controller(r: &mut Report, state: &State) {
    match state::artifact_root() {
        Ok((root, from)) if Path::new(&root).is_dir() => r.ok("artifact root", format!("{root} ({from})")),
        Ok((root, from)) => r.warn(
            "artifact root",
            format!("{root} ({from}) does not exist yet; the controller creates it for its record"),
        ),
        Err(e) => r.fail("artifact root", format!("{e:#}")),
    }
    for (tool, why) in [
        ("tmux", "the controller opens worker agents in tmux windows"),
        (
            "claude",
            "the controller agent itself; $AKAO_REPO_ROOT/utils/agent.sh installs it",
        ),
    ] {
        match which(tool) {
            Some(p) => r.ok(tool, p.display().to_string()),
            None => r.warn(tool, format!("not on PATH; {why}")),
        }
    }
    let infx = state.get("infx_local").ok().flatten().unwrap_or_default();
    match output(&["git", "-C", &infx, "log", "-1", "--format=%h %cs"]) {
        Some(head) => r.ok("inferencex", format!("{infx} at {}", head.trim())),
        None => r.warn(
            "inferencex",
            format!("{infx} is not a git checkout; akao mirror reads InferenceX configs from it (config infx_local)"),
        ),
    }
}

fn check_config(r: &mut Report, state: &State) {
    match state.get("default_image") {
        Ok(Some(i)) => r.ok("default_image", i),
        _ => r.fail("default_image", "unset; akao config set default_image <tag>"),
    }
}

fn check_hosts(r: &mut Report, state: &State) -> Vec<String> {
    match state.load_hosts() {
        Ok(h) if h.is_empty() => {
            r.warn("hosts", "hosts.tsv has no hosts; akao host add <nick> ...");
            Vec::new()
        }
        Ok(h) => {
            let nicks: Vec<String> = h.into_iter().map(|h| h.nick).collect();
            r.ok("hosts", nicks.join(" "));
            nicks
        }
        Err(e) => {
            r.fail("hosts", format!("{e:#}"));
            Vec::new()
        }
    }
}

/// docker's ssh:// contexts run the `ssh` found on PATH and cannot take -F, so that `ssh`
/// must be the wrapper that reads $AKAO_CONFIG_ROOT/.ssh/config.  Proof: for every host,
/// `ssh -G` through PATH resolves exactly like `ssh -F <config> -G`.
fn check_ssh(r: &mut Report, state: &State, nicks: &[String]) {
    let cfg = state.root.join(".ssh/config");
    let Some(path_ssh) = which("ssh") else {
        r.fail("ssh", "no ssh on PATH");
        return;
    };
    if !cfg.exists() {
        r.ok(
            "ssh",
            format!(
                "{} (no {}; plain ssh config applies)",
                path_ssh.display(),
                cfg.display()
            ),
        );
        return;
    }
    if nicks.is_empty() {
        r.warn("ssh", "no hosts to compare `ssh -G` on");
        return;
    }
    let cfg_s = cfg.to_string_lossy();
    let path_s = path_ssh.to_string_lossy();
    let ignored: Vec<&str> = nicks
        .iter()
        .filter(|n| {
            let via_path = output(&[&path_s, "-G", n]);
            via_path.is_none() || via_path != output(&[REAL_SSH, "-F", &cfg_s, "-G", n])
        })
        .map(String::as_str)
        .collect();
    if ignored.is_empty() {
        r.ok("ssh", format!("{} resolves every host through {}", path_s, cfg_s));
    } else {
        r.fail(
            "ssh",
            format!(
                "{path_s} ignores {cfg_s} for {}: docker contexts would use ~/.ssh/config.  Put the wrapper \
                 (container_home/.local/bin/ssh) ahead of /usr/bin on PATH and export AKAO_CONFIG_ROOT",
                ignored.join(" ")
            ),
        );
    }
}

fn check_template(r: &mut Report, state: &State) {
    let t = state.home_template();
    if !t.is_dir() {
        r.fail(
            "home template",
            format!("{} missing; init step 4 copies it", t.display()),
        );
    } else if !executable(&t.join(".local/bin/ssh")) {
        r.warn(
            "home template",
            format!("{} has no executable .local/bin/ssh wrapper", t.display()),
        );
    } else {
        r.ok("home template", t.display().to_string());
    }
}

fn check_deploy(r: &mut Report, state: &State) {
    let (Ok(src), Ok(paths)) = (state.require("deploy_src"), state.require("deploy_paths")) else {
        r.fail("deploy", "deploy_src / deploy_paths unset");
        return;
    };
    let missing: Vec<&str> = paths
        .split_whitespace()
        .filter(|p| !Path::new(&src).join(p).exists())
        .collect();
    if paths.trim().is_empty() {
        r.ok("deploy", "nothing beyond the repo (deploy_paths is empty)");
    } else if missing.is_empty() {
        r.ok("deploy", format!("{src}: {paths}"));
    } else {
        r.fail(
            "deploy",
            format!("missing under {src}: {} (init step 3 ships them)", missing.join(" ")),
        );
    }
}
