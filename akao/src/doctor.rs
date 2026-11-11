//! `akao doctor`: read-only checks of what akao needs on this machine.  Each line is
//! `ok`, `warn` or `FAIL` with the fix; any FAIL makes the command fail.

use crate::state::{self, State};
use anyhow::{bail, Result};
use std::fs;
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
            r.ok("config root", format!("{} ({})", s.root.display(), s.from));
            s
        }
        Err(e) => {
            r.fail("config root", format!("{e:#}"));
            bail!("1 check failed; the others need the config root");
        }
    };
    let nicks = check_hosts(&mut r, &state);
    check_credentials(&mut r, &state);
    check_ssh(&mut r, &state, &nicks);
    check_template(&mut r, &state);
    check_deploy(&mut r, &state);
    check_repo(&mut r);
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

/// The repo checkout init ships (CLAUDE.md, skills, oaka).
fn check_repo(r: &mut Report) {
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
    if !Path::new(&repo).join(".git").exists() || !Path::new(&repo).join("CLAUDE.md").is_file() {
        return r.fail(
            "repo",
            format!(
                "{repo} is not a checkout of akao-workflow with CLAUDE.md (init ships it to every \
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
}

/// `path`'s difference from the repo's utils/ssh, the wrapper's source: `None` when they
/// match or there is no source to compare with.
fn stale_wrapper(path: &Path) -> Option<String> {
    let (repo, _) = state::repo_root().ok()?;
    let src = Path::new(&repo).join("utils/ssh");
    let want = fs::read(&src).ok()?;
    (fs::read(path).ok()? != want).then(|| format!("{} differs from {}; cp it over", path.display(), src.display()))
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
        ("curl", "akao mirror --dashboard reads the InferenceX dashboard with it"),
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

/// akao's ssh credentials in the config root: `.ssh/config` (how to reach the boxes), the
/// private keys beside it, and every key the config names (ssh skips a missing
/// IdentityFile silently; the login fails later, far from the cause).
fn check_credentials(r: &mut Report, state: &State) {
    let dir = state.root.join(".ssh");
    let cfg = dir.join("config");
    let Ok(text) = fs::read_to_string(&cfg) else {
        return r.fail(
            "credentials",
            format!(
                "no {}: akao's ssh config (hosts, logins, the keys they use) lives there",
                cfg.display()
            ),
        );
    };
    let mut keys: Vec<String> = fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| private_key(&e.path()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    keys.sort();
    if keys.is_empty() {
        return r.fail(
            "credentials",
            format!(
                "no private key in {}: put the keys the boxes' logins use there",
                dir.display()
            ),
        );
    }
    let named = identity_files(&text, &std::env::var("HOME").unwrap_or_default());
    let missing: Vec<&str> = named
        .iter()
        .filter(|p| !Path::new(p).is_file())
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return r.fail(
            "credentials",
            format!(
                "{} names IdentityFile {} that does not exist (ssh skips it, the login fails)",
                cfg.display(),
                missing.join(" ")
            ),
        );
    }
    let outside: Vec<&str> = named
        .iter()
        .filter(|p| !Path::new(p).starts_with(&dir))
        .map(String::as_str)
        .collect();
    let mut detail = format!("{}: config, {}", dir.display(), keys.join(" "));
    if !outside.is_empty() {
        detail += &format!(" (keys outside it: {})", outside.join(" "));
    }
    r.ok("credentials", detail);
}

/// Whether `p` is a PEM or OpenSSH private key.
fn private_key(p: &Path) -> bool {
    p.is_file()
        && fs::read(p).is_ok_and(|b| {
            let head = String::from_utf8_lossy(&b[..b.len().min(64)]).into_owned();
            head.starts_with("-----BEGIN ") && head.contains("PRIVATE KEY-----")
        })
}

/// The IdentityFile paths an ssh config names, `~` and `%d` expanded to `home`; `none` and
/// paths ssh resolves per host (other `%` tokens, `${VAR}`, relative) are left out.
fn identity_files(config: &str, home: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in config.lines() {
        let Some((key, value)) = line.trim().split_once(|c: char| c.is_whitespace() || c == '=') else {
            continue;
        };
        if !key.eq_ignore_ascii_case("identityfile") {
            continue;
        }
        let v = value.trim_start_matches(|c: char| c.is_whitespace() || c == '=').trim();
        let v = v.trim_matches('"');
        let v = match v.strip_prefix("~/") {
            Some(rest) => format!("{home}/{rest}"),
            None => v.replace("%d", home),
        };
        if v.contains('%') || v.contains("${") || !v.starts_with('/') || out.contains(&v) {
            continue;
        }
        out.push(v);
    }
    out
}

/// docker's ssh:// contexts run the `ssh` found on PATH and cannot take -F, so that `ssh`
/// must be the wrapper that reads <config root>/.ssh/config.  Proof: for every host,
/// `ssh -G` through PATH resolves exactly like `ssh -F <config> -G`.
fn check_ssh(r: &mut Report, state: &State, nicks: &[String]) {
    let cfg = state.root.join(".ssh/config");
    let Some(path_ssh) = which("ssh") else {
        r.fail("ssh", "no ssh on PATH");
        return;
    };
    if !cfg.exists() {
        return; // check_credentials fails it
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
    // A host entry with a RemoteCommand (a login shell) refuses every ssh that carries a
    // command, docker's transport included, unless the wrapper turns it off for such calls.
    let login_shell: Vec<&str> = nicks
        .iter()
        .filter(|n| {
            let plain = output(&[REAL_SSH, "-F", &cfg_s, "-G", n]).unwrap_or_default();
            // With a command, ssh -G either prints the config or, keeping the RemoteCommand,
            // refuses like the real call would.
            let with_cmd = output(&[&path_s, "-G", n, "true"]);
            remote_command(&plain) && with_cmd.is_none_or(|g| remote_command(&g))
        })
        .map(String::as_str)
        .collect();
    if !login_shell.is_empty() {
        r.fail(
            "ssh",
            format!(
                "{} has a RemoteCommand that {path_s} keeps for calls with a command: docker and akao \
                 would be refused.  Install $AKAO_REPO_ROOT/utils/ssh as {path_s} and \
                 container_home/.local/bin/ssh",
                login_shell.join(" ")
            ),
        );
    }
    if let Some(stale) = stale_wrapper(&path_ssh) {
        r.warn("ssh", stale);
    }
    if ignored.is_empty() {
        r.ok("ssh", format!("{} resolves every host through {}", path_s, cfg_s));
    } else {
        r.fail(
            "ssh",
            format!(
                "{path_s} ignores {cfg_s} for {}: docker contexts would use ~/.ssh/config.  Put the wrapper \
                 (container_home/.local/bin/ssh) ahead of /usr/bin on PATH and export AKAO_CONFIG_ROOT (or \
                 keep the state in $AKAO_REPO_ROOT/.state)",
                ignored.join(" ")
            ),
        );
    }
}

/// Whether `ssh -G` output sets a RemoteCommand.
fn remote_command(g: &str) -> bool {
    g.lines()
        .any(|l| l.strip_prefix("remotecommand ").is_some_and(|v| v.trim() != "none"))
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
    } else if let Some(stale) = stale_wrapper(&t.join(".local/bin/ssh")) {
        r.warn("home template", format!("{stale} (new workers get the old one)"));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_files_expand_home_and_skip_what_ssh_resolves_per_host() {
        let cfg = "Host a\n\tIdentityFile /s/.ssh/id_ed25519\nHost b\n    identityfile ~/.ssh/id_rsa\n\
                   Host c\n  IdentityFile=\"%d/.ssh/k\"\n  IdentityFile ~/.ssh/id_%r\n  IdentityFile none\n\
                   # IdentityFile /commented\n  IdentityFile /s/.ssh/id_ed25519\n";
        assert_eq!(
            identity_files(cfg, "/root"),
            ["/s/.ssh/id_ed25519", "/root/.ssh/id_rsa", "/root/.ssh/k"]
        );
    }
}
