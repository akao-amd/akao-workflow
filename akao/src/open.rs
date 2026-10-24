//! `akao open <nick> <name>`: open the worker's agent (`claude` in akao_<name>) in a new window
//! of this console's tmux, named <name>, so the controller and the user see it side by side.
//!
//! The window runs `docker -H ssh://<nick> exec -it akao_<name> bash -lc 'exec claude ...'`:
//! the worker's login shell (its ~/.bash_profile holds the gateway key; nothing secret leaves
//! the console), with the worker's clone of the repo as an extra working directory.  The
//! window gets our AKAO_CONFIG_ROOT: docker's ssh goes through the `ssh` wrapper, which reads
//! akao's ssh config from it, and the tmux server's environment may lack it.

use crate::exec::{show, Runner};
use crate::state::{self, State};
use anyhow::{bail, Context, Result};

/// The agent in a worker container (installed by init step 9, utils/agent.sh).
const CLAUDE: &str = "/root/.local/bin/claude";

pub struct Options {
    pub nick: String,
    pub name: String,
    /// The directory the agent starts in (a task dir), instead of the container's root.
    pub cwd: Option<String>,
    /// Resume the directory's latest conversation.
    pub resume_latest: bool,
    /// Resume this conversation.
    pub resume: Option<String>,
    /// The tmux session [default: the current one].
    pub session: Option<String>,
}

/// The command the window runs.  `$AKAO_REPO_ROOT` is the worker's, expanded by its shell.
pub fn window_command(nick: &str, container: &str, opts: &Options) -> String {
    let mut claude = format!("exec {CLAUDE}");
    if opts.resume_latest {
        claude += " --continue";
    }
    if let Some(id) = &opts.resume {
        claude += &format!(" --resume {}", crate::exec::q(id));
    }
    claude += " --add-dir \"$AKAO_REPO_ROOT\"";
    let mut argv: Vec<String> = ["docker", "-H", &format!("ssh://{nick}"), "exec", "-it"]
        .map(String::from)
        .into();
    if let Some(cwd) = &opts.cwd {
        argv.extend(["-w".into(), cwd.clone()]);
    }
    argv.extend([container.to_string(), "bash".into(), "-lc".into(), claude]);
    show(&argv)
}

pub fn run(state: &State, r: &Runner, opts: &Options) -> Result<()> {
    let host = state.host(&opts.nick)?;
    let name = opts.name.strip_prefix("akao_").unwrap_or(&opts.name);
    if !state::valid_name(name) {
        bail!("invalid container name '{}'", opts.name);
    }
    if opts.resume_latest && opts.resume.is_some() {
        bail!("give --continue or --resume <id>, not both");
    }
    let container = format!("akao_{name}");
    let tmux = |args: &[&str]| -> Vec<String> {
        std::iter::once("tmux")
            .chain(args.iter().copied())
            .map(String::from)
            .collect()
    };

    // The session: given, else the one tmux calls current (this client's, or the most recent).
    let session = match &opts.session {
        Some(s) => s.trim_end_matches(':').to_string(),
        None => r
            .query(&tmux(&["display-message", "-p", "#{session_name}"]))
            .context("no tmux session to open the window in (start one, or give --session)")?
            .trim()
            .to_string(),
    };
    if r.probe(&tmux(&["has-session", "-t", &format!("={session}")]))?
        .is_none()
    {
        bail!("no tmux session '{session}'");
    }
    let windows = r.query(&tmux(&[
        "list-windows",
        "-t",
        &format!("={session}:"),
        "-F",
        "#{window_name}",
    ]))?;
    if windows.lines().any(|w| w == name) {
        bail!(
            "tmux session '{session}' already has a window '{name}' (one agent per window): \
             read it with `tmux capture-pane -p -t ={session}:{name}`, or close it first"
        );
    }

    let endpoint = format!("ssh://{}", host.nick);
    let docker = |args: &[&str]| -> Vec<String> {
        ["docker", "-H", &endpoint]
            .iter()
            .chain(args)
            .map(|s| s.to_string())
            .collect()
    };
    let running = r
        .query(&docker(&[
            "container",
            "inspect",
            "--format",
            "{{.State.Running}}",
            &container,
        ]))
        .with_context(|| format!("cannot inspect {container} on {}", host.nick))?;
    if running.trim() != "true" {
        bail!(
            "{container} is not running on {} (`akao init {} {name}` brings it up)",
            host.nick,
            host.nick
        );
    }
    if let Some(cwd) = &opts.cwd {
        if r.probe(&docker(&["exec", &container, "test", "-d", cwd]))?.is_none() {
            bail!("{cwd} is not a directory in {container} (`akao task` makes a task dir)");
        }
    }

    let config_root = format!("AKAO_CONFIG_ROOT={}", state.root.display());
    let command = window_command(&host.nick, &container, opts);
    r.run(&tmux(&[
        "new-window",
        "-d",
        "-P",
        "-F",
        "#{session_name}:#{window_index} #{window_id}",
        "-t",
        &format!("={session}:"),
        "-e",
        &config_root,
        "-n",
        name,
        &command,
    ]))?;
    if !r.dry_run {
        println!(
            "opened {container}'s agent in window '{name}' (the id above): answer its trust dialog, \
             then send the handshake (controller skill, \"The worker's agent\")"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> Options {
        Options {
            nick: "h13-16".into(),
            name: "exp".into(),
            cwd: None,
            resume_latest: false,
            resume: None,
            session: None,
        }
    }

    #[test]
    fn the_window_runs_claude_in_the_workers_login_shell() {
        assert_eq!(
            window_command("h13-16", "akao_exp", &opts()),
            "docker -H ssh://h13-16 exec -it akao_exp bash -lc \
             'exec /root/.local/bin/claude --add-dir \"$AKAO_REPO_ROOT\"'"
        );
        let o = Options {
            cwd: Some("/2026/ww43/exp/0002_a b".into()),
            resume_latest: true,
            ..opts()
        };
        let cmd = window_command("h13-16", "akao_exp", &o);
        assert!(cmd.contains("-w '/2026/ww43/exp/0002_a b' akao_exp"), "{cmd}");
        assert!(cmd.contains("claude --continue --add-dir"), "{cmd}");
        // What the console's shell runs splits back into the argv, $AKAO_REPO_ROOT unexpanded.
        let words = shlex::split(&cmd).unwrap();
        assert_eq!(
            words.last().unwrap(),
            "exec /root/.local/bin/claude --continue --add-dir \"$AKAO_REPO_ROOT\""
        );
    }
}
