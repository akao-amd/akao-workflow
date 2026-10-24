//! Subprocess execution. Every command is echoed before it runs; under --dry-run,
//! mutating commands are only echoed while read-only probes still run.

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Shell prelude: escalate with passwordless sudo if the login is not root and has it;
/// otherwise write as the login (a box whose dirs it owns; elsewhere the write fails loudly).
/// Prepend to remote scripts; use `$S cmd` afterwards.
pub const ESCALATE: &str = r#"if [ "$(id -u)" -ne 0 ] && sudo -n true 2>/dev/null; then S="sudo -n"; else S=; fi; "#;

/// Shell-quote a single token. Falls back to the raw string if quoting fails.
pub fn q(s: &str) -> String {
    shlex::try_quote(s)
        .map(|c| c.into_owned())
        .unwrap_or_else(|_| s.to_string())
}

/// Every ssh of akao's carries a command: a host entry's RemoteCommand (a login shell that
/// exports another HOME) would make ssh refuse it, and its RequestTTY force would put a tty
/// under binary pipes.  So akao's scripts never rely on the login's HOME: host paths come
/// from hosts.tsv (host_home).
pub const NO_LOGIN_SHELL: [&str; 4] = ["-o", "RemoteCommand=none", "-o", "RequestTTY=no"];

/// The default ssh gives up on a box that does not answer (sshd restarted, network gone)
/// instead of hanging: no connection in 10 s, or no answer to keepalives for a minute (a long
/// `akao cp` survives a blip).  It bounds connecting and dead connections, not how long a
/// command may run.  Given on the command line, these beat the host's own ssh config.  Not
/// BatchMode: a login may need a password typed by hand.  $AKAO_SSH replaces it all.
pub const FAIL_FAST: [&str; 6] = [
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
];

pub struct Runner {
    pub dry_run: bool,
    /// ssh argv prefix, from $AKAO_SSH (default "ssh" with FAIL_FAST and akao's -F).
    ssh: Vec<String>,
}

pub fn show(argv: &[String]) -> String {
    shlex::try_join(argv.iter().map(String::as_str)).unwrap_or_else(|_| argv.join(" "))
}

fn command(argv: &[String]) -> Command {
    let mut c = Command::new(&argv[0]);
    c.args(&argv[1..]);
    c
}

impl Runner {
    pub fn new(dry_run: bool) -> Result<Runner> {
        // If the caller has set AKAO_SSH, use it verbatim (they own the -F too).
        // Otherwise build "ssh <FAIL_FAST> [-F <config>]" from the default, adding -F when
        // $AKAO_CONFIG_ROOT/.ssh/config exists.
        let ssh = match std::env::var("AKAO_SSH") {
            Ok(s) if !s.trim().is_empty() => shlex::split(&s).context("cannot parse $AKAO_SSH")?,
            _ => {
                let mut base = vec!["ssh".to_string()];
                base.extend(FAIL_FAST.map(String::from));
                if let Ok(root) = std::env::var("AKAO_CONFIG_ROOT") {
                    let cfg = PathBuf::from(root).join(".ssh/config");
                    if cfg.exists() {
                        base.extend(["-F".to_string(), cfg.to_string_lossy().into_owned()]);
                    }
                }
                base
            }
        };
        Ok(Runner { dry_run, ssh })
    }

    /// argv for running a POSIX shell script on `nick`.
    pub fn ssh_argv(&self, nick: &str, script: &str) -> Vec<String> {
        let mut argv = self.ssh.clone();
        argv.extend(NO_LOGIN_SHELL.map(String::from));
        argv.extend([nick.into(), "--".into()]);
        argv.push(show(&["sh".into(), "-c".into(), script.into()]));
        argv
    }

    pub fn ssh_config_argv(&self, nick: &str) -> Vec<String> {
        let mut argv = self.ssh.clone();
        argv.extend(["-G".into(), nick.into()]);
        argv
    }

    /// Human-readable form of a command: ssh commands built by ssh_argv print as
    /// `[nick] <script>`, everything else shell-quoted.
    pub fn display(&self, argv: &[String]) -> String {
        let (n, o) = (self.ssh.len(), NO_LOGIN_SHELL.len());
        if argv.len() == n + o + 3
            && argv[..n] == self.ssh[..]
            && argv[n..n + o] == NO_LOGIN_SHELL
            && argv[n + o + 1] == "--"
        {
            if let Some(words) = shlex::split(&argv[n + o + 2]) {
                if let [sh, c, script] = &words[..] {
                    if sh == "sh" && c == "-c" {
                        return format!("[{}] {script}", argv[n + o]);
                    }
                }
            }
        }
        show(argv)
    }

    /// Run a mutating command, streaming its output. Skipped under --dry-run.
    pub fn run(&self, argv: &[String]) -> Result<()> {
        println!("  + {}", self.display(argv));
        if self.dry_run {
            return Ok(());
        }
        let status = command(argv)
            .status()
            .with_context(|| format!("cannot start {}", argv[0]))?;
        if !status.success() {
            bail!("command failed ({status}): {}", show(argv));
        }
        Ok(())
    }

    /// Run a mutating command with already-read input. Skipped under --dry-run.
    pub fn input(&self, argv: &[String], bytes: &[u8]) -> Result<()> {
        println!("  + {} (brief on stdin)", self.display(argv));
        if self.dry_run {
            return Ok(());
        }
        let mut child = command(argv)
            .stdin(Stdio::piped())
            .spawn()
            .with_context(|| format!("cannot start {}", argv[0]))?;
        let written = child.stdin.take().unwrap().write_all(bytes);
        let status = child.wait()?;
        if !status.success() {
            bail!("command failed ({status}): {}", show(argv));
        }
        written.context("sending brief on stdin")
    }

    /// Run `left | right`. Skipped under --dry-run.
    pub fn pipe(&self, left: &[String], right: &[String]) -> Result<()> {
        println!("  + {} | {}", self.display(left), self.display(right));
        if self.dry_run {
            return Ok(());
        }
        let mut l = command(left)
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("cannot start {}", left[0]))?;
        let r_status = command(right)
            .stdin(l.stdout.take().unwrap())
            .status()
            .with_context(|| format!("cannot start {}", right[0]))?;
        let l_status = l.wait()?;
        if !l_status.success() {
            bail!("command failed ({l_status}): {}", show(left));
        }
        if !r_status.success() {
            bail!("command failed ({r_status}): {}", show(right));
        }
        Ok(())
    }

    /// Run a read-only probe (also under --dry-run) and capture stdout.
    /// Returns Ok(None) when the command exits non-zero; its stderr is discarded.
    pub fn probe(&self, argv: &[String]) -> Result<Option<String>> {
        self.capture(argv, Stdio::null())
    }

    /// Like probe, but a non-zero exit is an error and stderr is shown.
    pub fn query(&self, argv: &[String]) -> Result<String> {
        self.capture(argv, Stdio::inherit())?
            .with_context(|| format!("command failed: {}", show(argv)))
    }

    fn capture(&self, argv: &[String], stderr: Stdio) -> Result<Option<String>> {
        println!("  ? {}", self.display(argv));
        let out = command(argv)
            .stdin(Stdio::null())
            .stderr(stderr)
            .output()
            .with_context(|| format!("cannot start {}", argv[0]))?;
        Ok(out
            .status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `$S` under ESCALATE for a login of uid `uid` whose `sudo -n` exits `sudo`.
    fn escalate_as(uid: u32, sudo: i32) -> String {
        use std::os::unix::fs::PermissionsExt;
        let bin = std::env::temp_dir().join(format!("akao-escalate-{uid}-{sudo}-{}", std::process::id()));
        std::fs::create_dir_all(&bin).unwrap();
        for (name, body) in [("id", format!("echo {uid}")), ("sudo", format!("exit {sudo}"))] {
            let f = bin.join(name);
            std::fs::write(&f, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let out = Command::new("sh")
            .args(["-c", &format!("{ESCALATE}echo \"[$S]\"")])
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&bin);
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn escalate_uses_sudo_only_when_it_works() {
        assert_eq!(escalate_as(0, 0), "[]");
        assert_eq!(escalate_as(1000, 0), "[sudo -n]");
        // No passwordless sudo (a GB300 box): write as the login.
        assert_eq!(escalate_as(1000, 1), "[]");
    }

    #[test]
    fn ssh_script_is_one_quoted_arg() {
        let r = Runner {
            dry_run: true,
            ssh: vec!["ssh".into()],
        };
        let argv = r.ssh_argv("h", "mkdir -p '/a b'");
        // A host's RemoteCommand (a login shell) never replaces akao's command.
        assert_eq!(
            argv[..7],
            ["ssh", "-o", "RemoteCommand=none", "-o", "RequestTTY=no", "h", "--"]
        );
        // What the remote shell sees must split back into exactly sh -c <script>.
        assert_eq!(shlex::split(&argv[7]).unwrap(), ["sh", "-c", "mkdir -p '/a b'"]);
        assert_eq!(r.display(&argv), "[h] mkdir -p '/a b'");
    }
}
