//! Subprocess execution. Every command is echoed before it runs; under --dry-run,
//! mutating commands are only echoed while read-only probes still run.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Shell prelude: escalate with passwordless sudo if the login is not root.
/// Prepend to remote scripts; use `$S cmd` afterwards.
pub const ESCALATE: &str = r#"if [ "$(id -u)" -eq 0 ]; then S=; else S="sudo -n"; fi; "#;

/// Shell-quote a single token. Falls back to the raw string if quoting fails.
pub fn q(s: &str) -> String {
    shlex::try_quote(s).map(|c| c.into_owned()).unwrap_or_else(|_| s.to_string())
}

pub struct Runner {
    pub dry_run: bool,
    /// ssh argv prefix, from $AKAO_SSH (default "ssh").
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
        // Otherwise build "ssh [-F <config>]" from the default, adding -F when
        // $AKAO_CONFIG_ROOT/.ssh/config exists.
        let ssh = match std::env::var("AKAO_SSH") {
            Ok(s) if !s.trim().is_empty() => shlex::split(&s).context("cannot parse $AKAO_SSH")?,
            _ => {
                let mut base = vec!["ssh".to_string()];
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
        let n = self.ssh.len();
        if argv.len() == n + 3 && argv[..n] == self.ssh[..] && argv[n + 1] == "--" {
            if let Some(words) = shlex::split(&argv[n + 2]) {
                if let [sh, c, script] = &words[..] {
                    if sh == "sh" && c == "-c" {
                        return format!("[{}] {script}", argv[n]);
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
        let status = command(argv).status().with_context(|| format!("cannot start {}", argv[0]))?;
        if !status.success() {
            bail!("command failed ({status}): {}", show(argv));
        }
        Ok(())
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
        self.capture(argv, Stdio::inherit())?.with_context(|| format!("command failed: {}", show(argv)))
    }

    fn capture(&self, argv: &[String], stderr: Stdio) -> Result<Option<String>> {
        println!("  ? {}", self.display(argv));
        let out = command(argv)
            .stdin(Stdio::null())
            .stderr(stderr)
            .output()
            .with_context(|| format!("cannot start {}", argv[0]))?;
        Ok(out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_script_is_one_quoted_arg() {
        let r = Runner { dry_run: true, ssh: vec!["ssh".into()] };
        let argv = r.ssh_argv("h", "mkdir -p '/a b'");
        assert_eq!(argv[..3], ["ssh", "h", "--"]);
        // What the remote shell sees must split back into exactly sh -c <script>.
        assert_eq!(shlex::split(&argv[3]).unwrap(), ["sh", "-c", "mkdir -p '/a b'"]);
        assert_eq!(r.display(&argv), "[h] mkdir -p '/a b'");
    }
}
