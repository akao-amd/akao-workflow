//! Subprocess execution. Every command is echoed before it runs; under --dry-run,
//! mutating commands are only echoed while read-only probes still run.

use anyhow::{bail, Context, Result};
use std::process::{Command, Stdio};

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
        let ssh = match std::env::var("AKAO_SSH") {
            Ok(s) if !s.trim().is_empty() => shlex::split(&s).context("cannot parse $AKAO_SSH")?,
            _ => vec!["ssh".into()],
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

    /// Run a mutating command, streaming its output. Skipped under --dry-run.
    pub fn run(&self, argv: &[String]) -> Result<()> {
        println!("  + {}", show(argv));
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
        println!("  + {} | {}", show(left), show(right));
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
    /// Returns Ok(None) when the command exits non-zero.
    pub fn probe(&self, argv: &[String]) -> Result<Option<String>> {
        println!("  ? {}", show(argv));
        let out = command(argv)
            .stdin(Stdio::null())
            .stderr(Stdio::inherit())
            .output()
            .with_context(|| format!("cannot start {}", argv[0]))?;
        Ok(out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned()))
    }

    /// Like probe, but a non-zero exit is an error.
    pub fn query(&self, argv: &[String]) -> Result<String> {
        self.probe(argv)?.with_context(|| format!("command failed: {}", show(argv)))
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
    }
}
