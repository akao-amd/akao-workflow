//! `akao cp <src> <dst>`: copy a path from one box to another.
//!
//! Addresses are `nick:rel-path` (relative to `<host_home>/<year>`) or a bare
//! local path (no colon).  Either side can be local or remote, in any combination.
//!
//! Semantics: `cp -a` style — the source **basename** is placed **inside** the
//! destination directory.  The destination is always created if absent.
//! Files are overwritten; there is no `--delete`.
//!
//! Implementation: `tar -C <src-parent> -czf - <src-base>  |  tar -xzf - -C <dst>`,
//! where each side is wrapped in `ssh nick -- sh -c '...'` when remote.
//! Writing to a remote host uses `sudo -n` escalation (skipped when already root).

use crate::exec::{q, Runner, ESCALATE};
use crate::state::{self, State};
use anyhow::{bail, Result};

/// A `nick:rel-path` or bare local path.
pub struct Addr {
    /// None means local (this machine).
    pub nick: Option<String>,
    /// Remote: relative to `<host_home>/<year>`.  Local: literal path.
    pub rel: String,
}

impl Addr {
    pub fn parse(s: &str) -> Addr {
        match s.find(':') {
            Some(i) => Addr {
                nick: Some(s[..i].to_string()),
                rel: s[i + 1..].to_string(),
            },
            None => Addr {
                nick: None,
                rel: s.to_string(),
            },
        }
    }

    /// Full absolute path on the respective host.
    fn full_path(&self, state: &State) -> Result<String> {
        match &self.nick {
            None => Ok(self.rel.clone()),
            Some(nick) => {
                let host = state.host(nick)?;
                let year = state::work_year();
                let rel = self.rel.trim_start_matches('/');
                Ok(format!("{}/{}/{}", host.home(), year, rel))
            }
        }
    }

    /// `(parent_dir, basename)` of the full path.
    fn split_path(&self, state: &State) -> Result<(String, String)> {
        let full = self.full_path(state)?;
        let trimmed = full.trim_end_matches('/');
        match trimmed.rfind('/') {
            Some(i) => Ok((trimmed[..i].to_string(), trimmed[i + 1..].to_string())),
            None => bail!("cannot determine parent of '{trimmed}'"),
        }
    }

    fn label(&self, state: &State) -> Result<String> {
        let full = self.full_path(state)?;
        Ok(match &self.nick {
            None => full,
            Some(nick) => format!("{nick}:{full}"),
        })
    }
}

pub fn run(state: &State, r: &Runner, src_str: &str, dst_str: &str) -> Result<()> {
    let src = Addr::parse(src_str);
    let dst = Addr::parse(dst_str);

    let (src_parent, src_base) = src.split_path(state)?;
    let dst_full = dst.full_path(state)?;

    println!("akao cp: {} -> {}", src.label(state)?, dst.label(state)?);

    // --- source: tar the basename out of its parent ---
    // Reading doesn't need escalation — world-readable is the norm.
    let read_script = format!("tar -C {} -czf - {}", q(&src_parent), q(&src_base));
    let tar_read: Vec<String> = match &src.nick {
        None => vec!["sh".into(), "-c".into(), read_script],
        Some(nick) => r.ssh_argv(nick, &read_script),
    };

    // --- destination: mkdir + extract (overwrite) ---
    // Writing into a root-owned /2026 on a remote host needs escalation.
    let dst_q = q(&dst_full);
    let tar_write: Vec<String> = match &dst.nick {
        None => {
            // Local: we are root inside the container, no sudo needed.
            let script = format!("mkdir -p {dst_q} && tar --owner=0 --group=0 -xzf - -C {dst_q}");
            vec!["sh".into(), "-c".into(), script]
        }
        Some(nick) => {
            let script = format!("{ESCALATE}$S mkdir -p {dst_q} && $S tar --owner=0 --group=0 -xzf - -C {dst_q}");
            r.ssh_argv(nick, &script)
        }
    };

    r.pipe(&tar_read, &tar_write)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_remote() {
        let a = Addr::parse("m15-21:ww41/dsv4/0001_results");
        assert_eq!(a.nick.as_deref(), Some("m15-21"));
        assert_eq!(a.rel, "ww41/dsv4/0001_results");
    }

    #[test]
    fn parse_local() {
        let a = Addr::parse("/tmp/foo/bar");
        assert!(a.nick.is_none());
        assert_eq!(a.rel, "/tmp/foo/bar");
    }

    #[test]
    fn parse_local_no_slash() {
        // A nick without a colon is ambiguous but treated as local.
        let a = Addr::parse("relative/path");
        assert!(a.nick.is_none());
    }

    #[test]
    fn parse_colon_in_path() {
        // Only the first colon is the nick separator.
        let a = Addr::parse("m15-21:ww41/a:b");
        assert_eq!(a.nick.as_deref(), Some("m15-21"));
        assert_eq!(a.rel, "ww41/a:b");
    }
}
