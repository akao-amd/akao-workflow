//! `akao cp <src> <dst>`: copy a path from one box to another.
//!
//! Addresses are `nick:rel-path` (relative to `<host_home>/<year>`) or a bare
//! local path (no colon).  Either side can be local or remote, in any combination.
//!
//! Semantics: `cp -a` style — the source **basename** is placed **inside** the
//! destination directory.  The destination is always created if absent.
//! Files are overwritten; there is no `--delete`.
//!
//! A destination of `nick:` (no path) keeps the source's path: it becomes the source's
//! parent (`same_path_parent`), so the boxes' trees still merge when the user rsyncs them
//! together.  `:` alone does the same on this machine (`/<year>/<parent>`), for a remote
//! source.  An explicit path is taken as given, `..` included.
//!
//! Implementation: `tar -C <src-parent> -czf - <src-base>  |  tar -xzf - -C <dst>`,
//! where each side is wrapped in `ssh nick -- sh -c '...'` when remote.
//! Remote reads and writes use `sudo -n` escalation (skipped when already root).

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

/// The destination path, relative to `<host_home>/<year>`, that keeps the source's path:
/// its parent, since the copy lands inside the destination.  A remote source gives its own
/// path; a local one must lie under this machine's `/<year>/`, the tree the boxes are
/// rsync'd into.
fn same_path_parent(src: &Addr, year: &str) -> Result<String> {
    let rel = match &src.nick {
        Some(_) => src.rel.trim_matches('/').to_string(),
        None => {
            let abs = std::path::absolute(&src.rel)?;
            let abs = abs.to_string_lossy();
            match abs.strip_prefix(&format!("/{year}/")) {
                Some(rel) => rel.trim_end_matches('/').to_string(),
                None => bail!(
                    "{abs} is not under /{year}/, so it has no path on a box to keep; \
                     name the destination path (nick:<path>)"
                ),
            }
        }
    };
    if rel.is_empty() {
        bail!("the whole /{year} tree has no parent to keep; name the destination path (nick:<path>)");
    }
    Ok(rel.rfind('/').map_or(String::new(), |i| rel[..i].to_string()))
}

pub fn run(state: &State, r: &Runner, src_str: &str, dst_str: &str) -> Result<()> {
    let src = Addr::parse(src_str);
    let mut dst = Addr::parse(dst_str);

    let same_path = dst.nick.is_some() && dst.rel.is_empty();
    if dst_str == ":" {
        // `:` alone: this machine, at the source's path under its /<year> (the superset).
        if src.nick.is_none() {
            bail!("{src_str} is already on this machine; `:` brings a box's path here");
        }
        let year = state::work_year();
        let parent = same_path_parent(&src, &year)?;
        dst = Addr {
            nick: None,
            rel: format!("/{year}/{parent}").trim_end_matches('/').to_string(),
        };
    } else if same_path {
        if dst.nick == src.nick {
            bail!("{src_str} is already at that path on {}", dst_str.trim_end_matches(':'));
        }
        dst.rel = same_path_parent(&src, &state::work_year())?;
    }

    let (src_parent, src_base) = src.split_path(state)?;
    let dst_full = dst.full_path(state)?;

    let note = if same_path { "  (the source's path)" } else { "" };
    println!("akao cp: {} -> {}{note}", src.label(state)?, dst.label(state)?);

    // --- source: tar the basename out of its parent ---
    // Worker artifacts may be root-only (for example admission receipts).
    let read_script = format!("tar -C {} -czf - {}", q(&src_parent), q(&src_base));
    let tar_read: Vec<String> = match &src.nick {
        None => vec!["sh".into(), "-c".into(), read_script],
        Some(nick) => r.ssh_argv(nick, &format!("{ESCALATE}$S {read_script}")),
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

    fn kept(src: &str) -> Result<String> {
        same_path_parent(&Addr::parse(src), "2026")
    }

    #[test]
    fn same_path_is_the_sources_parent() {
        assert_eq!(kept("f19-17:ww42/b0/8115_best").unwrap(), "ww42/b0");
        assert_eq!(kept("f19-17:/ww42/b0/8115_best/").unwrap(), "ww42/b0");
        assert_eq!(kept("f19-17:ww42/b0").unwrap(), "ww42");
        assert_eq!(kept("f19-17:ww42").unwrap(), "");
        assert_eq!(kept("/2026/ww42/a0/0003_x").unwrap(), "ww42/a0");
        assert_eq!(kept("/2026/ww42/a0/0003_x/").unwrap(), "ww42/a0");
    }

    #[test]
    fn same_path_refuses_what_has_no_place_on_a_box() {
        let e = kept("/tmp/0001_probe").unwrap_err().to_string();
        assert!(e.contains("not under /2026/"), "{e}");
        assert!(kept("/2026").is_err());
        let e = kept("f19-17:").unwrap_err().to_string();
        assert!(e.contains("whole /2026 tree"), "{e}");
    }

    #[test]
    fn parse_colon_in_path() {
        // Only the first colon is the nick separator.
        let a = Addr::parse("m15-21:ww41/a:b");
        assert_eq!(a.nick.as_deref(), Some("m15-21"));
        assert_eq!(a.rel, "ww41/a:b");
    }
}
