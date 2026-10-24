//! `akao rm <nick> <name>...`: retire workers — stop and remove akao_<name> on a box, and with
//! `--home` its container home.  Artifact roots under /<year> are never touched.
//!
//! Like init, every step checks first: an absent container (a successful `ps -a` that does not
//! list it) is reported, not an error, so running it again finishes what an earlier run left.
//! A failure with one name does not stop the others; the command fails at the end.

use crate::exec::{q, Runner, ESCALATE};
use crate::state::{self, State};
use anyhow::{bail, Context, Result};
use std::time::{Duration, Instant};

/// How long `docker stop` waits for the container's processes before killing them.
const STOP_SECS: &str = "30";
/// How long the container may take to disappear after its stop (`--rm` removes it
/// asynchronously).
const GONE_WITHIN: Duration = Duration::from_secs(60);

pub struct Options {
    pub nick: String,
    pub names: Vec<String>,
    pub home: bool,
}

/// `docker -H ssh://<nick> ...`: through the endpoint, as init's step 1 (no context needed,
/// none changed).
struct Docker<'a> {
    r: &'a Runner,
    program: &'a str,
    endpoint: String,
}

impl Docker<'_> {
    fn argv(&self, args: &[&str]) -> Vec<String> {
        [self.program, "-H", &self.endpoint]
            .iter()
            .chain(args)
            .map(|s| s.to_string())
            .collect()
    }

    /// The id of container `name`, None when a successful `ps -a` does not list it.
    fn id(&self, name: &str) -> Result<Option<String>> {
        let filter = format!("name=^/?{name}$");
        let out = self.r.query(&self.argv(&[
            "ps",
            "-a",
            "--no-trunc",
            "--filter",
            &filter,
            "--format",
            "{{.ID}} {{.Names}}",
        ]))?;
        Ok(out.lines().find_map(|l| match l.trim().split_once(' ') {
            Some((id, n)) if n == name => Some(id.to_string()),
            _ => None,
        }))
    }

    /// The state of container `id`, None once it is gone (proved by `ps -a`).
    fn status(&self, name: &str, id: &str) -> Result<Option<String>> {
        let inspect = self.argv(&["container", "inspect", "--format", "{{.State.Status}}", id]);
        if let Some(s) = self.r.probe(&inspect)? {
            return Ok(Some(s.trim().to_string()));
        }
        match self.id(name)? {
            None => Ok(None),
            Some(now) if now.starts_with(id) || id.starts_with(&now) => {
                bail!("cannot inspect {name} ({id}), though docker lists it")
            }
            Some(now) => bail!("{name} was replaced meanwhile (now {now}, was {id}); leaving it"),
        }
    }
}

/// Stop and remove one container; Ok(false) when it was not there.
fn retire(d: &Docker, nick: &str, c: &str) -> Result<bool> {
    retire_with_limits(d, nick, c, GONE_WITHIN, Duration::from_secs(2))
}

fn retire_with_limits(d: &Docker, nick: &str, c: &str, gone_within: Duration, poll_interval: Duration) -> Result<bool> {
    let Some(id) = d.id(c)? else {
        println!("  {c}: not there");
        return Ok(false);
    };
    let Some(status) = d.status(c, &id)? else {
        println!("  {c}: gone meanwhile");
        return Ok(false);
    };
    println!("  {c}: {id} {status}");
    if d.r.dry_run {
        println!("  {c}: would stop/remove as needed and verify absence");
    }
    // A failing stop or rm may only have lost a race with --rm: the wait below decides.
    let act = |args: &[&str]| {
        if let Err(e) = d.r.run(&d.argv(args)) {
            println!("  ({e:#}; checking whether it is gone)");
        }
    };
    match status.as_str() {
        "running" | "restarting" => act(&["stop", "-t", STOP_SECS, &id]),
        // A paused container's processes cannot take a stop signal.
        "paused" => act(&["rm", "-f", &id]),
        "removing" => {}
        _ => act(&["rm", &id]),
    }
    if d.r.dry_run {
        return Ok(true);
    }
    let start = Instant::now();
    // Only a stop leaves an rm still to send.
    let mut removed = !matches!(status.as_str(), "running" | "restarting");
    while let Some(now) = d.status(c, &id)? {
        if start.elapsed() > gone_within {
            bail!(
                "{c} is still {now} {}s after its stop; see `docker -H ssh://{nick} ps -a`",
                gone_within.as_secs()
            );
        }
        // Made without --rm, a stopped container stays: remove it once.
        if !removed && matches!(now.as_str(), "exited" | "created" | "dead") {
            act(&["rm", &id]);
            removed = true;
            continue;
        }
        std::thread::sleep(poll_interval);
    }
    println!("  {c}: removed");
    Ok(true)
}

/// Whether bind source `m` exposes `home`: the same path, inside it, or above it.
fn exposes(m: &str, home: &str) -> bool {
    if m.is_empty() {
        return false;
    }
    let m = m.trim_end_matches('/');
    m == home || m.starts_with(&format!("{home}/")) || home.starts_with(&format!("{m}/"))
}

pub fn run(state: &State, r: &Runner, opts: &Options) -> Result<()> {
    let host = state.host(&opts.nick)?;
    let nick = host.nick.clone();
    let containers = opts
        .names
        .iter()
        .map(|n| {
            let name = n.strip_prefix("akao_").unwrap_or(n);
            if !state::valid_name(name) {
                bail!("invalid container name '{n}'");
            }
            Ok(format!("akao_{name}"))
        })
        .collect::<Result<Vec<_>>>()?;
    println!(
        "akao rm: {} on {nick}{}",
        containers.join(" "),
        if r.dry_run { " [dry run]" } else { "" }
    );
    let d = Docker {
        r,
        program: "docker",
        endpoint: format!("ssh://{nick}"),
    };
    r.query(&d.argv(&["version", "--format", "{{.Server.Version}}"]))
        .with_context(|| format!("cannot reach docker on {nick} (its message is above)"))?;

    let mut failed = Vec::new();
    for c in &containers {
        if let Err(e) = retire(&d, &nick, c) {
            println!("  {c}: FAILED: {e:#}");
            failed.push(c.clone());
        }
    }

    if opts.home {
        // Every container on the box, stopped and foreign ones too: a home one of them still
        // mounts (or a dir above or below it) stays.  No census, no deletion.
        let ids = r.query(&d.argv(&["ps", "-aq"]))?;
        let ids: Vec<&str> = ids.split_whitespace().collect();
        let mounted = if ids.is_empty() {
            String::new()
        } else {
            let mut args = vec![
                "container",
                "inspect",
                "--format",
                "{{range .Mounts}}{{if eq .Type \"bind\"}}{{println .Source}}{{end}}{{end}}",
            ];
            args.extend(&ids);
            r.query(&d.argv(&args))
                .context("cannot read what the box's containers mount; no home removed")?
        };
        for c in &containers {
            if failed.contains(c) {
                println!("  {c}: home kept because retirement failed");
                continue;
            }
            let home = format!("{}/container_home/{c}", host.home());
            let present = r.query(&r.ssh_argv(
                &nick,
                &format!("if test -d {}; then echo present; else echo absent; fi", q(&home)),
            ));
            match present {
                Ok(out) if out.trim() == "present" => {}
                Ok(out) if out.trim() == "absent" => {
                    println!("  {home}: not there");
                    continue;
                }
                result => {
                    println!("  {home}: FAILED to check existence: {result:?}");
                    failed.push(format!("{c} (home)"));
                    continue;
                }
            }
            if let Some(m) = mounted.lines().find(|m| exposes(m, &home)) {
                println!("  {home}: kept, a container mounts {m}");
                failed.push(format!("{c} (home)"));
                continue;
            }
            match r.run(&r.ssh_argv(&nick, &format!("{ESCALATE}$S rm -rf -- {}", q(&home)))) {
                Ok(()) if r.dry_run => {}
                Ok(()) => println!("  {home}: removed"),
                Err(e) => {
                    println!("  {home}: FAILED: {e:#}");
                    failed.push(format!("{c} (home)"));
                }
            }
        }
    }
    if !failed.is_empty() {
        bail!("not retired: {}", failed.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_container_stuck_removing_reports_timeout_without_another_mutation() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("akao-rm-timeout-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let program = root.join("docker");
        fs::write(&program, "#!/bin/sh\ncase \"$*\" in\n*'ps -a'*) echo 'old-id akao_x';;\n*inspect*) echo removing;;\n*) exit 99;;\nesac\n").unwrap();
        fs::set_permissions(&program, fs::Permissions::from_mode(0o755)).unwrap();
        let r = Runner::new(false).unwrap();
        let d = Docker {
            r: &r,
            program: program.to_str().unwrap(),
            endpoint: "ssh://scratch".into(),
        };
        let err = retire_with_limits(&d, "scratch", "akao_x", Duration::ZERO, Duration::ZERO).unwrap_err();
        assert!(err.to_string().contains("still removing"), "{err:#}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_mount_exposes_the_home_at_or_around_it() {
        let home = "/home/akao/container_home/akao_x";
        assert!(exposes("/home/akao/container_home/akao_x", home));
        assert!(exposes("/home/akao/container_home/akao_x/", home));
        assert!(exposes("/home/akao/container_home/akao_x/.cache", home));
        assert!(exposes("/home/akao/container_home", home));
        assert!(!exposes("/home/akao/container_home/akao_xy", home));
        assert!(!exposes("/home/akao/2026", home));
        assert!(exposes("/", home));
        assert!(!exposes("", home));
        assert!(exposes("/host with space ", "/host with space /container_home/akao_x"));
        assert!(!exposes("/host with space", "/host with space /container_home/akao_x"));
    }
}
