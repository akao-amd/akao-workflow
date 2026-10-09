//! `akao init <nick> <name>`: bring up worker container akao_<name> on a remote box.
//!
//! Every step checks what already exists and reuses it, so re-running init on a
//! half-initialized worker resumes instead of failing.

use crate::exec::{q, show, Runner, ESCALATE};
use crate::state::{self, Host, State};
use anyhow::{bail, Result};
use std::path::Path;

/// The fixed part of every `docker run`; per-host mounts and rest args follow it.
const DOCKER_RUN_SKELETON: &[&str] = &[
    "run", "--rm", "-d", "--privileged", "--ulimit", "nofile=1048576", "--network=host",
    "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
    "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
    "-e", "PYTHONPATH=", "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8", "-e", "TERM=tmux-256color",
    "--ipc=host", "--shm-size=32g",
];

const APT_PACKAGES: &[&str] = &["vim", "less", "tmux", "docker.io"];

/// Excluded from the control-plane deploy.
const DEPLOY_EXCLUDES: &[&str] = &[".git", "__pycache__", "*.pyc", ".claude"];

pub struct Options {
    pub nick: String,
    pub name: String,
    pub week: Option<String>,
    pub skip_setup: bool,
}

/// Everything init needs, resolved from the arguments and state up front.
pub struct Plan {
    pub host: Host,
    /// docker context name (= nick)
    pub context: String,
    /// akao_<name>
    pub container: String,
    pub image: String,
    pub year: String,
    /// Container working directory, /<year>/<week>/<name>; same path under <host_home> on the host.
    pub workdir: String,
    /// <host_home>/container_home/akao_<name>
    pub host_container_home: String,
}

impl Plan {
    pub fn new(state: &State, opts: &Options) -> Result<Plan> {
        if !state::valid_name(&opts.name) {
            bail!("invalid container name '{}'", opts.name);
        }
        let name = opts.name.strip_prefix("akao_").unwrap_or(&opts.name);
        let host = state.host(&opts.nick)?;
        let image = match &host.image {
            Some(i) => i.clone(),
            None => state.require("default_image")?,
        };
        let week = match &opts.week {
            Some(w) => w.clone(),
            None => state::work_week(),
        };
        if !(week.len() == 4 && week.starts_with("ww") && week[2..].bytes().all(|b| b.is_ascii_digit())) {
            bail!("week must look like ww41, got '{week}'");
        }
        let year = state::work_year();
        let container = format!("akao_{name}");
        Ok(Plan {
            context: host.nick.clone(),
            workdir: format!("/{year}/{week}/{name}"),
            host_container_home: format!("{}/container_home/{container}", host.host_home),
            container,
            image,
            year,
            host,
        })
    }

    fn host_year_dir(&self) -> String {
        format!("{}/{}", self.host.host_home, self.year)
    }

    pub fn docker(&self, args: &[&str]) -> Vec<String> {
        let mut argv: Vec<String> = vec!["docker".into(), "--context".into(), self.context.clone()];
        argv.extend(args.iter().map(|s| s.to_string()));
        argv
    }

    pub fn docker_run_argv(&self) -> Result<Vec<String>> {
        let mut argv = self.docker(DOCKER_RUN_SKELETON);
        let h = &self.host;
        argv.push(format!("--name={}", self.container));
        for (src, dst) in [
            (h.model_path.clone(), "/model".to_string()),
            (h.docker_sock().to_string(), "/var/run/docker.sock".to_string()),
            (self.host_year_dir(), format!("/{}", self.year)),
            (self.host_container_home.clone(), "/root".to_string()),
        ] {
            argv.extend(["-v".into(), format!("{src}:{dst}")]);
        }
        argv.extend(["-w".into(), self.workdir.clone()]);
        argv.extend(h.rest_args()?);
        argv.extend([self.image.clone(), "sleep".into(), "infinity".into()]);
        Ok(argv)
    }

    fn exec(&self, opts: &[&str], script: &str) -> Vec<String> {
        let mut args = vec!["exec"];
        args.extend(opts);
        args.extend([self.container.as_str(), "bash", "-c", script]);
        self.docker(&args)
    }
}

/// Docker endpoint for `nick`, carrying the login user from `ssh -G` output.
/// docker runs plain `ssh` without our -F, so it would otherwise use the user from
/// ~/.ssh/config (or the local user) instead of the one in $AKAO_CONFIG_ROOT/.ssh/config.
fn docker_endpoint(ssh_g: &str, nick: &str) -> String {
    match ssh_g.lines().find_map(|l| l.strip_prefix("user ")) {
        Some(user) => format!("ssh://{}@{nick}", user.trim()),
        None => format!("ssh://{nick}"),
    }
}

fn tar_create(dir: &Path, paths: &[String], excludes: &[&str]) -> Vec<String> {
    let mut argv: Vec<String> = vec!["tar".into(), "-C".into(), dir.display().to_string()];
    argv.extend(["--owner=0".into(), "--group=0".into()]);
    argv.extend(excludes.iter().map(|e| format!("--exclude={e}")));
    argv.push("-czf".into());
    argv.push("-".into());
    argv.extend(paths.iter().cloned());
    argv
}

struct Steps {
    n: usize,
    total: usize,
}

impl Steps {
    fn next(&mut self, what: &str) {
        self.n += 1;
        println!("[{}/{}] {what}", self.n, self.total);
    }
}

pub fn run(state: &State, r: &Runner, opts: &Options) -> Result<()> {
    let p = Plan::new(state, opts)?;
    let nick = &p.host.nick;
    println!(
        "akao init: {} on {nick}{}\n  image {}\n  host home {}, workdir {}",
        p.container,
        if r.dry_run { " [dry run]" } else { "" },
        p.image,
        p.host.host_home,
        p.workdir
    );
    let mut s = Steps { n: 0, total: 8 };

    s.next("resolve host");
    let cfg = r.query(&r.ssh_config_argv(nick))?;
    let hostname = cfg.lines().find_map(|l| l.strip_prefix("hostname ")).unwrap_or(nick);
    let endpoint = docker_endpoint(&cfg, nick);
    println!("  {nick} -> {hostname}, docker {endpoint}");

    s.next("prepare host directories");
    let script = format!("{ESCALATE}$S mkdir -p {} {}/container_home", q(&format!("{}{}", p.host.host_home, p.workdir)), q(&p.host.host_home));
    r.run(&r.ssh_argv(nick, &script))?;

    s.next("deploy control plane");
    if opts.skip_setup {
        println!("  skipped (--skip-setup)");
    } else {
        let src = state.require("deploy_src")?;
        let paths: Vec<String> = state.require("deploy_paths")?.split_whitespace().map(String::from).collect();
        for path in &paths {
            if !Path::new(&src).join(path).exists() {
                bail!("deploy path missing locally: {src}/{path}");
            }
        }
        let extract = format!("{ESCALATE}$S tar -xzf - -C {}", q(&p.host_year_dir()));
        r.pipe(&tar_create(Path::new(&src), &paths, DEPLOY_EXCLUDES), &r.ssh_argv(nick, &extract))?;
    }

    s.next("container home");
    let home = q(&p.host_container_home);
    if r.probe(&r.ssh_argv(nick, &format!("test -d {home}")))?.is_some() {
        println!("  reusing existing {}", p.host_container_home);
    } else {
        let template = state.home_template();
        if !template.is_dir() {
            bail!("home template missing: {}", template.display());
        }
        let extract = format!("{ESCALATE}$S mkdir -p {home} && $S tar -xzf - -C {home}");
        r.pipe(&tar_create(&template, &[".".into()], &[]), &r.ssh_argv(nick, &extract))?;
    }

    s.next("docker context");
    // docker's ssh helper cannot take -F, so a failure here would otherwise surface
    // later as a silently failed probe (e.g. "container does not exist").
    let version = r.query(&["docker", "-H", &endpoint, "version", "--format", "{{.Server.Version}}"].map(String::from))?;
    println!("  docker {} reachable", version.trim());
    let inspect: Vec<String> = ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}", &p.context]
        .map(String::from)
        .into();
    let host_arg = format!("host={endpoint}");
    match r.probe(&inspect)? {
        Some(ep) if ep.trim() == endpoint => println!("  reusing context {}", p.context),
        Some(ep) => {
            println!("  context {} points at {}; updating", p.context, ep.trim());
            r.run(&["docker", "context", "update", &p.context, "--docker", &host_arg].map(String::from))?
        }
        None => r.run(&["docker", "context", "create", &p.context, "--docker", &host_arg].map(String::from))?,
    }
    r.run(&["docker", "context", "use", &p.context].map(String::from))?;

    s.next("container");
    let status = r.probe(&p.docker(&["container", "inspect", "--format", "{{.State.Status}}", &p.container]))?;
    match status.as_deref().map(str::trim) {
        Some("running") => println!("  reusing running {}", p.container),
        Some("created" | "exited") => r.run(&p.docker(&["start", &p.container]))?,
        Some(other) => bail!("{} exists but is {other}; resolve it by hand", p.container),
        None => r.run(&p.docker_run_argv()?)?,
    }

    s.next("install packages and agents");
    if opts.skip_setup {
        println!("  skipped (--skip-setup)");
    } else {
        let have_all = APT_PACKAGES
            .iter()
            .map(|p| format!("command -v {} >/dev/null", if *p == "docker.io" { "docker" } else { p }))
            .collect::<Vec<_>>()
            .join(" && ");
        let utils = format!("/{}/utils", p.year);
        let script = format!(
            "{have_all} || {{ export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y {}; }}",
            APT_PACKAGES.join(" ")
        );
        r.run(&p.exec(&[], &script))?;
        r.run(&p.exec(&[], &format!("command -v gh >/dev/null || bash {utils}/install_gh.sh")))?;
        r.run(&p.exec(&[], &format!("test -x /root/.local/bin/claude || bash {utils}/agent.sh --yes")))?;
    }

    s.next("tmux controller window");
    if r.probe(&p.exec(&[], "tmux has-session"))?.is_some() {
        println!("  tmux already running in {}; left as is", p.container);
    } else {
        r.run(&p.exec(&[], &format!("tmux new-session -d -n controller -c {}", q(&p.workdir))))?;
        r.run(&p.exec(&[], "tmux send-keys -t :controller claude Enter"))?;
    }

    let attach = p.docker(&["exec", "-it", &p.container, "tmux", "attach"]);
    println!("done. attach with:\n  {}", show(&attach));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Plan {
        Plan {
            host: Host {
                nick: "f19-11".into(),
                image: None,
                model_path: "/mnt/raid/models".into(),
                docker_sock: Some("/data/docker.sock".into()),
                host_home: "/root/akao".into(),
                rest: Some("--shm-size=64g -e 'FOO=a b'".into()),
            },
            context: "f19-11".into(),
            container: "akao_exp".into(),
            image: "img:tag".into(),
            year: "2026".into(),
            workdir: "/2026/ww41/exp".into(),
            host_container_home: "/root/akao/container_home/akao_exp".into(),
        }
    }

    #[test]
    fn docker_run_layout() {
        let argv = plan().docker_run_argv().unwrap();
        let s = argv.join(" ");
        assert!(s.starts_with("docker --context f19-11 run --rm -d --privileged"));
        assert!(s.contains("--name=akao_exp"));
        assert!(s.contains("-v /mnt/raid/models:/model"));
        assert!(s.contains("-v /data/docker.sock:/var/run/docker.sock"));
        assert!(s.contains("-v /root/akao/2026:/2026"));
        assert!(s.contains("-v /root/akao/container_home/akao_exp:/root"));
        assert!(s.contains("-w /2026/ww41/exp"));
        // rest args come after the skeleton but before the image.
        assert!(argv.ends_with(&["--shm-size=64g", "-e", "FOO=a b", "img:tag", "sleep", "infinity"].map(String::from)));
        assert!(argv.contains(&"PYTHONPATH=".to_string()));
    }

    #[test]
    fn docker_endpoint_carries_user() {
        let g = "host h21-5\nuser akao\nhostname ctheliosp-rck-g02-h21-5.rck.dcgpu\nport 22\n";
        assert_eq!(docker_endpoint(g, "h21-5"), "ssh://akao@h21-5");
        assert_eq!(docker_endpoint("hostname x\n", "h21-5"), "ssh://h21-5");
    }

    #[test]
    fn tar_owner_and_excludes() {
        let argv = tar_create(Path::new("/2026"), &["skills".into()], DEPLOY_EXCLUDES);
        assert_eq!(argv[..5], ["tar", "-C", "/2026", "--owner=0", "--group=0"]);
        assert!(argv.contains(&"--exclude=.claude".to_string()));
        assert!(argv.ends_with(&["-czf".into(), "-".into(), "skills".into()]));
    }
}
