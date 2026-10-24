mod cp;
mod dashboard;
mod doctor;
mod exec;
mod init;
mod mirror;
mod state;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use state::{Host, State};

/// Local driver: initialize and control worker containers on remote boxes.
///
/// State lives in $AKAO_CONFIG_ROOT (config.toml, hosts.tsv, container_home/).
#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("AKAO_GIT_SHA"), ")"))]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Copy a path from one box to another (or local <-> remote)
    ///
    /// Addresses are `nick:rel-path` (relative to <host_home>/<year>) or a bare
    /// local path.  The source basename is placed inside the destination directory.
    /// Files are always overwritten; nothing is deleted.
    ///
    /// A destination of `nick:` alone keeps the source's path, so the boxes' trees still merge
    /// when rsync'd together.  That is the normal use:
    ///
    ///   akao cp h21-5:ww42/a0/0003_x f19-17:      # -> f19-17:ww42/a0/0003_x
    ///
    ///   akao cp h21-5:ww42/a0 f19-17:             # -> f19-17:ww42/a0
    ///
    /// A local source must then lie under /<year>/.  An explicit destination path is taken as
    /// given (`..` included); the copy lands inside it.
    Cp {
        /// Source address: `nick:rel-path` or a local path
        src: String,
        /// Destination address: `nick:` (the source's path), `nick:rel-path` or a local path
        dst: String,
        /// Print the commands without running them
        #[arg(long)]
        dry_run: bool,
    },
    /// Initialize worker container akao_<name> on a remote host
    Init {
        /// Host nick (a row in hosts.tsv, an ssh Host alias)
        nick: String,
        /// Container name, without the akao_ prefix
        name: String,
        /// The container's image; required for a new container.  Omitted, an existing
        /// akao_<name> keeps the image it runs (a refresh); one running another image fails
        #[arg(long)]
        image: Option<String>,
        #[command(flatten)]
        flags: InitFlags,
    },
    /// A worker for one InferenceX benchmark config (init with its image, plus a brief)
    ///
    /// Reads the entries of InferenceX's *master.yaml from this machine's clone (config
    /// infx_local) at --rev; or, with --dashboard, what the InferenceX dashboard measured
    /// (config infx_dashboard): a curve's latest run, its image, the commit it ran at and
    /// the server logs.  Without a host, lists the matches or previews the brief.
    #[command(
        after_help = "Examples:\n  akao mirror --conf gptoss,mi355 --rev 4699ab81a^     # list\n  \
                            akao mirror h21-17 --conf gptoss,mi355,atom --rev 4699ab81a^\n  \
                            akao mirror --dashboard --conf mi355,vllm             # the dashboard's curves\n  \
                            akao mirror n10-17 --dashboard --conf gptoss,mi355,vllm"
    )]
    Mirror {
        /// Host nick to bring the worker up on; omit to list or preview
        nick: Option<String>,
        /// Container name, without the akao_ prefix [default: the entry's name]
        name: Option<String>,
        /// Comma-separated terms that name one entry (parts of its name, runner, framework,
        /// model), or the entry's full name
        #[arg(long, value_name = "TERMS")]
        conf: String,
        /// InferenceX revision to read the configs at
        #[arg(long, default_value = "HEAD", conflicts_with = "dashboard")]
        rev: String,
        /// Mirror what the InferenceX dashboard measured, not a config of this machine's clone
        #[arg(long)]
        dashboard: bool,
        /// The dashboard's data as of this date (YYYY-MM-DD) [default: the latest]
        #[arg(long, value_name = "YYYY-MM-DD", requires = "dashboard")]
        date: Option<String>,
        /// The curve's GitHub Actions run to mirror [default: its latest]
        #[arg(long, value_name = "ID", requires = "dashboard")]
        run: Option<String>,
        #[command(flatten)]
        flags: InitFlags,
    },
    /// Manage the remote host table (hosts.tsv)
    #[command(subcommand)]
    Host(HostCmd),
    /// Show or change settings (config.toml)
    #[command(subcommand)]
    Config(ConfigCmd),
    /// Check this machine's prerequisites for akao (read-only)
    Doctor,
}

/// What `init` and `mirror` share.
#[derive(clap::Args)]
struct InitFlags {
    /// Work week of the artifact root /<year>/<week>/<name> [default: ISO week of today]
    #[arg(long, conflicts_with = "artifact_root")]
    week: Option<String>,
    /// The worker's artifact root, under /<year>/ [default: /<year>/<week>/<name>];
    /// pinned into the container as $AKAO_ARTIFACT_ROOT and its working directory
    #[arg(long, value_name = "DIR")]
    artifact_root: Option<String>,
    /// Skip the control-plane deploy and the package/agent installation
    #[arg(long)]
    skip_setup: bool,
    /// Print the commands that would change anything instead of running them
    #[arg(long)]
    dry_run: bool,
}

#[derive(Subcommand)]
enum HostCmd {
    /// List known hosts
    Ls,
    /// Add a host (or replace it with --force)
    Add {
        /// ssh Host alias from ~/.ssh/config
        nick: String,
        /// Host home directory (absolute); holds <year>/ and container_home/
        #[arg(long)]
        home: String,
        /// Model directory on the host (absolute), mounted as /model
        #[arg(long)]
        model: String,
        /// Docker socket on the host [default: /var/run/docker.sock]
        #[arg(long)]
        sock: Option<String>,
        /// Extra `docker run` args appended to the skeleton, as one shell-quoted string
        #[arg(long, allow_hyphen_values = true)]
        rest: Option<String>,
        /// Replace an existing entry
        #[arg(long)]
        force: bool,
    },
    /// Remove a host
    Rm { nick: String },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Show every setting with its effective value
    Ls,
    /// Print one setting's effective value
    Get { key: String },
    /// Set a setting
    Set { key: String, value: String },
    /// Revert a setting to its default
    Unset { key: String },
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("akao: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    if let Cmd::Doctor = cli.cmd {
        return doctor::run(); // reports a missing AKAO_CONFIG_ROOT itself
    }
    let mut state = State::load()?;
    match cli.cmd {
        Cmd::Cp { src, dst, dry_run } => {
            let runner = exec::Runner::new(dry_run)?;
            cp::run(&state, &runner, &src, &dst)
        }
        Cmd::Init {
            nick,
            name,
            image,
            flags,
        } => {
            let runner = exec::Runner::new(flags.dry_run)?;
            init::run(&state, &runner, &flags.options(nick, name, image))?;
            Ok(())
        }
        Cmd::Mirror {
            nick,
            name,
            conf,
            rev,
            dashboard,
            date,
            run,
            flags,
        } => {
            if nick.is_none() && (flags.week.is_some() || flags.artifact_root.is_some() || flags.skip_setup) {
                bail!("--week, --artifact-root and --skip-setup apply when bringing a worker up: give a <nick>");
            }
            let runner = exec::Runner::new(flags.dry_run)?;
            mirror::run(
                &state,
                &runner,
                &mirror::Options {
                    conf,
                    rev: Some(rev),
                    dashboard,
                    date,
                    run,
                    nick,
                    name,
                    week: flags.week,
                    artifact_root: flags.artifact_root,
                    skip_setup: flags.skip_setup,
                },
            )
        }
        Cmd::Host(cmd) => host_cmd(&state, cmd),
        Cmd::Config(cmd) => config_cmd(&mut state, cmd),
        Cmd::Doctor => unreachable!(),
    }
}

impl InitFlags {
    fn options(self, nick: String, name: String, image: Option<String>) -> init::Options {
        init::Options {
            nick,
            name,
            week: self.week,
            artifact_root: self.artifact_root,
            image,
            skip_setup: self.skip_setup,
        }
    }
}

fn host_cmd(state: &State, cmd: HostCmd) -> Result<()> {
    let mut hosts = state.load_hosts()?;
    match cmd {
        HostCmd::Ls => {
            print!("{}", state::format_hosts(&hosts));
        }
        HostCmd::Add {
            nick,
            home,
            model,
            sock,
            rest,
            force,
        } => {
            let host = Host {
                nick,
                model_path: model,
                docker_sock: sock,
                host_home: home,
                rest,
            };
            host.validate()?;
            match hosts.iter().position(|h| h.nick == host.nick) {
                Some(_) if !force => bail!("host '{}' already exists; use --force to replace it", host.nick),
                Some(i) => hosts[i] = host,
                None => hosts.push(host),
            }
            state.save_hosts(&hosts)?;
        }
        HostCmd::Rm { nick } => {
            let before = hosts.len();
            hosts.retain(|h| h.nick != nick);
            if hosts.len() == before {
                bail!("host '{nick}' not found");
            }
            state.save_hosts(&hosts)?;
        }
    }
    Ok(())
}

fn config_cmd(state: &mut State, cmd: ConfigCmd) -> Result<()> {
    match cmd {
        ConfigCmd::Ls => {
            println!("# AKAO_CONFIG_ROOT={}", state.root.display());
            println!(
                "# work year {} / week {} (ISO, from today)",
                state::work_year(),
                state::work_week()
            );
            let (root, from) = state::artifact_root()?;
            println!("# artifact root {root} ({from})");
            let (repo, from) = state::repo_root()?;
            println!("# repo {repo} ({from})");
            for (key, _, desc) in state::KEYS {
                let v = state.get(key)?;
                let origin = if state.is_explicit(key) { "" } else { "  (default)" };
                println!("# {desc}");
                println!("{key} = {}{origin}", v.as_deref().unwrap_or("<unset>"));
            }
        }
        ConfigCmd::Get { key } => println!("{}", state.require(&key)?),
        ConfigCmd::Set { key, value } => state.set(&key, Some(&value))?,
        ConfigCmd::Unset { key } => state.set(&key, None)?,
    }
    Ok(())
}
