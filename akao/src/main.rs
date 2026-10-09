mod cp;
mod doctor;
mod exec;
mod init;
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
    /// Files are always overwritten.
    Cp {
        /// Source address: `nick:rel-path` or a local path
        src: String,
        /// Destination address: `nick:rel-path` or a local path
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
        /// Work week to place the workdir under [default: ISO week of today]
        #[arg(long)]
        week: Option<String>,
        /// Skip the control-plane deploy and the package/agent installation
        #[arg(long)]
        skip_setup: bool,
        /// Print the commands that would change anything instead of running them
        #[arg(long)]
        dry_run: bool,
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
        /// Image tag [default: config default_image]
        #[arg(long)]
        image: Option<String>,
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
            week,
            skip_setup,
            dry_run,
        } => {
            let runner = exec::Runner::new(dry_run)?;
            init::run(
                &state,
                &runner,
                &init::Options {
                    nick,
                    name,
                    week,
                    skip_setup,
                },
            )
        }
        Cmd::Host(cmd) => host_cmd(&state, cmd),
        Cmd::Config(cmd) => config_cmd(&mut state, cmd),
        Cmd::Doctor => unreachable!(),
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
            image,
            sock,
            rest,
            force,
        } => {
            let host = Host {
                nick,
                image,
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
