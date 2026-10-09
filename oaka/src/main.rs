mod compile;
mod doctor;
mod draft;
mod plan;
mod profile;
mod sys;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use profile::{Arg, Library, Profile, ProfileFile};
use std::path::{Path, PathBuf};
use toml::Value;

/// Worker-side plan generator and script compiler.
///
/// A plan (plan.toml in the Work Directory) names servers by profile plus overrides, and
/// the clients to run against them.  `oaka compile` turns it into stand-alone scripts in
/// scripts/; `oaka run` compiles and runs them.  Profiles live in the library,
/// $OAKA_LIB or /<year>/oaka.
#[derive(Parser)]
#[command(version = sys::VERSION)]
struct Cli {
    /// Work Directory holding plan.toml [default: current directory]
    #[arg(short = 'C', global = true, value_name = "DIR")]
    dir: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a plan.toml to start from, pre-filled from the hints and the machine
    Draft {
        /// Server profile (<model>/<recipe>); repeat for several servers
        #[arg(long = "profile", value_name = "NAME")]
        profiles: Vec<String>,
        /// GPUs the servers may use, e.g. 6,7
        #[arg(long, value_delimiter = ',')]
        gpus: Vec<u32>,
        /// Client to add for every server: gsm8k or fixed-seq; repeatable
        #[arg(long = "client", value_name = "KIND")]
        clients: Vec<String>,
        /// Overwrite an existing plan.toml
        #[arg(long)]
        force: bool,
    },
    /// Validate plan.toml against the library and this machine
    Check,
    /// Write scripts/ from plan.toml (automatic choices go to plan.lock.toml)
    Compile,
    /// Compile, then run scripts/run_all.sh
    Run,
    /// Check what compiled scripts need in this worker (read-only)
    Doctor,
    /// Browse and extend the profile library
    #[command(subcommand)]
    Profile(ProfileCmd),
}

#[derive(Subcommand)]
enum ProfileCmd {
    /// List profiles
    Ls,
    /// Show a profile with its extends chain applied
    Show { name: String },
    /// Save a plan server's overrides as a new profile extending its current one
    Save {
        /// Server name in plan.toml
        server: String,
        /// New profile name, <model>/<recipe>
        #[arg(long = "as", value_name = "NAME")]
        name: String,
        /// GPU architectures it is for [default: those of the profile it extends]
        #[arg(long)]
        arch: Vec<String>,
        /// One line on what this recipe is
        #[arg(long)]
        description: Option<String>,
        /// Replace an existing profile
        #[arg(long)]
        force: bool,
    },
    /// Show how two profiles differ
    Diff { a: String, b: String },
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("oaka: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    let dir = match cli.dir {
        Some(d) => d,
        None => std::env::current_dir()?,
    };
    let dir = dir
        .canonicalize()
        .with_context(|| format!("work directory {}", dir.display()))?;
    let lib = Library::locate();
    match cli.cmd {
        Cmd::Draft {
            profiles,
            gpus,
            clients,
            force,
        } => draft::run(
            &dir,
            &lib,
            &draft::Options {
                profiles,
                gpus,
                clients,
                force,
            },
        ),
        Cmd::Check => {
            let c = plan::check(&dir, &lib)?;
            describe(&c);
            println!("ok");
            Ok(())
        }
        Cmd::Compile => {
            compile_and_report(&dir, &lib)?;
            Ok(())
        }
        Cmd::Run => {
            compile_and_report(&dir, &lib)?;
            let script = dir.join(compile::SCRIPTS).join("run_all.sh");
            println!("+ bash {}", script.display());
            // Become run_all.sh, so Ctrl-C and exit codes behave as if it was run by hand.
            use std::os::unix::process::CommandExt;
            let err = std::process::Command::new("bash").arg(&script).current_dir(&dir).exec();
            Err(err).context("cannot exec bash")
        }
        Cmd::Doctor => doctor::run(&lib),
        Cmd::Profile(cmd) => profile_cmd(&dir, &lib, cmd),
    }
}

fn describe(c: &plan::Checked) {
    for s in &c.servers {
        println!(
            "server {}: profile {}, model {}, gpus {:?}, tp {}",
            s.name, s.base.name, s.model, s.gpus, s.tp
        );
        for o in compile::overrides(s) {
            println!("  override {o}");
        }
    }
    for (i, cl) in c.plan.clients.iter().enumerate() {
        println!("client {:02}: {} on {}", i + 1, cl.kind(), cl.server());
    }
    for w in &c.warnings {
        println!("warning: {w}");
    }
}

fn compile_and_report(dir: &Path, lib: &Library) -> Result<()> {
    let c = plan::check(dir, lib)?;
    describe(&c);
    let out = compile::compile(&c)?;
    for n in &out.notes {
        println!("note: {n}");
    }
    for p in &out.removed {
        println!("removed {}", p.display());
    }
    for p in &out.written {
        println!("wrote {}", p.display());
    }
    Ok(())
}

fn profile_cmd(dir: &Path, lib: &Library, cmd: ProfileCmd) -> Result<()> {
    match cmd {
        ProfileCmd::Ls => {
            let arch = sys::arch();
            println!(
                "# {}  (this machine: {}; ! = other arch, x = broken)",
                lib.profiles_dir().display(),
                arch.as_deref().unwrap_or("no single GPU arch")
            );
            for name in lib.list()? {
                let p = match lib.resolve(&name) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("x{name:<48} BROKEN: {e:#}");
                        continue;
                    }
                };
                let archs = if p.arch.is_empty() {
                    "any".to_string()
                } else {
                    p.arch.join(",")
                };
                let fits = arch.as_ref().is_none_or(|a| p.arch.is_empty() || p.arch.contains(a));
                println!(
                    "{}{name:<48} {archs:<16} {}",
                    if fits { " " } else { "!" },
                    p.description.as_deref().unwrap_or("")
                );
            }
        }
        ProfileCmd::Show { name } => show(lib, &name)?,
        ProfileCmd::Save {
            server,
            name,
            arch,
            description,
            force,
        } => {
            let c = plan::check(dir, lib)?;
            let s = c
                .servers
                .iter()
                .find(|s| s.name == server)
                .with_context(|| format!("no server {server:?} in the plan"))?;
            let mut file = delta(&s.base, &s.effective);
            if s.base.model.as_deref() != Some(s.model.as_str()) {
                file.model = Some(s.model.clone());
            }
            if file.env.is_empty() && file.args.is_empty() && file.model.is_none() {
                bail!("server {server} has no overrides over {}; nothing to save", s.base.name);
            }
            for a in &arch {
                if !sys::ARCHES.contains(&a.as_str()) {
                    bail!("arch {a:?} is not one of {}", sys::ARCHES.join(" "));
                }
            }
            file.arch = if arch.is_empty() { s.base.arch.clone() } else { arch };
            file.description =
                Some(description.unwrap_or_else(|| format!("{} with the overrides of server {server}", s.base.name)));
            file.extends = Some(s.base.name.clone());
            // History as prose, not a path field: the file must stand on its own on boxes
            // where the task directory does not exist.
            let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
            let header = format!(
                "# {}\n#\n# History: saved {} by `oaka profile save` from server '{server}' of the task\n\
                 # {} on {}.  Record below why each override is here and what it measured,\n\
                 # so this file stands on its own.\n\n",
                file.description.as_deref().unwrap_or_default(),
                chrono::Local::now().format("%Y-%m-%d"),
                dir.display(),
                host.trim(),
            );
            let path = lib.save(&name, &header, &file, force)?;
            println!("wrote {}", path.display());
            print!("{}", std::fs::read_to_string(&path)?);
        }
        ProfileCmd::Diff { a, b } => {
            let (pa, pb) = (lib.resolve(&a)?, lib.resolve(&b)?);
            println!("--- {a}\n+++ {b}");
            for line in diff(&pa, &pb) {
                println!("{line}");
            }
        }
    }
    Ok(())
}

fn show(lib: &Library, name: &str) -> Result<()> {
    let p = lib.resolve(name)?;
    let mut chain = vec![name.to_string()];
    while let Some(parent) = lib.load_file(chain.last().unwrap())?.extends {
        chain.push(parent);
    }
    println!("# {name}  ({})", lib.profile_path(name).display());
    if let Some(d) = &p.description {
        println!("# {d}");
    }
    if chain.len() > 1 {
        println!("# chain: {}", chain.join(" <- "));
    }
    println!(
        "# arch: {}",
        if p.arch.is_empty() {
            "any".into()
        } else {
            p.arch.join(" ")
        }
    );
    for (k, v) in &p.env {
        println!("{k}={}", sys::q(v));
    }
    println!("python3 -m sglang.launch_server \\");
    println!("    --model-path {} \\", p.model.as_deref().unwrap_or("<plan model>"));
    println!(
        "    --tp {} \\",
        p.tp.map(|t| t.to_string()).unwrap_or_else(|| "<number of gpus>".into())
    );
    println!("    --port <plan port> \\");
    let lines = p.launch_args();
    for (i, w) in lines.iter().enumerate() {
        let w: Vec<String> = w.iter().map(|x| sys::q(x)).collect();
        println!("    {}{}", w.join(" "), if i + 1 < lines.len() { " \\" } else { "" });
    }
    Ok(())
}

fn arg_value(v: &Arg) -> Value {
    let scalar = |s: &str| {
        s.parse::<i64>()
            .map(Value::Integer)
            .unwrap_or_else(|_| Value::String(s.to_string()))
    };
    match v {
        Arg::Flag => Value::Boolean(true),
        Arg::Value(s) => scalar(s),
        Arg::List(l) => Value::Array(l.iter().map(|s| scalar(s)).collect()),
    }
}

/// The overlay that turns `base` into `eff`.
fn delta(base: &Profile, eff: &Profile) -> ProfileFile {
    let mut f = ProfileFile::default();
    for (k, v) in &eff.env {
        if !base.env.iter().any(|(bk, bv)| bk == k && bv == v) {
            f.env.insert(k.clone(), Value::String(v.clone()));
        }
    }
    for (k, _) in &base.env {
        if !eff.env.iter().any(|(ek, _)| ek == k) {
            f.env.insert(k.clone(), Value::Boolean(false));
        }
    }
    for (k, v) in &eff.args {
        if base.arg(k) != Some(v) {
            f.args.insert(k.clone(), arg_value(v));
        }
    }
    for (k, _) in &base.args {
        if eff.arg(k).is_none() {
            f.args.insert(k.clone(), Value::Boolean(false));
        }
    }
    f
}

fn diff(a: &Profile, b: &Profile) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = |what: &str, x: Option<String>, y: Option<String>| {
        if x != y {
            if let Some(x) = x {
                out.push(format!("- {what} {x}"));
            }
            if let Some(y) = y {
                out.push(format!("+ {what} {y}"));
            }
        }
    };
    field("model", a.model.clone(), b.model.clone());
    field("tp", a.tp.map(|t| t.to_string()), b.tp.map(|t| t.to_string()));
    field("arch", Some(a.arch.join(" ")), Some(b.arch.join(" ")));
    let env = |p: &Profile, k: &str| {
        p.env
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| format!("{k}={}", sys::q(v)))
    };
    let mut seen = Vec::new();
    for (k, _) in a.env.iter().chain(&b.env) {
        if !seen.contains(&k) {
            seen.push(k);
            field("env", env(a, k), env(b, k));
        }
    }
    let arg = |p: &Profile, k: &str| {
        p.arg(k).map(|v| match v {
            Arg::Flag => format!("--{k}"),
            Arg::Value(s) => format!("--{k} {}", sys::q(s)),
            Arg::List(l) => format!("--{k} {}", l.iter().map(|s| sys::q(s)).collect::<Vec<_>>().join(" ")),
        })
    };
    let mut seen = Vec::new();
    for (k, _) in a.args.iter().chain(&b.args) {
        if !seen.contains(&k) {
            seen.push(k);
            field("args", arg(a, k), arg(b, k));
        }
    }
    out
}
