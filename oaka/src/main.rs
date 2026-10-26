mod compile;
mod doctor;
mod draft;
mod plan;
mod profile;
mod stack;
mod suite;
mod sys;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use profile::{flag_words, targets_text, Arg, Library, Profile, ProfileFile, Role, Target, ROUTER_COMMAND};
use std::path::{Path, PathBuf};
use toml::Value;

/// Worker-side plan generator and script compiler.
///
/// A plan (plan.toml in the Work Directory) names servers by profile plus overrides, and
/// the clients to run against them.  `oaka compile` turns it into stand-alone scripts in
/// scripts/; `oaka run` compiles and runs them.  Profiles live in the library,
/// $OAKA_LIB, else the oaka/ dir of the akao-workflow checkout ($AKAO_REPO_ROOT).
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
        /// Package of the library's stacks.toml to install from a tree, as <package>[@<commit>]; repeatable
        #[arg(long = "stack", value_name = "PACKAGE")]
        stacks: Vec<String>,
        /// gsm8k clients with thinking on (reasoning models such as gpt-oss)
        #[arg(long)]
        thinking: bool,
        /// A suite of the library: one plan per layout, in <dir>/<layout>/plan.toml, measuring
        /// --profile on --gpus (see `oaka suite ls`)
        #[arg(long, value_name = "NAME")]
        suite: Option<String>,
        /// With --suite: pin the servers' ports, from this one up (each layout reuses them)
        #[arg(long, requires = "suite")]
        port: Option<u16>,
        /// Overwrite an existing plan.toml
        #[arg(long)]
        force: bool,
    },
    /// Validate plan.toml against the library and this machine
    Check {
        /// Print JSON instead, for agents: {"ok": true, "servers": ...} or {"ok": false, "error": ...}
        #[arg(long)]
        json: bool,
    },
    /// Write scripts/ from plan.toml (automatic choices go to plan.lock.toml)
    Compile,
    /// Compile, then run scripts/run_all.sh
    Run,
    /// This worker: GPU arch, ROCm version and what compiled scripts need (read-only)
    Doctor {
        /// Print JSON instead, for agents: {"ok": ..., "checks": [{"status", "check", "detail"}]}
        #[arg(long)]
        json: bool,
    },
    /// Browse and extend the profile library
    #[command(subcommand)]
    Profile(ProfileCmd),
    /// Browse the library's suites (named benchmark matrices; `oaka draft --suite`)
    #[command(subcommand)]
    Suite(SuiteCmd),
}

#[derive(Subcommand)]
enum SuiteCmd {
    /// List suites
    Ls,
    /// Show a suite with its extends chain applied
    Show { name: String },
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
        /// What it is for: an arch, or arch:ROCm version (gfx950:10.1, *:10.1); repeatable
        /// [default: those of the profile it extends]
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
            stacks,
            thinking,
            suite: Some(suite),
            port,
            force,
        } => {
            if profiles.len() != 1 || !clients.is_empty() {
                bail!("--suite takes one --profile and no --client (the suite says what is measured)");
            }
            draft_suite(&dir, &lib, &suite, &profiles[0], &gpus, &stacks, thinking, port, force)
        }
        Cmd::Draft {
            profiles,
            gpus,
            clients,
            stacks,
            thinking,
            suite: None,
            port: _,
            force,
        } => draft::run(
            &dir,
            &lib,
            &draft::Options {
                profiles,
                gpus,
                clients,
                stacks,
                thinking,
                force,
            },
        ),
        Cmd::Check { json: false } => {
            let c = plan::check(&dir, &lib)?;
            describe(&c);
            println!("ok");
            Ok(())
        }
        Cmd::Check { json: true } => {
            // The verdict is in the JSON, errors included; the exit code still says it.
            let (out, ok) = match plan::check(&dir, &lib) {
                Ok(c) => (checked_json(&c), true),
                Err(e) => (serde_json::json!({ "ok": false, "error": format!("{e:#}") }), false),
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
            if !ok {
                std::process::exit(1);
            }
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
        Cmd::Doctor { json } => doctor::run(&lib, json),
        Cmd::Profile(cmd) => profile_cmd(&dir, &lib, cmd),
        Cmd::Suite(SuiteCmd::Ls) => {
            for name in lib.suites()? {
                let s = lib.suite(&name)?;
                println!("{name:<28} {}", s.description);
            }
            Ok(())
        }
        Cmd::Suite(SuiteCmd::Show { name }) => {
            print!("{}", lib.suite(&name)?.text());
            Ok(())
        }
    }
}

/// `oaka draft --suite`: <dir>/<layout>/plan.toml for each of the suite's layouts.
#[allow(clippy::too_many_arguments)]
fn draft_suite(
    dir: &Path,
    lib: &Library,
    name: &str,
    profile: &str,
    gpus: &[u32],
    stacks: &[String],
    thinking: bool,
    port: Option<u16>,
    force: bool,
) -> Result<()> {
    let s = lib.suite(name)?;
    let machine = plan::Machine::probe();
    let plans = suite::render(
        lib,
        &s,
        &suite::DraftOptions {
            profile,
            gpus,
            stacks,
            thinking,
            port,
            machine: &machine,
        },
    )?;
    for (layout, _) in &plans {
        let path = dir.join(layout).join(plan::PLAN);
        if path.exists() && !force {
            bail!("{} exists; use --force to overwrite it", path.display());
        }
    }
    for (layout, text) in &plans {
        let d = dir.join(layout);
        std::fs::create_dir_all(&d)?;
        std::fs::write(d.join(plan::PLAN), text)?;
        println!("wrote {}", d.join(plan::PLAN).display());
    }
    println!(
        "run each layout, one after another: {}",
        plans
            .iter()
            .map(|(l, _)| format!("oaka -C {} run", dir.join(l).display()))
            .collect::<Vec<_>>()
            .join(" && ")
    );
    Ok(())
}

/// "ROCm 10.0.0", "ROCm unknown", or "CUDA" on NVIDIA.
pub fn runtime_text(arch: Option<&str>, rocm: Option<&str>) -> String {
    if arch.is_some_and(sys::is_nvidia) {
        "CUDA".into()
    } else {
        format!("ROCm {}", rocm.unwrap_or("unknown"))
    }
}

/// "8 x gfx950, ROCm 10.0.0" (or "4 x sm_103, CUDA") for a machine.
fn machine_text(m: &plan::Machine) -> String {
    let gpus = match &m.gpus {
        None => "GPUs unknown".to_string(),
        Some(g) if g.is_empty() => "no GPUs".to_string(),
        Some(g) if g.iter().all(|a| *a == g[0]) => format!("{} x {}", g.len(), g[0]),
        Some(g) => g.join(","),
    };
    if m.nvidia() {
        return format!("{gpus}, CUDA");
    }
    format!("{gpus}, ROCm {}", m.rocm.as_deref().unwrap_or("unknown"))
}

/// What `describe` prints, as JSON for agents.
fn checked_json(c: &plan::Checked) -> serde_json::Value {
    use serde_json::json;
    let vary = match &c.vary {
        plan::Vary::No => serde_json::Value::Null,
        plan::Vary::Commits { package, commits } => {
            json!({ "kind": "commits", "package": package, "commits": commits })
        }
        plan::Vary::Bisect { package, good, bad } => {
            json!({ "kind": "bisect", "package": package, "good": good, "bad": bad })
        }
    };
    json!({
        "ok": true,
        "machine": { "gpus": c.machine.gpus, "rocm": c.machine.rocm },
        "servers": c.servers.iter().map(|s| json!({
            "name": s.name, "profile": s.base.name, "engine": s.effective.engine.as_str(), "model": s.model,
            "gpus": s.gpus, "tp": s.tp,
            "port": s.port, "overrides": compile::overrides(s),
            "pd": s.effective.pd.as_ref().map(|_| s.pd_servers().iter().map(|(name, role, gpus)| json!({
                "name": name, "role": role.as_str(), "gpus": gpus,
            })).collect::<Vec<_>>()),
        })).collect::<Vec<_>>(),
        "clients": c.plan.clients.iter().enumerate().map(|(i, cl)| json!({
            "index": i + 1, "kind": cl.kind(), "server": cl.server(), "lockstep": cl.lockstep(),
        })).collect::<Vec<_>>(),
        "stack": c.stack.iter().map(|p| json!({
            "package": p.name, "tree": p.tree, "commit": p.commit, "clean": p.clean.as_str(),
        })).collect::<Vec<_>>(),
        "vary": vary,
        "warnings": c.warnings,
    })
}

fn describe(c: &plan::Checked) {
    println!("machine: {}", machine_text(&c.machine));
    for s in &c.servers {
        println!(
            "server {}: profile {} ({}), model {}, gpus {:?}, tp {}",
            s.name, s.base.name, s.effective.engine, s.model, s.gpus, s.tp
        );
        for o in compile::overrides(s) {
            println!("  override {o}");
        }
        let pd = s.pd_servers();
        for (name, role, gpus) in &pd {
            println!("  {} server {name}: gpus {gpus:?}, tp {}", role.as_str(), s.tp);
        }
        if !pd.is_empty() {
            println!("  router {}.router: the server's port, for its clients", s.name);
        }
    }
    for (i, cl) in c.plan.clients.iter().enumerate() {
        match cl.lockstep() {
            Some(l) => println!("client {:02}: {} on {}, lockstep {l}", i + 1, cl.kind(), cl.server()),
            None => println!("client {:02}: {} on {}", i + 1, cl.kind(), cl.server()),
        }
    }
    for p in &c.stack {
        let what = match &c.vary {
            plan::Vary::Commits { package, commits } if *package == p.name => {
                format!("each of {}", commits.join(" "))
            }
            plan::Vary::Bisect { package, good, bad } if *package == p.name => format!("bisect good {good} bad {bad}"),
            _ => p.commit.clone().unwrap_or_else(|| "the tree as it is".into()),
        };
        println!("stack {}: {what} in {}, clean {}", p.name, p.tree, p.clean.as_str());
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
            let rocm = sys::rocm().map(|(v, _)| v);
            println!(
                "# {}  (this machine: {}, {}; ! = not for it, x = broken)",
                lib.profiles_dir().display(),
                arch.as_deref().unwrap_or("no single GPU arch"),
                runtime_text(arch.as_deref(), rocm.as_deref())
            );
            for name in lib.list()? {
                let p = match lib.resolve(&name) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("x{name:<48} BROKEN: {e:#}");
                        continue;
                    }
                };
                let archs = targets_text(&p.arch);
                let fits = p.fits(arch.as_deref(), rocm.as_deref());
                println!(
                    "{}{name:<48} {:<7}{archs:<16} {}",
                    if fits { " " } else { "!" },
                    p.engine.as_str(),
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
            let arch = arch
                .iter()
                .map(|a| Target::parse(a).with_context(|| format!("--arch {a}")))
                .collect::<Result<Vec<_>>>()?;
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
    println!("# arch: {}", targets_text(&p.arch));
    for (k, v) in &p.env {
        println!("{k}={}", sys::q(v));
    }
    println!("{} \\", p.engine.command());
    let model = p.model.as_deref().map(sys::q).unwrap_or_else(|| "<plan model>".into());
    let tp = p.tp.map(|t| t.to_string()).unwrap_or_else(|| "<number of gpus>".into());
    let mut lines: Vec<String> = p
        .engine
        .fixed_args(
            &model,
            &tp,
            if p.pd.is_some() {
                "<its own port>"
            } else {
                "<plan port>"
            },
        )
        .iter()
        .map(|w| w.join(" "))
        .collect();
    lines.extend(
        p.launch_args()
            .iter()
            .map(|w| w.iter().map(|x| sys::q(x)).collect::<Vec<_>>().join(" ")),
    );
    for (i, line) in lines.iter().enumerate() {
        println!("    {line}{}", if i + 1 < lines.len() { " \\" } else { "" });
    }
    if let Some(pd) = &p.pd {
        println!(
            "# [pd]: {} prefill + {} decode servers of the command above, {tp} GPUs each, behind a router",
            pd.prefill.count, pd.decode.count
        );
        for role in [Role::Prefill, Role::Decode] {
            let mut words = vec![format!("--disaggregation-mode {}", role.as_str())];
            if role == Role::Prefill {
                words.push("--disaggregation-bootstrap-port <port>".into());
            }
            words.extend(pd.role_delta(role, &p).into_iter().map(|(_, text)| text));
            println!("#   {}: {}", role.as_str(), words.join(" "));
        }
        for (k, v) in &pd.router_env {
            println!("{k}={}", sys::q(v));
        }
        let mut router = vec![
            format!("{ROUTER_COMMAND} --pd-disaggregation"),
            "--prefill <url> <bootstrap port> ... --decode <url> ...".into(),
            "--host 127.0.0.1 --port <plan port>".into(),
        ];
        router.extend(
            flag_words(&pd.router_args)
                .iter()
                .map(|w| w.iter().map(|x| sys::q(x)).collect::<Vec<_>>().join(" ")),
        );
        println!("{}", router.join(" \\\n    "));
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
    // What changed only a PD role's value: saved as the overlay that did it.
    let (env, args) = profile::role_only_changes(base, eff);
    for k in env {
        let v = eff
            .env
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| Value::String(v.clone()));
        f.env.insert(k, v.unwrap_or(Value::Boolean(false)));
    }
    for k in args {
        let v = eff.arg(&k).map(arg_value);
        f.args.insert(k, v.unwrap_or(Value::Boolean(false)));
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
    field("engine", Some(a.engine.to_string()), Some(b.engine.to_string()));
    field("model", a.model.clone(), b.model.clone());
    field("tp", a.tp.map(|t| t.to_string()), b.tp.map(|t| t.to_string()));
    field("arch", Some(targets_text(&a.arch)), Some(targets_text(&b.arch)));
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
    let (pa, pb) = (pd_lines(a), pd_lines(b));
    let mut seen = Vec::new();
    for (k, _) in pa.iter().chain(&pb) {
        if !seen.contains(&k) {
            seen.push(k);
            let line = |p: &[(String, String)]| p.iter().find(|(n, _)| n == k).map(|(_, l)| l.clone());
            field("pd", line(&pa), line(&pb));
        }
    }
    out
}

/// A profile's `[pd]` as (key, line) pairs for `diff`.
fn pd_lines(p: &Profile) -> Vec<(String, String)> {
    let Some(pd) = &p.pd else { return Vec::new() };
    let mut out = vec![(
        "servers".to_string(),
        format!("{} prefill + {} decode", pd.prefill.count, pd.decode.count),
    )];
    for role in [Role::Prefill, Role::Decode] {
        for (key, line) in pd.role_delta(role, p) {
            out.push((format!("{} {key}", role.as_str()), format!("{} {line}", role.as_str())));
        }
    }
    for (k, v) in &pd.router_env {
        out.push((format!("router env {k}"), format!("router env {k}={}", sys::q(v))));
    }
    for (w, (k, _)) in flag_words(&pd.router_args).iter().zip(&pd.router_args) {
        let line = w.iter().map(|x| sys::q(x)).collect::<Vec<_>>().join(" ");
        out.push((format!("router args {k}"), format!("router {line}")));
    }
    out
}
