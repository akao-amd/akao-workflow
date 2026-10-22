//! Server profiles: `<lib>/profiles/<model>/<recipe>.toml`.
//!
//! A profile is a launch recipe for a serving engine (`engine`: SGLang unless it says
//! vllm or ATOM): the model, env vars and flags.  `extends` makes a profile an overlay of another, which is how experiments
//! are consolidated (`oaka profile save` writes only the delta).
//!
//! A profile is self-contained: it travels to every box with the library, where nothing
//! else from the console exists, so `extends` (inside the library) is its only reference.
//! Provenance and evidence are written into the file as comments, never as paths.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use toml::{Table, Value};

/// Flags and env vars that oaka sets itself from the plan: the model, tp and port flags of
/// every engine, whichever engine the profile names.
pub const RESERVED_ARGS: &[&str] = &[
    "model-path",
    "model",
    "tp",
    "tp-size",
    "tensor-parallel-size",
    "port",
    "server-port",
];
pub const RESERVED_ENV: &[&str] = &["HIP_VISIBLE_DEVICES"];
/// What oaka sets on the SGLang servers of a `[pd]` profile: each one's role and the prefill
/// servers' bootstrap port, which the router is told.
pub const PD_RESERVED_ARGS: &[&str] = &["disaggregation-mode", "disaggregation-bootstrap-port"];
/// What oaka sets on a `[pd]` profile's router: its servers, and where it listens.
pub const ROUTER_RESERVED_ARGS: &[&str] = &["pd-disaggregation", "prefill", "decode", "host", "port"];
/// The router in front of a `[pd]` profile's servers.
pub const ROUTER_COMMAND: &str = "python3 -m sglang_router.launch_router";
pub const ROUTER_MODULE: &str = "sglang_router";

/// The server a profile launches.  Each takes oaka's model, tp and port in its own
/// spelling, and says differently when it is ready.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    #[default]
    Sglang,
    Vllm,
    Atom,
}

impl Engine {
    pub const ALL: [Engine; 3] = [Engine::Sglang, Engine::Vllm, Engine::Atom];

    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Sglang => "sglang",
            Engine::Vllm => "vllm",
            Engine::Atom => "atom",
        }
    }

    /// The command that starts a server, before its flags.
    pub fn command(self) -> &'static str {
        match self {
            Engine::Sglang => "python3 -m sglang.launch_server",
            Engine::Vllm => "vllm serve",
            Engine::Atom => "python3 -m atom.entrypoints.openai_server",
        }
    }

    /// The Python package that has to be importable for `command`.
    pub fn module(self) -> &'static str {
        match self {
            Engine::Sglang => "sglang",
            Engine::Vllm => "vllm",
            Engine::Atom => "atom",
        }
    }

    /// oaka's own flags, one argument group per line: the model, tp and port (`port` is
    /// written as given, e.g. "\"$PORT\"").  ATOM's spelling is InferenceX's ATOM scripts'.
    pub fn fixed_args(self, model: &str, tp: &str, port: &str) -> Vec<Vec<String>> {
        let s = |v: &[&str]| v.iter().map(|w| w.to_string()).collect::<Vec<_>>();
        match self {
            Engine::Sglang => vec![s(&["--model-path", model]), s(&["--tp", tp]), s(&["--port", port])],
            Engine::Vllm => vec![s(&[model]), s(&["--tensor-parallel-size", tp]), s(&["--port", port])],
            Engine::Atom => vec![s(&["--model", model]), s(&["-tp", tp]), s(&["--server-port", port])],
        }
    }

    /// The variable naming the engine's own cache root, if it has one.  SGLang's holds the
    /// Triton kernels it compiles (it points TRITON_CACHE_DIR there unless that is set).
    pub fn cache_env(self) -> Option<&'static str> {
        match self {
            Engine::Sglang => Some("SGLANG_CACHE_DIR"),
            Engine::Vllm => Some("VLLM_CACHE_ROOT"),
            Engine::Atom => None,
        }
    }

    /// SGLang says it is ready in its log once its warmup request went through; the others
    /// are ready when `GET /health` answers 200, as InferenceX's own scripts wait.
    pub fn ready_by_log(self) -> bool {
        self == Engine::Sglang
    }
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One profile file, as written on disk.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// What it is for, multiple choice: "gfx950" or ["gfx950", "10.0"]; empty = any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arch: Vec<Target>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,
    /// The server it launches; inherited through `extends`, SGLang when no profile says.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tp: Option<u32>,
    #[serde(default)]
    pub env: Table,
    #[serde(default)]
    pub args: Table,
    /// Prefill-decode disaggregation: the plan server becomes SGLang prefill and decode
    /// servers behind a router; inherited through `extends`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pd: Option<PdFile>,
}

/// `[pd]` as written: `[pd.prefill]` and `[pd.decode]` (each a `count`, default 1, and
/// `env`/`args` overlaid on the profile's for that role only), `[pd.router]` (`env`/`args`
/// of `sglang_router.launch_router`).
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PdFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill: Option<RoleFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<RoleFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterFile>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoleFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(default)]
    pub env: Table,
    #[serde(default)]
    pub args: Table,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouterFile {
    #[serde(default)]
    pub env: Table,
    #[serde(default)]
    pub args: Table,
}

/// A server's env vars and flags, in order.
pub type Env = Vec<(String, String)>;
pub type Args = Vec<(String, Arg)>;

/// The two kinds of SGLang server of a PD profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    Prefill,
    Decode,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Prefill => "prefill",
            Role::Decode => "decode",
        }
    }
}

/// One role's servers: how many, and their env and args (the profile's, with the role's
/// overlays applied where they stand in the `extends` chain).
#[derive(Debug, Clone, PartialEq)]
pub struct RoleSpec {
    pub count: u32,
    pub env: Env,
    pub args: Args,
}

/// `[pd]` with its `extends` chain applied.  The roles are resolved alongside the profile:
/// every `[env]`/`[args]` overlay (a profile's, then the plan's) reaches both roles, and a
/// role's own overlay goes on top of its profile's, so a later child or the plan still wins.
#[derive(Debug, Clone, PartialEq)]
pub struct Pd {
    pub prefill: RoleSpec,
    pub decode: RoleSpec,
    pub router_env: Env,
    pub router_args: Args,
}

impl Pd {
    /// Both roles start from the profile's env and args as resolved so far.
    fn new(env: &Env, args: &Args) -> Pd {
        let role = || RoleSpec {
            count: 1,
            env: env.clone(),
            args: args.clone(),
        };
        Pd {
            prefill: role(),
            decode: role(),
            router_env: Vec::new(),
            router_args: Vec::new(),
        }
    }

    pub fn role(&self, role: Role) -> &RoleSpec {
        match role {
            Role::Prefill => &self.prefill,
            Role::Decode => &self.decode,
        }
    }

    /// Prefill servers first, then decode: (role, index within the role).
    pub fn instances(&self) -> Vec<(Role, u32)> {
        let each = |r: Role| (0..self.role(r).count).map(move |i| (r, i));
        each(Role::Prefill).chain(each(Role::Decode)).collect()
    }

    /// How a role's servers differ from the profile's (`common`): (key, text) pairs, the
    /// text `env K=V`, `env K unset`, `--flag v` or `--flag dropped`.
    pub fn role_delta(&self, role: Role, common: &Profile) -> Vec<(String, String)> {
        let r = self.role(role);
        let mut out = Vec::new();
        for (k, v) in &r.env {
            if !common.env.iter().any(|(ck, cv)| ck == k && cv == v) {
                out.push((format!("env {k}"), format!("env {k}={}", crate::sys::q(v))));
            }
        }
        for (k, _) in &common.env {
            if !r.env.iter().any(|(rk, _)| rk == k) {
                out.push((format!("env {k}"), format!("env {k} unset")));
            }
        }
        for (words, (k, v)) in flag_words(&r.args).iter().zip(&r.args) {
            if common.arg(k) != Some(v) {
                let text = words.iter().map(|w| crate::sys::q(w)).collect::<Vec<_>>().join(" ");
                out.push((format!("args {k}"), text));
            }
        }
        for (k, _) in &common.args {
            if !r.args.iter().any(|(rk, _)| rk == k) {
                out.push((format!("args {k}"), format!("--{k} dropped")));
            }
        }
        out
    }

    /// An `[env]`/`[args]` overlay of a profile or a plan, on both roles.
    fn overlay_roles(&mut self, env: &Table, args: &Table) -> Result<()> {
        for r in [&mut self.prefill, &mut self.decode] {
            overlay_env(&mut r.env, env)?;
            overlay_args(&mut r.args, args)?;
        }
        Ok(())
    }

    /// A profile's `[pd]` tables.
    fn overlay(&mut self, f: &PdFile) -> Result<()> {
        for (role, file) in [(Role::Prefill, &f.prefill), (Role::Decode, &f.decode)] {
            let Some(file) = file else { continue };
            let at = format!("[pd.{}]", role.as_str());
            let spec = match role {
                Role::Prefill => &mut self.prefill,
                Role::Decode => &mut self.decode,
            };
            if let Some(n) = file.count {
                if n == 0 {
                    bail!("{at}: count must be at least 1");
                }
                spec.count = n;
            }
            overlay_env(&mut spec.env, &file.env).with_context(|| at.clone())?;
            overlay_args(&mut spec.args, &file.args).with_context(|| at.clone())?;
        }
        if let Some(r) = &f.router {
            overlay_env(&mut self.router_env, &r.env).context("[pd.router]")?;
            overlay_flags(
                &mut self.router_args,
                &r.args,
                ROUTER_RESERVED_ARGS,
                "for the router from [pd]",
            )
            .context("[pd.router]")?;
        }
        Ok(())
    }
}

/// What a profile is for: a GPU arch and a ROCm version, either `*` for any.  Written as
/// "gfx950" (any ROCm, as before ROCm versions mattered) or ["gfx950", "10.0"]; the
/// version matches as a prefix (sys::rocm_matches): "10.0" is any 10.0.x.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Target {
    Arch(String),
    Tuple([String; 2]),
}

impl Target {
    pub fn arch(&self) -> &str {
        match self {
            Target::Arch(a) | Target::Tuple([a, _]) => a,
        }
    }

    pub fn rocm(&self) -> &str {
        match self {
            Target::Arch(_) => "*",
            Target::Tuple([_, r]) => r,
        }
    }

    pub fn fits_arch(&self, arch: &str) -> bool {
        self.arch() == "*" || self.arch() == arch
    }

    /// None when the ROCm version matters to this target but is unknown.
    pub fn fits_rocm(&self, rocm: Option<&str>) -> Option<bool> {
        match (self.rocm(), rocm) {
            ("*", _) => Some(true),
            (_, None) => None,
            (p, Some(v)) => Some(crate::sys::rocm_matches(p, v)),
        }
    }

    /// From the command line: "gfx950" or "gfx950:10.0".
    pub fn parse(s: &str) -> Result<Target> {
        let t = match s.split_once(':') {
            None => Target::Arch(s.to_string()),
            Some((a, r)) => Target::Tuple([a.to_string(), r.to_string()]),
        };
        t.validate()?;
        Ok(t)
    }

    pub fn validate(&self) -> Result<()> {
        let a = self.arch();
        if a != "*" && !crate::sys::ARCHES.contains(&a) {
            bail!("{a:?} is not one of {} or *", crate::sys::ARCHES.join(" "));
        }
        if !crate::sys::valid_rocm_pattern(self.rocm()) {
            bail!(
                "ROCm version {:?} must be dotted numbers or *, e.g. \"10.0\" (any 10.0.x)",
                self.rocm()
            );
        }
        Ok(())
    }
}

impl std::fmt::Display for Target {
    /// "gfx950" for any ROCm, else "gfx950:10.0" (the command-line form).
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.rocm() {
            "*" => write!(f, "{}", self.arch()),
            r => write!(f, "{}:{r}", self.arch()),
        }
    }
}

/// "gfx942 gfx950:10.0" for messages; "any" when empty.
pub fn targets_text(arch: &[Target]) -> String {
    if arch.is_empty() {
        return "any".into();
    }
    arch.iter().map(Target::to_string).collect::<Vec<_>>().join(" ")
}

/// A server flag value.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// `--flag`
    Flag,
    /// `--flag value`
    Value(String),
    /// `--flag v1 v2 ...`
    List(Vec<String>),
}

/// A profile with its `extends` chain applied.
#[derive(Debug, Clone)]
pub struct Profile {
    pub name: String,
    pub description: Option<String>,
    pub arch: Vec<Target>,
    pub engine: Engine,
    pub model: Option<String>,
    pub tp: Option<u32>,
    pub env: Vec<(String, String)>,
    pub args: Vec<(String, Arg)>,
    pub pd: Option<Pd>,
}

impl Profile {
    /// How many servers a plan server of this profile starts besides a router: 1, or a PD
    /// profile's prefill and decode servers.  Each takes `tp` GPUs.
    pub fn instances(&self) -> u32 {
        self.pd.as_ref().map_or(1, |pd| pd.prefill.count + pd.decode.count)
    }

    /// `--flag value ...` for everything in `args`, in order.
    pub fn launch_args(&self) -> Vec<Vec<String>> {
        flag_words(&self.args)
    }

    /// Whether some target fits this arch and ROCm version (None = unknown, not held
    /// against it): for listing; `oaka check` is strict.
    pub fn fits(&self, arch: Option<&str>, rocm: Option<&str>) -> bool {
        self.arch.is_empty()
            || self
                .arch
                .iter()
                .any(|t| arch.is_none_or(|a| t.fits_arch(a)) && t.fits_rocm(rocm) != Some(false))
    }

    pub fn arg(&self, key: &str) -> Option<&Arg> {
        self.args.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Apply an `[env]` / `[args]` overlay (from a child profile or a plan).
    pub fn overlay(&mut self, env: &Table, args: &Table) -> Result<()> {
        overlay_env(&mut self.env, env)?;
        overlay_args(&mut self.args, args)?;
        if let Some(pd) = &mut self.pd {
            pd.overlay_roles(env, args)?;
        }
        Ok(())
    }
}

/// `--flag value ...` for each of `args`, in order: one argument group each.
pub fn flag_words(args: &[(String, Arg)]) -> Vec<Vec<String>> {
    args.iter()
        .map(|(k, v)| {
            let mut words = vec![format!("--{k}")];
            match v {
                Arg::Flag => {}
                Arg::Value(s) => words.push(s.clone()),
                Arg::List(l) => words.extend(l.iter().cloned()),
            }
            words
        })
        .collect()
}

fn scalar(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Integer(i) => Some(i.to_string()),
        Value::Float(f) => Some(f.to_string()),
        _ => None,
    }
}

fn set<T>(list: &mut Vec<(String, T)>, key: &str, value: T) {
    match list.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => list.push((key.to_string(), value)),
    }
}

pub fn overlay_env(env: &mut Vec<(String, String)>, t: &Table) -> Result<()> {
    for (k, v) in t {
        if RESERVED_ENV.contains(&k.as_str()) {
            bail!("env {k} is set by oaka from the plan's gpus; remove it");
        }
        if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            bail!("env name {k:?} is not a valid variable name");
        }
        match v {
            Value::Boolean(false) => env.retain(|(name, _)| name != k),
            _ => match scalar(v) {
                Some(s) => set(env, k, s),
                None => bail!("env {k}: expected a string, a number, or false (to unset), got {v}"),
            },
        }
    }
    Ok(())
}

pub fn overlay_args(args: &mut Vec<(String, Arg)>, t: &Table) -> Result<()> {
    if let Some((k, r)) = t.keys().find_map(|k| {
        PD_RESERVED_ARGS
            .iter()
            .find(|r| r.starts_with(k.as_str()))
            .map(|r| (k, r))
    }) {
        bail!("args {k}: --{r} is set by oaka from the profile's [pd]; remove it");
    }
    overlay_flags(args, t, RESERVED_ARGS, "from the plan")
}

/// Env vars and flags an overlay changed that show only in a PD role of `eff` (vs `base`): its
/// role values differ while the profile's own are the same, e.g. a plan setting a flag back to
/// the profile's value over a role's own.  An overlay reaches every role alike, so `eff`'s own
/// value (or its absence) is what the overlay set.
pub fn role_only_changes(base: &Profile, eff: &Profile) -> (Vec<String>, Vec<String>) {
    let (mut env, mut args) = (Vec::new(), Vec::new());
    let (Some(bp), Some(ep)) = (&base.pd, &eff.pd) else {
        return (env, args);
    };
    let get_env = |e: &Env, k: &str| e.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    let get_arg = |a: &Args, k: &str| a.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    for role in [Role::Prefill, Role::Decode] {
        let (b, e) = (bp.role(role), ep.role(role));
        for k in b.env.iter().chain(&e.env).map(|(k, _)| k) {
            if get_env(&b.env, k) != get_env(&e.env, k)
                && get_env(&base.env, k) == get_env(&eff.env, k)
                && !env.contains(k)
            {
                env.push(k.clone());
            }
        }
        for k in b.args.iter().chain(&e.args).map(|(k, _)| k) {
            if get_arg(&b.args, k) != get_arg(&e.args, k) && base.arg(k) == eff.arg(k) && !args.contains(k) {
                args.push(k.clone());
            }
        }
    }
    (env, args)
}

/// A flag as written in `[args]`: `[A-Za-z0-9][A-Za-z0-9_.-]*`.
fn valid_flag(k: &str) -> bool {
    let mut c = k.chars();
    matches!(c.next(), Some(f) if f.is_ascii_alphanumeric())
        && c.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Overlay `t` onto `args`; the `reserved` keys are oaka's, set `why`.
fn overlay_flags(args: &mut Vec<(String, Arg)>, t: &Table, reserved: &[&str], why: &str) -> Result<()> {
    for (k, v) in t {
        if k.starts_with('-') {
            bail!("args key {k:?}: write flags without the leading dashes");
        }
        if !valid_flag(k) {
            bail!("args key {k:?}: a flag name is letters, digits, '-', '_' and '.' (no '=': the value goes on the right)");
        }
        if reserved.contains(&k.as_str()) {
            bail!("args {k}: --{k} is set by oaka {why}; remove it");
        }
        // argparse takes an unambiguous prefix for the whole option.
        if let Some(r) = reserved.iter().find(|r| r.starts_with(k.as_str())) {
            bail!("args {k}: --{k} abbreviates --{r}, which oaka sets {why}; remove it");
        }
        let arg = match v {
            Value::Boolean(false) => {
                args.retain(|(name, _)| name != k);
                continue;
            }
            Value::Boolean(true) => Arg::Flag,
            Value::Array(items) => Arg::List(
                items
                    .iter()
                    .map(|i| scalar(i).with_context(|| format!("args {k}: list items must be strings or numbers")))
                    .collect::<Result<_>>()?,
            ),
            _ => Arg::Value(scalar(v).with_context(|| format!("args {k}: unsupported value {v}"))?),
        };
        // argparse reads such a word as the next flag, not as this one's value.
        let words = match &arg {
            Arg::Flag => vec![],
            Arg::Value(w) => vec![w],
            Arg::List(l) => l.iter().collect(),
        };
        if let Some(w) = words.iter().find(|w| w.starts_with("--")) {
            bail!("args {k}: value {w:?} would be read as a flag; give each flag a key of its own");
        }
        set(args, k, arg);
    }
    Ok(())
}

/// `<model>/<recipe>`, or `<model>` meaning `<model>/base`.
pub fn canonical(name: &str) -> Result<String> {
    let full = if name.contains('/') {
        name.to_string()
    } else {
        format!("{name}/base")
    };
    if !valid_name(&full) {
        bail!("invalid profile name {name:?}; expected <model>/<recipe> or <model>");
    }
    Ok(full)
}

/// `<model>/<recipe>`, each part [A-Za-z0-9][A-Za-z0-9_.-]*
pub fn valid_name(name: &str) -> bool {
    let part = |s: &str| {
        let mut c = s.chars();
        matches!(c.next(), Some(f) if f.is_ascii_alphanumeric())
            && c.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    };
    matches!(name.split('/').collect::<Vec<_>>()[..], [m, r] if part(m) && part(r))
}

/// The oaka library: `$OAKA_LIB`; else the `oaka/` dir of the akao-workflow checkout
/// (`$AKAO_REPO_ROOT`, default /root/akao-workflow), where profiles and stacks.toml are
/// tracked and a saved profile becomes a commit; else (a container without its clone yet)
/// the old `/<year>/oaka`.
pub struct Library {
    pub root: PathBuf,
}

impl Library {
    pub fn locate() -> Library {
        if let Some(r) = std::env::var_os("OAKA_LIB").filter(|r| !r.is_empty()) {
            return Library { root: PathBuf::from(r) };
        }
        let repo = std::env::var_os("AKAO_REPO_ROOT")
            .filter(|r| !r.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/root/akao-workflow"));
        let root = repo.join("oaka");
        if root.join("profiles").is_dir() {
            return Library { root };
        }
        Library {
            root: PathBuf::from(format!("/{}/oaka", crate::sys::work_year())),
        }
    }

    pub fn profiles_dir(&self) -> PathBuf {
        self.root.join("profiles")
    }

    pub fn profile_path(&self, name: &str) -> PathBuf {
        self.profiles_dir().join(format!("{name}.toml"))
    }

    /// All profile names, sorted.
    pub fn list(&self) -> Result<Vec<String>> {
        let dir = self.profiles_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut names = Vec::new();
        for model in fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .flatten()
        {
            if !model.path().is_dir() {
                continue;
            }
            for f in fs::read_dir(model.path())?.flatten() {
                let p = f.path();
                if p.extension().is_some_and(|e| e == "toml") {
                    let recipe = p.file_stem().unwrap().to_string_lossy();
                    names.push(format!("{}/{recipe}", model.file_name().to_string_lossy()));
                }
            }
        }
        names.sort();
        Ok(names)
    }

    pub fn load_file(&self, name: &str) -> Result<ProfileFile> {
        let name = &canonical(name)?;
        let path = self.profile_path(name);
        if !path.exists() {
            bail!("profile {name} not found ({}); see `oaka profile ls`", path.display());
        }
        read_profile(&path)
    }

    /// Load `name` and apply its `extends` chain.
    pub fn resolve(&self, name: &str) -> Result<Profile> {
        let name = &canonical(name)?;
        let mut chain: Vec<(String, ProfileFile)> = Vec::new();
        let mut next = Some(name.to_string());
        while let Some(n) = next {
            let n = canonical(&n)?;
            if chain.iter().any(|(c, _)| *c == n) {
                bail!("profile {name}: extends cycle through {n}");
            }
            let f = self
                .load_file(&n)
                .with_context(|| format!("resolving profile {name}"))?;
            next = f.extends.clone();
            chain.push((n, f));
        }
        let (_, top) = &chain[0];
        let mut p = Profile {
            name: name.to_string(),
            description: top.description.clone(),
            arch: top.arch.clone(),
            engine: Engine::default(),
            model: None,
            tp: None,
            env: Vec::new(),
            args: Vec::new(),
            pd: None,
        };
        for (n, f) in chain.iter().rev() {
            p.engine = f.engine.unwrap_or(p.engine);
            p.model = f.model.clone().or(p.model);
            p.tp = f.tp.or(p.tp);
            p.overlay(&f.env, &f.args).with_context(|| format!("profile {n}"))?;
            if let Some(pd) = &f.pd {
                let (env, args) = (&p.env, &p.args);
                p.pd.get_or_insert_with(|| Pd::new(env, args))
                    .overlay(pd)
                    .with_context(|| format!("profile {n}"))?;
            }
        }
        if p.pd.is_some() && p.engine != Engine::Sglang {
            bail!(
                "profile {name}: [pd] runs SGLang prefill and decode servers, but its engine is {}",
                p.engine
            );
        }
        Ok(p)
    }

    /// Write `file` as profile `name`, with `header` (comment lines) on top.
    pub fn save(&self, name: &str, header: &str, file: &ProfileFile, force: bool) -> Result<PathBuf> {
        let name = &canonical(name)?;
        let path = self.profile_path(name);
        if path.exists() && !force {
            bail!(
                "profile {name} already exists ({}); use --force to replace it",
                path.display()
            );
        }
        fs::create_dir_all(path.parent().unwrap())?;
        let text = format!("{header}{}", toml::to_string(file)?);
        fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

pub fn read_profile(path: &Path) -> Result<ProfileFile> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let raw: Table = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if raw.contains_key("origin") {
        bail!(
            "{}: `origin` is no longer a field; a profile must not point outside the library, \
             so write its provenance as # comments instead",
            path.display()
        );
    }
    let f: ProfileFile = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    for t in &f.arch {
        t.validate().with_context(|| format!("{}: arch", path.display()))?;
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(s: &str) -> Table {
        toml::from_str(s).unwrap()
    }

    #[test]
    fn args_overlay_and_render() {
        let mut args = Vec::new();
        overlay_args(
            &mut args,
            &table("trust-remote-code = true\npage-size = 1\nmem-fraction-static = 0.9\ncuda-graph-bs = [1, 2]"),
        )
        .unwrap();
        overlay_args(&mut args, &table("page-size = 64\ntrust-remote-code = false")).unwrap();
        let p = Profile {
            name: "m/r".into(),
            description: None,
            arch: vec![],
            engine: Engine::Sglang,
            model: None,
            tp: None,
            env: vec![],
            args,
            pd: None,
        };
        assert_eq!(
            p.launch_args(),
            vec![
                vec!["--page-size", "64"],
                vec!["--mem-fraction-static", "0.9"],
                vec!["--cuda-graph-bs", "1", "2"]
            ]
        );
    }

    #[test]
    fn reserved_and_dashes_rejected() {
        assert!(overlay_args(&mut vec![], &table("port = 1")).is_err());
        assert!(
            overlay_args(&mut vec![], &table("server-port = 1")).is_err(),
            "ATOM's port flag"
        );
        assert!(overlay_args(&mut vec![], &table("'--page-size' = 1")).is_err());
        assert!(overlay_env(&mut vec![], &table("HIP_VISIBLE_DEVICES = '0'")).is_err());
        assert!(overlay_env(&mut vec![], &table("A = true")).is_err());
    }

    #[test]
    fn env_unset() {
        let mut env = vec![("A".to_string(), "1".to_string())];
        overlay_env(&mut env, &table("A = false\nB = 2")).unwrap();
        assert_eq!(env, vec![("B".to_string(), "2".to_string())]);
    }

    #[test]
    fn targets() {
        let t: ProfileFile = toml::from_str("arch = ['gfx942', ['gfx950', '10.0'], ['*', '10.1']]").unwrap();
        let [a, b, c] = &t.arch[..] else { panic!() };
        assert!(a.fits_arch("gfx942") && a.fits_rocm(None) == Some(true));
        assert!(b.fits_arch("gfx950") && b.fits_rocm(Some("10.0.2")) == Some(true));
        assert_eq!(b.fits_rocm(Some("10.1.0")), Some(false));
        assert_eq!(b.fits_rocm(None), None, "unknown ROCm cannot be judged");
        assert!(c.fits_arch("gfx1250") && c.fits_rocm(Some("10.1.0")) == Some(true));
        assert_eq!(Target::parse("gfx950:10.0").unwrap(), *b);
        assert!(Target::parse("gfx90a").is_err() && Target::parse("gfx950:ten").is_err());
        // Old single-arch strings write back as strings, tuples as arrays.
        let text = toml::to_string(&t).unwrap();
        assert!(
            text.starts_with("arch = [\"gfx942\", [\"gfx950\", \"10.0\"], [\"*\", \"10.1\"]]\n"),
            "{text}"
        );
    }

    #[test]
    fn names() {
        assert!(valid_name("gpt-oss-120b/triton"));
        assert!(!valid_name("gpt-oss-120b"));
        assert_eq!(canonical("gpt-oss-120b").unwrap(), "gpt-oss-120b/base");
        assert_eq!(canonical("gpt-oss-120b/legacy").unwrap(), "gpt-oss-120b/legacy");
        assert!(canonical("../x").is_err());
        assert!(!valid_name("a/b/c"));
        assert!(!valid_name("../x"));
    }

    #[test]
    fn pd_is_inherited_and_overlaid() {
        let dir = std::env::temp_dir().join(format!("oaka-test-pd-{}", std::process::id()));
        let lib = Library { root: dir.clone() };
        fs::create_dir_all(lib.profiles_dir().join("m")).unwrap();
        let write = |n: &str, text: &str| fs::write(lib.profile_path(n), text).unwrap();
        write(
            "m/base",
            "model = '/model/m'\ntp = 2\n[args]\npage-size = 64\nmem-fraction-static = 0.9\n",
        );
        write(
            "m/pd",
            "extends = 'm/base'\n[pd.decode]\ncount = 2\n[pd.decode.args]\nmem-fraction-static = 0.8\n\
             [pd.router.args]\npolicy = 'round_robin'\n",
        );
        write(
            "m/pd2",
            "extends = 'm/pd'\n[pd.prefill.env]\nX = '1'\n[pd.decode.args]\npage-size = false\n",
        );
        let p = lib.resolve("m/pd2").unwrap();
        assert_eq!(p.instances(), 3);
        let pd = p.pd.as_ref().unwrap();
        assert_eq!(
            pd.instances(),
            [(Role::Prefill, 0), (Role::Decode, 0), (Role::Decode, 1)]
        );
        let decode = pd.role(Role::Decode);
        assert!(decode.env.is_empty());
        assert_eq!(
            decode.args,
            [("mem-fraction-static".to_string(), Arg::Value("0.8".into()))]
        );
        let prefill = pd.role(Role::Prefill);
        assert_eq!(prefill.env, [("X".to_string(), "1".to_string())]);
        assert_eq!(
            prefill.args.len(),
            2,
            "the prefill servers keep the profile's args: {prefill:?}"
        );
        assert_eq!(
            pd.role_delta(Role::Decode, &p),
            [
                (
                    "args mem-fraction-static".to_string(),
                    "--mem-fraction-static 0.8".to_string()
                ),
                ("args page-size".to_string(), "--page-size dropped".to_string()),
            ]
        );
        // A later overlay wins over an inherited role's: a child's [args], then the plan's.
        write("m/pd3", "extends = 'm/pd2'\n[args]\npage-size = 32\n");
        let mut p3 = lib.resolve("m/pd3").unwrap();
        p3.overlay(&Table::new(), &table("mem-fraction-static = 0.7")).unwrap();
        let decode = &p3.pd.as_ref().unwrap().decode;
        let arg = |k: &str| decode.args.iter().find(|(n, _)| n == k).unwrap().1.clone();
        assert_eq!(arg("page-size"), Arg::Value("32".into()));
        assert_eq!(arg("mem-fraction-static"), Arg::Value("0.7".into()));
        assert_eq!(
            pd.router_args,
            [("policy".to_string(), Arg::Value("round_robin".into()))]
        );
        assert_eq!(lib.resolve("m/base").unwrap().instances(), 1);

        let bad = |text: &str, needle: &str| {
            write("m/bad", text);
            let e = format!("{:#}", lib.resolve("m/bad").unwrap_err());
            assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        };
        bad("[pd.prefill]\ncount = 0\n", "[pd.prefill]: count must be at least 1");
        bad(
            "[pd.decode.args]\ndisaggregation-mode = 'prefill'\n",
            "--disaggregation-mode is set by oaka from the profile's [pd]",
        );
        bad("[args]\ndisaggregation-bootstrap-port = 1\n", "from the profile's [pd]");
        bad("[pd.router.args]\n'port=1' = true\n", "a flag name is letters, digits");
        bad("[args]\n'tp=4' = true\n", "a flag name is letters, digits");
        bad(
            "[pd.router.args]\npolicy = ['round_robin', '--port=1']\n",
            "value \"--port=1\" would be read as a flag",
        );
        bad(
            "[pd.router.args]\npo = 1\n",
            "--po abbreviates --port, which oaka sets for the router from [pd]",
        );
        bad(
            "[pd.prefill.args]\ndisaggregation-mo = 'decode'\n",
            "--disaggregation-mode is set by oaka from the profile's [pd]",
        );
        bad(
            "[pd.router.args]\nport = 1\n",
            "--port is set by oaka for the router from [pd]",
        );
        bad("[pd.router]\ncount = 2\n", "unknown field `count`");
        bad(
            "extends = 'm/pd'\nengine = 'vllm'\n",
            "[pd] runs SGLang prefill and decode servers, but its engine is vllm",
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn extends_chain() {
        let dir = std::env::temp_dir().join(format!("oaka-test-{}", std::process::id()));
        let lib = Library { root: dir.clone() };
        fs::create_dir_all(lib.profiles_dir().join("m")).unwrap();
        fs::write(
            lib.profile_path("m/base"),
            "model = '/model/m'\ntp = 2\n[env]\nX = '1'\n[args]\npage-size = 1\nfoo = true\n",
        )
        .unwrap();
        fs::write(
            lib.profile_path("m/var"),
            "extends = 'm'\narch = ['gfx942', ['gfx950', '10.1']]\n[env]\nY = '2'\n[args]\npage-size = 64\nfoo = false\n",
        )
        .unwrap();
        let p = lib.resolve("m/var").unwrap();
        assert_eq!(p.engine, Engine::Sglang, "no profile says: SGLang");
        assert_eq!(p.model.as_deref(), Some("/model/m"));
        assert_eq!(p.tp, Some(2));
        assert_eq!(targets_text(&p.arch), "gfx942 gfx950:10.1");
        fs::write(lib.profile_path("m/child"), "extends = 'm/var'\n").unwrap();
        assert!(lib.resolve("m/child").unwrap().arch.is_empty(), "arch is not inherited");
        fs::write(lib.profile_path("m/vllm"), "extends = 'm/var'\nengine = 'vllm'\n").unwrap();
        fs::write(lib.profile_path("m/vllm2"), "extends = 'm/vllm'\n").unwrap();
        assert_eq!(
            lib.resolve("m/vllm2").unwrap().engine,
            Engine::Vllm,
            "the engine is inherited"
        );
        fs::write(lib.profile_path("m/bad"), "engine = 'trt'\n").unwrap();
        assert!(lib.resolve("m/bad").is_err());
        for f in ["m/vllm", "m/vllm2", "m/bad"] {
            fs::remove_file(lib.profile_path(f)).unwrap();
        }
        fs::write(lib.profile_path("m/badrocm"), "arch = [['gfx950', '>=10.1']]\n").unwrap();
        let e = format!("{:#}", lib.resolve("m/badrocm").unwrap_err());
        assert!(e.contains("ROCm version \">=10.1\" must be dotted numbers or *"), "{e}");
        fs::remove_file(lib.profile_path("m/badrocm")).unwrap();
        fs::remove_file(lib.profile_path("m/child")).unwrap();
        assert_eq!(p.env.len(), 2);
        assert_eq!(p.args, vec![("page-size".to_string(), Arg::Value("64".into()))]);
        assert_eq!(lib.list().unwrap(), ["m/base", "m/var"]);
        assert_eq!(lib.resolve("m").unwrap().tp, Some(2));
        fs::write(lib.profile_path("m/old"), "# a comment\norigin = '/2026/ww41/x.sh'\n").unwrap();
        let e = format!("{:#}", lib.resolve("m/old").unwrap_err());
        assert!(e.contains("`origin` is no longer a field"), "{e}");
        fs::write(lib.profile_path("m/base"), "extends = 'm/var'\n").unwrap();
        assert!(lib.resolve("m/var").is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
