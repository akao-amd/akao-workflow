//! Server profiles: `<lib>/profiles/<model>/<recipe>.toml`.
//!
//! A profile is a launch recipe for `python3 -m sglang.launch_server`: the model, env vars
//! and flags.  `extends` makes a profile an overlay of another, which is how experiments
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

/// Flags and env vars that oaka sets itself from the plan.
pub const RESERVED_ARGS: &[&str] = &["model-path", "tp", "tp-size", "tensor-parallel-size", "port"];
pub const RESERVED_ENV: &[&str] = &["HIP_VISIBLE_DEVICES"];

/// One profile file, as written on disk.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Multiple choice from sys::ARCHES; empty = any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arch: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tp: Option<u32>,
    #[serde(default)]
    pub env: Table,
    #[serde(default)]
    pub args: Table,
}

/// A `launch_server` flag value.
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
    pub arch: Vec<String>,
    pub model: Option<String>,
    pub tp: Option<u32>,
    pub env: Vec<(String, String)>,
    pub args: Vec<(String, Arg)>,
}

impl Profile {
    /// `--flag value ...` for everything in `args`, in order.
    pub fn launch_args(&self) -> Vec<Vec<String>> {
        self.args
            .iter()
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

    pub fn arg(&self, key: &str) -> Option<&Arg> {
        self.args.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Apply an `[env]` / `[args]` overlay (from a child profile or a plan).
    pub fn overlay(&mut self, env: &Table, args: &Table) -> Result<()> {
        overlay_env(&mut self.env, env)?;
        overlay_args(&mut self.args, args)
    }
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
    for (k, v) in t {
        if k.starts_with('-') {
            bail!("args key {k:?}: write flags without the leading dashes");
        }
        if RESERVED_ARGS.contains(&k.as_str()) {
            bail!("args {k}: --{k} is set by oaka from the plan; remove it");
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

/// The oaka library: `$OAKA_LIB`, else `/<year>/oaka`.
pub struct Library {
    pub root: PathBuf,
}

impl Library {
    pub fn locate() -> Library {
        let root = match std::env::var_os("OAKA_LIB") {
            Some(r) if !r.is_empty() => PathBuf::from(r),
            _ => PathBuf::from(format!("/{}/oaka", crate::sys::work_year())),
        };
        Library { root }
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
            model: None,
            tp: None,
            env: Vec::new(),
            args: Vec::new(),
        };
        for (n, f) in chain.iter().rev() {
            p.model = f.model.clone().or(p.model);
            p.tp = f.tp.or(p.tp);
            p.overlay(&f.env, &f.args).with_context(|| format!("profile {n}"))?;
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
    for a in &f.arch {
        if !crate::sys::ARCHES.contains(&a.as_str()) {
            bail!(
                "{}: arch {a:?} is not one of {}",
                path.display(),
                crate::sys::ARCHES.join(" ")
            );
        }
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
            model: None,
            tp: None,
            env: vec![],
            args,
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
            "extends = 'm'\narch = ['gfx950']\n[env]\nY = '2'\n[args]\npage-size = 64\nfoo = false\n",
        )
        .unwrap();
        let p = lib.resolve("m/var").unwrap();
        assert_eq!(p.model.as_deref(), Some("/model/m"));
        assert_eq!(p.tp, Some(2));
        assert_eq!(p.arch, ["gfx950"]);
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
