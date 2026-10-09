//! Swappable packages: `<lib>/stacks.toml`.
//!
//! Each top-level table names a package that a plan may install from another tree or
//! commit (sglang, later aiter, triton): the repo its trees are worktrees of, the install
//! recipe (bash, inlined into the compiled scripts/stack.sh), how to verify it, and what
//! to clean.  Packages install in file order.  Like profiles, the file is self-contained:
//! provenance is `#` comments, and nothing points outside the library except the repo and
//! cache paths of the machine it describes.

use crate::profile::Library;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use toml::Table;

pub const STACKS: &str = "stacks.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub description: Option<String>,
    /// The git repo whose worktrees the plan's trees are, e.g. the image's checkout.
    pub repo: String,
    /// Python module that must import from inside the tree after an install.
    pub module: String,
    /// Tracked files the install edits in place: restored after every install.  Any
    /// other modified tracked file blocks a checkout.
    #[serde(default)]
    pub restore: Vec<String>,
    /// A directory of the tree (`.` = its root) that servers get first on PYTHONPATH, for
    /// packages the image also puts on PYTHONPATH (aiter, via /etc/bash.bashrc): such an
    /// entry beats any install, so the tree must come before it.
    pub pythonpath: Option<String>,
    /// bash, run with `set -euo pipefail` in the tree: one recipe, or one per GPU arch.
    pub install: Install,
    #[serde(default)]
    pub clean: Clean,
}

/// The install recipe: the same for every GPU arch, or a table keyed by arch where the
/// build differs (`"gfx942 gfx950" = ...`, `gfx1250 = ...`), as rocm.Dockerfile keys its
/// builds on GPU_ARCH_LIST.  An arch with no key has no recipe.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Install {
    Any(String),
    ByArch(BTreeMap<String, String>),
}

/// What may be deleted at any time at the cost of a rebuild: caches, never install state.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clean {
    /// Absolute paths or globs (`~/` = $HOME), e.g. JIT caches.
    #[serde(default)]
    pub paths: Vec<String>,
    /// Globs relative to the tree; only paths git ignores are removed.
    #[serde(default)]
    pub tree: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Stacks {
    pub path: PathBuf,
    /// In file order: the order of installation.
    pub packages: Vec<(String, Package)>,
}

impl Stacks {
    pub fn get(&self, name: &str) -> Option<&Package> {
        self.packages.iter().find(|(n, _)| n == name).map(|(_, p)| p)
    }

    pub fn names(&self) -> Vec<&str> {
        self.packages.iter().map(|(n, _)| n.as_str()).collect()
    }
}

pub fn path(lib: &Library) -> PathBuf {
    lib.root.join(STACKS)
}

/// `<lib>/stacks.toml`; no file means no swappable packages.
pub fn load(lib: &Library) -> Result<Stacks> {
    let path = path(lib);
    if !path.exists() {
        return Ok(Stacks {
            path,
            packages: Vec::new(),
        });
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut s = parse(&text).with_context(|| format!("{}", path.display()))?;
    s.path = path;
    Ok(s)
}

pub fn parse(text: &str) -> Result<Stacks> {
    let raw: Table = toml::from_str(text)?;
    let mut packages = Vec::new();
    for (name, v) in raw {
        if !valid_package_name(&name) {
            bail!("package name {name:?} must match [A-Za-z0-9][A-Za-z0-9_-]*");
        }
        let p: Package = v.try_into().with_context(|| format!("package {name}"))?;
        validate(&p).with_context(|| format!("package {name}"))?;
        packages.push((name, p));
    }
    Ok(Stacks {
        path: PathBuf::new(),
        packages,
    })
}

impl Package {
    /// The recipe for `arch` (this machine's GPU arch, None if unknown).
    pub fn recipe(&self, arch: Option<&str>) -> Result<&str> {
        let table = match &self.install {
            Install::Any(r) => return Ok(r),
            Install::ByArch(t) => t,
        };
        let Some(arch) = arch else {
            bail!(
                "its recipe depends on the GPU arch ({}), and this machine's is unknown \
                 (no GPUs, mixed archs, or set OAKA_GPUS)",
                self.archs().join(" ")
            );
        };
        table
            .iter()
            .find(|(k, _)| k.split_whitespace().any(|a| a == arch))
            .map(|(_, r)| r.as_str())
            .with_context(|| format!("no recipe for {arch}; it has recipes for: {}", self.archs().join(" ")))
    }

    /// The archs with a recipe, in crate::sys::ARCHES order; empty = any.
    pub fn archs(&self) -> Vec<&'static str> {
        match &self.install {
            Install::Any(_) => Vec::new(),
            Install::ByArch(t) => crate::sys::ARCHES
                .iter()
                .copied()
                .filter(|a| t.keys().any(|k| k.split_whitespace().any(|x| x == *a)))
                .collect(),
        }
    }

    /// The PYTHONPATH entry for `tree`, if the package has one.
    pub fn pythonpath_in(&self, tree: &str) -> Option<String> {
        self.pythonpath.as_ref().map(|pp| match pp.trim_end_matches('/') {
            "." | "" => tree.to_string(),
            rel => format!("{tree}/{}", rel.trim_start_matches("./")),
        })
    }
}

pub fn valid_package_name(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_alphanumeric())
        && c.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

/// Characters a clean or restore path may use: they are spliced into bash unquoted (so
/// globs expand), so nothing that could quote, expand or separate words.
fn plain_path(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-+@=,*?[]".contains(c))
        && !s.split('/').any(|part| part == "..")
}

fn validate(p: &Package) -> Result<()> {
    if !p.repo.starts_with('/') {
        bail!("repo {:?} must be an absolute path", p.repo);
    }
    if p.module.is_empty()
        || !p
            .module
            .split('.')
            .all(|m| !m.is_empty() && m.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
    {
        bail!("module {:?} is not a Python module name", p.module);
    }
    if let Install::ByArch(t) = &p.install {
        if t.is_empty() {
            bail!("install has no recipe");
        }
        let mut seen: Vec<&str> = Vec::new();
        for a in t.keys().flat_map(|k| k.split_whitespace()) {
            if !crate::sys::ARCHES.contains(&a) {
                bail!("install: {a:?} is not one of {}", crate::sys::ARCHES.join(" "));
            }
            if seen.contains(&a) {
                bail!("install: two recipes for {a}");
            }
            seen.push(a);
        }
    }
    for r in &p.restore {
        if r.starts_with('/') || !plain_path(r) || r.contains(['*', '?', '[']) {
            bail!("restore {r:?} must be a plain path relative to the tree");
        }
    }
    if let Some(pp) = &p.pythonpath {
        if pp != "." && (pp.starts_with('/') || !plain_path(pp) || pp.contains(['*', '?', '['])) {
            bail!("pythonpath {pp:?} must be `.` or a plain path relative to the tree");
        }
    }
    for c in &p.clean.paths {
        clean_path(c)?;
    }
    for t in &p.clean.tree {
        if t.starts_with('/') || !plain_path(t) || t.split('/').any(|part| part == ".git") {
            bail!("clean.tree {t:?} must be a glob relative to the tree, outside .git, without ..");
        }
    }
    Ok(())
}

/// A clean path as a double-quoted bash word, globs still unexpanded: `~/x*` becomes
/// `"$HOME/x*"`.  Rejects anything that could reach
/// `/`, `$HOME` itself or a parent directory.
pub fn clean_path(c: &str) -> Result<String> {
    let (home, rest) = match c.strip_prefix("~/").or_else(|| c.strip_prefix("$HOME/")) {
        Some(rest) => (true, rest),
        None if c.starts_with('/') => (false, &c[1..]),
        None => bail!("clean.paths {c:?} must be absolute or start with ~/"),
    };
    let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty() && *p != ".").collect();
    if !plain_path(rest) || parts.is_empty() || (!home && parts.len() < 2) {
        bail!("clean.paths {c:?} is too broad or has characters other than [A-Za-z0-9/._-+@=,] and globs");
    }
    // The leading directories (one under $HOME, two under /) must be literal.
    let fixed = if home { 1 } else { 2 };
    if parts[..fixed].iter().any(|p| p.contains(['*', '?', '['])) {
        bail!("clean.paths {c:?} is too broad: its first {fixed} directories must not be globs");
    }
    Ok(if home {
        format!("\"$HOME/{}\"", parts.join("/"))
    } else {
        format!("\"/{}\"", parts.join("/"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OK: &str = r#"
[zeta]
repo = "/r/z"
module = "z"
[zeta.install]
"gfx942 gfx950" = "build for 9xx"
gfx1250 = "build for 1250"

[alpha]
description = "a"
repo = "/r/a"
module = "a.b"
restore = ["python/pyproject.toml"]
pythonpath = "."
install = "pip install -e ."
[alpha.clean]
paths = ["~/.cache/a/jit", "/opt/venv/lib/x-*.egg"]
tree = ["**/__pycache__"]
"#;

    #[test]
    fn parses_in_file_order() {
        let s = parse(OK).unwrap();
        assert_eq!(s.names(), ["zeta", "alpha"]);
        assert_eq!(s.get("alpha").unwrap().clean.tree, ["**/__pycache__"]);
        assert_eq!(s.get("alpha").unwrap().pythonpath_in("/t").as_deref(), Some("/t"));
        assert_eq!(s.get("zeta").unwrap().pythonpath_in("/t"), None);
        let (zeta, alpha) = (s.get("zeta").unwrap(), s.get("alpha").unwrap());
        assert_eq!(zeta.recipe(Some("gfx950")).unwrap(), "build for 9xx");
        assert_eq!(zeta.recipe(Some("gfx1250")).unwrap(), "build for 1250");
        assert_eq!(alpha.recipe(None).unwrap(), "pip install -e .");
        let e = format!("{:#}", zeta.recipe(None).unwrap_err());
        assert!(e.contains("depends on the GPU arch (gfx942 gfx950 gfx1250)"), "{e}");
        let only = parse("[t]\nrepo = '/r'\nmodule = 't'\n[t.install]\ngfx1250 = 'x'\n").unwrap();
        let e = format!("{:#}", only.get("t").unwrap().recipe(Some("gfx950")).unwrap_err());
        assert!(e.contains("no recipe for gfx950; it has recipes for: gfx1250"), "{e}");
    }

    #[test]
    fn clean_paths_are_bounded() {
        assert_eq!(clean_path("~/.cache/a").unwrap(), "\"$HOME/.cache/a\"");
        assert_eq!(clean_path("$HOME/.cache").unwrap(), "\"$HOME/.cache\"");
        assert_eq!(clean_path("/opt/x/*.egg").unwrap(), "\"/opt/x/*.egg\"");
        for bad in [
            "/",
            "/opt",
            "~/",
            "~",
            "relative/x",
            "/opt/../etc",
            "~/a b",
            "/opt/$(x)",
            "~/*",
            "/*/*",
        ] {
            assert!(clean_path(bad).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn rejects() {
        let bad = |text: &str, needle: &str| {
            let e = format!("{:#}", parse(text).expect_err(needle));
            assert!(e.contains(needle), "{e:?} lacks {needle:?}");
        };
        bad(&OK.replace("/r/z", "r/z"), "absolute");
        bad(&OK.replace("\"z\"", "\"z-y\""), "Python module");
        bad(&OK.replace("**/__pycache__", "../x"), "clean.tree");
        bad(&OK.replace("**/__pycache__", ".git/hooks"), "clean.tree");
        bad(&OK.replace("python/pyproject.toml", "/etc/x"), "restore");
        bad(&OK.replace("pythonpath = \".\"", "pythonpath = \"/opt\""), "pythonpath");
        bad(
            &OK.replace("module = \"z\"", "module = \"z\"\nextra = 1"),
            "unknown field",
        );
        bad(&OK.replace("gfx1250 = ", "gfx90a = "), "not one of");
        bad(&OK.replace("gfx1250 = ", "gfx950 = "), "two recipes for gfx950");
    }
}
