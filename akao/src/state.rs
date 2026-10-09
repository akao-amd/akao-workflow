//! Persistent state under `$AKAO_CONFIG_ROOT`:
//!
//! ```text
//! $AKAO_CONFIG_ROOT/
//!   config.toml       key = "value" settings (see KEYS)
//!   hosts.tsv         one row per remote box (see Host)
//!   container_home/   template copied to <host_home>/container_home/akao_<name>
//! ```
//!
//! The work year and week are not stored: they are the ISO year/week of today.

use anyhow::{bail, Context, Result};
use chrono::Datelike;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_DOCKER_SOCK: &str = "/var/run/docker.sock";

/// Known config keys: (name, default, description). `{year}` expands to the work year.
pub const KEYS: &[(&str, Option<&str>, &str)] = &[
    ("default_image", None, "docker image tag for hosts without their own"),
    ("deploy_src", Some("/{year}"), "local directory holding the control plane"),
    (
        "deploy_paths",
        Some("CLAUDE.md AGENTS.md AGCP.md skills utils"),
        "space-separated paths under deploy_src shipped to <host_home>/<year>",
    ),
];

pub struct State {
    pub root: PathBuf,
    config: BTreeMap<String, String>,
}

impl State {
    pub fn load() -> Result<State> {
        let root = match std::env::var_os("AKAO_CONFIG_ROOT") {
            Some(r) if !r.is_empty() => PathBuf::from(r),
            _ => bail!("AKAO_CONFIG_ROOT is not set (e.g. export AKAO_CONFIG_ROOT=/2026/nocopy/akao-workflow-state)"),
        };
        if !root.is_dir() {
            bail!("AKAO_CONFIG_ROOT={} is not a directory", root.display());
        }
        let path = root.join("config.toml");
        let config = if path.exists() {
            let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
        } else {
            BTreeMap::new()
        };
        Ok(State { root, config })
    }

    pub fn hosts_path(&self) -> PathBuf {
        self.root.join("hosts.tsv")
    }

    pub fn home_template(&self) -> PathBuf {
        self.root.join("container_home")
    }

    /// Effective value of a known key: explicit setting, else default, `{year}` expanded.
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let (_, default, _) = KEYS
            .iter()
            .find(|(k, _, _)| *k == key)
            .with_context(|| format!("unknown config key '{key}' (see `akao config ls`)"))?;
        let v = self.config.get(key).map(String::as_str).or(*default);
        Ok(v.map(|v| v.replace("{year}", &work_year())))
    }

    pub fn require(&self, key: &str) -> Result<String> {
        self.get(key)?
            .with_context(|| format!("config key '{key}' is not set; run `akao config set {key} <value>`"))
    }

    pub fn is_explicit(&self, key: &str) -> bool {
        self.config.contains_key(key)
    }

    pub fn set(&mut self, key: &str, value: Option<&str>) -> Result<()> {
        self.get(key)?; // validates the key
        match value {
            Some(v) => self.config.insert(key.to_string(), v.to_string()),
            None => self.config.remove(key),
        };
        let path = self.root.join("config.toml");
        fs::write(&path, toml::to_string(&self.config)?).with_context(|| format!("writing {}", path.display()))
    }

    pub fn load_hosts(&self) -> Result<Vec<Host>> {
        let path = self.hosts_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        parse_hosts(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save_hosts(&self, hosts: &[Host]) -> Result<()> {
        write_atomic(&self.hosts_path(), &format_hosts(hosts))
    }

    pub fn host(&self, nick: &str) -> Result<Host> {
        self.load_hosts()?.into_iter().find(|h| h.nick == nick).with_context(|| {
            format!(
                "host '{nick}' is not in {}; add it with `akao host add {nick} --home <dir> --model <dir>`",
                self.hosts_path().display()
            )
        })
    }
}

fn write_atomic(path: &Path, text: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))
}

/// ISO year of today, e.g. "2026".
pub fn work_year() -> String {
    chrono::Local::now().iso_week().year().to_string()
}

/// ISO week of today, e.g. "ww41".
pub fn work_week() -> String {
    format!("ww{:02}", chrono::Local::now().iso_week().week())
}

/// One row of hosts.tsv. Optional columns are written as `-`.
#[derive(Debug, Clone, PartialEq)]
pub struct Host {
    /// ssh Host alias, resolved through ~/.ssh/config.
    pub nick: String,
    /// Image tag; None means config `default_image`.
    pub image: Option<String>,
    /// Model directory on the host, mounted as /model.
    pub model_path: String,
    /// Docker socket on the host; None means /var/run/docker.sock.
    pub docker_sock: Option<String>,
    /// The root of everything we create on the host (holds <year>/ and container_home/).
    pub host_home: String,
    /// Extra `docker run` arguments appended after the skeleton, shell-quoted.
    pub rest: Option<String>,
}

pub const HOSTS_HEADER: &str = "# nick\timage\tmodel_path\tdocker_sock\thost_home\trest";

impl Host {
    pub fn docker_sock(&self) -> &str {
        self.docker_sock.as_deref().unwrap_or(DEFAULT_DOCKER_SOCK)
    }

    pub fn rest_args(&self) -> Result<Vec<String>> {
        match &self.rest {
            None => Ok(Vec::new()),
            Some(r) => shlex::split(r).with_context(|| format!("host '{}': cannot parse rest args: {r}", self.nick)),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !valid_name(&self.nick) {
            bail!("invalid host nick '{}'", self.nick);
        }
        for (what, v) in [("model_path", &self.model_path), ("host_home", &self.host_home)] {
            if !v.starts_with('/') {
                bail!("host '{}': {what} must be an absolute path: {v}", self.nick);
            }
        }
        if let Some(s) = &self.docker_sock {
            if !s.starts_with('/') {
                bail!("host '{}': docker_sock must be an absolute path: {s}", self.nick);
            }
        }
        let fields = [Some(&self.nick), self.image.as_ref(), Some(&self.model_path), self.docker_sock.as_ref(), Some(&self.host_home), self.rest.as_ref()];
        for f in fields.into_iter().flatten() {
            if f.is_empty() || f == "-" || f.contains(['\t', '\n', '\r']) {
                bail!("host '{}': field {f:?} is empty, '-', or contains a tab/newline", self.nick);
            }
        }
        self.rest_args()?;
        Ok(())
    }
}

/// Docker container names and ssh aliases we accept: [A-Za-z0-9][A-Za-z0-9_.-]*
pub fn valid_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

fn opt(field: &str) -> Option<String> {
    (field != "-" && !field.is_empty()).then(|| field.to_string())
}

pub fn parse_hosts(text: &str) -> Result<Vec<Host>> {
    let mut hosts: Vec<Host> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 5 || f.len() > 6 {
            bail!("line {}: expected 5 or 6 tab-separated columns, got {}", i + 1, f.len());
        }
        let host = Host {
            nick: f[0].to_string(),
            image: opt(f[1]),
            model_path: f[2].to_string(),
            docker_sock: opt(f[3]),
            host_home: f[4].to_string(),
            rest: f.get(5).and_then(|r| opt(r)),
        };
        host.validate().with_context(|| format!("line {}", i + 1))?;
        if hosts.iter().any(|h| h.nick == host.nick) {
            bail!("line {}: duplicate host '{}'", i + 1, host.nick);
        }
        hosts.push(host);
    }
    Ok(hosts)
}

pub fn format_hosts(hosts: &[Host]) -> String {
    let dash = |o: &Option<String>| o.clone().unwrap_or_else(|| "-".into());
    let mut out = format!("{HOSTS_HEADER}\n");
    for h in hosts {
        out += &format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            h.nick,
            dash(&h.image),
            h.model_path,
            dash(&h.docker_sock),
            h.host_home,
            dash(&h.rest)
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Host {
        Host {
            nick: "f19-11".into(),
            image: None,
            model_path: "/mnt/raid/models".into(),
            docker_sock: Some("/data/docker.sock".into()),
            host_home: "/root/akao".into(),
            rest: Some("--shm-size=64g -e 'FOO=a b'".into()),
        }
    }

    #[test]
    fn hosts_roundtrip() {
        let hosts = vec![sample()];
        let text = format_hosts(&hosts);
        assert!(text.starts_with(HOSTS_HEADER));
        assert_eq!(parse_hosts(&text).unwrap(), hosts);
    }

    #[test]
    fn hosts_dash_and_short_rows() {
        let h = parse_hosts("h21-4\t-\t/m\t-\t/home/akao\n").unwrap();
        assert_eq!(h[0].image, None);
        assert_eq!(h[0].docker_sock(), DEFAULT_DOCKER_SOCK);
        assert_eq!(h[0].rest, None);
    }

    #[test]
    fn hosts_reject_bad_rows() {
        assert!(parse_hosts("a\tb\n").is_err());
        assert!(parse_hosts("a\t-\trelative\t-\t/h\n").is_err());
        assert!(parse_hosts("a\t-\t/m\t-\t/h\na\t-\t/m\t-\t/h\n").is_err());
        assert!(parse_hosts("a\t-\t/m\t-\t/h\t'unterminated\n").is_err());
    }

    #[test]
    fn rest_args_are_shell_split() {
        assert_eq!(sample().rest_args().unwrap(), ["--shm-size=64g", "-e", "FOO=a b"]);
    }

    #[test]
    fn names() {
        assert!(valid_name("dsv4_exp"));
        assert!(valid_name("h21-17"));
        assert!(!valid_name("_x"));
        assert!(!valid_name("a b"));
        assert!(!valid_name(""));
    }
}
