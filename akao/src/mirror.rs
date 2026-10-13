//! `akao mirror`: a worker for one InferenceX benchmark config.
//!
//! The entry comes from this machine's InferenceX clone (config `infx_local`), read through
//! git at a revision, never from the working tree: what is mirrored is a named commit, and
//! retired entries (gpt-oss, removed 2026-07-06) stay reachable with `--rev`.  The worker is
//! `akao init` with the entry's image, plus a brief in its artifact root,
//! `mirror/<entry>@<sha12>/`: MIRROR.md (what the entry is, its points, what oaka can and
//! cannot reproduce), entry.yaml (its text, comments kept) and the files that define its
//! server at that revision.  Turning those into an oaka profile is the worker agent's job.
//!
//! With `--dashboard` the source is what the InferenceX dashboard measured instead
//! (`dashboard.rs`): a curve's latest run, the image it ran, the commit it ran at (a
//! snapshot from GitHub, so this machine's clone plays no part), its measured points and
//! the server logs with the arguments each server really had, all added to the brief.

use crate::dashboard::{self, Evidence};
use crate::exec::{q, Runner, ESCALATE};
use crate::init;
use crate::state::State;
use anyhow::{bail, Context, Result};
use serde_yaml::{Mapping, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// InferenceX frameworks oaka has an engine for.
pub const ENGINES: &[&str] = &["sglang", "vllm", "atom"];

pub struct Options {
    /// Comma-separated terms naming one entry, e.g. "gptoss,mi355,atom".
    pub conf: String,
    pub rev: Option<String>,
    /// Read the dashboard (what it measured) instead of the clone at `rev`.
    pub dashboard: bool,
    /// Dashboard only: its snapshot as of this date (YYYY-MM-DD).
    pub date: Option<String>,
    /// Dashboard only: this GitHub Actions run of the curve instead of its latest.
    pub run: Option<String>,
    /// Bring it up on this host; None lists or previews the matches only.
    pub nick: Option<String>,
    pub name: Option<String>,
    pub week: Option<String>,
    pub artifact_root: Option<String>,
    pub skip_setup: bool,
}

/// `git -C <dir> <args>`: stdout, or an error with git's message.
fn git(dir: &str, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .context("cannot run git")?;
    if !out.status.success() {
        bail!(
            "git -C {dir} {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The InferenceX tree at one commit: this machine's clone read through git, or a snapshot
/// of the commit extracted into a directory (the dashboard's source).
pub struct Repo {
    pub dir: String,
    pub sha: String,
    /// "<sha7> <date> <subject>"
    pub title: String,
    pub files: Vec<String>,
    /// Files are read from `dir` itself (a snapshot), not through git.
    plain: bool,
}

impl Repo {
    /// The extracted tree of commit `sha` in `dir`.
    pub fn snapshot(dir: &Path, sha: &str, title: String) -> Result<Repo> {
        let mut files = Vec::new();
        let mut todo = vec![dir.to_path_buf()];
        while let Some(d) = todo.pop() {
            for e in fs::read_dir(&d).with_context(|| format!("reading {}", d.display()))? {
                let e = e?;
                if e.file_type()?.is_dir() {
                    todo.push(e.path());
                } else if let Ok(rel) = e.path().strip_prefix(dir) {
                    files.push(rel.to_string_lossy().into_owned());
                }
            }
        }
        files.sort();
        Ok(Repo {
            dir: dir.display().to_string(),
            sha: sha.to_string(),
            title,
            files,
            plain: true,
        })
    }

    pub fn open(dir: &str, rev: &str) -> Result<Repo> {
        let sha = git(
            dir,
            &[
                "rev-parse",
                "--verify",
                "--end-of-options",
                &format!("{rev}^{{commit}}"),
            ],
        )
        .with_context(|| format!("InferenceX revision {rev:?} in {dir} (config infx_local)"))?
        .trim()
        .to_string();
        let title = git(dir, &["log", "-1", "--format=%h %cs %s", &sha])?.trim().to_string();
        let files = git(dir, &["ls-tree", "-r", "--name-only", &sha])?
            .lines()
            .map(String::from)
            .collect();
        Ok(Repo {
            dir: dir.to_string(),
            sha,
            title,
            files,
            plain: false,
        })
    }

    pub fn show(&self, path: &str) -> Result<String> {
        if self.plain {
            let p = Path::new(&self.dir).join(path);
            return fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()));
        }
        git(&self.dir, &["show", &format!("{}:{path}", self.sha)])
    }
}

/// One entry of a `*master.yaml`.
pub struct Entry {
    pub name: String,
    /// The master config it is in, repo-relative.
    pub file: String,
    /// Where paths in the entry are relative to: "" or "inferencex-e2e/" (the dir above
    /// configs/, or above .github/configs/ as in mid-2026).
    pub root: String,
    /// The entry as written, comments included.
    pub raw: String,
    pub fields: Mapping,
}

fn scalar(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Lowercase without `-_.`: "gpt-oss" and "gptoss" compare equal.
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '-' | '_' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

impl Entry {
    pub fn get(&self, key: &str) -> Option<String> {
        self.fields.get(key).and_then(scalar)
    }

    fn field(&self, key: &str) -> String {
        self.get(key).unwrap_or_else(|| "?".into())
    }

    fn flag(&self, key: &str) -> bool {
        self.fields.get(key).and_then(Value::as_bool).unwrap_or(false)
    }

    /// The GPU of its runner, e.g. "mi355x" from "mi355x" or "cluster:mi355x-amds".
    pub fn gpu(&self) -> Option<String> {
        let runner = self.get("runner")?.to_lowercase();
        runner
            .split(|c: char| !c.is_ascii_alphanumeric())
            .find(|t| {
                ["mi", "b", "h", "gb"].iter().any(|p| {
                    t.strip_prefix(p)
                        .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
                })
            })
            .map(String::from)
    }

    /// The GPU arch its runner has, for the AMD GPUs oaka knows.
    pub fn arch(&self) -> Option<&'static str> {
        match self.gpu()?.as_str() {
            "mi300x" | "mi300a" | "mi325x" => Some("gfx942"),
            "mi350x" | "mi355x" => Some("gfx950"),
            _ => None,
        }
    }

    /// Why one container with oaka cannot run it; empty when it can.
    pub fn problems(&self) -> Vec<String> {
        let mut p = Vec::new();
        if self.flag("multinode") || self.flag("disagg") {
            p.push("multinode/disaggregated: one container cannot hold it".to_string());
        }
        let fw = self.field("framework");
        if !ENGINES.contains(&fw.as_str()) {
            p.push(format!("framework {fw}: oaka serves only {}", ENGINES.join(", ")));
        }
        if !self.gpu().is_some_and(|g| g.starts_with("mi")) {
            p.push(format!("runner {}: not an AMD Instinct GPU", self.field("runner")));
        }
        p
    }

    /// What `--conf` terms are matched against.
    fn tokens(&self) -> Vec<String> {
        let mut t: Vec<String> = self.name.split('-').map(norm).collect();
        for key in ["runner", "framework", "model-prefix", "precision"] {
            if let Some(v) = self.get(key) {
                t.extend(v.split([':', '-']).map(norm));
            }
        }
        if let Some(m) = self.get("model") {
            t.push(norm(m.rsplit('/').next().unwrap_or(&m)));
        }
        t
    }

    fn matches(&self, terms: &[String]) -> bool {
        let tokens = self.tokens();
        terms
            .iter()
            .all(|term| norm(term) == norm(&self.name) || tokens.iter().any(|t| t.starts_with(&norm(term))))
    }

    /// The InferenceX frameworks map one to one onto oaka engines.
    fn engine(&self) -> Option<String> {
        let fw = self.field("framework");
        ENGINES.contains(&fw.as_str()).then_some(fw)
    }
}

/// `<name>:` on a line of its own starts an entry; the next top-level key ends it.
fn raw_entry(text: &str, name: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let Some(start) = lines
        .iter()
        .position(|l| l.trim_end() == format!("{name}:") || l.starts_with(&format!("{name}: ")))
    else {
        return String::new();
    };
    let mut end = lines[start + 1..]
        .iter()
        .position(|l| !l.is_empty() && !l.starts_with([' ', '\t', '#']))
        .map_or(lines.len(), |i| start + 1 + i);
    // Comments and blank lines just above the next entry belong to it.
    while end > start + 1 && (lines[end - 1].trim().is_empty() || lines[end - 1].starts_with('#')) {
        end -= 1;
    }
    lines[start..end].join("\n") + "\n"
}

/// The tree's root for a config under `<above>configs/`: benchmarks/ sits beside configs/,
/// except when the configs live in .github/ (the layout of mid-2026).
fn entry_root(above: &str) -> String {
    above.strip_suffix(".github/").unwrap_or(above).to_string()
}

/// Every entry of every `*master.yaml` under a `configs/` dir at this revision.  A file
/// that does not parse is reported, not fatal.
pub fn entries(repo: &Repo) -> Result<(Vec<Entry>, Vec<String>)> {
    let mut out = Vec::new();
    let mut notes = Vec::new();
    for file in &repo.files {
        let Some(at) = file.find("configs/") else {
            continue;
        };
        if !(at == 0 || file[..at].ends_with('/')) || !file.ends_with("master.yaml") {
            continue;
        }
        let text = repo.show(file)?;
        let map: Mapping = match serde_yaml::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                notes.push(format!("{file}: not parsed ({e})"));
                continue;
            }
        };
        for (k, v) in map {
            let (Some(name), Value::Mapping(fields)) = (k.as_str(), v) else {
                continue;
            };
            out.push(Entry {
                raw: raw_entry(&text, name),
                name: name.to_string(),
                file: file.clone(),
                root: entry_root(&file[..at]),
                fields,
            });
        }
    }
    Ok((out, notes))
}

/// The entries `--conf` names: an exact entry name alone, else every entry each term
/// prefix-matches, those with the fewest name parts first (an `-mtp` or `-agentic`
/// variant has more, so naming the base entry does not also pick its variants).
pub fn select<'a>(entries: &'a [Entry], conf: &str) -> Vec<&'a Entry> {
    let terms: Vec<String> = conf
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if terms.is_empty() {
        return Vec::new();
    }
    if terms.len() == 1 {
        let exact: Vec<&Entry> = entries.iter().filter(|e| e.name == terms[0]).collect();
        if !exact.is_empty() {
            return exact;
        }
    }
    let mut found: Vec<&Entry> = entries.iter().filter(|e| e.matches(&terms)).collect();
    found.sort_by_key(|e| (e.name.split('-').count(), e.name.clone()));
    found
}

/// One row of a scenario's search space.
#[derive(Debug, Default)]
pub struct Point {
    pub scenario: String,
    pub isl: Option<String>,
    pub osl: Option<String>,
    pub tp: Option<String>,
    pub ep: Option<String>,
    pub dp_attn: bool,
    pub spec: Option<String>,
    pub conc: Vec<u64>,
    pub recipe: Option<String>,
    /// Keys of the row oaka has no notion of (kv-offloading, dcp-size, ...), as `k=v`.
    pub other: Vec<String>,
}

/// conc-list, or conc-start..conc-end doubling, end included (InferenceX's step).
fn concurrencies(row: &Mapping) -> Vec<u64> {
    if let Some(Value::Sequence(l)) = row.get("conc-list") {
        return l.iter().filter_map(Value::as_u64).collect();
    }
    let (Some(start), Some(end)) = (
        row.get("conc-start").and_then(Value::as_u64),
        row.get("conc-end").and_then(Value::as_u64),
    ) else {
        return Vec::new();
    };
    // InferenceX's _concurrency_range: double, but end on `end` even after overshoot.
    let mut c = Vec::new();
    let mut n = start;
    while n >= 1 && n <= end {
        c.push(n);
        if n == end {
            break;
        }
        n = n.saturating_mul(2).min(end);
    }
    c
}

pub fn points(e: &Entry) -> Vec<Point> {
    let mut out = Vec::new();
    let Some(Value::Mapping(scenarios)) = e.fields.get("scenarios") else {
        return out;
    };
    for (scenario, groups) in scenarios {
        let scenario = scenario.as_str().unwrap_or("?").to_string();
        for group in groups.as_sequence().into_iter().flatten() {
            let get = |k: &str| group.get(k).and_then(scalar);
            for row in group
                .get("search-space")
                .and_then(Value::as_sequence)
                .into_iter()
                .flatten()
            {
                let Some(row) = row.as_mapping() else { continue };
                let s = |k: &str| row.get(k).and_then(scalar);
                let known = [
                    "tp",
                    "ep",
                    "dp-attn",
                    "spec-decoding",
                    "conc-start",
                    "conc-end",
                    "conc-list",
                    "srt-recipe",
                ];
                let other = row
                    .iter()
                    .filter_map(|(k, v)| {
                        let k = k.as_str()?;
                        (!known.contains(&k)).then(|| {
                            format!(
                                "{k}={}",
                                scalar(v).unwrap_or_else(|| serde_yaml::to_string(v)
                                    .unwrap_or_default()
                                    .trim()
                                    .replace('\n', " "))
                            )
                        })
                    })
                    .collect();
                out.push(Point {
                    scenario: scenario.clone(),
                    isl: get("isl"),
                    osl: get("osl"),
                    tp: s("tp"),
                    ep: s("ep"),
                    dp_attn: row.get("dp-attn").and_then(Value::as_bool).unwrap_or(false),
                    spec: s("spec-decoding"),
                    conc: concurrencies(row),
                    recipe: s("srt-recipe"),
                    other,
                });
            }
        }
    }
    out
}

/// A file of the brief: where it is in the repo, and its text.
pub struct Shipped {
    pub path: String,
    pub text: String,
}

/// The files that define the entry's server at this revision: its srt recipes (with the
/// setup scripts they name), or the legacy bash script the runner's launcher would pick,
/// with the benchmark_lib.sh it sources.  Notes say what was looked for and not found.
pub fn recipe_files(repo: &Repo, e: &Entry, pts: &[Point]) -> Result<(Vec<Shipped>, Vec<String>)> {
    let mut paths: Vec<String> = Vec::new();
    let mut notes = Vec::new();
    for p in pts {
        if let Some(r) = &p.recipe {
            // A recipe may name a variant as file.yaml:selector.
            let file = r.split(':').next().unwrap_or(r);
            let full = format!("{}{file}", e.root);
            if !repo.files.contains(&full) {
                notes.push(format!("srt-recipe {r} is not in the tree at this revision"));
            } else if !paths.contains(&full) {
                paths.push(full);
            }
        }
    }
    // Rows without a recipe run the legacy script the runner's launcher picks: in the
    // scenario's directory (benchmarks/single_node/<dir>/, incl. deprecated/ under it),
    // <prefix>_<precision>_<gpu>_<framework><spec>.sh, else without the framework.
    let bench = format!("{}benchmarks/", e.root);
    let mut legacy: Vec<(&str, &str)> = pts
        .iter()
        .filter(|p| p.recipe.is_none())
        .map(|p| {
            let mtp = p.spec.as_deref() == Some("mtp") || e.name.ends_with("-mtp");
            (p.scenario.as_str(), if mtp { "_mtp" } else { "" })
        })
        .collect();
    legacy.sort();
    legacy.dedup();
    for (scenario, spec) in legacy {
        let base = format!(
            "{}_{}_{}",
            e.field("model-prefix"),
            e.field("precision"),
            e.gpu().unwrap_or_default()
        );
        let names = [
            format!("{base}_{}{spec}.sh", e.field("framework")),
            format!("{base}{spec}.sh"),
        ];
        match legacy_script(repo, &bench, scenario, &names) {
            Some(f) if !paths.contains(&f) => paths.push(f),
            Some(_) => {}
            None => notes.push(format!(
                "no {scenario} server script: looked for {} in {bench}single_node/{}/ (and outside any scenario dir)",
                names.join(" and "),
                scenario_dir(scenario)
            )),
        }
    }
    let lib = format!("{bench}benchmark_lib.sh");
    if paths.iter().any(|p| p.ends_with(".sh")) && repo.files.contains(&lib) {
        paths.push(lib);
    }
    let mut shipped: Vec<Shipped> = Vec::new();
    for path in &paths {
        shipped.push(Shipped {
            text: repo.show(path)?,
            path: path.clone(),
        });
    }
    // Setup scripts a recipe names by basename, wherever they live in the tree.
    let mut setups: Vec<String> = Vec::new();
    for s in &shipped {
        if let Ok(v) = serde_yaml::from_str::<Value>(&s.text) {
            collect_key(&v, "setup_script", &mut setups);
        }
    }
    setups.sort();
    setups.dedup();
    for name in setups {
        let hits: Vec<&String> = repo
            .files
            .iter()
            .filter(|f| f.rsplit('/').next() == Some(name.as_str()))
            .collect();
        match hits.as_slice() {
            [] => notes.push(format!("setup_script {name} is not in the tree at this revision")),
            _ => {
                for h in hits {
                    shipped.push(Shipped {
                        text: repo.show(h)?,
                        path: h.clone(),
                    });
                }
            }
        }
    }
    Ok((shipped, notes))
}

/// The launchers' directory for a scenario under benchmarks/single_node/.
fn scenario_dir(scenario: &str) -> &str {
    match scenario {
        "fixed-seq-len" => "fixed_seq_len",
        "agentic-coding" => "agentic",
        s => s,
    }
}

/// The first of `names` found in the scenario's directory (a current script before a
/// deprecated/ one), else in an older single-node layout (outside every scenario's
/// directory); never another scenario's or a multi-node script of the same name.
fn legacy_script(repo: &Repo, bench: &str, scenario: &str, names: &[String]) -> Option<String> {
    let own = format!("{bench}single_node/{}/", scenario_dir(scenario));
    let hits = |name: &str, place: &dyn Fn(&str) -> bool| {
        let mut h: Vec<&String> = repo
            .files
            .iter()
            .filter(|f| f.rsplit('/').next() == Some(name) && place(f))
            .collect();
        h.sort_by_key(|f| f.contains("/deprecated/"));
        h.first().map(|f| f.to_string())
    };
    let in_own = |f: &str| f.starts_with(&own);
    let older = |f: &str| {
        let rel = &f[bench.len().min(f.len())..];
        f.starts_with(bench)
            && !rel.starts_with("multi_node/")
            && !["fixed_seq_len", "agentic"]
                .iter()
                .any(|d| rel.starts_with(&format!("single_node/{d}/")))
    };
    names
        .iter()
        .find_map(|n| hits(n, &in_own))
        .or_else(|| names.iter().find_map(|n| hits(n, &older)))
}

/// Every value of `key` anywhere in `v`, list items one by one (a zip override's
/// per-point values are lists).
fn collect_key(v: &Value, key: &str, out: &mut Vec<String>) {
    match v {
        Value::Mapping(m) => {
            for (k, val) in m {
                if k.as_str() == Some(key) {
                    match val {
                        Value::Sequence(items) => out.extend(items.iter().filter_map(scalar)),
                        _ => out.extend(scalar(val)),
                    }
                }
                collect_key(val, key, out);
            }
        }
        Value::Sequence(s) => s.iter().for_each(|x| collect_key(x, key, out)),
        _ => {}
    }
}

/// What a recipe file says about the client and the server's life, for MIRROR.md.
fn recipe_facts(s: &Shipped) -> Vec<String> {
    let mut f = Vec::new();
    if s.path.ends_with(".sh") {
        // The client call: `run_benchmark_serving \` and its continuation lines.
        let mut call = String::new();
        let mut inside = false;
        for line in s.text.lines() {
            let l = line.trim_start();
            inside |= l.starts_with("run_benchmark_serving") && !l.starts_with("run_benchmark_serving(");
            if inside {
                call += line;
                call += "\n";
                if !line.trim_end().ends_with('\\') {
                    inside = false;
                }
            }
        }
        for flag in ["--use-chat-template", "--trust-remote-code", "--dsv4"] {
            if call.contains(flag) {
                f.push(format!(
                    "client flag `{flag}` (oaka's fixed-seq client cannot pass it: not comparable)"
                ));
            }
        }
        return f;
    }
    let Ok(v) = serde_yaml::from_str::<Value>(&s.text) else {
        return f;
    };
    let mut vals = Vec::new();
    // The client is each group's `benchmark.command` (base, override_*, ...); flags in it
    // are the client's (server flags are the roles' args), and oaka's client passes none.
    let commands: Vec<String> = match &v {
        Value::Mapping(groups) => groups
            .values()
            .filter_map(|g| g.get("benchmark")?.get("command"))
            .filter_map(scalar)
            .collect(),
        _ => Vec::new(),
    };
    let mut flags: Vec<String> = Vec::new();
    for c in &commands {
        match shlex::split(c) {
            Some(words) => flags.extend(words.into_iter().filter(|w| w.starts_with("--"))),
            None => f.push(format!(
                "benchmark.command `{c}` does not parse as shell words: read it yourself"
            )),
        }
    }
    flags.sort();
    flags.dedup();
    for flag in flags {
        f.push(format!(
            "client flag `{flag}` in benchmark.command (oaka's fixed-seq client cannot pass it: not comparable)"
        ));
    }
    for key in ["USE_CHAT_TEMPLATE", "RANDOM_RANGE_RATIO"] {
        vals.clear();
        collect_key(&v, key, &mut vals);
        vals.sort();
        vals.dedup();
        if !vals.is_empty() {
            let note = if key == "USE_CHAT_TEMPLATE" && vals.iter().any(|x| x == "true") {
                " (oaka's fixed-seq client sends no chat template: not comparable)"
            } else {
                ""
            };
            f.push(format!("{key} = {}{note}", vals.join(", ")));
        }
    }
    vals.clear();
    collect_key(&v, "data-parallel-size", &mut vals);
    if vals.iter().any(|d| d.parse::<u64>().is_ok_and(|n| n > 1)) {
        f.push("data-parallel-size > 1: not expressible in oaka yet (tp = the number of GPUs)".into());
    }
    if let Value::Mapping(m) = &v {
        let overrides: Vec<&str> = m
            .keys()
            .filter_map(Value::as_str)
            .filter(|k| k.starts_with("override") || k.starts_with("zip_override"))
            .collect();
        if !overrides.is_empty() {
            f.push(format!(
                "per-point server settings in {}: one profile overlay per distinct setting",
                overrides.join(", ")
            ));
        }
    }
    vals.clear();
    collect_key(&v, "setup_script", &mut vals);
    vals.sort();
    vals.dedup();
    if !vals.is_empty() {
        f.push(format!("setup script {} (shipped; never run by oaka)", vals.join(", ")));
    }
    if let Some(Value::Mapping(h)) = find_key(&v, "health_check") {
        let parts: Vec<String> = h
            .iter()
            .filter_map(|(k, v)| Some(format!("{}={}", k.as_str()?, scalar(v)?)))
            .collect();
        f.push(format!("health_check {} (oaka waits at most 1800 s)", parts.join(" ")));
    }
    f
}

fn find_key<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    match v {
        Value::Mapping(m) => m.iter().find_map(|(k, val)| {
            if k.as_str() == Some(key) {
                Some(val)
            } else {
                find_key(val, key)
            }
        }),
        Value::Sequence(s) => s.iter().find_map(|x| find_key(x, key)),
        _ => None,
    }
}

/// Where a brief goes, under the worker's artifact root.
pub fn brief_dir(e: &Entry, repo: &Repo) -> String {
    format!("mirror/{}@{}", e.name, &repo.sha[..12.min(repo.sha.len())])
}

/// Facts about the container the brief is for; None in a preview.
pub struct Target {
    pub container: String,
    pub host: String,
    pub host_archs: Option<Vec<String>>,
}

pub fn brief(
    repo: &Repo,
    e: &Entry,
    pts: &[Point],
    files: &[Shipped],
    notes: &[String],
    t: Option<&Target>,
    dash: Option<&Evidence>,
) -> String {
    let mut m = String::new();
    let rel = |p: &str| p.strip_prefix(&e.root).unwrap_or(p).to_string();
    m += &format!("# Mirror of InferenceX `{}`\n\n", e.name);
    m += &match t {
        Some(t) => format!(
            "Written by `akao mirror` on {} for `{}` on {}.\n\n",
            chrono::Local::now().format("%Y-%m-%d"),
            t.container,
            t.host
        ),
        None => "Preview (`akao mirror` without a host).\n\n".into(),
    };
    let arch = e.arch().unwrap_or("unknown to akao");
    let engine = e
        .engine()
        .map(|en| format!("oaka engine `{en}`"))
        .unwrap_or_else(|| "no oaka engine".into());
    m += "| | |\n|---|---|\n";
    if let Some(d) = dash {
        m += &format!(
            "| dashboard | {} ({}): curve `{}`, as measured on {} |\n",
            d.base,
            d.display,
            d.curve.name(),
            d.run.date
        );
        m += &format!("| run | [{}]({}) {} |\n", d.run.id, d.run.url, d.run_name);
        if let Some(desc) = &d.description {
            let pr = d.pr.as_deref().map(|p| format!(" ({p})")).unwrap_or_default();
            m += &format!("| change | {}{pr} |\n", desc.replace('\n', " ").replace('|', "\\|"));
        }
    }
    m += &format!("| InferenceX | `{}` ({}) |\n", repo.sha, repo.title);
    m += &format!("| config | `{}` |\n", e.file);
    match dash.and_then(|d| d.run.image.as_deref()) {
        Some(img) if img != e.field("image") => {
            m += &format!(
                "| image | `{img}` (what ran; the config says `{}`) |\n",
                e.field("image")
            )
        }
        _ => m += &format!("| image | `{}` |\n", e.field("image")),
    }
    m += &format!(
        "| model | `{}` (Hugging Face id; the weights are wherever this box keeps them, e.g. under /model) |\n",
        e.field("model")
    );
    m += &format!("| framework | {} → {engine} |\n", e.field("framework"));
    m += &format!("| precision | {} |\n", e.field("precision"));
    m += &format!(
        "| runner | {} → GPU arch {arch} (an arch is not a SKU: gfx942 is MI300X or MI325X) |\n",
        e.field("runner")
    );
    if let Some(archs) = t.and_then(|t| t.host_archs.as_ref()) {
        m += &format!("| this box | {} |\n", summarize(archs));
    }
    let problems = e.problems();
    if !problems.is_empty() {
        m += &format!("| cannot mirror | {} |\n", problems.join("; "));
    }

    m += "\n## Points\n\n";
    m += "| scenario | isl | osl | tp | ep | dp-attn | spec | concurrency | recipe | other |\n";
    m += "|---|---|---|---|---|---|---|---|---|---|\n";
    let d = |o: &Option<String>| o.clone().unwrap_or_else(|| "-".into());
    for p in pts {
        let conc: Vec<String> = p.conc.iter().map(u64::to_string).collect();
        m += &format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            p.scenario,
            d(&p.isl),
            d(&p.osl),
            d(&p.tp),
            d(&p.ep),
            if p.dp_attn { "yes" } else { "-" },
            d(&p.spec),
            conc.join(" "),
            p.recipe
                .as_deref()
                .map(|r| format!("`{r}`"))
                .unwrap_or_else(|| "-".into()),
            p.other.join(" ")
        );
    }
    let mut caveats: Vec<String> = Vec::new();
    if pts.iter().any(|p| p.scenario != "fixed-seq-len") {
        caveats.push("oaka has a client for fixed-seq-len only; other scenarios are for reading".into());
    }
    if pts.iter().any(|p| p.dp_attn) {
        caveats.push("dp-attn points: not expressible in oaka yet (tp = the number of GPUs)".into());
    }
    if pts.iter().any(|p| p.spec.is_some()) {
        caveats.push(
            "speculative decoding: InferenceX sends chat-templated prompts for these; oaka's client does not".into(),
        );
    }
    caveats.push(format!(
        "the client is this box's own InferenceX checkout (each fixed-seq run logs its sha), not {}: \
         the configs come from this revision, the benchmark code may be newer",
        &repo.sha[..12.min(repo.sha.len())]
    ));
    m += "\n";
    for c in caveats {
        m += &format!("- {c}\n");
    }
    if let Some(d) = dash {
        m += &measured(d);
    }

    m += "\n## Files\n\n- `entry.yaml`: the entry as written in the config, comments included\n";
    if let Some(d) = dash {
        m += "- `dashboard/rows.json`: the run's rows as the dashboard's API sent them, every metric\n";
        for l in &d.logs {
            m += &format!("- `{}`: the server log of {}\n", l.file, l.label);
        }
    }
    for f in files {
        m += &format!("- `recipes/{}`", rel(&f.path));
        let facts = recipe_facts(f);
        if facts.is_empty() {
            m += "\n";
        } else {
            m += ":\n";
            for fact in facts {
                m += &format!("  - {fact}\n");
            }
        }
    }
    for n in notes {
        m += &format!("- missing: {n}\n");
    }
    m += "\n## For the worker\n\n\
          Follow the worker skill's \"Mirror workers\" section (`$AKAO_REPO_ROOT/skills/worker/SKILL.md`): turn the\n\
          recipe into an oaka profile named `<model>/infx-<entry>[-<variant>]` with this revision in its\n\
          `#` provenance, prove it with a one-point fixed-seq plan, then run the entry's points.\n\
          Report every point oaka cannot reproduce as it stands (the caveats above), rather than\n\
          bending the plan to fit.\n";
    if dash.is_some() {
        m += "\nThe points were measured: report each of yours next to the dashboard's number above.\n\
              Where the server log's arguments and the recipe disagree, the log is what ran.\n";
    }
    m
}

/// MIRROR.md's section of what the dashboard measured, and with which server arguments.
fn measured(d: &Evidence) -> String {
    let mut m = format!(
        "\n## Measured on the dashboard\n\nRun {} on {}.  `out tok/s` is the dashboard's output tok/s per GPU times the \
         GPUs, the number oaka's fixed-seq summary calls `output_tok_s`.\n\n",
        d.run.id, d.run.date
    );
    m += "| isl/osl | tp | ep | dp-attn | gpus | conc | out tok/s | out tok/s/gpu | median ttft ms | median tpot ms | id |\n";
    m += "|---|---|---|---|---|---|---|---|---|---|---|\n";
    let f = |v: Option<f64>, p: usize| v.map(|x| format!("{x:.p$}")).unwrap_or_else(|| "-".into());
    for p in &d.points {
        m += &format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            p.seq,
            p.tp,
            p.ep,
            if p.dp_attn { "yes" } else { "-" },
            p.gpus,
            p.conc,
            f(p.out_tok_s, 1),
            f(p.out_per_gpu, 1),
            f(p.median_ttft_ms, 1),
            f(p.median_tpot_ms, 2),
            p.id
        );
    }
    if !d.others.is_empty() {
        m += "\nOther runs on this curve (another sequence or image; `akao mirror --dashboard --run <id>`):\n\n";
        for o in &d.others {
            m += &format!("- {o}\n");
        }
    }
    m += "\n### Server arguments, from the stored logs\n\n";
    if d.logs.is_empty() {
        m += "The dashboard has no server log for these points.\n";
    }
    for l in &d.logs {
        match &l.args {
            Some(a) => m += &format!("{} (`{}`):\n\n```\n{a}\n```\n\n", l.label, l.file),
            None => m += &format!("{} (`{}`): no argument line found; read the log.\n\n", l.label, l.file),
        }
    }
    m
}

fn summarize(archs: &[String]) -> String {
    match archs {
        [] => "no GPUs in its KFD topology".into(),
        [a, rest @ ..] if rest.iter().all(|x| x == a) => format!("{} x {a}", archs.len()),
        _ => archs.join(","),
    }
}

/// gfx name from a KFD `gfx_target_version` (as oaka's sys::gfx_name).
fn gfx_name(v: u32) -> String {
    format!("gfx{}{:x}{:x}", v / 10000, (v / 100) % 100, v % 100)
}

/// The host's GPU archs from its KFD topology, over ssh (read-only).  Values are marked,
/// so that a login banner on stdout cannot read as a GPU.
fn host_archs(r: &Runner, nick: &str) -> Result<Vec<String>> {
    let script = "cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
                  | sed -n 's/^gfx_target_version /OAKA_GFX=/p'";
    let out = r.query(&r.ssh_argv(nick, script))?;
    Ok(out
        .lines()
        .filter_map(|l| l.trim().strip_prefix("OAKA_GFX=")?.parse::<u32>().ok())
        .filter(|v| *v != 0)
        .map(gfx_name)
        .collect())
}

fn listing(found: &[&Entry]) -> String {
    found
        .iter()
        .map(|e| {
            let problems = e.problems();
            format!(
                "  {:<44} {:<7} {:<8} {}{}",
                e.name,
                e.field("framework"),
                e.field("runner"),
                e.field("image"),
                if problems.is_empty() {
                    String::new()
                } else {
                    format!("\n  {:<44} cannot mirror: {}", "", problems.join("; "))
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// What a mirror is made of: the InferenceX tree, its entry, the image to run and, from the
/// dashboard, what was measured with it.
struct Source {
    repo: Repo,
    entry: Entry,
    image: String,
    dash: Option<Evidence>,
    /// The dashboard's snapshot of the tree, removed when the mirror is done.
    _scratch: Option<dashboard::Scratch>,
}

/// The entry `--conf` names in this machine's clone at `--rev`; None when only listed.
fn from_git(state: &State, opts: &Options) -> Result<Option<Source>> {
    let dir = state.require("infx_local")?;
    let rev = opts.rev.clone().unwrap_or_else(|| "HEAD".into());
    let repo = Repo::open(&dir, &rev)?;
    println!("InferenceX {dir} at {}", repo.title);
    let (mut all, notes) = entries(&repo)?;
    for n in &notes {
        println!("  note: {n}");
    }
    let found = select(&all, &opts.conf);
    let name = match found.as_slice() {
        [] => {
            let first = opts.conf.split(',').next().unwrap_or_default().trim();
            // Retired entries live in history: the commits that last changed the first term.
            let hint = git(&dir, &["log", "-3", "--format=%h %cs %s", "-S", first, &repo.sha, "--", "*master.yaml"])
                .ok()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let lines: Vec<String> = l.lines().map(|x| format!("    {}", x.chars().take(100).collect::<String>())).collect();
                    format!(
                        "\n  the latest commits that changed {first:?} in a master config (retired? try --rev <one of them>^, \
                         or --dashboard for what the dashboard measured):\n{}",
                        lines.join("\n")
                    )
                })
                .unwrap_or_default();
            bail!("no entry matches --conf {} at {}{hint}", opts.conf, repo.title);
        }
        [one, rest @ ..] if rest.is_empty() || rest[0].name.split('-').count() > one.name.split('-').count() => {
            if !rest.is_empty() {
                println!(
                    "  also matched (name another part to pick one of them):\n{}",
                    listing(rest)
                );
            }
            one.name.clone()
        }
        _ if opts.nick.is_none() => {
            println!(
                "{} entries match; add a term to preview one:\n{}",
                found.len(),
                listing(&found)
            );
            return Ok(None);
        }
        _ => bail!(
            "--conf {} matches {} entries; add a term:\n{}",
            opts.conf,
            found.len(),
            listing(&found)
        ),
    };
    let entry = all.swap_remove(all.iter().position(|e| e.name == name).unwrap());
    let image = entry.get("image").context("the entry has no image")?;
    Ok(Some(Source {
        repo,
        entry,
        image,
        dash: None,
        _scratch: None,
    }))
}

/// The dashboard curve `--conf` names, its run's commit and the entry that run measured;
/// None when only listed.
fn from_dashboard(state: &State, opts: &Options) -> Result<Option<Source>> {
    let ask = dashboard::Ask {
        conf: &opts.conf,
        date: opts.date.as_deref(),
        run: opts.run.as_deref(),
    };
    let Some(crawled) = dashboard::crawl(state, &ask)? else {
        if opts.nick.is_some() {
            bail!(
                "--conf {} matches several dashboard curves (listed above); add a term",
                opts.conf
            );
        }
        return Ok(None);
    };
    let (mut all, notes) = entries(&crawled.repo)?;
    for n in &notes {
        println!("  note: {n}");
    }
    let d = crawled.evidence;
    let c = &d.curve;
    let fits = |e: &Entry| {
        e.gpu().as_deref() == Some(c.hardware.as_str())
            && e.field("framework") == c.framework
            && e.field("precision") == c.precision
    };
    // The run's changelog names the entries it measured; without one, the curve's own.
    let mut pick: Vec<usize> = (0..all.len())
        .filter(|i| d.config_keys.contains(&all[*i].name) && fits(&all[*i]))
        .collect();
    if pick.is_empty() {
        let agentic = c.kind != "single_turn";
        pick = (0..all.len())
            .filter(|i| {
                let e = &all[*i];
                fits(e)
                    && norm(&c.model).starts_with(&norm(&e.field("model-prefix")))
                    && e.name.contains("agentic") == agentic
                    && (c.spec == "none") != e.name.contains(&format!("-{}", c.spec))
            })
            .collect();
    }
    let i = match pick.as_slice() {
        [i] => *i,
        [] => bail!(
            "run {}'s commit {} has no config entry for {} (its changelog names: {})",
            d.run.id,
            crawled.repo.sha,
            c.name(),
            if d.config_keys.is_empty() {
                "nothing".to_string()
            } else {
                d.config_keys.join(" ")
            }
        ),
        many => bail!(
            "run {}'s commit has several entries for {}: {}",
            d.run.id,
            c.name(),
            many.iter().map(|i| all[*i].name.as_str()).collect::<Vec<_>>().join(" ")
        ),
    };
    let entry = all.swap_remove(i);
    println!(
        "  entry {} in {} at {}",
        entry.name,
        entry.file,
        &crawled.repo.sha[..12.min(crawled.repo.sha.len())]
    );
    let image = match &d.run.image {
        Some(img) => img.clone(),
        None => entry
            .get("image")
            .context("neither the dashboard nor the entry names an image")?,
    };
    Ok(Some(Source {
        repo: crawled.repo,
        entry,
        image,
        dash: Some(d),
        _scratch: Some(crawled.scratch),
    }))
}

pub fn run(state: &State, r: &Runner, opts: &Options) -> Result<()> {
    let source = if opts.dashboard {
        from_dashboard(state, opts)?
    } else {
        from_git(state, opts)?
    };
    let Some(Source {
        repo,
        entry,
        image,
        dash,
        _scratch,
    }) = source
    else {
        return Ok(());
    };
    let e = &entry;
    let pts = points(e);
    let (files, missing) = recipe_files(&repo, e, &pts)?;

    let Some(nick) = &opts.nick else {
        print!("\n{}", brief(&repo, e, &pts, &files, &missing, None, dash.as_ref()));
        return Ok(());
    };
    let problems = e.problems();
    if !problems.is_empty() {
        bail!("cannot mirror {}: {}", e.name, problems.join("; "));
    }
    let name = opts.name.clone().unwrap_or_else(|| e.name.clone());
    // The box must have the runner's GPUs before anything changes on it.
    let want = e.arch().with_context(|| {
        format!(
            "akao does not know the GPU arch of runner {}; add it to Entry::arch in akao/src/mirror.rs",
            e.field("runner")
        )
    })?;
    let archs = host_archs(r, nick)?;
    if archs.is_empty() {
        bail!("{nick} shows no AMD GPUs (no KFD topology on the host); pick a box with {want}");
    }
    if archs.iter().any(|a| a != want) {
        bail!(
            "{nick} has {}, but {} runs on {} ({want}); pick a box with {want}",
            summarize(&archs),
            e.name,
            e.field("runner")
        );
    }
    println!("  {nick}: {} as {} needs", summarize(&archs), e.field("runner"));
    let p = init::run(
        state,
        r,
        &init::Options {
            nick: nick.clone(),
            name,
            week: opts.week.clone(),
            artifact_root: opts.artifact_root.clone(),
            image,
            skip_setup: opts.skip_setup,
        },
    )?;

    println!("[mirror] brief");
    let image_id = r
        .probe(&p.docker(&["container", "inspect", "--format", "{{.Image}}", &p.container]))?
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let target = Target {
        container: format!("{} (image id {image_id})", p.container),
        host: nick.clone(),
        host_archs: Some(archs),
    };
    let rel = brief_dir(e, &repo);
    let md = brief(&repo, e, &pts, &files, &missing, Some(&target), dash.as_ref());
    let local = write_brief(&md, e, &files, &rel, dash.as_ref())?;
    let dest = format!("{}{}", p.host.home(), p.artifact_root);
    let tar: Vec<String> = ["tar", "-C"]
        .iter()
        .map(|s| s.to_string())
        .chain([
            local.display().to_string(),
            "--owner=0".into(),
            "--group=0".into(),
            "-czf".into(),
            "-".into(),
            "mirror".into(),
        ])
        .collect();
    let extract = format!("{ESCALATE}$S mkdir -p {} && $S tar -xzf - -C {}", q(&dest), q(&dest));
    let shipped = r.pipe(&tar, &r.ssh_argv(nick, &extract));
    let _ = fs::remove_dir_all(&local);
    shipped?;
    let md = format!("{}/{rel}/MIRROR.md", p.artifact_root);
    println!(
        "done. the worker's brief: {md}\n  brief its agent (controller skill, \"The worker's agent\"), adding:\n  \
         \"You are a mirror worker: read {md} first.\""
    );
    Ok(())
}

/// The brief (MIRROR.md's text `md`) in a local scratch dir, as `<scratch>/<rel>/...`;
/// returns the scratch dir.
fn write_brief(md: &str, e: &Entry, files: &[Shipped], rel: &str, dash: Option<&Evidence>) -> Result<PathBuf> {
    let scratch = std::env::temp_dir().join(format!("akao-mirror-{}", std::process::id()));
    let _ = fs::remove_dir_all(&scratch);
    let dir = scratch.join(rel);
    let write = |path: &Path, text: &str| -> Result<()> {
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, text).with_context(|| format!("writing {}", path.display()))
    };
    write(&dir.join("MIRROR.md"), md)?;
    write(&dir.join("entry.yaml"), &e.raw)?;
    for f in files {
        let rel = f.path.strip_prefix(&e.root).unwrap_or(&f.path);
        write(&dir.join("recipes").join(rel), &f.text)?;
    }
    if let Some(d) = dash {
        write(&dir.join("dashboard/rows.json"), &d.rows_json)?;
        for l in &d.logs {
            write(&dir.join(&l.file), &l.text)?;
        }
    }
    Ok(scratch)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER: &str = r#"# AMD entries
m1-fp4-mi355x-vllm:
  image: vllm/vllm:1
  model: org/M1-120B
  model-prefix: m1
  runner: mi355x
  precision: fp4
  framework: vllm
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - { tp: 1, conc-start: 4, conc-end: 32 }
      - { tp: 8, ep: 1, conc-start: 4, conc-end: 4 }

# WIP framework
m1-fp4-mi355x-atom:
  image: rocm/atom:1
  model: org/M1-120B
  model-prefix: m1
  runner: mi355x
  precision: fp4
  framework: atom   # no customers yet
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 8192
      osl: 1024
      search-space:
      - { tp: 8, dp-attn: true, conc-list: [4, 64] }

m1-fp4-mi355x-atom-mtp:
  image: rocm/atom:1
  model: org/M1-120B
  model-prefix: m1
  runner: mi355x
  precision: fp4
  framework: atom
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - { tp: 8, spec-decoding: mtp, conc-start: 4, conc-end: 8 }

m2-fp8-mi355x-sglang:
  image: lmsysorg/sglang:1
  model: org/M2
  model-prefix: m2
  runner: cluster:mi355x-amds
  precision: fp8
  framework: sglang
  scenarios:
    agentic-coding:
    - search-space:
      - { tp: 4, kv-offloading: none, conc-list: [1, 4], srt-recipe: benchmarks/single_node/srt-slurm-recipes/m2/sglang/agentic.yaml }

m1-fp4-b200-trt:
  image: nvcr/trt:1
  model: org/M1-120B
  model-prefix: m1
  runner: b200
  precision: fp4
  framework: trt
  scenarios: {}

m2-fp8-mi355x-sglang-disagg:
  image: lmsysorg/sglang:1
  model: org/M2
  model-prefix: m2
  runner: mi355x
  precision: fp8
  framework: sglang
  multinode: true
  disagg: true
  scenarios: {}
"#;

    const RECIPE: &str = "base:\n  setup_script: deps.sh\n  health_check:\n    max_attempts: 360\n  roles:\n    agg:\n      \
                          args:\n        data-parallel-size: 1\n        trust-remote-code: true\n  benchmark:\n    \
                          command: bash srt_fixed_sequence.sh '--trust-remote-code'\n    env:\n      USE_CHAT_TEMPLATE: 'true'\n      \
                          RANDOM_RANGE_RATIO: '0.8'\nzip_override_tp4:\n  roles:\n    agg:\n      args:\n        \
                          data-parallel-size: [1, 8]\n  benchmark:\n    env:\n      CONC: ['1', '4']\n";

    const VLLM_SH: &str = "vllm serve $MODEL --port $PORT\nrun_benchmark_serving \\\n    --model \"$MODEL\" \\\n    \
                           --trust-remote-code \\\n    --result-dir /workspace/\n";

    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        /// Two commits: the old layout (configs/ at the root, scripts under
        /// benchmarks/single_node/fixed_seq_len/), then everything under inferencex-e2e/
        /// with the config archived under configs/deprecated/.
        fn new(name: &str) -> (Fixture, String, String) {
            let dir = std::env::temp_dir().join(format!("akao-mirror-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let f = Fixture { dir };
            let put = |rel: &str, text: &str| {
                let p = f.dir.join(rel);
                fs::create_dir_all(p.parent().unwrap()).unwrap();
                fs::write(p, text).unwrap();
            };
            put("configs/amd-master.yaml", MASTER);
            put("configs/runners.yaml", "labels: {}\n");
            put(
                "benchmarks/benchmark_lib.sh",
                "run_benchmark_serving() { :; }  # --use-chat-template\n",
            );
            put("benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x.sh", VLLM_SH);
            put(
                "benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x_atom.sh",
                "python3 -m atom.entrypoints.openai_server\n",
            );
            // Another scenario's script of the same name must never be picked for fixed-seq.
            put("benchmarks/single_node/agentic/m1_fp4_mi355x_atom.sh", "agentic\n");
            // Nor a multi-node script, even with the framework's name (the vllm entry's
            // default-named single-node script must win over it).
            put("benchmarks/multi_node/m1_fp4_mi355x_vllm.sh", "multi node\n");
            put(
                "benchmarks/single_node/srt-slurm-recipes/m2/sglang/agentic.yaml",
                RECIPE,
            );
            put(
                "benchmarks/multi_node/srt-slurm-recipes/configs/deps.sh",
                "pip install x\n",
            );
            f.git(&["init", "-q"]);
            f.git(&["add", "-A"]);
            f.git(&["commit", "-q", "-m", "old layout"]);
            let old = f.git(&["rev-parse", "HEAD"]);
            f.git(&["mv", "configs", "x"]);
            f.git(&["mv", "benchmarks", "y"]);
            fs::create_dir_all(f.dir.join("inferencex-e2e/configs")).unwrap();
            f.git(&["mv", "x/amd-master.yaml", "x/deprecated.yaml"]);
            f.git(&["mv", "x", "inferencex-e2e/configs/deprecated"]);
            f.git(&[
                "mv",
                "inferencex-e2e/configs/deprecated/deprecated.yaml",
                "inferencex-e2e/configs/deprecated/amd-master.yaml",
            ]);
            f.git(&["mv", "y", "inferencex-e2e/benchmarks"]);
            f.git(&["commit", "-q", "-m", "e2e layout"]);
            let new = f.git(&["rev-parse", "HEAD"]);
            (f, old.trim().to_string(), new.trim().to_string())
        }

        fn git(&self, args: &[&str]) -> String {
            let out = Command::new("git")
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "init.defaultBranch=main",
                ])
                .args(args)
                .current_dir(&self.dir)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).into_owned()
        }

        fn repo(&self, rev: &str) -> Repo {
            Repo::open(&self.dir.display().to_string(), rev).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn names(found: &[&Entry]) -> Vec<String> {
        found.iter().map(|e| e.name.clone()).collect()
    }

    #[test]
    fn entries_select_and_points() {
        let (f, old, _) = Fixture::new("select");
        let repo = f.repo(&old);
        let (all, notes) = entries(&repo).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(all.len(), 6, "runners.yaml is not a master config");
        // The base entry before its -mtp variant; terms are prefixes, '-' and case ignored.
        assert_eq!(
            names(&select(&all, "M1, mi355 ,atom")),
            ["m1-fp4-mi355x-atom", "m1-fp4-mi355x-atom-mtp"]
        );
        assert_eq!(
            names(&select(&all, "m1-120b,trt")),
            ["m1-fp4-b200-trt"],
            "the model's basename"
        );
        assert_eq!(
            names(&select(&all, "m1-fp4-mi355x-atom-mtp")),
            ["m1-fp4-mi355x-atom-mtp"]
        );
        assert!(select(&all, "m1,mi300").is_empty());
        assert!(select(&all, " , ").is_empty());

        let by = |n: &str| all.iter().find(|e| e.name == n).unwrap();
        let atom = by("m1-fp4-mi355x-atom");
        assert!(atom.raw.starts_with("m1-fp4-mi355x-atom:\n"), "{}", atom.raw);
        assert!(
            atom.raw.contains("framework: atom   # no customers yet"),
            "comments kept"
        );
        assert!(
            !by("m1-fp4-mi355x-vllm").raw.contains("WIP framework"),
            "the next entry's comment"
        );
        assert!(atom.problems().is_empty());
        assert_eq!(atom.arch(), Some("gfx950"));
        assert_eq!(
            by("m2-fp8-mi355x-sglang").gpu().as_deref(),
            Some("mi355x"),
            "cluster:<gpu>-<site>"
        );
        assert_eq!(
            by("m1-fp4-b200-trt").problems().len(),
            2,
            "{:?}",
            by("m1-fp4-b200-trt").problems()
        );
        assert!(by("m2-fp8-mi355x-sglang-disagg").problems()[0].contains("multinode"));

        let pts = points(by("m1-fp4-mi355x-vllm"));
        assert_eq!(pts[0].conc, [4, 8, 16, 32]);
        let range = |a: u64, b: u64| {
            let mut m = Mapping::new();
            m.insert("conc-start".into(), a.into());
            m.insert("conc-end".into(), b.into());
            concurrencies(&m)
        };
        assert_eq!(range(4, 12), [4, 8, 12], "InferenceX ends on conc-end after overshoot");
        assert_eq!(range(4, 4), [4]);
        assert!(range(8, 4).is_empty() && range(0, 4).is_empty());
        assert_eq!(range(u64::MAX - 1, u64::MAX), [u64::MAX - 1, u64::MAX]);
        assert_eq!(pts[1].conc, [4]);
        assert_eq!(pts[1].ep.as_deref(), Some("1"));
        let pts = points(atom);
        assert_eq!((pts[0].conc.clone(), pts[0].dp_attn), (vec![4, 64], true));
        let pts = points(by("m2-fp8-mi355x-sglang"));
        assert_eq!(pts[0].scenario, "agentic-coding");
        assert_eq!(pts[0].other, ["kv-offloading=none"]);
    }

    #[test]
    fn recipes_follow_the_launcher_and_the_layout() {
        let (f, old, new) = Fixture::new("recipes");
        for (rev, root, sub) in [(&old, "", ""), (&new, "inferencex-e2e/", "")] {
            let repo = f.repo(rev);
            let (all, _) = entries(&repo).unwrap();
            let by = |n: &str| all.iter().find(|e| e.name == n).unwrap();
            let shipped = |n: &str| {
                let e = by(n);
                assert_eq!(e.root, root);
                let (files, notes) = recipe_files(&repo, e, &points(e)).unwrap();
                (files.iter().map(|f| f.path.clone()).collect::<Vec<_>>(), notes)
            };
            let bench = format!("{root}benchmarks/{sub}");
            // ATOM's own script first; vllm falls back to the GPU's default script.
            assert_eq!(
                shipped("m1-fp4-mi355x-atom").0,
                [
                    format!("{bench}single_node/fixed_seq_len/m1_fp4_mi355x_atom.sh"),
                    format!("{bench}benchmark_lib.sh")
                ]
            );
            assert_eq!(
                shipped("m1-fp4-mi355x-vllm").0[0],
                format!("{bench}single_node/fixed_seq_len/m1_fp4_mi355x.sh")
            );
            // An srt recipe brings the setup script it names, from wherever it lives.
            assert_eq!(
                shipped("m2-fp8-mi355x-sglang").0,
                [
                    format!("{bench}single_node/srt-slurm-recipes/m2/sglang/agentic.yaml"),
                    format!("{bench}multi_node/srt-slurm-recipes/configs/deps.sh")
                ]
            );
            let (files, notes) = shipped("m1-fp4-mi355x-atom-mtp");
            assert!(files.is_empty());
            assert!(
                notes[0].contains("looked for m1_fp4_mi355x_atom_mtp.sh and m1_fp4_mi355x_mtp.sh"),
                "{notes:?}"
            );
        }
        let repo = f.repo(&new);
        let (all, _) = entries(&repo).unwrap();
        assert_eq!(all[0].file, "inferencex-e2e/configs/deprecated/amd-master.yaml");
    }

    #[test]
    fn configs_under_github_root_at_the_top() {
        assert_eq!(entry_root(""), "");
        assert_eq!(entry_root("inferencex-e2e/"), "inferencex-e2e/");
        assert_eq!(entry_root(".github/"), "", "mid-2026: benchmarks/ beside .github/");
    }

    #[test]
    fn a_snapshot_reads_like_the_clone() {
        let dir = std::env::temp_dir().join(format!("akao-snapshot-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let put = |rel: &str, text: &str| {
            let p = dir.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, text).unwrap();
        };
        put(".github/configs/amd-master.yaml", MASTER);
        put("benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x.sh", VLLM_SH);
        put("benchmarks/benchmark_lib.sh", "run_benchmark_serving() { :; }\n");
        let repo = Repo::snapshot(&dir, "abc1234def", "t".into()).unwrap();
        assert_eq!(repo.files[0], ".github/configs/amd-master.yaml");
        let (all, _) = entries(&repo).unwrap();
        let e = all.iter().find(|e| e.name == "m1-fp4-mi355x-vllm").unwrap();
        assert_eq!(e.root, "");
        let (files, notes) = recipe_files(&repo, e, &points(e)).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(files[0].path, "benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x.sh");
        assert_eq!(files[0].text, VLLM_SH);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_brief_says_what_oaka_cannot_reproduce() {
        let (f, old, _) = Fixture::new("brief");
        let repo = f.repo(&old);
        let (all, _) = entries(&repo).unwrap();
        let brief_of = |n: &str| {
            let e = all.iter().find(|e| e.name == n).unwrap();
            let pts = points(e);
            let (files, notes) = recipe_files(&repo, e, &pts).unwrap();
            brief(&repo, e, &pts, &files, &notes, None, None)
        };
        let vllm = brief_of("m1-fp4-mi355x-vllm");
        assert!(vllm.contains(&format!("| InferenceX | `{old}`")), "{vllm}");
        assert!(vllm.contains("vllm → oaka engine `vllm`"), "{vllm}");
        assert!(
            vllm.contains("| fixed-seq-len | 1024 | 1024 | 1 | - | - | - | 4 8 16 32 | - |  |"),
            "{vllm}"
        );
        assert!(vllm.contains("client flag `--trust-remote-code`"), "{vllm}");
        assert!(
            !vllm.contains("--use-chat-template"),
            "benchmark_lib.sh only defines it: {vllm}"
        );
        let atom = brief_of("m1-fp4-mi355x-atom");
        assert!(atom.contains("dp-attn points: not expressible in oaka yet"), "{atom}");
        let sglang = brief_of("m2-fp8-mi355x-sglang");
        for want in [
            "USE_CHAT_TEMPLATE = true (oaka's fixed-seq client sends no chat template: not comparable)",
            "per-point server settings in zip_override_tp4",
            "setup script deps.sh (shipped; never run by oaka)",
            "health_check max_attempts=360",
            "client flag `--trust-remote-code` in benchmark.command (oaka's fixed-seq client cannot pass it",
            "data-parallel-size > 1: not expressible in oaka yet",
            "oaka has a client for fixed-seq-len only",
        ] {
            assert!(sglang.contains(want), "{want:?} not in {sglang}");
        }
        let trt = brief_of("m1-fp4-b200-trt");
        assert!(
            trt.contains("| cannot mirror | multinode") || trt.contains("| cannot mirror | framework trt"),
            "{trt}"
        );
        assert_eq!(
            brief_dir(all.iter().find(|e| e.name == "m1-fp4-b200-trt").unwrap(), &repo),
            format!("mirror/m1-fp4-b200-trt@{}", &old[..12])
        );
    }
}
