//! The InferenceX dashboard as a mirror's source: what it measured, rather than what a
//! config said at some revision of this machine's clone.
//!
//! Its public API (config `infx_dashboard`, `/api/openapi.json`) lists the curves it shows
//! (`availability`: model, GPU, framework, precision, speculative decoding, scenario), their
//! measured rows (`benchmarks`), the GitHub Actions run behind each row and that run's
//! InferenceX commit (`workflow-info`), and the server log each row was measured with
//! (`server-log`), whose arguments are the ones the server really ran with.  The commit's
//! tree comes from GitHub as a tarball (config `infx_snapshot`); it is often a PR branch's
//! commit that no clone of main has.  Reads go through `curl` and are printed, not run
//! through `Runner`: read-only, and they run under --dry-run too.

use crate::mirror::Repo;
use crate::state::State;
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// `curl` of one URL: the body, or an error with curl's message.
fn curl(url: &str) -> Result<Vec<u8>> {
    println!("  ? GET {url}");
    let out = Command::new("curl")
        .args(["-sS", "-f", "-L", "--compressed", "-m", "300", "-A", "akao-mirror", url])
        .stdin(Stdio::null())
        .output()
        .context("cannot run curl; akao mirror --dashboard reads the dashboard with it (akao doctor)")?;
    if !out.status.success() {
        bail!("GET {url}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(out.stdout)
}

/// Percent-encode a query value (RFC 3986 unreserved characters pass).
fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

pub struct Client {
    base: String,
}

impl Client {
    pub fn new(base: &str) -> Client {
        Client {
            base: base.trim_end_matches('/').to_string(),
        }
    }

    fn url(&self, path: &str, query: &[(&str, &str)]) -> String {
        let q: Vec<String> = query.iter().map(|(k, v)| format!("{k}={}", enc(v))).collect();
        match q.is_empty() {
            true => format!("{}{path}", self.base),
            false => format!("{}{path}?{}", self.base, q.join("&")),
        }
    }

    fn json(&self, path: &str, query: &[(&str, &str)]) -> Result<Value> {
        let url = self.url(path, query);
        serde_json::from_slice(&curl(&url)?).with_context(|| format!("GET {url}: not JSON"))
    }

    fn text(&self, path: &str, query: &[(&str, &str)]) -> Result<String> {
        Ok(String::from_utf8_lossy(&curl(&self.url(path, query))?).into_owned())
    }
}

/// A field as text, whether the API sent a string or a number ("26998598877" or 1024).
fn s(v: &Value, key: &str) -> Option<String> {
    match v.get(key)? {
        Value::String(x) => Some(x.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// A count or id, sent as a number or as a numeric string (benchmark row ids are strings).
fn n(v: &Value, key: &str) -> Option<u64> {
    match v.get(key)? {
        Value::String(x) => x.parse().ok(),
        x => x.as_u64(),
    }
}

/// Lowercase without `-_.`, as mirror::norm.
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '-' | '_' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

/// One line of the dashboard's charts: a model on a GPU with a framework, precision,
/// speculative decoding and scenario.  Its sequences (ISL/OSL) are points along it.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    /// The dashboard's model key, e.g. "gptoss120b" (its display name is in views/options).
    pub model: String,
    pub hardware: String,
    pub framework: String,
    pub precision: String,
    pub spec: String,
    pub disagg: bool,
    /// benchmark_type: "single_turn", "agentic_traces", ...
    pub kind: String,
    /// "1024/1024" ..., or "-" for scenarios without fixed sequences.
    pub seqs: Vec<String>,
    /// The latest date it has data for.
    pub last: String,
}

impl Curve {
    /// Named like InferenceX entries: gptoss120b-fp4-mi355x-vllm[-mtp][-agentic][-disagg].
    pub fn name(&self) -> String {
        let mut name = format!("{}-{}-{}-{}", self.model, self.precision, self.hardware, self.framework);
        if self.spec != "none" {
            name += &format!("-{}", self.spec.replace('_', "-"));
        }
        match self.kind.as_str() {
            "single_turn" => {}
            "agentic_traces" => name += "-agentic",
            k => name += &format!("-{}", k.replace('_', "-")),
        }
        if self.disagg && !self.framework.ends_with("disagg") {
            name += "-disagg";
        }
        name
    }

    /// Why one container with oaka cannot serve it, as mirror's Entry::problems; empty when it can.
    pub fn problems(&self) -> Vec<String> {
        let mut p = Vec::new();
        if self.disagg {
            p.push("disaggregated: one container cannot hold it".to_string());
        }
        if !crate::mirror::ENGINES.contains(&self.framework.as_str()) {
            p.push(format!(
                "framework {}: oaka serves only {}",
                self.framework,
                crate::mirror::ENGINES.join(", ")
            ));
        }
        if !self.hardware.starts_with("mi") {
            p.push(format!("{}: not an AMD Instinct GPU", self.hardware));
        }
        p
    }

    fn matches(&self, terms: &[String], display: &str) -> bool {
        let name = self.name();
        let mut tokens: Vec<String> = name.split('-').map(norm).collect();
        tokens.extend([norm(display), norm(&self.kind)]);
        terms
            .iter()
            .all(|t| norm(t) == norm(&name) || tokens.iter().any(|x| x.starts_with(&norm(t))))
    }

    /// Whether a benchmark row is a point of this curve.
    fn has(&self, row: &Value) -> bool {
        let eq = |k: &str, want: &str| s(row, k).as_deref() == Some(want);
        eq("model", &self.model)
            && eq("hardware", &self.hardware)
            && eq("framework", &self.framework)
            && eq("precision", &self.precision)
            && eq("spec_method", &self.spec)
            && eq("benchmark_type", &self.kind)
            && row.get("disagg").and_then(Value::as_bool) == Some(self.disagg)
    }
}

/// The curves of `availability` rows, sorted by name.
pub fn curves(availability: &Value) -> Vec<Curve> {
    let mut by: BTreeMap<String, Curve> = BTreeMap::new();
    for r in availability.as_array().into_iter().flatten() {
        let get = |k: &str| s(r, k).unwrap_or_default();
        let c = Curve {
            model: get("model"),
            hardware: get("hardware"),
            framework: get("framework"),
            precision: get("precision"),
            spec: get("spec_method"),
            disagg: r.get("disagg").and_then(Value::as_bool).unwrap_or(false),
            kind: get("benchmark_type"),
            seqs: Vec::new(),
            last: String::new(),
        };
        let seq = match (n(r, "isl"), n(r, "osl")) {
            (Some(i), Some(o)) => format!("{i}/{o}"),
            _ => "-".into(),
        };
        let date = get("date");
        let c = by.entry(c.name()).or_insert(c);
        if !c.seqs.contains(&seq) {
            c.seqs.push(seq);
        }
        if date > c.last {
            c.last = date;
        }
    }
    by.into_values().collect()
}

/// The curves `--conf` names: an exact curve name alone, else every curve each term
/// prefix-matches (name parts, the display name, the scenario), fewest name parts first.
pub fn select<'a>(all: &'a [Curve], conf: &str, display: &dyn Fn(&str) -> String) -> Vec<&'a Curve> {
    let terms: Vec<String> = conf
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if terms.is_empty() {
        return Vec::new();
    }
    if terms.len() == 1 {
        let exact: Vec<&Curve> = all.iter().filter(|c| c.name() == terms[0]).collect();
        if !exact.is_empty() {
            return exact;
        }
    }
    let mut found: Vec<&Curve> = all.iter().filter(|c| c.matches(&terms, &display(&c.model))).collect();
    found.sort_by_key(|c| (c.name().split('-').count(), c.name()));
    found
}

/// The rows of one GitHub Actions run of a curve: one image, one InferenceX commit.
#[derive(Debug)]
pub struct Run {
    pub id: String,
    pub attempt: String,
    pub url: String,
    pub date: String,
    pub started: String,
    pub image: Option<String>,
    pub rows: Vec<Value>,
}

/// "26998598877" and "1" from ".../actions/runs/26998598877/attempts/1".
fn run_of(url: &str) -> Option<(String, String)> {
    let rest = url.split("/actions/runs/").nth(1)?;
    let mut parts = rest.split('/');
    let id = parts
        .next()
        .filter(|i| !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit()))?;
    let attempt = match (parts.next(), parts.next()) {
        (Some("attempts"), Some(a)) => a.to_string(),
        _ => "1".to_string(),
    };
    Some((id.to_string(), attempt))
}

/// A curve's rows grouped by run, newest first.  The dashboard shows each sequence's latest
/// run, so one curve may hold several runs (and images); a container serves one of them.
pub fn runs(curve: &Curve, rows: &Value) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for r in rows.as_array().into_iter().flatten().filter(|r| curve.has(r)) {
        let url = s(r, "run_url").unwrap_or_default();
        let (id, attempt) = run_of(&url).unwrap_or_else(|| ("?".into(), "?".into()));
        let image = s(r, "image");
        match out.iter_mut().find(|x| x.id == id && x.image == image) {
            Some(run) => run.rows.push(r.clone()),
            None => out.push(Run {
                id,
                attempt,
                url,
                date: s(r, "date").unwrap_or_default(),
                started: s(r, "curve_run_started_at").unwrap_or_default(),
                image,
                rows: vec![r.clone()],
            }),
        }
    }
    for run in &mut out {
        run.rows.sort_by_key(|r| {
            (
                n(r, "isl"),
                n(r, "osl"),
                n(r, "decode_tp"),
                n(r, "decode_ep"),
                n(r, "conc"),
                n(r, "id"),
            )
        });
    }
    out.sort_by(|a, b| (&b.date, &b.started).cmp(&(&a.date, &a.started)));
    out
}

fn seq_of(r: &Value) -> String {
    match (n(r, "isl"), n(r, "osl")) {
        (Some(i), Some(o)) => format!("{i}/{o}"),
        _ => "-".into(),
    }
}

impl Run {
    pub fn summary(&self) -> String {
        let mut seqs: Vec<String> = self.rows.iter().map(seq_of).collect();
        seqs.dedup();
        format!(
            "run {} ({}, image {}): {} points, {}",
            self.id,
            self.date,
            self.image.as_deref().unwrap_or("?"),
            self.rows.len(),
            seqs.join(" ")
        )
    }
}

/// One measured point, for MIRROR.md.
#[derive(Debug)]
pub struct Measured {
    pub id: u64,
    pub seq: String,
    pub tp: String,
    pub ep: String,
    pub dp_attn: bool,
    pub gpus: u64,
    pub conc: u64,
    /// output_tput_per_gpu x gpus: what oaka's fixed-seq summary calls output_tok_s.
    pub out_tok_s: Option<f64>,
    pub out_per_gpu: Option<f64>,
    pub median_ttft_ms: Option<f64>,
    pub median_tpot_ms: Option<f64>,
}

impl Measured {
    fn of(r: &Value) -> Measured {
        let m = |k: &str| r.get("metrics").and_then(|m| m.get(k)).and_then(Value::as_f64);
        let gpus = n(r, "num_decode_gpu").or_else(|| n(r, "decode_tp")).unwrap_or(1);
        Measured {
            id: n(r, "id").unwrap_or(0),
            seq: seq_of(r),
            tp: s(r, "decode_tp").unwrap_or_else(|| "?".into()),
            ep: s(r, "decode_ep").unwrap_or_else(|| "?".into()),
            dp_attn: r.get("decode_dp_attention").and_then(Value::as_bool).unwrap_or(false),
            gpus,
            conc: n(r, "conc").unwrap_or(0),
            out_tok_s: m("output_tput_per_gpu").map(|t| t * gpus as f64),
            out_per_gpu: m("output_tput_per_gpu"),
            median_ttft_ms: m("median_ttft").map(|t| t * 1000.0),
            median_tpot_ms: m("median_tpot").map(|t| t * 1000.0),
        }
    }

    /// What starts a separate server: the same sequence, parallelism and GPUs.
    fn server_key(&self) -> (String, String, String, bool, u64) {
        (
            self.seq.clone(),
            self.tp.clone(),
            self.ep.clone(),
            self.dp_attn,
            self.gpus,
        )
    }
}

/// A stored server log, one per distinct server setting (its lowest concurrency).
pub struct Log {
    /// Brief-relative, e.g. dashboard/server_1024-1024_tp1_c4_419963.log
    pub file: String,
    /// "1024/1024 tp 1 ep 1"
    pub label: String,
    pub text: String,
    /// The line where the server prints its own arguments, if found.
    pub args: Option<String>,
}

/// vllm's `non-default args: {...}`, SGLang's `server_args=ServerArgs(...)`, ATOM's engine
/// args: the line that says how the server was really started.
pub fn server_args(log: &str) -> Option<String> {
    ["non-default args:", "server_args=", "engine args:"]
        .iter()
        .find_map(|mark| {
            log.lines()
                .find_map(|l| l.find(mark).map(|i| l[i..].trim_end_matches('\r').to_string()))
        })
}

/// Everything the brief needs from the dashboard.
pub struct Evidence {
    pub base: String,
    pub curve: Curve,
    pub display: String,
    pub run: Run,
    pub run_name: String,
    pub pr: Option<String>,
    pub description: Option<String>,
    pub config_keys: Vec<String>,
    pub points: Vec<Measured>,
    pub logs: Vec<Log>,
    /// The run's rows as the API sent them, for the worker to compare with.
    pub rows_json: String,
    /// Other runs on the curve (other sequences or images), one line each.
    pub others: Vec<String>,
}

/// An extracted snapshot, removed when dropped.
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// What the crawl found: the commit's tree, and the evidence.
pub struct Crawled {
    pub repo: Repo,
    pub evidence: Evidence,
    pub scratch: Scratch,
}

pub struct Ask<'a> {
    pub conf: &'a str,
    /// As-of date (YYYY-MM-DD) for the dashboard's snapshot; None = latest.
    pub date: Option<&'a str>,
    /// A GitHub Actions run id among the curve's; None = its latest.
    pub run: Option<&'a str>,
}

/// Find the curve, its run and that run's commit; None when only a listing was printed.
pub fn crawl(state: &State, ask: &Ask) -> Result<Option<Crawled>> {
    let base = state.require("infx_dashboard")?;
    let api = Client::new(&base);
    println!("InferenceX dashboard {base}");
    let options = api.json("/api/v1/views/options", &[])?;
    let names: Vec<(String, Vec<String>)> = options
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let keys = m
                .get("dbKeys")?
                .as_array()?
                .iter()
                .filter_map(|k| k.as_str().map(String::from));
            Some((s(m, "name")?, keys.collect()))
        })
        .collect();
    let display = |key: &str| {
        names
            .iter()
            .find(|(_, keys)| keys.iter().any(|k| k == key))
            .map(|(name, _)| name.clone())
            .unwrap_or_default()
    };
    let all = curves(&api.json("/api/v1/availability", &[])?);
    let found = select(&all, ask.conf, &display);
    let listing = |cs: &[&Curve]| {
        cs.iter()
            .map(|c| {
                let problems = c.problems();
                format!(
                    "  {:<44} {:<22} latest {}  {}{}",
                    c.name(),
                    display(&c.model),
                    c.last,
                    c.seqs.join(" "),
                    if problems.is_empty() {
                        String::new()
                    } else {
                        format!("\n  {:<44} cannot mirror: {}", "", problems.join("; "))
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let parts = |c: &Curve| c.name().split('-').count();
    let curve = match found.as_slice() {
        [] => bail!(
            "no dashboard curve matches --conf {} (terms match parts of names like gptoss120b-fp4-mi355x-vllm, \
             or the display model)",
            ask.conf
        ),
        [one, rest @ ..] if rest.is_empty() || parts(rest[0]) > parts(one) => {
            if !rest.is_empty() {
                println!(
                    "  also matched (name another part to pick one of them):\n{}",
                    listing(rest)
                );
            }
            *one
        }
        _ => {
            println!(
                "{} dashboard curves match; add a term (or give a full name):\n{}",
                found.len(),
                listing(&found)
            );
            return Ok(None);
        }
    };
    let model = display(&curve.model);
    if model.is_empty() {
        bail!(
            "the dashboard lists no display model for {} (views/options)",
            curve.model
        );
    }
    println!("  curve {} ({model}), latest {}", curve.name(), curve.last);

    let mut query = vec![("model", model.as_str())];
    if let Some(d) = ask.date {
        query.push(("date", d));
    }
    let rows = api.json("/api/v1/benchmarks", &query)?;
    let mut runs = runs(curve, &rows);
    if runs.is_empty() {
        bail!(
            "the dashboard has no rows of {} for {model}{}",
            curve.name(),
            ask.date.map(|d| format!(" on or before {d}")).unwrap_or_default()
        );
    }
    let i = match ask.run {
        Some(id) => runs.iter().position(|r| r.id == id).with_context(|| {
            let have: Vec<String> = runs.iter().map(|r| format!("  {}", r.summary())).collect();
            format!("run {id} is not among {}'s runs:\n{}", curve.name(), have.join("\n"))
        })?,
        None => 0,
    };
    let run = runs.remove(i);
    println!("  {}", run.summary());
    for o in &runs {
        println!("  also on the dashboard: {} (--run {} for it)", o.summary(), o.id);
    }
    let others: Vec<String> = runs.iter().map(Run::summary).collect();

    // The run's InferenceX commit and the change it measured.
    let wf = api.json("/api/v1/workflow-info", &[("date", run.date.as_str())])?;
    let list = |k: &str| wf.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
    let head = list("runConfigs")
        .iter()
        .filter(|c| s(c, "github_run_id").as_deref() == Some(run.id.as_str()))
        .find(|c| s(c, "hardware").as_deref() == Some(curve.hardware.as_str()) || s(c, "hardware").is_none())
        .and_then(|c| s(c, "head_sha"))
        .with_context(|| {
            format!(
                "the dashboard names no InferenceX commit for run {} (workflow-info?date={}); see {}",
                run.id, run.date, run.url
            )
        })?;
    if head.len() < 7 || !head.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("run {}: {head:?} is not a commit", run.id);
    }
    let run_name = list("runs")
        .iter()
        .find(|r| s(r, "github_run_id").as_deref() == Some(run.id.as_str()))
        .and_then(|r| s(r, "name"))
        .unwrap_or_default();
    let change = list("changelogs")
        .into_iter()
        .find(|c| s(c, "workflow_run_id").as_deref() == Some(run.id.as_str()));
    let config_keys: Vec<String> = change
        .as_ref()
        .and_then(|c| c.get("config_keys"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|k| k.as_str().map(String::from))
        .collect();

    // The commit's tree, from GitHub: no clone, no checkout.
    let url = state.require("infx_snapshot")?.replace("{sha}", &head);
    let dir = std::env::temp_dir().join(format!(
        "akao-snapshot-{}-{}",
        std::process::id(),
        &head[..12.min(head.len())]
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir)?;
    let scratch = Scratch(dir.clone());
    untar(&curl(&url)?, &dir).with_context(|| format!("extracting {url}"))?;
    let title = format!(
        "{} run {} attempt {} on {}: {run_name}",
        &head[..9.min(head.len())],
        run.id,
        run.attempt,
        run.date
    );
    let repo = Repo::snapshot(&dir, &head, title)?;

    // Every point, and one stored server log per distinct server.
    let points: Vec<Measured> = run.rows.iter().map(Measured::of).collect();
    let mut logs = Vec::new();
    let mut seen = Vec::new();
    for p in &points {
        if seen.contains(&p.server_key()) || p.id == 0 {
            continue;
        }
        seen.push(p.server_key());
        let id = p.id.to_string();
        let files = api
            .json("/api/v1/server-log-files", &[("id", &id)])
            .unwrap_or(Value::Null);
        let names: Vec<&str> = files
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let Some(file) = names.iter().find(|f| f.ends_with("server.log")).or(names.first()) else {
            continue;
        };
        let text = api.text("/api/v1/server-log", &[("id", &id), ("file", file), ("download", "1")])?;
        let ep = if p.ep != "1" {
            format!("_ep{}", p.ep)
        } else {
            String::new()
        };
        logs.push(Log {
            file: format!(
                "dashboard/server_{}_tp{}{ep}{}_c{}_{id}.log",
                p.seq.replace('/', "-"),
                p.tp,
                if p.dp_attn { "_dpattn" } else { "" },
                p.conc
            ),
            label: format!(
                "{} tp {} ep {}{}",
                p.seq,
                p.tp,
                p.ep,
                if p.dp_attn { " dp-attn" } else { "" }
            ),
            args: server_args(&text),
            text,
        });
    }
    let evidence = Evidence {
        base,
        curve: curve.clone(),
        display: model.clone(),
        rows_json: serde_json::to_string_pretty(&run.rows)? + "\n",
        run,
        run_name,
        pr: change.as_ref().and_then(|c| s(c, "pr_link")),
        description: change.as_ref().and_then(|c| s(c, "description")),
        config_keys,
        points,
        logs,
        others,
    };
    Ok(Some(Crawled {
        repo,
        evidence,
        scratch,
    }))
}

/// `tar -xz` of a GitHub tarball into `dir`, without its top-level `<repo>-<sha>/`.
fn untar(tgz: &[u8], dir: &Path) -> Result<()> {
    let mut tar = Command::new("tar")
        .args(["-xzf", "-", "--strip-components=1", "-C"])
        .arg(dir)
        .stdin(Stdio::piped())
        .spawn()
        .context("cannot run tar")?;
    tar.stdin.take().unwrap().write_all(tgz)?;
    if !tar.wait()?.success() {
        bail!("tar failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn avail() -> Value {
        let row = |model: &str, fw: &str, spec: &str, kind: &str, isl: Value, date: &str| {
            json!({ "model": model, "isl": isl, "osl": isl, "precision": "fp4", "hardware": "mi355x",
                    "framework": fw, "spec_method": spec, "disagg": false, "benchmark_type": kind, "date": date })
        };
        json!([
            row("gptoss120b", "vllm", "none", "single_turn", json!(1024), "2026-06-05"),
            row("gptoss120b", "vllm", "none", "single_turn", json!(8192), "2026-06-01"),
            row("gptoss120b", "atom", "none", "single_turn", json!(1024), "2026-06-05"),
            row("dsv4", "vllm", "mtp", "single_turn", json!(1024), "2026-07-04"),
            row("dsv4", "vllm", "none", "single_turn", json!(1024), "2026-07-04"),
            row("dsv4", "vllm", "mtp", "agentic_traces", Value::Null, "2026-09-14"),
        ])
    }

    fn display(key: &str) -> String {
        match key {
            "gptoss120b" => "gpt-oss-120b".into(),
            "dsv4" => "DeepSeek-V4-Pro".into(),
            _ => String::new(),
        }
    }

    #[test]
    fn curves_are_named_like_entries_and_selected_by_terms() {
        let all = curves(&avail());
        let names: Vec<String> = all.iter().map(Curve::name).collect();
        assert_eq!(
            names,
            [
                "dsv4-fp4-mi355x-vllm",
                "dsv4-fp4-mi355x-vllm-mtp",
                "dsv4-fp4-mi355x-vllm-mtp-agentic",
                "gptoss120b-fp4-mi355x-atom",
                "gptoss120b-fp4-mi355x-vllm"
            ]
        );
        let vllm = all.iter().find(|c| c.name() == "gptoss120b-fp4-mi355x-vllm").unwrap();
        assert_eq!(
            (vllm.seqs.join(" "), vllm.last.as_str()),
            ("1024/1024 8192/8192".into(), "2026-06-05")
        );
        assert_eq!(all[2].seqs, ["-"], "agentic curves have no fixed sequence");
        let pick = |conf: &str| -> Vec<String> { select(&all, conf, &display).iter().map(|c| c.name()).collect() };
        assert_eq!(pick("gptoss,mi355,vllm"), ["gptoss120b-fp4-mi355x-vllm"]);
        assert_eq!(
            pick("gpt-oss-120b, VLLM"),
            ["gptoss120b-fp4-mi355x-vllm"],
            "the display name"
        );
        // The base curve first, then its variants: fewest name parts.
        assert_eq!(
            pick("dsv4,vllm"),
            [
                "dsv4-fp4-mi355x-vllm",
                "dsv4-fp4-mi355x-vllm-mtp",
                "dsv4-fp4-mi355x-vllm-mtp-agentic"
            ]
        );
        assert_eq!(pick("deepseek,agentic"), ["dsv4-fp4-mi355x-vllm-mtp-agentic"]);
        assert_eq!(pick("dsv4-fp4-mi355x-vllm-mtp"), ["dsv4-fp4-mi355x-vllm-mtp"]);
        assert!(pick("gptoss,b200").is_empty() && pick(" , ").is_empty());
    }

    #[test]
    fn rows_group_into_runs_newest_first() {
        let curve = curves(&avail())
            .into_iter()
            .find(|c| c.name() == "gptoss120b-fp4-mi355x-vllm")
            .unwrap();
        let row = |id: u64, run: &str, date: &str, image: &str, isl: u64, conc: u64| {
            json!({ "id": id.to_string(), "model": "gptoss120b", "hardware": "mi355x", "framework": "vllm", "precision": "fp4",
                    "spec_method": "none", "disagg": false, "benchmark_type": "single_turn", "isl": isl, "osl": 1024,
                    "decode_tp": 1, "decode_ep": 1, "decode_dp_attention": false, "num_decode_gpu": 1, "conc": conc,
                    "image": image, "date": date,
                    "run_url": format!("https://github.com/o/r/actions/runs/{run}/attempts/2"),
                    "metrics": { "output_tput_per_gpu": 100.0 * conc as f64, "median_ttft": 0.05, "median_tpot": 0.005 } })
        };
        let rows = json!([
            // The API sends row ids as strings.
            row(3, "222", "2026-06-05", "img:2", 1024, 8),
            row(1, "111", "2026-03-09", "img:1", 8192, 4),
            row(2, "222", "2026-06-05", "img:2", 1024, 4),
            // Another curve's row (atom) is not one of its points.
            json!({ "id": 9, "model": "gptoss120b", "hardware": "mi355x", "framework": "atom", "precision": "fp4",
                    "spec_method": "none", "disagg": false, "benchmark_type": "single_turn" }),
        ]);
        let runs = runs(&curve, &rows);
        assert_eq!(runs.len(), 2);
        assert_eq!((runs[0].id.as_str(), runs[0].attempt.as_str()), ("222", "2"));
        let ids: Vec<u64> = runs[0].rows.iter().filter_map(|r| n(r, "id")).collect();
        assert_eq!(ids, [2, 3], "points in order");
        assert_eq!(
            runs[0].summary(),
            "run 222 (2026-06-05, image img:2): 2 points, 1024/1024"
        );
        let p = Measured::of(&runs[0].rows[1]);
        assert_eq!((p.conc, p.out_tok_s, p.median_ttft_ms), (8, Some(800.0), Some(50.0)));
        assert_eq!(run_of("https://x/actions/runs/42"), Some(("42".into(), "1".into())));
        assert_eq!(run_of("https://x/pulls/42"), None);
    }

    #[test]
    fn server_args_and_queries() {
        let log =
            "INFO banner\r\n(APIServer pid=1) INFO [utils.py:278] non-default args: {'max_model_len': 2304}\r\nmore\n";
        assert_eq!(
            server_args(log).as_deref(),
            Some("non-default args: {'max_model_len': 2304}")
        );
        assert_eq!(
            server_args("[2026] server_args=ServerArgs(model_path='m', tp_size=8)").as_deref(),
            Some("server_args=ServerArgs(model_path='m', tp_size=8)")
        );
        assert_eq!(server_args("nothing here"), None);
        let c = Client::new("https://d.example/");
        assert_eq!(
            c.url(
                "/api/v1/benchmarks",
                &[("model", "Qwen 3.5/A&B"), ("date", "2026-06-05")]
            ),
            "https://d.example/api/v1/benchmarks?model=Qwen%203.5%2FA%26B&date=2026-06-05"
        );
    }
}
