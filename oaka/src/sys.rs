//! What oaka reads from the machine: work year, GPUs, free ports.

use chrono::Datelike;
use std::fs;
use std::net::TcpListener;

/// GPU architectures a profile's `arch` field may name.
pub const ARCHES: &[&str] = &["gfx942", "gfx950", "gfx1250"];

/// Ports oaka picks from; 30000 is sglang's default and always taken by someone.
pub const PORT_RANGE: std::ops::RangeInclusive<u16> = 29900..=30100;
pub const PORT_AVOID: u16 = 30000;

/// ISO year of today, e.g. "2026" (same rule as akao).
pub fn work_year() -> String {
    chrono::Local::now().iso_week().year().to_string()
}

/// "0.1.0 (<git sha>)", stamped into every compiled script.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("OAKA_GIT_SHA"), ")");

/// Shell-quote one token.
pub fn q(s: &str) -> String {
    shlex::try_quote(s)
        .map(|c| c.into_owned())
        .unwrap_or_else(|_| s.to_string())
}

/// gfx name from a KFD `gfx_target_version` (major*10000 + minor*100 + stepping).
fn gfx_name(v: u32) -> String {
    format!("gfx{}{:x}{:x}", v / 10000, (v / 100) % 100, v % 100)
}

/// GPU architectures in HIP device order: $OAKA_GPUS (comma-separated archs; empty = no
/// GPUs) when set, else the KFD topology.  None when neither is available.
pub fn gpus() -> Option<Vec<String>> {
    if let Ok(v) = std::env::var("OAKA_GPUS") {
        return Some(
            v.split(',')
                .map(str::trim)
                .filter(|a| !a.is_empty())
                .map(String::from)
                .collect(),
        );
    }
    let dir = fs::read_dir("/sys/class/kfd/kfd/topology/nodes").ok()?;
    let mut nodes: Vec<(u32, String)> = Vec::new();
    for entry in dir.flatten() {
        let Ok(idx) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let Ok(props) = fs::read_to_string(entry.path().join("properties")) else {
            continue;
        };
        let v = props
            .lines()
            .find_map(|l| l.strip_prefix("gfx_target_version "))
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0);
        if v != 0 {
            nodes.push((idx, gfx_name(v)));
        }
    }
    nodes.sort();
    Some(nodes.into_iter().map(|(_, a)| a).collect())
}

/// The one architecture of this machine's GPUs, if they all agree.
pub fn arch() -> Option<String> {
    let g = gpus()?;
    let first = g.first()?.clone();
    g.iter().all(|a| *a == first).then_some(first)
}

/// This container's ROCm version ("10.0.0") and where it was read: $OAKA_ROCM, else the
/// first `.info/version` under $ROCM_PATH, $ROCM_HOME, /opt/rocm.  ROCm-10 images keep
/// ROCm in site-packages, with /opt/rocm a link; HIP's own version (7.15) is not it.
pub fn rocm() -> Option<(String, String)> {
    if let Ok(v) = std::env::var("OAKA_ROCM") {
        return Some((v.trim().to_string(), "OAKA_ROCM".into())).filter(|(v, _)| !v.is_empty());
    }
    let roots = ["ROCM_PATH", "ROCM_HOME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .chain(["/opt/rocm".to_string()]);
    for root in roots {
        let path = format!("{}/.info/version", root.trim_end_matches('/'));
        if let Some(v) = fs::read_to_string(&path).ok().and_then(|t| leading_version(&t)) {
            return Some((v, path));
        }
    }
    None
}

/// "7.2.0-43" -> "7.2.0": the leading dotted number.
fn leading_version(text: &str) -> Option<String> {
    let v: String = text
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let v = v.trim_end_matches('.');
    (!v.is_empty()).then(|| v.to_string())
}

/// A ROCm version pattern: dotted components, each a number or `*`, matched as a prefix:
/// "10.0" is any 10.0.x, "10.*.1" any 10.y.1, "*" anything.
pub fn valid_rocm_pattern(p: &str) -> bool {
    !p.is_empty()
        && p.split('.')
            .all(|c| c == "*" || (!c.is_empty() && c.chars().all(|d| d.is_ascii_digit())))
}

/// Whether `version` (e.g. "10.0.0") matches `pattern` (see valid_rocm_pattern).
pub fn rocm_matches(pattern: &str, version: &str) -> bool {
    let have: Vec<&str> = version.split('.').collect();
    pattern.split('.').enumerate().all(|(i, want)| {
        let got = have.get(i).copied().unwrap_or("0");
        want == "*" || want.parse::<u64>().ok() == got.parse::<u64>().ok()
    })
}

pub fn port_free(port: u16) -> bool {
    TcpListener::bind(("0.0.0.0", port)).is_ok()
}

/// A random free port in PORT_RANGE that is not in `taken`.
pub fn pick_port(taken: &[u16]) -> Option<u16> {
    let mut ports: Vec<u16> = PORT_RANGE.filter(|p| *p != PORT_AVOID && !taken.contains(p)).collect();
    fastrand::shuffle(&mut ports);
    ports.into_iter().find(|p| port_free(*p))
}

/// Directory names under /model, for draft's comments.
pub fn models() -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir("/model")
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picked_ports_stay_in_range() {
        for _ in 0..50 {
            let p = pick_port(&[29900]).unwrap();
            assert!((29901..=30100).contains(&p) && p != PORT_AVOID, "{p}");
        }
    }

    #[test]
    fn rocm_patterns() {
        assert!(rocm_matches("10.0", "10.0.0") && rocm_matches("10.0", "10.0.3") && !rocm_matches("10.0", "10.1.0"));
        assert!(rocm_matches("*", "7.2.0") && rocm_matches("10.*.1", "10.4.1") && !rocm_matches("10.*.1", "10.4.2"));
        assert!(rocm_matches("10.0.0", "10.0") && !rocm_matches("10", "1.0"));
        assert!(valid_rocm_pattern("10.1") && valid_rocm_pattern("*") && valid_rocm_pattern("10.*"));
        assert!(!valid_rocm_pattern(">=10.1") && !valid_rocm_pattern("10..1") && !valid_rocm_pattern(""));
        assert_eq!(leading_version("7.2.0-43\n").as_deref(), Some("7.2.0"));
        assert_eq!(leading_version("10.0.0\n").as_deref(), Some("10.0.0"));
        assert_eq!(leading_version("unknown"), None);
    }

    #[test]
    fn gfx_names() {
        assert_eq!(gfx_name(90402), "gfx942");
        assert_eq!(gfx_name(90500), "gfx950");
        assert_eq!(gfx_name(120500), "gfx1250");
        assert_eq!(gfx_name(90010), "gfx90a");
    }
}
