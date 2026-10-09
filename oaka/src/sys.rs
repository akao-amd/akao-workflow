//! What oaka reads from the machine: work year, GPUs, free ports.

use chrono::Datelike;
use std::fs;
use std::net::TcpListener;

/// GPU architectures a profile's `arch` field may name.
pub const ARCHES: &[&str] = &["gfx942", "gfx950", "gfx1250"];

/// Ports oaka picks from; 30000 is sglang's default and always taken by someone.
pub const PORT_RANGE: std::ops::RangeInclusive<u16> = 29900..=30050;
pub const PORT_AVOID: u16 = 30000;

/// ISO year of today, e.g. "2026" (same rule as akao).
pub fn work_year() -> String {
    chrono::Local::now().iso_week().year().to_string()
}

/// "0.1.0 (<git sha>)", stamped into every compiled script.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("OAKA_GIT_SHA"), ")");

/// Shell-quote one token.
pub fn q(s: &str) -> String {
    shlex::try_quote(s).map(|c| c.into_owned()).unwrap_or_else(|_| s.to_string())
}

/// gfx name from a KFD `gfx_target_version` (major*10000 + minor*100 + stepping).
fn gfx_name(v: u32) -> String {
    format!("gfx{}{:x}{:x}", v / 10000, (v / 100) % 100, v % 100)
}

/// GPU architectures in HIP device order, from the KFD topology.
/// None when the topology is unreadable (not a ROCm machine).
pub fn gpus() -> Option<Vec<String>> {
    let dir = fs::read_dir("/sys/class/kfd/kfd/topology/nodes").ok()?;
    let mut nodes: Vec<(u32, String)> = Vec::new();
    for entry in dir.flatten() {
        let Ok(idx) = entry.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(props) = fs::read_to_string(entry.path().join("properties")) else { continue };
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
    fn gfx_names() {
        assert_eq!(gfx_name(90402), "gfx942");
        assert_eq!(gfx_name(90500), "gfx950");
        assert_eq!(gfx_name(120500), "gfx1250");
        assert_eq!(gfx_name(90010), "gfx90a");
    }
}
