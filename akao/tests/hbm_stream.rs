//! The host-only HBM utility's reports, without ROCm, RVS, or a remote GPU.
use std::process::Command;

#[test]
fn hbm_reports_and_host_driver() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = Command::new("python3")
        .args(["-I", "-B"])
        .arg(root.join("utils/tests/test_hbm_stream.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
