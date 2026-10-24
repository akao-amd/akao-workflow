//! Local ':' copies exercise real tar, with '/'year destinations remapped by stand-ins.
//! The mkdir/tar stand-ins prevent any read/write to the real /<year> tree.
mod common;
use chrono::Datelike;
use common::{text, Sandbox};

const SSH: &str = r#"#!/usr/bin/python3 -I
import subprocess,sys
sys.exit(subprocess.run(['/bin/sh','-c',sys.argv[-1]]).returncode)
"#;

const FILE_TOOL: &str = r#"#!/usr/bin/python3 -I
import json,os,subprocess,sys
from pathlib import Path
r=Path(os.environ['FAKE_ROOT']); year=os.environ['FAKE_YEAR']
tool=Path(sys.argv[0]).name
a=sys.argv[1:]
with (r/'file-tools.jsonl').open('a') as f: f.write(json.dumps([tool]+a)+'\n')
prefix='/'+year
a=[str(r/'console'/year)+v[len(prefix):] if v==prefix or v.startswith(prefix+'/') else v for v in a]
if tool=='mkdir': paths=[v for v in a if not v.startswith('-')]
else: paths=[a[a.index('-C')+1]]
for v in paths: assert Path(v).resolve().is_relative_to(r), v
sys.exit(subprocess.run(['/bin/'+tool]+a).returncode)
"#;

fn sandbox(name: &str) -> (Sandbox, String) {
    let s = Sandbox::new(name);
    let year = chrono::Local::now().iso_week().year().to_string();
    s.exe("bin/ssh", SSH);
    s.exe("bin/mkdir", FILE_TOOL);
    s.exe("bin/tar", FILE_TOOL);
    s.put(&format!("host/{year}/ww42/w/0001_a/result.txt"), "new\n");
    (s, year)
}

#[test]
fn colon_destination_keeps_slashes_and_older_week_paths_and_merges_without_deleting() {
    let (s, year) = sandbox("cp-local");
    s.put(&format!("console/{year}/ww42/w/0001_a/keep.txt"), "preserved\n");
    s.put(&format!("console/{year}/ww42/w/0001_a/result.txt"), "old\n");
    let (code, out) = common::output(
        s.command(&["cp", "fakebox:/ww42/w/0001_a/", ":"])
            .env("FAKE_YEAR", &year),
    );
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("(the source's path)"), "{out}");
    assert_eq!(
        text(s.path(&format!("console/{year}/ww42/w/0001_a/result.txt"))),
        "new\n"
    );
    assert_eq!(
        text(s.path(&format!("console/{year}/ww42/w/0001_a/keep.txt"))),
        "preserved\n"
    );
    let (code, out) = common::output(
        s.command(&["cp", "fakebox:ww43/../ww42/w/0001_a", ":"])
            .env("FAKE_YEAR", &year),
    );
    // Traversal is explicitly taken at face value, so the source's intermediate dir must exist.
    assert_ne!(code, 0, "{out}");
    std::fs::create_dir(s.path(&format!("host/{year}/ww43"))).unwrap();
    let (code, out) = common::output(
        s.command(&["cp", "fakebox:ww43/../ww42/w/0001_a", ":"])
            .env("FAKE_YEAR", &year),
    );
    assert_eq!(code, 0, "{out}");
    let (code, out) = common::output(s.command(&["cp", "fakebox:ww42", ":"]).env("FAKE_YEAR", &year));
    assert_eq!(code, 0, "{out}");
    assert!(s.path(&format!("console/{year}/ww42/w/0001_a/result.txt")).exists());
}

#[test]
fn colon_destination_rejects_local_empty_and_malformed_sources_and_dry_run_writes_nothing() {
    let (s, year) = sandbox("cp-local-refuse");
    let local = s.path(&format!("host/{year}/ww42/w/0001_a"));
    for src in [local.to_str().unwrap(), "fakebox:", "fakebox:////", ":ww42/w"] {
        let (code, out) = common::output(s.command(&["cp", src, ":"]).env("FAKE_YEAR", &year));
        assert_ne!(code, 0, "{out}");
    }
    let (code, out) = common::output(
        s.command(&["cp", "fakebox:ww42/w/0001_a", ":", "--dry-run"])
            .env("FAKE_YEAR", &year),
    );
    assert_eq!(code, 0, "{out}");
    assert!(s.events("file-tools.jsonl").is_empty());
    assert!(!s.path("console").exists());
}

#[test]
fn explicit_local_destinations_keep_the_basename_inside_the_named_directory() {
    let (s, year) = sandbox("cp-explicit-local");
    let dst = s.path("explicit");
    let (code, out) = common::output(
        s.command(&["cp", "fakebox:ww42/w/0001_a", dst.to_str().unwrap()])
            .env("FAKE_YEAR", &year),
    );
    assert_eq!(code, 0, "{out}");
    assert_eq!(text(dst.join("0001_a/result.txt")), "new\n");
    assert!(!s.path("console").exists());
}
