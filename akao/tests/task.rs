//! Execute the task-creation bash against a stand-in Docker in scratch directories.
mod common;
use common::{output, text, Sandbox};
use std::fs;

const DOCKER: &str = r#"#!/usr/bin/python3 -I
import json, os, subprocess, sys
from pathlib import Path
a=sys.argv[1:]
r=Path(os.environ['FAKE_ROOT'])
with (r/'docker.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
assert a[:3] == ['-H','ssh://fakebox','exec'], a
if os.environ.get('FAKE_RUNNING') == 'false':
    print('worker is not running', file=sys.stderr); sys.exit(1)
a=a[3:]
if a[0]=='-i': a=a[1:]
assert a[0]=='akao_worker', a
sys.exit(subprocess.run(a[1:], cwd=r/'worker').returncode)
"#;

fn sandbox(name: &str) -> Sandbox {
    let s = Sandbox::new(name);
    s.exe("bin/docker", DOCKER);
    s
}

#[test]
fn tasks_use_the_worker_root_and_directory_numbers_and_install_the_exact_brief() {
    let s = sandbox("task-numbering");
    fs::create_dir(s.path("worker/0002_old")).unwrap();
    fs::create_dir(s.path("worker/0010_old")).unwrap();
    s.put("worker/9999_file", "unrelated file\n");
    fs::create_dir(s.path("worker/unrelated")).unwrap();
    let brief = s.path("brief with spaces.md");
    fs::write(&brief, b"brief\n$HOME `not executed`\0\xff").unwrap();
    let (code, out) = s.run(&["task", "fakebox", "worker", "a.b-c", "--brief", brief.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}");
    let dir = s.path("worker/0011_a.b-c");
    assert_eq!(fs::read(dir.join("BRIEF.md")).unwrap(), fs::read(brief).unwrap());
    assert!(!dir.join(".BRIEF.md.part").exists());
    assert_eq!(out.lines().last().unwrap(), format!("task dir {}", dir.display()));
    assert!(!s.root.join("0011_a.b-c").exists());
}

#[test]
fn task_directory_symlinks_count_but_files_do_not_reserve_a_number() {
    let s = sandbox("task-symlink");
    fs::create_dir(s.path("worker/0008_old")).unwrap();
    std::os::unix::fs::symlink("0008_old", s.path("worker/0012_link")).unwrap();
    let (code, out) = s.run(&["task", "fakebox", "akao_worker", "next"]);
    assert_eq!(code, 0, "{out}");
    assert!(s.path("worker/0013_next").is_dir());
    s.put("worker/7070_note.txt", "keep\n");
    let (code, out) = s.run(&["task", "fakebox", "worker", "chosen", "--number", "7070"]);
    assert_eq!(code, 0, "{out}");
    assert!(s.path("worker/7070_chosen").is_dir());
    assert_eq!(text(s.path("worker/7070_note.txt")), "keep\n");
    let (code, out) = s.run(&["task", "fakebox", "worker", "different", "--number", "7070"]);
    assert_ne!(code, 0, "{out}");
    assert!(!s.path("worker/7070_different").exists());
}

#[test]
fn task_rejects_bad_inputs_missing_roots_and_exhausted_numbering() {
    let s = sandbox("task-invalid");
    for number in ["", "0000", "7", "10000", "abcd"] {
        let (code, out) = s.run(&["task", "fakebox", "worker", "x", "--number", number]);
        assert_ne!(code, 0, "{number:?}: {out}");
        assert!(out.contains("four digits"), "{out}");
    }
    for description in ["../escape", "bad desc", "-bad", ""] {
        let (code, out) = s.run(&["task", "fakebox", "worker", "--", description]);
        assert_ne!(code, 0, "{out}");
    }
    assert!(s.events("docker.jsonl").is_empty());
    let (code, out) = output(
        s.command(&["task", "fakebox", "worker", "x"])
            .env("AKAO_ARTIFACT_ROOT", "relative"),
    );
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("not absolute"), "{out}");
    let (code, out) = output(
        s.command(&["task", "fakebox", "worker", "x"])
            .env("AKAO_ARTIFACT_ROOT", s.path("missing")),
    );
    assert_ne!(code, 0, "{out}");
    let (code, out) = output(
        s.command(&["task", "fakebox", "worker", "x"])
            .env_remove("AKAO_ARTIFACT_ROOT"),
    );
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("not under /<year>/"), "{out}");
    fs::create_dir(s.path("worker/9999_last")).unwrap();
    let (code, out) = s.run(&["task", "fakebox", "worker", "x"]);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("task 9999"), "{out}");
}

#[test]
fn task_dry_run_and_failures_preserve_all_existing_tasks() {
    let s = sandbox("task-failures");
    s.put("brief.md", "brief\n");
    let brief = s.path("brief.md");
    let args = ["task", "fakebox", "worker", "x", "--brief", brief.to_str().unwrap()];
    let mut dry = args.to_vec();
    dry.push("--dry-run");
    let (code, out) = s.run(&dry);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("would make"), "{out}");
    assert_eq!(fs::read_dir(s.path("worker")).unwrap().count(), 0);
    let (code, out) = output(s.command(&args).env("FAKE_RUNNING", "false"));
    assert_ne!(code, 0, "{out}");
    assert_eq!(fs::read_dir(s.path("worker")).unwrap().count(), 0);
    s.exe("bin/cat", "#!/bin/sh\nexit 17\n");
    let (code, out) = s.run(&args);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("made, but its BRIEF.md did not arrive"), "{out}");
    assert!(s.path("worker/0001_x").is_dir());
    assert!(!s.path("worker/0001_x/BRIEF.md").exists());
    s.exe("bin/cat", "#!/bin/sh\nexec /bin/cat \"$@\"\n");
    s.exe("bin/mv", "#!/bin/sh\nexit 18\n");
    let (code, out) = s.run(&args);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("made, but its BRIEF.md could not be installed"), "{out}");
    assert!(s.path("worker/0002_x").is_dir());
    assert!(!s.path("worker/0002_x/BRIEF.md").exists());
}

#[test]
fn task_reads_a_custom_worker_root_and_rejects_a_missing_brief_before_allocation() {
    let s = sandbox("task custom root");
    let root = s.path("another worker root");
    fs::create_dir(&root).unwrap();
    fs::create_dir(root.join("0100_old")).unwrap();
    let (code, out) = output(
        s.command(&["task", "fakebox", "worker", "x"])
            .env("AKAO_ARTIFACT_ROOT", &root),
    );
    assert_eq!(code, 0, "{out}");
    assert!(root.join("0101_x").is_dir());
    assert_eq!(fs::read_dir(s.path("worker")).unwrap().count(), 0);
    let missing = s.path("missing.md");
    let calls = s.events("docker.jsonl").len();
    let (code, out) = s.run(&["task", "fakebox", "worker", "x", "--brief", missing.to_str().unwrap()]);
    assert_ne!(code, 0, "{out}");
    assert_eq!(s.events("docker.jsonl").len(), calls);
}

#[test]
fn concurrent_identical_explicit_tasks_cannot_share_or_overwrite_the_directory() {
    let s = sandbox("task-concurrent");
    let args = ["task", "fakebox", "worker", "x", "--number", "0007"];
    let a = s
        .command(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let b = s
        .command(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let statuses = [
        a.wait_with_output().unwrap().status,
        b.wait_with_output().unwrap().status,
    ];
    assert_eq!(statuses.iter().filter(|s| s.success()).count(), 1);
    assert!(s.path("worker/0007_x").is_dir());
}

#[test]
fn concurrent_tasks_with_different_descriptions_preserve_each_successful_brief() {
    let s = sandbox("task-concurrent-descriptions");
    let mut children = Vec::new();
    for (desc, body) in [("first", "first brief\n"), ("second", "second brief\n")] {
        let brief = s.path(&format!("{desc}.md"));
        fs::write(&brief, body).unwrap();
        let child = s
            .command(&[
                "task",
                "fakebox",
                "worker",
                desc,
                "--number",
                "0007",
                "--brief",
                brief.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        children.push((desc, body, child));
    }
    let mut successes = 0;
    for (desc, body, child) in children {
        let result = child.wait_with_output().unwrap();
        let out = format!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        let dir = s.path(&format!("worker/0007_{desc}"));
        if result.status.success() {
            successes += 1;
            assert_eq!(text(dir.join("BRIEF.md")), body);
            assert!(!dir.join(".BRIEF.md.part").exists());
        } else {
            assert!(out.contains("task number 0007 is taken"), "{out}");
            assert!(!dir.exists());
        }
    }
    // Different descriptions can both win under the accepted best-effort numbering policy.
    assert!((1..=2).contains(&successes));
    assert_eq!(fs::read_dir(s.path("worker")).unwrap().count(), successes);
}
