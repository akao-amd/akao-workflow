//! The console tmux and worker Docker are stand-ins; launch quoting is exercised by shells.
mod common;
use common::{output, Sandbox};
use std::fs;

const TMUX: &str = r#"#!/usr/bin/python3 -I
import json, os, subprocess, sys
from pathlib import Path
a=sys.argv[1:]; r=Path(os.environ['FAKE_ROOT'])
with (r/'tmux.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
if a[0]=='display-message': print('0'); sys.exit(0)
if a[0]=='has-session': sys.exit(1 if os.environ.get('FAKE_NO_SESSION') else 0)
if a[0]=='list-windows': print(os.environ.get('FAKE_WINDOWS','worker-other')); sys.exit(0)
assert a[0]=='new-window', a
print('0:3 @17', flush=True)
sys.exit(subprocess.run(['/bin/sh','-c',a[-1]]).returncode)
"#;

const DOCKER: &str = r#"#!/usr/bin/python3 -I
import json, os, subprocess, sys
from pathlib import Path
a=sys.argv[1:]; r=Path(os.environ['FAKE_ROOT'])
with (r/'docker.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
assert a[:2]==['-H','ssh://fakebox'], a
if os.environ.get('FAKE_DOCKER_DOWN'): print('cannot connect to Docker',file=sys.stderr); sys.exit(31)
a=a[2:]
if a[:2]==['container','inspect']: print(os.environ.get('FAKE_RUNNING','true')); sys.exit(0)
assert a[0]=='exec',a
a=a[1:]
if a[:1]==['-it']: a=a[1:]
if a[:1]==['-w']: os.chdir(a[1]); a=a[2:]
assert a[0]=='akao_worker',a
a=a[1:]
if a[:2]==['test','-d']: sys.exit(0 if Path(a[2]).is_dir() else 1)
assert a[:2]==['bash','-lc'],a
# Model the worker shell without reading real login profiles or invoking a real agent.
os.environ['AKAO_REPO_ROOT']=os.environ.get('FAKE_WORKER_REPO','worker repo with spaces')
script=a[2].replace('/root/.local/bin/claude', str(r/'bin/agent'))
sys.exit(subprocess.run(['/bin/bash','-c',script]).returncode)
"#;

fn sandbox(name: &str) -> Sandbox {
    let s = Sandbox::new(name);
    s.exe("bin/tmux", TMUX);
    s.exe("bin/docker", DOCKER);
    s.exe(
        "bin/agent",
        r#"#!/usr/bin/python3 -I
import json, os, sys
from pathlib import Path
(Path(os.environ['FAKE_ROOT'])/'agent.jsonl').write_text(json.dumps(sys.argv[1:])+'\n')
"#,
    );
    s
}

#[test]
fn open_targets_the_exact_session_and_passes_environment_and_quoted_resume_arguments() {
    let s = sandbox("open-quoting");
    let dir = s.path("worker/task with spaces");
    fs::create_dir(&dir).unwrap();
    let resume = format!("saved session; touch {}", s.path("injected").display());
    let worker_repo = "worker repo $HOME `literal`";
    let (code, out) = output(
        s.command(&[
            "open",
            "fakebox",
            "worker",
            "--cwd",
            dir.to_str().unwrap(),
            "--session",
            "0:",
            "--resume",
            &resume,
        ])
        .env("FAKE_WORKER_REPO", worker_repo),
    );
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("0:3 @17"), "{out}");
    let tmux = s.events("tmux.jsonl");
    assert!(tmux.iter().any(|a| a == &["has-session", "-t", "=0"]));
    let new = tmux.iter().find(|a| a[0] == "new-window").unwrap();
    assert!(new.windows(2).any(|a| a == ["-t", "=0:"]));
    assert!(new
        .windows(2)
        .any(|a| a[0] == "-e" && a[1] == format!("AKAO_CONFIG_ROOT={}", s.path("state").display())));
    assert_eq!(
        s.events("agent.jsonl")[0],
        vec!["--resume", &resume, "--add-dir", worker_repo]
    );
    assert!(!s.path("injected").exists());
    assert!(s
        .events("docker.jsonl")
        .iter()
        .all(|a| a[..2] == ["-H", "ssh://fakebox"]));
}

#[test]
fn open_latest_uses_the_current_session_and_dry_run_creates_no_window() {
    let s = sandbox("open-latest");
    let (code, out) = s.run(&["open", "fakebox", "worker", "--continue"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(
        s.events("agent.jsonl")[0],
        vec!["--continue", "--add-dir", "worker repo with spaces"]
    );
    let s = sandbox("open-dry");
    let (code, out) = s.run(&["open", "fakebox", "worker", "--dry-run"]);
    assert_eq!(code, 0, "{out}");
    assert!(!s.events("tmux.jsonl").iter().any(|a| a[0] == "new-window"));
    assert!(s.events("agent.jsonl").is_empty());
    assert!(!out.contains("opened akao_worker"), "{out}");
}

#[test]
fn open_refuses_duplicate_windows_missing_sessions_workers_and_directories() {
    for (key, value, expected) in [
        ("FAKE_WINDOWS", "worker\nworker-other", "already has a window"),
        ("FAKE_NO_SESSION", "1", "no tmux session"),
        ("FAKE_RUNNING", "false", "is not running"),
        ("FAKE_DOCKER_DOWN", "1", "cannot inspect"),
    ] {
        let s = sandbox(key);
        let (code, out) = output(s.command(&["open", "fakebox", "worker"]).env(key, value));
        assert_ne!(code, 0, "{out}");
        assert!(out.contains(expected), "{out}");
        assert!(!s.events("tmux.jsonl").iter().any(|a| a[0] == "new-window"));
        if key == "FAKE_DOCKER_DOWN" {
            assert!(!out.contains("is not running"), "{out}");
        }
    }
    let s = sandbox("open-no-cwd");
    let cwd = s.path("missing");
    let (code, out) = s.run(&["open", "fakebox", "worker", "--cwd", cwd.to_str().unwrap()]);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("not a directory"), "{out}");
    assert!(s.events("agent.jsonl").is_empty());
    let (code, out) = s.run(&["open", "fakebox", "worker", "--continue", "--resume", "id"]);
    assert_ne!(code, 0, "{out}");
}
