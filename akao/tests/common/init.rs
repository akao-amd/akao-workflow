//! Stand-ins for init: queries are modeled and mutations only logged, never executed.
#![allow(dead_code)]
use crate::common::Sandbox;
use chrono::Datelike;

pub fn year() -> String {
    chrono::Local::now().iso_week().year().to_string()
}

pub fn existing(s: &Sandbox, image: &str, root: &str, pinned: bool) -> String {
    format!(
        "old-id\n{image}\n{root}\nmount /{} {}/{}\nmount /root {}/container_home/akao_worker\n{}",
        year(),
        s.path("host").display(),
        year(),
        s.path("host").display(),
        if pinned {
            format!("env AKAO_ARTIFACT_ROOT={root}\n")
        } else {
            String::new()
        }
    )
}

pub fn sandbox(name: &str) -> Sandbox {
    let s = Sandbox::new(name);
    s.exe(
        "bin/ssh",
        r#"#!/usr/bin/python3 -I
import json, os, shlex, sys
from pathlib import Path
a=sys.argv[1:]; r=Path(os.environ['FAKE_ROOT'])
with (r/'ssh.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
if '-G' in a: print('hostname example.invalid'); sys.exit(0)
script=shlex.split(a[-1])[-1]
if 'systemctl is-active' in script:
    if os.environ.get('FAKE_DIAG_FAIL'): sys.exit(55)
    print(os.environ.get('FAKE_DAEMON','active')); sys.exit(0)
if script.startswith('test -d '): sys.exit(1)
if os.environ.get('FAKE_SSH_DENIED'): print('Permission denied',file=sys.stderr); sys.exit(255)
sys.exit(0)
"#,
    );
    s.exe(
        "bin/docker",
        r#"#!/usr/bin/python3 -I
import json, os, sys
from pathlib import Path
a=sys.argv[1:]; r=Path(os.environ['FAKE_ROOT'])
with (r/'docker.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
if 'version' in a:
    if os.environ.get('FAKE_DOCKER_DOWN'):
        if os.environ.get('FAKE_REMOVE_SSH'): (r/'bin/ssh').unlink()
        print('original Docker transport error',file=sys.stderr); sys.exit(31)
    print('29.0'); sys.exit(0)
if 'ps' in a:
    if os.environ.get('FAKE_PS_DENIED'): print('cannot list containers',file=sys.stderr); sys.exit(32)
    if os.environ.get('FAKE_EXISTING'): print('akao_worker')
    sys.exit(0)
if 'container' in a and 'inspect' in a:
    if os.environ.get('FAKE_INSPECT_DENIED'): print('cannot inspect worker',file=sys.stderr); sys.exit(33)
    if not os.environ.get('FAKE_EXISTING'): sys.exit(1)
    if a[a.index('--format')+1]=='{{.Id}} {{.State.Status}}':
        print(os.environ.get('FAKE_REPLACED','old-id')+' running')
    else: print(os.environ['FAKE_EXISTING'])
    sys.exit(0)
if 'context' in a and 'inspect' in a: sys.exit(1)
sys.exit(0)
"#,
    );
    s
}
