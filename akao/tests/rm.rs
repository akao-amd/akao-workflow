//! Retirement and home cleanup against a stateful Docker and local SSH stand-in.
mod common;
use common::{output, text, Sandbox};
use serde_json::{json, Value};
use std::fs;

const DOCKER: &str = r#"#!/usr/bin/python3 -I
import json,os,re,sys
from pathlib import Path
r=Path(os.environ['FAKE_ROOT']); a=sys.argv[1:]
with (r/'docker.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
assert a[:2]==['-H','ssh://fakebox'],a
a=a[2:]; p=r/'scenario.json'; s=json.loads(p.read_text()); w=s['workers']
def save(): p.write_text(json.dumps(s))
def fail(msg): print(msg,file=sys.stderr); sys.exit(37)
if os.environ.get('FAKE_DOCKER_DOWN'): fail('Docker is unreachable')
if a[0]=='version': print('29.0'); sys.exit(0)
if a[0]=='ps':
    if '-aq' in a:
        if s.get('census_fail'): fail('census failed')
        print('\n'.join([x['id'] for x in w.values()]+(['foreign-id'] if s.get('mounts') else []))); sys.exit(0)
    if s.get('query_fail_after_stop') and s.get('stopped'): fail('absence query failed')
    pattern=a[a.index('--filter')+1].split('=',1)[1]
    for name,x in w.items():
        if re.fullmatch(pattern,name): print(x['id']+' '+name)
    sys.exit(0)
if a[:2]==['container','inspect']:
    fmt=a[a.index('--format')+1]
    if '.Mounts' in fmt:
        if s.get('mount_inspect_fail'): fail('mount inspection failed')
        print('\n'.join(s.get('mounts',[]))); sys.exit(0)
    identifier=a[-1]
    entry=next(((n,x) for n,x in w.items() if x['id']==identifier),None)
    if entry is None: sys.exit(1)
    name,x=entry
    if x.get('inspect_fail'): fail('cannot inspect worker')
    if x.get('remove_after'):
        x['polls']=x.get('polls',0)+1
        if x['polls']>=x['remove_after']: del w[name]; save(); sys.exit(1)
        save()
    print(x['status']); sys.exit(0)
if a[0] in ['stop','rm']:
    identifier=a[-1]
    entry=next(((n,x) for n,x in w.items() if x['id']==identifier),None)
    if entry is None: fail('already gone')
    name,x=entry
    if a[0]=='stop':
        s['stopped']=True
        if x.get('replace_on_stop'):
            w[name]={'id':'replacement-id','status':'running'}
        elif x.get('auto'): del w[name]
        else: x['status']='exited'
    else: del w[name]
    save()
    if x.get('race'): fail('auto-removal won race')
    sys.exit(0)
fail('unexpected command')
"#;

const SSH: &str = r#"#!/usr/bin/python3 -I
import json,os,subprocess,sys
from pathlib import Path
r=Path(os.environ['FAKE_ROOT']); a=sys.argv[1:]
with (r/'ssh.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
if os.environ.get('FAKE_SSH_FAIL'): print('SSH unreachable',file=sys.stderr); sys.exit(255)
sys.exit(subprocess.run(['/bin/sh','-c',a[-1]]).returncode)
"#;

fn sandbox(name: &str, scenario: Value) -> Sandbox {
    let s = Sandbox::new(name);
    s.exe("bin/docker", DOCKER);
    s.exe("bin/ssh", SSH);
    s.put("scenario.json", &scenario.to_string());
    s.put("worker/report.md", "preserve artifacts\n");
    s
}

#[test]
fn rm_handles_running_stopped_paused_dead_and_auto_removal_races_by_id() {
    for (state, auto, race) in [
        ("running", false, false),
        ("running", true, false),
        ("running", true, true),
        ("restarting", false, false),
        ("paused", false, false),
        ("exited", false, false),
        ("created", false, false),
        ("dead", false, true),
    ] {
        let id = "a".repeat(64);
        let s = sandbox(
            &format!("rm-{state}-{auto}-{race}"),
            json!({"workers":{"akao_x":{"id":id,"status":state,"auto":auto,"race":race}}}),
        );
        let (code, out) = s.run(&["rm", "fakebox", "x"]);
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("removed"), "{out}");
        let events = s.events("docker.jsonl");
        let mutations: Vec<_> = events.iter().filter(|a| a[2] == "rm" || a[2] == "stop").collect();
        assert!(mutations.iter().all(|a| a.last() == Some(&id)));
        if state == "running" && !auto {
            assert_eq!(mutations.len(), 2);
        }
        if state == "paused" {
            assert!(mutations[0].contains(&"-f".into()));
        }
        assert_eq!(text(s.path("worker/report.md")), "preserve artifacts\n");
    }
}

#[test]
fn rm_absence_dry_run_and_asynchronous_removal_are_idempotent() {
    let s = sandbox("rm-absent", json!({"workers":{}}));
    let (code, out) = s.run(&["rm", "fakebox", "missing"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("not there"), "{out}");
    let s = sandbox(
        "rm-dry",
        json!({"workers":{"akao_x":{"id":"old-id","status":"running"}}}),
    );
    let (code, out) = s.run(&["rm", "fakebox", "x", "--dry-run"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("would stop/remove"), "{out}");
    assert!(!out.contains("x: removed"), "{out}");
    assert!(!s.events("docker.jsonl").iter().any(|a| a[2] == "stop" || a[2] == "rm"));
    let s = sandbox(
        "rm-async",
        json!({"workers":{"akao_x":{"id":"old-id","status":"removing","remove_after":2}}}),
    );
    let (code, out) = s.run(&["rm", "fakebox", "x"]);
    assert_eq!(code, 0, "{out}");
    assert!(!s.events("docker.jsonl").iter().any(|a| a[2] == "stop" || a[2] == "rm"));
    let (code, out) = s.run(&["rm", "fakebox", "x"]);
    assert_eq!(code, 0, "{out}");
}

#[test]
fn rm_refuses_replacements_and_failed_queries_and_continues_other_names() {
    let s = sandbox(
        "rm-replaced",
        json!({"workers":{"akao_x":{"id":"old-id","status":"running","replace_on_stop":true}}}),
    );
    let (code, out) = s.run(&["rm", "fakebox", "x"]);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("replaced meanwhile"), "{out}");
    assert!(!s.events("docker.jsonl").iter().any(|a| a[2] == "rm"));
    let s = sandbox(
        "rm-continue",
        json!({"workers":{"akao_bad":{"id":"bad-id","status":"exited","inspect_fail":true},"akao_good":{"id":"good-id","status":"exited"}}}),
    );
    let (code, out) = s.run(&["rm", "fakebox", "bad", "good"]);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("good: removed"), "{out}");
    let s = sandbox(
        "rm-query-fail",
        json!({"query_fail_after_stop":true,"workers":{"akao_x":{"id":"old-id","status":"running","auto":true}}}),
    );
    let (code, out) = s.run(&["rm", "fakebox", "x"]);
    assert_ne!(code, 0, "{out}");
    assert!(!out.contains("x: removed"), "{out}");
    let s = sandbox("rm-down", json!({"workers":{}}));
    let (code, out) = output(s.command(&["rm", "fakebox", "missing"]).env("FAKE_DOCKER_DOWN", "1"));
    assert_ne!(code, 0, "{out}");
    assert!(!out.contains("not there"), "{out}");
}

#[test]
fn rm_home_requires_a_successful_mount_census_and_reports_ssh_failure() {
    let s = sandbox("rm-help", json!({"workers":{}}));
    let (code, help) = s.run(&["rm", "--help"]);
    assert_eq!(code, 0, "{help}");
    assert!(help.contains("clone") && help.contains("agent history"), "{help}");
    for shape in ["equal", "parent", "child", "root", "prefix", "space-parent"] {
        let s = sandbox(shape, json!({"workers":{}}));
        let home = if shape == "space-parent" {
            let host = s.path("host with space ");
            s.put("state/hosts.tsv", &format!("fakebox\t/model\t-\t{}\n", host.display()));
            host.join("container_home/akao_x")
        } else {
            s.path("host/container_home/akao_x")
        };
        fs::create_dir_all(&home).unwrap();
        let mount = match shape {
            "equal" => home.clone(),
            "parent" => home.parent().unwrap().to_path_buf(),
            "child" => home.join("cache"),
            "root" => "/".into(),
            "space-parent" => home.parent().unwrap().parent().unwrap().to_path_buf(),
            _ => s.path("host/container_home/akao_xy"),
        };
        s.put("scenario.json", &json!({"workers":{},"mounts":[mount]}).to_string());
        let (code, out) = s.run(&["rm", "fakebox", "x", "--home"]);
        if shape == "prefix" {
            assert_eq!(code, 0, "{out}");
            assert!(!home.exists());
        } else {
            assert_ne!(code, 0, "{out}");
            assert!(home.is_dir());
            assert!(out.contains("kept"), "{out}");
        }
    }
    for fail in ["census_fail", "mount_inspect_fail", "ssh"] {
        let s = sandbox(fail, json!({"workers":{},"mounts":["/unrelated"]}));
        let home = s.path("host/container_home/akao_x");
        fs::create_dir_all(&home).unwrap();
        if fail != "ssh" {
            s.put(
                "scenario.json",
                &json!({"workers":{},"mounts":["/unrelated"],fail:true}).to_string(),
            );
        }
        let mut command = s.command(&["rm", "fakebox", "x", "--home"]);
        if fail == "ssh" {
            command.env("FAKE_SSH_FAIL", "1");
        }
        let (code, out) = output(&mut command);
        assert_ne!(code, 0, "{out}");
        assert!(home.is_dir());
        assert!(!out.contains(&format!("{}: not there", home.display())), "{out}");
    }
}

#[test]
fn rm_home_dry_run_missing_homes_and_failed_retirement_preserve_data() {
    let s = sandbox("rm-home-dry", json!({"workers":{}}));
    s.put("host/container_home/akao_x/history", "preserved\n");
    let (code, out) = s.run(&["rm", "fakebox", "x", "--home", "--dry-run"]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(text(s.path("host/container_home/akao_x/history")), "preserved\n");
    let (code, out) = s.run(&["rm", "fakebox", "missing", "--home"]);
    assert_eq!(code, 0, "{out}");
    s.put(
        "scenario.json",
        &json!({"workers":{},"mounts":[s.path("host/container_home")]}).to_string(),
    );
    let (code, out) = s.run(&["rm", "fakebox", "missing", "--home"]);
    assert_eq!(code, 0, "{out}");
    assert!(!s.events("ssh.jsonl").iter().flatten().any(|a| a.contains("rm -rf")));
    let s = sandbox(
        "rm-home-failed",
        json!({"workers":{"akao_x":{"id":"old-id","status":"exited","inspect_fail":true}}}),
    );
    s.put("host/container_home/akao_x/history", "preserved\n");
    let (code, out) = s.run(&["rm", "fakebox", "x", "--home"]);
    assert_ne!(code, 0, "{out}");
    assert_eq!(text(s.path("host/container_home/akao_x/history")), "preserved\n");
}
