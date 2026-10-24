//! Runner transport defaults and override precedence, without making an SSH connection.
mod common;
use common::{output, Sandbox};

const SSH: &str = r#"#!/usr/bin/python3 -I
import json, os, sys
from pathlib import Path
with (Path(os.environ['FAKE_ROOT'])/'ssh.jsonl').open('a') as f:
    f.write(json.dumps(sys.argv[1:])+'\n')
print('stand-in refuses connection', file=sys.stderr)
sys.exit(23)
"#;

#[test]
fn default_ssh_uses_timeouts_and_the_config_without_batch_mode() {
    let s = Sandbox::new("ssh-default");
    s.exe("bin/ssh", SSH);
    s.put("state/.ssh/config", "Host fakebox\n ConnectTimeout 90\n");
    let local = s.path("local");
    let local = local.to_str().unwrap();
    let (code, out) = output(
        s.command(&["cp", "fakebox:ww43/x", local, "--dry-run"])
            .env_remove("AKAO_SSH"),
    );
    assert_eq!(code, 0, "{out}");
    // Dry-run suppresses the pipe, but its printed argv shows all transport options.
    assert!(out.contains("fakebox"), "{out}");
    assert!(s.events("ssh.jsonl").is_empty());
    let (code, out) = output(s.command(&["cp", "fakebox:ww43/x", local]).env_remove("AKAO_SSH"));
    assert_ne!(code, 0, "{out}");
    let calls = s.events("ssh.jsonl");
    let args = &calls[0];
    for opt in [
        "ConnectTimeout=10",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=4",
        "RequestTTY=no",
        "RemoteCommand=none",
    ] {
        assert!(args.iter().any(|a| a == opt), "{args:?}");
    }
    assert!(!args.iter().any(|a| a.contains("BatchMode")), "{args:?}");
    assert!(args
        .windows(2)
        .any(|a| a[0] == "-F" && a[1] == s.path("state/.ssh/config").to_str().unwrap()));
}

#[test]
fn explicit_ssh_override_replaces_defaults_and_automatic_config() {
    let s = Sandbox::new("ssh-override");
    s.exe("bin/ssh", SSH);
    s.put("state/.ssh/config", "Host fakebox\n HostName example.invalid\n");
    let override_argv = "ssh -o ConnectTimeout=2 -o ServerAliveInterval=3 -o ServerAliveCountMax=1 -F /dev/null";
    let local = s.path("local");
    let (code, out) = output(
        s.command(&["cp", "fakebox:ww43/x", local.to_str().unwrap()])
            .env("AKAO_SSH", override_argv),
    );
    assert_ne!(code, 0, "{out}");
    let calls = s.events("ssh.jsonl");
    let args = &calls[0];
    assert!(args.iter().any(|a| a == "ConnectTimeout=2"));
    assert!(!args.iter().any(|a| a == "ConnectTimeout=10"));
    assert!(args.windows(2).any(|a| a == ["-F", "/dev/null"]));
    assert!(!args.iter().any(|a| a == s.path("state/.ssh/config").to_str().unwrap()));
}

#[test]
fn default_ssh_without_a_config_keeps_timeouts_and_omits_f() {
    let s = Sandbox::new("ssh-no-config");
    s.exe("bin/ssh", SSH);
    let local = s.path("local");
    let (code, out) = output(
        s.command(&["cp", "fakebox:ww43/x", local.to_str().unwrap()])
            .env_remove("AKAO_SSH"),
    );
    assert_ne!(code, 0, "{out}");
    let calls = s.events("ssh.jsonl");
    assert!(calls[0].iter().any(|a| a == "ConnectTimeout=10"));
    assert!(!calls[0].iter().any(|a| a == "-F"));
}

#[test]
fn ssh_defaults_override_host_config_in_the_effective_local_configuration() {
    let s = Sandbox::new("ssh-effective");
    s.put("config", "Host fakebox\n HostName example.invalid\n ConnectTimeout 90\n ServerAliveInterval 99\n ServerAliveCountMax 9\n");
    let out = std::process::Command::new("/usr/bin/ssh")
        .args([
            "-G",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            "-F",
        ])
        .arg(s.path("config"))
        .arg("fakebox")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for line in ["connecttimeout 10", "serveraliveinterval 15", "serveralivecountmax 4"] {
        assert!(text.lines().any(|l| l == line), "{text}");
    }
}
