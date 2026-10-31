//! A failed Docker connection is preserved; only a definite stopped service adds a hint.
mod common;
#[path = "common/init.rs"]
mod fixture;
use common::output;

#[test]
fn stopped_docker_adds_a_hint_without_starting_any_service() {
    for status in ["inactive", "failed"] {
        let s = fixture::sandbox(status);
        let (code, out) = output(
            s.command(&["init", "fakebox", "worker", "--image", "img:tag", "--dry-run"])
                .env("FAKE_DOCKER_DOWN", "1")
                .env("FAKE_DAEMON", status),
        );
        assert_ne!(code, 0, "{out}");
        assert!(out.contains(&format!("docker is {status} on fakebox")), "{out}");
        assert!(out.contains("if starting it is authorized"), "{out}");
        assert!(out.contains("original Docker transport error"), "{out}");
        assert!(out.contains("command failed"), "{out}");
        assert!(!s
            .events("ssh.jsonl")
            .iter()
            .flatten()
            .any(|a| a.contains("systemctl start")));
        assert_eq!(s.events("docker.jsonl").len(), 1);
    }
}

#[test]
fn unknown_active_or_failed_secondary_probes_preserve_the_original_error() {
    for status in ["active", "activating", "unknown", ""] {
        let s = fixture::sandbox(&format!("daemon-{status}"));
        let (code, out) = output(
            s.command(&["init", "fakebox", "worker", "--image", "img:tag", "--dry-run"])
                .env("FAKE_DOCKER_DOWN", "1")
                .env("FAKE_DAEMON", status),
        );
        assert_ne!(code, 0, "{out}");
        assert!(out.contains("original Docker transport error"), "{out}");
        assert!(!out.contains("if starting it is authorized"), "{out}");
    }
    for failure in ["FAKE_DIAG_FAIL", "FAKE_REMOVE_SSH"] {
        let s = fixture::sandbox(failure);
        let (code, out) = output(
            s.command(&["init", "fakebox", "worker", "--image", "img:tag", "--dry-run"])
                .env("FAKE_DOCKER_DOWN", "1")
                .env(failure, "1"),
        );
        assert_ne!(code, 0, "{out}");
        assert!(out.contains("original Docker transport error"), "{out}");
        assert!(out.contains("command failed"), "{out}");
        assert!(!out.contains("cannot start"), "{out}");
        assert!(!out.contains("if starting it is authorized"), "{out}");
    }
}

#[test]
fn a_login_outside_the_sockets_group_is_named_with_the_fix() {
    let s = fixture::sandbox("sock-denied");
    let (code, out) = output(
        s.command(&["init", "fakebox", "worker", "--image", "img:tag", "--dry-run"])
            .env("FAKE_DOCKER_DOWN", "1")
            .env("FAKE_SOCK_DENIED", "1"),
    );
    assert_ne!(code, 0, "{out}");
    assert!(
        out.contains("amd on fakebox may not use /var/run/docker.sock (root:docker)"),
        "{out}"
    );
    assert!(out.contains("sudo usermod -aG docker amd"), "{out}");
    assert!(out.contains("original Docker transport error"), "{out}");
    assert!(!s.events("ssh.jsonl").iter().flatten().any(|a| a.contains("usermod")));
}
