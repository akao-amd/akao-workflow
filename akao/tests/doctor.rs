//! `akao doctor` against a scratch AKAO_CONFIG_ROOT: it passes on a sound setup and names
//! what is wrong otherwise.  Needs /usr/bin/ssh (only `ssh -G`, no network).

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The PATH wrapper from the home template (container_home/.local/bin/ssh).
const SSH_WRAPPER: &str = r#"#!/bin/bash
cfg="${AKAO_CONFIG_ROOT:+$AKAO_CONFIG_ROOT/.ssh/config}"
if [ -n "$cfg" ] && [ -f "$cfg" ]; then
    exec /usr/bin/ssh -F "$cfg" "$@"
fi
exec /usr/bin/ssh "$@"
"#;

struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Sandbox {
        let root = std::env::temp_dir().join(format!("akao-doctor-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let sb = Sandbox { root };
        let state = sb.state();
        let deploy = sb.root.join("deploy");
        sb.write(
            "state/config.toml",
            &format!(
                "default_image = 'img:tag'\ndeploy_src = '{}'\ndeploy_paths = 'x'\n",
                deploy.display()
            ),
        );
        sb.write("state/hosts.tsv", "fakebox\t-\t/m\t-\t/h\n");
        sb.write(
            "state/.ssh/config",
            "Host fakebox\n    HostName 10.9.8.7\n    User akao\n",
        );
        sb.exe("state/container_home/.local/bin/ssh", SSH_WRAPPER);
        sb.write("deploy/x", "");
        sb.exe("repo/oaka/bin/oaka", "#!/bin/sh\necho 'oaka 0.0.0 (test)'\n");
        sb.write("repo/year/CLAUDE.md", "# Working under /<year>\n");
        let git = Command::new("git")
            .args(["init", "-q"])
            .current_dir(sb.root.join("repo"))
            .status()
            .unwrap();
        assert!(git.success());
        sb.exe("bin/ssh", SSH_WRAPPER);
        sb.exe("bin/docker", "#!/bin/sh\necho 'Docker version 0.0 (fake)'\n");
        sb.exe("bin-nowrap/docker", "#!/bin/sh\necho 'Docker version 0.0 (fake)'\n");
        assert!(state.is_dir());
        sb
    }

    fn state(&self) -> PathBuf {
        self.root.join("state")
    }

    fn write(&self, rel: &str, text: &str) {
        let p = self.root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, text).unwrap();
    }

    fn exe(&self, rel: &str, text: &str) {
        self.write(rel, text);
        fs::set_permissions(self.root.join(rel), fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// `akao doctor` with `bin_dir` first on PATH (system dirs after it).
    fn doctor(&self, bin_dir: &str, config_root: Option<&Path>) -> (i32, String) {
        let mut c = Command::new(env!("CARGO_BIN_EXE_akao"));
        c.arg("doctor")
            .env("PATH", format!("{}:/usr/bin:/bin", self.root.join(bin_dir).display()))
            .env_remove("AKAO_ARTIFACT_ROOT")
            .env("AKAO_REPO_ROOT", self.root.join("repo"));
        match config_root {
            Some(r) => c.env("AKAO_CONFIG_ROOT", r),
            None => c.env_remove("AKAO_CONFIG_ROOT"),
        };
        let out: Output = c.output().unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code().unwrap_or(-1), text)
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn have_ssh() -> bool {
    let ok = Path::new("/usr/bin/ssh").exists();
    if !ok {
        eprintln!("skipped: no /usr/bin/ssh");
    }
    ok
}

#[test]
fn sound_setup_passes() {
    if !have_ssh() {
        return;
    }
    let sb = Sandbox::new("ok");
    let (code, out) = sb.doctor("bin", Some(&sb.state()));
    assert_eq!(code, 0, "{out}");
    for want in [
        "ok    hosts          fakebox",
        "resolves every host through",
        "ok    oaka",
        "ok    docker",
        "ok    repo",
    ] {
        assert!(out.contains(want), "{want:?} not in {out}");
    }
}

#[test]
fn ssh_without_the_wrapper_fails() {
    if !have_ssh() {
        return;
    }
    let sb = Sandbox::new("nowrap");
    let (code, out) = sb.doctor("bin-nowrap", Some(&sb.state()));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("FAIL  ssh            /usr/bin/ssh ignores"), "{out}");
    assert!(out.contains("for fakebox"), "{out}");
}

#[test]
fn missing_pieces_are_named() {
    let sb = Sandbox::new("missing");
    let (code, out) = sb.doctor("bin", None);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("FAIL  config root    AKAO_CONFIG_ROOT is not set"),
        "{out}"
    );

    fs::remove_file(sb.root.join("repo/oaka/bin/oaka")).unwrap();
    fs::remove_file(sb.root.join("deploy/x")).unwrap();
    fs::remove_dir_all(sb.state().join("container_home")).unwrap();
    fs::remove_file(sb.root.join("repo/year/CLAUDE.md")).unwrap();
    let (code, out) = sb.doctor("bin", Some(&sb.state()));
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("FAIL  deploy         missing under"), "{out}");
    assert!(out.contains("FAIL  oaka"), "{out}");
    assert!(out.contains("FAIL  home template"), "{out}");
    assert!(out.contains("FAIL  repo"), "{out}");
}

#[test]
fn controller_prerequisites_are_reported() {
    let sb = Sandbox::new("controller");
    let (code, out) = sb.doctor("bin", Some(&sb.state()));
    assert_eq!(code, 0, "{out}");
    // Not on the sandbox's PATH (system dirs aside): warned, never fatal.
    assert!(out.contains("artifact root  /"), "{out}");
    assert!(out.contains("(derived)"), "{out}");
    assert!(out.contains(" inferencex "), "{out}");
    // A relative AKAO_ARTIFACT_ROOT is an error everywhere akao reads it.
    let out = Command::new(env!("CARGO_BIN_EXE_akao"))
        .arg("doctor")
        .env("PATH", format!("{}:/usr/bin:/bin", sb.root.join("bin").display()))
        .env("AKAO_CONFIG_ROOT", sb.state())
        .env("AKAO_ARTIFACT_ROOT", "relative/dir")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(
        text.contains("FAIL  artifact root  AKAO_ARTIFACT_ROOT=relative/dir must be an absolute path"),
        "{text}"
    );
    let ok = Command::new(env!("CARGO_BIN_EXE_akao"))
        .arg("doctor")
        .env("PATH", format!("{}:/usr/bin:/bin", sb.root.join("bin").display()))
        .env("AKAO_CONFIG_ROOT", sb.state())
        .env("AKAO_ARTIFACT_ROOT", &sb.root)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&ok.stdout);
    assert!(
        text.contains(&format!(
            "ok    artifact root  {} (AKAO_ARTIFACT_ROOT)",
            sb.root.display()
        )),
        "{text}"
    );
}
