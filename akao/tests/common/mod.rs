//! Shared scratch directories and command capture for hermetic akao acceptance tests.
#![allow(dead_code)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Sandbox {
    pub root: PathBuf,
}

impl Sandbox {
    pub fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "akao-accept-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let s = Self { root };
        s.put("state/config.toml", "");
        s.put("state/container_home/.bashrc", "# scratch home template\n");
        s.put(
            "state/hosts.tsv",
            &format!("fakebox\t/model\t-\t{}\n", s.path("host").display()),
        );
        fs::create_dir_all(s.path("worker")).unwrap();
        fs::create_dir_all(s.path("repo/.git")).unwrap();
        s.put("repo/CLAUDE.md", "# scratch repo\n");
        s.exe("repo/oaka/bin/oaka", "#!/bin/sh\nexit 0\n");
        for tool in ["docker", "ssh", "tmux"] {
            s.exe(
                &format!("bin/{tool}"),
                "#!/bin/sh\necho 'unconfigured stand-in' >&2\nexit 99\n",
            );
        }
        s
    }

    pub fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    pub fn put(&self, rel: &str, body: &str) {
        let path = self.path(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }

    pub fn exe(&self, rel: &str, body: &str) {
        self.put(rel, body);
        fs::set_permissions(self.path(rel), fs::Permissions::from_mode(0o755)).unwrap();
    }

    pub fn command(&self, args: &[&str]) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_akao"));
        c.args(args)
            .env("AKAO_CONFIG_ROOT", self.path("state"))
            .env("AKAO_REPO_ROOT", self.path("repo"))
            .env("AKAO_ARTIFACT_ROOT", self.path("worker"))
            .env("AKAO_SSH", self.path("bin/ssh"))
            .env("PATH", format!("{}:/usr/bin:/bin", self.path("bin").display()))
            .env("FAKE_ROOT", &self.root)
            .env_remove("TMUX")
            .env_remove("TMUX_PANE");
        c
    }

    pub fn run(&self, args: &[&str]) -> (i32, String) {
        output(&mut self.command(args))
    }

    pub fn events(&self, rel: &str) -> Vec<Vec<String>> {
        fs::read_to_string(self.path(rel))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}

pub fn output(command: &mut Command) -> (i32, String) {
    let out = command.output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

pub fn text(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path).unwrap()
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
