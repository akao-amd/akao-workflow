//! `akao cp` between two scratch boxes whose homes are sandbox dirs, through a stand-in ssh
//! (AKAO_SSH) that runs the remote script here: what lands where, and what is refused.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

/// ssh: drop the options and the nick, run the command (`sh -c '<script>'`) locally.
const FAKE_SSH: &str = r#"#!/bin/sh
while [ "$1" != -- ]; do shift; done
shift
exec sh -c "$*"
"#;

struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Sandbox {
        let root = std::env::temp_dir().join(format!("akao-cp-it-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let sb = Sandbox { root };
        let (a, b) = (sb.root.join("a"), sb.root.join("b"));
        sb.put(
            "state/hosts.tsv",
            &format!("boxa\t/m\t-\t{}\nboxb\t/m\t-\t{}\n", a.display(), b.display()),
        );
        sb.put("state/config.toml", "");
        sb.put("bin/ssh", FAKE_SSH);
        fs::set_permissions(sb.root.join("bin/ssh"), fs::Permissions::from_mode(0o755)).unwrap();
        sb.put(&format!("a/{}/ww42/a0/0003_x/result.txt", year()), "42\n");
        sb
    }

    fn put(&self, rel: &str, text: &str) {
        let p = self.root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, text).unwrap();
    }

    fn cp(&self, args: &[&str]) -> (i32, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_akao"))
            .arg("cp")
            .args(args)
            .env("AKAO_CONFIG_ROOT", self.root.join("state"))
            .env("AKAO_SSH", self.root.join("bin/ssh"))
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code().unwrap_or(-1), text)
    }

    /// A path on box `b` under its `<host_home>/<year>`.
    fn on_b(&self, rel: &str) -> PathBuf {
        self.root.join("b").join(year()).join(rel)
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// The ISO year akao uses (its work year), without a date crate in the test.
fn year() -> String {
    let out = Command::new("date").arg("+%G").output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn a_bare_nick_destination_keeps_the_sources_path() {
    let sb = Sandbox::new("same");
    let (code, out) = sb.cp(&["boxa:ww42/a0/0003_x", "boxb:"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("(the source's path)"), "{out}");
    assert_eq!(
        fs::read_to_string(sb.on_b("ww42/a0/0003_x/result.txt")).unwrap(),
        "42\n"
    );

    // A whole worker root lands beside the receiving box's own roots.
    let (code, out) = sb.cp(&["boxa:ww42/a0", "boxb:"]);
    assert_eq!(code, 0, "{out}");
    assert!(sb.on_b("ww42/a0/0003_x/result.txt").is_file());
    assert!(!sb.on_b("a0").exists(), "{out}");
}

#[test]
fn an_explicit_destination_is_taken_as_given() {
    let sb = Sandbox::new("explicit");
    let (code, out) = sb.cp(&["boxa:ww42/a0/0003_x", "boxb:ww42/b0/../a0"]);
    assert_eq!(code, 0, "{out}");
    assert!(sb.on_b("ww42/a0/0003_x/result.txt").is_file());
    assert!(!out.contains("(the source's path)"), "{out}");
}

#[test]
fn a_bare_nick_destination_refuses_what_has_no_path_to_keep() {
    let sb = Sandbox::new("refuse");
    // The source's own box: it is already there, and tar would read what it overwrites.
    let (code, out) = sb.cp(&["boxa:ww42/a0/0003_x", "boxa:"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("already at that path on boxa"), "{out}");
    // A local source outside /<year>/ has no path on a box.
    let local = sb.root.join("a").join(year()).join("ww42/a0/0003_x");
    let (code, out) = sb.cp(&[local.to_str().unwrap(), "boxb:"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("name the destination path"), "{out}");
    assert!(!sb.on_b("ww42").exists(), "{out}");
}
