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
            .env(
                "PATH",
                format!("{}:{}", self.root.join("bin").display(), std::env::var("PATH").unwrap()),
            )
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

#[test]
fn remote_root_only_artifacts_are_read_with_available_escalation() {
    let sb = Sandbox::new("protected-read");
    // Model a non-root SSH account whose tar cannot read the artifact without sudo.
    // No privileges or host permissions are changed by these stand-ins.
    let tar = Command::new("sh").args(["-c", "command -v tar"]).output().unwrap();
    assert!(tar.status.success());
    let real_tar = String::from_utf8(tar.stdout).unwrap();
    sb.put("bin/id", "#!/bin/sh\nprintf '1000\\n'\n");
    sb.put(
        "bin/sudo",
        "#!/bin/sh\n[ \"$1\" = -n ] || exit 91\nshift\nexport AKAO_CP_TEST_ELEVATED=1\nexec \"$@\"\n",
    );
    sb.put(
        "bin/tar",
        &format!(
            "#!/bin/sh\nfor arg do\n  if [ \"$arg\" = -czf ] && [ \"${{AKAO_CP_TEST_ELEVATED:-}}\" != 1 ]; then\n    echo 'protected source: permission denied' >&2\n    exit 92\n  fi\ndone\nexec {} \"$@\"\n",
            real_tar.trim()
        ),
    );
    for name in ["id", "sudo", "tar"] {
        fs::set_permissions(sb.root.join("bin").join(name), fs::Permissions::from_mode(0o755)).unwrap();
    }
    let dst = sb.root.join("collected");
    let (code, out) = sb.cp(&["boxa:ww42/a0/0003_x", dst.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(fs::read_to_string(dst.join("0003_x/result.txt")).unwrap(), "42\n");
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
