//! `akao mirror` end to end under --dry-run, against a scratch InferenceX repo and stand-ins
//! for ssh (AKAO_SSH) and docker (PATH): what it would run on a box, and what it refuses
//! before touching one.  No network, no real docker.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

const MASTER: &str = "m1-fp4-mi355x-atom:
  image: rocm/atom:1
  model: org/M1-120B
  model-prefix: m1
  runner: mi355x
  precision: fp4
  framework: atom
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - { tp: 1, conc-start: 4, conc-end: 8 }
";

/// ssh: `-G` resolves; the KFD read answers $FAKE_GFX; `test -d` (the container home) fails.
const FAKE_SSH: &str = r#"#!/bin/sh
if [ "$1" = -G ]; then echo "hostname fakebox.example"; exit 0; fi
case "$*" in
    *kfd/topology*) echo "Welcome to fakebox 2026"; for v in $FAKE_GFX; do echo "OAKA_GFX=$v"; done ;;
    *"test -d"*) exit 1 ;;
esac
exit 0
"#;

/// docker: unreachable when $FAKE_DOCKER_DOWN is set; $FAKE_EXISTING describes an existing
/// akao_m1-fp4-mi355x-atom (inspect lines separated by |: id, image, workdir, mount/env
/// lines), running, until $FAKE_REPLACED_ID says another container took its name; without
/// it there is no container.
const FAKE_DOCKER: &str = r#"#!/bin/sh
case "$*" in
    *" version "*)
        [ -z "$FAKE_DOCKER_DOWN" ] || { echo "error during connect" >&2; exit 1; }
        echo 29.0 ;;
    *"ps -a"*) [ -z "$FAKE_EXISTING" ] || echo akao_m1-fp4-mi355x-atom ;;
    *"{{.State.Status}}"*)
        [ -n "$FAKE_EXISTING" ] || exit 1
        echo "${FAKE_REPLACED_ID:-${FAKE_EXISTING%%|*}} running" ;;
    *"container inspect"*)
        [ -n "$FAKE_EXISTING" ] || exit 1
        echo "$FAKE_EXISTING" | tr '|' '\n' ;;
    *"context inspect"*) exit 1 ;;
esac
exit 0
"#;

struct Sandbox {
    root: PathBuf,
    rev: String,
}

impl Sandbox {
    fn new(name: &str) -> Sandbox {
        let root = std::env::temp_dir().join(format!("akao-mirror-it-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let put = |rel: &str, text: &str, exe: bool| {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, text).unwrap();
            if exe {
                fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
            }
        };
        let infx = root.join("infx");
        put("infx/configs/amd-master.yaml", MASTER, false);
        put(
            "infx/benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x_atom.sh",
            "python3 -m atom.entrypoints.openai_server --model $MODEL\n",
            false,
        );
        let git = |args: &[&str]| {
            let out = Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                .args(args)
                .current_dir(&infx)
                .output()
                .unwrap();
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        git(&["init", "-q"]);
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "configs"]);
        let rev = git(&["rev-parse", "HEAD"]);
        put(
            "state/config.toml",
            &format!(
                "default_image = 'img:default'\ninfx_local = '{}'\ndeploy_src = '{}'\ndeploy_paths = 'x'\n",
                infx.display(),
                root.join("deploy").display()
            ),
            false,
        );
        put("deploy/x", "", false);
        put("state/hosts.tsv", "fakebox\t-\t/m\t-\t/h\n", false);
        put("state/container_home/.bashrc", "", false);
        // The repo init ships (AKAO_REPO_ROOT): year/CLAUDE.md with AGENTS.md a link to it.
        put("repo/year/CLAUDE.md", "# Working under /<year>\n", false);
        put("repo/oaka/bin/oaka", "#!/bin/sh\n", true);
        std::os::unix::fs::symlink("CLAUDE.md", root.join("repo/year/AGENTS.md")).unwrap();
        let out = Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "init.defaultBranch=main",
            ])
            .args(["init", "-q"])
            .current_dir(root.join("repo"))
            .output()
            .unwrap();
        assert!(out.status.success());
        put("bin/ssh", FAKE_SSH, true);
        put("bin/docker", FAKE_DOCKER, true);
        Sandbox { root, rev }
    }

    fn mirror(&self, args: &[&str], env: &[(&str, &str)]) -> (i32, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_akao"))
            .arg("mirror")
            .args(args)
            .env("AKAO_CONFIG_ROOT", self.root.join("state"))
            .env("AKAO_SSH", self.root.join("bin/ssh"))
            .env("PATH", format!("{}:/usr/bin:/bin", self.root.join("bin").display()))
            .env_remove("AKAO_ARTIFACT_ROOT")
            .env("AKAO_REPO_ROOT", self.root.join("repo"))
            .envs(env.iter().copied())
            .output()
            .unwrap();
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

#[test]
fn mirror_brings_up_the_entrys_image_with_a_brief() {
    let sb = Sandbox::new("up");
    let year = chrono_year();
    let (code, out) = sb.mirror(
        &[
            "fakebox",
            "--conf",
            "m1,atom",
            "--rev",
            &sb.rev,
            "--week",
            "ww07",
            "--dry-run",
        ],
        &[("FAKE_GFX", "90500 90500")],
    );
    assert_eq!(code, 0, "{out}");
    let root = format!("/{year}/ww07/m1-fp4-mi355x-atom");
    for want in [
        "fakebox: 2 x gfx950 as mi355x needs".to_string(),
        "image rocm/atom:1".to_string(),
        format!("-e 'AKAO_ARTIFACT_ROOT={root}' -e 'AKAO_REPO_ROOT=/root/akao-workflow' -w {root}"),
        // The orientation file from the repo, dereferenced, into the box's /<year>.
        "year -h '--owner=0' '--group=0' -czf - CLAUDE.md AGENTS.md | [fakebox]".to_string(),
        "[7/12] install packages".to_string(),
        "[9/12] install tools and agents".to_string(),
        "bash /root/akao-workflow/utils/install_gh.sh".to_string(),
        "bundle create - --branches --tags | docker --context fakebox exec -i akao_m1-fp4-mi355x-atom".to_string(),
        "cat >/root/.cache/akao-workflow.bundle".to_string(),
        // The hook's static oaka, beside the bundle (git-ignored), into the clone's oaka/bin.
        "oaka/bin/oaka | docker --context fakebox exec -i akao_m1-fp4-mi355x-atom".to_string(),
        "ln -sf /root/akao-workflow/oaka/bin/oaka /usr/local/bin/oaka".to_string(),
        "rocm/atom:1 sleep infinity".to_string(),
        "[mirror] brief".to_string(),
        format!("[fakebox] if [ \"$(id -u)\" -eq 0 ]; then S=; else S=\"sudo -n\"; fi; $S mkdir -p /h{root}"),
        format!(
            "done. the worker's brief: {root}/mirror/m1-fp4-mi355x-atom@{}/MIRROR.md",
            &sb.rev[..12]
        ),
    ] {
        assert!(out.contains(&want), "{want:?} not in {out}");
    }
}

/// FAKE_EXISTING for akao_m1-fp4-mi355x-atom as akao made it on fakebox (home /h).
fn existing(image: &str, root: &str) -> String {
    let year = root.split('/').nth(1).unwrap();
    format!(
        "cafe1234|{image}|{root}|mount /model /m|mount /{year} /h/{year}|mount /root /h/container_home/akao_m1-fp4-mi355x-atom|\
         env PATH=/usr/bin|env AKAO_ARTIFACT_ROOT={root}"
    )
}

#[test]
fn mirror_reuses_its_own_container() {
    let sb = Sandbox::new("reuse");
    let root = format!("/{}/ww07/m1-fp4-mi355x-atom", chrono_year());
    let (code, out) = sb.mirror(
        &["fakebox", "--conf", "m1,atom", "--rev", &sb.rev, "--dry-run"],
        &[
            ("FAKE_GFX", "90500"),
            ("FAKE_EXISTING", &existing("rocm/atom:1", &root)),
        ],
    );
    assert_eq!(code, 0, "{out}");
    // Its root wins over this week's.
    assert!(
        out.contains(&format!(
            "akao_m1-fp4-mi355x-atom exists: image rocm/atom:1, artifact root {root}"
        )),
        "{out}"
    );
    assert!(out.contains("reusing running akao_m1-fp4-mi355x-atom"), "{out}");
    assert!(
        out.contains(&format!("$S mkdir -p /h{root} && $S tar -xzf - -C /h{root}")),
        "{out}"
    );
    // Replaced under the same name between the checks and the reuse (another controller).
    let (code, out) = sb.mirror(
        &["fakebox", "--conf", "m1,atom", "--rev", &sb.rev, "--dry-run"],
        &[
            ("FAKE_GFX", "90500"),
            ("FAKE_EXISTING", &existing("rocm/atom:1", &root)),
            ("FAKE_REPLACED_ID", "beef5678"),
        ],
    );
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("akao_m1-fp4-mi355x-atom changed since step 1"), "{out}");
}

#[test]
fn mirror_refuses_before_touching_the_box() {
    let sb = Sandbox::new("refuse");
    let args = [
        "fakebox",
        "--conf",
        "m1,atom",
        "--rev",
        &sb.rev,
        "--week",
        "ww07",
        "--dry-run",
    ];
    let root = format!("/{}/ww07/m1-fp4-mi355x-atom", chrono_year());
    for (env, why) in [
        (
            vec![("FAKE_GFX", "90402")],
            "fakebox has 1 x gfx942, but m1-fp4-mi355x-atom runs on mi355x (gfx950)",
        ),
        (vec![("FAKE_GFX", "")], "fakebox shows no AMD GPUs"),
        (
            vec![("FAKE_GFX", "90500"), ("FAKE_DOCKER_DOWN", "1")],
            "command failed: docker -H ssh://fakebox version",
        ),
    ] {
        let (code, out) = sb.mirror(&args, &env);
        assert_eq!(code, 1, "{out}");
        assert!(out.contains(why), "{why:?} not in {out}");
        assert!(!out.contains("mkdir"), "{out}");
    }
    // A container of that name already runs another image, or mounts another host home.
    let other = existing("img:default", &root);
    let (code, out) = sb.mirror(&args, &[("FAKE_GFX", "90500"), ("FAKE_EXISTING", &other)]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("runs img:default, not the requested rocm/atom:1"), "{out}");
    assert!(!out.contains("mkdir"), "{out}");
    let moved = existing("rocm/atom:1", &root).replace(" /h/", " /h1/");
    let (code, out) = sb.mirror(&args, &[("FAKE_GFX", "90500"), ("FAKE_EXISTING", &moved)]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("but hosts.tsv now puts it at /h/"), "{out}");
    assert!(!out.contains("mkdir"), "{out}");
    // Without a host: a preview, nothing run.
    let (code, out) = sb.mirror(&["--conf", "m1-fp4-mi355x-atom", "--rev", &sb.rev], &[]);
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("# Mirror of InferenceX `m1-fp4-mi355x-atom`") && out.contains("Preview"),
        "{out}"
    );
    assert!(!out.contains("[fakebox]"), "{out}");
}

/// The ISO year akao uses (its work year), without a date crate in the test.
fn chrono_year() -> String {
    let out = Command::new("date").arg("+%G").output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
