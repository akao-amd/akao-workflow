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

/// ssh: `-G` resolves; the KFD read answers $FAKE_GFX; `test -d` (the container home) fails;
/// with $FAKE_NVIDIA set the box has no /dev/kfd (its /dev/nvidiactl answers like any probe).
const FAKE_SSH: &str = r#"#!/bin/sh
if [ "$1" = -G ]; then echo "hostname fakebox.example"; exit 0; fi
case "$*" in
    *kfd/topology*) echo "Welcome to fakebox 2026"; for v in $FAKE_GFX; do echo "OAKA_GFX=$v"; done ;;
    *"test -d"*) exit 1 ;;
    *"test -e /dev/kfd"*) [ -z "$FAKE_NVIDIA" ] ;;
esac
exit $?
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
                "infx_local = '{}'\ndeploy_src = '{}'\ndeploy_paths = 'x'\n",
                infx.display(),
                root.join("deploy").display()
            ),
            false,
        );
        put("deploy/x", "", false);
        put("state/hosts.tsv", "fakebox\t/m\t-\t/h\n", false);
        put("state/container_home/.bashrc", "", false);
        // The repo init ships (AKAO_REPO_ROOT): CLAUDE.md with AGENTS.md a link to it.
        put("repo/CLAUDE.md", "# akao-workflow\n", false);
        put("repo/oaka/bin/oaka", "#!/bin/sh\n", true);
        std::os::unix::fs::symlink("CLAUDE.md", root.join("repo/AGENTS.md")).unwrap();
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
        self.akao("mirror", args, env)
    }

    fn akao(&self, cmd: &str, args: &[&str], env: &[(&str, &str)]) -> (i32, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_akao"))
            .arg(cmd)
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
fn init_takes_its_image_which_an_existing_container_must_run() {
    let sb = Sandbox::new("init-image");
    let args = ["fakebox", "m1-fp4-mi355x-atom", "--image", "img:new", "--dry-run"];
    let (code, out) = sb.akao("init", &args, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("\n  image img:new\n"), "{out}");
    assert!(out.contains(" --entrypoint sleep img:new infinity"), "{out}");

    let year = chrono_year();
    let old = existing("img:old", &format!("/{year}/ww07/m1-fp4-mi355x-atom"));
    let (code, out) = sb.akao("init", &args, &[("FAKE_EXISTING", &old)]);
    assert_ne!(code, 0, "{out}");
    assert!(
        out.contains("runs img:old, not the requested img:new; pass --image img:old"),
        "{out}"
    );

    // No image, no default to fall back on.
    let (code, out) = sb.akao("init", &["fakebox", "x", "--dry-run"], &[]);
    assert_ne!(code, 0, "{out}");
    assert!(out.contains("--image <IMAGE>"), "{out}");
}

#[test]
fn init_runs_nvidia_boxes_with_gpus_all() {
    let sb = Sandbox::new("init-nvidia");
    let args = ["fakebox", "gb", "--image", "img:tag", "--dry-run"];
    let (code, out) = sb.akao("init", &args, &[("FAKE_NVIDIA", "1")]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("NVIDIA GPUs"), "{out}");
    assert!(out.contains(" run --rm -d --gpus all '--ipc=host'"), "{out}");
    assert!(!out.contains("--device=/dev/kfd"), "{out}");
    // Launched only: no setup steps, a shell to enter.
    assert!(out.contains("[6/6] container"), "{out}");
    assert!(!out.contains("[7/"), "{out}");
    assert!(out.contains("docker --context fakebox exec -it akao_gb bash"), "{out}");
    let (code, out) = sb.akao("init", &args, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains(" run --rm -d --privileged"), "{out}");
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
        "[3/12] deploy extra paths".to_string(),
        "[7/12] install packages".to_string(),
        "[9/12] install tools and agents".to_string(),
        "bash /root/akao-workflow/utils/install_gh.sh".to_string(),
        "bundle create - --branches --tags | docker --context fakebox exec -i akao_m1-fp4-mi355x-atom".to_string(),
        "cat >/root/.cache/akao-workflow.bundle".to_string(),
        // The hook's static oaka, beside the bundle (git-ignored), into the clone's oaka/bin.
        "oaka/bin/oaka | docker --context fakebox exec -i akao_m1-fp4-mi355x-atom".to_string(),
        "ln -sf /root/akao-workflow/oaka/bin/oaka /usr/local/bin/oaka".to_string(),
        "--entrypoint sleep rocm/atom:1 infinity".to_string(),
        "[mirror] brief".to_string(),
        format!(
            "[fakebox] if [ \"$(id -u)\" -ne 0 ] && sudo -n true 2>/dev/null; then S=\"sudo -n\"; else S=; fi; \
             $S mkdir -p /h{root}"
        ),
        format!(
            "done. the worker's brief: {root}/mirror/m1-fp4-mi355x-atom@{}/MIRROR.md",
            &sb.rev[..12]
        ),
    ] {
        assert!(out.contains(&want), "{want:?} not in {out}");
    }
    // The repo reaches the worker only as its clone: nothing of it under the box's /<year>.
    assert!(!out.contains("CLAUDE.md"), "{out}");
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
    let other = existing("img:other", &root);
    let (code, out) = sb.mirror(&args, &[("FAKE_GFX", "90500"), ("FAKE_EXISTING", &other)]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("runs img:other, not the requested rocm/atom:1"), "{out}");
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

/// curl: answers the dashboard's API from $FAKE_DASH, and the snapshot of a commit as a
/// GitHub tarball (`InferenceX-<sha>/...`).
const FAKE_CURL: &str = r#"#!/bin/sh
for a; do url=$a; done
echo "$url" >>"$FAKE_DASH/urls"
case "$url" in
    */api/v1/views/options) cat "$FAKE_DASH/options.json" ;;
    */api/v1/availability) cat "$FAKE_DASH/availability.json" ;;
    */api/v1/benchmarks\?*) cat "$FAKE_DASH/benchmarks.json" ;;
    */api/v1/workflow-info\?*) cat "$FAKE_DASH/workflow.json" ;;
    */api/v1/server-log-files\?*) echo '["results/server.log"]' ;;
    */api/v1/server-log\?*) id=${url#*id=}; cat "$FAKE_DASH/log_${id%%&*}.txt" ;;
    */snap/*) tar -C "$FAKE_DASH/snap" -czf - "InferenceX-${url##*/}" ;;
    *) echo "curl: (22) The requested URL returned error: 404" >&2; exit 22 ;;
esac
"#;

const DASH_SHA: &str = "770268c51c2b368e9d669096041c003520f14c3a";

/// The dashboard of M1 (key m1): a vllm curve with two runs (222, the latest, at DASH_SHA
/// with image vllm/vllm:2; 111, an older sequence), its -mtp variant and an ATOM curve.
/// The snapshot has the mid-2026 layout, configs under .github/; the sandbox's clone is
/// removed, so nothing can come from it.
fn dashboard(sb: &Sandbox) -> Vec<(&'static str, String)> {
    let dash = sb.root.join("dash");
    let put = |rel: &str, text: &str| {
        let p = dash.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, text).unwrap();
    };
    put("options.json", r#"{"models": [{"name": "M1-120B", "dbKeys": ["m1"]}]}"#);
    let avail = |fw: &str, spec: &str, isl: u32, date: &str| {
        format!(
            r#"{{"model": "m1", "isl": {isl}, "osl": 1024, "precision": "fp4", "hardware": "mi355x", "framework": "{fw}",
                "spec_method": "{spec}", "disagg": false, "benchmark_type": "single_turn", "date": "{date}"}}"#
        )
    };
    put(
        "availability.json",
        &format!(
            "[{}, {}, {}, {}]",
            avail("vllm", "none", 1024, "2026-06-05"),
            avail("vllm", "none", 8192, "2026-03-09"),
            avail("vllm", "mtp", 1024, "2026-06-05"),
            avail("atom", "none", 1024, "2026-06-05")
        ),
    );
    let row = |id: &str, run: &str, date: &str, image: &str, isl: u32, tp: u32, conc: u32| {
        format!(
            r#"{{"id": "{id}", "hardware": "mi355x", "framework": "vllm", "model": "m1", "precision": "fp4",
                "spec_method": "none", "disagg": false, "benchmark_type": "single_turn", "isl": {isl}, "osl": 1024,
                "decode_tp": {tp}, "decode_ep": 1, "decode_dp_attention": false, "num_decode_gpu": {tp}, "conc": {conc},
                "image": "{image}", "date": "{date}", "curve_run_started_at": "{date} 18:00:00+00",
                "run_url": "https://github.com/o/InferenceX/actions/runs/{run}/attempts/1",
                "metrics": {{"output_tput_per_gpu": {}, "median_ttft": 0.04, "median_tpot": 0.005}}}}"#,
            200 * conc / tp
        )
    };
    put(
        "benchmarks.json",
        &format!(
            "[{}, {}, {}, {}]",
            row("12", "222", "2026-06-05", "vllm/vllm:2", 1024, 1, 8),
            row("11", "222", "2026-06-05", "vllm/vllm:2", 1024, 1, 4),
            row("13", "222", "2026-06-05", "vllm/vllm:2", 1024, 8, 4),
            row("21", "111", "2026-03-09", "vllm/vllm:1", 8192, 1, 4)
        ),
    );
    put(
        "workflow.json",
        &format!(
            r#"{{"runs": [{{"github_run_id": "222", "name": "Run Sweep - m1 bump"}}],
                "changelogs": [{{"workflow_run_id": "222", "config_keys": ["m1-fp4-mi355x-vllm"],
                                 "description": "Bump M1 vLLM", "pr_link": "https://github.com/o/InferenceX/pull/7"}}],
                "configs": [],
                "runConfigs": [{{"github_run_id": "222", "hardware": "mi355x", "head_sha": "{DASH_SHA}"}}]}}"#
        ),
    );
    put(
        "log_11.txt",
        "banner\n(APIServer pid=1) INFO non-default args: {'max_model_len': 2304}\n",
    );
    put(
        "log_13.txt",
        "(APIServer pid=1) INFO non-default args: {'max_model_len': 2304, 'tensor_parallel_size': 8}\n",
    );
    let snap = format!("snap/InferenceX-{DASH_SHA}");
    put(
        &format!("{snap}/.github/configs/amd-master.yaml"),
        &MASTER
            .replace("m1-fp4-mi355x-atom", "m1-fp4-mi355x-vllm")
            .replace("rocm/atom:1", "vllm/vllm:old")
            .replace("framework: atom", "framework: vllm"),
    );
    put(
        &format!("{snap}/benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x.sh"),
        "export VLLM_ROCM_USE_AITER=1\nvllm serve $MODEL --max-model-len $MAX_MODEL_LEN\n",
    );
    put(
        &format!("{snap}/benchmarks/benchmark_lib.sh"),
        "run_benchmark_serving() { :; }\n",
    );
    let bin = sb.root.join("bin/curl");
    fs::write(&bin, FAKE_CURL).unwrap();
    fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).unwrap();
    let mut cfg = fs::read_to_string(sb.root.join("state/config.toml")).unwrap();
    cfg += "infx_dashboard = 'https://dash.example'\ninfx_snapshot = 'https://dash.example/snap/{sha}'\n";
    fs::write(sb.root.join("state/config.toml"), cfg).unwrap();
    fs::remove_dir_all(sb.root.join("infx")).unwrap();
    vec![("FAKE_DASH", dash.display().to_string())]
}

#[test]
fn dashboard_mirror_previews_what_was_measured() {
    let sb = Sandbox::new("dash-preview");
    let env = dashboard(&sb);
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let (code, out) = sb.mirror(&["--dashboard", "--conf", "m1,vllm"], &env);
    assert_eq!(code, 0, "{out}");
    for want in [
        "also matched (name another part to pick one of them):\n  m1-fp4-mi355x-vllm-mtp",
        "curve m1-fp4-mi355x-vllm (M1-120B), latest 2026-06-05",
        "run 222 (2026-06-05, image vllm/vllm:2): 3 points, 1024/1024",
        "also on the dashboard: run 111 (2026-03-09, image vllm/vllm:1): 1 points, 8192/1024 (--run 111 for it)",
        // The run's own commit, as a snapshot in the .github/configs layout.
        &format!("GET https://dash.example/snap/{DASH_SHA}"),
        &format!("entry m1-fp4-mi355x-vllm in .github/configs/amd-master.yaml at {}", &DASH_SHA[..12]),
        "| run | [222](https://github.com/o/InferenceX/actions/runs/222/attempts/1) Run Sweep - m1 bump |",
        "| change | Bump M1 vLLM (https://github.com/o/InferenceX/pull/7) |",
        "| image | `vllm/vllm:2` (what ran; the config says `vllm/vllm:old`) |",
        // Every point with its numbers; out tok/s is per GPU times the GPUs.
        "| 1024/1024 | 1 | 1 | - | 1 | 4 | 800.0 | 800.0 | 40.0 | 5.00 | 11 |",
        "| 1024/1024 | 1 | 1 | - | 1 | 8 | 1600.0 | 1600.0 | 40.0 | 5.00 | 12 |",
        "| 1024/1024 | 8 | 1 | - | 8 | 4 | 800.0 | 100.0 | 40.0 | 5.00 | 13 |",
        // One server log per distinct server, its lowest concurrency.
        "1024/1024 tp 1 ep 1 (`dashboard/server_1024-1024_tp1_c4_11.log`):\n\n```\nnon-default args: {'max_model_len': 2304}\n```",
        "non-default args: {'max_model_len': 2304, 'tensor_parallel_size': 8}",
        "- `recipes/benchmarks/single_node/fixed_seq_len/m1_fp4_mi355x.sh`",
        "report each of yours next to the dashboard's number above",
    ] {
        assert!(out.contains(want), "{want:?} not in {out}");
    }
    let urls = fs::read_to_string(sb.root.join("dash/urls")).unwrap();
    assert!(urls.contains("/api/v1/benchmarks?model=M1-120B\n"), "{urls}");
    assert!(urls.contains("/api/v1/workflow-info?date=2026-06-05\n"), "{urls}");
    assert!(!urls.contains("log?id=12"), "one log per server: {urls}");
    assert!(
        urls.contains("server-log?id=11&file=results%2Fserver.log&download=1"),
        "{urls}"
    );

    // Two curves of the fewest parts: every match listed, nothing picked.
    let (code, out) = sb.mirror(&["--dashboard", "--conf", "m1,mi355"], &env);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("3 dashboard curves match; add a term"), "{out}");
    assert!(!out.contains("benchmarks?"), "{out}");
}

#[test]
fn dashboard_mirror_brings_up_the_image_that_ran() {
    let sb = Sandbox::new("dash-up");
    let env = dashboard(&sb);
    let mut env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    env.push(("FAKE_GFX", "90500 90500"));
    let args = [
        "fakebox",
        "vllm_mirror",
        "--dashboard",
        "--conf",
        "m1,mi355,vllm",
        "--week",
        "ww07",
        "--dry-run",
    ];
    let (code, out) = sb.mirror(&args, &env);
    assert_eq!(code, 0, "{out}");
    let root = format!("/{}/ww07/vllm_mirror", chrono_year());
    for want in [
        "fakebox: 2 x gfx950 as mi355x needs".to_string(),
        "--name=akao_vllm_mirror".to_string(),
        "--entrypoint sleep vllm/vllm:2 infinity".to_string(),
        format!(
            "done. the worker's brief: {root}/mirror/m1-fp4-mi355x-vllm@{}/MIRROR.md",
            &DASH_SHA[..12]
        ),
    ] {
        assert!(out.contains(&want), "{want:?} not in {out}");
    }

    // Another run: the dashboard names no commit for it, so nothing is touched.
    let mut other = args.to_vec();
    other.extend(["--run", "111"]);
    let (code, out) = sb.mirror(&other, &env);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("names no InferenceX commit for run 111"), "{out}");
    assert!(!out.contains("mkdir"), "{out}");
    let mut other = args.to_vec();
    other.extend(["--run", "999"]);
    let (code, out) = sb.mirror(&other, &env);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("run 999 is not among m1-fp4-mi355x-vllm's runs"), "{out}");
    // A host with several matching curves: refused, not guessed.
    let (code, out) = sb.mirror(&["fakebox", "--dashboard", "--conf", "m1", "--dry-run"], &env);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("matches several dashboard curves"), "{out}");
    assert!(!out.contains("mkdir"), "{out}");
    // The dashboard is one source, --rev the other.
    let (code, out) = sb.mirror(&["--dashboard", "--rev", "x", "--conf", "m1"], &env);
    assert_eq!(code, 2, "{out}");
    let (code, out) = sb.mirror(&["--date", "2026-06-05", "--conf", "m1"], &env);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("--dashboard"), "{out}");
}

/// The ISO year akao uses (its work year), without a date crate in the test.
fn chrono_year() -> String {
    let out = Command::new("date").arg("+%G").output().unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
