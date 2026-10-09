//! End-to-end through the real `oaka` binary and the bash it compiles, with stand-ins for
//! sglang, InferenceX and sgl-eval.  Needs bash and python3, nothing else: no GPU, model,
//! network or installed sglang.  The real-hardware smoke run is in TEST.md.
//!
//! The stand-ins win over any installed package through PYTHONPATH (sglang), OAKA_INFX
//! (InferenceX) and PATH (sgl-eval).  Each records what it saw under `state/`.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// `python3 -m sglang.launch_server`: listens on --port, prints sglang's ready line, keeps
/// a child process like sglang's scheduler, and stops both on SIGTERM.
/// `--crash` exits like an argparse error; FAKE_READY_DELAY delays readiness.
const FAKE_SGLANG: &str = r#"
import json, os, signal, socket, subprocess, sys, time
args = sys.argv[1:]
if "--crash" in args:
    print("sglang serve: error: unrecognized arguments: --crash", flush=True)
    sys.exit(2)
port = int(args[args.index("--port") + 1])
def stop(*_):
    child.terminate(); child.wait(); sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)  # until the child exists: a TERM now must not orphan it
child = subprocess.Popen(["sleep", "300"], preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL))
signal.signal(signal.SIGTERM, stop)
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "pids"), "a") as f:
    f.write(f"{os.getpid()}\n{child.pid}\n")
json.dump({"argv": args, "gpus": os.environ.get("HIP_VISIBLE_DEVICES")}, open(os.path.join(state, f"server_{port}.json"), "w"))
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port)); s.listen(); s.settimeout(0.2)
time.sleep(float(os.environ.get("FAKE_READY_DELAY", "0")))
print("The server is fired up and ready to roll!", flush=True)
while True:
    try: s.accept()[0].close()
    except socket.timeout: pass
"#;

/// `python3 -m infx.bench fixed-seq point --k v ...`: writes a result JSON that echoes argv.
const FAKE_INFX: &str = r#"
import json, sys
a = sys.argv[1:]
assert a[:2] == ["fixed-seq", "point"], a
o = {a[i][2:]: a[i + 1] for i in range(2, len(a), 2)}
c = int(o["conc"])
json.dump({"output_throughput": 100.0 * c, "total_token_throughput": 200.0 * c, "mean_ttft_ms": 10.0,
           "mean_tpot_ms": 1.0, "completed": int(o["num-prompts"]), "argv": a}, open(o["result"], "w"))
"#;

/// `sgl-eval run gsm8k ...`: writes metrics.json with FAKE_GSM8K_SCORE (default 0.95).
const FAKE_SGL_EVAL: &str = r#"#!/usr/bin/env python3
import json, os, sys, time
a = sys.argv[1:]
d = os.path.join(a[a.index("--out-dir") + 1], f"sgl_eval_gsm8k_{time.time_ns()}")
os.makedirs(d)
score = float(os.environ.get("FAKE_GSM8K_SCORE", "0.95"))
json.dump({"aggregate": {"score": score}, "thinking": "--thinking" in a}, open(os.path.join(d, "metrics.json"), "w"))
"#;

struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Sandbox {
        let root = std::env::temp_dir().join(format!("oaka-it-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let write = |rel: &str, text: &str| {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, text).unwrap();
        };
        write("py/sglang/__init__.py", "");
        write("py/sglang/launch_server.py", FAKE_SGLANG);
        write("infx/infx/__init__.py", "");
        write("infx/infx/bench/__init__.py", "");
        write("infx/infx/bench/__main__.py", FAKE_INFX);
        write("infx/infx/bench/fixed_seq.py", "");
        write("bin/sgl-eval", FAKE_SGL_EVAL);
        Command::new("chmod")
            .arg("+x")
            .arg(root.join("bin/sgl-eval"))
            .status()
            .unwrap();
        let model = root.join("model");
        fs::create_dir_all(&model).unwrap();
        write(
            "lib/profiles/m/base.toml",
            &format!(
                "model = '{}'\n[env]\nSGLANG_X = '1'\n[args]\ntrust-remote-code = true\n",
                model.display()
            ),
        );
        fs::create_dir_all(root.join("work")).unwrap();
        fs::create_dir_all(root.join("state")).unwrap();
        Sandbox { root }
    }

    fn work(&self) -> PathBuf {
        self.root.join("work")
    }

    fn oaka(&self, args: &[&str]) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_oaka"));
        let path = format!(
            "{}:{}",
            self.root.join("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        c.args(args)
            .current_dir(self.work())
            .env("OAKA_LIB", self.root.join("lib"))
            .env("OAKA_INFX", self.root.join("infx"))
            .env("OAKA_GPUS", "gfx950,gfx950")
            .env("PYTHONPATH", self.root.join("py"))
            .env("PATH", path)
            .env("FAKE_STATE", self.root.join("state"))
            .stdin(Stdio::null());
        c
    }

    /// Write plan.toml and run `oaka run`.
    fn run(&self, plan: &str, envs: &[(&str, &str)]) -> (i32, String) {
        fs::write(self.work().join("plan.toml"), plan).unwrap();
        let out: Output = self.oaka(&["run"]).envs(envs.iter().copied()).output().unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code().unwrap_or(-1), text)
    }

    fn read(&self, rel: &str) -> String {
        fs::read_to_string(self.work().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
    }

    /// Every process a fake server started is gone.
    fn assert_no_leftovers(&self) {
        let pids = fs::read_to_string(self.root.join("state/pids")).unwrap_or_default();
        for pid in pids.lines() {
            assert!(!alive(pid), "process {pid} outlived the run");
        }
    }

    fn server_record(&self) -> Vec<String> {
        fs::read_dir(self.root.join("state"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("server_"))
            .map(|e| fs::read_to_string(e.path()).unwrap())
            .collect()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Running and not a zombie.
fn alive(pid: &str) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat"))
        .map(|s| {
            s.rsplit(')')
                .next()
                .is_some_and(|rest| !rest.trim_start().starts_with('Z'))
        })
        .unwrap_or(false)
}

/// A port from the OS's ephemeral range, outside oaka's 29900-30100, so tests running in
/// parallel (and servers already on this box) cannot collide with each other.
fn ephemeral_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn server(name: &str, gpu: u32, extra: &str) -> String {
    let port = ephemeral_port();
    format!("[[server]]\nname = '{name}'\nprofile = 'm/base'\ngpus = [{gpu}]\nport = {port}\n{extra}\n")
}

const GSM8K: &str = "[[client]]\nkind = 'gsm8k'\nserver = 'a'\nthinking = true\n";
const FIXED_SEQ: &str = "[[client]]\nkind = 'fixed-seq'\nserver = 'b'\nisl_osl = [[128, 32]]\nconc = [2, 4]\n";

#[test]
fn full_plan_runs_and_cleans_up() {
    let sb = Sandbox::new("full");
    // Server a has no port, so compile picks one and keeps it in plan.lock.toml.
    let a = "[[server]]\nname = 'a'\nprofile = 'm/base'\ngpus = [0]\n";
    let plan = format!(
        "{a}{}{GSM8K}{FIXED_SEQ}",
        server("b", 1, "[server.args]\npage-size = 64")
    );
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("[gsm8k] score 0.95 (min 0.90) PASS"), "{out}");

    // fixed-seq: InferenceX gets 10 prompts per concurrency and the vllm backend.
    let summary = sb.read("results/02_fixed-seq_b/summary.csv");
    assert_eq!(summary.lines().count(), 3, "{summary}");
    let point = sb.read("results/02_fixed-seq_b/b_isl128_osl32_c4_r1.json");
    for want in [
        r#""--num-prompts", "40""#,
        r#""--backend", "vllm""#,
        r#""--random-range-ratio", "0.8""#,
    ] {
        assert!(point.contains(want), "{want} not in {point}");
    }
    // Servers got their GPU, the plan override, and oaka's flags.
    let records = sb.server_record().join("\n");
    assert!(
        records.contains(r#""gpus": "1""#) && records.contains(r#""gpus": "0""#),
        "{records}"
    );
    assert!(records.contains(r#""--page-size", "64""#), "{records}");
    assert!(sb
        .read("logs/server_a.log")
        .contains("[server a] cmd python3 -m sglang.launch_server"));
    assert!(sb.read("logs/run_all.log").contains("] done"));
    sb.assert_no_leftovers();

    // Recompiling keeps the ports; a second run works the same way.
    let lock = sb.read("plan.lock.toml");
    assert!(lock.contains("a = "), "{lock}");
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(sb.read("plan.lock.toml"), lock);
    sb.assert_no_leftovers();
}

#[test]
fn server_crash_at_startup_stops_the_run() {
    let sb = Sandbox::new("crash");
    let plan = format!("{}{}", server("a", 0, "[server.args]\ncrash = true"), GSM8K);
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("server a died during startup"), "{out}");
    assert!(out.contains("unrecognized arguments: --crash"), "{out}");
    assert!(!sb.work().join("results").exists(), "a client ran after the crash");
}

#[test]
fn failed_gsm8k_gate_stops_the_run() {
    let sb = Sandbox::new("gate");
    let plan = format!("{}{}{GSM8K}{}", server("a", 0, ""), server("b", 1, ""), FIXED_SEQ);
    let (code, out) = sb.run(&plan, &[("FAKE_GSM8K_SCORE", "0.5")]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("score 0.5 (min 0.90) FAIL"), "{out}");
    assert!(
        !sb.work().join("results/02_fixed-seq_b").exists(),
        "fixed-seq ran after a failed gate"
    );
    sb.assert_no_leftovers();
}

#[test]
fn ctrl_c_during_startup_stops_servers() {
    use std::os::unix::process::CommandExt;
    let sb = Sandbox::new("int");
    fs::write(sb.work().join("plan.toml"), format!("{}{}", server("a", 0, ""), GSM8K)).unwrap();
    // Own process group, so the SIGINT reaches oaka/bash/tee the way a terminal's would.
    let mut child = sb
        .oaka(&["run"])
        .env("FAKE_READY_DELAY", "30")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .unwrap();
    let pids = sb.root.join("state/pids");
    let deadline = Instant::now() + Duration::from_secs(20);
    while !pids.exists() {
        assert!(Instant::now() < deadline, "server never started");
        std::thread::sleep(Duration::from_millis(100));
    }
    let group = format!("-{}", child.id());
    for _ in 0..2 {
        Command::new("kill").args(["-INT", "--", &group]).status().unwrap();
        std::thread::sleep(Duration::from_millis(300));
    }
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(130));
    sb.assert_no_leftovers();
    assert!(sb.read("logs/run_all.log").contains("stopping servers"));
}

#[test]
fn check_names_the_field_to_fix() {
    let sb = Sandbox::new("check");
    fs::write(sb.work().join("plan.toml"), server("a", 5, "")).unwrap();
    let out = sb.oaka(&["check"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("GPU 5 does not exist; this machine has 2 GPUs (0-1)"),
        "{err}"
    );
}

#[test]
fn doctor_reports_missing_prerequisites() {
    let sb = Sandbox::new("doctor");
    let out = sb.oaka(&["doctor"]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{text}");
    for want in [
        "ok    library",
        "ok    gpus        2 x gfx950",
        "ok    inferencex",
        "ok    sgl-eval",
    ] {
        assert!(text.contains(want), "{want:?} not in {text}");
    }
    // The stand-in sglang is the one found, so the check really looks at PYTHONPATH.
    assert!(
        text.contains(&format!("ok    sglang      {}", sb.root.join("py/sglang").display())),
        "{text}"
    );

    fs::remove_dir_all(sb.root.join("infx")).unwrap();
    let out = sb.oaka(&["doctor"]).env("OAKA_GPUS", "").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("FAIL  inferencex"), "{text}");
    assert!(text.contains("FAIL  gpus"), "{text}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("2 check(s) failed"));
}
