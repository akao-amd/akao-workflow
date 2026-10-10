//! End-to-end through the real `oaka` binary and the bash it compiles, with stand-ins for
//! sglang, InferenceX and sgl-eval.  Needs bash and python3, nothing else: no GPU, model,
//! network or installed sglang.  The real-hardware smoke run is in TEST.md.
//!
//! The stand-ins win over any installed package through PYTHONPATH (sglang), OAKA_INFX
//! (InferenceX) and PATH (sgl-eval).  Each records what it saw under `state/`.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

/// scripts/stack.sh refuses to swap the stack while any sglang server runs in the
/// container, which includes the stand-in servers of tests running in parallel.  Tests
/// that start servers share this lock; tests that install a stack hold it alone.
static SERVERS: RwLock<()> = RwLock::new(());

fn servers_shared() -> RwLockReadGuard<'static, ()> {
    SERVERS.read().unwrap_or_else(|e| e.into_inner())
}

fn servers_exclusive() -> RwLockWriteGuard<'static, ()> {
    SERVERS.write().unwrap_or_else(|e| e.into_inner())
}

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
try:
    import fakepkg
    pkg = os.path.realpath(fakepkg.__file__)
except ImportError:
    pkg = None
json.dump({"argv": args, "gpus": os.environ.get("HIP_VISIBLE_DEVICES"), "fakepkg": pkg},
          open(os.path.join(state, f"server_{port}.json"), "w"))
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port)); s.listen(); s.settimeout(0.2)
time.sleep(float(os.environ.get("FAKE_READY_DELAY", "0")))
print("The server is fired up and ready to roll!", flush=True)
while True:
    try: s.accept()[0].close()
    except socket.timeout: pass
"#;

/// `vllm serve <model> ...` and `python3 -m atom.entrypoints.openai_server ...`: listen on
/// --port (vllm) or --server-port (ATOM) and answer GET /health with 503 until
/// FAKE_READY_DELAY has passed, then 200; never print SGLang's ready line.  `--crash`
/// exits like an argparse error.
const FAKE_OPENAI_SERVER: &str = r#"
import http.server, json, os, signal, sys, time
args = sys.argv[1:]
if "--crash" in args:
    print("error: unrecognized arguments: --crash", flush=True)
    sys.exit(2)
flag = "--server-port" if "--server-port" in args else "--port"
port = int(args[args.index(flag) + 1])
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "pids"), "a") as f:
    f.write(f"{os.getpid()}\n")
json.dump({"argv": sys.argv, "gpus": os.environ.get("HIP_VISIBLE_DEVICES"), "env": dict(os.environ)},
          open(os.path.join(state, f"server_{port}.json"), "w"))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
ready_at = time.time() + float(os.environ.get("FAKE_READY_DELAY", "0"))
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" and time.time() >= ready_at else 503)
        self.end_headers()
    def log_message(self, *a):
        pass
print("INFO:     Started server process", flush=True)
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"#;

/// `python3 -m infx.bench fixed-seq point --k v ...`: writes a result JSON that echoes argv.
/// Output tok/s is conc times the installed stand-in package's SPEED (100 without one).
const FAKE_INFX: &str = r#"
import json, sys
try:
    import fakepkg
    speed = float(fakepkg.SPEED)
except ImportError:
    speed = 100.0
a = sys.argv[1:]
assert a[:2] == ["fixed-seq", "point"], a
o = {a[i][2:]: a[i + 1] for i in range(2, len(a), 2)}
c = int(o["conc"])
json.dump({"output_throughput": speed * c, "total_token_throughput": 200.0 * c, "mean_ttft_ms": 10.0,
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
        write("py/vllm/__init__.py", "");
        write("bin/vllm", &format!("#!/usr/bin/env python3\n{FAKE_OPENAI_SERVER}"));
        write("py/atom/__init__.py", "");
        write("py/atom/entrypoints/__init__.py", "");
        write("py/atom/entrypoints/openai_server.py", FAKE_OPENAI_SERVER);
        write("infx/infx/__init__.py", "");
        write("infx/infx/bench/__init__.py", "");
        write("infx/infx/bench/__main__.py", FAKE_INFX);
        write("infx/infx/bench/fixed_seq.py", "");
        write("bin/sgl-eval", FAKE_SGL_EVAL);
        Command::new("chmod")
            .arg("+x")
            .arg(root.join("bin/sgl-eval"))
            .arg(root.join("bin/vllm"))
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
        // The same model on the other engines, the way a mirror worker would write them.
        write(
            "lib/profiles/m/vllm.toml",
            "extends = 'm/base'\nengine = 'vllm'\n[args]\ntrust-remote-code = false\nmax-model-len = 10240\n",
        );
        write(
            "lib/profiles/m/atom.toml",
            "extends = 'm/base'\nengine = 'atom'\n[env]\nATOM_GPT_OSS_MODEL = '1'\n\
             [args]\ntrust-remote-code = false\nkv_cache_dtype = 'fp8'\n",
        );
        fs::create_dir_all(root.join("work")).unwrap();
        fs::create_dir_all(root.join("state")).unwrap();
        Sandbox { root }
    }

    fn work(&self) -> PathBuf {
        self.root.join("work")
    }

    fn oaka(&self, args: &[&str]) -> Command {
        let mut c = self.sandboxed(env!("CARGO_BIN_EXE_oaka"));
        c.args(args);
        c
    }

    /// `program` with the sandbox's environment, in the Work Directory.
    fn sandboxed(&self, program: &str) -> Command {
        let mut c = Command::new(program);
        let path = format!(
            "{}:{}",
            self.root.join("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        c.current_dir(self.work())
            .env("OAKA_LIB", self.root.join("lib"))
            .env("OAKA_INFX", self.root.join("infx"))
            .env("OAKA_GPUS", "gfx950,gfx950")
            .env("PYTHONPATH", self.root.join("py"))
            .env("PATH", path)
            .env("FAKE_STATE", self.root.join("state"))
            .env("OAKA_STACK_STATE", self.root.join("stack-state"))
            .env_remove("GPU_ARCH_LIST")
            .env_remove("AKAO_ARTIFACT_ROOT")
            .env("OAKA_ROCM", "10.0.0")
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

    fn read_tree(&self, rel: &str) -> String {
        fs::read_to_string(self.tree().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
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

/// The stand-in package's recipe: fails where BROKEN exists, edits setup.cfg in place
/// (stack.sh restores it), links the tree's fakepkg into the sandbox's PYTHONPATH and logs
/// each install to state/installs.
const FAKE_RECIPE: &str = r#"
if [ -e BROKEN ]; then echo "fakepkg: does not build here" >&2; exit 1; fi
echo "built for $GPU_ARCH" >>setup.cfg
ln -sfn "$TREE/fakepkg" "@ROOT@/py/fakepkg"
echo "$(git rev-parse --short HEAD)" >>"@ROOT@/state/installs"
"#;

impl Sandbox {
    /// A git repo `repo/` of the stand-in package fakepkg, one commit per entry of
    /// `speeds` (fakepkg.SPEED), with a BROKEN file in the commits whose indices are in
    /// `broken`, and the library's stacks.toml describing it.  Short shas, oldest first.
    fn fake_repo(&self, speeds: &[u32], broken: &[usize]) -> Vec<String> {
        let repo = self.root.join("repo");
        let git = |args: &[&str]| {
            let out = Command::new("git")
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "init.defaultBranch=main",
                ])
                .args(args)
                .current_dir(&repo)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        fs::create_dir_all(repo.join("fakepkg")).unwrap();
        git(&["init", "-q"]);
        fs::write(repo.join(".gitignore"), "__pycache__/\n").unwrap();
        fs::write(repo.join("setup.cfg"), "[fakepkg]\n").unwrap();
        let mut shas = Vec::new();
        for (i, speed) in speeds.iter().enumerate() {
            // Prints on import, as aiter does when it JIT-builds its core.
            let init = format!("print('[fakepkg] building core under /x/y')\nSPEED = {speed}\n");
            fs::write(repo.join("fakepkg/__init__.py"), init).unwrap();
            if broken.contains(&i) {
                fs::write(repo.join("BROKEN"), "").unwrap();
            } else {
                let _ = fs::remove_file(repo.join("BROKEN"));
            }
            git(&["add", "-A"]);
            git(&["commit", "-q", "--allow-empty", "-m", &format!("c{i} speed {speed}")]);
            shas.push(git(&["rev-parse", "--short", "HEAD"]));
        }
        let root = self.root.display().to_string();
        fs::write(
            self.root.join("lib/stacks.toml"),
            format!(
                "[fakepkg]\ndescription = 'stand-in'\nrepo = '{}'\nmodule = 'fakepkg'\nrestore = ['setup.cfg']\n\
                 install = '''{}'''\n[fakepkg.clean]\npaths = ['{root}/cache/jit']\ntree = ['**/__pycache__']\n",
                repo.display(),
                FAKE_RECIPE.replace("@ROOT@", &root)
            ),
        )
        .unwrap();
        shas
    }

    fn tree(&self) -> PathBuf {
        self.root.join("tree")
    }

    /// `git -C tree <args>`: success and stdout.
    fn git_tree(&self, args: &[&str]) -> (bool, String) {
        let out = Command::new("git")
            .arg("-C")
            .arg(self.tree())
            .args(args)
            .output()
            .unwrap();
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).trim().to_string(),
        )
    }

    /// Short shas the recipe installed, in order.
    fn installs(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("state/installs"))
            .unwrap_or_default()
            .lines()
            .map(String::from)
            .collect()
    }

    /// One server, one fixed-seq point at conc 1 (so tok/s = SPEED), and `[stack.fakepkg]`.
    fn stack_plan(&self, client_extra: &str, stack_extra: &str) -> String {
        format!(
            "{}[[client]]\nkind = 'fixed-seq'\nserver = 'a'\nisl_osl = [[128, 32]]\nconc = [1]\n{client_extra}\n\
             [stack.fakepkg]\ntree = '{}'\n{stack_extra}\n",
            server("a", 0, ""),
            self.tree().display()
        )
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
    let _servers = servers_shared();
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
    let _servers = servers_shared();
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
    let _servers = servers_shared();
    let sb = Sandbox::new("gate");
    let plan = format!("{}{}{GSM8K}{}", server("a", 0, ""), server("b", 1, ""), FIXED_SEQ);
    let (code, out) = sb.run(&plan, &[("FAKE_GSM8K_SCORE", "0.5")]);
    assert_eq!(code, 3, "a failed gate exits 3: {out}");
    assert!(out.contains("score 0.5 (min 0.90) FAIL"), "{out}");
    assert!(
        !sb.work().join("results/02_fixed-seq_b").exists(),
        "fixed-seq ran after a failed gate"
    );
    sb.assert_no_leftovers();
}

#[test]
fn ctrl_c_during_startup_stops_servers() {
    let _servers = servers_shared();
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
    // The same for agents, as JSON: the error in the document, the exit code unchanged.
    let out = sb.oaka(&["check", "--json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], false);
    assert!(v["error"].as_str().unwrap().contains("GPU 5 does not exist"), "{v}");

    fs::write(sb.work().join("plan.toml"), server("a", 1, "") + GSM8K).unwrap();
    let out = sb.oaka(&["check", "--json"]).output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["machine"]["rocm"], "10.0.0");
    assert_eq!(v["servers"][0]["gpus"], serde_json::json!([1]));
    assert_eq!(v["clients"][0]["kind"], "gsm8k");
    assert_eq!(v["vary"], serde_json::Value::Null);
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
    assert!(text.contains("warn  stacks      no "), "{text}");
    // With a stacks.toml: each package's repo and where its module imports from.
    sb.fake_repo(&[100], &[]);
    let text = String::from_utf8_lossy(&sb.oaka(&["doctor"]).output().unwrap().stdout).into_owned();
    assert!(
        text.contains("ok    stacks      ")
            && text.contains("stack fakepkg fakepkg imports from nowhere; recipes: any GPU arch"),
        "{text}"
    );
    fs::write(sb.root.join("lib/stacks.toml"), "[x]\nrepo = 'relative'\n").unwrap();
    let text = String::from_utf8_lossy(&sb.oaka(&["doctor"]).output().unwrap().stdout).into_owned();
    assert!(text.contains("FAIL  stacks"), "{text}");
    fs::remove_file(sb.root.join("lib/stacks.toml")).unwrap();
    // The stand-in sglang is the one found, so the check really looks at PYTHONPATH.
    assert!(
        text.contains(&format!("ok    sglang      {}", sb.root.join("py/sglang").display())),
        "{text}"
    );

    let out = sb.oaka(&["doctor", "--json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], true);
    let gpus = v["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["check"] == "gpus")
        .unwrap();
    assert_eq!(gpus["status"], "ok");
    assert!(gpus["detail"].as_str().unwrap().starts_with("2 x gfx950"), "{gpus}");

    fs::remove_dir_all(sb.root.join("infx")).unwrap();
    let out = sb.oaka(&["doctor"]).env("OAKA_GPUS", "").output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("FAIL  inferencex"), "{text}");
    assert!(text.contains("FAIL  gpus"), "{text}");
    assert!(String::from_utf8_lossy(&out.stderr).contains("2 check(s) failed"));
    let out = sb.oaka(&["doctor", "--json"]).env("OAKA_GPUS", "").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], false);
    assert!(v["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["check"] == "inferencex" && c["status"] == "fail"));
}

#[test]
fn stack_installs_a_commit_verifies_and_cleans() {
    let _servers = servers_exclusive();
    let sb = Sandbox::new("stack");
    let shas = sb.fake_repo(&[100, 200], &[]);
    let plan = sb.stack_plan("", &format!("commit = '{}'\nclean = 'before-install'", shas[0]));

    // First run creates the tree as a worktree of the repo at the commit.
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(sb.installs(), [shas[0].clone()]);
    assert!(out.contains("[stack") && out.contains("fakepkg ok:"), "{out}");
    assert!(sb.read("results/01_fixed-seq_a/summary.csv").contains(",100.0,"));
    // The recipe's in-place edit was put back.
    assert_eq!(sb.git_tree(&["status", "--porcelain", "--untracked-files=no"]).1, "");

    // Caches and ignored build products go before an install; user files stay.
    let pycache = sb.tree().join("fakepkg/__pycache__");
    fs::create_dir_all(&pycache).unwrap();
    fs::write(pycache.join("x.pyc"), "").unwrap();
    fs::write(sb.tree().join("notes.txt"), "mine").unwrap();
    fs::create_dir_all(sb.root.join("cache/jit/k")).unwrap();
    let plan = sb.stack_plan("", &format!("commit = '{}'\nclean = 'before-install'", shas[1]));
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(
        !pycache.join("x.pyc").exists() && !sb.root.join("cache/jit").exists(),
        "{out}"
    );
    assert!(sb.tree().join("notes.txt").exists());
    assert!(sb.read("results/01_fixed-seq_a/summary.csv").contains(",200.0,"));

    // clean = never keeps them and says so.
    fs::create_dir_all(sb.root.join("cache/jit/k")).unwrap();
    fs::write(sb.root.join("cache/jit/k/a.so"), "").unwrap();
    let (code, out) = sb.run(&sb.stack_plan("", &format!("commit = '{}'", shas[1])), &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("(1 files)"), "{out}");
    assert!(sb.root.join("cache/jit/k/a.so").exists());

    // A modified tracked file blocks a checkout: never reset someone's patch.
    fs::write(sb.tree().join("fakepkg/__init__.py"), "SPEED = 1  # my patch\n").unwrap();
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("has local changes"), "{out}");
    assert!(sb.read_tree("fakepkg/__init__.py").contains("my patch"));
    // ...but the tree as it is (no commit) installs with the patch, and says so.
    let (code, out) = sb.run(&sb.stack_plan("", ""), &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("with local changes"), "{out}");
    assert!(sb.read("results/01_fixed-seq_a/summary.csv").contains(",1.0,"));
    sb.assert_no_leftovers();
}

#[test]
fn stack_failures_stop_before_servers() {
    let _servers = servers_exclusive();
    let sb = Sandbox::new("stackfail");
    let shas = sb.fake_repo(&[100, 100], &[1]);
    // A recipe failure exits 2 and starts nothing; the edited file is still restored.
    let (code, out) = sb.run(&sb.stack_plan("", &format!("commit = '{}'", shas[1])), &[]);
    assert_eq!(code, 2, "{out}");
    assert!(
        out.contains("does not build here") && out.contains("install_fakepkg.sh exited 1"),
        "{out}"
    );
    assert!(!sb.work().join("logs/server_a.log").exists(), "a server started: {out}");
    assert_eq!(sb.git_tree(&["status", "--porcelain", "--untracked-files=no"]).1, "");

    // A running server in the container blocks any stack change.
    let port = ephemeral_port();
    let mut server = Command::new("python3")
        .args(["-m", "sglang.launch_server", "--port", &port.to_string()])
        .env("PYTHONPATH", sb.root.join("py"))
        .env("FAKE_STATE", sb.root.join("state"))
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while !sb.root.join("state/pids").exists() {
        assert!(Instant::now() < deadline, "server never started");
        std::thread::sleep(Duration::from_millis(100));
    }
    let (code, out) = sb.run(&sb.stack_plan("", &format!("commit = '{}'", shas[0])), &[]);
    Command::new("kill")
        .args(["-TERM", &server.id().to_string()])
        .status()
        .unwrap();
    server.wait().unwrap();
    sb.assert_no_leftovers();
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("a server runs in this container"), "{out}");
}

#[test]
fn commits_run_aba_and_compare() {
    let _servers = servers_exclusive();
    let sb = Sandbox::new("aba");
    let shas = sb.fake_repo(&[100, 50], &[]);
    let (a, b) = (&shas[0], &shas[1]);
    let plan = sb.stack_plan("", &format!("commits = ['{a}', '{b}', '{a}']"));
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(sb.installs(), [a.clone(), b.clone(), a.clone()]);
    for (n, sha, tok) in [(1, a, "100.0"), (2, b, "50.0"), (3, a, "100.0")] {
        let summary = sb.read(&format!("results/{n}_{sha}/01_fixed-seq_a/summary.csv"));
        assert!(summary.contains(&format!(",{tok},")), "{summary}");
    }
    let compare = sb.read("results/compare.csv");
    // metric, A1, B, A2, B/A1, A2/A1
    assert!(
        compare.contains("01_fixed-seq_a:isl128_osl32_c1,100.0,50.0,100.0,0.500,1.000"),
        "{compare}"
    );
    assert!(out.contains("A2/A1 should be ~1.000"), "{out}");
    sb.assert_no_leftovers();
}

#[test]
fn bisect_finds_the_regression_and_skips_broken_commits() {
    let _servers = servers_exclusive();
    let sb = Sandbox::new("bisect");
    // c0..c4 fast, c5 (the culprit) on slow; c3, git's first pick, does not build.
    let shas = sb.fake_repo(&[100, 100, 100, 100, 100, 50, 50, 50], &[3]);
    let plan = sb.stack_plan(
        "min_output_tok_s = 75",
        &format!("bisect = {{ good = '{}', bad = '{}' }}", shas[0], shas[7]),
    );
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    let log = sb.read("results/bisect/bisect.log");
    assert!(log.contains("# first bad commit:"), "{log}");
    let culprit = sb.git_tree(&["rev-parse", &shas[5]]).1;
    assert!(out.contains(&format!("first bad commit: [{culprit}]")), "{out}");
    let csv = sb.read("results/bisect/bisect.csv");
    assert!(
        csv.lines().next().unwrap().contains("01_fixed-seq_a:isl128_osl32_c1"),
        "{csv}"
    );
    assert!(csv.contains(&format!(",{},bad,ok,50.0,", shas[5])), "{csv}");
    assert!(csv.contains(&format!(",{},skip,install_failed,NA,", shas[3])), "{csv}");
    // No bisect left in progress; the venv was reinstalled from where the tree is now.
    assert!(!sb.git_tree(&["bisect", "log"]).0, "a bisect is still in progress");
    let (_, head) = sb.git_tree(&["rev-parse", "--short", "HEAD"]);
    assert_eq!(sb.installs().last(), Some(&head));
    assert_eq!(sb.git_tree(&["status", "--porcelain", "--untracked-files=no"]).1, "");
    sb.assert_no_leftovers();
}

#[test]
fn check_rejects_bad_stacks() {
    let sb = Sandbox::new("checkstack");
    let shas = sb.fake_repo(&[100, 50], &[]);
    let check = |plan: &str| {
        fs::write(sb.work().join("plan.toml"), plan).unwrap();
        let out = sb.oaka(&["check"]).output().unwrap();
        String::from_utf8_lossy(&out.stderr).into_owned()
    };
    let unknown = sb.stack_plan("", "").replace("[stack.fakepkg]", "[stack.nope]");
    assert!(check(&unknown).contains("no package nope in"), "{}", check(&unknown));
    let both = sb.stack_plan(
        "",
        &format!("commit = '{}'\ncommits = ['{}', '{}']", shas[0], shas[0], shas[1]),
    );
    assert!(check(&both).contains("set one of commit, commits, bisect"));
    let missing = sb.stack_plan("", "");
    assert!(
        check(&missing).contains("does not exist; set commit"),
        "{}",
        check(&missing)
    );
    let ungated = sb.stack_plan("", &format!("bisect = {{ good = '{}', bad = '{}' }}", shas[0], shas[1]));
    assert!(check(&ungated).contains("bisect needs a gate"), "{}", check(&ungated));
    let repo = sb.stack_plan("", "commit = 'x'").replace(
        &sb.tree().display().to_string(),
        &sb.root.join("repo").display().to_string(),
    );
    assert!(check(&repo).contains("use a worktree of it"), "{}", check(&repo));
}

#[test]
fn pythonpath_puts_the_tree_ahead_of_the_images_entries() {
    let _servers = servers_exclusive();
    let sb = Sandbox::new("pypath");
    let shas = sb.fake_repo(&[100], &[]);
    // As /etc/bash.bashrc does for aiter: a stale copy first on everyone's PYTHONPATH.
    let shadow = sb.root.join("shadow");
    fs::create_dir_all(shadow.join("fakepkg")).unwrap();
    fs::write(shadow.join("fakepkg/__init__.py"), "SPEED = 1\n").unwrap();
    let pythonpath = format!("{}:{}", shadow.display(), sb.root.join("py").display());
    let plan = sb.stack_plan("", &format!("commit = '{}'", shas[0]));

    // The install alone cannot win, and the verify says so.
    let (code, out) = sb.run(&plan, &[("PYTHONPATH", &pythonpath)]);
    assert_eq!(code, 2, "{out}");
    assert!(
        out.contains("not from") && out.contains("see pythonpath in stacks.toml"),
        "{out}"
    );

    // With `pythonpath`, the servers and the verify import from the tree.
    let stacks = sb.root.join("lib/stacks.toml");
    let text = fs::read_to_string(&stacks).unwrap();
    fs::write(
        &stacks,
        text.replace("module = 'fakepkg'\n", "module = 'fakepkg'\npythonpath = '.'\n"),
    )
    .unwrap();
    let (code, out) = sb.run(&plan, &[("PYTHONPATH", &pythonpath)]);
    assert_eq!(code, 0, "{out}");
    let tree = sb.tree().canonicalize().unwrap();
    let record = sb.server_record().join("\n");
    assert!(
        record.contains(&format!("\"fakepkg\": \"{}/fakepkg/__init__.py\"", tree.display())),
        "{record}"
    );
    sb.assert_no_leftovers();
}

#[test]
fn recipes_follow_the_gpu_arch() {
    let _servers = servers_exclusive();
    let sb = Sandbox::new("arch");
    let shas = sb.fake_repo(&[100], &[]);
    let stacks = sb.root.join("lib/stacks.toml");
    let text = fs::read_to_string(&stacks).unwrap().replace(
        "install = '''",
        "[fakepkg.install]\ngfx1250 = 'echo wrong recipe; exit 1'\n\"gfx942 gfx950\" = '''",
    );
    fs::write(&stacks, text).unwrap();
    let plan = sb.stack_plan("", &format!("commit = '{}'", shas[0]));

    // The plan's GPUs are gfx950 (OAKA_GPUS): that recipe, and only it, is compiled and run.
    let (code, out) = sb.run(&plan, &[]);
    assert_eq!(code, 0, "{out}");
    let script = sb.read("scripts/install_fakepkg.sh");
    assert!(script.contains("Install recipe of fakepkg for gfx950"), "{script}");
    assert!(
        script.contains("ln -sfn") && !script.contains("wrong recipe"),
        "{script}"
    );
    assert!(sb.read("scripts/stack.sh").contains("GPU_ARCH=gfx950 "));

    // An image built for another arch (its GPU_ARCH_LIST) refuses the recipe.
    let (code, out) = sb.run(&plan, &[("GPU_ARCH_LIST", "gfx942")]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("this image was built for GPU_ARCH_LIST=gfx942"), "{out}");

    // No recipe for the plan's arch: check names what there is.
    let out = sb.oaka(&["check"]).env("OAKA_GPUS", "gfx90a,gfx90a").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("no recipe for gfx90a; it has recipes for: gfx942 gfx950 gfx1250"),
        "{err}"
    );
}

#[test]
fn profiles_for_another_rocm_are_refused() {
    let _servers = servers_shared();
    let sb = Sandbox::new("rocm");
    fs::write(
        sb.root.join("lib/profiles/m/new.toml"),
        "extends = 'm/base'\narch = [['gfx950', '10.1'], 'gfx1250']\n",
    )
    .unwrap();
    let plan = server("a", 0, "").replace("m/base", "m/new") + GSM8K;
    fs::write(sb.work().join("plan.toml"), &plan).unwrap();

    // `oaka check` (and compile, run) aborts in a ROCm 10.0 container...
    let out = sb.oaka(&["check"]).output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        err.contains("this container has ROCm 10.0.0 on gfx950, but profile m/new is for gfx950:10.1 gfx1250"),
        "{err}"
    );

    // ...and scripts compiled in a 10.1 container refuse to start there when rerun by hand.
    let out = sb.oaka(&["compile"]).env("OAKA_ROCM", "10.1.2").output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let run = |rocm: &str| {
        let out = sb
            .sandboxed("bash")
            .arg("scripts/run_all.sh")
            .env("OAKA_ROCM", rocm)
            .output()
            .unwrap();
        (
            out.status.code(),
            format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        )
    };
    let (code, out) = run("10.0.0");
    assert_eq!(code, Some(1), "{out}");
    assert!(
        out.contains(
            "[server a] FATAL: this container has ROCm 10.0.0 on gfx950, but profile m/new is for gfx950:10.1 gfx1250"
        ),
        "{out}"
    );
    let (code, out) = run("10.1.0");
    assert_eq!(code, Some(0), "{out}");
    sb.assert_no_leftovers();
}

/// A server of profile `profile` on GPU `gpu`, on an ephemeral port.
fn server_of(name: &str, profile: &str, gpu: u32, extra: &str) -> (String, u16) {
    let port = ephemeral_port();
    (
        format!("[[server]]\nname = '{name}'\nprofile = '{profile}'\ngpus = [{gpu}]\nport = {port}\n{extra}\n"),
        port,
    )
}

#[test]
fn vllm_and_atom_servers_start_by_health_and_stop() {
    let _servers = servers_shared();
    let sb = Sandbox::new("engines");
    let (a, port_a) = server_of("a", "m/vllm", 0, "");
    let (b, port_b) = server_of("b", "m/atom", 1, "");
    let fixed =
        |s: &str| format!("[[client]]\nkind = 'fixed-seq'\nserver = '{s}'\nisl_osl = [[128, 32]]\nconc = [2]\n");
    let plan = format!("{a}{b}{}{}", fixed("a"), fixed("b"));
    // Not ready for the first second: /health answers 503, and no SGLang line ever comes.
    let (code, out) = sb.run(&plan, &[("FAKE_READY_DELAY", "1")]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("server a ready (GET /health)"), "{out}");
    assert!(out.contains("server b ready (GET /health)"), "{out}");
    let record = |port: u16| {
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(sb.root.join(format!("state/server_{port}.json"))).unwrap())
                .unwrap();
        v["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let model = sb.root.join("model").display().to_string();
    let vllm = record(port_a);
    assert!(
        vllm.contains(&format!(
            "vllm serve {model} --tensor-parallel-size 1 --port {port_a} --max-model-len 10240"
        )),
        "{vllm}"
    );
    assert!(!vllm.contains("trust-remote-code"), "{vllm}");
    let atom = record(port_b);
    assert!(
        atom.contains(&format!(
            "openai_server.py --model {model} -tp 1 --server-port {port_b} --kv_cache_dtype fp8"
        )),
        "{atom}"
    );
    assert!(sb
        .read("logs/server_b.log")
        .contains("[server b] cmd python3 -m atom.entrypoints.openai_server"));
    assert!(sb
        .read("logs/server_b.log")
        .contains("[server b] env ATOM_GPT_OSS_MODEL=1"));
    assert_eq!(sb.read("results/02_fixed-seq_b/summary.csv").lines().count(), 2);
    sb.assert_no_leftovers();

    // A vllm server that dies at startup is caught although it never prints anything.
    let (a, _) = server_of("a", "m/vllm", 0, "[server.args]\ncrash = true");
    let (code, out) = sb.run(&format!("{a}{}", fixed("a")), &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("server a died during startup"), "{out}");

    // Agents see the engine; a profile may not set the port in any engine's spelling.
    let out = sb.oaka(&["check", "--json"]).output().unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["servers"][0]["engine"], "vllm", "{v}");
    let (a, _) = server_of("a", "m/atom", 0, "[server.args]\nserver-port = 1");
    fs::write(sb.work().join("plan.toml"), a).unwrap();
    let out = sb.oaka(&["check"]).output().unwrap();
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--server-port is set by oaka from the plan"),
        "{out:?}"
    );
}

#[test]
fn doctor_reports_engines() {
    let sb = Sandbox::new("doctor-engines");
    let out = sb.oaka(&["doctor"]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(out.status.code(), Some(0), "{text}");
    for (engine, file) in [("vllm", "py/vllm/__init__.py"), ("atom", "py/atom/__init__.py")] {
        let want = format!("{engine:<11} {}", sb.root.join(file).display());
        assert!(text.contains(&format!("ok    {want}")), "{want:?} not in {text}");
    }
    // Without ATOM (unless this machine has a real one), its profile is the reason to warn.
    fs::remove_dir_all(sb.root.join("py/atom")).unwrap();
    let text = String::from_utf8_lossy(&sb.oaka(&["doctor"]).output().unwrap().stdout).into_owned();
    if !text.contains("ok    atom") {
        assert!(
            text.contains("warn  atom        python3 cannot find atom; profiles m/atom use it"),
            "{text}"
        );
    }
}

#[test]
fn artifact_root_is_reported_and_checked() {
    let sb = Sandbox::new("artifacts");
    let out = sb.oaka(&["doctor"]).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        text.contains("warn  artifacts   AKAO_ARTIFACT_ROOT is not set"),
        "{text}"
    );
    // The fallback the skills give, and the real way to pin one.
    assert!(text.contains("its root is its working directory"), "{text}");
    assert!(
        text.contains("docker --context <nick> rm -f akao_<name>, then akao init"),
        "{text}"
    );
    let root = sb.root.display().to_string();
    let out = sb.oaka(&["doctor"]).env("AKAO_ARTIFACT_ROOT", &root).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.contains(&format!("ok    artifacts   {root}")), "{text}");

    fs::write(sb.work().join("plan.toml"), server("a", 0, "") + GSM8K).unwrap();
    let warnings = |root: &str| {
        let out = sb
            .oaka(&["check", "--json"])
            .env("AKAO_ARTIFACT_ROOT", root)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        v["warnings"].to_string()
    };
    // The work dir is a task dir under the root: nothing to say.
    assert!(!warnings(&root).contains("artifact root"), "{}", warnings(&root));
    let work = sb.work().display().to_string();
    assert!(warnings(&work).contains("is the artifact root"), "{}", warnings(&work));
    assert!(
        warnings("/nonexistent/elsewhere").contains("is outside the artifact root"),
        "{}",
        warnings("/nonexistent/elsewhere")
    );
}

#[test]
fn a_model_path_is_never_shell_code() {
    let _servers = servers_shared();
    let sb = Sandbox::new("quoting");
    let model = sb.root.join("m $USD 'q' $(touch PWNED) `touch PWNED2`");
    fs::create_dir_all(&model).unwrap();
    let toml_model = model.display().to_string().replace('\\', "\\\\").replace('"', "\\\"");
    let plan = server("a", 0, &format!("model = \"{toml_model}\"")) + GSM8K;
    let (code, out) = sb.run(&plan, &[("USD", "100")]);
    assert_eq!(code, 0, "{out}");
    let log = sb.read("logs/server_a.log");
    assert!(log.contains(&format!("model {}  tp 1", model.display())), "{log}");
    let records = sb.server_record().join("\n");
    let argv_model = serde_json::to_string(&model.display().to_string()).unwrap();
    assert!(
        records.contains(&format!("\"--model-path\", {argv_model}")),
        "{records}"
    );
    for f in ["PWNED", "PWNED2"] {
        assert!(
            !sb.work().join(f).exists() && !sb.root.join(f).exists(),
            "{f} was created"
        );
    }
    sb.assert_no_leftovers();
}
