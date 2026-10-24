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
/// `--crash` exits like an argparse error; FAKE_READY_DELAY delays readiness;
/// `--exit-after-ready` exits half a second after it said it is ready.
///
/// It also serves what a trace client uses, over a simulated scheduler: `POST /v1/completions`
/// (`prompt_len` and `max_tokens` from the body; one EXTEND step admits every waiting request,
/// DECODE steps give each running one a token), `POST /flush_cache`, and profiler V2's by-stage
/// `/start_profile`: a capture starts at the first step of its stage, ends at the first step
/// of another stage or after num_steps + 1 steps, and writes one trace per TP rank; arming
/// while one is armed is refused, and so is any capture without SGLANG_PROFILE_V2=1.
/// Each step's span is written on the CPU and the GPU with one kernel, unless FAKE_TRACE_NO_GPU
/// (CPU events only, as torch 2.12 / ROCm 10.1 without sgl-project/sglang#44292).
/// FAKE_STEP_S is a step's time (bs = 1: a quarter of it); FAKE_PASS_PREFILL_MIXED makes the
/// trace clients' prefill captures (prefix pass*) also record the decode step that ends them.
/// Events go to state/events_<port>: `arm <prefix> <t>`, `flush <t>`, `done <tag> <t>`.
const FAKE_SGLANG: &str = r#"
import gzip, http.server, json, os, signal, socket, subprocess, sys, threading, time
args = sys.argv[1:]
if "--crash" in args:
    print("sglang serve: error: unrecognized arguments: --crash", flush=True)
    sys.exit(2)
port = int(args[args.index("--port") + 1])
tp = int(args[args.index("--tp") + 1]) if "--tp" in args else 1
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
# What each compile cache held at start; then "compile" a kernel into it, as sglang would.
caches = {}
for var in ("TRITON_CACHE_DIR", "XDG_CACHE_HOME", "SGLANG_CACHE_DIR"):
    path = os.environ.get(var)
    caches[var] = [path, sorted(os.listdir(path)) if path and os.path.isdir(path) else None]
    if path:
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, "kernel.bin"), "w").close()
json.dump({"argv": args, "gpus": os.environ.get("HIP_VISIBLE_DEVICES"), "fakepkg": pkg, "caches": caches,
           "profile_v2": os.environ.get("SGLANG_PROFILE_V2")},
          open(os.path.join(state, f"server_{port}.json"), "w"))

STEP_S = float(os.environ.get("FAKE_STEP_S", "0.01"))
V2 = os.environ.get("SGLANG_PROFILE_V2") == "1"
MIXED = bool(os.environ.get("FAKE_PASS_PREFILL_MIXED"))
GPU = not os.environ.get("FAKE_TRACE_NO_GPU")
lock = threading.Lock()
waiting, running = [], []
prof = None  # the armed capture: dict(stage, num_steps, dir, prefix, steps)
def event(*words):
    with open(os.path.join(state, f"events_{port}"), "a") as f:
        f.write(" ".join(map(str, words)) + f" {time.time()}\n")
def write_trace(p):
    for rank in range(tp):
        ev = []
        for i, name in enumerate(p["steps"]):
            ev.append({"name": name, "cat": "user_annotation", "ts": i * 1000, "ph": "X"})
            if GPU:
                ev.append({"name": name, "cat": "gpu_user_annotation", "ts": i * 1000 + 5, "ph": "X"})
                ev.append({"name": "_moe_gemm_a8w4", "cat": "kernel", "ts": i * 1000 + 6, "ph": "X"})
        path = os.path.join(p["dir"], f"{p['prefix']}-{time.time()}-TP-{rank}-{p['stage']}.trace.json.gz")
        with gzip.open(path + ".tmp", "wt") as f:
            json.dump({"traceEvents": ev}, f)
        os.rename(path + ".tmp", path)
def profile_step(mode, name):
    global prof
    if prof is None:
        return
    want = "EXTEND" if prof["stage"] == "prefill" else "DECODE"
    if not prof["steps"]:
        if mode == want:
            prof["steps"].append(name)
        return
    if mode != want:
        if MIXED and prof["stage"] == "prefill" and prof["prefix"].startswith("pass"):
            prof["steps"].append(name)
        write_trace(prof); prof = None
        return
    prof["steps"].append(name)
    if len(prof["steps"]) == prof["num_steps"] + 1:
        write_trace(prof); prof = None
def scheduler():
    while True:
        with lock:
            if waiting:
                batch = waiting[:]; del waiting[:]
                name = f"step[EXTEND bs={len(batch)} toks={sum(r['len'] for r in batch)}]"
                profile_step("EXTEND", name)
                for r in batch:
                    r["got"] = 1
                running.extend(batch); bs = len(batch)
            elif running:
                name = f"step[DECODE bs={len(running)}]"
                profile_step("DECODE", name)
                for r in running:
                    r["got"] += 1
                bs = len(running)
            else:
                bs = 0
            for r in [r for r in running if r["got"] >= r["max"]]:
                running.remove(r); r["done"].set()
        time.sleep(STEP_S / 4 if bs <= 1 else STEP_S)
threading.Thread(target=scheduler, daemon=True).start()
class H(http.server.BaseHTTPRequestHandler):
    def reply(self, code, text):
        body = text.encode()
        self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        global prof
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/v1/completions":
            prompt = str(body.get("prompt", ""))
            r = {"len": int(body.get("prompt_len", len(prompt.split()))), "max": max(1, int(body.get("max_tokens", 16))),
                 "got": 0, "done": threading.Event()}
            with lock:
                waiting.append(r)
            r["done"].wait()
            event("done", (prompt.split() or ["-"])[0])
            self.reply(200, json.dumps({"choices": [{"text": "x " * r["got"]}], "usage": {"completion_tokens": r["got"]}}))
        elif self.path == "/flush_cache":
            event("flush")
            self.reply(200, "Cache flushed.\n")
        elif self.path == "/start_profile":
            if not V2:
                self.reply(500, "profiler v1: by-stage profile_stages are not supported"); return
            if not body.get("profile_by_stage") or "merge_profiles" in body or "start_step" in body:
                self.reply(500, "AssertionError: V2 profiles by stage only"); return
            with lock:
                if prof is not None:
                    self.reply(400, "AssertionError: a capture is already armed"); return
                prof = {"stage": body["profile_stages"][0], "num_steps": int(body["num_steps"]),
                        "dir": body["output_dir"], "prefix": body["profile_prefix"], "steps": []}
                os.makedirs(prof["dir"], exist_ok=True)
            event("arm", body["profile_prefix"])
            self.reply(200, "Start profiling.\n")
        else:
            self.reply(404, "")
    def log_message(self, *a):
        pass
http.server.ThreadingHTTPServer.allow_reuse_address = True
http.server.ThreadingHTTPServer.daemon_threads = True
srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
time.sleep(float(os.environ.get("FAKE_READY_DELAY", "0")))
print("The server is fired up and ready to roll!", flush=True)
if "--exit-after-ready" in args:
    def leave():
        child.terminate(); child.wait(); os._exit(0)
    threading.Timer(0.5, leave).start()
srv.serve_forever()
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

/// `python3 -m sglang_router.launch_router --pd-disaggregation --prefill URL BOOT ... --decode URL
/// ... --port P`: records its argv in state/router_<port>.json, and answers `POST /generate` with
/// 200 once FAKE_ROUTER_DELAY has passed and every server behind it accepts connections, 503
/// before (as a router whose servers are not registered yet); with FAKE_ROUTER_HANG it never
/// answers.  Stops on SIGTERM.
const FAKE_ROUTER: &str = r#"
import http.server, json, os, signal, socket, sys, time
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
state = os.environ["FAKE_STATE"]
with open(os.path.join(state, "pids"), "a") as f:
    f.write(f"{os.getpid()}\n")
json.dump({"argv": args}, open(os.path.join(state, f"router_{port}.json"), "w"))
urls = [args[i + 1] for i, a in enumerate(args) if a in ("--prefill", "--decode")]
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
ready_at = time.time() + float(os.environ.get("FAKE_ROUTER_DELAY", "0"))
def up(url):
    try:
        socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[1])), timeout=1).close()
        return True
    except OSError:
        return False
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if os.environ.get("FAKE_ROUTER_HANG"):
            time.sleep(600)
        ok = self.path == "/generate" and time.time() >= ready_at and all(map(up, urls))
        self.send_response(200 if ok else 503)
        self.end_headers()
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

/// `python3 -m infx.bench fixed-seq point --k v ...`: writes a result JSON that echoes argv.
/// Output tok/s is conc times the installed stand-in package's SPEED (100 without one).
/// Each point appends `<result file> <start> <end>` to state/points; FAKE_INFX_SLOW
/// (`<port>:<seconds>`) makes the points against one server take that long, FAKE_INFX_FAIL
/// (`<port>:<conc>`) fails that server's points at that concurrency; with FAKE_INFX_COLD a
/// point's first run (`_r1`) is half as fast, so a test can tell which run a summary took;
/// FAKE_INFX_SCALE (`<port>:<conc>:<factor>,...`) scales that server's tok/s at that concurrency.
const FAKE_INFX: &str = r#"
import json, os, sys, time
try:
    import fakepkg
    speed = float(fakepkg.SPEED)
except ImportError:
    speed = 100.0
a = sys.argv[1:]
assert a[:2] == ["fixed-seq", "point"], a
o = {a[i][2:]: a[i + 1] for i in range(2, len(a), 2)}
c = int(o["conc"])
port = o["base-url"].rsplit(":", 1)[1]
start = time.time()
with open(os.path.join(os.environ["FAKE_STATE"], "pids"), "a") as f:
    f.write(f"{os.getpid()}\n")
slow = os.environ.get("FAKE_INFX_SLOW", ":").split(":")
if slow[0] == port:
    time.sleep(float(slow[1]))
if os.environ.get("FAKE_INFX_FAIL") == f"{port}:{c}":
    sys.exit("fake infx: failing as asked")
with open(os.path.join(os.environ["FAKE_STATE"], "points"), "a") as f:
    f.write(f"{os.path.basename(o['result'])} {start} {time.time()}\n")
if os.environ.get("FAKE_INFX_COLD") and o["result"].endswith("_r1.json"):
    speed /= 2
for scale in filter(None, os.environ.get("FAKE_INFX_SCALE", "").split(",")):
    p, k, x = scale.split(":")
    if p == port and int(k) == c:
        speed *= float(x)
json.dump({"output_throughput": speed * c, "total_token_throughput": 200.0 * c, "mean_ttft_ms": 10.0,
           "mean_tpot_ms": 1.0, "completed": int(o["num-prompts"]), "argv": a}, open(o["result"], "w"))
"#;

/// InferenceX's `infx.bench.fixed_seq` as trace_point.py uses it: `Point` and `client_argv`.
const FAKE_INFX_FIXED_SEQ: &str = r#"
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Point:
    base_url: str
    model: str
    backend: str
    isl: int
    osl: int
    random_range_ratio: str
    conc: int
    num_prompts: int
    result: Path
    tokenizer: str | None = None
    use_chat_template: bool = False
    trust_remote_code: bool = False

def client_argv(point: Point) -> list[str]:
    return ["python3", "-m", "infx.bench_serving.benchmark_serving", "--model", point.model,
            "--backend", point.backend, "--base-url", point.base_url, "--random-input-len", str(point.isl),
            "--random-output-len", str(point.osl), "--random-range-ratio", point.random_range_ratio,
            "--num-prompts", str(point.num_prompts), "--max-concurrency", str(point.conc),
            "--num-warmups", str(2 * point.conc), "--result-dir", str(point.result.parent),
            "--result-filename", point.result.name, "--tokenizer", point.tokenizer]
"#;

/// InferenceX's `infx.bench_serving.backend_request_func`: the request functions by backend.
const FAKE_INFX_REQUEST_FUNC: &str = r#"
import asyncio, json, urllib.request
from dataclasses import dataclass

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str

@dataclass
class RequestFuncOutput:
    success: bool = False
    output_tokens: int = 0

def _post(i):
    body = json.dumps({"model": i.model, "prompt": i.prompt, "prompt_len": i.prompt_len,
                       "max_tokens": i.output_len}).encode()
    req = urllib.request.Request(i.api_url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)

async def async_request_openai_completions(request_func_input, pbar=None):
    out = await asyncio.to_thread(_post, request_func_input)
    return RequestFuncOutput(success=True, output_tokens=out["usage"]["completion_tokens"])

ASYNC_REQUEST_FUNCS = {"vllm": async_request_openai_completions}
"#;

/// InferenceX's `infx.bench_serving.benchmark_serving`: `2 x conc` warm-ups of one prompt
/// (`WARMUP`), then with `--profile` the hook's `/start_profile`, the burst (`REQ <i>`, random
/// lengths within the range ratio) at `--max-concurrency`, and `/stop_profile`; the result JSON.
/// `FAKE_EXIT_NOISE`: the interpreter prints after the driver at exit, as GB300's triton does
/// (nanobind's leak report).
const FAKE_INFX_BENCH_SERVING: &str = r#"
import argparse, asyncio, atexit, json, os, random, sys, time
from .backend_request_func import ASYNC_REQUEST_FUNCS, RequestFuncInput

if os.environ.get("FAKE_EXIT_NOISE"):
    atexit.register(lambda: print("nanobind: leaked 4 instances! See https://nanobind.readthedocs.io", file=sys.stderr))

p = argparse.ArgumentParser()
for f in ["--model", "--backend", "--base-url", "--random-input-len", "--random-output-len",
          "--random-range-ratio", "--num-prompts", "--max-concurrency", "--num-warmups", "--result-dir",
          "--result-filename", "--tokenizer"]:
    p.add_argument(f)
p.add_argument("--profile", action="store_true")
a = p.parse_args()
isl, osl, ratio = int(a.random_input_len), int(a.random_output_len), float(a.random_range_ratio)
conc, n = int(a.max_concurrency), int(a.num_prompts)
rng = random.Random(0)
url = a.base_url + "/v1/completions"

async def main():
    request_func = ASYNC_REQUEST_FUNCS[a.backend]
    warm = RequestFuncInput("WARMUP", url, isl, osl, a.model)
    await asyncio.gather(*[request_func(request_func_input=warm) for _ in range(int(a.num_warmups))])
    if a.profile:
        await request_func(request_func_input=RequestFuncInput("", a.base_url + "/start_profile", 0, 0, a.model))
    sem = asyncio.Semaphore(conc)
    async def one(i):
        inp = RequestFuncInput(f"REQ {i}", url, rng.randint(int(isl * ratio), isl), rng.randint(int(osl * ratio), osl),
                               a.model)
        async with sem:
            return await request_func(request_func_input=inp)
    t0 = time.time()
    outs = await asyncio.gather(*[one(i) for i in range(n)])
    if a.profile:
        await request_func(request_func_input=RequestFuncInput("", a.base_url + "/stop_profile", 0, 0, a.model))
    toks = sum(o.output_tokens for o in outs)
    json.dump({"output_throughput": toks / (time.time() - t0), "completed": len(outs)},
              open(os.path.join(a.result_dir, a.result_filename), "w"))

asyncio.run(main())
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
        write("py/sglang_router/__init__.py", "");
        write("py/sglang_router/launch_router.py", FAKE_ROUTER);
        write("py/vllm/__init__.py", "");
        write("bin/vllm", &format!("#!/usr/bin/env python3\n{FAKE_OPENAI_SERVER}"));
        write("py/atom/__init__.py", "");
        write("py/atom/entrypoints/__init__.py", "");
        write("py/atom/entrypoints/openai_server.py", FAKE_OPENAI_SERVER);
        write("infx/infx/__init__.py", "");
        write("infx/infx/bench/__init__.py", "");
        write("infx/infx/bench/__main__.py", FAKE_INFX);
        write("infx/infx/bench/fixed_seq.py", FAKE_INFX_FIXED_SEQ);
        write("infx/infx/bench_serving/__init__.py", "");
        write(
            "infx/infx/bench_serving/backend_request_func.py",
            FAKE_INFX_REQUEST_FUNC,
        );
        write("infx/infx/bench_serving/benchmark_serving.py", FAKE_INFX_BENCH_SERVING);
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
            .env("AKAO_REPO_ROOT", self.root.join("repo-clone"))
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
        "{a}{}{GSM8K}{FIXED_SEQ}repeats = 2\nmin_output_tok_s = 90\n",
        server("b", 1, "[server.args]\npage-size = 64")
    );
    let (code, out) = sb.run(&plan, &[("FAKE_INFX_COLD", "1")]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("[gsm8k] score 0.95 (min 0.90) PASS"), "{out}");

    // fixed-seq with two runs per point: the first is the number (summary.csv and the gate),
    // the second (twice as fast here) is only kept and printed.
    let summary = sb.read("results/02_fixed-seq_b/summary.csv");
    assert_eq!(summary.lines().count(), 3, "{summary}");
    assert!(
        summary.contains("b_isl128_osl32_c2_r1,128,32,2,100.0,")
            && summary.contains("b_isl128_osl32_c4_r1,128,32,4,200.0,")
            && !summary.contains("_r2,"),
        "{summary}"
    );
    assert!(sb
        .work()
        .join("results/02_fixed-seq_b/b_isl128_osl32_c2_r2.json")
        .exists());
    assert!(out.contains("isl=128 osl=32 conc=2 100.0 tok/s (min 90) PASS"), "{out}");
    // InferenceX gets 10 prompts per concurrency and the vllm backend.
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

/// A lockstep's clients run at the same time, every run started together: the fast server's
/// next run waits for the slow one's.  One failing stops the other before its next run.
#[test]
fn lockstep_clients_run_together() {
    let _servers = servers_shared();
    let sb = Sandbox::new("lockstep");
    let (a, port_a) = server_of("a", "m/base", 0, "");
    let (b, port_b) = server_of("b", "m/base", 1, "");
    let fixed = |s: &str| {
        format!(
            "[[client]]\nkind = 'fixed-seq'\nserver = '{s}'\nisl_osl = [[128, 32]]\nconc = [1, 2]\n\
             repeats = 2\nlockstep = 'ab'\n"
        )
    };
    let plan = format!("{a}{b}{GSM8K}{}{}", fixed("a"), fixed("b"));
    let slow = format!("{port_b}:0.4");
    // a is the slower server at conc 1, b at conc 2; both runs of b's conc 2 are slow.
    let scale = format!("{port_a}:1:0.5,{port_b}:2:0.25");
    let (code, out) = sb.run(
        &plan,
        &[
            ("FAKE_INFX_SLOW", &slow),
            ("FAKE_INFX_SCALE", &scale),
            ("FAKE_INFX_COLD", "1"),
        ],
    );
    assert_eq!(code, 0, "{out}");
    let step = sb.read("scripts/step.sh");
    assert!(
        step.contains(
            "run_client client_01_gsm8k_a.sh\nrun_lockstep ab client_02_fixed-seq_a.sh client_03_fixed-seq_b.sh\n"
        ),
        "{step}"
    );
    for s in ["a", "b"] {
        let n = if s == "a" { "02" } else { "03" };
        let summary = sb.read(&format!("results/{n}_fixed-seq_{s}/summary.csv"));
        assert_eq!(summary.lines().count(), 3, "{summary}");
    }
    // The lockstep's summary: per point the slower server's first run (run 2 is faster, but
    // is not the number).
    let both = sb.read("results/lockstep_ab/summary.csv");
    assert_eq!(both.lines().count(), 3, "{both}");
    assert!(
        both.contains("a_isl128_osl32_c1_r1,128,32,1,25.0,") && both.contains("b_isl128_osl32_c2_r1,128,32,2,25.0,"),
        "{both}"
    );
    // Per run: (start, end) of each server, in run order.
    let points = fs::read_to_string(sb.root.join("state/points")).unwrap();
    let times = |s: &str| -> Vec<(f64, f64)> {
        ["c1_r1", "c1_r2", "c2_r1", "c2_r2"]
            .iter()
            .map(|run| {
                let line = points
                    .lines()
                    .find(|l| l.starts_with(&format!("{s}_isl128_osl32_{run}.json ")))
                    .unwrap_or_else(|| panic!("{s} {run} not in {points}"));
                let w: Vec<f64> = line.split(' ').skip(1).map(|x| x.parse().unwrap()).collect();
                (w[0], w[1])
            })
            .collect()
    };
    let (ta, tb) = (times("a"), times("b"));
    for k in 0..4 {
        assert!(
            (ta[k].0 - tb[k].0).abs() < 0.3,
            "run {k} did not start together: {points}"
        );
        if k > 0 {
            assert!(
                ta[k].0 >= tb[k - 1].1 - 0.05,
                "a's run {k} started before b's run {} ended: {points}",
                k - 1
            );
        }
    }
    sb.assert_no_leftovers();

    // a fails at conc 2: b finishes the run it is in, then stops; the step fails.
    fs::remove_dir_all(sb.work().join("results")).unwrap();
    let fail = format!("{port_a}:2");
    let (code, out) = sb.run(&plan, &[("FAKE_INFX_SLOW", &slow), ("FAKE_INFX_FAIL", &fail)]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("another client of the lockstep failed; stopping"), "{out}");
    assert!(out.contains("lockstep ab failed; stopping"), "{out}");
    assert!(sb
        .work()
        .join("results/03_fixed-seq_b/b_isl128_osl32_c2_r1.json")
        .exists());
    assert!(!sb
        .work()
        .join("results/03_fixed-seq_b/b_isl128_osl32_c2_r2.json")
        .exists());
    sb.assert_no_leftovers();
}

/// Every server starts on empty compile caches of its own, in the step's logs/, gone after
/// the step; a plan that points a cache elsewhere keeps it across runs.
#[test]
fn servers_start_with_empty_compile_caches() {
    let _servers = servers_shared();
    let sb = Sandbox::new("caches");
    let shared = sb.root.join("shared-triton");
    let caches_of = |name: &str| -> serde_json::Value {
        let port: u16 = sb
            .read("plan.lock.toml")
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{name} = ")).map(|p| p.trim().parse().unwrap()))
            .unwrap_or_else(|| panic!("no port for {name}"));
        let rec = fs::read(sb.root.join(format!("state/server_{port}.json"))).unwrap();
        serde_json::from_slice::<serde_json::Value>(&rec).unwrap()["caches"].clone()
    };
    let a = "[[server]]\nname = 'a'\nprofile = 'm/base'\ngpus = [0]\n";
    let b = "[[server]]\nname = 'b'\nprofile = 'm/base'\ngpus = [1]\n";
    let logs = sb.work().join("logs");
    // Twice: the second run's servers must not see the kernels the first run's compiled.
    for _ in 0..2 {
        let (code, out) = sb.run(&format!("{a}{b}{FIXED_SEQ}"), &[]);
        assert_eq!(code, 0, "{out}");
        for name in ["a", "b"] {
            let c = caches_of(name);
            for (var, sub) in [
                ("TRITON_CACHE_DIR", "triton"),
                ("XDG_CACHE_HOME", "xdg"),
                ("SGLANG_CACHE_DIR", "sglang"),
            ] {
                let want = logs.join(format!("cache_{name}/{sub}"));
                assert_eq!(c[var][0], want.to_str().unwrap(), "{var} of {name}: {c}");
                assert_eq!(
                    c[var][1],
                    serde_json::json!([]),
                    "{var} of {name} not empty at start: {c}"
                );
            }
        }
        assert!(sb
            .read("logs/server_a.log")
            .contains("[server a] env TRITON_CACHE_DIR="));
        let left: Vec<_> = fs::read_dir(&logs)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("cache_"))
            .collect();
        assert!(left.is_empty(), "caches left after the step: {left:?}");
    }
    // The plan's env wins: a's Triton cache is shared and keeps its kernels; the rest stay fresh.
    let a_shared = format!("{a}[server.env]\nTRITON_CACHE_DIR = '{}'\n", shared.display());
    for run in 0..2 {
        let (code, out) = sb.run(&format!("{a_shared}{b}{FIXED_SEQ}"), &[]);
        assert_eq!(code, 0, "{out}");
        let c = caches_of("a");
        assert_eq!(c["TRITON_CACHE_DIR"][0], shared.to_str().unwrap(), "{c}");
        let want = if run == 0 {
            serde_json::json!(null)
        } else {
            serde_json::json!(["kernel.bin"])
        };
        assert_eq!(c["TRITON_CACHE_DIR"][1], want, "{c}");
        assert_eq!(c["XDG_CACHE_HOME"][1], serde_json::json!([]), "{c}");
    }
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

/// Ctrl-C while a lockstep runs: its clients (own process groups, out of the terminal's
/// reach) are stopped with the servers.
#[test]
fn ctrl_c_during_lockstep_stops_its_clients() {
    let _servers = servers_shared();
    use std::os::unix::process::CommandExt;
    let sb = Sandbox::new("int-lockstep");
    let (a, _) = server_of("a", "m/base", 0, "");
    let (b, port_b) = server_of("b", "m/base", 1, "");
    let fixed = |s: &str| {
        format!("[[client]]\nkind = 'fixed-seq'\nserver = '{s}'\nisl_osl = [[128, 32]]\nconc = [1]\nlockstep = 'ab'\n")
    };
    fs::write(
        sb.work().join("plan.toml"),
        format!("{a}{b}{}{}", fixed("a"), fixed("b")),
    )
    .unwrap();
    let mut child = sb
        .oaka(&["run"])
        .env("FAKE_INFX_SLOW", format!("{port_b}:60"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .unwrap();
    // a's point is done when b's (60 s) is surely under way.
    let points = sb.root.join("state/points");
    let deadline = Instant::now() + Duration::from_secs(20);
    while !points.exists() {
        assert!(Instant::now() < deadline, "the lockstep never started");
        std::thread::sleep(Duration::from_millis(100));
    }
    let started = Instant::now();
    Command::new("kill")
        .args(["-INT", "--", &format!("-{}", child.id())])
        .status()
        .unwrap();
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(130));
    assert!(started.elapsed() < Duration::from_secs(30), "waited for b's point");
    sb.assert_no_leftovers();
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
        "ok    trace       InferenceX's client has what trace clients wrap",
        "ok    sgl-eval",
        "ok    pidfd       python3 and the kernel have pidfds",
        "warn  repo        no akao-workflow clone at",
    ] {
        assert!(text.contains(want), "{want:?} not in {text}");
    }
    // A python3 without pidfds (old Python or kernel): no step could start a server.
    let real = Command::new("sh").args(["-c", "command -v python3"]).output().unwrap();
    let real = String::from_utf8(real.stdout).unwrap().trim().to_string();
    let nopidfd = sb.root.join("nopidfd");
    fs::create_dir_all(&nopidfd).unwrap();
    fs::write(
        nopidfd.join("python3"),
        format!(
            "#!/bin/sh\ncase \"$*\" in *pidfd_open*) echo 'AttributeError: no pidfd_open' >&2; exit 1 ;; esac\n\
             exec {real} \"$@\"\n"
        ),
    )
    .unwrap();
    Command::new("chmod")
        .arg("+x")
        .arg(nopidfd.join("python3"))
        .status()
        .unwrap();
    let path = format!(
        "{}:{}:{}",
        nopidfd.display(),
        sb.root.join("bin").display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = sb.oaka(&["doctor"]).env("PATH", path).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(
        text.contains("FAIL  pidfd       python3 cannot use pidfds (AttributeError: no pidfd_open)"),
        "{text}"
    );
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

    // An InferenceX whose client lost the --profile hook: trace clients cannot work.
    let bs = sb.root.join("infx/infx/bench_serving/benchmark_serving.py");
    fs::write(
        &bs,
        fs::read_to_string(&bs).unwrap().replace("/start_profile", "/begin"),
    )
    .unwrap();
    let text = String::from_utf8_lossy(&sb.oaka(&["doctor"]).output().unwrap().stdout).into_owned();
    assert!(text.contains("warn  trace       trace clients cannot wrap InferenceX's client (AssertionError: benchmark_serving has no --profile hook"), "{text}");

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

/// A PD profile's prefill and decode servers start first, each on its share of the GPUs and
/// with its role's flags; the router starts once they are ready, on the plan's port, and is
/// ready when a generation goes through.  The clients reach the servers through it.
#[test]
fn pd_servers_start_behind_their_router() {
    let _servers = servers_shared();
    let sb = Sandbox::new("pd");
    fs::write(
        sb.root.join("lib/profiles/m/pd.toml"),
        "extends = 'm/base'\n[pd.decode.args]\nmem-fraction-static = 0.8\n[pd.router.args]\npolicy = 'round_robin'\n",
    )
    .unwrap();
    let port = ephemeral_port();
    let plan = format!(
        "[[server]]\nname = 'pd'\nprofile = 'm/pd'\ngpus = [0, 1]\nport = {port}\n\
         [[client]]\nkind = 'fixed-seq'\nserver = 'pd'\nisl_osl = [[128, 32]]\nconc = [4]\n"
    );
    // The servers' own ports, which compile would pick in 29900-30100 where tests running in
    // parallel pick theirs: from the lock instead, which compile keeps.
    fs::write(
        sb.work().join("plan.lock.toml"),
        format!(
            "[port]\n\"pd.prefill0\" = {}\n\"pd.prefill0.bootstrap\" = {}\n\"pd.decode0\" = {}\n",
            ephemeral_port(),
            ephemeral_port(),
            ephemeral_port()
        ),
    )
    .unwrap();
    // The router answers 503 for its first second, as when its servers are not registered yet.
    let (code, out) = sb.run(&plan, &[("FAKE_ROUTER_DELAY", "1")]);
    assert_eq!(code, 0, "{out}");
    let at = |needle: &str| out.find(needle).unwrap_or_else(|| panic!("{needle:?} not in {out}"));
    assert!(at("server pd.router started") > at("server pd.prefill0 ready"));
    assert!(at("server pd.router started") > at("server pd.decode0 ready"));
    at("server pd.router ready (POST /generate)");

    let argv = |file: String| -> Vec<String> {
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(sb.root.join("state").join(file)).unwrap()).unwrap();
        v["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap().to_string())
            .collect()
    };
    let router = argv(format!("router_{port}.json")).join(" ");
    let lock = sb.read("plan.lock.toml");
    let lock: toml::Table = toml::from_str(&lock).unwrap();
    let slot = |k: &str| lock["port"][k].as_integer().unwrap();
    let (p, b, d) = (slot("pd.prefill0"), slot("pd.prefill0.bootstrap"), slot("pd.decode0"));
    assert!(
        router.contains(&format!(
            "--pd-disaggregation --prefill http://127.0.0.1:{p} {b} --decode http://127.0.0.1:{d} \
             --host 127.0.0.1 --port {port} --policy round_robin"
        )),
        "{router}"
    );
    let prefill = argv(format!("server_{p}.json")).join(" ");
    assert!(
        prefill.contains(&format!(
            "--disaggregation-mode prefill --disaggregation-bootstrap-port {b}"
        )),
        "{prefill}"
    );
    let decode = argv(format!("server_{d}.json")).join(" ");
    assert!(
        decode.contains("--disaggregation-mode decode --trust-remote-code --mem-fraction-static 0.8"),
        "{decode}"
    );
    let records = sb.server_record().join("\n");
    assert!(
        records.contains(r#""gpus": "0""#) && records.contains(r#""gpus": "1""#),
        "{records}"
    );
    let point = sb.read("results/01_fixed-seq_pd/pd_isl128_osl32_c4_r1.json");
    assert!(
        point.contains(&format!("http://127.0.0.1:{port}")),
        "the client goes through the router: {point}"
    );
    assert!(sb
        .read("logs/server_pd.router.log")
        .contains("[server pd.router] cmd python3 -m sglang_router.launch_router"));
    sb.assert_no_leftovers();

    // Agents see the PD servers.
    let v: serde_json::Value = serde_json::from_slice(&sb.oaka(&["check", "--json"]).output().unwrap().stdout).unwrap();
    assert_eq!(v["servers"][0]["pd"][1]["name"], "pd.decode0", "{v}");
    assert_eq!(v["servers"][0]["pd"][1]["gpus"], serde_json::json!([1]), "{v}");

    // A prefill server that dies at startup stops the run; the router never starts.
    fs::write(
        sb.root.join("lib/profiles/m/pd.toml"),
        "extends = 'm/base'\n[pd.prefill.args]\ncrash = true\n",
    )
    .unwrap();
    let (code, out) = sb.run(&plan, &[("FAKE_READY_DELAY", "2")]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("server pd.prefill0 died during startup"), "{out}");
    assert!(!out.contains("server pd.router started"), "{out}");
    sb.assert_no_leftovers();

    // A decode server that exits once ready fails the run while its router is being waited for.
    fs::write(
        sb.root.join("lib/profiles/m/pd.toml"),
        "extends = 'm/base'\n[pd.decode.args]\nexit-after-ready = true\n",
    )
    .unwrap();
    let (code, out) = sb.run(&plan, &[("FAKE_ROUTER_DELAY", "3")]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("server pd.decode0 died during startup"), "{out}");
    assert!(!out.contains("server pd.router ready"), "{out}");
    sb.assert_no_leftovers();
}

/// Ctrl-C while step.sh waits for a router's answer stops everything at once, not after the
/// probe's timeout.
#[test]
fn ctrl_c_while_probing_a_router_stops_servers() {
    let _servers = servers_shared();
    use std::os::unix::process::CommandExt;
    let sb = Sandbox::new("pd-int");
    fs::write(sb.root.join("lib/profiles/m/pd.toml"), "extends = 'm/base'\n[pd]\n").unwrap();
    let port = ephemeral_port();
    let plan = format!("[[server]]\nname = 'pd'\nprofile = 'm/pd'\ngpus = [0, 1]\nport = {port}\n{GSM8K}")
        .replace("server = 'a'", "server = 'pd'");
    fs::write(sb.work().join("plan.toml"), plan).unwrap();
    fs::write(
        sb.work().join("plan.lock.toml"),
        format!(
            "[port]\n\"pd.prefill0\" = {}\n\"pd.prefill0.bootstrap\" = {}\n\"pd.decode0\" = {}\n",
            ephemeral_port(),
            ephemeral_port(),
            ephemeral_port()
        ),
    )
    .unwrap();
    let mut child = sb
        .oaka(&["run"])
        .env("FAKE_ROUTER_HANG", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let probing = || {
        fs::read_dir(sb.root.join("state"))
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().starts_with("router_"))
    };
    while !probing() {
        assert!(Instant::now() < deadline, "router never started");
        std::thread::sleep(Duration::from_millis(100));
    }
    std::thread::sleep(Duration::from_secs(2));
    let group = format!("-{}", child.id());
    let t0 = Instant::now();
    for _ in 0..2 {
        Command::new("kill").args(["-INT", "--", &group]).status().unwrap();
        std::thread::sleep(Duration::from_millis(300));
    }
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(130));
    assert!(t0.elapsed() < Duration::from_secs(15), "took {:?}", t0.elapsed());
    sb.assert_no_leftovers();
}

/// A plan override that changes only a PD role's value (here: back to the profile's own value
/// over the decode role's) is listed, and `profile save` keeps it.
#[test]
fn role_only_overrides_are_listed_and_saved() {
    let sb = Sandbox::new("pd-save");
    fs::write(
        sb.root.join("lib/profiles/m/pd.toml"),
        "extends = 'm/base'\n[args]\nmem-fraction-static = 0.9\n[pd.decode.args]\nmem-fraction-static = 0.8\n",
    )
    .unwrap();
    fs::write(
        sb.work().join("plan.toml"),
        "[[server]]\nname = 'pd'\nprofile = 'm/pd'\ngpus = [0, 1]\n[server.args]\nmem-fraction-static = 0.9\n",
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&sb.oaka(&["check", "--json"]).output().unwrap().stdout).unwrap();
    assert_eq!(
        v["servers"][0]["overrides"],
        serde_json::json!(["args --mem-fraction-static 0.9"]),
        "{v}"
    );
    let out = sb
        .oaka(&["profile", "save", "pd", "--as", "m/pd-saved"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let show =
        String::from_utf8_lossy(&sb.oaka(&["profile", "show", "m/pd-saved"]).output().unwrap().stdout).into_owned();
    assert!(
        show.contains("#   decode: --disaggregation-mode decode\n"),
        "the decode role is back to 0.9: {show}"
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
    // PD profiles' router is found; Mooncake (unless this machine has one) is missing for them.
    let want = format!(
        "ok    pd router   {}",
        sb.root.join("py/sglang_router/__init__.py").display()
    );
    assert!(text.contains(&want), "{want:?} not in {text}");
    fs::write(sb.root.join("lib/profiles/m/pd.toml"), "extends = 'm/base'\n[pd]\n").unwrap();
    let text = String::from_utf8_lossy(&sb.oaka(&["doctor"]).output().unwrap().stdout).into_owned();
    if !text.contains("ok    pd mooncake") {
        assert!(
            text.contains("warn  pd mooncake python3 cannot find mooncake; PD profiles m/pd use it"),
            "{text}"
        );
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

#[test]
fn the_library_is_the_repo_checkouts_oaka_dir() {
    let sb = Sandbox::new("library");
    // Without OAKA_LIB: $AKAO_REPO_ROOT/oaka, where a saved profile is a file git tracks.
    let repo = sb.root.join("akao-workflow");
    fs::create_dir_all(repo.join("oaka")).unwrap();
    fs::rename(sb.root.join("lib/profiles"), repo.join("oaka/profiles")).unwrap();
    let out = sb
        .oaka(&["profile", "ls"])
        .env_remove("OAKA_LIB")
        .env("AKAO_REPO_ROOT", &repo)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    assert!(
        text.starts_with(&format!("# {}", repo.join("oaka/profiles").display())),
        "{text}"
    );
    assert!(text.contains("m/vllm"), "{text}");
}

/// `[[client]] kind = 'trace'` on server a, one point.
fn trace_client(isl_osl: &str, conc: &str) -> String {
    format!("[[client]]\nkind = 'trace'\nserver = 'a'\nisl_osl = {isl_osl}\nconc = {conc}\nrange_ratio = 0.5\n")
}

/// The stand-in server's events: (what, word, time) per line of state/events_<port>.
fn events(sb: &Sandbox, port: u16) -> Vec<(String, String, f64)> {
    fs::read_to_string(sb.root.join(format!("state/events_{port}")))
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let w: Vec<&str> = l.split(' ').collect();
            let t = w.last().unwrap().parse().unwrap();
            (
                w[0].to_string(),
                if w.len() > 2 { w[1].to_string() } else { String::new() },
                t,
            )
        })
        .collect()
}

/// A trace client: oaka sets the profiler env, the probe passes, every point gets its warm
/// run and then flushed profiled passes whose captures are armed only within 50-85% of the
/// measured burst (never in the warm-ups), until a clean decode capture and 10 prefill steps
/// are on disk; one trace file per TP rank.
#[test]
fn trace_client_captures_the_steady_state() {
    let _servers = servers_shared();
    let sb = Sandbox::new("trace");
    let (a, port) = server_of("a", "m/base", 0, "");
    let plan = format!(
        "{}{}",
        a.replace("gpus = [0]", "gpus = [0, 1]"),
        trace_client("[[64, 40]]", "[4]")
    );
    let (code, out) = sb.run(&plan, &[("FAKE_STEP_S", "0.01"), ("FAKE_EXIT_NOISE", "1")]);
    assert_eq!(code, 0, "{out}");
    assert!(sb
        .read("scripts/server_a.sh")
        .contains("\nexport SGLANG_PROFILE_V2=1\n"));
    // Each pass's line is the driver's last, whatever the interpreter prints after it.
    assert!(
        out.contains("[trace_point pass 1] ") && !out.contains("nanobind"),
        "{out}"
    );
    let record: serde_json::Value = serde_json::from_str(&sb.server_record().join("")).unwrap();
    assert_eq!(record["profile_v2"], "1");
    for probe in [
        "A_prefill_ends_at_decode",
        "B_num_steps_9_gives_10",
        "C_decode_ends_at_prefill",
    ] {
        assert!(out.contains(&format!("[trace_probe] {probe}: PASS")), "{out}");
    }

    let point = sb.work().join("results/01_trace_a/a_isl64_osl40_c4");
    assert!(point.join("warm.json").exists());
    let state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(point.join("state.json")).unwrap()).unwrap();
    assert!(state["decode_clean"].is_string(), "{state}");
    assert!(state["prefill_steps"].as_u64().unwrap() >= 10, "{state}");
    for cap in state["captures"].as_array().unwrap() {
        let files = cap["files"].as_array().unwrap();
        assert_eq!(files.len(), 2, "one trace per TP rank: {cap}");
        assert!(files[0].as_str().unwrap().contains("-TP-0-"), "{cap}");
        if cap["use"] == "clean" {
            assert_eq!(cap["bs"], serde_json::json!(vec![4; 10]), "{cap}");
        }
    }
    let csv = sb.read("results/01_trace_a/captures.csv");
    assert!(csv.lines().count() > 2 && csv.contains(",clean"), "{csv}");

    // Each pass: a flush, then the warm-ups, then arms only within 50-85% of the 40 requests.
    let ev = events(&sb, port);
    let passes: Vec<usize> = ev
        .iter()
        .enumerate()
        .filter(|(_, e)| e.0 == "flush")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(passes.len() as u64, state["passes"].as_u64().unwrap(), "{ev:?}");
    for (k, &start) in passes.iter().enumerate() {
        let end = passes.get(k + 1).copied().unwrap_or(ev.len());
        let mut done = 0;
        let mut warmups_done = 0;
        for e in &ev[start..end] {
            match (e.0.as_str(), e.1.as_str()) {
                ("done", "REQ") => done += 1,
                ("done", "WARMUP") => {
                    assert_eq!(done, 0, "a warm-up after the burst began: {ev:?}");
                    warmups_done += 1
                }
                ("arm", prefix) if prefix.starts_with("pass") => {
                    assert_eq!(warmups_done, 8, "armed before the warm-ups were done: {ev:?}");
                    assert!((20..=34).contains(&done), "armed at {done} of 40 done: {ev:?}");
                }
                _ => {}
            }
        }
    }
    sb.assert_no_leftovers();
}

/// A point whose traces stay incomplete gets every pass, each flushed, and the client exits 3;
/// a profiler that breaks the practice's rules stops the client at the probe, before any run.
#[test]
fn trace_client_reports_incomplete_traces_and_a_failed_probe() {
    let _servers = servers_shared();
    let sb = Sandbox::new("trace-bad");
    let (a, port) = server_of("a", "m/base", 0, "");
    let plan = format!("{a}{}", trace_client("[[64, 24]]", "[2]"));
    // Prefill captures that also hold a decode step are never used.
    let (code, out) = sb.run(&plan, &[("FAKE_STEP_S", "0.004"), ("FAKE_PASS_PREFILL_MIXED", "1")]);
    assert_eq!(code, 3, "{out}");
    assert!(out.contains("a_isl64_osl24_c2: INCOMPLETE after 3 passes"), "{out}");
    let point = sb.work().join("results/01_trace_a/a_isl64_osl24_c2");
    let state: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(point.join("state.json")).unwrap()).unwrap();
    assert_eq!(state["passes"], 3);
    assert_eq!(state["prefill_steps"], 0);
    assert_eq!(events(&sb, port).iter().filter(|e| e.0 == "flush").count(), 3);
    assert!(sb
        .read("results/01_trace_a/captures.csv")
        .contains("unexpected steps: not used"));
    sb.assert_no_leftovers();

    // A server without profiler V2 (the plan cannot unset it, so its script is edited).
    let script = sb.work().join("scripts/server_a.sh");
    let text = sb.read("scripts/server_a.sh").replace("export SGLANG_PROFILE_V2=1", "");
    fs::write(&script, text).unwrap();
    let out = sb.sandboxed("bash").arg("scripts/run_all.sh").output().unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("[trace_probe] A_prefill_ends_at_decode: FAIL"), "{text}");
    assert!(
        text.contains("FATAL: the profiler does not behave as the capture practice assumes"),
        "{text}"
    );
    assert!(!point.exists(), "no warm run after a failed probe");
    sb.assert_no_leftovers();

    // A profiler that records the CPU only: traces without kernels are refused before any run.
    let (code, out) = sb.run(&plan, &[("FAKE_STEP_S", "0.004"), ("FAKE_TRACE_NO_GPU", "1")]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("[trace_probe] A_prefill_ends_at_decode: PASS"), "{out}");
    assert!(
        out.contains("[trace_probe] D_gpu_kernels_recorded: FAIL (GPU kernel events in B's trace; got 0)"),
        "{out}"
    );
    assert!(out.contains("sgl-project/sglang#44292"), "{out}");
    assert!(!point.exists(), "no warm run after a failed probe");
    sb.assert_no_leftovers();
}
