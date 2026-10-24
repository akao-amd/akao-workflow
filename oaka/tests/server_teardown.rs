//! Hermetic generated-server teardown: real CPU processes/pidfds, synthetic grace clock.
//! No framework, GPU, installed oaka, network, or process outside the fixture is used.
use std::fs;
use std::process::Command;

const CHECKS: &str = r#"
import importlib.util, json, os, shlex, shutil, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch

helper = Path(sys.argv.pop(1))

class Clock:
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds
        time.sleep(0.001)

class Teardown(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location('owned_server', helper)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.processes = []
        self.valid_states = []

    def tearDown(self):
        # Only recorded fixture groups are cleaned; corrupted test records are
        # never used to authorize this safety cleanup.
        for rec in self.valid_states:
            for member in self.m.members(rec):
                self.m.send(member, signal.SIGKILL)
        for proc in self.processes:
            proc.wait(timeout=5)
        self.tmp.cleanup()

    def wait_for(self, predicate):
        end = time.monotonic() + 5
        while not predicate():
            self.assertLess(time.monotonic(), end, 'fixture failed to become ready')
            time.sleep(0.01)

    def bootstrap(self, code, go=True):
        state = self.root / f'identity-{len(self.processes)}.json'
        source = self.root / f'server-{len(self.processes)}.py'
        source.write_text(code)
        script = source.with_suffix('.sh')
        script.write_text('exec ' + shlex.quote(sys.executable) + ' -I ' + shlex.quote(str(source)) + '\n')
        proc = subprocess.Popen([sys.executable, '-I', '-S', str(helper), 'start', str(state), str(script),
                                 str(os.getpid())])
        self.processes.append(proc)
        self.wait_for(state.exists)
        rec = self.m.load(state)
        self.valid_states.append(rec)
        if go:
            Path(str(state) + '.go').touch()
        return proc, state, rec

    def family(self, leader_exits=False):
        child = self.root / 'child.py'
        child.write_text('import os,signal,time\n'
                         'signal.signal(signal.SIGTERM, lambda *_: None)\n'
                         f'open({str(self.root / "child.pid")!r}, "w").write(str(os.getpid()))\n'
                         'while True: time.sleep(.05)\n')
        code = ('import os,signal,subprocess,sys,time\n'
                'signal.signal(signal.SIGTERM, lambda *_: None)\n'
                f'child=subprocess.Popen([sys.executable,"-I",{str(child)!r}])\n'
                f'while not os.path.exists({str(self.root / "child.pid")!r}): time.sleep(.01)\n')
        code += 'sys.exit(0)\n' if leader_exits else 'while True: time.sleep(.05)\n'
        proc, state, rec = self.bootstrap(code)
        self.wait_for((self.root / 'child.pid').exists)
        child_pid = int((self.root / 'child.pid').read_text())
        if leader_exits:
            self.assertEqual(proc.wait(timeout=5), 0)
        return proc, state, rec, child_pid

    def stop_with_clock(self, state):
        self.assertEqual(self.m.GRACE_SECONDS, 60)
        clock, events = Clock(), []
        original = self.m.send
        def send(rec, sig):
            events.append((rec['pid'], sig, clock.now))
            original(rec, sig)
        with patch.object(self.m, 'time', clock), patch.object(self.m, 'send', send):
            self.m.stop([(self.m.load(state)['pid'], str(state))])
        return events

    def test_term_only_leader_then_kill_members_after_full_grace(self):
        proc, state, rec, child = self.family()
        events = self.stop_with_clock(state)
        self.assertEqual([p for p,s,t in events if s == signal.SIGTERM], [rec['pid']])
        kills = [(p,t) for p,s,t in events if s == signal.SIGKILL]
        self.assertTrue({rec['pid'], child} <= {p for p,t in kills})
        self.assertTrue(all(t >= 60 for p,t in kills), events)
        self.assertFalse(self.m.members(rec))
        self.assertEqual(proc.wait(timeout=5), -signal.SIGKILL)

    def test_dead_leader_live_child_is_still_owned(self):
        proc, state, rec, child = self.family(leader_exits=True)
        self.assertTrue(self.m.running(self.m.identity(child)))
        events = self.stop_with_clock(state)
        self.assertFalse([e for e in events if e[1] == signal.SIGTERM])
        self.assertTrue(any(p == child and s == signal.SIGKILL and t >= 60 for p,s,t in events))
        self.assertFalse(self.m.members(rec))

    def test_wrong_start_decoy_refuses_before_any_signal(self):
        proc, state, rec, child = self.family()
        wrong = dict(rec, start=rec['start'] + 1)
        state.write_text(json.dumps(wrong))
        with patch.object(self.m, 'send') as send:
            with self.assertRaises(self.m.OwnershipError):
                self.m.stop([(proc.pid, str(state))])
            send.assert_not_called()
        self.assertIsNone(proc.poll())
        self.assertTrue(self.m.running(self.m.identity(child)))

    def test_pidfd_revalidation_never_signals_changed_identity(self):
        proc, state, rec, child = self.family()
        with patch.object(self.m, 'identity', return_value=dict(rec, start=rec['start'] + 1)):
            with patch.object(self.m.signal, 'pidfd_send_signal') as send:
                self.m.send(rec, signal.SIGTERM)
                send.assert_not_called()
        self.assertIsNone(proc.poll())

    def test_unknown_state_fails_closed(self):
        proc, state, rec, child = self.family()
        bad = self.root / 'bad.json'
        for text in (None, '{', '{}', json.dumps(dict(rec, start=0)), json.dumps(dict(rec, uid=rec['uid'] + 1))):
            if text is not None:
                bad.write_text(text)
            with patch.object(self.m, 'send') as send:
                with self.assertRaises(self.m.OwnershipError):
                    self.m.stop([(proc.pid, str(bad))])
                send.assert_not_called()
        self.assertIsNone(proc.poll())

    def test_cancel_before_exec_runs_no_server_code(self):
        marker = self.root / 'server-started'
        proc, state, rec = self.bootstrap(f'open({str(marker)!r}, "w").close()\n', go=False)
        Path(str(state) + '.stop').touch()
        self.assertNotEqual(proc.wait(timeout=5), 0)
        self.assertFalse(marker.exists())
        self.m.stop([(proc.pid, str(state))])

    def test_missing_pidfd_support_refuses_signaling(self):
        proc, state, rec, child = self.family()
        with patch.object(self.m.os, 'pidfd_open', side_effect=OSError('pidfds unavailable')):
            with self.assertRaises(self.m.OwnershipError):
                self.m.stop([(proc.pid, str(state))])
        self.assertIsNone(proc.poll())

    def test_unreadable_group_refuses_before_any_signal(self):
        proc, state, rec, child = self.family()
        with patch.object(self.m.os, 'listdir', side_effect=OSError('group query unavailable')):
            with patch.object(self.m, 'send') as send:
                with self.assertRaises(self.m.OwnershipError):
                    self.m.stop([(proc.pid, str(state))])
                send.assert_not_called()
        self.assertIsNone(proc.poll())

    def test_other_group_record_cannot_replace_registered_server(self):
        proc, state, rec, child = self.family()
        decoy, decoy_state, decoy_rec = self.bootstrap('import time\nwhile True: time.sleep(.05)\n')
        state.write_text(json.dumps(decoy_rec))
        with patch.object(self.m, 'send') as send:
            with self.assertRaises(self.m.OwnershipError):
                self.m.stop([(proc.pid, str(state))])
            send.assert_not_called()
        self.assertIsNone(proc.poll())
        self.assertIsNone(decoy.poll())

    def test_generated_step_unknown_cleanup_keeps_caches_and_failure_codes(self):
        for client_rc, expected_rc in ((0, 1), (3, 3)):
            with self.subTest(client_rc=client_rc):
                work = self.root / f'step-{client_rc}'
                scripts = work / 'scripts'
                shutil.copytree(helper.parent, scripts)
                # Stand-ins for only the server/client scripts. The compiled step
                # and ownership helper are used unchanged, without invoking oaka.
                (scripts / 'server_a.sh').write_text(
                    'mkdir -p "$OAKA_SERVER_CACHE/triton"\n'
                    'touch "$OAKA_SERVER_CACHE/triton/kernel"\n'
                    'echo "The server is fired up and ready"\n'
                    'exec sleep 300\n')
                valid = work / 'valid.json'
                client = work / 'client.py'
                client.write_text(
                    'import json, pathlib\n'
                    f'path = next(pathlib.Path({str(work)!r}).glob("logs/.server_ownership.*/a.json"))\n'
                    'rec = json.loads(path.read_text())\n'
                    f'pathlib.Path({str(valid)!r}).write_text(json.dumps(rec))\n'
                    'rec["start"] += 1\n'
                    'path.write_text(json.dumps(rec))\n')
                (scripts / 'client_01_fixed-seq_a.sh').write_text(
                    f'{shlex.quote(sys.executable)} -I {shlex.quote(str(client))}\nexit {client_rc}\n')
                result = subprocess.run(['bash', str(scripts / 'step.sh')], cwd=work,
                                        capture_output=True, text=True, timeout=10)
                rec = self.m.load(valid)
                self.valid_states.append(rec)
                self.assertEqual(result.returncode, expected_rc, result.stdout + result.stderr)
                self.assertIn('cleanup unverified', result.stdout + result.stderr)
                self.assertTrue((work / 'logs/cache_a/triton/kernel').exists())
                self.assertTrue(list((work / 'logs').glob('.server_ownership.*/a.json')))
                self.assertTrue(self.m.running(self.m.identity(rec['pid'])))

    def test_server_starts_with_default_pipe_and_file_size_signals(self):
        # Python ignores SIGPIPE/SIGXFSZ for itself; exec keeps ignored signals.
        state, out = self.root / 'signals.json', self.root / 'sigign'
        script = self.root / 'signals.sh'
        script.write_text(f'grep SigIgn /proc/self/status > {shlex.quote(str(out))}\n')
        proc = subprocess.Popen([sys.executable, '-I', '-S', str(helper), 'start', str(state), str(script),
                                 str(os.getpid())])
        self.processes.append(proc)
        self.wait_for(state.exists)
        Path(str(state) + '.go').touch()
        self.assertEqual(proc.wait(timeout=5), 0)
        ignored = int(out.read_text().split()[1], 16)
        self.assertFalse(ignored & (1 << (signal.SIGPIPE - 1)), hex(ignored))
        self.assertFalse(ignored & (1 << (signal.SIGXFSZ - 1)), hex(ignored))

    def test_survivors_of_kill_are_named_after_the_whole_ceiling(self):
        # 60 s of grace after TERM, 60 s more for KILLed GPU processes (D state) to go.
        self.assertEqual((self.m.GRACE_SECONDS, self.m.VERIFY_SECONDS), (60, 60))
        proc, state, rec, child = self.family()
        clock, original = Clock(), self.m.send
        def send(r, sig):
            if sig != signal.SIGKILL:  # a KILL that does not take effect in time
                original(r, sig)
        with patch.object(self.m, 'time', clock), patch.object(self.m, 'send', send):
            with self.assertRaises(self.m.OwnershipError) as err:
                self.m.stop([(rec['pid'], str(state))])
        self.assertGreaterEqual(clock.now, 120)
        msg = str(err.exception)
        self.assertIn('60 s after KILL', msg)
        self.assertIn(f"pid {rec['pid']} (", msg)
        self.assertIn(f'pid {child} (', msg)

    def run_unregistered_step(self, name, bootstrap, bash_env=None):
        # The compiled step and helper unchanged; a python3 first on PATH runs the helper's
        # `start` with `bootstrap` (a patch), everything else as is.
        work = self.root / name
        scripts = work / 'scripts'
        shutil.copytree(helper.parent, scripts)
        marker = work / 'server-code-ran'
        (scripts / 'server_a.sh').write_text(f'touch {shlex.quote(str(marker))}\n')
        bin_dir = work / 'bin'
        bin_dir.mkdir()
        py = shlex.quote(sys.executable)
        (bin_dir / 'python3').write_text(
            '#!/bin/sh\n'
            f'if [ "$4" = start ]; then h=$3; shift 4; exec {py} -I -S -c "$FAKE_BOOTSTRAP" "$h" start "$@"; fi\n'
            f'exec {py} "$@"\n')
        (bin_dir / 'python3').chmod(0o755)
        env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', FAKE_BOOTSTRAP=bootstrap)
        if bash_env:
            env['BASH_ENV'] = str(bash_env)
        result = subprocess.run(['bash', str(scripts / 'step.sh')], cwd=work, env=env,
                                capture_output=True, text=True, timeout=30)
        left = []
        for entry in os.listdir('/proc'):
            try:
                if entry.isdigit() and str(work).encode() in Path(f'/proc/{entry}/cmdline').read_bytes():
                    left.append(entry)
            except OSError:
                pass
        return result, result.stdout + result.stderr, marker, work, left

    RUN_HELPER = 'sys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name="__main__")\n'
    NO_PIDFD = ('import os, runpy, sys\n'
                'def unavailable(*a):\n    raise OSError("pidfds unavailable")\n'
                'os.pidfd_open = unavailable\n') + RUN_HELPER
    # The identity appears only after step.sh has given up on it (5 s) and written .stop.
    SLOW_IDENTITY = ('import runpy, sys, tempfile, time\n'
                     'mkstemp = tempfile.mkstemp\n'
                     'def slow(*a, **k):\n    time.sleep(7)\n    return mkstemp(*a, **k)\n'
                     'tempfile.mkstemp = slow\n') + RUN_HELPER

    def test_generated_step_reaps_a_bootstrap_that_never_registered(self):
        for name, bootstrap in (('no-pidfd', self.NO_PIDFD), ('slow-identity', self.SLOW_IDENTITY)):
            with self.subTest(name):
                result, out, marker, work, left = self.run_unregistered_step(name, bootstrap)
                self.assertEqual(result.returncode, 1, out)
                self.assertIn('identity missing; startup refused (see logs/server_a.log)', out)
                self.assertIn('server a never started (its bootstrap exited 2 before exec)', out)
                self.assertNotIn('cleanup unverified', out)
                self.assertFalse(marker.exists(), out)
                self.assertEqual(left, [], out)
                self.assertFalse(list((work / 'logs').glob('.server_ownership.*')), out)

    def test_generated_step_fails_closed_when_a_bootstrap_cannot_be_reaped(self):
        bash_env = self.root / 'no-wait.sh'
        bash_env.write_text('wait() { echo "wait: pid $1 is not a child of this shell" >&2; return 127; }\n')
        result, out, marker, work, left = self.run_unregistered_step('no-reap', self.NO_PIDFD, bash_env)
        self.assertEqual(result.returncode, 1, out)
        self.assertIn('could not be reaped: wait: pid', out)
        self.assertIn('cleanup unverified', out)
        self.assertNotIn('never started', out)
        self.assertFalse(marker.exists(), out)
        self.assertTrue(list((work / 'logs').glob('.server_ownership.*')), out)

    def test_missing_pidfd_support_refuses_before_server_exec(self):
        state, marker = self.root / 'unsupported.json', self.root / 'unexpected-exec'
        script = self.root / 'unexpected.sh'
        script.write_text(f'touch {shlex.quote(str(marker))}\n')
        code = ('import importlib.util,os,sys\n'
                'spec=importlib.util.spec_from_file_location("owned",sys.argv[1])\n'
                'm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
                'def unavailable(*a): raise OSError("pidfds unavailable")\n'
                'm.os.pidfd_open=unavailable\n'
                'm.start(sys.argv[2],sys.argv[3],os.getppid())\n')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(helper), str(state), str(script)],
                                capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())
        self.assertFalse(state.exists())

    def test_dead_parent_before_bootstrap_is_refused(self):
        state, script = self.root / 'orphan.json', self.root / 'orphan.sh'
        script.write_text('exit 0\n')
        with patch.object(self.m.os, 'getppid', return_value=1):
            with patch.object(self.m.os, 'pidfd_open') as pin:
                with self.assertRaises(self.m.OwnershipError):
                    self.m.start(str(state), str(script), os.getpid())
                pin.assert_not_called()
        self.assertFalse(state.exists())

    def test_missing_registration_ack_is_bounded(self):
        state, marker = self.root / 'unacknowledged.json', self.root / 'unexpected-start'
        script = self.root / 'unacknowledged.sh'
        script.write_text(f'touch {shlex.quote(str(marker))}\n')
        code = ('import importlib.util,sys\n'
                'spec=importlib.util.spec_from_file_location("owned",sys.argv[1])\n'
                'm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
                'm.START_ACK_SECONDS=.05\n'
                'm.start(sys.argv[2],sys.argv[3],int(sys.argv[4]))\n')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(helper), str(state), str(script),
                                 str(os.getpid())], capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(state.exists())
        self.assertIn(b'acknowledgement timed out', result.stderr)
        self.assertFalse(marker.exists())

unittest.main(verbosity=2)
"#;

#[test]
fn generated_server_teardown_uses_stable_identities() {
    let root = std::env::temp_dir().join(format!("oaka-teardown-{}-{}", std::process::id(), fastrand::u32(..)));
    fs::create_dir_all(root.join("lib/profiles/m")).unwrap();
    fs::create_dir_all(root.join("work")).unwrap();
    fs::write(root.join("lib/profiles/m/base.toml"), "model = '/model/fake'\n").unwrap();
    fs::write(
        root.join("work/plan.toml"),
        "[[server]]\nname = 'a'\nprofile = 'm/base'\ngpus = [0]\nport = 29950\n\
         [[client]]\nkind = 'fixed-seq'\nserver = 'a'\nisl_osl = [[128, 32]]\nconc = [1]\n",
    )
    .unwrap();
    let compiled = Command::new(env!("CARGO_BIN_EXE_oaka"))
        .arg("compile")
        .current_dir(root.join("work"))
        .env("OAKA_LIB", root.join("lib"))
        .env("OAKA_GPUS", "gfx950")
        .env("OAKA_ROCM", "10.0.0")
        .env_remove("AKAO_ARTIFACT_ROOT")
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    fs::write(root.join("checks.py"), CHECKS).unwrap();
    let checks = Command::new("python3")
        .arg("-I")
        .arg(root.join("checks.py"))
        .arg(root.join("work/scripts/server_processes.py"))
        .output()
        .unwrap();
    let _ = fs::remove_dir_all(&root);
    assert!(
        checks.status.success(),
        "{}{}",
        String::from_utf8_lossy(&checks.stdout),
        String::from_utf8_lossy(&checks.stderr)
    );
}
