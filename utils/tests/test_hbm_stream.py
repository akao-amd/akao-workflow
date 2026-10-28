"""Report arithmetic and host-driver failures using saved amd-smi/RVS stand-ins."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hbm_summary", UTILS / "hbm_stream_summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class Reports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hbm-stream-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.put("env.txt", "host: fake\nruns=2 num_iter=200 device=11 12 21 22 parallel=true\n")
        self.put("rvs_gpus.txt", "\n".join(
            f"{bdf} - GPU[ {n + 2} - {gpu_id}] AMD Radeon Graphics (Device 30151)"
            for n, (bdf, gpu_id) in enumerate([
                ("0001:04:00.0", 11), ("0001:04:00.1", 12),
                ("0002:04:00.0", 21), ("0002:04:00.1", 22), ("0003:04:00.0", 31)])))
        self.put("gpu_list.txt", "\n".join(
            f"GPU: {i}\n    BDF: {bdf}\n    KFD_ID: {gpu_id}\n"
            for i, (bdf, gpu_id) in enumerate([
                ("0001:04:00.0", 11), ("0001:04:00.1", 12),
                ("0002:04:00.0", 21), ("0002:04:00.1", 22), ("0003:04:00.0", 31)])))
        self.put("partition.txt", "GPU: 0\n    ACCELERATOR_PARTITION: DPX\n    MEMORY_PARTITION: NPS2\n"
                 "GPU: 2\n    ACCELERATOR_PARTITION: DPX\n    MEMORY_PARTITION: NPS2\n")
        self.put("ifwi.txt", "GPU: 0\n    NAME: AMD MI450X_MC_GENERIC\n    PART_NUMBER: 113-M4500001-700H\n    VERSION: 00207991\n"
                 "GPU: 1\n    VERSION: boot-firmware-not-physical-ifwi\nGPU: 2\n    VERSION: 00207991\n")
        self.put("run_status.tsv", "run\trc\telapsed_s\n1\t0\t1.25\n2\t0\t2.75\n")
        self.run_log(1, 1)
        self.run_log(2, 3)
        # Idle samples must not enter loaded means. Duplicate partition power and a
        # foreign physical GPU's 9000 W must not inflate selected physical totals.
        self.put("telemetry.txt", self.sample(0, 100, 100) + self.sample(95, 1000, 2000)
                 + self.sample(95, 2000, 4000))

    def put(self, name, text):
        (self.root / name).write_text(text)

    def run_log(self, run, scale, result="PASS", missing=False):
        self.put(f"run{run}.log", "\n".join(
            f"{gpu_id} {op} {scale * (1 if op == 'Read' else 2) * 1000000}"
            for gpu_id in (11, 12, 21, 22) for op in ("Read", "Triad")
            if not (missing and gpu_id == 22 and op == "Triad"))
            + f"\n| babel-float | BABEL | \x1b[32m{result}\x1b[0m |\n")

    def sample(self, activity, power0, power2):
        return "T 2026-10-28T12:00:00+00:00\n" + "\n".join(
            f"GPU: {i}\n    SOCKET_POWER: {power} W\n    GFX_0:\n        CLK: 1500 MHz\n"
            "    VCLK_0:\n        CLK: 66 MHz\n"
            f"    GFX_ACTIVITY: {activity} %\n    HOTSPOT: 70 C\n    MEM: 60 C\n"
            for i, power in ((0, power0), (1, power0), (2, power2), (3, power2), (4, 9000)))

    def fields(self):
        return dict(line.split(": ", 1) for line in (self.root / "checkme.txt").read_text().splitlines())

    def test_selected_physical_power_and_bandwidth_means(self):
        self.assertEqual(summary.build(self.root), 0)
        fields = self.fields()
        self.assertEqual(len(fields), 15)
        self.assertEqual(fields["Node Read"], "8.000 TB/s")
        self.assertEqual(fields["Node Triad"], "16.000 TB/s")
        self.assertIn("0001:04:00 4.000 TB/s", fields["Per physical GPU, Read"])
        self.assertIn("0002:04:00.1 2.000 TB/s", fields["Per partition, Read"])
        self.assertEqual(fields["Node GPU power, average"], "4500 W")
        self.assertIn("avg 1500 W, peak 2000 W", fields["Per GPU power under load"])
        self.assertIn("0001:04:00 1500 MHz", fields["Graphics clock under load"])
        self.assertNotIn("0003:04:00", (self.root / "checkme.txt").read_text())
        self.assertNotIn("boot-firmware-not-physical-ifwi", fields["IFWI"])
        self.assertIn("1002:75c7", fields["GPU type (PCI device ID)"])
        self.assertIn("DPX/NPS2", fields["Mode"])
        self.assertEqual(fields["Time per run"], "run1 1.25 s; run2 2.75 s")

    def test_failed_or_incomplete_runs_are_excluded_and_fail(self):
        for failure in ("nonzero", "FAIL", "missing", "no-footer", "no-status", "duplicate"):
            with self.subTest(failure=failure):
                self.run_log(2, 3, "FAIL" if failure == "FAIL" else "PASS", failure == "missing")
                rc = 7 if failure == "nonzero" else 0
                self.put("run_status.tsv", f"run\trc\telapsed_s\n1\t0\t1.25\n2\t{rc}\t2.75\n"
                         if failure != "no-status" else "run\trc\telapsed_s\n1\t0\t1.25\n")
                if failure == "no-footer":
                    self.put("run2.log", (self.root / "run2.log").read_text().split("| babel")[0])
                if failure == "duplicate":
                    with (self.root / "run2.log").open("a") as f:
                        f.write("11 Read 3000000\n")
                self.assertEqual(summary.build(self.root), 1)
                self.assertEqual(self.fields()["Node Read"], "4.000 TB/s (1/2 complete PASS runs)")
                self.assertEqual(self.fields()["Result"], "FAIL (1/2 complete PASS runs)")

    def test_missing_and_stale_telemetry_are_not_zero_or_loaded(self):
        self.put("telemetry.txt", self.sample(95, 1000, 2000)
                 + "T final\nGPU: 0\n    SOCKET_POWER: 10000 W\n    HOTSPOT: N/A\n    MEM: N/A\n")
        self.assertEqual(summary.build(self.root), 0)
        self.assertEqual(self.fields()["Node GPU power, average"], "3000 W")
        self.assertIn("avg 1000 W", self.fields()["Per GPU power under load"])
        self.put("telemetry.txt", "T only\nGPU: 0\n    GFX_ACTIVITY: 99 %\n    SOCKET_POWER: N/A\n")
        summary.build(self.root)
        self.assertEqual(self.fields()["Node GPU power, average"], "N/A")
        self.assertIn("0001:04:00 N/A / N/A", self.fields()["Max hotspot / HBM temperature"])

    def test_second_partition_uses_physical_sensors(self):
        self.put("env.txt", "host: fake\nruns=1 num_iter=200 device=12 parallel=false\n")
        self.put("run1.log", "12 Read 1000000\n12 Triad 2000000\n| BABEL | PASS |\n")
        summary.build(self.root)
        self.assertEqual(self.fields()["Node Read"], "1.000 TB/s")
        self.assertEqual(self.fields()["Node GPU power, average"], "1500 W")
        self.assertIn("00207991", self.fields()["IFWI"])

    def test_gpu_index_selection_and_invalid_ids(self):
        result = subprocess.run(["python3", "-I", str(UTILS / "hbm_stream_summary.py"),
                                 "--devices", "0,1,2,3", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "11 12 21 22")
        for indices in ("0,0", "8", "-1", "0,1;all"):
            result = subprocess.run(["python3", "-I", str(UTILS / "hbm_stream_summary.py"),
                                     "--devices", indices, str(self.root)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_config_must_select_requested_devices_before_benchmark(self):
        for config, ok in (("  device: 11 12\n", True), ("  device: all\n", False),
                           ("device: 11\ndevice: 12\n", False), ("# device: 11 12\n", False)):
            self.put("babel.conf", config)
            command = lambda: summary.main(["--validate-devices", "11 12", str(self.root)])
            if ok:
                self.assertEqual(command(), 0)
            else:
                with self.assertRaises(ValueError):
                    command()

    def test_driver_aborts_when_output_directory_or_gpu_probe_fails(self):
        smi = self.root / "amd-smi"
        smi.write_text("#!/bin/bash\n[ \"${FAIL_PROBE:-0}\" = 0 ] || exit 9\necho 'No running processes detected'\n")
        smi.chmod(0o755)
        rvs = self.root / "rvs"
        rvs.write_text(f"#!/bin/bash\ntouch '{self.root}/rvs-called'\n")
        rvs.chmod(0o755)
        config = self.root / "source.conf"
        config.write_text("  device: all\n  parallel: true\n  num_iter: 20000\n")
        blocker = self.root / "not-a-directory"
        blocker.touch()
        env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}", RVS_BIN=str(rvs),
                   BABEL_CONFIG=str(config), HBM_OUTPUT_ROOT=str(blocker), GPU="0,1,2,3")
        for fail in ("0", "1"):
            result = subprocess.run(["bash", str(UTILS / "hbm_stream_bench.sh"), "1", "200"],
                                     env=dict(env, FAIL_PROBE=fail), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((self.root / "rvs-called").exists())


if __name__ == "__main__":
    unittest.main()
