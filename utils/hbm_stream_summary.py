#!/usr/bin/env python3
"""Build summary.txt and checkme.txt from hbm_stream_bench.sh's saved artifacts."""

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read(root, name):
    path = root / name
    return path.read_text(errors="replace") if path.exists() else ""


def gpu_blocks(text):
    """Parse amd-smi text, preserving missing values and the GFX_0 clock."""
    parts = re.split(r"(?m)^GPU:\s*(\d+)\s*$", text)
    blocks = {}
    for index in range(1, len(parts), 2):
        body = parts[index + 1]
        block = dict(re.findall(r"(?m)^\s+([A-Z_0-9]+):[ \t]*([^\n]*)", body))
        clock = re.search(r"GFX_0:[ \t]*\n\s+CLK:[ \t]*([^\n]+)", body)
        if clock:
            block["GFX_0_CLK"] = clock[1]
        blocks[int(parts[index])] = block
    return blocks


def topology(root):
    smi = gpu_blocks(read(root, "gpu_list.txt"))
    by_bdf = {v["BDF"].lower(): k for k, v in smi.items() if "BDF" in v}
    devices = {}
    pattern = r"(?m)^([0-9a-fA-F:.]+) - GPU\[\s*\d+ -\s*(\d+)\](.*)"
    for bdf, gpu_id, description in re.findall(pattern, read(root, "rvs_gpus.txt")):
        bdf = bdf.lower()
        pci = re.search(r"\(Device (\d+)\)", description)
        devices[gpu_id] = {
            "bdf": bdf, "physical": bdf.rsplit(".", 1)[0],
            "smi": by_bdf.get(bdf), "pci": f"1002:{int(pci[1]):04x}" if pci else "N/A",
        }
    if not devices:
        raise ValueError("no RVS GPU mapping in rvs_gpus.txt")
    return devices, smi


def select(devices, value):
    ids = list(devices) if value == "all" else value.split()
    if not ids or len(ids) != len(set(ids)) or any(i not in devices for i in ids):
        raise ValueError("unknown, empty, or duplicate RVS device selection")
    return {i: devices[i] for i in ids}


def number(value):
    match = re.match(r"^(-?\d+(?:\.\d+)?)(?:\s|$)", value or "")
    return float(match[1]) if match else None


def average(values, unit, digits=0):
    return f"{statistics.mean(values):.{digits}f} {unit}" if values else "N/A"


def build(root):
    env = read(root, "env.txt")
    settings = re.search(r"runs=(\d+) num_iter=(\d+) device=(.*?) parallel=(true|false)", env)
    if not settings:
        raise ValueError("missing run settings in env.txt")
    runs = int(settings[1])
    devices, smi = topology(root)
    selected = select(devices, settings[3])
    physical = sorted({d["physical"] for d in selected.values()})
    # Socket sensors belong to a physical GPU, including when only its .1 is selected.
    owners = {}
    for bdf in physical:
        candidates = sorted((v["BDF"].lower(), i) for i, v in smi.items()
                            if v.get("BDF", "").lower().rsplit(".", 1)[0] == bdf)
        owners[bdf] = candidates[0][1] if candidates else None
    partition = gpu_blocks(read(root, "partition.txt"))
    ifwi = gpu_blocks(read(root, "ifwi.txt"))
    status = {int(row["run"]): row for row in csv.DictReader(
        read(root, "run_status.tsv").splitlines(), delimiter="\t")}
    expected = {(i, op) for i in selected for op in ("Read", "Triad")}
    values = defaultdict(list)
    totals = defaultdict(list)
    run_lines = []
    run_times = []
    successful = 0
    for run in range(1, runs + 1):
        log = re.sub(r"\x1b\[[0-9;]*m", "", read(root, f"run{run}.log"))
        results = re.findall(r"\|\s*BABEL\s*\|\s*(PASS|FAIL)\b", log, re.I)
        rows = defaultdict(list)
        for gpu_id, op, rate in re.findall(r"(?m)^\s*(\d+)\s+(Read|Triad)\s+(\d+(?:\.\d+)?)", log):
            rows[gpu_id, op].append(float(rate) / 1e6)
        record = status.get(run, {})
        rc = record.get("rc", "N/A")
        complete = set(rows) == expected and all(len(v) == 1 and v[0] > 0 for v in rows.values())
        passed = rc == "0" and bool(results) and all(r.upper() == "PASS" for r in results) and complete
        outcome = "PASS" if passed else "FAIL (exit status, BABEL result, or bandwidth data missing/failed)"
        run_lines.append(f"run{run}: {outcome}; rc={rc}")
        elapsed = number(record.get("elapsed_s"))
        run_times.append(f"run{run} {elapsed:.2f} s" if elapsed is not None else f"run{run} N/A")
        if passed:
            successful += 1
            for (gpu_id, op), rates in rows.items():
                values[selected[gpu_id]["bdf"], op].append(rates[0])
            for op in ("Read", "Triad"):
                total = sum(v[0] for (i, operation), v in rows.items() if operation == op)
                totals[op].append(total)
                run_lines.append(f"  node {op}: {total:.3f} TB/s")

    # Complete telemetry snapshots avoid depending on amd-smi's field order, carrying
    # stale activity across samples, or dropping the final sample at EOF.
    loaded_power = defaultdict(list)
    loaded_clock = defaultdict(list)
    hotspot = defaultdict(list)
    hbm = defaultdict(list)
    node_power = []
    for sample in re.split(r"(?m)^T [^\n]*\n", read(root, "telemetry.txt"))[1:]:
        blocks = gpu_blocks(sample)
        powers = []
        all_loaded = True
        for bdf in physical:
            block = blocks.get(owners[bdf], {})
            for key, target in (("HOTSPOT", hotspot), ("MEM", hbm)):
                value = number(block.get(key))
                if value is not None:
                    target[bdf].append(value)
            activity = number(block.get("GFX_ACTIVITY"))
            power = number(block.get("SOCKET_POWER"))
            loaded = activity is not None and activity >= 90
            if loaded:
                if power is not None and power > 0:
                    loaded_power[bdf].append(power)
                clock = number(block.get("GFX_0_CLK"))
                if clock is not None:
                    loaded_clock[bdf].append(clock)
            if loaded and power is not None and power > 0:
                powers.append(power)
            else:
                all_loaded = False
        if all_loaded and powers:
            node_power.append(sum(powers))

    def per_partition(op):
        return "; ".join(f"{d['bdf']} {average(values[d['bdf'], op], 'TB/s', 3)}"
                         for d in sorted(selected.values(), key=lambda d: d["bdf"]))

    def per_physical(op):
        return "; ".join(f"{bdf} {sum(statistics.mean(values[d['bdf'], op]) for d in selected.values() if d['physical'] == bdf):.3f} TB/s"
                         if successful else f"{bdf} N/A" for bdf in physical)

    def modes():
        return "; ".join(f"{bdf} " + "/".join(partition.get(owners[bdf], {}).get(k, "N/A")
                         for k in ("ACCELERATOR_PARTITION", "MEMORY_PARTITION")) for bdf in physical)

    def firmware():
        entries = []
        for bdf in physical:
            block = ifwi.get(owners[bdf], {})
            entries.append(f"{bdf} " + ", ".join(f"{k}={block.get(k, 'N/A')}"
                           for k in ("NAME", "PART_NUMBER", "VERSION")))
        return "; ".join(entries)

    def power_text(bdf):
        readings = loaded_power[bdf]
        return (f"avg {statistics.mean(readings):.0f} W, peak {max(readings):.0f} W"
                if readings else "N/A")

    def max_temp(readings):
        return f"{max(readings):.0f} C" if readings else "N/A"

    cohort = f" ({successful}/{runs} complete PASS runs)" if successful != runs else ""
    result = "PASS" if successful == runs else "FAIL"
    fields = [
        ("Mode", modes()),
        ("GPU type (PCI device ID)", "; ".join(f"{bdf} " + ", ".join(sorted({d['pci'] for d in selected.values() if d['physical'] == bdf})) for bdf in physical)),
        ("IFWI", firmware()),
        ("Node Read", average(totals["Read"], "TB/s", 3) + cohort),
        ("Node Triad", average(totals["Triad"], "TB/s", 3) + cohort),
        ("Per physical GPU, Read", per_physical("Read") + cohort),
        ("Per physical GPU, Triad", per_physical("Triad") + cohort),
        ("Per partition, Read", per_partition("Read") + cohort),
        ("Per partition, Triad", per_partition("Triad") + cohort),
        ("Per GPU power under load", "; ".join(f"{bdf} {power_text(bdf)}" for bdf in physical)),
        ("Node GPU power, average", average(node_power, "W")),
        ("Graphics clock under load", "; ".join(f"{bdf} {average(loaded_clock[bdf], 'MHz')}" for bdf in physical)),
        ("Max hotspot / HBM temperature", "; ".join(f"{bdf} {max_temp(hotspot[bdf])} / {max_temp(hbm[bdf])}" for bdf in physical)),
        ("Time per run", "; ".join(run_times)),
        ("Result", f"{result} ({successful}/{runs} complete PASS runs)"),
    ]
    checkme = "\n".join(f"{label}: {value}" for label, value in fields) + "\n"
    summary = [env.splitlines()[0], f"runs={runs} num_iter={settings[2]} device={settings[3]} parallel={settings[4]}",
               "Bandwidth: decimal TB/s, mean over complete PASS runs; physical totals sum selected partitions.",
               "Loaded: GFX_ACTIVITY >= 90%; power counted once per selected physical GPU.",
               "Node power uses samples with all selected physical GPUs loaded and valid power.",
               "Clock: mean GFX_0 CLK over loaded samples; temperatures: maximum over the capture.",
               "--- RVS result per run", *run_lines, "--- review fields", checkme.rstrip(),
               "--- per partition spread (max-min)/mean over complete PASS runs"]
    for (bdf, op), rates in sorted(values.items()):
        if rates:
            spread = (max(rates) - min(rates)) / statistics.mean(rates) * 100
            summary.append(f"{bdf} {op}: {spread:.2f}% ({len(rates)} runs)")
    summary.append("--- loaded telemetry samples per physical GPU")
    summary.extend(f"{bdf} (amd-smi GPU{owners[bdf]}): power {len(loaded_power[bdf])}, clock {len(loaded_clock[bdf])}" for bdf in physical)
    summary.append(f"Node power: {len(node_power)} samples; peak {max(node_power):.0f} W" if node_power else "Node power: N/A (no complete loaded samples)")
    messages = read(root, "dmesg_new.txt").splitlines()
    kernel_status = f"{len(messages)} (see dmesg_new.txt)" if (root / "dmesg_new.txt").exists() else "N/A"
    if "Operation not permitted" in read(root, "dmesg_live.txt") or "Permission denied" in read(root, "dmesg_live.txt"):
        kernel_status = "N/A (dmesg capture denied; see dmesg_live.txt)"
    summary.append(f"--- new kernel messages during the bench: {kernel_status}")
    (root / "checkme.txt").write_text(checkme)
    (root / "summary.txt").write_text("\n".join(summary) + "\n")
    return 0 if result == "PASS" else 1


def main(args):
    if len(args) == 3 and args[0] in ("--devices", "--validate-devices"):
        devices, _ = topology(Path(args[2]))
        if args[0] == "--devices":
            if not re.fullmatch(r"\d+(?:,\d+)*", args[1]):
                raise ValueError("GPU must be comma-separated amd-smi indices")
            indices = [int(i) for i in args[1].split(",")]
            ids = {d["smi"]: i for i, d in devices.items() if d["smi"] is not None}
            if len(indices) != len(set(indices)) or any(i not in ids for i in indices):
                raise ValueError("unknown or duplicate amd-smi GPU indices")
            print(" ".join(ids[i] for i in indices))
        else:
            requested = select(devices, args[1])
            config = read(Path(args[2]), "babel.conf")
            configured = re.findall(r"(?m)^\s*device:[ \t]*([^\n#]+)", config)
            if len(configured) != 1 or set(select(devices, configured[0].strip())) != set(requested):
                raise ValueError("babel.conf does not select the requested RVS devices")
        return 0
    if len(args) != 1:
        raise ValueError("usage: hbm_stream_summary.py <result-dir>")
    return build(Path(args[0]))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, KeyError, OSError) as error:
        print(f"HBM summary: {error}", file=sys.stderr)
        sys.exit(1)
