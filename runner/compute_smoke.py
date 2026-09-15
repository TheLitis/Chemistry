from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import math
import os
import time
from pathlib import Path

GIB = 1024 ** 3


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_memory_bytes() -> int:
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def choose_ram_bytes(available: int, max_gib: float = 8.0) -> int:
    if available <= 0 or max_gib <= 0:
        return 0
    return int(min(max_gib * GIB, available * 0.25))


def choose_matrix_size(free_vram_bytes: int) -> int:
    if free_vram_bytes <= 0:
        return 2048
    # Three FP16 matrices plus allocator/workspace headroom; hard cap keeps this a smoke test.
    estimated = int(math.sqrt(max(1, free_vram_bytes * 0.20 / 6.0)))
    return max(2048, min(8192, estimated))


def cpu_worker(iterations: int) -> float:
    acc = 0.0
    for i in range(iterations):
        acc += math.sin(i * 0.0001) * math.cos(i * 0.00013)
    return acc


def run_cpu_smoke() -> dict:
    workers = max(1, os.cpu_count() or 1)
    iterations = 250_000
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(cpu_worker, [iterations] * workers))
    return {
        "workers": workers,
        "iterations_per_worker": iterations,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "checksum": round(sum(values), 6),
    }


def run_ram_smoke(max_gib: float) -> dict:
    available = available_memory_bytes()
    target = choose_ram_bytes(available, max_gib)
    started = time.perf_counter()
    block = bytearray(target)
    page = 4096
    for offset in range(0, target, page):
        block[offset] = (offset // page) & 0xFF
    checksum = sum(block[:: max(page, target // 1024 or 1)]) if target else 0
    elapsed = time.perf_counter() - started
    del block
    return {
        "available_before_gib": round(available / GIB, 3),
        "allocated_gib": round(target / GIB, 3),
        "elapsed_seconds": round(elapsed, 4),
        "checksum": int(checksum),
    }


def run_gpu_smoke(require_cuda: bool) -> dict:
    try:
        import torch
    except Exception as exc:
        result = {"torch_available": False, "cuda_available": False, "error": repr(exc)}
        if require_cuda:
            result["required_capability_failed"] = True
        return result

    result = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        if require_cuda:
            result["required_capability_failed"] = True
        return result

    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    size = choose_matrix_size(int(free_bytes))
    torch.cuda.reset_peak_memory_stats(device)
    a = torch.randn((size, size), device=device, dtype=torch.float16)
    b = torch.randn((size, size), device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    out = None
    for _ in range(8):
        out = a @ b
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result.update(
        {
            "device": torch.cuda.get_device_name(0),
            "matrix_size": size,
            "iterations": 8,
            "elapsed_seconds": round(elapsed, 4),
            "free_vram_before_mib": round(free_bytes / 1024 ** 2, 1),
            "total_vram_mib": round(total_bytes / 1024 ** 2, 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 1),
            "checksum": float(out[0, 0].item()) if out is not None else None,
        }
    )
    del a, b, out
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded ChemistryPC CPU/RAM/GPU smoke test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-ram-gib", type=float, default=8.0)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": run_cpu_smoke(),
        "ram": run_ram_smoke(args.max_ram_gib),
        "gpu": run_gpu_smoke(args.require_cuda),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["gpu"].get("required_capability_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
