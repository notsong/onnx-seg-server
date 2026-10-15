"""Run the installed JLD ONNX through the real C++ HTTP service in both modes."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-deps"))
sys.path.insert(0, str(ROOT / "tests"))

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from email.parser import Parser

import numpy as np
from PIL import Image
from self_test import Server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    source_image = args.image.resolve()
    model = ROOT / "models" / "1001_004_JLD.onnx"
    config_file = ROOT / "config" / "1001_004.json"
    executable = ROOT / "build" / "Release" / "onnx-seg-server.exe"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    now = datetime.now(timezone(timedelta(hours=8)))
    run = ROOT / "tests" / "generated" / (
        "jld-" + now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    image_size = Image.open(source_image).size
    result = {"tested_at": now.isoformat(), "model": str(model),
              "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "image": str(source_image), "image_size": list(image_size),
              "image_sha256": hashlib.sha256(source_image.read_bytes()).hexdigest(),
              "threads": args.threads, "device": "CPU", "config": config, "modes": {}}
    palette = np.array(config["model_detail"]["colors"], np.uint8)
    masks = {}
    for mode in ("whole", "tiled"):
        print("Starting CPU " + mode + " model load and warm-up...", flush=True)
        if mode == "whole":
            server_root = run / "whole"
            (server_root / "config").mkdir(parents=True)
            (server_root / "models").mkdir()
            local_model = server_root / "models" / model.name
            try:
                os.link(model, local_model)
            except OSError:
                shutil.copy2(model, local_model)
            whole = json.loads(json.dumps(config))
            whole["model_detail"]["cut_flag"] = 0
            (server_root / "config" / config_file.name).write_text(
                json.dumps(whole, indent=2), encoding="utf-8")
        else:
            server_root = run / "tiled"
            (server_root / "config").mkdir(parents=True)
            (server_root / "models").mkdir()
            try:
                os.link(model, server_root / "models" / model.name)
            except OSError:
                shutil.copy2(model, server_root / "models" / model.name)
            tiled = json.loads(json.dumps(config))
            tiled["model_detail"]["cut_flag"] = 1
            (server_root / "config" / config_file.name).write_text(
                json.dumps(tiled, indent=2), encoding="utf-8")
        started = time.perf_counter()
        server = Server(executable, server_root, threads=args.threads,
                        startup_timeout=180, request_timeout=600)
        startup = time.perf_counter() - started
        mask_path = run / ("JLD_mask_" + mode + ".png")
        header_path = run / ("headers_" + mode + ".txt")
        try:
            print(f"{mode} ready after {startup:.2f}s; POST original JPEG {image_size}...", flush=True)
            started = time.perf_counter()
            process = subprocess.run(
                ["curl.exe", "--silent", "--show-error", "--fail-with-body",
                 "--max-time", "600", "-F", "file=@" + str(source_image),
                 "-F", "catagory=1001", "-F", "item=4", server.url + "/infer",
                 "--output", str(mask_path), "--dump-header", str(header_path)],
                capture_output=True, timeout=620,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            client_ms = (time.perf_counter() - started) * 1000
            if process.returncode:
                raise RuntimeError(process.stderr.decode(errors="replace") + "\n" +
                                   mask_path.read_text(errors="replace")[:1000])
            header_text = header_path.read_text(encoding="utf-8")
            blocks = [block for block in re.split(r"\r?\n\r?\n", header_text)
                      if block.startswith("HTTP/")]
            final_header = blocks[-1]
            assert final_header.startswith("HTTP/1.1 200"), header_text
            headers = Parser().parsestr(final_header.split("\n", 1)[1])
            assert headers["Content-Type"] == "image/png"
            with Image.open(mask_path) as image:
                assert image.size == image_size
                mask = np.array(image.convert("RGB"))
            labels = np.zeros(mask.shape[:2], np.uint8)
            covered = np.zeros(labels.shape, bool)
            percentages = {}
            for index, color in enumerate(palette):
                hit = np.all(mask == color, axis=2)
                covered |= hit
                labels[hit] = index
                percentages[str(index)] = round(float(hit.mean() * 100), 6)
            assert covered.all(), "Output contains colors outside config palette"
            masks[mode] = labels
            log_text = server.log_path.read_text(encoding="utf-8", errors="replace")
            (run / ("server_" + mode + ".log")).write_text(log_text, encoding="utf-8")
            stages = {}
            for name in ("decode", "preprocess", "inference", "postprocess", "encode", "total"):
                matches = re.findall(name + r"_ms=([0-9.]+)", log_text)
                if matches:
                    stages[name + "_ms"] = float(matches[-1])
            result["modes"][mode] = {
                "status": "passed", "http_status": 200, "mask": str(mask_path),
                "width": image_size[0], "height": image_size[1],
                "windows": int(headers["X-Tile-Count"]),
                "startup_and_warmup_seconds": round(startup, 3),
                "client_elapsed_ms": round(client_ms, 3),
                "inference_ms": float(headers["X-Inference-Ms"]),
                "server_total_ms": float(headers["X-Total-Ms"]),
                "class_pixel_percent": percentages, "stage_timings": stages}
            print(json.dumps(result["modes"][mode], indent=2), flush=True)
        finally:
            server.close()
    result["whole_vs_tiled_different_pixel_percent"] = round(
        float((masks["whole"] != masks["tiled"]).mean() * 100), 6)
    result["status"] = "passed"
    result["notes"] = [
        "Original JPEG uploaded without client resizing or crop.",
        "Black is class 0; white is class 1; no morphology/contour filtering.",
        "Single measured request per mode after model warm-up; not a stable benchmark.",
        "No ground truth or original DLL comparison; segmentation accuracy is not quantified."]
    report = run / "report.json"
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("REPORT: " + str(report), flush=True)


if __name__ == "__main__":
    main()
