"""Test the SX model using the actual C++ HTTP server and an original JPEG."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import argparse
from datetime import datetime, timezone, timedelta
from email.parser import Parser
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
from PIL import Image
from self_test import Server


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    source = args.image.resolve()
    model = ROOT / "models" / "1003_010_SX.onnx"
    config_path = ROOT / "config" / "1003_010.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["catagory"] == "1003" and config["item"] == 10
    assert config["model_detail"]["cut_flag"] == 0
    now = datetime.now(timezone(timedelta(hours=8)))
    run = ROOT / "tests" / "generated" / ("sx-" + now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    (run / "models").mkdir(parents=True)
    (run / "config").mkdir()
    shutil.copy2(config_path, run / "config" / config_path.name)
    try:
        os.link(model, run / "models" / model.name)
    except OSError:
        shutil.copy2(model, run / "models" / model.name)
    (ROOT / "tests" / "inputs").mkdir(exist_ok=True)
    shutil.copy2(source, run / "SX_input.jpg")
    with Image.open(source) as image:
        image_size = image.size
    result = {"tested_at": now.isoformat(), "device": "CPU", "threads": args.threads,
              "model": str(model), "model_sha256": sha256(model),
              "image": str(source), "image_size": list(image_size),
              "image_sha256": sha256(source), "config": config,
              "model_outputs": {"seg": [1, 2, 1024, 1024], "boundary": [1, 1, 1024, 1024]},
              "selected_output": "seg", "mode": "whole",
              "image_backup": str(run / "SX_input.jpg"),
              "preprocessing": {"resize_mode": config["model_detail"].get("resize_mode", "stretch"),
                                "mean": config["model_detail"].get("mean", [0, 0, 0]),
                                "std": config["model_detail"].get("std", [1, 1, 1])}}
    executable = ROOT / "build" / "Release" / "onnx-seg-server.exe"
    print("Starting SX CPU model load and warm-up...", flush=True)
    started = time.perf_counter()
    server = Server(executable, run, threads=args.threads,
                    startup_timeout=600, request_timeout=600)
    result["startup_and_warmup_seconds"] = round(time.perf_counter() - started, 3)
    mask_path = run / "SX_mask_whole.png"
    header_path = run / "headers.txt"
    try:
        print(f"SX ready after {result['startup_and_warmup_seconds']:.2f}s; uploading original {image_size} JPEG...", flush=True)
        started = time.perf_counter()
        process = subprocess.run(
            ["curl.exe", "--silent", "--show-error", "--fail-with-body", "--max-time", "600",
             "-F", "file=@" + str(source), "-F", "catagory=1003", "-F", "item=10",
             server.url + "/infer", "--output", str(mask_path), "--dump-header", str(header_path)],
            capture_output=True, timeout=620,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result["client_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if process.returncode:
            raise RuntimeError(process.stderr.decode(errors="replace") + "\n" +
                               mask_path.read_text(errors="replace")[:1000])
        blocks = [b for b in re.split(r"\r?\n\r?\n", header_path.read_text(encoding="utf-8"))
                  if b.startswith("HTTP/")]
        final = blocks[-1]
        assert final.startswith("HTTP/1.1 200"), final
        headers = Parser().parsestr(final.split("\n", 1)[1])
        assert headers["Content-Type"] == "image/png"
        assert headers["X-Model-Key"] == "1003_010"
        assert int(headers["X-Tile-Count"]) == 1
        with Image.open(mask_path) as image:
            assert image.size == image_size
            mask = np.array(image.convert("RGB"))
        colors = config["model_detail"]["colors"]
        palette = np.array(colors if isinstance(colors, list) else
                           [colors[str(i)] for i in range(len(colors))], np.uint8)
        covered = np.zeros(mask.shape[:2], bool)
        proportions = {}
        for index, color in enumerate(palette):
            hit = np.all(mask == color, axis=2)
            covered |= hit
            proportions[str(index)] = {"rgb": color.tolist(),
                                      "pixels": int(hit.sum()),
                                      "percent": round(float(hit.mean() * 100), 6)}
        assert covered.all(), "Mask contains colors outside the configured palette"
        log_text = server.log_path.read_text(encoding="utf-8", errors="replace")
        stages = {}
        for name in ("decode", "preprocess", "inference", "postprocess", "encode", "total"):
            matches = re.findall(name + r"_ms=([0-9.]+)", log_text)
            if matches:
                stages[name + "_ms"] = float(matches[-1])
        result.update(status="passed", http_status=200, mask=str(mask_path), windows=1,
                      width=image_size[0], height=image_size[1],
                      inference_ms=float(headers["X-Inference-Ms"]),
                      server_total_ms=float(headers["X-Total-Ms"]),
                      class_pixels=proportions, stage_timings=stages,
                      executable=str(executable))
    finally:
        server.close()
    result["notes"] = [
        "Original JPEG uploaded without client resize or crop; whole image processed according to config resize_mode.",
        "RGB/255 float32 NCHW with configured mean/std; remove letterbox pads if enabled, restore scores then argmax.",
        "Segmentation output seg is used; auxiliary boundary is not used in mask generation.",
        "Black is class 0; white is class 1. SX mean/std and gray letterbox were supplied in training transforms.",
        "One measured request after warm-up; no ground truth or original Python pipeline accuracy comparison."]
    report = run / "report.json"
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.publish:
        args.publish.mkdir(parents=True, exist_ok=True)
        for src, name in [(mask_path, "SX_mask_whole.png"), (report, "SX_test_report.json"),
                          (config_path, "1003_010.json")]:
            shutil.copy2(src, args.publish / name)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print("REPORT: " + str(report), flush=True)


if __name__ == "__main__":
    main()
