"""Compare an ImageNet mean/std trial with the previous SX RGB/255 mask."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-deps"))
sys.path.insert(0, str(ROOT / "tests"))

import argparse
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
import re
import shutil
import subprocess
import time
import uuid
from email.parser import Parser

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator
from PIL import Image
from self_test import Server

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_copy(source, destination):
    model = onnx.load(str(source), load_external_data=False)
    assert not any(t.external_data for t in model.graph.initializer)
    assert len(model.graph.input) == 1 and model.graph.input[0].name == "input"
    names = {n for node in model.graph.node for n in list(node.input) + list(node.output)}
    internal = ["__sx_trial_mean", "__sx_trial_std", "__sx_trial_centered", "__sx_trial_normalized"]
    assert not set(internal).intersection(names)
    nodes = [
        helper.make_node("Sub", ["input", internal[0]], [internal[2]]),
        helper.make_node("Div", [internal[2], internal[1]], [internal[3]])]
    initializers = [numpy_helper.from_array(MEAN, internal[0]), numpy_helper.from_array(STD, internal[1])]
    # Verify RGB channel broadcasting and negative values, without clipping to byte pixels.
    graph = helper.make_graph(
        nodes, "normalization_check",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 1])],
        [helper.make_tensor_value_info(internal[3], TensorProto.FLOAT, [1, 3, 1, 1])],
        initializers)
    probe_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    probe_model.ir_version = 10
    probe = np.array([0.0, 0.5, 1.0], np.float32).reshape(1, 3, 1, 1)
    actual = ReferenceEvaluator(probe_model).run(None, {"input": probe})[0]
    np.testing.assert_allclose(actual, (probe - MEAN) / STD, rtol=1e-6)
    assert actual.min() < 0 and actual.max() > 1
    existing = list(model.graph.node)
    for node in existing:
        for i, name in enumerate(node.input):
            if name == "input":
                node.input[i] = internal[3]
    del model.graph.node[:]
    model.graph.node.extend(nodes + existing)
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(destination))
    del model, existing
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    source_image = Path(baseline.get("image_backup", baseline["image"]))
    source_model = ROOT / "models" / "1003_010_SX.onnx"
    assert digest(source_model) == baseline["model_sha256"]
    assert digest(source_image) == baseline["image_sha256"]
    now = datetime.now(timezone(timedelta(hours=8)))
    run = ROOT / "tests" / "generated" / (
        "sx-imagenet-" + now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    (run / "models").mkdir(parents=True)
    (run / "config").mkdir()
    trial_model = run / "models" / "1003_010_SX_imagenet.onnx"
    print("Creating test-only ONNX copy with RGB mean/std normalization...", flush=True)
    normalized_copy(source_model, trial_model)
    cfg = json.loads((ROOT / "config" / "1003_010.json").read_text(encoding="utf-8"))
    cfg["model_detail"]["model_name"] = trial_model.name
    # Reproduce the original normalization-only trial without double normalization.
    cfg["model_detail"].pop("mean", None)
    cfg["model_detail"].pop("std", None)
    cfg["model_detail"]["resize_mode"] = "stretch"
    assert cfg["model_detail"]["cut_flag"] == 0
    (run / "config" / "1003_010.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    shutil.copy2(source_image, run / "SX_input.jpg")
    mask_path = run / "SX_mask_imagenet.png"
    header_path = run / "headers.txt"
    print("Loading normalized SX model into the C++ CPU HTTP service...", flush=True)
    started = time.perf_counter()
    server = Server(ROOT / "onnx-seg-server.exe", run, threads=4,
                    startup_timeout=600, request_timeout=600)
    startup_seconds = time.perf_counter() - started
    try:
        print(f"Ready after {startup_seconds:.2f}s; inferring the same original JPEG...", flush=True)
        process = subprocess.run(
            ["curl.exe", "--silent", "--show-error", "--fail-with-body", "--max-time", "600",
             "-F", "file=@" + str(source_image), "-F", "catagory=1003", "-F", "item=10",
             server.url + "/infer", "--output", str(mask_path), "--dump-header", str(header_path)],
            capture_output=True, timeout=620, creationflags=subprocess.CREATE_NO_WINDOW)
        if process.returncode:
            raise RuntimeError(process.stderr.decode(errors="replace") + "\n" +
                               mask_path.read_text(errors="replace")[:1000])
        blocks = [b for b in re.split(r"\r?\n\r?\n", header_path.read_text(encoding="utf-8"))
                  if b.startswith("HTTP/")]
        final = blocks[-1]
        assert final.startswith("HTTP/1.1 200"), final
        headers = Parser().parsestr(final.split("\n", 1)[1])
        assert headers["Content-Type"] == "image/png" and headers["X-Model-Key"] == "1003_010"
        assert int(headers["X-Tile-Count"]) == 1
        with Image.open(mask_path) as im:
            assert list(im.size) == baseline["image_size"]
            mask = np.array(im.convert("RGB"))
        with Image.open(baseline["mask"]) as im:
            old_mask = np.array(im.convert("RGB"))
        assert mask.shape == old_mask.shape
        assert np.all((mask == 0).all(axis=2) | (mask == 255).all(axis=2))
        foreground = (mask == 255).all(axis=2)
        old_foreground = (old_mask == 255).all(axis=2)
        changed = foreground != old_foreground
        result = {
            "status": "passed", "tested_at": now.isoformat(), "device": "CPU", "threads": 4,
            "mode": "whole", "selected_output": "seg", "image_size": baseline["image_size"],
            "image": str(run / "SX_input.jpg"), "image_sha256": baseline["image_sha256"],
            "original_model": str(source_model), "original_model_sha256": baseline["model_sha256"],
            "trial_model": str(trial_model), "trial_model_sha256": digest(trial_model),
            "normalization": {"input_before": "RGB/255 float32 NCHW",
                              "formula": "(x - mean) / std", "mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225],
                              "parameters_confirmed_from_training": False,
                              "placement": "Inside test ONNX, after service resize/RGB/255"},
            "mask": str(mask_path), "baseline_mask": baseline["mask"],
            "baseline_report": str(args.baseline.resolve()),
            "http_status": 200, "windows": 1,
            "inference_ms": float(headers["X-Inference-Ms"]),
            "server_total_ms": float(headers["X-Total-Ms"]),
            "startup_and_warmup_seconds": round(startup_seconds, 3),
            "white_class_pixel_percent_before": round(float(old_foreground.mean()*100), 6),
            "white_class_pixel_percent_after": round(float(foreground.mean()*100), 6),
            "different_pixel_count": int(changed.sum()),
            "different_pixel_percent": round(float(changed.mean()*100), 6),
            "notes": ["ImageNet parameters are an experiment; actual training preprocessing is unknown.",
                      "Score resize, argmax, palette, image and original model weights are unchanged.",
                      "Pixel difference is change relative to baseline, not an accuracy measurement.",
                      "Production EXE, model and config are unchanged; normalized copy is test-only."]}
        Image.fromarray(np.repeat((changed.astype(np.uint8)*255)[..., None], 3, axis=2)).save(run/"SX_changed_pixels.png")
        args.publish.mkdir(parents=True, exist_ok=True)
        for name in ("SX_mask_imagenet.png", "SX_changed_pixels.png"):
            shutil.copy2(run/name, args.publish/name)
        for path in (run/"report.json", args.publish/"SX_normalization_report.json"):
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
