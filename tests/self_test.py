"""End-to-end CPU tests using synthetic models, without business model weights."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-deps"))
sys.path.insert(0, str(ROOT / "scripts"))
import argparse
import copy
import io
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

PALETTE = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def multipart(fields, files):
    boundary = "seg-test-" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{name}"\r\n\r\n{value}\r\n').encode()
    for name, data in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{name}"; filename="image.png"\r\n'
                 "Content-Type: image/png\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), "multipart/form-data; boundary=" + boundary


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, executable, root, work_mb=1024, threads=1, startup_timeout=20, request_timeout=20,
                 backend=None):
        self.request_timeout = request_timeout
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = root / f"server-{self.port}.log"
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [str(executable), "--root", str(root), "--port", str(self.port),
             "--threads", str(threads), "--max-work-mb", str(work_mb)] +
            (["--backend", backend] if backend else []),
            stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(self.log_path.read_text(encoding="utf-8", errors="replace"))
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=0.5) as response:
                        if json.load(response)["status"] == "ok":
                            return
                except (OSError, urllib.error.URLError):
                    time.sleep(0.05)
            raise RuntimeError("Server did not become ready")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()

    def infer(self, array=None, item=1, fields=None, files=None, raw=None):
        if raw is None:
            if fields is None:
                fields = [("catagory", "1000"), ("item", str(item))]
            if files is None:
                files = [("file", png(array))]
            data, content_type = multipart(fields, files)
        else:
            data, content_type = raw, "application/json"
        request = urllib.request.Request(self.url + "/infer", data=data,
                                         headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()


def positions(length, window, stride):
    last = max(length - window, 0)
    return sorted(set(range(0, last + 1, stride)) | {last})


def fixture_config(item, model="rgb.onnx", shape=(32, 40), tiled=False, stride=24,
                   object_colors=False):
    colors = PALETTE.tolist()
    if object_colors:
        colors = {str(i): color for i, color in enumerate(colors)}
    return {"catagory": "1000", "item": item, "model_detail": {
        "model_name": model, "input_shape": list(shape), "num_classes": 3,
        "colors": colors, "cut_flag": int(tiled), "stride": stride}}


def write_config(root, cfg, filename=None):
    (root / "config").mkdir(parents=True, exist_ok=True)
    filename = filename or f"1000_{cfg['item']:03d}.json"
    (root / "config" / filename).write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def reference_position(array, stride):
    """Independent numpy score accumulation oracle for tile-local class biases."""
    height, width = array.shape[:2]
    sums = np.zeros((height, width, 3), np.float32)
    counts = np.zeros((height, width, 1), np.float32)
    bias = np.zeros((32, 40, 3), np.float32)
    bias[:, :, 0] = np.linspace(0, 0.9, 40, dtype=np.float32)[None, :]
    bias[:, :, 1] = 0.2
    for y in positions(height, 32, stride):
        for x in positions(width, 40, stride):
            rh, rw = min(32, height - y), min(40, width - x)
            sums[y:y+rh, x:x+rw] += array[y:y+rh, x:x+rw].astype(np.float32) / 255 + bias[:rh, :rw]
            counts[y:y+rh, x:x+rw] += 1
    assert np.all(counts > 0)
    return PALETTE[np.argmax(sums / counts, axis=2)]


def reference_resize(scores, height, width):
    # Half-pixel bilinear coordinates, evaluated independently from OpenCV.
    sh, sw = scores.shape[:2]
    ys = np.clip((np.arange(height) + 0.5) * sh / height - 0.5, 0, sh - 1)
    xs = np.clip((np.arange(width) + 0.5) * sw / width - 0.5, 0, sw - 1)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, sh - 1), np.minimum(x0 + 1, sw - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = scores[y0[:, None], x0[None, :]] * (1 - fx) + scores[y0[:, None], x1[None, :]] * fx
    bottom = scores[y1[:, None], x0[None, :]] * (1 - fx) + scores[y1[:, None], x1[None, :]] * fx
    return top * (1 - fy) + bottom * fy


def main():
    from make_demo import make_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path,
                        default=ROOT / "build" / "Release" / "onnx-seg-server.exe")
    parser.add_argument("--backend", choices=["onnxruntime", "openvino"])
    args = parser.parse_args()
    executable = args.exe.resolve()
    if not executable.is_file():
        raise RuntimeError("Build the server first")
    run = ROOT / "tests" / "generated" / ("run-" + uuid.uuid4().hex[:10])
    run.mkdir(parents=True)
    models = run / "models"
    make_model(models / "rgb.onnx")
    make_model(models / "square.onnx", 32, 32)
    make_model(models / "position.onnx", mode="position")
    make_model(models / "lowres.onnx", mode="lowres")
    make_model(models / "nonfinite.onnx", mode="nonfinite")
    make_model(models / "dynamic.onnx", dynamic=True)
    make_model(models / "模型测试.onnx")
    make_model(models / "binary.onnx", mode="binary")
    make_model(models / "auxiliary.onnx", mode="auxiliary")
    make_model(models / "ambiguous.onnx", mode="ambiguous")
    make_model(models / "geometry.onnx", mode="geometry")
    configurations = [
        fixture_config(1),
        fixture_config(2, tiled=True, object_colors=True),
        fixture_config(3, "square.onnx", (32, 32), True, 32),
        fixture_config(4, "position.onnx", tiled=True, stride=17),
        fixture_config(5, "lowres.onnx"),
        fixture_config(6, "nonfinite.onnx"),
        fixture_config(7, "dynamic.onnx"),
        fixture_config(8, "模型测试.onnx")]
    binary_cfg = fixture_config(9, "binary.onnx")
    binary_cfg["model_detail"].update(num_classes=2, colors=PALETTE[:2].tolist())
    configurations.append(binary_cfg)
    configurations.append(fixture_config(10, "auxiliary.onnx"))
    normalize_cfg = fixture_config(11)
    normalize_cfg["model_detail"].update(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    configurations.append(normalize_cfg)
    letterbox_cfg = fixture_config(12, "geometry.onnx")
    letterbox_cfg["model_detail"]["resize_mode"] = "letterbox"
    configurations.append(letterbox_cfg)
    combined_cfg = copy.deepcopy(normalize_cfg)
    combined_cfg["item"] = 13
    combined_cfg["model_detail"]["resize_mode"] = "letterbox"
    configurations.append(combined_cfg)
    tiled_normalize_cfg = copy.deepcopy(combined_cfg)
    tiled_normalize_cfg["item"] = 14
    tiled_normalize_cfg["model_detail"].update(cut_flag=1, stride=24)
    configurations.append(tiled_normalize_cfg)
    lowres_letterbox_cfg = fixture_config(15, "lowres.onnx")
    lowres_letterbox_cfg["model_detail"]["resize_mode"] = "letterbox"
    configurations.append(lowres_letterbox_cfg)
    for cfg in configurations:
        write_config(run, cfg)
    checks = []
    measurements = []

    def passed(name):
        checks.append(name)
        print("PASS", name, flush=True)

    def check_mask(server, array, item, expected, name, tile_count=None):
        status, headers, data = server.infer(array, item)
        assert status == 200, (name, status, data)
        assert headers.get_content_type() == "image/png"
        mask = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        assert mask.shape == array.shape, (name, mask.shape, array.shape)
        assert np.array_equal(mask, expected), (name, int(np.sum(mask != expected)))
        if tile_count is not None:
            assert int(headers["X-Tile-Count"]) == tile_count, headers
        assert headers["X-Model-Key"] == f"1000_{item:03d}"
        measurements.append({"case": name, "width": array.shape[1], "height": array.shape[0],
                             "windows": int(headers["X-Tile-Count"]),
                             "inference_ms": float(headers["X-Inference-Ms"]),
                             "total_ms": float(headers["X-Total-Ms"])})
        passed(name)
        return data

    server = Server(executable, run, backend=args.backend,
                    startup_timeout=120 if args.backend == "openvino" else 20)
    rng = np.random.default_rng(12345)
    try:
        array = PALETTE[rng.integers(0, 3, size=(32, 40))]
        sample = check_mask(server, array, 1, array, "whole RGB palette and exact dimensions", 1)
        (run / "mask_example.png").write_bytes(sample)
        check_mask(server, array, 10, array, "select segmentation when auxiliary output comes first", 1)
        for height, width in [(81, 113), (9, 13), (17, 95), (90, 11), (32, 40)]:
            array = PALETTE[rng.integers(0, 3, size=(height, width))]
            count = len(positions(height, 32, 24)) * len(positions(width, 40, 24))
            check_mask(server, array, 2, array, f"overlap/padding {height}x{width}", count)
        for height, width in [(64, 96), (61, 85)]:
            array = PALETTE[rng.integers(0, 3, size=(height, width))]
            count = len(positions(height, 32, 32)) * len(positions(width, 32, 32))
            check_mask(server, array, 3, array, f"stride=window {height}x{width}", count)
        array = np.tile(np.array([64, 128, 32], np.uint8), (77, 101, 1))
        count = len(positions(77, 32, 17)) * len(positions(101, 40, 17))
        check_mask(server, array, 4, reference_position(array, 17),
                   "overlap averages class scores before argmax", count)
        array = np.tile(PALETTE[1], (53, 79, 1))
        for item, name in [(1, "whole resize to original"), (5, "low-resolution output resize"),
                           (7, "dynamic ONNX dimensions"), (8, "Unicode model filename")]:
            check_mask(server, array, item, array, name, 1)

        array = PALETTE[rng.integers(0, 2, size=(32, 40))]
        check_mask(server, array, 9, array, "binary two-channel output", 1)
        # Smooth random scores avoid ambiguous argmax ties at interpolation boundaries.
        array = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
        pooled = (array.astype(np.float32) / 255).reshape(16, 2, 20, 2, 3).mean(axis=(1, 3))
        expected = PALETTE[np.argmax(reference_resize(pooled, 32, 40), axis=2)]
        check_mask(server, array, 5, expected, "bilinear score resize before argmax", 1)
        array = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)
        normalized = (array.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        check_mask(server, array, 11, PALETTE[np.argmax(normalized, axis=2)],
                   "RGB per-channel mean/std normalization", 1)
        bias = np.zeros((32, 40, 3), np.float32)
        bias[:, :, 0] = np.linspace(0, 0.913, 40, dtype=np.float32)[None, :]
        bias[:, :, 1] = np.linspace(0, 0.877, 32, dtype=np.float32)[:, None]
        bias[:, :, 2] = 0.417
        for height, width in ((9, 13), (17, 95), (90, 11), (23, 23)):
            scale = min(32 / height, 40 / width)
            rh, rw = max(1, int(height * scale)), max(1, int(width * scale))
            top, left = (32 - rh) // 2, (40 - rw) // 2
            cropped = bias[top:top+rh, left:left+rw]
            expected = PALETTE[np.argmax(reference_resize(cropped, height, width), axis=2)]
            check_mask(server, np.zeros((height, width, 3), np.uint8), 12, expected,
                       f"letterbox crop and restore {height}x{width}", 1)
        array = np.tile(PALETTE[1], (90, 11, 1))
        check_mask(server, array, 13, array, "letterbox plus normalization removes gray pads", 1)
        array = np.tile(PALETTE[1], (9, 13, 1))
        check_mask(server, array, 14, array, "normalized tiled small-image gray padding", 1)
        array = np.tile(PALETTE[1], (33, 11, 1))
        check_mask(server, array, 15, array, "low-resolution letterbox output restores before padding crop", 1)
        occupied = subprocess.run(
            [str(executable), "--root", str(run), "--port", str(server.port), "--threads", "1"],
            capture_output=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = (occupied.stdout + occupied.stderr).decode("utf-8", errors="replace")
        assert occupied.returncode == 1 and "Cannot bind host/port" in output, output
        passed("occupied port is rejected")

        valid_png = png(array)
        cases = [
            ("missing catagory", 400, {"fields": [("item", "1")], "files": [("file", valid_png)]}),
            ("invalid image", 400, {"files": [("file", b"not an image")]}),
            ("empty upload", 400, {"files": [("file", b"")]}),
            ("missing upload", 400, {"files": []}),
            ("category path traversal", 400, {"fields": [("catagory", "../1000"), ("item", "1")]}),
            ("noninteger item", 400, {"fields": [("catagory", "1000"), ("item", "1.1")]}),
            ("duplicate category", 400, {"fields": [("catagory", "1000"), ("catagory", "1000"),
                                                     ("item", "1")]}),
            ("duplicate upload", 400, {"files": [("file", valid_png), ("file", valid_png)]}),
            ("nonmultipart request", 400, {"raw": b"{}"}),
            ("unknown config", 404, {"item": 999})]
        for name, expected, kwargs in cases:
            status, headers, data = server.infer(array, **kwargs)
            assert status == expected, (name, status, data)
            assert headers.get_content_type() == "application/json"
            assert json.loads(data)["error"]
            passed(name)
        status, _, _ = server.infer(array, item="001")
        assert status == 200
        passed("zero-padded item accepted")
        status, _, data = server.infer(np.full((32, 40, 3), 255, np.uint8), item=6)
        assert status == 500, (status, data)
        assert json.loads(data)["error"]
        passed("nonfinite inference returns 500")
        with urllib.request.urlopen(server.url + "/health") as response:
            assert json.load(response)["status"] == "ok"
        passed("service remains available after inference failure")
    finally:
        server.close()

    invalid_cases = [
        ("filename mismatch", lambda d: d.update(item=2)),
        ("zero stride", lambda d: d["model_detail"].update(cut_flag=1, stride=0)),
        ("oversized stride", lambda d: d["model_detail"].update(cut_flag=1, stride=33)),
        ("missing stride", lambda d: (d["model_detail"].update(cut_flag=1),
                                     d["model_detail"].pop("stride"))),
        ("missing color", lambda d: d["model_detail"]["colors"].pop()),
        ("fractional color", lambda d: d["model_detail"]["colors"][0].__setitem__(0, 0.5)),
        ("class count mismatch", lambda d: d["model_detail"].update(
            num_classes=2, colors=PALETTE[:2].tolist())),
        ("input dimensions mismatch", lambda d: d["model_detail"].update(input_shape=[31, 40])),
        ("ambiguous segmentation outputs", lambda d: d["model_detail"].update(model_name="ambiguous.onnx")),
        ("unknown resize mode", lambda d: d["model_detail"].update(resize_mode="wrong")),
        ("missing normalization std", lambda d: d["model_detail"].update(mean=[0, 0, 0])),
        ("zero normalization std", lambda d: d["model_detail"].update(mean=[0, 0, 0], std=[1, 0, 1])),
        ("negative normalization std", lambda d: d["model_detail"].update(mean=[0, 0, 0], std=[1, -1, 1])),
        ("normalization length mismatch", lambda d: d["model_detail"].update(mean=[0, 0], std=[1, 1, 1])),
        ("nonnumeric normalization", lambda d: d["model_detail"].update(mean=[0, "x", 0], std=[1, 1, 1])),
        ("missing model", lambda d: d["model_detail"].update(model_name="missing.onnx")),
        ("model path traversal", lambda d: d["model_detail"].update(model_name="../other.onnx"))]
    for index, (name, mutate) in enumerate(invalid_cases):
        invalid_root = run / f"invalid-{index}"
        invalid_root.mkdir()
        # A junction is unnecessary: tiny models can simply be copied.
        (invalid_root / "models").mkdir()
        for model_name in ("rgb.onnx", "ambiguous.onnx"):
            (invalid_root / "models" / model_name).write_bytes((models / model_name).read_bytes())
        cfg = copy.deepcopy(configurations[0])
        mutate(cfg)
        write_config(invalid_root, cfg, "1000_001.json")
        process = subprocess.run(
            [str(executable), "--root", str(invalid_root), "--port", str(free_port()),
             "--threads", "1"], capture_output=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = (process.stdout + process.stderr).decode("utf-8", errors="replace")
        assert process.returncode == 1 and "Startup failed:" in output, (name, output)
        (invalid_root / "startup.log").write_text(output, encoding="utf-8")
        passed("startup rejects " + name)

    limited = Server(executable, run, work_mb=16, backend=args.backend,
                     startup_timeout=120 if args.backend == "openvino" else 20)
    try:
        array = np.zeros((1200, 1300, 3), np.uint8)
        status, _, data = limited.infer(array)
        assert status == 400, (status, data)
        assert "error" in json.loads(data)
        passed("image work-buffer limit")
    finally:
        limited.close()

    report = {"status": "passed", "check_count": len(checks), "checks": checks,
              "synthetic_cpu_measurements": measurements,
              "artifacts": str(run),
              "limitations": "Synthetic test models only; real model accuracy and CPU speed unverified."}
    report_path = run / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ALL {len(checks)} CHECKS PASSED. Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
