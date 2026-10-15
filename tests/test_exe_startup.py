"""Verify no-argument EXE launch, whole JLD inference and a Unicode portable folder."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".test-deps"))
import io
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from email.parser import Parser
import re
import numpy as np
from PIL import Image

RUN = ROOT / "tests" / "generated" / ("exe-startup-" + uuid.uuid4().hex[:10])
RUN.mkdir(parents=True)
FOREIGN_CWD = RUN / "unrelated-working-directory"
FOREIGN_CWD.mkdir()


def check_port_free():
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9600)) == 0:
            raise RuntimeError("Port 9600 is already occupied; leave the existing service untouched.")


def run_case(folder, name, fields, image_path, expected=None):
    check_port_free()
    log_path = RUN / (name + ".log")
    log = log_path.open("wb")
    process = subprocess.Popen([str(folder / "onnx-seg-server.exe")],
                               cwd=FOREIGN_CWD, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace"))
            try:
                with urllib.request.urlopen("http://127.0.0.1:9600/health", timeout=0.5) as response:
                    health = json.load(response)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("EXE did not become ready")
                time.sleep(0.1)
        print(name + " ready; uploading image...", flush=True)
        mask_path = RUN / (name + "_mask.png")
        header_path = RUN / (name + "_headers.txt")
        args = ["curl.exe", "--silent", "--show-error", "--fail-with-body", "--max-time", "600",
                "-F", "file=@" + str(image_path), "-F", "catagory=" + fields[0],
                "-F", "item=" + fields[1], "http://127.0.0.1:9600/infer",
                "--output", str(mask_path), "--dump-header", str(header_path)]
        response = subprocess.run(args, capture_output=True, timeout=620,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert response.returncode == 0, response.stderr
        blocks = [b for b in re.split(r"\r?\n\r?\n", header_path.read_text())
                  if b.startswith("HTTP/")]
        assert blocks[-1].startswith("HTTP/1.1 200")
        headers = Parser().parsestr(blocks[-1].split("\n", 1)[1])
        assert int(headers["X-Tile-Count"]) == 1
        assert headers["Content-Type"] == "image/png"
        assert headers["X-Model-Key"] == fields[0] + "_" + fields[1].zfill(3)
        with Image.open(mask_path) as image:
            mask = np.array(image.convert("RGB"))
            assert image.size == Image.open(image_path).size
        if expected is not None:
            assert np.array_equal(mask, expected), "RGB mask mismatch"
        else:
            assert np.all(np.logical_or(np.all(mask == 0, axis=2),
                                       np.all(mask == 255, axis=2)))
        result = {"name": name, "status": "passed", "arguments": [],
                  "cwd": str(FOREIGN_CWD), "executable_folder": str(folder), "health": health,
                  "mask": str(mask_path), "mask_width": mask.shape[1], "mask_height": mask.shape[0],
                  "windows": 1, "server_total_ms": float(headers["X-Total-Ms"])}
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        log.close()


def main():
    results = []
    actual_config = json.loads((ROOT / "config" / "1001_004.json").read_text(encoding="utf-8"))
    assert actual_config["model_detail"]["cut_flag"] == 0
    print("Starting installed EXE without arguments from a different folder...", flush=True)
    results.append(run_case(ROOT, "installed_JLD_whole", ("1001", "4"),
                            ROOT / "tests" / "inputs" / "JLD_input.jpg"))
    portable = RUN / "中文路径 便携服务"
    (portable / "models").mkdir(parents=True)
    (portable / "config").mkdir()
    for source in [ROOT / "onnx-seg-server.exe", *ROOT.glob("*.dll")]:
        shutil.copy2(source, portable / source.name)
    shutil.copy2(ROOT / "models" / "demo_rgb.onnx", portable / "models" / "demo_rgb.onnx")
    shutil.copy2(ROOT / "config" / "1000_001.json", portable / "config" / "1000_001.json")
    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    expected = palette[np.random.default_rng(123).integers(0, 3, size=(32, 40))]
    image_path = RUN / "rgb_input.png"
    Image.fromarray(expected).save(image_path)
    print("Starting a moved copy in a Unicode folder without arguments...", flush=True)
    results.append(run_case(portable, "unicode_portable_RGB", ("1000", "1"),
                            image_path, expected))
    report = {"status": "passed", "checks": results,
              "note": "Both EXEs launched without --root from an unrelated working directory."}
    path = RUN / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ALL EXE STARTUP CHECKS PASSED: " + str(path), flush=True)


if __name__ == "__main__":
    main()
