# Third-party sources and notices

Own service code is in src/. Full third-party license texts are retained in third_party/licenses/. Keep these notices and the applicable dependency licenses when redistributing this project or its binaries.

| Component | Version/source | Use | License text |
|---|---|---|---|
| ONNX Runtime | [Microsoft v1.22.0 CPU Windows x64](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0) | C++ inference API and CPU DLLs | onnxruntime-LICENSE.txt (MIT); onnxruntime-ThirdPartyNotices.txt |
| ONNX Runtime inference examples | [c_cxx at 16e32e87c81648773b8da2a80434a672e3b9dad6](https://github.com/microsoft/onnxruntime-inference-examples/tree/16e32e87c81648773b8da2a80434a672e3b9dad6/c_cxx) | Reference for session/tensor/run usage; service code written for this contract | onnxruntime-examples-LICENSE.txt (MIT) |
| cpp-httplib | [v0.26.0](https://github.com/yhirose/cpp-httplib/tree/v0.26.0) | Unmodified single header; HTTP/multipart/binary response | cpp-httplib-LICENSE.txt (MIT) |
| nlohmann/json | [v3.11.3](https://github.com/nlohmann/json/tree/v3.11.3) | Unmodified single header; configuration and JSON errors | nlohmann-json-LICENSE.txt (MIT) |
| OpenCV | [4.12.0](https://github.com/opencv/opencv/releases/tag/4.12.0) | Image codecs, RGB tensor construction, resize and pixel arithmetic | opencv-LICENSE.txt (Apache-2.0), opencv-3rdparty/ |
| MMSegmentation | [EncoderDecoder.slide_inference at b040e147adfa027bbc071b624bedf0ae84dfc922](https://github.com/open-mmlab/mmsegmentation/blob/b040e147adfa027bbc071b624bedf0ae84dfc922/mmseg/models/segmentors/encoder_decoder.py) | Adapted sliding-window positioning and score/count averaging | mmsegmentation-LICENSE.txt (Apache-2.0) |
| 7-Zip standalone 7zr | [Official download](https://www.7-zip.org/), SHA256 in deps.lock.json | Build-time extraction of official OpenCV package; not linked or needed by server | 7zip-LICENSE.txt and LGPL-2.1.txt; [official license information](https://www.7-zip.org/license.txt) |

## Adaptation notice: MMSegmentation

The C++ starts() function and tiled branch of Segmenter::predict in src/segmenter.cpp adapt the window-boundary positioning and overlapping prediction accumulation/count averaging approach from EncoderDecoder.slide_inference.

Upstream attribution: OpenMMLab / MMSegmentation contributors. Upstream license: Apache License, Version 2.0; full license retained in third_party/licenses/mmsegmentation-LICENSE.txt. Original source header: Copyright (c) OpenMMLab. All rights reserved.

Modifications in this project: translated the algorithm into C++17/OpenCV, restricted to one image and sequential CPU windows, right/bottom black-padded small images to a fixed ONNX input size, used configured shared integer stride, validated coverage, and rendered argmax predictions through an RGB palette. The upstream PyTorch implementation is not included as a runtime dependency.

## Dependency archives

deps.lock.json pins official download URLs and SHA256 values. bootstrap.ps1 copies the ONNX Runtime license and bundled ThirdPartyNotices, the OpenCV license, and the license/copyright files supplied in OpenCV's 3rdparty source directory. The OpenCV package also includes the original source and licenses under deps/opencv/sources; the copied 3rdparty directory lists supplied notices and does not imply every optional component is used by this server.

Python onnx/protobuf/numpy/Pillow are test-generation dependencies only and are not embedded into the C++ executable.

The full xmba15/onnx_runtime_cpp, onnxruntime-server and lite.ai.toolkit implementations are not included in this project.

## Windows compiler runtime deployment

CMake's InstallRequiredSystemLibraries module copies the unmodified MSVC Release runtime DLLs from the installed Visual Studio redistributable directory alongside the executable.
This local build uses the Visual Studio 18 Community MSVC 14.51.36231 redistributable files. Those Microsoft runtime binaries retain their own copyright and applicable license terms.


## Optional OpenVINO CPU runtime

The optional openvino/ directory contains native DLLs from Intel's official
onnxruntime-openvino 1.22.0 and OpenVINO 2025.1.0 Windows x64 wheels.
Only native DLLs are deployed; Python modules and bindings are not used by the C++ service.
Exact archive URLs and SHA256 hashes are pinned in deps.lock.json.

OpenVINO is licensed under Apache-2.0; ONNX Runtime is licensed under MIT.
Full license and bundled third-party notices, including oneDNN and oneTBB,
are retained under third_party/licenses/openvino/.
Keep them when redistributing the service.
Release compatibility source: https://github.com/intel/onnxruntime/releases/tag/v5.7
