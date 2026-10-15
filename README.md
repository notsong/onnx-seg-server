# onnx-seg-server

第一版 C++17 / Windows x64 / CPU 语义分割服务。默认监听 `127.0.0.1:9600`，上传图片、检测项和模型号，返回与原图同尺寸的彩色 mask PNG。

## 当前交付

- `src/main.cpp`：HTTP 请求、配置路由、会话加载与预热、错误响应、耗时日志。
- `src/segmenter.cpp`：配置校验、RGB/NCHW 张量、整图和滑窗推理、分数拼接、颜色绘制。
- `config/1000_001.json`：整图演示；`config/1000_002.json`：滑窗演示，步长 24。
- `scripts/`：依赖准备、编译、启动、演示模型生成。
- `tests/self_test.py`：56 项确定性模型端到端检查。
- `docs/MODEL_CONTRACT.md`：真实 ONNX 接入要求。
- `THIRD_PARTY_NOTICES.md`：第三方来源和许可证。

本机项目位于 D:\GitHubProject\onnx-seg-server，已经完成 Release 编译，根目录的 onnx-seg-server.exe 已部署好，旁边包含 ONNX/OpenCV 和 MSVC 运行库。配置修改后重启服务生效。

## 立即运行

直接双击 D:\GitHubProject\onnx-seg-server\onnx-seg-server.exe，看到 Listening on http://127.0.0.1:9600 后即可调用。
关闭控制台窗口或按 Ctrl+C 停止服务。启动出错时，双击创建的窗口会保留错误信息，按回车退出。

模型放 EXE 同级的 models 文件夹，JSON 放 config 文件夹；修改后重启服务。当前 1001_004 已按用户确认改为整图缩放（cut_flag=0）。
程序默认读取 EXE 所在目录，移动目录或通过快捷方式启动也能正确找到配置。请一并保留 EXE、DLL、models、config。
简明说明见根目录“使用说明.txt”。

也可在 PowerShell 中运行：

```powershell
Set-Location D:\GitHubProject\onnx-seg-server
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

另开一个终端调用。示例图片和 demo_rgb.onnx 已生成；它根据 RGB 通道的最大值分类，仅用于验证服务流程。

```powershell
Set-Location D:\GitHubProject\onnx-seg-server
curl.exe --fail-with-body -F "file=@tests/demo_input.png" -F "catagory=1000" -F "item=2" http://127.0.0.1:9600/infer --output mask.png
curl.exe http://127.0.0.1:9600/health
```

item=1 使用整图推理，item=2 使用滑窗推理。HTTP 成功响应直接是 PNG 字节，不需要解析 JSON 或 Base64。颜色为配置中的 RGB 值，PNG 为三通道彩色图；不包含原图叠加或类别编号图。

## 从源码准备依赖和编译

安装 Visual Studio 的“使用 C++ 的桌面开发”和 CMake。本机验证环境为 Visual Studio 2026 / MSVC 19.51 / CMake 4.3.1 / x64 Release。

```powershell
Set-Location D:\GitHubProject\onnx-seg-server
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

bootstrap 下载官方 ONNX Runtime CPU 1.22.0、OpenCV 4.12.0 Windows 包和构建时使用的 7zr 解压器；文件 SHA256 固定在 deps.lock.json。cpp-httplib 0.26.0 和 nlohmann/json 3.11.3 的单头文件已随源码保存。依赖保存在本项目 deps 内，不安装到全局环境。

首次准备依赖需要联网，OpenCV 包包含源码，解压后约 925 MB。7zr 的下载地址由官方复用，若以后内容变化，SHA256 校验会停止；核对官方版本后再更新锁定值。

build.ps1 会查找 PATH 中的 CMake，或 Visual Studio 自带 CMake。也可指定路径和生成器：

```powershell
.\scripts\build.ps1 -CMake "C:\path\to\cmake.exe" -Generator "Visual Studio 18 2026"
```

build.ps1 编译完成后，会把 EXE 和运行库部署到项目根目录。

CMake 支持通过 -DONNXRUNTIME_ROOT 和 -DOpenCV_DIR 指向自行准备的依赖。默认复制本项目固定版本的运行 DLL；自行替换版本时需对应调整或复制 DLL。

## 请求约定

POST /infer，Content-Type: multipart/form-data：

| 字段 | 内容 |
|---|---|
| file | 一张可解码图片，恰好一个非空文件 |
| catagory | 检测项字符串，1～12 个 ASCII 数字，保留旧字段拼写 |
| item | 非负整数模型号；1 和 001 均对应模型号 1 |

1000 + 1 对应 config/1000_001.json；模型号最少补足三位，1234 对应 1000_1234.json。每个文本字段应只提交一次。catagory/item 也可放在 URL 查询参数中，不能与表单重复提交。

成功响应：
- Content-Type: image/png；
- X-Model-Key：实际配置编号；
- X-Tile-Count：运行的窗口数；
- X-Inference-Ms：所有窗口的模型运行总耗时；
- X-Total-Ms：服务端处理耗时，不含排队、网络传输。

错误响应是 JSON，例如：
```json
{"error":"Config not found: 1000_999.json"}
```

| 状态码 | 场景 |
|---|---|
| 400 | 缺字段、重复字段、编号不合法、图片无法解码、工作缓冲超过限制 |
| 404 | 配置编号不存在 |
| 500 | 推理失败、运行输出不符合约定、分数存在 NaN/Inf |
| 413 | HTTP 上传正文超过 64 MiB |

GET /health 返回状态和已加载的配置数量。服务启动失败会输出 Startup failed 和具体配置名，并退出；端口已占用也会明确失败。

## 精简配置

config/1000_001.json：

```json
{
  "catagory": "1000",
  "item": 1,
  "model_detail": {
    "model_name": "your_model.onnx",
    "input_shape": [512, 512],
    "num_classes": 2,
    "colors": [[0, 0, 0], [255, 0, 0]],
    "cut_flag": 1,
    "stride": 384
  }
}
```

- model_name 相对于 models 文件夹，也可写子目录，例如 unet/v1.onnx。必须是该目录内已存在的 .onnx 文件。
- input_shape 是 [高,宽]，两个整数在 1～16384 内，也是滑窗尺寸。
- num_classes 是 2～4096 的整数，必须与模型输出通道数一致。
- colors 必须恰好包含 num_classes 个 [R,G,B]，每个颜色分量为 0～255 的整数。也支持 {"0":[0,0,0],"1":[255,0,0]}，键必须连续覆盖 0～C-1。
- cut_flag 支持 0/1 和 false/true。
- resize_mode 可选 stretch（默认）或 letterbox（保持比例、居中灰边）。
- mean/std 可选，必须同时填写，均为三个 RGB 通道的有限数字，std 必须为正。例如 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。ONNX 图里已有归一化时省略这两个字段。
- 开启切图时必须填写 stride，为两个方向共用的正整数步长，不超过窗口高和宽。关闭切图时 stride 不参与推理。
- 配置文件名必须与其 catagory/item 一致。服务启动时校验所有配置、创建会话、黑图预热；任何一份无效都会使启动失败。

接入真实模型时把 ONNX 放进 models，新增或修改配置并重启。首次交付的 1000_001/1000_002 配置指向测试模型。2026-10-10 新增真实晶粒模型 1001_004，输入为 1024×1024、两类输出、黑白调色板、默认整图缩放（cut_flag=0）；stride=768 保留但不参与整图推理。

## 推理方式

输入图片解码为三通道，将 BGR 转为 RGB、除以 255，生成 float32 [1,3,H,W]。可选 mean/std 按 RGB 通道执行 (x-mean)/std；未填写时只做 RGB/255。模型必须包含唯一一个 float32 [1,C,h,w] 类别分数输出，其中 C 对应 num_classes；可同时包含单通道 boundary 等辅助输出。多个输出都符合类别分数形状时会报告歧义并拒绝启动。

整图模式的 resize_mode 默认 stretch，直接缩放至输入尺寸。letterbox 则保持宽高比：scale=min(输入高/图高,输入宽/图宽)，尺寸按 int 截断并至少为 1，居中补 128 灰边，奇数余量放在下侧/右侧。输出每一类的分数先恢复输入空间、去掉灰边，再用双线性插值恢复原图尺寸，之后 argmax 并上色。请按训练/原 Python 推理流程选择。

滑窗模式在原图上切窗，末尾窗口贴齐右/下边界；不足一个窗口的维度向右或向下补黑，letterbox 配置采用 128 灰色补齐。mean/std 对每个窗口执行。每个窗口输出先恢复为窗口大小，仅取有效图像区域累计类别分数和覆盖次数；重叠区域平均后统一 argmax。不在窗口内先生成颜色图再拼接。即使 stride 等于窗口，非整除边界的末尾窗口仍可能重叠。

每份配置复用一个 ONNX Runtime CPU 会话。第一版由一个 HTTP 工作线程串行处理请求，窗口逐个运行；ONNX Runtime 内部使用 CPU 多线程。日志记录 decode/preprocess/inference/postprocess/encode/total 各阶段耗时。

可调整：
```powershell
.\scripts\run.ps1 -Port 9600 -Threads 4 -MaxWorkMb 1024
```

Threads=0 采用 ONNX Runtime 默认线程设置。MaxWorkMb 对图片分数/拼接等工作缓冲做保守估算，默认 1024 MiB；它不是整个进程的硬内存上限，模型权重、运行中间张量、图像解码和 HTTP 数据仍会另外占用内存。若需要其他电脑访问，可传 -HostAddress 0.0.0.0。

## 测试与重建演示模型

Python 仅用于生成测试模型和执行自测；C++ 服务运行不需要 Python。

```powershell
Set-Location D:\GitHubProject\onnx-seg-server
python -m pip install --target .test-deps -r .\tests\requirements.txt
python .\scripts\make_demo.py
python .\tests\self_test.py
python .\tests\test_exe_startup.py
```

自测使用随机空闲本地端口，结束后关闭自己创建的服务进程。产物保存在 tests/generated/run-*/，包括 report.json、示例 mask 和阶段耗时日志。

已通过 56 项检查：整图、RGB 两种调色板格式、二分类、输出分数插值、重叠平均、无重叠网格、小图、窄长图、非整步长、动态尺寸、中文文件名、HTTP 异常和启动校验。详见 docs/VALIDATION.md。

2026-10-10 已接入并测试 1001_004_JLD.onnx，使用用户提供的原始 2448×2048 图片。真实模型结果及 CPU 耗时见 docs/JLD_MODEL_TEST.md；尚未与原 DLL 或标注结果做准确率对照。其他模型仍需要分别验证，合成模型的毫秒数不能用于估算它们的速度。

2026-10-10 已接入 1003_010_SX.onnx，配置为 config/1003_010.json，调用 catagory=1003、item=10，整图 letterbox、ImageNet mean/std、两类黑白 mask。用户随后提供训练 transforms，正式配置已按该输入流程更新，最新结果见 docs/SX_PREPROCESS_TEST.md。原模型保留 seg 与 boundary 两个输出，服务使用 seg；无需重新导出。原始 2448×2048 图片的 CPU 测试见 docs/SX_MODEL_TEST.md，可运行 tests/test_sx_model.py --image <JPEG> 重现。


## 可选 OpenVINO CPU 模式

双击项目根目录的 启动_OpenVINO.cmd，使用原生 C++ OpenVINO CPU FP32 后端，端口仍为 9600。
模型与 JSON 继续使用原 models/、config/，接口字段与 PNG 输出保持一致。
普通模式继续双击根目录 onnx-seg-server.exe。切换模式前关闭原服务窗口。

服务运行无需 Python；原生 DLL 放在 openvino/。重建试验版运行 scripts/build-openvino.ps1。
官方依赖地址及 SHA256 固定在 deps.lock.json，许可证保留在 third_party/licenses/openvino/。
已验证四份正式配置启动与 56 项回归检查。SX 交替短测的中位计算时间约减少 7.6%，结果逐像素一致；机器性能状态波动明显，不能保证所有模型获得相同收益。
详细结果见 docs/SX_OPENVINO_TEST.md。
