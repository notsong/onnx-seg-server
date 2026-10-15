# SX OpenVINO CPU 试验结果

2026-10-10；Intel Core 7 150U，10 核、12 逻辑线程；原始 2448×2048 JPEG、同一 SX ONNX、CPU FP32。

已交付可选的 C++ OpenVINO 启动入口。这次交替短测显示小幅收益；普通 EXE 继续默认使用原 CPU 后端。

| 后端 | 模型计算中位数 | 三次范围 | 服务总耗时中位数 |
|---|---:|---:|---:|
| 普通 CPU | 25.28 秒 | 25.27～26.80 秒 | 25.90 秒 |
| OpenVINO CPU FP32 | 23.36 秒 | 21.72～24.31 秒 | 23.99 秒 |

两组中位计算时间相比较，减少 7.58%。这是本机短测结果，不能作为稳定加速比例。

两个服务均先预热，每次仅运行一个请求；相邻交替比较如下，单位为秒，第二组反转顺序：

| 配对 | 普通 CPU | OpenVINO | 本组耗时减少 |
|---|---:|---:|---:|
| 1 | 26.80 | 23.36 | 12.8% |
| 2 | 25.28 | 21.72 | 14.1% |
| 3 | 25.27 | 24.31 | 3.8% |

测试期间机器性能状态变化明显：最初普通 CPU 三次为 10.99～11.38 秒，后续独立复测为 27.96 秒；初始 OpenVINO 为 27.06～33.33 秒。这些非相邻测量不适合直接推断后端加速倍数。最终交付设置采用 batch_size=1、全部核心类型、单流和 FP32，并按上表交替比较。

## 输出与验证

- 初次七个 HTTP 请求及交替六个请求全部成功，共 13 次；全部 mask 与普通 CPU 逐像素一致，为原图尺寸黑白 PNG。
- 两个后端保留相同的 C++ 灰色 letterbox、RGB/255、ImageNet mean/std、分数恢复及 argmax。未量化，原 ONNX 的 SHA256 未改变。
- 初始独立性能记录显示 SX 图由一个 OpenVINOExecutionProvider 子图执行，没有普通 CPU 节点回退。文件为 openvino_provider_profile_*.json。
- OpenVINO 和普通模式各通过 56 项回归检查。图片、归一化、滑窗、letterbox 和 HTTP 错误使用所选后端；部分启动失败检查验证共享的 C++ 配置校验逻辑。
- 部署后的正式目录成功加载并预热 4 份配置，health 为 status=ok、models=4，该次启动耗时 50.59 秒；启动时间不计入请求耗时。
- 启动入口可从其他工作目录调用。所有测试使用空闲端口，结束后关闭自己的进程。
- 尚未用标注评估业务准确率；逐像素一致仅说明这张图在两个后端上的输出相同。

## 启动

双击 D:/GitHubProject/onnx-seg-server/启动_OpenVINO.cmd，等待 Listening on http://127.0.0.1:9600。
模型仍放根目录 models/，JSON 仍放 config/。请求仍上传 file、catagory、item；SX 为 catagory=1003、item=10。
切换模式前关闭之前的服务窗口，以免占用同一端口。普通模式继续双击根目录 onnx-seg-server.exe。

手动启动（从项目根目录）：

```powershell
.\openvino\onnx-seg-server.exe --root . --backend openvino --port 9600 --threads 0
```

运行服务无需 Python。OpenVINO 原生 DLL 独立放在 openvino/；移动项目时保留运行库、启动文件、models/、config/ 和许可证。

## 构建与复测

scripts/build-openvino.ps1 编译并整理运行目录。无本地依赖时下载 deps.lock.json 固定地址的官方 wheel 归档，验证 SHA256，直接提取原生 DLL。这里使用 ONNX Runtime OpenVINO 1.22.0 + OpenVINO 2025.1.0。
CPU 参数为 FP32、LATENCY、单流、ANY_CORE、CPU pinning，关闭超线程；按服务 batch=1 约定覆盖 batch_size 符号。OpenVINO 路径关闭 ORT 图优化，交由 OpenVINO 优化。

```powershell
.\scripts\build-openvino.ps1
python .\tests\self_test.py --exe .\openvino\onnx-seg-server.exe --backend openvino
python .\tests\test_sx_openvino_paired.py --image <JPEG> --openvino-exe .\openvino\onnx-seg-server.exe --publish <输出目录>
```

本机已验证使用现有隔离依赖的构建部署流程；无本地缓存时的下载分支尚未在干净机器上重测。

完整数据：paired/SX_openvino_paired_report.json；初次数据：SX_openvino_comparison_report.json；部署数据：SX_openvino_deployment_report.json。

官方参考：[OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)、[Intel v5.7 发布](https://github.com/intel/onnxruntime/releases/tag/v5.7)、[CPU 调度](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html)。
