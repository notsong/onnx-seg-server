# SX 模型测试记录

测试日期：2026-10-10。Windows x64，Intel Core 7 150U，ONNX Runtime CPU 1.22.0，OpenCV 4.12.0，4 个推理线程。

## 接入与结果

模型：models/1003_010_SX.onnx（400,622,642 字节）。
配置：config/1003_010.json，检测项 catagory=1003，模型号 item=10。
原模型未改写，SHA256：e3f66db5ce8c46849be80794e48da4ea233c846f00098962da12c3eb5b2717b9。

输入为 RGB float32 [batch,3,1024,1024]，实际导出文件 opset 18、IR 10。
用户提供导出代码中的 opset_version=17 是请求值；本次测试的实际文件为 opset 18，已由现有 ONNX Runtime 成功加载。
模型包含 seg=[1,2,1024,1024] 与 boundary=[1,1,1024,1024]。
服务现支持自动选择唯一符合配置类别数的语义分数输出，此次使用 seg 生成 mask。

原始 JPEG 为 **2448×2048**，上传时没有缩放或裁剪。整图模式 cut_flag=0，RGB/255，直接缩放到 1024×1024；按类别分数双线性恢复到原图尺寸，argmax 后绘制配置颜色。
返回 HTTP 200，PNG 尺寸 **2448×2048**，类别 0 黑色、类别 1 白色；全部像素均属于配置调色板。白色类别占 56.603%。

## 单次 CPU 耗时

| 阶段 | 时间 |
|---|---:|
| 模型加载与黑图预热 | 30.446 秒 |
| 图片解码 | 121.299 毫秒 |
| 预处理 | 28.117 毫秒 |
| 模型推理 | 31.896 秒 |
| 分数恢复与上色 | 48.130 毫秒 |
| PNG 编码 | 301.455 毫秒 |
| 服务端处理合计 | 32.404 秒 |
| 本地 HTTP 客户端用时 | 32.609 秒 |

这是预热后一次请求的记录，不代表稳定基准或并发性能。启动时会预热所有配置，加载多个真实模型会增加等待时间和内存占用。

## 预处理确认状态

用户提供的 torch.onnx.export 片段只定义模型导出、张量命名与动态维度，不会说明模型外部的数据预处理。检查本次实际 ONNX：input 直接进入 encoder.backbone.embeddings.patch_embeddings 的 Conv，输入路径没有显式的减 mean、除 std 操作。该检查不能证明训练时未做归一化；仍需查看训练/原 Python 推理时输入模型前的代码。

本次测试明确使用 RGB/255，没有额外 mean/std。没有提供类别语义、原 Python 推理结果或标注真值，尚不能据此评价分割准确率。如果训练还需要归一化，应对齐后再比较识别结果。

## 检查与可复现文件

- 测试脚本：tests/test_sx_model.py，使用实际 C++ exe、multipart 上传和 PNG 响应。
- 原始输入备份：tests/inputs/SX_input.jpg。
- 真实模型报告：tests/generated/sx-20261010-104214-b85e47/report.json。
- 同目录：SX_mask_whole.png、headers.txt、server-*.log。
- 回归测试：42 项通过，tests/generated/run-c5e22f9fe1/report.json。

## 根目录 EXE 交付验证

根目录 onnx-seg-server.exe 已更新，SHA256：9e87f4951681ff2c01f838c6ca95d291592ff14da9bc5d873403117ea508e53c。
从其他当前目录以零参数启动，默认 9600 端口可用，health 返回 models=4，包括 JLD、SX 和两份演示配置。默认 CPU 线程设置下本次所有配置加载及预热约 20.56 秒。此时间与上文单独 SX、4 线程的测试条件不同。相关记录：同测试目录的 SX_exe_startup_report.json、SX_exe_startup.log。测试服务已退出。

正常使用：双击 D:\GitHubProject\onnx-seg-server\onnx-seg-server.exe。POST /infer 上传 file、catagory=1003、item=10，即使用本次 SX 配置。若已有旧服务窗口，请先关闭后重新打开。

## 后续预处理更新

用户随后提供模型训练/验证 transforms，确认使用 ImageNet mean/std，并在 Dataset 中先做居中 128 灰边 letterbox。上文记录为早期直接缩放测试；当前正式 SX 配置已按提供流程更新，最新测试见 SX_PREPROCESS_TEST.md。
