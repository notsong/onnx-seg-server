# SX 已确认输入预处理测试

日期：2026-10-10。用户提供了模型侧 data/transforms.py：Dataset 中先保持比例 letterbox，居中补 RGB 128 灰边；验证/测试仅 Normalize，mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]，再 ToTensorV2。推理不执行随机翻转、仿射、弹性、亮度/噪声增强。

## 服务配置与实现

原始 ONNX：models/1003_010_SX.onnx，模型未改写。
配置：config/1003_010.json，catagory=1003、item=10、cut_flag=0。

新增可选字段：
- resize_mode=letterbox，默认 stretch。
- mean=[0.485,0.456,0.406]。
- std=[0.229,0.224,0.225]。

mean/std 在 RGB/255 的 float32 NCHW 张量上执行。未填写时 mean=0/std=1。ONNX 内已有归一化时应省略 mean/std，以免重复。JLD 等未设置新字段的配置继续按其原有处理运行。

本次原图 2448×2048：按 Python int 截断缩放到 1024×856，上下各补 84 像素灰边。输出先把类别分数恢复输入空间、裁去补边，再恢复原图分数并 argmax、上色。最终 mask 为 2448×2048，黑色为类别0、白色为类别1。

## 真实 CPU 结果

| 指标 | 结果 |
|---|---:|
| HTTP 状态 | 200 |
| 推理窗口数 | 1 |
| CPU 线程数 | 4 |
| 推理时间 | 16.606 秒 |
| 服务端合计 | 16.803 秒 |
| 白色类别占比 | 55.785% |
| 相对直接缩放 + mean/std 的类别变化 | 1.724% |
| 相对最初 RGB/255 直接缩放的类别变化 | 4.925% |

这些差异说明输入几何处理会影响结果；没有标注真值或原模型 Python 推理输出，不将类别变化比例解释为准确率。CPU 时间为预热后单次请求，不代表稳定基准。

## 验证

56 项自动检查通过：原有整图、滑窗、颜色和错误处理，以及新增 RGB 通道归一化、奇数补边截断/居中/裁除、组合归一化、低分辨率输出先恢复输入空间再裁灰边、非法 mean/std/resize_mode 启动拒绝。
自动测试：D:\GitHubProject\onnx-seg-server\tests\generated\run-cdcfc23991\report.json
真实模型日志、HTTP 头和原图备份：D:\GitHubProject\onnx-seg-server\tests\generated\sx-20261010-111602-228ff9

正式 EXE：D:/GitHubProject/onnx-seg-server/onnx-seg-server.exe。
SX 正式配置已更新；重新启动服务后，传 file、catagory=1003、item=10 即使用这套输入处理。

根目录 EXE 零参数启动检查通过：默认 9600 端口加载全部 4 份配置，加载和预热约 16.362 秒（默认 CPU 线程设置，和上文 4 线程单独 SX 测试条件不同）。EXE SHA256：755179bc161a7802d945352370c728c5997e1352ed8f6d573f72b7be37323570。测试创建的服务已关闭。
