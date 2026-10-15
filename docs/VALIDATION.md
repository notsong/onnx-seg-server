# 第一版验证记录

日期：2026-10-09  
项目：D:\GitHubProject\onnx-seg-server  
环境：Windows x64，Visual Studio 2026 / MSVC 19.51，CMake 4.3.1；ONNX Runtime CPU 1.22.0，OpenCV 4.12.0。

## 结果

Release 编译成功。tests/self_test.py 的 40 项检查全部通过；使用 Windows PowerShell 5.1 启动 scripts/run.ps1，采用默认 CPU 线程设置（Threads=0），在 9600 端口使用 curl.exe 上传演示图片，成功获得 HTTP 200 的原图尺寸彩色 PNG。

自动测试报告：
- tests/generated/run-5f7639885d/report.json
- 同目录 server-*.log：逐请求的 decode/preprocess/inference/postprocess/encode/total。
- tests/generated/default-port-smoke/report.json：默认端口联调。
- tests/generated/default-port-smoke/mask.png：91×73 演示输出。
- tests/generated/default-port-smoke/server.log：默认启动及阶段耗时。

测试进程已关闭，未留下常驻测试服务。正常使用时运行 scripts/run.ps1。

## 覆盖内容

| 类别 | 验证 |
|---|---|
| 整图 | RGB 通道次序、原图尺寸恢复、两通道二分类、低分辨率输出 |
| 插值顺序 | 独立 NumPy 双线性插值参考：先恢复类别分数，再 argmax |
| 滑窗 | 步长小于窗口、步长等于窗口、非整步长边缘贴齐 |
| 形状边界 | 9×13 小图、17×95 和 90×11 窄长图、32×40 同尺寸、61×85 非整除尺寸 |
| 拼接顺序 | 具有窗口位置偏置的 ONNX 与独立 NumPy 分数累加/平均参考逐像素一致 |
| 调色板 | RGB 数组及数字键对象，PNG 中的 RGB 值逐像素一致 |
| 模型加载 | 动态空间尺寸、中文 ONNX 文件名、会话预热 |
| HTTP | 缺字段、空/缺文件、错误图片、重复字段、重复上传、非法编号、非 multipart、未知配置 |
| 故障 | NaN/Inf 输出返回 500，之后服务仍可响应；端口占用启动失败 |
| 启动校验 | 文件编号、步长 0/过大/缺失、颜色数量/类型、类别/输入维度不匹配、模型缺失/路径越界 |
| 缓冲预算 | 图片工作缓冲超过 --max-work-mb 返回 400 |

测试模型由 scripts/make_demo.py 使用 ONNX 工具生成，核心为确定性的 1×1 Conv 或小型组合。合成模型验证了数据处理、滑窗算法和服务协议，不验证业务模型的识别质量。

## 默认端口示例记录

输入和输出都是 91×73，模型号 2，12 个窗口，返回调色板颜色与输入一致。

该次默认线程设置的服务日志：
| 阶段 | 毫秒 |
|---|---:|
| 图像解码 | 0.262 |
| 预处理（各窗口累计） | 0.084 |
| ONNX 推理（各窗口累计） | 0.377 |
| 分数恢复、拼接与上色 | 0.610 |
| PNG 编码 | 0.246 |
| 服务端合计 | 1.669 |

这是合成小模型的一次记录，不能据此估算真实 UNet/DINO/Mask2Former 的 CPU 速度。HTTP 头中的 X-Total-Ms 在设置响应头时读取，为 1.651 ms；日志稍后读取，包含后续响应头构建和日志格式化的少量时间。排队和网络传输均未计入。

## 真实模型待验证

2026-10-09 首次交付时尚未提供真实 ONNX。2026-10-10 已完成 1001_004_JLD 真实图片测试，详见 JLD_MODEL_TEST.md。其他模型接入后需对齐原 Python 的缩放、归一化、输出类别顺序和滑窗语义，验证 mask 一致性，并用真实图片记录上述 CPU 阶段耗时。本版要求一个 RGB float32 NCHW 输入和一个 [1,C,h,w] float32 语义分数输出；不直接处理检测框或实例分割原始输出。

## 2026-10-10 SX 适配验证

Release 重新编译成功，tests/self_test.py 的 42 项检查通过。新增检查验证辅助 boundary 输出在第一个位置时仍选择类别分数，并在两个输出都符合类别分数形状时拒绝歧义。完整报告：tests/generated/run-c5e22f9fe1/report.json。

真实 SX 模型通过 C++ HTTP 接口返回原图尺寸的黑白 PNG，详见 SX_MODEL_TEST.md。当前契约允许辅助输出，但仍要求唯一可识别的 [1,C,h,w] float32 语义分数输出。

## SX 训练预处理对齐

按用户提供的 data/transforms.py 新增可选 mean/std 与 resize_mode=letterbox。56 项检查通过，报告 tests/generated/run-cdcfc23991/report.json。相同 SX 原图通过真实 C++ HTTP 接口返回原尺寸 PNG，最新结果见 SX_PREPROCESS_TEST.md。
