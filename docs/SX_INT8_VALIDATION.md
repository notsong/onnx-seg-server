# SX 完整 INT8 验证结果

结论：本次直接静态量化版本不适合替换原模型。主要矩阵乘法和卷积均已验证采用整数计算，但未满足 5 秒服务目标，且分割精度大幅下降。

## 实测结果

CPU：Intel Core 7 150U；输入固定 1024×1024；RGB、ImageNet mean/std、128 灰色 letterbox；输出保持 2448×2048，黑色对应 sx。所有计时均排除模型加载及预热。

采用同一个 C++ EXE、同一套 ONNX Runtime CPU 运行库，一次只保留一个模型服务；先 FP32 三张，再 INT8 三张。分块顺序仍可能受温度及后台负载影响。

| 指标 | FP32 | 完整 INT8 |
|---|---:|---:|
| HTTP 总耗时中位数 | 21.45 秒 | 16.30 秒 |
| 推理耗时中位数 | 20.93 秒 | 15.75 秒 |
| HTTP 总耗时范围 | 19.55～21.83 秒 | 13.99～16.66 秒 |
| SX 类 IoU 的图片均值 | 94.89% | 39.73% |
| SX 类 Dice 的图片均值 | 97.38% | 56.69% |
| 模型文件大小（十进制 MB） | 400.62 | 103.34 |

此前使用根目录普通 CPU DLL、同时保留两个预热服务交替测试时，FP32/INT8 HTTP 中位数分别为 23.48/15.16 秒；两种服务测试条件都没有达到 5 秒。

独立 Python ONNX Runtime CPU 调用，INT8 第二次推理为 5.97 秒。它与 C++ 服务耗时仍有明显差异，本轮未定位全部原因；服务性能以 HTTP 实测为准，不能用独立调用时间承诺服务耗时。

## 校准与精度

用户目录共有 20 张不同的 JPG。17 张没有同名 JSON 的图片用于校准；4_10、4_8、4_9 三张有 LabelMe 标注的图片用于独立验证。SHA256 核对校准与验证无重复；用户附图与 4_8.jpg 相同。

标注 sx 的模型通道映射在 INT8 评估前固定为通道 0：原 FP32 在附图上的通道 0 IoU 为 94.42%，通道 1 仅 0.72%。输出颜色未改变。这里只报告 sx 类 IoU 的图片均值，并非两个类别的 mIoU。三个标注样本只能用于初步判断。

| 图片 | FP32 IoU | INT8 IoU |
|---|---:|---:|
| 4_10.jpg | 95.39% | 37.82% |
| 4_8.jpg | 94.42% | 46.44% |
| 4_9.jpg | 94.86% | 34.93% |

![附图分割对比](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/SX_INT8_comparison.png)

## 完整 INT8 的覆盖证据

使用 ONNX Runtime 1.22.0 静态 QDQ：MinMax 校准、U8 激活、S8 常量权重、常量权重按通道量化。reduce_range 使常量权重使用缩减范围，以缓解 AVX2 饱和风险；实际存储及计算仍采用 8 位类型。所有原始 96 个 MatMul（含 QK 和注意力×V）和 29 个 Conv 的输入均进入量化。

CPU 优化图含 96 个 QLinearMatMul、29 个 QLinearConv；两次实际推理分别记录到 192 和 58 次 CPU 整数核执行。输入输出仍为 float32，LayerNorm、GELU 以及部分图像/张量运算保留浮点。这不是整个图每个算子都只用整数。

OpenVINO 2025.1.0 的同一份 QDQ 模型运行图：72 个全连接和 29 个卷积采用整数，但 24 个注意力 MatMul 的执行类型是 jit_gemm_f32。因此该 OpenVINO 路径不能视作完整注意力 INT8；原生 OpenVINO 测试约 12.55 秒，精度同样大幅下降。

相关背景参考：[ONNX Runtime 官方量化说明](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)、[NNCF 2.16 的敏感模式定义](https://github.com/openvinotoolkit/nncf/blob/v2.16.0/nncf/common/graph/patterns/patterns.py)。本次性能及精度结论来自本机测试。

## 单图诊断

把进入 12 个注意力×V 乘法的 Softmax 概率解除量化，4_8 的 IoU 从 46.44% 回升到 52.25%，仍明显低于 FP32 的 94.42%。该诊断版本保留部分浮点乘法，只用于定位精度问题。

这些概率的量化步长约 0.0035～0.0039，而 4101 个 token 的平均注意力概率约 0.000244；许多小概率值会被舍入为零。该处存在可证实的量化风险，但解除这一处仍未恢复精度，不能把全部损失都归因于它。

后续若继续当前模型，应评估按敏感算子保留浮点、精度感知量化或量化感知训练。对 5 秒目标还需要实测更小输入或更轻模型，不能仅凭本次失败断言所有 INT8 方案都不可行。

## 文件与复现

- 原模型未修改：D:/GitHubProject/onnx-seg-server/models/1003_010_SX.onnx
- 原模型 SHA256：e3f66db5ce8c46849be80794e48da4ea233c846f00098962da12c3eb5b2717b9
- 完整 INT8 测试模型：D:/GitHubProject/onnx-seg-server/tests/generated/sx-int8-20261010/1003_010_SX_INT8_QDQ.onnx
- 数据拆分与 SHA256：[dataset_manifest.json](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/dataset_manifest.json)
- 量化记录：[quantization_QDQ_report.json](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/quantization_QDQ_report.json)
- 整数核证据：[ORT_INT8_kernel_report.json](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/cpu-kernels/ORT_INT8_kernel_report.json)
- C++ 服务测试：[SX_INT8_HTTP_report.json](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/http-isolated/SX_INT8_HTTP_report.json)
- OpenVINO 执行图：[INT8_runtime.json](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/native/INT8_runtime.json)
- 单图诊断：[attention_probability_diagnostic.json](C:/Users/13539/Documents/Codex/2026-10-09/wo-x/outputs/SX_int8/attention_probability_diagnostic.json)

默认 SX 配置、现有 EXE 和原 ONNX 均未替换。测试使用临时端口，测试服务已关闭。

依赖隔离在项目测试目录。最终工具不再依赖 NNCF；未修改系统 Python 环境。依赖版本：Python 3.12、NumPy 2.2.6、ONNX 1.17.0、ONNX Runtime 1.22.0、OpenVINO 2025.1.0、OpenCV Python 4.11.0、Pillow 12.3.0。

在对应 Python 环境安装 scripts/requirements-int8.txt 后，可执行：

    python scripts/quantize_sx_onnx_int8.py --data "D:/dataset/shuangxiang/SG5C/上钢五厂双相" --output tests/generated/sx-int8-new

    python tests/inspect_sx_int8_cpu.py --model tests/generated/sx-int8-new/1003_010_SX_INT8_QDQ.onnx --image "D:/dataset/shuangxiang/SG5C/上钢五厂双相/4_8.jpg" --publish tests/generated/sx-int8-new/cpu-kernels

    python tests/test_sx_int8_http.py --model tests/generated/sx-int8-new/1003_010_SX_INT8_QDQ.onnx --data "D:/dataset/shuangxiang/SG5C/上钢五厂双相" --publish tests/generated/sx-int8-new/http --exe openvino/onnx-seg-server.exe

    python tests/test_sx_int8_native.py --model tests/generated/sx-int8-new/1003_010_SX_INT8_QDQ.onnx --data "D:/dataset/shuangxiang/SG5C/上钢五厂双相" --publish tests/generated/sx-int8-new/native

校准脚本调整统计节点次序，生成矩阵后立即统计 min/max，降低注意力矩阵的同时驻留量；仅影响校准的执行顺序。全部 17 张图片都完成校准，本次校准与量化耗时约 15.4 分钟。
