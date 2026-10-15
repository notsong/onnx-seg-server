# ONNX 接入约定

平台通用性的边界是输入输出张量约定。满足以下约定的 UNet 或其他语义分割网络，服务无需知道其网络名称。

| 项目 | 要求 |
|---|---|
| 输入数量 | 恰好一个张量 |
| 输入类型/布局 | float32，RGB，[1,3,H,W]；默认 RGB/255，可配置额外 mean/std |
| 输入尺寸 | 使用配置中的 [H,W]；静态尺寸必须匹配 |
| 输出数量 | 至少一个输出；必须恰好一个符合下述语义分数布局，允许其他辅助输出 |
| 输出类型/布局 | float32，[1,C,h,w]，C 等于 num_classes，h/w 为正 |
| 输出内容 | 每个像素每个类别的可比较分数，顺序与 colors 一致 |
| 二分类 | C=2，不能仅导出单通道 sigmoid 或类别编号图 |
| 推理设备 | ONNX Runtime CPU 能支持图中所有算子 |

允许动态输入/输出尺寸，但每次仍按配置窗口大小运行；启动时会实际预热并校验。输出空间分辨率可与输入不同，服务恢复分数尺寸后再 argmax。相同最大分数出现并列时，使用较小的类别索引。输出含 NaN/Inf 会被拒绝。

多输出模型按 float32、四维、batch=1（或动态）、通道数等于 num_classes（或动态）筛选语义分数，不依赖输出排列顺序。没有匹配输出或存在多个匹配输出时启动失败。辅助单通道 boundary 可以保留，服务生成 mask 时只使用语义分数。

服务执行 RGB/255，可按配置执行 RGB 通道的 (x-mean)/std。resize_mode=stretch 直接缩放；letterbox 保持宽高比，以整数截断计算尺寸并居中补 [128,128,128] 灰边。输出先恢复到输入空间并去掉灰边，再恢复原图类别分数、argmax。未填写 resize_mode、mean、std 时保持 stretch 和 RGB/255。

mean/std 必须同时填写两个三元素数字数组；std 为正，所有值有限。如果 ONNX 内已包含归一化，配置应省略 mean/std，避免重复处理。模型特有 sigmoid、softmax、阈值、提示词或查询融合仍需在导出图中处理。

## 导出侧承担的处理

模型需要的 mean/std 可放在配置或导出图中；分割 head 和最终类别分数应合并在导出图中。下面是将归一化包进 PyTorch 导出包装的结构示意；network 的返回值应已经是语义类别分数，具体取得哪个字段要按你的网络实现调整。

```python
import torch

class ExportWrapper(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image_rgb_01):
        x = (image_rgb_01 - self.mean) / self.std
        scores = self.network(x)  # 按实际网络选出 [1,C,h,w]
        return scores

wrapper = ExportWrapper(network).eval().cpu()
sample = torch.zeros(1, 3, 512, 512)
# 用你当前 PyTorch 支持的 ONNX 导出方式导出 wrapper，
# 一个输入 image、一个输出 scores，例如选择 opset 17。
```

这里的 mean/std 只是常见示意值，请替换成训练使用的值。ONNX 本身包含计算图与权重，但不会自动携带 HTTP 字段、窗口步长和颜色配置。

- UNet：多类别 logits 输出通常可以直接接入，检查布局和归一化即可。
- 单通道二分类：若原模型输出 logit z，标准 sigmoid(z)>0.5 的判定可表示为两个分数 [0,z]。若训练阈值不同，应在导出包装中实现对应转换；不要用输出类别标签代替分数。
- DINO：若输出是特征/token，需要包含语义分割 head，导出最终 [1,C,h,w]。
- Mask2Former：通常存在 query 类别分数和 mask 分数，需要在导出包装中组合成最终语义分数并去掉 no-object 通道。将你的 Python 实现所使用的语义融合方式包入图中。
- YOLO 检测/实例分割原始输出不符合本版契约。只有已在导出侧转换成上述语义分数的模型才能直接接入。

滑窗会平均不同窗口的输出分数，所以必须保持每个窗口的分数含义一致。平台不自动对 logits/probability 做互换，需与原 Python 流程保持一致。

## 接入检查步骤

1. 在 Python 端按与服务一致的 RGB/255 输入测试导出的 ONNX，核对输入输出名称、数量、类型、布局和类别顺序。
2. 放到 models，填写配置，并用一次服务启动验证 CPU 算子支持、输入形状和输出形状。检查预热成功日志。
3. 用同一张图片对比 Python 与 C++ 的 mask，优先对齐缩放方式、颜色顺序、归一化和滑窗分数平均方式。
4. 记录真实图片的 decode/preprocess/inference/postprocess/encode/total。模型首次加载和黑图预热时间不计入 HTTP 推理耗时。

大型图像滑窗需要保存 C 份原图大小 float32 分数；仅此部分约为 4*C*H*W 字节。类别多或图像大时，通过 --max-work-mb 调整工作缓冲预算，并结合实际进程内存评估。切图降低单次模型输入尺寸，但不保证拼接缓冲更小。
