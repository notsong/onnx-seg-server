# 1001_004_JLD 真实图片测试

日期：2026-10-10。模型来自用户指定的 D:\work\Metis\Metis\bin\Debug\plugins\ONNX\1001_004_JLD.onnx，模型文件与接入副本的 SHA256 一致。

## 结论

该 ONNX 可以直接接入第一版服务，服务 C++ 源码和 ONNX 计算图均无需修改。已经通过实际 multipart HTTP 上传原始 JPEG，两种模式均返回 HTTP 200 的 RGB mask PNG，尺寸均为 2448×2048；输出像素仅包含配置中的黑色、白色。

新增 models/1001_004_JLD.onnx 和 config/1001_004.json。配置编号为 catagory=1001、item=4；本次对比测试时启用滑窗；按用户之后确认，当前项目配置已改为整图缩放（cut_flag=0）。原始输入图片另存为 tests/inputs/JLD_input.jpg。

## 测试设置

- CPU：Intel Core 7 150U，Windows x64。
- ONNX Runtime CPU 1.22.0，推理设置 4 个内部线程，HTTP 请求和窗口串行执行。
- 模型：约 95 MiB，PyTorch 1.12.1 导出，opset 12。
- 输入：images，float32 [1,3,1024,1024]。
- 输出：output，float32 [1,2,1024,1024]，类别分数。
- 预处理：RGB/255，无额外 mean/std。整图直接缩放；滑窗保持原图尺度。
- 调色板：类别 0=[0,0,0]，类别 1=[255,255,255]，沿用用户先前的黑白配置。
- 每种模式在模型加载和黑图预热后测量一次实际图片请求。数值是本次记录，不是多次稳定基准。

## 实测耗时

| 模式 | 窗口数 | 模型推理合计 | 服务端处理合计 | 类别 1 像素比例 |
|---|---:|---:|---:|---:|
| 整图缩放到 1024×1024 | 1 | 22.886 秒 | 23.659 秒 | 28.0017% |
| 1024×1024 滑窗，步长 768 | 9 | 169.685 秒 | 170.663 秒 | 26.0410% |

服务端处理包含解码、预处理、模型运行、分数恢复/拼接/绘色和 PNG 编码，不包含模型加载与预热、排队或网络传输。两次进程启动到就绪分别用了 19.647 秒、11.712 秒。

| 阶段 | 整图（毫秒） | 滑窗（毫秒） |
|---|---:|---:|
| 解码 | 222.471 | 110.653 |
| 预处理 | 51.611 | 127.248 |
| ONNX 推理 | 22886.216 | 169684.961 |
| 分数恢复/拼接/上色 | 62.532 | 279.658 |
| PNG 编码 | 423.034 | 424.112 |

本次主要耗时在模型运行。滑窗约 2 分 51 秒，因此调用方等待时间应覆盖这个耗时；下面的调用示例设置了 600 秒。

## 可视结果

两种模式都给出了主要晶界区域。滑窗结果线条更细，局部内部条纹也进入了前景。两张 mask 有 6.82022% 的像素类别不同，这是模式差异，不是误差率；整图改变了原图的空间尺度，两者本来就不保证逐像素一致。

没有使用形态学闭运算、轮廓筛选或额外阈值处理。原图右下角标尺也未裁掉。未提供标注，也未与原 Metis 的 OnnxInfer.dll 输出对比，当前不能量化准确率或宣称完全复现原软件。

## 产物与复现

完整结果位于：

tests/generated/jld-20261010-085211-24b27c/

- JLD_mask_whole.png：完整整图 mask。
- JLD_mask_tiled.png：完整滑窗 mask。
- JLD_comparison.png：原图、两种 mask 和同一区域的局部放大。
- report.json：模型与图片 SHA256、尺寸、配置、像素比例和耗时。
- server_whole.log / server_tiled.log：实际 C++ 服务输出。
- headers_whole.txt / headers_tiled.txt：包含大文件上传的 100 Continue 中间响应和最终 200 响应。

重新执行两种模式：

```powershell
Set-Location D:\GitHubProject\onnx-seg-server
python .\tests\test_jld_model.py --image .\tests\inputs\JLD_input.jpg --threads 4
```

正常启动服务：

```powershell
powershell -ExecutionPolicy Bypass -File D:\GitHubProject\onnx-seg-server\scripts\run.ps1 -Threads 4
```

看到 Listening 后，在另一个终端上传：

```powershell
Set-Location D:\GitHubProject\onnx-seg-server
curl.exe --fail-with-body --max-time 600 -F "file=@tests/inputs/JLD_input.jpg" -F "catagory=1001" -F "item=4" http://127.0.0.1:9600/infer --output JLD_mask.png
```

整图使用 cut_flag=0，滑窗使用 cut_flag=1；修改 config/1001_004.json 后重启。两种对比测试使用独立临时配置；当前正常启动的默认配置已经改为整图缩放。

测试过程中扩展了测试客户端的等待时间和线程参数，并修正大文件上传中间响应的解析。原有 40 项自动检查重新运行全部通过（tests/generated/run-6de95cc374/report.json）。测试创建的 C++ 服务进程已关闭。

## 后续启动交付

已提供项目根目录 onnx-seg-server.exe，可直接双击。模型位于 models，JSON 位于 config。exe-startup-4a509d346c 验证了无参数启动、不同工作目录和中文便携目录；真实模型整图结果与此前 mask 逐像素一致。
