# Beatrice v2 isolated probe

本目录是 Beatrice v2 的一次性研究与可运行性探针。它与正式音频链路完全隔离：不导入 `voice_change` 的生产模块，不修改 `AudioStream`、RVC、GUI 或打包配置，也不把 Beatrice 模型误识别为 RVC checkpoint。

## 结论

- 技术可行性：通过。官方 VCClient 2.2.2-beta 中的 Beatrice v2.0.0-rc.0 原生 Python Stable ABI 可被本项目 Python 3.11 环境加载，并完成真实 WAV 到 WAV 转换。
- 实时潜力：通过离线块级探针。运行时固定接收 16 kHz、160 samples（10 ms），返回 24 kHz、240 samples；三轮 16.74 秒音频的 wall RTF 为 0.07598，P99 块耗时为 1.266 ms，低于 10 ms deadline。
- 正式接入建议：**B：继续验证后再决定**。主要阻塞不是性能，而是原生推理库未公开源码、独立再分发授权不清晰，以及 JVS 示例模型禁止未经授权的商业用途。

这不是声卡回调、连续长时运行或听感验证，因此不能单凭本探针宣布正式实时链路已经完成。

## 固定的上游版本

查询日期：2026-08-08。

| 来源 | 分支提交 | 用途 |
| --- | --- | --- |
| `w-okada/voice-changer` | `f1caf8e7c39fd0d6866202be27bf142790191a51` (`master`) | 当前 VCClient 文档与 Beatrice 支持状态 |
| `fierce-cats/beatrice-trainer` | `f34836de014b86956096878aecb8d3b17feaaa0b` (`main`) | 官方训练代码、格式与采样率依据 |
| `wok000/vcclient_model` | `c46962577db8cf3fba2b3fc3526af98eb6611840` (`main`) | 官方 rc0 JVS 示例模型 |

当前 VCClient README 标记 v2.2.2-beta 支持 Beatrice v2.0.0-rc0，std/cuda/onnx edition 均列出 Beatrice 支持。公开仓库中的 `server/voice_changer/Beatrice/Beatrice.py` 仍是 `not implemented` 桩，因此实际 Beatrice 适配层与原生推理实现不能从公开源码完整审计；本探针依据官方发布物中的公开 Python API 和模型元数据做黑盒验证。

相关官方入口：

- https://github.com/w-okada/voice-changer
- https://github.com/w-okada/voice-changer/blob/master/server/voice_changer/Beatrice/Beatrice.py
- https://huggingface.co/fierce-cats/beatrice-trainer
- https://huggingface.co/wok000/vcclient_model/tree/main/beatrice_v2_rc0
- https://prj-beatrice.com/

## 许可证与再分发边界

必须将代码、运行时和模型分别判断：

1. `beatrice-trainer` 代码是 MIT；它的测试 Common Voice 子集声明为 CC0。
2. VCClient 公共仓库主体是 MIT，但 README 将 Beatrice v2 标为“独自/专有”许可；公共仓库许可不能自动覆盖发布包内的非公开 Beatrice 推理库。
3. 官方发布包的 bundled-license notice 表明其中包含非公开推理库，超出适用法律许可范围的使用需单独许可。当前探针只做本机研究，不复制到正式产品或 release。
4. rc0 JVS 模型的 TOML 明示模型训练使用 JVS corpus 与 JVS-MuSiC，须遵守两者条款，尤其禁止未经授权的商业使用。
5. 自训练模型能否发布还取决于训练数据、说话人授权和最终模型条款，不能仅凭 Trainer 的 MIT 许可推断。

因此：**当前原生运行时和 JVS 示例模型都不应进入 Git、安装包或 release。正式集成前需取得 Project Beatrice 对运行时嵌入/再分发的明确书面许可，并为计划使用的模型逐一审核数据与模型许可。**

## 实际实现路径

本次没有使用占位转换、RVC 兼容层或伪造结果。真实调用链为：

```text
mono WAV
  -> float32
  -> 必要时重采样到 16 kHz
  -> 连续 160-sample 状态块
  -> beatrice.load_beatrice("2.0.0-rc.0")
  -> v20rc0.SimpleBeatrice(...五个 bin 文件...)
  -> converter.convert(block)
  -> 连续 240-sample / 24 kHz 输出
  -> PCM16 WAV
```

官方 std_win 发布物提供 `beatrice/v20rc0.pyd` Stable-ABI 扩展。本机 Python 3.11 可直接导入；此路径不使用本环境的 Torch/CUDA/ONNX Runtime。Trainer 本身使用 PyTorch，但训练工具链和发布版推理工具链不是同一实现。VCClient 的 cuda/onnx edition 支持表也不能证明 Beatrice 模型本身是 ONNX 格式。

运行时 `.pyi` 与二进制存在两个值得锁定的 API 漂移点：二进制方法是 `get_max_formant_shift()`，不是 stub 中的 `get_max_formant_shift_index()`；`convert()` 实际返回 `(float32[240], int)`，而非仅数组。第二个整数未文档化，本探针只记录为 `unknown`，不臆测语义。

## 模型格式

模型不是单个 `.pth` 或 `.onnx`，而是一个 paraphernalia 目录：

```text
beatrice_paraphernalia_*.toml
phone_extractor.bin
pitch_estimator.bin
embedding_setter.bin
waveform_generator.bin
speaker_embeddings.bin
```

rc0 JVS 样例总大小 42,340,561 bytes，版本 `2.0.0-rc.0`，含 100 个目标说话人。可调参数包括 `target_speaker`、`formant_shift`、`pitch_shift_semitone`、源音高上下界和 VQ neighbors。运行时常量为：输入 16 kHz / 160 samples，输出 24 kHz / 240 samples，PHONE channels 128，pitch bins 448，codebook 512。

## 当前环境

- Python 3.11.15
- NumPy 1.26.4, SciPy 1.17.1, SoundFile 0.14.0, librosa 0.11.0
- Torch/Torchaudio 2.1.2+cu121；CUDA 可用，RTX 4060 Laptop，cuDNN 8801
- 未安装 ONNX / ONNX Runtime
- 未安装或升级任何包；基准资源统计也只使用标准库和 Windows API

环境快照是本地生成结果，应写入已忽略的 `results/`，不提交机器路径。

## 本地资产

`.gitignore` 排除了以下研究资产：

- `assets/runtime/`：从官方 VCClient std_win 2.2.2-beta 提取的 Beatrice 运行时
- `assets/model/`：官方 rc0 JVS 样例模型
- `results/`：机器相关 JSON
- `outputs/`：根仓库规则已忽略 WAV

下载件校验值：

- `vcclient_std_win_2.2.2-beta_only_beatrice.zip`: SHA256 `CA7DD6E1255277667AD6EF2128C1FBBFF0FAE3F19E96AA86C2F59CF75040A150`
- `beatrice_2.0.0-rc.0_20250824.zip`: SHA256 `48DAB9C4DE25C66FC21D8B54B6ADEC784B1F521800AC26CA45D0FE6BAA6D26A8`

固定输入 `assets/input/common_voice_ja_38833628_16k.wav` 来自 Trainer 官方测试资产，mono 16 kHz、5.58 秒；它同样只保留在本地，不纳入 Git。

## 运行命令

从仓库根目录执行：

```powershell
$python = 'python'
$probe = 'experiments\beatrice_probe'

& $python -u "$probe\inspect_runtime.py" `
  --runtime-root "$probe\assets\runtime" `
  --snapshot "$probe\results\environment_after.txt"

& $python -u "$probe\inspect_model.py" `
  "$probe\assets\model\jvs" `
  --runtime-root "$probe\assets\runtime" `
  --load-runtime `
  --json-out "$probe\results\model_inspection.json"

& $python -u "$probe\infer_wav.py" `
  --model "$probe\assets\model\jvs" `
  --runtime-root "$probe\assets\runtime" `
  --input "$probe\assets\input\common_voice_ja_38833628_16k.wav" `
  --output beatrice_jvs001.wav `
  --target-speaker 0 `
  --json-out "$probe\results\inference.json"

& $python -u "$probe\benchmark.py" `
  --model "$probe\assets\model\jvs" `
  --runtime-root "$probe\assets\runtime" `
  --input "$probe\assets\input\common_voice_ja_38833628_16k.wav" `
  --target-speaker 0 `
  --warmup-blocks 100 `
  --repeats 3 `
  --json-out "$probe\results\benchmark.json"
```

`infer_wav.py` 强制输出只能位于本探针的 `outputs/` 下，并在写入前后检查 NaN、Inf、全零、严重削波和时长偏差。

## 实测结果

真实 WAV 到 WAV：

- 输入：89,280 frames @ 16 kHz，5.58 s，peak 0.51236，RMS 0.07687
- 输出：133,920 frames @ 24 kHz，5.58 s，peak 0.55090，RMS 0.06629
- 结果：无 NaN/Inf、非全零、无严重削波、时长一致、WAV 成功回读
- 单次转换：558 blocks，wall 0.4669 s，RTF 0.08367，P95 1.153 ms，P99 1.394 ms

三轮基准（一次 1 秒静音 warmup 后）：

- 模型加载：0.0681 s
- 16.74 s 总音频：wall 1.2720 s，RTF 0.07598
- 1,674 blocks：mean 0.757 ms，P50 0.736 ms，P95 1.083 ms，P99 1.266 ms，max 2.225 ms
- 进程工作集：89.47 -> 160.95 MiB，增量 71.48 MiB
- 进程 CPU：约 98.3%（单核尺度），符合该 std runtime 的 CPU-oriented 行为
- GPU 只是前后快照，显存 1485 -> 1487 MiB，不能当作 Beatrice GPU 峰值或因果证据

这说明单个 10 ms 块的计算预算充足，但不包含设备 callback、双重重采样、线程调度、输出 ring buffer 和系统抖动。

## 与当前 RVC 结果的简单对照

仓库现存 `tests/output/rvc_realtime_benchmark/modelF_realtime_benchmark.json`（2026-08-01）记录了同机 RVC Model F：模型加载 4.520 s；325 ms chunk 平均 195.6 ms、P95 206.1 ms、RTF 0.602；500 ms chunk 平均 199.3 ms、P95 216.9 ms、RTF 0.399。

Beatrice 本探针为 10 ms 原生块、加载 0.068 s、P95 1.083 ms、RTF 0.076。它在块粒度和计算负载上明显更轻，但两组输入 WAV、采样率、模型、warmup 和测量脚本不同，不能把这个比值当作严格的质量或端到端延迟排名。RVC 数据还来自旧的 44.1 kHz 基准路径，只能作为现有历史参考。

## 正式接口映射草案

若许可证问题解决，建议单独新增 Beatrice adapter，不改 `AudioStream` 和 RVC 算法内部：

| 正式接口 | Beatrice 映射 |
| --- | --- |
| `load_model` | 解析 TOML 版本，校验五个 bin，按版本加载 native module，构造一个长期存活的 `SimpleBeatrice` |
| `unload` | 释放 converter/module 引用并清空输入输出缓冲；原生 API 没有显式 unload |
| `process_audio` | 保持有状态顺序，将 48 kHz 输入缓冲/重采样为 16 kHz 的 160-sample 块，再把 24 kHz 输出重采样回 48 kHz |
| `get_latency` | 至少报告 10 ms 算法块、输入聚合/输出排队和两次重采样；不能只报 native convert 时间 |
| `get_info` | 模型版本、说话人数、16/24 kHz、160/240 hop、配置范围、runtime/许可来源 |

当前全局链路权威采样率是 48 kHz，而 Beatrice 是 16 kHz 输入/24 kHz 输出。48 kHz 下 256-sample callback 只有约 5.33 ms，与 Beatrice 的 10 ms 原生块不对齐；adapter 必须维护输入累积与输出 ring buffer，不能每个 callback 新建 converter，也不能并行重排 160-sample 块。现有 backend-neutral worker 可以复用其生命周期/异步外壳，但 Beatrice 的状态、缓冲和重采样必须由专属 adapter 拥有。

## 下一步门槛

在正式接入前至少完成：

1. 获得原生 runtime 的嵌入、商业使用与再分发书面许可，并选用许可清晰的自训练模型。
2. 将 runtime 获取变成用户自备或经授权的可校验安装流程，不能从 VCClient 包私自复制进 release。
3. 对 `.pyd`/`.pyi` API 漂移做版本契约测试，明确 `convert()` 第二返回值。
4. 做 48 -> 16 -> 24 -> 48 kHz adapter 原型、连续 30 分钟压力测试、callback underflow/overflow 和端到端延迟测量。
5. 再做盲听、噪声/静音/爆音/不同音高输入、参数边界和设备实测。

在上述条件满足前，本目录保持实验性质，不进入正式 backend 注册、GUI 或发布流程。

See [STREAMING.md](STREAMING.md) for the 48 kHz / 256-sample callback adapter experiment.
