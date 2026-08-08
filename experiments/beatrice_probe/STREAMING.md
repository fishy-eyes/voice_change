# Beatrice v2 48 kHz streaming adapter probe

本文件记录 `48 kHz / 256 samples -> Beatrice v2 -> 48 kHz / 256 samples` 隔离实验。实验不导入或修改正式 `ai/`、`audio/`、`core/`、`effects/`、`gui/`、`main.py`。

## 结构

```text
48 kHz mono float32 / 256 samples
  -> stateful soxr QQ, 48 kHz -> 16 kHz
  -> input FIFO
  -> 严格顺序取 160 samples
  -> 长期存活的 SimpleBeatrice
  -> 240 samples @ 24 kHz
  -> stateful soxr QQ, 24 kHz -> 48 kHz
  -> output FIFO + 512-sample startup threshold
  -> 256 samples @ 48 kHz
```

实现文件：

- `streaming_adapter.py`：`BeatriceStreamingAdapter`、有界分块 FIFO、延迟和漂移统计。
- `streaming_probe.py`：真实 WAV callback 模拟、真实模型 10 分钟压力测试、生命周期验证。
- `test_streaming_adapter.py`：不依赖 proprietary runtime 的快速确定性测试。

## Resampler 选择

当前环境已经安装 `python-soxr 1.1.0`，其 `ResampleStream` 是真正有状态的连续重采样器，不需要修改环境。

默认使用 `QQ`：

- 48 -> 16 kHz 对连续 256-sample callback 产出 `85, 85, 86` 循环。
- 24 -> 48 kHz 首个 240-sample model block 产出 476 samples，之后每块 480。
- 输入端最大报告 delay：0.0417 ms。
- 输出端最大报告 delay：0.0833 ms。

`LQ/MQ/HQ/VHQ` 在此小块调用方式下会先缓存多个 callback，再批量产生 300/526/490/1100 等大小的输出；其中 HQ 输出端可累计约 1480 个 48 kHz delay samples。它们可以用于未来更高延迟的 Worker 实验，但不适合作为本轮最低延迟 callback 原型的默认值。

QQ 是 soxr 的低延迟快速模式，滤波质量低于 LQ/HQ。自动对齐比较结果很好，但最终是否存在金属感、周期噪声或高频损失必须人工试听，当前不能把 QQ 直接宣布为生产音质方案。

## 启动与 Buffer 策略

模型平均生产率和 callback 消费率完全相同，但 480-sample 输出以约每两个 callback 一次的 burst 到达。若得到首个 476 samples 后立刻输出，会在下一 callback 发生 starvation。

本原型采用：

- output FIFO 至少累计 512 samples 后才开始输出。
- 启动期返回完整零初始化 `float32[256]`。
- 实际在第 4 个 callback 开始返回转换音频。
- 固定启动 padding：768 samples，16.0 ms。
- 启动后不足时会安全补零并单独计为 underflow；实测没有发生。
- FIFO 上限 8192 samples，超过上限立即报错而不是静默丢音频。

真实短音频和 10 分钟测试的共同结果：

- input FIFO max：234 samples @ 16 kHz。
- output FIFO max：1020 samples @ 48 kHz。
- steady output FIFO P50：540 samples，即 11.25 ms。
- 10 分钟结束时 output FIFO：764 samples，是固定启动储备，不随时间增长。
- underflow after start：0。
- overflow：0。
- dropped samples：0。
- inserted silence：768 samples，全部来自明确的启动策略。

## Callback benchmark

完整 10 分钟真实模型测试：

- callback：112,500 次。
- deadline：5.333 ms。
- process P50：0.632 ms。
- process P95：1.202 ms。
- process P99：1.486 ms。
- process max：6.464 ms。
- deadline miss：2 次，约 0.00178%。

大多数 callback 只执行重采样、buffer 或输出；60,000 个 callback 执行一次 Beatrice convert。最大值说明偶发系统/推理抖动可以超过声卡 deadline，因此不建议把真实推理直接放入正式音频 callback。

## Beatrice benchmark

10 分钟、60,000 次真实 `converter.convert()`：

- inference P50：0.852 ms。
- inference P95：1.231 ms。
- inference P99：1.548 ms。
- inference max：6.320 ms。
- inference RTF：0.08731。
- adapter wall RTF：0.09673。
- 600 秒音频墙钟耗时：58.04 秒。

资源：

- 测试过程 RSS：162.54 -> 165.74 MiB，增加 3.20 MiB；其中包含统计数组，不能全部归因于 runtime。
- GPU/VRAM 只是系统快照：显存 1418 -> 1460 MiB，不能据此证明 CPU-oriented std runtime 使用 GPU。

## 延迟拆解

| 项目 | 实测/定义 |
| --- | ---: |
| callback block | 5.333 ms |
| Beatrice 原生输入帧 | 10.000 ms |
| callback 量化后的首帧输入可用时间 | 10.667 ms |
| input resampler 最大 delay | 0.0417 ms |
| Beatrice inference P50/P95/P99 | 0.852 / 1.231 / 1.548 ms |
| output resampler 最大 delay | 0.0833 ms |
| startup padding | 16.000 ms |
| 首个有效输出所在 callback 的数据可用时间 | 21.333 ms |
| steady output FIFO P50 | 11.250 ms |

已知部分估算的最小附加延迟约 16.98 ms。该值不包含 Beatrice native receptive-field 内部算法延迟、Worker queue、sounddevice/host API、声卡缓冲或系统输出链路，因此不能当作端到端实测。

## 长时间稳定性

真实 Beatrice 连续模拟 600 秒：

- input samples：28,800,000。
- output callback samples：28,800,000。
- Beatrice convert：60,000。
- 48 -> 16 kHz accounted samples：9,600,000，drift 0。
- 24 -> 48 kHz accounted samples：28,800,000，drift 0。
- time drift：0 ms。
- input FIFO final：0。
- output FIFO final：764，等于固定启动 reserve 扣除 soxr 内部 4-sample delay，不是累计漂移。
- underflow/overflow/drop：0/0/0。

没有发现“输入越来越多、输出越来越慢”或 buffer 随时间增长。

## 输出与人工 A/B

- 离线连续输出：`outputs/beatrice_jvs001.wav`，24 kHz。
- callback 模拟输出：`outputs/streaming_beatrice.wav`，48 kHz，开头含 16 ms 明确启动静音。

自动检查：shape/dtype 正确，无 NaN/Inf，非全零，无削波，保存并回读成功。streaming 文件较原始输入多 192 个 callback padding samples，即 4 ms。

把离线输出高质量重采样到 48 kHz 后，与 streaming 输出自动寻找最佳对齐：

- best lag：768 samples / 16.0 ms。
- aligned RMSE：0.001012。
- correlation：0.999884。

这只证明数值上高度接近，不能替代人工听评。请重点听边界点击、周期性电音、音高抖动、断字、10 ms 周期噪声、重采样金属感、开头静音和尾部截断。

## Reset / Close

原生 API 没有可确认的 converter reset 方法。实验策略：

- `reset()` 重新创建 converter，耗时约 80 ms。
- 同时重建两个 `ResampleStream`，清空 FIFO 和统计。
- 适用于模型切换、音频停止后重启和设备重新连接；不能在 audio callback 内执行。
- `close()` 清空 converter/module/resampler/buffer 引用。
- 重复 `close()` 安全；close 后 `process()` 明确拒绝。

## Worker 建议

正式方案推荐：

```text
Audio callback
  -> bounded input queue
  -> backend-neutral Worker
  -> BeatriceStreamingAdapter
  -> bounded output queue
  -> Audio callback
```

不推荐直接 callback 推理。虽然 P99 约 1.49 ms，但 10 分钟内仍出现 2 次大于 5.33 ms 的真实抖动。Worker 可隔离原生 runtime、模型 reset 和操作系统调度抖动；Beatrice adapter 仍必须独占 converter、resampler 和 FIFO，保持严格顺序。

## 技术结论与许可

纯技术评级：**A，可以进入正式 `BeatriceVoiceEngine` 开发阶段**，前提是采用 Worker + 专属 streaming adapter，并在正式接入前完成真实设备长时测试和人工 A/B。

许可结论不变：本地 `.pyd`、DLL、官方 JVS 模型、输入 WAV、输出 WAV 和 benchmark JSON 均被忽略，不能提交或随 release 重新分发。即使采用“用户自行提供 runtime 和模型”，仍需确认项目调用、商业使用、安装引导和兼容性校验是否获得 Project Beatrice 许可。

## 命令

```powershell
$python = 'python'
$probe = 'experiments\beatrice_probe'

& $python -u "$probe\test_streaming_adapter.py"

& $python -u "$probe\streaming_probe.py" `
  --model "$probe\assets\model\jvs" `
  --runtime-root "$probe\assets\runtime" `
  --input "$probe\assets\input\common_voice_ja_38833628_16k.wav" `
  --output streaming_beatrice.wav `
  --quality QQ `
  --startup-buffer-samples 512 `
  --stress-seconds 600 `
  --json-out "$probe\results\streaming_probe.json"
```

Dependency note: python-soxr 1.1.0 reports LGPL-2.1-or-later; any future packaged build must retain the required license notices and satisfy the LGPL terms.
