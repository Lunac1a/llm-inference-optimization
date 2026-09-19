# Mixed 注意力延迟优化：初步计划

状态：2026-09-18 完成初步定位及两种按来源读取候选。三段静态循环数值通过、micro 退化34%–47%，未采用；随后单循环整块分流通过数值、micro、两轮模型延迟、容量和30题独立质量门槛，16K首内容延迟降低10.1%/11.9%。成功候选保存在独立实验目录，产品默认尚未切换。本阶段累计使用 965.13 / 7200 秒 GPU 进程生命周期。

证据：[初步协议](mixed-latency-protocol.md)、[初步结果](mixed-latency-results.md)、[三段协议](mixed-split-protocol.md)、[三段结果](mixed-split-results.md)、[整块分流协议](mixed-branch-protocol.md)、[整块分流结果](mixed-branch-results.md)。

## 目标与边界

保留历史 FP8 KV / 当前 BF16 K/V 的分块 Prefill 语义和缓存容量，先定位延迟来源，再决定优化。当前 Decode 复用上游 FP8；缓存量化写入仍由上游完成。不要把 mixed 的速度收益简单等同于取消量化写入。

当前 RULER 16K 首字延迟中位数：BF16 Flash 4.091 秒、BF16 Triton 4.991 秒、mixed 5.431 秒、原生 FP8 9.051 秒。这是一轮端到端观察，不能用相减直接分解 kernel 成本。

## 顺序

1. 代码理解：启动注册 → FP8 缓存写入 → 请求路由 → 历史/当前两种输入 → 统一 softmax。区分上游实现和本地修改。
2. 冻结独立诊断协议：固定模型、输入、分块、精度、并发、冷热状态、输出长度、执行顺序和重复次数。主看 16K，4K/8K 作缩放对照；同后端 BF16/FP8/mixed 为主要归因对照，Flash 为性能目标。具体次数与停止条件在执行前确定，总阶段上限沿用 2 GPU 小时。
3. 性能定位：用时间线/算子记录区分 QKV 投影、量化写缓存、注意力 kernel、启动间隙及其他模型计算；单独记录无 profiler 的端到端基准。进一步检查路由层的逐序列调用和 GPU 元数据操作、历史 FP8 与 scale 读取、当前 BF16 读取、online softmax、tile 布局。无法取得硬件计数器时，不推断已测得带宽或缓存命中率。
4. 根据证据只选择一个改动：减少重复数据处理、调整 tile/执行布局或降低启动开销。没有明确瓶颈则报告未知，不扩大盲目调参。
5. 回归：对同一混合精度数学参考检查数值误差，再测固定工作负载延迟、实际缓存容量和独立质量样本（重点 CWE）。保留原 RULER 输入与证据，不把已见错题上的提升当作泛化证据。

## 输出与成功判断

先交付延迟构成和瓶颈证据，再制定具体优化验收门槛。目标是缩小与同后端 BF16 的差距，进一步比较 Flash；不预先承诺某个加速比例。只在延迟改善、容量保留、数值及质量检查均满足预定要求时接受候选，默认 baseline 不随实验改变。

## 阅读入口

- `src/inference_service/runtime.py`：配置与插件选择。
- `src/inference_service/plugins.py`：受 profile 限制的注册入口。
- `src/inference_service/backends/hybrid.py`、`routes.py`：继承上游缓存能力、构造历史/当前边界。
- `src/inference_service/backends/chunked_hybrid.py`：无历史 BF16、有历史多 token mixed、Decode 上游 FP8 三条路径。
- `src/inference_service/backends/dual_source_kernel.py`：按位置读取不同 K/V 来源，在一个 online softmax 中合并。
- `src/inference_service/backends/UPSTREAM_NOTICE.txt`：上游来源与本地修改范围。
