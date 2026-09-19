# Mixed 延迟初步诊断协议（2026-09-18）

本轮在 GPU 启动前冻结；不修改产品实现，不进行候选优化或质量重测。

- 固定 Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`、vLLM 0.23.0、BF16 权重。沿用 RULER 四组配置：16640 上下文、2048 分块、8 slots、4 GiB KV、NHD、eager，实际并发为 1。
- 输入使用已冻结的 RULER `niah_single_1`、sample 0，4K/8K/16K 各一题；不依据答案挑选。记录实际 token 数及输入哈希，不修改原始文件。
- 固定输出 1 token（ignore_eos=true），temperature=0、seed=42，专注 Prefill + 首 token，不据此评价 Decode 或质量。TTFT 指首个非空 SSE 内容块到达时间。
- 组顺序 BF16 Triton → mixed → FP8 Triton → BF16 Flash。每组一次服务启动；各长度各热身一次并丢弃，每请求前清空前缀缓存。随后固定三轮长度顺序：16K/4K/8K、8K/16K/4K、4K/8K/16K，共 9 次普通测量。报告全部样本及中位数/范围，不追加重复。
- 普通测量时 profiler 未启动；结束后仅对同一 16K 输入采集一次 CPU/CUDA trace，各组分开存储。追踪时间不纳入普通基准；禁止把不同组的 E2E 相减解释为算子成本。
- 四组总计 12 热身 + 36 普通测量 + 4 追踪请求。本次初步实验进程生命周期上限 1800 秒，计入独立延迟诊断阶段的 7200 秒总上限；单组最多 450 秒，启动最多 180 秒，请求最多 120 秒。故障/OOM/空内容/输出非 1 token/缓存命中/抢占/内存可用量不足 2 GiB/swap 增加超过 256 MiB 即停止，不自动重跑。
- 记录实际 KV 容量、源码/输入/协议哈希、版本、GPU 状态、启动命令、原始 SSE、前后 metrics、trace、进程时间账本。只清理由本轮创建的进程。
- 先分析 GPU kernel 总时长、调用次数、GPU 活动区间及空隙、CPU 算子/启动记录。融合 GEMM 无可靠形状映射时不硬拆 QKV 与 MLP；trace 无法拆分 attention 内部的历史/当前读取与 softmax，也不提供带宽/缓存命中率证据。
- 本轮为单输入、单并发、热 kernel/冷 prefix 的定位实验；组间热状态与时序偏差仍存在，不做统计显著性或普遍性能声明。只有证据明确后才另外冻结一个优化与数值、容量、独立质量回归协议。

产物目录：`.local/research/mixed-latency-2026-09-18/`。入口：`benchmarks/latency/diagnose.py`。
