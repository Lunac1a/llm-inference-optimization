# 0.1.3：mixed 默认服务与评测入口

2026-09-19。停止新增优化，交付已验证实现并扩大独立 benchmark。

- `inference-api serve` 默认运行 `chunked-hybrid`：历史 FP8、当前 BF16、2048 分块、4 GiB KV、16640 上下文、BLOCK_M=64。
- 单循环来源分支 kernel 与 2026-09-18 成功候选逐字节相同；服务适配器固定启用该分支，不再依赖研究环境变量。
- 失败的三段循环候选从活跃源码移出，原始候选、失败结果及整理前工作区副本完整保留。未修改上游 vLLM 安装。
- `document` 和 `hybrid` 作为显式比较入口保留。原生 FP8 由 benchmark 配置，不作为默认产品入口。
- 版本从 0.1.2 更新至 0.1.3。HTTP 请求格式保持兼容；默认 KV 精度、上下文与资源预算发生变化，既有用户可显式使用 `--profile document` 复现 BF16 配置。

## 仓库导航与证据

| 内容 | 入口 |
|---|---|
| 服务安装、调用、架构 | [安装](installation.md)、[API](api.md)、[架构](architecture.md) |
| 本轮扩大评测 | [完整结果与限制](final-benchmark-results.md)、[冻结协议](final-benchmark-protocol.md)、[执行说明](../benchmarks/evaluation/README.md) |
| 已集成 query tile 优化 | [历史数值与服务结果](attention-opt.md) |
| 首轮 780 响应 | [RULER 结论](ruler-findings.md) |
| 单循环来源分支候选 | [历史成功实验](mixed-branch-results.md) |
| 三段循环失败候选 | [失败分析](mixed-split-results.md) |
| 较早非 mixed 研究 | [归档索引](archive.md) |

历史协议、结果和执行脚本保留原位置及原文，不改写历史结论。它们记录当时的默认配置；当前行为以本页和服务文档为准。旧脚本含固定目录、固定版本及一次性运行限制，不应直接用于新评测。

整理前 65 个源码、测试、文档、工具和配置文件已复制并逐文件 SHA-256 校验至 `.local/archives/2026-09-19-release-preparation/`。已有未提交工作未被丢弃。新的原始评测证据单独保存在 `.local/research/ruler-final-2026-09-19/`；精简报告放入版本控制目录。

本页不把旧性能结果重新认定为新版本结果。本轮集成检查和 benchmark 状态在最终结果中分别报告。

## 整理与集成检查

- 源码17项回归、干净安装包17项回归全部通过。
- wheel内16个源码/数据文件与工作区逐字节一致；已移除的预算/CPU-KV模块未进入包。
- 不安装Torch/vLLM的API环境可正常导入，版本0.1.3、默认`chunked-hybrid`均核验通过。
- GPU集成后的20个固定数值用例通过；未重新搜索参数或运行新的优化实验。
- 扩大benchmark已完成并核验：2340/2340正式响应、54/54延迟测量，累计149.75分钟（含失败与续跑），未超过三小时预算。模型、端口及临时防休眠均已释放。
- 总体RULER分数mixed 92.868、BF16 Flash 93.103，相对下降0.252%；同等KV预算下容量1.94倍。16K质量相对BF16下降1.026%，测量时间存在跨段限制；详见[完整结果](final-benchmark-results.md)。[中断与暂停历史](final-benchmark-status.md)完整保留，没有重跑已完成响应。

机器检查记录位于整理归档目录的 `release-validation.json`、`installed-tests.log`，GPU数值记录位于新评测目录的 `numerical.json`。
