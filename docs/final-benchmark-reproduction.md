# 最终结果的离线核验与报告

本轮采集已完成，不应重新执行一次性GPU入口。数据位于`.local/research/ruler-final-2026-09-19/`，由原始运行、resume-1、resume-2及resume-3组成。原始数据在本地，不随Git发布；公开目录`benchmarks/results/final-2026-09-19/`保留逐题分数、答案差异、环境、冻结哈希、各次账本和汇总。

在原固定WSL环境和完整本地证据仍存在时，以下步骤只读取原始证据、评分和写入报告，不启动GPU：

```bash
/home/lunacia/.venvs/inference-vllm/bin/python benchmarks/evaluation/report_resumed3.py
/home/lunacia/.venvs/inference-vllm/bin/python benchmarks/evaluation/finalize_results.py
```

第一步验证原始、两次历史续跑及最后续跑的冻结哈希，逐项验证2340个响应与54条计时，合并部分FP8数据块的24+106个不同ID，并调用固定上游评分函数。第二步补充总体分数、主机事件记录、完成状态与可读结论。第二步不能代替第一步的原始证据核验。

评分与汇总不修改原始输入、响应、超时记录或暂停记录。16K第二块跨运行时段，报告中的耗时比例是观察值，不能据此宣称严格控制设备条件下的普遍加速。
