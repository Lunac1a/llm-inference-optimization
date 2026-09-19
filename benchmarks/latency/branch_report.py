"""Generate the final report from recorded gates; never changes the experiment."""
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / '.local/research/mixed-branch-2026-09-18'


def read(name): return json.loads((ROOT / name).read_text())


def main():
    micro = read('micro.json'); decision = read('decision.json'); validation = read('validation.json')
    accepted = decision['accepted']
    lines = ['# Mixed 单循环整块分流：实验结果', '', '2026-09-18。', '',
        '**全部冻结门槛通过；候选仅在本轮限定负载下获得接受证据。**' if accepted else
        f'**候选未被接受，停止于 `{decision["stopped_at"]}`；不再追加分流变体。**', '',
        '候选位于独立研究目录，未替换产品源码或切换默认配置。两种失败/成功实现的源码和证据分开保存。', '',
        '## 改动与边界', '',
        '在原始单 tile 循环的读取区域判断整块归属：历史只读 FP8/scales，当前只读 BF16，跨界块保留掩码。分支后共用 dot、causal mask、online softmax 和 acc 更新。当前块 scale 向量设为1；没有改变量化方式、缓存写入、BLOCK_M=64、TILE_SIZE=32 或 Decode 路由。', '',
        '原始对照来自初步诊断的完整源码快照，不是上一轮变慢的三段静态展开。候选显式设置 `INFERENCE_MIXED_BLOCK_SOURCE_BRANCH=1`；关闭时也做了输出等同性验证。', '',
        '## 数值与 kernel', '',
        f'20个固定用例全部通过。相对原始输出最大误差 {max(r["candidate_original"]["max_abs"] for r in micro["numerics"]):.9f}，最坏RMS {max(r["candidate_original"]["rms"] for r in micro["numerics"]):.9f}；阈值分别为0.002/0.0002。',
        f'相对混合精度FP32参考，候选最坏max_abs {max(r["candidate_reference"]["max_abs"] for r in micro["numerics"]):.9f}、RMS {max(r["candidate_reference"]["rms"] for r in micro["numerics"]):.9f}，满足0.02/0.002。关闭开关20/20与原始逐元素相同；cache字节和当前K/V调用前后均不变。', '',
        '每个性能形状20对AB/BA交替测量；每次CUDA event包含5次调用取单次平均，再取20个样本的中位数。两组各热身5次，不含profiler。', '',
        '| 历史 / 当前 tokens | 原始 ms | 候选 ms | 延迟变化 |', '|---|---:|---:|---:|']
    for r in micro['benchmarks']:
        lines.append(f'| {r["case"][0]} / 2048 | {r["medians_ms"]["original"]:.3f} | {r["medians_ms"]["split"]:.3f} | {(r["ratio"]-1)*100:+.1f}% |')
    resources = micro['benchmarks'][0]['resources']
    lines += ['', f'三形状几何均值比值 {micro["gate"]["geomean_ratio"]:.4f}，最坏比值 {micro["gate"]["worst_ratio"]:.4f}；门槛≤0.97/≤1.03。', '',
        '| 编译资源（32Q/8KV、128维） | 原始 | 候选 |', '|---|---:|---:|']
    for name in ['registers', 'spills', 'shared_bytes', 'num_warps']:
        lines.append(f'| {name} | {resources["original"][name]} | {resources["split"][name]} |')
    ptx = {arm: (ROOT / f'{arm}-8192.ptx').read_text() for arm in ['original', 'split']}
    lines += ['', f'PTX静态指令文本中，原始/候选的 `mma.sync.aligned` 数量为 {ptx["original"].count("mma.sync.aligned")}/{ptx["split"].count("mma.sync.aligned")}，`ex2.approx` 数量为 {ptx["original"].count("ex2.approx")}/{ptx["split"].count("ex2.approx")}。这支持共享计算体未像上一版那样静态展开成多份；不等于运行时指令计数或硬件occupancy测量。', '',
        '候选PTX文本仍比原始更大，不能以源码或文件更短解释加速。降低的寄存器、共享内存和spill指标是编译资源事实，不能单独证明完整因果；本轮没有带宽/缓存命中率/硬件occupancy计数器。']
    if (ROOT / 'latency-gate.json').exists():
        latency = read('latency-gate.json')
        lines += ['', '## 完整模型延迟与容量', '',
            '固定Qwen3-4B BF16权重、vLLM0.23.0、RTX4090 Laptop、4GiB KV、2048分块、NHD、eager、单并发。每次reset prefix，输出1 token。实际输入3960/8061/16245 tokens。执行顺序A1/B1/B2/A2；每格为3次普通测量的首个非空SSE内容块延迟中位数。', '',
            '| 组别 | 4K 秒 | 8K 秒 | 16K 秒 |', '|---|---:|---:|---:|']
        for name in ['A1', 'B1', 'B2', 'A2']:
            r = latency['medians'][name]
            lines.append(f'| {name}（{"原始" if name.startswith("A") else "候选"}） | {r["4096"]:.3f} | {r["8192"]:.3f} | {r["16384"]:.3f} |')
        ratios = latency['ratios_16k']
        lines += ['', f'两对16K延迟变化分别为 {(ratios[0]-1)*100:+.1f}% / {(ratios[1]-1)*100:+.1f}%；4K/8K合并中位数变化 {(latency["ratios_scaling"]["4096"]-1)*100:+.1f}% / {(latency["ratios_scaling"]["8192"]-1)*100:+.1f}%。四次实际KV容量均为 **56480 tokens**，无前缀命中、无抢占。', '',
            '这些是固定单输入、热kernel/冷prefix的描述性结果。ABBA降低顺序偏差但不消除温度/功率/频率漂移；没有生产p95、Decode或多并发性能声明。', '',
            '### 单独采集的16K性能追踪', '',
            '| 组别 | Attention GPU ms | 矩阵乘法 GPU ms | KV写入 GPU ms |', '|---|---:|---:|---:|']
        trace = read('trace-analysis.json')
        for name in ['A1', 'B1', 'B2', 'A2']:
            c = trace[name]['categories']
            lines.append(f'| {name} | {c["attention"]["ms"]:.1f} | {c["matrix_multiply"]["ms"]:.1f} | {c["kv_write"]["ms"]:.1f} |')
        lines += ['', '按原始kernel事件求和，不将包含kernel的annotation重复相加。追踪请求不进入普通延迟统计，不把组间E2E相减当作某个算子的精确成本。']
    if (ROOT / 'quality-gate.json').exists():
        quality = read('quality-gate.json')
        q = quality['rows']
        regressed = [r['id'] for r in q if r['split_score'] < r['original_score']]
        improved = [r['id'] for r in q if r['split_score'] > r['original_score']]
        lines += ['', '## 独立质量', '',
            '保留序号5/6：16K全部13个RULER任务，加4K/8K的CWE，共30题/实现；此前正式只评0–4、pilot用10。采用官方输出长度、正常EOS、非思考模板、greedy与seed42及官方评分，未按答案挑题。', '',
            f'逐题得分下降 **{len(regressed)}**，上升 **{len(improved)}**，文本完全相同 **{sum(r["same_text"] for r in q)}/30**。质量门槛结果：**{"通过" if quality["passed"] else "未通过"}**。门槛同时检查没有新增空输出和新增长度截断。', '',
            '| 样本 | 原始得分 | 候选得分 | 文本相同 |', '|---|---:|---:|---|']
        for r in q:
            lines.append(f'| {r["id"]} | {r["original_score"]:.3f} | {r["split_score"]:.3f} | {r["same_text"]} |')
        lines += ['', '小样本通过不证明普遍无损；这些样本今后不能再作为未见独立验收集。完整生成文本、原始SSE、usage和metrics保留在A-quality/B-quality目录。']
    lines += ['', '## 完整性、预算和交付', '',
        f'- 冻结文件 {validation["frozen_files_verified"]} 个全部复核；20个GPU数值用例和候选环境下17项CPU回归检查通过。',
        f'- 普通延迟/热身/追踪共 {validation["latency_requests"]} 请求；质量正式 {validation["quality_requests"]} 请求（另有每组1次短热身）。',
        f'- 本轮GPU实验进程生命周期 {validation["this_gpu_seconds"]:.3f} 秒；本阶段累计 {validation["stage_gpu_seconds"]:.3f}/7200秒。没有重跑或修改门槛。',
        '- 结束核验端口8034空闲，无GPU compute进程。',
        '- 候选保留在独立candidate-source目录；本次未替换产品文件或打包安装，不宣称生产默认已获得上述加速。', '',
        '协议：[mixed-branch-protocol.md](mixed-branch-protocol.md)。原始产物：`.local/research/mixed-branch-2026-09-18/`。',
        '关键证据：`micro.json`、`*.ptx`、`latency-gate.json`、`quality-gate.json`（如执行）、`trace-analysis.json`、`validation.json`、各组launch/log/results/responses/metrics/traces、`frozen.json`与`ledger.jsonl`。']
    (WORKSPACE / 'docs/mixed-branch-results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'accepted': accepted, 'report': 'docs/mixed-branch-results.md'}))


if __name__ == '__main__': main()
