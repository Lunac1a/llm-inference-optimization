# Source-aligned answer inspection

Inspected the 18 retained normal-EOS responses in r1-bf16-c3/quality-{0,1,2}
and r1-fp8-c3/quality-{0,1,2}, against the corresponding document text in
materials.json. Each wave contains one question per independent document.
This is a small inspected fact check, not the old exact-match scorer or a
general long-context quality benchmark. Review performed by the assistant.

| Document | Identifier | Owner | Official budget | Region | Documented risk |
| --- | --- | --- | --- | --- | --- |
| A | 云桥甲-17 | 林若安 | 480万元 | 墨尔本北区 | 供应商接口延迟 |
| B | 澄海乙-29 | 周明川 | 315万元 | 悉尼西区 | 节假日样本不足 |
| C | 星环丙-08 | 顾清禾 | 260万元 | 布里斯班南区 | 历史工单缺少标签 |

Source alignment: the opening paragraph explicitly labels the document identifier
and region; the middle owner record names the owner; the risk record identifies
the risk; the final confirmation explicitly states the official budget. The
intervening training budgets 820/560/410万元 are explicitly not project budgets.

Both arms correctly returned each document's identifier in wave 0; owner AND
official budget in wave 1; region AND risk in wave 2. All 18 answers preserve
every requested fact, without another document's value, omission, contradiction,
or selecting the training budget. All finish with stop, not length.

FP8 A's identifier answer is “云桥甲-17。” while BF16 says
“文档编号是云桥甲-17。” Both are correct; the extra framing is immaterial.
The other retained responses use the same natural phrasing in both arms.
Nine inspected answers per arm pass; no post-hoc repair or rescoring of historical
Stage 3 materials/results is implied. Dates, process ordering and broad reasoning
were not tested in this experiment.
