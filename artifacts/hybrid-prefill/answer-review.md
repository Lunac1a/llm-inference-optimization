# Source-aligned inspection of 27 normal-EOS answers

The assistant inspected all three retained quality waves for r1-bf16, r1-fp8
and r1-hybrid, against materials.json's unchanged doc-a/b/c text. Each wave asks
one question per document concurrently. All responses finished with stop.

| Document | Identifier | Owner | Official budget | Region | Risk |
| --- | --- | --- | --- | --- | --- |
| A | 云桥甲-17 | 林若安 | 480万元 | 墨尔本北区 | 供应商接口延迟 |
| B | 澄海乙-29 | 周明川 | 315万元 | 悉尼西区 | 节假日样本不足 |
| C | 星环丙-08 | 顾清禾 | 260万元 | 布里斯班南区 | 历史工单缺少标签 |

Source locations: the opening explicitly labels the document identifier and
region; the middle owner record supplies the owner; the risk record names the
risk; the final confirmation names the official budget. The documents explicitly
exclude the training-example budgets 820/560/410万元 from official facts.

All nine answers per arm contain every requested fact: identifier (wave 0),
owner AND budget (wave 1), region AND risk (wave 2). No omitted requested field,
other-document value, contradiction, invented fact or training-budget confusion
was observed. Short identifier phrasing and an introductory “文档编号是” are both
accepted. BF16 and hybrid A say “云桥甲-17。”; native FP8 says
“文档编号是云桥甲-17。” This is an equivalent answer, not a scoring discrepancy.

Result: 27/27 inspected answers correct, including 9/9 hybrid. No exact-string
scorer was used. This does not establish general QA quality, process reasoning,
dates, heterogeneous real documents or robustness beyond these source-aligned
facts. Fixed-128-token performance continuations are excluded from this review.
