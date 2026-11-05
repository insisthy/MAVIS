# Knowledge-card examples / 知识卡片示例

`examples/knowledge_cards_examples.jsonl` contains five literature-based examples
of the knowledge-card format, including source identifiers, applicability conditions
and usage boundaries.

| Card ID | Topic | Original source identifier(s) |
| --- | --- | --- |
| `SYMPTOM_CHESTPAIN_001` | Structured assessment of chest pain and anginal equivalents | `SRC_GUIDE_AHA_ACC_CHEST_PAIN_2021` |
| `SYMPTOM_CHESTPAIN_002` | Limitations of inferring low risk from non-classic symptoms | `SRC_GUIDE_AHA_ACC_CHEST_PAIN_2021` |
| `CLIN_RISK_DIABETES_001` | Diabetes as cardiovascular risk context | `SRC_GUIDE_ADA_CVD_DIABETES_2026` |
| `CLIN_RISK_HTN_001` | Hypertension as cardiovascular risk context | `SRC_GUIDE_ACC_AHA_PRIMARY_PREVENTION_2019`; `SRC_GUIDE_ADA_CVD_DIABETES_2026` |
| `CLIN_LIKELIHOOD_001` | Integrated clinical likelihood assessment | `SRC_GUIDE_ESC_CCS_2024` |

The default `knowledge_cards` setting in `configs/runtime.json` points to this
example file. For an evaluation, set this field to the versioned JSONL corpus
used for that evaluation. Retrieved evidence depends on the configured corpus.
