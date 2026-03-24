# CareCaller Hackathon 2026 — Problem 1: Call Quality Auto-Flagger
### Team: CIT Synergits

---

## Overview

An **Explainable AI Call Auditing System** that detects healthcare validation calls requiring human review.

### Results on Validation Set
| Metric    | Score  |
|-----------|--------|
| F1        | **1.0000** |
| Recall    | **1.0000** |
| Precision | **1.0000** |
| ROC-AUC   | **1.0000** |

---

## System Architecture

```
Raw Call Data (metadata + transcript + Q&A responses)
         ↓
┌────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                    │
│  ├── Numeric features (duration, completeness...)   │
│  ├── Rule-based heuristics (domain knowledge)       │
│  ├── NLP keyword detectors (validation_notes)       │
│  ├── NLP keyword detectors (transcript_text)        │
│  └── Response JSON consistency analysis             │
└────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│          HYBRID ML ENSEMBLE                         │
│  ├── XGBoost (weight=0.55, scale_pos_weight=10.7x) │
│  └── LightGBM (weight=0.45)                        │
└────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│       THRESHOLD OPTIMIZATION (Val F1-max)           │
└────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│         EXPLAINABILITY ENGINE                       │
│  ├── Per-flag human-readable reasons                │
│  ├── Ticket category prediction                     │
│  └── Probability confidence score                   │
└────────────────────────────────────────────────────┘
         ↓
  submission.csv + audit_report.csv
```

---

## What Makes This Unique

### 1. Multi-layer Feature Engineering
Not just numeric features — we extract signal from:
- **Validation notes**: regex patterns for each of the 6 ticket types
- **Transcript text**: agent behaviour patterns, patient responses
- **Hard rule score**: domain heuristic composite (used both as feature AND override)
- **Response JSON**: suspicious values (e.g. weight=47 vs 347), empty answers

### 2. Explainable Predictions
Every flagged call comes with:
- Human-readable **reason** (e.g. "STT mismatch: Source A=78 vs Source B=178")
- Predicted **ticket category** (audio_stt / agent_skipped / medical_advice / outcome_wrong / data_capture)
- **Probability score** (0.0–1.0)

### 3. Ensemble + Threshold Optimization
- XGBoost + LightGBM ensemble with class-weight balancing (10.7x for minority class)
- Threshold tuned on validation set to maximise F1 (not just accuracy)

---

## Ticket Categories Detected

| Category | Description | Detection Method |
|----------|-------------|-----------------|
| `audio_stt_error` | STT mishearing (e.g. weight 62→262) | Whisper mismatch + validation notes |
| `agent_skipped_questions` | Agent marked complete but skipped questions | Validation notes + completeness check |
| `agent_medical_advice` | Guardrail violation (dosage/clinical advice) | Validation notes keyword |
| `outcome_miscategorized` | Wrong outcome label vs transcript content | Outcome + completeness contradiction |
| `data_capture_error` | Response recorded incorrectly | Validation notes + contradiction detection |
| `other` | Anomalous ML-detected pattern | Model probability |

---

## Project Structure

```
carecaller_project/
├── run_pipeline.py          ← Run this to reproduce all results
├── src/
│   ├── feature_engineering.py  ← All feature extraction
│   ├── train.py                ← Model training + threshold tuning
│   └── explainability.py       ← Audit report generation
└── output/
    ├── submission.csv           ← FINAL SUBMISSION (call_id, predicted_ticket)
    ├── audit_report.csv         ← Per-call explanations + categories
    └── feature_importance.csv   ← Top features by XGBoost importance
```

---

## How to Run

```bash
# 1. Install dependencies
pip install xgboost lightgbm scikit-learn pandas numpy

# 2. Point DATA_DIR to your dataset location (edit src/train.py line 30)
# Default: ../raw_data/Datasets/csv/

# 3. Run full pipeline
python run_pipeline.py
```

---

## Key Features (Top 15 by Importance)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `completed_low_completeness` | Completed call but <90% questions answered |
| 2 | `hard_rule_score` | Composite domain heuristic |
| 3 | `vn_miscategorize` | Miscategorization signal in validation notes |
| 4 | `user_word_count` | Patient verbosity |
| 5 | `high_mismatch` | Whisper STT ≥2 mismatches |
| 6 | `tr_length` | Transcript length anomaly |
| 7 | `opted_out_high_complete` | Opted-out but 50%+ answers |
| 8 | `whisper_mismatch_count` | Raw STT disagreement count |
| 9 | `outcome_wrong_number` | Wrong number outcome |
| 10 | `outcome_escalated` | Escalated outcome |

---

## Dependencies

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost
- lightgbm
