"""
CareCaller Hackathon 2026 — Full Pipeline Runner
=================================================
Run this single file to:
1. Train the model
2. Generate submission.csv
3. Generate audit_report.csv (with explanations per flagged call)
4. Print evaluation summary

Usage:
    python run_pipeline.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Make sure src is on path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import build_features
from train import (
    load_data, prepare_datasets, get_class_weight,
    build_xgb, build_lgbm, optimize_threshold, hard_rule_score,
    predict_ensemble, OUT_DIR, DATA_DIR
)
from explainability import build_audit_report
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)


def run():
    print("=" * 60)
    print(" CareCaller Hackathon 2026 — Problem 1 Pipeline")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────
    df_train, df_val, df_test = load_data()

    # ── 2. Feature engineering ────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val = prepare_datasets(df_train, df_val, df_test)
    scale = get_class_weight(y_train)

    # ── 3. Train base models ──────────────────────────────────────────────
    print("\n[1/4] Training base models...")
    xgb  = build_xgb(scale)
    lgbm = build_lgbm(scale)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    models = [xgb, lgbm]
    mw = [0.55, 0.45]

    # ── 4. Validation + threshold tuning ─────────────────────────────────
    print("[2/4] Evaluating on validation set...")
    val_proba = np.column_stack([m.predict_proba(X_val)[:, 1] for m in models]) @ np.array(mw)
    best_thresh, _ = optimize_threshold(val_proba, y_val, metric='f1')
    val_preds = (val_proba >= best_thresh).astype(int)

    print(f"\n{'='*55}")
    print(f"  VALIDATION RESULTS  (threshold = {best_thresh:.2f})")
    print(f"{'='*55}")
    print(classification_report(y_val, val_preds, target_names=['No Ticket', 'Ticket']))
    print(f"  F1:        {f1_score(y_val, val_preds):.4f}")
    print(f"  Recall:    {recall_score(y_val, val_preds):.4f}")
    print(f"  Precision: {precision_score(y_val, val_preds, zero_division=0):.4f}")
    try:
        print(f"  ROC-AUC:   {roc_auc_score(y_val, val_proba):.4f}")
    except:
        pass

    # ── 5. Retrain on train+val for final prediction ──────────────────────
    print("\n[3/4] Retraining on full train+val data...")
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    X_all  = build_features(df_all)
    X_all['hard_rule_score'] = hard_rule_score(df_all).values
    X_all  = X_all.reindex(columns=X_train.columns, fill_value=0)
    y_all  = pd.concat([y_train, y_val], ignore_index=True)
    scale2 = get_class_weight(y_all)

    xgb_f  = build_xgb(scale2)
    lgbm_f = build_lgbm(scale2)
    xgb_f.fit(X_all, y_all)
    lgbm_f.fit(X_all, y_all)
    final_models = [xgb_f, lgbm_f]

    # ── 6. Predict on test set ─────────────────────────────────────────────
    print("[4/4] Generating test predictions...")
    test_proba, test_preds = predict_ensemble(final_models, X_test, best_thresh, mw)

    print(f"\n  Test: {test_preds.sum()} calls flagged out of {len(test_preds)}")
    print(f"  (Base rate ~8.6% → expected ~{int(0.086*len(test_preds))} tickets)")

    # ── 7. Save submission ────────────────────────────────────────────────
    submission = pd.DataFrame({
        'call_id':          df_test['call_id'],
        'predicted_ticket': test_preds.astype(bool),
    })
    sub_path = OUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)

    # ── 8. Build explainability audit report ─────────────────────────────
    audit = build_audit_report(df_test, test_proba, test_preds)
    audit_path = OUT_DIR / "audit_report.csv"
    audit.to_csv(audit_path, index=False)

    # ── 9. Print flagged calls summary ────────────────────────────────────
    flagged = audit[audit['predicted_ticket']].sort_values('ticket_probability', ascending=False)
    print(f"\n{'='*55}")
    print(f"  FLAGGED CALLS — AUDIT REPORT")
    print(f"{'='*55}")
    for _, row in flagged.iterrows():
        print(f"\n  Call ID: {row['call_id']}")
        print(f"  Outcome: {row['call_outcome']} | Prob: {row['ticket_probability']:.3f}")
        print(f"  Category: {row.get('predicted_category', 'N/A')}")
        print(f"  Reasons: {row.get('flag_reasons', 'N/A')}")

    # ── 10. Feature importance summary ───────────────────────────────────
    fi = pd.Series(xgb_f.feature_importances_, index=X_all.columns)
    fi_sorted = fi.sort_values(ascending=False)
    fi_path = OUT_DIR / "feature_importance.csv"
    fi_sorted.reset_index().rename(columns={'index': 'feature', 0: 'importance'}).to_csv(fi_path, index=False)

    print(f"\n{'='*55}")
    print("  OUTPUT FILES")
    print(f"{'='*55}")
    print(f"  ✅ submission.csv          → {sub_path}")
    print(f"  ✅ audit_report.csv        → {audit_path}")
    print(f"  ✅ feature_importance.csv  → {fi_path}")
    print(f"\n  submission.csv preview:")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    run()
