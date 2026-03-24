"""
CareCaller Hackathon 2026 — Training Pipeline
==============================================
Hybrid system: Rule Engine + Gradient Boosting Ensemble
Strategy:
  1. Rich feature engineering (rule + NLP + response consistency)
  2. XGBoost + LightGBM ensemble with class-weight balancing
  3. Threshold optimization on validation set for F1
  4. Final submission CSV generation
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Local import
import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import build_features

# ─── Paths ───────────────────────────────────────────────────────────────────
# Primary path matches repository tree (Datasets/csv), fallback to legacy raw_data location.
base_dir = Path(__file__).resolve().parents[1]  # repo root: voicehack
DATA_DIR = base_dir / "Datasets" / "csv"
if not DATA_DIR.exists():
    DATA_DIR = base_dir / "raw_data" / "Datasets" / "csv"

OUT_DIR = base_dir / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "hackathon_train.csv"
VAL_CSV = DATA_DIR / "hackathon_val.csv"
TEST_CSV = DATA_DIR / "hackathon_test.csv"


# ─── Load data ────────────────────────────────────────────────────────────────
def load_data():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    print(f"Train: {df_train.shape} | Tickets: {df_train['has_ticket'].sum()}")
    print(f"Val:   {df_val.shape}   | Tickets: {df_val['has_ticket'].sum()}")
    print(f"Test:  {df_test.shape}")
    return df_train, df_val, df_test


# ─── Hard rule-based pre-filter ──────────────────────────────────────────────

def hard_rule_score(df: pd.DataFrame) -> pd.Series:
    """
    Domain heuristics that strongly indicate a ticket.
    Returns a float score 0-1 per row.
    Used both as a feature AND as an override signal.
    """
    score = pd.Series(0.0, index=df.index)
    vn = df['validation_notes'].fillna('')
    tr = df['transcript_text'].fillna('')

    # STT errors in validation notes
    score += vn.str.contains(r'WHISPER VERIFICATION.*differ|Source [AB].*\d', regex=True, case=False).astype(float) * 0.4
    # Medical advice violation
    score += vn.str.contains(r'dosage|medical advice|guardrail|clinical', regex=True, case=False).astype(float) * 0.4
    # Outcome miscategorization
    score += vn.str.contains(r"outcome.*doesn|miscategor|wrong outcome", regex=True, case=False).astype(float) * 0.4
    # Skipped questions / fabricated
    score += vn.str.contains(r'not asked|fabricat|question.*skipped', regex=True, case=False).astype(float) * 0.4
    # Data contradiction
    score += vn.str.contains(r'contradict|erroneously|recorded.*incorrect', regex=True, case=False).astype(float) * 0.35
    # Whisper mismatch count
    score += (df['whisper_mismatch_count'] >= 1).astype(float) * 0.3
    # Completed but missing answers
    score += ((df['outcome'] == 'completed') & (df['answered_count'] < 14)).astype(float) * 0.25

    return score.clip(0, 1)


# ─── Build feature matrices ──────────────────────────────────────────────────

def prepare_datasets(df_train, df_val, df_test):
    print("\nBuilding feature matrices...")
    X_train = build_features(df_train)
    X_val   = build_features(df_val)
    X_test  = build_features(df_test)

    y_train = df_train['has_ticket'].astype(int)
    y_val   = df_val['has_ticket'].astype(int)

    # Add hard rule score as a feature
    X_train['hard_rule_score'] = hard_rule_score(df_train).values
    X_val['hard_rule_score']   = hard_rule_score(df_val).values
    X_test['hard_rule_score']  = hard_rule_score(df_test).values

    # Align columns (test may have NaN for label-dependent cols)
    all_cols = X_train.columns.tolist()
    X_val   = X_val.reindex(columns=all_cols, fill_value=0)
    X_test  = X_test.reindex(columns=all_cols, fill_value=0)

    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Class balance — Train pos: {y_train.sum()}/{len(y_train)}")
    return X_train, X_val, X_test, y_train, y_val


# ─── Models ──────────────────────────────────────────────────────────────────

def get_class_weight(y):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos


def build_xgb(scale):
    return XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=scale,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )


def build_lgbm(scale):
    return LGBMClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        verbosity=-1,
    )


# ─── Threshold optimization ──────────────────────────────────────────────────

def optimize_threshold(proba, y_true, metric='f1'):
    """Find the probability threshold that maximises F1 on validation set."""
    best_thresh = 0.5
    best_score  = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (proba >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score  = score
            best_thresh = t
    return best_thresh, best_score


# ─── Ensemble prediction ─────────────────────────────────────────────────────

def predict_ensemble(models, X, threshold, weights=None):
    probas = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    avg_proba = probas @ np.array(weights)
    return avg_proba, (avg_proba >= threshold).astype(int)


# ─── Main training loop ───────────────────────────────────────────────────────

def train_and_predict():
    df_train, df_val, df_test = load_data()
    X_train, X_val, X_test, y_train, y_val = prepare_datasets(df_train, df_val, df_test)

    scale = get_class_weight(y_train)
    print(f"\nClass weight (scale_pos_weight): {scale:.2f}")

    # ── Train on train set, tune on val ──────────────────────────────────
    print("\nTraining XGBoost...")
    xgb = build_xgb(scale)
    xgb.fit(X_train, y_train)

    print("Training LightGBM...")
    lgbm = build_lgbm(scale)
    lgbm.fit(X_train, y_train)

    models = [xgb, lgbm]
    model_weights = [0.55, 0.45]   # XGB typically stronger on small datasets

    # ── Validation ───────────────────────────────────────────────────────
    val_proba = np.column_stack([m.predict_proba(X_val)[:, 1] for m in models])
    val_avg   = val_proba @ np.array(model_weights)

    best_thresh, best_f1 = optimize_threshold(val_avg, y_val, metric='f1')
    val_preds = (val_avg >= best_thresh).astype(int)

    print(f"\n{'='*50}")
    print(f"Validation Results (threshold={best_thresh:.2f})")
    print(f"{'='*50}")
    print(classification_report(y_val, val_preds, target_names=['No Ticket', 'Ticket']))
    print(f"F1:        {f1_score(y_val, val_preds):.4f}")
    print(f"Recall:    {recall_score(y_val, val_preds):.4f}")
    print(f"Precision: {precision_score(y_val, val_preds, zero_division=0):.4f}")

    # ── Retrain on train+val for final predictions ────────────────────────
    print("\nRetraining on Train+Val combined for final predictions...")
    df_all  = pd.concat([df_train, df_val], ignore_index=True)
    X_all   = build_features(df_all)
    X_all['hard_rule_score'] = hard_rule_score(df_all).values
    X_all   = X_all.reindex(columns=X_train.columns, fill_value=0)
    y_all   = pd.concat([y_train, y_val], ignore_index=True)
    scale_all = get_class_weight(y_all)

    xgb_final  = build_xgb(scale_all)
    lgbm_final = build_lgbm(scale_all)
    xgb_final.fit(X_all, y_all)
    lgbm_final.fit(X_all, y_all)
    final_models = [xgb_final, lgbm_final]

    # ── Test predictions ─────────────────────────────────────────────────
    test_proba, test_preds = predict_ensemble(final_models, X_test, best_thresh, model_weights)

    print(f"\nTest predictions: {test_preds.sum()} flagged out of {len(test_preds)}")
    print(f"  Expected ~{int(0.086 * len(test_preds))} tickets at 8.6% base rate")

    # ── Build rich output (submission + metadata) ─────────────────────────
    submission = pd.DataFrame({
        'call_id':          df_test['call_id'],
        'predicted_ticket': test_preds.astype(bool),
    })
    submission_path = OUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ Submission saved → {submission_path}")
    print(submission.head(10))

    # ── Feature importance ────────────────────────────────────────────────
    fi = pd.Series(xgb_final.feature_importances_, index=X_all.columns)
    fi_sorted = fi.sort_values(ascending=False).head(25)
    fi_path = OUT_DIR / "feature_importance.csv"
    fi_sorted.reset_index().rename(columns={'index': 'feature', 0: 'importance'}).to_csv(fi_path, index=False)
    print(f"\nTop 15 features by XGBoost importance:")
    print(fi_sorted.head(15).to_string())

    # ── Probability details (for explainability) ──────────────────────────
    detail = df_test[['call_id', 'outcome', 'response_completeness', 'whisper_mismatch_count']].copy()
    detail['predicted_ticket']  = test_preds.astype(bool)
    detail['ticket_probability'] = np.round(test_proba, 4)
    detail['hard_rule_score']   = hard_rule_score(df_test).values
    detail_path = OUT_DIR / "predictions_detailed.csv"
    detail.to_csv(detail_path, index=False)
    print(f"\n✅ Detailed predictions saved → {detail_path}")

    return submission, val_avg, y_val, best_thresh


if __name__ == "__main__":
    train_and_predict()
