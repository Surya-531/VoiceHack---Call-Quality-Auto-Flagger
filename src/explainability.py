"""
CareCaller Hackathon 2026 — Explainability Module
===================================================
For each flagged call, generates a human-readable audit report with:
  - Ticket probability score
  - Predicted ticket category
  - Specific reasons for the flag
  - Supporting evidence from transcript/validation notes
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Ticket category classifier ─────────────────────────────────────────────

def predict_category(row: pd.Series) -> dict:
    """
    Rule-based ticket category predictor.
    Returns dict of {category: True/False}
    """
    vn = str(row.get('validation_notes', '') or '')
    tr = str(row.get('transcript_text', '') or '')
    cats = {}

    # STT / Audio issue
    cats['audio_stt_error'] = bool(re.search(
        r'(WHISPER VERIFICATION.*differ|Source [AB].*\d|STT error|speech.to.text|recorded.*incorrectly)',
        vn, re.IGNORECASE
    ) or row.get('whisper_mismatch_count', 0) >= 2)

    # AI/Agent issue (ElevenLabs voice, question skipping)
    cats['agent_skipped_questions'] = bool(re.search(
        r'(question.*not asked|fabricat|never asked|2\+ required questions|missed.*question)',
        vn, re.IGNORECASE
    ))

    # Medical advice violation (OpenAI guardrail)
    cats['agent_medical_advice'] = bool(re.search(
        r'(dosage.*guidance|medical advice|guardrail|clinical recommendation)',
        vn, re.IGNORECASE
    ))

    # Outcome miscategorization
    cats['outcome_miscategorized'] = bool(re.search(
        r"(outcome.*doesn|miscategor|wrong outcome|misclass)",
        vn, re.IGNORECASE
    ) or (
        row.get('outcome') in ['opted_out', 'wrong_number'] and
        row.get('response_completeness', 0) > 0.5
    ))

    # Data capture error
    cats['data_capture_error'] = bool(re.search(
        r'(contradict|erroneously|recorded.*wrong|response.*incorrect|fabricat)',
        vn, re.IGNORECASE
    ))

    # If nothing matched, mark as other
    if not any(cats.values()):
        cats['other'] = True
    else:
        cats['other'] = False

    return cats


def get_flag_reasons(row: pd.Series) -> list:
    """Produce a list of plain-English reasons for flagging a call."""
    reasons = []
    vn = str(row.get('validation_notes', '') or '')
    tr = str(row.get('transcript_text', '') or '')

    if row.get('whisper_mismatch_count', 0) >= 1:
        reasons.append(
            f"STT mismatch detected: {row['whisper_mismatch_count']} disagreement(s) between transcription systems"
        )

    if re.search(r'WHISPER VERIFICATION.*differ', vn, re.IGNORECASE):
        m = re.search(r'Source A: ([\w\d.]+).*?Source B: ([\w\d.]+)', vn)
        if m:
            reasons.append(f"Whisper verification found conflicting values: Source A={m.group(1)} vs Source B={m.group(2)}")
        else:
            reasons.append("Whisper verification found conflicting transcription values")

    if re.search(r'dosage|medical advice|guardrail|clinical', vn, re.IGNORECASE):
        reasons.append("Agent provided medical/dosage guidance — guardrail violation")

    if re.search(r'question.*not asked|fabricat|never asked', vn, re.IGNORECASE):
        reasons.append("Agent skipped required health questionnaire questions; answers appear fabricated")

    if re.search(r"outcome.*doesn|miscategor|wrong outcome", vn, re.IGNORECASE):
        reasons.append(f"Call outcome '{row.get('outcome')}' likely miscategorized based on transcript analysis")

    if re.search(r'contradict|erroneously|recorded.*wrong', vn, re.IGNORECASE):
        reasons.append("Recorded response contradicts what patient stated in the transcript")

    if row.get('outcome') == 'completed' and row.get('answered_count', 0) < 14:
        reasons.append(
            f"Call marked 'completed' but only {row.get('answered_count')}/14 questions answered"
        )

    if row.get('outcome') in ['opted_out', 'wrong_number'] and row.get('response_completeness', 0) > 0.5:
        reasons.append(
            f"Outcome '{row.get('outcome')}' but response_completeness={row.get('response_completeness'):.0%} — possible misclassification"
        )

    if not reasons:
        reasons.append("Anomalous call patterns detected by ML model (see probability score)")

    return reasons


def build_audit_report(df: pd.DataFrame, proba: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    """
    Build full audit report for all test calls.
    Flagged calls include explanations and category predictions.
    """
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        flagged = bool(preds[i])
        ticket_prob = float(proba[i])

        rec = {
            'call_id':           row['call_id'],
            'predicted_ticket':  flagged,
            'ticket_probability': round(ticket_prob, 4),
            'call_outcome':      row.get('outcome', ''),
            'response_completeness': row.get('response_completeness', 0),
            'whisper_mismatches':  row.get('whisper_mismatch_count', 0),
        }

        if flagged:
            reasons = get_flag_reasons(row)
            cats    = predict_category(row)
            rec['flag_reasons']          = ' | '.join(reasons)
            rec['predicted_category']    = ', '.join(k for k, v in cats.items() if v)
            rec['cat_audio_stt']         = cats.get('audio_stt_error', False)
            rec['cat_agent_skipped_q']   = cats.get('agent_skipped_questions', False)
            rec['cat_medical_advice']    = cats.get('agent_medical_advice', False)
            rec['cat_outcome_wrong']     = cats.get('outcome_miscategorized', False)
            rec['cat_data_capture']      = cats.get('data_capture_error', False)
        else:
            rec['flag_reasons']          = ''
            rec['predicted_category']    = ''

        records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # For standalone testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_engineering import build_features, hard_rule_score

    DATA_DIR = Path(__file__).parent.parent.parent / "raw_data" / "Datasets" / "csv"
    OUT_DIR  = Path(__file__).parent.parent / "output"

    df_test = pd.read_csv(DATA_DIR / "hackathon_test.csv")
    df_preds = pd.read_csv(OUT_DIR / "predictions_detailed.csv")

    proba = df_preds['ticket_probability'].values
    preds = df_preds['predicted_ticket'].astype(int).values

    report = build_audit_report(df_test, proba, preds)
    flagged = report[report['predicted_ticket']].sort_values('ticket_probability', ascending=False)
    print(f"\nFlagged calls: {len(flagged)}")
    print(flagged[['call_id', 'ticket_probability', 'predicted_category', 'flag_reasons']].to_string())

    report.to_csv(OUT_DIR / "audit_report.csv", index=False)
    print(f"\n✅ Audit report saved → {OUT_DIR / 'audit_report.csv'}")
