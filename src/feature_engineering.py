"""
CareCaller Hackathon 2026 — Feature Engineering Module
=======================================================
Builds a rich multi-source feature set from:
  - Call metadata (numeric)
  - Rule-based heuristics (domain knowledge)
  - NLP signal from validation_notes & transcript_text
  - Keyword/pattern detectors per ticket category
  - Response consistency analysis
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Rule-based heuristic signals ────────────────────────────────────────────

def rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-knowledge rules derived from EDA and the 6 ticket categories:
    1. Outcome miscategorization
    2. STT mishearing
    3. Agent skipped questions
    4. Wrong-number misclassification
    5. Agent gave medical advice
    6. Data capture errors
    """
    d = pd.DataFrame(index=df.index)

    # ── Outcome anomaly rules ────────────────────────────────────────────
    # Completed call but answerd_count < 14 (possible skipped questions)
    d['completed_incomplete'] = (
        (df['outcome'] == 'completed') & (df['answered_count'] < 14)
    ).astype(int)

    # Completed with very low completeness
    d['completed_low_completeness'] = (
        (df['outcome'] == 'completed') & (df['response_completeness'] < 0.9)
    ).astype(int)

    # opted_out / wrong_number with HIGH completeness (contradiction)
    d['opted_out_high_complete'] = (
        (df['outcome'].isin(['opted_out', 'wrong_number'])) &
        (df['response_completeness'] > 0.5)
    ).astype(int)

    # escalated but form submitted (unusual)
    d['escalated_form_submitted'] = (
        (df['outcome'] == 'escalated') & (df['form_submitted'] == True)
    ).astype(int)

    # ── STT / Whisper anomaly rules ──────────────────────────────────────
    d['has_mismatch'] = (df['whisper_mismatch_count'] > 0).astype(int)
    d['high_mismatch'] = (df['whisper_mismatch_count'] >= 2).astype(int)

    # ── Response completeness anomalies ──────────────────────────────────
    d['zero_answers_completed'] = (
        (df['outcome'] == 'completed') & (df['answered_count'] == 0)
    ).astype(int)

    # More questions asked than answered (incomplete call)
    d['questions_unanswered'] = (df['question_count'] - df['answered_count']).clip(lower=0)
    d['high_unanswered'] = (d['questions_unanswered'] >= 3).astype(int)

    # Duration vs completeness mismatch: long call but few answers
    d['long_call_few_answers'] = (
        (df['call_duration'] > 120) & (df['response_completeness'] < 0.3)
    ).astype(int)

    # Short call but marked completed
    d['short_completed'] = (
        (df['outcome'] == 'completed') & (df['call_duration'] < 60)
    ).astype(int)

    # ── Conversation structure anomalies ─────────────────────────────────
    # Agent spoke much more than patient (possible one-sided)
    ratio = df['agent_word_count'] / (df['user_word_count'] + 1)
    d['agent_dominance_ratio'] = ratio
    d['agent_dominated'] = (ratio > 5).astype(int)

    # Very few user turns for a completed call
    d['few_user_turns_completed'] = (
        (df['outcome'] == 'completed') & (df['user_turn_count'] < 5)
    ).astype(int)

    # High interruption count
    d['high_interruptions'] = (df['interruption_count'] > 3).astype(int)

    # ── Whisper status signals ───────────────────────────────────────────
    d['whisper_skipped'] = (df['whisper_status'] == 'skipped').astype(int)

    return d


# ─── NLP keyword detectors ───────────────────────────────────────────────────

# Validation notes patterns (very informative based on EDA)
VALIDATION_NOTE_PATTERNS = {
    'vn_stt_error':       r'(STT|speech.to.text|whisper.*differ|source [AB].*\d)',
    'vn_medical_advice':  r'(dosage|medical advice|guardrail|clinical recommendation|guidance)',
    'vn_miscategorize':   r'(miscategor|doesn.t match|wrong outcome|mismatch)',
    'vn_skipped_q':       r'(question.*not asked|skipped|never asked|fabricat)',
    'vn_contradict':      r'(contradict|differs|disagree|erroneously|incorrect)',
    'vn_whisper_issue':   r'WHISPER VERIFICATION.*differ',
    'vn_weight_error':    r'weight.*\d+.*\d+|recorded weight',
    'vn_outcome_issue':   r"outcome.*(doesn|wrong|incorrect|misclass)",
}

TRANSCRIPT_PATTERNS = {
    'tr_not_interested':  r'not interested|no thank you|please stop|remove me',
    'tr_wrong_person':    r"wrong number|wrong person|don't know|no one by that",
    'tr_medical_question':r'(what dose|how much should i take|is it safe|side effect.*serious|should i stop)',
    'tr_reschedule':      r'(call back|reschedule|not a good time|try later)',
    'tr_confirm_wrong':   r"(this is wrong|that.s not right|i said|no i said)",
    'tr_agent_advice':    r'\[AGENT\].*?(you should take|i recommend|try taking|increase|decrease|take \d)',
    'tr_escalate_signal': r'(talk to a doctor|speak with.*physician|emergency|urgent)',
}


def nlp_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract keyword/regex features from text columns."""
    d = pd.DataFrame(index=df.index)

    vn = df['validation_notes'].fillna('')
    tr = df['transcript_text'].fillna('')

    for feat, pattern in VALIDATION_NOTE_PATTERNS.items():
        d[feat] = vn.str.contains(pattern, case=False, regex=True).astype(int)

    for feat, pattern in TRANSCRIPT_PATTERNS.items():
        d[feat] = tr.str.contains(pattern, case=False, regex=True).astype(int)

    # Aggregate signal: total flags across keyword detectors
    d['total_vn_flags']  = d[[f for f in d.columns if f.startswith('vn_')]].sum(axis=1)
    d['total_tr_flags']  = d[[f for f in d.columns if f.startswith('tr_')]].sum(axis=1)
    d['total_kw_flags']  = d['total_vn_flags'] + d['total_tr_flags']

    # Text length features
    d['vn_length']        = vn.str.len()
    d['tr_length']        = tr.str.len()
    d['whisper_tr_length'] = df['whisper_transcript'].fillna('').str.len()

    return d


# ─── Response JSON consistency features ──────────────────────────────────────

NUMERIC_PATTERNS = re.compile(r'\b\d{1,4}(?:\.\d{1,2})?\b')

def parse_responses(responses_json: str) -> dict:
    """Parse responses_json and return analysis dict."""
    result = {
        'n_qa_pairs': 0,
        'n_empty_answers': 0,
        'has_weight': 0,
        'has_numeric': 0,
        'weight_value': np.nan,
        'suspicious_weight': 0,
    }
    try:
        data = json.loads(responses_json) if isinstance(responses_json, str) else []
        result['n_qa_pairs'] = len(data)
        empty = 0
        for pair in data:
            ans = str(pair.get('answer', '') or '')
            q   = str(pair.get('question', '') or '').lower()
            if ans.strip() in ('', 'null', 'none', 'n/a'):
                empty += 1
            if 'weight' in q:
                result['has_weight'] = 1
                nums = NUMERIC_PATTERNS.findall(ans)
                if nums:
                    try:
                        w = float(nums[0])
                        result['weight_value'] = w
                        # Suspiciously low weight for an adult (STT dropped a digit)
                        if w < 50 or w > 500:
                            result['suspicious_weight'] = 1
                    except:
                        pass
            if NUMERIC_PATTERNS.search(ans):
                result['has_numeric'] = 1
        result['n_empty_answers'] = empty
    except Exception:
        pass
    return result


def response_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse responses_json for each row."""
    parsed = df['responses_json'].apply(parse_responses).apply(pd.Series)
    parsed.index = df.index
    return parsed


# ─── Outcome encoding ─────────────────────────────────────────────────────────

OUTCOME_ORDER = {
    'completed': 0,
    'incomplete': 1,
    'opted_out':  2,
    'scheduled':  3,
    'escalated':  4,
    'wrong_number': 5,
    'voicemail':  6,
}

DAY_ORDER = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6,
}


def categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d['outcome_enc']       = df['outcome'].map(OUTCOME_ORDER).fillna(-1).astype(int)
    d['day_of_week_enc']   = df['day_of_week'].map(DAY_ORDER).fillna(-1).astype(int)
    d['whisper_status_enc'] = (df['whisper_status'] == 'completed').astype(int)
    d['form_submitted_enc'] = df['form_submitted'].astype(int)

    # One-hot outcome
    for outcome, code in OUTCOME_ORDER.items():
        d[f'outcome_{outcome}'] = (df['outcome'] == outcome).astype(int)

    return d


# ─── Full feature build ──────────────────────────────────────────────────────

NUMERIC_BASE_FEATURES = [
    'call_duration', 'attempt_number',
    'whisper_mismatch_count',
    'question_count', 'answered_count', 'response_completeness',
    'turn_count', 'user_turn_count', 'agent_turn_count',
    'user_word_count', 'agent_word_count',
    'avg_user_turn_words', 'avg_agent_turn_words',
    'interruption_count', 'max_time_in_call', 'hour_of_day',
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all feature groups into a single feature matrix."""
    parts = [
        df[NUMERIC_BASE_FEATURES].fillna(0),
        rule_features(df),
        nlp_keyword_features(df),
        response_features(df),
        categorical_features(df),
    ]
    X = pd.concat(parts, axis=1)
    X.index = df.index
    return X
