# CareCaller Hackathon 2026 — Data Dictionary

## Call Metadata
| Field | Type | Description |
|-------|------|-------------|
| call_id | UUID | Unique call identifier |
| outcome | string | Call result: completed, incomplete, opted_out, scheduled, escalated, wrong_number, voicemail |
| call_duration | int | Duration in seconds |
| attempt_number | int | Which attempt (1 = first, 2 = retry, etc.) |
| direction | string | outbound or inbound |
| attempted_at | datetime | When the call was placed (UTC) |
| scheduled_at | datetime | When the call was originally scheduled |
| whisper_status | string | STT verification status (completed, skipped) |
| whisper_mismatch_count | int | Number of disagreements between two STT systems |
| organization_id | string | Organization identifier |
| product_id | string | Product/medication identifier |
| patient_state | string | US state (2-letter code) |
| cycle_status | string | Case cycle status |
| form_submitted | bool | Whether the patient form was submitted |
| patient_name_anon | string | Synthetic patient name |

## Response Features
| Field | Type | Description |
|-------|------|-------------|
| question_count | int | Total health questions in questionnaire (14) |
| answered_count | int | How many questions got answers |
| response_completeness | float | Fraction answered (0.0 - 1.0) |

## Transcript Features
| Field | Type | Description |
|-------|------|-------------|
| turn_count | int | Total conversation turns |
| user_turn_count | int | Patient turns |
| agent_turn_count | int | Agent turns |
| user_word_count | int | Total patient words |
| agent_word_count | int | Total agent words |
| avg_user_turn_words | float | Average words per patient turn |
| avg_agent_turn_words | float | Average words per agent turn |
| interruption_count | int | Conversation interruptions |
| max_time_in_call | int | Maximum time in call (seconds) |
| hour_of_day | int | Hour call was placed (0-23) |
| day_of_week | string | Day of week |

## Text Fields
| Field | Type | Description |
|-------|------|-------------|
| transcript_text | string | Full conversation ([AGENT]: ... [USER]: ...) |
| validation_notes | string | Post-call AI validation analysis |
| responses_json | string | JSON array of {question, answer} pairs (14 items) |
| whisper_transcript | string | Raw STT transcript (no role markers) |

## Target Variables (Problem 1)
| Field | Type | Description |
|-------|------|-------------|
| has_ticket | bool | **PRIMARY TARGET** — was a review ticket raised? |
| ticket_has_reason | bool | Does the ticket have an explanation? |
| ticket_priority | string | urgent / high / normal |
| ticket_status | string | open / resolved |
| ticket_initial_notes | string | What was flagged (reason for the ticket) |
| ticket_resolution_notes | string | Resolution notes |

## Ticket Categories
| Field | Type | Description |
|-------|------|-------------|
| ticket_cat_audio_issue | string | Audio/STT issue: '' or 'issue' |
| ticket_cat_elevenlabs | string | Voice agent issue: '' or 'issue' |
| ticket_cat_openai | string | AI processing issue: '' or 'issue' |
| ticket_cat_supabase | string | Database issue: '' or 'issue' |
| ticket_cat_scheduler_aws | string | Scheduling issue: '' or 'issue' |
| ticket_cat_other | string | Other issue: '' or 'issue' |
| ticket_cat_*_notes | string | Notes for each category |
| ticket_raised_at | datetime | When ticket was created |
| ticket_resolved_at | datetime | When ticket was resolved |

## Notes
- **Test set labels are hidden** — `has_ticket` is null, all ticket fields are empty
- All patient names are **synthetic** (AI-generated, no real patients)
- Health data (weights, medications, side effects) is **fabricated**
- The 14 questionnaire questions follow the TrimRX medication refill check-in format
