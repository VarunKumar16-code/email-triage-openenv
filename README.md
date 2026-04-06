# Email Triage OpenEnv

## Description
A real-world OpenEnv environment where an AI agent triages incoming emails by classifying them as **urgent**, **normal**, **spam**, or **delete**, and optionally drafting replies to urgent messages.

## Motivation
Email overload is a real productivity problem. This environment trains agents to make accurate, context-aware triage decisions — a practical, high-value NLP task.

## Observation Space
| Field | Type | Description |
|---|---|---|
| email_id | string | Unique email identifier |
| subject | string | Email subject line |
| body | string | Email body text |
| sender | string | Sender email address |
| inbox_size | int | Remaining emails |
| current_step | int | Current step index |
| max_steps | int | Total steps in episode |

## Action Space
| Field | Type | Description |
|---|---|---|
| email_id | string | ID of email being acted on |
| label | string | One of: urgent, normal, spam, delete |
| reply | string (optional) | Draft reply text |

## Tasks
| Task | Difficulty | Steps | Description |
|---|---|---|---|
| easy | Easy | 3 | Clear-cut emails only |
| medium | Medium | 5 | Mixed inbox |
| hard | Hard | 8 | Full inbox + replies required |

## Reward Function
- **1.0** — Correct label
- **0.3** — Wrong label (non-critical)
- **0.1** — Spam labeled as urgent
- **0.0** — Urgent email labeled as spam/delete (critical miss)

## Setup
```bash
git clone https://github.com/yourusername/email-triage-openenv
cd email-triage-openenv
pip install -r requirements.txt