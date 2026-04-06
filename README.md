---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Email Triage OpenEnv

## Description
A real-world OpenEnv environment where an AI agent triages incoming emails by classifying them as **urgent**, **normal**, **spam**, or **delete**, and optionally drafting replies to urgent messages.

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
| Task | Difficulty | Steps |
|---|---|---|
| easy | Easy | 3 |
| medium | Medium | 5 |
| hard | Hard | 8 |

## Setup
```bash
pip install -r requirements.txt
python inference.py