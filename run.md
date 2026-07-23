# run.md

# BRIXTA Mattress Product Intelligence
## Development Journal & Terminal Runbook

Author: Zaheer Abbas

---

# Goal

Convert the old Thermal Simulator repository into a production-ready Mattress Product Intelligence platform capable of:

- discovering mattress products
- crawling official manufacturer websites
- collecting brochures/images
- OCR
- GPT Vision analysis
- configuration inference
- exporting Excel reports

---

# Initial Cleanup

Repository originally contained:

- Thermal simulator
- Research code
- Old crawler
- Streamlit prototype

Goal:

Turn everything into one coherent package.

---

# Package Rename

Old package naming was inconsistent.

Instead of using

```
mattress-thermal-prototype
```

the package itself became

```
mattress_intelligence
```

keeping repository name independent from Python package name.

---

# Virtual Environment

Create

```bash
python -m venv .venv
```

Activate

macOS/Linux

```bash
source .venv/bin/activate
```

---

# Install

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create

```
.env
```

Typical variables

```
OPENAI_API_KEY=

FIRECRAWL_API_KEY=

REDIS_URL=redis://localhost:6379/0

OBJECT_STORE_PATH=data

CELERY_BROKER_URL=redis://localhost:6379/0

CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

# Redis

Start

```bash
redis-server
```

Verify

```bash
redis-cli ping
```

Expected

```
PONG
```

---

# Celery Worker

Start worker

```bash
celery \
-A mattress_intelligence.celery_app \
worker \
--pool=solo \
-l INFO
```

---

# Streamlit

Run

```bash
streamlit run app.py
```

---

# Architecture

Final architecture

```
Streamlit

↓

Celery

↓

Firecrawl

↓

OCR

↓

GPT Vision

↓

Inference

↓

Excel Export
```

---

# Firecrawl Preference

Changed search order

OLD

```
Jina

↓

Firecrawl
```

NEW

```
Firecrawl

↓

Jina fallback
```

Reason:

Firecrawl gives better crawling.

---

# Progress Updates

Added progress stages

```
initializing

discovering

crawling

extracting

assets

ocr

vision

inferencing

exporting

finished
```

---

# UI Simplification

Old UI

Lots of config boxes.

New UI

Only

Company Name

Website

Start Research

Everything else automated.

---

# Celery Debugging

Observed

```
Task received
```

Meaning

✓ Redis works

✓ Streamlit submits

✓ Worker receives task

Later

```
STATE: PROGRESS

stage=crawling
```

Meaning

Pipeline actually started.

---

# Wrong Diagnosis

Initially suspected

Task timeout.

Checked

```python
Settings().celery_task_time_limit_seconds
```

Output

```
Hard limit

7200

Soft limit

7140
```

Conclusion

Timeout theory was WRONG.

---

# Actual Error

Observed

```
<class 'celery.concurrency.solo.TaskPool'>
does not implement kill_job
```

Meaning

Celery attempted to terminate a running task.

Likely causes

- revoke
- worker shutdown
- Ctrl+C
- terminate request

NOT

Hard timeout.

---

# Pylance Errors Fixed

Problem

```
Unknown attribute title
```

Solution

Added helper

```
_source_label(...)
```

instead of directly chaining

```
source_by_id.get(...)
```

---

Second Pylance warning

```
RobotFileParser.url unknown
```

Fixed by avoiding direct attribute access.

---

XML issue

```
etree unknown import
```

Resolved by importing from

```
lxml
```

instead of stdlib assumptions.

---

# Git Cleanup

Added

```
.gitignore
```

Final ignores

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/

.venv/

dist/
build/
*.egg-info/

.env
.env.*
!.env.example

artifacts/

data/*
!data/.gitkeep
!data/README.md

examples/*
!examples/.gitkeep
!examples/README.md

outputs/*
!outputs/.gitkeep
!outputs/README.md

*.sqlite3
*.sqlite3-*

*.xlsx
*.csv

.DS_Store
```

---

Created

```
data/.gitkeep

outputs/.gitkeep

examples/.gitkeep
```

---

# Git Commands Used

Repository status

```bash
git status
```

Branch

```bash
git branch --show-current
```

Current commit

```bash
git rev-parse HEAD
```

Remote commit

```bash
git rev-parse origin/main
```

Tree

```bash
git ls-tree -r --name-only HEAD
```

Tracked HTML

```bash
git ls-files \
| grep -Ei '\.(html?|htm)$'
```

Tracked runtime folders

```bash
git ls-files \
| grep -Ei '(^|/)(artifacts|outputs|data)/'
```

HTML sizes

```bash
git ls-files -z \
| grep -zEi '\.(html?|htm)$' \
| xargs -0 du -h \
| sort -h
```

Push

```bash
git push origin main
```

Fetch

```bash
git fetch origin
```

---

# GitHub Language Investigation

Repository initially showed

```
99.6% HTML
```

Hypotheses

- tracked HTML
- crawl artifacts
- generated files

Checked

```
git ls-files
```

No HTML.

Checked repository tree.

No HTML.

Checked uploaded ZIP.

No HTML.

Finally queried GitHub API

```bash
curl https://api.github.com/repos/habibieebhy/ThermalSimulatorVansvardhanProject/languages
```

Result

```json
{
  "Python": 424447,
  "Shell": 1126,
  "Makefile": 903,
  "Dockerfile": 763
}
```

Repository page later refreshed.

Final language

```
Python
```

Issue resolved.

---

# Mistakes Made

❌ Assumed Celery timeout.

Reality

Termination happened for another reason.

---

❌ Initially blamed HTML files.

Reality

GitHub language page cache.

---

❌ Package naming inconsistent.

Solved.

---

❌ Old UI exposed unnecessary controls.

Simplified.

---

❌ Firecrawl secondary.

Made primary.

---

❌ Synchronous execution.

Migrated to Celery.

---

# Things That Worked

✓ Redis

✓ Celery queue

✓ Streamlit

✓ Firecrawl integration

✓ Progress updates

✓ OCR pipeline

✓ Vision pipeline

✓ Excel export framework

✓ Git cleanup

✓ Repository cleanup

✓ Language detection

---

# Current Project Status

Working

✓ Streamlit

✓ Celery

✓ Redis

✓ Firecrawl

✓ Object Storage

✓ OCR

✓ Vision

✓ Product Extraction

Remaining

• Investigate Celery task termination while using solo pool.

• Continue improving inference quality.

• Improve evidence graph.

• Improve live progress UI.

---

# Useful Commands

Run app

```bash
streamlit run app.py
```

Run worker

```bash
celery -A mattress_intelligence.celery_app worker --pool=solo -l INFO
```

Run Redis

```bash
redis-server
```

Tests

```bash
pytest
```

Install editable

```bash
pip install -e .
```

Freeze packages

```bash
pip freeze > requirements.txt
```

---

# End of Session

Repository language fixed.

Repository cleaned.

Pipeline architecture stabilized.

Ready for next development session.


```
TERMINAL 1:docker compose up redis minio minio-init    

TERMINAL 2: ( (.venv) ) zaheerabbas@Zaheers-MacBook-Air mattress-thermal-prototype % python -m celery \
  -A mattress_intelligence.celery_app:celery_app \
  worker \
  --loglevel=INFO \
  --pool=prefork \
  --concurrency=1 \
  --hostname=brixta@%h

TERMINAL 3:python -m streamlit run app.py
```