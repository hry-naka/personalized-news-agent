# Personalized News Agent

A Python-based automated news curation system that fetches articles from RSS feeds, analyzes them using the Gemini API, generates a personalized HTML news digest based on a user profile, and optionally stores evaluation data for later analysis.  
This project is designed for daily automated execution (cron) and includes a full evaluation pipeline for measuring how well the curated articles match the user’s interests.

---

# Features

### RSS & Article Processing
- Fetches news articles from multiple RSS channels defined in `config.json`
- Resolves redirect URLs to obtain the final article URL
- Normalizes article metadata for consistent downstream processing

### Gemini‑Based Analysis & Curation
- Uses Gemini 2.5 to:
  - Analyze the user profile
  - Select the most relevant articles
  - Generate a structured HTML news digest
  - Include counter‑view articles to encourage intellectual diversity

### Email Delivery
- Sends the curated HTML report via SMTP (Gmail, SendGrid, Postfix, etc.)

### Evaluation & Reproducibility
- Optional evaluation mode (`--eval`) stores:
  - The exact prompt used
  - The generated HTML
  - The raw RSS article list
  - Metadata for reproducibility
- Fully compatible with the standalone evaluation tools:
  - `eval-prompt.py` (embedding-based quantitative evaluation)
  - `create-laaj-prompt.py` (LLM-as-a-Judge qualitative evaluation)
  - `summarize-eval.py` (statistical aggregation)

### Automation‑Friendly
- Designed for daily automated execution via cron or systemd
- Deterministic within a single run; evaluation data helps track consistency

---

# Requirements

- Python 3.12+
- A valid Gemini API key
- An SMTP server (Gmail, SendGrid, local Postfix, etc.)
- macOS, Linux (Ubuntu recommended), or WSL2

---

# Installation

Clone the repository:
```
git clone https://github.com/hry-naka/personalized-news-agent.git
cd personalized-news-agent
```
Install dependencies:
```
pip install -r requirements.txt
```
---

# Configuration

### 1. `.env` file

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASS=your_smtp_password
TO_EMAIL=recipient@example.com
```
### 2. `config.json`

Defines RSS channels, output and debugging settings:
```json
{
  "rss_channels": [
    { "query": "technology", "count": 30 },
    { "query": "innovation", "count": 30 },
    { "query": "culture japan", "count": 30 }
  ],
  "num_output_articles": 12,
  "retry_wait_seconds": [300,600,900],
  "force_error_status_test": false,
  "retry_wait_seconds_debug": [5,10,15],
  "force_error_status_code": 503
}
```
### 3. `user_profile.txt`
Defines the user’s interests, preferences, and intellectual tendencies.
This profile is injected directly into the main prompt and strongly influences article selection.

### 4. `main_prompt.txt`
Defines the HTML output structure and selection rules.
The script replaces:

```
{user_profile}
{articles_text}
{num_output_articles}
```
---
# Usage
If you do not need evaluation data, simply run the agent:
```
python news-agent.py "Daily News Digest"
```
This sends a curated email with the subject:
```
【AI news Agent】: Daily News Digest insights report
```

To store evaluation data:
```
python news-agent.py "Debug Run" --eval
```

This sends curated email and creates:
```
eval-data/YYYYMMDDHHMM/
 ├── prompt.txt
 ├── report.html
 ├── articles.json
 └── meta.json
```

---
# Evaluation Pipeline (Optional)

The project includes a complete evaluation pipeline combining:
* Embedding-based quantitative evaluation
* LLM-as-a-Judge qualitative evaluation
* Statistical aggregation across multiple LLMs

This allows you to measure:
* How well the digest matches the user profile
* How consistent the agent is over time
* How different LLMs judge the quality of the generated digest

### 1. Agent-Side Quantitative Evaluation (`eval-prompt.py`)

This tool evaluates:
* Titles
* Summaries
* Selection reasons
* Full article blocks
* Counter-view articles (inverted scoring)

All scores are cosine similarities between:
* The full prompt text
* Each component of the generated HTML digest

Run evaluation:
```
python eval-prompt.py -i latest -m all -o eval-data/eval-prompt.csv
```

Output CSV columns:

| column | description |
| ------ | ------------|
|timestamp|Evaluation timestamp|
|mail_subject|Subject used when generating the digest|
|article_index|- for summary row, otherwise article index|
|title_score|Similarity between prompt and article title|
|summary_score|Similarity between prompt and article summary|
|reason_score|Similarity between prompt and selection reason|
|article_score|Similarity between prompt and full article block|
|is_counter|1 for counter-view articles|

---

### 2. LLM-as-a-Judge Evaluation (`create-laaj-prompt.py`)

Embedding similarity alone cannot evaluate:
* Article selection quality
* Summary clarity
* HTML correctness
* Reasoning consistency
* Overall alignment with user interests

For this, the project supports LLM-as-a-Judge evaluation.

#### 2.1 Generate judge prompt:
```
python create-laaj-prompt.py -i eval-data/YYYYMMDDHHMM/ -o judge-prompt.txt
```

This prompt includes:
* User profile
* Generated HTML digest
* Original RSS article list
* Evaluation criteria (7 metrics)

#### 2.2 Evaluate using multiple LLMs

Feed judge-prompt.txt into:
* Claude
* Gemini
* GPT
* Copilot

Each LLM outputs a CSV:
```csv
model,metric,score,reason
Claude,article_selection,3.5,"..."
Claude,summary_quality,3.5,"..."
```

Store these files under:
judge-results/YYYYMMDD/

---
### 3. Statistical Aggregation (`summarize-eval.py`)

This tool combines:
* Agent-side embedding scores
* Judge-side LLM scores

Run aggregation:
```
python summarize-eval.py \
    --input eval-data/eval-prompt.csv \
    --from 202607151549 \
    --to   202607161508 \
    --llm-as-a-judge \
    --judge-dir judge-results/20260718 \
    --output summary.csv
```
Output structure:
First block: agent metrics (mean, median, stdev, count)
Blank line
Second block: judge metrics (LLM-averaged mean, median, stdev, count)

Example:
```csv
digest_score,0.7707,0.7707,0.0137,2
title_score,0.4887,0.5060,0.0513,10
...

article_selection,4.10,3.8,0.64,3
summary_quality,3.96,3.5,0.73,3
...
```

This provides a unified quantitative + qualitative evaluation of the agent’s performance.

---

# Automation (Ubuntu / macOS)

`run-news.sh` runs news-agent.py with `--eval` and eval-prompt.py sequentially. If you want to run news-agent.py only, edit this `run-news.sh`.
The single argument passed to run-news.sh is forwarded to news-agent.py as the mail_subject.

Example cron entry:
```
0 7 * * * /path/to/run-news.sh "Morning Digest"
```

---

# Project Structure

```
personalized-news-agent/
 ├── news-agent.py
 ├── eval-prompt.py
 ├── create-laaj-prompt.py
 ├── summarize-eval.py
 ├── main_prompt.txt
 ├── user_profile.txt
 ├── config.json
 ├── requirements.txt
 ├── run-news.sh
 ├── eval-data/
 ├── judge-results/
 └── README.md
```