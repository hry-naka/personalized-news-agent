# Personalized News Agent

A Python-based automated news curation system that fetches articles from RSS feeds, analyzes them using the Gemini API, generates a personalized HTML news digest based on a user profile, and optionally stores evaluation data for later analysis.
This project is designed for daily automated execution (cron/systemd) and supports a full evaluation pipeline for measuring how well the curated articles match the user’s interests.

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
- Fully compatible with the standalone evaluation tool `eval-prompt.py`

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
git clone https://github.com/yourname/personalized-news-agent.git
cd personalized-news-agent
```
Install dependencies:
```
pip install -r requirements.txt
```

---

# Configuration

### 1. `.env` file

Create a .env file in the project root:
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
```
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

"retry_wait_seconds" specifies the retry wait durations (in seconds) used during
normal execution.

When "force_error_status_test" is true, the agent intentionally triggers an
error with the status code defined in "force_error_status_code". In this mode,
the retry schedule switches to the shorter debug intervals defined in
"retry_wait_seconds_debug".

### 3. `user_profile.txt`

This file defines the user’s interests, preferences, and intellectual tendencies.
The news agent uses this profile to select and summarize articles that best match the user’s curiosity, aesthetic tastes, and analytical interests.

The content of user_profile.txt is injected directly into the main prompt and becomes the foundation for Gemini’s article‑selection logic. A well‑written profile significantly improves the relevance and quality of the curated news.
You can write this profile manually, or generate it interactively — for example, by asking Copilot to help articulate your interests, preferred topics, and the types of articles you enjoy. The agent will then use this profile to guide its curation process.

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

Run the agent:
```
python news-agent.py "Daily News Digest"
```
This will:
* Fetch RSS articles
* Build the Gemini prompt
* Generate curated HTML
* Send the email to TO_EMAIL

---

# Saving Evaluation Data

To store all inputs/outputs for later analysis:
```
python news-agent.py "Debug Run" --eval
```

This creates:
```
eval-data/YYYYMMDDHHMM/
 ├── prompt.txt
 ├── report.html
 ├── articles.json
 └── meta.json
```

These files can be analyzed using eval-prompt.py.

---

# Evaluation Pipeline (Optional)

Run evaluation:

```
python eval-prompt.py -i latest -m all -o eval-data/eval-prompt.csv
```

### Command-line options

- **`-i / --input`**  
  Specifies the evaluation target directory.  
  - `latest` — automatically selects the newest `eval-data/YYYYMMDDHHMM/` directory  
  - Or you can provide an explicit path:  
    `-i eval-data/202607161453`

- **`-o / --output`**  
  Appends evaluation results to the specified CSV file.  
  If the file does not exist, it will be created and the header will be written automatically.

### Modes
- **summary** — overall similarity between prompt and full HTML output  
- **articles** — per‑article similarity  
- **all** — both summary and per‑article rows

---
## CSV Output Format

The evaluation tool (`eval-prompt.py`) produces a unified CSV file containing
both summary‑level and per‑article evaluation results. All modes (`summary`,
`articles`, `all`) share the same column structure:

| column | description |
|--------|-------------|
| `timestamp` | Evaluation timestamp (YYYYMMDDHHMM) taken from `meta.json` |
| `mail_subject` | The mail subject string used when generating the digest |
| `article_index` | Article index in the HTML output. `-` indicates the summary row |
| `title_score` | Cosine similarity between the **full prompt text** and the article title extracted from the HTML |
| `summary_score` | Cosine similarity between the **full prompt text** and the article summary extracted from the HTML |
| `reason_score` | Cosine similarity between the **full prompt text** and the article’s “reason for selection” |
| `article_score` | Cosine similarity between the **full prompt text** and the entire article block (raw HTML text) |
| `is_counter` | `1` if the article is marked as a counter‑view (`data-view-type="counter"`), otherwise `0` |

### Counter‑view scoring
For counter‑view articles, all similarity scores are inverted:
```
score = 1.0 - cosine_similarity(prompt_vec, article_component_vec)
```

This allows counter‑view articles to be evaluated as “intentionally different”
from the main prompt.

---

### Row Structure by Mode

#### **summary mode**
- Produces **one row**
- `article_index = -`
- Only `article_score` is populated  
  (computed from the similarity between the full prompt and the entire HTML output)

#### **articles mode**
- Produces **one row per article**
- Each row contains all four scores:
  - `title_score`
  - `summary_score`
  - `reason_score`
  - `article_score`
- `is_counter` is set when the `<article>` tag has the attribute `data-view-type="counter"`

#### **all mode**
- First row: **summary row**
- Following rows: **per‑article rows**

---

### Notes
- All similarity scores are normalized between **0.0 and 1.0**
- Higher scores indicate stronger alignment between the prompt and the generated HTML
- CSV files can be aggregated over time to measure drift or changes in Gemini’s behavior

---

# Automation (Ubuntu / macOS)

Example cron entry:
```
0 7 * * * /usr/bin/python3 /path/to/news-agent.py "Morning Digest"
```

Or systemd service for more robust scheduling.

---

# Project Structure

```
personalized-news-agent/
 ├── news-agent.py
 ├── eval-prompt.py
 ├── main_prompt.txt
 ├── user_profile.txt
 ├── config.json
 ├── requirements.txt
 ├── eval-data/
 └── README.md
```
