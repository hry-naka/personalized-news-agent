# Personalized News Agent
A Python-based automated news curation system that fetches articles from RSS feeds, analyzes them using the Gemini API, generates a personalized HTML news digest based on a user profile, and optionally stores evaluation data for later analysis.
This project is designed for daily automated execution (cron/systemd) and supports a full evaluation pipeline for measuring how well the curated articles match the user’s interests.
---
# Features
Fetches news articles from multiple RSS channels defined in config.json
Resolves redirect URLs to real article URLs
Uses Gemini 2.5 to:
Analyze a user profile
Select relevant articles
Generate structured HTML output
Include counter‑view articles for intellectual diversity
Sends the curated HTML report via SMTP
Optional evaluation mode (--eval) stores:
Prompt used
Generated HTML
Raw RSS article list
Metadata for reproducibility
Compatible with the standalone evaluation tool eval-prompt.py
---
# Requirements
Python 3.12+
A valid Gemini API key
An SMTP server (Gmail, SendGrid, local Postfix, etc.)
macOS, Linux (Ubuntu recommended), or WSL2
---
# Installation
Clone the repository:
git clone https://github.com/yourname/personalized-news-agent.git
cd personalized-news-agent

Install dependencies:
pip install -r requirements.txt

---
# Configuration
1. `.env` file
Create a .env file in the project root:
GEMINI_API_KEY=your_gemini_api_key

SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASS=your_smtp_password

TO_EMAIL=recipient@example.com

2. `config.json`
Defines RSS channels and output settings:
{
  "rss_channels": [
    { "query": "technology", "count": 30 },
    { "query": "innovation", "count": 30 },
    { "query": "culture japan", "count": 30 }
  ],
  "num_output_articles": 12
}

3. `user_profile.txt`
This file defines the user’s interests, preferences, and intellectual tendencies.
The news agent uses this profile to select and summarize articles that best match the user’s curiosity, aesthetic tastes, and analytical interests.
The content of user_profile.txt is injected directly into the main prompt and becomes the foundation for Gemini’s article‑selection logic. A well‑written profile significantly improves the relevance and quality of the curated news.
You can write this profile manually, or generate it interactively — for example, by asking Copilot to help articulate your interests, preferred topics, and the types of articles you enjoy. The agent will then use this profile to guide its curation process.

4. `main_prompt.txt`
Defines the HTML output structure and selection rules.
The script replaces:
{user_profile}
{articles_text}
{num_output_articles}
---
# Usage
Run the agent:
python news-agent.py "Daily News Digest"

This will:
Fetch RSS articles
Build the Gemini prompt
Generate curated HTML
Send the email to TO_EMAIL
---
# Saving Evaluation Data
To store all inputs/outputs for later analysis:
python news-agent.py "Debug Run" --eval

This creates:
eval-data/YYYYMMDDHHMM/
 ├── prompt.txt
 ├── report.html
 ├── articles.json
 └── meta.json

These files can be analyzed using eval-prompt.py.
---
# Evaluation Pipeline (Optional)
Run evaluation:
python eval-prompt.py -i latest -m all -o eval-data/eval-prompt.csv

Modes:
summary — overall similarity between prompt and HTML
articles — per‑article similarity
all — both
---
# Automation (Ubuntu / macOS)
Example cron entry:
0 7 * * * /usr/bin/python3 /path/to/news-agent.py "Morning Digest"

Or systemd service for more robust scheduling.
---
# Project Structure
personalized-news-agent/
 ├── news-agent.py
 ├── eval-prompt.py
 ├── main_prompt.txt
 ├── user_profile.txt
 ├── config.json
 ├── requirements.txt
 ├── eval-data/
 └── README.md

---
# Notes
The agent relies on RSS feeds; article quality varies daily.
Gemini output is deterministic only within a single run; evaluation data helps track consistency.
HTML output is designed for email clients and avoids Markdown.