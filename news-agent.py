import os
import sys
import time
import json
import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import urllib.parse
import feedparser
import requests
from datetime import datetime as DT
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TO_EMAIL = os.environ.get("TO_EMAIL")

# SMTP server settings
SMTP_SERVER = os.environ.get("SMTP_SERVER", "127.0.0.1")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")

# File path constants
CONFIG_PATH = "config.json"
PROFILE_PATH = "user_profile.txt"
PROMPT_PATH = "main_prompt.txt"


def load_external_files():
    """Load config, profile, and prompt from external files."""
    if not os.path.exists(CONFIG_PATH):
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"Error: Required file '{CONFIG_PATH}' not found."
        )
        sys.exit(1)
    if not os.path.exists(PROFILE_PATH):
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"Error: Required file '{PROFILE_PATH}' not found."
        )
        sys.exit(1)
    if not os.path.exists(PROMPT_PATH):
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"Error: Required file '{PROMPT_PATH}' not found."
        )
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        user_profile = f.read()

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    return config_data, user_profile, prompt_template


def get_real_url(news_url):
    """Retrieve real URLs from News RSS."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(
            news_url,
            headers=headers,
            timeout=5,
            allow_redirects=True,
            stream=True,
        )
        return response.url
    except Exception as e:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"WARNING: Failed to get real URL, fallback to original: {e}"
        )
        return news_url


def fetch_news_from_rss(search_query, max_count):
    """Fetch and parse articles from Bing News RSS based on query."""
    encoded_query = urllib.parse.quote(search_query)
    rss_url = f"https://www.bing.com/news/search?q={encoded_query}&format=rss"
    print(
        f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f"INFO: Fetching RSS news for query: '{search_query}' (Max: {max_count})..."
    )

    try:
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:max_count]:
            full_title = entry.title
            encrypted_url = entry.link
            title = full_title
            source = "unknown"

            if " - " in full_title:
                parts = full_title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            real_url = get_real_url(encrypted_url)
            time.sleep(0.3)  # To avoid overwhelming the server
            articles.append({"title": title, "source": source, "url": real_url})
        return articles
    except Exception as e:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"ERROR: Failed to parse RSS feed for '{search_query}': {e}"
        )
        return []


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="News curation AI-agent.")
    parser.add_argument(
        "MailSubjectString",
        metavar="mail_subject_string",
        type=str,
        help="Subject string for the email report",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # 1. Check core environment variables
    if not GEMINI_API_KEY or not SMTP_USER or not SMTP_PASS or not TO_EMAIL:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"ERROR: Missing required settings in .env (API keys, SMTP credentials, etc.)."
        )
        return

    # 2. Load external settings and assets
    print(
        f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f"INFO: Loading configuration and asset files..."
    )
    config_data, user_profile, prompt_template = load_external_files()

    rss_channels = config_data.get("rss_channels", [])
    if not rss_channels:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"ERROR: No RSS channels defined in config.json."
        )
        return

    # 3. Dynamic RSS Looping based on config
    all_articles = []
    for channel in rss_channels:
        query = channel.get("query")
        count = channel.get("count", 30)
        if query:
            articles = fetch_news_from_rss(query, count)
            all_articles.extend(articles)

    if not all_articles:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"ERROR: Could not fetch any articles from RSS channels. Check network or configuration."
        )
        return

    print(
        f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f"SUCCESS: Fetched total {len(all_articles)} articles from RSS."
    )

    # 4. Format articles into plain text for Gemini
    articles_text = ""
    for i, article in enumerate(all_articles, 1):
        articles_text += f"\n[Article No.{i}]\n"
        articles_text += f"Title: {article['title']}\n"
        articles_text += f"Source: {article['source']}\n"
        articles_text += f"URL: {article['url']}\n"
        articles_text += "---------------------\n"

    # get the number of articles to output from config.json (default is 5)
    num_output_articles = config_data.get("num_output_articles", 5)

    # 5. Construct Main Prompt
    final_prompt = (
        prompt_template.replace("{user_profile}", user_profile)
        .replace("{articles_text}", articles_text)
        .replace("{num_output_articles}", str(num_output_articles))
    )

    # 6. Call Gemini 2.5 API
    print("INFO: Analyzing and curating articles with Gemini 2.5...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt,
        )
        report_content = response.text
    except Exception as e:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"ERROR: Gemini API execution failed: {e}"
        )
        return

    # 7. Send Curated Report Email
    print(
        f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f"INFO: Preparing and sending email..."
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"【AI news Agent】: {args.MailSubjectString} insights report"
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL

    html_part = MIMEText(report_content, "html", "utf-8")
    msg.attach(html_part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, TO_EMAIL, msg.as_string())
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"SUCCESS: Email sent successfully to {TO_EMAIL}"
        )
    except Exception as e:
        print(
            f"[{DT.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            f"ERROR: SMTP email transmission failed: {e}"
        )


if __name__ == "__main__":
    main()
