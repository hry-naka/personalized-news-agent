import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import urllib.parse
import feedparser
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

# load .env file
load_dotenv()

# load environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TO_EMAIL = os.environ.get("TO_EMAIL")

# smtp server settings
SMTP_SERVER = os.environ.get("SMTP_SERVER", "127.0.0.1")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
USER_PROFILE_BASE = os.environ.get("USER_PROFILE_BASE")

NUM_FETCH_ARTICLES = 50  # max number of rss articles to fetch per keyword
NUM_OUTPUT_ARTICLES = "10〜20"  # number of articles Gemini will curate and output
NUM_OUTPUT_TREND = 5  # number of lines for dynamic trend analysis


# retrieve real URLs from Google News RSS
def get_real_url(google_news_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(
            google_news_url,
            headers=headers,
            timeout=5,
            allow_redirects=True,
            stream=True,
        )
        return response.url
    except Exception as e:
        print(f"Error: Failed to get real URL {e}")
        return google_news_url


def fetch_news_from_rss(search_query, max_count):
    encoded_query = urllib.parse.quote(search_query)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    )

    print(f"Getting RSS news for: {search_query} ...")
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

        print(f"Unveiling URL for: {title[:20]}...")
        real_url = get_real_url(encrypted_url)

        articles.append({"title": title, "source": source, "url": real_url})

    return articles


# main
def main():
    # 必須の環境変数が揃っているかチェック
    if (
        not GEMINI_API_KEY
        or not SMTP_USER
        or not SMTP_PASS
        or not TO_EMAIL
        or not USER_PROFILE_BASE
    ):
        print(
            "Error: not found .env file with required settings (API keys, passwords, etc.)."
        )
        return
    # ------------------------------------------
    # 🚀 1. チャットの文脈から「最近のトレンド」を動的生成
    # ------------------------------------------
    dynamic_trend = "特になし"
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        trend_prompt = f"""あなたはユーザーの対話履歴を把握している優れたアナリストです。
これまでのユーザーとの会話の全体の流れを振り返り、
「彼が最近特に強い関心を持っているテーマ、知的な関心の変遷、または直近の時事的なトピック」を、ニュース選定の補助にするために箇条書きで {NUM_OUTPUT_TREND} 行程度】で簡潔に抽出してください。
余計な解説は省き、箇条書きのテキストだけを出力してください。"""
        print("Analyzing recent trends from chat context...")
        trend_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=trend_prompt,
        )
        if trend_response.text:
            dynamic_trend = trend_response.text.strip()
    except Exception as e:
        print(f"Error: fail to analyze dynamic trend: {e}")

    # 普遍的プロファイルと動的トレンドを合流させる
    final_profile = f"【普遍的な基本プロファイル】\n{USER_PROFILE_BASE}\n\n【AIが分析した直近の関心・対話の変遷トレンド】\n{dynamic_trend}"

    # get from RSS feeds
    nikkei_articles = fetch_news_from_rss("日本経済新聞", NUM_FETCH_ARTICLES)
    trend_articles = fetch_news_from_rss("ニュース", NUM_FETCH_ARTICLES)
    all_articles = nikkei_articles + trend_articles

    if not all_articles:
        print(
            "Error: could not fetch any articles from RSS feeds. Please check your network connection or the RSS feed URLs."
        )
        return

    print(f"Get total: {len(all_articles)} articles from RSS feeds.")

    # create prompt text for Gemini 2.5 SDK
    articles_text = ""
    for i, article in enumerate(all_articles, 1):
        articles_text += f"\n[記事No.{i}]\n"
        articles_text += f"タイトル: {article['title']}\n"
        articles_text += f"メディア: {article['source']}\n"
        articles_text += f"URL: {article['url']}\n"
        articles_text += "---------------------\n"

    # 3. Gemini 2.5 SDK を用いた呼び出し

    main_prompt = f"""以下の【ユーザープロファイル】を厳密に読み解き、提供された【ニュース記事候補リスト】の中から、彼の知的好奇心や関心に最も合致する記事を【{NUM_OUTPUT_ARTICLES}件程度】、厳選してください。

【ユーザープロファイル】
{final_profile}

【ニュース記事候補リスト】
{articles_text}

【出力フォーマット・極めて重要な指示】
厳選した記事について、必ず以下の【HTML形式】のみで出力してください。Markdown（# や - など）は一切使用しないでください。
URLには、提供されたリストにある本物のURL（httpから始まるURL）をそのままaタグのhref属性に埋め込んでください。

以下の記述パターンを正確にトレースして出力してください。
▪ <a href="URL" target="_blank" style="font-weight: bold; text-decoration: underline;">記事のタイトル</a><br>
<br>
【選定理由】 [なぜこのユーザーに選んだのかの理由を1文で記述]<br>
<br>
【記事の要約】 [アナリスト視点による、記事の本質を突いた深い要約を2〜3文で記述]<br>
<hr style="border: 0; border-top: 1px solid #555; margin: 20px 0;">"""

    print("Analyzing and curating articles with Gemini 2.5...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=main_prompt,
    )

    report_content = response.text

    # send mail via smtplib
    print("trying to send email...")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "【AI Agent】Morning Insight Report"
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL

    html_part = MIMEText(report_content, "html", "utf-8")
    msg.attach(html_part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, TO_EMAIL, msg.as_string())
        print(f"🎉 Email sent successfully! Recipient: {TO_EMAIL}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")


if __name__ == "__main__":
    main()
