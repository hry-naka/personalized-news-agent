import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import urllib.parse
import feedparser
import requests
from google import genai
from google.genai import types
# 💡 .env読み込み用のライブラリをインポート
from dotenv import load_dotenv

# 💡 同一ディレクトリにある .env ファイルを読み込みます
load_dotenv()

# ==========================================
# ⚙️ 【設定エリア】.env から安全に環境変数を取得
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TO_EMAIL = os.environ.get("TO_EMAIL")

# ✉️ SMTPサーバー設定
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.nifty.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER")
# ※ niftyなどのプロバイダで、ユーザー名がメールアドレスと同じ場合は 
# os.environ.get("SMTP_USER", TO_EMAIL) のようにフォールバックさせても便利です。
SMTP_PASS = os.environ.get("SMTP_PASS")

NUM_FETCH_ARTICLES = 50     # 各キーワードでRSSから取得する最大件数
NUM_OUTPUT_ARTICLES = "10〜20"  # Geminiが厳選して出力する記事数

USER_PROFILE_BASE = """
【思考・関心事】
- 映画の思想批評や、登場人物の行動原理の深掘りに関心がある。
- 生成AIを用いた自律型マルチエージェントシステムの構築や実務適用を探求している。
- 単なる技術の表面的な利用ではなく、構造的な理解やメタ認知的なアプローチを好む.
"""

# ==========================================
# 🌐 【本物URL抽出】GoogleニュースRSSから直リンクを回収する
# ==========================================
def get_real_url(google_news_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(google_news_url, headers=headers, timeout=5, allow_redirects=True, stream=True)
        return response.url
    except Exception as e:
        print(f"⚠️ URL転送追跡失敗: {e}")
        return google_news_url

def fetch_news_from_rss(search_query, max_count):
    encoded_query = urllib.parse.quote(search_query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    
    print(f"🔍 RSS取得中: {search_query} ...")
    feed = feedparser.parse(rss_url)
    articles = []
    
    for entry in feed.entries[:max_count]:
        full_title = entry.title
        encrypted_url = entry.link
        
        title = full_title
        source = "不明なメディア"
        if " - " in full_title:
            parts = full_title.rsplit(" - ", 1)
            title = parts[0].strip()
            source = parts[1].strip()
            
        print(f"🔗 URL解読中: {title[:20]}...")
        real_url = get_real_url(encrypted_url)
        
        articles.append({
            "title": title,
            "source": source,
            "url": real_url
        })
        
    return articles

# ==========================================
# 🧠 【メイン処理】ニュース選別とメール送信
# ==========================================
def main():
    # 必須の環境変数が揃っているかチェック
    if not GEMINI_API_KEY or not SMTP_USER or not SMTP_PASS:
        print("❌ エラー: .env ファイルに必要な設定（APIキーやパスワードなど）が見つかりません。")
        return

    # 1. RSSからニュースを一括取得
    nikkei_articles = fetch_news_from_rss("日本経済新聞", NUM_FETCH_ARTICLES)
    trend_articles = fetch_news_from_rss("ニュース", NUM_FETCH_ARTICLES)
    all_articles = nikkei_articles + trend_articles
    
    if not all_articles:
        print("❌ 有効な記事が1件も取得できませんでした。")
        return
        
    print(f"📊 合計 {len(all_articles)} 件の本物URL付き記事を回収しました。")
    
    # 2. Geminiに手渡すテキストの構築
    articles_text = ""
    for i, article in enumerate(all_articles, 1):
        articles_text += f"\n[記事No.{i}]\n"
        articles_text += f"タイトル: {article['title']}\n"
        articles_text += f"メディア: {article['source']}\n"
        articles_text += f"URL: {article['url']}\n"
        articles_text += "---------------------\n"

    # 3. Gemini 2.5 SDK を用いた呼び出し
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    main_prompt = f"""以下の【ユーザープロファイル】を厳密に読み解き、提供された【ニュース記事候補リスト】の中から、彼の知的好奇心や関心に最も合致する記事を【{NUM_OUTPUT_ARTICLES}件程度】、厳選してください。

【ユーザープロファイル】
{USER_PROFILE_BASE}

【ニュース記事候補リスト】
{articles_text}

【出力フォーマット・極めて重要な指示】
厳選した記事について、必ず以下の【HTML形式】のみで出力してください。Markdown（# や - など）は一切使用しないでください。
URLには、提供されたリストにある本物のURL（httpから始まるURL）をそのままaタグのhref属性に埋め込んでください。

<h3 style="margin-bottom: 5px;"><a href="本物のURL" target="_blank">記事のタイトル</a></h3>
<ul style="margin-top: 0;">
  <li><strong>メディア</strong>: [メディア名]</li>
  <li><strong>選定理由と知的な要約</strong>: [なぜ選んだのかの理由を交えたアナリスト視点の要約]</li>
</ul>
<br>"""

    print("🧠 Geminiによる厳選・アナリティクス要約を実行中...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=main_prompt,
    )
    
    report_content = response.text

    # 4. smtplibを用いたHTMLメールの送信
    print("📬 メール送信処理を開始...")
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
        print(f"🎉 メール送信大成功！ 送信先: {TO_EMAIL}")
    except Exception as e:
        print(f"❌ メール送信エラー: {e}")

if __name__ == "__main__":
    main()
