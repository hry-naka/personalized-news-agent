import os
import sys
import json
import argparse
import numpy as np
from google import genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

# Embedding model setting（Gemini 2.5 Flash Embedding）
EMBED_MODEL = "models/gemini-embedding-2"


header = (
    "timestamp,mail_subject,article_index,"
    "title_score,summary_score,reason_score,article_score,"
    "is_counter"
)

def load_meta(target_dir):
    """Load meta information from a JSON file."""
    meta_path = os.path.join(target_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: meta.json not found in {target_dir}")
        sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_file(target_dir, filename):
    """Load text content from a file."""
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Required file '{filename}' not found in {target_dir}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json_file(target_dir, filename):
    """Load JSON content from a file."""
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Required file '{filename}' not found in {target_dir}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_embedding(client, text):
    """Get embedding for a given text."""
    if not text or not text.strip():
        return np.zeros(768, dtype=float).tolist()
    response = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return response.embeddings[0].values


def detect_latest_eval_dir():
    """Detect the latest evaluation directory based on timestamp."""
    base = "eval-data"
    if not os.path.exists(base):
        print("ERROR: eval-data directory not found.")
        sys.exit(1)

    dirs = []
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if os.path.isdir(full) and d.isdigit():
            dirs.append(d)

    if not dirs:
        print("ERROR: No timestamp directories found in eval-data.")
        sys.exit(1)

    latest = sorted(dirs)[-1]
    return os.path.join(base, latest)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate prompt vs output HTML.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Evaluation target: 'latest' or path/to/eval-data/YYYYMMDDHHMM",
    )
    parser.add_argument("-o", "--output", help="Append CSV output to specified file")
    parser.add_argument("--header", action="store_true", help="Output CSV header only")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["summary", "articles", "all"],
        default="summary",
        help="Evaluation mode: summary, articles, or all",
    )
    return parser.parse_args()


def parse_articles_from_html(html_text):
    """Parse articles from HTML text and extract relevant information."""
    soup = BeautifulSoup(html_text, "html.parser")
    articles = []

    for idx, art in enumerate(soup.find_all("article"), start=1):
        # Title
        a_tag = art.find("a")
        title = a_tag.get_text(strip=True) if a_tag else ""

        # Reason
        reason_el = art.find(class_="reason")
        if not reason_el:
            reason_el = art.find(string=lambda x: "選定理由" in x)
        reason = (
            reason_el.get_text(strip=True) if hasattr(reason_el, "get_text") else ""
        )

        # Summary
        summary_el = art.find(class_="summary")
        if not summary_el:
            summary_el = art.find(string=lambda x: "記事の要約" in x)
        summary = (
            summary_el.get_text(strip=True) if hasattr(summary_el, "get_text") else ""
        )

        # Counter-view detection
        is_counter = art.get("data-view-type", "") == "counter"

        articles.append(
            {
                "index": idx,
                "title": title,
                "summary": summary,
                "reason": reason,
                "is_counter": is_counter,
                "raw_html": art.get_text(" ", strip=True),
            }
        )

    return articles


def eval_per_article(
    client, prompt_text, html_text, meta, output_path, header_only=False
):
    """Evaluate each article against the prompt and output scores."""
    if header_only:
        if output_path:
            write_header = not os.path.exists(output_path)
            with open(output_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header + "\n")
            print(f"INFO: Header written to {output_path}")
        else:
            print(header)
        return
    prompt_vec = get_embedding(client, prompt_text)
    articles = parse_articles_from_html(html_text)

    timestamp = meta["timestamp"]
    mail_subject = meta.get("mail_subject", "")

    if output_path:
        write_header = not os.path.exists(output_path)
        with open(output_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header + "\n")
            for art in articles:
                title_vec = get_embedding(client, art["title"])
                summary_vec = get_embedding(client, art["summary"])
                reason_vec = get_embedding(client, art["reason"])
                article_vec = get_embedding(client, art["raw_html"])

                title_sim = cosine_similarity(prompt_vec, title_vec)
                summary_sim = cosine_similarity(prompt_vec, summary_vec)
                reason_sim = cosine_similarity(prompt_vec, reason_vec)
                article_sim = cosine_similarity(prompt_vec, article_vec)

                if art["is_counter"]:
                    title_score = 1.0 - title_sim
                    summary_score = 1.0 - summary_sim
                    reason_score = 1.0 - reason_sim
                    article_score = 1.0 - article_sim
                else:
                    title_score = title_sim
                    summary_score = summary_sim
                    reason_score = reason_sim
                    article_score = article_sim

                row = (
                    f"{timestamp},{mail_subject},{art['index']},"
                    f"{title_score:.6f},{summary_score:.6f},{reason_score:.6f},{article_score:.6f},"
                    f"{int(art['is_counter'])}"
                )
                f.write(row + "\n")
        print(f"INFO: Appended per-article evaluation to {output_path}")
    else:
        print(header)
        for art in articles:
            title_vec = get_embedding(client, art["title"])
            summary_vec = get_embedding(client, art["summary"])
            reason_vec = get_embedding(client, art["reason"])
            article_vec = get_embedding(client, art["raw_html"])

            title_sim = cosine_similarity(prompt_vec, title_vec)
            summary_sim = cosine_similarity(prompt_vec, summary_vec)
            reason_sim = cosine_similarity(prompt_vec, reason_vec)
            article_sim = cosine_similarity(prompt_vec, article_vec)

            if art["is_counter"]:
                title_score = 1.0 - title_sim
                summary_score = 1.0 - summary_sim
                reason_score = 1.0 - reason_sim
                article_score = 1.0 - article_sim
            else:
                title_score = title_sim
                summary_score = summary_sim
                reason_score = reason_sim
                article_score = article_sim

            row = (
                f"{timestamp},{mail_subject},{art['index']},"
                f"{title_score:.6f},{summary_score:.6f},{reason_score:.6f},{article_score:.6f},"
                f"{int(art['is_counter'])}"
            )
            print(row)


def eval_summary(
    client, prompt_text, html_text, articles_list, meta, output_path, header_only=False
):
    """Evaluate the overall summary against the prompt and output scores."""

    if header_only:
        if output_path:
            write_header = not os.path.exists(output_path)
            with open(output_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header + "\n")
            print(f"INFO: Header written to {output_path}")
        else:
            print(header)
        return

    prompt_vec = get_embedding(client, prompt_text)
    html_vec = get_embedding(client, html_text)

    main_view_score = cosine_similarity(prompt_vec, html_vec)
    counter_view_score = 1.0 - main_view_score

    timestamp = meta["timestamp"]
    mail_subject = meta.get("mail_subject", "")

    # article_index is set to '-' for summary evaluation
    row = (
    f"{timestamp},{mail_subject},-,"
    f",,,,{main_view_score:.6f},0"
    )


    if output_path:
        write_header = not os.path.exists(output_path)
        with open(output_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header + "\n")
            f.write(row + "\n")
        print(f"INFO: Appended evaluation result to {output_path}")
    else:
        print(header)
        print(row)


def main():
    """Main function to handle evaluation based on command-line arguments."""
    args = parse_args()

    if args.input == "latest":
        target_dir = detect_latest_eval_dir()
    else:
        target_dir = args.input
        if not os.path.exists(target_dir):
            print(f"ERROR: Directory not found: {target_dir}")
            sys.exit(1)

    meta = load_meta(target_dir)
    prompt_text = load_text_file(target_dir, meta["prompt_file"])
    html_text = load_text_file(target_dir, meta["html_file"])
    articles_list = load_json_file(target_dir, meta["articles_file"])

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    if args.mode == "summary":
        eval_summary(
            client=client,
            prompt_text=prompt_text,
            html_text=html_text,
            articles_list=articles_list,
            meta=meta,
            output_path=args.output,
            header_only=args.header,
        )

    elif args.mode == "articles":
        eval_per_article(
            client=client,
            prompt_text=prompt_text,
            html_text=html_text,
            meta=meta,
            output_path=args.output,
            header_only=args.header,
        )

    elif args.mode == "all":
        eval_summary(
            client=client,
            prompt_text=prompt_text,
            html_text=html_text,
            articles_list=articles_list,
            meta=meta,
            output_path=args.output,
            header_only=args.header,
        )
        eval_per_article(
            client=client,
            prompt_text=prompt_text,
            html_text=html_text,
            meta=meta,
            output_path=args.output,
            header_only=False,
        )


if __name__ == "__main__":
    main()
