import os
import sys
import json
import yaml
import re
import argparse
import numpy as np
from google import genai
from bs4 import BeautifulSoup

# Default Embedding model
EMBED_MODEL = "models/gemini-embedding-2"

# Default Translation model (Gemini Flash)
TRANSLATE_MODEL = "gemini-2.5-flash"


header = (
    "timestamp,mail_subject,article_index,"
    "title_score,summary_score,reason_score,article_score,"
    "is_counter,is_translated"
)


def load_config():
    """Load config.yaml for API keys and settings."""
    if not os.path.exists("config.yaml"):
        print("ERROR: config.yaml not found.")
        sys.exit(1)
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def contains_non_ascii(text):
    return any(ord(ch) > 127 for ch in text)


def translate_text_to_english(client, text, config):
    """Translate ONLY non-English text to English using Gemini."""
    if not text or not text.strip():
        return text

    # If the text contains non-ASCII characters, assume translation is needed
    if contains_non_ascii(text):
        print(
            f"INFO: Translating text to English using model {config.get('gemini_llm_model', TRANSLATE_MODEL)}..."
        )
        try:
            response = client.models.generate_content(
                model=config.get("gemini_llm_model", TRANSLATE_MODEL),
                contents=config.get("translate_text_prompt") + f"{text}",
            )
            return response.text
        except Exception as e:
            print(f"WARNING: Translation failed, using original text: {e}")
            return text
    return text


def translate_html_to_english(client, html_text, config):
    """Translate ONLY non-English text inside HTML into English."""
    if not html_text or not html_text.strip():
        return html_text

    if contains_non_ascii(html_text):
        print(
            f"INFO: Translating HTML to English using model {config.get('gemini_llm_model', TRANSLATE_MODEL)}..."
        )
        response = client.models.generate_content(
            model=config.get("gemini_llm_model", TRANSLATE_MODEL),
            contents=config.get("translate_html_prompt") + f"{html_text}",
        )
        return response.text
    return html_text


def save_translated_files(target_dir, prompt_eng, html_eng):
    """Save translated prompt and HTML for debugging and evaluation verification."""
    prompt_path = os.path.join(target_dir, "prompt-eng.txt")
    html_path = os.path.join(target_dir, "report-eng.html")

    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_eng)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_eng)
        print(f"INFO: Saved translated files to {target_dir}")
    except Exception as e:
        print(f"WARNING: Failed to save translated files: {e}")


def translated_files_exist(target_dir):
    prompt_eng = os.path.join(target_dir, "prompt-eng.txt")
    html_eng = os.path.join(target_dir, "report-eng.html")
    return os.path.exists(prompt_eng) and os.path.exists(html_eng)


def load_meta(target_dir):
    meta_path = os.path.join(target_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: meta.json not found in {target_dir}")
        sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_file(target_dir, filename):
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Required file '{filename}' not found in {target_dir}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json_file(target_dir, filename):
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Required file '{filename}' not found in {target_dir}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def safe_embedding_text(label):
    """Return a safe default text for embedding."""
    return f"{label} not found"


def get_embedding(client, text, config):
    """Get embedding vector for the given text using Gemini embedding model."""
    if not text or not text.strip():
        # if text is empty or whitespace-only, use a safe default
        safe_text = safe_embedding_text("text")
        return (
            client.models.embed_content(
                model=config.get("gemini_embedding_model", EMBED_MODEL),
                contents=safe_text,
            )
            .embeddings[0]
            .values
        )

    response = client.models.embed_content(
        model=config.get("gemini_embedding_model", EMBED_MODEL), contents=text
    )
    return response.embeddings[0].values


def detect_latest_eval_dir():
    base = "eval-data"
    if not os.path.exists(base):
        print("ERROR: eval-data directory not found.")
        sys.exit(1)

    dirs = [d for d in os.listdir(base) if d.isdigit()]
    if not dirs:
        print("ERROR: No timestamp directories found in eval-data.")
        sys.exit(1)

    latest = sorted(dirs)[-1]
    return os.path.join(base, latest)


def parse_args():
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


def extract_reason(art):
    """Extract reason text from an article element, handling variations in HTML structure."""
    reason_el = art.find(class_="reason")
    if not reason_el:
        reason_el = art.find(
            string=lambda x: re.search(r"Reason", x or "", re.IGNORECASE)
        )
    if reason_el and hasattr(reason_el, "get_text"):
        return reason_el.get_text(strip=True)
    return "(reason not found)"


def extract_summary(art):
    """Extract summary text from an article element, handling variations in HTML structure."""
    summary_el = art.find(class_="summary")
    if not summary_el:
        summary_el = art.find(
            string=lambda x: re.search(r"Summary", x or "", re.IGNORECASE)
        )
    if summary_el and hasattr(summary_el, "get_text"):
        return summary_el.get_text(strip=True)
    return "(summary not found)"


def parse_articles_from_html(html_text):
    """Parse articles from HTML and extract title, summary, reason, and counter status."""
    soup = BeautifulSoup(html_text, "html.parser")
    articles = []

    for idx, art in enumerate(soup.find_all("article"), start=1):
        a_tag = art.find("a")
        title = a_tag.get_text(strip=True) if a_tag else ""

        # Updated labels for B-test
        reason = extract_reason(art)
        summary = extract_summary(art)

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


def get_prompt_and_html(
    client, config, prompt_text, html_text, target_dir, force_no_translation
):
    """Get prompt and HTML text, translating to English if needed."""
    if force_no_translation:
        return prompt_text, html_text

    if config.get("translate_when_evaluating", True):
        if translated_files_exist(target_dir):
            return load_text_file(target_dir, "prompt-eng.txt"), load_text_file(
                target_dir, "report-eng.html"
            )
        else:
            prompt_eng = translate_text_to_english(client, prompt_text, config)
            html_eng = translate_html_to_english(client, html_text, config)
            save_translated_files(target_dir, prompt_eng, html_eng)
            return prompt_eng, html_eng
    else:
        return prompt_text, html_text


def evaluate_prompt_and_html(client, config, prompt_text, html_text):
    prompt_vec = get_embedding(client, prompt_text, config)
    html_vec = get_embedding(client, html_text, config)
    main_score = cosine_similarity(prompt_vec, html_vec)
    return main_score


def evaluate_articles(client, config, prompt_text, html_text):
    """Evaluate each article's title, summary, reason, and overall content against the prompt."""
    prompt_vec = get_embedding(client, prompt_text, config)
    articles = parse_articles_from_html(html_text)
    results = []
    for art in articles:
        title_vec = get_embedding(client, art["title"], config)
        summary_vec = get_embedding(client, art["summary"], config)
        reason_vec = get_embedding(client, art["reason"], config)
        article_vec = get_embedding(client, art["raw_html"], config)
        if art["is_counter"]:
            title_score = 1.0 - cosine_similarity(prompt_vec, title_vec)
            summary_score = 1.0 - cosine_similarity(prompt_vec, summary_vec)
            reason_score = 1.0 - cosine_similarity(prompt_vec, reason_vec)
            article_score = 1.0 - cosine_similarity(prompt_vec, article_vec)
        else:
            title_score = cosine_similarity(prompt_vec, title_vec)
            summary_score = cosine_similarity(prompt_vec, summary_vec)
            reason_score = cosine_similarity(prompt_vec, reason_vec)
            article_score = cosine_similarity(prompt_vec, article_vec)
        results.append(
            {
                "index": art["index"],
                "title_score": title_score,
                "summary_score": summary_score,
                "reason_score": reason_score,
                "article_score": article_score,
                "is_counter": art["is_counter"],
            }
        )
    return results


def write_summary_row(f, timestamp, mail_subject, score, is_translated):
    """Write a single row of summary evaluation to the output file."""
    row = f"{timestamp},{mail_subject},-,,,," f"{score:.6f},0,{is_translated}"
    f.write(row + "\n")


def write_article_row(f, timestamp, mail_subject, result, is_translated):
    """Write a single row of article evaluation to the output file."""
    row = (
        f"{timestamp},{mail_subject},{result['index']},"
        f"{result['title_score']:.6f},{result['summary_score']:.6f},"
        f"{result['reason_score']:.6f},{result['article_score']:.6f},"
        f"{result['is_counter']},{is_translated}"
    )
    f.write(row + "\n")


def eval_per_article(
    client,
    config,
    prompt_text,
    html_text,
    meta,
    target_dir,
    output_path,
    header_only=False,
):
    """Evaluate each article against the prompt and write results to output file."""
    if header_only:
        if output_path:
            write_header = not os.path.exists(output_path)
            with open(output_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header + "\n")
        else:
            print(header)
        return

    if config.get("translate_when_evaluating", True):
        # translation enabled
        target_prompt, target_html = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=False,
        )
        results = evaluate_articles(client, config, target_prompt, target_html)
        with open(output_path, "a", encoding="utf-8") as f:
            for r in results:
                write_article_row(
                    f,
                    meta["timestamp"],
                    meta.get("mail_subject", ""),
                    r,
                    is_translated=1,
                )
        # translation disabled
        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        results_raw = evaluate_articles(client, config, prompt_raw, html_raw)
        with open(output_path, "a", encoding="utf-8") as f:
            for r in results_raw:
                write_article_row(
                    f,
                    meta["timestamp"],
                    meta.get("mail_subject", ""),
                    r,
                    is_translated=0,
                )
    else:
        # translation disabled only
        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        results_raw = evaluate_articles(client, config, prompt_raw, html_raw)
        with open(output_path, "a", encoding="utf-8") as f:
            for r in results_raw:
                write_article_row(
                    f,
                    meta["timestamp"],
                    meta.get("mail_subject", ""),
                    r,
                    is_translated=0,
                )


def eval_summary(
    client,
    config,
    prompt_text,
    html_text,
    articles_list,
    meta,
    target_dir,
    output_path,
    header_only=False,
):
    """Evaluate the overall prompt vs HTML and write summary results to output file."""
    if header_only:
        if output_path:
            write_header = not os.path.exists(output_path)
            with open(output_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header + "\n")
        else:
            print(header)
        return

    if config.get("translate_when_evaluating", True):
        # translation enabled
        target_prompt, target_html = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=False,
        )
        score = evaluate_prompt_and_html(client, config, target_prompt, target_html)
        with open(output_path, "a", encoding="utf-8") as f:
            write_summary_row(
                f,
                meta["timestamp"],
                meta.get("mail_subject", ""),
                score,
                is_translated=1,
            )

        # translation disabled
        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        score_raw = evaluate_prompt_and_html(client, config, prompt_raw, html_raw)
        with open(output_path, "a", encoding="utf-8") as f:
            write_summary_row(
                f,
                meta["timestamp"],
                meta.get("mail_subject", ""),
                score_raw,
                is_translated=0,
            )
    else:
        # translation disabled only
        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        score_raw = evaluate_prompt_and_html(client, config, prompt_raw, html_raw)
        with open(output_path, "a", encoding="utf-8") as f:
            write_summary_row(
                f,
                meta["timestamp"],
                meta.get("mail_subject", ""),
                score_raw,
                is_translated=0,
            )


def main():
    args = parse_args()

    # Load config.yaml
    config = load_config()
    GEMINI_API_KEY = config.get("gemini_api_key")
    if not GEMINI_API_KEY:
        print("ERROR: gemini_api_key missing in config.yaml")
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)

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

    if args.mode == "summary":
        eval_summary(
            client=client,
            config=config,
            prompt_text=prompt_text,
            html_text=html_text,
            articles_list=articles_list,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=args.header,
        )

    elif args.mode == "articles":
        eval_per_article(
            client=client,
            config=config,
            prompt_text=prompt_text,
            html_text=html_text,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=args.header,
        )

    elif args.mode == "all":
        eval_summary(
            client=client,
            config=config,
            prompt_text=prompt_text,
            html_text=html_text,
            articles_list=articles_list,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=args.header,
        )
        eval_per_article(
            client=client,
            config=config,
            prompt_text=prompt_text,
            html_text=html_text,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=False,
        )
    print("INFO: evaluation done.")


if __name__ == "__main__":
    main()
