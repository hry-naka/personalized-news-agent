import os
import sys
import time
import json
import yaml
import re
import argparse
import numpy as np
from google import genai
from bs4 import BeautifulSoup
from typing import List, Tuple, Iterable

from GeminiRateLimiter.tpm_window import TpmWindow
from GeminiRateLimiter.retry import RetryHandler
from GeminiRateLimiter.batcher import BatchEmbedder
from GeminiRateLimiter.embedder import GeminiEmbedder
from GeminiRateLimiter.embedder import RPDQuotaExceeded

# Default Embedding model
EMBED_MODEL = "models/gemini-embedding-2"

# Default Translation model (Gemini Flash)
TRANSLATE_MODEL = "gemini-2.5-flash"

header = (
    "timestamp,mail_subject,article_index,"
    "title_score,summary_score,reason_score,article_score,"
    "is_counter,is_translated"
)


def load_config() -> dict:
    """
    Load config.yaml for API keys and settings.

    Returns:
        dict: Parsed configuration.
    """
    if not os.path.exists("config.yaml"):
        print("ERROR: config.yaml not found.")
        sys.exit(1)
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def contains_non_ascii(text: str) -> bool:
    """
    Check if the text contains non-ASCII characters.

    Args:
        text (str): Input text.

    Returns:
        bool: True if non-ASCII characters are present.
    """
    return any(ord(ch) > 127 for ch in text)


def translate_text_to_english(client, text, config) -> str:
    """
    Translate ONLY non-English text to English using Gemini.

    Args:
        client: Google GenAI client instance.
        text (str): Input text.
        config (dict): Configuration dictionary.

    Returns:
        str: Translated or original text.
    """
    if not text or not text.strip():
        return text

    if contains_non_ascii(text):
        print(
            f"INFO: Translating text to English using model "
            f"{config.get('gemini_llm_model', TRANSLATE_MODEL)}..."
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


def translate_html_to_english(client, html_text, config) -> str:
    """
    Translate ONLY non-English text inside HTML into English.

    Args:
        client: Google GenAI client instance.
        html_text (str): HTML content.
        config (dict): Configuration dictionary.

    Returns:
        str: Translated or original HTML.
    """
    if not html_text or not html_text.strip():
        return html_text

    if contains_non_ascii(html_text):
        print(
            f"INFO: Translating HTML to English using model "
            f"{config.get('gemini_llm_model', TRANSLATE_MODEL)}..."
        )
        response = client.models.generate_content(
            model=config.get("gemini_llm_model", TRANSLATE_MODEL),
            contents=config.get("translate_html_prompt") + f"{html_text}",
        )
        return response.text
    return html_text


def save_translated_files(target_dir, prompt_eng, html_eng) -> None:
    """
    Save translated prompt and HTML for debugging and evaluation verification.

    Args:
        target_dir (str): Target directory path.
        prompt_eng (str): Translated prompt text.
        html_eng (str): Translated HTML text.
    """
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


def translated_files_exist(target_dir) -> bool:
    """
    Check if translated files already exist.

    Args:
        target_dir (str): Target directory path.

    Returns:
        bool: True if both translated files exist.
    """
    prompt_eng = os.path.join(target_dir, "prompt-eng.txt")
    html_eng = os.path.join(target_dir, "report-eng.html")
    return os.path.exists(prompt_eng) and os.path.exists(html_eng)


def load_meta(target_dir) -> dict:
    """
    Load meta.json from target directory.

    Args:
        target_dir (str): Target directory path.

    Returns:
        dict: Parsed meta information.
    """
    meta_path = os.path.join(target_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: meta.json not found in {target_dir}")
        sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_file(target_dir, filename) -> str:
    """
    Load a text file from target directory.

    Args:
        target_dir (str): Target directory path.
        filename (str): File name.

    Returns:
        str: File content.
    """
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Required file '{filename}' not found in {target_dir}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json_file(target_dir, filename) -> dict:
    """
    Load a JSON file from target directory.

    Args:
        target_dir (str): Target directory path.
        filename (str): File name.

    Returns:
        dict or list: Parsed JSON content.
    """
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Required file '{filename}' not found in {target_dir}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(v1, v2) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1 (List[float]): First vector.
        v2 (List[float]): Second vector.

    Returns:
        float: Cosine similarity.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def safe_embedding_text(label: str) -> str:
    """
    Return a safe default text for embedding.

    Args:
        label (str): Label describing missing content.

    Returns:
        str: Safe placeholder text.
    """
    return f"{label} not found"


def detect_latest_eval_dir() -> str:
    """
    Detect the latest evaluation directory under eval-data.

    Returns:
        str: Path to the latest evaluation directory.
    """
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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate prompt vs output HTML.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Evaluation target: 'latest' or path/to/eval-data/YYYYMMDDHHMM",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Append CSV output to specified file",
    )
    parser.add_argument(
        "--header", action="store_true", help="Force CSV header fix only"
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["summary", "articles", "all"],
        default="all",
        help="Evaluation mode: summary, articles, or all",
    )
    return parser.parse_args()


def ensure_csv_header(output_path: str) -> None:
    """
    Ensure the output CSV file has the correct header. If the file does not exist or is empty, create it and write the header. If the file exists but the header is incorrect, overwrite it with the correct header.
    """
    if output_path is None:
        return  # No output path provided, nothing to do

    # if the file does not exist, create it and write the header
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
        return

    # if the file exists → check its content
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # empty file → write header
    if len(lines) == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
        return

    # if the first line is not the correct header → overwrite with correct header
    first_line = lines[0].strip()
    if first_line != header:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.writelines(lines[0:])  # write the rest of the file after the header
        return


def extract_reason(art) -> str:
    """
    Extract reason text from an article element, handling variations in HTML structure.

    Args:
        art: BeautifulSoup article element.

    Returns:
        str: Extracted reason text or placeholder.
    """
    reason_el = art.find(class_="reason")
    if not reason_el:
        reason_el = art.find(
            string=lambda x: re.search(r"Reason", x or "", re.IGNORECASE)
        )
    if reason_el and hasattr(reason_el, "get_text"):
        return reason_el.get_text(strip=True)
    return "(reason not found)"


def extract_summary(art) -> str:
    """
    Extract summary text from an article element, handling variations in HTML structure.

    Args:
        art: BeautifulSoup article element.

    Returns:
        str: Extracted summary text or placeholder.
    """
    summary_el = art.find(class_="summary")
    if not summary_el:
        summary_el = art.find(
            string=lambda x: re.search(r"Summary", x or "", re.IGNORECASE)
        )
    if summary_el and hasattr(summary_el, "get_text"):
        return summary_el.get_text(strip=True)
    return "(summary not found)"


def parse_articles_from_html(html_text: str) -> List[dict]:
    """
    Parse articles from HTML and extract title, summary, reason, and counter status.

    Args:
        html_text (str): HTML content.

    Returns:
        List[dict]: List of article metadata.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    articles = []

    for idx, art in enumerate(soup.find_all("article"), start=1):
        a_tag = art.find("a")
        title = a_tag.get_text(strip=True) if a_tag else ""

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
) -> Tuple[str, str]:
    """
    Get prompt and HTML text, translating to English if needed.

    Args:
        client: Google GenAI client instance.
        config (dict): Configuration dictionary.
        prompt_text (str): Original prompt text.
        html_text (str): Original HTML text.
        target_dir (str): Target directory path.
        force_no_translation (bool): If True, skip translation.

    Returns:
        Tuple[str, str]: Prompt and HTML text (possibly translated).
    """
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


def evaluate_prompt_and_html(embedder, prompt_text, html_text) -> float:
    """
    Evaluate overall similarity between prompt and HTML using embeddings.

    Args:
        embedder (GeminiEmbedder): Embedding wrapper.
        prompt_text (str): Prompt text.
        html_text (str): HTML text.

    Returns:
        float: Cosine similarity score.
    """
    if not prompt_text or not prompt_text.strip():
        prompt_text = safe_embedding_text("prompt")
    if not html_text or not html_text.strip():
        html_text = safe_embedding_text("html")
    try:
        prompt_vec = embedder.embed(prompt_text)
        html_vec = embedder.embed(html_text)
    except RPDQuotaExceeded:
        print("INFO: RPD exceeded. Stopping evaluation for today.")
        sys.exit(1)
    return cosine_similarity(prompt_vec, html_vec)


def evaluate_articles(embedder, batcher, prompt_text, html_text) -> List[dict]:
    """
    Evaluate each article's title, summary, reason, and overall content against the prompt.

    Args:
        embedder (GeminiEmbedder): Embedding wrapper.
        batcher (BatchEmbedder): Batch helper.
        prompt_text (str): Prompt text.
        html_text (str): HTML text.

    Returns:
        List[dict]: Evaluation results per article.
    """
    if not prompt_text or not prompt_text.strip():
        prompt_text = safe_embedding_text("prompt")
    try:
        prompt_vec = embedder.embed(prompt_text)
        articles = parse_articles_from_html(html_text)
    except RPDQuotaExceeded:
        print("INFO: RPD exceeded. Stopping evaluation for today.")
        sys.exit(1)

    texts = []
    for art in articles:
        title = art["title"] or safe_embedding_text("title")
        summary = art["summary"] or safe_embedding_text("summary")
        reason = art["reason"] or safe_embedding_text("reason")
        raw_html = art["raw_html"] or safe_embedding_text("article")
        texts.extend([title, summary, reason, raw_html])

    try:
        vectors = batcher.embed_batches(texts, embedder.embed_batch)
    except RPDQuotaExceeded:
        print("INFO: RPD exceeded. Stopping evaluation for today.")
        sys.exit(1)

    results = []
    for i, art in enumerate(articles):
        base = i * 4
        title_vec = vectors[base]
        summary_vec = vectors[base + 1]
        reason_vec = vectors[base + 2]
        article_vec = vectors[base + 3]

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


def write_summary_row(f, timestamp, mail_subject, score, is_translated) -> None:
    """
    Write a single row of summary evaluation to the output file.

    Args:
        f: File object.
        timestamp (str): Timestamp string.
        mail_subject (str): Mail subject.
        score (float): Summary score.
        is_translated (int): 1 if translated, 0 otherwise.
    """
    row = f"{timestamp},{mail_subject},-,,,," f"{score:.6f},0,{is_translated}"
    f.write(row + "\n")


def write_article_row(f, timestamp, mail_subject, result, is_translated) -> None:
    """
    Write a single row of article evaluation to the output file.

    Args:
        f: File object.
        timestamp (str): Timestamp string.
        mail_subject (str): Mail subject.
        result (dict): Article evaluation result.
        is_translated (int): 1 if translated, 0 otherwise.
    """
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
    embedder,
    batcher,
    prompt_text,
    html_text,
    meta,
    target_dir,
    output_path,
    header_only=False,
) -> None:
    """
    Evaluate each article against the prompt and write results to output file.

    Args:
        client: Google GenAI client instance.
        config (dict): Configuration dictionary.
        embedder (GeminiEmbedder): Embedding wrapper.
        batcher (BatchEmbedder): Batch helper.
        prompt_text (str): Prompt text.
        html_text (str): HTML text.
        meta (dict): Meta information.
        target_dir (str): Target directory path.
        output_path (str): Output CSV file path.
        header_only (bool): If True, output only header.
    """
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
        target_prompt, target_html = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=False,
        )
        results = evaluate_articles(embedder, batcher, target_prompt, target_html)
        with open(output_path, "a", encoding="utf-8") as f:
            for r in results:
                write_article_row(
                    f,
                    meta["timestamp"],
                    meta.get("mail_subject", ""),
                    r,
                    is_translated=1,
                )

        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        results_raw = evaluate_articles(embedder, batcher, prompt_raw, html_raw)
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
        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        results_raw = evaluate_articles(embedder, batcher, prompt_raw, html_raw)
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
    embedder,
    prompt_text,
    html_text,
    articles_list,
    meta,
    target_dir,
    output_path,
    header_only=False,
) -> None:
    """
    Evaluate the overall prompt vs HTML and write summary results to output file.

    Args:
        client: Google GenAI client instance.
        config (dict): Configuration dictionary.
        embedder (GeminiEmbedder): Embedding wrapper.
        prompt_text (str): Prompt text.
        html_text (str): HTML text.
        articles_list (list): Articles list (unused, kept for compatibility).
        meta (dict): Meta information.
        target_dir (str): Target directory path.
        output_path (str): Output CSV file path.
        header_only (bool): If True, output only header.
    """
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
        target_prompt, target_html = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=False,
        )
        score = evaluate_prompt_and_html(embedder, target_prompt, target_html)
        with open(output_path, "a", encoding="utf-8") as f:
            write_summary_row(
                f,
                meta["timestamp"],
                meta.get("mail_subject", ""),
                score,
                is_translated=1,
            )

        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        score_raw = evaluate_prompt_and_html(embedder, prompt_raw, html_raw)
        with open(output_path, "a", encoding="utf-8") as f:
            write_summary_row(
                f,
                meta["timestamp"],
                meta.get("mail_subject", ""),
                score_raw,
                is_translated=0,
            )
    else:
        prompt_raw, html_raw = get_prompt_and_html(
            client,
            config,
            prompt_text,
            html_text,
            target_dir,
            force_no_translation=True,
        )
        score_raw = evaluate_prompt_and_html(embedder, prompt_raw, html_raw)
        with open(output_path, "a", encoding="utf-8") as f:
            write_summary_row(
                f,
                meta["timestamp"],
                meta.get("mail_subject", ""),
                score_raw,
                is_translated=0,
            )


def main():
    """
    Main entry point for evaluation script.
    """
    args = parse_args()
    if args.input == "latest":
        target_dir = detect_latest_eval_dir()
    else:
        target_dir = args.input
        if not os.path.exists(target_dir):
            print(f"ERROR: Directory not found: {target_dir}")
            sys.exit(1)

    if args.header:
        ensure_csv_header(args.output)
        print("INFO: CSV header ensured. Exiting as --header flag is set.")
        sys.exit(0)
    else:
        ensure_csv_header(args.output)
        print(
            f"INFO: CSV header ensured for output file: {args.output} and continuing with evaluation."
        )

    config = load_config()

    gemini_api_key = config.get("gemini_api_key")
    if not gemini_api_key:
        print("ERROR: gemini_api_key missing in config.yaml")
        sys.exit(1)
    client = genai.Client(api_key=gemini_api_key)

    meta = load_meta(target_dir)
    prompt_text = load_text_file(target_dir, meta["prompt_file"])
    html_text = load_text_file(target_dir, meta["html_file"])
    articles_list = load_json_file(target_dir, meta["articles_file"])

    tokenizer_model_path = config.get("tokenizer_model_path")
    token = config.get("huggingface_token")
    tpm = TpmWindow(tokenizer_model_path=tokenizer_model_path, token=token, limit=25000)
    retry = RetryHandler(default_sleep=60, verbose=True)
    embed_model = config.get("gemini_embedding_model", EMBED_MODEL)
    embedder = GeminiEmbedder(
        client=client,
        tpm=tpm,
        retry=retry,
        model=embed_model,
        max_retries=10,
    )

    batch_size = config.get("embedding_batch_size", 10)
    batcher = BatchEmbedder(batch_size=batch_size)

    if args.mode == "summary":
        eval_summary(
            client=client,
            config=config,
            embedder=embedder,
            prompt_text=prompt_text,
            html_text=html_text,
            articles_list=articles_list,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=False,
        )
    elif args.mode == "articles":
        eval_per_article(
            client=client,
            config=config,
            embedder=embedder,
            batcher=batcher,
            prompt_text=prompt_text,
            html_text=html_text,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=False,
        )
    elif args.mode == "all":
        eval_summary(
            client=client,
            config=config,
            embedder=embedder,
            prompt_text=prompt_text,
            html_text=html_text,
            articles_list=articles_list,
            meta=meta,
            target_dir=target_dir,
            output_path=args.output,
            header_only=False,
        )
        eval_per_article(
            client=client,
            config=config,
            embedder=embedder,
            batcher=batcher,
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
