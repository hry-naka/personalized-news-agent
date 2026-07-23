import argparse
import pandas as pd
import numpy as np
import glob
import csv

# Agent-side score columns
AGENT_COLUMNS = [
    "title_score",
    "summary_score",
    "reason_score",
    "article_score",
]

# Judge-side metrics
JUDGE_METRICS = [
    "article_selection",
    "summary_quality",
    "reasoning_quality",
    "html_quality",
    "overall_alignment",
    "trajectory_quality",
    "bias_stability",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics.")
    parser.add_argument("-i", "--input", required=True, help="eval-prompt.csv")
    parser.add_argument("-f", "--from", dest="from_ts", required=True)
    parser.add_argument("-t", "--to", dest="to_ts", required=True)
    parser.add_argument("-l", "--llm-as-a-judge", action="store_true")
    parser.add_argument("-j", "--judge-dir", default="judge-results")
    parser.add_argument("-o", "--output", default="summarized-eval.csv")
    return parser.parse_args()


def filter_by_time(df, ts_from, ts_to):
    df["timestamp"] = df["timestamp"].astype(str)
    return df[(df["timestamp"] >= ts_from) & (df["timestamp"] <= ts_to)]


def extract_digest_scores(df):
    digest_rows = df[df["article_index"] == "-"]
    raw = digest_rows["article_score"].tolist()
    return [float(x) for x in raw if x not in ("", None)]


def aggregate_scores(df):
    results = {}

    # digest score
    digest_scores = extract_digest_scores(df)
    if digest_scores:
        results["digest_score"] = {
            "mean": float(np.mean(digest_scores)),
            "median": float(np.median(digest_scores)),
            "stdev": float(np.std(digest_scores)),
            "count": len(digest_scores),
        }

    # normal articles
    article_rows = df[df["article_index"] != "-"]

    for col in AGENT_COLUMNS:
        raw = article_rows[col].dropna().tolist()
        scores = [float(x) for x in raw if x not in ("", None)]
        results[col] = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "stdev": float(np.std(scores)),
            "count": len(scores),
        }

    # counter articles
    counter_rows = article_rows[article_rows["is_counter"] == "1"]

    for col in AGENT_COLUMNS:
        raw = counter_rows[col].dropna().tolist()
        scores = [float(x) for x in raw if x not in ("", None)]
        results[col + "_counter"] = {
            "mean": float(np.mean(scores)) if scores else None,
            "median": float(np.median(scores)) if scores else None,
            "stdev": float(np.std(scores)) if scores else None,
            "count": len(scores),
        }

    return results


def aggregate_judge_scores(judge_dir):
    files = glob.glob(f"{judge_dir}/*.csv")
    if not files:
        return {}

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    results = {}
    for metric in JUDGE_METRICS:
        raw = df[df["metric"] == metric]["score"].tolist()
        scores = [float(x) for x in raw if x not in ("", None)]
        if scores:
            results[metric] = {
                "judge_mean": float(np.mean(scores)),
                "judge_median": float(np.median(scores)),
                "judge_stdev": float(np.std(scores)),
                "judge_count": len(scores),
            }
    return results


def save_csv(agent_results, judge_results, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)

        # header
        writer.writerow(["metric", "mean", "median", "stdev", "count"])

        # agent metrics
        for metric, data in agent_results.items():
            writer.writerow(
                [
                    metric,
                    data.get("mean"),
                    data.get("median"),
                    data.get("stdev"),
                    data.get("count"),
                ]
            )

        # blank line
        writer.writerow([])

        # judge metrics
        for metric, data in judge_results.items():
            writer.writerow(
                [
                    metric,
                    data.get("judge_mean"),
                    data.get("judge_median"),
                    data.get("judge_stdev"),
                    data.get("judge_count"),
                ]
            )


def main():
    args = parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, na_values=[""])
    df = filter_by_time(df, args.from_ts, args.to_ts)

    agent_results = aggregate_scores(df)

    judge_results = {}
    if args.llm_as_a_judge:
        judge_results = aggregate_judge_scores(args.judge_dir)

    save_csv(agent_results, judge_results, args.output)
    print(f"Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
