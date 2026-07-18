import argparse
import glob
import os

TEMPLATE_FILE = "laaj-prompt-template.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Create LLM-as-a-Judge prompt.")
    parser.add_argument("-i", "--input", required=True, help="latest | yyyymmddmm")
    return parser.parse_args()


def resolve_eval_dir(input_value):
    if input_value == "latest":
        all_entries = glob.glob("eval-data/*")
        dirs = [p for p in sorted(all_entries) if os.path.isdir(p)]
        if not dirs:
            raise FileNotFoundError(
                "No eval-data subdirectories found under eval-data/"
            )
        return dirs[-1]
    else:
        path = f"eval-data/{input_value}"
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} not found")
        return path


def load_text(path):
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def main():
    args = parse_args()

    # determine the eval-data/<id> directory
    eval_dir = resolve_eval_dir(args.input)

    # load necessary files
    prompt_text = load_text(f"{eval_dir}/prompt.txt")
    html_text = load_text(f"{eval_dir}/report.html")

    # load template
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as fp:
        template = fp.read()

    # embed the prompt and HTML into the template
    prompt = template.replace("{{PROMPT}}", prompt_text).replace("{{HTML}}", html_text)

    print(prompt)


if __name__ == "__main__":
    main()
