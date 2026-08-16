import argparse
import glob
import os
import re

TEMPLATE_FILE = "laaj-prompt-template.txt"
PROMPT_HEADER = (
    'I will send the judge prompt in multiple parts. Please wait until I send "END" '
    "and then combine."
)
FILE_SIZE_MIN = 9000
FILE_SIZE_MAX = 10000


def parse_args():
    parser = argparse.ArgumentParser(description="Create LLM-as-a-Judge prompt.")
    parser.add_argument("-i", "--input", required=True, help="latest | yyyymmddmm")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory for generated laaj-prompt-partN.txt files",
    )
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


def strip_part_markers(text):
    normalized = text.replace("\r\n", "\n").strip()
    normalized = re.sub(
        r'(?m)^I will send the judge prompt in multiple parts\. Please wait until I send "END" and then combine\.\n?',
        "",
        normalized,
    )
    normalized = re.sub(r"(?m)^===BEGIN PART \d+===$\n?", "", normalized)
    normalized = re.sub(r"(?m)^===END PART \d+===$\n?", "", normalized)
    normalized = re.sub(r"(?m)^END\s*$", "", normalized)
    return normalized.strip()


def build_part_text(part_number, body, is_final=False):
    part_begin = f"===BEGIN PART {part_number}==="
    part_end = f"===END PART {part_number}==="
    header = PROMPT_HEADER if part_number == 1 else ""
    text = f"{header}\n{part_begin}\n{body.rstrip()}\n{part_end}"
    if is_final:
        text += "\nEND"
    return text


def split_prompt_into_parts(prompt_text):
    content = strip_part_markers(prompt_text)
    if not content:
        return [build_part_text(1, "", is_final=True)]

    parts = []
    remaining = content
    part_number = 1

    while remaining:
        overhead = (
            (len(PROMPT_HEADER) if part_number == 1 else 0)
            + len(f"\n===BEGIN PART {part_number}===\n")
            + len(f"\n===END PART {part_number}===")
        )
        max_body_chars = FILE_SIZE_MAX - overhead
        min_body_chars = max(1, FILE_SIZE_MIN - overhead)

        if len(remaining) <= max_body_chars:
            chunk = remaining
            remaining = ""
            is_final = True
        else:
            is_final = False
            break_index = remaining.rfind("\n", min_body_chars, max_body_chars + 1)
            if break_index != -1 and break_index > min_body_chars:
                chunk = remaining[:break_index].rstrip()
            else:
                chunk = remaining[:max_body_chars].rstrip()
            remaining = remaining[len(chunk) :].lstrip("\n")

        parts.append(build_part_text(part_number, chunk, is_final=is_final))
        part_number += 1

    if not parts:
        return [build_part_text(1, "", is_final=True)]

    return parts


def main():
    args = parse_args()

    eval_dir = resolve_eval_dir(args.input)
    prompt_text = load_text(f"{eval_dir}/prompt.txt")
    html_text = load_text(f"{eval_dir}/report.html")

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as fp:
        template = fp.read()

    prompt = template.replace("{{PROMPT}}", prompt_text).replace("{{HTML}}", html_text)
    parts = split_prompt_into_parts(prompt)

    os.makedirs(args.output_dir, exist_ok=True)
    for existing_file in glob.glob(
        os.path.join(args.output_dir, "laaj-prompt-part*.txt")
    ):
        os.remove(existing_file)

    for index, part_text in enumerate(parts, start=1):
        output_path = os.path.join(args.output_dir, f"laaj-prompt-part{index}.txt")
        with open(output_path, "w", encoding="utf-8") as fp:
            fp.write(part_text)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
