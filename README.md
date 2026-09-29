# Personalized News Agent

This project fetches RSS news, ranks and curates articles with Gemini, and sends a personalized digest by email. It also includes an evaluation pipeline for prompt and article quality measurement.

## Features

- RSS article aggregation from multiple feeds configured in `config.yaml`
- Gemini-based article selection and summarization
- HTML report generation via `main_prompt.txt`
- SMTP delivery for the final digest
- Optional `--eval` output for reproducibility and review
- Secret management with `auto`, `gcp`, `env`, and `config` provider modes

## Requirements

- Python 3.12+
- A Gemini API key
- An SMTP account/server
- Optional: Google Cloud Project + Secret Manager for `gcp` mode (`pip install google-cloud-secret-manager`)
- macOS, Linux, or WSL2

## Installation

```bash
git clone https://github.com/hry-naka/personalized-news-agent.git
cd personalized-news-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Secret management

This project supports secret resolution in the following priority order when `secret_provider.type: auto` is used:

1. GCP Secret Manager
2. `.env`
3. `config.yaml`

The app also keeps backward compatibility with the legacy keys in `config.yaml`:

- `gemini_api_key`
- `smtp_pass`
- `huggingface_token`

When a secret is loaded from `config.yaml`, the code logs a warning to encourage migration.

### `secret_provider` configuration

```yaml
secret_provider:
  type: auto
```

Valid values:

- `auto`
- `gcp`
- `env`
- `config`

### `auto` mode

```yaml
secret_provider:
  type: auto
```

Resolved order:

1. GCP Secret Manager
2. `.env`
3. `config.yaml`

If none of the sources provide the value, the program exits with an explicit error.

### `gcp` mode

Use this when you want to read from Google Cloud Secret Manager only.

```yaml
secret_provider:
  type: gcp
```

Expected secret names:

- `gemini_api_key`
- `smtp_pass`
- `huggingface_token`

The code uses Application Default Credentials (ADC). Ensure the environment is authenticated before running the agent.

Example:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### `env` mode

Use this when you want to load from a `.env` file only.

```yaml
secret_provider:
  type: env
```

The keys must be defined as environment variables, normally via `.env`:

```env
GEMINI_API_KEY=...
SMTP_PASS=...
HUGGINGFACE_TOKEN=...
```

The project uses `python-dotenv` and calls `load_dotenv()` automatically.

### `config` mode

Use this only for compatibility or local test setups.

```yaml
secret_provider:
  type: config
```

This resolves from `config.yaml` directly, but emits a warning because it is a legacy path.

---

## Configuration

### `config.yaml.example`

Use [config.yaml.example](config.yaml.example) as the template. It intentionally omits real secrets and includes the new `secret_provider` section.

### `.env.example`

Create a local `.env` from [.env.example](.env.example) and keep it outside of source control.

```env
GEMINI_API_KEY=your_gemini_key
SMTP_PASS=your_smtp_password
HUGGINGFACE_TOKEN=your_huggingface_token
```

### Example `config.yaml`

```yaml
secret_provider:
  type: auto

gemini_llm_model: "gemini-2.5-flash"
gemini_embedding_model: "models/gemini-embedding-2"

smtp_server: "127.0.0.1"
smtp_port: 587
smtp_user: "your_email@example.com"
# legacy supported for backward compatibility
smtp_pass: ""
to_email: "destination@example.com"

rss_channels:
  - name: "日本経済新聞"
    query: "日本経済新聞"
    count: 20

num_output_articles: "10"
num_counter_articles: "2"
curate_language: "same"

language_instructions_same: |
  Write the [Reason] and [Summary] in the same language as the original article.

tokenizer_model_path: "tokenizer.model"
huggingface_token: ""
```

> The legacy YAML keys remain supported for compatibility, but they are not the preferred storage location.

---

## GCP Secret Manager setup

1. Enable Secret Manager for your project (Refer to https://docs.cloud.google.com/secret-manager/docs/reference/libraries#client-libraries-install-python).
2. Create secrets with the same names:
   - `gemini_api_key`
   - `smtp_pass`
   - `huggingface_token`
3. Grant the service account or your local ADC identity permission to access them.
4. Set the project ID:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
```

5. Configure:

```yaml
secret_provider:
  type: gcp
```

---

## `.env` setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key
SMTP_PASS=your_smtp_password
HUGGINGFACE_TOKEN=your_huggingface_token
```

Then configure:

```yaml
secret_provider:
  type: env
```

The app loads `.env` automatically via `python-dotenv`.

---

## Usage

Run the agent with the default secret resolution order:

```bash
python news-agent.py "Daily News Digest"
```

With evaluation data output:

```bash
python news-agent.py "Debug Run" --eval
```

---

## Evaluation

The repo includes the evaluation tools:

- `eval-prompt.py` for embedding-based quantitative evaluation (`-i` and `-o` are required)
- `create-laaj-prompt.py` for LLM-as-a-Judge prompt generation
- `summarize-eval.py` for aggregation

`eval-prompt.py` uses the configured secret provider for the Gemini and Hugging Face credentials. The prompt-generation and CSV-summary scripts do not call external APIs and do not require secrets.

Example evaluation commands, after a run has produced evaluation data:

```bash
python eval-prompt.py -i latest -o eval-data/eval-prompt.csv -m all
python create-laaj-prompt.py -i latest -o laaj-results
python summarize-eval.py -i eval-data/eval-prompt.csv -f 202609010000 -t 202609302359 -o summarized-eval.csv
```

The `--eval` option is intended to save `prompt.txt`, `report.html`, `articles.json`, and `meta.json` under a timestamped directory in `eval-data/`. Since the current `news-agent.py` exits before generation, it cannot create these files until the blocker above is fixed.

Run the available secret-provider unit tests with:

```bash
python -m unittest discover -s tests
```

---

## Notes on backward compatibility

The legacy fields below are still accepted:

```yaml
gemini_api_key: "..."
smtp_pass: "..."
huggingface_token: "..."
```

However, the code now treats them as a fallback source. When values originate from config, a warning is printed to indicate the recommended migration path.

---

## Project structure

```text
personalized-news-agent/
├── news-agent.py
├── eval-prompt.py
├── secret_manager.py
├── config.yaml
├── config.yaml.example
├── .env.example
├── main_prompt.txt
├── user_profile.txt
├── requirements.txt
├── README.md
├── tests/
├── eval-data/
├── laaj-results/
├── GeminiRateLimiter/
└── run-news.sh
```