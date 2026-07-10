# Personalized News Agent

A simple Python script that collects news articles from Google News RSS feeds, curates them using the Gemini API, and sends a formatted email report.

## Features

- Fetches news articles from Google News RSS for specified search queries
- Resolves Google News redirect URLs to real article URLs
- Uses Gemini API to curate and summarize articles based on a user profile
- Sends the curated report via SMTP email

## Requirements

- Python 3.9 or higher
- An SMTP server for sending email
- A Gemini API key

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following values:

```env
GEMINI_API_KEY=your_gemini_api_key
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASS=your_smtp_password
TO_EMAIL=recipient@example.com
USER_PROFILE_BASE=your_user_profile_text
```

- `GEMINI_API_KEY`: Gemini API key for generating article curation results.
- `SMTP_SERVER`: SMTP host for sending email.
- `SMTP_PORT`: SMTP port (default is 587).
- `SMTP_USER`: SMTP username or sender email address.
- `SMTP_PASS`: SMTP password.
- `TO_EMAIL`: Recipient email address.
- `USER_PROFILE_BASE`: Text describing the user profile used to guide article selection.

## Usage

Run the script with a mail subject string argument:

```bash
python news-agent.py "Daily News Digest"
```

The script will:

1. Fetch news articles from Google News RSS.
2. Curate an article list using Gemini.
3. Send the generated report in HTML format to the configured recipient.

## Notes

- Make sure the `.env` file is present and contains all required settings before running the script.
- If RSS fetch or Gemini generation fails, the script prints an error message and exits gracefully.
