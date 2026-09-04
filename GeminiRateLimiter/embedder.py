# embedder.py
import time
import json
from typing import List
from google.genai import types
from google.genai.errors import ClientError


class RPDQuotaExceeded(Exception):
    """Custom exception for RPD quota exceeded."""

    pass


class GeminiEmbedder:
    """
    Gemini Embedding wrapper with TPM control and retry handling.

    This class provides:
        - Single-text embedding
        - Batch embedding (list of texts)
        - TPM-aware request throttling
        - 429 retryDelay handling

    Attributes:
        client: Google GenAI client instance.
        tpm: TpmWindow instance for token-per-minute control.
        retry: RetryHandler instance for 429 retryDelay handling.
        model (str): Embedding model name.
        max_retries (int): Maximum retry attempts for 429 errors.
    """

    def __init__(
        self,
        client,
        tpm,
        retry,
        model: str = "gemini-2.0-flash-embed",
        max_retries: int = 5,
    ):
        """
        Initialize the embedder.

        Args:
            client: Google GenAI client instance.
            tpm (TpmWindow): Token-per-minute window manager.
            retry (RetryHandler): Retry handler for 429 errors.
            model (str): Embedding model name.
            max_retries (int): Maximum retry attempts.
        """
        self.client = client
        self.tpm = tpm
        self.retry = retry
        self.model = model
        self.max_retries = max_retries

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Embedding vector.
        """
        tokens_needed = self.tpm.estimate_tokens(text)
        embeddings = self._embed_content(text, tokens_needed)
        return embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        # Estimate total tokens for TPM control
        tokens_needed = sum(self.tpm.estimate_tokens(t) for t in texts)
        contents_list = [
            types.Content(parts=[types.Part.from_text(text=t)]) for t in texts
        ]
        return self._embed_content(contents_list, tokens_needed)

    def _embed_content(self, contents, tokens_needed: int) -> List[List[float]]:
        """Embed content while applying throttling and retry handling."""
        time.sleep(1.2)  # Small delay to avoid immediate rate limit issues
        self.tpm.wait_if_needed(tokens_needed)

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = self.client.models.embed_content(
                    model=self.model, contents=contents
                )
                vectors = [emb.values for emb in response.embeddings]
                self.tpm.add(tokens_needed)
                return vectors
            except ClientError as e:
                cause = "UNKNOWN"
                quota_id = None

                try:
                    error_json = e.response.json()
                    details = error_json.get("error", {}).get("details", [])

                    for d in details:
                        if (
                            d.get("@type")
                            == "type.googleapis.com/google.rpc.QuotaFailure"
                        ):
                            violations = d.get("violations", [])
                            if violations:
                                quota_metric = violations[0].get("quotaMetric")
                                quota_id = violations[0].get("quotaId")
                                quota_value = violations[0].get("quotaValue")
                                cause = f"{quota_metric} (id={quota_id}, value={quota_value})"
                        else:
                            cause = str(d)
                except Exception as parse_err:
                    cause = f"PARSE_ERROR: {parse_err}"
                if quota_id and quota_id.startswith("EmbedContentRequestsPerDay"):
                    print(
                        "INFO: 429 detected (RPD). No point in retrying. Aborting immediately."
                    )
                    raise RPDQuotaExceeded(
                        "ERROR: RPD quota exceeded. Try again tomorrow."
                    )
                print(f"INFO: 429 detected. Cause = {cause}")

                self.retry.sleep_for_retry(e)
                retry_count += 1

        raise RuntimeError("ERROR: Maximum retry attempts exceeded.")
