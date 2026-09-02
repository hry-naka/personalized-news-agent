# tpm_window.py
import time
from collections import deque
import sentencepiece as spm
from huggingface_hub import hf_hub_download


class TpmWindow:
    """
    Token-based sliding window rate limiter for Gemini Embedding API.

    This class loads the Gemma SentencePiece tokenizer and uses it to estimate
    token counts locally without consuming API requests. It maintains a sliding
    window of token usage over the last `window_sec` seconds and ensures that
    the total token usage does not exceed the configured TPM limit.

    Attributes:
        sp (SentencePieceProcessor): Loaded tokenizer model.
        limit (int): Maximum allowed tokens per minute (default: 30000).
        window_sec (int): Sliding window duration in seconds (default: 60).
        history (deque): Stores tuples of (timestamp, tokens).
    """

    def __init__(
        self,
        tokenizer_model_path="tokenizer.model",
        token="",
        limit=30000,
        window_sec=60,
    ):
        """
        Initialize the TPM window manager.

        Args:
            tokenizer_model_path (str): Path to Gemma tokenizer.model file.
            token(str): HuggingFace token for downloading the tokenizer model.
            limit (int): Token-per-minute limit.
            window_sec (int): Sliding window duration in seconds.

        Raises:
            RuntimeError: If tokenizer.model cannot be loaded.
        """
        try:
            model_path = hf_hub_download(
                repo_id="google/gemma-2b", filename=tokenizer_model_path, token=token
            )
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer model '{tokenizer_model_path}': {e}"
            )

        self.limit = limit
        self.window_sec = window_sec
        self.history = deque()

    def estimate_tokens(self, text):
        """
        Estimate token count using Gemma SentencePiece tokenizer.

        Args:
            text (str): Input text.

        Returns:
            int: Estimated token count.
        """
        if not text:
            return 0
        return len(self.sp.encode_as_ids(text))

    def _cleanup_history(self):
        """
        Remove entries older than the sliding window.
        """
        now = time.time()
        while self.history and self.history[0][0] < now - self.window_sec:
            self.history.popleft()

    def wait_if_needed(self, tokens_needed):
        """
        Check current token usage and sleep if the next request would exceed TPM.

        Args:
            tokens_needed (int): Tokens required for the upcoming request.

        Notes:
            This method may sleep for up to `window_sec` seconds if the TPM limit
            is close to being exceeded.
        """
        self._cleanup_history()

        now = time.time()
        current_tokens = sum(t for ts, t in self.history)
        print(
            f"INFO: Current tokens in window: {current_tokens}, Tokens needed: {tokens_needed}, Limit: {self.limit}"
        )

        if current_tokens + tokens_needed > self.limit:
            # Oldest timestamp determines when tokens will expire
            oldest_ts = self.history[0][0]
            sleep_sec = self.window_sec - (now - oldest_ts)
            if sleep_sec > 0:
                print(
                    f"INFO: TPM limit approaching. Sleeping {sleep_sec:.1f} seconds..."
                )
                time.sleep(sleep_sec)

            # Re-check recursively after sleeping
            self.wait_if_needed(tokens_needed)

    def add(self, tokens):
        """
        Add token usage to the sliding window after a successful API call.

        Args:
            tokens (int): Number of tokens consumed by the request.
        """
        self.history.append((time.time(), tokens))
