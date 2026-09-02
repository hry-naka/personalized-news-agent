# retry.py
import re
import time


class RetryHandler:
    """
    Handle retry logic for Gemini API 429 errors.

    Gemini's ClientError does not provide structured JSON for RetryInfo.
    Instead, the error message is embedded inside a single large string.
    This class extracts retryDelay values using robust regular expressions
    and sleeps accordingly.

    Attributes:
        default_sleep (int): Fallback sleep duration when retryDelay cannot be extracted.
        verbose (bool): Whether to print debug information.
    """

    def __init__(self, default_sleep: int = 60, verbose: bool = True):
        """
        Initialize the retry handler.

        Args:
            default_sleep (int): Fallback sleep duration in seconds.
            verbose (bool): Print debug information if True.
        """
        self.default_sleep = default_sleep
        self.verbose = verbose

    def _extract_retry_delay(self, raw: str) -> float:
        """
        Extract retryDelay from Gemini's error message.

        Gemini's 429 errors may contain:
            - "retryDelay': '2s'"
            - "Please retry in 2.471993926s."

        Args:
            raw (str): Raw error message (e.args[0]).

        Returns:
            float: Retry delay in seconds, or None if not found.
        """
        # Pattern 1: retryDelay': '2s'
        m = re.search(r"retryDelay': '(\d+)s", raw)
        if m:
            return float(m.group(1))

        # Pattern 2: Please retry in 2.47s
        m2 = re.search(r"Please retry in ([0-9.]+)s", raw)
        if m2:
            return float(m2.group(1))

        return None

    def sleep_for_retry(self, error: Exception) -> None:
        """
        Sleep for the duration specified by the retryDelay in the error message.

        Args:
            error (Exception): Gemini ClientError instance.
        """
        raw = str(error.args[0])
        delay = self._extract_retry_delay(raw)

        if delay is None:
            delay = self.default_sleep
            if self.verbose:
                print(
                    f"INFO: 429 detected without retryDelay. Sleeping {delay} seconds..."
                )
        else:
            if self.verbose:
                print(f"INFO: 429 detected. Sleeping {delay} seconds...")

        time.sleep(delay)
