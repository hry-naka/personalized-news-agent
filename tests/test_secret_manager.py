import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from secret_manager import get_secret, _resolve_secret_provider


class SecretManagerTests(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.copy()
        self.original_cwd = os.getcwd()
        self.tempdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tempdir.name)
        os.chdir(self.workdir)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)
        os.chdir(self.original_cwd)
        self.tempdir.cleanup()

    def write_config(self, content: str):
        (self.workdir / "config.yaml").write_text(content, encoding="utf-8")

    def test_auto_prefers_gcp_then_env_then_config(self):
        self.write_config("""
secret_provider:
  type: auto

gemini_api_key: config-value
smtp_pass: config-smtp
huggingface_token: config-hf
""".strip())
        os.environ["GEMINI_API_KEY"] = "env-gemini"
        os.environ["SMTP_PASS"] = "env-smtp"
        os.environ["HUGGINGFACE_TOKEN"] = "env-hf"

        with mock.patch(
            "secret_manager._load_from_gcp",
            side_effect=lambda name: {
                "gemini_api_key": "gcp-gemini",
                "smtp_pass": "gcp-smtp",
                "huggingface_token": "gcp-hf",
            }.get(name),
        ):
            self.assertEqual(get_secret("gemini_api_key"), "gcp-gemini")
            self.assertEqual(get_secret("smtp_pass"), "gcp-smtp")
            self.assertEqual(get_secret("huggingface_token"), "gcp-hf")

    def test_auto_falls_back_to_env(self):
        self.write_config("""
secret_provider:
  type: auto

gemini_api_key: config-gemini
""".strip())
        os.environ["GEMINI_API_KEY"] = "env-gemini"

        with mock.patch("secret_manager._load_from_gcp", return_value=None):
            self.assertEqual(get_secret("gemini_api_key"), "env-gemini")

    def test_auto_falls_back_to_config_with_warning(self):
        self.write_config("""
secret_provider:
  type: auto

gemini_api_key: config-gemini
""".strip())
        with mock.patch("secret_manager._load_from_gcp", return_value=None), mock.patch(
            "secret_manager._load_from_env", return_value=None
        ):
            self.assertEqual(get_secret("gemini_api_key"), "config-gemini")

    def test_config_value_warning(self):
        self.write_config("""
secret_provider:
  type: auto

gemini_api_key: legacy-gemini
""".strip())
        with mock.patch("secret_manager._load_from_gcp", return_value=None), mock.patch(
            "secret_manager._load_from_env", return_value=None
        ):
            self.assertEqual(get_secret("gemini_api_key"), "legacy-gemini")


if __name__ == "__main__":
    unittest.main()
