import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

try:
    from google.cloud import secretmanager
except ImportError:  # pragma: no cover - optional dependency for GCP mode
    secretmanager = None

SECRET_ENV_NAMES = {
    "gemini_api_key": "GEMINI_API_KEY",
    "smtp_pass": "SMTP_PASS",
    "huggingface_token": "HUGGINGFACE_TOKEN",
}


def _config_path() -> Path:
    return Path("config.yaml")


def _load_yaml_config() -> dict[str, Any]:
    config_path = _config_path()
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if isinstance(config, dict):
            return config
    except Exception:
        pass
    return {}


def _provider_type() -> str:
    config = _load_yaml_config()
    secret_provider = config.get("secret_provider") or {}
    if isinstance(secret_provider, dict):
        return str(secret_provider.get("type", "auto")).lower()
    return "auto"


def _resolve_secret_provider(name: str) -> str:
    provider = _provider_type()
    if provider == "auto":
        return "auto"
    if provider in {"gcp", "env", "config"}:
        return provider
    return "auto"


def _warn_legacy_config(name: str) -> None:
    logging.warning(
        "%s was loaded from config.yaml. Consider migrating to .env or GCP Secret Manager.",
        name,
    )


def _load_from_env(name: str) -> str | None:
    env_name = SECRET_ENV_NAMES.get(name)
    if not env_name:
        return None

    load_dotenv()
    value = os.getenv(env_name)
    if value is not None and value.strip():
        return value.strip()
    return None


def _load_from_config(name: str) -> str | None:
    value = _load_yaml_config().get(name)
    if value is not None and str(value).strip():
        _warn_legacy_config(name)
        return str(value).strip()
    return None


def _load_from_gcp(name: str) -> str | None:
    if secretmanager is None:
        return None

    project_id = (
        _load_yaml_config().get("gcp", {}).get("project_id")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )

    if not project_id:
        return None

    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_path = f"projects/{project_id}/secrets/{name}/versions/latest"
        response = client.access_secret_version(request={"name": secret_path})
        value = response.payload.data.decode("utf-8")
        if value and value.strip():
            return value.strip()
    except Exception as e:
        print("GCP secret error")
        print("secret =", name)
        print(repr(e))
        raise


def _require_with_provider(name: str, provider: str) -> str:
    if provider == "gcp":
        value = _load_from_gcp(name)
        if value is None:
            raise RuntimeError(
                f"Secret '{name}' could not be loaded from GCP Secret Manager. "
                "Set GOOGLE_CLOUD_PROJECT and ensure the secret exists."
            )
        return value

    if provider == "env":
        value = _load_from_env(name)
        if value is None:
            raise RuntimeError(
                f"Secret '{name}' could not be loaded from .env. "
                f"Set {SECRET_ENV_NAMES.get(name, name)} in the environment or .env file."
            )
        return value

    if provider == "config":
        value = _load_from_config(name)
        if value is None:
            raise RuntimeError(
                f"Secret '{name}' could not be loaded from config.yaml. "
                "Add the value or switch to .env/GCP Secret Manager."
            )
        return value

    raise ValueError(f"Unsupported secret provider type: {provider}")


def get_secret(name: str) -> str:
    """Resolve a secret from the configured provider hierarchy."""
    if not name or name not in SECRET_ENV_NAMES:
        raise ValueError(f"Unsupported secret name: {name}")

    provider = _resolve_secret_provider(name)
    if provider == "auto":
        for loader in (_load_from_gcp, _load_from_env, _load_from_config):
            value = loader(name)
            if value is not None and value.strip():
                return value
        raise RuntimeError(
            f"Secret '{name}' could not be found in GCP Secret Manager, .env, or config.yaml."
        )

    return _require_with_provider(name, provider)
