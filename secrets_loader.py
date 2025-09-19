"""Backward compatibility shim for secret loading."""

from infa_to_pyspark.secrets_loader import load_secrets_into_env

__all__ = ["load_secrets_into_env"]
