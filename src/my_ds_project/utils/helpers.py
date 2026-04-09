"""
Helper utility functions.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)

    if path.suffix.lower() in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
