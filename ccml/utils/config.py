"""Configuration management utilities for loading and merging YAML configs."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML config file

    Returns:
        Dictionary with config values

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not config_file:
        return {}

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_config_with_args(
    config: Dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser = None,
) -> argparse.Namespace:
    """
    Merge config file with command-line arguments.

    CLI arguments take precedence over config file values.
    Config values are only applied if the argument wasn't explicitly provided via CLI.

    Args:
        config: Dictionary from YAML config file
        args: argparse.Namespace from command-line
        parser: Optional ArgumentParser to extract flag-to-dest mappings

    Returns:
        argparse.Namespace with merged values
    """
    # Build flag-to-dest mapping from parser if provided
    flag_to_dest = {}
    if parser:
        for action in parser._actions:
            for option in action.option_strings:
                if option.startswith("--"):
                    flag_name = option.lstrip("-").replace("-", "_")
                    dest_name = action.dest
                    flag_to_dest[flag_name] = dest_name

    # Get CLI arguments that were explicitly provided
    cli_provided = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            # Extract argument name (handle both --arg and --arg=value)
            arg_name = arg.split("=")[0].lstrip("-").replace("-", "_")
            # Use dest name if we have the mapping, otherwise use arg name
            dest_name = flag_to_dest.get(arg_name, arg_name)
            cli_provided.add(dest_name)

    # Normalize config keys (hyphens to underscores)
    normalized_config = {}
    for key, value in config.items():
        normalized_key = key.replace("-", "_")
        normalized_config[normalized_key] = value

    # Apply config values only if not explicitly provided via CLI
    for key, value in normalized_config.items():
        if hasattr(args, key) and key not in cli_provided:
            existing_value = getattr(args, key)
            if existing_value is not None and value is not None:
                try:
                    value = type(existing_value)(value)
                except (ValueError, TypeError):
                    pass
            setattr(args, key, value)

    return args
