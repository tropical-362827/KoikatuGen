import json
import os
from typing import Any


def infer_data_prefix(parquet_path: str) -> str:
    filename = os.path.basename(parquet_path)
    if filename.startswith("kks_"):
        return "kks"
    if filename.startswith("kk_"):
        return "kk"
    raise ValueError(f"Unable to infer data prefix from parquet path: {parquet_path}")


def save_training_metadata(
    outdir: str, parquet_path: str, extra_metadata: dict[str, Any] | None = None
) -> None:
    metadata = {
        "parquet": parquet_path,
        "data_prefix": infer_data_prefix(parquet_path),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    metadata_path = os.path.join(outdir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_training_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    metadata_path = os.path.join(
        os.path.dirname(os.path.abspath(checkpoint_path)), "training_metadata.json"
    )
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_template_path(checkpoint_path: str) -> str:
    metadata = load_training_metadata(checkpoint_path)
    if metadata is None:
        raise ValueError(
            "Could not determine data prefix. Retrain to write training metadata."
        )
    data_prefix = metadata["data_prefix"]

    template_path = os.path.join("data", "templates", f"{data_prefix}_default.png")
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"Template not found: {template_path}. Prepare the template for '{data_prefix}'."
        )

    return template_path
