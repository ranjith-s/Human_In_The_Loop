"""Policy ingestion script for Lab 3.

This script loads policy documents into the ChromaDB collection named
"moderation_policies" and persists the vector DB at ./chroma_db.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "moderation_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _parse_metadata_from_filename(file_path: Path) -> dict[str, Any]:
    """Infer metadata from file name pattern: <category>__<severity>__<id>.txt.

    If the pattern is not present, safe defaults are used.
    """
    stem = file_path.stem
    parts = stem.split("__")

    category = parts[0] if len(parts) > 0 and parts[0] else "general"
    severity = parts[1] if len(parts) > 1 and parts[1] else "medium"
    policy_id = parts[2] if len(parts) > 2 and parts[2] else stem

    return {
        "policy_id": policy_id,
        "category": category,
        "severity": severity,
    }


def _read_policy_document(file_path: Path) -> tuple[str, dict[str, Any]]:
    """Read one policy file and return (content, metadata)."""
    text = file_path.read_text(encoding="utf-8").strip()
    metadata = _parse_metadata_from_filename(file_path)

    # Optional frontmatter support: a leading JSON line can override metadata.
    # Example first line: {"policy_id":"P123","category":"spam","severity":"high"}
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("{") and first_line.endswith("}"):
            try:
                override = json.loads(first_line)
                if isinstance(override, dict):
                    metadata.update({
                        "policy_id": override.get("policy_id", metadata["policy_id"]),
                        "category": override.get("category", metadata["category"]),
                        "severity": override.get("severity", metadata["severity"]),
                    })
                    text = "\n".join(lines[1:]).strip()
            except json.JSONDecodeError:
                # Ignore malformed frontmatter and keep full original text.
                pass

    return text, metadata


def _build_collection(chroma_dir: Path):
    """Create/open the moderation_policies collection with embedding function."""
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"created_by": "lab3_ingest"},
    )


def ingest_policies(data_dir: Path, chroma_dir: Path, reset: bool = False) -> None:
    """Ingest all policy docs from data_dir into ChromaDB."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {data_dir}")

    files = sorted(
        [p for p in data_dir.rglob("*") if p.suffix.lower() in {".txt", ".md"}]
    )
    if len(files) < 15:
        print(
            "Warning: fewer than 15 policy files found. "
            "Lab requirement recommends 15+ documents."
        )

    collection = _build_collection(chroma_dir)

    if reset:
        # Reset by deleting and recreating the collection.
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = _build_collection(chroma_dir)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for file_path in files:
        text, metadata = _read_policy_document(file_path)
        if not text:
            continue

        # Stable IDs make repeated ingest idempotent.
        digest = hashlib.md5(str(file_path).encode("utf-8")).hexdigest()[:12]
        doc_id = f"{metadata['policy_id']}_{digest}"

        ids.append(doc_id)
        documents.append(text)
        metadatas.append(metadata)

    if not documents:
        raise ValueError("No valid policy documents found to ingest.")

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(documents)} policy documents into '{COLLECTION_NAME}'.")
    print(f"ChromaDB persist directory: {chroma_dir}")


def main() -> None:
    """CLI entry point for policy ingestion."""
    parser = argparse.ArgumentParser(description="Ingest Lab 3 moderation policies.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ecommerce_policies"),
        help="Directory containing policy .txt/.md files",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Persist directory for ChromaDB",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate moderation_policies before ingest",
    )

    args = parser.parse_args()
    ingest_policies(args.data_dir, args.chroma_dir, reset=args.reset)


if __name__ == "__main__":
    main()
