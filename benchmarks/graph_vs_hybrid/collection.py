"""Mapping retrieved passages back to the gold keys of the corpus entries.

``MultimodalSample.from_dict`` drops the ``id`` field when reading a JSONL, so chunk ids
are assigned at index time (``str(hash(text))``) and cannot be predicted from the corpus
file. The gold key therefore travels in ``metadata.file_path``, and the collection is read
once per evaluation to build the chunk-id → gold-key map.
"""

import logging
from typing import Any, Dict, Iterable, List

from .config import GOLD_PREFIX, BenchConfig

logger = logging.getLogger(__name__)

_PAGE_SIZE = 1000


def _iter_rows(
    client, collection_name: str, fields: List[str]
) -> Iterable[Dict[str, Any]]:
    try:
        iterator = client.query_iterator(
            collection_name=collection_name,
            filter="",
            output_fields=fields,
            batch_size=_PAGE_SIZE,
        )
    except Exception as error:
        logger.debug(f"query_iterator unavailable ({error}); falling back to paging")
        offset = 0
        while True:
            page = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=fields,
                limit=_PAGE_SIZE,
                offset=offset,
            )
            if not page:
                return
            yield from page
            offset += len(page)
        return

    try:
        while True:
            page = iterator.next()
            if not page:
                return
            yield from page
    finally:
        iterator.close()


def indexed_passages(config: BenchConfig) -> List[Dict[str, str]]:
    """``{id, key, text}`` for every indexed passage carrying a gold key.

    The lexical baseline scores the passages *Milvus holds*, not the ones the corpus file
    holds, and keyed by the same chunk ids — otherwise it would be answering a different
    question from the retrievers it is compared against.
    """
    from pymilvus import MilvusClient

    client = MilvusClient(uri=config.milvus_uri, db_name="bench")
    try:
        rows = []
        for row in _iter_rows(
            client, config.index.collection_name, ["id", "file_path", "text"]
        ):
            path = row.get("file_path") or ""
            if path.startswith(GOLD_PREFIX):
                rows.append(
                    {
                        "id": row["id"],
                        "key": path[len(GOLD_PREFIX) :],
                        "text": row.get("text") or "",
                    }
                )
    finally:
        client.close()

    if not rows:
        raise ValueError(
            f"No document in collection '{config.index.collection_name}' carries an "
            f"'{GOLD_PREFIX}' file_path. Was the collection built by this benchmark's "
            "`prepare` + `index` stages?"
        )
    logger.info(f"Read {len(rows)} indexed passages")
    return rows


def key_by_chunk_id(config: BenchConfig) -> Dict[str, str]:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=config.milvus_uri, db_name="bench")
    try:
        mapping = {}
        for row in _iter_rows(
            client, config.index.collection_name, ["id", "file_path"]
        ):
            path = row.get("file_path") or ""
            if path.startswith(GOLD_PREFIX):
                mapping[row["id"]] = path[len(GOLD_PREFIX) :]
    finally:
        client.close()

    if not mapping:
        raise ValueError(
            f"No document in collection '{config.index.collection_name}' carries an "
            f"'{GOLD_PREFIX}' file_path. Was the collection built by this benchmark's "
            "`prepare` + `index` stages?"
        )
    logger.info(f"Resolved gold keys for {len(mapping)} indexed passages")
    return mapping
