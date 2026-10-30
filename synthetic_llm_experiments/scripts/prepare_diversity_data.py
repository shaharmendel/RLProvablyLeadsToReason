"""Prepare the two four-level diversity comparisons from pinned source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

OPS = (10, 13, 17, 18, 19, 20)
ARMS = ((10, 13, 17, 20), (17, 18, 19, 20))
SEED, LIMIT, ROWS = 20261029, 1024, 102299
HARD20_SHA256 = "5792eb48a3136e7fb5221116a4376f73fc739f1e1a15bb214c4a3b024e1c2e8a"
OUTPUT_SHA256 = {
    10: "7442a6a7387d4a7324de628c2c0250808b842b3cf472941f33790a2f8e27688e",
    13: "c0c580e7a4d3f6de5aeaaa8d2ee1ad631fc824fd07063c63247636c7e6c85a38",
    17: "7d1220bfbc3457661c562ba530ea00efd04de2aade8e2490e6903900ac8b0824",
    18: "3bd2a5437150ec1d41424009b76a699fc705f797083af8e78b8091bf1dac7398",
    19: "515bb07e473e295d4ada6cc9b84738f46c8cbd537b3871339c63f52fa67cb186",
    20: "37cb829eeabc598076e360fa32364204e1fb11fd2116af31d963d036af623e85",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_key(row):
    return hashlib.sha256(json.dumps(
        [row["problem"], row["question"]], ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def scan_file(path, tokenizer, compose_prompt, limit=LIMIT, batch_size=256):
    """Check both the original filter and item tokenization, without truncation."""
    eligible, excluded, batch = [], [], []
    digest = hashlib.sha256()
    rows = max_filter = max_item = 0

    def consume():
        nonlocal max_filter, max_item
        prompts = [compose_prompt(row)[0] for _, _, row in batch]
        options = dict(padding=False, truncation=False, return_attention_mask=False,
                       return_token_type_ids=False)
        filtered = tokenizer(prompts, add_special_tokens=True, **options)["input_ids"]
        items = tokenizer(prompts, add_special_tokens=False, **options)["input_ids"]
        for (index, raw, _), first, second in zip(batch, filtered, items, strict=True):
            max_filter, max_item = max(max_filter, len(first)), max(max_item, len(second))
            if len(first) <= limit and len(second) <= limit:
                eligible.append((index, hashlib.sha256(raw).hexdigest()))
            else:
                excluded.append(index)
        batch.clear()

    with Path(path).open("rb") as handle:
        for rows, raw in enumerate(handle, 1):
            digest.update(raw)
            batch.append((rows, raw, json.loads(raw)))
            if len(batch) >= batch_size:
                consume()
    if batch:
        consume()
    return eligible, {
        "source_rows": rows, "source_sha256": digest.hexdigest(),
        "eligible_rows": len(eligible), "excluded_source_indices": excluded,
        "max_filter_tokens": max_filter, "max_item_tokens": max_item,
    }


def select_indices(eligible, target=ROWS, seed=SEED):
    if len(eligible) < target:
        raise ValueError(f"Only {len(eligible)} eligible rows; need {target}")
    def rank(item):
        index, row_hash = item
        return hashlib.sha256(f"{seed}:{row_hash}".encode("ascii")).digest(), index
    return sorted(index for index, _ in sorted(eligible, key=rank)[:target])


def publish(staged, output, expected_sha256):
    """Create atomically or reuse identical bytes; never replace existing data."""
    if file_hash(staged) != expected_sha256:
        raise ValueError(f"Prepared bytes differ from pinned SHA256: {output}")
    try:
        if os.name == "nt":
            os.rename(staged, output)
        else:
            os.link(staged, output)
        return True
    except FileExistsError:
        if not output.is_file() or file_hash(output) != expected_sha256:
            raise FileExistsError(f"Refusing conflicting existing output: {output}")
        return False


def publish_verified(source, output, indices, expected_sha256, validation_keys=()):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source file")
    wanted = set(indices)
    if len(wanted) != len(indices) or any(index < 1 for index in wanted):
        raise ValueError("Selected indices must be distinct positive source rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".diversity-", dir=output.parent)
    staged, written = Path(name), 0
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as handle:
            for index, raw in enumerate(handle, 1):
                if index in wanted:
                    if prompt_key(json.loads(raw)) in validation_keys:
                        raise ValueError("Selected training prompt overlaps Hard20 validation")
                    destination.write(raw)
                    written += 1
            destination.flush()
            os.fsync(destination.fileno())
        if written != len(wanted):
            raise ValueError("Selected source row is missing")
        return publish(staged, output, expected_sha256)
    finally:
        staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    bundle = args.bundle.resolve()
    output = args.output_dir.resolve() if args.output_dir else bundle / "derived_data/appendix-1024/four"
    runtime = Path(os.environ.get("RUNTIME_DIR", bundle / ".runtime")).resolve()
    sys.path.insert(0, str(runtime))
    from transformers import AutoTokenizer
    from verl.dataset import compose_prompt

    model = bundle / "models/extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/base"
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("The pinned fast tokenizer is required")
    pinned = json.loads((bundle / "manifests/generated-datasets.json").read_text())
    validation = bundle / "data/composition/val/op20-200.jsonl"
    if file_hash(validation) != HARD20_SHA256:
        raise ValueError("Hard20 source differs from its pinned SHA256")
    included, stats = scan_file(validation, tokenizer, compose_prompt, batch_size=args.batch_size)
    if stats["source_rows"] != 200 or stats["excluded_source_indices"] != [12, 67, 92]:
        raise ValueError("Expected the same 197 eligible Hard20 examples (excluded rows 12, 67, 92)")
    validation_keys = {prompt_key(json.loads(raw)) for raw in validation.read_bytes().splitlines()}
    manifest = {
        "selection_seed": SEED, "selection_method": "lowest SHA256(seed:row_sha256), ties by source index",
        "source_index_base": 1, "prompt_limit": LIMIT, "rows_per_level": ROWS,
        "arms": [list(ops) for ops in ARMS], "rows_per_arm": 4 * ROWS,
        "tokenizer_sha256": file_hash(model / "tokenizer.json"),
        "dataset_code_sha256": file_hash(runtime / "verl/dataset.py"),
        "validation": {**stats, "included_source_indices": [index for index, _ in included]},
        "files": {},
    }
    for op in OPS:
        source = bundle / f"data/composition/heldout/op{op}-137500.jsonl"
        if file_hash(source) != pinned[source.name]:
            raise ValueError(f"Raw source differs from pinned SHA256: {source}")
        print(f"[prepare] filtering op{op}", flush=True)
        eligible, stats = scan_file(source, tokenizer, compose_prompt, batch_size=args.batch_size)
        if stats["source_rows"] != 137500 or stats["source_sha256"] != pinned[source.name]:
            raise ValueError(f"Unexpected or changed source: {source}")
        selected = select_indices(eligible)
        target = output / f"op{op}-{ROWS}.jsonl"
        created = publish_verified(source, target, selected, OUTPUT_SHA256[op], validation_keys)
        manifest["files"][target.name] = {
            **stats, "source": source.relative_to(bundle).as_posix(), "op": op,
            "rows": ROWS, "sha256": OUTPUT_SHA256[op], "selected_source_indices": selected,
        }
        print(f"[ok] {'created' if created else 'verified'} {target.name}: {ROWS} rows", flush=True)
    raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", dir=output)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw_manifest)
        publish(staged, output / "preparation-manifest.json", hashlib.sha256(raw_manifest).hexdigest())
    finally:
        staged.unlink(missing_ok=True)
    print(f"[ok] both arms: {4 * ROWS} rows; shared Hard20 mass 1/4; validation 197/200")


if __name__ == "__main__":
    main()
