"""CPU checks for the fixed diversity-ablation data preparation protocol."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import prepare_diversity_data as builder


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Tokenizer:
    def __init__(self, lengths):
        self.lengths = lengths

    def __call__(self, prompts, *, add_special_tokens, **options):
        assert options["padding"] is False
        assert options["truncation"] is False
        position = 0 if add_special_tokens else 1
        return {"input_ids": [[0] * self.lengths[prompt][position] for prompt in prompts]}


class DiversityDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_direct_selection_matches_nested_selection_with_reindexed_rows(self):
        # Repeated hashes also check that ties preserve original source order
        # when an intermediate selected file is numbered from one again.
        eligible = [(index, digest(f"row-{index % 13}".encode()))
                    for index in range(1, 51) if index % 4]
        intermediate = builder.select_indices(eligible, target=25, seed=20261029)
        row_hashes = dict(eligible)
        nested_eligible = [(index, row_hashes[source_index])
                           for index, source_index in enumerate(intermediate, 1)]
        nested = builder.select_indices(nested_eligible, target=12, seed=20261029)
        mapped_to_source = [intermediate[index - 1] for index in nested]
        direct = builder.select_indices(eligible, target=12, seed=20261029)
        self.assertEqual(mapped_to_source, direct)
        self.assertEqual(direct, sorted(direct))
        self.assertEqual(len(set(direct)), 12)

    def test_eligibility_checks_both_tokenizations_at_exact_boundary(self):
        lengths = {"both-at-limit": (1024, 1024),
                   "filter-too-long": (1025, 1000),
                   "item-too-long": (1000, 1025),
                   "item-at-limit": (1023, 1024)}
        lines = [json.dumps({"prompt": name}).encode() + b"\n" for name in lengths]
        source = self.directory / "source.jsonl"
        source.write_bytes(b"".join(lines))
        eligible, stats = builder.scan_file(
            source, Tokenizer(lengths), lambda row: (row["prompt"], ""), batch_size=2)
        self.assertEqual(eligible, [(1, digest(lines[0])), (4, digest(lines[3]))])
        self.assertEqual(stats["source_rows"], 4)
        self.assertEqual(stats["eligible_rows"], 2)
        self.assertEqual(stats["excluded_source_indices"], [2, 3])
        self.assertEqual(stats["max_filter_tokens"], 1025)
        self.assertEqual(stats["max_item_tokens"], 1025)
        self.assertEqual(stats["source_sha256"], digest(source.read_bytes()))

    def test_publication_preserves_raw_source_order_and_reuses_only_matching_file(self):
        lines = [b'{"problem": "first", "question": "x?"}\r\n',
                 b'{"problem":"second","question":"y?"}\n',
                 b'{"problem": "third", "question": "z?"}\n']
        source = self.directory / "source.jsonl"
        output = self.directory / "selected.jsonl"
        source.write_bytes(b"".join(lines))
        expected = lines[0] + lines[2]
        self.assertTrue(builder.publish_verified(source, output, [3, 1], digest(expected)))
        self.assertEqual(output.read_bytes(), expected)
        timestamp = output.stat().st_mtime_ns
        self.assertFalse(builder.publish_verified(source, output, [1, 3], digest(expected)))
        self.assertEqual(output.stat().st_mtime_ns, timestamp)
        output.write_bytes(b"existing conflicting data\n")
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            builder.publish_verified(source, output, [1, 3], digest(expected))
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(source.read_bytes(), b"".join(lines))

    def test_wrong_pinned_output_hash_never_publishes_or_changes_existing_file(self):
        source = self.directory / "source.jsonl"
        output = self.directory / "selected.jsonl"
        original = b'{"problem":"original","question":"x?"}\n'
        source.write_bytes(original)
        with self.assertRaises(ValueError):
            builder.publish_verified(source, output, [1], "0" * 64)
        self.assertFalse(output.exists())
        output.write_bytes(original)
        timestamp = output.stat().st_mtime_ns
        with self.assertRaises(ValueError):
            builder.publish_verified(source, output, [1], "0" * 64)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(output.stat().st_mtime_ns, timestamp)
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
