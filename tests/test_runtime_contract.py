"""Identity integration tests with synthetic metadata only; no ML or bank media."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cashwatch import runtime_contract as contracts
from cashwatch.settings import Settings


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "private-checkpoint-location"
        self.model.mkdir()
        self.manifest = {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct", "resolved_revision": "a" * 40,
            "downloaded_at_utc": "2026-10-17T00:00:00Z", "license": "apache-2.0",
            "files": {
                "config.json": {"bytes": 10, "sha256": "1" * 64},
                "tokenizer_config.json": {"bytes": 20, "sha256": "2" * 64},
                "model.safetensors": {"bytes": 30, "sha256": "3" * 64},
            },
        }
        self.write_manifest()
        self.rules = self.root / "private-rules.json"
        self.rules.write_text(json.dumps({"schema_version": 1, "rules": []}))
        self.sources = self.root / "synthetic-package"
        self.sources.mkdir()
        (self.sources / "__init__.py").write_text('"""Synthetic sources."""\n')
        (self.sources / "engine.py").write_text("VALUE = 1\n")
        source_patch = patch.object(contracts, "_PACKAGE_ROOT", self.sources)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        self.settings = Settings(api_key="x" * 24, model_dir=self.model, rules_path=self.rules, device="cuda")

    def write_manifest(self):
        (self.model / "cashwatch_model_manifest.json").write_text(json.dumps(self.manifest))

    def build(self):
        return contracts.build_runtime_contract(self.settings)

    def test_complete_declared_identity_does_not_read_weight_files_or_claim_readiness(self):
        # There are deliberately no real weight/config files. Completeness means
        # declared identity, while model loading and active-rule readiness are separate.
        original_open = Path.open

        def guard_weight_read(path, *args, **kwargs):
            if path.suffix == ".safetensors":
                raise AssertionError("Unexpected weight read")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guard_weight_read):
            result = self.build()
        self.assertTrue(result["complete"])
        self.assertEqual(result["issues"], [])
        self.assertIs(result["weights_hashes_verified"], False)
        self.assertEqual(contracts.validate_runtime_contract(result, require_complete=True), result)

    def test_missing_provenance_is_incomplete_and_strict_validation_fails_closed(self):
        (self.model / "cashwatch_model_manifest.json").unlink()
        result = self.build()
        self.assertFalse(result["complete"])
        self.assertEqual(result["model"], {"model_id": None, "revision": None, "declared_files_sha256": {}})
        self.assertEqual(result["issues"], ["model_provenance_unavailable"])
        self.assertEqual(contracts.validate_runtime_contract(result), result)
        with self.assertRaisesRegex(ValueError, "^runtime_contract_incomplete$"):
            contracts.validate_runtime_contract(result, require_complete=True)

    def test_relocation_keys_download_time_and_manifest_order_do_not_change_identity(self):
        first = self.build()
        moved = self.root / "different-private-location"
        shutil.copytree(self.model, moved)
        manifest = copy.deepcopy(self.manifest)
        manifest["downloaded_at_utc"] = "2099-01-01T00:00:00Z"
        manifest["files"] = dict(reversed(list(manifest["files"].items())))
        manifest["unrelated_absolute_path"] = "/private/ignored/download/path"
        (moved / "cashwatch_model_manifest.json").write_text(json.dumps(manifest, indent=4))
        moved_rules = self.root / "relocated-rules.json"
        shutil.copyfile(self.rules, moved_rules)
        moved_sources = self.root / "relocated-source-package"
        shutil.copytree(self.sources, moved_sources)
        with patch.object(contracts, "_PACKAGE_ROOT", moved_sources):
            second = contracts.build_runtime_contract(replace(
                self.settings, api_key="z" * 30, data_dir=self.root / "other-runtime",
                model_dir=moved, rules_path=moved_rules, retention_hours=2,
            ))
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(contracts.compare_runtime_contracts(first, second), [])
        serialized = json.dumps(second)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("z" * 30, serialized)
        self.assertNotIn("downloaded_at", serialized)

    def test_every_runtime_setting_is_part_of_identity(self):
        original = self.build()
        changes = {
            "device": "cpu", "sample_fps": 2.0, "window_seconds": 11.0,
            "overlap_seconds": 1.0, "max_frames_per_window": 20,
            "max_new_tokens": 1024, "max_image_edge": 384,
            "max_duration_sec": 160.0, "analysis_timeout_sec": 270.0,
            "max_inflight": 2, "max_upload_mb": 80,
        }
        for name, value in changes.items():
            with self.subTest(setting=name):
                changed = contracts.build_runtime_contract(replace(self.settings, **{name: value}))
                self.assertNotEqual(original["fingerprint"], changed["fingerprint"])
                self.assertEqual(contracts.compare_runtime_contracts(original, changed), [f"settings.{name}"])

    def test_model_revision_declared_hash_rules_and_sources_are_independent_identity_fields(self):
        first = self.build()
        self.manifest["resolved_revision"] = "b" * 40
        self.write_manifest()
        revision_changed = self.build()
        self.assertEqual(contracts.compare_runtime_contracts(first, revision_changed), ["model.revision"])
        self.manifest["files"]["model.safetensors"]["sha256"] = "4" * 64
        self.write_manifest()
        hashes_changed = self.build()
        self.assertEqual(contracts.compare_runtime_contracts(revision_changed, hashes_changed), ["model.declared_files_sha256"])
        self.rules.write_text(self.rules.read_text() + "\n")
        rules_changed = self.build()
        self.assertEqual(contracts.compare_runtime_contracts(hashes_changed, rules_changed), ["rules_sha256"])
        (self.sources / "engine.py").write_text("VALUE = 2\n")
        sources_changed = self.build()
        self.assertEqual(contracts.compare_runtime_contracts(rules_changed, sources_changed), ["pipeline_sha256"])

    def test_corrupted_fingerprint_is_rejected(self):
        result = self.build()
        result["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "^runtime_contract_fingerprint_mismatch$"):
            contracts.validate_runtime_contract(result)

    def test_complete_issues_and_weight_verification_cannot_be_overstated(self):
        for field, value in (("complete", False), ("issues", ["private-input-value"]), ("weights_hashes_verified", True)):
            with self.subTest(field=field):
                result = self.build()
                result[field] = value
                with self.assertRaisesRegex(ValueError, "^runtime_contract_invalid$"):
                    contracts.validate_runtime_contract(result)

    def test_unsafe_and_colliding_normalized_model_filenames_make_provenance_incomplete(self):
        original = copy.deepcopy(self.manifest)
        for name in ("../private-file", "/absolute-file", "C:\\private-file", "nested//file"):
            with self.subTest(name=name):
                self.manifest = copy.deepcopy(original)
                self.manifest["files"][name] = {"bytes": 1, "sha256": "5" * 64}
                self.write_manifest()
                self.assertEqual(self.build()["issues"], ["model_provenance_unavailable"])
        self.manifest = copy.deepcopy(original)
        self.manifest["files"]["nested/file.json"] = {"bytes": 1, "sha256": "5" * 64}
        self.manifest["files"]["nested\\file.json"] = {"bytes": 1, "sha256": "5" * 64}
        self.write_manifest()
        self.assertFalse(self.build()["complete"])

    def test_duplicate_json_keys_and_oversized_provenance_fail_closed(self):
        path = self.model / "cashwatch_model_manifest.json"
        path.write_text('{"files":{},"files":{}}')
        self.assertEqual(self.build()["issues"], ["model_provenance_unavailable"])
        path.write_bytes(b"x" * 1025)
        with patch.object(contracts, "_JSON_LIMIT", 1024):
            self.assertEqual(self.build()["issues"], ["model_provenance_unavailable"])

    def test_branch_revision_and_invalid_hash_cannot_establish_model_identity(self):
        for field, value in (("resolved_revision", "main"), ("resolved_revision", None)):
            with self.subTest(value=value):
                self.manifest[field] = value
                self.write_manifest()
                self.assertFalse(self.build()["complete"])
        self.manifest["resolved_revision"] = "a" * 40
        self.manifest["files"]["model.safetensors"]["sha256"] = "private-invalid-digest"
        self.write_manifest()
        self.assertFalse(self.build()["complete"])

    def test_missing_sources_and_invalid_rules_do_not_crash_capabilities(self):
        self.rules.write_text('{"schema_version":1,"rules":"invalid"}')
        with patch.object(contracts, "_PACKAGE_ROOT", self.root / "absent-package"):
            result = self.build()
        self.assertEqual(result["issues"], ["pipeline_identity_unavailable", "rules_identity_unavailable"])
        self.assertFalse(result["complete"])
        self.assertEqual(contracts.validate_runtime_contract(result), result)

    def test_invalid_settings_are_explicitly_incomplete_not_nonfinite_json(self):
        result = contracts.build_runtime_contract(replace(self.settings, sample_fps=float("nan")))
        self.assertEqual(result["issues"], ["runtime_settings_invalid"])
        self.assertTrue(all(value is None for value in result["settings"].values()))
        self.assertEqual(contracts.validate_runtime_contract(result), result)
        json.dumps(result, allow_nan=False)

    def test_malformed_payload_errors_do_not_include_private_values(self):
        for malformed in (None, [], {"private-filename": "private-value"}):
            with self.subTest(payload_type=type(malformed).__name__):
                with self.assertRaisesRegex(ValueError, "^runtime_contract_invalid$"):
                    contracts.validate_runtime_contract(malformed)
        result = self.build()
        result["model"]["declared_files_sha256"]["private-new-name"] = "invalid"
        with self.assertRaisesRegex(ValueError, "^runtime_contract_invalid$"):
            contracts.compare_runtime_contracts(self.build(), result)


if __name__ == "__main__":
    unittest.main()
