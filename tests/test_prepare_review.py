"""Synthetic checks for exposure closure and preservation of disputed records."""
import copy
import hashlib
import io
import json
import stat
import sys
from pathlib import Path

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_review import audit, prepare
import prepare_review as REVIEW


def source(name, digest, group='unassigned', split='unassigned', timestamp=2):
    return dict(video_id=name, path=name + '.mov', sha256=digest * 64,
                duration_sec=10, fps=30, width=10, height=10, label=1,
                events=[dict(timecode_sec=timestamp, rule_id=None)],
                group_id=group, camera_id='unassigned', split=split)


def fixture():
    videos = [source('a', 'a'), source('alias', 'a', 'g'),
              source('b', 'b', 'g'), source('b-alias', 'b', 'h'),
              source('c', 'c', 'h', 'holdout'),
              source('d', 'd', timestamp=3), source('d-alias', 'd', timestamp=7)]
    manifest = dict(schema_version=1, videos=videos)
    ledger = dict(schema_version=1, sha_exposure_reservations_local_only=[dict(
        sha256='a' * 64, aliases=[dict(video_id=v['video_id'], group_id=v['group_id'])
                                  for v in videos[:2]])])
    return manifest, ledger


def check_transitive_exposure_unknown_groups_and_official_records_preserved():
    manifest, ledger = fixture()
    before = copy.deepcopy(manifest)
    mapping, aggregate, queue = audit(manifest, ledger)
    assert manifest == before
    assert aggregate['direct_exposed_aliases'] == 2
    assert aggregate['excluded_by_exposure_sources'] == 5
    assert aggregate['annotation_conflict_sha_groups'] == 1
    assert aggregate['unknown_group_count'] is None
    assert aggregate['independent_evaluation_authorized'] is False
    assert all(v['split'] != 'holdout' for v in queue)
    conflict = next(c for c in mapping['conflicts'] if c['kind'] == 'identical_bytes_differing_annotations')
    assert [v['events'][0]['timecode_sec'] for v in conflict['original_records']] == [3, 7]
    assert any(c['kind'] == 'exposure_in_evaluation' for c in mapping['conflicts'])


def check_ledger_alias_mismatch_is_reported_not_silently_repaired():
    manifest, ledger = fixture()
    ledger['sha_exposure_reservations_local_only'][0]['aliases'].pop()
    mapping, _, _ = audit(manifest, ledger)
    assert any(c['kind'] == 'ledger_alias_mismatch' for c in mapping['conflicts'])


def check_private_outputs_preserve_original_bytes_and_refuse_overwrite(tmp_path):
    manifest, ledger = fixture()
    mp, lp = tmp_path / 'm.json', tmp_path / 'l.json'
    mp.write_text(json.dumps(manifest))
    ledger['source_manifest_sha256'] = hashlib.sha256(mp.read_bytes()).hexdigest()
    lp.write_text(json.dumps(ledger))
    rules = Path(__file__).resolve().parents[1] / 'config/rules.json'
    out = tmp_path / 'review'
    summary = prepare(rules, out, mp, lp)
    assert summary['manifest_matches_original_ledger_digest'] is True
    assert (out / 'manifest.original.local.json').read_bytes() == mp.read_bytes()
    assert (out / 'training_exposure_ledger.original.local.json').read_bytes() == lp.read_bytes()
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in out.iterdir())
    rows = json.loads((out / 'applicability.local.json').read_text())['rows']
    assert all(r['confirmed_conditions'] == [] and r['applicability_decision'] == 'unknown' for r in rows)
    assert all(r['required_continuous_interval']['duration_sec'] is None for r in rows)
    with unittest.TestCase().assertRaises(FileExistsError):
        prepare(rules, out, mp, lp)


def check_missing_inputs_create_only_explicit_templates(tmp_path):
    rules = Path(__file__).resolve().parents[1] / 'config/rules.json'
    summary = prepare(rules, tmp_path / 'review')
    assert summary['data_audit_executed'] is False
    assert summary['rules_unknown'] == 3
    assert summary['rules_proposed_for_enablement'] == []
    table = json.loads((tmp_path / 'review/applicability.local.json').read_text())
    assert table['template_only'] is True
    assert len(table['rows']) == 3


class ReviewTests(unittest.TestCase):
    def test_exposure(self):
        check_transitive_exposure_unknown_groups_and_official_records_preserved()

    def test_alias_mismatch(self):
        check_ledger_alias_mismatch_is_reported_not_silently_repaired()

    def test_private_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            check_private_outputs_preserve_original_bytes_and_refuse_overwrite(Path(directory))

    def test_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            check_missing_inputs_create_only_explicit_templates(Path(directory))

    def test_review_queue_excludes_transitive_evaluation_aliases_and_historical_ledger_groups(self):
        for protected_split in ('validation', 'holdout', 'historical_holdout'):
            with self.subTest(protected_split=protected_split):
                videos = [
                    source('bridge-other-alias', 'b', 'second-group', 'train'),
                    source('protected', 'a', split='train' if protected_split == 'historical_holdout' else protected_split),
                    source('protected-train-alias', 'a', 'first-group', 'train'),
                    source('bridge', 'b', 'first-group', 'train'),
                    source('reserved', 'c', 'second-group'),
                    source('historical-group-neighbor', 'd', 'old-group', 'train'),
                    source('safe-development', 'e', 'separate-group', 'train'),
                ]
                manifest = {'schema_version': 1, 'videos': videos}
                ledger = {'schema_version': 1, 'sha_exposure_reservations_local_only': [{
                    'sha256': 'c' * 64, 'aliases': [{
                        'video_id': 'reserved', 'group_id': 'old-group',
                        'original_split': 'holdout' if protected_split == 'historical_holdout' else 'unassigned',
                    }],
                }]}
                original_manifest, original_ledger = copy.deepcopy(manifest), copy.deepcopy(ledger)
                mapping, summary, queue = audit(manifest, ledger)
                self.assertEqual([video['video_id'] for video in queue], ['safe-development'])
                self.assertEqual(summary['protected_evaluation_component_sha'], 4)
                self.assertEqual(summary['review_queue_sources'], 1)
                self.assertGreaterEqual(summary['review_candidates_blocked_by_evaluation'], 5)
                self.assertTrue(all(record['review_blockers'] for record in mapping['sources']
                                    if record['sha256'] != 'e' * 64))
                self.assertEqual(manifest, original_manifest)
                self.assertEqual(ledger, original_ledger)

    def test_audit_digest_and_copies_share_one_snapshot_even_if_input_files_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ledger = fixture()
            manifest_bytes = (json.dumps(manifest, indent=3) + '\n').encode()
            ledger['source_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
            ledger_bytes = (json.dumps(ledger, indent=4) + '\n').encode()
            mp, lp = root / 'manifest.json', root / 'ledger.json'
            mp.write_bytes(manifest_bytes)
            lp.write_bytes(ledger_bytes)

            def changing_inputs(manifest_snapshot, ledger_snapshot):
                result = audit(manifest_snapshot, ledger_snapshot)
                # Simulate another local editor after parsing; never use real data.
                mp.write_text('{"later_edit":true}')
                lp.write_text('{"later_edit":true}')
                return result

            rules = Path(__file__).resolve().parents[1] / 'config/rules.json'
            with patch.object(REVIEW, 'audit', side_effect=changing_inputs):
                summary = prepare(rules, root / 'review', mp, lp)
            self.assertTrue(summary['manifest_matches_original_ledger_digest'])
            self.assertEqual((root / 'review/manifest.original.local.json').read_bytes(), manifest_bytes)
            self.assertEqual((root / 'review/training_exposure_ledger.original.local.json').read_bytes(), ledger_bytes)

    def test_malformed_rules_fail_before_creating_output_or_copying_private_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, ledger = fixture()
            mp, lp, rp = root / 'manifest.json', root / 'ledger.json', root / 'rules.json'
            mp.write_text(json.dumps(manifest))
            lp.write_text(json.dumps(ledger))
            rp.write_text(json.dumps({'schema_version': 1, 'rules': [{
                'rule_id': 'synthetic_rule', 'title': 'Synthetic rule', 'requirement': 'Synthetic scope',
                'disqualifiers': [], 'source': {},  # required_conditions deliberately missing
            }]}))
            with self.assertRaises(KeyError):
                prepare(rp, root / 'review', mp, lp)
            self.assertFalse((root / 'review').exists())
            self.assertEqual(json.loads(mp.read_text()), manifest)
            self.assertEqual(json.loads(lp.read_text()), ledger)

    def test_invalid_json_diagnostics_do_not_reveal_private_values_and_read_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private_marker = 'synthetic-private-annotation-marker'
            mp, lp = root / (private_marker + '.json'), root / 'ledger.json'
            mp.write_text(json.dumps({private_marker: 1})[:-1] + ',"' + private_marker + '":2}')
            lp.write_text('{}')
            rules = Path(__file__).resolve().parents[1] / 'config/rules.json'
            stdout, stderr = io.StringIO(), io.StringIO()
            arguments = ['prepare_review.py', '--manifest', str(mp), '--ledger', str(lp),
                         '--rules', str(rules), '--out', str(root / 'review')]
            with patch.object(sys, 'argv', arguments), redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(REVIEW.main(), 1)
            self.assertEqual(stdout.getvalue(), '')
            self.assertNotIn(private_marker, stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())
            self.assertFalse((root / 'review').exists())
            with patch.object(REVIEW, 'MAX_JSON_BYTES', 8), self.assertRaisesRegex(ValueError, 'size limit'):
                REVIEW.json_snapshot(mp)


if __name__ == '__main__':
    unittest.main()
