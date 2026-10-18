#!/usr/bin/env python3
"""Prepare private human-review worksheets; never decode media or create labels.

Only aggregate counters reach stdout. Existing manifests and ledgers are copied
byte for byte into a new private run directory, never corrected or overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from evaluate import read_registry, validate_manifest
from prepare_dataset import DatasetError, UNKNOWN_GROUPS


MAX_JSON_BYTES = 16 * 1024 * 1024
EVALUATION_SPLITS = {'validation', 'holdout'}


def known(value):
    return isinstance(value, str) and value.strip().casefold() not in UNKNOWN_GROUPS


def private_write(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def json_snapshot(path):
    """Read each input once so audit, digest and copied bytes cannot diverge."""
    if not path.is_file():
        raise DatasetError('Expected a local JSON file')
    with path.open('rb') as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise DatasetError('Review JSON exceeds the size limit')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid_constant(_value):
        raise DatasetError('Non-finite JSON value')

    return data, json.loads(data.decode('utf-8-sig'), object_pairs_hook=unique_object,
                            parse_constant=invalid_constant)


def linked_closure(sha_seeds, group_seeds, groups_by_sha):
    """Follow all SHA/known-group links, without inventing group annotations."""
    linked_sha, linked_groups = set(sha_seeds), set(group_seeds)
    changed = True
    while changed:
        previous = (len(linked_sha), len(linked_groups))
        for digest, groups in groups_by_sha.items():
            if digest in linked_sha or linked_groups.intersection(groups):
                linked_sha.add(digest)
                linked_groups.update(groups)
        changed = previous != (len(linked_sha), len(linked_groups))
    return linked_sha, linked_groups


def applicability_rows(registry, sources):
    rows = []
    for source in sources or [None]:
        for rule in registry['rules']:
            rows.append({
                'video_id': source['video_id'] if source else None,
                'rule_id': rule['rule_id'],
                'process_and_scope': rule['requirement'],
                'necessary_conditions': rule['required_conditions'],
                'confirmed_conditions': [],
                'unknown_conditions': rule['required_conditions'],
                'exceptions_to_check': rule['disqualifiers'],
                'exception_review': 'unknown',
                'required_continuous_interval': {
                    'start_sec': None, 'end_sec': None, 'duration_sec': None,
                    'coverage_verified': False,
                    'requirement': 'Full relevant operation, including before/during/after; player marks are navigation only.'},
                'applicability_decision': 'unknown',
                'decision_basis': 'No human scene review supplied. Public normative text verified separately.',
                'public_source': rule['source'],
                'human_review_json_local_path': None,
                'reviewer_local_id': None,
                'reviewed': False,
            })
    return rows


def audit(manifest, ledger):
    videos = validate_manifest(manifest)
    by_sha = defaultdict(list)
    by_id = {}
    for video in videos:
        if video['video_id'] in by_id:
            raise DatasetError('Duplicate source identifier')
        by_id[video['video_id']] = video
        by_sha[video['sha256'].lower()].append(video)
    if not isinstance(ledger, dict):
        raise DatasetError('Invalid original exposure ledger')
    reservations = ledger.get('sha_exposure_reservations_local_only')
    if (type(ledger.get('schema_version')) is not int or ledger['schema_version'] != 1
            or not isinstance(reservations, list) or not reservations):
        raise DatasetError('Missing exposure reservations')
    groups_by_sha = {digest: {v['group_id'] for v in aliases if known(v['group_id'])}
                     for digest, aliases in by_sha.items()}
    protected_roots = {digest for digest, aliases in by_sha.items()
                       if any(v['split'] in EVALUATION_SPLITS for v in aliases)}
    exposed = set()
    groups = set()
    conflicts = []
    for reservation in reservations:
        digest = reservation['sha256'].lower()
        if digest in exposed or digest not in by_sha:
            raise DatasetError('Duplicate or missing reserved SHA')
        exposed.add(digest)
        aliases = reservation['aliases']
        if not isinstance(aliases, list) or not aliases:
            raise DatasetError('Missing reserved aliases')
        actual = {v['video_id'] for v in by_sha[digest]}
        recorded = {v['video_id'] for v in aliases}
        if actual != recorded or len(recorded) != len(aliases):
            conflicts.append({'kind': 'ledger_alias_mismatch', 'sha256': digest,
                              'recorded': sorted(recorded), 'actual': sorted(actual)})
        recorded_groups = {a['group_id'] for a in aliases if known(a.get('group_id'))}
        groups.update(recorded_groups)
        groups_by_sha[digest].update(recorded_groups)
        if any(a.get('original_split') in EVALUATION_SPLITS for a in aliases):
            protected_roots.add(digest)
    direct_exposed = set(exposed)
    exposed, groups = linked_closure(exposed, groups, groups_by_sha)
    # A train alias or a third video from a known linked group is still related
    # to protected evaluation media. Logging the conflict alone must not queue it
    # for rule tuning. Include historical ledger links if group labels changed.
    protected_sha, _ = linked_closure(protected_roots, set(), groups_by_sha)
    ambiguous = set()
    for digest, aliases in by_sha.items():
        signatures = {json.dumps({'label': v['label'], 'events': sorted(
            v['events'], key=lambda e: (e['timecode_sec'], e.get('rule_id') or ''))}, sort_keys=True)
            for v in aliases}
        if len(signatures) > 1:
            ambiguous.add(digest)
            conflicts.append({'kind': 'identical_bytes_differing_annotations',
                              'sha256': digest, 'original_records': copy.deepcopy(aliases),
                              'resolution': None})
        splits = {v['split'] for v in aliases} - {'unassigned'}
        if len(splits) > 1:
            conflicts.append({'kind': 'sha_cross_split', 'sha256': digest})
        if digest in exposed and splits.intersection({'validation', 'holdout'}):
            conflicts.append({'kind': 'exposure_in_evaluation', 'sha256': digest})
    group_splits = defaultdict(set)
    for v in videos:
        if known(v['group_id']) and v['split'] != 'unassigned':
            group_splits[v['group_id']].add(v['split'])
    conflicts.extend({'kind': 'group_cross_split', 'group_id': group}
                     for group, splits in group_splits.items() if len(splits) > 1)
    source_map = []
    for digest, aliases in sorted(by_sha.items()):
        reasons = []
        if digest in exposed:
            reasons.append('training_exposure_or_linked_group')
        if digest in ambiguous:
            reasons.append('unresolved_official_annotations')
        if any(not known(v['group_id']) for v in aliases):
            reasons.append('unknown_scenario_group')
        source_map.append({'sha256': digest, 'original_records': copy.deepcopy(aliases),
                           'evaluation_blockers': reasons,
                           'review_blockers': ['validation_or_holdout_linked_sha_or_group']
                           if digest in protected_sha else [],
                           'independence_verified': False,
                           'human_group_basis': None, 'human_camera_basis': None})
    review_candidates = [v for v in videos
                         if (v['sha256'].lower() in exposed or v['split'] == 'train')
                         and v['split'] not in EVALUATION_SPLITS]
    review_sources = [v for v in review_candidates if v['sha256'].lower() not in protected_sha]
    aggregate = {
        'source_entries': len(videos), 'unique_sha': len(by_sha),
        'duplicate_sha_groups': sum(len(a) > 1 for a in by_sha.values()),
        'annotation_conflict_sha_groups': len(ambiguous),
        'direct_exposed_sha': len(direct_exposed),
        'direct_exposed_aliases': sum(len(by_sha[s]) for s in direct_exposed),
        'excluded_by_exposure_sha': len(exposed),
        'excluded_by_exposure_sources': sum(len(by_sha[s]) for s in exposed),
        'protected_evaluation_component_sha': len(protected_sha),
        'review_candidates_blocked_by_evaluation': len(review_candidates) - len(review_sources),
        'review_queue_sources': len(review_sources),
        'declared_group_ids': len({v['group_id'] for v in videos if known(v['group_id'])}),
        'sources_with_unknown_group': sum(not known(v['group_id']) for v in videos),
        'sources_with_unknown_camera': sum(not known(v['camera_id']) for v in videos),
        'human_verified_groups': None, 'unknown_group_count': None,
        'unresolved_conflicts': len(conflicts), 'independent_evaluation_authorized': False,
        'media_hashes_recomputed': False,
    }
    aggregate['matches_documented_counts'] = all(aggregate[k] == n for k, n in {
        'source_entries': 42, 'unique_sha': 40, 'duplicate_sha_groups': 2,
        'annotation_conflict_sha_groups': 1, 'direct_exposed_sha': 4,
        'direct_exposed_aliases': 5}.items())
    # Review only prior exposure/train, outside all protected evaluation links.
    return {'sources': source_map, 'conflicts': conflicts}, aggregate, review_sources


def prepare(rules, output, manifest_path=None, ledger_path=None):
    _, registry = json_snapshot(rules)
    read_registry(registry)
    if bool(manifest_path) != bool(ledger_path):
        raise DatasetError('Manifest and original ledger must be supplied together')
    aggregate = {'data_audit_executed': False, 'rules_reviewed': 0,
                 'rules_confirmed': 0, 'rules_unknown': len(registry['rules']),
                 'rules_not_applicable': 0, 'rules_proposed_for_enablement': []}
    mapping, sources = {'status': 'missing_manifest_and_original_ledger', 'sources': [], 'conflicts': []}, []
    snapshots = []
    if manifest_path:
        manifest_bytes, manifest = json_snapshot(manifest_path)
        ledger_bytes, ledger = json_snapshot(ledger_path)
        mapping, counters, sources = audit(manifest, ledger)
        aggregate.update(counters, data_audit_executed=True)
        aggregate['manifest_matches_original_ledger_digest'] = (
            hashlib.sha256(manifest_bytes).hexdigest() == ledger.get('source_manifest_sha256'))
        snapshots = [('manifest.original.local.json', manifest_bytes),
                     ('training_exposure_ledger.original.local.json', ledger_bytes)]
    worksheet = {
        'kind': 'cashwatch_applicability_worksheet', 'schema_version': 1,
        'is_ground_truth': False, 'template_only': not bool(sources),
        'rows': applicability_rows(registry, sources)}
    # Discover malformed rule fields before creating/copying any output files.
    for payload in (worksheet, mapping, aggregate):
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    # Every invocation creates a new directory; no overwrite of human work.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, data in snapshots:
        with (output / name).open('xb') as stream:
            os.chmod(output / name, 0o600)
            stream.write(data)
    private_write(output / 'applicability.local.json', worksheet)
    private_write(output / 'groups_aliases.local.json', mapping)
    private_write(output / 'summary.aggregate.json', aggregate)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rules', type=Path, default=Path('config/rules.json'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.rules, args.out, args.manifest, args.ledger)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        # Do not copy source paths, annotation text or arbitrary exception data.
        print('Review preparation failed. Check inputs locally; no existing files were replaced.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
