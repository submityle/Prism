//! Workspace payload CAS lifecycle tests. Ordering is explicit: no sleeps or
//! scheduling assumptions are needed to establish lease and GC boundaries.

use prism_asset_build_core::{
    AssetId, ContentHash, EditableAsset, EditableAssetError, EditablePayload,
    EditablePayloadGcReport, EditablePayloadRef, EditableWorkspace, EditableWorkspaceEdit,
    EditableWorkspaceMutation as Mutation, SourceRecord, MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES,
    MAX_EDITABLE_WORKSPACE_ENCODED_BYTES as BUDGET,
};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::{tempdir, TempDir};

const PAYLOADS: &str = ".prism-editable-payloads";
const HISTORY: &str = ".prism-editable-history";

fn asset(id: u128, payload: impl Into<EditablePayload>) -> EditableAsset {
    EditableAsset::new(
        AssetId(id),
        7,
        0,
        Vec::new(),
        SourceRecord::embedded("source.raw", "test/raw", b"source".to_vec()).unwrap(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        payload,
        None,
    )
    .unwrap()
}

fn hash(value: &EditableAsset) -> ContentHash {
    ContentHash::digest(&value.encode().unwrap())
}

fn put(path: &str, previous: Option<&EditableAsset>, value: EditableAsset) -> Mutation {
    Mutation::Put {
        relative_path: path.into(),
        expected_hash: previous.map(hash),
        replacement: value,
    }
}

fn remove(value: &EditableAsset) -> Mutation {
    Mutation::Remove {
        asset: value.asset,
        expected_hash: hash(value),
    }
}

struct Fixture {
    directory: TempDir,
    workspace: EditableWorkspace,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempdir().unwrap();
        let workspace = EditableWorkspace::open(directory.path()).unwrap();
        Self {
            directory,
            workspace,
        }
    }

    fn root(&self) -> &Path {
        self.directory.path()
    }

    fn reopen(&self) -> EditableWorkspace {
        EditableWorkspace::open(self.root()).unwrap()
    }

    fn blob(&self, reference: EditablePayloadRef) -> PathBuf {
        let hex = reference.content_hash.to_hex();
        self.root().join(PAYLOADS).join(&hex[..2]).join(&hex[2..])
    }

    fn pin(&self, id: u128) -> PathBuf {
        self.root().join(HISTORY).join(format!("{id:032x}.pin"))
    }

    fn store_unrooted(&self, bytes: &[u8]) -> EditablePayloadRef {
        let lease = self
            .workspace
            .store_payload(bytes, bytes.len() as u64)
            .unwrap();
        lease.reference()
    }

    fn create(&self, name: &str, value: &EditableAsset) {
        self.workspace
            .mutate(vec![put(name, None, value.clone())], BUDGET)
            .unwrap();
    }

    fn collect(&self) -> EditablePayloadGcReport {
        self.workspace.collect_payloads(Duration::ZERO).unwrap()
    }

    fn collect_result(&self) -> Result<EditablePayloadGcReport, EditableAssetError> {
        self.reopen().collect_payloads(Duration::ZERO)
    }

    fn assert_package_unchanged(&self, name: &str, before: &[u8]) {
        assert_eq!(fs::read(self.root().join(name)).unwrap(), before);
        // Open runs transaction recovery: a rejected edit must not publish on
        // the next open either.
        drop(self.reopen());
        assert_eq!(fs::read(self.root().join(name)).unwrap(), before);
    }
}

fn regular_files(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    if !root.exists() {
        return result;
    }
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let kind = entry.file_type().unwrap();
        if kind.is_dir() {
            result.extend(regular_files(&entry.path()));
        } else {
            result.push(entry.path());
        }
    }
    result.sort();
    result
}

/// Exercises short reads, one recoverable interruption, and optionally a hard
/// read failure *after* bytes have already reached the CAS staging file.
struct Chunked<'a> {
    bytes: &'a [u8],
    interrupted: bool,
    fail_at_end: bool,
}

impl Read for Chunked<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if !self.interrupted {
            self.interrupted = true;
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.bytes.is_empty() && self.fail_at_end {
            return Err(io::Error::other("injected source read failure"));
        }
        let count = output.len().min(self.bytes.len()).min(3);
        output[..count].copy_from_slice(&self.bytes[..count]);
        self.bytes = &self.bytes[count..];
        Ok(count)
    }
}

#[test]
fn streaming_short_reads_interruption_dedup_and_empty_objects_round_trip() {
    let fixture = Fixture::new();
    let bytes = b"streamed authoring bytes";
    let first = fixture
        .workspace
        .store_payload(
            Chunked {
                bytes,
                interrupted: false,
                fail_at_end: false,
            },
            bytes.len() as u64,
        )
        .unwrap();
    let second = fixture
        .reopen()
        .store_payload(bytes.as_slice(), bytes.len() as u64)
        .unwrap();
    assert_eq!(first.reference(), second.reference());
    assert_eq!(first.reference().content_hash, ContentHash::digest(bytes));
    assert_eq!(first.reference().byte_len, bytes.len() as u64);
    assert_eq!(
        regular_files(&fixture.root().join(PAYLOADS)),
        [fixture.blob(first.reference())]
    );
    let empty = fixture.workspace.store_payload(io::empty(), 0).unwrap();
    assert_eq!(
        empty.reference(),
        EditablePayloadRef {
            content_hash: ContentHash::digest(b""),
            byte_len: 0,
        }
    );
    let snapshot = fixture.workspace.snapshot(BUDGET).unwrap();
    assert_eq!(
        &*snapshot
            .resolve_payload(
                &EditablePayload::External(first.reference()),
                bytes.len() as u64
            )
            .unwrap(),
        bytes
    );
    assert!(snapshot
        .resolve_payload(&EditablePayload::External(empty.reference()), 0)
        .unwrap()
        .is_empty());
    assert_eq!(regular_files(&fixture.root().join(PAYLOADS)).len(), 2);
}

#[test]
fn failed_reads_and_limit_overruns_publish_neither_prefix_nor_temporary_file() {
    let fixture = Fixture::new();
    let existing = fixture.store_unrooted(b"unrelated valid object");
    let before = regular_files(&fixture.root().join(PAYLOADS));
    let prefix = b"accepted prefix before read error";
    let failed = fixture.workspace.store_payload(
        Chunked {
            bytes: prefix,
            interrupted: false,
            fail_at_end: true,
        },
        1024,
    );
    assert!(matches!(
        failed,
        Err(EditableAssetError::Io {
            operation: "read authoring payload",
            kind: io::ErrorKind::Other
        })
    ));
    assert_eq!(regular_files(&fixture.root().join(PAYLOADS)), before);
    assert!(!fixture
        .blob(EditablePayloadRef {
            content_hash: ContentHash::digest(prefix),
            byte_len: prefix.len() as u64
        })
        .exists());

    // Three bytes are staged before the next short read exceeds the limit.
    assert!(fixture
        .workspace
        .store_payload(
            Chunked {
                bytes: b"abcdef",
                interrupted: false,
                fail_at_end: false,
            },
            4
        )
        .is_err());
    assert!(fixture.workspace.store_payload(&b"x"[..], 0).is_err());
    assert!(matches!(
        fixture
            .workspace
            .store_payload(io::empty(), MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES + 1),
        Err(EditableAssetError::PayloadBudgetExceeded { .. })
    ));
    assert_eq!(regular_files(&fixture.root().join(PAYLOADS)), before);
    assert_eq!(
        fs::read(fixture.blob(existing)).unwrap(),
        b"unrelated valid object"
    );
    // No failed operation leaves a shared lease behind.
    assert_eq!(fixture.collect().sweep.deleted, 1);
}

#[test]
fn writer_clone_protects_unpublished_bytes_across_workspace_handles_until_final_drop() {
    let fixture = Fixture::new();
    let writer = fixture
        .workspace
        .store_payload(&b"unpublished"[..], 64)
        .unwrap();
    let reference = writer.reference();
    let clone = writer.clone();
    let collector = fixture.reopen();
    drop(writer);
    let deferred = collector.collect_payloads(Duration::ZERO).unwrap();
    assert!(deferred.deferred_due_to_active_readers);
    assert_eq!(deferred.sweep.deleted, 0);
    assert!(fixture.blob(reference).is_file());
    drop(clone);
    let collected = collector.collect_payloads(Duration::ZERO).unwrap();
    assert!(!collected.deferred_due_to_active_readers);
    assert_eq!(collected.sweep.deleted, 1);
    assert!(!fixture.blob(reference).exists());
}

#[test]
fn snapshot_clone_resolves_deleted_package_payload_until_its_final_drop() {
    let fixture = Fixture::new();
    let reference = fixture.store_unrooted(b"snapshot retained authoring");
    let original = asset(1, EditablePayload::External(reference));
    fixture.create("one.prism", &original);
    let snapshot = fixture.workspace.snapshot(BUDGET).unwrap();
    let clone = snapshot.clone();
    drop(snapshot);
    let other = fixture.reopen();
    other.mutate(vec![remove(&original)], BUDGET).unwrap();
    assert!(!fixture.root().join("one.prism").exists());
    assert!(
        other
            .collect_payloads(Duration::ZERO)
            .unwrap()
            .deferred_due_to_active_readers
    );
    assert_eq!(
        &*clone
            .resolve_payload(&clone[&original.asset].authoring_payload, 1024)
            .unwrap(),
        b"snapshot retained authoring"
    );
    drop(clone);
    assert_eq!(
        other
            .collect_payloads(Duration::ZERO)
            .unwrap()
            .sweep
            .deleted,
        1
    );
    assert!(!fixture.blob(reference).exists());
}

#[test]
fn current_package_roots_survive_but_replaced_and_deleted_roots_are_collected() {
    let fixture = Fixture::new();
    let first = fixture.store_unrooted(b"first payload");
    let second = fixture.store_unrooted(b"second payload");
    let orphan = fixture.store_unrooted(b"never published");
    let original = asset(1, EditablePayload::External(first));
    fixture.create("one.prism", &original);
    assert_eq!(fixture.collect().sweep.deleted, 2);
    assert!(fixture.blob(first).exists());
    assert!(!fixture.blob(second).exists());
    assert!(!fixture.blob(orphan).exists());

    let writer = fixture
        .workspace
        .store_payload(&b"second payload"[..], 64)
        .unwrap();
    let replacement = asset(1, EditablePayload::External(writer.reference()));
    fixture
        .workspace
        .save(
            vec![EditableWorkspaceEdit {
                expected_hash: hash(&original),
                replacement: replacement.clone(),
            }],
            BUDGET,
        )
        .unwrap();
    drop(writer);
    let report = fixture.collect();
    assert_eq!(report.sweep.deleted, 1);
    assert_eq!(report.sweep.retained, 1);
    assert!(!fixture.blob(first).exists());
    assert!(fixture.blob(second).exists());
    fixture
        .workspace
        .mutate(vec![remove(&replacement)], BUDGET)
        .unwrap();
    assert_eq!(fixture.collect().sweep.deleted, 1);
    assert!(!fixture.blob(second).exists());
}

#[test]
fn durable_snapshot_and_explicit_pins_survive_reopen_and_release_independently() {
    let fixture = Fixture::new();
    let writer = fixture
        .workspace
        .store_payload(&b"durable undo"[..], 64)
        .unwrap();
    let reference = writer.reference();
    let original = asset(1, EditablePayload::External(reference));
    fixture.create("one.prism", &original);
    let snapshot = fixture.workspace.snapshot(BUDGET).unwrap();
    fixture.workspace.retain_snapshot(11, &snapshot).unwrap();
    fixture
        .workspace
        .retain_payloads(12, &[reference, reference])
        .unwrap();
    fixture
        .workspace
        .mutate(vec![remove(&original)], BUDGET)
        .unwrap();
    drop(snapshot);
    drop(writer);
    let reopened = fixture.reopen();
    let report = reopened.collect_payloads(Duration::ZERO).unwrap();
    assert!(!report.deferred_due_to_active_readers);
    assert_eq!(report.history_pins, 2);
    assert_eq!(report.sweep.retained, 1);
    assert_eq!(report.sweep.deleted, 0);
    reopened.release_payloads(11).unwrap();
    reopened.release_payloads(11).unwrap();
    assert_eq!(
        fixture
            .reopen()
            .collect_payloads(Duration::ZERO)
            .unwrap()
            .history_pins,
        1
    );
    assert!(fixture.blob(reference).exists());
    fixture.reopen().release_payloads(12).unwrap();
    assert_eq!(fixture.collect().sweep.deleted, 1);
    assert!(!fixture.blob(reference).exists());
}

#[test]
fn replacing_a_history_pin_changes_its_roots_atomically_and_invalid_updates_preserve_it() {
    let fixture = Fixture::new();
    let old = fixture.store_unrooted(b"old history");
    let next = fixture.store_unrooted(b"next history");
    fixture.workspace.retain_payloads(1, &[old]).unwrap();
    let before = fs::read(fixture.pin(1)).unwrap();
    let invalid = EditablePayloadRef {
        byte_len: next.byte_len + 1,
        ..next
    };
    assert!(fixture.workspace.retain_payloads(1, &[invalid]).is_err());
    assert_eq!(fs::read(fixture.pin(1)).unwrap(), before);
    assert_eq!(
        fixture.workspace.retain_payloads(0, &[next]),
        Err(EditableAssetError::InvalidPayloadRetention)
    );
    assert_eq!(
        fixture.workspace.retain_payloads(
            2,
            &[
                old,
                EditablePayloadRef {
                    byte_len: old.byte_len + 1,
                    ..old
                }
            ]
        ),
        Err(EditableAssetError::InvalidPayloadRetention)
    );
    fixture.reopen().retain_payloads(1, &[next]).unwrap();
    let report = fixture.collect();
    assert_eq!(report.history_pins, 1);
    assert_eq!(report.sweep.deleted, 1);
    assert!(!fixture.blob(old).exists());
    assert!(fixture.blob(next).exists());
    fixture.workspace.release_payloads(1).unwrap();
    assert_eq!(fixture.collect().sweep.deleted, 1);
}

#[test]
fn snapshot_from_another_workspace_cannot_create_a_durable_pin() {
    let fixture = Fixture::new();
    let foreign = Fixture::new();
    let snapshot = fixture.workspace.snapshot(BUDGET).unwrap();
    assert_eq!(
        foreign.workspace.retain_snapshot(1, &snapshot),
        Err(EditableAssetError::InvalidPayloadRetention)
    );
    assert!(!foreign.pin(1).exists());
}

#[test]
fn corrupt_pin_checksum_aborts_gc_before_deleting_any_unreachable_object() {
    let fixture = Fixture::new();
    let rooted = fixture.store_unrooted(b"history root");
    let orphan = fixture.store_unrooted(b"unreachable but must survive failed sweep");
    fixture.workspace.retain_payloads(7, &[rooted]).unwrap();
    let mut pin = fs::read(fixture.pin(7)).unwrap();
    *pin.last_mut().unwrap() ^= 1;
    fs::write(fixture.pin(7), &pin).unwrap();
    let reopened = fixture.reopen();
    assert!(matches!(
        reopened.collect_payloads(Duration::ZERO),
        Err(EditableAssetError::InvalidPayloadRetention)
    ));
    assert!(fixture.blob(rooted).exists());
    assert!(fixture.blob(orphan).exists());
    // Retention must not silently replace/ignore corrupt existing metadata.
    assert!(reopened.retain_payloads(8, &[orphan]).is_err());
    assert!(!fixture.pin(8).exists());
    assert_eq!(fs::read(fixture.pin(7)).unwrap(), pin);
    reopened.release_payloads(7).unwrap();
    assert_eq!(fixture.collect().sweep.deleted, 2);
}

#[test]
fn truncated_or_unrecognized_history_roots_fail_closed() {
    for name in ["00000000000000000000000000000001.pin", "not-a-pin"] {
        let fixture = Fixture::new();
        let orphan = fixture.store_unrooted(b"must remain on invalid metadata");
        fs::create_dir(fixture.root().join(HISTORY)).unwrap();
        fs::write(fixture.root().join(HISTORY).join(name), b"truncated").unwrap();
        assert!(
            fixture.reopen().collect_payloads(Duration::ZERO).is_err(),
            "{name}"
        );
        assert!(fixture.blob(orphan).exists());
    }
}

/// Deliberately corrupt semantic metadata while retaining a valid checksum,
/// so the length-conflict test cannot pass merely by rejecting a checksum.
fn change_single_pin_length(path: &Path, length: u64) {
    let mut bytes = fs::read(path).unwrap();
    let checksum_start = bytes.len() - 32;
    bytes[checksum_start - 8..checksum_start].copy_from_slice(&length.to_le_bytes());
    let checksum = ContentHash::digest(&bytes[..checksum_start]);
    bytes[checksum_start..].copy_from_slice(&checksum.0);
    fs::write(path, bytes).unwrap();
}

#[test]
fn conflicting_hash_lengths_across_pins_or_between_pin_and_current_root_fail_closed() {
    for conflict_with_current in [false, true] {
        let fixture = Fixture::new();
        let reference = fixture.store_unrooted(b"same hash must have one length");
        let orphan = fixture.store_unrooted(b"unrelated collectible candidate");
        fixture.workspace.retain_payloads(1, &[reference]).unwrap();
        if conflict_with_current {
            fixture.create("one.prism", &asset(1, EditablePayload::External(reference)));
        } else {
            fixture.workspace.retain_payloads(2, &[reference]).unwrap();
        }
        change_single_pin_length(&fixture.pin(1), reference.byte_len + 1);
        assert!(matches!(
            fixture.reopen().collect_payloads(Duration::ZERO),
            Err(EditableAssetError::InvalidPayloadRetention)
        ));
        assert!(fixture.blob(reference).exists());
        assert!(fixture.blob(orphan).exists());
        if !conflict_with_current {
            assert_eq!(
                fixture.workspace.retain_payloads(3, &[orphan]),
                Err(EditableAssetError::InvalidPayloadRetention)
            );
            assert!(!fixture.pin(3).exists());
        }
        fixture.workspace.release_payloads(1).unwrap();
        assert_eq!(fixture.collect().sweep.deleted, 1);
        assert!(fixture.blob(reference).exists());
    }
}

#[test]
fn malformed_current_package_aborts_before_sweep() {
    let fixture = Fixture::new();
    let orphan = fixture.store_unrooted(b"unrooted object");
    fs::write(
        fixture.root().join("broken.prism"),
        b"not an editable package",
    )
    .unwrap();
    assert!(fixture.collect_result().is_err());
    assert!(fixture.blob(orphan).exists());
}

#[test]
fn conflicting_lengths_in_two_current_packages_abort_before_sweep() {
    let fixture = Fixture::new();
    let reference = fixture.store_unrooted(b"published object");
    let orphan = fixture.store_unrooted(b"unrooted object");
    fixture.create("one.prism", &asset(1, EditablePayload::External(reference)));
    // Bypass admission only to emulate corrupt on-disk metadata for the
    // collector. Normal mutation must reject this reference before publication.
    asset(
        2,
        EditablePayload::External(EditablePayloadRef {
            byte_len: reference.byte_len + 1,
            ..reference
        }),
    )
    .write_to_path(fixture.root().join("two.prism"))
    .unwrap();
    assert!(matches!(
        fixture.collect_result(),
        Err(EditableAssetError::InvalidPayloadReference)
    ));
    assert!(fixture.blob(reference).exists());
    assert!(fixture.blob(orphan).exists());
}

#[test]
fn missing_corrupt_and_wrong_length_external_replacements_never_publish_any_batch_member() {
    for damage in ["missing", "hash", "length"] {
        let fixture = Fixture::new();
        let original = asset(1, b"old inline authoring".to_vec());
        let neighbor = asset(2, b"unchanged neighbor".to_vec());
        fixture.create("one.prism", &original);
        fixture.create("two.prism", &neighbor);
        let before = fs::read(fixture.root().join("one.prism")).unwrap();
        let neighbor_before = fs::read(fixture.root().join("two.prism")).unwrap();
        let mut reference = fixture.store_unrooted(b"replacement external");
        match damage {
            "missing" => fs::remove_file(fixture.blob(reference)).unwrap(),
            "hash" => fs::write(
                fixture.blob(reference),
                vec![0xEE; reference.byte_len as usize],
            )
            .unwrap(),
            "length" => reference.byte_len += 1,
            _ => unreachable!(),
        }
        let replacement = asset(1, EditablePayload::External(reference));
        let next_neighbor = asset(2, b"neighbor changed in same batch".to_vec());
        assert!(
            fixture
                .workspace
                .save(
                    vec![
                        EditableWorkspaceEdit {
                            expected_hash: hash(&original),
                            replacement: replacement.clone()
                        },
                        EditableWorkspaceEdit {
                            expected_hash: hash(&neighbor),
                            replacement: next_neighbor
                        },
                    ],
                    BUDGET
                )
                .is_err(),
            "{damage}"
        );
        fixture.assert_package_unchanged("one.prism", &before);
        fixture.assert_package_unchanged("two.prism", &neighbor_before);
        assert!(
            fixture
                .workspace
                .mutate(
                    vec![put(
                        "new.prism",
                        None,
                        asset(3, replacement.authoring_payload)
                    )],
                    BUDGET
                )
                .is_err(),
            "{damage}"
        );
        assert!(!fixture.root().join("new.prism").exists());
        drop(fixture.reopen());
        assert!(!fixture.root().join("new.prism").exists());
        fixture.assert_package_unchanged("one.prism", &before);
        fixture.assert_package_unchanged("two.prism", &neighbor_before);
    }
}

#[test]
fn snapshot_resolution_enforces_budget_length_and_hash_even_after_capture() {
    for damage in ["missing", "hash", "length"] {
        let fixture = Fixture::new();
        let reference = fixture.store_unrooted(b"verified authoring");
        let value = asset(1, EditablePayload::External(reference));
        fixture.create("one.prism", &value);
        let snapshot = fixture.workspace.snapshot(BUDGET).unwrap();
        let payload = &snapshot[&value.asset].authoring_payload;
        assert!(matches!(
            snapshot.resolve_payload(payload, reference.byte_len - 1),
            Err(EditableAssetError::PayloadBudgetExceeded { .. })
        ));
        assert!(matches!(
            snapshot.resolve_payload(payload, MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES + 1),
            Err(EditableAssetError::PayloadBudgetExceeded { .. })
        ));
        assert_eq!(
            &*snapshot
                .resolve_payload(payload, reference.byte_len)
                .unwrap(),
            b"verified authoring"
        );
        match damage {
            "missing" => fs::remove_file(fixture.blob(reference)).unwrap(),
            "hash" => fs::write(
                fixture.blob(reference),
                vec![0xCC; reference.byte_len as usize],
            )
            .unwrap(),
            "length" => fs::write(fixture.blob(reference), b"short").unwrap(),
            _ => unreachable!(),
        }
        assert!(
            snapshot
                .resolve_payload(payload, reference.byte_len)
                .is_err(),
            "{damage}"
        );
        assert!(
            fixture.workspace.retain_payloads(1, &[reference]).is_err(),
            "{damage}"
        );
        assert!(!fixture.pin(1).exists());
    }
}

#[test]
fn reserved_storage_subtrees_cannot_be_used_as_package_destinations() {
    let fixture = Fixture::new();
    for path in [
        ".prism-editable-payloads/injected.prism",
        ".prism-editable-history/injected.prism",
        "nested/.prism-editable-payloads/injected.prism",
        "nested/.prism-editable-history/injected.prism",
    ] {
        assert_eq!(
            fixture
                .workspace
                .mutate(vec![put(path, None, asset(1, Vec::new()))], BUDGET),
            Err(EditableAssetError::PackageOutsideWorkspace)
        );
        assert!(!fixture.root().join(path).exists());
    }
    assert!(fixture.workspace.snapshot(BUDGET).unwrap().is_empty());
}

#[cfg(unix)]
#[test]
fn dangling_history_or_payload_directory_is_not_treated_as_absent() {
    use std::os::unix::fs::symlink;
    for reserved in [HISTORY, PAYLOADS] {
        let fixture = Fixture::new();
        let reference = fixture.store_unrooted(b"must not collect through a dangling link");
        let original = fixture.root().join(reserved);
        let preserved = fixture.root().join("preserved-storage");
        if reserved == PAYLOADS {
            fs::rename(&original, &preserved).unwrap();
        }
        symlink(fixture.root().join("does-not-exist"), &original).unwrap();
        assert!(matches!(
            fixture.collect_result(),
            Err(EditableAssetError::NonRegularFile)
        ));
        if reserved == HISTORY {
            assert_eq!(
                fixture.workspace.release_payloads(1),
                Err(EditableAssetError::NonRegularFile)
            );
            assert!(fixture.blob(reference).exists());
        } else {
            let hex = reference.content_hash.to_hex();
            assert!(preserved.join(&hex[..2]).join(&hex[2..]).exists());
            assert!(fixture
                .workspace
                .store_payload(&b"new object"[..], 64)
                .is_err());
        }
    }
}

#[cfg(unix)]
#[test]
fn symlinked_history_directory_rejects_gc_retention_and_release() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let outside = tempdir().unwrap();
    let reference = fixture.store_unrooted(b"history directory must be workspace owned");
    symlink(outside.path(), fixture.root().join(HISTORY)).unwrap();
    assert!(matches!(
        fixture.collect_result(),
        Err(EditableAssetError::NonRegularFile)
    ));
    assert_eq!(
        fixture.workspace.retain_payloads(1, &[reference]),
        Err(EditableAssetError::NonRegularFile)
    );
    assert_eq!(
        fixture.workspace.release_payloads(1),
        Err(EditableAssetError::NonRegularFile)
    );
    assert!(fixture.blob(reference).exists());
    assert!(regular_files(outside.path()).is_empty());
}

#[cfg(unix)]
#[test]
fn symlinked_storage_roots_shards_and_objects_are_rejected_without_external_writes() {
    use std::os::unix::fs::symlink;
    for location in ["root", "shard", "object", "lease"] {
        let fixture = Fixture::new();
        let outside = tempdir().unwrap();
        let bytes = b"symlink target payload";
        let reference = EditablePayloadRef {
            content_hash: ContentHash::digest(bytes),
            byte_len: bytes.len() as u64,
        };
        let outside_file = outside.path().join("sentinel");
        fs::write(&outside_file, bytes).unwrap();
        match location {
            "root" => symlink(outside.path(), fixture.root().join(PAYLOADS)).unwrap(),
            "shard" => {
                fs::create_dir(fixture.root().join(PAYLOADS)).unwrap();
                symlink(outside.path(), fixture.blob(reference).parent().unwrap()).unwrap();
            }
            "object" => {
                fs::create_dir_all(fixture.blob(reference).parent().unwrap()).unwrap();
                symlink(&outside_file, fixture.blob(reference)).unwrap();
            }
            "lease" => symlink(
                &outside_file,
                fixture.root().join(".prism-editable-payloads.lock"),
            )
            .unwrap(),
            _ => unreachable!(),
        }
        if location == "object" {
            let snapshot = fixture.workspace.snapshot(BUDGET).unwrap();
            assert!(snapshot
                .resolve_payload(&EditablePayload::External(reference), 1024)
                .is_err());
            assert!(fixture.workspace.retain_payloads(1, &[reference]).is_err());
        }
        assert!(
            fixture
                .workspace
                .store_payload(bytes.as_slice(), 1024)
                .is_err(),
            "{location}"
        );
        assert_eq!(fs::read(&outside_file).unwrap(), bytes);
        assert_eq!(regular_files(outside.path()), [outside_file]);
    }
}

#[cfg(unix)]
#[test]
fn symlinked_history_pin_and_package_parent_cannot_escape_workspace() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let outside = tempdir().unwrap();
    let reference = fixture.store_unrooted(b"retain this object");
    fixture.workspace.retain_payloads(1, &[reference]).unwrap();
    let outside_pin = outside.path().join("outside.pin");
    fs::rename(fixture.pin(1), &outside_pin).unwrap();
    let before = fs::read(&outside_pin).unwrap();
    symlink(&outside_pin, fixture.pin(1)).unwrap();
    assert!(matches!(
        fixture.collect_result(),
        Err(EditableAssetError::NonRegularFile)
    ));
    assert_eq!(
        fixture.workspace.release_payloads(1),
        Err(EditableAssetError::NonRegularFile)
    );
    assert!(fixture.workspace.retain_payloads(1, &[reference]).is_err());
    assert_eq!(fs::read(&outside_pin).unwrap(), before);
    assert!(fixture.blob(reference).exists());
    symlink(outside.path(), fixture.root().join("linked")).unwrap();
    assert!(fixture
        .workspace
        .mutate(
            vec![put("linked/escaped.prism", None, asset(1, Vec::new()))],
            BUDGET
        )
        .is_err());
    assert!(!outside.path().join("escaped.prism").exists());
}

#[test]
fn external_payload_above_inline_limit_stays_within_a_small_encoded_package_budget() {
    let fixture = Fixture::new();
    let length = 32 * 1024 * 1024 + 1;
    let writer = fixture
        .workspace
        .store_payload(io::repeat(0x5A).take(length), length)
        .unwrap();
    let reference = writer.reference();
    assert_eq!(reference.byte_len, length);
    let value = asset(1, EditablePayload::External(reference));
    let encoded = value.encode().unwrap();
    assert!(encoded.len() < 4096);
    fixture
        .workspace
        .mutate(vec![put("large.prism", None, value.clone())], 4096)
        .unwrap();
    let unchanged = fixture
        .workspace
        .save(
            vec![EditableWorkspaceEdit {
                expected_hash: hash(&value),
                replacement: value.clone(),
            }],
            encoded.len() as u64,
        )
        .unwrap();
    assert!(unchanged.updated_assets.is_empty());
    drop(writer);
    let snapshot = fixture.reopen().snapshot(encoded.len() as u64).unwrap();
    assert_eq!(snapshot[&value.asset], value);
    assert_eq!(
        fs::metadata(fixture.root().join("large.prism"))
            .unwrap()
            .len(),
        encoded.len() as u64
    );
    let resolved = snapshot
        .resolve_payload(&snapshot[&value.asset].authoring_payload, length)
        .unwrap();
    assert_eq!(resolved.len() as u64, length);
    assert!(resolved.iter().all(|byte| *byte == 0x5A));
    assert_eq!(ContentHash::digest(&resolved), reference.content_hash);
    drop(snapshot);
    assert_eq!(fixture.collect().sweep.retained, 1);
    fixture
        .workspace
        .mutate(vec![remove(&value)], 4096)
        .unwrap();
    assert_eq!(fixture.collect().sweep.deleted, 1);
}
