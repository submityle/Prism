use prism_asset_build_core::cas::{CasError, CasWriteResult, ContentAddressedStore};
use prism_asset_build_core::ContentHash;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

fn files_under(root: &Path) -> BTreeSet<PathBuf> {
    let mut files = BTreeSet::new();
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            files.extend(files_under(&entry.path()));
        } else {
            files.insert(entry.path());
        }
    }
    files
}

fn payload() -> Vec<u8> {
    (0..(3 * 64 * 1024 + 37))
        .map(|index| (index % 251) as u8)
        .collect()
}

fn assert_result(cas: &ContentAddressedStore, result: CasWriteResult, bytes: &[u8]) {
    assert_eq!(result.content_hash, ContentHash::digest(bytes));
    assert_eq!(result.byte_len, bytes.len() as u64);
    assert_eq!(cas.verify(result.content_hash).unwrap(), result.byte_len);
    assert_eq!(cas.read(result.content_hash).unwrap(), bytes);
    assert_eq!(
        files_under(cas.root()),
        BTreeSet::from([cas.blob_path(result.content_hash)])
    );
}

#[test]
fn uneven_chunks_hash_and_publish_exact_bytes_only_on_finish() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let bytes = payload();
    let hash = ContentHash::digest(&bytes);
    let mut writer = cas.begin_write(bytes.len() as u64 + 100).unwrap();
    assert_eq!(writer.write(&[]).unwrap(), 0);
    for chunk in bytes.chunks(4093) {
        writer.write_all(chunk).unwrap();
        writer.flush().unwrap();
        assert!(!cas.blob_path(hash).exists());
    }
    let temporary = files_under(root.path());
    assert_eq!(temporary.len(), 1);
    assert_eq!(temporary.first().unwrap().parent().unwrap(), cas.root());
    let result = writer.finish().unwrap();
    assert_result(&cas, result, &bytes);
    // Interoperability with existing put/read clients keeps the same identity.
    assert_eq!(cas.put(&bytes).unwrap(), result.content_hash);
}

#[test]
fn zero_budget_can_publish_and_deduplicate_empty_data() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    for _ in 0..2 {
        let mut writer = cas.begin_write(0).unwrap();
        assert_eq!(writer.write(&[]).unwrap(), 0);
        writer.flush().unwrap();
        assert_result(&cas, writer.finish().unwrap(), &[]);
    }
}

#[test]
fn exact_budget_is_accepted_across_multiple_writes() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let mut writer = cas.begin_write(7).unwrap();
    writer.write_all(b"abc").unwrap();
    writer.write_all(b"defg").unwrap();
    assert_eq!(writer.write(&[]).unwrap(), 0);
    assert_result(&cas, writer.finish().unwrap(), b"abcdefg");
}

#[test]
fn over_budget_write_poisons_finish_and_never_publishes_a_successful_prefix() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let mut writer = cas.begin_write(4).unwrap();
    writer.write_all(b"1234").unwrap();
    let failure = writer.write_all(b"5").unwrap_err();
    assert_eq!(failure.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(failure.to_string(), "CAS write exceeds maximum byte budget");
    // Cleanup happens on the first failure, even before finish or Drop.
    assert!(files_under(root.path()).is_empty());
    let retry = writer.write(b"x").unwrap_err();
    assert_eq!(retry.kind(), failure.kind());
    assert_eq!(retry.to_string(), failure.to_string());
    assert_eq!(writer.flush().unwrap_err().kind(), failure.kind());
    assert!(matches!(
        writer.finish(),
        Err(CasError::Io(error))
            if error.kind() == failure.kind() && error.to_string() == failure.to_string()
    ));
    assert!(!cas.blob_path(ContentHash::digest(b"1234")).exists());
    assert!(files_under(root.path()).is_empty());
}

#[test]
fn rejected_first_write_does_not_publish_empty_or_partial_data() {
    for budget in [0, 3] {
        let root = tempfile::tempdir().unwrap();
        let cas = ContentAddressedStore::open(root.path()).unwrap();
        let mut writer = cas.begin_write(budget).unwrap();
        assert_eq!(
            writer.write(b"too long").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(matches!(writer.finish(), Err(CasError::Io(_))));
        assert!(files_under(root.path()).is_empty());
    }
}

#[test]
fn dropping_unfinished_writers_removes_only_their_temporary_outputs() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let existing = cas.put(b"previously committed bytes").unwrap();
    let before = files_under(root.path());
    let mut first = cas.begin_write(64).unwrap();
    let mut second = cas.begin_write(64).unwrap();
    first.write_all(b"first unfinished output").unwrap();
    second.write_all(b"second unfinished output").unwrap();
    assert_eq!(files_under(root.path()).len(), before.len() + 2);
    drop(first);
    assert_eq!(files_under(root.path()).len(), before.len() + 1);
    drop(second);
    assert_eq!(files_under(root.path()), before);
    assert_eq!(cas.read(existing).unwrap(), b"previously committed bytes");
}

#[test]
fn stream_owns_its_store_path_and_can_finish_after_original_handle_drops() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = {
        let cas = ContentAddressedStore::open(root.path()).unwrap();
        cas.begin_write(64).unwrap()
    };
    writer.write_all(b"independent writer lifetime").unwrap();
    let result = writer.finish().unwrap();
    let reopened = ContentAddressedStore::open(root.path()).unwrap();
    assert_result(&reopened, result, b"independent writer lifetime");
}

#[test]
fn deduplication_verifies_and_preserves_an_existing_valid_blob() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let bytes = payload();
    let hash = cas.put(&bytes).unwrap();
    let before = fs::metadata(cas.blob_path(hash)).unwrap();
    let mut writer = cas.begin_write(bytes.len() as u64).unwrap();
    for chunk in bytes.chunks(1021) {
        writer.write_all(chunk).unwrap();
    }
    assert_result(&cas, writer.finish().unwrap(), &bytes);
    let after = fs::metadata(cas.blob_path(hash)).unwrap();
    assert_eq!(before.modified().unwrap(), after.modified().unwrap());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(before.dev(), after.dev());
        assert_eq!(before.ino(), after.ino());
    }
}

#[test]
fn existing_corrupt_or_truncated_blob_is_repaired_from_the_complete_stream() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let bytes = payload();
    let hash = cas.put(&bytes).unwrap();
    let mut tampered = bytes.clone();
    *tampered.last_mut().unwrap() ^= 1;
    for damaged in [tampered, bytes[..64 * 1024].to_vec()] {
        fs::write(cas.blob_path(hash), &damaged).unwrap();
        assert!(matches!(
            cas.verify(hash),
            Err(CasError::Corrupt { expected, actual })
                if expected == hash && actual == ContentHash::digest(&damaged)
        ));
        let mut writer = cas.begin_write(bytes.len() as u64).unwrap();
        for chunk in bytes.chunks(8191) {
            writer.write_all(chunk).unwrap();
        }
        assert_result(&cas, writer.finish().unwrap(), &bytes);
    }
}

#[test]
fn concurrently_finished_identical_streams_publish_one_verified_blob() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let bytes = payload();
    let mut writers = Vec::new();
    for chunk_size in [997, 4093, 8191, 64 * 1024] {
        let mut writer = cas.begin_write(bytes.len() as u64).unwrap();
        for chunk in bytes.chunks(chunk_size) {
            writer.write_all(chunk).unwrap();
        }
        writers.push(writer);
    }
    let barrier = Arc::new(Barrier::new(writers.len()));
    let handles: Vec<_> = writers
        .into_iter()
        .map(|writer| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                writer.finish()
            })
        })
        .collect();
    // Join every writer before asserting so a failure cannot race TempDir Drop.
    let results: Vec<_> = handles.into_iter().map(|handle| handle.join()).collect();
    for result in results {
        assert_result(&cas, result.unwrap().unwrap(), &bytes);
    }
}

#[test]
fn finish_directory_error_discards_unpublished_output() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let bytes = b"cannot publish past a file-shaped shard";
    let hash = ContentHash::digest(bytes);
    let path = cas.blob_path(hash);
    let shard = path.parent().unwrap();
    let mut writer = cas.begin_write(bytes.len() as u64).unwrap();
    writer.write_all(bytes).unwrap();
    fs::write(shard, b"not a directory").unwrap();
    assert!(matches!(writer.finish(), Err(CasError::Io(_))));
    assert_eq!(
        files_under(root.path()),
        BTreeSet::from([shard.to_path_buf()])
    );
    assert_eq!(fs::read(shard).unwrap(), b"not a directory");
}

#[test]
fn existing_non_file_is_rejected_without_clobbering_or_leaving_temporary_output() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let bytes = b"a directory cannot be a CAS blob";
    let path = cas.blob_path(ContentHash::digest(bytes));
    fs::create_dir_all(&path).unwrap();
    let mut writer = cas.begin_write(bytes.len() as u64).unwrap();
    writer.write_all(bytes).unwrap();
    assert!(matches!(writer.finish(), Err(CasError::Io(_))));
    assert!(path.is_dir());
    assert!(files_under(root.path()).is_empty());
}

#[test]
fn ordinary_cas_sweep_ignores_the_unpublished_stream_file() {
    let root = tempfile::tempdir().unwrap();
    let cas = ContentAddressedStore::open(root.path()).unwrap();
    let mut writer = cas.begin_write(64).unwrap();
    writer.write_all(b"incomplete").unwrap();
    let before = files_under(root.path());
    let report = cas
        .sweep_unreachable(&BTreeSet::new(), Duration::ZERO)
        .unwrap();
    assert_eq!(report.scanned, 0);
    assert_eq!(report.deleted, 0);
    assert_eq!(files_under(root.path()), before);
    writer.write_all(b" becomes complete").unwrap();
    assert_result(
        &cas,
        writer.finish().unwrap(),
        b"incomplete becomes complete",
    );
}
