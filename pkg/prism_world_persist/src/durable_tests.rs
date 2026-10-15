use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::durable::test_support::read_journal_bytes;

pub(super) fn transaction(id: u128, clock: u64, sequence: u64, value: i64) -> Transaction {
    let forward = OverrideDelta::single(FieldId(1), OverrideValue::Integer(value))
        .expect("valid forward delta");
    let inverse =
        OverrideDelta::single(FieldId(1), OverrideValue::Delete).expect("valid inverse delta");
    Transaction::new(
        TransactionId(id),
        ActorId(7),
        LogicalClock(clock),
        sequence,
        PersistentId(99),
        forward,
        inverse,
    )
    .expect("valid transaction")
}

pub(super) fn temporary_path(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "prism-world-{label}-{}-{nonce}",
        std::process::id()
    ))
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn journal_restarts_with_all_durable_transactions() {
    let path = temporary_path("restart");
    let _cleanup = RemoveOnDrop(path.clone());
    let first = transaction(1, 1, 0, 10);
    let second = transaction(2, 2, 1, 20);

    {
        let mut journal = FileJournal::open(&path).expect("open journal");
        journal
            .append_batch(&[first.clone(), second.clone()])
            .expect("append batch");
        assert_eq!(journal.log().as_slice(), &[first.clone(), second.clone()]);
    }

    let reopened = FileJournal::open(&path).expect("reopen journal");
    assert_eq!(reopened.recovery(), None);
    assert_eq!(reopened.log().as_slice(), &[first, second]);
}

#[test]
fn journal_rejects_duplicate_without_mutating_file_or_log() {
    let path = temporary_path("duplicate");
    let _cleanup = RemoveOnDrop(path.clone());
    let first = transaction(3, 1, 0, 10);
    let mut journal = FileJournal::open(&path).expect("open journal");
    journal.append(first.clone()).expect("append first");
    let before_len = fs::metadata(&path).expect("metadata").len();

    assert!(matches!(
        journal.append(first),
        Err(JournalError::Log(LogError::DuplicateTransaction(
            TransactionId(3)
        )))
    ));
    assert_eq!(fs::metadata(&path).expect("metadata").len(), before_len);
    assert_eq!(journal.log().len(), 1);
    journal
        .append(transaction(30, 2, 1, 20))
        .expect("validation failure must not poison the journal");
}

#[test]
fn journal_exclusive_lock_rejects_second_handle_until_drop() {
    let path = temporary_path("exclusive-lock");
    let first = transaction(101, 1, 0, 10);
    let second = transaction(102, 2, 1, 20);
    let mut journal = FileJournal::open(&path).expect("first handle");
    // Even an empty journal is protected before any append.
    let header = read_journal_bytes(&mut journal);
    assert!(matches!(
        FileJournal::open(&path),
        Err(JournalError::Locked)
    ));
    assert_eq!(read_journal_bytes(&mut journal), header);
    journal.append(first.clone()).expect("first append");
    let before = read_journal_bytes(&mut journal);
    assert!(matches!(
        FileJournal::open(&path),
        Err(JournalError::Locked)
    ));
    assert_eq!(read_journal_bytes(&mut journal), before);
    drop(journal);

    let mut reopened = FileJournal::open(&path).expect("lock released on drop");
    assert_eq!(reopened.log().as_slice(), &[first.clone()]);
    reopened.append(second.clone()).expect("second append");
    drop(reopened);
    let reopened = FileJournal::open(&path).expect("restart");
    assert_eq!(reopened.log().as_slice(), &[first, second]);
    assert_eq!(reopened.recovery(), None);
}

#[test]
fn journal_exclusive_lock_is_visible_to_another_process() {
    const CHILD_PATH: &str = "PRISM_JOURNAL_LOCK_TEST_CHILD_PATH";
    if let Some(path) = std::env::var_os(CHILD_PATH) {
        let raw = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("child opens actual file");
        assert!(matches!(raw.try_lock(), Err(fs::TryLockError::WouldBlock)));
        assert!(matches!(
            FileJournal::open(&path),
            Err(JournalError::Locked)
        ));
        return;
    }

    let path = temporary_path("process-lock");
    let mut journal = FileJournal::open(&path).expect("parent journal");
    journal.append(transaction(103, 1, 0, 10)).unwrap();
    let before = read_journal_bytes(&mut journal);
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "durable_tests::journal_exclusive_lock_is_visible_to_another_process",
            "--nocapture",
        ])
        .env(CHILD_PATH, &path)
        .output()
        .expect("start lock contender process");
    assert!(
        child.status.success(),
        "child failed: {}\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(String::from_utf8_lossy(&child.stdout).contains("1 passed"));
    assert_eq!(read_journal_bytes(&mut journal), before);
    drop(journal);
    assert!(FileJournal::open(&path).is_ok());
}

#[test]
fn journal_truncates_a_cut_tail_and_keeps_valid_prefix() {
    let path = temporary_path("truncated");
    let _cleanup = RemoveOnDrop(path.clone());
    let first = transaction(4, 1, 0, 10);
    let second = transaction(5, 2, 1, 20);
    let valid_prefix_len;
    {
        let mut journal = FileJournal::open(&path).expect("open journal");
        journal.append(first.clone()).expect("append first");
        valid_prefix_len = fs::metadata(&path).expect("metadata").len();
        journal.append(second).expect("append second");
    }
    let original_len = fs::metadata(&path).expect("metadata").len();
    let truncated_len = original_len - 3;
    let file = fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open for truncation");
    file.set_len(truncated_len).expect("truncate tail");

    let reopened = FileJournal::open(&path).expect("recover journal");
    let recovery = reopened.recovery().expect("recovery details");
    assert_eq!(recovery.reason(), RecoveryReason::Truncated);
    assert_eq!(recovery.truncated_bytes(), truncated_len - valid_prefix_len);
    assert_eq!(reopened.log().as_slice(), &[first]);
    assert_eq!(
        fs::metadata(&path).expect("metadata").len(),
        valid_prefix_len
    );
}

#[test]
fn journal_discards_checksum_corruption_and_allows_append_after_recovery() {
    let path = temporary_path("checksum");
    let _cleanup = RemoveOnDrop(path.clone());
    let first = transaction(6, 1, 0, 10);
    let second = transaction(7, 2, 1, 20);
    {
        let mut journal = FileJournal::open(&path).expect("open journal");
        journal
            .append_batch(&[first.clone(), second])
            .expect("append batch");
    }
    let mut bytes = fs::read(&path).expect("read journal");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x80;
    fs::write(&path, bytes).expect("corrupt journal");

    let mut reopened = FileJournal::open(&path).expect("recover checksum");
    assert_eq!(
        reopened.recovery().map(JournalRecovery::reason),
        Some(RecoveryReason::ChecksumMismatch)
    );
    assert_eq!(reopened.log().as_slice(), &[first]);
    reopened
        .append(transaction(8, 3, 1, 30))
        .expect("append after recovery");
    assert_eq!(reopened.log().len(), 2);
}

#[test]
fn snapshot_store_round_trips_and_replaces_atomically() {
    let path = temporary_path("snapshot");
    let _cleanup = RemoveOnDrop(path.clone());
    let mut timeline = Timeline::new();
    timeline
        .commit(
            TransactionId(20),
            ActorId(1),
            LogicalClock(1),
            PersistentId(2),
            OverrideDelta::single(FieldId(9), OverrideValue::Integer(42)).expect("valid delta"),
        )
        .expect("commit");
    let snapshot = timeline.snapshot();

    SnapshotStore::save(&path, &snapshot).expect("save snapshot");
    assert_eq!(SnapshotStore::load(&path).expect("load snapshot"), snapshot);
    let temporary_files = fs::read_dir(path.parent().expect("snapshot parent"))
        .expect("read parent")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_name().to_string_lossy().starts_with(&format!(
                ".{}.tmp-",
                path.file_name().unwrap().to_string_lossy()
            ))
        })
        .count();
    assert_eq!(temporary_files, 0);
}
