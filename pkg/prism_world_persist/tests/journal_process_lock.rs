#![cfg(feature = "std")]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use prism_world_persist::{
    ActorId, FieldId, FileJournal, JournalError, LogicalClock, OverrideDelta, OverrideValue,
    PersistentId, Transaction, TransactionId,
};

const CHILD_MODE: &str = "PRISM_WORLD_PERSIST_PROCESS_LOCK_PROBE_MODE";
const CHILD_PATH: &str = "PRISM_WORLD_PERSIST_PROCESS_LOCK_PROBE_PATH";
const CHILD_TEST: &str = "journal_process_lock_child_probe";

fn committed_prefix() -> Vec<Transaction> {
    [10, 20]
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            Transaction::new(
                TransactionId(index as u128 + 1),
                ActorId(7),
                LogicalClock(index as u64 + 1),
                index as u64,
                PersistentId(99),
                OverrideDelta::single(FieldId(1), OverrideValue::Integer(value)).unwrap(),
                OverrideDelta::single(FieldId(1), OverrideValue::Delete).unwrap(),
            )
            .unwrap()
        })
        .collect()
}

fn fixture_path() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    for attempt in 0..100 {
        let directory = std::env::temp_dir().join(format!(
            "prism-journal-process-lock-{}-{nonce}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&directory) {
            Ok(()) => return directory.join("journal"),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create unique journal fixture: {error}"),
        }
    }
    panic!("could not reserve a unique journal fixture");
}

fn run_child_probe(path: &Path, mode: &str) {
    let output = Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(CHILD_MODE, mode)
        .env(CHILD_PATH, path)
        .output()
        .expect("run child lock probe");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "child {mode} probe failed: {}\n{stdout}\n{stderr}",
        output.status
    );
    // A successful exit with zero matching tests must not count as a probe.
    assert!(
        stdout.contains(&format!("journal-process-lock-probe:{mode}:ok")),
        "child helper did not acknowledge {mode}:\n{stdout}\n{stderr}"
    );
}

#[test]
fn journal_process_lock_child_probe() {
    let Some(mode) = std::env::var_os(CHILD_MODE) else {
        // The helper is inert during ordinary test discovery/execution.
        return;
    };
    let path = PathBuf::from(std::env::var_os(CHILD_PATH).expect("child journal path"));
    let mode = mode.to_str().expect("UTF-8 child probe mode");
    match mode {
        "locked" => match FileJournal::open(&path) {
            Err(JournalError::Locked) => {}
            Err(error) => panic!("expected Locked, got {error:?}"),
            Ok(_) => panic!("child acquired a journal still owned by the parent"),
        },
        "released" => {
            let journal = FileJournal::open(&path).expect("child opens after parent drop");
            assert_eq!(journal.recovery(), None);
            assert_eq!(journal.log().as_slice(), committed_prefix().as_slice());
        }
        other => panic!("unknown child probe mode: {other}"),
    }
    println!("journal-process-lock-probe:{mode}:ok");
}

#[test]
fn journal_exclusive_lock_rejects_child_then_allows_reopen_without_mutation() {
    let path = fixture_path();
    let expected = committed_prefix();
    {
        let mut journal = FileJournal::open(&path).expect("create journal fixture");
        journal.append_batch(&expected).expect("commit prefix");
    }
    // Snapshot only while unlocked: another read handle would fail under
    // Windows byte-range locking. No lock is released during the probe below.
    let before = fs::read(&path).expect("snapshot committed journal");
    let journal = FileJournal::open(&path).expect("parent acquires exclusive lock");
    assert_eq!(journal.recovery(), None);
    assert_eq!(journal.log().as_slice(), expected.as_slice());

    // output() waits for the probe to exit while this exact parent owns the lock.
    run_child_probe(&path, "locked");
    assert_eq!(journal.log().as_slice(), expected.as_slice());
    drop(journal);
    assert_eq!(fs::read(&path).expect("bytes after rejected child"), before);

    run_child_probe(&path, "released");
    assert_eq!(
        fs::read(&path).expect("bytes after successful child"),
        before
    );
    let reopened = FileJournal::open(&path).expect("parent reopens after child exit");
    assert_eq!(reopened.recovery(), None);
    assert_eq!(reopened.log().as_slice(), expected.as_slice());
}
