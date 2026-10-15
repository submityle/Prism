//! Faults wrap a real locked file and exercise the production writer, not a
//! separate test implementation. New fixtures are left in the temporary dir.

use std::io::{self, Seek, SeekFrom, Write};

use super::journal_write::JournalIo;
use super::test_support::read_journal_bytes;
use super::*;
use crate::durable_tests::{temporary_path, transaction};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Failure {
    Seek,
    Write,
    Flush,
    Sync,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RollbackFailure {
    Truncate,
    Seek,
    Flush,
    Sync,
}

struct FaultFile<'a> {
    file: &'a mut File,
    failure: Failure,
    rollback_failure: Option<RollbackFailure>,
    bytes_left: usize,
    failed: bool,
}

impl<'a> FaultFile<'a> {
    fn new(
        file: &'a mut File,
        failure: Failure,
        rollback_failure: Option<RollbackFailure>,
        bytes_left: usize,
    ) -> Self {
        Self {
            file,
            failure,
            rollback_failure,
            bytes_left,
            failed: false,
        }
    }

    fn fail_operation(&mut self) -> io::Error {
        self.failed = true;
        io::Error::other("injected append failure")
    }

    fn rollback_fault(&self, stage: RollbackFailure) -> io::Result<()> {
        if self.failed && self.rollback_failure == Some(stage) {
            Err(io::Error::other("injected rollback failure"))
        } else {
            Ok(())
        }
    }
}

impl Write for FaultFile<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.failed && self.failure == Failure::Write {
            if self.bytes_left == 0 {
                return Err(self.fail_operation());
            }
            let length = bytes.len().min(self.bytes_left);
            let written = self.file.write(&bytes[..length])?;
            self.bytes_left -= written;
            return Ok(written);
        }
        self.file.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.rollback_fault(RollbackFailure::Flush)?;
        if !self.failed && self.failure == Failure::Flush {
            return Err(self.fail_operation());
        }
        self.file.flush()
    }
}

impl Seek for FaultFile<'_> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.rollback_fault(RollbackFailure::Seek)?;
        if !self.failed && self.failure == Failure::Seek {
            return Err(self.fail_operation());
        }
        self.file.seek(position)
    }
}

impl JournalIo for FaultFile<'_> {
    fn set_len(&mut self, length: u64) -> io::Result<()> {
        self.rollback_fault(RollbackFailure::Truncate)?;
        self.file.set_len(length)
    }

    fn sync_all(&mut self) -> io::Result<()> {
        self.rollback_fault(RollbackFailure::Sync)?;
        if !self.failed && self.failure == Failure::Sync {
            return Err(self.fail_operation());
        }
        self.file.sync_all()
    }
}

fn assert_original_error(error: &JournalError) {
    match error {
        JournalError::Io(error) => assert_eq!(error.to_string(), "injected append failure"),
        other => panic!("expected original I/O error, got {other:?}"),
    }
}

fn assert_poisoned_without_io(journal: &mut FileJournal) {
    let before = read_journal_bytes(journal);
    assert!(matches!(
        journal.append(transaction(900, 900, 0, 90)),
        Err(JournalError::Poisoned)
    ));
    assert!(matches!(
        journal.append_batch(&[transaction(901, 901, 0, 91)]),
        Err(JournalError::Poisoned)
    ));
    assert!(matches!(
        journal.append_batch(&[]),
        Err(JournalError::Poisoned)
    ));
    assert!(matches!(journal.flush(), Err(JournalError::Poisoned)));
    assert!(matches!(journal.sync(), Err(JournalError::Poisoned)));
    assert!(matches!(
        FileJournal::open(journal.path()),
        Err(JournalError::Locked)
    ));
    assert_eq!(read_journal_bytes(journal), before);
}

#[test]
fn append_faults_poison_even_when_rollback_succeeds_and_preserve_both_errors_when_it_fails() {
    for failure in [Failure::Seek, Failure::Write, Failure::Flush, Failure::Sync] {
        for rollback in [
            None,
            Some(RollbackFailure::Truncate),
            Some(RollbackFailure::Seek),
            Some(RollbackFailure::Flush),
            Some(RollbackFailure::Sync),
        ] {
            let path = temporary_path("append-fault");
            let first = transaction(201, 1, 0, 10);
            let second = transaction(202, 2, 1, 20);
            let mut journal = FileJournal::open(&path).unwrap();
            journal.append(first.clone()).unwrap();
            let before = read_journal_bytes(&mut journal);
            let error = journal
                .writer
                .append(
                    &mut FaultFile::new(&mut journal.file, failure, rollback, 7),
                    &mut journal.log,
                    second.clone(),
                )
                .expect_err("injected append must fail");
            if rollback.is_some() {
                match &error {
                    JournalError::RollbackFailed {
                        operation,
                        rollback,
                    } => {
                        assert_original_error(operation);
                        assert_eq!(rollback.to_string(), "injected rollback failure");
                        assert!(std::error::Error::source(&error).is_some());
                    }
                    other => panic!("missing rollback error for {failure:?}: {other:?}"),
                }
            } else {
                assert_original_error(&error);
                assert_eq!(read_journal_bytes(&mut journal), before);
            }
            assert_eq!(journal.log().as_slice(), &[first.clone()]);
            assert_poisoned_without_io(&mut journal);
            drop(journal);

            let mut reopened = FileJournal::open(&path).expect("reopen recovers file");
            let complete_failed_record = rollback == Some(RollbackFailure::Truncate)
                && matches!(failure, Failure::Flush | Failure::Sync);
            if complete_failed_record {
                assert_eq!(reopened.log().as_slice(), &[first.clone(), second.clone()]);
            } else {
                assert_eq!(reopened.log().as_slice(), &[first.clone()]);
                reopened
                    .append(second.clone())
                    .expect("new handle is not poisoned");
            }
            reopened.append(transaction(203, 3, 2, 30)).unwrap();
            drop(reopened);
            let reopened = FileJournal::open(&path).unwrap();
            assert_eq!(reopened.log().len(), 3);
            assert_eq!(reopened.recovery(), None);
        }
    }
}

#[test]
fn failed_batch_rollback_recovers_complete_prefix_before_allowing_more_appends() {
    for rollback_fails in [false, true] {
        let path = temporary_path("batch-fault");
        let first = transaction(301, 1, 0, 10);
        let batch = [
            transaction(302, 2, 1, 20),
            transaction(303, 3, 2, 30),
            transaction(304, 4, 3, 40),
        ];
        let bytes_to_write = batch
            .iter()
            .map(|tx| encode_record(tx).unwrap().len())
            .sum::<usize>()
            - 1;
        let mut journal = FileJournal::open(&path).unwrap();
        journal.append(first.clone()).unwrap();
        let before = read_journal_bytes(&mut journal);
        let error = journal
            .writer
            .append_batch(
                &mut FaultFile::new(
                    &mut journal.file,
                    Failure::Write,
                    rollback_fails.then_some(RollbackFailure::Truncate),
                    bytes_to_write,
                ),
                &mut journal.log,
                &batch,
            )
            .expect_err("batch write must fail before final checksum byte");
        assert_eq!(
            matches!(error, JournalError::RollbackFailed { .. }),
            rollback_fails
        );
        assert_eq!(journal.log().as_slice(), &[first.clone()]);
        if rollback_fails {
            assert_eq!(
                fs::metadata(&path).unwrap().len(),
                (before.len() + bytes_to_write) as u64
            );
        } else {
            assert_eq!(read_journal_bytes(&mut journal), before);
        }
        assert_poisoned_without_io(&mut journal);
        drop(journal);

        let mut reopened = FileJournal::open(&path).unwrap();
        if rollback_fails {
            assert_eq!(
                reopened.log().as_slice(),
                &[first, batch[0].clone(), batch[1].clone()]
            );
            let recovery = reopened.recovery().expect("recover partial final record");
            assert_eq!(recovery.reason(), RecoveryReason::Truncated);
            assert_eq!(
                recovery.truncated_bytes(),
                encode_record(&batch[2]).unwrap().len() as u64 - 1
            );
            reopened.append(batch[2].clone()).unwrap();
        } else {
            assert_eq!(reopened.log().as_slice(), &[first]);
            assert_eq!(reopened.recovery(), None);
            reopened.append_batch(&batch).unwrap();
        }
        drop(reopened);
        let reopened = FileJournal::open(&path).unwrap();
        assert_eq!(reopened.log().len(), 4);
        assert_eq!(reopened.recovery(), None);
    }
}

#[test]
fn explicit_flush_and_sync_errors_poison_without_unpoisoning_on_retry() {
    for failure in [Failure::Flush, Failure::Sync] {
        let path = temporary_path("explicit-io-fault");
        let mut journal = FileJournal::open(&path).unwrap();
        let mut file = FaultFile::new(&mut journal.file, failure, None, 0);
        let error = if failure == Failure::Flush {
            journal.writer.flush(&mut file)
        } else {
            journal.writer.sync(&mut file)
        }
        .unwrap_err();
        assert_original_error(&error);
        assert_poisoned_without_io(&mut journal);
        drop(journal);
        let mut reopened = FileJournal::open(&path).unwrap();
        reopened.append(transaction(401, 1, 0, 10)).unwrap();
    }
}
