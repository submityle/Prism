//! Shared production write path with a private I/O seam for fault injection.

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};

use super::{encode_record, JournalError};
use crate::{AppendOnlyLog, Transaction};

pub(super) trait JournalIo: Write + Seek {
    fn set_len(&mut self, length: u64) -> io::Result<()>;
    fn sync_all(&mut self) -> io::Result<()>;
}

impl JournalIo for File {
    fn set_len(&mut self, length: u64) -> io::Result<()> {
        File::set_len(self, length)
    }

    fn sync_all(&mut self) -> io::Result<()> {
        File::sync_all(self)
    }
}

pub(super) struct JournalWriter {
    end_offset: u64,
    poisoned: bool,
}

impl JournalWriter {
    pub(super) const fn new(end_offset: u64) -> Self {
        Self {
            end_offset,
            poisoned: false,
        }
    }

    fn ensure_writable(&self) -> Result<(), JournalError> {
        if self.poisoned {
            Err(JournalError::Poisoned)
        } else {
            Ok(())
        }
    }

    pub(super) fn append(
        &mut self,
        file: &mut impl JournalIo,
        log: &mut AppendOnlyLog,
        transaction: Transaction,
    ) -> Result<(), JournalError> {
        self.ensure_writable()?;
        log.validate_append(&transaction)?;
        let record = encode_record(&transaction)?;
        if let Err(error) = self.write_durable(file, &record) {
            return Err(self.failed_append(file, error.into()));
        }
        if let Err(error) = log.append(transaction) {
            return Err(self.failed_append(file, error.into()));
        }
        self.end_offset += record.len() as u64;
        Ok(())
    }

    pub(super) fn append_batch(
        &mut self,
        file: &mut impl JournalIo,
        log: &mut AppendOnlyLog,
        transactions: &[Transaction],
    ) -> Result<(), JournalError> {
        self.ensure_writable()?;
        if transactions.is_empty() {
            return Ok(());
        }
        let mut candidate = log.clone();
        let mut bytes = Vec::new();
        for transaction in transactions {
            candidate.validate_append(transaction)?;
            bytes.extend_from_slice(&encode_record(transaction)?);
            candidate.append(transaction.clone())?;
        }
        if let Err(error) = self.write_durable(file, &bytes) {
            return Err(self.failed_append(file, error.into()));
        }
        *log = candidate;
        self.end_offset += bytes.len() as u64;
        Ok(())
    }

    pub(super) fn flush(&mut self, file: &mut impl JournalIo) -> Result<(), JournalError> {
        self.ensure_writable()?;
        let result = file.flush();
        self.finish_io(result)
    }

    pub(super) fn sync(&mut self, file: &mut impl JournalIo) -> Result<(), JournalError> {
        self.ensure_writable()?;
        let result = file.sync_all();
        self.finish_io(result)
    }

    fn finish_io(&mut self, result: io::Result<()>) -> Result<(), JournalError> {
        result.map_err(|error| {
            self.poisoned = true;
            JournalError::Io(error)
        })
    }

    fn write_durable(&self, file: &mut impl JournalIo, bytes: &[u8]) -> io::Result<()> {
        file.seek(SeekFrom::Start(self.end_offset))?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()
    }

    fn failed_append(
        &mut self,
        file: &mut impl JournalIo,
        operation: JournalError,
    ) -> JournalError {
        // Never reuse the pre-failure offset, even after a successful rollback.
        // Only opening a new handle can establish a trusted on-disk prefix.
        self.poisoned = true;
        match self.rollback_to_end(file) {
            Ok(()) => operation,
            Err(rollback) => JournalError::RollbackFailed {
                operation: Box::new(operation),
                rollback,
            },
        }
    }

    fn rollback_to_end(&self, file: &mut impl JournalIo) -> io::Result<()> {
        file.set_len(self.end_offset)?;
        file.seek(SeekFrom::Start(self.end_offset))?;
        file.flush()?;
        file.sync_all()
    }
}
