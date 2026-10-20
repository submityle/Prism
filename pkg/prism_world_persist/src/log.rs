use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::{DeltaError, LogicalClock};
use crate::{PersistentId, Transaction, TransactionId};

/// Append-only log validation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogError {
    /// The transaction ID was already appended.
    DuplicateTransaction(TransactionId),
    /// The transaction sequence was not the next expected sequence.
    OutOfOrder {
        /// Next sequence required by the log.
        expected: u64,
        /// Sequence supplied by the rejected transaction.
        actual: u64,
    },
    /// The logical clock regressed relative to the previous append.
    ClockRegression {
        /// Clock of the last accepted transaction.
        previous: LogicalClock,
        /// Clock supplied by the rejected transaction.
        actual: LogicalClock,
    },
    /// No further sequence number can be represented.
    SequenceExhausted,
    /// The transaction's field data was invalid.
    InvalidTransaction(DeltaError),
}

/// Deterministic iterator over transactions affecting one object.
pub struct ObjectLogIter<'a> {
    entries: &'a [Transaction],
    indices: &'a [usize],
    position: usize,
}

impl<'a> Iterator for ObjectLogIter<'a> {
    type Item = &'a Transaction;

    fn next(&mut self) -> Option<Self::Item> {
        let index = *self.indices.get(self.position)?;
        self.position += 1;
        self.entries.get(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len().saturating_sub(self.position);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ObjectLogIter<'_> {}

/// Validated append-only transaction log with a deterministic object index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AppendOnlyLog {
    entries: Vec<Transaction>,
    object_index: BTreeMap<PersistentId, Vec<usize>>,
    transaction_ids: BTreeSet<TransactionId>,
    next_sequence: u64,
    last_clock: Option<LogicalClock>,
}

impl AppendOnlyLog {
    /// Creates an empty log whose first transaction must have sequence zero.
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            object_index: BTreeMap::new(),
            transaction_ids: BTreeSet::new(),
            next_sequence: 0,
            last_clock: None,
        }
    }

    /// Appends a transaction after validating uniqueness, sequence, clock,
    /// and field data. No mutation occurs when validation fails.
    pub fn append(&mut self, transaction: Transaction) -> Result<(), LogError> {
        self.validate_append(&transaction)?;

        let index = self.entries.len();
        self.entries.push(transaction.clone());
        self.transaction_ids.insert(transaction.id());
        self.object_index
            .entry(transaction.object())
            .or_default()
            .push(index);
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.last_clock = Some(transaction.clock());
        Ok(())
    }

    /// Appends a batch after validating all entries against the prospective
    /// log. Validation failures leave the log unchanged.
    pub fn append_batch(&mut self, transactions: &[Transaction]) -> Result<(), LogError> {
        let mut next_sequence = self.next_sequence;
        let mut last_clock = self.last_clock;
        for (index, transaction) in transactions.iter().enumerate() {
            if self.transaction_ids.contains(&transaction.id())
                || transactions[..index]
                    .iter()
                    .any(|previous| previous.id() == transaction.id())
            {
                return Err(LogError::DuplicateTransaction(transaction.id()));
            }
            if transaction.sequence() != next_sequence {
                return Err(LogError::OutOfOrder {
                    expected: next_sequence,
                    actual: transaction.sequence(),
                });
            }
            if let Some(previous) = last_clock
                && transaction.clock() < previous
            {
                return Err(LogError::ClockRegression {
                    previous,
                    actual: transaction.clock(),
                });
            }
            if next_sequence == u64::MAX {
                return Err(LogError::SequenceExhausted);
            }
            transaction
                .validate()
                .map_err(LogError::InvalidTransaction)?;
            next_sequence += 1;
            last_clock = Some(transaction.clock());
        }

        // Validation is complete. Use the normal append path so all indexes
        // and cursors stay identical to individual appends.
        for transaction in transactions {
            self.append(transaction.clone())
                .expect("prevalidated append batch remains valid");
        }
        Ok(())
    }

    pub(crate) fn validate_append(&self, transaction: &Transaction) -> Result<(), LogError> {
        if self.transaction_ids.contains(&transaction.id()) {
            return Err(LogError::DuplicateTransaction(transaction.id()));
        }
        if transaction.sequence() != self.next_sequence {
            return Err(LogError::OutOfOrder {
                expected: self.next_sequence,
                actual: transaction.sequence(),
            });
        }
        if let Some(previous) = self.last_clock
            && transaction.clock() < previous
        {
            return Err(LogError::ClockRegression {
                previous,
                actual: transaction.clock(),
            });
        }
        if self.next_sequence == u64::MAX {
            return Err(LogError::SequenceExhausted);
        }
        transaction
            .validate()
            .map_err(LogError::InvalidTransaction)?;
        Ok(())
    }

    /// Returns the next sequence required by [`Self::append`].
    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Returns the number of transactions in this log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when no transactions have been appended.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates all transactions in append sequence order.
    pub fn iter(&self) -> core::slice::Iter<'_, Transaction> {
        self.entries.iter()
    }

    /// Iterates one object's transactions in append sequence order.
    pub fn iter_object(&self, object: PersistentId) -> ObjectLogIter<'_> {
        let indices = self
            .object_index
            .get(&object)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        ObjectLogIter {
            entries: &self.entries,
            indices,
            position: 0,
        }
    }

    /// Returns whether a transaction ID is already present.
    pub fn contains(&self, id: TransactionId) -> bool {
        self.transaction_ids.contains(&id)
    }

    /// Returns the transaction at an append index.
    pub fn get(&self, index: usize) -> Option<&Transaction> {
        self.entries.get(index)
    }

    /// Returns the log entries as a slice in append order.
    pub fn as_slice(&self) -> &[Transaction] {
        &self.entries
    }
}
