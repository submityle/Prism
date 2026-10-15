use alloc::collections::{BTreeMap, BTreeSet};

use crate::{ActorId, DeltaError, FieldId, LogicalClock, OverrideValue, PersistentId};
use crate::{Transaction, TransactionId};

/// Deterministic key for one object's one field.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MergeKey {
    /// Object identity.
    pub object: PersistentId,
    /// Field identity.
    pub field: FieldId,
}

/// One last-writer-wins register value and its ordering metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LwwRegister {
    clock: LogicalClock,
    actor: ActorId,
    transaction: TransactionId,
    value: OverrideValue,
}

impl LwwRegister {
    /// Creates a register candidate.
    pub fn new(
        clock: LogicalClock,
        actor: ActorId,
        transaction: TransactionId,
        value: OverrideValue,
    ) -> Result<Self, DeltaError> {
        value.validate().map_err(DeltaError::InvalidValue)?;
        Ok(Self {
            clock,
            actor,
            transaction,
            value,
        })
    }

    /// Returns the logical clock.
    pub const fn clock(&self) -> LogicalClock {
        self.clock
    }

    /// Returns the actor tie-break key.
    pub const fn actor(&self) -> ActorId {
        self.actor
    }

    /// Returns the transaction tie-break key.
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    /// Returns the stored value, including [`OverrideValue::Delete`].
    pub const fn value(&self) -> &OverrideValue {
        &self.value
    }

    fn ordering_key(&self) -> (LogicalClock, ActorId, TransactionId) {
        (self.clock, self.actor, self.transaction)
    }

    fn wins_over(&self, other: &Self) -> bool {
        self.ordering_key() > other.ordering_key()
    }
}

/// CRDT merge failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CrdtError {
    /// A transaction ID was supplied more than once.
    DuplicateTransaction(TransactionId),
    /// A transaction's field data was invalid.
    InvalidTransaction(DeltaError),
}

/// Deterministic field-level LWW state over object/field keys.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CrdtState {
    registers: BTreeMap<MergeKey, LwwRegister>,
    transactions: BTreeSet<TransactionId>,
}

impl CrdtState {
    /// Creates an empty CRDT state.
    pub const fn new() -> Self {
        Self {
            registers: BTreeMap::new(),
            transactions: BTreeSet::new(),
        }
    }

    /// Merges transactions in any arrival order. The resulting registers are
    /// determined only by `(clock, actor, transaction_id)` ordering.
    pub fn merge<'a, I>(&mut self, transactions: I) -> Result<(), CrdtError>
    where
        I: IntoIterator<Item = &'a Transaction>,
    {
        for transaction in transactions {
            if self.transactions.contains(&transaction.id()) {
                return Err(CrdtError::DuplicateTransaction(transaction.id()));
            }
            transaction
                .validate()
                .map_err(CrdtError::InvalidTransaction)?;
            self.transactions.insert(transaction.id());
            for change in transaction.delta().iter() {
                let key = MergeKey {
                    object: transaction.object(),
                    field: change.field,
                };
                let candidate = LwwRegister::new(
                    transaction.clock(),
                    transaction.actor(),
                    transaction.id(),
                    change.value.clone(),
                )
                .map_err(CrdtError::InvalidTransaction)?;
                match self.registers.get(&key) {
                    Some(current) if !candidate.wins_over(current) => {}
                    _ => {
                        self.registers.insert(key, candidate);
                    }
                }
            }
        }
        Ok(())
    }

    /// Merges a complete peer state and rejects repeated transaction IDs.
    pub fn merge_state(&mut self, other: &Self) -> Result<(), CrdtError> {
        if other
            .transactions
            .iter()
            .any(|id| self.transactions.contains(id))
        {
            let duplicate = other
                .transactions
                .iter()
                .find(|id| self.transactions.contains(id));
            if let Some(id) = duplicate {
                return Err(CrdtError::DuplicateTransaction(*id));
            }
        }
        self.transactions.extend(other.transactions.iter().copied());
        for (key, candidate) in &other.registers {
            match self.registers.get(key) {
                Some(current) if !candidate.wins_over(current) => {}
                _ => {
                    self.registers.insert(*key, candidate.clone());
                }
            }
        }
        Ok(())
    }

    /// Returns the winning register for an object/field key.
    pub fn register(&self, object: PersistentId, field: FieldId) -> Option<&LwwRegister> {
        self.registers.get(&MergeKey { object, field })
    }

    /// Returns the winning materialized value, hiding a deletion tombstone.
    pub fn get(&self, object: PersistentId, field: FieldId) -> Option<&OverrideValue> {
        let register = self.register(object, field)?;
        match register.value() {
            OverrideValue::Delete => None,
            value => Some(value),
        }
    }

    /// Iterates winning registers by object then field.
    pub fn iter(&self) -> alloc::collections::btree_map::Iter<'_, MergeKey, LwwRegister> {
        self.registers.iter()
    }

    /// Returns the number of object/field registers.
    pub fn len(&self) -> usize {
        self.registers.len()
    }

    /// Returns true when no registers exist.
    pub fn is_empty(&self) -> bool {
        self.registers.is_empty()
    }
}
