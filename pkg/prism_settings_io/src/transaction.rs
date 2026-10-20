//! Operation-level transaction execution over the durable settings journal.

use crate::{
    reconstruct_transaction, recover_journal, ApplyOp, Journal, JournalError, JournalEvent,
    Reversibility, TxMode, TxOutcome,
};
use std::{error::Error, fmt};

/// Ordered set of side-effect operations produced by preflight.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyPlan {
    /// Operations in execution order.
    pub ops: Vec<ApplyOp>,
    /// Highest reversibility class in the plan.
    pub max_reversibility: Reversibility,
    /// First irreversible operation, if any.
    pub irreversible_index: Option<u16>,
}

impl ApplyPlan {
    /// Builds and validates a deterministic plan.
    pub fn new(mut ops: Vec<ApplyOp>) -> Result<Self, ApplyPlanError> {
        ops.sort_by_key(|op| op.index);
        for (expected, op) in ops.iter().enumerate() {
            let expected =
                u16::try_from(expected).map_err(|_| ApplyPlanError::TooManyOperations)?;
            if op.index != expected {
                return Err(ApplyPlanError::NonContiguousIndex {
                    expected,
                    actual: op.index,
                });
            }
        }
        let max_reversibility = ops
            .iter()
            .map(|op| op.reversibility)
            .max()
            .unwrap_or(Reversibility::Reversible);
        let irreversible_index = ops
            .iter()
            .find(|op| op.reversibility == Reversibility::Irreversible)
            .map(|op| op.index);
        if let Some(index) = irreversible_index
            && ops.iter().any(|op| op.index > index)
        {
            return Err(ApplyPlanError::IrreversibleNotLast { index });
        }
        Ok(Self {
            ops,
            max_reversibility,
            irreversible_index,
        })
    }
}

/// Errors found before any side effect is invoked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApplyPlanError {
    /// The plan contains more than `u16::MAX` operations.
    TooManyOperations,
    /// Operation indices are not zero-based and contiguous.
    NonContiguousIndex {
        /// Expected index.
        expected: u16,
        /// Actual index.
        actual: u16,
    },
    /// An irreversible operation is not at the forward-only boundary.
    IrreversibleNotLast {
        /// First irreversible operation index.
        index: u16,
    },
}

impl fmt::Display for ApplyPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyOperations => formatter.write_str("apply plan has too many operations"),
            Self::NonContiguousIndex { expected, actual } => {
                write!(
                    formatter,
                    "apply plan expected op index {expected}, got {actual}"
                )
            }
            Self::IrreversibleNotLast { index } => {
                write!(
                    formatter,
                    "irreversible op {index} is not at the forward-only boundary"
                )
            }
        }
    }
}

impl Error for ApplyPlanError {}

/// Result returned by an applier after an operation invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpOutcome {
    /// The effect was applied and may return an opaque recovery handle.
    Applied {
        /// Opaque handle for probing or compensation.
        effect_handle: Option<String>,
        /// Digest of result metadata.
        result_digest: u64,
    },
    /// The operation failed; the handle is retained if a partial effect exists.
    Failed {
        /// Redacted error text.
        error: String,
        /// Opaque partial-effect handle.
        effect_handle: Option<String>,
    },
    /// The operation was queued without producing a side effect.
    Deferred,
    /// A non-idempotent operation was already attempted.
    AlreadyAttempted,
}

/// Result returned by an applier during compensation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompensateOutcome {
    /// The old effect was restored.
    Done {
        /// Whether restoration was independently verified.
        verified: bool,
    },
    /// Restoration failed.
    Failed {
        /// Redacted error text.
        error: String,
    },
    /// This operation cannot be compensated.
    Unsupported,
}

/// Applies and probes one operation group.
pub trait SettingsApplier {
    /// Stable group name matched against [`ApplyOp::group`].
    fn group(&self) -> &str;
    /// Deterministic group order used by the caller when constructing plans.
    fn order(&self) -> i32;
    /// Applies one operation after its `OpBegin` record is durable.
    fn apply_op(&mut self, op: &ApplyOp) -> OpOutcome;
    /// Probes an effect after an ambiguous crash.
    fn probe(&self, op: &ApplyOp) -> crate::EffectState;
    /// Attempts to restore the old effect.
    fn compensate(&mut self, op: &ApplyOp) -> CompensateOutcome;
}

/// Errors raised while executing a journaled transaction.
#[derive(Debug)]
pub enum TransactionError {
    /// The plan is invalid.
    Plan(ApplyPlanError),
    /// The durable journal does not describe a committed state exchange matching this plan.
    InvalidTransaction(String),
    /// The journal could not be written or synced.
    Journal(JournalError),
    /// No applier was registered for an operation group.
    MissingApplier(String),
    /// A deferred or already-attempted operation cannot complete a direct apply.
    OperationNotApplied(u16),
}

impl fmt::Display for TransactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => write!(formatter, "invalid apply plan: {error}"),
            Self::InvalidTransaction(error) => write!(formatter, "invalid transaction: {error}"),
            Self::Journal(error) => write!(formatter, "transaction journal failed: {error}"),
            Self::MissingApplier(group) => write!(formatter, "no applier for group `{group}`"),
            Self::OperationNotApplied(index) => {
                write!(formatter, "operation {index} did not apply")
            }
        }
    }
}

impl Error for TransactionError {}

impl From<ApplyPlanError> for TransactionError {
    fn from(value: ApplyPlanError) -> Self {
        Self::Plan(value)
    }
}

impl From<JournalError> for TransactionError {
    fn from(value: JournalError) -> Self {
        Self::Journal(value)
    }
}

/// Result of a transaction after operation execution and compensation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionRunOutcome {
    /// Every operation completed successfully.
    Committed,
    /// State remains committed but an external/irreversible effect degraded.
    Degraded,
    /// Reversible effects were compensated.
    RolledBack,
    /// Compensation or journal state is ambiguous.
    RecoveryRequired,
}

/// Executes a validated plan and records every operation boundary.
pub fn run_transaction(
    journal: &mut Journal,
    plan: &ApplyPlan,
    appliers: &mut [&mut dyn SettingsApplier],
) -> Result<TransactionRunOutcome, TransactionError> {
    validate_transaction(journal, plan)?;
    let mut applied = Vec::new();
    let mut degraded = false;
    for op in &plan.ops {
        let Some(applier) = appliers
            .iter_mut()
            .find(|applier| applier.group() == op.group)
        else {
            return Err(TransactionError::MissingApplier(op.group.clone()));
        };
        if op.reversibility == Reversibility::Irreversible {
            journal.append(JournalEvent::IrreversibleBarrier { op_index: op.index })?;
        }
        journal.append(JournalEvent::OpBegin { op: op.clone() })?;
        let outcome = applier.apply_op(op);
        match outcome {
            OpOutcome::Applied {
                effect_handle,
                result_digest,
            } => {
                if let Err(error) = journal.append(JournalEvent::OpApplied {
                    op_index: op.index,
                    effect_handle: effect_handle.clone(),
                    result_digest,
                }) {
                    return uncertain_after_effect(journal, error);
                }
                let mut applied_op = op.clone();
                applied_op.effect_handle = effect_handle;
                applied.push(applied_op);
                if op.reversibility == Reversibility::Irreversible
                    || op.reversibility == Reversibility::External
                {
                    degraded = true;
                }
            }
            OpOutcome::Failed {
                error,
                effect_handle,
            } => {
                if let Err(journal_error) = journal.append(JournalEvent::OpFailed {
                    op_index: op.index,
                    error,
                    effect_handle: effect_handle.clone(),
                }) {
                    return uncertain_after_effect(journal, journal_error);
                }
                if effect_handle.is_some()
                    && op.reversibility != Reversibility::External
                    && op.reversibility != Reversibility::Irreversible
                {
                    let mut partial = op.clone();
                    partial.effect_handle = effect_handle;
                    applied.push(partial);
                }
                // Once an irreversible operation is invoked, the transaction is
                // forward-only even when the operation reports failure. External
                // effects are outside our rollback authority as well.
                if op.reversibility == Reversibility::Irreversible {
                    return finish(journal, TransactionRunOutcome::Degraded);
                }
                if op.reversibility == Reversibility::External {
                    return finish(journal, TransactionRunOutcome::RecoveryRequired);
                }
                if applied.iter().any(|applied| {
                    applied.reversibility == Reversibility::Irreversible
                        || applied.reversibility == Reversibility::External
                }) {
                    return finish(journal, TransactionRunOutcome::Degraded);
                }
                return compensate_applied(journal, appliers, applied);
            }
            OpOutcome::Deferred => {
                journal.append(JournalEvent::OpFailed {
                    op_index: op.index,
                    error: "deferred operation is not supported by direct apply".to_owned(),
                    effect_handle: None,
                })?;
                return Err(TransactionError::OperationNotApplied(op.index));
            }
            OpOutcome::AlreadyAttempted => {
                journal.append(JournalEvent::OpFailed {
                    op_index: op.index,
                    error: "operation was already attempted".to_owned(),
                    effect_handle: None,
                })?;
                return Err(TransactionError::OperationNotApplied(op.index));
            }
        }
    }
    finish(
        journal,
        if degraded {
            TransactionRunOutcome::Degraded
        } else {
            TransactionRunOutcome::Committed
        },
    )
}

fn validate_transaction(journal: &Journal, plan: &ApplyPlan) -> Result<(), TransactionError> {
    let records = recover_journal(journal.path())?.records;
    let tx = reconstruct_transaction(&records)
        .map_err(|error| TransactionError::InvalidTransaction(error.to_string()))?;
    let Some(JournalEvent::TxBegin { mode, op_count, .. }) = tx.begin else {
        return Err(TransactionError::InvalidTransaction(
            "durable TxBegin is missing".to_owned(),
        ));
    };
    if tx.ended.is_some() {
        return Err(TransactionError::InvalidTransaction(
            "transaction already has a terminal outcome".to_owned(),
        ));
    }
    if !tx.snapshot_committed {
        return Err(TransactionError::InvalidTransaction(
            "snapshot has not been durably committed".to_owned(),
        ));
    }
    if usize::from(op_count) != plan.ops.len() {
        return Err(TransactionError::InvalidTransaction(format!(
            "TxBegin declares {op_count} operations but plan contains {}",
            plan.ops.len()
        )));
    }
    if mode == TxMode::Preview
        && plan.ops.iter().any(|op| {
            matches!(
                op.reversibility,
                Reversibility::Irreversible | Reversibility::External
            )
        })
    {
        return Err(TransactionError::InvalidTransaction(
            "preview plans cannot contain irreversible or external operations".to_owned(),
        ));
    }
    Ok(())
}

fn compensate_applied(
    journal: &mut Journal,
    appliers: &mut [&mut dyn SettingsApplier],
    mut applied: Vec<ApplyOp>,
) -> Result<TransactionRunOutcome, TransactionError> {
    while let Some(op) = applied.pop() {
        if op.reversibility == Reversibility::Irreversible
            || op.reversibility == Reversibility::External
        {
            return finish(journal, TransactionRunOutcome::Degraded);
        }
        if let Err(error) = journal.append(JournalEvent::CompensateBegin {
            op_index: op.index,
            reason: "operation failure".to_owned(),
        }) {
            return uncertain_after_effect(journal, error);
        }
        let Some(applier) = appliers
            .iter_mut()
            .find(|applier| applier.group() == op.group)
        else {
            return finish(journal, TransactionRunOutcome::RecoveryRequired);
        };
        match applier.compensate(&op) {
            CompensateOutcome::Done { verified } => {
                if let Err(error) = journal.append(JournalEvent::Compensated {
                    op_index: op.index,
                    verified,
                }) {
                    return uncertain_after_effect(journal, error);
                }
                if !verified {
                    return finish(journal, TransactionRunOutcome::RecoveryRequired);
                }
            }
            CompensateOutcome::Failed { error } => {
                if let Err(journal_error) = journal.append(JournalEvent::CompensateFailed {
                    op_index: op.index,
                    error,
                }) {
                    return uncertain_after_effect(journal, journal_error);
                }
                return finish(journal, TransactionRunOutcome::RecoveryRequired);
            }
            CompensateOutcome::Unsupported => {
                journal.append(JournalEvent::CompensateFailed {
                    op_index: op.index,
                    error: "unsupported".to_owned(),
                })?;
                return finish(journal, TransactionRunOutcome::RecoveryRequired);
            }
        }
    }
    finish(journal, TransactionRunOutcome::RolledBack)
}

fn finish(
    journal: &mut Journal,
    outcome: TransactionRunOutcome,
) -> Result<TransactionRunOutcome, TransactionError> {
    let durable_outcome = match outcome {
        TransactionRunOutcome::Committed => TxOutcome::Committed,
        TransactionRunOutcome::Degraded => TxOutcome::Degraded,
        TransactionRunOutcome::RolledBack => TxOutcome::RolledBack,
        TransactionRunOutcome::RecoveryRequired => TxOutcome::RecoveryRequired,
    };
    journal.append(JournalEvent::TxEnd {
        outcome: durable_outcome,
    })?;
    Ok(outcome)
}

fn uncertain_after_effect(
    journal: &mut Journal,
    original_error: JournalError,
) -> Result<TransactionRunOutcome, TransactionError> {
    match finish(journal, TransactionRunOutcome::RecoveryRequired) {
        Ok(outcome) => Ok(outcome),
        Err(_) => Err(TransactionError::Journal(original_error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reconstruct_transaction, recover_journal, ApplyOp, EffectState, JournalEvent};
    use prism_settings::SettingKey;
    use std::cell::Cell;
    use std::cell::RefCell;
    use tempfile::tempdir;

    struct MockApplier {
        fail_on: Option<u16>,
        verify_compensation: bool,
        compensated: Cell<u32>,
        compensation_order: RefCell<Vec<u16>>,
    }

    impl SettingsApplier for MockApplier {
        fn group(&self) -> &str {
            "Audio"
        }

        fn order(&self) -> i32 {
            0
        }

        fn apply_op(&mut self, op: &ApplyOp) -> OpOutcome {
            if self.fail_on == Some(op.index) {
                OpOutcome::Failed {
                    error: "mock failure".to_owned(),
                    effect_handle: None,
                }
            } else {
                OpOutcome::Applied {
                    effect_handle: Some(format!("handle:{}", op.index)),
                    result_digest: 42,
                }
            }
        }

        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }

        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            self.compensated.set(self.compensated.get() + 1);
            self.compensation_order.borrow_mut().push(_op.index);
            CompensateOutcome::Done {
                verified: self.verify_compensation,
            }
        }
    }

    fn op(index: u16) -> ApplyOp {
        ApplyOp {
            index,
            id: format!("audio-{index}"),
            group: "Audio".to_owned(),
            key: SettingKey::new(format!("audio.level.{index}")),
            action: crate::OpAction::Set,
            reversibility: Reversibility::Reversible,
            idempotent: true,
            input_digest: u64::from(index),
            effect_handle: None,
        }
    }

    fn begin_transaction(journal: &mut Journal, mode: TxMode, op_count: usize) {
        journal
            .append(JournalEvent::TxBegin {
                mode,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: u16::try_from(op_count).unwrap(),
                snapshot_gen: 4,
                last_confirmed: None,
            })
            .unwrap();
        journal
            .append(JournalEvent::SnapshotCommitted { snapshot_gen: 5 })
            .unwrap();
    }

    #[test]
    fn run_records_boundaries_and_terminal_commit() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-run").unwrap();
        let plan = ApplyPlan::new(vec![op(0)]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        let mut applier = MockApplier {
            fail_on: None,
            verify_compensation: true,
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            run_transaction(&mut journal, &plan, &mut appliers).unwrap(),
            TransactionRunOutcome::Committed
        );
        let records = recover_journal(&path).unwrap().records;
        let tx = reconstruct_transaction(&records).unwrap();
        assert_eq!(tx.ended, Some(TxOutcome::Committed));
        assert!(records
            .iter()
            .any(|record| matches!(record.event, JournalEvent::OpBegin { .. })));
    }

    #[test]
    fn failed_operation_compensates_in_reverse_order() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-fail").unwrap();
        let plan = ApplyPlan::new(vec![op(0), op(1), op(2)]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        let mut applier = MockApplier {
            fail_on: Some(2),
            verify_compensation: true,
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            run_transaction(&mut journal, &plan, &mut appliers).unwrap(),
            TransactionRunOutcome::RolledBack
        );
        assert_eq!(applier.compensated.get(), 2);
        assert_eq!(&*applier.compensation_order.borrow(), &[1, 0]);
    }

    #[test]
    fn failed_irreversible_operation_is_degraded_without_compensation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-irreversible-fail").unwrap();
        let mut irreversible = op(0);
        irreversible.reversibility = Reversibility::Irreversible;
        let plan = ApplyPlan::new(vec![irreversible]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        let mut applier = MockApplier {
            fail_on: Some(0),
            verify_compensation: true,
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert_eq!(
            run_transaction(&mut journal, &plan, &mut appliers).unwrap(),
            TransactionRunOutcome::Degraded
        );
        assert_eq!(applier.compensated.get(), 0);
        let records = recover_journal(&path).unwrap().records;
        assert!(records.iter().any(|record| matches!(
            record.event,
            JournalEvent::TxEnd {
                outcome: TxOutcome::Degraded
            }
        )));
    }

    #[test]
    fn unverified_compensation_requires_recovery() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-unverified").unwrap();
        let plan = ApplyPlan::new(vec![op(0), op(1)]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        let mut applier = MockApplier {
            fail_on: Some(1),
            verify_compensation: false,
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert_eq!(
            run_transaction(&mut journal, &plan, &mut appliers).unwrap(),
            TransactionRunOutcome::RecoveryRequired
        );
        assert_eq!(applier.compensated.get(), 1);
    }
}
