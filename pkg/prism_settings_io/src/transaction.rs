//! Operation-level transaction execution over the durable settings journal.

use crate::{
    reconstruct_transaction, recover_journal, ApplyOp, ApplyOpState, Journal, JournalError,
    JournalEvent, Reversibility, TxMode, TxOutcome,
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

/// Restores the settings state represented by a durable generation.
///
/// The journal intentionally contains hashes and generation identities rather
/// than setting values. A host supplies the durable snapshot lookup at this
/// boundary during process-start recovery.
pub trait SnapshotRecovery {
    /// Restores the requested generation and verifies its hash.
    fn restore(&mut self, generation: u64, hash: u64) -> Result<(), String>;
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
    /// An irreversible effect crossed its barrier and requires a restart.
    NeedsRestart,
    /// Compensation or journal state is ambiguous.
    RecoveryRequired,
    /// A reversible preview is installed and awaits confirmation.
    PendingConfirmation,
}

/// Executes a validated plan and records every operation boundary.
pub fn run_transaction(
    journal: &mut Journal,
    plan: &ApplyPlan,
    appliers: &mut [&mut dyn SettingsApplier],
) -> Result<TransactionRunOutcome, TransactionError> {
    validate_transaction(journal, plan, TxMode::Direct)?;
    let outcome = execute_operations(journal, plan, appliers)?;
    if reconstruct_transaction(&recover_journal(journal.path())?.records)?
        .ended
        .is_some()
    {
        return Ok(outcome);
    }
    finish(journal, outcome)
}

/// Executes a reversible preview and leaves a durable pending-confirmation
/// marker when every operation succeeds.
pub fn run_preview_transaction(
    journal: &mut Journal,
    plan: &ApplyPlan,
    appliers: &mut [&mut dyn SettingsApplier],
) -> Result<TransactionRunOutcome, TransactionError> {
    validate_transaction(journal, plan, TxMode::Preview)?;
    let outcome = execute_operations(journal, plan, appliers)?;
    if outcome == TransactionRunOutcome::Committed {
        journal.append(JournalEvent::TxEnd {
            outcome: TxOutcome::PendingConfirmation,
        })?;
        Ok(TransactionRunOutcome::PendingConfirmation)
    } else if reconstruct_transaction(&recover_journal(journal.path())?.records)?
        .ended
        .is_some()
    {
        Ok(outcome)
    } else {
        finish(journal, outcome)
    }
}

struct ExecutionState {
    applied: Vec<ApplyOp>,
    degraded: bool,
    needs_restart: bool,
}

fn execute_operations(
    journal: &mut Journal,
    plan: &ApplyPlan,
    appliers: &mut [&mut dyn SettingsApplier],
) -> Result<TransactionRunOutcome, TransactionError> {
    let mut state = ExecutionState {
        applied: Vec::new(),
        degraded: false,
        needs_restart: false,
    };
    let durable = reconstruct_transaction(&recover_journal(journal.path())?.records)
        .map_err(|error| TransactionError::InvalidTransaction(error.to_string()))?;
    for op in &plan.ops {
        let Some(applier) = appliers
            .iter_mut()
            .find(|applier| applier.group() == op.group)
        else {
            return Err(TransactionError::MissingApplier(op.group.clone()));
        };

        if let Some(existing) = durable.operations.get(&op.index) {
            if !same_operation_metadata(&existing.op, op) {
                return Err(TransactionError::InvalidTransaction(format!(
                    "journal operation {} does not match the supplied plan",
                    op.index
                )));
            }
            match existing.state {
                ApplyOpState::Applied => {
                    let mut applied = op.clone();
                    applied.effect_handle = existing.op.effect_handle.clone();
                    state.applied.push(applied);
                    if matches!(op.reversibility, Reversibility::External) {
                        state.degraded = true;
                    }
                    if op.reversibility == Reversibility::Irreversible {
                        state.needs_restart = true;
                    }
                    continue;
                }
                ApplyOpState::Unknown => match applier.probe(op) {
                    crate::EffectState::Applied => {
                        journal.append(JournalEvent::OpApplied {
                            op_index: op.index,
                            effect_handle: existing.op.effect_handle.clone(),
                            result_digest: 0,
                        })?;
                        let mut applied = op.clone();
                        applied.effect_handle = existing.op.effect_handle.clone();
                        state.applied.push(applied);
                        if !op.idempotent {
                            state.degraded = true;
                        }
                        continue;
                    }
                    crate::EffectState::NotApplied if op.idempotent => {
                        // A durable OpBegin without an effect can be retried
                        // only when the operation explicitly promises idempotence.
                    }
                    crate::EffectState::NotApplied => {
                        return finish(journal, TransactionRunOutcome::RecoveryRequired);
                    }
                    crate::EffectState::Unknown => {
                        return finish(journal, TransactionRunOutcome::RecoveryRequired);
                    }
                },
                ApplyOpState::Failed if !op.idempotent => {
                    return finish(journal, TransactionRunOutcome::RecoveryRequired);
                }
                ApplyOpState::Failed => {}
                ApplyOpState::Pending => {}
            }
        }

        if op.reversibility == Reversibility::Irreversible {
            journal.append(JournalEvent::IrreversibleBarrier { op_index: op.index })?;
        }
        journal.append(JournalEvent::OpBegin { op: op.clone() })?;
        match applier.apply_op(op) {
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
                let mut applied = op.clone();
                applied.effect_handle = effect_handle;
                state.applied.push(applied);
                if matches!(op.reversibility, Reversibility::External) {
                    state.degraded = true;
                }
                if op.reversibility == Reversibility::Irreversible {
                    state.needs_restart = true;
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
                    state.applied.push(partial);
                }
                if op.reversibility == Reversibility::Irreversible {
                    return Ok(TransactionRunOutcome::Degraded);
                }
                if op.reversibility == Reversibility::External {
                    return Ok(TransactionRunOutcome::RecoveryRequired);
                }
                if state.applied.iter().any(|applied| {
                    matches!(
                        applied.reversibility,
                        Reversibility::Irreversible | Reversibility::External
                    )
                }) {
                    return Ok(TransactionRunOutcome::Degraded);
                }
                return compensate_applied(journal, appliers, state.applied);
            }
            OpOutcome::Deferred | OpOutcome::AlreadyAttempted => {
                journal.append(JournalEvent::OpFailed {
                    op_index: op.index,
                    error: "operation did not produce a direct effect".to_owned(),
                    effect_handle: None,
                })?;
                return Ok(TransactionRunOutcome::RecoveryRequired);
            }
        }
    }
    Ok(if state.needs_restart {
        TransactionRunOutcome::NeedsRestart
    } else if state.degraded {
        TransactionRunOutcome::Degraded
    } else {
        TransactionRunOutcome::Committed
    })
}

fn same_operation_metadata(left: &ApplyOp, right: &ApplyOp) -> bool {
    left.index == right.index
        && left.id == right.id
        && left.group == right.group
        && left.key == right.key
        && left.action == right.action
        && left.reversibility == right.reversibility
        && left.idempotent == right.idempotent
        && left.input_digest == right.input_digest
}

fn validate_transaction(
    journal: &Journal,
    plan: &ApplyPlan,
    expected_mode: TxMode,
) -> Result<(), TransactionError> {
    let records = recover_journal(journal.path())?.records;
    let tx = reconstruct_transaction(&records)
        .map_err(|error| TransactionError::InvalidTransaction(error.to_string()))?;
    let Some(JournalEvent::TxBegin { mode, op_count, .. }) = tx.begin else {
        return Err(TransactionError::InvalidTransaction(
            "durable TxBegin is missing".to_owned(),
        ));
    };
    if tx
        .ended
        .is_some_and(|outcome| outcome != TxOutcome::PendingConfirmation)
    {
        return Err(TransactionError::InvalidTransaction(
            "transaction already has a terminal outcome".to_owned(),
        ));
    }
    if mode != expected_mode {
        return Err(TransactionError::InvalidTransaction(
            "transaction mode does not match the requested executor".to_owned(),
        ));
    }
    if !tx.snapshot_committed {
        return Err(TransactionError::InvalidTransaction(
            "snapshot has not been durably committed".to_owned(),
        ));
    }
    if mode == TxMode::Direct && !tx.persisted {
        return Err(TransactionError::InvalidTransaction(
            "direct operations require a matching durable PersistEnd".to_owned(),
        ));
    }
    if mode == TxMode::Direct && !tx.operations.is_empty() {
        return Err(TransactionError::InvalidTransaction(
            "direct operation recovery must use the journal recovery executor".to_owned(),
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

/// Rolls back an open transaction, including a preview waiting for confirmation.
pub fn rollback_open_transaction(
    journal: &mut Journal,
    appliers: &mut [&mut dyn SettingsApplier],
    snapshots: &mut dyn SnapshotRecovery,
) -> Result<TransactionRunOutcome, TransactionError> {
    let records = recover_journal(journal.path())?.records;
    let tx = reconstruct_transaction(&records)
        .map_err(|error| TransactionError::InvalidTransaction(error.to_string()))?;
    if tx.ended == Some(TxOutcome::Committed)
        || tx.ended == Some(TxOutcome::RolledBack)
        || tx.ended == Some(TxOutcome::Degraded)
        || tx.ended == Some(TxOutcome::NeedsRestart)
        || tx.ended == Some(TxOutcome::RecoveryRequired)
    {
        return Err(TransactionError::InvalidTransaction(
            "transaction is not an open preview".to_owned(),
        ));
    }
    rollback_reconstructed(journal, &tx, appliers, snapshots)
}

/// Recovers an interrupted transaction from its durable prefix.
pub fn recover_open_transaction(
    journal: &mut Journal,
    snapshots: &mut dyn SnapshotRecovery,
    appliers: &mut [&mut dyn SettingsApplier],
) -> Result<TransactionRunOutcome, TransactionError> {
    let records = recover_journal(journal.path())?.records;
    let tx = reconstruct_transaction(&records)
        .map_err(|error| TransactionError::InvalidTransaction(error.to_string()))?;
    if tx.persistence_in_flight {
        if let Some(JournalEvent::TxBegin {
            snapshot_gen,
            hash_before,
            ..
        }) = tx.begin.as_ref()
        {
            snapshots
                .restore(*snapshot_gen, *hash_before)
                .map_err(TransactionError::InvalidTransaction)?;
        }
        return finish(journal, TransactionRunOutcome::RecoveryRequired);
    }
    match tx.recovery_decision() {
        crate::TxRecoveryDecision::Completed => {
            return Ok(match tx.ended {
                Some(TxOutcome::Committed) => TransactionRunOutcome::Committed,
                Some(TxOutcome::Degraded) => TransactionRunOutcome::Degraded,
                Some(TxOutcome::RolledBack) => TransactionRunOutcome::RolledBack,
                Some(TxOutcome::RecoveryRequired) => TransactionRunOutcome::RecoveryRequired,
                Some(TxOutcome::NeedsRestart) => TransactionRunOutcome::NeedsRestart,
                Some(TxOutcome::PendingConfirmation) | None => {
                    TransactionRunOutcome::RecoveryRequired
                }
            });
        }
        crate::TxRecoveryDecision::Discard => {
            return finish(journal, TransactionRunOutcome::RolledBack);
        }
        crate::TxRecoveryDecision::NeedsRestart => {
            return finish(journal, TransactionRunOutcome::NeedsRestart);
        }
        crate::TxRecoveryDecision::Commit => {
            return finish(journal, TransactionRunOutcome::Committed);
        }
        crate::TxRecoveryDecision::RecoveryRequired => {
            return finish(journal, TransactionRunOutcome::RecoveryRequired);
        }
        crate::TxRecoveryDecision::PendingConfirmation | crate::TxRecoveryDecision::Rollback => {}
    }

    if let Some(barrier) = tx.irreversible_barrier {
        let Some(operation) = tx.operations.get(&barrier) else {
            return rollback_reconstructed(journal, &tx, appliers, snapshots);
        };
        let state = match operation.state {
            ApplyOpState::Applied => crate::EffectState::Applied,
            ApplyOpState::Unknown => appliers
                .iter()
                .find(|applier| applier.group() == operation.op.group)
                .ok_or_else(|| TransactionError::MissingApplier(operation.op.group.clone()))?
                .probe(&operation.op),
            ApplyOpState::Failed | ApplyOpState::Pending => crate::EffectState::NotApplied,
        };
        match state {
            crate::EffectState::Applied => {
                journal.append(JournalEvent::OpApplied {
                    op_index: barrier,
                    effect_handle: operation.op.effect_handle.clone(),
                    result_digest: 0,
                })?;
                return finish(journal, TransactionRunOutcome::NeedsRestart);
            }
            crate::EffectState::NotApplied => {
                journal.append(JournalEvent::OpFailed {
                    op_index: barrier,
                    error: "recovery probe: effect not applied".to_owned(),
                    effect_handle: None,
                })?;
                return finish(journal, TransactionRunOutcome::NeedsRestart);
            }
            crate::EffectState::Unknown => {
                return finish(journal, TransactionRunOutcome::RecoveryRequired);
            }
        }
    }

    rollback_reconstructed(journal, &tx, appliers, snapshots)
}

fn rollback_reconstructed(
    journal: &mut Journal,
    tx: &crate::JournalTx,
    appliers: &mut [&mut dyn SettingsApplier],
    snapshots: &mut dyn SnapshotRecovery,
) -> Result<TransactionRunOutcome, TransactionError> {
    let mut operations: Vec<ApplyOp> = tx
        .operations
        .values()
        .filter(|state| state.state == ApplyOpState::Applied)
        .map(|state| state.op.clone())
        .collect();
    operations.sort_by_key(|op| std::cmp::Reverse(op.index));

    for state in tx.operations.values() {
        if state.compensation == crate::CompensationState::Unknown
            || state.compensation == crate::CompensationState::Failed
        {
            return finish(journal, TransactionRunOutcome::RecoveryRequired);
        }
    }

    for state in tx.operations.values() {
        if state.state != ApplyOpState::Unknown {
            continue;
        }
        let Some(applier) = appliers
            .iter_mut()
            .find(|applier| applier.group() == state.op.group)
        else {
            return Err(TransactionError::MissingApplier(state.op.group.clone()));
        };
        match applier.probe(&state.op) {
            crate::EffectState::Applied => operations.push(state.op.clone()),
            crate::EffectState::NotApplied => {}
            crate::EffectState::Unknown
                if state.op.idempotent
                    && matches!(
                        state.op.reversibility,
                        Reversibility::Reversible | Reversibility::Compensable
                    ) =>
            {
                operations.push(state.op.clone())
            }
            crate::EffectState::Unknown => {
                return finish(journal, TransactionRunOutcome::RecoveryRequired);
            }
        }
    }
    operations.sort_by_key(|op| std::cmp::Reverse(op.index));
    operations.dedup_by_key(|op| op.index);
    for op in operations {
        let Some(applier) = appliers
            .iter_mut()
            .find(|applier| applier.group() == op.group)
        else {
            return Err(TransactionError::MissingApplier(op.group.clone()));
        };
        journal.append(JournalEvent::CompensateBegin {
            op_index: op.index,
            reason: "journal recovery".to_owned(),
        })?;
        match applier.compensate(&op) {
            CompensateOutcome::Done { verified } => {
                journal.append(JournalEvent::Compensated {
                    op_index: op.index,
                    verified,
                })?;
                if !verified {
                    return finish(journal, TransactionRunOutcome::RecoveryRequired);
                }
            }
            CompensateOutcome::Failed { error } => {
                journal.append(JournalEvent::CompensateFailed {
                    op_index: op.index,
                    error,
                })?;
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

    let Some(JournalEvent::TxBegin {
        snapshot_gen,
        hash_before,
        ..
    }) = tx.begin.as_ref()
    else {
        return finish(journal, TransactionRunOutcome::RolledBack);
    };
    snapshots
        .restore(*snapshot_gen, *hash_before)
        .map_err(TransactionError::InvalidTransaction)?;
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
        TransactionRunOutcome::NeedsRestart => TxOutcome::NeedsRestart,
        TransactionRunOutcome::RecoveryRequired => TxOutcome::RecoveryRequired,
        TransactionRunOutcome::PendingConfirmation => TxOutcome::PendingConfirmation,
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
        applied: Cell<u32>,
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
            self.applied.set(self.applied.get() + 1);
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
        if mode == TxMode::Direct {
            journal
                .append(JournalEvent::PersistBegin {
                    generation: 6,
                    manifest_hash: 7,
                })
                .unwrap();
            journal
                .append(JournalEvent::PersistEnd {
                    generation: 6,
                    manifest_hash: 7,
                })
                .unwrap();
        }
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
            applied: Cell::new(0),
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
    fn direct_execution_requires_persist_end_before_any_effect() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-direct-not-persisted").unwrap();
        let plan = ApplyPlan::new(vec![op(0)]).unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: plan.ops.len() as u16,
                snapshot_gen: 4,
                last_confirmed: None,
            })
            .unwrap();
        journal
            .append(JournalEvent::SnapshotCommitted { snapshot_gen: 5 })
            .unwrap();
        let mut applier = MockApplier {
            fail_on: None,
            verify_compensation: true,
            applied: Cell::new(0),
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert!(matches!(
            run_transaction(&mut journal, &plan, &mut appliers),
            Err(TransactionError::InvalidTransaction(_))
        ));
        assert_eq!(applier.applied.get(), 0);
    }

    #[test]
    fn direct_execution_rejects_an_existing_operation_prefix() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-direct-prefix").unwrap();
        let plan = ApplyPlan::new(vec![op(0)]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        journal
            .append(JournalEvent::PersistBegin {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::PersistBegin {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal.append(JournalEvent::OpBegin { op: op(0) }).unwrap();
        let mut applier = MockApplier {
            fail_on: None,
            verify_compensation: true,
            applied: Cell::new(0),
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert!(matches!(
            run_transaction(&mut journal, &plan, &mut appliers),
            Err(TransactionError::InvalidTransaction(_))
        ));
        assert_eq!(applier.applied.get(), 0);
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
            applied: Cell::new(0),
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
            applied: Cell::new(0),
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
            applied: Cell::new(0),
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

    #[test]
    fn replay_skips_a_durably_applied_idempotent_operation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-replay").unwrap();
        let plan = ApplyPlan::new(vec![op(0)]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        journal
            .append(JournalEvent::PersistBegin {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal.append(JournalEvent::OpBegin { op: op(0) }).unwrap();
        journal
            .append(JournalEvent::OpApplied {
                op_index: 0,
                effect_handle: Some("already-applied".to_owned()),
                result_digest: 9,
            })
            .unwrap();

        let mut applier = MockApplier {
            fail_on: None,
            verify_compensation: true,
            applied: Cell::new(0),
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        let mut snapshots = RecoverySnapshots {
            restored: Cell::new(0),
        };
        assert_eq!(
            recover_open_transaction(&mut journal, &mut snapshots, &mut appliers).unwrap(),
            TransactionRunOutcome::Committed
        );
        assert_eq!(applier.applied.get(), 0);
    }

    #[test]
    fn non_idempotent_in_flight_operation_is_not_replayed() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-non-idempotent").unwrap();
        let mut non_idempotent = op(0);
        non_idempotent.idempotent = false;
        let plan = ApplyPlan::new(vec![non_idempotent.clone()]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        journal
            .append(JournalEvent::PersistBegin {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::OpBegin { op: non_idempotent })
            .unwrap();

        let mut applier = MockApplier {
            fail_on: None,
            verify_compensation: true,
            applied: Cell::new(0),
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert!(matches!(
            run_transaction(&mut journal, &plan, &mut appliers),
            Err(TransactionError::InvalidTransaction(_))
        ));
        assert_eq!(applier.applied.get(), 0);
    }

    struct RecoverySnapshots {
        restored: Cell<u32>,
    }

    impl SnapshotRecovery for RecoverySnapshots {
        fn restore(&mut self, _generation: u64, _hash: u64) -> Result<(), String> {
            self.restored.set(self.restored.get() + 1);
            Ok(())
        }
    }

    #[test]
    fn irreversible_recovery_returns_needs_restart() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-needs-restart").unwrap();
        let mut irreversible = op(0);
        irreversible.reversibility = Reversibility::Irreversible;
        let plan = ApplyPlan::new(vec![irreversible.clone()]).unwrap();
        begin_transaction(&mut journal, TxMode::Direct, plan.ops.len());
        journal
            .append(JournalEvent::IrreversibleBarrier { op_index: 0 })
            .unwrap();
        journal
            .append(JournalEvent::OpBegin {
                op: irreversible.clone(),
            })
            .unwrap();
        journal
            .append(JournalEvent::OpApplied {
                op_index: 0,
                effect_handle: Some("irreversible".to_owned()),
                result_digest: 1,
            })
            .unwrap();

        let mut snapshots = RecoverySnapshots {
            restored: Cell::new(0),
        };
        let mut applier = MockApplier {
            fail_on: None,
            verify_compensation: true,
            applied: Cell::new(0),
            compensated: Cell::new(0),
            compensation_order: RefCell::new(Vec::new()),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            recover_open_transaction(&mut journal, &mut snapshots, &mut appliers).unwrap(),
            TransactionRunOutcome::NeedsRestart
        );
        assert_eq!(snapshots.restored.get(), 0);
        assert_eq!(
            reconstruct_transaction(&recover_journal(&path).unwrap().records)
                .unwrap()
                .ended,
            Some(TxOutcome::NeedsRestart)
        );
    }
}
