//! App-owned lifecycle for one in-process settings preview transaction.
//!
//! This module is intentionally separate from [`crate::settings_io`]: callers
//! provide the runtime, persistence backend, and effect appliers for each
//! operation, so the ECS resource never owns external handles.

#![cfg(feature = "settings-io")]

use std::sync::{
    mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use prism_ecs::resource::Resource;
use prism_ecs::system::{Res, ResMut};
use prism_settings_io::{
    PipelineError, PipelineOutcome, SettingsApplier, SettingsPersistence,
    SettingsTransaction as IoTransaction, TxOutcome,
};

use crate::{
    settings_io::SettingsRuntimeController,
    settings_runtime::{SettingsRuntime, SettingsRuntimeCheckpoint},
};

/// Observable lifecycle of the manager.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SettingsTransactionStatus {
    /// No active transaction; a new one may be started.
    Idle,
    /// A preview is installed and awaits confirmation or expiry.
    PreviewPending {
        /// Stable transaction identity.
        tx_id: String,
        /// Monotonic confirmation deadline.
        deadline: Instant,
    },
    /// An expired Preview is being rolled back off the frame thread.
    PreviewRollingBack {
        /// Stable transaction identity.
        tx_id: String,
    },
    /// A durable or runtime operation could not be proven complete.
    RecoveryRequired {
        /// Transaction that needs recovery.
        tx_id: String,
        /// Redacted reason for the recovery state.
        reason: String,
    },
    /// A transaction committed with a degraded external effect.
    Degraded {
        /// Transaction that committed with degraded effects.
        tx_id: String,
    },
    /// The transaction crossed an irreversible boundary and requires restart.
    NeedsRestart {
        /// Transaction whose effects require restart.
        tx_id: String,
    },
}

/// Lifecycle errors returned by [`SettingsTransactionManager`].
#[derive(Debug)]
pub enum SettingsTransactionManagerError {
    /// A transaction is already active or the manager requires recovery.
    Busy(SettingsTransactionStatus),
    /// The supplied ID does not identify the active transaction.
    IdMismatch {
        /// ID carried by the transaction/journal.
        expected: String,
        /// ID supplied by the caller.
        actual: String,
    },
    /// There is no active preview to confirm, reject, or expire.
    NoPendingPreview,
    /// The requested preview has not reached its deadline.
    NotExpired {
        /// Confirmation deadline.
        deadline: Instant,
        /// Clock value supplied by the caller.
        now: Instant,
    },
    /// Deadline arithmetic overflowed.
    DeadlineOverflow,
    /// The underlying transaction pipeline failed.
    Pipeline(PipelineError),
    /// Pipeline returned an outcome incompatible with the requested operation.
    UnexpectedOutcome(PipelineOutcome),
}

impl std::fmt::Display for SettingsTransactionManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(status) => write!(f, "settings transaction manager is not idle: {status:?}"),
            Self::IdMismatch { expected, actual } => write!(
                f,
                "settings transaction ID mismatch: expected `{expected}`, got `{actual}`"
            ),
            Self::NoPendingPreview => f.write_str("no settings preview is pending"),
            Self::NotExpired { deadline, now } => write!(
                f,
                "settings preview expires at {deadline:?}, current time is {now:?}"
            ),
            Self::DeadlineOverflow => f.write_str("settings preview deadline overflowed"),
            Self::Pipeline(error) => write!(f, "settings transaction pipeline failed: {error}"),
            Self::UnexpectedOutcome(outcome) => {
                write!(f, "unexpected settings transaction outcome: {outcome:?}")
            }
        }
    }
}

impl std::error::Error for SettingsTransactionManagerError {}

struct PendingPreview {
    tx_id: String,
    transaction: IoTransaction,
    checkpoint: SettingsRuntimeCheckpoint,
    deadline: Instant,
}

struct ExpiryWorker {
    events: Mutex<Receiver<ExpiryWorkerEvent>>,
    result: Mutex<Receiver<Result<PipelineOutcome, PipelineError>>>,
}

enum ExpiryWorkerEvent {
    RestoreBefore {
        acknowledge: SyncSender<Result<(), String>>,
    },
}

struct DeferredSnapshotRestore {
    events: Sender<ExpiryWorkerEvent>,
}

impl prism_settings_io::SnapshotController for DeferredSnapshotRestore {
    fn install_snapshot(&mut self, _snapshot: prism_settings::Snapshot) -> Result<(), String> {
        Err("background Preview rollback cannot install snapshots".to_owned())
    }

    fn restore_generation(&mut self, _generation: u64, _hash: u64) -> Result<(), String> {
        Err("background Preview rollback cannot restore generations".to_owned())
    }

    fn restore_before(&mut self, _snapshot: &prism_settings::Snapshot) -> Result<(), String> {
        let (acknowledge, acknowledged) = mpsc::sync_channel(1);
        self.events
            .send(ExpiryWorkerEvent::RestoreBefore { acknowledge })
            .map_err(|_| "App dropped the Preview rollback restore request".to_owned())?;
        acknowledged
            .recv()
            .map_err(|_| "App dropped the Preview rollback acknowledgement".to_owned())?
    }
}

/// ECS resource owning the lifecycle state and at most one pending preview.
///
/// Runtime, persistence, and appliers are deliberately borrowed per operation;
/// only the transaction and complete runtime rollback checkpoint survive frames.
pub struct SettingsTransactionManager {
    status: SettingsTransactionStatus,
    pending: Option<PendingPreview>,
    expiry_error: Option<String>,
    expiry_worker: Option<ExpiryWorker>,
}

impl Resource for SettingsTransactionManager {}

struct SettingsTransactionServicesInner {
    _persistence: Box<dyn SettingsPersistence + Send>,
    appliers: Vec<Box<dyn SettingsApplier + Send>>,
}

/// Thread-safe host services used by the App-owned transaction lifecycle.
///
/// The manager retains only transaction data and rollback state. This resource
/// owns the mutable persistence and side-effect providers so the built-in
/// expiry system can perform a rollback at a frame boundary without requiring
/// every caller to manually pass those providers back into the manager.
pub struct SettingsTransactionServices {
    inner: Arc<Mutex<SettingsTransactionServicesInner>>,
}

impl Resource for SettingsTransactionServices {}

impl SettingsTransactionServices {
    /// Creates lifecycle services from one persistence backend and ordered
    /// effect appliers.
    pub fn new(
        persistence: impl SettingsPersistence + Send + 'static,
        appliers: Vec<Box<dyn SettingsApplier + Send>>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SettingsTransactionServicesInner {
                _persistence: Box::new(persistence),
                appliers,
            })),
        }
    }

    fn spawn_preview_rollback(&self, transaction: IoTransaction) -> Result<ExpiryWorker, String> {
        let services = Arc::clone(&self.inner);
        let (event_tx, event_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        thread::Builder::new()
            .name("prism-settings-preview-rollback".to_owned())
            .spawn(move || {
                let mut inner = services
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let SettingsTransactionServicesInner { appliers, .. } = &mut *inner;
                let mut applier_refs = appliers
                    .iter_mut()
                    .map(|applier| applier.as_mut() as &mut dyn SettingsApplier)
                    .collect::<Vec<_>>();
                let mut snapshots = DeferredSnapshotRestore { events: event_tx };
                let result = transaction.rollback_preview(&mut snapshots, &mut applier_refs);
                let _ = result_tx.send(result);
            })
            .map_err(|error| format!("failed to start Preview rollback worker: {error}"))?;
        Ok(ExpiryWorker {
            events: Mutex::new(event_rx),
            result: Mutex::new(result_rx),
        })
    }
}

impl Default for SettingsTransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SettingsTransactionManager {
    /// Creates an idle lifecycle manager.
    pub const fn new() -> Self {
        Self {
            status: SettingsTransactionStatus::Idle,
            pending: None,
            expiry_error: None,
            expiry_worker: None,
        }
    }

    /// Returns the current externally observable lifecycle status.
    pub fn status(&self) -> &SettingsTransactionStatus {
        &self.status
    }

    /// Returns the most recent automatic expiry failure, if any.
    ///
    /// Missing services do not discard the preview or its rollback checkpoint;
    /// the frame-boundary system checks again on subsequent frames. A pipeline
    /// rollback failure transitions the manager to `RecoveryRequired` and is
    /// not retried automatically. Host diagnostics can inspect this field and
    /// install/fix the required lifecycle services or begin explicit recovery.
    pub fn expiry_error(&self) -> Option<&str> {
        self.expiry_error.as_deref()
    }

    /// Returns whether the retained Preview deadline has elapsed.
    #[must_use]
    pub fn is_expired(&self, now: Instant) -> bool {
        matches!(
            self.status,
            SettingsTransactionStatus::PreviewPending { deadline, .. } if now >= deadline
        )
    }

    fn schedule_expiry(&mut self, services: &SettingsTransactionServices) -> Result<(), String> {
        let Some(pending) = self.pending.as_ref() else {
            return Err("expired Preview has no retained transaction".to_owned());
        };
        if self.expiry_worker.is_some() {
            return Ok(());
        }
        let tx_id = pending.tx_id.clone();
        let worker = services.spawn_preview_rollback(pending.transaction.clone())?;
        self.expiry_worker = Some(worker);
        self.status = SettingsTransactionStatus::PreviewRollingBack { tx_id };
        self.expiry_error = None;
        Ok(())
    }

    /// Polls the off-thread rollback without waiting, restoring the App-owned
    /// runtime only at the worker's explicit pre-finalization handshake.
    fn poll_expiry_worker(&mut self, runtime: &mut SettingsRuntime) {
        let Some(worker) = self.expiry_worker.as_ref() else {
            return;
        };
        match worker
            .events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_recv()
        {
            Ok(ExpiryWorkerEvent::RestoreBefore { acknowledge }) => {
                let restored = self
                    .pending
                    .as_ref()
                    .ok_or_else(|| "Preview rollback checkpoint was lost".to_owned())
                    .and_then(|pending| {
                        runtime
                            .restore_checkpoint(pending.checkpoint.clone())
                            .map_err(|errors| format!("Preview runtime restore failed: {errors:?}"))
                    });
                if let Err(error) = &restored {
                    self.expiry_error = Some(error.clone());
                }
                let _ = acknowledge.send(restored);
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {}
        }

        let completion = match worker
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .try_recv()
        {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(PipelineError::InvalidState(
                "Preview rollback worker terminated without a result".to_owned(),
            ))),
        };
        let Some(completion) = completion else {
            return;
        };
        let tx_id = self
            .pending
            .as_ref()
            .map(|pending| pending.tx_id.clone())
            .unwrap_or_else(|| "unknown".to_owned());
        self.expiry_worker = None;
        match completion {
            Ok(PipelineOutcome::RolledBack) => {
                self.pending = None;
                self.status = SettingsTransactionStatus::Idle;
                self.expiry_error = None;
            }
            Ok(outcome) => {
                let reason = format!("automatic Preview rollback returned {outcome:?}");
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id,
                    reason: reason.clone(),
                };
                self.expiry_error = Some(reason);
            }
            Err(error) => {
                let reason = error.to_string();
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id,
                    reason: reason.clone(),
                };
                self.expiry_error = Some(reason);
            }
        }
    }

    /// Executes a transaction. A preview awaiting confirmation is retained
    /// together with its runtime checkpoint and monotonic deadline.
    pub fn begin(
        &mut self,
        tx_id: impl Into<String>,
        transaction: IoTransaction,
        timeout: Duration,
        runtime: &mut SettingsRuntime,
        persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn SettingsApplier],
    ) -> Result<PipelineOutcome, SettingsTransactionManagerError> {
        if self.pending.is_some() || !matches!(self.status, SettingsTransactionStatus::Idle) {
            return Err(SettingsTransactionManagerError::Busy(self.status.clone()));
        }
        let tx_id = tx_id.into();
        if tx_id != transaction.tx_id() {
            return Err(SettingsTransactionManagerError::IdMismatch {
                expected: transaction.tx_id().to_owned(),
                actual: tx_id,
            });
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(SettingsTransactionManagerError::DeadlineOverflow)?;
        let fallback_checkpoint = runtime.checkpoint();
        let mut controller = SettingsRuntimeController::new(runtime);
        let result = transaction.execute(&mut controller, persistence, appliers);
        let checkpoint = controller.take_checkpoint().unwrap_or(fallback_checkpoint);
        match result {
            Ok(PipelineOutcome::PendingConfirmation) => {
                self.expiry_error = None;
                self.status = SettingsTransactionStatus::PreviewPending {
                    tx_id: tx_id.clone(),
                    deadline,
                };
                self.pending = Some(PendingPreview {
                    tx_id,
                    transaction,
                    checkpoint,
                    deadline,
                });
                Ok(PipelineOutcome::PendingConfirmation)
            }
            Ok(outcome @ (PipelineOutcome::RecoveryRequired | PipelineOutcome::NeedsRestart)) => {
                self.status = if outcome == PipelineOutcome::NeedsRestart {
                    SettingsTransactionStatus::NeedsRestart { tx_id }
                } else {
                    SettingsTransactionStatus::RecoveryRequired {
                        tx_id,
                        reason: format!("pipeline outcome: {outcome:?}"),
                    }
                };
                Err(SettingsTransactionManagerError::UnexpectedOutcome(outcome))
            }
            Ok(outcome @ PipelineOutcome::Degraded) => {
                self.status = SettingsTransactionStatus::Degraded { tx_id };
                Ok(outcome)
            }
            Ok(outcome) => Ok(outcome),
            Err(error) => {
                if transaction.terminal_outcome().ok() == Some(Some(TxOutcome::RolledBack)) {
                    self.status = SettingsTransactionStatus::Idle;
                    self.expiry_error = None;
                } else {
                    self.status = SettingsTransactionStatus::RecoveryRequired {
                        tx_id,
                        reason: error.to_string(),
                    };
                }
                Err(SettingsTransactionManagerError::Pipeline(error))
            }
        }
    }

    /// Confirms the active preview and clears its retained transaction only
    /// after durable finalization succeeds.
    pub fn confirm(
        &mut self,
        tx_id: &str,
        runtime: &mut SettingsRuntime,
        persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn SettingsApplier],
    ) -> Result<PipelineOutcome, SettingsTransactionManagerError> {
        self.validate_id(tx_id)?;
        let pending = self.pending.as_ref().expect("validated pending preview");
        let mut controller =
            SettingsRuntimeController::with_checkpoint(runtime, pending.checkpoint.clone());
        let result = pending
            .transaction
            .confirm_preview(&mut controller, persistence, appliers);
        self.finish_preview_result(tx_id, result)
    }

    /// Rejects the active preview and restores runtime state. Any ambiguous
    /// rollback outcome is retained as `RecoveryRequired`.
    pub fn reject(
        &mut self,
        tx_id: &str,
        runtime: &mut SettingsRuntime,
        _persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn SettingsApplier],
    ) -> Result<PipelineOutcome, SettingsTransactionManagerError> {
        self.rollback(tx_id, runtime, _persistence, appliers)
    }

    /// Rolls back the preview if `now` has reached its deadline.
    pub fn expire(
        &mut self,
        now: Instant,
        runtime: &mut SettingsRuntime,
        _persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn SettingsApplier],
    ) -> Result<Option<PipelineOutcome>, SettingsTransactionManagerError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(SettingsTransactionManagerError::NoPendingPreview)?;
        if now < pending.deadline {
            return Err(SettingsTransactionManagerError::NotExpired {
                deadline: pending.deadline,
                now,
            });
        }
        let id = pending.tx_id.clone();
        self.rollback(&id, runtime, _persistence, appliers)
            .map(Some)
    }

    fn validate_id(&self, tx_id: &str) -> Result<(), SettingsTransactionManagerError> {
        if matches!(
            self.status,
            SettingsTransactionStatus::PreviewRollingBack { .. }
        ) {
            return Err(SettingsTransactionManagerError::Busy(self.status.clone()));
        }
        let Some(pending) = self.pending.as_ref() else {
            return Err(SettingsTransactionManagerError::NoPendingPreview);
        };
        if pending.tx_id != tx_id {
            return Err(SettingsTransactionManagerError::IdMismatch {
                expected: pending.tx_id.clone(),
                actual: tx_id.to_owned(),
            });
        }
        Ok(())
    }

    fn rollback(
        &mut self,
        tx_id: &str,
        runtime: &mut SettingsRuntime,
        _persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn SettingsApplier],
    ) -> Result<PipelineOutcome, SettingsTransactionManagerError> {
        self.validate_id(tx_id)?;
        let pending = self.pending.as_ref().expect("validated pending preview");
        let mut controller =
            SettingsRuntimeController::with_checkpoint(runtime, pending.checkpoint.clone());
        let result = pending
            .transaction
            .rollback_preview(&mut controller, appliers);
        match result {
            Ok(PipelineOutcome::RolledBack) => {
                self.pending = None;
                self.status = SettingsTransactionStatus::Idle;
                self.expiry_error = None;
                Ok(PipelineOutcome::RolledBack)
            }
            Ok(
                outcome @ (PipelineOutcome::RecoveryRequired
                | PipelineOutcome::NeedsRestart
                | PipelineOutcome::Degraded),
            ) => {
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id: tx_id.to_owned(),
                    reason: format!("rollback outcome: {outcome:?}"),
                };
                Err(SettingsTransactionManagerError::UnexpectedOutcome(outcome))
            }
            Ok(outcome) => {
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id: tx_id.to_owned(),
                    reason: format!("unexpected rollback outcome: {outcome:?}"),
                };
                Err(SettingsTransactionManagerError::UnexpectedOutcome(outcome))
            }
            Err(error) => {
                if self
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.tx_id == tx_id)
                    && self
                        .pending
                        .as_ref()
                        .and_then(|pending| pending.transaction.terminal_outcome().ok().flatten())
                        == Some(TxOutcome::RolledBack)
                {
                    self.pending = None;
                    self.status = SettingsTransactionStatus::Idle;
                    self.expiry_error = None;
                    return Err(SettingsTransactionManagerError::Pipeline(error));
                }
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id: tx_id.to_owned(),
                    reason: error.to_string(),
                };
                Err(SettingsTransactionManagerError::Pipeline(error))
            }
        }
    }

    fn finish_preview_result(
        &mut self,
        tx_id: &str,
        result: Result<PipelineOutcome, PipelineError>,
    ) -> Result<PipelineOutcome, SettingsTransactionManagerError> {
        match result {
            Ok(PipelineOutcome::Committed) => {
                self.pending = None;
                self.status = SettingsTransactionStatus::Idle;
                self.expiry_error = None;
                Ok(PipelineOutcome::Committed)
            }
            Ok(
                outcome @ (PipelineOutcome::RecoveryRequired
                | PipelineOutcome::NeedsRestart
                | PipelineOutcome::Degraded),
            ) => {
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id: tx_id.to_owned(),
                    reason: format!("confirmation outcome: {outcome:?}"),
                };
                Err(SettingsTransactionManagerError::UnexpectedOutcome(outcome))
            }
            Ok(outcome) => {
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id: tx_id.to_owned(),
                    reason: format!("unexpected confirmation outcome: {outcome:?}"),
                };
                Err(SettingsTransactionManagerError::UnexpectedOutcome(outcome))
            }
            Err(error) => {
                if self
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.tx_id == tx_id)
                    && self
                        .pending
                        .as_ref()
                        .and_then(|pending| pending.transaction.terminal_outcome().ok().flatten())
                        == Some(TxOutcome::RolledBack)
                {
                    self.pending = None;
                    self.status = SettingsTransactionStatus::Idle;
                    self.expiry_error = None;
                    return Err(SettingsTransactionManagerError::Pipeline(error));
                }
                self.status = SettingsTransactionStatus::RecoveryRequired {
                    tx_id: tx_id.to_owned(),
                    reason: error.to_string(),
                };
                Err(SettingsTransactionManagerError::Pipeline(error))
            }
        }
    }
}

/// Automatically rolls back an expired Preview at the App frame boundary.
///
/// The system is installed by [`App::new`](crate::App::new) under the
/// `settings-io` feature. It is inert until a caller installs
/// [`SettingsTransactionServices`] in the main World.
pub fn expire_settings_transaction_system(
    mut manager: ResMut<SettingsTransactionManager>,
    mut runtime: ResMut<SettingsRuntime>,
    services: Option<Res<SettingsTransactionServices>>,
) {
    let now = Instant::now();
    manager.poll_expiry_worker(&mut runtime);
    if !manager.is_expired(now) {
        return;
    }
    let Some(services) = services else {
        manager.expiry_error =
            Some("expired settings preview is waiting for SettingsTransactionServices".to_owned());
        return;
    };
    if let Err(error) = manager.schedule_expiry(&services) {
        manager.expiry_error = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::App;
    use prism_settings::{Domain, LayerKind, RegistryBuilder, SettingDef, TrustLevel, Value};
    use prism_settings_io::{
        load_with_backup, ApplyOp, ApplyPlan, CandidateBuilder, CandidateSnapshot,
        CompensateOutcome, EffectState, FileSettingsPersistence, OpAction, OpOutcome,
        Reversibility, SettingsApplier, SettingsDocument, SettingsTransactionSpec, SourceSpec,
        TxMode,
    };
    use std::{
        fs,
        path::PathBuf,
        sync::{
            atomic::{AtomicU64, AtomicUsize, Ordering},
            mpsc,
        },
        thread,
    };

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "prism-settings-transaction-manager-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct NoopPersistence;

    #[test]
    fn frame_expiry_retries_after_transaction_services_lock_contention() {
        let registry = registry();
        let directory = TempDir::new();
        let mut app = App::new();
        app.install_settings_runtime(registry.clone(), Domain::Runtime)
            .unwrap();

        let mut persistence = NoopPersistence;
        let mut begin_appliers = appliers();
        let mut manager = app
            .world_mut()
            .remove_resource::<SettingsTransactionManager>()
            .unwrap();
        let mut runtime = app
            .world_mut()
            .remove_resource::<SettingsRuntime>()
            .unwrap();
        manager
            .begin(
                "lock-contention",
                transaction(&registry, &directory, "lock-contention", 0.75),
                Duration::ZERO,
                &mut runtime,
                &mut persistence,
                &mut begin_appliers,
            )
            .unwrap();
        app.insert_resource(manager);
        app.insert_resource(runtime);

        let services = SettingsTransactionServices::new(NoopPersistence, Vec::new());
        let shared_lock = Arc::clone(&services.inner);
        app.insert_resource(services);
        let (locked_tx, locked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = thread::spawn(move || {
            let _guard = shared_lock.lock().unwrap();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        locked_rx.recv().unwrap();

        app.update();
        let manager = app.world().resource::<SettingsTransactionManager>();
        assert!(matches!(
            manager.status(),
            SettingsTransactionStatus::PreviewRollingBack { .. }
        ));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
        run_until(&mut app, |manager| {
            matches!(manager.status(), SettingsTransactionStatus::Idle)
        });
    }

    impl SettingsPersistence for NoopPersistence {
        fn persist(
            &mut self,
            _document: &SettingsDocument,
            generation: u64,
        ) -> Result<u64, String> {
            Ok(generation)
        }

        fn rollback(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    struct PersistFailure;

    impl SettingsPersistence for PersistFailure {
        fn persist(
            &mut self,
            _document: &SettingsDocument,
            _generation: u64,
        ) -> Result<u64, String> {
            Err("injected persistence failure".to_owned())
        }

        fn rollback(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    struct FailOncePersistence {
        fail_next: bool,
        attempts: usize,
        rollbacks: usize,
        finalizations: usize,
    }

    impl SettingsPersistence for FailOncePersistence {
        fn persist(
            &mut self,
            _document: &SettingsDocument,
            generation: u64,
        ) -> Result<u64, String> {
            self.attempts += 1;
            if std::mem::take(&mut self.fail_next) {
                Err("injected one-shot persistence failure".to_owned())
            } else {
                Ok(generation)
            }
        }

        fn rollback(&mut self) -> Result<(), String> {
            self.rollbacks += 1;
            Ok(())
        }

        fn finalize(&mut self) -> Result<(), String> {
            self.finalizations += 1;
            Ok(())
        }
    }

    struct CountingApplier {
        applied: Arc<AtomicUsize>,
        compensated: Arc<AtomicUsize>,
    }

    impl SettingsApplier for CountingApplier {
        fn group(&self) -> &str {
            "Display"
        }

        fn order(&self) -> i32 {
            0
        }

        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            self.applied.fetch_add(1, Ordering::SeqCst);
            OpOutcome::Applied {
                effect_handle: Some("display-handle".to_owned()),
                result_digest: 1,
            }
        }

        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }

        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            self.compensated.fetch_add(1, Ordering::SeqCst);
            CompensateOutcome::Done { verified: true }
        }
    }

    fn registry() -> prism_settings::SettingsRegistry {
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(SettingDef::new("render.scale", "render", Value::Float(1.0)));
        builder.finish().unwrap()
    }

    fn candidate(registry: &prism_settings::SettingsRegistry, value: f64) -> CandidateSnapshot {
        let source = SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file");
        let mut document = SettingsDocument::new(1);
        document.insert("render.scale", Value::Float(value));
        CandidateBuilder::new(registry, Domain::Runtime, source)
            .build(document, &[], 1, false)
            .unwrap()
    }

    fn transaction(
        registry: &prism_settings::SettingsRegistry,
        directory: &TempDir,
        id: &str,
        value: f64,
    ) -> IoTransaction {
        transaction_with_plan(
            registry,
            directory,
            id,
            value,
            ApplyPlan::new(Vec::new()).unwrap(),
        )
    }

    fn transaction_with_plan(
        registry: &prism_settings::SettingsRegistry,
        directory: &TempDir,
        id: &str,
        value: f64,
        plan: ApplyPlan,
    ) -> IoTransaction {
        let before = SettingsRuntime::new(registry.clone(), Domain::Runtime).unwrap();
        IoTransaction::new(SettingsTransactionSpec {
            tx_id: id.to_owned(),
            journal_path: directory.0.join(format!("{id}.journal")),
            mode: TxMode::Preview,
            before_hash: 0,
            before_snapshot: before.snapshot().clone(),
            candidate: candidate(registry, value),
            plan,
            snapshot_gen: 0,
            last_confirmed: None,
        })
    }

    fn runtime(registry: &prism_settings::SettingsRegistry) -> SettingsRuntime {
        SettingsRuntime::new(registry.clone(), Domain::Runtime).unwrap()
    }

    fn appliers() -> Vec<&'static mut dyn SettingsApplier> {
        Vec::new()
    }

    fn run_until(app: &mut App, predicate: impl Fn(&SettingsTransactionManager) -> bool) {
        for _ in 0..10_000 {
            app.update();
            if predicate(app.world().resource::<SettingsTransactionManager>()) {
                return;
            }
            thread::yield_now();
        }
        panic!("settings transaction did not reach the expected state");
    }

    struct CompensationFailureApplier {
        compensations: Arc<AtomicUsize>,
    }

    struct BlockingCompensationApplier {
        entered: Sender<()>,
        release: Arc<Mutex<Receiver<()>>>,
    }

    impl SettingsApplier for BlockingCompensationApplier {
        fn group(&self) -> &str {
            "Display"
        }
        fn order(&self) -> i32 {
            0
        }
        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            OpOutcome::Applied {
                effect_handle: Some("display-handle".to_owned()),
                result_digest: 1,
            }
        }
        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }
        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            let _ = self.entered.send(());
            self.release.lock().unwrap().recv().unwrap();
            CompensateOutcome::Done { verified: true }
        }
    }

    struct ApplyFailureApplier;

    impl SettingsApplier for ApplyFailureApplier {
        fn group(&self) -> &str {
            "Display"
        }

        fn order(&self) -> i32 {
            0
        }

        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            OpOutcome::Failed {
                error: "injected apply failure".to_owned(),
                effect_handle: None,
            }
        }

        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }

        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            CompensateOutcome::Done { verified: true }
        }
    }

    impl SettingsApplier for CompensationFailureApplier {
        fn group(&self) -> &str {
            "Display"
        }

        fn order(&self) -> i32 {
            0
        }

        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            OpOutcome::Applied {
                effect_handle: Some("display-handle".to_owned()),
                result_digest: 1,
            }
        }

        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }

        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            self.compensations.fetch_add(1, Ordering::SeqCst);
            CompensateOutcome::Failed {
                error: "injected compensation failure".to_owned(),
            }
        }
    }

    fn reversible_display_plan() -> ApplyPlan {
        ApplyPlan::new(vec![ApplyOp {
            index: 0,
            id: "display-mode".to_owned(),
            group: "Display".to_owned(),
            key: "render.scale".into(),
            action: OpAction::Set,
            reversibility: Reversibility::Reversible,
            idempotent: true,
            input_digest: 1,
            effect_handle: None,
        }])
        .unwrap()
    }

    #[test]
    fn rejects_a_second_pending_transaction() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let mut persistence = NoopPersistence;
        let mut first_appliers = appliers();
        assert_eq!(
            manager
                .begin(
                    "first",
                    transaction(&registry, &directory, "first", 0.75),
                    Duration::from_secs(30),
                    &mut runtime,
                    &mut persistence,
                    &mut first_appliers,
                )
                .unwrap(),
            PipelineOutcome::PendingConfirmation
        );
        let mut second_appliers = appliers();
        let error = manager
            .begin(
                "second",
                transaction(&registry, &directory, "second", 0.5),
                Duration::from_secs(30),
                &mut runtime,
                &mut persistence,
                &mut second_appliers,
            )
            .unwrap_err();
        assert!(matches!(error, SettingsTransactionManagerError::Busy(_)));
    }

    #[test]
    fn clean_direct_rollback_error_does_not_lock_manager_in_recovery() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let before = runtime.snapshot().clone();
        let before_snapshot = before.clone();
        let transaction = IoTransaction::new(SettingsTransactionSpec {
            tx_id: "direct-persist-failure".to_owned(),
            journal_path: directory.0.join("direct-persist-failure.journal"),
            mode: TxMode::Direct,
            before_hash: 0,
            before_snapshot,
            candidate: candidate(&registry, 0.75),
            plan: ApplyPlan::new(Vec::new()).unwrap(),
            snapshot_gen: 0,
            last_confirmed: None,
        });
        let mut persistence = PersistFailure;
        let mut appliers = appliers();
        let error = manager
            .begin(
                "direct-persist-failure",
                transaction,
                Duration::from_secs(30),
                &mut runtime,
                &mut persistence,
                &mut appliers,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            SettingsTransactionManagerError::Pipeline(_)
        ));
        assert_eq!(manager.status(), &SettingsTransactionStatus::Idle);
        assert_eq!(runtime.snapshot(), &before);
    }

    #[test]
    fn manager_direct_apply_failure_restores_file_runtime_and_journal() {
        let registry = registry();
        let directory = TempDir::new();
        let settings_path = directory.0.join("settings.toml");
        let backup_path = directory.0.join("settings.bak");
        let tx_id = "manager-direct-rollback";
        let journal_path = directory.0.join(format!("{tx_id}.journal"));
        let mut old_document = SettingsDocument::new(1);
        old_document.insert("render.scale", Value::Float(0.5));
        prism_settings_io::write_atomic(&settings_path, &old_document, Some(&backup_path)).unwrap();

        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let before = runtime.checkpoint();
        let mut persistence = FileSettingsPersistence::new(&settings_path, Some(&backup_path), 2);
        let mut applier = ApplyFailureApplier;
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        let candidate = transaction_with_plan(
            &registry,
            &directory,
            tx_id,
            0.75,
            reversible_display_plan(),
        )
        .candidate()
        .clone();
        let tx = IoTransaction::new(SettingsTransactionSpec {
            tx_id: tx_id.to_owned(),
            journal_path: journal_path.clone(),
            mode: TxMode::Direct,
            before_hash: 0,
            before_snapshot: runtime.snapshot().clone(),
            candidate,
            plan: reversible_display_plan(),
            snapshot_gen: 0,
            last_confirmed: None,
        });

        assert_eq!(
            manager
                .begin(
                    tx_id,
                    tx,
                    Duration::from_secs(30),
                    &mut runtime,
                    &mut persistence,
                    &mut appliers,
                )
                .unwrap(),
            PipelineOutcome::RolledBack
        );
        assert_eq!(manager.status(), &SettingsTransactionStatus::Idle);
        assert_eq!(runtime.checkpoint(), before);
        let restored = load_with_backup(&settings_path, Some(&backup_path), Some(&registry))
            .unwrap()
            .document;
        assert_eq!(
            restored.candidates().get(&"render.scale".into()),
            Some(&Value::Float(0.5))
        );
        let journal = prism_settings_io::inspect_journal(&journal_path).unwrap();
        let recovered = prism_settings_io::reconstruct_transaction(&journal.records).unwrap();
        assert_eq!(recovered.ended, Some(TxOutcome::RolledBack));
    }

    #[test]
    fn timeout_rolls_back_and_returns_to_idle() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let before = runtime.snapshot().clone();
        let mut persistence = NoopPersistence;
        let mut begin_appliers = appliers();
        manager
            .begin(
                "expire",
                transaction(&registry, &directory, "expire", 0.75),
                Duration::ZERO,
                &mut runtime,
                &mut persistence,
                &mut begin_appliers,
            )
            .unwrap();
        let mut rollback_appliers = appliers();
        assert_eq!(
            manager
                .expire(
                    Instant::now(),
                    &mut runtime,
                    &mut persistence,
                    &mut rollback_appliers,
                )
                .unwrap(),
            Some(PipelineOutcome::RolledBack)
        );
        assert_eq!(runtime.snapshot(), &before);
        assert_eq!(manager.status(), &SettingsTransactionStatus::Idle);
    }

    #[test]
    fn rejects_an_id_mismatch_without_touching_the_pending_preview() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let mut persistence = NoopPersistence;
        let mut begin_appliers = appliers();
        manager
            .begin(
                "actual",
                transaction(&registry, &directory, "actual", 0.75),
                Duration::from_secs(30),
                &mut runtime,
                &mut persistence,
                &mut begin_appliers,
            )
            .unwrap();
        let mut confirm_appliers = appliers();
        let error = manager
            .confirm(
                "wrong",
                &mut runtime,
                &mut persistence,
                &mut confirm_appliers,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            SettingsTransactionManagerError::IdMismatch { .. }
        ));
        assert!(matches!(
            manager.status(),
            SettingsTransactionStatus::PreviewPending { .. }
        ));
    }

    #[test]
    fn successful_confirm_clears_the_pending_state() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let mut persistence = NoopPersistence;
        let mut begin_appliers = appliers();
        manager
            .begin(
                "confirm",
                transaction(&registry, &directory, "confirm", 0.75),
                Duration::from_secs(30),
                &mut runtime,
                &mut persistence,
                &mut begin_appliers,
            )
            .unwrap();
        let mut confirm_appliers = appliers();
        assert_eq!(
            manager
                .confirm(
                    "confirm",
                    &mut runtime,
                    &mut persistence,
                    &mut confirm_appliers
                )
                .unwrap(),
            PipelineOutcome::Committed
        );
        assert_eq!(manager.status(), &SettingsTransactionStatus::Idle);
    }

    #[test]
    fn failed_preview_confirmation_that_rolls_back_clears_manager_state() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let before = runtime.snapshot().clone();
        let mut persistence = FailOncePersistence {
            fail_next: true,
            attempts: 0,
            rollbacks: 0,
            finalizations: 0,
        };
        let applied = Arc::new(AtomicUsize::new(0));
        let compensated = Arc::new(AtomicUsize::new(0));
        let mut applier = CountingApplier {
            applied: applied.clone(),
            compensated: compensated.clone(),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        let tx = IoTransaction::new(SettingsTransactionSpec {
            tx_id: "confirm-persist-failure".to_owned(),
            journal_path: directory.0.join("confirm-persist-failure.journal"),
            mode: TxMode::Preview,
            before_hash: 0,
            before_snapshot: before.clone(),
            candidate: candidate(&registry, 0.75),
            plan: reversible_display_plan(),
            snapshot_gen: 0,
            last_confirmed: None,
        });

        assert_eq!(
            manager
                .begin(
                    "confirm-persist-failure",
                    tx,
                    Duration::from_secs(30),
                    &mut runtime,
                    &mut persistence,
                    &mut appliers,
                )
                .unwrap(),
            PipelineOutcome::PendingConfirmation
        );
        assert_eq!(applied.load(Ordering::SeqCst), 1);
        assert_ne!(runtime.snapshot(), &before);

        assert!(matches!(
            manager.confirm(
                "confirm-persist-failure",
                &mut runtime,
                &mut persistence,
                &mut appliers,
            ),
            Err(SettingsTransactionManagerError::Pipeline(_))
        ));
        assert_eq!(manager.status(), &SettingsTransactionStatus::Idle);
        assert_eq!(runtime.snapshot(), &before);
        assert_eq!(applied.load(Ordering::SeqCst), 1);
        assert_eq!(compensated.load(Ordering::SeqCst), 1);
        assert_eq!(persistence.attempts, 1);
        assert_eq!(persistence.rollbacks, 1);
        assert_eq!(persistence.finalizations, 1);

        assert!(matches!(
            manager.confirm(
                "confirm-persist-failure",
                &mut runtime,
                &mut persistence,
                &mut appliers,
            ),
            Err(SettingsTransactionManagerError::NoPendingPreview)
        ));
        assert_eq!(applied.load(Ordering::SeqCst), 1);
        assert_eq!(compensated.load(Ordering::SeqCst), 1);
        assert_eq!(persistence.attempts, 1);
    }

    #[test]
    fn rejects_a_transaction_id_that_differs_from_the_journal_id() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let mut persistence = NoopPersistence;
        let mut appliers = appliers();

        let error = manager
            .begin(
                "caller-id",
                transaction(&registry, &directory, "journal-id", 0.75),
                Duration::from_secs(30),
                &mut runtime,
                &mut persistence,
                &mut appliers,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            SettingsTransactionManagerError::IdMismatch { .. }
        ));
        assert_eq!(manager.status(), &SettingsTransactionStatus::Idle);
    }

    #[test]
    fn app_installs_the_transaction_manager_resource() {
        let app = App::new();
        assert_eq!(
            app.world()
                .resource::<SettingsTransactionManager>()
                .status(),
            &SettingsTransactionStatus::Idle
        );
    }

    #[test]
    fn frame_boundary_system_expires_a_pending_preview() {
        let registry = registry();
        let directory = TempDir::new();
        let mut app = App::new();
        app.install_settings_runtime(registry.clone(), Domain::Runtime)
            .unwrap();
        let mut persistence = NoopPersistence;
        let mut appliers = appliers();
        let mut manager = app
            .world_mut()
            .remove_resource::<SettingsTransactionManager>()
            .unwrap();
        let mut runtime = app
            .world_mut()
            .remove_resource::<SettingsRuntime>()
            .unwrap();
        manager
            .begin(
                "auto-expire",
                transaction(&registry, &directory, "auto-expire", 0.75),
                Duration::ZERO,
                &mut runtime,
                &mut persistence,
                &mut appliers,
            )
            .unwrap();
        app.insert_resource(manager);
        app.insert_resource(runtime);

        app.insert_resource(SettingsTransactionServices::new(
            NoopPersistence,
            Vec::new(),
        ));
        run_until(&mut app, |manager| {
            matches!(manager.status(), SettingsTransactionStatus::Idle)
        });
    }

    #[test]
    fn expired_preview_reports_missing_services_and_retries_after_install() {
        let registry = registry();
        let directory = TempDir::new();
        let mut app = App::new();
        app.install_settings_runtime(registry.clone(), Domain::Runtime)
            .unwrap();
        let mut persistence = NoopPersistence;
        let mut begin_appliers = appliers();
        let mut manager = app
            .world_mut()
            .remove_resource::<SettingsTransactionManager>()
            .unwrap();
        let mut runtime = app
            .world_mut()
            .remove_resource::<SettingsRuntime>()
            .unwrap();
        manager
            .begin(
                "delayed-services",
                transaction(&registry, &directory, "delayed-services", 0.75),
                Duration::ZERO,
                &mut runtime,
                &mut persistence,
                &mut begin_appliers,
            )
            .unwrap();
        app.insert_resource(manager);
        app.insert_resource(runtime);

        app.update();
        let manager = app.world().resource::<SettingsTransactionManager>();
        assert!(matches!(
            manager.status(),
            SettingsTransactionStatus::PreviewPending { .. }
        ));
        assert!(manager
            .expiry_error()
            .is_some_and(|error| error.contains("SettingsTransactionServices")));

        app.insert_resource(SettingsTransactionServices::new(
            NoopPersistence,
            Vec::new(),
        ));
        app.update();
        let manager = app.world().resource::<SettingsTransactionManager>();
        assert!(matches!(
            manager.status(),
            SettingsTransactionStatus::PreviewRollingBack { .. }
        ));
        run_until(&mut app, |manager| {
            matches!(manager.status(), SettingsTransactionStatus::Idle)
        });
        assert_eq!(
            app.world()
                .resource::<SettingsTransactionManager>()
                .expiry_error(),
            None
        );
    }

    #[test]
    fn failed_preview_expiry_retains_recovery_state_without_automatic_retry() {
        let registry = registry();
        let directory = TempDir::new();
        let mut manager = SettingsTransactionManager::new();
        let mut runtime = runtime(&registry);
        let mut persistence = NoopPersistence;
        let compensations = Arc::new(AtomicUsize::new(0));
        let mut applier = CompensationFailureApplier {
            compensations: compensations.clone(),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert_eq!(
            manager
                .begin(
                    "failed-expiry",
                    transaction_with_plan(
                        &registry,
                        &directory,
                        "failed-expiry",
                        0.75,
                        reversible_display_plan(),
                    ),
                    Duration::ZERO,
                    &mut runtime,
                    &mut persistence,
                    &mut appliers,
                )
                .unwrap(),
            PipelineOutcome::PendingConfirmation
        );

        let error = manager
            .expire(
                Instant::now(),
                &mut runtime,
                &mut persistence,
                &mut appliers,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            SettingsTransactionManagerError::UnexpectedOutcome(_)
        ));
        assert!(matches!(
            manager.status(),
            SettingsTransactionStatus::RecoveryRequired { tx_id, .. } if tx_id == "failed-expiry"
        ));
        assert!(!manager.is_expired(Instant::now()));
        assert_eq!(compensations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn app_expiry_persists_recovery_required_after_compensation_failure() {
        let registry = registry();
        let directory = TempDir::new();
        let tx_id = "app-expiry-recovery";
        let journal_path = directory.0.join(format!("{tx_id}.journal"));
        let compensations = Arc::new(AtomicUsize::new(0));
        let mut app = App::new();
        app.install_settings_runtime(registry.clone(), Domain::Runtime)
            .unwrap();

        let mut manager = app
            .world_mut()
            .remove_resource::<SettingsTransactionManager>()
            .unwrap();
        let mut runtime = app
            .world_mut()
            .remove_resource::<SettingsRuntime>()
            .unwrap();
        let mut persistence = NoopPersistence;
        let mut begin_applier = CompensationFailureApplier {
            compensations: compensations.clone(),
        };
        let mut begin_appliers: Vec<&mut dyn SettingsApplier> = vec![&mut begin_applier];
        assert_eq!(
            manager
                .begin(
                    tx_id,
                    transaction_with_plan(
                        &registry,
                        &directory,
                        tx_id,
                        0.75,
                        reversible_display_plan(),
                    ),
                    Duration::ZERO,
                    &mut runtime,
                    &mut persistence,
                    &mut begin_appliers,
                )
                .unwrap(),
            PipelineOutcome::PendingConfirmation
        );
        app.insert_resource(manager);
        app.insert_resource(runtime);
        app.insert_resource(SettingsTransactionServices::new(
            NoopPersistence,
            vec![Box::new(CompensationFailureApplier {
                compensations: compensations.clone(),
            })],
        ));

        run_until(&mut app, |manager| {
            matches!(
                manager.status(),
                SettingsTransactionStatus::RecoveryRequired { .. }
            )
        });
        assert!(matches!(
            app.world()
                .resource::<SettingsTransactionManager>()
                .status(),
            SettingsTransactionStatus::RecoveryRequired { tx_id: id, .. } if id == tx_id
        ));
        assert_eq!(compensations.load(Ordering::SeqCst), 1);

        let journal = prism_settings_io::inspect_journal(&journal_path).unwrap();
        let recovered = prism_settings_io::reconstruct_transaction(&journal.records).unwrap();
        assert_eq!(recovered.ended, Some(TxOutcome::RecoveryRequired));
        assert!(journal.records.iter().any(|record| matches!(
            record.event,
            prism_settings_io::JournalEvent::CompensateFailed { op_index: 0, .. }
        )));

        app.update();
        assert_eq!(compensations.load(Ordering::SeqCst), 1);
        assert!(matches!(
            app.world()
                .resource::<SettingsTransactionManager>()
                .status(),
            SettingsTransactionStatus::RecoveryRequired { tx_id: id, .. } if id == tx_id
        ));
    }

    #[test]
    fn blocked_expiry_compensation_does_not_block_app_frame() {
        let registry = registry();
        let directory = TempDir::new();
        let tx_id = "blocked-app-expiry";
        let journal_path = directory.0.join(format!("{tx_id}.journal"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let mut app = App::new();
        app.install_settings_runtime(registry.clone(), Domain::Runtime)
            .unwrap();
        let mut manager = app
            .world_mut()
            .remove_resource::<SettingsTransactionManager>()
            .unwrap();
        let mut runtime = app
            .world_mut()
            .remove_resource::<SettingsRuntime>()
            .unwrap();
        let mut persistence = NoopPersistence;
        let mut begin_applier = CountingApplier {
            applied: Arc::new(AtomicUsize::new(0)),
            compensated: Arc::new(AtomicUsize::new(0)),
        };
        let mut begin_appliers: Vec<&mut dyn SettingsApplier> = vec![&mut begin_applier];
        manager
            .begin(
                tx_id,
                transaction_with_plan(
                    &registry,
                    &directory,
                    tx_id,
                    0.75,
                    reversible_display_plan(),
                ),
                Duration::ZERO,
                &mut runtime,
                &mut persistence,
                &mut begin_appliers,
            )
            .unwrap();
        app.insert_resource(manager);
        app.insert_resource(runtime);
        app.insert_resource(SettingsTransactionServices::new(
            NoopPersistence,
            vec![Box::new(BlockingCompensationApplier {
                entered: entered_tx,
                release: Arc::clone(&release_rx),
            })],
        ));

        // This must return even though the rollback worker will block in compensate.
        app.update();
        entered_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("worker did not enter compensation");
        assert!(matches!(
            app.world()
                .resource::<SettingsTransactionManager>()
                .status(),
            SettingsTransactionStatus::PreviewRollingBack { .. }
        ));
        release_tx.send(()).unwrap();
        run_until(&mut app, |manager| {
            matches!(manager.status(), SettingsTransactionStatus::Idle)
        });

        let journal = prism_settings_io::inspect_journal(&journal_path).unwrap();
        let recovered = prism_settings_io::reconstruct_transaction(&journal.records).unwrap();
        assert_eq!(recovered.ended, Some(TxOutcome::RolledBack));
    }
}
