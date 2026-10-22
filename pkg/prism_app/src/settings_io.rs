//! Optional adapter between [`SettingsRuntime`] and the canonical settings
//! file transaction crate.
//!
//! Enabled only by `settings-io`; default and `no_std + alloc` builds do not
//! include filesystem I/O.

use prism_settings::Snapshot;
use prism_settings_io::{CandidateSnapshot, SnapshotController};
#[cfg(test)]
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::settings_runtime::{PreparedSettingsUpdate, SettingsRuntime, SettingsRuntimeCheckpoint};

/// Transaction snapshot adapter that commits source sets and effective
/// snapshots atomically.
///
/// For a preview, retain the transaction and checkpoint across frames. After
/// `PendingConfirmation`, take the checkpoint and construct a new controller
/// with [`with_checkpoint`](Self::with_checkpoint) for a possible rollback.
pub struct SettingsRuntimeController<'a> {
    runtime: &'a mut SettingsRuntime,
    prepared: Option<PreparedSettingsUpdate>,
    checkpoint: Option<SettingsRuntimeCheckpoint>,
}

impl<'a> SettingsRuntimeController<'a> {
    /// Creates an adapter for one in-process transaction.
    pub fn new(runtime: &'a mut SettingsRuntime) -> Self {
        Self {
            runtime,
            prepared: None,
            checkpoint: None,
        }
    }

    /// Creates an adapter that can roll back to a retained preview checkpoint.
    pub fn with_checkpoint(
        runtime: &'a mut SettingsRuntime,
        checkpoint: SettingsRuntimeCheckpoint,
    ) -> Self {
        Self {
            runtime,
            prepared: None,
            checkpoint: Some(checkpoint),
        }
    }

    /// Returns the checkpoint created when a candidate was installed.
    pub fn take_checkpoint(&mut self) -> Option<SettingsRuntimeCheckpoint> {
        self.checkpoint.take()
    }

    /// Borrows the adapted runtime.
    pub fn runtime(&self) -> &SettingsRuntime {
        self.runtime
    }
}

impl SnapshotController for SettingsRuntimeController<'_> {
    fn install_candidate(&mut self, candidate: &CandidateSnapshot) -> Result<(), String> {
        if self.checkpoint.is_some() || self.prepared.is_some() {
            return Err("settings runtime controller already owns a transaction".to_owned());
        }
        let prepared = self
            .runtime
            .prepare_candidate_source(candidate.source.clone(), &candidate.snapshot)
            .map_err(|error| format!("settings candidate rejected: {error:?}"))?;
        self.prepared = Some(prepared);
        let prepared = self.prepared.take().expect("prepared candidate was set");
        self.checkpoint = Some(
            self.runtime
                .commit_prepared_source(prepared)
                .map_err(|error| format!("settings candidate commit failed: {error:?}"))?,
        );
        Ok(())
    }

    fn install_snapshot(&mut self, snapshot: Snapshot) -> Result<(), String> {
        self.restore_before(&snapshot)
    }

    fn restore_before(&mut self, snapshot: &Snapshot) -> Result<(), String> {
        let Some(checkpoint) = self.checkpoint.as_ref() else {
            return Err(
                "settings runtime rollback requires its complete source checkpoint".to_owned(),
            );
        };
        self.runtime
            .restore_checkpoint(checkpoint.clone())
            .map_err(|errors| format!("settings runtime checkpoint restore failed: {errors:?}"))?;
        if !same_snapshot_values_and_sources(self.runtime.snapshot(), snapshot) {
            return Err("restored settings checkpoint does not match transaction snapshot".into());
        }
        Ok(())
    }

    fn restore_generation(&mut self, _generation: u64, _hash: u64) -> Result<(), String> {
        Err("settings startup recovery requires rebuilding authoritative sources before journal recovery".to_owned())
    }
}

fn same_snapshot_values_and_sources(left: &Snapshot, right: &Snapshot) -> bool {
    left.domain() == right.domain()
        && left.len() == right.len()
        && (0..left.len()).all(|index| {
            let id = prism_settings::SettingId(index as u32);
            left.get(id) == right.get(id) && left.source(id) == right.source(id)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_settings::{
        Constraints, Domain, LayerKind, RegistryBuilder, SettingDef, TrustLevel, Value,
    };
    use prism_settings_io::{
        load_with_backup, ApplyOp, ApplyPlan, CandidateBuilder, CompensateOutcome, EffectState,
        FileSettingsPersistence, OpAction, OpOutcome, Reversibility, SettingsApplier,
        SettingsDocument, SettingsTransaction, SettingsTransactionSpec, SourceSpec, TxMode,
    };

    struct TestTempDir(PathBuf);

    impl TestTempDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "prism-app-settings-io-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn registry() -> prism_settings::SettingsRegistry {
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        let mut scale = SettingDef::new("render.scale", "render", Value::Float(1.0));
        scale.constraints = Constraints {
            min: Some(0.25),
            max: Some(2.0),
            ..Constraints::default()
        };
        builder.register(scale);
        builder.finish().unwrap()
    }

    fn candidate(registry: &prism_settings::SettingsRegistry) -> CandidateSnapshot {
        let source = SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file");
        let mut document = SettingsDocument::new(1);
        document.insert("render.scale", Value::Float(0.75));
        CandidateBuilder::new(registry, Domain::Runtime, source)
            .build(document, &[], 1, false)
            .unwrap()
    }

    #[test]
    fn candidate_install_and_checkpoint_rollback_keep_sources_and_snapshot_aligned() {
        let registry = registry();
        let mut runtime = SettingsRuntime::new(registry.clone(), Domain::Runtime).unwrap();
        let before = runtime.snapshot().clone();
        let checkpoint = {
            let mut controller = SettingsRuntimeController::new(&mut runtime);
            controller.install_candidate(&candidate(&registry)).unwrap();
            controller.take_checkpoint().unwrap()
        };
        let id = registry.id(&"render.scale".into()).unwrap();
        assert_eq!(runtime.snapshot().get(id), Some(&Value::Float(0.75)));
        assert!(runtime.source(LayerKind::User, "user-file").is_some());

        let mut controller = SettingsRuntimeController::with_checkpoint(&mut runtime, checkpoint);
        controller.install_snapshot(before.clone()).unwrap();
        assert_eq!(controller.runtime().snapshot(), &before);
        assert!(controller
            .runtime()
            .source(LayerKind::User, "user-file")
            .is_none());
    }

    #[test]
    fn startup_recovery_refuses_to_invent_authoritative_source_state() {
        let mut runtime = SettingsRuntime::new(registry(), Domain::Runtime).unwrap();
        let mut controller = SettingsRuntimeController::new(&mut runtime);
        assert!(controller.restore_generation(0, 0).is_err());
    }

    struct FailingScaleApplier;

    impl SettingsApplier for FailingScaleApplier {
        fn group(&self) -> &str {
            "Render"
        }

        fn order(&self) -> i32 {
            0
        }

        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            OpOutcome::Failed {
                error: "injected renderer reconfiguration failure".to_owned(),
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

    #[test]
    fn direct_apply_failure_restores_persisted_document_and_runtime_source() {
        let registry = registry();
        let directory = TestTempDir::new();
        let settings_path = directory.path().join("settings.toml");
        let backup_path = directory.path().join("settings.bak");
        let journal_path = directory.path().join("settings.journal");
        let mut previous_document = SettingsDocument::new(1);
        previous_document.insert("render.scale", Value::Float(0.5));
        prism_settings_io::write_atomic(&settings_path, &previous_document, Some(&backup_path))
            .unwrap();

        let mut runtime = SettingsRuntime::new(registry.clone(), Domain::Runtime).unwrap();
        let mut previous_source =
            prism_settings::LayerSource::new(LayerKind::User, TrustLevel::User, "user-file");
        previous_source.set("render.scale", Value::Float(0.5));
        runtime.replace_source(previous_source.clone()).unwrap();
        let before_snapshot = runtime.snapshot().clone();
        let base_sources = runtime.sources().cloned().collect::<Vec<_>>();

        let mut next_document = SettingsDocument::new(1);
        next_document.insert("render.scale", Value::Float(0.8));
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(next_document, &base_sources, 2, false)
        .unwrap();
        let plan = ApplyPlan::new(vec![ApplyOp {
            index: 0,
            id: "render-scale".to_owned(),
            group: "Render".to_owned(),
            key: "render.scale".into(),
            action: OpAction::Set,
            reversibility: Reversibility::Reversible,
            idempotent: true,
            input_digest: 80,
            effect_handle: None,
        }])
        .unwrap();
        let transaction = SettingsTransaction::new(SettingsTransactionSpec {
            tx_id: "direct-rollback-regression".to_owned(),
            journal_path,
            mode: TxMode::Direct,
            before_hash: 50,
            before_snapshot,
            candidate,
            plan,
            snapshot_gen: 1,
            last_confirmed: Some(1),
        });

        let mut snapshots = SettingsRuntimeController::new(&mut runtime);
        let mut persistence = FileSettingsPersistence::new(&settings_path, Some(&backup_path), 2);
        let mut applier = FailingScaleApplier;
        let mut appliers: [&mut dyn SettingsApplier; 1] = [&mut applier];
        let outcome = transaction
            .execute(&mut snapshots, &mut persistence, &mut appliers)
            .unwrap();
        assert_eq!(outcome, prism_settings_io::PipelineOutcome::RolledBack);

        let restored_document =
            load_with_backup(&settings_path, Some(&backup_path), Some(&registry))
                .unwrap()
                .document;
        assert_eq!(
            restored_document.candidates().get(&"render.scale".into()),
            Some(&Value::Float(0.5))
        );
        let id = registry.id(&"render.scale".into()).unwrap();
        assert_eq!(runtime.snapshot().get(id), Some(&Value::Float(0.5)));
        assert_eq!(
            runtime.source(LayerKind::User, "user-file"),
            Some(&previous_source)
        );
    }
}
