//! Completion-gated publish and reclaim state for material-VT GPU updates.

use alloc::vec::Vec;

use prism_render_architecture::gpu_scene::GpuCompletionValue;

use crate::shading::texel_cache::gpu::page_table::MaterialVirtualTexturePageTableSnapshot;

use super::{
    abi::MaterialVirtualTextureGpuConfig,
    staging::{MaterialVirtualTextureStaging, MaterialVirtualTextureUploadRanges},
};

/// Error from the bundle's staging/submission/publication state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVirtualTextureLifecycleError {
    /// A previous update is still awaiting completion/publication.
    UploadInFlight,
    /// The token is not the update currently owned by this state machine.
    InvalidToken,
    /// Completion values must be nonzero.
    InvalidCompletion,
    /// The queue has not completed this update yet.
    CompletionNotReached,
    /// A publication token is stale or already consumed.
    InvalidPublication,
    /// A table generation/epoch rolled backwards.
    SnapshotRollback,
    /// The staging payload or resource slice cannot be paired with this update.
    InvalidResourceSlot,
    /// The staging bytes/config do not describe the supplied page-table snapshot.
    InvalidStaging,
    /// A submitted update cannot be aborted because the GPU may still consume it.
    SubmissionInFlight,
}

/// External queue completion proof associated with one upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureSubmission {
    serial: u64,
    completion: GpuCompletionValue,
    resource_slot: u8,
}

impl MaterialVirtualTextureSubmission {
    /// Upload serial used to match staging, completion, and publication.
    #[must_use]
    pub const fn serial(self) -> u64 {
        self.serial
    }

    /// Queue completion value supplied by the real submission tracker.
    #[must_use]
    pub const fn completion(self) -> GpuCompletionValue {
        self.completion
    }

    /// Resource slice that received the queue writes for this submission.
    #[must_use]
    pub const fn resource_slot(self) -> u8 {
        self.resource_slot
    }
}

/// Data that may be published after the submission's completion is observed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureReadyToPublish {
    serial: u64,
    completion: GpuCompletionValue,
    snapshot: MaterialVirtualTexturePageTableSnapshot,
    config: MaterialVirtualTextureGpuConfig,
    resource_slot: u8,
}

impl MaterialVirtualTextureReadyToPublish {
    /// Upload serial represented by this token.
    #[must_use]
    pub const fn serial(&self) -> u64 {
        self.serial
    }

    /// Snapshot whose page table bytes are now GPU-visible.
    #[must_use]
    pub fn snapshot(&self) -> &MaterialVirtualTexturePageTableSnapshot {
        &self.snapshot
    }

    /// Config whose bytes are now GPU-visible.
    #[must_use]
    pub const fn config(&self) -> MaterialVirtualTextureGpuConfig {
        self.config
    }

    /// Resource slice whose complete upload is now safe to publish.
    #[must_use]
    pub const fn resource_slot(&self) -> u8 {
        self.resource_slot
    }
}

#[derive(Clone, Debug)]
struct InFlight {
    serial: u64,
    staging: MaterialVirtualTextureStaging,
    ranges: MaterialVirtualTextureUploadRanges,
    submission: Option<MaterialVirtualTextureSubmission>,
    snapshot: MaterialVirtualTexturePageTableSnapshot,
}

#[derive(Clone, Debug)]
struct RetiredPublication {
    serial: u64,
    retire_after: GpuCompletionValue,
    snapshot: MaterialVirtualTexturePageTableSnapshot,
}

/// CPU lifecycle coordinator for the independent GPU bundle.
///
/// It does not guess queue progress. The caller must pass the completion value
/// returned by its actual queue submission tracker and a separately observed
/// completed value before `complete` or `reclaim` can advance state.
#[derive(Clone, Debug, Default)]
pub struct MaterialVirtualTextureLifecycle {
    next_serial: u64,
    active: Option<MaterialVirtualTextureReadyToPublish>,
    in_flight: Option<InFlight>,
    retired: Vec<RetiredPublication>,
}

impl MaterialVirtualTextureLifecycle {
    /// Creates an empty lifecycle with no published material-VT table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_serial: 0,
            active: None,
            in_flight: None,
            retired: Vec::new(),
        }
    }

    /// Reserves a staged update before its queue writes are recorded.
    ///
    /// The caller must either call [`mark_submitted`](Self::mark_submitted)
    /// after the queue writes succeed, or call [`abort`](Self::abort) while
    /// no writes have been recorded. This ordering keeps a failed upload from
    /// releasing ownership of a resource slot that the GPU may still read.
    pub fn enqueue(
        &mut self,
        staging: MaterialVirtualTextureStaging,
        snapshot: MaterialVirtualTexturePageTableSnapshot,
        ranges: MaterialVirtualTextureUploadRanges,
    ) -> Result<u64, MaterialVirtualTextureLifecycleError> {
        if self.in_flight.is_some() {
            return Err(MaterialVirtualTextureLifecycleError::UploadInFlight);
        }
        if ranges.resource_slot >= 2 {
            return Err(MaterialVirtualTextureLifecycleError::InvalidResourceSlot);
        }
        let config = staging.config();
        let expected_entry_count = u32::try_from(snapshot.len())
            .map_err(|_| MaterialVirtualTextureLifecycleError::InvalidStaging)?;
        if config.entry_count != expected_entry_count
            || config.enabled != u32::from(!snapshot.is_empty())
            || staging.page_table_words() != snapshot.gpu_words()
        {
            return Err(MaterialVirtualTextureLifecycleError::InvalidStaging);
        }
        if let Some(active) = &self.active {
            if snapshot.epoch() < active.snapshot.epoch()
                || (snapshot.epoch() == active.snapshot.epoch()
                    && snapshot.generation() <= active.snapshot.generation())
            {
                return Err(MaterialVirtualTextureLifecycleError::SnapshotRollback);
            }
            if ranges.resource_slot == active.resource_slot {
                return Err(MaterialVirtualTextureLifecycleError::InvalidResourceSlot);
            }
        }
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or(MaterialVirtualTextureLifecycleError::InvalidToken)?;
        let serial = self.next_serial;
        self.in_flight = Some(InFlight {
            serial,
            staging,
            ranges,
            submission: None,
            snapshot,
        });
        Ok(serial)
    }

    /// Records the actual queue timeline value after `upload_staging`.
    pub fn mark_submitted(
        &mut self,
        serial: u64,
        completion: GpuCompletionValue,
    ) -> Result<MaterialVirtualTextureSubmission, MaterialVirtualTextureLifecycleError> {
        if completion.0 == 0 {
            return Err(MaterialVirtualTextureLifecycleError::InvalidCompletion);
        }
        let in_flight = self
            .in_flight
            .as_mut()
            .ok_or(MaterialVirtualTextureLifecycleError::InvalidToken)?;
        if in_flight.serial != serial || in_flight.submission.is_some() {
            return Err(MaterialVirtualTextureLifecycleError::InvalidToken);
        }
        let resource_slot = in_flight.ranges.resource_slot;
        if resource_slot >= 2 {
            return Err(MaterialVirtualTextureLifecycleError::InvalidResourceSlot);
        }
        let submission = MaterialVirtualTextureSubmission {
            serial,
            completion,
            resource_slot,
        };
        in_flight.submission = Some(submission);
        Ok(submission)
    }

    /// Exposes the exact ranges retained for diagnostics until completion.
    #[must_use]
    pub fn uploaded_ranges(&self, serial: u64) -> Option<&MaterialVirtualTextureUploadRanges> {
        self.in_flight
            .as_ref()
            .filter(|upload| upload.serial == serial)
            .map(|upload| &upload.ranges)
    }

    /// Converts a submitted update to a publication token after completion.
    pub fn complete(
        &mut self,
        submission: MaterialVirtualTextureSubmission,
        completed: GpuCompletionValue,
    ) -> Result<MaterialVirtualTextureReadyToPublish, MaterialVirtualTextureLifecycleError> {
        let in_flight = self
            .in_flight
            .as_ref()
            .ok_or(MaterialVirtualTextureLifecycleError::InvalidToken)?;
        if in_flight.submission != Some(submission) {
            return Err(MaterialVirtualTextureLifecycleError::InvalidToken);
        }
        if completed.0 < submission.completion.0 {
            return Err(MaterialVirtualTextureLifecycleError::CompletionNotReached);
        }
        // Do not consume `in_flight` here. Publication is a second state
        // transition and may fail validation (for example when a caller
        // supplies an invalid retirement proof). Keeping the submission
        // owned by the lifecycle makes that failure retryable instead of
        // leaving the producer with a pending token that can never complete.
        Ok(MaterialVirtualTextureReadyToPublish {
            serial: in_flight.serial,
            completion: submission.completion,
            snapshot: in_flight.snapshot.clone(),
            config: in_flight.staging.config(),
            resource_slot: submission.resource_slot,
        })
    }

    /// Aborts an update before a queue completion proof has been recorded.
    ///
    /// Once [`mark_submitted`](Self::mark_submitted) succeeds, the queue may
    /// still read the staging slice. Such an update must remain owned by this
    /// lifecycle until completion and publication; callers must not drop it as
    /// a way to recover from a later publication error.
    pub fn abort(&mut self, serial: u64) -> Result<(), MaterialVirtualTextureLifecycleError> {
        let in_flight = self
            .in_flight
            .as_ref()
            .ok_or(MaterialVirtualTextureLifecycleError::InvalidToken)?;
        if in_flight.serial != serial {
            return Err(MaterialVirtualTextureLifecycleError::InvalidToken);
        }
        if in_flight.submission.is_some() {
            return Err(MaterialVirtualTextureLifecycleError::SubmissionInFlight);
        }
        self.in_flight = None;
        Ok(())
    }

    /// Publishes a completion-verified table and retires the old one.
    pub fn publish(
        &mut self,
        ready: MaterialVirtualTextureReadyToPublish,
        retire_after: GpuCompletionValue,
    ) -> Result<(), MaterialVirtualTextureLifecycleError> {
        if retire_after.0 == 0 {
            return Err(MaterialVirtualTextureLifecycleError::InvalidCompletion);
        }
        let in_flight = self
            .in_flight
            .as_ref()
            .filter(|in_flight| in_flight.serial == ready.serial)
            .ok_or(MaterialVirtualTextureLifecycleError::InvalidPublication)?;
        let Some(submission) = in_flight.submission else {
            return Err(MaterialVirtualTextureLifecycleError::InvalidPublication);
        };
        if submission.completion != ready.completion
            || submission.resource_slot != ready.resource_slot
            || ready.snapshot != in_flight.snapshot
            || ready.config != in_flight.staging.config()
        {
            return Err(MaterialVirtualTextureLifecycleError::InvalidPublication);
        }
        if retire_after.0 < submission.completion.0 {
            return Err(MaterialVirtualTextureLifecycleError::InvalidCompletion);
        }
        if self
            .active
            .as_ref()
            .is_some_and(|active| ready.serial <= active.serial)
        {
            return Err(MaterialVirtualTextureLifecycleError::InvalidPublication);
        }
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.resource_slot == ready.resource_slot)
        {
            return Err(MaterialVirtualTextureLifecycleError::InvalidResourceSlot);
        }
        if let Some(active) = self.active.replace(ready.clone()) {
            self.retired.push(RetiredPublication {
                serial: active.serial,
                retire_after,
                snapshot: active.snapshot,
            });
        }
        self.in_flight = None;
        Ok(())
    }

    /// Last completion-verified publication, if any.
    #[must_use]
    pub fn active(&self) -> Option<&MaterialVirtualTextureReadyToPublish> {
        self.active.as_ref()
    }

    /// Drops retired CPU publication snapshots once their GPU proof completes.
    pub fn reclaim(&mut self, completed: GpuCompletionValue) -> usize {
        let before = self.retired.len();
        self.retired.retain(|retired| {
            let _ = retired.serial;
            let _ = retired.snapshot.generation();
            retired.retire_after.0 > completed.0
        });
        before - self.retired.len()
    }

    /// Number of retired publications still retained for GPU safety.
    #[must_use]
    pub fn retired_len(&self) -> usize {
        self.retired.len()
    }
}
