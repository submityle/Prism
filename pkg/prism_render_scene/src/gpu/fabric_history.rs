//! Completion-gated scene-side history for Render Fabric.
//!
//! This module is deliberately a small CPU state machine. It does not allocate
//! GPU resources, submit command buffers, infer completion from recording, or
//! use a frame-count delay. The caller stages exact [`FrameResourceBinding`]
//! values before recording, attaches the completion value returned by the
//! actual host queue after submission, and polls the same
//! [`GpuCompletionTracker`] before allowing publication or retirement.

use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt;

use bevy_ecs::{
    prelude::{Res, ResMut},
    resource::Resource,
};
use bevy_render::renderer::RenderQueue;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_architecture::render_fabric::{
    FrameResourceBinding, HistoryCompletion, HistoryError, HistoryPublication, ResourceHistory,
    ResourceKey, ResourceTemplate,
};

use crate::completion::GpuCompletionTracker;

/// A binding paired with the semantic template that validates it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FabricHistoryBinding {
    pub template: ResourceTemplate,
    pub binding: FrameResourceBinding,
}

/// Opaque CPU serial for one staged Fabric submission.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FabricHistorySerial(u64);

impl FabricHistorySerial {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The real host-queue completion attached to a staged submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FabricHistorySubmission {
    serial: FabricHistorySerial,
    completion: GpuCompletionValue,
}

impl FabricHistorySubmission {
    #[must_use]
    pub const fn serial(self) -> FabricHistorySerial {
        self.serial
    }

    #[must_use]
    pub const fn completion(self) -> GpuCompletionValue {
        self.completion
    }
}

/// Progress made when an observed completion value is polled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FabricHistoryProgress {
    /// Number of exact bindings moved into completed `ResourceHistory`.
    pub published_bindings: usize,
    /// Number of explicit retirement requests that advanced a resource
    /// retirement watermark.
    pub retired_ranges: usize,
}

impl FabricHistoryProgress {
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.published_bindings == 0 && self.retired_ranges == 0
    }
}

/// Errors from the scene-side Fabric history lifecycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FabricHistoryError {
    InvalidCompletion,
    CompletionOutOfOrder {
        previous: GpuCompletionValue,
        next: GpuCompletionValue,
    },
    CompletionRegressed {
        previous: GpuCompletionValue,
        next: GpuCompletionValue,
    },
    UnknownSubmission(FabricHistorySerial),
    SubmissionAlreadyMarked(FabricHistorySerial),
    BindingAlreadyPending {
        resource: ResourceKey,
        frame: u64,
    },
    BindingAlreadyRetired {
        resource: ResourceKey,
        frame: u64,
        first_live_frame: u64,
    },
    History(HistoryError),
}

impl fmt::Display for FabricHistoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "scene Fabric history error: {self:?}")
    }
}

impl std::error::Error for FabricHistoryError {}

impl From<HistoryError> for FabricHistoryError {
    fn from(value: HistoryError) -> Self {
        Self::History(value)
    }
}

#[derive(Clone, Debug)]
struct PendingSubmission {
    bindings: Vec<FabricHistoryBinding>,
    completion: Option<GpuCompletionValue>,
}

#[derive(Clone, Copy, Debug)]
struct PendingRetirement {
    resource: ResourceKey,
    first_live_frame: u64,
    completion: GpuCompletionValue,
    order: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CompletionEvent {
    Publication {
        completion: GpuCompletionValue,
        serial: FabricHistorySerial,
    },
    Retirement {
        completion: GpuCompletionValue,
        order: u64,
    },
}

impl CompletionEvent {
    fn key(self) -> (GpuCompletionValue, u8, u64) {
        match self {
            // Publish before a retirement with the same token. This models a
            // replacement whose producer and prior-resource retirement were
            // both proven by the same host submission.
            Self::Publication { completion, serial } => (completion, 0, serial.0),
            Self::Retirement { completion, order } => (completion, 1, order),
        }
    }
}

/// Pure CPU coordinator for exact Fabric history publication and retirement.
///
/// The lifecycle has three observable phases:
///
/// 1. [`Self::enqueue`] retains exact semantic bindings before recording.
/// 2. [`Self::mark_host_submitted`] attaches a real completion token after the
///    host `RenderQueue` submission.
/// 3. [`Self::poll_completed`] publishes and retires only after the tracker has
///    observed that token complete.
///
/// `ResourceHistory` is therefore never populated by recording alone, and a
/// missing exact publication never falls back to an older frame.
#[derive(Clone, Debug, Default)]
pub struct FabricHistoryLifecycle {
    next_serial: u64,
    next_retirement_order: u64,
    last_submission_completion: Option<GpuCompletionValue>,
    last_observed_completion: GpuCompletionValue,
    pending: BTreeMap<FabricHistorySerial, PendingSubmission>,
    retirements: Vec<PendingRetirement>,
    retired_before: BTreeMap<ResourceKey, u64>,
    history: ResourceHistory,
}

impl FabricHistoryLifecycle {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_serial: 0,
            next_retirement_order: 0,
            last_submission_completion: None,
            last_observed_completion: GpuCompletionValue(0),
            pending: BTreeMap::new(),
            retirements: Vec::new(),
            retired_before: BTreeMap::new(),
            history: ResourceHistory::default(),
        }
    }

    /// Stages exact bindings before the producing commands are recorded.
    ///
    /// Staging does not publish anything and does not imply that the GPU has
    /// seen the binding. The returned serial must be marked with the token of
    /// the host submission that contains the producing commands.
    pub fn enqueue(
        &mut self,
        bindings: impl IntoIterator<Item = FabricHistoryBinding>,
    ) -> Result<FabricHistorySerial, FabricHistoryError> {
        let bindings: Vec<_> = bindings.into_iter().collect();
        for (index, current) in bindings.iter().enumerate() {
            if let Some(&first_live_frame) = self.retired_before.get(&current.binding.key)
                && current.binding.frame < first_live_frame
            {
                return Err(FabricHistoryError::BindingAlreadyRetired {
                    resource: current.binding.key,
                    frame: current.binding.frame,
                    first_live_frame,
                });
            }
            if bindings[..index].iter().any(|previous| {
                previous.binding.key == current.binding.key
                    && previous.binding.frame == current.binding.frame
            }) || self.pending.values().any(|submission| {
                submission.bindings.iter().any(|previous| {
                    previous.binding.key == current.binding.key
                        && previous.binding.frame == current.binding.frame
                })
            }) {
                return Err(FabricHistoryError::BindingAlreadyPending {
                    resource: current.binding.key,
                    frame: current.binding.frame,
                });
            }
        }

        self.next_serial =
            self.next_serial
                .checked_add(1)
                .ok_or(FabricHistoryError::UnknownSubmission(FabricHistorySerial(
                    u64::MAX,
                )))?;
        let serial = FabricHistorySerial(self.next_serial);
        self.pending.insert(
            serial,
            PendingSubmission {
                bindings,
                completion: None,
            },
        );
        Ok(serial)
    }

    /// Attaches a completion value returned by the actual submitting queue.
    ///
    /// This is the CPU/test seam. Production render-world code should use
    /// [`Self::mark_host_submitted`] so the value comes from the host queue's
    /// completion callback registration.
    pub fn mark_submitted(
        &mut self,
        serial: FabricHistorySerial,
        completion: GpuCompletionValue,
    ) -> Result<FabricHistorySubmission, FabricHistoryError> {
        self.mark_submitted_batch(core::slice::from_ref(&serial), completion)?;
        Ok(FabricHistorySubmission { serial, completion })
    }

    /// Attaches a token from the real Bevy host queue after its submission.
    ///
    /// The caller must invoke this after the host has submitted the command
    /// buffers containing the staged bindings. Registering the callback is not
    /// itself treated as completion; the binding remains pending until a later
    /// [`Self::poll_completed`] observes the tracker value.
    pub fn mark_host_submitted(
        &mut self,
        serial: FabricHistorySerial,
        tracker: &GpuCompletionTracker,
        queue: &RenderQueue,
    ) -> Result<FabricHistorySubmission, FabricHistoryError> {
        self.validate_mark_submitted(serial)?;
        let completion = tracker.track_host(queue);
        self.mark_submitted(serial, completion)
    }

    /// Attaches one real host-queue completion token to every staged binding
    /// that has not already been attached to a submission.
    ///
    /// `RenderGraphSystems::Finish` runs after Bevy's
    /// `submit_pending_command_buffers` system. The callback is therefore
    /// registered only after the render-world command buffers have entered the
    /// host queue. One callback is shared by all batches from that graph
    /// submission; registering one callback per batch would add avoidable
    /// queue bookkeeping without providing stronger lifetime proof.
    pub(crate) fn mark_pending_host_submissions(
        &mut self,
        tracker: &GpuCompletionTracker,
        queue: &RenderQueue,
    ) -> Result<usize, FabricHistoryError> {
        self.mark_pending_host_submissions_with(|| tracker.track_host(queue))
    }

    /// Requests explicit retirement of one resource's old history prefix.
    ///
    /// The prefix is removed only after `completion` is observed complete. A
    /// caller must supply the token proving that no GPU use of the retired
    /// prefix remains in flight.
    pub fn retire_before_after(
        &mut self,
        resource: ResourceKey,
        first_live_frame: u64,
        completion: GpuCompletionValue,
    ) -> Result<(), FabricHistoryError> {
        if completion.0 == 0 {
            return Err(FabricHistoryError::InvalidCompletion);
        }
        if self
            .retired_before
            .get(&resource)
            .is_some_and(|&current| first_live_frame <= current)
        {
            return Ok(());
        }
        self.next_retirement_order = self
            .next_retirement_order
            .checked_add(1)
            .ok_or(FabricHistoryError::InvalidCompletion)?;
        self.retirements.push(PendingRetirement {
            resource,
            first_live_frame,
            completion,
            order: self.next_retirement_order,
        });
        Ok(())
    }

    /// Polls the actual completion value owned by the render-world tracker.
    pub fn poll_completed(
        &mut self,
        tracker: &GpuCompletionTracker,
    ) -> Result<FabricHistoryProgress, FabricHistoryError> {
        self.publish_completed(tracker.completed_value())
    }

    /// CPU/test seam for advancing the state machine with an observed queue
    /// completion value. No frame count or recording state is consulted.
    pub fn publish_completed(
        &mut self,
        completed: GpuCompletionValue,
    ) -> Result<FabricHistoryProgress, FabricHistoryError> {
        if completed < self.last_observed_completion {
            return Err(FabricHistoryError::CompletionRegressed {
                previous: self.last_observed_completion,
                next: completed,
            });
        }

        let mut events = Vec::new();
        for (&serial, submission) in &self.pending {
            if let Some(completion) = submission.completion
                && completion <= completed
            {
                events.push(CompletionEvent::Publication { completion, serial });
            }
        }
        for retirement in &self.retirements {
            if retirement.completion <= completed {
                events.push(CompletionEvent::Retirement {
                    completion: retirement.completion,
                    order: retirement.order,
                });
            }
        }
        events.sort_by_key(|event| event.key());

        let mut projected_history = self.history.clone();
        let mut projected_retired_before = self.retired_before.clone();
        let mut progress = FabricHistoryProgress::default();

        for event in events {
            match event {
                CompletionEvent::Publication { completion, serial } => {
                    let submission = self
                        .pending
                        .get(&serial)
                        .expect("completion event references pending submission");
                    for staged in &submission.bindings {
                        if let Some(&first_live_frame) =
                            projected_retired_before.get(&staged.binding.key)
                            && staged.binding.frame < first_live_frame
                        {
                            return Err(FabricHistoryError::BindingAlreadyRetired {
                                resource: staged.binding.key,
                                frame: staged.binding.frame,
                                first_live_frame,
                            });
                        }
                        projected_history.publish(
                            staged.template,
                            staged.binding,
                            HistoryCompletion(completion.0),
                        )?;
                        progress.published_bindings += 1;
                    }
                }
                CompletionEvent::Retirement { order, .. } => {
                    let retirement = self
                        .retirements
                        .iter()
                        .find(|retirement| retirement.order == order)
                        .expect("completion event references pending retirement");
                    let should_advance = projected_retired_before
                        .get(&retirement.resource)
                        .is_none_or(|&current| retirement.first_live_frame > current);
                    if should_advance {
                        projected_retired_before
                            .insert(retirement.resource, retirement.first_live_frame);
                        projected_history
                            .retire_before(retirement.resource, retirement.first_live_frame);
                        progress.retired_ranges += 1;
                    }
                }
            }
        }

        self.history = projected_history;
        self.retired_before = projected_retired_before;
        self.last_observed_completion = completed;
        self.pending.retain(|_, submission| {
            submission
                .completion
                .is_none_or(|completion| completion > completed)
        });
        self.retirements
            .retain(|retirement| retirement.completion > completed);
        Ok(progress)
    }

    /// Validates an exact `(resource, frame, version, physical, layout)` import
    /// against a completion-published binding.
    pub fn validate_import(
        &self,
        binding: FrameResourceBinding,
    ) -> Result<HistoryPublication, HistoryError> {
        self.history.validate_import(binding)
    }

    #[must_use]
    pub fn history(&self) -> &ResourceHistory {
        &self.history
    }

    #[must_use]
    pub fn pending_submission_count(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub fn pending_binding_count(&self) -> usize {
        self.pending
            .values()
            .map(|submission| submission.bindings.len())
            .sum()
    }

    #[must_use]
    pub fn pending_retirement_count(&self) -> usize {
        self.retirements.len()
    }

    #[must_use]
    pub fn retired_before(&self, resource: ResourceKey) -> Option<u64> {
        self.retired_before.get(&resource).copied()
    }

    fn validate_mark_submitted(
        &self,
        serial: FabricHistorySerial,
    ) -> Result<(), FabricHistoryError> {
        let pending = self
            .pending
            .get(&serial)
            .ok_or(FabricHistoryError::UnknownSubmission(serial))?;
        if pending.completion.is_some() {
            return Err(FabricHistoryError::SubmissionAlreadyMarked(serial));
        }
        Ok(())
    }

    fn mark_submitted_batch(
        &mut self,
        serials: &[FabricHistorySerial],
        completion: GpuCompletionValue,
    ) -> Result<(), FabricHistoryError> {
        if completion.0 == 0 {
            return Err(FabricHistoryError::InvalidCompletion);
        }
        if let Some(previous) = self.last_submission_completion
            && completion <= previous
        {
            return Err(FabricHistoryError::CompletionOutOfOrder {
                previous,
                next: completion,
            });
        }
        for &serial in serials {
            self.validate_mark_submitted(serial)?;
        }
        for &serial in serials {
            self.pending
                .get_mut(&serial)
                .expect("validated pending submission disappeared")
                .completion = Some(completion);
        }
        self.last_submission_completion = Some(completion);
        Ok(())
    }

    fn mark_pending_host_submissions_with(
        &mut self,
        register_completion: impl FnOnce() -> GpuCompletionValue,
    ) -> Result<usize, FabricHistoryError> {
        let serials: Vec<_> = self
            .pending
            .iter()
            .filter_map(|(&serial, submission)| submission.completion.is_none().then_some(serial))
            .collect();
        if serials.is_empty() {
            return Ok(0);
        }
        let completion = register_completion();
        self.mark_submitted_batch(&serials, completion)?;
        Ok(serials.len())
    }
}

/// Render-world resource owning the scene-side Fabric history coordinator.
#[derive(Resource, Debug, Default)]
pub struct SceneFabricHistory {
    pub lifecycle: FabricHistoryLifecycle,
    /// The last lifecycle error observed by the Finish-system poller. Keeping
    /// it in the resource makes a failed publication visible and retryable;
    /// the poller never discards a pending binding on error.
    pub last_error: Option<FabricHistoryError>,
}

/// Finish-stage render-world hook. The queue tracker is the only source of
/// progress: a frame count, command recording, or callback registration alone
/// cannot move a binding into completed history.
pub(crate) fn poll_scene_fabric_history(
    mut scene_history: ResMut<SceneFabricHistory>,
    tracker: Res<GpuCompletionTracker>,
) {
    match scene_history.lifecycle.poll_completed(&tracker) {
        Ok(_) => scene_history.last_error = None,
        Err(error) => {
            if scene_history.last_error.as_ref() != Some(&error) {
                bevy_log::warn!("Fabric history completion poll failed: {error:?}");
            }
            scene_history.last_error = Some(error);
        }
    }
}

/// Finish-stage submission hook for the scene-side Fabric history.
///
/// A missing render queue is treated as an unavailable submission boundary,
/// not as completion. This keeps headless/initializing render worlds pending
/// until a real host queue is available.
pub(crate) fn mark_scene_fabric_history_submissions(
    mut scene_history: ResMut<SceneFabricHistory>,
    tracker: Res<GpuCompletionTracker>,
    queue: Option<Res<RenderQueue>>,
) {
    let Some(queue) = queue else {
        return;
    };
    if let Err(error) = scene_history
        .lifecycle
        .mark_pending_host_submissions(&tracker, &queue)
    {
        if scene_history.last_error.as_ref() != Some(&error) {
            bevy_log::warn!("Fabric history submission hook failed: {error:?}");
        }
        scene_history.last_error = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{prelude::World, schedule::Schedule};
    use prism_render_architecture::frame_graph::ResourceLifetime;
    use prism_render_architecture::render_fabric::{
        BufferId, BufferUsages, FeatureId, PhysicalResource, ResourceLayout,
    };

    const FEATURE: FeatureId = FeatureId(9000);
    const RESOURCE: ResourceKey = ResourceKey::of(FEATURE, 0);

    fn template() -> ResourceTemplate {
        ResourceTemplate {
            key: RESOURCE,
            lifetime: ResourceLifetime::Persistent,
            layout: ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE,
            },
            alignment: 16,
        }
    }

    fn binding(frame: u64, version: u32, id: u32) -> FrameResourceBinding {
        FrameResourceBinding {
            key: RESOURCE,
            frame,
            version,
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(id, 1),
                offset: 0,
            },
            layout: template().layout,
            alignment: template().alignment,
        }
    }

    fn staged(frame: u64, version: u32, id: u32) -> FabricHistoryBinding {
        FabricHistoryBinding {
            template: template(),
            binding: binding(frame, version, id),
        }
    }

    #[test]
    fn recording_and_submission_do_not_publish_history() {
        let mut lifecycle = FabricHistoryLifecycle::new();
        let frame = binding(40, 7, 1);
        let serial = lifecycle
            .enqueue([FabricHistoryBinding {
                template: template(),
                binding: frame,
            }])
            .unwrap();
        let submission = lifecycle
            .mark_submitted(serial, GpuCompletionValue(3))
            .unwrap();

        assert_eq!(submission.completion(), GpuCompletionValue(3));
        assert_eq!(lifecycle.pending_binding_count(), 1);
        assert!(lifecycle.validate_import(frame).is_err());
        assert_eq!(
            lifecycle.publish_completed(GpuCompletionValue(2)),
            Ok(FabricHistoryProgress::default())
        );
        assert!(lifecycle.validate_import(frame).is_err());
    }

    #[test]
    fn exact_binding_publishes_only_at_observed_completion() {
        let mut lifecycle = FabricHistoryLifecycle::new();
        let frame = binding(41, 8, 2);
        let serial = lifecycle.enqueue([staged(41, 8, 2)]).unwrap();
        lifecycle
            .mark_submitted(serial, GpuCompletionValue(11))
            .unwrap();

        let progress = lifecycle.publish_completed(GpuCompletionValue(11)).unwrap();
        assert_eq!(progress.published_bindings, 1);
        let publication = lifecycle.validate_import(frame).unwrap();
        assert_eq!(publication.binding(), frame);
        assert_eq!(publication.completion(), HistoryCompletion(11));

        let mut stale = frame;
        stale.version += 1;
        assert!(matches!(
            lifecycle.validate_import(stale),
            Err(HistoryError::BindingMismatch { .. })
        ));
    }

    #[test]
    fn retirement_waits_for_its_own_completion_proof() {
        let mut lifecycle = FabricHistoryLifecycle::new();
        let old = binding(5, 1, 3);
        let serial = lifecycle.enqueue([staged(5, 1, 3)]).unwrap();
        lifecycle
            .mark_submitted(serial, GpuCompletionValue(20))
            .unwrap();
        lifecycle.publish_completed(GpuCompletionValue(20)).unwrap();

        lifecycle
            .retire_before_after(RESOURCE, 6, GpuCompletionValue(25))
            .unwrap();
        assert_eq!(lifecycle.pending_retirement_count(), 1);
        assert!(lifecycle.validate_import(old).is_ok());

        let progress = lifecycle.publish_completed(GpuCompletionValue(24)).unwrap();
        assert!(progress.is_empty());
        assert!(lifecycle.validate_import(old).is_ok());

        let progress = lifecycle.publish_completed(GpuCompletionValue(25)).unwrap();
        assert_eq!(progress.retired_ranges, 1);
        assert!(lifecycle.validate_import(old).is_err());
        assert_eq!(lifecycle.retired_before(RESOURCE), Some(6));
    }

    #[test]
    fn retirement_watermark_prevents_old_pending_history_from_reappearing() {
        let mut lifecycle = FabricHistoryLifecycle::new();
        let old = staged(2, 1, 4);
        lifecycle
            .retire_before_after(RESOURCE, 3, GpuCompletionValue(8))
            .unwrap();
        lifecycle.publish_completed(GpuCompletionValue(8)).unwrap();
        let error = lifecycle.enqueue([old]).unwrap_err();
        assert_eq!(
            error,
            FabricHistoryError::BindingAlreadyRetired {
                resource: RESOURCE,
                frame: 2,
                first_live_frame: 3,
            }
        );
    }

    #[test]
    fn completion_tokens_are_monotonic_and_nonzero() {
        let mut lifecycle = FabricHistoryLifecycle::new();
        let first = lifecycle.enqueue([staged(1, 1, 5)]).unwrap();
        assert_eq!(
            lifecycle.mark_submitted(first, GpuCompletionValue(0)),
            Err(FabricHistoryError::InvalidCompletion)
        );
        lifecycle
            .mark_submitted(first, GpuCompletionValue(2))
            .unwrap();
        let second = lifecycle.enqueue([staged(2, 2, 6)]).unwrap();
        assert_eq!(
            lifecycle.mark_submitted(second, GpuCompletionValue(2)),
            Err(FabricHistoryError::CompletionOutOfOrder {
                previous: GpuCompletionValue(2),
                next: GpuCompletionValue(2),
            })
        );
    }

    #[test]
    fn host_submission_hook_uses_one_token_for_all_unmarked_batches() {
        let mut lifecycle = FabricHistoryLifecycle::new();
        let first = lifecycle.enqueue([staged(10, 1, 7)]).unwrap();
        let second = lifecycle.enqueue([staged(11, 2, 8)]).unwrap();

        let mut registrations = 0;
        let marked = lifecycle
            .mark_pending_host_submissions_with(|| {
                registrations += 1;
                GpuCompletionValue(31)
            })
            .unwrap();

        assert_eq!(registrations, 1);
        assert_eq!(marked, 2);
        assert_eq!(lifecycle.pending_submission_count(), 2);
        assert_eq!(lifecycle.pending_binding_count(), 2);
        assert_eq!(
            lifecycle.publish_completed(GpuCompletionValue(30)),
            Ok(FabricHistoryProgress::default())
        );
        assert!(lifecycle.validate_import(binding(10, 1, 7)).is_err());
        assert!(lifecycle.validate_import(binding(11, 2, 8)).is_err());

        let progress = lifecycle.publish_completed(GpuCompletionValue(31)).unwrap();
        assert_eq!(progress.published_bindings, 2);
        assert_eq!(lifecycle.pending_submission_count(), 0);
        assert_eq!(first.get() + 1, second.get());

        let marked = lifecycle
            .mark_pending_host_submissions_with(|| {
                panic!("an already attached submission must not register another callback")
            })
            .unwrap();
        assert_eq!(marked, 0);
        assert_eq!(registrations, 1);
    }

    #[test]
    fn render_world_hook_never_marks_without_a_real_render_queue() {
        let mut world = World::new();
        let mut scene_history = SceneFabricHistory::default();
        scene_history.lifecycle.enqueue([staged(20, 1, 9)]).unwrap();
        world.insert_resource(scene_history);
        world.insert_resource(GpuCompletionTracker::default());

        let mut schedule = Schedule::default();
        schedule.add_systems(mark_scene_fabric_history_submissions);
        schedule.run(&mut world);

        let scene_history = world.resource::<SceneFabricHistory>();
        assert_eq!(scene_history.lifecycle.pending_submission_count(), 1);
        assert_eq!(scene_history.lifecycle.pending_binding_count(), 1);
        assert!(scene_history.last_error.is_none());
    }
}
