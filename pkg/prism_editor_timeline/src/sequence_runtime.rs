//! Runtime adapters for authored sequence camera and resident-audio tracks.
//!
//! This module is intentionally behind the sequence-runtime feature. The
//! timeline kernel remains independent of an ECS, camera renderer, and audio
//! graph. When enabled, the adapter consumes real sampled frames and applies
//! only explicitly bound native Prism components. It never interprets event
//! names as gameplay policy.

use crate::{SequenceFrame, SequenceInstance, SequenceProjectToken, TimelineError, TrackId};
use alloc::{collections::BTreeMap, vec::Vec};
use core::fmt;
use prism_audio_assets::asset::AudioAssetHandle;
use prism_audio_ecs::{AudioAssetStore, AudioPlaybackState, AudioPlayer};
use prism_camera_ecs::{CameraIntentComponent, CameraPoseComponent, ViewHistoryComponent};
use prism_ecs::{entity::Entity, world::World, world::WorldId};
use prism_math::{DQuat, DVec3};

/// Hard upper bound for one prepared frame.
pub const MAX_RUNTIME_CHANGES: usize = 4096;

/// Stable path key disambiguating reused and repeated subsequences.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RuntimePath(pub Vec<SequenceInstance>);

impl RuntimePath {
    /// Creates a path key from a sequence frame path.
    #[must_use]
    pub fn new(path: &[SequenceInstance]) -> Self {
        Self(path.to_vec())
    }
}

/// Explicit camera pose channel binding. No project role is inferred.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CameraTrackBinding {
    /// Sequence document containing the channels.
    pub sequence: crate::SequenceId,
    /// Nested instance path containing the channels.
    pub path: RuntimePath,
    /// Native camera entity owned by the host.
    pub entity: Entity,
    /// Position channel, exactly three finite values.
    pub position: TrackId,
    /// Quaternion channel in x,y,z,w order, exactly four finite values.
    pub rotation: TrackId,
    /// Optional world-origin channel, exactly three finite values.
    pub world_origin: Option<TrackId>,
    /// Optional seek-safe cut selector. A sampled integer selects a camera.
    pub selector: Option<TrackId>,
    /// Value selected by the optional selector track.
    pub cut_value: u64,
    /// Explicit cut group; only cameras in this group are toggled.
    pub cut_group: u128,
}

/// Explicit camera switch authored by a stable event ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CameraCutBinding {
    /// Source sequence containing the event.
    pub sequence: crate::SequenceId,
    /// Nested instance path containing the event.
    pub path: RuntimePath,
    /// Stable authored event identity.
    pub event_id: u128,
    /// Camera selected by this cut.
    pub target: Entity,
    /// Cut group to reconcile.
    pub cut_group: u128,
}

/// Audio time mapping from a local authored tick to source frames.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioTimeBinding {
    /// Sequence track containing the authored local tick.
    pub track: TrackId,
    /// Source-frame offset at local tick zero.
    pub offset_frames: u64,
}

/// Explicit audio event operation. Event names are never parsed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioEventAction {
    /// Start without resetting source position.
    Start,
    /// Pause while retaining the exact source frame.
    Pause,
    /// Stop and silence while retaining the entity.
    Stop,
    /// Seek to the frame carried by the matching sample.
    Seek,
}

/// Audio player binding owned by the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioTrackBinding {
    /// Sequence document containing the source-time track.
    pub sequence: crate::SequenceId,
    /// Nested instance path containing the source-time track.
    pub path: RuntimePath,
    /// Native player entity owned by the host.
    pub entity: Entity,
    /// Source-time track and offset.
    pub time: AudioTimeBinding,
    /// Stable asset identity expected on the native `AudioPlayer`.
    pub asset: AudioAssetHandle,
}

/// Explicit audio event operation bound by stable source identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioEventBinding {
    /// Source sequence containing the event.
    pub sequence: crate::SequenceId,
    /// Nested instance path containing the event.
    pub path: RuntimePath,
    /// Stable authored event identity.
    pub event_id: u128,
    /// Native player entity owned by the host.
    pub entity: Entity,
    /// Operation to apply after all samples are validated.
    pub action: AudioEventAction,
}

/// Explicit transport state for one prepared frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeTransport {
    /// Apply sampled state and crossed events.
    Playing,
    /// Apply sampled state while preserving audio position.
    Paused,
    /// Apply final camera silence and audio stop without events.
    Stopped,
    /// Reconstruct state from samples only; seek emits no events.
    Seek,
}

/// Adapter resource limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceRuntimeLimits {
    /// Maximum native component changes in one frame.
    pub max_changes: usize,
    /// Maximum event operations in one frame.
    pub max_events: usize,
}
impl Default for SequenceRuntimeLimits {
    fn default() -> Self {
        Self {
            max_changes: 1024,
            max_events: 1024,
        }
    }
}

/// Errors are reported before native mutation whenever possible. Rollback is
/// rejected after publication or an intervening component change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequenceRuntimeError {
    /// Resource policy is outside the bounded contract.
    InvalidLimits,
    /// Project token, revision, or ECS world is stale.
    StaleFence,
    /// The host attempted to reuse a published frame.
    Published,
    /// A binding referenced a missing native component.
    MissingEntity(Entity),
    /// Two bindings would write the same native property ambiguously.
    AmbiguousBinding,
    /// A sampled channel has wrong arity or a non-finite value.
    InvalidChannel(TrackId),
    /// A quaternion is zero-length or non-finite.
    InvalidRotation,
    /// Native `AudioPlayer` asset differs from its binding.
    AudioAssetMismatch(Entity),
    /// Audio content is unavailable, streaming, or outside source range.
    AudioUnavailable(Entity),
    /// Integer conversion overflowed.
    ArithmeticOverflow,
    /// Audio route epoch cannot advance without wrapping.
    EpochExhausted(Entity),
    /// Requested operation exceeds a frame budget.
    Limit(&'static str),
    /// A real timeline sample/evaluation failure.
    Timeline(TimelineError),
}
impl fmt::Display for SequenceRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sequence runtime {self:?}")
    }
}
impl core::error::Error for SequenceRuntimeError {}
impl From<TimelineError> for SequenceRuntimeError {
    fn from(value: TimelineError) -> Self {
        Self::Timeline(value)
    }
}

/// Native world identity and project revision captured at preparation.
#[derive(Clone)]
pub struct RuntimeFence {
    token: SequenceProjectToken,
    revision: u64,
    world: WorldId,
}
impl RuntimeFence {
    /// Captures the exact project token, revision, and ECS world.
    #[must_use]
    pub fn new(token: SequenceProjectToken, revision: u64, world: &World) -> Self {
        Self {
            token,
            revision,
            world: world.id(),
        }
    }
}

#[derive(Clone)]
struct CameraChange {
    entity: Entity,
    before_intent: CameraIntentComponent,
    before_pose: CameraPoseComponent,
    before_history: ViewHistoryComponent,
    after_intent: CameraIntentComponent,
    after_pose: CameraPoseComponent,
    after_history: ViewHistoryComponent,
}
#[derive(Clone)]
struct AudioChange {
    entity: Entity,
    before: AudioPlayer,
    after: AudioPlayer,
}

/// A fully validated, not-yet-applied native update.
pub struct PreparedRuntimeFrame {
    fence: RuntimeFence,
    transport: RuntimeTransport,
    cameras: Vec<CameraChange>,
    audio: Vec<AudioChange>,
    published: bool,
}
impl PreparedRuntimeFrame {
    /// Transport represented by this prepared update.
    #[must_use]
    pub const fn transport(&self) -> RuntimeTransport {
        self.transport
    }
    /// Number of native component records in the update.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.cameras.len() + self.audio.len()
    }
}

/// Bounded adapter for real camera components and resident audio routing.
pub struct SequenceRuntimeAdapter {
    limits: SequenceRuntimeLimits,
    camera_tracks: Vec<CameraTrackBinding>,
    camera_cuts: Vec<CameraCutBinding>,
    audio_tracks: Vec<AudioTrackBinding>,
    audio_events: Vec<AudioEventBinding>,
}
impl SequenceRuntimeAdapter {
    /// Creates an empty adapter. Bindings are explicit host-owned contracts.
    pub fn new(limits: SequenceRuntimeLimits) -> Result<Self, SequenceRuntimeError> {
        if limits.max_changes == 0
            || limits.max_changes > MAX_RUNTIME_CHANGES
            || limits.max_events == 0
            || limits.max_events > MAX_RUNTIME_CHANGES
        {
            return Err(SequenceRuntimeError::InvalidLimits);
        }
        Ok(Self {
            limits,
            camera_tracks: Vec::new(),
            camera_cuts: Vec::new(),
            audio_tracks: Vec::new(),
            audio_events: Vec::new(),
        })
    }
    /// Adds a camera channel binding; duplicate ownership is rejected.
    pub fn bind_camera(&mut self, binding: CameraTrackBinding) -> Result<(), SequenceRuntimeError> {
        if self
            .camera_tracks
            .iter()
            .any(|old| old.entity == binding.entity)
        {
            return Err(SequenceRuntimeError::AmbiguousBinding);
        }
        self.camera_tracks.push(binding);
        Ok(())
    }
    /// Adds a stable camera-cut binding.
    pub fn bind_camera_cut(
        &mut self,
        binding: CameraCutBinding,
    ) -> Result<(), SequenceRuntimeError> {
        self.camera_cuts.push(binding);
        Ok(())
    }
    /// Adds a resident source-time binding.
    pub fn bind_audio(&mut self, binding: AudioTrackBinding) -> Result<(), SequenceRuntimeError> {
        if self
            .audio_tracks
            .iter()
            .any(|old| old.entity == binding.entity)
        {
            return Err(SequenceRuntimeError::AmbiguousBinding);
        }
        self.audio_tracks.push(binding);
        Ok(())
    }
    /// Adds a stable audio event binding.
    pub fn bind_audio_event(
        &mut self,
        binding: AudioEventBinding,
    ) -> Result<(), SequenceRuntimeError> {
        self.audio_events.push(binding);
        Ok(())
    }

    /// Preflights a complete sampled frame without mutating the native world.
    pub fn prepare<'a>(
        &self,
        frame: &SequenceFrame<'a>,
        transport: RuntimeTransport,
        fence: RuntimeFence,
        world: &World,
    ) -> Result<PreparedRuntimeFrame, SequenceRuntimeError> {
        if fence.world != world.id() {
            return Err(SequenceRuntimeError::StaleFence);
        }
        let event_count = if matches!(transport, RuntimeTransport::Playing) {
            frame.events.len()
        } else {
            0
        };
        if event_count > self.limits.max_events {
            return Err(SequenceRuntimeError::Limit("events"));
        }
        let mut cameras = self.prepare_cameras(frame, transport, world)?;
        let mut audio = self.prepare_audio(frame, transport, world)?;
        if matches!(transport, RuntimeTransport::Playing) {
            self.apply_camera_events(&mut cameras, frame);
            self.apply_audio_events(&mut audio, frame, world)?;
        }
        let count = cameras
            .len()
            .checked_add(audio.len())
            .ok_or(SequenceRuntimeError::ArithmeticOverflow)?;
        if count > self.limits.max_changes {
            return Err(SequenceRuntimeError::Limit("changes"));
        }
        Ok(PreparedRuntimeFrame {
            fence,
            transport,
            cameras,
            audio,
            published: false,
        })
    }

    /// Applies only when every fence and preflight snapshot remains exact.
    pub fn apply(
        &self,
        prepared: &mut PreparedRuntimeFrame,
        world: &mut World,
        token: &SequenceProjectToken,
        revision: u64,
    ) -> Result<(), SequenceRuntimeError> {
        if prepared.published
            || prepared.fence.world != world.id()
            || prepared.fence.revision != revision
            || prepared.fence.token != *token
        {
            return Err(SequenceRuntimeError::StaleFence);
        }
        for change in &prepared.cameras {
            if world.get::<CameraIntentComponent>(change.entity).copied()
                != Some(change.before_intent)
                || world.get::<CameraPoseComponent>(change.entity).copied()
                    != Some(change.before_pose)
                || world.get::<ViewHistoryComponent>(change.entity).copied()
                    != Some(change.before_history)
            {
                return Err(SequenceRuntimeError::StaleFence);
            }
        }
        for change in &prepared.audio {
            if world.get::<AudioPlayer>(change.entity).copied() != Some(change.before) {
                return Err(SequenceRuntimeError::StaleFence);
            }
        }
        for change in &prepared.cameras {
            world.insert(
                change.entity,
                (change.after_intent, change.after_pose, change.after_history),
            );
        }
        for change in &prepared.audio {
            world.insert(change.entity, change.after);
        }
        Ok(())
    }
    /// Marks the point after which native publication may be externally visible.
    pub fn publication_barrier(
        &self,
        prepared: &mut PreparedRuntimeFrame,
    ) -> Result<(), SequenceRuntimeError> {
        if prepared.published {
            return Err(SequenceRuntimeError::Published);
        }
        prepared.published = true;
        Ok(())
    }
    /// Restores the before-image only before publication and only if unchanged.
    pub fn rollback(
        &self,
        prepared: &mut PreparedRuntimeFrame,
        world: &mut World,
    ) -> Result<(), SequenceRuntimeError> {
        if prepared.published || prepared.fence.world != world.id() {
            return Err(if prepared.published {
                SequenceRuntimeError::Published
            } else {
                SequenceRuntimeError::StaleFence
            });
        }
        for change in &prepared.cameras {
            if world.get::<CameraIntentComponent>(change.entity).copied()
                != Some(change.after_intent)
                || world.get::<CameraPoseComponent>(change.entity).copied()
                    != Some(change.after_pose)
                || world.get::<ViewHistoryComponent>(change.entity).copied()
                    != Some(change.after_history)
            {
                return Err(SequenceRuntimeError::StaleFence);
            }
        }
        for change in &prepared.audio {
            if world.get::<AudioPlayer>(change.entity).copied() != Some(change.after) {
                return Err(SequenceRuntimeError::StaleFence);
            }
        }
        for change in &prepared.cameras {
            world.insert(
                change.entity,
                (
                    change.before_intent,
                    change.before_pose,
                    change.before_history,
                ),
            );
        }
        for change in &prepared.audio {
            world.insert(change.entity, change.before);
        }
        Ok(())
    }

    fn prepare_cameras<'a>(
        &self,
        frame: &SequenceFrame<'a>,
        transport: RuntimeTransport,
        world: &World,
    ) -> Result<Vec<CameraChange>, SequenceRuntimeError> {
        let mut result = Vec::new();
        for binding in &self.camera_tracks {
            let before_intent = world
                .get::<CameraIntentComponent>(binding.entity)
                .copied()
                .ok_or(SequenceRuntimeError::MissingEntity(binding.entity))?;
            let before_pose = world
                .get::<CameraPoseComponent>(binding.entity)
                .copied()
                .ok_or(SequenceRuntimeError::MissingEntity(binding.entity))?;
            let before_history = world
                .get::<ViewHistoryComponent>(binding.entity)
                .copied()
                .ok_or(SequenceRuntimeError::MissingEntity(binding.entity))?;
            let mut after_pose = before_pose;
            let mut after_intent = before_intent;
            let mut found = false;
            for sample in &frame.samples {
                if sample.sequence != binding.sequence
                    || RuntimePath::new(&sample.path) != binding.path
                {
                    continue;
                }
                let pos = channel(&sample.tracks, binding.position, 3)?;
                let rot = channel(&sample.tracks, binding.rotation, 4)?;
                after_pose.0.position = DVec3::new(pos[0], pos[1], pos[2]);
                let quat = DQuat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
                if !quat.is_finite() || quat.length_squared() <= f64::EPSILON {
                    return Err(SequenceRuntimeError::InvalidRotation);
                }
                after_pose.0.rotation = quat.normalize();
                if let Some(track) = binding.world_origin {
                    let v = channel(&sample.tracks, track, 3)?;
                    after_pose.0.world_origin = DVec3::new(v[0], v[1], v[2]);
                }
                if let Some(selector) = binding.selector {
                    let v = channel(&sample.tracks, selector, 1)?;
                    after_intent.0.active = v[0].round() as u64 == binding.cut_value;
                }
                found = true;
            }
            if matches!(transport, RuntimeTransport::Stopped) {
                after_intent.0.active = false;
            }
            let cut_affects = matches!(transport, RuntimeTransport::Playing)
                && frame.events.iter().any(|event| {
                    self.camera_cuts.iter().any(|cut| {
                        cut.sequence == event.sequence
                            && cut.path == RuntimePath::new(&event.path)
                            && cut.cut_group == binding.cut_group
                            && cut.event_id == event.event.id
                    })
                });
            if found || cut_affects || after_intent != before_intent || after_pose != before_pose {
                let mut history = before_history;
                if matches!(
                    transport,
                    RuntimeTransport::Seek | RuntimeTransport::Stopped
                ) {
                    history.0.clear();
                }
                result.push(CameraChange {
                    entity: binding.entity,
                    before_intent,
                    before_pose,
                    before_history,
                    after_intent,
                    after_pose,
                    after_history: history,
                });
            }
        }
        Ok(result)
    }

    fn prepare_audio<'a>(
        &self,
        frame: &SequenceFrame<'a>,
        transport: RuntimeTransport,
        world: &World,
    ) -> Result<Vec<AudioChange>, SequenceRuntimeError> {
        let mut result = Vec::new();
        let store = world.get_resource::<AudioAssetStore>();
        for binding in &self.audio_tracks {
            let before = world
                .get::<AudioPlayer>(binding.entity)
                .copied()
                .ok_or(SequenceRuntimeError::MissingEntity(binding.entity))?;
            // Stopping is a control-plane operation. It must remain possible
            // after an asset unload or before the audio runtime is installed;
            // requiring decoder metadata here would leave a native player
            // visibly playing when its source is already unavailable.
            if matches!(transport, RuntimeTransport::Stopped) {
                let mut after = before;
                after.stop_requested = true;
                after.playing = false;
                replace_audio(&mut result, binding.entity, before, after);
                continue;
            }
            if before.asset != Some(binding.asset) {
                return Err(SequenceRuntimeError::AudioAssetMismatch(binding.entity));
            }
            let store = store.ok_or(SequenceRuntimeError::AudioUnavailable(binding.entity))?;
            let meta = store
                .resolve(binding.asset)
                .ok_or(SequenceRuntimeError::AudioUnavailable(binding.entity))?;
            let frame_count = meta
                .stream
                .frame_count
                .ok_or(SequenceRuntimeError::AudioUnavailable(binding.entity))?;
            if meta.stream.channels == 0 || meta.stream.channels > 2 || meta.stream.sample_rate == 0
            {
                return Err(SequenceRuntimeError::AudioUnavailable(binding.entity));
            }
            let mut after = before;
            for sample in &frame.samples {
                if sample.sequence != binding.sequence
                    || RuntimePath::new(&sample.path) != binding.path
                {
                    continue;
                }
                let tick_value = channel(&sample.tracks, binding.time.track, 1)?[0];
                if tick_value < 0.0 || tick_value.fract() != 0.0 || tick_value > u64::MAX as f64 {
                    return Err(SequenceRuntimeError::InvalidChannel(binding.time.track));
                }
                let tick = tick_value as u64;
                let numerator = tick
                    .checked_mul(u64::from(meta.stream.sample_rate))
                    .ok_or(SequenceRuntimeError::ArithmeticOverflow)?;
                if sample.document.ticks_per_second == 0 {
                    return Err(SequenceRuntimeError::ArithmeticOverflow);
                }
                let source = numerator
                    .checked_div(sample.document.ticks_per_second)
                    .ok_or(SequenceRuntimeError::ArithmeticOverflow)?
                    .checked_add(binding.time.offset_frames)
                    .ok_or(SequenceRuntimeError::ArithmeticOverflow)?;
                if source >= frame_count {
                    return Err(SequenceRuntimeError::AudioUnavailable(binding.entity));
                }
                after.playhead = source;
                if matches!(transport, RuntimeTransport::Playing) {
                    after.playing = true;
                    after.stop_requested = false;
                }
                if matches!(transport, RuntimeTransport::Paused | RuntimeTransport::Seek) {
                    after.playing = false;
                    after.stop_requested = false;
                }
                after.route_epoch = after
                    .route_epoch
                    .checked_add(1)
                    .ok_or(SequenceRuntimeError::EpochExhausted(binding.entity))?;
                after.state = AudioPlaybackState::WaitingForAsset;
                replace_audio(&mut result, binding.entity, before, after);
            }
        }
        Ok(result)
    }
    fn apply_camera_events(&self, changes: &mut [CameraChange], frame: &SequenceFrame<'_>) {
        for event in &frame.events {
            for cut in self.camera_cuts.iter().filter(|c| {
                c.sequence == event.sequence
                    && c.path == RuntimePath::new(&event.path)
                    && c.event_id == event.event.id
            }) {
                for change in changes.iter_mut().filter(|change| {
                    self.camera_tracks
                        .iter()
                        .any(|b| b.entity == change.entity && b.cut_group == cut.cut_group)
                }) {
                    change.after_intent.0.active = change.entity == cut.target;
                }
            }
        }
    }
    fn apply_audio_events(
        &self,
        changes: &mut Vec<AudioChange>,
        frame: &SequenceFrame<'_>,
        world: &World,
    ) -> Result<(), SequenceRuntimeError> {
        for event in &frame.events {
            for binding in self.audio_events.iter().filter(|b| {
                b.sequence == event.sequence
                    && b.path == RuntimePath::new(&event.path)
                    && b.event_id == event.event.id
            }) {
                let before = world
                    .get::<AudioPlayer>(binding.entity)
                    .copied()
                    .ok_or(SequenceRuntimeError::MissingEntity(binding.entity))?;
                let mut after = changes
                    .iter()
                    .find(|c| c.entity == binding.entity)
                    .map_or(before, |c| c.after);
                match binding.action {
                    AudioEventAction::Start => {
                        after.playing = true;
                        after.stop_requested = false;
                    }
                    AudioEventAction::Pause => after.playing = false,
                    AudioEventAction::Stop => {
                        after.stop_requested = true;
                        after.playing = false;
                    }
                    AudioEventAction::Seek => {}
                }
                replace_audio(changes, binding.entity, before, after);
            }
        }
        Ok(())
    }
}

fn channel<'a>(
    tracks: &'a BTreeMap<TrackId, Vec<f64>>,
    id: TrackId,
    arity: usize,
) -> Result<&'a [f64], SequenceRuntimeError> {
    let values = tracks
        .get(&id)
        .ok_or(SequenceRuntimeError::InvalidChannel(id))?;
    if values.len() != arity || values.iter().any(|value| !value.is_finite()) {
        return Err(SequenceRuntimeError::InvalidChannel(id));
    }
    Ok(values)
}
fn replace_audio(
    changes: &mut Vec<AudioChange>,
    entity: Entity,
    before: AudioPlayer,
    after: AudioPlayer,
) {
    if let Some(change) = changes.iter_mut().find(|change| change.entity == entity) {
        change.after = after;
    } else {
        changes.push(AudioChange {
            entity,
            before,
            after,
        });
    }
}
