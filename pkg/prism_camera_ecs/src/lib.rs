//! ECS and App assembly for [`prism_camera`].
//!
//! The mathematical crate intentionally does not own an ECS or an application
//! lifecycle. This adapter supplies that seam without adding gameplay policy:
//! entities carry camera intent, pose, and temporal history; one PostUpdate
//! system derives immutable [`ViewData`] snapshots and publishes them through
//! the engine-neutral view registry.

#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "rig")]
use alloc::boxed::Box;
use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use prism_app::TaskPoolResource;
use prism_app::{App, EngineClocks, Plugin, PostUpdate};
use prism_camera::{
    derive_view, publish_view, CameraError, CameraIntent, CameraPose, ViewData, ViewHandle,
    ViewHistory,
};
#[cfg(feature = "cutscene")]
use prism_ecs::event::{Event, Events};
use prism_ecs::prelude::{
    Bundle, Component, Entity, IntoSystemConfigs, Query, Res, ResMut, Resource, SystemSet,
};
use prism_render_architecture::view_family::{ViewImportance, ViewRegistry};

#[cfg(feature = "rig")]
use prism_math::Vec2;

#[cfg(any(feature = "director", feature = "rig"))]
use prism_camera_rig as rig;

#[cfg(any(feature = "director", feature = "rig"))]
pub use prism_camera_rig;

/// Ordered anchors for the camera runtime's PostUpdate pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, SystemSet)]
pub enum CameraSystemSet {
    /// Clears per-frame diagnostics before runtime evaluation.
    Prepare,
    /// Selects and blends virtual cameras into output poses.
    Director,
    /// Normalizes application input into controller commands.
    Control,
    /// Resolves body, aim, controllers, noise, and impulses into output poses.
    Rig,
    /// Applies fixed-tick authored camera tracks and publishes crossed events.
    #[cfg(feature = "cutscene")]
    Cutscene,
    /// Derives immutable views and publishes them to the view registry.
    Publication,
    /// Computes editor-facing composition guides from published view snapshots.
    #[cfg(feature = "composition_aid")]
    Composition,
}

#[cfg(feature = "cutscene")]
/// One authored track occurrence published from a camera playback instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CameraTrackEvent {
    /// Generational camera entity that owns this playback instance.
    pub camera: Entity,
    /// Playback generation; changes on explicit seek/restart.
    pub playback_epoch: u64,
    /// Authored keyframe tick, unique within one validated track.
    pub tick: u64,
    /// Application-defined event payload key.
    pub event_key: u64,
}

#[cfg(feature = "cutscene")]
impl Event for CameraTrackEvent {}

#[cfg(feature = "cutscene")]
/// Per-entity, fixed-tick validated track playback state.
#[derive(Clone, Debug, Component)]
pub struct CameraTrackPlayerComponent {
    /// Immutable validated track sampled by this playback instance.
    pub track: prism_camera::ValidatedCameraTrack,
    /// Current authored fixed tick. Callers advance it only while playing.
    pub tick: u64,
    /// Stable generation used to distinguish replayed event occurrences.
    pub playback_epoch: u64,
    /// Whether the component currently drives the owning camera pose.
    pub playing: bool,
    /// Latest deterministic lens/exposure result sampled by the Cutscene set.
    pub sample: prism_camera::CameraTrackSample,
    /// Optional shutter intent consumed by the next immutable render snapshot.
    pub shutter: Option<prism_camera::Shutter>,
    cursor: prism_camera::CameraTrackEventCursor,
    occurrences: Vec<prism_camera::CameraTrackEventOccurrence>,
}

#[cfg(feature = "cutscene")]
impl CameraTrackPlayerComponent {
    /// Creates a player at an initial tick without dispatching earlier events.
    #[must_use]
    pub fn new(track: prism_camera::ValidatedCameraTrack, tick: u64) -> Self {
        let fallback = prism_camera::CameraTrackSample {
            pose: CameraPose::default(),
            focal_length_mm: 50.0,
            exposure: prism_camera::Exposure::default(),
            event_key: None,
            active: true,
            handoff_to_previous: false,
        };
        let sample = track.sample_state(tick, fallback);
        Self {
            track,
            tick,
            playback_epoch: 0,
            playing: true,
            sample,
            shutter: None,
            cursor: prism_camera::CameraTrackEventCursor::default(),
            occurrences: Vec::new(),
        }
    }

    /// Seeks without replaying the interval between the previous and new tick.
    /// The next update establishes a new baseline under a fresh event identity.
    pub fn seek(&mut self, tick: u64) {
        self.tick = tick;
        self.playback_epoch = self.playback_epoch.wrapping_add(1);
        self.cursor = prism_camera::CameraTrackEventCursor::default();
        self.occurrences.clear();
    }
}

#[cfg(feature = "composition_aid")]
/// Optional per-camera composition-guide configuration.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct CompositionGuideComponent(pub prism_camera::CompositionGuide);

#[cfg(feature = "composition_aid")]
/// One entity-owned collection of normalized, jitter-independent guide lines.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraCompositionGizmos {
    /// Camera entity that owns the guide configuration.
    pub entity: Entity,
    /// Guide lines in normalized viewport coordinates.
    pub gizmos: Vec<prism_camera::CompositionGizmo>,
    /// Number of invalid-view or conversion conditions encountered.
    pub error_count: u32,
}

#[cfg(feature = "composition_aid")]
/// Immutable composition-gizmo output produced after view publication.
#[derive(Debug, Default, Resource)]
pub struct PublishedCompositionGizmos {
    entries: Vec<CameraCompositionGizmos>,
}

#[cfg(feature = "composition_aid")]
impl PublishedCompositionGizmos {
    /// Iterates outputs in the same deterministic `(order, entity)` order as published views.
    pub fn iter(&self) -> impl Iterator<Item = &CameraCompositionGizmos> {
        self.entries.iter()
    }
}

#[cfg(feature = "view_lod")]
/// Optional per-camera update and resolution budget.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct CameraLodComponent(pub prism_camera::ViewLod);

#[cfg(feature = "view_lod")]
/// Published result of the camera-side LOD scheduler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraLodStatus {
    /// ECS entity that owns the camera view.
    pub entity: Entity,
    /// Whether the camera was derived this frame instead of reusing history.
    pub updated: bool,
    /// Resolution scale requested by the camera policy.
    pub resolution_scale: f32,
    /// Number of history frames the downstream temporal consumer should retain.
    pub history_frames: u32,
}

#[cfg(feature = "view_lod")]
/// Per-frame LOD decisions in published-view order.
#[derive(Debug, Default, Resource)]
pub struct PublishedCameraLod {
    entries: Vec<CameraLodStatus>,
}

#[cfg(feature = "view_lod")]
impl PublishedCameraLod {
    /// Iterates the current frame's deterministic LOD decisions.
    pub fn iter(&self) -> impl Iterator<Item = &CameraLodStatus> {
        self.entries.iter()
    }
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Copy, Debug)]
struct CameraLodState {
    entity: Entity,
    last_update_frame: u64,
    last_view: PublishedView,
    last_intent: CameraIntent,
    last_policy: prism_camera::ViewLod,
    last_target_size: [u32; 2],
}

#[cfg(feature = "view_lod")]
#[derive(Debug, Default, Resource)]
struct CameraLodRuntime {
    states: Vec<CameraLodState>,
    active_entities: Vec<Entity>,
}

#[cfg(feature = "view_lod")]
impl CameraLodRuntime {
    fn state(&self, entity: Entity) -> Option<CameraLodState> {
        self.states
            .binary_search_by_key(&entity, |state| state.entity)
            .ok()
            .map(|index| self.states[index])
    }

    fn record(
        &mut self,
        entity: Entity,
        frame: u64,
        view: PublishedView,
        intent: CameraIntent,
        policy: prism_camera::ViewLod,
        target_size: [u32; 2],
    ) {
        match self
            .states
            .binary_search_by_key(&entity, |state| state.entity)
        {
            Ok(index) => {
                self.states[index] = CameraLodState {
                    entity,
                    last_update_frame: frame,
                    last_view: view,
                    last_intent: intent,
                    last_policy: policy,
                    last_target_size: target_size,
                }
            }
            Err(index) => self.states.insert(
                index,
                CameraLodState {
                    entity,
                    last_update_frame: frame,
                    last_view: view,
                    last_intent: intent,
                    last_policy: policy,
                    last_target_size: target_size,
                },
            ),
        }
    }
}

#[cfg(feature = "rig")]
/// Optional physics/query provider used by camera rigs.
#[derive(Default, Resource)]
pub struct CameraSceneQueryResource {
    /// The application-owned scene query implementation.
    pub query: Option<Box<dyn rig::SceneQuery + Send + Sync>>,
}

#[cfg(feature = "rig")]
/// Optional target snapshot provider used by targeted and grouped rigs.
#[derive(Default, Resource)]
pub struct CameraTargetQueryResource {
    /// The application-owned target query implementation.
    pub query: Option<Box<dyn rig::TargetQuery + Send + Sync>>,
}

#[cfg(feature = "director")]
/// ECS-owned virtual-camera candidate. The mathematical candidate remains in
/// `prism_camera_rig`; this wrapper only gives it entity ownership.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct VirtualCameraComponent(pub rig::VirtualCamera);

#[cfg(feature = "director")]
/// ECS director state for one output camera.
#[derive(Clone, Debug, Component)]
pub struct CameraBrainComponent {
    /// Candidate entities, evaluated in this explicit order before the
    /// director applies its stable priority/key ordering.
    pub candidates: Vec<Entity>,
    pub brain: rig::CameraBrain,
    pub default_pose: CameraPose,
    /// Optional fixed override. `NaN` means use [`CameraFrame::dt_seconds`].
    pub dt_seconds: f32,
    scratch: Vec<rig::VirtualCamera>,
}

#[cfg(feature = "director")]
impl Default for CameraBrainComponent {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
            brain: rig::CameraBrain::default(),
            default_pose: CameraPose::default(),
            dt_seconds: f32::NAN,
            scratch: Vec::new(),
        }
    }
}

#[cfg(feature = "director")]
impl CameraBrainComponent {
    pub const MAX_CANDIDATES: usize = 64;

    /// Replaces the candidate list while enforcing the bounded hot-path
    /// contract. The first 64 entities are retained deterministically.
    pub fn set_candidates<I>(&mut self, candidates: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        self.candidates.clear();
        self.candidates
            .extend(candidates.into_iter().take(Self::MAX_CANDIDATES));
    }
}

#[cfg(feature = "rig")]
/// ECS-owned body/aim rig. Physics queries remain an injected concern of the
/// pure `SceneQuery` API; this component handles the deterministic pose path.
#[derive(Clone, Debug, Component)]
pub struct CameraRigComponent {
    pub body: rig::BodyRig,
    pub aim: rig::AimRig,
    pub world_origin: prism_math::DVec3,
    pub fallback: CameraPose,
    pub noise: Option<rig::CameraNoise>,
    pub impulse: Option<rig::Impulse>,
    pub time_seconds: f32,
    /// Previous resolved spring-arm distance; `NaN` means initialize from the first query.
    pub spring_distance: f32,
    /// Optional fixed override. `NaN` means use [`CameraFrame::dt_seconds`].
    pub dt_seconds: f32,
}

#[cfg(feature = "rig")]
/// Controller mode driven by an application-fed normalized camera command.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraControllerMode {
    /// Free-fly controller for debug and DCC-style cameras.
    Fly(rig::FlyController),
    /// Orbit controller around a target snapshot.
    Orbit(rig::OrbitController),
    /// Local-plane pan controller for map and editor cameras.
    Pan(rig::PanController),
}

#[cfg(feature = "rig")]
/// ECS-owned controller input seam. Platform events are converted by the
/// application into [`rig::CameraCommand`] before this system runs.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct CameraControllerComponent {
    /// Command accumulated for the current frame.
    pub command: rig::CameraCommand,
    /// Device-independent normalization and sensitivity policy.
    pub input_map: rig::CameraInputMap,
    /// Active controller implementation.
    pub mode: CameraControllerMode,
}

#[cfg(feature = "rig")]
impl Default for CameraControllerComponent {
    fn default() -> Self {
        Self {
            command: rig::CameraCommand::default(),
            input_map: rig::CameraInputMap::default(),
            mode: CameraControllerMode::Fly(rig::FlyController {
                speed: 5.0,
                ..Default::default()
            }),
        }
    }
}

#[cfg(feature = "rig")]
impl Default for CameraRigComponent {
    fn default() -> Self {
        Self {
            body: rig::BodyRig::default(),
            aim: rig::AimRig::default(),
            world_origin: prism_math::DVec3::ZERO,
            fallback: CameraPose::default(),
            noise: None,
            impulse: None,
            time_seconds: 0.0,
            spring_distance: f32::NAN,
            dt_seconds: f32::NAN,
        }
    }
}

/// ECS component carrying the engine-neutral camera intent.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct CameraIntentComponent(pub CameraIntent);

impl From<CameraIntent> for CameraIntentComponent {
    fn from(value: CameraIntent) -> Self {
        Self(value)
    }
}

/// ECS component carrying the world-space pose consumed by view derivation.
#[derive(Clone, Copy, Debug, PartialEq, Component)]
pub struct CameraPoseComponent(pub CameraPose);

impl From<CameraPose> for CameraPoseComponent {
    fn from(value: CameraPose) -> Self {
        Self(value)
    }
}

/// ECS component owning temporal state for one camera entity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
pub struct ViewHistoryComponent(pub ViewHistory);

/// A camera entity bundle for the standard intent -> pose -> history path.
#[derive(Bundle)]
pub struct CameraBundle {
    /// Intent written by gameplay, editor, or a camera director.
    pub intent: CameraIntentComponent,
    /// World-space pose written by a rig or external pose provider.
    pub pose: CameraPoseComponent,
    /// Temporal state retained between publication frames.
    pub history: ViewHistoryComponent,
}

impl Default for CameraBundle {
    fn default() -> Self {
        Self {
            intent: CameraIntentComponent(CameraIntent::default()),
            pose: CameraPoseComponent(CameraPose::default()),
            history: ViewHistoryComponent::default(),
        }
    }
}

/// Per-frame camera input owned by the application layer.
#[derive(Clone, Copy, Debug, PartialEq, Resource)]
pub struct CameraFrame {
    /// Monotonic frame number passed to jitter and temporal derivation.
    pub frame: u64,
    /// Target extent used to resolve the camera viewport.
    pub target_size: [u32; 2],
    /// Variable-step duration used by Director, Rig, and controller systems.
    pub dt_seconds: f32,
    /// Nominal frame rate used by frame-budgeted camera policies such as LOD.
    pub frame_rate_hz: f32,
}

/// Workload policy for view derivation.
///
/// A camera view is a small, fixed-cost job. Dispatching a handful of these
/// jobs through a general work-stealing pool costs more than evaluating them
/// on the publication thread, so the adapter only fans out a batch after the
/// work is large enough to amortize the fork/join barrier. Applications with
/// unusually expensive projection extensions may lower the thresholds by
/// inserting this resource before [`CameraPlugin`] is built.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy, Debug, PartialEq, Resource)]
pub struct CameraDerivePolicy {
    /// Minimum active-view count before pool dispatch is considered.
    pub parallel_min_views: usize,
    /// Minimum views assigned per worker before pool dispatch is considered.
    pub min_views_per_worker: usize,
}

#[cfg(feature = "parallel")]
impl Default for CameraDerivePolicy {
    fn default() -> Self {
        Self {
            parallel_min_views: 1024,
            min_views_per_worker: 64,
        }
    }
}

#[cfg(feature = "parallel")]
impl CameraDerivePolicy {
    fn allows_parallel(self, view_count: usize, worker_count: usize) -> bool {
        let worker_count = worker_count.max(1);
        let min_views = self
            .parallel_min_views
            .max(self.min_views_per_worker.saturating_mul(worker_count))
            .max(1);
        view_count >= min_views
    }
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            frame: 0,
            target_size: [1, 1],
            dt_seconds: 1.0 / 60.0,
            frame_rate_hz: 60.0,
        }
    }
}

impl CameraFrame {
    /// Creates a frame context with a caller-provided target extent.
    #[must_use]
    pub const fn new(target_size: [u32; 2]) -> Self {
        Self {
            frame: 0,
            target_size,
            dt_seconds: 1.0 / 60.0,
            frame_rate_hz: 60.0,
        }
    }
}

impl CameraFrame {
    /// Returns a finite, non-negative frame delta for runtime systems.
    #[must_use]
    pub fn effective_dt(self) -> f32 {
        if self.dt_seconds.is_finite() {
            self.dt_seconds.max(0.0)
        } else {
            1.0 / 60.0
        }
    }

    /// Returns a finite positive rate for frame-budgeted camera policies.
    #[must_use]
    pub fn effective_frame_rate(self) -> f32 {
        if self.frame_rate_hz.is_finite() && self.frame_rate_hz > 0.0 {
            self.frame_rate_hz.clamp(1.0, 1_000.0)
        } else {
            60.0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CameraViewBinding {
    entity: Entity,
    handle: ViewHandle,
}

/// The ECS-owned session registry that maps camera entities to stable views.
///
/// The underlying generational registry remains the authoritative handle
/// store. The adapter adds entity ownership and frame reconciliation so removed
/// entities cannot leak live view slots across frames.
#[derive(Debug, Default, Resource)]
pub struct CameraViewRegistry {
    registry: ViewRegistry,
    bindings: Vec<CameraViewBinding>,
    seen: Vec<Entity>,
}

impl CameraViewRegistry {
    /// Returns the number of currently published camera views.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Returns the engine-neutral registry for read-only aggregation queries.
    #[must_use]
    pub fn registry(&self) -> &ViewRegistry {
        &self.registry
    }

    fn begin_frame(&mut self) {
        self.seen.clear();
    }

    fn binding(&self, entity: Entity) -> Option<CameraViewBinding> {
        self.bindings
            .binary_search_by_key(&entity, |binding| binding.entity)
            .ok()
            .map(|index| self.bindings[index])
    }

    fn remove(&mut self, entity: Entity) {
        if let Ok(index) = self
            .bindings
            .binary_search_by_key(&entity, |binding| binding.entity)
        {
            let binding = self.bindings.remove(index);
            self.registry.remove(binding.handle);
        }
    }

    fn publish(
        &mut self,
        entity: Entity,
        data: &mut ViewData,
        importance: ViewImportance,
    ) -> Result<(), CameraError> {
        self.seen.push(entity);
        let mut handle = self
            .binding(entity)
            .map_or(ViewHandle::INVALID, |binding| binding.handle);

        match publish_view(&mut self.registry, &mut handle, data, importance) {
            Ok(()) => {}
            Err(error) if error.code == prism_camera::CameraErrorCode::StaleViewHandle => {
                self.remove(entity);
                handle = ViewHandle::INVALID;
                publish_view(&mut self.registry, &mut handle, data, importance)?;
            }
            Err(error) => return Err(error),
        }

        match self
            .bindings
            .binary_search_by_key(&entity, |binding| binding.entity)
        {
            Ok(index) => self.bindings[index].handle = handle,
            Err(index) => self
                .bindings
                .insert(index, CameraViewBinding { entity, handle }),
        }
        Ok(())
    }

    fn finish_frame(&mut self) {
        self.seen.sort_unstable();
        let registry = &mut self.registry;
        self.bindings.retain(|binding| {
            if self.seen.binary_search(&binding.entity).is_ok() {
                true
            } else {
                registry.remove(binding.handle);
                false
            }
        });
    }
}

/// One successfully published view and its ECS owner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedView {
    /// ECS entity that owns the camera intent.
    pub entity: Entity,
    /// Immutable view snapshot consumed by extraction/render code.
    pub data: ViewData,
}

/// Per-frame immutable camera output consumed by extraction/render code.
#[derive(Debug, Default, Resource)]
pub struct PublishedViews {
    views: Vec<PublishedView>,
    last_valid: Vec<PublishedView>,
}

/// Optional per-view cinematic parameters sampled before camera publication.
#[derive(Clone, Debug, PartialEq)]
pub struct PublishedCameraRenderParams {
    /// Generational camera entity owning these parameters.
    pub entity: Entity,
    /// Physical focal length in millimetres; `None` keeps the camera lens intent.
    pub focal_length_mm: Option<f32>,
    /// Authored exposure result; `None` keeps the camera intent exposure.
    pub exposure: Option<prism_camera::Exposure>,
    /// Sanitized shutter result for consistent exposure and motion-blur units.
    #[cfg(feature = "cutscene")]
    pub shutter: Option<prism_camera::ShutterSnapshot>,
    /// Normalized authored post stack; `None` selects the renderer default stack.
    pub post_stack: Option<prism_camera::PostStack>,
}

/// Same-frame immutable camera render parameters, sorted by `(order, entity)`.
#[derive(Debug, Default, Resource)]
pub struct PublishedCameraRenderParamsSet {
    entries: Vec<PublishedCameraRenderParams>,
}

impl PublishedCameraRenderParamsSet {
    /// Iterates per-camera parameters in deterministic publication order.
    pub fn iter(&self) -> impl Iterator<Item = &PublishedCameraRenderParams> {
        self.entries.iter()
    }

    /// Finds a parameter snapshot by full generational camera entity.
    pub fn get(&self, entity: Entity) -> Option<&PublishedCameraRenderParams> {
        self.entries.iter().find(|entry| entry.entity == entity)
    }
}

impl PublishedViews {
    /// Returns the views in deterministic `(order, entity)` order.
    #[must_use]
    pub fn iter(&self) -> impl Iterator<Item = &PublishedView> {
        self.views.iter()
    }

    /// Returns the number of valid views published this frame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Returns whether no valid camera view was published.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    fn clear(&mut self) {
        self.views.clear();
    }

    fn push(&mut self, view: PublishedView) {
        self.views.push(view);
    }

    fn remember_valid(&mut self, view: PublishedView) {
        if let Some(previous) = self
            .last_valid
            .iter_mut()
            .find(|previous| previous.entity == view.entity)
        {
            *previous = view;
        } else {
            self.last_valid.push(view);
        }
    }

    fn previous(&self, entity: Entity) -> Option<PublishedView> {
        self.last_valid
            .iter()
            .copied()
            .find(|view| view.entity == entity)
    }

    fn remove(&mut self, entity: Entity) {
        self.last_valid.retain(|view| view.entity != entity);
    }

    fn sort_deterministically(&mut self) {
        self.views.sort_unstable_by(|a, b| {
            a.data
                .order
                .cmp(&b.data.order)
                .then_with(|| a.entity.cmp(&b.entity))
        });
    }
}

/// A derivation failure associated with one camera entity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraViewError {
    /// ECS entity whose input failed validation or derivation.
    pub entity: Entity,
    /// Stable camera error returned by the math layer.
    pub error: CameraError,
}

/// Bounded per-frame diagnostics for invalid camera inputs.
#[derive(Debug, Default, Resource)]
pub struct CameraDiagnostics {
    errors: Vec<CameraViewError>,
}

/// Rolling camera-stage cost monitor with hysteretic non-primary rig shedding.
/// The sample ring is fixed-size so observing a frame never grows storage.
#[derive(Clone, Debug, Resource)]
pub struct CameraPerformanceBudget {
    samples_ns: [u64; Self::SAMPLE_CAPACITY],
    sample_count: usize,
    next_sample: usize,
    over_budget_frames: u32,
    under_budget_frames: u32,
    degraded: bool,
    last_p99_ns: u64,
    last_view_count: usize,
    frame_started: std::time::Instant,
}

impl CameraPerformanceBudget {
    const SAMPLE_CAPACITY: usize = 64;
    const SHED_AFTER_FRAMES: u32 = 3;
    const RECOVER_AFTER_FRAMES: u32 = 120;
    // These guard the complete ECS camera stage; pure derive_view latency has
    // separate microbenchmark targets and is not observable from this system.

    /// Whether non-primary camera rigs are currently suppressed.
    #[must_use]
    pub const fn is_degraded(&self) -> bool {
        self.degraded
    }

    /// Rolling P99 camera-stage duration in nanoseconds.
    #[must_use]
    pub const fn p99_ns(&self) -> u64 {
        self.last_p99_ns
    }

    /// Active camera count from the latest completed frame.
    #[must_use]
    pub const fn view_count(&self) -> usize {
        self.last_view_count
    }

    fn stage_budget_ns(view_count: usize) -> u64 {
        match view_count {
            0 => u64::MAX,
            1 => 50_000,
            2..=16 => 50_000 + ((view_count - 1) as u64 * 50_000).div_ceil(15),
            count => (100_000_u64.saturating_mul(count as u64)).div_ceil(16),
        }
    }

    fn observe(&mut self, elapsed_ns: u64, view_count: usize) {
        if view_count != self.last_view_count {
            self.samples_ns.fill(0);
            self.sample_count = 0;
            self.next_sample = 0;
            self.over_budget_frames = 0;
            self.under_budget_frames = 0;
        }
        self.samples_ns[self.next_sample] = elapsed_ns;
        self.next_sample = (self.next_sample + 1) % Self::SAMPLE_CAPACITY;
        self.sample_count = (self.sample_count + 1).min(Self::SAMPLE_CAPACITY);
        self.last_view_count = view_count;

        let mut ordered = self.samples_ns;
        ordered[..self.sample_count].sort_unstable();
        let p99_index = ((self.sample_count - 1) * 99).div_ceil(100);
        self.last_p99_ns = ordered[p99_index];
        let over_budget = self.last_p99_ns > Self::stage_budget_ns(view_count);
        if over_budget {
            self.over_budget_frames = self.over_budget_frames.saturating_add(1);
            self.under_budget_frames = 0;
            if self.over_budget_frames >= Self::SHED_AFTER_FRAMES {
                self.degraded = true;
            }
        } else {
            self.under_budget_frames = self.under_budget_frames.saturating_add(1);
            self.over_budget_frames = 0;
            if self.under_budget_frames >= Self::RECOVER_AFTER_FRAMES {
                self.degraded = false;
            }
        }
    }

    fn begin_frame(&mut self) {
        self.frame_started = std::time::Instant::now();
    }

    fn finish_frame(&mut self, view_count: usize) {
        self.observe(
            self.frame_started
                .elapsed()
                .as_nanos()
                .min(u64::MAX as u128) as u64,
            view_count,
        );
    }
}

impl Default for CameraPerformanceBudget {
    fn default() -> Self {
        Self {
            samples_ns: [0; Self::SAMPLE_CAPACITY],
            sample_count: 0,
            next_sample: 0,
            over_budget_frames: 0,
            under_budget_frames: 0,
            degraded: false,
            last_p99_ns: 0,
            last_view_count: 0,
            frame_started: std::time::Instant::now(),
        }
    }
}

#[derive(Clone, Copy)]
struct CameraDeriveInput {
    entity: Entity,
    intent: CameraIntent,
    pose: CameraPose,
    history: ViewHistory,
}

struct CameraDeriveOutput {
    entity: Entity,
    intent: CameraIntent,
    history: ViewHistory,
    result: Result<ViewData, CameraError>,
}

/// Reused input/output storage for the camera publication hot path.
#[derive(Default, Resource)]
struct CameraDeriveScratch {
    inputs: Vec<CameraDeriveInput>,
    outputs: Vec<Option<CameraDeriveOutput>>,
}

impl CameraDiagnostics {
    /// Returns all derivation errors reported during the latest publication.
    #[must_use]
    pub fn errors(&self) -> &[CameraViewError] {
        &self.errors
    }

    fn clear(&mut self) {
        self.errors.clear();
    }

    fn push(&mut self, error: CameraViewError) {
        self.errors.push(error);
    }
}

#[cfg(feature = "director")]
/// Evaluates every ECS-owned director before view derivation.
pub fn apply_camera_directors(
    mut outputs: Query<(
        &mut CameraPoseComponent,
        &mut CameraIntentComponent,
        &mut CameraBrainComponent,
    )>,
    candidates: Query<&VirtualCameraComponent>,
    frame: Res<CameraFrame>,
) {
    for (mut pose, mut intent, mut brain) in outputs.iter_mut() {
        brain.scratch.clear();
        for index in 0..brain.candidates.len() {
            let entity = brain.candidates[index];
            if let Ok(candidate) = candidates.get(entity) {
                brain.scratch.push(candidate.0);
            }
        }
        let selected = rig::CameraBrain::choose(&brain.scratch);
        let dt_seconds = if brain.dt_seconds.is_finite() {
            brain.dt_seconds.max(0.0)
        } else {
            frame.effective_dt()
        };
        let default_pose = brain.default_pose;
        let mut director = brain.brain;
        let output = director.update(&brain.scratch, dt_seconds, default_pose);
        brain.brain = director;
        pose.0 = output;
        intent.0.lens = selected.and_then(|camera| camera.lens);
    }
}

#[cfg(feature = "rig")]
/// Applies normalized controller commands before procedural rig resolution.
pub fn apply_camera_controls(
    mut cameras: Query<(&mut CameraPoseComponent, &mut CameraControllerComponent)>,
    frame: Res<CameraFrame>,
) {
    let dt = frame.effective_dt();
    for (mut pose, mut controller) in cameras.iter_mut() {
        let command = controller.input_map.normalize(controller.command);
        controller.command = rig::CameraCommand::default();
        match &mut controller.mode {
            CameraControllerMode::Fly(fly) => {
                fly.move_axis = command.move_axis;
                pose.0 = fly.apply(pose.0, dt);
            }
            CameraControllerMode::Orbit(orbit) => {
                orbit.yaw += command.look_axis.x;
                orbit.pitch = (orbit.pitch - command.look_axis.y).clamp(-1.56, 1.56);
                orbit.radius = (orbit.radius - command.zoom).max(0.0);
                pose.0 = orbit.pose(pose.0.world_origin);
            }
            CameraControllerMode::Pan(pan) => {
                pan.axis = Vec2::new(command.move_axis.x, command.move_axis.y);
                pose.0 = pan.apply(pose.0, dt);
            }
        }
    }
}

#[cfg(feature = "rig")]
/// Resolves the pure body/aim path and applies optional deterministic camera
/// noise and one-shot impulse data before publication.
pub fn apply_camera_rigs(
    mut cameras: Query<(
        &CameraIntentComponent,
        &mut CameraPoseComponent,
        &mut CameraRigComponent,
    )>,
    frame: Res<CameraFrame>,
    budget: Res<CameraPerformanceBudget>,
    scene_query: Option<Res<CameraSceneQueryResource>>,
    target_query: Option<Res<CameraTargetQueryResource>>,
) {
    let frame_dt = frame.effective_dt();
    let scene = scene_query.as_deref().and_then(|resource| {
        resource
            .query
            .as_deref()
            .map(|query| query as &dyn rig::SceneQuery)
    });
    let targets = target_query.as_deref().and_then(|resource| {
        resource
            .query
            .as_deref()
            .map(|query| query as &dyn rig::TargetQuery)
    });
    for (intent, mut pose, mut rig_state) in cameras.iter_mut() {
        if budget.is_degraded()
            && intent.0.kind != prism_render_architecture::view_family::ViewKind::Main
        {
            continue;
        }
        let dt = if rig_state.dt_seconds.is_finite() {
            rig_state.dt_seconds.max(0.0)
        } else {
            frame_dt
        };
        let output = rig::compose_rig_with_queries(
            &rig_state.body,
            rig_state.aim,
            rig_state.world_origin,
            rig::RigResolveInput {
                fallback: rig_state.fallback,
                target_position: rig_state.fallback.position,
                target_rotation: rig_state.fallback.rotation,
                target_velocity: prism_math::DVec3::ZERO,
                group_direction: prism_math::DVec3::Z,
                spring_socket_offset: prism_math::Vec3::ZERO,
                spring_direction: prism_math::Vec3::NEG_Z,
                previous_position: Some(rig_state.fallback.position),
                previous_rotation: Some(rig_state.fallback.rotation),
                previous_spring_distance: rig_state.spring_distance,
                dt,
            },
            targets,
            scene,
        );
        let mut next_pose = output.pose;
        if let Some(distance) = output.spring_distance {
            rig_state.spring_distance = distance;
        }
        if let Some(noise) = rig_state.noise {
            next_pose = noise.sample(next_pose, rig_state.time_seconds);
        }
        if let Some(impulse) = rig_state.impulse {
            let (impulsed, remaining) = impulse.apply(next_pose, dt);
            next_pose = impulsed;
            rig_state.impulse = (remaining.remaining_s > 0.0).then_some(remaining);
        }
        rig_state.time_seconds = (rig_state.time_seconds + dt).max(0.0);
        rig_state.fallback = next_pose;
        pose.0 = next_pose;
    }
}

/// Captures the engine clock before director, controller, and rig systems run.
/// Every camera stage in one PostUpdate therefore observes the same delta.
pub fn prepare_camera_frame(
    mut frame: ResMut<CameraFrame>,
    mut budget: ResMut<CameraPerformanceBudget>,
    clocks: Option<Res<EngineClocks>>,
) {
    budget.begin_frame();
    if let Some(clocks) = clocks {
        frame.dt_seconds = clocks.delta_secs();
    }
}

/// Derives and publishes every camera entity for the current frame.
pub(crate) fn publish_camera_views(
    mut cameras: Query<(
        Entity,
        &CameraIntentComponent,
        &CameraPoseComponent,
        &mut ViewHistoryComponent,
    )>,
    mut frame: ResMut<CameraFrame>,
    mut registry: ResMut<CameraViewRegistry>,
    mut published: ResMut<PublishedViews>,
    mut diagnostics: ResMut<CameraDiagnostics>,
    mut budget: ResMut<CameraPerformanceBudget>,
    mut scratch: ResMut<CameraDeriveScratch>,
    #[cfg(feature = "parallel")] derive_policy: Res<CameraDerivePolicy>,
    #[cfg(feature = "parallel")] task_pool: Option<Res<TaskPoolResource>>,
    #[cfg(feature = "view_lod")] lods: Query<Option<&CameraLodComponent>>,
    #[cfg(feature = "view_lod")] mut lod_runtime: ResMut<CameraLodRuntime>,
    #[cfg(feature = "view_lod")] mut published_lod: ResMut<PublishedCameraLod>,
) {
    registry.begin_frame();
    published.clear();
    diagnostics.clear();
    scratch.inputs.clear();
    #[cfg(feature = "view_lod")]
    published_lod.entries.clear();

    let frame_number = frame.frame;
    let target_size = frame.target_size;
    #[cfg(feature = "view_lod")]
    let frame_rate = frame.effective_frame_rate();
    #[cfg(feature = "view_lod")]
    lod_runtime.active_entities.clear();
    for (entity, intent, pose, mut history) in cameras.iter_mut() {
        if !intent.0.active {
            history.0.clear();
            registry.remove(entity);
            published.remove(entity);
            #[cfg(feature = "view_lod")]
            lod_runtime.states.retain(|state| state.entity != entity);
            continue;
        }
        #[cfg(feature = "view_lod")]
        lod_runtime.active_entities.push(entity);
        #[cfg(feature = "view_lod")]
        if let Some(lod) = lods.get(entity).ok().flatten() {
            let protected = matches!(
                intent.0.kind,
                prism_camera::ViewKind::Main
                    | prism_camera::ViewKind::StereoEye
                    | prism_camera::ViewKind::Reflection
            );
            if let Some(state) = lod_runtime.state(entity) {
                if !protected
                    && state.last_intent == intent.0
                    && state.last_policy == lod.0
                    && state.last_target_size == target_size
                    && !lod
                        .0
                        .should_update(frame_number, state.last_update_frame, frame_rate)
                {
                    let mut reused = state.last_view;
                    if registry
                        .publish(entity, &mut reused.data, intent.0.importance)
                        .is_ok()
                    {
                        published.push(reused);
                        published.remember_valid(reused);
                        published_lod.entries.push(CameraLodStatus {
                            entity,
                            updated: false,
                            resolution_scale: lod.0.resolution_scale(),
                            history_frames: lod.0.effective_history_frames(),
                        });
                        continue;
                    }
                    lod_runtime
                        .states
                        .retain(|candidate| candidate.entity != entity);
                }
            }
        }
        scratch.inputs.push(CameraDeriveInput {
            entity,
            intent: intent.0,
            pose: pose.0,
            history: history.0,
        });
    }
    scratch.inputs.sort_unstable_by_key(|input| input.entity);
    let input_count = scratch.inputs.len();
    scratch.outputs.resize_with(input_count, || None);
    for output in &mut scratch.outputs[..input_count] {
        *output = None;
    }
    let CameraDeriveScratch { inputs, outputs } = &mut *scratch;

    #[cfg(feature = "parallel")]
    if let Some(task_pool) =
        task_pool.filter(|pool| derive_policy.allows_parallel(input_count, pool.0.worker_count()))
    {
        task_pool
            .0
            .par_iter_mut(&mut outputs[..input_count])
            .with_min_len(4)
            .enumerate_for_each(|index, output| {
                let input = inputs[index];
                let mut history = input.history;
                let result = derive_view(
                    input.intent,
                    input.pose,
                    frame_number,
                    target_size,
                    &mut history,
                );
                *output = Some(CameraDeriveOutput {
                    entity: input.entity,
                    intent: input.intent,
                    history,
                    result,
                });
            });
    } else {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            let mut history = input.history;
            let result = derive_view(
                input.intent,
                input.pose,
                frame_number,
                target_size,
                &mut history,
            );
            *output = Some(CameraDeriveOutput {
                entity: input.entity,
                intent: input.intent,
                history,
                result,
            });
        }
    }
    #[cfg(not(feature = "parallel"))]
    for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
        let mut history = input.history;
        let result = derive_view(
            input.intent,
            input.pose,
            frame_number,
            target_size,
            &mut history,
        );
        *output = Some(CameraDeriveOutput {
            entity: input.entity,
            intent: input.intent,
            history,
            result,
        });
    }
    for output in &mut outputs[..input_count] {
        let CameraDeriveOutput {
            entity,
            intent,
            mut history,
            result,
        } = output.take().expect("every camera derive slot is filled");
        if let Ok((_, _, _, mut history_component)) = cameras.get_mut(entity) {
            history_component.0 = history;
        }
        match result {
            Ok(mut data) => {
                if let Err(error) = registry.publish(entity, &mut data, intent.importance) {
                    diagnostics.push(CameraViewError { entity, error });
                    if let Ok((_, _, _, mut history_component)) = cameras.get_mut(entity) {
                        history_component.0.clear();
                    }
                    registry.remove(entity);
                } else {
                    let view = PublishedView { entity, data };
                    published.remember_valid(view);
                    published.push(view);
                    #[cfg(feature = "view_lod")]
                    if let Some(lod) = lods.get(entity).ok().flatten() {
                        lod_runtime.record(entity, frame_number, view, intent, lod.0, target_size);
                        published_lod.entries.push(CameraLodStatus {
                            entity,
                            updated: true,
                            resolution_scale: lod.0.resolution_scale(),
                            history_frames: lod.0.effective_history_frames(),
                        });
                    }
                }
            }
            Err(error) => {
                diagnostics.push(CameraViewError { entity, error });
                if let Some(mut previous) = published.previous(entity) {
                    previous.data.status = prism_camera::ViewStatus::Invalid(error.code);
                    if registry
                        .publish(entity, &mut previous.data, intent.importance)
                        .is_ok()
                    {
                        published.push(previous);
                    } else {
                        history.clear();
                        if let Ok((_, _, _, mut history_component)) = cameras.get_mut(entity) {
                            history_component.0.clear();
                        }
                        registry.remove(entity);
                        published.remove(entity);
                    }
                } else {
                    history.clear();
                    if let Ok((_, _, _, mut history_component)) = cameras.get_mut(entity) {
                        history_component.0.clear();
                    }
                    registry.remove(entity);
                    published.remove(entity);
                }
            }
        }
    }
    registry.finish_frame();
    published.sort_deterministically();
    #[cfg(not(feature = "view_lod"))]
    published.last_valid.retain(|previous| {
        inputs
            .binary_search_by_key(&previous.entity, |input| input.entity)
            .is_ok()
    });
    #[cfg(feature = "view_lod")]
    {
        lod_runtime.active_entities.sort_unstable();
        let CameraLodRuntime {
            states,
            active_entities,
        } = &mut *lod_runtime;
        states.retain(|state| active_entities.binary_search(&state.entity).is_ok());
        published
            .last_valid
            .retain(|previous| active_entities.binary_search(&previous.entity).is_ok());
        published_lod
            .entries
            .sort_unstable_by_key(|entry| entry.entity);
    }
    budget.finish_frame(input_count);
    frame.frame = frame.frame.wrapping_add(1);
}

#[cfg(feature = "composition_aid")]
fn publish_composition_gizmos(
    published: Res<PublishedViews>,
    guides: Query<&CompositionGuideComponent>,
    mut output: ResMut<PublishedCompositionGizmos>,
) {
    let mut output_index = 0;
    for view in published.iter() {
        let Ok(guide) = guides.get(view.entity) else {
            continue;
        };
        if output_index == output.entries.len() {
            output.entries.push(CameraCompositionGizmos {
                entity: view.entity,
                gizmos: Vec::with_capacity(8),
                error_count: 0,
            });
        }
        let entry = &mut output.entries[output_index];
        entry.entity = view.entity;
        entry.error_count = guide.0.compute_into(&view.data, &mut entry.gizmos);
        output_index += 1;
    }
    output.entries.truncate(output_index);
}

/// Installs camera resources and the PostUpdate publication system.
#[derive(Clone, Copy, Debug)]
pub struct CameraPlugin {
    target_size: [u32; 2],
}

impl Default for CameraPlugin {
    fn default() -> Self {
        Self::new([1, 1])
    }
}

impl CameraPlugin {
    /// Creates the plugin with the target extent used by its frame resource.
    #[must_use]
    pub const fn new(target_size: [u32; 2]) -> Self {
        Self { target_size }
    }
}

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "cutscene")]
        app.add_event::<CameraTrackEvent>();
        app.insert_resource(CameraFrame::new(self.target_size))
            .init_resource::<CameraViewRegistry>()
            .init_resource::<PublishedViews>()
            .init_resource::<PublishedCameraRenderParamsSet>()
            .init_resource::<CameraDiagnostics>()
            .init_resource::<CameraPerformanceBudget>()
            .init_resource::<CameraDeriveScratch>();
        #[cfg(feature = "composition_aid")]
        app.init_resource::<PublishedCompositionGizmos>();
        #[cfg(feature = "view_lod")]
        app.init_resource::<CameraLodRuntime>()
            .init_resource::<PublishedCameraLod>();
        #[cfg(feature = "parallel")]
        app.init_resource::<CameraDerivePolicy>();
        #[cfg(feature = "rig")]
        app.init_resource::<CameraSceneQueryResource>()
            .init_resource::<CameraTargetQueryResource>();

        #[cfg(all(feature = "director", feature = "rig"))]
        app.add_systems(
            PostUpdate,
            (
                prepare_camera_frame.in_set(CameraSystemSet::Prepare),
                apply_camera_directors.in_set(CameraSystemSet::Director),
                apply_camera_controls.in_set(CameraSystemSet::Control),
                apply_camera_rigs.in_set(CameraSystemSet::Rig),
                publish_camera_views.in_set(CameraSystemSet::Publication),
            )
                .chain(),
        );
        #[cfg(all(feature = "director", not(feature = "rig")))]
        app.add_systems(
            PostUpdate,
            (
                prepare_camera_frame.in_set(CameraSystemSet::Prepare),
                apply_camera_directors.in_set(CameraSystemSet::Director),
                publish_camera_views.in_set(CameraSystemSet::Publication),
            )
                .chain(),
        );
        #[cfg(all(feature = "rig", not(feature = "director")))]
        app.add_systems(
            PostUpdate,
            (
                prepare_camera_frame.in_set(CameraSystemSet::Prepare),
                apply_camera_controls.in_set(CameraSystemSet::Control),
                apply_camera_rigs.in_set(CameraSystemSet::Rig),
                publish_camera_views.in_set(CameraSystemSet::Publication),
            )
                .chain(),
        );
        #[cfg(not(any(feature = "director", feature = "rig")))]
        app.add_systems(
            PostUpdate,
            (
                prepare_camera_frame.in_set(CameraSystemSet::Prepare),
                publish_camera_views.in_set(CameraSystemSet::Publication),
            )
                .chain(),
        );
        #[cfg(feature = "cutscene")]
        app.add_systems(
            PostUpdate,
            apply_camera_tracks
                .in_set(CameraSystemSet::Cutscene)
                .after(CameraSystemSet::Rig)
                .before(CameraSystemSet::Publication),
        );
        app.add_systems(
            PostUpdate,
            publish_camera_render_params.after(CameraSystemSet::Publication),
        );
        #[cfg(feature = "composition_aid")]
        app.add_systems(
            PostUpdate,
            publish_composition_gizmos
                .in_set(CameraSystemSet::Composition)
                .after(CameraSystemSet::Publication),
        );
    }
}

#[cfg(feature = "cutscene")]
fn apply_camera_tracks(
    mut players: Query<(
        Entity,
        &mut CameraTrackPlayerComponent,
        &mut CameraPoseComponent,
    )>,
    mut events: ResMut<Events<CameraTrackEvent>>,
) {
    for (entity, mut player, mut pose) in players.iter_mut() {
        if !player.playing {
            continue;
        }
        let fallback = prism_camera::CameraTrackSample {
            pose: pose.0,
            focal_length_mm: 50.0,
            exposure: prism_camera::Exposure::default(),
            event_key: None,
            active: true,
            handoff_to_previous: false,
        };
        let sample = player.track.sample_state(player.tick, fallback);
        player.sample = sample;
        if sample.active && !sample.handoff_to_previous {
            pose.0 = sample.pose;
        }
        let tick = player.tick;
        let playback_epoch = player.playback_epoch;
        let CameraTrackPlayerComponent {
            track,
            cursor,
            occurrences,
            ..
        } = &mut *player;
        if cursor.advance(track, tick, occurrences).is_ok() {
            for occurrence in occurrences.drain(..) {
                events.send(CameraTrackEvent {
                    camera: entity,
                    playback_epoch,
                    tick: occurrence.tick,
                    event_key: occurrence.event_key,
                });
            }
        }
    }
}

#[cfg(feature = "cutscene")]
fn publish_camera_render_params(
    players: Query<(Entity, &CameraTrackPlayerComponent)>,
    published: Res<PublishedViews>,
    frame: Res<CameraFrame>,
    mut output: ResMut<PublishedCameraRenderParamsSet>,
) {
    output.entries.clear();
    for view in published.iter() {
        let Ok((_, player)) = players.get(view.entity) else {
            continue;
        };
        if !player.playing {
            continue;
        }
        let sample = player.sample;
        if !sample.active || sample.handoff_to_previous {
            continue;
        }
        let post_stack = player.track.post_stack().cloned();
        let shutter = player.shutter.map(|intent| {
            let valid_angle = intent.shutter_angle_deg.is_finite();
            let valid_fps = intent.fps.is_finite() && intent.fps > 0.0;
            let safe = prism_camera::Shutter {
                shutter_angle_deg: if valid_angle {
                    intent.shutter_angle_deg.clamp(0.0, 360.0)
                } else {
                    prism_camera::Shutter::default().shutter_angle_deg
                },
                fps: frame.effective_frame_rate(),
                aperture_f_stop: if intent.aperture_f_stop.is_finite()
                    && intent.aperture_f_stop > 0.0
                {
                    intent.aperture_f_stop
                } else {
                    prism_camera::Shutter::default().aperture_f_stop
                },
            };
            let status = match (valid_fps, valid_angle) {
                (true, true) => prism_camera::ShutterRuntimeStatus::Applied,
                (false, true) => prism_camera::ShutterRuntimeStatus::InvalidFrameRate,
                (true, false) => prism_camera::ShutterRuntimeStatus::InvalidAngle,
                (false, false) => prism_camera::ShutterRuntimeStatus::InvalidFrameRateAndAngle,
            };
            prism_camera::ShutterSnapshot {
                shutter_angle_deg: safe.shutter_angle_deg,
                fps: safe.fps,
                exposure_seconds: safe.exposure_seconds(),
                exposure: safe.physical_exposure(100.0),
                aperture_f_stop: safe.aperture_f_stop,
                motion_blur_exposure_fraction: safe.shutter_angle_deg / 360.0,
                status,
            }
        });
        output.entries.push(PublishedCameraRenderParams {
            entity: view.entity,
            focal_length_mm: Some(sample.focal_length_mm),
            exposure: Some(sample.exposure),
            #[cfg(feature = "cutscene")]
            shutter,
            post_stack,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::App;
    use prism_camera::{CameraErrorCode, ViewKind};
    #[cfg(feature = "cutscene")]
    use prism_ecs::event::EventCursor;

    #[cfg(feature = "cutscene")]
    fn validated_track() -> prism_camera::ValidatedCameraTrack {
        let mut track = prism_camera::CameraTrack::default();
        for (tick, event_key) in [(2, 7), (4, 7), (6, 9)] {
            assert!(track.push(prism_camera::CameraKeyframe {
                tick,
                pose: CameraPose::default(),
                focal_length_mm: 50.0,
                exposure: prism_camera::Exposure::default(),
                event_key,
            }));
        }
        track.into_validated().unwrap()
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn camera_track_player_publishes_each_crossed_keyframe_occurrence_once() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        let camera = app.world_mut().spawn(CameraBundle::default());
        app.world_mut().insert(
            camera,
            CameraTrackPlayerComponent::new(validated_track(), 0),
        );
        let mut cursor = EventCursor::<CameraTrackEvent>::default();

        app.update();
        {
            let mut player = app
                .world_mut()
                .get_mut::<CameraTrackPlayerComponent>(camera)
                .unwrap();
            player.tick = 5;
        }
        app.update();
        let occurrences: Vec<_> = cursor
            .read(app.world().resource::<Events<CameraTrackEvent>>())
            .copied()
            .collect();
        assert_eq!(
            occurrences,
            [
                CameraTrackEvent {
                    camera,
                    playback_epoch: 0,
                    tick: 2,
                    event_key: 7
                },
                CameraTrackEvent {
                    camera,
                    playback_epoch: 0,
                    tick: 4,
                    event_key: 7
                },
            ]
        );

        app.update();
        assert_eq!(
            cursor
                .read(app.world().resource::<Events<CameraTrackEvent>>())
                .count(),
            0
        );
        app.world_mut()
            .get_mut::<CameraTrackPlayerComponent>(camera)
            .unwrap()
            .tick = 6;
        app.update();
        assert_eq!(
            cursor
                .read(app.world().resource::<Events<CameraTrackEvent>>())
                .copied()
                .collect::<Vec<_>>(),
            [CameraTrackEvent {
                camera,
                playback_epoch: 0,
                tick: 6,
                event_key: 9
            }]
        );
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn camera_track_seek_rebaselines_and_paused_players_do_not_emit() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        let camera = app.world_mut().spawn(CameraBundle::default());
        app.world_mut().insert(
            camera,
            CameraTrackPlayerComponent::new(validated_track(), 0),
        );
        let mut cursor = EventCursor::<CameraTrackEvent>::default();
        app.update();
        app.world_mut()
            .get_mut::<CameraTrackPlayerComponent>(camera)
            .unwrap()
            .seek(5);
        app.update();
        assert_eq!(
            cursor
                .read(app.world().resource::<Events<CameraTrackEvent>>())
                .count(),
            0
        );
        {
            let mut player = app
                .world_mut()
                .get_mut::<CameraTrackPlayerComponent>(camera)
                .unwrap();
            player.playing = false;
            player.tick = 6;
        }
        app.update();
        assert_eq!(
            cursor
                .read(app.world().resource::<Events<CameraTrackEvent>>())
                .count(),
            0
        );
    }

    #[test]
    fn performance_budget_sheds_after_sustained_overrun_and_recovers_with_hysteresis() {
        let mut budget = CameraPerformanceBudget::default();
        for _ in 0..CameraPerformanceBudget::SHED_AFTER_FRAMES {
            budget.observe(101_000, 16);
        }
        assert!(budget.is_degraded());
        assert_eq!(budget.p99_ns(), 101_000);

        // First age out the over-budget samples from the P99 window.
        while budget.p99_ns() > 90_000 {
            budget.observe(90_000, 16);
        }
        for _ in 0..(CameraPerformanceBudget::RECOVER_AFTER_FRAMES - 2) {
            budget.observe(90_000, 16);
        }
        assert!(budget.is_degraded());
        budget.observe(90_000, 16);
        assert!(!budget.is_degraded());
    }

    #[test]
    fn performance_budget_keeps_single_view_and_tolerates_short_spikes() {
        let mut budget = CameraPerformanceBudget::default();
        budget.observe(100_000, 1);
        budget.observe(1_000, 1);
        assert!(!budget.is_degraded());
        assert_eq!(CameraPerformanceBudget::stage_budget_ns(16), 100_000);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn derive_policy_keeps_small_camera_batches_on_the_publication_thread() {
        let policy = CameraDerivePolicy::default();
        assert!(!policy.allows_parallel(16, 8));
        assert!(!policy.allows_parallel(1023, 8));
        assert!(policy.allows_parallel(1024, 8));

        let expensive_view_policy = CameraDerivePolicy {
            parallel_min_views: 8,
            min_views_per_worker: 2,
        };
        assert!(!expensive_view_policy.allows_parallel(7, 4));
        assert!(expensive_view_policy.allows_parallel(8, 4));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_and_serial_publication_produce_identical_stable_views() {
        let mut parallel_app = App::new();
        parallel_app.add_plugins(CameraPlugin::new([1920, 1080]));
        assert!(parallel_app.task_pool().is_some());
        let mut serial_app = App::new();
        serial_app.add_plugins(CameraPlugin::new([1920, 1080]));
        assert!(serial_app
            .world_mut()
            .remove_resource::<TaskPoolResource>()
            .is_some());

        for index in 0..16 {
            let mut intent = CameraIntent {
                order: ((index * 7) % 16) as i32,
                ..Default::default()
            };
            intent.target = match index % 3 {
                0 => prism_camera::RenderTarget::PrimaryWindow,
                1 => prism_camera::RenderTarget::Window((index / 3 + 1) as u64),
                _ => prism_camera::RenderTarget::Texture((index / 3 + 1) as u64),
            };
            let pose = CameraPoseComponent(CameraPose {
                position: DVec3::new(index as f64 * 0.25, 1.0, -2.0),
                ..Default::default()
            });
            parallel_app.world_mut().spawn(CameraBundle {
                intent: CameraIntentComponent(intent),
                pose,
                ..Default::default()
            });
            serial_app.world_mut().spawn(CameraBundle {
                intent: CameraIntentComponent(intent),
                pose,
                ..Default::default()
            });
        }

        for _ in 0..3 {
            parallel_app.update();
            serial_app.update();
            let parallel_views: Vec<_> = parallel_app
                .world()
                .resource::<PublishedViews>()
                .iter()
                .map(|view| (view.data.order, view.data.target, view.data.world_position))
                .collect();
            let serial_views: Vec<_> = serial_app
                .world()
                .resource::<PublishedViews>()
                .iter()
                .map(|view| (view.data.order, view.data.target, view.data.world_position))
                .collect();
            assert_eq!(parallel_views, serial_views);
            assert_eq!(parallel_views.len(), 16);
        }
    }

    #[cfg(any(feature = "director", feature = "rig"))]
    use prism_math::{DQuat, DVec3, Vec3};

    #[cfg(feature = "director")]
    #[test]
    fn director_system_selects_candidate_and_publishes_output_pose() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        let candidate = app
            .world_mut()
            .spawn(VirtualCameraComponent(rig::VirtualCamera {
                key: 7,
                priority: 10,
                pose: CameraPose {
                    position: DVec3::new(3.0, 2.0, 1.0),
                    ..Default::default()
                },
                ..Default::default()
            }));
        let output = app.world_mut().spawn(CameraBundle::default());
        let mut brain = CameraBrainComponent::default();
        brain.set_candidates([candidate]);
        assert!(app.world_mut().insert(output, brain));

        app.update();

        let pose = app.world().get::<CameraPoseComponent>(output).unwrap();
        assert_eq!(pose.0.position, DVec3::new(3.0, 2.0, 1.0));
        assert_eq!(app.world().resource::<PublishedViews>().len(), 1);
    }

    #[cfg(feature = "rig")]
    #[test]
    fn rig_system_composes_body_and_aim_before_publication() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        let output = app.world_mut().spawn(CameraBundle::default());
        assert!(app.world_mut().insert(
            output,
            CameraRigComponent {
                body: rig::BodyRig::Fixed {
                    position: DVec3::new(4.0, 5.0, 6.0),
                },
                aim: rig::AimRig::default(),
                ..Default::default()
            },
        ));

        app.update();

        let pose = app.world().get::<CameraPoseComponent>(output).unwrap();
        assert_eq!(pose.0.position, DVec3::new(4.0, 5.0, 6.0));
    }

    #[cfg(feature = "rig")]
    #[test]
    fn rig_system_consumes_injected_target_and_scene_queries() {
        struct Query;

        impl rig::TargetQuery for Query {
            fn resolve_batch(&self, entities: &[rig::Entity], states: &mut [rig::TargetState]) {
                for (entity, state) in entities.iter().zip(states.iter_mut()) {
                    if *entity == 42 {
                        *state = rig::TargetState {
                            valid: true,
                            position: DVec3::ZERO,
                            rotation: DQuat::IDENTITY,
                            velocity: DVec3::ZERO,
                        };
                    }
                }
            }
        }

        impl rig::SceneQuery for Query {
            fn cast_batch(&self, _requests: &[rig::CastRequest], hits: &mut [rig::CastHit]) {
                for hit in hits {
                    hit.hit = true;
                    hit.distance = 1.0;
                    hit.normal = Vec3::Y;
                }
            }
        }

        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        app.world_mut()
            .resource_mut::<CameraSceneQueryResource>()
            .query = Some(Box::new(Query));
        app.world_mut()
            .resource_mut::<CameraTargetQueryResource>()
            .query = Some(Box::new(Query));
        let output = app.world_mut().spawn(CameraBundle::default());
        assert!(app.world_mut().insert(
            output,
            CameraRigComponent {
                body: rig::BodyRig::SpringArmTargeted(rig::SpringArmBody {
                    target: 42,
                    arm: rig::SpringArm {
                        arm_length: 4.0,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                ..Default::default()
            },
        ));

        app.update();

        let pose = app.world().get::<CameraPoseComponent>(output).unwrap();
        assert!((pose.0.position.z + 0.95).abs() < 1.0e-6);
    }

    #[cfg(feature = "rig")]
    #[test]
    fn rig_system_retains_spring_arm_distance_between_frames() {
        struct Query;

        impl rig::SceneQuery for Query {
            fn cast_batch(&self, _requests: &[rig::CastRequest], hits: &mut [rig::CastHit]) {
                for hit in hits {
                    hit.hit = true;
                    hit.distance = 1.0;
                    hit.normal = Vec3::Y;
                }
            }
        }

        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        app.world_mut()
            .resource_mut::<CameraSceneQueryResource>()
            .query = Some(Box::new(Query));
        let output = app.world_mut().spawn(CameraBundle::default());
        assert!(app.world_mut().insert(
            output,
            CameraRigComponent {
                body: rig::BodyRig::SpringArm(rig::SpringArm {
                    arm_length: 4.0,
                    lag_half_life: 0.2,
                    collision_margin: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ));

        app.update();
        let first = app
            .world()
            .get::<CameraRigComponent>(output)
            .unwrap()
            .spring_distance;
        app.update();
        let second = app
            .world()
            .get::<CameraRigComponent>(output)
            .unwrap()
            .spring_distance;
        assert!(first.is_finite());
        assert_eq!(second, first);
    }

    #[test]
    fn app_update_publishes_intent_to_view_data_and_reuses_history() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::new([1920, 1080]));
        let entity = app.world_mut().spawn(CameraBundle::default());

        app.update();
        let first = app.world().resource::<PublishedViews>();
        assert_eq!(first.len(), 1);
        assert_eq!(first.iter().next().unwrap().entity, entity);
        assert_eq!(first.iter().next().unwrap().data.kind, ViewKind::Main);
        assert!(!first.iter().next().unwrap().data.history_valid);
        let first_handle = first.iter().next().unwrap().data.handle;
        assert_eq!(app.world().resource::<CameraViewRegistry>().len(), 1);
        assert_eq!(app.world().resource::<CameraFrame>().frame, 1);

        app.update();
        let second = app.world().resource::<PublishedViews>();
        assert_eq!(second.len(), 1);
        assert!(second.iter().next().unwrap().data.history_valid);
        assert_eq!(second.iter().next().unwrap().data.handle, first_handle);
    }

    #[cfg(feature = "view_lod")]
    #[test]
    fn view_lod_reuses_non_primary_views_and_keeps_primary_views_live() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::new([1920, 1080]));
        let mut capture = CameraBundle::default();
        capture.intent.0.kind = ViewKind::SceneCapture;
        let capture_entity = app.world_mut().spawn((
            capture,
            CameraLodComponent(prism_camera::ViewLod {
                max_update_hz: 30.0,
                min_resolution_scale: 0.5,
                history_frames: 4,
            }),
        ));
        let main_entity = app.world_mut().spawn((
            CameraBundle::default(),
            CameraLodComponent(prism_camera::ViewLod {
                max_update_hz: 1.0,
                min_resolution_scale: 0.75,
                history_frames: 2,
            }),
        ));

        app.update();
        let first_capture = app
            .world()
            .resource::<PublishedViews>()
            .iter()
            .find(|view| view.entity == capture_entity)
            .expect("scene capture is published")
            .data;
        let first_lod = app
            .world()
            .resource::<PublishedCameraLod>()
            .iter()
            .find(|entry| entry.entity == capture_entity)
            .copied()
            .expect("scene capture has a lod decision");
        assert!(first_lod.updated);
        assert_eq!(first_lod.resolution_scale, 0.5);
        assert_eq!(first_lod.history_frames, 4);

        app.world_mut()
            .get_mut::<CameraPoseComponent>(capture_entity)
            .unwrap()
            .0
            .position
            .x = 10.0;
        app.update();
        let second_capture = app
            .world()
            .resource::<PublishedViews>()
            .iter()
            .find(|view| view.entity == capture_entity)
            .expect("scene capture remains published")
            .data;
        let second_lod = app
            .world()
            .resource::<PublishedCameraLod>()
            .iter()
            .find(|entry| entry.entity == capture_entity)
            .copied()
            .expect("scene capture has a reused lod decision");
        assert!(!second_lod.updated);
        assert_eq!(
            second_capture.view_from_world,
            first_capture.view_from_world
        );
        assert!(app
            .world()
            .resource::<PublishedCameraLod>()
            .iter()
            .find(|entry| entry.entity == main_entity)
            .is_some_and(|entry| entry.updated));

        app.update();
        let third_lod = app
            .world()
            .resource::<PublishedCameraLod>()
            .iter()
            .find(|entry| entry.entity == capture_entity)
            .copied()
            .expect("scene capture has a scheduled update");
        assert!(third_lod.updated);
        assert_ne!(
            app.world()
                .resource::<PublishedViews>()
                .iter()
                .find(|view| view.entity == capture_entity)
                .unwrap()
                .data
                .view_from_world,
            first_capture.view_from_world
        );
    }

    #[test]
    fn invalid_target_is_reported_without_publishing() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::new([0, 0]));
        app.world_mut().spawn(CameraBundle::default());

        app.update();

        assert!(app.world().resource::<PublishedViews>().is_empty());
        let errors = app.world().resource::<CameraDiagnostics>().errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].error.code, CameraErrorCode::InvalidTargetSize);
        assert_eq!(app.world().resource::<CameraViewRegistry>().len(), 0);
    }

    #[test]
    fn default_plugin_uses_a_valid_bootstrap_extent() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        app.world_mut().spawn(CameraBundle::default());

        app.update();

        assert_eq!(app.world().resource::<PublishedViews>().len(), 1);
        assert!(app
            .world()
            .resource::<CameraDiagnostics>()
            .errors()
            .is_empty());
    }

    #[cfg(feature = "composition_aid")]
    #[test]
    fn composition_guides_follow_published_view_and_clear_when_camera_is_inactive() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::new([1280, 720]));
        let intent = CameraBundle::default();
        let entity = app.world_mut().spawn((
            intent,
            CompositionGuideComponent(prism_camera::CompositionGuide {
                thirds: true,
                golden_ratio: false,
                title_safe: 0.9,
                action_safe: 0.8,
            }),
        ));

        app.update();

        let output = app.world().resource::<PublishedCompositionGizmos>();
        let entry = output.iter().next().expect("guide output is published");
        assert_eq!(entry.entity, entity);
        assert_eq!(entry.gizmos.len(), 6);
        assert_eq!(entry.error_count, 0);
        let gizmo_buffer = entry.gizmos.as_ptr();
        let gizmo_capacity = entry.gizmos.capacity();

        app.update();
        let entry = app
            .world()
            .resource::<PublishedCompositionGizmos>()
            .iter()
            .next()
            .expect("guide output remains published");
        assert_eq!(entry.gizmos.as_ptr(), gizmo_buffer);
        assert_eq!(entry.gizmos.capacity(), gizmo_capacity);

        app.world_mut()
            .get_mut::<CameraIntentComponent>(entity)
            .unwrap()
            .0
            .active = false;
        app.update();
        assert!(app
            .world()
            .resource::<PublishedCompositionGizmos>()
            .iter()
            .next()
            .is_none());
    }

    #[test]
    fn inactive_camera_releases_its_registered_view() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::new([64, 64]));
        let entity = app.world_mut().spawn(CameraBundle::default());
        app.update();
        assert_eq!(app.world().resource::<CameraViewRegistry>().len(), 1);

        app.world_mut()
            .get_mut::<CameraIntentComponent>(entity)
            .unwrap()
            .0
            .active = false;
        app.update();

        assert!(app.world().resource::<PublishedViews>().is_empty());
        assert_eq!(app.world().resource::<CameraViewRegistry>().len(), 0);
    }

    #[test]
    fn invalid_update_retains_previous_view_with_structured_status() {
        let mut app = App::new();
        app.add_plugins(CameraPlugin::default());
        let entity = app.world_mut().spawn(CameraBundle::default());

        app.update();
        assert_eq!(app.world().resource::<PublishedViews>().len(), 1);

        assert!(app.world_mut().insert(
            entity,
            CameraIntentComponent(CameraIntent {
                viewport: prism_camera::Viewport::from_pixels([0, 0], [0, 1]),
                ..CameraIntent::default()
            }),
        ));
        app.update();

        let published = app.world().resource::<PublishedViews>();
        let view = published.iter().next().expect("previous view is retained");
        assert_eq!(view.entity, entity);
        assert_eq!(
            view.data.status,
            prism_camera::ViewStatus::Invalid(CameraErrorCode::InvalidViewport)
        );
        assert_eq!(app.world().resource::<CameraViewRegistry>().len(), 1);
        assert_eq!(
            app.world().resource::<CameraDiagnostics>().errors().len(),
            1
        );
    }
}

#[cfg(feature = "cutscene")]
#[test]
fn cutscene_lens_exposure_post_stack_and_shutter_publish_with_the_view() {
    let mut app = App::new();
    app.add_plugins(CameraPlugin::new([64, 64]));
    let camera = app.world_mut().spawn(CameraBundle::default());
    let mut track = prism_camera::CameraTrack::default();
    let mut stack = prism_camera::PostStack::default();
    stack.push(prism_camera::PostNode::Vignette(0.2)).unwrap();
    stack.normalize().unwrap();
    track.post_stack = Some(stack.clone());
    assert!(track.push(prism_camera::CameraKeyframe {
        tick: 0,
        pose: CameraPose::default(),
        focal_length_mm: 35.0,
        exposure: prism_camera::Exposure::ManualEv100 { ev100: 1.5 },
        event_key: 0,
    }));
    let mut player = CameraTrackPlayerComponent::new(track.into_validated().unwrap(), 0);
    player.shutter = Some(prism_camera::Shutter {
        shutter_angle_deg: 180.0,
        fps: 30.0,
        aperture_f_stop: 4.0,
    });
    app.world_mut().insert(camera, player);
    app.update();

    let params = app
        .world()
        .resource::<PublishedCameraRenderParamsSet>()
        .get(camera)
        .expect("active published camera has a cinematic parameter snapshot");
    assert_eq!(params.focal_length_mm, Some(35.0));
    assert_eq!(
        params.exposure,
        Some(prism_camera::Exposure::ManualEv100 { ev100: 1.5 })
    );
    assert_eq!(params.post_stack.as_ref(), Some(&stack));
    let shutter = params
        .shutter
        .expect("shutter is converted for the render adapter");
    assert_eq!(
        shutter.fps, 60.0,
        "render tick uses the authoritative frame rate"
    );
    assert_eq!(shutter.motion_blur_exposure_fraction(), 0.5);
    assert!((shutter.exposure_seconds - 1.0 / 120.0).abs() < 1.0e-7);
}

#[cfg(feature = "cutscene")]
#[test]
fn cutscene_shutter_snapshot_reports_invalid_authored_rate_and_sanitizes_aperture() {
    let mut app = App::new();
    app.add_plugins(CameraPlugin::new([64, 64]));
    let camera = app.world_mut().spawn(CameraBundle::default());
    let mut track = prism_camera::CameraTrack::default();
    assert!(track.push(prism_camera::CameraKeyframe {
        tick: 0,
        pose: CameraPose::default(),
        focal_length_mm: 50.0,
        exposure: prism_camera::Exposure::default(),
        event_key: 0,
    }));
    let mut player = CameraTrackPlayerComponent::new(track.into_validated().unwrap(), 0);
    player.shutter = Some(prism_camera::Shutter {
        shutter_angle_deg: f32::NAN,
        fps: 0.0,
        aperture_f_stop: f32::NAN,
    });
    app.world_mut().insert(camera, player);
    app.update();

    let shutter = app
        .world()
        .resource::<PublishedCameraRenderParamsSet>()
        .get(camera)
        .unwrap()
        .shutter
        .unwrap();
    assert_eq!(
        shutter.status,
        prism_camera::ShutterRuntimeStatus::InvalidFrameRateAndAngle
    );
    assert_eq!(shutter.shutter_angle_deg, 180.0);
    assert_eq!(shutter.fps, 60.0);
    assert_eq!(shutter.aperture_f_stop, 2.8);
    assert_eq!(shutter.exposure_seconds, 1.0 / 120.0);
}
