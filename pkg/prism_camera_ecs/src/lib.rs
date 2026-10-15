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
use prism_app::{App, EngineClocks, Plugin, PostUpdate};
use prism_camera::{
    derive_view, publish_view, CameraError, CameraIntent, CameraPose, ViewData, ViewHandle,
    ViewHistory,
};
#[cfg(any(feature = "director", feature = "rig"))]
use prism_ecs::prelude::IntoSystemConfigs;
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
    /// Derives immutable views and publishes them to the view registry.
    Publication,
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
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            frame: 0,
            target_size: [1, 1],
            dt_seconds: 1.0 / 60.0,
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
    mut cameras: Query<(&mut CameraPoseComponent, &mut CameraRigComponent)>,
    frame: Res<CameraFrame>,
    scene_query: Option<Res<CameraSceneQueryResource>>,
    target_query: Option<Res<CameraTargetQueryResource>>,
) {
    let frame_dt = frame.effective_dt();
    let scene = scene_query
        .as_deref()
        .and_then(|resource| {
            resource
                .query
                .as_deref()
                .map(|query| query as &dyn rig::SceneQuery)
        });
    let targets = target_query
        .as_deref()
        .and_then(|resource| {
            resource
                .query
                .as_deref()
                .map(|query| query as &dyn rig::TargetQuery)
        });
    for (mut pose, mut rig_state) in cameras.iter_mut() {
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
pub fn prepare_camera_frame(mut frame: ResMut<CameraFrame>, clocks: Option<Res<EngineClocks>>) {
    if let Some(clocks) = clocks {
        frame.dt_seconds = clocks.delta_secs();
    }
}

/// Derives and publishes every camera entity for the current frame.
pub fn publish_camera_views(
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
) {
    registry.begin_frame();
    published.clear();
    diagnostics.clear();
    let mut active_entities = Vec::new();

    let frame_number = frame.frame;
    let target_size = frame.target_size;
    for (entity, intent, pose, mut history) in cameras.iter_mut() {
        if !intent.0.active {
            history.0.clear();
            registry.remove(entity);
            published.remove(entity);
            continue;
        }
        active_entities.push(entity);

        match derive_view(intent.0, pose.0, frame_number, target_size, &mut history.0) {
            Ok(mut data) => {
                if let Err(error) = registry.publish(entity, &mut data, intent.0.importance) {
                    diagnostics.push(CameraViewError { entity, error });
                    history.0.clear();
                    registry.remove(entity);
                } else {
                    let view = PublishedView { entity, data };
                    published.remember_valid(view);
                    published.push(view);
                }
            }
            Err(error) => {
                diagnostics.push(CameraViewError { entity, error });
                if let Some(mut previous) = published.previous(entity) {
                    previous.data.status = prism_camera::ViewStatus::Invalid(error.code);
                    if registry
                        .publish(entity, &mut previous.data, intent.0.importance)
                        .is_ok()
                    {
                        published.push(previous);
                    } else {
                        history.0.clear();
                        registry.remove(entity);
                        published.remove(entity);
                    }
                } else {
                    history.0.clear();
                    registry.remove(entity);
                    published.remove(entity);
                }
            }
        }
    }
    registry.finish_frame();
    published.sort_deterministically();
    published
        .last_valid
        .retain(|previous| active_entities.contains(&previous.entity));
    frame.frame = frame.frame.wrapping_add(1);
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
        app.insert_resource(CameraFrame::new(self.target_size))
            .init_resource::<CameraViewRegistry>()
            .init_resource::<PublishedViews>()
            .init_resource::<CameraDiagnostics>();
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::App;
    use prism_camera::{CameraErrorCode, ViewKind};

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
