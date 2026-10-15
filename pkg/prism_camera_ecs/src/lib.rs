//! ECS and App assembly for [`prism_camera`].
//!
//! The mathematical crate intentionally does not own an ECS or an application
//! lifecycle. This adapter supplies that seam without adding gameplay policy:
//! entities carry camera intent, pose, and temporal history; one PostUpdate
//! system derives immutable [`ViewData`] snapshots and publishes them through
//! the engine-neutral view registry.

#![forbid(unsafe_code)]

use prism_app::{App, Plugin, PostUpdate};
use prism_camera::{
    derive_view, publish_view, CameraError, CameraIntent, CameraPose, ViewData, ViewHandle,
    ViewHistory,
};
use prism_ecs::prelude::{Bundle, Component, Entity, Query, ResMut, Resource};
use prism_render_architecture::view_family::{ViewImportance, ViewRegistry};

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
}

impl Default for CameraFrame {
    fn default() -> Self {
        Self {
            frame: 0,
            target_size: [1, 1],
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

    let frame_number = frame.frame;
    let target_size = frame.target_size;
    for (entity, intent, pose, mut history) in cameras.iter_mut() {
        if !intent.0.active {
            history.0.clear();
            registry.remove(entity);
            continue;
        }

        match derive_view(intent.0, pose.0, frame_number, target_size, &mut history.0) {
            Ok(mut data) => {
                if let Err(error) = registry.publish(entity, &mut data, intent.0.importance) {
                    diagnostics.push(CameraViewError { entity, error });
                    history.0.clear();
                    registry.remove(entity);
                } else {
                    published.push(PublishedView { entity, data });
                }
            }
            Err(error) => {
                diagnostics.push(CameraViewError { entity, error });
                history.0.clear();
                registry.remove(entity);
            }
        }
    }
    registry.finish_frame();
    published.sort_deterministically();
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
            .init_resource::<CameraDiagnostics>()
            .add_systems(PostUpdate, publish_camera_views);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::App;
    use prism_camera::{CameraErrorCode, ViewKind};

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
}
