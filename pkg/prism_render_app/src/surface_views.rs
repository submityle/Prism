//! Native camera snapshots for surface passes. Call once per extraction, after
//! main-world camera/transform updates. No camera or prerequisite is synthesized.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;

use prism_camera::{ClearMode, RenderTarget};
use prism_camera_ecs::PublishedViews;
use prism_ecs::prelude::{Entity, LifecycleEvent, ObserverContext, Resource, World};
use prism_ecs::world::WorldId;
use prism_math::Mat4;
use prism_transform::GlobalTransform;

use crate::visibility::{PrismCamera, PrismFrustum, ViewFlags};

/// A validated scene camera, with column-major world-to-clip matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceView {
    /// Main-world identity, including the entity generation.
    pub camera: Entity,
    /// Stable destination identity from the published camera intent.
    pub target: RenderTarget,
    /// Target clear behavior from the published camera intent.
    pub clear: ClearMode,
    /// Camera composition order; ties are resolved by generational entity.
    pub order: i32,
    /// The camera's supplied matrix, without applying its pose a second time.
    pub clip_from_world: [[f32; 4]; 4],
    /// Last extraction's matrix when history is valid; otherwise current.
    pub previous_clip_from_world: [[f32; 4]; 4],
    /// Pixel rectangle `[x, y, width, height]`.
    pub viewport: [u32; 4],
    /// Application-controlled temporal reset epoch.
    pub history_epoch: u64,
    /// True only for a continuous camera incarnation without cut/epoch/rect reset.
    pub history_valid: bool,
}

/// Why a camera was omitted. Prerequisite errors precede value validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceViewExtractError {
    /// No native culling frustum exists on the camera entity.
    MissingFrustum,
    /// No propagated native pose exists on the camera entity.
    MissingGlobalTransform,
    /// At least one clip matrix element is NaN or infinite.
    NonFiniteClipFromWorld,
    /// The supplied world-to-clip matrix has zero determinant.
    SingularClipFromWorld,
    /// At least one global affine element is NaN or infinite.
    NonFiniteGlobalTransform,
    /// The global affine has zero determinant.
    SingularGlobalTransform,
    /// At least one native frustum plane coefficient is NaN or infinite.
    NonFiniteFrustum,
    /// Width or height is zero.
    EmptyViewport,
    /// A viewport endpoint cannot be represented by a u32.
    ViewportOverflow,
}

/// One rejected camera; unrelated valid cameras are still published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceViewRejection {
    /// Main-world camera identity.
    pub camera: Entity,
    /// First validation failure for this camera.
    pub error: SurfaceViewExtractError,
}

/// Render-world snapshot, replaced on every call, including empty/failed calls.
///
/// Lookup uses the full scene [`Entity`]. Iteration and diagnostics follow
/// ascending `(entity.index(), entity.generation())` order, independent of ECS
/// archetype traversal. Private tokens distinguish remove/readd even within a
/// single ECS change tick. Dropping this resource resets all retained history.
#[derive(Resource, Default, Debug)]
pub struct ExtractedSurfaceViews {
    views: BTreeMap<Entity, SurfaceView>,
    rejections: Vec<SurfaceViewRejection>,
    incarnations: BTreeMap<Entity, Arc<()>>,
    source: Option<WorldId>,
}

impl ExtractedSurfaceViews {
    /// Find an accepted view by its main-world camera entity.
    pub fn get(&self, camera: Entity) -> Option<&SurfaceView> {
        self.views.get(&camera)
    }

    /// Accepted views in deterministic entity order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SurfaceView> + DoubleEndedIterator {
        self.views.values()
    }

    /// Number of accepted views in this extraction.
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether this extraction has no accepted cameras.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Rejected cameras in deterministic entity order, for this extraction only.
    pub fn rejections(&self) -> &[SurfaceViewRejection] {
        &self.rejections
    }
}

// Installed lazily in main. Observe removal, not replacement: an ordinary
// matrix/pose update must preserve history. Tokens are retained only for the
// currently accepted cameras, so despawns/invalid cameras cannot grow the table.
#[derive(Resource, Default)]
struct SurfaceViewIncarnations(BTreeMap<Entity, Arc<()>>);

fn forget_incarnation(context: &mut ObserverContext<'_>) {
    if let Some(tokens) = context.world.get_resource_mut::<SurfaceViewIncarnations>() {
        tokens.0.remove(&context.entity);
    }
}

/// Extract native surface cameras once, preserving only consecutive history.
///
/// Installs a private main-world resource and removal observers on first use;
/// it does not alter camera, frustum, transform, flags, or the ECS change tick.
/// Requires `PrismCamera + PrismFrustum + GlobalTransform`. Missing prerequisites,
/// nonfinite/singular matrices, nonfinite planes, and empty/overflowing rectangles
/// are diagnosed in [`ExtractedSurfaceViews::rejections`] and never published.
/// Constant zero-normal planes are allowed, matching native infinite-far views.
///
/// A successful view inherits history only from this render world's immediately
/// preceding extraction, with the same main world, entity generation, component
/// incarnation, epoch and viewport, and no current
/// [`ViewFlags::CAMERA_CUT`]. A cut frame can seed history for the following frame.
/// Call even when no cameras exist, so disappearance invalidates history.
pub fn extract_surface_views(main: &mut World, render: &mut World) {
    if let Some(published) = main.get_resource::<PublishedViews>() {
        extract_published_surface_views(main, render, published);
        return;
    }
    if main.get_resource::<SurfaceViewIncarnations>().is_none() {
        main.insert_resource(SurfaceViewIncarnations::default());
        main.observe::<PrismCamera, _>(LifecycleEvent::Remove, forget_incarnation);
        main.observe::<PrismFrustum, _>(LifecycleEvent::Remove, forget_incarnation);
        main.observe::<GlobalTransform, _>(LifecycleEvent::Remove, forget_incarnation);
    }

    // Query cameras alone so incomplete candidates produce diagnostics.
    let mut query = main.query::<(Entity, &PrismCamera)>();
    let cameras: BTreeMap<_, _> = query
        .iter(main)
        .map(|(entity, camera)| (entity, *camera))
        .collect();
    let previous = render
        .remove_resource::<ExtractedSurfaceViews>()
        .unwrap_or_default();
    let source = Some(main.id());
    let mut extracted = ExtractedSurfaceViews {
        source,
        ..Default::default()
    };

    for (entity, camera) in cameras {
        let clip_from_world = match validate_camera(main, entity, &camera) {
            Ok(matrix) => matrix,
            Err(error) => {
                extracted.rejections.push(SurfaceViewRejection {
                    camera: entity,
                    error,
                });
                continue;
            }
        };
        let incarnation = main
            .resource_mut::<SurfaceViewIncarnations>()
            .0
            .entry(entity)
            .or_default()
            .clone();
        let old = previous.get(entity).filter(|old| {
            previous.source == source
                && previous
                    .incarnations
                    .get(&entity)
                    .is_some_and(|token| Arc::ptr_eq(token, &incarnation))
                && old.history_epoch == camera.history_epoch
                && old.viewport == camera.viewport
                && !camera.flags.contains(ViewFlags::CAMERA_CUT)
        });
        extracted.views.insert(
            entity,
            SurfaceView {
                camera: entity,
                target: RenderTarget::PrimaryWindow,
                clear: ClearMode::ColorAndDepth,
                order: 0,
                clip_from_world,
                previous_clip_from_world: old.map_or(clip_from_world, |old| old.clip_from_world),
                viewport: camera.viewport,
                history_epoch: camera.history_epoch,
                history_valid: old.is_some(),
            },
        );
        extracted.incarnations.insert(entity, incarnation);
    }
    main.resource_mut::<SurfaceViewIncarnations>()
        .0
        .retain(|entity, _| extracted.views.contains_key(entity));
    render.insert_resource(extracted);
}

fn extract_published_surface_views(main: &World, render: &mut World, published: &PublishedViews) {
    let previous = render
        .remove_resource::<ExtractedSurfaceViews>()
        .unwrap_or_default();
    let source = Some(main.id());
    let mut extracted = ExtractedSurfaceViews {
        source,
        ..Default::default()
    };
    for published_view in published.iter() {
        let view = published_view.data;
        // Native scene transforms are absolute-world values. Match the
        // visibility bridge by converting the camera's origin-relative matrix
        // to that same coordinate space before validation and publication.
        let origin = view.world_origin.as_vec3();
        let matrix = columns(view.clip_from_world * Mat4::from_translation(-origin));
        let error = match validate_matrix(matrix) {
            Err(true) => Some(SurfaceViewExtractError::SingularClipFromWorld),
            Err(false) => Some(SurfaceViewExtractError::NonFiniteClipFromWorld),
            Ok(()) if !view.world_from_view.is_finite() => {
                Some(SurfaceViewExtractError::NonFiniteGlobalTransform)
            }
            Ok(()) if view.viewport_px.size[0] == 0 || view.viewport_px.size[1] == 0 => {
                Some(SurfaceViewExtractError::EmptyViewport)
            }
            Ok(())
                if view.viewport_px.origin[0]
                    .checked_add(view.viewport_px.size[0])
                    .is_none()
                    || view.viewport_px.origin[1]
                        .checked_add(view.viewport_px.size[1])
                        .is_none() =>
            {
                Some(SurfaceViewExtractError::ViewportOverflow)
            }
            Ok(()) => None,
        };
        if let Some(error) = error {
            extracted.rejections.push(SurfaceViewRejection {
                camera: published_view.entity,
                error,
            });
            continue;
        }
        let prior = previous.get(published_view.entity).filter(|old| {
            previous.source == source
                && view.history_valid
                && !view.teleport
                && old.history_epoch == view.history_epoch
                && old.viewport
                    == [
                        view.viewport_px.origin[0],
                        view.viewport_px.origin[1],
                        view.viewport_px.size[0],
                        view.viewport_px.size[1],
                    ]
        });
        extracted.views.insert(
            published_view.entity,
            SurfaceView {
                camera: published_view.entity,
                target: view.target,
                clear: view.clear,
                order: view.order,
                clip_from_world: matrix,
                previous_clip_from_world: prior.map_or(matrix, |old| old.clip_from_world),
                viewport: [
                    view.viewport_px.origin[0],
                    view.viewport_px.origin[1],
                    view.viewport_px.size[0],
                    view.viewport_px.size[1],
                ],
                history_epoch: view.history_epoch,
                history_valid: prior.is_some(),
            },
        );
    }
    render.insert_resource(extracted);
}

fn columns(matrix: Mat4) -> [[f32; 4]; 4] {
    [
        matrix.x_axis.to_array(),
        matrix.y_axis.to_array(),
        matrix.z_axis.to_array(),
        matrix.w_axis.to_array(),
    ]
}

fn validate_matrix(matrix: [[f32; 4]; 4]) -> Result<(), bool> {
    if !matrix.iter().flatten().all(|value| value.is_finite()) {
        return Err(false);
    }
    // Evaluate in f64: finite invertible f32 matrices can have determinants
    // that underflow/overflow f32. No arbitrary epsilon rejects small scales.
    let [a, b, c, d] = matrix.map(|column| column.map(f64::from));
    let minor = |i: usize, j: usize| c[i] * d[j] - c[j] * d[i];
    let determinant = a[0] * (b[1] * minor(2, 3) - b[2] * minor(1, 3) + b[3] * minor(1, 2))
        - a[1] * (b[0] * minor(2, 3) - b[2] * minor(0, 3) + b[3] * minor(0, 2))
        + a[2] * (b[0] * minor(1, 3) - b[1] * minor(0, 3) + b[3] * minor(0, 1))
        - a[3] * (b[0] * minor(1, 2) - b[1] * minor(0, 2) + b[2] * minor(0, 1));
    if determinant == 0.0 {
        Err(true)
    } else {
        Ok(())
    }
}

fn validate_camera(
    main: &World,
    entity: Entity,
    camera: &PrismCamera,
) -> Result<[[f32; 4]; 4], SurfaceViewExtractError> {
    use SurfaceViewExtractError as Error;

    let frustum = main
        .get::<PrismFrustum>(entity)
        .ok_or(Error::MissingFrustum)?;
    let global = main
        .get::<GlobalTransform>(entity)
        .ok_or(Error::MissingGlobalTransform)?;
    let matrix = columns(camera.clip_from_world);
    validate_matrix(matrix).map_err(|singular| {
        if singular {
            Error::SingularClipFromWorld
        } else {
            Error::NonFiniteClipFromWorld
        }
    })?;
    validate_matrix(columns(global.to_matrix())).map_err(|singular| {
        if singular {
            Error::SingularGlobalTransform
        } else {
            Error::NonFiniteGlobalTransform
        }
    })?;
    if !frustum.0.planes.iter().all(|plane| {
        plane
            .normal
            .to_array()
            .iter()
            .all(|value| value.is_finite())
            && plane.d.is_finite()
    }) {
        return Err(Error::NonFiniteFrustum);
    }
    let [x, y, width, height] = camera.viewport;
    if width == 0 || height == 0 {
        return Err(Error::EmptyViewport);
    }
    if x.checked_add(width).is_none() || y.checked_add(height).is_none() {
        return Err(Error::ViewportOverflow);
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::{Frustum, Plane, Vec3, Vec4};
    use prism_transform::Transform;

    fn camera_matrix(offset: f32) -> Mat4 {
        // Distinct nonsymmetric columns catch transposition and double-pose bugs.
        Mat4::from_cols(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.5, 3.0, 0.0, 0.0),
            Vec4::new(0.0, 0.25, 4.0, -1.0),
            Vec4::new(offset, 6.0, 7.0, 1.0),
        )
    }

    fn camera(offset: f32) -> PrismCamera {
        PrismCamera::new(camera_matrix(offset), [3, 7, 640, 360])
    }

    fn frustum() -> PrismFrustum {
        PrismFrustum(Frustum::from_planes([
            Plane::new(Vec3::X, 10.0),
            Plane::new(-Vec3::X, 10.0),
            Plane::new(Vec3::Y, 10.0),
            Plane::new(-Vec3::Y, 10.0),
            Plane::new(Vec3::Z, 10.0),
            // Valid infinite-far constant plane, following native visibility.
            Plane::new(Vec3::ZERO, 1.0),
        ]))
    }

    fn global() -> GlobalTransform {
        GlobalTransform::from_transform(&Transform::from_xyz(19.0, -23.0, 11.0))
    }

    fn spawn(main: &mut World, offset: f32) -> Entity {
        main.spawn((camera(offset), frustum(), global()))
    }

    fn view(render: &World, entity: Entity) -> SurfaceView {
        *render
            .resource::<ExtractedSurfaceViews>()
            .get(entity)
            .unwrap()
    }

    #[test]
    fn published_target_and_clear_are_copied_to_surface_view() {
        use prism_app::{App, Plugin};
        use prism_camera::{CameraIntent, ClearMode, RenderTarget, Viewport};
        use prism_camera_ecs::{CameraBundle, CameraIntentComponent, CameraPlugin};

        let mut app = App::new();
        app.add_plugins(CameraPlugin::new([640, 360]));
        let entity = app.world_mut().spawn(CameraBundle {
            intent: CameraIntentComponent(CameraIntent {
                target: RenderTarget::Texture(0xCAFE),
                clear: ClearMode::DepthOnly,
                viewport: Viewport::FULL,
                ..Default::default()
            }),
            ..Default::default()
        });
        app.update();

        let mut render = World::new();
        extract_surface_views(app.world_mut(), &mut render);
        let extracted = view(&render, entity);
        assert_eq!(extracted.target, RenderTarget::Texture(0xCAFE));
        assert_eq!(extracted.clear, ClearMode::DepthOnly);
    }

    fn assert_reset(render: &World, entity: Entity) {
        let view = view(render, entity);
        assert!(!view.history_valid);
        assert_eq!(view.previous_clip_from_world, view.clip_from_world);
    }

    fn assert_rejected(render: &World, entity: Entity, error: SurfaceViewExtractError) {
        let views = render.resource::<ExtractedSurfaceViews>();
        assert!(views.get(entity).is_none());
        assert_eq!(
            views.rejections(),
            &[SurfaceViewRejection {
                camera: entity,
                error
            }]
        );
    }

    #[test]
    fn empty_world_and_default_resource_never_publish_a_camera() {
        let mut main = World::new();
        let mut render = World::new();
        assert!(ExtractedSurfaceViews::default().is_empty());
        extract_surface_views(&mut main, &mut render);
        let views = render.resource::<ExtractedSurfaceViews>();
        assert!(views.is_empty());
        assert_eq!(views.iter().count(), 0);
        assert!(views.rejections().is_empty());
    }

    #[test]
    fn current_is_exact_camera_matrix_and_previous_is_last_extraction() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 5.0);
        let tick = main.change_tick();
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
        assert_eq!(view(&render, entity).target, RenderTarget::PrimaryWindow);
        assert_eq!(view(&render, entity).clear, ClearMode::ColorAndDepth);
        assert_eq!(
            view(&render, entity).clip_from_world,
            columns(camera_matrix(5.0))
        );
        assert_eq!(main.change_tick(), tick);
        assert_eq!(*main.get::<GlobalTransform>(entity).unwrap(), global());

        main.get_mut::<PrismCamera>(entity).unwrap().clip_from_world = camera_matrix(8.0);
        extract_surface_views(&mut main, &mut render);
        let second = view(&render, entity);
        assert!(second.history_valid);
        assert_eq!(second.camera, entity);
        assert_eq!(second.viewport, [3, 7, 640, 360]);
        assert_eq!(second.clip_from_world, columns(camera_matrix(8.0)));
        assert_eq!(second.previous_clip_from_world, columns(camera_matrix(5.0)));
        extract_surface_views(&mut main, &mut render);
        assert_eq!(
            view(&render, entity).previous_clip_from_world,
            second.clip_from_world
        );
    }

    #[test]
    fn multiple_cameras_keep_independent_history_in_entity_order() {
        let mut main = World::new();
        let mut render = World::new();
        let first = spawn(&mut main, 2.0);
        let second = spawn(&mut main, 4.0);
        let third = spawn(&mut main, 6.0);
        // Force an archetype move, so query traversal need not be entity order.
        assert!(main.remove::<(PrismFrustum,)>(first));
        assert!(main.insert(first, frustum()));
        extract_surface_views(&mut main, &mut render);
        main.get_mut::<PrismCamera>(first).unwrap().clip_from_world = camera_matrix(9.0);
        extract_surface_views(&mut main, &mut render);
        let views = render.resource::<ExtractedSurfaceViews>();
        assert_eq!(views.len(), 3);
        assert_eq!(
            views.iter().map(|v| v.camera).collect::<Vec<_>>(),
            [first, second, third]
        );
        assert_eq!(
            view(&render, first).previous_clip_from_world,
            columns(camera_matrix(2.0))
        );
        assert_eq!(
            view(&render, second).previous_clip_from_world,
            columns(camera_matrix(4.0))
        );
        assert!(views.iter().all(|v| v.history_valid));
    }

    #[test]
    fn cut_epoch_and_any_viewport_change_reset_then_recover() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        extract_surface_views(&mut main, &mut render);
        assert!(view(&render, entity).history_valid);

        main.get_mut::<PrismCamera>(entity).unwrap().flags = ViewFlags::CAMERA_CUT;
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity); // A persistent cut keeps invalidating.
        main.get_mut::<PrismCamera>(entity).unwrap().flags = ViewFlags::REVERSE_Z;
        extract_surface_views(&mut main, &mut render);
        assert!(view(&render, entity).history_valid);

        main.get_mut::<PrismCamera>(entity).unwrap().history_epoch = 41;
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
        assert_eq!(view(&render, entity).history_epoch, 41);
        extract_surface_views(&mut main, &mut render);
        assert!(view(&render, entity).history_valid);
        for viewport in [
            [3, 7, 800, 360],
            [3, 7, 800, 480],
            [4, 7, 800, 480],
            [4, 8, 800, 480],
        ] {
            main.get_mut::<PrismCamera>(entity).unwrap().viewport = viewport;
            extract_surface_views(&mut main, &mut render);
            assert_reset(&render, entity);
            extract_surface_views(&mut main, &mut render);
            assert!(view(&render, entity).history_valid);
        }
    }

    #[test]
    fn removed_camera_absent_for_an_extraction_cannot_reuse_history() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        assert!(main.remove::<(PrismCamera,)>(entity));
        extract_surface_views(&mut main, &mut render);
        assert!(render.resource::<ExtractedSurfaceViews>().is_empty());
        assert!(render
            .resource::<ExtractedSurfaceViews>()
            .rejections()
            .is_empty());
        assert!(main.resource::<SurfaceViewIncarnations>().0.is_empty());
        assert!(main.insert(entity, camera(7.0)));
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
    }

    #[test]
    fn same_tick_remove_readd_of_any_prerequisite_starts_new_incarnation() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        let tick = main.change_tick();
        for prerequisite in 0..3 {
            extract_surface_views(&mut main, &mut render);
            assert!(view(&render, entity).history_valid);
            match prerequisite {
                0 => {
                    assert!(main.remove::<(PrismCamera,)>(entity));
                    assert!(main.insert(entity, camera(2.0)));
                }
                1 => {
                    assert!(main.remove::<(PrismFrustum,)>(entity));
                    assert!(main.insert(entity, frustum()));
                }
                _ => {
                    assert!(main.remove::<(GlobalTransform,)>(entity));
                    assert!(main.insert(entity, global()));
                }
            }
            assert_eq!(main.change_tick(), tick);
            extract_surface_views(&mut main, &mut render);
            assert_reset(&render, entity);
        }
    }

    #[test]
    fn ordinary_component_replacement_preserves_history() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        assert!(main.insert(entity, (camera(7.0), frustum(), global())));
        extract_surface_views(&mut main, &mut render);
        assert!(view(&render, entity).history_valid);
        assert_eq!(
            view(&render, entity).previous_clip_from_world,
            columns(camera_matrix(2.0))
        );
    }

    #[test]
    fn despawned_slot_reuse_never_inherits_the_old_entity_history() {
        let mut main = World::new();
        let mut render = World::new();
        let old = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        assert!(main.despawn(old));
        let new = spawn(&mut main, 3.0);
        assert_eq!(old.index(), new.index());
        assert_ne!(old.generation(), new.generation());
        extract_surface_views(&mut main, &mut render);
        assert!(render
            .resource::<ExtractedSurfaceViews>()
            .get(old)
            .is_none());
        assert_reset(&render, new);
    }

    #[test]
    fn missing_prerequisites_are_diagnosed_and_do_not_publish() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = main.spawn(camera(1.0));
        extract_surface_views(&mut main, &mut render);
        assert_rejected(&render, entity, SurfaceViewExtractError::MissingFrustum);
        assert!(main.insert(entity, frustum()));
        extract_surface_views(&mut main, &mut render);
        assert_rejected(
            &render,
            entity,
            SurfaceViewExtractError::MissingGlobalTransform,
        );
        assert!(main.insert(entity, global()));
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
        assert!(render
            .resource::<ExtractedSurfaceViews>()
            .rejections()
            .is_empty());
    }

    #[test]
    fn rejected_frame_breaks_history_and_recovery_seeds_new_history() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        main.get_mut::<PrismCamera>(entity)
            .unwrap()
            .clip_from_world
            .x_axis
            .x = f32::NAN;
        extract_surface_views(&mut main, &mut render);
        assert_rejected(
            &render,
            entity,
            SurfaceViewExtractError::NonFiniteClipFromWorld,
        );
        assert!(main.insert(entity, camera(8.0)));
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
        extract_surface_views(&mut main, &mut render);
        assert!(view(&render, entity).history_valid);
    }

    #[test]
    fn every_clip_element_rejects_nan_and_both_infinities() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for column in 0..4 {
                for row in 0..4 {
                    let mut data = columns(camera_matrix(2.0));
                    data[column][row] = value;
                    main.get_mut::<PrismCamera>(entity).unwrap().clip_from_world = Mat4::from_cols(
                        Vec4::from_array(data[0]),
                        Vec4::from_array(data[1]),
                        Vec4::from_array(data[2]),
                        Vec4::from_array(data[3]),
                    );
                    extract_surface_views(&mut main, &mut render);
                    assert_rejected(
                        &render,
                        entity,
                        SurfaceViewExtractError::NonFiniteClipFromWorld,
                    );
                }
            }
        }
    }

    #[test]
    fn singular_clip_including_nonzero_dependent_columns_is_rejected() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        let mut dependent = camera_matrix(2.0);
        dependent.y_axis = dependent.x_axis * 2.0;
        for matrix in [Mat4::from_scale(Vec3::ZERO), dependent] {
            main.get_mut::<PrismCamera>(entity).unwrap().clip_from_world = matrix;
            extract_surface_views(&mut main, &mut render);
            assert_rejected(
                &render,
                entity,
                SurfaceViewExtractError::SingularClipFromWorld,
            );
        }
    }

    #[test]
    fn finite_invertible_extreme_scales_are_not_misclassified_as_singular() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        for scale in [1.0e-20, 1.0e20] {
            main.get_mut::<PrismCamera>(entity).unwrap().clip_from_world =
                Mat4::from_scale(Vec3::splat(scale));
            extract_surface_views(&mut main, &mut render);
            assert!(render
                .resource::<ExtractedSurfaceViews>()
                .get(entity)
                .is_some());
            assert!(render
                .resource::<ExtractedSurfaceViews>()
                .rejections()
                .is_empty());
        }
    }

    #[test]
    fn malformed_global_pose_is_rejected_without_recomputing_clip() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        let mut invalid = global();
        invalid.0.translation.x = f32::NAN;
        assert!(main.insert(entity, invalid));
        extract_surface_views(&mut main, &mut render);
        assert_rejected(
            &render,
            entity,
            SurfaceViewExtractError::NonFiniteGlobalTransform,
        );
        invalid = global();
        invalid.0.matrix3.x_axis = Vec3::ZERO;
        assert!(main.insert(entity, invalid));
        extract_surface_views(&mut main, &mut render);
        assert_rejected(
            &render,
            entity,
            SurfaceViewExtractError::SingularGlobalTransform,
        );
    }

    #[test]
    fn nonfinite_frustum_is_rejected_but_constant_infinite_far_plane_is_valid() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
        for normal in [false, true] {
            let mut invalid = frustum();
            if normal {
                invalid.0.planes[2].normal.y = f32::NAN;
            } else {
                invalid.0.planes[5].d = f32::INFINITY;
            }
            assert!(main.insert(entity, invalid));
            extract_surface_views(&mut main, &mut render);
            assert_rejected(&render, entity, SurfaceViewExtractError::NonFiniteFrustum);
        }
    }

    #[test]
    fn empty_and_overflowing_viewports_are_rejected_but_max_endpoint_is_valid() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        for (viewport, error) in [
            ([0, 0, 0, 1], SurfaceViewExtractError::EmptyViewport),
            ([0, 0, 1, 0], SurfaceViewExtractError::EmptyViewport),
            (
                [u32::MAX, 0, 1, 1],
                SurfaceViewExtractError::ViewportOverflow,
            ),
            (
                [0, u32::MAX, 1, 1],
                SurfaceViewExtractError::ViewportOverflow,
            ),
            (
                [9, 0, u32::MAX, 1],
                SurfaceViewExtractError::ViewportOverflow,
            ),
        ] {
            main.get_mut::<PrismCamera>(entity).unwrap().viewport = viewport;
            extract_surface_views(&mut main, &mut render);
            assert_rejected(&render, entity, error);
        }
        main.get_mut::<PrismCamera>(entity).unwrap().viewport = [u32::MAX - 1, u32::MAX - 1, 1, 1];
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
    }

    #[test]
    fn invalid_camera_does_not_suppress_valid_camera_and_diagnostics_are_sorted() {
        let mut main = World::new();
        let mut render = World::new();
        let missing_frustum = main.spawn(camera(1.0));
        let valid = spawn(&mut main, 2.0);
        let missing_global = main.spawn((camera(3.0), frustum()));
        extract_surface_views(&mut main, &mut render);
        let views = render.resource::<ExtractedSurfaceViews>();
        assert_eq!(views.len(), 1);
        assert_eq!(views.iter().next().unwrap().camera, valid);
        assert_eq!(
            views.rejections(),
            &[
                SurfaceViewRejection {
                    camera: missing_frustum,
                    error: SurfaceViewExtractError::MissingFrustum
                },
                SurfaceViewRejection {
                    camera: missing_global,
                    error: SurfaceViewExtractError::MissingGlobalTransform
                },
            ]
        );
    }

    #[test]
    fn different_main_world_with_identical_entity_bits_cannot_inherit_history() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        let mut replacement = World::new();
        assert_eq!(spawn(&mut replacement, 7.0), entity);
        extract_surface_views(&mut replacement, &mut render);
        assert_reset(&render, entity);
    }

    #[test]
    fn resetting_render_resource_discards_previous_history() {
        let mut main = World::new();
        let mut render = World::new();
        let entity = spawn(&mut main, 2.0);
        extract_surface_views(&mut main, &mut render);
        extract_surface_views(&mut main, &mut render);
        assert!(view(&render, entity).history_valid);
        render.insert_resource(ExtractedSurfaceViews::default());
        extract_surface_views(&mut main, &mut render);
        assert_reset(&render, entity);
    }
}
