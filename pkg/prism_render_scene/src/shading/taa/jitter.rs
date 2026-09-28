//! Per-frame sub-pixel camera jitter injection for TAA.
//!
//! A TAA *resolve* with no camera jitter is only a temporal denoise: every
//! frame samples the very same pixel centres, so the history can never resolve
//! detail finer than one pixel and edges stay aliased. Real supersampling comes
//! from nudging the camera's projection by a fraction of a pixel each frame
//! along a low-discrepancy [`Halton`](prism_render_shading::halton)(2, 3)
//! sequence; integrated over the cycle by the motion-reprojected history blend,
//! the jittered samples reconstruct a supersampled image.
//!
//! Bevy already carries the whole clip-space half of this contract: attaching a
//! [`TemporalJitter`] to a view makes
//! [`prepare_view_uniforms`](bevy_render::view::prepare_view_uniforms) shear the
//! projection (`ViewUniform.clip_from_world`) by `2 * offset / viewport` while
//! keeping a separate `unjittered_clip_from_world`. The visibility raster
//! projects through the standard `clip_from_world`, so it rasterises jittered
//! automatically; the resolve rebuilds world positions from the visibility
//! buffer's barycentrics and world-space vertices (camera-matrix independent),
//! and the motion-vector G-buffer is built from the *unjittered* projection, so
//! history reprojection tracks true surface motion rather than the artificial
//! wobble. All this module has to do is stamp the right per-frame offset.
//!
//! The offset comes straight from the golden
//! [`taa_jitter`](prism_render_shading::taa_jitter) so the CPU reference and the
//! live render agree on the sequence. The system runs in
//! [`RenderSystems::PrepareViews`](bevy_render::RenderSystems::PrepareViews),
//! which is ordered before the `PrepareResources` set that prepares the view
//! uniforms, guaranteeing the jitter lands before the projection is baked.
//! Extraction removes `TemporalJitter` every frame (the main-world camera never
//! carries it), so re-stamping it each frame is both required and idempotent;
//! when TAA is disabled the component simply stays removed.

use bevy_camera::Camera3d;
use bevy_ecs::prelude::*;
use bevy_render::{camera::TemporalJitter, view::ExtractedView};
use prism_render_shading::{taa_jitter, DEFAULT_TAA_JITTER_LEN};

use super::super::runtime::PrismShadingSettings;

/// `PrepareViews` system stamping this frame's Halton sub-pixel offset onto
/// every 3D view's [`TemporalJitter`], so the visibility raster jitters and the
/// TAA resolve accumulates a supersampled image.
///
/// The `Local` frame counter advances only while TAA is enabled and wraps with
/// the Halton cycle, so toggling TAA never leaves the sequence on a stale phase.
/// Only [`Camera3d`] views are jittered: shadow and other auxiliary subviews
/// drive their own uniforms and must stay unjittered. Skipped entirely when TAA
/// is disabled, in which case extraction leaves the component removed.
pub(crate) fn prepare_taa_jitter(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut frame: Local<u64>,
    views: Query<Entity, (With<ExtractedView>, With<Camera3d>)>,
) {
    if !settings.enable_taa {
        return;
    }

    // One-based inside `taa_jitter`, so frame 0 already skips the zero offset;
    // the counter wraps with the Halton length so it never overflows in
    // practice and the sequence stays phase-stable across a wrap.
    let offset = taa_jitter(*frame, DEFAULT_TAA_JITTER_LEN);
    *frame = frame.wrapping_add(1);

    for entity in &views {
        commands.entity(entity).insert(TemporalJitter { offset });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::entity::Entity;
    use bevy_math::{Mat4, UVec4};
    use bevy_render::{
        render_resource::TextureFormat,
        sync_world::MainEntity,
        view::{ColorGrading, RetainedViewEntity},
    };
    use bevy_transform::components::GlobalTransform;

    /// A minimal render-world 3D view: enough of an [`ExtractedView`] for the
    /// query filter plus the [`Camera3d`] marker.
    fn spawn_view(world: &mut World, main_bits: u64) -> Entity {
        world
            .spawn((
                ExtractedView {
                    retained_view_entity: RetainedViewEntity::new(
                        MainEntity::from(Entity::from_bits(main_bits)),
                        None,
                        0,
                    ),
                    clip_from_view: Mat4::IDENTITY,
                    world_from_view: GlobalTransform::IDENTITY,
                    clip_from_world: None,
                    target_format: TextureFormat::Rgba16Float,
                    viewport: UVec4::new(0, 0, 1920, 1080),
                    color_grading: ColorGrading::default(),
                    invert_culling: false,
                },
                Camera3d::default(),
            ))
            .id()
    }

    fn run(world: &mut World, schedule: &mut Schedule) {
        schedule.run(world);
    }

    #[test]
    fn stamps_the_golden_halton_sequence_per_frame() {
        let mut world = World::new();
        world.insert_resource(PrismShadingSettings {
            enable_taa: true,
            ..Default::default()
        });
        let view = spawn_view(&mut world, 1);

        let mut schedule = Schedule::default();
        schedule.add_systems(prepare_taa_jitter);

        // Two frames must reproduce `taa_jitter(0)` then `taa_jitter(1)`; the
        // persistent `Local` counter lives in the schedule's system instance.
        run(&mut world, &mut schedule);
        assert_eq!(
            world.get::<TemporalJitter>(view).unwrap().offset,
            taa_jitter(0, DEFAULT_TAA_JITTER_LEN),
        );

        run(&mut world, &mut schedule);
        assert_eq!(
            world.get::<TemporalJitter>(view).unwrap().offset,
            taa_jitter(1, DEFAULT_TAA_JITTER_LEN),
        );
    }

    #[test]
    fn all_views_share_the_frame_offset() {
        let mut world = World::new();
        world.insert_resource(PrismShadingSettings {
            enable_taa: true,
            ..Default::default()
        });
        let a = spawn_view(&mut world, 1);
        let b = spawn_view(&mut world, 2);

        let mut schedule = Schedule::default();
        schedule.add_systems(prepare_taa_jitter);
        run(&mut world, &mut schedule);

        let offset_a = world.get::<TemporalJitter>(a).unwrap().offset;
        let offset_b = world.get::<TemporalJitter>(b).unwrap().offset;
        assert_eq!(offset_a, offset_b);
        assert_eq!(offset_a, taa_jitter(0, DEFAULT_TAA_JITTER_LEN));
    }

    #[test]
    fn disabled_taa_never_stamps_jitter() {
        let mut world = World::new();
        world.insert_resource(PrismShadingSettings::default());
        let view = spawn_view(&mut world, 1);

        let mut schedule = Schedule::default();
        schedule.add_systems(prepare_taa_jitter);
        run(&mut world, &mut schedule);

        assert!(world.get::<TemporalJitter>(view).is_none());
    }
}
