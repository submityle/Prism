//! Per-view motion-matrix uniform + motion-vector history for the resolve pass.
//!
//! The shading-resolve shader reconstructs each covered pixel's world position
//! for both the current and previous frame (`world_cur`/`world_prev`, the latter
//! placed with `scene_previous_transforms`) and projects both through the
//! matrices in this uniform to write a screen-space motion vector. That vector
//! captures camera *and* per-object motion, so the SSR temporal pass (and later
//! TAA) can reproject history along true surface motion instead of the
//! camera-only reprojection that ghosts on moving geometry.
//!
//! The previous-frame view-projection is not carried on [`ExtractedView`], so
//! this module keeps its own per-view history keyed by
//! [`RetainedViewEntity`], mirroring the visibility subsystem's
//! `previous_clip_from_world` cache but self-contained to the resolve pass.

use bevy_ecs::prelude::*;
use bevy_math::Mat4;
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    view::{ExtractedView, RetainedViewEntity},
};
use bytemuck::{Pod, Zeroable};

/// Current + previous view-projection uploaded to `shading_resolve.wesl`
/// (`@group(0) @binding(11)`).
///
/// Mirrors the `MotionMatrices` struct in `shaders/shading_resolve.wesl`
/// byte-for-byte: two column-major `mat4x4<f32>` back to back, 128 bytes total,
/// no trailing padding needed (a `mat4x4` is already 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct MotionMatrices {
    /// This frame's `clip_from_world` (world -> current clip).
    pub clip_from_world: [[f32; 4]; 4],
    /// Last frame's `clip_from_world` (world -> previous clip); equals the
    /// current matrix on the first frame a view is seen, so a fresh view emits
    /// zero camera motion.
    pub previous_clip_from_world: [[f32; 4]; 4],
}

/// Render-world cache of each view's previous-frame `clip_from_world`, keyed by
/// its stable [`RetainedViewEntity`] so a camera that persists across frames
/// reprojects against its own history.
#[derive(Resource, Default)]
pub(crate) struct ResolveMotionHistory {
    previous: HashMap<RetainedViewEntity, [[f32; 4]; 4]>,
}

/// Per-view GPU uniform holding [`MotionMatrices`] for the resolve bind group.
#[derive(Component)]
pub(crate) struct ViewMotionUniform {
    pub(crate) buffer: Buffer,
}

/// `PrepareResources` system: for every extracted view, resolve the current
/// `clip_from_world`, pair it with the cached previous one (falling back to the
/// current matrix on the first sighting), upload the 128-byte uniform and
/// refresh the history.
pub(crate) fn prepare_resolve_motion(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut history: ResMut<ResolveMotionHistory>,
    views: Query<(Entity, &ExtractedView)>,
) {
    let mut seen: HashMap<RetainedViewEntity, [[f32; 4]; 4]> = HashMap::default();
    for (entity, view) in &views {
        // Same reconstruction the visibility pass uses: prefer the explicit
        // `clip_from_world`, else compose it from the projection and the
        // inverse view transform.
        let clip: Mat4 = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let clip_array = clip.to_cols_array_2d();
        let retained = view.retained_view_entity;
        // No history on the first frame -> previous == current -> zero motion.
        let previous = history
            .previous
            .get(&retained)
            .copied()
            .unwrap_or(clip_array);

        let matrices = MotionMatrices {
            clip_from_world: clip_array,
            previous_clip_from_world: previous,
        };
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism resolve motion matrices"),
            size: size_of::<MotionMatrices>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&matrices));
        commands.entity(entity).insert(ViewMotionUniform { buffer });

        seen.insert(retained, clip_array);
    }
    // Replace the history wholesale so views that vanished do not leak matrices.
    history.previous = seen;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_matrices_layout_matches_the_wgsl_uniform_block() {
        // Two column-major mat4x4<f32> back to back; each is 64 bytes and
        // 16-byte aligned, so the block is exactly 128 bytes with no padding.
        assert_eq!(size_of::<MotionMatrices>(), 128);
        assert_eq!(align_of::<MotionMatrices>(), 4);
    }
}
