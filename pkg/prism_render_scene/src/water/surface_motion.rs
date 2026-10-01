//! The water-surface pass's **motion-vector** `@group(4)` plumbing: the `CPU`
//! half that lets the transparent surface fragment stage write its own
//! screen-space motion vector into the shared motion G-buffer
//! ([`ViewVisibilityBuffer::motion_vectors_view`](crate::shading::ViewVisibilityBuffer)),
//! so the `SSR` temporal reprojection (and later `TAA` / temporal upsampling)
//! can reproject history along the water surface instead of ghosting it against
//! the submerged opaque geometry behind it.
//!
//! ## Why the water pass has to write motion itself
//!
//! The opaque `shading_resolve` compute pass fills the `Rg16Float` motion
//! G-buffer for every covered *opaque* pixel (camera **and** per-object motion;
//! see [`super::super::shading::resolve`]'s `motion` module). The water surface
//! is a separate transparent raster pass drawn *after* that resolve, and it does
//! not write depth, so nothing downstream knows which screen pixels the water
//! now occupies. Left alone, those pixels keep the motion vector of the opaque
//! surface *behind* the water — the lake bed, not the surface — and the temporal
//! reprojection smears the reflection/refraction every time the camera moves.
//! Writing the surface's own motion as a second render target (`MRT`) over the
//! same G-buffer fixes the reprojection basis for exactly the water pixels the
//! raster covers, mirroring how `UE5` Single Layer Water and `Frostbite` emit
//! water into the velocity buffer.
//!
//! ## Scope: camera reprojection this slice
//!
//! The vector this slice emits is the **camera** reprojection of the surface's
//! current world position: both the current and previous `clip_from_world` are
//! applied to the *same* displaced world position, so a static wave under a
//! moving camera reprojects correctly, but the wave's own displacement motion
//! (this frame's crest versus last frame's) is **not** yet captured — that needs
//! a double-buffered displacement field and is a separate following slice. This
//! is an honest increment: it closes the camera-motion half of the gap without
//! claiming the full surface-motion solution.
//!
//! ## Why a per-view prepare system
//!
//! The previous-frame view-projection is not carried on [`ExtractedView`], so —
//! exactly like the resolve pass's `ResolveMotionHistory` — this module keeps
//! its own per-view history keyed by [`RetainedViewEntity`] and refreshes it
//! wholesale each frame (so a camera that vanishes cannot leak matrices). The
//! prepare system builds the per-view [`ViewWaterMotionUniform`] the raster draw
//! node binds as `@group(4)`; the sibling [`super::surface_pipeline`] adds the
//! fifth bind-group layout and the second colour target that carries the vector.

use bevy_ecs::prelude::*;
use bevy_material::bind_group_layout_entries::{
    binding_types::uniform_buffer_sized, BindGroupLayoutEntries,
};
use bevy_math::Mat4;
use bevy_platform::collections::HashMap;
use bevy_render::{
    render_resource::{BindGroupLayoutEntry, Buffer, BufferDescriptor, BufferUsages, ShaderStages},
    renderer::{RenderDevice, RenderQueue},
    view::{ExtractedView, RetainedViewEntity},
};
use bytemuck::{Pod, Zeroable};

/// `GPU`-side mirror of the shader's `WaterMotionConfig` uniform (the water
/// `@group(4) @binding(0)` block).
///
/// Layout matches the `WGSL` struct byte-for-byte: two column-major
/// `mat4x4<f32>` back to back, 128 bytes total, no trailing padding (a
/// `mat4x4<f32>` is already 16-byte aligned, which is the alignment the `WGSL`
/// uniform address space requires for a struct whose widest member is a
/// `mat4x4<f32>`). The fragment stage projects the surface's current world
/// position through both matrices and writes `cur_uv - prev_uv`, the exact
/// `cur_uv - prev_uv` convention the opaque resolve pass's `MotionMatrices`
/// encodes, so the two writers agree on the shared G-buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterMotionConfig {
    /// This frame's `clip_from_world` (world -> current clip), column-major.
    pub clip_from_world: [[f32; 4]; 4],
    /// Last frame's `clip_from_world` (world -> previous clip), column-major;
    /// equals the current matrix on the first frame a view is seen, so a fresh
    /// view emits zero camera motion.
    pub previous_clip_from_world: [[f32; 4]; 4],
}

impl GpuWaterMotionConfig {
    /// Packs the current and previous view-projection into the uniform, each
    /// uploaded column-major to match the `WGSL` `mat4x4<f32>` memory order.
    pub(crate) fn new(clip: Mat4, previous: Mat4) -> Self {
        Self {
            clip_from_world: clip.to_cols_array_2d(),
            previous_clip_from_world: previous.to_cols_array_2d(),
        }
    }
}

/// Render-world cache of each view's previous-frame `clip_from_world`, keyed by
/// its stable [`RetainedViewEntity`] so a camera that persists across frames
/// reprojects against its own history. Refreshed wholesale every frame by
/// [`prepare_water_surface_motion`] so a view that disappears does not leak its
/// matrix, mirroring the resolve pass's `ResolveMotionHistory`.
#[derive(Resource, Default)]
pub(crate) struct WaterMotionHistory {
    previous: HashMap<RetainedViewEntity, Mat4>,
}

/// Per-view `GPU` uniform holding the [`GpuWaterMotionConfig`] the surface
/// raster draw binds as `@group(4)`.
#[derive(Component)]
pub(crate) struct ViewWaterMotionUniform {
    /// The 128-byte current+previous view-projection uniform buffer.
    pub(crate) buffer: Buffer,
}

/// Builds the water-surface `@group(4)` motion layout: a single
/// [`ShaderStages::FRAGMENT`] uniform (the water fragment stage writes the
/// motion vector, so only that stage reads the matrices).
pub(crate) fn motion_layout_entries() -> [BindGroupLayoutEntry; 1] {
    BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, uniform_buffer_sized(false, None))
}

/// `Prepare` system: for every extracted view, resolve the current
/// `clip_from_world`, pair it with the cached previous one (falling back to the
/// current matrix on the first sighting so a fresh view emits zero motion),
/// upload the 128-byte uniform and refresh the history.
pub(crate) fn prepare_water_surface_motion(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut history: ResMut<WaterMotionHistory>,
    views: Query<(Entity, &ExtractedView)>,
) {
    let mut seen: HashMap<RetainedViewEntity, Mat4> = HashMap::default();
    for (entity, view) in &views {
        // Same reconstruction the visibility and resolve passes use: prefer the
        // explicit `clip_from_world`, else compose it from the projection and
        // the inverse view transform.
        let clip: Mat4 = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let retained = view.retained_view_entity;
        // No history on the first frame -> previous == current -> zero motion.
        let previous = history.previous.get(&retained).copied().unwrap_or(clip);

        let config = GpuWaterMotionConfig::new(clip, previous);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism water surface motion config"),
            size: size_of::<GpuWaterMotionConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&config));
        commands
            .entity(entity)
            .insert(ViewWaterMotionUniform { buffer });

        seen.insert(retained, clip);
    }
    // Replace the history wholesale so views that vanished do not leak matrices.
    history.previous = seen;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn config_matches_the_wgsl_uniform_block_size_and_alignment() {
        // Two column-major `mat4x4<f32>` back to back; each is 64 bytes and
        // 16-byte aligned, so the block is exactly 128 bytes with no padding.
        assert_eq!(size_of::<GpuWaterMotionConfig>(), 128);
        assert_eq!(align_of::<GpuWaterMotionConfig>(), 4);
    }

    #[test]
    fn new_uploads_each_matrix_column_major() {
        let clip = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let prev = Mat4::from_cols_array(&[
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ]);
        let cfg = GpuWaterMotionConfig::new(clip, prev);
        assert_eq!(cfg.clip_from_world, clip.to_cols_array_2d());
        assert_eq!(cfg.previous_clip_from_world, prev.to_cols_array_2d());
    }

    #[test]
    fn first_sighting_pairs_the_current_matrix_with_itself() {
        // When `previous == current` the shader projects the same world point
        // through identical matrices, so `cur_uv - prev_uv` is exactly zero: a
        // view's first frame emits no camera motion.
        let clip = Mat4::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ]);
        let cfg = GpuWaterMotionConfig::new(clip, clip);
        assert_eq!(cfg.clip_from_world, cfg.previous_clip_from_world);
    }

    #[test]
    fn layout_declares_one_fragment_uniform_binding() {
        let entries = motion_layout_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].binding, 0);
        assert!(entries[0].visibility.contains(ShaderStages::FRAGMENT));
    }
}
