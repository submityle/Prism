//! Per-view scratch texture backing the world-space `ReSTIR` composite's
//! copy + fold two-pass substitution.
//!
//! The composite folds the resolve's direct-illumination export (`gi_out`)
//! back over `scene_color` under an energy-conserving substitution: it removes
//! the clustered punctual diffuse the shading resolve already folded in and
//! adds the `ReSTIR` estimate in its place (see [`super`]'s module docs for the
//! fold algebra). `scene_color` is an `rgba16float` storage image, which is not
//! read-write storage-capable, so the fold cannot read and write it in a single
//! pass. A preceding copy pass lifts the shading-resolved `scene_color` into
//! this scratch `gi_base` texture; the fold then reads the base from `gi_base`
//! and writes only `scene_color`, keeping the pass free of any storage
//! read/write aliasing hazard — exactly mirroring
//! [`super::super::super::world_space_gi`]'s composite scratch.
//!
//! The scratch is sized to the resolve export it layers over, so it exists
//! exactly when the resolve export does (i.e. while the subsystem is enabled
//! and the view carries a resident [`ViewWorldRestirResolve`]) and is only
//! reallocated on a viewport resize.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages, TextureView},
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
};

use super::super::super::resources::SCENE_COLOR_FORMAT;
use super::super::resolve::ViewWorldRestirResolve;
use super::super::settings::PrismWorldRestirSettings;

/// Per-view scratch texture backing the composite's `scene_color` -> `gi_base`
/// copy, present only while the resolve export it layers over is resident.
#[derive(Component)]
pub(crate) struct ViewWorldRestirComposite {
    /// Full-resolution scratch (`rgba16float`): the 1:1 lift of the
    /// shading-resolved `scene_color` the fold reads as its base, avoiding the
    /// read/write hazard the fold would otherwise hit on a single storage image.
    gi_base: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for; the
    /// only reallocation trigger.
    pub(crate) size: UVec2,
}

impl ViewWorldRestirComposite {
    /// Storage view of the base-copy scratch written by `wr_copy_base` and read
    /// by `wr_composite`.
    pub(crate) fn gi_base_view(&self) -> &TextureView {
        &self.gi_base.default_view
    }
}

/// (Re)allocates [`ViewWorldRestirComposite`] for every view whose resolve
/// export is live while the subsystem is enabled, and removes it otherwise.
///
/// Gated on [`PrismWorldRestirSettings::enabled`] and the presence of
/// [`ViewWorldRestirResolve`] (the resolve export the composite layers over).
/// The scratch is sized to that export; a steady-state frame at the same
/// resolution reuses the resident texture, and only a viewport resize triggers
/// a realloc.
pub(crate) fn prepare_world_restir_composite(
    mut commands: Commands,
    settings: Res<PrismWorldRestirSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        Option<&ViewWorldRestirResolve>,
        Option<&ViewWorldRestirComposite>,
    )>,
) {
    for (entity, resolve, existing) in &views {
        let enabled = settings.enabled && resolve.is_some();
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewWorldRestirComposite>();
            }
            continue;
        }
        // Size the scratch to the resolve export it copies and folds over.
        let size = resolve.expect("gated on resolve.is_some()").size;
        // Steady state: an existing scratch already sized for this viewport is
        // reused as-is (the copy pass overwrites every pixel each frame).
        if existing.is_some_and(|resources| resources.size == size) {
            continue;
        }

        let gi_base = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism world-space ReSTIR composite base"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(ViewWorldRestirComposite { gi_base, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_scratch_shares_the_scene_colour_format() {
        // The scratch is a 1:1 lift of `scene_color`, so it must share
        // `scene_color`'s format exactly like the world-space GI composite base.
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
