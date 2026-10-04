//! Per-view resident direct-illumination export produced by the world-space
//! `ReSTIR` resolve pass and folded into `scene_color` by a downstream
//! composite.
//!
//! The resolve runs one compute invocation per screen pixel, re-hashes the
//! pixel's world shading point into the resident `SHARC` reservoir table and
//! writes the finalised cell's reconnection-shifted direct irradiance (`rgb`) +
//! hit confidence (`a`) into this `gi_out` texture. The texture is a GI-style
//! *export* (`rgba16float`, `STORAGE_BINDING | TEXTURE_BINDING`), never copied
//! back over `scene_color` by this pass — exactly mirroring
//! [`super::super::super::world_space_gi`]'s `gi_out`.
//!
//! The export is sized to the camera viewport, so the only reallocation trigger
//! is a viewport resize; a steady-state frame at the same resolution reuses the
//! resident texture (the resolve overwrites every pixel each frame). The
//! subsystem is opt-in and consumes both the SSR prepass (depth + normal) and
//! the resident reservoir table, so the export exists exactly when
//! [`PrismWorldRestirSettings::enabled`] holds and the view carries both a
//! resident [`ViewSsrTextures`] prepass and a [`ViewWorldRestir`] table.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages, TextureView},
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
};

use super::super::super::resources::SCENE_COLOR_FORMAT;
use super::super::super::ssr::ViewSsrTextures;
use super::super::resources::ViewWorldRestir;
use super::super::settings::PrismWorldRestirSettings;

/// The per-view world-space `ReSTIR` resolve export, present only while the
/// resolve is enabled and its backing SSR prepass + reservoir table are
/// resident.
#[derive(Component)]
pub(crate) struct ViewWorldRestirResolve {
    /// Full-resolution direct-illumination export (`rgba16float`): `rgb` =
    /// pre-BRDF direct irradiance, `a` = `[0, 1]` hit confidence. Written by
    /// `resolve_main` (storage) and sampled by the downstream composite
    /// (texture); never copied over `scene_color` by this pass.
    gi_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for; the
    /// only reallocation trigger.
    pub(crate) size: UVec2,
}

impl ViewWorldRestirResolve {
    /// Storage view of the direct-illumination export written by `resolve_main`.
    pub(crate) fn gi_out_view(&self) -> &TextureView {
        &self.gi_out.default_view
    }
}

/// (Re)allocates [`ViewWorldRestirResolve`] for every camera view whose SSR
/// prepass and resident reservoir table are live while the resolve is enabled,
/// and removes it otherwise.
///
/// Gated on [`PrismWorldRestirSettings::enabled`], the presence of
/// [`ViewSsrTextures`] (the resolve reconstructs its shading point from the SSR
/// prepass depth + packed normal, exactly as the producer did) and
/// [`ViewWorldRestir`] (the resident reservoir table it probes). The export is
/// sized to the camera viewport; a steady-state frame at the same resolution
/// reuses the resident texture, and only a viewport resize triggers a realloc.
pub(crate) fn prepare_world_restir_resolve(
    mut commands: Commands,
    settings: Res<PrismWorldRestirSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&ViewSsrTextures>,
        Option<&ViewWorldRestir>,
        Option<&ViewWorldRestirResolve>,
    )>,
) {
    for (entity, camera, ssr, restir, existing) in &views {
        let enabled = settings.enabled && ssr.is_some() && restir.is_some();
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewWorldRestirResolve>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        // Steady state: an existing export already sized for this viewport is
        // reused as-is (the resolve overwrites every pixel each frame).
        if existing.is_some_and(|resources| resources.size == size) {
            continue;
        }

        // Direct-illumination export: full-resolution wide HDR, written by
        // `resolve_main` (storage) and sampled by the composite (texture).
        let gi_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism world-space ReSTIR resolve output"),
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
            .insert(ViewWorldRestirResolve { gi_out, size });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_export_shares_the_scene_colour_format() {
        // The export is a GI-style wide-HDR buffer the composite samples, so it
        // must share `scene_color`'s format exactly like SSGI/world-space GI.
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
