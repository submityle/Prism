//! Per-view GPU texture backing the film-grain pass, plus the per-frame seed
//! advance that animates the grain.
//!
//! The film grain runs as a single full-screen compute pass over the
//! pre-exposed HDR scene colour: `film_grain_main` adds the golden hash-noise
//! grain to each texel and writes the result to a dedicated output that the
//! dispatch then copies back over `scene_color`. This module owns that one
//! cached texture:
//!
//! * `film_grain_out` — the full-resolution grained HDR output (`rgba16float`),
//!   copied back over `scene_color` after the pass.
//!
//! The [`ExtractedCamera`]-driven prepare system (re)allocates it to match the
//! viewport, mirroring [`super::super::vignette::resources`]. It is gated on the
//! film-grain enable *and* on the presence of the visibility buffer (source of
//! the pre-exposed scene colour the pass reads and copies back over). Like the
//! vignette, the grain needs no depth, so it does not couple to the SSR prepass.
//!
//! The same system also advances the render-world [`PrismFilmGrainSettings`]
//! `frame` counter exactly once per frame (this prepare system runs once per
//! frame, unlike the per-view dispatch), giving the grain a deterministic,
//! self-contained per-frame animation seed without depending on the engine
//! `Time` / `FrameCount` resources (which this crate does not link).

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages, TextureView},
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::settings::PrismFilmGrainSettings;

/// The per-view film-grain output texture, present only while the pass is
/// enabled and the backing visibility buffer is resident.
#[derive(Component)]
pub(crate) struct ViewFilmGrain {
    /// Full-resolution grained HDR output written by `film_grain_main`
    /// (`rgba16float`), copied back over `scene_color`.
    film_grain_out: CachedTexture,
    /// Full-resolution framebuffer extent this allocation was sized for.
    pub(crate) size: UVec2,
}

impl ViewFilmGrain {
    /// Storage view of the grained HDR output written by `film_grain_main`.
    pub(crate) fn film_grain_out_view(&self) -> &TextureView {
        &self.film_grain_out.default_view
    }

    /// The grained-output GPU texture itself, for the `copy_texture_to_texture`
    /// that writes it back over `scene_color`.
    pub(crate) fn film_grain_out_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.film_grain_out.texture
    }
}

/// (Re)allocates [`ViewFilmGrain`] for every view that has a resident visibility
/// buffer while the film grain is enabled, removes it otherwise, and advances
/// the per-frame animation seed.
///
/// Gated on `settings.enabled` plus the presence of [`ViewVisibilityBuffer`]
/// (the pre-exposed scene colour) and on single-sample views (the scene colour
/// buffer is itself single-sample). The texture is re-created whenever the
/// viewport size changes, exactly like the buffer it shadows. The `frame`
/// counter is advanced once per system run (once per frame) so the grain
/// animates deterministically.
pub(crate) fn prepare_film_grain_textures(
    mut commands: Commands,
    mut settings: ResMut<PrismFilmGrainSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&ViewFilmGrain>,
    )>,
) {
    // Advance the per-frame animation seed exactly once per frame (this system
    // runs once per frame, before the per-view dispatch reads it). Wraps
    // naturally; the `params` builder folds it modulo `FRAME_SEED_PERIOD`.
    settings.frame = settings.frame.wrapping_add(1);

    for (entity, camera, msaa, visibility, existing) in &views {
        let enabled = settings.enabled
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewFilmGrain>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|textures| textures.size == size) {
            continue;
        }

        // Grained output: full-resolution wide HDR, written by `film_grain_main`
        // (storage) and copied back over `scene_color`.
        let film_grain_out = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism film grain output"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by `film_grain_main`. COPY_SRC: copied
                // back over `scene_color` after the pass so the downstream post
                // chain reads the grained image.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewFilmGrain {
            film_grain_out,
            size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_grain_output_shares_the_scene_colour_format() {
        assert_eq!(
            SCENE_COLOR_FORMAT,
            bevy_render::render_resource::TextureFormat::Rgba16Float
        );
    }
}
