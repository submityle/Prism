//! `CPU` bake and `GPU` upload of the Linearly Transformed Cosines (`LTC`)
//! look-up table backing the area-light passes.
//!
//! The golden [`prism_render_shading::gi::area_light::ltc_lut`] fits, per
//! `(n·v, roughness)` grid texel, the five non-zero coefficients of the `LTC`
//! inverse transform `M⁻¹` plus the lobe amplitude (directional albedo). This
//! module bakes that table on the `CPU` (no device required) and uploads it to
//! two `Rgba32Float` textures the resolve path samples bilinearly:
//!
//! * `coeffs` texture — `(a00, a02, a11, a20)` per texel.
//! * `amp` texture — `(a22, amplitude, 0, 0)` per texel.
//!
//! Splitting the six coefficients across two four-channel textures keeps every
//! value full `f32` precision and lets a single `n·v` x roughness sample fetch
//! the whole transform in two taps, matching the `CPU` reference exactly
//! (clamp-to-edge, bilinear) when block 2 wires it into the clustered-lighting
//! resolve.
//!
//! The bake is pure and deterministic; [`pack_ltc_lut`] is unit-tested against
//! the golden fitter without a `GPU`, so the uploaded bytes are provably the
//! `CPU` reference table.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    render_resource::{
        AddressMode, FilterMode, Sampler, SamplerDescriptor, Texture, TextureDataOrder,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor,
    },
    renderer::{RenderDevice, RenderQueue},
};
use prism_render_shading::gi::area_light::ltc_lut::{bake_ltc_lut, LtcLut};

use super::settings::PrismAreaLightSettings;

/// `Rgba32Float` is four `f32` channels — 16 bytes per texel.
const LTC_TEXEL_FLOATS: usize = 4;

/// The baked `LTC` look-up table split into the two `Rgba32Float` channel sets
/// the twin textures carry.
///
/// `coeffs` and `amp` are both row-major `size * size * 4` long (roughness
/// indexes rows, `n·v` indexes columns), matching the golden
/// [`LtcLut::texels`] ordering.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AreaLightLtcTexels {
    /// Grid resolution per axis.
    pub size: u32,
    /// `(a00, a02, a11, a20)` per texel, row-major.
    pub coeffs: Vec<f32>,
    /// `(a22, amplitude, 0, 0)` per texel, row-major.
    pub amp: Vec<f32>,
}

/// Packs a baked [`LtcLut`] into the two `Rgba32Float` channel sets.
///
/// Pure and `GPU`-free: the output is exactly the golden coefficients laid out
/// four floats per texel, so a device upload of these bytes reproduces the
/// `CPU` reference table bit-for-bit.
pub(crate) fn pack_ltc_lut(lut: &LtcLut) -> AreaLightLtcTexels {
    let size = lut.size();
    let texels = lut.texels();
    let mut coeffs = Vec::with_capacity(texels.len() * LTC_TEXEL_FLOATS);
    let mut amp = Vec::with_capacity(texels.len() * LTC_TEXEL_FLOATS);
    for c in texels {
        coeffs.push(c.a00);
        coeffs.push(c.a02);
        coeffs.push(c.a11);
        coeffs.push(c.a20);
        amp.push(c.a22);
        amp.push(c.amplitude);
        amp.push(0.0);
        amp.push(0.0);
    }
    AreaLightLtcTexels { size, coeffs, amp }
}

/// The uploaded `LTC` `LUT`: the two coefficient textures, their views and the
/// shared linear clamp-to-edge sampler the resolve path binds.
///
/// The views and sampler are consumed by the block-2 clustered-lighting resolve
/// integration; block 1 only bakes and uploads them.
#[derive(Resource)]
pub(crate) struct AreaLightLtcLut {
    #[expect(
        dead_code,
        reason = "retains the coeffs texture; the view is bound by the block-2 resolve wiring"
    )]
    coeffs_texture: Texture,
    #[expect(
        dead_code,
        reason = "retains the amplitude texture; the view is bound by the block-2 resolve wiring"
    )]
    amp_texture: Texture,
    #[expect(
        dead_code,
        reason = "bound by the block-2 clustered-lighting resolve integration"
    )]
    coeffs_view: TextureView,
    #[expect(
        dead_code,
        reason = "bound by the block-2 clustered-lighting resolve integration"
    )]
    amp_view: TextureView,
    #[expect(
        dead_code,
        reason = "bound by the block-2 clustered-lighting resolve integration"
    )]
    sampler: Sampler,
    #[expect(
        dead_code,
        reason = "read by the block-2 clustered-lighting resolve integration"
    )]
    size: UVec2,
}

/// Uploads one `Rgba32Float` `size x size` `LUT` texture from packed `f32`
/// channel data.
fn upload_channel(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &'static str,
    size: u32,
    data: &[f32],
) -> Texture {
    device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some(label),
            size: UVec2::splat(size).to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        TextureDataOrder::default(),
        bytemuck::cast_slice(data),
    )
}

/// Bakes the `LTC` `LUT` at `size` / `grid` and uploads the two coefficient
/// textures plus a linear clamp-to-edge sampler.
pub(crate) fn bake_and_upload_ltc_lut(
    device: &RenderDevice,
    queue: &RenderQueue,
    size: u32,
    grid: u32,
) -> AreaLightLtcLut {
    let lut = bake_ltc_lut(size, grid);
    let packed = pack_ltc_lut(&lut);
    let coeffs_texture = upload_channel(
        device,
        queue,
        "prism area-light LTC coeffs LUT",
        packed.size,
        &packed.coeffs,
    );
    let amp_texture = upload_channel(
        device,
        queue,
        "prism area-light LTC amplitude LUT",
        packed.size,
        &packed.amp,
    );
    let coeffs_view = coeffs_texture.create_view(&TextureViewDescriptor::default());
    let amp_view = amp_texture.create_view(&TextureViewDescriptor::default());
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism area-light LTC LUT linear-clamp sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });
    AreaLightLtcLut {
        coeffs_texture,
        amp_texture,
        coeffs_view,
        amp_view,
        sampler,
        size: UVec2::splat(packed.size),
    }
}

/// `RenderStartup` system that bakes and uploads the `LTC` `LUT` once when the
/// area-light subsystem is enabled.
///
/// The table is static (independent of the frame), so it is baked a single time
/// at startup; the `enabled` gate keeps the upload off the default path. Block 2
/// consumes the resulting [`AreaLightLtcLut`] from the resolve bind groups.
pub(crate) fn init_area_light_ltc_lut(
    mut commands: Commands,
    settings: Res<PrismAreaLightSettings>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if !settings.enabled {
        return;
    }
    let lut = bake_and_upload_ltc_lut(&device, &queue, settings.lut_size, settings.fit_grid);
    commands.insert_resource(lut);
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::gi::area_light::ltc_lut::{fit_ltc, LtcCoeffs};

    #[test]
    fn pack_lays_out_four_floats_per_texel() {
        let lut = bake_ltc_lut(8, 12);
        let packed = pack_ltc_lut(&lut);
        assert_eq!(packed.size, 8);
        assert_eq!(packed.coeffs.len(), 8 * 8 * LTC_TEXEL_FLOATS);
        assert_eq!(packed.amp.len(), 8 * 8 * LTC_TEXEL_FLOATS);
        // Every value is finite (the golden fitter sanitizes to finite coeffs).
        assert!(packed.coeffs.iter().all(|v| v.is_finite()));
        assert!(packed.amp.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pack_matches_the_golden_coefficients_exactly() {
        let size = 8u32;
        let grid = 12u32;
        let lut = bake_ltc_lut(size, grid);
        let packed = pack_ltc_lut(&lut);
        // Spot-check a handful of texels against a direct golden fit at the same
        // texel centres (roughness indexes rows, n·v indexes columns).
        let inv = 1.0 / size as f32;
        for &(col, row) in &[(0u32, 0u32), (3, 5), (7, 7), (2, 6)] {
            let n_dot_v = (col as f32 + 0.5) * inv;
            let roughness = (row as f32 + 0.5) * inv;
            let expected: LtcCoeffs = fit_ltc(n_dot_v, roughness, grid);
            let base = ((row * size + col) as usize) * LTC_TEXEL_FLOATS;
            assert_eq!(packed.coeffs[base], expected.a00);
            assert_eq!(packed.coeffs[base + 1], expected.a02);
            assert_eq!(packed.coeffs[base + 2], expected.a11);
            assert_eq!(packed.coeffs[base + 3], expected.a20);
            assert_eq!(packed.amp[base], expected.a22);
            assert_eq!(packed.amp[base + 1], expected.amplitude);
            assert_eq!(packed.amp[base + 2], 0.0);
            assert_eq!(packed.amp[base + 3], 0.0);
        }
    }

    #[test]
    fn amplitude_channel_is_a_bounded_albedo() {
        let packed = pack_ltc_lut(&bake_ltc_lut(16, 16));
        for texel in packed.amp.chunks_exact(LTC_TEXEL_FLOATS) {
            // amplitude is the directional albedo in [0, 1].
            assert!((0.0..=1.0).contains(&texel[1]), "amp={}", texel[1]);
            // The padding channels are exactly zero.
            assert_eq!(texel[2], 0.0);
            assert_eq!(texel[3], 0.0);
        }
    }
}
