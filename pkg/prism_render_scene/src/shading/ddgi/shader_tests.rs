//! WESL compilation coverage, ABI layout guards, and CPU-mirror parity tests
//! for the DDGI irradiance-volume sample subsystem.
//!
//! The sandbox has no GPU, so the compile test runs `ddgi_sample.wesl` through
//! the same `ShaderCache` / `wesl` pipeline the render world uses, validating
//! that the kernel parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl`).
//!
//! The parity tests transcribe the WESL sample maths op-for-op into Rust and
//! assert agreement with the CPU golden
//! ([`prism_render_shading::gi::irradiance_volume`]) to `1e-6`. They are what
//! keep this block honest: a green result proves the on-device octahedral
//! addressing, the trilinear / back-face / Chebyshev interpolation weights, and
//! the padded-atlas bilinear taps match the reference numerically, not just
//! that the shader compiles.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::irradiance_volume::{
    normal_backface_weight, trilinear_weights, DdgiDepthOct, IrradianceOct,
};
use prism_render_shading::gi::world_space::octahedral::dir_to_oct;
use prism_render_shading::gi::world_space::visibility::chebyshev_weight;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("DDGI sample shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `ddgi_sample.wesl`, proving the per-pixel irradiance-volume sample
/// kernel parses and type-checks exactly as it will in the render world (depth
/// + normal reads, the volume uniform, the probe-meta storage buffer, the two
/// octahedral atlas textures and the GI export storage write).
#[test]
fn ddgi_sample_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let sample = shader_id(0x4444_4749_5f53_414d_504c_455f_5745_0001);
    cache.set_shader(
        sample,
        Shader::from_wesl(
            include_str!("../../shaders/ddgi_sample.wesl"),
            "embedded://prism_render_scene/shaders/ddgi_sample.wesl",
        ),
    );

    cache
        .get(0, sample, &[])
        .unwrap_or_else(|error| panic!("ddgi_sample.wesl failed to compile: {error}"));
}

/// Guards the Rust ABIs against drift from the WESL structs: the volume uniform
/// twin is the 80-byte block, the sample immediate is the 80-byte `mat4x4`-led
/// block, and the workgroup constant matches `@workgroup_size(8, 8, 1)`.
#[test]
fn ddgi_abi_matches_the_shader_layout() {
    use super::abi::{GpuDdgiSampleParams, GpuDdgiVolume, DDGI_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuDdgiVolume>(), 80);
    assert_eq!(align_of::<GpuDdgiVolume>(), 4);
    assert_eq!(size_of::<GpuDdgiSampleParams>(), 80);
    assert_eq!(align_of::<GpuDdgiSampleParams>(), 4);
    assert_eq!(DDGI_WORKGROUP_SIZE, 8);
}

// --- CPU mirror of the WESL sample kernel ------------------------------------

const PARITY_EPS: f32 = 1.0e-6;

/// A deterministic spread of unit directions covering the sphere, used to probe
/// the octahedral addressing from every hemisphere and across the seams.
fn parity_directions() -> [Vec3; 26] {
    let d = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(-1.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, -1.0),
        Vec3::new(0.0, -1.0, 1.0),
        Vec3::new(0.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, -1.0),
    ];
    let mut out = [Vec3::ZERO; 26];
    for (slot, raw) in out.iter_mut().zip(d.iter()) {
        *slot = raw.normalize();
    }
    out
}

/// Mirror of the WESL `clamp_padded`: a floored, UV-scaled coordinate clamped
/// into `0..=last`. WESL truncates with `i32(value)`; the golden with
/// `value as usize`. Both agree for the non-negative floored inputs produced by
/// the addressing, so this exercises the exact device path.
fn mirror_clamp_padded(value: f32, last: i32) -> i32 {
    if value <= 0.0 {
        0
    } else {
        let v = value as i32;
        if v > last {
            last
        } else {
            v
        }
    }
}

/// Mirror of the WESL `oct_bilinear_irradiance` addressing + bilinear taps,
/// reading golden texels through [`IrradianceOct::texel_at`] so a match against
/// [`IrradianceOct::sample`] proves the device atlas lookup is arm-for-arm.
fn mirror_oct_irradiance(field: &IrradianceOct, normal: Vec3) -> Vec3 {
    let interior = field.interior() as f32;
    let padded = field.padded();
    let oct = dir_to_oct(normal);
    let fx = oct.x * interior + 0.5;
    let fy = oct.y * interior + 0.5;
    let x0f = fx.floor();
    let y0f = fy.floor();
    let tx = (fx - x0f).clamp(0.0, 1.0);
    let ty = (fy - y0f).clamp(0.0, 1.0);
    let last = padded as i32 - 1;
    let x0 = mirror_clamp_padded(x0f, last) as usize;
    let y0 = mirror_clamp_padded(y0f, last) as usize;
    let x1 = mirror_clamp_padded(x0f + 1.0, last) as usize;
    let y1 = mirror_clamp_padded(y0f + 1.0, last) as usize;
    let c00 = field.texel_at(x0, y0);
    let c10 = field.texel_at(x1, y0);
    let c01 = field.texel_at(x0, y1);
    let c11 = field.texel_at(x1, y1);
    let top = c00 + (c10 - c00) * tx;
    let bottom = c01 + (c11 - c01) * tx;
    let out = top + (bottom - top) * ty;
    Vec3::new(out.x.max(0.0), out.y.max(0.0), out.z.max(0.0))
}

/// Mirror of the WESL `oct_bilinear_depth` addressing + bilinear taps, reading
/// golden moments through [`DdgiDepthOct::moments_at`].
fn mirror_oct_depth(field: &DdgiDepthOct, dir: Vec3) -> [f32; 2] {
    let interior = field.interior() as f32;
    let padded = field.padded();
    let oct = dir_to_oct(dir);
    let fx = oct.x * interior + 0.5;
    let fy = oct.y * interior + 0.5;
    let x0f = fx.floor();
    let y0f = fy.floor();
    let tx = (fx - x0f).clamp(0.0, 1.0);
    let ty = (fy - y0f).clamp(0.0, 1.0);
    let last = padded as i32 - 1;
    let x0 = mirror_clamp_padded(x0f, last) as usize;
    let y0 = mirror_clamp_padded(y0f, last) as usize;
    let x1 = mirror_clamp_padded(x0f + 1.0, last) as usize;
    let y1 = mirror_clamp_padded(y0f + 1.0, last) as usize;
    let m00 = field.moments_at(x0, y0);
    let m10 = field.moments_at(x1, y0);
    let m01 = field.moments_at(x0, y1);
    let m11 = field.moments_at(x1, y1);
    let mut out = [0.0f32; 2];
    for c in 0..2 {
        let top = m00[c] + (m10[c] - m00[c]) * tx;
        let bottom = m01[c] + (m11[c] - m01[c]) * tx;
        out[c] = (top + (bottom - top) * ty).max(0.0);
    }
    out
}

/// Mirror of the WESL `trilinear_weights` corner expansion (`out[c] =
/// wx[c&1]*wy[(c>>1)&1]*wz[(c>>2)&1]`).
fn mirror_trilinear_weights(fx: f32, fy: f32, fz: f32) -> [f32; 8] {
    let fx = fx.clamp(0.0, 1.0);
    let fy = fy.clamp(0.0, 1.0);
    let fz = fz.clamp(0.0, 1.0);
    let mut out = [0.0f32; 8];
    for (c, slot) in out.iter_mut().enumerate() {
        let wx = if c & 1 == 1 { fx } else { 1.0 - fx };
        let wy = if (c >> 1) & 1 == 1 { fy } else { 1.0 - fy };
        let wz = if (c >> 2) & 1 == 1 { fz } else { 1.0 - fz };
        *slot = wx * wy * wz;
    }
    out
}

/// Mirror of the WESL `normal_backface_weight`.
fn mirror_backface_weight(point: Vec3, normal: Vec3, probe: Vec3) -> f32 {
    let to_probe = probe - point;
    let len_sq = to_probe.length_squared();
    let n_len_sq = normal.length_squared();
    if len_sq <= f32::MIN_POSITIVE || n_len_sq <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let dir = to_probe * len_sq.sqrt().recip();
    let n = normal * n_len_sq.sqrt().recip();
    let wrap = (0.5 + 0.5 * dir.dot(n)).max(0.0);
    wrap * wrap
}

/// Mirror of the WESL `chebyshev_weight`.
fn mirror_chebyshev_weight(mean: f32, mean_sq: f32, distance: f32) -> f32 {
    if distance <= mean {
        return 1.0;
    }
    let variance = (mean_sq - mean * mean).max(0.0);
    let delta = distance - mean;
    (variance / (variance + delta * delta)).clamp(0.0, 1.0)
}

#[test]
fn mirror_irradiance_sample_matches_golden() {
    // Build a directional (non-uniform) irradiance field so the bilinear taps
    // and the addressing both bite; `hysteresis = 0` keeps the integrated
    // radiance so the field is not flattened toward the zero prior.
    let rays = [
        (Vec3::X, Vec3::new(1.0, 0.1, 0.1)),
        (-Vec3::X, Vec3::new(0.1, 0.1, 1.0)),
        (Vec3::Y, Vec3::new(0.2, 1.0, 0.3)),
        (-Vec3::Y, Vec3::new(0.3, 0.2, 0.1)),
        (Vec3::Z, Vec3::new(0.9, 0.6, 0.2)),
        (-Vec3::Z, Vec3::new(0.1, 0.8, 0.7)),
    ];
    let mut field = IrradianceOct::new(6);
    field.update(&rays, 0.0);

    for normal in parity_directions() {
        let golden = field.sample(normal);
        let mirror = mirror_oct_irradiance(&field, normal);
        assert!(
            (golden - mirror).length() < PARITY_EPS,
            "irradiance sample drift for {normal:?}: golden {golden:?} vs mirror {mirror:?}",
        );
    }
}

#[test]
fn mirror_depth_sample_matches_golden() {
    let rays = [
        (Vec3::X, 2.0f32),
        (-Vec3::X, 5.0),
        (Vec3::Y, 3.5),
        (-Vec3::Y, 7.0),
        (Vec3::Z, 4.0),
        (-Vec3::Z, 6.0),
    ];
    let mut field = DdgiDepthOct::new(16);
    // `update` takes (dir, distance) rays and a sharpness exponent; use the
    // golden default sharpness so the moments are the on-device values.
    field.update(&rays, 0.0, 50.0, 1.0e4);

    for dir in parity_directions() {
        let golden = field.sample(dir);
        let mirror = mirror_oct_depth(&field, dir);
        assert!(
            (golden[0] - mirror[0]).abs() < PARITY_EPS
                && (golden[1] - mirror[1]).abs() < PARITY_EPS,
            "depth-moment drift for {dir:?}: golden {golden:?} vs mirror {mirror:?}",
        );
    }
}

#[test]
fn mirror_trilinear_weights_matches_golden() {
    let fracs = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.25, 0.5, 0.75),
        Vec3::new(0.9, 0.1, 0.4),
        Vec3::new(0.33, 0.66, 0.99),
    ];
    for frac in fracs {
        let golden = trilinear_weights(frac);
        let mirror = mirror_trilinear_weights(frac.x, frac.y, frac.z);
        for c in 0..8 {
            assert!(
                (golden[c] - mirror[c]).abs() < PARITY_EPS,
                "trilinear weight {c} drift for {frac:?}: {} vs {}",
                golden[c],
                mirror[c],
            );
        }
    }
}

#[test]
fn mirror_backface_weight_matches_golden() {
    let point = Vec3::new(0.5, 1.0, -0.3);
    let normal = Vec3::new(0.2, 0.9, 0.1).normalize();
    for probe in parity_directions() {
        let probe = point + probe * 2.0;
        let golden = normal_backface_weight(point, normal, probe);
        let mirror = mirror_backface_weight(point, normal, probe);
        assert!(
            (golden - mirror).abs() < PARITY_EPS,
            "backface weight drift for probe {probe:?}: {golden} vs {mirror}",
        );
    }
    // Degenerate probe / normal fall back to the no-op weight on both sides.
    assert!((normal_backface_weight(point, Vec3::ZERO, point + Vec3::X) - 1.0).abs() < PARITY_EPS);
    assert!((mirror_backface_weight(point, Vec3::ZERO, point + Vec3::X) - 1.0).abs() < PARITY_EPS);
}

#[test]
fn mirror_chebyshev_weight_matches_golden() {
    let cases = [
        (5.0f32, 26.0f32, 3.0f32),
        (5.0, 26.0, 5.0),
        (5.0, 26.0, 8.0),
        (2.0, 4.0, 10.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
    ];
    for (mean, mean_sq, distance) in cases {
        let golden = chebyshev_weight(mean, mean_sq, distance);
        let mirror = mirror_chebyshev_weight(mean, mean_sq, distance);
        assert!(
            (golden - mirror).abs() < PARITY_EPS,
            "chebyshev drift for ({mean}, {mean_sq}, {distance}): {golden} vs {mirror}",
        );
    }
}
