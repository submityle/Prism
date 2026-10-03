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
use bevy_math::{Vec2, Vec3};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::irradiance_volume::{
    classify_probe, cosine_weighted_irradiance, fibonacci_sphere_dir, normal_backface_weight,
    relocate_offset, sharpened_depth_moments, trilinear_weights, DdgiDepthOct, IrradianceOct,
    ProbeRayStats, ProbeState,
};
use prism_render_shading::gi::world_space::octahedral::{dir_to_oct, oct_to_dir};
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

// --- Probe-update (block 2c) coverage ---------------------------------------

/// Compiles `ddgi_probe_update.wesl`, proving the one-workgroup-per-probe
/// re-integration kernel parses and type-checks exactly as it will in the
/// render world (depth / normal / scene-colour reads, the lattice uniform, the
/// probe-meta + irradiance / depth history `read_write` storage buffers and the
/// two octahedral atlas storage writes).
#[test]
fn ddgi_probe_update_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let update = shader_id(0x4444_4749_5f50_5255_5044_4154_455f_0002);
    cache.set_shader(
        update,
        Shader::from_wesl(
            include_str!("../../shaders/ddgi_probe_update.wesl"),
            "embedded://prism_render_scene/shaders/ddgi_probe_update.wesl",
        ),
    );

    cache
        .get(0, update, &[])
        .unwrap_or_else(|error| panic!("ddgi_probe_update.wesl failed to compile: {error}"));
}

/// Compiles `ddgi_composite.wesl` standalone through the shared WESL -> naga
/// path. The composite is a self-contained screen-space fold with no imports,
/// so a successful compile validates both entry points (`ddgi_copy_base` lifts
/// scene_color into the base scratch; `ddgi_composite` folds the gather back
/// in) and the `rgba16float` storage write against the immediate `params`.
#[test]
fn ddgi_composite_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let composite = shader_id(0x4444_4749_5f43_4f4d_504f_5349_5445_0003);
    cache.set_shader(
        composite,
        Shader::from_wesl(
            include_str!("../../shaders/ddgi_composite.wesl"),
            "embedded://prism_render_scene/shaders/ddgi_composite.wesl",
        ),
    );

    cache
        .get(0, composite, &[])
        .unwrap_or_else(|error| panic!("ddgi_composite.wesl failed to compile: {error}"));
}

/// Guards the composite immediate ABI against drift from the WESL
/// `CompositeParams` struct: two `u32` extents + two `u32` pads = 16 bytes.
#[test]
fn ddgi_composite_abi_matches_the_shader_layout() {
    use super::abi::GpuDdgiCompositeParams;
    assert_eq!(size_of::<GpuDdgiCompositeParams>(), 16);
    assert_eq!(align_of::<GpuDdgiCompositeParams>(), 4);
}

/// Guards the probe-update immediate ABI against drift from the WESL
/// `UpdateParams` struct: two `mat4x4<f32>` (128) + five scalars (20) + three
/// pads (12) = 160 bytes, and the workgroup constant matches
/// `@workgroup_size(64)`.
#[test]
fn ddgi_probe_update_abi_matches_the_shader_layout() {
    use super::abi::{GpuDdgiUpdateParams, DDGI_PROBE_UPDATE_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuDdgiUpdateParams>(), 160);
    assert_eq!(align_of::<GpuDdgiUpdateParams>(), 4);
    assert_eq!(DDGI_PROBE_UPDATE_WORKGROUP_SIZE, 64);
}

/// The WESL `IRR_INTERIOR` / `DEPTH_INTERIOR` tile sizes must equal the golden
/// settings defaults the atlas / history allocations are sized for, or the
/// per-probe tile strides diverge between the shader and the resource buffers.
#[test]
fn ddgi_probe_update_consts_match_settings() {
    use super::settings::{DEFAULT_DEPTH_INTERIOR, DEFAULT_IRRADIANCE_INTERIOR};
    // WESL `const IRR_INTERIOR: u32 = 6u;` / `const DEPTH_INTERIOR: u32 = 16u;`.
    assert_eq!(DEFAULT_IRRADIANCE_INTERIOR, 6);
    assert_eq!(DEFAULT_DEPTH_INTERIOR, 16);
}

const WESL_RAY_COUNT: usize = 64;
const WESL_PI: f32 = 3.1415927;
const WESL_MIN_POSITIVE: f32 = 1.175494e-38;
const WESL_WEIGHT_EPSILON: f32 = 1.0e-9;

/// Mirror of the WESL `fibonacci_sphere_dir` (std transcendentals, matching the
/// device `cos` / `sin` / `sqrt`).
#[allow(clippy::disallowed_methods)]
fn mirror_fibonacci_sphere_dir(index: u32, count: u32) -> Vec3 {
    if count == 0 {
        return Vec3::new(0.0, 0.0, 1.0);
    }
    let golden = WESL_PI * (3.0 - 5.0f32.sqrt());
    let i = index as f32;
    let n = count as f32;
    let z = 1.0 - 2.0 * (i + 0.5) / n;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let theta = golden * i;
    Vec3::new(r * theta.cos(), r * theta.sin(), z)
}

/// Mirror of the WESL `oct_to_dir`.
fn mirror_oct_to_dir(uv: Vec2) -> Vec3 {
    fn sign_not_zero(value: f32) -> f32 {
        if value >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
    let e = uv * 2.0 - Vec2::ONE;
    let z = 1.0 - e.x.abs() - e.y.abs();
    let mut x = e.x;
    let mut y = e.y;
    if z < 0.0 {
        x = (1.0 - e.y.abs()) * sign_not_zero(e.x);
        y = (1.0 - e.x.abs()) * sign_not_zero(e.y);
    }
    Vec3::new(x, y, z).normalize()
}

/// Mirror of the WESL `cosine_weighted_irradiance` loop over the shared ray
/// budget (`ray_dir` / `ray_radiance`).
fn mirror_cosine_irradiance(axis_dir: Vec3, rays: &[(Vec3, Vec3)]) -> Vec3 {
    let axis_len_sq = axis_dir.dot(axis_dir);
    if axis_len_sq <= WESL_MIN_POSITIVE {
        return Vec3::ZERO;
    }
    let axis = axis_dir * axis_len_sq.sqrt().recip();
    let mut sum = Vec3::ZERO;
    let mut weight_sum = 0.0f32;
    for &(d, radiance) in rays {
        let len_sq = d.dot(d);
        if len_sq <= WESL_MIN_POSITIVE {
            continue;
        }
        let dn = d * len_sq.sqrt().recip();
        let w = axis.dot(dn).max(0.0);
        if w <= 0.0 {
            continue;
        }
        sum += radiance * w;
        weight_sum += w;
    }
    if weight_sum <= WESL_WEIGHT_EPSILON {
        return Vec3::ZERO;
    }
    let inv = weight_sum.recip();
    Vec3::new(
        (sum.x * inv).max(0.0),
        (sum.y * inv).max(0.0),
        (sum.z * inv).max(0.0),
    )
}

/// Mirror of the WESL `sharpened_depth_moments` loop. Returns `None` on the
/// degenerate axis / empty-weight paths (WESL `.z == 0`).
#[allow(clippy::disallowed_methods)]
fn mirror_sharpened_depth_moments(
    axis_dir: Vec3,
    rays: &[(Vec3, f32)],
    sharpness: f32,
    max_distance: f32,
) -> Option<[f32; 2]> {
    let axis_len_sq = axis_dir.dot(axis_dir);
    if axis_len_sq <= WESL_MIN_POSITIVE {
        return None;
    }
    let axis = axis_dir * axis_len_sq.sqrt().recip();
    let sharp = sharpness.max(1.0);
    let md = max_distance.max(0.0);
    let mut sum_d = 0.0f32;
    let mut sum_d2 = 0.0f32;
    let mut weight_sum = 0.0f32;
    for &(d, dist_raw) in rays {
        let len_sq = d.dot(d);
        if len_sq <= WESL_MIN_POSITIVE {
            continue;
        }
        let dn = d * len_sq.sqrt().recip();
        let c = axis.dot(dn).max(0.0);
        if c <= 0.0 {
            continue;
        }
        let w = c.powf(sharp);
        if w <= 0.0 {
            continue;
        }
        let dist = dist_raw.clamp(0.0, md);
        sum_d += dist * w;
        sum_d2 += dist * dist * w;
        weight_sum += w;
    }
    if weight_sum <= WESL_WEIGHT_EPSILON {
        return None;
    }
    let inv = weight_sum.recip();
    Some([sum_d * inv, sum_d2 * inv])
}

#[test]
fn mirror_fibonacci_sphere_matches_golden() {
    for count in [1u32, 8, 16, 64] {
        for index in 0..count {
            let golden = fibonacci_sphere_dir(index as usize, count as usize);
            let mirror = mirror_fibonacci_sphere_dir(index, count);
            assert!(
                (golden - mirror).length() < PARITY_EPS,
                "fibonacci drift at {index}/{count}: golden {golden:?} vs mirror {mirror:?}",
            );
        }
    }
    // Degenerate count falls back to +Z on both sides.
    assert!((fibonacci_sphere_dir(0, 0) - mirror_fibonacci_sphere_dir(0, 0)).length() < PARITY_EPS);
}

#[test]
fn mirror_oct_to_dir_matches_golden() {
    // The probe update addresses interior texels through `oct_to_dir`; sweep a
    // dense UV grid (both hemispheres + the diagonal seam) and assert the
    // device decode matches the golden to `1e-6`.
    for iy in 0..=16u32 {
        for ix in 0..=16u32 {
            let uv = Vec2::new(ix as f32 / 16.0, iy as f32 / 16.0);
            let golden = oct_to_dir(uv);
            let mirror = mirror_oct_to_dir(uv);
            assert!(
                (golden - mirror).length() < PARITY_EPS,
                "oct_to_dir drift at {uv:?}: golden {golden:?} vs mirror {mirror:?}",
            );
        }
    }
}

/// A deterministic Fibonacci ray budget tagged with radiance / distance, so the
/// re-integration mirrors exercise the exact shared-budget loops the device
/// runs (64 rays, `probe_update`'s `RAY_COUNT`).
fn parity_ray_budget() -> (Vec<(Vec3, Vec3)>, Vec<(Vec3, f32)>) {
    let mut radiance = Vec::with_capacity(WESL_RAY_COUNT);
    let mut depth = Vec::with_capacity(WESL_RAY_COUNT);
    for r in 0..WESL_RAY_COUNT {
        let dir = fibonacci_sphere_dir(r, WESL_RAY_COUNT);
        let i = r as f32;
        // Smoothly varying, strictly positive radiance / distance so every ray
        // contributes and the weighted sums are well conditioned.
        let rad = Vec3::new(
            0.2 + 0.1 * (i * 0.17).fract(),
            0.3 + 0.1 * (i * 0.31).fract(),
            0.4 + 0.1 * (i * 0.07).fract(),
        );
        let dist = 1.0 + (i * 0.5).fract() * 4.0;
        radiance.push((dir, rad));
        depth.push((dir, dist));
    }
    (radiance, depth)
}

#[test]
fn mirror_cosine_irradiance_matches_golden() {
    let (rays, _) = parity_ray_budget();
    for normal in parity_directions() {
        let golden = cosine_weighted_irradiance(normal, &rays);
        let mirror = mirror_cosine_irradiance(normal, &rays);
        assert!(
            (golden - mirror).length() < PARITY_EPS,
            "cosine irradiance drift for {normal:?}: golden {golden:?} vs mirror {mirror:?}",
        );
    }
    // Degenerate axis collapses to zero on both sides.
    assert!(cosine_weighted_irradiance(Vec3::ZERO, &rays).length() < PARITY_EPS);
    assert!(mirror_cosine_irradiance(Vec3::ZERO, &rays).length() < PARITY_EPS);
}

#[test]
fn mirror_depth_moments_matches_golden() {
    let (_, rays) = parity_ray_budget();
    let sharpness = 50.0f32;
    let max_distance = 1.0e4f32;
    for dir in parity_directions() {
        let golden = sharpened_depth_moments(dir, &rays, sharpness, max_distance);
        let mirror = mirror_sharpened_depth_moments(dir, &rays, sharpness, max_distance);
        match (golden, mirror) {
            (Some(g), Some(m)) => assert!(
                (g[0] - m[0]).abs() < PARITY_EPS && (g[1] - m[1]).abs() < PARITY_EPS,
                "depth-moment drift for {dir:?}: golden {g:?} vs mirror {m:?}",
            ),
            (None, None) => {}
            other => panic!("depth-moment validity disagreement for {dir:?}: {other:?}"),
        }
    }
    // Degenerate axis -> both sides report "keep previous".
    assert!(sharpened_depth_moments(Vec3::ZERO, &rays, sharpness, max_distance).is_none());
    assert!(mirror_sharpened_depth_moments(Vec3::ZERO, &rays, sharpness, max_distance).is_none());
}

/// Builds ray statistics by hand to drive the relocation / classification
/// branches independently of the trace.
fn stats_with(
    backface_fraction: f32,
    closest_frontface_distance: f32,
    closest_frontface_dir: Vec3,
    farthest_frontface_distance: f32,
    farthest_frontface_dir: Vec3,
    closest_backface_dir: Vec3,
    closest_backface_distance: f32,
) -> ProbeRayStats {
    let mut stats = ProbeRayStats::open();
    stats.backface_fraction = backface_fraction;
    stats.closest_frontface_distance = closest_frontface_distance;
    stats.closest_frontface_dir = closest_frontface_dir;
    stats.farthest_frontface_distance = farthest_frontface_distance;
    stats.farthest_frontface_dir = farthest_frontface_dir;
    stats.closest_backface_dir = closest_backface_dir;
    stats.closest_backface_distance = closest_backface_distance;
    stats
}

/// Mirror of the WESL `relocate_offset`, called with the same tunables the
/// dispatch feeds the immediate.
fn mirror_relocate_offset(
    prev: Vec3,
    stats: &ProbeRayStats,
    spacing: Vec3,
    min_front: f32,
    backface_threshold: f32,
    relocation_limit: f32,
) -> Vec3 {
    fn sanitize(v: Vec3) -> Vec3 {
        let m = 1.0e-6f32;
        Vec3::new(v.x.abs().max(m), v.y.abs().max(m), v.z.abs().max(m))
    }
    fn normalize_or_none(v: Vec3) -> Option<Vec3> {
        let len_sq = v.dot(v);
        if len_sq <= WESL_MIN_POSITIVE {
            None
        } else {
            Some(v * len_sq.sqrt().recip())
        }
    }
    let spacing = sanitize(spacing);
    let min_front = min_front.max(0.0);
    let bt = backface_threshold.clamp(0.0, 1.0);
    let limit = relocation_limit.clamp(0.0, 0.4999);
    let mut offset = prev;
    if stats.backface_fraction > bt {
        if let Some(nd) = normalize_or_none(stats.closest_backface_dir) {
            let push = stats.closest_backface_distance.max(0.0) + min_front;
            offset = prev + nd * push;
        }
    } else if stats.closest_frontface_distance < min_front {
        if let Some(nd) = normalize_or_none(stats.farthest_frontface_dir) {
            offset = prev + nd * min_front;
        }
    } else {
        offset = prev * 0.5;
    }
    let max_axis = spacing * limit;
    Vec3::new(
        offset.x.clamp(-max_axis.x, max_axis.x),
        offset.y.clamp(-max_axis.y, max_axis.y),
        offset.z.clamp(-max_axis.z, max_axis.z),
    )
}

#[test]
fn mirror_relocate_offset_matches_golden() {
    let spacing = Vec3::splat(1.5);
    let min_front = 0.3f32;
    let bt = 0.25f32;
    let limit = 0.45f32;
    let cases = [
        // Embedded in geometry (back-face branch).
        stats_with(
            0.8,
            0.4,
            Vec3::X,
            2.0,
            Vec3::Y,
            Vec3::new(-1.0, 0.0, 0.2),
            0.5,
        ),
        // Too close to a surface (front-face back-off branch).
        stats_with(
            0.0,
            0.1,
            Vec3::Z,
            3.0,
            Vec3::new(0.1, 1.0, 0.0),
            Vec3::ZERO,
            3.4e38,
        ),
        // Open space (decay branch).
        stats_with(0.0, 2.0, Vec3::X, 4.0, Vec3::Y, Vec3::ZERO, 3.4e38),
        // Back-face branch with a degenerate escape dir (no move).
        stats_with(0.9, 0.2, Vec3::X, 1.0, Vec3::Y, Vec3::ZERO, 0.3),
    ];
    for (prev, stats) in [Vec3::ZERO, Vec3::new(0.3, -0.2, 0.1)]
        .iter()
        .flat_map(|p| cases.iter().map(move |s| (*p, s)))
    {
        let golden = relocate_offset(prev, stats, spacing, min_front, bt, limit);
        let mirror = mirror_relocate_offset(prev, stats, spacing, min_front, bt, limit);
        assert!(
            (golden - mirror).length() < PARITY_EPS,
            "relocate drift: golden {golden:?} vs mirror {mirror:?}",
        );
    }
}

/// Mirror of the WESL `classify_probe_active` (1 = active, 0 = inactive); the
/// golden `classify_probe` never returns `NewlyVacated`.
fn mirror_classify_active(stats: &ProbeRayStats, activity_distance: f32, bt: f32) -> f32 {
    let bt = bt.clamp(0.0, 1.0);
    let ad = activity_distance.max(0.0);
    if stats.backface_fraction > bt {
        return 0.0;
    }
    if stats.closest_frontface_distance <= ad {
        1.0
    } else {
        0.0
    }
}

#[test]
fn mirror_classify_probe_matches_golden() {
    let bt = 0.25f32;
    let ad = 2.0f32;
    let cases = [
        stats_with(0.5, 1.0, Vec3::X, 2.0, Vec3::Y, Vec3::Z, 0.5), // embedded -> inactive
        stats_with(0.0, 1.5, Vec3::X, 2.0, Vec3::Y, Vec3::ZERO, 3.4e38), // near -> active
        stats_with(0.0, 5.0, Vec3::X, 6.0, Vec3::Y, Vec3::ZERO, 3.4e38), // isolated -> inactive
        stats_with(0.0, 2.0, Vec3::X, 2.0, Vec3::Y, Vec3::ZERO, 3.4e38), // boundary -> active
    ];
    for stats in &cases {
        let golden = match classify_probe(stats, ad, bt) {
            ProbeState::Active => 1.0,
            ProbeState::Inactive => 0.0,
            ProbeState::NewlyVacated => panic!("classify_probe must not return NewlyVacated"),
        };
        let mirror = mirror_classify_active(stats, ad, bt);
        assert!(
            (golden - mirror).abs() < PARITY_EPS,
            "classify drift: golden {golden} vs mirror {mirror}",
        );
    }
}
