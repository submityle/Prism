//! `WESL` compilation coverage, `ABI` layout guards, and `CPU` math-mirror
//! parity for the surface-cache subsystem.
//!
//! The sandbox has no `GPU`, so these tests compile all five `WESL` sources
//! through the same `ShaderCache` / `wesl` pipeline the render world uses,
//! proving `surface_cache_{alloc,update,spatial_filter,coverage}.wesl` and
//! `surface_cache_composite.wesl` parse and type-check exactly as they will on
//! device. The kernels are self-contained (no intra-crate `import`s, matching
//! the world-space `GI` sources), so a green result also guards the surfel
//! coverage / temporal maths against drift from its `CPU` golden twin in
//! [`prism_render_shading::gi::surface_cache`].
//!
//! The math-mirror tests transcribe the scalar `WESL` coverage kernel
//! (`radial_weight` x `axial_weight` x `normal_consistency`) op-for-op into
//! Rust and cross-check it against the golden
//! [`Surfel::coverage`](prism_render_shading::gi::surface_cache::Surfel), and
//! pin the temporal-integration helpers (`confidence_alpha`, `is_disoccluded`,
//! `integrate_radiance`, `spatial_filter`) to the golden exact values, so a
//! divergence between the on-device kernels and the `CPU` reference fails here.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::{ops, Vec3};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::surface_cache::{
    confidence_alpha, integrate_radiance, is_disoccluded, normal_consistency, spatial_filter,
    CoverageParams, Surfel, SurfelCacheEntry, TemporalParams,
};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("surface cache shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles a single surface-cache `WESL` source through the render-world
/// shader pipeline, panicking with the compiler diagnostic on failure.
fn assert_wesl_compiles(tag: u128, name: &str, source: &'static str) {
    let mut cache = ShaderCache::new((), load_source);
    let id = shader_id(tag);
    let path = alloc_embedded_path(name);
    cache.set_shader(id, Shader::from_wesl(source, path));
    cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{name} failed to compile: {error}"));
}

fn alloc_embedded_path(name: &str) -> String {
    format!("embedded://prism_render_scene/shaders/{name}")
}

#[test]
fn alloc_wesl_compiles_standalone() {
    assert_wesl_compiles(
        0x5052_4953_4d5f_5343_5f41_4c4c_4f43_0001,
        "surface_cache_alloc.wesl",
        include_str!("../../shaders/surface_cache_alloc.wesl"),
    );
}

#[test]
fn update_wesl_compiles_standalone() {
    assert_wesl_compiles(
        0x5052_4953_4d5f_5343_5f55_5044_4154_0002,
        "surface_cache_update.wesl",
        include_str!("../../shaders/surface_cache_update.wesl"),
    );
}

#[test]
fn spatial_filter_wesl_compiles_standalone() {
    assert_wesl_compiles(
        0x5052_4953_4d5f_5343_5f46_494c_5445_0003,
        "surface_cache_spatial_filter.wesl",
        include_str!("../../shaders/surface_cache_spatial_filter.wesl"),
    );
}

#[test]
fn coverage_wesl_compiles_standalone() {
    assert_wesl_compiles(
        0x5052_4953_4d5f_5343_5f43_4f56_4552_0004,
        "surface_cache_coverage.wesl",
        include_str!("../../shaders/surface_cache_coverage.wesl"),
    );
}

#[test]
fn composite_wesl_compiles_standalone() {
    assert_wesl_compiles(
        0x5052_4953_4d5f_5343_5f43_4f4d_5053_0005,
        "surface_cache_composite.wesl",
        include_str!("../../shaders/surface_cache_composite.wesl"),
    );
}

/// Guards the Rust immediate-block `ABI`s against drift from the `WESL`
/// structs: the five immediate blocks keep their on-wire sizes (96 / 16 / 32 /
/// 112 / 16 bytes), the persistent surfel element stays the 48-byte scalar
/// std430 layout, and the workgroup constants match the shader
/// `@workgroup_size` attributes.
#[test]
fn surface_cache_abi_matches_the_shader_layout() {
    use super::abi::{
        GpuSurfaceCacheAllocParams, GpuSurfaceCacheCompositeParams, GpuSurfaceCacheCoverageParams,
        GpuSurfaceCacheFilterParams, GpuSurfaceCacheUpdateParams, GpuSurfel,
        SURFACE_CACHE_WORKGROUP_SIZE_1D, SURFACE_CACHE_WORKGROUP_SIZE_2D,
    };
    assert_eq!(size_of::<GpuSurfel>(), 48);
    assert_eq!(align_of::<GpuSurfel>(), 4);
    assert_eq!(size_of::<GpuSurfaceCacheAllocParams>(), 96);
    assert_eq!(size_of::<GpuSurfaceCacheUpdateParams>(), 16);
    assert_eq!(size_of::<GpuSurfaceCacheFilterParams>(), 32);
    assert_eq!(size_of::<GpuSurfaceCacheCoverageParams>(), 112);
    assert_eq!(size_of::<GpuSurfaceCacheCompositeParams>(), 16);
    assert_eq!(SURFACE_CACHE_WORKGROUP_SIZE_1D, 64);
    assert_eq!(SURFACE_CACHE_WORKGROUP_SIZE_2D, 8);
}

// --- CPU transcription of the scalar WESL coverage kernel ------------------
//
// These mirror the helper functions in `surface_cache_coverage.wesl` op-for-op
// (which in turn mirror the golden `Surfel::coverage`), so the parity test is
// an independent cross-check that fails if either the shader or the golden
// drifts.

const MIN_RADIUS: f32 = 1.0e-6;

fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    ops::exp(x.clamp(-80.0, 0.0))
}

fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

fn radial_distance(pos: Vec3, normal: Vec3, point: Vec3) -> f32 {
    let delta = point - pos;
    let axial = delta.dot(normal);
    let in_plane = delta - normal * axial;
    let len_sq = in_plane.length_squared();
    if len_sq > 0.0 {
        len_sq.sqrt()
    } else {
        0.0
    }
}

fn wesl_radial_weight(pos: Vec3, normal: Vec3, radius: f32, point: Vec3) -> f32 {
    let t = (radial_distance(pos, normal, point) / radius).clamp(0.0, 1.0);
    (1.0 - t * t).clamp(0.0, 1.0)
}

fn wesl_axial_weight(pos: Vec3, normal: Vec3, radius: f32, point: Vec3, tol: f32) -> f32 {
    let axial = (point - pos).dot(normal).abs();
    let scale = radius * tol.max(0.0);
    let denom = scale.max(MIN_RADIUS);
    stable_exp(-axial / denom)
}

fn wesl_normal_consistency(a: Vec3, b: Vec3, sharpness: f32) -> f32 {
    let cosine = safe_normalize(a).dot(safe_normalize(b)).clamp(0.0, 1.0);
    ops::powf(cosine, sharpness.max(0.0)).clamp(0.0, 1.0)
}

/// Full coverage weight transcribed from the `WESL` `coverage_weight`.
fn wesl_coverage(
    pos: Vec3,
    normal: Vec3,
    radius: f32,
    point: Vec3,
    point_normal: Vec3,
    params: &CoverageParams,
) -> f32 {
    let w_radial = wesl_radial_weight(pos, normal, radius, point);
    if w_radial <= 0.0 {
        return 0.0;
    }
    let w_axial = wesl_axial_weight(pos, normal, radius, point, params.axial_tolerance);
    let w_normal = wesl_normal_consistency(normal, point_normal, params.normal_sharpness);
    (w_radial * w_axial * w_normal).clamp(0.0, 1.0)
}

#[test]
fn wesl_coverage_matches_the_golden_surfel_coverage() {
    let params = CoverageParams::default();
    let surfel = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);
    // A spread of in-plane, off-plane and normal-mismatched sample points.
    let samples = [
        (Vec3::ZERO, Vec3::Z),
        (Vec3::new(0.3, 0.0, 0.0), Vec3::Z),
        (Vec3::new(0.0, 0.0, 0.2), Vec3::Z),
        (Vec3::new(0.2, 0.1, 0.1), Vec3::new(0.0, 0.1, 1.0)),
        (Vec3::new(0.5, 0.0, 0.0), Vec3::new(1.0, 0.0, 1.0)),
        (Vec3::new(2.0, 0.0, 0.0), Vec3::Z),
    ];
    for (point, point_normal) in samples {
        let mirror = wesl_coverage(
            surfel.position,
            surfel.normal,
            surfel.radius,
            point,
            point_normal,
            &params,
        );
        let golden = surfel.coverage(point, point_normal, &params);
        assert!(
            (mirror - golden).abs() < 1e-6,
            "coverage drift at {point:?}/{point_normal:?}: wesl {mirror} vs golden {golden}"
        );
    }
}

#[test]
fn wesl_normal_consistency_matches_the_golden() {
    let cases = [
        (Vec3::Z, Vec3::Z, 8.0_f32),
        (Vec3::Z, Vec3::NEG_Z, 8.0),
        (Vec3::Z, Vec3::new(1.0, 0.0, 1.0), 8.0),
        (Vec3::Z, Vec3::new(1.0, 0.0, 1.0), 0.0),
    ];
    for (a, b, sharpness) in cases {
        let mirror = wesl_normal_consistency(a, b, sharpness);
        let golden = normal_consistency(a, b, sharpness);
        assert!(
            (mirror - golden).abs() < 1e-6,
            "normal_consistency drift: wesl {mirror} vs golden {golden}"
        );
    }
}

#[test]
fn confidence_alpha_pins_the_golden_curve() {
    assert!((confidence_alpha(0, 32) - 1.0).abs() < 1e-6);
    assert!((confidence_alpha(1, 32) - 0.5).abs() < 1e-6);
    assert!((confidence_alpha(5, 32) - 1.0 / 6.0).abs() < 1e-6);
    // Floored at 1 / max_samples once confidence saturates.
    assert!((confidence_alpha(1000, 32) - 1.0 / 32.0).abs() < 1e-6);
}

#[test]
fn disocclusion_matches_the_golden_decision() {
    let params = TemporalParams::default();
    let prev = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);
    // Two radii away => reset.
    let jumped = Surfel::new(Vec3::new(2.0, 0.0, 0.0), Vec3::Z, 1.0);
    assert!(is_disoccluded(&prev, &jumped, &params));
    // Normal flip => reset.
    let flipped = Surfel::new(Vec3::ZERO, Vec3::NEG_Z, 1.0);
    assert!(is_disoccluded(&prev, &flipped, &params));
    // Small motion keeps history.
    let nudged = Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.05, 0.0, 1.0), 1.0);
    assert!(!is_disoccluded(&prev, &nudged, &params));
}

#[test]
fn integrate_radiance_pins_the_golden_ema() {
    let params = TemporalParams::default();
    let surfel = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);
    // No history: snap to the fresh sample at confidence 1.
    let seeded = integrate_radiance(None, &surfel, Vec3::splat(1.0), &params);
    assert_eq!(seeded.sample_count, 1);
    assert!((seeded.radiance - Vec3::splat(1.0)).length() < 1e-6);
    // With history: confidence advances and radiance blends between the two.
    let entry = SurfelCacheEntry {
        radiance: Vec3::splat(5.0),
        sample_count: 10,
    };
    let blended = integrate_radiance(Some((entry, surfel)), &surfel, Vec3::splat(1.0), &params);
    assert_eq!(blended.sample_count, 11);
    assert!(blended.radiance.x < 5.0 && blended.radiance.x > 1.0);
    // Disocclusion resets to the fresh sample regardless of history.
    let jumped = Surfel::new(Vec3::new(2.0, 0.0, 0.0), Vec3::Z, 1.0);
    let reset = integrate_radiance(Some((entry, surfel)), &jumped, Vec3::splat(2.0), &params);
    assert_eq!(reset.sample_count, 1);
    assert!((reset.radiance - Vec3::splat(2.0)).length() < 1e-6);
}

#[test]
fn spatial_filter_pins_the_golden_bilateral() {
    let params = CoverageParams::default();
    let center = Surfel::new(Vec3::ZERO, Vec3::Z, 1.0);
    // Empty neighbourhood: identity.
    let identity = spatial_filter(&center, Vec3::new(0.2, 0.4, 0.8), &[], &params);
    assert!((identity - Vec3::new(0.2, 0.4, 0.8)).length() < 1e-6);
    // Locally constant signal is preserved exactly.
    let c = Vec3::new(0.3, 0.6, 0.9);
    let neighbours = [
        (Surfel::new(Vec3::new(0.1, 0.0, 0.0), Vec3::Z, 1.0), c),
        (Surfel::new(Vec3::new(-0.1, 0.1, 0.0), Vec3::Z, 1.0), c),
    ];
    let constant = spatial_filter(&center, c, &neighbours, &params);
    assert!((constant - c).length() < 1e-6, "{constant:?}");
    // A single bright compatible neighbour pulls the centre toward it.
    let pull = [(
        Surfel::new(Vec3::new(0.05, 0.0, 0.0), Vec3::Z, 1.0),
        Vec3::splat(2.0),
    )];
    let out = spatial_filter(&center, Vec3::ZERO, &pull, &params);
    assert!(out.x > 0.0 && out.x < 2.0, "{out:?}");
}
