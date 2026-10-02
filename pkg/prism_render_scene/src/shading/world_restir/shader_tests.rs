//! WESL compilation coverage + `SHARC` hash parity for the world-space
//! `ReSTIR` DI fill pass.
//!
//! The sandbox has no GPU, so correctness of `world_restir_fill.wesl` is
//! proven on the CPU in two independent ways:
//!
//! * [`fill_wesl_compiles_standalone`] runs the shader through the same
//!   `ShaderCache` / `wesl` pipeline the render world uses, so a parse or
//!   type-check regression fails the build exactly as it would on device.
//! * [`fill_shader_hash_matches_cpu_golden`] re-implements the shader's
//!   hand-rolled 64-bit (`vec2<u32>`) `SHARC` arithmetic bit-for-bit in Rust
//!   and asserts it equals the authoritative `u64` golden
//!   ([`prism_render_shading::gi::world_restir::spatial_hash`]). WGSL has no
//!   64-bit integers, so this parity is the only CPU-checkable guarantee that
//!   the kernel addresses the same reservoir slots (`hash_key` / `checksum` /
//!   `bucket_index`) as the CPU path it mirrors.
//!
//! [`world_restir_abi_matches_the_shader_layout`] pins the device `ABI` strides
//! and the workgroup size against the WESL struct / `@workgroup_size` so a
//! layout drift between the two also fails the build.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::IVec3;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::gi::world_restir::spatial_hash::{self, HashGridKey};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("world-space ReSTIR shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `world_restir_fill.wesl`, proving the GRIS spatial-reuse + finalize
/// kernel parses and type-checks exactly as it will in the render world (the
/// `src`/`dst` reservoir storage bindings and the `FillParams` immediate).
#[test]
fn fill_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let fill = shader_id(0x5052_4953_4d5f_5753_5244_5f46_494c_0001);
    cache.set_shader(
        fill,
        Shader::from_wesl(
            include_str!("../../shaders/world_restir_fill.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_fill.wesl",
        ),
    );

    cache
        .get(0, fill, &[])
        .unwrap_or_else(|error| panic!("world_restir_fill.wesl failed to compile: {error}"));
}

/// Compiles `world_restir_seed.wesl`, proving the per-cell light `RIS` seed
/// kernel parses and type-checks exactly as it will in the render world (the
/// `src`/`dst` reservoir storage bindings, the `@binding(2)` light list, and
/// the `SeedParams` immediate).
#[test]
fn seed_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let seed = shader_id(0x5052_4953_4d5f_5753_5244_5f46_494c_0002);
    cache.set_shader(
        seed,
        Shader::from_wesl(
            include_str!("../../shaders/world_restir_seed.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_seed.wesl",
        ),
    );

    cache
        .get(0, seed, &[])
        .unwrap_or_else(|error| panic!("world_restir_seed.wesl failed to compile: {error}"));
}

/// Compiles `world_restir_inject.wesl`, proving the hash-grid injection /
/// slot-claim kernel parses and type-checks exactly as it will in the render
/// world (the visible-point storage binding, the `read_write` reservoir table,
/// the `atomic<u32>` slot-state array, and the `InjectParams` immediate).
#[test]
fn inject_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let inject = shader_id(0x5052_4953_4d5f_5753_5244_5f46_494c_0003);
    cache.set_shader(
        inject,
        Shader::from_wesl(
            include_str!("../../shaders/world_restir_inject.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_inject.wesl",
        ),
    );

    cache
        .get(0, inject, &[])
        .unwrap_or_else(|error| panic!("world_restir_inject.wesl failed to compile: {error}"));
}

/// Guards the Rust device `ABI` against drift from the WESL twin: the reservoir
/// slot is the 80-byte five-lane record, the fill immediate is the 64-byte
/// four-lane block, and the workgroup constant matches `@workgroup_size(64, 1,
/// 1)`.
#[test]
fn world_restir_abi_matches_the_shader_layout() {
    use super::abi::{
        GpuWorldRestirFillParams, GpuWorldRestirInjectParams, GpuWorldRestirInjectPoint,
        GpuWorldRestirLight, GpuWorldRestirReservoir, GpuWorldRestirSeedParams,
        WORLD_RESTIR_INJECT_POINT_STRIDE, WORLD_RESTIR_INJECT_WORKGROUP_SIZE,
        WORLD_RESTIR_LIGHT_STRIDE, WORLD_RESTIR_RESERVOIR_STRIDE, WORLD_RESTIR_SEED_WORKGROUP_SIZE,
        WORLD_RESTIR_WORKGROUP_SIZE,
    };

    // Fill pass: 80-byte five-lane reservoir slot + 64-byte four-lane immediate.
    assert_eq!(size_of::<GpuWorldRestirReservoir>(), 80);
    assert_eq!(align_of::<GpuWorldRestirReservoir>(), 4);
    assert_eq!(WORLD_RESTIR_RESERVOIR_STRIDE, 80);
    assert_eq!(size_of::<GpuWorldRestirFillParams>(), 64);
    assert_eq!(align_of::<GpuWorldRestirFillParams>(), 4);
    assert_eq!(WORLD_RESTIR_WORKGROUP_SIZE, 64);

    // Seed pass: 32-byte two-lane light record + 32-byte two-lane immediate.
    assert_eq!(size_of::<GpuWorldRestirLight>(), 32);
    assert_eq!(align_of::<GpuWorldRestirLight>(), 4);
    assert_eq!(WORLD_RESTIR_LIGHT_STRIDE, 32);
    assert_eq!(size_of::<GpuWorldRestirSeedParams>(), 32);
    assert_eq!(align_of::<GpuWorldRestirSeedParams>(), 4);
    assert_eq!(WORLD_RESTIR_SEED_WORKGROUP_SIZE, 64);

    // Inject pass: 32-byte two-lane visible point + 48-byte three-lane immediate.
    assert_eq!(size_of::<GpuWorldRestirInjectPoint>(), 32);
    assert_eq!(align_of::<GpuWorldRestirInjectPoint>(), 4);
    assert_eq!(WORLD_RESTIR_INJECT_POINT_STRIDE, 32);
    assert_eq!(size_of::<GpuWorldRestirInjectParams>(), 48);
    assert_eq!(align_of::<GpuWorldRestirInjectParams>(), 4);
    assert_eq!(WORLD_RESTIR_INJECT_WORKGROUP_SIZE, 64);
}

/// Bit-exact Rust re-implementation of the WESL `vec2<u32>` 64-bit `SHARC`
/// arithmetic in `world_restir_fill.wesl`. Each `U64` is `(low, high)` and every
/// step uses wrapping `u32` ops to reproduce WGSL's wrapping integer semantics,
/// so proving this equal to the `u64` golden proves the shader does too.
mod hash_mirror {
    /// A 64-bit value split as `(low 32 bits, high 32 bits)`.
    pub type U64 = (u32, u32);

    // 64-bit mixing constants as (low, high) u32 pairs (see the WESL header).
    const SEED_KEY: U64 = (0x7f4a_7c15, 0x9e37_79b9);
    const SEED_SUM: U64 = (0x27d4_eb4f, 0xc2b2_ae3d);
    const MIX_C1: U64 = (0xed55_8ccd, 0xff51_afd7);
    const MIX_C2: U64 = (0x1a85_ec53, 0xc4ce_b9fe);
    const LEVEL_MUL: U64 = SEED_KEY;
    const NORMAL_MUL: U64 = MIX_C1;
    const COORD_BIAS: u32 = 1_048_576;
    const COORD_MASK: u32 = 0x001f_ffff;

    /// Full 32x32 -> 64 bit unsigned product (golden `mul32x32_64`).
    fn mul32x32_64(a: u32, b: u32) -> U64 {
        let a_lo = a & 0xffff;
        let a_hi = a >> 16;
        let b_lo = b & 0xffff;
        let b_hi = b >> 16;
        let t0 = a_lo.wrapping_mul(b_lo);
        let t1 = a_hi.wrapping_mul(b_lo).wrapping_add(t0 >> 16);
        let t2 = a_lo.wrapping_mul(b_hi).wrapping_add(t1 & 0xffff);
        let lo = (t2 << 16) | (t0 & 0xffff);
        let hi = a_hi
            .wrapping_mul(b_hi)
            .wrapping_add(t1 >> 16)
            .wrapping_add(t2 >> 16);
        (lo, hi)
    }

    /// Low 64 bits of a 64x64 unsigned product (golden `u64_mul`).
    fn u64_mul(a: U64, b: U64) -> U64 {
        let p = mul32x32_64(a.0, b.0);
        let hi =
            p.1.wrapping_add(a.0.wrapping_mul(b.1))
                .wrapping_add(a.1.wrapping_mul(b.0));
        (p.0, hi)
    }

    /// Logical right shift by 33 bits (golden `u64_shr33`).
    fn u64_shr33(a: U64) -> U64 {
        (a.1 >> 1, 0)
    }

    fn u64_xor(a: U64, b: U64) -> U64 {
        (a.0 ^ b.0, a.1 ^ b.1)
    }

    /// `MurmurHash3` 64-bit finalizer (golden `fmix64`).
    fn fmix64(mut x: U64) -> U64 {
        x = u64_xor(x, u64_shr33(x));
        x = u64_mul(x, MIX_C1);
        x = u64_xor(x, u64_shr33(x));
        x = u64_mul(x, MIX_C2);
        x = u64_xor(x, u64_shr33(x));
        x
    }

    /// Biases a signed axis into its low 21 bits (golden `encode_axis`).
    fn encode_axis(v: i32) -> u32 {
        (v as u32).wrapping_add(COORD_BIAS) & COORD_MASK
    }

    /// Packs a key's fields into a 64-bit value (golden `pack_key`).
    fn pack_key(coord: [i32; 3], level: i32, normal_bin: u32) -> U64 {
        let ex = encode_axis(coord[0]);
        let ey = encode_axis(coord[1]);
        let ez = encode_axis(coord[2]);
        let packed = (ex | (ey << 21), (ey >> 11) | (ez << 10));
        let level = u64_mul((level as u32, 0), LEVEL_MUL);
        let normal = u64_mul((normal_bin, 0), NORMAL_MUL);
        u64_xor(u64_xor(packed, level), normal)
    }

    /// 64-bit bucket hash of a key (golden `hash_key`).
    pub fn hash_key(coord: [i32; 3], level: i32, normal_bin: u32) -> U64 {
        fmix64(u64_xor(pack_key(coord, level, normal_bin), SEED_KEY))
    }

    /// 32-bit collision checksum (golden `key_checksum`).
    pub fn checksum(coord: [i32; 3], level: i32, normal_bin: u32) -> u32 {
        let h = fmix64(u64_xor(pack_key(coord, level, normal_bin), SEED_SUM));
        h.1 ^ h.0
    }

    /// `hash % max(capacity, 1)` via bit-serial long division, MSB first
    /// (golden `bucket_index`); assumes `capacity <= 2^31`.
    pub fn bucket_index(coord: [i32; 3], level: i32, normal_bin: u32, capacity: u32) -> u32 {
        let m = capacity.max(1);
        if m <= 1 {
            return 0;
        }
        let h = hash_key(coord, level, normal_bin);
        let mut rem: u32 = 0;
        for i in (0..32).rev() {
            let bit = (h.1 >> i) & 1;
            rem = (rem << 1) | bit;
            if rem >= m {
                rem -= m;
            }
        }
        for i in (0..32).rev() {
            let bit = (h.0 >> i) & 1;
            rem = (rem << 1) | bit;
            if rem >= m {
                rem -= m;
            }
        }
        rem
    }
}

/// Asserts the shader's `vec2<u32>` hash mirror reproduces the `u64` golden
/// `hash_key` / `checksum` / `bucket_index` for a sweep covering signed cell
/// extremes (including the +/- 2^20 bias edges), every grid level in
/// `[0, MAX_LEVEL]`, several normal bins, and a range of table capacities
/// (including the default `131072`). This is the device-equivalence proof for
/// the SHARC port under the no-GPU sandbox.
#[test]
fn fill_shader_hash_matches_cpu_golden() {
    let coords: [[i32; 3]; 8] = [
        [0, 0, 0],
        [1, 2, 3],
        [-1, -2, -3],
        [1_048_575, -1_048_576, 7],
        [-1_048_576, 1_048_575, -7],
        [12_345, -54_321, 9_999],
        [-100_000, 100_000, -1],
        [524_288, -524_288, 262_144],
    ];
    let normal_bins: [u32; 6] = [0, 1, 7, 31, 63, 4095];
    let capacities: [u32; 7] = [1, 2, 1024, 4096, 131_072, 1u32 << 20, 1u32 << 30];

    for &coord in &coords {
        for level in 0..=spatial_hash::MAX_LEVEL {
            for &normal_bin in &normal_bins {
                let key = HashGridKey {
                    cell_coord: IVec3::new(coord[0], coord[1], coord[2]),
                    level,
                    normal_bin,
                };

                let golden_hash = spatial_hash::hash_key(&key);
                let mirror_hash = hash_mirror::hash_key(coord, level, normal_bin);
                assert_eq!(
                    (golden_hash as u32, (golden_hash >> 32) as u32),
                    mirror_hash,
                    "hash_key mismatch at coord={coord:?} level={level} normal_bin={normal_bin}"
                );

                assert_eq!(
                    spatial_hash::checksum(&key),
                    hash_mirror::checksum(coord, level, normal_bin),
                    "checksum mismatch at coord={coord:?} level={level} normal_bin={normal_bin}"
                );

                for &capacity in &capacities {
                    assert_eq!(
                        spatial_hash::bucket_index(&key, capacity),
                        hash_mirror::bucket_index(coord, level, normal_bin, capacity),
                        "bucket mismatch at coord={coord:?} level={level} \
                         normal_bin={normal_bin} capacity={capacity}"
                    );
                }
            }
        }
    }
}

/// Asserts two `Vec3`s are bit-identical component-by-component. Any NaN would
/// make a plain `==` lie, so the only faithful equality for a `GPU`-mirrored
/// float is on the raw bit pattern. `ctx` is the `(frame, slot, candidate_count)`
/// coordinate, surfaced for diagnostics.
fn assert_vec3_bits(
    golden: bevy_math::Vec3,
    mirror: bevy_math::Vec3,
    field: &str,
    ctx: (u32, u32, u32),
) {
    assert_eq!(
        golden.x.to_bits(),
        mirror.x.to_bits(),
        "{field}.x mismatch at frame/slot/candidates {ctx:?}"
    );
    assert_eq!(
        golden.y.to_bits(),
        mirror.y.to_bits(),
        "{field}.y mismatch at frame/slot/candidates {ctx:?}"
    );
    assert_eq!(
        golden.z.to_bits(),
        mirror.z.to_bits(),
        "{field}.z mismatch at frame/slot/candidates {ctx:?}"
    );
}

/// Bit-exact Rust transcription of `world_restir_seed.wesl`'s `RIS` estimator.
///
/// Every function mirrors a WESL twin line-for-line (`fmix32` / `rng01` /
/// `pick_light` / `luminance` / `geometric_term` / `target_function` /
/// `build_sample` / `ris_weight` and the `seed_main` resampling loop), with the
/// WESL `x == x` NaN probe written as `!x.is_nan()`. Under the finite-only input
/// sweep in [`seed_shader_matches_cpu_golden`] this is the device-equivalence
/// proof that the kernel's inlined estimator matches the authoritative
/// `Reservoir` golden it ports (`WGSL` has no 64-bit ints, so a CPU mirror is
/// the only no-GPU check available).
pub(super) mod seed_mirror {
    use bevy_math::Vec3;
    use prism_render_shading::gi::screen_probe::restir::GiSample;

    /// Smallest positive normal `f32` (golden `f32::MIN_POSITIVE`, WESL `MIN_POSITIVE`).
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    /// Rec. 709 luminance weights (golden/WESL `LUMA_R` / `LUMA_G` / `LUMA_B`).
    const LUMA_R: f32 = 0.212_639;
    const LUMA_G: f32 = 0.715_169;
    const LUMA_B: f32 = 0.072_192;

    /// One candidate emitter — mirror of the WESL `WorldRestirLight` fields the
    /// estimator consumes (`position` / `intensity` / `color`).
    #[derive(Clone, Copy, Debug)]
    pub struct Light {
        pub position: Vec3,
        pub intensity: f32,
        pub color: Vec3,
    }

    /// The seeded reservoir summary the kernel serialises: the surviving sample,
    /// its finalised contribution weight `W`, and the capped confidence `m`.
    pub struct SeedResult {
        pub sample: Option<GiSample>,
        pub w: f32,
        pub m: f32,
    }

    /// `MurmurHash3` 32-bit finalizer (WESL `fmix32`).
    pub fn fmix32(x_in: u32) -> u32 {
        let mut x = x_in;
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    }

    /// Uniform in `[0, 1)` seeded by `(frame, slot, counter)` (WESL `rng01`).
    pub fn rng01(frame: u32, slot: u32, counter: u32) -> f32 {
        let h = fmix32(
            frame
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(fmix32(slot.wrapping_mul(0x85eb_ca6b).wrapping_add(counter))),
        );
        (h >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Picks a light index in `[0, light_count)` for candidate `k` (WESL
    /// `pick_light`); callers guarantee `light_count > 0`.
    pub fn pick_light(frame: u32, slot: u32, k: u32, light_count: u32) -> u32 {
        fmix32(
            frame
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(slot.wrapping_mul(0x85eb_ca6b))
                .wrapping_add(k.wrapping_mul(0x1656_67b1)),
        ) % light_count
    }

    /// Rec. 709 luminance of a linear RGB triple (WESL `luminance`).
    fn luminance(rgb: Vec3) -> f32 {
        LUMA_R * rgb.x.max(0.0) + LUMA_G * rgb.y.max(0.0) + LUMA_B * rgb.z.max(0.0)
    }

    /// Surface-to-surface geometry factor `cos_v * cos_s / dist^2` (WESL
    /// `geometric_term`); zero for degenerate / back-facing / NaN pairs. Uses
    /// `dist_sq.sqrt().recip()` to reproduce the golden's exact IEEE reciprocal
    /// square root bit-for-bit.
    fn geometric_term(
        visible_point: Vec3,
        visible_normal: Vec3,
        sample_point: Vec3,
        sample_normal: Vec3,
    ) -> f32 {
        let delta = sample_point - visible_point;
        let dist_sq = delta.length_squared();
        if dist_sq > MIN_POSITIVE {
            let inv_dist = dist_sq.sqrt().recip();
            let dir = delta * inv_dist;
            let cos_v = visible_normal.dot(dir).max(0.0);
            let cos_s = sample_normal.dot(-dir).max(0.0);
            let term = cos_v * cos_s / dist_sq;
            // `!term.is_nan()` mirrors the WESL `term == term` NaN reject.
            if !term.is_nan() {
                return term.max(0.0);
            }
        }
        0.0
    }

    /// Scalar resampling target `p_hat` (WESL `target_function`).
    fn target_function(s: &GiSample) -> f32 {
        let g = geometric_term(
            s.visible_point,
            s.visible_normal,
            s.sample_point,
            s.sample_normal,
        );
        let t = luminance(s.radiance) * g;
        if t.is_nan() {
            0.0
        } else {
            t.max(0.0)
        }
    }

    /// Resampled-importance weight `target_function / source_pdf` (WESL
    /// `ris_weight`). The WESL guard is `!(source_pdf > 0.0)`; for the finite
    /// source pdfs the seed ever sees (`1 / light_count`) that is exactly
    /// `source_pdf <= 0.0`.
    fn ris_weight(s: &GiSample, source_pdf: f32) -> f32 {
        if source_pdf <= 0.0 {
            return 0.0;
        }
        let w = target_function(s) / source_pdf;
        if w.is_nan() {
            0.0
        } else {
            w.max(0.0)
        }
    }

    /// Builds a light candidate from the cell's visible geometry and one emitter
    /// (WESL `build_sample`): the secondary point is the emitter position, its
    /// normal faces the visible point, and the radiance is the emitter colour
    /// scaled by its intensity and the artistic gain. Shared verbatim by the
    /// golden and mirror sweeps so both resolve the identical `GiSample`.
    pub fn build_sample(
        visible_point: Vec3,
        visible_normal: Vec3,
        light: &Light,
        intensity: f32,
    ) -> GiSample {
        let sample_point = light.position;
        let delta = visible_point - sample_point;
        let dist_sq = delta.length_squared();
        let sample_normal = if dist_sq > MIN_POSITIVE {
            delta * dist_sq.sqrt().recip()
        } else {
            Vec3::Z
        };
        GiSample {
            visible_point,
            visible_normal,
            sample_point,
            sample_normal,
            radiance: light.color * (light.intensity * intensity),
        }
    }

    /// Hand-rolls the WESL `seed_main` streaming `RIS` loop for one occupied
    /// cell: draws `candidate_count` lights, folds each positive weight into the
    /// reservoir, caps the confidence, then finalizes `W` from the surviving
    /// sample's own target density. Bit-equal to the golden `stream_candidate` +
    /// `cap_confidence` + `finalize` chain on finite inputs.
    #[expect(
        clippy::too_many_arguments,
        reason = "arm-for-arm mirror of the WESL `seed_main` inputs; bundling them would obscure the one-to-one port"
    )]
    pub fn seed_cell(
        visible_point: Vec3,
        visible_normal: Vec3,
        lights: &[Light],
        candidate_count: u32,
        frame: u32,
        slot: u32,
        intensity: f32,
        m_cap: f32,
    ) -> SeedResult {
        let light_count = lights.len() as u32;
        let mut selected: Option<GiSample> = None;
        let mut w_sum = 0.0f32;
        let mut m = 0.0f32;
        if light_count > 0 {
            let source_pdf = 1.0 / light_count as f32;
            for k in 0..candidate_count {
                let li = pick_light(frame, slot, k, light_count) as usize;
                let candidate = build_sample(visible_point, visible_normal, &lights[li], intensity);
                let weight = ris_weight(&candidate, source_pdf);
                if weight > 0.0 {
                    w_sum += weight;
                    m += 1.0;
                    let u = rng01(frame, slot, k).clamp(0.0, 1.0);
                    if u * w_sum <= weight {
                        selected = Some(candidate);
                    }
                }
            }
        }
        if m_cap >= 0.0 && m > m_cap {
            m = m_cap;
        }
        let mut w = 0.0f32;
        if let Some(s) = selected {
            let p_hat = target_function(&s);
            // `!w_sum.is_nan()` mirrors the golden `w_sum.is_finite()` guard
            // (equivalent on the finite sweep); `p_hat > 0.0` is the finalize
            // positivity test.
            if m > 0.0 && !w_sum.is_nan() && p_hat > 0.0 {
                let cw = (w_sum / m) / p_hat;
                w = if !cw.is_nan() && cw >= 0.0 { cw } else { 0.0 };
            }
        }
        SeedResult {
            sample: selected,
            w,
            m,
        }
    }
}

/// Asserts the seed kernel's inlined `RIS` estimator (transcribed by
/// [`seed_mirror`]) reproduces the authoritative `Reservoir` golden
/// (`stream_candidate` / `cap_confidence` / `finalize`) bit-for-bit across a
/// sweep of cell geometries, light sets (bright / single / dark / coincident),
/// candidate budgets, `M`-caps, artistic gains, and frames. The golden and the
/// mirror share `rng01` / `pick_light` / `build_sample`, so the only variable
/// under test is the resampling arithmetic. This is the device-equivalence
/// proof for the seed port under the no-GPU sandbox.
#[test]
fn seed_shader_matches_cpu_golden() {
    use bevy_math::Vec3;
    use prism_render_shading::gi::screen_probe::restir::{GiSample, Reservoir};
    use prism_render_shading::gi::world_restir::world_reservoir::{finalize, stream_candidate};
    use seed_mirror::{build_sample, pick_light, rng01, seed_cell, Light};

    let cells: [(Vec3, Vec3); 5] = [
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Z),
        (Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0)),
        (Vec3::new(-4.0, 0.5, 2.0), Vec3::new(1.0, 0.0, 0.0)),
        (Vec3::new(2.5, -1.0, -3.0), Vec3::new(0.0, 0.0, -1.0)),
        (
            Vec3::new(-1.0, -2.0, 1.5),
            Vec3::new(1.0, 1.0, 1.0).normalize(),
        ),
    ];
    let cell0_vp = cells[0].0;
    let light_sets: [Vec<Light>; 4] = [
        // Bright: three well-separated emitters with distinct colours.
        vec![
            Light {
                position: Vec3::new(5.0, 4.0, 1.0),
                intensity: 3.0,
                color: Vec3::new(1.0, 0.8, 0.6),
            },
            Light {
                position: Vec3::new(-3.0, 6.0, -2.0),
                intensity: 1.5,
                color: Vec3::new(0.4, 0.9, 1.0),
            },
            Light {
                position: Vec3::new(0.0, -5.0, 4.0),
                intensity: 2.0,
                color: Vec3::new(0.7, 0.7, 0.2),
            },
        ],
        // Single emitter.
        vec![Light {
            position: Vec3::new(2.0, 3.0, -1.0),
            intensity: 4.0,
            color: Vec3::new(0.9, 0.5, 0.3),
        }],
        // Dark: zero radiance (zero intensity / zero colour) -> every candidate
        // weight is 0 -> no selection on either side.
        vec![
            Light {
                position: Vec3::new(1.0, 1.0, 1.0),
                intensity: 0.0,
                color: Vec3::new(1.0, 1.0, 1.0),
            },
            Light {
                position: Vec3::new(-2.0, 2.0, 3.0),
                intensity: 5.0,
                color: Vec3::ZERO,
            },
        ],
        // Coincident: the first emitter sits on cell 0's visible point
        // (degenerate geometry there -> zero geometric term -> skipped) plus one
        // healthy emitter.
        vec![
            Light {
                position: cell0_vp,
                intensity: 2.0,
                color: Vec3::new(1.0, 1.0, 1.0),
            },
            Light {
                position: Vec3::new(3.0, 1.0, -2.0),
                intensity: 1.0,
                color: Vec3::new(0.6, 0.8, 1.0),
            },
        ],
    ];
    let candidate_counts: [u32; 4] = [0, 1, 4, 16];
    let m_caps: [f32; 4] = [-1.0, 0.0, 2.0, 32.0];
    let intensities: [f32; 3] = [0.5, 1.0, 2.0];

    for (cell_idx, &(vp, vn)) in cells.iter().enumerate() {
        let slot = (cell_idx as u32) * 7 + 1;
        for lights in &light_sets {
            let light_count = lights.len() as u32;
            for &candidate_count in &candidate_counts {
                for &m_cap in &m_caps {
                    for &intensity in &intensities {
                        for frame in 0..4u32 {
                            // Golden: the authoritative Reservoir RIS path.
                            let mut r = Reservoir::<GiSample>::new();
                            if light_count > 0 {
                                let source_pdf = 1.0 / light_count as f32;
                                for k in 0..candidate_count {
                                    let li = pick_light(frame, slot, k, light_count) as usize;
                                    let sample = build_sample(vp, vn, &lights[li], intensity);
                                    stream_candidate(
                                        &mut r,
                                        sample,
                                        source_pdf,
                                        rng01(frame, slot, k),
                                    );
                                }
                            }
                            r.cap_confidence(m_cap);
                            finalize(&mut r);

                            // Mirror: the kernel's inlined estimator.
                            let mirror = seed_cell(
                                vp,
                                vn,
                                lights,
                                candidate_count,
                                frame,
                                slot,
                                intensity,
                                m_cap,
                            );

                            let ctx = (frame, slot, candidate_count);
                            assert_eq!(
                                r.contribution_weight().to_bits(),
                                mirror.w.to_bits(),
                                "W mismatch at {ctx:?} (m_cap={m_cap}, intensity={intensity}, lights={light_count})"
                            );
                            assert_eq!(
                                r.confidence().to_bits(),
                                mirror.m.to_bits(),
                                "m mismatch at {ctx:?} (m_cap={m_cap}, intensity={intensity}, lights={light_count})"
                            );
                            match (r.sample(), mirror.sample) {
                                (Some(g), Some(mi)) => {
                                    assert_vec3_bits(
                                        g.visible_point,
                                        mi.visible_point,
                                        "visible_point",
                                        ctx,
                                    );
                                    assert_vec3_bits(
                                        g.visible_normal,
                                        mi.visible_normal,
                                        "visible_normal",
                                        ctx,
                                    );
                                    assert_vec3_bits(
                                        g.sample_point,
                                        mi.sample_point,
                                        "sample_point",
                                        ctx,
                                    );
                                    assert_vec3_bits(
                                        g.sample_normal,
                                        mi.sample_normal,
                                        "sample_normal",
                                        ctx,
                                    );
                                    assert_vec3_bits(g.radiance, mi.radiance, "radiance", ctx);
                                }
                                (None, None) => {}
                                (g, mi) => {
                                    panic!("selection divergence at {ctx:?}: golden={g:?} mirror={mi:?}")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Serial CPU mirror of `world_restir_inject.wesl`'s `inject_main` open-address
/// slot claim. `state[idx]` is `None` for an empty slot or `Some(checksum)` for
/// a claimed one, reproducing the shader's `atomic<u32>` `slot_state`
/// (`EMPTY_SLOT = 0`). The probe base, step bound (`PROBE_LIMIT`), and checksum
/// are the golden `spatial_hash` / `WorldHashGrid::find_or_alloc` values, so the
/// claimed slot sequence is bit-identical to the device path.
mod inject_mirror {
    use bevy_math::Vec3;
    use prism_render_shading::gi::world_restir::spatial_hash::{self, HashGridKey, HashGridParams};
    use prism_render_shading::gi::world_restir::world_reservoir::PROBE_LIMIT;

    /// The outcome of claiming a slot for one injected visible point.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Outcome {
        /// First invocation to own the cell: it claims `state[idx]` and
        /// pre-seeds the reservoir (shader `claim.exchanged` branch).
        Fresh(usize),
        /// The cell was already claimed by a prior point (shader `owner == cs`
        /// first-wins branch); the slot is left untouched.
        Reuse(usize),
        /// The probe window was full of other cells; the point is dropped
        /// rather than evicting a live cell.
        Drop,
    }

    /// Hashes `(position, normal)` into its `SHARC` cell and claims the slot
    /// owning that cell under the shader's linear probe, mutating `state`.
    /// Returns the claim outcome plus the resolved key and its checksum.
    pub fn claim(
        state: &mut [Option<u32>],
        position: Vec3,
        normal: Vec3,
        camera: Vec3,
        params: &HashGridParams,
    ) -> (Outcome, HashGridKey, u32) {
        let cap = state.len() as u32;
        let key = spatial_hash::compute_key(position, normal, camera, params);
        let cs = spatial_hash::checksum(&key);
        let base = spatial_hash::bucket_index(&key, cap);
        let steps = PROBE_LIMIT.min(cap);
        for i in 0..steps {
            let idx = ((base + i) % cap) as usize;
            match state[idx] {
                None => {
                    state[idx] = Some(cs);
                    return (Outcome::Fresh(idx), key, cs);
                }
                Some(owner) if owner == cs => return (Outcome::Reuse(idx), key, cs),
                Some(_) => {}
            }
        }
        (Outcome::Drop, key, cs)
    }
}

/// Device-equivalence proof for `world_restir_inject.wesl` under the no-GPU
/// sandbox. The shader's atomic open-address claim is mirrored on the CPU
/// ([`inject_mirror::claim`]) and cross-checked against the authoritative
/// `WorldHashGrid::find_or_alloc` (reached through `insert_candidate`), which
/// shares the identical probe base, step bound, and checksum. For a
/// deterministic point stream with repeated cells (driving the first-wins
/// `Reuse` path) across table capacities that force collisions and
/// probe-window-full `Drop`s, it asserts:
///
/// * every checksum is non-zero and distinct cells get distinct checksums, so
///   the shader's `EMPTY_SLOT = 0` sentinel is never ambiguous with a claim;
/// * each `Fresh` claim allocates exactly one new grid cell while `Reuse` /
///   `Drop` allocate none, so grid occupancy equals the distinct-claimed count;
/// * a cell's reservoir is present in the grid iff the mirror claimed it;
/// * the pre-seeded reservoir is byte-for-byte what `make_reservoir` writes and
///   `seed_main` reads back (`visible_point` / `visible_normal` geometry, the
///   `checksum`, `valid = 1`, and zeroed `w` / `m` / sample / radiance lanes).
#[test]
fn inject_shader_matches_cpu_golden() {
    use super::abi::GpuWorldRestirReservoir;
    use bevy_math::Vec3;
    use inject_mirror::{claim, Outcome};
    use prism_render_shading::gi::screen_probe::restir::GiSample;
    use prism_render_shading::gi::world_restir::spatial_hash::HashGridParams;
    use prism_render_shading::gi::world_restir::world_reservoir::WorldHashGrid;

    let camera = Vec3::new(0.0, 1.5, 4.0);
    let params = HashGridParams::DEFAULT;

    // Deterministic visible-point stream: six well-separated cells with three
    // exact repeats interleaved so the first-wins `Reuse` branch is exercised.
    let points: [(Vec3, Vec3); 10] = [
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        (Vec3::new(3.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::Z),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        (Vec3::new(0.0, 0.0, -3.0), Vec3::new(0.0, -1.0, 0.0)),
        (Vec3::new(3.0, 0.0, 0.0), Vec3::X),
        (
            Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0).normalize(),
        ),
        (
            Vec3::new(-3.0, 0.0, 3.0),
            Vec3::new(-1.0, 0.0, 1.0).normalize(),
        ),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::Z),
    ];

    for &cap in &[1u32, 4, 8, 131_072] {
        let mut state: Vec<Option<u32>> = vec![None; cap as usize];
        let mut grid = WorldHashGrid::new(cap);
        // Distinct resolved cells in first-seen order: (key, checksum, claimed?).
        let mut seen: Vec<(HashGridKey, u32, bool)> = Vec::new();
        let mut fresh_count = 0usize;

        for &(pos, normal) in &points {
            let (outcome, key, cs) = claim(&mut state, pos, normal, camera, &params);
            assert_ne!(
                cs, 0,
                "checksum must be non-zero so EMPTY_SLOT=0 is unambiguous (cap={cap}, key={key:?})"
            );

            // The authoritative grid shares the identical probe window and
            // checksum, so `find_or_alloc` (via `insert_candidate`) allocates a
            // cell exactly when the mirror reports `Fresh`.
            let before = grid.occupied_len();
            let dummy = GiSample {
                visible_point: pos,
                visible_normal: normal,
                ..GiSample::ZERO
            };
            grid.insert_candidate(&key, dummy, 1.0, 0.5);
            let after = grid.occupied_len();

            match outcome {
                Outcome::Fresh(_) => {
                    fresh_count += 1;
                    assert_eq!(
                        after,
                        before + 1,
                        "fresh claim must allocate one grid cell (cap={cap}, key={key:?})"
                    );
                }
                Outcome::Reuse(_) | Outcome::Drop => {
                    assert_eq!(
                        after, before,
                        "reuse/drop must not allocate (cap={cap}, outcome={outcome:?}, key={key:?})"
                    );
                }
            }

            let claimed_now = matches!(outcome, Outcome::Fresh(_) | Outcome::Reuse(_));
            match seen.iter_mut().find(|(k, _, _)| *k == key) {
                Some((_, prev_cs, prev_claimed)) => {
                    assert_eq!(
                        *prev_cs, cs,
                        "a cell's checksum must be stable (cap={cap}, key={key:?})"
                    );
                    // No eviction, so a cell's claim state is deterministic
                    // across repeats (claimed stays claimed, dropped stays
                    // dropped).
                    assert_eq!(
                        *prev_claimed, claimed_now,
                        "a cell's claim state must be stable across repeats (cap={cap}, key={key:?})"
                    );
                }
                None => seen.push((key, cs, claimed_now)),
            }
        }

        // Distinct cells carry distinct checksums in this sweep, so the shader's
        // EMPTY_SLOT remap can never be reached.
        let mut checksums: Vec<u32> = seen.iter().map(|&(_, cs, _)| cs).collect();
        let distinct = checksums.len();
        checksums.sort_unstable();
        checksums.dedup();
        assert_eq!(
            checksums.len(),
            distinct,
            "distinct cells must have distinct checksums (cap={cap})"
        );

        // Grid occupancy equals the number of distinctly-claimed cells, and each
        // cell's reservoir is present iff the mirror claimed it.
        let claimed_cells = seen.iter().filter(|&&(_, _, c)| c).count();
        assert_eq!(
            grid.occupied_len(),
            claimed_cells,
            "grid occupancy must equal distinct claimed cells (cap={cap})"
        );
        assert_eq!(
            fresh_count, claimed_cells,
            "exactly one Fresh per distinct claimed cell (cap={cap})"
        );
        for (key, _, claimed) in &seen {
            assert_eq!(
                grid.reservoir(key).is_some(),
                *claimed,
                "grid reservoir presence must match the mirror claim (cap={cap}, key={key:?})"
            );
        }
    }

    // Pre-seeded reservoir bit-exactness: construct what `make_reservoir` writes
    // for a fresh claim and assert it is byte-for-byte the `seed_main`-readable
    // record (geometry + checksum + valid flag, every other lane zeroed).
    let (pos, normal) = points[0];
    let key = spatial_hash::compute_key(pos, normal, camera, &params);
    let cs = spatial_hash::checksum(&key);
    let injected = GpuWorldRestirReservoir {
        visible_point: pos.to_array(),
        w: 0.0,
        visible_normal: normal.to_array(),
        m: 0.0,
        sample_point: [0.0; 3],
        checksum: cs,
        sample_normal: [0.0; 3],
        valid: 1,
        radiance: [0.0; 3],
        _pad0: 0,
    };
    assert_eq!(
        injected.valid, 1,
        "injected slot must be marked valid for seed_main"
    );
    assert_eq!(
        injected.checksum, cs,
        "injected checksum must be the key's SHARC checksum"
    );
    assert_ne!(cs, 0, "the injected checksum must be non-zero");
    assert_eq!(
        injected.w.to_bits(),
        0.0_f32.to_bits(),
        "no W is written at injection"
    );
    assert_eq!(
        injected.m.to_bits(),
        0.0_f32.to_bits(),
        "no confidence is written at injection"
    );
    assert_eq!(injected._pad0, 0, "the std430 pad word stays zero");
    for (got, want) in injected.visible_point.iter().zip(pos.to_array().iter()) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "visible_point must round-trip the cell geometry"
        );
    }
    for (got, want) in injected.visible_normal.iter().zip(normal.to_array().iter()) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "visible_normal must round-trip the cell geometry"
        );
    }
    for got in injected.sample_point {
        assert_eq!(
            got.to_bits(),
            0.0_f32.to_bits(),
            "no light-sample point at injection"
        );
    }
    for got in injected.sample_normal {
        assert_eq!(
            got.to_bits(),
            0.0_f32.to_bits(),
            "no light-sample normal at injection"
        );
    }
    for got in injected.radiance {
        assert_eq!(got.to_bits(), 0.0_f32.to_bits(), "no radiance at injection");
    }
}
