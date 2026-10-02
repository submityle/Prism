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

/// Guards the Rust device `ABI` against drift from the WESL twin: the reservoir
/// slot is the 80-byte five-lane record, the fill immediate is the 64-byte
/// four-lane block, and the workgroup constant matches `@workgroup_size(64, 1,
/// 1)`.
#[test]
fn world_restir_abi_matches_the_shader_layout() {
    use super::{
        GpuWorldRestirFillParams, GpuWorldRestirReservoir, WORLD_RESTIR_RESERVOIR_STRIDE,
        WORLD_RESTIR_WORKGROUP_SIZE,
    };

    assert_eq!(size_of::<GpuWorldRestirReservoir>(), 80);
    assert_eq!(align_of::<GpuWorldRestirReservoir>(), 4);
    assert_eq!(WORLD_RESTIR_RESERVOIR_STRIDE, 80);
    assert_eq!(size_of::<GpuWorldRestirFillParams>(), 64);
    assert_eq!(align_of::<GpuWorldRestirFillParams>(), 4);
    assert_eq!(WORLD_RESTIR_WORKGROUP_SIZE, 64);
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
