//! Real-device parity for the fixed-width `u32` bit-pack twin:
//! [`GpuBitPackU32`](prism_volumetric_gpu::bit_pack_u32::GpuBitPackU32) must
//! reproduce the `CPU` golden
//! [`bit_pack_u32`](prism_render_architecture::particle::bit_pack_u32) element
//! for element across both the forward `pack` and the reverse `unpack`
//! transforms, plus the full round trip.
//!
//! The fixtures cover the full span of edge widths — `bits == 1` (bitmap),
//! `7`, `8`, `16` (exact two-per-word), `31` (every value crosses a boundary),
//! and `32` (the verbatim-copy word-boundary degenerate case) — alongside the
//! empty input, a single value, values deliberately straddling word boundaries,
//! the all-`0` stream, the all-`1` (mask-saturated) stream, and a deterministic
//! integer-`LCG` random sweep. The `LCG` lives on the host in `u64`, exactly as
//! the golden test fixture; the kernels themselves stay pure `u32`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every transform is pure `u32` bit algebra with no rounding anywhere, so
//! `CPU` and `GPU` must agree bit for bit. The comparison is an exact `==` on
//! every `u32` output, with no tolerance: any mismatch is a genuine port bug.
//! `WGSL` has no `u64`, and the kernel is already `u32`-only, so nothing is out
//! of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_pack_u32`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::bit_pack_u32::{pack, packed_len_words, unpack};
use prism_volumetric_gpu::bit_pack_u32::GpuBitPackU32;
use prism_volumetric_gpu::GpuContext;

/// Returns a mask with the low `bits` bits set (`bits` in `1..=32`), avoiding a
/// `1u32 << 32` shift for `bits == 32`.
fn low_mask(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

/// Small linear-congruential generator for deterministic random samples, mirroring
/// the golden fixture. The `u64` state lives on the host; the kernel stays `u32`.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes constants.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }
}

/// The edge bit widths the house rules call out: `1`, `7`, `8`, `16`, `31` and
/// the `32`-bit integer-word degenerate case.
const EDGE_BITS: [u32; 6] = [1, 7, 8, 16, 31, 32];

#[test]
fn pack_matches_reference_across_edge_widths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    for bits in EDGE_BITS {
        let mask = low_mask(bits);
        // 37 values is deliberately non-aligned to the word size for every width.
        let values: Vec<u32> = (0..37)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B1) & mask)
            .collect();
        let got = gpu.pack(&ctx, &values, bits);
        let expected = pack(&values, bits);
        assert_eq!(got.len(), packed_len_words(values.len(), bits));
        assert_eq!(got, expected, "pack mismatch at bits = {bits}");
    }
}

#[test]
fn unpack_matches_reference_across_edge_widths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    for bits in EDGE_BITS {
        let mask = low_mask(bits);
        let values: Vec<u32> = (0..37)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B1) & mask)
            .collect();
        let packed = pack(&values, bits);
        let got = gpu.unpack(&ctx, &packed, bits, values.len());
        assert_eq!(got, values, "unpack mismatch at bits = {bits}");
    }
}

#[test]
fn roundtrip_matches_reference_all_bit_widths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // Every width 1..=32: pack on device, unpack on device, must recover inputs.
    for bits in 1..=32u32 {
        let mask = low_mask(bits);
        let values: Vec<u32> = (0..41)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761) & mask)
            .collect();
        let packed = gpu.pack(&ctx, &values, bits);
        assert_eq!(
            packed,
            pack(&values, bits),
            "pack mismatch at bits = {bits}"
        );
        let got = gpu.unpack(&ctx, &packed, bits, values.len());
        assert_eq!(got, values, "roundtrip mismatch at bits = {bits}");
    }
}

#[test]
fn single_value_roundtrips_every_width() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    for bits in 1..=32u32 {
        let value = 0x9E37_79B9 & low_mask(bits);
        let packed = gpu.pack(&ctx, &[value], bits);
        assert_eq!(packed, pack(&[value], bits), "single pack bits = {bits}");
        assert_eq!(
            gpu.unpack(&ctx, &packed, bits, 1),
            vec![value],
            "single unpack bits = {bits}"
        );
    }
}

#[test]
fn value_spanning_word_boundary_explicit_layout() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // 7 values of 5 bits: the 7th starts at bit 30 and spills 3 bits into word 1.
    let values = [0u32, 0, 0, 0, 0, 0, 0x1F];
    let packed = gpu.pack(&ctx, &values, 5);
    assert_eq!(packed.len(), 2); // 35 bits -> 2 words
    assert_eq!(packed[0], 0xC000_0000); // bits 30,31 set
    assert_eq!(packed[1], 0b111); // top 3 bits of 0x1F
    assert_eq!(gpu.unpack(&ctx, &packed, 5, values.len()), values.to_vec());
}

#[test]
fn cross_word_widths_roundtrip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // 3, 12, 17 all force values to straddle word boundaries at various offsets.
    for bits in [3u32, 12, 17] {
        let mask = low_mask(bits);
        let values: Vec<u32> = (0..30).map(|i| (i as u32 * 1103) & mask).collect();
        let packed = gpu.pack(&ctx, &values, bits);
        assert_eq!(packed, pack(&values, bits), "cross-word pack bits = {bits}");
        assert_eq!(
            gpu.unpack(&ctx, &packed, bits, values.len()),
            values,
            "cross-word unpack bits = {bits}"
        );
    }
}

#[test]
fn all_zero_and_all_ones_streams() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    for bits in EDGE_BITS {
        let mask = low_mask(bits);
        // All-zero stream: packs to all-zero words, unpacks back to zeros.
        let zeros = vec![0u32; 50];
        let packed_zero = gpu.pack(&ctx, &zeros, bits);
        assert_eq!(
            packed_zero,
            pack(&zeros, bits),
            "all-zero pack bits = {bits}"
        );
        assert_eq!(
            gpu.unpack(&ctx, &packed_zero, bits, zeros.len()),
            zeros,
            "all-zero unpack bits = {bits}"
        );
        // All-ones (mask-saturated) stream: every value fills all `bits` bits.
        let ones = vec![mask; 50];
        let packed_ones = gpu.pack(&ctx, &ones, bits);
        assert_eq!(
            packed_ones,
            pack(&ones, bits),
            "all-ones pack bits = {bits}"
        );
        assert_eq!(
            gpu.unpack(&ctx, &packed_ones, bits, ones.len()),
            ones,
            "all-ones unpack bits = {bits}"
        );
    }
}

#[test]
fn out_of_range_values_are_truncated() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // 3-bit width: 0xFF masks to 0x7, 0x08 masks to 0x0, matching the reference.
    let values = [0xFFu32, 0x08, 0x05];
    let packed = gpu.pack(&ctx, &values, 3);
    assert_eq!(packed, pack(&values, 3));
    assert_eq!(gpu.unpack(&ctx, &packed, 3, 3), vec![0x7u32, 0x0, 0x5]);
}

#[test]
fn bits32_is_verbatim_copy() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // bits == 32 degenerates to a straight word copy on both sides.
    let values = [0xDEAD_BEEFu32, 0x0000_0000, 0xFFFF_FFFF, 0x1234_5678];
    let packed = gpu.pack(&ctx, &values, 32);
    assert_eq!(packed, values.to_vec(), "bits=32 pack is verbatim");
    assert_eq!(
        gpu.unpack(&ctx, &packed, 32, values.len()),
        values.to_vec(),
        "bits=32 unpack is verbatim"
    );
}

#[test]
fn lcg_random_roundtrip_all_bit_widths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
    for bits in 1..=32u32 {
        let mask = low_mask(bits);
        let values: Vec<u32> = (0..200).map(|_| rng.next_u32() & mask).collect();
        let packed = gpu.pack(&ctx, &values, bits);
        assert_eq!(packed, pack(&values, bits), "random pack bits = {bits}");
        assert_eq!(
            gpu.unpack(&ctx, &packed, bits, values.len()),
            values,
            "random roundtrip bits = {bits}"
        );
    }
}

#[test]
fn unpack_fewer_than_packed_capacity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // Pack 20 values but decode only the first 5, as the reference permits.
    let values: Vec<u32> = (0..20).map(|i| i as u32 & 0xF).collect();
    let packed = gpu.pack(&ctx, &values, 4);
    assert_eq!(
        gpu.unpack(&ctx, &packed, 4, 5),
        unpack(&packed, 4, 5),
        "partial unpack mismatch"
    );
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitPackU32::new(&ctx);
    // No dispatch is issued and both entry points return an empty vector.
    assert!(gpu.pack(&ctx, &[], 7).is_empty());
    assert!(gpu.unpack(&ctx, &[], 7, 0).is_empty());
}
