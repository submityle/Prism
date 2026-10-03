//! GPU hardware-decode parity for the pure-CPU block-compressed texture codecs.
//!
//! Strategy: use the crate's own *encoders* to synthesise valid blocks of a
//! known mode, then decode the identical bytes two ways and compare:
//!   * the pure-CPU decoder in `prism_render_material::texture_codec`, and
//!   * the platform's native hardware texture-decompression unit, driven by
//!     [`prism_render_material_gpu::BlockOracle`].
//!
//! Because both paths decode the *same* bytes, encoder quality is irrelevant;
//! the test isolates decoder agreement against hardware ground truth. On a host
//! without a usable adapter the oracle returns `None` and the test skips.

use prism_render_material::{
    decode_astc_4x4_hdr, decode_astc_4x4_ldr, decode_astc_4x4_weights, decode_astc_4x4_weights_ise,
    decode_astc_void_extent_hdr, decode_astc_void_extent_ldr, decode_bc1, decode_bc3,
    decode_bc6h_mode10_signed, decode_bc6h_mode10_unsigned, decode_bc6h_mode12_signed,
    decode_bc6h_mode12_unsigned, decode_bc6h_mode13_signed, decode_bc6h_mode13_unsigned,
    decode_bc6h_mode14_signed, decode_bc6h_mode14_unsigned, decode_bc6h_mode1_signed,
    decode_bc6h_mode1_unsigned, decode_bc6h_mode2_signed, decode_bc6h_mode2_unsigned,
    decode_bc6h_mode3_signed, decode_bc6h_mode3_unsigned, decode_bc6h_mode4_signed,
    decode_bc6h_mode4_unsigned, decode_bc6h_mode5_signed, decode_bc6h_mode5_unsigned,
    decode_bc6h_mode6_signed, decode_bc6h_mode6_unsigned, decode_bc6h_mode7_signed,
    decode_bc6h_mode7_unsigned, decode_bc6h_mode8_signed, decode_bc6h_mode8_unsigned,
    decode_bc6h_mode9_signed, decode_bc6h_mode9_unsigned, decode_bc6h_signed, decode_bc6h_unsigned,
    decode_bc7, decode_bc7_mode0, decode_bc7_mode1, decode_bc7_mode2, decode_bc7_mode3,
    decode_bc7_mode7, encode_bc1, encode_bc3, encode_bc6h_mode11_unsigned, encode_bc7_mode4,
    encode_bc7_mode5, encode_bc7_mode6,
};
use prism_render_material_gpu::BlockOracle;
use wgpu::{Features, TextureFormat};

/// Tiny deterministic xorshift32 PRNG (tests must be reproducible).
struct Rng(u32);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }
    /// A finite, non-negative f16 bit pattern (sign=0, exponent kept moderate so
    /// the value is neither Inf/NaN nor denormal-tiny).
    fn half_bits(&mut self) -> u16 {
        let mant = (self.next_u32() & 0x3ff) as u16;
        // exponent field in [8, 22] -> roughly [2^-7, 2^7], always finite.
        let exp = (8 + (self.next_u32() % 15)) as u16;
        (exp << 10) | mant
    }
}

fn rgba_tile(rng: &mut Rng) -> [[u8; 4]; 16] {
    let mut tile = [[0u8; 4]; 16];
    for texel in &mut tile {
        texel[0] = rng.byte();
        texel[1] = rng.byte();
        texel[2] = rng.byte();
        texel[3] = rng.byte();
    }
    tile
}

fn hdr_tile(rng: &mut Rng) -> [[u16; 3]; 16] {
    let mut tile = [[0u16; 3]; 16];
    for texel in &mut tile {
        texel[0] = rng.half_bits();
        texel[1] = rng.half_bits();
        texel[2] = rng.half_bits();
    }
    tile
}

/// Max absolute per-channel difference between two `[[u8; 4]; 16]` tiles.
fn max_abs_u8(a: &[[u8; 4]; 16], b: &[[u8; 4]; 16]) -> i32 {
    let mut m = 0i32;
    for (ta, tb) in a.iter().zip(b.iter()) {
        for c in 0..4 {
            m = m.max((i32::from(ta[c]) - i32::from(tb[c])).abs());
        }
    }
    m
}

#[test]
fn bc_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping parity test");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x1234_5678);

    // ---- BC7 (modes 4/5/6): spec-exact weight tables, expect <= 1 LSB. ----
    for _ in 0..64 {
        for encode in [encode_bc7_mode6, encode_bc7_mode5, encode_bc7_mode4] {
            let tile = rgba_tile(&mut rng);
            let block = encode(&tile);
            let cpu = decode_bc7(&block).expect("encoder emits a decodable BC7 block");
            let gpu = oracle.decode_unorm8(TextureFormat::Bc7RgbaUnorm, &block);
            let d = max_abs_u8(&cpu, &gpu);
            assert!(
                d <= 1,
                "BC7 CPU vs GPU diff {d} > 1 LSB for block {block:?}"
            );
        }
    }

    // ---- BC1: 565 endpoints + 2-bit indices. GPU interpolation rounding is
    // implementation-defined, so allow a small tolerance on RGB; the opaque
    // path keeps alpha at 255 on both sides. ----
    for _ in 0..64 {
        let tile = rgba_tile(&mut rng);
        let block = encode_bc1(&tile);
        let cpu = decode_bc1(&block);
        let gpu = oracle.decode_unorm8(TextureFormat::Bc1RgbaUnorm, &block);
        let d = max_abs_u8(&cpu, &gpu);
        assert!(d <= 4, "BC1 CPU vs GPU diff {d} > 4 for block {block:?}");
    }

    // ---- BC3: BC1 color + BC4 alpha ramp. ----
    for _ in 0..64 {
        let tile = rgba_tile(&mut rng);
        let block = encode_bc3(&tile);
        let cpu = decode_bc3(&block);
        let gpu = oracle.decode_unorm8(TextureFormat::Bc3RgbaUnorm, &block);
        let d = max_abs_u8(&cpu, &gpu);
        assert!(d <= 4, "BC3 CPU vs GPU diff {d} > 4 for block {block:?}");
    }

    // ---- BC6H mode 11 (unsigned HDR): compare decoded f16 values. ----
    for _ in 0..64 {
        let tile = hdr_tile(&mut rng);
        let block = encode_bc6h_mode11_unsigned(&tile);
        let cpu = decode_bc6h_unsigned(&block).expect("mode-11 block decodes");
        let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbUfloat, &block);
        for (tc, tg) in cpu.iter().zip(gpu.iter()) {
            for c in 0..3 {
                let (a, b) = (tc[c], tg[c]);
                let tol = 1e-2 * a.abs().max(b.abs()).max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "BC6H CPU {a} vs GPU {b} exceeds tol {tol} for block {block:?}"
                );
            }
        }
    }
}

/// BPTC 2-subset anchor table (Khronos Data Format Spec). Mirrors the private
/// `BPTC_ANCHORS_2` in `prism_render_material`; the oracle test owns its own
/// copy so it can assemble spec-correct mode-1 blocks (there is no mode-1
/// encoder). `ANCHORS_2[p]` is the texel holding subset 1's fixed-high-bit-zero
/// index; subset 0's anchor is always texel 0.
#[rustfmt::skip]
const ANCHORS_2: [usize; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,
     2,  8,  2,  2,  8,  8,  2,  2,
    15, 15,  6,  8,  2,  8, 15, 15,
     2,  8,  2,  2,  2, 15, 15,  6,
     6,  2,  6,  8, 15, 15,  2,  2,
    15, 15, 15, 15, 15,  2,  2, 15,
];

/// LSB-first bit writer for assembling 128-bit BC7 blocks in tests.
struct BlockWriter {
    bytes: [u8; 16],
    pos: usize,
}

impl BlockWriter {
    fn new() -> Self {
        Self {
            bytes: [0u8; 16],
            pos: 0,
        }
    }
    fn write(&mut self, value: u32, n: u32) {
        for i in 0..n {
            if (value >> i) & 1 == 1 {
                self.bytes[self.pos / 8] |= 1 << (self.pos % 8);
            }
            self.pos += 1;
        }
    }
}

/// Assemble a two-subset RGB BC7 mode-1 block for `partition`, with four 6-bit
/// RGB endpoints, two shared P-bits, and sixteen indices (anchor texels carry a
/// 2-bit index, all others 3-bit) laid out exactly as the Khronos spec and the
/// CPU decoder expect.
fn make_mode1_block(
    partition: usize,
    rgb: [[u32; 3]; 4],
    pbit: [u32; 2],
    idx: [u8; 16],
) -> [u8; 16] {
    let mut w = BlockWriter::new();
    w.write(0b10, 2); // mode-1 unary marker: a 0 then a 1.
    w.write(partition as u32, 6);
    for e in rgb {
        w.write(e[0], 6);
    }
    for e in rgb {
        w.write(e[1], 6);
    }
    for e in rgb {
        w.write(e[2], 6);
    }
    w.write(pbit[0], 1);
    w.write(pbit[1], 1);
    let anchor1 = ANCHORS_2[partition];
    for (t, &i) in idx.iter().enumerate() {
        let n = if t == 0 || t == anchor1 { 2 } else { 3 };
        w.write(u32::from(i), n);
    }
    assert_eq!(w.pos, 128, "mode-1 block must be exactly 128 bits");
    w.bytes
}

/// BC7 mode 1 (two-subset RGB) parity against GPU hardware across **all 64
/// partitions**. There is no mode-1 encoder, so the test synthesises blocks
/// directly; a wrong partition row in the CPU decoder would surface here as a
/// per-texel subset mismatch versus the hardware unit.
#[test]
fn bc7_mode1_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping mode-1 parity");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x0BAD_F00D);
    for (partition, &anchor1) in ANCHORS_2.iter().enumerate() {
        // A couple of random endpoint/index draws per partition widens coverage
        // of the subset -> endpoint mapping and the interpolation ramp.
        for _ in 0..3 {
            let mut rgb = [[0u32; 3]; 4];
            for e in &mut rgb {
                e[0] = rng.next_u32() & 0x3f;
                e[1] = rng.next_u32() & 0x3f;
                e[2] = rng.next_u32() & 0x3f;
            }
            let pbit = [rng.next_u32() & 1, rng.next_u32() & 1];
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 || t == anchor1 {
                    (rng.next_u32() & 0x3) as u8 // 2-bit anchor index
                } else {
                    (rng.next_u32() & 0x7) as u8 // 3-bit index
                };
            }
            let block = make_mode1_block(partition, rgb, pbit, idx);
            let cpu = decode_bc7_mode1(&block);
            let via_dispatch = decode_bc7(&block).expect("mode-1 block decodes");
            assert_eq!(
                cpu, via_dispatch,
                "dispatch must match direct mode-1 decode"
            );
            let gpu = oracle.decode_unorm8(TextureFormat::Bc7RgbaUnorm, &block);
            let d = max_abs_u8(&cpu, &gpu);
            assert!(
                d <= 1,
                "BC7 mode-1 CPU vs GPU diff {d} > 1 LSB, partition {partition}\n cpu={cpu:?}\n gpu={gpu:?}\n block={block:?}"
            );
        }
    }
}

/// Assemble a two-subset RGB BC7 mode-3 block for `partition`, with four 7-bit
/// RGB endpoints, four per-endpoint P-bits, and sixteen 2-bit indices (anchor
/// texels carry a 1-bit index, all others 2-bit) laid out exactly as the
/// Khronos spec and the CPU decoder expect.
fn make_mode3_block(
    partition: usize,
    rgb: [[u32; 3]; 4],
    pbit: [u32; 4],
    idx: [u8; 16],
) -> [u8; 16] {
    let mut w = BlockWriter::new();
    w.write(0b1000, 4); // mode-3 unary marker: three 0s then a 1.
    w.write(partition as u32, 6);
    for e in rgb {
        w.write(e[0], 7);
    }
    for e in rgb {
        w.write(e[1], 7);
    }
    for e in rgb {
        w.write(e[2], 7);
    }
    for p in pbit {
        w.write(p, 1);
    }
    let anchor1 = ANCHORS_2[partition];
    for (t, &i) in idx.iter().enumerate() {
        let n = if t == 0 || t == anchor1 { 1 } else { 2 };
        w.write(u32::from(i), n);
    }
    assert_eq!(w.pos, 128, "mode-3 block must be exactly 128 bits");
    w.bytes
}

/// BC7 mode 3 (two-subset RGB, 7-bit endpoints + per-endpoint P-bit, 2-bit
/// indices) parity against GPU hardware across **all 64 partitions**. There is
/// no mode-3 encoder, so the test synthesises blocks directly; a wrong
/// partition row, anchor width, or P-bit assignment in the CPU decoder would
/// surface here as a per-texel mismatch versus the hardware unit.
#[test]
fn bc7_mode3_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping mode-3 parity");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x3C7B_1E55);
    for (partition, &anchor1) in ANCHORS_2.iter().enumerate() {
        for _ in 0..3 {
            let mut rgb = [[0u32; 3]; 4];
            for e in &mut rgb {
                e[0] = rng.next_u32() & 0x7f;
                e[1] = rng.next_u32() & 0x7f;
                e[2] = rng.next_u32() & 0x7f;
            }
            let pbit = [
                rng.next_u32() & 1,
                rng.next_u32() & 1,
                rng.next_u32() & 1,
                rng.next_u32() & 1,
            ];
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 || t == anchor1 {
                    (rng.next_u32() & 0x1) as u8 // 1-bit anchor index
                } else {
                    (rng.next_u32() & 0x3) as u8 // 2-bit index
                };
            }
            let block = make_mode3_block(partition, rgb, pbit, idx);
            let cpu = decode_bc7_mode3(&block);
            let via_dispatch = decode_bc7(&block).expect("mode-3 block decodes");
            assert_eq!(
                cpu, via_dispatch,
                "dispatch must match direct mode-3 decode"
            );
            let gpu = oracle.decode_unorm8(TextureFormat::Bc7RgbaUnorm, &block);
            let d = max_abs_u8(&cpu, &gpu);
            assert!(
                d <= 1,
                "BC7 mode-3 CPU vs GPU diff {d} > 1 LSB, partition {partition}\n cpu={cpu:?}\n gpu={gpu:?}\n block={block:?}"
            );
        }
    }
}

/// Assemble a two-subset RGBA BC7 mode-7 block for `partition`, with four 5-bit
/// RGBA endpoints, four per-endpoint P-bits, and sixteen 2-bit indices (anchor
/// texels carry a 1-bit index, all others 2-bit) shared by colour and alpha,
/// laid out exactly as the Khronos spec and the CPU decoder expect.
fn make_mode7_block(
    partition: usize,
    rgba: [[u32; 4]; 4],
    pbit: [u32; 4],
    idx: [u8; 16],
) -> [u8; 16] {
    let mut w = BlockWriter::new();
    w.write(0b1000_0000, 8); // mode-7 unary marker: seven 0s then a 1.
    w.write(partition as u32, 6);
    for e in rgba {
        w.write(e[0], 5);
    }
    for e in rgba {
        w.write(e[1], 5);
    }
    for e in rgba {
        w.write(e[2], 5);
    }
    for e in rgba {
        w.write(e[3], 5);
    }
    for p in pbit {
        w.write(p, 1);
    }
    let anchor1 = ANCHORS_2[partition];
    for (t, &i) in idx.iter().enumerate() {
        let n = if t == 0 || t == anchor1 { 1 } else { 2 };
        w.write(u32::from(i), n);
    }
    assert_eq!(w.pos, 128, "mode-7 block must be exactly 128 bits");
    w.bytes
}

/// BC7 mode 7 (two-subset RGBA, 5-bit endpoints + per-endpoint P-bit, 2-bit
/// shared colour/alpha indices) parity against GPU hardware across **all 64
/// partitions**. There is no mode-7 encoder, so the test synthesises blocks
/// directly; a wrong partition row, anchor width, P-bit assignment, or alpha
/// path in the CPU decoder would surface here as a per-texel mismatch versus
/// the hardware unit.
#[test]
fn bc7_mode7_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping mode-7 parity");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x7A1E_9C3D);
    for (partition, &anchor1) in ANCHORS_2.iter().enumerate() {
        for _ in 0..3 {
            let mut rgba = [[0u32; 4]; 4];
            for e in &mut rgba {
                e[0] = rng.next_u32() & 0x1f;
                e[1] = rng.next_u32() & 0x1f;
                e[2] = rng.next_u32() & 0x1f;
                e[3] = rng.next_u32() & 0x1f;
            }
            let pbit = [
                rng.next_u32() & 1,
                rng.next_u32() & 1,
                rng.next_u32() & 1,
                rng.next_u32() & 1,
            ];
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 || t == anchor1 {
                    (rng.next_u32() & 0x1) as u8 // 1-bit anchor index
                } else {
                    (rng.next_u32() & 0x3) as u8 // 2-bit index
                };
            }
            let block = make_mode7_block(partition, rgba, pbit, idx);
            let cpu = decode_bc7_mode7(&block);
            let via_dispatch = decode_bc7(&block).expect("mode-7 block decodes");
            assert_eq!(
                cpu, via_dispatch,
                "dispatch must match direct mode-7 decode"
            );
            let gpu = oracle.decode_unorm8(TextureFormat::Bc7RgbaUnorm, &block);
            let d = max_abs_u8(&cpu, &gpu);
            assert!(
                d <= 1,
                "BC7 mode-7 CPU vs GPU diff {d} > 1 LSB, partition {partition}\n cpu={cpu:?}\n gpu={gpu:?}\n block={block:?}"
            );
        }
    }
}

/// Anchor index of subset 1 for 3-subset partitioning (Khronos "Fixup" table).
#[rustfmt::skip]
const ANCHORS_3_2: [usize; 64] = [
     3, 3,15,15, 8, 3,15,15, 8, 8, 6, 6, 6, 5, 3, 3,
     3, 3, 8,15, 3, 3, 6,10, 5, 8, 8, 6, 8, 5,15,15,
     8,15, 3, 5, 6,10, 8,15,15, 3,15, 5,15,15,15,15,
     3,15, 5, 5, 5, 8, 5,10, 5,10, 8,13,15,12, 3, 3,
];

/// Anchor index of subset 2 for 3-subset partitioning (Khronos "Fixup" table).
#[rustfmt::skip]
const ANCHORS_3_3: [usize; 64] = [
    15, 8, 8, 3,15,15, 3, 8,15,15,15,15,15,15,15, 8,
    15, 8,15, 3,15, 8,15, 8, 3,15, 6,10,15,15,10, 8,
    15, 3,15,10,10, 8, 9,10, 6,15, 8,15, 3, 6, 6, 8,
    15, 3,15,15,15,15,15,15,15,15,15,15, 3,15,15, 8,
];

/// Assemble a three-subset RGB BC7 mode-2 block for `partition`: six 5-bit RGB
/// endpoints, no P-bits, and sixteen 2-bit indices (the three anchor texels
/// carry a 1-bit index, all others 2-bit).
fn make_mode2_block(partition: usize, rgb: [[u32; 3]; 6], idx: [u8; 16]) -> [u8; 16] {
    let mut w = BlockWriter::new();
    w.write(0b100, 3); // mode-2 unary marker: two 0s then a 1.
    w.write(partition as u32, 6);
    for e in rgb {
        w.write(e[0], 5);
    }
    for e in rgb {
        w.write(e[1], 5);
    }
    for e in rgb {
        w.write(e[2], 5);
    }
    let a1 = ANCHORS_3_2[partition];
    let a2 = ANCHORS_3_3[partition];
    for (t, &i) in idx.iter().enumerate() {
        let n = if t == 0 || t == a1 || t == a2 { 1 } else { 2 };
        w.write(u32::from(i), n);
    }
    assert_eq!(w.pos, 128, "mode-2 block must be exactly 128 bits");
    w.bytes
}

/// BC7 mode 2 (three-subset RGB, 5-bit endpoints, no P-bits, 2-bit indices)
/// parity against GPU hardware across **all 64 partitions**. A wrong 3-subset
/// partition row or either anchor would surface as a per-texel mismatch.
#[test]
fn bc7_mode2_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping mode-2 parity");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x2ABC_D105);
    for partition in 0..64usize {
        let a1 = ANCHORS_3_2[partition];
        let a2 = ANCHORS_3_3[partition];
        for _ in 0..3 {
            let mut rgb = [[0u32; 3]; 6];
            for e in &mut rgb {
                e[0] = rng.next_u32() & 0x1f;
                e[1] = rng.next_u32() & 0x1f;
                e[2] = rng.next_u32() & 0x1f;
            }
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 || t == a1 || t == a2 {
                    (rng.next_u32() & 0x1) as u8 // 1-bit anchor index
                } else {
                    (rng.next_u32() & 0x3) as u8 // 2-bit index
                };
            }
            let block = make_mode2_block(partition, rgb, idx);
            let cpu = decode_bc7_mode2(&block);
            let via_dispatch = decode_bc7(&block).expect("mode-2 block decodes");
            assert_eq!(
                cpu, via_dispatch,
                "dispatch must match direct mode-2 decode"
            );
            let gpu = oracle.decode_unorm8(TextureFormat::Bc7RgbaUnorm, &block);
            let d = max_abs_u8(&cpu, &gpu);
            assert!(
                d <= 1,
                "BC7 mode-2 CPU vs GPU diff {d} > 1 LSB, partition {partition}\n cpu={cpu:?}\n gpu={gpu:?}\n block={block:?}"
            );
        }
    }
}

/// Assemble a three-subset RGB BC7 mode-0 block for `partition` (only 16
/// partitions): six 4-bit RGB endpoints, six per-endpoint P-bits, and sixteen
/// 3-bit indices (the three anchor texels carry a 2-bit index, all others
/// 3-bit).
fn make_mode0_block(
    partition: usize,
    rgb: [[u32; 3]; 6],
    pbit: [u32; 6],
    idx: [u8; 16],
) -> [u8; 16] {
    let mut w = BlockWriter::new();
    w.write(0b1, 1); // mode-0 unary marker: a single 1.
    w.write(partition as u32, 4);
    for e in rgb {
        w.write(e[0], 4);
    }
    for e in rgb {
        w.write(e[1], 4);
    }
    for e in rgb {
        w.write(e[2], 4);
    }
    for p in pbit {
        w.write(p, 1);
    }
    let a1 = ANCHORS_3_2[partition];
    let a2 = ANCHORS_3_3[partition];
    for (t, &i) in idx.iter().enumerate() {
        let n = if t == 0 || t == a1 || t == a2 { 2 } else { 3 };
        w.write(u32::from(i), n);
    }
    assert_eq!(w.pos, 128, "mode-0 block must be exactly 128 bits");
    w.bytes
}

/// BC7 mode 0 (three-subset RGB, 4-bit endpoints + per-endpoint P-bit, 3-bit
/// indices, 16 partitions) parity against GPU hardware across **all 16
/// partitions**. A wrong 3-subset partition row, anchor, or P-bit assignment
/// would surface as a per-texel mismatch versus the hardware unit.
#[test]
fn bc7_mode0_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping mode-0 parity");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x0DEF_0007);
    for partition in 0..16usize {
        let a1 = ANCHORS_3_2[partition];
        let a2 = ANCHORS_3_3[partition];
        for _ in 0..3 {
            let mut rgb = [[0u32; 3]; 6];
            for e in &mut rgb {
                e[0] = rng.next_u32() & 0xf;
                e[1] = rng.next_u32() & 0xf;
                e[2] = rng.next_u32() & 0xf;
            }
            let mut pbit = [0u32; 6];
            for p in &mut pbit {
                *p = rng.next_u32() & 1;
            }
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 || t == a1 || t == a2 {
                    (rng.next_u32() & 0x3) as u8 // 2-bit anchor index
                } else {
                    (rng.next_u32() & 0x7) as u8 // 3-bit index
                };
            }
            let block = make_mode0_block(partition, rgb, pbit, idx);
            let cpu = decode_bc7_mode0(&block);
            let via_dispatch = decode_bc7(&block).expect("mode-0 block decodes");
            assert_eq!(
                cpu, via_dispatch,
                "dispatch must match direct mode-0 decode"
            );
            let gpu = oracle.decode_unorm8(TextureFormat::Bc7RgbaUnorm, &block);
            let d = max_abs_u8(&cpu, &gpu);
            assert!(
                d <= 1,
                "BC7 mode-0 CPU vs GPU diff {d} > 1 LSB, partition {partition}\n cpu={cpu:?}\n gpu={gpu:?}\n block={block:?}"
            );
        }
    }
}

/// Assemble a single-subset BC6H delta block (modes 12/13/14) directly from
/// its fields: `mode_bits` is the 5-bit mode field, `base` the three
/// `base_prec`-bit base endpoint components (low 10 bits inline, high bits
/// relocated after each channel's delta), `delta` the three `delta_bits`-bit
/// signed-delta fields, `idx` the sixteen indices (texel 0 is the 3-bit anchor
/// with an implicit high zero, the other fifteen are 4-bit). There is no CPU
/// encoder for these modes, so the parity test owns this spec-exact assembler.
fn make_bc6h_delta_block(
    mode_bits: u32,
    base_prec: u32,
    delta_bits: u32,
    base: [u32; 3],
    delta: [u32; 3],
    idx: [u8; 16],
) -> [u8; 16] {
    let mut w = BlockWriter::new();
    let hi_bits = base_prec - 10;
    w.write(mode_bits, 5);
    // Base low 10 bits inline, then per channel the delta immediately followed
    // by that channel's relocated high base bits (bit 10..base_prec-1).
    w.write(base[0] & 0x3FF, 10);
    w.write(base[1] & 0x3FF, 10);
    w.write(base[2] & 0x3FF, 10);
    for c in 0..3 {
        w.write(delta[c], delta_bits);
        // High base bits most-significant-first.
        for k in (0..hi_bits).rev() {
            w.write((base[c] >> (10 + k)) & 1, 1);
        }
    }
    w.write(u32::from(idx[0]), 3);
    for &i in idx.iter().skip(1) {
        w.write(u32::from(i), 4);
    }
    assert_eq!(w.pos, 128, "BC6H delta fields must fill the block exactly");
    w.bytes
}

/// `(mode_bits, base_prec, delta_bits)` for the three single-subset delta modes.
const BC6H_DELTA_LAYOUTS: [(u32, u32, u32); 3] = [
    (0b00111, 11, 9), // mode 12
    (0b01011, 12, 8), // mode 13
    (0b01111, 16, 4), // mode 14
];

/// Compare `cpu` and `gpu` HDR texels with a relative tolerance (the hardware
/// unquantize/interpolate rounding is implementation-defined within an LSB).
fn assert_rgb_f32_close(cpu: &[[f32; 3]; 16], gpu: &[[f32; 3]; 16], ctx: &str) {
    for (tc, tg) in cpu.iter().zip(gpu.iter()) {
        for c in 0..3 {
            let (a, b) = (tc[c], tg[c]);
            let tol = 1e-2 * a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "BC6H CPU {a} vs GPU {b} exceeds tol {tol} ({ctx})"
            );
        }
    }
}

/// BC6H single-subset **unsigned** delta modes (12/13/14) parity against GPU
/// hardware. Each mode's inverse transform reconstructs endpoint 1 as
/// `(base + signed delta)` wrapped to the base precision; a wrong base/delta
/// field width, wrap mask, or unquantize precision would mismatch hardware.
#[test]
fn bc6h_single_subset_delta_unsigned_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping BC6H delta unsigned");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x6C12_0D01);
    for &(mode_bits, base_prec, delta_bits) in &BC6H_DELTA_LAYOUTS {
        let base_mask = if base_prec >= 32 {
            u32::MAX
        } else {
            (1u32 << base_prec) - 1
        };
        let delta_mask = (1u32 << delta_bits) - 1;
        for _ in 0..24 {
            let base = [
                rng.next_u32() & base_mask,
                rng.next_u32() & base_mask,
                rng.next_u32() & base_mask,
            ];
            let delta = [
                rng.next_u32() & delta_mask,
                rng.next_u32() & delta_mask,
                rng.next_u32() & delta_mask,
            ];
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 {
                    (rng.next_u32() & 0x7) as u8
                } else {
                    (rng.next_u32() & 0xf) as u8
                };
            }
            let block = make_bc6h_delta_block(mode_bits, base_prec, delta_bits, base, delta, idx);
            let cpu = decode_bc6h_unsigned(&block).expect("delta mode decodes");
            let direct = match mode_bits {
                0b00111 => decode_bc6h_mode12_unsigned(&block),
                0b01011 => decode_bc6h_mode13_unsigned(&block),
                _ => decode_bc6h_mode14_unsigned(&block),
            };
            assert_eq!(cpu, direct, "dispatch must match direct decode");
            let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbUfloat, &block);
            assert_rgb_f32_close(&cpu, &gpu, &format!("unsigned mode {mode_bits:#07b}"));
        }
    }
}

/// BC6H single-subset **signed** (`SF16`) delta modes (12/13/14) parity against
/// GPU hardware. Identical field layout to the unsigned path, but base and the
/// wrapped endpoint are sign-extended at the base precision before the signed
/// unquantize, so this isolates the signed transform + finish arithmetic.
#[test]
fn bc6h_single_subset_delta_signed_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU adapter unreachable in sandbox, graceful skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping BC6H delta signed");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x6C12_0D5E);
    for &(mode_bits, base_prec, delta_bits) in &BC6H_DELTA_LAYOUTS {
        let base_mask = if base_prec >= 32 {
            u32::MAX
        } else {
            (1u32 << base_prec) - 1
        };
        let delta_mask = (1u32 << delta_bits) - 1;
        for _ in 0..24 {
            let base = [
                rng.next_u32() & base_mask,
                rng.next_u32() & base_mask,
                rng.next_u32() & base_mask,
            ];
            let delta = [
                rng.next_u32() & delta_mask,
                rng.next_u32() & delta_mask,
                rng.next_u32() & delta_mask,
            ];
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = if t == 0 {
                    (rng.next_u32() & 0x7) as u8
                } else {
                    (rng.next_u32() & 0xf) as u8
                };
            }
            let block = make_bc6h_delta_block(mode_bits, base_prec, delta_bits, base, delta, idx);
            let cpu = decode_bc6h_signed(&block).expect("delta mode decodes");
            let direct = match mode_bits {
                0b00111 => decode_bc6h_mode12_signed(&block),
                0b01011 => decode_bc6h_mode13_signed(&block),
                _ => decode_bc6h_mode14_signed(&block),
            };
            assert_eq!(cpu, direct, "dispatch must match direct decode");
            let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbFloat, &block);
            assert_rgb_f32_close(&cpu, &gpu, &format!("signed mode {mode_bits:#07b}"));
        }
    }
}

// ---------------------------------------------------------------------------
// BC6H two-subset mode 1 GPU parity.
// ---------------------------------------------------------------------------

/// Header-field ids for the BC6H mode-1 descriptor (mirror of the private
/// `Bc6hField` in the decoder). The descriptor + assembler live here so the GPU
/// parity test drives the exact spec bit-layout independently of the decoder's
/// internal tables; a wrong layout would mismatch the hardware decode.
#[derive(Clone, Copy)]
enum F6 {
    Rw,
    Gw,
    Bw,
    Rx,
    Gx,
    Bx,
    Ry,
    Gy,
    By,
    Rz,
    Gz,
    Bz,
    D,
    M,
}

#[rustfmt::skip]
const BC6H_MODE1_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (Gy, 4), (By, 4), (Bz, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Rw, 8), (Rw, 9), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Gw, 8), (Gw, 9), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bw, 8), (Bw, 9), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Gz, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

/// Subset-1 anchor texel per 2-subset partition (first 32 rows apply to BC6H's
/// 5-bit partition index). Mirrors `BPTC_ANCHORS_2` in the decoder.
#[rustfmt::skip]
const BC6H_ANCHORS_2: [usize; 32] = [
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,
     2,  8,  2,  2,  8,  8,  2,  2,
];

/// Assemble a BC6H mode-1 block from its stored field values: `w` is the base
/// endpoint (subset 0 A), `x`/`y`/`z` the three raw 5-bit delta fields (subset
/// 0 B, subset 1 A, subset 1 B), `d` the 5-bit partition, and `idx` the sixteen
/// indices (subset anchors are 2-bit, the rest 3-bit).
fn make_bc6h_mode1_block(
    w: [u32; 3],
    x: [u32; 3],
    y: [u32; 3],
    z: [u32; 3],
    d: u32,
    idx: [u8; 16],
) -> [u8; 16] {
    let fv = [
        w[0], w[1], w[2], x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2], d, 0b00,
    ];
    let mut bw = BlockWriter::new();
    for &(f, bit) in BC6H_MODE1_DESC {
        bw.write((fv[f as usize] >> bit) & 1, 1);
    }
    let anchor1 = BC6H_ANCHORS_2[d as usize];
    for (t, &i) in idx.iter().enumerate() {
        let bits = if t == 0 || t == anchor1 { 2 } else { 3 };
        bw.write(u32::from(i), bits);
    }
    assert_eq!(
        bw.pos, 128,
        "BC6H mode-1 fields must fill the block exactly"
    );
    bw.bytes
}

/// Draw random mode-1 fields for `partition` and return the assembled block.
fn random_mode1_block(rng: &mut Rng, partition: u32) -> [u8; 16] {
    let w = [
        rng.next_u32() & 0x3FF,
        rng.next_u32() & 0x3FF,
        rng.next_u32() & 0x3FF,
    ];
    let x = [
        rng.next_u32() & 0x1F,
        rng.next_u32() & 0x1F,
        rng.next_u32() & 0x1F,
    ];
    let y = [
        rng.next_u32() & 0x1F,
        rng.next_u32() & 0x1F,
        rng.next_u32() & 0x1F,
    ];
    let z = [
        rng.next_u32() & 0x1F,
        rng.next_u32() & 0x1F,
        rng.next_u32() & 0x1F,
    ];
    let anchor1 = BC6H_ANCHORS_2[partition as usize];
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        *slot = if t == 0 || t == anchor1 {
            (rng.next_u32() & 0x3) as u8
        } else {
            (rng.next_u32() & 0x7) as u8
        };
    }
    make_bc6h_mode1_block(w, x, y, z, partition, idx)
}

/// BC6H two-subset **mode 1 unsigned** parity against GPU hardware. Sweeps all
/// 32 partitions with random 10-bit base + 5-bit deltas and random indices; a
/// wrong header-bit descriptor, partition/anchor table, delta transform, or
/// 3-bit interpolation would diverge from the hardware decode.
#[test]
fn bc6h_mode1_unsigned_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU unreachable, skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping BC6H mode1 unsigned");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x6C12_1001);
    for partition in 0..32u32 {
        for _ in 0..8 {
            let block = random_mode1_block(&mut rng, partition);
            let cpu = decode_bc6h_unsigned(&block).expect("mode-1 block decodes");
            let direct = decode_bc6h_mode1_unsigned(&block);
            assert_eq!(cpu, direct, "dispatch must match direct mode-1 decode");
            let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbUfloat, &block);
            assert_rgb_f32_close(&cpu, &gpu, &format!("mode1 unsigned partition {partition}"));
        }
    }
}

/// BC6H two-subset **mode 1 signed** (`SF16`) parity against GPU hardware. Same
/// layout as unsigned, but the base/endpoints are sign-extended at the base
/// precision before the signed unquantize + finish, isolating the signed path.
#[test]
fn bc6h_mode1_signed_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        #[expect(
            clippy::print_stderr,
            reason = "test diagnostic: GPU unreachable, skip"
        )]
        {
            eprintln!("no GPU adapter with BC support reachable; skipping BC6H mode1 signed");
        }
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));

    let mut rng = Rng(0x6C12_1051);
    for partition in 0..32u32 {
        for _ in 0..8 {
            let block = random_mode1_block(&mut rng, partition);
            let cpu = decode_bc6h_signed(&block).expect("mode-1 block decodes");
            let direct = decode_bc6h_mode1_signed(&block);
            assert_eq!(cpu, direct, "dispatch must match direct mode-1 decode");
            let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbFloat, &block);
            assert_rgb_f32_close(&cpu, &gpu, &format!("mode1 signed partition {partition}"));
        }
    }
}

// ---------------------------------------------------------------------------
// BC6H two-subset mode 2 GPU parity.
//
// Mode 2 (2-bit mode `0b01`): 7-bit base endpoint, 6/6/6 transformed deltas.
// The generic two-subset assembler below drives the exact spec bit-layout
// independently of the decoder's descriptor so a wrong transcription diverges
// from the hardware decode.
// ---------------------------------------------------------------------------

/// Assemble a BC6H two-subset block from a mode `desc`riptor and the fourteen
/// field values (`fv` indexed by [`F6`]), the 5-bit partition `d`, and sixteen
/// indices (subset anchors are 2-bit, the rest 3-bit). Shared by every mode.
fn make_bc6h_two_subset_block(
    desc: &[(F6, u8)],
    fv: &[u32; 14],
    d: u32,
    idx: [u8; 16],
) -> [u8; 16] {
    let mut bw = BlockWriter::new();
    for &(f, bit) in desc {
        bw.write((fv[f as usize] >> bit) & 1, 1);
    }
    let anchor1 = BC6H_ANCHORS_2[d as usize];
    for (t, &i) in idx.iter().enumerate() {
        let bits = if t == 0 || t == anchor1 { 2 } else { 3 };
        bw.write(u32::from(i), bits);
    }
    assert_eq!(
        bw.pos, 128,
        "BC6H two-subset fields must fill the block exactly"
    );
    bw.bytes
}

/// Draw random two-subset fields with a `base_mask` (base endpoint) and
/// `delta_mask` (three delta fields), writing `mode_bits` into the `M` slot.
fn random_two_subset_block(
    rng: &mut Rng,
    desc: &[(F6, u8)],
    partition: u32,
    mode_bits: u32,
    base_mask: u32,
    delta_mask: [u32; 3],
) -> [u8; 16] {
    let mut fv = [0u32; 14];
    for c in 0..3 {
        fv[c] = rng.next_u32() & base_mask;
    }
    for d in 3..12 {
        fv[d] = rng.next_u32() & delta_mask[(d - 3) % 3];
    }
    fv[12] = partition;
    fv[13] = mode_bits;
    let anchor1 = BC6H_ANCHORS_2[partition as usize];
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        *slot = if t == 0 || t == anchor1 {
            (rng.next_u32() & 0x3) as u8
        } else {
            (rng.next_u32() & 0x7) as u8
        };
    }
    make_bc6h_two_subset_block(desc, &fv, partition, idx)
}

#[rustfmt::skip]
const BC6H_MODE2_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (Gy, 5), (Gz, 4), (Gz, 5), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Bz, 0), (Bz, 1), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (By, 5), (Bz, 2), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bz, 3), (Bz, 5), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Rx, 5), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Gx, 5), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bx, 5), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Ry, 5), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Rz, 5), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

/// BC6H two-subset **mode 2 unsigned** parity against GPU hardware (7-bit base,
/// 6-bit deltas, 2-bit mode `0b01`). Sweeps all 32 partitions.
#[test]
fn bc6h_mode2_unsigned_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter with BC support reachable; skipping BC6H mode2 unsigned");
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));
    let mut rng = Rng(0x6C12_2001);
    for partition in 0..32u32 {
        for _ in 0..8 {
            let block = random_two_subset_block(
                &mut rng,
                BC6H_MODE2_DESC,
                partition,
                0b01,
                0x7F,
                [0x3F; 3],
            );
            let cpu = decode_bc6h_unsigned(&block).expect("mode-2 block decodes");
            let direct = decode_bc6h_mode2_unsigned(&block);
            assert_eq!(cpu, direct, "dispatch must match direct mode-2 decode");
            let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbUfloat, &block);
            assert_rgb_f32_close(&cpu, &gpu, &format!("mode2 unsigned partition {partition}"));
        }
    }
}

/// BC6H two-subset **mode 2 signed** (`SF16`) parity against GPU hardware.
#[test]
fn bc6h_mode2_signed_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter with BC support reachable; skipping BC6H mode2 signed");
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));
    let mut rng = Rng(0x6C12_2051);
    for partition in 0..32u32 {
        for _ in 0..8 {
            let block = random_two_subset_block(
                &mut rng,
                BC6H_MODE2_DESC,
                partition,
                0b01,
                0x7F,
                [0x3F; 3],
            );
            let cpu = decode_bc6h_signed(&block).expect("mode-2 block decodes");
            let direct = decode_bc6h_mode2_signed(&block);
            assert_eq!(cpu, direct, "dispatch must match direct mode-2 decode");
            let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbFloat, &block);
            assert_rgb_f32_close(&cpu, &gpu, &format!("mode2 signed partition {partition}"));
        }
    }
}

// ---------------------------------------------------------------------------
// BC6H two-subset modes 3-10 GPU parity (parametric).
//
// Each 5-bit mode carries its own DirectXTex ModeDescriptor (scrambled bit
// order), base precision and per-channel delta widths. The table below drives
// the shared assembler + both signed/unsigned decoders against the hardware
// oracle; a wrong descriptor diverges from the GPU decode.
// ---------------------------------------------------------------------------

/// One BC6H two-subset mode under test: name, 5-bit mode field, base precision,
/// per-channel (R,G,B) delta widths (0 == non-transformed raw endpoints), and
/// the 82-entry descriptor.
struct TwoSubsetSpec {
    name: &'static str,
    mode_bits: u32,
    base_prec: u32,
    delta: [u32; 3],
    desc: &'static [(F6, u8)],
}

#[rustfmt::skip]
const BC6H_MODE3_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Rw, 8), (Rw, 9), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Gw, 8), (Gw, 9), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bw, 8), (Bw, 9), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Rw, 10), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gw, 10),
        (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bw, 10),
        (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE4_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Rw, 8), (Rw, 9), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Gw, 8), (Gw, 9), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bw, 8), (Bw, 9), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rw, 10),
        (Gz, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Gw, 10), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bw, 10),
        (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Bz, 0),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Gy, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE5_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Rw, 8), (Rw, 9), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Gw, 8), (Gw, 9), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bw, 8), (Bw, 9), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rw, 10),
        (By, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gw, 10),
        (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bw, 10), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Bz, 1),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Bz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE6_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Rw, 8), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Gw, 8), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bw, 8), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Gz, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE7_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Gz, 4), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Bz, 2), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bz, 3), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Rx, 5), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Ry, 5), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Rz, 5), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE8_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Bz, 0), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (Gy, 5), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Gz, 5), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Gz, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Gx, 5), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE9_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Rw, 6), (Rw, 7), (Bz, 1), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gw, 6), (Gw, 7), (By, 5), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Bw, 6), (Bw, 7), (Bz, 5), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Gz, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bx, 5), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

#[rustfmt::skip]
const BC6H_MODE10_DESC: &[(F6, u8)] = {
    use F6::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    &[
        (M, 0), (M, 1), (M, 2), (M, 3), (M, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
        (Rw, 5), (Gz, 4), (Bz, 0), (Bz, 1), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
        (Gw, 5), (Gy, 5), (By, 5), (Bz, 2), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
        (Bw, 5), (Gz, 5), (Bz, 3), (Bz, 5), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
        (Rx, 5), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
        (Gx, 5), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
        (Bx, 5), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
        (Ry, 5), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Rz, 5), (D, 0), (D, 1), (D, 2),
        (D, 3), (D, 4),
    ]
};

const BC6H_TWO_SUBSET_SPECS: &[TwoSubsetSpec] = &[
    TwoSubsetSpec {
        name: "mode3",
        mode_bits: 0b00010,
        base_prec: 11,
        delta: [5, 4, 4],
        desc: BC6H_MODE3_DESC,
    },
    TwoSubsetSpec {
        name: "mode4",
        mode_bits: 0b00110,
        base_prec: 11,
        delta: [4, 5, 4],
        desc: BC6H_MODE4_DESC,
    },
    TwoSubsetSpec {
        name: "mode5",
        mode_bits: 0b01010,
        base_prec: 11,
        delta: [4, 4, 5],
        desc: BC6H_MODE5_DESC,
    },
    TwoSubsetSpec {
        name: "mode6",
        mode_bits: 0b01110,
        base_prec: 9,
        delta: [5, 5, 5],
        desc: BC6H_MODE6_DESC,
    },
    TwoSubsetSpec {
        name: "mode7",
        mode_bits: 0b10010,
        base_prec: 8,
        delta: [6, 5, 5],
        desc: BC6H_MODE7_DESC,
    },
    TwoSubsetSpec {
        name: "mode8",
        mode_bits: 0b10110,
        base_prec: 8,
        delta: [5, 6, 5],
        desc: BC6H_MODE8_DESC,
    },
    TwoSubsetSpec {
        name: "mode9",
        mode_bits: 0b11010,
        base_prec: 8,
        delta: [5, 5, 6],
        desc: BC6H_MODE9_DESC,
    },
    TwoSubsetSpec {
        name: "mode10",
        mode_bits: 0b11110,
        base_prec: 6,
        delta: [6, 6, 6],
        desc: BC6H_MODE10_DESC,
    },
];

/// Dispatch to the direct per-mode decoder so the parametric test also proves
/// the exported single-mode entry points match the generic dispatcher.
fn decode_two_subset_direct(mode_bits: u32, block: &[u8; 16], signed: bool) -> [[f32; 3]; 16] {
    match (mode_bits, signed) {
        (0b00010, false) => decode_bc6h_mode3_unsigned(block),
        (0b00010, true) => decode_bc6h_mode3_signed(block),
        (0b00110, false) => decode_bc6h_mode4_unsigned(block),
        (0b00110, true) => decode_bc6h_mode4_signed(block),
        (0b01010, false) => decode_bc6h_mode5_unsigned(block),
        (0b01010, true) => decode_bc6h_mode5_signed(block),
        (0b01110, false) => decode_bc6h_mode6_unsigned(block),
        (0b01110, true) => decode_bc6h_mode6_signed(block),
        (0b10010, false) => decode_bc6h_mode7_unsigned(block),
        (0b10010, true) => decode_bc6h_mode7_signed(block),
        (0b10110, false) => decode_bc6h_mode8_unsigned(block),
        (0b10110, true) => decode_bc6h_mode8_signed(block),
        (0b11010, false) => decode_bc6h_mode9_unsigned(block),
        (0b11010, true) => decode_bc6h_mode9_signed(block),
        (0b11110, false) => decode_bc6h_mode10_unsigned(block),
        (0b11110, true) => decode_bc6h_mode10_signed(block),
        _ => unreachable!("unhandled two-subset spec mode {mode_bits:#07b}"),
    }
}

fn delta_mask(delta: [u32; 3]) -> [u32; 3] {
    let m = |w: u32| {
        if w == 0 {
            (1u32 << 6) - 1
        } else {
            (1u32 << w) - 1
        }
    };
    [m(delta[0]), m(delta[1]), m(delta[2])]
}

/// BC6H two-subset modes 3-10 **unsigned** parity against GPU hardware.
#[test]
fn bc6h_two_subset_modes_unsigned_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter with BC support reachable; skipping BC6H two-subset unsigned");
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));
    for (si, spec) in BC6H_TWO_SUBSET_SPECS.iter().enumerate() {
        let mut rng = Rng(0x6C13_0000 ^ (si as u32).wrapping_mul(0x9E37_79B1));
        let base_mask = (1u32 << spec.base_prec) - 1;
        let dmask = delta_mask(spec.delta);
        for partition in 0..32u32 {
            for _ in 0..8 {
                let block = random_two_subset_block(
                    &mut rng,
                    spec.desc,
                    partition,
                    spec.mode_bits,
                    base_mask,
                    dmask,
                );
                let cpu = decode_bc6h_unsigned(&block).expect("two-subset block decodes");
                let direct = decode_two_subset_direct(spec.mode_bits, &block, false);
                assert_eq!(
                    cpu, direct,
                    "dispatch must match direct {} decode",
                    spec.name
                );
                let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbUfloat, &block);
                assert_rgb_f32_close(
                    &cpu,
                    &gpu,
                    &format!("{} unsigned partition {partition}", spec.name),
                );
            }
        }
    }
}

/// BC6H two-subset modes 3-10 **signed** (`SF16`) parity against GPU hardware.
#[test]
fn bc6h_two_subset_modes_signed_parity_against_gpu() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter with BC support reachable; skipping BC6H two-subset signed");
        return;
    };
    assert!(oracle.features().contains(Features::TEXTURE_COMPRESSION_BC));
    for (si, spec) in BC6H_TWO_SUBSET_SPECS.iter().enumerate() {
        let mut rng = Rng(0x6C13_5000 ^ (si as u32).wrapping_mul(0x9E37_79B1));
        let base_mask = (1u32 << spec.base_prec) - 1;
        let dmask = delta_mask(spec.delta);
        for partition in 0..32u32 {
            for _ in 0..8 {
                let block = random_two_subset_block(
                    &mut rng,
                    spec.desc,
                    partition,
                    spec.mode_bits,
                    base_mask,
                    dmask,
                );
                let cpu = decode_bc6h_signed(&block).expect("two-subset block decodes");
                let direct = decode_two_subset_direct(spec.mode_bits, &block, true);
                assert_eq!(
                    cpu, direct,
                    "dispatch must match direct {} decode",
                    spec.name
                );
                let gpu = oracle.decode_rgb_f32(TextureFormat::Bc6hRgbFloat, &block);
                assert_rgb_f32_close(
                    &cpu,
                    &gpu,
                    &format!("{} signed partition {partition}", spec.name),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ETC2 RGB8 extended-mode (T / H / Planar) GPU parity.
// ---------------------------------------------------------------------------

/// ETC2 is a bit-exact format: every CPU-decoded texel must equal the GPU
/// hardware decode exactly (tolerance 0). This test forces the differential
/// flag so random blocks fall into the three extended modes (`T`, `H`,
/// `Planar`), classifies them with [`etc2_rgb8_mode`], and compares the
/// pure-CPU [`decode_etc2_rgb8`] output against the Metal hardware oracle for
/// at least 128 blocks per mode.
#[test]
fn etc2_extended_mode_parity() {
    use prism_render_material::{decode_etc2_rgb8, etc2_rgb8_mode, Etc2Mode};
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ETC2 parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ETC2)
    {
        eprintln!("adapter lacks ETC2 support; skipping ETC2 parity");
        return;
    }
    let mut rng = Rng(0x5EED_1234);
    const TARGET: u32 = 128;
    let (mut nt, mut nh, mut np) = (0u32, 0u32, 0u32);
    let mut checked = 0u32;
    for _ in 0..20_000_000u32 {
        if nt >= TARGET && nh >= TARGET && np >= TARGET {
            break;
        }
        let mut b = [0u8; 8];
        for by in &mut b {
            *by = rng.byte();
        }
        b[3] |= 0x02; // force the diff flag so we exercise the extended modes
        let mode = etc2_rgb8_mode(&b);
        let slot = match mode {
            Etc2Mode::T => &mut nt,
            Etc2Mode::H => &mut nh,
            Etc2Mode::Planar => &mut np,
            _ => continue,
        };
        if *slot >= TARGET {
            continue;
        }
        *slot += 1;
        let cpu = decode_etc2_rgb8(&b).expect("extended mode decodes");
        let gpu = oracle.decode_unorm8(TextureFormat::Etc2Rgb8Unorm, &b);
        for t in 0..16 {
            assert_eq!(
                cpu[t][..3],
                gpu[t][..3],
                "{mode:?} block={b:02x?} texel {t}: cpu={:?} gpu={:?}",
                cpu[t],
                gpu[t]
            );
            assert_eq!(cpu[t][3], 255, "{mode:?} alpha must be opaque");
        }
        checked += 1;
    }
    assert!(
        nt >= TARGET && nh >= TARGET && np >= TARGET,
        "insufficient coverage: T={nt} H={nh} Planar={np}"
    );
    eprintln!("ETC2 extended-mode parity: {checked} blocks (T={nt} H={nh} Planar={np})");
}

// ---------------------------------------------------------------------------
// EAC R11 / RG11 (11-bit scalar) GPU parity.
// ---------------------------------------------------------------------------

/// EAC is a bit-exact 11-bit format. The Metal hardware decode returns the
/// unorm value normalised as `v / 2047.0`; recovering the integer with
/// `round(f32 * 2047.0)` reproduces the pure-CPU `[u16; 16]` output exactly
/// (verified empirically: normalisation, texel orientation `t = y*4 + x`, and
/// both the `mult == 0` and `mult != 0` formula paths all match with tol 0).
#[test]
fn eac_r11_parity_against_gpu_hardware_decode() {
    use prism_render_material::decode_eac_r11_unorm;
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping EAC R11 parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ETC2)
    {
        eprintln!("adapter lacks ETC2/EAC support; skipping EAC R11 parity");
        return;
    }
    let mut rng = Rng(0xEAC_1111);
    const COUNT: u32 = 200;
    for _ in 0..COUNT {
        let mut b = [0u8; 8];
        for by in &mut b {
            *by = rng.byte();
        }
        let cpu = decode_eac_r11_unorm(&b);
        let gpu = oracle.decode_raw(TextureFormat::EacR11Unorm, &b);
        for t in 0..16 {
            let g = (gpu[t][0] * 2047.0).round() as i32;
            assert_eq!(
                cpu[t] as i32, g,
                "R11 block={b:02x?} texel {t}: cpu={} gpu*2047={g} (raw {})",
                cpu[t], gpu[t][0]
            );
        }
    }
    eprintln!("EAC R11 parity: {COUNT} blocks bit-exact");
}

/// `EAC_RG11` is two independent R11 channels (red = bytes `0..8`, green =
/// bytes `8..16`). The GPU returns red in channel 0 and green in channel 1;
/// both must match the pure-CPU `[[u16; 2]; 16]` output bit-exactly.
#[test]
fn eac_rg11_parity_against_gpu_hardware_decode() {
    use prism_render_material::decode_eac_rg11_unorm;
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping EAC RG11 parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ETC2)
    {
        eprintln!("adapter lacks ETC2/EAC support; skipping EAC RG11 parity");
        return;
    }
    let mut rng = Rng(0xEAC_2222);
    const COUNT: u32 = 200;
    for _ in 0..COUNT {
        let mut b = [0u8; 16];
        for by in &mut b {
            *by = rng.byte();
        }
        let cpu = decode_eac_rg11_unorm(&b);
        let gpu = oracle.decode_raw(TextureFormat::EacRg11Unorm, &b);
        for t in 0..16 {
            let gr = (gpu[t][0] * 2047.0).round() as i32;
            let gg = (gpu[t][1] * 2047.0).round() as i32;
            assert_eq!(
                cpu[t][0] as i32, gr,
                "RG11 block={b:02x?} texel {t} RED: cpu={} gpu*2047={gr}",
                cpu[t][0]
            );
            assert_eq!(
                cpu[t][1] as i32, gg,
                "RG11 block={b:02x?} texel {t} GREEN: cpu={} gpu*2047={gg}",
                cpu[t][1]
            );
        }
    }
    eprintln!("EAC RG11 parity: {COUNT} blocks bit-exact (red+green)");
}

/// Signed `EAC_R11` is bit-exact: the Metal hardware snorm decode normalises as
/// `v / 1023.0`, so `round(f32 * 1023.0)` recovers the pure-CPU `[i16; 16]`
/// output (`-1023..=1023`) exactly. Confirmed empirically for positive and
/// negative base codewords, `-1024` clamp, and both formula paths (no `+4`).
#[test]
fn eac_r11_snorm_parity_against_gpu_hardware_decode() {
    use prism_render_material::decode_eac_r11_snorm;
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping EAC R11 snorm parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ETC2)
    {
        eprintln!("adapter lacks ETC2/EAC support; skipping EAC R11 snorm parity");
        return;
    }
    let mut rng = Rng(0x5EAC_1111);
    const COUNT: u32 = 200;
    for _ in 0..COUNT {
        let mut b = [0u8; 8];
        for by in &mut b {
            *by = rng.byte();
        }
        let cpu = decode_eac_r11_snorm(&b);
        let gpu = oracle.decode_raw(TextureFormat::EacR11Snorm, &b);
        for t in 0..16 {
            let g = (gpu[t][0] * 1023.0).round() as i32;
            assert_eq!(
                cpu[t] as i32, g,
                "R11_SNORM block={b:02x?} texel {t}: cpu={} gpu*1023={g} (raw {})",
                cpu[t], gpu[t][0]
            );
        }
    }
    eprintln!("EAC R11 snorm parity: {COUNT} blocks bit-exact");
}

/// Signed `EAC_RG11` is two independent signed R11 channels (red = bytes
/// `0..8`, green = bytes `8..16`), the ETC2 analogue of `BC5_SNORM`. Both
/// channels must match the pure-CPU `[[i16; 2]; 16]` output bit-exactly.
#[test]
fn eac_rg11_snorm_parity_against_gpu_hardware_decode() {
    use prism_render_material::decode_eac_rg11_snorm;
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping EAC RG11 snorm parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ETC2)
    {
        eprintln!("adapter lacks ETC2/EAC support; skipping EAC RG11 snorm parity");
        return;
    }
    let mut rng = Rng(0x5EAC_2222);
    const COUNT: u32 = 200;
    for _ in 0..COUNT {
        let mut b = [0u8; 16];
        for by in &mut b {
            *by = rng.byte();
        }
        let cpu = decode_eac_rg11_snorm(&b);
        let gpu = oracle.decode_raw(TextureFormat::EacRg11Snorm, &b);
        for t in 0..16 {
            let gr = (gpu[t][0] * 1023.0).round() as i32;
            let gg = (gpu[t][1] * 1023.0).round() as i32;
            assert_eq!(
                cpu[t][0] as i32, gr,
                "RG11_SNORM block={b:02x?} texel {t} RED: cpu={} gpu*1023={gr}",
                cpu[t][0]
            );
            assert_eq!(
                cpu[t][1] as i32, gg,
                "RG11_SNORM block={b:02x?} texel {t} GREEN: cpu={} gpu*1023={gg}",
                cpu[t][1]
            );
        }
    }
    eprintln!("EAC RG11 snorm parity: {COUNT} blocks bit-exact (red+green)");
}

/// Build an LDR void-extent ASTC block carrying the given 16-bit UNORM channels
/// and a degenerate (all-ones) extent, matching the CPU decoder's expectation.
fn astc_void_extent_ldr(r: u16, g: u16, b: u16, a: u16) -> [u8; 16] {
    let mut blk = [0u8; 16];
    let mut lo: u64 = 0;
    lo |= 0b1_1111_1100u64; // void-extent signature, bits [0..9)
                            // bit 9 = 0 selects LDR.
    lo |= 0b11u64 << 10; // reserved, bits [10..12)
    lo |= ((1u64 << 52) - 1) << 12; // extent coordinates, bits [12..64)
    blk[0..8].copy_from_slice(&lo.to_le_bytes());
    blk[8..10].copy_from_slice(&r.to_le_bytes());
    blk[10..12].copy_from_slice(&g.to_le_bytes());
    blk[12..14].copy_from_slice(&b.to_le_bytes());
    blk[14..16].copy_from_slice(&a.to_le_bytes());
    blk
}

#[test]
fn astc_void_extent_ldr_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC void-extent parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC void-extent parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0xA57C_0001);
    const COUNT: u32 = 256;
    for _ in 0..COUNT {
        let r = (rng.next_u32() & 0xFFFF) as u16;
        let g = (rng.next_u32() & 0xFFFF) as u16;
        let b = (rng.next_u32() & 0xFFFF) as u16;
        let a = (rng.next_u32() & 0xFFFF) as u16;
        let blk = astc_void_extent_ldr(r, g, b, a);
        let cpu = decode_astc_void_extent_ldr(&blk).expect("valid LDR void-extent");
        let gpu = oracle.decode_unorm8(format, &blk);
        for t in 0..16 {
            for c in 0..4 {
                let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                assert!(
                    d <= 1,
                    "ASTC void-extent block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                    cpu[t][c],
                    gpu[t][c]
                );
            }
        }
    }
    eprintln!("ASTC void-extent LDR parity: {COUNT} blocks within 1 LSB of hardware");
}

/// Build an HDR void-extent ASTC block carrying the given FP16 channels.
fn astc_void_extent_hdr(r: u16, g: u16, b: u16, a: u16) -> [u8; 16] {
    let mut blk = astc_void_extent_ldr(r, g, b, a);
    blk[1] |= 1 << (9 - 8); // set bit 9: HDR dynamic-range flag
    blk[8..10].copy_from_slice(&r.to_le_bytes());
    blk[10..12].copy_from_slice(&g.to_le_bytes());
    blk[12..14].copy_from_slice(&b.to_le_bytes());
    blk[14..16].copy_from_slice(&a.to_le_bytes());
    blk
}

#[test]
fn astc_void_extent_hdr_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC HDR void-extent parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC_HDR)
    {
        eprintln!("adapter lacks ASTC HDR support; skipping ASTC HDR void-extent parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Hdr,
    };
    let mut rng = Rng(0xA57C_7D11);
    const COUNT: u32 = 256;
    for _ in 0..COUNT {
        let r = rng.half_bits();
        let g = rng.half_bits();
        let b = rng.half_bits();
        let a = rng.half_bits();
        let blk = astc_void_extent_hdr(r, g, b, a);
        let cpu = decode_astc_void_extent_hdr(&blk).expect("valid HDR void-extent");
        let gpu = oracle.decode_rgb_f32(format, &blk);
        for t in 0..16 {
            for c in 0..3 {
                let (cv, gv) = (cpu[t][c], gpu[t][c]);
                let tol = cv.abs() * 1e-3 + 1e-3;
                assert!(
                    (cv - gv).abs() <= tol,
                    "ASTC HDR void-extent block={blk:02x?} texel {t} chan {c}: cpu={cv} gpu={gv}"
                );
            }
        }
    }
    eprintln!("ASTC HDR void-extent parity: {COUNT} blocks match hardware (RGB FP16)");
}

// ---------------------------------------------------------------------------
// ASTC single-plane 4x4 weight-grid parity.
//
// The weight read + unquantization is proven in isolation against the hardware
// decoder by building a single-partition CEM8 (direct LDR RGB) block mode 578
// whose endpoints are forced to black (e0) and white (e1); each unquantized
// weight 0..=64 then renders as a pure gray level, so the GPU output is a
// direct readout of `decode_astc_4x4_weights`.
// ---------------------------------------------------------------------------

/// Set `count` little-endian bits of `val` starting at bit `lo`.
fn astc_set_bits(blk: &mut [u8; 16], lo: u32, count: u32, val: u32) {
    for i in 0..count {
        if (val >> i) & 1 == 1 {
            let p = lo + i;
            blk[(p >> 3) as usize] |= 1 << (p & 7);
        }
    }
}

/// Build a single-partition, CEM8 block of block mode `bm` with black (e0) and
/// white (e1) endpoints. bits[0..11)=block mode, bits[11..13)=0 (one
/// partition), bits[13..17)=CEM 8; endpoint ISE bits {25,40,55} drive the
/// QUANT_96 endpoints to (0,0,0)->(255,255,255) (hardware-derived config).
fn astc_bm578_black_white() -> [u8; 16] {
    let mut b = [0u8; 16];
    astc_set_bits(&mut b, 0, 11, 578);
    astc_set_bits(&mut b, 13, 4, 8);
    for p in [25u32, 40, 55] {
        b[(p >> 3) as usize] |= 1 << (p & 7);
    }
    b
}

/// Lay texel `t`'s 4-bit weight `w` into the bit-reversed weight region: value
/// bit `b` occupies block bit `127 - (4*t + b)`.
fn astc_set_weight4(blk: &mut [u8; 16], t: u32, w: u32) {
    for b in 0..4u32 {
        if (w >> b) & 1 == 1 {
            let pos = 127 - (4 * t + b);
            blk[(pos >> 3) as usize] |= 1 << (pos & 7);
        }
    }
}

/// Interpolate the gray level a black->white LDR block produces for an
/// unquantized weight `wu` in 0..=64: e0=0, e1=0xFFFF, UNORM16 lerp, then the
/// UNORM16->UNORM8 round the hardware applies.
fn astc_gray_from_weight(wu: u8) -> u8 {
    let c16 = ((65535u32 * wu as u32) + 32) >> 6;
    ((c16 * 255 + 32767) / 65535) as u8
}

#[test]
fn astc_single_plane_weights_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC weight parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC weight parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0xA57C_11E6);
    const COUNT: u32 = 256;
    for _ in 0..COUNT {
        let mut blk = astc_bm578_black_white();
        let mut weights = [0u32; 16];
        for (t, wt) in weights.iter_mut().enumerate() {
            let w = rng.next_u32() & 0xF;
            *wt = w;
            astc_set_weight4(&mut blk, t as u32, w);
        }
        let unquant = decode_astc_4x4_weights(&blk, 4);
        let gpu = oracle.decode_unorm8(format, &blk);
        for t in 0..16 {
            let expected = astc_gray_from_weight(unquant[t]);
            for c in 0..3 {
                let d = (expected as i32 - gpu[t][c] as i32).abs();
                assert!(
                    d <= 1,
                    "ASTC weight block={blk:02x?} texel {t} chan {c}: raw_w={} unquant={} cpu_gray={expected} gpu={} (|d|={d})",
                    weights[t],
                    unquant[t],
                    gpu[t][c]
                );
            }
        }
    }
    eprintln!("ASTC single-plane weight parity: {COUNT} blocks within 1 LSB of hardware");
}

// ---------------------------------------------------------------------------
// ASTC single-plane 4x4 TRIT/QUINT weight-grid parity.
//
// Extends the bit-only (QUANT_16) proof above to the trit- and quint-form
// weight ranges, which unquantize through the astcenc `unscramble_and_unquant_map`
// tables rather than by bit replication. We synthesise single-partition CEM8
// (direct LDR RGB) blocks with black (e0) / white (e1) 8-bit (QUANT_256)
// endpoints so each unquantized weight 0..=64 renders as a pure gray level,
// then compare `decode_astc_4x4_weights_ise` against the Metal hardware
// decoder.
//
// Block modes (wx=wy=4, single plane, D=0). The endpoint quant is derived as
// `color_bits = 111 - weight_bits`, which clamps to QUANT_256 (8-bit, identity
// unquant) for both modes:
//   * QUANT_6  trit  (bits=1): block mode 67,  weight_bits = 42
//   * QUANT_10 quint (bits=1): block mode 577, weight_bits = 54
// ---------------------------------------------------------------------------

/// astcenc `INTEGER_OF_TRITS` packing table (inverse of the decoder table),
/// transcribed from `prism_render_material`'s own verified round-trip fixtures.
/// Used here only to *synthesise* valid trit weight ISE streams.
#[rustfmt::skip]
const INTEGER_OF_TRITS: [u8; 243] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 3, 7, 15, 19, 23, 27,
    12, 13, 14, 32, 33, 34, 36, 37, 38, 40, 41, 42,
    48, 49, 50, 52, 53, 54, 56, 57, 58, 35, 39, 47,
    51, 55, 59, 44, 45, 46, 64, 65, 66, 68, 69, 70,
    72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90,
    67, 71, 79, 83, 87, 91, 76, 77, 78, 128, 129, 130,
    132, 133, 134, 136, 137, 138, 144, 145, 146, 148, 149, 150,
    152, 153, 154, 131, 135, 143, 147, 151, 155, 140, 141, 142,
    160, 161, 162, 164, 165, 166, 168, 169, 170, 176, 177, 178,
    180, 181, 182, 184, 185, 186, 163, 167, 175, 179, 183, 187,
    172, 173, 174, 192, 193, 194, 196, 197, 198, 200, 201, 202,
    208, 209, 210, 212, 213, 214, 216, 217, 218, 195, 199, 207,
    211, 215, 219, 204, 205, 206, 96, 97, 98, 100, 101, 102,
    104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122,
    99, 103, 111, 115, 119, 123, 108, 109, 110, 224, 225, 226,
    228, 229, 230, 232, 233, 234, 240, 241, 242, 244, 245, 246,
    248, 249, 250, 227, 231, 239, 243, 247, 251, 236, 237, 238,
    28, 29, 30, 60, 61, 62, 92, 93, 94, 156, 157, 158,
    188, 189, 190, 220, 221, 222, 31, 63, 127, 159, 191, 255,
    252, 253, 254,
];

/// astcenc `INTEGER_OF_QUINTS` packing table (inverse of the decoder table).
#[rustfmt::skip]
const INTEGER_OF_QUINTS: [u8; 125] = [
    0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 16, 17,
    18, 19, 20, 24, 25, 26, 27, 28, 5, 13, 21, 29,
    6, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 48,
    49, 50, 51, 52, 56, 57, 58, 59, 60, 37, 45, 53,
    61, 14, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76,
    80, 81, 82, 83, 84, 88, 89, 90, 91, 92, 69, 77,
    85, 93, 22, 96, 97, 98, 99, 100, 104, 105, 106, 107,
    108, 112, 113, 114, 115, 116, 120, 121, 122, 123, 124, 101,
    109, 117, 125, 30, 102, 103, 70, 71, 38, 110, 111, 78,
    79, 46, 118, 119, 86, 87, 54, 126, 127, 94, 95, 62,
    39, 47, 55, 63, 31,
];

fn enc_trit(t: [u8; 5]) -> u8 {
    INTEGER_OF_TRITS[((((t[4] as usize * 3 + t[3] as usize) * 3 + t[2] as usize) * 3
        + t[1] as usize)
        * 3)
        + t[0] as usize]
}

fn enc_quint(q: [u8; 3]) -> u8 {
    INTEGER_OF_QUINTS[(q[2] as usize * 5 + q[1] as usize) * 5 + q[0] as usize]
}

/// astcenc-order trit ISE encoder (mirror of the decoder's bit-collection
/// order). `vals` are raw BISE values `low | (trit << bits)`.
fn astc_encode_trit(block: &mut [u8; 16], start: u32, bits: u32, vals: &[u8]) {
    let mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };
    let count = vals.len();
    let mut off = start;
    let mut i = 0usize;
    let full = count / 5;
    for _ in 0..full {
        let t = enc_trit([
            vals[i] >> bits,
            vals[i + 1] >> bits,
            vals[i + 2] >> bits,
            vals[i + 3] >> bits,
            vals[i + 4] >> bits,
        ]) as u32;
        let shifts = [0u32, 2, 4, 5, 7];
        let tb = [2u32, 2, 1, 2, 1];
        for e in 0..5 {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> shifts[e]) & ((1 << tb[e]) - 1)) << bits);
            astc_set_bits(block, off, bits + tb[e], pack);
            off += bits + tb[e];
            i += 1;
        }
    }
    if i != count {
        let g = |k: usize| {
            if i + k >= count {
                0
            } else {
                vals[i + k] >> bits
            }
        };
        let t = enc_trit([g(0), g(1), g(2), g(3), 0]) as u32;
        let tbits = [2u32, 2, 1, 2];
        let tshift = [0u32, 2, 4, 5];
        let mut j = 0usize;
        while i < count {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);
            astc_set_bits(block, off, bits + tbits[j], pack);
            off += bits + tbits[j];
            i += 1;
            j += 1;
        }
    }
}

/// astcenc-order quint ISE encoder (mirror of the decoder's bit-collection
/// order). `vals` are raw BISE values `low | (quint << bits)`.
fn astc_encode_quint(block: &mut [u8; 16], start: u32, bits: u32, vals: &[u8]) {
    let mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };
    let count = vals.len();
    let mut off = start;
    let mut i = 0usize;
    let full = count / 3;
    for _ in 0..full {
        let t = enc_quint([vals[i] >> bits, vals[i + 1] >> bits, vals[i + 2] >> bits]) as u32;
        let shifts = [0u32, 3, 5];
        let tb = [3u32, 2, 2];
        for e in 0..3 {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> shifts[e]) & ((1 << tb[e]) - 1)) << bits);
            astc_set_bits(block, off, bits + tb[e], pack);
            off += bits + tb[e];
            i += 1;
        }
    }
    if i != count {
        let g = |k: usize| {
            if i + k >= count {
                0
            } else {
                vals[i + k] >> bits
            }
        };
        let t = enc_quint([g(0), g(1), 0]) as u32;
        let tbits = [3u32, 2];
        let tshift = [0u32, 3];
        let mut j = 0usize;
        while i < count {
            let pack =
                ((vals[i] as u32) & mask) | (((t >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);
            astc_set_bits(block, off, bits + tbits[j], pack);
            off += bits + tbits[j];
            i += 1;
            j += 1;
        }
    }
}

#[derive(Clone, Copy)]
enum WeightForm {
    Trit,
    Quint,
}

/// Lay sixteen raw weight values into the bit-reversed weight region (the top
/// `weight_bits` bits) by encoding them LSB-first into a scratch "reversed"
/// block, then mirroring that block end-for-end into `blk` (logical stream bit
/// `p` -> real block bit `127 - p`). This is the exact inverse of the mirror +
/// `decode_ise` read in `decode_astc_4x4_weights_ise`.
fn astc_set_weights_ise(blk: &mut [u8; 16], form: WeightForm, bits: u32, weights: &[u8; 16]) {
    let mut tmp = [0u8; 16];
    match form {
        WeightForm::Trit => astc_encode_trit(&mut tmp, 0, bits, weights),
        WeightForm::Quint => astc_encode_quint(&mut tmp, 0, bits, weights),
    }
    for p in 0..128u32 {
        if (tmp[(p >> 3) as usize] >> (p & 7)) & 1 == 1 {
            let real = 127 - p;
            blk[(real >> 3) as usize] |= 1 << (real & 7);
        }
    }
}

/// Build a single-partition CEM8 block of block mode `bm` with black (e0) and
/// white (e1) 8-bit (QUANT_256) endpoints. The six 8-bit endpoint values sit
/// at bits [17..65): v0=v2=v4=0 (e0 black) and v1=v3=v5=255 (e1 white). Since
/// sum(e0)=0 < sum(e1)=765 there is no blue-contraction swap.
fn astc_bw_quant256(bm: u32) -> [u8; 16] {
    let mut b = [0u8; 16];
    astc_set_bits(&mut b, 0, 11, bm);
    astc_set_bits(&mut b, 13, 4, 8);
    for off in [25u32, 41, 57] {
        astc_set_bits(&mut b, off, 8, 255);
    }
    b
}

/// Modes exercised by the trit/quint weight parity test: `(block_mode, form,
/// low_bits, level_count)`.
const TRIT_QUINT_MODES: [(u32, WeightForm, u32, u32); 2] = [
    (67, WeightForm::Trit, 1, 6),
    (577, WeightForm::Quint, 1, 10),
];

#[test]
fn astc_trit_quint_weights_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC trit/quint weight parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC trit/quint weight parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x7217_0a1c);
    const PER_MODE: u32 = 128;
    for (bm, form, bits, levels) in TRIT_QUINT_MODES {
        for _ in 0..PER_MODE {
            let mut blk = astc_bw_quant256(bm);
            let mut raw = [0u8; 16];
            for r in raw.iter_mut() {
                *r = (rng.next_u32() % levels) as u8;
            }
            astc_set_weights_ise(&mut blk, form, bits, &raw);
            let unquant =
                decode_astc_4x4_weights_ise(&blk, levels).expect("level count is a valid range");
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                let expected = astc_gray_from_weight(unquant[t]);
                for c in 0..3 {
                    let d = (expected as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC trit/quint weight bm={bm} block={blk:02x?} texel {t} chan {c}: raw={} unquant={} cpu_gray={expected} gpu={} (|d|={d})",
                        raw[t],
                        unquant[t],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC trit/quint weight parity: {} blocks within 1 LSB of hardware",
        PER_MODE * 2
    );
}

// ---------------------------------------------------------------------------
// ASTC full single-partition CEM8 (direct LDR RGB) parity.
//
// The two tests above isolate the weight path by pinning the endpoints to
// black/white. This test exercises the complete single-partition pipeline:
// six *random* 8-bit (QUANT_256) endpoint integers decoded into the two RGB
// endpoint colours (with the astcenc blue-contraction + endpoint-swap applied
// when sum(e0_rgb) > sum(e1_rgb)), then interpolated per texel by random
// trit/quint weights. We compare the full `decode_astc_4x4_ldr` output against
// the Metal hardware decoder on all four channels (CEM8 forces alpha = 255 on
// both paths). Random endpoints mean ~half the blocks drive the hardware
// blue-contraction path, so that branch is proven here too.
// ---------------------------------------------------------------------------

/// Build a single-partition CEM8 block of block mode `bm` with six explicit
/// 8-bit (QUANT_256) endpoint integers at bits [17..65):
/// `ep = [v0, v1, v2, v3, v4, v5]` where (v0,v2,v4) is endpoint 0's RGB and
/// (v1,v3,v5) is endpoint 1's RGB (CEM8 interleave).
fn astc_cem8_block(bm: u32, ep: [u8; 6]) -> [u8; 16] {
    let mut b = [0u8; 16];
    astc_set_bits(&mut b, 0, 11, bm);
    astc_set_bits(&mut b, 13, 4, 8);
    for (i, v) in ep.iter().enumerate() {
        astc_set_bits(&mut b, 17 + i as u32 * 8, 8, u32::from(*v));
    }
    b
}

#[test]
fn astc_full_single_partition_cem8_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC full single-partition parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC full single-partition parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x0C7A_5EED);
    const PER_MODE: u32 = 128;
    let mut contracted = 0u32;
    for (bm, form, bits, levels) in TRIT_QUINT_MODES {
        for _ in 0..PER_MODE {
            let mut ep = [0u8; 6];
            for e in ep.iter_mut() {
                *e = (rng.next_u32() & 0xFF) as u8;
            }
            let mut blk = astc_cem8_block(bm, ep);
            let mut raw = [0u8; 16];
            for r in raw.iter_mut() {
                *r = (rng.next_u32() % levels) as u8;
            }
            astc_set_weights_ise(&mut blk, form, bits, &raw);

            // sum(e0_rgb) > sum(e1_rgb) drives the blue-contraction + swap path
            // on both the CPU decoder and the hardware.
            let s0 = u32::from(ep[0]) + u32::from(ep[2]) + u32::from(ep[4]);
            let s1 = u32::from(ep[1]) + u32::from(ep[3]) + u32::from(ep[5]);
            if s0 > s1 {
                contracted += 1;
            }

            let cpu = decode_astc_4x4_ldr(&blk).expect("supported single-partition CEM8 block");
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC full CEM8 bm={bm} block={blk:02x?} texel {t} chan {c}: ep={ep:?} raw_w={} cpu={} gpu={} (|d|={d})",
                        raw[t],
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC full single-partition CEM8 parity: {} blocks within 1 LSB of hardware ({contracted} exercised blue-contraction)",
        PER_MODE * 2
    );
}

// ---------------------------------------------------------------------------
// ASTC single-partition CEM8 (direct LDR RGB) parity with a NON-QUANT_256
// colour range.
//
// The three ASTC colour tests above all drive QUANT_256 (8-bit identity)
// endpoints, so they never exercise the colour *unquantization* tables. This
// test uses block mode 578 (4x4, QUANT_16 bit-only weights, 64 weight bits),
// whose single-partition single-plane colour budget is
// `color_bits = 111 - 64 = 47`, which the reference `quant_mode_table` maps to
// QUANT_192 (colour level index 15, a trit-form range with 6 low bits). The
// six colour integers are therefore encoded as a trit ISE stream at bit 17 and
// decoded through the `color_scrambled_pquant_to_uquant_q192` table on the CPU
// side, then compared against the Metal hardware decoder on all four channels
// (CEM8 forces alpha = 255 on both paths). Random packed endpoints drive the
// blue-contraction + swap branch on roughly half the blocks.
// ---------------------------------------------------------------------------

/// `color_scrambled_pquant_to_uquant_q192` (QUANT_192, colour level index 15),
/// transcribed verbatim from astcenc. Mirrors `color_unquant::Q192` in the
/// CPU decoder; duplicated here only to count blue-contracted blocks for the
/// diagnostic, so a drift between the two is itself a useful tripwire.
#[rustfmt::skip]
const COLOR_Q192: [u8; 192] = [
    0, 255, 4, 251, 8, 247, 12, 243, 16, 239, 20, 235, 24, 231, 28, 227, 32, 223, 36, 219, 40, 215,
    44, 211, 48, 207, 52, 203, 56, 199, 60, 195, 64, 191, 68, 187, 72, 183, 76, 179, 80, 175, 84,
    171, 88, 167, 92, 163, 96, 159, 100, 155, 104, 151, 108, 147, 112, 143, 116, 139, 120, 135,
    124, 131, 1, 254, 5, 250, 9, 246, 13, 242, 17, 238, 21, 234, 25, 230, 29, 226, 33, 222, 37,
    218, 41, 214, 45, 210, 49, 206, 53, 202, 57, 198, 61, 194, 65, 190, 69, 186, 73, 182, 77, 178,
    81, 174, 85, 170, 89, 166, 93, 162, 97, 158, 101, 154, 105, 150, 109, 146, 113, 142, 117, 138,
    121, 134, 125, 130, 2, 253, 6, 249, 10, 245, 14, 241, 18, 237, 22, 233, 26, 229, 30, 225, 34,
    221, 38, 217, 42, 213, 46, 209, 50, 205, 54, 201, 58, 197, 62, 193, 66, 189, 70, 185, 74, 181,
    78, 177, 82, 173, 86, 169, 90, 165, 94, 161, 98, 157, 102, 153, 106, 149, 110, 145, 114, 141,
    118, 137, 122, 133, 126, 129,
];

#[test]
fn astc_cem8_color_quant192_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC QUANT_192 colour parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC QUANT_192 colour parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x1992_C01E);
    const COUNT: u32 = 256;
    let mut contracted = 0u32;
    for _ in 0..COUNT {
        // Mode 578 (4x4 QUANT_16 weights, 64 weight bits) + CEM 8.
        let mut blk = [0u8; 16];
        astc_set_bits(&mut blk, 0, 11, 578);
        astc_set_bits(&mut blk, 13, 4, 8);

        // Six QUANT_192 packed colour integers: `low | (trit << 6)` with
        // `low` in 0..64 and `trit` in 0..3, encoded as a trit ISE at bit 17.
        let mut packed = [0u8; 6];
        for p in packed.iter_mut() {
            let low = rng.next_u32() % 64;
            let trit = rng.next_u32() % 3;
            *p = (low | (trit << 6)) as u8;
        }
        astc_encode_trit(&mut blk, 17, 6, &packed);

        // Sixteen random 4-bit (QUANT_16) weights, bit-reversed from the top.
        let mut weights = [0u32; 16];
        for (t, wt) in weights.iter_mut().enumerate() {
            let w = rng.next_u32() & 0xF;
            *wt = w;
            astc_set_weight4(&mut blk, t as u32, w);
        }

        // Count blocks that drive blue-contraction + swap, computed on the
        // *unquantized* colours exactly as the CPU decoder does.
        let u = |i: usize| COLOR_Q192[packed[i] as usize] as u32;
        let s0 = u(0) + u(2) + u(4);
        let s1 = u(1) + u(3) + u(5);
        if s0 > s1 {
            contracted += 1;
        }

        let cpu = decode_astc_4x4_ldr(&blk).expect("supported QUANT_192 CEM8 block");
        let gpu = oracle.decode_unorm8(format, &blk);
        for t in 0..16 {
            for c in 0..4 {
                let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                assert!(
                    d <= 1,
                    "ASTC QUANT_192 CEM8 block={blk:02x?} texel {t} chan {c}: packed={packed:?} raw_w={} cpu={} gpu={} (|d|={d})",
                    weights[t],
                    cpu[t][c],
                    gpu[t][c]
                );
            }
        }
    }
    eprintln!(
        "ASTC QUANT_192 colour parity: {COUNT} blocks within 1 LSB of hardware ({contracted} exercised blue-contraction)"
    );
}

// ---------------------------------------------------------------------------
// ASTC single-partition multi-CEM LDR endpoint parity (Milestone #5).
//
// The earlier ASTC colour tests only exercised CEM 8 (direct LDR RGB). This
// proves every one of the ten LDR Colour Endpoint Modes
// (0/1/4/5/6/8/9/10/12/13) against the Metal hardware decoder. Each block uses
// block mode 67 (4x4, single plane, QUANT_6 trit weights, weight_bits = 42),
// whose single-partition colour budget is `color_bits = 111 - 42 = 69`. At 69
// colour bits every `quant_mode_table` row maps to QUANT_256, so the N colour
// integers for each CEM are plain 8-bit binary values at bits [17..17+8N),
// with no colour-unquantization table in the path. The weights remain a
// genuine 6-level trit ISE stream, so endpoint interpolation is still
// exercised end-to-end. Random endpoint integers naturally drive the
// blue-contraction, delta sign-extension and RGB-scale branches on both the
// CPU and hardware sides; parity is checked on all four channels to 1 LSB.
// ---------------------------------------------------------------------------

/// `(cem, integer_count)` for the ten LDR Colour Endpoint Modes.
const LDR_CEMS: [(u32, u32); 10] = [
    (0, 2),  // LUM
    (1, 2),  // LUM_DELTA
    (4, 4),  // LUM_ALPHA
    (5, 4),  // LUM_ALPHA_DELTA
    (6, 4),  // RGB_SCALE
    (8, 6),  // RGB
    (9, 6),  // RGB_DELTA
    (10, 6), // RGB_SCALE_ALPHA
    (12, 8), // RGBA
    (13, 8), // RGBA_DELTA
];

#[test]
fn astc_multi_cem_ldr_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-CEM LDR parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC multi-CEM LDR parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x05EC_0DE5);
    const PER_CEM: u32 = 128;
    for (cem, integer_count) in LDR_CEMS {
        for _ in 0..PER_CEM {
            // Mode 67 (4x4, single plane, QUANT_6 trit weights) + this CEM.
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, 67);
            astc_set_bits(&mut blk, 13, 4, cem);

            // N raw 8-bit (QUANT_256) colour integers at bits [17..17+8N).
            for i in 0..integer_count {
                let v = rng.byte() as u32;
                astc_set_bits(&mut blk, 17 + i * 8, 8, v);
            }

            // Sixteen 6-level trit weights (low in 0..2, trit in 0..3).
            let mut weights = [0u8; 16];
            for w in weights.iter_mut() {
                let low = (rng.next_u32() % 2) as u8;
                let trit = (rng.next_u32() % 3) as u8;
                *w = low | (trit << 1);
            }
            astc_set_weights_ise(&mut blk, WeightForm::Trit, 1, &weights);

            let cpu = decode_astc_4x4_ldr(&blk)
                .unwrap_or_else(|e| panic!("CEM {cem} block {blk:02x?} rejected: {e:?}"));
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC CEM {cem} block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-CEM LDR parity: {} LDR CEMs x {PER_CEM} blocks within 1 LSB of hardware",
        LDR_CEMS.len()
    );
}

/// Weight ISE forms spanning every single-plane range the infill test needs:
/// pure-binary ranges in addition to the trit/quint ranges [`WeightForm`]
/// already covers.
#[derive(Clone, Copy)]
enum GridWeightForm {
    Bits,
    Trit,
    Quint,
}

/// Map a BISE weight level count to its ISE form and low (pure-binary) bit
/// width, matching `IseRange::from_num_levels` on the decode side.
fn levels_to_grid_form(levels: u32) -> (GridWeightForm, u32) {
    match levels {
        2 => (GridWeightForm::Bits, 1),
        3 => (GridWeightForm::Trit, 0),
        4 => (GridWeightForm::Bits, 2),
        5 => (GridWeightForm::Quint, 0),
        6 => (GridWeightForm::Trit, 1),
        8 => (GridWeightForm::Bits, 3),
        10 => (GridWeightForm::Quint, 1),
        12 => (GridWeightForm::Trit, 2),
        16 => (GridWeightForm::Bits, 4),
        20 => (GridWeightForm::Quint, 2),
        24 => (GridWeightForm::Trit, 3),
        32 => (GridWeightForm::Bits, 5),
        _ => panic!("unsupported weight level count {levels}"),
    }
}

/// Draw a random raw weight for `form`/`bits`: a `bits`-wide binary value, or a
/// `low | (digit << bits)` trit/quint pack, exactly what the encoders consume.
fn rand_grid_weight(rng: &mut Rng, form: GridWeightForm, bits: u32) -> u8 {
    match form {
        GridWeightForm::Bits => (rng.next_u32() % (1 << bits)) as u8,
        GridWeightForm::Trit => {
            let low = rng.next_u32() % (1 << bits);
            let trit = rng.next_u32() % 3;
            (low | (trit << bits)) as u8
        }
        GridWeightForm::Quint => {
            let low = rng.next_u32() % (1 << bits);
            let quint = rng.next_u32() % 5;
            (low | (quint << bits)) as u8
        }
    }
}

/// Lay `vals` (a single-plane weight grid of arbitrary length) into the
/// bit-reversed weight region: encode LSB-first into a scratch block at bit 0
/// using `form`/`bits`, then mirror end-for-end (logical bit `p` -> real block
/// bit `127 - p`), the exact inverse of the decoder's mirror + `decode_ise`.
fn astc_set_grid_weights(blk: &mut [u8; 16], form: GridWeightForm, bits: u32, vals: &[u8]) {
    let mut tmp = [0u8; 16];
    match form {
        GridWeightForm::Bits => {
            let mut off = 0u32;
            for &v in vals {
                astc_set_bits(&mut tmp, off, bits, v as u32);
                off += bits;
            }
        }
        GridWeightForm::Trit => astc_encode_trit(&mut tmp, 0, bits, vals),
        GridWeightForm::Quint => astc_encode_quint(&mut tmp, 0, bits, vals),
    }
    for p in 0..128u32 {
        if (tmp[(p >> 3) as usize] >> (p & 7)) & 1 == 1 {
            let real = 127 - p;
            blk[(real >> 3) as usize] |= 1 << (real & 7);
        }
    }
}

/// Non-4x4 single-plane modes exercised by the bilinear-infill parity test:
/// `(block_mode, weights_x, weights_y, weight_levels)`. Each uses CEM8 with six
/// raw 8-bit (QUANT_256) colour integers, and a weight grid that is resampled
/// to the 4x4 footprint by the Khronos bilinear infill. The grids span wide,
/// tall and square shapes and all three weight ISE forms (bits/trit/quint).
// Every grid here fits inside the 4x4 texel footprint (wx<=4 && wy<=4) and is
// NOT the 4x4 baseline, so each is a LEGAL non-4x4 single-plane block mode that
// conformant ASTC hardware accepts. Grids larger than the footprint (e.g. 8x2,
// 5x2, 4x8) are illegal for a 4x4 block and the hardware rejects them, so they
// are intentionally excluded. Shapes cover 4x2/4x3/2x4/3x4/2x3/3x2/3x3 and the
// weight ISE forms span pure-bit (4/8/16), trit (6/24), and quint (5/10) ranges.
const INFILL_MODES: [(u32, u32, u32, u32); 12] = [
    (19, 4, 2, 8),
    (34, 4, 3, 4),
    (35, 4, 3, 6),
    (50, 4, 3, 5),
    (351, 2, 4, 8),
    (431, 3, 3, 6),
    (462, 3, 4, 4),
    (478, 3, 4, 5),
    (814, 2, 3, 16),
    (910, 3, 2, 16),
    (941, 3, 3, 10),
    (943, 3, 3, 24),
];

/// GPU parity for the non-4x4 single-plane weight-grid bilinear infill: for a
/// spread of grid shapes and all three weight ISE forms, build CEM8 QUANT_256
/// blocks with random endpoints and random grid weights, then confirm the CPU
/// `decode_astc_4x4_ldr` matches the Metal ASTC hardware decoder within 1 LSB
/// on every texel and channel.
#[test]
fn astc_infill_non44_single_plane_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC non-4x4 infill parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC non-4x4 infill parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x14F1_11AB);
    const PER_MODE: u32 = 128;
    for (mode, wx, wy, levels) in INFILL_MODES {
        let (form, bits) = levels_to_grid_form(levels);
        let weight_count = (wx * wy) as usize;
        for _ in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 13, 4, 8); // CEM 8 (LDR direct RGB)

            // Six raw 8-bit (QUANT_256) colour integers at bits [17..65).
            for i in 0..6u32 {
                astc_set_bits(&mut blk, 17 + i * 8, 8, rng.byte() as u32);
            }

            // weights_x * weights_y random grid weights in the mode's range.
            let mut weights = [0u8; 64];
            for w in weights.iter_mut().take(weight_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &weights[..weight_count]);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!("mode {mode} ({wx}x{wy}, {levels} levels) block {blk:02x?} rejected: {e:?}")
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC infill mode {mode} ({wx}x{wy}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC non-4x4 infill parity: {} grid modes x {PER_MODE} blocks within 1 LSB of hardware",
        INFILL_MODES.len()
    );
}

// -------------------------------------------------------------------------
// ASTC dual-plane single-partition LDR parity (Milestone #7).
//
// A dual-plane block stores two interleaved weight planes plus a 2-bit colour
// component selector (CCS): the selected channel interpolates with plane 1,
// the other three with plane 0. We build CEM8 QUANT_256 blocks (six raw 8-bit
// colour integers at bits [17..65)) across a spread of legal dual-plane grid
// shapes and all three weight ISE forms, write 2*wx*wy interleaved grid
// weights, set the CCS just below the weight region, and confirm the CPU
// `decode_astc_4x4_ldr` matches the Metal ASTC hardware decoder within 1 LSB.
//
// Each tuple is (block_mode, weights_x, weights_y, weight_levels, weight_bits);
// weight_bits positions the CCS at `128 - weight_bits - 2`. Every mode keeps
// color_bits = 109 - weight_bits >= 48 so CEM8 stays at QUANT_256.
const DUAL_PLANE_MODES: [(u32, u32, u32, u32, u32); 12] = [
    (1057, 4, 3, 2, 24),
    (1026, 4, 2, 4, 32),
    (1043, 4, 2, 8, 48),
    (1806, 2, 2, 16, 32),
    (1089, 4, 4, 2, 32),
    (1041, 4, 2, 3, 26),
    (1027, 4, 2, 6, 42),
    (1105, 4, 4, 3, 52),
    (1470, 3, 3, 5, 42),
    (1805, 2, 2, 10, 27),
    (1359, 2, 4, 6, 42),
    (1485, 3, 4, 2, 24),
];

#[test]
fn astc_dual_plane_single_partition_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC dual-plane parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC dual-plane parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x5EED_D0AB);
    const PER_MODE: u32 = 128;
    for (mode, wx, wy, levels, weight_bits) in DUAL_PLANE_MODES {
        let (form, bits) = levels_to_grid_form(levels);
        // Dual plane stores two interleaved weights per grid point.
        let seq_count = (wx * wy * 2) as usize;
        let ccs_pos = 128 - weight_bits - 2;
        for n in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 13, 4, 8); // CEM 8 (LDR direct RGB)

            // Six raw 8-bit (QUANT_256) colour integers at bits [17..65).
            for i in 0..6u32 {
                astc_set_bits(&mut blk, 17 + i * 8, 8, rng.byte() as u32);
            }

            // 2 * wx * wy interleaved grid weights (even -> plane 0, odd ->
            // plane 1), laid into the bit-reversed weight region.
            let mut seq = [0u8; 64];
            for w in seq.iter_mut().take(seq_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &seq[..seq_count]);

            // Colour component selector: cycle through all four channels so
            // every plane-routing case is exercised.
            let ccs = n % 4;
            astc_set_bits(&mut blk, ccs_pos, 2, ccs);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!(
                    "dual-plane mode {mode} ({wx}x{wy}, {levels} levels, ccs {ccs}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC dual-plane mode {mode} ({wx}x{wy}, ccs {ccs}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC dual-plane single-partition parity: {} grid modes x {PER_MODE} blocks within 1 LSB of hardware",
        DUAL_PLANE_MODES.len()
    );
}

// -------------------------------------------------------------------------
// ASTC multi-partition single-plane LDR parity (Milestone #8, shared CEM class).
//
// A multi-partition block splits the 4x4 footprint into 2/3/4 regions via the
// procedural partition hash (10-bit seed in bits [13,23)); each region carries
// its own endpoint pair while a single weight stream (bit-reversed at the top)
// is shared. We exercise the "all partitions share one colour class" CEM form
// (base class 0) at QUANT_256, so every colour integer is a raw 8-bit value
// written forward from bit 29. Across 2/3/4 partitions and the luminance
// (CEM0), luminance+alpha (CEM4) and RGB base+scale (CEM6) formats, we confirm
// the CPU `decode_astc_4x4_ldr` matches the Metal ASTC hardware decoder within
// 1 LSB on every texel and channel.
//
// Each tuple is (block_mode, weights_x, weights_y, weight_levels, weight_bits,
// partition_count, cem, color_integer_count). color_bits = 99 - weight_bits
// (the shared class spends no CEM high part); every config keeps
// color_integer_count*8 <= color_bits so the colour ISE stays at QUANT_256, and
// the raw 8-bit endpoint run [29, 29 + 8*n) never overlaps the top weight
// region [128 - weight_bits, 128). All seven modes are single-plane grids drawn
// from the committed infill set (weight_bits == 24).
const MULTI_PART_MODES: [(u32, u32, u32, u32, u32, u32, u32, u32); 7] = [
    (19, 4, 2, 8, 24, 2, 0, 4),
    (814, 2, 3, 16, 24, 2, 0, 4),
    (431, 3, 3, 6, 24, 3, 0, 6),
    (34, 4, 3, 4, 24, 2, 6, 8),
    (351, 2, 4, 8, 24, 2, 4, 8),
    (462, 3, 4, 4, 24, 4, 0, 8),
    (910, 3, 2, 16, 24, 4, 0, 8),
];

#[test]
fn astc_multi_partition_single_plane_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-partition parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC multi-partition parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x6A17_2C0B);
    const PER_MODE: u32 = 128;
    for (mode, wx, wy, levels, weight_bits, pc, cem, n_int) in MULTI_PART_MODES {
        let _ = weight_bits; // documented invariant; weights placed by grid form
        let (form, bits) = levels_to_grid_form(levels);
        let weight_count = (wx * wy) as usize;
        for _ in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, pc - 1); // partition count minus one
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF); // partition seed

            // Shared CEM class (base class 0): the 6-bit field at [23,29) holds
            // the colour format in bits 2..6, so the two low base-class bits are
            // zero and no CEM high part is spent.
            astc_set_bits(&mut blk, 23, 6, cem << 2);

            // n_int raw 8-bit (QUANT_256) colour integers written forward from
            // bit 29 (= 19 + PARTITION_INDEX_BITS).
            for i in 0..n_int {
                astc_set_bits(&mut blk, 29 + i * 8, 8, rng.byte() as u32);
            }

            // Shared single-plane weight stream (bit-reversed at the top).
            let mut weights = [0u8; 64];
            for w in weights.iter_mut().take(weight_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &weights[..weight_count]);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!(
                    "multi-part mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC multi-part mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition single-plane parity: {} configs x {PER_MODE} blocks within 1 LSB of hardware",
        MULTI_PART_MODES.len()
    );
}

// ---------------------------------------------------------------------------
// ASTC multi-partition low-quant colour + per-partition CEM parity (M#8-ext).
//
// The committed multi-partition test only drives shared-class CEM at the
// identity QUANT_256 colour level, so two decode paths that are already
// implemented in `multi_partition.rs` remained unproven on hardware:
//   (A) non-identity colour ISE levels (QUANT_16/24/40/48/64 and the trit and
//       quint colour forms), exercising `quant_mode::color_quant_level`
//       ROW5-ROW9 and `color_unquant` for low-quant colour on 2/3/4 partitions;
//   (B) the per-partition colour-endpoint-mode class field (base class != 0),
//       exercising the per-partition CEM decode branch and its high-part bits
//       just below the weight stream.
// Both tests build whole blocks by hand and compare the CPU
// `decode_astc_4x4_ldr` against the Metal ASTC hardware decoder within 1 LSB.
// ---------------------------------------------------------------------------

/// Number of colour integers a given LDR colour-endpoint-mode consumes
/// (`cem_integer_count` in the CPU decoder): `((cem >> 2) + 1) * 2`.
fn cem_int_count(cem: u32) -> u32 {
    ((cem >> 2) + 1) * 2
}

/// Reference `quant_mode_table[integer_count / 2][color_bits]` restricted to
/// the 10/12/14/16/18-integer rows (ROW5-ROW9), transcribed verbatim from the
/// CPU `quant_mode` module so the test derives the exact same colour quant
/// level the decoder will use. A colour-bits value past the table saturates at
/// QUANT_256 (level 20).
#[rustfmt::skip]
fn color_quant_level_test(integer_count: u32, color_bits: usize) -> i8 {
    const ROW5: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5,
        5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10,
        10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14,
        15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 19, 19, 19, 19,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    ];
    const ROW6: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
        8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11,
        12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,
        16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    ];
    const ROW7: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6,
        6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9,
        9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13,
        13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16,
        16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    ];
    const ROW8: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,
        17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19,
    ];
    const ROW9: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4,
        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
        6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9,
        9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
        14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17,
    ];
    let row: &[i8; 128] = match integer_count >> 1 {
        5 => &ROW5,
        6 => &ROW6,
        7 => &ROW7,
        8 => &ROW8,
        9 => &ROW9,
        other => panic!("test only exercises 10/12/14/16/18 colour integers, got {other} halves"),
    };
    *row.get(color_bits).unwrap_or(&20)
}

/// (low bit width, is-trit, is-quint) for colour quant level `0..=20`, the exact
/// inverse of `color_unquant::NUM_LEVELS` so endpoints can be ISE-encoded in the
/// level the decoder expects.
fn color_btq(level: i8) -> (u32, bool, bool) {
    match level {
        0 => (1, false, false),
        1 => (0, true, false),
        2 => (2, false, false),
        3 => (0, false, true),
        4 => (1, true, false),
        5 => (3, false, false),
        6 => (1, false, true),
        7 => (2, true, false),
        8 => (4, false, false),
        9 => (2, false, true),
        10 => (3, true, false),
        11 => (5, false, false),
        12 => (3, false, true),
        13 => (4, true, false),
        14 => (6, false, false),
        15 => (4, false, true),
        16 => (5, true, false),
        17 => (7, false, false),
        18 => (5, false, true),
        19 => (6, true, false),
        20 => (8, false, false),
        other => panic!("unsupported colour quant level {other}"),
    }
}

/// Number of distinct quant steps at colour level `level`.
fn color_num_levels(level: i8) -> u32 {
    let (bits, trit, quint) = color_btq(level);
    if trit {
        3 << bits
    } else if quint {
        5 << bits
    } else {
        1 << bits
    }
}

/// Write `ic` random colour endpoint integers forward from bit `start` at colour
/// quant `level`, in astcenc ISE order (the exact inverse of the decoder's
/// `decode_ise` over the concatenated endpoint run). Each raw value is a valid
/// packed `low | (digit << bits)` because it is drawn modulo the level's step
/// count.
fn set_color_endpoints(blk: &mut [u8; 16], start: u32, level: i8, ic: usize, rng: &mut Rng) {
    let (bits, trit, quint) = color_btq(level);
    let nl = color_num_levels(level);
    let mut vals = [0u8; 18];
    for v in vals.iter_mut().take(ic) {
        *v = (rng.next_u32() % nl) as u8;
    }
    if trit {
        astc_encode_trit(blk, start, bits, &vals[..ic]);
    } else if quint {
        astc_encode_quint(blk, start, bits, &vals[..ic]);
    } else {
        for (i, &v) in vals[..ic].iter().enumerate() {
            astc_set_bits(blk, start + (i as u32) * bits, bits, v as u32);
        }
    }
}

/// Encode a per-partition colour-endpoint-mode field (base class != 0) for the
/// `cems` list. The low 6 bits land at [23, 29); the `3*pc - 4` high bits land
/// just below the weight stream at `128 - weight_bits - highpart`, exactly where
/// `multi_partition::decode_multi_partition_4x4_ldr` reads them back. All CEM
/// classes must lie within one step of the minimum class so the shared base-class
/// encoding is representable.
fn set_cem_per_partition(blk: &mut [u8; 16], cems: &[u32], weight_bits: u32) {
    let pc = cems.len() as u32;
    let base = cems
        .iter()
        .map(|&c| c >> 2)
        .min()
        .expect("non-empty CEM list");
    let baseclass = base + 1;
    assert!(
        (1..=3).contains(&baseclass),
        "base class {baseclass} out of range"
    );
    let highpart = 3 * pc - 4;
    let mut enc: u32 = baseclass & 0x3;
    for (i, &c) in cems.iter().enumerate() {
        let diff = (c >> 2) - base;
        assert!(diff <= 1, "CEM classes must differ by <= 1");
        enc |= diff << (2 + i as u32);
    }
    for (i, &c) in cems.iter().enumerate() {
        enc |= (c & 0x3) << (2 + pc + 2 * i as u32);
    }
    astc_set_bits(blk, 23, 6, enc & 0x3F);
    astc_set_bits(blk, 128 - weight_bits - highpart, highpart, enc >> 6);
}

/// Shared-class low-quant colour configs:
/// `(block_mode, weights_x, weights_y, weight_levels, weight_bits, partition_count, cem)`.
/// Each exercises a non-identity colour quant level across 2/3/4 partitions and
/// all three colour ISE forms (bit / trit / quint).
const LOWQUANT_SHARED: [(u32, u32, u32, u32, u32, u32, u32); 6] = [
    (19, 4, 2, 8, 24, 2, 8),    // ic12, cb75 -> QUANT_64  (bits)  ROW6
    (50, 4, 3, 5, 28, 2, 8),    // ic12, cb71 -> QUANT_48  (trit)  ROW6
    (35, 4, 3, 6, 32, 2, 8),    // ic12, cb67 -> QUANT_40  (quint) ROW6
    (19, 4, 2, 8, 24, 3, 8),    // ic18, cb75 -> QUANT_16  (bits)  ROW9
    (814, 2, 3, 16, 24, 2, 12), // ic16, cb75 -> QUANT_24  (trit)  ROW8
    (462, 3, 4, 4, 24, 4, 6),   // ic16, cb75 -> QUANT_24  (trit)  ROW8, 4 partitions
];

/// GPU parity for multi-partition single-plane LDR blocks whose colour endpoints
/// use a non-identity (low) quant level in a shared CEM class. Proves the
/// `color_quant_level` ROW5-ROW9 path and the trit/quint colour ISE forms match
/// the Metal ASTC hardware decoder within 1 LSB on every texel and channel.
#[test]
fn astc_multi_partition_lowquant_shared_cem_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-partition low-quant parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC multi-partition low-quant parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x4C51_A17C);
    const PER_MODE: u32 = 128;
    for (mode, wx, wy, levels, weight_bits, pc, cem) in LOWQUANT_SHARED {
        let (form, bits) = levels_to_grid_form(levels);
        let weight_count = (wx * wy) as usize;
        let ic = (pc * cem_int_count(cem)) as usize;
        let color_bits = 99 - weight_bits as i32; // COLOR_BITS_ARR[2..=4] == 99, shared class
        assert!(color_bits >= 0);
        let level = color_quant_level_test(ic as u32, color_bits as usize);
        assert!(
            level >= 4,
            "config mode {mode} pc {pc} cem {cem} derived level {level} below QUANT_6"
        );
        for _ in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, pc - 1);
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF);
            astc_set_bits(&mut blk, 23, 6, cem << 2); // shared class (base class 0)
            set_color_endpoints(&mut blk, 29, level, ic, &mut rng);

            let mut weights = [0u8; 64];
            for w in weights.iter_mut().take(weight_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &weights[..weight_count]);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!(
                    "low-quant mode {mode} (pc {pc}, cem {cem}, level {level}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC low-quant mode {mode} (pc {pc}, cem {cem}, level {level}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition low-quant shared-CEM parity: {} configs x {PER_MODE} blocks within 1 LSB of hardware",
        LOWQUANT_SHARED.len()
    );
}

/// Per-partition CEM-class configs:
/// `(block_mode, weights_x, weights_y, weight_levels, weight_bits, &[cem_per_partition])`.
/// Every class pair differs by at most one step so the base-class encoding is
/// representable; all use weight_bits == 24.
const PER_PARTITION_CEM: [(u32, u32, u32, u32, u32, &[u32]); 4] = [
    (19, 4, 2, 8, 24, &[8, 4]),     // ic10 classes {2,1} base1 -> ROW5
    (814, 2, 3, 16, 24, &[12, 8]),  // ic14 classes {3,2} base2 -> ROW7
    (431, 3, 3, 6, 24, &[8, 8, 8]), // ic18 classes {2,2,2} base2, highpart 5 -> ROW9
    (34, 4, 3, 4, 24, &[8, 8]),     // ic12 classes {2,2} base2 -> ROW6
];

/// GPU parity for multi-partition single-plane LDR blocks that use the
/// per-partition colour-endpoint-mode class field (base class != 0). Proves the
/// per-partition CEM decode branch and its high-part bit placement match the
/// Metal ASTC hardware decoder within 1 LSB on every texel and channel.
#[test]
fn astc_multi_partition_per_partition_cem_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC per-partition CEM parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC per-partition CEM parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x9E_C0DE11);
    const PER_MODE: u32 = 128;
    for (mode, wx, wy, levels, weight_bits, cems) in PER_PARTITION_CEM {
        let pc = cems.len() as u32;
        let (form, bits) = levels_to_grid_form(levels);
        let weight_count = (wx * wy) as usize;
        let ic = cems.iter().map(|&c| cem_int_count(c)).sum::<u32>() as usize;
        let highpart = 3 * pc - 4;
        let color_bits = 99 - weight_bits as i32 - highpart as i32;
        assert!(color_bits >= 0);
        let level = color_quant_level_test(ic as u32, color_bits as usize);
        assert!(
            level >= 4,
            "config mode {mode} cems {cems:?} derived level {level} below QUANT_6"
        );
        for _ in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, pc - 1);
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF);
            set_cem_per_partition(&mut blk, cems, weight_bits);
            set_color_endpoints(&mut blk, 29, level, ic, &mut rng);

            let mut weights = [0u8; 64];
            for w in weights.iter_mut().take(weight_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &weights[..weight_count]);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!(
                    "per-partition mode {mode} (cems {cems:?}, level {level}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC per-partition mode {mode} (cems {cems:?}, level {level}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition per-partition-CEM parity: {} configs x {PER_MODE} blocks within 1 LSB of hardware",
        PER_PARTITION_CEM.len()
    );
}

// ---------------------------------------------------------------------------
// ASTC multi-partition DUAL-PLANE LDR parity (Milestone #9: dual-plane x
// multi-partition).
//
// Previously `multi_partition.rs` rejected every dual-plane block; it now
// decodes two- and three-partition dual-plane blocks (four-partition dual-plane
// is forbidden by the spec and still rejected). A dual-plane block carries two
// interleaved weight planes plus a 2-bit colour component selector (CCS) that
// routes one channel to plane 1 while the other three use plane 0; the CCS sits
// immediately below the weight region and, when the per-partition CEM form is
// used, below its high part as well. The partition hash still assigns each
// texel to a region, and that region's endpoint pair is interpolated with the
// per-channel plane weight.
//
// Test A drives the shared CEM class (base class 0) at identity QUANT_256
// colour so endpoints are raw 8-bit integers written forward from bit 29. Each
// tuple is (block_mode, weights_x, weights_y, weight_levels, weight_bits,
// partition_count, cem, color_integer_count). color_bits =
// 99 - weight_bits - 2 (the dual-plane CCS steals two bits; the shared class
// spends no CEM high part); every config keeps color_integer_count*8 <=
// color_bits so the colour ISE stays at QUANT_256, and the endpoint run
// [29, 29 + 8*n_int) never reaches the CCS/weight region at the top. The grid
// shapes and weight ISE forms (pure-bit, trit, quint) are drawn from the
// GPU-proven single-partition dual-plane mode list.
const MULTI_PART_DUAL_PLANE_SHARED: [(u32, u32, u32, u32, u32, u32, u32, u32); 7] = [
    (1057, 4, 3, 2, 24, 2, 0, 4),
    (1041, 4, 2, 3, 26, 2, 6, 8),
    (1805, 2, 2, 10, 27, 2, 4, 8),
    (1089, 4, 4, 2, 32, 2, 6, 8),
    (1026, 4, 2, 4, 32, 3, 0, 6),
    (1057, 4, 3, 2, 24, 3, 0, 6),
    (1041, 4, 2, 3, 26, 3, 0, 6),
];

#[test]
fn astc_multi_partition_dual_plane_shared_cem_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-partition dual-plane parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC multi-partition dual-plane parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x3C9A_71E5);
    const PER_MODE: u32 = 128;
    for (mode, wx, wy, levels, weight_bits, pc, cem, n_int) in MULTI_PART_DUAL_PLANE_SHARED {
        let (form, bits) = levels_to_grid_form(levels);
        // Dual-plane stores two interleaved weights per grid point.
        let seq_count = (wx * wy * 2) as usize;
        // Shared class spends no CEM high part, so the CCS sits directly below
        // the weight region.
        let ccs_pos = 128 - weight_bits - 2;
        for n in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, pc - 1); // partition count minus one
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF); // partition seed

            // Shared CEM class (base class 0): the colour format sits in bits
            // 2..6 of the 6-bit field at [23,29); the two low base-class bits
            // are zero so no CEM high part is spent.
            astc_set_bits(&mut blk, 23, 6, cem << 2);

            // n_int raw 8-bit (QUANT_256) colour integers written forward from
            // bit 29 (= 19 + PARTITION_INDEX_BITS).
            for i in 0..n_int {
                astc_set_bits(&mut blk, 29 + i * 8, 8, rng.byte() as u32);
            }

            // 2 * wx * wy interleaved grid weights (even -> plane 0, odd ->
            // plane 1), laid into the bit-reversed weight region.
            let mut seq = [0u8; 64];
            for w in seq.iter_mut().take(seq_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &seq[..seq_count]);

            // Colour component selector: cycle all four channels.
            let ccs = n % 4;
            astc_set_bits(&mut blk, ccs_pos, 2, ccs);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!(
                    "mp dual-plane shared mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}, ccs {ccs}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC mp dual-plane shared mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}, ccs {ccs}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition dual-plane (shared CEM) parity: {} configs x {PER_MODE} blocks within 1 LSB of hardware",
        MULTI_PART_DUAL_PLANE_SHARED.len()
    );
}

// Test B drives the PER-PARTITION CEM class form (base class != 0) together
// with dual-plane weights, so the CEM high part (3*pc - 4 bits, here 2 bits for
// two partitions) sits just below the weight region and the CCS sits two bits
// below *that*. Both partitions use CEM 4 (luminance+alpha, endpoint class 1)
// expressed through the per-partition encoding: the 6-bit field at [23,29) is
// 0b000010 (base class 2 => class 1 per partition, all class/low bits zero) and
// the 2-bit high part is zero. color_bits = 99 - weight_bits - 2 (high part) -
// 2 (CCS); every config keeps 8*8 = 64 <= color_bits so colour stays at
// QUANT_256 and endpoints remain raw 8-bit integers from bit 29.
const MULTI_PART_DUAL_PLANE_PERPART: [(u32, u32, u32, u32, u32); 3] = [
    (1057, 4, 3, 2, 24),
    (1041, 4, 2, 3, 26),
    (1805, 2, 2, 10, 27),
];

#[test]
fn astc_multi_partition_dual_plane_per_partition_cem_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-partition dual-plane per-partition parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC)
    {
        eprintln!("adapter lacks ASTC support; skipping ASTC mp dual-plane per-partition parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Unorm,
    };
    let mut rng = Rng(0x1B57_9EEF);
    const PER_MODE: u32 = 128;
    const PC: u32 = 2;
    const N_INT: u32 = 8; // 2 partitions x CEM4 (4 integers each)
    for (mode, wx, wy, levels, weight_bits) in MULTI_PART_DUAL_PLANE_PERPART {
        let (form, bits) = levels_to_grid_form(levels);
        let seq_count = (wx * wy * 2) as usize;
        // Per-partition form spends a (3*pc - 4)-bit high part just below the
        // weight region; the CCS sits two bits below the high part.
        let highpart = 3 * PC - 4; // == 2 for two partitions
        let highpart_pos = 128 - weight_bits - highpart;
        let ccs_pos = highpart_pos - 2;
        for n in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, PC - 1);
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF);

            // Per-partition CEM: 6-bit field = base class 2 (=> class 1 = CEM4
            // for both partitions), high part = 0.
            astc_set_bits(&mut blk, 23, 6, 0b00_0010);
            astc_set_bits(&mut blk, highpart_pos, highpart, 0);

            for i in 0..N_INT {
                astc_set_bits(&mut blk, 29 + i * 8, 8, rng.byte() as u32);
            }

            let mut seq = [0u8; 64];
            for w in seq.iter_mut().take(seq_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &seq[..seq_count]);

            let ccs = n % 4;
            astc_set_bits(&mut blk, ccs_pos, 2, ccs);

            let cpu = decode_astc_4x4_ldr(&blk).unwrap_or_else(|e| {
                panic!(
                    "mp dual-plane per-part mode {mode} ({wx}x{wy}, ccs {ccs}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_unorm8(format, &blk);
            for t in 0..16 {
                for c in 0..4 {
                    let d = (cpu[t][c] as i32 - gpu[t][c] as i32).abs();
                    assert!(
                        d <= 1,
                        "ASTC mp dual-plane per-part mode {mode} ({wx}x{wy}, ccs {ccs}) block={blk:02x?} texel {t} chan {c}: cpu={} gpu={} (|d|={d})",
                        cpu[t][c],
                        gpu[t][c]
                    );
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition dual-plane (per-partition CEM) parity: {} configs x {PER_MODE} blocks within 1 LSB of hardware",
        MULTI_PART_DUAL_PLANE_PERPART.len()
    );
}

// ---------------------------------------------------------------------------
// ASTC single-partition HDR parity (the six HDR Colour Endpoint Modes).
//
// Proves `decode_astc_4x4_hdr` against the Metal ASTC-HDR hardware decoder.
// Each block uses block mode 67 (trit weights, 1 low bit, 6 levels), whose
// single-partition colour budget is `color_bits = 111 - 42 = 69`, mapping to
// QUANT_256 for every HDR integer count (<= 8), so the colour integers are the
// raw 8-bit values laid directly at bit 17 (no colour unquant table). For each
// HDR CEM we emit random 8-bit endpoint integers and random trit weights, then
// compare the RGB FP16 output. The oracle drops the alpha lane, so alpha is
// implemented (CEM 14 linear, others LNS) but only RGB is hardware-proven here.
//
// Both the Metal decoder and this CPU path follow the Khronos ASTC spec: the
// endpoint lanes are interpolated in the integer domain, then mapped to FP16
// via a deterministic LNS / UNORM16 conversion, so agreement is near-exact.
// FP16 saturates at 65504 (0x7BFF); texels that reach that clamp (or that the
// hardware renders as non-finite) are a spec-boundary ambiguity and are counted
// and skipped rather than asserted, so the proof covers the representable range.
// ---------------------------------------------------------------------------

/// Build a single-partition HDR block of block mode `bm` and CEM `cem`, laying
/// `ep` as consecutive 8-bit (QUANT_256) colour integers at bits [17..].
fn astc_hdr_block(bm: u32, cem: u32, ep: &[u8]) -> [u8; 16] {
    let mut b = [0u8; 16];
    astc_set_bits(&mut b, 0, 11, bm);
    astc_set_bits(&mut b, 13, 4, cem);
    for (i, v) in ep.iter().enumerate() {
        astc_set_bits(&mut b, 17 + i as u32 * 8, 8, u32::from(*v));
    }
    b
}

#[test]
fn astc_single_partition_hdr_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC single-partition HDR parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC_HDR)
    {
        eprintln!("adapter lacks ASTC HDR support; skipping ASTC single-partition HDR parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Hdr,
    };
    // (CEM, integer_count): LUM_LARGE=2, LUM_SMALL=2, RGB_SCALE=4, RGB=6,
    // RGB_LDR_ALPHA=8, RGB_HDR_ALPHA=8.
    const HDR_CEMS: [(u32, usize); 6] = [(2, 2), (3, 2), (7, 4), (11, 6), (14, 8), (15, 8)];
    const BM: u32 = 67; // trit weights, 1 low bit, 6 levels
    const PER_CEM: u32 = 128;
    let mut rng = Rng(0x4D7_0C7A);
    let mut compared = 0u64;
    let mut skipped = 0u64;
    for (cem, int_count) in HDR_CEMS {
        for _ in 0..PER_CEM {
            let mut ep = [0u8; 8];
            for e in ep[..int_count].iter_mut() {
                *e = (rng.next_u32() & 0xFF) as u8;
            }
            let mut blk = astc_hdr_block(BM, cem, &ep[..int_count]);
            let mut raw = [0u8; 16];
            for r in raw.iter_mut() {
                *r = (rng.next_u32() % 6) as u8;
            }
            astc_set_weights_ise(&mut blk, WeightForm::Trit, 1, &raw);

            let cpu = decode_astc_4x4_hdr(&blk).expect("supported single-partition HDR block");
            let gpu = oracle.decode_rgb_f32(format, &blk);
            for t in 0..16 {
                for c in 0..3 {
                    let (cv, gv) = (cpu[t][c], gpu[t][c]);
                    // Skip the FP16 saturation boundary: at/above ~65504 the CPU
                    // clamps to the max finite half while hardware may emit +Inf.
                    if !cv.is_finite() || !gv.is_finite() || cv.abs() >= 6.5e4 {
                        skipped += 1;
                        continue;
                    }
                    let tol = cv.abs() * 1e-3 + 1e-3;
                    assert!(
                        (cv - gv).abs() <= tol,
                        "ASTC HDR cem={cem} bm={BM} block={blk:02x?} texel {t} chan {c}: ep={:?} raw_w={} cpu={cv} gpu={gv}",
                        &ep[..int_count],
                        raw[t]
                    );
                    compared += 1;
                }
            }
        }
    }
    eprintln!(
        "ASTC single-partition HDR parity: {compared} RGB lanes match hardware across {} CEMs x {PER_CEM} blocks ({skipped} saturated lanes skipped)",
        HDR_CEMS.len()
    );
}

// ---------------------------------------------------------------------------
// ASTC multi-partition HDR parity (Milestone #11: multi-partition x HDR).
//
// `decode_astc_4x4_hdr` now decodes 2/3/4-partition blocks when *every*
// partition uses an HDR Colour Endpoint Mode. The header, partition seed and
// colour integer-sequence parse are shared verbatim with the LDR
// multi-partition path (`parse_multi_partition_color`); only the per-partition
// endpoint expansion (HDR unpack) and the logarithmic FP16 interpolation
// differ. These tests prove that path bit-for-bit against the Metal ASTC-HDR
// hardware decoder.
//
// Both tests drive the shared CEM class (base class 0) at identity QUANT_256
// colour, so each colour integer is a raw 8-bit value written forward from bit
// 29 (= 19 + PARTITION_INDEX_BITS). Each tuple is (block_mode, weights_x,
// weights_y, weight_levels, weight_bits, partition_count, cem,
// color_integer_count); color_bits = 99 - weight_bits (single plane) or
// 99 - weight_bits - 2 (dual plane steals two CCS bits), and every config keeps
// color_integer_count*8 <= color_bits so the colour ISE stays at QUANT_256 and
// the endpoint run never reaches the top weight/CCS region.
//
// As with the single-partition HDR proof, the oracle drops alpha (RGB only) and
// FP16 saturates at ~65504; lanes at/above that clamp (or that hardware renders
// non-finite) are a spec-boundary ambiguity and are counted and skipped rather
// than asserted.

/// Single-plane multi-partition HDR configs across 2/3/4 partitions, exercising
/// three HDR CEMs (2 LUM_LARGE, 3 LUM_SMALL, 7 RGB_SCALE) and the pure-bit,
/// trit and quint weight-grid forms. Modes are drawn from the GPU-proven
/// single-plane multi-partition infill set (weight_bits == 24).
const MULTI_PART_HDR_SINGLE_PLANE: [(u32, u32, u32, u32, u32, u32, u32, u32); 7] = [
    (19, 4, 2, 8, 24, 2, 2, 4),
    (814, 2, 3, 16, 24, 2, 3, 4),
    (431, 3, 3, 6, 24, 3, 2, 6),
    (34, 4, 3, 4, 24, 2, 7, 8),
    (351, 2, 4, 8, 24, 2, 7, 8),
    (462, 3, 4, 4, 24, 4, 2, 8),
    (910, 3, 2, 16, 24, 4, 3, 8),
];

/// GPU parity for single-plane multi-partition HDR blocks: proves per-partition
/// HDR endpoint unpack + logarithmic interpolation + the shared partition hash
/// match the Metal ASTC-HDR hardware decoder across 2/3/4 partitions.
#[test]
fn astc_multi_partition_hdr_single_plane_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-partition HDR parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC_HDR)
    {
        eprintln!("adapter lacks ASTC HDR support; skipping ASTC multi-partition HDR parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Hdr,
    };
    let mut rng = Rng(0x51D2_7E4B);
    const PER_MODE: u32 = 128;
    let mut compared = 0u64;
    let mut skipped = 0u64;
    for (mode, wx, wy, _levels_doc, _wb_doc, pc, cem, n_int) in MULTI_PART_HDR_SINGLE_PLANE {
        let (form, bits) = levels_to_grid_form(_levels_doc);
        let weight_count = (wx * wy) as usize;
        for _ in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, pc - 1); // partition count minus one
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF); // partition seed
                                                                     // Shared CEM class (base class 0): colour format in bits 2..6 of the
                                                                     // 6-bit field at [23,29); no CEM high part is spent.
            astc_set_bits(&mut blk, 23, 6, cem << 2);
            // n_int raw 8-bit (QUANT_256) colour integers forward from bit 29.
            for i in 0..n_int {
                astc_set_bits(&mut blk, 29 + i * 8, 8, rng.byte() as u32);
            }
            // Shared single-plane weight stream (bit-reversed at the top).
            let mut weights = [0u8; 64];
            for w in weights.iter_mut().take(weight_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &weights[..weight_count]);

            let cpu = decode_astc_4x4_hdr(&blk).unwrap_or_else(|e| {
                panic!(
                    "mp HDR mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_rgb_f32(format, &blk);
            for t in 0..16 {
                for c in 0..3 {
                    let (cv, gv) = (cpu[t][c], gpu[t][c]);
                    if !cv.is_finite() || !gv.is_finite() || cv.abs() >= 6.5e4 {
                        skipped += 1;
                        continue;
                    }
                    let tol = cv.abs() * 1e-3 + 1e-3;
                    assert!(
                        (cv - gv).abs() <= tol,
                        "ASTC mp HDR mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}) block={blk:02x?} texel {t} chan {c}: cpu={cv} gpu={gv}"
                    );
                    compared += 1;
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition HDR single-plane parity: {compared} RGB lanes match hardware across {} configs x {PER_MODE} blocks ({skipped} saturated lanes skipped)",
        MULTI_PART_HDR_SINGLE_PLANE.len()
    );
}

/// Dual-plane multi-partition HDR configs (2/3 partitions; four-partition
/// dual-plane is spec-forbidden). Modes are drawn from the GPU-proven
/// single-partition dual-plane set; the CCS sits directly below the weight
/// region (shared class spends no CEM high part).
const MULTI_PART_HDR_DUAL_PLANE: [(u32, u32, u32, u32, u32, u32, u32, u32); 7] = [
    (1057, 4, 3, 2, 24, 2, 7, 8),
    (1041, 4, 2, 3, 26, 2, 7, 8),
    (1805, 2, 2, 10, 27, 2, 7, 8),
    (1089, 4, 4, 2, 32, 2, 3, 4),
    (1026, 4, 2, 4, 32, 3, 2, 6),
    (1057, 4, 3, 2, 24, 3, 2, 6),
    (1041, 4, 2, 3, 26, 3, 3, 6),
];

/// GPU parity for dual-plane multi-partition HDR blocks: proves the two-plane
/// colour-component selector routing combined with per-partition HDR endpoints
/// and logarithmic interpolation matches the Metal ASTC-HDR hardware decoder.
#[test]
fn astc_multi_partition_hdr_dual_plane_parity_against_gpu_hardware_decode() {
    let Some(oracle) = BlockOracle::try_new() else {
        eprintln!("no GPU adapter; skipping ASTC multi-partition HDR dual-plane parity");
        return;
    };
    if !oracle
        .features()
        .contains(Features::TEXTURE_COMPRESSION_ASTC_HDR)
    {
        eprintln!("adapter lacks ASTC HDR support; skipping ASTC mp HDR dual-plane parity");
        return;
    }
    let format = TextureFormat::Astc {
        block: wgpu::AstcBlock::B4x4,
        channel: wgpu::AstcChannel::Hdr,
    };
    let mut rng = Rng(0x2B8E_15C7);
    const PER_MODE: u32 = 128;
    let mut compared = 0u64;
    let mut skipped = 0u64;
    for (mode, wx, wy, levels, weight_bits, pc, cem, n_int) in MULTI_PART_HDR_DUAL_PLANE {
        let (form, bits) = levels_to_grid_form(levels);
        let seq_count = (wx * wy * 2) as usize;
        // Shared class spends no CEM high part, so the CCS sits directly below
        // the weight region.
        let ccs_pos = 128 - weight_bits - 2;
        for n in 0..PER_MODE {
            let mut blk = [0u8; 16];
            astc_set_bits(&mut blk, 0, 11, mode);
            astc_set_bits(&mut blk, 11, 2, pc - 1);
            astc_set_bits(&mut blk, 13, 10, rng.next_u32() & 0x3FF);
            astc_set_bits(&mut blk, 23, 6, cem << 2);
            for i in 0..n_int {
                astc_set_bits(&mut blk, 29 + i * 8, 8, rng.byte() as u32);
            }
            // 2 * wx * wy interleaved grid weights (even -> plane 0, odd ->
            // plane 1), laid into the bit-reversed weight region.
            let mut seq = [0u8; 64];
            for w in seq.iter_mut().take(seq_count) {
                *w = rand_grid_weight(&mut rng, form, bits);
            }
            astc_set_grid_weights(&mut blk, form, bits, &seq[..seq_count]);
            // Colour component selector: cycle all four channels.
            let ccs = n % 4;
            astc_set_bits(&mut blk, ccs_pos, 2, ccs);

            let cpu = decode_astc_4x4_hdr(&blk).unwrap_or_else(|e| {
                panic!(
                    "mp HDR dual-plane mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}, ccs {ccs}) block {blk:02x?} rejected: {e:?}"
                )
            });
            let gpu = oracle.decode_rgb_f32(format, &blk);
            for t in 0..16 {
                for c in 0..3 {
                    let (cv, gv) = (cpu[t][c], gpu[t][c]);
                    if !cv.is_finite() || !gv.is_finite() || cv.abs() >= 6.5e4 {
                        skipped += 1;
                        continue;
                    }
                    let tol = cv.abs() * 1e-3 + 1e-3;
                    assert!(
                        (cv - gv).abs() <= tol,
                        "ASTC mp HDR dual-plane mode {mode} ({wx}x{wy}, pc {pc}, cem {cem}, ccs {ccs}) block={blk:02x?} texel {t} chan {c}: cpu={cv} gpu={gv}"
                    );
                    compared += 1;
                }
            }
        }
    }
    eprintln!(
        "ASTC multi-partition HDR dual-plane parity: {compared} RGB lanes match hardware across {} configs x {PER_MODE} blocks ({skipped} saturated lanes skipped)",
        MULTI_PART_HDR_DUAL_PLANE.len()
    );
}
