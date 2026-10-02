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
    decode_bc1, decode_bc3, decode_bc6h_mode12_signed, decode_bc6h_mode12_unsigned,
    decode_bc6h_mode13_signed, decode_bc6h_mode13_unsigned, decode_bc6h_mode14_signed,
    decode_bc6h_mode14_unsigned, decode_bc6h_mode1_signed, decode_bc6h_mode1_unsigned,
    decode_bc6h_mode2_signed, decode_bc6h_mode2_unsigned, decode_bc6h_mode3_signed,
    decode_bc6h_mode3_unsigned, decode_bc6h_mode4_signed, decode_bc6h_mode4_unsigned,
    decode_bc6h_signed, decode_bc6h_unsigned, decode_bc7, decode_bc7_mode0, decode_bc7_mode1,
    decode_bc7_mode2, decode_bc7_mode3, decode_bc7_mode7, encode_bc1, encode_bc3,
    encode_bc6h_mode11_unsigned, encode_bc7_mode4, encode_bc7_mode5, encode_bc7_mode6,
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
];

/// Dispatch to the direct per-mode decoder so the parametric test also proves
/// the exported single-mode entry points match the generic dispatcher.
fn decode_two_subset_direct(mode_bits: u32, block: &[u8; 16], signed: bool) -> [[f32; 3]; 16] {
    match (mode_bits, signed) {
        (0b00010, false) => decode_bc6h_mode3_unsigned(block),
        (0b00010, true) => decode_bc6h_mode3_signed(block),
        (0b00110, false) => decode_bc6h_mode4_unsigned(block),
        (0b00110, true) => decode_bc6h_mode4_signed(block),
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
