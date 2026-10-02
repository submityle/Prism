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
    decode_bc1, decode_bc3, decode_bc6h_unsigned, decode_bc7, decode_bc7_mode0, decode_bc7_mode1,
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
