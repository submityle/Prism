//! Independent bitstream construction; successful bytes must come from GPU.

use prism_gdeflate_gpu::{
    DecodePath, DeflateDecodeError, GDeflateDecoder, GpuContext, GpuDeflate, GpuFallbackReason,
};
use prism_render_architecture::compression::gdeflate_compress;

#[derive(Default)]
struct Bits(Vec<bool>);

impl Bits {
    fn put(&mut self, value: u32, count: usize) {
        self.0.extend((0..count).map(|i| value & (1 << i) != 0));
    }

    fn code(&mut self, lengths: &[u8], symbol: usize) {
        let width = lengths[symbol];
        assert!(width > 0);
        let mut counts = [0u32; 16];
        for &bits in lengths {
            if bits > 0 {
                counts[bits as usize] += 1;
            }
        }
        let mut first = 0;
        for bits in 1..=usize::from(width) {
            first = (first + counts[bits - 1]) << 1;
        }
        let code = first
            + lengths[..symbol]
                .iter()
                .filter(|&&bits| bits == width)
                .count() as u32;
        self.0
            .extend((0..width).rev().map(|i| code & (1 << i) != 0));
    }

    fn align(&mut self) {
        while !self.0.len().is_multiple_of(8) {
            self.0.push(false);
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0; self.0.len().div_ceil(8)];
        for (i, &bit) in self.0.iter().enumerate() {
            out[i / 8] |= u8::from(bit) << (i % 8);
        }
        out
    }
}

const ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn cl_lengths() -> Vec<u8> {
    (0..19).map(|i| if i < 13 { 4 } else { 5 }).collect()
}

fn header(w: &mut Bits, last: bool, hlit: usize, hdist: usize, cl: &[u8]) {
    w.put(u32::from(last), 1);
    w.put(2, 2);
    w.put((hlit - 257) as u32, 5);
    w.put((hdist - 1) as u32, 5);
    w.put(15, 4);
    for i in ORDER {
        w.put(u32::from(cl[i]), 3);
    }
}

// Return the actual repeat symbols emitted so tests prove all 16/17/18 paths.
fn dynamic(w: &mut Bits, last: bool, lit: &[u8], dist: &[u8]) -> Vec<usize> {
    let cl = cl_lengths();
    header(w, last, lit.len(), dist.len(), &cl);
    let lengths: Vec<u8> = lit.iter().chain(dist).copied().collect();
    let mut i = 0;
    let mut repeats = Vec::new();
    while i < lengths.len() {
        let value = lengths[i];
        let run = lengths[i..].iter().take_while(|&&v| v == value).count();
        let (symbol, count, extra, width) = if value == 0 && run >= 11 {
            (18, run.min(138), run.min(138) - 11, 7)
        } else if value == 0 && run >= 3 {
            (17, run.min(10), run.min(10) - 3, 3)
        } else if i > 0 && value == lengths[i - 1] && run >= 3 {
            (16, run.min(6), run.min(6) - 3, 2)
        } else {
            (usize::from(value), 1, 0, 0)
        };
        w.code(&cl, symbol);
        w.put(extra as u32, width);
        if symbol >= 16 {
            repeats.push(symbol);
        }
        i += count;
    }
    repeats
}

fn balanced() -> (Vec<u8>, Vec<u8>) {
    (
        (0..286).map(|i| if i < 226 { 8 } else { 9 }).collect(),
        vec![5; 32],
    )
}

fn fixed_lengths() -> Vec<u8> {
    (0..288)
        .map(|i| match i {
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        })
        .collect()
}

fn stored(w: &mut Bits, last: bool, data: &[u8]) {
    w.put(u32::from(last), 1);
    w.put(0, 2);
    w.align();
    w.put(data.len() as u32, 16);
    w.put(u32::from(!(data.len() as u16)), 16);
    for &byte in data {
        w.put(u32::from(byte), 8);
    }
}

fn interleave(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0; raw.len().div_ceil(128) * 128];
    let rounds = out.len() / 128;
    for (i, &byte) in raw.iter().enumerate() {
        let word = i / 4;
        out[(word % 32 * rounds + word / 32) * 4 + i % 4] = byte;
    }
    out
}

fn container(raw: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = b"GDF1\x01\x20\x10\0".to_vec();
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    let hash = data.iter().fold(0x811c9dc5u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x01000193)
    });
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&interleave(raw));
    out
}

fn check(ctx: &GpuContext, gpu: &GpuDeflate, raw: &[u8], expected: &[u8]) {
    assert_eq!(
        gpu.decompress(ctx, &interleave(raw), raw.len(), expected.len())
            .unwrap(),
        expected
    );
    let decoded = GDeflateDecoder::new(ctx)
        .decode(ctx, &container(raw, expected))
        .unwrap();
    assert_eq!(decoded.bytes, expected);
    assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
    assert_eq!(decoded.report.fallback, None);
    assert!(decoded.report.gpu_attempted);
}

#[test]
fn dynamic_all_literals_repeats_and_fifteen_bit_codes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let (lit, dist) = balanced();
    let mut w = Bits::default();
    assert!(dynamic(&mut w, true, &lit, &dist).contains(&16));
    for b in 0..256 {
        w.code(&lit, b);
    }
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), &(0..=255).collect::<Vec<u8>>());

    // A complete comb tree exercises every width 1..15, including EOB=15.
    let mut lit = vec![0; 257];
    for (i, value) in lit.iter_mut().take(14).enumerate() {
        *value = (i + 1) as u8;
    }
    lit[14] = 15;
    lit[256] = 15;
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &[0]);
    for b in 0..15 {
        w.code(&lit, b);
    }
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), &(0..15).collect::<Vec<u8>>());

    // Degenerate legal EOB-only and literals-only trees, empty distance tree.
    let mut lit = vec![0; 257];
    lit[256] = 1;
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &[0]);
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), &[]);
    lit[65] = 1;
    let mut w = Bits::default();
    let repeats = dynamic(&mut w, true, &lit, &[0; 5]);
    assert!(repeats.contains(&17) && repeats.contains(&18));
    w.code(&lit, 65);
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), b"A");

    // Repeat 16 spans the literal/distance boundary, as required by DEFLATE.
    let mut lit = vec![0; 257];
    lit[65] = 1;
    lit[255] = 2;
    lit[256] = 2;
    let mut w = Bits::default();
    assert!(dynamic(&mut w, true, &lit, &[2; 4]).contains(&16));
    w.code(&lit, 65);
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), b"A");
}

#[test]
fn mixed_every_transition_alignment_and_history_across_blocks() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let (lit, dist) = balanced();
    let fixed = fixed_lengths();
    // Every ordered pair of block types, and every starting bit alignment
    // before a stored block. Dynamic matches reference earlier blocks.
    for a in 0..3 {
        for b in 0..3 {
            for offset in 0..8 {
                let mut w = Bits::default();
                w.put(0, 1);
                w.put(1, 2);
                for _ in 0..offset {
                    w.code(&fixed, 200);
                }
                w.code(&fixed, 256);
                let mut expected = vec![200; offset];
                for (index, kind) in [a, b, 2].into_iter().enumerate() {
                    let last = index == 2;
                    if kind == 0 {
                        stored(&mut w, last, b"abc");
                        expected.extend_from_slice(b"abc");
                        continue;
                    }
                    let codes = if kind == 1 {
                        w.put(u32::from(last), 1);
                        w.put(1, 2);
                        &fixed
                    } else {
                        dynamic(&mut w, last, &lit, &dist);
                        &lit
                    };
                    if index == 0 {
                        w.code(codes, 65);
                        expected.push(65);
                    }
                    // length=3, distance=1, overlapping copy across boundary.
                    w.code(codes, 257);
                    w.code(if kind == 1 { &[5; 32] } else { &dist }, 0);
                    expected.extend_from_slice(&[*expected.last().unwrap(); 3]);
                    w.code(codes, 256);
                }
                check(&ctx, &gpu, &w.bytes(), &expected);
            }
        }
    }
}

#[test]
fn dynamic_max_tile_long_distance_and_tree_replacement() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let (lit, _) = balanced();
    let mut w = Bits::default();
    let prefix: Vec<u8> = (0..32768).map(|i| (i * 37) as u8).collect();
    stored(&mut w, false, &prefix);
    let mut deep_dist = vec![0; 30];
    for (i, bits) in deep_dist.iter_mut().take(14).enumerate() {
        *bits = (i + 1) as u8;
    }
    deep_dist[14] = 15;
    deep_dist[29] = 15;
    dynamic(&mut w, false, &lit, &deep_dist);
    w.code(&lit, 285); // length 258
    w.code(&deep_dist, 29);
    w.put(8191, 13); // distance 32768
    w.code(&lit, 256);
    let mut expected = prefix.clone();
    expected.extend_from_slice(&prefix[..258]);
    // Different dynamic alphabet, one-bit distance tree; reset old tables.
    let mut lit = vec![0; 286];
    lit[65] = 1;
    lit[256] = 2;
    lit[285] = 2;
    dynamic(&mut w, false, &lit, &[1]);
    w.code(&lit, 65);
    expected.push(65);
    w.code(&lit, 285);
    w.code(&[1], 0);
    expected.extend_from_slice(&[65; 258]);
    w.code(&lit, 256);
    let tail = vec![0x7a; 65536 - expected.len()];
    stored(&mut w, true, &tail);
    expected.extend_from_slice(&tail);
    check(&ctx, &gpu, &w.bytes(), &expected);
}

#[test]
fn dynamic_every_length_and_distance_extra_bit_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let (lit, dist) = balanced();
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &dist);
    w.code(&lit, 65);
    let length_base = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    let length_extra = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    let mut size = 1;
    for length in 3..=258 {
        let index = length_base
            .iter()
            .rposition(|&base| length >= base)
            .unwrap();
        w.code(&lit, 257 + index);
        w.put((length - length_base[index]) as u32, length_extra[index]);
        w.code(&dist, 0);
        size += length;
    }
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), &vec![65; size]);

    let distance_base = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    let distance_extra = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    let prefix: Vec<u8> = (0..32768usize)
        .map(|i| (i.wrapping_mul(71) ^ (i >> 8)) as u8)
        .collect();
    for first in (1..=32768).step_by(8192) {
        let mut w = Bits::default();
        stored(&mut w, false, &prefix);
        dynamic(&mut w, true, &lit, &dist);
        let mut expected = prefix.clone();
        for distance in first..first + 8192 {
            let index = distance_base
                .iter()
                .rposition(|&base| distance >= base)
                .unwrap();
            w.code(&lit, 257);
            w.code(&dist, index);
            w.put(
                (distance - distance_base[index]) as u32,
                distance_extra[index],
            );
            for _ in 0..3 {
                expected.push(expected[expected.len() - distance]);
            }
        }
        w.code(&lit, 256);
        check(&ctx, &gpu, &w.bytes(), &expected);
    }
}

#[test]
fn dynamic_code_length_alphabet_seven_bits_and_minimum_header() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    // Complete CL comb: lengths 1..6 and two length-7 codes. Real table
    // expansion uses the deepest CL symbols, not merely declared lengths.
    let mut cl = vec![0; 19];
    for (symbol, bits) in [
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (7, 6),
        (0, 7),
        (1, 7),
    ] {
        cl[symbol] = bits;
    }
    let mut lit = vec![0; 257];
    lit[65] = 1;
    lit[256] = 1;
    let mut w = Bits::default();
    header(&mut w, true, 257, 1, &cl);
    for &length in lit.iter().chain(&[0]) {
        w.code(&cl, usize::from(length));
    }
    w.code(&lit, 65);
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), b"A");

    // HCLEN=5, the smallest header that can describe length 8. A complete
    // 256-code literal tree includes EOB and excludes byte 255.
    let mut w = Bits::default();
    w.put(1, 1);
    w.put(2, 2);
    w.put(0, 5);
    w.put(0, 5);
    w.put(1, 4);
    for length in [0, 0, 0, 1, 1] {
        w.put(length, 3);
    }
    let mut cl = vec![0; 19];
    cl[0] = 1;
    cl[8] = 1;
    let mut lit = vec![8; 257];
    lit[255] = 0;
    for &length in lit.iter().chain(&[0]) {
        w.code(&cl, usize::from(length));
    }
    for b in 0..255 {
        w.code(&lit, b);
    }
    w.code(&lit, 256);
    check(&ctx, &gpu, &w.bytes(), &(0..255).collect::<Vec<u8>>());

    // Minimum HCLEN=4: only CL symbols 18 and 0 have a one-bit code.
    // A malformed missing-EOB tree must fail without reading stale CL state.
    let mut w = Bits::default();
    w.put(1, 1);
    w.put(2, 2);
    w.put(0, 5);
    w.put(0, 5);
    w.put(0, 4);
    for length in [0, 0, 1, 1] {
        w.put(length, 3);
    }
    let mut cl = vec![0; 19];
    cl[0] = 1;
    cl[18] = 1;
    w.code(&cl, 18);
    w.put(127, 7);
    w.code(&cl, 18);
    w.put(109, 7);
    invalid(&ctx, &gpu, &w, 0);
}

fn invalid(ctx: &GpuContext, gpu: &GpuDeflate, w: &Bits, size: usize) {
    let raw = w.bytes();
    assert!(
        gpu.decompress(ctx, &interleave(&raw), raw.len(), size)
            .is_err(),
        "accepted malformed stream: {raw:?}"
    );
}

#[test]
fn dynamic_rejects_invalid_trees_headers_repeats_and_reserved_symbols() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    for cl in [vec![0; 19], vec![1; 19], vec![7; 19]] {
        let mut w = Bits::default();
        header(&mut w, true, 257, 1, &cl);
        invalid(&ctx, &gpu, &w, 0);
    }
    // HLIT 287/288 is reserved, even when the rest of the tile is padded.
    for hlit in [287, 288] {
        let mut w = Bits::default();
        header(&mut w, true, hlit, 1, &cl_lengths());
        invalid(&ctx, &gpu, &w, 0);
    }
    let cl = cl_lengths();
    for repeat in [16, 17, 18] {
        let mut w = Bits::default();
        header(&mut w, true, 257, 1, &cl);
        if repeat != 16 {
            // Leave exactly one slot, then overrun it with a zero repeat.
            for _ in 0..257 {
                w.code(&cl, 0);
            }
        }
        w.code(&cl, repeat);
        w.put(
            0,
            if repeat == 18 {
                7
            } else if repeat == 17 {
                3
            } else {
                2
            },
        );
        invalid(&ctx, &gpu, &w, 0);
    }
    let (base, dist) = balanced();
    let mut missing_eob = base.clone();
    missing_eob[256] = 0;
    let mut oversubscribed = base.clone();
    oversubscribed[0] = 1;
    let mut incomplete = vec![0; 257];
    incomplete[65] = 2;
    incomplete[256] = 2;
    for lit in [missing_eob, oversubscribed, incomplete] {
        let mut w = Bits::default();
        dynamic(&mut w, true, &lit, &dist);
        invalid(&ctx, &gpu, &w, 8);
    }
    for bad_dist in [vec![1, 1, 1], vec![2, 2]] {
        let mut w = Bits::default();
        dynamic(&mut w, true, &base, &bad_dist);
        invalid(&ctx, &gpu, &w, 8);
    }
    for symbol in [30, 31] {
        let mut w = Bits::default();
        dynamic(&mut w, true, &base, &dist);
        w.code(&base, 65);
        w.code(&base, 257);
        w.code(&dist, symbol);
        invalid(&ctx, &gpu, &w, 4);
    }
    // Empty distance table may be declared but must never decode a match.
    let mut w = Bits::default();
    dynamic(&mut w, true, &base, &[0]);
    w.code(&base, 65);
    w.code(&base, 257);
    w.put(0, 16);
    invalid(&ctx, &gpu, &w, 4);
}

#[test]
fn dynamic_truncation_output_bounds_distance_and_stored_integrity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let (lit, dist) = balanced();
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &dist);
    for b in b"hello GPU" {
        w.code(&lit, usize::from(*b));
    }
    w.code(&lit, 256);
    let raw = w.bytes();
    for end in 1..raw.len() {
        let padded = interleave(&raw); // valid data beyond logical end must not be read
        assert!(
            gpu.decompress(&ctx, &padded, end, 9).is_err(),
            "prefix {end} accepted"
        );
    }
    for size in [0, 8, 10, 65537] {
        assert!(gpu
            .decompress(&ctx, &interleave(&raw), raw.len(), size)
            .is_err());
    }
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &dist);
    w.code(&lit, 65);
    w.code(&lit, 285);
    w.code(&dist, 0);
    w.code(&lit, 256);
    let raw_match = w.bytes();
    assert_eq!(
        gpu.decompress(&ctx, &interleave(&raw_match), raw_match.len(), 258),
        Err(DeflateDecodeError::OutputSizeMismatch)
    );
    for end in 1..raw_match.len() {
        assert!(gpu
            .decompress(&ctx, &interleave(&raw_match), end, 259)
            .is_err());
    }
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &dist);
    w.code(&lit, 257);
    w.code(&dist, 0);
    assert_eq!(
        gpu.decompress(&ctx, &interleave(&w.bytes()), w.bytes().len(), 3),
        Err(DeflateDecodeError::DistanceTooFar)
    );
    let mut w = Bits::default();
    stored(&mut w, true, b"abc");
    let raw = w.bytes();
    for end in 1..raw.len() {
        assert!(gpu.decompress(&ctx, &interleave(&raw), end, 3).is_err());
    }
    for size in [2, 4] {
        assert!(gpu
            .decompress(&ctx, &interleave(&raw), raw.len(), size)
            .is_err());
    }
    let mut bad = raw.clone();
    bad[3] ^= 1;
    assert!(gpu
        .decompress(&ctx, &interleave(&bad), bad.len(), 3)
        .is_err());
    let mut corrupt = container(&raw, b"abc");
    corrupt[28] ^= 1;
    assert!(GDeflateDecoder::new(&ctx).decode(&ctx, &corrupt).is_err());
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "record the real GPU adapter used by this test"
)]
fn independent_zlib_dynamic_fixture_decodes_on_gpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    eprintln!("full DEFLATE GPU adapter: {:?}", ctx.adapter_info());
    let gpu = GpuDeflate::new(&ctx);
    // Produced by Python zlib.compressobj(9, zlib.DEFLATED, -15), not Prism's
    // encoder or the handcrafted canonical-code helper in this test file.
    let hex = "edcab10d83301000c0559e0932062d4d0608f0912c2516022301d3932dd25c73d5f5c333e6b3bebe658ab6666ef18852e75cf2476d717dca18ef72b47dcd2e7adbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb66ddbb6ffb86f";
    let raw: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let expected = b"GPU dynamic trees / independent zlib fixture! ".repeat(200);
    assert_eq!(raw.len(), 93);
    assert_eq!(expected.len(), 9200);
    assert_eq!((raw[0] >> 1) & 3, 2);
    check(&ctx, &gpu, &raw, &expected);
}

#[test]
fn real_compressor_dynamic_multitile_never_uses_cpu_inflate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let data: Vec<u8> = (0..90000u32)
        .map(|i| {
            if i % 97 == 0 {
                (0x80 + i % 127) as u8
            } else {
                b'a'
            }
        })
        .collect();
    let encoded = gdeflate_compress(&data);
    // First tile payload follows two descriptors. Check real compressor chose
    // BTYPE=2 rather than silently exercising only stored or fixed Huffman.
    assert_eq!((encoded[44] >> 1) & 3, 2);
    let decoded = GDeflateDecoder::new(&ctx).decode(&ctx, &encoded).unwrap();
    assert_eq!(decoded.bytes, data);
    assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
    assert_eq!(decoded.report.fallback, None);
}

#[test]
fn cpu_retry_is_explicit_for_nonconforming_tree_accepted_by_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // The CPU golden is permissive about incomplete trees. GPU rejects the
    // invalid two-bit singleton rather than calling its output GPU-produced.
    let mut lit = vec![0; 257];
    lit[65] = 2;
    lit[256] = 2;
    let mut w = Bits::default();
    dynamic(&mut w, true, &lit, &[0]);
    w.code(&lit, 65);
    w.code(&lit, 256);
    let decoded = GDeflateDecoder::new(&ctx)
        .decode(&ctx, &container(&w.bytes(), b"A"))
        .unwrap();
    assert_eq!(decoded.bytes, b"A");
    assert_eq!(decoded.report.path, DecodePath::CpuFallback);
    assert_eq!(
        decoded.report.fallback,
        Some(GpuFallbackReason::DecodeRejected(
            DeflateDecodeError::MalformedStream
        ))
    );
}
