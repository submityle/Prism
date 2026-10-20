//! Real-device fixed-Huffman `GDeflate` coverage.
//!
//! The successful assertions compare device output directly with bytes built
//! by this test. They do not call the CPU inflate implementation.

use prism_gdeflate_gpu::{
    DecodePath, FixedDecodeError, GDeflateDecoder, GpuContext, GpuFixedGDeflate, GROUP, WORD,
};

#[expect(
    clippy::print_stderr,
    reason = "device identity and skipped dispatch evidence"
)]
fn gpu_context() -> Option<GpuContext> {
    let ctx = GpuContext::try_headless();
    if let Some(ctx) = &ctx {
        eprintln!("fixed-Huffman test device: {:?}", ctx.adapter_info());
    } else {
        eprintln!("[skip] fixed-Huffman GPU tests: no adapter");
    }
    ctx
}

struct BitWriter {
    bytes: Vec<u8>,
    pending: u8,
    bits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            pending: 0,
            bits: 0,
        }
    }

    fn bits(&mut self, mut value: u32, count: u8) {
        for _ in 0..count {
            self.pending |= ((value & 1) as u8) << self.bits;
            self.bits += 1;
            value >>= 1;
            if self.bits == 8 {
                self.bytes.push(self.pending);
                self.pending = 0;
                self.bits = 0;
            }
        }
    }

    fn fixed_code(&mut self, code: u16, bits: u8) {
        let mut reversed = 0u32;
        for index in 0..bits {
            reversed |= u32::from((code >> index) & 1) << (bits - index - 1);
        }
        self.bits(reversed, bits);
    }

    fn symbol(&mut self, symbol: u16) {
        let (code, bits) = fixed_litlen_code(symbol);
        self.fixed_code(code, bits);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits != 0 {
            self.bytes.push(self.pending);
        }
        self.bytes
    }
}

fn fixed_litlen_code(symbol: u16) -> (u16, u8) {
    match symbol {
        0..=143 => (0x30 + symbol, 8),
        144..=255 => (0x190 + symbol - 144, 9),
        256..=279 => (symbol - 256, 7),
        280..=287 => (0xc0 + symbol - 280, 8),
        _ => panic!("fixed literal/length symbol out of range"),
    }
}

fn length_symbol(length: usize) -> (u16, u32, u8) {
    const BASE: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    let index = BASE
        .windows(2)
        .position(|window| length >= window[0] && length < window[1])
        .unwrap_or(BASE.len() - 1);
    (
        257 + index as u16,
        (length - BASE[index]) as u32,
        EXTRA[index],
    )
}

fn distance_symbol(distance: usize) -> (u16, u32, u8) {
    const BASE: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    let index = BASE
        .windows(2)
        .position(|window| distance >= window[0] && distance < window[1])
        .unwrap_or(BASE.len() - 1);
    (index as u16, (distance - BASE[index]) as u32, EXTRA[index])
}

fn fixed_stream(blocks: &[(&[u8], Option<(usize, usize)>)]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    for (block_index, (literals, match_tail)) in blocks.iter().enumerate() {
        writer.bits(u32::from(block_index + 1 == blocks.len()), 1);
        writer.bits(1, 2); // fixed Huffman
        for &byte in *literals {
            let (code, bits) = fixed_litlen_code(u16::from(byte));
            writer.fixed_code(code, bits);
        }
        if let Some((length, distance)) = match_tail {
            let (symbol, extra, extra_bits) = length_symbol(*length);
            let (code, bits) = fixed_litlen_code(symbol);
            writer.fixed_code(code, bits);
            writer.bits(extra, extra_bits);
            let (distance_symbol, distance_extra, distance_extra_bits) = distance_symbol(*distance);
            writer.fixed_code(distance_symbol, 5);
            writer.bits(distance_extra, distance_extra_bits);
        }
        let (code, bits) = fixed_litlen_code(256);
        writer.fixed_code(code, bits);
    }
    writer.finish()
}

fn interleave(linear: &[u8]) -> Vec<u8> {
    let padded_len = linear.len().div_ceil(GROUP) * GROUP;
    let mut padded = vec![0u8; padded_len];
    padded[..linear.len()].copy_from_slice(linear);
    let rounds = padded_len / GROUP;
    let mut output = vec![0u8; padded_len];
    for lane in 0..32usize {
        for round in 0..rounds {
            let logical = round * 32 + lane;
            let stored = lane * rounds + round;
            output[stored * WORD..stored * WORD + WORD]
                .copy_from_slice(&padded[logical * WORD..logical * WORD + WORD]);
        }
    }
    output
}

fn checksum(data: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn container(tiles: &[(&[u8], Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GDF1");
    out.extend_from_slice(&[1, 32, 16, 0]);
    out.extend_from_slice(&(tiles.len() as u32).to_le_bytes());
    let total = tiles
        .iter()
        .map(|(bytes, _)| bytes.len() as u64)
        .sum::<u64>();
    out.extend_from_slice(&total.to_le_bytes());
    for (bytes, deflated) in tiles {
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(deflated.len() as u32).to_le_bytes());
        out.extend_from_slice(&checksum(bytes).to_le_bytes());
    }
    for (_, deflated) in tiles {
        out.extend_from_slice(&interleave(deflated));
    }
    out
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test must explain why it did not dispatch"
)]
#[test]
fn gpu_fixed_huffman_decodes_literals_back_reference_and_multiple_blocks() {
    let first = b"abc";
    let first_stream = fixed_stream(&[(first, Some((9, 3)))]);
    let first_expected = b"abcabcabcabc";
    let second = b"HELLO";
    let second_stream = fixed_stream(&[(second, None), (b"!", None)]);
    let second_expected = b"HELLO!";
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!(
            "[skip] gpu_fixed_huffman_decodes_literals_back_reference_and_multiple_blocks: no GPU"
        );
        return;
    };
    let decoder = GpuFixedGDeflate::new(&ctx);
    assert_eq!(
        decoder.decompress(
            &ctx,
            &interleave(&first_stream),
            first_stream.len(),
            first_expected.len(),
        ),
        Ok(first_expected.to_vec())
    );
    assert_eq!(
        decoder.decompress(
            &ctx,
            &interleave(&second_stream),
            second_stream.len(),
            second_expected.len(),
        ),
        Ok(second_expected.to_vec())
    );
    let long_prefix: Vec<u8> = (0..385u16).map(|value| value as u8).collect();
    let long_stream = fixed_stream(&[(&long_prefix, Some((3, 385)))]);
    let mut long_expected = long_prefix.clone();
    long_expected.extend_from_slice(&long_prefix[..3]);
    assert_eq!(
        decoder.decompress(
            &ctx,
            &interleave(&long_stream),
            long_stream.len(),
            long_expected.len(),
        ),
        Ok(long_expected)
    );
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test must explain why it did not dispatch"
)]
#[test]
fn unified_decoder_uses_full_gpu_path_and_handles_multiple_tiles() {
    let first = b"abc";
    let first_expected = b"abcabcabcabc";
    let second = b"HELLO";
    let second_expected = b"HELLO!";
    let encoded = container(&[
        (first_expected, fixed_stream(&[(first, Some((9, 3)))])),
        (
            second_expected,
            fixed_stream(&[(second, None), (b"!", None)]),
        ),
    ]);
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("[skip] unified_decoder_uses_full_gpu_path_and_handles_multiple_tiles: no GPU");
        return;
    };
    let decoded = GDeflateDecoder::new(&ctx)
        .decode(&ctx, &encoded)
        .expect("fixed-Huffman tiles should decode on the GPU");
    assert_eq!(
        decoded.bytes,
        [first_expected.as_slice(), second_expected].concat()
    );
    assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
}

#[test]
fn gpu_fixed_covers_all_literals_lengths_distances_and_tile_limit() {
    let Some(ctx) = gpu_context() else { return };
    let decoder = GpuFixedGDeflate::new(&ctx);
    let mut cases = Vec::new();

    let literals: Vec<u8> = (0..=255).collect();
    cases.push((fixed_stream(&[(&literals, None)]), literals));

    // Every length 3..=258, including all extra-bit values and symbol 285.
    // Distance 1 forces overlap, including copies within the same output word.
    let blocks: Vec<_> = (3..=258)
        .map(|length| (b"A".as_slice(), Some((length, 1))))
        .collect();
    let output_len = (3..=258).map(|length| length + 1).sum();
    cases.push((fixed_stream(&blocks), vec![b'A'; output_len]));

    // Both ends of every distance code's extra-bit interval, through 32768.
    let prefix: Vec<u8> = (0..32768usize)
        .map(|i| (i.wrapping_mul(71) ^ (i >> 8)) as u8)
        .collect();
    let mut expected = prefix.clone();
    let mut distance_blocks = vec![(prefix.as_slice(), None)];
    for distance in 1..=32768usize {
        let (_, extra, bits) = distance_symbol(distance);
        if extra == 0 || extra == (1u32 << bits) - 1 {
            distance_blocks.push((b"".as_slice(), Some((3, distance))));
            for _ in 0..3 {
                expected.push(expected[expected.len() - distance]);
            }
        }
    }
    cases.push((fixed_stream(&distance_blocks), expected));

    // Exercise every distance, not only code boundaries. Four independent
    // tiles keep each output below 64 KiB while references span word, lane,
    // round and block boundaries. Build expected bytes without CPU inflate.
    for first_distance in (1..=32768usize).step_by(8192) {
        let mut blocks = vec![(prefix.as_slice(), None)];
        let mut expected = prefix.clone();
        for distance in first_distance..first_distance + 8192 {
            blocks.push((b"".as_slice(), Some((3, distance))));
            for _ in 0..3 {
                expected.push(expected[expected.len() - distance]);
            }
        }
        cases.push((fixed_stream(&blocks), expected));
    }
    cases.push((
        fixed_stream(&[(b"HELLO", None), (b"", Some((7, 5)))]),
        b"HELLOHELLOHE".to_vec(),
    ));
    cases.push((fixed_stream(&[(b"", None)]), Vec::new()));
    let tile: Vec<u8> = (0..65536usize).map(|i| i as u8).collect();
    cases.push((fixed_stream(&[(&tile, None)]), tile));

    for (stream, expected) in cases {
        assert_eq!(
            decoder.decompress(&ctx, &interleave(&stream), stream.len(), expected.len()),
            Ok(expected)
        );
    }
}

#[test]
fn gpu_fixed_rejects_malformed_unsupported_and_wrong_size_without_partial_output() {
    let Some(ctx) = gpu_context() else { return };
    let decoder = GpuFixedGDeflate::new(&ctx);
    let mut cases = Vec::new();
    for symbol in [286, 287] {
        let mut writer = BitWriter::new();
        writer.bits(3, 3); // final fixed block
        writer.symbol(symbol);
        cases.push((writer.finish(), 3, FixedDecodeError::MalformedStream));
    }
    for distance in [30, 31] {
        let mut writer = BitWriter::new();
        writer.bits(3, 3);
        writer.symbol(257); // length 3, reserved distance code
        writer.fixed_code(distance, 5);
        cases.push((writer.finish(), 3, FixedDecodeError::MalformedStream));
    }
    cases.push((
        fixed_stream(&[(b"", Some((3, 1)))]),
        3,
        FixedDecodeError::DistanceTooFar,
    ));
    for size in [0, 2, 4] {
        cases.push((
            fixed_stream(&[(b"abc", None)]),
            size,
            FixedDecodeError::OutputSizeMismatch,
        ));
    }
    cases.push((
        fixed_stream(&[(b"a", Some((258, 1)))]),
        258,
        FixedDecodeError::OutputSizeMismatch,
    ));
    let full = fixed_stream(&[(b"abc", Some((258, 3)))]);
    for end in 1..full.len() {
        cases.push((full[..end].to_vec(), 261, FixedDecodeError::MalformedStream));
    }
    // Stored, dynamic and reserved BTYPE are never accepted as fixed blocks.
    for header in [1, 5, 7] {
        cases.push((vec![header], 0, FixedDecodeError::UnsupportedBlockType));
    }
    for (stream, size, error) in cases {
        assert_eq!(
            decoder.decompress(&ctx, &interleave(&stream), stream.len(), size),
            Err(error),
            "stream {stream:02x?}, output size {size}"
        );
    }
    let stream = fixed_stream(&[(b"abc", None)]);
    let interleaved = interleave(&stream);
    for (bytes, compressed, output) in [
        (&[][..], 1, 3),
        (&interleaved[..GROUP - 1], stream.len(), 3),
        (&interleaved, 0, 3),
        (&interleaved, GROUP + 1, 3),
        (&interleaved, stream.len(), 65537),
    ] {
        assert_eq!(
            decoder.decompress(&ctx, bytes, compressed, output),
            Err(FixedDecodeError::InvalidInput)
        );
    }
}

#[test]
fn unified_decoder_rejects_checksum_corruption_and_decodes_mixed_blocks_on_gpu() {
    let Some(ctx) = gpu_context() else { return };
    let decoder = GDeflateDecoder::new(&ctx);
    let mut encoded = container(&[(b"abc", fixed_stream(&[(b"abc", None)]))]);
    encoded[28] ^= 1; // first tile checksum
    assert!(decoder.decode(&ctx, &encoded).is_err());

    // A non-final empty fixed block consumes 10 bits, then a final stored
    // block aligns to the next byte and supplies three literal bytes.
    let mut mixed = BitWriter::new();
    mixed.bits(2, 3);
    mixed.symbol(256);
    mixed.bits(1, 3);
    let mut mixed = mixed.finish();
    mixed.extend_from_slice(&[3, 0, 0xfc, 0xff, b'a', b'b', b'c']);
    let encoded = container(&[(b"abc", mixed)]);
    let decoded = decoder
        .decode(&ctx, &encoded)
        .expect("mixed blocks decode completely on the GPU");
    assert_eq!(decoded.bytes, b"abc");
    assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
    assert!(decoded.report.gpu_attempted);
}
