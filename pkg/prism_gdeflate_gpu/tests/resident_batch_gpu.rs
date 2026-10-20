//! Independent raw streams and downstream GPU copies of retained batch output.

use prism_gdeflate_gpu::{
    buffer, DeflateBatchError, DeflateBatchLimits, DeflateDecodeError, DeflateTile, GpuContext,
    GpuDeflate, GpuFixedGDeflate, ResidentDeflateTile, GROUP, WORD,
};

#[expect(
    clippy::print_stderr,
    reason = "record real GPU execution or explicit skip"
)]
fn gpu_context() -> Option<GpuContext> {
    // try_headless already enforces PRISM_REQUIRE_GPU. Also honor the shorter
    // REQUIRE_GPU spelling used by standalone integration runners.
    let context = GpuContext::try_headless();
    assert!(
        context.is_some() || std::env::var_os("REQUIRE_GPU").is_none(),
        "REQUIRE_GPU is set but no usable GPU was found",
    );
    if let Some(context) = &context {
        eprintln!("resident batch GPU: {:?}", context.adapter_info());
    } else {
        eprintln!("[skip] resident batch GPU: no usable adapter");
    }
    context
}

#[derive(Default)]
struct Bits(Vec<bool>);

impl Bits {
    fn put(&mut self, value: u32, count: usize) {
        self.0.extend((0..count).map(|bit| value & (1 << bit) != 0));
    }

    fn code(&mut self, lengths: &[u8], symbol: usize) {
        let width = lengths[symbol];
        assert!(width > 0);
        let mut counts = [0u32; 16];
        for &bits in lengths {
            if bits != 0 {
                counts[usize::from(bits)] += 1;
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
            .extend((0..width).rev().map(|bit| code & (1 << bit) != 0));
    }

    fn bytes(self) -> Vec<u8> {
        let mut bytes = vec![0; self.0.len().div_ceil(8)];
        for (index, bit) in self.0.into_iter().enumerate() {
            bytes[index / 8] |= u8::from(bit) << (index % 8);
        }
        bytes
    }

    fn stored(&mut self, last: bool, data: &[u8]) {
        assert!(data.len() <= u16::MAX as usize);
        self.put(u32::from(last), 1);
        self.put(0, 2);
        while !self.0.len().is_multiple_of(8) {
            self.0.push(false);
        }
        self.put(data.len() as u32, 16);
        self.put(u32::from(!(data.len() as u16)), 16);
        for &byte in data {
            self.put(u32::from(byte), 8);
        }
    }

    fn dynamic_header(&mut self, last: bool) -> Vec<u8> {
        const ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let lit: Vec<u8> = (0..286).map(|i| if i < 226 { 8 } else { 9 }).collect();
        let cl: Vec<u8> = (0..19).map(|i| if i < 13 { 4 } else { 5 }).collect();
        self.put(u32::from(last), 1);
        self.put(2, 2);
        self.put(29, 5); // 286 literal/length codes
        self.put(31, 5); // 32 distance codes
        self.put(15, 4); // all 19 code-length codes
        for index in ORDER {
            self.put(u32::from(cl[index]), 3);
        }
        for &width in lit.iter().chain([5; 32].iter()) {
            self.code(&cl, usize::from(width));
        }
        lit
    }
}

fn fixed_lengths() -> Vec<u8> {
    (0..288)
        .map(|symbol| match symbol {
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        })
        .collect()
}

fn fixed(data: &[u8]) -> Vec<u8> {
    let mut bits = Bits::default();
    bits.put(1, 1);
    bits.put(1, 2);
    let lengths = fixed_lengths();
    for &byte in data {
        bits.code(&lengths, usize::from(byte));
    }
    bits.code(&lengths, 256);
    bits.bytes()
}

fn stored(data: &[u8]) -> Vec<u8> {
    let mut bits = Bits::default();
    let chunks = data.len().div_ceil(u16::MAX as usize).max(1);
    if data.is_empty() {
        bits.stored(true, &[]);
    } else {
        for (index, chunk) in data.chunks(u16::MAX as usize).enumerate() {
            bits.stored(index + 1 == chunks, chunk);
        }
    }
    bits.bytes()
}

fn dynamic(data: &[u8]) -> Vec<u8> {
    let mut bits = Bits::default();
    let lengths = bits.dynamic_header(true);
    for &byte in data {
        bits.code(&lengths, usize::from(byte));
    }
    bits.code(&lengths, 256);
    bits.bytes()
}

fn mixed() -> Vec<u8> {
    let mut bits = Bits::default();
    bits.stored(false, b"abc");
    bits.put(0, 1);
    bits.put(1, 2);
    let lengths = fixed_lengths();
    bits.code(&lengths, 257); // length 3
    bits.code(&[5; 32], 2); // distance 3: references the stored block
    bits.code(&lengths, 256);
    let lengths = bits.dynamic_header(true);
    bits.code(&lengths, 257); // length 3
    bits.code(&[5; 32], 4);
    bits.put(1, 1); // distance 6: history spans both earlier blocks
    bits.code(&lengths, usize::from(b'!'));
    bits.code(&lengths, 256);
    bits.bytes()
}

fn interleave(raw: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; raw.len().div_ceil(GROUP) * GROUP];
    let rounds = bytes.len() / GROUP;
    for (index, &byte) in raw.iter().enumerate() {
        let word = index / WORD;
        bytes[(word % 32 * rounds + word / 32) * WORD + index % WORD] = byte;
    }
    bytes
}

fn request<'a>(interleaved: &'a [u8], raw: &[u8], expected: &[u8]) -> DeflateTile<'a> {
    DeflateTile {
        interleaved,
        compressed_size: raw.len(),
        expected_output_size: expected.len(),
    }
}

fn admission() -> DeflateBatchLimits {
    DeflateBatchLimits {
        max_tiles: 32,
        max_device_bytes: 8 << 20,
    }
}

fn consume(ctx: &GpuContext, outputs: &[ResidentDeflateTile], expected: &[Vec<u8>]) {
    assert_eq!(outputs.len(), expected.len());
    let stage_size: u64 = outputs.iter().map(|tile| tile.buffer().size()).sum();
    let stage = buffer::staging(ctx.device(), "resident_consumer_readback", stage_size);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    let mut offset = 0;
    for (tile, bytes) in outputs.iter().zip(expected) {
        assert_eq!(tile.byte_len(), bytes.len() as u64);
        assert_eq!(
            tile.buffer().size(),
            bytes.len().div_ceil(WORD).max(1) as u64 * WORD as u64
        );
        assert!(tile
            .buffer()
            .usage()
            .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC));
        assert!(!tile.buffer().usage().contains(wgpu::BufferUsages::MAP_READ));
        // Exercise a separate consumer-owned GPU allocation before readback.
        let consumer =
            buffer::storage_rw_zeroed(ctx.device(), "resident_consumer", tile.buffer().size());
        encoder.copy_buffer_to_buffer(tile.buffer(), 0, &consumer, 0, tile.buffer().size());
        encoder.copy_buffer_to_buffer(&consumer, 0, &stage, offset, tile.buffer().size());
        offset += tile.buffer().size();
    }
    let submission = ctx.queue().submit([encoder.finish()]);
    let words = buffer::read_back::<u32>(ctx, &stage, submission).unwrap();
    let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let mut offset = 0;
    for (tile, expected) in outputs.iter().zip(expected) {
        assert_eq!(&bytes[offset..offset + expected.len()], expected);
        offset += tile.buffer().size() as usize;
    }
}

#[test]
fn resident_batch_mixed_blocks_empty_tails_and_retained_downstream_copy() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let expected = vec![
        b"stored".to_vec(),
        b"fixed".to_vec(),
        (0..=255).collect(),
        b"abcabcabc!".to_vec(),
        vec![],
        vec![],
        vec![],
        vec![1],
        vec![2, 3],
        vec![4, 5, 6],
        (0..65536).map(|i| (i * 37) as u8).collect(),
    ];
    // Sources and the compiled decoder are destroyed before downstream use.
    let mut outputs = {
        let gpu = GpuDeflate::new(&ctx);
        let raw = vec![
            stored(&expected[0]),
            fixed(&expected[1]),
            dynamic(&expected[2]),
            mixed(),
            stored(&[]),
            fixed(&[]),
            dynamic(&[]),
            fixed(&expected[7]),
            stored(&expected[8]),
            dynamic(&expected[9]),
            stored(&expected[10]),
        ];
        let lane: Vec<_> = raw.iter().map(|bytes| interleave(bytes)).collect();
        let requests: Vec<_> = lane
            .iter()
            .zip(&raw)
            .zip(&expected)
            .map(|((lane, raw), expected)| request(lane, raw, expected))
            .collect();
        gpu.decompress_batch_resident(&ctx, &requests, admission())
            .unwrap()
    };
    consume(&ctx, &outputs, &expected);
    let retained = outputs.remove(3);
    drop(outputs);
    consume(&ctx, &[retained], &[expected[3].clone()]);
}

#[test]
fn resident_admission_counts_all_allocations_and_rejects_invalid_late_shapes() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let raw = fixed(&[]);
    let lane = interleave(&raw);
    let tile = request(&lane, &raw, &[]);
    assert!(gpu
        .decompress_batch_resident(
            &ctx,
            &[],
            DeflateBatchLimits {
                max_tiles: 0,
                max_device_bytes: 0
            }
        )
        .unwrap()
        .is_empty());
    assert_eq!(
        gpu.decompress_batch_resident(
            &ctx,
            &[tile; 2],
            DeflateBatchLimits {
                max_tiles: 1,
                max_device_bytes: u64::MAX
            }
        )
        .unwrap_err(),
        DeflateBatchError::TooManyTiles {
            requested: 2,
            limit: 1
        },
    );
    // Per tile: padded source + minimum output + uniform + status + readback.
    let required = 2 * (GROUP as u64 + 4 + 32 + 8 + 8);
    assert_eq!(
        gpu.decompress_batch_resident(
            &ctx,
            &[tile; 2],
            DeflateBatchLimits {
                max_tiles: 2,
                max_device_bytes: required - 1
            }
        )
        .unwrap_err(),
        DeflateBatchError::AllocationBudgetExceeded {
            required,
            limit: required - 1
        },
    );
    let outputs = gpu
        .decompress_batch_resident(
            &ctx,
            &[tile; 2],
            DeflateBatchLimits {
                max_tiles: 2,
                max_device_bytes: required,
            },
        )
        .unwrap();
    consume(&ctx, &outputs, &[vec![], vec![]]);
    let scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for invalid in [
        DeflateTile {
            interleaved: &lane[..GROUP - 1],
            ..tile
        },
        DeflateTile {
            compressed_size: 0,
            ..tile
        },
        DeflateTile {
            compressed_size: lane.len() + 1,
            ..tile
        },
        DeflateTile {
            expected_output_size: 65537,
            ..tile
        },
    ] {
        assert_eq!(
            gpu.decompress_batch_resident(&ctx, &[tile, invalid], admission())
                .unwrap_err(),
            DeflateBatchError::Tile {
                index: 1,
                error: DeflateDecodeError::InvalidInput
            },
        );
    }
    assert!(futures_lite::future::block_on(scope.pop()).is_none());
    let recovered = gpu
        .decompress_batch_resident(&ctx, &[tile], admission())
        .unwrap();
    consume(&ctx, &recovered, &[vec![]]);
}

#[test]
fn resident_shader_failure_is_atomic_and_next_batch_recovers() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let raw = fixed(b"retained");
    let lane = interleave(&raw);
    let good = request(&lane, &raw, b"retained");
    let retained = gpu
        .decompress_batch_resident(&ctx, &[good], admission())
        .unwrap();
    let malformed = interleave(&[7]); // reserved BTYPE
    let truncated = interleave(&[3]); // fixed header without complete EOB
    let scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
    for (bad, error) in [
        (
            DeflateTile {
                interleaved: &malformed,
                compressed_size: 1,
                expected_output_size: 0,
            },
            DeflateDecodeError::UnsupportedBlockType,
        ),
        (
            DeflateTile {
                interleaved: &truncated,
                compressed_size: 1,
                expected_output_size: 0,
            },
            DeflateDecodeError::MalformedStream,
        ),
        (
            DeflateTile {
                expected_output_size: 7,
                ..good
            },
            DeflateDecodeError::OutputSizeMismatch,
        ),
        (
            DeflateTile {
                expected_output_size: 9,
                ..good
            },
            DeflateDecodeError::OutputSizeMismatch,
        ),
    ] {
        // A successful first tile must never escape a failed second tile.
        assert_eq!(
            gpu.decompress_batch_resident(&ctx, &[good, bad], admission())
                .unwrap_err(),
            DeflateBatchError::Tile { index: 1, error },
        );
        let recovered = gpu
            .decompress_batch_resident(&ctx, &[good], admission())
            .unwrap();
        consume(&ctx, &recovered, &[b"retained".to_vec()]);
    }
    assert!(futures_lite::future::block_on(scope.pop()).is_none());
    drop(gpu);
    consume(&ctx, &retained, &[b"retained".to_vec()]);
}

#[test]
fn resident_fixed_contract_and_wrong_device_errors_are_preserved() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let gpu = GpuFixedGDeflate::new(&ctx);
    let raw = fixed(b"fixed");
    let lane = interleave(&raw);
    let good = request(&lane, &raw, b"fixed");
    for raw in [stored(&[]), dynamic(&[])] {
        let lane = interleave(&raw);
        assert_eq!(
            gpu.decompress_batch_resident(&ctx, &[good, request(&lane, &raw, &[])], admission())
                .unwrap_err(),
            DeflateBatchError::Tile {
                index: 1,
                error: DeflateDecodeError::UnsupportedBlockType
            },
        );
    }
    let outputs = gpu
        .decompress_batch_resident(&ctx, &[good], admission())
        .unwrap();
    assert_eq!(gpu.decompress(&ctx, &lane, raw.len(), 5).unwrap(), b"fixed");
    consume(&ctx, &outputs, &[b"fixed".to_vec()]);
    let Some(other) = gpu_context() else {
        return;
    };
    // Independent contexts own different instances. Their local DeviceIds may
    // compare equal; rejection must depend on the instance as well as device.
    let scope = other
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    assert_eq!(
        gpu.decompress_batch_resident(&other, &[good], admission())
            .unwrap_err(),
        DeflateBatchError::Device(DeflateDecodeError::DeviceMismatch),
    );
    let complete = GpuDeflate::new(&ctx);
    assert_eq!(
        complete
            .decompress_batch_resident(&other, &[good], admission())
            .unwrap_err(),
        DeflateBatchError::Device(DeflateDecodeError::DeviceMismatch),
    );
    assert_eq!(
        gpu.decompress(&other, &lane, raw.len(), 5),
        Err(DeflateDecodeError::DeviceMismatch)
    );
    assert_eq!(
        complete.decompress(&other, &lane, raw.len(), 5),
        Err(DeflateDecodeError::DeviceMismatch)
    );
    assert!(futures_lite::future::block_on(scope.pop()).is_none());
    let cloned = ctx.clone();
    let accepted = gpu
        .decompress_batch_resident(&cloned, &[good], admission())
        .unwrap();
    consume(&cloned, &accepted, &[b"fixed".to_vec()]);
    let accepted = complete
        .decompress_batch_resident(&cloned, &[good], admission())
        .unwrap();
    consume(&cloned, &accepted, &[b"fixed".to_vec()]);
    // Decoder remains usable after host-side wrong-device rejection.
    let recovered = gpu
        .decompress_batch_resident(&ctx, &[good], admission())
        .unwrap();
    consume(&ctx, &recovered, &[b"fixed".to_vec()]);
}

#[test]
fn resident_device_loss_cannot_publish_even_empty_output() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let gpu = GpuDeflate::new(&ctx);
    let raw = fixed(&[]);
    let lane = interleave(&raw);
    let tile = request(&lane, &raw, &[]);
    ctx.device().destroy();
    assert!(matches!(
        gpu.decompress_batch_resident(&ctx, &[tile, tile], admission()),
        Err(DeflateBatchError::Device(
            DeflateDecodeError::DeviceResource | DeflateDecodeError::Readback(_)
        ))
    ));
}
