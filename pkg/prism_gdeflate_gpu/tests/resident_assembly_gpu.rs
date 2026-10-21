//! Byte-exact resident assembly using independent stored-DEFLATE GPU inputs.

use core::ops::Range;

use prism_gdeflate_gpu::{
    buffer, DeflateBatchLimits, DeflateTile, FixedDecodeError, GpuContext, GpuDeflate,
    GpuResidentAssembler, ResidentAssemblyError, ResidentAssemblyLimits, ResidentByteRange,
    ResidentDeflateTile, GROUP, LANES, WORD,
};

#[expect(
    clippy::print_stderr,
    reason = "record real GPU execution or an explicit skip"
)]
fn gpu_context() -> Option<GpuContext> {
    let context = GpuContext::try_headless();
    assert!(
        context.is_some()
            || (std::env::var_os("PRISM_REQUIRE_GPU").is_none()
                && std::env::var_os("REQUIRE_GPU").is_none()),
        "a GPU was required but no usable device was found",
    );
    if let Some(context) = &context {
        eprintln!("resident assembly GPU: {:?}", context.adapter_info());
    } else {
        eprintln!("[skip] resident assembly GPU: no usable adapter");
    }
    context
}

fn limits() -> ResidentAssemblyLimits {
    ResidentAssemblyLimits {
        max_slices: 128,
        max_device_bytes: 16 << 20,
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|index| (index as u8).wrapping_mul(37).wrapping_add(seed) ^ ((index >> 8) as u8))
        .collect()
}

// RFC 1951 stored blocks, written directly without a shared compressor, CPU
// inflater, or production framing helper. A 64-KiB fixture needs two blocks.
fn stored(data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 65536);
    if data.is_empty() {
        return vec![1, 0, 0, 255, 255];
    }
    let count = data.len().div_ceil(u16::MAX as usize);
    let mut raw = Vec::new();
    for (index, chunk) in data.chunks(u16::MAX as usize).enumerate() {
        raw.push(u8::from(index + 1 == count)); // BFINAL, BTYPE=00, alignment.
        let len = u16::try_from(chunk.len()).unwrap();
        raw.extend_from_slice(&len.to_le_bytes());
        raw.extend_from_slice(&(!len).to_le_bytes());
        raw.extend_from_slice(chunk);
    }
    raw
}

fn interleave(raw: &[u8]) -> Vec<u8> {
    let mut lane = vec![0; raw.len().div_ceil(GROUP) * GROUP];
    let rounds = lane.len() / GROUP;
    let lanes = LANES as usize;
    for (index, &byte) in raw.iter().enumerate() {
        let word = index / WORD;
        lane[(word % lanes * rounds + word / lanes) * WORD + index % WORD] = byte;
    }
    lane
}

fn decode(ctx: &GpuContext, expected: &[Vec<u8>]) -> Vec<ResidentDeflateTile> {
    let raw: Vec<_> = expected.iter().map(|bytes| stored(bytes)).collect();
    let lane: Vec<_> = raw.iter().map(|bytes| interleave(bytes)).collect();
    let requests: Vec<_> = lane
        .iter()
        .zip(&raw)
        .zip(expected)
        .map(|((lane, raw), expected)| DeflateTile {
            interleaved: lane,
            compressed_size: raw.len(),
            expected_output_size: expected.len(),
        })
        .collect();
    let decoder = GpuDeflate::new(ctx);
    // The decoder, all request slices, and compressed uploads die on return.
    // No decoded payload readback is performed to construct assembly inputs.
    decoder
        .decompress_batch_resident(
            ctx,
            &requests,
            DeflateBatchLimits {
                max_tiles: expected.len(),
                max_device_bytes: 16 << 20,
            },
        )
        .unwrap()
}

fn byte_range(tile: &ResidentDeflateTile, range: Range<u64>) -> ResidentByteRange<'_> {
    ResidentByteRange { tile, range }
}

// Only this test consumer reads payloads back. Consume owned assembly outputs,
// exercise a separate downstream GPU allocation, and inspect every padding byte.
fn consume(ctx: &GpuContext, outputs: Vec<(ResidentDeflateTile, Vec<u8>)>) {
    assert!(!outputs.is_empty());
    let stage_size = outputs.iter().map(|(tile, _)| tile.buffer().size()).sum();
    let stage = buffer::staging(ctx.device(), "assembly_test_readback", stage_size);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    let mut expected_ranges = Vec::new();
    let mut offset = 0;
    for (tile, expected) in outputs {
        assert_eq!(tile.byte_len(), expected.len() as u64);
        let size = expected.len().div_ceil(WORD).max(1) as u64 * WORD as u64;
        assert_eq!(tile.buffer().size(), size);
        assert!(tile
            .buffer()
            .usage()
            .contains(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC));
        assert!(!tile.buffer().usage().contains(wgpu::BufferUsages::MAP_READ));
        let (owned, len) = tile.into_buffer();
        assert_eq!(len, expected.len() as u64);
        let consumer = buffer::storage_rw_zeroed(ctx.device(), "assembly_test_consumer", size);
        encoder.copy_buffer_to_buffer(&owned, 0, &consumer, 0, size);
        encoder.copy_buffer_to_buffer(&consumer, 0, &stage, offset, size);
        expected_ranges.push((offset as usize, size as usize, expected));
        offset += size;
    }
    let submission = ctx.queue().submit([encoder.finish()]);
    let bytes: Vec<u8> = buffer::read_back::<u32>(ctx, &stage, submission)
        .unwrap()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    for (offset, size, expected) in expected_ranges {
        assert_eq!(&bytes[offset..offset + expected.len()], expected);
        assert!(bytes[offset + expected.len()..offset + size]
            .iter()
            .all(|&byte| byte == 0));
    }
}

#[test]
fn all_source_and_destination_byte_alignments_concatenate_exactly() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let data = vec![pattern(259, 11), pattern(79, 173), vec![]];
    let tiles = decode(&ctx, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let mut outputs = Vec::new();
    for source_mod in 0..4usize {
        for destination_mod in 0..4usize {
            for len in [1usize, 2, 3, 7] {
                let prefix_len = if destination_mod == 0 {
                    4
                } else {
                    destination_mod
                };
                assert_eq!(prefix_len % 4, destination_mod);
                let ranges = [
                    byte_range(&tiles[1], 2..(2 + prefix_len) as u64),
                    byte_range(&tiles[2], 0..0),
                    byte_range(&tiles[0], source_mod as u64..(source_mod + len) as u64),
                    byte_range(&tiles[0], 259..259),
                    byte_range(&tiles[1], 11..16),
                    byte_range(&tiles[0], (source_mod + 1) as u64..(source_mod + 4) as u64),
                ];
                let mut expected = data[1][2..2 + prefix_len].to_vec();
                expected.extend_from_slice(&data[0][source_mod..source_mod + len]);
                expected.extend_from_slice(&data[1][11..16]);
                expected.extend_from_slice(&data[0][source_mod + 1..source_mod + 4]);
                let output = assembler.assemble(&ctx, &ranges, limits()).unwrap();
                outputs.push((output, expected));
            }
        }
    }
    drop(assembler);
    drop(tiles);
    drop(data);
    consume(&ctx, outputs);
}

#[test]
fn four_one_byte_slices_share_one_word_without_losing_any_lane() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let data = vec![
        pattern(17, 5),
        pattern(17, 91),
        pattern(17, 143),
        pattern(17, 237),
    ];
    let tiles = decode(&ctx, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let mut outputs = Vec::new();
    for rotation in 0..4usize {
        for count in 1..=4usize {
            for insert_empty in [false, true] {
                let mut ranges = Vec::new();
                let mut expected = Vec::new();
                for lane in 0..count {
                    // For count=4 all four independent dispatches target
                    // distinct byte lanes of the SAME destination word.
                    // Rotate source alignment through all mod-4 values for
                    // each destination lane, not just aligned byte copies.
                    let source_offset = 4 + (lane + rotation) % 4;
                    ranges.push(byte_range(
                        &tiles[lane],
                        source_offset as u64..(source_offset + 1) as u64,
                    ));
                    expected.push(data[lane][source_offset]);
                    if insert_empty {
                        ranges.push(byte_range(&tiles[lane], 17..17));
                    }
                }
                let output = assembler.assemble(&ctx, &ranges, limits()).unwrap();
                assert_eq!(output.byte_len(), count as u64);
                assert_eq!(output.buffer().size(), WORD as u64);
                // Counts 1/2/3 additionally require untouched shared-word
                // tail lanes to remain zero after the atomic seam writes.
                outputs.push((output, expected));
            }
        }
    }
    drop(assembler);
    drop(tiles);
    drop(data);
    consume(&ctx, outputs);
}

#[test]
fn large_ranges_cross_dispatch_boundaries_and_assembled_outputs_are_inputs() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let (output, expected) = {
        let data = vec![pattern(65536, 17), pattern(5003, 193)];
        let tiles = decode(&ctx, &data);
        let ranges = [
            byte_range(&tiles[0], 1..65535),
            byte_range(&tiles[1], 3..5003),
            byte_range(&tiles[0], 0..65536),
            byte_range(&tiles[1], 4998..5003),
        ];
        let mut expected = data[0][1..65535].to_vec();
        expected.extend_from_slice(&data[1][3..]);
        expected.extend_from_slice(&data[0]);
        expected.extend_from_slice(&data[1][4998..]);
        assert!(expected.len() > 2 * 65536);
        (
            assembler.assemble(&ctx, &ranges, limits()).unwrap(),
            expected,
        )
    };
    // The first selection crosses a seam in the already-assembled GPU buffer.
    let chained = assembler
        .assemble(
            &ctx,
            &[byte_range(&output, 65531..65548), byte_range(&output, 2..7)],
            limits(),
        )
        .unwrap();
    let mut chained_expected = expected[65531..65548].to_vec();
    chained_expected.extend_from_slice(&expected[2..7]);
    drop(assembler);
    consume(&ctx, vec![(output, expected), (chained, chained_expected)]);
}

#[test]
fn empty_lists_empty_eof_ranges_and_odd_tails_have_zero_word_padding() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let data = vec![vec![0xff; 17], vec![]];
    let tiles = decode(&ctx, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let mut outputs = vec![(
        assembler
            .assemble(
                &ctx,
                &[],
                ResidentAssemblyLimits {
                    max_slices: 0,
                    max_device_bytes: limits().max_device_bytes,
                },
            )
            .unwrap(),
        vec![],
    )];
    for ranges in [
        vec![byte_range(&tiles[0], 17..17)],
        vec![byte_range(&tiles[1], 0..0)],
        vec![
            byte_range(&tiles[0], 0..0),
            byte_range(&tiles[1], 0..0),
            byte_range(&tiles[0], 17..17),
        ],
    ] {
        outputs.push((assembler.assemble(&ctx, &ranges, limits()).unwrap(), vec![]));
    }
    for len in 1..=9u64 {
        outputs.push((
            assembler
                .assemble(&ctx, &[byte_range(&tiles[0], 3..3 + len)], limits())
                .unwrap(),
            vec![0xff; len as usize],
        ));
    }
    drop(assembler);
    drop(tiles);
    consume(&ctx, outputs);
}

#[test]
fn invalid_ranges_report_the_late_index_without_poisoning_future_assembly() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let data = vec![pattern(17, 71)];
    let tiles = decode(&ctx, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut outputs = Vec::new();
    for (start, end) in [
        (18, 18),
        (17, 18), // GPU allocation is padded to 20, logical EOF is only 17.
        (16, 18),
        (0, 18),
        (4, 3),
        (0, u64::MAX),
        (u64::MAX, u64::MAX),
        (u64::MAX, 0),
    ] {
        let ranges = [
            byte_range(&tiles[0], 1..4),
            byte_range(&tiles[0], Range { start, end }),
        ];
        assert!(matches!(
            assembler.assemble(&ctx, &ranges, limits()),
            Err(ResidentAssemblyError::InvalidRange { index: 1 }),
        ));
        let recovered = assembler
            .assemble(&ctx, &[byte_range(&tiles[0], 1..6)], limits())
            .unwrap();
        outputs.push((recovered, data[0][1..6].to_vec()));
    }
    assert!(futures_lite::future::block_on(scope.pop()).is_none());
    drop(assembler);
    drop(tiles);
    consume(&ctx, outputs);
}

#[test]
fn slice_and_allocation_budget_rejections_recover_at_the_exact_limit() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let data = vec![pattern(17, 41)];
    let tiles = decode(&ctx, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let ranges = [
        byte_range(&tiles[0], 1..4),
        byte_range(&tiles[0], 17..17),
        byte_range(&tiles[0], 5..9),
    ];
    for limit in [0, 2] {
        assert!(matches!(
            assembler.assemble(
                &ctx,
                &ranges,
                ResidentAssemblyLimits {
                    max_slices: limit,
                    max_device_bytes: limits().max_device_bytes,
                },
            ),
            Err(ResidentAssemblyError::TooManySlices {
                requested: 3,
                limit: actual,
            }) if actual == limit,
        ));
    }
    // Zero-length ranges still count as slices for admission.
    assert!(matches!(
        assembler.assemble(
            &ctx,
            &[byte_range(&tiles[0], 17..17)],
            ResidentAssemblyLimits {
                max_slices: 0,
                max_device_bytes: limits().max_device_bytes,
            },
        ),
        Err(ResidentAssemblyError::TooManySlices {
            requested: 1,
            limit: 0,
        }),
    ));
    let mut expected = data[0][1..4].to_vec();
    expected.extend_from_slice(&data[0][5..9]);
    let mut outputs = Vec::new();
    // Discover the declared allocation total rather than coupling this test
    // to an implementation-specific uniform/descriptor ABI. Require the same
    // total on retry and successful admission at exactly that limit.
    for (ranges, expected) in [
        (ranges.as_slice(), expected),
        (&[] as &[ResidentByteRange<'_>], vec![]),
    ] {
        let required = match assembler.assemble(
            &ctx,
            ranges,
            ResidentAssemblyLimits {
                max_slices: ranges.len(),
                max_device_bytes: 0,
            },
        ) {
            Err(ResidentAssemblyError::AllocationBudgetExceeded { required, limit: 0 }) => required,
            other => panic!("expected explicit device-allocation admission failure: {other:?}"),
        };
        let padded = expected.len().div_ceil(WORD).max(1) as u64 * WORD as u64;
        assert!(required >= padded, "must charge at least the padded output");
        assert!(matches!(
            assembler.assemble(
                &ctx,
                ranges,
                ResidentAssemblyLimits {
                    max_slices: ranges.len(),
                    max_device_bytes: required - 1,
                },
            ),
            Err(ResidentAssemblyError::AllocationBudgetExceeded {
                required: retry_required,
                limit,
            }) if retry_required == required && limit == required - 1,
        ));
        outputs.push((
            assembler
                .assemble(
                    &ctx,
                    ranges,
                    ResidentAssemblyLimits {
                        max_slices: ranges.len(),
                        max_device_bytes: required,
                    },
                )
                .unwrap(),
            expected,
        ));
    }
    assert!(futures_lite::future::block_on(scope.pop()).is_none());
    drop(assembler);
    drop(tiles);
    consume(&ctx, outputs);
}

#[test]
fn foreign_contexts_and_foreign_tiles_fail_closed_but_context_clones_work() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let Some(other) = gpu_context() else {
        return;
    };
    let data = vec![pattern(17, 101), vec![]];
    let local = decode(&ctx, &data);
    let foreign = decode(&other, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let other_assembler = GpuResidentAssembler::new(&other).unwrap();
    let local_scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let foreign_scope = other
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    // Each headless probe creates an independent instance. Device IDs alone
    // are insufficient, even when both probes choose the same physical GPU.
    for ranges in [
        vec![],
        vec![byte_range(&local[0], 1..6)],
        vec![byte_range(&foreign[0], 1..6)],
    ] {
        assert!(matches!(
            assembler.assemble(&other, &ranges, limits()),
            Err(ResidentAssemblyError::Device(
                FixedDecodeError::DeviceMismatch
            )),
        ));
    }
    for ranges in [
        vec![byte_range(&foreign[0], 1..6)],
        vec![byte_range(&foreign[0], 17..17)],
        vec![byte_range(&local[0], 1..4), byte_range(&foreign[1], 0..0)],
    ] {
        assert!(matches!(
            assembler.assemble(&ctx, &ranges, limits()),
            Err(ResidentAssemblyError::Device(
                FixedDecodeError::DeviceMismatch
            )),
        ));
    }
    assert!(matches!(
        other_assembler.assemble(&other, &[byte_range(&local[0], 1..6)], limits()),
        Err(ResidentAssemblyError::Device(
            FixedDecodeError::DeviceMismatch
        )),
    ));
    let cloned = ctx.clone();
    let accepted_clone = assembler
        .assemble(&cloned, &[byte_range(&local[0], 1..6)], limits())
        .unwrap();
    let recovered = assembler
        .assemble(&ctx, &[byte_range(&local[0], 2..9)], limits())
        .unwrap();
    let recovered_other = other_assembler
        .assemble(&other, &[byte_range(&foreign[0], 3..10)], limits())
        .unwrap();
    assert!(futures_lite::future::block_on(foreign_scope.pop()).is_none());
    assert!(futures_lite::future::block_on(local_scope.pop()).is_none());
    drop(assembler);
    drop(other_assembler);
    drop(local);
    drop(foreign);
    consume(
        &cloned,
        vec![
            (accepted_clone, data[0][1..6].to_vec()),
            (recovered, data[0][2..9].to_vec()),
        ],
    );
    consume(&other, vec![(recovered_other, data[0][3..10].to_vec())]);
}

#[test]
fn different_devices_in_the_same_instance_are_not_interchangeable() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let other = GpuContext::request_shared(
        ctx.instance(),
        &wgpu::RequestAdapterOptions::default(),
        &wgpu::DeviceDescriptor::default(),
    )
    .expect("second logical device in the same instance");
    assert_eq!(ctx.instance(), other.instance());
    assert_ne!(ctx.device(), other.device());
    let data = vec![pattern(17, 83)];
    let local = decode(&ctx, &data);
    let foreign = decode(&other, &data);
    let assembler = GpuResidentAssembler::new(&ctx).unwrap();
    let local_scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let other_scope = other
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    assert!(matches!(
        assembler.assemble(&other, &[byte_range(&foreign[0], 1..6)], limits()),
        Err(ResidentAssemblyError::Device(
            FixedDecodeError::DeviceMismatch
        )),
    ));
    assert!(matches!(
        assembler.assemble(&ctx, &[byte_range(&foreign[0], 1..6)], limits()),
        Err(ResidentAssemblyError::Device(
            FixedDecodeError::DeviceMismatch
        )),
    ));
    let recovered = assembler
        .assemble(&ctx, &[byte_range(&local[0], 1..6)], limits())
        .unwrap();
    assert!(futures_lite::future::block_on(other_scope.pop()).is_none());
    assert!(futures_lite::future::block_on(local_scope.pop()).is_none());
    drop(assembler);
    drop(local);
    drop(foreign);
    consume(&ctx, vec![(recovered, data[0][1..6].to_vec())]);
}

#[test]
fn owned_output_survives_sources_decoder_assembler_and_original_context() {
    let Some(ctx) = gpu_context() else {
        return;
    };
    let consumer_ctx = ctx.clone();
    let (output, expected) = {
        let data = vec![pattern(67, 29), pattern(23, 211)];
        let sources = decode(&ctx, &data);
        let assembler = GpuResidentAssembler::new(&ctx).unwrap();
        let ranges = [
            byte_range(&sources[1], 3..16),
            byte_range(&sources[0], 1..66),
            byte_range(&sources[1], 0..5),
        ];
        let mut expected = data[1][3..16].to_vec();
        expected.extend_from_slice(&data[0][1..66]);
        expected.extend_from_slice(&data[1][..5]);
        (
            assembler.assemble(&ctx, &ranges, limits()).unwrap(),
            expected,
        )
    };
    drop(ctx);
    consume(&consumer_ctx, vec![(output, expected)]);
}
