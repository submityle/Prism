//! Strict physical-GPU regression for the planetary placement producer.
#![cfg(feature = "terrain_scatter")]

use std::{convert::Infallible, sync::Arc, time::Duration};

use prism_world_gpu::{GpuContext, GpuPlanetPlacement, GpuPlanetPlacementWorkBudget};
use prism_world_terrain::{
    planet_placement_rank, BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldSpec, PlanetPlacementConfig, PlanetPlacementKey, PlanetPlacementRegion, TileKey,
    PLANET_PLACEMENT_VERSION,
};

struct Flat;

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(0))
    }
}

struct Ramp;

impl HeightSampler for Ramp {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(
            i64::from(sample.u_q16() / 8192) - i64::from(sample.v_q16() / 8192),
        ))
    }
}

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 6,
        max_bindings_per_bind_group: 6,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical planetary placement device acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(context.evidence().is_hardware());
    eprintln!("[planet_placement_gpu] {:?}", context.evidence());
    context
}

fn tile() -> Arc<prism_world_terrain::HeightfieldTile> {
    Arc::new(
        prism_world_terrain::HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveZ),
            HeightfieldSpec::new(4, 1).unwrap(),
            &Flat,
            BakeLimits {
                max_samples: 16,
                max_bytes: 32,
            },
        )
        .unwrap(),
    )
}

fn ramp_tile() -> Arc<prism_world_terrain::HeightfieldTile> {
    Arc::new(
        prism_world_terrain::HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveZ),
            HeightfieldSpec::new(4, 1).unwrap(),
            &Ramp,
            BakeLimits {
                max_samples: 16,
                max_bytes: 32,
            },
        )
        .unwrap(),
    )
}

fn read(context: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn f32_word(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(word(bytes, offset))
}

fn i128_words(value: i128) -> [u32; 4] {
    let value = value as u128;
    [
        value as u32,
        (value >> 32) as u32,
        (value >> 64) as u32,
        (value >> 96) as u32,
    ]
}

#[test]
fn planetary_placement_dispatch_produces_all_flat_tile_candidates() {
    let context = context();
    let tile = tile();
    let mut placement = GpuPlanetPlacement::new(&context, 4, 16, 3, u64::MAX).unwrap();
    placement
        .write_request(
            PlanetPlacementConfig {
                seed: 7,
                version: 3,
                salt: 11,
                radius_mm: 6_371_000_000,
                origin_mm: [0; 3],
                vertical_offset_mm: 0,
                acceptance_threshold: u16::MAX,
            },
            PlanetPlacementRegion::new(tile.tile(), [0, 0], [2, 2]).unwrap(),
            &tile,
            GpuPlanetPlacementWorkBudget {
                max_candidate_evaluations: 4,
                max_sample_reads: 16,
            },
        )
        .unwrap();
    let registration = placement.registration_snapshot().unwrap();
    assert_eq!(registration.request_generation, 1);
    assert_eq!(registration.candidate_count, 4);
    assert_eq!(registration.params.bytes(), 192);
    assert_eq!(registration.records.bytes(), 4 * 80);
    assert_eq!(registration.heights.bytes(), 16 * 16);
    assert_eq!(registration.compact.bytes(), 4 * 4);
    assert_eq!(registration.status.bytes(), 8);
    assert_eq!(registration.indirect.bytes(), 16);
    assert_eq!(registration.active.as_ref().unwrap().dispatch(), [1, 1, 1]);
    assert_eq!(
        registration.finalize.as_ref().unwrap().dispatch(),
        [1, 1, 1]
    );

    let readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement GPU readback"),
        size: 24,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let records = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement GPU records readback"),
        size: 4 * 80,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement GPU test encoder"),
        });
    registration.encode(&mut encoder);
    placement.encode_readback(&mut encoder, &readback).unwrap();
    placement
        .encode_records_readback(&mut encoder, &records)
        .unwrap();
    context.queue().submit(Some(encoder.finish()));
    context.wait();

    let bytes = read(&context, &readback);
    assert_eq!(
        word(&bytes, 0),
        4,
        "all four flat candidates should pass the threshold"
    );
    assert_eq!(
        word(&bytes, 4),
        0,
        "the flat tile must not report projection failures"
    );
    assert_eq!(
        word(&bytes, 12),
        4,
        "indirect instance count must match accepted candidates"
    );
    assert_eq!(word(&bytes, 16), 0);
    assert_eq!(word(&bytes, 20), 0);

    let record_bytes = read(&context, &records);
    for index in 0..4_u32 {
        let base = index as usize * 80;
        let cell = [index % 2, index / 2];
        let key = PlanetPlacementKey {
            placement_version: PLANET_PLACEMENT_VERSION,
            seed: 7,
            version: 3,
            salt: 11,
            tile: tile.tile(),
            cell,
        };
        assert_eq!(word(&record_bytes, base), 4);
        assert_eq!(word(&record_bytes, base + 4), 0);
        assert_eq!(word(&record_bytes, base + 8), 0);
        assert_eq!(word(&record_bytes, base + 12), 0);
        assert_eq!(word(&record_bytes, base + 16), cell[0]);
        assert_eq!(word(&record_bytes, base + 20), cell[1]);
        assert_eq!(word(&record_bytes, base + 48), 1);
        assert_eq!(
            word(&record_bytes, base + 52),
            u32::from(planet_placement_rank(key))
        );
        assert_eq!(
            &record_bytes[base + 56..base + 72],
            bytemuck::cast_slice::<u32, u8>(&i128_words(0))
        );
        assert_eq!(word(&record_bytes, base + 72), 0);
        assert_eq!(word(&record_bytes, base + 76), 0);
    }
}

#[test]
fn planetary_placement_records_match_cpu_semantics_for_nonzero_heights_and_threshold() {
    let context = context();
    let tile = ramp_tile();
    let config = PlanetPlacementConfig {
        seed: 0x1234_5678_9abc_def0,
        version: 19,
        salt: 0xfedc_ba98_7654_3210,
        radius_mm: 6_371_000_000,
        origin_mm: [12, -34, 56],
        vertical_offset_mm: 123,
        acceptance_threshold: 0x7fff,
    };
    let region = PlanetPlacementRegion::new(tile.tile(), [0, 0], [2, 2]).unwrap();
    let mut placement = GpuPlanetPlacement::new(&context, 4, 16, 3, u64::MAX).unwrap();
    placement
        .write_request(
            config,
            region,
            &tile,
            GpuPlanetPlacementWorkBudget {
                max_candidate_evaluations: 4,
                max_sample_reads: 16,
            },
        )
        .unwrap();
    let records = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement semantic parity readback"),
        size: 4 * 80,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement semantic parity encoder"),
        });
    placement.encode(&mut encoder).unwrap();
    placement
        .encode_records_readback(&mut encoder, &records)
        .unwrap();
    context.queue().submit(Some(encoder.finish()));
    context.wait();
    let bytes = read(&context, &records);

    for index in 0..4_u32 {
        let base = index as usize * 80;
        let cell = [index % 2, index / 2];
        let key = PlanetPlacementKey {
            placement_version: PLANET_PLACEMENT_VERSION,
            seed: config.seed,
            version: config.version,
            salt: config.salt,
            tile: tile.tile(),
            cell,
        };
        let samples = tile.samples();
        let side = tile.spec().samples_per_edge() as usize;
        let at = |u: usize, v: usize| i128::from(samples[v * side + u]);
        let numerator = at(cell[0] as usize, cell[1] as usize)
            + at(cell[0] as usize + 1, cell[1] as usize)
            + at(cell[0] as usize, cell[1] as usize + 1)
            + at(cell[0] as usize + 1, cell[1] as usize + 1);
        let expected_rank = planet_placement_rank(key);
        let expected_keep = expected_rank <= config.acceptance_threshold;

        // Independent f64 projection oracle. The shader intentionally emits
        // f32 positions, so compare with a radius-scaled millimetre error bound
        // rather than claiming bitwise parity with the CPU i64 output.
        let side_cells = f64::from(tile.spec().samples_per_edge() - 1);
        let u = -1.0 + 2.0 * (f64::from(cell[0]) + 0.5) / side_cells;
        let v = -1.0 + 2.0 * (f64::from(cell[1]) + 0.5) / side_cells;
        let length = (u * u + v * v + 1.0).sqrt();
        let normal = [u / length, v / length, 1.0 / length];
        let radial = (config.radius_mm + config.vertical_offset_mm) as f64 + numerator as f64 / 4.0;
        let expected_position = [
            config.origin_mm[0] as f64 + normal[0] * radial,
            config.origin_mm[1] as f64 + normal[1] * radial,
            config.origin_mm[2] as f64 + normal[2] * radial,
        ];
        let position_tolerance_mm = 2_048.0;
        for axis in 0..3 {
            let actual_position = f64::from(f32_word(&bytes, base + 24 + axis * 4));
            assert!(
                (actual_position - expected_position[axis]).abs() <= position_tolerance_mm,
                "cell {cell:?} position axis {axis}: GPU {actual_position}, oracle {}, tolerance {position_tolerance_mm} mm",
                expected_position[axis]
            );
            let actual_normal = f64::from(f32_word(&bytes, base + 36 + axis * 4));
            assert!(
                (actual_normal - normal[axis]).abs() <= 2.0e-6,
                "cell {cell:?} normal axis {axis}: GPU {actual_normal}, oracle {}",
                normal[axis]
            );
        }
        assert_eq!(word(&bytes, base + 16), cell[0]);
        assert_eq!(word(&bytes, base + 20), cell[1]);
        assert_eq!(word(&bytes, base + 48), u32::from(expected_keep));
        assert_eq!(word(&bytes, base + 52), u32::from(expected_rank));
        assert_eq!(
            &bytes[base + 56..base + 72],
            bytemuck::cast_slice::<u32, u8>(&i128_words(numerator))
        );
        assert_eq!(word(&bytes, base + 72), 0);
    }
}
