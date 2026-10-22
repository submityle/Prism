//! Strict physical-GPU regression for the planetary placement producer.
#![cfg(feature = "terrain_scatter")]

use std::{convert::Infallible, sync::Arc, time::Duration};

use prism_world_gpu::{GpuContext, GpuPlanetPlacement, GpuPlanetPlacementWorkBudget};
use prism_world_terrain::{
    planet_placement_rank, BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldSpec, PlanetPlacementBudget, PlanetPlacementConfig, PlanetPlacementKey,
    PlanetPlacementRegion, TileKey, PLANET_PLACEMENT_VERSION,
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

fn i128_word(bytes: &[u8], offset: usize) -> i128 {
    i128::from_le_bytes(bytes[offset..offset + 16].try_into().unwrap())
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
    assert_eq!(registration.records.bytes(), 4 * 116);
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
        size: 4 * 116,
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
        let base = index as usize * 116;
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
        assert_eq!(word(&record_bytes, base + 84), 1);
        assert_eq!(
            word(&record_bytes, base + 88),
            u32::from(planet_placement_rank(key))
        );
        assert_eq!(
            &record_bytes[base + 92..base + 108],
            bytemuck::cast_slice::<u32, u8>(&i128_words(0))
        );
        assert_eq!(word(&record_bytes, base + 108), 0);
        assert_eq!(word(&record_bytes, base + 112), 0);
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
        size: 4 * 116,
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
        let base = index as usize * 116;
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

        // Independent f64 projection oracle for the integer-mm GPU output.
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
        for axis in 0..3 {
            let actual_position = i128_word(&bytes, base + 24 + axis * 16) as f64;
            assert!(
                (actual_position - expected_position[axis]).abs() <= 1.0,
                "cell {cell:?} position axis {axis}: GPU {actual_position}, oracle {}",
                expected_position[axis]
            );
            let actual_normal = f64::from(f32_word(&bytes, base + 72 + axis * 4));
            assert!(
                (actual_normal - normal[axis]).abs() <= 2.0e-6,
                "cell {cell:?} normal axis {axis}: GPU {actual_normal}, oracle {}",
                normal[axis]
            );
        }
        assert_eq!(word(&bytes, base + 16), cell[0]);
        assert_eq!(word(&bytes, base + 20), cell[1]);
        assert_eq!(word(&bytes, base + 84), u32::from(expected_keep));
        assert_eq!(word(&bytes, base + 88), u32::from(expected_rank));
        assert_eq!(
            &bytes[base + 92..base + 108],
            bytemuck::cast_slice::<u32, u8>(&i128_words(numerator))
        );
        assert_eq!(word(&bytes, base + 108), 0);
    }
}

#[test]
fn planetary_placement_frame_preserves_millimetre_local_offsets_at_planet_scale() {
    let context = context();
    let tile = ramp_tile();
    let config = PlanetPlacementConfig {
        seed: 42,
        version: 2,
        salt: 77,
        radius_mm: 6_371_000_000,
        origin_mm: [4_000_000_000_000, -7_000_000_000_000, 9_000_000_000_000],
        vertical_offset_mm: 250,
        acceptance_threshold: u16::MAX,
    };
    let region = PlanetPlacementRegion::new(tile.tile(), [0, 0], [1, 1]).unwrap();
    let cpu = prism_world_terrain::place_on_planet_heightfield(
        config,
        region,
        &tile,
        PlanetPlacementBudget {
            max_candidate_evaluations: 1,
            max_output_points: 1,
            max_output_bytes: core::mem::size_of::<prism_world_terrain::PlanetPlacementPoint>(),
        },
    )
    .unwrap();
    let frame_origin = cpu.points[0].position_mm;

    let mut placement = GpuPlanetPlacement::new(&context, 1, 16, 3, u64::MAX).unwrap();
    placement
        .write_request_in_frame(
            config,
            region,
            &tile,
            frame_origin,
            GpuPlanetPlacementWorkBudget {
                max_candidate_evaluations: 1,
                max_sample_reads: 4,
            },
        )
        .unwrap();
    let registration = placement.registration_snapshot().unwrap();
    assert_eq!(registration.output_origin_mm, frame_origin);
    let records = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement local-frame parity readback"),
        size: 116,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement local-frame parity encoder"),
        });
    registration.encode(&mut encoder);
    placement
        .encode_records_readback(&mut encoder, &records)
        .unwrap();
    context.queue().submit(Some(encoder.finish()));
    context.wait();
    let bytes = read(&context, &records);
    let cells = f64::from(tile.spec().samples_per_edge() - 1);
    let face_tiles = f64::from(tile.tile().side_len());
    let u = -1.0 + 2.0 * (f64::from(tile.tile().x()) + 0.5 / cells) / face_tiles;
    let v = -1.0 + 2.0 * (f64::from(tile.tile().y()) + 0.5 / cells) / face_tiles;
    let cube = [u, v, 1.0];
    let length = cube
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let radial_mm = config.radius_mm as f64
        + config.vertical_offset_mm as f64
        + cpu.points[0].surface_height_numerator_mm as f64 / 4.0;
    for axis in 0..3 {
        let expected_local = config.origin_mm[axis] as f64 - frame_origin[axis] as f64
            + cube[axis] / length * radial_mm;
        let actual_local = i128_word(&bytes, 24 + axis * 16) as f64;
        assert!(
            (actual_local - expected_local).abs() <= 1.0,
            "axis {axis}: GPU local {actual_local} mm, CPU local {expected_local} mm"
        );
    }
}
