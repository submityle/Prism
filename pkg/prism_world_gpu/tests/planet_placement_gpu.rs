//! Strict physical-GPU regression for the planetary placement producer.
#![cfg(feature = "terrain_scatter")]

use std::{convert::Infallible, sync::Arc, time::Duration};

use prism_world_gpu::{GpuContext, GpuPlanetPlacement, GpuPlanetPlacementWorkBudget};
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
    PlanetPlacementConfig, PlanetPlacementRegion, TileKey,
};

struct Flat;

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(0))
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

    let readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement GPU readback"),
        size: 24,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement GPU test encoder"),
        });
    placement.encode(&mut encoder).unwrap();
    placement.encode_readback(&mut encoder, &readback).unwrap();
    context.queue().submit(Some(encoder.finish()));
    context.wait();

    let bytes = read(&context, &readback);
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(
        word(0),
        4,
        "all four flat candidates should pass the threshold"
    );
    assert_eq!(
        word(4),
        0,
        "the flat tile must not report projection failures"
    );
    assert_eq!(
        word(12),
        4,
        "indirect instance count must match accepted candidates"
    );
    assert_eq!(word(16), 0);
    assert_eq!(word(20), 0);
}
