//! Strict physical-GPU parity for planetary placement full-key deletion.
#![cfg(feature = "terrain_scatter")]

use std::{
    convert::Infallible,
    sync::{mpsc, Arc},
    time::Duration,
};

use prism_world_gpu::{
    GpuContext, GpuPlanetPlacement, GpuPlanetPlacementDeletionCapacity,
    GpuPlanetPlacementDeletionWorkBudget, GpuPlanetPlacementDeletions,
    GpuPlanetPlacementWorkBudget,
};
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
    PlanetPlacementConfig, PlanetPlacementDeletionBudget, PlanetPlacementDeletionSet,
    PlanetPlacementKey, PlanetPlacementRegion, TileKey, PLANET_PLACEMENT_VERSION,
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
        max_bindings_per_bind_group: 7,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical planetary deletion device acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(context.evidence().is_hardware());
    eprintln!("[planet_placement_deletion] {:?}", context.evidence());
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

fn config() -> PlanetPlacementConfig {
    PlanetPlacementConfig {
        seed: 7,
        version: 3,
        salt: 11,
        radius_mm: 6_371_000_000,
        origin_mm: [0; 3],
        vertical_offset_mm: 0,
        acceptance_threshold: u16::MAX,
    }
}

fn deletion_set(key: PlanetPlacementKey) -> PlanetPlacementDeletionSet {
    PlanetPlacementDeletionSet::new(
        vec![key],
        PlanetPlacementDeletionBudget {
            max_keys: 1,
            max_key_bytes: 48,
        },
    )
    .unwrap()
}

fn read(context: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap()
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
fn deletes_one_complete_planetary_key_and_preserves_other_cells() {
    let context = context();
    let tile = tile();
    let config = config();
    let region = PlanetPlacementRegion::new(tile.tile(), [0, 0], [2, 2]).unwrap();
    let n = 4usize;
    let placement_budget = GpuPlanetPlacementWorkBudget {
        max_candidate_evaluations: n as u64,
        max_sample_reads: (n * 4) as u64,
    };
    let key = PlanetPlacementKey {
        placement_version: PLANET_PLACEMENT_VERSION,
        seed: config.seed,
        version: config.version,
        salt: config.salt,
        tile: tile.tile(),
        cell: [0, 0],
    };
    let deletions = deletion_set(key);
    let deletion_budget = GpuPlanetPlacementDeletionWorkBudget {
        max_input_slots: n as u64,
        max_keys: 1,
        max_key_comparisons: n as u64 * 2,
    };
    let placement = GpuPlanetPlacement::new(&context, n, 16, 3, u64::MAX).unwrap();
    let mut gpu = GpuPlanetPlacementDeletions::new(
        &context,
        placement,
        GpuPlanetPlacementDeletionCapacity {
            max_keys: 1,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: 48,
        },
    )
    .unwrap();
    gpu.write_request(
        config,
        region,
        &tile,
        placement_budget,
        &deletions,
        deletion_budget,
    )
    .unwrap();

    let record_bytes = n * 80;
    let compact_offset = 16usize;
    let records_offset = compact_offset + n * 4;
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement deletion readback"),
        size: (records_offset + record_bytes) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &staging, 0, 16);
    encoder.copy_buffer_to_buffer(
        gpu.compact_indices(),
        0,
        &staging,
        compact_offset as u64,
        (n * 4) as u64,
    );
    encoder.copy_buffer_to_buffer(
        gpu.records(),
        0,
        &staging,
        records_offset as u64,
        record_bytes as u64,
    );
    context.queue().submit(Some(encoder.finish()));
    context.wait();
    let bytes = read(&context, &staging);
    let word = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(word(4), 3, "one complete key must be removed");
    for index in 0..3usize {
        let source_slot = word(compact_offset + index * 4) as usize;
        let record = records_offset + source_slot * 80;
        assert_ne!([word(record + 16), word(record + 20)], [0, 0]);
    }
}
