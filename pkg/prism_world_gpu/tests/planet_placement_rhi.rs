//! Strict physical-GPU coverage of planetary placement RHI submission leases.
#![cfg(feature = "placement_rhi")]

use std::{convert::Infallible, num::NonZeroUsize, sync::Arc, time::Duration};

use prism_world_gpu::{
    planet_placement_rhi::{PlanetPlacementRenderGraphPlan, PlanetPlacementRhiBridge},
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
        max_storage_buffers_per_shader_stage: 7,
        max_bindings_per_bind_group: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("planetary RHI device acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(context.evidence().is_hardware());
    eprintln!("[planet_placement_rhi] {:?}", context.evidence());
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

fn map_u32s(context: &GpuContext, buffer: &wgpu::Buffer, words: usize) -> Vec<u32> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context.wait();
    receiver
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let mapped = buffer.get_mapped_range(..).unwrap();
    let values = mapped[..words * 4]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    drop(mapped);
    buffer.unmap();
    values
}

fn read_indirect_after_bridge_submit(
    context: &GpuContext,
    bridge: &mut PlanetPlacementRhiBridge,
    source: &wgpu::Buffer,
) -> Vec<u32> {
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement RHI indirect verification"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement RHI verification copy"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, 16);
    context.queue().submit([encoder.finish()]);
    context.wait();
    let values = map_u32s(context, &staging, 4);
    assert_eq!(bridge.poll().unwrap(), 1);
    values
}

#[test]
fn placement_rhi_submit_executes_producer_and_retires_only_after_completion() {
    let context = context();
    let tile = tile();
    let region = PlanetPlacementRegion::new(tile.tile(), [0, 0], [2, 2]).unwrap();
    let mut producer = GpuPlanetPlacement::new(&context, 4, 16, 3, u64::MAX).unwrap();
    producer
        .write_request(
            config(),
            region,
            &tile,
            GpuPlanetPlacementWorkBudget {
                max_candidate_evaluations: 4,
                max_sample_reads: 16,
            },
        )
        .unwrap();

    let mut bridge = PlanetPlacementRhiBridge::new(&context, NonZeroUsize::new(1).unwrap());
    let registered = bridge.register(producer).unwrap();
    let snapshot = registered.producer().registration_snapshot().unwrap();
    assert_eq!(
        PlanetPlacementRenderGraphPlan::lower(&snapshot)
            .unwrap()
            .execution
            .pass_count(),
        4
    );
    let receipt = bridge.submit(&registered).unwrap();
    assert_eq!(bridge.pending_count(), 1);
    assert!(!receipt.is_complete());

    let indirect = read_indirect_after_bridge_submit(
        &context,
        &mut bridge,
        registered.producer().indirect_args(),
    );
    assert_eq!(indirect, [3, 4, 0, 0]);
    assert!(receipt.is_complete());
    assert_eq!(bridge.pending_count(), 0);
}

#[test]
fn planetary_deletion_rhi_submit_executes_filter_and_retires_snapshot() {
    let context = context();
    let tile = tile();
    let config = config();
    let region = PlanetPlacementRegion::new(tile.tile(), [0, 0], [2, 2]).unwrap();
    let key = PlanetPlacementKey {
        placement_version: PLANET_PLACEMENT_VERSION,
        seed: config.seed,
        version: config.version,
        salt: config.salt,
        tile: tile.tile(),
        cell: [0, 0],
    };
    let deletions = PlanetPlacementDeletionSet::new(
        vec![key],
        PlanetPlacementDeletionBudget {
            max_keys: 1,
            max_key_bytes: 48,
        },
    )
    .unwrap();
    let placement = GpuPlanetPlacement::new(&context, 4, 16, 3, u64::MAX).unwrap();
    let mut producer = GpuPlanetPlacementDeletions::new(
        &context,
        placement,
        GpuPlanetPlacementDeletionCapacity {
            max_keys: 1,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: 48,
        },
    )
    .unwrap();
    producer
        .write_request(
            config,
            region,
            &tile,
            GpuPlanetPlacementWorkBudget {
                max_candidate_evaluations: 4,
                max_sample_reads: 16,
            },
            &deletions,
            GpuPlanetPlacementDeletionWorkBudget {
                max_input_slots: 4,
                max_keys: 1,
                max_key_comparisons: 8,
            },
        )
        .unwrap();

    let mut bridge = PlanetPlacementRhiBridge::new(&context, NonZeroUsize::new(1).unwrap());
    let registered = bridge.register_deletions(producer).unwrap();
    let snapshot = registered.producer().registration_snapshot().unwrap();
    assert_eq!(
        PlanetPlacementRenderGraphPlan::lower_deletions(&snapshot)
            .unwrap()
            .execution
            .pass_count(),
        6
    );
    let receipt = bridge.submit_deletions(&registered).unwrap();
    assert_eq!(bridge.pending_count(), 1);
    assert!(!receipt.is_complete());

    let indirect = read_indirect_after_bridge_submit(
        &context,
        &mut bridge,
        registered.producer().indirect_args(),
    );
    assert_eq!(indirect, [3, 3, 0, 0]);
    assert!(receipt.is_complete());
    assert_eq!(bridge.pending_count(), 0);
}
