//! Physical-GPU parity for the independent 56-byte placement deletion chain.
#![cfg(feature = "terrain_scatter")]

use std::{
    convert::Infallible,
    sync::{mpsc, Arc},
    time::Duration,
};

use prism_world_gpu::{
    GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementDeletionCapacity,
    GpuHeightfieldPlacementDeletionWorkBudget, GpuHeightfieldPlacementDeletions,
    GpuHeightfieldPlacementWorkBudget,
};
use prism_world_residency::{
    ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample,
    SectorCoord, SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    estimate_heightfield_placement_work, scatter_on_heightfield, BakeLimits, CubeFace,
    HeightSample, HeightSampleValue, HeightSampler, HeightfieldPlacementBudget,
    HeightfieldPlacementConfig, HeightfieldPlacementDeletionBudget,
    HeightfieldPlacementDeletionSet, HeightfieldPlacementFilters, HeightfieldPlacementKey,
    HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey, HEIGHTFIELD_PLACEMENT_VERSION,
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
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical placement deletion device acquisition failed")
    .expect("physical GPU required; no silent skip");
    let evidence = context.evidence();
    assert!(matches!(
        evidence.device_type.as_str(),
        "IntegratedGpu" | "DiscreteGpu"
    ));
    eprintln!("[heightfield_placement_deletion] {evidence:?}");
    context
}

fn surface() -> PlanarHeightfieldSurface {
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(4, 1).unwrap(),
        &Flat,
        BakeLimits {
            max_samples: 16,
            max_bytes: 32,
        },
    )
    .unwrap();
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [0, 0, 0],
            sample_spacing_mm: 1_000,
        },
    )
    .unwrap()
}

fn field() -> ScatterDensityField {
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 19,
            version: 2,
            base_density: u16::MAX,
            base_species: 7,
            noise: None,
        },
        Vec::new(),
        ScatterFieldBuildBudget {
            max_brushes: 0,
            max_brush_bytes: 0,
        },
    )
    .unwrap()
}

fn config() -> HeightfieldPlacementConfig {
    HeightfieldPlacementConfig {
        seed: 0x1234_5678_9abc_def0,
        version: 3,
        salt: 0x0fed_cba9_8765_4321,
        grid_spacing_mm: 1_000,
        min_distance_mm: 0,
        vertical_offset_mm: 0,
    }
}

fn region() -> HeightfieldPlacementRegion {
    HeightfieldPlacementRegion::new([0, 0], [3_000, 3_000]).unwrap()
}

fn filters() -> HeightfieldPlacementFilters {
    HeightfieldPlacementFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}

fn budgets(
    c: HeightfieldPlacementConfig,
    r: HeightfieldPlacementRegion,
    f: &ScatterDensityField,
) -> (
    GpuHeightfieldPlacementWorkBudget,
    HeightfieldPlacementBudget,
    usize,
) {
    let work = estimate_heightfield_placement_work(c, r, f).unwrap();
    let n = work.primary_candidates as usize;
    (
        GpuHeightfieldPlacementWorkBudget {
            max_candidate_evaluations: work.max_candidate_evaluations,
            max_sample_reads: work.max_sample_reads,
            max_field_work: work.max_field_work,
        },
        HeightfieldPlacementBudget {
            max_candidate_evaluations: work.max_candidate_evaluations,
            max_sample_reads: work.max_sample_reads,
            max_field_work: work.max_field_work,
            max_output_points: n,
            max_output_bytes: n
                * (std::mem::size_of::<prism_world_terrain::HeightfieldPlacementPoint>()
                    + std::mem::size_of::<ScatterFieldSample>()),
        },
        n,
    )
}

fn staging(context: &GpuContext, size: u64) -> wgpu::Buffer {
    context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("placement deletion readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
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

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn placement_key(c: HeightfieldPlacementConfig, x: i64, z: i64) -> HeightfieldPlacementKey {
    HeightfieldPlacementKey {
        placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
        source: prism_world_residency::ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: c.seed,
            version: c.version,
            salt: c.salt,
            grid_spacing_mm: c.grid_spacing_mm,
            grid: SectorCoord::new(x, 0, z),
        },
    }
}

fn keys_from_compact(
    bytes: &[u8],
    n: usize,
    c: HeightfieldPlacementConfig,
) -> Vec<HeightfieldPlacementKey> {
    let all = words(bytes);
    let count = all[1] as usize;
    let records = &all[4..4 + n * 16];
    let compact = &all[4 + n * 16..4 + n * 16 + n];
    compact[..count]
        .iter()
        .map(|slot| {
            let base = *slot as usize * 16;
            let x = i64::from_le_bytes(
                (u64::from(records[base]) | u64::from(records[base + 1]) << 32).to_le_bytes(),
            );
            let z = i64::from_le_bytes(
                (u64::from(records[base + 4]) | u64::from(records[base + 5]) << 32).to_le_bytes(),
            );
            placement_key(c, x, z)
        })
        .collect()
}

fn submit_read(
    context: &GpuContext,
    gpu: &GpuHeightfieldPlacementDeletions,
    n: usize,
    c: HeightfieldPlacementConfig,
) -> Vec<HeightfieldPlacementKey> {
    let output = staging(context, (16 + 64 * n + 4 * n) as u64);
    let mut encoder = context.device().create_command_encoder(&Default::default());
    gpu.encode(&mut encoder).unwrap();
    encoder.copy_buffer_to_buffer(gpu.indirect_args(), 0, &output, 0, 16);
    encoder.copy_buffer_to_buffer(gpu.records(), 0, &output, 16, 64 * n as u64);
    encoder.copy_buffer_to_buffer(
        gpu.compact_indices(),
        0,
        &output,
        16 + 64 * n as u64,
        4 * n as u64,
    );
    context.queue().submit([encoder.finish()]);
    keys_from_compact(&read(context, &output), n, c)
}

#[test]
fn physical_gpu_filters_exact_full_placement_keys_and_preserves_upstream_records() {
    let context = context();
    let c = config();
    let r = region();
    let f = field();
    let s = surface();
    let selection = filters();
    let (placement_budget, cpu_budget, n) = budgets(c, r, &f);
    let cpu = scatter_on_heightfield(c, r, &f, &s, selection, cpu_budget).unwrap();
    assert!(cpu.points.len() > 2);
    let deletion_key = cpu.points[0].key;
    let one_deleted = HeightfieldPlacementDeletionSet::new(
        vec![deletion_key, deletion_key],
        HeightfieldPlacementDeletionBudget {
            max_keys: n,
            max_key_bytes: n * std::mem::size_of::<HeightfieldPlacementKey>(),
        },
    )
    .unwrap();
    let empty = HeightfieldPlacementDeletionSet::new(
        Vec::new(),
        HeightfieldPlacementDeletionBudget {
            max_keys: n,
            max_key_bytes: n * std::mem::size_of::<HeightfieldPlacementKey>(),
        },
    )
    .unwrap();
    let all = HeightfieldPlacementDeletionSet::new(
        cpu.points.iter().map(|point| point.key).collect(),
        HeightfieldPlacementDeletionBudget {
            max_keys: n,
            max_key_bytes: n * std::mem::size_of::<HeightfieldPlacementKey>(),
        },
    )
    .unwrap();
    let mut gpu = GpuHeightfieldPlacementDeletions::new(
        &context,
        GpuHeightfieldPlacement::new(&context, n, 0, 16, 3, u64::MAX).unwrap(),
        GpuHeightfieldPlacementDeletionCapacity {
            max_keys: n,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: n * 56,
        },
    )
    .unwrap();
    for (deletions, expected_removed) in [
        (&one_deleted, 1usize),
        (&empty, 0usize),
        (&all, cpu.points.len()),
    ] {
        gpu.write_request(
            c,
            r,
            &f,
            &s,
            selection,
            placement_budget,
            deletions,
            GpuHeightfieldPlacementDeletionWorkBudget {
                max_input_slots: n as u64,
                max_keys: n,
                max_key_comparisons: u64::MAX,
            },
        )
        .unwrap();
        let actual = submit_read(&context, &gpu, n, c);
        let expected: Vec<_> = cpu
            .points
            .iter()
            .map(|point| point.key)
            .filter(|key| !deletions.is_deleted(*key))
            .collect();
        assert_eq!(cpu.points.len() - actual.len(), expected_removed);
        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            actual
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
        );
        assert_eq!(gpu.cache_key().unwrap().deletions, *deletions);
    }
}
