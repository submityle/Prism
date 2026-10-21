//! Physical-GPU parity for the independent 56-byte placement deletion chain.
#![cfg(feature = "terrain_scatter")]

use std::{
    convert::Infallible,
    sync::{mpsc, Arc},
    time::Duration,
};

use prism_world_gpu::{
    GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementDeletionCapacity,
    GpuHeightfieldPlacementDeletionError, GpuHeightfieldPlacementDeletionWorkBudget,
    GpuHeightfieldPlacementDeletions, GpuHeightfieldPlacementWorkBudget,
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
    context_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
}

fn context_with_limits(limits: wgpu::Limits) -> GpuContext {
    let context = GpuContext::try_headless_with_limits(limits)
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
            sample_origin_mm: [-3_000, 0, -3_000],
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
    HeightfieldPlacementRegion::new([-3_000, -3_000], [0, 0]).unwrap()
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
    let mut results = Vec::new();
    for native_snapshot in [false, true] {
        // Keep the cloned handles alive through submission and readback.
        let snapshot = gpu.registration_snapshot().unwrap();
        let output = staging(context, (16 + 64 * n + 4 * n) as u64);
        let mut encoder = context.device().create_command_encoder(&Default::default());
        if native_snapshot {
            let mut stages = vec![&snapshot.upstream.reset];
            if let Some(active) = &snapshot.upstream.active {
                stages.extend([
                    &active.source,
                    &active.project,
                    &active.field,
                    &active.finalize,
                ]);
            }
            for stage in stages {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(stage.pipeline());
                pass.set_bind_group(0, stage.bind_group(), &[]);
                let [x, y, z] = stage.dispatch();
                pass.dispatch_workgroups(x, y, z);
            }
            encoder.copy_buffer_to_buffer(
                snapshot.upstream.indirect.buffer(),
                0,
                snapshot.indirect.buffer(),
                0,
                16,
            );
            for stage in
                std::iter::once(&snapshot.clear_instance_count).chain(snapshot.active.as_ref())
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(stage.pipeline());
                pass.set_bind_group(0, stage.bind_group(), &[]);
                let [x, y, z] = stage.dispatch();
                pass.dispatch_workgroups(x, y, z);
            }
        } else {
            gpu.encode(&mut encoder).unwrap();
        }
        encoder.copy_buffer_to_buffer(snapshot.indirect.buffer(), 0, &output, 0, 16);
        encoder.copy_buffer_to_buffer(
            snapshot.upstream.records.buffer(),
            0,
            &output,
            16,
            64 * n as u64,
        );
        encoder.copy_buffer_to_buffer(
            snapshot.compact.buffer(),
            0,
            &output,
            16 + 64 * n as u64,
            4 * n as u64,
        );
        context.queue().submit([encoder.finish()]);
        let bytes = read(context, &output);
        let all = words(&bytes);
        assert_eq!(all[0], 3);
        assert_eq!(&all[2..4], &[0, 0]);
        results.push(keys_from_compact(&bytes, n, c));
    }
    assert_eq!(results[0].len(), results[1].len());
    assert_eq!(
        results[0]
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        results[1]
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        "native snapshot stages must match direct encoding",
    );
    results.remove(0)
}

fn deletion_set(
    keys: Vec<HeightfieldPlacementKey>,
    max_keys: usize,
) -> HeightfieldPlacementDeletionSet {
    HeightfieldPlacementDeletionSet::new(
        keys,
        HeightfieldPlacementDeletionBudget {
            max_keys,
            max_key_bytes: max_keys * std::mem::size_of::<HeightfieldPlacementKey>(),
        },
    )
    .unwrap()
}

fn gpu(
    context: &GpuContext,
    placement_capacity: usize,
    deletion_capacity: usize,
) -> GpuHeightfieldPlacementDeletions {
    GpuHeightfieldPlacementDeletions::new(
        context,
        GpuHeightfieldPlacement::new(context, placement_capacity, 0, 16, 3, u64::MAX).unwrap(),
        GpuHeightfieldPlacementDeletionCapacity {
            max_keys: deletion_capacity,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: deletion_capacity * 56,
        },
    )
    .unwrap()
}

fn deletion_budget(n: usize, d: usize) -> GpuHeightfieldPlacementDeletionWorkBudget {
    GpuHeightfieldPlacementDeletionWorkBudget {
        max_input_slots: n as u64,
        max_keys: d,
        max_key_comparisons: n as u64 * u64::from(usize::BITS - d.leading_zeros()),
    }
}

#[test]
fn physical_gpu_filters_exact_full_placement_keys_and_preserves_upstream_records() {
    // Force a padded 2D dispatch without allocating millions of candidates.
    let context = context_with_limits(wgpu::Limits {
        max_compute_workgroups_per_dimension: 2,
        max_storage_buffers_per_shader_stage: 8,
        ..wgpu::Limits::downlevel_defaults()
    });
    let c = HeightfieldPlacementConfig {
        grid_spacing_mm: 250,
        ..config()
    };
    let r = region();
    let f = field();
    let s = surface();
    let selection = filters();
    let (placement_budget, cpu_budget, n) = budgets(c, r, &f);
    assert!((129..=256).contains(&n), "exercise a padded 2x2 dispatch");
    let cpu = scatter_on_heightfield(c, r, &f, &s, selection, cpu_budget).unwrap();
    assert!(cpu.points.len() > 2);
    let deletion_key = cpu.points[0].key;
    let absent_key = placement_key(c, 99, 99);
    let one_deleted = HeightfieldPlacementDeletionSet::new(
        vec![deletion_key, deletion_key, absent_key],
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
    let mut previous_snapshot = None;
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
        let snapshot = gpu.registration_snapshot().unwrap();
        assert_eq!(snapshot.request.candidate_count, n as u32);
        assert_eq!(snapshot.request.candidate_count, gpu.candidate_count());
        assert_eq!(snapshot.request.key_count, deletions.keys().len() as u32);
        assert_eq!(snapshot.request.workgroups, [2, 2]);
        assert_eq!(
            snapshot.upstream.candidate_count,
            snapshot.request.candidate_count
        );
        assert_eq!(snapshot.clear_instance_count.dispatch(), [1, 1, 1]);
        assert_eq!(snapshot.active.as_ref().unwrap().dispatch(), [2, 2, 1]);
        assert_eq!(
            snapshot.upstream.active.as_ref().unwrap().source.dispatch(),
            snapshot.active.as_ref().unwrap().dispatch()
        );
        assert_eq!(snapshot.params.bytes(), 128);
        assert_eq!(snapshot.keys.bytes(), 56 * n as u64);
        assert_eq!(snapshot.compact.bytes(), 4 * n as u64);
        assert_eq!(snapshot.indirect.bytes(), 16);
        for buffer in [
            &snapshot.params,
            &snapshot.keys,
            &snapshot.compact,
            &snapshot.indirect,
        ] {
            assert_eq!(buffer.bytes(), buffer.buffer().size());
        }
        assert_eq!(snapshot.compact.buffer(), gpu.compact_indices());
        assert_eq!(snapshot.indirect.buffer(), gpu.indirect_args());
        assert_eq!(
            snapshot.upstream.compact.buffer(),
            gpu.upstream_compact_indices()
        );
        assert_eq!(snapshot.upstream.records.buffer(), gpu.records());
        assert_eq!(snapshot.upstream.samples.buffer(), gpu.surface_samples());
        assert_eq!(snapshot.upstream.attributes.buffer(), gpu.attributes());
        assert_ne!(
            snapshot.compact.buffer(),
            snapshot.upstream.compact.buffer()
        );
        assert_ne!(
            snapshot.indirect.buffer(),
            snapshot.upstream.indirect.buffer()
        );
        if let Some((generation, metadata)) = previous_snapshot {
            assert_eq!(snapshot.upstream.request_generation, generation + 1);
            assert_ne!(snapshot.request.key_count, metadata);
        }
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
        previous_snapshot = Some((
            snapshot.upstream.request_generation,
            snapshot.request.key_count,
        ));
    }
}

#[test]
fn admission_is_atomic_and_encode_requires_an_uploaded_request() {
    let context = context();
    let c = config();
    let r = region();
    let f = field();
    let s = surface();
    let selection = filters();
    let (placement_budget, cpu_budget, n) = budgets(c, r, &f);
    let cpu = scatter_on_heightfield(c, r, &f, &s, selection, cpu_budget).unwrap();
    assert!(cpu.points.len() > 2);

    let mut gpu = gpu(&context, n, 1);
    let mut encoder = context.device().create_command_encoder(&Default::default());
    assert_eq!(
        gpu.encode(&mut encoder),
        Err(GpuHeightfieldPlacementDeletionError::RequestNotUploaded)
    );
    assert!(matches!(
        gpu.registration_snapshot(),
        Err(GpuHeightfieldPlacementDeletionError::RequestNotUploaded)
    ));

    let valid = deletion_set(vec![cpu.points[0].key], n);
    gpu.write_request(
        c,
        r,
        &f,
        &s,
        selection,
        placement_budget,
        &valid,
        deletion_budget(n, 1),
    )
    .unwrap();
    let before = submit_read(&context, &gpu, n, c);
    let before_cache = gpu.cache_key().cloned();
    let before_stats = gpu.upload_stats();
    let before_snapshot = gpu.registration_snapshot().unwrap();
    assert_eq!(before_snapshot.request.key_count, 1);
    assert_eq!(before_snapshot.keys.bytes(), 56);
    assert_eq!(before_stats.uniform_bytes, before_snapshot.params.bytes());
    assert_eq!(
        before_stats.key_bytes,
        56 * u64::from(before_snapshot.request.key_count)
    );

    let too_many = deletion_set(vec![cpu.points[0].key, cpu.points[1].key], n);
    let cases = [
        (
            valid.clone(),
            GpuHeightfieldPlacementDeletionWorkBudget {
                max_input_slots: (n - 1) as u64,
                ..deletion_budget(n, 1)
            },
            GpuHeightfieldPlacementDeletionError::InputBudgetExceeded {
                required: n as u64,
                limit: (n - 1) as u64,
            },
        ),
        (
            valid.clone(),
            GpuHeightfieldPlacementDeletionWorkBudget {
                max_keys: 0,
                ..deletion_budget(n, 1)
            },
            GpuHeightfieldPlacementDeletionError::KeyBudgetExceeded {
                required: 1,
                limit: 0,
            },
        ),
        (
            valid.clone(),
            GpuHeightfieldPlacementDeletionWorkBudget {
                max_key_comparisons: (n - 1) as u64,
                ..deletion_budget(n, 1)
            },
            GpuHeightfieldPlacementDeletionError::ComparisonBudgetExceeded {
                required: n as u64,
                limit: (n - 1) as u64,
            },
        ),
        (
            too_many,
            deletion_budget(n, 2),
            GpuHeightfieldPlacementDeletionError::KeyCapacityExceeded {
                required: 2,
                capacity: 1,
            },
        ),
    ];

    for (deletions, budget, expected) in cases {
        assert_eq!(
            gpu.validate_request(
                c,
                r,
                &f,
                &s,
                selection,
                placement_budget,
                &deletions,
                budget,
            ),
            Err(expected.clone())
        );
        assert_eq!(
            gpu.write_request(
                c,
                r,
                &f,
                &s,
                selection,
                placement_budget,
                &deletions,
                budget,
            ),
            Err(expected)
        );
        assert_eq!(gpu.cache_key(), before_cache.as_ref());
        assert_eq!(gpu.upload_stats(), before_stats);
        let snapshot = gpu.registration_snapshot().unwrap();
        assert_eq!(snapshot.request, before_snapshot.request);
        assert_eq!(
            snapshot.upstream.request_generation,
            before_snapshot.upstream.request_generation
        );
        assert_eq!(snapshot.params.buffer(), before_snapshot.params.buffer());
        assert_eq!(snapshot.keys.buffer(), before_snapshot.keys.buffer());
        assert_eq!(
            snapshot.clear_instance_count.pipeline(),
            before_snapshot.clear_instance_count.pipeline()
        );
        assert_eq!(
            snapshot.clear_instance_count.bind_group(),
            before_snapshot.clear_instance_count.bind_group()
        );
        assert_eq!(submit_read(&context, &gpu, n, c), before);
    }
    gpu.write_request(
        c,
        r,
        &f,
        &s,
        selection,
        placement_budget,
        &valid,
        deletion_budget(n, 1),
    )
    .unwrap();
    let repeated = gpu.registration_snapshot().unwrap();
    assert_eq!(repeated.request, before_snapshot.request);
    assert_eq!(
        repeated.upstream.request_generation,
        before_snapshot.upstream.request_generation + 1
    );
    assert_eq!(gpu.upload_stats().uniform_bytes, 0);
    assert_eq!(gpu.upload_stats().key_bytes, 0);
    assert_eq!(submit_read(&context, &gpu, n, c), before);
}

#[test]
fn constructor_enforces_scratch_budget_and_zero_capacity_is_valid() {
    let context = context();
    let c = config();
    let f = field();
    let s = surface();
    let placement = GpuHeightfieldPlacement::new(&context, 1, 0, 16, 3, u64::MAX).unwrap();
    assert!(matches!(
        GpuHeightfieldPlacementDeletions::new(
            &context,
            placement,
            GpuHeightfieldPlacementDeletionCapacity {
                max_keys: 2,
                max_buffer_bytes: u64::MAX,
                max_scratch_bytes: 2 * 56 - 1,
            },
        ),
        Err(
            GpuHeightfieldPlacementDeletionError::ScratchBudgetExceeded {
                required: 112,
                limit: 111,
            },
        )
    ));

    let zero_region = HeightfieldPlacementRegion::new([0, 0], [0, 0]).unwrap();
    let (zero_placement_budget, zero_budget, zero_n) = budgets(c, zero_region, &f);
    assert_eq!(zero_n, 0);
    let mut zero = gpu(&context, 0, 0);
    let empty = deletion_set(Vec::new(), 0);
    let selection = filters();
    zero.write_request(
        c,
        zero_region,
        &f,
        &s,
        selection,
        zero_placement_budget,
        &empty,
        deletion_budget(0, 0),
    )
    .unwrap();
    assert_eq!(zero.candidate_count(), 0);
    assert_eq!(zero.deletions(), Some(&empty));
    assert_eq!(zero_budget.max_output_points, 0);
    let snapshot = zero.registration_snapshot().unwrap();
    assert_eq!(snapshot.request.candidate_count, 0);
    assert_eq!(snapshot.request.key_count, 0);
    assert_eq!(snapshot.request.workgroups, [0, 0]);
    assert_eq!(snapshot.upstream.candidate_count, 0);
    assert!(snapshot.upstream.active.is_none());
    assert!(snapshot.active.is_none());
    assert_eq!(snapshot.upstream.reset.dispatch(), [1, 1, 1]);
    assert_eq!(snapshot.clear_instance_count.dispatch(), [1, 1, 1]);
    assert_eq!(
        [
            snapshot.params.bytes(),
            snapshot.keys.bytes(),
            snapshot.compact.bytes(),
            snapshot.indirect.bytes()
        ],
        [128, 56, 4, 16]
    );
    for buffer in [
        &snapshot.params,
        &snapshot.keys,
        &snapshot.compact,
        &snapshot.indirect,
    ] {
        assert_eq!(buffer.bytes(), buffer.buffer().size());
    }
    assert_eq!(zero.deletion_buffer_bytes(), 128 + 56 + 4 + 16);
    let _ = submit_read(&context, &zero, 0, c);
}
