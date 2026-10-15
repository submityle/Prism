//! Public indexed-brush parity on a physical GPU. Readback is test-only;
//! compact ordering, unused tails, and GPU timing are not part of the contract.

#![cfg(feature = "gpu")]

#[path = "support/scatter_deletion.rs"]
#[allow(dead_code)]
mod support;

use std::collections::{BTreeMap, BTreeSet};

use prism_world_gpu::{
    GpuContext, GpuScatterDeletionCapacity, GpuScatterDeletionWorkBudget, GpuScatterDeletions,
    GpuScatterError, GpuScatterField, GpuScatterFieldError, GpuScatterFieldRequest,
    GpuScatterSourceRequest,
};
use prism_world_residency::{
    estimate_scatter_work, DensityBrush, DensityBrushOp, DensityNoise, FieldScatterCacheKey,
    ScatterCacheKey, ScatterCandidateKey, ScatterConfig, ScatterDensityField, ScatterError,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldError, ScatterFieldSample,
    ScatterPoint, ScatterRegion, SectorCoord, SCATTER_FIELD_VERSION, SCATTER_KERNEL_VERSION,
};
use support::{deletions, read, region, Capture};

const VERTICES: u32 = 9;

fn context(storage_buffers: u32) -> GpuContext {
    let ctx = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: storage_buffers,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("indexed scatter must acquire a real device")
    .expect("a physical GPU is required; skipping is forbidden");
    let evidence = ctx.evidence();
    let actual = match evidence.backend.as_str() {
        "Metal" => wgpu::Backends::METAL,
        "Vulkan" => wgpu::Backends::VULKAN,
        "Dx12" => wgpu::Backends::DX12,
        other => panic!("unsupported indexed scatter backend {other}"),
    };
    if let Some(requested) = wgpu::Backends::from_env() {
        assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
    }
    assert!(
        matches!(
            evidence.device_type.as_str(),
            "IntegratedGpu" | "DiscreteGpu"
        ),
        "physical GPU required: {evidence:?}"
    );
    assert_eq!(
        ctx.device().limits().max_storage_buffers_per_shader_stage,
        storage_buffers
    );
    eprintln!("[scatter_brush_index] adapter={evidence:?}");
    ctx
}

fn config() -> ScatterConfig {
    ScatterConfig {
        grid_spacing_mm: 1,
        ..support::config()
    }
}

fn field_config() -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 0xfedc_ba98_7654_3210,
        version: 7,
        base_density: 32_768,
        base_species: 91,
        noise: None,
    }
}

fn field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(config, brushes, budget).unwrap()
}

fn brush(id: u64, center_mm: [i64; 3]) -> DensityBrush {
    DensityBrush {
        id,
        center_mm,
        inner_radius_mm: 5,
        outer_radius_mm: 19,
        opacity: u16::MAX,
        density: DensityBrushOp::Replace(u16::MAX),
        species: Some(id as u32),
    }
}

fn nodes(brushes: usize) -> usize {
    if brushes == 0 {
        0
    } else {
        2 * brushes - 1
    }
}

fn bytes(capacity: usize, brushes: usize, indexed: bool) -> u64 {
    let records = brushes + if indexed { nodes(brushes) } else { 0 };
    88 * capacity.max(1) as u64 + 352 + 64 * records.max(1) as u64
}

fn sample_work(f: &ScatterDensityField, indexed: bool) -> u64 {
    f.work_per_sample()
        + if indexed {
            nodes(f.brushes().len()) as u64
        } else {
            0
        }
}

fn slots(cfg: ScatterConfig, bounds: ScatterRegion) -> usize {
    let count = estimate_scatter_work(cfg, bounds)
        .unwrap()
        .primary_candidates;
    assert!(count <= 2048, "indexed fixtures must remain bounded");
    count as usize
}

fn upload(
    gpu: &mut GpuScatterField,
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
) {
    let work = estimate_scatter_work(cfg, bounds).unwrap();
    let per_sample = sample_work(f, gpu.brush_index_enabled());
    assert_eq!(gpu.max_sample_work(f), Ok(per_sample));
    gpu.write_request(
        cfg,
        bounds,
        work.max_candidate_evaluations,
        f,
        work.primary_candidates * per_sample,
    )
    .unwrap();
    assert_eq!(gpu.candidate_count() as usize, slots(cfg, bounds));
    assert_eq!(
        gpu.cache_key(),
        Some(&FieldScatterCacheKey {
            scatter: ScatterCacheKey {
                kernel_version: SCATTER_KERNEL_VERSION,
                config: cfg,
                region: bounds
            },
            field_version: SCATTER_FIELD_VERSION,
            field: f.clone(),
        })
    );
}

fn capture(ctx: &GpuContext, gpu: &GpuScatterField, encode: bool) -> Capture {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    if encode {
        gpu.encode(&mut encoder).unwrap();
    }
    let result = Capture::new(
        ctx,
        &mut encoder,
        gpu.candidate_count(),
        gpu.records(),
        Some(gpu.attributes()),
        gpu.compact_indices(),
        gpu.indirect_args(),
    );
    ctx.queue().submit([encoder.finish()]);
    result
}

type Accepted = BTreeMap<ScatterCandidateKey, (ScatterPoint, ScatterFieldSample)>;

#[derive(Debug, Eq, PartialEq)]
struct Snapshot {
    args: [u32; 4],
    records: Vec<[u32; 16]>,
    attributes: Vec<[u32; 4]>,
    compact: BTreeSet<u32>,
    accepted: Accepted,
}

// Check *every* primary slot, including field rejects and zeroed source rejects.
// The linear GPU additionally supplies a byte-exact source-record ABI oracle.
fn verify(
    ctx: &GpuContext,
    output: &Capture,
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
) -> Snapshot {
    let data = read(ctx, &output.buffer);
    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = slots(cfg, bounds);
    assert_eq!(data.len(), 16 + 84 * n);
    let source: BTreeMap<_, _> = support::source(cfg, bounds)
        .into_iter()
        .map(|p| (p.key, p))
        .collect();
    let mut snapshot = Snapshot {
        args: words[..4].try_into().unwrap(),
        records: Vec::new(),
        attributes: Vec::new(),
        compact: BTreeSet::new(),
        accepted: BTreeMap::new(),
    };
    let mut expected_slots = BTreeSet::new();
    if n != 0 {
        let spacing = i64::from(cfg.grid_spacing_mm);
        let min = bounds.min_mm.map(|v| v.div_euclid(spacing));
        let max = bounds.max_mm.map(|v| (v - 1).div_euclid(spacing));
        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    let slot = snapshot.records.len();
                    let record: [u32; 16] = words[4 + 16 * slot..4 + 16 * (slot + 1)]
                        .try_into()
                        .unwrap();
                    let attrs: [u32; 4] = words[4 + 16 * n + 4 * slot..4 + 16 * n + 4 * (slot + 1)]
                        .try_into()
                        .unwrap();
                    let key = ScatterCandidateKey {
                        kernel_version: SCATTER_KERNEL_VERSION,
                        seed: cfg.seed,
                        version: cfg.version,
                        salt: cfg.salt,
                        grid_spacing_mm: cfg.grid_spacing_mm,
                        grid: SectorCoord::new(x, y, z),
                    };
                    let pair = |i| (u64::from(record[i]) | (u64::from(record[i + 1]) << 32)) as i64;
                    assert_eq!(
                        [pair(0), pair(2), pair(4)],
                        [x, y, z],
                        "global source key slot {slot}"
                    );
                    assert_eq!(record[12], u32::from(source.contains_key(&key)));
                    assert_eq!(record[15], 0, "source padding");
                    match key.rebuild() {
                        Ok(p) => assert_eq!([pair(6), pair(8), pair(10)], p.position_mm),
                        Err(ScatterError::CoordinateOverflow) => {
                            assert_eq!(&record[6..12], &[0; 6])
                        }
                        Err(error) => panic!("unexpected source rebuild: {error:?}"),
                    }
                    if let Some(&p) = source.get(&key) {
                        let sample = f.sample(p.position_mm);
                        let accepted = f.accepts(key, sample.density);
                        assert_eq!(
                            attrs,
                            [
                                u32::from(sample.density),
                                sample.species,
                                u32::from(accepted),
                                f.candidate_rank(key)
                            ],
                            "CPU field slot {slot}: {key:?}"
                        );
                        if accepted {
                            snapshot.accepted.insert(key, (p, sample));
                            expected_slots.insert(slot as u32);
                        }
                    } else {
                        assert_eq!(attrs, [0; 4], "source-rejected attributes must clear");
                    }
                    snapshot.records.push(record);
                    snapshot.attributes.push(attrs);
                }
            }
        }
    }
    assert_eq!(snapshot.records.len(), n);
    assert_eq!(
        snapshot.args,
        [VERTICES, snapshot.accepted.len() as u32, 0, 0]
    );
    for &slot in &words[4 + 20 * n..4 + 20 * n + snapshot.accepted.len()] {
        assert!((slot as usize) < n);
        assert!(
            snapshot.compact.insert(slot),
            "duplicate compact source slot"
        );
    }
    assert_eq!(snapshot.compact, expected_slots);
    snapshot
}

struct Pair {
    indexed: GpuScatterField,
    linear: GpuScatterField,
}

impl Pair {
    fn new(ctx: &GpuContext, capacity: usize, brushes: usize) -> Self {
        let indexed = GpuScatterField::new_indexed(
            ctx,
            capacity,
            brushes,
            VERTICES,
            bytes(capacity, brushes, true),
        )
        .unwrap();
        let linear = GpuScatterField::new(
            ctx,
            capacity,
            brushes,
            VERTICES,
            bytes(capacity, brushes, false),
        )
        .unwrap();
        assert!(indexed.brush_index_enabled());
        assert!(!linear.brush_index_enabled());
        Self { indexed, linear }
    }

    fn submit(
        &mut self,
        ctx: &GpuContext,
        cfg: ScatterConfig,
        bounds: ScatterRegion,
        f: &ScatterDensityField,
    ) -> [Capture; 2] {
        upload(&mut self.indexed, cfg, bounds, f);
        let indexed = capture(ctx, &self.indexed, true);
        upload(&mut self.linear, cfg, bounds, f);
        let linear = capture(ctx, &self.linear, true);
        assert_eq!(
            self.indexed.cache_key(),
            self.linear.cache_key(),
            "index is not semantic cache identity"
        );
        [indexed, linear]
    }

    fn run(
        &mut self,
        ctx: &GpuContext,
        cfg: ScatterConfig,
        bounds: ScatterRegion,
        f: &ScatterDensityField,
    ) -> Snapshot {
        let outputs = self.submit(ctx, cfg, bounds, f);
        verify_pair(ctx, &outputs, cfg, bounds, f)
    }
}

fn verify_pair(
    ctx: &GpuContext,
    outputs: &[Capture; 2],
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    f: &ScatterDensityField,
) -> Snapshot {
    let indexed = verify(ctx, &outputs[0], cfg, bounds, f);
    assert_eq!(
        indexed,
        verify(ctx, &outputs[1], cfg, bounds, f),
        "indexed/linear complete ABI parity"
    );
    indexed
}

#[test]
fn indexed_constructor_exact_budget_one_below_and_zero_sentinels() {
    let ctx = context(5);
    for (capacity, brushes) in [(0, 0), (0, 3), (1, 1), (65, 2), (129, 17)] {
        let expected = bytes(capacity, brushes, true);
        assert!(
            matches!(GpuScatterField::new_indexed(&ctx, capacity, brushes, VERTICES, expected - 1),
            Err(GpuScatterFieldError::Scatter(GpuScatterError::BufferBudgetExceeded { required, limit }))
            if required == expected && limit == expected - 1)
        );
        let gpu =
            GpuScatterField::new_indexed(&ctx, capacity, brushes, VERTICES, expected).unwrap();
        assert!(gpu.brush_index_enabled());
        assert_eq!(gpu.allocated_buffer_bytes(), expected);
        assert_eq!(gpu.capacity(), capacity);
        assert!(gpu.brush_scratch_bytes() >= 64 * (brushes + nodes(brushes)));
        assert_eq!(gpu.candidate_count(), 0);
        assert!(gpu.cache_key().is_none());
    }
    assert!(matches!(
        GpuScatterField::new_indexed(&ctx, usize::MAX, 0, VERTICES, u64::MAX),
        Err(GpuScatterFieldError::Scatter(
            GpuScatterError::ArithmeticOverflow
        ))
    ));
    assert!(matches!(
        GpuScatterField::new_indexed(&ctx, 1, usize::MAX, VERTICES, u64::MAX),
        Err(GpuScatterFieldError::Scatter(
            GpuScatterError::ArithmeticOverflow
        ))
    ));
    let mut pair = Pair::new(&ctx, 0, 0);
    let empty = field(field_config(), vec![]);
    for axis in 0..3 {
        let mut max = [i64::MAX; 3];
        max[axis] = i64::MIN;
        assert!(pair
            .run(&ctx, config(), region([i64::MIN; 3], max), &empty)
            .accepted
            .is_empty());
    }
}

#[test]
fn public_work_bound_uses_actual_brush_count_and_validates_capacity_first() {
    let ctx = context(5);
    let pair = Pair::new(&ctx, 1, 7);
    for count in [0, 1, 2, 5, 7] {
        for (amplitude, warp_amplitude_mm) in [(0, u32::MAX), (12345, 0), (12345, u32::MAX)] {
            let f = field(
                ScatterFieldConfig {
                    noise: Some(DensityNoise {
                        cell_size_mm: 17,
                        amplitude,
                        warp_cell_size_mm: 13,
                        warp_amplitude_mm,
                    }),
                    ..field_config()
                },
                (0..count).map(|i| brush(i as u64, [0; 3])).collect(),
            );
            assert_eq!(
                pair.indexed.max_sample_work(&f),
                Ok(f.work_per_sample() + nodes(count) as u64)
            );
            assert_eq!(pair.linear.max_sample_work(&f), Ok(f.work_per_sample()));
        }
    }
    let too_many = field(field_config(), (0..8).map(|i| brush(i, [0; 3])).collect());
    for gpu in [&pair.indexed, &pair.linear] {
        assert_eq!(
            gpu.max_sample_work(&too_many),
            Err(GpuScatterFieldError::BrushCapacityExceeded {
                required: 8,
                capacity: 7
            })
        );
        assert!(gpu.cache_key().is_none());
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        assert_eq!(
            gpu.encode(&mut encoder),
            Err(GpuScatterFieldError::Scatter(
                GpuScatterError::RequestNotUploaded
            ))
        );
    }
}

#[test]
fn empty_shrink_grow_and_source_only_edits_reuse_fixed_allocations() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 129, 9);
    let handles = [
        pair.indexed.records().clone(),
        pair.indexed.attributes().clone(),
        pair.indexed.compact_indices().clone(),
        pair.indexed.indirect_args().clone(),
    ];
    let scratch = pair.indexed.brush_scratch_bytes();
    let allocated = pair.indexed.allocated_buffer_bytes();
    let mut pending = Vec::new();
    for (revision, count) in [9, 1, 0, 5, 2, 9, 0, 1].into_iter().enumerate() {
        let f = field(
            ScatterFieldConfig {
                version: revision as u32,
                ..field_config()
            },
            (0..count)
                .map(|i| DensityBrush {
                    density: DensityBrushOp::Add(if i % 2 == 0 { 12000 } else { -9000 }),
                    ..brush(i as u64, [i as i64 * 13 - 52, 0, 0])
                })
                .collect(),
        );
        for bounds in [
            region([-64, 0, 0], [65, 1, 1]),
            region([0; 3], [0, 1, 1]),
            region([-3, 0, 0], [4, 1, 1]),
        ] {
            let cfg = ScatterConfig {
                seed: config().seed ^ revision as u64,
                ..config()
            };
            let output = pair.submit(&ctx, cfg, bounds, &f);
            pending.push((cfg, bounds, f.clone(), output));
            assert_eq!(pair.indexed.brush_scratch_bytes(), scratch);
            assert_eq!(pair.indexed.allocated_buffer_bytes(), allocated);
            assert_eq!(pair.indexed.records(), &handles[0]);
            assert_eq!(pair.indexed.attributes(), &handles[1]);
            assert_eq!(pair.indexed.compact_indices(), &handles[2]);
            assert_eq!(pair.indexed.indirect_args(), &handles[3]);
        }
    }
    // All submissions precede any wait/readback; stale roots must not survive
    // an actual-count change, even though allocated brush capacity stays nine.
    for (cfg, bounds, f, output) in pending {
        verify_pair(&ctx, &output, cfg, bounds, &f);
    }
}

#[test]
fn global_i64_centers_and_opposite_world_corners_do_not_wrap_bounds() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 64, 3);
    let mut origins = vec![[-(1_i64 << 32) - 2, -2, -1], [(1_i64 << 60) + 29, -2, -1]];
    for mask in 0..8 {
        origins.push(std::array::from_fn(|axis| {
            if mask & (1 << axis) == 0 {
                i64::MIN
            } else {
                i64::MAX - 4
            }
        }));
    }
    for min in origins {
        let max = min.map(|v| v + 4);
        let f = field(
            field_config(),
            vec![
                DensityBrush {
                    density: DensityBrushOp::Replace(12345),
                    ..brush(u64::MAX, min)
                },
                DensityBrush {
                    center_mm: min.map(|v| if v < 0 { i64::MAX } else { i64::MIN }),
                    inner_radius_mm: u32::MAX,
                    outer_radius_mm: u32::MAX,
                    density: DensityBrushOp::Replace(0),
                    ..brush(0, [0; 3])
                },
                DensityBrush {
                    density: DensityBrushOp::Add(7000),
                    ..brush(1_u64 << 32, max)
                },
            ],
        );
        pair.run(&ctx, config(), region(min, max), &f);
    }
}

#[test]
fn max_radius_aabbs_preserve_hard_outer_boundary_and_squared_falloff() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 3, 2);
    let radius = i64::from(u32::MAX);
    for center in [[0; 3], [i64::MIN; 3], [i64::MAX - 1; 3]] {
        for inner in [0, u32::MAX] {
            let f = field(
                ScatterFieldConfig {
                    base_density: 0,
                    ..field_config()
                },
                vec![
                    DensityBrush {
                        inner_radius_mm: inner,
                        outer_radius_mm: u32::MAX,
                        species: Some(u32::MAX),
                        ..brush(9, center)
                    },
                    DensityBrush {
                        opacity: 0,
                        ..brush(0, center)
                    },
                ],
            );
            pair.run(&ctx, config(), region(center, center.map(|v| v + 1)), &f);
            for axis in 0..3 {
                let sign = if center[axis] > 0 { -1 } else { 1 };
                let mut min = center;
                min[axis] += sign * radius - 1;
                let mut max = min.map(|v| v + 1);
                max[axis] += 2;
                pair.run(&ctx, config(), region(min, max), &f);
                let mut on = center;
                on[axis] += sign * radius;
                assert_eq!(f.sample(on).density, 0, "outer boundary has zero weight");
            }
        }
    }
}

#[test]
fn noise_and_warp_match_linear_with_active_and_remote_brushes() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 64, 3);
    for (cell, amplitude, warp_cell, warp) in [
        (17, 0, 13, u32::MAX),
        (17, 30000, 13, 0),
        (17, 30000, 13, 23),
        (1, u16::MAX, 1, u32::MAX),
        (u32::MAX, u16::MAX, u32::MAX, u32::MAX),
    ] {
        for origin in [-18, -(1_i64 << 32) - 1, (1_i64 << 60) + 29] {
            let min = [origin, -2, -1];
            let f = field(
                ScatterFieldConfig {
                    seed: u64::MAX,
                    base_species: u32::MAX,
                    noise: Some(DensityNoise {
                        cell_size_mm: cell,
                        amplitude,
                        warp_cell_size_mm: warp_cell,
                        warp_amplitude_mm: warp,
                    }),
                    ..field_config()
                },
                vec![
                    DensityBrush {
                        opacity: 32768,
                        density: DensityBrushOp::Add(-13000),
                        ..brush(99, min)
                    },
                    DensityBrush {
                        density: DensityBrushOp::Add(7000),
                        ..brush(1, [origin + 3, 0, 0])
                    },
                    brush(7, [i64::MAX; 3]),
                ],
            );
            pair.run(&ctx, config(), region(min, [origin + 4, 2, 3]), &f);
        }
    }
    // Warp can leave the representable world even though brush centers cannot.
    for min in [[i64::MIN; 3], [i64::MAX - 4; 3]] {
        let f = field(
            ScatterFieldConfig {
                noise: Some(DensityNoise {
                    cell_size_mm: 1,
                    amplitude: u16::MAX,
                    warp_cell_size_mm: 1,
                    warp_amplitude_mm: u32::MAX,
                }),
                ..field_config()
            },
            vec![DensityBrush {
                opacity: 32767,
                ..brush(1, min)
            }],
        );
        pair.run(&ctx, config(), region(min, min.map(|v| v + 4)), &f);
    }
}

#[test]
fn spatially_shuffled_ids_keep_noncommuting_stable_composition_order() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 129, 5);
    let ids = [0, 1_u64 << 32, 1_u64 << 63, u64::MAX - 1, u64::MAX];
    let centers = [120, -120, 60, -60, 0];
    let ops = [
        DensityBrushOp::Add(-65535),
        DensityBrushOp::Add(50000),
        DensityBrushOp::Replace(20000),
        DensityBrushOp::Add(-10000),
        DensityBrushOp::Add(0),
    ];
    let strokes: Vec<_> = (0..5)
        .map(|i| DensityBrush {
            inner_radius_mm: 1000,
            outer_radius_mm: 1000,
            opacity: if i == 2 { 32768 } else { u16::MAX },
            density: ops[i],
            species: Some(if i == 4 { u32::MAX } else { i as u32 }),
            ..brush(ids[i], [centers[i], 0, 0])
        })
        .collect();
    let f = field(field_config(), strokes.iter().rev().copied().collect());
    assert_eq!(f, field(field_config(), strokes.clone()));
    let bounds = region([-64, 0, 0], [65, 1, 1]);
    let original = pair.run(&ctx, config(), bounds, &f);
    assert!(!original.accepted.is_empty() && original.accepted.len() < 129);
    assert!(original.attributes.iter().all(|a| a[1] == u32::MAX));
    // Reversing vector input is neutral; reversing composition IDs is not.
    assert_eq!(
        original,
        pair.run(
            &ctx,
            config(),
            bounds,
            &field(field_config(), strokes.clone())
        )
    );
    let reversed = field(
        field_config(),
        strokes
            .into_iter()
            .enumerate()
            .map(|(i, b)| DensityBrush {
                id: ids[4 - i],
                ..b
            })
            .collect(),
    );
    assert_ne!(f.sample([0; 3]), reversed.sample([0; 3]));
    pair.run(&ctx, config(), bounds, &reversed);
}

#[test]
fn sparse_and_fully_overlapping_odd_even_trees_match_every_sample() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 129, 31);
    for count in [1, 2, 17, 30, 31] {
        for sparse in [true, false] {
            let f = field(
                field_config(),
                (0..count)
                    .map(|i| {
                        let x = if sparse {
                            (i as i64 - count as i64 / 2) * 10000
                        } else {
                            (i as i64 * 19 % 31) - 15
                        };
                        DensityBrush {
                            inner_radius_mm: if sparse { 5 } else { 200 },
                            outer_radius_mm: if sparse { 19 } else { 200 },
                            opacity: if i % 7 == 0 { 0 } else { u16::MAX },
                            density: DensityBrushOp::Add(if i % 2 == 0 { 7000 } else { -3000 }),
                            ..brush((count - i) as u64, [x, 0, 0])
                        }
                    })
                    .collect(),
            );
            pair.run(&ctx, config(), region([-64, 0, 0], [65, 1, 1]), &f);
        }
    }
}

#[test]
fn partition_union_and_source_min_distance_are_index_independent() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 256, 3);
    let f = field(
        ScatterFieldConfig {
            base_density: 55000,
            ..field_config()
        },
        vec![
            DensityBrush {
                density: DensityBrushOp::Add(-25000),
                ..brush(99, [-9, 0, 0])
            },
            DensityBrush {
                density: DensityBrushOp::Add(13000),
                ..brush(0, [12, 0, 0])
            },
            brush(u64::MAX, [10000; 3]),
        ],
    );
    let whole = region([-35, -7, -7], [36, 8, 8]);
    for distance in [0, 7, 10] {
        let cfg = ScatterConfig {
            grid_spacing_mm: 7,
            min_distance_mm: distance,
            ..config()
        };
        let expected = pair.run(&ctx, cfg, whole, &f).accepted;
        assert!(!expected.is_empty());
        let mut union = BTreeMap::new();
        for (x0, x1) in [(17, 36), (0, 17), (-1, 0), (-35, -1)] {
            for (z0, z1) in [(0, 8), (-7, 0)] {
                let part = pair.run(&ctx, cfg, region([x0, -7, z0], [x1, 8, z1]), &f);
                for (key, value) in part.accepted {
                    assert!(union.insert(key, value).is_none());
                }
            }
        }
        assert_eq!(union, expected, "arbitrary half-open partition ownership");
        let points: Vec<_> = union.values().map(|(p, _)| p.position_mm).collect();
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                let squared: i128 = (0..3)
                    .map(|axis| {
                        let d = i128::from(a[axis]) - i128::from(b[axis]);
                        d * d
                    })
                    .sum();
                assert!(squared >= i128::from(distance) * i128::from(distance));
            }
        }
    }
}

#[test]
fn indexed_work_and_capacity_preflight_errors_preserve_old_request_and_gpu_bytes() {
    let ctx = context(5);
    let mut pair = Pair::new(&ctx, 65, 3);
    let cfg = ScatterConfig {
        min_distance_mm: 2,
        ..config()
    };
    let bounds = region([-32, 0, 0], [33, 1, 1]);
    let old = field(
        field_config(),
        vec![brush(1, [-10, 0, 0]), brush(9, [10, 0, 0])],
    );
    let original = pair.run(&ctx, cfg, bounds, &old);
    let key = pair.indexed.cache_key().cloned();
    let scratch = pair.indexed.brush_scratch_bytes();
    let changed = ScatterConfig {
        seed: !cfg.seed,
        min_distance_mm: 0,
        ..cfg
    };
    let changed_field = field(
        ScatterFieldConfig {
            seed: 0,
            base_density: 0,
            ..field_config()
        },
        vec![brush(77, [i64::MAX; 3])],
    );
    let work = estimate_scatter_work(changed, bounds).unwrap();
    let required = work.primary_candidates * sample_work(&changed_field, true);
    let too_many = field(field_config(), (0..4).map(|i| brush(i, [0; 3])).collect());
    let cases = [
        (
            &changed_field,
            bounds,
            work.max_candidate_evaluations,
            required - 1,
            GpuScatterFieldError::Field(ScatterFieldError::WorkBudgetExceeded {
                required,
                limit: required - 1,
            }),
        ),
        // The former linear budget is insufficient even with no field survivors.
        (
            &changed_field,
            bounds,
            work.max_candidate_evaluations,
            work.primary_candidates * changed_field.work_per_sample(),
            GpuScatterFieldError::Field(ScatterFieldError::WorkBudgetExceeded {
                required,
                limit: work.primary_candidates * changed_field.work_per_sample(),
            }),
        ),
        (
            &changed_field,
            bounds,
            work.max_candidate_evaluations - 1,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::Scatter(
                ScatterError::CandidateBudgetExceeded {
                    required: work.max_candidate_evaluations,
                    limit: work.max_candidate_evaluations - 1,
                },
            )),
        ),
        (
            &changed_field,
            region([-32, 0, 0], [34, 1, 1]),
            u64::MAX,
            u64::MAX,
            GpuScatterFieldError::Scatter(GpuScatterError::CapacityExceeded {
                required: 66,
                capacity: 65,
            }),
        ),
        (
            &too_many,
            bounds,
            u64::MAX,
            0,
            GpuScatterFieldError::BrushCapacityExceeded {
                required: 4,
                capacity: 3,
            },
        ),
    ];
    for (f, bounds_changed, source_limit, field_limit, error) in cases {
        let before = read(&ctx, &capture(&ctx, &pair.indexed, false).buffer);
        assert_eq!(
            pair.indexed
                .write_request(changed, bounds_changed, source_limit, f, field_limit),
            Err(error)
        );
        assert_eq!(pair.indexed.cache_key(), key.as_ref());
        assert_eq!(pair.indexed.candidate_count(), 65);
        assert_eq!(pair.indexed.brush_scratch_bytes(), scratch);
        let after = read(&ctx, &capture(&ctx, &pair.indexed, false).buffer);
        let defined = 16 + 80 * 65 + 4 * original.accepted.len();
        assert_eq!(
            &before[..defined],
            &after[..defined],
            "error changed old GPU output"
        );
        let replay = capture(&ctx, &pair.indexed, true);
        assert_eq!(
            verify(&ctx, &replay, cfg, bounds, &old),
            original,
            "error enqueued stray source/brush/node writes"
        );
    }
    assert_eq!(
        pair.indexed.max_sample_work(&too_many),
        Err(GpuScatterFieldError::BrushCapacityExceeded {
            required: 4,
            capacity: 3
        })
    );
    assert_eq!(pair.indexed.cache_key(), key.as_ref());
    // Exact indexed admission succeeds after failures; budgets do not enter identity.
    pair.indexed
        .write_request(
            changed,
            bounds,
            work.max_candidate_evaluations,
            &changed_field,
            required,
        )
        .unwrap();
    verify(
        &ctx,
        &capture(&ctx, &pair.indexed, true),
        changed,
        bounds,
        &changed_field,
    );
}

fn deletion_work(n: usize, count: usize) -> GpuScatterDeletionWorkBudget {
    GpuScatterDeletionWorkBudget {
        max_input_slots: n as u64,
        max_keys: count,
        max_key_comparisons: n as u64 * u64::from(usize::BITS - count.leading_zeros()),
    }
}

#[test]
fn indexed_field_managed_deletions_keep_upstream_abi_and_cache_identity() {
    let ctx = context(6);
    let cfg = config();
    let bounds = region([-64, 0, 0], [65, 1, 1]);
    let f = field(
        ScatterFieldConfig {
            base_density: 55000,
            ..field_config()
        },
        vec![
            DensityBrush {
                density: DensityBrushOp::Add(-20000),
                ..brush(7, [-20, 0, 0])
            },
            brush(1, [20, 0, 0]),
        ],
    );
    let mut pair = Pair::new(&ctx, 129, 2);
    let baseline = pair.run(&ctx, cfg, bounds, &f);
    let key = pair.indexed.cache_key().unwrap().clone();
    let request = GpuScatterFieldRequest {
        source: GpuScatterSourceRequest {
            config: cfg,
            region: bounds,
            max_candidate_evaluations: estimate_scatter_work(cfg, bounds)
                .unwrap()
                .max_candidate_evaluations,
        },
        field: &f,
        max_field_work: 129 * sample_work(&f, true),
    };
    let mut gpu = GpuScatterDeletions::new(
        &ctx,
        pair.indexed,
        GpuScatterDeletionCapacity {
            max_keys: 129,
            max_buffer_bytes: 1_048_576,
            max_scratch_bytes: 52 * 129,
        },
    )
    .unwrap();
    assert!(baseline.accepted.len() > 3);
    let keys: Vec<_> = baseline.accepted.keys().copied().collect();
    for removed in [vec![], keys.iter().step_by(2).copied().collect(), keys] {
        let set = deletions(removed);
        gpu.write_request(&request, &set, deletion_work(129, set.keys().len()))
            .unwrap();
        assert_eq!(gpu.upstream_cache_key(), Some(&key));
        assert_eq!(gpu.deletions(), Some(&set));
        let expected: Vec<_> = baseline
            .accepted
            .values()
            .filter(|(p, _)| set.keys().binary_search(&p.key).is_err())
            .map(|(p, _)| *p)
            .collect();
        let attrs: BTreeMap<_, _> = baseline
            .accepted
            .iter()
            .map(|(k, (_, s))| {
                (
                    *k,
                    [u32::from(s.density), s.species, 1, f.candidate_rank(*k)],
                )
            })
            .collect();
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        gpu.encode(&mut encoder).unwrap();
        let output = Capture::new(
            &ctx,
            &mut encoder,
            gpu.candidate_count(),
            gpu.records(),
            gpu.attributes(),
            gpu.compact_indices(),
            gpu.indirect_args(),
        );
        ctx.queue().submit([encoder.finish()]);
        output.verify(&ctx, cfg, &expected, Some(&attrs), VERTICES);
        let data = read(&ctx, &output.buffer);
        let words: Vec<_> = data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(&words[4..4 + 16 * 129], baseline.records.as_flattened());
        assert_eq!(
            &words[4 + 16 * 129..4 + 20 * 129],
            baseline.attributes.as_flattened()
        );
    }
}

#[cfg(feature = "terrain_scatter")]
mod terrain {
    use super::*;
    use prism_world_gpu::{
        GpuScatterHeightfield, GpuScatterHeightfieldError, GpuScatterHeightfieldRequest,
    };
    use prism_world_residency::{scatter_with_density_field, FieldScatterBudget, ScatterBudget};
    use prism_world_terrain::{
        scatter_with_heightfield, BakeLimits, CubeFace, HeightSample, HeightSampleValue,
        HeightSampler, HeightfieldScatterBudget, HeightfieldScatterFilters, HeightfieldSpec,
        HeightfieldTile, PlanarHeightfieldEmbedding, PlanarHeightfieldSurface, TileKey,
    };
    use std::{convert::Infallible, sync::Arc};

    struct Flat;
    impl HeightSampler for Flat {
        type Error = Infallible;
        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(0))
        }
    }

    #[test]
    fn indexed_heightfield_and_managed_deletions_match_public_cpu_composition() {
        let ctx = context(6);
        let cfg = ScatterConfig {
            min_distance_mm: 2,
            ..config()
        };
        let bounds = region([-6, -3, -6], [7, 4, 7]);
        let n = slots(cfg, bounds);
        let f = field(
            ScatterFieldConfig {
                base_density: 55000,
                ..field_config()
            },
            vec![
                DensityBrush {
                    density: DensityBrushOp::Add(-17000),
                    ..brush(9, [2, 0, 0])
                },
                DensityBrush {
                    density: DensityBrushOp::Add(7000),
                    ..brush(1, [-2, 0, 0])
                },
            ],
        );
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(3, 1).unwrap(),
            &Flat,
            BakeLimits {
                max_samples: 9,
                max_bytes: 9 * size_of::<i16>(),
            },
        )
        .unwrap();
        let surface = PlanarHeightfieldSurface::new(
            Arc::new(tile),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-8, 0, -8],
                sample_spacing_mm: 8,
            },
        )
        .unwrap();
        let filters = HeightfieldScatterFilters {
            surface_height_mm: [0, 0],
            clearance_mm: [-1, 1],
            max_slope_rise: 0,
            max_slope_run: 1,
        };
        let surface_budget = HeightfieldScatterBudget {
            max_sample_reads: 4 * n as u64,
        };
        let source_budget = ScatterBudget {
            max_candidate_evaluations: estimate_scatter_work(cfg, bounds)
                .unwrap()
                .max_candidate_evaluations,
            max_output_points: n,
            max_output_bytes: n * size_of::<ScatterPoint>(),
        };
        let field_budget = FieldScatterBudget {
            max_field_work: n as u64 * f.work_per_sample(),
            max_output_bytes: n * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
        };
        let cpu = scatter_with_heightfield(
            cfg,
            bounds,
            source_budget,
            &f,
            field_budget,
            &surface,
            filters,
            surface_budget,
        )
        .unwrap();
        let field_only =
            scatter_with_density_field(cfg, bounds, source_budget, &f, field_budget).unwrap();
        assert!(
            cpu.points.len() > 3 && cpu.points.len() < field_only.points.len(),
            "terrain must actually thin the field"
        );
        let budget = bytes(n, 2, true) + 20 * n.max(1) as u64 + 4 * 9 + 144;
        assert!(
            matches!(GpuScatterHeightfield::new_indexed(&ctx, n, 2, 9, VERTICES, budget - 1),
            Err(GpuScatterHeightfieldError::Field(GpuScatterFieldError::Scatter(GpuScatterError::BufferBudgetExceeded { required, limit })))
            if required == budget && limit == budget - 1)
        );
        let mut upstream =
            GpuScatterHeightfield::new_indexed(&ctx, n, 2, 9, VERTICES, budget).unwrap();
        assert!(upstream.brush_index_enabled());
        assert_eq!(upstream.max_sample_work(&f), Ok(sample_work(&f, true)));
        assert_eq!(upstream.allocated_buffer_bytes(), budget);
        let request = GpuScatterHeightfieldRequest {
            field: GpuScatterFieldRequest {
                source: GpuScatterSourceRequest {
                    config: cfg,
                    region: bounds,
                    max_candidate_evaluations: source_budget.max_candidate_evaluations,
                },
                field: &f,
                max_field_work: n as u64 * sample_work(&f, true),
            },
            surface: &surface,
            filters,
            surface_budget,
        };
        upstream
            .write_request(
                cfg,
                bounds,
                source_budget.max_candidate_evaluations,
                &f,
                request.field.max_field_work,
                &surface,
                filters,
                surface_budget,
            )
            .unwrap();
        assert_eq!(upstream.cache_key(), Some(&cpu.cache_key));
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        upstream.encode(&mut encoder).unwrap();
        let baseline = Capture::new(
            &ctx,
            &mut encoder,
            upstream.candidate_count(),
            upstream.records(),
            Some(upstream.attributes()),
            upstream.compact_indices(),
            upstream.indirect_args(),
        );
        ctx.queue().submit([encoder.finish()]);
        let attrs: BTreeMap<_, _> = cpu
            .points
            .iter()
            .zip(&cpu.attributes)
            .map(|(p, s)| {
                (
                    p.key,
                    [u32::from(s.density), s.species, 1, f.candidate_rank(p.key)],
                )
            })
            .collect();
        baseline.verify(&ctx, cfg, &cpu.points, Some(&attrs), VERTICES);
        let prefix = read(&ctx, &baseline.buffer)[16..16 + 80 * n].to_vec();
        let mut gpu = GpuScatterDeletions::new(
            &ctx,
            upstream,
            GpuScatterDeletionCapacity {
                max_keys: n,
                max_buffer_bytes: budget + 52 * n as u64 + 4 * n as u64 + 144,
                max_scratch_bytes: 52 * n,
            },
        )
        .unwrap();
        let keys: Vec<_> = cpu.points.iter().map(|p| p.key).collect();
        for removed in [vec![], keys.iter().step_by(2).copied().collect(), keys] {
            let set = deletions(removed);
            gpu.write_request(&request, &set, deletion_work(n, set.keys().len()))
                .unwrap();
            assert_eq!(gpu.upstream_cache_key(), Some(&cpu.cache_key));
            let expected: Vec<_> = cpu
                .points
                .iter()
                .filter(|p| set.keys().binary_search(&p.key).is_err())
                .copied()
                .collect();
            let mut encoder = ctx.device().create_command_encoder(&Default::default());
            gpu.encode(&mut encoder).unwrap();
            let output = Capture::new(
                &ctx,
                &mut encoder,
                gpu.candidate_count(),
                gpu.records(),
                gpu.attributes(),
                gpu.compact_indices(),
                gpu.indirect_args(),
            );
            ctx.queue().submit([encoder.finish()]);
            output.verify(&ctx, cfg, &expected, Some(&attrs), VERTICES);
            assert_eq!(
                &read(&ctx, &output.buffer)[16..16 + 80 * n],
                prefix.as_slice(),
                "managed deletion changed upstream source/terrain attributes"
            );
        }
    }
}
