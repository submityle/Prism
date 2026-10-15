//! Physical-device deletion composition against public CPU field/terrain APIs.
//! Readback is test-only; compact order and unused compact tails are unspecified.

#![cfg(feature = "gpu")]

#[path = "support/scatter_deletion.rs"]
mod support;

use std::collections::BTreeMap;

use prism_world_gpu::{
    GpuContext, GpuScatterDeletionCacheKey, GpuScatterDeletionCapacity, GpuScatterDeletionError,
    GpuScatterDeletionUpstream, GpuScatterDeletionWorkBudget, GpuScatterDeletions, GpuScatterError,
    GpuScatterField, GpuScatterFieldError, GpuScatterFieldRequest, GpuScatterSourceRequest,
    GPU_SCATTER_DELETION_KEY_ABI_VERSION, GPU_SCATTER_DELETION_VERSION,
};
use prism_world_residency::{
    estimate_scatter_work, scatter_with_density_field, DensityBrush, DensityBrushOp, DensityNoise,
    FieldScatterBudget, FieldScatterOutput, ScatterBudget, ScatterCandidateKey, ScatterConfig,
    ScatterDeletionSet, ScatterDensityField, ScatterError, ScatterFieldBuildBudget,
    ScatterFieldConfig, ScatterFieldError, ScatterFieldSample, ScatterPoint, ScatterRegion,
    SCATTER_DELETION_VERSION,
};
use support::{context, deletions, read, region, Capture};

const GPU_BYTES: u64 = 1_048_576;
const VERTICES: u32 = 9;

fn field_config(density: u16) -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 91,
        version: 3,
        base_density: density,
        base_species: 11,
        noise: None,
    }
}

fn make_field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(config, brushes, budget).unwrap()
}

fn constant(density: u16) -> ScatterDensityField {
    make_field(field_config(density), vec![])
}

fn brush(id: u64, x: i64) -> DensityBrush {
    DensityBrush {
        id,
        center_mm: [x, 0, 0],
        inner_radius_mm: 40,
        outer_radius_mm: 80,
        opacity: u16::MAX,
        density: DensityBrushOp::Add(-12_000),
        species: Some(if x < 0 { 23 } else { 97 }),
    }
}

fn painted() -> ScatterDensityField {
    make_field(
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                cell_size_mm: 29,
                amplitude: 12_000,
                warp_cell_size_mm: 41,
                warp_amplitude_mm: 7,
            }),
            ..field_config(46_000)
        },
        vec![brush(2, -30), brush(9, 30)],
    )
}

fn fixture_config() -> ScatterConfig {
    ScatterConfig {
        min_distance_mm: 13,
        ..support::config()
    }
}

fn bounds() -> ScatterRegion {
    region([-48, -24, -48], [49, 25, 49])
}

fn slots(cfg: ScatterConfig, bounds: ScatterRegion) -> usize {
    let n = estimate_scatter_work(cfg, bounds)
        .unwrap()
        .primary_candidates;
    assert!(n <= 8_192, "composed oracle fixtures must remain bounded");
    usize::try_from(n).unwrap()
}

fn source_budget(cfg: ScatterConfig, bounds: ScatterRegion) -> ScatterBudget {
    let n = slots(cfg, bounds);
    ScatterBudget {
        max_candidate_evaluations: estimate_scatter_work(cfg, bounds)
            .unwrap()
            .max_candidate_evaluations,
        max_output_points: n,
        max_output_bytes: n * size_of::<ScatterPoint>(),
    }
}

fn field_budget(
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    field: &ScatterDensityField,
) -> FieldScatterBudget {
    let n = slots(cfg, bounds);
    FieldScatterBudget {
        max_field_work: n as u64 * field.work_per_sample(),
        max_output_bytes: n * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
    }
}

fn cpu_field(
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    field: &ScatterDensityField,
) -> FieldScatterOutput {
    scatter_with_density_field(
        cfg,
        bounds,
        source_budget(cfg, bounds),
        field,
        field_budget(cfg, bounds, field),
    )
    .unwrap()
}

fn field_request(
    cfg: ScatterConfig,
    bounds: ScatterRegion,
    field: &ScatterDensityField,
) -> GpuScatterFieldRequest<'_> {
    GpuScatterFieldRequest {
        source: GpuScatterSourceRequest {
            config: cfg,
            region: bounds,
            max_candidate_evaluations: estimate_scatter_work(cfg, bounds)
                .unwrap()
                .max_candidate_evaluations,
        },
        field,
        max_field_work: field_budget(cfg, bounds, field).max_field_work,
    }
}

fn work(n: usize, set: &ScatterDeletionSet) -> GpuScatterDeletionWorkBudget {
    let d = set.keys().len();
    GpuScatterDeletionWorkBudget {
        max_input_slots: n as u64,
        max_keys: d,
        max_key_comparisons: n as u64 * u64::from(usize::BITS - d.leading_zeros()),
    }
}

fn wrap<U: GpuScatterDeletionUpstream>(
    ctx: &GpuContext,
    upstream: U,
    max_keys: usize,
) -> GpuScatterDeletions<U> {
    GpuScatterDeletions::new(
        ctx,
        upstream,
        GpuScatterDeletionCapacity {
            max_keys,
            max_buffer_bytes: GPU_BYTES,
            max_scratch_bytes: 52 * max_keys,
        },
    )
    .unwrap()
}

fn capture<U: GpuScatterDeletionUpstream>(
    ctx: &GpuContext,
    gpu: &GpuScatterDeletions<U>,
    encode: bool,
) -> Capture {
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    if encode {
        gpu.encode(&mut encoder).unwrap();
    }
    let result = Capture::new(
        ctx,
        &mut encoder,
        gpu.candidate_count(),
        gpu.records(),
        gpu.attributes(),
        gpu.compact_indices(),
        gpu.indirect_args(),
    );
    ctx.queue().submit([encoder.finish()]);
    result
}

fn attributes(
    points: &[ScatterPoint],
    values: &[ScatterFieldSample],
    field: &ScatterDensityField,
) -> BTreeMap<ScatterCandidateKey, [u32; 4]> {
    assert_eq!(points.len(), values.len());
    points
        .iter()
        .zip(values)
        .map(|(p, a)| {
            (
                p.key,
                [
                    u32::from(a.density),
                    a.species,
                    1,
                    field.candidate_rank(p.key),
                ],
            )
        })
        .collect()
}

fn prefix(ctx: &GpuContext, capture: &Capture, n: usize) -> Vec<u8> {
    // Every candidate record/attribute byte, including upstream rejects. Never
    // compare compact tails or assume deterministic cross-workgroup order.
    read(ctx, &capture.buffer)[16..16 + 80 * n].to_vec()
}

fn accepted_flags(ctx: &GpuContext, capture: &Capture, n: usize) -> (usize, usize) {
    let data = prefix(ctx, capture, n);
    let word = |offset: usize| u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    let source = (0..n).filter(|&i| word(64 * i + 48) == 1).count();
    let final_upstream = (0..n).filter(|&i| word(64 * n + 16 * i + 8) == 1).count();
    (source, final_upstream)
}

fn assert_cache<U: GpuScatterDeletionUpstream>(
    gpu: &GpuScatterDeletions<U>,
    upstream: &U::CacheKey,
    set: &ScatterDeletionSet,
) {
    let expected = GpuScatterDeletionCacheKey {
        upstream: upstream.clone(),
        deletions: set.clone(),
        deletion_version: SCATTER_DELETION_VERSION,
        gpu_deletion_version: GPU_SCATTER_DELETION_VERSION,
        key_abi_version: GPU_SCATTER_DELETION_KEY_ABI_VERSION,
        draw_args: [VERTICES, 0, 0],
    };
    assert_eq!(gpu.cache_key(), Some(&expected));
    assert_eq!(gpu.upstream_cache_key(), Some(upstream));
    assert_eq!(gpu.deletions(), Some(set));
}

fn outside_key(point: ScatterPoint) -> ScatterCandidateKey {
    ScatterCandidateKey {
        grid: prism_world_residency::SectorCoord::new(10_000, 0, 0),
        ..point.key
    }
}

fn sets(points: &[ScatterPoint]) -> [ScatterDeletionSet; 3] {
    assert!(points.len() > 3);
    [
        ScatterDeletionSet::default(),
        deletions(
            points
                .iter()
                .step_by(2)
                .map(|p| p.key)
                .chain([outside_key(points[0])])
                .collect(),
        ),
        deletions(points.iter().map(|p| p.key).collect()),
    ]
}

struct OutputOracle<'a> {
    config: ScatterConfig,
    points: &'a [ScatterPoint],
    attributes: &'a BTreeMap<ScatterCandidateKey, [u32; 4]>,
}

fn failed_request_preserves<U: GpuScatterDeletionUpstream>(
    ctx: &GpuContext,
    gpu: &mut GpuScatterDeletions<U>,
    request: &U::Request<'_>,
    set: &ScatterDeletionSet,
    budget: GpuScatterDeletionWorkBudget,
    expected_error: GpuScatterDeletionError,
    oracle: &OutputOracle<'_>,
) {
    let OutputOracle {
        config: cfg,
        points: expected,
        attributes: attrs,
    } = *oracle;
    let before = capture(ctx, gpu, true);
    before.verify(ctx, cfg, expected, Some(attrs), VERTICES);
    let bytes = read(ctx, &before.buffer);
    let n = gpu.candidate_count() as usize;
    let cache = gpu.cache_key().cloned();
    let stats = gpu.upload_stats();
    assert_eq!(
        gpu.validate_request(request, set, budget),
        Err(expected_error.clone())
    );
    assert_eq!(gpu.write_request(request, set, budget), Err(expected_error));
    assert_eq!(gpu.cache_key(), cache.as_ref());
    assert_eq!(gpu.upload_stats(), stats);
    assert_eq!(gpu.candidate_count() as usize, n);
    let untouched = capture(ctx, gpu, false);
    let after = read(ctx, &untouched.buffer);
    let defined_bytes = 16 + 80 * n + 4 * expected.len();
    assert_eq!(
        &after[..defined_bytes],
        &bytes[..defined_bytes],
        "failed admission changed existing GPU outputs"
    );
    // Also replay the retained request: stray parameter/recipe queue writes on
    // an error must not silently change the next encoding of the old request.
    let replay = capture(ctx, gpu, true);
    replay.verify(ctx, cfg, expected, Some(attrs), VERTICES);
    assert_eq!(prefix(ctx, &replay, n), bytes[16..16 + 80 * n]);
}

#[test]
fn field_empty_partial_all_deletions_match_public_cpu_and_preserve_full_prefixes() {
    let ctx = context();
    let cfg = fixture_config();
    let field = painted();
    let cpu = cpu_field(cfg, bounds(), &field);
    let raw = support::source(cfg, bounds());
    assert!(cpu.points.len() > 3 && cpu.points.len() < raw.len());
    assert!(cpu
        .attributes
        .windows(2)
        .any(|w| w[0].density != w[1].density));
    assert!(cpu
        .attributes
        .windows(2)
        .any(|w| w[0].species != w[1].species));
    let n = slots(cfg, bounds());
    let request = field_request(cfg, bounds(), &field);
    let mut upstream = GpuScatterField::new(&ctx, n, 2, VERTICES, GPU_BYTES).unwrap();
    upstream
        .write_request(
            cfg,
            bounds(),
            request.source.max_candidate_evaluations,
            &field,
            request.max_field_work,
        )
        .unwrap();
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
    baseline.verify(
        &ctx,
        cfg,
        &cpu.points,
        Some(&attributes(&cpu.points, &cpu.attributes, &field)),
        VERTICES,
    );
    // Establish the entire prefix before the deletion adapter ever owns it:
    // even corruption shared by empty/partial/full deletion must be detected.
    let original_prefix = prefix(&ctx, &baseline, n);
    let mut gpu = wrap(&ctx, upstream, n + 4);
    for set in sets(&cpu.points) {
        let expected = set.filter_field(cpu_field(cfg, bounds(), &field)).unwrap();
        let attrs = attributes(&expected.source.points, &expected.source.attributes, &field);
        gpu.write_request(&request, &set, work(n, &set)).unwrap();
        assert_cache(&gpu, &cpu.cache_key, &set);
        let snapshot = capture(&ctx, &gpu, true);
        snapshot.verify(&ctx, cfg, &expected.source.points, Some(&attrs), VERTICES);
        let (source_flags, final_flags) = accepted_flags(&ctx, &snapshot, n);
        assert_eq!(source_flags, raw.len());
        assert_eq!(
            final_flags,
            cpu.points.len(),
            "deletion must not rewrite field accepted flags"
        );
        let current = prefix(&ctx, &snapshot, n);
        assert_eq!(
            current, original_prefix,
            "all source/attribute slots must remain byte-exact to the unwrapped upstream"
        );
    }
}

#[test]
fn empty_field_final_compact_never_resurrects_raw_source_flags() {
    let ctx = context();
    let cfg = fixture_config();
    let field = constant(0);
    let raw = support::source(cfg, bounds());
    assert!(!raw.is_empty());
    let cpu = cpu_field(cfg, bounds(), &field);
    assert!(cpu.points.is_empty());
    let n = slots(cfg, bounds());
    let mut gpu = wrap(
        &ctx,
        GpuScatterField::new(&ctx, n, 0, VERTICES, GPU_BYTES).unwrap(),
        1,
    );
    for set in [
        ScatterDeletionSet::default(),
        deletions(vec![outside_key(raw[0])]),
    ] {
        gpu.write_request(&field_request(cfg, bounds(), &field), &set, work(n, &set))
            .unwrap();
        let snapshot = capture(&ctx, &gpu, true);
        snapshot.verify(&ctx, cfg, &[], Some(&BTreeMap::new()), VERTICES);
        assert_eq!(accepted_flags(&ctx, &snapshot, n), (raw.len(), 0));
        assert_cache(&gpu, &cpu.cache_key, &set);
    }
}

#[test]
fn field_cache_retains_inactive_recipe_full_deletions_and_new_namespace() {
    let ctx = context();
    let cfg = fixture_config();
    let mut inactive = brush(55, 10_000);
    let first_field = make_field(field_config(u16::MAX), vec![inactive]);
    inactive.opacity -= 1;
    let second_field = make_field(field_config(u16::MAX), vec![inactive]);
    let first = cpu_field(cfg, bounds(), &first_field);
    let second = cpu_field(cfg, bounds(), &second_field);
    assert_eq!(first.points, second.points);
    assert_eq!(first.attributes, second.attributes);
    assert_ne!(first.cache_key, second.cache_key);
    let n = slots(cfg, bounds());
    let mut gpu = wrap(
        &ctx,
        GpuScatterField::new(&ctx, n, 1, VERTICES, GPU_BYTES).unwrap(),
        n + 4,
    );
    let absent = deletions(vec![outside_key(first.points[0])]);
    let empty = ScatterDeletionSet::default();
    let mut old = None;
    for (field, set) in [
        (&first_field, &empty),
        (&first_field, &absent),
        (&second_field, &absent),
    ] {
        let cpu = cpu_field(cfg, bounds(), field);
        gpu.write_request(&field_request(cfg, bounds(), field), set, work(n, set))
            .unwrap();
        assert_cache(&gpu, &cpu.cache_key, set);
        if let Some(previous) = old {
            assert_ne!(gpu.cache_key().unwrap(), &previous);
        }
        old = gpu.cache_key().cloned();
        let snapshot = capture(&ctx, &gpu, true);
        snapshot.verify(
            &ctx,
            cfg,
            &cpu.points,
            Some(&attributes(&cpu.points, &cpu.attributes, field)),
            VERTICES,
        );
    }
    let old_keys = deletions(first.points.iter().map(|p| p.key).collect());
    let namespace = ScatterConfig {
        seed: cfg.seed ^ (1 << 40),
        version: cfg.version + 1,
        salt: cfg.salt ^ (1 << 41),
        ..cfg
    };
    let cpu = cpu_field(namespace, bounds(), &first_field);
    assert!(!cpu.points.is_empty());
    assert!(cpu.points.iter().all(|p| !old_keys.is_deleted(p.key)));
    gpu.write_request(
        &field_request(namespace, bounds(), &first_field),
        &old_keys,
        work(n, &old_keys),
    )
    .unwrap();
    assert_cache(&gpu, &cpu.cache_key, &old_keys);
    capture(&ctx, &gpu, true).verify(
        &ctx,
        namespace,
        &cpu.points,
        Some(&attributes(&cpu.points, &cpu.attributes, &first_field)),
        VERTICES,
    );
}

#[test]
fn field_full_chain_preflight_failures_preserve_old_gpu_output_and_request() {
    let ctx = context();
    let cfg = fixture_config();
    let field = painted();
    let n = slots(cfg, bounds());
    let cpu = cpu_field(cfg, bounds(), &field);
    let set = deletions(vec![cpu.points[0].key]);
    let expected = set.filter_field(cpu_field(cfg, bounds(), &field)).unwrap();
    let attrs = attributes(&expected.source.points, &expected.source.attributes, &field);
    let oracle = OutputOracle {
        config: cfg,
        points: &expected.source.points,
        attributes: &attrs,
    };
    let mut gpu = wrap(
        &ctx,
        GpuScatterField::new(&ctx, n, 2, VERTICES, GPU_BYTES).unwrap(),
        2,
    );
    let request = field_request(cfg, bounds(), &field);
    gpu.write_request(&request, &set, work(n, &set)).unwrap();
    let mut changed = field_request(
        ScatterConfig {
            seed: cfg.seed + 1,
            ..cfg
        },
        bounds(),
        &field,
    );
    changed.max_field_work = 0;
    let field_error = GpuScatterDeletionError::Field(GpuScatterFieldError::Field(
        ScatterFieldError::WorkBudgetExceeded {
            required: n as u64 * field.work_per_sample(),
            limit: 0,
        },
    ));
    failed_request_preserves(
        &ctx,
        &mut gpu,
        &changed,
        &set,
        work(n, &set),
        field_error,
        &oracle,
    );
    let mut invalid = request;
    invalid.source.config.grid_spacing_mm = 0;
    failed_request_preserves(
        &ctx,
        &mut gpu,
        &invalid,
        &set,
        work(n, &set),
        GpuScatterDeletionError::Field(GpuScatterFieldError::Scatter(GpuScatterError::Scatter(
            ScatterError::InvalidGridSpacing,
        ))),
        &oracle,
    );
    let oversized_field = make_field(
        field_config(u16::MAX),
        vec![brush(1, -30), brush(2, 0), brush(3, 30)],
    );
    failed_request_preserves(
        &ctx,
        &mut gpu,
        &field_request(cfg, bounds(), &oversized_field),
        &set,
        work(n, &set),
        GpuScatterDeletionError::Field(GpuScatterFieldError::BrushCapacityExceeded {
            required: 3,
            capacity: 2,
        }),
        &oracle,
    );
    let too_many = deletions(cpu.points.iter().take(3).map(|p| p.key).collect());
    failed_request_preserves(
        &ctx,
        &mut gpu,
        &request,
        &too_many,
        work(n, &too_many),
        GpuScatterDeletionError::KeyCapacityExceeded {
            required: 3,
            capacity: 2,
        },
        &oracle,
    );
    let budget = work(n, &set);
    for (budget, error) in [
        (
            GpuScatterDeletionWorkBudget {
                max_input_slots: 0,
                ..budget
            },
            GpuScatterDeletionError::InputBudgetExceeded {
                required: n as u64,
                limit: 0,
            },
        ),
        (
            GpuScatterDeletionWorkBudget {
                max_keys: 0,
                ..budget
            },
            GpuScatterDeletionError::KeyBudgetExceeded {
                required: 1,
                limit: 0,
            },
        ),
        (
            GpuScatterDeletionWorkBudget {
                max_key_comparisons: 0,
                ..budget
            },
            GpuScatterDeletionError::ComparisonBudgetExceeded {
                required: n as u64,
                limit: 0,
            },
        ),
    ] {
        failed_request_preserves(&ctx, &mut gpu, &request, &set, budget, error, &oracle);
    }
}

#[cfg(feature = "terrain_scatter")]
mod terrain {
    use std::{convert::Infallible, sync::Arc};

    use super::*;
    use prism_world_gpu::{
        GpuScatterHeightfield, GpuScatterHeightfieldError, GpuScatterHeightfieldRequest,
    };
    use prism_world_terrain::{
        filter_heightfield_scatter, scatter_with_heightfield, BakeLimits, CubeFace, HeightSample,
        HeightSampleValue, HeightSampler, HeightfieldQueryError, HeightfieldScatterBudget,
        HeightfieldScatterError, HeightfieldScatterFilters, HeightfieldScatterOutput,
        HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding, PlanarHeightfieldSurface,
        TileKey,
    };

    struct Heights<'a>(&'a [i64; 9]);
    impl HeightSampler for Heights<'_> {
        type Error = Infallible;
        fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            let index = |q: u16| ((u64::from(q) * 2 + 32_767) / 65_535) as usize;
            Ok(HeightSampleValue::from_millimetres(
                self.0[index(sample.v_q16()) * 3 + index(sample.u_q16())],
            ))
        }
    }

    fn surface(heights: &[i64; 9]) -> PlanarHeightfieldSurface {
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(3, 1).unwrap(),
            &Heights(heights),
            BakeLimits {
                max_samples: 9,
                max_bytes: 9 * size_of::<i16>(),
            },
        )
        .unwrap();
        PlanarHeightfieldSurface::new(
            Arc::new(tile),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-8, 0, -8],
                sample_spacing_mm: 8,
            },
        )
        .unwrap()
    }

    fn cfg() -> ScatterConfig {
        ScatterConfig {
            grid_spacing_mm: 1,
            min_distance_mm: 2,
            ..support::config()
        }
    }

    fn bounds() -> ScatterRegion {
        region([-6, -3, -6], [7, 4, 7])
    }

    fn unrestricted() -> HeightfieldScatterFilters {
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            clearance_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        }
    }

    fn filters() -> HeightfieldScatterFilters {
        HeightfieldScatterFilters {
            surface_height_mm: [-2, 2],
            clearance_mm: [-1, 1],
            max_slope_rise: 1,
            max_slope_run: 2,
        }
    }

    fn request<'a>(
        cfg: ScatterConfig,
        bounds: ScatterRegion,
        field: &'a ScatterDensityField,
        surface: &'a PlanarHeightfieldSurface,
        filters: HeightfieldScatterFilters,
    ) -> GpuScatterHeightfieldRequest<'a> {
        GpuScatterHeightfieldRequest {
            field: field_request(cfg, bounds, field),
            surface,
            filters,
            surface_budget: HeightfieldScatterBudget {
                max_sample_reads: 4 * slots(cfg, bounds) as u64,
            },
        }
    }

    fn cpu(request: GpuScatterHeightfieldRequest<'_>) -> HeightfieldScatterOutput {
        let cfg = request.field.source.config;
        let bounds = request.field.source.region;
        scatter_with_heightfield(
            cfg,
            bounds,
            source_budget(cfg, bounds),
            request.field.field,
            field_budget(cfg, bounds, request.field.field),
            request.surface,
            request.filters,
            request.surface_budget,
        )
        .unwrap()
    }

    fn new_gpu(
        ctx: &GpuContext,
        n: usize,
        max_keys: usize,
    ) -> GpuScatterDeletions<GpuScatterHeightfield> {
        wrap(
            ctx,
            GpuScatterHeightfield::new(ctx, n, 2, 9, VERTICES, GPU_BYTES).unwrap(),
            max_keys,
        )
    }

    #[test]
    fn heightfield_empty_partial_all_match_cpu_final_compact_and_full_prefixes() {
        let ctx = context();
        let cfg = cfg();
        let field = painted();
        let surface = surface(&[0, 2, 4, -2, 0, 2, -4, -2, 0]);
        let request = request(cfg, bounds(), &field, &surface, filters());
        let original = cpu(request);
        assert!(original.points.len() > 3);
        let field_only = cpu_field(cfg, bounds(), &field);
        assert!(
            original.points.len() < field_only.points.len(),
            "terrain must actually reject field survivors"
        );
        let n = slots(cfg, bounds());
        let mut upstream = GpuScatterHeightfield::new(&ctx, n, 2, 9, VERTICES, GPU_BYTES).unwrap();
        upstream
            .write_request(
                cfg,
                bounds(),
                request.field.source.max_candidate_evaluations,
                &field,
                request.field.max_field_work,
                &surface,
                request.filters,
                request.surface_budget,
            )
            .unwrap();
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
        baseline.verify(
            &ctx,
            cfg,
            &original.points,
            Some(&attributes(&original.points, &original.attributes, &field)),
            VERTICES,
        );
        let original_prefix = prefix(&ctx, &baseline, n);
        let mut gpu = wrap(&ctx, upstream, n + 4);
        for set in sets(&original.points) {
            let expected = filter_heightfield_scatter(cpu(request), &set).unwrap();
            gpu.write_request(&request, &set, work(n, &set)).unwrap();
            assert_cache(&gpu, &original.cache_key, &set);
            assert!(Arc::ptr_eq(
                gpu.upstream_cache_key().unwrap().surface.tile(),
                surface.tile()
            ));
            let snapshot = capture(&ctx, &gpu, true);
            snapshot.verify(
                &ctx,
                cfg,
                &expected.source.points,
                Some(&attributes(
                    &expected.source.points,
                    &expected.source.attributes,
                    &field,
                )),
                VERTICES,
            );
            let (raw_flags, final_flags) = accepted_flags(&ctx, &snapshot, n);
            assert!(raw_flags > original.points.len());
            assert_eq!(
                final_flags,
                original.points.len(),
                "terrain flags precede deletion"
            );
            let current = prefix(&ctx, &snapshot, n);
            assert_eq!(
                current, original_prefix,
                "all source/attribute slots must remain byte-exact to the unwrapped upstream"
            );
        }
    }

    #[test]
    fn empty_heightfield_final_compact_does_not_resurrect_raw_or_field_flags() {
        let ctx = context();
        let cfg = cfg();
        let field = constant(u16::MAX);
        let surface = surface(&[0; 9]);
        let request = request(
            cfg,
            bounds(),
            &field,
            &surface,
            HeightfieldScatterFilters {
                surface_height_mm: [100, 100],
                ..unrestricted()
            },
        );
        let original = cpu(request);
        let raw = support::source(cfg, bounds());
        assert!(!raw.is_empty());
        assert_eq!(raw, cpu_field(cfg, bounds(), &field).points);
        assert!(original.points.is_empty());
        let n = slots(cfg, bounds());
        let mut gpu = new_gpu(&ctx, n, 1);
        for set in [
            ScatterDeletionSet::default(),
            deletions(vec![outside_key(raw[0])]),
        ] {
            gpu.write_request(&request, &set, work(n, &set)).unwrap();
            assert_cache(&gpu, &original.cache_key, &set);
            let snapshot = capture(&ctx, &gpu, true);
            snapshot.verify(&ctx, cfg, &[], Some(&BTreeMap::new()), VERTICES);
            assert_eq!(accepted_flags(&ctx, &snapshot, n), (raw.len(), 0));
        }
    }

    #[test]
    fn heightfield_cache_keeps_full_inactive_payload_embedding_filters_and_deletions() {
        let ctx = context();
        let cfg = ScatterConfig {
            min_distance_mm: 0,
            ..cfg()
        };
        let bounds = region([-7, -1, -7], [-5, 2, -5]);
        let field = constant(u16::MAX);
        let first = surface(&[0; 9]);
        let equal = surface(&[0; 9]);
        let changed = surface(&[0, 0, 0, 0, 0, 0, 0, 0, 9]);
        assert!(!Arc::ptr_eq(first.tile(), equal.tile()));
        assert_eq!(first.tile().revision(), changed.tile().revision());
        let shifted = PlanarHeightfieldSurface::new(
            first.tile().clone(),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [-8, 1, -8],
                ..first.embedding()
            },
        )
        .unwrap();
        let original = cpu(request(cfg, bounds, &field, &first, unrestricted()));
        let absent = deletions(vec![outside_key(original.points[0])]);
        let empty = ScatterDeletionSet::default();
        let n = slots(cfg, bounds);
        let mut gpu = new_gpu(&ctx, n, 1);
        let mut previous = None;
        for (index, (surface, filters, set)) in [
            (&first, unrestricted(), &empty),
            (&equal, unrestricted(), &empty),
            (&changed, unrestricted(), &empty),
            (&shifted, unrestricted(), &empty),
            (
                &shifted,
                HeightfieldScatterFilters {
                    clearance_mm: [-100, 100],
                    ..unrestricted()
                },
                &empty,
            ),
            (
                &shifted,
                HeightfieldScatterFilters {
                    clearance_mm: [-100, 100],
                    ..unrestricted()
                },
                &absent,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let request = request(cfg, bounds, &field, surface, filters);
            let cpu = cpu(request);
            assert_eq!(cpu.points, original.points);
            assert_eq!(cpu.attributes, original.attributes);
            gpu.write_request(&request, set, work(n, set)).unwrap();
            assert_cache(&gpu, &cpu.cache_key, set);
            if let Some(ref old) = previous {
                if index == 1 {
                    assert_eq!(gpu.cache_key().unwrap(), old);
                } else {
                    assert_ne!(gpu.cache_key().unwrap(), old);
                }
            }
            previous = gpu.cache_key().cloned();
            capture(&ctx, &gpu, true).verify(
                &ctx,
                cfg,
                &cpu.points,
                Some(&attributes(&cpu.points, &cpu.attributes, &field)),
                VERTICES,
            );
        }
    }

    #[test]
    fn heightfield_full_chain_preflight_errors_preserve_old_gpu_output_and_request() {
        let ctx = context();
        let cfg = cfg();
        let field = painted();
        let surface = surface(&[0; 9]);
        let good = request(cfg, bounds(), &field, &surface, filters());
        let n = slots(cfg, bounds());
        let original = cpu(good);
        let set = deletions(vec![original.points[0].key]);
        let expected = filter_heightfield_scatter(cpu(good), &set).unwrap();
        let attrs = attributes(&expected.source.points, &expected.source.attributes, &field);
        let oracle = OutputOracle {
            config: cfg,
            points: &expected.source.points,
            attributes: &attrs,
        };
        let mut gpu = new_gpu(&ctx, n, 2);
        gpu.write_request(&good, &set, work(n, &set)).unwrap();
        let invalid_filters = GpuScatterHeightfieldRequest {
            filters: HeightfieldScatterFilters {
                max_slope_run: 0,
                ..good.filters
            },
            ..good
        };
        let insufficient_reads = GpuScatterHeightfieldRequest {
            surface_budget: HeightfieldScatterBudget {
                max_sample_reads: 0,
            },
            ..good
        };
        let uncovered = PlanarHeightfieldSurface::new(
            surface.tile().clone(),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [100, 0, 100],
                ..surface.embedding()
            },
        )
        .unwrap();
        let uncovered_request = GpuScatterHeightfieldRequest {
            surface: &uncovered,
            ..good
        };
        for (request, error) in [
            (invalid_filters, HeightfieldScatterError::InvalidFilters),
            (
                insufficient_reads,
                HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                    required: 4 * n as u64,
                    limit: 0,
                },
            ),
            (
                uncovered_request,
                HeightfieldScatterError::Surface(HeightfieldQueryError::OutsideSurface),
            ),
        ] {
            failed_request_preserves(
                &ctx,
                &mut gpu,
                &request,
                &set,
                work(n, &set),
                GpuScatterDeletionError::Heightfield(GpuScatterHeightfieldError::Terrain(error)),
                &oracle,
            );
        }
        let mut changed = good;
        changed.field.source.config.seed += 1;
        changed.field.max_field_work = 0;
        failed_request_preserves(
            &ctx,
            &mut gpu,
            &changed,
            &set,
            work(n, &set),
            GpuScatterDeletionError::Heightfield(GpuScatterHeightfieldError::Field(
                GpuScatterFieldError::Field(ScatterFieldError::WorkBudgetExceeded {
                    required: n as u64 * field.work_per_sample(),
                    limit: 0,
                }),
            )),
            &oracle,
        );
        let budget = GpuScatterDeletionWorkBudget {
            max_key_comparisons: 0,
            ..work(n, &set)
        };
        failed_request_preserves(
            &ctx,
            &mut gpu,
            &good,
            &set,
            budget,
            GpuScatterDeletionError::ComparisonBudgetExceeded {
                required: n as u64,
                limit: 0,
            },
            &oracle,
        );
    }
}
