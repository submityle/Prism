//! Release-only CPU placement benchmark with an independent integer oracle.
#![expect(clippy::print_stdout, reason = "benchmark measurements")]
#![expect(
    unsafe_code,
    reason = "forwarding allocation counter, as in sibling benches"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

use prism_world_residency::{
    DensityBrush, DensityBrushOp, ScatterCandidateKey, ScatterDensityField,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample, SectorCoord,
    SCATTER_FIELD_VERSION, SCATTER_KERNEL_VERSION,
};
use prism_world_terrain::{
    estimate_heightfield_placement_work, scatter_on_heightfield, BakeLimits, CubeFace,
    HeightSample, HeightSampleValue, HeightSampler, HeightfieldPlacementBudget,
    HeightfieldPlacementCacheKey, HeightfieldPlacementConfig, HeightfieldPlacementError,
    HeightfieldPlacementFilters, HeightfieldPlacementKey, HeightfieldPlacementOutput,
    HeightfieldPlacementPoint, HeightfieldPlacementRegion, HeightfieldPlacementWork,
    HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding, PlanarHeightfieldSurface,
    TileKey, HEIGHTFIELD_PLACEMENT_VERSION, HEIGHTFIELD_SURFACE_VERSION,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 32;
const PRIMARY: usize = 10_000;
const BRUSHES: usize = 3;
const OUTPUT_BYTES_LIMIT: usize = 16 * 1024 * 1024;
const BRUSH_BYTES_LIMIT: usize = BRUSHES * size_of::<DensityBrush>();
const GRID_MIN: i64 = -50;
const GRID_MAX: i64 = 50;

// Same forwarding convention as indexed_heightfield_scatter.rs. Default
// realloc/alloc_zeroed route through alloc, so extra output growth is counted.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
struct Counter;

// SAFETY: every allocation/deallocation forwards the unchanged layout/pointer
// to System; the only additional operation is a relaxed atomic increment.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the caller's valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout originate from this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counter = Counter;

struct Ramp;

impl HeightSampler for Ramp {
    type Error = core::convert::Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(
            i64::from(sample.u_q16()) / 256 - i64::from(sample.v_q16()) / 256,
        ))
    }
}

fn surface() -> PlanarHeightfieldSurface {
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(65, 1).expect("65x65 millimetre encoding"),
        &Ramp,
        BakeLimits {
            max_samples: 65 * 65,
            max_bytes: 2 * 65 * 65,
        },
    )
    .expect("bounded real heightfield bake");
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-32_000, 0, -32_000],
            sample_spacing_mm: 1_000,
        },
    )
    .expect("valid planar embedding")
}

fn field() -> ScatterDensityField {
    // Full-opacity hard spheres make an independent exact attribute oracle
    // simple. Reverse IDs exercise canonical, noncommutative composition.
    let brushes = vec![
        DensityBrush {
            id: 24,
            center_mm: [0, 7, 2_500],
            inner_radius_mm: 2_000,
            outer_radius_mm: 2_000,
            opacity: u16::MAX,
            density: DensityBrushOp::Add(-8_000),
            species: Some(24),
        },
        DensityBrush {
            id: 15,
            center_mm: [1_500, 7, 0],
            inner_radius_mm: 2_700,
            outer_radius_mm: 2_700,
            opacity: u16::MAX,
            density: DensityBrushOp::Replace(u16::MAX),
            species: Some(15),
        },
        DensityBrush {
            id: 3,
            center_mm: [-2_500, 7, 0],
            inner_radius_mm: 2_300,
            outer_radius_mm: 2_300,
            opacity: u16::MAX,
            density: DensityBrushOp::Replace(0),
            species: Some(3),
        },
    ];
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 40_000,
            base_species: 7,
            noise: None,
        },
        brushes,
        ScatterFieldBuildBudget {
            max_brushes: BRUSHES,
            max_brush_bytes: BRUSH_BYTES_LIMIT,
        },
    )
    .expect("nonempty bounded author field")
}

// Independent scalar PCG specification: widened arithmetic modulo 2^64,
// rather than calling production pcg_hash_u32 or its private placement rank.
fn oracle_pcg(input: u64) -> u32 {
    let mask = u128::from(u64::MAX);
    let initial = (u128::from(input) + 0x853c_49e6_748f_ea9b) & mask;
    let state = ((initial * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & mask) as u64;
    let word = (((state >> 18) ^ state) >> 27) as u32;
    word.rotate_right((state >> 59) as u32)
}

fn oracle_mix(input: u64) -> u64 {
    u64::from(oracle_pcg(input)) * (1_u64 << 32)
        + u64::from(oracle_pcg(input ^ 0x9e37_79b9_7f4a_7c15))
}

fn oracle_priority(key: ScatterCandidateKey) -> u64 {
    [
        u64::from(key.kernel_version),
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.z as u64,
    ]
    .into_iter()
    .fold(oracle_mix(0x706c_6163_655f_787a ^ 1), |rank, word| {
        oracle_mix(rank ^ word)
    })
}

fn oracle_field_rank(seed: u64, key: ScatterCandidateKey) -> u32 {
    let hash = [
        u64::from(key.kernel_version),
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.y as u64,
        key.grid.z as u64,
    ]
    .into_iter()
    .fold(oracle_mix(seed ^ 0x6669_656c_645f_726b), |rank, word| {
        oracle_mix(rank ^ word)
    });
    oracle_pcg(hash)
}

fn oracle_attributes(field: &ScatterDensityField, position: [i64; 3]) -> ScatterFieldSample {
    assert_eq!(field.config().noise, None);
    let mut sample = ScatterFieldSample {
        density: field.config().base_density,
        species: field.config().base_species,
    };
    for brush in field.brushes() {
        assert_eq!(brush.opacity, u16::MAX);
        assert_eq!(brush.inner_radius_mm, brush.outer_radius_mm);
        let squared_distance: i128 = position
            .into_iter()
            .zip(brush.center_mm)
            .map(|(a, b)| (i128::from(a) - i128::from(b)).pow(2))
            .sum();
        if squared_distance < i128::from(brush.outer_radius_mm).pow(2) {
            sample.density = match brush.density {
                DensityBrushOp::Replace(value) => value,
                DensityBrushOp::Add(delta) => {
                    (i32::from(sample.density) + delta).clamp(0, 65_535) as u16
                }
            };
            if let Some(species) = brush.species {
                sample.species = species;
            }
        }
    }
    assert_eq!(
        field.sample(position),
        sample,
        "independent hard-sphere attributes"
    );
    sample
}

#[derive(Clone, Copy)]
struct RawCandidate {
    key: HeightfieldPlacementKey,
    xz: [i64; 2],
    priority: u64,
}

fn raw(config: HeightfieldPlacementConfig, x: i64, z: i64) -> RawCandidate {
    let source = ScatterCandidateKey {
        kernel_version: SCATTER_KERNEL_VERSION,
        seed: config.seed,
        version: config.version,
        salt: config.salt,
        grid_spacing_mm: config.grid_spacing_mm,
        grid: SectorCoord::new(x, 0, z),
    };
    let point = source
        .rebuild()
        .expect("real raw source key reconstruction");
    RawCandidate {
        key: HeightfieldPlacementKey {
            placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
            source,
        },
        xz: [point.position_mm[0], point.position_mm[2]],
        priority: oracle_priority(source),
    }
}

fn distance_squared(a: [i64; 2], b: [i64; 2]) -> i128 {
    (i128::from(a[0]) - i128::from(b[0])).pow(2) + (i128::from(a[1]) - i128::from(b[1])).pow(2)
}

fn oracle_point(
    candidate: RawCandidate,
    surface: &PlanarHeightfieldSurface,
    offset: i64,
) -> HeightfieldPlacementPoint {
    let embedding = surface.embedding();
    let spacing = i128::from(embedding.sample_spacing_mm);
    let local = [
        i128::from(candidate.xz[0]) - i128::from(embedding.sample_origin_mm[0]),
        i128::from(candidate.xz[1]) - i128::from(embedding.sample_origin_mm[2]),
    ];
    let [x, z] = local.map(|v| (v / spacing) as u32);
    let [rx, rz] = local.map(|v| v % spacing);
    let height = |x, z| surface.tile().height_mm([x, z]).expect("real baked corner");
    // Independent barycentric interpolation and cross-product face normal.
    let (vertices, weights) = if rx + rz <= spacing {
        (
            [
                [0, height(x, z), 0],
                [0, height(x, z + 1), spacing],
                [spacing, height(x + 1, z), 0],
            ],
            [spacing - rx - rz, rz, rx],
        )
    } else {
        (
            [
                [spacing, height(x + 1, z), 0],
                [0, height(x, z + 1), spacing],
                [spacing, height(x + 1, z + 1), spacing],
            ],
            [spacing - rz, spacing - rx, rx + rz - spacing],
        )
    };
    let numerator = i128::from(embedding.sample_origin_mm[1]) * spacing
        + vertices
            .into_iter()
            .zip(weights)
            .map(|(v, w)| v[1] * w)
            .sum::<i128>();
    let edge1: [i128; 3] = core::array::from_fn(|axis| vertices[1][axis] - vertices[0][axis]);
    let edge2: [i128; 3] = core::array::from_fn(|axis| vertices[2][axis] - vertices[0][axis]);
    let normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ]
    .map(|component| component / spacing);
    // Obtain the real opaque public sample, then verify every exact component.
    // There are no synthetic samples, float approximations or fake source Ys.
    let sample = surface
        .sample_xz_mm(candidate.xz)
        .expect("covered real surface");
    assert_eq!(sample.height_numerator_mm(), numerator);
    assert_eq!(i128::from(sample.height_denominator()), spacing);
    assert_eq!(sample.normal_mm(), normal);
    let offset_numerator = numerator + i128::from(offset) * spacing;
    let magnitude = (2 * offset_numerator.abs() + spacing) / (2 * spacing);
    let y =
        i64::try_from(offset_numerator.signum() * magnitude).expect("bounded integer projection");
    let point = HeightfieldPlacementPoint {
        key: candidate.key,
        position_mm: [candidate.xz[0], y, candidate.xz[1]],
        surface_sample: sample,
    };
    assert_eq!(
        candidate
            .key
            .rebuild(surface, offset)
            .expect("placement key rebuild"),
        point
    );
    point
}

#[derive(Debug, Default)]
struct Rejections {
    distance: u64,
    terrain: u64,
    field: u64,
    first_blocker_outside_request: u64,
}

fn reference(
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: &ScatterDensityField,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldPlacementFilters,
) -> (HeightfieldPlacementOutput, Rejections) {
    // Pin the independently specified version-1 priority and field algorithms.
    assert_eq!(HEIGHTFIELD_PLACEMENT_VERSION, 1);
    assert_eq!(SCATTER_KERNEL_VERSION, 1);
    assert_eq!(SCATTER_FIELD_VERSION, 1);
    assert_eq!(HEIGHTFIELD_SURFACE_VERSION, 1);
    let halo = i64::from(config.min_distance_mm.div_ceil(config.grid_spacing_mm));
    let side = halo * 2 + 1;
    let radius_squared = i128::from(config.min_distance_mm).pow(2);
    let candidates: Vec<_> = (GRID_MIN - halo..GRID_MAX + halo)
        .flat_map(|x| (GRID_MIN - halo..GRID_MAX + halo).map(move |z| raw(config, x, z)))
        .collect();
    let mut expected = HeightfieldPlacementOutput {
        points: Vec::new(),
        attributes: Vec::new(),
        cache_key: HeightfieldPlacementCacheKey {
            placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
            source_kernel_version: SCATTER_KERNEL_VERSION,
            surface_version: HEIGHTFIELD_SURFACE_VERSION,
            field_version: SCATTER_FIELD_VERSION,
            config,
            region,
            filters,
            surface: surface.clone(),
            field: field.clone(),
        },
        candidate_evaluations: PRIMARY as u64,
        surface_sample_reads: 0,
        field_work: 0,
    };
    let mut rejected = Rejections::default();
    for x in GRID_MIN..GRID_MAX {
        for z in GRID_MIN..GRID_MAX {
            let current = raw(config, x, z);
            assert!(
                region.contains(current.xz),
                "all 100x100 aligned primaries are owned"
            );
            // All-pairs membership oracle, not production's regenerating halo
            // walk: compare every raw candidate (including outside ownership),
            // even candidates later rejected by terrain, field or other ranks.
            let blocker = if config.min_distance_mm == 0 {
                None
            } else {
                candidates
                    .iter()
                    .filter(|other| {
                        distance_squared(current.xz, other.xz) < radius_squared
                            && (other.priority, other.key) < (current.priority, current.key)
                    })
                    .min_by_key(|other| other.key)
            };
            if let Some(other) = blocker {
                let dx = other.key.source.grid.x - x;
                let dz = other.key.source.grid.z - z;
                assert!(dx.abs() <= halo && dz.abs() <= halo);
                // Derive the first blocker's probe ordinal algebraically;
                // production skips the centre and stops at this lexicographic
                // neighbour. No duplicate production halo traversal is used.
                let ordinal = (dx + halo) * side + dz + halo;
                let centre = halo * side + halo;
                expected.candidate_evaluations +=
                    (ordinal + 1 - i64::from(ordinal > centre)) as u64;
                rejected.distance += 1;
                rejected.first_blocker_outside_request += u64::from(!region.contains(other.xz));
                continue;
            }
            expected.candidate_evaluations += (side * side - 1) as u64;
            let point = oracle_point(current, surface, config.vertical_offset_mm);
            expected.surface_sample_reads += 4;
            let sample = point.surface_sample;
            let numerator = sample.height_numerator_mm();
            let denominator = i128::from(sample.height_denominator());
            let [nx, ny, nz] = sample.normal_mm();
            let slope_ok = (nx * nx + nz * nz) * i128::from(filters.max_slope_run).pow(2)
                <= ny * ny * i128::from(filters.max_slope_rise).pow(2);
            if numerator < i128::from(filters.surface_height_mm[0]) * denominator
                || numerator > i128::from(filters.surface_height_mm[1]) * denominator
                || !slope_ok
            {
                rejected.terrain += 1;
                continue;
            }
            let attributes = oracle_attributes(field, point.position_mm);
            expected.field_work += (1 + BRUSHES) as u64;
            let rank = oracle_field_rank(field.config().seed, current.key.source);
            assert_eq!(field.candidate_rank(current.key.source), rank);
            let accepted = attributes.density == u16::MAX
                || (attributes.density != 0 && rank < u32::from(attributes.density) * 65_537);
            if !accepted {
                rejected.field += 1;
                continue;
            }
            expected.points.push(point);
            expected.attributes.push(attributes);
        }
    }
    assert!(!expected.points.is_empty());
    assert!(rejected.terrain > 0 && rejected.field > 0);
    assert_eq!(
        expected.points.len() as u64 + rejected.distance + rejected.terrain + rejected.field,
        PRIMARY as u64
    );
    if config.min_distance_mm > 0 {
        assert!(rejected.distance > 0);
        assert!(
            rejected.first_blocker_outside_request > 0,
            "halo outside the request must affect membership"
        );
    }
    (expected, rejected)
}

fn verify(
    output: &HeightfieldPlacementOutput,
    expected: &HeightfieldPlacementOutput,
    surface: &PlanarHeightfieldSurface,
    field: &ScatterDensityField,
    config: HeightfieldPlacementConfig,
) {
    assert_eq!(
        output, expected,
        "all ordered keys/positions/exact samples/attributes/counters/cache"
    );
    assert_eq!(output.points.len(), output.attributes.len());
    for (index, point) in output.points.iter().enumerate() {
        assert_eq!(point.key.source.grid.y, 0);
        assert_eq!(
            point
                .key
                .rebuild(surface, config.vertical_offset_mm)
                .expect("every output key rebuilds"),
            *point
        );
        assert_eq!(field.sample(point.position_mm), output.attributes[index]);
        if config.min_distance_mm != 0 {
            // Unconditional integer all-pairs horizontal-spacing oracle, not
            // 3D distance, a grid approximation or a quality/distribution test.
            for other in &output.points[..index] {
                assert!(
                    distance_squared(
                        [point.position_mm[0], point.position_mm[2]],
                        [other.position_mm[0], other.position_mm[2]],
                    ) >= i128::from(config.min_distance_mm).pow(2)
                );
            }
        }
    }
}

fn measure(
    run: impl FnOnce() -> Result<HeightfieldPlacementOutput, HeightfieldPlacementError>,
) -> (HeightfieldPlacementOutput, f64) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let result = black_box(run());
    let micros = start.elapsed().as_secs_f64() * 1e6;
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    // Result handling, all oracles, buffer checks and drops are outside timing.
    let output = result.expect("bounded production placement succeeds");
    assert_eq!(
        allocations, 2,
        "one geometry and one attribute Vec allocation"
    );
    (output, micros)
}

fn buffer_stats(output: &HeightfieldPlacementOutput) -> [usize; 4] {
    [
        output.points.capacity(),
        output.points.capacity() * size_of::<HeightfieldPlacementPoint>(),
        output.attributes.capacity(),
        output.attributes.capacity() * size_of::<ScatterFieldSample>(),
    ]
}

fn percentiles(mut times: [f64; SAMPLES]) -> [f64; 3] {
    times.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| times[(SAMPLES * p).div_ceil(100) - 1])
}

fn benchmark(surface: &PlanarHeightfieldSurface, field: &ScatterDensityField, radius: u32) {
    let config = HeightfieldPlacementConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
        grid_spacing_mm: 100,
        min_distance_mm: radius,
        vertical_offset_mm: 7,
    };
    let region = HeightfieldPlacementRegion::new([-5_000; 2], [5_000; 2])
        .expect("100x100 aligned XZ primary cells; no Y slab");
    let filters = HeightfieldPlacementFilters {
        surface_height_mm: [-20, 20],
        max_slope_rise: 1,
        max_slope_run: 100,
    };
    assert_eq!(field.brushes().len(), BRUSHES);
    assert_eq!(field.work_per_sample(), (1 + BRUSHES) as u64);
    let halo = u64::from(radius.div_ceil(config.grid_spacing_mm));
    let work = HeightfieldPlacementWork {
        primary_candidates: PRIMARY as u64,
        max_candidate_evaluations: PRIMARY as u64 * (2 * halo + 1).pow(2),
        max_sample_reads: PRIMARY as u64 * 4,
        max_field_work: PRIMARY as u64 * (1 + BRUSHES) as u64,
    };
    assert_eq!(
        estimate_heightfield_placement_work(config, region, field).expect("finite work estimate"),
        work
    );
    let budget = HeightfieldPlacementBudget {
        max_candidate_evaluations: work.max_candidate_evaluations,
        max_sample_reads: 40_000,
        max_field_work: 40_000,
        max_output_points: PRIMARY,
        max_output_bytes: OUTPUT_BYTES_LIMIT,
    };
    // No production call supplies the expected result. All fixture creation,
    // raw source/priority/geometry/author oracles precede the warmup loop.
    let (expected, rejected) = reference(config, region, field, surface, filters);
    assert!(expected.candidate_evaluations <= budget.max_candidate_evaluations);
    assert!(expected.surface_sample_reads <= budget.max_sample_reads);
    assert!(expected.field_work <= budget.max_field_work);
    let mut times = [0.0; SAMPLES];
    let mut buffers = None;
    for iteration in 0..WARMUP + SAMPLES {
        let (output, micros) = measure(|| {
            scatter_on_heightfield(
                black_box(config),
                black_box(region),
                black_box(field),
                black_box(surface),
                black_box(filters),
                black_box(budget),
            )
        });
        verify(&output, &expected, surface, field, config);
        let actual_buffers = buffer_stats(&output);
        assert!(actual_buffers[0] >= PRIMARY && actual_buffers[2] >= PRIMARY);
        assert!(actual_buffers[1] + actual_buffers[3] <= budget.max_output_bytes);
        assert_eq!(
            *buffers.get_or_insert(actual_buffers),
            actual_buffers,
            "repeatable actual capacity payload"
        );
        if iteration >= WARMUP {
            times[iteration - WARMUP] = micros;
        }
    }
    let [point_capacity, point_bytes, attr_capacity, attr_bytes] =
        buffers.expect("40 verified calls");
    let [p50, p95, p99] = percentiles(times);
    println!("# config={config:?} region={region:?} filters={filters:?}");
    println!("# independent_preflight={work:?} budget={budget:?} rejections={rejected:?}");
    println!("painted radius{radius}: points={} attrs={} points_capacity={point_capacity} points_capacity_bytes={point_bytes} attrs_capacity={attr_capacity} attrs_capacity_bytes={attr_bytes} total_output_capacity_bytes={} candidate_evaluations={} surface_reads={} actual_field_work={} allocations=2 us_P50/P95/P99={p50:.3}/{p95:.3}/{p99:.3}",
        expected.points.len(), expected.attributes.len(), point_bytes + attr_bytes,
        expected.candidate_evaluations, expected.surface_sample_reads, expected.field_work);
}

fn main() {
    if cfg!(debug_assertions) {
        panic!("run this benchmark with a release/bench profile");
    }
    println!("# CPU host wall clock only; release; per case {WARMUP} warmups + {SAMPLES} samples; nearest-rank P50/P95/P99 (ranks16/31/32); no timing pass/fail threshold");
    println!("# Timed: black_box complete production scatter_on_heightfield call including two output allocations; excludes fixture/oracle construction, Result handling, all validation, printing and drops");
    println!("# Real bake: PositiveY root, 65x65 i16 samples, scale1mm, ramp=u_q16/256-v_q16/256; embedding origin[-32000,0,-32000] spacing1000mm; shared height payload8450B; XZ100x100 candidates at100mm; radius0 and150mm (halo0/2)");
    println!("# Membership oracle: raw source key reconstruction + independent widened PCG priority/field ranks; positive-radius all-pairs over104x104 raw candidates including unrequested halo; integer barycentric samples/normals/rounding, hard-sphere attributes, exact early-exit counters, full cache, every point key rebuilt, every output pair checked in XZ");
    println!("# Logical admission probes are not CPU instruction/time budgets; output bytes charge actual Vec capacities, excluding allocator metadata/shared tile+field/stack/RSS; no GPU, Poisson, blue-noise, maximal-fill or cross-machine performance claim");
    let surface = surface();
    let field = field();
    println!("# field_config={:?} canonical_brushes={:?} brush_build_limit={BRUSH_BYTES_LIMIT}B; full-opacity hard spheres, reverse input IDs24/15/3, noise=None", field.config(), field.brushes());
    benchmark(&surface, &field, 0);
    benchmark(&surface, &field, 150);
}
