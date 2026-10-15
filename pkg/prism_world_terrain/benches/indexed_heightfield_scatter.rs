//! Release-only CPU wall-clock A/B: linear versus indexed heightfield scatter.
#![expect(clippy::print_stdout, reason = "benchmark measurements")]
#![expect(unsafe_code, reason = "same forwarding counter as heightfield_scatter")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

use prism_world_residency::{
    estimate_scatter_work, DensityBrush, DensityBrushOp, FieldScatterBudget, ScatterBrushIndex,
    ScatterBrushIndexBudget, ScatterBudget, ScatterConfig, ScatterDensityField,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldSample, ScatterPoint, ScatterRegion,
};
use prism_world_terrain::{
    scatter_with_heightfield, scatter_with_indexed_heightfield, BakeLimits, CubeFace, HeightSample,
    HeightSampleValue, HeightSampler, HeightfieldScatterBudget, HeightfieldScatterError,
    HeightfieldScatterFilters, HeightfieldScatterOutput, HeightfieldSpec, HeightfieldTile,
    PlanarHeightfieldEmbedding, PlanarHeightfieldSurface, TileKey,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 32;
const BRUSHES: usize = 1_024;
const PRIMARY: usize = 10_000;
const OUTPUT_LIMIT: usize = 16 * 1024 * 1024;
const BRUSH_BYTES_LIMIT: usize = BRUSHES * size_of::<DensityBrush>();
const INDEX_BYTES_LIMIT: usize = 256 * 1024;
const SCRATCH_BYTES_LIMIT: usize = BRUSHES * size_of::<usize>();

// Reuse the forwarding allocation-counter design in heightfield_scatter.rs.
// The default GlobalAlloc realloc/alloc_zeroed also forward through alloc, so
// unexpected buffer growth is counted. Counts are read before any validation.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
struct Counter;

unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarding the caller's valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout originate from this allocator.
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
        HeightfieldSpec::new(65, 1).expect("65x65 millimetre height encoding"),
        &Ramp,
        BakeLimits {
            max_samples: 65 * 65,
            max_bytes: 2 * 65 * 65,
        },
    )
    .expect("bounded real terrain bake");
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-32_000, 0, -32_000],
            sample_spacing_mm: 1_000,
        },
    )
    .expect("valid ramp embedding")
}

fn field(overlap: bool) -> ScatterDensityField {
    let mut brushes = Vec::with_capacity(BRUSHES);
    // Reverse input IDs exercise canonical, noncommutative composition. Sparse
    // centres form a 16x4x16 3D lattice, not a long one-dimensional source strip.
    for id in (0..BRUSHES).rev() {
        let center_mm = if overlap {
            [0, 0, 0]
        } else {
            [
                -1_500 + (id % 16) as i64 * 200,
                -900 + ((id / 16) % 4) as i64 * 600,
                -1_500 + (id / 64) as i64 * 200,
            ]
        };
        brushes.push(DensityBrush {
            id: id as u64,
            center_mm,
            inner_radius_mm: if overlap { 200 } else { 40 },
            outer_radius_mm: if overlap { 10_000 } else { 80 },
            opacity: 40_000,
            density: match id % 3 {
                0 => DensityBrushOp::Replace((30_000 + id * 17 % 20_000) as u16),
                1 => DensityBrushOp::Add(1_000),
                _ => DensityBrushOp::Add(-300),
            },
            species: Some(id as u32),
        });
    }
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
    .expect("fixed bounded brush recipe")
}

fn percentiles(mut times: [f64; SAMPLES]) -> [f64; 3] {
    times.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| times[(SAMPLES * p).div_ceil(100) - 1])
}

fn measure(
    run: impl FnOnce() -> Result<HeightfieldScatterOutput, HeightfieldScatterError>,
) -> (HeightfieldScatterOutput, f64) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let result = black_box(run());
    let micros = start.elapsed().as_secs_f64() * 1e6;
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    // Everything below, including Result handling and output destruction, is
    // outside the timer. Only the production call and black_box are measured.
    let output = result.expect("bounded scatter must succeed");
    assert_eq!(allocations, 2, "one points and one attributes allocation");
    (output, micros)
}

fn assert_equivalent(linear: &HeightfieldScatterOutput, indexed: &HeightfieldScatterOutput) {
    assert_eq!(
        linear.points, indexed.points,
        "complete points and stable IDs"
    );
    assert_eq!(
        linear.attributes, indexed.attributes,
        "complete aligned attributes"
    );
    assert_eq!(
        linear.cache_key, indexed.cache_key,
        "complete structural cache identity"
    );
    assert_eq!(linear.candidate_evaluations, indexed.candidate_evaluations);
    assert_eq!(linear.surface_sample_reads, indexed.surface_sample_reads);
    // field_work is intentionally NOT normalized: both actual values are
    // checked against their own reference output and printed separately.
}

fn buffer_stats(output: &HeightfieldScatterOutput) -> [usize; 4] {
    [
        output.points.capacity(),
        output.points.capacity() * size_of::<ScatterPoint>(),
        output.attributes.capacity(),
        output.attributes.capacity() * size_of::<ScatterFieldSample>(),
    ]
}

fn report(label: &str, output: &HeightfieldScatterOutput, times: [f64; SAMPLES]) {
    let [point_capacity, point_bytes, attr_capacity, attr_bytes] = buffer_stats(output);
    let [p50, p95, p99] = percentiles(times);
    println!(
        "{label}: points={} attrs={} points_capacity={point_capacity} points_capacity_bytes={point_bytes} attrs_capacity={attr_capacity} attrs_capacity_bytes={attr_bytes} total_output_capacity_bytes={} candidate_evaluations={} actual_field_work={} surface_reads={} allocations=2 us_P50/P95/P99={p50:.3}/{p95:.3}/{p99:.3}",
        output.points.len(), output.attributes.len(), point_bytes + attr_bytes,
        output.candidate_evaluations, output.field_work, output.surface_sample_reads,
    );
}

fn benchmark(surface: &PlanarHeightfieldSurface, overlap: bool) {
    // All terrain/field/index/scratch construction and full reference outputs
    // precede the warmup and measurement loop, and never enter either timer.
    let field = field(overlap);
    let index = ScatterBrushIndex::new(
        field.clone(),
        ScatterBrushIndexBudget {
            max_brushes: BRUSHES,
            max_index_bytes: INDEX_BYTES_LIMIT,
        },
    )
    .expect("bounded brush index");
    let mut scratch = index
        .new_scratch(SCRATCH_BYTES_LIMIT)
        .expect("preallocated scratch");
    let scratch_capacity = scratch.capacity();
    let scratch_bytes = scratch.allocated_bytes();
    let index_bytes = index.allocated_bytes();
    assert!(scratch_capacity >= BRUSHES);
    assert!(scratch_bytes <= SCRATCH_BYTES_LIMIT);
    assert!(index_bytes <= INDEX_BYTES_LIMIT);

    let config = ScatterConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
        grid_spacing_mm: 100,
        min_distance_mm: 0,
    };
    let region = ScatterRegion::new([-1_000, -1_200, -1_000], [1_000, 1_300, 1_000])
        .expect("20x25x20 aligned 3D source");
    let estimate = estimate_scatter_work(config, region).expect("bounded source estimate");
    assert_eq!(estimate.primary_candidates, PRIMARY as u64);
    assert_eq!(estimate.max_candidate_evaluations, PRIMARY as u64);
    let source_budget = ScatterBudget {
        max_candidate_evaluations: PRIMARY as u64,
        max_output_points: PRIMARY,
        max_output_bytes: OUTPUT_LIMIT,
    };
    let linear_preflight = (PRIMARY as u64)
        .checked_mul(field.work_per_sample())
        .expect("bounded linear field work");
    let indexed_preflight = (PRIMARY as u64)
        .checked_mul(index.max_sample_work())
        .expect("bounded indexed field work");
    // Identical finite admission limits for A and B; budgets are not a time cap.
    let field_budget = FieldScatterBudget {
        max_field_work: linear_preflight.max(indexed_preflight),
        max_output_bytes: OUTPUT_LIMIT,
    };
    let filters = HeightfieldScatterFilters {
        surface_height_mm: [-100, 100],
        clearance_mm: [-250, 250],
        max_slope_rise: 1,
        max_slope_run: 10,
    };
    let surface_budget = HeightfieldScatterBudget {
        max_sample_reads: PRIMARY as u64 * 4,
    };

    let expected_linear = scatter_with_heightfield(
        config,
        region,
        source_budget,
        &field,
        field_budget,
        surface,
        filters,
        surface_budget,
    )
    .expect("complete linear reference");
    let expected_indexed = scatter_with_indexed_heightfield(
        config,
        region,
        source_budget,
        &index,
        &mut scratch,
        field_budget,
        surface,
        filters,
        surface_budget,
    )
    .expect("complete indexed reference");
    assert_equivalent(&expected_linear, &expected_indexed);
    assert_eq!(expected_linear.candidate_evaluations, PRIMARY as u64);
    assert_eq!(expected_linear.field_work, linear_preflight);
    assert!(expected_indexed.field_work <= indexed_preflight);
    assert_eq!(expected_linear.surface_sample_reads % 4, 0);
    assert!(expected_linear.surface_sample_reads <= surface_budget.max_sample_reads);
    let field_survivors = expected_linear.surface_sample_reads / 4;
    assert!(
        !expected_linear.points.is_empty(),
        "fixture must exercise output"
    );
    assert!(
        expected_linear.points.len() < field_survivors as usize,
        "fixture must exercise terrain thinning"
    );
    assert_eq!(scratch.capacity(), scratch_capacity);
    assert_eq!(scratch.allocated_bytes(), scratch_bytes);
    let linear_buffers = buffer_stats(&expected_linear);
    let indexed_buffers = buffer_stats(&expected_indexed);

    let mut linear_times = [0.0; SAMPLES];
    let mut indexed_times = [0.0; SAMPLES];
    for iteration in 0..WARMUP + SAMPLES {
        let linear_run = || {
            scatter_with_heightfield(
                black_box(config),
                black_box(region),
                black_box(source_budget),
                black_box(&field),
                black_box(field_budget),
                black_box(surface),
                black_box(filters),
                black_box(surface_budget),
            )
        };
        let indexed_run = || {
            scatter_with_indexed_heightfield(
                black_box(config),
                black_box(region),
                black_box(source_budget),
                black_box(&index),
                black_box(&mut scratch),
                black_box(field_budget),
                black_box(surface),
                black_box(filters),
                black_box(surface_budget),
            )
        };
        // Eight warmup pairs and 32 measured pairs, with 16 AB and 16 BA
        // measured orders. Both outputs remain live until validation finishes.
        let ((linear, linear_us), (indexed, indexed_us)) = if iteration % 2 == 0 {
            let linear = measure(linear_run);
            let indexed = measure(indexed_run);
            (linear, indexed)
        } else {
            let indexed = measure(indexed_run);
            let linear = measure(linear_run);
            (linear, indexed)
        };
        // Full Vec contents (including keys), attrs, cache and every counter
        // are checked after every call, warmups included. Equality to the own
        // reference also checks each path's actual field_work for repeatability.
        assert_eq!(linear, expected_linear, "complete repeatable A output");
        assert_eq!(indexed, expected_indexed, "complete repeatable B output");
        assert_equivalent(&linear, &indexed);
        assert_eq!(buffer_stats(&linear), linear_buffers);
        assert_eq!(buffer_stats(&indexed), indexed_buffers);
        assert!(linear_buffers[1] + linear_buffers[3] <= field_budget.max_output_bytes);
        assert!(indexed_buffers[1] + indexed_buffers[3] <= field_budget.max_output_bytes);
        assert_eq!(
            scratch.capacity(),
            scratch_capacity,
            "scratch must never grow"
        );
        assert_eq!(scratch.allocated_bytes(), scratch_bytes);
        assert_eq!(index.allocated_bytes(), index_bytes);
        if iteration >= WARMUP {
            linear_times[iteration - WARMUP] = linear_us;
            indexed_times[iteration - WARMUP] = indexed_us;
        }
    }

    println!("# workload={} brushes={BRUSHES} primary={PRIMARY} field_survivors={field_survivors} noise=None", if overlap { "all-overlapping" } else { "sparse-3D" });
    println!(
        "# config={config:?} region={region:?} field_config={:?} filters={filters:?}",
        field.config()
    );
    println!("# brush_recipe: descending input IDs 0..1024; opacity40000; ID%3: Replace(30000+ID*17%20000)/Add(1000)/Add(-300); species=ID; sparse centres[-1500+(ID%16)*200,-900+((ID/16)%4)*600,-1500+(ID/64)*200] inner40/outer80mm; overlap centre[0,0,0] inner200/outer10000mm");
    println!("# index_nodes={} index_actual_capacity_bytes={index_bytes} scratch_actual_capacity_indices={scratch_capacity} scratch_actual_capacity_bytes={scratch_bytes} brush_bytes_limit={BRUSH_BYTES_LIMIT} index_bytes_limit={INDEX_BYTES_LIMIT} scratch_bytes_limit={SCRATCH_BYTES_LIMIT}", index.node_count());
    println!("# source_budget={source_budget:?} field_budget={field_budget:?} surface_budget={surface_budget:?} linear_logical_work_preflight={linear_preflight} indexed_logical_work_preflight={indexed_preflight}; logical probes exclude sorting comparisons and are not CPU/time budgets");
    report("A linear", &expected_linear, linear_times);
    report("B indexed", &expected_indexed, indexed_times);
}

fn main() {
    if cfg!(debug_assertions) {
        panic!("run this benchmark with a release/bench profile");
    }
    println!("# CPU host wall clock only; release; {WARMUP} warmup pairs + {SAMPLES} measured pairs; alternating AB/BA; nearest-rank P50/P95/P99 (ranks16/31/32); no performance pass/fail threshold");
    println!("# Timed: complete production scatter call (source+field+terrain), output Vec allocations included; construction, reference generation, full result/capacity checks, printing and drops excluded");
    println!("# Public bake API: PositiveY root, 65x65 i16 heights, scale1mm, ramp=u_q16/256-v_q16/256; embedding origin[-32000,0,-32000] spacing1000mm; 8450B shared height payload; source20x25x20 at100mm, minimum-distance0");
    let surface = surface();
    benchmark(&surface, false);
    benchmark(&surface, true);
}
