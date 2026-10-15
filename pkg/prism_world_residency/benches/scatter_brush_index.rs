//! CPU brush BVH versus the exact linear oracle; not renderer/frame timings.
#![expect(clippy::print_stdout, reason = "benchmark measurements")]
#![expect(
    unsafe_code,
    reason = "forwarding allocation counter for hot-path validation"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use prism_world_residency::{
    scatter_with_density_field, scatter_with_indexed_density_field, DensityBrush, DensityBrushOp,
    DensityNoise, FieldScatterBudget, FieldScatterOutput, ScatterBrushIndex,
    ScatterBrushIndexBudget, ScatterBudget, ScatterConfig, ScatterDensityField,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterRegion,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;

struct Counter;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwards the caller's valid allocation layout unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer/layout originate from this forwarding allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counter = Counter;

fn field(brush_count: usize, overlap: bool) -> ScatterDensityField {
    let brushes: Vec<_> = (0..brush_count)
        .rev()
        .map(|id| DensityBrush {
            id: id as u64,
            center_mm: [if overlap { 0 } else { id as i64 * 10_000 }, 0, 0],
            inner_radius_mm: 200,
            outer_radius_mm: if overlap { 10_000 } else { 1_000 },
            opacity: 40_000,
            density: match id % 3 {
                0 => DensityBrushOp::Replace((id * 17 % 65536) as u16),
                1 => DensityBrushOp::Add(1000),
                _ => DensityBrushOp::Add(-300),
            },
            species: Some(id as u32),
        })
        .collect();
    let bytes = brushes.capacity() * size_of::<DensityBrush>();
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 32768,
            base_species: 42,
            noise: Some(DensityNoise {
                cell_size_mm: 311,
                amplitude: 24000,
                warp_cell_size_mm: 127,
                warp_amplitude_mm: 201,
            }),
        },
        brushes,
        ScatterFieldBuildBudget {
            max_brushes: brush_count,
            max_brush_bytes: bytes,
        },
    )
    .expect("valid bounded benchmark field")
}

fn digest(samples: impl Iterator<Item = (u16, u32)>) -> u64 {
    samples.fold(0_u64, |value, (density, species)| {
        value.rotate_left(7) ^ u64::from(density) ^ (u64::from(species) << 32)
    })
}

fn measure<T>(run: impl FnOnce() -> T) -> (T, f64) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let result = black_box(run());
    let micros = start.elapsed().as_secs_f64() * 1e6;
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "queries must not allocate"
    );
    (result, micros)
}

fn measure_scatter(run: impl FnOnce() -> FieldScatterOutput) -> (FieldScatterOutput, f64) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let result = black_box(run());
    let micros = start.elapsed().as_secs_f64() * 1e6;
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        2,
        "one points and one attributes allocation"
    );
    (result, micros)
}

fn full_scatter() {
    let field = field(1024, false);
    let index = ScatterBrushIndex::new(
        field.clone(),
        ScatterBrushIndexBudget {
            max_brushes: 1024,
            max_index_bytes: 32 * 1024 * 1024,
        },
    )
    .expect("bounded full scatter index");
    let mut scratch = index
        .new_scratch(1024 * size_of::<usize>())
        .expect("scratch");
    let config = ScatterConfig {
        grid_spacing_mm: 100,
        min_distance_mm: 0,
        seed: 42,
        version: 1,
        salt: 7,
    };
    let region = ScatterRegion {
        min_mm: [0, 0, 0],
        max_mm: [1_000_000, 100, 100],
    };
    let source_budget = ScatterBudget {
        max_output_points: 10_000,
        max_output_bytes: 16 * 1024 * 1024,
        max_candidate_evaluations: 10_000,
    };
    let budget = FieldScatterBudget {
        max_field_work: 10_000 * index.max_sample_work(),
        max_output_bytes: 16 * 1024 * 1024,
    };
    let mut linear_times = [0.0; SAMPLES];
    let mut index_times = [0.0; SAMPLES];
    let mut stats = (0, 0, 0, 0);
    for iteration in 0..WARMUP + SAMPLES {
        let linear_run = || {
            scatter_with_density_field(config, region, source_budget, &field, budget)
                .expect("linear full scatter")
        };
        let indexed_run = || {
            scatter_with_indexed_density_field(
                config,
                region,
                source_budget,
                &index,
                &mut scratch,
                budget,
            )
            .expect("indexed full scatter")
        };
        let ((linear, linear_us), (indexed, indexed_us)) = if iteration % 2 == 0 {
            let linear = measure_scatter(linear_run);
            (linear, measure_scatter(indexed_run))
        } else {
            let indexed = measure_scatter(indexed_run);
            (measure_scatter(linear_run), indexed)
        };
        assert_eq!(indexed.points, linear.points);
        assert_eq!(indexed.attributes, linear.attributes);
        assert_eq!(indexed.cache_key, linear.cache_key);
        assert_eq!(indexed.candidate_evaluations, linear.candidate_evaluations);
        stats = (
            indexed.points.len(),
            indexed.points.capacity() * size_of::<prism_world_residency::ScatterPoint>()
                + indexed.attributes.capacity()
                    * size_of::<prism_world_residency::ScatterFieldSample>(),
            linear.field_work,
            indexed.field_work,
        );
        if iteration >= WARMUP {
            linear_times[iteration - WARMUP] = linear_us;
            index_times[iteration - WARMUP] = indexed_us;
        }
    }
    linear_times.sort_by(f64::total_cmp);
    index_times.sort_by(f64::total_cmp);
    println!("full_scatter B=1024 sparse primary=10000 accepted={} output_capacity_bytes={} allocations=2 linear_work={} indexed_work={} linear_us_P50/P95/P99={:.3}/{:.3}/{:.3} indexed_us_P50/P95/P99={:.3}/{:.3}/{:.3}; excludes drop/oracle/index construction",
        stats.0, stats.1, stats.2, stats.3, linear_times[31], linear_times[60], linear_times[63],
        index_times[31], index_times[60], index_times[63]);
}

fn main() {
    println!("brush BVH CPU release: {WARMUP} warmup/{SAMPLES} samples; full oracle checks outside every measured pair; alternating order; query batch only, no rank/render/GPU timing");
    for brush_count in [32, 1024, 10_000] {
        for overlap in [false, true] {
            let field = field(brush_count, overlap);
            let positions: Vec<_> = (0..if brush_count == 10_000 { 256 } else { 1024 })
                .map(|i| {
                    [
                        if overlap {
                            i as i64 % 257 - 128
                        } else {
                            (i % brush_count) as i64 * 10_000 + i as i64 % 257 - 128
                        },
                        i as i64 % 127 - 63,
                        i as i64 % 61 - 30,
                    ]
                })
                .collect();
            let expected: Vec<_> = positions.iter().map(|&p| field.sample(p)).collect();
            let expected_digest = digest(expected.iter().map(|s| (s.density, s.species)));
            let build_start = Instant::now();
            let index = ScatterBrushIndex::new(
                field.clone(),
                ScatterBrushIndexBudget {
                    max_brushes: brush_count,
                    max_index_bytes: 32 * 1024 * 1024,
                },
            )
            .expect("bounded index build");
            let build_us = build_start.elapsed().as_secs_f64() * 1e6;
            let mut scratch = index
                .new_scratch(brush_count * size_of::<usize>())
                .expect("bounded scratch");
            let mut linear_times = [0.0; SAMPLES];
            let mut index_times = [0.0; SAMPLES];
            let mut probes = (0, 0, 0);
            for iteration in 0..WARMUP + SAMPLES {
                let linear_run = || {
                    digest(positions.iter().map(|&p| {
                        let s = black_box(&field).sample(black_box(p));
                        (s.density, s.species)
                    }))
                };
                let indexed_run = || {
                    let mut nodes = 0;
                    let mut brushes = 0;
                    let mut work = 0;
                    let value = digest(positions.iter().map(|&p| {
                        let s = black_box(&index)
                            .sample(black_box(p), &mut scratch, index.max_sample_work())
                            .expect("preflighted query");
                        nodes += s.node_tests;
                        brushes += s.brush_tests;
                        work += s.field_work;
                        (s.sample.density, s.sample.species)
                    }));
                    (value, nodes, brushes, work)
                };
                let ((linear_digest, linear_us), ((value, nodes, brushes, work), index_us)) =
                    if iteration % 2 == 0 {
                        let linear = measure(linear_run);
                        (linear, measure(indexed_run))
                    } else {
                        let indexed = measure(indexed_run);
                        (measure(linear_run), indexed)
                    };
                assert_eq!(linear_digest, expected_digest);
                assert_eq!(value, expected_digest);
                // Compare every full sample, not only an aggregate digest.
                for (&position, &expected) in positions.iter().zip(&expected) {
                    assert_eq!(
                        index
                            .sample(position, &mut scratch, index.max_sample_work())
                            .expect("oracle query")
                            .sample,
                        expected
                    );
                }
                probes = (nodes, brushes, work);
                if iteration >= WARMUP {
                    linear_times[iteration - WARMUP] = linear_us;
                    index_times[iteration - WARMUP] = index_us;
                }
            }
            linear_times.sort_by(f64::total_cmp);
            index_times.sort_by(f64::total_cmp);
            println!("B={brush_count} overlap={overlap} N={} build_us={build_us:.3} nodes={} index_bytes={} scratch_bytes={} digest={expected_digest:016x} linear_us_P50/P95/P99={:.3}/{:.3}/{:.3} indexed_us_P50/P95/P99={:.3}/{:.3}/{:.3} node_tests={} brush_tests={} work={} query_allocations=0",
                positions.len(), index.node_count(), index.allocated_bytes(), scratch.allocated_bytes(),
                linear_times[31], linear_times[60], linear_times[63],
                index_times[31], index_times[60], index_times[63], probes.0, probes.1, probes.2);
        }
    }
    full_scatter();
}
