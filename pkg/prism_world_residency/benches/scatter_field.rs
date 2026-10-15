//! Bounded integer author-field CPU benchmark; no renderer/frame-time claims.
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
    scatter, scatter_with_density_field, DensityBrush, DensityBrushOp, DensityNoise,
    FieldScatterBudget, ScatterBudget, ScatterConfig, ScatterDensityField, ScatterFieldBuildBudget,
    ScatterFieldConfig, ScatterRegion,
};

const POINTS: usize = 10_000;
const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const OUTPUT_LIMIT: usize = 16 * 1024 * 1024;

struct Counter;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded with the caller's valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout originate from this forwarding allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counter = Counter;

fn field(noise: bool, warp: bool, brush_count: usize) -> ScatterDensityField {
    let brushes: Vec<_> = (0..brush_count)
        .map(|id| DensityBrush {
            id: id as u64,
            center_mm: [500, 500, 50],
            inner_radius_mm: 50 + id as u32,
            outer_radius_mm: 2_000,
            opacity: 8192,
            density: DensityBrushOp::Add(if id % 2 == 0 { 1000 } else { -1000 }),
            species: Some(id as u32 + 1),
        })
        .collect();
    let capacity_bytes = brushes.capacity() * size_of::<DensityBrush>();
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 32768,
            base_species: 7,
            noise: noise.then_some(DensityNoise {
                cell_size_mm: 311,
                amplitude: 24000,
                warp_cell_size_mm: 127,
                warp_amplitude_mm: if warp { 201 } else { 0 },
            }),
        },
        brushes,
        ScatterFieldBuildBudget {
            max_brushes: brush_count,
            max_brush_bytes: capacity_bytes,
        },
    )
    .expect("valid bounded benchmark field")
}

fn main() {
    let config = ScatterConfig {
        grid_spacing_mm: 50,
        min_distance_mm: 0,
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
    };
    let region = ScatterRegion {
        min_mm: [0, 0, 0],
        max_mm: [1000, 1000, 1250],
    };
    let source_budget = ScatterBudget {
        max_output_points: POINTS,
        max_output_bytes: OUTPUT_LIMIT,
        max_candidate_evaluations: POINTS as u64,
    };
    let source = scatter(config, region, source_budget).expect("bounded base scatter");
    assert_eq!(source.points.len(), POINTS);
    println!(
        "scatter_field CPU release: {POINTS} samples, {WARMUP} warmup, {SAMPLES} measurements"
    );
    for (label, field) in [
        ("constant", field(false, false, 0)),
        ("noise", field(true, false, 0)),
        ("warp", field(true, true, 0)),
        ("warp+32-overlapping-brushes", field(true, true, 32)),
    ] {
        let run = || {
            let mut accepted = 0;
            let mut digest = 0_u64;
            for point in &source.points {
                let field = black_box(&field);
                let sample = field.sample(black_box(point.position_mm));
                accepted += usize::from(field.accepts(point.key, sample.density));
                digest = digest.rotate_left(7)
                    ^ u64::from(sample.density)
                    ^ (u64::from(sample.species) << 32);
            }
            black_box((accepted, digest))
        };
        let expected = run();
        let mut times = [0.0; SAMPLES];
        for iteration in 0..WARMUP + SAMPLES {
            ALLOCATIONS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            let result = run();
            let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
            let allocations = ALLOCATIONS.load(Ordering::Relaxed);
            assert_eq!(result, expected, "stable sample/rank output");
            assert_eq!(allocations, 0, "sampling must not allocate");
            if iteration >= WARMUP {
                times[iteration - WARMUP] = elapsed_us;
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{label}: accepted={} digest={:016x}, sample+rank us P50/P95/P99={:.3}/{:.3}/{:.3}, allocations=0 work/sample={}",
            expected.0, expected.1, times[31], times[60], times[63], field.work_per_sample()
        );
    }
    let field = field(true, true, 32);
    let mut times = [0.0; SAMPLES];
    let mut expected = None;
    for iteration in 0..WARMUP + SAMPLES {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        let start = Instant::now();
        let output = scatter_with_density_field(
            config,
            region,
            source_budget,
            &field,
            FieldScatterBudget {
                max_field_work: POINTS as u64 * field.work_per_sample(),
                max_output_bytes: OUTPUT_LIMIT,
            },
        )
        .expect("bounded full field scatter");
        black_box(&output);
        let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        let stats = (
            output.points.len(),
            output.points.capacity() * size_of::<prism_world_residency::ScatterPoint>()
                + output.attributes.capacity()
                    * size_of::<prism_world_residency::ScatterFieldSample>(),
            allocations,
        );
        if let Some(expected) = expected {
            assert_eq!(stats, expected);
        } else {
            expected = Some(stats);
        }
        assert_eq!(allocations, 2, "one points buffer and one attribute buffer");
        if iteration >= WARMUP {
            times[iteration - WARMUP] = elapsed_us;
        }
    }
    times.sort_by(f64::total_cmp);
    let (accepted, bytes, allocations) = expected.expect("nonempty measurement set");
    println!(
        "full scatter+warp+32 brushes: accepted={accepted} output capacity bytes={bytes} allocations={allocations}, us P50/P95/P99={:.3}/{:.3}/{:.3} (excludes output drop)",
        times[31], times[60], times[63]
    );
}
