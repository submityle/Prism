//! Bounded CPU heightfield-query and full scatter-filter measurements.
#![expect(clippy::print_stdout, reason = "benchmark measurements")]
#![expect(unsafe_code, reason = "forwarding allocation counter")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::mem::size_of;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

use prism_world_residency::{
    estimate_scatter_work, scatter_with_density_field, DensityNoise, FieldScatterBudget,
    ScatterBudget, ScatterConfig, ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig,
    ScatterRegion,
};
use prism_world_terrain::{
    scatter_with_heightfield, BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldScatterBudget, HeightfieldScatterFilters, HeightfieldSpec, HeightfieldTile,
    PlanarHeightfieldEmbedding, PlanarHeightfieldSurface, TileKey,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const QUERIES: usize = 100_000;
const PRIMARY: usize = 10_000;
const OUTPUT_LIMIT: usize = 16 * 1024 * 1024;
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

fn percentiles(mut times: [f64; SAMPLES]) -> [f64; 3] {
    times.sort_by(f64::total_cmp);
    [50_usize, 95, 99].map(|p| times[(SAMPLES * p).div_ceil(100) - 1])
}

fn main() {
    let tile = Arc::new(
        HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(65, 1).unwrap(),
            &Ramp,
            BakeLimits {
                max_samples: 65 * 65,
                max_bytes: 2 * 65 * 65,
            },
        )
        .unwrap(),
    );
    let surface = PlanarHeightfieldSurface::new(
        Arc::clone(&tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-32_000, 0, -32_000],
            sample_spacing_mm: 1_000,
        },
    )
    .unwrap();
    // Fixed negative/positive coordinates crossing cells and both triangles.
    let positions: Vec<_> = (0..QUERIES as i64)
        .map(|i| {
            [
                (i * 97).rem_euclid(64_000) - 32_000,
                (i * 193).rem_euclid(64_000) - 32_000,
            ]
        })
        .collect();
    let expected: Vec<_> = positions
        .iter()
        .map(|&p| surface.sample_xz_mm(p).unwrap())
        .collect();
    let query = || {
        let mut digest = 0_u64;
        for (&position, expected) in positions.iter().zip(&expected) {
            let result = black_box(&surface)
                .sample_xz_mm(black_box(position))
                .unwrap();
            // Verification is intentionally included in this query batch's
            // time; no claim of isolated interpolation instruction time.
            assert_eq!(result, *expected);
            digest = digest.rotate_left(7) ^ result.height_numerator_mm() as u64;
        }
        black_box(digest)
    };
    let expected_digest = query();
    let mut times = [0.0; SAMPLES];
    for i in 0..WARMUP + SAMPLES {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        let start = Instant::now();
        let digest = query();
        let us = start.elapsed().as_secs_f64() * 1e6;
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        assert_eq!(digest, expected_digest);
        assert_eq!(allocations, 0, "surface queries must not allocate");
        if i >= WARMUP {
            times[i - WARMUP] = us;
        }
    }
    let p = percentiles(times);
    println!("# CPU host wall clock, release, {WARMUP} warmup/{SAMPLES} samples, nearest-rank P50/P95/P99; not GPU time or product frame time");
    println!("query+exact-result-check: queries={QUERIES} digest={expected_digest:016x} allocations=0 us={:.3}/{:.3}/{:.3}", p[0], p[1], p[2]);

    let config = ScatterConfig {
        seed: 0x0123_4567_89ab_cdef,
        version: 1,
        salt: 22,
        grid_spacing_mm: 100,
        min_distance_mm: 0,
    };
    let region = ScatterRegion::new([-1_000, -1_200, -1_000], [1_000, 1_300, 1_000]).unwrap();
    assert_eq!(
        estimate_scatter_work(config, region)
            .unwrap()
            .primary_candidates,
        PRIMARY as u64,
        "fixture bounds must align to exactly 20x25x20 global source cells"
    );
    let source_budget = ScatterBudget {
        max_candidate_evaluations: PRIMARY as u64,
        max_output_points: PRIMARY,
        max_output_bytes: OUTPUT_LIMIT,
    };
    let field = ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: 40_000,
            base_species: 7,
            noise: Some(DensityNoise {
                cell_size_mm: 311,
                amplitude: 10_000,
                warp_cell_size_mm: 127,
                warp_amplitude_mm: 201,
            }),
        },
        Vec::new(),
        ScatterFieldBuildBudget {
            max_brushes: 0,
            max_brush_bytes: 0,
        },
    )
    .unwrap();
    let field_budget = FieldScatterBudget {
        max_field_work: PRIMARY as u64 * field.work_per_sample(),
        max_output_bytes: OUTPUT_LIMIT,
    };
    let filters = HeightfieldScatterFilters {
        surface_height_mm: [-100, 100],
        clearance_mm: [-250, 250],
        max_slope_rise: 1,
        max_slope_run: 10,
    };
    let source =
        scatter_with_density_field(config, region, source_budget, &field, field_budget).unwrap();
    // Independent bounded-fixture selection oracle: exact scalar rational
    // comparisons, and direct squaring is safe for this small i16/scale fixture.
    let expected: Vec<_> = source
        .points
        .iter()
        .copied()
        .zip(source.attributes.iter().copied())
        .filter(|(point, _)| {
            let s = surface
                .sample_xz_mm([point.position_mm[0], point.position_mm[2]])
                .unwrap();
            let h = s.height_numerator_mm();
            let d = i128::from(s.height_denominator());
            let c = i128::from(point.position_mm[1]) * d - h;
            let n = s.normal_mm();
            h >= -100 * d
                && h <= 100 * d
                && c >= -250 * d
                && c <= 250 * d
                && (n[0] * n[0] + n[2] * n[2]) * 100 <= n[1] * n[1]
        })
        .collect();
    assert!(!expected.is_empty() && expected.len() < source.points.len());
    let expected_reads = source.points.len() as u64 * 4;
    let mut expected_bytes = None;
    let mut times = [0.0; SAMPLES];
    for i in 0..WARMUP + SAMPLES {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        let start = Instant::now();
        let output = scatter_with_heightfield(
            config,
            region,
            source_budget,
            &field,
            field_budget,
            &surface,
            filters,
            HeightfieldScatterBudget {
                max_sample_reads: PRIMARY as u64 * 4,
            },
        )
        .unwrap();
        black_box(&output);
        let us = start.elapsed().as_secs_f64() * 1e6;
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        assert_eq!(
            allocations, 2,
            "only the reused source points/attributes buffers"
        );
        assert_eq!(output.surface_sample_reads, expected_reads);
        assert_eq!(output.points.len(), expected.len());
        for ((point, attr), expected) in output.points.iter().zip(&output.attributes).zip(&expected)
        {
            assert_eq!(
                (*point, *attr),
                *expected,
                "full result validation outside timer"
            );
        }
        let bytes = output.points.capacity() * size_of::<prism_world_residency::ScatterPoint>()
            + output.attributes.capacity() * size_of::<prism_world_residency::ScatterFieldSample>();
        if let Some(expected_bytes) = expected_bytes {
            assert_eq!(bytes, expected_bytes);
        } else {
            expected_bytes = Some(bytes);
        }
        if i >= WARMUP {
            times[i - WARMUP] = us;
        }
    }
    let p = percentiles(times);
    println!("# actual baked 65x65 ramp, 1000mm spacing, origin[-32000,0,-32000], 8450B shared height payload; 20x25x20 source voxels at100mm, radius0; density40000+noise/warp, no brushes");
    println!("# height[-100,100]mm clearance[-250,250]mm slope<=1/10; complete result validated after every timed iteration; build/drop/query-oracle preparation excluded from full-scatter timer");
    println!("source+field+surface: primary={PRIMARY} field_survivors={} final_survivors={} surface_reads={expected_reads} output_capacity_bytes={} allocations=2 us={:.3}/{:.3}/{:.3}", source.points.len(), expected.len(), expected_bytes.unwrap(), p[0], p[1], p[2]);
}
