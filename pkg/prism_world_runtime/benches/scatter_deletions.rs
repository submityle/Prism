//! Reproducible host-wallclock scatter deletion and separate codec benchmark.
//!
//! Run with `cargo bench -p prism_world_runtime --bench scatter_deletions
//! --offline -j1 --target-dir /private/tmp/prism-world-sector-build-EFoWO8`.
//! Source generation, buffer/key cloning, set construction, validation and drops
//! are outside filtering timing. Codec timing includes its own allocations and
//! key validation, but not source filtering or result validation/destruction.
//! Vec payload capacity and wire length are not process-memory measurements.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::hint::black_box;
use std::mem::size_of;
use std::time::{Duration, Instant};

use prism_world_residency::{
    scatter, DeletionFilteredScatter, ScatterBudget, ScatterCandidateKey, ScatterConfig,
    ScatterDeletionBudget, ScatterDeletionSet, ScatterOutput, ScatterPoint, ScatterRegion,
};
use prism_world_runtime::{
    decode_scatter_deletions, encode_scatter_deletions, ScatterDeletionCodecBudget,
    SCATTER_DELETION_HEADER_BYTES, SCATTER_DELETION_KEY_BYTES,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const DELETE_EVERY: usize = 4;
const SEED: u64 = 0x0123_4567_89ab_cdef;

struct Fixture {
    source: ScatterOutput,
    expected: Vec<ScatterPoint>,
    deleted_keys: BTreeSet<ScatterCandidateKey>,
    deletions: ScatterDeletionSet,
    codec_budget: ScatterDeletionCodecBudget,
    wire: Vec<u8>,
}

impl Fixture {
    fn new(count: usize) -> Self {
        // An aligned 20 * 20 * Z grid gives exactly N candidates, with no
        // thinning or request-boundary rejection. Every fourth source row dies.
        assert_eq!(count % 400, 0);
        let config = ScatterConfig {
            seed: SEED,
            version: 1,
            salt: 22,
            grid_spacing_mm: 50,
            min_distance_mm: 0,
        };
        let region = ScatterRegion {
            min_mm: [0, 0, 0],
            max_mm: [1_000, 1_000, i64::try_from(count / 400).unwrap() * 50],
        };
        let source = scatter(
            config,
            region,
            ScatterBudget {
                max_candidate_evaluations: count as u64,
                max_output_points: count,
                max_output_bytes: count * size_of::<ScatterPoint>(),
            },
        )
        .expect("aligned bounded source scatter");
        assert_eq!(source.points.len(), count);
        assert_eq!(source.candidate_evaluations, count as u64);

        let d = count / DELETE_EVERY;
        let mut keys = Vec::with_capacity(d);
        keys.extend(source.points.iter().step_by(DELETE_EVERY).map(|p| p.key));
        assert_eq!(keys.len(), d);
        // Independent membership and ordinal oracles; neither invokes the
        // filtering implementation being benchmarked to compute expectations.
        let deleted_keys = keys.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(deleted_keys.len(), d);
        let expected = source
            .points
            .iter()
            .enumerate()
            .filter(|(index, _)| index % DELETE_EVERY != 0)
            .map(|(_, point)| *point)
            .collect();
        let budget = ScatterDeletionBudget {
            max_keys: d,
            max_key_bytes: keys.capacity() * size_of::<ScatterCandidateKey>(),
        };
        let deletions = ScatterDeletionSet::new(keys, budget).expect("bounded deletion keys");
        assert!(deletions
            .keys()
            .iter()
            .copied()
            .eq(deleted_keys.iter().copied()));
        let max_encoded_bytes = SCATTER_DELETION_HEADER_BYTES + d * SCATTER_DELETION_KEY_BYTES;
        let codec_budget = ScatterDeletionCodecBudget {
            max_encoded_bytes,
            deletions: budget,
        };
        let wire = encode_scatter_deletions(&deletions, max_encoded_bytes)
            .expect("canonical reference encoding");
        assert_eq!(wire.len(), max_encoded_bytes);
        Self {
            source,
            expected,
            deleted_keys,
            deletions,
            codec_budget,
            wire,
        }
    }

    fn source_buffer(&self) -> ScatterOutput {
        ScatterOutput {
            points: self.source.points.clone(),
            cache_key: self.source.cache_key,
            candidate_evaluations: self.source.candidate_evaluations,
        }
    }

    fn validate(&self, output: &DeletionFilteredScatter) {
        assert_eq!(output.removed, self.deleted_keys.len());
        assert_eq!(
            output.source.points.len() + output.removed,
            self.source.points.len()
        );
        assert_eq!(output.source.cache_key, self.source.cache_key);
        assert_eq!(
            output.source.candidate_evaluations,
            self.source.candidate_evaluations
        );
        assert_eq!(output.deletions, self.deletions);
        // Full point equality checks every structural key and position, and
        // proves survivor order, completeness and absence of duplicates.
        assert_eq!(output.source.points.as_slice(), self.expected.as_slice());
        assert!(output
            .source
            .points
            .iter()
            .all(|p| !self.deleted_keys.contains(&p.key)));
    }

    fn run(&self) {
        let mut filtering = [Duration::ZERO; SAMPLES];
        let mut output_capacity_bytes = None;
        for iteration in 0..WARMUP + SAMPLES {
            // Clone the already-generated source before starting the clock.
            // One buffer per iteration avoids holding 72 copies concurrently.
            let buffer = black_box(self.source_buffer());
            let pointer = buffer.points.as_ptr();
            let capacity = buffer.points.capacity();
            let deletions = black_box(&self.deletions);
            let start = Instant::now();
            let output = black_box(deletions.filter_source(buffer));
            let elapsed = start.elapsed();

            self.validate(&output);
            assert_eq!(output.source.points.as_ptr(), pointer, "in-place filter");
            assert_eq!(output.source.points.capacity(), capacity);
            let bytes = capacity * size_of::<ScatterPoint>();
            assert_eq!(*output_capacity_bytes.get_or_insert(bytes), bytes);
            if iteration >= WARMUP {
                filtering[iteration - WARMUP] = elapsed;
            }
            // Output destruction is outside the timed interval.
        }

        let mut codec = [Duration::ZERO; SAMPLES];
        let mut decoded_key_bytes = None;
        for iteration in 0..WARMUP + SAMPLES {
            let deletions = black_box(&self.deletions);
            let budget = black_box(self.codec_budget);
            let start = Instant::now();
            let bytes = encode_scatter_deletions(deletions, budget.max_encoded_bytes)
                .expect("encode benchmark deletions");
            let decoded = decode_scatter_deletions(black_box(&bytes), budget)
                .expect("decode benchmark deletions");
            black_box((&bytes, &decoded));
            let elapsed = start.elapsed();

            assert_eq!(bytes, self.wire, "canonical bytes must be identical");
            assert_eq!(
                decoded.keys(),
                self.deletions.keys(),
                "exact full-key roundtrip"
            );
            let key_bytes = decoded.allocated_key_bytes();
            assert!(key_bytes <= budget.deletions.max_key_bytes);
            assert_eq!(*decoded_key_bytes.get_or_insert(key_bytes), key_bytes);
            // Each codec result must also reproduce every survivor, not merely
            // roundtrip its byte count. This filtering and cloning are untimed.
            self.validate(&decoded.filter_source(self.source_buffer()));
            if iteration >= WARMUP {
                codec[iteration - WARMUP] = elapsed;
            }
        }

        println!(
            "N={} D={} deleted=25% input_points={} output_points={} removed={} \
             input_point_vec_capacity_bytes={} output_point_vec_capacity_bytes={} \
             key_vec_capacity_bytes={} decoded_key_vec_capacity_bytes={} wirebytes={}",
            self.source.points.len(),
            self.deletions.keys().len(),
            self.source.points.len(),
            self.expected.len(),
            self.deleted_keys.len(),
            self.source.points.capacity() * size_of::<ScatterPoint>(),
            output_capacity_bytes.unwrap(),
            self.deletions.allocated_key_bytes(),
            decoded_key_bytes.unwrap(),
            self.wire.len(),
        );
        report("DeletionSet.filter_source", &mut filtering);
        report("codec.encode+decode", &mut codec);
        println!("  validation=passed for all {WARMUP} warmups + {SAMPLES} samples per operation");
    }
}

fn report(label: &str, samples: &mut [Duration; SAMPLES]) {
    samples.sort_unstable();
    let nearest_rank = |percent: usize| samples[(SAMPLES * percent).div_ceil(100) - 1];
    println!(
        "  {label}: host_wallclock_us nearest_rank P50={:.3} P95={:.3} P99={:.3}",
        nearest_rank(50).as_secs_f64() * 1e6,
        nearest_rank(95).as_secs_f64() * 1e6,
        nearest_rank(99).as_secs_f64() * 1e6,
    );
}

fn main() {
    assert!(
        !black_box(cfg!(debug_assertions)),
        "run this benchmark with cargo bench (release)"
    );
    println!(
        "scatter_deletions release host benchmark: os={} arch={} warmup={WARMUP} samples={SAMPLES} \
         seed={SEED:#018x} version=1 salt=22 spacing_mm=50 min_distance_mm=0 delete=source_index%4==0",
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    println!(
        "filter timing excludes generation, source/key clones, set construction, codec, validation and output drop; \
         codec timing includes encode+decode only (including their allocations/key validation), excluding result drop"
    );
    println!(
        "bytes are Vec capacity payload (excluding headers/allocator metadata) and canonical wire length; \
         no GPU, frame-time, allocation-count or RSS claims; nearest-rank indices=32/61/64 (1-based)"
    );
    for count in [10_000, 100_000] {
        Fixture::new(count).run();
    }
}
