//! Release host-wallclock placement deletion and separate encode+decode benchmark.
//!
//! Run with `cargo bench -p prism_world_runtime --features placement
//! --bench placement_deletions`. Source generation, cloning, set construction,
//! oracle checks and result destruction are untimed. Codec timing includes its
//! allocations and internal key validation, not external roundtrip/filter checks.
//! Reported Vec payload capacities and wire lengths are not process memory.
#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::hint::black_box;
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

use prism_world_residency::{
    DensityBrush, DensityBrushOp, ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig,
    ScatterFieldSample,
};
use prism_world_runtime::{
    decode_placement_deletions, encode_placement_deletions, PlacementDeletionCodecBudget,
    PLACEMENT_DELETION_HEADER_BYTES, PLACEMENT_DELETION_KEY_BYTES,
};
use prism_world_terrain::{
    estimate_heightfield_placement_work, filter_heightfield_placement, scatter_on_heightfield,
    BakeLimits, CubeFace, DeletionFilteredHeightfieldPlacement, HeightSample, HeightSampleValue,
    HeightSampler, HeightfieldPlacementBudget, HeightfieldPlacementConfig,
    HeightfieldPlacementDeletionBudget, HeightfieldPlacementDeletionSet,
    HeightfieldPlacementFilters, HeightfieldPlacementKey, HeightfieldPlacementOutput,
    HeightfieldPlacementPoint, HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile,
    PlanarHeightfieldEmbedding, PlanarHeightfieldSurface, TileKey,
};

const WARMUP: usize = 8;
const SAMPLES: usize = 64;
const SEED: u64 = 0x0123_4567_89ab_cdef;
const GRID_COLUMNS: usize = 100;
const SPACING_MM: u32 = 50;

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
    .expect("bounded real nonflat heightfield bake");
    assert_ne!(tile.height_mm([0, 0]), tile.height_mm([64, 0]));
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [0, 0, 0],
            sample_spacing_mm: 1_000,
        },
    )
    .expect("64m square planar surface covers both aligned requests")
}

fn field() -> ScatterDensityField {
    // Full density everywhere guarantees exactly one output per owned cell.
    // A real species-only hard brush makes the attribute column nonconstant.
    let brushes = vec![DensityBrush {
        id: 15,
        center_mm: [2_500, 7, 2_500],
        inner_radius_mm: 2_000,
        outer_radius_mm: 2_000,
        opacity: u16::MAX,
        density: DensityBrushOp::Add(0),
        species: Some(15),
    }];
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 0xfedc_ba98_7654_3210,
            version: 1,
            base_density: u16::MAX,
            base_species: 7,
            noise: None,
        },
        brushes,
        budget,
    )
    .expect("bounded full-density author field with species variation")
}

fn source(count: usize) -> HeightfieldPlacementOutput {
    assert!(matches!(count, 10_000 | 100_000));
    assert_eq!(count % GRID_COLUMNS, 0);
    let surface = surface();
    let field = field();
    let config = HeightfieldPlacementConfig {
        seed: SEED,
        version: 1,
        salt: 22,
        grid_spacing_mm: SPACING_MM,
        min_distance_mm: 0,
        vertical_offset_mm: 7,
    };
    // 100x100 / 100x1000 whole cells: no boundary rejection or density thinning.
    let region = HeightfieldPlacementRegion::new(
        [0, 0],
        [
            i64::try_from(GRID_COLUMNS).unwrap() * i64::from(SPACING_MM),
            i64::try_from(count / GRID_COLUMNS).unwrap() * i64::from(SPACING_MM),
        ],
    )
    .expect("aligned half-open XZ footprint");
    let work = estimate_heightfield_placement_work(config, region, &field)
        .expect("bounded dense placement work");
    assert_eq!(work.primary_candidates, count as u64);
    assert_eq!(work.max_candidate_evaluations, count as u64);
    assert_eq!(work.max_sample_reads, 4 * count as u64);
    assert_eq!(work.max_field_work, 2 * count as u64);
    let output = scatter_on_heightfield(
        config,
        region,
        &field,
        &surface,
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: 1,
            max_slope_run: 1,
        },
        HeightfieldPlacementBudget {
            max_candidate_evaluations: work.max_candidate_evaluations,
            max_sample_reads: work.max_sample_reads,
            max_field_work: work.max_field_work,
            max_output_points: count,
            max_output_bytes: count
                * (size_of::<HeightfieldPlacementPoint>() + size_of::<ScatterFieldSample>()),
        },
    )
    .expect("real heightfield placements with author attributes");
    assert_eq!(output.points.len(), count);
    assert_eq!(output.attributes.len(), count);
    assert_eq!(output.candidate_evaluations, work.max_candidate_evaluations);
    assert_eq!(output.surface_sample_reads, work.max_sample_reads);
    assert_eq!(output.field_work, work.max_field_work);
    let mut keys = BTreeSet::new();
    for (point, attribute) in output.points.iter().zip(&output.attributes) {
        assert!(keys.insert(point.key), "source identities must be unique");
        assert_eq!(
            point
                .key
                .rebuild(&surface, config.vertical_offset_mm)
                .expect("projected identity reconstructs against the real surface"),
            *point,
        );
        assert_eq!(*attribute, field.sample(point.position_mm));
        assert_eq!(attribute.density, u16::MAX);
        let [nx, _, nz] = point.surface_sample.normal_mm();
        assert!(nx != 0 || nz != 0, "sampled geometry must be nonflat");
    }
    assert!(output.attributes.iter().any(|row| row.species == 7));
    assert!(output.attributes.iter().any(|row| row.species == 15));
    output
}

fn deleted_ordinal(ordinal: usize, percent: usize) -> bool {
    match percent {
        0 => false,
        25 => ordinal.is_multiple_of(4),
        100 => true,
        _ => unreachable!("only 0/25/100 percent fixtures"),
    }
}

type BufferIdentity = (
    *const HeightfieldPlacementPoint,
    usize,
    *const ScatterFieldSample,
    usize,
);

fn buffers(output: &HeightfieldPlacementOutput) -> BufferIdentity {
    (
        output.points.as_ptr(),
        output.points.capacity(),
        output.attributes.as_ptr(),
        output.attributes.capacity(),
    )
}

fn capacity_bytes(output: &HeightfieldPlacementOutput) -> [usize; 2] {
    [
        output.points.capacity() * size_of::<HeightfieldPlacementPoint>(),
        output.attributes.capacity() * size_of::<ScatterFieldSample>(),
    ]
}

struct Fixture<'a> {
    source: &'a HeightfieldPlacementOutput,
    percent: usize,
    deleted_keys: BTreeSet<HeightfieldPlacementKey>,
    deletions: HeightfieldPlacementDeletionSet,
    codec_budget: PlacementDeletionCodecBudget,
    wire: Vec<u8>,
}

impl<'a> Fixture<'a> {
    fn new(source: &'a HeightfieldPlacementOutput, percent: usize) -> Self {
        let count = source.points.len() * percent / 100;
        let mut keys = Vec::with_capacity(count);
        keys.extend(
            source
                .points
                .iter()
                .enumerate()
                .filter(|(ordinal, _)| deleted_ordinal(*ordinal, percent))
                .map(|(_, point)| point.key),
        );
        let deleted_keys: BTreeSet<_> = keys.iter().copied().collect();
        assert_eq!(keys.len(), count);
        assert_eq!(deleted_keys.len(), count);
        let budget = HeightfieldPlacementDeletionBudget {
            max_keys: count,
            max_key_bytes: keys.capacity() * size_of::<HeightfieldPlacementKey>(),
        };
        let deletions = HeightfieldPlacementDeletionSet::new(keys, budget)
            .expect("bounded complete placement deletion identities");
        assert!(deletions
            .keys()
            .iter()
            .copied()
            .eq(deleted_keys.iter().copied()));
        let max_encoded_bytes =
            PLACEMENT_DELETION_HEADER_BYTES + count * PLACEMENT_DELETION_KEY_BYTES;
        let codec_budget = PlacementDeletionCodecBudget {
            max_encoded_bytes,
            deletions: budget,
        };
        let wire = encode_placement_deletions(&deletions, max_encoded_bytes)
            .expect("canonical reference placement wire");
        assert_eq!(wire.len(), max_encoded_bytes);
        Self {
            source,
            percent,
            deleted_keys,
            deletions,
            codec_budget,
            wire,
        }
    }

    fn source_buffer(&self) -> HeightfieldPlacementOutput {
        HeightfieldPlacementOutput {
            points: self.source.points.clone(),
            attributes: self.source.attributes.clone(),
            cache_key: self.source.cache_key.clone(),
            candidate_evaluations: self.source.candidate_evaluations,
            surface_sample_reads: self.source.surface_sample_reads,
            field_work: self.source.field_work,
        }
    }

    fn validate(&self, output: &DeletionFilteredHeightfieldPlacement, original: BufferIdentity) {
        assert_eq!(
            buffers(&output.source),
            original,
            "both Vec pointers/capacities retained"
        );
        assert_eq!(output.source.cache_key, self.source.cache_key);
        assert_eq!(
            output.source.candidate_evaluations,
            self.source.candidate_evaluations
        );
        assert_eq!(
            output.source.surface_sample_reads,
            self.source.surface_sample_reads
        );
        assert_eq!(output.source.field_work, self.source.field_work);
        assert_eq!(output.deletions, self.deletions);
        assert_eq!(output.removed, self.deleted_keys.len());
        assert_eq!(
            output.source.points.len(),
            self.source.points.len() - output.removed
        );
        assert_eq!(output.source.points.len(), output.source.attributes.len());
        assert!(output
            .deletions
            .keys()
            .iter()
            .copied()
            .eq(self.deleted_keys.iter().copied()));

        // Independent source-ordinal oracle, not another call to the filter.
        // Check every original key against BTreeSet membership, then compare
        // every surviving full point (key/position/exact sample) and attribute.
        let mut survivor = 0;
        for (ordinal, (point, attribute)) in self
            .source
            .points
            .iter()
            .zip(&self.source.attributes)
            .enumerate()
        {
            let deleted = deleted_ordinal(ordinal, self.percent);
            assert_eq!(self.deleted_keys.contains(&point.key), deleted);
            if !deleted {
                assert_eq!(output.source.points.get(survivor), Some(point));
                assert_eq!(output.source.attributes.get(survivor), Some(attribute));
                survivor += 1;
            }
        }
        assert_eq!(
            survivor,
            output.source.points.len(),
            "no missing/extra/reordered rows"
        );
    }

    fn run(&self) {
        let mut filtering = [Duration::ZERO; SAMPLES];
        let mut filtered_capacity = None;
        for iteration in 0..WARMUP + SAMPLES {
            // All source/cache clones, pointer/capacity capture and setup precede timing.
            let buffer = black_box(self.source_buffer());
            let original = buffers(&buffer);
            let deletions = black_box(&self.deletions);
            let start = Instant::now();
            let result = black_box(filter_heightfield_placement(buffer, deletions));
            let elapsed = start.elapsed();

            let output = result.expect("aligned in-place placement filter");
            self.validate(&output, original);
            let capacity = capacity_bytes(&output.source);
            assert_eq!(*filtered_capacity.get_or_insert(capacity), capacity);
            if iteration >= WARMUP {
                filtering[iteration - WARMUP] = elapsed;
            }
            // Filter result destruction is outside the timer, including delete-all.
        }

        let mut codec = [Duration::ZERO; SAMPLES];
        let mut decoded_capacity = None;
        let mut wire_capacity = None;
        for iteration in 0..WARMUP + SAMPLES {
            let deletions = black_box(&self.deletions);
            let budget = black_box(self.codec_budget);
            let start = Instant::now();
            let bytes = encode_placement_deletions(deletions, budget.max_encoded_bytes)
                .expect("encode placement deletions");
            let decoded = decode_placement_deletions(black_box(&bytes), budget)
                .expect("decode placement deletions");
            black_box((&bytes, &decoded));
            let elapsed = start.elapsed();

            assert_eq!(bytes, self.wire, "canonical wire equality");
            assert_eq!(decoded.keys(), self.deletions.keys(), "full-key roundtrip");
            let capacity = decoded.allocated_key_bytes();
            assert!(capacity <= budget.deletions.max_key_bytes);
            assert_eq!(*decoded_capacity.get_or_insert(capacity), capacity);
            assert_eq!(
                *wire_capacity.get_or_insert(bytes.capacity()),
                bytes.capacity()
            );
            // Every decoded set must reproduce the complete row oracle too.
            // Cloning, filtering, validation and all drops here are untimed.
            let buffer = self.source_buffer();
            let original = buffers(&buffer);
            let output = filter_heightfield_placement(buffer, &decoded)
                .expect("filter with roundtripped placement keys");
            self.validate(&output, original);
            if iteration >= WARMUP {
                codec[iteration - WARMUP] = elapsed;
            }
        }

        let [source_points, source_attributes] = capacity_bytes(self.source);
        let [filtered_points, filtered_attributes] = filtered_capacity.unwrap();
        println!(
            "N={} D={} deleted={}% input_points={} output_points={} removed={} \
             candidate_evaluations={} surface_sample_reads={} field_work={}",
            self.source.points.len(),
            self.deleted_keys.len(),
            self.percent,
            self.source.points.len(),
            self.source.points.len() - self.deleted_keys.len(),
            self.deleted_keys.len(),
            self.source.candidate_evaluations,
            self.source.surface_sample_reads,
            self.source.field_work,
        );
        println!(
            "  source_point_vec_capacity_bytes={source_points} \
             source_attr_vec_capacity_bytes={source_attributes} \
             filtered_point_vec_capacity_bytes={filtered_points} \
             filtered_attr_vec_capacity_bytes={filtered_attributes} \
             key_vec_capacity_bytes={} decoded_key_vec_capacity_bytes={} \
             wirebytes={} wire_vec_capacity_bytes={}",
            self.deletions.allocated_key_bytes(),
            decoded_capacity.unwrap(),
            self.wire.len(),
            wire_capacity.unwrap(),
        );
        report("filter_heightfield_placement", &mut filtering);
        report("codec.encode+decode", &mut codec);
        println!(
            "  validation=passed for all {WARMUP} warmups + {SAMPLES} samples per operation; \
             decoded-set filtering checked every codec run"
        );
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
        "placement_deletions release host benchmark: os={} arch={} warmup={WARMUP} \
         samples={SAMPLES} seed={SEED:#018x} version=1 salt=22 spacing_mm={SPACING_MM} \
         min_distance_mm=0 full_density=65535 offset_mm=7 heightfield=65x65_nonflat \
         deletion=0:none,25:source_ordinal%4==0,100:all",
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    println!(
        "filter timing excludes source generation, source/cache/key clones, set construction, \
         codec, oracle checks and result drop; codec timing includes encode+decode allocations \
         and internal key validation, excluding external validation, decoded-set filtering and result drop"
    );
    println!(
        "bytes are Vec capacity payloads (excluding Vec/Arc headers, allocator metadata, \
         shared terrain/field and oracle storage) or canonical wire length; \
         no GPU, frame-time, allocation-count, RSS or time-bound claims; \
         nearest-rank indices=32/61/64 (1-based)"
    );
    for count in [10_000, 100_000] {
        let source = source(count);
        for percent in [0, 25, 100] {
            Fixture::new(&source, percent).run();
        }
    }
}
