//! CPU-only public-contract tests; linear terrain scatter is the result oracle.
#![cfg(feature = "scatter")]

use std::{collections::BTreeMap, convert::Infallible, error::Error, mem::size_of, sync::Arc};

use prism_world_residency::{
    estimate_scatter_work, scatter, scatter_with_indexed_density_field, DensityBrush,
    DensityBrushOp, DensityNoise, FieldScatterBudget, ScatterBrushIndex, ScatterBrushIndexBudget,
    ScatterBrushIndexError, ScatterBrushScratch, ScatterBudget, ScatterCandidateKey, ScatterConfig,
    ScatterDeletionBudget, ScatterDeletionSet, ScatterDensityField, ScatterError,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldError, ScatterFieldSample,
    ScatterPoint, ScatterRegion,
};
use prism_world_terrain::{
    filter_heightfield_scatter, scatter_with_heightfield, scatter_with_indexed_heightfield,
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldQueryError,
    HeightfieldScatterBudget, HeightfieldScatterError, HeightfieldScatterFilters,
    HeightfieldScatterOutput, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey, HEIGHTFIELD_SCATTER_FILTER_VERSION,
    HEIGHTFIELD_SURFACE_VERSION,
};

const LIMIT: usize = 8_192;
const HEIGHTS: [i64; 9] = [-4, 0, 5, -2, 2, 8, 0, 4, 10];

struct GridHeights([i64; 9]);

impl HeightSampler for GridHeights {
    type Error = Infallible;

    fn sample_height(&self, sample: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        let index = |q: u16| ((u64::from(q) * 2 + 32_767) / 65_535) as usize;
        Ok(HeightSampleValue::from_millimetres(
            self.0[index(sample.v_q16()) * 3 + index(sample.u_q16())],
        ))
    }
}

fn surface(origin: [i64; 3], spacing: u32, heights: [i64; 9]) -> PlanarHeightfieldSurface {
    let tile = HeightfieldTile::bake(
        TileKey::root(CubeFace::PositiveY),
        HeightfieldSpec::new(3, 1).unwrap(),
        &GridHeights(heights),
        BakeLimits {
            max_samples: 9,
            max_bytes: 9 * size_of::<i16>(),
        },
    )
    .unwrap();
    PlanarHeightfieldSurface::new(
        Arc::new(tile),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: origin,
            sample_spacing_mm: spacing,
        },
    )
    .unwrap()
}

fn field_config() -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 91,
        version: 3,
        base_density: u16::MAX,
        base_species: 11,
        noise: None,
    }
}

fn noise() -> DensityNoise {
    DensityNoise {
        cell_size_mm: 7,
        amplitude: 12_000,
        warp_cell_size_mm: 11,
        warp_amplitude_mm: 5,
    }
}

fn field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(config, brushes, budget).unwrap()
}

fn brush(id: u64) -> DensityBrush {
    DensityBrush {
        id,
        center_mm: [0; 3],
        inner_radius_mm: 8,
        outer_radius_mm: 18,
        opacity: u16::MAX,
        density: DensityBrushOp::Add(0),
        species: Some(id as u32),
    }
}

fn painted_field() -> ScatterDensityField {
    // Input and spatial traversal order both disagree with canonical ID order.
    let brushes = [
        (50, -6, DensityBrushOp::Add(12_345), None),
        (20, 3, DensityBrushOp::Add(20_000), Some(2)),
        (40, -3, DensityBrushOp::Add(-3_000), None),
        (10, 6, DensityBrushOp::Replace(60_000), Some(1)),
        (30, 0, DensityBrushOp::Replace(1_000), Some(3)),
    ]
    .into_iter()
    .map(|(id, x, density, species)| DensityBrush {
        center_mm: [x, 0, 0],
        density,
        species,
        ..brush(id)
    })
    .collect();
    field(
        ScatterFieldConfig {
            base_density: 46_000,
            noise: Some(noise()),
            ..field_config()
        },
        brushes,
    )
}

fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

fn unrestricted() -> HeightfieldScatterFilters {
    HeightfieldScatterFilters {
        surface_height_mm: [i64::MIN, i64::MAX],
        clearance_mm: [i64::MIN, i64::MAX],
        max_slope_rise: u32::MAX,
        max_slope_run: 1,
    }
}

#[derive(Clone, Copy)]
struct Limits {
    source: ScatterBudget,
    field: FieldScatterBudget,
    surface: HeightfieldScatterBudget,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            source: ScatterBudget {
                max_candidate_evaluations: 1_000_000,
                max_output_points: LIMIT,
                max_output_bytes: LIMIT * size_of::<ScatterPoint>(),
            },
            field: FieldScatterBudget {
                max_field_work: 10_000_000,
                max_output_bytes: LIMIT
                    * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
            },
            surface: HeightfieldScatterBudget {
                max_sample_reads: 4 * LIMIT as u64,
            },
        }
    }
}

impl Limits {
    fn zero() -> Self {
        Self {
            source: ScatterBudget {
                max_candidate_evaluations: 0,
                max_output_points: 0,
                max_output_bytes: 0,
            },
            field: FieldScatterBudget {
                max_field_work: 0,
                max_output_bytes: 0,
            },
            surface: HeightfieldScatterBudget {
                max_sample_reads: 0,
            },
        }
    }
}

struct Fixture {
    config: ScatterConfig,
    region: ScatterRegion,
    index: ScatterBrushIndex,
    surface: PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
}

impl Fixture {
    fn new(field: ScatterDensityField) -> Self {
        Self {
            config: ScatterConfig {
                seed: 0x3a65_c807_9742_1eab,
                version: 7,
                salt: 0xf09c_1172,
                grid_spacing_mm: 3,
                min_distance_mm: 0,
            },
            region: region([-11, -6, -11], [12, 7, 12]),
            index: ScatterBrushIndex::new(
                field,
                ScatterBrushIndexBudget {
                    max_brushes: 128,
                    max_index_bytes: 128 * 1024,
                },
            )
            .unwrap(),
            surface: surface([-16, 0, -16], 16, HEIGHTS),
            filters: HeightfieldScatterFilters {
                surface_height_mm: [-4, 8],
                clearance_mm: [-3, 4],
                max_slope_rise: 2,
                max_slope_run: 1,
            },
        }
    }

    fn scratch(&self) -> ScatterBrushScratch {
        self.index
            .new_scratch(self.index.field().brushes().len() * size_of::<usize>())
            .unwrap()
    }

    fn linear(&self, limits: Limits) -> Result<HeightfieldScatterOutput, HeightfieldScatterError> {
        scatter_with_heightfield(
            self.config,
            self.region,
            limits.source,
            self.index.field(),
            limits.field,
            &self.surface,
            self.filters,
            limits.surface,
        )
    }

    fn indexed(
        &self,
        scratch: &mut ScatterBrushScratch,
        limits: Limits,
    ) -> Result<HeightfieldScatterOutput, HeightfieldScatterError> {
        scatter_with_indexed_heightfield(
            self.config,
            self.region,
            limits.source,
            &self.index,
            scratch,
            limits.field,
            &self.surface,
            self.filters,
            limits.surface,
        )
    }

    fn parity(&self, scratch: &mut ScatterBrushScratch) -> HeightfieldScatterOutput {
        let before = (scratch.capacity(), scratch.allocated_bytes());
        let limits = Limits::default();
        let expected = self.linear(limits).unwrap();
        let actual = self.indexed(scratch, limits).unwrap();
        assert_parity(&actual, &expected);
        let source = scatter(self.config, self.region, limits.source).unwrap();
        let density = scatter_with_indexed_density_field(
            self.config,
            self.region,
            limits.source,
            &self.index,
            scratch,
            limits.field,
        )
        .unwrap();
        // Count every source survivor, including subsequent density/terrain rejects.
        let work: u64 = source
            .points
            .iter()
            .map(|p| {
                let sample = self
                    .index
                    .sample(p.position_mm, scratch, self.index.max_sample_work())
                    .unwrap();
                assert_eq!(sample.sample, self.index.field().sample(p.position_mm));
                sample.field_work
            })
            .sum();
        assert_eq!(actual.field_work, work);
        assert_eq!(actual.field_work, density.field_work);
        assert_eq!(actual.candidate_evaluations, source.candidate_evaluations);
        assert_eq!(actual.surface_sample_reads, density.points.len() as u64 * 4);
        assert_eq!(actual.cache_key.source, density.cache_key);
        // Terrain thinning retains the two charged source capacities, even at len 0.
        assert_eq!(actual.points.capacity(), density.points.capacity());
        assert_eq!(actual.attributes.capacity(), density.attributes.capacity());
        for (point, attribute) in rows(&actual) {
            assert_eq!(point.key.rebuild().unwrap(), point);
            assert!(source.points.contains(&point));
            assert_eq!(attribute, self.index.field().sample(point.position_mm));
        }
        assert_eq!(before, (scratch.capacity(), scratch.allocated_bytes()));
        actual
    }
}

fn assert_parity(actual: &HeightfieldScatterOutput, expected: &HeightfieldScatterOutput) {
    assert_eq!(actual.points, expected.points);
    assert_eq!(actual.attributes, expected.attributes);
    assert_eq!(actual.cache_key, expected.cache_key);
    assert_eq!(actual.candidate_evaluations, expected.candidate_evaluations);
    assert_eq!(actual.surface_sample_reads, expected.surface_sample_reads);
    assert_eq!(actual.points.capacity(), expected.points.capacity());
    assert_eq!(actual.attributes.capacity(), expected.attributes.capacity());
    assert_eq!(actual.points.len(), actual.attributes.len());
    // field_work measures different algorithms; parity() checks its actual probes.
}

fn rows(output: &HeightfieldScatterOutput) -> Vec<(ScatterPoint, ScatterFieldSample)> {
    output
        .points
        .iter()
        .copied()
        .zip(output.attributes.iter().copied())
        .collect()
}

fn assert_error(actual: HeightfieldScatterError, expected: HeightfieldScatterError) {
    assert_eq!(actual, expected);
    assert!(!actual.to_string().is_empty());
    match expected {
        HeightfieldScatterError::Index(inner) => {
            assert_eq!(actual.source().unwrap().downcast_ref(), Some(&inner));
            assert!(actual.to_string().contains(&inner.to_string()));
        }
        HeightfieldScatterError::Source(inner) => {
            assert_eq!(actual.source().unwrap().downcast_ref(), Some(&inner));
            assert!(actual.to_string().contains(&inner.to_string()));
        }
        HeightfieldScatterError::Surface(inner) => {
            assert_eq!(actual.source().unwrap().downcast_ref(), Some(&inner));
            assert!(actual.to_string().contains(&inner.to_string()));
        }
        _ => assert!(actual.source().is_none()),
    }
}

#[test]
fn empty_axes_and_brushless_fields_match_linear_including_zero_budgets() {
    for noisy in [false, true] {
        for density in [0, u16::MAX] {
            let mut case = Fixture::new(field(
                ScatterFieldConfig {
                    base_density: density,
                    noise: noisy.then_some(noise()),
                    ..field_config()
                },
                vec![],
            ));
            let mut scratch = case.scratch();
            assert_eq!((case.index.node_count(), scratch.allocated_bytes()), (0, 0));
            let output = case.parity(&mut scratch);
            assert_eq!(
                output.field_work,
                case.linear(Limits::default()).unwrap().field_work
            );
            if !noisy && density == 0 {
                assert!(output.points.is_empty());
                assert_eq!(output.surface_sample_reads, 0);
                assert!(output.field_work > 0);
            }
            for axis in 0..3 {
                let min = [100, -100, 100];
                let mut max = [101, -99, 101];
                max[axis] = min[axis];
                case.region = region(min, max); // Empty requests need no terrain coverage.
                let actual = case.indexed(&mut scratch, Limits::zero()).unwrap();
                assert_eq!(actual, case.linear(Limits::zero()).unwrap());
                assert!(actual.points.is_empty());
                assert_eq!(
                    (
                        actual.candidate_evaluations,
                        actual.field_work,
                        actual.surface_sample_reads
                    ),
                    (0, 0, 0)
                );
            }
        }
    }
}

#[test]
fn warped_noise_and_unordered_noncommutative_brushes_preserve_rows_and_identity() {
    let original = painted_field();
    assert_eq!(
        original.sample([0; 3]),
        ScatterFieldSample {
            density: 12_345,
            species: 3
        }
    );
    let case = Fixture::new(original.clone());
    let expected = case.parity(&mut case.scratch());
    assert!(!expected.points.is_empty());
    assert!(expected
        .attributes
        .windows(2)
        .any(|w| w[0].density != w[1].density));
    assert!(expected
        .attributes
        .windows(2)
        .any(|w| w[0].species != w[1].species));
    let mut brushes = original.brushes().to_vec();
    brushes.reverse();
    for _ in 0..3 {
        brushes.rotate_left(1);
        let other = Fixture::new(field(original.config(), brushes.clone()));
        assert_eq!(other.index.field(), &original);
        let actual = other.parity(&mut other.scratch());
        assert_eq!(actual, expected);
    }
}

#[test]
fn sparse_and_fully_overlapping_indices_charge_actual_work_not_linear_work() {
    for overlap in [false, true] {
        let brushes = (0..33)
            .map(|id| DensityBrush {
                center_mm: if overlap {
                    [0; 3]
                } else {
                    [1_000 + id as i64 * 1_000, 0, 0]
                },
                inner_radius_mm: if overlap { 64 } else { 5 },
                outer_radius_mm: if overlap { 128 } else { 10 },
                species: None,
                ..brush(id)
            })
            .collect();
        let case = Fixture::new(field(field_config(), brushes));
        let actual = case.parity(&mut case.scratch());
        let source = scatter(case.config, case.region, Limits::default().source).unwrap();
        let base = field(field_config(), vec![]).work_per_sample();
        let per_point = if overlap {
            case.index.max_sample_work()
        } else {
            base + 1
        };
        assert_eq!(actual.field_work, source.points.len() as u64 * per_point);
        let linear = case.linear(Limits::default()).unwrap();
        assert!(!actual.points.is_empty());
        if overlap {
            assert!(actual.field_work > linear.field_work);
        } else {
            assert!(actual.field_work < linear.field_work);
        }
    }
}

#[test]
fn i64_edges_keep_brush_bounds_warp_height_and_clearance_widened() {
    for low in [true, false] {
        let edge = if low { i64::MIN } else { i64::MAX - 16 };
        let mut case = Fixture::new(field(
            ScatterFieldConfig {
                noise: Some(DensityNoise {
                    warp_amplitude_mm: u32::MAX,
                    ..noise()
                }),
                ..field_config()
            },
            vec![DensityBrush {
                center_mm: [if low { i64::MIN } else { i64::MAX }; 3],
                inner_radius_mm: u32::MAX,
                outer_radius_mm: u32::MAX,
                ..brush(7)
            }],
        ));
        case.config.grid_spacing_mm = 1;
        case.region = region([edge; 3], [edge + 4; 3]);
        case.surface = surface([edge; 3], 8, [0; 9]);
        case.filters = unrestricted();
        assert!(!case.parity(&mut case.scratch()).points.is_empty());
        // Opposite-end Y produces an unrepresentable i64 clearance, not a wrap.
        let y = if low { i64::MAX - 1 } else { i64::MIN };
        case.region = region([edge, y, edge], [edge + 1, y + 1, edge + 1]);
        assert!(case.parity(&mut case.scratch()).points.is_empty());
        // A decoded surface height outside i64 must not saturate into full filters.
        let origin_y = if low { i64::MIN } else { i64::MAX };
        case.surface = surface([edge, origin_y, edge], 8, [if low { -1 } else { 1 }; 9]);
        case.region = region([edge; 3], [edge + 1; 3]);
        assert!(case.parity(&mut case.scratch()).points.is_empty());
    }
}

#[test]
fn reversed_nonaligned_xyz_partitions_preserve_full_rows_and_source_minimum_distance() {
    let mut case = Fixture::new(painted_field());
    case.config.min_distance_mm = 4;
    let mut scratch = case.scratch();
    let whole = case.parity(&mut scratch);
    assert!(!whole.points.is_empty());
    let expected: BTreeMap<_, _> = rows(&whole)
        .into_iter()
        .map(|(p, a)| (p.key, (p, a)))
        .collect();
    let mut union = BTreeMap::new();
    let cuts = [[-11, -2, 12], [-6, 1, 7], [-11, 2, 12]];
    assert!(cuts
        .iter()
        .all(|axis| axis[1] % i64::from(case.config.grid_spacing_mm) != 0));
    for x in (0..2).rev() {
        for y in (0..2).rev() {
            for z in (0..2).rev() {
                case.region = region(
                    [cuts[0][x], cuts[1][y], cuts[2][z]],
                    [cuts[0][x + 1], cuts[1][y + 1], cuts[2][z + 1]],
                );
                for (point, attribute) in rows(&case.parity(&mut scratch)) {
                    assert!(case.region.contains_mm(point.position_mm));
                    assert!(
                        union.insert(point.key, (point, attribute)).is_none(),
                        "duplicate partition ownership"
                    );
                }
            }
        }
    }
    assert_eq!(union, expected);
    for (i, a) in whole.points.iter().enumerate() {
        for b in &whole.points[i + 1..] {
            let squared: i128 = (0..3)
                .map(|axis| {
                    let d = i128::from(a.position_mm[axis]) - i128::from(b.position_mm[axis]);
                    d * d
                })
                .sum();
            assert!(squared >= i128::from(case.config.min_distance_mm).pow(2));
        }
    }
}

#[test]
fn exact_and_one_less_work_reads_and_output_capacity_are_not_relaxed_by_thinning() {
    for density in [0, u16::MAX] {
        let mut case = Fixture::new(field(
            ScatterFieldConfig {
                base_density: density,
                ..field_config()
            },
            vec![DensityBrush {
                opacity: 0,
                ..brush(1)
            }],
        ));
        case.config.grid_spacing_mm = 1;
        case.region = region([0; 3], [2; 3]);
        case.surface = surface([0; 3], 1, [0; 9]);
        case.filters = HeightfieldScatterFilters {
            surface_height_mm: [1, 1],
            ..unrestricted()
        };
        let mut scratch = case.scratch();
        let baseline = case.parity(&mut scratch);
        assert!(baseline.points.is_empty());
        let bytes = baseline.points.capacity() * size_of::<ScatterPoint>()
            + baseline.attributes.capacity() * size_of::<ScatterFieldSample>();
        assert!(bytes > 0);
        let estimate = estimate_scatter_work(case.config, case.region).unwrap();
        let work = estimate.primary_candidates * case.index.max_sample_work();
        let reads = 4 * estimate.primary_candidates;
        let exact = Limits {
            source: ScatterBudget {
                max_candidate_evaluations: estimate.max_candidate_evaluations,
                max_output_points: 8,
                max_output_bytes: 8 * size_of::<ScatterPoint>(),
            },
            field: FieldScatterBudget {
                max_field_work: work,
                max_output_bytes: bytes,
            },
            surface: HeightfieldScatterBudget {
                max_sample_reads: reads,
            },
        };
        assert_eq!(case.indexed(&mut scratch, exact).unwrap(), baseline);
        assert_eq!(baseline.field_work, work);
        assert_eq!(
            baseline.surface_sample_reads,
            if density == 0 { 0 } else { reads }
        );
        let mut limited = exact;
        limited.field.max_field_work -= 1;
        limited.source = Limits::zero().source;
        assert_error(
            case.indexed(&mut scratch, limited).unwrap_err(),
            HeightfieldScatterError::Index(ScatterBrushIndexError::WorkBudgetExceeded {
                required: work,
                limit: work - 1,
            }),
        );
        limited = exact;
        limited.surface.max_sample_reads -= 1;
        limited.field = Limits::zero().field;
        limited.source = Limits::zero().source;
        assert_error(
            case.indexed(&mut scratch, limited).unwrap_err(),
            HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                required: reads,
                limit: reads - 1,
            },
        );
        limited = exact;
        limited.field.max_output_bytes -= 1;
        assert_error(
            case.indexed(&mut scratch, limited).unwrap_err(),
            HeightfieldScatterError::Index(ScatterBrushIndexError::Field(
                ScatterFieldError::OutputBytesBudgetExceeded {
                    required: bytes,
                    limit: bytes - 1,
                },
            )),
        );
        for source in [
            ScatterBudget {
                max_candidate_evaluations: estimate.max_candidate_evaluations - 1,
                ..exact.source
            },
            ScatterBudget {
                max_output_points: 0,
                ..exact.source
            },
            ScatterBudget {
                max_output_bytes: 0,
                ..exact.source
            },
        ] {
            let expected = scatter(case.config, case.region, source).unwrap_err();
            assert_error(
                case.indexed(&mut scratch, Limits { source, ..exact })
                    .unwrap_err(),
                HeightfieldScatterError::Index(ScatterBrushIndexError::Field(
                    ScatterFieldError::Scatter(expected),
                )),
            );
        }
    }
}

#[test]
fn terrain_preflight_precedes_index_and_source_and_empty_still_checks_scratch() {
    let empty = Fixture::new(field(field_config(), vec![]));
    let mut scratch = empty.scratch();
    let before = (scratch.capacity(), scratch.allocated_bytes());
    let mut case = Fixture::new(field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            opacity: 0,
            ..brush(1)
        }],
    ));
    let valid_region = case.region;
    let insufficient =
        HeightfieldScatterError::Index(ScatterBrushIndexError::ScratchCapacityExceeded {
            required: 1,
            limit: 0,
        });
    for filters in [
        HeightfieldScatterFilters {
            surface_height_mm: [1, -1],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            clearance_mm: [1, -1],
            ..unrestricted()
        },
        HeightfieldScatterFilters {
            max_slope_run: 0,
            ..unrestricted()
        },
    ] {
        case.filters = filters;
        assert_error(
            case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
            HeightfieldScatterError::InvalidFilters,
        );
    }
    case.filters = unrestricted();
    case.config.grid_spacing_mm = 0;
    assert_error(
        case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::Scatter(
            ScatterError::InvalidGridSpacing,
        )),
    );
    case.config.grid_spacing_mm = 3;
    case.region = ScatterRegion {
        min_mm: [1, 0, 0],
        max_mm: [0, 1, 1],
    };
    assert_error(
        case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
        HeightfieldScatterError::Source(ScatterFieldError::Scatter(ScatterError::InvalidRegion)),
    );
    // All four footprint edges must be covered even with density zero and no budgets.
    for request in [
        region([-17, 0, 0], [0, 1, 1]),
        region([0, 0, -17], [1, 1, 0]),
        region([0; 3], [17, 1, 1]),
        region([0; 3], [1, 1, 17]),
    ] {
        case.region = request;
        let expected = case.linear(Limits::zero()).unwrap_err();
        assert_eq!(
            expected,
            HeightfieldScatterError::Surface(HeightfieldQueryError::OutsideSurface)
        );
        assert_error(
            case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
            expected,
        );
    }
    case.region = valid_region;
    let expected = case.linear(Limits::zero()).unwrap_err();
    assert!(matches!(
        expected,
        HeightfieldScatterError::SurfaceWorkBudgetExceeded { .. }
    ));
    assert_error(
        case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
        expected,
    );
    // Scratch admission wins over indexed work and source limits after terrain preflight.
    assert_error(
        case.indexed(
            &mut scratch,
            Limits {
                surface: Limits::default().surface,
                ..Limits::zero()
            },
        )
        .unwrap_err(),
        insufficient,
    );
    for axis in 0..3 {
        let min = [100; 3];
        let mut max = [101; 3];
        max[axis] = min[axis];
        case.region = region(min, max);
        assert_error(
            case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
            insufficient,
        );
        let mut adequate = case.scratch();
        let output = case.indexed(&mut adequate, Limits::zero()).unwrap();
        assert_eq!(output, case.linear(Limits::zero()).unwrap());
        assert!(output.points.is_empty());
    }
    assert_eq!(before, (scratch.capacity(), scratch.allocated_bytes()));
}

#[test]
fn enormous_read_and_index_work_preflights_fail_without_source_enumeration() {
    let mut case = Fixture::new(field(field_config(), (0..6).map(brush).collect()));
    case.config.grid_spacing_mm = 1;
    let mut scratch = case.scratch();
    for (edge, expected) in [
        (
            1_u32 << 20,
            HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                required: 1_u64 << 62,
                limit: 0,
            },
        ),
        (1_u32 << 21, HeightfieldScatterError::SurfaceWorkOverflow),
    ] {
        case.surface = surface([0; 3], edge, [0; 9]);
        case.region = region([0; 3], [i64::from(edge); 3]);
        assert_error(
            case.indexed(&mut scratch, Limits::zero()).unwrap_err(),
            expected,
        );
        assert_eq!(case.linear(Limits::zero()).unwrap_err(), expected);
    }
    case.region = region([0; 3], [1 << 20; 3]);
    let primary = estimate_scatter_work(case.config, case.region)
        .unwrap()
        .primary_candidates;
    assert!(primary.checked_mul(case.index.max_sample_work()).is_none());
    assert_error(
        case.indexed(
            &mut scratch,
            Limits {
                field: FieldScatterBudget {
                    max_field_work: u64::MAX,
                    max_output_bytes: 0,
                },
                surface: HeightfieldScatterBudget {
                    max_sample_reads: u64::MAX,
                },
                ..Limits::zero()
            },
        )
        .unwrap_err(),
        HeightfieldScatterError::Index(ScatterBrushIndexError::ArithmeticOverflow),
    );
}

#[test]
fn index_cache_retains_complete_inactive_field_recipe_and_shared_terrain_identity() {
    let inactive = DensityBrush {
        center_mm: [1_000; 3],
        opacity: 0,
        ..brush(42)
    };
    let disabled = DensityNoise {
        amplitude: 0,
        warp_amplitude_mm: 0,
        ..noise()
    };
    let config = ScatterFieldConfig {
        noise: Some(disabled),
        ..field_config()
    };
    let case = Fixture::new(field(config, vec![inactive]));
    let baseline = case.parity(&mut case.scratch());
    assert_eq!(baseline.cache_key.source.field, *case.index.field());
    assert_eq!(
        baseline.cache_key.source.field.brushes().as_ptr(),
        case.index.field().brushes().as_ptr()
    );
    assert!(Arc::ptr_eq(
        baseline.cache_key.surface.tile(),
        case.surface.tile()
    ));
    assert_eq!(
        baseline.cache_key.surface_version,
        HEIGHTFIELD_SURFACE_VERSION
    );
    assert_eq!(
        baseline.cache_key.filter_version,
        HEIGHTFIELD_SCATTER_FILTER_VERSION
    );
    let equal = Fixture::new(field(config, vec![inactive]));
    assert!(!Arc::ptr_eq(equal.surface.tile(), case.surface.tile()));
    assert_eq!(
        baseline.cache_key,
        equal.parity(&mut equal.scratch()).cache_key
    );
    for changed in [
        DensityBrush { id: 43, ..inactive },
        DensityBrush {
            center_mm: [999; 3],
            ..inactive
        },
        DensityBrush {
            inner_radius_mm: 9,
            ..inactive
        },
        DensityBrush {
            outer_radius_mm: 19,
            ..inactive
        },
        DensityBrush {
            opacity: 1,
            ..inactive
        },
        DensityBrush {
            density: DensityBrushOp::Replace(0),
            ..inactive
        },
        DensityBrush {
            species: None,
            ..inactive
        },
    ] {
        let other = Fixture::new(field(config, vec![changed]));
        let output = other.parity(&mut other.scratch());
        assert_eq!(rows(&output), rows(&baseline));
        assert_ne!(output.cache_key, baseline.cache_key);
    }
    let mut configs = vec![
        ScatterFieldConfig {
            seed: config.seed ^ 1,
            ..config
        },
        ScatterFieldConfig {
            version: config.version + 1,
            ..config
        },
        ScatterFieldConfig {
            base_density: config.base_density - 1,
            ..config
        },
        ScatterFieldConfig {
            base_species: config.base_species + 1,
            ..config
        },
        ScatterFieldConfig {
            noise: None,
            ..config
        },
    ];
    for changed in [
        DensityNoise {
            cell_size_mm: 8,
            ..disabled
        },
        DensityNoise {
            amplitude: 1,
            ..disabled
        },
        DensityNoise {
            warp_cell_size_mm: 12,
            ..disabled
        },
        DensityNoise {
            warp_amplitude_mm: 1,
            ..disabled
        },
    ] {
        configs.push(ScatterFieldConfig {
            noise: Some(changed),
            ..config
        });
    }
    for changed in configs {
        let other = Fixture::new(field(changed, vec![inactive]));
        assert_ne!(
            other.parity(&mut other.scratch()).cache_key,
            baseline.cache_key
        );
    }
}

#[test]
fn composed_deletions_use_full_keys_and_preserve_original_rows_cache_and_both_vecs() {
    let case = Fixture::new(painted_field());
    let mut scratch = case.scratch();
    let baseline = case.parity(&mut scratch);
    assert!(baseline.points.len() > 3);
    let key = baseline.points[0].key;
    let mut near = [key; 7];
    near[0].seed ^= 1;
    near[1].version += 1;
    near[2].salt ^= 1;
    near[3].grid_spacing_mm += 1;
    near[4].grid.x += 1;
    near[5].grid.y += 1;
    near[6].grid.z += 1;
    let deletion_set = |keys: Vec<ScatterCandidateKey>| {
        let budget = ScatterDeletionBudget {
            max_keys: keys.len(),
            max_key_bytes: keys.capacity() * size_of::<ScatterCandidateKey>(),
        };
        ScatterDeletionSet::new(keys, budget).unwrap()
    };
    // A valid near-match is not a deletion of the original full candidate key.
    for alternative in near {
        alternative.rebuild().unwrap();
        let mut one = case.indexed(&mut scratch, Limits::default()).unwrap();
        one.points.truncate(1);
        one.attributes.truncate(1);
        let result = filter_heightfield_scatter(one, &deletion_set(vec![alternative])).unwrap();
        assert_eq!(result.removed, 0);
        assert_eq!(rows(&result.source), rows(&baseline)[..1]);
    }
    for mode in 0..3 {
        let original = case.indexed(&mut scratch, Limits::default()).unwrap();
        let pointers = (original.points.as_ptr(), original.attributes.as_ptr());
        let capacities = (original.points.capacity(), original.attributes.capacity());
        let source_key = original.cache_key.clone();
        let counters = (
            original.candidate_evaluations,
            original.field_work,
            original.surface_sample_reads,
        );
        let mut absent = key;
        absent.grid.x += 10_000;
        let mut keys: Vec<_> = original
            .points
            .iter()
            .enumerate()
            .filter(|(i, _)| mode == 2 || (mode == 1 && i % 3 == 1))
            .map(|(_, p)| p.key)
            .collect();
        keys.push(absent);
        let deletions = deletion_set(keys);
        let expected: Vec<_> = rows(&original)
            .into_iter()
            .filter(|(p, _)| !deletions.is_deleted(p.key))
            .collect();
        let result = filter_heightfield_scatter(original, &deletions).unwrap();
        assert_eq!(rows(&result.source), expected);
        assert_eq!(result.removed, baseline.points.len() - expected.len());
        assert_eq!(result.source.cache_key, source_key);
        assert_eq!(result.deletions, deletions);
        assert_eq!(result.deletions.keys().as_ptr(), deletions.keys().as_ptr());
        assert!(result.deletions.is_deleted(absent));
        assert_eq!(
            (
                result.source.points.as_ptr(),
                result.source.attributes.as_ptr()
            ),
            pointers
        );
        assert_eq!(
            (
                result.source.points.capacity(),
                result.source.attributes.capacity()
            ),
            capacities
        );
        assert_eq!(
            (
                result.source.candidate_evaluations,
                result.source.field_work,
                result.source.surface_sample_reads
            ),
            counters
        );
        assert!(Arc::ptr_eq(
            result.source.cache_key.surface.tile(),
            case.surface.tile()
        ));
    }
    assert_eq!(
        case.indexed(&mut scratch, Limits::default()).unwrap(),
        baseline
    );
}

#[test]
fn scratch_reuses_across_indices_requests_and_failures_without_capacity_growth() {
    let painted = Fixture::new(painted_field());
    let empty = Fixture::new(field(field_config(), vec![]));
    let single = Fixture::new(field(field_config(), vec![brush(1)]));
    let mut scratch = painted.scratch();
    let before = (scratch.capacity(), scratch.allocated_bytes());
    for _ in 0..3 {
        for case in [&painted, &empty, &single, &painted] {
            let baseline = case.parity(&mut scratch);
            assert_error(
                case.indexed(
                    &mut scratch,
                    Limits {
                        field: FieldScatterBudget {
                            max_field_work: 0,
                            ..Limits::default().field
                        },
                        ..Limits::default()
                    },
                )
                .unwrap_err(),
                HeightfieldScatterError::Index(ScatterBrushIndexError::WorkBudgetExceeded {
                    required: estimate_scatter_work(case.config, case.region)
                        .unwrap()
                        .primary_candidates
                        * case.index.max_sample_work(),
                    limit: 0,
                }),
            );
            assert_eq!(
                case.indexed(&mut scratch, Limits::default()).unwrap(),
                baseline
            );
            assert_eq!(before, (scratch.capacity(), scratch.allocated_bytes()));
        }
    }
}
