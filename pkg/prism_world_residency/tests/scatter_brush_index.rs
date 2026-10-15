//! Public-contract tests: the linear field is the oracle, never BVH internals.

use std::mem::size_of;

use prism_world_residency::{
    estimate_scatter_work, scatter, scatter_with_density_field, scatter_with_indexed_density_field,
    DensityBrush, DensityBrushOp, DensityNoise, FieldScatterBudget, FieldScatterOutput,
    IndexedFieldSample, ScatterBrushIndex, ScatterBrushIndexBudget, ScatterBrushIndexError,
    ScatterBrushScratch, ScatterBudget, ScatterConfig, ScatterDensityField, ScatterError,
    ScatterFieldBuildBudget, ScatterFieldConfig, ScatterFieldError, ScatterFieldSample,
    ScatterPoint, ScatterRegion,
};

fn field_config() -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 0x9abc_def0_1234_5678,
        version: 19,
        base_density: 29_731,
        base_species: 87,
        noise: None,
    }
}

fn brush(id: u64) -> DensityBrush {
    DensityBrush {
        id,
        center_mm: [0; 3],
        inner_radius_mm: 20,
        outer_radius_mm: 100,
        opacity: u16::MAX,
        density: DensityBrushOp::Replace(55_555),
        species: Some(id as u32),
    }
}

fn field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    let budget = ScatterFieldBuildBudget {
        max_brushes: brushes.len(),
        max_brush_bytes: brushes.capacity() * size_of::<DensityBrush>(),
    };
    ScatterDensityField::new(config, brushes, budget).unwrap()
}

fn index_budget() -> ScatterBrushIndexBudget {
    ScatterBrushIndexBudget {
        max_brushes: usize::MAX,
        max_index_bytes: usize::MAX,
    }
}

fn index(field: ScatterDensityField) -> ScatterBrushIndex {
    ScatterBrushIndex::new(field, index_budget()).unwrap()
}

fn sample_parity(
    index: &ScatterBrushIndex,
    scratch: &mut ScatterBrushScratch,
    position: [i64; 3],
) -> IndexedFieldSample {
    let before = (scratch.capacity(), scratch.allocated_bytes());
    let actual = index
        .sample(position, scratch, index.max_sample_work())
        .unwrap();
    assert_eq!(
        actual.sample,
        index.field().sample(position),
        "{position:?}"
    );
    assert!(actual.node_tests <= index.node_count() as u64);
    assert!(actual.brush_tests <= index.field().brushes().len() as u64);
    assert!(actual.field_work <= index.max_sample_work());
    assert!(actual.field_work > 0);
    assert_eq!(before, (scratch.capacity(), scratch.allocated_bytes()));
    actual
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        seed: 0xfedc_ba98_7654_3210,
        version: 3,
        salt: 711,
        grid_spacing_mm: 17,
        min_distance_mm: 0,
    }
}

fn source_budget() -> ScatterBudget {
    ScatterBudget {
        max_candidate_evaluations: 10_000_000,
        max_output_points: 100_000,
        max_output_bytes: 16 * 1024 * 1024,
    }
}

fn field_budget() -> FieldScatterBudget {
    FieldScatterBudget {
        max_field_work: u64::MAX,
        max_output_bytes: 32 * 1024 * 1024,
    }
}

fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min, max).unwrap()
}

fn assert_output_parity(actual: &FieldScatterOutput, expected: &FieldScatterOutput) {
    assert_eq!(actual.points, expected.points);
    assert_eq!(actual.attributes, expected.attributes);
    assert_eq!(actual.cache_key, expected.cache_key);
    assert_eq!(actual.candidate_evaluations, expected.candidate_evaluations);
    assert_eq!(actual.points.len(), actual.attributes.len());
    // field_work deliberately measures a different algorithm on the two paths.
}

fn scatter_parity(
    config: ScatterConfig,
    region: ScatterRegion,
    index: &ScatterBrushIndex,
    scratch: &mut ScatterBrushScratch,
) -> FieldScatterOutput {
    let expected = scatter_with_density_field(
        config,
        region,
        source_budget(),
        index.field(),
        field_budget(),
    )
    .unwrap();
    let actual = scatter_with_indexed_density_field(
        config,
        region,
        source_budget(),
        index,
        scratch,
        field_budget(),
    )
    .unwrap();
    assert_output_parity(&actual, &expected);

    let source = scatter(config, region, source_budget()).unwrap();
    let mut expected_work = 0;
    for point in &source.points {
        expected_work += sample_parity(index, scratch, point.position_mm).field_work;
    }
    assert_eq!(actual.field_work, expected_work);
    assert!(actual.field_work <= source.points.len() as u64 * index.max_sample_work());
    actual
}

// Fixed seed and explicit wrapping make the corpus replayable without rand.
struct Corpus(u64);

impl Corpus {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 ^ (self.0 >> 29)
    }

    fn coordinate(&mut self, extent: u64) -> i64 {
        (self.next() % (2 * extent + 1)) as i64 - extent as i64
    }
}

fn mixed_brushes() -> Vec<DensityBrush> {
    let mut rng = Corpus(0x048c_217e_feda_53b9);
    (0..96_u64)
        .map(|i| {
            let outer = 1 + (rng.next() % 401) as u32;
            DensityBrush {
                // A permutation with IDs that differ in their high bits, too.
                id: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                center_mm: [
                    rng.coordinate(300),
                    rng.coordinate(300),
                    rng.coordinate(300),
                ],
                inner_radius_mm: (rng.next() % (u64::from(outer) + 1)) as u32,
                outer_radius_mm: outer,
                opacity: match i % 7 {
                    0 => 0,
                    1 => u16::MAX,
                    2 => 32_767,
                    3 => 32_768,
                    _ => rng.next() as u16,
                },
                density: match i % 5 {
                    0 => DensityBrushOp::Replace(rng.next() as u16),
                    1 => DensityBrushOp::Add(65_535),
                    2 => DensityBrushOp::Add(-65_535),
                    3 => DensityBrushOp::Add(0),
                    _ => DensityBrushOp::Add((rng.next() % 131_071) as i32 - 65_535),
                },
                species: match i % 4 {
                    0 => None,
                    1 => Some(0),
                    2 => Some(u32::MAX),
                    _ => Some(rng.next() as u32),
                },
            }
        })
        .collect()
}

#[test]
fn canonical_id_order_preserves_noncommutative_density_and_species() {
    let edits = [
        (10, DensityBrushOp::Replace(60_000), Some(1)),
        (20, DensityBrushOp::Add(20_000), Some(2)),
        (30, DensityBrushOp::Replace(1_000), Some(3)),
        (40, DensityBrushOp::Add(-3_000), None),
        (u64::MAX, DensityBrushOp::Add(12_345), None),
    ];
    let brushes: Vec<_> = edits
        .into_iter()
        .enumerate()
        .map(|(i, (id, density, species))| DensityBrush {
            center_mm: [(2 - i as i64) * 30, 0, 0],
            inner_radius_mm: 400,
            outer_radius_mm: 500,
            density,
            species,
            ..brush(id)
        })
        .collect();
    let forward = field(field_config(), brushes.clone());
    let mut reversed = brushes.clone();
    reversed.reverse();
    let reversed = field(field_config(), reversed);
    let mut permuted = brushes;
    permuted.rotate_left(2);
    let permuted = field(field_config(), permuted);
    assert_eq!(forward, reversed);
    assert_eq!(forward, permuted);
    let expected = ScatterFieldSample {
        density: 12_345,
        species: 3,
    };
    assert_eq!(forward.sample([0; 3]), expected);

    let indexes = [index(forward.clone()), index(reversed), index(permuted)];
    for other in &indexes {
        assert_eq!(other.field(), &forward);
        assert_eq!(other.node_count(), indexes[0].node_count());
        assert_eq!(other.allocated_bytes(), indexes[0].allocated_bytes());
        assert_eq!(other.max_sample_work(), indexes[0].max_sample_work());
    }
    let mut scratches: Vec<_> = indexes
        .iter()
        .map(|index| index.new_scratch(usize::MAX).unwrap())
        .collect();
    for position in [[0; 3], [29, -5, 7], [-999; 3], [499, 0, 0]] {
        let reference = sample_parity(&indexes[0], &mut scratches[0], position);
        for i in 1..indexes.len() {
            let actual = sample_parity(&indexes[i], &mut scratches[i], position);
            assert_eq!(actual.sample, reference.sample);
            assert_eq!(actual.node_tests, reference.node_tests);
            assert_eq!(actual.brush_tests, reference.brush_tests);
            assert_eq!(actual.field_work, reference.field_work);
        }
    }
}

#[test]
fn randomized_overlap_noise_and_warp_match_the_complete_baseline_sample() {
    let brushes = mixed_brushes();
    let noise_modes = [
        None,
        Some(DensityNoise {
            cell_size_mm: 17,
            amplitude: 0,
            warp_cell_size_mm: 13,
            warp_amplitude_mm: u32::MAX,
        }),
        Some(DensityNoise {
            cell_size_mm: 17,
            amplitude: 31_337,
            warp_cell_size_mm: 13,
            warp_amplitude_mm: 0,
        }),
        Some(DensityNoise {
            cell_size_mm: 17,
            amplitude: u16::MAX,
            warp_cell_size_mm: 13,
            warp_amplitude_mm: 67,
        }),
        Some(DensityNoise {
            cell_size_mm: 1,
            amplitude: 49_999,
            warp_cell_size_mm: u32::MAX,
            warp_amplitude_mm: u32::MAX,
        }),
    ];
    let mut positions = vec![[0; 3], [-1; 3], [i64::MIN; 3], [i64::MAX; 3]];
    let mut rng = Corpus(0x76bb_aa99_0012_3456);
    for _ in 0..400 {
        positions.push([
            rng.coordinate(800),
            rng.coordinate(800),
            rng.coordinate(800),
        ]);
    }
    for brush in &brushes {
        positions.push(brush.center_mm);
        for axis in 0..3 {
            for offset in [-1_i64, 0, 1] {
                let mut position = brush.center_mm;
                position[axis] += i64::from(brush.outer_radius_mm) + offset;
                positions.push(position);
            }
        }
    }
    for noise in noise_modes {
        let original = field(
            ScatterFieldConfig {
                noise,
                ..field_config()
            },
            brushes.clone(),
        );
        let mut reverse = brushes.clone();
        reverse.reverse();
        let reversed = index(field(original.config(), reverse));
        let original = index(original);
        let mut scratch = original.new_scratch(usize::MAX).unwrap();
        for &position in &positions {
            let first = sample_parity(&original, &mut scratch, position);
            let second = sample_parity(&reversed, &mut scratch, position);
            assert_eq!(first.sample, second.sample);
            assert_eq!(first.node_tests, second.node_tests);
            assert_eq!(first.brush_tests, second.brush_tests);
            assert_eq!(first.field_work, second.field_work);
        }
    }
}

#[test]
fn integer_world_edges_and_maximum_radius_do_not_wrap_or_clip() {
    let radius = i64::from(u32::MAX);
    let centers = [
        [i64::MIN; 3],
        [i64::MAX; 3],
        [i64::MIN, 0, i64::MAX],
        [i64::MAX, i64::MIN, 0],
        [0; 3],
    ];
    let brushes: Vec<_> = centers
        .into_iter()
        .enumerate()
        .map(|(i, center_mm)| DensityBrush {
            center_mm,
            inner_radius_mm: if i % 2 == 0 { u32::MAX } else { 0 },
            outer_radius_mm: u32::MAX,
            density: if i % 2 == 0 {
                DensityBrushOp::Replace(60_001)
            } else {
                DensityBrushOp::Add(-54_321)
            },
            species: Some(i as u32 + 100),
            ..brush(i as u64)
        })
        .collect();
    for noise in [
        None,
        Some(DensityNoise {
            cell_size_mm: u32::MAX,
            amplitude: u16::MAX,
            warp_cell_size_mm: 1,
            warp_amplitude_mm: u32::MAX,
        }),
    ] {
        let index = index(field(
            ScatterFieldConfig {
                noise,
                ..field_config()
            },
            brushes.clone(),
        ));
        let mut scratch = index.new_scratch(usize::MAX).unwrap();
        for center in centers {
            sample_parity(&index, &mut scratch, center);
            for axis in 0..3 {
                for offset in [
                    -radius - 1,
                    -radius,
                    -radius + 1,
                    -1,
                    1,
                    radius - 1,
                    radius,
                    radius + 1,
                ] {
                    let mut position = center;
                    if let Some(value) = center[axis].checked_add(offset) {
                        position[axis] = value;
                        sample_parity(&index, &mut scratch, position);
                    }
                }
            }
        }
        for x in [i64::MIN, -1, 0, 1, i64::MAX] {
            for y in [i64::MIN, 0, i64::MAX] {
                for z in [i64::MIN, 0, i64::MAX] {
                    sample_parity(&index, &mut scratch, [x, y, z]);
                }
            }
        }
    }
    // Opposite-world-corner distance cannot overflow its widened accumulator.
    let isolated = index(field(
        field_config(),
        vec![DensityBrush {
            center_mm: [i64::MIN; 3],
            inner_radius_mm: u32::MAX,
            outer_radius_mm: u32::MAX,
            ..brush(1)
        }],
    ));
    let mut scratch = isolated.new_scratch(usize::MAX).unwrap();
    let untouched = sample_parity(&isolated, &mut scratch, [i64::MAX; 3]);
    assert_eq!(untouched.sample.density, field_config().base_density);
    assert_eq!(untouched.sample.species, field_config().base_species);
}

#[test]
fn species_threshold_and_hard_outer_boundary_are_exact() {
    for opacity in [0, 1, 32_767, 32_768, u16::MAX] {
        let index = index(field(
            field_config(),
            vec![DensityBrush {
                inner_radius_mm: 10,
                outer_radius_mm: 10,
                opacity,
                density: DensityBrushOp::Add(0),
                species: Some(u32::MAX),
                ..brush(1)
            }],
        ));
        let mut scratch = index.new_scratch(usize::MAX).unwrap();
        for position in [
            [0; 3],
            [9, 0, 0],
            [10, 0, 0],
            [-10, 0, 0],
            [6, 8, 0],
            [11, 0, 0],
        ] {
            let actual = sample_parity(&index, &mut scratch, position);
            let inside = position == [0; 3] || position == [9, 0, 0];
            assert_eq!(actual.sample.density, field_config().base_density);
            assert_eq!(
                actual.sample.species,
                if inside && opacity >= 32_768 {
                    u32::MAX
                } else {
                    field_config().base_species
                },
            );
        }
    }
}

#[test]
fn empty_index_requires_no_payload_and_still_samples_noise() {
    for noise in [
        None,
        Some(DensityNoise {
            cell_size_mm: 7,
            amplitude: 32_000,
            warp_cell_size_mm: 11,
            warp_amplitude_mm: 19,
        }),
    ] {
        let field = field(
            ScatterFieldConfig {
                noise,
                ..field_config()
            },
            vec![],
        );
        let expected_work = field.work_per_sample();
        let index = ScatterBrushIndex::new(
            field,
            ScatterBrushIndexBudget {
                max_brushes: 0,
                max_index_bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(index.node_count(), 0);
        assert_eq!(index.allocated_bytes(), 0);
        assert_eq!(index.max_sample_work(), expected_work);
        let mut scratch = index.new_scratch(0).unwrap();
        assert_eq!(scratch.capacity(), 0);
        assert_eq!(scratch.allocated_bytes(), 0);
        for position in [[i64::MIN; 3], [-1; 3], [0; 3], [i64::MAX; 3]] {
            let actual = sample_parity(&index, &mut scratch, position);
            assert_eq!(actual.node_tests, 0);
            assert_eq!(actual.brush_tests, 0);
            assert_eq!(actual.field_work, expected_work);
            assert_eq!(
                index
                    .sample(position, &mut scratch, expected_work - 1)
                    .unwrap_err(),
                ScatterBrushIndexError::WorkBudgetExceeded {
                    required: expected_work,
                    limit: expected_work - 1,
                },
            );
        }
    }
}

#[test]
fn sparse_query_is_pruned_and_work_budget_uses_actual_probes() {
    let brushes: Vec<_> = (0..257)
        .map(|i| DensityBrush {
            center_mm: [i * 10_000, (i % 3) * 1000, 0],
            inner_radius_mm: 5,
            outer_radius_mm: 10,
            ..brush(i as u64)
        })
        .collect();
    let index = index(field(field_config(), brushes));
    assert!(index.node_count() > 1);
    let mut scratch = index.new_scratch(usize::MAX).unwrap();
    let position = [-1_000_000; 3];
    let outside = sample_parity(&index, &mut scratch, position);
    assert_eq!(outside.brush_tests, 0);
    assert!(outside.node_tests > 0);
    assert!(outside.node_tests < index.node_count() as u64);
    assert!(outside.field_work < index.max_sample_work());
    assert!(outside.field_work < index.field().work_per_sample());
    let exact = index
        .sample(position, &mut scratch, outside.field_work)
        .unwrap();
    assert_eq!(exact.sample, outside.sample);
    assert_eq!(exact.field_work, outside.field_work);
    assert_eq!(
        index
            .sample(position, &mut scratch, outside.field_work - 1)
            .unwrap_err(),
        ScatterBrushIndexError::WorkBudgetExceeded {
            required: outside.field_work,
            limit: outside.field_work - 1,
        },
    );
    for position in [
        [0; 3],
        [1_280_000, 2000, 0],
        [2_560_010, 1000, 0],
        [7, 7, 7],
    ] {
        let actual = sample_parity(&index, &mut scratch, position);
        let exact = index
            .sample(position, &mut scratch, actual.field_work)
            .unwrap();
        assert_eq!(actual.sample, exact.sample);
        assert_eq!(actual.node_tests, exact.node_tests);
        assert_eq!(actual.brush_tests, exact.brush_tests);
        assert_eq!(actual.field_work, exact.field_work);
        assert_eq!(
            index
                .sample(position, &mut scratch, actual.field_work - 1)
                .unwrap_err(),
            ScatterBrushIndexError::WorkBudgetExceeded {
                required: actual.field_work,
                limit: actual.field_work - 1,
            },
        );
        sample_parity(&index, &mut scratch, position);
    }
}

#[test]
fn index_and_scratch_payload_budgets_accept_exact_limits_and_reject_one_less() {
    let field = field(field_config(), mixed_brushes());
    let count = field.brushes().len();
    assert_eq!(
        ScatterBrushIndex::new(
            field.clone(),
            ScatterBrushIndexBudget {
                max_brushes: count - 1,
                max_index_bytes: usize::MAX,
            },
        )
        .unwrap_err(),
        ScatterBrushIndexError::BrushBudgetExceeded {
            required: count,
            limit: count - 1,
        },
    );
    let generous = index(field.clone());
    let bytes = generous.allocated_bytes();
    assert!(bytes > 0);
    let exact = ScatterBrushIndex::new(
        field.clone(),
        ScatterBrushIndexBudget {
            max_brushes: count,
            max_index_bytes: bytes,
        },
    )
    .unwrap();
    assert_eq!(exact.field(), &field);
    assert_eq!(exact.allocated_bytes(), bytes);
    assert_eq!(exact.node_count(), generous.node_count());
    assert_eq!(
        ScatterBrushIndex::new(
            field,
            ScatterBrushIndexBudget {
                max_brushes: count,
                max_index_bytes: bytes - 1,
            },
        )
        .unwrap_err(),
        ScatterBrushIndexError::IndexBytesBudgetExceeded {
            required: bytes,
            limit: bytes - 1,
        },
    );

    let scratch = exact.new_scratch(usize::MAX).unwrap();
    assert!(scratch.capacity() >= count);
    let scratch_bytes = scratch.allocated_bytes();
    assert!(scratch_bytes > 0);
    assert_eq!(
        exact.new_scratch(scratch_bytes - 1).unwrap_err(),
        ScatterBrushIndexError::ScratchBytesBudgetExceeded {
            required: scratch_bytes,
            limit: scratch_bytes - 1,
        },
    );
    let mut scratch = exact.new_scratch(scratch_bytes).unwrap();
    assert_eq!(scratch.allocated_bytes(), scratch_bytes);
    for position in [[0; 3], [200; 3], [i64::MIN; 3], [0; 3]] {
        sample_parity(&exact, &mut scratch, position);
    }
}

#[test]
fn scratch_reuses_across_indexes_and_rejects_insufficient_capacity_without_growth() {
    let small = index(field(field_config(), vec![brush(1)]));
    let mut small_scratch = small.new_scratch(usize::MAX).unwrap();
    let before = (small_scratch.capacity(), small_scratch.allocated_bytes());
    let required = before.0 + 7;
    let large = index(field(
        ScatterFieldConfig {
            base_species: u32::MAX,
            ..field_config()
        },
        (0..required)
            .map(|i| DensityBrush {
                center_mm: [i as i64, 0, 0],
                density: DensityBrushOp::Replace(i as u16),
                ..brush(i as u64)
            })
            .collect(),
    ));
    for position in [[0; 3], [i64::MAX; 3]] {
        // Capacity is a hard contract even when this query would hit no brush.
        assert_eq!(
            large
                .sample(position, &mut small_scratch, large.max_sample_work())
                .unwrap_err(),
            ScatterBrushIndexError::ScratchCapacityExceeded {
                required,
                limit: before.0,
            },
        );
        assert_eq!(
            before,
            (small_scratch.capacity(), small_scratch.allocated_bytes())
        );
        sample_parity(&small, &mut small_scratch, [0; 3]);
    }
    // Check capacity before work so an inadequate buffer never starts sampling.
    assert_eq!(
        large.sample([0; 3], &mut small_scratch, 0).unwrap_err(),
        ScatterBrushIndexError::ScratchCapacityExceeded {
            required,
            limit: before.0,
        },
    );
    let empty = index(field(field_config(), vec![]));
    let mut scratch = large.new_scratch(usize::MAX).unwrap();
    let large_before = (scratch.capacity(), scratch.allocated_bytes());
    for _ in 0..3 {
        for index in [&large, &small, &empty, &small, &large] {
            for position in [[0; 3], [i64::MAX; 3], [7, 1, -3], [0; 3]] {
                sample_parity(index, &mut scratch, position);
            }
        }
    }
    assert_eq!(
        large_before,
        (scratch.capacity(), scratch.allocated_bytes())
    );
}

#[test]
fn inactive_brushes_and_disabled_noise_remain_structural_cache_dependencies() {
    let inactive = DensityBrush {
        opacity: 0,
        density: DensityBrushOp::Replace(0),
        species: Some(u32::MAX),
        ..brush(42)
    };
    let disabled_noise = DensityNoise {
        cell_size_mm: 13,
        amplitude: 0,
        warp_cell_size_mm: 17,
        warp_amplitude_mm: 0,
    };
    let fields = [
        field(field_config(), vec![]),
        field(field_config(), vec![inactive]),
        field(field_config(), vec![DensityBrush { id: 43, ..inactive }]),
        field(
            field_config(),
            vec![DensityBrush {
                center_mm: [999; 3],
                ..inactive
            }],
        ),
        field(
            ScatterFieldConfig {
                noise: Some(disabled_noise),
                ..field_config()
            },
            vec![],
        ),
        field(
            ScatterFieldConfig {
                noise: Some(DensityNoise {
                    warp_amplitude_mm: u32::MAX,
                    ..disabled_noise
                }),
                ..field_config()
            },
            vec![],
        ),
    ];
    let request = region([-34; 3], [34; 3]);
    let mut outputs = Vec::new();
    for field in fields {
        let index = index(field.clone());
        assert_eq!(index.field(), &field);
        let mut scratch = index.new_scratch(usize::MAX).unwrap();
        for position in [[0; 3], [999; 3], [i64::MIN; 3], [i64::MAX; 3]] {
            let sampled = sample_parity(&index, &mut scratch, position);
            assert_eq!(
                sampled.sample,
                ScatterFieldSample {
                    density: field_config().base_density,
                    species: field_config().base_species,
                }
            );
        }
        outputs.push(scatter_parity(
            scatter_config(),
            request,
            &index,
            &mut scratch,
        ));
    }
    assert!(!outputs[0].points.is_empty());
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(output.points, outputs[0].points);
        assert_eq!(output.attributes, outputs[0].attributes);
        for other in &outputs[..i] {
            assert_ne!(output.cache_key, other.cache_key);
        }
    }
}

#[test]
fn scatter_matches_all_observable_baseline_results_with_and_without_min_distance() {
    for noise in [
        None,
        Some(DensityNoise {
            cell_size_mm: 23,
            amplitude: 29_111,
            warp_cell_size_mm: 19,
            warp_amplitude_mm: 37,
        }),
    ] {
        let index = index(field(
            ScatterFieldConfig {
                noise,
                ..field_config()
            },
            mixed_brushes(),
        ));
        let mut scratch = index.new_scratch(usize::MAX).unwrap();
        for min_distance_mm in [0, 19, 35] {
            let config = ScatterConfig {
                min_distance_mm,
                ..scatter_config()
            };
            for request in [
                region([-68; 3], [68; 3]),
                region([-49, -21, -33], [59, 47, 42]),
            ] {
                let output = scatter_parity(config, request, &index, &mut scratch);
                assert!(
                    !output.points.is_empty(),
                    "exercise attributes, not only empty parity"
                );
                for point in &output.points {
                    assert!(request.contains_mm(point.position_mm));
                }
                let min_squared = i128::from(min_distance_mm).pow(2);
                for (i, point) in output.points.iter().enumerate() {
                    for other in &output.points[..i] {
                        let distance: i128 = point
                            .position_mm
                            .into_iter()
                            .zip(other.position_mm)
                            .map(|(a, b)| (i128::from(a) - i128::from(b)).pow(2))
                            .sum();
                        assert!(distance >= min_squared);
                    }
                }
            }
        }
    }
}

#[test]
fn partitioned_reordered_scatter_has_identical_points_and_aligned_attributes() {
    let index = index(field(
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                cell_size_mm: 19,
                amplitude: 27_000,
                warp_cell_size_mm: 11,
                warp_amplitude_mm: 41,
            }),
            ..field_config()
        },
        mixed_brushes(),
    ));
    let mut scratch = index.new_scratch(usize::MAX).unwrap();
    for min_distance_mm in [0, 23] {
        let config = ScatterConfig {
            min_distance_mm,
            ..scatter_config()
        };
        let whole = scatter_parity(config, region([-51; 3], [51; 3]), &index, &mut scratch);
        let mut expected: Vec<_> = whole
            .points
            .iter()
            .copied()
            .zip(whole.attributes.iter().copied())
            .collect();
        expected.sort_unstable_by_key(|(point, _)| point.key);
        let mut parts = Vec::new();
        // Neither the split at -7 nor 13 coincides with the global 17-mm grid.
        for (x0, x1) in [(-51, -7), (-7, 51)] {
            for (y0, y1) in [(-51, 13), (13, 51)] {
                for (z0, z1) in [(-51, -1), (-1, 51)] {
                    parts.push(region([x0, y0, z0], [x1, y1, z1]));
                }
            }
        }
        parts.reverse();
        let mut actual = Vec::new();
        for part in parts {
            let output = scatter_parity(config, part, &index, &mut scratch);
            actual.extend(output.points.into_iter().zip(output.attributes));
        }
        actual.sort_unstable_by_key(|(point, _)| point.key);
        assert!(!expected.is_empty());
        assert_eq!(actual, expected);
        assert!(actual.windows(2).all(|pair| pair[0].0.key != pair[1].0.key));
        // Work and candidate counts need not be additive: partition halos overlap.
    }
}

#[test]
fn scatter_preflights_primary_times_max_work_before_source_generation() {
    let index = index(field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            opacity: 0,
            ..brush(1)
        }],
    ));
    let mut scratch = index.new_scratch(usize::MAX).unwrap();
    let request = region([-34; 3], [34; 3]);
    let config = ScatterConfig {
        min_distance_mm: 31,
        ..scatter_config()
    };
    let primary = estimate_scatter_work(config, request)
        .unwrap()
        .primary_candidates;
    let required = primary * index.max_sample_work();
    let denied_source = ScatterBudget {
        max_candidate_evaluations: 0,
        max_output_points: 0,
        max_output_bytes: 0,
    };
    assert_eq!(
        scatter_with_indexed_density_field(
            config,
            request,
            denied_source,
            &index,
            &mut scratch,
            FieldScatterBudget {
                max_field_work: required - 1,
                max_output_bytes: 0
            },
        )
        .unwrap_err(),
        ScatterBrushIndexError::WorkBudgetExceeded {
            required,
            limit: required - 1
        },
    );
    // At the exact preflight limit, execution reaches the independent source budget.
    let estimate = estimate_scatter_work(config, request).unwrap();
    assert_eq!(
        scatter_with_indexed_density_field(
            config,
            request,
            denied_source,
            &index,
            &mut scratch,
            FieldScatterBudget {
                max_field_work: required,
                ..field_budget()
            },
        )
        .unwrap_err(),
        ScatterBrushIndexError::Field(ScatterFieldError::Scatter(
            ScatterError::CandidateBudgetExceeded {
                required: estimate.max_candidate_evaluations,
                limit: 0,
            },
        )),
    );
    let output = scatter_with_indexed_density_field(
        config,
        request,
        source_budget(),
        &index,
        &mut scratch,
        FieldScatterBudget {
            max_field_work: required,
            ..field_budget()
        },
    )
    .unwrap();
    assert!(output.points.is_empty());
    assert!(output.attributes.is_empty());
    assert!(
        output.field_work > 0,
        "density-rejected source survivors are still sampled"
    );
    assert!(output.field_work <= required);
}

#[test]
fn work_preflight_charges_primary_candidates_even_when_region_has_no_survivor() {
    let config = scatter_config();
    let cell = scatter(config, region([0; 3], [17; 3]), source_budget()).unwrap();
    assert_eq!(cell.points.len(), 1);
    let corner = if cell.points[0].position_mm == [0; 3] {
        [1; 3]
    } else {
        [0; 3]
    };
    let request = region(corner, corner.map(|x| x + 1));
    assert!(scatter(config, request, source_budget())
        .unwrap()
        .points
        .is_empty());
    assert_eq!(
        estimate_scatter_work(config, request)
            .unwrap()
            .primary_candidates,
        1
    );
    let index = index(field(field_config(), mixed_brushes()));
    let mut scratch = index.new_scratch(usize::MAX).unwrap();
    let required = index.max_sample_work();
    assert_eq!(
        scatter_with_indexed_density_field(
            config,
            request,
            source_budget(),
            &index,
            &mut scratch,
            FieldScatterBudget {
                max_field_work: required - 1,
                ..field_budget()
            },
        )
        .unwrap_err(),
        ScatterBrushIndexError::WorkBudgetExceeded {
            required,
            limit: required - 1
        },
    );
    let output = scatter_with_indexed_density_field(
        config,
        request,
        source_budget(),
        &index,
        &mut scratch,
        FieldScatterBudget {
            max_field_work: required,
            ..field_budget()
        },
    )
    .unwrap();
    assert!(output.points.is_empty());
    assert!(output.attributes.is_empty());
    assert_eq!(output.field_work, 0);
}

#[test]
fn scatter_preflight_multiplication_overflow_is_reported_without_enumeration() {
    let index = index(field(field_config(), vec![brush(1), brush(2)]));
    let mut scratch = index.new_scratch(usize::MAX).unwrap();
    let config = ScatterConfig {
        grid_spacing_mm: 1,
        ..scatter_config()
    };
    let request = region([0; 3], [1 << 21; 3]);
    let estimate = estimate_scatter_work(config, request).unwrap();
    assert_eq!(estimate.primary_candidates, 1_u64 << 63);
    assert!(estimate
        .primary_candidates
        .checked_mul(index.max_sample_work())
        .is_none());
    assert_eq!(
        scatter_with_indexed_density_field(
            config,
            request,
            ScatterBudget {
                max_candidate_evaluations: 0,
                max_output_points: 0,
                max_output_bytes: 0
            },
            &index,
            &mut scratch,
            field_budget(),
        )
        .unwrap_err(),
        ScatterBrushIndexError::ArithmeticOverflow,
    );
}

#[test]
fn source_and_combined_output_limits_are_not_relaxed_by_field_thinning() {
    let index = index(field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![],
    ));
    let mut scratch = index.new_scratch(0).unwrap();
    let request = region([0; 3], [34; 3]);
    let source = scatter(scatter_config(), request, source_budget()).unwrap();
    assert_eq!(source.points.len(), 8);
    for limited in [
        ScatterBudget {
            max_output_points: 0,
            ..source_budget()
        },
        ScatterBudget {
            max_output_bytes: 0,
            ..source_budget()
        },
        ScatterBudget {
            max_candidate_evaluations: 0,
            ..source_budget()
        },
    ] {
        let expected = scatter(scatter_config(), request, limited).unwrap_err();
        assert_eq!(
            scatter_with_indexed_density_field(
                scatter_config(),
                request,
                limited,
                &index,
                &mut scratch,
                field_budget(),
            )
            .unwrap_err(),
            ScatterBrushIndexError::Field(ScatterFieldError::Scatter(expected)),
        );
    }
    let generous = scatter_parity(scatter_config(), request, &index, &mut scratch);
    assert!(generous.points.is_empty());
    let bytes = generous.points.capacity() * size_of::<ScatterPoint>()
        + generous.attributes.capacity() * size_of::<ScatterFieldSample>();
    assert!(
        bytes > 0,
        "budgets count retained capacity, not surviving length"
    );
    assert_eq!(
        scatter_with_indexed_density_field(
            scatter_config(),
            request,
            source_budget(),
            &index,
            &mut scratch,
            FieldScatterBudget {
                max_output_bytes: bytes - 1,
                ..field_budget()
            },
        )
        .unwrap_err(),
        ScatterBrushIndexError::Field(ScatterFieldError::OutputBytesBudgetExceeded {
            required: bytes,
            limit: bytes - 1,
        }),
    );
    let exact = scatter_with_indexed_density_field(
        scatter_config(),
        request,
        source_budget(),
        &index,
        &mut scratch,
        FieldScatterBudget {
            max_output_bytes: bytes,
            ..field_budget()
        },
    )
    .unwrap();
    assert_output_parity(&exact, &generous);
    assert_eq!(exact.field_work, generous.field_work);
}

#[test]
fn empty_scatter_region_needs_no_work_or_output_payload() {
    let index = index(field(field_config(), mixed_brushes()));
    let mut scratch = index.new_scratch(usize::MAX).unwrap();
    for request in [
        region([-50; 3], [-50, 50, 50]),
        region([i64::MAX; 3], [i64::MAX; 3]),
    ] {
        let output = scatter_with_indexed_density_field(
            scatter_config(),
            request,
            ScatterBudget {
                max_candidate_evaluations: 0,
                max_output_points: 0,
                max_output_bytes: 0,
            },
            &index,
            &mut scratch,
            FieldScatterBudget {
                max_field_work: 0,
                max_output_bytes: 0,
            },
        )
        .unwrap();
        assert!(output.points.is_empty());
        assert!(output.attributes.is_empty());
        assert_eq!(output.points.capacity(), 0);
        assert_eq!(output.attributes.capacity(), 0);
        assert_eq!(output.candidate_evaluations, 0);
        assert_eq!(output.field_work, 0);
        let expected = scatter_with_density_field(
            scatter_config(),
            request,
            source_budget(),
            index.field(),
            field_budget(),
        )
        .unwrap();
        assert_output_parity(&output, &expected);
    }
}
