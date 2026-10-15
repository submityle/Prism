//! Contract tests for field revision 1. Reference arithmetic deliberately does
//! not call the production PCG, mix, smoothstep, lerp, or value-noise helpers.

use super::*;
use crate::{SectorCoord, SCATTER_KERNEL_VERSION};
use alloc::{collections::BTreeSet, vec};

const SCALE: u128 = 65_535;

fn field_config() -> ScatterFieldConfig {
    ScatterFieldConfig {
        seed: 0x1234_5678_9abc_def0,
        version: 7,
        base_density: 32_768,
        base_species: 91,
        noise: None,
    }
}

fn noise() -> DensityNoise {
    DensityNoise {
        cell_size_mm: 17,
        amplitude: 30_000,
        warp_cell_size_mm: 13,
        warp_amplitude_mm: 0,
    }
}

fn brush(id: u64) -> DensityBrush {
    DensityBrush {
        id,
        center_mm: [0; 3],
        inner_radius_mm: 5,
        outer_radius_mm: 10,
        opacity: u16::MAX,
        density: DensityBrushOp::Replace(u16::MAX),
        species: Some(123),
    }
}

fn build_budget() -> ScatterFieldBuildBudget {
    ScatterFieldBuildBudget {
        max_brushes: usize::MAX,
        max_brush_bytes: usize::MAX,
    }
}

fn field(config: ScatterFieldConfig, brushes: Vec<DensityBrush>) -> ScatterDensityField {
    ScatterDensityField::new(config, brushes, build_budget()).unwrap()
}

fn scatter_config() -> ScatterConfig {
    ScatterConfig {
        seed: field_config().seed,
        version: 7,
        salt: 91,
        grid_spacing_mm: 17,
        min_distance_mm: 0,
    }
}

fn source_budget() -> ScatterBudget {
    ScatterBudget {
        max_candidate_evaluations: 5_000_000,
        max_output_points: 20_000,
        max_output_bytes: 20_000 * size_of::<ScatterPoint>(),
    }
}

fn execution_budget() -> FieldScatterBudget {
    FieldScatterBudget {
        max_field_work: 5_000_000,
        max_output_bytes: 20_000 * (size_of::<ScatterPoint>() + size_of::<ScatterFieldSample>()),
    }
}

fn region(min_mm: [i64; 3], max_mm: [i64; 3]) -> ScatterRegion {
    ScatterRegion::new(min_mm, max_mm).unwrap()
}

fn candidate_key() -> ScatterCandidateKey {
    ScatterCandidateKey {
        kernel_version: SCATTER_KERNEL_VERSION,
        seed: scatter_config().seed,
        version: 7,
        salt: 91,
        grid_spacing_mm: 17,
        grid: SectorCoord::new(-7, 4, 13),
    }
}

fn cache_key(field: ScatterDensityField) -> FieldScatterCacheKey {
    FieldScatterCacheKey {
        scatter: ScatterCacheKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            config: scatter_config(),
            region: region([-51; 3], [51; 3]),
        },
        field_version: SCATTER_FIELD_VERSION,
        field,
    }
}

fn sorted_pairs(output: &FieldScatterOutput) -> Vec<(ScatterPoint, ScatterFieldSample)> {
    assert_eq!(output.points.len(), output.attributes.len());
    let mut pairs: Vec<_> = output
        .points
        .iter()
        .copied()
        .zip(output.attributes.iter().copied())
        .collect();
    pairs.sort_unstable_by_key(|(point, _)| point.key);
    pairs
}

fn authored_field() -> ScatterDensityField {
    field(
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                warp_amplitude_mm: 23,
                ..noise()
            }),
            ..field_config()
        },
        vec![
            DensityBrush {
                center_mm: [-11, 4, -7],
                inner_radius_mm: 17,
                outer_radius_mm: 67,
                opacity: 47_000,
                density: DensityBrushOp::Add(-19_321),
                species: Some(0),
                ..brush(9)
            },
            DensityBrush {
                center_mm: [19, -3, 5],
                inner_radius_mm: 11,
                outer_radius_mm: 53,
                density: DensityBrushOp::Replace(58_000),
                species: Some(u32::MAX),
                ..brush(2)
            },
        ],
    )
}

// All modular LCG arithmetic is widened before the explicit 64-bit mask.
// Rotate is expressed as shifts rather than using the production operation.
fn scalar_pcg(input: u64) -> u32 {
    let mask = u128::from(u64::MAX);
    let initial = (u128::from(input) + 0x853c_49e6_748f_ea9b) & mask;
    let state = (initial * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & mask;
    let word = (((state >> 18) ^ state) >> 27) as u32;
    let rotation = (state >> 59) as u32;
    ((u64::from(word) >> rotation) | (u64::from(word) << ((32 - rotation) % 32))) as u32
}

fn scalar_mix(value: u64) -> u64 {
    u64::from(scalar_pcg(value)) * (1_u64 << 32)
        + u64::from(scalar_pcg(value ^ 0x9e37_79b9_7f4a_7c15))
}

fn scalar_smoothstep(value: u16) -> u16 {
    let t = u128::from(value);
    let denominator = SCALE * SCALE;
    ((t * t * (3 * SCALE - 2 * t) + denominator / 2) / denominator) as u16
}

fn scalar_lerp(a: u16, b: u16, weight: u16) -> u16 {
    let w = u128::from(weight);
    ((u128::from(a) * (SCALE - w) + u128::from(b) * w + SCALE / 2) / SCALE) as u16
}

fn scalar_noise(position: [i128; 3], cell_size: u32, seed: u64, channel: u64) -> u16 {
    let size = i128::from(cell_size);
    let mut grid = [0_i128; 3];
    let mut weights = [0_u16; 3];
    for axis in 0..3 {
        // Derive Euclidean quotient/remainder without div_euclid/rem_euclid.
        let mut quotient = position[axis] / size;
        let mut remainder = position[axis] % size;
        if remainder < 0 {
            quotient -= 1;
            remainder += size;
        }
        grid[axis] = quotient;
        weights[axis] =
            scalar_smoothstep((remainder as u128 * SCALE / u128::from(cell_size)) as u16);
    }
    let tag = (u128::from(channel) * 0xd1b5_4a32_d192_ed03) as u64;
    let root = scalar_mix(seed ^ tag);
    let mut values = [0_u16; 8];
    for (index, value) in values.iter_mut().enumerate() {
        let mut hash = root;
        for (axis, grid_coordinate) in grid.iter().enumerate() {
            let corner_bit = (index >> (2 - axis)) & 1;
            let coordinate = *grid_coordinate + corner_bit as i128;
            hash = scalar_mix(hash ^ coordinate as u64);
        }
        *value = (scalar_pcg(hash) >> 16) as u16;
    }
    // Pairwise reduction preserves the specified Z, then Y, then X rounding.
    // One giant eight-corner rational sum would have a different contract.
    let mut count = 8;
    for weight in weights.into_iter().rev() {
        for index in 0..count / 2 {
            values[index] = scalar_lerp(values[2 * index], values[2 * index + 1], weight);
        }
        count /= 2;
    }
    values[0]
}

fn scalar_warp(position: [i64; 3], config: ScatterFieldConfig) -> [i128; 3] {
    let origin = position.map(i128::from);
    let mut warped = origin;
    if let Some(noise) = config
        .noise
        .filter(|noise| noise.amplitude != 0 && noise.warp_amplitude_mm != 0)
    {
        for (axis, coordinate) in warped.iter_mut().enumerate() {
            let value = scalar_noise(
                origin,
                noise.warp_cell_size_mm,
                config.seed,
                axis as u64 + 1,
            );
            let signed = 2 * i128::from(value) - SCALE as i128;
            *coordinate += signed * i128::from(noise.warp_amplitude_mm) / SCALE as i128;
        }
    }
    warped
}

fn scalar_density(position: [i64; 3], config: ScatterFieldConfig) -> u16 {
    let mut value = i128::from(config.base_density);
    if let Some(noise) = config.noise.filter(|noise| noise.amplitude != 0) {
        let sample = scalar_noise(
            scalar_warp(position, config),
            noise.cell_size_mm,
            config.seed,
            0,
        );
        value +=
            (2 * i128::from(sample) - SCALE as i128) * i128::from(noise.amplitude) / SCALE as i128;
    }
    value.clamp(0, SCALE as i128) as u16
}

fn scalar_rank(seed: u64, key: ScatterCandidateKey) -> u32 {
    let words = [
        u64::from(key.kernel_version),
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.y as u64,
        key.grid.z as u64,
    ];
    let mut state = scalar_mix(seed ^ 0x6669_656c_645f_726b);
    for word in words {
        state = scalar_mix(state ^ word);
    }
    scalar_pcg(state)
}

#[test]
fn invalid_noise_cell_sizes_are_rejected_even_when_noise_and_warp_are_inactive() {
    for amplitude in [0, 1, u16::MAX] {
        for warp_amplitude_mm in [0, u32::MAX] {
            for (cell_size_mm, warp_cell_size_mm) in [(0, 13), (17, 0), (0, 0)] {
                let config = ScatterFieldConfig {
                    noise: Some(DensityNoise {
                        cell_size_mm,
                        amplitude,
                        warp_cell_size_mm,
                        warp_amplitude_mm,
                    }),
                    ..field_config()
                };
                assert_eq!(
                    ScatterDensityField::new(config, vec![], build_budget()),
                    Err(ScatterFieldError::InvalidNoiseCellSize)
                );
            }
        }
    }
}

#[test]
fn invalid_brush_radius_and_delta_are_validated_including_zero_opacity() {
    for opacity in [0, u16::MAX] {
        for (inner_radius_mm, outer_radius_mm) in [(0, 0), (1, 0), (11, 10)] {
            let stroke = DensityBrush {
                inner_radius_mm,
                outer_radius_mm,
                opacity,
                ..brush(1)
            };
            assert_eq!(
                ScatterDensityField::new(field_config(), vec![stroke], build_budget()),
                Err(ScatterFieldError::InvalidBrushRadius)
            );
        }
        for delta in [i32::MIN, -65_536, 65_536, i32::MAX] {
            let stroke = DensityBrush {
                density: DensityBrushOp::Add(delta),
                opacity,
                ..brush(1)
            };
            assert_eq!(
                ScatterDensityField::new(field_config(), vec![stroke], build_budget()),
                Err(ScatterFieldError::InvalidBrushDelta)
            );
        }
        for delta in [-65_535, 0, 65_535] {
            let stroke = DensityBrush {
                density: DensityBrushOp::Add(delta),
                opacity,
                ..brush(1)
            };
            assert!(ScatterDensityField::new(field_config(), vec![stroke], build_budget()).is_ok());
        }
    }
}

#[test]
fn duplicate_ids_are_rejected_after_sort_even_for_distinct_inactive_strokes() {
    for id in [0, 42, u64::MAX] {
        let a = DensityBrush {
            opacity: 0,
            ..brush(id)
        };
        let b = DensityBrush {
            center_mm: [i64::MAX; 3],
            species: None,
            ..a
        };
        let other = brush(if id == 0 { 1 } else { 0 });
        assert_eq!(
            ScatterDensityField::new(field_config(), vec![a, other, b], build_budget()),
            Err(ScatterFieldError::DuplicateBrushId(id))
        );
    }
    assert_eq!(
        field(field_config(), vec![brush(u64::MAX), brush(0)]).brushes()[0].id,
        0
    );
}

#[test]
fn canonical_sort_controls_noncommutative_composition_and_value_equality() {
    let replace = DensityBrush {
        density: DensityBrushOp::Replace(1_000),
        ..brush(2)
    };
    let add = DensityBrush {
        density: DensityBrushOp::Add(7),
        ..brush(9)
    };
    let a = field(field_config(), vec![add, replace]);
    let mut spare = Vec::with_capacity(19);
    spare.extend([replace, add]);
    let b = field(field_config(), spare);
    assert_eq!(a.brushes(), &[replace, add]);
    assert_eq!(a, b);
    assert_eq!(a, a.clone());
    assert_eq!(a.sample([0; 3]).density, 1_007);
    assert_eq!(cache_key(a.clone()), cache_key(b));
    let swapped = field(
        field_config(),
        vec![
            DensityBrush { id: 9, ..replace },
            DensityBrush { id: 2, ..add },
        ],
    );
    assert_eq!(swapped.sample([0; 3]).density, 1_000);
    assert_ne!(a, swapped);
}

#[test]
fn field_and_cache_equality_include_every_config_and_inactive_noise_input() {
    let inactive = DensityNoise {
        amplitude: 0,
        ..noise()
    };
    let baseline_config = ScatterFieldConfig {
        noise: Some(inactive),
        ..field_config()
    };
    let baseline = field(baseline_config, vec![]);
    assert_eq!(baseline.config(), baseline_config);
    let variants = [
        ScatterFieldConfig {
            seed: baseline_config.seed ^ (1 << 63),
            ..baseline_config
        },
        ScatterFieldConfig {
            version: 8,
            ..baseline_config
        },
        ScatterFieldConfig {
            base_density: 32_769,
            ..baseline_config
        },
        ScatterFieldConfig {
            base_species: u32::MAX,
            ..baseline_config
        },
        ScatterFieldConfig {
            noise: None,
            ..baseline_config
        },
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                cell_size_mm: 18,
                ..inactive
            }),
            ..baseline_config
        },
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                amplitude: 1,
                ..inactive
            }),
            ..baseline_config
        },
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                warp_cell_size_mm: 14,
                ..inactive
            }),
            ..baseline_config
        },
        ScatterFieldConfig {
            noise: Some(DensityNoise {
                warp_amplitude_mm: u32::MAX,
                ..inactive
            }),
            ..baseline_config
        },
    ];
    for changed in variants {
        let other = field(changed, vec![]);
        assert_ne!(baseline, other, "{changed:?}");
        assert_ne!(cache_key(baseline.clone()), cache_key(other));
    }
    assert_eq!(
        baseline.sample([i64::MAX; 3]),
        field(field_config(), vec![]).sample([i64::MAX; 3])
    );
}

#[test]
fn full_field_equality_includes_every_inactive_brush_member_and_brush_count() {
    let stroke = DensityBrush {
        opacity: 0,
        ..brush(7)
    };
    let baseline = field(field_config(), vec![stroke]);
    let variants = [
        DensityBrush { id: 8, ..stroke },
        DensityBrush {
            center_mm: [1, 0, 0],
            ..stroke
        },
        DensityBrush {
            center_mm: [0, -1, 0],
            ..stroke
        },
        DensityBrush {
            center_mm: [0, 0, i64::MIN],
            ..stroke
        },
        DensityBrush {
            inner_radius_mm: 4,
            ..stroke
        },
        DensityBrush {
            outer_radius_mm: 11,
            ..stroke
        },
        DensityBrush {
            opacity: 1,
            ..stroke
        },
        DensityBrush {
            density: DensityBrushOp::Replace(65_534),
            ..stroke
        },
        DensityBrush {
            density: DensityBrushOp::Add(0),
            ..stroke
        },
        DensityBrush {
            species: None,
            ..stroke
        },
        DensityBrush {
            species: Some(u32::MAX),
            ..stroke
        },
    ];
    for changed in variants {
        let other = field(field_config(), vec![changed]);
        assert_ne!(baseline, other, "{changed:?}");
        assert_ne!(cache_key(baseline.clone()), cache_key(other.clone()));
        assert_eq!(baseline.sample([100; 3]), other.sample([100; 3]));
    }
    assert_ne!(baseline, field(field_config(), vec![]));
    assert_ne!(
        baseline,
        field(
            field_config(),
            vec![stroke, DensityBrush { id: 8, ..stroke }]
        )
    );
}

#[test]
fn build_budgets_count_inactive_brushes_and_supplied_capacity_including_empty_vec() {
    let mut strokes = Vec::with_capacity(17);
    strokes.extend([
        DensityBrush {
            opacity: 0,
            ..brush(1)
        },
        brush(2),
    ]);
    let bytes = strokes.capacity() * size_of::<DensityBrush>();
    assert_eq!(
        ScatterDensityField::new(
            field_config(),
            strokes,
            ScatterFieldBuildBudget {
                max_brushes: 2,
                max_brush_bytes: bytes - 1
            }
        ),
        Err(ScatterFieldError::BrushBytesBudgetExceeded {
            required: bytes,
            limit: bytes - 1
        })
    );
    let mut strokes = Vec::with_capacity(17);
    strokes.extend([
        DensityBrush {
            opacity: 0,
            ..brush(1)
        },
        brush(2),
    ]);
    let bytes = strokes.capacity() * size_of::<DensityBrush>();
    assert_eq!(
        ScatterDensityField::new(
            field_config(),
            strokes,
            ScatterFieldBuildBudget {
                max_brushes: 2,
                max_brush_bytes: bytes
            }
        )
        .unwrap()
        .brushes()
        .len(),
        2
    );
    assert_eq!(
        ScatterDensityField::new(
            field_config(),
            vec![DensityBrush {
                opacity: 0,
                ..brush(1)
            }],
            ScatterFieldBuildBudget {
                max_brushes: 0,
                max_brush_bytes: 0
            }
        ),
        Err(ScatterFieldError::BrushBudgetExceeded {
            required: 1,
            limit: 0
        })
    );
    let empty = Vec::<DensityBrush>::with_capacity(11);
    let bytes = empty.capacity() * size_of::<DensityBrush>();
    assert_eq!(
        ScatterDensityField::new(
            field_config(),
            empty,
            ScatterFieldBuildBudget {
                max_brushes: 0,
                max_brush_bytes: 0
            }
        ),
        Err(ScatterFieldError::BrushBytesBudgetExceeded {
            required: bytes,
            limit: 0
        })
    );
    assert!(ScatterDensityField::new(
        field_config(),
        vec![],
        ScatterFieldBuildBudget {
            max_brushes: 0,
            max_brush_bytes: 0
        }
    )
    .is_ok());
}

#[test]
fn work_per_sample_charges_rank_all_brushes_and_only_enabled_noise_channels() {
    for (noise, expected) in [
        (None, 1),
        (
            Some(DensityNoise {
                amplitude: 0,
                warp_amplitude_mm: u32::MAX,
                ..noise()
            }),
            1,
        ),
        (Some(noise()), 9),
        (
            Some(DensityNoise {
                warp_amplitude_mm: 1,
                ..noise()
            }),
            33,
        ),
    ] {
        let config = ScatterFieldConfig {
            noise,
            ..field_config()
        };
        assert_eq!(field(config, vec![]).work_per_sample(), expected);
        assert_eq!(
            field(
                config,
                vec![
                    DensityBrush {
                        opacity: 0,
                        ..brush(1)
                    },
                    brush(2)
                ]
            )
            .work_per_sample(),
            expected + 2
        );
    }
}

#[test]
fn brush_inner_outer_endpoints_squared_falloff_and_opacity_have_exact_values() {
    let full = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![brush(1)],
    );
    let half = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            opacity: 32_768,
            ..brush(1)
        }],
    );
    // Golden values use squared distance, not a floating-point square root.
    for (position, expected, half_expected) in [
        ([0, 0, 0], 65_535, 32_768),
        ([3, 4, 0], 65_535, 32_768),
        ([6, 0, 0], 61_719, 30_860),
        ([7, 0, 0], 49_697, 24_849),
        ([8, 0, 0], 30_801, 15_401),
        ([9, 0, 0], 10_486, 5_243),
        ([10, 0, 0], 0, 0),
        ([6, 8, 0], 0, 0),
        ([11, 0, 0], 0, 0),
    ] {
        for position in [position, position.map(|v| -v)] {
            assert_eq!(full.sample(position).density, expected, "{position:?}");
            assert_eq!(half.sample(position).density, half_expected, "{position:?}");
        }
    }
}

#[test]
fn species_replacement_uses_effective_weight_32768_and_add_zero_is_species_only() {
    for (opacity, expected_species) in [(0, 91), (32_767, 91), (32_768, 123), (65_535, 123)] {
        let f = field(
            field_config(),
            vec![DensityBrush {
                opacity,
                density: DensityBrushOp::Add(0),
                ..brush(1)
            }],
        );
        assert_eq!(
            f.sample([0; 3]),
            ScatterFieldSample {
                density: 32_768,
                species: expected_species
            }
        );
        assert_eq!(
            f.sample([10, 0, 0]),
            ScatterFieldSample {
                density: 32_768,
                species: 91
            }
        );
        // At radius 8 even full opacity has effective weight only 30801.
        assert_eq!(f.sample([8, 0, 0]).species, 91);
    }
    let f = field(
        field_config(),
        vec![DensityBrush {
            species: None,
            ..brush(1)
        }],
    );
    assert_eq!(f.sample([0; 3]).species, 91);
    let f = field(
        field_config(),
        vec![
            DensityBrush {
                species: Some(u32::MAX),
                ..brush(1)
            },
            DensityBrush {
                species: Some(0),
                density: DensityBrushOp::Add(0),
                ..brush(2)
            },
        ],
    );
    assert_eq!(f.sample([0; 3]).species, 0);
}

#[test]
fn replace_and_signed_add_round_differently_and_saturate_at_both_endpoints() {
    for (base_density, density, opacity, expected) in [
        (10_000, DensityBrushOp::Add(-1), 32_768, 10_000),
        (10_000, DensityBrushOp::Add(1), 32_768, 10_000),
        (10_000, DensityBrushOp::Add(-3), 32_768, 9_999),
        (10_000, DensityBrushOp::Add(3), 32_768, 10_001),
        (10_000, DensityBrushOp::Add(-65_535), 65_535, 0),
        (10_000, DensityBrushOp::Add(65_535), 65_535, 65_535),
        (0, DensityBrushOp::Replace(65_535), 32_768, 32_768),
        (65_535, DensityBrushOp::Replace(0), 32_768, 32_767),
        (123, DensityBrushOp::Replace(0), 65_535, 0),
        (123, DensityBrushOp::Replace(65_535), 65_535, 65_535),
        (123, DensityBrushOp::Replace(65_535), 0, 123),
    ] {
        let f = field(
            ScatterFieldConfig {
                base_density,
                ..field_config()
            },
            vec![DensityBrush {
                density,
                opacity,
                ..brush(1)
            }],
        );
        assert_eq!(
            f.sample([0; 3]).density,
            expected,
            "{density:?}, opacity {opacity}"
        );
    }
    // Saturation happens per stroke, not after summing all signed deltas.
    let f = field(
        ScatterFieldConfig {
            base_density: 1,
            ..field_config()
        },
        vec![
            DensityBrush {
                density: DensityBrushOp::Add(-65_535),
                ..brush(1)
            },
            DensityBrush {
                density: DensityBrushOp::Add(7),
                ..brush(2)
            },
        ],
    );
    assert_eq!(f.sample([0; 3]).density, 7);
}

#[test]
fn hard_brush_inner_equals_outer_never_divides_by_zero_and_excludes_outer_surface() {
    for radius in [1, 5, u32::MAX] {
        let stroke = DensityBrush {
            inner_radius_mm: radius,
            outer_radius_mm: radius,
            ..brush(1)
        };
        let f = field(
            ScatterFieldConfig {
                base_density: 0,
                ..field_config()
            },
            vec![stroke],
        );
        let radius = i64::from(radius);
        assert_eq!(f.sample([0; 3]).density, 65_535);
        assert_eq!(f.sample([radius - 1, 0, 0]).density, 65_535);
        assert_eq!(f.sample([radius, 0, 0]).density, 0);
        assert_eq!(f.sample([-radius, 0, 0]).density, 0);
    }
    let point_inner = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            inner_radius_mm: 0,
            outer_radius_mm: 1,
            ..brush(1)
        }],
    );
    assert_eq!(point_inner.sample([0; 3]).density, 65_535);
    assert_eq!(point_inner.sample([1, 0, 0]).density, 0);
}

#[test]
fn extreme_i64_brush_distances_do_not_overflow_square_sum_wrap_or_lose_one_mm() {
    for center_mm in [[i64::MIN; 3], [i64::MAX; 3]] {
        let stroke = DensityBrush {
            center_mm,
            inner_radius_mm: u32::MAX,
            outer_radius_mm: u32::MAX,
            ..brush(1)
        };
        let f = field(
            ScatterFieldConfig {
                base_density: 0,
                ..field_config()
            },
            vec![stroke],
        );
        assert_eq!(f.sample(center_mm).density, 65_535);
        let opposite = if center_mm[0] == i64::MIN {
            i64::MAX
        } else {
            i64::MIN
        };
        assert_eq!(f.sample([opposite; 3]).density, 0);
        let direction = if center_mm[0] == i64::MIN { 1 } else { -1 };
        let mut near = center_mm;
        near[0] += direction * (i64::from(u32::MAX) - 1);
        assert_eq!(f.sample(near).density, 65_535);
        near[0] += direction;
        assert_eq!(f.sample(near).density, 0);
        let diagonal = center_mm.map(|v| v + direction * 3_000_000_000);
        assert_eq!(f.sample(diagonal).density, 0);
    }
    let center_mm = [(1_i64 << 53) + 1, -(1_i64 << 53) - 3, 0];
    let f = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            center_mm,
            inner_radius_mm: 1,
            outer_radius_mm: 1,
            ..brush(1)
        }],
    );
    assert_eq!(f.sample(center_mm).density, 65_535);
    assert_eq!(f.sample([center_mm[0] + 1, center_mm[1], 0]).density, 0);
}

#[test]
fn unorm_smoothstep_and_lerp_match_widened_oracle_for_every_weight() {
    let mut previous = 0;
    for weight in 0..=u16::MAX {
        let actual = smoothstep(weight);
        assert_eq!(actual, scalar_smoothstep(weight));
        assert!(actual >= previous);
        previous = actual;
        for (a, b) in [
            (0, 65_535),
            (65_535, 0),
            (1, 2),
            (2, 1),
            (123, 60_001),
            (17, 17),
        ] {
            assert_eq!(lerp(a, b, weight), scalar_lerp(a, b, weight));
        }
    }
    assert_eq!(smoothstep(0), 0);
    assert_eq!(smoothstep(u16::MAX), u16::MAX);
    assert_eq!(smoothstep(32_767), 32_767);
    assert_eq!(smoothstep(32_768), 32_768);
}

#[test]
fn independent_pcg_and_trilinear_noise_are_pinned_to_fixed_golden_samples() {
    assert_eq!(
        SCATTER_FIELD_VERSION, 1,
        "arithmetic goldens require an explicit revision review"
    );
    for (input, expected) in [
        (0, 1_055_580_183),
        (1, 3_817_416_052),
        (field_config().seed, 593_296_677),
        (u64::MAX, 1_228_489_125),
    ] {
        assert_eq!(scalar_pcg(input), expected);
    }
    for (position, size, channel, expected) in [
        ([0, 0, 0], 17, 0, 36_820),
        ([1, 2, 3], 17, 0, 37_567),
        ([-1, -17, -18], 17, 0, 4_131),
        (
            [9_007_199_254_740_993, -9_007_199_254_740_995, 7],
            31,
            0,
            25_542,
        ),
        (
            [i128::from(i64::MAX), i128::from(i64::MIN), 0],
            1,
            0,
            49_330,
        ),
        ([-31, 17, 63], 31, 2, 44_401),
        ([5, 11, 29], u32::MAX, 3, 28_152),
    ] {
        assert_eq!(
            scalar_noise(position, size, field_config().seed, channel),
            expected
        );
        assert_eq!(
            value_noise(position, size, field_config().seed, channel),
            expected
        );
    }
}

#[test]
fn scalar_noise_oracle_covers_negative_lattices_large_integers_clamping_and_period() {
    let positions = [
        [0; 3],
        [1, 2, 3],
        [-1; 3],
        [-17, -18, 16],
        [16, 17, 18],
        [31, -31, 62],
        [(1_i64 << 53) + 1, -(1_i64 << 53) - 3, 7],
        [(1_i64 << 53) + 2, -(1_i64 << 53) - 4, 8],
        [i64::MIN; 3],
        [i64::MAX; 3],
        [i64::MIN, i64::MAX, -1],
    ];
    let mut saw_zero = false;
    let mut saw_full = false;
    for seed in [0, field_config().seed, u64::MAX] {
        for cell_size_mm in [1, 2, 17, 31, u32::MAX] {
            for amplitude in [0, 1, u16::MAX] {
                for base_density in [0, 32_768, u16::MAX] {
                    let config = ScatterFieldConfig {
                        seed,
                        base_density,
                        noise: Some(DensityNoise {
                            cell_size_mm,
                            amplitude,
                            // Even a configured extreme warp must be skipped
                            // when the density amplitude itself is zero.
                            warp_amplitude_mm: if amplitude == 0 { u32::MAX } else { 0 },
                            ..noise()
                        }),
                        ..field_config()
                    };
                    let f = field(config, vec![]);
                    for position in positions {
                        let expected = scalar_density(position, config);
                        let actual = f.sample(position);
                        assert_eq!(actual.density, expected, "{config:?}, {position:?}");
                        assert_eq!(actual.species, config.base_species);
                        saw_zero |= actual.density == 0;
                        saw_full |= actual.density == u16::MAX;
                    }
                }
            }
        }
    }
    assert!(saw_zero && saw_full);
    // Hash period is in lattice coordinates, not millimetres. Exercise virtual
    // neighbours beyond i64 rather than accidentally clamping them to the edge.
    for size in [1, 17, u32::MAX] {
        let origin = [-19_i128, 7, i128::from(i64::MAX)];
        for channel in 0..4 {
            let expected = scalar_noise(origin, size, field_config().seed, channel);
            for axis in 0..3 {
                for direction in [-1_i128, 1] {
                    let mut shifted = origin;
                    shifted[axis] += direction * (1_i128 << 64) * i128::from(size);
                    assert_eq!(
                        scalar_noise(shifted, size, field_config().seed, channel),
                        expected
                    );
                    assert_eq!(
                        value_noise(shifted, size, field_config().seed, channel),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn domain_warp_matches_independent_signed_oracle_and_widens_past_both_world_edges() {
    let golden_config = ScatterFieldConfig {
        noise: Some(DensityNoise {
            warp_amplitude_mm: u32::MAX,
            ..noise()
        }),
        ..field_config()
    };
    let golden = field(golden_config, vec![]);
    for (position, expected) in [
        ([0; 3], 44_231),
        ([1, 2, 3], 16_930),
        ([-1, -17, -18], 44_696),
        ([i64::MAX, i64::MIN, 0], 50_930),
    ] {
        assert_eq!(scalar_density(position, golden_config), expected);
        assert_eq!(golden.sample(position).density, expected);
    }
    let mut below_min = false;
    let mut above_max = false;
    let mut negative_delta = false;
    let mut positive_delta = false;
    let mut changed_by_warp = false;
    for seed in [0, 1, field_config().seed, u64::MAX] {
        for (cell_size_mm, warp_cell_size_mm) in [(1, 1), (17, 13), (u32::MAX, u32::MAX)] {
            for warp_amplitude_mm in [1, 23, u32::MAX] {
                let config = ScatterFieldConfig {
                    seed,
                    noise: Some(DensityNoise {
                        cell_size_mm,
                        warp_cell_size_mm,
                        warp_amplitude_mm,
                        ..noise()
                    }),
                    ..field_config()
                };
                let f = field(config, vec![]);
                let unwarped = ScatterFieldConfig {
                    noise: Some(DensityNoise {
                        warp_amplitude_mm: 0,
                        ..config.noise.unwrap()
                    }),
                    ..config
                };
                for position in [
                    [0; 3],
                    [-18, -1, 17],
                    [i64::MIN; 3],
                    [i64::MAX; 3],
                    [(1_i64 << 53) + 1, 3, -7],
                ] {
                    let warped = scalar_warp(position, config);
                    for axis in 0..3 {
                        below_min |= warped[axis] < i128::from(i64::MIN);
                        above_max |= warped[axis] > i128::from(i64::MAX);
                        negative_delta |= warped[axis] < i128::from(position[axis]);
                        positive_delta |= warped[axis] > i128::from(position[axis]);
                    }
                    let expected = scalar_density(position, config);
                    assert_eq!(
                        f.sample(position).density,
                        expected,
                        "{config:?}, {position:?}"
                    );
                    changed_by_warp |= expected != scalar_density(position, unwarped);
                }
            }
        }
    }
    assert!(below_min && above_max && negative_delta && positive_delta && changed_by_warp);
}

#[test]
fn rank_matches_independent_golden_and_uses_the_full_candidate_and_field_seed_namespace() {
    let f = field(field_config(), vec![]);
    let original = candidate_key();
    assert_eq!(scalar_rank(field_config().seed, original), 2_846_899_230);
    assert_eq!(f.candidate_rank(original), 2_846_899_230);
    // High bits and each signed coordinate matter, not just their low u32.
    let variants = [
        ScatterCandidateKey {
            kernel_version: original.kernel_version ^ 0x8000_0000,
            ..original
        },
        ScatterCandidateKey {
            seed: original.seed ^ (1 << 63),
            ..original
        },
        ScatterCandidateKey {
            version: original.version ^ 0x8000_0000,
            ..original
        },
        ScatterCandidateKey {
            salt: original.salt ^ (1 << 63),
            ..original
        },
        ScatterCandidateKey {
            grid_spacing_mm: original.grid_spacing_mm ^ 0x8000_0000,
            ..original
        },
        ScatterCandidateKey {
            grid: SectorCoord::new(-7 + (1_i64 << 32), 4, 13),
            ..original
        },
        ScatterCandidateKey {
            grid: SectorCoord::new(-7, 4 + (1_i64 << 32), 13),
            ..original
        },
        ScatterCandidateKey {
            grid: SectorCoord::new(-7, 4, 13 + (1_i64 << 32)),
            ..original
        },
        ScatterCandidateKey {
            grid: SectorCoord::new(i64::MIN, i64::MAX, -1),
            ..original
        },
    ];
    for key in variants {
        assert_eq!(f.candidate_rank(key), scalar_rank(field_config().seed, key));
        assert_ne!(
            f.candidate_rank(key),
            f.candidate_rank(original),
            "fixed namespace fixture {key:?}"
        );
    }
    let other_seed = field_config().seed ^ (1 << 63);
    let other = field(
        ScatterFieldConfig {
            seed: other_seed,
            ..field_config()
        },
        vec![],
    );
    assert_eq!(
        other.candidate_rank(original),
        scalar_rank(other_seed, original)
    );
    assert_ne!(other.candidate_rank(original), f.candidate_rank(original));
    // Recipe revision, brushes, species, density and noise invalidate the cache,
    // but do not perturb the independent selection ranks when seed is fixed.
    let edited = field(
        ScatterFieldConfig {
            version: u32::MAX,
            base_density: 0,
            base_species: u32::MAX,
            noise: Some(DensityNoise {
                warp_amplitude_mm: u32::MAX,
                ..noise()
            }),
            ..field_config()
        },
        vec![brush(u64::MAX)],
    );
    assert_ne!(f, edited);
    for key in variants.into_iter().chain(core::iter::once(original)) {
        assert_eq!(edited.candidate_rank(key), f.candidate_rank(key));
    }
}

#[test]
fn accepts_has_exact_endpoints_strict_threshold_and_exhaustive_density_monotonicity() {
    let f = field(field_config(), vec![]);
    for grid in [
        SectorCoord::new(-7, 4, 13),
        SectorCoord::new(0, 0, 0),
        SectorCoord::new(i64::MIN, i64::MAX, -1),
        SectorCoord::new(18_428, 4, 13),
    ] {
        let key = ScatterCandidateKey {
            grid,
            ..candidate_key()
        };
        let rank = u64::from(scalar_rank(field_config().seed, key));
        assert!(!f.accepts(key, 0));
        assert!(f.accepts(key, u16::MAX));
        let mut accepted_before = false;
        for density in 0..=u16::MAX {
            let expected =
                density == u16::MAX || (density != 0 && rank < u64::from(density) * 65_537);
            let actual = f.accepts(key, density);
            assert_eq!(actual, expected, "{key:?}, density {density}");
            assert!(!accepted_before || actual);
            accepted_before |= actual;
        }
        let first = (rank / 65_537 + 1).min(u64::from(u16::MAX)) as u16;
        assert!(f.accepts(key, first));
        assert!(!f.accepts(key, first - 1));
    }
    // Fixed rank is exactly density * 65537: this distinguishes < from <=,
    // unlike a probabilistic collection of arbitrary keys.
    let tie = ScatterCandidateKey {
        grid: SectorCoord::new(18_428, 4, 13),
        ..candidate_key()
    };
    assert_eq!(f.candidate_rank(tie), 2_411_630_526);
    assert_eq!(2_411_630_526_u64, 36_798 * 65_537);
    assert!(!f.accepts(tie, 36_798));
    assert!(f.accepts(tie, 36_799));
}

#[test]
fn field_scatter_matches_manual_cpu_scatter_sample_accepts_and_attribute_order() {
    let f = authored_field();
    let mut saw_accepted = false;
    let mut saw_rejected = false;
    let mut species = BTreeSet::new();
    for min_distance_mm in [0, 11, 29] {
        let config = ScatterConfig {
            min_distance_mm,
            ..scatter_config()
        };
        for bounds in [
            region([-51; 3], [51; 3]),
            region([-35, -23, -19], [40, 39, 31]),
            region([(1_i64 << 53) + 1, -17, -17], [(1_i64 << 53) + 70, 18, 18]),
            region([i64::MIN, -17, -17], [i64::MIN + 35, 18, 18]),
            region([i64::MAX - 35, -17, -17], [i64::MAX, 18, 18]),
        ] {
            let source = scatter(config, bounds, source_budget()).unwrap();
            let mut expected_points = Vec::new();
            let mut expected_attributes = Vec::new();
            for point in &source.points {
                let sample = f.sample(point.position_mm);
                if f.accepts(point.key, sample.density) {
                    expected_points.push(*point);
                    expected_attributes.push(sample);
                    species.insert(sample.species);
                    saw_accepted = true;
                } else {
                    saw_rejected = true;
                }
            }
            let actual =
                scatter_with_density_field(config, bounds, source_budget(), &f, execution_budget())
                    .unwrap();
            assert_eq!(actual.points, expected_points);
            assert_eq!(actual.attributes, expected_attributes);
            assert_eq!(actual.candidate_evaluations, source.candidate_evaluations);
            assert_eq!(
                actual.field_work,
                source.points.len() as u64 * f.work_per_sample()
            );
            assert_eq!(actual.cache_key.scatter, source.cache_key);
            assert_eq!(actual.cache_key.field, f);
            assert_eq!(actual.cache_key.field_version, SCATTER_FIELD_VERSION);
            assert!(actual
                .points
                .windows(2)
                .all(|pair| pair[0].key < pair[1].key));
            assert!(actual
                .points
                .iter()
                .all(|point| bounds.contains_mm(point.position_mm)));
        }
    }
    assert!(saw_accepted && saw_rejected);
    assert!(species.contains(&0) && species.contains(&u32::MAX));
}

#[test]
fn zero_and_full_density_filter_none_or_all_without_changing_source_work_or_positions() {
    for min_distance_mm in [0, 23] {
        let config = ScatterConfig {
            min_distance_mm,
            ..scatter_config()
        };
        let bounds = region([-51; 3], [51; 3]);
        let source = scatter(config, bounds, source_budget()).unwrap();
        assert!(!source.points.is_empty());
        for density in [0, u16::MAX] {
            let f = field(
                ScatterFieldConfig {
                    base_density: density,
                    base_species: u32::MAX,
                    ..field_config()
                },
                vec![],
            );
            let out =
                scatter_with_density_field(config, bounds, source_budget(), &f, execution_budget())
                    .unwrap();
            assert_eq!(out.candidate_evaluations, source.candidate_evaluations);
            assert_eq!(out.field_work, source.points.len() as u64);
            assert_eq!(out.points.capacity(), source.points.capacity());
            assert!(out.attributes.capacity() >= source.points.len());
            if density == 0 {
                assert!(out.points.is_empty() && out.attributes.is_empty());
            } else {
                assert_eq!(out.points, source.points);
                assert_eq!(
                    out.attributes,
                    vec![
                        ScatterFieldSample {
                            density,
                            species: u32::MAX
                        };
                        source.points.len()
                    ]
                );
            }
        }
    }
}

#[test]
fn arbitrary_three_axis_region_splits_and_reverse_order_preserve_points_and_attributes() {
    let f = authored_field();
    for min_distance_mm in [0, 11, 29] {
        let config = ScatterConfig {
            min_distance_mm,
            ..scatter_config()
        };
        // Cover many exclusion volumes even at the largest thinning radius:
        // the smaller manual-scatter fixture can legitimately filter to zero.
        let bounds = region([-103, -89, -79], [112, 107, 101]);
        let whole =
            scatter_with_density_field(config, bounds, source_budget(), &f, execution_budget())
                .unwrap();
        assert!(
            whole.points.len() > 1,
            "fixture must retain multiple points at radius {min_distance_mm}"
        );
        for axis in 0..3 {
            assert!(
                whole.points.iter().any(|point| point.position_mm[axis] < 0)
                    && whole
                        .points
                        .iter()
                        .any(|point| point.position_mm[axis] >= 0),
                "fixture must populate both sides of axis {axis} at radius {min_distance_mm}"
            );
        }
        let mut requests = Vec::new();
        // Deliberately not aligned to the 17-mm generation lattice, and with
        // a one-millimetre slice straddling the origin on each axis.
        for xs in [-103, -3, 0, 1, 12, 112].windows(2) {
            for ys in [-89, 0, 1, 107].windows(2) {
                for zs in [-79, -1, 0, 4, 101].windows(2) {
                    requests.push(region([xs[0], ys[0], zs[0]], [xs[1], ys[1], zs[1]]));
                }
            }
        }
        let expected = sorted_pairs(&whole);
        for reverse in [false, true] {
            let mut combined = Vec::new();
            for offset in 0..requests.len() {
                let index = if reverse {
                    requests.len() - 1 - offset
                } else {
                    offset
                };
                let request = requests[index];
                let out = scatter_with_density_field(
                    config,
                    request,
                    source_budget(),
                    &f,
                    execution_budget(),
                )
                .unwrap();
                assert!(out
                    .points
                    .iter()
                    .all(|point| request.contains_mm(point.position_mm)));
                combined.extend(sorted_pairs(&out));
            }
            let unique: BTreeSet<_> = combined.iter().map(|(point, _)| point.key).collect();
            assert_eq!(
                unique.len(),
                combined.len(),
                "duplicate ownership at a split"
            );
            combined.sort_unstable_by_key(|(point, _)| point.key);
            assert_eq!(combined, expected);
        }
    }
    // Also split at single-mm boundaries beyond exact f64 integer precision
    // and at both representable world edges.
    for start in [
        i64::MIN,
        -(1_i64 << 53) - 3,
        (1_i64 << 53) + 1,
        i64::MAX - 4,
    ] {
        let config = ScatterConfig {
            grid_spacing_mm: 1,
            ..scatter_config()
        };
        let whole = scatter_with_density_field(
            config,
            region([start, 0, 0], [start + 4, 1, 1]),
            source_budget(),
            &f,
            execution_budget(),
        )
        .unwrap();
        let mut split = Vec::new();
        for offset in (0..4).rev() {
            let x = start + offset;
            let out = scatter_with_density_field(
                config,
                region([x, 0, 0], [x + 1, 1, 1]),
                source_budget(),
                &f,
                execution_budget(),
            )
            .unwrap();
            split.extend(sorted_pairs(&out));
        }
        split.sort_unstable_by_key(|(point, _)| point.key);
        assert_eq!(split, sorted_pairs(&whole));
    }
}

#[test]
fn field_filter_preserves_minimum_distance_across_seams_and_does_not_repack_holes() {
    let config = ScatterConfig {
        min_distance_mm: 23,
        ..scatter_config()
    };
    let f = authored_field();
    let mut pairs = Vec::new();
    for x in -2..=1 {
        let bounds = ScatterRegion::from_cell(SectorCoord::new(x, 0, 0), 35).unwrap();
        let out =
            scatter_with_density_field(config, bounds, source_budget(), &f, execution_budget())
                .unwrap();
        pairs.extend(sorted_pairs(&out));
    }
    assert!(pairs.len() > 1);
    for (index, (a, _)) in pairs.iter().enumerate() {
        for (b, _) in &pairs[index + 1..] {
            let squared: u128 = (0..3)
                .map(|axis| {
                    let delta = (i128::from(a.position_mm[axis]) - i128::from(b.position_mm[axis]))
                        .unsigned_abs();
                    delta * delta
                })
                .sum();
            assert!(squared >= 23_u128 * 23);
            assert_ne!(a.key, b.key);
        }
    }
    let bounds = region([-51; 3], [51; 3]);
    let raw = scatter(
        ScatterConfig {
            min_distance_mm: 0,
            ..config
        },
        bounds,
        source_budget(),
    )
    .unwrap();
    let thinned = scatter(config, bounds, source_budget()).unwrap();
    let survivor_keys: BTreeSet<_> = thinned.points.iter().map(|point| point.key).collect();
    let blocked = raw
        .points
        .iter()
        .find(|point| !survivor_keys.contains(&point.key))
        .expect("fixture must contain a source-blocked candidate");
    // Paint only the blocked candidate's exact integer position to full
    // density. Removing every blocker by the field must not resurrect it.
    let isolated = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            center_mm: blocked.position_mm,
            inner_radius_mm: 1,
            outer_radius_mm: 1,
            ..brush(1)
        }],
    );
    assert_eq!(isolated.sample(blocked.position_mm).density, u16::MAX);
    let out = scatter_with_density_field(
        config,
        bounds,
        source_budget(),
        &isolated,
        execution_budget(),
    )
    .unwrap();
    assert!(out.points.is_empty() && out.attributes.is_empty());
    assert_eq!(
        out.field_work,
        thinned.points.len() as u64 * isolated.work_per_sample()
    );
}

#[test]
fn cache_describes_all_source_inputs_and_rebuilds_independently_of_successful_budgets() {
    let f = authored_field();
    let config = scatter_config();
    let bounds = region([-51; 3], [51; 3]);
    let generous =
        scatter_with_density_field(config, bounds, source_budget(), &f, execution_budget())
            .unwrap();
    let estimate = estimate_scatter_work(config, bounds).unwrap();
    let source = scatter(config, bounds, source_budget()).unwrap();
    let exact_source = ScatterBudget {
        max_candidate_evaluations: estimate.max_candidate_evaluations,
        max_output_points: source.points.len(),
        max_output_bytes: source.points.capacity() * size_of::<ScatterPoint>(),
    };
    let exact_field = FieldScatterBudget {
        max_field_work: estimate.primary_candidates * f.work_per_sample(),
        max_output_bytes: generous.points.capacity() * size_of::<ScatterPoint>()
            + generous.attributes.capacity() * size_of::<ScatterFieldSample>(),
    };
    let key = &generous.cache_key;
    let rebuilt = scatter_with_density_field(
        key.scatter.config,
        key.scatter.region,
        exact_source,
        &key.field,
        exact_field,
    )
    .unwrap();
    assert_eq!(rebuilt, generous);
    for changed in [
        ScatterConfig {
            seed: config.seed ^ (1 << 63),
            ..config
        },
        ScatterConfig {
            version: config.version + 1,
            ..config
        },
        ScatterConfig {
            salt: config.salt ^ (1 << 63),
            ..config
        },
        ScatterConfig {
            grid_spacing_mm: 18,
            ..config
        },
        ScatterConfig {
            min_distance_mm: 1,
            ..config
        },
    ] {
        let out =
            scatter_with_density_field(changed, bounds, source_budget(), &f, execution_budget())
                .unwrap();
        assert_ne!(generous.cache_key, out.cache_key);
    }
    for axis in 0..3 {
        for change_min in [false, true] {
            let mut changed = bounds;
            if change_min {
                changed.min_mm[axis] += 1;
            } else {
                changed.max_mm[axis] -= 1;
            }
            let out = scatter_with_density_field(
                config,
                changed,
                source_budget(),
                &f,
                execution_budget(),
            )
            .unwrap();
            assert_ne!(generous.cache_key, out.cache_key);
        }
    }
    let mut changed = generous.cache_key.clone();
    changed.field_version += 1;
    assert_ne!(generous.cache_key, changed);
    let mut changed = generous.cache_key.clone();
    changed.scatter.kernel_version += 1;
    assert_ne!(generous.cache_key, changed);
}

#[test]
fn field_work_is_preflighted_against_all_primary_candidates_not_survivors_or_density() {
    let config = ScatterConfig {
        min_distance_mm: 29,
        ..scatter_config()
    };
    let bounds = region([-51; 3], [51; 3]);
    let f = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![DensityBrush {
            opacity: 0,
            ..brush(1)
        }],
    );
    let estimate = estimate_scatter_work(config, bounds).unwrap();
    let required = estimate.primary_candidates * f.work_per_sample();
    let source = scatter(config, bounds, source_budget()).unwrap();
    let actual_work = source.points.len() as u64 * f.work_per_sample();
    assert!(actual_work < required);
    assert_eq!(
        scatter_with_density_field(
            config,
            bounds,
            source_budget(),
            &f,
            FieldScatterBudget {
                max_field_work: required - 1,
                ..execution_budget()
            }
        ),
        Err(ScatterFieldError::WorkBudgetExceeded {
            required,
            limit: required - 1
        })
    );
    let out = scatter_with_density_field(
        config,
        bounds,
        source_budget(),
        &f,
        FieldScatterBudget {
            max_field_work: required,
            ..execution_budget()
        },
    )
    .unwrap();
    assert_eq!(out.field_work, actual_work);
    assert!(out.points.is_empty());
}

#[test]
fn combined_output_budget_counts_source_spare_capacity_and_all_pre_filter_attributes() {
    let config = ScatterConfig {
        min_distance_mm: 29,
        ..scatter_config()
    };
    let bounds = region([-51; 3], [51; 3]);
    let f = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![],
    );
    let source = scatter(config, bounds, source_budget()).unwrap();
    assert!(!source.points.is_empty());
    assert!(source.points.capacity() > source.points.len());
    let point_bytes = source.points.capacity() * size_of::<ScatterPoint>();
    let required = point_bytes + source.points.len() * size_of::<ScatterFieldSample>();
    assert_eq!(
        scatter_with_density_field(
            config,
            bounds,
            source_budget(),
            &f,
            FieldScatterBudget {
                max_output_bytes: required - 1,
                ..execution_budget()
            }
        ),
        Err(ScatterFieldError::OutputBytesBudgetExceeded {
            required,
            limit: required - 1
        })
    );
    let roomy = scatter_with_density_field(config, bounds, source_budget(), &f, execution_budget())
        .unwrap();
    let actual_bytes = roomy.points.capacity() * size_of::<ScatterPoint>()
        + roomy.attributes.capacity() * size_of::<ScatterFieldSample>();
    let exact = scatter_with_density_field(
        config,
        bounds,
        source_budget(),
        &f,
        FieldScatterBudget {
            max_output_bytes: actual_bytes,
            ..execution_budget()
        },
    )
    .unwrap();
    assert_eq!(roomy, exact);
    assert!(exact.points.is_empty() && exact.attributes.is_empty());
    assert_eq!(exact.points.capacity(), source.points.capacity());
    assert!(exact.attributes.capacity() >= source.points.len());
    assert!(actual_bytes >= required);
    // Even zero survivors can leave primary-candidate spare point capacity.
    let point = candidate_key().rebuild().unwrap();
    let mut unowned = point.position_mm;
    let voxel_start = point.key.grid.x * i64::from(point.key.grid_spacing_mm);
    unowned[0] = voxel_start + i64::from(point.position_mm[0] == voxel_start);
    let tiny = region(unowned, unowned.map(|v| v + 1));
    let sparse = scatter(scatter_config(), tiny, source_budget()).unwrap();
    assert!(sparse.points.is_empty());
    assert!(sparse.points.capacity() > 0);
    let required = sparse.points.capacity() * size_of::<ScatterPoint>();
    assert_eq!(
        scatter_with_density_field(
            scatter_config(),
            tiny,
            source_budget(),
            &f,
            FieldScatterBudget {
                max_output_bytes: 0,
                ..execution_budget()
            }
        ),
        Err(ScatterFieldError::OutputBytesBudgetExceeded { required, limit: 0 })
    );
    let out = scatter_with_density_field(
        scatter_config(),
        tiny,
        source_budget(),
        &f,
        FieldScatterBudget {
            max_output_bytes: required,
            ..execution_budget()
        },
    )
    .unwrap();
    assert!(out.points.is_empty());
    assert_eq!(out.points.capacity(), sparse.points.capacity());
    assert_eq!(out.attributes.capacity(), 0);
    assert_eq!(out.field_work, 0);
}

#[test]
fn output_byte_precheck_handles_exact_zero_limits_and_both_overflow_paths() {
    let attribute_size = size_of::<ScatterFieldSample>();
    assert_eq!(check_output_bytes(0, 0, 0), Ok(()));
    assert_eq!(check_output_bytes(7, 3, 7 + 3 * attribute_size), Ok(()));
    assert_eq!(
        check_output_bytes(7, 3, 7 + 3 * attribute_size - 1),
        Err(ScatterFieldError::OutputBytesBudgetExceeded {
            required: 7 + 3 * attribute_size,
            limit: 7 + 3 * attribute_size - 1
        })
    );
    assert!(attribute_size > 1);
    assert_eq!(
        check_output_bytes(0, usize::MAX / attribute_size + 1, usize::MAX),
        Err(ScatterFieldError::ArithmeticOverflow)
    );
    assert_eq!(
        check_output_bytes(usize::MAX, 1, usize::MAX),
        Err(ScatterFieldError::ArithmeticOverflow)
    );
    assert_eq!(check_output_bytes(usize::MAX, 0, usize::MAX), Ok(()));
    assert_eq!(
        check_output_bytes(usize::MAX, 1, 0),
        Err(ScatterFieldError::ArithmeticOverflow)
    );
}

#[test]
fn work_multiplication_and_source_estimate_overflows_fail_before_allocation_or_enumeration() {
    let config = ScatterConfig {
        grid_spacing_mm: 1,
        ..scatter_config()
    };
    let f = field(
        field_config(),
        vec![DensityBrush {
            opacity: 0,
            ..brush(1)
        }],
    );
    let line = region([i64::MIN, 0, 0], [i64::MAX, 1, 1]);
    assert_eq!(
        estimate_scatter_work(config, line)
            .unwrap()
            .primary_candidates,
        u64::MAX
    );
    let no_source = ScatterBudget {
        max_candidate_evaluations: 0,
        max_output_points: 0,
        max_output_bytes: 0,
    };
    assert_eq!(
        scatter_with_density_field(config, line, no_source, &f, execution_budget()),
        Err(ScatterFieldError::ArithmeticOverflow)
    );
    let huge = region([i64::MIN; 3], [i64::MAX; 3]);
    assert_eq!(
        scatter_with_density_field(config, huge, no_source, &f, execution_budget()),
        Err(ScatterFieldError::Scatter(ScatterError::ArithmeticOverflow))
    );
    let extreme_halo = ScatterConfig {
        min_distance_mm: u32::MAX,
        ..config
    };
    assert_eq!(
        scatter_with_density_field(
            extreme_halo,
            region([0; 3], [1; 3]),
            no_source,
            &f,
            execution_budget()
        ),
        Err(ScatterFieldError::Scatter(ScatterError::ArithmeticOverflow))
    );
}

#[test]
fn empty_requests_need_no_source_or_field_budget_but_still_validate_source_inputs() {
    let config = ScatterConfig {
        grid_spacing_mm: 1,
        min_distance_mm: u32::MAX,
        ..scatter_config()
    };
    let f = authored_field();
    let source_zero = ScatterBudget {
        max_candidate_evaluations: 0,
        max_output_points: 0,
        max_output_bytes: 0,
    };
    let field_zero = FieldScatterBudget {
        max_field_work: 0,
        max_output_bytes: 0,
    };
    for axis in 0..3 {
        let mut max_mm = [i64::MAX; 3];
        max_mm[axis] = i64::MIN;
        let bounds = region([i64::MIN; 3], max_mm);
        let out = scatter_with_density_field(config, bounds, source_zero, &f, field_zero).unwrap();
        assert_eq!(out.points.capacity(), 0);
        assert_eq!(out.attributes.capacity(), 0);
        assert_eq!(out.candidate_evaluations, 0);
        assert_eq!(out.field_work, 0);
        assert_eq!(out.cache_key.scatter.region, bounds);
        assert_eq!(out.cache_key.field, f);
        assert_eq!(
            scatter_with_density_field(
                ScatterConfig {
                    grid_spacing_mm: 0,
                    ..config
                },
                bounds,
                source_zero,
                &f,
                field_zero
            ),
            Err(ScatterFieldError::Scatter(ScatterError::InvalidGridSpacing))
        );
    }
    // An empty X axis must not hide an inverted Y axis.
    let invalid = ScatterRegion {
        min_mm: [0, 1, 0],
        max_mm: [0, 0, 1],
    };
    assert_eq!(
        scatter_with_density_field(config, invalid, source_zero, &f, field_zero),
        Err(ScatterFieldError::Scatter(ScatterError::InvalidRegion))
    );
}

#[test]
fn error_priority_is_validation_then_field_work_then_source_budgets_then_combined_bytes() {
    let config = ScatterConfig {
        grid_spacing_mm: 1,
        ..scatter_config()
    };
    let f = field(
        ScatterFieldConfig {
            base_density: 0,
            ..field_config()
        },
        vec![],
    );
    let bounds = region([0; 3], [2, 1, 1]);
    let zero_source = ScatterBudget {
        max_candidate_evaluations: 0,
        max_output_points: 0,
        max_output_bytes: 0,
    };
    let zero_field = FieldScatterBudget {
        max_field_work: 0,
        max_output_bytes: 0,
    };
    let invalid = ScatterRegion {
        min_mm: [1; 3],
        max_mm: [0; 3],
    };
    assert_eq!(
        scatter_with_density_field(
            ScatterConfig {
                grid_spacing_mm: 0,
                ..config
            },
            invalid,
            zero_source,
            &f,
            zero_field
        ),
        Err(ScatterFieldError::Scatter(ScatterError::InvalidGridSpacing))
    );
    assert_eq!(
        scatter_with_density_field(config, invalid, zero_source, &f, zero_field),
        Err(ScatterFieldError::Scatter(ScatterError::InvalidRegion))
    );
    assert_eq!(
        scatter_with_density_field(config, bounds, zero_source, &f, zero_field),
        Err(ScatterFieldError::WorkBudgetExceeded {
            required: 2,
            limit: 0
        })
    );
    let enough_work = FieldScatterBudget {
        max_field_work: 2,
        max_output_bytes: 0,
    };
    assert_eq!(
        scatter_with_density_field(config, bounds, zero_source, &f, enough_work),
        Err(ScatterFieldError::Scatter(
            ScatterError::CandidateBudgetExceeded {
                required: 2,
                limit: 0
            }
        ))
    );
    let no_count = ScatterBudget {
        max_output_points: 0,
        ..source_budget()
    };
    assert_eq!(
        scatter_with_density_field(config, bounds, no_count, &f, enough_work),
        Err(ScatterFieldError::Scatter(
            ScatterError::OutputCountBudgetExceeded {
                required: 1,
                limit: 0
            }
        ))
    );
    let size = size_of::<ScatterPoint>();
    let no_bytes = ScatterBudget {
        max_output_bytes: 2 * size - 1,
        ..source_budget()
    };
    assert_eq!(
        scatter_with_density_field(config, bounds, no_bytes, &f, enough_work),
        Err(ScatterFieldError::Scatter(
            ScatterError::OutputBytesBudgetExceeded {
                required: 2 * size,
                limit: 2 * size - 1
            }
        ))
    );
    let source = scatter(config, bounds, source_budget()).unwrap();
    let required =
        source.points.capacity() * size + source.points.len() * size_of::<ScatterFieldSample>();
    assert_eq!(
        scatter_with_density_field(config, bounds, source_budget(), &f, enough_work),
        Err(ScatterFieldError::OutputBytesBudgetExceeded { required, limit: 0 })
    );
    // Constructor precedence is similarly deliberate: invalid noise, count,
    // capacity, then brush validation, and only then duplicate detection.
    let invalid_brush = DensityBrush {
        outer_radius_mm: 0,
        ..brush(1)
    };
    let no_brushes = ScatterFieldBuildBudget {
        max_brushes: 0,
        max_brush_bytes: 0,
    };
    assert_eq!(
        ScatterDensityField::new(
            ScatterFieldConfig {
                noise: Some(DensityNoise {
                    cell_size_mm: 0,
                    ..noise()
                }),
                ..field_config()
            },
            vec![invalid_brush],
            no_brushes
        ),
        Err(ScatterFieldError::InvalidNoiseCellSize)
    );
    assert_eq!(
        ScatterDensityField::new(field_config(), vec![invalid_brush], no_brushes),
        Err(ScatterFieldError::BrushBudgetExceeded {
            required: 1,
            limit: 0
        })
    );
    let strokes = vec![invalid_brush];
    let required = strokes.capacity() * size_of::<DensityBrush>();
    assert_eq!(
        ScatterDensityField::new(
            field_config(),
            strokes,
            ScatterFieldBuildBudget {
                max_brushes: 1,
                max_brush_bytes: 0
            }
        ),
        Err(ScatterFieldError::BrushBytesBudgetExceeded { required, limit: 0 })
    );
    assert_eq!(
        ScatterDensityField::new(
            field_config(),
            vec![invalid_brush, brush(1)],
            build_budget()
        ),
        Err(ScatterFieldError::InvalidBrushRadius)
    );
}
