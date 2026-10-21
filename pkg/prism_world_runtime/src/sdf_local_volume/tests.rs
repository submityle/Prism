use super::{WorldSdfLocalVolume, WorldSdfLocalVolumeError};
use alloc::{format, vec, vec::Vec};

fn count(dimensions: [u32; 3]) -> usize {
    dimensions.into_iter().map(|n| n as usize).product()
}

fn index(dimensions: [u32; 3], [x, y, z]: [u32; 3]) -> usize {
    (u64::from(x)
        + u64::from(dimensions[0]) * (u64::from(y) + u64::from(dimensions[1]) * u64::from(z)))
        as usize
}

fn is_boundary(dimensions: [u32; 3], coordinate: [u32; 3]) -> bool {
    (0..3).any(|axis| coordinate[axis] == 0 || coordinate[axis] == dimensions[axis] - 1)
}

fn grid(dimensions: [u32; 3], mut interior: impl FnMut([u32; 3]) -> i16) -> Vec<i16> {
    let mut samples = vec![i16::MAX; count(dimensions)];
    for z in 1..dimensions[2] - 1 {
        for y in 1..dimensions[1] - 1 {
            for x in 1..dimensions[0] - 1 {
                let coordinate = [x, y, z];
                samples[index(dimensions, coordinate)] = interior(coordinate);
            }
        }
    }
    samples
}

fn volume(dimensions: [u32; 3], spacing: u32, samples: Vec<i16>) -> WorldSdfLocalVolume {
    let budget = samples.len();
    WorldSdfLocalVolume::new(dimensions, spacing, samples, budget).unwrap()
}

fn cube(values: [i16; 8], spacing: u32) -> WorldSdfLocalVolume {
    volume(
        [4; 3],
        spacing,
        grid([4; 3], |[x, y, z]| {
            values[((x - 1) + 2 * (y - 1) + 4 * (z - 1)) as usize]
        }),
    )
}

// Independent polynomial oracle rather than the production corner-weight sum.
// Unlike the implementation it evaluates boundary cells too, including their
// upper endpoints, and rounds the absolute rational via a doubled numerator.
fn oracle(dimensions: [u32; 3], spacing: u32, samples: &[i16], position: [i128; 3]) -> i16 {
    let s = i128::from(spacing);
    let mut cell = [0_u32; 3];
    let mut t = [0_i128; 3];
    for axis in 0..3 {
        if position[axis] < 0 || position[axis] > i128::from(dimensions[axis] - 1) * s {
            return i16::MAX;
        }
        cell[axis] = (position[axis] / s).min(i128::from(dimensions[axis] - 2)) as u32;
        t[axis] = position[axis] - i128::from(cell[axis]) * s;
    }
    let value = |dx, dy, dz| {
        i128::from(samples[index(dimensions, [cell[0] + dx, cell[1] + dy, cell[2] + dz])])
    };
    let a = value(0, 0, 0);
    let b = value(1, 0, 0);
    let c = value(0, 1, 0);
    let d = value(1, 1, 0);
    let e = value(0, 0, 1);
    let f = value(1, 0, 1);
    let g = value(0, 1, 1);
    let h = value(1, 1, 1);
    let [x, y, z] = t;
    let denominator = s * s * s;
    let numerator = a * denominator
        + (b - a) * x * s * s
        + (c - a) * y * s * s
        + (e - a) * z * s * s
        + (d - b - c + a) * x * y * s
        + (f - b - e + a) * x * z * s
        + (g - c - e + a) * y * z * s
        + (h - d - f - g + b + c + e - a) * x * y * z;
    let rounded_magnitude = (numerator.abs() * 2 + denominator) / (denominator * 2);
    if numerator < 0 {
        (-rounded_magnitude) as i16
    } else {
        rounded_magnitude as i16
    }
}

#[test]
fn reports_geometry_and_preserves_x_fastest_lattice_samples() {
    let dimensions = [5, 4, 6];
    let spacing = 7;
    let samples = grid(dimensions, |[x, y, z]| {
        (1000 * x as i32 - 300 * y as i32 + 17 * z as i32) as i16
    });
    let field = volume(dimensions, spacing, samples.clone());
    assert_eq!(field.dimensions(), dimensions);
    assert_eq!(field.voxel_size_mm(), spacing);
    assert_eq!(field.max_local_mm(), [28, 21, 35]);
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let coordinate = [x, y, z];
                let position = coordinate.map(|n| i128::from(n) * i128::from(spacing));
                assert_eq!(
                    field.sample(position),
                    samples[index(dimensions, coordinate)],
                    "grid vertex {coordinate:?}"
                );
            }
        }
    }
}

#[test]
fn exhaustive_non_cubic_grid_matches_independent_integer_oracle() {
    let dimensions = [4, 5, 6];
    let samples = grid(dimensions, |[x, y, z]| {
        ((x * 21_413 + y * 7439 + z * 10_007) % 65_536) as i32 as i16
    });
    for spacing in [1, 2, 3, 4, 5] {
        let field = volume(dimensions, spacing, samples.clone());
        let max = field.max_local_mm();
        for z in -1..=max[2] + 1 {
            for y in -1..=max[1] + 1 {
                for x in -1..=max[0] + 1 {
                    let position = [i128::from(x), i128::from(y), i128::from(z)];
                    assert_eq!(
                        field.sample(position),
                        oracle(dimensions, spacing, &samples, position),
                        "spacing={spacing} position={position:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn halves_round_away_from_zero_on_each_axis() {
    for axis in 0..3 {
        for (low, expected) in [(-2, -2), (-1, -1), (0, 1), (1, 2)] {
            let samples = grid([4; 3], |coordinate| low + (coordinate[axis] - 1) as i16);
            let field = volume([4; 3], 2, samples);
            let mut position = [2; 3];
            position[axis] = 3;
            assert_eq!(field.sample(position), expected, "axis={axis} low={low}");
        }
    }
}

#[test]
fn rounds_only_after_all_three_axes_not_after_individual_lerps() {
    for sign in [-1, 1] {
        let mut values = [0; 8];
        values[7] = sign;
        let field = cube(values, 2);
        // +/-1/8 rounds to zero; individually rounded lerps would produce +/-1.
        assert_eq!(field.sample([3; 3]), 0);
    }
    let field = cube([-1, 0, -1, 0, -1, 0, -1, 0], 2);
    assert_eq!(field.sample([3; 3]), -1);
}

#[test]
fn signed_rounding_covers_both_sides_of_a_half() {
    for sign in [-1_i16, 1] {
        let field = cube([0, sign, 0, sign, 0, sign, 0, sign], 10);
        for (x, expected_magnitude) in [(11, 0), (14, 0), (15, 1), (16, 1), (19, 1)] {
            assert_eq!(field.sample([x, 10, 10]), sign * expected_magnitude);
        }
    }
}

#[test]
fn extreme_quantized_distances_remain_exact_and_do_not_saturate() {
    for value in [i16::MIN, i16::MAX, -123, 0, 456] {
        let field = cube([value; 8], 13);
        for z in 13..=26 {
            for y in 13..=26 {
                for x in 13..=26 {
                    assert_eq!(field.sample([x, y, z]), value);
                }
            }
        }
    }
    let field = cube(
        [
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX,
            i16::MIN,
            i16::MAX,
        ],
        2,
    );
    // The exact midpoint is -0.5, not zero or the positive far sentinel.
    assert_eq!(field.sample([3; 3]), -1);
    assert_eq!(field.sample([2; 3]), i16::MIN);
    assert_eq!(field.sample([4, 2, 2]), i16::MAX);
}

#[test]
fn maximum_u32_spacing_matches_oracle_without_i128_overflow() {
    let spacing = u32::MAX;
    let s = i128::from(spacing);
    let dimensions = [4; 3];
    let values = [i16::MIN, i16::MAX, -17, 3, i16::MAX, i16::MIN, 0, -1];
    let samples = grid(dimensions, |[x, y, z]| {
        values[((x - 1) + 2 * (y - 1) + 4 * (z - 1)) as usize]
    });
    let field = volume(dimensions, spacing, samples.clone());
    assert_eq!(field.max_local_mm(), [3 * i64::from(spacing); 3]);
    let positions = [
        0,
        1,
        s - 1,
        s,
        s + 1,
        s + s / 2,
        2 * s - 1,
        2 * s,
        3 * s - 1,
        3 * s,
    ];
    for x in positions {
        for y in positions {
            for z in positions {
                let position = [x, y, z];
                assert_eq!(
                    field.sample(position),
                    oracle(dimensions, spacing, &samples, position),
                    "position={position:?}"
                );
            }
        }
    }
    // Exercise the largest negative numerator and maximum positive numerator,
    // not just mixed corner values with cancellation.
    for value in [i16::MIN, i16::MAX] {
        assert_eq!(cube([value; 8], spacing).sample([s + s / 2; 3]), value);
    }
}

#[test]
fn spacing_never_implicitly_rescales_quantized_distance_units() {
    for spacing in [1, 2, 17, 1000, u32::MAX] {
        let field = cube([-193; 8], spacing);
        assert_eq!(field.sample([i128::from(spacing); 3]), -193);
    }
}

#[test]
fn far_corners_interpolate_numerically_inside_boundary_cells() {
    let field = cube([-1; 8], 2);
    assert_eq!(field.sample([0, 2, 2]), i16::MAX);
    assert_eq!(field.sample([1, 2, 2]), 16_383);
    assert_eq!(field.sample([2, 2, 2]), -1);
    assert_eq!(field.sample([5, 2, 2]), 16_383);
    assert_eq!(field.sample([6, 2, 2]), i16::MAX);
}

#[test]
fn every_face_edge_corner_and_arbitrary_outside_position_is_far() {
    let field = volume([4, 5, 6], 3, grid([4, 5, 6], |_| i16::MIN));
    let max = field.max_local_mm().map(i128::from);
    for z in 0..=max[2] {
        for y in 0..=max[1] {
            for x in 0..=max[0] {
                let position = [x, y, z];
                if (0..3).any(|axis| position[axis] == 0 || position[axis] == max[axis]) {
                    assert_eq!(field.sample(position), i16::MAX, "{position:?}");
                }
            }
        }
    }
    for axis in 0..3 {
        for outside in [i128::MIN, -1, max[axis] + 1, i128::MAX] {
            let mut position = [3; 3];
            position[axis] = outside;
            assert_eq!(field.sample(position), i16::MAX, "{position:?}");
        }
    }
    assert_eq!(field.sample([i128::MIN; 3]), i16::MAX);
    assert_eq!(field.sample([i128::MAX; 3]), i16::MAX);
}

#[test]
fn two_sample_axes_have_no_non_far_interior_vertices() {
    for dimensions in [[2, 4, 5], [4, 2, 5], [4, 5, 2], [2; 3]] {
        let field = volume(dimensions, 3, vec![i16::MAX; count(dimensions)]);
        let max = field.max_local_mm();
        for z in 0..=max[2] {
            for y in 0..=max[1] {
                for x in 0..=max[0] {
                    assert_eq!(field.sample([x.into(), y.into(), z.into()]), i16::MAX);
                }
            }
        }
    }
}

#[test]
fn rejects_each_invalid_dimension_before_other_checks() {
    for axis in 0..3 {
        for dimension in [0, 1] {
            let mut dimensions = [2; 3];
            dimensions[axis] = dimension;
            assert_eq!(
                WorldSdfLocalVolume::new(dimensions, 0, Vec::new(), 0).unwrap_err(),
                WorldSdfLocalVolumeError::InvalidDimension { axis, dimension }
            );
        }
    }
}

#[test]
fn rejects_zero_spacing_before_storage_or_budget_validation() {
    assert_eq!(
        WorldSdfLocalVolume::new([2; 3], 0, Vec::new(), 0).unwrap_err(),
        WorldSdfLocalVolumeError::InvalidVoxelSize
    );
}

#[test]
fn detects_sample_product_overflow_without_allocating_sample_storage() {
    assert_eq!(
        WorldSdfLocalVolume::new([u32::MAX; 3], 1, Vec::new(), usize::MAX).unwrap_err(),
        WorldSdfLocalVolumeError::SampleCountOverflow
    );
}

#[test]
fn checks_each_extent_without_narrowing_or_allocating_giant_storage() {
    for axis in 0..3 {
        let mut dimensions = [2; 3];
        dimensions[axis] = u32::MAX;
        let expected = if usize::BITS >= 64 {
            WorldSdfLocalVolumeError::ExtentOverflow { axis }
        } else {
            WorldSdfLocalVolumeError::SampleCountOverflow
        };
        assert_eq!(
            WorldSdfLocalVolume::new(dimensions, u32::MAX, Vec::new(), usize::MAX).unwrap_err(),
            expected
        );
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn extent_limit_accepts_largest_valid_dimension_and_rejects_next() {
    // With spacing u32::MAX, 2^31 intervals fit i64, but 2^31 + 1 do not.
    for axis in 0..3 {
        let mut dimensions = [2; 3];
        dimensions[axis] = (1_u32 << 31) + 1;
        let required = dimensions[axis] as usize * 4;
        assert_eq!(
            WorldSdfLocalVolume::new(dimensions, u32::MAX, Vec::new(), 0).unwrap_err(),
            WorldSdfLocalVolumeError::SampleBudgetExceeded {
                required,
                max_samples: 0
            }
        );
        dimensions[axis] += 1;
        assert_eq!(
            WorldSdfLocalVolume::new(dimensions, u32::MAX, Vec::new(), usize::MAX).unwrap_err(),
            WorldSdfLocalVolumeError::ExtentOverflow { axis }
        );
    }
}

#[test]
fn rejects_budget_before_storage_length_or_boundary_scan() {
    for samples in [Vec::new(), vec![0; 63], vec![0; 64], vec![0; 65]] {
        for max_samples in [0, 63] {
            assert_eq!(
                WorldSdfLocalVolume::new([4; 3], 1, samples.clone(), max_samples).unwrap_err(),
                WorldSdfLocalVolumeError::SampleBudgetExceeded {
                    required: 64,
                    max_samples
                }
            );
        }
    }
    assert!(WorldSdfLocalVolume::new([4; 3], 1, vec![i16::MAX; 64], 64).is_ok());
}

#[test]
fn rejects_both_incomplete_and_excess_storage_before_boundary_scan() {
    for actual in [0, 1, 63, 65, 128] {
        assert_eq!(
            WorldSdfLocalVolume::new([4; 3], 1, vec![0; actual], 64).unwrap_err(),
            WorldSdfLocalVolumeError::SampleCountMismatch {
                expected: 64,
                actual
            }
        );
    }
}

#[test]
fn rejects_every_non_far_boundary_vertex_including_edges_and_corners() {
    let dimensions = [4, 5, 6];
    let valid = grid(dimensions, |_| i16::MIN);
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let coordinate = [x, y, z];
                if !is_boundary(dimensions, coordinate) {
                    continue;
                }
                for value in [i16::MIN, 0, i16::MAX - 1] {
                    let mut samples = valid.clone();
                    samples[index(dimensions, coordinate)] = value;
                    assert_eq!(
                        WorldSdfLocalVolume::new(dimensions, 1, samples, count(dimensions))
                            .unwrap_err(),
                        WorldSdfLocalVolumeError::InvalidBoundarySample { coordinate, value }
                    );
                }
            }
        }
    }
    assert!(WorldSdfLocalVolume::new(dimensions, 1, valid, count(dimensions)).is_ok());
}

#[test]
fn owns_original_allocation_without_copying_or_reallocating() {
    let mut samples = Vec::with_capacity(128);
    samples.resize(64, i16::MAX);
    let pointer = samples.as_ptr();
    let capacity = samples.capacity();
    let field = WorldSdfLocalVolume::new([4; 3], 2, samples, 64).unwrap();
    assert_eq!(field.samples.as_ptr(), pointer);
    assert_eq!(field.samples.capacity(), capacity);
    assert_eq!(field.samples.len(), 64);
    for x in -1..=7 {
        assert_eq!(field.sample([x, 3, 3]), i16::MAX);
    }
    assert_eq!(field.samples.as_ptr(), pointer);
    assert_eq!(field.samples.capacity(), capacity);
}

#[test]
fn every_error_has_core_error_and_actionable_display() {
    let cases = [
        (
            WorldSdfLocalVolumeError::InvalidDimension {
                axis: 1,
                dimension: 0,
            },
            "local SDF dimension on axis 1 must be at least 2, got 0",
        ),
        (
            WorldSdfLocalVolumeError::InvalidVoxelSize,
            "local SDF voxel spacing must be positive",
        ),
        (
            WorldSdfLocalVolumeError::SampleCountOverflow,
            "local SDF sample count exceeds the platform usize range",
        ),
        (
            WorldSdfLocalVolumeError::ExtentOverflow { axis: 2 },
            "local SDF extent on axis 2 exceeds i64::MAX",
        ),
        (
            WorldSdfLocalVolumeError::SampleBudgetExceeded {
                required: 64,
                max_samples: 63,
            },
            "local SDF requires 64 samples, exceeding budget 63",
        ),
        (
            WorldSdfLocalVolumeError::SampleCountMismatch {
                expected: 64,
                actual: 65,
            },
            "local SDF requires exactly 64 samples, got 65",
        ),
        (
            WorldSdfLocalVolumeError::InvalidBoundarySample {
                coordinate: [1, 0, 2],
                value: -1,
            },
            "local SDF boundary sample at [1, 0, 2] must be i16::MAX, got -1",
        ),
    ];
    for (error, expected) in cases {
        let erased: &dyn core::error::Error = &error;
        assert!(erased.source().is_none());
        assert_eq!(format!("{erased}"), expected);
    }
}
