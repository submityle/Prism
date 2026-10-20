//! Analytic fields verify interpolation, derivatives, addressing and missing data.

use prism_world_physics_bridge::{
    SdfBridgeError, SdfDistanceGradient, SdfPhysicsBridgeBudget, SdfPhysicsSnapshot,
    SdfQueryResult, WorldSdfPhysicsBridge, WorldSdfSnapshot, WorldSdfTransform,
};
use prism_world_residency::{BrickCoord, SdfError, SparseSdfBrickGrid};

fn snapshot(grid: SparseSdfBrickGrid, transform: WorldSdfTransform) -> SdfPhysicsSnapshot {
    WorldSdfPhysicsBridge::new(SdfPhysicsBridgeBudget::new(64, 4096, 16384).expect("test budget"))
        .stage(WorldSdfSnapshot::new(transform, 7, grid))
        .expect("test snapshot fits")
}

fn add_brick(
    grid: &mut SparseSdfBrickGrid,
    coord: BrickCoord,
    field: impl Fn(i64, i64, i64) -> i16,
) {
    let edge = i64::from(grid.brick_size());
    let mut samples = Vec::new();
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                samples.push(field(
                    i64::from(coord.x) * edge + x,
                    i64::from(coord.y) * edge + y,
                    i64::from(coord.z) * edge + z,
                ));
            }
        }
    }
    grid.insert_brick(coord, samples).expect("valid brick");
}

fn value(snapshot: &SdfPhysicsSnapshot, position: [i64; 3], mip: u8) -> SdfDistanceGradient {
    match snapshot
        .query_world_distance_gradient_mm(position, mip)
        .expect("query")
    {
        SdfQueryResult::Value(sample) => sample,
        other => panic!("fully resident query returned {other:?}"),
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-10,
        "{actual} != {expected}"
    );
}

/// Integer subtraction preserves subvoxel weights near both i64 endpoints.
#[test]
fn affine_field_preserves_millimetres_at_large_world_origins() {
    let origin = [i64::MAX - 1000, i64::MIN + 1000, 9_000_000_000_000_000_000];
    let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    add_brick(&mut grid, BrickCoord::new(0, 0, 0, 0), |x, y, z| {
        i16::try_from(2 * x - 3 * y + 5 * z - 4).expect("field")
    });
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new(origin, 100, 10).expect("transform"),
    );
    let sample = value(
        &snapshot,
        [origin[0] + 25, origin[1] + 50, origin[2] + 75],
        0,
    );
    close(sample.distance_mm, -12.5);
    for (actual, expected) in sample.gradient.into_iter().zip([0.2, -0.3, 0.5]) {
        close(actual, expected);
    }
    let normal = sample.unit_normal().expect("nonzero gradient");
    close(normal.iter().map(|v| v * v).sum::<f64>(), 1.0);
    assert!(normal[0] > 0.0 && normal[1] < 0.0 && normal[2] > 0.0);
}

/// Eight distinct bricks meet around this negatively addressed lattice cell.
#[test]
fn interpolation_crosses_negative_brick_boundaries() {
    let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    for z in -1..=0 {
        for y in 0..=1 {
            for x in -1..=0 {
                add_brick(&mut grid, BrickCoord::new(x, y, z, 0), |x, y, z| {
                    i16::try_from(2 * x + 3 * y - 4 * z + 1).expect("field")
                });
            }
        }
    }
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new([0; 3], 100, 2).expect("transform"),
    );
    let sample = value(&snapshot, [-25, 175, -75], 0);
    close(sample.distance_mm, 17.5);
    for (actual, expected) in sample.gradient.into_iter().zip([0.04, 0.06, -0.08]) {
        close(actual, expected);
    }
}

/// Mixed xyz terms expose wrong corner ordering and incorrect derivative axes.
#[test]
fn mixed_field_matches_analytic_and_finite_difference_gradients() {
    let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    add_brick(&mut grid, BrickCoord::new(0, 0, 0, 0), |x, y, z| {
        i16::try_from(7 + 2 * x - 3 * y + 5 * z + 4 * x * y - 2 * x * z + 3 * y * z + 6 * x * y * z)
            .expect("field")
    });
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new([0; 3], 100, 3).expect("transform"),
    );
    let before = snapshot.clone();
    for px in [25, 50, 75] {
        for py in [25, 50, 75] {
            for pz in [25, 50, 75] {
                let position = [px, py, pz];
                let [x, y, z] = position.map(|v| v as f64 / 100.0);
                let sample = value(&snapshot, position, 0);
                close(
                    sample.distance_mm,
                    3.0 * (7.0 + 2.0 * x - 3.0 * y + 5.0 * z + 4.0 * x * y - 2.0 * x * z
                        + 3.0 * y * z
                        + 6.0 * x * y * z),
                );
                let expected = [
                    2.0 + 4.0 * y - 2.0 * z + 6.0 * y * z,
                    -3.0 + 4.0 * x + 3.0 * z + 6.0 * x * z,
                    5.0 - 2.0 * x + 3.0 * y + 6.0 * x * y,
                ];
                for axis in 0..3 {
                    close(sample.gradient[axis], expected[axis] * 0.03);
                    let mut lower = position;
                    let mut upper = position;
                    lower[axis] -= 1;
                    upper[axis] += 1;
                    let difference = (value(&snapshot, upper, 0).distance_mm
                        - value(&snapshot, lower, 0).distance_mm)
                        / 2.0;
                    close(sample.gradient[axis], difference);
                }
            }
        }
    }
    assert_eq!(snapshot, before, "queries do not alter published data");
}

/// Missing any corner invalidates the whole sample, even at a lattice knot.
#[test]
fn every_missing_corner_is_unknown_without_a_fabricated_normal() {
    for missing in 0..8 {
        let mut grid = SparseSdfBrickGrid::new(1, 1).expect("grid");
        for index in 0..8 {
            if index != missing {
                add_brick(
                    &mut grid,
                    BrickCoord::new(index & 1, (index >> 1) & 1, index >> 2, 0),
                    |_, _, _| 5,
                );
            }
        }
        let snapshot = snapshot(
            grid,
            WorldSdfTransform::new([0; 3], 100, 1).expect("transform"),
        );
        for position in [[25, 50, 75], [0; 3]] {
            assert_eq!(
                snapshot.query_world_distance_gradient_mm(position, 0),
                Ok(SdfQueryResult::Unknown)
            );
        }
    }
}

/// Pairwise differences keep constant fields exactly flat despite fractional weights.
#[test]
fn saturated_constant_fields_have_no_normal() {
    for constant in [i16::MIN, i16::MAX] {
        let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
        add_brick(&mut grid, BrickCoord::new(0, 0, 0, 0), |_, _, _| constant);
        let snapshot = snapshot(
            grid,
            WorldSdfTransform::new([0; 3], 100, u32::MAX).expect("transform"),
        );
        let sample = value(&snapshot, [17, 31, 83], 0);
        assert_eq!(
            sample.distance_mm,
            f64::from(constant) * f64::from(u32::MAX)
        );
        assert_eq!(sample.gradient, [0.0; 3]);
        assert_eq!(sample.unit_normal(), None);
    }
}

/// Physical derivatives divide by mip spacing, not just mip-zero spacing.
#[test]
fn mip_and_distance_scales_are_independent() {
    let mut grid = SparseSdfBrickGrid::new(2, 3).expect("grid");
    add_brick(&mut grid, BrickCoord::new(0, 0, 0, 2), |x, y, z| {
        i16::try_from(x + y + z).expect("field")
    });
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new([0; 3], 25, 5).expect("transform"),
    );
    let sample = value(&snapshot, [25, 50, 75], 2);
    assert_eq!(sample.distance_mm, 7.5);
    assert_eq!(sample.gradient, [0.05; 3]);
    assert_eq!(
        snapshot.query_world_distance_gradient_mm([25, 50, 75], 0),
        Ok(SdfQueryResult::Unknown)
    );
    assert_eq!(
        snapshot.query_world_distance_gradient_mm([0; 3], 3),
        Err(SdfBridgeError::GridQuery(SdfError::InvalidMip))
    );
}

/// Upper corners and extreme world deltas fail before any overflowing address math.
#[test]
fn coordinate_and_mip_overflows_are_errors() {
    let grid = SparseSdfBrickGrid::new(2, 255).expect("grid");
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new([0; 3], u32::MAX, 1).expect("transform"),
    );
    assert_eq!(
        snapshot.query_world_distance_gradient_mm([0; 3], 127),
        Err(SdfBridgeError::MipScaleOverflow)
    );
    assert_eq!(
        snapshot.query_world_distance_gradient_mm([0; 3], 96),
        Err(SdfBridgeError::MipScaleOverflow)
    );
    let grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    let snapshot = self::snapshot(
        grid,
        WorldSdfTransform::new([0; 3], 1, 1).expect("transform"),
    );
    for x in [
        i64::from(i32::MAX),
        i64::from(i32::MIN) - 1,
        i64::MAX,
        i64::MIN,
    ] {
        assert_eq!(
            snapshot.query_world_distance_gradient_mm([x, 0, 0], 0),
            Err(SdfBridgeError::VoxelCoordinateOutOfRange)
        );
    }
}

/// Euclidean local addressing works even when a brick's origin lies below i32::MIN.
#[test]
fn non_power_of_two_bricks_support_the_lowest_voxel() {
    let mut grid = SparseSdfBrickGrid::new(3, 1).expect("grid");
    add_brick(
        &mut grid,
        BrickCoord::new(i32::MIN.div_euclid(3), 0, 0, 0),
        |x, y, z| i16::try_from(x - i64::from(i32::MIN) + y + z).expect("field"),
    );
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new([0; 3], 1, 2).expect("transform"),
    );
    let sample = value(&snapshot, [i64::from(i32::MIN), 0, 0], 0);
    assert_eq!(sample.distance_mm, 0.0);
    assert_eq!(sample.gradient, [2.0; 3]);
}

/// Normalization rejects invalid external values and handles finite extremes.
#[test]
fn unit_normal_handles_external_floating_point_extremes() {
    for magnitude in [f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX] {
        let normal = SdfDistanceGradient {
            distance_mm: 0.0,
            gradient: [magnitude; 3],
        }
        .unit_normal()
        .expect("finite normal");
        for component in normal {
            close(component, 1.0 / 3.0_f64.sqrt());
        }
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            SdfDistanceGradient {
                distance_mm: 0.0,
                gradient: [invalid, 1.0, 0.0]
            }
            .unit_normal(),
            None
        );
    }
}

/// Values remain continuous at lattice knots while the documented derivative
/// uses the positive-side cell rather than averaging away a sharp change.
#[test]
fn lattice_knot_uses_the_positive_side_derivative() {
    let mut grid = SparseSdfBrickGrid::new(3, 1).expect("grid");
    add_brick(&mut grid, BrickCoord::new(0, 0, 0, 0), |x, _, _| {
        i16::try_from(x * x).expect("field")
    });
    let snapshot = snapshot(
        grid,
        WorldSdfTransform::new([0; 3], 100, 1).expect("transform"),
    );
    let before = value(&snapshot, [99, 0, 0], 0);
    let at = value(&snapshot, [100, 0, 0], 0);
    close(before.distance_mm, 0.99);
    assert_eq!(before.gradient, [0.01, 0.0, 0.0]);
    assert_eq!(at.distance_mm, 1.0);
    assert_eq!(at.gradient, [0.03, 0.0, 0.0]);
}
