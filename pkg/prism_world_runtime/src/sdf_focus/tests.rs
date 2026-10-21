use super::*;
use crate::{WorldSdfClipmapConfig, WorldSdfClipmapScheduler};
use prism_world_residency::SdfClipmapConfig;

fn config(margins: [u64; 3]) -> WorldSdfFocusConfig {
    WorldSdfFocusConfig::new([0; 3], 25, 4, margins).unwrap()
}

fn apply(driver: &mut WorldSdfFocusDriver, position: [i64; 3]) -> WorldSdfFocusUpdate<()> {
    driver.update_with(position, |_| Ok::<_, ()>(())).unwrap()
}

#[test]
fn validates_geometry_literals_and_each_margin() {
    for (voxel, brick) in [(0, 4), (1, 0), (1, 33), (1, u8::MAX)] {
        let invalid = WorldSdfFocusConfig {
            mip0_voxel_size_mm: voxel,
            brick_size: brick,
            ..config([0; 3])
        };
        assert!(matches!(
            WorldSdfFocusDriver::new(invalid),
            Err(WorldSdfFocusError::InvalidGeometry)
        ));
    }
    for (i, axis) in AXES.into_iter().enumerate() {
        for margin in [100, u64::MAX] {
            let mut invalid = config([0; 3]);
            invalid.hysteresis_mm[i] = margin;
            assert!(
                matches!(WorldSdfFocusDriver::new(invalid), Err(WorldSdfFocusError::InvalidHysteresis { axis: a }) if a == axis)
            );
        }
    }
    assert!(WorldSdfFocusDriver::new(config([99; 3])).is_ok());
}

#[test]
fn single_slice_coverage_rejects_y_lag_and_zero_radius_horizontal_lag() {
    assert_eq!(
        config([1, 0, 0]).validate_single_slice_coverage(0),
        Err(WorldSdfFocusError::InvalidCoverage {
            axis: ClipmapAxis::X
        })
    );
    assert_eq!(
        config([0, 0, 1]).validate_single_slice_coverage(0),
        Err(WorldSdfFocusError::InvalidCoverage {
            axis: ClipmapAxis::Z
        })
    );
    assert_eq!(
        config([0, 1, 0]).validate_single_slice_coverage(u32::MAX),
        Err(WorldSdfFocusError::InvalidCoverage {
            axis: ClipmapAxis::Y
        })
    );
    assert!(config([99, 0, 99])
        .validate_single_slice_coverage(1)
        .is_ok());
    assert!(config([0; 3]).validate_single_slice_coverage(0).is_ok());
}

#[test]
fn admitted_deadbands_keep_raw_focus_resident_at_every_mip() {
    use prism_world_residency::SdfToroidalClipmap;

    for (margins, radius) in [([0; 3], 0), ([99, 0, 99], 1), ([3, 0, 71], 1)] {
        let cfg = WorldSdfFocusConfig {
            origin_mm: [-9001, 7003, -3307],
            ..config(margins)
        };
        cfg.validate_single_slice_coverage(radius).unwrap();
        let mut driver = WorldSdfFocusDriver::new(cfg).unwrap();
        let mut ring =
            SdfToroidalClipmap::new(SdfClipmapConfig::new(8, radius, 64 * 1024).unwrap()).unwrap();
        for n in -700_i64..=700 {
            // Monotone and returning axes cross negative/positive boundaries;
            // alternating Z teleports ensure retained focus is not assumed.
            let relative = [n, -n * 3, if n % 2 == 0 { n * 11 } else { -n * 7 }];
            let position = core::array::from_fn(|i| cfg.origin_mm[i] + relative[i]);
            driver
                .update_with(position, |center| ring.update_center(center))
                .unwrap();
            for mip in 0..8 {
                let extent = 100_i64 << mip;
                let raw = relative.map(|v| v.div_euclid(extent) as i32);
                assert!(
                    ring.slot_index(BrickCoord::new(raw[0], raw[1], raw[2], mip))
                        .is_some(),
                    "focus left ring: margins={margins:?} position={relative:?} mip={mip}"
                );
            }
        }
    }
}

#[test]
fn offset_positive_negative_and_exact_boundaries() {
    let cfg = WorldSdfFocusConfig {
        origin_mm: [1000, -1000, 33],
        ..config([0; 3])
    };
    let mut driver = WorldSdfFocusDriver::new(cfg).unwrap();
    for (offset, expected) in [
        (-201, -3),
        (-200, -2),
        (-101, -2),
        (-100, -1),
        (-1, -1),
        (0, 0),
        (99, 0),
        (100, 1),
    ] {
        assert_eq!(
            apply(&mut driver, cfg.origin_mm.map(|v| v + offset)).center,
            BrickCoord::new(expected, expected, expected, 0)
        );
    }
    assert_eq!(driver.config(), cfg);
}

#[test]
fn floor_mapping_matches_independent_integer_oracle() {
    let mut driver = WorldSdfFocusDriver::new(config([0; 3])).unwrap();
    for n in -5001_i64..=5001 {
        let position = [n, -n * 7, n * 13];
        let expected = position.map(|v| {
            let quotient = v / 100;
            (quotient - i64::from(v < 0 && v % 100 != 0)) as i32
        });
        assert_eq!(
            apply(&mut driver, position).center,
            BrickCoord::new(expected[0], expected[1], expected[2], 0)
        );
    }
}

#[test]
fn extreme_origin_subtraction_and_extent_are_widened() {
    let cfg = WorldSdfFocusConfig::new([i64::MIN; 3], u32::MAX, 32, [0; 3]).unwrap();
    assert_eq!(cfg.brick_extent_mm(), u64::from(u32::MAX) * 32);
    let mut driver = WorldSdfFocusDriver::new(cfg).unwrap();
    let expected =
        ((i128::from(i64::MAX) - i128::from(i64::MIN)) / i128::from(cfg.brick_extent_mm())) as i32;
    assert_eq!(
        apply(&mut driver, [i64::MAX; 3]).center,
        BrickCoord::new(expected, expected, expected, 0)
    );
    let cfg = WorldSdfFocusConfig {
        origin_mm: [i64::MAX; 3],
        ..cfg
    };
    let mut driver = WorldSdfFocusDriver::new(cfg).unwrap();
    assert_eq!(
        apply(&mut driver, [i64::MIN; 3]).center,
        BrickCoord::new(-expected - 1, -expected - 1, -expected - 1, 0)
    );
}

#[test]
fn raw_overflow_on_every_axis_rejects_before_callback_and_keeps_center() {
    for (i, axis) in AXES.into_iter().enumerate() {
        for value in [
            i64::MIN,
            i64::MAX,
            (i64::from(i32::MAX) + 1) * 100,
            i64::from(i32::MIN) * 100 - 1,
        ] {
            let mut driver = WorldSdfFocusDriver::new(config([99; 3])).unwrap();
            apply(&mut driver, [0; 3]);
            let mut position = [0; 3];
            position[i] = value;
            let result = driver.update_with(position, |_| -> Result<(), ()> {
                panic!("overflow invoked downstream")
            });
            assert_eq!(
                result,
                Err(WorldSdfFocusUpdateError::Focus(
                    WorldSdfFocusError::CoordinateOverflow { axis }
                ))
            );
            assert_eq!(driver.center(), Some(BrickCoord::new(0, 0, 0, 0)));
        }
    }
}

#[test]
fn deadband_does_not_hide_raw_i32_boundary_overflow() {
    let mut driver = WorldSdfFocusDriver::new(config([99; 3])).unwrap();
    let max = i64::from(i32::MAX) * 100;
    assert_eq!(apply(&mut driver, [max; 3]).center.x, i32::MAX);
    assert!(matches!(
        driver.update_with([max + 100; 3], |_| Ok::<_, ()>(())),
        Err(WorldSdfFocusUpdateError::Focus(
            WorldSdfFocusError::CoordinateOverflow {
                axis: ClipmapAxis::X
            }
        ))
    ));
    assert_eq!(driver.center().unwrap().x, i32::MAX);
    let min = i64::from(i32::MIN) * 100;
    assert_eq!(apply(&mut driver, [min; 3]).center.x, i32::MIN);
    assert!(driver
        .update_with([min - 1; 3], |_| Ok::<_, ()>(()))
        .is_err());
    assert_eq!(driver.center().unwrap().x, i32::MIN);
}

#[test]
fn independent_xyz_lower_inclusive_upper_exclusive_and_negative_return() {
    let mut driver = WorldSdfFocusDriver::new(config([10, 0, 25])).unwrap();
    apply(&mut driver, [-100; 3]);
    assert_eq!(apply(&mut driver, [-110, -100, -125]).applied, None);
    assert_eq!(apply(&mut driver, [9, -1, 24]).applied, None);
    assert_eq!(
        apply(&mut driver, [10, -1, 24]).center,
        BrickCoord::new(0, -1, -1, 0)
    );
    assert_eq!(
        apply(&mut driver, [10, 0, 24]).center,
        BrickCoord::new(0, 0, -1, 0)
    );
    assert_eq!(
        apply(&mut driver, [10, 0, 25]).center,
        BrickCoord::new(0, 0, 0, 0)
    );
    assert_eq!(apply(&mut driver, [-10, 0, -25]).applied, None);
    assert_eq!(
        apply(&mut driver, [-11, 0, -26]).center,
        BrickCoord::new(-1, 0, -1, 0)
    );
    assert_eq!(
        apply(&mut driver, [-111, -1, -126]).center,
        BrickCoord::new(-2, -1, -2, 0)
    );
}

#[test]
fn retained_jitter_skips_downstream_and_teleport_snaps_directly() {
    let mut driver = WorldSdfFocusDriver::new(config([99; 3])).unwrap();
    apply(&mut driver, [0; 3]);
    for value in -99..199 {
        let result = driver
            .update_with([value; 3], |_| -> Result<(), ()> {
                panic!("retained focus callback")
            })
            .unwrap();
        assert_eq!(result.applied, None);
    }
    assert_eq!(
        apply(&mut driver, [100_000_000, -100_000_001, 200_000_000]).center,
        BrickCoord::new(1_000_000, -1_000_001, 2_000_000, 0)
    );
}

#[test]
fn failed_first_apply_does_not_arm_hysteresis_and_retry_is_exact() {
    let mut driver = WorldSdfFocusDriver::new(config([99; 3])).unwrap();
    assert_eq!(
        driver.update_with([100; 3], |_| Err::<(), _>("budget")),
        Err(WorldSdfFocusUpdateError::Apply("budget"))
    );
    assert_eq!(driver.center(), None);
    assert_eq!(
        apply(&mut driver, [99; 3]).center,
        BrickCoord::new(0, 0, 0, 0)
    );
    assert_eq!(
        driver.update_with([1000; 3], |_| Err::<(), _>("budget")),
        Err(WorldSdfFocusUpdateError::Apply("budget"))
    );
    assert_eq!(driver.center(), Some(BrickCoord::new(0, 0, 0, 0)));
    assert_eq!(
        apply(&mut driver, [1000; 3]).center,
        BrickCoord::new(10, 10, 10, 0)
    );
}

#[test]
fn real_scheduler_retained_focus_keeps_dirty_revision_and_failed_radius_roll_is_atomic() {
    let cfg =
        WorldSdfClipmapConfig::new(SdfClipmapConfig::new(2, 1, 16384).unwrap(), 18, 4096).unwrap();
    let mut scheduler = WorldSdfClipmapScheduler::new(cfg).unwrap();
    let mut driver = WorldSdfFocusDriver::new(config([10, 0, 25])).unwrap();
    driver
        .update_with([0; 3], |center| scheduler.update_center(center))
        .unwrap();
    let revision = scheduler.revision();
    let dirty = scheduler.dirty_bricks().to_vec();
    let result = driver
        .update_with([-10, 0, 124], |center| scheduler.update_center(center))
        .unwrap();
    assert!(result.applied.is_none());
    assert_eq!(scheduler.revision(), revision);
    assert_eq!(scheduler.dirty_bricks(), dirty);
    // Mapping is legal but center+radius is not: both owners retain old state.
    assert!(driver
        .update_with([i64::from(i32::MAX) * 100, 0, 0], |center| scheduler
            .update_center(center))
        .is_err());
    assert_eq!(scheduler.revision(), revision);
    assert_eq!(scheduler.dirty_bricks(), dirty);
    assert_eq!(driver.center(), scheduler.center());
    driver
        .update_with([110, 0, 0], |center| scheduler.update_center(center))
        .unwrap();
    assert_eq!(driver.center(), scheduler.center());
    assert!(scheduler.revision() > revision);
}

#[test]
fn real_scheduler_initial_budget_failure_does_not_publish_either_owner() {
    let cfg =
        WorldSdfClipmapConfig::new(SdfClipmapConfig::new(1, 1, 4096).unwrap(), 8, 4096).unwrap();
    let mut scheduler = WorldSdfClipmapScheduler::new(cfg).unwrap();
    let mut driver = WorldSdfFocusDriver::new(config([0; 3])).unwrap();
    assert!(matches!(
        driver.update_with([0; 3], |center| scheduler.update_center(center)),
        Err(WorldSdfFocusUpdateError::Apply(_))
    ));
    assert_eq!(driver.center(), None);
    assert_eq!(scheduler.center(), None);
    assert_eq!(scheduler.revision(), 0);
    assert!(scheduler.dirty_bricks().is_empty());
}

#[cfg(feature = "std")]
#[test]
fn downstream_panic_does_not_commit_focus() {
    let mut driver = WorldSdfFocusDriver::new(config([0; 3])).unwrap();
    apply(&mut driver, [0; 3]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        driver.update_with([100; 3], |_| -> Result<(), ()> {
            panic!("downstream failed")
        })
    }));
    assert!(result.is_err());
    assert_eq!(driver.center(), Some(BrickCoord::new(0, 0, 0, 0)));
    apply(&mut driver, [100; 3]);
    assert_eq!(driver.center(), Some(BrickCoord::new(1, 1, 1, 0)));
}
