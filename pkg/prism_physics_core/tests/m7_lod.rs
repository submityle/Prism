//! Integration tests for the M7 adaptive level-of-detail controllers.

use prism_physics_core::{LodConfig, SpatialController, TemporalController};

#[test]
fn temporal_controller_scales_with_speed() {
    let temporal = TemporalController::new(0.5, 0.1, 1, 32);
    let dt = 1.0 / 60.0;
    let slow = temporal.substeps(dt, 0.5);
    let fast = temporal.substeps(dt, 50.0);
    assert!(
        slow <= fast,
        "faster motion should need at least as many substeps"
    );
    assert!(temporal.substep_dt(dt, 50.0) <= temporal.substep_dt(dt, 0.5));
}

#[test]
fn spatial_controller_reduces_distant_budget() {
    let spatial = SpatialController::new(vec![5.0, 20.0], vec![1.0, 0.5, 0.25], 2);
    let near = spatial.scaled_iterations(16, 1.0);
    let mid = spatial.scaled_iterations(16, 10.0);
    let far = spatial.scaled_iterations(16, 100.0);
    assert!(near >= mid && mid >= far);
    assert_eq!(near, 16);
    assert_eq!(far, 4);
}

#[test]
fn config_builds_both_controllers() {
    let config = LodConfig::default();
    let temporal = config.temporal();
    let spatial = config.spatial();
    assert_eq!(temporal.bounds(), (1, 16));
    assert_eq!(spatial.tier_count(), 3);
}
