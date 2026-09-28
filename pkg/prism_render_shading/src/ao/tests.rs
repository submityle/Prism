//! Golden tests for the GTAO reference: analytic edge cases on the closed-form
//! integral plus behavioural integration tests on synthetic depth buffers.

use super::integral::{combine_horizon, distance_weight, slice_visibility, HALF_PI};
use super::{compute_gtao, gtao_pixel, GtaoBuffers, GtaoCamera, GtaoConfig};

const CAMERA: GtaoCamera = GtaoCamera {
    tan_half_fov_x: 0.5,
    tan_half_fov_y: 0.5,
};

/// Flat fronto-parallel plane at constant depth facing the camera.
fn flat_plane(width: usize, height: usize, depth: f32) -> (Vec<f32>, Vec<[f32; 3]>) {
    (vec![depth; width * height], vec![[0.0, 0.0, 1.0]; width * height])
}

/// Left half is the floor at depth 10, right half a step raised 0.8 closer.
fn stepped_scene(width: usize, height: usize) -> (Vec<f32>, Vec<[f32; 3]>) {
    let mut depth = vec![10.0_f32; width * height];
    for y in 0..height {
        for x in (width / 2)..width {
            depth[y * width + x] = 9.2;
        }
    }
    (depth, vec![[0.0, 0.0, 1.0]; width * height])
}

#[test]
fn open_plane_is_fully_visible() {
    let (width, height) = (32, 32);
    let (depth, normals) = flat_plane(width, height, 10.0);
    let buffers = GtaoBuffers { width, height, linear_depth: &depth, view_normals: &normals };
    let center = gtao_pixel(buffers, CAMERA, GtaoConfig::default(), 16, 16);
    assert!(center > 0.98, "open plane center should be unoccluded, got {center}");
}

#[test]
fn background_pixels_are_unoccluded() {
    let (width, height) = (8, 8);
    let mut depth = vec![10.0_f32; width * height];
    depth[3 * width + 3] = 0.0; // sky / no geometry
    let normals = vec![[0.0, 0.0, 1.0]; width * height];
    let buffers = GtaoBuffers { width, height, linear_depth: &depth, view_normals: &normals };
    assert_eq!(gtao_pixel(buffers, CAMERA, GtaoConfig::default(), 3, 3), 1.0);
}

#[test]
fn nearer_step_darkens_adjacent_floor() {
    let (width, height) = (64, 64);
    let (depth, normals) = stepped_scene(width, height);
    let buffers = GtaoBuffers { width, height, linear_depth: &depth, view_normals: &normals };
    let config = GtaoConfig { world_radius: 1.5, ..GtaoConfig::default() };
    let near_seam = gtao_pixel(buffers, CAMERA, config, 30, 32);
    let far_floor = gtao_pixel(buffers, CAMERA, config, 2, 32);
    assert!(far_floor > 0.9, "far floor should stay open, got {far_floor}");
    assert!(
        near_seam < far_floor - 0.05,
        "seam pixel should be more occluded: near={near_seam} far={far_floor}"
    );
}

#[test]
fn larger_radius_reaches_more_occluders() {
    let (width, height) = (64, 64);
    let (depth, normals) = stepped_scene(width, height);
    let buffers = GtaoBuffers { width, height, linear_depth: &depth, view_normals: &normals };
    // A pixel a few texels from the seam is only reached by the larger radius.
    let x = 26;
    let small = GtaoConfig { world_radius: 0.8, ..GtaoConfig::default() };
    let large = GtaoConfig { world_radius: 3.0, ..GtaoConfig::default() };
    let occ_small = gtao_pixel(buffers, CAMERA, small, x, 32);
    let occ_large = gtao_pixel(buffers, CAMERA, large, x, 32);
    assert!(
        occ_large < occ_small,
        "wider radius must find the step: small={occ_small} large={occ_large}"
    );
}

#[test]
fn visibility_stays_in_unit_range() {
    let (width, height) = (24, 24);
    let mut depth = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            depth[y * width + x] = 4.0 + ((x * 7 + y * 3) % 11) as f32 * 0.5;
        }
    }
    let normals = vec![[0.0, 0.0, 1.0]; width * height];
    let buffers = GtaoBuffers { width, height, linear_depth: &depth, view_normals: &normals };
    let config = GtaoConfig { world_radius: 2.0, power: 1.5, ..GtaoConfig::default() };
    for value in compute_gtao(buffers, CAMERA, config) {
        assert!((0.0..=1.0).contains(&value), "AO out of range: {value}");
    }
}

#[test]
fn higher_power_darkens_occluded_pixels() {
    let (width, height) = (64, 64);
    let (depth, normals) = stepped_scene(width, height);
    let buffers = GtaoBuffers { width, height, linear_depth: &depth, view_normals: &normals };
    let soft = GtaoConfig { world_radius: 1.5, power: 1.0, ..GtaoConfig::default() };
    let sharp = GtaoConfig { power: 3.0, ..soft };
    let seam_soft = gtao_pixel(buffers, CAMERA, soft, 30, 32);
    let seam_sharp = gtao_pixel(buffers, CAMERA, sharp, 30, 32);
    assert!(
        seam_sharp < seam_soft,
        "higher power should darken: soft={seam_soft} sharp={seam_sharp}"
    );
}

#[test]
fn slice_integral_open_hemisphere_is_unity() {
    let visibility = slice_visibility(0.0, 0.0, 0.0, 1.0);
    assert!((visibility - 1.0).abs() < 1.0e-5, "expected 1.0, got {visibility}");
}

#[test]
fn slice_integral_closed_hemisphere_is_zero() {
    let visibility = slice_visibility(1.0, 1.0, 0.0, 1.0);
    assert!(visibility.abs() < 1.0e-5, "expected 0.0, got {visibility}");
}

#[test]
fn slice_integral_is_monotonic_in_horizon() {
    let open = slice_visibility(0.0, 0.0, 0.0, 1.0);
    let half = slice_visibility(0.0, 0.5, 0.0, 1.0);
    let closed = slice_visibility(0.0, 1.0, 0.0, 1.0);
    assert!(open > half && half > closed, "open={open} half={half} closed={closed}");
}

#[test]
fn tilted_normal_reduces_projected_weight() {
    assert_eq!(HALF_PI, core::f32::consts::FRAC_PI_2);
    let straight = slice_visibility(0.0, 0.0, 0.0, 1.0);
    let tilted = slice_visibility(0.0, 0.0, 0.0, 0.25);
    assert!(tilted < straight, "tilted={tilted} straight={straight}");
}

#[test]
fn horizon_combination_keeps_the_maximum() {
    assert_eq!(combine_horizon(0.2, 0.6), 0.6);
    assert_eq!(combine_horizon(0.6, 0.1), 0.6);
}

#[test]
fn distance_weight_eases_from_start_to_radius() {
    // Solid inside the falloff start, zero at (and past) the radius, monotone
    // decreasing in between.
    assert_eq!(distance_weight(0.5, 1.0, 2.0), 1.0);
    assert_eq!(distance_weight(1.0, 1.0, 2.0), 1.0);
    assert!((distance_weight(2.0, 1.0, 2.0)).abs() < 1.0e-6);
    assert!(distance_weight(3.0, 1.0, 2.0).abs() < 1.0e-6);
    let mid = distance_weight(1.5, 1.0, 2.0);
    assert!(mid > 0.0 && mid < 1.0, "midpoint weight {mid} should be in (0, 1)");
}

#[test]
fn camera_reconstruct_round_trips_center_and_scales_radius() {
    let camera = GtaoCamera::from_projection(2.0, 2.0); // tan_half = 0.5
    let center = camera.reconstruct([0.5, 0.5], 7.0);
    assert!(center[0].abs() < 1.0e-6 && center[1].abs() < 1.0e-6);
    assert!((center[2] + 7.0).abs() < 1.0e-6);
    let near = camera.uv_radius(1.0, 5.0)[0];
    let far = camera.uv_radius(1.0, 10.0)[0];
    assert!(far < near, "near={near} far={far}");
}
