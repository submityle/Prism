//! Integration tests for the M9 convex Voronoi fracture module.
//!
//! These cover the externally observable guarantees of the fracture pipeline:
//! the fragments tile the source solid (volume conservation), a single seed
//! returns the whole shape, a symmetric two-seed cut splits the volume evenly,
//! impact clustering concentrates many small fragments near the contact, and a
//! single-seed cube reproduces the analytic rigid-body inertia tensor.

use glam::Vec3;
use prism_physics_core::{
    fracture_aabb, shatter_box, shatter_box_impact, FractureConfig, Fragment, Real,
};

/// Total volume of a slice of fragments.
fn total_volume(fragments: &[Fragment]) -> Real {
    let mut sum = 0.0;
    for fragment in fragments {
        sum += fragment.volume();
    }
    sum
}

/// Sum of fragment volumes for a uniformly shattered box must recover the box
/// volume, because the fragments form a partition of the solid.
#[test]
fn uniform_shatter_conserves_volume() {
    let config = FractureConfig {
        seed_count: 12,
        min_fragment_volume: 0.0,
        ..FractureConfig::default()
    };
    let fragments = shatter_box(Vec3::ZERO, Vec3::splat(2.0), &config);
    assert!(fragments.len() >= 2);

    let total = total_volume(&fragments);
    let box_volume = 8.0;
    assert!(
        (total - box_volume).abs() < 0.05 * box_volume,
        "total fragment volume {total} deviates from box volume {box_volume}"
    );
}

/// A single seed cannot be cut against anything, so the whole box is returned
/// as one fragment carrying the full volume.
#[test]
fn single_seed_returns_whole_box() {
    let config = FractureConfig {
        seed_count: 1,
        ..FractureConfig::default()
    };
    let fragments = shatter_box(Vec3::ZERO, Vec3::splat(2.0), &config);
    assert_eq!(fragments.len(), 1);
    assert!((fragments[0].volume() - 8.0).abs() < 1e-3);
}

/// Two seeds placed symmetrically across the centre are separated by the plane
/// `x = 0`, so each cell is exactly half of the unit-ish cube.
#[test]
fn symmetric_two_seeds_split_in_half() {
    let config = FractureConfig {
        min_fragment_volume: 0.0,
        ..FractureConfig::default()
    };
    let sites = [Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)];
    let fragments = fracture_aabb(Vec3::splat(-1.0), Vec3::splat(1.0), &sites, &config);
    assert_eq!(fragments.len(), 2);
    for fragment in &fragments {
        assert!(
            (fragment.volume() - 4.0).abs() < 1e-2,
            "half-cell volume {} is not ~4.0",
            fragment.volume()
        );
    }
}

/// Impact clustering must place more, smaller fragments near the contact point
/// than in the far field.
#[test]
fn impact_clustering_concentrates_fragments() {
    let impact = Vec3::splat(0.3);
    let config = FractureConfig {
        seed_count: 32,
        impact_cluster_fraction: 0.75,
        impact_cluster_radius: 0.3,
        min_fragment_volume: 0.0,
        ..FractureConfig::default()
    };
    let fragments = shatter_box_impact(Vec3::ZERO, Vec3::splat(2.0), impact, &config);
    assert!(fragments.len() >= 4);

    let near_radius_sq = 0.6 * 0.6;
    let far_radius_sq = 1.2 * 1.2;

    let mut near_count = 0usize;
    let mut far_count = 0usize;
    let mut near_volume = 0.0;
    let mut far_volume = 0.0;
    for fragment in &fragments {
        let dist_sq = fragment.centroid().distance_squared(impact);
        if dist_sq < near_radius_sq {
            near_count += 1;
            near_volume += fragment.volume();
        } else if dist_sq > far_radius_sq {
            far_count += 1;
            far_volume += fragment.volume();
        }
    }

    assert!(near_count > 0, "expected fragments near the impact");
    assert!(far_count > 0, "expected fragments away from the impact");
    assert!(
        near_count > far_count,
        "impact should crowd fragments near the contact ({near_count} near vs {far_count} far)"
    );

    let mean_near = near_volume / (near_count as Real);
    let mean_far = far_volume / (far_count as Real);
    assert!(
        mean_near < mean_far,
        "near fragments (mean {mean_near}) should be smaller than far ones (mean {mean_far})"
    );
}

/// A single-seed side-2 cube at unit density must reproduce the analytic mass,
/// centroid, and inertia tensor of a solid cube.
#[test]
fn single_seed_cube_matches_analytic_inertia() {
    let config = FractureConfig {
        seed_count: 1,
        ..FractureConfig::default()
    };
    let fragments = shatter_box(Vec3::splat(-1.0), Vec3::splat(1.0), &config);
    assert_eq!(fragments.len(), 1);

    let mass = fragments[0].mass_properties(1.0);
    assert!((mass.volume - 8.0).abs() < 1e-3);
    assert!((mass.mass - 8.0).abs() < 1e-3);
    assert!(mass.centroid.length() < 1e-3);

    // Solid cube: I = m * s^2 / 6 on each diagonal (m = 8, s = 2).
    let expected = 8.0 * 4.0 / 6.0;
    assert!((mass.inertia.x_axis.x - expected).abs() < 1e-2);
    assert!((mass.inertia.y_axis.y - expected).abs() < 1e-2);
    assert!((mass.inertia.z_axis.z - expected).abs() < 1e-2);
    // Off-diagonal terms vanish for a centred cube.
    assert!(mass.inertia.x_axis.y.abs() < 1e-2);
    assert!(mass.inertia.x_axis.z.abs() < 1e-2);
    assert!(mass.inertia.y_axis.z.abs() < 1e-2);
}
