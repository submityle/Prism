//! Cross-module acceptance tests for particle surface reconstruction (M5.5).
//!
//! Validates the field-sampling + marching-tetrahedra pipeline: a dense
//! particle blob reconstructs a closed, edge-manifold mesh with valid indices;
//! a spherical cluster reconstructs an approximately spherical bounding box;
//! and empty input yields an empty mesh.

use std::collections::HashMap;

use glam::Vec3;
use prism_physics_core::math::scalar::Real;
use prism_physics_core::reconstruct::{triangulate, ScalarField, SurfaceMesh};

fn edge_share_counts(mesh: &SurfaceMesh) -> HashMap<(u32, u32), u32> {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a <= b { (a, b) } else { (b, a) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

#[test]
fn empty_input_reconstructs_empty_mesh() {
    let field = ScalarField::from_particles(&[], 0.2, 0.1, 2);
    let mesh = triangulate(&field, 0.5);
    assert!(mesh.is_empty());
    assert_eq!(mesh.triangle_count(), 0);
    assert_eq!(mesh.vertex_count(), 0);
}

#[test]
fn dense_blob_reconstructs_closed_manifold_mesh() {
    let mut pts = Vec::new();
    for k in 0..6 {
        for j in 0..6 {
            for i in 0..6 {
                pts.push(Vec3::new(i as Real, j as Real, k as Real) * 0.05);
            }
        }
    }
    let field = ScalarField::from_particles(&pts, 0.13, 0.05, 2);
    let mesh = triangulate(&field, 0.5);

    assert!(mesh.triangle_count() > 0, "expected non-empty surface");
    assert!(mesh.indices_are_valid(), "indices out of range");

    // Watertight & edge-manifold: every edge shared by exactly two triangles.
    for (&(a, b), &c) in &edge_share_counts(&mesh) {
        assert_eq!(c, 2, "edge ({a},{b}) shared by {c} triangles");
    }

    // Normals are unit length.
    for n in &mesh.normals {
        assert!((n.length() - 1.0).abs() < 1.0e-3, "non-unit normal {n:?}");
    }
}

#[test]
fn spherical_cluster_reconstructs_ball_shaped_bounds() {
    // Particles filling a ball of radius `r` about the origin.
    let r = 0.5;
    let step = 0.05;
    let mut pts = Vec::new();
    let n = 20;
    for iz in 0..=n {
        for iy in 0..=n {
            for ix in 0..=n {
                let p = Vec3::new(
                    -r + ix as Real * step,
                    -r + iy as Real * step,
                    -r + iz as Real * step,
                );
                if p.length() <= r {
                    pts.push(p);
                }
            }
        }
    }
    assert!(!pts.is_empty());

    let field = ScalarField::from_particles(&pts, 0.12, 0.05, 2);
    let mesh = triangulate(&field, 0.5);
    assert!(mesh.triangle_count() > 0);
    assert!(mesh.indices_are_valid());

    // Bounding box of the reconstructed surface should be roughly cubic
    // (a ball's AABB has near-equal extents on every axis).
    let mut lo = Vec3::splat(Real::INFINITY);
    let mut hi = Vec3::splat(Real::NEG_INFINITY);
    for &p in &mesh.positions {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let ext = hi - lo;
    let mean = (ext.x + ext.y + ext.z) / 3.0;
    assert!(mean > 0.0);
    for axis in [ext.x, ext.y, ext.z] {
        assert!(
            (axis - mean).abs() / mean < 0.2,
            "extent {axis} deviates from mean {mean} by >20%"
        );
    }

    // Still a closed manifold.
    for &c in edge_share_counts(&mesh).values() {
        assert_eq!(c, 2, "non-manifold edge, share count {c}");
    }
}
