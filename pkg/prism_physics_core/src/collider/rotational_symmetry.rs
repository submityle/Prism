//! Rotational (n-fold) symmetry detection for triangle meshes.
//!
//! Beyond bilateral mirror symmetry (see
//! [`detect_mirror_symmetry`](crate::collider::detect_mirror_symmetry)), many
//! authored assets have a rotational symmetry about an axis: a bolt head, a
//! gear, a column, a wheel. A cooker can exploit this the same way it exploits
//! mirror symmetry — store one angular sector and instance it, or constrain a
//! decimator to keep the sectors matched — so detecting the dominant rotation
//! axis and its order is worthwhile.
//!
//! The search is aligned to the shape's principal frame: for a body with a
//! single rotation axis, that axis is one of the principal axes (the rotation
//! makes the in-plane second moments isotropic, which pins the axis exactly).
//! The three candidate axes pass through the bounding-box centre. For each axis
//! and each candidate order `n` in `2..=max_order`, every area-weighted surface
//! sample is rotated by `2*pi/n` about the axis and its distance back to the
//! surface is measured with
//! [`MeshBvh::closest_point`](crate::collider::MeshBvh::closest_point). The
//! residual, normalised by the bounding diagonal, is the symmetry error. For a
//! given axis the reported order is the one with the smallest residual, with
//! near-ties broken toward the higher order (a 4-fold body is also 2-fold, so
//! the stronger symmetry is reported). The axis with the smallest residual is
//! the dominant rotation axis. The result is deterministic for a fixed seed.
//!
//! This is pure triangle-soup geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use glam::{Mat3, Vec3};

use crate::collider::inertia::principal_axes;
use crate::collider::mesh_bvh::MeshBvh;
use crate::collider::surface_sampling::{sample_surface, SurfaceSampleParams};

/// Tuning for [`detect_rotational_symmetry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotationalSymmetryParams {
    /// Number of area-weighted surface samples used for the rotation test.
    pub sample_count: usize,
    /// Seed for the deterministic area-weighted surface sampler.
    pub seed: u64,
    /// Largest rotational order to test (inclusive). Orders `2..=max_order` are
    /// evaluated; a value below `2` makes [`detect_rotational_symmetry`] return
    /// `None` because there is nothing to test.
    pub max_order: u32,
}

impl Default for RotationalSymmetryParams {
    /// 1024 samples with a fixed seed, testing orders 2 through 8.
    fn default() -> Self {
        Self {
            sample_count: 1024,
            seed: 0x524F_5441_5359_4D4D,
            max_order: 8,
        }
    }
}

/// A candidate rotation axis, the best order found about it, and how closely
/// the mesh matches that rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationalAxis {
    /// Unit direction of the rotation axis.
    pub axis: Vec3,
    /// A point on the axis (the bounding-box centre).
    pub point: Vec3,
    /// The rotational order with the lowest residual about this axis (the `n`
    /// in a `2*pi/n` rotation), in `2..=max_order`.
    pub order: u32,
    /// Root-mean-square distance from rotated samples back to the surface at
    /// the reported order, in mesh units.
    pub rms_error: f32,
    /// Largest distance from any rotated sample back to the surface at the
    /// reported order.
    pub max_error: f32,
    /// `rms_error` divided by the bounding diagonal, in `[0, 1]` for typical
    /// meshes. Scale-independent, so it is the value to threshold on.
    pub normalized_error: f32,
}

impl RotationalAxis {
    /// A `[0, 1]` symmetry score (`1 - normalized_error`, clamped). Higher is
    /// more symmetric.
    #[must_use]
    pub fn score(&self) -> f32 {
        (1.0 - self.normalized_error).clamp(0.0, 1.0)
    }

    /// Whether the mesh is rotationally symmetric about this axis at its
    /// reported order within `tolerance` (a normalised-error threshold, e.g.
    /// `0.02`).
    #[must_use]
    pub fn is_symmetric(&self, tolerance: f32) -> bool {
        self.normalized_error <= tolerance
    }
}

/// The rotational-symmetry analysis of a mesh over its three principal axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshRotationalSymmetry {
    /// The three candidate axes, one per principal axis.
    pub axes: [RotationalAxis; 3],
    /// Index into [`axes`](MeshRotationalSymmetry::axes) of the best
    /// (lowest-error) axis.
    pub best_index: usize,
}

impl MeshRotationalSymmetry {
    /// The dominant (lowest-error) rotation axis.
    #[must_use]
    pub fn best(&self) -> RotationalAxis {
        self.axes[self.best_index]
    }

    /// Whether the best axis is rotationally symmetric within `tolerance`.
    #[must_use]
    pub fn is_rotational(&self, tolerance: f32) -> bool {
        self.best().is_symmetric(tolerance)
    }
}

/// Rotates `point` by `angle` radians about the line through `centre` with unit
/// direction `axis`, via Rodrigues' rotation formula.
fn rotate_about_axis(point: Vec3, centre: Vec3, axis: Vec3, cos: f32, sin: f32) -> Vec3 {
    let v = point - centre;
    let rotated = v * cos + axis.cross(v) * sin + axis * (axis.dot(v) * (1.0 - cos));
    centre + rotated
}

/// Measures the rotation residual for one axis at one order, as
/// `(rms_error, max_error)` in mesh units.
fn order_residual(
    bvh: &MeshBvh,
    samples: &[Vec3],
    centre: Vec3,
    axis: Vec3,
    order: u32,
) -> (f64, f64) {
    let angle = core::f64::consts::TAU / f64::from(order);
    let cos = angle.cos() as f32;
    let sin = angle.sin() as f32;
    let mut sum_sq = 0.0_f64;
    let mut max = 0.0_f64;
    for &p in samples {
        let rotated = rotate_about_axis(p, centre, axis, cos, sin);
        if let Some(hit) = bvh.closest_point(rotated) {
            let d = f64::from(hit.distance_sq).max(0.0).sqrt();
            sum_sq += d * d;
            if d > max {
                max = d;
            }
        }
    }
    let n = samples.len() as f64;
    ((sum_sq / n).sqrt(), max)
}

/// Detects the dominant rotation axis and order of a triangle mesh.
///
/// Returns `None` when the mesh is empty, when `sample_count` is zero, when
/// `max_order < 2`, when a BVH cannot be built, when surface sampling fails, or
/// when the shape is degenerate (zero bounding diagonal). The analysis tests
/// the three principal axes through the bounding-box centre and is
/// deterministic for a fixed seed.
#[must_use]
pub fn detect_rotational_symmetry(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: RotationalSymmetryParams,
) -> Option<MeshRotationalSymmetry> {
    if vertices.is_empty() || indices.is_empty() || params.sample_count == 0 || params.max_order < 2
    {
        return None;
    }

    let bvh = MeshBvh::build(vertices, indices)?;
    let (bmin, bmax) = bvh.local_aabb();
    let diagonal = (bmax - bmin).length();
    if !(diagonal.is_finite() && diagonal > 0.0) {
        return None;
    }

    let surface = sample_surface(
        vertices,
        indices,
        SurfaceSampleParams {
            count: params.sample_count,
            seed: params.seed,
        },
    )?;
    if surface.is_empty() {
        return None;
    }
    let positions: Vec<Vec3> = surface.iter().map(|s| s.position).collect();

    // The rotation axis of a symmetric body passes through its centroid, not its
    // bounding-box centre (consider a triangular prism, whose cross-section
    // centroid is offset from the AABB centre). Use the exact area-weighted
    // surface centroid, which is deterministic and noise-free.
    let (mut cx, mut cy, mut cz) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut area_acc = 0.0_f64;
    for tri in indices {
        let p0 = vertices[tri[0] as usize];
        let p1 = vertices[tri[1] as usize];
        let p2 = vertices[tri[2] as usize];
        let area = 0.5 * f64::from((p1 - p0).cross(p2 - p0).length());
        let c = (p0 + p1 + p2) / 3.0;
        cx += f64::from(c.x) * area;
        cy += f64::from(c.y) * area;
        cz += f64::from(c.z) * area;
        area_acc += area;
    }
    let centre = if area_acc > 0.0 {
        Vec3::new(
            (cx / area_acc) as f32,
            (cy / area_acc) as f32,
            (cz / area_acc) as f32,
        )
    } else {
        (bmin + bmax) * 0.5
    };

    // Vertex covariance about the centre pins the principal frame noise-free:
    // for a rotationally symmetric body the in-plane moments are isotropic and
    // the rotation axis lands exactly on a principal axis, which a sampled
    // covariance only approximates.
    let vcount = vertices.len() as f64;
    let (mut xx, mut yy, mut zz) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut xy, mut xz, mut yz) = (0.0_f64, 0.0_f64, 0.0_f64);
    for v in vertices {
        let dx = f64::from(v.x - centre.x);
        let dy = f64::from(v.y - centre.y);
        let dz = f64::from(v.z - centre.z);
        xx += dx * dx;
        yy += dy * dy;
        zz += dz * dz;
        xy += dx * dy;
        xz += dx * dz;
        yz += dy * dz;
    }
    let covariance = Mat3::from_cols(
        Vec3::new(
            (xx / vcount) as f32,
            (xy / vcount) as f32,
            (xz / vcount) as f32,
        ),
        Vec3::new(
            (xy / vcount) as f32,
            (yy / vcount) as f32,
            (yz / vcount) as f32,
        ),
        Vec3::new(
            (xz / vcount) as f32,
            (yz / vcount) as f32,
            (zz / vcount) as f32,
        ),
    );
    let frame = principal_axes(covariance);
    let candidate_axes = [
        frame.axes.x_axis.normalize_or_zero(),
        frame.axes.y_axis.normalize_or_zero(),
        frame.axes.z_axis.normalize_or_zero(),
    ];

    let diagonal_f64 = f64::from(diagonal);
    let mut axes = [RotationalAxis {
        axis: Vec3::X,
        point: centre,
        order: 2,
        rms_error: diagonal,
        max_error: diagonal,
        normalized_error: 1.0,
    }; 3];

    for (slot, &axis) in candidate_axes.iter().enumerate() {
        // A degenerate axis cannot define a rotation; mark it maximally
        // asymmetric.
        if axis.length_squared() < 0.5 {
            continue;
        }

        // Evaluate every order, then pick the smallest residual with near-ties
        // broken toward the higher order (a 4-fold body is also 2-fold).
        let mut best_order = 2_u32;
        let mut best_rms = f64::INFINITY;
        let mut best_max = 0.0_f64;
        for order in 2..=params.max_order {
            let (rms, max) = order_residual(&bvh, &positions, centre, axis, order);
            // Prefer a strictly lower residual; within a small tolerance of the
            // best so far, prefer the higher order (orders are ascending, so a
            // later order wins the tie).
            if rms <= best_rms + 1e-4 * diagonal_f64 {
                best_order = order;
                best_rms = rms;
                best_max = max;
            }
        }

        axes[slot] = RotationalAxis {
            axis,
            point: centre,
            order: best_order,
            rms_error: best_rms as f32,
            max_error: best_max as f32,
            normalized_error: (best_rms / diagonal_f64) as f32,
        };
    }

    // The dominant axis is the strongest symmetry: among axes whose residual is
    // within a small margin of the global minimum, prefer the higher order (a
    // 4-fold axis beats a 2-fold one), breaking remaining ties by lower error.
    // A square or triangular prism, for example, has both its principal
    // rotation axis and secondary 2-fold axes, and the principal one should win.
    let min_err = axes
        .iter()
        .map(|a| a.normalized_error)
        .fold(f32::INFINITY, f32::min);
    let margin = 0.01_f32;
    let best_index = axes
        .iter()
        .enumerate()
        .filter(|(_, a)| a.normalized_error <= min_err + margin)
        .max_by(|a, b| {
            a.1.order
                .cmp(&b.1.order)
                .then_with(|| b.1.normalized_error.total_cmp(&a.1.normalized_error))
        })
        .map_or(0, |(i, _)| i);

    Some(MeshRotationalSymmetry { axes, best_index })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square prism: a `2*2` square cross-section in the XY plane extruded
    /// along Z from `-half_length` to `half_length`. It is 4-fold symmetric
    /// about Z.
    fn square_prism(half_length: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let z0 = -half_length;
        let z1 = half_length;
        let verts = vec![
            Vec3::new(-1.0, -1.0, z0),
            Vec3::new(1.0, -1.0, z0),
            Vec3::new(1.0, 1.0, z0),
            Vec3::new(-1.0, 1.0, z0),
            Vec3::new(-1.0, -1.0, z1),
            Vec3::new(1.0, -1.0, z1),
            Vec3::new(1.0, 1.0, z1),
            Vec3::new(-1.0, 1.0, z1),
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        (verts, tris)
    }

    /// A regular triangular prism: an equilateral triangle (circumradius 1) in
    /// the XY plane extruded along Z. It is 3-fold symmetric about Z.
    fn triangular_prism(half_length: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let z0 = -half_length;
        let z1 = half_length;
        let s = 0.866_025_4_f32; // sqrt(3)/2
        let a = Vec3::new(0.0, 1.0, 0.0);
        let b = Vec3::new(-s, -0.5, 0.0);
        let c = Vec3::new(s, -0.5, 0.0);
        let verts = vec![
            Vec3::new(a.x, a.y, z0),
            Vec3::new(b.x, b.y, z0),
            Vec3::new(c.x, c.y, z0),
            Vec3::new(a.x, a.y, z1),
            Vec3::new(b.x, b.y, z1),
            Vec3::new(c.x, c.y, z1),
        ];
        let tris = vec![
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ];
        (verts, tris)
    }

    /// An irregular (scalene) tetrahedron with no rotational symmetry.
    fn irregular_tetrahedron() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        (verts, tris)
    }

    #[test]
    fn empty_or_degenerate_params_rejected() {
        let (verts, tris) = square_prism(2.0);
        assert!(
            detect_rotational_symmetry(&[], &[], RotationalSymmetryParams::default()).is_none()
        );
        let zero_samples = RotationalSymmetryParams {
            sample_count: 0,
            ..RotationalSymmetryParams::default()
        };
        assert!(detect_rotational_symmetry(&verts, &tris, zero_samples).is_none());
        let no_order = RotationalSymmetryParams {
            max_order: 1,
            ..RotationalSymmetryParams::default()
        };
        assert!(detect_rotational_symmetry(&verts, &tris, no_order).is_none());
    }

    #[test]
    fn square_prism_is_four_fold_about_its_long_axis() {
        let (verts, tris) = square_prism(2.0);
        let sym = detect_rotational_symmetry(&verts, &tris, RotationalSymmetryParams::default())
            .expect("prism analyses");
        let best = sym.best();
        assert!(
            sym.is_rotational(0.02),
            "best normalized error = {}",
            best.normalized_error
        );
        assert_eq!(
            best.order, 4,
            "expected 4-fold symmetry, got {}",
            best.order
        );
        // The dominant axis must be the long (Z) axis of the prism.
        assert!(best.axis.normalize_or_zero().z.abs() > 0.99);
        assert!(best.score() > 0.95);
    }

    #[test]
    fn triangular_prism_is_three_fold_about_its_long_axis() {
        let (verts, tris) = triangular_prism(2.0);
        let sym = detect_rotational_symmetry(&verts, &tris, RotationalSymmetryParams::default())
            .expect("prism analyses");
        let best = sym.best();
        assert!(
            sym.is_rotational(0.03),
            "best normalized error = {}",
            best.normalized_error
        );
        assert_eq!(
            best.order, 3,
            "expected 3-fold symmetry, got {}",
            best.order
        );
        assert!(best.axis.normalize_or_zero().z.abs() > 0.99);
    }

    #[test]
    fn irregular_tetrahedron_has_no_rotational_symmetry() {
        let (verts, tris) = irregular_tetrahedron();
        let sym = detect_rotational_symmetry(&verts, &tris, RotationalSymmetryParams::default())
            .expect("prism analyses");
        assert!(
            sym.best().normalized_error > 0.03,
            "best normalized error unexpectedly low = {}",
            sym.best().normalized_error
        );
    }

    #[test]
    fn result_is_deterministic_for_fixed_seed() {
        let (verts, tris) = square_prism(2.5);
        let params = RotationalSymmetryParams {
            sample_count: 512,
            seed: 777,
            max_order: 6,
        };
        let a = detect_rotational_symmetry(&verts, &tris, params).expect("run a");
        let b = detect_rotational_symmetry(&verts, &tris, params).expect("run b");
        assert_eq!(a, b);
    }
}
