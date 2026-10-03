//! Mesh solidity (convexity ratio) for collision cooking heuristics.
//!
//! A collision cooker often needs to decide whether a shape is "convex enough"
//! to approximate with a single hull or whether it should be split by convex
//! decomposition. The standard scalar for that decision is *solidity*: the
//! ratio of the mesh's enclosed volume to the volume of its convex hull.
//!
//! - A perfectly convex solid has solidity `1`.
//! - The deeper the concavities (or the farther apart disjoint parts sit), the
//!   smaller the ratio.
//!
//! AAA cookers (`PhysX`, `Jolt`) use exactly this measure to gate single-hull
//! versus decomposed collision. This module reuses the existing closed-mesh
//! volume integral ([`full_inertia_tensor`](crate::collider::full_inertia_tensor))
//! and hull builder ([`convex_hull`](crate::collider::convex_hull)); it is pure
//! triangle-soup geometry with no coupling to the collision pipeline, and
//! nothing here is derived from Unreal Engine source.

use glam::Vec3;

use crate::collider::hull::convex_hull;
use crate::collider::inertia::full_inertia_tensor;

/// Unit density used purely to turn the mass integral into a volume reading.
const UNIT_DENSITY: f32 = 1.0;

/// Default tolerance under which [`MeshSolidity::is_convex`] still reports
/// convex, absorbing hull/volume round-off.
pub const DEFAULT_CONVEX_TOLERANCE: f32 = 1.0e-3;

/// The solidity (convexity ratio) of a closed triangle mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshSolidity {
    /// Enclosed volume of the input mesh, in cubic metres.
    pub mesh_volume: f32,
    /// Enclosed volume of the mesh's convex hull, in cubic metres.
    pub convex_volume: f32,
    /// `mesh_volume / convex_volume`, clamped to `[0, 1]`. A convex solid is
    /// `1`; concavities and disjoint parts drive it toward `0`.
    pub solidity: f32,
}

impl MeshSolidity {
    /// Whether the mesh is convex to within `tolerance` (solidity at least
    /// `1 - tolerance`).
    #[must_use]
    pub fn is_convex(&self, tolerance: f32) -> bool {
        self.solidity >= 1.0 - tolerance
    }

    /// Whether the mesh is convex to within [`DEFAULT_CONVEX_TOLERANCE`].
    #[must_use]
    pub fn is_convex_default(&self) -> bool {
        self.is_convex(DEFAULT_CONVEX_TOLERANCE)
    }
}

/// Measures the solidity of a closed, consistently wound triangle mesh.
///
/// Returns `None` when the mesh is empty, when it does not enclose a positive
/// volume (open, flat, or degenerate), or when a convex hull cannot be built
/// from its vertices. The winding may be outward or inward; the enclosed-volume
/// sign is normalised internally. The result is deterministic.
#[must_use]
pub fn measure_solidity(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<MeshSolidity> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let mesh = full_inertia_tensor(vertices, indices, UNIT_DENSITY)?;
    let mesh_volume = mesh.volume;
    if !(mesh_volume.is_finite() && mesh_volume > 0.0) {
        return None;
    }

    let (hull_vertices, hull_indices) = convex_hull(vertices)?;
    let hull = full_inertia_tensor(&hull_vertices, &hull_indices, UNIT_DENSITY)?;
    let convex_volume = hull.volume;
    if !(convex_volume.is_finite() && convex_volume > 0.0) {
        return None;
    }

    let solidity = (mesh_volume / convex_volume).clamp(0.0, 1.0);
    Some(MeshSolidity {
        mesh_volume,
        convex_volume,
        solidity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube translated by `offset`, outward wound and closed.
    fn cube(offset: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let base = [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let v = base.iter().map(|p| *p + offset).collect();
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    #[test]
    fn empty_input_is_rejected() {
        let (v, f) = cube(Vec3::ZERO);
        assert!(measure_solidity(&[], &f).is_none());
        assert!(measure_solidity(&v, &[]).is_none());
    }

    #[test]
    fn open_mesh_is_rejected() {
        // A single triangle encloses no volume.
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(measure_solidity(&verts, &[[0, 1, 2]]).is_none());
    }

    #[test]
    fn convex_cube_is_fully_solid() {
        let (v, f) = cube(Vec3::ZERO);
        let s = measure_solidity(&v, &f).unwrap();
        assert!(
            (s.mesh_volume - 1.0).abs() < 1.0e-4,
            "cube volume {}",
            s.mesh_volume
        );
        assert!((s.convex_volume - 1.0).abs() < 1.0e-4);
        assert!((s.solidity - 1.0).abs() < 1.0e-4, "solidity {}", s.solidity);
        assert!(s.is_convex_default());
    }

    #[test]
    fn two_disjoint_cubes_are_half_solid() {
        // Two unit cubes offset along x by 3 units. Combined mesh volume is 2;
        // their convex hull is exactly the 4x1x1 bounding box (volume 4), so
        // solidity is 1/2.
        let (mut v, mut f) = cube(Vec3::ZERO);
        let (v2, f2) = cube(Vec3::new(3.0, 0.0, 0.0));
        let base = v.len() as u32;
        v.extend(v2);
        f.extend(f2.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));

        let s = measure_solidity(&v, &f).unwrap();
        assert!(
            (s.mesh_volume - 2.0).abs() < 1.0e-4,
            "mesh volume {}",
            s.mesh_volume
        );
        assert!(
            (s.convex_volume - 4.0).abs() < 1.0e-3,
            "hull volume {}",
            s.convex_volume
        );
        assert!((s.solidity - 0.5).abs() < 1.0e-3, "solidity {}", s.solidity);
        assert!(!s.is_convex_default());
    }

    #[test]
    fn solidity_is_bounded_and_deterministic() {
        let (mut v, mut f) = cube(Vec3::ZERO);
        let (v2, f2) = cube(Vec3::new(3.0, 0.0, 0.0));
        let base = v.len() as u32;
        v.extend(v2);
        f.extend(f2.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));

        let a = measure_solidity(&v, &f).unwrap();
        let b = measure_solidity(&v, &f).unwrap();
        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a.solidity));
    }
}
