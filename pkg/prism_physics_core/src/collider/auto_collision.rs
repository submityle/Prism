//! Automatic collision-shape selection (the collision "cooking strategy").
//!
//! Authored art is a raw triangle soup, but a physics runtime wants the
//! *cheapest shape that still collides correctly*. Production engines therefore
//! pick a collision representation per object: a sphere, capsule, or box when a
//! primitive fits tightly; a single convex hull when the mesh is (nearly)
//! convex; a small set of convex hulls when it is concave but closed; and a raw
//! triangle mesh only as a last resort for open or thin surfaces. Unreal's
//! "simple vs complex collision", `PhysX` cooking and Jolt all expose exactly
//! this ladder.
//!
//! [`cook_auto_collision`] implements that ladder on top of Prism's existing,
//! independently written geometry kit:
//!
//! 1. [`cook_collision_shells`] welds the soup and splits it into clean,
//!    connected shells.
//! 2. Each shell is classified by [`select_representation`], which tries the
//!    cheapest adequate representation first and only falls back to richer ones.
//!
//! The whole pass is deterministic: every building block
//! ([`convex_hull`](crate::collider::convex_hull), [`fit_obb`],
//! [`minimal_bounding_sphere`], [`fit_bounding_capsule`],
//! [`measure_solidity`], [`convex_decompose`]) is bit-reproducible, and the
//! shells arrive in a fixed order.
//!
//! # Provenance
//!
//! This module only *composes* Prism primitives behind a decision policy that
//! follows the publicly documented behaviour of collision cookers. It contains
//! **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::collider::bounding_capsule::{fit_bounding_capsule, BoundingCapsule};
use crate::collider::bounding_sphere::{minimal_bounding_sphere, BoundingSphere};
use crate::collider::convex_mesh::ConvexMeshData;
use crate::collider::cook_shells::{cook_collision_shells, CookShellParams};
use crate::collider::decompose::{convex_decompose, DecompositionParams};
use crate::collider::obb::{fit_obb, Obb};
use crate::collider::solidity::measure_solidity;

/// Tuning for [`cook_auto_collision`].
#[derive(Clone, Copy, Debug)]
pub struct AutoCollisionParams {
    /// Welding/splitting parameters applied before classification.
    pub shells: CookShellParams,
    /// A primitive (sphere, capsule, box) is accepted only when its volume
    /// exceeds the shell's convex-hull volume by at most this fraction. Because
    /// every bounding primitive encloses the hull, the ratio is always `>= 1`;
    /// a value of `0.2` therefore accepts a primitive that over-approximates the
    /// hull by no more than 20%.
    pub primitive_volume_tolerance: f32,
    /// A shell whose [`solidity`](crate::collider::MeshSolidity::solidity) is at
    /// least this is represented by a single convex hull rather than being
    /// decomposed. In `[0, 1]`; `1` means "only perfectly convex shells become a
    /// single hull".
    pub convex_solidity_threshold: f32,
    /// Decomposition parameters used when a closed shell is too concave for a
    /// single hull.
    pub decomposition: DecompositionParams,
}

impl Default for AutoCollisionParams {
    fn default() -> Self {
        Self {
            shells: CookShellParams::default(),
            primitive_volume_tolerance: 0.2,
            convex_solidity_threshold: 0.92,
            decomposition: DecompositionParams::default(),
        }
    }
}

impl AutoCollisionParams {
    /// Clamps the knobs into their valid ranges so a caller cannot force a
    /// nonsensical decision (negative tolerance, out-of-range solidity).
    #[must_use]
    fn sanitized(mut self) -> Self {
        if !self.primitive_volume_tolerance.is_finite() || self.primitive_volume_tolerance < 0.0 {
            self.primitive_volume_tolerance = 0.0;
        }
        if !self.convex_solidity_threshold.is_finite() {
            self.convex_solidity_threshold = 1.0;
        }
        self.convex_solidity_threshold = self.convex_solidity_threshold.clamp(0.0, 1.0);
        self
    }
}

/// A single convex hull piece produced by decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexHullPiece {
    /// Hull vertices.
    pub vertices: Vec<Vec3>,
    /// Outward-wound hull triangles indexing into [`Self::vertices`].
    pub indices: Vec<[u32; 3]>,
}

/// The collision representation chosen for one shell, cheapest first.
#[derive(Clone, Debug, PartialEq)]
pub enum CollisionRepresentation {
    /// A bounding sphere tightly enclosed the shell.
    Sphere(BoundingSphere),
    /// A bounding capsule tightly enclosed the shell.
    Capsule(BoundingCapsule),
    /// An oriented bounding box tightly enclosed the shell.
    Box(Obb),
    /// The shell is (nearly) convex; its convex hull is used directly.
    ConvexHull(ConvexHullPiece),
    /// The shell is concave but closed; it is approximated by several hulls.
    ConvexHulls(Vec<ConvexHullPiece>),
    /// The shell is open or too thin to enclose a volume; the raw triangle mesh
    /// is kept for exact (complex) collision.
    TriangleMesh {
        /// Welded shell vertices.
        vertices: Vec<Vec3>,
        /// Shell triangles indexing into [`Self::TriangleMesh::vertices`].
        indices: Vec<[u32; 3]>,
    },
}

impl CollisionRepresentation {
    /// A short, stable tag naming the chosen representation, for logging and
    /// test assertions.
    #[must_use]
    pub fn kind(&self) -> RepresentationKind {
        match self {
            CollisionRepresentation::Sphere(_) => RepresentationKind::Sphere,
            CollisionRepresentation::Capsule(_) => RepresentationKind::Capsule,
            CollisionRepresentation::Box(_) => RepresentationKind::Box,
            CollisionRepresentation::ConvexHull(_) => RepresentationKind::ConvexHull,
            CollisionRepresentation::ConvexHulls(_) => RepresentationKind::ConvexHulls,
            CollisionRepresentation::TriangleMesh { .. } => RepresentationKind::TriangleMesh,
        }
    }
}

/// The representation family, independent of its payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationKind {
    /// [`CollisionRepresentation::Sphere`].
    Sphere,
    /// [`CollisionRepresentation::Capsule`].
    Capsule,
    /// [`CollisionRepresentation::Box`].
    Box,
    /// [`CollisionRepresentation::ConvexHull`].
    ConvexHull,
    /// [`CollisionRepresentation::ConvexHulls`].
    ConvexHulls,
    /// [`CollisionRepresentation::TriangleMesh`].
    TriangleMesh,
}

/// The representation chosen for one connected shell of the input.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellCollision {
    /// Index of this shell in the cooked, deterministically ordered shell list.
    pub shell_index: usize,
    /// The chosen representation.
    pub representation: CollisionRepresentation,
}

/// The complete auto-cooked collision geometry: one entry per surviving shell.
#[derive(Clone, Debug, PartialEq)]
pub struct AutoCollisionResult {
    /// Per-shell chosen representations, in cooked shell order.
    pub shells: Vec<ShellCollision>,
}

impl AutoCollisionResult {
    /// Number of shells represented.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shells.len()
    }

    /// Whether no shell was produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shells.is_empty()
    }

    /// Counts how many shells use the given representation family.
    #[must_use]
    pub fn count_of(&self, kind: RepresentationKind) -> usize {
        self.shells
            .iter()
            .filter(|s| s.representation.kind() == kind)
            .count()
    }
}

/// Minimum convex-hull volume (cubic units) a shell must enclose before any
/// solid (primitive/hull/decomposition) representation is considered; below it
/// the shell is treated as an open or thin surface.
const MIN_SOLID_VOLUME: f32 = 1.0e-9;

/// Cooks a raw triangle soup into per-shell collision geometry, choosing the
/// cheapest adequate representation for each connected shell.
///
/// Returns `None` when [`cook_collision_shells`] rejects the input (empty soup,
/// invalid weld epsilon, or every shell filtered out as noise).
#[must_use]
pub fn cook_auto_collision(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: AutoCollisionParams,
) -> Option<AutoCollisionResult> {
    let params = params.sanitized();
    let cooked = cook_collision_shells(vertices, indices, params.shells)?;

    let shells = cooked
        .shells
        .iter()
        .enumerate()
        .map(|(shell_index, shell)| ShellCollision {
            shell_index,
            representation: select_representation(&shell.vertices, &shell.indices, params),
        })
        .collect();

    Some(AutoCollisionResult { shells })
}

/// Classifies a single welded shell into the cheapest adequate representation.
///
/// The ladder is: tight primitive (sphere -> capsule -> box) -> single convex
/// hull (when solid enough) -> convex decomposition (concave closed shells) ->
/// raw triangle mesh (open/thin shells). The shell is expected to be welded, as
/// produced by [`cook_collision_shells`].
#[must_use]
pub fn select_representation(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: AutoCollisionParams,
) -> CollisionRepresentation {
    let params = params.sanitized();

    // A convex hull is the yardstick for "how much volume is really here". If we
    // cannot even build one, the shell is degenerate (collinear/coplanar/empty):
    // keep it as a triangle mesh.
    let Some(hull) = ConvexMeshData::from_points(vertices) else {
        return triangle_mesh(vertices, indices);
    };
    let hull_volume = hull.volume();
    if hull_volume.is_nan() || hull_volume <= MIN_SOLID_VOLUME {
        return triangle_mesh(vertices, indices);
    }

    // 1) Tight bounding primitives, cheapest collision cost first.
    if let Some(primitive) = fit_tight_primitive(vertices, hull_volume, params) {
        return primitive;
    }

    // 2) A single convex hull for (nearly) convex solids. `measure_solidity`
    // needs a closed, consistently wound shell; an open shell returns `None`.
    match measure_solidity(vertices, indices) {
        Some(solidity) if solidity.solidity >= params.convex_solidity_threshold => {
            convex_hull_from(&hull)
        }
        Some(_) => decompose_or_hull(vertices, indices, &hull, params),
        // Open/thin shell: no reliable enclosed volume -> exact triangle mesh.
        None => triangle_mesh(vertices, indices),
    }
}

/// Tries sphere, then capsule, then box, returning the first whose volume is
/// within tolerance of the hull volume and which encloses every shell vertex.
fn fit_tight_primitive(
    vertices: &[Vec3],
    hull_volume: f32,
    params: AutoCollisionParams,
) -> Option<CollisionRepresentation> {
    let max_ratio = 1.0 + params.primitive_volume_tolerance;

    if let Some(sphere) = minimal_bounding_sphere(vertices)
        && within_ratio(sphere.volume(), hull_volume, max_ratio)
        && vertices.iter().all(|&p| sphere.contains_point(p))
    {
        return Some(CollisionRepresentation::Sphere(sphere));
    }

    if let Some(capsule) = fit_bounding_capsule(vertices) {
        let eps = contain_eps(vertices);
        if within_ratio(capsule.volume(), hull_volume, max_ratio)
            && vertices.iter().all(|&p| capsule.contains(p, eps))
        {
            return Some(CollisionRepresentation::Capsule(capsule));
        }
    }

    if let Some(obb) = fit_obb(vertices)
        && within_ratio(obb.volume(), hull_volume, max_ratio)
        && vertices.iter().all(|&p| obb.contains_point(p))
    {
        return Some(CollisionRepresentation::Box(obb));
    }

    None
}

/// Decomposes a concave closed shell, falling back to its single hull when the
/// decomposer yields nothing usable.
fn decompose_or_hull(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    hull: &ConvexMeshData,
    params: AutoCollisionParams,
) -> CollisionRepresentation {
    let parts = convex_decompose(vertices, indices, params.decomposition);
    let pieces: Vec<ConvexHullPiece> = parts.iter().map(piece_from).collect();
    match pieces.len() {
        0 => convex_hull_from(hull),
        1 => CollisionRepresentation::ConvexHull(pieces.into_iter().next().unwrap()),
        _ => CollisionRepresentation::ConvexHulls(pieces),
    }
}

/// Whether `enclosing >= reference` and overshoots it by at most `max_ratio`.
fn within_ratio(enclosing: f32, reference: f32, max_ratio: f32) -> bool {
    enclosing.is_finite() && enclosing >= reference && enclosing <= reference * max_ratio
}

/// A small, scale-aware containment tolerance derived from the point spread.
fn contain_eps(vertices: &[Vec3]) -> f32 {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &p in vertices {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let extent = (hi - lo).max_element();
    1.0e-4 + 1.0e-4 * extent.max(0.0)
}

/// Builds a [`ConvexHullPiece`] from a cooked [`ConvexMeshData`].
fn piece_from(mesh: &ConvexMeshData) -> ConvexHullPiece {
    ConvexHullPiece {
        vertices: mesh.vertices().to_vec(),
        indices: mesh.triangles().to_vec(),
    }
}

/// Wraps a cooked hull as a single-hull representation.
fn convex_hull_from(hull: &ConvexMeshData) -> CollisionRepresentation {
    CollisionRepresentation::ConvexHull(piece_from(hull))
}

/// Copies a shell into a triangle-mesh representation.
fn triangle_mesh(vertices: &[Vec3], indices: &[[u32; 3]]) -> CollisionRepresentation {
    CollisionRepresentation::TriangleMesh {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    /// Builds a unit-radius icosphere subdivided `levels` times (closed,
    /// shared-vertex). High subdivision makes the hull volume approach the
    /// bounding-sphere volume, so the auto-cooker reads it as a sphere.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f64.sqrt()) as f32;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ]
        .into_iter()
        .map(|v| v.normalize())
        .collect();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut mid: StdHashMap<(u32, u32), u32> = StdHashMap::new();
            let mut next = Vec::new();
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = mid.get(&key) {
                    return m;
                }
                let m = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                let idx = verts.len() as u32;
                verts.push(m);
                mid.insert(key, idx);
                idx
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        (verts, faces)
    }

    /// An axis-aligned box of the given half-extents centred at `center`.
    fn box_mesh(center: Vec3, he: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts: Vec<Vec3> = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ]
        .into_iter()
        .map(|v| center + v * he)
        .collect();
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ];
        (verts, faces)
    }

    /// A closed L-shaped prism: concave in cross-section, so a single box
    /// over-approximates it and decomposition is required.
    fn l_prism() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        // L polygon, counter-clockwise, star-shaped about vertex 0.
        let poly = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        let n = poly.len() as u32;
        let mut verts = Vec::new();
        for &(x, y) in &poly {
            verts.push(Vec3::new(x, y, 0.0));
        }
        for &(x, y) in &poly {
            verts.push(Vec3::new(x, y, 1.0));
        }
        let mut faces = Vec::new();
        // Bottom fan (z = 0), top fan (z = 1).
        for i in 1..(n - 1) {
            faces.push([0, i, i + 1]);
            faces.push([n, n + i + 1, n + i]);
        }
        // Side quads.
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push([i, j, n + j]);
            faces.push([i, n + j, n + i]);
        }
        (verts, faces)
    }

    /// Samples points on the surface of a capsule aligned with the z-axis.
    fn capsule_cloud(radius: f32, half_segment: f32) -> Vec<Vec3> {
        let mut pts = Vec::new();
        let rings = 12;
        let sectors = 16;
        for r in 0..=rings {
            let z = (r as f32 / rings as f32 - 0.5) * 2.0 * half_segment;
            for s in 0..sectors {
                let a = 2.0 * core::f64::consts::PI * (s as f64) / (sectors as f64);
                let x = (radius as f64 * a.cos()) as f32;
                let y = (radius as f64 * a.sin()) as f32;
                pts.push(Vec3::new(x, y, z));
            }
        }
        // Hemispherical caps.
        let cap = 6;
        for c in 1..=cap {
            let phi = core::f64::consts::FRAC_PI_2 * (c as f64) / (cap as f64);
            let rr = (radius as f64 * phi.cos()) as f32;
            let dz = (radius as f64 * phi.sin()) as f32;
            for s in 0..sectors {
                let a = 2.0 * core::f64::consts::PI * (s as f64) / (sectors as f64);
                let x = (rr as f64 * a.cos()) as f32;
                let y = (rr as f64 * a.sin()) as f32;
                pts.push(Vec3::new(x, y, half_segment + dz));
                pts.push(Vec3::new(x, y, -half_segment - dz));
            }
        }
        pts
    }

    #[test]
    fn rejects_empty_input() {
        assert!(cook_auto_collision(&[], &[], AutoCollisionParams::default()).is_none());
    }

    #[test]
    fn sphere_mesh_becomes_sphere() {
        let (v, i) = icosphere(3);
        let result = cook_auto_collision(&v, &i, AutoCollisionParams::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.shells[0].representation.kind(),
            RepresentationKind::Sphere
        );
        if let CollisionRepresentation::Sphere(s) = &result.shells[0].representation {
            assert!((s.radius - 1.0).abs() < 0.05, "radius = {}", s.radius);
            assert!(s.center.length() < 0.05, "center = {:?}", s.center);
        } else {
            panic!("expected sphere");
        }
    }

    #[test]
    fn box_mesh_becomes_box() {
        let (v, i) = box_mesh(Vec3::ZERO, Vec3::splat(1.0));
        let result = cook_auto_collision(&v, &i, AutoCollisionParams::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.shells[0].representation.kind(),
            RepresentationKind::Box
        );
        if let CollisionRepresentation::Box(b) = &result.shells[0].representation {
            assert!((b.volume() - 8.0).abs() < 0.5, "volume = {}", b.volume());
        } else {
            panic!("expected box");
        }
    }

    #[test]
    fn two_disjoint_boxes_make_two_box_shells() {
        let (mut v, mut i) = box_mesh(Vec3::new(-5.0, 0.0, 0.0), Vec3::splat(1.0));
        let (v2, i2) = box_mesh(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0));
        let base = v.len() as u32;
        v.extend_from_slice(&v2);
        i.extend(i2.iter().map(|f| [f[0] + base, f[1] + base, f[2] + base]));
        let result = cook_auto_collision(&v, &i, AutoCollisionParams::default()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.count_of(RepresentationKind::Box), 2);
    }

    #[test]
    fn flat_quad_becomes_triangle_mesh() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let i = vec![[0, 1, 2], [0, 2, 3]];
        let result = cook_auto_collision(&v, &i, AutoCollisionParams::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.shells[0].representation.kind(),
            RepresentationKind::TriangleMesh
        );
    }

    #[test]
    fn concave_prism_decomposes_into_convex_hulls() {
        let (v, i) = l_prism();
        // Tight tolerance so the loose 2x2x1 bounding box is rejected.
        let params = AutoCollisionParams {
            primitive_volume_tolerance: 0.05,
            ..AutoCollisionParams::default()
        };
        let result = cook_auto_collision(&v, &i, params).unwrap();
        assert_eq!(result.len(), 1);
        match &result.shells[0].representation {
            CollisionRepresentation::ConvexHulls(pieces) => {
                assert!(pieces.len() >= 2, "pieces = {}", pieces.len());
                for piece in pieces {
                    assert!(!piece.vertices.is_empty());
                    assert!(!piece.indices.is_empty());
                }
            }
            other => panic!("expected convex hulls, got {:?}", other.kind()),
        }
    }

    #[test]
    fn elongated_cloud_becomes_capsule() {
        let pts = capsule_cloud(0.5, 1.5);
        let rep = select_representation(&pts, &[], AutoCollisionParams::default());
        assert_eq!(rep.kind(), RepresentationKind::Capsule);
        if let CollisionRepresentation::Capsule(c) = rep {
            assert!((c.radius - 0.5).abs() < 0.1, "radius = {}", c.radius);
            assert!((c.height() - 3.0).abs() < 0.3, "height = {}", c.height());
        } else {
            panic!("expected capsule");
        }
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = l_prism();
        let a = cook_auto_collision(&v, &i, AutoCollisionParams::default()).unwrap();
        let b = cook_auto_collision(&v, &i, AutoCollisionParams::default()).unwrap();
        assert_eq!(a, b);
    }
}
