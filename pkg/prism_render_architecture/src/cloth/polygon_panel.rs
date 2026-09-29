//! Arbitrary simple-polygon pattern panels triangulated into a cloth membrane.
//!
//! The sibling [`super::panel`] module cuts a garment from *rectangular*
//! warp/weft grids. Real pattern authoring (`Marvelous` Designer / CLO, and the
//! panel workflow UE5 `Chaos` Cloth ships) cuts panels as *arbitrary flat
//! polygons* — a sleeve, a gore, a collar facing — whose silhouette is a simple
//! (non-self-intersecting) closed outline, not a box. This module turns one
//! such authored polygon into a solver-ready sim mesh: a triangulated membrane
//! of [`ClothParticle`]s plus the woven [`Constraint`] graph the `XPBD` solver
//! projects.
//!
//! The triangulation is a deterministic **ear-clipping** pass (O(n²), the
//! textbook simple-polygon algorithm). Ear clipping needs no external
//! dependency, handles convex and concave outlines, and — clipping the
//! lowest-index valid ear each step over the crate-local [`Vec3`] / 2D math —
//! is bit-reproducible: the same outline always yields the same triangle list,
//! so the mesh can be golden-tested. Only `sqrt` is used (via
//! [`Vec3::distance`]); no transcendental is called.
//!
//! Once triangulated the module weaves the physical constraints the way a
//! production triangle-mesh cloth does:
//!
//! * every unique triangle edge becomes a structural [`ConstraintKind::Stretch`]
//!   distance constraint, classified as warp- or weft-aligned from its
//!   panel-local direction so the fabric keeps its authored anisotropy;
//! * every interior edge shared by two triangles becomes a
//!   [`ConstraintKind::Bend`] distance constraint spanning the two opposing apex
//!   vertices — the standard discrete dihedral-bending stand-in for a triangle
//!   membrane.
//!
//! Node masses are **lumped** from the incident triangle areas (each triangle
//! donates a third of `density * area` to each of its vertices), which is the
//! physically faithful mass distribution for an irregular mesh rather than the
//! uniform per-node mass a regular grid can assume.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::asset::{FabricMaterial, Panel, PanelId};
use super::{ClothParticle, Compliance, Constraint, ConstraintKind, Vec3};

/// Twice-area magnitude below which a triangle (or an ear candidate) is treated
/// as degenerate, so a collinear or zero-area sliver never becomes an ear and
/// never contributes a `NaN` normal or a division by zero.
const TWICE_AREA_EPS: f32 = 1.0e-9;

/// Distance below which a generated constraint is dropped as degenerate, so two
/// coincident polygon vertices never emit a `rest_length == 0` spring.
const MIN_REST: f32 = 1.0e-6;

/// Lower bound on a node mass before taking its reciprocal, so a zero-density
/// fabric (or an unreferenced vertex) yields a finite, very large inverse mass
/// instead of a division by zero or an infinite/`NaN` value.
const MIN_MASS: f32 = 1.0e-6;

/// An arbitrary flat pattern panel: a simple closed polygon in panel-local 2D
/// coordinates, placed into world space by an orthonormal basis.
///
/// `boundary` lists the polygon's corner points in order around the outline
/// (either winding is accepted; the triangulator normalizes to counter-
/// clockwise internally). Point `[u, v]` maps to the world position
/// `origin + u * warp_axis + v * weft_axis`, so `u` runs along the warp
/// (lengthwise) yarns and `v` along the weft (crosswise) yarns. The caller
/// supplies the `warp_axis` / `weft_axis` basis (expected orthonormal); this
/// type only scales and adds, so no transcendental rotation is needed.
#[derive(Clone, Debug, PartialEq)]
pub struct PolygonPanel {
    /// Stable identity of this panel within the garment.
    pub id: PanelId,
    /// The closed outline, in panel-local `[warp, weft]` coordinates (metres).
    /// The closing edge from the last point back to the first is implicit.
    pub boundary: Vec<[f32; 2]>,
    /// World-space position of panel-local origin `[0, 0]`.
    pub origin: Vec3,
    /// Unit warp (lengthwise) axis; panel-local `u` scales this.
    pub warp_axis: Vec3,
    /// Unit weft (crosswise) axis; panel-local `v` scales this.
    pub weft_axis: Vec3,
}

impl PolygonPanel {
    /// Builds a polygon panel from its outline and placement basis.
    #[must_use]
    pub fn new(
        id: PanelId,
        boundary: Vec<[f32; 2]>,
        origin: Vec3,
        warp_axis: Vec3,
        weft_axis: Vec3,
    ) -> Self {
        Self {
            id,
            boundary,
            origin,
            warp_axis,
            weft_axis,
        }
    }

    /// Number of outline corners (and therefore sim-mesh particles).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.boundary.len()
    }

    /// Maps a panel-local `[u, v]` point to its world-space position.
    #[must_use]
    pub fn position_3d(&self, local: [f32; 2]) -> Vec3 {
        self.origin
            .add(self.warp_axis.scale(local[0]))
            .add(self.weft_axis.scale(local[1]))
    }

    /// Signed area of the outline (positive when the points wind
    /// counter-clockwise), via the shoelace formula.
    ///
    /// Used both to normalize the winding for triangulation and to golden-test
    /// that the triangle fan conserves the polygon's area.
    #[must_use]
    pub fn signed_area(&self) -> f32 {
        signed_area_2d(&self.boundary)
    }

    /// Triangulates the outline into a fan of triangles indexing the original
    /// `boundary` order, by ear clipping.
    ///
    /// Returns an empty list when the outline has fewer than three corners or
    /// is fully degenerate (zero area). Each triangle is wound
    /// counter-clockwise in panel-local space regardless of the input winding,
    /// so a consistent front face is emitted.
    #[must_use]
    pub fn triangulate(&self) -> Vec<[u32; 3]> {
        ear_clip(&self.boundary)
    }

    /// Triangulates the panel and weaves it into a solver-ready membrane.
    ///
    /// The returned [`PolygonPanelMesh`] carries the sim particles (lumped-mass
    /// from the incident triangle areas), the de-duplicated structural / bending
    /// constraint graph, the triangle list (for aerodynamics, self-collision and
    /// render embedding) and the panel's compact vertex-range record. The build
    /// is a pure function of the panel and material, so calling it twice yields
    /// identical output.
    #[must_use]
    pub fn build(&self, material: FabricMaterial) -> PolygonPanelMesh {
        let triangles = self.triangulate();
        let particles = self.lumped_particles(&triangles, material);
        let constraints = weave_constraints(&self.boundary, &triangles, &particles, material);
        let panel = Panel::new(self.id, 0, self.vertex_count() as u32);
        PolygonPanelMesh {
            particles,
            constraints,
            triangles,
            panel,
        }
    }

    /// Builds the sim particles, distributing fabric mass by lumping a third of
    /// each incident triangle's `density * area` onto each of its vertices.
    ///
    /// A vertex touched by no triangle (only possible for a degenerate outline)
    /// keeps the `MIN_MASS` floor, so its inverse mass is finite. The result is
    /// never pinned here; pins are an authoring concern layered on afterwards.
    fn lumped_particles(
        &self,
        triangles: &[[u32; 3]],
        material: FabricMaterial,
    ) -> Vec<ClothParticle> {
        let density = material.sanitized().density;
        let count = self.boundary.len();
        let mut mass = alloc::vec![0.0_f32; count];

        for tri in triangles {
            let area = triangle_area_2d(&self.boundary, *tri);
            let share = density * area / 3.0;
            for &vi in tri {
                if let Some(slot) = mass.get_mut(vi as usize) {
                    *slot += share;
                }
            }
        }

        (0..count)
            .map(|i| {
                let m = mass[i].max(MIN_MASS);
                let inverse_mass = 1.0 / m;
                ClothParticle::new(self.position_3d(self.boundary[i]), inverse_mass)
            })
            .collect()
    }
}

/// A triangulated polygon panel welded into a solver-ready membrane.
///
/// Mirrors the shape of [`super::panel::GarmentMesh`] but additionally carries
/// the `triangles` the aerodynamics, self-collision and render-embed passes
/// consume — a triangle membrane is defined by its faces, not just its edges.
#[derive(Clone, Debug, PartialEq)]
pub struct PolygonPanelMesh {
    /// Sim-mesh particles, one per outline corner, in `boundary` order.
    pub particles: Vec<ClothParticle>,
    /// De-duplicated woven constraint graph over the particles.
    pub constraints: Vec<Constraint>,
    /// The counter-clockwise triangle fan covering the polygon.
    pub triangles: Vec<[u32; 3]>,
    /// This panel's ownership of the compact sim-vertex range.
    pub panel: Panel,
}

/// Signed area of a 2D polygon via the shoelace formula (positive == CCW).
fn signed_area_2d(points: &[[f32; 2]]) -> f32 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        acc += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * acc
}

/// Twice the (unsigned) area of triangle `tri` over the polygon `points`, i.e.
/// the magnitude of the 2D cross product of its two edges.
fn twice_area_of(points: &[[f32; 2]], a: usize, b: usize, c: usize) -> f32 {
    let pa = points[a];
    let pb = points[b];
    let pc = points[c];
    let abx = pb[0] - pa[0];
    let aby = pb[1] - pa[1];
    let acx = pc[0] - pa[0];
    let acy = pc[1] - pa[1];
    abx * acy - aby * acx
}

/// Unsigned area of one triangle of the polygon, guarding against a missing
/// index (a degenerate triangle then contributes zero).
fn triangle_area_2d(points: &[[f32; 2]], tri: [u32; 3]) -> f32 {
    let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
    if a >= points.len() || b >= points.len() || c >= points.len() {
        return 0.0;
    }
    0.5 * twice_area_of(points, a, b, c).abs()
}

/// Returns `true` when point `p` lies inside the CCW triangle `(a, b, c)`,
/// treating the closed triangle (edges included) as containment.
///
/// Uses the three edge cross-product signs. A point lying on an edge (a
/// near-zero sign) counts as *contained*: this is what makes ear clipping
/// robust on concave outlines, where a reflex vertex can sit exactly on a
/// candidate ear's diagonal. Accepting such an ear would carve a triangle
/// across the concavity, so the on-edge case must block the ear. The caller
/// never passes the ear's own three corners, so a shared vertex never blocks
/// its own ear. All comparisons are against a small epsilon, never bit
/// equality.
fn point_in_ccw_triangle(points: &[[f32; 2]], a: usize, b: usize, c: usize, p: usize) -> bool {
    let edge = |u: usize, v: usize| -> f32 {
        let pu = points[u];
        let pv = points[v];
        let pp = points[p];
        (pv[0] - pu[0]) * (pp[1] - pu[1]) - (pv[1] - pu[1]) * (pp[0] - pu[0])
    };
    let d0 = edge(a, b);
    let d1 = edge(b, c);
    let d2 = edge(c, a);
    d0 >= -TWICE_AREA_EPS && d1 >= -TWICE_AREA_EPS && d2 >= -TWICE_AREA_EPS
}

/// Ear-clips a simple polygon into a CCW triangle fan indexing `points`.
///
/// Normalizes the winding to counter-clockwise, then repeatedly clips the
/// lowest-index convex vertex whose ear triangle contains no other reflex
/// vertex. The scan is deterministic, so the same outline always produces the
/// same triangle list. A simple polygon always has an ear, so the only way the
/// loop exits early is a degenerate or self-intersecting input, in which case
/// the triangles produced so far are returned rather than looping forever.
fn ear_clip(points: &[[f32; 2]]) -> Vec<[u32; 3]> {
    let n = points.len();
    if n < 3 {
        return Vec::new();
    }

    // Working ring of original indices, ordered counter-clockwise.
    let mut ring: Vec<usize> = (0..n).collect();
    if signed_area_2d(points) < 0.0 {
        ring.reverse();
    }

    let mut triangles = Vec::with_capacity(n.saturating_sub(2));

    // Each iteration removes one vertex; the inner scan is O(n), so the whole
    // pass is O(n²). `guard` bounds the outer loop so a pathological input can
    // never spin: at most `n` successful clips are possible.
    while ring.len() > 3 {
        let m = ring.len();
        let mut clipped = false;

        for k in 0..m {
            let prev = ring[(k + m - 1) % m];
            let curr = ring[k];
            let next = ring[(k + 1) % m];

            // Convex corner test (CCW): the ear apex must turn left.
            if twice_area_of(points, prev, curr, next) <= TWICE_AREA_EPS {
                continue;
            }

            // No other (reflex) vertex may fall inside the candidate ear.
            let mut contains = false;
            for &other in &ring {
                if other == prev || other == curr || other == next {
                    continue;
                }
                if point_in_ccw_triangle(points, prev, curr, next, other) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }

            triangles.push([prev as u32, curr as u32, next as u32]);
            ring.remove(k);
            clipped = true;
            break;
        }

        if !clipped {
            // Degenerate/self-intersecting outline: stop rather than loop.
            return triangles;
        }
    }

    if ring.len() == 3 && twice_area_of(points, ring[0], ring[1], ring[2]).abs() > TWICE_AREA_EPS {
        // A non-degenerate final triangle; a fully collinear remainder is
        // dropped so a zero-area outline yields no triangles.
        triangles.push([ring[0] as u32, ring[1] as u32, ring[2] as u32]);
    }
    triangles
}

/// Classifies a structural edge as warp- or weft-aligned from its panel-local
/// direction and returns the matching compliance.
///
/// The dominant local axis (larger absolute component) picks the yarn family,
/// so an edge running mostly along `u` uses the warp compliance and one running
/// mostly along `v` uses the weft compliance, preserving the fabric anisotropy
/// on an irregular triangle mesh.
fn structural_compliance(
    points: &[[f32; 2]],
    a: usize,
    b: usize,
    material: FabricMaterial,
) -> Compliance {
    let du = (points[b][0] - points[a][0]).abs();
    let dv = (points[b][1] - points[a][1]).abs();
    if du >= dv {
        material.warp_compliance()
    } else {
        material.weft_compliance()
    }
}

/// Weaves the structural (per triangle edge) and bending (per shared interior
/// edge) constraints over the triangulated polygon.
///
/// Structural edges are de-duplicated so an interior edge shared by two
/// triangles emits a single stretch spring; bending constraints span the two
/// apex vertices opposite each interior edge. Every pair is measured from the
/// welded particle positions, self edges and near-zero rest lengths are
/// dropped, and the `(min, max, kind)` key keeps a stretch and a bend over the
/// same pair distinct while never emitting a duplicate.
fn weave_constraints(
    points: &[[f32; 2]],
    triangles: &[[u32; 3]],
    particles: &[ClothParticle],
    material: FabricMaterial,
) -> Vec<Constraint> {
    let mat = material.sanitized();
    let bend_c = mat.bend_compliance();

    let mut out = Vec::new();
    let mut seen: BTreeSet<(u32, u32, u8)> = BTreeSet::new();
    // Sorted interior-edge -> the apex vertices opposite it, to pair the two
    // triangles that share an edge into one bending constraint deterministically.
    let mut edge_apex: BTreeMap<(u32, u32), Vec<u32>> = BTreeMap::new();

    for tri in triangles {
        let [i0, i1, i2] = *tri;
        let edges = [(i0, i1, i2), (i1, i2, i0), (i2, i0, i1)];
        for (a, b, apex) in edges {
            let comp = structural_compliance(points, a as usize, b as usize, mat);
            push_constraint(
                &mut out,
                &mut seen,
                a,
                b,
                particles,
                comp,
                ConstraintKind::Stretch,
            );
            let key = if a < b { (a, b) } else { (b, a) };
            edge_apex.entry(key).or_default().push(apex);
        }
    }

    for (_, apexes) in edge_apex {
        // An interior edge is shared by exactly two triangles; pair their two
        // opposing apexes into a bending constraint. A boundary edge has one
        // apex (no bend); a non-manifold edge with more than two is paired in
        // ascending order so the output stays deterministic.
        if apexes.len() < 2 {
            continue;
        }
        let mut sorted = apexes;
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            push_constraint(
                &mut out,
                &mut seen,
                pair[0],
                pair[1],
                particles,
                bend_c,
                ConstraintKind::Bend,
            );
        }
    }

    out
}

/// Stable `u8` tag for a [`ConstraintKind`], used only as part of the
/// de-duplication key so two different kinds over the same vertex pair are kept
/// distinct.
fn kind_key(kind: ConstraintKind) -> u8 {
    match kind {
        ConstraintKind::Stretch => 0,
        ConstraintKind::Bend => 1,
        ConstraintKind::Shear => 2,
        ConstraintKind::Lra => 3,
        ConstraintKind::Tether => 4,
    }
}

/// Pushes one constraint between two particles, skipping self edges, degenerate
/// near-zero rest lengths and any `(min, max, kind)` pair already emitted. The
/// rest length is measured from the particle positions.
fn push_constraint(
    out: &mut Vec<Constraint>,
    seen: &mut BTreeSet<(u32, u32, u8)>,
    a: u32,
    b: u32,
    particles: &[ClothParticle],
    compliance: Compliance,
    kind: ConstraintKind,
) {
    if a == b {
        return;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (Some(pa), Some(pb)) = (particles.get(a as usize), particles.get(b as usize)) else {
        return;
    };
    let rest = pa.position.distance(pb.position);
    if rest < MIN_REST {
        return;
    }
    if seen.insert((lo, hi, kind_key(kind))) {
        out.push(Constraint::new(lo, hi, rest, compliance, kind));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small tolerance for the `f32` invariant checks; comparisons are always
    /// magnitude-based, never bit equality.
    const EPS: f32 = 1.0e-4;

    /// Builds an axis-aligned panel basis so panel-local `[u, v]` maps directly
    /// to world `(x, y)`, keeping the area/geometry assertions readable.
    fn flat_panel(boundary: Vec<[f32; 2]>) -> PolygonPanel {
        PolygonPanel::new(
            PanelId(7),
            boundary,
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    /// Sums the unsigned areas of a triangulation over `points`, for the
    /// area-conservation invariant (the fan must tile the whole polygon).
    fn triangulated_area(points: &[[f32; 2]], tris: &[[u32; 3]]) -> f32 {
        let mut acc = 0.0;
        for tri in tris {
            acc += triangle_area_2d(points, *tri);
        }
        acc
    }

    /// Counts constraints of one kind, so the woven graph can be checked
    /// structurally without depending on emission order.
    fn count_kind(cs: &[Constraint], kind: ConstraintKind) -> usize {
        cs.iter().filter(|c| c.kind == kind).count()
    }

    fn unit_square_ccw() -> Vec<[f32; 2]> {
        alloc::vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn square_triangulates_into_two_triangles() {
        let panel = flat_panel(unit_square_ccw());
        let tris = panel.triangulate();
        assert_eq!(tris.len(), 2);
        // n - 2 triangles for any simple polygon.
        assert_eq!(tris.len(), panel.vertex_count() - 2);
    }

    #[test]
    fn square_area_is_conserved() {
        let panel = flat_panel(unit_square_ccw());
        let tris = panel.triangulate();
        let area = triangulated_area(&panel.boundary, &tris);
        assert!((area - panel.signed_area().abs()).abs() < EPS);
        assert!((area - 1.0).abs() < EPS);
    }

    #[test]
    fn square_weaves_five_stretch_and_one_bend() {
        let panel = flat_panel(unit_square_ccw());
        let mesh = panel.build(FabricMaterial::default());
        // Four boundary edges plus the one shared diagonal.
        assert_eq!(count_kind(&mesh.constraints, ConstraintKind::Stretch), 5);
        // The single interior diagonal yields exactly one bending constraint,
        // spanning the two corners opposite that diagonal.
        let bends: Vec<&Constraint> = mesh
            .constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Bend)
            .collect();
        assert_eq!(bends.len(), 1);
        let bend = bends[0];
        // The bend joins the two corners not on the shared diagonal; for the
        // unit square those two corners sit opposite each other on the ring, so
        // their indices differ by two.
        let span = if bend.a < bend.b {
            bend.b - bend.a
        } else {
            bend.a - bend.b
        };
        assert_eq!(span, 2);
    }

    #[test]
    fn clockwise_input_is_normalized() {
        // Same square, wound clockwise: the triangulator must normalize the
        // winding and still yield a valid two-triangle, area-conserving fan.
        let cw = alloc::vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let panel = flat_panel(cw);
        let tris = panel.triangulate();
        assert_eq!(tris.len(), 2);
        let area = triangulated_area(&panel.boundary, &tris);
        assert!((area - 1.0).abs() < EPS);
        let mesh = panel.build(FabricMaterial::default());
        assert_eq!(count_kind(&mesh.constraints, ConstraintKind::Stretch), 5);
        assert_eq!(count_kind(&mesh.constraints, ConstraintKind::Bend), 1);
    }

    #[test]
    fn concave_l_shape_triangulates_and_conserves_area() {
        // An L-shaped hexagon (a reflex corner at index 3): ear clipping must
        // avoid cutting across the concavity while still tiling the polygon.
        let l_shape = alloc::vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let panel = flat_panel(l_shape);
        let tris = panel.triangulate();
        assert_eq!(tris.len(), panel.vertex_count() - 2);
        let area = triangulated_area(&panel.boundary, &tris);
        // The L covers three unit squares.
        assert!((area - panel.signed_area().abs()).abs() < EPS);
        assert!((area - 3.0).abs() < EPS);
    }

    #[test]
    fn arrow_concave_polygon_conserves_area() {
        // A four-point chevron with a deep reflex notch at the tail (index 3).
        let arrow = alloc::vec![[0.0, 0.0], [4.0, 2.0], [0.0, 4.0], [1.0, 2.0]];
        let panel = flat_panel(arrow);
        let tris = panel.triangulate();
        assert_eq!(tris.len(), panel.vertex_count() - 2);
        let area = triangulated_area(&panel.boundary, &tris);
        assert!((area - panel.signed_area().abs()).abs() < EPS);
    }

    #[test]
    fn build_is_deterministic() {
        let panel = flat_panel(unit_square_ccw());
        let a = panel.build(FabricMaterial::default());
        let b = panel.build(FabricMaterial::default());
        assert_eq!(a, b);
    }

    #[test]
    fn lumped_masses_are_finite_and_positive() {
        let panel = flat_panel(unit_square_ccw());
        let mesh = panel.build(FabricMaterial::default());
        assert_eq!(mesh.particles.len(), 4);
        for particle in &mesh.particles {
            assert!(particle.inverse_mass.is_finite());
            assert!(particle.inverse_mass > 0.0);
        }
    }

    #[test]
    fn rest_lengths_and_constraints_are_finite() {
        let panel = flat_panel(unit_square_ccw());
        let mesh = panel.build(FabricMaterial::default());
        for c in &mesh.constraints {
            assert!(c.rest_length.is_finite());
            assert!(c.rest_length > 0.0);
            assert!(c.a != c.b);
        }
    }

    #[test]
    fn degenerate_too_few_vertices_yields_empty_mesh() {
        let two = flat_panel(alloc::vec![[0.0, 0.0], [1.0, 0.0]]);
        assert!(two.triangulate().is_empty());
        let mesh = two.build(FabricMaterial::default());
        assert!(mesh.triangles.is_empty());
        assert!(mesh.constraints.is_empty());
        // Both particles still get a finite floored inverse mass, no panic.
        assert_eq!(mesh.particles.len(), 2);
        for particle in &mesh.particles {
            assert!(particle.inverse_mass.is_finite());
            assert!(particle.inverse_mass > 0.0);
        }
    }

    #[test]
    fn collinear_triangle_is_degenerate_without_panic() {
        // Three collinear points have zero area: no ear can be clipped, so the
        // triangulation is empty and nothing panics.
        let line = flat_panel(alloc::vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        assert!(line.triangulate().is_empty());
        let mesh = line.build(FabricMaterial::default());
        assert!(mesh.triangles.is_empty());
        for particle in &mesh.particles {
            assert!(particle.inverse_mass.is_finite());
        }
    }

    #[test]
    fn signed_area_sign_tracks_winding() {
        let ccw = flat_panel(unit_square_ccw());
        assert!(ccw.signed_area() > 0.0);
        let cw = flat_panel(alloc::vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
        assert!(cw.signed_area() < 0.0);
    }

    #[test]
    fn position_3d_maps_local_axes_into_world() {
        let panel = PolygonPanel::new(
            PanelId(1),
            unit_square_ccw(),
            Vec3::new(10.0, 20.0, 30.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let p = panel.position_3d([2.0, 3.0]);
        assert!((p.x - 12.0).abs() < EPS);
        assert!((p.y - 20.0).abs() < EPS);
        assert!((p.z - 33.0).abs() < EPS);
    }
}
