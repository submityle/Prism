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
        self.build_refined(material, 0)
    }

    /// Triangulates the outline, densifies its interior by `subdivisions`
    /// uniform 1->4 midpoint-subdivision levels, and weaves the densified mesh
    /// into a solver-ready membrane.
    ///
    /// Boundary-only ear clipping fans a large panel into a few oversized
    /// triangles, which under-resolves the drape and starves the aerodynamic /
    /// self-collision passes of surface samples. Each refinement level splits
    /// every triangle into four by its edge midpoints, so `subdivisions = k`
    /// multiplies the triangle count by `4^k` and roughly halves every edge
    /// length per level, giving the interior real resolution while the outline
    /// shape is preserved exactly (a boundary edge's midpoint stays on that
    /// straight segment). Midpoints shared by two triangles are welded to a
    /// single interior vertex, so the refined mesh stays watertight and
    /// manifold. `subdivisions = 0` reproduces the boundary-only triangulation.
    ///
    /// The build is a pure function of the panel, material and level, so calling
    /// it twice yields identical output.
    #[must_use]
    pub fn build_refined(&self, material: FabricMaterial, subdivisions: u32) -> PolygonPanelMesh {
        let (points, triangles) = self.triangulate_refined(subdivisions);
        let particles = self.lumped_particles(&points, &triangles, material);
        let constraints = weave_constraints(&points, &triangles, &particles, material);
        let panel = Panel::new(self.id, 0, points.len() as u32);
        PolygonPanelMesh {
            particles,
            constraints,
            triangles,
            panel,
        }
    }

    /// Ear-clips the outline, then applies `subdivisions` uniform 1->4 midpoint
    /// refinement levels, returning the densified panel-local `[warp, weft]`
    /// vertex list and the conforming CCW triangle list over it.
    ///
    /// The first [`vertex_count`](Self::vertex_count) entries are the original
    /// outline corners in `boundary` order (so a seam or pin that references a
    /// boundary vertex survives refinement unchanged); the generated interior
    /// midpoints follow, appended in a deterministic scan order. `subdivisions
    /// = 0` returns the raw ear-clipped triangulation over the untouched
    /// boundary.
    #[must_use]
    pub fn triangulate_refined(&self, subdivisions: u32) -> (Vec<[f32; 2]>, Vec<[u32; 3]>) {
        let mut points = self.boundary.clone();
        let mut triangles = ear_clip(&points);
        for _ in 0..subdivisions {
            triangles = subdivide_once(&mut points, &triangles);
        }
        (points, triangles)
    }

    /// Chooses the smallest uniform subdivision level whose longest resulting
    /// triangle edge is at or below `target_edge_length` world units, capped at
    /// `max_levels` so a tiny target can never explode the mesh.
    ///
    /// Each level roughly halves every edge, so the count is derived by halving
    /// the longest base-triangulation edge until it meets the target (an integer
    /// loop, never a transcendental `log`). A non-positive or already-satisfied
    /// target returns `0`.
    #[must_use]
    pub fn subdivisions_for_edge_length(&self, target_edge_length: f32, max_levels: u32) -> u32 {
        if !(target_edge_length > 0.0) {
            return 0;
        }
        let triangles = self.triangulate();
        let mut longest = 0.0_f32;
        for tri in &triangles {
            for e in 0..3 {
                let a = tri[e] as usize;
                let b = tri[(e + 1) % 3] as usize;
                let pa = self.position_3d(self.boundary[a]);
                let pb = self.position_3d(self.boundary[b]);
                longest = longest.max(pa.distance(pb));
            }
        }

        let mut level = 0;
        let mut edge = longest;
        while edge > target_edge_length && level < max_levels {
            edge *= 0.5;
            level += 1;
        }
        level
    }

    /// Builds the sim particles, distributing fabric mass by lumping a third of
    /// each incident triangle's `density * area` onto each of its vertices.
    ///
    /// A vertex touched by no triangle (only possible for a degenerate outline)
    /// keeps the `MIN_MASS` floor, so its inverse mass is finite. The result is
    /// never pinned here; pins are an authoring concern layered on afterwards.
    fn lumped_particles(
        &self,
        points: &[[f32; 2]],
        triangles: &[[u32; 3]],
        material: FabricMaterial,
    ) -> Vec<ClothParticle> {
        let density = material.sanitized().density;
        let count = points.len();
        let mut mass = alloc::vec![0.0_f32; count];

        for tri in triangles {
            let area = triangle_area_2d(points, *tri);
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
                ClothParticle::new(self.position_3d(points[i]), inverse_mass)
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

/// Applies one uniform 1->4 midpoint subdivision level to `triangles`.
///
/// Every triangle is split into four by the midpoints of its three edges:
/// three corner children and one central (medial) child, all wound
/// counter-clockwise like their parent. Edge midpoints are appended to
/// `points` and cached by their ascending index pair, so an edge shared by two
/// triangles is split at exactly one common vertex and the refined mesh stays
/// watertight and manifold. Uses only midpoint averaging (no transcendental),
/// so the pass is bit-reproducible.
fn subdivide_once(points: &mut Vec<[f32; 2]>, triangles: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let mut midpoints: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut refined = Vec::with_capacity(triangles.len() * 4);
    for tri in triangles {
        let [a, b, c] = *tri;
        let ab = edge_midpoint(points, &mut midpoints, a, b);
        let bc = edge_midpoint(points, &mut midpoints, b, c);
        let ca = edge_midpoint(points, &mut midpoints, c, a);
        // Three corner children then the central medial child; each preserves
        // the parent's CCW winding.
        refined.push([a, ab, ca]);
        refined.push([b, bc, ab]);
        refined.push([c, ca, bc]);
        refined.push([ab, bc, ca]);
    }
    refined
}

/// Returns the index of the midpoint of edge `(a, b)`, creating and appending
/// it to `points` on first use and reusing the cached vertex thereafter.
///
/// The cache key is the ascending index pair, so the two triangles across a
/// shared edge always resolve to the same midpoint regardless of the local
/// winding they enumerate the edge in — the guarantee that keeps subdivision
/// conforming (no T-junctions) and the vertex set free of duplicates.
fn edge_midpoint(
    points: &mut Vec<[f32; 2]>,
    midpoints: &mut BTreeMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&existing) = midpoints.get(&key) {
        return existing;
    }
    let pa = points[a as usize];
    let pb = points[b as usize];
    let index = points.len() as u32;
    points.push([0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])]);
    midpoints.insert(key, index);
    index
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

    /// Longest triangle edge length (world space) over a densified mesh, for
    /// the target-edge-length assertions.
    fn longest_edge(panel: &PolygonPanel, points: &[[f32; 2]], tris: &[[u32; 3]]) -> f32 {
        let mut longest = 0.0_f32;
        for tri in tris {
            for e in 0..3 {
                let a = tri[e] as usize;
                let b = tri[(e + 1) % 3] as usize;
                let pa = panel.position_3d(points[a]);
                let pb = panel.position_3d(points[b]);
                longest = longest.max(pa.distance(pb));
            }
        }
        longest
    }

    /// Asserts a triangle mesh is edge-manifold: every undirected edge is shared
    /// by at most two triangles (boundary edges once, interior edges twice).
    fn assert_edge_manifold(tris: &[[u32; 3]]) {
        let mut edge_uses: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for tri in tris {
            for e in 0..3 {
                let a = tri[e];
                let b = tri[(e + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_uses.entry(key).or_default() += 1;
            }
        }
        assert!(
            edge_uses.values().all(|&n| n == 1 || n == 2),
            "a densified edge is shared by more than two triangles (non-manifold)"
        );
        // A refined disk keeps at least one interior (twice-shared) edge.
        assert!(edge_uses.values().any(|&n| n == 2));
    }

    #[test]
    fn one_level_quadruples_triangles() {
        let panel = flat_panel(unit_square_ccw());
        let base = panel.triangulate().len();
        let (_, tris) = panel.triangulate_refined(1);
        assert_eq!(tris.len(), base * 4);
        let (_, tris2) = panel.triangulate_refined(2);
        assert_eq!(tris2.len(), base * 16);
    }

    #[test]
    fn refinement_preserves_boundary_vertices() {
        let panel = flat_panel(unit_square_ccw());
        let (points, _) = panel.triangulate_refined(2);
        // The original corners stay first, in outline order, untouched.
        for (i, corner) in panel.boundary.iter().enumerate() {
            assert!((points[i][0] - corner[0]).abs() < EPS);
            assert!((points[i][1] - corner[1]).abs() < EPS);
        }
        // Interior midpoints were appended, growing the vertex set.
        assert!(points.len() > panel.boundary.len());
    }

    #[test]
    fn refinement_conserves_area() {
        let panel = flat_panel(unit_square_ccw());
        for level in 0..=3 {
            let (points, tris) = panel.triangulate_refined(level);
            let area = triangulated_area(&points, &tris);
            assert!(
                (area - 1.0).abs() < EPS,
                "level {level} did not tile the unit square (area = {area})"
            );
        }
    }

    #[test]
    fn refined_mesh_is_watertight_and_manifold() {
        let panel = flat_panel(unit_square_ccw());
        let (_, tris) = panel.triangulate_refined(3);
        assert_edge_manifold(&tris);
    }

    #[test]
    fn refined_triangles_are_all_ccw() {
        let panel = flat_panel(unit_square_ccw());
        let (points, tris) = panel.triangulate_refined(2);
        for tri in &tris {
            let corners = [
                points[tri[0] as usize],
                points[tri[1] as usize],
                points[tri[2] as usize],
            ];
            assert!(
                signed_area_2d(&corners) > TWICE_AREA_EPS,
                "a refined child triangle flipped winding"
            );
        }
    }

    #[test]
    fn refined_masses_are_finite_and_positive() {
        let panel = flat_panel(unit_square_ccw());
        let mesh = panel.build_refined(FabricMaterial::default(), 3);
        assert!(mesh.particles.len() > 4);
        for particle in &mesh.particles {
            assert!(particle.inverse_mass.is_finite());
            assert!(particle.inverse_mass > 0.0);
        }
    }

    #[test]
    fn refined_constraint_graph_grows_with_resolution() {
        let panel = flat_panel(unit_square_ccw());
        let coarse = panel.build_refined(FabricMaterial::default(), 0);
        let fine = panel.build_refined(FabricMaterial::default(), 2);
        assert!(fine.constraints.len() > coarse.constraints.len());
        assert!(count_kind(&fine.constraints, ConstraintKind::Bend) > 0);
        // Rest lengths stay finite and non-degenerate after refinement.
        for c in &fine.constraints {
            assert!(c.rest_length.is_finite());
            assert!(c.rest_length > 0.0);
        }
    }

    #[test]
    fn build_refined_is_deterministic() {
        let panel = flat_panel(unit_square_ccw());
        let a = panel.build_refined(FabricMaterial::default(), 3);
        let b = panel.build_refined(FabricMaterial::default(), 3);
        assert_eq!(a, b);
    }

    #[test]
    fn subdivisions_for_edge_length_meets_target() {
        let panel = flat_panel(unit_square_ccw());
        // The unit square's ear-clip diagonal is sqrt(2) ~= 1.414 world units.
        let target = 0.3;
        let levels = panel.subdivisions_for_edge_length(target, 8);
        assert!(levels > 0);
        let (points, tris) = panel.triangulate_refined(levels);
        assert!(longest_edge(&panel, &points, &tris) <= target + EPS);
    }

    #[test]
    fn subdivisions_for_edge_length_respects_cap_and_noop() {
        let panel = flat_panel(unit_square_ccw());
        // Non-positive target and an already-satisfied target both refine nothing.
        assert_eq!(panel.subdivisions_for_edge_length(0.0, 8), 0);
        assert_eq!(panel.subdivisions_for_edge_length(-1.0, 8), 0);
        assert_eq!(panel.subdivisions_for_edge_length(100.0, 8), 0);
        // A tiny target is clamped to the level cap rather than exploding.
        assert_eq!(panel.subdivisions_for_edge_length(1.0e-6, 3), 3);
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
        let span = bend.a.abs_diff(bend.b);
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
