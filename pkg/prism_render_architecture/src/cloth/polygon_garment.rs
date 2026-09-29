//! Multi-panel garment assembly for arbitrary *polygon* pattern panels.
//!
//! The sibling [`super::panel`] module stitches a garment out of rectangular
//! [`GridPanel`](super::panel::GridPanel)s; [`super::polygon_panel`] cuts a
//! single arbitrary [`PolygonPanel`] into a triangulated membrane. Real
//! garments (a shirt, a sleeve, a gored skirt) are several *polygon* panels
//! sewn edge to edge, so this module supplies the missing assembler:
//! [`PolygonGarmentBuilder`] accumulates any number of polygon panels — each
//! with its own interior-refinement level — plus the seams that sew their
//! boundary edges together and the pins that fix vertices in place, then welds
//! the whole thing into one solver-ready [`PolygonGarmentMesh`].
//!
//! The weld pipeline mirrors [`super::panel::GarmentBuilder`]:
//!
//! 1. **Lay out** every panel's refined sim mesh into one raw vertex list, each
//!    panel occupying a contiguous `base..base + count` block.
//! 2. **Union** each seam's paired boundary vertices in a union-find. A seam
//!    joins one *boundary edge* of panel A to one of panel B; the vertices that
//!    lie along each edge after refinement (its two corners plus the midpoints
//!    the 1->4 subdivision dropped onto that straight segment) are recovered
//!    geometrically, ordered along the edge, and paired one-to-one (optionally
//!    reversing B so the two edges run the same way).
//! 3. **Compact** the welded sets to a dense index range by an ascending scan,
//!    so a panel's welded vertices stay a contiguous block.
//! 4. **Weld particles**: a welded vertex's position is the mean of its raw
//!    positions and its inverse mass is the parallel combination of the raw
//!    inverse masses (a single pinned raw vertex pins the weld).
//! 5. **Remap** every panel's woven constraints and triangles through the weld,
//!    dropping self / duplicate / degenerate edges and collapsed triangles and
//!    recomputing each constraint's rest length from the welded geometry.
//!
//! The build is a pure function of the accumulated panels, seams, and pins, so
//! calling it twice yields identical output; only `sqrt` is used (via
//! [`Vec3::distance`]), no transcendental.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::asset::{FabricMaterial, Panel, PanelId, Seam};
use super::polygon_panel::PolygonPanel;
use super::{ClothParticle, Compliance, Constraint, ConstraintKind, Vec3, EPS_LEN_SQ};

/// Distance below which a generated constraint is dropped as degenerate, so two
/// welded-coincident vertices never emit a `rest_length == 0` spring.
const MIN_REST: f32 = 1.0e-6;

/// A refined vertex counts as lying *on* a boundary edge when its perpendicular
/// distance to that straight segment is within this fraction of the edge's
/// world length (plus [`ON_EDGE_ABS_TOL`] so a very short edge keeps a floor).
/// Midpoint subdivision keeps split points exactly on the parent segment, so
/// only floating-point rounding is being tolerated here.
const ON_EDGE_REL_TOL: f32 = 1.0e-3;

/// Absolute floor for the on-edge perpendicular tolerance (metres).
const ON_EDGE_ABS_TOL: f32 = 1.0e-5;

/// Slack on the `[0, 1]` parameter test so a vertex sitting exactly on a corner
/// (`t == 0` or `t == 1`) is still counted on the edge despite rounding.
const PARAM_EPS: f32 = 1.0e-4;

/// A seam sewing one boundary edge of `panel_a` to one boundary edge of
/// `panel_b`.
///
/// A polygon panel's boundary is its ordered corner list; `edge` names the
/// boundary edge running from corner `edge` to corner `edge + 1` (wrapping),
/// and the seam pairs every vertex lying along `edge_a` with the corresponding
/// vertex along `edge_b`. When the two edges are authored running in opposite
/// directions (the usual case for two panels meeting face to face), set
/// `reversed` so `panel_b`'s vertex order is flipped before pairing.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PolygonSeamSpec {
    /// The first panel joined by this seam.
    pub panel_a: PanelId,
    /// Boundary edge of `panel_a` (corner `edge_a` -> corner `edge_a + 1`).
    pub edge_a: u32,
    /// The second panel joined by this seam.
    pub panel_b: PanelId,
    /// Boundary edge of `panel_b` (corner `edge_b` -> corner `edge_b + 1`).
    pub edge_b: u32,
    /// When `true`, `panel_b`'s edge vertex order is reversed before pairing.
    pub reversed: bool,
}

impl PolygonSeamSpec {
    /// Builds a seam specification between two panel boundary edges.
    #[must_use]
    pub const fn new(
        panel_a: PanelId,
        edge_a: u32,
        panel_b: PanelId,
        edge_b: u32,
        reversed: bool,
    ) -> Self {
        Self {
            panel_a,
            edge_a,
            panel_b,
            edge_b,
            reversed,
        }
    }
}

/// A pin constraint that fixes garment vertices in place (inverse mass forced to
/// zero), modelling a waistband, collar, or anim-driven attachment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PolygonPinSpec {
    /// Pin a single boundary corner of a panel.
    Corner {
        /// The panel whose corner is pinned.
        panel: PanelId,
        /// Boundary corner index (into the panel's outline).
        corner: u32,
    },
    /// Pin every vertex lying along one boundary edge of a panel (its two
    /// corners plus any refined midpoints on that segment).
    Edge {
        /// The panel whose edge is pinned.
        panel: PanelId,
        /// Boundary edge index (corner `edge` -> corner `edge + 1`).
        edge: u32,
    },
}

/// The welded simulation mesh produced by [`PolygonGarmentBuilder::build`].
///
/// `particles`, `constraints`, and `triangles` are the solver- and
/// render-ready sim mesh (compact-indexed `0..particles.len()`); `panels` and
/// `seams` are the [`asset`](super::asset)-level records describing how the
/// compact range decomposes back into the authored panels and how many stitches
/// each seam realized.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolygonGarmentMesh {
    /// Welded sim-mesh particles, compact-indexed `0..particles.len()`.
    pub particles: Vec<ClothParticle>,
    /// De-duplicated woven constraint graph over the welded particles.
    pub constraints: Vec<Constraint>,
    /// The counter-clockwise triangle membrane over the welded particles.
    pub triangles: Vec<[u32; 3]>,
    /// Per-panel ownership of the compact sim-vertex range.
    pub panels: Vec<Panel>,
    /// Realized seams with their actual welded stitch counts.
    pub seams: Vec<Seam>,
}

/// Accumulates polygon panels, seams, and pins into a garment, then welds them
/// into a [`PolygonGarmentMesh`].
///
/// All panels share one [`FabricMaterial`]; add panels (each with its interior
/// refinement level), seams, and pins in any order, then call
/// [`PolygonGarmentBuilder::build`].
#[derive(Clone, Debug, PartialEq)]
pub struct PolygonGarmentBuilder {
    /// Woven fabric parameters shared by every panel of the garment.
    material: FabricMaterial,
    /// Panels with their refinement level, in insertion order; the order fixes
    /// raw and compact indexing.
    panels: Vec<(PolygonPanel, u32)>,
    /// Seam specifications in insertion order.
    seams: Vec<PolygonSeamSpec>,
    /// Pin specifications in insertion order.
    pins: Vec<PolygonPinSpec>,
}

impl PolygonGarmentBuilder {
    /// Starts an empty garment cut from the given fabric.
    #[must_use]
    pub fn new(material: FabricMaterial) -> Self {
        Self {
            material,
            panels: Vec::new(),
            seams: Vec::new(),
            pins: Vec::new(),
        }
    }

    /// Adds a polygon panel refined by `subdivisions` uniform 1->4 midpoint
    /// levels and returns `self` for chaining.
    pub fn add_panel(&mut self, panel: PolygonPanel, subdivisions: u32) -> &mut Self {
        self.panels.push((panel, subdivisions));
        self
    }

    /// Adds a seam and returns `self` for chaining.
    pub fn add_seam(&mut self, seam: PolygonSeamSpec) -> &mut Self {
        self.seams.push(seam);
        self
    }

    /// Adds a pin and returns `self` for chaining.
    pub fn add_pin(&mut self, pin: PolygonPinSpec) -> &mut Self {
        self.pins.push(pin);
        self
    }

    /// Returns the index of the panel with the given id, or `None` when no such
    /// panel was added (so a stale seam/pin reference is skipped, not panicked
    /// on).
    fn panel_index(&self, id: PanelId) -> Option<usize> {
        self.panels.iter().position(|(p, _)| p.id == id)
    }

    /// Welds the accumulated panels, seams, and pins into a solver-ready mesh.
    #[must_use]
    pub fn build(&self) -> PolygonGarmentMesh {
        // 1. Refine every panel independently into its own sim mesh.
        let meshes: Vec<_> = self
            .panels
            .iter()
            .map(|(panel, level)| panel.build_refined(self.material, *level))
            .collect();

        // Prefix sum of per-panel vertex counts: bases[k] is the raw index of
        // panel k's first vertex; the final entry is the total raw count.
        let mut bases = Vec::with_capacity(self.panels.len() + 1);
        let mut acc: u32 = 0;
        bases.push(0);
        for mesh in &meshes {
            acc = acc.saturating_add(mesh.particles.len() as u32);
            bases.push(acc);
        }
        let total_raw = *bases.last().unwrap_or(&0);

        // 2. Lay out raw positions and per-raw inverse masses.
        let mut raw_positions = alloc::vec![Vec3::ZERO; total_raw as usize];
        let mut raw_inverse_mass = alloc::vec![0.0_f32; total_raw as usize];
        for (k, mesh) in meshes.iter().enumerate() {
            let base = bases[k] as usize;
            for (local, particle) in mesh.particles.iter().enumerate() {
                raw_positions[base + local] = particle.position;
                raw_inverse_mass[base + local] = particle.inverse_mass;
            }
        }

        let raw_pinned = self.raw_pins(&bases, &raw_positions, &meshes, total_raw);

        // 3. Union seam-paired boundary vertices.
        let mut uf = UnionFind::new(total_raw);
        let realized_seams = self.weld_seams(&bases, &raw_positions, &meshes, &mut uf);

        // 4. Compact and weld.
        let (raw_to_compact, compact_count, rep_prefix) = compact_indices(&uf, total_raw);
        let particles = weld_particles(
            &raw_positions,
            &raw_inverse_mass,
            &raw_pinned,
            &raw_to_compact,
            compact_count,
        );

        // 5. Remap constraints and triangles through the weld.
        let constraints = self.weld_constraints(&bases, &meshes, &raw_to_compact, &particles);
        let triangles = weld_triangles(&bases, &meshes, &raw_to_compact);
        let panels = self.compact_panels(&bases, &rep_prefix);

        PolygonGarmentMesh {
            particles,
            constraints,
            triangles,
            panels,
            seams: realized_seams,
        }
    }

    /// Resolves every [`PolygonPinSpec`] to raw vertices and returns a
    /// per-raw-vertex pinned flag.
    fn raw_pins(
        &self,
        bases: &[u32],
        raw_positions: &[Vec3],
        meshes: &[super::polygon_panel::PolygonPanelMesh],
        total_raw: u32,
    ) -> Vec<bool> {
        let mut pinned = alloc::vec![false; total_raw as usize];
        for pin in &self.pins {
            match *pin {
                PolygonPinSpec::Corner { panel, corner } => {
                    if let Some(k) = self.panel_index(panel) {
                        let count = meshes[k].particles.len() as u32;
                        if corner < count {
                            let raw = bases[k].saturating_add(corner) as usize;
                            if raw < pinned.len() {
                                pinned[raw] = true;
                            }
                        }
                    }
                }
                PolygonPinSpec::Edge { panel, edge } => {
                    if let Some(k) = self.panel_index(panel) {
                        let count = meshes[k].particles.len() as u32;
                        for raw in
                            edge_chain_raw(&self.panels[k].0, bases[k], raw_positions, count, edge)
                        {
                            if (raw as usize) < pinned.len() {
                                pinned[raw as usize] = true;
                            }
                        }
                    }
                }
            }
        }
        pinned
    }

    /// Unions each seam's paired boundary vertices in the union-find and returns
    /// the realized [`Seam`] records. Seams referencing a missing panel are
    /// skipped.
    fn weld_seams(
        &self,
        bases: &[u32],
        raw_positions: &[Vec3],
        meshes: &[super::polygon_panel::PolygonPanelMesh],
        uf: &mut UnionFind,
    ) -> Vec<Seam> {
        let mut realized = Vec::with_capacity(self.seams.len());
        for spec in &self.seams {
            let (Some(ka), Some(kb)) = (
                self.panel_index(spec.panel_a),
                self.panel_index(spec.panel_b),
            ) else {
                continue;
            };
            let count_a = meshes[ka].particles.len() as u32;
            let count_b = meshes[kb].particles.len() as u32;
            let chain_a = edge_chain_raw(
                &self.panels[ka].0,
                bases[ka],
                raw_positions,
                count_a,
                spec.edge_a,
            );
            let mut chain_b = edge_chain_raw(
                &self.panels[kb].0,
                bases[kb],
                raw_positions,
                count_b,
                spec.edge_b,
            );
            if spec.reversed {
                chain_b.reverse();
            }
            let stitches = chain_a.len().min(chain_b.len());
            for s in 0..stitches {
                uf.union(chain_a[s], chain_b[s]);
            }
            realized.push(Seam::new(spec.panel_a, spec.panel_b, stitches as u32));
        }
        realized
    }

    /// Remaps every panel's woven constraints through the weld, de-duplicating
    /// by `(min, max, kind)` and recomputing each rest length from the welded
    /// particle positions.
    fn weld_constraints(
        &self,
        bases: &[u32],
        meshes: &[super::polygon_panel::PolygonPanelMesh],
        raw_to_compact: &[u32],
        particles: &[ClothParticle],
    ) -> Vec<Constraint> {
        let mut out = Vec::new();
        let mut seen: BTreeSet<(u32, u32, u8)> = BTreeSet::new();
        for (k, mesh) in meshes.iter().enumerate() {
            let base = bases[k];
            for c in &mesh.constraints {
                let ra = base.saturating_add(c.a) as usize;
                let rb = base.saturating_add(c.b) as usize;
                let (Some(&ca), Some(&cb)) = (raw_to_compact.get(ra), raw_to_compact.get(rb))
                else {
                    continue;
                };
                push_welded_constraint(
                    &mut out,
                    &mut seen,
                    ca,
                    cb,
                    particles,
                    c.compliance,
                    c.kind,
                );
            }
        }
        out
    }

    /// Builds the per-panel [`Panel`] records mapping each panel onto its
    /// contiguous block of compact sim vertices.
    fn compact_panels(&self, bases: &[u32], rep_prefix: &[u32]) -> Vec<Panel> {
        let mut panels = Vec::with_capacity(self.panels.len());
        for (k, (panel, _)) in self.panels.iter().enumerate() {
            let start = rep_prefix[bases[k] as usize];
            let end = rep_prefix[bases[k + 1] as usize];
            panels.push(Panel::new(panel.id, start, end - start));
        }
        panels
    }
}

/// Recovers the raw vertex indices lying along one boundary edge of a panel,
/// ordered from the edge's start corner to its end corner.
///
/// The edge runs from boundary corner `edge` to corner `edge + 1` (wrapping).
/// A vertex is on the edge when it projects inside the segment and its
/// perpendicular distance is within tolerance; results are ordered by the
/// projection parameter so two panels' chains pair up along the same geometry.
fn edge_chain_raw(
    panel: &PolygonPanel,
    base: u32,
    raw_positions: &[Vec3],
    count: u32,
    edge: u32,
) -> Vec<u32> {
    let n = panel.boundary.len();
    if n < 2 {
        return Vec::new();
    }
    let e = (edge as usize) % n;
    let a = panel.position_3d(panel.boundary[e]);
    let b = panel.position_3d(panel.boundary[(e + 1) % n]);
    let d = b.sub(a);
    let len2 = d.dot(d);
    if len2 <= EPS_LEN_SQ {
        // Degenerate edge collapses to its start corner.
        return alloc::vec![base.saturating_add(e as u32)];
    }
    let tol = (ON_EDGE_REL_TOL * len2.sqrt()).max(ON_EDGE_ABS_TOL);
    let param_range = -PARAM_EPS..=(1.0 + PARAM_EPS);

    let mut chain: Vec<(f32, u32)> = Vec::new();
    for local in 0..count {
        let raw = base.saturating_add(local);
        let Some(&p) = raw_positions.get(raw as usize) else {
            continue;
        };
        let t = p.sub(a).dot(d) / len2;
        if !param_range.contains(&t) {
            continue;
        }
        let proj = a.add(d.scale(t));
        if p.distance(proj) > tol {
            continue;
        }
        chain.push((t.clamp(0.0, 1.0), raw));
    }
    chain.sort_by(|x, y| x.0.total_cmp(&y.0));
    chain.into_iter().map(|(_, raw)| raw).collect()
}

/// Remaps and de-duplicates every panel's triangles through the weld, dropping
/// any triangle that collapses (two welded-equal corners) or repeats.
fn weld_triangles(
    bases: &[u32],
    meshes: &[super::polygon_panel::PolygonPanelMesh],
    raw_to_compact: &[u32],
) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    let mut seen: BTreeSet<[u32; 3]> = BTreeSet::new();
    for (k, mesh) in meshes.iter().enumerate() {
        let base = bases[k];
        for tri in &mesh.triangles {
            let mapped: Option<[u32; 3]> = (|| {
                let mut m = [0u32; 3];
                for (slot, &v) in m.iter_mut().zip(tri.iter()) {
                    let raw = base.saturating_add(v) as usize;
                    *slot = *raw_to_compact.get(raw)?;
                }
                Some(m)
            })();
            let Some(m) = mapped else {
                continue;
            };
            if m[0] == m[1] || m[1] == m[2] || m[0] == m[2] {
                continue;
            }
            let mut key = m;
            key.sort_unstable();
            if seen.insert(key) {
                out.push(m);
            }
        }
    }
    out
}

/// Pushes one constraint between two compact particles, skipping self edges
/// (welded collapse), degenerate near-zero rest lengths, and any `(min, max,
/// kind)` pair already emitted. The rest length is measured from the welded
/// particle positions.
fn push_welded_constraint(
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

/// Stable `u8` tag for a [`ConstraintKind`], used only as part of the
/// de-duplication key so two different kinds over the same pair stay distinct.
fn kind_key(kind: ConstraintKind) -> u8 {
    match kind {
        ConstraintKind::Stretch => 0,
        ConstraintKind::Bend => 1,
        ConstraintKind::Shear => 2,
        ConstraintKind::Lra => 3,
        ConstraintKind::Tether => 4,
    }
}

/// Assigns compact sim-vertex indices from raw indices.
///
/// Returns `(raw_to_compact, compact_count, rep_prefix)` where
/// `raw_to_compact[r]` is the compact index of raw vertex `r`, `compact_count`
/// is the number of distinct welded vertices, and `rep_prefix[k]` counts the
/// representatives in `0..k` (so a panel's compact block is
/// `rep_prefix[base]..rep_prefix[next_base]`). A representative is the lowest
/// raw index of its welded set, so it is the first member seen scanning up.
fn compact_indices(uf: &UnionFind, total_raw: u32) -> (Vec<u32>, u32, Vec<u32>) {
    let total = total_raw as usize;
    let mut compact_of_root = alloc::vec![u32::MAX; total];
    let mut rep_prefix = alloc::vec![0u32; total + 1];
    let mut next: u32 = 0;
    for (r, slot) in compact_of_root.iter_mut().enumerate() {
        rep_prefix[r] = next;
        if uf.find(r as u32) == r as u32 {
            *slot = next;
            next += 1;
        }
    }
    rep_prefix[total] = next;

    let mut raw_to_compact = alloc::vec![0u32; total];
    for (r, slot) in raw_to_compact.iter_mut().enumerate() {
        *slot = compact_of_root[uf.find(r as u32) as usize];
    }
    (raw_to_compact, next, rep_prefix)
}

/// Builds the welded particle list from raw geometry.
///
/// Each welded particle's position is the mean of the raw positions welded into
/// it, and its inverse mass is the parallel combination of the raw inverse
/// masses (a single pinned raw vertex pins the weld). Velocity starts at zero.
fn weld_particles(
    raw_positions: &[Vec3],
    raw_inverse_mass: &[f32],
    raw_pinned: &[bool],
    raw_to_compact: &[u32],
    compact_count: u32,
) -> Vec<ClothParticle> {
    let count = compact_count as usize;
    let mut sums = alloc::vec![Vec3::ZERO; count];
    let mut tallies = alloc::vec![0u32; count];
    // Inverse mass accumulated in parallel; `NaN` marks "no contribution yet".
    let mut inv_mass = alloc::vec![f32::NAN; count];

    for (raw, &compact) in raw_to_compact.iter().enumerate() {
        let c = compact as usize;
        if c >= count {
            continue;
        }
        sums[c] = sums[c].add(raw_positions[raw]);
        tallies[c] += 1;
        let this = if raw_pinned[raw] {
            0.0
        } else {
            raw_inverse_mass[raw]
        };
        inv_mass[c] = combine_inverse_mass(inv_mass[c], this);
    }

    let mut particles = Vec::with_capacity(count);
    for c in 0..count {
        let n = tallies[c].max(1) as f32;
        let position = sums[c].scale(1.0 / n);
        let mass = if inv_mass[c].is_nan() {
            0.0
        } else {
            inv_mass[c]
        };
        particles.push(ClothParticle::new(position, mass));
    }
    particles
}

/// Combines two inverse masses of welded vertices in parallel.
///
/// A `NaN` accumulator means "no contribution yet" and returns `incoming`
/// unchanged. A non-positive inverse mass (pinned / infinite mass) on either
/// side wins, keeping the weld pinned. Otherwise the physical masses add
/// (`1/m = 1/m_a + 1/m_b`), i.e. the inverse masses combine as
/// `(a * b) / (a + b)`.
fn combine_inverse_mass(acc: f32, incoming: f32) -> f32 {
    if acc.is_nan() {
        return incoming;
    }
    if acc <= 0.0 || incoming <= 0.0 {
        return 0.0;
    }
    (acc * incoming) / (acc + incoming)
}

/// A disjoint-set (union-find) forest over raw vertex indices.
///
/// The representative of each set is always its lowest raw index (union always
/// re-parents the higher root onto the lower), which makes the compact-index
/// scan deterministic. `find` applies iterative path following.
#[derive(Clone, Debug, PartialEq)]
struct UnionFind {
    /// Parent link per element; a root points at itself.
    parent: Vec<u32>,
}

impl UnionFind {
    /// Creates `count` singleton sets.
    fn new(count: u32) -> Self {
        let mut parent = Vec::with_capacity(count as usize);
        for i in 0..count {
            parent.push(i);
        }
        Self { parent }
    }

    /// Returns the representative (lowest index) of `x`'s set. An out-of-range
    /// index is returned unchanged.
    fn find(&self, x: u32) -> u32 {
        let mut cur = x;
        while let Some(&p) = self.parent.get(cur as usize) {
            if p == cur {
                return cur;
            }
            cur = p;
        }
        x
    }

    /// Merges the sets containing `a` and `b`, re-parenting the higher-indexed
    /// root onto the lower so the representative stays the minimum index.
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        if let Some(slot) = self.parent.get_mut(hi as usize) {
            *slot = lo;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit-square CCW panel on the world XY plane at `origin_x`.
    fn unit_square(id: u32, origin_x: f32) -> PolygonPanel {
        PolygonPanel::new(
            PanelId(id),
            alloc::vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            Vec3::new(origin_x, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    /// Total membrane area of a welded mesh from its welded triangle positions.
    fn welded_area(mesh: &PolygonGarmentMesh) -> f32 {
        let mut area = 0.0;
        for tri in &mesh.triangles {
            let p0 = mesh.particles[tri[0] as usize].position;
            let p1 = mesh.particles[tri[1] as usize].position;
            let p2 = mesh.particles[tri[2] as usize].position;
            area += 0.5 * p1.sub(p0).cross(p2.sub(p0)).length();
        }
        area
    }

    /// Two unit squares meeting along `x = 1`: A's right edge (corner 1->2)
    /// sewn to B's left edge (corner 3->0), reversed so they run together.
    fn two_square_garment(subdivisions: u32) -> PolygonGarmentBuilder {
        let mut b = PolygonGarmentBuilder::new(FabricMaterial::default());
        b.add_panel(unit_square(0, 0.0), subdivisions);
        b.add_panel(unit_square(1, 1.0), subdivisions);
        b.add_seam(PolygonSeamSpec::new(PanelId(0), 1, PanelId(1), 3, true));
        b
    }

    #[test]
    fn seam_welds_shared_boundary_vertices() {
        let mesh = two_square_garment(0).build();
        // Each square: 4 corners. Seam pairs the 2 shared corners on x=1.
        assert_eq!(mesh.seams.len(), 1);
        assert_eq!(mesh.seams[0].stitch_count, 2);
        assert_eq!(mesh.particles.len(), 4 + 4 - 2);
    }

    #[test]
    fn seam_stitch_count_grows_with_refinement() {
        // One subdivision drops a midpoint onto every edge, so the shared edge
        // carries 3 vertices (2 corners + 1 midpoint) => 3 stitches.
        let mesh = two_square_garment(1).build();
        assert_eq!(mesh.seams[0].stitch_count, 3);
        // A single square refined once has 4 corners + 5 edge midpoints = 9
        // vertices; two of them weld away 3 shared, leaving 9 + 9 - 3 = 15.
        assert_eq!(mesh.particles.len(), 15);
    }

    #[test]
    fn welded_area_equals_sum_of_panels() {
        for level in 0..3 {
            let mesh = two_square_garment(level).build();
            assert!(
                (welded_area(&mesh) - 2.0).abs() < 1.0e-4,
                "level {level} area {}",
                welded_area(&mesh)
            );
        }
    }

    #[test]
    fn build_is_deterministic() {
        let a = two_square_garment(2).build();
        let b = two_square_garment(2).build();
        assert_eq!(a, b);
    }

    #[test]
    fn triangles_reference_valid_and_nondegenerate_particles() {
        let mesh = two_square_garment(2).build();
        let n = mesh.particles.len() as u32;
        assert!(!mesh.triangles.is_empty());
        for tri in &mesh.triangles {
            assert!(tri.iter().all(|&v| v < n));
            assert!(tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]);
        }
    }

    #[test]
    fn no_self_constraints_after_weld() {
        let mesh = two_square_garment(2).build();
        assert!(!mesh.constraints.is_empty());
        for c in &mesh.constraints {
            assert_ne!(c.a, c.b);
            assert!(c.rest_length >= MIN_REST);
        }
    }

    #[test]
    fn pin_edge_survives_weld() {
        let mut builder = two_square_garment(1);
        // Pin A's left edge (corner 3->0, the x=0 boundary).
        builder.add_pin(PolygonPinSpec::Edge {
            panel: PanelId(0),
            edge: 3,
        });
        let mesh = builder.build();
        let pinned = mesh.particles.iter().filter(|p| p.is_pinned()).count();
        // The x=0 edge of a once-refined square carries 3 vertices.
        assert_eq!(pinned, 3);
    }

    #[test]
    fn pin_corner_survives_weld() {
        let mut builder = two_square_garment(0);
        builder.add_pin(PolygonPinSpec::Corner {
            panel: PanelId(1),
            corner: 2,
        });
        let mesh = builder.build();
        assert_eq!(mesh.particles.iter().filter(|p| p.is_pinned()).count(), 1);
    }

    #[test]
    fn panels_partition_compact_range() {
        let mesh = two_square_garment(1).build();
        assert_eq!(mesh.panels.len(), 2);
        assert_eq!(mesh.panels[0].sim_vertex_start, 0);
        let mut cursor = 0;
        for panel in &mesh.panels {
            assert_eq!(panel.sim_vertex_start, cursor);
            cursor += panel.sim_vertex_count;
        }
        assert_eq!(cursor as usize, mesh.particles.len());
    }

    #[test]
    fn reversed_flag_flips_pairing() {
        // Without reversing, A's edge 1 ((1,0)->(1,1)) pairs with B's edge 3
        // ((1,1)->(1,0)) end-to-end: corner (1,0) of A welds to corner (1,1)
        // of B and vice versa. Both orientations still weld exactly 2 vertices
        // but produce distinct welded geometry, so the meshes differ.
        let mut aligned = PolygonGarmentBuilder::new(FabricMaterial::default());
        aligned.add_panel(unit_square(0, 0.0), 0);
        aligned.add_panel(unit_square(1, 1.0), 0);
        aligned.add_seam(PolygonSeamSpec::new(PanelId(0), 1, PanelId(1), 3, false));
        let crossed = aligned.build();
        let straight = two_square_garment(0).build();
        assert_eq!(crossed.seams[0].stitch_count, 2);
        assert_eq!(straight.seams[0].stitch_count, 2);
        assert_ne!(crossed.particles, straight.particles);
    }

    #[test]
    fn missing_panel_seam_is_skipped() {
        let mut b = PolygonGarmentBuilder::new(FabricMaterial::default());
        b.add_panel(unit_square(0, 0.0), 0);
        b.add_seam(PolygonSeamSpec::new(PanelId(0), 1, PanelId(99), 3, true));
        let mesh = b.build();
        assert!(mesh.seams.is_empty());
        assert_eq!(mesh.particles.len(), 4);
    }

    #[test]
    fn seam_weld_averages_coincident_positions() {
        // The two shared corners sit at identical world points, so welding
        // leaves the welded vertex exactly on that point.
        let mesh = two_square_garment(0).build();
        let on_seam = mesh
            .particles
            .iter()
            .filter(|p| (p.position.x - 1.0).abs() < 1.0e-6)
            .count();
        // x=1 line carries: A corners (1,0),(1,1) welded with B corners => 2.
        assert_eq!(on_seam, 2);
    }
}
