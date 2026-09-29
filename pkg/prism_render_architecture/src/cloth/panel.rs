//! 2D pattern panels stitched into a 3D garment sim mesh.
//!
//! This module turns an authored set of flat rectangular *pattern panels*
//! (the warp/weft grid a garment is cut from) plus a list of *seams* into a
//! single welded simulation mesh: [`ClothParticle`]s and a woven
//! [`Constraint`] graph. It mirrors the panel/seam authoring model of
//! `Marvelous` Designer / CLO and the constraint layout of production XPBD
//! cloth (UE5 `Chaos` Cloth, NVIDIA `NvCloth`), at the algorithm level only.
//!
//! A seam is a *weld*, not a spring: the two paired boundary vertices become
//! the same simulation particle (their positions are averaged and their masses
//! combined in parallel). Welding, rather than a stiff seam spring, keeps the
//! seam infinitely rigid without adding an ill-conditioned high-stiffness
//! constraint to the solver, which is how a sewn garment behaves once the
//! stitch is closed.
//!
//! The build is fully deterministic given a fixed panel/seam/pin order:
//! vertices are welded with a union-find whose representative is always the
//! lowest raw index, compact indices are assigned by a single ascending scan,
//! and constraints are de-duplicated through an ordered
//! [`BTreeSet`](alloc::collections::BTreeSet) key.
//!
//! `prism_render_architecture` is a dependency-free contracts crate, so the
//! geometry is spelled out with the crate-local [`Vec3`] math (only `sqrt` is
//! used, via [`Vec3::distance`]); panels carry an explicit orthonormal
//! (`warp_axis`, `weft_axis`) basis so this module never calls a transcendental
//! rotation.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::asset::{FabricMaterial, Panel, PanelId, Seam};
use super::{ClothParticle, Compliance, Constraint, ConstraintKind, Vec3};

/// Distance below which a generated constraint is dropped as degenerate, so a
/// zero-length weld collapse never produces a `rest_length == 0` spring.
const MIN_REST: f32 = 1.0e-6;

/// Lower bound on a node mass before taking its reciprocal, so a zero-density
/// fabric yields a finite (very large) inverse mass instead of a division by
/// zero or an infinite/`NaN` value.
const MIN_MASS: f32 = 1.0e-6;

/// One of the four boundary edges of a rectangular [`GridPanel`].
///
/// Edge vertices are enumerated in a fixed order (ascending warp index `i`
/// along `Top`/`Bottom`, ascending weft index `j` along `Left`/`Right`) so a
/// seam pairs the two edges deterministically.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PanelEdge {
    /// The `j == rows - 1` row, enumerated over `i` ascending.
    Top,
    /// The `j == 0` row, enumerated over `i` ascending.
    Bottom,
    /// The `i == 0` column, enumerated over `j` ascending.
    Left,
    /// The `i == cols - 1` column, enumerated over `j` ascending.
    Right,
}

/// A flat rectangular pattern panel sampled on a warp/weft grid.
///
/// Node `(i, j)` (with `i` the warp index in `0..cols` and `j` the weft index
/// in `0..rows`) sits at
/// `origin + i * warp_spacing * warp_axis + j * weft_spacing * weft_axis`.
/// The local (panel-relative) vertex index is `j * cols + i`, so a panel owns
/// `cols * rows` contiguous vertices before welding. The caller supplies the
/// `warp_axis`/`weft_axis` basis (expected orthonormal); this type only scales
/// and adds, so no transcendental rotation is needed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPanel {
    /// Stable identity of this panel within the garment.
    pub id: PanelId,
    /// Number of nodes along the warp (lengthwise) direction.
    pub cols: u32,
    /// Number of nodes along the weft (crosswise) direction.
    pub rows: u32,
    /// Rest spacing between adjacent warp nodes, in metres.
    pub warp_spacing: f32,
    /// Rest spacing between adjacent weft nodes, in metres.
    pub weft_spacing: f32,
    /// World-space position of node `(0, 0)`.
    pub origin: Vec3,
    /// Unit direction of increasing warp index `i`.
    pub warp_axis: Vec3,
    /// Unit direction of increasing weft index `j`.
    pub weft_axis: Vec3,
}

impl GridPanel {
    /// Builds a panel with the given grid resolution, spacings, and placement.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "A grid panel is defined by its resolution, two spacings, and a full placement frame; grouping them into an ad hoc struct would only move the same fields."
    )]
    pub const fn new(
        id: PanelId,
        cols: u32,
        rows: u32,
        warp_spacing: f32,
        weft_spacing: f32,
        origin: Vec3,
        warp_axis: Vec3,
        weft_axis: Vec3,
    ) -> Self {
        Self {
            id,
            cols,
            rows,
            warp_spacing,
            weft_spacing,
            origin,
            warp_axis,
            weft_axis,
        }
    }

    /// Number of grid nodes this panel owns before welding (`cols * rows`),
    /// computed with a saturating product so a pathological resolution clamps
    /// instead of wrapping.
    #[must_use]
    pub fn node_count(self) -> u32 {
        self.cols.saturating_mul(self.rows)
    }

    /// The panel-local vertex index of node `(i, j)`, i.e. `j * cols + i`,
    /// computed with saturating arithmetic so it never wraps.
    #[must_use]
    pub fn local_index(self, i: u32, j: u32) -> u32 {
        j.saturating_mul(self.cols).saturating_add(i)
    }

    /// World-space rest position of node `(i, j)`.
    #[must_use]
    pub fn node_position(self, i: u32, j: u32) -> Vec3 {
        let along_warp = self.warp_axis.scale(i as f32 * self.warp_spacing);
        let along_weft = self.weft_axis.scale(j as f32 * self.weft_spacing);
        self.origin.add(along_warp).add(along_weft)
    }

    /// The panel-local vertex indices along one boundary edge, in the fixed
    /// enumeration order for that edge (ascending `i` for `Top`/`Bottom`,
    /// ascending `j` for `Left`/`Right`). Returns an empty vector for a
    /// degenerate (zero-width or zero-height) panel.
    #[must_use]
    pub fn boundary_indices(self, edge: PanelEdge) -> Vec<u32> {
        let mut out = Vec::new();
        if self.cols == 0 || self.rows == 0 {
            return out;
        }
        match edge {
            PanelEdge::Bottom => {
                for i in 0..self.cols {
                    out.push(self.local_index(i, 0));
                }
            }
            PanelEdge::Top => {
                let j = self.rows - 1;
                for i in 0..self.cols {
                    out.push(self.local_index(i, j));
                }
            }
            PanelEdge::Left => {
                for j in 0..self.rows {
                    out.push(self.local_index(0, j));
                }
            }
            PanelEdge::Right => {
                let i = self.cols - 1;
                for j in 0..self.rows {
                    out.push(self.local_index(i, j));
                }
            }
        }
        out
    }
}

/// A seam that welds one edge of `panel_a` to one edge of `panel_b`.
///
/// The two edges are enumerated in their fixed orders and paired index-by-index
/// up to the shorter edge; any surplus vertices on the longer edge are left
/// unwelded (a deterministic truncation, never a panic). Set `reversed` when
/// the panels face each other so the edges run in opposite directions, which
/// reverses `panel_b`'s sequence before pairing.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SeamSpec {
    /// The first panel joined by this seam.
    pub panel_a: PanelId,
    /// The edge of `panel_a` that is sewn.
    pub edge_a: PanelEdge,
    /// The second panel joined by this seam.
    pub panel_b: PanelId,
    /// The edge of `panel_b` that is sewn.
    pub edge_b: PanelEdge,
    /// When `true`, `panel_b`'s edge sequence is reversed before pairing.
    pub reversed: bool,
}

impl SeamSpec {
    /// Builds a seam specification between two panel edges.
    #[must_use]
    pub const fn new(
        panel_a: PanelId,
        edge_a: PanelEdge,
        panel_b: PanelId,
        edge_b: PanelEdge,
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

/// A pin constraint that fixes one or more sim vertices in place (inverse mass
/// forced to zero), modelling a waistband, collar, or anim-driven attachment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PinSpec {
    /// Pin a single grid node `(i, j)` of one panel.
    Node {
        /// The panel whose node is pinned.
        panel: PanelId,
        /// Warp index of the pinned node.
        i: u32,
        /// Weft index of the pinned node.
        j: u32,
    },
    /// Pin every vertex along one boundary edge of a panel.
    Edge {
        /// The panel whose edge is pinned.
        panel: PanelId,
        /// The pinned boundary edge.
        edge: PanelEdge,
    },
}

/// The welded simulation mesh produced by [`GarmentBuilder::build`].
///
/// `particles` and `constraints` are the solver-ready sim mesh; `panels` and
/// `seams` are the [`asset`](super::asset)-level records describing how the
/// compact vertex range decomposes back into the authored panels and seams.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GarmentMesh {
    /// Welded sim-mesh particles, compact-indexed `0..particles.len()`.
    pub particles: Vec<ClothParticle>,
    /// De-duplicated woven constraint graph over the welded particles.
    pub constraints: Vec<Constraint>,
    /// Per-panel ownership of the compact sim-vertex range.
    pub panels: Vec<Panel>,
    /// Realized seams with their actual welded stitch counts.
    pub seams: Vec<Seam>,
}

/// Accumulates panels, seams, and pins into a garment, then welds them into a
/// [`GarmentMesh`].
///
/// All panels share one [`FabricMaterial`]; add panels, seams, and pins in any
/// order, then call [`GarmentBuilder::build`]. The build is a pure function of
/// the accumulated state, so calling it twice yields identical output.
#[derive(Clone, Debug, PartialEq)]
pub struct GarmentBuilder {
    /// Woven fabric parameters shared by every panel of the garment.
    material: FabricMaterial,
    /// Panels in insertion order; the order fixes raw and compact indexing.
    panels: Vec<GridPanel>,
    /// Seam specifications in insertion order.
    seams: Vec<SeamSpec>,
    /// Pin specifications in insertion order.
    pins: Vec<PinSpec>,
}

impl GarmentBuilder {
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

    /// Adds a pattern panel and returns `self` for chaining.
    pub fn add_panel(&mut self, panel: GridPanel) -> &mut Self {
        self.panels.push(panel);
        self
    }

    /// Adds a seam and returns `self` for chaining.
    pub fn add_seam(&mut self, seam: SeamSpec) -> &mut Self {
        self.seams.push(seam);
        self
    }

    /// Adds a pin and returns `self` for chaining.
    pub fn add_pin(&mut self, pin: PinSpec) -> &mut Self {
        self.pins.push(pin);
        self
    }

    /// Returns the index of the panel with the given id, or `None` when no such
    /// panel was added (so a stale seam/pin reference is skipped, not panicked
    /// on).
    fn panel_index(&self, id: PanelId) -> Option<usize> {
        self.panels.iter().position(|p| p.id == id)
    }

    /// Welds the accumulated panels, seams, and pins into a solver-ready mesh.
    ///
    /// The pipeline is: lay every panel's grid into a raw vertex list, union
    /// seam-paired boundary vertices, mark pinned vertices, assign compact
    /// indices by an ascending scan, average welded positions while combining
    /// masses in parallel, then emit the de-duplicated woven constraint graph.
    #[must_use]
    pub fn build(&self) -> GarmentMesh {
        let bases = self.panel_bases();
        let total_raw = *bases.last().unwrap_or(&0);

        let raw_positions = self.raw_positions(&bases, total_raw);
        let raw_pinned = self.raw_pins(&bases, total_raw);

        let mut uf = UnionFind::new(total_raw);
        let realized_seams = self.weld_seams(&bases, &mut uf);

        let (raw_to_compact, compact_count, rep_prefix) = compact_indices(&uf, total_raw);
        let particles = weld_particles(
            &raw_positions,
            &raw_pinned,
            &raw_to_compact,
            compact_count,
            self.node_inverse_mass(),
        );
        let panels = self.compact_panels(&bases, &rep_prefix);
        let constraints = self.weld_constraints(&bases, &raw_to_compact, &particles);

        GarmentMesh {
            particles,
            constraints,
            panels,
            seams: realized_seams,
        }
    }

    /// Prefix sum of panel node counts: `bases[k]` is the raw index of the
    /// first vertex of panel `k`, and the final entry is the total raw vertex
    /// count. Length is `panels.len() + 1`.
    fn panel_bases(&self) -> Vec<u32> {
        let mut bases = Vec::with_capacity(self.panels.len() + 1);
        let mut acc: u32 = 0;
        bases.push(0);
        for panel in &self.panels {
            acc = acc.saturating_add(panel.node_count());
            bases.push(acc);
        }
        bases
    }

    /// The uniform per-node inverse mass for this fabric: `1 / (density *
    /// warp_spacing * weft_spacing)`, clamped so a zero/degenerate area yields a
    /// finite value. Uses the first panel's spacings as the representative cell
    /// size; an empty garment falls back to unit spacing.
    fn node_inverse_mass(&self) -> f32 {
        let (ws, wf) = self
            .panels
            .first()
            .map_or((1.0, 1.0), |p| (p.warp_spacing, p.weft_spacing));
        let area = (ws * wf).max(0.0);
        let mass = (self.material.sanitized().density * area).max(MIN_MASS);
        1.0 / mass
    }

    /// Lays every panel's grid into one raw position list indexed by
    /// `base + local_index`.
    fn raw_positions(&self, bases: &[u32], total_raw: u32) -> Vec<Vec3> {
        let mut positions = alloc::vec![Vec3::ZERO; total_raw as usize];
        for (k, panel) in self.panels.iter().enumerate() {
            let base = bases[k];
            for j in 0..panel.rows {
                for i in 0..panel.cols {
                    let raw = base.saturating_add(panel.local_index(i, j)) as usize;
                    if raw < positions.len() {
                        positions[raw] = panel.node_position(i, j);
                    }
                }
            }
        }
        positions
    }

    /// Resolves every [`PinSpec`] to raw vertices and returns a per-raw-vertex
    /// pinned flag.
    fn raw_pins(&self, bases: &[u32], total_raw: u32) -> Vec<bool> {
        let mut pinned = alloc::vec![false; total_raw as usize];
        for pin in &self.pins {
            match *pin {
                PinSpec::Node { panel, i, j } => {
                    if let Some(k) = self.panel_index(panel) {
                        let p = self.panels[k];
                        if i < p.cols && j < p.rows {
                            let raw = bases[k].saturating_add(p.local_index(i, j)) as usize;
                            if raw < pinned.len() {
                                pinned[raw] = true;
                            }
                        }
                    }
                }
                PinSpec::Edge { panel, edge } => {
                    if let Some(k) = self.panel_index(panel) {
                        let base = bases[k];
                        for local in self.panels[k].boundary_indices(edge) {
                            let raw = base.saturating_add(local) as usize;
                            if raw < pinned.len() {
                                pinned[raw] = true;
                            }
                        }
                    }
                }
            }
        }
        pinned
    }

    /// Unions each seam's paired boundary vertices in the union-find and
    /// returns the realized [`Seam`] records (with the actual welded stitch
    /// count). Seams referencing a missing panel are skipped.
    fn weld_seams(&self, bases: &[u32], uf: &mut UnionFind) -> Vec<Seam> {
        let mut realized = Vec::with_capacity(self.seams.len());
        for spec in &self.seams {
            let (Some(ka), Some(kb)) = (
                self.panel_index(spec.panel_a),
                self.panel_index(spec.panel_b),
            ) else {
                continue;
            };
            let edge_a = self.panels[ka].boundary_indices(spec.edge_a);
            let mut edge_b = self.panels[kb].boundary_indices(spec.edge_b);
            if spec.reversed {
                edge_b.reverse();
            }
            let stitches = edge_a.len().min(edge_b.len());
            for s in 0..stitches {
                let raw_a = bases[ka].saturating_add(edge_a[s]);
                let raw_b = bases[kb].saturating_add(edge_b[s]);
                uf.union(raw_a, raw_b);
            }
            realized.push(Seam::new(spec.panel_a, spec.panel_b, stitches as u32));
        }
        realized
    }

    /// Builds the per-panel [`Panel`] records mapping each panel onto its
    /// contiguous block of compact sim vertices.
    ///
    /// A panel owns exactly the welded vertices whose union-find representative
    /// first appears inside that panel's raw range; because compact indices are
    /// assigned by a single ascending scan, those indices form a contiguous
    /// block `[rep_prefix[base], rep_prefix[next_base])`.
    fn compact_panels(&self, bases: &[u32], rep_prefix: &[u32]) -> Vec<Panel> {
        let mut panels = Vec::with_capacity(self.panels.len());
        for (k, panel) in self.panels.iter().enumerate() {
            let start = rep_prefix[bases[k] as usize];
            let end = rep_prefix[bases[k + 1] as usize];
            panels.push(Panel::new(panel.id, start, end.saturating_sub(start)));
        }
        panels
    }

    /// Emits the woven constraint graph over the welded (compact) particles.
    ///
    /// For every panel it generates warp/weft structural stretch edges,
    /// quad-diagonal shear constraints, and second-neighbour cross-bending
    /// constraints, mapping each grid neighbour through the weld before
    /// de-duplicating by `(min, max, kind)` so a shared seam edge or a collapsed
    /// weld never emits a duplicate or a self constraint.
    fn weld_constraints(
        &self,
        bases: &[u32],
        raw_to_compact: &[u32],
        particles: &[ClothParticle],
    ) -> Vec<Constraint> {
        let mat = self.material.sanitized();
        let warp_c = mat.warp_compliance();
        let weft_c = mat.weft_compliance();
        let bend_c = mat.bend_compliance();
        // Shear resists in-plane distortion; the diagonal is not a warp or weft
        // yarn, so it uses the softer bending compliance as its rest stiffness.
        let shear_c = mat.bend_compliance();

        let mut out = Vec::new();
        let mut seen: BTreeSet<(u32, u32, u8)> = BTreeSet::new();

        for (k, panel) in self.panels.iter().enumerate() {
            let base = bases[k];
            let compact = |i: u32, j: u32| -> u32 {
                let raw = base.saturating_add(panel.local_index(i, j)) as usize;
                raw_to_compact[raw]
            };

            for j in 0..panel.rows {
                for i in 0..panel.cols {
                    // Warp structural edge (i, j)-(i+1, j).
                    if i + 1 < panel.cols {
                        push_constraint(
                            &mut out,
                            &mut seen,
                            compact(i, j),
                            compact(i + 1, j),
                            particles,
                            warp_c,
                            ConstraintKind::Stretch,
                        );
                    }
                    // Weft structural edge (i, j)-(i, j+1).
                    if j + 1 < panel.rows {
                        push_constraint(
                            &mut out,
                            &mut seen,
                            compact(i, j),
                            compact(i, j + 1),
                            particles,
                            weft_c,
                            ConstraintKind::Stretch,
                        );
                    }
                    // Quad shear diagonals of cell (i, j).
                    if i + 1 < panel.cols && j + 1 < panel.rows {
                        push_constraint(
                            &mut out,
                            &mut seen,
                            compact(i, j),
                            compact(i + 1, j + 1),
                            particles,
                            shear_c,
                            ConstraintKind::Shear,
                        );
                        push_constraint(
                            &mut out,
                            &mut seen,
                            compact(i + 1, j),
                            compact(i, j + 1),
                            particles,
                            shear_c,
                            ConstraintKind::Shear,
                        );
                    }
                    // Warp bending: second neighbour (i, j)-(i+2, j).
                    if i + 2 < panel.cols {
                        push_constraint(
                            &mut out,
                            &mut seen,
                            compact(i, j),
                            compact(i + 2, j),
                            particles,
                            bend_c,
                            ConstraintKind::Bend,
                        );
                    }
                    // Weft bending: second neighbour (i, j)-(i, j+2).
                    if j + 2 < panel.rows {
                        push_constraint(
                            &mut out,
                            &mut seen,
                            compact(i, j),
                            compact(i, j + 2),
                            particles,
                            bend_c,
                            ConstraintKind::Bend,
                        );
                    }
                }
            }
        }
        out
    }
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

/// Pushes one constraint between two compact particles, skipping self edges
/// (welded collapse), degenerate near-zero rest lengths, and any `(min, max,
/// kind)` pair already emitted. The rest length is measured from the welded
/// particle positions.
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

/// Assigns compact sim-vertex indices from raw indices.
///
/// Returns `(raw_to_compact, compact_count, rep_prefix)` where
/// `raw_to_compact[r]` is the compact index of raw vertex `r`, `compact_count`
/// is the number of distinct welded vertices, and `rep_prefix[k]` is the number
/// of representatives in `0..k` (so a panel's compact block is
/// `rep_prefix[base]..rep_prefix[next_base]`). A representative is the lowest
/// raw index of its welded set, so it is always the first member seen in the
/// ascending scan.
fn compact_indices(uf: &UnionFind, total_raw: u32) -> (Vec<u32>, u32, Vec<u32>) {
    let total = total_raw as usize;
    let mut compact_of_root = alloc::vec![u32::MAX; total];
    let mut rep_prefix = alloc::vec![0u32; total + 1];
    let mut next: u32 = 0;
    for r in 0..total {
        rep_prefix[r] = next;
        if uf.find(r as u32) == r as u32 {
            compact_of_root[r] = next;
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
/// Each compact particle's position is the mean of the raw positions welded
/// into it, and its inverse mass is the parallel combination of the raw inverse
/// masses (a single pinned raw vertex pins the weld). Velocity starts at zero.
fn weld_particles(
    raw_positions: &[Vec3],
    raw_pinned: &[bool],
    raw_to_compact: &[u32],
    compact_count: u32,
    node_inverse_mass: f32,
) -> Vec<ClothParticle> {
    let count = compact_count as usize;
    let mut sums = alloc::vec![Vec3::ZERO; count];
    let mut tallies = alloc::vec![0u32; count];
    // Inverse mass accumulated in parallel; `0.0` marks "pinned so far".
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
            node_inverse_mass
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
/// side wins, keeping the weld pinned. Otherwise the masses add
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
/// scan deterministic: a set's representative is the first of its members
/// reached by an ascending scan. `find` applies iterative path halving.
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

    /// Returns the representative (lowest index) of `x`'s set, halving the path
    /// along the way. An out-of-range index is returned unchanged.
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

    /// A unit-spaced panel on the XY plane at the origin.
    fn xy_panel(id: u32, cols: u32, rows: u32) -> GridPanel {
        GridPanel::new(
            PanelId(id),
            cols,
            rows,
            1.0,
            1.0,
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn node_count_and_local_index_match_grid() {
        let p = xy_panel(0, 4, 3);
        assert_eq!(p.node_count(), 12);
        assert_eq!(p.local_index(0, 0), 0);
        assert_eq!(p.local_index(3, 0), 3);
        assert_eq!(p.local_index(0, 1), 4);
        assert_eq!(p.local_index(3, 2), 11);
    }

    #[test]
    fn node_position_is_linear_in_basis() {
        let p = xy_panel(0, 3, 3);
        assert_eq!(p.node_position(0, 0), Vec3::ZERO);
        assert_eq!(p.node_position(2, 0), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(p.node_position(0, 2), Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(p.node_position(1, 1), Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn boundary_edges_enumerate_expected_indices() {
        let p = xy_panel(0, 3, 3);
        assert_eq!(p.boundary_indices(PanelEdge::Bottom), alloc::vec![0, 1, 2]);
        assert_eq!(p.boundary_indices(PanelEdge::Top), alloc::vec![6, 7, 8]);
        assert_eq!(p.boundary_indices(PanelEdge::Left), alloc::vec![0, 3, 6]);
        assert_eq!(p.boundary_indices(PanelEdge::Right), alloc::vec![2, 5, 8]);
    }

    #[test]
    fn degenerate_panel_has_no_boundary() {
        let p = xy_panel(0, 0, 5);
        assert!(p.boundary_indices(PanelEdge::Top).is_empty());
        assert_eq!(p.node_count(), 0);
    }

    #[test]
    fn single_panel_counts_particles_and_constraints() {
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 3, 3));
        let mesh = b.build();
        // 3x3 grid: 9 unwelded particles.
        assert_eq!(mesh.particles.len(), 9);
        assert_eq!(mesh.panels.len(), 1);
        assert_eq!(mesh.panels[0].sim_vertex_start, 0);
        assert_eq!(mesh.panels[0].sim_vertex_count, 9);
        // Every constraint references two valid, distinct particles.
        for c in &mesh.constraints {
            assert_ne!(c.a, c.b);
            assert!((c.a as usize) < mesh.particles.len());
            assert!((c.b as usize) < mesh.particles.len());
            assert!(c.rest_length >= MIN_REST);
        }
        assert!(!mesh.constraints.is_empty());
    }

    #[test]
    fn stretch_rest_length_equals_spacing() {
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 2, 2));
        let mesh = b.build();
        for c in &mesh.constraints {
            if c.kind == ConstraintKind::Stretch {
                assert!((c.rest_length - 1.0).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn seam_weld_shares_vertices() {
        // Two 3x3 panels; weld panel 0's Right edge to panel 1's Left edge.
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 3, 3));
        b.add_panel(xy_panel(1, 3, 3));
        b.add_seam(SeamSpec::new(
            PanelId(0),
            PanelEdge::Right,
            PanelId(1),
            PanelEdge::Left,
            false,
        ));
        let mesh = b.build();
        // 18 raw - 3 welded pairs = 15 particles.
        assert_eq!(mesh.particles.len(), 15);
        // Panel 0 owns its 9; panel 1 owns the 6 it introduces.
        assert_eq!(mesh.panels[0].sim_vertex_start, 0);
        assert_eq!(mesh.panels[0].sim_vertex_count, 9);
        assert_eq!(mesh.panels[1].sim_vertex_start, 9);
        assert_eq!(mesh.panels[1].sim_vertex_count, 6);
        assert_eq!(mesh.seams.len(), 1);
        assert_eq!(mesh.seams[0].stitch_count, 3);
    }

    #[test]
    fn build_is_deterministic() {
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 4, 3));
        b.add_panel(xy_panel(1, 4, 3));
        b.add_seam(SeamSpec::new(
            PanelId(0),
            PanelEdge::Top,
            PanelId(1),
            PanelEdge::Bottom,
            false,
        ));
        let first = b.build();
        let second = b.build();
        assert_eq!(first, second);
    }

    #[test]
    fn pinned_edge_has_zero_inverse_mass() {
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 3, 3));
        b.add_pin(PinSpec::Edge {
            panel: PanelId(0),
            edge: PanelEdge::Top,
        });
        let mesh = b.build();
        // Top edge = compact vertices 6,7,8 in a lone 3x3 panel.
        for idx in [6usize, 7, 8] {
            assert_eq!(mesh.particles[idx].inverse_mass, 0.0);
        }
        for idx in [0usize, 1, 2] {
            assert!(mesh.particles[idx].inverse_mass > 0.0);
        }
    }

    #[test]
    fn pin_survives_weld() {
        // Pin panel 1's Left edge, then weld it to panel 0's Right edge: the
        // shared particles must remain pinned.
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 2, 3));
        b.add_panel(xy_panel(1, 2, 3));
        b.add_pin(PinSpec::Edge {
            panel: PanelId(1),
            edge: PanelEdge::Left,
        });
        b.add_seam(SeamSpec::new(
            PanelId(0),
            PanelEdge::Right,
            PanelId(1),
            PanelEdge::Left,
            false,
        ));
        let mesh = b.build();
        let pinned = mesh
            .particles
            .iter()
            .filter(|p| p.inverse_mass == 0.0)
            .count();
        assert_eq!(pinned, 3);
    }

    #[test]
    fn mismatched_seam_lengths_truncate() {
        // Panel 0 Right edge has 4 vertices, panel 1 Left edge has 3.
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 2, 4));
        b.add_panel(xy_panel(1, 2, 3));
        b.add_seam(SeamSpec::new(
            PanelId(0),
            PanelEdge::Right,
            PanelId(1),
            PanelEdge::Left,
            false,
        ));
        let mesh = b.build();
        // min(4, 3) = 3 stitches welded.
        assert_eq!(mesh.seams[0].stitch_count, 3);
        // 8 + 6 raw - 3 welds = 11 particles.
        assert_eq!(mesh.particles.len(), 11);
    }

    #[test]
    fn missing_panel_seam_is_skipped() {
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 2, 2));
        b.add_seam(SeamSpec::new(
            PanelId(0),
            PanelEdge::Right,
            PanelId(99),
            PanelEdge::Left,
            false,
        ));
        // Must not panic; the dangling seam contributes nothing.
        let mesh = b.build();
        assert!(mesh.seams.is_empty());
        assert_eq!(mesh.particles.len(), 4);
    }

    #[test]
    fn no_self_constraints_after_weld() {
        let mut b = GarmentBuilder::new(FabricMaterial::default());
        b.add_panel(xy_panel(0, 3, 3));
        b.add_panel(xy_panel(1, 3, 3));
        b.add_seam(SeamSpec::new(
            PanelId(0),
            PanelEdge::Right,
            PanelId(1),
            PanelEdge::Left,
            false,
        ));
        let mesh = b.build();
        let mut keys = BTreeSet::new();
        for c in &mesh.constraints {
            assert_ne!(c.a, c.b);
            // No duplicate (min,max,kind) triples.
            assert!(keys.insert((c.a.min(c.b), c.a.max(c.b), kind_key(c.kind))));
        }
    }
}
