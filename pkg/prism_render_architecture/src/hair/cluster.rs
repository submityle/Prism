//! `Nanite`-style strand clustering with per-cluster cull verdicts and
//! continuous strand decimation (design doc §8.5 item6).
//!
//! A dense groom is millions of render strands; drawing or simulating every one
//! regardless of where the camera looks is wasteful. The `Nanite`-geometry
//! answer is to break the primitive stream into small, spatially-local
//! *clusters* and then cull and `LOD` whole clusters at once on the `GPU`: a
//! cluster that leaves the view frustum, that a nearer occluder fully hides
//! (hierarchical-Z / `HiZ`), or that faces away from the eye contributes no
//! pixels and is dropped before any per-strand work. This mirrors how `UE5`
//! Groom buckets its strands and is the hair-side analogue of the virtual
//! geometry cluster cull in [`crate::virtual_geometry::cull`].
//!
//! This module is the deterministic, panic-free *contract* layer for that. It
//! does three jobs, all pure (array in, array out, `golden`-comparable, no
//! device state):
//!
//! * **Clustering** — [`cluster_strands`] assigns each strand to a spatial grid
//!   cell and groups strands sharing a cell into a [`StrandCluster`] (its
//!   axis-aligned bounds, mean growth tangent, and member strand indices).
//!   Clusters are capped in size so one dense cell fans out into several
//!   bounded clusters, exactly like fixed-size `Nanite` clusters. Cell keys are
//!   quantised integer lattice coordinates, so the grouping is fully
//!   reproducible frame to frame.
//! * **Cull verdicts** — [`cluster_cull_verdict`] rejects a cluster by frustum
//!   (bounds vs six inward planes), then by back-facing (mean tangent vs the
//!   view-to-eye direction), then by occlusion (nearest cluster depth vs a
//!   conservative `HiZ` occluder sample). Frustum rejection takes precedence,
//!   matching the virtual-geometry convention.
//! * **Continuous decimation** — [`strand_keep_ratio`] maps a cluster's
//!   projected pixel footprint to a keep ratio with a *linear* ramp between a
//!   full-detail and a cull footprint, so the §4 continuous `LOD` ladder thins
//!   strands without a pop as a groom recedes.
//!
//! [`bin_clusters`] fans a whole cluster list into per-verdict buckets in one
//! deterministic pass (preserving input order within each bucket), reusing the
//! `push`/`total`/`is_empty` bucket shape of
//! [`crate::virtual_geometry::bins`]. The real `GPU` cull/decimate dispatch is
//! owned by the render graph; this layer only produces the `CPU` contract the
//! dispatch is validated against. It performs **no** transcendental math — only
//! multiplies, `sqrt` for vector length, and integer lattice quantisation — so
//! it needs no `libm` determinism shim.

use alloc::vec::Vec;

/// Shared epsilon for zero-length / degenerate-span guards and test comparisons.
const EPS: f32 = 1.0e-6;

/// Default back-facing cosine margin: a cluster is only culled when its mean
/// tangent points more than this far away from the eye. Conservative on
/// purpose — hair is near-cylindrical and a tangent is a weak facing cue, so we
/// would rather keep a cluster than drop one that could still shade pixels.
pub const DEFAULT_BACKFACE_BIAS: f32 = 0.5;

/// Default maximum strands per cluster, matching the small fixed cluster size
/// that keeps `Nanite`-style `GPU` cull batches uniform.
pub const DEFAULT_MAX_STRANDS_PER_CLUSTER: usize = 128;

// ---------------------------------------------------------------------------
// Small hand-written vector helpers (no external math crate in this crate).
// ---------------------------------------------------------------------------

#[must_use]
fn sanitize_coord(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[must_use]
fn sanitize_vec3(v: [f32; 3]) -> [f32; 3] {
    [
        sanitize_coord(v[0]),
        sanitize_coord(v[1]),
        sanitize_coord(v[2]),
    ]
}

#[must_use]
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        0.0
    }
}

#[must_use]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[must_use]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[must_use]
fn length_sq3(v: [f32; 3]) -> f32 {
    dot3(v, v)
}

/// Returns `v` scaled to unit length, or `[0, 0, 0]` when `v` is degenerate
/// (shorter than [`EPS`]), so normalisation never divides by zero or yields a
/// `NaN`.
#[must_use]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = length_sq3(v);
    if len_sq <= EPS * EPS {
        return [0.0, 0.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

// ---------------------------------------------------------------------------
// Axis-aligned bounds.
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box around a cluster's strands.
///
/// Built empty (`min = +inf`, `max = -inf`) and grown point by point. After at
/// least one [`Aabb::expand`] it satisfies `min <= max` on every axis; an
/// un-grown box reports [`Aabb::is_valid`] `false` and yields a zero centre /
/// zero half-extents rather than `NaN`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl Default for Aabb {
    fn default() -> Self {
        Self::empty()
    }
}

impl Aabb {
    /// An inverted empty box that any [`Aabb::expand`] immediately corrects.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    /// Grows the box to enclose a (sanitised) point.
    pub fn expand(&mut self, point: [f32; 3]) {
        let p = sanitize_vec3(point);
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(p[axis]);
            self.max[axis] = self.max[axis].max(p[axis]);
        }
    }

    /// Whether the box has enclosed at least one point (`min <= max` all axes).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }

    /// Box centre, or `[0, 0, 0]` for an empty box.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        if !self.is_valid() {
            return [0.0, 0.0, 0.0];
        }
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Non-negative half-extents, or `[0, 0, 0]` for an empty box.
    #[must_use]
    pub fn half_extents(&self) -> [f32; 3] {
        if !self.is_valid() {
            return [0.0, 0.0, 0.0];
        }
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }
}

// ---------------------------------------------------------------------------
// Clustering input + spatial grid.
// ---------------------------------------------------------------------------

/// One strand sampled for clustering: its root and tip (which bracket the
/// strand's spatial extent) plus a growth `tangent` used for the cluster-level
/// back-facing test. All three vectors are sanitised (non-finite components ->
/// `0`) before use, so malformed grooms never panic or poison the bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrandSample {
    /// Strand root (scalp) position.
    pub root: [f32; 3],
    /// Strand tip position.
    pub tip: [f32; 3],
    /// Strand growth / orientation tangent (need not be unit length).
    pub tangent: [f32; 3],
}

impl StrandSample {
    /// A strand sample from explicit root, tip, and tangent.
    #[must_use]
    pub const fn new(root: [f32; 3], tip: [f32; 3], tangent: [f32; 3]) -> Self {
        Self { root, tip, tangent }
    }

    /// This sample with every non-finite vector component repaired to `0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            root: sanitize_vec3(self.root),
            tip: sanitize_vec3(self.tip),
            tangent: sanitize_vec3(self.tangent),
        }
    }
}

/// Uniform spatial grid used to assign strands to clusters by locality.
///
/// A strand is placed in the cell `floor(root / cell_size)` on each axis;
/// strands landing in the same integer cell cluster together. A non-finite or
/// non-positive `cell_size` is repaired to `1.0` by [`ClusterGrid::sanitized`]
/// (a zero cell would collapse every strand into one cell).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterGrid {
    /// Edge length of one cubic grid cell in world units.
    pub cell_size: f32,
}

impl Default for ClusterGrid {
    fn default() -> Self {
        Self { cell_size: 1.0 }
    }
}

impl ClusterGrid {
    /// A grid with an explicit cell size.
    #[must_use]
    pub const fn new(cell_size: f32) -> Self {
        Self { cell_size }
    }

    /// This grid with a non-finite / non-positive `cell_size` repaired to `1.0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let cell_size = if self.cell_size.is_finite() && self.cell_size > 0.0 {
            self.cell_size
        } else {
            1.0
        };
        Self { cell_size }
    }

    /// Integer lattice cell containing a (sanitised) world position.
    #[must_use]
    pub fn cell_of(&self, position: [f32; 3]) -> [i32; 3] {
        let cs = self.sanitized().cell_size;
        let p = sanitize_vec3(position);
        [
            (p[0] / cs).floor() as i32,
            (p[1] / cs).floor() as i32,
            (p[2] / cs).floor() as i32,
        ]
    }
}

// ---------------------------------------------------------------------------
// Strand cluster + clustering pass.
// ---------------------------------------------------------------------------

/// A spatially-local group of strands with the data a cull/`LOD` dispatch needs.
///
/// `strand_indices` holds the member strands' indices into the original sample
/// slice (the deterministic, locality-grouped equivalent of a packed strand
/// range). `mean_tangent` is unit length (or `[0, 0, 0]` when the members'
/// tangents cancel) and feeds the back-facing test.
#[derive(Clone, Debug, PartialEq)]
pub struct StrandCluster {
    /// Sequential cluster id, equal to the cluster's index in the output list.
    pub id: u32,
    /// Bounds enclosing every member strand's root and tip.
    pub bounds: Aabb,
    /// Unit mean growth tangent over the member strands.
    pub mean_tangent: [f32; 3],
    /// Member strand indices into the input sample slice, in input order.
    pub strand_indices: Vec<u32>,
}

impl StrandCluster {
    /// Number of strands in this cluster.
    #[must_use]
    pub fn strand_count(&self) -> usize {
        self.strand_indices.len()
    }

    /// Whether this cluster has no member strands.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strand_indices.is_empty()
    }
}

/// Groups strands into spatially-local clusters of at most `max_strands`.
///
/// Each strand is assigned to its [`ClusterGrid`] cell; strands in the same cell
/// join the same cluster until it reaches `max_strands`, at which point a fresh
/// cluster opens for that cell (so one dense cell yields several bounded
/// clusters). Cluster ids and member order are fully determined by the input
/// order and the grid, so the result is reproducible. A `max_strands` of `0` is
/// treated as `1` to guarantee forward progress. An empty input returns an
/// empty list.
#[must_use]
pub fn cluster_strands(
    samples: &[StrandSample],
    grid: ClusterGrid,
    max_strands: usize,
) -> Vec<StrandCluster> {
    let grid = grid.sanitized();
    let cap = max_strands.max(1);
    let mut clusters: Vec<StrandCluster> = Vec::new();
    // Maps a live cell to the index of its currently-open (not-yet-full) cluster.
    let mut open: Vec<([i32; 3], usize)> = Vec::new();

    for (strand_index, raw) in samples.iter().enumerate() {
        let sample = raw.sanitized();
        let cell = grid.cell_of(sample.root);

        // Find an existing open cluster for this cell.
        let mut slot: Option<usize> = None;
        for (entry_index, (entry_cell, _)) in open.iter().enumerate() {
            if *entry_cell == cell {
                slot = Some(entry_index);
                break;
            }
        }

        let cluster_index = match slot {
            Some(entry_index) => {
                let current = open[entry_index].1;
                if clusters[current].strand_indices.len() < cap {
                    current
                } else {
                    // Open cluster is full: start a new one for the same cell.
                    let fresh = clusters.len();
                    clusters.push(empty_cluster(fresh));
                    open[entry_index].1 = fresh;
                    fresh
                }
            }
            None => {
                let fresh = clusters.len();
                clusters.push(empty_cluster(fresh));
                open.push((cell, fresh));
                fresh
            }
        };

        let cluster = &mut clusters[cluster_index];
        cluster.strand_indices.push(strand_index as u32);
        cluster.bounds.expand(sample.root);
        cluster.bounds.expand(sample.tip);
        // Accumulate the raw tangent sum; normalised in the finalise pass.
        cluster.mean_tangent[0] += sample.tangent[0];
        cluster.mean_tangent[1] += sample.tangent[1];
        cluster.mean_tangent[2] += sample.tangent[2];
    }

    for cluster in &mut clusters {
        cluster.mean_tangent = normalize3(cluster.mean_tangent);
    }
    clusters
}

#[must_use]
fn empty_cluster(id: usize) -> StrandCluster {
    StrandCluster {
        id: id as u32,
        bounds: Aabb::empty(),
        mean_tangent: [0.0, 0.0, 0.0],
        strand_indices: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Frustum + occlusion primitives (self-contained, trig/transcendental-free).
// ---------------------------------------------------------------------------

/// An inward-facing frustum plane: the half-space `dot(normal, p) + distance >=
/// 0` is the interior. `normal` is expected unit length (planes arrive
/// normalised from the render layer) so the `AABB` projected-radius test is
/// exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// Unit inward normal.
    pub normal: [f32; 3],
    /// Signed distance of the plane from the origin along `normal`.
    pub distance: f32,
}

impl Plane {
    /// A plane from a (unit) inward normal and signed origin distance.
    #[must_use]
    pub const fn new(normal: [f32; 3], distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Signed distance from `point` to the plane; positive is interior.
    #[must_use]
    pub fn signed_distance(&self, point: [f32; 3]) -> f32 {
        dot3(self.normal, point) + self.distance
    }

    /// The box's support radius along this plane's axis (projection of the
    /// half-extents onto `|normal|`).
    #[must_use]
    fn projected_extent(&self, half_extents: [f32; 3]) -> f32 {
        self.normal[0].abs() * half_extents[0]
            + self.normal[1].abs() * half_extents[1]
            + self.normal[2].abs() * half_extents[2]
    }
}

/// Six inward-facing frustum planes (plane order is not significant).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    /// The six bounding planes.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Wraps six already-normalised inward-facing planes.
    #[must_use]
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// Whether an [`Aabb`] is at least partially inside the frustum, using the
    /// exact projected-radius test. An empty (un-grown) box is treated as
    /// outside, since it bounds nothing.
    #[must_use]
    pub fn intersects_aabb(&self, bounds: &Aabb) -> bool {
        if !bounds.is_valid() {
            return false;
        }
        let center = bounds.center();
        let half = bounds.half_extents();
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(center) >= -plane.projected_extent(half))
    }
}

/// Conservative occlusion probe for a cluster's screen footprint.
///
/// Depths are view-space linear distances where a smaller value is nearer. A
/// cluster is occluded when its nearest point is strictly farther than the
/// nearest occluder covering its footprint (e.g. a hierarchical-Z / `HiZ`
/// sample).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionProbe {
    /// Depth of the cluster's closest point (smaller is nearer).
    pub closest_depth: f32,
    /// Conservative nearest occluder depth over the cluster footprint.
    pub occluder_depth: f32,
}

impl OcclusionProbe {
    /// A probe from an explicit cluster depth and occluder depth.
    #[must_use]
    pub const fn new(closest_depth: f32, occluder_depth: f32) -> Self {
        Self {
            closest_depth,
            occluder_depth,
        }
    }

    /// Whether the cluster is fully behind the nearest occluder.
    #[must_use]
    pub fn is_occluded(&self) -> bool {
        self.closest_depth > self.occluder_depth
    }
}

/// The viewing state needed for the per-cluster facing test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterView {
    /// Eye (camera) position in the same space as the cluster bounds.
    pub eye: [f32; 3],
}

impl ClusterView {
    /// A view from an explicit eye position.
    #[must_use]
    pub const fn new(eye: [f32; 3]) -> Self {
        Self { eye }
    }
}

// ---------------------------------------------------------------------------
// Cull verdict.
// ---------------------------------------------------------------------------

/// Why a cluster received its [`ClusterCullVerdict`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CullReason {
    /// Passes every test; submit it.
    #[default]
    Visible,
    /// Rejected: fully outside the view frustum.
    Frustum,
    /// Rejected: the cluster's mean tangent faces away from the eye.
    Backface,
    /// Rejected: fully hidden behind a nearer occluder.
    Occlusion,
}

/// Outcome of culling one cluster.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusterCullVerdict {
    /// Whether the cluster should be drawn.
    pub visible: bool,
    /// The deciding reason (first failing test, or [`CullReason::Visible`]).
    pub reason: CullReason,
}

impl ClusterCullVerdict {
    /// A visible verdict.
    #[must_use]
    pub const fn visible() -> Self {
        Self {
            visible: true,
            reason: CullReason::Visible,
        }
    }

    /// A culled verdict carrying the deciding reason.
    #[must_use]
    pub const fn culled(reason: CullReason) -> Self {
        Self {
            visible: false,
            reason,
        }
    }
}

/// Whether a cluster is back-facing for a given to-eye direction.
///
/// `mean_tangent` is the cluster's unit mean growth tangent and `to_eye` is the
/// (not-necessarily-unit) direction from the cluster centre toward the eye;
/// both are normalised defensively. The cluster is back-facing only when the
/// tangent points away from the eye by more than the `bias` cosine margin, i.e.
/// `dot(tangent, to_eye) <= -bias`. A degenerate tangent or to-eye direction
/// yields a dot of `0`, which never trips a positive `bias`, so degenerate
/// input is never culled.
#[must_use]
pub fn is_cluster_backfacing(mean_tangent: [f32; 3], to_eye: [f32; 3], bias: f32) -> bool {
    let bias = bias.clamp(0.0, 1.0);
    let facing = dot3(normalize3(mean_tangent), normalize3(to_eye));
    facing <= -bias
}

/// Culls one cluster against the frustum, facing, and an optional occlusion
/// probe.
///
/// Tests run in the order frustum, back-facing, occlusion, and the first
/// failing test decides the verdict (frustum rejection takes precedence, as in
/// [`crate::virtual_geometry::cull`]). Passing `None` for `occlusion` skips the
/// occlusion phase (e.g. the first depth-prepass wave before an `HiZ` pyramid
/// exists).
#[must_use]
pub fn cluster_cull_verdict(
    cluster: &StrandCluster,
    frustum: &Frustum,
    view: ClusterView,
    occlusion: Option<OcclusionProbe>,
    backface_bias: f32,
) -> ClusterCullVerdict {
    if !frustum.intersects_aabb(&cluster.bounds) {
        return ClusterCullVerdict::culled(CullReason::Frustum);
    }
    let to_eye = sub3(view.eye, cluster.bounds.center());
    if is_cluster_backfacing(cluster.mean_tangent, to_eye, backface_bias) {
        return ClusterCullVerdict::culled(CullReason::Backface);
    }
    if let Some(probe) = occlusion
        && probe.is_occluded()
    {
        return ClusterCullVerdict::culled(CullReason::Occlusion);
    }
    ClusterCullVerdict::visible()
}

// ---------------------------------------------------------------------------
// Continuous strand decimation (anti-pop LOD).
// ---------------------------------------------------------------------------

/// Screen-footprint thresholds that drive the continuous strand keep ratio.
///
/// At a projected footprint `>= full_px` the cluster keeps every strand
/// (ratio `1`); at `<= cull_px` it keeps only `min_ratio` of them; between the
/// two the ratio ramps linearly, so a receding groom thins out without a pop.
/// Invariant after [`DecimationThresholds::sanitized`]: `full_px >= cull_px >=
/// 0`, all finite, and `min_ratio` in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecimationThresholds {
    /// At or above this footprint every strand is kept.
    pub full_px: f32,
    /// At or below this footprint only `min_ratio` of strands are kept.
    pub cull_px: f32,
    /// Floor keep ratio at and below `cull_px`.
    pub min_ratio: f32,
}

impl Default for DecimationThresholds {
    fn default() -> Self {
        Self {
            full_px: 64.0,
            cull_px: 4.0,
            min_ratio: 0.05,
        }
    }
}

impl DecimationThresholds {
    /// Thresholds from explicit footprints and floor ratio.
    #[must_use]
    pub const fn new(full_px: f32, cull_px: f32, min_ratio: f32) -> Self {
        Self {
            full_px,
            cull_px,
            min_ratio,
        }
    }

    /// Repairs non-finite / out-of-order inputs: both footprints become
    /// finite and non-negative with `full_px >= cull_px`, and `min_ratio` is
    /// clamped into `[0, 1]`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let full = sanitize_nonneg(self.full_px);
        let cull = sanitize_nonneg(self.cull_px).min(full);
        let min_ratio = if self.min_ratio.is_finite() {
            self.min_ratio.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            full_px: full,
            cull_px: cull,
            min_ratio,
        }
    }
}

/// Continuous strand keep ratio in `[min_ratio, 1]` for a cluster's projected
/// pixel footprint.
///
/// Monotone non-decreasing in `cluster_px`: a larger footprint keeps at least
/// as many strands. A degenerate band (`full_px == cull_px`) becomes a hard
/// step at that footprint (still panic-free). Non-finite / negative footprints
/// sanitise to `0` (the smallest, most-decimated footprint).
#[must_use]
pub fn strand_keep_ratio(cluster_px: f32, thresholds: DecimationThresholds) -> f32 {
    let t = thresholds.sanitized();
    let px = if cluster_px.is_finite() {
        cluster_px.max(0.0)
    } else {
        0.0
    };
    if px >= t.full_px {
        return 1.0;
    }
    if px <= t.cull_px {
        return t.min_ratio;
    }
    let span = t.full_px - t.cull_px;
    if span <= EPS {
        return t.min_ratio;
    }
    let blend = (px - t.cull_px) / span;
    t.min_ratio + (1.0 - t.min_ratio) * blend
}

/// Number of strands kept for `total` members at a given keep ratio.
///
/// The ratio is clamped into `[0, 1]`, scaled, rounded to the nearest strand,
/// and clamped to never exceed `total`, so the result is always a valid member
/// count and never panics on a non-finite ratio.
#[must_use]
pub fn kept_strand_count(total: u32, ratio: f32) -> u32 {
    let r = if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let kept = (total as f32 * r).round();
    let kept = kept.max(0.0) as u32;
    kept.min(total)
}

// ---------------------------------------------------------------------------
// Per-verdict binning.
// ---------------------------------------------------------------------------

/// A visible cluster paired with its continuous decimation result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibleCluster {
    /// Cluster id (its index in the clustering output).
    pub id: u32,
    /// Continuous keep ratio from the cluster's screen footprint.
    pub keep_ratio: f32,
    /// Strands kept this frame (`keep_ratio` applied to the member count).
    pub kept_strands: u32,
}

/// A cluster list partitioned by cull verdict.
///
/// The render graph consumes the `visible` bucket (each entry already carries
/// its decimation result) and may use the culled id buckets for statistics or
/// a second occlusion pass. Within every bucket, order matches the input
/// cluster order, which keeps dispatch deterministic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterBins {
    /// Clusters that passed every test, with their decimation result.
    pub visible: Vec<VisibleCluster>,
    /// Ids of clusters rejected by the frustum.
    pub frustum_culled: Vec<u32>,
    /// Ids of clusters rejected as back-facing.
    pub backface_culled: Vec<u32>,
    /// Ids of clusters rejected by occlusion.
    pub occlusion_culled: Vec<u32>,
}

impl ClusterBins {
    /// Total number of clusters across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.visible.len()
            + self.frustum_culled.len()
            + self.backface_culled.len()
            + self.occlusion_culled.len()
    }

    /// Returns `true` when no cluster landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.visible.is_empty()
            && self.frustum_culled.is_empty()
            && self.backface_culled.is_empty()
            && self.occlusion_culled.is_empty()
    }

    /// Routes one culled id into the bucket named by `reason`. A
    /// [`CullReason::Visible`] reason is a no-op (visible clusters carry a
    /// decimation payload and are pushed via the visible bucket instead).
    fn push_culled(&mut self, reason: CullReason, id: u32) {
        match reason {
            CullReason::Frustum => self.frustum_culled.push(id),
            CullReason::Backface => self.backface_culled.push(id),
            CullReason::Occlusion => self.occlusion_culled.push(id),
            CullReason::Visible => {}
        }
    }
}

/// Fans a cluster list into per-verdict buckets in one deterministic pass.
///
/// For cluster `i`, the occlusion probe is `probes.get(i)` (a missing or `None`
/// entry skips the occlusion phase for that cluster) and the projected
/// footprint is `coverage_px.get(i)` (a missing entry is treated as `0`, the
/// most-decimated footprint). Parallel slices shorter than `clusters` therefore
/// degrade gracefully rather than panicking, mirroring the out-of-range skip in
/// [`crate::virtual_geometry::bins`]. Visible clusters carry their
/// [`strand_keep_ratio`] / [`kept_strand_count`] result.
#[must_use]
pub fn bin_clusters(
    clusters: &[StrandCluster],
    frustum: &Frustum,
    view: ClusterView,
    probes: &[Option<OcclusionProbe>],
    coverage_px: &[f32],
    backface_bias: f32,
    decimation: DecimationThresholds,
) -> ClusterBins {
    let mut bins = ClusterBins::default();
    for (index, cluster) in clusters.iter().enumerate() {
        let probe = probes.get(index).copied().flatten();
        let verdict = cluster_cull_verdict(cluster, frustum, view, probe, backface_bias);
        if verdict.visible {
            let px = coverage_px.get(index).copied().unwrap_or(0.0);
            let keep_ratio = strand_keep_ratio(px, decimation);
            let kept_strands = kept_strand_count(cluster.strand_count() as u32, keep_ratio);
            bins.visible.push(VisibleCluster {
                id: cluster.id,
                keep_ratio,
                kept_strands,
            });
        } else {
            bins.push_culled(verdict.reason, cluster.id);
        }
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-5
    }

    /// Axis-aligned box frustum: |x| <= 10, |y| <= 10, 0 <= z <= 100.
    fn box_frustum() -> Frustum {
        Frustum::from_planes([
            Plane::new([1.0, 0.0, 0.0], 10.0),
            Plane::new([-1.0, 0.0, 0.0], 10.0),
            Plane::new([0.0, 1.0, 0.0], 10.0),
            Plane::new([0.0, -1.0, 0.0], 10.0),
            Plane::new([0.0, 0.0, 1.0], 0.0),
            Plane::new([0.0, 0.0, -1.0], 100.0),
        ])
    }

    fn sample(root: [f32; 3], tip: [f32; 3], tangent: [f32; 3]) -> StrandSample {
        StrandSample::new(root, tip, tangent)
    }

    #[test]
    fn clusters_group_strands_sharing_a_cell() {
        // Two strands in cell (0,0,0), one in a far cell.
        let samples = [
            sample([0.1, 0.1, 0.1], [0.1, 1.0, 0.1], [0.0, 1.0, 0.0]),
            sample([0.4, 0.2, 0.3], [0.4, 1.2, 0.3], [0.0, 1.0, 0.0]),
            sample([5.5, 5.5, 5.5], [5.5, 6.5, 5.5], [0.0, 1.0, 0.0]),
        ];
        let clusters = cluster_strands(&samples, ClusterGrid::new(1.0), 128);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].id, 0);
        assert_eq!(clusters[0].strand_indices, vec![0, 1]);
        assert_eq!(clusters[1].id, 1);
        assert_eq!(clusters[1].strand_indices, vec![2]);
    }

    #[test]
    fn cluster_bounds_enclose_root_and_tip() {
        let samples = [sample([0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0])];
        let clusters = cluster_strands(&samples, ClusterGrid::new(1.0), 128);
        let b = &clusters[0].bounds;
        assert!(b.is_valid());
        assert!(close(b.min[1], 0.0));
        assert!(close(b.max[1], 2.0));
        let center = b.center();
        assert!(close(center[1], 1.0));
    }

    #[test]
    fn mean_tangent_is_unit_length() {
        let samples = [
            sample([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 3.0, 0.0]),
            sample([0.1, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 5.0, 0.0]),
        ];
        let clusters = cluster_strands(&samples, ClusterGrid::new(1.0), 128);
        let t = clusters[0].mean_tangent;
        assert!(close(length_sq3(t), 1.0));
        assert!(close(t[1], 1.0));
    }

    #[test]
    fn clusters_split_at_max_strands() {
        // Four strands all in one cell, cap of 2 -> two clusters of two.
        let samples = [
            sample([0.1, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 1.0, 0.0]),
            sample([0.2, 0.0, 0.0], [0.2, 1.0, 0.0], [0.0, 1.0, 0.0]),
            sample([0.3, 0.0, 0.0], [0.3, 1.0, 0.0], [0.0, 1.0, 0.0]),
            sample([0.4, 0.0, 0.0], [0.4, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let clusters = cluster_strands(&samples, ClusterGrid::new(1.0), 2);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].strand_indices, vec![0, 1]);
        assert_eq!(clusters[1].strand_indices, vec![2, 3]);
    }

    #[test]
    fn zero_max_strands_is_treated_as_one() {
        let samples = [
            sample([0.1, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 1.0, 0.0]),
            sample([0.2, 0.0, 0.0], [0.2, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let clusters = cluster_strands(&samples, ClusterGrid::new(1.0), 0);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].strand_count(), 1);
    }

    #[test]
    fn empty_input_produces_no_clusters() {
        let clusters = cluster_strands(&[], ClusterGrid::default(), 128);
        assert!(clusters.is_empty());
    }

    #[test]
    fn non_finite_samples_are_sanitized() {
        let samples = [sample(
            [f32::NAN, 0.0, f32::INFINITY],
            [0.0, f32::NEG_INFINITY, 0.0],
            [f32::NAN, 1.0, 0.0],
        )];
        let clusters = cluster_strands(&samples, ClusterGrid::new(1.0), 128);
        assert_eq!(clusters.len(), 1);
        let b = &clusters[0].bounds;
        assert!(b.is_valid());
        for axis in 0..3 {
            assert!(b.min[axis].is_finite());
            assert!(b.max[axis].is_finite());
        }
        assert!(clusters[0].mean_tangent[0].is_finite());
    }

    #[test]
    fn grid_sanitizes_bad_cell_size() {
        assert!(close(ClusterGrid::new(-2.0).sanitized().cell_size, 1.0));
        assert!(close(ClusterGrid::new(f32::NAN).sanitized().cell_size, 1.0));
        assert!(close(ClusterGrid::new(2.0).sanitized().cell_size, 2.0));
    }

    #[test]
    fn plane_signed_distance_sign_matches_interior() {
        let p = Plane::new([1.0, 0.0, 0.0], 10.0);
        assert!(p.signed_distance([0.0, 0.0, 0.0]) >= 0.0);
        assert!(p.signed_distance([-20.0, 0.0, 0.0]) < 0.0);
    }

    #[test]
    fn frustum_keeps_inside_and_rejects_outside() {
        let f = box_frustum();
        let mut inside = Aabb::empty();
        inside.expand([0.0, 0.0, 49.0]);
        inside.expand([1.0, 1.0, 51.0]);
        assert!(f.intersects_aabb(&inside));

        let mut outside = Aabb::empty();
        outside.expand([99.0, 0.0, 49.0]);
        outside.expand([101.0, 1.0, 51.0]);
        assert!(!f.intersects_aabb(&outside));

        // An empty box bounds nothing and is treated as outside.
        assert!(!f.intersects_aabb(&Aabb::empty()));
    }

    #[test]
    fn occlusion_probe_detects_hidden_cluster() {
        assert!(OcclusionProbe::new(60.0, 50.0).is_occluded());
        assert!(!OcclusionProbe::new(40.0, 50.0).is_occluded());
    }

    fn cluster_at(center: [f32; 3], tangent: [f32; 3]) -> StrandCluster {
        let samples = [sample(
            [center[0], center[1] - 0.5, center[2]],
            [center[0], center[1] + 0.5, center[2]],
            tangent,
        )];
        cluster_strands(&samples, ClusterGrid::new(1000.0), 128)
            .into_iter()
            .next()
            .expect("one cluster")
    }

    #[test]
    fn verdict_frustum_culls_outside_cluster() {
        let f = box_frustum();
        let cluster = cluster_at([100.0, 0.0, 50.0], [0.0, 1.0, 0.0]);
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let v = cluster_cull_verdict(&cluster, &f, view, None, DEFAULT_BACKFACE_BIAS);
        assert!(!v.visible);
        assert_eq!(v.reason, CullReason::Frustum);
    }

    #[test]
    fn verdict_visible_inside_cluster() {
        let f = box_frustum();
        // Tangent sideways so the facing test never trips.
        let cluster = cluster_at([0.0, 0.0, 50.0], [1.0, 0.0, 0.0]);
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let v = cluster_cull_verdict(&cluster, &f, view, None, DEFAULT_BACKFACE_BIAS);
        assert!(v.visible);
        assert_eq!(v.reason, CullReason::Visible);
    }

    #[test]
    fn verdict_backface_culls_tangent_facing_away() {
        let f = box_frustum();
        // Eye at -z; a cluster whose tangent points toward +z faces away.
        let cluster = cluster_at([0.0, 0.0, 50.0], [0.0, 0.0, 1.0]);
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let v = cluster_cull_verdict(&cluster, &f, view, None, DEFAULT_BACKFACE_BIAS);
        assert!(!v.visible);
        assert_eq!(v.reason, CullReason::Backface);
    }

    #[test]
    fn verdict_occlusion_culls_hidden_cluster() {
        let f = box_frustum();
        let cluster = cluster_at([0.0, 0.0, 50.0], [1.0, 0.0, 0.0]);
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let probe = Some(OcclusionProbe::new(60.0, 40.0));
        let v = cluster_cull_verdict(&cluster, &f, view, probe, DEFAULT_BACKFACE_BIAS);
        assert!(!v.visible);
        assert_eq!(v.reason, CullReason::Occlusion);
    }

    #[test]
    fn verdict_frustum_takes_precedence_over_occlusion() {
        let f = box_frustum();
        let cluster = cluster_at([100.0, 0.0, 50.0], [1.0, 0.0, 0.0]);
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        // Even with an occluding probe, frustum rejection reports first.
        let probe = Some(OcclusionProbe::new(60.0, 40.0));
        let v = cluster_cull_verdict(&cluster, &f, view, probe, DEFAULT_BACKFACE_BIAS);
        assert_eq!(v.reason, CullReason::Frustum);
    }

    #[test]
    fn backface_never_culls_degenerate_tangent() {
        // Zero tangent -> facing 0, never below -bias for positive bias.
        assert!(!is_cluster_backfacing(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            DEFAULT_BACKFACE_BIAS
        ));
    }

    #[test]
    fn keep_ratio_is_full_at_or_above_full_px() {
        let t = DecimationThresholds::new(64.0, 4.0, 0.05);
        assert!(close(strand_keep_ratio(64.0, t), 1.0));
        assert!(close(strand_keep_ratio(200.0, t), 1.0));
    }

    #[test]
    fn keep_ratio_is_floor_at_or_below_cull_px() {
        let t = DecimationThresholds::new(64.0, 4.0, 0.05);
        assert!(close(strand_keep_ratio(4.0, t), 0.05));
        assert!(close(strand_keep_ratio(0.0, t), 0.05));
    }

    #[test]
    fn keep_ratio_ramps_monotonically() {
        let t = DecimationThresholds::new(64.0, 4.0, 0.0);
        let lo = strand_keep_ratio(16.0, t);
        let mid = strand_keep_ratio(34.0, t);
        let hi = strand_keep_ratio(52.0, t);
        assert!(lo < mid);
        assert!(mid < hi);
        // Midpoint of a 0..1 floor ramp over [4,64] at px=34 is 0.5.
        assert!(close(mid, 0.5));
    }

    #[test]
    fn keep_ratio_sanitizes_bad_footprint() {
        let t = DecimationThresholds::new(64.0, 4.0, 0.05);
        // Non-finite / negative px -> treated as 0 -> floor ratio.
        assert!(close(strand_keep_ratio(f32::NAN, t), 0.05));
        assert!(close(strand_keep_ratio(-5.0, t), 0.05));
    }

    #[test]
    fn keep_ratio_degenerate_band_is_hard_step() {
        let t = DecimationThresholds::new(10.0, 10.0, 0.2);
        assert!(close(strand_keep_ratio(10.0, t), 1.0));
        assert!(close(strand_keep_ratio(9.5, t), 0.2));
    }

    #[test]
    fn kept_strand_count_rounds_and_clamps() {
        assert_eq!(kept_strand_count(100, 1.0), 100);
        assert_eq!(kept_strand_count(100, 0.0), 0);
        assert_eq!(kept_strand_count(100, 0.5), 50);
        assert_eq!(kept_strand_count(10, 0.25), 3); // 2.5 -> 3 (round half up)
                                                    // Non-finite / out-of-range ratio is clamped.
        assert_eq!(kept_strand_count(100, f32::NAN), 0);
        assert_eq!(kept_strand_count(100, 2.0), 100);
    }

    #[test]
    fn bin_routes_and_preserves_order() {
        let f = box_frustum();
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let clusters = vec![
            cluster_with_id(cluster_at([0.0, 0.0, 50.0], [1.0, 0.0, 0.0]), 0), // visible
            cluster_with_id(cluster_at([100.0, 0.0, 50.0], [1.0, 0.0, 0.0]), 1), // frustum
            cluster_with_id(cluster_at([0.0, 0.0, 50.0], [0.0, 0.0, 1.0]), 2), // backface
        ];
        let probes = [None, None, None];
        let coverage = [64.0, 64.0, 64.0];
        let bins = bin_clusters(
            &clusters,
            &f,
            view,
            &probes,
            &coverage,
            DEFAULT_BACKFACE_BIAS,
            DecimationThresholds::default(),
        );
        assert_eq!(bins.total(), 3);
        assert_eq!(bins.visible.len(), 1);
        assert_eq!(bins.visible[0].id, 0);
        assert!(close(bins.visible[0].keep_ratio, 1.0));
        assert_eq!(bins.frustum_culled, vec![1]);
        assert_eq!(bins.backface_culled, vec![2]);
        assert!(bins.occlusion_culled.is_empty());
    }

    #[test]
    fn bin_tolerates_short_parallel_slices() {
        let f = box_frustum();
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let clusters = vec![
            cluster_with_id(cluster_at([0.0, 0.0, 50.0], [1.0, 0.0, 0.0]), 0),
            cluster_with_id(cluster_at([0.0, 0.0, 50.0], [1.0, 0.0, 0.0]), 1),
        ];
        // Empty probe/coverage slices: no occlusion phase, px defaults to 0.
        let bins = bin_clusters(
            &clusters,
            &f,
            view,
            &[],
            &[],
            DEFAULT_BACKFACE_BIAS,
            DecimationThresholds::default(),
        );
        assert_eq!(bins.visible.len(), 2);
        // px=0 -> floor keep ratio from the default thresholds.
        assert!(close(bins.visible[0].keep_ratio, 0.05));
    }

    #[test]
    fn bin_empty_input_is_empty() {
        let f = box_frustum();
        let view = ClusterView::new([0.0, 0.0, -10.0]);
        let bins = bin_clusters(
            &[],
            &f,
            view,
            &[],
            &[],
            DEFAULT_BACKFACE_BIAS,
            DecimationThresholds::default(),
        );
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    fn cluster_with_id(mut cluster: StrandCluster, id: u32) -> StrandCluster {
        cluster.id = id;
        cluster
    }
}
