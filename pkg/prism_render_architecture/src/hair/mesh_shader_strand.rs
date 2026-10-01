//! `mesh-shader` / `amplification`-shader strand subdivision contract
//! (design doc §8.5 item13).
//!
//! Interpolating guide strands into render strands and tessellating each render
//! strand's spline is embarrassingly parallel, yet a classic pipeline pays for
//! it by expanding every strand on the `CPU`, uploading the fat vertex stream,
//! and reading cull results back. Moving the expansion into a `mesh-shader`
//! (optionally fronted by an `amplification` shader that fans out workgroups)
//! keeps the interpolation and subdivision on the `GPU`, so no inflated vertex
//! buffer ever crosses the bus and no `CPU`<->`GPU` round trip stalls the frame.
//! This is the strand-side analogue of how `UE5` `Nanite` emits `meshlet`
//! workgroups for triangle geometry.
//!
//! This module is the deterministic, panic-free *contract* layer for that
//! expansion. It owns **no** device state and issues **no** dispatch: the real
//! `mesh-shader` / `amplification` dispatch is driven by the render graph in
//! `prism_render_scene`. All this layer does is describe, as plain integer
//! layout, how a run of render strands is partitioned into bounded
//! `meshlet`-style workgroups, so the on-`GPU` expansion can be validated value
//! for value against a `CPU` `golden`. Every function is pure (counts in, layout
//! out), order-preserving, and free of transcendental or floating-point math —
//! it is integer budgeting only, so it needs no `libm` determinism shim.
//!
//! * **Budgeting** — [`MeshStrandParams`] holds the per-workgroup vertex and
//!   primitive budgets plus the segment and strand caps a `mesh-shader` can
//!   emit, each clamped into a legal range by [`MeshStrandParams::sanitized`].
//! * **Packing** — [`pack_strand_meshlets`] slices a contiguous render-strand
//!   range into [`StrandMeshlet`] groups that each respect the budget, fanning a
//!   dense run into several bounded groups exactly like fixed-size `Nanite`
//!   clusters.
//! * **Binning** — [`bin_strand_meshlets`] and [`bin_selected_meshlets`] fan a
//!   group list into full / partial buckets in one deterministic pass,
//!   preserving input order and skipping out-of-range selections rather than
//!   panicking, reusing the `push` / `total` / `is_empty` bucket shape of
//!   [`crate::virtual_geometry::bins`].
//! * **Dispatch sizing** — [`AmplificationDispatch`] derives the workgroup
//!   dispatch count from a `meshlet` total and a per-group fan-out with plain
//!   ceiling integer division.

use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Hard limits + defaults.
// ---------------------------------------------------------------------------

/// Hard ceiling on vertices one `mesh-shader` workgroup may emit. Mirrors the
/// typical hardware `mesh-shader` output cap; [`MeshStrandParams::sanitized`]
/// clamps the configured budget to this.
pub const MAX_VERTICES_PER_GROUP_LIMIT: u32 = 256;

/// Hard ceiling on primitives (triangles) one `mesh-shader` workgroup may emit.
pub const MAX_PRIMITIVES_PER_GROUP_LIMIT: u32 = 256;

/// Hard ceiling on segments per strand. A render strand is a short polyline, so
/// this bounds the per-strand vertex and primitive expansion.
pub const MAX_SEGMENTS_PER_STRAND_LIMIT: u32 = 64;

/// Hard ceiling on how many strands one `meshlet`-style workgroup may own,
/// independent of the vertex / primitive budgets.
pub const MAX_STRANDS_PER_MESHLET_LIMIT: u32 = 256;

/// Clamps a per-workgroup budget field into `[1, limit]` so a sanitized budget
/// is always legal and always allows forward progress.
#[must_use]
fn clamp_budget(value: u32, limit: u32) -> u32 {
    value.max(1).min(limit)
}

/// Clamps a segment count into `[1, MAX_SEGMENTS_PER_STRAND_LIMIT]`.
#[must_use]
fn sanitize_segments(segments_per_strand: u32) -> u32 {
    clamp_budget(segments_per_strand, MAX_SEGMENTS_PER_STRAND_LIMIT)
}

/// Vertices a single strand of `segments` segments expands to: one vertex per
/// cross-section, i.e. `segments + 1`.
#[must_use]
fn vertices_per_strand(segments: u32) -> u32 {
    segments.saturating_add(1)
}

/// Primitives (triangles) a single strand of `segments` segments expands to:
/// two triangles per segment, i.e. `segments * 2`.
#[must_use]
fn primitives_per_strand(segments: u32) -> u32 {
    segments.saturating_mul(2)
}

// ---------------------------------------------------------------------------
// Per-workgroup budget parameters.
// ---------------------------------------------------------------------------

/// Per-`mesh-shader`-workgroup output budget for strand expansion.
///
/// A `mesh-shader` workgroup can only emit so many vertices and primitives, so
/// the number of strands one workgroup owns is bounded by whichever of the
/// vertex budget, the primitive budget, or the explicit strand cap is tightest.
/// `segments_per_strand` is the authoring default used when a caller does not
/// pass an explicit per-groom segment count; packing always honours the segment
/// count handed to [`pack_strand_meshlets`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshStrandParams {
    /// Maximum vertices one workgroup may emit.
    pub max_vertices_per_group: u32,
    /// Maximum primitives (triangles) one workgroup may emit.
    pub max_primitives_per_group: u32,
    /// Authoring default segments per strand.
    pub segments_per_strand: u32,
    /// Upper bound on strands per workgroup, independent of the budgets.
    pub strands_per_meshlet: u32,
}

impl Default for MeshStrandParams {
    fn default() -> Self {
        Self {
            max_vertices_per_group: 128,
            max_primitives_per_group: 256,
            segments_per_strand: 8,
            strands_per_meshlet: 32,
        }
    }
}

impl MeshStrandParams {
    /// Explicit budget parameters.
    #[must_use]
    pub const fn new(
        max_vertices_per_group: u32,
        max_primitives_per_group: u32,
        segments_per_strand: u32,
        strands_per_meshlet: u32,
    ) -> Self {
        Self {
            max_vertices_per_group,
            max_primitives_per_group,
            segments_per_strand,
            strands_per_meshlet,
        }
    }

    /// These parameters with every field clamped into its legal range: budgets
    /// to `[1, hard-limit]`, so a sanitized budget is never zero and never
    /// exceeds what a workgroup can physically emit.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            max_vertices_per_group: clamp_budget(
                self.max_vertices_per_group,
                MAX_VERTICES_PER_GROUP_LIMIT,
            ),
            max_primitives_per_group: clamp_budget(
                self.max_primitives_per_group,
                MAX_PRIMITIVES_PER_GROUP_LIMIT,
            ),
            segments_per_strand: sanitize_segments(self.segments_per_strand),
            strands_per_meshlet: clamp_budget(
                self.strands_per_meshlet,
                MAX_STRANDS_PER_MESHLET_LIMIT,
            ),
        }
    }

    /// Largest strand count a single workgroup may own for a groom whose strands
    /// have `segments_per_strand` segments.
    ///
    /// This is the minimum of the vertex-budget capacity, the primitive-budget
    /// capacity, and the explicit strand cap, but never less than `1`: even a
    /// degenerate budget too small for one strand still packs one strand per
    /// group so packing always makes forward progress.
    #[must_use]
    pub fn max_strands_per_meshlet(self, segments_per_strand: u32) -> u32 {
        let params = self.sanitized();
        let segments = sanitize_segments(segments_per_strand);
        let by_vertices = params.max_vertices_per_group / vertices_per_strand(segments);
        let by_primitives = params.max_primitives_per_group / primitives_per_strand(segments);
        by_vertices
            .min(by_primitives)
            .min(params.strands_per_meshlet)
            .max(1)
    }
}

// ---------------------------------------------------------------------------
// Strand meshlet description.
// ---------------------------------------------------------------------------

/// Describes one `amplification` / `mesh-shader` workgroup's strand group.
///
/// A group owns `strand_count` contiguous render strands starting at
/// `first_render_strand`, each expanded with `segments_per_strand` segments. The
/// derived vertex and primitive counts are what the workgroup emits; packing
/// guarantees they stay within the active [`MeshStrandParams`] budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrandMeshlet {
    /// Index of this group's first render strand in the source run.
    pub first_render_strand: u32,
    /// Number of render strands this group owns.
    pub strand_count: u32,
    /// Segments each owned strand is subdivided into.
    pub segments_per_strand: u32,
}

impl StrandMeshlet {
    /// Explicit strand group.
    #[must_use]
    pub const fn new(
        first_render_strand: u32,
        strand_count: u32,
        segments_per_strand: u32,
    ) -> Self {
        Self {
            first_render_strand,
            strand_count,
            segments_per_strand,
        }
    }

    /// Index one past this group's last render strand.
    #[must_use]
    pub fn end_render_strand(self) -> u32 {
        self.first_render_strand.saturating_add(self.strand_count)
    }

    /// Vertices this group emits: `strand_count * (segments_per_strand + 1)`.
    /// Saturating so a hand-built oversized group reports a capped count instead
    /// of overflowing.
    #[must_use]
    pub fn vertex_count(self) -> u32 {
        self.strand_count
            .saturating_mul(vertices_per_strand(self.segments_per_strand))
    }

    /// Primitives this group emits: `strand_count * segments_per_strand * 2`.
    #[must_use]
    pub fn primitive_count(self) -> u32 {
        self.strand_count
            .saturating_mul(primitives_per_strand(self.segments_per_strand))
    }

    /// Whether this group's emitted vertices and primitives both fit the
    /// sanitized budget.
    #[must_use]
    pub fn fits_budget(self, params: MeshStrandParams) -> bool {
        let params = params.sanitized();
        self.vertex_count() <= params.max_vertices_per_group
            && self.primitive_count() <= params.max_primitives_per_group
    }

    /// Whether this group is packed full for the budget: its strand count equals
    /// the per-group maximum for its segment count.
    #[must_use]
    pub fn is_full(self, params: MeshStrandParams) -> bool {
        self.strand_count == params.max_strands_per_meshlet(self.segments_per_strand)
    }

    /// This group's full / partial fill class under `params`.
    #[must_use]
    pub fn fill(self, params: MeshStrandParams) -> MeshletFill {
        if self.is_full(params) {
            MeshletFill::Full
        } else {
            MeshletFill::Partial
        }
    }
}

// ---------------------------------------------------------------------------
// Packing a render-strand run into meshlets.
// ---------------------------------------------------------------------------

/// Partitions a contiguous run of `render_strand_count` render strands into
/// budget-bounded [`StrandMeshlet`] groups.
///
/// Each group owns at most [`MeshStrandParams::max_strands_per_meshlet`] strands
/// for the (sanitized) `segments_per_strand`, so a dense run fans out into
/// several bounded groups, with only the final group left partial. Groups are
/// emitted in strand order, so the output is fully deterministic. A
/// `render_strand_count` of `0` returns an empty list; the per-group maximum is
/// always at least `1`, so packing can never stall or loop forever.
#[must_use]
pub fn pack_strand_meshlets(
    render_strand_count: u32,
    segments_per_strand: u32,
    params: MeshStrandParams,
) -> Vec<StrandMeshlet> {
    let mut meshlets: Vec<StrandMeshlet> = Vec::new();
    if render_strand_count == 0 {
        return meshlets;
    }
    let segments = sanitize_segments(segments_per_strand);
    let per_group = params.max_strands_per_meshlet(segments);
    let mut start = 0u32;
    while start < render_strand_count {
        let remaining = render_strand_count - start;
        let count = per_group.min(remaining);
        meshlets.push(StrandMeshlet::new(start, count, segments));
        start += count;
    }
    meshlets
}

// ---------------------------------------------------------------------------
// Fill classification + binning.
// ---------------------------------------------------------------------------

/// Whether a [`StrandMeshlet`] is packed full to its budget or left partial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshletFill {
    /// Strand count equals the per-group maximum for the budget.
    Full,
    /// Strand count is below the per-group maximum (the trailing remainder).
    Partial,
}

/// A `meshlet` list partitioned by fill class.
///
/// The render graph can submit the `full` bucket as one uniform dispatch and
/// handle the `partial` remainder groups separately. Within each bucket, order
/// matches the input order, keeping dispatch deterministic.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeshletBins {
    /// Groups packed full to the per-group budget.
    pub full: Vec<StrandMeshlet>,
    /// Groups below the per-group budget (trailing remainders).
    pub partial: Vec<StrandMeshlet>,
}

impl MeshletBins {
    /// Routes one group into its fill bucket under `params`.
    pub fn push(&mut self, meshlet: StrandMeshlet, params: MeshStrandParams) {
        match meshlet.fill(params) {
            MeshletFill::Full => self.full.push(meshlet),
            MeshletFill::Partial => self.partial.push(meshlet),
        }
    }

    /// Immutable view of the bucket for a given fill class.
    #[must_use]
    pub fn bucket(&self, fill: MeshletFill) -> &[StrandMeshlet] {
        match fill {
            MeshletFill::Full => &self.full,
            MeshletFill::Partial => &self.partial,
        }
    }

    /// Total number of groups across both buckets.
    #[must_use]
    pub fn total(&self) -> usize {
        self.full.len() + self.partial.len()
    }

    /// Returns `true` when no group landed in either bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.full.is_empty() && self.partial.is_empty()
    }
}

/// Fans a `meshlet` list into full / partial buckets in one deterministic pass,
/// preserving input order within each bucket.
#[must_use]
pub fn bin_strand_meshlets(meshlets: &[StrandMeshlet], params: MeshStrandParams) -> MeshletBins {
    let mut bins = MeshletBins::default();
    for &meshlet in meshlets {
        bins.push(meshlet, params);
    }
    bins
}

/// Fans a selected subset of a `meshlet` list into full / partial buckets.
///
/// Each entry of `selection` indexes `meshlets`; an out-of-range index is
/// skipped rather than panicking, mirroring the out-of-range skip in
/// [`crate::virtual_geometry::bins`]. Selected groups are binned in `selection`
/// order, so the result stays deterministic.
#[must_use]
pub fn bin_selected_meshlets(
    meshlets: &[StrandMeshlet],
    selection: &[u32],
    params: MeshStrandParams,
) -> MeshletBins {
    let mut bins = MeshletBins::default();
    for &index in selection {
        if let Some(&meshlet) = meshlets.get(index as usize) {
            bins.push(meshlet, params);
        }
    }
    bins
}

// ---------------------------------------------------------------------------
// Amplification dispatch sizing.
// ---------------------------------------------------------------------------

/// Workgroups needed to cover `meshlet_count` groups at `meshlets_per_group` per
/// `amplification` workgroup: `ceil(meshlet_count / meshlets_per_group)`.
///
/// A `meshlets_per_group` of `0` is treated as `1` for forward progress, and a
/// `meshlet_count` of `0` dispatches nothing. The ceiling division is computed
/// in 64-bit so the `+ (per - 1)` rounding step can never overflow.
#[must_use]
pub fn dispatch_group_count(meshlet_count: u32, meshlets_per_group: u32) -> u32 {
    let per_group = meshlets_per_group.max(1);
    if meshlet_count == 0 {
        return 0;
    }
    let count = meshlet_count as u64;
    let per = per_group as u64;
    ((count + per - 1) / per) as u32
}

/// `amplification`-shader dispatch dimensions for a packed `meshlet` list.
///
/// The render graph binds `meshlet_count` groups and launches `dispatch_groups`
/// `amplification` workgroups, each responsible for up to `meshlets_per_group`
/// groups. All three fields are plain integers so a `golden` can compare them
/// bit for bit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AmplificationDispatch {
    /// Total `meshlet` groups to dispatch.
    pub meshlet_count: u32,
    /// `meshlet` groups each `amplification` workgroup covers (at least `1`).
    pub meshlets_per_group: u32,
    /// `amplification` workgroups to launch (ceiling division).
    pub dispatch_groups: u32,
}

impl AmplificationDispatch {
    /// Derives the dispatch dimensions for `meshlet_count` groups fanned out at
    /// `meshlets_per_group` per `amplification` workgroup.
    #[must_use]
    pub fn for_meshlets(meshlet_count: u32, meshlets_per_group: u32) -> Self {
        let per_group = meshlets_per_group.max(1);
        Self {
            meshlet_count,
            meshlets_per_group: per_group,
            dispatch_groups: dispatch_group_count(meshlet_count, per_group),
        }
    }

    /// `meshlet` slots the launched workgroups cover, i.e.
    /// `dispatch_groups * meshlets_per_group`. This is `>= meshlet_count`; the
    /// difference is the trailing-workgroup slack the last workgroup leaves
    /// idle. Saturating so a hand-built dispatch cannot overflow.
    #[must_use]
    pub fn covered_slots(self) -> u32 {
        self.dispatch_groups.saturating_mul(self.meshlets_per_group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // A small, internally consistent budget for packing tests.
    // segs = 3 -> verts/strand = 4, prims/strand = 6.
    // by_vertices = 32 / 4 = 8, by_primitives = 48 / 6 = 8, cap = 8 -> 8 strands.
    fn budget() -> MeshStrandParams {
        MeshStrandParams::new(32, 48, 3, 8)
    }

    #[test]
    fn max_strands_balances_all_three_limits() {
        assert_eq!(budget().max_strands_per_meshlet(3), 8);
    }

    #[test]
    fn max_strands_limited_by_vertex_budget() {
        // verts/strand = 4; a 20-vertex budget fits 5 strands even though the
        // primitive budget and cap allow more.
        let params = MeshStrandParams::new(20, 256, 3, 64);
        assert_eq!(params.max_strands_per_meshlet(3), 5);
    }

    #[test]
    fn max_strands_limited_by_primitive_budget() {
        // prims/strand = 6; a 20-primitive budget fits 3 strands.
        let params = MeshStrandParams::new(256, 20, 3, 64);
        assert_eq!(params.max_strands_per_meshlet(3), 3);
    }

    #[test]
    fn max_strands_limited_by_strand_cap() {
        // Budgets allow many, but the explicit cap pins it to 2.
        let params = MeshStrandParams::new(256, 256, 3, 2);
        assert_eq!(params.max_strands_per_meshlet(3), 2);
    }

    #[test]
    fn max_strands_is_at_least_one_for_tiny_budget() {
        // A single strand (segs=8 -> 9 verts) already exceeds a 4-vertex budget,
        // yet packing still owns one strand per group for forward progress.
        let params = MeshStrandParams::new(4, 4, 8, 8);
        assert_eq!(params.max_strands_per_meshlet(8), 1);
    }

    #[test]
    fn pack_exact_multiple_fills_every_group() {
        // 16 strands / 8 per group = exactly 2 full groups.
        let meshlets = pack_strand_meshlets(16, 3, budget());
        assert_eq!(meshlets.len(), 2);
        assert_eq!(meshlets[0], StrandMeshlet::new(0, 8, 3));
        assert_eq!(meshlets[1], StrandMeshlet::new(8, 8, 3));
    }

    #[test]
    fn pack_fans_out_with_partial_remainder() {
        // 19 strands / 8 per group = 8, 8, 3.
        let meshlets = pack_strand_meshlets(19, 3, budget());
        assert_eq!(meshlets.len(), 3);
        assert_eq!(meshlets[0], StrandMeshlet::new(0, 8, 3));
        assert_eq!(meshlets[1], StrandMeshlet::new(8, 8, 3));
        assert_eq!(meshlets[2], StrandMeshlet::new(16, 3, 3));
    }

    #[test]
    fn pack_derives_vertex_and_primitive_counts_within_budget() {
        let meshlets = pack_strand_meshlets(10, 3, budget());
        for m in &meshlets {
            assert!(m.vertex_count() <= budget().max_vertices_per_group);
            assert!(m.primitive_count() <= budget().max_primitives_per_group);
            assert!(m.fits_budget(budget()));
        }
        // First full group: 8 strands * 4 verts = 32, 8 * 6 prims = 48.
        assert_eq!(meshlets[0].vertex_count(), 32);
        assert_eq!(meshlets[0].primitive_count(), 48);
        // Trailing group: 2 strands.
        assert_eq!(meshlets[1], StrandMeshlet::new(8, 2, 3));
        assert_eq!(meshlets[1].vertex_count(), 8);
        assert_eq!(meshlets[1].primitive_count(), 12);
    }

    #[test]
    fn pack_zero_strands_is_empty() {
        assert!(pack_strand_meshlets(0, 3, budget()).is_empty());
    }

    #[test]
    fn pack_zero_segments_is_sanitized_to_one() {
        // segs=0 -> sanitized to 1; verts/strand=2, prims/strand=2.
        let meshlets = pack_strand_meshlets(3, 0, MeshStrandParams::new(256, 256, 8, 64));
        assert!(!meshlets.is_empty());
        for m in &meshlets {
            assert_eq!(m.segments_per_strand, 1);
        }
    }

    #[test]
    fn pack_is_bit_exact_deterministic() {
        let a = pack_strand_meshlets(37, 5, MeshStrandParams::default());
        let b = pack_strand_meshlets(37, 5, MeshStrandParams::default());
        assert_eq!(a, b);
    }

    #[test]
    fn sanitize_clamps_zero_and_overflow() {
        let s = MeshStrandParams::new(0, u32::MAX, 0, 0).sanitized();
        assert_eq!(s.max_vertices_per_group, 1);
        assert_eq!(s.max_primitives_per_group, MAX_PRIMITIVES_PER_GROUP_LIMIT);
        assert_eq!(s.segments_per_strand, 1);
        assert_eq!(s.strands_per_meshlet, 1);
    }

    #[test]
    fn fill_classifies_full_and_partial() {
        let full = StrandMeshlet::new(0, 8, 3);
        let partial = StrandMeshlet::new(8, 3, 3);
        assert_eq!(full.fill(budget()), MeshletFill::Full);
        assert!(full.is_full(budget()));
        assert_eq!(partial.fill(budget()), MeshletFill::Partial);
        assert!(!partial.is_full(budget()));
    }

    #[test]
    fn bin_routes_full_and_partial_in_order() {
        // 19 strands -> [full(0,8), full(8,8), partial(16,3)].
        let meshlets = pack_strand_meshlets(19, 3, budget());
        let bins = bin_strand_meshlets(&meshlets, budget());
        assert_eq!(bins.total(), 3);
        assert_eq!(bins.full.len(), 2);
        assert_eq!(bins.partial.len(), 1);
        assert_eq!(bins.full[0].first_render_strand, 0);
        assert_eq!(bins.full[1].first_render_strand, 8);
        assert_eq!(bins.partial[0], StrandMeshlet::new(16, 3, 3));
        assert_eq!(bins.bucket(MeshletFill::Partial), &bins.partial[..]);
    }

    #[test]
    fn bin_empty_input_is_empty() {
        let bins = bin_strand_meshlets(&[], budget());
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    #[test]
    fn bin_selected_skips_out_of_range_and_preserves_order() {
        // Pack yields three groups; index 9 is out of range and must be skipped.
        let meshlets = pack_strand_meshlets(19, 3, budget());
        let bins = bin_selected_meshlets(&meshlets, &[2, 9, 0], budget());
        assert_eq!(bins.total(), 2);
        assert_eq!(bins.partial.len(), 1);
        assert_eq!(bins.full.len(), 1);
        assert_eq!(bins.partial[0].first_render_strand, 16);
        assert_eq!(bins.full[0].first_render_strand, 0);
    }

    #[test]
    fn dispatch_exact_multiple() {
        assert_eq!(dispatch_group_count(16, 4), 4);
    }

    #[test]
    fn dispatch_rounds_up_on_remainder() {
        assert_eq!(dispatch_group_count(17, 4), 5);
        assert_eq!(dispatch_group_count(1, 8), 1);
    }

    #[test]
    fn dispatch_zero_meshlets_is_zero_groups() {
        assert_eq!(dispatch_group_count(0, 8), 0);
    }

    #[test]
    fn dispatch_zero_per_group_is_sanitized_to_one() {
        // per_group 0 -> 1, so one workgroup per meshlet.
        assert_eq!(dispatch_group_count(5, 0), 5);
    }

    #[test]
    fn dispatch_does_not_overflow_on_huge_count() {
        // u32::MAX / 1 = u32::MAX without overflow in the ceiling step.
        assert_eq!(dispatch_group_count(u32::MAX, 1), u32::MAX);
    }

    #[test]
    fn amplification_dispatch_derives_fields() {
        let d = AmplificationDispatch::for_meshlets(17, 4);
        assert_eq!(d.meshlet_count, 17);
        assert_eq!(d.meshlets_per_group, 4);
        assert_eq!(d.dispatch_groups, 5);
        assert_eq!(d.covered_slots(), 20);
    }

    #[test]
    fn amplification_dispatch_clamps_per_group() {
        let d = AmplificationDispatch::for_meshlets(3, 0);
        assert_eq!(d.meshlets_per_group, 1);
        assert_eq!(d.dispatch_groups, 3);
    }
}
