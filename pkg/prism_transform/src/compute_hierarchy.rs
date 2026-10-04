//! GPU-side hierarchy propagation (§24.7): the **device-free** scheduling and
//! upload-payload layer.
//!
//! Massive hierarchies (cluster animation, foliage, crowds) can bottleneck a
//! CPU propagation pass. The design (§24.7) pushes propagation onto a GPU
//! compute shader: the parent-index topology and the authoritative local
//! transforms are uploaded once, a compute pass walks the forest **by depth
//! level** multiplying each child by its already-resolved parent, and the world
//! matrices stay resident in a GPU buffer for instancing / indirect draw with
//! no CPU readback.
//!
//! The real WGSL dispatch and the device-resident buffer wiring live in the
//! consuming render/driver crate (see [`crate::compute_hierarchy`] honest
//! boundary and the design doc §24.9). What *is* portable — and what this
//! module owns — is everything a compute pass needs that does not touch a
//! device:
//!
//! - [`LevelSchedule`]: the per-level dispatch plan. Nodes are bucketed by
//!   depth; a level's nodes are mutually independent (their parents all live in
//!   shallower, already-finalized levels), so the GPU can dispatch one level at
//!   a time and run every node in it in parallel. Within a level the order is
//!   sorted by [`NodeId`] so the schedule is bit-identical across runs.
//! - [`propagate_by_levels`]: the CPU reference that evaluates the exact same
//!   per-level multiply-accumulate the shader performs. It is the **parity
//!   oracle** for a GPU twin and is bit-for-bit equal to the serial
//!   [`crate::propagate`] (same [`Affine3`](prism_math::Affine3) composition,
//!   same operand order).
//! - [`ComputeHierarchyInput`]: a device-free upload payload — the parent-index
//!   array (`-1` marks a root), the local matrices packed row-major with the
//!   same [`MatrixLayout`] the output path uses, and the flattened dispatch
//!   order plus per-level ranges. A driver uploads these verbatim.
//!
//! This module is `no_std + alloc`; it never allocates a device, spawns a
//! thread, or reads a clock.

use alloc::vec::Vec;
use core::ops::Range;

use prism_math::Affine3;

use crate::gpu_upload::MatrixLayout;
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::{GlobalTransform, Transform};

/// A depth-bucketed dispatch plan for a fixed hierarchy topology.
///
/// Building the schedule is `O(n log n)` (the `log` is the per-level
/// determinism sort); it stays valid as long as the parent/child edges are
/// unchanged, so a static topology can reuse it across frames while only the
/// [`Transform`] *values* change. This is the GPU analogue of
/// [`LevelPlan`](crate::parallel::LevelPlan): `LevelPlan` drives the `std`
/// thread pool, whereas `LevelSchedule` is `no_std` and models the per-level
/// compute **dispatch** (flattened order + explicit level ranges + per-node
/// depth) a shader and its upload payload consume.
#[derive(Clone, Debug, Default)]
pub struct LevelSchedule {
    /// Every node id, grouped by depth (shallow to deep); within a level the
    /// ids are ascending so the dispatch order is deterministic.
    order: Vec<NodeId>,
    /// `level_ranges[d]` slices [`LevelSchedule::order`] to the nodes at depth
    /// `d`. Dispatch one level's range at a time, shallow to deep.
    level_ranges: Vec<Range<usize>>,
    /// Depth of each node, indexed by [`NodeId::index`] (root depth is 0).
    depth: Vec<u32>,
}

impl LevelSchedule {
    /// Build a schedule from `hierarchy`.
    ///
    /// # Errors
    /// [`HierarchyError::Cycle`] if the hierarchy is not a forest.
    pub fn build(hierarchy: &Hierarchy) -> Result<Self, HierarchyError> {
        // `compute_order` validates acyclicity and lists parents before
        // children, so a single forward pass resolves every depth.
        let order_pbc = hierarchy.compute_order()?;
        let n = hierarchy.len();

        let mut depth = alloc::vec![0u32; n];
        let mut max_depth = 0u32;
        for &node in &order_pbc {
            let d = match hierarchy.parent(node) {
                None => 0,
                Some(parent) => depth[parent.index()] + 1,
            };
            depth[node.index()] = d;
            max_depth = max_depth.max(d);
        }

        // Counting sort of nodes into contiguous depth buckets.
        let num_levels = if n == 0 { 0 } else { (max_depth as usize) + 1 };
        let mut counts = alloc::vec![0usize; num_levels];
        for &node in &order_pbc {
            counts[depth[node.index()] as usize] += 1;
        }
        let mut level_ranges = Vec::with_capacity(num_levels);
        let mut acc = 0usize;
        for &c in &counts {
            level_ranges.push(acc..acc + c);
            acc += c;
        }
        let mut cursor: Vec<usize> = level_ranges.iter().map(|r| r.start).collect();
        let mut order = alloc::vec![NodeId::new(0); n];
        for &node in &order_pbc {
            let d = depth[node.index()] as usize;
            order[cursor[d]] = node;
            cursor[d] += 1;
        }

        // Make each level's dispatch order independent of child-insertion
        // order: sort by node index so the plan is identical across runs.
        for range in &level_ranges {
            order[range.clone()].sort_unstable_by_key(|id| id.index());
        }

        Ok(Self {
            order,
            level_ranges,
            depth,
        })
    }

    /// Number of depth levels (0 for an empty hierarchy).
    #[inline]
    pub fn level_count(&self) -> usize {
        self.level_ranges.len()
    }

    /// The deepest depth present, or `None` when the hierarchy is empty.
    #[inline]
    pub fn max_depth(&self) -> Option<u32> {
        if self.level_ranges.is_empty() {
            None
        } else {
            Some((self.level_ranges.len() - 1) as u32)
        }
    }

    /// Total number of nodes covered.
    #[inline]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the schedule covers no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The flattened dispatch order: every node id, shallow level first.
    #[inline]
    pub fn order(&self) -> &[NodeId] {
        &self.order
    }

    /// The ids at depth `level` (the slice a shader dispatches for that level).
    ///
    /// # Panics
    /// Panics if `level >= self.level_count()`.
    #[inline]
    pub fn level(&self, level: usize) -> &[NodeId] {
        &self.order[self.level_ranges[level].clone()]
    }

    /// The per-level slice ranges into [`LevelSchedule::order`].
    #[inline]
    pub fn level_ranges(&self) -> &[Range<usize>] {
        &self.level_ranges
    }

    /// The depth of `node` (root is 0).
    #[inline]
    pub fn node_depth(&self, node: NodeId) -> u32 {
        self.depth[node.index()]
    }
}

/// Evaluate the hierarchy level by level, exactly as the GPU compute pass does.
///
/// This is the CPU **parity oracle** for a GPU twin: it walks
/// [`LevelSchedule::order`] one depth level at a time, computing each node's
/// world transform from its parent's already-finalized world transform. Because
/// it uses the same [`Affine3`](prism_math::Affine3) composition and operand
/// order as [`crate::propagate`], the result is bit-for-bit identical to a
/// serial pass; only the evaluation *grouping* (by level) differs.
///
/// # Errors
/// - [`HierarchyError::LengthMismatch`] if either buffer length differs from
///   the schedule's node count.
pub fn propagate_by_levels(
    hierarchy: &Hierarchy,
    schedule: &LevelSchedule,
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) -> Result<(), HierarchyError> {
    let n = schedule.len();
    if locals.len() != n || globals.len() != n {
        return Err(HierarchyError::LengthMismatch);
    }
    for level in 0..schedule.level_count() {
        for &node in schedule.level(level) {
            let local = &locals[node.index()];
            let world = match hierarchy.parent(node) {
                None => GlobalTransform::from_transform(local),
                Some(parent) => globals[parent.index()].mul_transform(local),
            };
            globals[node.index()] = world;
        }
    }
    Ok(())
}

/// A device-free upload payload for a GPU compute-hierarchy pass.
///
/// Everything here is laid out so a driver can `memcpy` it into storage buffers
/// and dispatch one group per level without further CPU work. The matrices use
/// the same row-major [`MatrixLayout`] as [`crate::gpu_upload`], so a shader can
/// share the decode with the output/instancing path.
#[derive(Clone, Debug)]
pub struct ComputeHierarchyInput {
    /// Parent index per node (node-indexed). A root stores `-1`; otherwise the
    /// value is the parent's [`NodeId::index`]. `i32` matches a GPU `i32`
    /// storage buffer and bounds the addressable node count to `i32::MAX`.
    parents: Vec<i32>,
    /// Local transforms packed row-major, node-indexed, `layout.stride()` bytes
    /// each. These are the authoritative inputs the shader multiplies up the
    /// tree; the output world buffer is produced device-side.
    local_matrices: Vec<u8>,
    /// Flattened per-level dispatch order (node indices), shallow level first.
    dispatch_order: Vec<u32>,
    /// `(start, end)` into [`ComputeHierarchyInput::dispatch_order`] per level.
    level_ranges: Vec<(u32, u32)>,
    /// Matrix encoding used for [`ComputeHierarchyInput::local_matrices`].
    layout: MatrixLayout,
}

impl ComputeHierarchyInput {
    /// Build the payload for `hierarchy` with the given `schedule` and `locals`.
    ///
    /// `locals[i]` is node `i`'s authoritative local transform. The local
    /// matrices are packed from [`Transform::to_affine`] with `layout`.
    ///
    /// # Panics
    /// Panics if `locals.len()` differs from the hierarchy node count or the
    /// node count exceeds `i32::MAX` (parents are encoded as `i32`).
    pub fn pack(
        hierarchy: &Hierarchy,
        schedule: &LevelSchedule,
        locals: &[Transform],
        layout: MatrixLayout,
    ) -> Self {
        let n = hierarchy.len();
        assert_eq!(
            locals.len(),
            n,
            "locals length must equal the hierarchy node count"
        );
        assert!(
            n <= i32::MAX as usize,
            "hierarchy node count exceeds the i32 parent-index range"
        );

        let mut parents = alloc::vec![-1i32; n];
        for (i, slot) in parents.iter_mut().enumerate() {
            let node = NodeId::new(i as u32);
            if let Some(parent) = hierarchy.parent(node) {
                *slot = parent.index() as i32;
            }
        }

        let stride = layout.stride();
        let mut local_matrices = alloc::vec![0u8; n * stride];
        for (i, local) in locals.iter().enumerate() {
            let affine = local.to_affine();
            let out = &mut local_matrices[i * stride..(i + 1) * stride];
            pack_affine_row_major(layout, &affine, out);
        }

        let dispatch_order: Vec<u32> =
            schedule.order().iter().map(|id| id.index() as u32).collect();
        let level_ranges: Vec<(u32, u32)> = schedule
            .level_ranges()
            .iter()
            .map(|r| (r.start as u32, r.end as u32))
            .collect();

        Self {
            parents,
            local_matrices,
            dispatch_order,
            level_ranges,
            layout,
        }
    }

    /// Parent index per node (`-1` for a root).
    #[inline]
    pub fn parents(&self) -> &[i32] {
        &self.parents
    }

    /// Packed local matrices (node-indexed, `matrix_stride()` bytes each).
    #[inline]
    pub fn local_matrices(&self) -> &[u8] {
        &self.local_matrices
    }

    /// Flattened per-level dispatch order (node indices).
    #[inline]
    pub fn dispatch_order(&self) -> &[u32] {
        &self.dispatch_order
    }

    /// `(start, end)` dispatch-order slices per level.
    #[inline]
    pub fn level_ranges(&self) -> &[(u32, u32)] {
        &self.level_ranges
    }

    /// Matrix encoding of the local matrices.
    #[inline]
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Bytes per packed matrix.
    #[inline]
    pub fn matrix_stride(&self) -> usize {
        self.layout.stride()
    }

    /// Number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.parents.len()
    }

    /// Number of dispatch levels.
    #[inline]
    pub fn level_count(&self) -> usize {
        self.level_ranges.len()
    }
}

/// Pack one affine into `out` row-major, little-endian, matching the layout the
/// output/instancing path in [`crate::gpu_upload`] uses so a shader shares one
/// decode. `out.len()` must equal `layout.stride()`.
fn pack_affine_row_major(layout: MatrixLayout, affine: &Affine3, out: &mut [u8]) {
    let m = affine.matrix3;
    let t = affine.translation;
    // Row-major rows: row r = [basis.x[r], basis.y[r], basis.z[r], translation[r]].
    let rows_3x4 = [
        m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x, //
        m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y, //
        m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z,
    ];
    match layout {
        MatrixLayout::RowMajor3x4 => write_floats(&rows_3x4, out),
        MatrixLayout::RowMajor4x4 => {
            let mut rows_4x4 = [0.0f32; 16];
            rows_4x4[..12].copy_from_slice(&rows_3x4);
            rows_4x4[15] = 1.0;
            write_floats(&rows_4x4, out);
        }
    }
}

/// Write `floats` little-endian into `out` (`out.len() == 4 * floats.len()`).
#[inline]
fn write_floats(floats: &[f32], out: &mut [u8]) {
    for (f, chunk) in floats.iter().zip(out.chunks_exact_mut(4)) {
        chunk.copy_from_slice(&f.to_le_bytes());
    }
}
