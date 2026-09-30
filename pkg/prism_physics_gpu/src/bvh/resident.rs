//! Device-resident `LBVH` produced by [`GpuLbvh::build_resident`].
//!
//! [`GpuResidentLbvh`] holds a fully built hierarchy in `GPU` memory: the node
//! links, the order-encoded internal-node bounds, the original-order primitive
//! boxes, and the sorted leaf-slot primitive indices, all left in the device
//! buffers the build wrote. A query binds them directly, so neither the build's
//! result nor the query's inputs make a host round-trip; this is the standard
//! `AAA` arrangement, where the acceleration structure stays resident on device
//! between build and traversal.
//!
//! Trees with fewer than two leaves have no traversable hierarchy, so they hold
//! no buffers and a query over them yields no pairs.
//!
//! # Provenance
//!
//! Plain resource ownership over the linear `BVH` of Karras, "Maximizing
//! Parallelism in the Construction of BVHs, Octrees, and k-d Trees" (High
//! Performance Graphics 2012). No Unreal Engine source or derived code.

use wgpu::Buffer;

use crate::radix::gpu::SortedBuffers;

use super::gpu::RecordedBuild;

/// A device-resident `LBVH`: the built hierarchy kept in `GPU` memory so a query
/// can bind it directly, with no host round-trip between build and traversal.
pub struct GpuResidentLbvh {
    /// The resident device buffers, or `None` for fewer than two leaves (no
    /// traversable hierarchy).
    inner: Option<ResidentInner>,
    /// Number of leaf primitives (`0` or `1` when `inner` is `None`).
    num_leaves: usize,
}

/// The device buffers of a resident `LBVH` with at least two leaves.
///
/// Every field names a buffer the build left in device memory, kept alive so a
/// query can bind it without any re-upload.
pub(crate) struct ResidentInner {
    /// Original-order primitive box minimum corners, `vec4` lanes (`w` unused).
    pub(crate) aabb_min: Buffer,
    /// Original-order primitive box maximum corners, `vec4` lanes (`w` unused).
    pub(crate) aabb_max: Buffer,
    /// Sorted radix output; its values are the leaf-slot primitive indices.
    pub(crate) sorted: SortedBuffers,
    /// Left child (encoded id) per internal node.
    pub(crate) left: Buffer,
    /// Right child (encoded id) per internal node.
    pub(crate) right: Buffer,
    /// Parent (encoded id) of every node, indexed by encoded id.
    pub(crate) parent: Buffer,
    /// Order-encoded internal-node minimum bounds, three lanes per node.
    pub(crate) node_min: Buffer,
    /// Order-encoded internal-node maximum bounds, three lanes per node.
    pub(crate) node_max: Buffer,
    /// Number of internal nodes (`num_leaves - 1`).
    pub(crate) num_internal: usize,
    /// Encoded id of the root node (always `0` for this builder).
    pub(crate) root: u32,
}

impl GpuResidentLbvh {
    /// The resident tree for a trivial input of `num_leaves <= 1`: it owns no
    /// device buffers, and any query over it yields no pairs.
    #[must_use]
    pub(crate) fn empty(num_leaves: usize) -> GpuResidentLbvh {
        GpuResidentLbvh {
            inner: None,
            num_leaves,
        }
    }

    /// Wraps the buffers of a just-recorded build as a resident tree.
    ///
    /// Consumes the [`RecordedBuild`] so its buffers keep living inside the
    /// resident tree; the build's per-pass bind groups are no longer needed once
    /// the recorded passes have been submitted, so they are dropped here.
    #[must_use]
    pub(crate) fn from_recorded(rec: RecordedBuild) -> GpuResidentLbvh {
        let num_leaves = rec.num_leaves;
        GpuResidentLbvh {
            inner: Some(ResidentInner {
                aabb_min: rec.aabb_min,
                aabb_max: rec.aabb_max,
                sorted: rec.sorted,
                left: rec.left,
                right: rec.right,
                parent: rec.parent,
                node_min: rec.node_min,
                node_max: rec.node_max,
                num_internal: rec.num_internal,
                root: 0,
            }),
            num_leaves,
        }
    }

    /// The number of leaf primitives in the tree.
    #[must_use]
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// The number of internal nodes, or `0` for a trivial (fewer than two leaf)
    /// tree.
    #[must_use]
    pub fn num_internal(&self) -> usize {
        self.inner.as_ref().map_or(0, |inner| inner.num_internal)
    }

    /// The resident device buffers, or `None` for a trivial (fewer than two
    /// leaf) tree that has none.
    #[must_use]
    pub(crate) fn buffers(&self) -> Option<&ResidentInner> {
        self.inner.as_ref()
    }
}
