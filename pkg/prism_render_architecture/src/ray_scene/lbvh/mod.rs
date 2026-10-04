//! Parallel linear bounding-volume hierarchy (`LBVH`), Karras-style.
//!
//! The binned-surface-area-heuristic (`SAH`) builder in
//! [`crate::ray_scene::bvh`] is the quality reference, but it is serial. For
//! realtime, rebuild-every-frame dynamic geometry (skinned meshes, destruction,
//! particles promoted to triangles) the renderer needs a hierarchy it can
//! reconstruct in parallel on the `GPU` each frame. The linear `BVH` of Karras
//! (2012) does exactly that in four data-parallel phases:
//!
//! 1. **Quantize** each primitive centroid to a 30-bit `Morton` key
//!    ([`morton`]).
//! 2. **Sort** the keys with a radix sort ([`radix_sort`]).
//! 3. **Emit** a binary radix tree whose internal nodes are each derived
//!    independently from the sorted keys ([`radix_tree`]).
//! 4. **Fit** bounding boxes bottom-up and flatten to the shared depth-first
//!    layout ([`build`]).
//!
//! Every phase is a pure classical computation with a deterministic result, so
//! this `CPU` module doubles as the golden reference a `GPU` build kernel must
//! match bit-for-bit. The output [`Bvh`](crate::ray_scene::bvh::Bvh) is
//! interchangeable with a `SAH`-built one for traversal and `GPU` upload.

pub mod build;
pub mod morton;
pub mod radix_sort;
pub mod radix_tree;

pub use build::LinearBvh;

#[cfg(test)]
mod tests;
