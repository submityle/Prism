//! CPU-side packing and incremental upload planning for GPU transform columns.
//!
//! A renderer that draws many instances uploads each visible entity's world
//! matrix to the `GPU` (instanced / indirect draw). [`GlobalTransform`] already
//! holds the world [`Affine3`](prism_math::Affine3); this module packs it into a
//! byte buffer laid out exactly the way a shader expects, and — crucially —
//! re-emits only the entries that changed this frame so the extract stage moves
//! the per-frame dirty delta instead of re-uploading the whole scene every
//! frame (design doc §13).
//!
//! This layer is deliberately device-free: it produces the packed bytes and the
//! compact dirty byte ranges, and a renderer feeds those to its `RHI`
//! (`prism_render_driver`) as buffer writes. No `GPU` device, queue, or
//! backend is required or referenced here.
//!
//! ## Layout
//! [`MatrixLayout`] selects between a compact 48-byte 3x4 affine and a full
//! 64-byte 4x4 matrix, both **row-major** `f32`. Row-major means consecutive
//! floats form a row, and the fourth column of each row carries that row's
//! translation component — the common instance-buffer convention and what a
//! `std140`/`std430` `mat3x4`/`mat4` row-major binding reads. Every scalar is
//! encoded little-endian via [`f32::to_le_bytes`], matching the byte order a
//! little-endian host streams straight to the `GPU`.
//!
//! ## Incremental diff
//! [`GpuColumnBuffer::pack_dirty`] takes the set of entries that changed this
//! frame (the [`DirtyPropagator`](crate::dirty::DirtyPropagator) sweep set,
//! surfaced by [`crate::TransformGraph::recomputed_entries`]), repacks exactly
//! those entries in place, and merges consecutive entry indices into the fewest
//! possible contiguous [`UploadRange`] spans. A frame in which nothing changed
//! yields an empty plan and performs no writes.

use alloc::vec;
use alloc::vec::Vec;

use prism_math::Affine3;

use crate::GlobalTransform;

/// Byte layout of one packed transform column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixLayout {
    /// Row-major 3x4 affine: three rows of four `f32` (48 bytes). Rows 0..3
    /// hold the linear basis in their first three lanes and the translation in
    /// the fourth. The implicit bottom row `[0, 0, 0, 1]` is not stored.
    RowMajor3x4,
    /// Row-major 4x4 matrix: four rows of four `f32` (64 bytes). Identical to
    /// [`MatrixLayout::RowMajor3x4`] with an explicit `[0, 0, 0, 1]` bottom row.
    RowMajor4x4,
}

impl MatrixLayout {
    /// Number of `f32` scalars one entry occupies in this layout.
    #[inline]
    pub const fn float_count(self) -> usize {
        match self {
            MatrixLayout::RowMajor3x4 => 12,
            MatrixLayout::RowMajor4x4 => 16,
        }
    }

    /// Byte size of one packed entry (its stride in the column buffer).
    #[inline]
    pub const fn stride(self) -> usize {
        self.float_count() * size_of::<f32>()
    }
}

/// A half-open byte span `[offset, offset + len)` inside a
/// [`GpuColumnBuffer`]'s packed bytes that a renderer should re-upload.
///
/// Spans are the minimal cover of the dirty entries: consecutive dirty entry
/// indices are merged into a single range, so a renderer issues one buffer
/// write per contiguous run rather than one per entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UploadRange {
    /// Byte offset of the span from the start of [`GpuColumnBuffer::as_bytes`].
    pub offset: usize,
    /// Length of the span in bytes (always a positive multiple of the stride).
    pub len: usize,
}

/// A packed, `GPU`-ready buffer of per-entity world matrices plus the machinery
/// to re-emit only the entries that changed since the last pack.
///
/// Create one with a [`MatrixLayout`], call [`GpuColumnBuffer::pack_all`] once
/// to establish the baseline, then [`GpuColumnBuffer::pack_dirty`] each frame
/// with that frame's changed entries. The returned [`UploadRange`]s are what a
/// renderer writes to its device buffer.
#[derive(Clone, Debug)]
pub struct GpuColumnBuffer {
    layout: MatrixLayout,
    buffer: Vec<u8>,
    count: usize,
}

impl GpuColumnBuffer {
    /// Create an empty buffer with the given layout.
    #[inline]
    pub const fn new(layout: MatrixLayout) -> Self {
        Self { layout, buffer: Vec::new(), count: 0 }
    }

    /// Create an empty buffer with capacity preallocated for `entries` columns.
    #[inline]
    pub fn with_capacity(layout: MatrixLayout, entries: usize) -> Self {
        Self { layout, buffer: Vec::with_capacity(entries * layout.stride()), count: 0 }
    }

    /// The layout entries are packed in.
    #[inline]
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Byte stride of one entry.
    #[inline]
    pub fn stride(&self) -> usize {
        self.layout.stride()
    }

    /// Number of transform columns currently packed.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.count
    }

    /// Whether the buffer holds no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Total packed size in bytes (`entry_count * stride`).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.buffer.len()
    }

    /// The packed bytes, ready to hand to a buffer-write call.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// The byte span of entry `index` in the packed buffer.
    #[inline]
    pub fn entry_range(&self, index: usize) -> UploadRange {
        let stride = self.stride();
        UploadRange { offset: index * stride, len: stride }
    }

    /// (Re)pack **every** entry from `globals`, resizing the buffer to match.
    ///
    /// Returns a plan covering the whole buffer as a single [`UploadRange`] (or
    /// an empty plan when there are no entries) — a full re-upload.
    pub fn pack_all(&mut self, globals: &[GlobalTransform]) -> Vec<UploadRange> {
        let stride = self.stride();
        self.count = globals.len();
        self.buffer.clear();
        self.buffer.resize(self.count * stride, 0);
        for (i, global) in globals.iter().enumerate() {
            pack_entry(self.layout, &global.affine(), &mut self.buffer[i * stride..(i + 1) * stride]);
        }
        if self.buffer.is_empty() {
            Vec::new()
        } else {
            vec![UploadRange { offset: 0, len: self.buffer.len() }]
        }
    }

    /// Incrementally repack only the entries listed in `dirty` (as entry
    /// indices) and return the merged byte ranges to re-upload.
    ///
    /// `globals` is the current, fully-propagated world-transform array;
    /// `dirty` are the indices whose world transform changed this frame (order
    /// and duplicates do not matter — they are sorted and de-duplicated
    /// internally). If `globals` grew since the last pack the buffer grows to
    /// fit; the caller is expected to include any newly-spawned indices in
    /// `dirty` so they are packed.
    ///
    /// When `dirty` is empty this writes nothing and returns an empty plan, so
    /// a static frame uploads zero bytes.
    pub fn pack_dirty(
        &mut self,
        globals: &[GlobalTransform],
        dirty: &[u32],
    ) -> Vec<UploadRange> {
        let stride = self.stride();

        // Keep the backing store sized to the live entity set. Growth is the
        // common case (spawns); a shrink drops trailing columns.
        if globals.len() != self.count {
            self.count = globals.len();
            self.buffer.resize(self.count * stride, 0);
        }

        if dirty.is_empty() {
            return Vec::new();
        }

        // Sort + dedup the dirty indices so consecutive entries can be merged
        // into contiguous spans. Out-of-range indices (e.g. a stale index after
        // a shrink) are skipped.
        let mut indices: Vec<usize> = dirty
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| i < self.count)
            .collect();
        indices.sort_unstable();
        indices.dedup();

        if indices.is_empty() {
            return Vec::new();
        }

        // Repack each dirty entry in place.
        for &i in &indices {
            pack_entry(
                self.layout,
                &globals[i].affine(),
                &mut self.buffer[i * stride..(i + 1) * stride],
            );
        }

        // Merge consecutive entry indices into the fewest byte spans.
        let mut ranges: Vec<UploadRange> = Vec::new();
        let mut run_start = indices[0];
        let mut run_end = indices[0]; // inclusive
        for &i in &indices[1..] {
            if i == run_end + 1 {
                run_end = i;
            } else {
                ranges.push(UploadRange {
                    offset: run_start * stride,
                    len: (run_end - run_start + 1) * stride,
                });
                run_start = i;
                run_end = i;
            }
        }
        ranges.push(UploadRange {
            offset: run_start * stride,
            len: (run_end - run_start + 1) * stride,
        });
        ranges
    }
}

/// Pack one world affine into `out`, which must be exactly `layout.stride()`
/// bytes. Scalars are written row-major, little-endian.
fn pack_entry(layout: MatrixLayout, affine: &Affine3, out: &mut [u8]) {
    match layout {
        MatrixLayout::RowMajor3x4 => write_floats(&rows_3x4(affine), out),
        MatrixLayout::RowMajor4x4 => write_floats(&rows_4x4(affine), out),
    }
}

/// The 12 row-major scalars of a 3x4 affine: row `r` is
/// `[basis[0][r], basis[1][r], basis[2][r], translation[r]]`.
#[inline]
fn rows_3x4(a: &Affine3) -> [f32; 12] {
    let m = a.matrix3;
    let t = a.translation;
    [
        m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x,
        m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y,
        m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z,
    ]
}

/// The 16 row-major scalars of a 4x4 matrix: the 3x4 rows plus the implicit
/// `[0, 0, 0, 1]` bottom row.
#[inline]
fn rows_4x4(a: &Affine3) -> [f32; 16] {
    let m = a.matrix3;
    let t = a.translation;
    [
        m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x,
        m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y,
        m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Write `floats` little-endian into `out` (`out.len() == 4 * floats.len()`).
#[inline]
fn write_floats(floats: &[f32], out: &mut [u8]) {
    for (f, chunk) in floats.iter().zip(out.chunks_exact_mut(4)) {
        chunk.copy_from_slice(&f.to_le_bytes());
    }
}
