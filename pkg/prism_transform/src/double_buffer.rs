//! §24.3 Double / multi-buffer transforms (render-simulation read consistency).
//!
//! When the render thread (or the next frame's extract stage) reads world
//! transforms while simulation is still writing them, it can observe a
//! half-updated frame: some entities moved, some not — a *torn* snapshot. The
//! fix mirrors graphics back-buffering: keep the **simulation-write** column
//! and the **render-read** column in separate storage, and expose an atomic
//! [`MultiBuffer::publish`] that makes the just-written column the new readable
//! snapshot without ever mutating a column a reader might hold.
//!
//! [`MultiBuffer`] generalises this to any buffer count:
//! - **Double** (2 buffers): sim writes the back column, render reads the front
//!   column; `publish` swaps them. Correct when a reader always finishes within
//!   the frame (it reads the published snapshot before the next `publish`).
//! - **Triple** (3+ buffers): sim can begin writing frame *N+1* while render is
//!   still reading the frame *N* snapshot it captured via
//!   [`MultiBuffer::read_token`]. The write column rotates away from both the
//!   newest snapshot and the one a reader is holding, so a captured snapshot
//!   survives exactly `buffer_count - 2` further publishes untouched — no
//!   stalls, no tearing.
//!
//! The type is a pure data structure: `no_std` + `alloc`, no atomics, no
//! threads of its own. It encodes the *ownership discipline* (which column is
//! writable, which are frozen snapshots) that a scheduler then upholds across
//! threads. Writers borrow the back column mutably; readers take a `Copy`
//! [`ReadToken`] and later re-borrow that exact snapshot immutably, so the
//! borrow checker already forbids reading a column while it is being written.

use alloc::vec::Vec;

use crate::GlobalTransform;

/// A `Copy` handle to a published snapshot: the backing buffer index plus the
/// frame version it was published at.
///
/// Capture one with [`MultiBuffer::read_token`], then read its columns with
/// [`MultiBuffer::columns`]. The paired version lets a reader detect — via
/// [`MultiBuffer::is_token_live`] — whether the sim has since rotated back onto
/// that buffer (so the snapshot it names may now be overwritten) without
/// holding a borrow across the intervening [`MultiBuffer::publish`] calls.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReadToken {
    buffer: usize,
    version: u64,
}

impl ReadToken {
    /// The frame version this snapshot was published at. The first published
    /// frame is version `1`; version `0` means "nothing published yet".
    #[inline]
    #[must_use]
    pub const fn version(self) -> u64 {
        self.version
    }
}

/// A rotating set of per-node transform columns providing a tear-free
/// simulation-write / render-read split. See the [module docs](self).
///
/// `T` is the per-node payload; the common case is [`GlobalTransform`], for
/// which [`TransformDoubleBuffer`] / [`TransformTripleBuffer`] are provided.
#[derive(Clone, Debug)]
pub struct MultiBuffer<T> {
    /// `buffers.len()` columns, each `len` entries long.
    buffers: Vec<Vec<T>>,
    /// Entries per column (node count).
    len: usize,
    /// Index of the column simulation currently writes into (the back buffer).
    write_idx: usize,
    /// Index of the newest published column (the front buffer render reads).
    read_idx: usize,
    /// Monotonic published-frame counter; `0` until the first [`publish`](Self::publish).
    version: u64,
}

impl<T: Copy> MultiBuffer<T> {
    /// Create a buffer with `buffer_count` columns (clamped to at least 2) of
    /// `len` entries each, every entry initialised to `fill`.
    ///
    /// The write and read columns start distinct, so a reader before the first
    /// `publish` never shares storage with the writer.
    #[must_use]
    pub fn new(buffer_count: usize, len: usize, fill: T) -> Self {
        let count = buffer_count.max(2);
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            let mut column = Vec::with_capacity(len);
            column.resize(len, fill);
            buffers.push(column);
        }
        Self {
            buffers,
            len,
            // Read column 0 is the initial published-but-empty snapshot; sim
            // writes column 1 so the two never alias before the first publish.
            write_idx: 1,
            read_idx: 0,
            version: 0,
        }
    }

    /// Create a classic double buffer (2 columns).
    #[inline]
    #[must_use]
    pub fn double(len: usize, fill: T) -> Self {
        Self::new(2, len, fill)
    }

    /// Create a triple buffer (3 columns): a captured [`ReadToken`] survives one
    /// further `publish` untouched, fully decoupling a reader from the sim.
    #[inline]
    #[must_use]
    pub fn triple(len: usize, fill: T) -> Self {
        Self::new(3, len, fill)
    }

    /// Number of columns.
    #[inline]
    #[must_use]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Entries per column (node count).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether each column has no entries.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The newest published frame version (`0` before the first `publish`).
    #[inline]
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The simulation-writable back column. Writes here are invisible to
    /// readers of [`MultiBuffer::read`] until the next [`MultiBuffer::publish`].
    #[inline]
    pub fn write(&mut self) -> &mut [T] {
        &mut self.buffers[self.write_idx]
    }

    /// Write a single entry in the back column.
    ///
    /// # Panics
    /// Panics if `index >= len`.
    #[inline]
    pub fn write_at(&mut self, index: usize, value: T) {
        self.buffers[self.write_idx][index] = value;
    }

    /// Copy `src` into the back column wholesale.
    ///
    /// # Panics
    /// Panics if `src.len() != len`.
    pub fn write_all(&mut self, src: &[T]) {
        assert_eq!(src.len(), self.len, "write_all: slice length must equal len");
        self.buffers[self.write_idx].copy_from_slice(src);
    }

    /// Seed the back column with the current published snapshot, so a frame that
    /// only edits a few entries carries the rest forward instead of starting
    /// from stale 2-frames-ago data.
    pub fn prime_write_from_read(&mut self) {
        if self.read_idx != self.write_idx {
            let (read, write) = index_two_mut(&mut self.buffers, self.read_idx, self.write_idx);
            write.copy_from_slice(read);
        }
    }

    /// The newest published snapshot — the tear-free column a renderer reads.
    /// Before the first `publish` this is the initial `fill` column.
    #[inline]
    #[must_use]
    pub fn read(&self) -> &[T] {
        &self.buffers[self.read_idx]
    }

    /// A `Copy` [`ReadToken`] naming the newest published snapshot, for a reader
    /// that must hold it across subsequent `publish` calls without keeping a
    /// borrow.
    #[inline]
    #[must_use]
    pub fn read_token(&self) -> ReadToken {
        ReadToken {
            buffer: self.read_idx,
            version: self.version,
        }
    }

    /// The columns a [`ReadToken`] named at capture time.
    ///
    /// With triple+ buffering the sim writes elsewhere for `buffer_count - 2`
    /// publishes, so these entries stay the exact snapshot that was published —
    /// check [`MultiBuffer::is_token_live`] first if more publishes may have
    /// elapsed.
    ///
    /// # Panics
    /// Panics if the token did not originate from this buffer.
    #[inline]
    #[must_use]
    pub fn columns(&self, token: ReadToken) -> &[T] {
        &self.buffers[token.buffer]
    }

    /// Whether the snapshot a [`ReadToken`] names is still intact — i.e. the sim
    /// has not yet rotated its write column back onto that buffer. A live token
    /// reads consistent, un-torn data from [`MultiBuffer::columns`].
    ///
    /// A snapshot stays live for `buffer_count - 1` publishes after it was
    /// taken (one for double buffering, two for triple, and so on).
    #[inline]
    #[must_use]
    pub fn is_token_live(&self, token: ReadToken) -> bool {
        // The token's buffer is reused once the write column lands back on it,
        // which happens exactly `buffer_count - 1` publishes after capture.
        self.version.saturating_sub(token.version) < self.buffer_count() as u64
            && token.buffer != self.write_idx
    }

    /// Publish the back column as the new readable snapshot and rotate the write
    /// column to the next buffer (never the just-published one). Increments the
    /// version. This is the atomic hand-off a scheduler performs between the sim
    /// finishing a frame and render starting to read it.
    pub fn publish(&mut self) {
        self.read_idx = self.write_idx;
        // Rotate: next column, which (for count >= 2) is never the new read_idx.
        self.write_idx = (self.write_idx + 1) % self.buffers.len();
        self.version += 1;
    }

    /// Convenience: copy `src` into the back column and immediately publish it.
    ///
    /// # Panics
    /// Panics if `src.len() != len`.
    pub fn publish_from(&mut self, src: &[T]) {
        self.write_all(src);
        self.publish();
    }

    /// Grow every column to `new_len` entries, filling new slots with `fill`.
    /// Never shrinks. Does not alter the published version or rotation state.
    pub fn resize(&mut self, new_len: usize, fill: T) {
        if new_len <= self.len {
            return;
        }
        for column in &mut self.buffers {
            column.resize(new_len, fill);
        }
        self.len = new_len;
    }
}

/// Split-borrow two distinct indices of a slice of columns mutably.
///
/// # Panics
/// Panics if `a == b` or either index is out of bounds.
fn index_two_mut<U>(buffers: &mut [U], a: usize, b: usize) -> (&mut U, &mut U) {
    assert_ne!(a, b, "index_two_mut: indices must differ");
    if a < b {
        let (lo, hi) = buffers.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = buffers.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// A double-buffered column of world [`GlobalTransform`]s (sim-write / render-read).
pub type TransformDoubleBuffer = MultiBuffer<GlobalTransform>;

/// A triple-buffered column of world [`GlobalTransform`]s: a captured render
/// snapshot survives one extra `publish`, so render never stalls the sim.
pub type TransformTripleBuffer = MultiBuffer<GlobalTransform>;
