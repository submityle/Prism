//! A sparse, growable storage-buffer vector of atomic POD records.
//!
//! The retained GPU scene addresses rows by a stable handle index that can be
//! sparse (a freed slot is simply absent, not compacted away). This type holds
//! those rows on the CPU and uploads them to one storage buffer, growing the
//! buffer as the high-water mark rises.
//!
//! It is a Prism replacement for the equivalent `bevy_render` structure, built
//! directly on the RHI so the retained scene's buffers no longer depend on
//! `bevy_render`. Updates are uploaded with
//! [`RenderQueue::write_buffer`](prism_render_driver::RenderQueue::write_buffer)
//! (a correct full upload of the touched image); a delta-job compute path is a
//! later optimization and is deliberately not attempted here.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;

use prism_render_driver::{BufferDescriptor, BufferId, BufferUsages, RenderDevice, RenderQueue};

use crate::atomic_pod::AtomicPod;

/// The smallest capacity (in rows) a buffer allocates.
const MIN_CAPACITY: u32 = 1;

/// A sparse vector of [`AtomicPod`] rows backed by one RHI storage buffer.
pub struct SparseBufferVec<T: AtomicPod> {
    /// The rows present, keyed by handle index.
    rows: BTreeMap<u32, T>,
    /// The debug label surfaced in GPU tooling.
    label: Arc<str>,
    /// The requested usage of the storage buffer.
    usage: BufferUsages,
    /// The number of rows the GPU buffer can hold.
    capacity: u32,
    /// The live RHI buffer, once uploaded at least once.
    buffer: Option<BufferId>,
    /// The byte size of the live buffer (for growth/recreate decisions).
    uploaded_bytes: u64,
}

impl<T: AtomicPod> SparseBufferVec<T> {
    /// Creates an empty vector that will back `usage` (usually
    /// [`BufferUsages::STORAGE`]).
    #[must_use]
    pub fn new(usage: BufferUsages, label: Arc<str>) -> Self {
        Self {
            rows: BTreeMap::new(),
            label,
            usage,
            capacity: 0,
            buffer: None,
            uploaded_bytes: 0,
        }
    }

    /// The number of rows the buffer can currently hold.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The live RHI buffer, or `None` before the first upload.
    #[must_use]
    pub fn buffer(&self) -> Option<BufferId> {
        self.buffer
    }

    /// Stores `value` at sparse `index`, growing the capacity if needed.
    pub fn grow_and_set(&mut self, index: u32, value: T) {
        if index >= self.capacity {
            // Double from the current capacity (or start at the floor) until the
            // index fits, so repeated inserts amortize to O(1) growth.
            let mut capacity = self.capacity.max(MIN_CAPACITY);
            while capacity <= index {
                capacity = capacity.saturating_mul(2);
            }
            self.capacity = capacity;
        }
        self.rows.insert(index, value);
    }

    /// The number of present rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no rows are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Reads back a present row.
    #[must_use]
    pub fn get(&self, index: u32) -> Option<&T> {
        self.rows.get(&index)
    }

    /// (Re)creates the GPU buffer if needed and uploads every present row.
    ///
    /// Absent (sparse) rows are written as zero so the shader sees a defined
    /// value at every allocated index.
    pub fn write_buffer<D: RenderDevice, Q: RenderQueue>(&mut self, device: &D, queue: &Q) {
        let row_size = size_of::<T>();
        let capacity = self.capacity.max(MIN_CAPACITY);
        let byte_len = capacity as u64 * row_size as u64;

        if self.buffer.is_none() || self.uploaded_bytes != byte_len {
            if let Some(old) = self.buffer.take() {
                device.destroy_buffer(old);
            }
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(format!("{}", self.label)),
                size: byte_len.max(1),
                // `COPY_DST` for the upload; `COPY_SRC` so golden/read-back and
                // future compaction passes can read the image back.
                usage: self
                    .usage
                    .union(BufferUsages::COPY_DST)
                    .union(BufferUsages::COPY_SRC),
                mapped_at_creation: false,
            });
            self.buffer = Some(buffer);
            self.uploaded_bytes = byte_len;
        }

        let mut image = vec![0u8; byte_len as usize];
        for (index, row) in &self.rows {
            let offset = *index as usize * row_size;
            image[offset..offset + row_size].copy_from_slice(bytemuck::bytes_of(row));
        }
        if let Some(buffer) = self.buffer {
            queue.write_buffer(buffer, 0, &image);
        }
    }
}

impl<T: AtomicPod> core::fmt::Debug for SparseBufferVec<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseBufferVec")
            .field("label", &self.label)
            .field("rows", &self.rows.len())
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::SparseBufferVec;
    use prism_render_driver::BufferUsages;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct Row {
        a: u32,
        b: u32,
    }

    crate::impl_atomic_pod!(Row, RowBlob);

    fn vec() -> SparseBufferVec<Row> {
        SparseBufferVec::new(BufferUsages::STORAGE, alloc::sync::Arc::from("test rows"))
    }

    #[test]
    fn grows_capacity_to_fit_the_highest_index() {
        let mut v = vec();
        v.grow_and_set(0, Row { a: 1, b: 2 });
        assert_eq!(v.capacity(), 1);
        v.grow_and_set(5, Row { a: 3, b: 4 });
        assert!(v.capacity() > 5, "capacity covers the sparse index");
        assert_eq!(v.len(), 2, "only present rows are stored");
        assert_eq!(v.get(5), Some(&Row { a: 3, b: 4 }));
        assert_eq!(v.get(2), None, "an absent slot reads back as missing");
        assert!(!v.is_empty());
    }

    #[test]
    fn stores_rows_sparsely_without_compacting() {
        let mut v = vec();
        v.grow_and_set(100, Row { a: 7, b: 8 });
        assert_eq!(v.len(), 1);
        assert_eq!(v.get(100), Some(&Row { a: 7, b: 8 }));
    }

    /// Whether a missing adapter must fail instead of skipping.
    fn require_gpu() -> bool {
        std::env::var_os("PRISM_REQUIRE_GPU").is_some()
    }

    #[test]
    fn gpu_upload_round_trips_the_sparse_image() {
        use prism_render_driver_wgpu::WgpuInstance;

        let Some((device, queue)) = WgpuInstance::new().request_default_device().ok() else {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU is set but no adapter was found"
            );
            return;
        };

        let mut v = vec();
        v.grow_and_set(
            3,
            Row {
                a: 0x1122_3344,
                b: 0x5566_7788,
            },
        );
        v.write_buffer(&device, &queue);
        assert!(device.poll_wait(), "the device stays healthy");

        let capacity = v.capacity();
        assert_eq!(capacity, 4, "capacity doubles to fit index 3");
        let bytes = queue.read_buffer(v.buffer().expect("uploaded"), capacity as u64 * 8);

        // The gap at row 0 is zero-filled.
        assert_eq!(&bytes[0..8], &[0u8; 8], "an absent row uploads as zeros");
        // Row 3 carries its little-endian words verbatim.
        assert_eq!(
            &bytes[24..32],
            &[0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55],
            "the present row round-trips byte-for-byte"
        );
    }
}
