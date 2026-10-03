//! Buffer creation descriptors.

use crate::flags::BufferUsages;
use alloc::string::String;

/// A request to create a GPU buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The size in bytes.
    pub size: u64,
    /// The permitted GPU usages.
    pub usage: BufferUsages,
    /// Whether the buffer is mapped (CPU-accessible) immediately on creation.
    pub mapped_at_creation: bool,
}

impl BufferDescriptor {
    /// Creates an unmapped buffer descriptor.
    #[must_use]
    pub const fn new(size: u64, usage: BufferUsages) -> Self {
        Self {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        }
    }
}
