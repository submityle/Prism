//! A mesh's index list, in whichever width the author had.

use alloc::vec::Vec;

use prism_render_driver::IndexFormat;

/// A mesh's indices, stored at the width they were authored in.
///
/// Keeping the authored width matters: a 16-bit index list costs half the upload
/// and half the index-buffer bandwidth of a 32-bit one, and widening it "for
/// safety" is exactly the kind of silent cost a renderer should not pay. Callers
/// that need one width can ask for it with [`to_u32`](Indices::to_u32).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Indices {
    /// 16-bit indices.
    Uint16(Vec<u16>),
    /// 32-bit indices.
    Uint32(Vec<u32>),
}

impl Indices {
    /// The number of indices.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Uint16(values) => values.len(),
            Self::Uint32(values) => values.len(),
        }
    }

    /// Whether there are no indices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The index width the GPU fetches.
    #[must_use]
    pub const fn format(&self) -> IndexFormat {
        match self {
            Self::Uint16(_) => IndexFormat::Uint16,
            Self::Uint32(_) => IndexFormat::Uint32,
        }
    }

    /// The index buffer bytes, little-endian at the authored width.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Uint16(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 2);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            Self::Uint32(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
        }
    }

    /// The indices widened to `u32`, for callers that index other arrays.
    #[must_use]
    pub fn to_u32(&self) -> Vec<u32> {
        match self {
            Self::Uint16(values) => values.iter().map(|value| u32::from(*value)).collect(),
            Self::Uint32(values) => values.clone(),
        }
    }

    /// The largest index in the list, or `None` when it is empty.
    ///
    /// The renderer uses it to check that a mesh's indices stay inside its own
    /// vertex range — an out-of-range index is a content bug that would otherwise
    /// read another mesh's vertices on the GPU.
    #[must_use]
    pub fn max_index(&self) -> Option<u32> {
        match self {
            Self::Uint16(values) => values.iter().copied().max().map(u32::from),
            Self::Uint32(values) => values.iter().copied().max(),
        }
    }
}

