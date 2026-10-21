//! Storage-independent identity for saved authoring bytes.

use crate::ContentHash;

/// Maximum external authoring object length. Materialization has a separate
/// per-operation budget; a small package never authorizes an unbounded read.
pub const MAX_EDITABLE_EXTERNAL_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct EditablePayloadRef {
    pub content_hash: ContentHash,
    pub byte_len: u64,
}

/// External references address the owning workspace's immutable payload CAS.
/// They do not address the build store or the original import source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditablePayload {
    Inline(Vec<u8>),
    External(EditablePayloadRef),
}

impl From<Vec<u8>> for EditablePayload {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Inline(bytes)
    }
}

impl EditablePayload {
    pub fn byte_len(&self) -> u64 {
        match self {
            Self::Inline(bytes) => bytes.len() as u64,
            Self::External(reference) => reference.byte_len,
        }
    }

    pub fn content_hash(&self) -> ContentHash {
        match self {
            Self::Inline(bytes) => ContentHash::digest(bytes),
            Self::External(reference) => reference.content_hash,
        }
    }

    pub fn inline_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Inline(bytes) => Some(bytes),
            Self::External(_) => None,
        }
    }

    /// Compares content identity without fetching an external object. Readers
    /// must still verify external bytes before making them visible to a cooker.
    pub fn semantic_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Inline(a), Self::Inline(b)) => a == b,
            _ => self.byte_len() == other.byte_len() && self.content_hash() == other.content_hash(),
        }
    }

    pub(crate) fn encoded_len(&self) -> usize {
        match self {
            Self::Inline(bytes) => 1usize.saturating_add(4).saturating_add(bytes.len()),
            Self::External(_) => 1 + 32 + 8,
        }
    }
}
