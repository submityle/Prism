//! Opaque, generational handle identifying a proxy stored in a [`DynamicBvh`].
//!
//! [`DynamicBvh`]: crate::bvh::DynamicBvh

/// A stable, opaque handle to a leaf proxy in a [`DynamicBvh`].
///
/// It pairs a slot index with a generation counter so that a handle to a
/// removed proxy is not accidentally confused with a later proxy that reuses
/// the same slot. Compare handles for equality rather than inspecting the
/// [`ProxyId::index`] directly.
///
/// [`DynamicBvh`]: crate::bvh::DynamicBvh
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ProxyId {
    /// Index of the backing node slot in the tree's node pool.
    index: u32,
    /// Generation of the slot at the time this handle was issued.
    generation: u32,
}

impl ProxyId {
    /// A sentinel handle that never refers to a live proxy.
    ///
    /// Useful as an initial or "cleared" value; it never compares equal to a
    /// handle returned by [`DynamicBvh::insert`].
    ///
    /// [`DynamicBvh::insert`]: crate::bvh::DynamicBvh::insert
    pub const NONE: ProxyId = ProxyId {
        index: u32::MAX,
        generation: u32::MAX,
    };

    /// Creates a handle from raw parts.
    ///
    /// This is crate-internal: only the tree issues valid handles.
    pub(crate) const fn new(index: u32, generation: u32) -> ProxyId {
        ProxyId { index, generation }
    }

    /// Returns the backing node-slot index.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation stamped on this handle.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl Default for ProxyId {
    #[inline]
    fn default() -> Self {
        ProxyId::NONE
    }
}

#[cfg(test)]
mod tests {
    use super::ProxyId;

    #[test]
    fn none_is_distinct_and_accessors_round_trip() {
        let id = ProxyId::new(3, 7);
        assert_eq!(id.index(), 3);
        assert_eq!(id.generation(), 7);
        assert_ne!(id, ProxyId::NONE);
        assert_eq!(ProxyId::default(), ProxyId::NONE);
    }

    #[test]
    fn equality_considers_generation() {
        let a = ProxyId::new(1, 1);
        let b = ProxyId::new(1, 2);
        assert_ne!(a, b);
        assert_eq!(a, ProxyId::new(1, 1));
    }
}
