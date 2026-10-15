//! Deterministic per-frame world state hashing (design §14: 逐帧状态哈希去同步).
//!
//! Rollback networking (Quantum / GGPO form) detects desync by comparing a
//! compact hash of each simulated frame between peers. The hash must be
//! build-stable and independent of allocation order, so it is folded in a fixed,
//! deterministic traversal order: entities ascending by [`Entity::to_bits`], and
//! within the world, columns ascending by [`ComponentId`].
//!
//! Honesty: a value only contributes its semantic hash when its component
//! was registered as hashable
//! ([`register_snapshot_component_hashable`](crate::world::World::register_snapshot_component_hashable)).
//! Components that are merely cloneable (e.g. ones holding `f32`, which is not
//! [`core::hash::Hash`]) still contribute their *structural* presence —
//! `(entity, component_id)` — but not their value. The hash can therefore
//! detect structural divergence and value divergence for
//! hash-registered components. A matching hash is not proof of value equality:
//! hashes can collide. Delta reuse and snapshot equality use separately opted-in
//! typed comparison callbacks, never object representations or hash equality.

use core::alloc::Layout;
use core::any::{Any, TypeId};

use crate::collections::HashMap;
use crate::component::{Component, ComponentId};
use crate::resource::{Resource, ResourceId};
use crate::world::World;

/// A captured comparison contract for one concrete Rust type. Its callbacks
/// compare typed values rather than reading their raw object representations.
#[derive(Clone, Copy)]
pub(super) struct SnapshotEq {
    type_id: TypeId,
    layout: Layout,
    equal_ptr: unsafe fn(*const u8, *const u8) -> bool,
    equal_any: fn(&dyn Any, &dyn Any) -> Option<bool>,
}

impl SnapshotEq {
    pub(super) fn of<T: PartialEq + 'static>() -> Self {
        // SAFETY: callers must supply two initialized values of this exact T.
        /// # Safety
        /// Both pointers must be valid, properly aligned, initialized `T`
        /// values for the duration of the comparison.
        unsafe fn equal_ptr<T: PartialEq>(a: *const u8, b: *const u8) -> bool {
            // SAFETY: the function contract guarantees two valid T values.
            unsafe { *a.cast::<T>() == *b.cast::<T>() }
        }
        fn equal_any<T: PartialEq + 'static>(a: &dyn Any, b: &dyn Any) -> Option<bool> {
            Some(a.downcast_ref::<T>()? == b.downcast_ref::<T>()?)
        }
        Self {
            type_id: TypeId::of::<T>(),
            layout: Layout::new::<T>(),
            equal_ptr: equal_ptr::<T>,
            equal_any: equal_any::<T>,
        }
    }

    pub(super) fn matches(&self, type_id: TypeId, layout: Layout) -> bool {
        self.type_id == type_id && self.layout == layout
    }

    pub(super) fn compatible(&self, other: &Self) -> bool {
        self.matches(other.type_id, other.layout)
    }

    /// # Safety
    /// Both pointers must reference initialized values of the type this callback
    /// was created for, including its alignment and aliasing requirements.
    pub(super) unsafe fn compare_ptr(&self, a: *const u8, b: *const u8) -> bool {
        // SAFETY: forwarded typed-value contract.
        unsafe { (self.equal_ptr)(a, b) }
    }

    pub(super) fn compare_any(&self, a: &dyn Any, b: &dyn Any) -> Option<bool> {
        (self.equal_any)(a, b)
    }
}

/// Comparison registration is infrastructure, not rollback state. Keeping it
/// in a private, non-snapshotted resource leaves Component's Clone-only contract
/// and the component registry ABI unchanged.
#[derive(Default)]
struct SnapshotComparisons {
    components: HashMap<ComponentId, SnapshotEq>,
    resources: HashMap<ResourceId, SnapshotEq>,
}

impl Resource for SnapshotComparisons {}

impl World {
    /// Register clone capture/restore and typed value comparison for `T`.
    ///
    /// This is independent of hash registration. The comparison calls `T`'s
    /// `PartialEq`; standard String/Vec implementations compare contents without
    /// reading padding or allocation addresses. Custom implementations define
    /// the value equivalence used for delta reuse. Clone-only registration
    /// remains sufficient for capture and restore; without comparison glue,
    /// independent values are incomparable and delta encoding must retain a
    /// fresh clone conservatively.
    /// Registration affects future captures; existing snapshots keep their
    /// captured comparison contract. A non-reflexive `PartialEq` (such as NaN)
    /// can deliberately prevent equality/reuse even for an unchanged value.
    /// Captures still use `T::clone`; registering comparison does not turn a
    /// shallow clone of shared interior-mutable data into an independent copy.
    pub fn register_snapshot_component_equatable<T: Component + Clone + PartialEq>(
        &mut self,
    ) -> ComponentId {
        let id = self.register_snapshot_component::<T>();
        self.resources.init::<SnapshotComparisons>();
        self.resources
            .get_mut::<SnapshotComparisons>()
            .expect("comparison registry was initialized")
            .components
            .insert(id, SnapshotEq::of::<T>());
        id
    }

    /// Register opt-in clone capture/restore and typed comparison for resource
    /// `T`. Hash equality is never used as a substitute for this comparison.
    /// Existing Clone-only and hashable registration APIs remain available.
    pub fn register_snapshot_resource_equatable<T: Resource + Clone + PartialEq>(
        &mut self,
    ) -> ResourceId {
        let id = self.register_snapshot_resource::<T>();
        self.resources.init::<SnapshotComparisons>();
        self.resources
            .get_mut::<SnapshotComparisons>()
            .expect("comparison registry was initialized")
            .resources
            .insert(id, SnapshotEq::of::<T>());
        id
    }
}

/// Resolve and validate a component's callback before capturing it. Dynamic
/// schemas without a concrete Rust TypeId cannot use a stale typed callback.
pub(super) fn component_equality(world: &World, id: ComponentId) -> Option<SnapshotEq> {
    let eq = *world
        .resources
        .get::<SnapshotComparisons>()?
        .components
        .get(&id)?;
    let info = world.components.info(id)?;
    eq.matches(info.type_id()?, info.layout()).then_some(eq)
}

pub(super) fn resource_equality(world: &World, id: ResourceId) -> Option<SnapshotEq> {
    let eq = *world
        .resources
        .get::<SnapshotComparisons>()?
        .resources
        .get(&id)?;
    (world.resources.type_id_of(id)? == eq.type_id).then_some(eq)
}

/// A small, dependency-free [`core::hash::Hasher`] implementing 64-bit FNV-1a.
///
/// FNV-1a is chosen for determinism and simplicity (no SipHash keys, no std):
/// identical byte streams always fold to the same 64-bit value on every target,
/// which is exactly the contract a cross-peer desync check needs.
/// Integer writes use little-endian encoding, with pointer-sized integers
/// normalized to 64 bits. A custom `Hash` implementation must itself avoid
/// address-dependent or platform-dependent byte streams; the hasher cannot
/// normalize arbitrary bytes passed to `write`.
pub struct FnvHasher {
    state: u64,
}

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl FnvHasher {
    /// Create a hasher seeded with the FNV offset basis.
    #[inline]
    pub fn new() -> Self {
        Self { state: FNV_OFFSET }
    }
}

impl Default for FnvHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl core::hash::Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.state;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        self.state = hash;
    }

    fn write_u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    fn write_u16(&mut self, value: u16) {
        self.write(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn write_u128(&mut self, value: u128) {
        self.write(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_i8(&mut self, value: i8) {
        self.write_u8(value as u8);
    }

    fn write_i16(&mut self, value: i16) {
        self.write_u16(value as u16);
    }

    fn write_i32(&mut self, value: i32) {
        self.write_u32(value as u32);
    }

    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }

    fn write_i128(&mut self, value: i128) {
        self.write_u128(value as u128);
    }

    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }
}
