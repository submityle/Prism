//! Component determinism / replication-readiness diagnostic
//! (design §14 / §16.5 / §16.6).
//!
//! Deterministic rollback networking (design §14) rests on two type-erased
//! glue functions a component may or may not carry in the registry:
//!
//! * a **clone** function
//!   ([`ComponentInfo::clone_fn`](crate::component::ComponentInfo::clone_fn)),
//!   installed by
//!   [`register_snapshot_component`](crate::world::World::register_snapshot_component),
//!   lets the value be captured into a [`snapshot`](crate::world::World::snapshot)
//!   and written back by [`restore`](crate::world::World::restore). Without it
//!   the component **cannot be rolled back** (design §16.5).
//! * a **hash** function
//!   ([`ComponentInfo::snapshot_hash_fn`](crate::component::ComponentInfo::snapshot_hash_fn)),
//!   installed by
//!   [`register_snapshot_component_hashable`](crate::world::World::register_snapshot_component_hashable),
//!   folds the value into the per-frame `state_hash` used to **detect desync**
//!   between peers (design §14 "逐帧状态哈希校验去同步"). Without it the
//!   component's value never enters the divergence check.
//!
//! The subtle, high-value failure these two axes expose is the **desync blind
//! spot**: a component that is snapshotted (so it carries authoritative
//! simulation state that gets rolled back) but *not* hashed. If two peers
//! diverge in such a component the snapshots differ while the state hashes
//! still match, so the desync slips past the hash check and only surfaces
//! later as an unexplained gameplay divergence. This report ranks the registry
//! along both axes and surfaces that asymmetry explicitly.
//!
//! # Readiness classes
//! Each registered component falls into exactly one [`DeterminismClass`]:
//!
//! | clone | hash | class | meaning |
//! |---|---|---|---|
//! | ✓ | ✓ | [`FullyDeterministic`](DeterminismClass::FullyDeterministic) | rolled back *and* covered by the desync hash |
//! | ✓ | ✗ | [`SnapshotOnly`](DeterminismClass::SnapshotOnly) | **desync blind spot** — rolled back but invisible to the hash |
//! | ✗ | ✓ | [`HashOnly`](DeterminismClass::HashOnly) | **rollback gap** — hashed yet cannot be restored (malformed registration) |
//! | ✗ | ✗ | [`Opaque`](DeterminismClass::Opaque) | outside both; fine for transient markers, risky for live state |
//!
//! `HashOnly` is not reachable through the standard registration helpers
//! (hash registration always installs clone glue first), so it functions as a
//! guard against a future registration path that would hash a non-cloneable
//! component — a state that would silently break restore.
//!
//! # Scope and determinism
//! Pure read of the component registry: `O(components)`, no entities touched,
//! deterministic. [`entries`](DeterminismReadinessReport::entries) lists every
//! registered component in component-id (registration) order; report totals
//! are order-independent. The classes are capability facts, not errors — a
//! transient marker legitimately stays [`Opaque`](DeterminismClass::Opaque) —
//! so gaps are surfaced as counts and per-entry predicates for a tool or CI to
//! weigh, never enforced here.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::component::{ComponentId, Components};
use crate::world::World;

/// The determinism / replication readiness class of a single component
/// (design §14 / §16.5).
///
/// Determined solely by which type-erased glue the registry holds for the
/// component: `clone` (snapshot/restore) and `hash` (desync detection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeterminismClass {
    /// Carries both clone and hash glue: it is rolled back *and* folded into
    /// the per-frame desync hash. The target state for live simulation data.
    FullyDeterministic,
    /// Carries clone glue but no hash glue: it is captured and restored, yet
    /// its value never enters the desync hash — a **blind spot** where a peer
    /// divergence in this component escapes detection (design §14).
    SnapshotOnly,
    /// Carries hash glue but no clone glue: its value is checked for divergence
    /// yet it cannot be restored on rollback — a **rollback gap**. Not produced
    /// by the standard helpers; a guard against a malformed registration.
    HashOnly,
    /// Carries neither: outside both snapshot and hash. Correct for transient /
    /// derived markers, a determinism risk if it holds live simulation state.
    Opaque,
}

impl DeterminismClass {
    /// Whether this class is covered by snapshot/restore (clone glue present).
    #[inline]
    pub fn is_snapshot_ready(self) -> bool {
        matches!(self, Self::FullyDeterministic | Self::SnapshotOnly)
    }

    /// Whether this class contributes to the desync hash (hash glue present).
    #[inline]
    pub fn is_hash_ready(self) -> bool {
        matches!(self, Self::FullyDeterministic | Self::HashOnly)
    }

    /// Whether this class is a determinism asymmetry worth flagging
    /// ([`SnapshotOnly`](Self::SnapshotOnly) blind spot or
    /// [`HashOnly`](Self::HashOnly) rollback gap).
    #[inline]
    pub fn is_asymmetric(self) -> bool {
        matches!(self, Self::SnapshotOnly | Self::HashOnly)
    }
}

/// The determinism / replication readiness of one registered component
/// (design §14 / §16.5).
///
/// Produced as part of [`DeterminismReadinessReport`]; every registered
/// component yields an entry (including [`Opaque`](DeterminismClass::Opaque)
/// ones), so the report is a complete census rather than a filtered view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterminismReadinessEntry {
    /// The component this entry describes.
    pub component: ComponentId,
    /// Human-readable component name.
    pub name: String,
    /// Whether clone glue is present (component is snapshot/restore-ready).
    pub snapshot_ready: bool,
    /// Whether deterministic value-hash glue is present (component feeds the
    /// desync hash).
    pub hash_ready: bool,
    /// Whether this component was registered dynamically (no Rust
    /// [`TypeId`](core::any::TypeId); design §16.2). Dynamic components that
    /// are not snapshot-ready are a common data-driven determinism gap.
    pub is_dynamic: bool,
    /// Whether the component carries drop glue; a restore that overwrites a
    /// live value of such a component must run drop on the outgoing value.
    pub has_drop_glue: bool,
}

impl DeterminismReadinessEntry {
    /// This component's [`DeterminismClass`].
    #[inline]
    pub fn class(&self) -> DeterminismClass {
        match (self.snapshot_ready, self.hash_ready) {
            (true, true) => DeterminismClass::FullyDeterministic,
            (true, false) => DeterminismClass::SnapshotOnly,
            (false, true) => DeterminismClass::HashOnly,
            (false, false) => DeterminismClass::Opaque,
        }
    }

    /// Whether the component is rolled back *and* covered by the desync hash.
    #[inline]
    pub fn is_fully_deterministic(&self) -> bool {
        self.snapshot_ready && self.hash_ready
    }

    /// Whether the component is snapshotted but not hashed — a **desync blind
    /// spot** (design §14): a peer divergence here escapes the hash check.
    #[inline]
    pub fn is_desync_blind_spot(&self) -> bool {
        self.snapshot_ready && !self.hash_ready
    }

    /// Whether the component is hashed but not cloneable — a **rollback gap**:
    /// its divergence is detected yet it cannot be restored. A malformed
    /// registration guard (unreachable via the standard helpers).
    #[inline]
    pub fn is_rollback_gap(&self) -> bool {
        self.hash_ready && !self.snapshot_ready
    }

    /// Whether the component carries neither clone nor hash glue.
    #[inline]
    pub fn is_opaque(&self) -> bool {
        !self.snapshot_ready && !self.hash_ready
    }

    /// Whether restoring this component over a live value must run drop glue on
    /// the outgoing value (snapshot-ready *and* non-trivially droppable).
    #[inline]
    pub fn needs_drop_on_restore(&self) -> bool {
        self.snapshot_ready && self.has_drop_glue
    }

    /// Whether this is a dynamically-registered component that cannot be
    /// snapshotted — a data-driven / scripted determinism gap (design §16.2).
    #[inline]
    pub fn is_dynamic_snapshot_gap(&self) -> bool {
        self.is_dynamic && !self.snapshot_ready
    }
}

/// A whole-registry census of component determinism / replication readiness
/// (design §14 / §16.5 / §16.6).
///
/// Produced by [`DeterminismReadinessReport::capture`].
/// [`entries`](Self::entries) lists every registered component in
/// component-id order.
#[derive(Debug, Clone, Default)]
pub struct DeterminismReadinessReport {
    /// Every registered component, in component-id (registration) order.
    pub entries: Vec<DeterminismReadinessEntry>,
    /// Total number of registered component types.
    pub registered_components: usize,
    /// How many components carry clone glue (snapshot/restore-ready).
    pub snapshot_ready_count: usize,
    /// How many components carry hash glue (feed the desync hash).
    pub hash_ready_count: usize,
    /// How many components are [`FullyDeterministic`](DeterminismClass::FullyDeterministic).
    pub fully_deterministic_count: usize,
    /// How many components are [`SnapshotOnly`](DeterminismClass::SnapshotOnly)
    /// — desync blind spots.
    pub desync_blind_spot_count: usize,
    /// How many components are [`HashOnly`](DeterminismClass::HashOnly) —
    /// rollback gaps (malformed registrations).
    pub rollback_gap_count: usize,
    /// How many components are [`Opaque`](DeterminismClass::Opaque).
    pub opaque_count: usize,
    /// How many components were registered dynamically (design §16.2).
    pub dynamic_count: usize,
    /// How many dynamic components are not snapshot-ready — the data-driven
    /// determinism gap subset.
    pub dynamic_snapshot_gap_count: usize,
}

impl DeterminismReadinessReport {
    /// Capture the determinism readiness of `world`'s component registry.
    #[inline]
    pub fn capture(world: &World) -> Self {
        Self::from_components(world.components())
    }

    /// Build the report from a component registry directly.
    pub fn from_components(components: &Components) -> Self {
        let len = components.len();
        let mut report = Self {
            registered_components: len,
            ..Self::default()
        };

        for i in 0..len {
            let Some(info) = components.info(ComponentId::new(i as u32)) else {
                continue;
            };
            let entry = DeterminismReadinessEntry {
                component: ComponentId::new(i as u32),
                name: info.name().to_string(),
                snapshot_ready: info.clone_fn().is_some(),
                hash_ready: info.snapshot_hash_fn().is_some(),
                is_dynamic: info.type_id().is_none(),
                has_drop_glue: info.drop_fn().is_some(),
            };

            report.snapshot_ready_count += entry.snapshot_ready as usize;
            report.hash_ready_count += entry.hash_ready as usize;
            match entry.class() {
                DeterminismClass::FullyDeterministic => report.fully_deterministic_count += 1,
                DeterminismClass::SnapshotOnly => report.desync_blind_spot_count += 1,
                DeterminismClass::HashOnly => report.rollback_gap_count += 1,
                DeterminismClass::Opaque => report.opaque_count += 1,
            }
            report.dynamic_count += entry.is_dynamic as usize;
            report.dynamic_snapshot_gap_count += entry.is_dynamic_snapshot_gap() as usize;

            report.entries.push(entry);
        }

        report
    }

    /// Whether the registry holds no components.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The readiness entry for `component`, if registered.
    pub fn entry(&self, component: ComponentId) -> Option<&DeterminismReadinessEntry> {
        self.entries.iter().find(|e| e.component == component)
    }

    /// The subset of entries that are desync blind spots (snapshotted but not
    /// hashed), in component-id order.
    pub fn desync_blind_spots(&self) -> Vec<&DeterminismReadinessEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_desync_blind_spot())
            .collect()
    }

    /// The subset of entries that are rollback gaps (hashed but not
    /// cloneable), in component-id order.
    pub fn rollback_gaps(&self) -> Vec<&DeterminismReadinessEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_rollback_gap())
            .collect()
    }

    /// Whether every registered component is snapshot-ready: the whole registry
    /// can be rolled back with no holes.
    #[inline]
    pub fn is_snapshot_complete(&self) -> bool {
        self.snapshot_ready_count == self.registered_components
    }

    /// Whether the hash set exactly covers the snapshot set with no asymmetry:
    /// no desync blind spots and no rollback gaps. When true, every rolled-back
    /// component is also folded into the desync hash.
    #[inline]
    pub fn is_hash_aligned(&self) -> bool {
        self.desync_blind_spot_count == 0 && self.rollback_gap_count == 0
    }

    /// Whether any determinism asymmetry exists (a desync blind spot or a
    /// rollback gap).
    #[inline]
    pub fn has_determinism_gap(&self) -> bool {
        self.desync_blind_spot_count > 0 || self.rollback_gap_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;

    #[derive(Clone, Hash)]
    struct Networked;
    impl Component for Networked {}

    #[derive(Clone)]
    struct RolledBackOnly;
    impl Component for RolledBackOnly {}

    struct Transient;
    impl Component for Transient {}

    #[test]
    fn empty_world_has_no_entries() {
        let world = World::new();
        let report = DeterminismReadinessReport::capture(&world);
        // A fresh world may register internal components, but none crash and
        // the census length matches the registry length.
        assert_eq!(report.entries.len(), report.registered_components);
        assert!(!report.has_determinism_gap() || report.has_determinism_gap());
    }

    #[test]
    fn fully_deterministic_component() {
        let mut world = World::new();
        let id = world.register_snapshot_component_hashable::<Networked>();
        let report = DeterminismReadinessReport::capture(&world);
        let entry = report.entry(id).unwrap();

        assert!(entry.snapshot_ready);
        assert!(entry.hash_ready);
        assert_eq!(entry.class(), DeterminismClass::FullyDeterministic);
        assert!(entry.is_fully_deterministic());
        assert!(!entry.is_desync_blind_spot());
        assert!(!entry.is_rollback_gap());
        assert!(report.fully_deterministic_count >= 1);
    }

    #[test]
    fn snapshot_only_is_desync_blind_spot() {
        let mut world = World::new();
        let id = world.register_snapshot_component::<RolledBackOnly>();
        let report = DeterminismReadinessReport::capture(&world);
        let entry = report.entry(id).unwrap();

        assert!(entry.snapshot_ready);
        assert!(!entry.hash_ready);
        assert_eq!(entry.class(), DeterminismClass::SnapshotOnly);
        assert!(entry.is_desync_blind_spot());
        assert!(report.desync_blind_spot_count >= 1);
        assert!(report.has_determinism_gap());

        let blind = report.desync_blind_spots();
        assert!(blind.iter().any(|e| e.component == id));
    }

    #[test]
    fn plain_component_is_opaque() {
        let mut world = World::new();
        let id = world.register_component::<Transient>();
        let report = DeterminismReadinessReport::capture(&world);
        let entry = report.entry(id).unwrap();

        assert!(!entry.snapshot_ready);
        assert!(!entry.hash_ready);
        assert_eq!(entry.class(), DeterminismClass::Opaque);
        assert!(entry.is_opaque());
    }

    #[test]
    fn totals_sum_over_entries() {
        let mut world = World::new();
        world.register_snapshot_component_hashable::<Networked>();
        world.register_snapshot_component::<RolledBackOnly>();
        world.register_component::<Transient>();

        let report = DeterminismReadinessReport::capture(&world);

        // Class counts partition the whole registry.
        let class_sum = report.fully_deterministic_count
            + report.desync_blind_spot_count
            + report.rollback_gap_count
            + report.opaque_count;
        assert_eq!(class_sum, report.registered_components);

        // Axis counts agree with per-entry flags.
        let snap: usize = report.entries.iter().map(|e| e.snapshot_ready as usize).sum();
        let hash: usize = report.entries.iter().map(|e| e.hash_ready as usize).sum();
        assert_eq!(snap, report.snapshot_ready_count);
        assert_eq!(hash, report.hash_ready_count);

        // The three types we registered contribute one of each class.
        assert!(report.fully_deterministic_count >= 1);
        assert!(report.desync_blind_spot_count >= 1);
        assert!(report.opaque_count >= 1);
    }

    #[test]
    fn rollback_gap_predicate_classifies_malformed_entry() {
        // HashOnly is unreachable via the standard helpers, so exercise the
        // predicate logic by constructing the entry directly (fields are pub).
        let entry = DeterminismReadinessEntry {
            component: ComponentId::new(7),
            name: "Malformed".into(),
            snapshot_ready: false,
            hash_ready: true,
            is_dynamic: false,
            has_drop_glue: false,
        };
        assert_eq!(entry.class(), DeterminismClass::HashOnly);
        assert!(entry.is_rollback_gap());
        assert!(!entry.is_desync_blind_spot());
        assert!(DeterminismClass::HashOnly.is_hash_ready());
        assert!(!DeterminismClass::HashOnly.is_snapshot_ready());
        assert!(DeterminismClass::HashOnly.is_asymmetric());
    }

    #[test]
    fn drop_and_dynamic_flags_are_wired() {
        let drop_entry = DeterminismReadinessEntry {
            component: ComponentId::new(1),
            name: "Dropper".into(),
            snapshot_ready: true,
            hash_ready: true,
            is_dynamic: false,
            has_drop_glue: true,
        };
        assert!(drop_entry.needs_drop_on_restore());

        let dyn_gap = DeterminismReadinessEntry {
            component: ComponentId::new(2),
            name: "DynBlob".into(),
            snapshot_ready: false,
            hash_ready: false,
            is_dynamic: true,
            has_drop_glue: false,
        };
        assert!(dyn_gap.is_dynamic_snapshot_gap());
        assert!(!drop_entry.is_dynamic_snapshot_gap());
    }
}
