//! Component lifecycle-hook coverage diagnostic (design §12 / §16.6).
//!
//! Lifecycle hooks (design §12) are the low-level, synchronous callbacks the
//! world fires as a component is structurally added, overwritten, or removed:
//! `on_add`, `on_insert`, `on_replace`, `on_remove` (see
//! [`ComponentHooks`](crate::component_hooks::ComponentHooks)). They are the
//! canonical place to **acquire** a side resource when a component appears
//! (register a GPU buffer, allocate a derived-index slot, open a handle) and to
//! **release** it when the component goes away. Getting that acquire/release
//! pairing wrong leaks the side resource silently — exactly the class of bug
//! that never shows up in a value-only inspector.
//!
//! This report maps which of the four hooks each component registers and
//! flags the asymmetries that are concrete leak smells. It reads only the
//! registry's public hook metadata ([`ComponentInfo::hooks`]), so it is
//! well-defined on an entity-less world and never runs a hook.
//!
//! # Trigger model recap (why the asymmetry flags are what they are)
//! For one structural operation the engine fires, per affected component:
//!
//! | Transition | Hooks (in order) |
//! |---|---|
//! | newly added | `on_add` then `on_insert` |
//! | value overwritten in place | `on_replace` then `on_insert` |
//! | removed (incl. despawn) | `on_replace` then `on_remove` |
//!
//! So `on_replace` is the only teardown hook that fires on **both** overwrite
//! and removal, while `on_remove` fires on removal **only**. A component that
//! acquires a resource in `on_add` must therefore release it in `on_replace`
//! (covers overwrite + removal) — or accept that an in-place overwrite leaks
//! the previous value if it only cleans up in `on_remove`.
//!
//! # What each entry reports
//! For every component carrying at least one hook:
//!
//! * Which of the four hooks are present
//!   ([`on_add`](HookCoverageEntry::on_add) … [`on_remove`](HookCoverageEntry::on_remove)).
//! * [`acquires_without_release`](HookCoverageEntry::acquires_without_release) —
//!   acquires on add with no teardown hook at all: a definite leak on despawn.
//! * [`release_misses_overwrite`](HookCoverageEntry::release_misses_overwrite) —
//!   tears down only in `on_remove`, so an in-place overwrite can leak the
//!   outgoing value (the robust teardown hook is `on_replace`).
//!
//! # Scope and determinism
//! Pure read of the component registry: `O(components)`, no entities touched,
//! deterministic. Entries list hooked components in component-id (registration)
//! order; report totals are order-independent. The asymmetry flags are
//! documented leak **heuristics**, not hard errors — a hook may legitimately
//! acquire nothing — so they are surfaced as counts and per-entry predicates
//! for a tool or CI to weigh, never enforced here.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::component::{ComponentId, Components};
use crate::world::World;

/// Which lifecycle hooks one component registers, plus leak-asymmetry flags
/// (design §12).
///
/// Produced as part of [`HookCoverageReport`]; only components with at least
/// one registered hook are emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookCoverageEntry {
    /// The component this entry describes.
    pub component: ComponentId,
    /// Human-readable component name.
    pub name: String,
    /// Whether an `on_add` hook is registered (fires when newly added).
    pub on_add: bool,
    /// Whether an `on_insert` hook is registered (fires on every write).
    pub on_insert: bool,
    /// Whether an `on_replace` hook is registered (fires before overwrite or
    /// removal — the teardown hook that covers both).
    pub on_replace: bool,
    /// Whether an `on_remove` hook is registered (fires before removal only).
    pub on_remove: bool,
}

impl HookCoverageEntry {
    /// How many of the four lifecycle hooks this component registers (1..=4).
    #[inline]
    pub fn hook_count(&self) -> usize {
        self.on_add as usize
            + self.on_insert as usize
            + self.on_replace as usize
            + self.on_remove as usize
    }

    /// Whether this component acquires on first add (`on_add` present).
    #[inline]
    pub fn has_acquire(&self) -> bool {
        self.on_add
    }

    /// Whether this component has any teardown hook (`on_replace` or
    /// `on_remove`) that could release a resource.
    #[inline]
    pub fn has_teardown(&self) -> bool {
        self.on_replace || self.on_remove
    }

    /// Leak heuristic: acquires a resource on add but registers **no** teardown
    /// hook at all, so the resource leaks when the component is removed or the
    /// entity is despawned.
    #[inline]
    pub fn acquires_without_release(&self) -> bool {
        self.on_add && !self.has_teardown()
    }

    /// Leak heuristic: tears down only in `on_remove`, which does not fire on an
    /// in-place overwrite, so overwriting the value can leak the previous one.
    /// The robust teardown hook for a value-holding resource is `on_replace`.
    #[inline]
    pub fn release_misses_overwrite(&self) -> bool {
        self.on_remove && !self.on_replace
    }
}

/// A whole-registry map of component lifecycle-hook coverage (design §12 /
/// §16.6).
///
/// Produced by [`HookCoverageReport::capture`]. [`entries`](Self::entries)
/// lists every hooked component in component-id order.
#[derive(Debug, Clone, Default)]
pub struct HookCoverageReport {
    /// Hooked components, in component-id (registration) order.
    pub entries: Vec<HookCoverageEntry>,
    /// Total number of registered component types (hooked or not).
    pub registered_components: usize,
    /// How many registered components carry at least one hook.
    pub hooked_components: usize,
    /// How many components register an `on_add` hook.
    pub on_add_count: usize,
    /// How many components register an `on_insert` hook.
    pub on_insert_count: usize,
    /// How many components register an `on_replace` hook.
    pub on_replace_count: usize,
    /// How many components register an `on_remove` hook.
    pub on_remove_count: usize,
    /// How many components match [`acquires_without_release`] — a definite
    /// leak-on-despawn smell.
    ///
    /// [`acquires_without_release`]: HookCoverageEntry::acquires_without_release
    pub acquire_without_release_count: usize,
    /// How many components match [`release_misses_overwrite`] — a possible
    /// leak-on-overwrite smell.
    ///
    /// [`release_misses_overwrite`]: HookCoverageEntry::release_misses_overwrite
    pub release_misses_overwrite_count: usize,
}

impl HookCoverageReport {
    /// Capture the lifecycle-hook coverage of `world`'s component registry.
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
            let hooks = info.hooks();
            if hooks.is_empty() {
                continue;
            }
            let entry = HookCoverageEntry {
                component: ComponentId::new(i as u32),
                name: info.name().to_string(),
                on_add: hooks.on_add().is_some(),
                on_insert: hooks.on_insert().is_some(),
                on_replace: hooks.on_replace().is_some(),
                on_remove: hooks.on_remove().is_some(),
            };
            report.on_add_count += entry.on_add as usize;
            report.on_insert_count += entry.on_insert as usize;
            report.on_replace_count += entry.on_replace as usize;
            report.on_remove_count += entry.on_remove as usize;
            report.acquire_without_release_count += entry.acquires_without_release() as usize;
            report.release_misses_overwrite_count += entry.release_misses_overwrite() as usize;
            report.entries.push(entry);
        }

        report.hooked_components = report.entries.len();
        report
    }

    /// Whether no component registers any lifecycle hook.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The coverage entry for `component`, if it carries any hook.
    pub fn entry(&self, component: ComponentId) -> Option<&HookCoverageEntry> {
        self.entries.iter().find(|e| e.component == component)
    }

    /// The hooked component registering the most lifecycle hooks, or `None`
    /// when nothing is hooked. Ties resolve to the lower component id.
    pub fn most_hooked(&self) -> Option<&HookCoverageEntry> {
        self.entries.iter().max_by(|a, b| {
            a.hook_count()
                .cmp(&b.hook_count())
                .then_with(|| b.component.index().cmp(&a.component.index()))
        })
    }

    /// The subset of entries that leak on despawn (acquire with no teardown).
    pub fn acquire_without_release(&self) -> Vec<&HookCoverageEntry> {
        self.entries
            .iter()
            .filter(|e| e.acquires_without_release())
            .collect()
    }

    /// Whether any component shows a leak-asymmetry smell (either heuristic).
    #[inline]
    pub fn has_leak_risk(&self) -> bool {
        self.acquire_without_release_count > 0 || self.release_misses_overwrite_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::component_hooks::{ComponentHooks, HookContext};
    use crate::world::World;

    fn noop(_ctx: HookContext) {}

    struct GpuHandle;
    impl Component for GpuHandle {}

    struct DerivedIndex;
    impl Component for DerivedIndex {}

    struct FullyGuarded;
    impl Component for FullyGuarded {}

    struct Plain;
    impl Component for Plain {}

    #[test]
    fn no_hooks_reports_empty() {
        let mut world = World::new();
        world.register_component::<Plain>();
        let report = HookCoverageReport::capture(&world);
        assert!(report.is_empty());
        assert_eq!(report.hooked_components, 0);
        assert!(!report.has_leak_risk());
        assert!(report.most_hooked().is_none());
    }

    #[test]
    fn acquire_without_release_is_flagged() {
        let mut world = World::new();
        // Acquires a GPU handle on add but never releases it.
        world.register_component_hooks::<GpuHandle>(ComponentHooks::new().with_on_add(noop));

        let report = HookCoverageReport::capture(&world);
        let id = world.components().id_of::<GpuHandle>().unwrap();
        let entry = report.entry(id).unwrap();

        assert!(entry.has_acquire());
        assert!(!entry.has_teardown());
        assert!(entry.acquires_without_release());
        assert_eq!(entry.hook_count(), 1);
        assert_eq!(report.acquire_without_release_count, 1);
        assert!(report.has_leak_risk());
        assert_eq!(report.acquire_without_release().len(), 1);
    }

    #[test]
    fn remove_only_release_misses_overwrite() {
        let mut world = World::new();
        // Releases on remove, but an in-place overwrite would leak the old
        // value because on_replace is absent.
        world.register_component_hooks::<DerivedIndex>(
            ComponentHooks::new().with_on_add(noop).with_on_remove(noop),
        );

        let report = HookCoverageReport::capture(&world);
        let id = world.components().id_of::<DerivedIndex>().unwrap();
        let entry = report.entry(id).unwrap();

        assert!(entry.has_teardown());
        assert!(!entry.acquires_without_release());
        assert!(entry.release_misses_overwrite());
        assert_eq!(report.release_misses_overwrite_count, 1);
        assert_eq!(report.acquire_without_release_count, 0);
    }

    #[test]
    fn on_replace_teardown_is_leak_safe() {
        let mut world = World::new();
        // Acquires on add, releases in on_replace (fires on both overwrite and
        // removal) -> neither leak heuristic trips.
        world.register_component_hooks::<FullyGuarded>(
            ComponentHooks::new()
                .with_on_add(noop)
                .with_on_replace(noop),
        );

        let report = HookCoverageReport::capture(&world);
        let id = world.components().id_of::<FullyGuarded>().unwrap();
        let entry = report.entry(id).unwrap();

        assert!(entry.has_acquire());
        assert!(entry.has_teardown());
        assert!(!entry.acquires_without_release());
        assert!(!entry.release_misses_overwrite());
        assert!(!report.has_leak_risk());
        assert_eq!(report.most_hooked().unwrap().component, id);
    }

    #[test]
    fn totals_sum_over_entries() {
        let mut world = World::new();
        world.register_component_hooks::<GpuHandle>(ComponentHooks::new().with_on_add(noop));
        world.register_component_hooks::<DerivedIndex>(
            ComponentHooks::new().with_on_add(noop).with_on_remove(noop),
        );
        world.register_component_hooks::<FullyGuarded>(
            ComponentHooks::new()
                .with_on_add(noop)
                .with_on_replace(noop),
        );
        world.register_component::<Plain>();

        let report = HookCoverageReport::capture(&world);
        assert_eq!(report.hooked_components, 3);
        assert_eq!(report.on_add_count, 3);
        assert_eq!(report.on_remove_count, 1);
        assert_eq!(report.on_replace_count, 1);

        let sum_add: usize = report.entries.iter().map(|e| e.on_add as usize).sum();
        assert_eq!(sum_add, report.on_add_count);
    }
}
