//! Generational registry of live views and their importance aggregation.
//!
//! A frame renders a *family* of views — the main camera, stereo eyes, several
//! shadow views, reflection probes, capture and offline tiles — that come and go
//! as the scene changes. [`ViewRegistry`] owns that set, handing out
//! [`ViewHandle`]s (generational handles) so a handle to a removed view can never
//! silently alias a later view that reuses its slot. Removing a view bumps its
//! slot generation, so any stale handle is detected and rejected.
//!
//! The registry also aggregates the family's [`ViewImportance`] two ways: the
//! component-wise maximum (serve the most demanding view) and a kind-weighted sum
//! (share the shared budget across views by their [`ViewKind::budget_weight`]).
//! Both feed the shared virtual-resource budget. Iteration is in slot-index
//! order, so aggregation is deterministic. No `GPU` handle lives here.

use super::{ViewHandle, ViewImportance, ViewKind};
use alloc::vec::Vec;

/// Registered state of one view.
///
/// Holds a floating-point [`ViewImportance`], so it is only [`PartialEq`], never
/// [`Eq`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewEntry {
    /// What kind of view this is.
    pub kind: ViewKind,
    /// Streaming and `LOD` importance driving the shared budget.
    pub importance: ViewImportance,
    /// Whether the view participates in this frame's budget and culling.
    pub enabled: bool,
}

impl ViewEntry {
    /// Builds an entry, enabled, with the given kind and importance.
    #[must_use]
    fn new(kind: ViewKind, importance: ViewImportance) -> Self {
        Self {
            kind,
            importance,
            enabled: true,
        }
    }
}

/// One registry slot: a generation counter plus its optional occupant.
#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u32,
    entry: Option<ViewEntry>,
}

/// Session-lifetime registry of views keyed by generational [`ViewHandle`].
#[derive(Clone, Debug, Default)]
pub struct ViewRegistry {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
}

impl ViewRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Number of live (registered, not removed) views.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether no views are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Registers a view of `kind` with that kind's default importance.
    pub fn register(&mut self, kind: ViewKind) -> ViewHandle {
        self.register_with(kind, kind.default_importance())
    }

    /// Registers a view of `kind` with an explicit `importance`.
    ///
    /// Reuses a freed slot when one is available (adopting its bumped
    /// generation), otherwise appends a new slot.
    pub fn register_with(&mut self, kind: ViewKind, importance: ViewImportance) -> ViewHandle {
        let entry = ViewEntry::new(kind, importance);
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.entry = Some(entry);
            return ViewHandle {
                index,
                generation: slot.generation,
            };
        }
        let index = self.slots.len() as u32;
        self.slots.push(Slot {
            generation: 0,
            entry: Some(entry),
        });
        ViewHandle {
            index,
            generation: 0,
        }
    }

    /// Resolves `handle` to its live slot index, validating the generation.
    #[must_use]
    fn resolve(&self, handle: ViewHandle) -> Option<usize> {
        if !handle.is_valid() {
            return None;
        }
        let index = handle.index as usize;
        let slot = self.slots.get(index)?;
        if slot.generation == handle.generation && slot.entry.is_some() {
            Some(index)
        } else {
            None
        }
    }

    /// Whether `handle` refers to a live view.
    #[must_use]
    pub fn contains(&self, handle: ViewHandle) -> bool {
        self.resolve(handle).is_some()
    }

    /// Borrows the entry for `handle`, or `None` if it is stale or removed.
    #[must_use]
    pub fn get(&self, handle: ViewHandle) -> Option<&ViewEntry> {
        let index = self.resolve(handle)?;
        self.slots[index].entry.as_ref()
    }

    /// Mutably borrows the entry for `handle`, or `None` if stale or removed.
    #[must_use]
    pub fn get_mut(&mut self, handle: ViewHandle) -> Option<&mut ViewEntry> {
        let index = self.resolve(handle)?;
        self.slots[index].entry.as_mut()
    }

    /// Replaces the importance of `handle`; returns `false` if it is stale.
    pub fn set_importance(&mut self, handle: ViewHandle, importance: ViewImportance) -> bool {
        if let Some(entry) = self.get_mut(handle) {
            entry.importance = importance;
            true
        } else {
            false
        }
    }

    /// Enables or disables `handle`; returns `false` if it is stale.
    ///
    /// A disabled view stays registered but contributes nothing to importance
    /// aggregation or culling.
    pub fn set_enabled(&mut self, handle: ViewHandle, enabled: bool) -> bool {
        if let Some(entry) = self.get_mut(handle) {
            entry.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Removes the view `handle` refers to, bumping its slot generation so the
    /// handle can never be reused.
    ///
    /// Returns `true` if a live view was removed, `false` for a stale handle.
    pub fn remove(&mut self, handle: ViewHandle) -> bool {
        let Some(index) = self.resolve(handle) else {
            return false;
        };
        let slot = &mut self.slots[index];
        slot.entry = None;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(index as u32);
        self.live -= 1;
        true
    }

    /// Iterates `(handle, entry)` for every live view in slot-index order.
    pub fn iter(&self) -> impl Iterator<Item = (ViewHandle, &ViewEntry)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.entry.as_ref().map(|entry| {
                (
                    ViewHandle {
                        index: index as u32,
                        generation: slot.generation,
                    },
                    entry,
                )
            })
        })
    }

    /// Iterates the entries of every live, enabled view in slot-index order.
    fn enabled_entries(&self) -> impl Iterator<Item = &ViewEntry> {
        self.slots
            .iter()
            .filter_map(|slot| slot.entry.as_ref())
            .filter(|entry| entry.enabled)
    }

    /// Component-wise maximum importance across all live, enabled views.
    ///
    /// This drives the shared budget when it should satisfy the most demanding
    /// view. With no enabled views the result is [`ViewImportance::ZERO`].
    #[must_use]
    pub fn aggregate_importance(&self) -> ViewImportance {
        self.enabled_entries()
            .fold(ViewImportance::ZERO, |acc, entry| {
                acc.combine_max(entry.importance)
            })
    }

    /// Kind-weighted sum of importance across all live, enabled views.
    ///
    /// Each view's importance is scaled by its [`ViewKind::budget_weight`] and
    /// accumulated, biasing the shared budget toward primary views. With no
    /// enabled views the result is [`ViewImportance::ZERO`].
    #[must_use]
    pub fn weighted_importance(&self) -> ViewImportance {
        self.enabled_entries()
            .fold(ViewImportance::ZERO, |acc, entry| {
                acc.combine_add(entry.importance.scaled(entry.kind.budget_weight()))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get_round_trips() {
        let mut reg = ViewRegistry::new();
        let h = reg.register(ViewKind::Main);
        assert_eq!(reg.len(), 1);
        let entry = reg.get(h).unwrap();
        assert_eq!(entry.kind, ViewKind::Main);
        assert_eq!(entry.importance, ViewImportance::uniform(1.0));
        assert!(entry.enabled);
    }

    #[test]
    fn stale_handle_is_rejected_after_removal() {
        let mut reg = ViewRegistry::new();
        let h = reg.register(ViewKind::Shadow);
        assert!(reg.remove(h));
        assert_eq!(reg.len(), 0);
        // The old handle no longer resolves.
        assert!(reg.get(h).is_none());
        assert!(!reg.contains(h));
        // Removing an already-stale handle is a no-op false.
        assert!(!reg.remove(h));
    }

    #[test]
    fn reused_slot_gets_a_fresh_generation() {
        let mut reg = ViewRegistry::new();
        let old = reg.register(ViewKind::Shadow);
        reg.remove(old);
        let new = reg.register(ViewKind::Reflection);
        // Same slot index is reused, but the generation advanced.
        assert_eq!(old.index, new.index);
        assert_ne!(old.generation, new.generation);
        // The stale handle must not alias the new view.
        assert!(reg.get(old).is_none());
        assert_eq!(reg.get(new).unwrap().kind, ViewKind::Reflection);
    }

    #[test]
    fn default_handle_never_resolves() {
        let reg = ViewRegistry::new();
        assert!(reg.get(ViewHandle::INVALID).is_none());
    }

    #[test]
    fn aggregate_takes_component_peaks_across_views() {
        let mut reg = ViewRegistry::new();
        reg.register_with(ViewKind::Main, ViewImportance::new(0.2, 0.9, 0.1));
        reg.register_with(ViewKind::Shadow, ViewImportance::new(0.8, 0.3, 0.5));
        assert_eq!(
            reg.aggregate_importance(),
            ViewImportance::new(0.8, 0.9, 0.5)
        );
    }

    #[test]
    fn disabled_views_are_excluded_from_aggregation() {
        let mut reg = ViewRegistry::new();
        let a = reg.register_with(ViewKind::Main, ViewImportance::uniform(1.0));
        reg.register_with(ViewKind::Shadow, ViewImportance::uniform(0.25));
        assert!(reg.set_enabled(a, false));
        // With the main view disabled, only the shadow view's 0.25 remains.
        assert_eq!(reg.aggregate_importance(), ViewImportance::uniform(0.25));
    }

    #[test]
    fn weighted_importance_scales_by_kind_weight() {
        let mut reg = ViewRegistry::new();
        // Main weight 1.0, SceneCapture weight 0.25.
        reg.register_with(ViewKind::Main, ViewImportance::uniform(1.0));
        reg.register_with(ViewKind::SceneCapture, ViewImportance::uniform(1.0));
        // 1.0 * 1.0 + 1.0 * 0.25 = 1.25 per component.
        assert_eq!(reg.weighted_importance(), ViewImportance::uniform(1.25));
    }

    #[test]
    fn empty_registry_aggregates_to_zero() {
        let reg = ViewRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.aggregate_importance(), ViewImportance::ZERO);
        assert_eq!(reg.weighted_importance(), ViewImportance::ZERO);
    }

    #[test]
    fn iter_is_deterministic_in_slot_order() {
        let mut reg = ViewRegistry::new();
        let h0 = reg.register(ViewKind::Main);
        let h1 = reg.register(ViewKind::Shadow);
        let h2 = reg.register(ViewKind::Reflection);
        reg.remove(h1);
        let kinds: Vec<ViewKind> = reg.iter().map(|(_, e)| e.kind).collect();
        assert_eq!(kinds, alloc::vec![ViewKind::Main, ViewKind::Reflection]);
        // Surviving handles still resolve; the removed one does not.
        assert!(reg.contains(h0));
        assert!(reg.contains(h2));
        assert!(!reg.contains(h1));
    }

    #[test]
    fn set_importance_and_enabled_reject_stale_handles() {
        let mut reg = ViewRegistry::new();
        let h = reg.register(ViewKind::Main);
        reg.remove(h);
        assert!(!reg.set_importance(h, ViewImportance::uniform(0.5)));
        assert!(!reg.set_enabled(h, false));
    }
}
