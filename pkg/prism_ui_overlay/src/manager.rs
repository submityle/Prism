//! The overlay stack and its render composition.

use alloc::format;
use alloc::vec::Vec;

use prism_ui::Element;

use crate::id::OverlayId;
use crate::kind::OverlayKind;

/// Stable class/key prefix for the composition root produced by
/// [`OverlayManager::render`].
const ROOT_KEY: &str = "prism-overlay-root";
/// Stable class/key for the portal layer that hosts every overlay.
const PORTAL_KEY: &str = "prism-overlay-portal";
/// Class applied to each overlay's wrapping layer.
const LAYER_CLASS: &str = "prism-overlay-layer";
/// Class applied to a modal's backdrop/scrim.
const BACKDROP_CLASS: &str = "prism-overlay-backdrop";

/// A single overlay held by an [`OverlayManager`].
#[derive(Clone, Debug)]
pub struct OverlayEntry {
    id: OverlayId,
    kind: OverlayKind,
    element: Element,
    dismissible: bool,
    seq: u64,
}

impl OverlayEntry {
    /// This overlay's stable id.
    #[must_use]
    pub fn id(&self) -> OverlayId {
        self.id
    }

    /// This overlay's kind.
    #[must_use]
    pub fn kind(&self) -> OverlayKind {
        self.kind
    }

    /// The content element supplied when this overlay was pushed.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }

    /// Whether this overlay may be dismissed by escape/scrim policies.
    #[must_use]
    pub fn is_dismissible(&self) -> bool {
        self.dismissible
    }
}

/// Manages a stack of overlays and composes them into a portal layer.
///
/// Overlays are stored in insertion order but always observed in deterministic
/// **z-order**: ascending [`OverlayKind::z_priority`], ties broken by insertion
/// order. The last entry in z-order is the top-most, as returned by
/// [`top`](OverlayManager::top).
#[derive(Clone, Debug, Default)]
pub struct OverlayManager {
    entries: Vec<OverlayEntry>,
    counter: u64,
}

impl OverlayManager {
    /// Creates an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of live overlays.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no live overlays.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pushes a dismissible overlay of `kind` with `element` as its content and
    /// returns its freshly allocated id.
    pub fn push(&mut self, kind: OverlayKind, element: Element) -> OverlayId {
        self.push_with(kind, element, true)
    }

    /// Pushes an overlay, explicitly choosing whether it is dismissible.
    ///
    /// Non-dismissible overlays are skipped by [`on_escape`](Self::on_escape)
    /// and [`on_scrim_click`](Self::on_scrim_click) but can still be removed by
    /// an explicit [`dismiss`](Self::dismiss).
    pub fn push_with(
        &mut self,
        kind: OverlayKind,
        element: Element,
        dismissible: bool,
    ) -> OverlayId {
        let id = OverlayId(self.counter);
        let seq = self.counter;
        self.counter += 1;
        self.entries.push(OverlayEntry {
            id,
            kind,
            element,
            dismissible,
            seq,
        });
        id
    }

    /// Removes the overlay with `id`, returning whether it existed.
    pub fn dismiss(&mut self, id: OverlayId) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.id == id) {
            self.entries.remove(pos);
            true
        } else {
            false
        }
    }

    /// Removes the top-most overlay (highest z-order), returning its id.
    pub fn dismiss_top(&mut self) -> Option<OverlayId> {
        let top_id = self.top().map(OverlayEntry::id)?;
        self.dismiss(top_id);
        Some(top_id)
    }

    /// Sets whether the overlay with `id` is dismissible, returning whether it
    /// existed.
    pub fn set_dismissible(&mut self, id: OverlayId, dismissible: bool) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.dismissible = dismissible;
            true
        } else {
            false
        }
    }

    /// Borrows the entry with `id`, if present.
    #[must_use]
    pub fn get(&self, id: OverlayId) -> Option<&OverlayEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// The top-most overlay in z-order, if any.
    #[must_use]
    pub fn top(&self) -> Option<&OverlayEntry> {
        self.z_order_indices().last().map(|&i| &self.entries[i])
    }

    /// Iterates overlays in ascending z-order (bottom-most first, top-most
    /// last). This is the order in which they are composited.
    #[must_use]
    pub fn iter(&self) -> OverlayIter<'_> {
        OverlayIter {
            manager: self,
            order: self.z_order_indices(),
            pos: 0,
        }
    }

    /// Dismisses the top-most *dismissible* overlay, returning its id.
    ///
    /// This is the handler to wire to the Escape key: it scans from the top of
    /// the stack downward and removes the first overlay that opted into
    /// dismissal, leaving pinned overlays untouched.
    pub fn on_escape(&mut self) -> Option<OverlayId> {
        let order = self.z_order_indices();
        for &i in order.iter().rev() {
            if self.entries[i].dismissible {
                let id = self.entries[i].id;
                self.entries.remove(i);
                return Some(id);
            }
        }
        None
    }

    /// Dismisses the top-most modal when a scrim/backdrop is clicked, provided
    /// that modal is dismissible.
    ///
    /// If the top-most modal is pinned (non-dismissible), nothing happens and
    /// `None` is returned: a scrim click must not fall through to a modal
    /// beneath it.
    pub fn on_scrim_click(&mut self) -> Option<OverlayId> {
        let order = self.z_order_indices();
        for &i in order.iter().rev() {
            if self.entries[i].kind.is_modal() {
                if self.entries[i].dismissible {
                    let id = self.entries[i].id;
                    self.entries.remove(i);
                    return Some(id);
                }
                return None;
            }
        }
        None
    }

    /// Composes `base` with a portal layer that hosts every overlay in z-order.
    ///
    /// The returned tree is a root box whose first child is `base` and whose
    /// second child is the portal. Within the portal, each modal contributes a
    /// backdrop element immediately before its content layer, so the backdrop
    /// always paints beneath the modal it belongs to. Every generated node is
    /// given a stable `key_str` derived from the overlay id, so `prism_ui`'s
    /// keyed reconciler reuses nodes across frames.
    #[must_use]
    pub fn render(&self, base: Element) -> Element {
        Element::box_()
            .class(ROOT_KEY)
            .key_str(ROOT_KEY)
            .child(base)
            .child(self.portal())
    }

    /// Builds the portal layer holding every overlay in z-order.
    fn portal(&self) -> Element {
        let mut layers: Vec<Element> = Vec::new();
        for entry in self.iter() {
            if entry.kind.has_backdrop() {
                layers.push(backdrop(entry.id));
            }
            layers.push(layer(entry));
        }
        Element::box_()
            .class(PORTAL_KEY)
            .key_str(PORTAL_KEY)
            .children(layers)
    }

    /// Returns entry indices sorted into ascending z-order.
    ///
    /// Keys `(z_priority, seq)` are unique, so an unstable sort is fully
    /// deterministic here while remaining `alloc`-only.
    fn z_order_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.entries.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            let ea = &self.entries[a];
            let eb = &self.entries[b];
            ea.kind
                .z_priority()
                .cmp(&eb.kind.z_priority())
                .then(ea.seq.cmp(&eb.seq))
        });
        indices
    }
}

/// Builds a modal's backdrop element with a stable key.
fn backdrop(id: OverlayId) -> Element {
    Element::box_()
        .class(BACKDROP_CLASS)
        .key_str(format!("{BACKDROP_CLASS}-{id}"))
}

/// Wraps an overlay's content in a keyed layer box.
fn layer(entry: &OverlayEntry) -> Element {
    Element::box_()
        .class(LAYER_CLASS)
        .key_str(format!("{LAYER_CLASS}-{}", entry.id))
        .child(entry.element.clone())
}

/// Iterator over overlays in ascending z-order, returned by
/// [`OverlayManager::iter`].
#[derive(Debug)]
pub struct OverlayIter<'a> {
    manager: &'a OverlayManager,
    order: Vec<usize>,
    pos: usize,
}

impl<'a> Iterator for OverlayIter<'a> {
    type Item = &'a OverlayEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let &index = self.order.get(self.pos)?;
        self.pos += 1;
        Some(&self.manager.entries[index])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.order.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for OverlayIter<'_> {}

#[cfg(test)]
mod tests {
    use super::{OverlayManager, BACKDROP_CLASS, LAYER_CLASS};
    use crate::id::OverlayId;
    use crate::kind::OverlayKind;
    use alloc::vec::Vec;
    use prism_ui::Element;

    fn ids_in_z_order(m: &OverlayManager) -> Vec<u64> {
        m.iter().map(|e| e.id().to_raw()).collect()
    }

    #[test]
    fn z_order_sorts_by_kind_then_insertion() {
        let mut m = OverlayManager::new();
        let toast = m.push(OverlayKind::Toast, Element::text("t"));
        let modal = m.push(OverlayKind::Modal, Element::text("m"));
        let tip = m.push(OverlayKind::Tooltip, Element::text("tip"));
        let modal2 = m.push(OverlayKind::Modal, Element::text("m2"));

        // Modals first (ascending priority), insertion order within a kind,
        // then tooltip, then toast on top.
        assert_eq!(
            ids_in_z_order(&m),
            [modal, modal2, tip, toast].map(OverlayId::to_raw)
        );
        assert_eq!(m.top().unwrap().id(), toast);
    }

    #[test]
    fn dismiss_by_id_and_top() {
        let mut m = OverlayManager::new();
        let a = m.push(OverlayKind::Popover, Element::text("a"));
        let b = m.push(OverlayKind::Toast, Element::text("b"));
        assert_eq!(m.len(), 2);

        assert!(m.dismiss(a));
        assert!(!m.dismiss(a));
        assert_eq!(m.len(), 1);

        assert_eq!(m.dismiss_top(), Some(b));
        assert!(m.is_empty());
        assert_eq!(m.dismiss_top(), None);
    }

    #[test]
    fn render_puts_backdrop_before_modal_content() {
        let mut m = OverlayManager::new();
        m.push(OverlayKind::Modal, Element::text("dialog"));

        let view = m.render(Element::box_());
        assert_eq!(view.child_elements().len(), 2);
        let portal = &view.child_elements()[1];
        let layers = portal.child_elements();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].class_names(), [BACKDROP_CLASS]);
        assert_eq!(layers[1].class_names(), [LAYER_CLASS]);
    }

    #[test]
    fn render_has_no_backdrop_without_modal() {
        let mut m = OverlayManager::new();
        m.push(OverlayKind::Tooltip, Element::text("tip"));
        m.push(OverlayKind::Toast, Element::text("toast"));

        let view = m.render(Element::box_());
        let portal = &view.child_elements()[1];
        let layers = portal.child_elements();
        assert_eq!(layers.len(), 2);
        for layer in layers {
            assert_eq!(layer.class_names(), [LAYER_CLASS]);
        }
    }

    #[test]
    fn scrim_click_only_dismisses_dismissible_top_modal() {
        let mut m = OverlayManager::new();
        let pinned = m.push_with(OverlayKind::Modal, Element::text("pinned"), false);
        assert_eq!(m.on_scrim_click(), None);
        assert!(m.get(pinned).is_some());

        let open = m.push(OverlayKind::Modal, Element::text("open"));
        assert_eq!(m.on_scrim_click(), Some(open));
        // The pinned modal beneath is untouched.
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn escape_skips_pinned_overlays() {
        let mut m = OverlayManager::new();
        let pinned_toast = m.push_with(OverlayKind::Toast, Element::text("sticky"), false);
        let modal = m.push(OverlayKind::Modal, Element::text("m"));
        // Toast is top-most but pinned, so escape falls through to the modal.
        assert_eq!(m.on_escape(), Some(modal));
        assert_eq!(m.on_escape(), None);
        assert!(m.get(pinned_toast).is_some());
    }
}
