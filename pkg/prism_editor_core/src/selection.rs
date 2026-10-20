use alloc::{collections::BTreeSet, string::String, vec::Vec};

/// Kind of object that can be selected by the editor.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum SelectionKind {
    /// A runtime or document entity.
    Entity(u128),
    /// An imported or authored asset.
    Asset(u128),
    /// A named sub-object within another selected object.
    SubObject {
        /// Identity of the owning object.
        owner: u128,
        /// Domain that defines the sub-object.
        domain: String,
        /// Stable index within the domain.
        index: u32,
    },
}

/// Stable selectable item stored in a [`Selection`].
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SelectionItem {
    kind: SelectionKind,
}

impl SelectionItem {
    /// Creates an entity selection item.
    #[must_use]
    pub const fn entity(id: u128) -> Self {
        Self {
            kind: SelectionKind::Entity(id),
        }
    }

    /// Creates an asset selection item.
    #[must_use]
    pub const fn asset(id: u128) -> Self {
        Self {
            kind: SelectionKind::Asset(id),
        }
    }

    /// Creates a sub-object selection item.
    #[must_use]
    pub fn sub_object(owner: u128, domain: impl Into<String>, index: u32) -> Self {
        Self {
            kind: SelectionKind::SubObject {
                owner,
                domain: domain.into(),
                index,
            },
        }
    }

    /// Returns the typed selection identity.
    #[must_use]
    pub const fn kind(&self) -> &SelectionKind {
        &self.kind
    }
}

/// Ordered, duplicate-free selection with an explicit primary item.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Selection {
    items: Vec<SelectionItem>,
    membership: BTreeSet<SelectionItem>,
    primary: Option<SelectionItem>,
}

impl Selection {
    /// Returns selected items in insertion order.
    #[must_use]
    pub fn items(&self) -> &[SelectionItem] {
        &self.items
    }

    /// Returns the current primary item, if any.
    #[must_use]
    pub fn primary(&self) -> Option<&SelectionItem> {
        self.primary.as_ref()
    }

    /// Clears the selection and reports whether it changed.
    pub fn clear(&mut self) -> bool {
        if self.items.is_empty() {
            return false;
        }
        self.items.clear();
        self.membership.clear();
        self.primary = None;
        true
    }

    /// Replaces the selection with one primary item.
    pub fn replace(&mut self, item: SelectionItem) {
        self.clear();
        self.membership.insert(item.clone());
        self.items.push(item.clone());
        self.primary = Some(item);
    }

    /// Adds an item and makes it primary; returns whether it was new.
    pub fn add(&mut self, item: SelectionItem) -> bool {
        if !self.membership.insert(item.clone()) {
            self.primary = Some(item);
            return false;
        }
        self.items.push(item.clone());
        self.primary = Some(item);
        true
    }

    /// Removes an item and promotes the last remaining item when needed.
    pub fn remove(&mut self, item: &SelectionItem) -> bool {
        if !self.membership.remove(item) {
            return false;
        }
        self.items.retain(|entry| entry != item);
        if self.primary.as_ref() == Some(item) {
            self.primary = self.items.last().cloned();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_ordered_deduplicated_and_has_a_primary() {
        let mut selection = Selection::default();
        assert!(selection.add(SelectionItem::entity(2)));
        assert!(selection.add(SelectionItem::entity(1)));
        assert!(!selection.add(SelectionItem::entity(1)));
        assert_eq!(selection.items().len(), 2);
        assert_eq!(selection.primary(), Some(&SelectionItem::entity(1)));
        selection.remove(&SelectionItem::entity(1));
        assert_eq!(selection.primary(), Some(&SelectionItem::entity(2)));
    }
}
