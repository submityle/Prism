use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub enum SelectionKind {
    Entity(u128),
    Asset(u128),
    SubObject {
        owner: u128,
        domain: String,
        index: u32,
    },
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SelectionItem {
    kind: SelectionKind,
}

impl SelectionItem {
    #[must_use]
    pub const fn entity(id: u128) -> Self {
        Self {
            kind: SelectionKind::Entity(id),
        }
    }

    #[must_use]
    pub const fn asset(id: u128) -> Self {
        Self {
            kind: SelectionKind::Asset(id),
        }
    }

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

    #[must_use]
    pub const fn kind(&self) -> &SelectionKind {
        &self.kind
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Selection {
    items: Vec<SelectionItem>,
    membership: BTreeSet<SelectionItem>,
    primary: Option<SelectionItem>,
}

impl Selection {
    #[must_use]
    pub fn items(&self) -> &[SelectionItem] {
        &self.items
    }

    #[must_use]
    pub fn primary(&self) -> Option<&SelectionItem> {
        self.primary.as_ref()
    }

    pub fn clear(&mut self) -> bool {
        if self.items.is_empty() {
            return false;
        }
        self.items.clear();
        self.membership.clear();
        self.primary = None;
        true
    }

    pub fn replace(&mut self, item: SelectionItem) {
        self.clear();
        self.membership.insert(item.clone());
        self.items.push(item.clone());
        self.primary = Some(item);
    }

    pub fn add(&mut self, item: SelectionItem) -> bool {
        if !self.membership.insert(item.clone()) {
            self.primary = Some(item);
            return false;
        }
        self.items.push(item.clone());
        self.primary = Some(item);
        true
    }

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
