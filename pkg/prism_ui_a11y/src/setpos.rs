//! Set position computation (`posinset` / `setsize`).
//!
//! Assistive technologies announce a set item as "N of M" — "item 2 of 5",
//! "tab 3 of 4" — so the user knows where they are within a collection. The
//! web accessibility model derives this from an item's same-role siblings
//! under an owning container, exposed as `aria-posinset` and `aria-setsize`.
//!
//! Loom's [`A11yTree`] is a flat, document-ordered collection without explicit
//! parent links, so this module infers a set from a **maximal contiguous run
//! of same-role items**, the grouping assistive technologies fall back to when
//! no explicit container is present. Only the recognised item roles
//! ([`SetItem` roles](is_set_item_role): list item, menu item, tab, radio)
//! participate; every other role returns [`None`].
//!
//! Hidden nodes are removed from the accessibility tree, so they neither count
//! toward a set nor break a run: two list items separated only by a hidden
//! node are adjacent members of the same set. Disabled items remain in the
//! tree and are counted (a screen reader still says "dimmed, 2 of 5").
//!
//! # Example
//!
//! ```
//! use prism_ui::Key;
//! use prism_ui_a11y::{A11yNode, A11yTree, Role, set_position};
//!
//! let mut tree = A11yTree::new();
//! for i in 1..=3 {
//!     tree.insert(A11yNode::builder(Key::Int(i), Role::ListItem).build());
//! }
//!
//! let pos = set_position(&tree, &Key::Int(2)).expect("list item is a set member");
//! assert_eq!(pos.pos_in_set, 2);
//! assert_eq!(pos.set_size, 3);
//! assert_eq!(pos.phrase(), "2 of 3");
//! ```

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use prism_ui::Key;

use crate::role::Role;
use crate::tree::A11yTree;

/// The position of an item within its inferred set.
///
/// Both fields are one-based and satisfy `1 <= pos_in_set <= set_size`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SetPosition {
    /// The item's one-based position within the set.
    pub pos_in_set: usize,
    /// The total number of items in the set.
    pub set_size: usize,
}

impl SetPosition {
    /// Formats the spoken position phrase, e.g. `"2 of 5"`.
    #[must_use]
    pub fn phrase(&self) -> String {
        let mut out = self.pos_in_set.to_string();
        out.push_str(" of ");
        out.push_str(&self.set_size.to_string());
        out
    }
}

/// Returns `true` when `role` denotes an item that participates in set
/// position computation: a list item, menu item, tab, or radio button.
#[must_use]
pub fn is_set_item_role(role: &Role) -> bool {
    matches!(
        role,
        Role::ListItem | Role::MenuItem | Role::Tab | Role::Radio
    )
}

/// Computes the [`SetPosition`] of the node identified by `key`, or `None` when
/// the node is absent, hidden, or not a [set item role](is_set_item_role).
///
/// The set is the maximal contiguous run of same-role items surrounding the
/// node in document order, ignoring hidden nodes.
#[must_use]
pub fn set_position(tree: &A11yTree, key: &Key) -> Option<SetPosition> {
    let node = tree.get(key)?;
    if node.state.hidden || !is_set_item_role(&node.role) {
        return None;
    }
    let role = &node.role;

    // The visible (non-hidden) document-ordered sequence; hidden nodes are
    // absent from the accessibility tree and so neither count nor split a run.
    let visible = tree
        .iter()
        .filter(|n| !n.state.hidden)
        .collect::<Vec<_>>();

    let here = visible.iter().position(|n| &n.key == key)?;

    // Expand left and right across the maximal run of the same role.
    let mut lo = here;
    while lo > 0 && &visible[lo - 1].role == role {
        lo -= 1;
    }
    let mut hi = here;
    while hi + 1 < visible.len() && &visible[hi + 1].role == role {
        hi += 1;
    }

    Some(SetPosition {
        pos_in_set: here - lo + 1,
        set_size: hi - lo + 1,
    })
}

/// Composes the screen-reader description of `key` and appends its set position
/// when the node is a set item, e.g. `"radio button, Male, not checked, 2 of 3"`.
///
/// For non-item nodes this is identical to
/// [`screen_reader_text`](crate::screen_reader_text).
#[must_use]
pub fn screen_reader_text_with_position(tree: &A11yTree, key: &Key) -> String {
    let mut text = crate::sr::screen_reader_text(tree, key);
    if let Some(pos) = set_position(tree, key)
        && !text.is_empty()
    {
        text.push_str(", ");
        text.push_str(&pos.phrase());
    }
    text
}

#[cfg(test)]
mod tests {
    use super::{is_set_item_role, screen_reader_text_with_position, set_position, SetPosition};
    use crate::label::Label;
    use crate::node::A11yNode;
    use crate::role::Role;
    use crate::state::AriaState;
    use crate::tree::A11yTree;
    use alloc::vec::Vec;
    use prism_ui::Key;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    fn role_for(tag: u64) -> Role {
        match tag {
            0 => Role::ListItem,
            1 => Role::MenuItem,
            2 => Role::Tab,
            3 => Role::Radio,
            4 => Role::Button,
            5 => Role::Group,
            _ => Role::Heading { level: 1 },
        }
    }

    /// Independent oracle: build the visible sequence, find the key, and expand
    /// the same-role run by hand.
    fn oracle(nodes: &[(Key, Role, bool)], key: &Key) -> Option<SetPosition> {
        let target = nodes.iter().find(|(k, _, _)| k == key)?;
        if target.2 || !is_set_item_role(&target.1) {
            return None;
        }
        let role = &target.1;
        let visible: Vec<&(Key, Role, bool)> = nodes.iter().filter(|(_, _, h)| !*h).collect();
        let here = visible.iter().position(|(k, _, _)| k == key)?;
        let mut lo = here;
        while lo > 0 && &visible[lo - 1].1 == role {
            lo -= 1;
        }
        let mut hi = here;
        while hi + 1 < visible.len() && &visible[hi + 1].1 == role {
            hi += 1;
        }
        Some(SetPosition {
            pos_in_set: here - lo + 1,
            set_size: hi - lo + 1,
        })
    }

    #[test]
    fn matches_independent_oracle_over_random_trees() {
        let mut rng = SplitMix64(0xA11A_0000_0000_0001);
        for _ in 0..3_000 {
            let count = 1 + rng.below(10) as i64;
            let mut spec: Vec<(Key, Role, bool)> = Vec::new();
            let mut tree = A11yTree::new();
            for i in 0..count {
                let role = role_for(rng.below(7));
                let hidden = rng.below(4) == 0;
                let key = Key::Int(i);
                spec.push((key.clone(), role.clone(), hidden));
                tree.insert(
                    A11yNode::builder(key, role)
                        .state(AriaState::new().hidden(hidden))
                        .build(),
                );
            }
            for i in 0..count {
                let key = Key::Int(i);
                assert_eq!(
                    set_position(&tree, &key),
                    oracle(&spec, &key),
                    "mismatch at {key:?} spec={spec:?}"
                );
            }
        }
    }

    #[test]
    fn simple_run_positions() {
        let mut tree = A11yTree::new();
        for i in 1..=4 {
            tree.insert(A11yNode::builder(Key::Int(i), Role::Tab).build());
        }
        assert_eq!(
            set_position(&tree, &Key::Int(1)),
            Some(SetPosition { pos_in_set: 1, set_size: 4 })
        );
        assert_eq!(
            set_position(&tree, &Key::Int(4)),
            Some(SetPosition { pos_in_set: 4, set_size: 4 })
        );
    }

    #[test]
    fn different_role_breaks_the_run() {
        let mut tree = A11yTree::new();
        tree.insert(A11yNode::builder(Key::Int(1), Role::MenuItem).build());
        tree.insert(A11yNode::builder(Key::Int(2), Role::MenuItem).build());
        tree.insert(A11yNode::builder(Key::Int(3), Role::Group).build());
        tree.insert(A11yNode::builder(Key::Int(4), Role::MenuItem).build());

        assert_eq!(
            set_position(&tree, &Key::Int(2)),
            Some(SetPosition { pos_in_set: 2, set_size: 2 })
        );
        // The group splits the run, so the trailing item starts a new set.
        assert_eq!(
            set_position(&tree, &Key::Int(4)),
            Some(SetPosition { pos_in_set: 1, set_size: 1 })
        );
        // The group itself is not a set item.
        assert_eq!(set_position(&tree, &Key::Int(3)), None);
    }

    #[test]
    fn hidden_item_is_skipped_not_a_separator() {
        let mut tree = A11yTree::new();
        tree.insert(A11yNode::builder(Key::Int(1), Role::Radio).build());
        tree.insert(
            A11yNode::builder(Key::Int(2), Role::Radio)
                .state(AriaState::new().hidden(true))
                .build(),
        );
        tree.insert(A11yNode::builder(Key::Int(3), Role::Radio).build());

        // The hidden radio is removed: 1 and 3 are adjacent, a set of two.
        assert_eq!(
            set_position(&tree, &Key::Int(1)),
            Some(SetPosition { pos_in_set: 1, set_size: 2 })
        );
        assert_eq!(
            set_position(&tree, &Key::Int(3)),
            Some(SetPosition { pos_in_set: 2, set_size: 2 })
        );
        // The hidden item itself has no position.
        assert_eq!(set_position(&tree, &Key::Int(2)), None);
    }

    #[test]
    fn disabled_item_still_counts() {
        let mut tree = A11yTree::new();
        tree.insert(A11yNode::builder(Key::Int(1), Role::ListItem).build());
        tree.insert(
            A11yNode::builder(Key::Int(2), Role::ListItem)
                .state(AriaState::new().disabled(true))
                .build(),
        );
        tree.insert(A11yNode::builder(Key::Int(3), Role::ListItem).build());
        assert_eq!(
            set_position(&tree, &Key::Int(3)),
            Some(SetPosition { pos_in_set: 3, set_size: 3 })
        );
    }

    #[test]
    fn non_item_and_missing_are_none() {
        let mut tree = A11yTree::new();
        tree.insert(A11yNode::builder(Key::Int(1), Role::Button).build());
        assert_eq!(set_position(&tree, &Key::Int(1)), None);
        assert_eq!(set_position(&tree, &Key::Int(99)), None);
    }

    #[test]
    fn phrase_formats_n_of_m() {
        let pos = SetPosition { pos_in_set: 2, set_size: 5 };
        assert_eq!(pos.phrase(), "2 of 5");
    }

    #[test]
    fn text_with_position_appends_phrase() {
        let mut tree = A11yTree::new();
        tree.insert(
            A11yNode::builder(Key::Int(1), Role::Radio)
                .label(Label::text("Male"))
                .state(AriaState::new().checked(false))
                .build(),
        );
        tree.insert(
            A11yNode::builder(Key::Int(2), Role::Radio)
                .label(Label::text("Female"))
                .state(AriaState::new().checked(true))
                .build(),
        );
        assert_eq!(
            screen_reader_text_with_position(&tree, &Key::Int(1)),
            "radio button, Male, not checked, 1 of 2"
        );
        // A non-item node gets the plain description, no position suffix.
        tree.insert(
            A11yNode::builder(Key::Int(3), Role::Button)
                .label(Label::text("Go"))
                .build(),
        );
        assert_eq!(
            screen_reader_text_with_position(&tree, &Key::Int(3)),
            "button, Go"
        );
    }
}
