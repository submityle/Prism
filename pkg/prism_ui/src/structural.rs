//! Structural view helpers: declarative conditionals and keyed lists.
//!
//! These are small, allocation-light functions that build `[Vec]<[Element]>`
//! fragments, so they drop straight into [`Element::children`]. They mirror the
//! control-flow primitives of modern declarative UIs:
//!
//! * [`show`] / [`show_or`] — conditional rendering (`SolidJS`'s `<Show>`).
//! * [`for_each`] — map a collection to **keyed** children so the reconciler
//!   reuses nodes across reorders (`SolidJS`'s `<For>`).
//!
//! Returning a fragment (`Vec<Element>`) rather than a single node lets a
//! conditional collapse to *nothing* without leaving a placeholder, while still
//! composing with ordinary children:
//!
//! ```
//! use prism_ui::{Element, structural};
//!
//! let logged_in = true;
//! let view = Element::box_().children(
//!     structural::show(logged_in, || Element::text("welcome")),
//! );
//! assert_eq!(view.child_elements().len(), 1);
//! ```

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::element::Element;

/// A value that can be attached to an [`Element`] as its reconciliation key.
///
/// Implemented for the integer and string key kinds so [`for_each`] can accept
/// either without the caller spelling out [`Element::key_int`] /
/// [`Element::key_str`].
pub trait IntoElementKey {
    /// Attaches `self` as the key of `element`.
    fn apply_key(self, element: Element) -> Element;
}

impl IntoElementKey for i64 {
    fn apply_key(self, element: Element) -> Element {
        element.key_int(self)
    }
}

impl IntoElementKey for i32 {
    fn apply_key(self, element: Element) -> Element {
        element.key_int(i64::from(self))
    }
}

impl IntoElementKey for usize {
    fn apply_key(self, element: Element) -> Element {
        // A `usize` index is a common key source; clamp defensively into the
        // `i64` key space (indices never approach `i64::MAX` in practice).
        element.key_int(i64::try_from(self).unwrap_or(i64::MAX))
    }
}

impl IntoElementKey for String {
    fn apply_key(self, element: Element) -> Element {
        element.key_str(self)
    }
}

impl IntoElementKey for &str {
    fn apply_key(self, element: Element) -> Element {
        element.key_str(self)
    }
}

/// Renders `then` only when `cond` is `true`, otherwise nothing.
///
/// The result is a fragment: one child when `cond` holds, zero children when it
/// does not.
pub fn show<F>(cond: bool, then: F) -> Vec<Element>
where
    F: FnOnce() -> Element,
{
    if cond {
        vec![then()]
    } else {
        Vec::new()
    }
}

/// Renders `then` when `cond` is `true`, else `otherwise` — a declarative
/// if/else that always yields exactly one child.
pub fn show_or<T, E>(cond: bool, then: T, otherwise: E) -> Vec<Element>
where
    T: FnOnce() -> Element,
    E: FnOnce() -> Element,
{
    if cond {
        vec![then()]
    } else {
        vec![otherwise()]
    }
}

/// Maps `items` to a **keyed** fragment.
///
/// For each item, `key_of` derives a stable reconciliation key and `view_of`
/// builds its [`Element`]; the key is then attached automatically. Because the
/// children are keyed, reordering the input moves retained nodes instead of
/// destroying and rebuilding them (see the keyed diff in `prism_ui_tree`).
pub fn for_each<T, I, KF, K, V>(items: I, key_of: KF, view_of: V) -> Vec<Element>
where
    I: IntoIterator<Item = T>,
    KF: Fn(&T) -> K,
    K: IntoElementKey,
    V: Fn(&T) -> Element,
{
    let iter = items.into_iter();
    let (lower, _) = iter.size_hint();
    let mut out = Vec::with_capacity(lower);
    for item in iter {
        let key = key_of(&item);
        let element = view_of(&item);
        out.push(key.apply_key(element));
    }
    out
}

/// Maps `items` to a keyed fragment using each element's **positional index**
/// as its key.
///
/// Convenient when the collection order is itself stable and meaningful;
/// prefer [`for_each`] with an identity-based key when items can be reordered.
pub fn for_each_indexed<T, I, V>(items: I, view_of: V) -> Vec<Element>
where
    I: IntoIterator<Item = T>,
    V: Fn(usize, &T) -> Element,
{
    let mut out = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        let element = view_of(index, &item);
        out.push(index.apply_key(element));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn show_includes_child_when_true() {
        let frag = show(true, || Element::text("yes"));
        assert_eq!(frag.len(), 1);
        assert_eq!(frag[0].text_content(), Some("yes"));
    }

    #[test]
    fn show_is_empty_when_false() {
        let frag = show(false, || Element::text("no"));
        assert!(frag.is_empty());
    }

    #[test]
    fn show_or_picks_the_right_branch() {
        let a = show_or(true, || Element::text("a"), || Element::text("b"));
        assert_eq!(a[0].text_content(), Some("a"));
        let b = show_or(false, || Element::text("a"), || Element::text("b"));
        assert_eq!(b[0].text_content(), Some("b"));
    }

    #[test]
    fn for_each_attaches_integer_keys() {
        let items = [10_i64, 20, 30];
        let frag = for_each(items, |&id| id, |&id| Element::text(itoa(id)));
        assert_eq!(frag.len(), 3);
        for (item, element) in items.iter().zip(&frag) {
            assert_eq!(element.text_content(), Some(itoa(*item).as_str()));
            assert!(element.explicit_key().is_some());
        }
    }

    #[test]
    fn for_each_indexed_keys_by_position() {
        let frag = for_each_indexed(["a", "b", "c"], |_, &s| Element::text(s));
        assert_eq!(frag.len(), 3);
        assert_eq!(frag[1].text_content(), Some("b"));
        assert!(frag[0].explicit_key().is_some());
    }

    #[test]
    fn for_each_empty_input_yields_empty_fragment() {
        let empty: [i64; 0] = [];
        let frag = for_each(empty, |&id| id, |&id| Element::text(itoa(id)));
        assert!(frag.is_empty());
    }

    // Tiny integer-to-string helper so tests avoid pulling in extra deps and
    // stay free of floating point (banned by the workspace lints).
    fn itoa(mut value: i64) -> String {
        if value == 0 {
            return String::from("0");
        }
        let negative = value < 0;
        let mut buf = Vec::new();
        while value != 0 {
            let digit = (value % 10).unsigned_abs() as u8;
            buf.push(b'0' + digit);
            value /= 10;
        }
        if negative {
            buf.push(b'-');
        }
        buf.reverse();
        String::from_utf8(buf).expect("ascii digits are valid utf-8")
    }
}
