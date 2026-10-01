//! Deriving accessibility data from [`prism_ui`] elements.
//!
//! These helpers bridge raw [`Element`] trees into accessibility primitives:
//! extracting a stable [`Key`], a flattened accessible name from text runs, and
//! a convenience [`Label`] for an element.

use alloc::string::String;

use prism_ui::{Element, Key};

use crate::label::Label;

/// Resolves the reconciliation [`Key`] for `element` at sibling `index`.
///
/// Mirrors the runtime's rule: an explicit key wins, otherwise the position is
/// used as a [`Key::Index`].
#[must_use]
pub fn element_key(element: &Element, index: usize) -> Key {
    element.explicit_key().cloned().unwrap_or(Key::Index(index))
}

/// Collects the accessible text of `element`: its own text run if present,
/// otherwise the in-order concatenation of its descendants' text runs,
/// separated by single spaces.
///
/// Returns `None` when the subtree contains no text at all.
///
/// # Examples
///
/// ```
/// use prism_ui::Element;
/// use prism_ui_a11y::accessible_text;
///
/// let button = Element::box_().child(Element::text("Save")).child(Element::text("now"));
/// assert_eq!(accessible_text(&button).as_deref(), Some("Save now"));
/// ```
#[must_use]
pub fn accessible_text(element: &Element) -> Option<String> {
    let mut out = String::new();
    collect_text(element, &mut out);
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn collect_text(element: &Element, out: &mut String) {
    if let Some(text) = element.text_content() {
        push_word(out, text);
    }
    for child in element.child_elements() {
        collect_text(child, out);
    }
}

fn push_word(out: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(text);
}

/// Builds a [`Label`] from an element's accessible text, or [`Label::None`]
/// when the subtree has no text.
#[must_use]
pub fn label_from_element(element: &Element) -> Label {
    match accessible_text(element) {
        Some(text) => Label::Text(text),
        None => Label::None,
    }
}

#[cfg(test)]
mod tests {
    use super::{accessible_text, element_key, label_from_element};
    use crate::label::Label;
    use prism_ui::{Element, Key};

    #[test]
    fn explicit_key_wins_otherwise_index() {
        let keyed = Element::text("x").key_int(7);
        assert_eq!(element_key(&keyed, 3), Key::Int(7));

        let unkeyed = Element::text("x");
        assert_eq!(element_key(&unkeyed, 3), Key::Index(3));
    }

    #[test]
    fn accessible_text_flattens_descendants() {
        let tree = Element::box_()
            .child(Element::text("Hello"))
            .child(Element::box_().child(Element::text("world")));
        assert_eq!(accessible_text(&tree).as_deref(), Some("Hello world"));
    }

    #[test]
    fn accessible_text_none_when_empty() {
        assert_eq!(accessible_text(&Element::box_()), None);
    }

    #[test]
    fn label_from_element_bridges() {
        assert_eq!(
            label_from_element(&Element::text("Submit")),
            Label::Text("Submit".into())
        );
        assert_eq!(label_from_element(&Element::box_()), Label::None);
    }
}
