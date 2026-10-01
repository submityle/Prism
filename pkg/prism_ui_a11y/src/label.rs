//! Accessible names.
//!
//! A [`Label`] is the source of an element's accessible name. It is either an
//! inline string, a reference to another element that supplies the name
//! (`labelled-by`), or absent.

use alloc::string::String;

use prism_ui::Key;

/// The source of an element's accessible name.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Label {
    /// An inline, literal accessible name.
    Text(String),
    /// The name is taken from another element identified by its [`Key`].
    LabelledBy(Key),
    /// The element has no accessible name of its own.
    #[default]
    None,
}

impl Label {
    /// Creates a [`Label::Text`] from anything string-like.
    #[must_use]
    pub fn text(value: impl Into<String>) -> Self {
        Label::Text(value.into())
    }

    /// Creates a [`Label::LabelledBy`] pointing at `key`.
    #[must_use]
    pub fn labelled_by(key: Key) -> Self {
        Label::LabelledBy(key)
    }

    /// Returns the inline text if this is a [`Label::Text`], otherwise `None`.
    ///
    /// This does *not* resolve [`Label::LabelledBy`]; use
    /// [`A11yTree::label_text`](crate::A11yTree::label_text) for full
    /// resolution against a tree.
    #[must_use]
    pub fn inline_text(&self) -> Option<&str> {
        match self {
            Label::Text(s) => Some(s.as_str()),
            Label::LabelledBy(_) | Label::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Label;
    use prism_ui::Key;

    #[test]
    fn constructors() {
        assert_eq!(Label::text("hi"), Label::Text("hi".into()));
        assert_eq!(
            Label::labelled_by(Key::Int(3)),
            Label::LabelledBy(Key::Int(3))
        );
        assert_eq!(Label::default(), Label::None);
    }

    #[test]
    fn inline_text_only_for_text() {
        assert_eq!(Label::text("x").inline_text(), Some("x"));
        assert_eq!(Label::LabelledBy(Key::Int(1)).inline_text(), None);
        assert_eq!(Label::None.inline_text(), None);
    }
}
