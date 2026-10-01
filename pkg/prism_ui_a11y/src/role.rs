//! Accessibility roles.
//!
//! A [`Role`] classifies what an element *is* to assistive technology,
//! mirroring the vocabulary of the web accessibility role model (button,
//! heading, checkbox and so on). The role drives both the spoken description
//! produced by [`screen_reader_text`](crate::screen_reader_text) and the
//! keyboard semantics resolved by [`KeyboardNav`](crate::KeyboardNav).

use alloc::format;
use alloc::string::String;

/// The semantic role of an accessible element.
///
/// Roles are intentionally a closed set covering the common interactive and
/// structural widgets. Each role maps to a stable, human-readable name via
/// [`Role::describe`], used when composing screen-reader output.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// A clickable command control.
    Button,
    /// A hyperlink to another location or resource.
    Link,
    /// A section heading at the given `level` (1 is the most important).
    Heading {
        /// The heading level, where `1` is the top-level heading.
        level: u8,
    },
    /// A container for an ordered or unordered set of [`Role::ListItem`]s.
    List,
    /// A single item within a [`Role::List`].
    ListItem,
    /// A modal or non-modal dialog window.
    Dialog,
    /// A single-line or multi-line text input.
    Textbox,
    /// A two- or three-state checkable control.
    Checkbox,
    /// A mutually exclusive checkable control within a group.
    Radio,
    /// A single tab within a tab list.
    Tab,
    /// A menu containing [`Role::MenuItem`]s.
    Menu,
    /// A single item within a [`Role::Menu`].
    MenuItem,
    /// A graphic or picture.
    Image,
    /// A generic grouping of related elements.
    Group,
    /// A purely presentational element with no semantics of its own.
    Presentation,
}

impl Role {
    /// Returns the human-readable description of this role.
    ///
    /// The text is deterministic and lower-case, suitable for direct inclusion
    /// in a spoken phrase. Headings include their level, e.g.
    /// `"heading level 2"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_ui_a11y::Role;
    ///
    /// assert_eq!(Role::Button.describe(), "button");
    /// assert_eq!(Role::Heading { level: 2 }.describe(), "heading level 2");
    /// ```
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Role::Button => String::from("button"),
            Role::Link => String::from("link"),
            Role::Heading { level } => format!("heading level {level}"),
            Role::List => String::from("list"),
            Role::ListItem => String::from("list item"),
            Role::Dialog => String::from("dialog"),
            Role::Textbox => String::from("text field"),
            Role::Checkbox => String::from("checkbox"),
            Role::Radio => String::from("radio button"),
            Role::Tab => String::from("tab"),
            Role::Menu => String::from("menu"),
            Role::MenuItem => String::from("menu item"),
            Role::Image => String::from("image"),
            Role::Group => String::from("group"),
            Role::Presentation => String::from("presentation"),
        }
    }

    /// Returns `true` when this role denotes a composite container whose items
    /// are navigated with the arrow keys (list, menu or tab list).
    #[must_use]
    pub fn is_arrow_navigable(&self) -> bool {
        matches!(
            self,
            Role::List | Role::ListItem | Role::Menu | Role::MenuItem | Role::Tab
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Role;

    #[test]
    fn describe_is_stable() {
        assert_eq!(Role::Button.describe(), "button");
        assert_eq!(Role::Link.describe(), "link");
        assert_eq!(Role::Heading { level: 3 }.describe(), "heading level 3");
        assert_eq!(Role::ListItem.describe(), "list item");
        assert_eq!(Role::Checkbox.describe(), "checkbox");
        assert_eq!(Role::Presentation.describe(), "presentation");
    }

    #[test]
    fn arrow_navigable_roles() {
        assert!(Role::Menu.is_arrow_navigable());
        assert!(Role::Tab.is_arrow_navigable());
        assert!(Role::ListItem.is_arrow_navigable());
        assert!(!Role::Button.is_arrow_navigable());
        assert!(!Role::Dialog.is_arrow_navigable());
    }
}
