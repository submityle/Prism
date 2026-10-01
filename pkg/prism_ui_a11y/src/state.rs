//! Accessible state flags.
//!
//! [`AriaState`] captures the dynamic, assistive-technology-visible state of an
//! element independent of its visual styling: whether it is disabled, hidden
//! from the accessibility tree, checked, expanded, selected or required.

/// The dynamic accessibility state of an element.
///
/// Tri-state fields use `Option<bool>`: `None` means the state is not
/// applicable to the element (for example, a plain button has no `checked`
/// state), while `Some(_)` carries the current value.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AriaState {
    /// The element is present but cannot be interacted with.
    pub disabled: bool,
    /// The element is removed from the accessibility tree entirely.
    pub hidden: bool,
    /// Checkable state for checkboxes and radios, if applicable.
    pub checked: Option<bool>,
    /// Expanded/collapsed state for disclosure controls, if applicable.
    pub expanded: Option<bool>,
    /// Selection state for tabs, options and list items, if applicable.
    pub selected: Option<bool>,
    /// The element must have a value before a form can be submitted.
    pub required: bool,
}

impl AriaState {
    /// Creates an empty state with every flag cleared and every tri-state
    /// field set to `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `disabled` flag.
    #[must_use]
    pub fn disabled(mut self, value: bool) -> Self {
        self.disabled = value;
        self
    }

    /// Sets the `hidden` flag.
    #[must_use]
    pub fn hidden(mut self, value: bool) -> Self {
        self.hidden = value;
        self
    }

    /// Sets the `checked` tri-state.
    #[must_use]
    pub fn checked(mut self, value: bool) -> Self {
        self.checked = Some(value);
        self
    }

    /// Sets the `expanded` tri-state.
    #[must_use]
    pub fn expanded(mut self, value: bool) -> Self {
        self.expanded = Some(value);
        self
    }

    /// Sets the `selected` tri-state.
    #[must_use]
    pub fn selected(mut self, value: bool) -> Self {
        self.selected = Some(value);
        self
    }

    /// Sets the `required` flag.
    #[must_use]
    pub fn required(mut self, value: bool) -> Self {
        self.required = value;
        self
    }

    /// Returns `true` when the element should be excluded from focus and
    /// navigation because it is either `disabled` or `hidden`.
    #[must_use]
    pub fn is_inert(&self) -> bool {
        self.disabled || self.hidden
    }
}

#[cfg(test)]
mod tests {
    use super::AriaState;

    #[test]
    fn builder_sets_fields() {
        let s = AriaState::new()
            .disabled(true)
            .checked(false)
            .expanded(true)
            .selected(true)
            .required(true);
        assert!(s.disabled);
        assert!(!s.hidden);
        assert_eq!(s.checked, Some(false));
        assert_eq!(s.expanded, Some(true));
        assert_eq!(s.selected, Some(true));
        assert!(s.required);
    }

    #[test]
    fn default_is_empty() {
        let s = AriaState::default();
        assert!(!s.disabled);
        assert!(!s.hidden);
        assert_eq!(s.checked, None);
        assert_eq!(s.expanded, None);
        assert_eq!(s.selected, None);
        assert!(!s.required);
        assert!(!s.is_inert());
    }

    #[test]
    fn inert_when_disabled_or_hidden() {
        assert!(AriaState::new().disabled(true).is_inert());
        assert!(AriaState::new().hidden(true).is_inert());
        assert!(!AriaState::new().checked(true).is_inert());
    }
}
