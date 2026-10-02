//! The capability whitelist.
//!
//! A [`CapabilitySet`] is the single source of truth for what a remote
//! document is allowed to reference: which component kinds may appear, which
//! style tokens may be applied and which event names may be bound. Anything
//! absent from the set is denied by the [`Sandbox`](crate::Sandbox); there is
//! no implicit escalation.

use alloc::collections::BTreeSet;
use alloc::string::String;

/// An allow-list of component kinds, style tokens and event names.
///
/// The set starts empty: a freshly constructed `CapabilitySet` permits nothing,
/// so capabilities must be opted into explicitly. Internally every allow-list
/// is an ordered set, which keeps membership queries cheap and iteration
/// deterministic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    kinds: BTreeSet<String>,
    styles: BTreeSet<String>,
    events: BTreeSet<String>,
}

impl CapabilitySet {
    /// Creates an empty set that permits nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allows a single component kind.
    #[must_use]
    pub fn allow_kind(mut self, kind: impl Into<String>) -> Self {
        self.kinds.insert(kind.into());
        self
    }

    /// Allows many component kinds.
    #[must_use]
    pub fn allow_kinds<I, S>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.kinds.extend(kinds.into_iter().map(Into::into));
        self
    }

    /// Allows a single style token.
    #[must_use]
    pub fn allow_style(mut self, token: impl Into<String>) -> Self {
        self.styles.insert(token.into());
        self
    }

    /// Allows many style tokens.
    #[must_use]
    pub fn allow_styles<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.styles.extend(tokens.into_iter().map(Into::into));
        self
    }

    /// Allows a single event name.
    #[must_use]
    pub fn allow_event(mut self, name: impl Into<String>) -> Self {
        self.events.insert(name.into());
        self
    }

    /// Allows many event names.
    #[must_use]
    pub fn allow_events<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.events.extend(names.into_iter().map(Into::into));
        self
    }

    /// Whether `kind` is a whitelisted component kind.
    #[must_use]
    pub fn allows_kind(&self, kind: &str) -> bool {
        self.kinds.contains(kind)
    }

    /// Whether `token` is a whitelisted style token.
    #[must_use]
    pub fn allows_style(&self, token: &str) -> bool {
        self.styles.contains(token)
    }

    /// Whether `name` is a whitelisted event name.
    #[must_use]
    pub fn allows_event(&self, name: &str) -> bool {
        self.events.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::CapabilitySet;

    #[test]
    fn empty_set_permits_nothing() {
        let caps = CapabilitySet::new();
        assert!(!caps.allows_kind("box"));
        assert!(!caps.allows_style("card"));
        assert!(!caps.allows_event("tap"));
    }

    #[test]
    fn builder_allows_requested_capabilities() {
        let caps = CapabilitySet::new()
            .allow_kinds(["box", "text"])
            .allow_style("card")
            .allow_events(["tap", "hover"]);
        assert!(caps.allows_kind("box"));
        assert!(caps.allows_kind("text"));
        assert!(!caps.allows_kind("script"));
        assert!(caps.allows_style("card"));
        assert!(!caps.allows_style("danger"));
        assert!(caps.allows_event("tap"));
        assert!(caps.allows_event("hover"));
        assert!(!caps.allows_event("drop"));
    }

    #[test]
    fn capabilities_are_independent_namespaces() {
        // A kind named "card" does not grant the style token "card".
        let caps = CapabilitySet::new().allow_kind("card");
        assert!(caps.allows_kind("card"));
        assert!(!caps.allows_style("card"));
        assert!(!caps.allows_event("card"));
    }
}
