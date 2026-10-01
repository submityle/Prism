//! The [`Workbench`] registry: hierarchical, stable organisation of stories.
//!
//! Stories are grouped by a slash-delimited *group path* (e.g.
//! `"Forms/Button"`), mirroring Storybook's sidebar hierarchy. The registry
//! keeps groups and the stories within them in sorted [`BTreeMap`] order, so
//! listing is deterministic regardless of insertion order — the property that
//! makes rendered snapshots reproducible.

#![forbid(unsafe_code)]

use alloc::collections::btree_map::Entry;
use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::story::Story;

/// A registry of stories organised into named, sorted groups.
///
/// Each group holds its stories keyed by [`Story::name`]. Both levels are
/// [`BTreeMap`]s, giving a stable, sorted traversal order.
#[derive(Debug, Default)]
pub struct Workbench {
    groups: BTreeMap<String, BTreeMap<String, Story>>,
}

impl Workbench {
    /// Creates an empty workbench.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }

    /// Adds `story` under the group path `group`, keyed by its name.
    ///
    /// When a story with the same name already exists in that group it is
    /// replaced, and the previous story is returned.
    pub fn add(&mut self, group: impl Into<String>, story: Story) -> Option<Story> {
        let group = group.into();
        match self.groups.entry(group) {
            Entry::Vacant(slot) => {
                let mut stories = BTreeMap::new();
                let previous = stories.insert(story.name().into(), story);
                slot.insert(stories);
                previous
            }
            Entry::Occupied(mut slot) => slot.get_mut().insert(story.name().into(), story),
        }
    }

    /// Borrows the story named `name` within `group`, if present.
    #[must_use]
    pub fn get(&self, group: &str, name: &str) -> Option<&Story> {
        self.groups.get(group).and_then(|stories| stories.get(name))
    }

    /// Returns `true` when a story named `name` exists in `group`.
    #[must_use]
    pub fn contains(&self, group: &str, name: &str) -> bool {
        self.get(group, name).is_some()
    }

    /// Iterates the group paths in sorted order.
    pub fn groups(&self) -> impl Iterator<Item = &str> {
        self.groups.keys().map(String::as_str)
    }

    /// The number of groups in the workbench.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The total number of stories across all groups.
    #[must_use]
    pub fn len(&self) -> usize {
        self.groups.values().map(BTreeMap::len).sum()
    }

    /// Returns `true` when the workbench holds no stories.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.groups.values().all(BTreeMap::is_empty)
    }

    /// Iterates the stories in `group` in sorted name order.
    ///
    /// Yields nothing when the group does not exist.
    pub fn group(&self, group: &str) -> impl Iterator<Item = &Story> {
        self.groups
            .get(group)
            .into_iter()
            .flat_map(BTreeMap::values)
    }

    /// Iterates every story as `(group, name, story)` in sorted order.
    ///
    /// Groups are visited in sorted path order and, within each group, stories
    /// in sorted name order.
    pub fn stories(&self) -> impl Iterator<Item = (&str, &str, &Story)> {
        self.groups.iter().flat_map(|(group, stories)| {
            stories
                .iter()
                .map(move |(name, story)| (group.as_str(), name.as_str(), story))
        })
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::controls::ControlValue;
    use alloc::vec::Vec;
    use prism_ui::Element;

    fn leaf(name: &str) -> Story {
        Story::builder(name).build(|_ctx| Element::box_())
    }

    #[test]
    fn add_and_get_roundtrip() {
        let mut wb = Workbench::new();
        wb.add("Forms/Button", leaf("Primary"));

        assert!(wb.get("Forms/Button", "Primary").is_some());
        assert!(wb.get("Forms/Button", "Missing").is_none());
        assert!(wb.contains("Forms/Button", "Primary"));
    }

    #[test]
    fn add_replaces_same_name_and_returns_previous() {
        let mut wb = Workbench::new();
        assert!(wb.add("Forms/Button", leaf("Primary")).is_none());

        let replacement = Story::builder("Primary")
            .arg("flag", ControlValue::Bool(true))
            .build(|_ctx| Element::box_());
        let previous = wb.add("Forms/Button", replacement).expect("replaced");
        assert_eq!(previous.name(), "Primary");
        assert_eq!(wb.len(), 1);
    }

    #[test]
    fn listing_is_sorted_and_stable() {
        let mut wb = Workbench::new();
        wb.add("Forms/Button", leaf("Secondary"));
        wb.add("Forms/Button", leaf("Primary"));
        wb.add("Layout/Stack", leaf("Vertical"));

        let groups: Vec<&str> = wb.groups().collect();
        assert_eq!(groups, ["Forms/Button", "Layout/Stack"]);

        let button_stories: Vec<&str> = wb.group("Forms/Button").map(Story::name).collect();
        assert_eq!(button_stories, ["Primary", "Secondary"]);

        let all: Vec<(&str, &str)> = wb.stories().map(|(group, name, _)| (group, name)).collect();
        assert_eq!(
            all,
            [
                ("Forms/Button", "Primary"),
                ("Forms/Button", "Secondary"),
                ("Layout/Stack", "Vertical"),
            ]
        );
    }

    #[test]
    fn counts_and_emptiness() {
        let mut wb = Workbench::new();
        assert!(wb.is_empty());
        assert_eq!(wb.len(), 0);

        wb.add("Forms/Button", leaf("Primary"));
        wb.add("Layout/Stack", leaf("Vertical"));
        assert!(!wb.is_empty());
        assert_eq!(wb.group_count(), 2);
        assert_eq!(wb.len(), 2);
    }

    #[test]
    fn missing_group_yields_empty_iterator() {
        let wb = Workbench::new();
        assert_eq!(wb.group("Nope").count(), 0);
    }
}
