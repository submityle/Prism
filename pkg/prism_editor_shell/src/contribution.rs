//! Deterministic menu, toolbar, and status-bar contribution aggregation.

use std::collections::BTreeMap;

use crate::{ContextExpr, ContextKeySet};

/// A token that can unregister one surface contribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContributionRegistration {
    serial: u64,
    kind: ContributionKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContributionKind {
    Menu,
    Toolbar,
    Status,
}

/// An error raised while registering a contribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContributionError {
    /// The contribution ID is empty.
    EmptyId,
    /// The contribution ID is already registered for this surface.
    DuplicateId(String),
}

impl std::fmt::Display for ContributionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId => formatter.write_str("contribution id cannot be empty"),
            Self::DuplicateId(id) => write!(formatter, "duplicate contribution id {id}"),
        }
    }
}

impl std::error::Error for ContributionError {}

/// A menu item contribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuContribution {
    /// Stable contribution ID.
    pub id: String,
    /// Menu surface, such as `file` or `edit`.
    pub menu_id: String,
    /// Optional parent item ID for nested menus.
    pub parent_id: Option<String>,
    /// Display label.
    pub label: String,
    /// Command invoked by this item, if it is not a submenu/separator.
    pub command_id: Option<String>,
    /// Sort group.
    pub group: String,
    /// Order within a group.
    pub order: i32,
    /// Context controlling visibility.
    pub when: ContextExpr,
}

/// A resolved menu item returned for a context.
pub type MenuItem = MenuContribution;

/// A toolbar item contribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolbarContribution {
    /// Stable contribution ID.
    pub id: String,
    /// Toolbar surface, such as `main`.
    pub toolbar_id: String,
    /// Display label.
    pub label: String,
    /// Command invoked by this item.
    pub command_id: String,
    /// Sort group.
    pub group: String,
    /// Order within a group.
    pub order: i32,
    /// Context controlling visibility.
    pub when: ContextExpr,
}

/// A resolved toolbar item returned for a context.
pub type ToolbarItem = ToolbarContribution;

/// Status-bar placement region.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StatusRegion {
    /// Left side of the status bar.
    Left,
    /// Center of the status bar.
    Center,
    /// Right side of the status bar.
    Right,
}

/// A status-bar contribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StatusContribution {
    /// Stable contribution ID.
    pub id: String,
    /// Target status-bar region.
    pub region: StatusRegion,
    /// Render-neutral text payload.
    pub text: String,
    /// Larger priorities sort first within a region.
    pub priority: i32,
    /// Context controlling visibility.
    pub when: ContextExpr,
}

/// A resolved status-bar item returned for a context.
pub type StatusItem = StatusContribution;

struct Registered<T> {
    serial: u64,
    value: T,
}

/// Registry and aggregator for shell surface contributions.
#[derive(Default)]
pub struct ShellContributions {
    menus: BTreeMap<String, Registered<MenuContribution>>,
    toolbars: BTreeMap<String, Registered<ToolbarContribution>>,
    statuses: BTreeMap<String, Registered<StatusContribution>>,
    next_serial: u64,
}

impl ShellContributions {
    /// Registers a menu contribution.
    pub fn register_menu(
        &mut self,
        contribution: MenuContribution,
    ) -> Result<ContributionRegistration, ContributionError> {
        let serial = register_into(
            &mut self.menus,
            &mut self.next_serial,
            contribution.id.clone(),
            contribution,
        )?;
        Ok(ContributionRegistration {
            serial,
            kind: ContributionKind::Menu,
        })
    }

    /// Registers a toolbar contribution.
    pub fn register_toolbar(
        &mut self,
        contribution: ToolbarContribution,
    ) -> Result<ContributionRegistration, ContributionError> {
        let serial = register_into(
            &mut self.toolbars,
            &mut self.next_serial,
            contribution.id.clone(),
            contribution,
        )?;
        Ok(ContributionRegistration {
            serial,
            kind: ContributionKind::Toolbar,
        })
    }

    /// Registers a status-bar contribution.
    pub fn register_status(
        &mut self,
        contribution: StatusContribution,
    ) -> Result<ContributionRegistration, ContributionError> {
        let serial = register_into(
            &mut self.statuses,
            &mut self.next_serial,
            contribution.id.clone(),
            contribution,
        )?;
        Ok(ContributionRegistration {
            serial,
            kind: ContributionKind::Status,
        })
    }

    /// Unregisters a contribution token from its original surface.
    pub fn unregister(&mut self, registration: ContributionRegistration) -> bool {
        match registration.kind {
            ContributionKind::Menu => remove_serial(&mut self.menus, registration.serial),
            ContributionKind::Toolbar => remove_serial(&mut self.toolbars, registration.serial),
            ContributionKind::Status => remove_serial(&mut self.statuses, registration.serial),
        }
    }

    /// Returns visible menu items sorted by group, order, and ID.
    #[must_use]
    pub fn menu(&self, menu_id: &str, context: &ContextKeySet) -> Vec<MenuItem> {
        let mut result = self
            .menus
            .values()
            .filter(|item| item.value.menu_id == menu_id && item.value.when.matches(context))
            .map(|item| item.value.clone())
            .collect::<Vec<_>>();
        result.sort_by(|left, right| {
            left.parent_id
                .cmp(&right.parent_id)
                .then_with(|| left.group.cmp(&right.group))
                .then_with(|| left.order.cmp(&right.order))
                .then_with(|| left.id.cmp(&right.id))
        });
        result
    }

    /// Returns visible toolbar items sorted by group, order, and ID.
    #[must_use]
    pub fn toolbar(&self, toolbar_id: &str, context: &ContextKeySet) -> Vec<ToolbarItem> {
        let mut result = self
            .toolbars
            .values()
            .filter(|item| item.value.toolbar_id == toolbar_id && item.value.when.matches(context))
            .map(|item| item.value.clone())
            .collect::<Vec<_>>();
        result.sort_by(|left, right| {
            left.group
                .cmp(&right.group)
                .then_with(|| left.order.cmp(&right.order))
                .then_with(|| left.id.cmp(&right.id))
        });
        result
    }

    /// Returns visible status items sorted by region, descending priority, and ID.
    #[must_use]
    pub fn status(&self, context: &ContextKeySet) -> Vec<StatusItem> {
        let mut result = self
            .statuses
            .values()
            .filter(|item| item.value.when.matches(context))
            .map(|item| item.value.clone())
            .collect::<Vec<_>>();
        result.sort_by(|left, right| {
            left.region
                .cmp(&right.region)
                .then_with(|| right.priority.cmp(&left.priority))
                .then_with(|| left.id.cmp(&right.id))
        });
        result
    }

    /// Returns the number of registered menu contributions.
    #[must_use]
    pub fn menu_count(&self) -> usize {
        self.menus.len()
    }

    /// Returns the number of registered toolbar contributions.
    #[must_use]
    pub fn toolbar_count(&self) -> usize {
        self.toolbars.len()
    }

    /// Returns the number of registered status-bar contributions.
    #[must_use]
    pub fn status_count(&self) -> usize {
        self.statuses.len()
    }
}

fn register_into<T>(
    map: &mut BTreeMap<String, Registered<T>>,
    next_serial: &mut u64,
    id: String,
    value: T,
) -> Result<u64, ContributionError> {
    if id.trim().is_empty() {
        return Err(ContributionError::EmptyId);
    }
    if map.contains_key(&id) {
        return Err(ContributionError::DuplicateId(id));
    }
    *next_serial = next_serial.saturating_add(1);
    map.insert(
        id,
        Registered {
            serial: *next_serial,
            value,
        },
    );
    Ok(*next_serial)
}

fn remove_serial<T>(map: &mut BTreeMap<String, Registered<T>>, serial: u64) -> bool {
    let Some(id) = map
        .iter()
        .find_map(|(id, value)| (value.serial == serial).then(|| id.clone()))
    else {
        return false;
    };
    map.remove(&id).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(expression: &str) -> ContextExpr {
        ContextExpr::parse(expression).expect("context")
    }

    #[test]
    fn aggregation_filters_and_sorts_deterministically() {
        let mut contributions = ShellContributions::default();
        contributions
            .register_menu(MenuContribution {
                id: "edit.copy".into(),
                menu_id: "edit".into(),
                parent_id: None,
                label: "Copy".into(),
                command_id: Some("copy".into()),
                group: "clipboard".into(),
                order: 20,
                when: ContextExpr::Any,
            })
            .expect("register");
        contributions
            .register_menu(MenuContribution {
                id: "edit.paste".into(),
                menu_id: "edit".into(),
                parent_id: None,
                label: "Paste".into(),
                command_id: Some("paste".into()),
                group: "clipboard".into(),
                order: 10,
                when: context("clipboard.has_data"),
            })
            .expect("register");

        let empty = ContextKeySet::new();
        assert_eq!(contributions.menu("edit", &empty)[0].id, "edit.copy");
        let mut active = ContextKeySet::new();
        active.insert("clipboard.has_data");
        assert_eq!(
            contributions
                .menu("edit", &active)
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ["edit.paste", "edit.copy"]
        );

        contributions
            .register_toolbar(ToolbarContribution {
                id: "main.save".into(),
                toolbar_id: "main".into(),
                label: "Save".into(),
                command_id: "scene.save".into(),
                group: "file".into(),
                order: 0,
                when: ContextExpr::Any,
            })
            .expect("toolbar");
        contributions
            .register_status(StatusContribution {
                id: "status.mode".into(),
                region: StatusRegion::Right,
                text: "Editing".into(),
                priority: 5,
                when: ContextExpr::Any,
            })
            .expect("status");
        assert_eq!(contributions.toolbar("main", &active).len(), 1);
        assert_eq!(contributions.status(&active)[0].id, "status.mode");
    }

    #[test]
    fn registrations_are_surface_scoped_and_disposable() {
        let mut contributions = ShellContributions::default();
        let menu = contributions
            .register_menu(MenuContribution {
                id: "file.open".into(),
                menu_id: "file".into(),
                parent_id: None,
                label: "Open".into(),
                command_id: Some("open".into()),
                group: "file".into(),
                order: 0,
                when: ContextExpr::Any,
            })
            .expect("register");
        let toolbar = contributions
            .register_toolbar(ToolbarContribution {
                id: "main.open".into(),
                toolbar_id: "main".into(),
                label: "Open".into(),
                command_id: "open".into(),
                group: "file".into(),
                order: 0,
                when: ContextExpr::Any,
            })
            .expect("register");
        assert!(contributions.unregister(menu));
        assert!(!contributions.unregister(menu));
        assert_eq!(contributions.toolbar_count(), 1);
        assert!(contributions.unregister(toolbar));
        assert_eq!(contributions.toolbar_count(), 0);
    }
}
