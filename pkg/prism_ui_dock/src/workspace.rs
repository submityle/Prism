//! Pure-data workspace and perspective management.
//!
//! A [`Perspective`] owns one [`DockLayout`] snapshot. [`Workspace`] and
//! [`WorkspaceManager`] retain snapshots in insertion order and expose only
//! cloned layouts at their boundaries. This module deliberately does not
//! perform file I/O, autosave, rendering, or widget construction.

use alloc::{string::String, vec::Vec};
use core::fmt;

use crate::DockLayout;

/// The named object for which a name was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NameKind {
    /// A workspace name.
    Workspace,
    /// A perspective name.
    Perspective,
}

impl fmt::Display for NameKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace => formatter.write_str("workspace"),
            Self::Perspective => formatter.write_str("perspective"),
        }
    }
}

/// A structured failure from workspace or perspective management.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    /// A workspace or perspective name is empty or whitespace-only.
    EmptyName {
        /// The kind of object whose name was rejected.
        kind: NameKind,
    },
    /// A workspace with this name already exists.
    DuplicateWorkspace {
        /// The conflicting workspace name.
        name: String,
    },
    /// A perspective with this name already exists in the workspace.
    DuplicatePerspective {
        /// The containing workspace name.
        workspace: String,
        /// The conflicting perspective name.
        perspective: String,
    },
    /// The requested workspace does not exist.
    WorkspaceNotFound {
        /// The requested workspace name.
        name: String,
    },
    /// The requested perspective does not exist in the workspace.
    PerspectiveNotFound {
        /// The containing workspace name.
        workspace: String,
        /// The requested perspective name.
        perspective: String,
    },
    /// An operation requires an active workspace, but none is selected.
    NoActiveWorkspace,
    /// An operation requires an active perspective, but none is selected.
    NoActivePerspective {
        /// The active workspace name.
        workspace: String,
    },
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName { kind } => write!(formatter, "{kind} name must not be empty"),
            Self::DuplicateWorkspace { name } => {
                write!(formatter, "workspace name is already registered: {name}")
            }
            Self::DuplicatePerspective {
                workspace,
                perspective,
            } => write!(
                formatter,
                "perspective name is already registered in workspace {workspace}: {perspective}"
            ),
            Self::WorkspaceNotFound { name } => {
                write!(formatter, "workspace was not found: {name}")
            }
            Self::PerspectiveNotFound {
                workspace,
                perspective,
            } => write!(
                formatter,
                "perspective was not found in workspace {workspace}: {perspective}"
            ),
            Self::NoActiveWorkspace => formatter.write_str("no active workspace is selected"),
            Self::NoActivePerspective { workspace } => write!(
                formatter,
                "no active perspective is selected in workspace {workspace}"
            ),
        }
    }
}

fn validate_name(kind: NameKind, name: &str) -> Result<(), WorkspaceError> {
    if name.trim().is_empty() {
        Err(WorkspaceError::EmptyName { kind })
    } else {
        Ok(())
    }
}

/// A named, independently owned [`DockLayout`] snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Perspective {
    name: String,
    layout: DockLayout,
}

impl Perspective {
    /// Creates a perspective by deep-copying the supplied layout.
    pub fn new(name: impl Into<String>, layout: &DockLayout) -> Result<Self, WorkspaceError> {
        let name = name.into();
        validate_name(NameKind::Perspective, &name)?;
        Ok(Self {
            name,
            layout: layout.clone(),
        })
    }

    /// Returns the stable perspective name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns an independently owned copy of the saved layout.
    #[must_use]
    pub fn layout(&self) -> DockLayout {
        self.layout.clone()
    }

    /// Replaces the saved layout with a deep copy.
    pub fn replace_layout(&mut self, layout: &DockLayout) {
        self.layout = layout.clone();
    }
}

/// An ordered named collection of independent perspectives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Workspace {
    name: String,
    perspectives: Vec<Perspective>,
    active_perspective: Option<usize>,
}

impl Workspace {
    /// Creates an empty workspace.
    pub fn new(name: impl Into<String>) -> Result<Self, WorkspaceError> {
        let name = name.into();
        validate_name(NameKind::Workspace, &name)?;
        Ok(Self {
            name,
            perspectives: Vec::new(),
            active_perspective: None,
        })
    }

    /// Returns the stable workspace name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds a perspective at the end of the deterministic insertion order.
    ///
    /// The layout is cloned before this method returns. Empty names and
    /// duplicate names are rejected without mutating the workspace.
    pub fn add_perspective(
        &mut self,
        name: impl Into<String>,
        layout: &DockLayout,
    ) -> Result<(), WorkspaceError> {
        let name = name.into();
        let perspective = Perspective::new(name, layout)?;
        if self
            .perspectives
            .iter()
            .any(|existing| existing.name == perspective.name)
        {
            return Err(WorkspaceError::DuplicatePerspective {
                workspace: self.name.clone(),
                perspective: perspective.name,
            });
        }

        self.perspectives.push(perspective);
        if self.active_perspective.is_none() {
            self.active_perspective = Some(0);
        }
        Ok(())
    }

    /// Returns a perspective by name without exposing mutable state.
    #[must_use]
    pub fn perspective(&self, name: &str) -> Option<&Perspective> {
        self.perspectives
            .iter()
            .find(|perspective| perspective.name == name)
    }

    /// Returns perspectives in insertion order.
    pub fn perspectives(&self) -> core::slice::Iter<'_, Perspective> {
        self.perspectives.iter()
    }

    /// Returns the number of perspectives.
    #[must_use]
    pub const fn perspective_count(&self) -> usize {
        self.perspectives.len()
    }

    /// Returns whether this workspace has no perspectives.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.perspectives.is_empty()
    }

    /// Selects a perspective and returns a deep copy of its layout.
    pub fn switch_perspective(&mut self, name: &str) -> Result<DockLayout, WorkspaceError> {
        let index = self
            .perspectives
            .iter()
            .position(|perspective| perspective.name == name)
            .ok_or_else(|| WorkspaceError::PerspectiveNotFound {
                workspace: self.name.clone(),
                perspective: String::from(name),
            })?;
        self.active_perspective = Some(index);
        Ok(self.perspectives[index].layout())
    }

    /// Returns the active perspective name, if one is selected.
    #[must_use]
    pub fn active_perspective_name(&self) -> Option<&str> {
        self.active_perspective
            .and_then(|index| self.perspectives.get(index))
            .map(Perspective::name)
    }

    /// Returns a deep copy of the active layout, if one is selected.
    #[must_use]
    pub fn active_layout(&self) -> Option<DockLayout> {
        self.active_perspective
            .and_then(|index| self.perspectives.get(index))
            .map(Perspective::layout)
    }

    /// Replaces the active perspective's layout with a deep copy.
    pub fn save_active_layout(&mut self, layout: &DockLayout) -> Result<(), WorkspaceError> {
        let index = self
            .active_perspective
            .ok_or_else(|| WorkspaceError::NoActivePerspective {
                workspace: self.name.clone(),
            })?;
        self.perspectives[index].replace_layout(layout);
        Ok(())
    }

    /// Replaces a named perspective's layout with a deep copy.
    pub fn save_layout(&mut self, name: &str, layout: &DockLayout) -> Result<(), WorkspaceError> {
        let workspace_name = self.name.clone();
        let perspective =
            self.perspective_mut(name)
                .ok_or_else(|| WorkspaceError::PerspectiveNotFound {
                    workspace: workspace_name,
                    perspective: String::from(name),
                })?;
        perspective.replace_layout(layout);
        Ok(())
    }

    fn perspective_mut(&mut self, name: &str) -> Option<&mut Perspective> {
        self.perspectives
            .iter_mut()
            .find(|perspective| perspective.name == name)
    }
}

impl<'a> IntoIterator for &'a Workspace {
    type Item = &'a Perspective;
    type IntoIter = core::slice::Iter<'a, Perspective>;

    fn into_iter(self) -> Self::IntoIter {
        self.perspectives()
    }
}

/// An ordered collection of named workspaces and the current selection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceManager {
    workspaces: Vec<Workspace>,
    active_workspace: Option<usize>,
}

impl WorkspaceManager {
    /// Creates an empty manager.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            workspaces: Vec::new(),
            active_workspace: None,
        }
    }

    /// Adds a workspace at the end of the deterministic insertion order.
    pub fn add_workspace(&mut self, name: impl Into<String>) -> Result<(), WorkspaceError> {
        let workspace = Workspace::new(name)?;
        if self
            .workspaces
            .iter()
            .any(|existing| existing.name == workspace.name)
        {
            return Err(WorkspaceError::DuplicateWorkspace {
                name: workspace.name,
            });
        }

        self.workspaces.push(workspace);
        if self.active_workspace.is_none() {
            self.active_workspace = Some(0);
        }
        Ok(())
    }

    /// Returns a workspace by name without exposing mutable state.
    #[must_use]
    pub fn workspace(&self, name: &str) -> Option<&Workspace> {
        self.workspaces
            .iter()
            .find(|workspace| workspace.name == name)
    }

    /// Returns workspaces in insertion order.
    pub fn workspaces(&self) -> core::slice::Iter<'_, Workspace> {
        self.workspaces.iter()
    }

    /// Returns the number of workspaces.
    #[must_use]
    pub const fn workspace_count(&self) -> usize {
        self.workspaces.len()
    }

    /// Returns whether no workspaces are registered.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.workspaces.is_empty()
    }

    /// Selects a workspace without changing any stored perspective.
    pub fn switch_workspace(&mut self, name: &str) -> Result<(), WorkspaceError> {
        let index = self
            .workspaces
            .iter()
            .position(|workspace| workspace.name == name)
            .ok_or_else(|| WorkspaceError::WorkspaceNotFound {
                name: String::from(name),
            })?;
        self.active_workspace = Some(index);
        Ok(())
    }

    /// Returns the active workspace without exposing mutable state.
    #[must_use]
    pub fn active_workspace(&self) -> Option<&Workspace> {
        self.active_workspace
            .and_then(|index| self.workspaces.get(index))
    }

    /// Returns the active workspace name, if one is selected.
    #[must_use]
    pub fn active_workspace_name(&self) -> Option<&str> {
        self.active_workspace().map(Workspace::name)
    }

    /// Adds a perspective to the active workspace.
    pub fn add_perspective(
        &mut self,
        name: impl Into<String>,
        layout: &DockLayout,
    ) -> Result<(), WorkspaceError> {
        let workspace = self.active_workspace_mut()?;
        workspace.add_perspective(name, layout)
    }

    /// Selects a perspective in the active workspace and returns a deep copy.
    pub fn switch_perspective(&mut self, name: &str) -> Result<DockLayout, WorkspaceError> {
        let workspace = self.active_workspace_mut()?;
        workspace.switch_perspective(name)
    }

    /// Returns a deep copy of the active workspace's active layout.
    #[must_use]
    pub fn active_layout(&self) -> Option<DockLayout> {
        self.active_workspace().and_then(Workspace::active_layout)
    }

    /// Replaces the active perspective's layout with a deep copy.
    pub fn save_active_layout(&mut self, layout: &DockLayout) -> Result<(), WorkspaceError> {
        let workspace = self.active_workspace_mut()?;
        workspace.save_active_layout(layout)
    }

    /// Replaces a named perspective in the active workspace with a deep copy.
    pub fn save_layout(&mut self, name: &str, layout: &DockLayout) -> Result<(), WorkspaceError> {
        let workspace = self.active_workspace_mut()?;
        workspace.save_layout(name, layout)
    }

    fn active_workspace_mut(&mut self) -> Result<&mut Workspace, WorkspaceError> {
        let index = self
            .active_workspace
            .ok_or(WorkspaceError::NoActiveWorkspace)?;
        Ok(&mut self.workspaces[index])
    }
}

impl<'a> IntoIterator for &'a WorkspaceManager {
    type Item = &'a Workspace;
    type IntoIter = core::slice::Iter<'a, Workspace>;

    fn into_iter(self) -> Self::IntoIter {
        self.workspaces()
    }
}
