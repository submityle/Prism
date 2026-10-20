use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

/// Stable identity assigned to a registered editor contribution.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ContributionId(u128);

impl ContributionId {
    /// Creates an identifier from its persisted numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Returns the persisted numeric representation.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Kind of contribution tracked by a [`ContributionRegistry`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationKind {
    /// A command palette or shell command.
    Command,
    /// A dockable editor panel.
    Panel,
}

/// Metadata used to register an editor command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandDescriptor {
    /// Stable command identifier.
    pub id: String,
    /// Human-readable command title.
    pub title: String,
    /// Optional default keyboard shortcut.
    pub default_key: Option<String>,
    /// Optional context expression in which the command is active.
    pub context: Option<String>,
}

/// Metadata used to register an editor panel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PanelDescriptor {
    /// Stable panel identifier.
    pub id: String,
    /// Human-readable panel title.
    pub title: String,
    /// Whether only one instance of the panel may exist.
    pub singleton: bool,
}

/// Error returned when a contribution identifier is already registered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContributionError {
    kind: RegistrationKind,
    id: String,
}

impl fmt::Display for ContributionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "duplicate {:?} contribution {}",
            self.kind, self.id
        )
    }
}

impl core::error::Error for ContributionError {}

/// Disposable token for one registered contribution.
pub struct Registration {
    id: ContributionId,
    kind: RegistrationKind,
}

impl Registration {
    /// Returns the stable identifier assigned to this registration.
    #[must_use]
    pub const fn id(&self) -> ContributionId {
        self.id
    }

    /// Returns the kind of contribution owned by this registration.
    #[must_use]
    pub const fn kind(&self) -> RegistrationKind {
        self.kind
    }
}

/// Provider of contributions that can be installed into an editor registry.
pub trait EditorModule {
    /// Returns the stable module identifier.
    fn id(&self) -> &str;
    /// Registers all contributions owned by this module.
    fn register(
        &self,
        registry: &mut ContributionRegistry,
    ) -> Result<Vec<Registration>, ContributionError>;
}

#[derive(Default)]
/// Registry of command and panel contribution metadata.
pub struct ContributionRegistry {
    commands: BTreeMap<String, (ContributionId, CommandDescriptor)>,
    panels: BTreeMap<String, (ContributionId, PanelDescriptor)>,
    next_id: u128,
}

impl ContributionRegistry {
    /// Registers a command descriptor and returns its disposable token.
    pub fn register_command(
        &mut self,
        descriptor: CommandDescriptor,
    ) -> Result<Registration, ContributionError> {
        if self.commands.contains_key(&descriptor.id) {
            return Err(ContributionError {
                kind: RegistrationKind::Command,
                id: descriptor.id,
            });
        }
        let registration = Registration {
            id: self.allocate_id(),
            kind: RegistrationKind::Command,
        };
        self.commands
            .insert(descriptor.id.clone(), (registration.id, descriptor));
        Ok(registration)
    }

    /// Registers a panel descriptor and returns its disposable token.
    pub fn register_panel(
        &mut self,
        descriptor: PanelDescriptor,
    ) -> Result<Registration, ContributionError> {
        if self.panels.contains_key(&descriptor.id) {
            return Err(ContributionError {
                kind: RegistrationKind::Panel,
                id: descriptor.id,
            });
        }
        let registration = Registration {
            id: self.allocate_id(),
            kind: RegistrationKind::Panel,
        };
        self.panels
            .insert(descriptor.id.clone(), (registration.id, descriptor));
        Ok(registration)
    }

    /// Removes the contribution represented by `registration`.
    pub fn unregister(&mut self, registration: Registration) -> bool {
        let before = self.commands.len() + self.panels.len();
        match registration.kind {
            RegistrationKind::Command => self.commands.retain(|_, (id, _)| *id != registration.id),
            RegistrationKind::Panel => self.panels.retain(|_, (id, _)| *id != registration.id),
        }
        before != self.commands.len() + self.panels.len()
    }

    /// Looks up a command descriptor by stable identifier.
    #[must_use]
    pub fn command(&self, id: &str) -> Option<&CommandDescriptor> {
        self.commands.get(id).map(|(_, descriptor)| descriptor)
    }

    /// Looks up a panel descriptor by stable identifier.
    #[must_use]
    pub fn panel(&self, id: &str) -> Option<&PanelDescriptor> {
        self.panels.get(id).map(|(_, descriptor)| descriptor)
    }

    /// Returns the number of registered commands.
    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Returns the number of registered panels.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    fn allocate_id(&mut self) -> ContributionId {
        self.next_id = self.next_id.saturating_add(1);
        ContributionId::from_raw(self.next_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registrations_are_disposable_and_duplicate_safe() {
        let mut registry = ContributionRegistry::default();
        let registration = registry
            .register_command(CommandDescriptor {
                id: "editor.save".into(),
                title: "Save".into(),
                default_key: Some("Ctrl+S".into()),
                context: None,
            })
            .expect("register");
        assert!(registry.command("editor.save").is_some());
        assert!(registry
            .register_command(CommandDescriptor {
                id: "editor.save".into(),
                title: "Duplicate".into(),
                default_key: None,
                context: None,
            })
            .is_err());
        assert!(registry.unregister(registration));
        assert_eq!(registry.command_count(), 0);
    }
}
