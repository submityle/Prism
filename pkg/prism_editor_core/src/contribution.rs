use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ContributionId(u128);

impl ContributionId {
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationKind {
    Command,
    Panel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandDescriptor {
    pub id: String,
    pub title: String,
    pub default_key: Option<String>,
    pub context: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PanelDescriptor {
    pub id: String,
    pub title: String,
    pub singleton: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContributionError {
    kind: RegistrationKind,
    id: String,
}

impl std::fmt::Display for ContributionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "duplicate {:?} contribution {}",
            self.kind, self.id
        )
    }
}

impl std::error::Error for ContributionError {}

pub struct Registration {
    id: ContributionId,
    kind: RegistrationKind,
}

impl Registration {
    #[must_use]
    pub const fn id(&self) -> ContributionId {
        self.id
    }

    #[must_use]
    pub const fn kind(&self) -> RegistrationKind {
        self.kind
    }
}

pub trait EditorModule {
    fn id(&self) -> &str;
    fn register(
        &self,
        registry: &mut ContributionRegistry,
    ) -> Result<Vec<Registration>, ContributionError>;
}

#[derive(Default)]
pub struct ContributionRegistry {
    commands: BTreeMap<String, (ContributionId, CommandDescriptor)>,
    panels: BTreeMap<String, (ContributionId, PanelDescriptor)>,
    next_id: u128,
}

impl ContributionRegistry {
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

    pub fn unregister(&mut self, registration: Registration) -> bool {
        let before = self.commands.len() + self.panels.len();
        match registration.kind {
            RegistrationKind::Command => self.commands.retain(|_, (id, _)| *id != registration.id),
            RegistrationKind::Panel => self.panels.retain(|_, (id, _)| *id != registration.id),
        }
        before != self.commands.len() + self.panels.len()
    }

    #[must_use]
    pub fn command(&self, id: &str) -> Option<&CommandDescriptor> {
        self.commands.get(id).map(|(_, descriptor)| descriptor)
    }

    #[must_use]
    pub fn panel(&self, id: &str) -> Option<&PanelDescriptor> {
        self.panels.get(id).map(|(_, descriptor)| descriptor)
    }

    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

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
