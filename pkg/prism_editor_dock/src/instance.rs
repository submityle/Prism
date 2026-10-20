//! Panel-instance lifecycle adaptation.

use prism_ui_dock::{
    PanelInstance, PanelInstanceId, PanelInstanceRegistry, PanelInstanceRegistryError,
};

use crate::DockContributionRegistry;

/// A lifecycle operation rejected by the dock adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PanelLifecycleError {
    /// The panel ID is not mapped into the dock registry.
    UnknownPanel(String),
    /// The underlying deterministic instance registry rejected the operation.
    Registry(PanelInstanceRegistryError),
}

impl std::fmt::Display for PanelLifecycleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPanel(id) => write!(formatter, "panel is not mapped into dock: {id}"),
            Self::Registry(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PanelLifecycleError {}

/// The pure-data result of a panel lifecycle transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PanelLifecycleEvent {
    /// A new stable instance was opened.
    Created(PanelInstance),
    /// An open instance was closed but retained for restoration.
    Closed(PanelInstance),
    /// A retained instance was reopened.
    Restored(PanelInstance),
}

/// Adapter that makes dock instance policy available to the editor layer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PanelLifecycleAdapter {
    registry: PanelInstanceRegistry,
}

impl PanelLifecycleAdapter {
    /// Creates an empty lifecycle adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            registry: PanelInstanceRegistry::new(),
        }
    }

    /// Opens an instance for a mapped panel.
    pub fn create(
        &mut self,
        panels: &DockContributionRegistry,
        panel_id: &str,
    ) -> Result<PanelLifecycleEvent, PanelLifecycleError> {
        if !panels.dock_registry().contains(panel_id) {
            return Err(PanelLifecycleError::UnknownPanel(panel_id.into()));
        }
        self.registry
            .create(panels.dock_registry(), panel_id)
            .map(PanelLifecycleEvent::Created)
            .map_err(PanelLifecycleError::Registry)
    }

    /// Closes an instance while retaining its stable identity.
    pub fn close(
        &mut self,
        instance_id: PanelInstanceId,
    ) -> Result<PanelLifecycleEvent, PanelLifecycleError> {
        self.registry
            .close(instance_id)
            .map(PanelLifecycleEvent::Closed)
            .map_err(PanelLifecycleError::Registry)
    }

    /// Restores a retained instance.
    pub fn restore(
        &mut self,
        instance_id: PanelInstanceId,
    ) -> Result<PanelLifecycleEvent, PanelLifecycleError> {
        self.registry
            .restore(instance_id)
            .map(PanelLifecycleEvent::Restored)
            .map_err(PanelLifecycleError::Registry)
    }

    /// Returns the retained instance registry for snapshotting.
    #[must_use]
    pub const fn registry(&self) -> &PanelInstanceRegistry {
        &self.registry
    }

    /// Replaces the retained instance registry after the owner validates it.
    pub(crate) fn replace_registry(&mut self, registry: PanelInstanceRegistry) {
        self.registry = registry;
    }

    /// Returns an instance by stable identity.
    #[must_use]
    pub fn get(&self, instance_id: PanelInstanceId) -> Option<&PanelInstance> {
        self.registry.get(instance_id)
    }
}

impl From<PanelInstanceRegistryError> for PanelLifecycleError {
    fn from(error: PanelInstanceRegistryError) -> Self {
        Self::Registry(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_editor_core::PanelDescriptor;

    #[test]
    fn close_and_restore_retain_identity() {
        let mut panels = DockContributionRegistry::new();
        panels
            .register(&PanelDescriptor {
                id: "logs".into(),
                title: "Logs".into(),
                singleton: false,
            })
            .unwrap();
        let mut lifecycle = PanelLifecycleAdapter::new();
        let PanelLifecycleEvent::Created(instance) = lifecycle.create(&panels, "logs").unwrap()
        else {
            panic!("create must emit Created");
        };
        let id = instance.id();
        assert!(matches!(
            lifecycle.close(id),
            Ok(PanelLifecycleEvent::Closed(_))
        ));
        assert!(
            matches!(lifecycle.restore(id), Ok(PanelLifecycleEvent::Restored(restored)) if restored.id() == id)
        );
    }

    #[test]
    fn singleton_policy_is_enforced_by_the_adapter() {
        let mut panels = DockContributionRegistry::new();
        panels
            .register(&PanelDescriptor {
                id: "inspector".into(),
                title: "Inspector".into(),
                singleton: true,
            })
            .unwrap();
        let mut lifecycle = PanelLifecycleAdapter::new();
        lifecycle.create(&panels, "inspector").unwrap();
        assert!(matches!(
            lifecycle.create(&panels, "inspector"),
            Err(PanelLifecycleError::Registry(
                PanelInstanceRegistryError::SingletonAlreadyExists { .. }
            ))
        ));
    }
}
