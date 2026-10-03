//! Hot-reload event descriptors for authoring data.
//!
//! When a sound designer edits an authored asset (an event, a container, a
//! bank, or a patch), the engine must react without a restart. This module
//! defines the serializable [`HotReloadEvent`] contract that names which asset
//! changed and how. It deliberately stops at the contract: deciding to
//! recompile a subgraph and atomically swapping it at a block boundary is the
//! engine's responsibility. The one piece of policy expressed here is
//! [`HotReloadEvent::requires_exec_plan_rebuild`], which reports whether a
//! change touches graph structure (and therefore needs a recompiled plan) or is
//! a lighter metadata change that can be applied in place.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the authoring-data hot-reload trigger of design section 38. A
//! structural change requests the recompiled plan of design section 29 and the
//! atomic block-boundary swap of design section 30; this module only carries
//! the request and never performs the recompilation itself.

/// Which kind of authored asset a reload concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AssetKind {
    /// A playable event (trigger metadata and action list).
    Event,
    /// A container that structures how child sounds are chosen or layered.
    Container,
    /// A bank of loaded media and its lookup tables.
    Bank,
    /// A node-graph patch (an effect chain or routing fragment).
    Patch,
}

impl AssetKind {
    /// Returns `true` when editing this kind of asset changes graph structure.
    ///
    /// Containers, banks, and patches shape the render graph, so a change to
    /// them needs a recompiled plan. Events carry trigger metadata that can be
    /// re-indexed in place.
    #[must_use]
    pub const fn is_graph_structural(self) -> bool {
        match self {
            AssetKind::Container | AssetKind::Bank | AssetKind::Patch => true,
            AssetKind::Event => false,
        }
    }
}

/// What happened to an asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ChangeKind {
    /// The asset was newly created.
    Added,
    /// The asset's contents changed.
    Modified,
    /// The asset was deleted.
    Removed,
}

/// Stable identifier for an authored asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AssetId(pub u64);

/// A single authoring-data reload request.
///
/// It names the asset, its kind, the change, and a monotonically increasing
/// revision so a consumer can order and deduplicate reload requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HotReloadEvent {
    /// Which asset changed.
    pub asset: AssetId,
    /// What kind of asset it is.
    pub kind: AssetKind,
    /// How it changed.
    pub change: ChangeKind,
    /// A monotonically increasing revision for ordering and deduplication.
    pub revision: u64,
}

impl HotReloadEvent {
    /// Builds a reload event.
    #[must_use]
    pub const fn new(
        asset: AssetId,
        kind: AssetKind,
        change: ChangeKind,
        revision: u64,
    ) -> Self {
        Self {
            asset,
            kind,
            change,
            revision,
        }
    }

    /// Returns `true` when applying this change requires a recompiled plan.
    ///
    /// A removal of any asset drops nodes from the graph and so needs a
    /// rebuild; an addition or modification needs a rebuild only when the asset
    /// is graph-structural (see [`AssetKind::is_graph_structural`]).
    #[must_use]
    pub const fn requires_exec_plan_rebuild(&self) -> bool {
        match self.change {
            ChangeKind::Removed => true,
            ChangeKind::Added | ChangeKind::Modified => self.kind.is_graph_structural(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_modification_is_in_place() {
        let ev = HotReloadEvent::new(AssetId(1), AssetKind::Event, ChangeKind::Modified, 1);
        assert!(!ev.requires_exec_plan_rebuild());
    }

    #[test]
    fn patch_modification_rebuilds() {
        let ev = HotReloadEvent::new(AssetId(2), AssetKind::Patch, ChangeKind::Modified, 2);
        assert!(ev.requires_exec_plan_rebuild());
    }

    #[test]
    fn container_add_rebuilds() {
        let ev = HotReloadEvent::new(AssetId(3), AssetKind::Container, ChangeKind::Added, 3);
        assert!(ev.requires_exec_plan_rebuild());
    }

    #[test]
    fn any_removal_rebuilds() {
        let ev = HotReloadEvent::new(AssetId(4), AssetKind::Event, ChangeKind::Removed, 4);
        assert!(ev.requires_exec_plan_rebuild());
    }

    #[test]
    fn structural_classification() {
        assert!(AssetKind::Container.is_graph_structural());
        assert!(AssetKind::Bank.is_graph_structural());
        assert!(AssetKind::Patch.is_graph_structural());
        assert!(!AssetKind::Event.is_graph_structural());
    }

    #[test]
    fn fields_round_trip() {
        let ev = HotReloadEvent::new(AssetId(9), AssetKind::Bank, ChangeKind::Modified, 42);
        assert_eq!(ev.asset, AssetId(9));
        assert_eq!(ev.kind, AssetKind::Bank);
        assert_eq!(ev.change, ChangeKind::Modified);
        assert_eq!(ev.revision, 42);
    }
}
