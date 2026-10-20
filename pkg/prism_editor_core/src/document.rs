use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

/// Stable identity for an open editor document.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct DocumentId(u128);

impl DocumentId {
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

/// Stable identity for one document layer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct LayerId(u128);

impl LayerId {
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

/// Semantic kind of a document layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    /// Authoritative base content.
    Base,
    /// User or tool override content.
    Override,
    /// Runtime-observed content that is not editable by default.
    Runtime,
}

/// Metadata and routing state for one ordered document layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocumentLayer {
    id: LayerId,
    name: String,
    kind: LayerKind,
    visible: bool,
    editable: bool,
}

impl DocumentLayer {
    /// Creates a visible layer, editable unless it is a runtime layer.
    #[must_use]
    pub fn new(id: LayerId, name: impl Into<String>, kind: LayerKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            visible: true,
            editable: !matches!(kind, LayerKind::Runtime),
        }
    }

    /// Returns the layer identity.
    #[must_use]
    pub const fn id(&self) -> LayerId {
        self.id
    }

    /// Returns the display name of the layer.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the semantic layer kind.
    #[must_use]
    pub const fn kind(&self) -> LayerKind {
        self.kind
    }

    /// Returns whether the layer is visible.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Changes the layer visibility.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Returns whether commands may edit the layer.
    #[must_use]
    pub const fn is_editable(&self) -> bool {
        self.editable
    }

    /// Changes whether commands may edit the layer.
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
    }
}

/// Type-independent document state shared by UI, automation, and adapters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocumentState {
    id: DocumentId,
    layers: Vec<DocumentLayer>,
    active_layer: Option<LayerId>,
    revision: u64,
    dirty: bool,
}

impl DocumentState {
    /// Creates an empty document with revision zero and no dirty changes.
    #[must_use]
    pub fn new(id: DocumentId) -> Self {
        Self {
            id,
            layers: Vec::new(),
            active_layer: None,
            revision: 0,
            dirty: false,
        }
    }

    /// Returns the document identity.
    #[must_use]
    pub const fn id(&self) -> DocumentId {
        self.id
    }

    /// Returns layers in insertion order.
    #[must_use]
    pub fn layers(&self) -> &[DocumentLayer] {
        &self.layers
    }

    /// Returns the currently active layer, if any.
    #[must_use]
    pub const fn active_layer(&self) -> Option<LayerId> {
        self.active_layer
    }

    /// Returns the monotonically increasing document revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns whether the document has unsaved changes.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Adds a layer while enforcing uniqueness and the base-layer-first rule.
    pub fn add_layer(&mut self, layer: DocumentLayer) -> Result<(), DocumentError> {
        if self.layers.iter().any(|entry| entry.id == layer.id) {
            return Err(DocumentError::DuplicateLayer(layer.id));
        }
        if self.layers.is_empty() && !matches!(layer.kind, LayerKind::Base) {
            return Err(DocumentError::FirstLayerMustBeBase);
        }
        if self.active_layer.is_none() {
            self.active_layer = Some(layer.id);
        }
        self.layers.push(layer);
        self.touch();
        Ok(())
    }

    /// Removes a non-base layer and returns its previous value.
    pub fn remove_layer(&mut self, id: LayerId) -> Result<DocumentLayer, DocumentError> {
        let index = self
            .layers
            .iter()
            .position(|layer| layer.id == id)
            .ok_or(DocumentError::MissingLayer(id))?;
        if matches!(self.layers[index].kind, LayerKind::Base) {
            return Err(DocumentError::CannotRemoveBase);
        }
        let removed = self.layers.remove(index);
        if self.active_layer == Some(id) {
            self.active_layer = self.layers.last().map(DocumentLayer::id);
        }
        self.touch();
        Ok(removed)
    }

    /// Selects an existing layer as active.
    pub fn set_active_layer(&mut self, id: LayerId) -> Result<(), DocumentError> {
        if !self.layers.iter().any(|layer| layer.id == id) {
            return Err(DocumentError::MissingLayer(id));
        }
        self.active_layer = Some(id);
        self.touch();
        Ok(())
    }

    /// Clears the dirty flag without changing the revision.
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    /// Marks the document dirty and advances its revision.
    pub fn mark_dirty(&mut self) {
        self.touch();
    }

    fn touch(&mut self) {
        self.revision = self.revision.saturating_add(1);
        self.dirty = true;
    }
}

/// Failure produced by an invalid document or layer operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DocumentError {
    /// A layer with the same identity is already present.
    DuplicateLayer(LayerId),
    /// The first layer must be a base layer.
    FirstLayerMustBeBase,
    /// The requested layer does not exist.
    MissingLayer(LayerId),
    /// The base layer is required and cannot be removed.
    CannotRemoveBase,
}

impl fmt::Display for DocumentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateLayer(id) => write!(formatter, "layer {} already exists", id.raw()),
            Self::FirstLayerMustBeBase => {
                formatter.write_str("the first layer must be a base layer")
            }
            Self::MissingLayer(id) => write!(formatter, "layer {} does not exist", id.raw()),
            Self::CannotRemoveBase => formatter.write_str("the base layer cannot be removed"),
        }
    }
}

impl core::error::Error for DocumentError {}

/// A deterministic metadata index useful to adapters that need stable names.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DocumentMetadata {
    values: BTreeMap<String, String>,
}

impl DocumentMetadata {
    /// Inserts or replaces a string value for `key`.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.values.insert(key.into(), value.into());
    }

    /// Returns the value associated with `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_requires_base_layer_and_tracks_dirty_state() {
        let mut document = DocumentState::new(DocumentId::from_raw(1));
        assert_eq!(
            document.add_layer(DocumentLayer::new(
                LayerId::from_raw(2),
                "override",
                LayerKind::Override,
            )),
            Err(DocumentError::FirstLayerMustBeBase)
        );
        document
            .add_layer(DocumentLayer::new(
                LayerId::from_raw(1),
                "base",
                LayerKind::Base,
            ))
            .expect("base layer");
        assert!(document.is_dirty());
        document.mark_saved();
        assert!(!document.is_dirty());
    }
}
