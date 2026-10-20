use std::collections::BTreeMap;

/// Stable identity for an open editor document.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct DocumentId(u128);

impl DocumentId {
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

/// Stable identity for one document layer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct LayerId(u128);

impl LayerId {
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
pub enum LayerKind {
    Base,
    Override,
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

    #[must_use]
    pub const fn id(&self) -> LayerId {
        self.id
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub const fn kind(&self) -> LayerKind {
        self.kind
    }

    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    #[must_use]
    pub const fn is_editable(&self) -> bool {
        self.editable
    }

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

    #[must_use]
    pub const fn id(&self) -> DocumentId {
        self.id
    }

    #[must_use]
    pub fn layers(&self) -> &[DocumentLayer] {
        &self.layers
    }

    #[must_use]
    pub const fn active_layer(&self) -> Option<LayerId> {
        self.active_layer
    }

    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.dirty
    }

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

    pub fn set_active_layer(&mut self, id: LayerId) -> Result<(), DocumentError> {
        if !self.layers.iter().any(|layer| layer.id == id) {
            return Err(DocumentError::MissingLayer(id));
        }
        self.active_layer = Some(id);
        self.touch();
        Ok(())
    }

    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    pub fn mark_dirty(&mut self) {
        self.touch();
    }

    fn touch(&mut self) {
        self.revision = self.revision.saturating_add(1);
        self.dirty = true;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DocumentError {
    DuplicateLayer(LayerId),
    FirstLayerMustBeBase,
    MissingLayer(LayerId),
    CannotRemoveBase,
}

impl std::fmt::Display for DocumentError {
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

impl std::error::Error for DocumentError {}

/// A deterministic metadata index useful to adapters that need stable names.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DocumentMetadata {
    values: BTreeMap<String, String>,
}

impl DocumentMetadata {
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.values.insert(key.into(), value.into());
    }

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
