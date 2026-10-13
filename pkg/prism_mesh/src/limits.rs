//! Device-independent mesh quantity limits.
//!
//! The render driver owns device limits.  This module narrows the subset that
//! a static mesh contract needs and gives the three count domains distinct
//! types so callers do not accidentally compare a stream count with a vertex
//! count.

use crate::error::MeshError;

/// Maximum number of vertices accepted by a mesh contract.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexLimit(u64);

impl VertexLimit {
    /// Creates a vertex limit.
    #[must_use]
    pub const fn new(maximum: u64) -> Self {
        Self(maximum)
    }

    /// The inclusive maximum.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Whether `count` is accepted.
    #[must_use]
    pub const fn accepts(self, count: u64) -> bool {
        count <= self.0
    }
}

/// Maximum number of vertex attributes accepted by a mesh contract.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AttributeLimit(u32);

impl AttributeLimit {
    /// Creates an attribute limit.
    #[must_use]
    pub const fn new(maximum: u32) -> Self {
        Self(maximum)
    }

    /// The inclusive maximum.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether `count` is accepted.
    #[must_use]
    pub const fn accepts(self, count: u32) -> bool {
        count <= self.0
    }
}

/// Maximum number of vertex streams accepted by a mesh contract.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StreamLimit(u32);

impl StreamLimit {
    /// Creates a stream limit.
    #[must_use]
    pub const fn new(maximum: u32) -> Self {
        Self(maximum)
    }

    /// The inclusive maximum.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether `count` is accepted.
    #[must_use]
    pub const fn accepts(self, count: u32) -> bool {
        count <= self.0
    }
}

/// Mesh-facing limits extracted from the RHI's device limits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshLimits {
    /// Maximum vertex count. Static mesh indices are `u32`, so the default is
    /// the largest addressable vertex count.
    pub max_vertices: VertexLimit,
    /// Maximum number of declared vertex attributes.
    pub max_attributes: AttributeLimit,
    /// Maximum number of vertex streams/buffers.
    pub max_streams: StreamLimit,
    /// Maximum material slot value plus one. A zero value rejects all slots.
    pub max_material_slots: u32,
}

impl MeshLimits {
    /// Constructs explicit mesh-facing limits.
    #[must_use]
    pub const fn new(max_vertices: u64, max_attributes: u32, max_streams: u32) -> Self {
        Self {
            max_vertices: VertexLimit::new(max_vertices),
            max_attributes: AttributeLimit::new(max_attributes),
            max_streams: StreamLimit::new(max_streams),
            max_material_slots: u32::MAX,
        }
    }

    /// Sets the maximum number of material slots and returns the updated
    /// limits.
    #[must_use]
    pub const fn with_material_slots(mut self, max_material_slots: u32) -> Self {
        self.max_material_slots = max_material_slots;
        self
    }

    /// The conservative render-driver portability profile.
    #[must_use]
    pub const fn baseline() -> Self {
        Self::new(u32::MAX as u64, 16, 8)
    }

    /// The higher-tier desktop render-driver profile.
    #[must_use]
    pub const fn desktop() -> Self {
        Self::new(u32::MAX as u64, 32, 16)
    }

    /// Extracts mesh limits from the RHI's device limits without coupling mesh
    /// data to a concrete backend.
    #[must_use]
    pub const fn from_render_limits(limits: prism_render_driver::Limits) -> Self {
        Self::new(
            u32::MAX as u64,
            limits.max_vertex_attributes,
            limits.max_vertex_buffers,
        )
    }

    /// Validates a vertex/attribute/stream count tuple.
    pub fn validate_counts(
        &self,
        vertex_count: usize,
        attribute_count: usize,
        stream_count: usize,
    ) -> Result<(), MeshError> {
        let vertex_count = u64::try_from(vertex_count).unwrap_or(u64::MAX);
        if !self.max_vertices.accepts(vertex_count) {
            return Err(MeshError::VertexLimitExceeded {
                actual: vertex_count,
                maximum: self.max_vertices.get(),
            });
        }

        let attribute_count = u32::try_from(attribute_count).unwrap_or(u32::MAX);
        if !self.max_attributes.accepts(attribute_count) {
            return Err(MeshError::AttributeLimitExceeded {
                actual: attribute_count,
                maximum: self.max_attributes.get(),
            });
        }

        let stream_count = u32::try_from(stream_count).unwrap_or(u32::MAX);
        if !self.max_streams.accepts(stream_count) {
            return Err(MeshError::StreamLimitExceeded {
                actual: stream_count,
                maximum: self.max_streams.get(),
            });
        }

        Ok(())
    }

    /// Validates a material slot against this profile.
    pub const fn validate_material_slot(&self, material_slot: u16) -> Result<(), MeshError> {
        if (material_slot as u32) < self.max_material_slots {
            Ok(())
        } else {
            Err(MeshError::MaterialSlotOutOfRange {
                slot: material_slot,
                maximum: self.max_material_slots,
            })
        }
    }
}

impl Default for MeshLimits {
    fn default() -> Self {
        Self::baseline()
    }
}
