//! Vertex buffer layout description.

use crate::format::VertexFormat;
use alloc::vec::Vec;

/// How vertex buffer data advances across invocations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum VertexStepMode {
    /// Advance once per vertex.
    #[default]
    Vertex,
    /// Advance once per instance.
    Instance,
}

/// One attribute within a vertex buffer: its format, byte offset, and the
/// shader location it feeds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexAttribute {
    /// The scalar/vector format of the attribute.
    pub format: VertexFormat,
    /// The byte offset of the attribute within its vertex.
    pub offset: u64,
    /// The `@location(n)` the attribute binds to in the shader.
    pub shader_location: u32,
}

/// The layout of one vertex buffer: its stride, step mode, and attributes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VertexBufferLayout {
    /// The byte stride between consecutive elements.
    pub array_stride: u64,
    /// Whether elements advance per vertex or per instance.
    pub step_mode: VertexStepMode,
    /// The attributes packed into each element.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    /// Builds a tightly packed layout from a sequence of formats, assigning
    /// sequential shader locations starting at `first_location` and computing
    /// each offset from the preceding formats' sizes. The resulting
    /// `array_stride` is the sum of all attribute sizes.
    #[must_use]
    pub fn packed(
        step_mode: VertexStepMode,
        first_location: u32,
        formats: &[VertexFormat],
    ) -> Self {
        let mut attributes = Vec::with_capacity(formats.len());
        let mut offset = 0u64;
        for (i, &format) in formats.iter().enumerate() {
            attributes.push(VertexAttribute {
                format,
                offset,
                shader_location: first_location + u32::try_from(i).expect("location fits u32"),
            });
            offset += format.size();
        }
        Self {
            array_stride: offset,
            step_mode,
            attributes,
        }
    }

    /// Whether every attribute lies fully within `array_stride`. A layout that
    /// fails this check would read out of bounds and backends must reject it.
    #[must_use]
    pub fn is_within_stride(&self) -> bool {
        self.attributes
            .iter()
            .all(|attr| attr.offset + attr.format.size() <= self.array_stride)
    }
}
