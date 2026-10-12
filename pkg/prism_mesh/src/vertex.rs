//! The vertex-attribute model: what a vertex is, and how a mesh's vertices are
//! laid out in a buffer.

use prism_render_driver::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

/// The per-attribute format for `position`, packed as three `f32`s.
const POSITION_FORMAT: VertexFormat = VertexFormat::Float32x3;
/// The per-attribute format for `normal`, packed as three `f32`s.
const NORMAL_FORMAT: VertexFormat = VertexFormat::Float32x3;
/// The per-attribute format for `uv`, packed as two `f32`s.
const UV_FORMAT: VertexFormat = VertexFormat::Float32x2;
/// The per-attribute format for `tangent`, packed as four `f32`s.
const TANGENT_FORMAT: VertexFormat = VertexFormat::Float32x4;

/// The fixed shader location for `position`.
const LOCATION_POSITION: u32 = 0;
/// The fixed shader location for `normal`.
const LOCATION_NORMAL: u32 = 1;
/// The fixed shader location for `uv`.
const LOCATION_UV: u32 = 2;
/// The fixed shader location for `tangent`.
const LOCATION_TANGENT: u32 = 3;

/// The byte size of a packed `position` attribute.
const POSITION_SIZE: usize = POSITION_FORMAT.size() as usize;
/// The byte size of a packed `normal` attribute.
const NORMAL_SIZE: usize = NORMAL_FORMAT.size() as usize;
/// The byte size of a packed `uv` attribute.
const UV_SIZE: usize = UV_FORMAT.size() as usize;

/// One vertex as an engine-neutral value.
///
/// Every field always exists in the struct; a [`VertexLayout`] decides which of
/// them are actually written to (and read from) a buffer. `tangent` is a
/// four-component vector so a handedness sign can ride in `w`; it is only packed
/// when the layout enables tangents.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Object-space normal, expected to be unit length.
    pub normal: [f32; 3],
    /// Texture coordinate, conventionally within `[0, 1]`.
    pub uv: [f32; 2],
    /// Tangent with handedness in `w`; used only when a layout enables it.
    pub tangent: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0; 3],
            uv: [0.0; 2],
            tangent: [0.0; 4],
        }
    }
}

impl Vertex {
    /// Builds a vertex from position, normal, and uv, leaving the tangent
    /// zeroed.
    #[must_use]
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            tangent: [0.0; 4],
        }
    }

    /// Returns this vertex with its tangent (and handedness in `w`) set.
    #[must_use]
    pub const fn with_tangent(mut self, tangent: [f32; 4]) -> Self {
        self.tangent = tangent;
        self
    }

    /// Appends this vertex's present attributes to `out` in the canonical order
    /// `position, normal, uv, tangent`, matching [`VertexLayout::to_rhi`].
    pub(crate) fn encode(&self, layout: VertexLayout, out: &mut Vec<u8>) {
        out.reserve(layout.stride() as usize);
        if layout.position {
            push_f32(out, &self.position);
        }
        if layout.normal {
            push_f32(out, &self.normal);
        }
        if layout.uv {
            push_f32(out, &self.uv);
        }
        if layout.tangent {
            push_f32(out, &self.tangent);
        }
    }

    /// Reads one vertex back out of `bytes`, the exact inverse of
    /// [`encode`](Vertex::encode).
    ///
    /// Attributes absent from `layout` are left at their [`Default`] value.
    pub(crate) fn decode(layout: VertexLayout, bytes: &[u8]) -> Self {
        let mut vertex = Self::default();
        let mut offset = 0usize;
        if layout.position {
            vertex.position = read_f32::<3>(bytes, offset);
            offset += POSITION_SIZE;
        }
        if layout.normal {
            vertex.normal = read_f32::<3>(bytes, offset);
            offset += NORMAL_SIZE;
        }
        if layout.uv {
            vertex.uv = read_f32::<2>(bytes, offset);
            offset += UV_SIZE;
        }
        if layout.tangent {
            vertex.tangent = read_f32::<4>(bytes, offset);
        }
        vertex
    }
}

/// Appends `values` to `out` as little-endian `f32`s.
fn push_f32(out: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

/// Reads `N` little-endian `f32`s starting at `offset`.
fn read_f32<const N: usize>(bytes: &[u8], offset: usize) -> [f32; N] {
    let mut out = [0.0f32; N];
    for (index, slot) in out.iter_mut().enumerate() {
        let start = offset + index * 4;
        let mut raw = [0u8; 4];
        raw.copy_from_slice(&bytes[start..start + 4]);
        *slot = f32::from_le_bytes(raw);
    }
    out
}

/// Which attributes a vertex carries.
///
/// Attributes are packed in the canonical order `position, normal, uv,
/// tangent`, and each present attribute keeps a fixed format and shader
/// location. A layout with only `position` and `uv` present therefore still
/// feeds `@location(2)` for `uv`, so a shader written against one layout keeps
/// working across the others.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexLayout {
    /// Whether a `position` attribute is present.
    pub position: bool,
    /// Whether a `normal` attribute is present.
    pub normal: bool,
    /// Whether a `uv` attribute is present.
    pub uv: bool,
    /// Whether a `tangent` attribute is present.
    pub tangent: bool,
}

impl VertexLayout {
    /// The layout the primitive generators produce: position, normal, and uv.
    pub const POSITION_NORMAL_UV: Self = Self {
        position: true,
        normal: true,
        uv: true,
        tangent: false,
    };

    /// The layout with position, normal, uv, and tangent.
    pub const POSITION_NORMAL_UV_TANGENT: Self = Self {
        position: true,
        normal: true,
        uv: true,
        tangent: true,
    };

    /// The byte stride between consecutive packed vertices.
    #[must_use]
    pub const fn stride(self) -> u64 {
        let mut stride = 0;
        if self.position {
            stride += POSITION_FORMAT.size();
        }
        if self.normal {
            stride += NORMAL_FORMAT.size();
        }
        if self.uv {
            stride += UV_FORMAT.size();
        }
        if self.tangent {
            stride += TANGENT_FORMAT.size();
        }
        stride
    }

    /// How many attributes the layout contains.
    #[must_use]
    pub const fn attribute_count(self) -> u32 {
        self.position as u32 + self.normal as u32 + self.uv as u32 + self.tangent as u32
    }

    /// Builds the RHI buffer layout that feeds the vertex stage.
    ///
    /// The offsets match the byte order [`Vertex::encode`] writes, so the
    /// resulting layout describes the packed vertex bytes exactly.
    #[must_use]
    pub fn to_rhi(self) -> VertexBufferLayout {
        let mut attributes = Vec::with_capacity(4);
        let mut offset = 0u64;
        if self.position {
            attributes.push(VertexAttribute {
                format: POSITION_FORMAT,
                offset,
                shader_location: LOCATION_POSITION,
            });
            offset += POSITION_FORMAT.size();
        }
        if self.normal {
            attributes.push(VertexAttribute {
                format: NORMAL_FORMAT,
                offset,
                shader_location: LOCATION_NORMAL,
            });
            offset += NORMAL_FORMAT.size();
        }
        if self.uv {
            attributes.push(VertexAttribute {
                format: UV_FORMAT,
                offset,
                shader_location: LOCATION_UV,
            });
            offset += UV_FORMAT.size();
        }
        if self.tangent {
            attributes.push(VertexAttribute {
                format: TANGENT_FORMAT,
                offset,
                shader_location: LOCATION_TANGENT,
            });
            offset += TANGENT_FORMAT.size();
        }
        VertexBufferLayout {
            array_stride: offset,
            step_mode: VertexStepMode::Vertex,
            attributes,
        }
    }
}
