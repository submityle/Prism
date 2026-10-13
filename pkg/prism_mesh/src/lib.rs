//! # `prism_mesh`
//!
//! The mesh asset: what a mesh *is*, independent of who draws it.
//!
//! A [`Mesh`] holds named vertex [`attributes`](MeshVertexAttribute) in typed
//! values, an optional [`Indices`] list at the width it was authored in, and a
//! [`PrimitiveTopology`]. From that it derives the three things a renderer needs:
//!
//! - the interleaved [`VertexBufferLayout`] a pipeline binds,
//! - the packed [`vertex-buffer bytes`](Mesh::vertex_buffer_data) to upload, and
//! - local-space [`bounds`](Mesh::bounds) for culling.
//!
//! It is deliberately the *data* half of the mesh pipeline: no ECS, no GPU
//! handles, no staging, so a mesh can be authored, validated, and asserted on in
//! a unit test. The renderer's GPU form (buffers, upload, release) is
//! `prism_render_app`'s `RenderMesh`, prepared through the render-asset pipeline.
//!
//! The vertex vocabulary is the **RHI's** ([`VertexFormat`], [`VertexBufferLayout`],
//! [`PrimitiveTopology`], [`IndexFormat`]): a mesh describes exactly what a
//! pipeline will bind, and a second set of enums here would be a second truth
//! that could drift from the one the GPU is configured with.
//!
//! [`VertexFormat`]: prism_render_driver::VertexFormat
//! [`VertexBufferLayout`]: prism_render_driver::VertexBufferLayout
//! [`PrimitiveTopology`]: prism_render_driver::PrimitiveTopology
//! [`IndexFormat`]: prism_render_driver::IndexFormat

#![forbid(unsafe_code)]

extern crate alloc;

mod attribute;
mod bounds;
mod error;
mod hash;
mod indices;
mod limits;
mod mesh;
mod morph;
pub mod primitives;
mod skin;
mod streams;
mod submesh;
mod topology;

#[cfg(test)]
mod tests;

pub use attribute::{MeshVertexAttribute, VertexAttributeValues};
pub use bounds::MeshBounds;
pub use error::MeshError;
pub use hash::{canonical_bytes, content_hash, MeshContentHash, MESH_HASH_SCHEMA};
pub use indices::Indices;
pub use limits::{AttributeLimit, MeshLimits, StreamLimit, VertexLimit};
pub use mesh::{Mesh, POSITION_ATTRIBUTE};
pub use morph::{MorphDelta, MorphError, MorphTarget, MorphTargetData};
pub use skin::{SkinError, SkinInfluence, SkinningData};
pub use streams::{StreamAttribute, StreamLayoutProfile, VertexStreamKind, VertexStreamLayout};
pub use submesh::{validate_submeshes, validate_submeshes_with_limits, SubMesh};
pub use topology::{elements_per_primitive, primitive_count, PrimitiveTopologyExt};
