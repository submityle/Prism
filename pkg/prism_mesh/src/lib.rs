//! # `prism_mesh`
//!
//! Engine-neutral mesh geometry for Prism's own renderer: the piece that
//! replaces `bevy_mesh`'s mesh types on the RHI render path. It knows nothing
//! about any ECS, scene, or GPU backend — a [`Mesh`] is just packed vertex
//! bytes, index data, and the [`VertexLayout`] describing them.
//!
//! ## Layout
//! - [`vertex`]: the vertex-attribute model ([`Vertex`], [`VertexLayout`]).
//! - [`mesh`]: [`Mesh`] — packed vertex bytes plus indices and a layout.
//! - [`primitives`]: deterministic [`plane`](primitives::plane),
//!   [`cube`](primitives::cube), [`uv_sphere`](primitives::uv_sphere), and
//!   [`cylinder`](primitives::cylinder) generators.
//! - [`gpu`]: [`GpuMesh`] and the RHI upload.
//!
//! The vertex layout is the single source of truth: it drives both how
//! [`Vertex`] values are packed into bytes and the
//! [`prism_render_driver::VertexBufferLayout`] the vertex stage is fed, so the
//! two can never drift apart.
#![forbid(unsafe_code)]

mod gpu;
mod mesh;
pub mod primitives;
mod vertex;

pub use gpu::GpuMesh;
pub use mesh::Mesh;
pub use vertex::{Vertex, VertexLayout};

#[cfg(test)]
mod tests;
