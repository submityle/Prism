//! Uninhabited domain markers shared by the string [`Interner`](super::Interner)
//! and the content-addressed [`InternCache`](super::InternCache).
//!
//! Each marker is a separate, uninhabited type (no values ever exist). A
//! handle is generic over its marker, so handles from different domains are
//! **distinct types** and the compiler rejects mixing, for example, an
//! asset-path handle with a tag handle even though both are a `u32` index
//! under the hood. Markers live only inside a `PhantomData` and cost nothing
//! at runtime.
//!
//! Domain separation is also the mechanism for bounded lifetime: each domain
//! is a separate [`Interner`](super::Interner) / [`InternCache`](super::InternCache)
//! instance, so reclaiming a whole domain is just dropping (or
//! [`clear`](super::InternCache::clear)-ing) that one table, which keeps any
//! single interning table from growing without bound (design doc §23 risk).

/// General-purpose tags / names (the default engine "name" domain, mirroring
/// Unreal's `FName`).
#[derive(Debug)]
pub enum Tag {}

/// Filesystem- or asset-style paths.
#[derive(Debug)]
pub enum Path {}

/// Human-readable debug labels.
#[derive(Debug)]
pub enum Debug {}

/// Type identities (the domain `prism_reflect` keys its registry by).
#[derive(Debug)]
pub enum Type {}

/// Asset identifiers (keys for the content-addressed asset cache).
#[derive(Debug)]
pub enum Asset {}

/// Deduplicated mesh blocks (geometry content addressing).
#[derive(Debug)]
pub enum Mesh {}

/// Deduplicated texture blocks (image content addressing).
#[derive(Debug)]
pub enum Texture {}
