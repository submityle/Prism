//! Surface cache (Lumen-style surfel radiance cache): persistent per-surfel
//! irradiance with temporal integration, atlas addressing, and spatial filtering.
//!
//! # Conventions
//! * A surfel is a world-space oriented disc; radiance is accumulated by a
//!   confidence-weighted exponential moving average with disocclusion reset.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).
//!
//! # Modules
//! * [`surfel`] — the world-space oriented-disc primitive ([`surfel::Surfel`])
//!   plus its coverage / weight kernels (radial + off-plane + normal-agreement)
//!   and the geometric reuse weight shared with the spatial filter.
//! * [`atlas`] — deterministic surfel-atlas addressing: a tile allocator mapping
//!   surfel id to atlas tile, and per-surfel octahedral direction storage that
//!   reuses [`crate::gi::world_space::octahedral`].
//! * [`integration`] — confidence-weighted temporal EMA with disocclusion reset
//!   and a bilateral spatial filter over geometrically compatible neighbours.
//! * [`gpu`] — on-device producer twins (`WESL` compute kernels), their
//!   `repr(C)` host/device `ABI`, and the scalar `CPU` mirrors that pin each
//!   kernel bit-for-bit to the golden above.

pub mod atlas;
pub mod gpu;
pub mod integration;
pub mod surfel;

pub use atlas::{AtlasTexel, SurfelAtlas, TileCoord};
pub use integration::{
    confidence_alpha, integrate_radiance, is_disoccluded, spatial_filter, SurfelCacheEntry,
    TemporalParams,
};
pub use surfel::{normal_consistency, CoverageParams, Surfel};
