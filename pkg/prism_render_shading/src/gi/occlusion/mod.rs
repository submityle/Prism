//! Bent-normal and specular-occlusion reconstruction for GI (CPU golden).
//!
//! A scalar ambient-occlusion (AO) term throws away *where* the sky is still
//! visible.  This module reconstructs that directional information as a golden,
//! GPU-free numerical reference the WESL/Metal twin must reproduce:
//!
//! * [`bent_normal`] fits a cone of unoccluded directions — a bent-normal
//!   axis, a half-angle aperture, and a scalar visibility — to a shading
//!   point's hemispherical visibility samples, following GTAO (Jimenez et al.
//!   2016) and *Ambient Aperture Lighting* (Oat & Sander 2007).
//! * [`specular_occlusion`] remaps diffuse AO into a view- and
//!   roughness-dependent specular-occlusion factor (Lagarde & de Rousiers,
//!   *Moving Frostbite to PBR*, 2014) and derives a horizon-occlusion term for
//!   reflections from the bent-normal cone, sharing a cone-overlap primitive.
//!
//! Every item is a deterministic, allocation-free pure function with unit
//! tests covering its boundary, degenerate, and numerical-property behaviour.

pub mod bent_normal;
pub mod specular_occlusion;

pub use bent_normal::{accumulate_bent_normal, bent_normal_from_cosine_hemisphere, BentNormalCone};
pub use specular_occlusion::{
    cone_cone_intersection, horizon_occlusion, specular_occlusion,
};
