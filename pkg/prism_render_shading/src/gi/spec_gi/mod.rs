//! Specular / glossy global illumination: GGX VNDF lobe sampling, glossy
//! ReSTIR reservoirs with a directional target, and BRDF/light MIS.
//!
//! # Conventions
//! * Glossy reuse weights the stored sample by the destination's GGX lobe so a
//!   neighbour's radiance is re-targeted to the current view/roughness.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).

pub mod ggx_lobe;
pub mod glossy_reservoir;
pub mod brdf_mis;
