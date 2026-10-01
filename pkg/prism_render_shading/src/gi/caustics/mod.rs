//! Caustics: photon splatting + manifold next-event estimation for specular-to-
//! diffuse light transport (water / glass focused light).
//!
//! The caustic reference is split into three cooperating pieces:
//!
//! * [`photon`] — flux-carrying [`Photon`](photon::Photon)s and their transport
//!   through smooth dielectric interfaces: Snell [`refract`](photon::refract) /
//!   mirror [`reflect`](photon::reflect) with total-internal-reflection
//!   handling, and an exact-Fresnel energy split
//!   ([`split_at_interface`](photon::split_at_interface) /
//!   [`scatter_at_interface`](photon::scatter_at_interface)).
//! * [`density_estimate`] — bounded (compact-support) kernel density estimation
//!   that splats each photon's flux over a footprint
//!   ([`Kernel`](density_estimate::Kernel),
//!   [`kernel_density`](density_estimate::kernel_density),
//!   [`estimate_irradiance`](density_estimate::estimate_irradiance)); the kernel
//!   integrates to one so total energy is conserved.
//! * [`manifold_nee`] — manifold next-event estimation that Newton-solves the
//!   half-vector constraint for the specular connection point through a
//!   [`SpecularPlane`](manifold_nee::SpecularPlane)
//!   ([`solve_manifold`](manifold_nee::solve_manifold)), with Jacobian clamping
//!   and a finite convergence-failure fall-back.
//!
//! # Conventions
//! * Photons carry flux; density estimation uses a bounded kernel footprint.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).

pub mod density_estimate;
pub mod manifold_nee;
pub mod photon;
