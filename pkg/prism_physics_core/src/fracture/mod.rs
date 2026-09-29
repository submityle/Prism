//! Convex Voronoi fracture: shatter a solid into simulation-ready fragments.
//!
//! This module carves a convex solid into a partition of convex cells, one per
//! scattered seed *site*, using bounded Voronoi decomposition. Each cell is an
//! intersection of half-spaces (the bounding shape plus one bisector per rival
//! site), measured with exact divergence-theorem volume and mass integrals, so
//! the resulting [`Fragment`]s can immediately become dynamic rigid bodies.
//!
//! # Layout
//!
//! - [`config`] — authoring tunables ([`FractureConfig`]).
//! - [`rng`] — the deterministic `xorshift64*` generator.
//! - [`predicates`] — epsilon-consistent sign tests and symbolic perturbation.
//! - [`plane`] — oriented planes, bisectors, and three-plane intersection.
//! - [`polyhedron`] — the convex [`ConvexPolyhedron`] and half-space carving.
//! - [`mass`] — rigid-body [`MassProperties`] of a convex solid.
//! - [`pattern`] — uniform and impact-clustered seed scattering.
//! - [`fragment`] — a carved [`Fragment`] (cell plus generating site).
//! - [`voronoi`] — the decomposition entry points.
//!
//! # Determinism
//!
//! Every step is a pure function of the [`FractureConfig`] seed and the input
//! geometry: the generator is fixed-algorithm, ties are broken by an
//! index-based symbolic perturbation, and no transcendental functions are used.
//! Identical inputs therefore produce byte-identical fragments across runs and
//! machines, which is required for baked caches and networked simulation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Bounded
//! Voronoi fracture by per-site half-space intersection, H-representation
//! vertex enumeration, and the divergence-theorem mass integrals are standard,
//! publicly documented computational-geometry results.

pub mod config;
pub mod fragment;
pub mod mass;
pub mod pattern;
pub mod plane;
pub mod polyhedron;
pub mod predicates;
pub mod rng;
pub mod voronoi;

pub use config::FractureConfig;
pub use fragment::Fragment;
pub use mass::MassProperties;
pub use pattern::{scatter_impact, scatter_uniform};
pub use plane::Plane;
pub use polyhedron::ConvexPolyhedron;
pub use rng::DeterministicRng;
pub use voronoi::{fracture_aabb, fracture_convex, shatter_box, shatter_box_impact};
