//! Particle surface reconstruction via marching tetrahedra (milestone M5.5).
//!
//! Fluid and MPM solvers produce clouds of particles; to hand a renderable or
//! collidable surface to downstream systems we reconstruct an iso-surface. A
//! scalar field is sampled onto a regular grid from the particle set (a
//! smoothed density / blobby field), then a marching-tetrahedra pass extracts
//! a watertight triangle mesh at a chosen iso-level. The output is a pure
//! geometry buffer (positions + normals + indices); this module performs **no
//! rendering** and has zero dependency on any render backend.
//!
//! Marching tetrahedra (rather than the 256-case Marching Cubes table) is used
//! deliberately: the Freudenthal–Kuhn cube subdivision tiles space as one
//! consistent simplicial complex, so the extracted surface is guaranteed
//! edge-manifold and free of the ambiguous-face cracks that the raw MC table
//! can introduce between neighbouring cells.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! marching-tetrahedra extraction and case tables follow Doi & Koide 1991 and
//! Bourke, "Polygonising a scalar field using tetrahedrons"; the cube
//! subdivision is the classical Freudenthal–Kuhn triangulation; the
//! particle-to-field kernels follow Blinn 1982 and Zhu & Bridson 2005.

pub mod field;
pub mod marching_cubes;
pub mod mesh;
pub mod tables;

pub use field::ScalarField;
pub use marching_cubes::triangulate;
pub use mesh::SurfaceMesh;
