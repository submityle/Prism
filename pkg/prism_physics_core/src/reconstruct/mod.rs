//! Particle surface reconstruction via Marching Cubes (milestone M5.5).
//!
//! Fluid and MPM solvers produce clouds of particles; to hand a renderable or
//! collidable surface to downstream systems we reconstruct an iso-surface. A
//! scalar field is sampled onto a grid from the particle set (a smoothed
//! density / signed-distance field), then the Marching Cubes algorithm
//! extracts a watertight triangle mesh at a chosen iso-level. The output is a
//! pure geometry buffer (positions + indices + normals); this module performs
//! **no rendering** and has zero dependency on any render backend.
//!
//! This module is populated by milestone M5.5.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! Marching Cubes edge/triangle tables and the particle-to-field kernels are
//! implemented from the standard, publicly documented literature (Lorensen &
//! Cline 1987; Zhu & Bridson 2005).
