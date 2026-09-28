//! FLIP/APIC free-surface fluid solver on a MAC grid (milestone M5.5).
//!
//! This solver simulates incompressible liquid with marker particles advected
//! through a staggered Marker-And-Cell (MAC) velocity grid. Each step splats
//! particle velocities to the grid (P2G), applies body forces, enforces
//! incompressibility with a pressure projection (a discrete Poisson solve),
//! then interpolates the corrected velocity field back to the particles using
//! a PIC/FLIP blend (optionally the APIC affine transfer) for low numerical
//! dissipation and lively splashes.
//!
//! This module is populated by milestone M5.5.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The MAC
//! grid, PIC/FLIP/APIC transfers, and pressure projection are implemented from
//! standard, publicly documented computational-fluid-dynamics literature
//! (Bridson, "Fluid Simulation for Computer Graphics"; Zhu & Bridson 2005).
