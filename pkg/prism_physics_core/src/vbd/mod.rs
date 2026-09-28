//! Vertex Block Descent (VBD) solver (milestone M7).
//!
//! VBD (Chen et al., SIGGRAPH 2024) minimises the implicit-Euler variational
//! energy by block coordinate descent, one vertex at a time. For each vertex a
//! local 3x3 system built from the incident elastic-energy Hessians is solved
//! and the vertex position is updated. The method is unconditionally stable,
//! handles very stiff materials and large deformation without the drift XPBD
//! shows at low iteration counts, and parallelises by graph colouring.
//!
//! This module is populated by milestone M7.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The VBD
//! energy formulation, per-vertex Gauss-Seidel block descent, and the spring
//! and stable-Neo-Hookean tetrahedral energies are implemented from standard,
//! publicly documented literature (Chen et al. 2024; Smith et al. 2018).
