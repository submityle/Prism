//! `GPU` per-triangle cloth aerodynamics (wind drag/lift velocity pre-pass).
//!
//! Wind only reads as *cloth* when it pushes on the mesh face by face: a sheet
//! broadside to a gust catches far more force than one edge-on to it, so the
//! force must be computed *per triangle* from the face normal and then spread
//! across the face's three vertices as an external velocity increment, run once
//! before the `XPBD` substep prediction. The authoritative sequential model is
//! [`prism_physics_core`]'s `apply_aero_forces`; this module is its race-free
//! `GPU` twin.
//!
//! # Why two passes
//!
//! The sequential reference walks triangles in order and reads each vertex's
//! *already updated* velocity (a Gauss-Seidel coupling), which a compute kernel
//! cannot reproduce because warp order is undefined. The twin therefore runs
//! the **Jacobi** reformulation the hardware can actually execute: every face
//! force is computed from one frozen velocity snapshot, then each vertex gathers
//! the forces of its incident faces and writes only its own slot. The two agree
//! exactly when no vertex is shared, and the Jacobi form is the well-defined
//! reference the parity suite checks (identical to the scene-side
//! `prism_render_architecture` aero gather twin).
//!
//! - [`prep`] — the host-side deterministic build: triangle filtering, the
//!   per-triangle wind (steady field plus the integer-hashed turbulence), and
//!   the per-vertex incidence `CSR`, all in the golden's exact order.
//! - [`cpu`] — the [`cpu_cloth_aero`] golden twin (delegating the per-face force
//!   to [`prism_physics_core`]) plus an independent brute-force anchor.
//! - [`gpu`] — the real-device [`GpuClothAero`] two-pass pipeline.
//!
//! # Correctness model
//!
//! Triangle filtering and the incidence `CSR` are integer-exact (built on the
//! host in [`prep`]), so the only floating-point divergence is the per-face
//! drag/lift arithmetic (`GPU` fused multiply-add and division/square-root
//! rounding), and parity is verified within a tight tolerance rather than
//! bit-for-bit — the same model the rest of the solver uses.
//!
//! # Provenance
//!
//! The per-triangle drag/lift decomposition, the optional quadratic
//! dynamic-pressure term, and the integer-hash turbulence are standard,
//! publicly documented cloth-aerodynamics techniques. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};

pub mod cpu;
pub mod gpu;
pub mod prep;

pub use cpu::cpu_cloth_aero;
pub use gpu::GpuClothAero;
pub use prep::{build as build_cloth_aero_prep, ClothAeroPrep};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// The wind field and aerodynamic coefficients for one aero pass.
///
/// This is the `GPU` crate's flat, upload-friendly re-statement of
/// [`prism_physics_core`]'s `WindField` + `AeroParams` pair: the host sanitizes
/// those types and copies their scalars here before building the pass. The
/// per-triangle turbulence jitter is folded into each face's wind on the host
/// (see [`prep::build`]), so only the steady [`velocity`](Self::velocity) is
/// carried on this struct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothAeroParams {
    /// Steady world-space wind velocity (world units per second).
    pub velocity: [Real; 3],
    /// Per-triangle turbulence amplitude, clamped to `0..=1` on the host.
    pub turbulence: Real,
    /// Normal-direction (drag) coefficient; clamped non-negative on the host.
    pub drag: Real,
    /// In-plane (lift) coefficient; clamped non-negative on the host.
    pub lift: Real,
    /// Fluid (air) density: `<= 0` selects the linear `area * relative_wind`
    /// model, `> 0` the quadratic (airspeed-squared) model.
    pub air_density: Real,
}

impl ClothAeroParams {
    /// Builds parameters from a steady wind, turbulence, and drag/lift
    /// coefficients, leaving the air density at zero (linear model).
    #[must_use]
    pub const fn new(velocity: [Real; 3], turbulence: Real, drag: Real, lift: Real) -> Self {
        ClothAeroParams {
            velocity,
            turbulence,
            drag,
            lift,
            air_density: 0.0,
        }
    }

    /// Returns a copy opting into the quadratic (airspeed-squared) model with
    /// the given air density.
    #[must_use]
    pub const fn with_air_density(mut self, air_density: Real) -> Self {
        self.air_density = air_density;
        self
    }
}

/// A single mesh triangle for the aero pass, over raw particle indices.
///
/// Padded to a `vec4<u32>` so it maps directly to the device buffer stride; the
/// fourth lane is unused.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct ClothAeroTriangle {
    /// First corner particle index.
    pub i0: u32,
    /// Second corner particle index.
    pub i1: u32,
    /// Third corner particle index.
    pub i2: u32,
    /// Unused padding lane (always `0`).
    pub _pad: u32,
}

impl ClothAeroTriangle {
    /// Builds a triangle from three raw particle indices.
    #[must_use]
    pub const fn new(i0: u32, i1: u32, i2: u32) -> Self {
        ClothAeroTriangle {
            i0,
            i1,
            i2,
            _pad: 0,
        }
    }

    /// The three corner indices as a plain array.
    #[must_use]
    pub const fn indices(&self) -> [u32; 3] {
        [self.i0, self.i1, self.i2]
    }
}
