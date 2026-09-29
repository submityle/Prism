//! Incompressible grid-fluid, combustion, and vorticity-confinement contracts
//! for next-generation pyro/smoke effects (design §10).
//!
//! This is the `CPU`-verifiable contract layer for the voxel-fluid link that a
//! future `GPU` compute backend fills in. It models — but does not fully
//! implement as production kernels — the per-frame stable-fluids pipeline used
//! by `Houdini` Pyro, `EmberGen`, and Unreal `Niagara`'s fluid solver:
//!
//! 1. **Advection** — move the velocity (and scalar) fields along themselves,
//!    either plain semi-Lagrangian back-tracing or a `MacCormack` correction
//!    that cancels most of the first-order numerical diffusion.
//! 2. **Diffusion** — a viscous relaxation of the velocity field.
//! 3. **Pressure projection** — solve the pressure Poisson equation so the
//!    velocity field becomes (near) divergence-free, in an approximate `Jacobi`
//!    tier or an exact multigrid tier, then subtract the pressure gradient.
//! 4. **Velocity write-back** — the projected field becomes next frame's input
//!    and is sampled to advect the visible pyro particles.
//!
//! On top of the plain solver this module adds a three-channel combustion
//! coupling (temperature / fuel / smoke) that drives buoyancy and a black-body
//! self-emission approximation (design §17), plus a vorticity-confinement force
//! that reinjects the small-scale curl the advection step numerically damps.
//!
//! Determinism rules match the sibling particle modules: the only floating-point
//! primitive beyond ordinary arithmetic is `sqrt` (through [`Vec3`]); there are
//! no transcendental calls, so combustion falloff and black-body colour use
//! polynomial (multiply-only) approximations rather than `exp`/`pow`. The result
//! is bit-reproducible against a `GPU` kernel and free of the hash-`RNG` used by
//! the emission stages. Time steps are expected to respect the advection `CFL`
//! limit (see [`stable_timestep`]).

use alloc::vec;
use alloc::vec::Vec;

use super::stages::FluidStage;
use super::{Vec3, EPS_LEN_SQ};

/// Resolution of a uniform voxel fluid grid (design §10).
///
/// The cinematic default is a cubic `128³` domain; the quality ladder scales
/// this up to `256³` or down to `64³` (see [`fluid_quality_profile`]). Counts
/// are stored per axis so anisotropic domains (a tall smoke column) are also
/// expressible.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GridResolution {
    /// Voxel count along X.
    pub nx: u32,
    /// Voxel count along Y.
    pub ny: u32,
    /// Voxel count along Z.
    pub nz: u32,
}

impl GridResolution {
    /// The cinematic-default cubic `128³` grid.
    pub const CINEMATIC_128: Self = Self::uniform(128);

    /// Builds a cubic grid with `n` voxels on every axis.
    #[must_use]
    pub const fn uniform(n: u32) -> Self {
        Self {
            nx: n,
            ny: n,
            nz: n,
        }
    }

    /// Builds a grid from explicit per-axis counts.
    #[must_use]
    pub const fn new(nx: u32, ny: u32, nz: u32) -> Self {
        Self { nx, ny, nz }
    }

    /// Total voxel count, saturating instead of overflowing so a `256³` grid
    /// (~16.7M voxels) and larger never wrap.
    #[must_use]
    pub fn voxel_count(self) -> u32 {
        self.nx.saturating_mul(self.ny).saturating_mul(self.nz)
    }

    /// Row-major linear index of voxel `(x, y, z)`.
    ///
    /// The caller is responsible for supplying in-range coordinates; this is a
    /// pure index computation used by the field samplers and the pressure
    /// solve.
    #[must_use]
    pub fn linear_index(self, x: u32, y: u32, z: u32) -> u32 {
        (z * self.ny + y) * self.nx + x
    }
}

/// How the advection stage moves fields along the flow (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AdvectionScheme {
    /// First-order semi-Lagrangian back-trace: unconditionally stable but
    /// numerically diffusive (smoke detail smears out over time).
    SemiLagrangian,
    /// `MacCormack` (`BFECC`-style) forward/back correction on top of the
    /// semi-Lagrangian trace: cancels most of the first-order diffusion so
    /// smoke keeps its crisp edges, at the cost of two extra samples.
    MacCormack,
}

impl AdvectionScheme {
    /// Whether the scheme actively cancels numerical diffusion (`true` for
    /// `MacCormack`).
    #[must_use]
    pub fn corrects_numerical_diffusion(self) -> bool {
        matches!(self, AdvectionScheme::MacCormack)
    }

    /// Number of semi-Lagrangian sampling passes the scheme performs per step
    /// (one for [`AdvectionScheme::SemiLagrangian`], three for the forward /
    /// back / correct sequence of [`AdvectionScheme::MacCormack`]).
    #[must_use]
    pub fn sampling_passes(self) -> u32 {
        match self {
            AdvectionScheme::SemiLagrangian => 1,
            AdvectionScheme::MacCormack => 3,
        }
    }
}

/// Which pressure-projection solver a quality tier uses (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProjectionMethod {
    /// Fixed-iteration `Jacobi` relaxation of the pressure Poisson equation:
    /// cheap and trivially parallel, but only an *approximate* projection at a
    /// practical iteration budget (residual left non-zero).
    JacobiApprox,
    /// A multigrid V-cycle that drives the residual down to tolerance in far
    /// fewer passes: the *exact* tier used for hero shots. Its smoother is
    /// still a `Jacobi` relaxation, so it emits the same per-voxel stage
    /// primitive.
    Multigrid,
}

impl ProjectionMethod {
    /// Whether the method solves the projection to tolerance (`true` for
    /// [`ProjectionMethod::Multigrid`]) rather than stopping at a fixed
    /// iteration count.
    #[must_use]
    pub fn is_exact(self) -> bool {
        matches!(self, ProjectionMethod::Multigrid)
    }

    /// A sensible default relaxation-iteration budget for the method.
    ///
    /// The multigrid tier needs far fewer top-level relaxations than the flat
    /// `Jacobi` tier to reach the same residual.
    #[must_use]
    pub fn default_iterations(self) -> u32 {
        match self {
            ProjectionMethod::JacobiApprox => 40,
            ProjectionMethod::Multigrid => 8,
        }
    }

    /// The per-voxel [`FluidStage`] primitive this method dispatches during the
    /// projection. Both tiers relax with a `Jacobi` sweep, so both map to
    /// [`FluidStage::PressureJacobi`].
    #[must_use]
    pub fn relaxation_stage(self) -> FluidStage {
        FluidStage::PressureJacobi
    }
}

/// The stopping contract for a pressure-projection solve (design §10).
///
/// A solve runs until it either performs `max_iterations` relaxations or drives
/// the `L2` residual to at most `residual_tolerance`, whichever comes first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionPlan {
    /// The solver family (approximate `Jacobi` or exact multigrid).
    pub method: ProjectionMethod,
    /// Hard cap on relaxation iterations.
    pub max_iterations: u32,
    /// Residual (`L2` norm of `∇²p − div`) at or below which the solve stops.
    pub residual_tolerance: f32,
}

impl ProjectionPlan {
    /// Builds a plan from a method plus its default iteration budget and a
    /// residual tolerance.
    #[must_use]
    pub fn new(method: ProjectionMethod, residual_tolerance: f32) -> Self {
        Self {
            method,
            max_iterations: method.default_iterations(),
            residual_tolerance,
        }
    }
}

/// A logical field the fluid solver reads or writes (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FluidField {
    /// The primary velocity field (one [`Vec3`] per voxel).
    Velocity,
    /// The scratch velocity field written by advection before diffusion.
    VelocityScratch,
    /// The scalar pressure field solved during projection.
    Pressure,
    /// The scalar velocity divergence field.
    Divergence,
    /// The combustion temperature field.
    Temperature,
    /// The combustion fuel field.
    Fuel,
    /// The advected smoke-density field.
    Smoke,
}

/// The read/write field contract of one [`FluidStage`] (design §10).
///
/// Slices are `'static` because the read/write sets of each stage are fixed at
/// compile time; a scheduler uses them to place the minimal barriers between
/// passes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageFieldIo {
    /// Fields the stage reads.
    pub reads: &'static [FluidField],
    /// Fields the stage writes.
    pub writes: &'static [FluidField],
}

/// Returns the read/write field contract for a fluid stage (design §10).
///
/// This pins down the input/output fields of every pass in the
/// [`FluidStage`] schedule so a dependency-tracking scheduler never has to
/// guess which buffers a pass touches.
#[must_use]
pub fn stage_field_io(stage: FluidStage) -> StageFieldIo {
    match stage {
        FluidStage::AddForces => StageFieldIo {
            reads: &[
                FluidField::Velocity,
                FluidField::Temperature,
                FluidField::Smoke,
            ],
            writes: &[FluidField::Velocity],
        },
        FluidStage::Advect => StageFieldIo {
            reads: &[FluidField::Velocity],
            writes: &[FluidField::VelocityScratch],
        },
        FluidStage::Diffuse => StageFieldIo {
            reads: &[FluidField::VelocityScratch],
            writes: &[FluidField::Velocity],
        },
        FluidStage::ComputeDivergence => StageFieldIo {
            reads: &[FluidField::Velocity],
            writes: &[FluidField::Divergence],
        },
        FluidStage::PressureJacobi => StageFieldIo {
            reads: &[FluidField::Pressure, FluidField::Divergence],
            writes: &[FluidField::Pressure],
        },
        FluidStage::SubtractGradient => StageFieldIo {
            reads: &[FluidField::Pressure, FluidField::Velocity],
            writes: &[FluidField::Velocity],
        },
    }
}

/// The advection `CFL` number `|v|·dt / h` for a max speed, step, and cell size.
///
/// Semi-Lagrangian advection is stable for any number, but keeping it near or
/// below one bounds the back-trace to roughly a single cell and preserves
/// detail; the projection and combustion falloffs assume the same.
#[must_use]
pub fn cfl_number(max_velocity: f32, dt: f32, cell_size: f32) -> f32 {
    if cell_size.abs() > EPS_LEN_SQ {
        (max_velocity * dt) / cell_size
    } else {
        0.0
    }
}

/// The largest stable time step for a target `CFL` number.
///
/// Solves `|v|·dt / h ≤ cfl_target` for `dt`; a zero max speed yields a zero
/// step (nothing is moving) rather than dividing by zero.
#[must_use]
pub fn stable_timestep(max_velocity: f32, cell_size: f32, cfl_target: f32) -> f32 {
    if max_velocity.abs() > EPS_LEN_SQ {
        (cfl_target * cell_size) / max_velocity
    } else {
        0.0
    }
}

/// Semi-Lagrangian back-trace: the position a parcel arriving at `pos` came
/// from one step ago, `pos − v·dt` (design §10).
///
/// Multiply-add only, so it is bit-reproducible on the `GPU`.
#[must_use]
pub fn semi_lagrangian_backtrace(pos: Vec3, velocity: Vec3, dt: f32) -> Vec3 {
    pos.sub(velocity.scale(dt))
}

/// The `MacCormack` diffusion correction.
///
/// Given the forward-advected value, the original value, and the value obtained
/// by advecting the forward result back again, returns
/// `forward + 0.5·(original − back)`. This cancels most of the first-order
/// error the plain semi-Lagrangian trace introduces. Multiply-add only.
#[must_use]
pub fn maccormack_corrected(forward: Vec3, original: Vec3, back_advected: Vec3) -> Vec3 {
    forward.add(original.sub(back_advected).scale(0.5))
}

/// Trilinear interpolation weights for the eight corners of the voxel enclosing
/// a fractional position (design §10).
///
/// `frac` holds the in-cell fractions on `[0, 1]` per axis. The returned array
/// is indexed by a corner bitmask (bit 0 = X, bit 1 = Y, bit 2 = Z) and sums to
/// one. Multiply-only.
#[must_use]
pub fn trilinear_weights(frac: Vec3) -> [f32; 8] {
    let (fx, fy, fz) = (frac.x, frac.y, frac.z);
    let (gx, gy, gz) = (1.0 - fx, 1.0 - fy, 1.0 - fz);
    [
        gx * gy * gz,
        fx * gy * gz,
        gx * fy * gz,
        fx * fy * gz,
        gx * gy * fz,
        fx * gy * fz,
        gx * fy * fz,
        fx * fy * fz,
    ]
}

/// Trilinearly blends eight corner velocities by precomputed `weights`
/// (design §10).
///
/// Corner order matches [`trilinear_weights`]. Multiply-add only, so a
/// particle sampling the velocity field never invokes a transcendental.
#[must_use]
pub fn trilinear_sample(corners: [Vec3; 8], weights: [f32; 8]) -> Vec3 {
    corners
        .iter()
        .zip(weights.iter())
        .fold(Vec3::ZERO, |acc, (corner, weight)| {
            acc.add(corner.scale(*weight))
        })
}

/// Samples a voxel velocity field at grid-space position `pos` (design §10).
///
/// `field` is row-major with `res.voxel_count()` entries. The position is
/// clamped into the grid and the eight surrounding voxels are blended with
/// [`trilinear_sample`]; out-of-range or empty grids return [`Vec3::ZERO`].
/// This is the contract a pyro particle uses to be advected by the flow.
#[must_use]
pub fn sample_velocity_field(field: &[Vec3], res: GridResolution, pos: Vec3) -> Vec3 {
    let count = res.voxel_count() as usize;
    if count == 0 || field.len() < count {
        return Vec3::ZERO;
    }
    let max_x = res.nx - 1;
    let max_y = res.ny - 1;
    let max_z = res.nz - 1;
    let gx = clamp_scalar(pos.x, 0.0, max_x as f32);
    let gy = clamp_scalar(pos.y, 0.0, max_y as f32);
    let gz = clamp_scalar(pos.z, 0.0, max_z as f32);
    let x0f = gx.floor();
    let y0f = gy.floor();
    let z0f = gz.floor();
    let frac = Vec3::new(gx - x0f, gy - y0f, gz - z0f);
    let x0 = x0f as u32;
    let y0 = y0f as u32;
    let z0 = z0f as u32;
    let x1 = (x0 + 1).min(max_x);
    let y1 = (y0 + 1).min(max_y);
    let z1 = (z0 + 1).min(max_z);
    let fetch = |x: u32, y: u32, z: u32| field[res.linear_index(x, y, z) as usize];
    let corners = [
        fetch(x0, y0, z0),
        fetch(x1, y0, z0),
        fetch(x0, y1, z0),
        fetch(x1, y1, z0),
        fetch(x0, y0, z1),
        fetch(x1, y0, z1),
        fetch(x0, y1, z1),
        fetch(x1, y1, z1),
    ];
    trilinear_sample(corners, trilinear_weights(frac))
}

/// Integrates a particle one step under an already-sampled velocity, `pos +
/// v·dt` (design §10). Multiply-add only.
#[must_use]
pub fn advect_particle(pos: Vec3, velocity: Vec3, dt: f32) -> Vec3 {
    pos.add(velocity.scale(dt))
}

/// Advects a particle by sampling a voxel velocity field at its position and
/// integrating one step (design §10).
///
/// Composes [`sample_velocity_field`] with [`advect_particle`]; the whole path
/// is multiply-add plus the sampler's `floor`, so smoke motion is
/// bit-reproducible on the `GPU`.
#[must_use]
pub fn advect_particle_in_field(pos: Vec3, field: &[Vec3], res: GridResolution, dt: f32) -> Vec3 {
    let velocity = sample_velocity_field(field, res, pos);
    advect_particle(pos, velocity, dt)
}

/// Six axis-aligned scalar neighbor samples around a voxel, used for central
/// differences (design §10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeighborScalars {
    /// Sample at `+X`.
    pub x_plus: f32,
    /// Sample at `−X`.
    pub x_minus: f32,
    /// Sample at `+Y`.
    pub y_plus: f32,
    /// Sample at `−Y`.
    pub y_minus: f32,
    /// Sample at `+Z`.
    pub z_plus: f32,
    /// Sample at `−Z`.
    pub z_minus: f32,
}

/// Central-difference gradient of a scalar field from six neighbor samples.
///
/// `inv_2h` is `1 / (2·h)` for cell size `h`. Multiply-add only. Used both for
/// the pressure gradient in projection and for the temperature gradient that
/// drives heat-haze distortion.
#[must_use]
pub fn central_gradient(n: NeighborScalars, inv_2h: f32) -> Vec3 {
    Vec3::new(
        (n.x_plus - n.x_minus) * inv_2h,
        (n.y_plus - n.y_minus) * inv_2h,
        (n.z_plus - n.z_minus) * inv_2h,
    )
}

/// Removes the divergent part of a velocity by subtracting the pressure
/// gradient, `v − ∇p` (design §10). This is the velocity write-back contract.
#[must_use]
pub fn subtract_pressure_gradient(velocity: Vec3, pressure_gradient: Vec3) -> Vec3 {
    velocity.sub(pressure_gradient)
}

/// The result of a pressure-projection solve (design §10).
#[derive(Clone, Debug, PartialEq)]
pub struct PressureSolveResult {
    /// The solved pressure field, row-major, one scalar per voxel.
    pub pressure: Vec<f32>,
    /// The final `L2` residual `‖∇²p − div‖`.
    pub residual: f32,
    /// How many relaxation iterations actually ran.
    pub iterations_run: u32,
}

/// `L2` residual of a pressure field against a divergence field (design §10).
///
/// Computes `sqrt(mean((div − ∇²p)²))` with a unit cell size and homogeneous
/// (`p = 0`) Dirichlet boundaries. A shrinking residual across iterations is
/// the convergence signal for the `Jacobi`/multigrid solve.
#[must_use]
pub fn pressure_residual_l2(pressure: &[f32], divergence: &[f32], res: GridResolution) -> f32 {
    let count = res.voxel_count() as usize;
    if count == 0 || pressure.len() < count || divergence.len() < count {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let center = pressure[idx];
                let laplacian = neighbor_sum(pressure, res, x, y, z) - 6.0 * center;
                let r = divergence[idx] - laplacian;
                sum_sq += r * r;
            }
        }
    }
    (sum_sq / count as f32).sqrt()
}

/// Solves the pressure Poisson equation with damped `Jacobi` relaxation
/// (design §10).
///
/// Discretizes `∇²p = div` on a unit-spaced grid with homogeneous Dirichlet
/// boundaries, so each sweep sets `p ← (Σ neighbors − div) / 6`. The solve
/// stops when it hits `plan.max_iterations` or the `L2` residual falls to
/// `plan.residual_tolerance`. This is the approximate `Jacobi` tier; the
/// multigrid tier reaches the same residual in fewer top-level sweeps but is
/// contract-compatible (same inputs and outputs). Multiply-add plus one `sqrt`
/// in the residual.
#[must_use]
pub fn jacobi_pressure_solve(
    divergence: &[f32],
    res: GridResolution,
    plan: ProjectionPlan,
) -> PressureSolveResult {
    let count = res.voxel_count() as usize;
    if count == 0 || divergence.len() < count {
        return PressureSolveResult {
            pressure: Vec::new(),
            residual: 0.0,
            iterations_run: 0,
        };
    }
    let mut pressure = vec![0.0f32; count];
    let mut residual = pressure_residual_l2(&pressure, divergence, res);
    let mut iterations_run = 0;
    while iterations_run < plan.max_iterations && residual > plan.residual_tolerance {
        let mut next = vec![0.0f32; count];
        for z in 0..res.nz {
            for y in 0..res.ny {
                for x in 0..res.nx {
                    let idx = res.linear_index(x, y, z) as usize;
                    next[idx] = (neighbor_sum(&pressure, res, x, y, z) - divergence[idx]) / 6.0;
                }
            }
        }
        pressure = next;
        iterations_run += 1;
        residual = pressure_residual_l2(&pressure, divergence, res);
    }
    PressureSolveResult {
        pressure,
        residual,
        iterations_run,
    }
}

/// Three-channel combustion state at one voxel (design §10, §17).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CombustionState {
    /// Temperature (drives buoyancy and black-body emission).
    pub temperature: f32,
    /// Remaining fuel available to burn.
    pub fuel: f32,
    /// Accumulated smoke density.
    pub smoke: f32,
}

impl CombustionState {
    /// Builds a combustion state.
    #[must_use]
    pub const fn new(temperature: f32, fuel: f32, smoke: f32) -> Self {
        Self {
            temperature,
            fuel,
            smoke,
        }
    }
}

/// Parameters coupling the temperature / fuel / smoke channels (design §10).
///
/// All falloffs are linear so no `exp`/`pow` is needed: fuel burns at a fixed
/// rate once the ignition temperature is reached, each unit of burned fuel adds
/// `heat_yield` to temperature and `smoke_yield` to smoke, and temperature
/// relaxes toward ambient at a linear (`Newton`-style) cooling rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CombustionParams {
    /// Temperature at or above which fuel ignites.
    pub ignition_temperature: f32,
    /// Fuel burned per unit time while ignited.
    pub burn_rate: f32,
    /// Smoke produced per unit of burned fuel.
    pub smoke_yield: f32,
    /// Temperature added per unit of burned fuel.
    pub heat_yield: f32,
    /// Linear cooling coefficient toward ambient.
    pub cooling_rate: f32,
    /// Ambient temperature the field relaxes toward.
    pub ambient_temperature: f32,
    /// Upward buoyancy per degree above ambient.
    pub buoyancy_alpha: f32,
    /// Downward drag per unit of smoke density (heavy soot sinks).
    pub buoyancy_beta: f32,
}

/// Advances one combustion voxel by `dt` (design §10).
///
/// Burns fuel (clamped to what remains) when the voxel is at or above the
/// ignition temperature, converts it to smoke and heat, then applies linear
/// cooling toward ambient. Multiply-add only — no `exp`/`pow`.
#[must_use]
pub fn step_combustion(
    state: CombustionState,
    params: CombustionParams,
    dt: f32,
) -> CombustionState {
    let mut temperature = state.temperature;
    let mut fuel = state.fuel;
    let mut smoke = state.smoke;
    if temperature >= params.ignition_temperature && fuel > 0.0 {
        let mut burned = params.burn_rate * dt;
        if burned > fuel {
            burned = fuel;
        }
        fuel -= burned;
        smoke += burned * params.smoke_yield;
        temperature += burned * params.heat_yield;
    }
    let cooled = params.cooling_rate * dt * (temperature - params.ambient_temperature);
    temperature -= cooled;
    CombustionState {
        temperature,
        fuel,
        smoke,
    }
}

/// The buoyancy body force for a combustion voxel (design §10).
///
/// Hot voxels rise (`+Y`) proportional to their temperature above ambient;
/// dense smoke drags the parcel back down. Returned as a force injected during
/// the [`FluidStage::AddForces`] pass. Multiply-add only.
#[must_use]
pub fn buoyancy_force(state: CombustionState, params: CombustionParams) -> Vec3 {
    let lift = (state.temperature - params.ambient_temperature) * params.buoyancy_alpha
        - state.smoke * params.buoyancy_beta;
    Vec3::new(0.0, lift, 0.0)
}

/// Parameters for the polynomial black-body emission approximation
/// (design §17).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissionParams {
    /// Temperature at which the fire just starts to glow (emission ≈ 0).
    pub low_temperature: f32,
    /// Temperature at which the glow reaches full white-hot.
    pub white_temperature: f32,
    /// Overall emissive intensity scale fed to the shading model.
    pub intensity_scale: f32,
}

/// Approximates black-body self-emission colour from temperature (design §17).
///
/// Maps temperature onto `[0, 1]` between the low and white points, then ramps
/// red first, green next, and blue last (the red→orange→yellow→white glow of
/// a `Houdini`/`EmberGen` fire) using multiply-only polynomials instead of a
/// Planck `exp`. The returned colour is premultiplied by the emissive
/// intensity; every channel is monotonic in temperature.
#[must_use]
pub fn blackbody_emission(temperature: f32, params: EmissionParams) -> Vec3 {
    let span = params.white_temperature - params.low_temperature;
    let t = if span > EPS_LEN_SQ {
        clamp01((temperature - params.low_temperature) / span)
    } else {
        0.0
    };
    let intensity = t * params.intensity_scale;
    let r = clamp01(t * 1.6);
    let g = clamp01(t * t * 1.3);
    let b = clamp01(t * t * t);
    Vec3::new(r, g, b).scale(intensity)
}

/// Screen-space heat-haze refraction offset from a temperature gradient
/// (design §10, §17).
///
/// Hot air bends light; the distortion offset is proportional to the local
/// temperature gradient scaled by `strength`. Multiply only.
#[must_use]
pub fn heat_haze_distortion(temperature_gradient: Vec3, strength: f32) -> Vec3 {
    temperature_gradient.scale(strength)
}

/// Six axis-aligned velocity neighbor samples around a voxel, for the finite
/// difference curl (design §10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeighborVelocities {
    /// Velocity sample at `+X`.
    pub x_plus: Vec3,
    /// Velocity sample at `−X`.
    pub x_minus: Vec3,
    /// Velocity sample at `+Y`.
    pub y_plus: Vec3,
    /// Velocity sample at `−Y`.
    pub y_minus: Vec3,
    /// Velocity sample at `+Z`.
    pub z_plus: Vec3,
    /// Velocity sample at `−Z`.
    pub z_minus: Vec3,
}

impl NeighborVelocities {
    /// Central-difference curl (vorticity) `ω = ∇ × v` at the voxel center
    /// (design §10).
    ///
    /// `inv_2h` is `1 / (2·h)`. Multiply-add only. This is the vorticity
    /// estimate the confinement force restores after advection damps it.
    #[must_use]
    pub fn curl(self, inv_2h: f32) -> Vec3 {
        let dwz_dy = self.y_plus.z - self.y_minus.z;
        let dvy_dz = self.z_plus.y - self.z_minus.y;
        let dux_dz = self.z_plus.x - self.z_minus.x;
        let dwz_dx = self.x_plus.z - self.x_minus.z;
        let dvy_dx = self.x_plus.y - self.x_minus.y;
        let dux_dy = self.y_plus.x - self.y_minus.x;
        Vec3::new(
            (dwz_dy - dvy_dz) * inv_2h,
            (dux_dz - dwz_dx) * inv_2h,
            (dvy_dx - dux_dy) * inv_2h,
        )
    }
}

/// Strength of the vorticity-confinement force (design §10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VorticityParams {
    /// Confinement coefficient `ε`: how hard curl detail is pushed back in.
    pub epsilon: f32,
}

/// The vorticity-confinement force injected into the velocity field
/// (design §10).
///
/// Restores small-scale swirl that advection numerically dissipated, keeping
/// smoke rolls crisp. The location force is `ε·h·(N × ω)` where `N` is the
/// normalized gradient of `‖ω‖` (pointing toward higher vorticity) and `ω` is
/// the local curl. `N` is built with [`Vec3::normalize_or_zero`] so a flat
/// vorticity region contributes no force instead of a `NaN`. Multiply-add plus
/// the normalize `sqrt`.
#[must_use]
pub fn vorticity_confinement_force(
    curl: Vec3,
    magnitude_gradient: Vec3,
    params: VorticityParams,
    cell_size: f32,
) -> Vec3 {
    let n = magnitude_gradient.normalize_or_zero();
    n.cross(curl).scale(params.epsilon * cell_size)
}

/// A rendering-driven quality tier for the fluid solve (design §10, §28).
///
/// Ordered coarsest-first. The tier selects grid resolution, projection solver,
/// advection scheme, and whether vorticity confinement runs, so the whole
/// pipeline degrades together under budget pressure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FluidQuality {
    /// Lowest cost: coarse `64³` grid, approximate `Jacobi` projection, plain
    /// semi-Lagrangian advection, no vorticity confinement.
    Low,
    /// Balanced `96³` grid, approximate `Jacobi` projection, plain advection,
    /// vorticity confinement on.
    Medium,
    /// Cinematic `128³` grid, exact multigrid projection, `MacCormack`
    /// advection, vorticity confinement on.
    High,
    /// Hero-shot `256³` grid, exact multigrid projection, `MacCormack`
    /// advection, vorticity confinement on.
    Ultra,
}

impl FluidQuality {
    /// Fidelity rank (`0` lowest).
    #[must_use]
    pub fn rank(self) -> u8 {
        match self {
            FluidQuality::Low => 0,
            FluidQuality::Medium => 1,
            FluidQuality::High => 2,
            FluidQuality::Ultra => 3,
        }
    }

    /// The next lower tier, or [`None`] at [`FluidQuality::Low`] (one rung of
    /// the degradation staircase).
    #[must_use]
    pub fn degrade(self) -> Option<Self> {
        match self {
            FluidQuality::Ultra => Some(FluidQuality::High),
            FluidQuality::High => Some(FluidQuality::Medium),
            FluidQuality::Medium => Some(FluidQuality::Low),
            FluidQuality::Low => None,
        }
    }
}

/// The resolved solver configuration for a [`FluidQuality`] tier (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FluidQualityProfile {
    /// Voxel grid resolution.
    pub resolution: GridResolution,
    /// Pressure-projection solver family.
    pub projection: ProjectionMethod,
    /// Relaxation-iteration budget for the projection.
    pub projection_iterations: u32,
    /// Advection scheme.
    pub advection: AdvectionScheme,
    /// Whether vorticity confinement runs this tier.
    pub vorticity_enabled: bool,
}

/// Maps a [`FluidQuality`] tier to its solver configuration (design §10).
///
/// This is the degradation matrix: resolution, projection tier, advection
/// scheme, and vorticity toggle all step down together as quality drops, so a
/// budget-limited frame produces a coherent (if softer) sim rather than an
/// inconsistent mix.
#[must_use]
pub fn fluid_quality_profile(quality: FluidQuality) -> FluidQualityProfile {
    match quality {
        FluidQuality::Ultra => FluidQualityProfile {
            resolution: GridResolution::uniform(256),
            projection: ProjectionMethod::Multigrid,
            projection_iterations: ProjectionMethod::Multigrid.default_iterations(),
            advection: AdvectionScheme::MacCormack,
            vorticity_enabled: true,
        },
        FluidQuality::High => FluidQualityProfile {
            resolution: GridResolution::CINEMATIC_128,
            projection: ProjectionMethod::Multigrid,
            projection_iterations: ProjectionMethod::Multigrid.default_iterations(),
            advection: AdvectionScheme::MacCormack,
            vorticity_enabled: true,
        },
        FluidQuality::Medium => FluidQualityProfile {
            resolution: GridResolution::uniform(96),
            projection: ProjectionMethod::JacobiApprox,
            projection_iterations: ProjectionMethod::JacobiApprox.default_iterations(),
            advection: AdvectionScheme::SemiLagrangian,
            vorticity_enabled: true,
        },
        FluidQuality::Low => FluidQualityProfile {
            resolution: GridResolution::uniform(64),
            projection: ProjectionMethod::JacobiApprox,
            projection_iterations: ProjectionMethod::JacobiApprox.default_iterations(),
            advection: AdvectionScheme::SemiLagrangian,
            vorticity_enabled: false,
        },
    }
}

/// Clamps a scalar into `[lo, hi]`.
fn clamp_scalar(v: f32, lo: f32, hi: f32) -> f32 {
    v.clamp(lo, hi)
}

/// Clamps a scalar into `[0, 1]`.
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Sum of the six face-neighbor scalars around `(x, y, z)` with homogeneous
/// (`0`) Dirichlet boundaries outside the grid.
fn neighbor_sum(field: &[f32], res: GridResolution, x: u32, y: u32, z: u32) -> f32 {
    let mut sum = 0.0;
    if x + 1 < res.nx {
        sum += field[res.linear_index(x + 1, y, z) as usize];
    }
    if x > 0 {
        sum += field[res.linear_index(x - 1, y, z) as usize];
    }
    if y + 1 < res.ny {
        sum += field[res.linear_index(x, y + 1, z) as usize];
    }
    if y > 0 {
        sum += field[res.linear_index(x, y - 1, z) as usize];
    }
    if z + 1 < res.nz {
        sum += field[res.linear_index(x, y, z + 1) as usize];
    }
    if z > 0 {
        sum += field[res.linear_index(x, y, z - 1) as usize];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const F_EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < F_EPS
    }

    #[test]
    fn grid_resolution_counts_and_index() {
        let res = GridResolution::CINEMATIC_128;
        assert_eq!(res.voxel_count(), 128 * 128 * 128);
        let small = GridResolution::new(4, 3, 2);
        assert_eq!(small.voxel_count(), 24);
        assert_eq!(small.linear_index(0, 0, 0), 0);
        assert_eq!(small.linear_index(3, 2, 1), 23);
    }

    #[test]
    fn voxel_count_saturates_instead_of_overflowing() {
        let huge = GridResolution::uniform(u32::MAX);
        assert_eq!(huge.voxel_count(), u32::MAX);
    }

    #[test]
    fn advection_scheme_flags_and_passes() {
        assert!(!AdvectionScheme::SemiLagrangian.corrects_numerical_diffusion());
        assert!(AdvectionScheme::MacCormack.corrects_numerical_diffusion());
        assert_eq!(AdvectionScheme::SemiLagrangian.sampling_passes(), 1);
        assert_eq!(AdvectionScheme::MacCormack.sampling_passes(), 3);
    }

    #[test]
    fn projection_method_exactness_and_stage() {
        assert!(!ProjectionMethod::JacobiApprox.is_exact());
        assert!(ProjectionMethod::Multigrid.is_exact());
        assert!(
            ProjectionMethod::JacobiApprox.default_iterations()
                > ProjectionMethod::Multigrid.default_iterations()
        );
        assert_eq!(
            ProjectionMethod::Multigrid.relaxation_stage(),
            FluidStage::PressureJacobi
        );
        assert_eq!(
            ProjectionMethod::JacobiApprox.relaxation_stage(),
            FluidStage::PressureJacobi
        );
    }

    #[test]
    fn projection_plan_pulls_default_iterations() {
        let plan = ProjectionPlan::new(ProjectionMethod::JacobiApprox, 0.01);
        assert_eq!(plan.max_iterations, 40);
        assert_eq!(plan.method, ProjectionMethod::JacobiApprox);
    }

    #[test]
    fn stage_field_io_writes_are_consistent() {
        let advect = stage_field_io(FluidStage::Advect);
        assert_eq!(advect.reads, &[FluidField::Velocity]);
        assert_eq!(advect.writes, &[FluidField::VelocityScratch]);
        let jacobi = stage_field_io(FluidStage::PressureJacobi);
        assert!(jacobi.reads.contains(&FluidField::Divergence));
        assert_eq!(jacobi.writes, &[FluidField::Pressure]);
        let project = stage_field_io(FluidStage::SubtractGradient);
        assert!(project.reads.contains(&FluidField::Pressure));
        assert_eq!(project.writes, &[FluidField::Velocity]);
    }

    #[test]
    fn cfl_and_stable_timestep_round_trip() {
        // A dt at the target CFL reproduces the target CFL number.
        let dt = stable_timestep(2.0, 0.5, 1.0);
        assert!(approx(dt, 0.25));
        assert!(approx(cfl_number(2.0, dt, 0.5), 1.0));
        // Degenerate inputs are safe.
        assert!(approx(stable_timestep(0.0, 0.5, 1.0), 0.0));
        assert!(approx(cfl_number(2.0, dt, 0.0), 0.0));
    }

    #[test]
    fn semi_lagrangian_backtrace_walks_upstream() {
        let pos = Vec3::new(5.0, 5.0, 5.0);
        let vel = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(
            semi_lagrangian_backtrace(pos, vel, 2.0),
            Vec3::new(3.0, 5.0, 5.0)
        );
    }

    #[test]
    fn maccormack_correction_recovers_original_when_symmetric() {
        // When the back-advection recovers the original value (no round-trip
        // error), the `0.5·(original − back)` correction vanishes and the
        // forward-advected value is returned unchanged.
        let forward = Vec3::new(1.0, 2.0, 3.0);
        let original = Vec3::new(4.0, 4.0, 4.0);
        assert_eq!(maccormack_corrected(forward, original, original), forward);
        // A back-advection error is half-corrected.
        let back = Vec3::new(0.0, 0.0, 0.0);
        let corrected = maccormack_corrected(forward, original, back);
        assert_eq!(corrected, Vec3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn trilinear_weights_partition_unity() {
        let w = trilinear_weights(Vec3::new(0.5, 0.5, 0.5));
        for weight in w {
            assert!(approx(weight, 0.125));
        }
        let sum: f32 = trilinear_weights(Vec3::new(0.2, 0.7, 0.9)).iter().sum();
        assert!(approx(sum, 1.0));
    }

    #[test]
    fn trilinear_sample_at_corner_and_center() {
        // Field value == corner x-coordinate: corners at x=0 are 0, x=1 are 1.
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        // At the lower-x corner the sample is 0.
        let at_corner = trilinear_sample(corners, trilinear_weights(Vec3::ZERO));
        assert!(approx(at_corner.x, 0.0));
        // At the x-midpoint the sample is 0.5 (linear along x).
        let at_mid = trilinear_sample(corners, trilinear_weights(Vec3::new(0.5, 0.5, 0.5)));
        assert!(approx(at_mid.x, 0.5));
    }

    #[test]
    fn sample_velocity_field_interpolates_a_ramp() {
        // 2x1x1 grid with velocity.x ramping 0 -> 10 across the cell.
        let res = GridResolution::new(2, 1, 1);
        let field = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)];
        let mid = sample_velocity_field(&field, res, Vec3::new(0.5, 0.0, 0.0));
        assert!(approx(mid.x, 5.0));
        // Clamped past the far edge -> the last voxel value.
        let past = sample_velocity_field(&field, res, Vec3::new(9.0, 0.0, 0.0));
        assert!(approx(past.x, 10.0));
        // Empty grid is safe.
        assert_eq!(
            sample_velocity_field(&[], GridResolution::new(0, 0, 0), Vec3::ZERO),
            Vec3::ZERO
        );
    }

    #[test]
    fn particle_advects_downwind() {
        let res = GridResolution::new(2, 1, 1);
        let field = [Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)];
        let moved = advect_particle_in_field(Vec3::new(0.0, 0.0, 0.0), &field, res, 0.25);
        assert!(approx(moved.x, 1.0));
    }

    #[test]
    fn jacobi_pressure_solve_residual_trends_down() {
        let res = GridResolution::uniform(4);
        let count = res.voxel_count() as usize;
        let mut divergence = vec![0.0f32; count];
        // A point source at the center voxel.
        divergence[res.linear_index(2, 2, 2) as usize] = 1.0;
        // One iteration vs. many: the residual must shrink toward convergence.
        let one = jacobi_pressure_solve(
            &divergence,
            res,
            ProjectionPlan {
                method: ProjectionMethod::JacobiApprox,
                max_iterations: 1,
                residual_tolerance: 0.0,
            },
        );
        let many = jacobi_pressure_solve(
            &divergence,
            res,
            ProjectionPlan {
                method: ProjectionMethod::JacobiApprox,
                max_iterations: 60,
                residual_tolerance: 0.0,
            },
        );
        assert_eq!(one.iterations_run, 1);
        assert_eq!(many.iterations_run, 60);
        assert!(many.residual < one.residual);
        assert!(many.residual >= 0.0);
    }

    #[test]
    fn jacobi_pressure_solve_stops_at_tolerance() {
        let res = GridResolution::uniform(4);
        let count = res.voxel_count() as usize;
        let divergence = vec![0.0f32; count];
        // A zero source is already converged, so no iterations run.
        let result = jacobi_pressure_solve(
            &divergence,
            res,
            ProjectionPlan {
                method: ProjectionMethod::JacobiApprox,
                max_iterations: 10,
                residual_tolerance: 0.001,
            },
        );
        assert_eq!(result.iterations_run, 0);
        assert!(approx(result.residual, 0.0));
    }

    #[test]
    fn subtract_pressure_gradient_removes_divergence() {
        let v = Vec3::new(3.0, 1.0, -2.0);
        let grad = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(
            subtract_pressure_gradient(v, grad),
            Vec3::new(2.0, 0.0, -3.0)
        );
    }

    #[test]
    fn central_gradient_is_finite_difference() {
        let n = NeighborScalars {
            x_plus: 2.0,
            x_minus: 0.0,
            y_plus: 0.0,
            y_minus: 0.0,
            z_plus: 4.0,
            z_minus: 0.0,
        };
        let g = central_gradient(n, 0.5);
        assert_eq!(g, Vec3::new(1.0, 0.0, 2.0));
    }

    fn burn_params() -> CombustionParams {
        CombustionParams {
            ignition_temperature: 1.0,
            burn_rate: 1.0,
            smoke_yield: 2.0,
            heat_yield: 5.0,
            cooling_rate: 0.1,
            ambient_temperature: 0.0,
            buoyancy_alpha: 1.0,
            buoyancy_beta: 0.5,
        }
    }

    #[test]
    fn combustion_couples_fuel_heat_and_smoke() {
        let params = burn_params();
        let before = CombustionState::new(2.0, 1.0, 0.0);
        let after = step_combustion(before, params, 0.5);
        // Fuel is consumed, smoke and (net) temperature rise.
        assert!(after.fuel < before.fuel);
        assert!(after.smoke > before.smoke);
        assert!(after.temperature > before.temperature);
        // Burned 0.5 fuel -> +1.0 smoke, +2.5 heat, then linear cooling.
        assert!(approx(after.fuel, 0.5));
        assert!(approx(after.smoke, 1.0));
    }

    #[test]
    fn combustion_below_ignition_only_cools() {
        let params = burn_params();
        let before = CombustionState::new(0.5, 1.0, 0.0);
        let after = step_combustion(before, params, 1.0);
        // No ignition: fuel and smoke are untouched, temperature relaxes down.
        assert!(approx(after.fuel, 1.0));
        assert!(approx(after.smoke, 0.0));
        assert!(after.temperature < before.temperature);
    }

    #[test]
    fn buoyancy_lifts_hot_and_sinks_smoky() {
        let params = burn_params();
        let hot = buoyancy_force(CombustionState::new(3.0, 0.0, 0.0), params);
        assert!(hot.y > 0.0);
        assert!(approx(hot.x, 0.0));
        assert!(approx(hot.z, 0.0));
        // Heavy cold smoke sinks.
        let smoky = buoyancy_force(CombustionState::new(0.0, 0.0, 4.0), params);
        assert!(smoky.y < 0.0);
    }

    #[test]
    fn blackbody_emission_is_monotonic_in_temperature() {
        let params = EmissionParams {
            low_temperature: 1.0,
            white_temperature: 5.0,
            intensity_scale: 2.0,
        };
        let cold = blackbody_emission(0.5, params);
        let warm = blackbody_emission(2.0, params);
        let hot = blackbody_emission(5.0, params);
        // Below the glow point there is no emission.
        assert!(approx(cold.length(), 0.0));
        // Brightness only increases with temperature.
        assert!(hot.length() > warm.length());
        assert!(warm.length() > 0.0);
    }

    #[test]
    fn heat_haze_scales_with_gradient() {
        let offset = heat_haze_distortion(Vec3::new(0.0, 2.0, 0.0), 0.5);
        assert_eq!(offset, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn curl_of_rigid_rotation_points_along_axis() {
        // Field v = (-y, x, 0) rotates about +Z; its curl is (0, 0, 2).
        let neighbors = NeighborVelocities {
            x_plus: Vec3::new(0.0, 1.0, 0.0),
            x_minus: Vec3::new(0.0, -1.0, 0.0),
            y_plus: Vec3::new(-1.0, 0.0, 0.0),
            y_minus: Vec3::new(1.0, 0.0, 0.0),
            z_plus: Vec3::ZERO,
            z_minus: Vec3::ZERO,
        };
        let curl = neighbors.curl(0.5);
        assert!(approx(curl.x, 0.0));
        assert!(approx(curl.y, 0.0));
        assert!(approx(curl.z, 2.0));
    }

    #[test]
    fn vorticity_confinement_force_is_perpendicular() {
        // Gradient of |curl| along +X, curl along +Z -> force along -Y.
        let curl = Vec3::new(0.0, 0.0, 2.0);
        let grad = Vec3::new(3.0, 0.0, 0.0);
        let force = vorticity_confinement_force(curl, grad, VorticityParams { epsilon: 1.0 }, 1.0);
        assert!(approx(force.x, 0.0));
        assert!(force.y < 0.0);
        assert!(approx(force.z, 0.0));
        // A flat vorticity region (zero gradient) yields no force, not a NaN.
        let none =
            vorticity_confinement_force(curl, Vec3::ZERO, VorticityParams { epsilon: 1.0 }, 1.0);
        assert_eq!(none, Vec3::ZERO);
    }

    #[test]
    fn quality_ladder_degrades_monotonically() {
        assert_eq!(FluidQuality::Ultra.degrade(), Some(FluidQuality::High));
        assert_eq!(FluidQuality::High.degrade(), Some(FluidQuality::Medium));
        assert_eq!(FluidQuality::Medium.degrade(), Some(FluidQuality::Low));
        assert_eq!(FluidQuality::Low.degrade(), None);
        assert!(FluidQuality::Ultra.rank() > FluidQuality::Low.rank());
    }

    #[test]
    fn quality_profile_matrix_steps_down_together() {
        let ultra = fluid_quality_profile(FluidQuality::Ultra);
        let high = fluid_quality_profile(FluidQuality::High);
        let medium = fluid_quality_profile(FluidQuality::Medium);
        let low = fluid_quality_profile(FluidQuality::Low);
        // Resolution shrinks as quality drops.
        assert!(ultra.resolution.voxel_count() > high.resolution.voxel_count());
        assert!(high.resolution.voxel_count() > medium.resolution.voxel_count());
        assert!(medium.resolution.voxel_count() > low.resolution.voxel_count());
        // Exact projection and MacCormack only at the top tiers.
        assert!(ultra.projection.is_exact());
        assert!(high.projection.is_exact());
        assert!(!medium.projection.is_exact());
        assert!(!low.projection.is_exact());
        assert!(high.advection.corrects_numerical_diffusion());
        assert!(!medium.advection.corrects_numerical_diffusion());
        // Vorticity confinement is dropped at the cheapest tier.
        assert!(high.vorticity_enabled);
        assert!(medium.vorticity_enabled);
        assert!(!low.vorticity_enabled);
    }
}
