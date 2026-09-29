//! ABI shared between the water compute passes and the sibling `WESL` shaders
//! `shaders/water_ocean.wesl`, `shaders/water_flip.wesl`,
//! `shaders/water_pbf.wesl`, `shaders/water_surface.wesl` and
//! `shaders/water_render_fx.wesl`.
//!
//! Every record here is a `#[repr(C)]` host mirror of a `WESL` `struct`, laid
//! out byte-for-byte so a plain `bytemuck` cast can upload the `CPU`-golden
//! solver state that the `prism_render_architecture::water::gpu` scheduler
//! sizes. The `size_of` contract tests below pin each record to the resident
//! stride constants exported by
//! [`prism_render_architecture::water::gpu::buffers`], so a drift between the
//! host allocation, the shader `struct` and the golden buffer sizing fails the
//! build rather than corrupting a dispatch at run time. (The `Gerstner`
//! wave record in particular carries such a guard: the golden stride is the
//! full two-`vec4` layout, not a truncated single `vec4`.)
//!
//! `WESL`/`WGSL` layout rules mirrored here:
//!
//! * A `var<storage>` array element uses its `std430` size, which for these
//!   all-scalar / packed-`vec` records equals the packed `#[repr(C)]` size.
//! * A `var<uniform>` block is rounded up to a 16-byte multiple, so the
//!   uniform mirrors carry explicit trailing pad words and the host upload
//!   covers the full binding.
//! * A `vec3<f32>` has 16-byte alignment, so a trailing scalar packs into the
//!   same 16-byte row (the classic `vec3 + f32` slot).

#![allow(
    dead_code,
    reason = "the GPU water ABI records and shader-mirror constants are the verified layout foundation of this subsystem; the pipeline / bind-group / dispatch slices that consume them land in the following slices, and the `size_of` contract tests exercise every record now"
)]

use bytemuck::{Pod, Zeroable};

/// Planar tile edge of every per-texel / per-cell water compute entry point.
/// Must match every `@workgroup_size(8, 8, 1)` in the water shaders (spectrum
/// `IFFT`, `Gerstner`, `SWE`, foam, waterline, caustics, dispersion, wetness).
pub(crate) const WATER_TEXEL_TILE: u32 = 8;

/// Volumetric tile edge of the froxel / `MAC`-grid entry points. Must match
/// every `@workgroup_size(4, 4, 4)` in the water shaders (`FLIP` pressure and
/// the underwater froxel volume).
pub(crate) const WATER_VOXEL_TILE: u32 = 4;

/// Linear workgroup size of the per-particle / per-query water entry points.
/// Must match every `@workgroup_size(64)` in the water shaders (`FLIP`
/// `P2G`/`G2P`, `PBF` density solve, spray, coupling read-back).
pub(crate) const WATER_LINEAR_WORKGROUP: u32 = 64;

// ===========================================================================
// Ocean: spectrum `IFFT` + analytic `Gerstner` (water_ocean.wesl)
// ===========================================================================

/// Per-frame `Tessendorf` spectrum scalars. Byte-compatible with
/// `WaterSpectrumParams` in `water_ocean.wesl`, padded to the 32-byte
/// 16-byte-rounded uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterSpectrumParams {
    /// Grid resolution `N`; the spectrum buffers are `N*N` complex amplitudes.
    pub grid_size: u32,
    /// Physical patch size `L` (m).
    pub patch_size: f32,
    /// Simulation time `t` (s) fed to the Hermitian phase advance.
    pub time: f32,
    /// Horizontal choppiness scale `lambda` (the `Tessendorf` `-i*k_hat` term).
    pub choppiness: f32,
    /// Fold threshold: `J <= foam_threshold` flags a breaking crest.
    pub foam_threshold: f32,
    /// Trailing pad to the 32-byte uniform stride; never read.
    pub _pad: [u32; 3],
}

/// One analytic `Gerstner` wave train. Byte-compatible with `GerstnerWave` in
/// `water_ocean.wesl` and the golden `GERSTNER_WAVE_STRIDE` (32 bytes: eight
/// `f32` lanes = two `vec4<f32>` rows).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuGerstnerWave {
    /// Unit propagation direction, x component (xz plane).
    pub dir_x: f32,
    /// Unit propagation direction, z component (xz plane).
    pub dir_z: f32,
    /// Crest amplitude `A` (m).
    pub amplitude: f32,
    /// Wavelength `L` (m); wave number `k = 2*PI / L`.
    pub wavelength: f32,
    /// Steepness `Q` in `[0, 1]` (`0` = sine, `1` = sharp cusp).
    pub steepness: f32,
    /// Phase-speed multiplier on deep-water `omega = sqrt(g*k)`.
    pub speed: f32,
    /// Constant phase offset (rad) so trains de-correlate.
    pub phase: f32,
    /// Padding lane for the 16-byte row alignment; ignored.
    pub _pad: f32,
}

/// Per-frame `Gerstner` scalars. Byte-compatible with `WaterGerstnerParams` in
/// `water_ocean.wesl`, padded to the 32-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterGerstnerParams {
    /// Grid resolution `N`; output textures are `N*N` texels.
    pub grid_size: u32,
    /// Physical patch size `L` (m).
    pub patch_size: f32,
    /// Simulation time `t` (s) advancing every wave's phase.
    pub time: f32,
    /// Number of active `GpuGerstnerWave` entries to sum.
    pub wave_count: u32,
    /// Rest water level (world y, m) added to the summed vertical displacement.
    pub base_level: f32,
    /// Trailing pad to the 32-byte uniform stride; never read.
    pub _pad: [u32; 3],
}

// ===========================================================================
// Volume `FLIP`/`APIC` (water_flip.wesl)
// ===========================================================================

/// One `FLIP`/`APIC` particle. Byte-compatible with `FlipParticle` in
/// `water_flip.wesl` and the golden `FLIP_PARTICLE_STRIDE` (80 bytes: five
/// `vec4<f32>` lanes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuFlipParticle {
    /// World position (`xyz`) and active flag (`w`).
    pub pos: [f32; 4],
    /// Particle velocity (`xyz`).
    pub vel: [f32; 4],
    /// `APIC` affine matrix row 0 (`xyz`).
    pub c0: [f32; 4],
    /// `APIC` affine matrix row 1 (`xyz`).
    pub c1: [f32; 4],
    /// `APIC` affine matrix row 2 (`xyz`).
    pub c2: [f32; 4],
}

/// Global `FLIP`/`APIC` grid scalars. Byte-compatible with `FlipSimParams` in
/// `water_flip.wesl` (two `vec4` rows + eight scalars = 64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuFlipSimParams {
    /// World-space origin of cell `(0, 0, 0)`'s corner (`xyz`).
    pub origin: [f32; 4],
    /// Grid resolution in cells along x, y, z (`w` unused).
    pub dim: [u32; 4],
    /// `MAC` cell edge length (`> 0`).
    pub dx: f32,
    /// Reciprocal cell edge length `1 / dx`.
    pub inv_dx: f32,
    /// `FLIP`/`PIC` blend in `0..=1` (`0` = `PIC`, `1` = `FLIP`).
    pub flip_blend: f32,
    /// Per-particle mass weighting the `P2G` scatter (`> 0`).
    pub particle_mass: f32,
    /// Damped-`Jacobi` relaxation factor in `(0, 1]`.
    pub jacobi_omega: f32,
    /// `1` if the `APIC` affine field is carried, `0` for plain `PIC`/`FLIP`.
    pub use_affine: u32,
    /// Live particle count bounding the particle dispatches.
    pub particle_count: u32,
    /// Total grid cell count bounding the pressure dispatch.
    pub cell_count: u32,
}

/// Screen-space reconstruction scalars. Byte-compatible with
/// `FlipSurfaceParams` in `water_flip.wesl` (`van der Laan` smooth + normal),
/// padded to the 32-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuFlipSurfaceParams {
    /// Viewport resolution in pixels (`xy`).
    pub resolution: [u32; 2],
    /// Bilateral filter half-width in pixels.
    pub filter_radius: i32,
    /// Spatial Gaussian falloff denominator `2*sigma_spatial^2` (`> 0`).
    pub spatial_sigma2: f32,
    /// Range (depth) Gaussian falloff denominator `2*sigma_range^2` (`> 0`).
    pub range_sigma2: f32,
    /// View-space scale of a one-pixel step at unit depth (normal differences).
    pub pixel_world_scale: f32,
    /// Trailing pad to the 32-byte uniform stride; never read.
    pub _pad: [u32; 2],
}

// ===========================================================================
// Volume `PBF` + spray (water_pbf.wesl)
// ===========================================================================

/// `PBF` density-solve tuning + grid description. Byte-compatible with
/// `PbfParams` in `water_pbf.wesl` (`vec3 + f32` row + twelve scalars = 64).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuPbfParams {
    /// Minimum corner of the spatial-hash grid, world space.
    pub grid_origin: [f32; 3],
    /// Grid cell edge length (`> 0`), equal to the smoothing radius.
    pub cell_size: f32,
    /// Target rest density `rho_0` (`> 0`).
    pub rest_density: f32,
    /// Per-particle mass (`> 0`) for the `SPH` density sum.
    pub particle_mass: f32,
    /// Smoothing radius `h` (`> 0`); support radius of both kernels.
    pub smoothing_radius: f32,
    /// `XPBD` relaxation term added to the `lambda` denominator.
    pub relaxation_epsilon: f32,
    /// Artificial-pressure strength `k` (`>= 0`) fighting clustering.
    pub artificial_pressure_k: f32,
    /// Fraction of `h` at which the artificial-pressure reference is sampled.
    pub artificial_pressure_delta_q: f32,
    /// Artificial-pressure exponent `n` (`>= 1`).
    pub artificial_pressure_n: u32,
    /// Particle count bounding the per-particle dispatch.
    pub particle_count: u32,
    /// Grid cell count along x (`>= 1`).
    pub grid_nx: u32,
    /// Grid cell count along y (`>= 1`).
    pub grid_ny: u32,
    /// Grid cell count along z (`>= 1`).
    pub grid_nz: u32,
    /// Padding to the 16-byte multiple for `uniform` layout.
    pub _pad: u32,
}

/// One candidate crest sample for the spray classifier. Byte-compatible with
/// `SpraySource` in `water_pbf.wesl` (three `vec3 + f32` rows = 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSpraySource {
    /// World origin of the burst.
    pub position: [f32; 3],
    /// Displacement-gradient steepness (slope magnitude).
    pub steepness: f32,
    /// Surface tangent (crest flow direction) for the launch.
    pub tangent: [f32; 3],
    /// Choppy-displacement Jacobian (a fold at or below zero).
    pub jacobian: f32,
    /// Surface normal (upward jet direction).
    pub normal: [f32; 3],
    /// Local crest curvature magnitude.
    pub curvature: f32,
}

/// One planned spray burst. Byte-compatible with `SprayParticle` in
/// `water_pbf.wesl` (`vec3 + u32` twice = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSprayParticle {
    /// World origin of the burst.
    pub position: [f32; 3],
    /// Number of spray particles this source emits (`0` when not breaking).
    pub count: u32,
    /// Launch velocity (tangent + upward jet, scaled by intensity).
    pub velocity: [f32; 3],
    /// Padding to the 16-byte multiple.
    pub _pad: u32,
}

/// Header of the spray spawn buffer: the `atomic<u32>` total counter followed
/// (in the device buffer) by one `GpuSprayParticle` per source slot. The
/// counter sits in the first 16-byte row because `GpuSprayParticle`'s `vec3`
/// forces the array to start at offset 16. The host uploads this header zeroed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSpraySpawnHeader {
    /// Accumulated total spawned particle count (order-independent sum).
    pub counter: u32,
    /// Padding to the 16-byte start of the burst array; never read.
    pub _pad: [u32; 3],
}

/// Breaking classifier thresholds + spray tuning. Byte-compatible with
/// `SprayParams` in `water_pbf.wesl` (eight scalars = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSprayParams {
    /// Steepness at which cresting begins (saturates at twice it).
    pub steepness_threshold: f32,
    /// Jacobian at or below which a fold is counted.
    pub jacobian_fold_threshold: f32,
    /// Curvature at which the crest is sharp enough to spume.
    pub curvature_threshold: f32,
    /// Intensity at or above which a sample is fully breaking.
    pub breaking_intensity: f32,
    /// Launch-speed scale for the crest-spray jet.
    pub jet_speed: f32,
    /// Number of source samples bounding the dispatch.
    pub source_count: u32,
    /// Capacity of the per-source spawn record array.
    pub spawn_capacity: u32,
    /// Upper bound on spray particle count for one source.
    pub max_spray_count: u32,
}

// ===========================================================================
// Surface `SWE` / foam / waterline (water_surface.wesl)
// ===========================================================================

/// Shallow-water grid + `CFL`/timestep scalars. Byte-compatible with
/// `WaterSweParams` in `water_surface.wesl` (eight scalars = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterSweParams {
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
    /// Square cell size in meters (`> 0`).
    pub dx: f32,
    /// Gravitational acceleration (wave celerity + pressure term).
    pub gravity: f32,
    /// Linear velocity damping per second in `0..=1`.
    pub damping: f32,
    /// Requested frame timestep, seconds (clamped to the `CFL` bound).
    pub dt: f32,
    /// Courant number the explicit step must satisfy.
    pub cfl_number: f32,
    /// Largest grid signal speed this step (host `swe::max_wave_speed`).
    pub max_wave_speed: f32,
}

/// Foam advection / decay scalars. Byte-compatible with `WaterFoamParams` in
/// `water_surface.wesl`, padded to the 32-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterFoamParams {
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
    /// Square cell size in meters (`> 0`).
    pub dx: f32,
    /// Advection / decay timestep, seconds.
    pub dt: f32,
    /// Baseline decay rate per second at or above `reference_speed`.
    pub base_decay: f32,
    /// Fraction of `base_decay` still applied in still water, `0..=1`.
    pub persistence_floor: f32,
    /// Flow speed at which decay reaches the full `base_decay` (`> 0`).
    pub reference_speed: f32,
    /// Trailing pad to the 32-byte uniform stride; never read.
    pub _pad: u32,
}

/// Waterline mask scalars. Byte-compatible with `WaterWaterlineParams` in
/// `water_surface.wesl` (four scalars = one 16-byte uniform row).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterWaterlineParams {
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
    /// Half-width of the soft transition straddling the surface, meters.
    pub transition_half_width: f32,
    /// Total water depth below which a submerged sample joins the shoreline.
    pub shoreline_depth: f32,
}

// ===========================================================================
// Render `FX`: caustics / dispersion / underwater / wetness / coupling
// (water_render_fx.wesl)
// ===========================================================================

/// `RT`/Jacobian caustics scalars. Byte-compatible with `WaterCausticsParams`
/// in `water_render_fx.wesl` (eight scalars = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterCausticsParams {
    /// Incident irradiance reaching the receiver before focusing.
    pub incident: f32,
    /// Upper bound on the Jacobian gain (cap a near-singular focus).
    pub max_gain: f32,
    /// Per-photon power for the `RT`/photon splat-density term.
    pub photon_power: f32,
    /// Photon gather radius for the splat-density term.
    pub splat_radius: f32,
    /// Receiver-plane texel size (finite-difference step for the Jacobian).
    pub texel_size: f32,
    /// Target width in texels.
    pub width: u32,
    /// Target height in texels.
    pub height: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad: u32,
}

/// Chromatic dispersion scalars. Byte-compatible with `WaterDispersionParams`
/// in `water_render_fx.wesl` (`Cauchy` `IOR` + eight-scalar 32-byte block).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterDispersionParams {
    /// `Cauchy` baseline index `a`.
    pub cauchy_a: f32,
    /// `Cauchy` dispersion coefficient `b`, in micrometre-squared.
    pub cauchy_b: f32,
    /// Screen-space refraction offset gain folded with water thickness.
    pub strength: f32,
    /// Target width in texels.
    pub width: u32,
    /// Target height in texels.
    pub height: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad0: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad1: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad2: u32,
}

/// Underwater volumetric scalars. Byte-compatible with `WaterUnderwaterParams`
/// in `water_render_fx.wesl` (two `vec3 + f32` rows + scalar tail = 64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterUnderwaterParams {
    /// `RGB` extinction coefficients (`r > g > b` for the blue-green shift).
    pub extinction: [f32; 3],
    /// Thickness of one `froxel` depth slice, in metres.
    pub slice_thickness: f32,
    /// Base surface radiance colour entering the medium.
    pub base_color: [f32; 3],
    /// `Henyey-Greenstein` asymmetry `g` in `(-1, 1)`.
    pub phase_g: f32,
    /// Cosine of the angle between the view and sun directions.
    pub sun_cos: f32,
    /// Single-scattering albedo for the multiple-scatter boost.
    pub scatter_albedo: f32,
    /// `froxel` grid width.
    pub width: u32,
    /// `froxel` grid height.
    pub height: u32,
    /// `froxel` grid depth (slice count).
    pub depth: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad0: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad1: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad2: u32,
}

/// Surface wetness scalars. Byte-compatible with `WaterWetnessParams` in
/// `water_render_fx.wesl` (twelve scalars = 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterWetnessParams {
    /// Maximum capillary rise / wet-band reach, in metres.
    pub max_capillary_height: f32,
    /// Absorption rate per second on contact or full rain.
    pub absorb_rate: f32,
    /// Drying rate per second when exposed and rain-free.
    pub dry_rate: f32,
    /// Peak fractional albedo darkening at full saturation, `0..=1`.
    pub darkening_strength: f32,
    /// Puddle depth above which a cell seeds `SWE` ripples.
    pub puddle_threshold: f32,
    /// Normalised rain wetting drive for this step.
    pub rain_rate: f32,
    /// Puddle drainage rate for this step.
    pub drain_rate: f32,
    /// Step duration, seconds.
    pub dt: f32,
    /// Height above the waterline sampled for the capillary band, in metres.
    pub dist_above_water: f32,
    /// Non-zero when the surface is in direct water contact this step.
    pub water_contact: u32,
    /// Field width in cells.
    pub width: u32,
    /// Field height in cells.
    pub height: u32,
}

/// One two-way coupling query. Byte-compatible with `WaterCouplingQuery` in
/// `water_render_fx.wesl` (four scalars = 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterCouplingQuery {
    /// Volume of the body below the surface.
    pub submerged_volume: f32,
    /// Total volume of the body.
    pub total_volume: f32,
    /// Drag cross-section presented to the flow.
    pub cross_section: f32,
    /// Relative body/fluid speed.
    pub rel_speed: f32,
}

/// Coupling read-back pack scalars. Byte-compatible with `WaterCouplingParams`
/// in `water_render_fx.wesl` (eight scalars = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterCouplingParams {
    /// Density of the surrounding fluid.
    pub fluid_density: f32,
    /// Quadratic form-drag coefficient.
    pub drag_coeff: f32,
    /// Added-mass coefficient for the body shape.
    pub added_mass_coeff: f32,
    /// Gravitational acceleration magnitude.
    pub gravity: f32,
    /// Number of queries requested this frame.
    pub query_count: u32,
    /// Upper bound on the read-back batch (never a full-field read-back).
    pub max_readback: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad0: u32,
    /// Padding to a 16-byte boundary; never read.
    pub _pad1: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::water::gpu::buffers::{
        FLIP_PARTICLE_STRIDE, GERSTNER_WAVE_STRIDE, GRID_SCALAR_STRIDE, PBF_PARTICLE_STRIDE,
        SPECTRUM_AMPLITUDE_STRIDE, SWE_CELL_STRIDE,
    };

    /// The ocean `h0(+/-k)` amplitude buffers are `array<vec2<f32>>`, so their
    /// element matches the golden complex-amplitude stride.
    #[test]
    fn spectrum_amplitude_matches_golden_stride() {
        assert_eq!(size_of::<[f32; 2]>() as u32, SPECTRUM_AMPLITUDE_STRIDE);
    }

    /// The wave-train buffer element equals the golden `Gerstner` stride. This
    /// is the guard that pins the full two-`vec4` (32-byte) layout, so a mirror
    /// truncated to a single `vec4` fails the build.
    #[test]
    fn gerstner_wave_matches_golden_stride() {
        assert_eq!(size_of::<GpuGerstnerWave>() as u32, GERSTNER_WAVE_STRIDE);
        assert_eq!(size_of::<GpuGerstnerWave>(), 32);
    }

    /// The spectrum uniform rounds up to a 32-byte uniform stride.
    #[test]
    fn spectrum_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuWaterSpectrumParams>(), 32);
        assert_eq!(size_of::<GpuWaterSpectrumParams>() % 16, 0);
    }

    /// The `Gerstner` uniform rounds up to a 32-byte uniform stride.
    #[test]
    fn gerstner_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuWaterGerstnerParams>(), 32);
        assert_eq!(size_of::<GpuWaterGerstnerParams>() % 16, 0);
    }

    /// The `FLIP` particle equals the golden five-`vec4` stride.
    #[test]
    fn flip_particle_matches_golden_stride() {
        assert_eq!(size_of::<GpuFlipParticle>() as u32, FLIP_PARTICLE_STRIDE);
        assert_eq!(size_of::<GpuFlipParticle>(), 80);
    }

    /// The `FLIP` grid scalars (pressure ping-pong, scatter lanes) match the
    /// golden per-scalar stride.
    #[test]
    fn flip_grid_scalar_matches_golden_stride() {
        assert_eq!(size_of::<f32>() as u32, GRID_SCALAR_STRIDE);
        assert_eq!(size_of::<u32>() as u32, GRID_SCALAR_STRIDE);
    }

    /// The `FLIP` sim uniform is two `vec4` rows plus eight scalars (64 bytes).
    #[test]
    fn flip_sim_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuFlipSimParams>(), 64);
        assert_eq!(size_of::<GpuFlipSimParams>() % 16, 0);
    }

    /// The `FLIP` surface uniform rounds up to a 32-byte uniform stride.
    #[test]
    fn flip_surface_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuFlipSurfaceParams>(), 32);
        assert_eq!(size_of::<GpuFlipSurfaceParams>() % 16, 0);
    }

    /// The `PBF` position ping-pong buffers are `array<vec4<f32>>`, matching
    /// the golden particle stride.
    #[test]
    fn pbf_particle_matches_golden_stride() {
        assert_eq!(size_of::<[f32; 4]>() as u32, PBF_PARTICLE_STRIDE);
    }

    /// The `PBF` uniform is a `vec3 + f32` row plus twelve scalars (64 bytes).
    #[test]
    fn pbf_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuPbfParams>(), 64);
        assert_eq!(size_of::<GpuPbfParams>() % 16, 0);
    }

    /// A spray source packs three `vec3 + f32` rows (48 bytes).
    #[test]
    fn spray_source_is_forty_eight_bytes() {
        assert_eq!(size_of::<GpuSpraySource>(), 48);
        assert_eq!(size_of::<GpuSpraySource>() % 16, 0);
    }

    /// A spray burst record is two `vec3 + u32` rows (32 bytes), and the spawn
    /// header reserves the first 16-byte row before the burst array.
    #[test]
    fn spray_records_are_sixteen_aligned() {
        assert_eq!(size_of::<GpuSprayParticle>(), 32);
        assert_eq!(size_of::<GpuSprayParticle>() % 16, 0);
        assert_eq!(size_of::<GpuSpraySpawnHeader>(), 16);
    }

    /// The spray classifier uniform is eight scalars (32 bytes).
    #[test]
    fn spray_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuSprayParams>(), 32);
        assert_eq!(size_of::<GpuSprayParams>() % 16, 0);
    }

    /// The `SWE` per-cell interaction source is an `array<vec4<f32>>`, matching
    /// the golden cell stride; the state fields are per-scalar arrays.
    #[test]
    fn swe_cell_matches_golden_stride() {
        assert_eq!(size_of::<[f32; 4]>() as u32, SWE_CELL_STRIDE);
        assert_eq!(size_of::<f32>() as u32, GRID_SCALAR_STRIDE);
    }

    /// The `SWE` uniform is eight scalars (32 bytes).
    #[test]
    fn swe_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuWaterSweParams>(), 32);
        assert_eq!(size_of::<GpuWaterSweParams>() % 16, 0);
    }

    /// The foam and waterline uniforms round up to their 16-byte multiples.
    #[test]
    fn surface_fx_params_are_uniform_stride() {
        assert_eq!(size_of::<GpuWaterFoamParams>(), 32);
        assert_eq!(size_of::<GpuWaterFoamParams>() % 16, 0);
        assert_eq!(size_of::<GpuWaterWaterlineParams>(), 16);
    }

    /// The caustics and dispersion uniforms are eight-scalar 32-byte blocks.
    #[test]
    fn render_fx_2d_params_are_uniform_stride() {
        assert_eq!(size_of::<GpuWaterCausticsParams>(), 32);
        assert_eq!(size_of::<GpuWaterCausticsParams>() % 16, 0);
        assert_eq!(size_of::<GpuWaterDispersionParams>(), 32);
        assert_eq!(size_of::<GpuWaterDispersionParams>() % 16, 0);
    }

    /// The underwater and wetness uniforms round up to their 16-byte multiples.
    #[test]
    fn volumetric_fx_params_are_uniform_stride() {
        assert_eq!(size_of::<GpuWaterUnderwaterParams>(), 64);
        assert_eq!(size_of::<GpuWaterUnderwaterParams>() % 16, 0);
        assert_eq!(size_of::<GpuWaterWetnessParams>(), 48);
        assert_eq!(size_of::<GpuWaterWetnessParams>() % 16, 0);
    }

    /// A coupling query is four scalars (16 bytes); its params uniform is
    /// eight scalars (32 bytes).
    #[test]
    fn coupling_records_are_uniform_stride() {
        assert_eq!(size_of::<GpuWaterCouplingQuery>(), 16);
        assert_eq!(size_of::<GpuWaterCouplingParams>(), 32);
        assert_eq!(size_of::<GpuWaterCouplingParams>() % 16, 0);
    }

    /// The shared workgroup tiles match the `@workgroup_size(...)` literals in
    /// the water shaders.
    #[test]
    fn workgroup_tiles_match_shaders() {
        assert_eq!(WATER_TEXEL_TILE, 8);
        assert_eq!(WATER_VOXEL_TILE, 4);
        assert_eq!(WATER_LINEAR_WORKGROUP, 64);
    }
}
