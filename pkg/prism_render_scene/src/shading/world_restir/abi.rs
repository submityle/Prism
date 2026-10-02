//! ABI shared between the world-space `ReSTIR` direct-illumination fill pass and
//! its WESL twin (`shaders/world_restir_fill.wesl`), plus the resident
//! reservoir-cell storage layout the water-surface `@group(9)` consumer reads
//! back.
//!
//! World-space `ReSTIR` keeps a `SHARC`-style spatial-hash cache of streaming
//! `RIS` reservoirs: one open-addressed slot per hashed world cell, each
//! holding the surviving light sample (a [`prism_render_shading`] `GiSample`
//! twin) and its finalised unbiased contribution weight `W`. The CPU golden
//! lives in [`prism_render_shading::gi::world_restir`] (hash keys +
//! cross-cell `GRIS` merge) and [`prism_render_shading::gi::screen_probe`]'s
//! `restir` (the streaming reservoir + `target_function`); this module freezes
//! the byte layouts both the producer compute pass and the consumer sample.
//!
//! Both blocks are laid out so a machine with and without a GPU agree with the
//! CPU golden: the reservoir cell packs five `vec4` lanes (`vec3` + trailing
//! scalar each) for 80 bytes, and the fill immediate block packs four `vec4`
//! lanes for 64 bytes, each a multiple of the 16-byte `WGSL` alignment with no
//! implicit padding.

use bevy_math::Vec3;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismWorldRestirSettings;

/// Workgroup size (1-D) of the world-space `ReSTIR` fill entry point.
///
/// Must match `@workgroup_size(N, 1, 1)` in `world_restir_fill.wesl`; the
/// dispatch rounds the reservoir-table capacity up to a multiple of this and
/// the shader bounds-checks every invocation against the live capacity.
pub(crate) const WORLD_RESTIR_WORKGROUP_SIZE: u32 = 64;

/// Per-cell reservoir storage stride in bytes: five `vec4<f32>` lanes (a
/// `vec3` payload plus one trailing scalar each) = 80 bytes, matching the WGSL
/// `WorldRestirReservoir` struct's std430 layout and its 16-byte array stride.
pub(crate) const WORLD_RESTIR_RESERVOIR_STRIDE: u64 = 80;

/// `GPU` twin of one hashed world-cell `ReSTIR` reservoir (the resident
/// reservoir-table slot).
///
/// Mirrors a [`prism_render_shading`] `GiSample` (five `Vec3`s: the visible /
/// shading point, the secondary light-sample point, their normals, and the RGB
/// radiance leaving the sample towards the visible point) plus the reservoir
/// metadata the consumer needs to reconstruct the estimator: the finalised
/// unbiased contribution weight `w` (`W`), the confidence `m`, the slot
/// collision `checksum` (`SHARC` open-addressing guard), and a `valid` flag.
///
/// Layout (five `vec4` lanes, std430, no implicit padding):
/// 0. `visible_point.xyz` + `w`
/// 1. `visible_normal.xyz` + `m`
/// 2. `sample_point.xyz` + `checksum`
/// 3. `sample_normal.xyz` + `valid`
/// 4. `radiance.xyz` + one pad word
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirReservoir {
    /// Visible-point (shading-point) world position `x_v`: the cell centroid
    /// the golden `target_function` re-evaluates a shift-mapped neighbour
    /// against.
    pub visible_point: [f32; 3],
    /// Finalised unbiased contribution weight `W = (1 / p_hat) * (w_sum / m)`
    /// (golden `Reservoir::contribution_weight`); `0` on an empty slot.
    pub w: f32,
    /// Unit surface normal at the visible point.
    pub visible_normal: [f32; 3],
    /// Reservoir confidence `m` (effective candidate count after the `GRIS`
    /// merge + `M`-cap).
    pub m: f32,
    /// Secondary light-sample world position `x_s` (the point on the chosen
    /// emitter the surviving candidate sampled).
    pub sample_point: [f32; 3],
    /// `SHARC` collision checksum of the slot's hash key (golden
    /// `spatial_hash::checksum`); the consumer rejects a slot whose checksum
    /// does not match its own recomputed key.
    pub checksum: u32,
    /// Unit surface normal at the light-sample point (the emitter orientation).
    pub sample_normal: [f32; 3],
    /// `1` when the slot holds a surviving candidate this frame, `0` otherwise.
    pub valid: u32,
    /// Linear RGB radiance leaving `x_s` towards `x_v`.
    pub radiance: [f32; 3],
    /// Padding word rounding the final lane to the 16-byte std430 stride.
    pub _pad0: u32,
}

impl GpuWorldRestirReservoir {
    /// An empty, zeroed reservoir slot carrying no energy and flagged invalid.
    ///
    /// `wgpu` zero-initialises a freshly created storage buffer, so the resident
    /// tables already start as all-`EMPTY` slots without the host ever writing
    /// one; this named constant documents that frozen zero-slot ABI and anchors
    /// the layout tests. The seed pass (a follow-up slice) is the first host
    /// path to construct it explicitly, so off-test it is intentionally unused.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "frozen GPU zero-slot ABI; wgpu zero-init yields it on the device and the                       seed pass constructs it in a follow-up slice, so no host path reads it yet"
        )
    )]
    pub(crate) const EMPTY: Self = Self {
        visible_point: [0.0; 3],
        w: 0.0,
        visible_normal: [0.0; 3],
        m: 0.0,
        sample_point: [0.0; 3],
        checksum: 0,
        sample_normal: [0.0; 3],
        valid: 0,
        radiance: [0.0; 3],
        _pad0: 0,
    };
}

/// Immediate (push-constant) block consumed by the `world_restir_fill` entry
/// point.
///
/// One invocation per reservoir slot: it streams the frame's light candidates
/// through the slot's `RIS` reservoir (golden `Reservoir::update` /
/// `ris_weight`), optionally merges a jittered ring of spatial neighbours
/// (golden `merge_spatial` / `pairwise_mis_weight`), caps the confidence and
/// finalises `W` (golden `finalize`). The hash-grid tunables mirror the golden
/// [`prism_render_shading::gi::world_restir::spatial_hash`] `HashGridParams`
/// so the device cache keys agree with the CPU reference.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirFillParams {
    /// Camera world position, forwarded to the golden `grid_level` so the
    /// cell size grows with viewing distance (constant screen footprint).
    pub camera_position: [f32; 3],
    /// Reservoir-table capacity in slots (`>= 1`), the dispatch extent and the
    /// open-addressing modulus (golden `bucket_index`).
    pub capacity: u32,
    /// Grid-phase jitter in `[0, 1)^3` cell units (golden `HashGridParams`).
    pub jitter: [f32; 3],
    /// Number of active lights the `RIS` candidate stream draws from.
    pub light_count: u32,
    /// Edge length of a level-0 cell in world units (golden
    /// `HashGridParams::base_cell_size`, clamped positive by the shader).
    pub base_cell_size: f32,
    /// Distance-to-cell-size scale (golden `HashGridParams::level_scale`);
    /// `<= 0` disables level scaling (uniform fine grid).
    pub level_scale: f32,
    /// Side count of the `resolution x resolution` normal-bin grid (golden
    /// `HashGridParams::normal_resolution`, clamped to at least `1`).
    pub normal_resolution: u32,
    /// `ReSTIR` `M`-cap bounding how much temporal history a slot accumulates
    /// (golden `Reservoir::cap_confidence`).
    pub m_cap: f32,
    /// Artistic gain baked into the reservoir radiance before `W` is applied.
    pub intensity: f32,
    /// Monotonic frame index seeding the per-slot streaming `RNG`.
    pub frame: u32,
    /// Number of spatial neighbours each slot merges per frame (golden
    /// `GRIS` reuse); `0` disables spatial reuse.
    pub spatial_samples: u32,
    /// Spatial-reuse search radius in cells around each slot.
    pub spatial_radius: u32,
}

impl GpuWorldRestirFillParams {
    /// Builds the fill immediate block from the camera position, the per-frame
    /// jitter + light count + frame index, and the live settings.
    pub(crate) fn from_settings(
        camera_position: Vec3,
        jitter: Vec3,
        light_count: u32,
        frame: u32,
        settings: &PrismWorldRestirSettings,
    ) -> Self {
        Self {
            camera_position: camera_position.to_array(),
            capacity: settings.capacity.max(1),
            jitter: jitter.to_array(),
            light_count,
            base_cell_size: settings.base_cell_size,
            level_scale: settings.level_scale,
            normal_resolution: settings.normal_resolution,
            m_cap: settings.m_cap,
            intensity: settings.intensity,
            frame,
            spatial_samples: settings.spatial_samples,
            spatial_radius: settings.spatial_radius,
        }
    }
}

/// Workgroup size (1-D) of the world-space `ReSTIR` seed entry point.
///
/// Must match `@workgroup_size(N, 1, 1)` in `world_restir_seed.wesl`; the seed
/// dispatch rounds the reservoir-table capacity up to a multiple of this and
/// the shader bounds-checks every invocation against the live capacity, exactly
/// as the fill pass does.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "seed dispatch extent constant; consumed once the seed pipeline is wired in a                   follow-up slice, so no host path reads it yet"
    )
)]
pub(crate) const WORLD_RESTIR_SEED_WORKGROUP_SIZE: u32 = 64;

/// Per-light candidate storage stride in bytes: two `vec4<f32>` lanes (a `vec3`
/// payload plus one trailing scalar each) = 32 bytes, matching the WGSL
/// `WorldRestirLight` struct's std430 layout and its 16-byte array stride.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "seed light-buffer stride; consumed once the seed bind group binds the light                   list in a follow-up slice, so no host path reads it yet"
    )
)]
pub(crate) const WORLD_RESTIR_LIGHT_STRIDE: u64 = 32;

/// `GPU` twin of one candidate light the seed pass's `RIS` stream draws from
/// (the per-frame light-list record the seed bind group binds at `@binding(2)`).
///
/// The seed kernel reads the emitter world position and its scalar intensity
/// from the first lane and the linear RGB colour from the second, builds a
/// `GiSample` towards the cell's visible point, and resamples it under
/// `ris_weight` (see `world_restir_seed.wesl`). The layout matches the WESL
/// `WorldRestirLight` struct: two `vec4` lanes (a `vec3` payload plus one
/// trailing scalar each) for 32 bytes, a multiple of the 16-byte std430 array
/// stride with no implicit padding.
///
/// Layout (two `vec4` lanes, std430, no implicit padding):
/// 0. `position.xyz` + `intensity`
/// 1. `color.xyz` + one pad word
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirLight {
    /// Emitter world position; the secondary sample point `x_s` of the built
    /// candidate.
    pub position: [f32; 3],
    /// Scalar radiant intensity scaling the emitter colour before the artistic
    /// gain.
    pub intensity: f32,
    /// Linear RGB emitter colour.
    pub color: [f32; 3],
    /// Padding word rounding the record to the 16-byte std430 stride.
    pub _pad0: f32,
}

/// Immediate (push-constant) block consumed by the `world_restir_seed` entry
/// point.
///
/// One invocation per reservoir slot: an occupied slot streams
/// `candidate_count` light candidates through its `RIS` reservoir (golden
/// `stream_candidate`), caps the confidence at `m_cap` (golden
/// `cap_confidence`) and finalises `W` (golden `finalize`); an empty slot is
/// copied through so the ping-pong preserves vacancy. The `intensity` artistic
/// gain and the `m_cap` mirror the fill block's so the two passes agree, and
/// `frame` seeds the per-slot streaming `RNG`.
///
/// Layout (two `vec4` lanes, std430, no implicit padding):
/// 0. (`capacity`, `light_count`, `candidate_count`, `frame`)
/// 1. (`intensity`, `m_cap`, pad, pad)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirSeedParams {
    /// Reservoir-table capacity in slots (`>= 1`), the dispatch extent and the
    /// per-invocation bounds check.
    pub capacity: u32,
    /// Number of active lights the `RIS` candidate stream draws from; `0`
    /// leaves every slot's reservoir empty.
    pub light_count: u32,
    /// Candidates streamed per occupied slot per frame (the `RIS` budget).
    pub candidate_count: u32,
    /// Monotonic frame index seeding the per-slot streaming `RNG`.
    pub frame: u32,
    /// Artistic gain baked into the candidate radiance before `W` is applied;
    /// `1` reproduces the golden magnitude exactly.
    pub intensity: f32,
    /// `ReSTIR` `M`-cap bounding how much confidence a slot accumulates (golden
    /// `Reservoir::cap_confidence`); negative disables the cap.
    pub m_cap: f32,
    /// Padding word rounding the second lane to the 16-byte std430 stride.
    pub _pad0: f32,
    /// Padding word rounding the second lane to the 16-byte std430 stride.
    pub _pad1: f32,
}

impl GpuWorldRestirSeedParams {
    /// Builds the seed immediate block from the per-frame light count + frame
    /// index and the live settings (capacity floored at `1`, the candidate
    /// budget / artistic gain / `M`-cap forwarded verbatim).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "seed immediate builder; called once the seed dispatch is wired in a                       follow-up slice, so no host path constructs it yet"
        )
    )]
    pub(crate) fn from_settings(
        light_count: u32,
        frame: u32,
        settings: &PrismWorldRestirSettings,
    ) -> Self {
        Self {
            capacity: settings.capacity.max(1),
            light_count,
            candidate_count: settings.candidate_count,
            frame,
            intensity: settings.intensity,
            m_cap: settings.m_cap,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Workgroup size (1-D) of the world-space `ReSTIR` injection entry point.
///
/// Must match `@workgroup_size(N, 1, 1)` in `world_restir_inject.wesl`; the
/// inject dispatch rounds the visible-point count up to a multiple of this and
/// the shader bounds-checks every invocation against the live point count.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inject dispatch extent constant; consumed once the inject pipeline is wired in                   a follow-up slice, so no host path reads it yet"
    )
)]
pub(crate) const WORLD_RESTIR_INJECT_WORKGROUP_SIZE: u32 = 64;

/// Per-point injection storage stride in bytes: two `vec4<f32>` lanes (a `vec3`
/// payload plus one trailing pad word each) = 32 bytes, matching the WGSL
/// `InjectPoint` struct's std430 layout and its 16-byte array stride.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inject point-buffer stride; consumed once the inject bind group binds the                   visible-point list in a follow-up slice, so no host path reads it yet"
    )
)]
pub(crate) const WORLD_RESTIR_INJECT_POINT_STRIDE: u64 = 32;

/// `GPU` twin of one visible shading point the injection pass claims a slot for
/// (the per-frame visible-point list the inject bind group binds at
/// `@binding(0)`).
///
/// The inject kernel hashes the world position + normal into its `SHARC` cell,
/// claims the open-addressed reservoir slot, and pre-seeds it with this
/// geometry (see `world_restir_inject.wesl`). The layout matches the WESL
/// `InjectPoint` struct: two `vec4` lanes (a `vec3` payload plus one trailing
/// pad word each) for 32 bytes, a multiple of the 16-byte std430 array stride
/// with no implicit padding.
///
/// Layout (two `vec4` lanes, std430, no implicit padding):
/// 0. `world_position.xyz` + one pad word
/// 1. `world_normal.xyz` + one pad word
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirInjectPoint {
    /// Visible-point (shading-point) world position; the cell centroid the
    /// seed pass re-exposes as the reservoir's `visible_point`.
    pub world_position: [f32; 3],
    /// Padding word rounding the first lane to the 16-byte std430 stride.
    pub _pad0: f32,
    /// Unit surface normal at the visible point; octahedrally binned into the
    /// hash key's `normal_bin`.
    pub world_normal: [f32; 3],
    /// Padding word rounding the record to the 16-byte std430 stride.
    pub _pad1: f32,
}

/// Immediate (push-constant) block consumed by the `world_restir_inject` entry
/// point.
///
/// One invocation per visible point: it hashes the point into its `SHARC` cell
/// (golden `spatial_hash::compute_key`), claims the open-addressed slot owning
/// that cell under a linear probe (golden `WorldHashGrid::find_or_alloc`), and
/// pre-seeds the slot with the cell geometry + `valid` flag. The hash-grid
/// tunables mirror the golden
/// [`prism_render_shading::gi::world_restir::spatial_hash`] `HashGridParams`
/// so the device cache keys agree with the CPU reference.
///
/// The `u32` counts occupy their own lane (unlike the fill block, which
/// float-encodes `light_count` and recovers it with a `u32(...)` cast); the
/// host writes them directly with no bitcast.
///
/// Layout (three `vec4` lanes, std430, no implicit padding):
/// 0. `camera_position.xyz` + `base_cell_size`
/// 1. `jitter.xyz` + `level_scale`
/// 2. (`capacity`, `point_count`, `normal_resolution`, `frame`)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirInjectParams {
    /// Camera world position, forwarded to the golden `grid_level` so the cell
    /// size grows with viewing distance (constant screen footprint).
    pub camera_position: [f32; 3],
    /// Edge length of a level-0 cell in world units (golden
    /// `HashGridParams::base_cell_size`, clamped positive by the shader).
    pub base_cell_size: f32,
    /// Grid-phase jitter in `[0, 1)^3` cell units (golden `HashGridParams`).
    pub jitter: [f32; 3],
    /// Distance-to-cell-size scale (golden `HashGridParams::level_scale`);
    /// `<= 0` disables level scaling (uniform fine grid).
    pub level_scale: f32,
    /// Reservoir-table capacity in slots (`>= 1`), the open-addressing modulus
    /// (golden `bucket_index`) and the probe-window bound.
    pub capacity: u32,
    /// Number of visible points to inject this frame, the dispatch extent and
    /// the per-invocation bounds check.
    pub point_count: u32,
    /// Side count of the `resolution x resolution` normal-bin grid (golden
    /// `HashGridParams::normal_resolution`, clamped to at least `1`).
    pub normal_resolution: u32,
    /// Monotonic frame index (reserved for per-frame slot-state reset / debug;
    /// carried for parity with the seed / fill immediates).
    pub frame: u32,
}

impl GpuWorldRestirInjectParams {
    /// Builds the inject immediate block from the camera position, the
    /// per-frame jitter + visible-point count + frame index, and the live
    /// settings (capacity floored at `1`, the grid tunables forwarded verbatim).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "inject immediate builder; called once the inject dispatch is wired in a                       follow-up slice, so no host path constructs it yet"
        )
    )]
    pub(crate) fn from_settings(
        camera_position: Vec3,
        jitter: Vec3,
        point_count: u32,
        frame: u32,
        settings: &PrismWorldRestirSettings,
    ) -> Self {
        Self {
            camera_position: camera_position.to_array(),
            base_cell_size: settings.base_cell_size,
            jitter: jitter.to_array(),
            level_scale: settings.level_scale,
            capacity: settings.capacity.max(1),
            point_count,
            normal_resolution: settings.normal_resolution,
            frame,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn reservoir_is_the_80_byte_five_lane_slot() {
        // Five `vec4` lanes (a `vec3` payload + trailing scalar each) = 80
        // bytes, a multiple of the 16-byte std430 array stride.
        assert_eq!(size_of::<GpuWorldRestirReservoir>(), 80);
        assert_eq!(align_of::<GpuWorldRestirReservoir>(), 4);
        assert_eq!(WORLD_RESTIR_RESERVOIR_STRIDE, 80);
        assert_eq!(WORLD_RESTIR_RESERVOIR_STRIDE % 16, 0);
        assert_eq!(
            size_of::<GpuWorldRestirReservoir>() as u64,
            WORLD_RESTIR_RESERVOIR_STRIDE
        );
    }

    #[test]
    fn fill_params_is_the_64_byte_four_lane_block() {
        // Four `vec4` lanes = 64 bytes, a multiple of the 16-byte immediate
        // alignment with no implicit padding.
        assert_eq!(size_of::<GpuWorldRestirFillParams>(), 64);
        assert_eq!(align_of::<GpuWorldRestirFillParams>(), 4);
    }

    #[test]
    fn empty_reservoir_is_zeroed_and_invalid() {
        let empty = GpuWorldRestirReservoir::EMPTY;
        assert_eq!(empty, GpuWorldRestirReservoir::zeroed());
        assert_eq!(empty.valid, 0);
        assert_eq!(empty.w, 0.0);
        assert_eq!(empty.m, 0.0);
    }

    #[test]
    fn fill_params_from_settings_forwards_the_grid_tunables() {
        let settings = PrismWorldRestirSettings::default();
        let params = GpuWorldRestirFillParams::from_settings(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::ZERO,
            7,
            42,
            &settings,
        );
        assert_eq!(params.camera_position, [1.0, 2.0, 3.0]);
        assert_eq!(params.capacity, settings.capacity);
        assert_eq!(params.light_count, 7);
        assert_eq!(params.frame, 42);
        assert_eq!(params.base_cell_size, settings.base_cell_size);
        assert_eq!(params.level_scale, settings.level_scale);
        assert_eq!(params.normal_resolution, settings.normal_resolution);
        assert_eq!(params.m_cap, settings.m_cap);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.spatial_samples, settings.spatial_samples);
        assert_eq!(params.spatial_radius, settings.spatial_radius);
    }

    #[test]
    fn fill_params_floors_capacity_at_one() {
        let settings = PrismWorldRestirSettings {
            capacity: 0,
            ..Default::default()
        };
        let params =
            GpuWorldRestirFillParams::from_settings(Vec3::ZERO, Vec3::ZERO, 0, 0, &settings);
        assert_eq!(params.capacity, 1);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(WORLD_RESTIR_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn light_is_the_32_byte_two_lane_record() {
        // Two `vec4` lanes (a `vec3` payload + trailing scalar each) = 32
        // bytes, a multiple of the 16-byte std430 array stride.
        assert_eq!(size_of::<GpuWorldRestirLight>(), 32);
        assert_eq!(align_of::<GpuWorldRestirLight>(), 4);
        assert_eq!(WORLD_RESTIR_LIGHT_STRIDE, 32);
        assert_eq!(WORLD_RESTIR_LIGHT_STRIDE % 16, 0);
        assert_eq!(
            size_of::<GpuWorldRestirLight>() as u64,
            WORLD_RESTIR_LIGHT_STRIDE
        );
    }

    #[test]
    fn seed_params_is_the_32_byte_two_lane_block() {
        // Two `vec4` lanes = 32 bytes, a multiple of the 16-byte immediate
        // alignment with no implicit padding.
        assert_eq!(size_of::<GpuWorldRestirSeedParams>(), 32);
        assert_eq!(align_of::<GpuWorldRestirSeedParams>(), 4);
    }

    #[test]
    fn seed_workgroup_constant_matches_the_shader() {
        assert_eq!(WORLD_RESTIR_SEED_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn seed_params_from_settings_forwards_the_tunables() {
        let settings = PrismWorldRestirSettings::default();
        let params = GpuWorldRestirSeedParams::from_settings(5, 11, &settings);
        assert_eq!(params.capacity, settings.capacity);
        assert_eq!(params.light_count, 5);
        assert_eq!(params.candidate_count, settings.candidate_count);
        assert_eq!(params.frame, 11);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.m_cap, settings.m_cap);
    }

    #[test]
    fn seed_params_floors_capacity_at_one() {
        let settings = PrismWorldRestirSettings {
            capacity: 0,
            ..Default::default()
        };
        let params = GpuWorldRestirSeedParams::from_settings(0, 0, &settings);
        assert_eq!(params.capacity, 1);
    }

    #[test]
    fn inject_point_is_the_32_byte_two_lane_record() {
        // Two `vec4` lanes (a `vec3` payload + trailing pad word each) = 32
        // bytes, a multiple of the 16-byte std430 array stride.
        assert_eq!(size_of::<GpuWorldRestirInjectPoint>(), 32);
        assert_eq!(align_of::<GpuWorldRestirInjectPoint>(), 4);
        assert_eq!(WORLD_RESTIR_INJECT_POINT_STRIDE, 32);
        assert_eq!(WORLD_RESTIR_INJECT_POINT_STRIDE % 16, 0);
        assert_eq!(
            size_of::<GpuWorldRestirInjectPoint>() as u64,
            WORLD_RESTIR_INJECT_POINT_STRIDE
        );
    }

    #[test]
    fn inject_params_is_the_48_byte_three_lane_block() {
        // Three `vec4` lanes = 48 bytes, a multiple of the 16-byte immediate
        // alignment with no implicit padding.
        assert_eq!(size_of::<GpuWorldRestirInjectParams>(), 48);
        assert_eq!(align_of::<GpuWorldRestirInjectParams>(), 4);
    }

    #[test]
    fn inject_workgroup_constant_matches_the_shader() {
        assert_eq!(WORLD_RESTIR_INJECT_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn inject_params_from_settings_forwards_the_grid_tunables() {
        let settings = PrismWorldRestirSettings::default();
        let params = GpuWorldRestirInjectParams::from_settings(
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(0.25, 0.5, 0.75),
            13,
            9,
            &settings,
        );
        assert_eq!(params.camera_position, [4.0, 5.0, 6.0]);
        assert_eq!(params.base_cell_size, settings.base_cell_size);
        assert_eq!(params.jitter, [0.25, 0.5, 0.75]);
        assert_eq!(params.level_scale, settings.level_scale);
        assert_eq!(params.capacity, settings.capacity);
        assert_eq!(params.point_count, 13);
        assert_eq!(params.normal_resolution, settings.normal_resolution);
        assert_eq!(params.frame, 9);
    }

    #[test]
    fn inject_params_floors_capacity_at_one() {
        let settings = PrismWorldRestirSettings {
            capacity: 0,
            ..Default::default()
        };
        let params =
            GpuWorldRestirInjectParams::from_settings(Vec3::ZERO, Vec3::ZERO, 0, 0, &settings);
        assert_eq!(params.capacity, 1);
    }
}
