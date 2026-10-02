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
}
