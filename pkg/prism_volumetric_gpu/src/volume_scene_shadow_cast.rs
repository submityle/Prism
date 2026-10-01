//! `wgpu` compute twin of the `CPU` golden particle-volume soft-shadow caster
//! [`VolumeShadowCaster`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster)
//! (design section 20, "也可向场景投体积软阴影（高档）"; see
//! `docs/prism_particle_engine_design_zh.md` section 20).
//!
//! A particle volume acts as an occluder that darkens the lit scene behind it:
//! along the "scene receiver -> light" direction the volume's extinction is
//! integrated front-to-back as an opacity product, and the surviving product is
//! the soft-shadow attenuation in `0..=1`. The `CPU` golden
//! [`VolumeShadowCaster`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster)
//! owns that march math; [`GpuVolumeShadowCast`] is the on-device twin,
//! validated against that reference so a passing real-device parity test is
//! direct evidence the ported kernel marches the same transmittance, not merely
//! that its shader compiles.
//!
//! # Algorithm
//!
//! The twin reproduces the reference exactly. For each ray the direction is
//! normalized by [`Vec3::normalize_or_zero`](prism_render_architecture::particle::Vec3::normalize_or_zero)
//! (the one place `sqrt` is used); a degenerate direction or a collapsed step
//! length yields a fully-lit `1.0` so a malformed query never fabricates
//! occlusion. Otherwise the ray samples the shared
//! [`FroxelDensityField`](prism_render_architecture::particle::volumetrics::FroxelDensityField)
//! at `start + direction * step_length * (i + 0.5)` for `i` in `0..max_steps`
//! (the step midpoint), folding a per-step survival
//! `1 - clamp(density * sigma * step_length, 0, 1)` into a running transmittance
//! product. The same march, recorded at each step, bakes a
//! [`DeepOpacityLayer`](prism_render_architecture::particle::volumetrics::DeepOpacityLayer)
//! curve: layer `0` at depth `0.0` transmittance `1.0`, then one per step at
//! depth `step_length * (i + 1)` carrying the running transmittance after that
//! step. The final product is the `cast` / `transmittance_along` attenuation and
//! the full curve is the `bake_light_ray` deep opacity map.
//!
//! The on-device density lookup mirrors
//! [`FroxelGrid::cell_index`](prism_render_architecture::particle::volumetrics::FroxelGrid::cell_index):
//! the per-axis cell index is `floor((world - origin) / cell_size)` with the
//! same out-of-bounds and non-positive-cell-size guards, flattened row-major as
//! `x + y * dims.x + z * dims.x * dims.y`, returning `0.0` outside the grid.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `min`, `max`,
//! `clamp`, `floor` (through the truncating `u32` cast), `+ - * /` and integer
//! comparison — with no `exp`, `pow`, `log`, trig or `inverseSqrt` and no
//! optional device feature, so it runs unmodified on Metal, Vulkan and DX12.
//! The forbidden `Beer-Lambert` `exp` is replaced by the same algebraic
//! increment product `transmittance *= 1 - alpha` the `CPU` reference uses.
//!
//! # Correctness model
//!
//! The recurrence contains no transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form algebra in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a tolerance (`abs_diff < 1e-5` or `rel_diff < 1e-4`), tight enough to
//! catch a genuinely wrong port yet loose enough to admit legal fused
//! multiply-add contraction over the per-step survival product chain.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Nelson-Max front-to-back `1 - alpha` volume compositing
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::volumetrics::{DeepOpacityLayer, FroxelDensityField};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` shadow-cast kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` march exactly; see the
/// module documentation for the algorithm.
const VOLUME_SCENE_SHADOW_CAST_WGSL: &str = r#"
// Particle-volume soft-shadow caster twin: one thread per ray normalizes the
// shadow direction, marches the froxel density field front-to-back in step
// midpoints, and accumulates transmittance by the algebraic step-opacity
// recurrence `transmittance *= 1 - clamp(density * sigma * step_length, 0, 1)`,
// writing `layer_stride` `(depth, transmittance)` layers plus the valid layer
// count. It mirrors the CPU golden `VolumeShadowCaster`, uses only the portable
// core-WGSL subset (sqrt/min/max/clamp, truncating u32 cast and + - * /), and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Nelson-Max front-to-back `1 - alpha` volume compositing;
// no Unreal Engine source or derived code.

// Squared-length threshold below which a direction normalizes to zero; mirrors
// the `EPS_LEN_SQ` guard in `Vec3::normalize_or_zero`.
const EPS_LEN_SQ: f32 = 1e-12;
// Degenerate-direction / collapsed-step guard; mirrors the `CMP_EPS` guard in
// the CPU reference's `transmittance_along`.
const CMP_EPS: f32 = 1e-6;
// Non-positive-cell-size / grid-bounds guard; mirrors the `EPS` floor in
// `FroxelGrid::cell_index`.
const CELL_EPS: f32 = 1e-6;
// Fraction of a march step at which the volume is sampled: the step midpoint,
// mirroring the CPU reference's `STEP_MIDPOINT`.
const STEP_MIDPOINT: f32 = 0.5;

struct Params {
    // Cell counts along X, Y, Z.
    dims: vec3<u32>,
    // Number of rays marched in this dispatch (one thread each).
    ray_count: u32,
    // World-space minimum corner of the grid (cell [0, 0, 0]).
    origin: vec3<f32>,
    // Density-to-opacity scale `sigma_t` (already floored at zero by the host).
    sigma: f32,
    // World-space size of one cell along each axis.
    cell_size: vec3<f32>,
    // World-space march increment.
    step_length: f32,
    // Maximum number of march steps.
    max_steps: u32,
    // Layers written per ray (`max_steps + 1`): the near seed plus one per step.
    layer_stride: u32,
    // Padding to round the uniform struct to a 16-byte multiple.
    pad0: u32,
    pad1: u32,
}

// One recorded layer. 16-byte std430 stride: the pair plus two pad words.
struct Layer {
    depth: f32,
    transmittance: f32,
    pad0: f32,
    pad1: f32,
}

// One ray. 32-byte std430 stride: two vec3 payloads each padded to 16 bytes.
struct Ray {
    start: vec3<f32>,
    pad0: f32,
    direction: vec3<f32>,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> density: array<f32>;
@group(0) @binding(2) var<storage, read> rays: array<Ray>;
@group(0) @binding(3) var<storage, read_write> layers: array<Layer>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;

// Clamps `x` into [0, 1], mirroring the reference's first-order absorber.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Unit vector along `v`, or the zero vector when `v` is numerically zero;
// mirrors `Vec3::normalize_or_zero` (the single use of `sqrt`).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Resolves one axis cell index, or a negative sentinel when the cell size is
// non-positive or the local coordinate is out of bounds; mirrors the CPU
// `axis_index`. The truncating `u32` cast equals `floor` for the non-negative
// in-bounds local coordinate.
fn axis_index(local: f32, size: f32, dim: u32) -> i32 {
    if (size <= CELL_EPS || local < 0.0) {
        return -1;
    }
    let idx = u32(local / size);
    if (idx < dim) {
        return i32(idx);
    }
    return -1;
}

// Density at a world point (`0.0` outside the grid); mirrors `density_at` plus
// `FroxelGrid::cell_index` and `linear_index`.
fn density_at(point: vec3<f32>) -> f32 {
    let local = point - params.origin;
    let ix = axis_index(local.x, params.cell_size.x, params.dims.x);
    let iy = axis_index(local.y, params.cell_size.y, params.dims.y);
    let iz = axis_index(local.z, params.cell_size.z, params.dims.z);
    if (ix < 0 || iy < 0 || iz < 0) {
        return 0.0;
    }
    // Row-major flatten `x + y * dims.x + z * dims.x * dims.y`.
    let ux = u32(ix);
    let uy = u32(iy);
    let uz = u32(iz);
    let index = ux + uy * params.dims.x + uz * params.dims.x * params.dims.y;
    return density[index];
}

@compute @workgroup_size(64)
fn march_shadows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray = gid.x;
    if (ray >= params.ray_count) {
        return;
    }

    let base = ray * params.layer_stride;

    // Layer 0: the fully-lit near side; nothing has absorbed yet.
    var seed: Layer;
    seed.depth = 0.0;
    seed.transmittance = 1.0;
    seed.pad0 = 0.0;
    seed.pad1 = 0.0;
    layers[base] = seed;

    let dir = normalize_or_zero(rays[ray].direction);
    // A degenerate direction or collapsed step length stays fully lit: one
    // layer, transmittance 1.0, exactly like the CPU early return.
    if (dot(dir, dir) <= CMP_EPS || params.step_length <= CMP_EPS) {
        // Fill the unused tail with the seed so the readback buffer is
        // deterministic; the host only reads the first `count` layers.
        var tail = 1u;
        while (tail < params.layer_stride) {
            layers[base + tail] = seed;
            tail = tail + 1u;
        }
        counts[ray] = 1u;
        return;
    }

    let start = rays[ray].start;
    var transmittance = 1.0;
    var step = 0u;
    while (step < params.max_steps) {
        let offset = params.step_length * (f32(step) + STEP_MIDPOINT);
        let point = start + dir * offset;
        let d = density_at(point);
        let opacity = clamp01(d * params.sigma * params.step_length);
        transmittance = transmittance * (1.0 - opacity);
        let depth = params.step_length * (f32(step) + 1.0);
        var out: Layer;
        out.depth = depth;
        out.transmittance = transmittance;
        out.pad0 = 0.0;
        out.pad1 = 0.0;
        layers[base + step + 1u] = out;
        step = step + 1u;
    }
    counts[ray] = params.layer_stride;
}
"#;

/// Uniform parameters for one shadow-cast dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`VOLUME_SCENE_SHADOW_CAST_WGSL`]: the `12`-byte
/// `dims` vector packs `ray_count` into its trailing word, then the `16`-byte
/// aligned `origin` plus `sigma`, the `16`-byte aligned `cell_size` plus
/// `step_length`, the two index words and two pad words — `64` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Cell counts along `X`, `Y`, `Z`.
    dims: [u32; 3],
    /// Number of rays marched in this dispatch.
    ray_count: u32,
    /// World-space minimum corner of the grid.
    origin: [f32; 3],
    /// Density-to-opacity scale `sigma_t`, floored at zero by the host.
    sigma: f32,
    /// World-space size of one cell along each axis.
    cell_size: [f32; 3],
    /// World-space march increment.
    step_length: f32,
    /// Maximum number of march steps.
    max_steps: u32,
    /// Layers written per ray (`max_steps + 1`).
    layer_stride: u32,
    /// Padding word keeping the struct a `16`-byte multiple.
    pad0: u32,
    /// Padding word keeping the struct a `16`-byte multiple.
    pad1: u32,
}

/// One march ray uploaded to the kernel. `repr(C)` `std430` layout matching
/// `Ray` in the shader: a `vec3` `start` plus a pad word, then a `vec3`
/// `direction` plus a pad word — `32` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRay {
    /// World-space ray origin (the receiver / entry point).
    start: [f32; 3],
    /// Padding word keeping `direction` `16`-byte aligned.
    pad0: f32,
    /// Shadow direction before normalization (the kernel normalizes it).
    direction: [f32; 3],
    /// Padding word keeping the stride a `16`-byte multiple.
    pad1: f32,
}

/// One recorded layer as read back. `16`-byte `std430` stride matching `Layer`
/// in the shader: the `(depth, transmittance)` pair plus two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLayer {
    /// Distance along the shadow ray at which this layer was recorded.
    depth: f32,
    /// Surviving transmittance at `depth`, in `0..=1`.
    transmittance: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One shadow-march ray: an origin and a to-be-normalized direction.
///
/// Mirrors the arguments of
/// [`VolumeShadowCaster::transmittance_along`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster::transmittance_along).
/// For a `cast` query the caller resolves the "toward light" direction with
/// [`SceneLight::toward_light`](prism_render_architecture::particle::volume_scene_shadow_cast::SceneLight::toward_light)
/// and passes it here; for a `bake_light_ray` query `start` is the ray entry
/// and `direction` is the light's travel direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeShadowRay {
    /// World-space ray origin.
    pub start: [f32; 3],
    /// Shadow direction before normalization.
    pub direction: [f32; 3],
}

/// The result of marching one [`VolumeShadowRay`] on the device.
///
/// `transmittance` reproduces
/// [`VolumeShadowCaster::transmittance_along`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster::transmittance_along)
/// (and therefore `cast` once the light direction is resolved); `layers`
/// reproduces
/// [`VolumeShadowCaster::bake_light_ray`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster::bake_light_ray):
/// a single fully-lit layer for a degenerate ray, otherwise `max_steps + 1`
/// layers ascending in depth and non-increasing in transmittance.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowMarch {
    /// Final soft-shadow attenuation in `0..=1` (`1.0` fully lit).
    pub transmittance: f32,
    /// The baked deep opacity map along the ray.
    pub layers: Vec<DeepOpacityLayer>,
}

/// A compiled, reusable particle-volume soft-shadow caster pipeline.
pub struct GpuVolumeShadowCast {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVolumeShadowCast {
    /// Compiles the shadow-cast kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVolumeShadowCast {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast"),
            source: ShaderSource::Wgsl(VOLUME_SCENE_SHADOW_CAST_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("march_shadows"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVolumeShadowCast {
            module,
            layout,
            pipeline,
        }
    }

    /// Marches every ray through `field` and returns one [`ShadowMarch`] per ray
    /// in input order.
    ///
    /// `extinction` is the density-to-opacity scale `sigma_t` (a negative value
    /// is floored to zero, exactly like
    /// [`VolumeShadowCaster::new`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster::new)),
    /// `step_length` is the world-space march increment and `max_steps` caps the
    /// ray length. Each returned `transmittance` equals
    /// [`VolumeShadowCaster::transmittance_along`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster::transmittance_along)
    /// on the same ray and each `layers` curve equals
    /// [`VolumeShadowCaster::bake_light_ray`](prism_render_architecture::particle::volume_scene_shadow_cast::VolumeShadowCaster::bake_light_ray),
    /// to within the tolerance documented on this module. An empty `rays` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return.
    #[must_use]
    pub fn march(
        &self,
        ctx: &GpuContext,
        field: &FroxelDensityField,
        extinction: f32,
        step_length: f32,
        max_steps: u32,
        rays: &[VolumeShadowRay],
    ) -> Vec<ShadowMarch> {
        if rays.is_empty() {
            return Vec::new();
        }

        let grid = field.grid();
        // Mirror `VolumeShadowCaster::new`: a negative extinction never adds
        // light, so it is floored to zero.
        let sigma = if extinction > 0.0 { extinction } else { 0.0 };
        // One near seed layer plus one layer per march step.
        let layer_stride = max_steps + 1;

        let gpu_rays: Vec<GpuRay> = rays
            .iter()
            .map(|r| GpuRay {
                start: r.start,
                pad0: 0.0,
                direction: r.direction,
                pad1: 0.0,
            })
            .collect();

        let gpu_params = Params {
            dims: grid.dims,
            ray_count: rays.len() as u32,
            origin: [grid.origin.x, grid.origin.y, grid.origin.z],
            sigma,
            cell_size: [grid.cell_size.x, grid.cell_size.y, grid.cell_size.z],
            step_length,
            max_steps,
            layer_stride,
            pad0: 0,
            pad1: 0,
        };

        let device = ctx.device();

        // The density grid uploaded verbatim; storage buffers cannot be
        // zero-sized, so a degenerate empty grid is padded to one zero cell.
        let densities = field.densities();
        let density_upload: Vec<f32> = if densities.is_empty() {
            vec![0.0]
        } else {
            densities.to_vec()
        };

        let layers_len = rays.len() * (layer_stride as usize);
        let layers_bytes = (layers_len as u64) * (size_of::<GpuLayer>() as u64);
        let counts_bytes = (rays.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let density_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_density"),
            contents: bytemuck::cast_slice(&density_upload),
            usage: BufferUsages::STORAGE,
        });
        let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_rays"),
            contents: bytemuck::cast_slice(&gpu_rays),
            usage: BufferUsages::STORAGE,
        });
        let layers_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_layers"),
            size: layers_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let layers_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_layers_stage"),
            size: layers_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_counts"),
            size: counts_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_counts_stage"),
            size: counts_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: density_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: rays_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: layers_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: counts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_volume_scene_shadow_cast_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_volume_scene_shadow_cast_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per ray, in workgroups of 64 (the kernel's size).
            let groups = (rays.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&layers_buf, 0, &layers_stage, 0, layers_bytes);
        encoder.copy_buffer_to_buffer(&counts_buf, 0, &counts_stage, 0, counts_bytes);
        ctx.queue().submit([encoder.finish()]);

        layers_stage.slice(..).map_async(MapMode::Read, |_| {});
        counts_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let layers_view = layers_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped layers readback range should be available after poll");
        let gpu_layers = bytemuck::cast_slice::<u8, GpuLayer>(&layers_view).to_vec();
        drop(layers_view);
        layers_stage.unmap();

        let counts_view = counts_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped counts readback range should be available after poll");
        let gpu_counts = bytemuck::cast_slice::<u8, u32>(&counts_view).to_vec();
        drop(counts_view);
        counts_stage.unmap();

        debug_assert_eq!(gpu_layers.len(), layers_len);
        debug_assert_eq!(gpu_counts.len(), rays.len());

        let stride = layer_stride as usize;
        gpu_counts
            .iter()
            .enumerate()
            .map(|(ray, &count)| {
                let base = ray * stride;
                let valid = (count as usize).min(stride);
                let layers: Vec<DeepOpacityLayer> = gpu_layers[base..base + valid]
                    .iter()
                    .map(|layer| DeepOpacityLayer::new(layer.depth, layer.transmittance))
                    .collect();
                // The final attenuation is the deepest recorded layer: the near
                // seed for a degenerate ray, the full product otherwise.
                let transmittance = layers.last().map_or(1.0, |layer| layer.transmittance);
                ShadowMarch {
                    transmittance,
                    layers,
                }
            })
            .collect()
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
