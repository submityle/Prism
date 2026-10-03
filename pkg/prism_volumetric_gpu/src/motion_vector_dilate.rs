//! `wgpu` compute twin of the closest-depth velocity dilation primitive
//! ([`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth),
//! motion module `pkg/prism_render_architecture/src/motion/dilation.rs`).
//!
//! The `CPU` golden
//! [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth)
//! is the classic `TAA` fix that stops thin, fast foreground silhouettes from
//! tearing: for each output pixel it scans a border-clamped
//! `(2 * radius + 1)` square window of the paired
//! [`VelocityField`](prism_render_architecture::motion::dilation::VelocityField)
//! and [`DepthField`](prism_render_architecture::motion::dilation::DepthField),
//! seeds the search with the center pixel, and adopts a neighbor's velocity only
//! when that neighbor is *strictly* nearer the camera under the
//! [`DepthOrder`](prism_render_architecture::motion::dilation::DepthOrder). The
//! strict replacement plus the fixed row-major scan order make the result fully
//! deterministic: ties keep the incumbent, so the first-seen nearest pixel wins.
//!
//! [`GpuMotionVectorDilate`] is the on-device twin of that per-pixel reduction.
//! One thread solves one output pixel, scanning the identical clamped window in
//! the identical `ny`-outer / `nx`-inner order and applying the identical strict
//! [`is_closer`](prism_render_architecture::motion::dilation::DepthOrder::is_closer)
//! predicate, so a passing real-device parity test is direct evidence the ported
//! kernel reproduces the exact tie-breaking and edge-clamping the reference
//! fixes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The whole closest-depth pass: the window clamp
//! (`x_lo = x.saturating_sub(radius)`, `x_hi = (x + radius).min(width - 1)`, and
//! the same for `y`), the center-pixel seed, the strict-nearest neighbor scan,
//! and the velocity copy of the winning neighbor. The scan performs no floating
//! point arithmetic — only depth magnitude comparisons and a velocity copy — so
//! for inputs with no tied depths the `GPU` output velocity equals the `CPU`
//! golden *bit for bit*.
//!
//! # What stays on the host
//!
//! The variable-length container plumbing stays on the host, exactly as for the
//! other twins in this crate: the dimension-mismatch `None` guard of
//! [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth),
//! allocation of the row-major grids, and the empty-field short circuit. The
//! host uploads the velocity and depth grids as `std430` storage arrays, issues
//! a single dispatch, and reads back one velocity per pixel; a zero-pixel input
//! never reaches the device, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned index
//! arithmetic, magnitude comparison, `min`/`max`-free clamped bounds built from
//! compares, and a `vec2<f32>` copy — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `round`, no `smoothstep`, and only
//! `i32`/`u32`/`f32`/`bool` (no `u64`). No optional device feature is required,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::dilation`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// `WGSL` op selector mirroring
/// [`DepthOrder::SmallerIsCloser`](prism_render_architecture::motion::dilation::DepthOrder::SmallerIsCloser):
/// the smaller stored depth is nearer (classic forward-`Z`).
const ORDER_SMALLER_IS_CLOSER: u32 = 0;

/// `WGSL` op selector mirroring
/// [`DepthOrder::LargerIsCloser`](prism_render_architecture::motion::dilation::DepthOrder::LargerIsCloser):
/// the larger stored depth is nearer (reversed-`Z`).
const ORDER_LARGER_IS_CLOSER: u32 = 1;

/// The portable core-`WGSL` closest-depth dilation kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `dilate`
/// mirrors the `CPU` golden
/// [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth)
/// per-pixel reduction; see the module documentation for the algorithm.
const MOTION_VECTOR_DILATE_WGSL: &str = r#"
// Closest-depth velocity dilation twin: one thread computes one output pixel's
// dilated velocity by scanning a border-clamped (2*radius+1) square window in
// `ny`-outer / `nx`-inner order, seeding with the center pixel and adopting a
// neighbor's velocity only when strictly nearer the camera under `order`. It
// performs no floating-point arithmetic (only depth compares and a vec2 copy),
// so it reproduces the CPU golden `motion::dilation::dilate_closest_depth`
// bit-for-bit on tie-free inputs. The dimension guard, grid allocation and
// empty short-circuit stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::dilation；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Grid width in pixels.
    width: u32,
    // Grid height in pixels.
    height: u32,
    // Square dilation radius in pixels.
    radius: u32,
    // Depth ordering: 0 = smaller-is-closer, 1 = larger-is-closer.
    order: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> velocities: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec2<f32>>;

// Strict "candidate is nearer than reference" predicate, a verbatim port of the
// reference `DepthOrder::is_closer`: strictness keeps ties on the incumbent.
fn is_closer(candidate: f32, reference: f32) -> bool {
    if (params.order == 0u) {
        return candidate < reference;
    }
    return candidate > reference;
}

@compute @workgroup_size(64)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = params.width * params.height;
    if (idx >= count) {
        return;
    }

    let x = idx % params.width;
    let y = idx / params.width;

    // Seed with the center pixel so a fully tied window is a no-op copy.
    var best_depth = depths[idx];
    var best_velocity = velocities[idx];

    // Clamp the window to the borders, mirroring the reference's
    // `saturating_sub` low bound and `.min(width - 1)` / `.min(height - 1)`
    // high bound. `count > 0` guarantees width >= 1 and height >= 1.
    var x_lo = 0u;
    if (x > params.radius) {
        x_lo = x - params.radius;
    }
    var x_hi = x + params.radius;
    let x_max = params.width - 1u;
    if (x_hi > x_max) {
        x_hi = x_max;
    }
    var y_lo = 0u;
    if (y > params.radius) {
        y_lo = y - params.radius;
    }
    var y_hi = y + params.radius;
    let y_max = params.height - 1u;
    if (y_hi > y_max) {
        y_hi = y_max;
    }

    // Scan ny-outer, nx-inner exactly as the reference, replacing only on a
    // strictly nearer depth.
    for (var ny = y_lo; ny <= y_hi; ny = ny + 1u) {
        for (var nx = x_lo; nx <= x_hi; nx = nx + 1u) {
            let n_idx = ny * params.width + nx;
            let candidate_depth = depths[n_idx];
            if (is_closer(candidate_depth, best_depth)) {
                best_depth = candidate_depth;
                best_velocity = velocities[n_idx];
            }
        }
    }

    results[idx] = best_velocity;
}
"#;

/// Uniform parameters for one dispatch: the grid dimensions, the dilation radius
/// and the depth-ordering selector, matching `Params` in
/// [`MOTION_VECTOR_DILATE_WGSL`] (`16` bytes, four `u32` words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Grid width in pixels.
    width: u32,
    /// Grid height in pixels.
    height: u32,
    /// Square dilation radius in pixels.
    radius: u32,
    /// Depth ordering selector ([`ORDER_SMALLER_IS_CLOSER`] or
    /// [`ORDER_LARGER_IS_CLOSER`]).
    order: u32,
}

/// `repr(C)` `std430` layout of one velocity: two `f32` components matching the
/// `WGSL` `vec2<f32>` element (`8`-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVelocity {
    /// `X` component (previous-to-current pixel displacement along `+x`).
    x: f32,
    /// `Y` component (previous-to-current pixel displacement along `+y`).
    y: f32,
}

/// Which numeric direction is "closer to the camera", a twin of the reference
/// [`DepthOrder`](prism_render_architecture::motion::dilation::DepthOrder) that
/// keeps this crate free of a re-exported golden type.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::motion::dilation`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MotionVectorDilateOrder {
    /// Smaller stored depth is nearer (classic forward-`Z`, `0` = near),
    /// mirroring
    /// [`DepthOrder::SmallerIsCloser`](prism_render_architecture::motion::dilation::DepthOrder::SmallerIsCloser).
    SmallerIsCloser,
    /// Larger stored depth is nearer (reversed-`Z`, `1` = near), mirroring
    /// [`DepthOrder::LargerIsCloser`](prism_render_architecture::motion::dilation::DepthOrder::LargerIsCloser).
    LargerIsCloser,
}

impl MotionVectorDilateOrder {
    /// Returns the `WGSL` op selector for this ordering.
    fn op_code(self) -> u32 {
        match self {
            MotionVectorDilateOrder::SmallerIsCloser => ORDER_SMALLER_IS_CLOSER,
            MotionVectorDilateOrder::LargerIsCloser => ORDER_LARGER_IS_CLOSER,
        }
    }
}

/// One dilation request: a row-major velocity grid and its paired depth grid,
/// plus the dilation radius and depth ordering.
///
/// Element `(x, y)` lives at linear index `y * width + x`, matching the
/// reference pixel convention (origin top-left, `+x` right, `+y` down). Both
/// slices must hold exactly `width * height` elements.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::motion::dilation`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug)]
pub struct MotionVectorDilateInput<'a> {
    /// Grid width in pixels.
    pub width: u32,
    /// Grid height in pixels.
    pub height: u32,
    /// Square dilation radius in pixels (a `radius` of `0` copies the input).
    pub radius: u32,
    /// Depth ordering deciding which direction is nearer the camera.
    pub order: MotionVectorDilateOrder,
    /// Row-major velocity grid, `width * height` elements of `[x, y]`.
    pub velocities: &'a [[f32; 2]],
    /// Row-major depth grid, `width * height` elements.
    pub depths: &'a [f32],
}

/// A compiled, reusable closest-depth velocity dilation compute pipeline,
/// twinning the `CPU` golden
/// [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth).
pub struct GpuMotionVectorDilate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
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

impl GpuMotionVectorDilate {
    /// Compiles the closest-depth dilation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionVectorDilate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate"),
            source: ShaderSource::Wgsl(MOTION_VECTOR_DILATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("dilate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionVectorDilate {
            module,
            layout,
            pipeline,
        }
    }

    /// Dilates every pixel of `input` and returns one `[x, y]` velocity per
    /// pixel in row-major order.
    ///
    /// Each output equals the `CPU` golden
    /// [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth)
    /// at the matching pixel; because the kernel only compares depths and copies
    /// a velocity, the agreement is bit-exact for tie-free depth inputs. An
    /// empty grid (`width * height == 0`) returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics when `input.velocities` or `input.depths` does not hold exactly
    /// `width * height` elements, mirroring the reference length contract.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, input: &MotionVectorDilateInput) -> Vec<[f32; 2]> {
        let count = (input.width as usize) * (input.height as usize);
        if count == 0 {
            return Vec::new();
        }
        assert_eq!(
            input.velocities.len(),
            count,
            "velocity grid length must equal width * height"
        );
        assert_eq!(
            input.depths.len(),
            count,
            "depth grid length must equal width * height"
        );

        let device = ctx.device();

        let params = GpuParams {
            width: input.width,
            height: input.height,
            radius: input.radius,
            order: input.order.op_code(),
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let velocities: Vec<GpuVelocity> = input
            .velocities
            .iter()
            .map(|v| GpuVelocity { x: v[0], y: v[1] })
            .collect();
        let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: BufferUsages::STORAGE,
        });
        let depths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_depths"),
            contents: bytemuck::cast_slice(input.depths),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuVelocity>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: velocities_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: depths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_vector_dilate_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_vector_dilate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output pixel, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuVelocity>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(|v| [v.x, v.y]).collect()
    }
}
