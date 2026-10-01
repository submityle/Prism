//! `wgpu` compute twin of the `Brown-Conrady` radial + tangential
//! lens-distortion model
//! ([`lens_distortion`](prism_render_architecture::particle::lens_distortion),
//! design §16-§21).
//!
//! A real lens does not image a straight world line to a straight image line: a
//! wide-angle element bows lines outward (*barrel*) and a telephoto pinches them
//! inward (*pincushion*), while a decentred element adds an asymmetric
//! *tangential* smear. The photogrammetry standard describing both is the
//! `Brown-Conrady` polynomial: a radial series in even powers of the image
//! radius plus a two-term tangential correction. The `CPU` golden
//! [`lens_distortion`](prism_render_architecture::particle::lens_distortion)
//! owns that math; [`GpuLensDistortion`] is the on-device twin that runs one
//! thread per point and reproduces the same coordinates the batch form
//! [`distort_grid`](prism_render_architecture::particle::lens_distortion::distort_grid)
//! produces. A passing real-device parity test is therefore direct evidence the
//! ported kernels evaluate the same polynomial, the same tangential decentring
//! and the same fixed-point inverse the reference does, not merely that the
//! shaders compile.
//!
//! # What is twinned
//!
//! Three maps are reproduced, one `WGSL` entry point each, all sharing the same
//! coefficient uniform and point buffers:
//!
//! * [`GpuLensDistortion::distort`] mirrors the forward map
//!   [`distort`](prism_render_architecture::particle::lens_distortion::distort):
//!   scale by the radial factor `1 + k1*r^2 + k2*r^4 + k3*r^6` and add the
//!   tangential offset.
//! * [`GpuLensDistortion::undistort`] mirrors the inverse
//!   [`undistort`](prism_render_architecture::particle::lens_distortion::undistort):
//!   a fixed
//!   [`UNDISTORT_ITERATIONS`](prism_render_architecture::particle::lens_distortion::UNDISTORT_ITERATIONS)-step
//!   fixed-point refinement seeded at the observed point, using only
//!   multiply/add/divide.
//! * [`GpuLensDistortion::distort_radius`] mirrors the scalar radial map
//!   [`distort_radius`](prism_render_architecture::particle::lens_distortion::distort_radius):
//!   `r * (1 + k1*r^2 + k2*r^4 + k3*r^6)` along a ray through the optical
//!   centre.
//!
//! The internal helpers `radial_factor` and `tangential_offset` are reproduced
//! term for term, forming the even powers by repeated multiplication
//! (`r4 = r2*r2`, `r6 = r4*r2`) and never by an integer-power intrinsic, exactly
//! as the reference does.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `+ - * /` and
//! unsigned integer index arithmetic — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `sqrt`, `smoothstep` or optional device feature, so they run
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no transcendental call
//! on this path: the whole model is multiplies, adds and the divides of the
//! fixed-point inverse, whose radial-factor denominator stays close to one for
//! the physical coefficients a real lens exhibits and so never approaches zero.
//!
//! # Correctness model
//!
//! Each output coordinate is a fixed, non-reorderable sequence of multiplies,
//! adds and (for the inverse) divides, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, and across the ten inverse
//! iterations that perturbation compounds. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a
//! genuinely wrong port (a swapped tangential sign, a dropped radial term, a
//! wrong iteration count) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Brown-Conrady` radial + tangential lens-distortion
//! model and its `OpenCV`-style fixed-point inverse plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::lens_distortion::DistortionCoeffs;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly width
/// the sibling twins use for a one-thread-per-element one-`D` dispatch.
const WORKGROUP_SIZE: u32 = 64;

/// Inlined `WGSL` for the three lens-distortion kernels.
///
/// `distort_main` is the forward map, `undistort_main` the fixed-point inverse
/// and `distort_radius_main` the scalar radial map (reading the radius from each
/// point's `x` slot and leaving `y` zero). All three share one coefficient
/// uniform and the same `read` / `read_write` point buffers. The even powers are
/// formed by repeated multiplication and the inverse loop runs the same fixed
/// `UNDISTORT_ITERATIONS` count the reference does, so `CPU` and `GPU` evaluate
/// the identical closed form. The kernels use only the portable core-`WGSL`
/// subset (`+ - * /` and unsigned index math) and take no optional feature, so
/// they run unmodified on `Metal`, `Vulkan` and `DX12`.
///
/// Provenance: standard `Brown-Conrady` lens distortion; no third-party engine
/// source or derived code.
const LENS_DISTORTION_WGSL: &str = r#"
// Fixed-point refinement count for the inverse, matching the reference
// `UNDISTORT_ITERATIONS`.
const UNDISTORT_ITERATIONS: u32 = 10u;

struct Params {
    // Brown-Conrady coefficients: radial (k*) then tangential (p*).
    k1: f32,
    k2: f32,
    k3: f32,
    p1: f32,
    p2: f32,
    // Number of points to process (one thread each).
    count: u32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

// One normalized image-plane coordinate. Matches `GpuPoint` in
// `src/lens_distortion.rs`.
struct Point {
    x: f32,
    y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> results: array<Point>;

// Radial magnification factor `1 + k1*r^2 + k2*r^4 + k3*r^6`, with the even
// powers formed by repeated multiplication, mirroring the reference
// `radial_factor` term for term.
fn radial_factor(r2: f32) -> f32 {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    return 1.0 + params.k1 * r2 + params.k2 * r4 + params.k3 * r6;
}

// Brown-Conrady tangential (decentring) offset `[dx, dy]`, mirroring the
// reference `tangential_offset` exactly.
fn tangential_offset(x: f32, y: f32, r2: f32) -> vec2<f32> {
    let dx = 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x);
    let dy = params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y;
    return vec2<f32>(dx, dy);
}

// Forward map: scale by the radial factor and add the tangential offset,
// mirroring the reference `distort`.
fn distort_point(p: vec2<f32>) -> vec2<f32> {
    let r2 = p.x * p.x + p.y * p.y;
    let radial = radial_factor(r2);
    let t = tangential_offset(p.x, p.y, r2);
    return vec2<f32>(p.x * radial + t.x, p.y * radial + t.y);
}

// Inverse map: a fixed `UNDISTORT_ITERATIONS`-step fixed-point refinement seeded
// at the observed point, using only multiply/add/divide, mirroring the
// reference `undistort`.
fn undistort_point(d: vec2<f32>) -> vec2<f32> {
    var x = d.x;
    var y = d.y;
    for (var i = 0u; i < UNDISTORT_ITERATIONS; i = i + 1u) {
        let r2 = x * x + y * y;
        let radial = radial_factor(r2);
        let t = tangential_offset(x, y, r2);
        x = (d.x - t.x) / radial;
        y = (d.y - t.y) / radial;
    }
    return vec2<f32>(x, y);
}

// Scalar radial map `r * (1 + k1*r^2 + k2*r^4 + k3*r^6)`, mirroring the
// reference `distort_radius`.
fn distort_radius(r: f32) -> f32 {
    let r2 = r * r;
    return r * radial_factor(r2);
}

@compute @workgroup_size(64)
fn distort_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = vec2<f32>(points[idx].x, points[idx].y);
    let o = distort_point(p);
    results[idx].x = o.x;
    results[idx].y = o.y;
}

@compute @workgroup_size(64)
fn undistort_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let d = vec2<f32>(points[idx].x, points[idx].y);
    let o = undistort_point(d);
    results[idx].x = o.x;
    results[idx].y = o.y;
}

@compute @workgroup_size(64)
fn distort_radius_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // The scalar radius travels in each point's `x` slot; `y` is unused on
    // input and written zero so the readback shape stays uniform.
    let r = points[idx].x;
    results[idx].x = distort_radius(r);
    results[idx].y = 0.0;
}
"#;

/// One normalized image-plane coordinate as uploaded. `8`-byte `repr(C)`
/// matching `Point` in [`LENS_DISTORTION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPoint {
    /// `x` component (or the scalar radius for the radial kernel).
    x: f32,
    /// `y` component (unused on input for the radial kernel).
    y: f32,
}

/// Uniform parameters for one dispatch. `32`-byte `repr(C)` matching `Params` in
/// [`LENS_DISTORTION_WGSL`]: the five `Brown-Conrady` coefficients, the point
/// count and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// First radial coefficient (scales `r^2`).
    k1: f32,
    /// Second radial coefficient (scales `r^4`).
    k2: f32,
    /// Third radial coefficient (scales `r^6`).
    k3: f32,
    /// First tangential coefficient.
    p1: f32,
    /// Second tangential coefficient.
    p2: f32,
    /// Number of points to process.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

impl GpuParams {
    /// Packs one coefficient block and the point `count` into the uniform
    /// layout the shader expects.
    fn new(coeffs: &DistortionCoeffs, count: usize) -> GpuParams {
        GpuParams {
            k1: coeffs.k1,
            k2: coeffs.k2,
            k3: coeffs.k3,
            p1: coeffs.p1,
            p2: coeffs.p2,
            count: count as u32,
            pad0: 0,
            pad1: 0,
        }
    }
}

/// A compiled, reusable lens-distortion pipeline trio (forward, inverse and
/// scalar radial), sharing one bind-group layout and shader module.
pub struct GpuLensDistortion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_distort: ComputePipeline,
    pipeline_undistort: ComputePipeline,
    pipeline_distort_radius: ComputePipeline,
}

impl GpuLensDistortion {
    /// Compiles the forward, inverse and scalar-radial kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLensDistortion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_lens_distortion"),
            source: ShaderSource::Wgsl(LENS_DISTORTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_lens_distortion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_lens_distortion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_distort = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lens_distortion_distort_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("distort_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_undistort = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lens_distortion_undistort_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("undistort_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_distort_radius = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lens_distortion_distort_radius_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("distort_radius_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLensDistortion {
            module,
            layout,
            pipeline_distort,
            pipeline_undistort,
            pipeline_distort_radius,
        }
    }

    /// Applies the forward `Brown-Conrady` map to every point in `points`,
    /// returning one distorted coordinate per input in order.
    ///
    /// The result for point `p` equals the `CPU`
    /// [`distort`](prism_render_architecture::particle::lens_distortion::distort)`(coeffs, p)`
    /// to within the tolerance documented on this module. An empty `points`
    /// slice yields an empty vector — storage buffers cannot be zero-sized, so
    /// it is handled by an early return with no dispatch issued.
    #[must_use]
    pub fn distort(
        &self,
        ctx: &GpuContext,
        coeffs: &DistortionCoeffs,
        points: &[[f32; 2]],
    ) -> Vec<[f32; 2]> {
        self.run(ctx, &self.pipeline_distort, coeffs, points)
    }

    /// Inverts the forward map for every point in `distorted`, recovering the
    /// ideal coordinate [`distort`](Self::distort) would have carried there.
    ///
    /// The result for point `p` equals the `CPU`
    /// [`undistort`](prism_render_architecture::particle::lens_distortion::undistort)`(coeffs, p)`
    /// to within the tolerance documented on this module. An empty slice yields
    /// an empty vector with no dispatch issued.
    #[must_use]
    pub fn undistort(
        &self,
        ctx: &GpuContext,
        coeffs: &DistortionCoeffs,
        distorted: &[[f32; 2]],
    ) -> Vec<[f32; 2]> {
        self.run(ctx, &self.pipeline_undistort, coeffs, distorted)
    }

    /// Applies the purely radial map to every scalar radius in `radii`,
    /// returning one distorted radius per input in order.
    ///
    /// The result for radius `r` equals the `CPU`
    /// [`distort_radius`](prism_render_architecture::particle::lens_distortion::distort_radius)`(coeffs, r)`
    /// to within the tolerance documented on this module. An empty slice yields
    /// an empty vector with no dispatch issued. Each radius travels in a point's
    /// `x` slot and the result is read back from the output `x`.
    #[must_use]
    pub fn distort_radius(
        &self,
        ctx: &GpuContext,
        coeffs: &DistortionCoeffs,
        radii: &[f32],
    ) -> Vec<f32> {
        let points: Vec<[f32; 2]> = radii.iter().map(|&r| [r, 0.0]).collect();
        let out = self.run(ctx, &self.pipeline_distort_radius, coeffs, &points);
        out.into_iter().map(|p| p[0]).collect()
    }

    /// Uploads `points`, dispatches `pipeline` one thread per point and reads the
    /// result back in order.
    ///
    /// Shared by every public entry point; the only difference between them is
    /// which pipeline is bound. An empty input short-circuits to an empty vector
    /// so no zero-sized storage buffer is ever allocated.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        coeffs: &DistortionCoeffs,
        points: &[[f32; 2]],
    ) -> Vec<[f32; 2]> {
        if points.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_points: Vec<GpuPoint> = points
            .iter()
            .map(|p| GpuPoint { x: p[0], y: p[1] })
            .collect();
        let gpu_params = GpuParams::new(coeffs, points.len());
        let out_bytes = (points.len() as u64) * (size_of::<GpuPoint>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lens_distortion_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lens_distortion_points"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lens_distortion_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lens_distortion_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_lens_distortion_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_lens_distortion_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_lens_distortion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per point, flattened to a one-`D` dispatch.
            let groups = (points.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuPoint>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), points.len());
        raw.into_iter().map(|p| [p.x, p.y]).collect()
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
