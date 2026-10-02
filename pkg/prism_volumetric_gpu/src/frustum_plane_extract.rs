//! `wgpu` compute twin of the Gribb-Hartmann view-frustum plane extraction
//! golden
//! ([`frustum_plane_extract`](prism_render_architecture::particle::frustum_plane_extract),
//! particle design §12, §13).
//!
//! The `CPU` golden
//! [`frustum_plane_extract`](prism_render_architecture::particle::frustum_plane_extract)
//! owns the analytic extraction of the six oriented planes that bound a view
//! volume from a combined view-projection
//! [`Mat4`](prism_render_architecture::particle::frustum_plane_extract::Mat4).
//! Each clip-space inequality `-w <= x,y,z <= w` of the canonical `OpenGL`-style
//! cube is a linear combination of the matrix rows, so a plane is just a row sum
//! or difference — no eigenvalues, no iteration, no trigonometry — followed by a
//! normalization that divides the coefficients by `|n|` (one
//! [`f32::sqrt`](f32::sqrt)) so
//! [`Plane::signed_distance`](prism_render_architecture::particle::frustum_plane_extract::Plane::signed_distance)
//! becomes a true Euclidean distance.
//!
//! [`GpuFrustumPlaneExtract`] is the on-device twin: one thread per matrix
//! reproduces the same six-plane
//! [`extract_frustum_planes`](prism_render_architecture::particle::frustum_plane_extract::extract_frustum_planes)
//! combination table and the same normalization branch for branch, so a passing
//! real-device parity test is direct evidence the ported kernel extracts the
//! same planes and classifies the same degenerate normal the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every value the reference reports is reproduced per matrix: the full
//! six-plane
//! [`FrustumPlanes`](prism_render_architecture::particle::frustum_plane_extract::FrustumPlanes)
//! block in the fixed order left, right, bottom, top, near, far, each plane's
//! `nx, ny, nz, d` carried through the Gribb-Hartmann row combination and the
//! unit-normal normalization. The reference's single degenerate branch is
//! mirrored: a normal whose squared length is below
//! [`NORMALIZE_EPS`](prism_render_architecture::particle::frustum_plane_extract::NORMALIZE_EPS)
//! is left unscaled rather than dividing by ~zero.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `vec4` add and
//! subtract, scalar multiply and one `sqrt` for the genuine normal length — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each matrix is a fixed, non-reorderable sequence of adds, multiplies, a
//! divide and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the
//! same order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every `f32` plane coefficient,
//! tight enough to catch a genuinely wrong port (a swapped row, a wrong sign, a
//! dropped normalization) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
//! classic Gribb-Hartmann plane extraction plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::frustum_plane_extract::{
    extract_frustum_planes, FrustumPlanes, Mat4, Plane, PLANE_COUNT,
};
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

/// The portable core-`WGSL` frustum-plane-extraction kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `extract`
/// mirrors the `CPU` golden
/// [`extract_frustum_planes`](prism_render_architecture::particle::frustum_plane_extract::extract_frustum_planes)
/// row combination for row combination; see the module documentation for the
/// algorithm.
const FRUSTUM_PLANE_EXTRACT_WGSL: &str = r#"
// Gribb-Hartmann view-frustum plane extraction twin: one thread per matrix
// reproduces the six oriented, normalized planes bounding a view volume. It
// mirrors the CPU golden `particle::frustum_plane_extract` row combination for
// row combination, uses only the portable core-WGSL subset (vec4 add/subtract,
// scalar multiply and one sqrt for the normal length) and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::frustum_plane_extract; no third-party engine source or derived
// code.

// Squared-length floor below which a normal is treated as degenerate and the
// plane is returned unscaled rather than divided by ~zero. Matches the
// reference `NORMALIZE_EPS`.
const NORMALIZE_EPS: f32 = 1.0e-12;

struct Params {
    // Number of matrices in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Mat4 {
    // The four row-major rows of the view-projection matrix; row `i` holds the
    // coefficients used to dot a column point `(x, y, z, 1)`.
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    row3: vec4<f32>,
}

struct Planes {
    // The six planes in the fixed order left, right, bottom, top, near, far;
    // each lane carries `(nx, ny, nz, d)` for `n*p + d = 0`.
    left: vec4<f32>,
    right: vec4<f32>,
    bottom: vec4<f32>,
    top: vec4<f32>,
    plane_near: vec4<f32>,
    plane_far: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrices: array<Mat4>;
@group(0) @binding(2) var<storage, read_write> planes_out: array<Planes>;

// Returns an equivalent plane with a unit-length normal, mirroring the
// reference `Plane::normalized`: dividing `n` and `d` by `|n|` preserves the
// zero set and the sign of the signed distance. A degenerate normal (squared
// length below NORMALIZE_EPS) is returned unchanged rather than dividing by
// ~zero.
fn normalize_plane(pl: vec4<f32>) -> vec4<f32> {
    let len_sq = pl.x * pl.x + pl.y * pl.y + pl.z * pl.z;
    if (len_sq < NORMALIZE_EPS) {
        return pl;
    }
    let inv = 1.0 / sqrt(len_sq);
    return pl * inv;
}

@compute @workgroup_size(64)
fn extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let m = matrices[idx];
    let r0 = m.row0;
    let r1 = m.row1;
    let r2 = m.row2;
    let r3 = m.row3;

    // The clip-volume inequalities `-w <= x,y,z <= w` rearranged into
    // `n*p + d >= 0` are row sums and differences, each then normalized.
    var out: Planes;
    out.left = normalize_plane(r3 + r0);
    out.right = normalize_plane(r3 - r0);
    out.bottom = normalize_plane(r3 + r1);
    out.top = normalize_plane(r3 - r1);
    out.plane_near = normalize_plane(r3 + r2);
    out.plane_far = normalize_plane(r3 - r2);
    planes_out[idx] = out;
}
"#;

/// Evaluates the `CPU` golden for one matrix, delegating to the reference
/// [`extract_frustum_planes`](prism_render_architecture::particle::frustum_plane_extract::extract_frustum_planes)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(mat: &Mat4) -> FrustumPlanes {
    extract_frustum_planes(mat)
}

/// `repr(C)` `std430` layout of one packed matrix: four `vec4` row slots in
/// row-major order — `64` bytes, each row on its `16`-byte-aligned slot exactly
/// as the `WGSL` `Mat4` struct reads it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
/// no third-party engine source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMat4 {
    /// Row `0` of the matrix.
    row0: [f32; 4],
    /// Row `1` of the matrix.
    row1: [f32; 4],
    /// Row `2` of the matrix.
    row2: [f32; 4],
    /// Row `3` of the matrix.
    row3: [f32; 4],
}

impl GpuMat4 {
    /// Packs one [`Mat4`](prism_render_architecture::particle::frustum_plane_extract::Mat4)
    /// into its `std430` image, splitting the sixteen row-major coefficients
    /// into four `vec4` rows.
    fn new(mat: &Mat4) -> GpuMat4 {
        let r0 = mat.row(0);
        let r1 = mat.row(1);
        let r2 = mat.row(2);
        let r3 = mat.row(3);
        GpuMat4 {
            row0: r0,
            row1: r1,
            row2: r2,
            row3: r3,
        }
    }
}

/// `repr(C)` `std430` layout of one result: six `vec4` plane slots in the fixed
/// order left, right, bottom, top, near, far — `96` bytes matching the `WGSL`
/// `Planes` struct, each lane carrying `(nx, ny, nz, d)`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
/// no third-party engine source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPlanes {
    /// Left clipping plane.
    left: [f32; 4],
    /// Right clipping plane.
    right: [f32; 4],
    /// Bottom clipping plane.
    bottom: [f32; 4],
    /// Top clipping plane.
    top: [f32; 4],
    /// Near clipping plane.
    plane_near: [f32; 4],
    /// Far clipping plane.
    plane_far: [f32; 4],
}

/// Uniform parameters for one dispatch: the matrix count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
/// no third-party engine source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of matrices in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable frustum-plane-extraction compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
/// no third-party engine source or derived code.
pub struct GpuFrustumPlaneExtract {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFrustumPlaneExtract {
    /// Compiles the frustum-plane-extraction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFrustumPlaneExtract {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract"),
            source: ShaderSource::Wgsl(FRUSTUM_PLANE_EXTRACT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("extract"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFrustumPlaneExtract {
            module,
            layout,
            pipeline,
        }
    }

    /// Extracts the six frustum planes from every matrix on-device and returns
    /// one [`FrustumPlanes`](prism_render_architecture::particle::frustum_plane_extract::FrustumPlanes)
    /// per input, in order.
    ///
    /// Each result equals the reference
    /// [`extract_frustum_planes`](prism_render_architecture::particle::frustum_plane_extract::extract_frustum_planes)
    /// to within the tolerance documented on this module. An empty input returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_plane_extract`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, matrices: &[Mat4]) -> Vec<FrustumPlanes> {
        if matrices.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = matrices.len();

        let packed: Vec<GpuMat4> = matrices.iter().map(GpuMat4::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuPlanes>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_frustum_plane_extract_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_frustum_plane_extract_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per matrix, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuPlanes>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuPlanes`] into the public
/// [`FrustumPlanes`](prism_render_architecture::particle::frustum_plane_extract::FrustumPlanes),
/// rebuilding the six planes in the fixed left, right, bottom, top, near, far
/// order.
fn decode_result(raw: &GpuPlanes) -> FrustumPlanes {
    let mut planes = [Plane::new(0.0, 0.0, 0.0, 0.0); PLANE_COUNT];
    planes[0] = Plane::new(raw.left[0], raw.left[1], raw.left[2], raw.left[3]);
    planes[1] = Plane::new(raw.right[0], raw.right[1], raw.right[2], raw.right[3]);
    planes[2] = Plane::new(raw.bottom[0], raw.bottom[1], raw.bottom[2], raw.bottom[3]);
    planes[3] = Plane::new(raw.top[0], raw.top[1], raw.top[2], raw.top[3]);
    planes[4] = Plane::new(
        raw.plane_near[0],
        raw.plane_near[1],
        raw.plane_near[2],
        raw.plane_near[3],
    );
    planes[5] = Plane::new(
        raw.plane_far[0],
        raw.plane_far[1],
        raw.plane_far[2],
        raw.plane_far[3],
    );
    FrustumPlanes::new(planes)
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
