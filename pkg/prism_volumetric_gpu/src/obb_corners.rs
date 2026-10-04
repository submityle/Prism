//! `wgpu` compute twin of the oriented-bounding-box corner helper from the
//! `CPU` golden `prism_physics_core::collider::obb::Obb::corners`.
//!
//! An oriented bounding box is a center, three orthonormal axes, and the box
//! half-extents along those axes. The golden expands that compact description
//! into its eight world-space corners by scaling each axis by its half-extent
//! and adding the three signed half-axis vectors to the center in a fixed sign
//! pattern. This module ports that stateless, no-`RNG`, branch-free closed form
//! onto the device: one compute thread resolves one box, so a passing
//! real-device parity test is direct evidence the kernel reproduces the exact
//! corner ordering and sign pattern, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one box: `center`, `axis0`, `axis1`, `axis2`, `half_extents`.
//! For each box the kernel reproduces the reference closed form exactly:
//!
//! * `hx = axis0 * he.x`, `hy = axis1 * he.y`, `hz = axis2 * he.z`;
//! * the eight corners in the golden's order, `c0 = center - hx - hy - hz` …
//!   `c7 = center + hx + hy + hz`, cycling the `hx` sign fastest and the `hz`
//!   sign slowest.
//!
//! There is no division and no branch: the computation is pure multiply-add, so
//! the kernel provably terminates and `valid` is always `1`.
//!
//! # Correctness model
//!
//! Each corner is a short multiply-add chain, so `CPU` and `GPU` are not
//! required to be bit-exact. The parity test asserts a tolerance (`abs_diff <=
//! 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each of the twenty-four
//! corner components; the discrete `valid` flag is compared exactly. The kernel
//! has no comparisons at all, so no bare float equality and no fast-math `NaN`
//! sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every center, axis triple and half-extent
//! yields eight well-defined corners, so `valid` is always `1`. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - *` and unsigned
//! index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no
//! `round`, no float modulo and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb::Obb::corners`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` OBB-corners kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `project` mirrors the
/// `CPU` golden `corners`; see the module documentation for the algorithm.
const OBB_CORNERS_WGSL: &str = r#"
// OBB-corners twin: one thread per box reproduces the eight world-space corners
// the golden Obb::corners builds from a center, three orthonormal axes and the
// half-extents. It mirrors the CPU golden operation for operation, uses only
// the portable core-WGSL subset (+ - * plus unsigned index math), takes no
// optional feature, and has no loop, so the kernel provably terminates. There
// is no division and no branch, so no float equality or fast-math sentinel is
// involved.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::obb::Obb::corners；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of boxes in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Box center.
    cx: f32,
    cy: f32,
    cz: f32,
    // Orthonormal axis 0.
    a0x: f32,
    a0y: f32,
    a0z: f32,
    // Orthonormal axis 1.
    a1x: f32,
    a1y: f32,
    a1z: f32,
    // Orthonormal axis 2.
    a2x: f32,
    a2y: f32,
    a2z: f32,
    // Half-extents along axis 0, 1, 2.
    hex: f32,
    hey: f32,
    hez: f32,
}

struct Result {
    // Eight world-space corners, each stored as three flat scalars, in the
    // golden's order (hx sign fastest, hz sign slowest).
    c0x: f32, c0y: f32, c0z: f32,
    c1x: f32, c1y: f32, c1z: f32,
    c2x: f32, c2y: f32, c2z: f32,
    c3x: f32, c3y: f32, c3z: f32,
    c4x: f32, c4y: f32, c4z: f32,
    c5x: f32, c5y: f32, c5z: f32,
    c6x: f32, c6y: f32, c6z: f32,
    c7x: f32, c7y: f32, c7z: f32,
    // Always 1: the closed form has no degenerate branch.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let center = vec3<f32>(q.cx, q.cy, q.cz);
    let axis0 = vec3<f32>(q.a0x, q.a0y, q.a0z);
    let axis1 = vec3<f32>(q.a1x, q.a1y, q.a1z);
    let axis2 = vec3<f32>(q.a2x, q.a2y, q.a2z);

    let hx = axis0 * q.hex;
    let hy = axis1 * q.hey;
    let hz = axis2 * q.hez;

    let c0 = center - hx - hy - hz;
    let c1 = center + hx - hy - hz;
    let c2 = center - hx + hy - hz;
    let c3 = center + hx + hy - hz;
    let c4 = center - hx - hy + hz;
    let c5 = center + hx - hy + hz;
    let c6 = center - hx + hy + hz;
    let c7 = center + hx + hy + hz;

    var out: Result;
    out.c0x = c0.x; out.c0y = c0.y; out.c0z = c0.z;
    out.c1x = c1.x; out.c1y = c1.y; out.c1z = c1.z;
    out.c2x = c2.x; out.c2y = c2.y; out.c2z = c2.z;
    out.c3x = c3.x; out.c3y = c3.y; out.c3z = c3.z;
    out.c4x = c4.x; out.c4y = c4.y; out.c4z = c4.z;
    out.c5x = c5.x; out.c5y = c5.y; out.c5z = c5.z;
    out.c6x = c6.x; out.c6y = c6.y; out.c6z = c6.z;
    out.c7x = c7.x; out.c7y = c7.y; out.c7z = c7.z;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`OBB_CORNERS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// All components are scalar `f32` so the slot contains no `vec3` and the host
/// and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cx: f32,
    cy: f32,
    cz: f32,
    a0x: f32,
    a0y: f32,
    a0z: f32,
    a1x: f32,
    a1y: f32,
    a1z: f32,
    a2x: f32,
    a2y: f32,
    a2z: f32,
    hex: f32,
    hey: f32,
    hez: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The eight corners are stored as flat scalars in the golden's order;
/// the trailing `valid` word keeps the discrete flag beside them.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    corners: [f32; 24],
    valid: u32,
}

/// One query for the OBB-corners twin: a center, three orthonormal axes and the
/// box half-extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbCornersQuery {
    /// Box center.
    pub center: [f32; 3],
    /// Orthonormal axis `0`, paired with `half_extents[0]`.
    pub axis0: [f32; 3],
    /// Orthonormal axis `1`, paired with `half_extents[1]`.
    pub axis1: [f32; 3],
    /// Orthonormal axis `2`, paired with `half_extents[2]`.
    pub axis2: [f32; 3],
    /// Half the box size along each corresponding axis.
    pub half_extents: [f32; 3],
}

impl ObbCornersQuery {
    /// Builds a query from the center, the three orthonormal axes and the
    /// half-extents.
    #[must_use]
    pub fn new(
        center: [f32; 3],
        axis0: [f32; 3],
        axis1: [f32; 3],
        axis2: [f32; 3],
        half_extents: [f32; 3],
    ) -> ObbCornersQuery {
        ObbCornersQuery {
            center,
            axis0,
            axis1,
            axis2,
            half_extents,
        }
    }
}

/// One resolved answer for a single box: the eight world-space corners in the
/// golden's order, plus the `valid` flag (always `1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbCornersResult {
    /// The eight corners, each `[x, y, z]`, in the golden's order: `hx` sign
    /// cycles fastest, `hz` sign slowest.
    pub corners: [[f32; 3]; 8],
    /// Always `1`: the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`ObbCornersQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ObbCornersQuery) -> GpuQuery {
    GpuQuery {
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        a0x: q.axis0[0],
        a0y: q.axis0[1],
        a0z: q.axis0[2],
        a1x: q.axis1[0],
        a1y: q.axis1[1],
        a1z: q.axis1[2],
        a2x: q.axis2[0],
        a2y: q.axis2[1],
        a2z: q.axis2[2],
        hex: q.half_extents[0],
        hey: q.half_extents[1],
        hez: q.half_extents[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ObbCornersResult`].
fn decode_result(raw: &GpuResult) -> ObbCornersResult {
    let mut corners = [[0.0_f32; 3]; 8];
    for (i, corner) in corners.iter_mut().enumerate() {
        *corner = [
            raw.corners[i * 3],
            raw.corners[i * 3 + 1],
            raw.corners[i * 3 + 2],
        ];
    }
    ObbCornersResult {
        corners,
        valid: raw.valid,
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

/// A compiled, reusable OBB-corners compute pipeline, twinning the `CPU` golden
/// `Obb::corners`.
pub struct GpuObbCorners {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuObbCorners {
    /// Compiles the OBB-corners kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbCorners {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_obb_corners"),
            source: ShaderSource::Wgsl(OBB_CORNERS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_obb_corners_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_obb_corners_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_obb_corners_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbCorners {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every box in `queries` and returns one [`ObbCornersResult`] per
    /// input, in order.
    ///
    /// Each corner component matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[ObbCornersQuery]) -> Vec<ObbCornersResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_corners_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_obb_corners_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_corners_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_obb_corners_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_obb_corners_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_obb_corners_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_obb_corners_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per box, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
