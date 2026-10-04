//! `wgpu` compute twin of the rigid-body generalized inverse mass from the
//! `CPU` golden `prism_physics_core::solver::xpbd::rigid::generalized_inverse_mass`.
//!
//! An `XPBD` rigid-body constraint needs the effective inverse mass a body
//! presents along a unit direction applied at a world lever arm. This is the
//! stateless scalar kernel that computes it:
//!
//! ```text
//! rn   = cross(r, direction)
//! w    = inv_mass + dot(rn, I_inv_world * rn)
//! ```
//!
//! where `I_inv_world` is the world-space inverse inertia tensor, laid out
//! column-major (`col0, col1, col2`) to match the `rigid_inv_inertia_world`
//! output, so the matrix-vector product is
//! `I_inv_world * rn = col0 * rn.x + col1 * rn.y + col2 * rn.z`.
//!
//! # What is twinned
//!
//! The single stateless, no-`RNG` entry point `generalized_inverse_mass`. It is
//! pure arithmetic with no degenerate branch, so the `valid` flag is always
//! `1`; it is carried only to keep the result layout uniform with the other
//! twins in this crate.
//!
//! # Correctness model
//!
//! Each quantity threads through multiplies, adds and one cross/dot pair, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the
//! continuous `w`; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! There is none: with a zero lever arm (`r = 0`) or a lever arm parallel to
//! the direction the cross product `rn` is zero and `w` collapses to
//! `inv_mass`, which the kernel produces by plain arithmetic with no special
//! case. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, a hand
//! written cross and dot — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! `round`, no `f32` remainder, no `u64`/`i64`, no `f64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid::generalized_inverse_mass`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` generalized-inverse-mass kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `evaluate`
/// mirrors the `CPU` golden `generalized_inverse_mass`; see the module
/// documentation for the algorithm.
const RIGID_GENERALIZED_INVERSE_MASS_WGSL: &str = r#"
// Rigid-body generalized inverse mass twin: one thread per query computes the
// effective inverse mass a body presents along a unit direction applied at a
// world lever arm, w = inv_mass + dot(rn, I_inv_world * rn) with
// rn = cross(r, direction). The inverse inertia tensor is column-major
// (col0, col1, col2), so I_inv_world * rn = col0*rn.x + col1*rn.y + col2*rn.z.
// It uses only the portable core-WGSL subset (+ - * / with a hand-written cross
// and dot) with no u64/i64/f64 and no transcendental, so it runs unmodified on
// Metal, Vulkan and DX12. The formula has no degenerate branch; valid is always
// 1 and is carried only to keep the result layout uniform.
//
// Provenance: 孪生自本仓 prism_physics_core::solver::xpbd::rigid::generalized_inverse_mass；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the scalar inverse mass, the world inverse inertia tensor stored
// column-major as nine scalars, the world lever arm r and the unit direction.
// Every vector and matrix is flattened to scalar lanes so the std430 layout
// never trips a 16-byte vector-alignment rule; the kernel rebuilds each vec3.
struct Query {
    inv_mass: f32,
    i00: f32,
    i01: f32,
    i02: f32,
    i10: f32,
    i11: f32,
    i12: f32,
    i20: f32,
    i21: f32,
    i22: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

// One result: the generalized inverse mass and a valid flag.
struct Res {
    w: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let lever = vec3<f32>(q.rx, q.ry, q.rz);
    let dir = vec3<f32>(q.dx, q.dy, q.dz);

    // rn = cross(lever, dir).
    let rn = vec3<f32>(
        lever.y * dir.z - lever.z * dir.y,
        lever.z * dir.x - lever.x * dir.z,
        lever.x * dir.y - lever.y * dir.x,
    );

    // Column-major inverse inertia tensor; columns are (i00,i01,i02) etc.
    let col0 = vec3<f32>(q.i00, q.i01, q.i02);
    let col1 = vec3<f32>(q.i10, q.i11, q.i12);
    let col2 = vec3<f32>(q.i20, q.i21, q.i22);

    // I_inv_world * rn = col0*rn.x + col1*rn.y + col2*rn.z.
    let i_rn = col0 * rn.x + col1 * rn.y + col2 * rn.z;

    var out: Res;
    out.w = q.inv_mass + dot(rn, i_rn);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RIGID_GENERALIZED_INVERSE_MASS_WGSL`].
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
/// The vector and matrix inputs are flattened to scalar lanes so the layout
/// never trips a `16`-byte vector-alignment rule; the kernel rebuilds each
/// `vec3<f32>`. The inertia tensor is stored column-major (`col0, col1, col2`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    inv_mass: f32,
    i00: f32,
    i01: f32,
    i02: f32,
    i10: f32,
    i11: f32,
    i12: f32,
    i20: f32,
    i21: f32,
    i22: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    w: f32,
    valid: u32,
}

/// One query for the generalized-inverse-mass twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidGeneralizedInverseMassQuery {
    /// Scalar inverse mass of the body.
    pub inv_mass: f32,
    /// World-space inverse inertia tensor, stored column-major as
    /// `[col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z]`
    /// to match the `rigid_inv_inertia_world` output.
    pub inv_inertia_world: [f32; 9],
    /// World lever arm (offset from the center of mass).
    pub r: [f32; 3],
    /// Unit direction the constraint acts along.
    pub direction: [f32; 3],
}

impl RigidGeneralizedInverseMassQuery {
    /// Builds a query from the inverse mass, column-major inverse inertia
    /// tensor, world lever arm and direction.
    #[must_use]
    pub fn new(
        inv_mass: f32,
        inv_inertia_world: [f32; 9],
        r: [f32; 3],
        direction: [f32; 3],
    ) -> RigidGeneralizedInverseMassQuery {
        RigidGeneralizedInverseMassQuery {
            inv_mass,
            inv_inertia_world,
            r,
            direction,
        }
    }
}

/// One resolved answer for a single query, mirroring the scalar returned by the
/// reference `generalized_inverse_mass`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidGeneralizedInverseMassResult {
    /// The generalized inverse mass `w = inv_mass + dot(rn, I_inv_world * rn)`.
    pub w: f32,
    /// Always `1`: the formula is pure arithmetic with no degenerate branch.
    /// The flag is carried only to keep the result layout uniform with the
    /// other twins in this crate.
    pub valid: u32,
}

/// Encodes one [`RigidGeneralizedInverseMassQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &RigidGeneralizedInverseMassQuery) -> GpuQuery {
    let i = q.inv_inertia_world;
    GpuQuery {
        inv_mass: q.inv_mass,
        i00: i[0],
        i01: i[1],
        i02: i[2],
        i10: i[3],
        i11: i[4],
        i12: i[5],
        i20: i[6],
        i21: i[7],
        i22: i[8],
        rx: q.r[0],
        ry: q.r[1],
        rz: q.r[2],
        dx: q.direction[0],
        dy: q.direction[1],
        dz: q.direction[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RigidGeneralizedInverseMassResult`].
fn decode_result(raw: &GpuResult) -> RigidGeneralizedInverseMassResult {
    RigidGeneralizedInverseMassResult {
        w: raw.w,
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

/// A compiled, reusable generalized-inverse-mass compute pipeline, twinning the
/// `CPU` golden `generalized_inverse_mass`.
pub struct GpuRigidGeneralizedInverseMass {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRigidGeneralizedInverseMass {
    /// Compiles the generalized-inverse-mass kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidGeneralizedInverseMass {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass"),
            source: ShaderSource::Wgsl(RIGID_GENERALIZED_INVERSE_MASS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRigidGeneralizedInverseMass {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`RigidGeneralizedInverseMassResult`] per input, in order.
    ///
    /// The continuous `w` matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RigidGeneralizedInverseMassQuery],
    ) -> Vec<RigidGeneralizedInverseMassResult> {
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
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_bind_group"),
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
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rigid_generalized_inverse_mass_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rigid_generalized_inverse_mass_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
