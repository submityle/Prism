//! `wgpu` compute twin of the trigonometry-free octahedral direction/square
//! map and its solid-angle Jacobian, mirroring this repository's
//! `prism_render_architecture::reference_pt::octahedral` module.
//!
//! The golden module folds the unit sphere onto the `L1` octahedron
//! `|x| + |y| + |z| = 1` and unfolds that onto the square `[-1, 1]^2` using
//! only additions, multiplications, absolute values and sign copies (the
//! workspace determinism policy forbids transcendentals other than `sqrt`).
//! This twin reproduces, for one query per thread, all three closed forms:
//!
//! - [`direction_to_square`](prism_render_architecture::reference_pt::octahedral::direction_to_square):
//!   the `L1` projection of a (not necessarily unit) direction, with the lower
//!   hemisphere reflected across the diagonals.
//! - [`square_to_direction`](prism_render_architecture::reference_pt::octahedral::square_to_direction):
//!   the inverse map, reconstructing and normalizing the direction.
//! - [`solid_angle_jacobian`](prism_render_architecture::reference_pt::octahedral::solid_angle_jacobian):
//!   the measure correction `d_omega / dA = |d|_1^3`.
//!
//! # Sign handling
//!
//! The golden `sign_unit` is `copysign(1.0, x)`: a non-negative component (and
//! `+0`) maps to `+1`, a negative component (and `-0`) to `-1`. The kernel
//! reproduces this with the sign bit, not a bare `x >= 0.0` compare, so a
//! negative zero on a fold seam takes the same branch as the host and the maps
//! stay exact inverses there.
//!
//! # What is twinned
//!
//! [`GpuOctahedralMap`] packs both halves into one query: the direction drives
//! the forward map and the Jacobian, the independent `(u, v)` square point
//! drives the inverse map. Each result carries the forward square coordinate,
//! the reconstructed direction, the Jacobian and a `valid` flag (always `1`,
//! since there is no degenerate rejection; a zero direction simply maps to the
//! square origin and a degenerate inverse normalizes to the zero vector).
//!
//! # Precision model
//!
//! The golden path evaluates in `f32`; this twin and its host oracle both
//! evaluate the same closed form in `f32`, each query a fixed, non-reorderable
//! sequence of multiplies, adds, absolute values and one `sqrt`, so `CPU` and
//! `GPU` compute the same arithmetic in the same order. They are not bit-exact:
//! a `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! parity test therefore asserts `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous output and an exact `==` on
//! `valid`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sqrt`, `select`, `bitcast` and `+ - * /` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, `round` or optional device feature, and no `u64`,
//! `i64` or `f64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! only non-rational operation is the inverse-map normalization `sqrt`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::octahedral`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly
/// default shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `octmap` mirrors the forward and
/// inverse octahedral maps and the solid-angle Jacobian of the `CPU` golden
/// module; see the module documentation for the algorithm.
const OCTAHEDRAL_MAP_WGSL: &str = r#"
// Octahedral direction/square map twin: one thread per query evaluates the
// forward L1 projection direction_to_square, the inverse square_to_direction
// (with normalization) and the solid-angle Jacobian |d|_1^3. It uses only the
// portable core-WGSL subset (abs/min/max/sqrt/select/bitcast and + - * /) with
// no u64/i64/f64 and no transcendental, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::octahedral；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: a (not necessarily unit) direction lifted to a vec4 slot, plus the
// independent square point (u, v) that drives the inverse map.
struct Query {
    dir: vec4<f32>,
    u: f32,
    v: f32,
    pad0: f32,
    pad1: f32,
}

// One result: the forward square coordinate, the reconstructed direction, the
// Jacobian and the valid flag (plus a pad word to round to 32 bytes).
struct Res {
    sq_u: f32,
    sq_v: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    jac: f32,
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// copysign(1.0, x): +1 for a non-negative component (incl +0), -1 for a
// negative component (incl -0). Reads the sign bit directly so -0 is treated as
// negative, exactly like the golden sign_unit; a bare x >= 0.0 would misclassify
// -0 as positive and break the inverse on the fold seams.
fn sign_unit(x: f32) -> f32 {
    let neg = (bitcast<u32>(x) & 0x80000000u) != 0u;
    return select(1.0, -1.0, neg);
}

@compute @workgroup_size(64)
fn octmap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let dx = q.dir.x;
    let dy = q.dir.y;
    let dz = q.dir.z;

    // Forward map: L1 projection, lower hemisphere reflected across diagonals.
    let l1 = abs(dx) + abs(dy) + abs(dz);
    let inv = select(0.0, 1.0 / l1, l1 > 0.0);
    let px = dx * inv;
    let py = dy * inv;
    var sq_u: f32;
    var sq_v: f32;
    if (dz >= 0.0) {
        sq_u = px;
        sq_v = py;
    } else {
        sq_u = (1.0 - abs(py)) * sign_unit(px);
        sq_v = (1.0 - abs(px)) * sign_unit(py);
    }

    // Inverse map: fold height z, reflected on the lower-hemisphere wedge, then
    // normalize (zero vector on a degenerate point).
    let u = q.u;
    let v = q.v;
    let z = 1.0 - abs(u) - abs(v);
    var x: f32;
    var y: f32;
    if (z >= 0.0) {
        x = u;
        y = v;
    } else {
        x = (1.0 - abs(v)) * sign_unit(u);
        y = (1.0 - abs(u)) * sign_unit(v);
    }
    let len2 = x * x + y * y + z * z;
    var rx: f32 = 0.0;
    var ry: f32 = 0.0;
    var rz: f32 = 0.0;
    if (len2 > 1e-12) {
        let inv_len = 1.0 / sqrt(len2);
        rx = x * inv_len;
        ry = y * inv_len;
        rz = z * inv_len;
    }

    // Solid-angle Jacobian d_omega / dA = |d|_1^3.
    let jl1 = abs(dx) + abs(dy) + abs(dz);
    let jac = jl1 * jl1 * jl1;

    var out: Res;
    out.sq_u = sq_u;
    out.sq_v = sq_v;
    out.dir_x = rx;
    out.dir_y = ry;
    out.dir_z = rz;
    out.jac = jac;
    out.valid = 1u;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// One octahedral-map query: a direction for the forward map and Jacobian plus
/// an independent square point `(u, v)` for the inverse map.
///
/// `dir` need not be a unit vector; the forward map normalizes it through the
/// `L1` projection and the Jacobian is the `L1` norm cubed. Derives only
/// [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctahedralMapQuery {
    /// Direction driving the forward square map and the Jacobian.
    pub dir: [f32; 3],
    /// Square `u` coordinate driving the inverse map.
    pub u: f32,
    /// Square `v` coordinate driving the inverse map.
    pub v: f32,
}

impl OctahedralMapQuery {
    /// Builds a query from a direction and an inverse-map square point.
    #[must_use]
    pub const fn new(dir: [f32; 3], u: f32, v: f32) -> OctahedralMapQuery {
        OctahedralMapQuery { dir, u, v }
    }
}

/// The octahedral-map outputs for one query, the host-side mirror of the
/// kernel's `Res` lane.
///
/// `square` is the forward `direction_to_square` coordinate, `direction` the
/// reconstructed `square_to_direction` unit vector, `jacobian` the solid-angle
/// correction `|d|_1^3` and `valid` the always-`1` flag. Derives only
/// [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctahedralMapResult {
    /// Forward square coordinate `(sq_u, sq_v)`.
    pub square: [f32; 2],
    /// Reconstructed unit direction `(x, y, z)` from the inverse map.
    pub direction: [f32; 3],
    /// Solid-angle-to-area Jacobian `|d|_1^3`.
    pub jacobian: f32,
    /// Always `1`: there is no degenerate rejection.
    pub valid: u32,
}

/// `repr(C)` `std430` layout of one packed query: the direction `vec3` lifted
/// to a `vec4` slot, then `u`, `v` and two pad lanes — `32` bytes, exactly as
/// the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Direction, `xyz` plus a pad lane.
    dir: [f32; 4],
    /// Inverse-map square `u`.
    u: f32,
    /// Inverse-map square `v`.
    v: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &OctahedralMapQuery) -> GpuQuery {
        GpuQuery {
            dir: [query.dir[0], query.dir[1], query.dir[2], 0.0],
            u: query.u,
            v: query.v,
            pad0: 0.0,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the six scalar `f32` outputs, the
/// `valid` flag and one pad word in the same order as the `WGSL` `Res` struct —
/// `32` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Forward square `u`.
    sq_u: f32,
    /// Forward square `v`.
    sq_v: f32,
    /// Reconstructed direction `x`.
    dir_x: f32,
    /// Reconstructed direction `y`.
    dir_y: f32,
    /// Reconstructed direction `z`.
    dir_z: f32,
    /// Solid-angle Jacobian.
    jac: f32,
    /// Always-`1` valid flag.
    valid: u32,
    /// Padding word.
    pad0: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// round the uniform block out to `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Decodes one packed `GpuResult` into the public [`OctahedralMapResult`].
fn decode_result(raw: &GpuResult) -> OctahedralMapResult {
    OctahedralMapResult {
        square: [raw.sq_u, raw.sq_v],
        direction: [raw.dir_x, raw.dir_y, raw.dir_z],
        jacobian: raw.jac,
        valid: raw.valid,
    }
}

/// Builds one storage/uniform buffer bind-group-layout entry.
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

/// On-device twin of the octahedral direction/square map and its Jacobian.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuOctahedralMap::new`] and reuse it across
/// [`GpuOctahedralMap::evaluate`] calls.
pub struct GpuOctahedralMap {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `octmap` entry point.
    pipeline: ComputePipeline,
}

impl GpuOctahedralMap {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOctahedralMap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_octahedral_map_shader"),
            source: ShaderSource::Wgsl(OCTAHEDRAL_MAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_octahedral_map_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_octahedral_map_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_octahedral_map_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("octmap"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOctahedralMap {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every octahedral-map query on-device and returns one
    /// [`OctahedralMapResult`] per input, in order.
    ///
    /// Each result equals the reference closed form to within the tolerance
    /// documented on this module. An empty input returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[OctahedralMapQuery],
    ) -> Vec<OctahedralMapResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_octahedral_map_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_octahedral_map_output"),
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
            label: Some("prism_volumetric_octahedral_map_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_octahedral_map_bind_group"),
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
            label: Some("prism_volumetric_octahedral_map_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_octahedral_map_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_octahedral_map_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
