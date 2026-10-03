//! `wgpu` compute twin of the isometric-bending Gauss-Seidel solve from the
//! cloth bending module
//! ([`bending`](prism_render_architecture::cloth::bending)).
//!
//! Bending controls how sharply a garment folds along an interior edge. The
//! golden
//! [`apply_bending`](prism_render_architecture::cloth::bending::apply_bending)
//! runs `iterations` Gauss-Seidel sweeps over a hinge set, projecting each
//! four-vertex isometric-bending constraint in place against the predicted
//! positions. The projection math lives once in
//! [`project_isometric_bending`](prism_physics_core::soft::constraint::project_isometric_bending):
//! it forms the bend vector `S = Σ wᵢ · xᵢ`, the energy `E = ½ · scale · |S|²`,
//! an `XPBD` denominator `Σ wmassᵢ · |gradᵢ|² + compliance / dt²`, and the
//! mass-weighted correction `Δxᵢ = inv_massᵢ · Δλ · scale · wᵢ · S`.
//!
//! The sweep is *order sensitive*: each constraint reads the positions updated
//! by the constraints before it, so a parallel-per-constraint kernel would
//! diverge. [`GpuClothBendingApply`] therefore maps one whole cloth system onto
//! one thread (`@workgroup_size(1)`): the thread copies its system's positions
//! into private storage, then replays the exact iteration-outer /
//! constraint-inner loop in place before writing the solved positions and the
//! first-sweep energy back. A passing real-device parity test is direct
//! evidence the ported kernel reproduces the reference solve, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! For one query carrying the system's particle positions, their inverse
//! masses, the hinge stencils (`vertices`/`weights`/`scale`/`compliance`), the
//! sweep count `iterations`, and the substep `dt`, the kernel reproduces
//! `apply_bending`:
//! - `iterations` is clamped to at least one;
//! - for each sweep, each constraint in slice order projects in place: a
//!   per-constraint `dt <= 0` no-op, the bend vector `S`, a `|S|² <= 1e-12`
//!   degeneracy no-op, the energy, the guarded `denom <= 0` no-op, and the
//!   per-free-vertex position correction;
//! - only the first sweep's energies are summed into the returned
//!   `first_energy`.
//!
//! # Semantics
//!
//! The bend vector `S` sums every in-bounds slot (even pinned ones, since it
//! reads positions, not inverse mass), while the denominator and the position
//! correction touch only free slots (`inverse_mass > 0`). Vertex indices past
//! the particle count contribute nothing, mirroring the golden's
//! `positions.get` / `unwrap_or(0.0)`.
//!
//! # Correctness model
//!
//! Positions and energy are continuous `f32` quantities, so parity is asserted
//! with the crate tolerance (`abs <= 2e-4` or `rel <= 2e-3`, floor `1e-6`),
//! relaxed modestly from the base `1e-4`/`1e-3` because multiple in-place
//! sweeps accumulate rounding. The branch cuts (`dt <= 0`, `|S|² <= 1e-12`,
//! `denom <= 0`) are all inequalities the fixtures keep away from.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `dot`,
//! `min`, `max` — with no transcendental call and no `64`-bit integers or
//! floats, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::bending`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. The bending solve is sequential *within* a
/// system, so one thread owns one entire cloth system; a batch of systems is
/// one independent thread each.
const WORKGROUP_SIZE: u32 = 1;

/// Maximum number of particles one query may carry.
///
/// The twin uploads fixed-length `std430` arrays, so a single cloth system in
/// one query is capped at this many particles. Systems in the test fixtures sit
/// well under the cap.
pub const MAX_PARTICLES: usize = 16;

/// Maximum number of bending hinges one query may carry.
pub const MAX_CONSTRAINTS: usize = 16;

/// The portable core-`WGSL` isometric-bending kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `bending_apply`
/// resolves one cloth system per thread, serially.
const CLOTH_BENDING_APPLY_WGSL: &str = r#"
// Cloth isometric-bending twin: one thread owns one cloth system and replays
// the golden apply_bending / apply_isometric_bending Gauss-Seidel sweep in
// place. The sweep is order sensitive, so the loop is serial per system;
// independent systems in a batch run on independent threads. Pure arithmetic;
// no transcendental.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::bending；无第三方引擎源码
// 或衍生代码。

const MAX_PARTICLES: u32 = 16u;
const MAX_CONSTRAINTS: u32 = 16u;
const EPS_LEN_SQ: f32 = 1e-12;

struct Params {
    // Number of systems in the batch; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle positions, flattened [x0, y0, z0, x1, y1, z1, ...].
    pos: array<f32, 48>,
    // Inverse masses, one per particle slot.
    inv_mass: array<f32, 16>,
    // Hinge vertex indices, four per constraint, flattened.
    vertices: array<u32, 64>,
    // Hinge weights, four per constraint, flattened.
    weights: array<f32, 64>,
    // Per-constraint area scale.
    scale: array<f32, 16>,
    // Per-constraint XPBD compliance.
    compliance: array<f32, 16>,
    // Valid particle / constraint counts.
    particle_count: u32,
    constraint_count: u32,
    // Clamped-at-least-one Gauss-Seidel sweep count.
    iterations: u32,
    // Substep dt; <= 0 makes every projection a no-op.
    dt: f32,
}

struct Outcome {
    // Solved particle positions, flattened like Query.pos.
    pos: array<f32, 48>,
    // First-sweep total corrected energy.
    first_energy: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

@compute @workgroup_size(1)
fn bending_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) {
        return;
    }
    let q = queries[gid.x];

    let pc = min(q.particle_count, MAX_PARTICLES);
    let cc = min(q.constraint_count, MAX_CONSTRAINTS);
    let iters = max(q.iterations, 1u);
    let dt = q.dt;

    // Private working copy of the positions, evolved in place.
    var pos: array<vec3<f32>, 16>;
    for (var i: u32 = 0u; i < pc; i = i + 1u) {
        let b = i * 3u;
        pos[i] = vec3<f32>(q.pos[b + 0u], q.pos[b + 1u], q.pos[b + 2u]);
    }

    var first_energy: f32 = 0.0;

    for (var it: u32 = 0u; it < iters; it = it + 1u) {
        for (var c: u32 = 0u; c < cc; c = c + 1u) {
            let base = c * 4u;
            let scale = q.scale[c];
            let compliance = q.compliance[c];
            var energy: f32 = 0.0;

            if (dt > 0.0) {
                // S = sum w_i * x_i over in-bounds slots.
                var s = vec3<f32>(0.0, 0.0, 0.0);
                for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                    let idx = q.vertices[base + k];
                    if (idx < pc) {
                        s = s + pos[idx] * q.weights[base + k];
                    }
                }
                let s_len_sq = dot(s, s);
                if (s_len_sq > EPS_LEN_SQ) {
                    energy = 0.5 * scale * s_len_sq;

                    // Denominator over free (inv_mass > 0) slots only.
                    var sum_w_grad: f32 = 0.0;
                    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                        let idx = q.vertices[base + k];
                        var im: f32 = 0.0;
                        if (idx < pc) {
                            im = q.inv_mass[idx];
                        }
                        if (im > 0.0) {
                            let grad_scalar = scale * q.weights[base + k];
                            sum_w_grad = sum_w_grad + im * grad_scalar * grad_scalar * s_len_sq;
                        }
                    }
                    let alpha_tilde = compliance / (dt * dt);
                    let denom = sum_w_grad + alpha_tilde;
                    if (denom > 0.0) {
                        let d_lambda = -energy / denom;
                        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
                            let idx = q.vertices[base + k];
                            var im: f32 = 0.0;
                            if (idx < pc) {
                                im = q.inv_mass[idx];
                            }
                            if (im > 0.0) {
                                pos[idx] = pos[idx] + s * (im * d_lambda * scale * q.weights[base + k]);
                            }
                        }
                    }
                }
            }

            if (it == 0u) {
                first_energy = first_energy + energy;
            }
        }
    }

    var out: Outcome;
    for (var i: u32 = 0u; i < MAX_PARTICLES; i = i + 1u) {
        let b = i * 3u;
        if (i < pc) {
            out.pos[b + 0u] = pos[i].x;
            out.pos[b + 1u] = pos[i].y;
            out.pos[b + 2u] = pos[i].z;
        } else {
            // Out-of-range slots are passed through untouched.
            out.pos[b + 0u] = q.pos[b + 0u];
            out.pos[b + 1u] = q.pos[b + 1u];
            out.pos[b + 2u] = q.pos[b + 2u];
        }
    }
    out.first_energy = first_energy;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[gid.x] = out;
}
"#;

/// Uniform parameters for the dispatch: the system count and three pad words,
/// filling a `16`-byte uniform struct matching `Params` in
/// [`CLOTH_BENDING_APPLY_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid systems in the batch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one cloth-system query, matching the `WGSL`
/// `Query` struct. Every member is a `f32`/`u32` scalar array so the struct
/// packs tightly on `4`-byte boundaries.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Flattened particle positions `[x, y, z]` per slot.
    pos: [f32; 48],
    /// Inverse masses, one per particle slot.
    inv_mass: [f32; 16],
    /// Hinge vertex indices, four per constraint, flattened.
    vertices: [u32; 64],
    /// Hinge weights, four per constraint, flattened.
    weights: [f32; 64],
    /// Per-constraint area scale.
    scale: [f32; 16],
    /// Per-constraint `XPBD` compliance.
    compliance: [f32; 16],
    /// Valid particle count.
    particle_count: u32,
    /// Valid constraint count.
    constraint_count: u32,
    /// Gauss-Seidel sweep count (clamped to at least one in-kernel).
    iterations: u32,
    /// Substep `dt`.
    dt: f32,
}

/// `repr(C)` `std430` layout of one solved system, matching the `WGSL`
/// `Outcome` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOutcome {
    /// Flattened solved particle positions.
    pos: [f32; 48],
    /// First-sweep total corrected energy.
    first_energy: f32,
    /// Padding word rounding the struct to a `16`-byte multiple.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One cloth-system query for the twin: the particle positions and inverse
/// masses, the bending hinges, and the sweep schedule.
///
/// Positions and inverse masses are parallel: `inverse_masses[i]` is the
/// inverse mass of `positions[i]`. The four constraint vectors are parallel
/// too: hinge `c` is `(vertices[c], weights[c], scales[c], compliances[c])`.
/// Up to [`MAX_PARTICLES`] particles and [`MAX_CONSTRAINTS`] hinges are
/// honored; anything beyond is clamped off.
#[derive(Clone, Debug, PartialEq)]
pub struct ClothBendingApplyQuery {
    /// Particle positions as `[x, y, z]`.
    pub positions: Vec<[f32; 3]>,
    /// Inverse masses aligned with `positions` (`0` pins a particle).
    pub inverse_masses: Vec<f32>,
    /// Hinge vertex stencils `[edge0, edge1, apex_a, apex_b]`.
    pub vertices: Vec<[u32; 4]>,
    /// Hinge weight stencils aligned with `vertices`.
    pub weights: Vec<[f32; 4]>,
    /// Per-hinge area scale aligned with `vertices`.
    pub scales: Vec<f32>,
    /// Per-hinge `XPBD` compliance aligned with `vertices`.
    pub compliances: Vec<f32>,
    /// Gauss-Seidel sweep count (clamped to at least one).
    pub iterations: u32,
    /// Substep `dt`.
    pub dt: f32,
}

/// One solved cloth system: the post-solve particle positions and the
/// first-sweep corrected energy, mirroring the reference
/// [`apply_bending`](prism_render_architecture::cloth::bending::apply_bending).
#[derive(Clone, Debug, PartialEq)]
pub struct ClothBendingApplyResult {
    /// Solved particle positions as `[x, y, z]`, one per input particle.
    pub positions: Vec<[f32; 3]>,
    /// Total bending energy corrected on the first sweep.
    pub first_energy: f32,
}

/// Encodes one [`ClothBendingApplyQuery`] into its `std430` [`GpuQuery`] slot,
/// clamping to the fixed [`MAX_PARTICLES`] / [`MAX_CONSTRAINTS`] capacities.
fn encode_query(q: &ClothBendingApplyQuery) -> GpuQuery {
    let mut g = GpuQuery {
        pos: [0.0; 48],
        inv_mass: [0.0; 16],
        vertices: [0; 64],
        weights: [0.0; 64],
        scale: [0.0; 16],
        compliance: [0.0; 16],
        particle_count: 0,
        constraint_count: 0,
        iterations: q.iterations,
        dt: q.dt,
    };

    let pc = q.positions.len().min(MAX_PARTICLES);
    for (i, p) in q.positions.iter().take(pc).enumerate() {
        g.pos[i * 3] = p[0];
        g.pos[i * 3 + 1] = p[1];
        g.pos[i * 3 + 2] = p[2];
    }
    for (i, m) in q.inverse_masses.iter().take(pc).enumerate() {
        g.inv_mass[i] = *m;
    }
    g.particle_count = pc as u32;

    let cc = q.vertices.len().min(MAX_CONSTRAINTS);
    for (c, v) in q.vertices.iter().take(cc).enumerate() {
        for (k, idx) in v.iter().enumerate() {
            g.vertices[c * 4 + k] = *idx;
        }
    }
    for (c, w) in q.weights.iter().take(cc).enumerate() {
        for (k, weight) in w.iter().enumerate() {
            g.weights[c * 4 + k] = *weight;
        }
    }
    for (c, s) in q.scales.iter().take(cc).enumerate() {
        g.scale[c] = *s;
    }
    for (c, comp) in q.compliances.iter().take(cc).enumerate() {
        g.compliance[c] = *comp;
    }
    g.constraint_count = cc as u32;

    g
}

/// Decodes one `std430` [`GpuOutcome`] into a [`ClothBendingApplyResult`],
/// returning `particle_count` solved positions.
fn decode_outcome(o: &GpuOutcome, particle_count: usize) -> ClothBendingApplyResult {
    let count = particle_count.min(MAX_PARTICLES);
    let positions = (0..count)
        .map(|i| [o.pos[i * 3], o.pos[i * 3 + 1], o.pos[i * 3 + 2]])
        .collect();
    ClothBendingApplyResult {
        positions,
        first_energy: o.first_energy,
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

/// A compiled, reusable cloth isometric-bending compute pipeline, twinning the
/// `CPU` golden
/// [`apply_bending`](prism_render_architecture::cloth::bending::apply_bending)
/// from [`bending`](prism_render_architecture::cloth::bending).
pub struct GpuClothBendingApply {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothBendingApply {
    /// Compiles the bending kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothBendingApply {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply"),
            source: ShaderSource::Wgsl(CLOTH_BENDING_APPLY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bending_apply"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothBendingApply {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every cloth system in `queries` and returns one
    /// [`ClothBendingApplyResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothBendingApplyQuery],
    ) -> Vec<ClothBendingApplyResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let result_bytes = (count * size_of::<GpuOutcome>()) as u64;
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_results"),
            size: result_bytes,
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
            label: Some("prism_volumetric_cloth_bending_apply_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_bending_apply_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cloth system, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_bending_apply_stage"),
            size: result_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&results_buf, 0, &stage, 0, result_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let outcomes = bytemuck::cast_slice::<u8, GpuOutcome>(&view).to_vec();
        drop(view);
        stage.unmap();

        queries
            .iter()
            .zip(outcomes.iter())
            .map(|(q, o)| decode_outcome(o, q.positions.len()))
            .collect()
    }
}
