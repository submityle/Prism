//! `wgpu` compute twin of the per-triangle mass-property contribution kernel
//! underlying this repository's
//! `prism_render_architecture::ray_scene::mesh_mass_properties::mass_properties`.
//!
//! The `CPU` golden `mass_properties` walks a whole closed
//! [`TriangleMesh`](prism_render_architecture::ray_scene::triangle_mesh::TriangleMesh),
//! accumulating each triangle's surface area, signed tetrahedron volume, first
//! moment and second-moment (covariance) matrix in `f64`, and separately
//! tallies directed edges in a `HashMap` for the watertight test before a final
//! centroid, inertia and watertight reduction. That whole-mesh reduction, the
//! directed-edge `HashMap`, the centroid and inertia derivation and the
//! watertight flag all stay on the host.
//!
//! This twin isolates the **per-triangle closed-form contribution**: it treats
//! a single triangle as one tetrahedron joined to the origin (Blow & Binstock)
//! and reproduces, term for term, the exact arithmetic the golden path applies
//! once per triangle:
//!
//! - the surface-area contribution `0.5 * |(b - a) x (c - a)|`,
//! - the signed tetrahedron volume `det(a, b, c) / 6` where the determinant
//!   has columns `a`, `b`, `c`,
//! - the first-moment contribution `tet_volume * (a + b + c) / 4`,
//! - the second-moment contribution `det * A * Ccanon * A^T`, with `A` holding
//!   columns `a`, `b`, `c` and `Ccanon` the canonical reference covariance
//!   scaled by `1/120`.
//!
//! # What is twinned
//!
//! [`GpuMeshVolumeContribution`] reproduces, for one triangle per query, the
//! `area`, `signed_volume`, `moment1` and `moment2` the golden path would add
//! into its running sums for that triangle. There are no discrete outputs.
//!
//! # What stays on the host
//!
//! The whole-mesh reduction (summing area, volume, first moment and
//! second-moment matrix across all triangles), the directed-edge `HashMap`
//! watertight tally, the final centroid, the second-moment translation to the
//! centroid and the inertia tensor derivation are a variable-length reduction
//! that is not twinned; only the per-triangle closed form above runs on the
//! device.
//!
//! # Precision model
//!
//! The golden path accumulates in `f64`; this twin and its host oracle both
//! evaluate the per-triangle closed form in `f32` so the comparison is a fair
//! device-against-device check of the same arithmetic. Each query is a fixed,
//! non-reorderable sequence of multiplies and adds, so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4 || rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous output.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `cross`, `length`,
//! `sqrt` and `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`,
//! `round` or optional device feature, and no `u64`, `i64` or `f64`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The only non-rational operation
//! is the area `sqrt`, matching the reference's `f64::sqrt` lowered to `f32`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_mass_properties`；无第三方引擎源码或衍生代码。

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
/// single source file. The single entry point `contribute` mirrors the
/// per-triangle closed form of the `CPU` golden `mass_properties`; see the
/// module documentation for the algorithm.
const MESH_VOLUME_CONTRIBUTION_WGSL: &str = r#"
// Per-triangle mass-property contribution twin: one thread per query computes
// the surface-area, signed tetrahedron volume, first moment and second-moment
// (covariance) contribution a single triangle adds to the golden
// mass_properties reduction (Blow & Binstock signed-tetrahedron decomposition).
// It uses only the portable core-WGSL subset (cross/length/sqrt and + - * /)
// with no u64/i64/f64 and no transcendental, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::mesh_mass_properties；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: a single triangle's three vertices, each vec3 lifted to a vec4
// slot so the std430 layout is explicit.
struct Query {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

// One result: the triangle's area and signed volume contribution, its first
// moment (3 scalars) and its second-moment matrix (9 scalars, row-major).
struct Res {
    area: f32,
    signed_volume: f32,
    m1x: f32,
    m1y: f32,
    m1z: f32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn contribute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a = q.a.xyz;
    let b = q.b.xyz;
    let c = q.c.xyz;

    // Surface area via half the cross-product magnitude.
    let ab = b - a;
    let ac = c - a;
    let area = 0.5 * length(cross(ab, ac));

    // Determinant of the matrix whose columns are a, b, c, matching the golden
    // term order exactly.
    let det = a.x * (b.y * c.z - b.z * c.y)
        - a.y * (b.x * c.z - b.z * c.x)
        + a.z * (b.x * c.y - b.y * c.x);
    let tet_volume = det / 6.0;

    // Tetrahedron centroid is the average of its four vertices, one of which is
    // the origin, so (a + b + c) / 4.
    let tc = (a + b + c) / 4.0;
    let m1 = vec3<f32>(tet_volume * tc.x, tet_volume * tc.y, tet_volume * tc.z);

    // Canonical reference covariance scaled by 1/120 (Blow & Binstock).
    let c2 = 2.0 / 120.0;
    let c1 = 1.0 / 120.0;

    // cat[k] = sum_m Ccanon[k][m] * col[m], where col[0]=a, col[1]=b, col[2]=c
    // and the vec3 ranges over the free column index j.
    let cat0 = c2 * a + c1 * b + c1 * c;
    let cat1 = c1 * a + c2 * b + c1 * c;
    let cat2 = c1 * a + c1 * b + c2 * c;

    // out row i = det * (a[i]*cat0 + b[i]*cat1 + c[i]*cat2); each row ranges
    // over j, matching the golden covariance_contribution summation order.
    let row0 = det * (a.x * cat0 + b.x * cat1 + c.x * cat2);
    let row1 = det * (a.y * cat0 + b.y * cat1 + c.y * cat2);
    let row2 = det * (a.z * cat0 + b.z * cat1 + c.z * cat2);

    var out: Res;
    out.area = area;
    out.signed_volume = tet_volume;
    out.m1x = m1.x;
    out.m1y = m1.y;
    out.m1z = m1.z;
    out.m00 = row0.x;
    out.m01 = row0.y;
    out.m02 = row0.z;
    out.m10 = row1.x;
    out.m11 = row1.y;
    out.m12 = row1.z;
    out.m20 = row2.x;
    out.m21 = row2.y;
    out.m22 = row2.z;
    results[idx] = out;
}
"#;

/// One per-triangle mass-property query: a single triangle's three vertices.
///
/// Mirrors a single reference triangle treated as one tetrahedron joined to the
/// origin. Vertices are ordered `a`, `b`, `c` to match the golden winding.
/// Derives only [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32`
/// geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVolumeContributionQuery {
    /// Position of the first vertex.
    pub a: [f32; 3],
    /// Position of the second vertex.
    pub b: [f32; 3],
    /// Position of the third vertex.
    pub c: [f32; 3],
}

impl MeshVolumeContributionQuery {
    /// Builds a query over the triangle's three vertices.
    #[must_use]
    pub const fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> MeshVolumeContributionQuery {
        MeshVolumeContributionQuery { a, b, c }
    }
}

/// The per-triangle mass-property contribution for one triangle query, the
/// host-side mirror of the kernel's `Res` lane.
///
/// `area` is the surface-area contribution, `signed_volume` the signed
/// tetrahedron volume, `moment1` the first-moment contribution and `moment2`
/// the second-moment (covariance) contribution as a row-major `3x3` matrix.
/// Derives only [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32`
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVolumeContributionResult {
    /// Surface-area contribution `0.5 * |(b - a) x (c - a)|`.
    pub area: f32,
    /// Signed tetrahedron volume `det(a, b, c) / 6`.
    pub signed_volume: f32,
    /// First-moment contribution `signed_volume * (a + b + c) / 4`.
    pub moment1: [f32; 3],
    /// Second-moment (covariance) contribution, row-major `3x3`.
    pub moment2: [f32; 9],
}

/// `repr(C)` `std430` layout of one packed query: three vertices, each `vec3`
/// lifted to a `vec4` slot — `48` bytes, exactly as the `WGSL` `Query` struct
/// reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First vertex, `xyz` plus a pad lane.
    a: [f32; 4],
    /// Second vertex, `xyz` plus a pad lane.
    b: [f32; 4],
    /// Third vertex, `xyz` plus a pad lane.
    c: [f32; 4],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &MeshVolumeContributionQuery) -> GpuQuery {
        GpuQuery {
            a: [query.a[0], query.a[1], query.a[2], 0.0],
            b: [query.b[0], query.b[1], query.b[2], 0.0],
            c: [query.c[0], query.c[1], query.c[2], 0.0],
        }
    }
}

/// `repr(C)` `std430` layout of one result: the `14` scalar `f32` outputs in
/// the same order as the `WGSL` `Res` struct — `56` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface-area contribution.
    area: f32,
    /// Signed tetrahedron volume.
    signed_volume: f32,
    /// First-moment `x`.
    m1x: f32,
    /// First-moment `y`.
    m1y: f32,
    /// First-moment `z`.
    m1z: f32,
    /// Second-moment row `0`, column `0`.
    m00: f32,
    /// Second-moment row `0`, column `1`.
    m01: f32,
    /// Second-moment row `0`, column `2`.
    m02: f32,
    /// Second-moment row `1`, column `0`.
    m10: f32,
    /// Second-moment row `1`, column `1`.
    m11: f32,
    /// Second-moment row `1`, column `2`.
    m12: f32,
    /// Second-moment row `2`, column `0`.
    m20: f32,
    /// Second-moment row `2`, column `1`.
    m21: f32,
    /// Second-moment row `2`, column `2`.
    m22: f32,
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

/// Decodes one packed `GpuResult` into the public
/// [`MeshVolumeContributionResult`].
fn decode_result(raw: &GpuResult) -> MeshVolumeContributionResult {
    MeshVolumeContributionResult {
        area: raw.area,
        signed_volume: raw.signed_volume,
        moment1: [raw.m1x, raw.m1y, raw.m1z],
        moment2: [
            raw.m00, raw.m01, raw.m02, raw.m10, raw.m11, raw.m12, raw.m20, raw.m21, raw.m22,
        ],
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

/// On-device twin of the per-triangle mass-property contribution kernel.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuMeshVolumeContribution::new`] and reuse it across
/// [`GpuMeshVolumeContribution::evaluate`] calls.
pub struct GpuMeshVolumeContribution {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `contribute` entry point.
    pipeline: ComputePipeline,
}

impl GpuMeshVolumeContribution {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshVolumeContribution {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_shader"),
            source: ShaderSource::Wgsl(MESH_VOLUME_CONTRIBUTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("contribute"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshVolumeContribution {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every triangle query on-device and returns one
    /// [`MeshVolumeContributionResult`] per input, in order.
    ///
    /// Each result equals the per-triangle reference contribution to within the
    /// tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MeshVolumeContributionQuery],
    ) -> Vec<MeshVolumeContributionResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_output"),
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
            label: Some("prism_volumetric_mesh_volume_contribution_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_bind_group"),
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
            label: Some("prism_volumetric_mesh_volume_contribution_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_volume_contribution_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_volume_contribution_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triangle query, flattened to a 1-D dispatch.
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
