//! `wgpu` compute twin of the meshlet software-raster per-vertex projection
//! stage inside the virtual-geometry pipeline
//! ([`software_raster`](prism_render_architecture::virtual_geometry::software_raster)).
//!
//! The `CPU` golden
//! [`software_raster`](prism_render_architecture::virtual_geometry::software_raster)
//! transforms a world-space vertex through a column-major clip matrix into
//! y-down pixel space with a reversed-Z depth, and separately encodes that depth
//! into a compositing key. Two pure functions carry that contract:
//! [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
//! (matrix-vector product, perspective divide, `ndc_to_uv`, viewport scale, with
//! a near-plane cull at `clip.w <= 0`) and
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
//! (clamp to `0..=1`, then the raw `IEEE` bit pattern). Both are pure
//! arithmetic plus one `bitcast`, with no transcendental and no `f32` equality
//! test.
//!
//! [`GpuGeomRasterProject`] is the on-device twin of exactly those two cores.
//! One thread projects one vertex — its validity flag, y-down pixel position,
//! reversed-Z depth and encoded depth key — reproducing the reference's exact
//! closed form, so a passing real-device parity test is direct evidence the
//! ported kernel projects the same way the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one vertex the kernel reproduces, in closed form:
//!
//! - [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex):
//!   the column-major product `clip[r] = sum over c of clip_from_world[c][r] *
//!   world_h[c]` (with `world_h = (world, 1)`), the near-plane cull
//!   `clip.w <= 0` reported as a `valid` flag, the perspective divide
//!   `ndc = clip.xyz / clip.w`, the `ndc_to_uv` map
//!   `uv = ndc.xy * (0.5, -0.5) + 0.5`, the viewport scale `pos = uv * viewport`
//!   (y-down pixels) and the reversed-Z `depth = ndc.z`.
//! - [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth):
//!   `bitcast<u32>(clamp(depth, 0, 1))`, matching the golden's
//!   `depth.clamp(0, 1).to_bits()` bit-for-bit.
//!
//! The emitted result carries the `valid` flag, the two `screen_pos`
//! components, the reversed-Z `depth_ndc`, and the `encoded_depth` key.
//!
//! # What stays on the host
//!
//! Triangle and cluster rasterization, the vis-buffer packing (which uses
//! `u64`), near-plane clipping topology and every variable-length aggregate stay
//! host-side; the device sees one independent vertex per thread. An empty batch
//! short-circuits with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The `valid` flag and the `encoded_depth` bit pattern are discrete, asserted
//! bit-exact (`==`) in the parity test; the screen position and depth thread
//! through multiplies and a guarded divide, so a `GPU` divide may land a few
//! units in the last place from the scalar reference and are asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor `1e-6`). The cull
//! boundary (`clip.w ~ 0`) is a discontinuity that flips `valid`; fixtures and
//! the randomized sweep keep every sample's `clip.w` well clear of zero so `CPU`
//! and `GPU` cannot straddle it.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `+ - * /`,
//! `bitcast`, unsigned index arithmetic and `select` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no
//! `round` and no `sqrt`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` software-raster vertex-projection twin, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
/// and
/// [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
/// closed forms; see the module documentation for the algorithm.
const GEOM_RASTER_PROJECT_WGSL: &str = r#"
// Software-raster vertex projection twin: one thread projects one world-space
// vertex through a column-major clip matrix into y-down pixel space with a
// reversed-Z depth, flags the near-plane cull (clip.w <= 0), and encodes a
// clamped depth into its bit-pattern key. Mirrors the CPU golden
// `virtual_geometry::software_raster` closed forms with only clamp, bitcast,
// select and + - * /. It owns no triangle/cluster rasterization and no
// vis-buffer packing; those stay host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::virtual_geometry::software_raster；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of vertices in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Column-major 4x4 clip-from-world matrix: m[c * 4 + r] is column c, row r.
    m: array<f32, 16>,
    // World-space position (homogeneous w = 1).
    wx: f32,
    wy: f32,
    wz: f32,
    // Viewport size in pixels.
    vp_x: f32,
    vp_y: f32,
    // Depth fed to encode_depth (independent of the projection).
    depth: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // 1 when clip.w > 0 (projectable), else 0 (near-plane culled).
    valid: u32,
    // bitcast<u32>(clamp(depth, 0, 1)): the compositing depth key.
    encoded_depth: u32,
    // y-down pixel-space position.
    screen_x: f32,
    screen_y: f32,
    // Reversed-Z NDC depth (ndc.z).
    depth_ndc: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Homogeneous world position.
    let w0 = q.wx;
    let w1 = q.wy;
    let w2 = q.wz;
    let w3 = 1.0;

    // Column-major matrix-vector product: clip[r] = sum_c m[c*4+r] * world_h[c].
    let c0 = q.m[0] * w0 + q.m[4] * w1 + q.m[8] * w2 + q.m[12] * w3;
    let c1 = q.m[1] * w0 + q.m[5] * w1 + q.m[9] * w2 + q.m[13] * w3;
    let c2 = q.m[2] * w0 + q.m[6] * w1 + q.m[10] * w2 + q.m[14] * w3;
    let c3 = q.m[3] * w0 + q.m[7] * w1 + q.m[11] * w2 + q.m[15] * w3;

    // Near-plane cull: the perspective divide is undefined for clip.w <= 0.
    let valid = select(0u, 1u, c3 > 0.0);

    var sx = 0.0;
    var sy = 0.0;
    var dz = 0.0;
    if (c3 > 0.0) {
        let inv_w = 1.0 / c3;
        let ndc0 = c0 * inv_w;
        let ndc1 = c1 * inv_w;
        let ndc2 = c2 * inv_w;
        // ndc_to_uv flips y so uv is y-down, then scales into the viewport.
        let u = ndc0 * 0.5 + 0.5;
        let v = ndc1 * -0.5 + 0.5;
        sx = u * q.vp_x;
        sy = v * q.vp_y;
        dz = ndc2;
    }

    // encode_depth: clamp to [0,1] then take the raw IEEE bit pattern.
    let encoded = bitcast<u32>(clamp(q.depth, 0.0, 1.0));

    var out: Result;
    out.valid = valid;
    out.encoded_depth = encoded;
    out.screen_x = sx;
    out.screen_y = sy;
    out.depth_ndc = dz;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the vertex count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`GEOM_RASTER_PROJECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid vertices in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one vertex query, matching the `WGSL` `Query`
/// struct: the column-major clip matrix flattened to `16` words, the world
/// position, the viewport, the depth to encode, and two pad words to a
/// `96`-byte stride of `24` words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Column-major clip-from-world matrix, `m[c * 4 + r]` is column `c`, row
    /// `r`.
    m: [f32; 16],
    /// World-space `x`.
    wx: f32,
    /// World-space `y`.
    wy: f32,
    /// World-space `z`.
    wz: f32,
    /// Viewport width in pixels.
    vp_x: f32,
    /// Viewport height in pixels.
    vp_y: f32,
    /// Depth fed to `encode_depth`.
    depth: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one vertex result, matching the `WGSL` `Result`
/// struct: the validity flag, the encoded depth key, the two screen-position
/// components, the reversed-Z depth, and three pad words to a `32`-byte stride
/// of eight words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when projectable, else `0`.
    valid: u32,
    /// `bitcast<u32>(clamp(depth, 0, 1))`.
    encoded_depth: u32,
    /// y-down pixel-space `x`.
    screen_x: f32,
    /// y-down pixel-space `y`.
    screen_y: f32,
    /// Reversed-Z NDC depth.
    depth_ndc: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One per-vertex query for the software-raster projection twin: the
/// column-major clip matrix, the world position, the viewport, and the depth to
/// encode.
///
/// [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
/// reads [`clip_from_world`](Self::clip_from_world), [`world_pos`](Self::world_pos)
/// and [`viewport`](Self::viewport);
/// [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
/// reads [`depth`](Self::depth) independently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomRasterProjectQuery {
    /// Column-major `4x4` clip-from-world matrix: `clip_from_world[c]` is column
    /// `c`, matching the golden signature.
    pub clip_from_world: [[f32; 4]; 4],
    /// World-space position (homogeneous `w = 1`).
    pub world_pos: [f32; 3],
    /// Viewport size in pixels `(width, height)`.
    pub viewport: [f32; 2],
    /// Depth fed to `encode_depth`.
    pub depth: f32,
}

impl GeomRasterProjectQuery {
    /// Builds a query from the clip matrix, world position, viewport and depth.
    #[must_use]
    pub const fn new(
        clip_from_world: [[f32; 4]; 4],
        world_pos: [f32; 3],
        viewport: [f32; 2],
        depth: f32,
    ) -> GeomRasterProjectQuery {
        GeomRasterProjectQuery {
            clip_from_world,
            world_pos,
            viewport,
            depth,
        }
    }
}

/// One projected vertex of the software-raster twin, mirroring the golden
/// [`project_vertex`](prism_render_architecture::virtual_geometry::software_raster::project_vertex)
/// screen vertex and the
/// [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
/// key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomRasterProjectResult {
    /// `1` when the vertex is projectable (`clip.w > 0`), else `0`.
    pub valid: u32,
    /// y-down pixel-space position (meaningful only when [`valid`](Self::valid)
    /// is `1`).
    pub screen_pos: [f32; 2],
    /// Reversed-Z NDC depth (meaningful only when [`valid`](Self::valid) is
    /// `1`).
    pub depth_ndc: f32,
    /// `bitcast<u32>(clamp(depth, 0, 1))`, the compositing depth key.
    pub encoded_depth: u32,
}

/// Encodes one [`GeomRasterProjectQuery`] into its `std430` [`GpuQuery`] slot.
/// The matrix is flattened column-major (`m[c * 4 + r]`); the pad words are
/// zeroed.
fn encode_query(q: &GeomRasterProjectQuery) -> GpuQuery {
    let mut m = [0.0_f32; 16];
    let mut col = 0;
    while col < 4 {
        let mut row = 0;
        while row < 4 {
            m[col * 4 + row] = q.clip_from_world[col][row];
            row += 1;
        }
        col += 1;
    }
    GpuQuery {
        m,
        wx: q.world_pos[0],
        wy: q.world_pos[1],
        wz: q.world_pos[2],
        vp_x: q.viewport[0],
        vp_y: q.viewport[1],
        depth: q.depth,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GeomRasterProjectResult`].
fn decode_result(raw: &GpuResult) -> GeomRasterProjectResult {
    GeomRasterProjectResult {
        valid: raw.valid,
        screen_pos: [raw.screen_x, raw.screen_y],
        depth_ndc: raw.depth_ndc,
        encoded_depth: raw.encoded_depth,
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

/// A compiled, reusable software-raster vertex-projection compute pipeline,
/// twinning the numeric core of the `CPU` golden
/// [`software_raster`](prism_render_architecture::virtual_geometry::software_raster).
pub struct GpuGeomRasterProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGeomRasterProject {
    /// Compiles the software-raster projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGeomRasterProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_geom_raster_project"),
            source: ShaderSource::Wgsl(GEOM_RASTER_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_geom_raster_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_geom_raster_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_geom_raster_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGeomRasterProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every vertex in `queries` and returns one
    /// [`GeomRasterProjectResult`] per input, in order.
    ///
    /// The validity flag and encoded depth match the reference bit-exactly; the
    /// screen position and depth match within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GeomRasterProjectQuery],
    ) -> Vec<GeomRasterProjectResult> {
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
            label: Some("prism_volumetric_geom_raster_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_geom_raster_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_geom_raster_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_geom_raster_project_bind_group"),
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
            label: Some("prism_volumetric_geom_raster_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_geom_raster_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_geom_raster_project_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per vertex, flattened to a 1-D dispatch.
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
