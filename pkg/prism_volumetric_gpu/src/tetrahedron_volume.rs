//! `wgpu` compute twin of the signed-tetrahedron-volume geometry contract
//! ([`tetrahedron_volume`](prism_render_architecture::particle::tetrahedron_volume),
//! particle design §8.2, §11).
//!
//! The `CPU` golden
//! [`tetrahedron_volume`](prism_render_architecture::particle::tetrahedron_volume)
//! owns the pure solid-volume math for a tetrahedron: the raw orientation
//! determinant
//! [`orient3d_det`](prism_render_architecture::particle::tetrahedron_volume::orient3d_det)
//! (the scalar triple product `dot(b - a, cross(c - a, d - a))`), the
//! [`signed_volume`](prism_render_architecture::particle::tetrahedron_volume::signed_volume)
//! that is simply that determinant divided by six, and the
//! [`orient3d`](prism_render_architecture::particle::tetrahedron_volume::orient3d)
//! predicate that classifies the determinant's sign into
//! [`Orientation`](prism_render_architecture::particle::tetrahedron_volume::Orientation).
//! [`GpuTetrahedronVolume`] is the on-device twin: one thread per tetrahedron
//! reproduces all three answers (plus the absolute volume), so a passing
//! real-device parity test is direct evidence the ported kernel solves the same
//! geometry and classifies the same degenerate (coplanar) case the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-tetrahedron answer the reference computes is reproduced: the
//! orientation determinant, the signed volume (`det / 6`), the absolute volume
//! (`|signed_volume|`), and the three-way
//! [`Orientation`](prism_render_architecture::particle::tetrahedron_volume::Orientation)
//! classification. The sign classification mirrors the reference branch for
//! branch: a determinant whose magnitude is at or below the compare epsilon is
//! `Coplanar` (the degenerate case), a strictly positive determinant is
//! `Positive`, and a strictly negative one is `Negative`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs` and
//! `+ - * /` — with no `sqrt` (volume needs none), no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, `round` or `cbrt`, and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each tetrahedron is a fixed, non-reorderable sequence of multiplies, adds and
//! one divide, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, while the
//! integer orientation code is compared exactly: fixtures stay well clear of the
//! degeneracy crack so both devices land on the same side of the epsilon.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`tetrahedron_volume`](prism_render_architecture::particle::tetrahedron_volume);
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::tetrahedron_volume::Orientation;
use prism_render_architecture::particle::Vec3;
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

/// Orientation code written by the kernel for a positively-oriented tetrahedron.
const ORIENT_POSITIVE: u32 = 0;
/// Orientation code written by the kernel for a negatively-oriented tetrahedron.
const ORIENT_NEGATIVE: u32 = 1;

/// The portable core-`WGSL` tetrahedron-volume kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`tetrahedron_volume`](prism_render_architecture::particle::tetrahedron_volume);
/// see the module documentation for the algorithm.
const TETRAHEDRON_VOLUME_WGSL: &str = r#"
// Signed-tetrahedron-volume twin: one thread per tetrahedron reproduces the
// orientation determinant dot(b - a, cross(c - a, d - a)), the signed volume
// (det / 6), the absolute volume (|signed volume|) and the three-way
// orientation classification. It mirrors the CPU golden
// `particle::tetrahedron_volume` branch for branch, uses only the portable
// core-WGSL subset (abs and + - * /), needs no sqrt (volume needs none) and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::tetrahedron_volume; no
// third-party engine source or derived code.

// Magnitude below which the orientation determinant is treated as zero
// (coplanar / degenerate), never written as an exact == / != on an f32.
// Matches the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

// Orientation codes shared with the host-side decode.
const ORIENT_POSITIVE: u32 = 0u;
const ORIENT_NEGATIVE: u32 = 1u;
const ORIENT_COPLANAR: u32 = 2u;

struct Params {
    // Number of tetrahedra in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Tetra {
    // First vertex a; a pad lane follows.
    va: vec3<f32>,
    pad0: f32,
    // Second vertex b; a pad lane follows.
    vb: vec3<f32>,
    pad1: f32,
    // Third vertex c; a pad lane follows.
    vc: vec3<f32>,
    pad2: f32,
    // Fourth vertex d; a pad lane follows.
    vd: vec3<f32>,
    pad3: f32,
}

struct Result {
    // Signed volume, absolute volume, orientation determinant and the integer
    // orientation code: four scalars filling one vec4 slot.
    signed_volume: f32,
    abs_volume: f32,
    det: f32,
    orientation: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tetra: array<Tetra>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let t = tetra[idx];
    let va = t.va;
    let vb = t.vb;
    let vc = t.vc;
    let vd = t.vd;

    // Scalar triple product dot(b - a, cross(c - a, d - a)): six times the
    // signed volume, i.e. the raw orientation determinant.
    let det = dot(vb - va, cross(vc - va, vd - va));
    let signed_vol = det / 6.0;
    let abs_vol = abs(signed_vol);

    // Classify the sign, treating a near-zero determinant as coplanar.
    var orient: u32 = ORIENT_COPLANAR;
    if (abs(det) <= CMP_EPS) {
        orient = ORIENT_COPLANAR;
    } else if (det > 0.0) {
        orient = ORIENT_POSITIVE;
    } else {
        orient = ORIENT_NEGATIVE;
    }

    var out: Result;
    out.signed_volume = signed_vol;
    out.abs_volume = abs_vol;
    out.det = det;
    out.orientation = orient;
    results[idx] = out;
}
"#;

/// One tetrahedron-volume query: the four vertices `(a, b, c, d)` the reference
/// [`orient3d_det`](prism_render_architecture::particle::tetrahedron_volume::orient3d_det),
/// [`signed_volume`](prism_render_architecture::particle::tetrahedron_volume::signed_volume)
/// and
/// [`orient3d`](prism_render_architecture::particle::tetrahedron_volume::orient3d)
/// consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetrahedronVolumeQuery {
    /// First vertex `a` (the base of the orientation winding).
    pub a: Vec3,
    /// Second vertex `b`.
    pub b: Vec3,
    /// Third vertex `c`.
    pub c: Vec3,
    /// Fourth vertex `d` (whose side of the `(a, b, c)` plane sets the sign).
    pub d: Vec3,
}

impl TetrahedronVolumeQuery {
    /// Builds a query from the four tetrahedron vertices.
    #[must_use]
    pub const fn new(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> TetrahedronVolumeQuery {
        TetrahedronVolumeQuery { a, b, c, d }
    }
}

/// The resolved answer for one tetrahedron, mirroring every value the reference
/// reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetrahedronVolumeResult {
    /// Signed volume `det / 6`, matching
    /// [`signed_volume`](prism_render_architecture::particle::tetrahedron_volume::signed_volume).
    pub signed_volume: f32,
    /// Absolute (unsigned) volume `|signed_volume|`.
    pub abs_volume: f32,
    /// Orientation determinant `dot(b - a, cross(c - a, d - a))`, matching
    /// [`orient3d_det`](prism_render_architecture::particle::tetrahedron_volume::orient3d_det).
    pub det: f32,
    /// Three-way orientation classification, matching
    /// [`orient3d`](prism_render_architecture::particle::tetrahedron_volume::orient3d).
    pub orientation: Orientation,
}

/// `repr(C)` `std430` layout of one packed tetrahedron: four `vec4` slots
/// holding `(a.xyz, pad)`, `(b.xyz, pad)`, `(c.xyz, pad)` and `(d.xyz, pad)` —
/// `64` bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Tetra` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTetra {
    /// First vertex `a`.
    a: [f32; 3],
    /// Padding lane after `a`.
    pad0: f32,
    /// Second vertex `b`.
    b: [f32; 3],
    /// Padding lane after `b`.
    pad1: f32,
    /// Third vertex `c`.
    c: [f32; 3],
    /// Padding lane after `c`.
    pad2: f32,
    /// Fourth vertex `d`.
    d: [f32; 3],
    /// Padding lane after `d`.
    pad3: f32,
}

impl GpuTetra {
    /// Packs one query into its `std430` image.
    fn new(query: &TetrahedronVolumeQuery) -> GpuTetra {
        GpuTetra {
            a: [query.a.x, query.a.y, query.a.z],
            pad0: 0.0,
            b: [query.b.x, query.b.y, query.b.z],
            pad1: 0.0,
            c: [query.c.x, query.c.y, query.c.z],
            pad2: 0.0,
            d: [query.d.x, query.d.y, query.d.z],
            pad3: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar `vec4` slot holding
/// `(signed_volume, abs_volume, det, orientation)` — `16` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed volume `det / 6`.
    signed_volume: f32,
    /// Absolute volume `|signed_volume|`.
    abs_volume: f32,
    /// Orientation determinant.
    det: f32,
    /// Integer orientation code (`0` positive, `1` negative, `2` coplanar).
    orientation: u32,
}

/// Uniform parameters for one dispatch: the tetrahedron count plus three pad
/// words to fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of tetrahedra in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable tetrahedron-volume compute pipeline.
pub struct GpuTetrahedronVolume {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTetrahedronVolume {
    /// Compiles the tetrahedron-volume kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTetrahedronVolume {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume"),
            source: ShaderSource::Wgsl(TETRAHEDRON_VOLUME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTetrahedronVolume {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every tetrahedron on-device and returns one
    /// [`TetrahedronVolumeResult`] per input, in order.
    ///
    /// Each result equals the reference answers (`orient3d_det`,
    /// `signed_volume`, its absolute value and `orient3d`) to within the
    /// tolerance documented on this module, with the orientation code exact. An
    /// empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[TetrahedronVolumeQuery],
    ) -> Vec<TetrahedronVolumeResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuTetra> = queries.iter().map(GpuTetra::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_output"),
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
            label: Some("prism_volumetric_tetrahedron_volume_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_bind_group"),
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
            label: Some("prism_volumetric_tetrahedron_volume_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_tetrahedron_volume_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tetrahedron_volume_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per tetrahedron, flattened to a 1-D dispatch.
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

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TetrahedronVolumeResult`].
fn decode_result(raw: &GpuResult) -> TetrahedronVolumeResult {
    let orientation = match raw.orientation {
        ORIENT_POSITIVE => Orientation::Positive,
        ORIENT_NEGATIVE => Orientation::Negative,
        _ => Orientation::Coplanar,
    };
    TetrahedronVolumeResult {
        signed_volume: raw.signed_volume,
        abs_volume: raw.abs_volume,
        det: raw.det,
        orientation,
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
