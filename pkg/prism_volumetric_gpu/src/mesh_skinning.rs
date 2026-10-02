//! `wgpu` compute twin of the `CPU` golden linear-blend skinning
//! (`LBS`) pure functions
//! ([`mesh_emission`](prism_render_architecture::particle::mesh_emission),
//! design §12).
//!
//! Posing a skeletal mesh deforms every bind-pose vertex by a weighted blend of
//! its influencing bone transforms. The `CPU` golden
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission) module
//! owns the math for the two pure functions this twin reproduces, and
//! [`GpuMeshSkinning`] is the on-device twin validated against that reference so
//! a passing real-device parity test is direct evidence the ported kernel blends
//! the same transforms in the same order, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Each thread owns one vertex and produces, in a single [`GpuSkinResult`], the
//! output of two golden functions evaluated at that vertex:
//!
//! - [`position`](GpuSkinResult::position) is the skinned position
//!   `Σ wᵢ·(Bᵢ · p)` over the (up to four) influences, twinning
//!   [`skin_position`](prism_render_architecture::particle::mesh_emission::skin_position).
//! - [`normal`](GpuSkinResult::normal) is the skinned, renormalized normal,
//!   twinning
//!   [`skin_normal`](prism_render_architecture::particle::mesh_emission::skin_normal)
//!   (basis-only transform, then
//!   [`normalize_or_zero`](prism_render_architecture::particle::Vec3::normalize_or_zero)).
//!
//! A single [`BoneTransform`](prism_render_architecture::particle::mesh_emission::BoneTransform)
//! is a `3x4` affine map, so the kernel applies it as the left-associated sum
//! `basis_x·p.x + basis_y·p.y + basis_z·p.z + translation`, matching the
//! reference tap for tap; the normal uses the same basis without the
//! translation. Near-zero weights (`|w| < EPS`) and out-of-range bone indices
//! are skipped on both sides.
//!
//! # Correctness model
//!
//! Every output is a multiply-add plus the normal's `sqrt`-based renormalize;
//! there is no transcendental call, so `CPU` and `GPU` evaluate the same closed
//! form. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few `ULP`.
//! The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`), tight enough to catch a genuinely wrong port (a swapped
//! basis column, a dropped translation, a missing weight skip) yet loose enough
//! to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! An empty vertex batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized). An empty bone palette uploads one placeholder
//! bone the kernel never reads because `bone_count` is zero, so every influence
//! is skipped and the result is the zero position and the zero normal. A vertex
//! whose blended normal collapses to (numerically) the zero vector
//! (`len² <= EPS_LEN_SQ`) renormalizes to
//! [`Vec3::ZERO`](prism_render_architecture::particle::Vec3::ZERO), exactly as
//! the reference guard returns.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ − × ÷` on `f32` vectors and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `pow`, optional device feature or `u64`, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission` 的
//! 线性混合蒙皮纯函数；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::mesh_emission::{BoneTransform, SkinnedVertex};
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

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` skinning kernel, embedded inline so the twin ships
/// as a single source file. One thread owns one vertex; see the module
/// documentation for the algorithm.
const MESH_SKINNING_WGSL: &str = r#"
// Linear-blend skinning twin: one thread owns one vertex and reproduces two CPU
// golden functions in one result record (the skinned position and the skinned,
// renormalized normal). It uses only the portable core-WGSL subset (integer
// index math plus + - * / on f32 vectors, abs, sqrt), takes no optional
// feature, and runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::mesh_emission 的线性混合蒙皮纯函数；无第三方引擎源码或衍生代码。

struct Params {
    // Number of bone transforms in the palette; influences at or past this index
    // are skipped, matching the reference's bounds check.
    bone_count: u32,
    // Number of valid vertices; threads past this short-circuit.
    vertex_count: u32,
    pad0: u32,
    pad1: u32,
}

struct Vertex {
    // Bind-pose position (xyz; w unused).
    position: vec4<f32>,
    // Bind-pose normal (xyz; w unused).
    normal: vec4<f32>,
    // Per-influence blend weights.
    weights: vec4<f32>,
    // Per-influence bone-palette indices.
    bones: vec4<u32>,
}

struct SkinResult {
    // Skinned position (xyz; w zero).
    position: vec4<f32>,
    // Skinned, renormalized normal (xyz; w zero).
    normal: vec4<f32>,
}

// The weight-skip threshold, matching mesh_emission::EPS.
const EPS: f32 = 1e-6;
// The squared-length floor below which a normal renormalizes to zero, matching
// particle::EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1e-12;

@group(0) @binding(0) var<uniform> params: Params;
// The bone palette, four vec4 per bone (basis_x, basis_y, basis_z, translation;
// xyz used, w pad).
@group(0) @binding(1) var<storage, read> bones: array<vec4<f32>>;
// The per-thread vertex batch.
@group(0) @binding(2) var<storage, read> vertices: array<Vertex>;
// The per-thread result batch.
@group(0) @binding(3) var<storage, read_write> results: array<SkinResult>;

@compute @workgroup_size(64)
fn skin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.vertex_count) {
        return;
    }

    let v = vertices[idx];
    let pos = v.position.xyz;
    let nrm = v.normal.xyz;

    var acc_pos = vec3<f32>(0.0, 0.0, 0.0);
    var acc_nrm = vec3<f32>(0.0, 0.0, 0.0);

    // Bounded four-influence blend loop; skips near-zero weights and
    // out-of-range bone indices exactly as the reference does.
    for (var i = 0u; i < 4u; i = i + 1u) {
        let w = v.weights[i];
        if (abs(w) < EPS) {
            continue;
        }
        let bone_idx = v.bones[i];
        if (bone_idx >= params.bone_count) {
            continue;
        }
        let base = bone_idx * 4u;
        let basis_x = bones[base].xyz;
        let basis_y = bones[base + 1u].xyz;
        let basis_z = bones[base + 2u].xyz;
        let translation = bones[base + 3u].xyz;
        // Affine point transform (left-associated), matching transform_point.
        let tp = basis_x * pos.x + basis_y * pos.y + basis_z * pos.z + translation;
        // Basis-only vector transform, matching transform_vector.
        let tv = basis_x * nrm.x + basis_y * nrm.y + basis_z * nrm.z;
        acc_pos = acc_pos + tp * w;
        acc_nrm = acc_nrm + tv * w;
    }

    // Renormalize the blended normal, falling back to zero for a degenerate
    // (numerically zero) accumulation, matching normalize_or_zero.
    let len_sq = dot(acc_nrm, acc_nrm);
    var out_nrm = vec3<f32>(0.0, 0.0, 0.0);
    if (len_sq > EPS_LEN_SQ) {
        out_nrm = acc_nrm * (1.0 / sqrt(len_sq));
    }

    results[idx].position = vec4<f32>(acc_pos, 0.0);
    results[idx].normal = vec4<f32>(out_nrm, 0.0);
}
"#;

/// Uniform parameters for one skinning dispatch. `repr(C)` `std140` layout
/// matching `Params` in [`MESH_SKINNING_WGSL`]: the bone count, the vertex count
/// and two pad words — `16` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of bone transforms in the palette.
    bone_count: u32,
    /// Number of valid vertices in the batch.
    vertex_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One bone transform as uploaded. `64`-byte `std430` stride matching four
/// consecutive `vec4<f32>` lanes in the shader: the three basis columns plus the
/// translation, each `xyz` carrying the data and `w` held at zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBone {
    /// Image of the object-space `+X` axis (`xyz`; `w` zero pad).
    basis_x: [f32; 4],
    /// Image of the object-space `+Y` axis (`xyz`; `w` zero pad).
    basis_y: [f32; 4],
    /// Image of the object-space `+Z` axis (`xyz`; `w` zero pad).
    basis_z: [f32; 4],
    /// Translation column (`xyz`; `w` zero pad).
    translation: [f32; 4],
}

impl GpuBone {
    /// Packs a [`BoneTransform`] into the padded device layout.
    ///
    /// Provenance: local device-upload helper for `GpuMeshSkinning`; no third
    /// party engine source or derived code.
    fn from_bone(b: &BoneTransform) -> Self {
        GpuBone {
            basis_x: [b.basis_x.x, b.basis_x.y, b.basis_x.z, 0.0],
            basis_y: [b.basis_y.x, b.basis_y.y, b.basis_y.z, 0.0],
            basis_z: [b.basis_z.x, b.basis_z.y, b.basis_z.z, 0.0],
            translation: [b.translation.x, b.translation.y, b.translation.z, 0.0],
        }
    }
}

/// One vertex as uploaded to the device. `64`-byte `std430` stride matching
/// `Vertex` in [`MESH_SKINNING_WGSL`]: the padded position and normal, the four
/// blend weights and the four bone indices (widened from `u16` to `u32`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertexRaw {
    /// Bind-pose position (`xyz`; `w` unused).
    position: [f32; 4],
    /// Bind-pose normal (`xyz`; `w` unused).
    normal: [f32; 4],
    /// Per-influence blend weights.
    weights: [f32; 4],
    /// Per-influence bone-palette indices, widened to `u32` for the device.
    bones: [u32; 4],
}

/// One result as read back from the device. `32`-byte `std430` stride matching
/// `SkinResult` in [`MESH_SKINNING_WGSL`]: the skinned position and the skinned,
/// renormalized normal, each `xyz` carrying the data and `w` zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResultRaw {
    /// Skinned position (`xyz`; `w` zero).
    position: [f32; 4],
    /// Skinned, renormalized normal (`xyz`; `w` zero).
    normal: [f32; 4],
}

/// The two golden skinning outputs evaluated at one
/// [`SkinnedVertex`](prism_render_architecture::particle::mesh_emission::SkinnedVertex).
///
/// Provenance: twin output record for the golden
/// [`mesh_emission`](prism_render_architecture::particle::mesh_emission) skinning
/// pure functions; no third party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSkinResult {
    /// The skinned position `Σ wᵢ·(Bᵢ · p)` over the influences.
    pub position: Vec3,
    /// The skinned, renormalized normal
    /// ([`Vec3::ZERO`](prism_render_architecture::particle::Vec3::ZERO) for a
    /// degenerate accumulation).
    pub normal: Vec3,
}

/// A compiled, reusable linear-blend skinning kernel, twinning the `CPU` golden
/// [`mesh_emission`](prism_render_architecture::particle::mesh_emission) skinning
/// pure functions.
///
/// Provenance: on-device twin of the golden
/// [`mesh_emission`](prism_render_architecture::particle::mesh_emission) skinning
/// pure functions; no third party engine source or derived code.
pub struct GpuMeshSkinning {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMeshSkinning {
    /// Compiles the skinning kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    ///
    /// Provenance: pipeline construction for the golden
    /// [`mesh_emission`](prism_render_architecture::particle::mesh_emission)
    /// skinning twin; no third party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMeshSkinning {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_skinning_module"),
            source: ShaderSource::Wgsl(MESH_SKINNING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_skinning_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_skinning_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_skinning_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("skin"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMeshSkinning {
            module,
            layout,
            pipeline,
        }
    }

    /// Skins every vertex under the shared bone palette `bones`, returning one
    /// [`GpuSkinResult`] per vertex in input order.
    ///
    /// Each result matches the golden functions to within the tolerance
    /// documented on this module. An empty vertex batch returns an empty vector
    /// with no dispatch issued (a storage buffer cannot be zero-sized). An empty
    /// bone palette uploads one placeholder bone the kernel never reads because
    /// `bone_count` is zero, so every influence is skipped and the result is the
    /// zero position and the zero normal. Near-zero weights and out-of-range bone
    /// indices are skipped, matching the reference guards.
    ///
    /// Provenance: dispatch and read-back for the golden
    /// [`mesh_emission`](prism_render_architecture::particle::mesh_emission)
    /// skinning twin; no third party engine source or derived code.
    #[must_use]
    pub fn skin(
        &self,
        ctx: &GpuContext,
        vertices: &[SkinnedVertex],
        bones: &[BoneTransform],
    ) -> Vec<GpuSkinResult> {
        if vertices.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();

        let gpu_params = Params {
            bone_count: bones.len() as u32,
            vertex_count: vertices.len() as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_skinning_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });

        // A storage buffer cannot be zero-sized; for an empty palette upload a
        // single placeholder bone the kernel never reads because `bone_count` is
        // zero.
        let placeholder = [GpuBone {
            basis_x: [0.0; 4],
            basis_y: [0.0; 4],
            basis_z: [0.0; 4],
            translation: [0.0; 4],
        }];
        let packed: Vec<GpuBone> = bones.iter().map(GpuBone::from_bone).collect();
        let bone_contents: &[GpuBone] = if packed.is_empty() {
            &placeholder
        } else {
            &packed
        };
        let bones_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_skinning_bones"),
            contents: bytemuck::cast_slice(bone_contents),
            usage: BufferUsages::STORAGE,
        });

        let raw_vertices: Vec<GpuVertexRaw> = vertices
            .iter()
            .map(|v| GpuVertexRaw {
                position: [v.position.x, v.position.y, v.position.z, 0.0],
                normal: [v.normal.x, v.normal.y, v.normal.z, 0.0],
                weights: [v.weights[0], v.weights[1], v.weights[2], v.weights[3]],
                bones: [
                    u32::from(v.bones[0]),
                    u32::from(v.bones[1]),
                    u32::from(v.bones[2]),
                    u32::from(v.bones[3]),
                ],
            })
            .collect();
        let vertices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_skinning_vertices"),
            contents: bytemuck::cast_slice(&raw_vertices),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (vertices.len() * size_of::<GpuResultRaw>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_skinning_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_skinning_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bones_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: vertices_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_skinning_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_skinning_encoder"),
        });
        {
            let groups = (vertices.len() as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_skinning_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResultRaw>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(raw.len(), vertices.len());

        raw.into_iter()
            .map(|r| GpuSkinResult {
                position: Vec3::new(r.position[0], r.position[1], r.position[2]),
                normal: Vec3::new(r.normal[0], r.normal[1], r.normal[2]),
            })
            .collect()
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
