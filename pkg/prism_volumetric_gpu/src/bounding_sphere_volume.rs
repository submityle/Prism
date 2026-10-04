//! `wgpu` compute twin of the bounding-sphere volume closed form, from the
//! `CPU` golden `prism_physics_core::collider::bounding_sphere`'s
//! `BoundingSphere::volume`.
//!
//! The volume of a sphere of radius `r` is `V = (4/3) * pi * r^3`, the standard
//! closed form. This module ports that single stateless closed form onto the
//! device: one thread resolves one query, so a passing real-device parity test
//! is direct evidence the ported kernel computes the same volume the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `volume` for one radius:
//!
//! * If the radius is non-finite or `< 0`, the query is invalid
//!   (`valid = 0`, `volume = 0`).
//! * Otherwise `volume = (4.0 / 3.0) * pi * r * r * r`, evaluated in exactly the
//!   golden operator order (the `4/3` ratio, then `pi`, then three multiplies by
//!   `r`).
//!
//! # Correctness model
//!
//! The continuous arithmetic (a constant ratio, a constant multiply and three
//! `r` multiplies) threads through operators a `GPU` may contract, so `CPU` and
//! `GPU` are not necessarily bit-exact; the valid `volume` scalar is compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). For a
//! large radius the cube is large, so the relative arm absorbs the fused
//! multiply-add drift. The discrete `valid` flag is compared exactly; the parity
//! sweep keeps random radii strictly positive and finite so the validity
//! decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite radius or a radius `< 0` yields `valid = 0` with `volume = 0`.
//! The volume itself is a pure product with no division, so there is no divisor
//! to guard; the invalid branch simply forces the output to `0` through a
//! `select`. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and validity with ordered `>= 0`; there is no
//! `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_sphere`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` bounding-sphere volume kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `BoundingSphere::volume`; see the module
/// documentation for the closed form.
const BOUNDING_SPHERE_VOLUME_WGSL: &str = r#"
// Bounding-sphere volume twin: one thread per query reproduces volume. It uses
// only the portable core-WGSL subset (abs, + - * /, select plus unsigned index
// math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and validity an ordered >= 0 compare, both fed to select.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sphere radius.
    radius: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Sphere volume (4/3)*pi*r^3 when valid, else 0.
    volume: f32,
    // 1 when the radius is finite and non-negative, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let radius = q.radius;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false), plus non-negativity. No bare f32 equality
    // anywhere.
    let finite = abs(radius) < FINITE_LIMIT;
    let non_negative = radius >= 0.0;
    let ok = finite && non_negative;

    // Golden operator order: (4/3) * pi, then three multiplies by r. Pure
    // product, so no divisor to guard; the invalid branch forces 0 via select.
    let vol = (4.0 / 3.0) * PI * radius * radius * radius;

    var out: Result;
    out.volume = select(0.0, vol, ok);
    out.valid = select(0u, 1u, ok);
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// The radius is padded to `4` `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    radius: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the volume and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    volume: f32,
    valid: u32,
}

/// One bounding-sphere volume query: the sphere radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphereVolumeQuery {
    /// Sphere radius.
    pub radius: f32,
}

impl BoundingSphereVolumeQuery {
    /// Builds a query from the sphere radius.
    #[must_use]
    pub fn new(radius: f32) -> BoundingSphereVolumeQuery {
        BoundingSphereVolumeQuery { radius }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `BoundingSphere::volume` output for that radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphereVolumeResult {
    /// The sphere volume `(4/3) * pi * r^3` when valid, else `0`.
    pub volume: f32,
    /// `1` when the radius is finite and non-negative, else `0`.
    pub valid: u32,
}

/// Encodes one [`BoundingSphereVolumeQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &BoundingSphereVolumeQuery) -> GpuQuery {
    GpuQuery {
        radius: q.radius,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BoundingSphereVolumeResult`].
fn decode_result(raw: &GpuResult) -> BoundingSphereVolumeResult {
    BoundingSphereVolumeResult {
        volume: raw.volume,
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

/// A compiled, reusable bounding-sphere volume compute pipeline, twinning the
/// `CPU` golden `BoundingSphere::volume`.
pub struct GpuBoundingSphereVolume {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBoundingSphereVolume {
    /// Compiles the bounding-sphere volume kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBoundingSphereVolume {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume"),
            source: ShaderSource::Wgsl(BOUNDING_SPHERE_VOLUME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBoundingSphereVolume {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BoundingSphereVolumeResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `volume` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BoundingSphereVolumeQuery],
    ) -> Vec<BoundingSphereVolumeResult> {
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
            label: Some("prism_volumetric_bounding_sphere_volume_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_bind_group"),
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
            label: Some("prism_volumetric_bounding_sphere_volume_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bounding_sphere_volume_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bounding_sphere_volume_pass"),
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
