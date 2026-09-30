//! `wgpu` compute twin of Prism's ray-traced-reflection role decision
//! ([`resolve_rt_role`](prism_render_architecture::hair::rt_proxy::resolve_rt_role)).
//!
//! Tracing individual strands in reflections is prohibitively expensive, so a
//! groom is either registered as a cheap LOD proxy, traced as real strands (a
//! hero-shot luxury) or excluded from reflections entirely. That policy choice
//! is a pure, per-instance function of the groom's LOD tier, its screen coverage
//! and a shared [`RtProxyPolicy`], with no cross-instance dependency, so a scene
//! of many groom instances classifies one `GPU` thread per instance. The `CPU`
//! golden is
//! [`resolve_rt_role`](prism_render_architecture::hair::rt_proxy::resolve_rt_role);
//! this crate is the on-device twin that runs the identical two-gate decision so
//! a passing real-device parity test is direct evidence the ported kernel
//! resolves the same role as the reference — not merely that its shader
//! compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRtProxy::eval`] takes a shared [`RtProxyPolicy`] and a batch of
//! per-groom [`RtProxyQuery`] (LOD tier plus screen coverage) and returns one
//! [`RtReflectionRole`] per groom, applying the reference's visibility gate
//! (coverage below `min_coverage_for_proxy` is excluded) then representation
//! gate (strand-based tiers trace real strands only when the policy opts in;
//! everything else is a proxy).
//!
//! # Portability
//!
//! The kernel uses only integer arithmetic and one float compare in the
//! portable core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The path has no transcendental and no fused multiply-add — the only float
//! operation is the gate comparison — so the `CPU` and `GPU` produce the
//! identical role code for every finite-coverage groom. The parity test
//! therefore asserts exact equality rather than a tolerance. A non-finite
//! (`NaN`) coverage degrades to [`RtReflectionRole::Excluded`] in the reference;
//! the kernel implements the same guard but the parity test exercises only
//! finite coverages, since a driver's fast-math mode may make on-device `NaN`
//! handling nondeterministic (the reference's own `NaN` case is covered by its
//! unit tests).
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard LOD-driven RT reflection proxy/exclusion policy plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::rt_proxy::{resolve_rt_role, RtProxyPolicy, RtReflectionRole};
use prism_render_architecture::hair::HairLodTier;

use crate::context::GpuContext;

/// Uniform parameters for one role dispatch. Layout matches `Params` in
/// `shaders/rt_proxy.wesl`: the groom count bounding the dispatch, the
/// `allow_strands` flag (`0`/`1`) and the coverage gate, padded to one
/// `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    query_count: u32,
    allow_strands: u32,
    min_coverage_for_proxy: f32,
    pad: u32,
}

/// One groom instance to classify: its LOD tier and screen coverage.
///
/// The tier is carried as its [`HairLodTier`] code (`0`=Strands,
/// `1`=`ReducedStrands`, `2`=Cards, `3`=Mesh) so the layout is a plain pair of
/// 32-bit words matching `Query` in `shaders/rt_proxy.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Query {
    tier: u32,
    coverage: f32,
}

/// Maps a [`HairLodTier`] to the `u32` tier code the kernel and reference share.
#[must_use]
pub fn tier_code(tier: HairLodTier) -> u32 {
    match tier {
        HairLodTier::Strands => 0,
        HairLodTier::ReducedStrands => 1,
        HairLodTier::Cards => 2,
        HairLodTier::Mesh => 3,
    }
}

/// Maps an [`RtReflectionRole`] to the `u32` role code the kernel emits.
#[must_use]
pub fn role_code(role: RtReflectionRole) -> u32 {
    match role {
        RtReflectionRole::FullStrands => 0,
        RtReflectionRole::Proxy => 1,
        RtReflectionRole::Excluded => 2,
    }
}

/// Decodes a kernel-emitted `u32` role code back into an [`RtReflectionRole`].
///
/// Any value other than `0`/`1` decodes to [`RtReflectionRole::Excluded`], the
/// safe "does not enter the BVH" default; the kernel only ever writes `0`, `1`
/// or `2`.
#[must_use]
pub fn role_from_code(code: u32) -> RtReflectionRole {
    match code {
        0 => RtReflectionRole::FullStrands,
        1 => RtReflectionRole::Proxy,
        _ => RtReflectionRole::Excluded,
    }
}

/// The `CPU` reference: resolves the RT reflection role for one groom exactly as
/// the kernel does, delegating to the golden
/// [`resolve_rt_role`](prism_render_architecture::hair::rt_proxy::resolve_rt_role).
#[must_use]
pub fn reference_rt_role(
    tier: HairLodTier,
    coverage: f32,
    policy: RtProxyPolicy,
) -> RtReflectionRole {
    resolve_rt_role(tier, coverage, policy)
}

/// A compiled, reusable per-groom RT-reflection-role pipeline.
pub struct GpuHairRtProxy {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRtProxy {
    /// Compiles the per-groom role kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRtProxy {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rt_proxy"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rt_proxy.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rt_proxy_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rt_proxy_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rt_proxy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRtProxy {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the RT reflection role for every groom in `queries` under the
    /// shared `policy`, returning one [`RtReflectionRole`] per groom in order.
    ///
    /// Each `(tier, coverage)` maps to the same role the `CPU` golden
    /// [`resolve_rt_role`](prism_render_architecture::hair::rt_proxy::resolve_rt_role)
    /// yields for the same inputs — the path is transcendental-free and fma-free,
    /// so the twin is bit-identical, not merely close. An empty query batch
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        policy: RtProxyPolicy,
        queries: &[(HairLodTier, f32)],
    ) -> Vec<RtReflectionRole> {
        if queries.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            query_count: queries.len() as u32,
            allow_strands: u32::from(policy.allow_strands_in_rt),
            min_coverage_for_proxy: policy.min_coverage_for_proxy,
            pad: 0,
        };

        let gpu_queries: Vec<Query> = queries
            .iter()
            .map(|&(tier, coverage)| Query {
                tier: tier_code(tier),
                coverage,
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_proxy_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let query_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_proxy_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_proxy_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_proxy_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rt_proxy_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: query_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_rt_proxy_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rt_proxy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let codes = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        codes.into_iter().map(role_from_code).collect()
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
