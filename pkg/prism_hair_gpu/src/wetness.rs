//! `wgpu` compute twin of Prism's wet-hair saturation coupling
//! ([`wet_hair_response`](prism_render_architecture::hair::wetness::wet_hair_response),
//! batch form
//! [`wet_hair_response_map`](prism_render_architecture::hair::wetness::wet_hair_response_map)).
//!
//! When hair takes on water the change is physically coherent and driven by a
//! single scalar, the water saturation `w` in `[0, 1]`. From that one quantity
//! the groom pipeline derives five parameter *modifiers* that make wet hair
//! behave and read differently from dry hair: the clump radius tightens, mass
//! and damping rise, pigment absorption deepens (darker wet hair) and the
//! apparent roughness drops (the wet sheen). Each modifier is a straight linear
//! interpolation between the dry endpoint (`w = 0`) and the wet endpoint
//! (`w = 1`), so the whole map is pure multiply/add with no transcendental
//! term — exactly golden-comparable, array in, array out, panic-free.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairWetness::eval`] takes a batch of wetness fractions and returns, for
//! each one, the five modifiers packed as five consecutive `f32`s in the golden
//! [`WetHairResponse`](prism_render_architecture::hair::wetness::WetHairResponse)
//! field order (`clump_scale`, `mass_mul`, `damping_mul`, `sigma_a_mul`,
//! `roughness_delta`), preserving input order. The element index is the
//! invocation id (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the element count early-return.
//!
//! # The endpoints live on the host, the saturation varies per thread
//!
//! The five wet endpoints (`CLUMP_SCALE_WET`, `MASS_MUL_WET`, `DAMPING_MUL_WET`,
//! `SIGMA_A_MUL_WET`, `ROUGHNESS_DELTA_WET`) are passed as uniform parameters so
//! the shader never duplicates the golden's magic constants and cannot drift
//! from them; the per-thread variation is the saturation read from the storage
//! input. The wetness is sanitized in-shader bit-faithfully to the golden's
//! `sanitize_wetness`, so the host uploads the raw authored value unchanged.
//!
//! # Portability
//!
//! The kernel uses only multiply/add and `clamp` — no `exp`, `pow`, `sin` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! Each modifier is a single multiply-add a `GPU` may fuse, so `CPU` and `GPU`
//! agree to within the documented fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than bit-for-bit. The saturation guard mirrors the
//! golden's `sanitize_wetness` exactly: the finite test (`w == w` rejects `NaN`,
//! the finite-magnitude bound rejects `+/-inf`) collapses non-finite inputs to
//! fully dry, and finite out-of-range inputs clamp into `[0, 1]`, so negative,
//! greater-than-one and non-finite inputs produce the same bounded modifiers
//! and the result stays finite for every input.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own deterministic wet-hair coupling plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::wetness::{
    wet_hair_response, WetHairResponse, CLUMP_SCALE_WET, DAMPING_MUL_WET, MASS_MUL_WET,
    ROUGHNESS_DELTA_WET, SIGMA_A_MUL_WET,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of `f32` modifiers emitted per wetness element, matching the five
/// fields of the golden `WetHairResponse`.
pub const MODIFIERS_PER_ELEMENT: usize = 5;

/// Uniform parameters for one wetness dispatch. Layout matches `Params` in
/// `shaders/wetness.wesl`: the four multiplicative wet endpoints packed as one
/// `16`-byte `vec4`, then the additive roughness-delta endpoint and the element
/// count padded out to a second `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    wet: [f32; 4],
    roughness_delta_wet: f32,
    element_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-element wet-hair coupling pipeline.
pub struct GpuHairWetness {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairWetness {
    /// Compiles the per-element wet-hair coupling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairWetness {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_wetness"),
            source: ShaderSource::Wgsl(include_str!("../shaders/wetness.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_wetness_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_wetness_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_wetness_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairWetness {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each wetness fraction to its five response modifiers, returning them
    /// flattened five `f32`s per element in input order (field order
    /// `clump_scale`, `mass_mul`, `damping_mul`, `sigma_a_mul`,
    /// `roughness_delta`).
    ///
    /// The five values for element `i` equal the fields of the `CPU` golden
    /// [`wet_hair_response`](prism_render_architecture::hair::wetness::wet_hair_response)
    /// of `wetness[i]` to within the module's documented fma tolerance, with
    /// negative, greater-than-one and non-finite inputs collapsing to the same
    /// clamped modifiers. An empty batch yields an empty vector without a
    /// dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, wetness: &[f32]) -> Vec<f32> {
        let element_count = wetness.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            wet: [
                CLUMP_SCALE_WET,
                MASS_MUL_WET,
                DAMPING_MUL_WET,
                SIGMA_A_MUL_WET,
            ],
            roughness_delta_wet: ROUGHNESS_DELTA_WET,
            element_count: element_count as u32,
            pad0: 0,
            pad1: 0,
        };

        // Output is five f32s (one WetHairResponse worth) per element.
        let out_len = element_count * MODIFIERS_PER_ELEMENT;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_wetness_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let wetness_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_wetness_input"),
            contents: bytemuck::cast_slice(wetness),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_wetness_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_wetness_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_wetness_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wetness_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_wetness_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_wetness_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (element_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden response for one wetness fraction, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_response(wetness: f32) -> WetHairResponse {
    wet_hair_response(wetness)
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
