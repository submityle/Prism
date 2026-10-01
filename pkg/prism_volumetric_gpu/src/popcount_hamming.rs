//! `wgpu` compute twin of the integer population-count / `Hamming` primitives
//! ([`popcount_hamming`](prism_render_architecture::particle::popcount_hamming)).
//!
//! The `CPU` golden standard owns the bit math: a `SWAR` (SIMD Within A
//! Register) population count
//! ([`popcount_u32`](prism_render_architecture::particle::popcount_hamming::popcount_u32)),
//! the `Hamming` distance as `popcount(a ^ b)`
//! ([`hamming_distance_u32`](prism_render_architecture::particle::popcount_hamming::hamming_distance_u32)),
//! and the folded-`XOR` bit `parity`
//! ([`parity_u32`](prism_render_architecture::particle::popcount_hamming::parity_u32)).
//! [`GpuPopcountHamming`] is the on-device twin that runs one thread per array
//! element and reproduces those values **bit-for-bit**, so a passing real-device
//! parity test is direct evidence the ported kernels implement the identical
//! integer algorithm, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Three per-element kernels share a single `SWAR` `popcount_u32` helper whose
//! magic masks (`0x55555555`, `0x33333333`, `0x0F0F0F0F`) and final multiply by
//! `0x01010101` match the golden word for word:
//!
//! - [`GpuPopcountHamming::popcount`] maps `popcount_u32` over a `u32` storage
//!   array, emitting one set-bit count per element.
//! - [`GpuPopcountHamming::hamming_distance`] maps `popcount_u32(a ^ b)` over two
//!   equal-length `u32` arrays, emitting one differing-bit count per element.
//! - [`GpuPopcountHamming::parity`] maps the folded-`XOR` `parity_u32` over a
//!   `u32` array, emitting `1` for odd popcount and `0` for even.
//!
//! The byte-slice reductions
//! [`hamming_weight_slice`](prism_render_architecture::particle::popcount_hamming::hamming_weight_slice)
//! and
//! [`hamming_distance_slice`](prism_render_architecture::particle::popcount_hamming::hamming_distance_slice)
//! are refactored into their `u32`-array form: the device emits the per-element
//! counts and the whole-slice total is recovered by summing those counts on the
//! host, which pins both the per-element math and the reduction without an
//! on-device atomic.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the bitwise operators
//! `& | ^ << >> ~`, `+ - * /`, and unsigned index arithmetic — with no
//! transcendental call and no optional device feature, so they run unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no `u64` type in `WGSL`, so the `u64`
//! golden variants (`popcount_u64`, `hamming_distance_u64`, `parity_u64`) are
//! intentionally **not** twinned here.
//!
//! # Correctness model
//!
//! Every operation is pure unsigned integer arithmetic with no reordering: the
//! `SWAR` reduction, the `XOR`, and the `parity` fold evaluate the same closed
//! form `WGSL` defines to wrap modulo `2^32`, exactly as the golden's
//! `wrapping_mul` and never-underflowing subtraction do. `CPU` and `GPU` are
//! therefore **bit-exact**; the parity test asserts a precise `==` with no
//! tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! `prism_render_architecture::particle::popcount_hamming` plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

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
/// shared by the other twins in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Uniform parameters for one dispatch: the element count plus three pad words
/// so the struct is a `16`-byte, `16`-byte-aligned uniform. Layout matches
/// `Params` in [`POPCOUNT_HAMMING_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// The portable core-`WGSL` population-count / `Hamming` / `parity` kernels,
/// embedded inline so the twin ships as a single source file. The three entry
/// points `popcount_main`, `hamming_main` and `parity_main` mirror the `CPU`
/// golden
/// [`popcount_u32`](prism_render_architecture::particle::popcount_hamming::popcount_u32),
/// [`hamming_distance_u32`](prism_render_architecture::particle::popcount_hamming::hamming_distance_u32)
/// and
/// [`parity_u32`](prism_render_architecture::particle::popcount_hamming::parity_u32)
/// bit for bit; see the module documentation for the algorithm.
const POPCOUNT_HAMMING_WGSL: &str = r#"
// Population-count / Hamming / parity twin: one thread per array element.
// `popcount_u32` is the classic SWAR parallel-bit-count with masks 0x55555555,
// 0x33333333, 0x0F0F0F0F and a final multiply by 0x01010101; the Hamming
// distance is popcount(a ^ b); parity folds the word with XOR shifts. All math
// is pure unsigned integer arithmetic (WGSL defines u32 ops to wrap modulo
// 2^32, matching the golden's wrapping_mul and never-underflowing subtract), so
// CPU and GPU are bit-exact. The kernels use only the portable core-WGSL subset
// (& | ^ << >> plus + - *) and take no optional feature, so they run unmodified
// on Metal, Vulkan and DX12. There is no u64 in WGSL, so the u64 golden variants
// are not twinned.
//
// Provenance: twinned from this repository's
// prism_render_architecture::particle::popcount_hamming; no third-party engine
// source or derived code.

struct Params {
    // Number of elements to process; threads at or beyond this index return.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
// Second operand for the Hamming-distance kernel only; the popcount and parity
// kernels do not reference it, so their pipeline layout omits this binding.
@group(0) @binding(3) var<storage, read> input_b: array<u32>;

// SWAR population count: identical to the golden `popcount_u32`. The subtraction
// never underflows (the SWAR invariant), and `*` wraps modulo 2^32 exactly as
// the golden's `wrapping_mul`.
fn popcount_u32(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    return ((v * 0x01010101u) >> 24u) & 0x3Fu;
}

// Folded-XOR bit parity: identical shift ladder to the golden `parity_u32`,
// returning 1 for an odd popcount and 0 for an even one.
fn parity_u32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v ^ (v >> 8u);
    v = v ^ (v >> 4u);
    v = v ^ (v >> 2u);
    v = v ^ (v >> 1u);
    return v & 1u;
}

@compute @workgroup_size(64)
fn popcount_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    output[idx] = popcount_u32(input_a[idx]);
}

@compute @workgroup_size(64)
fn hamming_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Hamming distance = popcount of the XOR, exactly as the golden defines it.
    output[idx] = popcount_u32(input_a[idx] ^ input_b[idx]);
}

@compute @workgroup_size(64)
fn parity_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    output[idx] = parity_u32(input_a[idx]);
}
"#;

/// A compiled, reusable population-count / `Hamming` / `parity` pipeline trio.
///
/// All three kernels share one `SWAR` `popcount_u32` helper and one `WGSL`
/// module; the unary kernels (`popcount`, `parity`) bind one input, the binary
/// kernel (`hamming_distance`) binds two.
pub struct GpuPopcountHamming {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    unary_layout: BindGroupLayout,
    binary_layout: BindGroupLayout,
    popcount_pipeline: ComputePipeline,
    hamming_pipeline: ComputePipeline,
    parity_pipeline: ComputePipeline,
}

impl GpuPopcountHamming {
    /// Compiles the population-count, `Hamming`-distance and `parity` kernels on
    /// `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPopcountHamming {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_popcount_hamming"),
            source: ShaderSource::Wgsl(POPCOUNT_HAMMING_WGSL.into()),
        });
        // Unary layout: params + one input + one output. Used by the popcount
        // and parity kernels, which never reference `input_b`.
        let unary_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_popcount_hamming_unary_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        // Binary layout: adds the second input at binding 3 for the Hamming
        // kernel.
        let binary_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_popcount_hamming_binary_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let unary_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_popcount_hamming_unary_pipeline_layout"),
            bind_group_layouts: &[Some(&unary_layout)],
            immediate_size: 0,
        });
        let binary_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_popcount_hamming_binary_pipeline_layout"),
            bind_group_layouts: &[Some(&binary_layout)],
            immediate_size: 0,
        });
        let popcount_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_popcount_hamming_popcount_pipeline"),
            layout: Some(&unary_pipeline_layout),
            module: &module,
            entry_point: Some("popcount_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let hamming_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_popcount_hamming_hamming_pipeline"),
            layout: Some(&binary_pipeline_layout),
            module: &module,
            entry_point: Some("hamming_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let parity_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_popcount_hamming_parity_pipeline"),
            layout: Some(&unary_pipeline_layout),
            module: &module,
            entry_point: Some("parity_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPopcountHamming {
            module,
            unary_layout,
            binary_layout,
            popcount_pipeline,
            hamming_pipeline,
            parity_pipeline,
        }
    }

    /// Returns the per-element `SWAR` population count of `input`, twinning
    /// [`popcount_u32`](prism_render_architecture::particle::popcount_hamming::popcount_u32)
    /// over the array. Summing the result reproduces the whole-slice
    /// [`hamming_weight_slice`](prism_render_architecture::particle::popcount_hamming::hamming_weight_slice).
    ///
    /// An empty input returns an empty vector without issuing a dispatch (a
    /// storage buffer cannot be zero-sized).
    #[must_use]
    pub fn popcount(&self, ctx: &GpuContext, input: &[u32]) -> Vec<u32> {
        self.run_unary(ctx, &self.popcount_pipeline, input)
    }

    /// Returns the per-element bit `parity` of `input` (`1` for an odd
    /// population count, `0` for even), twinning
    /// [`parity_u32`](prism_render_architecture::particle::popcount_hamming::parity_u32)
    /// over the array.
    ///
    /// An empty input returns an empty vector without issuing a dispatch.
    #[must_use]
    pub fn parity(&self, ctx: &GpuContext, input: &[u32]) -> Vec<u32> {
        self.run_unary(ctx, &self.parity_pipeline, input)
    }

    /// Returns the per-element `Hamming` distance `popcount(a ^ b)`, twinning
    /// [`hamming_distance_u32`](prism_render_architecture::particle::popcount_hamming::hamming_distance_u32)
    /// over two equal-length arrays. Summing the result reproduces the
    /// whole-slice
    /// [`hamming_distance_slice`](prism_render_architecture::particle::popcount_hamming::hamming_distance_slice).
    ///
    /// Equal-length inputs return an empty vector without issuing a dispatch;
    /// otherwise the two arrays are processed element for element.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` have different lengths, mirroring the golden
    /// `hamming_distance_slice`, which has no defined distance for mismatched
    /// lengths (it returns `None`).
    #[must_use]
    pub fn hamming_distance(&self, ctx: &GpuContext, a: &[u32], b: &[u32]) -> Vec<u32> {
        assert_eq!(
            a.len(),
            b.len(),
            "Hamming distance requires equal-length inputs"
        );
        if a.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = a.len();
        let bytes = size_of_val(a) as u64;
        let gpu_params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_popcount_hamming_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_a_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_popcount_hamming_input_a"),
            contents: bytemuck::cast_slice(a),
            usage: BufferUsages::STORAGE,
        });
        let input_b_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_popcount_hamming_input_b"),
            contents: bytemuck::cast_slice(b),
            usage: BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_popcount_hamming_output"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_popcount_hamming_stage"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_popcount_hamming_binary_bind_group"),
            layout: &self.binary_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_a_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: input_b_buf.as_entire_binding(),
                },
            ],
        });
        dispatch_and_read(
            ctx,
            &self.hamming_pipeline,
            &bind_group,
            count,
            &output_buf,
            &stage,
        )
    }

    /// Shared helper for the unary (`popcount`, `parity`) kernels: uploads
    /// `input`, dispatches one thread per element and reads the per-element
    /// `u32` result back.
    fn run_unary(&self, ctx: &GpuContext, pipeline: &ComputePipeline, input: &[u32]) -> Vec<u32> {
        if input.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = input.len();
        let bytes = size_of_val(input) as u64;
        let gpu_params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_popcount_hamming_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_popcount_hamming_input_a"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_popcount_hamming_output"),
            size: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_popcount_hamming_stage"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_popcount_hamming_unary_bind_group"),
            layout: &self.unary_layout,
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
        dispatch_and_read(ctx, pipeline, &bind_group, count, &output_buf, &stage)
    }
}

/// Records one 1-D dispatch of `count` threads (`64` per workgroup, `div_ceil`
/// rounding), copies the output into `stage`, and reads it back as a `Vec<u32>`.
fn dispatch_and_read(
    ctx: &GpuContext,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
    count: usize,
    output_buf: &wgpu::Buffer,
    stage: &wgpu::Buffer,
) -> Vec<u32> {
    let device = ctx.device();
    let bytes = (count * size_of::<u32>()) as u64;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism_volumetric_popcount_hamming_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_volumetric_popcount_hamming_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        // One thread per array element, flattened to a 1-D dispatch.
        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(output_buf, 0, stage, 0, bytes);
    ctx.queue().submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();
    debug_assert_eq!(out.len(), count);
    out
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
