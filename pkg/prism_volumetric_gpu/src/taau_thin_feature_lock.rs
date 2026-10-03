//! `wgpu` compute twin of the thin-feature lock numeric core inside the
//! temporal-upscale reconstruction contract
//! ([`lock`](prism_render_architecture::temporal_upscale::lock)).
//!
//! The `CPU` golden
//! [`lock`](prism_render_architecture::temporal_upscale::lock) owns the
//! per-pixel thin-feature lifecycle used to protect one-pixel luminance
//! features (a power line, a railing, a specular glint) from the neighborhood
//! clamp that would otherwise eat them. Two of its functions are pure,
//! transcendental-free closed forms over a single pixel and are twinned here:
//!
//! - [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength)
//!   scores how strongly the center pixel is a one-pixel luminance extremum
//!   along both screen axes, from the five linear `RGB` colors of the cross
//!   around it.
//! - [`rejection_scale`](prism_render_architecture::temporal_upscale::lock::rejection_scale)
//!   turns a [`LockState`](prism_render_architecture::temporal_upscale::lock::LockState)
//!   into the fraction of the normal color rejection to keep for a locked
//!   pixel.
//!
//! [`GpuTaauThinFeatureLock`] is the on-device twin of those two closed forms.
//! One thread solves one pixel, emitting both scalars, reproducing the
//! reference's exact arithmetic, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same strength and rejection scale the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a pixel the kernel takes the Rec. 709 luminance of the center and its
//! four axis neighbors (the same
//! [`luminance`](prism_render_architecture::temporal_upscale::color::luminance)
//! weights, inlined), forms the vertical and horizontal extrema, the local luma
//! `range`, the `bright`/`dark` separations, and returns
//! `(separation / range).min(1)` guarded to zero when the cross is flat or the
//! center is not a two-axis extremum — the exact
//! [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength)
//! closed form. It also takes a lock's `locked` flag and `lifetime` and returns
//! `1` when unlocked, else `1 - life * (1 - MIN_SCALE)` with
//! `life = (lifetime / INITIAL_LOCK_LIFETIME).clamp(0, 1)` — the exact
//! [`rejection_scale`](prism_render_architecture::temporal_upscale::lock::rejection_scale)
//! closed form.
//!
//! # What stays on the host
//!
//! The per-frame lock state machine
//! [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
//! — the disocclusion break, the relative-luma staleness break, the one-frame
//! decay and the create/refresh rule — is sequential frame infrastructure with
//! no fixed-width per-pixel device analogue, so it stays on the host and is
//! never dispatched. The degenerate non-finite input guard (a `NaN` local luma
//! range) also stays on the host: the host only enqueues finite cross colors,
//! so the device never sees a `NaN` range and needs no `NaN` test.
//!
//! # Correctness model
//!
//! Both outputs thread through subtracts, a divide and `min`/`max`/`clamp`, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` divide may land a few units in the
//! last place from the scalar reference. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on both scalars, tight
//! enough to catch a genuinely wrong port (a dropped guard, a swapped axis, a
//! wrong weight) yet loose enough to admit a legal last-place divide difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep` and no
//! `round`. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::lock`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` thin-feature lock kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength)
/// and
/// [`rejection_scale`](prism_render_architecture::temporal_upscale::lock::rejection_scale)
/// closed forms; see the module documentation for the algorithm.
const TAAU_THIN_FEATURE_LOCK_WGSL: &str = r#"
// Thin-feature lock twin: one thread computes one pixel's thin-feature strength
// (a two-axis luminance extremum detector over the cross around it) and its
// lock rejection scale, mirroring the CPU golden
// `temporal_upscale::lock` closed forms with only min/max/clamp and + - * /. It
// owns no advance_lock frame state machine; that sequential infrastructure
// stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::lock；无第三方
// 引擎源码或衍生代码。

// Lifetime, in frames, granted to a freshly created lock; the rejection-scale
// normalizer. Mirrors the golden INITIAL_LOCK_LIFETIME.
const INITIAL_LOCK_LIFETIME: f32 = 4.0;
// Minimum fraction of rejection kept at full lock. Mirrors the golden MIN_SCALE.
const MIN_SCALE: f32 = 0.05;

struct Params {
    // Number of pixels in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Linear RGB of the center pixel; w lane is unused padding.
    center: vec4<f32>,
    // Linear RGB of the north (vertical) neighbor; w lane is unused padding.
    north: vec4<f32>,
    // Linear RGB of the south (vertical) neighbor; w lane is unused padding.
    south: vec4<f32>,
    // Linear RGB of the west (horizontal) neighbor; w lane is unused padding.
    west: vec4<f32>,
    // Linear RGB of the east (horizontal) neighbor; w lane is unused padding.
    east: vec4<f32>,
    // 1 when the pixel holds a live lock (golden is_locked), else 0.
    locked: u32,
    // Remaining lock lifetime in frames.
    lifetime: f32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Thin-feature strength in [0, 1].
    strength: f32,
    // Rejection scale in [0, 1]: 1 unlocked, approaching MIN_SCALE at full lock.
    rejection: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Rec. 709 relative luminance of a linear RGB color, mirroring the golden
// `temporal_upscale::color::luminance` weights exactly.
fn luminance(rgb: vec3<f32>) -> f32 {
    return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // --- thin_feature_strength -------------------------------------------
    let c = luminance(q.center.xyz);
    let n = luminance(q.north.xyz);
    let s = luminance(q.south.xyz);
    let w = luminance(q.west.xyz);
    let e = luminance(q.east.xyz);

    let v_max = max(n, s);
    let v_min = min(n, s);
    let h_max = max(w, e);
    let h_min = min(w, e);

    let ring_max = max(v_max, h_max);
    let ring_min = min(v_min, h_min);
    // Full local luma range including the center; guards the normalizer so a
    // flat cross has no feature. The host never enqueues a non-finite range, so
    // only the `<= 0` branch of the golden guard is reachable on-device.
    let range = max(ring_max, c) - min(ring_min, c);

    var strength: f32 = 0.0;
    if (range <= 0.0) {
        strength = 0.0;
    } else {
        // Bright feature: center above both axis maxima. Dark feature: center
        // below both axis minima. The separation is the weaker of the two axes.
        let bright = min(c - v_max, c - h_max);
        let dark = min(v_min - c, h_min - c);
        let separation = max(bright, dark);
        if (separation <= 0.0) {
            strength = 0.0;
        } else {
            strength = min(separation / range, 1.0);
        }
    }

    // --- rejection_scale --------------------------------------------------
    var rejection: f32 = 1.0;
    if (q.locked != 0u) {
        // Normalized remaining life in [0, 1]; more life -> stronger
        // suppression.
        let life = clamp(q.lifetime / INITIAL_LOCK_LIFETIME, 0.0, 1.0);
        rejection = 1.0 - life * (1.0 - MIN_SCALE);
    }

    var out: Result;
    out.strength = strength;
    out.rejection = rejection;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pixel count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_THIN_FEATURE_LOCK_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pixels in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pixel query, matching the `WGSL` `Query`
/// struct: five `16`-byte color lanes (each an `RGB` triple plus an unused pad
/// lane) followed by the lock `locked` flag, `lifetime`, and two pad words to a
/// `96`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear `RGB` of the center pixel, padded to a `16`-byte lane.
    center: [f32; 4],
    /// Linear `RGB` of the north neighbor, padded to a `16`-byte lane.
    north: [f32; 4],
    /// Linear `RGB` of the south neighbor, padded to a `16`-byte lane.
    south: [f32; 4],
    /// Linear `RGB` of the west neighbor, padded to a `16`-byte lane.
    west: [f32; 4],
    /// Linear `RGB` of the east neighbor, padded to a `16`-byte lane.
    east: [f32; 4],
    /// `1` when the pixel holds a live lock, `0` otherwise.
    locked: u32,
    /// Remaining lock lifetime in frames.
    lifetime: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one pixel result, matching the `WGSL` `Result`
/// struct: the thin-feature strength, the rejection scale, and two pad words to
/// a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Thin-feature strength in `[0, 1]`.
    strength: f32,
    /// Rejection scale in `[0, 1]`.
    rejection: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One per-pixel query for the thin-feature lock twin: the five linear `RGB`
/// colors of the cross around the pixel plus the pixel's lock state.
///
/// The host owns the surrounding frame state machine — the
/// [`advance_lock`](prism_render_architecture::temporal_upscale::lock::advance_lock)
/// decay, break and create rules — and enqueues one
/// [`TaauThinFeatureLockQuery`] per pixel to resolve, mirroring the inputs the
/// reference
/// [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength)
/// and
/// [`rejection_scale`](prism_render_architecture::temporal_upscale::lock::rejection_scale)
/// take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauThinFeatureLockQuery {
    /// Linear `RGB` color of the center pixel.
    pub center: [f32; 3],
    /// Linear `RGB` color of the north (vertical) neighbor.
    pub north: [f32; 3],
    /// Linear `RGB` color of the south (vertical) neighbor.
    pub south: [f32; 3],
    /// Linear `RGB` color of the west (horizontal) neighbor.
    pub west: [f32; 3],
    /// Linear `RGB` color of the east (horizontal) neighbor.
    pub east: [f32; 3],
    /// Whether the pixel currently holds a live lock (the golden
    /// [`LockState::is_locked`](prism_render_architecture::temporal_upscale::lock::LockState::is_locked)).
    pub locked: bool,
    /// Remaining lock lifetime in frames (the golden `LockState` `lifetime`).
    pub lifetime: f32,
}

impl TaauThinFeatureLockQuery {
    /// Builds a query from the cross colors and the pixel's lock state.
    #[must_use]
    pub const fn new(
        center: [f32; 3],
        north: [f32; 3],
        south: [f32; 3],
        west: [f32; 3],
        east: [f32; 3],
        locked: bool,
        lifetime: f32,
    ) -> TaauThinFeatureLockQuery {
        TaauThinFeatureLockQuery {
            center,
            north,
            south,
            west,
            east,
            locked,
            lifetime,
        }
    }
}

/// One resolved pixel of the thin-feature lock twin, mirroring the two scalars
/// the reference closed forms produce.
///
/// `strength` is the
/// [`thin_feature_strength`](prism_render_architecture::temporal_upscale::lock::thin_feature_strength)
/// in `[0, 1]`; `rejection` is the
/// [`rejection_scale`](prism_render_architecture::temporal_upscale::lock::rejection_scale)
/// in `[0, 1]` (`1` for an unlocked pixel, approaching `MIN_SCALE` for a freshly
/// created lock).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauThinFeatureLockResult {
    /// Thin-feature strength in `[0, 1]`.
    pub strength: f32,
    /// Rejection scale in `[0, 1]`.
    pub rejection: f32,
}

/// Encodes one [`TaauThinFeatureLockQuery`] into its `std430` [`GpuQuery`] slot,
/// turning the `locked` [`bool`] into a `u32` word and padding each color to a
/// `16`-byte lane.
fn encode_query(q: &TaauThinFeatureLockQuery) -> GpuQuery {
    let pad = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
    GpuQuery {
        center: pad(q.center),
        north: pad(q.north),
        south: pad(q.south),
        west: pad(q.west),
        east: pad(q.east),
        locked: u32::from(q.locked),
        lifetime: q.lifetime,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauThinFeatureLockResult`].
fn decode_result(raw: &GpuResult) -> TaauThinFeatureLockResult {
    TaauThinFeatureLockResult {
        strength: raw.strength,
        rejection: raw.rejection,
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

/// A compiled, reusable thin-feature lock compute pipeline, twinning the numeric
/// core of the `CPU` golden
/// [`lock`](prism_render_architecture::temporal_upscale::lock).
pub struct GpuTaauThinFeatureLock {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauThinFeatureLock {
    /// Compiles the thin-feature lock kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauThinFeatureLock {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock"),
            source: ShaderSource::Wgsl(TAAU_THIN_FEATURE_LOCK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauThinFeatureLock {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every pixel in `queries` and returns one
    /// [`TaauThinFeatureLockResult`] per input, in order.
    ///
    /// Both the strength and the rejection scale match the reference to within
    /// the tolerance documented on this module. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauThinFeatureLockQuery],
    ) -> Vec<TaauThinFeatureLockResult> {
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
            label: Some("prism_volumetric_taau_thin_feature_lock_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_bind_group"),
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
            label: Some("prism_volumetric_taau_thin_feature_lock_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_thin_feature_lock_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_thin_feature_lock_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, flattened to a 1-D dispatch.
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
