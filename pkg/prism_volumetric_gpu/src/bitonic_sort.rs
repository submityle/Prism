//! `wgpu` compute twin of the Batcher bitonic sorting-network `u32` sort golden
//! ([`BitonicSort`](prism_render_architecture::particle::bitonic_sort::BitonicSort),
//! particle design §12 sort ordering).
//!
//! The §12 strategy matrix routes small per-emitter sorts to a bitonic network
//! (the companion radix count-sort handles large runs). The `CPU` golden
//! [`BitonicSort`](prism_render_architecture::particle::bitonic_sort::BitonicSort)
//! owns the fixed compare-exchange schedule — the stage/pass count, the
//! per-invocation partner index and the per-element sort direction — and
//! [`sort_keys`](prism_render_architecture::particle::bitonic_sort::BitonicSort::sort_keys)
//! replays that schedule serially on the `CPU`. [`GpuBitonicSortU32`] is the
//! on-device twin: it uploads the keys once, then replays the *same* network as
//! a sequence of compare-exchange dispatches (one per `(stage, pass)` step),
//! one thread per padded element, so a passing real-device parity test is
//! direct evidence the ported kernel reproduces the golden network topology and
//! direction bit for bit, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! A bitonic network sorts a power-of-two element count in
//! `stage * (stage + 1) / 2` compare-exchange passes. The host rounds the live
//! key count up to the next power of two ([`BitonicSort::padded_count`]), pads
//! the key buffer to [`MAX_N`] slots with the sentinel `u32::MAX`, and walks the
//! golden [`steps`](prism_render_architecture::particle::bitonic_sort::BitonicSort::steps):
//! each [`CompareStep`](prism_render_architecture::particle::bitonic_sort::CompareStep)
//! contributes one dispatch carrying its `XOR` compare distance
//! (`1 << (stage - net_pass)`) and box size (`1 << stage`). In each dispatch,
//! thread `i` pairs with `partner = i ^ distance`; only the lower index of the
//! pair acts, the box bit (`i & box_size`) selects the ascending or descending
//! comparison, and the pair swaps when out of order — the exact `partner <= i`
//! guard, box-bit direction and `out_of_order` test the golden
//! [`sort_keys`](prism_render_architecture::particle::bitonic_sort::BitonicSort::sort_keys)
//! performs. Padding with the maximum key keeps the padded tail sorted to the
//! end, so an ascending network leaves the real keys ordered at the front. A
//! final pack dispatch writes the live `count` and the full [`MAX_N`] slot block
//! into [`GpuBitonicSort`], the sentinel tail intact.
//!
//! # Correctness model
//!
//! Every value on the path is a `u32`: partner indices are shifts and `XOR`s,
//! the comparison is an integer `>` or `<`, and the exchange is a swap. There is
//! no floating point, so the parity test asserts **exact per-element equality**
//! on the live sorted prefix; any mismatch is a genuine port bug (a wrong
//! distance, a flipped direction, a dropped swap) rather than a rounding
//! artifact.
//!
//! # Degenerate inputs
//!
//! An empty key array is short-circuited on the host — a storage buffer may not
//! be zero-sized — and returns the all-sentinel [`GpuBitonicSort`] with
//! [`count`](GpuBitonicSort::count) `= 0`. A single-element array has an empty
//! step schedule (no network passes) and only the pack dispatch runs, matching
//! the reference no-op for lengths below two. The host rejects an array longer
//! than [`MAX_N`].
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer shifts, masks,
//! `XOR`s and comparisons — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! inverse trigonometry, no `sqrt`, no `u64` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Successive dispatches in
//! the single compute pass observe each other's writes to the shared key buffer
//! through the implicit storage barrier `WebGPU` inserts between dispatches.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::bitonic_sort::BitonicSort;
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
/// that divides evenly across `Metal`, `Vulkan` and `DX12`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of `u32` keys one bitonic sort handles. The key buffer travels
/// to the device as a fixed `std430` slot block of this width (`count` live keys
/// plus the `u32::MAX` sentinel padding), matching the `MAX_N` constant baked
/// into the shader. It is a power of two, so [`BitonicSort::padded_count`] of any
/// in-range key count never exceeds it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
pub const MAX_N: usize = 1024;

/// Core-`WGSL` bitonic compare-exchange kernel, inlined so this twin lives
/// entirely in the crate with no external `.wesl`. The `bitonic_step_main` entry
/// runs one compare-exchange pass over the shared key buffer (one thread per
/// padded element), and the `bitonic_pack_main` entry writes the live `count`
/// and the full `MAX_N` slot block into the output, mirroring the `CPU` golden
/// `particle::bitonic_sort::BitonicSort::sort_keys`.
///
/// The WGSL reserved word `pass` is avoided; the per-stage pass index is called
/// `net_pass` wherever it appears.
const BITONIC_SORT_WGSL: &str = r#"
// Bitonic sort u32 twin. The key buffer is a fixed MAX_N-slot block: `count`
// live keys at the front followed by the u32::MAX sentinel padding. The host
// replays the Batcher network as a sequence of compare-exchange dispatches, one
// per (stage, net_pass) step, each carrying its XOR compare distance and box
// size. A final pack dispatch copies the sorted block into the output. Mirrors
// the CPU golden `particle::bitonic_sort::BitonicSort::sort_keys`.

// Fixed per-sort slot width; mirrors the host MAX_N.
const MAX_N: u32 = 1024u;
// Padding sentinel: the maximum u32, so the padded tail sorts to the end.
const SENTINEL: u32 = 4294967295u;

// One compare-exchange step as a push-constant-sized uniform. `distance` and
// `box_size` describe the current (stage, net_pass); `count` is consumed only by
// the pack entry.
struct Params {
    distance: u32,
    box_size: u32,
    padded: u32,
    count: u32,
}

// The packed output: the live key count and the full MAX_N slot block (sorted
// prefix plus the u32::MAX sentinel tail).
struct BitonicOut {
    count: u32,
    sorted: array<u32, 1024>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: BitonicOut;

// One compare-exchange pass: thread `i` pairs with `partner = i ^ distance`.
// Only the lower index of each pair acts, so every pair is touched once (the
// golden `partner <= i` guard, which also turns a zero-distance self-pair into a
// no-op). The box bit `i & box_size` selects ascending vs descending, exactly
// the golden `CompareStep::sort_ascending`, and the swap fires on the golden
// `out_of_order` test.
@compute @workgroup_size(64)
fn bitonic_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.padded) {
        return;
    }
    let partner = i ^ params.distance;
    if (partner <= i) {
        return;
    }
    let ascending = (i & params.box_size) == 0u;
    let a = data[i];
    let b = data[partner];
    var out_of_order = false;
    if (ascending) {
        out_of_order = a > b;
    } else {
        out_of_order = a < b;
    }
    if (out_of_order) {
        data[i] = b;
        data[partner] = a;
    }
}

// Pack entry: write the live count (once, from lane 0) and the full MAX_N slot
// block. Slots in [0, padded) copy the sorted network buffer; the tail up to
// MAX_N is the sentinel (already true for [count, padded) after the sort).
@compute @workgroup_size(64)
fn bitonic_pack_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= MAX_N) {
        return;
    }
    if (i == 0u) {
        result.count = params.count;
    }
    if (i < params.padded) {
        result.sorted[i] = data[i];
    } else {
        result.sorted[i] = SENTINEL;
    }
}
"#;

/// Uniform parameters for one dispatch. `16`-byte `repr(C)` matching `Params` in
/// the inlined shader: the compare distance and box size of the current
/// `(stage, net_pass)` step, the padded element count and the live key count.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    distance: u32,
    box_size: u32,
    padded: u32,
    count: u32,
}

/// One sorted array as the device returns it, mirroring `BitonicOut` in the
/// inlined shader: the live key `count` followed by the full [`MAX_N`] slot block
/// (the ascending sorted prefix plus the `u32::MAX` sentinel tail).
///
/// `repr(C)` with no padding (`4 + 4 * MAX_N` bytes, `4`-byte aligned), so it is
/// `Pod` and reads back directly without a decode step. Only the first `count`
/// slots are live keys; everything from `count` to [`MAX_N`] is the sentinel.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuBitonicSort {
    /// Number of live keys written into the ascending [`sorted`](Self::sorted)
    /// prefix.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`。
    pub count: u32,
    /// The full [`MAX_N`] slot block: the ascending sorted keys in
    /// `[0, count)`, the `u32::MAX` sentinel padding thereafter.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`。
    pub sorted: [u32; MAX_N],
}

impl GpuBitonicSort {
    /// The empty-input result: `count` `= 0` and every slot the `u32::MAX`
    /// sentinel, matching the all-padding slot block the dispatch path leaves
    /// for a zero-length sort.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`。
    #[must_use]
    fn empty() -> GpuBitonicSort {
        GpuBitonicSort {
            count: 0,
            sorted: [u32::MAX; MAX_N],
        }
    }

    /// The live ascending key prefix (`sorted[0..count]`), dropping the sentinel
    /// padding tail.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`。
    #[must_use]
    pub fn sorted_keys(&self) -> &[u32] {
        &self.sorted[..self.count as usize]
    }
}

/// Builds a compute-visible buffer binding layout entry.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
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

/// A compiled, reusable bitonic `u32`-sort compute pipeline, twinning the `CPU`
/// golden
/// [`BitonicSort`](prism_render_architecture::particle::bitonic_sort::BitonicSort).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
pub struct GpuBitonicSortU32 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    step_pipeline: ComputePipeline,
    pack_pipeline: ComputePipeline,
}

impl GpuBitonicSortU32 {
    /// Compiles the bitonic `u32`-sort kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBitonicSortU32 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bitonic_sort_shader"),
            source: ShaderSource::Wgsl(BITONIC_SORT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bitonic_sort_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bitonic_sort_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let step_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bitonic_sort_step_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bitonic_step_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pack_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bitonic_sort_pack_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bitonic_pack_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBitonicSortU32 {
            module,
            layout,
            step_pipeline,
            pack_pipeline,
        }
    }

    /// Sorts `keys` ascending on-device and returns them in one
    /// [`GpuBitonicSort`].
    ///
    /// The [`sorted`](GpuBitonicSort::sorted) prefix equals the `CPU` golden
    /// [`sort_keys`](prism_render_architecture::particle::bitonic_sort::BitonicSort::sort_keys)
    /// element for element, with the `u32::MAX` sentinel filling the slots from
    /// [`count`](GpuBitonicSort::count) up to [`MAX_N`]. An empty `keys` slice
    /// issues **no dispatch** — a storage buffer may not be zero-sized — and
    /// returns the all-sentinel [`GpuBitonicSort`] with
    /// [`count`](GpuBitonicSort::count) `= 0`.
    ///
    /// # Panics
    ///
    /// Panics if `keys.len()` exceeds [`MAX_N`]; the fixed-length device buffer
    /// cannot hold a longer array.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bitonic_sort`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, keys: &[u32]) -> GpuBitonicSort {
        let count = keys.len();
        if count == 0 {
            // No key to dispatch, and a storage buffer cannot be zero-sized, so
            // return the all-sentinel result directly.
            return GpuBitonicSort::empty();
        }
        assert!(
            count <= MAX_N,
            "bitonic_sort twin handles at most MAX_N keys"
        );
        let device = ctx.device();

        // The golden descriptor supplies the exact network: padded power-of-two
        // size and the ordered compare-exchange steps the GPU replays.
        let sorter = BitonicSort::new(count);
        let padded = sorter.padded_count();
        debug_assert!(padded <= MAX_N, "padded count must fit the slot block");
        let steps = sorter.steps();
        let padded_u = padded as u32;
        let count_u = count as u32;

        // Key buffer: a fixed MAX_N slot block, live keys at the front and the
        // u32::MAX sentinel everywhere else, so the padded tail sorts to the end.
        let mut data_init = [u32::MAX; MAX_N];
        data_init[..count].copy_from_slice(keys);
        let data_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bitonic_sort_data"),
            contents: bytemuck::cast_slice(&data_init),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of::<GpuBitonicSort>() as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bitonic_sort_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // One uniform buffer and bind group per compare-exchange step, plus one
        // for the final pack. Kept alive in these vectors until after submit.
        let mut param_bufs = Vec::with_capacity(steps.len() + 1);
        let mut bind_groups = Vec::with_capacity(steps.len() + 1);
        let make_bind_group = |params: &GpuParams| {
            let params_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("prism_volumetric_bitonic_sort_params"),
                contents: bytemuck::bytes_of(params),
                usage: BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_volumetric_bitonic_sort_bind_group"),
                layout: &self.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: data_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });
            (params_buf, bind_group)
        };

        for step in &steps {
            let params = GpuParams {
                distance: step.compare_distance() as u32,
                box_size: step.box_size() as u32,
                padded: padded_u,
                count: count_u,
            };
            let (params_buf, bind_group) = make_bind_group(&params);
            param_bufs.push(params_buf);
            bind_groups.push(bind_group);
        }
        // The pack step ignores distance/box size; it only needs padded + count.
        let pack_params = GpuParams {
            distance: 0,
            box_size: 0,
            padded: padded_u,
            count: count_u,
        };
        let (pack_buf, pack_bind_group) = make_bind_group(&pack_params);
        param_bufs.push(pack_buf);

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bitonic_sort_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bitonic_sort_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bitonic_sort_pass"),
                timestamp_writes: None,
            });
            // One thread per padded element, flattened to a 1-D dispatch. Each
            // step dispatch observes the previous step's writes to the shared
            // key buffer through WebGPU's implicit between-dispatch barrier.
            let step_groups = padded_u.div_ceil(WORKGROUP_SIZE);
            pass.set_pipeline(&self.step_pipeline);
            for bind_group in &bind_groups {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(step_groups, 1, 1);
            }
            // Pack the sorted block (one thread per MAX_N slot) into the output.
            let pack_groups = (MAX_N as u32).div_ceil(WORKGROUP_SIZE);
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_bind_group, &[]);
            pass.dispatch_workgroups(pack_groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::pod_read_unaligned::<GpuBitonicSort>(&view);
        drop(view);
        stage.unmap();
        raw
    }
}
