//! `wgpu` compute twin of the deterministic least-significant-`digit` (`LSD`)
//! `u32` `radix` sort
//! ([`radix_sort_u32`](prism_render_architecture::particle::radix_sort_u32),
//! particle design §12 sort ordering).
//!
//! A production `GPU` VFX stack sorts millions of quantized view-depth keys per
//! frame with a multi-pass `LSD` `radix` sort. The `CPU` golden
//! [`radix_sort_u32`](prism_render_architecture::particle::radix_sort_u32) owns
//! the serial reference of that scheme: four stable 8-bit counting passes
//! (bytes `0`, `1`, `2`, `3` from least to most significant) that sort the whole
//! `u32` key space, exposed as
//! [`sort`](prism_render_architecture::particle::radix_sort_u32::sort) (a sorted
//! copy of the keys) and
//! [`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort) (a
//! stable permutation of indices). [`GpuRadixSortU32`] is the on-device twin:
//! one thread runs the entire serial sort of one array (up to [`MAX_N`] keys),
//! so a passing real-device parity test is direct evidence the ported kernel
//! produces the same ordering and the same stable permutation the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Both golden entry points are reproduced in a single dispatch. The kernel
//! carries each key together with its original index through the four counting
//! passes, so after the final pass the key lane holds exactly the
//! [`sort`](prism_render_architecture::particle::radix_sort_u32::sort) output and
//! the index lane holds exactly the
//! [`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort)
//! permutation. Each pass counts digit occurrences into a fixed `256`-bucket
//! `histogram`, turns the `histogram` into per-bucket start offsets with an
//! in-place exclusive prefix sum, then scatters every `(key, index)` pair front
//! to back into monotonically advancing bucket cursors — the exact counting,
//! prefix sum and scatter the golden performs, so stability (equal keys keep
//! their original order) is reproduced bit for bit.
//!
//! # Correctness model
//!
//! Every value on the path is a `u32`: `digit` extraction is a shift and a
//! mask, counting and prefix sum are integer additions bounded by the key count,
//! and the scatter is an index copy. There is no floating point, so the parity
//! test asserts **exact per-element equality** on both the sorted keys and the
//! `argsort` permutation; any mismatch is a genuine port bug (a dropped
//! increment, a wrong `digit`, a broken cursor) rather than a rounding artifact.
//!
//! # Degenerate inputs
//!
//! An empty key array is short-circuited on the host — a storage buffer may not
//! be zero-sized — and returns the all-zero [`GpuRadixSort`] with
//! [`count`](GpuRadixSort::count) `= 0`. A single-element array passes through
//! the four passes unchanged, matching the reference no-op for lengths below
//! two. The host rejects an array longer than [`MAX_N`].
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer shifts, masks
//! and adds plus bounded `for` loops over fixed-length `array` buffers — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, no inverse trigonometry, no `sqrt`, no
//! `u64` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every loop bound is a compile-time `const` or the
//! host-bounded key count, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
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
/// used across this crate's compute kernels; only the first lane does work here.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Number of independent sort jobs issued per dispatch: the twin sorts one array
/// per call, so a single thread runs the whole serial sort.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
const JOBS: u32 = 1;

/// Maximum number of keys one sort job handles, matching the `WGSL` `MAX_N`
/// fixed-length working buffers.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
pub const MAX_N: usize = 64;

/// The portable core-`WGSL` `radix`-sort kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `radix_sort_u32_main`
/// mirrors the `CPU` golden
/// [`radix_sort_u32`](prism_render_architecture::particle::radix_sort_u32) pass
/// for pass; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
const RADIX_SORT_U32_WGSL: &str = r#"
// Radix-sort u32 twin: a single thread runs the full four-pass LSD radix sort of
// one u32 array (up to MAX_N = 64 keys), reproducing the CPU golden
// `particle::radix_sort_u32::sort` and `::argsort` bit for bit. Each pass is a
// stable 8-bit counting sort over a 256-bucket histogram: count digit
// occurrences, turn the histogram into exclusive start offsets with a prefix
// sum, then scatter every (key, index) pair front to back into monotonically
// advancing bucket cursors. Because the scatter walks the working buffer in
// order and each cursor only advances, equal keys keep their original order, so
// the pass is stable and the four least-significant-first passes sort the whole
// key space. Carrying the source index alongside each key yields the stable
// argsort permutation in the same passes. Pure integer arithmetic: shifts, masks
// and adds only. No u64, no transcendental, no sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::radix_sort_u32；无第三方
// 引擎源码或衍生代码。

// Maximum key count a single sort job handles; matches the host MAX_N.
const MAX_N: u32 = 64u;
// Number of key bits consumed per counting pass.
const DIGIT_BITS: u32 = 8u;
// Histogram buckets in one pass (1u << DIGIT_BITS = 256).
const BUCKETS: u32 = 256u;
// Bit mask isolating one 8-bit digit (0xFF).
const DIGIT_MASK: u32 = 255u;
// Passes needed to consume every bit of a u32 key (32u / DIGIT_BITS = 4).
const PASSES: u32 = 4u;

struct Params {
    // Number of keys in the input array; in [1, MAX_N] whenever dispatched.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct SortResult {
    // Number of sorted keys written (equals params.count).
    count: u32,
    // Ascending keys in lanes [0, count); lanes [count, MAX_N) left zero.
    sorted: array<u32, MAX_N>,
    // Stable sort permutation in lanes [0, count); lanes [count, MAX_N) zero.
    argsort: array<u32, MAX_N>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: SortResult;

// Extract the `byte_index`-th 8-bit digit of `key`, least significant first; mirrors
// the reference `digit_of`.
fn digit_of(key: u32, byte_index: u32) -> u32 {
    return (key >> (byte_index * DIGIT_BITS)) & DIGIT_MASK;
}

@compute @workgroup_size(64)
fn radix_sort_u32_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // A single thread performs the whole serial sort; extra lanes short-circuit.
    if (gid.x >= 1u) {
        return;
    }
    let count = params.count;

    // Working buffers: `src` holds the current order, `dst` receives each
    // scattered pass. Indices ride along to build the stable argsort permutation
    // in the same passes.
    var src_key: array<u32, MAX_N>;
    var src_idx: array<u32, MAX_N>;
    var dst_key: array<u32, MAX_N>;
    var dst_idx: array<u32, MAX_N>;

    // Seed the working buffers with the input keys and the identity permutation.
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        src_key[i] = keys[i];
        src_idx[i] = i;
    }

    for (var byte_index: u32 = 0u; byte_index < PASSES; byte_index = byte_index + 1u) {
        // Zero the 256-bucket histogram for this pass.
        var counts: array<u32, BUCKETS>;
        for (var b: u32 = 0u; b < BUCKETS; b = b + 1u) {
            counts[b] = 0u;
        }
        // Count digit occurrences across the current order.
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let d = digit_of(src_key[i], byte_index);
            counts[d] = counts[d] + 1u;
        }
        // Exclusive prefix sum: counts[b] becomes bucket b's start offset.
        var running: u32 = 0u;
        for (var b: u32 = 0u; b < BUCKETS; b = b + 1u) {
            let here = counts[b];
            counts[b] = running;
            running = running + here;
        }
        // Stable scatter front to back into monotonically advancing cursors.
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            let bucket = digit_of(src_key[i], byte_index);
            let slot = counts[bucket];
            dst_key[slot] = src_key[i];
            dst_idx[slot] = src_idx[i];
            counts[bucket] = slot + 1u;
        }
        // Copy the scattered pass back into `src` for the next pass. PASSES is
        // even (4), so the fully sorted data lands in `src` after the last pass.
        for (var i: u32 = 0u; i < count; i = i + 1u) {
            src_key[i] = dst_key[i];
            src_idx[i] = dst_idx[i];
        }
    }

    // Write the result, explicitly zeroing lanes past `count` so the readback is
    // deterministic regardless of the backend's buffer-clearing policy.
    result.count = count;
    for (var i: u32 = 0u; i < MAX_N; i = i + 1u) {
        if (i < count) {
            result.sorted[i] = src_key[i];
            result.argsort[i] = src_idx[i];
        } else {
            result.sorted[i] = 0u;
            result.argsort[i] = 0u;
        }
    }
}
"#;

/// Uniform parameters for one dispatch: the key count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RADIX_SORT_U32_WGSL`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid keys in the input array.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// The on-device sort result for one array, matching the `WGSL` `SortResult`
/// struct byte for byte: the written key count, the ascending keys, and the
/// stable `argsort` permutation.
///
/// Lanes `[count, MAX_N)` of [`sorted`](GpuRadixSort::sorted) and
/// [`argsort`](GpuRadixSort::argsort) are zero; use
/// [`sorted_keys`](GpuRadixSort::sorted_keys) and
/// [`argsort_indices`](GpuRadixSort::argsort_indices) to read only the valid
/// prefix. The `sorted` prefix equals
/// [`sort`](prism_render_architecture::particle::radix_sort_u32::sort) and the
/// `argsort` prefix equals
/// [`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuRadixSort {
    /// Number of sorted keys written, equal to the input key count.
    pub count: u32,
    /// Ascending keys in lanes `[0, count)`; lanes `[count, MAX_N)` are zero.
    pub sorted: [u32; MAX_N],
    /// Stable sort permutation in lanes `[0, count)`; lanes `[count, MAX_N)` are
    /// zero. Indexing the original keys by this prefix yields `sorted`.
    pub argsort: [u32; MAX_N],
}

impl GpuRadixSort {
    /// Returns the ascending key prefix actually written, of length
    /// [`count`](GpuRadixSort::count).
    ///
    /// This slice equals the `CPU` golden
    /// [`sort`](prism_render_architecture::particle::radix_sort_u32::sort) output
    /// element for element.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn sorted_keys(&self) -> &[u32] {
        &self.sorted[..self.count as usize]
    }

    /// Returns the stable permutation prefix actually written, of length
    /// [`count`](GpuRadixSort::count).
    ///
    /// This slice equals the `CPU` golden
    /// [`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort)
    /// permutation element for element (as `u32`).
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn argsort_indices(&self) -> &[u32] {
        &self.argsort[..self.count as usize]
    }
}

/// Builds a compute-visible buffer binding layout entry.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
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

/// A compiled, reusable `u32` `radix`-sort compute pipeline, twinning the `CPU`
/// golden
/// [`radix_sort_u32`](prism_render_architecture::particle::radix_sort_u32).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
pub struct GpuRadixSortU32 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRadixSortU32 {
    /// Compiles the `u32` `radix`-sort kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRadixSortU32 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_shader"),
            source: ShaderSource::Wgsl(RADIX_SORT_U32_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("radix_sort_u32_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRadixSortU32 {
            module,
            layout,
            pipeline,
        }
    }

    /// Sorts `keys` on-device and returns the sorted keys plus the stable
    /// `argsort` permutation in one [`GpuRadixSort`].
    ///
    /// The [`sorted`](GpuRadixSort::sorted) prefix equals the `CPU` golden
    /// [`sort`](prism_render_architecture::particle::radix_sort_u32::sort) and
    /// the [`argsort`](GpuRadixSort::argsort) prefix equals the `CPU` golden
    /// [`argsort`](prism_render_architecture::particle::radix_sort_u32::argsort),
    /// element for element, including the stability of equal keys. An empty
    /// `keys` slice issues **no dispatch** — a storage buffer may not be
    /// zero-sized — and returns the all-zero [`GpuRadixSort`] with
    /// [`count`](GpuRadixSort::count) `= 0`.
    ///
    /// # Panics
    ///
    /// Panics if `keys.len()` exceeds [`MAX_N`]; the fixed-length device buffers
    /// cannot hold a longer array.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::radix_sort_u32`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, keys: &[u32]) -> GpuRadixSort {
        let count = keys.len();
        if count == 0 {
            // No key to dispatch, and a storage buffer cannot be zero-sized, so
            // return the all-zero result directly.
            return GpuRadixSort::zeroed();
        }
        assert!(
            count <= MAX_N,
            "radix_sort_u32 twin handles at most MAX_N keys"
        );
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let keys_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_keys"),
            contents: bytemuck::cast_slice(keys),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of::<GpuRadixSort>() as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: keys_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_radix_sort_u32_encoder"),
        });
        {
            // One sort job, so a single thread; flattened to a 1-D dispatch.
            let groups = JOBS.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_radix_sort_u32_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
        let raw = bytemuck::pod_read_unaligned::<GpuRadixSort>(&view);
        drop(view);
        stage.unmap();
        raw
    }
}
