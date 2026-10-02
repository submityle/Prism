//! `wgpu` compute twin of the stable `u32` merge-sort contract
//! ([`merge_sort_stable`](prism_render_architecture::particle::merge_sort_stable),
//! particle design §12 sort ordering).
//!
//! The `CPU` golden
//! [`merge_sort_stable`](prism_render_architecture::particle::merge_sort_stable)
//! owns the serial *stable* comparison sort a `GPU` VFX stack needs when two
//! particles share a quantized depth or draw-order key: the earlier element must
//! still appear earlier after sorting, so tie-broken layering, additive blend
//! order and deterministic `CPU`/`GPU` cross-checks stay reproducible. The
//! reference exposes three building blocks:
//! [`merge`](prism_render_architecture::particle::merge_sort_stable::merge)
//! merges two already-sorted adjacent runs with the `left <= right` tie rule
//! (equal keys take the left, earlier run first),
//! [`merge_runs`](prism_render_architecture::particle::merge_sort_stable::merge_runs)
//! is the bottom-up driver that doubles the run width `1`, `2`, `4`, ... while
//! carrying a parallel payload slice, and
//! [`merge_sort_u32`](prism_render_architecture::particle::merge_sort_stable::merge_sort_u32)
//! sorts a raw `u32` slice ascending through that engine.
//!
//! [`GpuMergeSortStable`] is the on-device twin: a single thread sorts one
//! `u32` array of up to [`MAX_N`] elements with the identical iterative,
//! bottom-up merge, so a passing real-device parity test is direct evidence the
//! ported kernel reproduces the reference ordering value for value, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel sorts the `count` supplied keys ascending and, to make stability
//! observable, carries each key's original index through the same merges. It
//! emits a [`GpuMergeSort`] holding the sorted keys and the stable *argsort* —
//! the permutation of original indices. The sorted keys reproduce
//! [`merge_sort_u32`](prism_render_architecture::particle::merge_sort_stable::merge_sort_u32)
//! and the argsort reproduces
//! [`merge_runs`](prism_render_architecture::particle::merge_sort_stable::merge_runs)
//! run with the original indices as the payload: within any equal-key group the
//! original indices stay strictly increasing.
//!
//! # Correctness model
//!
//! Everything is exact unsigned integer work: keys compare with `<=` on `u32`,
//! indices are `u32` additions bounded by [`MAX_N`], and run widths double by a
//! left shift. Nothing divides and no floating point or transcendental call
//! appears, so the `CPU` and `GPU` orderings are *bit-exact*. The parity test
//! therefore asserts an exact `==` on every sorted key and every argsort index.
//!
//! # Degenerate inputs
//!
//! A length `0` or `1` array is already sorted. An empty array short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized; a
//! single element returns through the kernel's skipped merge loop unchanged.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `<=`, `+`, a
//! left shift and unsigned index arithmetic over fixed-length `array<u32, 64>`
//! scratch buffers — with no `sqrt`, no transcendental call, no recursion and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every loop bound is `MAX_N`, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::merge_sort_stable`；无第三方引擎源码或衍生代码。
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
/// used across this crate's one-thread-per-element kernels; here only lane `0`
/// performs the single sort task.
const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of keys one dispatch sorts, matching the fixed `std430` slot
/// count of the `WGSL` scratch and output arrays.
pub const MAX_N: usize = 64;

/// The portable core-`WGSL` stable merge-sort kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `sort` mirrors the
/// `CPU` golden
/// [`merge_sort_stable`](prism_render_architecture::particle::merge_sort_stable)
/// bottom-up driver; see the module documentation for the algorithm.
const MERGE_SORT_STABLE_WGSL: &str = r#"
// Stable merge-sort twin: lane 0 iteratively (bottom-up) merges the `count`
// input keys ascending while carrying each key's original index, so the output
// exposes both the sorted keys and the stable argsort permutation. The merge
// step uses the `left <= right` tie rule, so equal keys keep their original
// order. It mirrors the CPU golden `particle::merge_sort_stable` driver, uses
// only the portable core-WGSL subset (min, <=, +, a left shift and unsigned
// index math over fixed-length array<u32, 64> scratch), needs no sqrt and no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. Every loop bound is the fixed capacity, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::merge_sort_stable；无第三方
// 引擎源码或衍生代码。

// Fixed capacity: the length of every scratch and output array.
const CAP: u32 = 64u;

struct Params {
    // Number of valid keys to sort; must be in 0..=CAP (0 is short-circuited on
    // the host, so a dispatched call always has count >= 1).
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct MergeSort {
    // Number of valid entries in `sorted` and `argsort`.
    count: u32,
    // Ascending stable-sorted keys in lanes 0..count; trailing lanes are zero.
    sorted: array<u32, 64>,
    // Original index of each sorted key (the stable argsort); trailing lanes are
    // zero.
    argsort: array<u32, 64>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: MergeSort;

@compute @workgroup_size(64)
fn sort(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Exactly one sort task: only lane 0 does work, the rest short-circuit.
    if (gid.x >= 1u) {
        return;
    }

    let n = params.count;

    // Working buffers plus their merge-pass scratch; WGSL zero-initializes var
    // arrays, so trailing lanes past `n` stay zero.
    var keys_buf: array<u32, 64>;
    var idx_buf: array<u32, 64>;
    var keys_scratch: array<u32, 64>;
    var idx_scratch: array<u32, 64>;

    // Seed with the input keys and their original indices.
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        keys_buf[i] = keys_in[i];
        idx_buf[i] = i;
    }

    // Bottom-up merge: run width 1, 2, 4, ... until a single run covers n. Each
    // pass merges adjacent runs into scratch, then copies scratch back.
    var width: u32 = 1u;
    while (width < n) {
        let step = width << 1u;
        var lo: u32 = 0u;
        while (lo < n) {
            let mid = min(lo + width, n);
            let hi = min(lo + step, n);
            var left = lo;
            var right = mid;
            var out = lo;
            // `<=` keeps the left (earlier) element first on ties: the entire
            // stability guarantee.
            while (left < mid && right < hi) {
                if (keys_buf[left] <= keys_buf[right]) {
                    keys_scratch[out] = keys_buf[left];
                    idx_scratch[out] = idx_buf[left];
                    left = left + 1u;
                } else {
                    keys_scratch[out] = keys_buf[right];
                    idx_scratch[out] = idx_buf[right];
                    right = right + 1u;
                }
                out = out + 1u;
            }
            while (left < mid) {
                keys_scratch[out] = keys_buf[left];
                idx_scratch[out] = idx_buf[left];
                left = left + 1u;
                out = out + 1u;
            }
            while (right < hi) {
                keys_scratch[out] = keys_buf[right];
                idx_scratch[out] = idx_buf[right];
                right = right + 1u;
                out = out + 1u;
            }
            lo = lo + step;
        }
        // Copy the merged pass back so keys_buf/idx_buf hold the partial order.
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            keys_buf[i] = keys_scratch[i];
            idx_buf[i] = idx_scratch[i];
        }
        width = step;
    }

    result.count = n;
    for (var i: u32 = 0u; i < CAP; i = i + 1u) {
        result.sorted[i] = keys_buf[i];
        result.argsort[i] = idx_buf[i];
    }
}
"#;

/// Uniform parameters for one dispatch: the key count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MERGE_SORT_STABLE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid keys to sort.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of the single sort output, matching the `WGSL`
/// `MergeSort` struct: a count word followed by the fixed `MAX_N`-slot sorted
/// keys and argsort indices.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Number of valid entries in `sorted` and `argsort`.
    count: u32,
    /// Ascending stable-sorted keys; trailing slots are zero.
    sorted: [u32; MAX_N],
    /// Original index of each sorted key; trailing slots are zero.
    argsort: [u32; MAX_N],
}

/// One resolved stable sort: the ascending keys and the stable argsort (the
/// permutation of original indices), both valid in lanes `0..count`.
///
/// `sorted` reproduces
/// [`merge_sort_u32`](prism_render_architecture::particle::merge_sort_stable::merge_sort_u32)
/// and `argsort` reproduces
/// [`merge_runs`](prism_render_architecture::particle::merge_sort_stable::merge_runs)
/// carrying the original indices: within any equal-key group the indices in
/// `argsort` stay strictly increasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMergeSort {
    /// Number of valid entries in `sorted` and `argsort`.
    pub count: u32,
    /// Ascending stable-sorted keys; lanes `0..count` are valid, the rest zero.
    pub sorted: [u32; MAX_N],
    /// Stable argsort permutation of original indices; lanes `0..count` are
    /// valid, the rest zero.
    pub argsort: [u32; MAX_N],
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

/// A compiled, reusable stable merge-sort compute pipeline, twinning the `CPU`
/// golden
/// [`merge_sort_stable`](prism_render_architecture::particle::merge_sort_stable).
pub struct GpuMergeSortStable {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMergeSortStable {
    /// Compiles the stable merge-sort kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMergeSortStable {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_merge_sort_stable"),
            source: ShaderSource::Wgsl(MERGE_SORT_STABLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sort"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMergeSortStable {
            module,
            layout,
            pipeline,
        }
    }

    /// Stably sorts `keys` ascending and returns the sorted keys plus the stable
    /// argsort permutation of original indices.
    ///
    /// The sorted keys and argsort indices match the `CPU` golden exactly (an
    /// exact `==`), since the kernel does pure unsigned integer work. An empty
    /// `keys` slice returns a zeroed [`GpuMergeSort`] with `count` `0` and no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics if `keys` has more than [`MAX_N`] elements, which exceeds the
    /// fixed `std430` slot count the kernel is built for.
    #[must_use]
    pub fn sort(&self, ctx: &GpuContext, keys: &[u32]) -> GpuMergeSort {
        let count = keys.len();
        assert!(
            count <= MAX_N,
            "merge_sort_stable twin sorts at most {MAX_N} keys, got {count}"
        );
        if count == 0 {
            return GpuMergeSort {
                count: 0,
                sorted: [0u32; MAX_N],
                argsort: [0u32; MAX_N],
            };
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let keys_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_keys"),
            contents: bytemuck::cast_slice(keys),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of::<GpuResult>() as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_bind_group"),
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
            label: Some("prism_volumetric_merge_sort_stable_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_merge_sort_stable_encoder"),
        });
        {
            // Exactly one sort task, flattened to a 1-D dispatch.
            let groups = 1u32.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_merge_sort_stable_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view)[0];
        drop(view);
        stage.unmap();

        GpuMergeSort {
            count: raw.count,
            sorted: raw.sorted,
            argsort: raw.argsort,
        }
    }
}
