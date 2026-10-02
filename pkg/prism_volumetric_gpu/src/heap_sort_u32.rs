//! `wgpu` compute twin of the in-place binary max-heap `u32` sort golden
//! ([`heap_sort`](prism_render_architecture::particle::heap_sort_u32::heap_sort),
//! design §12 sort ordering).
//!
//! The particle subsystem orders small per-tile or per-bucket `u32` key runs
//! ascending with no auxiliary buffer; the `CPU` golden
//! [`heap_sort`](prism_render_architecture::particle::heap_sort_u32::heap_sort)
//! owns that in-place heapsort and
//! [`is_max_heap`](prism_render_architecture::particle::heap_sort_u32::is_max_heap)
//! owns the companion max-heap predicate over the *original* input.
//! [`GpuHeapSortU32`] is the on-device twin that runs one thread per array and
//! reproduces both: it heap-sorts the array in place and reports whether the
//! untouched input already satisfied the max-heap property. A passing
//! real-device parity test is therefore direct evidence the ported kernel folds
//! the same `sift_down` control flow and the same ascending extraction the
//! reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Each array has at most [`MAX_N`] elements and reaches the device in a fixed
//! `std430` slot block (`count` live elements plus zero padding). One thread
//! copies its slot into an invocation-private scratch array, evaluates the
//! max-heap predicate on the pristine copy (parents `0..count / 2`, each
//! compared against both present children with `>`), then runs the two-phase
//! heapsort: `build_max_heap` sifts every internal node from the last parent
//! (`count / 2 - 1`) down to the root, and the extraction loop swaps the root
//! to the current end of the live heap, shrinks the heap by one, and sifts the
//! new root back down until the slice is sorted ascending. The `sift_down`
//! inner loop mirrors the golden guard for guard: pick the larger of the two
//! present children, stop when the parent already dominates it (`>=`), else
//! swap down and continue.
//!
//! # Portability
//!
//! The kernel is pure integer arithmetic — index shifts and adds, integer
//! comparisons (`<`, `>=`, `>`) and element swaps — using only the portable
//! core-`WGSL` subset with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`,
//! `smoothstep`, no `atomic`, no `u64` and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Sorted keys and the predicate flag are integers, so the parity test asserts
//! **exact equality**, not a float tolerance: the full `u32` key range is
//! reproduced bit for bit, and the `is_max_heap_input` flag is an exact `u32`
//! boolean. There is no floating point anywhere on the path, hence no `ULP`
//! boundary and no degenerate region to avoid. Any mismatch is a genuine port
//! bug (a wrong child pick, a dropped swap, a miscounted heap bound).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::heap_sort_u32`；无第三方引擎源码或衍生代码。
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
/// that divides evenly across `Metal`, `Vulkan` and `DX12`.
const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of `u32` keys in one heap-sorted array. Arrays travel to the
/// device in a fixed `std430` slot block of this width (`count` live elements
/// plus zero padding), matching the `MAX_N` constant baked into the shader.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::heap_sort_u32`。
pub const MAX_N: usize = 64;

/// Core-`WGSL` heapsort kernel, inlined so this twin lives entirely in the
/// crate with no external `.wesl`. One thread per array copies its fixed slot
/// block into an invocation-private scratch array, records whether the pristine
/// input was already a max-heap, then heap-sorts the live `count` prefix
/// ascending in place, mirroring the `CPU` golden
/// `particle::heap_sort_u32::heap_sort` and `::is_max_heap`.
const HEAP_SORT_U32_WGSL: &str = r#"
// Heapsort u32 twin: one thread per array. Each array is a fixed MAX_N-slot
// block (`count` live keys plus zero padding). The thread copies the block into
// an invocation-private scratch array, evaluates the max-heap predicate on the
// pristine copy, then runs the two-phase in-place heapsort over the live
// prefix. Mirrors the CPU golden `particle::heap_sort_u32::{heap_sort,
// is_max_heap}`. No u64, no transcendental, no sqrt, no atomic.

// Fixed per-array slot width; mirrors the host `MAX_N`.
const MAX_N: u32 = 64u;

struct Params {
    num: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One input array: `count` live keys followed by zero padding up to MAX_N.
struct HeapInput {
    count: u32,
    data: array<u32, 64>,
}

// One sorted output: `count`, the full MAX_N slot block (sorted prefix plus
// zero padding) and the pre-sort max-heap predicate as a u32 boolean.
struct HeapOutput {
    count: u32,
    sorted: array<u32, 64>,
    is_max_heap_input: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<HeapInput>;
@group(0) @binding(2) var<storage, read_write> outputs: array<HeapOutput>;

// Invocation-private scratch heap: each thread owns its own instance, so the
// in-place sort never races a sibling lane.
var<private> scratch: array<u32, 64>;

// Restore the max-heap property for the subtree rooted at `root_in` over the
// live heap `scratch[0..heap_len]`. The node sinks toward the leaves, swapping
// with its larger present child whenever that child is strictly greater, until
// it dominates its larger child or becomes a leaf. Mirrors the golden
// `sift_down_by` guard for guard.
fn sift_down(root_in: u32, heap_len: u32) {
    var root = root_in;
    loop {
        let left = 2u * root + 1u;
        // A missing left child means `root` is a leaf within the live heap.
        if (left >= heap_len) {
            break;
        }
        // Pick the larger of the two children (right child only if present).
        let right = left + 1u;
        var largest = left;
        if (right < heap_len && scratch[right] > scratch[left]) {
            largest = right;
        }
        // If the parent already dominates its larger child, the subtree is a
        // valid max-heap and the node has reached its resting place.
        if (scratch[root] >= scratch[largest]) {
            break;
        }
        let tmp = scratch[root];
        scratch[root] = scratch[largest];
        scratch[largest] = tmp;
        root = largest;
    }
}

// Transform `scratch[0..n]` into a max-heap by sifting down every internal node
// from the last parent (`n / 2 - 1`) up to the root. Mirrors the golden
// `build_max_heap_by`: empty and single-element prefixes are already heaps.
fn build_max_heap(n: u32) {
    if (n < 2u) {
        return;
    }
    var node = n / 2u;
    loop {
        if (node == 0u) {
            break;
        }
        node = node - 1u;
        sift_down(node, n);
    }
}

// Sort `scratch[0..n]` ascending in place with an in-place binary max-heap.
// Mirrors the golden `heap_sort`: build the heap, then repeatedly swap the root
// maximum to the current end of the live heap, shrink it, and sift the new root
// back down so the sorted suffix grows from the back.
fn heap_sort(n: u32) {
    if (n < 2u) {
        return;
    }
    build_max_heap(n);
    var heap_len = n;
    loop {
        if (heap_len <= 1u) {
            break;
        }
        heap_len = heap_len - 1u;
        let tmp = scratch[0];
        scratch[0] = scratch[heap_len];
        scratch[heap_len] = tmp;
        sift_down(0u, heap_len);
    }
}

@compute @workgroup_size(64)
fn heap_sort_u32_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num) {
        return;
    }
    let n = inputs[idx].count;
    // Load the fixed slot block: live keys verbatim, padding zeroed.
    for (var i = 0u; i < MAX_N; i = i + 1u) {
        if (i < n) {
            scratch[i] = inputs[idx].data[i];
        } else {
            scratch[i] = 0u;
        }
    }
    // Evaluate the max-heap predicate on the pristine (pre-sort) copy: only
    // internal nodes `0..n / 2` can violate it; each is compared against both
    // present children with `>`. Any violation clears the flag.
    var ismh = 1u;
    let half = n / 2u;
    for (var parent = 0u; parent < half; parent = parent + 1u) {
        let left = 2u * parent + 1u;
        let right = left + 1u;
        if (left < n && scratch[left] > scratch[parent]) {
            ismh = 0u;
        }
        if (right < n && scratch[right] > scratch[parent]) {
            ismh = 0u;
        }
    }
    // Sort the live prefix in place, then write back the whole slot block.
    heap_sort(n);
    outputs[idx].count = n;
    for (var i = 0u; i < MAX_N; i = i + 1u) {
        outputs[idx].sorted[i] = scratch[i];
    }
    outputs[idx].is_max_heap_input = ismh;
}
"#;

/// One heapsort request: the `u32` keys of a single array to sort ascending.
///
/// The slice may hold at most [`MAX_N`] keys; it is zero-padded to the fixed
/// `std430` slot width on the way to the device. An empty slice is valid and
/// yields a zero-count result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeapSortU32Query {
    /// The `u32` keys to sort ascending, at most [`MAX_N`] of them.
    pub data: Vec<u32>,
}

/// Uniform parameters for one dispatch. `16`-byte `repr(C)` matching `Params`
/// in the inlined shader: the array count plus padding to the uniform
/// alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    num: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One input array as the device sees it: `count` live keys followed by zero
/// padding, matching `HeapInput` in the inlined shader. `repr(C)` with no
/// padding (`4 + 4 * MAX_N` bytes, `4`-byte aligned), so it is `Pod`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHeapInput {
    count: u32,
    data: [u32; MAX_N],
}

impl GpuHeapInput {
    /// Builds the fixed-slot device input for `query`, copying its live keys and
    /// zero-padding the remaining slots.
    ///
    /// # Panics
    ///
    /// Panics if `query.data` holds more than [`MAX_N`] keys, which cannot fit
    /// the fixed `std430` slot block.
    fn from_query(query: &HeapSortU32Query) -> GpuHeapInput {
        assert!(
            query.data.len() <= MAX_N,
            "heap-sort array length {} exceeds MAX_N {MAX_N}",
            query.data.len()
        );
        let mut data = [0u32; MAX_N];
        data[..query.data.len()].copy_from_slice(&query.data);
        GpuHeapInput {
            count: query.data.len() as u32,
            data,
        }
    }
}

/// One sorted array as the device returns it, mirroring `HeapOutput` in the
/// inlined shader: the live key `count`, the full [`MAX_N`] slot block (sorted
/// prefix plus zero padding) and the pre-sort max-heap predicate as a `u32`
/// boolean (`1` when the pristine input was already a max-heap, else `0`).
///
/// `repr(C)` with no padding (`4 + 4 * MAX_N + 4` bytes, `4`-byte aligned), so
/// it is `Pod` and reads back directly without a decode step.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuHeapSort {
    /// Number of live keys in the sorted prefix (`sorted[..count]`).
    pub count: u32,
    /// The fixed slot block: the ascending sorted prefix in `sorted[..count]`
    /// followed by zero padding in `sorted[count..]`.
    pub sorted: [u32; MAX_N],
    /// `1` when the pristine input array was already a valid max-heap (per the
    /// golden `is_max_heap`), else `0`.
    pub is_max_heap_input: u32,
}

/// A compiled, reusable in-place `u32` heapsort pipeline.
pub struct GpuHeapSortU32 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHeapSortU32 {
    /// Compiles the in-place `u32` heapsort kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHeapSortU32 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_shader"),
            source: ShaderSource::Wgsl(HEAP_SORT_U32_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("heap_sort_u32_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHeapSortU32 {
            module,
            layout,
            pipeline,
        }
    }

    /// Heap-sorts each array in `queries` ascending in place on-device.
    ///
    /// Returns one [`GpuHeapSort`] per query, in query order: its `sorted`
    /// prefix equals the `CPU` golden
    /// [`heap_sort`](prism_render_architecture::particle::heap_sort_u32::heap_sort)
    /// output exactly, and `is_max_heap_input` matches the golden
    /// [`is_max_heap`](prism_render_architecture::particle::heap_sort_u32::is_max_heap)
    /// over the pristine input. An empty `queries` slice issues **no dispatch**
    /// — a storage buffer cannot be zero-sized — so it is handled by an early
    /// return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[HeapSortU32Query]) -> Vec<GpuHeapSort> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            num: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_inputs: Vec<GpuHeapInput> = queries.iter().map(GpuHeapInput::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuHeapSort>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let outputs_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_outputs"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let outputs_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_outputs_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: outputs_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_heap_sort_u32_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_heap_sort_u32_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per array, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&outputs_buf, 0, &outputs_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        outputs_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = outputs_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let results = bytemuck::cast_slice::<u8, GpuHeapSort>(&view).to_vec();
        drop(view);
        outputs_stage.unmap();
        debug_assert_eq!(results.len(), queries.len());
        results
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
