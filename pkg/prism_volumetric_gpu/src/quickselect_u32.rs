//! `wgpu` compute twin of the in-place Hoare/Lomuto `quickselect` order-statistic
//! contract
//! ([`quickselect_u32`](prism_render_architecture::particle::quickselect_u32),
//! design §12 sort ordering, order statistics).
//!
//! A production `GPU` `VFX` pipeline frequently needs a single order statistic
//! rather than a total order: the median particle depth for a soft cutoff, the
//! `k`-th nearest-neighbour distance for a density estimate, or a percentile
//! threshold for adaptive culling. Fully sorting the buffer to read one value
//! wastes work. The `CPU` golden
//! [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
//! and
//! [`median`](prism_render_architecture::particle::quickselect_u32::median) own
//! that selection; [`GpuQuickselectU32`] is the on-device twin that runs one
//! thread per `u32` array and reproduces every lane, so a passing real-device
//! parity test is direct evidence the ported kernel visits the same pivots and
//! partitions the same way the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden control flow function for function: the
//! deterministic
//! [`median_of_three`](prism_render_architecture::particle::quickselect_u32::median_of_three)
//! pivot (three comparisons, no `RNG`), the Lomuto
//! [`partition`](prism_render_architecture::particle::quickselect_u32::partition)
//! with its single write cursor, the iterative
//! [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
//! range narrowing, and the lower-median rule of
//! [`median`](prism_render_architecture::particle::quickselect_u32::median)
//! (sorted index `len / 2 - 1` for even lengths, `len / 2` for odd). Because
//! [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
//! reorders its input in place, each thread works on a private function-scope
//! copy of the array and runs the selected-rank and median passes on two
//! independent copies so the in-place reorder of one can never leak into the
//! other.
//!
//! # Degenerate inputs
//!
//! An empty array yields `None` for both the selected rank and the median, and
//! an out-of-range rank (`k >= count`) yields `None` for the selected rank,
//! matching the reference short circuits. Both outcomes are carried back as a
//! `0`/`1` validity flag rather than a sentinel value, so the host decodes the
//! same [`Option`] the reference returns. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized; an
//! array longer than [`MAX_N`] is clamped to that many keys on upload.
//!
//! # Portability
//!
//! The kernel is pure integer work — `u32` `<=` comparisons, index arithmetic
//! and element swaps — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, no
//! floating point at all, no `u64` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Each loop is bounded by the key
//! count, itself capped at [`MAX_N`], so the kernel provably terminates.
//!
//! # Correctness model
//!
//! A selected key, a rank, a median and a validity flag are all integers, so
//! the parity test asserts **exact** `==` on every field: there is no floating
//! point anywhere on the path and therefore no rounding, no `ULP` boundary and
//! no tie band. Any mismatch is a genuine port bug (a wrong pivot, a dropped
//! swap, a miscounted rank).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::quickselect_u32`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::quickselect_u32::{median, quickselect};
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
/// shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of `u32` keys one array slot holds on device.
///
/// Each array is uploaded into a fixed-length `std430` slot of this many `u32`
/// lanes, so an input with more keys is clamped on upload. `64` matches the
/// serial reference's working-set expectation for a single-thread selection.
///
/// Provenance: `MAX_N` chosen for this twin's fixed per-thread slot; mirrors no
/// reference constant.
pub const MAX_N: usize = 64;

/// The portable core-`WGSL` `quickselect` kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`quickselect_u32`](prism_render_architecture::particle::quickselect_u32)
/// branch for branch; see the module documentation for the algorithm.
const QUICKSELECT_U32_WGSL: &str = r#"
// Quickselect u32 twin: one thread selects one order statistic from one array
// and reproduces the CPU golden `particle::quickselect_u32` branch for branch --
// the deterministic median-of-three pivot, the Lomuto partition write cursor,
// the iterative range narrowing, and the lower-median rule. `queries` holds one
// fixed-length array per thread; `results` receives one record per array. Each
// thread copies its array into a private function-scope buffer and runs the
// selected-rank and median passes on two independent copies so the in-place
// reorder of one cannot leak into the other. The kernel uses only the portable
// core-WGSL subset (u32 <= comparisons, index math and swaps), has no floating
// point and no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. Each loop is bounded by the key count (<= MAX_N), so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::quickselect_u32；无第三方
// 引擎源码或衍生代码。

// Fixed number of key lanes in each array slot; mirrors the host `MAX_N`.
const MAX_N: u32 = 64u;

struct Params {
    // Number of arrays in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Fixed-length key slot: the first `count` lanes are live keys.
    values: array<u32, 64>,
    // Number of live keys in `values` (0..=64).
    count: u32,
    // Zero-based rank requested from `quickselect`.
    k: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Live key count (clamped to MAX_N), echoed back for the host decode.
    count: u32,
    // Rank requested, echoed back for the host decode.
    k: u32,
    // The k-th smallest key when `selected_valid` is 1, else 0.
    selected_value: u32,
    // 1 when the selected rank is in range (k < count), else 0.
    selected_valid: u32,
    // The lower median key when `median_valid` is 1, else 0.
    median_value: u32,
    // 1 when the array is non-empty, else 0.
    median_valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Swap two lanes of a private key buffer, mirroring the reference `slice::swap`.
fn swap_keys(data: ptr<function, array<u32, 64>>, i: u32, j: u32) {
    let tmp = (*data)[i];
    (*data)[i] = (*data)[j];
    (*data)[j] = tmp;
}

// Index of the median of the three keys at `lo`, the midpoint, and `hi`,
// mirroring the reference `median_of_three` comparison tree exactly. No element
// is moved; only three comparisons decide the pivot index.
fn median_of_three(data: ptr<function, array<u32, 64>>, lo: u32, hi: u32) -> u32 {
    let mid = lo + (hi - lo) / 2u;
    let a = (*data)[lo];
    let b = (*data)[mid];
    let c = (*data)[hi];
    if (a <= b) {
        if (b <= c) {
            return mid;
        } else if (a <= c) {
            return hi;
        } else {
            return lo;
        }
    } else if (a <= c) {
        return lo;
    } else if (b <= c) {
        return hi;
    } else {
        return mid;
    }
}

// Lomuto partition of `data[lo..=hi]` around the pivot initially at `pivot`,
// mirroring the reference `partition`: swap the pivot to `hi`, sweep a single
// write cursor moving every key no greater than the pivot to the front, then
// swap the pivot into the cursor position. Returns the pivot's resting index.
fn lomuto_partition(data: ptr<function, array<u32, 64>>, lo: u32, hi: u32, pivot: u32) -> u32 {
    swap_keys(data, pivot, hi);
    let pivot_key = (*data)[hi];
    var store = lo;
    var scan = lo;
    // Walk the range with an explicit cursor; both `scan` and `store` advance.
    while (scan < hi) {
        if ((*data)[scan] <= pivot_key) {
            swap_keys(data, store, scan);
            store = store + 1u;
        }
        scan = scan + 1u;
    }
    swap_keys(data, store, hi);
    return store;
}

// Reorder `data[0..len]` in place so the `k`-th smallest key sits at index `k`,
// returning that key, mirroring the reference `quickselect` range narrowing.
// The caller guarantees `k < len` and `len >= 1`.
fn quickselect_value(data: ptr<function, array<u32, 64>>, k: u32, len: u32) -> u32 {
    var lo = 0u;
    var hi = len - 1u;
    loop {
        if (lo == hi) {
            return (*data)[lo];
        }
        let pivot = median_of_three(data, lo, hi);
        let p = lomuto_partition(data, lo, hi, pivot);
        if (p == k) {
            return (*data)[p];
        } else if (k < p) {
            // Target lies strictly left of the pivot; drop the right side.
            hi = p - 1u;
        } else {
            // Target lies strictly right of the pivot; drop the left side.
            lo = p + 1u;
        }
    }
    // Unreachable: the loop only exits through a return above.
    return 0u;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var n = queries[idx].count;
    if (n > MAX_N) {
        n = MAX_N;
    }
    let k = queries[idx].k;

    var res: Result;
    res.count = n;
    res.k = k;
    res.selected_value = 0u;
    res.selected_valid = 0u;
    res.median_value = 0u;
    res.median_valid = 0u;

    // Selected order statistic: an out-of-range rank (k >= n, which also covers
    // the empty array) is a miss, matching the reference `None`. Runs on its own
    // private copy so the in-place reorder cannot leak into the median run.
    if (k < n) {
        var sel_data: array<u32, 64>;
        for (var i = 0u; i < n; i = i + 1u) {
            sel_data[i] = queries[idx].values[i];
        }
        res.selected_value = quickselect_value(&sel_data, k, n);
        res.selected_valid = 1u;
    }

    // Lower median: `None` only for an empty array; otherwise the sorted index
    // is `n / 2 - 1` for even lengths and `n / 2` for odd, matching the
    // reference `median`. Its own private copy, independent of the selected run.
    if (n > 0u) {
        var med_k = n / 2u;
        if ((n & 1u) == 0u) {
            med_k = n / 2u - 1u;
        }
        var med_data: array<u32, 64>;
        for (var i = 0u; i < n; i = i + 1u) {
            med_data[i] = queries[idx].values[i];
        }
        res.median_value = quickselect_value(&med_data, med_k, n);
        res.median_valid = 1u;
    }

    results[idx] = res;
}
"#;

/// One `quickselect` request: an open array of `u32` keys and the zero-based
/// rank to select from it.
///
/// Mirrors a single reference
/// [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
/// call paired with a
/// [`median`](prism_render_architecture::particle::quickselect_u32::median) over
/// the same keys. Arrays longer than [`MAX_N`] are clamped to that many keys on
/// upload. Derives [`Eq`] because it holds only integer keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuickselectQuery {
    /// The `u32` keys to select over; clamped to [`MAX_N`] on upload.
    pub values: Vec<u32>,
    /// The zero-based rank passed to
    /// [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect).
    pub k: u32,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `selected_value` is the `k`-th smallest key and `selected_valid` is its
/// `0`/`1` presence flag, together decoding the [`Option`] from
/// [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect);
/// `median_value` and `median_valid` decode the [`Option`] from
/// [`median`](prism_render_architecture::particle::quickselect_u32::median). The
/// `count` and `k` fields echo the clamped live-key count and the requested
/// rank. Derives [`Eq`] because every field is an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuQuickselect {
    /// Live key count after clamping to [`MAX_N`].
    pub count: u32,
    /// The zero-based rank that was requested.
    pub k: u32,
    /// The `k`-th smallest key; meaningful only when `selected_valid` is `1`.
    pub selected_value: u32,
    /// Selected-rank presence flag (`1` when `k < count`, else `0`).
    pub selected_valid: u32,
    /// The lower median key; meaningful only when `median_valid` is `1`.
    pub median_value: u32,
    /// Median presence flag (`1` when the array is non-empty, else `0`).
    pub median_valid: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUICKSELECT_U32_WGSL`].
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

/// `repr(C)` `std430` layout of one query slot, matching the `WGSL` `Query`
/// struct: `MAX_N` `u32` key lanes followed by the live key count, the rank and
/// two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `MAX_N` `u32` key lanes; the first `count` are live.
    values: [u32; MAX_N],
    /// Number of live keys in `values`.
    count: u32,
    /// Zero-based rank passed to the kernel's `quickselect`.
    k: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: six `u32` lanes carrying the clamped count, the rank, the selected
/// value and its flag, and the median value and its flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Live key count after clamping.
    count: u32,
    /// Rank requested.
    k: u32,
    /// Selected `k`-th smallest key (`0` when invalid).
    selected_value: u32,
    /// Selected-rank presence flag.
    selected_valid: u32,
    /// Lower median key (`0` when invalid).
    median_value: u32,
    /// Median presence flag.
    median_valid: u32,
}

impl GpuQuery {
    /// Packs a [`QuickselectQuery`] into the fixed-length `std430` upload
    /// layout, clamping to [`MAX_N`] live keys.
    fn from_query(query: &QuickselectQuery) -> GpuQuery {
        let count = query.values.len().min(MAX_N);
        let mut values = [0u32; MAX_N];
        values[..count].copy_from_slice(&query.values[..count]);
        GpuQuery {
            values,
            count: count as u32,
            k: query.k,
            pad0: 0,
            pad1: 0,
        }
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuQuickselect`].
fn decode_result(raw: &GpuResult) -> GpuQuickselect {
    GpuQuickselect {
        count: raw.count,
        k: raw.k,
        selected_value: raw.selected_value,
        selected_valid: raw.selected_valid,
        median_value: raw.median_value,
        median_valid: raw.median_valid,
    }
}

/// The `CPU` golden verdict for one query, dispatching to the reference
/// [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
/// and
/// [`median`](prism_render_architecture::particle::quickselect_u32::median) so
/// callers (and the parity test) can pin the twin lane for lane.
///
/// The array is clamped to [`MAX_N`] keys to match the on-device slot, then the
/// selected rank and the lower median are computed on two independent copies —
/// each entry point reorders its input in place — and returned as explicit
/// `0`/`1` validity flags beside their values.
#[must_use]
pub fn cpu_reference(query: &QuickselectQuery) -> GpuQuickselect {
    let count = query.values.len().min(MAX_N);
    let keys = &query.values[..count];
    let mut sel = keys.to_vec();
    let selected = quickselect(&mut sel, query.k as usize);
    let mut med = keys.to_vec();
    let median_value = median(&mut med);
    GpuQuickselect {
        count: count as u32,
        k: query.k,
        selected_value: selected.unwrap_or(0),
        selected_valid: u32::from(selected.is_some()),
        median_value: median_value.unwrap_or(0),
        median_valid: u32::from(median_value.is_some()),
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

/// A compiled, reusable `quickselect` compute pipeline, twinning the `CPU`
/// golden
/// [`quickselect_u32`](prism_render_architecture::particle::quickselect_u32).
pub struct GpuQuickselectU32 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuickselectU32 {
    /// Compiles the `quickselect` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuickselectU32 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quickselect_u32"),
            source: ShaderSource::Wgsl(QUICKSELECT_U32_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quickselect_u32_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quickselect_u32_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quickselect_u32_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuickselectU32 {
            module,
            layout,
            pipeline,
        }
    }

    /// Selects the requested rank and the lower median for every query in
    /// `queries`, returning one [`GpuQuickselect`] per input in order.
    ///
    /// The returned record for query `q` mirrors
    /// [`quickselect`](prism_render_architecture::particle::quickselect_u32::quickselect)
    /// and
    /// [`median`](prism_render_architecture::particle::quickselect_u32::median)
    /// evaluated on `q.values` (clamped to [`MAX_N`]) and `q.k`, exactly: every
    /// field is an integer and matches the reference without tolerance. An empty
    /// `queries` batch yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[QuickselectQuery]) -> Vec<GpuQuickselect> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quickselect_u32_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quickselect_u32_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quickselect_u32_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quickselect_u32_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quickselect_u32_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quickselect_u32_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quickselect_u32_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
