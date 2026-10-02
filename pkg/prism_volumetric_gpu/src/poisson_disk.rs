//! `wgpu` compute twin of the deterministic, closed-form core of `Bridson`'s
//! blue-noise Poisson-disk sampler
//! ([`poisson_disk`](prism_render_architecture::particle::poisson_disk),
//! particle design §16-§21, §25).
//!
//! The `CPU` golden
//! [`poisson_disk`](prism_render_architecture::particle::poisson_disk) owns a
//! stateful sampling pass: an active-list frontier, a variable-length grid of
//! occupied cells and a growing output `Vec`. None of that fits a
//! one-thread-one-element kernel. What *does* port cleanly is the pass's pure
//! numeric spine: the `splitmix32` integer-hash stream
//! ([`Rng::next_u32`](prism_render_architecture::particle::poisson_disk::Rng::next_u32),
//! [`Rng::next_unit`](prism_render_architecture::particle::poisson_disk::Rng::next_unit)),
//! the background-grid cell size
//! ([`cell_size`](prism_render_architecture::particle::poisson_disk::cell_size)),
//! the squared/linear point-distance core shared by
//! [`min_pairwise_distance`](prism_render_architecture::particle::poisson_disk::min_pairwise_distance)
//! and the private `candidate_fits` predicate, the private `cell_index_1d`
//! coordinate-to-cell map, and the per-candidate min-distance accept/reject
//! test over a host-supplied neighbor set.
//!
//! [`GpuPoissonDisk`] is the on-device twin: one thread solves one
//! [`PoissonDiskQuery`] and writes one [`PoissonDiskResult`], so a passing
//! real-device parity test is direct evidence the ported kernel folds the same
//! words, scalars, indices and verdicts the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Each query selects one routine by a `u32` tag and the kernel reproduces it
//! branch for branch: the `splitmix32` bump-and-mix word, the `2^-32` unit-draw
//! scaling, the `radius / sqrt(2)` cell size, the `dx*dx + dy*dy` squared
//! distance and its `sqrt`, the `floor(coord / cell)` cell index folded into a
//! `row * cols + col` linear index, and the "every neighbor is at least `r`
//! away" acceptance predicate.
//!
//! # What is left on the host
//!
//! The variable-state machinery is deliberately *not* twinned here, since a
//! single thread cannot own unbounded, growing state:
//! [`poisson_disk_sample`](prism_render_architecture::particle::poisson_disk::poisson_disk_sample)
//! drives an active-list frontier and emits a variable-length `Vec`; the full
//! [`min_pairwise_distance`](prism_render_architecture::particle::poisson_disk::min_pairwise_distance)
//! walks an `O(n^2)` all-pairs loop; the acceleration grid stores variable-length
//! occupancy; and the annulus-dart rejection loop and point insertion mutate that
//! shared grid. The host runs those and feeds the kernel the fixed-length,
//! stateless slices (a generator state word, two points, a candidate plus its
//! occupied-cell neighbors) that each twinned routine consumes.
//!
//! # No transcendental math
//!
//! Every routine is integer bit-mixing or multiply-add plus at most one `sqrt`
//! (the pair-distance report). The kernel uses no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no inverse trigonometry and no `smoothstep` or `round`; the
//! annulus geometry that *would* need trigonometry stays on the host, which
//! draws it by rejection sampling.
//!
//! # Correctness model
//!
//! The dispatch tag is an integer classification, so the kernel runs exactly the
//! branch the host requested. The integer routines (`next_u32`, the cell index
//! and the accept/reject verdict) are bit-exact and are compared with `==`. The
//! continuous entries (`next_unit`, `cell_size`, the pair distance) thread
//! through a `u32`-to-`f32` widen, a divide or a `sqrt`, so `CPU` and `GPU` are
//! not guaranteed bit-exact: a `GPU` may round a widen or a divide a hair
//! differently. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous quantity.
//!
//! # Degenerate inputs
//!
//! The reference only ever calls `cell_index_1d` and `candidate_fits` with a
//! strictly positive cell size and in-domain coordinates, so the twin and its
//! fixtures keep the cell size positive and the coordinates clear of exact cell
//! boundaries, where a one-`ULP` `floor` disagreement could flip an index.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::poisson_disk`；无第三方引擎源码或衍生代码。
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
use prism_render_architecture::particle::poisson_disk::{cell_size, min_pairwise_distance, Rng};

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Upper bound on the occupied-cell neighbors the host may hand a single
/// [`PoissonDiskQuery::CandidateFits`] query. `Bridson`'s `r / sqrt(2)` grid
/// holds at most one point per cell and the min-distance test only ever consults
/// the `5x5` window, whose blue-noise-occupied subset is tiny; `8` is a safe cap
/// for that conflict set.
const MAX_NEIGHBORS: usize = 8;

/// The portable core-`WGSL` Poisson-disk kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`poisson_disk`](prism_render_architecture::particle::poisson_disk) numeric
/// core branch for branch; see the module documentation for the algorithm.
const POISSON_DISK_WGSL: &str = r#"
// Poisson-disk twin: one thread per query runs the routine its `tag` selects,
// reproducing the CPU golden `particle::poisson_disk` numeric core branch for
// branch. It uses only the portable core-WGSL subset (floor/max/min/sqrt, the
// u32 bit operations and + - * / plus unsigned index math), needs no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. The only loop is bounded by a small constant, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::poisson_disk；无第三方引擎源码
// 或衍生代码。

// 2^-32, the exact reciprocal of 2^32 (a power of two), scaling a u32 draw into
// the unit interval; mirrors the reference `INV_U32_SPAN`.
const INV_U32_SPAN: f32 = 2.3283064365386963e-10;

// sqrt(2); the Bridson cell size is `radius / SQRT_2`. The f32 round of this
// literal equals the reference `core::f32::consts::SQRT_2`.
const SQRT_2: f32 = 1.4142135623730951;

// Upper bound on neighbors inspected by the accept/reject test; matches the host
// `MAX_NEIGHBORS` cap.
const MAX_NEIGHBORS: u32 = 8u;

// Routine tags; the host casts its query discriminant straight to these codes.
const TAG_CANDIDATE_FITS: u32 = 0u;
const TAG_CELL_INDEX: u32 = 1u;
const TAG_CELL_SIZE: u32 = 2u;
const TAG_NEXT_U32: u32 = 3u;
const TAG_NEXT_UNIT: u32 = 4u;
const TAG_PAIR_DISTANCE: u32 = 5u;
const TAG_PAIR_DISTANCE_SQUARED: u32 = 6u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Routine selector matching the TAG_* codes.
    tag: u32,
    // Generator state word for next_u32 / next_unit.
    state: u32,
    // Occupied-neighbor count for candidate_fits.
    neighbor_count: u32,
    // Grid column count for the cell_index linear fold.
    cols: u32,
    // General scalar lane: cell_index cell size, cell_size radius, or
    // candidate_fits r^2 in x.
    scalar: vec4<f32>,
    // Point lane: (a.x, a.y, b.x, b.y) for the pair distances, (coord.x,
    // coord.y, _, _) for cell_index, (candidate.x, candidate.y, _, _) for
    // candidate_fits.
    ab: vec4<f32>,
    // Packed neighbors 0,1 as (x0, y0, x1, y1).
    n0: vec4<f32>,
    // Packed neighbors 2,3.
    n1: vec4<f32>,
    // Packed neighbors 4,5.
    n2: vec4<f32>,
    // Packed neighbors 6,7.
    n3: vec4<f32>,
}

struct Result {
    // Scalar lane: next_unit / cell_size / pair distance in x.
    scalar: vec4<f32>,
    // Word lane: next_u32 word, cell linear index, or 0/1 fit verdict in x.
    word: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// splitmix32 finalizer: bump the state by the golden-ratio odd constant then mix
// with two xor-shift multiplies. Mirrors the reference `Rng::next_u32`; the u32
// multiplies wrap by definition in WGSL, matching the host `wrapping_mul`.
fn next_u32(state: u32) -> u32 {
    let s = state + 0x9E3779B9u;
    var z = s;
    z = (z ^ (z >> 16u)) * 0x21F0AAADu;
    z = (z ^ (z >> 15u)) * 0x735A2D97u;
    return z ^ (z >> 15u);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.scalar = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.word = vec4<u32>(0u, 0u, 0u, 0u);

    if (q.tag == TAG_CANDIDATE_FITS) {
        // Accept only when every neighbor is at least r away (compared as r^2);
        // the reference rejects on the first neighbor with squared distance < r^2.
        let cx = q.ab.x;
        let cy = q.ab.y;
        let r2 = q.scalar.x;
        var nx: array<f32, 8>;
        var ny: array<f32, 8>;
        nx[0] = q.n0.x; ny[0] = q.n0.y;
        nx[1] = q.n0.z; ny[1] = q.n0.w;
        nx[2] = q.n1.x; ny[2] = q.n1.y;
        nx[3] = q.n1.z; ny[3] = q.n1.w;
        nx[4] = q.n2.x; ny[4] = q.n2.y;
        nx[5] = q.n2.z; ny[5] = q.n2.w;
        nx[6] = q.n3.x; ny[6] = q.n3.y;
        nx[7] = q.n3.z; ny[7] = q.n3.w;
        let n = min(q.neighbor_count, MAX_NEIGHBORS);
        var fits = 1u;
        for (var i = 0u; i < n; i = i + 1u) {
            let dx = nx[i] - cx;
            let dy = ny[i] - cy;
            if (dx * dx + dy * dy < r2) {
                fits = 0u;
            }
        }
        out.word.x = fits;
    } else if (q.tag == TAG_CELL_INDEX) {
        // floor(coord / cell), clamped to zero, folded as row * cols + col.
        let cell = q.scalar.x;
        let col = u32(max(floor(q.ab.x / cell), 0.0));
        let row = u32(max(floor(q.ab.y / cell), 0.0));
        out.word.x = row * q.cols + col;
    } else if (q.tag == TAG_CELL_SIZE) {
        out.scalar.x = q.scalar.x / SQRT_2;
    } else if (q.tag == TAG_NEXT_U32) {
        out.word.x = next_u32(q.state);
    } else if (q.tag == TAG_NEXT_UNIT) {
        out.scalar.x = f32(next_u32(q.state)) * INV_U32_SPAN;
    } else if (q.tag == TAG_PAIR_DISTANCE) {
        let dx = q.ab.x - q.ab.z;
        let dy = q.ab.y - q.ab.w;
        out.scalar.x = sqrt(dx * dx + dy * dy);
    } else if (q.tag == TAG_PAIR_DISTANCE_SQUARED) {
        let dx = q.ab.x - q.ab.z;
        let dy = q.ab.y - q.ab.w;
        out.scalar.x = dx * dx + dy * dy;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`POISSON_DISK_WGSL`].
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
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Routine selector matching the `WGSL` `TAG_*` codes.
    tag: u32,
    /// Generator state word for `next_u32` / `next_unit`.
    state: u32,
    /// Occupied-neighbor count for `candidate_fits`.
    neighbor_count: u32,
    /// Grid column count for the `cell_index` linear fold.
    cols: u32,
    /// General scalar lane (`cell`, `radius` or `r^2` in `x`).
    scalar: [f32; 4],
    /// Point lane `(a.x, a.y, b.x, b.y)` or `(coord.x, coord.y, _, _)`.
    ab: [f32; 4],
    /// Packed neighbors `0,1` as `(x0, y0, x1, y1)`.
    n0: [f32; 4],
    /// Packed neighbors `2,3`.
    n1: [f32; 4],
    /// Packed neighbors `4,5`.
    n2: [f32; 4],
    /// Packed neighbors `6,7`.
    n3: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Scalar lane: `next_unit` / `cell_size` / pair distance in `x`.
    scalar: [f32; 4],
    /// Word lane: `next_u32` word, cell linear index, or `0`/`1` verdict in `x`.
    word: [u32; 4],
}

/// One query for the Poisson-disk twin: a tagged union selecting which golden
/// routine to run with its typed inputs.
///
/// Each variant twins exactly one deterministic, closed-form routine of the
/// golden [`poisson_disk`](prism_render_architecture::particle::poisson_disk)
/// pass; the stateful sampler itself stays on the host (see the module docs).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::poisson_disk`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoissonDiskQuery {
    /// The per-candidate min-distance accept/reject test, twinning the squared
    /// min-distance core of the golden private `candidate_fits` predicate over a
    /// host-supplied occupied-neighbor set.
    CandidateFits {
        /// Candidate point `(x, y)` under test.
        candidate: [f32; 2],
        /// Occupied neighbors from the candidate's `5x5` grid window; only the
        /// first `neighbor_count` entries are inspected.
        neighbors: [[f32; 2]; MAX_NEIGHBORS],
        /// Number of valid entries in `neighbors`.
        neighbor_count: u32,
        /// Squared minimum spacing `r^2`.
        r_squared: f32,
    },
    /// The grid cell coordinate-to-linear-index map, twinning the golden private
    /// `cell_index_1d` fold `row * cols + col`.
    CellIndex {
        /// Domain coordinate `(x, y)`.
        coord: [f32; 2],
        /// Background-grid cell size.
        cell: f32,
        /// Number of grid columns.
        cols: u32,
    },
    /// The `Bridson` background-grid cell size, twinning
    /// [`cell_size`](prism_render_architecture::particle::poisson_disk::cell_size).
    CellSize {
        /// Minimum spacing `r`.
        radius: f32,
    },
    /// The next pseudo-random word, twinning
    /// [`Rng::next_u32`](prism_render_architecture::particle::poisson_disk::Rng::next_u32)
    /// evaluated from the given generator state.
    NextU32 {
        /// Generator state word fed to the `splitmix32` finalizer.
        state: u32,
    },
    /// The next unit-interval draw, twinning
    /// [`Rng::next_unit`](prism_render_architecture::particle::poisson_disk::Rng::next_unit)
    /// evaluated from the given generator state.
    NextUnit {
        /// Generator state word fed to the `splitmix32` finalizer.
        state: u32,
    },
    /// The Euclidean distance between two points, twinning the single-pair core
    /// of
    /// [`min_pairwise_distance`](prism_render_architecture::particle::poisson_disk::min_pairwise_distance).
    PairDistance {
        /// First point `(x, y)`.
        a: [f32; 2],
        /// Second point `(x, y)`.
        b: [f32; 2],
    },
    /// The squared distance between two points, the comparison quantity the
    /// sampler keeps in squared space.
    PairDistanceSquared {
        /// First point `(x, y)`.
        a: [f32; 2],
        /// Second point `(x, y)`.
        b: [f32; 2],
    },
}

impl PoissonDiskQuery {
    /// Returns the `WGSL` routine tag for this query.
    fn tag(&self) -> u32 {
        match self {
            PoissonDiskQuery::CandidateFits { .. } => 0,
            PoissonDiskQuery::CellIndex { .. } => 1,
            PoissonDiskQuery::CellSize { .. } => 2,
            PoissonDiskQuery::NextU32 { .. } => 3,
            PoissonDiskQuery::NextUnit { .. } => 4,
            PoissonDiskQuery::PairDistance { .. } => 5,
            PoissonDiskQuery::PairDistanceSquared { .. } => 6,
        }
    }
}

/// One resolved answer for a single query: a tagged union whose variant matches
/// the routine the corresponding [`PoissonDiskQuery`] selected.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::poisson_disk`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoissonDiskResult {
    /// A boolean accept/reject verdict (`candidate_fits`).
    Fits(bool),
    /// A grid linear index (`cell_index`).
    Index(u32),
    /// A continuous scalar (`cell_size`, `next_unit` or a pair distance).
    Scalar(f32),
    /// A pseudo-random word (`next_u32`).
    Word(u32),
}

/// Reimplements the golden private `cell_index_1d`: floor a non-negative,
/// cell-scaled coordinate to its `1D` grid index.
///
/// The reference narrows an always-non-negative, bounded `f32` floor to an
/// index; the fixtures keep the cell size positive and the coordinate clear of
/// cell boundaries so this narrowing is exact and the twin agrees.
fn cell_index_1d(coord: f32, cell: f32) -> u32 {
    (coord / cell).floor().max(0.0) as u32
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points (or faithfully re-deriving the private closed forms) so callers (and
/// the parity test) can pin the twin lane for lane.
///
/// The public golden
/// [`cell_size`](prism_render_architecture::particle::poisson_disk::cell_size),
/// [`Rng::next_u32`](prism_render_architecture::particle::poisson_disk::Rng::next_u32),
/// [`Rng::next_unit`](prism_render_architecture::particle::poisson_disk::Rng::next_unit)
/// and
/// [`min_pairwise_distance`](prism_render_architecture::particle::poisson_disk::min_pairwise_distance)
/// are called directly; the private `cell_index_1d`, the squared-distance core
/// and the `candidate_fits` predicate are re-derived from their closed forms,
/// since they are not exported.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::poisson_disk`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn cpu_reference(query: &PoissonDiskQuery) -> PoissonDiskResult {
    match query {
        PoissonDiskQuery::CandidateFits {
            candidate,
            neighbors,
            neighbor_count,
            r_squared,
        } => {
            let n = (*neighbor_count as usize).min(MAX_NEIGHBORS);
            let mut fits = true;
            for p in &neighbors[..n] {
                let dx = p[0] - candidate[0];
                let dy = p[1] - candidate[1];
                if dx * dx + dy * dy < *r_squared {
                    fits = false;
                }
            }
            PoissonDiskResult::Fits(fits)
        }
        PoissonDiskQuery::CellIndex { coord, cell, cols } => {
            let col = cell_index_1d(coord[0], *cell);
            let row = cell_index_1d(coord[1], *cell);
            PoissonDiskResult::Index(row * *cols + col)
        }
        PoissonDiskQuery::CellSize { radius } => PoissonDiskResult::Scalar(cell_size(*radius)),
        PoissonDiskQuery::NextU32 { state } => PoissonDiskResult::Word(Rng::new(*state).next_u32()),
        PoissonDiskQuery::NextUnit { state } => {
            PoissonDiskResult::Scalar(Rng::new(*state).next_unit())
        }
        PoissonDiskQuery::PairDistance { a, b } => {
            PoissonDiskResult::Scalar(min_pairwise_distance(&[*a, *b]))
        }
        PoissonDiskQuery::PairDistanceSquared { a, b } => {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            PoissonDiskResult::Scalar(dx * dx + dy * dy)
        }
    }
}

/// Encodes one [`PoissonDiskQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &PoissonDiskQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    g.tag = q.tag();
    match q {
        PoissonDiskQuery::CandidateFits {
            candidate,
            neighbors,
            neighbor_count,
            r_squared,
        } => {
            g.neighbor_count = (*neighbor_count).min(MAX_NEIGHBORS as u32);
            g.scalar = [*r_squared, 0.0, 0.0, 0.0];
            g.ab = [candidate[0], candidate[1], 0.0, 0.0];
            g.n0 = [
                neighbors[0][0],
                neighbors[0][1],
                neighbors[1][0],
                neighbors[1][1],
            ];
            g.n1 = [
                neighbors[2][0],
                neighbors[2][1],
                neighbors[3][0],
                neighbors[3][1],
            ];
            g.n2 = [
                neighbors[4][0],
                neighbors[4][1],
                neighbors[5][0],
                neighbors[5][1],
            ];
            g.n3 = [
                neighbors[6][0],
                neighbors[6][1],
                neighbors[7][0],
                neighbors[7][1],
            ];
        }
        PoissonDiskQuery::CellIndex { coord, cell, cols } => {
            g.cols = *cols;
            g.scalar = [*cell, 0.0, 0.0, 0.0];
            g.ab = [coord[0], coord[1], 0.0, 0.0];
        }
        PoissonDiskQuery::CellSize { radius } => {
            g.scalar = [*radius, 0.0, 0.0, 0.0];
        }
        PoissonDiskQuery::NextU32 { state } | PoissonDiskQuery::NextUnit { state } => {
            g.state = *state;
        }
        PoissonDiskQuery::PairDistance { a, b }
        | PoissonDiskQuery::PairDistanceSquared { a, b } => {
            g.ab = [a[0], a[1], b[0], b[1]];
        }
    }
    g
}

/// Decodes one packed [`GpuResult`] into the public [`PoissonDiskResult`],
/// selecting the variant from the query's routine.
fn decode_result(q: &PoissonDiskQuery, raw: &GpuResult) -> PoissonDiskResult {
    match q {
        PoissonDiskQuery::CandidateFits { .. } => PoissonDiskResult::Fits(raw.word[0] != 0),
        PoissonDiskQuery::CellIndex { .. } => PoissonDiskResult::Index(raw.word[0]),
        PoissonDiskQuery::NextU32 { .. } => PoissonDiskResult::Word(raw.word[0]),
        PoissonDiskQuery::CellSize { .. }
        | PoissonDiskQuery::NextUnit { .. }
        | PoissonDiskQuery::PairDistance { .. }
        | PoissonDiskQuery::PairDistanceSquared { .. } => PoissonDiskResult::Scalar(raw.scalar[0]),
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

/// A compiled, reusable Poisson-disk compute pipeline, twinning the `CPU` golden
/// [`poisson_disk`](prism_render_architecture::particle::poisson_disk) numeric
/// core.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::poisson_disk`；无第三方引擎源码或衍生代码。
pub struct GpuPoissonDisk {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPoissonDisk {
    /// Compiles the Poisson-disk kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPoissonDisk {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_poisson_disk"),
            source: ShaderSource::Wgsl(POISSON_DISK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_poisson_disk_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_poisson_disk_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_poisson_disk_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPoissonDisk {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`PoissonDiskResult`] per
    /// input, in order.
    ///
    /// The result variant matches the routine each query selected, matching the
    /// reference to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PoissonDiskQuery],
    ) -> Vec<PoissonDiskResult> {
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
            label: Some("prism_volumetric_poisson_disk_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_poisson_disk_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_poisson_disk_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_poisson_disk_bind_group"),
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
            label: Some("prism_volumetric_poisson_disk_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_poisson_disk_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_poisson_disk_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
