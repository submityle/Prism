//! `wgpu` compute twin of the simulation-space and large-world relocation
//! contracts
//! ([`sim_space`](prism_render_architecture::particle::sim_space), particle
//! design §26).
//!
//! The `CPU` golden module resolves, for one emitter's
//! [`SimSpace`](prism_render_architecture::particle::SimSpace), the spawn/update
//! coordinate contract
//! ([`plan`](prism_render_architecture::particle::sim_space::plan),
//! [`storage_space`](prism_render_architecture::particle::sim_space::storage_space)),
//! the rigid-frame local↔world maps
//! ([`TransformFrame`](prism_render_architecture::particle::sim_space::TransformFrame)),
//! the floating-origin rebase trigger
//! ([`RebaseConfig::should_rebase`](prism_render_architecture::particle::sim_space::RebaseConfig::should_rebase)),
//! the two-layer chunk coordinate
//! ([`ChunkGrid`](prism_render_architecture::particle::sim_space::ChunkGrid) /
//! [`ChunkCoord`](prism_render_architecture::particle::sim_space::ChunkCoord)),
//! the rebase helpers
//! ([`rebase_offset`](prism_render_architecture::particle::sim_space::rebase_offset),
//! [`offset_is_significant`](prism_render_architecture::particle::sim_space::offset_is_significant),
//! [`apply_rebase_position`](prism_render_architecture::particle::sim_space::apply_rebase_position),
//! [`apply_rebase_velocity`](prism_render_architecture::particle::sim_space::apply_rebase_velocity),
//! [`rebase_particle`](prism_render_architecture::particle::sim_space::rebase_particle)),
//! and the `fp32` `ULP` estimator
//! ([`fp32_ulp`](prism_render_architecture::particle::sim_space::fp32_ulp)).
//! [`GpuSimSpace`] is the on-device twin: one thread resolves one aggregate
//! query, so a passing real-device parity test is direct evidence the ported
//! kernel runs the same classification, linear algebra, floored chunk split and
//! bit-level `ULP` reconstruction the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each per-query answer the reference computes is reproduced in one aggregate
//! [`GpuSimSpaceResult`]: the resolved plan (`spawn_bake_to_world` flag and
//! storage frame) and bare storage frame; the four frame maps
//! (`transform_point`, `transform_direction`, `inverse_transform_point`,
//! `inverse_transform_direction`); the rebase trigger; the chunk `split`
//! (`chunk` index and intra-chunk `offset`), `compose`, `compose_relative` and
//! `snap_to_chunk`; the chunk-aligned `rebase_offset`; the significance test;
//! the position/velocity/space rebase helpers; and the `fp32` `ULP` estimate.
//! The kernel mirrors each reference branch — the `Local`-vs-`World` storage
//! split, the floored chunk division's integer truncation, and the
//! `zero`/`subnormal` branch of the `ULP` reconstruction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `dot`,
//! `+ - *` and integer bit operations (`bitcast`, shifts, masks) — with no
//! `sqrt`, `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no rounding intrinsic and
//! no optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The chunk index is carried as `i32`, the only portable signed
//! integer, matching the reference's `i32` chunk index.
//!
//! # Correctness model
//!
//! The continuous answers (frame maps, chunk offsets, rebase vectors and the
//! `ULP` magnitude) are fixed, non-reorderable sequences of multiplies and
//! adds, so `CPU` and `GPU` evaluate the same closed form. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on those, while the discrete answers — classification
//! codes, booleans and the `i32` chunk index — are compared exactly, and the
//! bit-reconstructed `fp32_ulp` is compared bit-for-bit (its reconstruction is
//! pure integer math on an un-recomputed input, so both devices emit the same
//! bits).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`sim_space`](prism_render_architecture::particle::sim_space);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sim_space::{
    ChunkCoord, ChunkGrid, SimSpacePlan, StorageSpace, TransformFrame,
};
use prism_render_architecture::particle::{SimSpace, Vec3};
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

/// The portable core-`WGSL` simulation-space kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`sim_space`](prism_render_architecture::particle::sim_space) function by
/// function; see the module documentation for the algorithm.
const SIM_SPACE_WGSL: &str = r#"
// Simulation-space twin: one thread resolves one aggregate query. It mirrors the
// CPU golden `particle::sim_space` branch for branch, uses only the portable
// core-WGSL subset (floor/dot/+ - * and integer bit ops) with no sqrt and no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::sim_space;
// no third-party engine source or derived code.

// Squared-length floor below which a rebase offset counts as insignificant,
// matching the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // SimSpace classification code: Local=0, World=1, Hybrid=2.
    space_code: u32,
    // TransformFrame as four packed vec3s: origin, right, up, forward.
    frame: array<f32, 12>,
    // Local input shared by transform_point and transform_direction.
    local: array<f32, 3>,
    // World input shared by inverse_transform_point and inverse_transform_direction.
    world: array<f32, 3>,
    // Camera position for should_rebase and rebase_offset.
    camera: array<f32, 3>,
    // RebaseConfig trigger distance.
    threshold: f32,
    // ChunkGrid edge length.
    chunk_size: f32,
    // Position fed to split and snap_to_chunk.
    split_in: array<f32, 3>,
    // ChunkCoord chunk index for compose / compose_relative.
    coord_chunk: array<i32, 3>,
    // ChunkCoord intra-chunk offset for compose / compose_relative.
    coord_offset: array<f32, 3>,
    // Reference chunk index for compose_relative.
    reference: array<i32, 3>,
    // Rebase offset for offset_is_significant / apply_rebase_position / rebase_particle.
    offset: array<f32, 3>,
    // World-space position for apply_rebase_position / rebase_particle.
    pos: array<f32, 3>,
    // World-space velocity for apply_rebase_velocity.
    vel: array<f32, 3>,
    // Magnitude for the fp32 ULP estimate.
    ulp_value: f32,
}

struct Result {
    // plan().spawn_bake_to_world as 0/1.
    plan_spawn_bake: u32,
    // plan().storage code: Local=0, World=1.
    plan_storage: u32,
    // storage_space() code: Local=0, World=1.
    storage_space: u32,
    // should_rebase as 0/1.
    should_rebase: u32,
    // offset_is_significant as 0/1.
    offset_is_significant: u32,
    // split().chunk integer index.
    split_chunk: array<i32, 3>,
    // transform_point result.
    transform_point: array<f32, 3>,
    // transform_direction result.
    transform_direction: array<f32, 3>,
    // inverse_transform_point result.
    inverse_transform_point: array<f32, 3>,
    // inverse_transform_direction result.
    inverse_transform_direction: array<f32, 3>,
    // split().offset result.
    split_offset: array<f32, 3>,
    // compose result.
    compose: array<f32, 3>,
    // compose_relative result.
    compose_relative: array<f32, 3>,
    // snap_to_chunk result.
    snap_to_chunk: array<f32, 3>,
    // rebase_offset result.
    rebase_offset: array<f32, 3>,
    // apply_rebase_position result.
    apply_rebase_position: array<f32, 3>,
    // apply_rebase_velocity result.
    apply_rebase_velocity: array<f32, 3>,
    // rebase_particle result.
    rebase_particle: array<f32, 3>,
    // fp32_ulp result.
    fp32_ulp: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Reconstructs a vec3 from a packed array<f32, 3>.
fn pack3(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}

// Flattens a vec3 into a packed array<f32, 3>.
fn flat3(v: vec3<f32>) -> array<f32, 3> {
    return array<f32, 3>(v.x, v.y, v.z);
}

// Order-of-magnitude fp32 ULP, rebuilt with pure integer bit math, mirroring
// the reference `fp32_ulp`: zero/subnormal report the smallest positive
// subnormal.
fn fp32_ulp(value: f32) -> f32 {
    let mag_bits = bitcast<u32>(value) & 0x7fffffffu;
    let exp = (mag_bits >> 23u) & 0xffu;
    if (exp == 0u) {
        return bitcast<f32>(1u);
    }
    let ulp_exp = i32(exp) - 23;
    if (ulp_exp <= 0) {
        return bitcast<f32>(1u);
    }
    return bitcast<f32>(u32(ulp_exp) << 23u);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let origin = vec3<f32>(q.frame[0], q.frame[1], q.frame[2]);
    let right = vec3<f32>(q.frame[3], q.frame[4], q.frame[5]);
    let up = vec3<f32>(q.frame[6], q.frame[7], q.frame[8]);
    let forward = vec3<f32>(q.frame[9], q.frame[10], q.frame[11]);
    let local = pack3(q.local);
    let world = pack3(q.world);
    let camera = pack3(q.camera);
    let split_in = pack3(q.split_in);
    let coord_offset = pack3(q.coord_offset);
    let offset = pack3(q.offset);
    let pos = pack3(q.pos);
    let vel = pack3(q.vel);
    let cs = q.chunk_size;

    var out: Result;

    // plan / storage_space: Local stores Local; World and Hybrid store World.
    var storage_code: u32 = 1u;
    var spawn_bake: u32 = 1u;
    if (q.space_code == 0u) {
        storage_code = 0u;
        spawn_bake = 0u;
    }
    out.plan_spawn_bake = spawn_bake;
    out.plan_storage = storage_code;
    out.storage_space = storage_code;

    // Rigid-frame maps: a plain linear combination of the basis columns.
    let tp = origin + right * local.x + up * local.y + forward * local.z;
    let td = right * local.x + up * local.y + forward * local.z;
    let d = world - origin;
    let itp = vec3<f32>(dot(d, right), dot(d, up), dot(d, forward));
    let itd = vec3<f32>(dot(world, right), dot(world, up), dot(world, forward));
    out.transform_point = flat3(tp);
    out.transform_direction = flat3(td);
    out.inverse_transform_point = flat3(itp);
    out.inverse_transform_direction = flat3(itd);

    // Rebase trigger: squared magnitudes, so no sqrt and no fp32 equality.
    var should_rebase: u32 = 0u;
    if (dot(camera, camera) > q.threshold * q.threshold) {
        should_rebase = 1u;
    }
    out.should_rebase = should_rebase;

    // Chunk split: floored quotient plus the in-range remainder.
    let fx = floor(split_in.x / cs);
    let fy = floor(split_in.y / cs);
    let fz = floor(split_in.z / cs);
    out.split_chunk = array<i32, 3>(i32(fx), i32(fy), i32(fz));
    out.split_offset = flat3(vec3<f32>(
        split_in.x - fx * cs,
        split_in.y - fy * cs,
        split_in.z - fz * cs,
    ));

    // Compose: absolute reconstruction and reference-relative reconstruction.
    let cx = f32(q.coord_chunk[0]);
    let cy = f32(q.coord_chunk[1]);
    let cz = f32(q.coord_chunk[2]);
    out.compose = flat3(vec3<f32>(cx * cs, cy * cs, cz * cs) + coord_offset);
    let rdx = f32(q.coord_chunk[0] - q.reference[0]);
    let rdy = f32(q.coord_chunk[1] - q.reference[1]);
    let rdz = f32(q.coord_chunk[2] - q.reference[2]);
    out.compose_relative = flat3(vec3<f32>(rdx * cs, rdy * cs, rdz * cs) + coord_offset);

    // snap_to_chunk of split_in, and the chunk-aligned rebase offset of camera.
    out.snap_to_chunk = flat3(vec3<f32>(
        floor(split_in.x / cs) * cs,
        floor(split_in.y / cs) * cs,
        floor(split_in.z / cs) * cs,
    ));
    out.rebase_offset = flat3(vec3<f32>(
        floor(camera.x / cs) * cs,
        floor(camera.y / cs) * cs,
        floor(camera.z / cs) * cs,
    ));

    // Significance test, position/velocity rebase and space-aware rebase.
    var significant: u32 = 0u;
    if (dot(offset, offset) > EPS_LEN_SQ) {
        significant = 1u;
    }
    out.offset_is_significant = significant;
    out.apply_rebase_position = flat3(pos - offset);
    out.apply_rebase_velocity = flat3(vel);
    if (storage_code == 1u) {
        out.rebase_particle = flat3(pos - offset);
    } else {
        out.rebase_particle = flat3(pos);
    }

    out.fp32_ulp = fp32_ulp(q.ulp_value);

    results[idx] = out;
}
"#;

/// One aggregate simulation-space query: the emitter's
/// [`SimSpace`](prism_render_architecture::particle::SimSpace), the rigid
/// [`TransformFrame`](prism_render_architecture::particle::sim_space::TransformFrame),
/// the [`ChunkGrid`](prism_render_architecture::particle::sim_space::ChunkGrid)
/// and every scalar/vector input the twinned `CPU` golden functions consume.
///
/// Provenance: inputs to this repository's
/// [`sim_space`](prism_render_architecture::particle::sim_space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSimSpaceQuery {
    /// The emitter's simulation space.
    pub space: SimSpace,
    /// The rigid transform frame.
    pub frame: TransformFrame,
    /// The chunk lattice.
    pub grid: ChunkGrid,
    /// The reference chunk index for `compose_relative`.
    pub reference: [i32; 3],
    /// Local input shared by `transform_point` and `transform_direction`.
    pub local: Vec3,
    /// World input shared by `inverse_transform_point` and
    /// `inverse_transform_direction`.
    pub world: Vec3,
    /// Camera position for `should_rebase` and `rebase_offset`.
    pub camera: Vec3,
    /// Rebase trigger distance.
    pub threshold: f32,
    /// Position fed to `split` and `snap_to_chunk`.
    pub split_in: Vec3,
    /// Chunk index of the coordinate fed to `compose` / `compose_relative`.
    pub coord_chunk: [i32; 3],
    /// Intra-chunk offset of the coordinate fed to `compose` /
    /// `compose_relative`.
    pub coord_offset: Vec3,
    /// Rebase offset for `offset_is_significant`, `apply_rebase_position` and
    /// `rebase_particle`.
    pub offset: Vec3,
    /// World-space position for `apply_rebase_position` and `rebase_particle`.
    pub pos: Vec3,
    /// World-space velocity for `apply_rebase_velocity`.
    pub vel: Vec3,
    /// Magnitude for the `fp32` `ULP` estimate.
    pub ulp_value: f32,
}

/// The resolved answers for one query, each matching the corresponding `CPU`
/// golden [`sim_space`](prism_render_architecture::particle::sim_space)
/// function.
///
/// Provenance: outputs of this repository's
/// [`sim_space`](prism_render_architecture::particle::sim_space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSimSpaceResult {
    /// The resolved spawn/update plan.
    pub plan: SimSpacePlan,
    /// The bare storage frame.
    pub storage_space: StorageSpace,
    /// `transform_point(local)`.
    pub transform_point: Vec3,
    /// `transform_direction(local)`.
    pub transform_direction: Vec3,
    /// `inverse_transform_point(world)`.
    pub inverse_transform_point: Vec3,
    /// `inverse_transform_direction(world)`.
    pub inverse_transform_direction: Vec3,
    /// `should_rebase(camera)`.
    pub should_rebase: bool,
    /// `split(split_in)`.
    pub split: ChunkCoord,
    /// `compose(coord)`.
    pub compose: Vec3,
    /// `compose_relative(coord, reference)`.
    pub compose_relative: Vec3,
    /// `snap_to_chunk(split_in)`.
    pub snap_to_chunk: Vec3,
    /// `rebase_offset(camera, grid)`.
    pub rebase_offset: Vec3,
    /// `offset_is_significant(offset)`.
    pub offset_is_significant: bool,
    /// `apply_rebase_position(pos, offset)`.
    pub apply_rebase_position: Vec3,
    /// `apply_rebase_velocity(vel)`.
    pub apply_rebase_velocity: Vec3,
    /// `rebase_particle(space, pos, offset)`.
    pub rebase_particle: Vec3,
    /// `fp32_ulp(ulp_value)`.
    pub fp32_ulp: f32,
}

/// Maps a [`SimSpace`](prism_render_architecture::particle::SimSpace) to the
/// classification code the kernel expects: `Local=0`, `World=1`, `Hybrid=2`.
fn space_code(space: SimSpace) -> u32 {
    match space {
        SimSpace::Local => 0,
        SimSpace::World => 1,
        SimSpace::Hybrid => 2,
    }
}

/// Maps a storage classification code back to a
/// [`StorageSpace`](prism_render_architecture::particle::sim_space::StorageSpace).
fn storage_from_code(code: u32) -> StorageSpace {
    match code {
        0 => StorageSpace::Local,
        _ => StorageSpace::World,
    }
}

/// `repr(C)` `std430` layout of one packed query. Every field is a `4`-byte
/// scalar (or scalar array), so the struct needs no interior padding and maps
/// `1:1` to the `WGSL` `Query` struct, whose fixed-size `f32`/`i32` arrays carry
/// a `4`-byte stride in the storage address space.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Simulation-space classification code.
    space_code: u32,
    /// Transform frame packed as `origin`, `right`, `up`, `forward`.
    frame: [f32; 12],
    /// Local input vector.
    local: [f32; 3],
    /// World input vector.
    world: [f32; 3],
    /// Camera position.
    camera: [f32; 3],
    /// Rebase trigger distance.
    threshold: f32,
    /// Chunk edge length.
    chunk_size: f32,
    /// Position fed to `split` / `snap_to_chunk`.
    split_in: [f32; 3],
    /// Coordinate chunk index.
    coord_chunk: [i32; 3],
    /// Coordinate intra-chunk offset.
    coord_offset: [f32; 3],
    /// Reference chunk index.
    reference: [i32; 3],
    /// Rebase offset.
    offset: [f32; 3],
    /// World-space position.
    pos: [f32; 3],
    /// World-space velocity.
    vel: [f32; 3],
    /// Magnitude for the `ULP` estimate.
    ulp_value: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &GpuSimSpaceQuery) -> GpuQuery {
        let f = query.frame;
        GpuQuery {
            space_code: space_code(query.space),
            frame: [
                f.origin.x,
                f.origin.y,
                f.origin.z,
                f.right.x,
                f.right.y,
                f.right.z,
                f.up.x,
                f.up.y,
                f.up.z,
                f.forward.x,
                f.forward.y,
                f.forward.z,
            ],
            local: [query.local.x, query.local.y, query.local.z],
            world: [query.world.x, query.world.y, query.world.z],
            camera: [query.camera.x, query.camera.y, query.camera.z],
            threshold: query.threshold,
            chunk_size: query.grid.chunk_size,
            split_in: [query.split_in.x, query.split_in.y, query.split_in.z],
            coord_chunk: query.coord_chunk,
            coord_offset: [
                query.coord_offset.x,
                query.coord_offset.y,
                query.coord_offset.z,
            ],
            reference: query.reference,
            offset: [query.offset.x, query.offset.y, query.offset.z],
            pos: [query.pos.x, query.pos.y, query.pos.z],
            vel: [query.vel.x, query.vel.y, query.vel.z],
            ulp_value: query.ulp_value,
        }
    }
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct
/// field for field; every member is a `4`-byte scalar or scalar array, so no
/// interior padding is introduced.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `plan().spawn_bake_to_world` as `0`/`1`.
    plan_spawn_bake: u32,
    /// `plan().storage` classification code.
    plan_storage: u32,
    /// `storage_space()` classification code.
    storage_space: u32,
    /// `should_rebase` as `0`/`1`.
    should_rebase: u32,
    /// `offset_is_significant` as `0`/`1`.
    offset_is_significant: u32,
    /// `split().chunk` integer index.
    split_chunk: [i32; 3],
    /// `transform_point` result.
    transform_point: [f32; 3],
    /// `transform_direction` result.
    transform_direction: [f32; 3],
    /// `inverse_transform_point` result.
    inverse_transform_point: [f32; 3],
    /// `inverse_transform_direction` result.
    inverse_transform_direction: [f32; 3],
    /// `split().offset` result.
    split_offset: [f32; 3],
    /// `compose` result.
    compose: [f32; 3],
    /// `compose_relative` result.
    compose_relative: [f32; 3],
    /// `snap_to_chunk` result.
    snap_to_chunk: [f32; 3],
    /// `rebase_offset` result.
    rebase_offset: [f32; 3],
    /// `apply_rebase_position` result.
    apply_rebase_position: [f32; 3],
    /// `apply_rebase_velocity` result.
    apply_rebase_velocity: [f32; 3],
    /// `rebase_particle` result.
    rebase_particle: [f32; 3],
    /// `fp32_ulp` result.
    fp32_ulp: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable simulation-space compute pipeline.
pub struct GpuSimSpace {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSimSpace {
    /// Compiles the simulation-space kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSimSpace {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sim_space"),
            source: ShaderSource::Wgsl(SIM_SPACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sim_space_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sim_space_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sim_space_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSimSpace {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query on-device and returns one [`GpuSimSpaceResult`] per
    /// input, in order.
    ///
    /// Each result equals the corresponding `CPU` golden
    /// [`sim_space`](prism_render_architecture::particle::sim_space) functions to
    /// within the tolerance documented on this module (continuous answers) or
    /// exactly (classification codes, booleans, the `i32` chunk index and the
    /// bit-reconstructed `fp32_ulp`). An empty input returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuSimSpaceQuery]) -> Vec<GpuSimSpaceResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sim_space_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sim_space_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sim_space_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sim_space_bind_group"),
            layout: &self.layout,
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

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sim_space_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sim_space_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sim_space_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`GpuSimSpaceResult`].
fn decode_result(raw: &GpuResult) -> GpuSimSpaceResult {
    GpuSimSpaceResult {
        plan: SimSpacePlan {
            spawn_bake_to_world: raw.plan_spawn_bake != 0,
            storage: storage_from_code(raw.plan_storage),
        },
        storage_space: storage_from_code(raw.storage_space),
        transform_point: vec3(raw.transform_point),
        transform_direction: vec3(raw.transform_direction),
        inverse_transform_point: vec3(raw.inverse_transform_point),
        inverse_transform_direction: vec3(raw.inverse_transform_direction),
        should_rebase: raw.should_rebase != 0,
        split: ChunkCoord {
            chunk: raw.split_chunk,
            offset: vec3(raw.split_offset),
        },
        compose: vec3(raw.compose),
        compose_relative: vec3(raw.compose_relative),
        snap_to_chunk: vec3(raw.snap_to_chunk),
        rebase_offset: vec3(raw.rebase_offset),
        offset_is_significant: raw.offset_is_significant != 0,
        apply_rebase_position: vec3(raw.apply_rebase_position),
        apply_rebase_velocity: vec3(raw.apply_rebase_velocity),
        rebase_particle: vec3(raw.rebase_particle),
        fp32_ulp: raw.fp32_ulp,
    }
}

/// Rebuilds a [`Vec3`](prism_render_architecture::particle::Vec3) from a packed
/// three-component array.
fn vec3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
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
