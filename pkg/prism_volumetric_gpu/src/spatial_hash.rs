//! `wgpu` compute twin of the uniform-grid spatial-hash build kernels
//! ([`spatial_hash`](prism_render_architecture::particle::spatial_hash), design
//! §7 `PerNeighborCell`, §10).
//!
//! Broad-phase neighbor acceleration bins entities into a uniform grid and
//! builds a counting-sort bucket layout: fold each cell coordinate into a
//! hash-table slot, count how many entities land in each cell, exclusive
//! prefix-sum the counts into offsets, then stably scatter indices into the
//! buckets. The `CPU` golden
//! [`spatial_hash`](prism_render_architecture::particle::spatial_hash) owns that
//! whole contract; [`GpuSpatialHash`] is the on-device twin of its two
//! self-contained, one-`dispatch` building blocks, so a passing real-device
//! parity test is direct evidence the ported kernels fold and bucket exactly as
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Two kernels, each one `dispatch` and self-contained:
//!
//! * **Hash** ([`hash_cells`](GpuSpatialHash::hash_cells)): one thread per cell
//!   coordinate `[i32; 3]`, folding it into a slot in `[0, table_size)` by
//!   mixing the three axes with the standard large primes (Teschner et al.),
//!   mirroring
//!   [`hash_cell`](prism_render_architecture::particle::spatial_hash::hash_cell).
//!   A `table_size` of `0` yields `0`, matching the golden early return.
//! * **Count** ([`count_cells`](GpuSpatialHash::count_cells)): one thread per
//!   flattened cell index, `atomicAdd`-ing `1u` into its bucket, mirroring
//!   [`count_cells`](prism_render_architecture::particle::spatial_hash::count_cells).
//!   Indices at or beyond `cell_count` are skipped, exactly as the golden skips
//!   out-of-range slots.
//!
//! The golden's `prefix_sum` / `scatter_stable` stay on the `CPU` as the
//! bucket-layout reference; the `neighbor_cell_offsets_3x3x3` cube is a compile
//! -time constant and needs no kernel. The host-side
//! [`UniformGrid::cell_count`](prism_render_architecture::particle::spatial_hash::UniformGrid::cell_count)
//! uses a `u64` saturating multiply purely as an overflow guard for enormous
//! grids; that is **not** twinned — `WGSL` has no `u64`, and the two ported
//! kernels are pure `u32`/`i32` arithmetic.
//!
//! # Wrap semantics
//!
//! The golden mixes the signed coordinates with `i32::wrapping_mul` and then
//! reinterprets the bit pattern to `u32` (`from_ne_bytes`) before the modulus.
//! The twin performs the identical mix in the `u32` domain — `bitcast<u32>` on
//! each coordinate, multiply by the same prime bit patterns, `xor` — because
//! the low `32` bits of a two's-complement signed multiply equal the low `32`
//! bits of the unsigned multiply of the same bit patterns, and `WGSL` `u32`
//! arithmetic wraps modulo `2^32`. The resulting key is therefore bit-identical
//! to the golden's `u32::from_ne_bytes(mixed.to_ne_bytes())`, and `key %
//! table_size` matches the reference exactly.
//!
//! # `atomicAdd` versus `saturating_add`
//!
//! The `CPU` golden accumulates with `saturating_add` so a pathological input
//! can never wrap a bucket count; the `WGSL` kernel accumulates with
//! `atomicAdd`, which wraps on `u32` overflow. This is a deliberate, documented
//! difference, observable only when a single bucket would exceed `2^32`
//! entities — far above any real-device parity fixture — so the two
//! accumulators are equivalent at test scale and the parity test keeps every
//! bucket count well under `u32::MAX`.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `bitcast`, integer
//! `* ^`, `%`, a `<` compare and `atomicAdd` — with no transcendental call, no
//! optional device feature and no `u64`, so they run unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Both kernels are pure integer arithmetic with no rounding anywhere, so `CPU`
//! and `GPU` must agree bit for bit. The parity test asserts an exact `==` on
//! every hash slot and every bucket count, with no tolerance: any mismatch is a
//! genuine port bug (a wrong prime, a dropped `xor`, a miscounted bucket).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::spatial_hash`；无第三方引擎源码或衍生代码。
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

/// Core-`WGSL` spatial-hash kernels, inlined so this twin lives entirely in the
/// crate with no external `.wgsl`/`.wesl` asset. Two entry points mirror the
/// `CPU` golden
/// [`hash_cell`](prism_render_architecture::particle::spatial_hash::hash_cell)
/// and
/// [`count_cells`](prism_render_architecture::particle::spatial_hash::count_cells);
/// the mixing primes are copied verbatim from the golden `HASH_P1`/`HASH_P2`/
/// `HASH_P3`.
const SPATIAL_HASH_WGSL: &str = r#"
// Spatial-hash build twin: one thread per element. Two entry points mirror the
// CPU golden `particle::spatial_hash`: `hash_cell` folds a signed cell
// coordinate into [0, table_size); `count_cells` atomicAdds each flattened cell
// index into its bucket.
//
// The hash mix runs in the u32 domain (bitcast each coordinate, multiply by the
// prime bit patterns, xor) so the key is bit-identical to the golden's
// `i32::wrapping_mul` followed by `u32::from_ne_bytes`: the low 32 bits of a
// signed multiply equal the low 32 bits of the unsigned multiply of the same
// bits, and WGSL u32 arithmetic wraps modulo 2^32. Accumulation uses atomicAdd
// (wraps on overflow) where the CPU uses saturating_add; equivalent below 2^32
// counts.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::spatial_hash；无第三方
// 引擎源码或衍生代码。

// Mixing primes, the bit patterns of the golden i32 HASH_P1/HASH_P2/HASH_P3
// (all positive, so the u32 value equals the signed value).
const HASH_P1: u32 = 73856093u;
const HASH_P2: u32 = 19349663u;
const HASH_P3: u32 = 83492791u;

struct Params {
    // Number of valid elements; threads past this short-circuit.
    count: u32,
    // Hash kernel: the hash-table size (modulus); 0 yields 0.
    // Count kernel: the cell count (indices at or beyond it are skipped).
    arg: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Hash kernel bindings (group 0): flat x,y,z triples in, one slot out.
@group(0) @binding(1) var<storage, read> coords: array<i32>;
@group(0) @binding(2) var<storage, read_write> hashes: array<u32>;
// Count kernel bindings (group 0): flattened cell indices in, bucket counts out.
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn hash_cell_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let table_size = params.arg;
    if (table_size == 0u) {
        // Golden `hash_cell` returns 0 for a zero-sized table; also avoids a
        // modulo by zero.
        hashes[idx] = 0u;
        return;
    }
    let base = idx * 3u;
    // Mix in the u32 domain; see the module header for the wrap-equivalence
    // argument with the golden `i32::wrapping_mul` + `from_ne_bytes`.
    let kx = bitcast<u32>(coords[base + 0u]) * HASH_P1;
    let ky = bitcast<u32>(coords[base + 1u]) * HASH_P2;
    let kz = bitcast<u32>(coords[base + 2u]) * HASH_P3;
    let key = kx ^ ky ^ kz;
    hashes[idx] = key % table_size;
}

@compute @workgroup_size(64)
fn count_cells_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let slot = indices[idx];
    if (slot < params.arg) {
        atomicAdd(&counts[slot], 1u);
    }
}
"#;

/// Uniform parameters for one dispatch: the element `count` and the kernel
/// argument (`table_size` for the hash kernel, `cell_count` for the count
/// kernel), padded to a `16`-byte, `std140`-aligned uniform struct matching
/// `Params` in [`SPATIAL_HASH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements in the input buffer.
    count: u32,
    /// `table_size` (hash) or `cell_count` (count).
    arg: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// A compiled, reusable pair of spatial-hash kernels (hash and count), mirroring
/// the `CPU` golden
/// [`spatial_hash`](prism_render_architecture::particle::spatial_hash) building
/// blocks tap for tap.
pub struct GpuSpatialHash {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    hash_layout: BindGroupLayout,
    count_layout: BindGroupLayout,
    pipeline_hash: ComputePipeline,
    pipeline_count: ComputePipeline,
}

impl GpuSpatialHash {
    /// Compiles the two spatial-hash kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpatialHash {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_spatial_hash"),
            source: ShaderSource::Wgsl(SPATIAL_HASH_WGSL.into()),
        });
        // The hash entry point uses bindings 0,1,2; the count entry point uses
        // bindings 0,3,4. Each kernel gets its own layout holding only the
        // bindings it touches, so no dummy buffer is ever bound.
        let hash_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let count_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let hash_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_pipeline_layout"),
            bind_group_layouts: &[Some(&hash_layout)],
            immediate_size: 0,
        });
        let count_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_pipeline_layout"),
            bind_group_layouts: &[Some(&count_layout)],
            immediate_size: 0,
        });
        let pipeline_hash = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_pipeline"),
            layout: Some(&hash_pipeline_layout),
            module: &module,
            entry_point: Some("hash_cell_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_count = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_pipeline"),
            layout: Some(&count_pipeline_layout),
            module: &module,
            entry_point: Some("count_cells_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpatialHash {
            module,
            hash_layout,
            count_layout,
            pipeline_hash,
            pipeline_count,
        }
    }

    /// Hashes each cell coordinate into a slot in `[0, table_size)`, one thread
    /// per cell, returning a `Vec<u32>` whose entry `i` equals
    /// [`hash_cell`](prism_render_architecture::particle::spatial_hash::hash_cell)
    /// `(cells[i], table_size)`, exactly (see the module-level correctness
    /// model). A `table_size` of `0` yields all-zero slots, matching the golden
    /// early return. An empty input issues **no dispatch** — a storage buffer
    /// cannot be zero-sized — and returns an empty vector.
    #[must_use]
    pub fn hash_cells(&self, ctx: &GpuContext, cells: &[[i32; 3]], table_size: u32) -> Vec<u32> {
        let count = cells.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            arg: table_size,
            pad0: 0,
            pad1: 0,
        };
        // Flatten the x,y,z triples into a single i32 storage array.
        let mut flat: Vec<i32> = Vec::with_capacity(count * 3);
        for cell in cells {
            flat.extend_from_slice(cell);
        }

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let coords_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spatial_hash_coords"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (count * size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spatial_hash_hashes"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_bind_group"),
            layout: &self.hash_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: coords_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_spatial_hash_hash_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spatial_hash_hash_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_hash);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
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
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(result.len(), count);
        result
    }

    /// Counts how many entries land in each cell, one thread per entry, each
    /// `atomicAdd`-ing `1u` into its bucket. Returns a `Vec<u32>` of length
    /// `cell_count` whose entry `k` equals the count
    /// [`count_cells`](prism_render_architecture::particle::spatial_hash::count_cells)
    /// produces for cell `k`, exactly (see the module-level correctness model).
    /// Indices at or beyond `cell_count` are skipped. A `cell_count` of `0`
    /// returns an empty vector; an empty input issues **no dispatch** and
    /// returns the all-zero histogram of length `cell_count`.
    #[must_use]
    pub fn count_cells(&self, ctx: &GpuContext, indices: &[u32], cell_count: u32) -> Vec<u32> {
        let len = usize::try_from(cell_count).unwrap_or(usize::MAX);
        // No cell to count into: a storage buffer cannot be zero-sized, and the
        // golden returns an empty vector for a zero cell count.
        if len == 0 {
            return Vec::new();
        }
        // No entry to dispatch: return the correctly sized zero histogram.
        if indices.is_empty() {
            return vec![0u32; len];
        }
        let device = ctx.device();

        let params = GpuParams {
            count: indices.len() as u32,
            arg: cell_count,
            pad0: 0,
            pad1: 0,
        };
        let out_bytes = (len * size_of::<u32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let indices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spatial_hash_indices"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::STORAGE,
        });
        // Zero-initialized explicitly so every `atomicAdd` accumulates from `0`
        // regardless of the backend's buffer-clearing policy.
        let zeros = vec![0u32; len];
        let counts_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spatial_hash_counts"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_bind_group"),
            layout: &self.count_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: indices_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: counts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_spatial_hash_count_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spatial_hash_count_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_count);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per entry, flattened to a 1-D dispatch.
            let groups = (indices.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&counts_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(result.len(), len);
        result
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
