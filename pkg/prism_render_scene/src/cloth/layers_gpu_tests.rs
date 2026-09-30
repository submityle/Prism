//! Real-machine GPU coverage of the inter-layer garment coupling kernel in
//! `cloth_layers.wesl`: the `cloth_layer_coupling` entry.
//!
//! CPU golden:
//! `prism_render_architecture::cloth::layers_jacobi` -- the own-slot Jacobi form
//! of the Gauss-Seidel core in `cloth::layers`, which keeps stacked garments from
//! interpenetrating while preserving their inner-to-outer stacking order. The
//! Jacobi golden is exposed only through `resolve_layer_coupling_jacobi`, which
//! applies each particle's accumulated correction in place; this module recovers
//! the per-particle correction as `after - before` and asserts the on-device
//! `corrections[a]` matches it.
//!
//! The kernel does no spatial hashing itself: the host prebuilds, in the exact
//! golden traversal order, each particle's cross-layer candidate neighbours and
//! uploads them as a `CSR` adjacency (offsets + entries), mirroring how the
//! aerodynamics gather kernel consumes its vertex-triangle `CSR`. Same-layer
//! pairs and the self index are filtered out on the host, so the kernel only
//! sums half corrections along its own row.
//!
//! The parity fixtures use a single cross-layer contact per particle with
//! axis-aligned normals and deltas, so the float reduction is order-free and the
//! reciprocal-sqrt lands exact -- the on-device result equals the golden within a
//! few ULP, far inside `PARITY_EPS`.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter
//! returns `None` from `try_compute_device`, so each on-device test prints a skip
//! note and passes. A device-free compile test still guards that the shader
//! parses everywhere.

use alloc::collections::BTreeMap;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::layers::LayerParams;
use prism_render_architecture::cloth::layers_jacobi::resolve_layer_coupling_jacobi;
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::GpuClothLayerParams;
use super::gpu_test_support::{
    compile_layers_wgsl, find_entry_point, storage_from_slice, try_compute_device, PARITY_EPS,
};

/// A free particle (unit inverse mass) at `(x, y, z)`.
fn free(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 1.0)
}

/// A pinned particle (zero inverse mass) at `(x, y, z)`.
fn pinned(x: f32, y: f32, z: f32) -> ClothParticle {
    ClothParticle::new(Vec3::new(x, y, z), 0.0)
}

/// Host upload layout for the particle buffer: `xyz` = position, `w` = inverse
/// mass, matching `layer_positions` in the WESL.
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// Host upload layout for the outward normals: `xyz` = normal, `w` = 0.
fn upload_normals(normals: &[Vec3]) -> Vec<[f32; 4]> {
    normals.iter().map(|n| [n.x, n.y, n.z, 0.0]).collect()
}

/// Replicates `cloth::layers::cell_of` (which is crate-private in the
/// architecture crate): maps a world position to its saturating integer
/// spatial-hash cell for a positive `cell_size`.
fn cell_of(pos: [f32; 4], cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    let cx = (pos[0] * inv).floor() as i32;
    let cy = (pos[1] * inv).floor() as i32;
    let cz = (pos[2] * inv).floor() as i32;
    (cx, cy, cz)
}

/// Prebuilds the per-particle cross-layer `CSR` neighbourhood in the exact golden
/// traversal order: bucket every particle by its frozen cell, then for each
/// particle walk its 27-cell neighbourhood (dx, dy, dz each in -1..=1) in
/// ascending bucket order, skipping the self index and same-layer neighbours.
/// Returns `(offsets, entries)` where `offsets` has `positions.len() + 1` rows.
fn build_layer_csr(
    positions: &[[f32; 4]],
    layer_of: &[u32],
    params: LayerParams,
) -> (Vec<u32>, Vec<u32>) {
    let params = params.sanitized();
    let n = positions.len();
    let mut offsets: Vec<u32> = Vec::with_capacity(n + 1);
    let mut entries: Vec<u32> = Vec::new();

    if params.thickness <= 0.0 || params.cell_size <= 0.0 || n < 2 {
        offsets.resize(n + 1, 0);
        return (offsets, entries);
    }

    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, pos) in positions.iter().enumerate() {
        if index < layer_of.len() {
            let cell = cell_of(*pos, params.cell_size);
            grid.entry(cell).or_default().push(index as u32);
        }
    }

    for a in 0..n as u32 {
        offsets.push(entries.len() as u32);
        let ai = a as usize;
        if ai >= layer_of.len() {
            continue;
        }
        let cell = cell_of(positions[ai], params.cell_size);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    let Some(nbucket) = grid.get(&neighbor) else {
                        continue;
                    };
                    for &b in nbucket {
                        if b == a {
                            continue;
                        }
                        let bi = b as usize;
                        if layer_of[ai] == layer_of[bi] {
                            continue;
                        }
                        entries.push(b);
                    }
                }
            }
        }
    }
    offsets.push(entries.len() as u32);
    (offsets, entries)
}

/// The group-0 layout: five read-only storage buffers (positions, layer numbers,
/// normals, CSR offsets, CSR entries), a read-write correction buffer and the
/// `params` uniform, matching `@binding(0..6)` in the WESL.
fn layer_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
    let storage = |binding: u32, read_only: bool| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("cloth_layers_parity_group0"),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, true),
            storage(3, true),
            storage(4, true),
            storage(5, false),
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Uploads the fixture and runs one `cloth_layer_coupling` dispatch, returning
/// each particle's `xyz` position correction read back from the device.
#[expect(
    clippy::too_many_arguments,
    reason = "真机 parity replay 需一次上传位置/层号/法向/CSR 双缓冲/参数，拆结构反更难读"
)]
fn run_layer_coupling_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    layer_of: &[u32],
    normals: &[[f32; 4]],
    offsets: &[u32],
    entries: &[u32],
    params: GpuClothLayerParams,
) -> Vec<[f32; 3]> {
    let layout = layer_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_layers_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_layers_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_layer_coupling_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let count = positions.len();
    let positions_buf = storage_from_slice(device, "cloth_layers_positions", positions, [0.0f32; 4]);
    let layer_buf = storage_from_slice(device, "cloth_layers_layer_of", layer_of, 0u32);
    let normals_buf = storage_from_slice(device, "cloth_layers_normals", normals, [0.0f32; 4]);
    let offsets_buf = storage_from_slice(device, "cloth_layers_offsets", offsets, 0u32);
    let entries_buf = storage_from_slice(device, "cloth_layers_entries", entries, 0u32);

    let corr_init = vec![[0.0f32; 4]; count.max(1)];
    let corrections_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_layers_corrections"),
        contents: bytemuck::cast_slice(&corr_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_layers_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_layers_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: layer_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: normals_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: offsets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: entries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: corrections_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage_bytes = (count.max(1) * size_of::<[f32; 4]>()) as u64;
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_layers_corrections_stage"),
        size: stage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_layers_parity_encoder"),
    });
    let groups = (count as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_layer_coupling_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&corrections_buf, 0, &stage, 0, stage_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted layer-coupling work");
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped layer correction readback range should be available after poll");
    let raw = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    stage.unmap();
    raw.into_iter()
        .take(count)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Runs the Jacobi golden and the on-device twin over the same fixture, then
/// asserts every per-particle correction agrees within `PARITY_EPS`. The golden
/// correction is recovered as the in-place displacement
/// `resolve_layer_coupling_jacobi` applies (`after - before`), which is exactly
/// the `corrections[a]` the kernel writes.
fn assert_layer_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) {
    let mut moved = particles.to_vec();
    resolve_layer_coupling_jacobi(&mut moved, layer_of, normals, params);
    let golden: Vec<[f32; 3]> = moved
        .iter()
        .zip(particles.iter())
        .map(|(a, b)| {
            [
                a.position.x - b.position.x,
                a.position.y - b.position.y,
                a.position.z - b.position.z,
            ]
        })
        .collect();

    let positions = upload_positions(particles);
    let normals_up = upload_normals(normals);
    let (offsets, entries) = build_layer_csr(&positions, layer_of, params);
    let gpu_params = GpuClothLayerParams {
        thickness: params.sanitized().thickness,
        particle_count: particles.len() as u32,
        pad0: 0,
        pad1: 0,
    };
    let entry = find_entry_point(wgsl, "layer_coupling");
    let gpu = run_layer_coupling_on_gpu(
        device,
        queue,
        wgsl,
        &entry,
        &positions,
        layer_of,
        &normals_up,
        &offsets,
        &entries,
        gpu_params,
    );

    assert_eq!(gpu.len(), golden.len(), "correction readback length mismatch");
    for (i, (g, c)) in gpu.iter().zip(golden.iter()).enumerate() {
        for axis in 0..3 {
            assert!(
                (g[axis] - c[axis]).abs() <= PARITY_EPS,
                "particle {i} axis {axis}: gpu {} drifted from golden {} beyond {PARITY_EPS}",
                g[axis],
                c[axis]
            );
        }
    }
}

/// The shader must parse and expose its entry point on every host, including
/// headless CI with no GPU adapter. No device is needed, so this never skips.
#[test]
fn cloth_layers_wesl_compiles() {
    let wgsl = compile_layers_wgsl();
    let _ = find_entry_point(&wgsl, "layer_coupling");
}

/// A single oriented cross-layer contact (inner carries a `+Y` normal, outer
/// sunk below it) must be pushed apart exactly like the Jacobi golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn oriented_contact_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_layers oriented_contact: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_layers_wgsl();
    let particles = [free(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let params = LayerParams {
        thickness: 0.1,
        cell_size: 0.2,
    };
    assert_layer_parity(&device, &queue, &wgsl, &particles, &layer_of, &normals, params);
}

/// The radial fallback (no usable inner normal) along an axis-aligned delta must
/// also match the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn radial_contact_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_layers radial_contact: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_layers_wgsl();
    let particles = [free(0.0, 0.0, 0.0), free(0.02, 0.0, 0.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::ZERO, Vec3::ZERO];
    let params = LayerParams {
        thickness: 0.1,
        cell_size: 0.2,
    };
    assert_layer_parity(&device, &queue, &wgsl, &particles, &layer_of, &normals, params);
}

/// A pinned inner particle takes none of the correction; the whole push lands on
/// the outer particle, matching the golden's mass weighting.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn pinned_inner_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_layers pinned_inner: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_layers_wgsl();
    let particles = [pinned(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let params = LayerParams {
        thickness: 0.1,
        cell_size: 0.2,
    };
    assert_layer_parity(&device, &queue, &wgsl, &particles, &layer_of, &normals, params);
}

/// Same-layer particles never couple: the golden leaves them untouched and the
/// host CSR emits no cross-layer neighbours, so every correction is zero.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "on a host with no wgpu device the skip note must reach the test log"
)]
fn same_layer_no_correction() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("cloth_layers same_layer: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_layers_wgsl();
    let particles = [free(0.0, 0.0, 0.0), free(0.0, 0.01, 0.0)];
    let layer_of = [2u32, 2u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    let params = LayerParams {
        thickness: 0.1,
        cell_size: 0.2,
    };
    assert_layer_parity(&device, &queue, &wgsl, &particles, &layer_of, &normals, params);
}
