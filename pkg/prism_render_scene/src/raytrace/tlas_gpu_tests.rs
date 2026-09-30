//! Real-device `GPU` parity coverage for the top-level `TLAS` traversal kernel.
//!
//! The sibling [`shader_tests`](super::shader_tests) module proves
//! `shaders/tlas_traverse.wesl` parses and type-checks through the render
//! world's [`ShaderCache`], guarding the kernel's *shape*. These tests close the
//! behavioural gap: they build a real two-level scene — several distinct
//! [`Bvh`] bottom-level structures pooled by [`GpuBlasPool::from_blases`] and a
//! [`Tlas`] of affine-transformed [`Instance`]s packed by
//! [`GpuTlasBuffers::from_tlas`] — bind the actual `tlas_traverse` compute
//! pipeline on a live `Metal` (or any native `wgpu`) device, dispatch a batch of
//! world-space rays, read the hits back and assert them ray-for-ray against the
//! `CPU` golden walks [`GpuTlasBuffers::closest_hit`] and
//! [`GpuTlasBuffers::any_hit`].
//!
//! Because the `WESL` kernel and its packed `CPU` twin share byte-identical
//! arithmetic — the per-instance `world_to_object` affine transform (linear
//! part carried without renormalizing so `t` is preserved), the reciprocal-slab
//! rejection, the near/far child ordering, the cross-instance `best_t` shrink
//! and the double-sided Möller–Trumbore test — a green run is direct on-device
//! evidence the ported two-level walk matches its reference to `float32`
//! rounding, not merely that it compiles.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter [`try_solver_device`] returns `None` and the test skips with a
//! printed notice instead of failing.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance as WgpuInstance, InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions,
    PollType, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

use prism_render_architecture::ray_scene::{
    Affine3, Bvh, BvhBuildConfig, GpuBlasPool, GpuTlasBuffers, Instance, Ray, Tlas, Triangle,
};

use super::abi::{
    GpuRayTraverseParams, MISS_PRIMITIVE, POSITIVE_INF_BITS, RAYTRACE_MODE_ANY,
    RAYTRACE_MODE_CLOSEST, RAY_WORDS, TLAS_HIT_WORDS,
};

/// Absolute per-scalar tolerance for the `GPU`-versus-`CPU` `t`/`u`/`v`
/// comparison; the object-space affine transform adds a few fused-multiply-add
/// contractions on top of the bottom-level walk, still far inside this margin.
const PARITY_EPS: f32 = 1.0e-3;

/// Tiny deterministic `xorshift` `RNG`, byte-reproducible across hosts.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() as f32) / (u32::MAX as f32);
        lo + (hi - lo) * unit
    }

    fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }
}

/// Builds one deterministic bottom-level triangle soup inside a small box,
/// forcing a genuine multi-level `BVH` (interior nodes plus several leaves).
fn build_blas_soup(seed: u64, count: u32) -> Vec<Triangle> {
    let mut rng = Rng::new(seed);
    let mut triangles = Vec::new();
    let mut primitive = 0u32;
    while primitive < count {
        let center = rng.point(-1.5, 1.5);
        let e0 = rng.point(-0.4, 0.4);
        let e1 = rng.point(-0.4, 0.4);
        let v0 = center;
        let v1 = [center[0] + e0[0], center[1] + e0[1], center[2] + e0[2]];
        let v2 = [center[0] + e1[0], center[1] + e1[1], center[2] + e1[2]];
        triangles.push(Triangle::new(v0, v1, v2, primitive));
        primitive += 1;
    }
    triangles
}

/// A built two-level scene: the pooled `BLAS` buffers, the packed `TLAS`
/// buffers, the object→world transforms (kept to aim rays at instance space),
/// each instance's `BLAS` soup (kept to recover object-space centroids) and the
/// per-instance `BLAS` index.
struct Scene {
    pool: GpuBlasPool,
    tlas_buffers: GpuTlasBuffers,
    object_to_world: Vec<Affine3>,
    soups: Vec<Vec<Triangle>>,
    instance_blas: Vec<usize>,
}

/// Assembles two distinct `BLAS` soups and four affine-transformed instances
/// (translation, non-uniform scale and quaternion rotation) so the dispatch
/// exercises the full object-space transform path and cross-instance pruning.
fn build_scene() -> Scene {
    let soup0 = build_blas_soup(0x1111_2222_3333_4444, 18);
    let soup1 = build_blas_soup(0x5555_6666_7777_8888, 22);
    let blases = vec![Bvh::build(&soup0), Bvh::build(&soup1)];

    // object→world transforms: distinct placements so instances do not overlap
    // in world space, giving each ray an unambiguous nearest hit.
    let placements: Vec<(Affine3, usize, u32)> = vec![
        (Affine3::from_translation([6.0, 0.0, 0.0]), 0, 10),
        (
            Affine3::from_translation([-6.0, 0.0, 0.0]).compose(&Affine3::from_scale([
                1.5, 0.75, 1.25,
            ])),
            0,
            20,
        ),
        (
            Affine3::from_translation([0.0, 6.0, 0.0])
                .compose(&Affine3::from_quaternion([0.9239, 0.0, 0.3827, 0.0])),
            1,
            30,
        ),
        (
            Affine3::from_translation([0.0, -6.0, 3.0]).compose(&Affine3::from_scale([
                2.0, 2.0, 2.0,
            ])),
            1,
            40,
        ),
    ];

    let mut instances = Vec::new();
    let mut object_to_world = Vec::new();
    let mut instance_blas = Vec::new();
    for (m, blas, id) in placements {
        let inst = Instance::new(m, blas, id).expect("instance transform must be invertible");
        instances.push(inst);
        object_to_world.push(m);
        instance_blas.push(blas);
    }

    // Force a branching top-level hierarchy (max_leaf_primitives = 1) so the
    // four instances split into internal nodes; this makes the dispatch
    // exercise the `TLAS` node stack instead of a single trivial leaf.
    let tlas = Tlas::build_with(
        &instances,
        &blases,
        BvhBuildConfig {
            max_leaf_primitives: 1,
            ..BvhBuildConfig::default()
        },
    );
    let tlas_buffers = GpuTlasBuffers::from_tlas(&tlas);
    let pool = GpuBlasPool::from_blases(&blases);

    // The `TLAS` reorders instances during the build; recover the packed order
    // so the ray-aim book-keeping tracks the buffer the kernel actually reads.
    let mut reordered_o2w = Vec::new();
    let mut reordered_blas = Vec::new();
    let mut reordered_soups = Vec::new();
    for inst in tlas.instances() {
        let id = inst.instance_id();
        // Find the original placement with this stable id.
        let orig = instances
            .iter()
            .position(|i| i.instance_id() == id)
            .expect("instance id must survive the build");
        reordered_o2w.push(object_to_world[orig]);
        reordered_blas.push(instance_blas[orig]);
        let soup = if instance_blas[orig] == 0 {
            soup0.clone()
        } else {
            soup1.clone()
        };
        reordered_soups.push(soup);
    }

    Scene {
        pool,
        tlas_buffers,
        object_to_world: reordered_o2w,
        soups: reordered_soups,
        instance_blas: reordered_blas,
    }
}

/// Builds a batch of world-space rays that provably exercises both hit and miss
/// paths: one ray aimed at each instance's transformed triangle centroid, plus
/// guaranteed misses (a too-short interval and a ray pointing away).
fn build_rays(scene: &Scene) -> Vec<Ray> {
    let mut rng = Rng::new(0x0BAD_F00D_C0FF_EE00);
    let mut rays = Vec::new();

    for (idx, o2w) in scene.object_to_world.iter().enumerate() {
        let soup = &scene.soups[idx];
        // Aim at a handful of triangles per instance to land many closest-hits.
        for tri in soup.iter().take(6) {
            let c_obj = [
                (tri.v0[0] + tri.v1[0] + tri.v2[0]) / 3.0,
                (tri.v0[1] + tri.v1[1] + tri.v2[1]) / 3.0,
                (tri.v0[2] + tri.v1[2] + tri.v2[2]) / 3.0,
            ];
            let c = o2w.transform_point(c_obj);
            let origin = [
                c[0] + rng.range(-10.0, 10.0),
                c[1] + rng.range(-10.0, 10.0),
                c[2] + rng.range(-10.0, 10.0),
            ];
            let dir = [c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]];
            rays.push(Ray::new(origin, dir, 0.0, f32::INFINITY));
        }
    }

    // Guaranteed misses.
    rays.push(Ray::new([40.0, 40.0, 40.0], [1.0, 1.0, 1.0], 0.0, 0.5));
    rays.push(Ray::new([40.0, 40.0, 40.0], [1.0, 1.0, 1.0], 0.0, f32::INFINITY));
    rays.push(Ray::new([0.0, 0.0, 40.0], [0.0, 0.0, 1.0], 0.0, f32::INFINITY));

    let _ = &scene.instance_blas;
    let _ = &scene.pool;
    rays
}

/// Packs the ray batch into the kernel's `RAY_WORDS`-stride `u32` buffer, using
/// the clamped [`Ray::t_min`]/[`Ray::t_max`] so the `CPU` golden and the `GPU`
/// walk see an identical interval and the un-normalized direction the kernel
/// reciprocates itself.
fn pack_rays(rays: &[Ray]) -> Vec<u32> {
    let mut words = vec![0u32; rays.len() * RAY_WORDS];
    for (i, ray) in rays.iter().enumerate() {
        let base = i * RAY_WORDS;
        let o = ray.origin();
        let d = ray.direction();
        words[base] = o[0].to_bits();
        words[base + 1] = o[1].to_bits();
        words[base + 2] = o[2].to_bits();
        words[base + 3] = ray.t_min().to_bits();
        words[base + 4] = d[0].to_bits();
        words[base + 5] = d[1].to_bits();
        words[base + 6] = d[2].to_bits();
        words[base + 7] = ray.t_max().to_bits();
    }
    words
}

/// A top-level hit decoded from the kernel's `TLAS_HIT_WORDS`-stride output.
#[derive(Clone, Copy, Debug)]
struct GpuTlasHit {
    t: f32,
    u: f32,
    v: f32,
    primitive: u32,
    instance_id: u32,
    instance_index: u32,
}

/// Streams the `Wgsl` source back out of the shader cache without a device.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("the TLAS traversal shader is WESL"),
    }
}

/// Compiles `tlas_traverse.wesl` and returns its `Wgsl` translation.
fn compile_tlas_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5449_4c41_5354_5256_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/tlas_traverse.wesl"),
            "embedded://prism_render_scene/shaders/tlas_traverse.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("tlas_traverse.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
fn find_entry_point(wgsl: &str, needle: &str) -> String {
    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains(needle) {
                return name.to_string();
            }
        }
    }
    panic!("no compute entry point containing `{needle}` in compiled Wgsl");
}

/// Best-effort acquisition of a native compute device and queue.
fn try_solver_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = WgpuInstance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Records the `tlas_traverse` dispatch for `mode` and reads the hits back.
///
/// Binds the packed top-level `tlas_nodes`/`instances`, the shared pool
/// `pool_nodes`/`pool_triangles`/`pool_offsets`, the packed `rays`, an `RW`
/// `hits` output and the [`GpuRayTraverseParams`] uniform on `@group(0)`
/// bindings 0..8, dispatches `ray_count.div_ceil(64)` workgroups and maps the
/// results back.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch mirrors the kernel's eight-binding group; grouping them into a struct would only obscure the one-to-one binding map"
)]
fn dispatch_tlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    scene: &Scene,
    ray_words: &[u32],
    ray_count: u32,
    mode: u32,
) -> Vec<GpuTlasHit> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("tlas_traverse_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("tlas_traverse_parity_pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let hit_words = ray_count as usize * TLAS_HIT_WORDS;
    let hit_bytes = (hit_words * size_of::<u32>()) as u64;

    let tlas_nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_nodes"),
        contents: bytemuck::cast_slice(&scene.tlas_buffers.nodes),
        usage: BufferUsages::STORAGE,
    });
    let instances_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_instances"),
        contents: bytemuck::cast_slice(&scene.tlas_buffers.instances),
        usage: BufferUsages::STORAGE,
    });
    let pool_nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_pool_nodes"),
        contents: bytemuck::cast_slice(&scene.pool.nodes),
        usage: BufferUsages::STORAGE,
    });
    let pool_triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_pool_triangles"),
        contents: bytemuck::cast_slice(&scene.pool.triangles),
        usage: BufferUsages::STORAGE,
    });
    let pool_offsets_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_pool_offsets"),
        contents: bytemuck::cast_slice(&scene.pool.offsets),
        usage: BufferUsages::STORAGE,
    });
    let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_rays"),
        contents: bytemuck::cast_slice(ray_words),
        usage: BufferUsages::STORAGE,
    });
    let hits_buf = device.create_buffer(&BufferDescriptor {
        label: Some("tlas_hits"),
        size: hit_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = GpuRayTraverseParams {
        ray_count,
        mode,
        pad0: 0,
        pad1: 0,
    };
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("tlas_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("tlas_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tlas_nodes_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: instances_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: pool_nodes_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: pool_triangles_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: pool_offsets_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: rays_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: hits_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("tlas_hits_stage"),
        size: hit_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("tlas_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("tlas_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(ray_count.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&hits_buf, 0, &stage, 0, hit_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let raw: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();

    let mut hits = Vec::with_capacity(ray_count as usize);
    for i in 0..ray_count as usize {
        let base = i * TLAS_HIT_WORDS;
        hits.push(GpuTlasHit {
            t: f32::from_bits(raw[base]),
            u: f32::from_bits(raw[base + 1]),
            v: f32::from_bits(raw[base + 2]),
            primitive: raw[base + 3],
            instance_id: raw[base + 4],
            instance_index: raw[base + 5],
        });
    }
    hits
}

/// Nearest-hit walk parity: `tlas_traverse` in `mode = 0` must reproduce
/// [`GpuTlasBuffers::closest_hit`] for every ray in the batch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn tlas_traverse_matches_closest_hit_on_device() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping tlas_traverse closest-hit parity: no wgpu adapter");
        return;
    };

    let scene = build_scene();
    assert!(
        scene.tlas_buffers.node_count() > 1,
        "scene must build a multi-node TLAS"
    );
    let rays = build_rays(&scene);
    let ray_words = pack_rays(&rays);

    let wgsl = compile_tlas_wgsl();
    let entry = find_entry_point(&wgsl, "tlas_traverse");
    let gpu = dispatch_tlas(
        &device,
        &queue,
        &wgsl,
        &entry,
        &scene,
        &ray_words,
        rays.len() as u32,
        RAYTRACE_MODE_CLOSEST,
    );

    let mut hit_count = 0usize;
    let mut miss_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu = scene.tlas_buffers.closest_hit(ray, &scene.pool);
        let g = gpu[i];
        match cpu {
            Some(hit) => {
                hit_count += 1;
                assert_eq!(
                    g.primitive, hit.primitive,
                    "ray {i}: GPU primitive {} != CPU {}",
                    g.primitive, hit.primitive
                );
                assert_eq!(
                    g.instance_id, hit.instance_id,
                    "ray {i}: GPU instance_id {} != CPU {}",
                    g.instance_id, hit.instance_id
                );
                assert_eq!(
                    g.instance_index, hit.instance_index,
                    "ray {i}: GPU instance_index {} != CPU {}",
                    g.instance_index, hit.instance_index
                );
                assert!(
                    (g.t - hit.t).abs() <= PARITY_EPS,
                    "ray {i}: GPU t {} != CPU t {}",
                    g.t,
                    hit.t
                );
                assert!(
                    (g.u - hit.u).abs() <= PARITY_EPS && (g.v - hit.v).abs() <= PARITY_EPS,
                    "ray {i}: GPU uv ({},{}) != CPU ({},{})",
                    g.u,
                    g.v,
                    hit.u,
                    hit.v
                );
            }
            None => {
                miss_count += 1;
                assert_eq!(
                    g.primitive, MISS_PRIMITIVE,
                    "ray {i}: CPU missed but GPU reported primitive {}",
                    g.primitive
                );
                assert_eq!(
                    g.t.to_bits(),
                    POSITIVE_INF_BITS,
                    "ray {i}: GPU miss must write +inf t"
                );
            }
        }
    }
    assert!(hit_count > 0, "batch must exercise the hit path");
    assert!(miss_count > 0, "batch must exercise the miss path");
}

/// Occlusion-walk parity: `tlas_traverse` in `mode = 1` must agree with
/// [`GpuTlasBuffers::any_hit`] on whether each ray is blocked.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn tlas_traverse_matches_any_hit_on_device() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping tlas_traverse any-hit parity: no wgpu adapter");
        return;
    };

    let scene = build_scene();
    let rays = build_rays(&scene);
    let ray_words = pack_rays(&rays);

    let wgsl = compile_tlas_wgsl();
    let entry = find_entry_point(&wgsl, "tlas_traverse");
    let gpu = dispatch_tlas(
        &device,
        &queue,
        &wgsl,
        &entry,
        &scene,
        &ray_words,
        rays.len() as u32,
        RAYTRACE_MODE_ANY,
    );

    let mut blocked_count = 0usize;
    let mut clear_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu_blocked = scene.tlas_buffers.any_hit(ray, &scene.pool);
        let gpu_blocked = gpu[i].primitive != MISS_PRIMITIVE;
        assert_eq!(
            gpu_blocked, cpu_blocked,
            "ray {i}: GPU blocked={gpu_blocked} != CPU blocked={cpu_blocked}"
        );
        if cpu_blocked {
            blocked_count += 1;
        } else {
            clear_count += 1;
        }
    }
    assert!(blocked_count > 0, "batch must exercise the blocked path");
    assert!(clear_count > 0, "batch must exercise the clear path");
}
