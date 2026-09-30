//! Real-device `GPU` parity coverage for the ray-traversal kernel.
//!
//! The sibling [`shader_tests`](super::shader_tests) module proves
//! `shaders/ray_traverse.wesl` parses and type-checks through the render
//! world's [`ShaderCache`]. That guards the kernel's *shape*, but it never runs
//! it: a shader can compile cleanly and still walk the tree wrong. The tests
//! here close that gap by building a real [`Bvh`], packing it with
//! [`GpuBvhBuffers::from_bvh`] (the authoritative, float-audited layout owned by
//! `prism_render_architecture::ray_scene`), binding the actual `ray_traverse`
//! compute pipeline on a live `Metal` (or any native `wgpu`) device, dispatching
//! a batch of rays, reading the hits back, and asserting them ray-for-ray
//! against the `CPU` golden walks [`GpuBvhBuffers::closest_hit`] and
//! [`GpuBvhBuffers::any_hit`].
//!
//! Because the `WESL` kernel and the packed `CPU` twin share byte-identical
//! arithmetic — the reciprocal-slab rejection, the near/far child ordering by
//! split-axis sign, the running `t_max` shrink and the double-sided
//! Möller–Trumbore test (`EPS = 1e-8`) — a green run is direct on-device
//! evidence that the ported kernel matches its reference to `float32` rounding,
//! not merely that it compiles.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter [`try_solver_device`] returns `None` and the test skips with a
//! printed notice instead of failing, so the suite stays green everywhere while
//! still exercising the full dispatch on any machine with a real device.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

use prism_render_architecture::ray_scene::{Bvh, GpuBvhBuffers, Ray, Triangle};

use super::abi::{
    GpuRayTraverseParams, HIT_WORDS, MISS_PRIMITIVE, POSITIVE_INF_BITS, RAYTRACE_MODE_ANY,
    RAYTRACE_MODE_CLOSEST, RAY_WORDS,
};

/// Absolute per-scalar tolerance for the `GPU`-versus-`CPU` `t`/`u`/`v`
/// comparison.
///
/// The two paths run the same `float32` arithmetic, so agreement is far tighter
/// than this in practice; the margin only absorbs a driver's fused-multiply-add
/// contraction and reordering freedom.
const PARITY_EPS: f32 = 1.0e-3;

/// Streams the `Wgsl` source back out of the shader cache without a device.
///
/// Mirrors the closure [`shader_tests`](super::shader_tests) uses so the `WESL`
/// is composed through the exact render-world pipeline; here we keep the
/// compiled `Wgsl` string (rather than a device module) so the parity test can
/// hand it to a raw `wgpu` device it created itself.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("the ray-traversal shader is WESL"),
    }
}

/// Compiles `ray_traverse.wesl` and returns its `Wgsl` translation.
fn compile_traverse_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5241_5954_5241_5645_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/ray_traverse.wesl"),
            "embedded://prism_render_scene/shaders/ray_traverse.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("ray_traverse.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
///
/// The `WESL` compiler may prefix module-local names, so the parity test locates
/// the `ray_traverse` entry by substring rather than assuming a fixed symbol.
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
///
/// Returns `None` (rather than panicking) when no adapter is available so the
/// suite stays green on headless hosts; on a machine with a real `GPU` this
/// yields a live device the parity test dispatches against.
fn try_solver_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
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

/// Tiny deterministic `xorshift` `RNG` so the scene is byte-reproducible across
/// runs and hosts without pulling in a dependency.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    /// Uniform `f32` in `[lo, hi)` using only the four basic operators.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() as f32) / (u32::MAX as f32);
        lo + (hi - lo) * unit
    }

    fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }
}

/// Builds a deterministic, non-trivial triangle soup.
///
/// A few dozen small randomly placed triangles inside a box force a genuine
/// multi-level `BVH` (interior nodes plus several leaves), so the dispatch
/// exercises the stack, the split-axis near/far ordering and the running
/// `t_max` shrink rather than a single-leaf trivial walk.
fn build_scene() -> Vec<Triangle> {
    let mut rng = Rng::new(0x00C0_FFEE_1234_5678);
    let mut triangles = Vec::new();
    let mut primitive = 0u32;
    let count = 96;
    while primitive < count {
        let center = rng.point(-6.0, 6.0);
        let e0 = rng.point(-0.7, 0.7);
        let e1 = rng.point(-0.7, 0.7);
        let v0 = center;
        let v1 = [center[0] + e0[0], center[1] + e0[1], center[2] + e0[2]];
        let v2 = [center[0] + e1[0], center[1] + e1[1], center[2] + e1[2]];
        triangles.push(Triangle::new(v0, v1, v2, primitive));
        primitive += 1;
    }
    triangles
}

/// Builds a batch of rays that provably exercises both the hit and the miss
/// paths.
///
/// The first block aims a ray from a random origin straight at each triangle's
/// centroid (so the batch lands many closest-hits, some occluded by a nearer
/// triangle); the second block fires rays whose interval is too short to reach
/// anything and rays pointing away from the soup (guaranteed misses).
fn build_rays(triangles: &[Triangle]) -> Vec<Ray> {
    let mut rng = Rng::new(0x0BAD_F00D_DEAD_BEEF);
    let mut rays = Vec::new();

    for tri in triangles {
        let c = [
            (tri.v0[0] + tri.v1[0] + tri.v2[0]) / 3.0,
            (tri.v0[1] + tri.v1[1] + tri.v2[1]) / 3.0,
            (tri.v0[2] + tri.v1[2] + tri.v2[2]) / 3.0,
        ];
        let origin = [
            rng.range(-14.0, 14.0),
            rng.range(-14.0, 14.0),
            rng.range(-14.0, 14.0),
        ];
        let dir = [c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]];
        rays.push(Ray::new(origin, dir, 0.0, f32::INFINITY));
    }

    // Guaranteed misses: a ray whose finite `t_max` stops well short of the
    // soup, and a ray pointing away from it.
    rays.push(Ray::new([20.0, 20.0, 20.0], [1.0, 1.0, 1.0], 0.0, 0.5));
    rays.push(Ray::new([20.0, 20.0, 20.0], [1.0, 1.0, 1.0], 0.0, f32::INFINITY));
    rays.push(Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0, 200.0));

    rays
}

/// Packs the ray batch into the kernel's `RAY_WORDS`-stride `u32` buffer.
///
/// The interval fields use the *clamped* [`Ray::t_min`]/[`Ray::t_max`] so the
/// `CPU` golden and the `GPU` walk see the identical interval; direction is left
/// un-normalized exactly as [`Ray::new`] stores it (the shader derives
/// `inv_dir = 1.0 / dir`, matching the `CPU` `reciprocal`).
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

/// A hit decoded from the kernel's `HIT_WORDS`-stride output buffer.
#[derive(Clone, Copy, Debug)]
struct GpuHit {
    t: f32,
    u: f32,
    v: f32,
    primitive: u32,
}

/// Records the `ray_traverse` dispatch for `mode` and reads the hits back.
///
/// Binds the packed `nodes`/`triangles`, the packed `rays`, an `RW` `hits`
/// output and the [`GpuRayTraverseParams`] uniform on `@group(0)` bindings 0..5,
/// dispatches `ray_count.div_ceil(64)` workgroups and maps the results back.
fn dispatch_traverse(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    buffers: &GpuBvhBuffers,
    ray_words: &[u32],
    ray_count: u32,
    mode: u32,
) -> Vec<GpuHit> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray_traverse_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("ray_traverse_parity_pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let hit_words = ray_count as usize * HIT_WORDS;
    let hit_bytes = (hit_words * size_of::<u32>()) as u64;

    let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("rt_nodes"),
        contents: bytemuck::cast_slice(&buffers.nodes),
        usage: BufferUsages::STORAGE,
    });
    let triangles_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("rt_triangles"),
        contents: bytemuck::cast_slice(&buffers.triangles),
        usage: BufferUsages::STORAGE,
    });
    let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("rt_rays"),
        contents: bytemuck::cast_slice(ray_words),
        usage: BufferUsages::STORAGE,
    });
    let hits_buf = device.create_buffer(&BufferDescriptor {
        label: Some("rt_hits"),
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
        label: Some("rt_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("rt_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: nodes_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: triangles_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: rays_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: hits_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("rt_hits_stage"),
        size: hit_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("rt_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("rt_parity_pass"),
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
        let base = i * HIT_WORDS;
        hits.push(GpuHit {
            t: f32::from_bits(raw[base]),
            u: f32::from_bits(raw[base + 1]),
            v: f32::from_bits(raw[base + 2]),
            primitive: raw[base + 3],
        });
    }
    hits
}

/// Nearest-hit walk parity: `ray_traverse` in `mode = 0` must reproduce
/// [`GpuBvhBuffers::closest_hit`] for every ray in the batch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn ray_traverse_matches_closest_hit_on_device() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping ray_traverse closest-hit parity: no wgpu adapter");
        return;
    };

    let triangles = build_scene();
    let bvh = Bvh::build(&triangles);
    assert!(bvh.node_count() > 1, "scene must build a multi-node BVH");
    let buffers = GpuBvhBuffers::from_bvh(&bvh);
    let rays = build_rays(&triangles);
    let ray_words = pack_rays(&rays);

    let wgsl = compile_traverse_wgsl();
    let entry = find_entry_point(&wgsl, "ray_traverse");
    let gpu = dispatch_traverse(
        &device,
        &queue,
        &wgsl,
        &entry,
        &buffers,
        &ray_words,
        rays.len() as u32,
        RAYTRACE_MODE_CLOSEST,
    );

    let mut hit_count = 0usize;
    let mut miss_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu = buffers.closest_hit(ray);
        let g = gpu[i];
        match cpu {
            Some(hit) => {
                hit_count += 1;
                assert_eq!(
                    g.primitive, hit.primitive,
                    "ray {i}: GPU primitive {} != CPU {}",
                    g.primitive, hit.primitive
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

/// Occlusion-walk parity: `ray_traverse` in `mode = 1` must agree with
/// [`GpuBvhBuffers::any_hit`] on whether each ray is blocked.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn ray_traverse_matches_any_hit_on_device() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping ray_traverse any-hit parity: no wgpu adapter");
        return;
    };

    let triangles = build_scene();
    let bvh = Bvh::build(&triangles);
    let buffers = GpuBvhBuffers::from_bvh(&bvh);
    let rays = build_rays(&triangles);
    let ray_words = pack_rays(&rays);

    let wgsl = compile_traverse_wgsl();
    let entry = find_entry_point(&wgsl, "ray_traverse");
    let gpu = dispatch_traverse(
        &device,
        &queue,
        &wgsl,
        &entry,
        &buffers,
        &ray_words,
        rays.len() as u32,
        RAYTRACE_MODE_ANY,
    );

    let mut blocked_count = 0usize;
    let mut clear_count = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu_blocked = buffers.any_hit(ray);
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
