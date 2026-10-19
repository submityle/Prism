//! Bounded, standalone GPU software-BVH trace pass.
//!
//! The packed geometry ABI is the architecture crate's `GpuBvhBuffers` ABI:
//! 12 `u32` words per node and triangle. This module consumes those words
//! directly and performs a bounded stack traversal in WGSL.

use std::sync::Arc;

use prism_render_architecture::ray_scene::{GpuBvhBuffers, NODE_WORDS, TRIANGLE_WORDS};
use prism_render_driver as rhi;
use rhi::{RenderDevice, RenderQueue};

const WORKGROUP_SIZE: u32 = 64;
const MAX_STACK: usize = 64;
const MAX_NODE_COUNT: usize = 1_000_000;
const MAX_TRIANGLE_COUNT: usize = 2_000_000;
const MAX_RAY_COUNT: usize = 1_000_000;
const RAY_WORDS: usize = 8;
const HIT_WORDS: usize = 8;

/// Input ray, with a closed lower and upper parameter interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceRay {
    /// World-space ray origin.
    pub origin: [f32; 3],
    /// Inclusive lower ray parameter.
    pub t_min: f32,
    /// World-space ray direction; it need not be normalized.
    pub direction: [f32; 3],
    /// Inclusive upper ray parameter.
    pub t_max: f32,
}

/// Stable result. A miss has `hit == false`, `primitive == u32::MAX`, and zero
/// `t/u/v`; traversal faults are returned as errors, never disguised as misses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceHit {
    /// Whether the ray intersects a triangle in the interval.
    pub hit: bool,
    /// Stable primitive identifier, or `u32::MAX` for a miss.
    pub primitive: u32,
    /// Ray parameter of the nearest hit; zero for a miss.
    pub t: f32,
    /// First barycentric coordinate; zero for a miss.
    pub u: f32,
    /// Second barycentric coordinate; zero for a miss.
    pub v: f32,
}

impl TraceHit {
    const MISS: Self = Self {
        hit: false,
        primitive: u32::MAX,
        t: 0.0,
        u: 0.0,
        v: 0.0,
    };
}

/// Explicit limits and malformed-input errors for this bounded slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceGpuError {
    /// No nodes or triangles were supplied.
    EmptyGeometry,
    /// Packed arrays do not match the frozen ABI strides.
    MalformedPackedLength,
    /// Node or triangle count exceeds the bounded slice's limit.
    GeometryLimit,
    /// Ray count exceeds the bounded slice's limit.
    RayLimit,
    /// A node has invalid bounds, children, or primitive range.
    InvalidNode(usize),
    /// A triangle is non-finite or degenerate.
    InvalidTriangle(usize),
    /// A ray is non-finite or has an invalid interval/direction.
    InvalidRay(usize),
    /// The node records do not form one valid reachable tree.
    InvalidTree,
    /// The tree exceeds the bounded WGSL traversal stack.
    StackDepthExceeded,
    /// A zero-length ray batch cannot be dispatched.
    EmptyRays,
    /// A storage binding exceeds the device's declared limit.
    StorageBindingLimit,
    /// A packed input byte size cannot be represented without overflow.
    BufferSizeOverflow,
    /// The dispatch exceeds the device's workgroup limit.
    DispatchLimit,
    /// The caller-provided hit buffer is smaller than the packed hit output.
    OutputBufferTooSmall,
    /// The GPU detected a traversal safety-bound violation.
    TraversalFault(usize),
}

/// Validated packed scene. Words use the architecture crate's frozen ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedTraceScene {
    /// Flat `NODE_WORDS` records in depth-first order.
    pub nodes: Vec<u32>,
    /// Flat `TRIANGLE_WORDS` records indexed by leaf ranges.
    pub triangles: Vec<u32>,
}

impl From<&GpuBvhBuffers> for PackedTraceScene {
    fn from(value: &GpuBvhBuffers) -> Self {
        Self {
            nodes: value.nodes.clone(),
            triangles: value.triangles.clone(),
        }
    }
}

impl PackedTraceScene {
    /// Checks ABI lengths, finite ordered bounds, leaf ranges, triangle data,
    /// reachability, cycles, and the exact maximum traversal stack occupancy.
    pub fn validate(&self) -> Result<(), TraceGpuError> {
        if self.nodes.is_empty() || self.triangles.is_empty() {
            return Err(TraceGpuError::EmptyGeometry);
        }
        if self.nodes.len() % NODE_WORDS != 0 || self.triangles.len() % TRIANGLE_WORDS != 0 {
            return Err(TraceGpuError::MalformedPackedLength);
        }
        let node_count = self.nodes.len() / NODE_WORDS;
        let triangle_count = self.triangles.len() / TRIANGLE_WORDS;
        if node_count > MAX_NODE_COUNT || triangle_count > MAX_TRIANGLE_COUNT {
            return Err(TraceGpuError::GeometryLimit);
        }
        for n in 0..node_count {
            let b = n * NODE_WORDS;
            for a in 0..3 {
                let lo = f32::from_bits(self.nodes[b + a]);
                let hi = f32::from_bits(self.nodes[b + 3 + a]);
                if !lo.is_finite() || !hi.is_finite() || lo > hi {
                    return Err(TraceGpuError::InvalidNode(n));
                }
            }
            let count = self.nodes[b + 8];
            if count == 0 {
                let second = self.nodes[b + 7] as usize;
                if second <= n + 1
                    || second >= node_count
                    || self.nodes[b + 9] > 2
                    || n + 1 >= node_count
                {
                    return Err(TraceGpuError::InvalidNode(n));
                }
            } else {
                let start = self.nodes[b + 6] as usize;
                if start
                    .checked_add(count as usize)
                    .is_none_or(|end| end > triangle_count)
                {
                    return Err(TraceGpuError::InvalidNode(n));
                }
            }
        }
        for t in 0..triangle_count {
            let b = t * TRIANGLE_WORDS;
            let v = (0..9).all(|i| f32::from_bits(self.triangles[b + i]).is_finite());
            let a = vec3(&self.triangles, b);
            let e1 = sub(vec3(&self.triangles, b + 3), a);
            let e2 = sub(vec3(&self.triangles, b + 6), a);
            if !v || dot(cross(e1, e2), cross(e1, e2)) <= 1.0e-20 {
                return Err(TraceGpuError::InvalidTriangle(t));
            }
        }
        let mut seen = vec![false; node_count];
        let mut stack = vec![(0usize, 0usize)];
        let mut max_stack = 0;
        while let Some((n, depth)) = stack.pop() {
            if n >= node_count || seen[n] {
                return Err(TraceGpuError::InvalidTree);
            }
            seen[n] = true;
            max_stack = max_stack.max(depth);
            let b = n * NODE_WORDS;
            if self.nodes[b + 8] == 0 {
                max_stack = max_stack.max(depth + 1);
                stack.push((self.nodes[b + 7] as usize, depth + 1));
                stack.push((n + 1, depth + 1));
            }
        }
        if seen.iter().any(|v| !v) {
            return Err(TraceGpuError::InvalidTree);
        }
        if max_stack > MAX_STACK {
            return Err(TraceGpuError::StackDepthExceeded);
        }
        Ok(())
    }
}

/// CPU oracle over the exact packed input accepted by the GPU component.
pub fn trace_reference(
    scene: &PackedTraceScene,
    rays: &[TraceRay],
) -> Result<Vec<TraceHit>, TraceGpuError> {
    scene.validate()?;
    validate_rays(rays)?;
    Ok(rays.iter().map(|ray| trace_one(scene, *ray)).collect())
}

/// Real resident upload-buffer capacities owned by a trace pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceBufferCapacityStats {
    /// Capacity of the packed node upload buffer, in bytes.
    pub nodes_bytes: u64,
    /// Capacity of the packed triangle upload buffer, in bytes.
    pub triangles_bytes: u64,
    /// Capacity of the packed ray upload buffer, in bytes.
    pub rays_bytes: u64,
    /// Number of buffers retained after capacity growth until this pass drops.
    pub retired_buffers: usize,
    /// Number of upload-buffer allocations performed by this pass.
    pub allocations: u64,
}

/// A buffer that the render graph must import before recording a trace pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceGraphBuffer {
    /// Backend buffer identity.
    pub id: rhi::BufferId,
    /// Minimum byte range used by this trace dispatch.
    pub size_bytes: u64,
}

/// Persistent input resources and dispatch recorder for one graph trace pass.
///
/// The originating pass retains every graph bind group until it is dropped,
/// and retired input buffers are retained for the same period. Keep the pass
/// alive through graph execution and queue submission. The graph must import
/// `nodes`, `triangles`, and `rays` as
/// storage reads and `hits` as a storage write. The prepared input buffers are
/// not replaced while this recorder exists because preparation requires a
/// mutable borrow of the pass.
#[derive(Clone, Copy)]
pub struct TraceGraphRecorder {
    pub nodes: TraceGraphBuffer,
    pub triangles: TraceGraphBuffer,
    pub rays: TraceGraphBuffer,
    pub hits: TraceGraphBuffer,
    pipeline: rhi::ComputePipelineId,
    bind_group: rhi::BindGroupId,
    workgroups: u32,
}

impl TraceGraphRecorder {
    /// Appends the traversal to a caller-owned encoder; never submits it.
    pub fn record(&self, encoder: &mut rhi::CommandEncoder) {
        encoder.push_compute_pass(
            Some("lighting-software-bvh-trace".into()),
            vec![
                rhi::ComputeCommand::SetPipeline(self.pipeline),
                rhi::ComputeCommand::SetBindGroup {
                    index: 0,
                    bind_group: self.bind_group,
                    dynamic_offsets: vec![],
                },
                rhi::ComputeCommand::Dispatch {
                    x: self.workgroups,
                    y: 1,
                    z: 1,
                },
            ],
        );
    }

    /// Records this dispatch into the render graph's compute execute context.
    pub fn record_graph(&self, context: &mut prism_render_graph::ExecuteContext<'_>) {
        context
            .set_compute_pipeline(self.pipeline)
            .set_compute_bind_group(0, self.bind_group, vec![])
            .dispatch(self.workgroups, 1, 1);
    }
}

#[derive(Clone, Copy)]
struct ResidentBuffer {
    id: rhi::BufferId,
    capacity_bytes: u64,
}

/// Owned compute pipeline and capacity-managed upload buffers.
///
/// A pass is device-affine: use only the queue belonging to the device passed
/// to [`Self::new`]. Calls require exclusive mutable access and are intended
/// for sequential render-thread dispatches. Grown-out buffers remain resident
/// until this pass is dropped because the RHI exposes ordered submissions but
/// no completion token for safe earlier retirement. Dropping the pass releases
/// every resident and retired allocation through that same device.
pub struct LightingTraceGpu {
    device: Arc<dyn RenderDevice + Send + Sync>,
    layout: rhi::BindGroupLayoutId,
    pipeline_layout: rhi::PipelineLayoutId,
    shader: rhi::ShaderModuleId,
    pipeline: rhi::ComputePipelineId,
    nodes: Option<ResidentBuffer>,
    triangles: Option<ResidentBuffer>,
    rays: Option<ResidentBuffer>,
    retired_buffers: Vec<rhi::BufferId>,
    graph_bind_groups: Vec<rhi::BindGroupId>,
    packed_rays: Vec<u32>,
    allocations: u64,
}

impl LightingTraceGpu {
    pub fn new(device: Arc<dyn RenderDevice + Send + Sync>) -> Self {
        let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("lighting-trace-gpu-layout".into()),
            entries: (0..4)
                .map(|binding| rhi::BindGroupLayoutEntry {
                    binding,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: rhi::BindingType::Buffer {
                        ty: rhi::BufferBindingType::Storage {
                            read_only: binding != 3,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect(),
        });
        let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
            label: Some("lighting-trace-gpu-pipeline-layout".into()),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
        });
        let shader = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(TRACE_WGSL));
        let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("lighting-software-bvh-trace".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "main".into(),
        });
        Self {
            device,
            layout,
            pipeline_layout,
            shader,
            pipeline,
            nodes: None,
            triangles: None,
            rays: None,
            retired_buffers: Vec::new(),
            graph_bind_groups: Vec::new(),
            packed_rays: Vec::new(),
            allocations: 0,
        }
    }

    /// Reports the actual resident upload capacities and allocation count.
    pub fn buffer_capacity_stats(&self) -> TraceBufferCapacityStats {
        TraceBufferCapacityStats {
            nodes_bytes: self.nodes.map_or(0, |buffer| buffer.capacity_bytes),
            triangles_bytes: self.triangles.map_or(0, |buffer| buffer.capacity_bytes),
            rays_bytes: self.rays.map_or(0, |buffer| buffer.capacity_bytes),
            retired_buffers: self.retired_buffers.len(),
            allocations: self.allocations,
        }
    }

    /// Uploads scene and ray inputs into persistent capacity-managed buffers,
    /// then prepares a recorder for insertion into a caller-owned graph pass.
    ///
    /// No command buffer is created and the queue is never submitted here.
    /// `output` must have storage usage and at least `rays.len() * 32` bytes.
    /// Keep `self` alive through graph execution and submission. Bind groups
    /// are retained until `self` is dropped; the returned recorder is a small
    /// copyable command description suitable for a `'static` graph closure.
    pub fn prepare_graph(
        &mut self,
        queue: &dyn RenderQueue,
        scene: &PackedTraceScene,
        rays: &[TraceRay],
        output: rhi::BufferId,
        output_size_bytes: u64,
    ) -> Result<TraceGraphRecorder, TraceGpuError> {
        scene.validate()?;
        validate_rays(rays)?;
        if rays.is_empty() {
            return Err(TraceGpuError::EmptyRays);
        }
        let sizes = [
            byte_size(scene.nodes.len(), 4)?,
            byte_size(scene.triangles.len(), 4)?,
            byte_size(rays.len(), RAY_WORDS * 4)?,
            byte_size(rays.len(), HIT_WORDS * 4)?,
        ];
        validate_output_size(output_size_bytes, sizes[3])?;
        let caps = self.device.capabilities();
        let limit = caps.limits.max_storage_buffer_binding_size;
        if sizes.iter().any(|size| *size > limit) {
            return Err(TraceGpuError::StorageBindingLimit);
        }
        let workgroups = u32::try_from(rays.len())
            .map_err(|_| TraceGpuError::DispatchLimit)?
            .div_ceil(WORKGROUP_SIZE);
        if workgroups > caps.limits.max_compute_workgroups_per_dimension {
            return Err(TraceGpuError::DispatchLimit);
        }
        let node_buffer = ensure_resident_buffer(
            &*self.device,
            &mut self.nodes,
            &mut self.retired_buffers,
            &mut self.allocations,
            sizes[0],
            limit,
        );
        let triangle_buffer = ensure_resident_buffer(
            &*self.device,
            &mut self.triangles,
            &mut self.retired_buffers,
            &mut self.allocations,
            sizes[1],
            limit,
        );
        let ray_buffer = ensure_resident_buffer(
            &*self.device,
            &mut self.rays,
            &mut self.retired_buffers,
            &mut self.allocations,
            sizes[2],
            limit,
        );
        queue.write_buffer(node_buffer, 0, bytemuck::cast_slice(&scene.nodes));
        queue.write_buffer(triangle_buffer, 0, bytemuck::cast_slice(&scene.triangles));
        self.pack_rays(rays, sizes[2])?;
        queue.write_buffer(ray_buffer, 0, bytemuck::cast_slice(&self.packed_rays));
        let group = self.device.create_bind_group(&rhi::BindGroupDescriptor {
            label: Some("lighting-trace-gpu-graph-resources".into()),
            layout: Some(self.layout),
            entries: [node_buffer, triangle_buffer, ray_buffer, output]
                .into_iter()
                .zip(sizes)
                .enumerate()
                .map(|(binding, (buffer, size))| rhi::BindGroupEntry {
                    binding: binding as u32,
                    resource: rhi::BindingResource::Buffer {
                        buffer,
                        offset: 0,
                        size: Some(size),
                    },
                })
                .collect(),
        });
        self.graph_bind_groups.push(group);
        Ok(TraceGraphRecorder {
            nodes: TraceGraphBuffer {
                id: node_buffer,
                size_bytes: sizes[0],
            },
            triangles: TraceGraphBuffer {
                id: triangle_buffer,
                size_bytes: sizes[1],
            },
            rays: TraceGraphBuffer {
                id: ray_buffer,
                size_bytes: sizes[2],
            },
            hits: TraceGraphBuffer {
                id: output,
                size_bytes: sizes[3],
            },
            pipeline: self.pipeline,
            bind_group: group,
            workgroups,
        })
    }

    fn pack_rays(&mut self, rays: &[TraceRay], byte_count: u64) -> Result<(), TraceGpuError> {
        self.packed_rays.clear();
        let word_count =
            usize::try_from(byte_count / 4).map_err(|_| TraceGpuError::BufferSizeOverflow)?;
        if self.packed_rays.capacity() < word_count {
            self.packed_rays.reserve(word_count);
        }
        for ray in rays {
            self.packed_rays.extend([
                ray.origin[0].to_bits(),
                ray.origin[1].to_bits(),
                ray.origin[2].to_bits(),
                ray.t_min.to_bits(),
                ray.direction[0].to_bits(),
                ray.direction[1].to_bits(),
                ray.direction[2].to_bits(),
                ray.t_max.to_bits(),
            ]);
        }
        Ok(())
    }
}

impl Drop for LightingTraceGpu {
    fn drop(&mut self) {
        for bind_group in self.graph_bind_groups.drain(..) {
            self.device.destroy_bind_group(bind_group);
        }
        for buffer in [self.nodes, self.triangles, self.rays]
            .into_iter()
            .flatten()
        {
            self.device.destroy_buffer(buffer.id);
        }
        for buffer in self.retired_buffers.drain(..) {
            self.device.destroy_buffer(buffer);
        }
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device.destroy_shader_module(self.shader);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device.destroy_bind_group_layout(self.layout);
    }
}

fn byte_size(count: usize, stride: usize) -> Result<u64, TraceGpuError> {
    let bytes = count
        .checked_mul(stride)
        .ok_or(TraceGpuError::BufferSizeOverflow)?;
    u64::try_from(bytes).map_err(|_| TraceGpuError::BufferSizeOverflow)
}

fn validate_output_size(output_size: u64, required_size: u64) -> Result<(), TraceGpuError> {
    if output_size < required_size {
        Err(TraceGpuError::OutputBufferTooSmall)
    } else {
        Ok(())
    }
}

fn grown_capacity(current: u64, required: u64, limit: u64) -> u64 {
    let base = current.max(required).max(4);
    let rounded = base.checked_next_power_of_two().unwrap_or(base);
    if rounded <= limit {
        rounded
    } else {
        required
    }
}

fn ensure_resident_buffer(
    device: &dyn RenderDevice,
    slot: &mut Option<ResidentBuffer>,
    retired_buffers: &mut Vec<rhi::BufferId>,
    allocations: &mut u64,
    required_bytes: u64,
    binding_limit: u64,
) -> rhi::BufferId {
    if let Some(buffer) = slot.filter(|buffer| buffer.capacity_bytes >= required_bytes) {
        return buffer.id;
    }
    let capacity_bytes = grown_capacity(
        slot.map_or(0, |buffer| buffer.capacity_bytes),
        required_bytes,
        binding_limit,
    );
    let id = device.create_buffer(&rhi::BufferDescriptor::new(
        capacity_bytes,
        rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_DST,
    ));
    if let Some(previous) = slot.replace(ResidentBuffer { id, capacity_bytes }) {
        retired_buffers.push(previous.id);
    }
    *allocations = (*allocations).saturating_add(1);
    id
}

fn validate_rays(rays: &[TraceRay]) -> Result<(), TraceGpuError> {
    if rays.len() > MAX_RAY_COUNT {
        return Err(TraceGpuError::RayLimit);
    }
    for (i, r) in rays.iter().enumerate() {
        if !r
            .origin
            .iter()
            .chain(r.direction.iter())
            .all(|v| v.is_finite())
            || !r.t_min.is_finite()
            || !r.t_max.is_finite()
            || r.t_min < 0.0
            || r.t_max < r.t_min
            || dot(r.direction, r.direction) <= 1.0e-20
        {
            return Err(TraceGpuError::InvalidRay(i));
        }
    }
    Ok(())
}

fn trace_one(scene: &PackedTraceScene, ray: TraceRay) -> TraceHit {
    let mut stack = [0u32; MAX_STACK];
    let mut sp = 0usize;
    let mut node = 0usize;
    let mut closest = ray.t_max;
    let mut hit = TraceHit::MISS;
    loop {
        let b = node * NODE_WORDS;
        if box_hit(&scene.nodes, b, ray, closest) {
            let count = scene.nodes[b + 8] as usize;
            if count > 0 {
                let start = scene.nodes[b + 6] as usize;
                for tri in start..start + count {
                    if let Some((t, u, v)) =
                        triangle_hit(&scene.triangles, tri * TRIANGLE_WORDS, ray, closest)
                    {
                        closest = t;
                        hit = TraceHit {
                            hit: true,
                            primitive: scene.triangles[tri * TRIANGLE_WORDS + 9],
                            t,
                            u,
                            v,
                        };
                    }
                }
                if sp == 0 {
                    break;
                }
                sp -= 1;
                node = stack[sp] as usize;
            } else {
                let axis = scene.nodes[b + 9] as usize;
                let left = node + 1;
                let right = scene.nodes[b + 7] as usize;
                let (near, far) = if ray.direction[axis] < 0.0 {
                    (right, left)
                } else {
                    (left, right)
                };
                stack[sp] = far as u32;
                sp += 1;
                node = near;
            }
        } else if sp > 0 {
            sp -= 1;
            node = stack[sp] as usize;
        } else {
            break;
        }
    }
    hit
}

fn vec3(w: &[u32], b: usize) -> [f32; 3] {
    [
        f32::from_bits(w[b]),
        f32::from_bits(w[b + 1]),
        f32::from_bits(w[b + 2]),
    ]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn box_hit(w: &[u32], b: usize, r: TraceRay, max_t: f32) -> bool {
    let mut lo = r.t_min;
    let mut hi = max_t;
    for a in 0..3 {
        let o = r.origin[a];
        let d = r.direction[a];
        let mn = f32::from_bits(w[b + a]);
        let mx = f32::from_bits(w[b + 3 + a]);
        if d == 0.0 {
            if o < mn || o > mx {
                return false;
            }
        } else {
            let inv = 1.0 / d;
            let mut x = (mn - o) * inv;
            let mut y = (mx - o) * inv;
            if x > y {
                core::mem::swap(&mut x, &mut y);
            }
            lo = lo.max(x);
            hi = hi.min(y);
            if lo > hi {
                return false;
            }
        }
    }
    true
}
fn triangle_hit(w: &[u32], b: usize, r: TraceRay, max_t: f32) -> Option<(f32, f32, f32)> {
    let a = vec3(w, b);
    let e1 = sub(vec3(w, b + 3), a);
    let e2 = sub(vec3(w, b + 6), a);
    let p = cross(r.direction, e2);
    let det = dot(e1, p);
    if det.abs() <= 1.0e-8 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = sub(r.origin, a);
    let u = dot(tv, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(tv, e1);
    let v = dot(r.direction, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, q) * inv;
    if t < r.t_min || t > max_t {
        None
    } else {
        Some((t, u, v))
    }
}

const TRACE_WGSL: &str = r#"
struct Params { limits: vec4<u32> }
@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> triangles: array<u32>;
@group(0) @binding(2) var<storage, read> rays: array<u32>;
@group(0) @binding(3) var<storage, read_write> hits: array<u32>;
fn f(w:u32)->f32{return bitcast<f32>(w);}
fn vec3at(b:u32)->vec3<f32>{return vec3<f32>(f(triangles[b]),f(triangles[b+1u]),f(triangles[b+2u]));}
fn slab(b:u32,o:vec3<f32>,d:vec3<f32>,tmin:f32,tmax:f32)->bool{
 var lo=tmin; var hi=tmax;
 for(var a=0u;a<3u;a++){let mn=f(nodes[b+a]);let mx=f(nodes[b+3u+a]);let x=o[a];let v=d[a];if(v==0.0){if(x<mn||x>mx){return false;}}else{let inv=1.0/v;var t0=(mn-x)*inv;var t1=(mx-x)*inv;if(t0>t1){let tmp=t0;t0=t1;t1=tmp;}lo=max(lo,t0);hi=min(hi,t1);if(lo>hi){return false;}}} return true;
}
fn tri_hit(b:u32,o:vec3<f32>,d:vec3<f32>,tmin:f32,tmax:f32)->vec3<f32>{
 let a=vec3at(b);let e1=vec3at(b+3u)-a;let e2=vec3at(b+6u)-a;let p=cross(d,e2);let det=dot(e1,p);if(abs(det)<=1.0e-8){return vec3<f32>(-1.0);}let inv=1.0/det;let tv=o-a;let u=dot(tv,p)*inv;if(u<0.0||u>1.0){return vec3<f32>(-1.0);}let q=cross(tv,e1);let v=dot(d,q)*inv;if(v<0.0||u+v>1.0){return vec3<f32>(-1.0);}let t=dot(e2,q)*inv;if(t<tmin||t>tmax){return vec3<f32>(-1.0);}return vec3<f32>(t,u,v);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){let ri=gid.x;if(ri>=arrayLength(&rays)/8u){return;}let rb=ri*8u;let o=vec3<f32>(f(rays[rb]),f(rays[rb+1u]),f(rays[rb+2u]));let tmin=f(rays[rb+3u]);let d=vec3<f32>(f(rays[rb+4u]),f(rays[rb+5u]),f(rays[rb+6u]));var closest=f(rays[rb+7u]);var best_id=0xffffffffu;var bu=0.0;var bv=0.0;var node=0u;var stack:array<u32,64>;var sp=0u;var steps=0u;
 loop{if(steps>=arrayLength(&nodes)/12u*2u+1u){let hb=ri*8u;hits[hb]=2u;hits[hb+1u]=0xffffffffu;hits[hb+2u]=0u;hits[hb+3u]=0u;hits[hb+4u]=0u;return;}steps++;let nb=node*12u;if(slab(nb,o,d,tmin,closest)){let count=nodes[nb+8u];if(count>0u){let start=nodes[nb+6u];for(var i=0u;i<count;i++){let tb=(start+i)*12u;let h=tri_hit(tb,o,d,tmin,closest);if(h.x>=0.0){closest=h.x;best_id=triangles[tb+9u];bu=h.y;bv=h.z;}}if(sp==0u){break;}sp--;node=stack[sp];}else{let left=node+1u;let right=nodes[nb+7u];let axis=nodes[nb+9u];var near=left;var far=right;if(d[axis]<0.0){near=right;far=left;}if(sp>=64u){let hb=ri*8u;hits[hb]=2u;hits[hb+1u]=0xffffffffu;hits[hb+2u]=0u;hits[hb+3u]=0u;hits[hb+4u]=0u;return;}stack[sp]=far;sp++;node=near;}}else{if(sp==0u){break;}sp--;node=stack[sp];}}
 let hb=ri*8u;if(best_id==0xffffffffu){hits[hb]=0u;hits[hb+1u]=0xffffffffu;hits[hb+2u]=0u;hits[hb+3u]=0u;hits[hb+4u]=0u;}else{hits[hb]=1u;hits[hb+1u]=best_id;hits[hb+2u]=bitcast<u32>(closest);hits[hb+3u]=bitcast<u32>(bu);hits[hb+4u]=bitcast<u32>(bv);}hits[hb+5u]=0u;hits[hb+6u]=0u;hits[hb+7u]=0u;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::ray_scene::{Bvh, BvhBuildConfig, Triangle};
    use prism_render_driver::{BufferDescriptor, BufferState, BufferUsages, Instance as _};
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
    use prism_render_graph::{BufferDesc, ImportedBuffer, PassFlags, RenderGraph};

    fn fixture() -> (PackedTraceScene, Vec<TraceRay>) {
        // The production builder defaults to four primitives per leaf, so the
        // fixture explicitly forces a tree. These slanted triangles have
        // overlapping Z bounds across the split branches, and the center ray
        // intersects every triangle. Thus traversal must consider both
        // children while retaining the nearest result (primitive 17).
        let tris: Vec<_> = (0..8)
            .map(|i| {
                let z = -(i as f32);
                Triangle::new(
                    [-1.0, -1.0, z - 5.0],
                    [1.0, -1.0, z - 5.0],
                    [0.0, 1.0, z + 1.0],
                    if i == 0 { 17 } else { 29 + i as u32 },
                )
            })
            .collect();
        let bvh = Bvh::build_with(
            &tris,
            BvhBuildConfig {
                max_leaf_primitives: 1,
                ..BvhBuildConfig::default()
            },
        );
        let packed = prism_render_architecture::ray_scene::GpuBvhBuffers::from_bvh(&bvh);
        let rays = (0..65)
            .map(|i| TraceRay {
                origin: if i == 63 {
                    [10.0, 0.0, 2.0]
                } else {
                    [0.0, 0.0, 2.0]
                },
                t_min: 0.0,
                direction: [0.0, 0.0, -1.0],
                t_max: 10.0,
            })
            .collect();
        (PackedTraceScene::from(&packed), rays)
    }

    #[test]
    fn packed_reference_reports_stable_nearest_hits_and_misses() {
        let (scene, rays) = fixture();
        let node_count = scene.nodes.len() / NODE_WORDS;
        assert!(node_count >= 7, "fixture must contain internal BVH nodes");
        assert!(
            (0..node_count).any(|node| scene.nodes[node * NODE_WORDS + 8] == 0),
            "fixture must exercise internal-node traversal"
        );
        let result = trace_reference(&scene, &rays).unwrap();
        assert_eq!(result.len(), 65);
        assert!(result[..63]
            .iter()
            .all(|h| h.hit && h.primitive == 17 && h.t == 4.0));
        assert!(
            !result[63].hit,
            "misses remain explicit in the stable ray slot"
        );
        assert!(result[64].hit && result[64].primitive == 17 && result[64].t == 4.0);
    }

    #[test]
    fn malformed_packed_ranges_and_rays_are_rejected() {
        let (mut scene, _rays) = fixture();
        scene.nodes[7] = u32::MAX;
        assert_eq!(scene.validate(), Err(TraceGpuError::InvalidNode(0)));
        let (scene, mut rays) = fixture();
        rays[0].direction = [0.0; 3];
        assert_eq!(
            trace_reference(&scene, &rays),
            Err(TraceGpuError::InvalidRay(0))
        );
    }

    #[test]
    fn resident_capacity_growth_is_geometric_bounded_and_never_shrinks() {
        assert_eq!(grown_capacity(0, 12, 1024), 16);
        assert_eq!(grown_capacity(16, 16, 1024), 16);
        assert_eq!(grown_capacity(16, 17, 1024), 32);
        assert_eq!(grown_capacity(64, 32, 1024), 64);
        assert_eq!(grown_capacity(0, 65, 100), 65);
    }

    #[test]
    fn upload_byte_size_rejects_integer_overflow() {
        assert_eq!(
            byte_size(usize::MAX, 4),
            Err(TraceGpuError::BufferSizeOverflow)
        );
        assert_eq!(byte_size(16, 4), Ok(64));
    }

    #[test]
    fn graph_output_range_must_hold_every_packed_hit() {
        assert_eq!(
            validate_output_size(255, 256),
            Err(TraceGpuError::OutputBufferTooSmall)
        );
        assert_eq!(validate_output_size(256, 256), Ok(()));
        assert_eq!(validate_output_size(512, 256), Ok(()));
    }

    #[test]
    fn graph_dispatch_count_rounds_up_without_empty_dispatches() {
        assert_eq!(1_u32.div_ceil(WORKGROUP_SIZE), 1);
        assert_eq!(64_u32.div_ceil(WORKGROUP_SIZE), 1);
        assert_eq!(65_u32.div_ceil(WORKGROUP_SIZE), 2);
    }

    #[test]
    fn graph_prepare_rejects_short_output_and_records_on_null_graph() {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance
            .request_adapter(&rhi::RequestAdapterOptions::default())
            .unwrap();
        let (device, queue) = instance
            .create_device(&adapter, &rhi::DeviceDescriptor::default())
            .unwrap();
        let device = Arc::new(device);
        let trace_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let (scene, rays) = fixture();
        let required = (rays.len() * HIT_WORDS * 4) as u64;
        let output = device.create_buffer(&BufferDescriptor::new(
            required,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let mut trace = LightingTraceGpu::new(trace_device);
        assert!(matches!(
            trace.prepare_graph(&queue, &scene, &rays, output, required - 1),
            Err(TraceGpuError::OutputBufferTooSmall)
        ));
        let recorder = trace
            .prepare_graph(&queue, &scene, &rays, output, required)
            .unwrap();
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let nodes = graph.import_buffer(
            "trace-nodes",
            BufferDesc::new(recorder.nodes.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.nodes.id,
                entry_state: BufferState::initial(),
            },
        );
        let triangles = graph.import_buffer(
            "trace-triangles",
            BufferDesc::new(recorder.triangles.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.triangles.id,
                entry_state: BufferState::initial(),
            },
        );
        let ray_buffer = graph.import_buffer(
            "trace-rays",
            BufferDesc::new(recorder.rays.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.rays.id,
                entry_state: BufferState::initial(),
            },
        );
        let hits = graph.import_buffer(
            "trace-hits",
            BufferDesc::new(required, BufferUsages::STORAGE | BufferUsages::COPY_SRC),
            ImportedBuffer {
                buffer: output,
                entry_state: BufferState::initial(),
            },
        );
        let graph_recorder = recorder;
        graph.add_compute_pass("trace", PassFlags::SIDE_EFFECT, move |builder| {
            builder.read_storage_buffer(nodes);
            builder.read_storage_buffer(triangles);
            builder.read_storage_buffer(ray_buffer);
            builder.write_storage_buffer(hits);
            move |context| graph_recorder.record_graph(context)
        });
        let recorded = graph.execute(device.as_ref(), &queue).unwrap();
        recorded.release(device.as_ref());
        drop(recorder);
        drop(trace);
        device.destroy_buffer(output);
    }

    #[test]
    fn hardware_wgsl_traversal_matches_cpu_across_workgroups() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no physical adapter was found"
                );
                eprintln!("SKIPPED trace GPU parity: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires physical GPU: {evidence}"
                );
                eprintln!("SKIPPED trace GPU parity: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(e)) => {
                panic!("hardware GPU device creation failed: {e}")
            }
        };
        let device = Arc::new(device);
        let trace_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let (scene, rays) = fixture();
        let expected = trace_reference(&scene, &rays).unwrap();
        let node_count = scene.nodes.len() / NODE_WORDS;
        assert!(node_count >= 7, "fixture must contain internal BVH nodes");
        assert!(
            (0..node_count).any(|node| scene.nodes[node * NODE_WORDS + 8] == 0),
            "fixture must exercise internal-node traversal"
        );
        let output = device.create_buffer(&BufferDescriptor::new(
            (rays.len() * HIT_WORDS * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let mut pass = LightingTraceGpu::new(trace_device);
        let recorder = pass
            .prepare_graph(
                &queue,
                &scene,
                &rays,
                output,
                (rays.len() * HIT_WORDS * 4) as u64,
            )
            .unwrap();
        assert_eq!(
            recorder.hits.size_bytes,
            (rays.len() * HIT_WORDS * 4) as u64
        );
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        let nodes = graph.import_buffer(
            "trace-nodes",
            BufferDesc::new(recorder.nodes.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.nodes.id,
                entry_state: BufferState::initial(),
            },
        );
        let triangles = graph.import_buffer(
            "trace-triangles",
            BufferDesc::new(recorder.triangles.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.triangles.id,
                entry_state: BufferState::initial(),
            },
        );
        let ray_buffer = graph.import_buffer(
            "trace-rays",
            BufferDesc::new(recorder.rays.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.rays.id,
                entry_state: BufferState::initial(),
            },
        );
        let hits = graph.import_buffer(
            "trace-hits",
            BufferDesc::new(
                recorder.hits.size_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            ImportedBuffer {
                buffer: recorder.hits.id,
                entry_state: BufferState::initial(),
            },
        );
        let graph_recorder = recorder;
        graph.add_compute_pass(
            "software-bvh-trace",
            PassFlags::SIDE_EFFECT,
            move |builder| {
                builder.read_storage_buffer(nodes);
                builder.read_storage_buffer(triangles);
                builder.read_storage_buffer(ray_buffer);
                builder.write_storage_buffer(hits);
                move |context| graph_recorder.record_graph(context)
            },
        );
        let recorded = graph.execute(device.as_ref(), &queue).unwrap();
        let bytes = queue.read_buffer(output, (rays.len() * HIT_WORDS * 4) as u64);
        let words: &[u32] = bytemuck::try_cast_slice(&bytes).unwrap();
        for (i, hit) in expected.iter().enumerate() {
            let b = i * HIT_WORDS;
            assert_eq!(words[b], u32::from(hit.hit), "ray {i} hit flag");
            assert_eq!(words[b + 1], hit.primitive, "ray {i} primitive");
            let expected_tuv = [hit.t, hit.u, hit.v];
            for (offset, expected_value) in expected_tuv.into_iter().enumerate() {
                assert!(
                    (f32::from_bits(words[b + 2 + offset]) - expected_value).abs() < 1e-5,
                    "ray {i} t/u/v word {}",
                    offset + 2
                );
            }
            assert_eq!(&words[b + 5..b + 8], &[0, 0, 0], "ray {i} padding");
        }
        recorded.release(device.as_ref());
        drop(recorder);
        drop(pass);
        device.destroy_buffer(output);
    }
}
