//! Bounded, standalone GPU software-BVH trace pass.
//!
//! The packed geometry ABI is the architecture crate's `GpuBvhBuffers` ABI:
//! 12 `u32` words per node and triangle. This module consumes those words
//! directly and performs a bounded stack traversal in WGSL.

use std::sync::Arc;

use crate::lighting::{LightingGiCandidateInput, LightingSurfaceInput};
use crate::lighting_gpu::LightingGiCandidateTarget;
use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::raygen::{
    append_surface_gi_rays, GiRay, SurfaceRayGenInput,
};
use prism_render_architecture::ray_scene::{
    GpuBvhBuffers, RayHitAttributeRecord, NODE_WORDS, TRIANGLE_WORDS,
};
use prism_render_driver as rhi;
use rhi::{RenderDevice, RenderQueue};

const WORKGROUP_SIZE: u32 = 64;
const MAX_STACK: usize = 64;
const MAX_NODE_COUNT: usize = 1_000_000;
const MAX_TRIANGLE_COUNT: usize = 2_000_000;
const MAX_RAY_COUNT: usize = 1_000_000;
const RAY_WORDS: usize = 12;
const HIT_WORDS: usize = 16;
/// Host GI trace work is explicitly bounded to 64 rays for each of 1024
/// extracted surfaces; a following GPU pass builds surface-major candidates.
pub const MAX_HOST_TRACE_RAYS: usize = 65_536;
/// Bound matching the architecture lighting surface budget and sample policy.
pub const MAX_HOST_TRACE_RAYS_PER_SURFACE: u32 = 64;
pub(crate) const HOST_TRACE_OUTPUT_BYTES: u64 = (MAX_HOST_TRACE_RAYS * HIT_WORDS * 4) as u64;

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
    /// Solid-angle density of the sampled direction.
    pub direction_pdf: f32,
    /// Deterministic reservoir-selection variate, shared by CPU and GPU paths.
    pub selection_u01: f32,
}

/// Stable result. A miss has `hit == false`, `primitive == u32::MAX`, and zero
/// `t/u/v`; traversal faults are returned as errors, never disguised as misses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceHit {
    /// Whether the ray intersects a triangle in the interval.
    pub hit: bool,
    /// Stable primitive identifier, or `u32::MAX` for a miss.
    pub primitive: u32,
    /// Packed triangle record index, or `u32::MAX` for a miss.
    pub triangle_index: u32,
    /// Stable instance identity carried by the hit payload.
    pub instance_id: u32,
    /// Stable geometry identity carried by the hit payload.
    pub geometry_id: u32,
    /// Material content generation carried by the hit payload.
    pub material_generation: u32,
    /// Interpolated texture coordinate carried by the hit payload.
    pub uv: [f32; 2],
    /// Interpolated shading normal carried by the hit payload.
    pub shading_normal: [f32; 3],
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
        triangle_index: u32::MAX,
        instance_id: u32::MAX,
        geometry_id: u32::MAX,
        material_generation: u32::MAX,
        uv: [0.0; 2],
        shading_normal: [0.0; 3],
        t: 0.0,
        u: 0.0,
        v: 0.0,
    };
}

/// Invalid input to [`build_gi_candidates_reference`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GiCandidateReferenceError {
    /// A surface batch is empty or ray count is not divisible by surface count.
    InvalidBatchShape,
    /// A surface position, normal, or albedo is non-finite or outside its domain.
    InvalidSurface(usize),
    /// A ray is invalid, including a non-positive/non-finite direction PDF.
    InvalidRay(usize),
    /// The hit record is internally inconsistent or contains invalid values.
    InvalidHit(usize),
    /// A hit refers to a triangle outside the packed scene or disagrees with its primitive.
    InvalidTriangleIndex(usize),
    /// A committed hit has no validated instance/geometry/material identity
    /// and surface attributes, so secondary shading is rejected.
    MissingHitAttributes(usize),
    /// The hit payload disagrees with the packed triangle's authoritative
    /// attributes or contains an invalid interpolated normal/UV.
    InvalidHitAttributes(usize),
    /// The outgoing-radiance table does not cover every packed triangle.
    RadianceCountMismatch { expected: usize, actual: usize },
    /// Outgoing radiance contains a negative or non-finite channel.
    InvalidRadiance(usize),
    /// The packed BVH failed validation.
    InvalidScene(TraceGpuError),
}

/// Builds one solid-angle-measure RIS proposal batch per surface from trace results.
///
/// Rays and hits are surface-major: all rays for surface 0, then surface 1, etc.
/// The per-surface batch length is the proposal count `N`; misses and zero-target
/// hits remain present, preserving the estimator denominator. `outgoing_radiance`
/// is indexed by packed triangle record index. The target is the luminance of
/// `albedo * Le * max(n_primary·direction, 0) / π` (Lambert integrand in solid
/// angle); `source_pdf` is exactly the sampled direction density.
pub fn build_gi_candidates_reference(
    scene: &PackedTraceScene,
    rays: &[TraceRay],
    hits: &[TraceHit],
    surfaces: &[LightingSurfaceInput],
    stable_surface_ids: &[u64],
    frame_index: u64,
    outgoing_radiance: &[[f32; 3]],
) -> Result<Vec<Vec<LightingGiCandidateInput>>, GiCandidateReferenceError> {
    scene
        .validate()
        .map_err(GiCandidateReferenceError::InvalidScene)?;
    scene
        .validate_secondary_attributes()
        .map_err(GiCandidateReferenceError::InvalidScene)?;
    let triangle_count = scene.triangles.len() / TRIANGLE_WORDS;
    if outgoing_radiance.len() != triangle_count {
        return Err(GiCandidateReferenceError::RadianceCountMismatch {
            expected: triangle_count,
            actual: outgoing_radiance.len(),
        });
    }
    if surfaces.is_empty()
        || surfaces.len() != stable_surface_ids.len()
        || rays.is_empty()
        || rays.len() != hits.len()
        || rays.len() % surfaces.len() != 0
    {
        return Err(GiCandidateReferenceError::InvalidBatchShape);
    }
    let samples_per_surface = rays.len() / surfaces.len();
    let pi = core::f32::consts::PI;
    for (index, surface) in surfaces.iter().enumerate() {
        if !surface.position.into_iter().all(f32::is_finite)
            || !surface.normal.into_iter().all(f32::is_finite)
            || !surface
                .albedo
                .into_iter()
                .all(|v| v.is_finite() && v >= 0.0)
            || dot(surface.normal, surface.normal) <= 1.0e-20
        {
            return Err(GiCandidateReferenceError::InvalidSurface(index));
        }
    }
    for (index, radiance) in outgoing_radiance.iter().enumerate() {
        if !radiance.into_iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(GiCandidateReferenceError::InvalidRadiance(index));
        }
    }

    let mut result = Vec::with_capacity(surfaces.len());
    for (surface_index, (surface, &stable_id)) in
        surfaces.iter().zip(stable_surface_ids).enumerate()
    {
        let mut batch = Vec::with_capacity(samples_per_surface);
        for local_ray_index in 0..samples_per_surface {
            let ray_index = surface_index * samples_per_surface + local_ray_index;
            let ray = rays[ray_index];
            let hit = hits[ray_index];
            let direction_length_squared = dot(ray.direction, ray.direction);
            if !ray.origin.into_iter().all(f32::is_finite)
                || !ray.direction.into_iter().all(f32::is_finite)
                || !ray.t_min.is_finite()
                || !ray.t_max.is_finite()
                || ray.t_min < 0.0
                || ray.t_max < ray.t_min
                || !direction_length_squared.is_finite()
                || direction_length_squared <= 1.0e-20
                || !ray.direction_pdf.is_finite()
                || ray.direction_pdf <= 0.0
                || !ray.selection_u01.is_finite()
                || !(0.0..1.0).contains(&ray.selection_u01)
            {
                return Err(GiCandidateReferenceError::InvalidRay(ray_index));
            }
            let dir = normalize(ray.direction);
            let mut candidate = LightingGiCandidateInput {
                position: [0.0; 3],
                source_pdf: ray.direction_pdf,
                normal: [0.0, 1.0, 0.0],
                target_pdf: 0.0,
                radiance: [0.0; 3],
                selection_u01: selection_hash(frame_index, stable_id, local_ray_index as u64),
                visibility: 0.0,
                _padding: [0.0; 3],
            };
            if hit.hit {
                if !hit.t.is_finite()
                    || hit.t < ray.t_min
                    || hit.t > ray.t_max
                    || !hit.u.is_finite()
                    || !hit.v.is_finite()
                    || hit.u < 0.0
                    || hit.v < 0.0
                    || hit.u + hit.v > 1.0 + 1.0e-5
                {
                    return Err(GiCandidateReferenceError::InvalidHit(ray_index));
                }
                if hit.triangle_index as usize >= triangle_count {
                    return Err(GiCandidateReferenceError::InvalidTriangleIndex(ray_index));
                }
                let triangle = hit.triangle_index as usize;
                let base = triangle * TRIANGLE_WORDS;
                if scene.triangles[base + 9] != hit.primitive {
                    return Err(GiCandidateReferenceError::InvalidTriangleIndex(ray_index));
                }
                if hit.instance_id == u32::MAX
                    || hit.geometry_id == u32::MAX
                    || hit.material_generation == u32::MAX
                {
                    return Err(GiCandidateReferenceError::MissingHitAttributes(ray_index));
                }
                let attributes = scene
                    .hit_attributes()
                    .and_then(|records| records.get(triangle))
                    .and_then(|record| record.interpolate(hit.u, hit.v).ok())
                    .ok_or(GiCandidateReferenceError::InvalidHitAttributes(ray_index))?;
                if attributes.identity.instance_id != hit.instance_id
                    || attributes.identity.geometry_id != hit.geometry_id
                    || attributes.identity.material_generation != hit.material_generation
                    || attributes
                        .uv
                        .into_iter()
                        .zip(hit.uv)
                        .any(|(expected, actual)| expected.to_bits() != actual.to_bits())
                    || attributes
                        .shading_normal
                        .into_iter()
                        .zip(hit.shading_normal)
                        .any(|(expected, actual)| expected.to_bits() != actual.to_bits())
                {
                    return Err(GiCandidateReferenceError::InvalidHitAttributes(ray_index));
                }
                let a = vec3(&scene.triangles, base);
                let b = vec3(&scene.triangles, base + 3);
                let c = vec3(&scene.triangles, base + 6);
                let unnormalized_normal = cross(sub(b, a), sub(c, a));
                let normal_length_squared = dot(unnormalized_normal, unnormalized_normal);
                if !normal_length_squared.is_finite() || normal_length_squared <= 1.0e-20 {
                    return Err(GiCandidateReferenceError::InvalidTriangleIndex(ray_index));
                }
                let geometric_normal = normalize(unnormalized_normal);
                if dot(attributes.shading_normal, geometric_normal) < -1.0e-5 {
                    return Err(GiCandidateReferenceError::InvalidHitAttributes(ray_index));
                }
                let position = add(ray.origin, mul(ray.direction, hit.t));
                if !position.into_iter().all(f32::is_finite) {
                    return Err(GiCandidateReferenceError::InvalidHit(ray_index));
                }
                let barycentric_position = add(
                    add(mul(a, 1.0 - hit.u - hit.v), mul(b, hit.u)),
                    mul(c, hit.v),
                );
                let scale = position
                    .into_iter()
                    .chain(barycentric_position)
                    .map(f32::abs)
                    .fold(1.0_f32, f32::max);
                let delta = sub(position, barycentric_position);
                if dot(delta, delta) > (1.0e-4 * scale).powi(2) {
                    return Err(GiCandidateReferenceError::InvalidHit(ray_index));
                }
                candidate.position = position;
                candidate.normal = attributes.shading_normal;
                let radiance = outgoing_radiance[triangle];
                candidate.radiance = radiance;
                candidate.visibility = 1.0;
                let primary_normal = normalize(surface.normal);
                let cos_primary = dot(primary_normal, dir).max(0.0);
                if cos_primary > 0.0 {
                    let contribution = [
                        surface.albedo[0] * radiance[0] * cos_primary / pi,
                        surface.albedo[1] * radiance[1] * cos_primary / pi,
                        surface.albedo[2] * radiance[2] * cos_primary / pi,
                    ];
                    let target = luminance(contribution);
                    if !target.is_finite() {
                        return Err(GiCandidateReferenceError::InvalidRadiance(triangle));
                    }
                    candidate.target_pdf = target;
                }
            } else if hit.primitive != u32::MAX
                || hit.triangle_index != u32::MAX
                || hit.t != 0.0
                || hit.u != 0.0
                || hit.v != 0.0
            {
                return Err(GiCandidateReferenceError::InvalidHit(ray_index));
            }
            batch.push(candidate);
        }
        result.push(batch);
    }
    Ok(result)
}

fn unit_vector(v: [f32; 3]) -> bool {
    let length_squared = dot(v, v);
    length_squared.is_finite() && (length_squared - 1.0).abs() <= 1.0e-3
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let inv_len = dot(v, v).sqrt().recip();
    mul(v, inv_len)
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn selection_hash(frame: u64, surface: u64, ray: u64) -> f32 {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    let bits = mix(frame
        ^ mix(surface.wrapping_add(0x9e3779b97f4a7c15))
        ^ mix(ray.wrapping_add(0xd1b54a32d192ed03)));
    (bits >> 40) as f32 * (1.0 / 16_777_216.0)
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
    /// The trace scene has no per-triangle identity/attribute payload.
    MissingHitAttributePayload,
    /// A per-triangle identity/attribute payload is malformed.
    InvalidHitAttributePayload(usize),
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
    /// A candidate-build storage binding exceeds the device limit.
    CandidateStorageBindingLimit,
    /// A candidate-build buffer cannot cover this frame's indexed accesses.
    CandidateBufferTooSmall,
    /// Trace ray count does not match the candidate surface-major batch shape.
    CandidateBatchShape,
    /// The GPU detected a traversal safety-bound violation.
    TraversalFault(usize),
}

/// Host-side validation outcome for preparing the optional frame trace pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TracePreparationError {
    /// No trace scene was supplied by the host.
    NoScene,
    /// The current frame has no surfaces to trace.
    NoSurfaces,
    /// Stable IDs are not one-to-one with the extracted surfaces.
    SurfaceIdCountMismatch {
        /// Number of extracted surface samples.
        surfaces: usize,
        /// Number of provided stable surface IDs.
        ids: usize,
    },
    /// Per-surface samples are zero or exceed the bounded frame capacity.
    RayCapacity,
    /// Two surfaces claim the same stable identity.
    DuplicateSurfaceId(u64),
    /// The outgoing-radiance table must have exactly one entry per packed triangle.
    RadianceCountMismatch {
        /// Packed BVH triangle count.
        triangles: usize,
        /// Supplied radiance-record count.
        radiance: usize,
    },
    /// Two radiance records claim the same stable primitive ID.
    DuplicateRadiancePrimitive(u32),
    /// The packed BVH contains a primitive ID more than once.
    DuplicatePackedPrimitive(u32),
    /// The radiance table's primitive IDs do not exactly match the BVH.
    RadiancePrimitiveSetMismatch {
        /// Primitive ID from the outgoing-radiance table.
        radiance_id: u32,
        /// Primitive ID from the packed BVH at the same sorted position.
        packed_id: u32,
    },
    /// A packed triangle has no matching stable radiance record.
    MissingRadiancePrimitive {
        /// Index of the triangle in packed BVH order.
        packed_index: usize,
        /// Stable primitive ID stored in that triangle.
        primitive_id: u32,
    },
    /// The radiance table exceeds the bounded resident GPU allocation.
    RadianceCapacity,
    /// An outgoing-radiance value is negative or non-finite.
    InvalidRadiance(usize),
    /// Scene validation or GPU preparation rejected the packed input.
    Trace(TraceGpuError),
    /// Architecture ray generation rejected a surface or allocation.
    RayGeneration(prism_render_architecture::lighting::raygen::RayGenError),
}

/// Observable status of the latest host trace preparation attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TracePreparationStatus {
    /// The trace feature is not configured by the host.
    Disabled,
    /// No eligible surface batch was available.
    NoSurfaces,
    /// Input validation or device preparation failed.
    Failed(TracePreparationError),
    /// Trace dispatch was prepared; hits remain an intermediate output.
    Prepared { rays: u32, frame_index: u64 },
}

/// Render-world owner for trace resources and the most recent preparation result.
#[derive(Resource, Default)]
pub struct LightingTraceState {
    generation: Option<u64>,
    gpu: Option<LightingTraceGpu>,
    output: Option<TraceOutputBuffer>,
    status: TracePreparationStatus,
    validated_scene: Option<Arc<GpuBvhBuffers>>,
    surface_id_scratch: Vec<u64>,
    ray_scratch: Vec<TraceRay>,
    raygen_scratch: Vec<GiRay>,
}

impl Default for TracePreparationStatus {
    fn default() -> Self {
        Self::Disabled
    }
}

impl LightingTraceState {
    /// Latest trace preparation status, suitable for diagnostics and tests.
    pub const fn status(&self) -> TracePreparationStatus {
        self.status
    }

    /// Prepares a deterministic per-surface trace batch for this frame.
    pub fn prepare(
        &mut self,
        input: Option<&crate::LightingTraceSceneInput>,
        surfaces: &[crate::LightingSurfaceInput],
        frame_index: u64,
        device_generation: u64,
        device: Arc<dyn RenderDevice + Send + Sync>,
        queue: &dyn RenderQueue,
        candidate_target: Option<LightingGiCandidateTarget>,
    ) -> Option<TraceGraphRecorder> {
        // A device-generation change invalidates every trace/candidate object,
        // even when this frame has no trace input.  Do this before the early
        // returns below so recovery cannot leave the previous device's GPU
        // resources retained indefinitely during an input-free frame.
        if self.generation != Some(device_generation) {
            self.gpu = None;
            self.output = None;
            self.generation = Some(device_generation);
        }
        let Some(input) = input else {
            self.status = TracePreparationStatus::Disabled;
            return None;
        };
        if input.outgoing_radiance.len() > MAX_TRIANGLE_COUNT {
            self.status = TracePreparationStatus::Failed(TracePreparationError::RadianceCapacity);
            return None;
        }
        if let Err(error) = input.packed_outgoing_radiance() {
            self.status = TracePreparationStatus::Failed(error);
            return None;
        }
        if surfaces.is_empty() {
            self.status = TracePreparationStatus::NoSurfaces;
            return None;
        }
        if input.surface_ids.len() != surfaces.len() {
            self.status =
                TracePreparationStatus::Failed(TracePreparationError::SurfaceIdCountMismatch {
                    surfaces: surfaces.len(),
                    ids: input.surface_ids.len(),
                });
            return None;
        }
        let ray_count = surfaces.len().checked_mul(input.rays_per_surface as usize);
        if input.rays_per_surface == 0
            || input.rays_per_surface > MAX_HOST_TRACE_RAYS_PER_SURFACE
            || ray_count.is_none_or(|n| n > MAX_HOST_TRACE_RAYS)
        {
            self.status = TracePreparationStatus::Failed(TracePreparationError::RayCapacity);
            return None;
        }
        let ray_count = ray_count.expect("checked above");
        if ray_count == 0 {
            self.status = TracePreparationStatus::NoSurfaces;
            return None;
        }
        self.surface_id_scratch.clear();
        if self.surface_id_scratch.capacity() < input.surface_ids.len()
            && self
                .surface_id_scratch
                .try_reserve_exact(input.surface_ids.len())
                .is_err()
        {
            self.status = TracePreparationStatus::Failed(TracePreparationError::RayCapacity);
            return None;
        }
        self.surface_id_scratch
            .extend_from_slice(&input.surface_ids);
        self.surface_id_scratch.sort_unstable();
        if let Some(pair) = self
            .surface_id_scratch
            .windows(2)
            .find(|pair| pair[0] == pair[1])
        {
            let id = pair[0];
            self.status =
                TracePreparationStatus::Failed(TracePreparationError::DuplicateSurfaceId(id));
            return None;
        }
        if self.gpu.is_none() {
            self.gpu = Some(LightingTraceGpu::new(Arc::clone(&device)));
        }
        if self.output.is_none() {
            let limit = device.capabilities().limits.max_storage_buffer_binding_size;
            if HOST_TRACE_OUTPUT_BYTES > limit {
                self.status = TracePreparationStatus::Failed(TracePreparationError::Trace(
                    TraceGpuError::StorageBindingLimit,
                ));
                return None;
            }
            let id = device.create_buffer(&rhi::BufferDescriptor::new(
                HOST_TRACE_OUTPUT_BYTES,
                rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_SRC,
            ));
            self.output = Some(TraceOutputBuffer {
                id,
                capacity_bytes: HOST_TRACE_OUTPUT_BYTES,
                device: Arc::clone(&device),
            });
        }

        if self.ray_scratch.capacity() < ray_count
            && self
                .ray_scratch
                .try_reserve_exact(ray_count - self.ray_scratch.len())
                .is_err()
        {
            self.status = TracePreparationStatus::Failed(TracePreparationError::RayCapacity);
            return None;
        }
        if self.raygen_scratch.capacity() < ray_count
            && self
                .raygen_scratch
                .try_reserve_exact(ray_count - self.raygen_scratch.len())
                .is_err()
        {
            self.status = TracePreparationStatus::Failed(TracePreparationError::RayCapacity);
            return None;
        }
        self.ray_scratch.clear();
        self.raygen_scratch.clear();
        let scene = BorrowedPackedTraceScene::from(input.bvh.as_ref());
        if self
            .validated_scene
            .as_ref()
            .is_none_or(|scene| !Arc::ptr_eq(scene, &input.bvh))
        {
            if let Err(error) = scene.validate() {
                self.status = TracePreparationStatus::Failed(TracePreparationError::Trace(error));
                return None;
            }
            self.validated_scene = Some(Arc::clone(&input.bvh));
        }
        for (surface, &stable_id) in surfaces.iter().zip(&input.surface_ids) {
            if let Err(error) = append_surface_gi_rays(
                SurfaceRayGenInput {
                    position: prism_math::Vec3::new(
                        surface.position[0],
                        surface.position[1],
                        surface.position[2],
                    ),
                    normal: prism_math::Vec3::new(
                        surface.normal[0],
                        surface.normal[1],
                        surface.normal[2],
                    ),
                    surface_stable_id: stable_id,
                    frame_index,
                    sample_index: 0,
                    ray_count: input.rays_per_surface,
                    epsilon: input.ray_epsilon,
                    tmax: input.ray_tmax,
                },
                &mut self.raygen_scratch,
            ) {
                self.status =
                    TracePreparationStatus::Failed(TracePreparationError::RayGeneration(error));
                return None;
            }
        }
        for (surface_index, rays) in self
            .raygen_scratch
            .chunks(input.rays_per_surface as usize)
            .enumerate()
        {
            let stable_id = input.surface_ids[surface_index];
            self.ray_scratch
                .extend(rays.iter().enumerate().map(|(sample_index, ray)| TraceRay {
                    origin: ray.origin.to_array(),
                    t_min: ray.tmin,
                    direction: ray.direction.to_array(),
                    t_max: ray.tmax,
                    direction_pdf: ray.pdf,
                    selection_u01: selection_hash(frame_index, stable_id, sample_index as u64),
                }));
        }
        let output = self.output.as_ref().expect("output allocated above");
        let recorder = self
            .gpu
            .as_mut()
            .expect("GPU pass created above")
            .prepare_graph(
                queue,
                &scene,
                &self.ray_scratch,
                output.id,
                output.capacity_bytes,
                candidate_target,
            );
        match recorder {
            Ok(recorder) => {
                self.status = TracePreparationStatus::Prepared {
                    rays: ray_count as u32,
                    frame_index,
                };
                Some(recorder)
            }
            Err(error) => {
                self.status = TracePreparationStatus::Failed(TracePreparationError::Trace(error));
                None
            }
        }
    }
}

/// Validated packed scene. Words use the architecture crate's frozen ABI.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedTraceScene {
    /// Flat `NODE_WORDS` records in depth-first order.
    pub nodes: Vec<u32>,
    /// Flat `TRIANGLE_WORDS` records indexed by leaf ranges.
    pub triangles: Vec<u32>,
    /// Optional per-triangle secondary-shading identity and attributes.
    pub hit_attributes: Option<Vec<RayHitAttributeRecord>>,
}

impl From<&GpuBvhBuffers> for PackedTraceScene {
    fn from(value: &GpuBvhBuffers) -> Self {
        Self {
            nodes: value.nodes.clone(),
            triangles: value.triangles.clone(),
            hit_attributes: value.hit_attributes.clone(),
        }
    }
}

/// Borrowed packed geometry view; validation and upload do not clone the BVH.
#[derive(Clone, Copy, Debug)]
pub struct BorrowedPackedTraceScene<'a> {
    /// Flat node ABI words.
    pub nodes: &'a [u32],
    /// Flat triangle ABI words.
    pub triangles: &'a [u32],
    /// Optional borrowed per-triangle secondary-shading payload.
    pub hit_attributes: Option<&'a [RayHitAttributeRecord]>,
}

impl<'a> From<&'a GpuBvhBuffers> for BorrowedPackedTraceScene<'a> {
    fn from(value: &'a GpuBvhBuffers) -> Self {
        Self {
            nodes: &value.nodes,
            triangles: &value.triangles,
            hit_attributes: value.hit_attributes.as_deref(),
        }
    }
}

impl<'a> From<&'a PackedTraceScene> for BorrowedPackedTraceScene<'a> {
    fn from(value: &'a PackedTraceScene) -> Self {
        Self {
            nodes: &value.nodes,
            triangles: &value.triangles,
            hit_attributes: value.hit_attributes.as_deref(),
        }
    }
}

trait PackedSceneWords {
    fn nodes(&self) -> &[u32];
    fn triangles(&self) -> &[u32];
    fn hit_attributes(&self) -> Option<&[RayHitAttributeRecord]>;
}

impl PackedSceneWords for PackedTraceScene {
    fn nodes(&self) -> &[u32] {
        &self.nodes
    }
    fn triangles(&self) -> &[u32] {
        &self.triangles
    }
    fn hit_attributes(&self) -> Option<&[RayHitAttributeRecord]> {
        self.hit_attributes.as_deref()
    }
}

impl PackedSceneWords for BorrowedPackedTraceScene<'_> {
    fn nodes(&self) -> &[u32] {
        self.nodes
    }
    fn triangles(&self) -> &[u32] {
        self.triangles
    }
    fn hit_attributes(&self) -> Option<&[RayHitAttributeRecord]> {
        self.hit_attributes
    }
}

impl PackedTraceScene {
    /// Checks ABI lengths, finite ordered bounds, leaf ranges, triangle data,
    /// reachability, cycles, and the exact maximum traversal stack occupancy.
    pub fn validate(&self) -> Result<(), TraceGpuError> {
        validate_packed_scene(self)
    }

    /// Validates the material-facing payload required by GI shading.
    pub fn validate_secondary_attributes(&self) -> Result<(), TraceGpuError> {
        validate_secondary_attributes(self)
    }
}

impl BorrowedPackedTraceScene<'_> {
    /// Validates the packed ABI without copying either backing array.
    pub fn validate(&self) -> Result<(), TraceGpuError> {
        validate_packed_scene(self)
    }

    /// Validates the optional material-facing payload required by GI shading.
    pub fn validate_secondary_attributes(&self) -> Result<(), TraceGpuError> {
        validate_secondary_attributes(self)
    }
}

fn validate_packed_scene(scene: &impl PackedSceneWords) -> Result<(), TraceGpuError> {
    let nodes = scene.nodes();
    let triangles = scene.triangles();
    if nodes.is_empty() || triangles.is_empty() {
        return Err(TraceGpuError::EmptyGeometry);
    }
    if nodes.len() % NODE_WORDS != 0 || triangles.len() % TRIANGLE_WORDS != 0 {
        return Err(TraceGpuError::MalformedPackedLength);
    }
    let node_count = nodes.len() / NODE_WORDS;
    let triangle_count = triangles.len() / TRIANGLE_WORDS;
    if node_count > MAX_NODE_COUNT || triangle_count > MAX_TRIANGLE_COUNT {
        return Err(TraceGpuError::GeometryLimit);
    }
    for n in 0..node_count {
        let b = n * NODE_WORDS;
        for a in 0..3 {
            let lo = f32::from_bits(nodes[b + a]);
            let hi = f32::from_bits(nodes[b + 3 + a]);
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(TraceGpuError::InvalidNode(n));
            }
        }
        let count = nodes[b + 8];
        if count == 0 {
            let second = nodes[b + 7] as usize;
            if second <= n + 1 || second >= node_count || nodes[b + 9] > 2 || n + 1 >= node_count {
                return Err(TraceGpuError::InvalidNode(n));
            }
        } else {
            let start = nodes[b + 6] as usize;
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
        let v = (0..9).all(|i| f32::from_bits(triangles[b + i]).is_finite());
        let a = vec3(triangles, b);
        let e1 = sub(vec3(triangles, b + 3), a);
        let e2 = sub(vec3(triangles, b + 6), a);
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
        if nodes[b + 8] == 0 {
            max_stack = max_stack.max(depth + 1);
            stack.push((nodes[b + 7] as usize, depth + 1));
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

fn validate_secondary_attributes(scene: &impl PackedSceneWords) -> Result<(), TraceGpuError> {
    let Some(attributes) = scene.hit_attributes() else {
        return Err(TraceGpuError::MissingHitAttributePayload);
    };
    let triangle_count = scene.triangles().len() / TRIANGLE_WORDS;
    if attributes.len() != triangle_count {
        return Err(TraceGpuError::InvalidHitAttributePayload(attributes.len()));
    }
    for (index, record) in attributes.iter().enumerate() {
        record
            .validate()
            .map_err(|_| TraceGpuError::InvalidHitAttributePayload(index))?;
    }
    Ok(())
}

/// CPU oracle over the exact packed input accepted by the GPU component.
pub fn trace_reference(
    scene: &PackedTraceScene,
    rays: &[TraceRay],
) -> Result<Vec<TraceHit>, TraceGpuError> {
    scene.validate()?;
    validate_rays(rays)?;
    let borrowed = BorrowedPackedTraceScene::from(scene);
    Ok(rays.iter().map(|ray| trace_one(&borrowed, *ray)).collect())
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
    /// Number of times the packed BVH node and triangle data was uploaded.
    /// A stable scene should increment this once for the lifetime of the pass.
    pub scene_uploads: u64,
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
/// `nodes`, `triangles`, `attributes`, and `rays` as
/// storage reads and `hits` as a storage write. The prepared input buffers are
/// not replaced while this recorder exists because preparation requires a
/// mutable borrow of the pass.
#[derive(Clone, Copy)]
pub struct TraceGraphRecorder {
    pub nodes: TraceGraphBuffer,
    pub triangles: TraceGraphBuffer,
    /// Per-triangle material-facing hit attributes.
    pub attributes: TraceGraphBuffer,
    pub rays: TraceGraphBuffer,
    pub hits: TraceGraphBuffer,
    /// Candidate buffer written by the post-trace candidate-build dispatch.
    pub candidates: Option<TraceGraphBuffer>,
    /// Radiance buffer consumed by candidate build.
    pub radiance: Option<TraceGraphBuffer>,
    /// Surface buffer consumed by candidate build.
    pub surfaces: Option<TraceGraphBuffer>,
    /// Lighting parameters consumed by candidate build.
    pub params: Option<TraceGraphBuffer>,
    pipeline: rhi::ComputePipelineId,
    bind_group: rhi::BindGroupId,
    workgroups: u32,
    candidate_pipeline: Option<rhi::ComputePipelineId>,
    candidate_bind_group: Option<rhi::BindGroupId>,
    candidate_workgroups: u32,
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

    /// Records the candidate-build dispatch after trace hits are available.
    pub fn record_candidate_graph(&self, context: &mut prism_render_graph::ExecuteContext<'_>) {
        if let (Some(pipeline), Some(bind_group)) =
            (self.candidate_pipeline, self.candidate_bind_group)
        {
            context
                .set_compute_pipeline(pipeline)
                .set_compute_bind_group(0, bind_group, vec![])
                .dispatch(self.candidate_workgroups, 1, 1);
        }
    }
}

#[derive(Clone, Copy)]
struct ResidentBuffer {
    id: rhi::BufferId,
    capacity_bytes: u64,
}

struct TraceOutputBuffer {
    id: rhi::BufferId,
    capacity_bytes: u64,
    device: Arc<dyn RenderDevice + Send + Sync>,
}

impl Drop for TraceOutputBuffer {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.id);
    }
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
    candidate_layout: rhi::BindGroupLayoutId,
    candidate_pipeline_layout: rhi::PipelineLayoutId,
    candidate_shader: rhi::ShaderModuleId,
    candidate_pipeline: rhi::ComputePipelineId,
    nodes: Option<ResidentBuffer>,
    triangles: Option<ResidentBuffer>,
    attributes: Option<ResidentBuffer>,
    rays: Option<ResidentBuffer>,
    retired_buffers: Vec<rhi::BufferId>,
    graph_bind_group: Option<(rhi::BindGroupId, [rhi::BufferId; 5])>,
    retired_bind_groups: Vec<rhi::BindGroupId>,
    candidate_bind_group: Option<(rhi::BindGroupId, [rhi::BufferId; 8])>,
    retired_candidate_bind_groups: Vec<rhi::BindGroupId>,
    packed_rays: Vec<u32>,
    uploaded_scene_identity: Option<(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        rhi::BufferId,
        rhi::BufferId,
        rhi::BufferId,
    )>,
    allocations: u64,
    scene_uploads: u64,
}

impl LightingTraceGpu {
    pub fn new(device: Arc<dyn RenderDevice + Send + Sync>) -> Self {
        let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("lighting-trace-gpu-layout".into()),
            entries: (0..5)
                .map(|binding| rhi::BindGroupLayoutEntry {
                    binding,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: rhi::BindingType::Buffer {
                        ty: rhi::BufferBindingType::Storage {
                            read_only: binding != 4,
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
        let candidate_layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("lighting-gi-candidate-build-layout".into()),
            entries: (0..8)
                .map(|binding| rhi::BindGroupLayoutEntry {
                    binding,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: if binding == 6 {
                        rhi::BindingType::Buffer {
                            ty: rhi::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(32),
                        }
                    } else {
                        rhi::BindingType::Buffer {
                            ty: rhi::BufferBindingType::Storage {
                                read_only: binding != 7,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    },
                    count: None,
                })
                .collect(),
        });
        let candidate_pipeline_layout =
            device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
                label: Some("lighting-gi-candidate-build-pipeline-layout".into()),
                bind_group_layouts: vec![candidate_layout],
                push_constant_ranges: vec![],
            });
        let candidate_shader =
            device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(CANDIDATE_WGSL));
        let candidate_pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("lighting-gi-candidate-build".into()),
            layout: Some(candidate_pipeline_layout),
            module: candidate_shader,
            entry_point: "main".into(),
        });
        Self {
            device,
            layout,
            pipeline_layout,
            shader,
            pipeline,
            candidate_layout,
            candidate_pipeline_layout,
            candidate_shader,
            candidate_pipeline,
            nodes: None,
            triangles: None,
            attributes: None,
            rays: None,
            retired_buffers: Vec::new(),
            graph_bind_group: None,
            retired_bind_groups: Vec::new(),
            candidate_bind_group: None,
            retired_candidate_bind_groups: Vec::new(),
            packed_rays: Vec::new(),
            uploaded_scene_identity: None,
            allocations: 0,
            scene_uploads: 0,
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
            scene_uploads: self.scene_uploads,
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
        scene: &BorrowedPackedTraceScene<'_>,
        rays: &[TraceRay],
        output: rhi::BufferId,
        output_size_bytes: u64,
        candidate_target: Option<LightingGiCandidateTarget>,
    ) -> Result<TraceGraphRecorder, TraceGpuError> {
        scene.validate()?;
        validate_rays(rays)?;
        if rays.is_empty() {
            return Err(TraceGpuError::EmptyRays);
        }
        let sizes = [
            byte_size(scene.nodes.len(), 4)?,
            byte_size(scene.triangles.len(), 4)?,
            byte_size(
                scene
                    .hit_attributes()
                    .map_or(1, <[RayHitAttributeRecord]>::len),
                core::mem::size_of::<RayHitAttributeRecord>(),
            )?,
            byte_size(rays.len(), RAY_WORDS * 4)?,
            byte_size(rays.len(), HIT_WORDS * 4)?,
        ];
        if candidate_target.is_some() {
            scene.validate_secondary_attributes()?;
        }
        validate_output_size(output_size_bytes, sizes[4])?;
        let caps = self.device.capabilities();
        let limit = caps.limits.max_storage_buffer_binding_size;
        if sizes.iter().any(|size| *size > limit) {
            return Err(TraceGpuError::StorageBindingLimit);
        }
        if let Some(target) = candidate_target {
            let expected_ray_count = (target.surface_count as usize)
                .checked_mul(target.rays_per_surface as usize)
                .ok_or(TraceGpuError::BufferSizeOverflow)?;
            if target.surface_count == 0
                || target.rays_per_surface == 0
                || target.rays_per_surface > MAX_HOST_TRACE_RAYS_PER_SURFACE
                || expected_ray_count != rays.len()
            {
                return Err(TraceGpuError::CandidateBatchShape);
            }
            let candidate_bytes =
                byte_size(rays.len(), core::mem::size_of::<LightingGiCandidateInput>())?;
            let radiance_bytes = byte_size(scene.triangles.len() / TRIANGLE_WORDS, 16)?;
            let surface_bytes = byte_size(target.surface_count as usize, 80)?;
            let candidate_binding_sizes = [
                sizes[3],
                sizes[4],
                sizes[1],
                sizes[2],
                target.radiance_size_bytes,
                target.surfaces_size_bytes,
                target.params_size_bytes,
                target.candidates_size_bytes,
            ];
            if candidate_binding_sizes.iter().any(|size| *size > limit) {
                return Err(TraceGpuError::CandidateStorageBindingLimit);
            }
            if target.candidates_size_bytes < candidate_bytes
                || target.radiance_size_bytes < radiance_bytes
                || target.surfaces_size_bytes < surface_bytes
                || target.params_size_bytes < 32
            {
                return Err(TraceGpuError::CandidateBufferTooSmall);
            }
            if target.params_size_bytes > caps.limits.max_uniform_buffer_binding_size {
                return Err(TraceGpuError::CandidateStorageBindingLimit);
            }
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
        let attribute_buffer = ensure_resident_buffer(
            &*self.device,
            &mut self.attributes,
            &mut self.retired_buffers,
            &mut self.allocations,
            sizes[2],
            limit,
        );
        let ray_buffer = ensure_resident_buffer(
            &*self.device,
            &mut self.rays,
            &mut self.retired_buffers,
            &mut self.allocations,
            sizes[3],
            limit,
        );
        let scene_identity = (
            scene.nodes.as_ptr() as usize,
            scene.nodes.len(),
            scene.triangles.as_ptr() as usize,
            scene.triangles.len(),
            scene
                .hit_attributes()
                .map_or(0, |attributes| attributes.as_ptr() as usize),
            scene
                .hit_attributes()
                .map_or(0, <[RayHitAttributeRecord]>::len),
            node_buffer,
            triangle_buffer,
            attribute_buffer,
        );
        if self.uploaded_scene_identity != Some(scene_identity) {
            queue.write_buffer(node_buffer, 0, bytemuck::cast_slice(scene.nodes));
            queue.write_buffer(triangle_buffer, 0, bytemuck::cast_slice(scene.triangles));
            let attributes = scene.hit_attributes().unwrap_or(&[]);
            if attributes.is_empty() {
                let invalid = [0u32; RayHitAttributeRecord::WORDS];
                queue.write_buffer(attribute_buffer, 0, bytemuck::cast_slice(&invalid));
            } else {
                queue.write_buffer(attribute_buffer, 0, bytemuck::cast_slice(attributes));
            }
            self.uploaded_scene_identity = Some(scene_identity);
            self.scene_uploads = self.scene_uploads.saturating_add(1);
        }
        self.pack_rays(rays, sizes[3])?;
        queue.write_buffer(ray_buffer, 0, bytemuck::cast_slice(&self.packed_rays));
        let buffers = [
            node_buffer,
            triangle_buffer,
            attribute_buffer,
            ray_buffer,
            output,
        ];
        let binding_sizes = [
            self.nodes.expect("node buffer was ensured").capacity_bytes,
            self.triangles
                .expect("triangle buffer was ensured")
                .capacity_bytes,
            self.attributes
                .expect("attribute buffer was ensured")
                .capacity_bytes,
            self.rays.expect("ray buffer was ensured").capacity_bytes,
            output_size_bytes,
        ];
        let group = if let Some((group, previous_buffers)) = self.graph_bind_group {
            if previous_buffers == buffers {
                group
            } else {
                self.retired_bind_groups.push(group);
                let group = self.create_graph_bind_group(buffers, binding_sizes);
                self.graph_bind_group = Some((group, buffers));
                group
            }
        } else {
            let group = self.create_graph_bind_group(buffers, binding_sizes);
            self.graph_bind_group = Some((group, buffers));
            group
        };
        let candidate = if let Some(target) = candidate_target {
            let candidate_buffers = [
                ray_buffer,
                output,
                triangle_buffer,
                attribute_buffer,
                target.radiance,
                target.surfaces,
                target.params,
                target.candidates,
            ];
            let candidate_sizes = [
                self.rays.expect("ray buffer was ensured").capacity_bytes,
                output_size_bytes,
                self.triangles
                    .expect("triangle buffer was ensured")
                    .capacity_bytes,
                self.attributes
                    .expect("attribute buffer was ensured")
                    .capacity_bytes,
                target.radiance_size_bytes,
                target.surfaces_size_bytes,
                target.params_size_bytes,
                target.candidates_size_bytes,
            ];
            let group = if let Some((group, previous)) = self.candidate_bind_group {
                if previous == candidate_buffers {
                    group
                } else {
                    self.retired_candidate_bind_groups.push(group);
                    let group =
                        self.create_candidate_bind_group(candidate_buffers, candidate_sizes);
                    self.candidate_bind_group = Some((group, candidate_buffers));
                    group
                }
            } else {
                let group = self.create_candidate_bind_group(candidate_buffers, candidate_sizes);
                self.candidate_bind_group = Some((group, candidate_buffers));
                group
            };
            Some((group, candidate_buffers))
        } else {
            None
        };
        let candidate_enabled = candidate.is_some();
        Ok(TraceGraphRecorder {
            nodes: TraceGraphBuffer {
                id: node_buffer,
                size_bytes: sizes[0],
            },
            triangles: TraceGraphBuffer {
                id: triangle_buffer,
                size_bytes: sizes[1],
            },
            attributes: TraceGraphBuffer {
                id: attribute_buffer,
                size_bytes: sizes[2],
            },
            rays: TraceGraphBuffer {
                id: ray_buffer,
                size_bytes: sizes[3],
            },
            hits: TraceGraphBuffer {
                id: output,
                size_bytes: sizes[4],
            },
            pipeline: self.pipeline,
            bind_group: group,
            workgroups,
            candidates: candidate_target.map(|target| TraceGraphBuffer {
                id: target.candidates,
                size_bytes: target.candidates_size_bytes,
            }),
            radiance: candidate_target.map(|target| TraceGraphBuffer {
                id: target.radiance,
                size_bytes: target.radiance_size_bytes,
            }),
            surfaces: candidate_target.map(|target| TraceGraphBuffer {
                id: target.surfaces,
                size_bytes: target.surfaces_size_bytes,
            }),
            params: candidate_target.map(|target| TraceGraphBuffer {
                id: target.params,
                size_bytes: target.params_size_bytes,
            }),
            candidate_pipeline: candidate_enabled.then_some(self.candidate_pipeline),
            candidate_bind_group: candidate.map(|(group, _)| group),
            candidate_workgroups: workgroups,
        })
    }

    fn create_candidate_bind_group(
        &self,
        buffers: [rhi::BufferId; 8],
        sizes: [u64; 8],
    ) -> rhi::BindGroupId {
        self.device.create_bind_group(&rhi::BindGroupDescriptor {
            label: Some("lighting-gi-candidate-build-bind-group".into()),
            layout: Some(self.candidate_layout),
            entries: buffers
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
        })
    }

    fn create_graph_bind_group(
        &self,
        buffers: [rhi::BufferId; 5],
        sizes: [u64; 5],
    ) -> rhi::BindGroupId {
        self.device.create_bind_group(&rhi::BindGroupDescriptor {
            label: Some("lighting-trace-gpu-graph-resources".into()),
            layout: Some(self.layout),
            entries: buffers
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
                ray.direction_pdf.to_bits(),
                ray.selection_u01.to_bits(),
                0,
                0,
            ]);
        }
        Ok(())
    }
}

impl Drop for LightingTraceGpu {
    fn drop(&mut self) {
        if let Some((bind_group, _)) = self.graph_bind_group.take() {
            self.device.destroy_bind_group(bind_group);
        }
        for bind_group in self.retired_bind_groups.drain(..) {
            self.device.destroy_bind_group(bind_group);
        }
        if let Some((bind_group, _)) = self.candidate_bind_group.take() {
            self.device.destroy_bind_group(bind_group);
        }
        for bind_group in self.retired_candidate_bind_groups.drain(..) {
            self.device.destroy_bind_group(bind_group);
        }
        for buffer in [self.nodes, self.triangles, self.attributes, self.rays]
            .into_iter()
            .flatten()
        {
            self.device.destroy_buffer(buffer.id);
        }
        for buffer in self.retired_buffers.drain(..) {
            self.device.destroy_buffer(buffer);
        }
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device
            .destroy_compute_pipeline(self.candidate_pipeline);
        self.device.destroy_shader_module(self.shader);
        self.device.destroy_shader_module(self.candidate_shader);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device
            .destroy_pipeline_layout(self.candidate_pipeline_layout);
        self.device.destroy_bind_group_layout(self.layout);
        self.device.destroy_bind_group_layout(self.candidate_layout);
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
            || !r.direction_pdf.is_finite()
            || r.direction_pdf <= 0.0
            || !r.selection_u01.is_finite()
            || !(0.0..1.0).contains(&r.selection_u01)
            || r.t_min < 0.0
            || r.t_max < r.t_min
            || dot(r.direction, r.direction) <= 1.0e-20
        {
            return Err(TraceGpuError::InvalidRay(i));
        }
    }
    Ok(())
}

fn trace_one(scene: &BorrowedPackedTraceScene<'_>, ray: TraceRay) -> TraceHit {
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
                            triangle_index: tri as u32,
                            instance_id: u32::MAX,
                            geometry_id: u32::MAX,
                            material_generation: u32::MAX,
                            uv: [0.0; 2],
                            shading_normal: [0.0; 3],
                            t,
                            u,
                            v,
                        };
                        if let Some(attributes) = scene
                            .hit_attributes()
                            .and_then(|records| records.get(tri))
                            .and_then(|record| record.interpolate(u, v).ok())
                        {
                            hit.instance_id = attributes.identity.instance_id;
                            hit.geometry_id = attributes.identity.geometry_id;
                            hit.material_generation = attributes.identity.material_generation;
                            hit.uv = attributes.uv;
                            hit.shading_normal = attributes.shading_normal;
                        }
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
@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> triangles: array<u32>;
@group(0) @binding(2) var<storage, read> attributes: array<u32>;
@group(0) @binding(3) var<storage, read> rays: array<u32>;
@group(0) @binding(4) var<storage, read_write> hits: array<u32>;
fn f(w:u32)->f32{return bitcast<f32>(w);}
fn vec3at(b:u32)->vec3<f32>{return vec3<f32>(f(triangles[b]),f(triangles[b+1u]),f(triangles[b+2u]));}
fn finite(x:f32)->bool{return ((bitcast<u32>(x)>>23u)&255u)!=255u;}
fn finite3(v:vec3<f32>)->bool{return finite(v.x)&&finite(v.y)&&finite(v.z);}
fn slab(b:u32,o:vec3<f32>,d:vec3<f32>,tmin:f32,tmax:f32)->bool{
 var lo=tmin; var hi=tmax;
 for(var a=0u;a<3u;a++){let mn=f(nodes[b+a]);let mx=f(nodes[b+3u+a]);let x=o[a];let v=d[a];if(v==0.0){if(x<mn||x>mx){return false;}}else{let inv=1.0/v;var t0=(mn-x)*inv;var t1=(mx-x)*inv;if(t0>t1){let tmp=t0;t0=t1;t1=tmp;}lo=max(lo,t0);hi=min(hi,t1);if(lo>hi){return false;}}} return true;
}
fn tri_hit(b:u32,o:vec3<f32>,d:vec3<f32>,tmin:f32,tmax:f32)->vec3<f32>{
 let a=vec3at(b);let e1=vec3at(b+3u)-a;let e2=vec3at(b+6u)-a;let p=cross(d,e2);let det=dot(e1,p);if(abs(det)<=1.0e-8){return vec3<f32>(-1.0);}let inv=1.0/det;let tv=o-a;let u=dot(tv,p)*inv;if(u<0.0||u>1.0){return vec3<f32>(-1.0);}let q=cross(tv,e1);let v=dot(d,q)*inv;if(v<0.0||u+v>1.0){return vec3<f32>(-1.0);}let t=dot(e2,q)*inv;if(t<tmin||t>tmax){return vec3<f32>(-1.0);}return vec3<f32>(t,u,v);
}
fn clear_hit(hb:u32,state:u32){hits[hb]=state;for(var i=1u;i<16u;i++){hits[hb+i]=0u;}hits[hb+1u]=0xffffffffu;hits[hb+5u]=0xffffffffu;hits[hb+6u]=0xffffffffu;hits[hb+7u]=0xffffffffu;hits[hb+8u]=0xffffffffu;}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let ri=gid.x;if(ri>=arrayLength(&rays)/12u){return;}
 let rb=ri*12u;let hb=ri*16u;let o=vec3<f32>(f(rays[rb]),f(rays[rb+1u]),f(rays[rb+2u]));let tmin=f(rays[rb+3u]);let d=vec3<f32>(f(rays[rb+4u]),f(rays[rb+5u]),f(rays[rb+6u]));var closest=f(rays[rb+7u]);var best_id=0xffffffffu;var best_triangle=0xffffffffu;var bu=0.0;var bv=0.0;var node=0u;var stack:array<u32,64>;var sp=0u;var steps=0u;
 loop{if(steps>=arrayLength(&nodes)/12u*2u+1u){clear_hit(hb,2u);return;}steps++;let nb=node*12u;if(slab(nb,o,d,tmin,closest)){let count=nodes[nb+8u];if(count>0u){let start=nodes[nb+6u];for(var i=0u;i<count;i++){let ti=start+i;let tb=ti*12u;let h=tri_hit(tb,o,d,tmin,closest);if(h.x>=0.0){closest=h.x;best_id=triangles[tb+9u];best_triangle=ti;bu=h.y;bv=h.z;}}if(sp==0u){break;}sp--;node=stack[sp];}else{let left=node+1u;let right=nodes[nb+7u];let axis=nodes[nb+9u];var near=left;var far=right;if(d[axis]<0.0){near=right;far=left;}if(sp>=64u){clear_hit(hb,2u);return;}stack[sp]=far;sp++;node=near;}}else{if(sp==0u){break;}sp--;node=stack[sp];}}
 if(best_id==0xffffffffu){clear_hit(hb,0u);return;}
 hits[hb]=1u;hits[hb+1u]=best_id;hits[hb+2u]=bitcast<u32>(closest);hits[hb+3u]=bitcast<u32>(bu);hits[hb+4u]=bitcast<u32>(bv);hits[hb+5u]=best_triangle;
 hits[hb+6u]=0xffffffffu;hits[hb+7u]=0xffffffffu;hits[hb+8u]=0xffffffffu;hits[hb+14u]=0u;hits[hb+15u]=0u;
 let ab=best_triangle*20u;if(ab+19u>=arrayLength(&attributes)){return;}
 let ids=vec3<u32>(attributes[ab],attributes[ab+1u],attributes[ab+2u]);if(ids.x==0xffffffffu||ids.y==0xffffffffu||ids.z==0xffffffffu){return;}
 let w=1.0-bu-bv;let uv=vec2<f32>(w*f(attributes[ab+4u])+bu*f(attributes[ab+6u])+bv*f(attributes[ab+8u]),w*f(attributes[ab+5u])+bu*f(attributes[ab+7u])+bv*f(attributes[ab+9u]));let nraw=vec3<f32>(w*f(attributes[ab+10u])+bu*f(attributes[ab+13u])+bv*f(attributes[ab+16u]),w*f(attributes[ab+11u])+bu*f(attributes[ab+14u])+bv*f(attributes[ab+17u]),w*f(attributes[ab+12u])+bu*f(attributes[ab+15u])+bv*f(attributes[ab+18u]));let nlen=dot(nraw,nraw);if(!finite(uv.x)||!finite(uv.y)||!finite3(nraw)||!finite(nlen)||nlen<=1.0e-20){return;}let n=nraw*inverseSqrt(nlen);if(!finite3(n)){return;}hits[hb+6u]=ids.x;hits[hb+7u]=ids.y;hits[hb+8u]=ids.z;hits[hb+9u]=bitcast<u32>(uv.x);hits[hb+10u]=bitcast<u32>(uv.y);hits[hb+11u]=bitcast<u32>(n.x);hits[hb+12u]=bitcast<u32>(n.y);hits[hb+13u]=bitcast<u32>(n.z);
}
"#;

const CANDIDATE_WGSL: &str = r#"
struct Params { counts: vec4<u32>, flags: vec4<u32> }
struct Surface { position: vec4<f32>, normal: vec4<f32>, albedo: vec4<f32>, ambient: vec4<f32>, candidate_range: vec4<u32> }
struct Candidate { position_source_pdf: vec4<f32>, normal_target_pdf: vec4<f32>, radiance_selection_u01: vec4<f32>, visibility_padding: vec4<f32> }
@group(0) @binding(0) var<storage, read> rays: array<u32>;
@group(0) @binding(1) var<storage, read> hits: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read> attributes: array<u32>;
@group(0) @binding(4) var<storage, read> outgoing: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> surfaces: array<Surface>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read_write> candidates: array<Candidate>;
fn f(w:u32)->f32{return bitcast<f32>(w);}
fn finite(x:f32)->bool{return ((bitcast<u32>(x)>>23u)&255u)!=255u;}
fn finite3(v:vec3<f32>)->bool{return finite(v.x)&&finite(v.y)&&finite(v.z);}
fn safe_normalize(v:vec3<f32>)->vec3<f32>{let n=dot(v,v);if(n>1.0e-20&&finite(n)){return v*inverseSqrt(n);}return vec3<f32>(0.0,1.0,0.0);}
fn luminance(v:vec3<f32>)->f32{return dot(v,vec3<f32>(0.2126,0.7152,0.0722));}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let ri=gid.x; let rays_per_surface=params.counts.w; if(rays_per_surface==0u||ri>=arrayLength(&rays)/12u){return;}
 let si=ri/rays_per_surface; let local=ri%rays_per_surface; if(si>=params.counts.x||local>=64u){return;}
 let surface=surfaces[si]; let rb=ri*12u; let hb=ri*16u; let base=surface.candidate_range.x+local; if(base>=arrayLength(&candidates)){return;}
 var c:Candidate; c.position_source_pdf=vec4<f32>(0.0,0.0,0.0,f(rays[rb+8u])); c.normal_target_pdf=vec4<f32>(0.0,1.0,0.0,0.0); c.radiance_selection_u01=vec4<f32>(0.0,0.0,0.0,0.0); c.visibility_padding=vec4<f32>(0.0);
 c.radiance_selection_u01.w=f(rays[rb+9u]);
 if(!finite(c.radiance_selection_u01.w)||c.radiance_selection_u01.w<0.0||c.radiance_selection_u01.w>=1.0){return;}
 let hit_state=hits[hb]; if(hit_state==2u){c.visibility_padding.x=-1.0;candidates[base]=c;return;} if(hit_state==0u){candidates[base]=c;return;} if(hit_state!=1u){c.visibility_padding.x=-1.0;candidates[base]=c;return;}
 let tri=hits[hb+5u]; if(tri>=params.flags.w||tri>=arrayLength(&outgoing)||tri>=(arrayLength(&triangles)/12u)){c.visibility_padding.x=-1.0;candidates[base]=c;return;} let tc=tri*12u; let ab=tri*20u;
 if(triangles[tc+9u]!=hits[hb+1u]||ab+19u>=arrayLength(&attributes)){c.visibility_padding.x=-1.0;candidates[base]=c;return;}
 if(attributes[ab]==0xffffffffu||attributes[ab+1u]==0xffffffffu||attributes[ab+2u]==0xffffffffu||attributes[ab]!=hits[hb+6u]||attributes[ab+1u]!=hits[hb+7u]||attributes[ab+2u]!=hits[hb+8u]){c.visibility_padding.x=-1.0;candidates[base]=c;return;}
 let o=vec3<f32>(f(rays[rb]),f(rays[rb+1u]),f(rays[rb+2u])); let d=vec3<f32>(f(rays[rb+4u]),f(rays[rb+5u]),f(rays[rb+6u])); let t=f(hits[hb+2u]); let u=f(hits[hb+3u]); let v=f(hits[hb+4u]);
 if(!finite(t)||!finite(u)||!finite(v)||t<f(rays[rb+3u])||t>f(rays[rb+7u])||u<0.0||v<0.0||u+v>1.00001){c.visibility_padding.x=-1.0;candidates[base]=c;return;}
 let w=1.0-u-v;let uv=vec2<f32>(w*f(attributes[ab+4u])+u*f(attributes[ab+6u])+v*f(attributes[ab+8u]),w*f(attributes[ab+5u])+u*f(attributes[ab+7u])+v*f(attributes[ab+9u]));let nraw=vec3<f32>(w*f(attributes[ab+10u])+u*f(attributes[ab+13u])+v*f(attributes[ab+16u]),w*f(attributes[ab+11u])+u*f(attributes[ab+14u])+v*f(attributes[ab+17u]),w*f(attributes[ab+12u])+u*f(attributes[ab+15u])+v*f(attributes[ab+18u]));let nlen=dot(nraw,nraw);if(!finite(uv.x)||!finite(uv.y)||!finite3(nraw)||!finite(nlen)||nlen<=1.0e-20){c.visibility_padding.x=-1.0;candidates[base]=c;return;}let n=nraw*inverseSqrt(nlen);if(!finite3(n)){c.visibility_padding.x=-1.0;candidates[base]=c;return;}let hit_uv=vec2<f32>(f(hits[hb+9u]),f(hits[hb+10u]));let hit_n=vec3<f32>(f(hits[hb+11u]),f(hits[hb+12u]),f(hits[hb+13u]));if(any(abs(uv-hit_uv)>vec2<f32>(1.0e-5))||any(abs(n-hit_n)>vec3<f32>(1.0e-5))){c.visibility_padding.x=-1.0;candidates[base]=c;return;} let p=o+d*t; let wi=safe_normalize(d); let cos_primary=max(dot(safe_normalize(surface.normal.xyz),wi),0.0); let le=outgoing[tri].xyz; if(!finite3(le)||any(le<vec3<f32>(0.0))){c.visibility_padding.x=-1.0;candidates[base]=c;return;} let target_density=luminance(surface.albedo.xyz*le*(cos_primary/3.141592653589793)); if(!finite(target_density)||target_density<0.0){c.visibility_padding.x=-1.0;candidates[base]=c;return;}
 c.position_source_pdf=vec4<f32>(p,f(rays[rb+8u])); c.normal_target_pdf=vec4<f32>(n,target_density); c.radiance_selection_u01=vec4<f32>(le,c.radiance_selection_u01.w); c.visibility_padding=vec4<f32>(1.0,0.0,0.0,0.0); candidates[base]=c;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::ray_scene::{Bvh, BvhBuildConfig, Triangle};
    use prism_render_driver::{BufferDescriptor, BufferState, BufferUsages, Instance as _};
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
    use prism_render_graph::{BufferDesc, ImportedBuffer, PassFlags, PassKind, RenderGraph};

    // Capture the descriptors sent by the production constructor/graph path;
    // the null backend itself intentionally does not validate storage access.
    struct DescriptorDevice {
        inner: prism_render_driver_null::NullDevice,
        layouts: std::sync::Mutex<Vec<rhi::BindGroupLayoutDescriptor>>,
        groups: std::sync::Mutex<Vec<rhi::BindGroupDescriptor>>,
    }

    macro_rules! forward_device_calls {
        ($($name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*) => {
            $(fn $name(&self, $($arg: $ty),*) $(-> $ret)? {
                self.inner.$name($($arg),*)
            })*
        };
    }

    impl RenderDevice for DescriptorDevice {
        fn capabilities(&self) -> &rhi::DeviceCapabilities {
            self.inner.capabilities()
        }

        fn create_bind_group_layout(
            &self,
            descriptor: &rhi::BindGroupLayoutDescriptor,
        ) -> rhi::BindGroupLayoutId {
            self.layouts.lock().unwrap().push(descriptor.clone());
            self.inner.create_bind_group_layout(descriptor)
        }

        fn create_bind_group(&self, descriptor: &rhi::BindGroupDescriptor) -> rhi::BindGroupId {
            self.groups.lock().unwrap().push(descriptor.clone());
            self.inner.create_bind_group(descriptor)
        }

        forward_device_calls! {
            create_buffer(descriptor: &rhi::BufferDescriptor) -> rhi::BufferId;
            create_texture(descriptor: &rhi::TextureDescriptor) -> rhi::TextureId;
            create_texture_view(texture: rhi::TextureId, descriptor: &rhi::TextureViewDescriptor) -> rhi::TextureViewId;
            create_sampler(descriptor: &rhi::SamplerDescriptor) -> rhi::SamplerId;
            create_shader_module(descriptor: &rhi::ShaderModuleDescriptor) -> rhi::ShaderModuleId;
            create_pipeline_layout(descriptor: &rhi::PipelineLayoutDescriptor) -> rhi::PipelineLayoutId;
            create_render_pipeline(descriptor: &rhi::RenderPipelineDescriptor) -> rhi::RenderPipelineId;
            create_compute_pipeline(descriptor: &rhi::ComputePipelineDescriptor) -> rhi::ComputePipelineId;
            destroy_buffer(id: rhi::BufferId);
            destroy_texture(id: rhi::TextureId);
            destroy_texture_view(id: rhi::TextureViewId);
            destroy_sampler(id: rhi::SamplerId);
            destroy_shader_module(id: rhi::ShaderModuleId);
            destroy_bind_group_layout(id: rhi::BindGroupLayoutId);
            destroy_bind_group(id: rhi::BindGroupId);
            destroy_pipeline_layout(id: rhi::PipelineLayoutId);
            destroy_render_pipeline(id: rhi::RenderPipelineId);
            destroy_compute_pipeline(id: rhi::ComputePipelineId);
        }
    }

    fn descriptor_device() -> (Arc<DescriptorDevice>, prism_render_driver_null::NullQueue) {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance
            .request_adapter(&rhi::RequestAdapterOptions::default())
            .unwrap();
        let (inner, queue) = instance
            .create_device(&adapter, &rhi::DeviceDescriptor::default())
            .unwrap();
        (
            Arc::new(DescriptorDevice {
                inner,
                layouts: std::sync::Mutex::new(Vec::new()),
                groups: std::sync::Mutex::new(Vec::new()),
            }),
            queue,
        )
    }

    fn assert_layout_bindings(
        layout: &rhi::BindGroupLayoutDescriptor,
        shader: &str,
        expected: &[(rhi::BindingType, &str)],
    ) {
        assert!(layout.has_unique_bindings());
        assert_eq!(layout.entries.len(), expected.len());
        let declarations: Vec<_> = shader
            .lines()
            .filter(|line| line.starts_with("@group(0) @binding("))
            .collect();
        assert_eq!(declarations.len(), expected.len());
        for (binding, &(ty, declaration)) in expected.iter().enumerate() {
            assert_eq!(
                layout.entries[binding],
                rhi::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty,
                    count: None,
                },
                "layout binding {binding}: {declaration}"
            );
            assert_eq!(
                declarations[binding],
                format!("@group(0) @binding({binding}) {declaration}")
            );
        }
    }

    #[test]
    fn trace_layout_descriptor_matches_attributes_rays_and_writable_hits() {
        use rhi::BufferBindingType::Storage;
        let (device, _) = descriptor_device();
        let _trace = LightingTraceGpu::new(device.clone());
        let layouts = device.layouts.lock().unwrap();
        assert_eq!(layouts.len(), 2);
        assert_eq!(
            layouts[0].label.as_deref(),
            Some("lighting-trace-gpu-layout")
        );
        assert_layout_bindings(
            &layouts[0],
            TRACE_WGSL,
            &[
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> nodes: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> triangles: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> attributes: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> rays: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read_write> hits: array<u32>;",
                ),
            ],
        );
    }

    #[test]
    fn candidate_layout_descriptor_matches_uniform_params_and_writable_candidates() {
        use rhi::BufferBindingType::{Storage, Uniform};
        let (device, _) = descriptor_device();
        let _trace = LightingTraceGpu::new(device.clone());
        let layouts = device.layouts.lock().unwrap();
        assert_eq!(layouts.len(), 2);
        assert_eq!(
            layouts[1].label.as_deref(),
            Some("lighting-gi-candidate-build-layout")
        );
        assert_layout_bindings(
            &layouts[1],
            CANDIDATE_WGSL,
            &[
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> rays: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> hits: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> triangles: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> attributes: array<u32>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> outgoing: array<vec4<f32>>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read> surfaces: array<Surface>;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(32),
                    },
                    "var<uniform> params: Params;",
                ),
                (
                    rhi::BindingType::Buffer {
                        ty: Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "var<storage, read_write> candidates: array<Candidate>;",
                ),
            ],
        );
    }

    #[test]
    fn graph_bind_group_descriptors_preserve_ray_hit_abi_resource_order_and_ranges() {
        let (device, queue) = descriptor_device();
        let mut trace = LightingTraceGpu::new(device.clone());
        let (scene, ray, _) = candidate_fixture();
        // Distinct, padded ranges expose swapped resources and size indices.
        let sizes = [128, 144, 160, 176, 192];
        let buffers = sizes.map(|size| {
            device.create_buffer(&BufferDescriptor::new(
                size,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            ))
        });
        let [hits, radiance, surfaces, params, candidates] = buffers;
        let recorder = trace
            .prepare_graph(
                &queue,
                &BorrowedPackedTraceScene::from(&scene),
                &[ray],
                hits,
                sizes[0],
                Some(LightingGiCandidateTarget {
                    candidates,
                    candidates_size_bytes: sizes[4],
                    radiance,
                    radiance_size_bytes: sizes[1],
                    surfaces,
                    surfaces_size_bytes: sizes[2],
                    params,
                    params_size_bytes: sizes[3],
                    surface_count: 1,
                    rays_per_surface: 1,
                }),
            )
            .unwrap();
        let nodes = trace.nodes.unwrap();
        let triangles = trace.triangles.unwrap();
        let attributes = trace.attributes.unwrap();
        let rays = trace.rays.unwrap();
        let groups = device.groups.lock().unwrap();
        assert_eq!(groups.len(), 2);
        for (group, layout, label, expected) in [
            (
                &groups[0],
                trace.layout,
                "lighting-trace-gpu-graph-resources",
                vec![
                    (nodes.id, nodes.capacity_bytes),
                    (triangles.id, triangles.capacity_bytes),
                    (attributes.id, attributes.capacity_bytes),
                    (rays.id, rays.capacity_bytes),
                    (hits, sizes[0]),
                ],
            ),
            (
                &groups[1],
                trace.candidate_layout,
                "lighting-gi-candidate-build-bind-group",
                vec![
                    (rays.id, rays.capacity_bytes),
                    (hits, sizes[0]),
                    (triangles.id, triangles.capacity_bytes),
                    (attributes.id, attributes.capacity_bytes),
                    (radiance, sizes[1]),
                    (surfaces, sizes[2]),
                    (params, sizes[3]),
                    (candidates, sizes[4]),
                ],
            ),
        ] {
            assert_eq!(group.label.as_deref(), Some(label));
            assert_eq!(group.layout, Some(layout));
            assert!(group.has_unique_bindings());
            let entries: Vec<_> = expected
                .into_iter()
                .enumerate()
                .map(|(binding, (buffer, size))| rhi::BindGroupEntry {
                    binding: binding as u32,
                    resource: rhi::BindingResource::Buffer {
                        buffer,
                        offset: 0,
                        size: Some(size),
                    },
                })
                .collect();
            assert_eq!(group.entries, entries);
        }
        drop(groups);
        drop(recorder);
        drop(trace);
        for buffer in buffers {
            device.destroy_buffer(buffer);
        }
        assert_eq!(device.inner.live_resource_count(), 0);
    }

    fn candidate_fixture() -> (PackedTraceScene, TraceRay, TraceHit) {
        use prism_render_architecture::ray_scene::{
            Bvh, RayHitAttributeRecord, RayHitIdentity, Triangle,
        };
        let bvh = Bvh::build(&[Triangle::new(
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            19,
        )]);
        let packed_bvh = GpuBvhBuffers::from_bvh(&bvh)
            .with_hit_attributes(vec![RayHitAttributeRecord::new(
                RayHitIdentity::new(3, 19, 7),
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0, 1.0]; 3],
            )])
            .expect("valid hit attributes");
        let packed = PackedTraceScene::from(&packed_bvh);
        let ray = TraceRay {
            origin: [0.0, 0.0, 0.0],
            t_min: 0.0,
            direction: [0.0, 0.0, -1.0],
            t_max: 10.0,
            direction_pdf: 0.25,
            selection_u01: 0.5,
        };
        let hit = TraceHit {
            hit: true,
            primitive: 19,
            triangle_index: 0,
            instance_id: 3,
            geometry_id: 19,
            material_generation: 7,
            uv: [0.25, 0.5],
            shading_normal: [0.0, 0.0, 1.0],
            t: 1.0,
            u: 0.25,
            v: 0.5,
        };
        (packed, ray, hit)
    }

    fn candidate_inputs() -> (Vec<LightingSurfaceInput>, Vec<u64>, Vec<[f32; 3]>) {
        (
            vec![LightingSurfaceInput {
                position: [0.0; 3],
                normal: [0.0, 0.0, -1.0],
                albedo: [0.8, 0.5, 0.25],
                ambient: [0.0; 3],
            }],
            vec![42],
            vec![[4.0, 2.0, 1.0]],
        )
    }

    fn packed_scene_with_attributes(packed: &GpuBvhBuffers) -> PackedTraceScene {
        let attributes = packed
            .triangles
            .chunks_exact(TRIANGLE_WORDS)
            .map(|triangle| {
                let a = vec3(triangle, 0);
                let b = vec3(triangle, 3);
                let c = vec3(triangle, 6);
                let normal = normalize(cross(sub(b, a), sub(c, a)));
                RayHitAttributeRecord::new(
                    prism_render_architecture::ray_scene::RayHitIdentity::new(1, triangle[9], 7),
                    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    [normal; 3],
                )
            })
            .collect();
        PackedTraceScene::from(
            &packed
                .clone()
                .with_hit_attributes(attributes)
                .expect("test attributes match packed triangles"),
        )
    }

    #[test]
    fn gi_candidate_reference_hit_uses_solid_angle_lambert_integrand_and_geometric_normal() {
        let (scene, ray, hit) = candidate_fixture();
        let (surfaces, ids, radiance) = candidate_inputs();
        let batches =
            build_gi_candidates_reference(&scene, &[ray], &[hit], &surfaces, &ids, 7, &radiance)
                .unwrap();
        let candidate = batches[0][0];
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(candidate.source_pdf, 0.25);
        assert_eq!(candidate.position, [0.0, 0.0, -1.0]);
        assert_eq!(candidate.normal, [0.0, 0.0, 1.0]);
        assert_eq!(candidate.radiance, radiance[0]);
        let expected =
            (0.8 * 4.0 * 0.2126 + 0.5 * 2.0 * 0.7152 + 0.25 * 1.0 * 0.0722) / core::f32::consts::PI;
        assert!((candidate.target_pdf - expected).abs() < 1.0e-6);
        assert_eq!(candidate.visibility, 1.0);
    }

    #[test]
    fn gi_candidate_reference_miss_remains_a_zero_target_proposal() {
        let (scene, ray, _) = candidate_fixture();
        let (surfaces, ids, radiance) = candidate_inputs();
        let batches = build_gi_candidates_reference(
            &scene,
            &[ray],
            &[TraceHit::MISS],
            &surfaces,
            &ids,
            7,
            &radiance,
        )
        .unwrap();
        assert_eq!(batches[0].len(), 1, "N includes missed proposals");
        assert_eq!(batches[0][0].source_pdf, ray.direction_pdf);
        assert_eq!(batches[0][0].target_pdf, 0.0);
        assert_eq!(batches[0][0].visibility, 0.0);
    }

    #[test]
    fn gi_candidate_reference_zero_contribution_hit_remains_in_the_proposal_batch() {
        let (scene, ray, hit) = candidate_fixture();
        let (mut surfaces, ids, radiance) = candidate_inputs();
        surfaces[0].albedo = [0.0; 3];
        let batches =
            build_gi_candidates_reference(&scene, &[ray], &[hit], &surfaces, &ids, 7, &radiance)
                .unwrap();
        assert_eq!(batches[0].len(), 1, "N includes zero-contribution hits");
        assert_eq!(batches[0][0].source_pdf, ray.direction_pdf);
        assert_eq!(batches[0][0].target_pdf, 0.0);
        assert_eq!(batches[0][0].radiance, radiance[0]);
    }

    #[test]
    fn gi_candidate_reference_hash_is_deterministic_and_keyed_by_frame_surface_ray() {
        let (scene, ray, hit) = candidate_fixture();
        let (surfaces, ids, radiance) = candidate_inputs();
        let run = |frame| {
            build_gi_candidates_reference(&scene, &[ray], &[hit], &surfaces, &ids, frame, &radiance)
                .unwrap()[0][0]
                .selection_u01
        };
        let a = run(7);
        assert_eq!(a, run(7));
        assert!((0.0..1.0).contains(&a));
        assert_ne!(a, run(8));
        let other_surface =
            build_gi_candidates_reference(&scene, &[ray], &[hit], &surfaces, &[43], 7, &radiance)
                .unwrap()[0][0]
                .selection_u01;
        assert_ne!(a, other_surface);
        let mut second_ray = ray;
        second_ray.origin[0] = -0.2;
        let mut second_hit = hit;
        second_hit.u = 0.15;
        second_hit.uv = [second_hit.u, second_hit.v];
        let ray_indexed = build_gi_candidates_reference(
            &scene,
            &[ray, second_ray],
            &[hit, second_hit],
            &surfaces,
            &ids,
            7,
            &radiance,
        )
        .unwrap();
        assert_ne!(
            ray_indexed[0][0].selection_u01,
            ray_indexed[0][1].selection_u01
        );
    }

    #[test]
    fn gi_candidate_reference_rejects_malformed_batches_pdf_indices_and_radiance() {
        let (scene, ray, hit) = candidate_fixture();
        let (surfaces, ids, radiance) = candidate_inputs();
        assert_eq!(
            build_gi_candidates_reference(&scene, &[], &[], &surfaces, &ids, 0, &radiance),
            Err(GiCandidateReferenceError::InvalidBatchShape)
        );
        let mut bad_ray = ray;
        bad_ray.direction_pdf = f32::NAN;
        assert_eq!(
            build_gi_candidates_reference(
                &scene,
                &[bad_ray],
                &[hit],
                &surfaces,
                &ids,
                0,
                &radiance
            ),
            Err(GiCandidateReferenceError::InvalidRay(0))
        );
        let mut bad_hit = hit;
        bad_hit.triangle_index = 3;
        assert_eq!(
            build_gi_candidates_reference(
                &scene,
                &[ray],
                &[bad_hit],
                &surfaces,
                &ids,
                0,
                &radiance
            ),
            Err(GiCandidateReferenceError::InvalidTriangleIndex(0))
        );
        assert_eq!(
            build_gi_candidates_reference(
                &scene,
                &[ray],
                &[hit],
                &surfaces,
                &ids,
                0,
                &[[f32::INFINITY, 0.0, 0.0]]
            ),
            Err(GiCandidateReferenceError::InvalidRadiance(0))
        );
    }

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
                direction_pdf: 1.0 / core::f32::consts::PI,
                selection_u01: 0.5,
            })
            .collect();
        (packed_scene_with_attributes(&packed), rays)
    }

    #[test]
    fn host_trace_preparation_generates_bounded_rays_and_rejects_duplicate_surface_ids() {
        use prism_render_architecture::ray_scene::{Bvh, Triangle};

        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance
            .request_adapter(&rhi::RequestAdapterOptions::default())
            .unwrap();
        let (device, queue) = instance
            .create_device(&adapter, &rhi::DeviceDescriptor::default())
            .unwrap();
        let device = Arc::new(device);
        let trace_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let bvh = Bvh::build(&[Triangle::new(
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            7,
        )]);
        let input = crate::LightingTraceSceneInput::new(
            Arc::new(GpuBvhBuffers::from_bvh(&bvh)),
            vec![42],
            2,
            0.001,
            10.0,
            Arc::from([crate::LightingTriangleRadiance {
                primitive_id: 7,
                rgb: [1.0, 1.0, 1.0],
            }]),
            1,
        );
        let surfaces = [crate::LightingSurfaceInput::default()];
        let mut state = LightingTraceState::default();
        let bad_radiance = crate::LightingTraceSceneInput::new(
            Arc::clone(&input.bvh),
            input.surface_ids.clone(),
            input.rays_per_surface,
            input.ray_epsilon,
            input.ray_tmax,
            Arc::from([]),
            input.radiance_revision,
        );
        assert!(state
            .prepare(
                Some(&bad_radiance),
                &surfaces,
                3,
                1,
                Arc::clone(&trace_device),
                &queue,
                None,
            )
            .is_none());
        assert_eq!(
            state.status(),
            TracePreparationStatus::Failed(TracePreparationError::RadianceCountMismatch {
                triangles: 1,
                radiance: 0,
            })
        );
        let recorder = state.prepare(
            Some(&input),
            &surfaces,
            3,
            1,
            Arc::clone(&trace_device),
            &queue,
            None,
        );
        assert!(recorder.is_some());
        assert_eq!(
            state.status(),
            TracePreparationStatus::Prepared {
                rays: 2,
                frame_index: 3
            }
        );
        let surface_id_capacity = state.surface_id_scratch.capacity();
        assert!(state
            .prepare(
                Some(&input),
                &surfaces,
                4,
                1,
                Arc::clone(&trace_device),
                &queue,
                None,
            )
            .is_some());
        assert_eq!(state.surface_id_scratch.capacity(), surface_id_capacity);

        let duplicate_ids = crate::LightingTraceSceneInput::new(
            Arc::clone(&input.bvh),
            vec![42, 42],
            input.rays_per_surface,
            input.ray_epsilon,
            input.ray_tmax,
            Arc::clone(&input.outgoing_radiance),
            input.radiance_revision,
        );
        let two_surfaces = [
            crate::LightingSurfaceInput::default(),
            crate::LightingSurfaceInput::default(),
        ];
        assert!(state
            .prepare(
                Some(&duplicate_ids),
                &two_surfaces,
                4,
                1,
                trace_device,
                &queue,
                None,
            )
            .is_none());
        assert_eq!(
            state.status(),
            TracePreparationStatus::Failed(TracePreparationError::DuplicateSurfaceId(42))
        );
    }

    #[test]
    fn stable_arc_bvh_uploads_once_replacement_reuploads_and_device_generation_rebuilds() {
        use prism_render_architecture::ray_scene::{Bvh, Triangle};

        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance
            .request_adapter(&rhi::RequestAdapterOptions::default())
            .unwrap();
        let (device, queue) = instance
            .create_device(&adapter, &rhi::DeviceDescriptor::default())
            .unwrap();
        let device = Arc::new(device);
        let trace_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let make_scene = |z: f32| {
            let bvh = Bvh::build(&[Triangle::new(
                [-1.0, -1.0, z],
                [1.0, -1.0, z],
                [0.0, 1.0, z],
                7,
            )]);
            crate::LightingTraceSceneInput::new(
                Arc::new(GpuBvhBuffers::from_bvh(&bvh)),
                vec![42],
                1,
                0.001,
                10.0,
                Arc::from([crate::LightingTriangleRadiance {
                    primitive_id: 7,
                    rgb: [1.0, 1.0, 1.0],
                }]),
                1,
            )
        };
        let surface = [crate::LightingSurfaceInput::default()];
        let first_scene = make_scene(-1.0);
        let mut state = LightingTraceState::default();

        for frame_index in 0..3 {
            assert!(state
                .prepare(
                    Some(&first_scene),
                    &surface,
                    frame_index,
                    1,
                    Arc::clone(&trace_device),
                    &queue,
                    None,
                )
                .is_some());
        }
        assert_eq!(
            state
                .gpu
                .as_ref()
                .unwrap()
                .buffer_capacity_stats()
                .scene_uploads,
            1,
            "the same retained Arc scene must not be uploaded again each frame"
        );

        let replacement_scene = make_scene(-2.0);
        assert!(state
            .prepare(
                Some(&replacement_scene),
                &surface,
                3,
                1,
                Arc::clone(&trace_device),
                &queue,
                None,
            )
            .is_some());
        assert_eq!(
            state
                .gpu
                .as_ref()
                .unwrap()
                .buffer_capacity_stats()
                .scene_uploads,
            2,
            "a different Arc with different BVH contents must be uploaded"
        );

        assert!(state
            .prepare(
                Some(&replacement_scene),
                &surface,
                4,
                2,
                trace_device,
                &queue,
                None,
            )
            .is_some());
        assert_eq!(
            state
                .gpu
                .as_ref()
                .unwrap()
                .buffer_capacity_stats()
                .scene_uploads,
            1,
            "device-generation replacement creates fresh buffers and uploads the scene once"
        );
    }

    #[test]
    fn device_generation_change_reclaims_trace_resources_before_input_validation() {
        use prism_render_architecture::ray_scene::{Bvh, Triangle};

        let make_device = || {
            let instance = prism_render_driver_null::NullInstance::new();
            let adapter = instance
                .request_adapter(&rhi::RequestAdapterOptions::default())
                .unwrap();
            instance
                .create_device(&adapter, &rhi::DeviceDescriptor::default())
                .unwrap()
        };

        let (old_device, old_queue) = make_device();
        let old_device = Arc::new(old_device);
        let old_trace_device: Arc<dyn RenderDevice + Send + Sync> = old_device.clone();
        let bvh = Bvh::build(&[Triangle::new(
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            7,
        )]);
        let input = crate::LightingTraceSceneInput::new(
            Arc::new(GpuBvhBuffers::from_bvh(&bvh)),
            vec![42],
            1,
            0.001,
            10.0,
            Arc::from([crate::LightingTriangleRadiance {
                primitive_id: 7,
                rgb: [1.0, 1.0, 1.0],
            }]),
            1,
        );
        let surfaces = [crate::LightingSurfaceInput::default()];
        let mut state = LightingTraceState::default();
        assert!(state
            .prepare(
                Some(&input),
                &surfaces,
                0,
                1,
                old_trace_device,
                &old_queue,
                None,
            )
            .is_some());
        assert!(old_device.live_resource_count() > 0);

        let (new_device, new_queue) = make_device();
        let new_device = Arc::new(new_device);
        let new_trace_device: Arc<dyn RenderDevice + Send + Sync> = new_device.clone();
        assert!(state
            .prepare(None, &[], 1, 2, new_trace_device, &new_queue, None)
            .is_none());
        assert_eq!(
            old_device.live_resource_count(),
            0,
            "device recovery must release the previous trace/candidate generation even when the current frame has no input"
        );
        assert_eq!(new_device.live_resource_count(), 0);
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
        let borrowed_scene = BorrowedPackedTraceScene::from(&scene);
        let required = (rays.len() * HIT_WORDS * 4) as u64;
        let output = device.create_buffer(&BufferDescriptor::new(
            required,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let mut trace = LightingTraceGpu::new(trace_device);
        assert!(matches!(
            trace.prepare_graph(&queue, &borrowed_scene, &rays, output, required - 1, None),
            Err(TraceGpuError::OutputBufferTooSmall)
        ));
        let recorder = trace
            .prepare_graph(&queue, &borrowed_scene, &rays, output, required, None)
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
        let attributes = graph.import_buffer(
            "trace-attributes",
            BufferDesc::new(recorder.attributes.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.attributes.id,
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
        graph.add_compute_pass("trace", PassFlags::SIDE_EFFECT, |builder| {
            builder.read_storage_buffer(nodes);
            builder.read_storage_buffer(triangles);
            builder.read_storage_buffer(attributes);
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
    fn hardware_candidate_build_matches_cpu_reference_for_hit_and_miss() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no physical adapter was found"
                );
                eprintln!("SKIPPED candidate GPU parity: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires physical GPU: {evidence}"
                );
                eprintln!("SKIPPED candidate GPU parity: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("hardware GPU device creation failed: {error}")
            }
        };
        let device = Arc::new(device);
        let trace_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let bvh = Bvh::build(&[Triangle::new(
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            19,
        )]);
        let scene = packed_scene_with_attributes(&GpuBvhBuffers::from_bvh(&bvh));
        let stable_id = 42_u64;
        let frame_index = 7_u64;
        let rays = [
            TraceRay {
                origin: [0.0, 0.0, 0.0],
                t_min: 0.001,
                direction: [0.0, 0.0, -1.0],
                t_max: 10.0,
                direction_pdf: 1.0 / core::f32::consts::PI,
                selection_u01: selection_hash(frame_index, stable_id, 0),
            },
            TraceRay {
                origin: [10.0, 0.0, 0.0],
                t_min: 0.001,
                direction: [0.0, 0.0, -1.0],
                t_max: 10.0,
                direction_pdf: 1.0 / core::f32::consts::PI,
                selection_u01: selection_hash(frame_index, stable_id, 1),
            },
        ];
        let surfaces = [LightingSurfaceInput {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            albedo: [0.8, 0.5, 0.25],
            ambient: [0.0; 3],
        }];
        let outgoing = [[4.0, 2.0, 1.0]];
        let cpu_hits = trace_reference(&scene, &rays).unwrap();
        let expected = build_gi_candidates_reference(
            &scene,
            &rays,
            &cpu_hits,
            &surfaces,
            &[stable_id],
            frame_index,
            &outgoing,
        )
        .unwrap();
        let make_buffer = |size, usage| device.create_buffer(&BufferDescriptor::new(size, usage));
        let hits = make_buffer(
            2 * HIT_WORDS as u64 * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let radiance = make_buffer(16, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let gpu_surfaces = make_buffer(80, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let params = make_buffer(32, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let candidates = make_buffer(
            2 * core::mem::size_of::<LightingGiCandidateInput>() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        queue.write_buffer(
            radiance,
            0,
            bytemuck::cast_slice(&[[4.0_f32, 2.0, 1.0, 0.0]]),
        );
        let packed_surface = [
            [0.0_f32, 0.0, 0.0, 0.0],
            [0.0_f32, 0.0, -1.0, 0.0],
            [0.8_f32, 0.5, 0.25, 0.0],
            [0.0_f32; 4],
        ];
        let mut surface_words = bytemuck::cast_slice::<[f32; 4], u32>(&packed_surface).to_vec();
        surface_words.extend([0, 2, stable_id as u32, (stable_id >> 32) as u32]);
        queue.write_buffer(gpu_surfaces, 0, bytemuck::cast_slice(&surface_words));
        let params_words = [
            1_u32,
            0,
            0,
            2,
            0,
            frame_index as u32,
            (frame_index >> 32) as u32,
            1,
        ];
        queue.write_buffer(params, 0, bytemuck::cast_slice(&params_words));
        let target = LightingGiCandidateTarget {
            candidates,
            candidates_size_bytes: 2 * core::mem::size_of::<LightingGiCandidateInput>() as u64,
            surface_count: 1,
            rays_per_surface: 2,
            surfaces: gpu_surfaces,
            surfaces_size_bytes: 80,
            params,
            params_size_bytes: 32,
            radiance,
            radiance_size_bytes: 16,
        };
        let mut pass = LightingTraceGpu::new(trace_device);
        let undersized_target = LightingGiCandidateTarget {
            candidates_size_bytes: 0,
            ..target
        };
        assert!(matches!(
            pass.prepare_graph(
                &queue,
                &BorrowedPackedTraceScene::from(&scene),
                &rays,
                hits,
                2 * HIT_WORDS as u64 * 4,
                Some(undersized_target),
            ),
            Err(TraceGpuError::CandidateBufferTooSmall)
        ));
        let recorder = pass
            .prepare_graph(
                &queue,
                &BorrowedPackedTraceScene::from(&scene),
                &rays,
                hits,
                2 * HIT_WORDS as u64 * 4,
                Some(target),
            )
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
        let attributes = graph.import_buffer(
            "trace-attributes",
            BufferDesc::new(recorder.attributes.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.attributes.id,
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
        let mut hit_buffer = graph.import_buffer(
            "trace-hits",
            BufferDesc::new(recorder.hits.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.hits.id,
                entry_state: BufferState::initial(),
            },
        );
        let mut candidate_buffer = graph.import_buffer(
            "gi-candidates",
            BufferDesc::new(
                target.candidates_size_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            ImportedBuffer {
                buffer: candidates,
                entry_state: BufferState::initial(),
            },
        );
        let radiance_buffer = graph.import_buffer(
            "gi-radiance",
            BufferDesc::new(16, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: radiance,
                entry_state: BufferState::initial(),
            },
        );
        let surface_buffer = graph.import_buffer(
            "gi-surfaces",
            BufferDesc::new(80, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: gpu_surfaces,
                entry_state: BufferState::initial(),
            },
        );
        let params_buffer = graph.import_buffer(
            "gi-params",
            BufferDesc::new(32, BufferUsages::UNIFORM),
            ImportedBuffer {
                buffer: params,
                entry_state: BufferState::initial(),
            },
        );
        let trace_recorder = recorder;
        let trace_pass = graph
            .try_add_pass_with_dependencies(
                "trace",
                PassKind::Compute,
                PassFlags::SIDE_EFFECT,
                Vec::new(),
                |builder| {
                    builder.read_storage_buffer(nodes);
                    builder.read_storage_buffer(triangles);
                    builder.read_storage_buffer(attributes);
                    builder.read_storage_buffer(ray_buffer);
                    hit_buffer = builder.write_storage_buffer(hit_buffer);
                    move |context| trace_recorder.record_graph(context)
                },
            )
            .expect("trace pass fits the parity graph");
        let trace_hit_output = hit_buffer;
        graph
            .try_add_pass_with_dependencies(
                "candidate-build",
                PassKind::Compute,
                PassFlags::SIDE_EFFECT,
                vec![trace_pass],
                |builder| {
                    builder.read_storage_buffer(ray_buffer);
                    builder.read_storage_buffer(trace_hit_output);
                    builder.read_storage_buffer(triangles);
                    builder.read_storage_buffer(attributes);
                    builder.read_storage_buffer(radiance_buffer);
                    builder.read_storage_buffer(surface_buffer);
                    builder.read_uniform(params_buffer);
                    candidate_buffer = builder.write_storage_buffer(candidate_buffer);
                    move |context| recorder.record_candidate_graph(context)
                },
            )
            .expect("candidate pass fits the parity graph");
        let recorded = graph.execute(device.as_ref(), &queue).unwrap();
        let hit_bytes = queue.read_buffer(hits, 2 * HIT_WORDS as u64 * 4);
        let hit_words: &[u32] = bytemuck::try_cast_slice(&hit_bytes).unwrap();
        assert_eq!(hit_words[0], 1, "first candidate ray must hit");
        assert_eq!(hit_words[HIT_WORDS], 0, "second candidate ray must miss");
        let bytes = queue.read_buffer(candidates, target.candidates_size_bytes);
        let actual: &[LightingGiCandidateInput] = bytemuck::try_cast_slice(&bytes).unwrap();
        for (candidate_index, (actual, expected)) in actual.iter().zip(&expected[0]).enumerate() {
            let float_fields = actual
                .position
                .into_iter()
                .chain([actual.source_pdf])
                .chain(actual.normal)
                .chain([actual.target_pdf])
                .chain(actual.radiance)
                .chain([actual.selection_u01, actual.visibility]);
            let expected_fields = expected
                .position
                .into_iter()
                .chain([expected.source_pdf])
                .chain(expected.normal)
                .chain([expected.target_pdf])
                .chain(expected.radiance)
                .chain([expected.selection_u01, expected.visibility]);
            for (actual, expected) in float_fields.zip(expected_fields) {
                assert!(
                    (actual - expected).abs() <= 2.0e-5,
                    "candidate {candidate_index}: {actual} != {expected}; gpu={:?}, cpu={:?}",
                    actual,
                    expected
                );
            }
        }
        recorded.release(device.as_ref());
        drop(recorder);
        drop(pass);
        for buffer in [hits, radiance, gpu_surfaces, params, candidates] {
            device.destroy_buffer(buffer);
        }
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
        let borrowed_scene = BorrowedPackedTraceScene::from(&scene);
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
                &borrowed_scene,
                &rays,
                output,
                (rays.len() * HIT_WORDS * 4) as u64,
                None,
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
        let attributes = graph.import_buffer(
            "trace-attributes",
            BufferDesc::new(recorder.attributes.size_bytes, BufferUsages::STORAGE),
            ImportedBuffer {
                buffer: recorder.attributes.id,
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
                builder.read_storage_buffer(attributes);
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
            assert_eq!(words[b + 5], hit.triangle_index, "ray {i} triangle index");
            assert_eq!(words[b + 6], hit.instance_id, "ray {i} instance identity");
            assert_eq!(words[b + 7], hit.geometry_id, "ray {i} geometry identity");
            assert_eq!(
                words[b + 8],
                hit.material_generation,
                "ray {i} material generation"
            );
            for (offset, expected_value) in hit.uv.into_iter().enumerate() {
                assert!(
                    (f32::from_bits(words[b + 9 + offset]) - expected_value).abs() < 1e-5,
                    "ray {i} uv word {offset}"
                );
            }
            for (offset, expected_value) in hit.shading_normal.into_iter().enumerate() {
                assert!(
                    (f32::from_bits(words[b + 11 + offset]) - expected_value).abs() < 1e-5,
                    "ray {i} shading normal word {offset}"
                );
            }
        }
        recorded.release(device.as_ref());
        drop(recorder);
        drop(pass);
        device.destroy_buffer(output);
    }
}
