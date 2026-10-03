//! First-order specular reflection kernel (image-source method).
//!
//! One `GPU` invocation per (query, triangle) pair mirrors the emitter across
//! the candidate face, validates the bounce (same side of the plane, on-face,
//! both legs clear of geometry, above the gain floor), and writes a reflection
//! candidate. The dispatch is a flat 1D grid over `query_count * triangle_count`
//! invocations, so slot `q * triangle_count + t` holds query `q`'s candidate off
//! triangle `t`. Rejected slots keep their zero bytes, which decode as
//! `valid == 0`; the host ([`crate::backend`]) merges, de-duplicates, sorts
//! loudest-first, and caps the survivors.
//!
//! [`cpu_reflection`] is the host twin: a line-for-line port of
//! [`shaders/reflection.wgsl`](../src/shaders/reflection.wgsl) that the parity
//! tests compare the device output against, and that is in turn checked against
//! the `CPU`
//! [`resolve_reflections`](prism_audio_geometry::reflection_path::resolve_reflections).
//!
//! # Provenance
//!
//! Original work; standard `wgpu` compute mirroring the classic image-source
//! construction already implemented on the `CPU` in [`prism_audio_geometry`]; no
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google
//! Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Driven by [`crate::backend`], which uploads the scene once and decodes the
//! surviving candidates into [`prism_audio_spatial`] reflection paths.

use alloc::vec::Vec;

use bevy_math::ops::sqrt;
use bevy_math::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::layout::{buffer_entry, entry};
use crate::params::GeometryParams;
use crate::query::{GpuQuery, GpuReflectionCandidate};
use crate::scene_upload::{GpuScene, GpuTriangle};
use crate::trace::{first_hit, quat_conj, quat_rotate, vec3, COINCIDENT};

/// Smallest denominator the plane-intersection and barycentric solves accept
/// before treating the configuration as degenerate; mirrors `F32_EPSILON` in
/// the shader (`f32::EPSILON`).
const F32_EPSILON: f32 = 1.192_092_9e-7;

/// Barycentric containment slack, mirroring `BARY_TOL` in the shader. A small
/// positive tolerance keeps a reflection point that lands exactly on a shared
/// edge from being rejected by rounding.
const BARY_TOL: f32 = 1.0e-4;

/// Compiled reflection compute pipeline and its bind-group layout.
pub(crate) struct ReflectionKernel {
    /// The compiled `resolve_reflection` pipeline.
    pipeline: ComputePipeline,
    /// The bind-group layout the dispatch binds against.
    layout: BindGroupLayout,
}

impl ReflectionKernel {
    /// Compiles `shaders/reflection.wgsl` into a ready-to-dispatch pipeline.
    #[must_use]
    pub(crate) fn new(ctx: &GpuContext) -> ReflectionKernel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_acoustics_reflection_shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/reflection.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_acoustics_reflection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_acoustics_reflection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_acoustics_reflection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve_reflection"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        ReflectionKernel { pipeline, layout }
    }

    /// Resolves one reflection candidate per (query, triangle) pair.
    ///
    /// Returns a flat vector of `queries.len() * params.triangle_count`
    /// candidates in row-major (query-major) order; slot `q * triangle_count + t`
    /// is query `q`'s candidate off triangle `t`. An empty batch, or a scene with
    /// no triangles, returns an empty vector without a dispatch, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub(crate) fn dispatch(
        &self,
        ctx: &GpuContext,
        scene: &GpuScene,
        params: &GeometryParams,
        queries: &[GpuQuery],
    ) -> Vec<GpuReflectionCandidate> {
        let triangle_count = params.triangle_count as usize;
        let total = queries.len() * triangle_count;
        if total == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let result_bytes = (total * size_of::<GpuReflectionCandidate>()) as u64;

        let params_buf = buffer::uniform(device, "prism_acoustics_reflection_params", params);
        let query_buf = buffer::storage_read(device, "prism_acoustics_reflection_queries", queries);
        let result_buf =
            buffer::storage_rw_zeroed(device, "prism_acoustics_reflection_results", result_bytes);
        let staging = buffer::staging(device, "prism_acoustics_reflection_staging", result_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_acoustics_reflection_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, scene.buffer()),
                entry(2, &query_buf),
                entry(3, &result_buf),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_acoustics_reflection_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_acoustics_reflection_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (total as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &result_buf, &staging, result_bytes);
        ctx.queue().submit([encoder.finish()]);
        buffer::read_back::<GpuReflectionCandidate>(ctx, &staging)
    }
}

/// Shrinks the segment by `eps` at both ends and tests whether any triangle
/// blocks the interior; mirrors the shader's `segment_blocked` and the `CPU`
/// `AcousticScene::segment_blocked`.
///
/// Pulling both endpoints inward by `eps` keeps a surface an endpoint already
/// lies on (the reflecting face itself, at the bounce point) from counting as a
/// blocker.
#[must_use]
fn segment_blocked(triangles: &[GpuTriangle], from: Vec3, to: Vec3, eps: f32) -> bool {
    let delta = to - from;
    let len = sqrt(delta.dot(delta));
    let e = eps.max(0.0);
    if len <= 2.0 * e {
        return false;
    }
    let dir = delta / len;
    let origin = from + dir * e;
    first_hit(triangles, origin, dir, len - 2.0 * e).valid
}

/// Coplanar barycentric containment with a small positive slack, mirroring the
/// shader's `point_in_triangle` and the `CPU` equivalent.
#[must_use]
fn point_in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;
    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);
    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() <= F32_EPSILON {
        return false;
    }
    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;
    u >= -BARY_TOL && v >= -BARY_TOL && (u + v) <= 1.0 + BARY_TOL
}

/// Host twin of the reflection kernel: resolves one (query, triangle) candidate.
///
/// A line-for-line port of `resolve_reflection` in `shaders/reflection.wgsl`,
/// used both to validate the device output and to drive the `CPU` fallback path
/// in [`crate::backend`]. `tri_index` selects the candidate reflecting face in
/// `triangles`.
#[must_use]
pub(crate) fn cpu_reflection(
    triangles: &[GpuTriangle],
    query: &GpuQuery,
    tri_index: usize,
    params: &GeometryParams,
) -> GpuReflectionCandidate {
    let mut out = GpuReflectionCandidate {
        direction: [0.0, 0.0, 0.0, 0.0],
        delay_seconds: 0.0,
        gain: 0.0,
        valid: 0,
        _pad: 0.0,
        bands: [0.0, 0.0, 0.0, 0.0],
    };

    let tri = triangles[tri_index];
    let normal = vec3(tri.normal);
    if normal.dot(normal) <= 0.0 {
        return out;
    }
    let a = vec3(tri.a);
    let b = vec3(tri.b);
    let c = vec3(tri.c);

    let lp = vec3(query.listener_pos);
    let ep = vec3(query.emitter_pos);
    let orient = query.listener_orient;

    let d_listener = (lp - a).dot(normal);
    let d_source = (ep - a).dot(normal);
    if d_listener * d_source <= 0.0 {
        return out;
    }

    let image = ep - normal * (2.0 * d_source);
    let direction_v = image - lp;
    let denom = direction_v.dot(normal);
    if denom.abs() <= F32_EPSILON {
        return out;
    }
    let tt = -d_listener / denom;
    if !(tt > 0.0 && tt < 1.0) {
        return out;
    }
    let point = lp + direction_v * tt;
    if !point_in_triangle(point, a, b, c) {
        return out;
    }

    let eps = params.surface_epsilon.max(0.0);
    if segment_blocked(triangles, lp, point, eps) || segment_blocked(triangles, point, ep, eps) {
        return out;
    }

    let leg_in = point - lp;
    let leg_out = ep - point;
    let path_length = sqrt(leg_in.dot(leg_in)) + sqrt(leg_out.dot(leg_out));
    if path_length <= 0.0 {
        return out;
    }
    let base_vec = ep - lp;
    let base = sqrt(base_vec.dot(base_vec));
    let spreading = (base / path_length).clamp(0.0, 1.0);
    // Specular share weighted by sqrt(1 - scattering) band by band, then scaled
    // by the spreading factor; mirrors the CPU
    // `specular_reflection().scaled(spreading)` and the shader's inline split.
    let spec_weight = sqrt((1.0 - tri.scattering).max(0.0));
    let mut effective = [0.0_f32; 3];
    for (slot, &reflection) in effective.iter_mut().zip(tri.reflection.iter()) {
        let specular = (reflection * spec_weight).clamp(0.0, 1.0);
        *slot = (specular * spreading).clamp(0.0, 1.0);
    }
    let peak = effective[0].max(effective[1]).max(effective[2]);
    if peak <= params.min_gain {
        return out;
    }
    let colour = if peak > 0.0 {
        [effective[0] / peak, effective[1] / peak, effective[2] / peak]
    } else {
        [0.0, 0.0, 0.0]
    };
    let gain = peak;

    let to_point = point - lp;
    let d = sqrt(to_point.dot(to_point));
    let mut ldir = Vec3::new(0.0, 0.0, -1.0);
    if d > COINCIDENT {
        ldir = quat_rotate(quat_conj(orient), to_point / d);
    }

    out.direction = [ldir.x, ldir.y, ldir.z, 0.0];
    out.delay_seconds = path_length / params.speed_of_sound;
    out.gain = gain;
    out.valid = 1;
    out._pad = 0.0;
    out.bands = [colour[0], colour[1], colour[2], 0.0];
    out
}
