//! Direct line-of-sight arrival and transmission kernel.
//!
//! One `GPU` invocation per query marches the listener-to-emitter segment
//! through the scene, folds each crossed partition's transmission gain, and
//! writes the direct (or transmitted) arrival plus the occlusion it implies.
//! The dispatch binds the shared [`GeometryParams`](crate::params::GeometryParams)
//! uniform, the uploaded [`GpuScene`] triangles, the batch of
//! [`GpuQuery`](crate::query::GpuQuery) records, and a zero-initialised result
//! buffer that decodes an untouched slot as "no arrival".
//!
//! [`cpu_direct`] is the host twin: a line-for-line port of
//! [`shaders/direct.wgsl`](../src/shaders/direct.wgsl) that the parity tests
//! compare the device output against, and that is in turn checked against the
//! `CPU` [`resolve_direct`](prism_audio_geometry::direct_path::resolve_direct).
//!
//! # Provenance
//!
//! Original work; standard `wgpu` compute mirroring the classic ray-march
//! already implemented on the `CPU` in [`prism_audio_geometry`]; no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance
//! Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Driven by [`crate::backend`], which uploads the scene once and decodes the
//! results into [`prism_audio_spatial`] propagation paths.

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
use crate::query::{GpuDirectResult, GpuQuery, DIRECT_KIND_DIRECT, DIRECT_KIND_TRANSMISSION};
use crate::scene_upload::{GpuScene, GpuTriangle};
use crate::trace::{first_hit, quat_conj, quat_rotate, vec3, COINCIDENT};

/// Full-band low-pass corner (Hz); the direct stage never filters, matching
/// `FULL_BAND_CUTOFF_HZ` in [`prism_audio_spatial`].
const FULL_BAND: f32 = 1.0e6;

/// Maximum partitions one march folds before giving up, matching
/// `MAX_MARCH_HITS` in the shader and the `CPU` `march_segment`.
const MAX_MARCH_HITS: u32 = 64;

/// Root-mean-square of three per-band gains, matching `BandGains::broadband_rms`
/// and the shader's `broadband_rms3`: the single broadband amplitude that
/// preserves energy when the three-band spectrum collapses to one number.
#[must_use]
fn broadband_rms3(v: [f32; 3]) -> f32 {
    sqrt((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 3.0)
}

/// Factors three per-band gains into the peak scalar and the normalised colour,
/// mirroring `BandGains::split_peak` and the shader's inline peak/colour split.
///
/// A spectrum whose peak is zero has no colour to recover and factors into a
/// zero scalar and a silent colour, exactly as the spatial crate does.
#[must_use]
fn split_peak3(v: [f32; 3]) -> (f32, [f32; 3]) {
    let peak = v[0].max(v[1]).max(v[2]);
    if peak <= 0.0 {
        (0.0, [0.0, 0.0, 0.0])
    } else {
        (peak, [v[0] / peak, v[1] / peak, v[2] / peak])
    }
}

/// Compiled direct-path compute pipeline and its bind-group layout.
pub(crate) struct DirectKernel {
    /// The compiled `resolve_direct` pipeline.
    pipeline: ComputePipeline,
    /// The bind-group layout the dispatch binds against.
    layout: BindGroupLayout,
}

impl DirectKernel {
    /// Compiles `shaders/direct.wgsl` into a ready-to-dispatch pipeline.
    #[must_use]
    pub(crate) fn new(ctx: &GpuContext) -> DirectKernel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_acoustics_direct_shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/direct.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_acoustics_direct_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_acoustics_direct_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_acoustics_direct_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve_direct"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        DirectKernel { pipeline, layout }
    }

    /// Resolves the direct arrival for every query in `queries`.
    ///
    /// Returns one [`GpuDirectResult`] per query in input order. An empty batch
    /// returns an empty vector without a dispatch, since storage buffers cannot
    /// be zero-sized.
    #[must_use]
    pub(crate) fn dispatch(
        &self,
        ctx: &GpuContext,
        scene: &GpuScene,
        params: &GeometryParams,
        queries: &[GpuQuery],
    ) -> Vec<GpuDirectResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let result_bytes = (queries.len() * size_of::<GpuDirectResult>()) as u64;

        let params_buf = buffer::uniform(device, "prism_acoustics_direct_params", params);
        let query_buf = buffer::storage_read(device, "prism_acoustics_direct_queries", queries);
        let result_buf =
            buffer::storage_rw_zeroed(device, "prism_acoustics_direct_results", result_bytes);
        let staging = buffer::staging(device, "prism_acoustics_direct_staging", result_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_acoustics_direct_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, scene.buffer()),
                entry(2, &query_buf),
                entry(3, &result_buf),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_acoustics_direct_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_acoustics_direct_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &result_buf, &staging, result_bytes);
        ctx.queue().submit([encoder.finish()]);
        buffer::read_back::<GpuDirectResult>(ctx, &staging)
    }
}

/// Host twin of the direct kernel: resolves one query against `triangles`.
///
/// A line-for-line port of `resolve_direct` in `shaders/direct.wgsl`, used both
/// to validate the device output and to drive the `CPU` fallback path in
/// [`crate::backend`].
#[must_use]
pub(crate) fn cpu_direct(
    triangles: &[GpuTriangle],
    query: &GpuQuery,
    params: &GeometryParams,
) -> GpuDirectResult {
    let lp = vec3(query.listener_pos);
    let ep = vec3(query.emitter_pos);
    let orient = query.listener_orient;

    let to_source = ep - lp;
    let dist = sqrt(to_source.dot(to_source));
    let mut direction = Vec3::new(0.0, 0.0, -1.0);
    let mut distance = 0.0;
    if dist > COINCIDENT {
        direction = quat_rotate(quat_conj(orient), to_source / dist);
        distance = dist;
    }
    let delay = distance / params.speed_of_sound;

    let eps = params.surface_epsilon.max(0.0);
    let mut transmitted = [1.0_f32, 1.0, 1.0];
    let mut crossings = 0u32;
    if dist > 0.0 {
        let dir = to_source / dist;
        let mut cursor = lp;
        let mut remaining = dist;
        for _ in 0..MAX_MARCH_HITS {
            if remaining <= eps {
                break;
            }
            let hit = first_hit(triangles, cursor, dir, remaining);
            if !hit.valid {
                break;
            }
            let band = triangles[hit.index].transmission;
            for k in 0..3 {
                transmitted[k] *= band[k];
            }
            crossings += 1;
            if !matches!(
                broadband_rms3(transmitted).partial_cmp(&params.min_gain),
                Some(core::cmp::Ordering::Greater)
            ) {
                break;
            }
            let step = hit.t + eps;
            cursor += dir * step;
            remaining -= step;
        }
    }

    let mut out = GpuDirectResult {
        direction: [direction.x, direction.y, direction.z, 0.0],
        delay_seconds: delay,
        gain: 0.0,
        cutoff_hz: FULL_BAND,
        base_distance: distance,
        obstruction: 0.0,
        occlusion: 0.0,
        kind: DIRECT_KIND_DIRECT,
        audible: 0,
        bands: [0.0, 0.0, 0.0, 0.0],
    };
    if crossings == 0 {
        out.gain = 1.0;
        out.bands = [1.0, 1.0, 1.0, 0.0];
        out.obstruction = 0.0;
        out.occlusion = 0.0;
        out.kind = DIRECT_KIND_DIRECT;
        out.audible = 1;
    } else {
        let survived = broadband_rms3(transmitted);
        let blocked = (1.0 - survived).clamp(0.0, 1.0);
        out.obstruction = blocked;
        out.occlusion = blocked;
        out.kind = DIRECT_KIND_TRANSMISSION;
        let (peak, colour) = split_peak3(transmitted);
        out.bands = [colour[0], colour[1], colour[2], 0.0];
        let audible = params.transmission_enabled != 0 && survived > params.min_gain;
        if audible {
            out.gain = peak;
            out.audible = 1;
        } else {
            out.gain = 0.0;
            out.audible = 0;
        }
    }
    out
}
