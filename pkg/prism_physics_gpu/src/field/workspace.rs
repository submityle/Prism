//! Capacity-bounded, reusable native field upload/dispatch/readback workspace.

use std::{
    future::Future,
    pin::pin,
    sync::mpsc,
    task::{Context, Poll, Waker},
    time::Duration,
};

use bytemuck::Zeroable;
use glam::Vec3;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

use super::gpu::{Params, WORKGROUP};
use super::packed::{PackedBody, PackedField, PackedSample};
use super::{FieldBody, FieldBufferStats, FieldGpuError, FieldRegistry, FieldSample};
use crate::{buffer, context::GpuContext};

#[derive(Default)]
pub(super) struct Workspace {
    buffers: Option<Buffers>,
    fields: Vec<PackedField>,
    bodies: Vec<PackedBody>,
    allocations: u64,
}

struct Buffers {
    params: Buffer,
    fields: Buffer,
    bodies: Buffer,
    output: Buffer,
    staging: Buffer,
    bind_group: wgpu::BindGroup,
    body_capacity: usize,
    field_capacity: usize,
}

/// Native wgpu 30 pops scopes synchronously (its futures are `ready`). Poll
/// exactly once instead of introducing an unbounded executor wait. A different
/// backend must report unresolved error status, never pretend it succeeded.
fn resolve_scope(
    operation: &'static str,
    future: impl Future<Output = Option<wgpu::Error>>,
) -> Result<Option<wgpu::Error>, FieldGpuError> {
    match pin!(future).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(error) => Ok(error),
        Poll::Pending => Err(FieldGpuError::ErrorScopePending { operation }),
    }
}

fn scoped_error(operation: &'static str, error: wgpu::Error) -> FieldGpuError {
    match error {
        wgpu::Error::OutOfMemory { source } => FieldGpuError::OutOfMemory {
            operation,
            // Unlike Error::Display's generic "Out of Memory", this retains
            // the context and lower-level source of an allocation failure.
            message: format!("{source:?}"),
        },
        wgpu::Error::Validation { description, .. } => FieldGpuError::Validation {
            operation,
            message: description,
        },
        wgpu::Error::Internal { description, .. } => FieldGpuError::Internal {
            operation,
            message: description,
        },
    }
}

/// Capture native GPU errors on this thread without an unbounded scope wait.
/// Shared with pipeline construction; callers supply the diagnostic operation.
pub(super) fn scoped_gpu<T>(
    device: &Device,
    operation: &'static str,
    op: impl FnOnce() -> Result<T, FieldGpuError>,
) -> Result<T, FieldGpuError> {
    // wgpu 30 guards are thread-local: keep the operation and all pops on this
    // thread, and pop EVERY scope in reverse order even when `op` returns Err.
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = op();
    let validation = resolve_scope(operation, validation.pop());
    let internal = resolve_scope(operation, internal.pop());
    let oom = resolve_scope(operation, oom.pop());
    // Allocation failure can also cause cascading invalid-resource errors.
    // Prefer the root OOM/internal error over the resulting validation error.
    let mut unresolved = None;
    for outcome in [oom, internal, validation] {
        match outcome {
            Ok(Some(error)) => return Err(scoped_error(operation, error)),
            Ok(None) => {}
            Err(error) => unresolved = Some(error),
        }
    }
    if let Some(error) = unresolved {
        return Err(error);
    }
    result
}

/// Validate before packing/allocating. Synthetic limits exercise boundary
/// failures without allocating enormous host arrays or requiring a GPU.
fn validate_limits(
    limits: &wgpu::Limits,
    bodies: usize,
    fields: usize,
) -> Result<Params, FieldGpuError> {
    let body_count =
        u32::try_from(bodies).map_err(|_| FieldGpuError::BodyCountTooLarge { count: bodies })?;
    let field_count =
        u32::try_from(fields).map_err(|_| FieldGpuError::FieldCountTooLarge { count: fields })?;
    let workgroups = body_count.div_ceil(WORKGROUP);
    if workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(FieldGpuError::DispatchLimit {
            requested: workgroups,
            limit: limits.max_compute_workgroups_per_dimension,
        });
    }
    let limit = storage_limit(limits);
    for (buffer, count, stride) in [
        ("fields", fields.max(1), size_of::<PackedField>()),
        ("bodies", bodies, size_of::<PackedBody>()),
        ("samples", bodies, size_of::<PackedSample>()),
    ] {
        let requested = (count as u64)
            .checked_mul(stride as u64)
            .unwrap_or(u64::MAX);
        if requested > limit {
            return Err(FieldGpuError::BufferLimit {
                buffer,
                requested,
                limit,
            });
        }
    }
    Ok(Params {
        body_count,
        field_count,
        _pad: [0; 2],
    })
}

fn storage_limit(limits: &wgpu::Limits) -> u64 {
    limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size))
}

fn grown_capacity(required: usize, previous: usize, maximum: usize) -> usize {
    required
        .max(previous)
        .checked_next_power_of_two()
        .unwrap_or(maximum)
        .min(maximum)
}

impl Buffers {
    fn new(
        device: &Device,
        layout: &wgpu::BindGroupLayout,
        bodies: usize,
        fields: usize,
    ) -> Result<Self, FieldGpuError> {
        scoped_gpu(device, "allocation", || {
            let create = |label, size, usage| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let params = create(
                "prism_field_params",
                size_of::<Params>() as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            );
            let fields_buf = create(
                "prism_field_fields",
                (fields * size_of::<PackedField>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            let bodies_buf = create(
                "prism_field_bodies",
                (bodies * size_of::<PackedBody>()) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            let output = buffer::storage_rw_zeroed(
                device,
                "prism_field_output",
                (bodies * size_of::<PackedSample>()) as u64,
            );
            let staging = buffer::staging(
                device,
                "prism_field_stage",
                (bodies * size_of::<PackedSample>()) as u64,
            );
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism_field_bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: fields_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bodies_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            Ok(Self {
                params,
                fields: fields_buf,
                bodies: bodies_buf,
                output,
                staging,
                bind_group,
                body_capacity: bodies,
                field_capacity: fields,
            })
        })
    }
}

impl Workspace {
    pub(super) fn stats(&self) -> FieldBufferStats {
        FieldBufferStats {
            body_capacity: self.buffers.as_ref().map_or(0, |b| b.body_capacity),
            field_capacity: self.buffers.as_ref().map_or(0, |b| b.field_capacity),
            allocations: self.allocations,
        }
    }

    pub(super) fn sample(
        &mut self,
        ctx: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        registry: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, FieldGpuError> {
        if bodies.is_empty() {
            return Ok(Vec::new());
        }
        let limits = ctx.device().limits();
        let params = validate_limits(&limits, bodies.len(), registry.fields().len())?;
        let stats = self.stats();
        if bodies.len() > stats.body_capacity
            || registry.fields().len().max(1) > stats.field_capacity
        {
            let limit = storage_limit(&limits) as usize;
            let body_capacity = grown_capacity(
                bodies.len(),
                stats.body_capacity,
                limit / size_of::<PackedBody>().max(size_of::<PackedSample>()),
            );
            let field_capacity = grown_capacity(
                registry.fields().len().max(1),
                stats.field_capacity,
                limit / size_of::<PackedField>(),
            );
            let replacement = Buffers::new(ctx.device(), layout, body_capacity, field_capacity)?;
            // Commit only a fully validated allocation; keep the previous set
            // and its diagnostics available when growth fails.
            self.buffers = Some(replacement);
            self.allocations += 1;
        }
        self.fields.clear();
        self.fields
            .extend(registry.fields().iter().map(PackedField::from));
        if self.fields.is_empty() {
            self.fields.push(PackedField::zeroed());
        }
        self.bodies.clear();
        self.bodies.extend(bodies.iter().map(PackedBody::from));
        let buffers = self
            .buffers
            .as_ref()
            .expect("nonempty validated batch has capacity");
        let queue = ctx.queue();
        scoped_gpu(ctx.device(), "upload", || {
            queue.write_buffer(&buffers.params, 0, bytemuck::bytes_of(&params));
            queue.write_buffer(&buffers.fields, 0, bytemuck::cast_slice(&self.fields));
            queue.write_buffer(&buffers.bodies, 0, bytemuck::cast_slice(&self.bodies));
            Ok(())
        })?;
        let output_bytes = (bodies.len() * size_of::<PackedSample>()) as u64;
        let commands = scoped_gpu(ctx.device(), "encoding", || {
            let mut encoder =
                ctx.device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("prism_field_encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prism_field_sample"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &buffers.bind_group, &[]);
                pass.dispatch_workgroups(params.body_count.div_ceil(WORKGROUP), 1, 1);
            }
            buffer::copy(
                &mut encoder,
                &buffers.output,
                &buffers.staging,
                output_bytes,
            );
            Ok(encoder.finish())
        })?;
        // A failed submit still returns an index in wgpu. Check its scope
        // before mapping or waiting on an index that may not represent work.
        let submission = scoped_gpu(ctx.device(), "submission", || Ok(queue.submit([commands])))?;
        scoped_gpu(ctx.device(), "readback", || {
            let slice = buffers.staging.slice(..output_bytes);
            let (sender, receiver) = mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let result = (|| {
                ctx.device()
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission),
                        timeout: Some(Duration::from_secs(30)),
                    })
                    .map_err(|error| FieldGpuError::Poll(error.to_string()))?;
                receiver
                    .try_recv()
                    .map_err(|error| error.to_string())
                    .and_then(|result| result.map_err(|error| error.to_string()))
                    .map_err(FieldGpuError::Mapping)?;
                match slice.get_mapped_range() {
                    Ok(view) => Ok(bytemuck::cast_slice::<u8, PackedSample>(&view)
                        .iter()
                        .map(|sample| FieldSample {
                            force: Vec3::new(sample.force[0], sample.force[1], sample.force[2]),
                            torque: Vec3::new(sample.torque[0], sample.torque[1], sample.torque[2]),
                        })
                        .collect()),
                    Err(error) => Err(FieldGpuError::Mapping(error.to_string())),
                }
            })();
            // Also cancel a pending map after timeout/error. Unmap while the
            // readback scopes are still active, and after any mapped view drops.
            buffers.staging.unmap();
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<GpuContext> {
        let ctx = GpuContext::try_headless();
        if ctx.is_none() {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU is set but no real GPU context is available"
            );
            eprintln!("skipping field workspace GPU regression: no GPU context available");
        }
        ctx
    }

    fn pipeline(device: &Device, source: &str) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_workspace_regression"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("field_workspace_regression"),
            layout: None,
            module: &module,
            entry_point: Some("sample_fields"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline.get_bind_group_layout(0), pipeline)
    }

    #[test]
    fn classifies_scoped_errors_and_rejects_pending_without_waiting() {
        // Do not exhaust a real device to test OOM/internal classification.
        let oom = scoped_error(
            "allocation",
            wgpu::Error::OutOfMemory {
                source: Box::new(std::io::Error::other("synthetic allocation failure")),
            },
        );
        assert!(matches!(
            oom,
            FieldGpuError::OutOfMemory { operation: "allocation", message }
                if message.contains("synthetic allocation failure")
        ));
        assert_eq!(
            scoped_error(
                "submission",
                wgpu::Error::Validation {
                    source: Box::new(std::io::Error::other("synthetic source")),
                    description: "original validation diagnostic".into(),
                }
            ),
            FieldGpuError::Validation {
                operation: "submission",
                message: "original validation diagnostic".into(),
            }
        );
        assert_eq!(
            scoped_error(
                "upload",
                wgpu::Error::Internal {
                    source: Box::new(std::io::Error::other("synthetic source")),
                    description: "original internal diagnostic".into(),
                }
            ),
            FieldGpuError::Internal {
                operation: "upload",
                message: "original internal diagnostic".into(),
            }
        );
        assert_eq!(
            resolve_scope("allocation", std::future::pending()).err(),
            Some(FieldGpuError::ErrorScopePending {
                operation: "allocation",
            })
        );
        assert!(matches!(
            resolve_scope("allocation", std::future::ready(None)),
            Ok(None)
        ));
    }

    #[test]
    fn real_gpu_scopes_capture_allocation_upload_and_submission_errors() {
        let Some(ctx) = gpu() else { return };
        let device = ctx.device();
        // No inner errors may escape into the caller's scope. Its final pop
        // also checks that every nested scope was popped, even on early Err.
        let outer = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let invalid = scoped_gpu(device, "allocation", || {
            Ok(device.create_buffer(&BufferDescriptor {
                label: Some("field_deliberately_invalid_usage"),
                size: 4,
                // MAP_READ cannot be combined with STORAGE.
                usage: BufferUsages::MAP_READ | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        });
        assert!(matches!(
            invalid,
            Err(FieldGpuError::Validation {
                operation: "allocation",
                ..
            })
        ));
        let source = device.create_buffer(&BufferDescriptor {
            label: Some("field_submission_source"),
            size: 4,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dest = buffer::staging(device, "field_submission_destination", 4);
        let upload = scoped_gpu(device, "upload", || {
            ctx.queue().write_buffer(&source, 4, &[0; 4]);
            Ok(())
        });
        assert!(matches!(
            upload,
            Err(FieldGpuError::Validation {
                operation: "upload",
                ..
            })
        ));
        let commands = scoped_gpu(device, "encoding", || {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&source, 0, &dest, 0, 4);
            Ok(encoder.finish())
        })
        .expect("valid encoding before destroying its source");
        source.destroy();
        let submission = scoped_gpu(device, "submission", || Ok(ctx.queue().submit([commands])));
        assert!(matches!(
            submission,
            Err(FieldGpuError::Validation {
                operation: "submission",
                ..
            })
        ));
        let early: Result<(), _> = scoped_gpu(device, "readback", || {
            Err(FieldGpuError::Mapping("synthetic early return".into()))
        });
        assert_eq!(
            early,
            Err(FieldGpuError::Mapping("synthetic early return".into()))
        );
        // A rejected submission must not prevent a subsequent valid one.
        let submission = scoped_gpu(device, "submission", || Ok(ctx.queue().submit([])))
            .expect("valid submission after scoped failures");
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("bounded poll of valid submission");
        assert!(matches!(resolve_scope("outer", outer.pop()), Ok(None)));
    }

    #[test]
    fn real_gpu_failed_growth_and_encoding_preserve_workspace_reuse() {
        let Some(ctx) = gpu() else { return };
        let device = ctx.device();
        let (layout, pipeline) = pipeline(device, include_str!("field.wgsl"));
        let wrong_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field_deliberately_incompatible_layout"),
            entries: &[],
        });
        let body = FieldBody {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            mass: 1.0,
            target_mask: super::super::FieldTargetMask::ALL,
        };
        let registry = FieldRegistry::new();
        let mut workspace = Workspace::default();
        assert!(matches!(
            workspace.sample(&ctx, &wrong_layout, &pipeline, &registry, &[body]),
            Err(FieldGpuError::Validation {
                operation: "allocation",
                ..
            })
        ));
        assert_eq!(workspace.stats(), FieldBufferStats::default());
        assert!(workspace.buffers.is_none());
        let expected = vec![FieldSample::default()];
        assert_eq!(
            workspace
                .sample(&ctx, &layout, &pipeline, &registry, &[body])
                .unwrap(),
            expected
        );
        let stats = workspace.stats();
        let retained = workspace.buffers.as_ref().unwrap().bodies.clone();
        assert!(matches!(
            workspace.sample(&ctx, &wrong_layout, &pipeline, &registry, &[body; 3]),
            Err(FieldGpuError::Validation {
                operation: "allocation",
                ..
            })
        ));
        assert_eq!(workspace.stats(), stats);
        assert_eq!(workspace.buffers.as_ref().unwrap().bodies, retained);
        assert_eq!(
            workspace
                .sample(&ctx, &layout, &pipeline, &registry, &[body])
                .unwrap(),
            expected
        );
        assert_eq!(workspace.stats(), stats);
        // Deliberately incompatible pipeline fails during encoding rather
        // than returning old samples or submitting an invalid command buffer.
        let (_, incompatible) = self::pipeline(
            device,
            "@group(0) @binding(0) var<storage, read_write> wrong: array<u32>;
             @compute @workgroup_size(1) fn sample_fields() { wrong[0] = 1u; }",
        );
        assert!(matches!(
            workspace.sample(&ctx, &layout, &incompatible, &registry, &[body]),
            Err(FieldGpuError::Validation {
                operation: "encoding",
                ..
            })
        ));
        assert_eq!(workspace.stats(), stats);
        assert_eq!(
            workspace
                .sample(&ctx, &layout, &pipeline, &registry, &[body])
                .unwrap(),
            expected
        );
        assert_eq!(workspace.buffers.as_ref().unwrap().bodies, retained);
        assert_eq!(workspace.stats(), stats);
        assert_eq!(
            workspace
                .sample(&ctx, &layout, &pipeline, &registry, &[body; 3])
                .unwrap(),
            vec![FieldSample::default(); 3]
        );
        assert_eq!(workspace.stats().allocations, stats.allocations + 1);
        assert!(workspace.stats().body_capacity >= 3);
    }

    #[test]
    fn validates_exact_storage_boundaries_and_growth() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 96 * 3,
            max_buffer_size: 1 << 20,
            ..Default::default()
        };
        assert!(validate_limits(&limits, 6, 3).is_ok());
        assert_eq!(
            validate_limits(&limits, 7, 3).err(),
            Some(FieldGpuError::BufferLimit {
                buffer: "bodies",
                requested: 336,
                limit: 288
            })
        );
        assert_eq!(
            validate_limits(&limits, 1, 4).err(),
            Some(FieldGpuError::BufferLimit {
                buffer: "fields",
                requested: 384,
                limit: 288
            })
        );
        let limits = wgpu::Limits {
            max_buffer_size: 96,
            ..limits
        };
        assert_eq!(
            validate_limits(&limits, 3, 0).err(),
            Some(FieldGpuError::BufferLimit {
                buffer: "bodies",
                requested: 144,
                limit: 96
            })
        );
        assert!(validate_limits(&limits, 2, 0).is_ok());
        assert_eq!(grown_capacity(5, 0, 6), 6);
        assert_eq!(grown_capacity(1, 6, 6), 6);
        assert_eq!(grown_capacity(5, 0, 16), 8);
    }

    #[test]
    fn validates_dispatch_and_index_overflow_without_allocating() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..Default::default()
        };
        assert!(validate_limits(&limits, 64, 1).is_ok());
        assert_eq!(
            validate_limits(&limits, 65, 1).err(),
            Some(FieldGpuError::DispatchLimit {
                requested: 2,
                limit: 1
            })
        );
        if usize::BITS > 32 {
            let huge = u32::MAX as usize + 1;
            assert_eq!(
                validate_limits(&limits, huge, 1).err(),
                Some(FieldGpuError::BodyCountTooLarge { count: huge })
            );
            assert_eq!(
                validate_limits(&limits, 1, huge).err(),
                Some(FieldGpuError::FieldCountTooLarge { count: huge })
            );
        }
    }
}
