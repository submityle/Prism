//! Real-device `wgpu` sampler for the M11 analytic field registry.
//!
//! One compute invocation owns one [`FieldBody`](super::FieldBody) and walks
//! the registry in its stable `FieldId` order. The host only packs immutable
//! parameters and reads the completed samples back: shape membership,
//! attenuation, direction calculation, force construction, and accumulation
//! all execute in `field.wgsl`.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

use super::workspace::{scoped_gpu, Workspace};
use super::{FieldBody, FieldRegistry, FieldSample};

/// Number of sample bodies processed by one compute workgroup.
pub(super) const WORKGROUP: u32 = 64;

/// Pre-allocation rejections, scoped native GPU errors, and readback failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldGpuError {
    /// The body slice does not fit the kernel's `u32` indexing contract.
    BodyCountTooLarge {
        /// Requested number of bodies.
        count: usize,
    },
    /// The registry does not fit the kernel's `u32` indexing contract.
    FieldCountTooLarge {
        /// Requested number of fields.
        count: usize,
    },
    /// Requested storage exceeds a binding or allocation limit.
    BufferLimit {
        buffer: &'static str,
        requested: u64,
        limit: u64,
    },
    /// The one-dimensional dispatch exceeds the device limit.
    DispatchLimit { requested: u32, limit: u32 },
    /// Submission used a different device from the compiled pipeline.
    DeviceMismatch,
    /// A previous panicking caller poisoned the serialized workspace.
    WorkspacePoisoned,
    /// A workspace operation exhausted GPU memory (not host allocator failure).
    OutOfMemory {
        /// Workspace operation that failed.
        operation: &'static str,
        /// Lower-level wgpu diagnostic, including its source.
        message: String,
    },
    /// wgpu rejected a workspace GPU operation.
    Validation {
        /// Workspace operation that failed.
        operation: &'static str,
        /// Original wgpu validation diagnostic.
        message: String,
    },
    /// wgpu reported an internal error during a workspace operation.
    Internal {
        /// Workspace operation that failed.
        operation: &'static str,
        /// Original wgpu internal-error diagnostic.
        message: String,
    },
    /// A popped scope did not resolve immediately as the native wgpu 30 backend
    /// does. The operation is not assumed successful and no unbounded wait is used.
    ErrorScopePending {
        /// Workspace operation whose GPU error status could not be established.
        operation: &'static str,
    },
    /// Polling failed or exceeded the bounded readback wait.
    Poll(String),
    /// Mapping failed, including unavailable mapped bytes or callback.
    Mapping(String),
}

/// Retained GPU capacity and allocation diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldBufferStats {
    /// Body/sample records retained without replacing GPU buffers.
    pub body_capacity: usize,
    /// Field records retained without replacing GPU buffers.
    pub field_capacity: usize,
    /// Buffer-set allocations, not sample calls.
    pub allocations: u64,
}

/// Uniform parameters shared with `Params` in `field.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Params {
    pub body_count: u32,
    pub field_count: u32,
    pub _pad: [u32; 2],
}

/// A compiled reusable sampler for core's analytic [`FieldRegistry`].
pub struct GpuFieldSampler {
    device_identity: Arc<()>,
    workspace: Mutex<Workspace>,
    #[expect(
        dead_code,
        reason = "kept alive so the compute pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    sample: ComputePipeline,
}

impl GpuFieldSampler {
    /// Compiles the analytic field kernel on `ctx`.
    ///
    /// Returns scoped shader, layout, pipeline validation and GPU allocation
    /// failures without installing an uncaptured-error handler or CPU fallback.
    pub fn new(ctx: &GpuContext) -> Result<Self, FieldGpuError> {
        Self::compile(ctx, include_str!("field.wgsl"))
    }

    fn compile(ctx: &GpuContext, source: &str) -> Result<Self, FieldGpuError> {
        let device = ctx.device();
        scoped_gpu(device, "pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("prism_field_sample"),
                source: ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("prism_field_sample_layout"),
                entries: &[
                    buffer_entry(0, BufferBindingType::Uniform),
                    buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                    buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                    buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("prism_field_sample_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let sample = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("prism_field_sample_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("sample_fields"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });
            Ok(Self {
                device_identity: Arc::clone(ctx.device_identity()),
                workspace: Mutex::new(Workspace::default()),
                module,
                layout,
                sample,
            })
        })
    }

    /// Inspect retained capacity without synchronizing the device.
    pub fn buffer_stats(&self) -> Result<FieldBufferStats, FieldGpuError> {
        Ok(self
            .workspace
            .lock()
            .map_err(|_| FieldGpuError::WorkspacePoisoned)?
            .stats())
    }

    /// Samples every field for every `body` on the GPU.
    ///
    /// The returned samples retain input body order. Empty body batches need no
    /// device allocation; nonempty batches dispatch even for an empty registry,
    /// proving zero-field output is also written by the compute kernel.
    ///
    /// # Errors
    ///
    /// Rejects device/count/buffer/dispatch mismatches before allocation.
    /// Workspace allocation/upload/encoding/submission errors are captured by
    /// native wgpu error scopes. Failed growth preserves the previous buffers.
    /// Mapping and bounded native device polling failures are returned, not
    /// converted to a CPU fallback. The workspace is serialized and reused.
    /// Pipeline construction has its own fallible boundary in [`Self::new`].
    /// Host allocation failure and arbitrary backend/application panics are
    /// not converted into GPU errors.
    pub fn sample(
        &self,
        ctx: &GpuContext,
        registry: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, FieldGpuError> {
        // wgpu backend IDs may collide across independently created instances.
        // Compare retained host ownership, not the backend resource handles.
        if !Arc::ptr_eq(ctx.device_identity(), &self.device_identity) {
            return Err(FieldGpuError::DeviceMismatch);
        }
        self.workspace
            .lock()
            .map_err(|_| FieldGpuError::WorkspacePoisoned)?
            .sample(ctx, &self.layout, &self.sample, registry, bodies)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_gpu_pipeline_errors_are_returned_and_do_not_poison_device() {
        let Some(context) = GpuContext::try_headless() else {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU is set but no real GPU context is available"
            );
            eprintln!("skipping field pipeline regression: no GPU context available");
            return;
        };
        let result = GpuFieldSampler::compile(&context, "invalid WGSL source");
        assert!(matches!(
            result,
            Err(FieldGpuError::Validation { operation: "pipeline", message })
                if !message.is_empty()
        ));
        let sampler =
            GpuFieldSampler::new(&context).expect("valid pipeline after a rejected shader");
        let body = FieldBody {
            position: glam::Vec3::ZERO,
            velocity: glam::Vec3::ZERO,
            mass: 1.0,
            target_mask: super::super::FieldTargetMask::ALL,
        };
        assert_eq!(
            sampler
                .sample(&context, &FieldRegistry::new(), &[body])
                .unwrap(),
            vec![FieldSample::default()]
        );
    }
}
