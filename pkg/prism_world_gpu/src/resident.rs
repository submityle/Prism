//! Fixed-capacity proxy visibility and one GPU-generated instanced draw.

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{
    buffer_binding, storage_entry, uniform_entry, GpuParams, GpuRecord, WORKGROUP_SIZE,
};
use crate::{GpuContext, GpuVisibility, ProxyInstance, ProxySoa, VisibilityInput};

/// Validation failures detected before allocating resources or uploading columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuResidentError {
    /// A host count, range, or byte-size calculation overflowed the GPU ABI.
    ArithmeticOverflow,
    /// A requested range extends beyond the fixed allocation.
    CapacityExceeded { end: usize, capacity: usize },
    /// All three public SoA columns must have the same length.
    ColumnLengthMismatch,
    /// A row contains invalid proxy values or non-finite padding.
    InvalidProxy { index: usize },
    /// A dense prefix would include slots never successfully uploaded.
    UninitializedDenseRange {
        count: usize,
        initialized_prefix: usize,
    },
    /// Camera values must be finite; projection and range must be non-negative.
    InvalidCamera,
    /// Upload camera/count with `write_camera` before encoding this frame.
    CameraNotUploaded,
    /// The allocation or dispatch cannot fit the logical device's limits.
    DeviceLimit {
        limit: &'static str,
        required: u64,
        available: u64,
    },
}

impl std::fmt::Display for GpuResidentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArithmeticOverflow => f.write_str("resident proxy range or byte size overflow"),
            Self::CapacityExceeded { end, capacity } => {
                write!(f, "proxy range ends at {end}, beyond capacity {capacity}")
            }
            Self::ColumnLengthMismatch => f.write_str("proxy SoA column lengths differ"),
            Self::InvalidProxy { index } => write!(f, "invalid proxy SoA row {index}"),
            Self::UninitializedDenseRange {
                count,
                initialized_prefix,
            } => {
                write!(
                    f,
                    "dense count {count} exceeds uploaded prefix {initialized_prefix}"
                )
            }
            Self::InvalidCamera => f.write_str("invalid visibility camera parameters"),
            Self::CameraNotUploaded => {
                f.write_str("camera or dense count changed; call write_camera before encode")
            }
            Self::DeviceLimit {
                limit,
                required,
                available,
            } => {
                write!(
                    f,
                    "{limit}: requires {required}, device supports {available}"
                )
            }
        }
    }
}

impl std::error::Error for GpuResidentError {}

/// Reusable GPU resources for a caller-maintained dense proxy prefix.
///
/// Resident buffers, bind groups, and pipelines are created only in `new`.
/// Upload staging is managed by wgpu's queue. Uploads never change the active
/// count: call `set_dense_count` explicitly, then `write_camera` before encoding.
/// Compact indices are
/// source *slots*, not proxy IDs, and only the first indirect `instanceCount`
/// entries are valid. Their order is intentionally unspecified.
///
/// Queue ordering contract: submit an encoded frame before another `encode`
/// or any upload on this object. Like ordinary `Queue::write_buffer`, camera and
/// column uploads execute before the next submission, not at their position in
/// an encoder. Sequential submissions need no host wait, mapping, or readback.
pub struct GpuResidentVisibility {
    queue: wgpu::Queue,
    capacity: usize,
    count: u32,
    initialized: Vec<bool>,
    initialized_prefix: usize,
    positions: wgpu::Buffer,
    errors: wgpu::Buffer,
    metadata: wgpu::Buffer,
    params: wgpu::Buffer,
    uploaded_params: Option<GpuParams>,
    records: wgpu::Buffer,
    indices: wgpu::Buffer,
    indirect: wgpu::Buffer,
    visibility: GpuVisibility,
    visibility_bind_group: wgpu::BindGroup,
    compact_pipeline: wgpu::ComputePipeline,
    compact_bind_group: wgpu::BindGroup,
}

impl GpuResidentVisibility {
    /// Allocates capacity once and initializes `[vertex_count, 0, 0, 0]`.
    ///
    /// Zero capacity is supported using one unaddressed physical slot. Limits
    /// are checked before any GPU allocation; physical allocation/device loss
    /// errors remain subject to the caller's normal wgpu error handling.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        vertex_count: u32,
    ) -> Result<Self, GpuResidentError> {
        validate_capacity(capacity, &ctx.device().limits())?;
        let device = ctx.device();
        let slots = capacity.max(1) as u64;
        let storage = |label, size, extra| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let positions = storage(
            "resident proxy positions",
            slots * 16,
            wgpu::BufferUsages::COPY_DST,
        );
        let errors = storage(
            "resident proxy errors",
            slots * 16,
            wgpu::BufferUsages::COPY_DST,
        );
        let metadata = storage(
            "resident proxy metadata",
            slots * 16,
            wgpu::BufferUsages::COPY_DST,
        );
        let records = storage(
            "resident visibility records",
            slots * size_of::<GpuRecord>() as u64,
            wgpu::BufferUsages::COPY_SRC,
        );
        let indices = storage(
            "resident compact indices",
            slots * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("resident non-indexed indirect draw"),
            contents: bytemuck::cast_slice(&[vertex_count, 0, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident visibility params"),
            size: size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visibility = GpuVisibility::new(ctx);
        let visibility_bind_group =
            visibility.bind_group(device, &params, &positions, &errors, &metadata, &records);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident visibility compact shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/proxy_compact.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident visibility compact layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resident visibility compact pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resident visibility compact pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let compact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident visibility compact bind group"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, &records),
                buffer_binding(2, &indices),
                buffer_binding(3, &indirect),
            ],
        });
        Ok(Self {
            queue: ctx.queue().clone(),
            capacity,
            count: 0,
            initialized: vec![false; capacity],
            initialized_prefix: 0,
            positions,
            errors,
            metadata,
            params,
            uploaded_params: None,
            records,
            indices,
            indirect,
            visibility,
            visibility_bind_group,
            compact_pipeline,
            compact_bind_group,
        })
    }

    /// Validates the entire public SoA and destination before queuing any write.
    ///
    /// An error leaves all columns, initialization tracking, and count unchanged.
    /// Successful writes affect exactly `offset..offset + soa.len()`; count is
    /// unchanged. Empty writes at capacity are allowed.
    pub fn write_range(&mut self, offset: usize, soa: &ProxySoa) -> Result<(), GpuResidentError> {
        let end = validate_write(self.capacity, offset, soa)?;
        if soa.is_empty() {
            return Ok(());
        }
        let byte_offset = (offset as u64) * 16;
        self.queue.write_buffer(
            &self.positions,
            byte_offset,
            bytemuck::cast_slice(&soa.positions),
        );
        self.queue
            .write_buffer(&self.errors, byte_offset, bytemuck::cast_slice(&soa.errors));
        self.queue.write_buffer(
            &self.metadata,
            byte_offset,
            bytemuck::cast_slice(&soa.metadata),
        );
        self.initialized[offset..end].fill(true);
        while self.initialized_prefix < self.capacity && self.initialized[self.initialized_prefix] {
            self.initialized_prefix += 1;
        }
        Ok(())
    }

    /// Selects the initialized prefix `0..count`, rejecting holes and overflow.
    /// Shrinking does not discard uploaded data; growing can reuse those slots.
    pub fn set_dense_count(&mut self, count: usize) -> Result<(), GpuResidentError> {
        checked_end(self.capacity, 0, count)?;
        if count > self.initialized_prefix {
            return Err(GpuResidentError::UninitializedDenseRange {
                count,
                initialized_prefix: self.initialized_prefix,
            });
        }
        self.count = count as u32;
        Ok(())
    }

    /// Validates and queues the shared 128-byte camera/count uniform. Call after
    /// selecting the dense count, before `encode`. An unchanged uniform reuses
    /// its previous upload. Queue staging is managed by wgpu; this method does
    /// not recreate resident buffers or submit commands.
    pub fn write_camera(&mut self, camera: VisibilityInput) -> Result<(), GpuResidentError> {
        validate_camera(camera)?;
        let params = GpuParams::new(camera, self.count);
        if !self.camera_is_uploaded(&params) {
            self.queue
                .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
            self.uploaded_params = Some(params);
        }
        Ok(())
    }

    /// Only records reset + visibility + compact into the caller's encoder.
    /// Never uploads, submits, polls, maps, reads back, or creates GPU resources.
    /// For a nonempty frame, `write_camera(camera)` must have uploaded this
    /// camera and dense count. An error leaves the encoder unchanged.
    ///
    /// Even for zero proxies the indirect instance count is cleared in encoder
    /// order. Only that word is reset; the caller's vertex count stays fixed.
    pub fn encode(
        &self,
        camera: VisibilityInput,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuResidentError> {
        validate_camera(camera)?;
        if self.count != 0 && !self.camera_is_uploaded(&GpuParams::new(camera, self.count)) {
            return Err(GpuResidentError::CameraNotUploaded);
        }
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        self.visibility
            .encode(encoder, &self.visibility_bind_group, self.count);
        if self.count != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident visibility compact pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, &self.compact_bind_group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        Ok(())
    }

    fn camera_is_uploaded(&self, params: &GpuParams) -> bool {
        self.uploaded_params
            .as_ref()
            .is_some_and(|uploaded| bytemuck::bytes_of(uploaded) == bytemuck::bytes_of(params))
    }

    /// Fixed logical slot capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Active dense prefix length; independent of the last write's extent.
    #[must_use]
    pub fn dense_count(&self) -> u32 {
        self.count
    }

    /// Sum of resident buffer sizes in bytes, excluding driver allocations,
    /// pipelines, queue staging, and CPU initialization tracking. This is a
    /// logical buffer budget, not a measurement of physical VRAM residency.
    #[must_use]
    pub fn allocated_buffer_bytes(&self) -> u64 {
        [
            &self.positions,
            &self.errors,
            &self.metadata,
            &self.params,
            &self.records,
            &self.indices,
            &self.indirect,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    /// Read-only downstream storage: packed u32 source slots, in arbitrary order.
    /// Only the prefix of length indirect `instanceCount` is valid.
    #[must_use]
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.indices
    }

    /// Four u32 words `[vertex_count, visible_count, 0, 0]`, usable directly by
    /// `RenderPass::draw_indirect(buffer, 0)` after `encode` in the same encoder.
    #[must_use]
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }

    /// Per-slot visibility ABI, shared with the parity path: 20-byte records of
    /// `[visible:u32, lod:u32, error_quantized:u32, proxy_id:u32, screen_error:f32]`.
    /// Only `0..dense_count()` records belong to the current frame.
    #[must_use]
    pub fn visibility_records(&self) -> &wgpu::Buffer {
        &self.records
    }
}

fn require_limit(
    limit: &'static str,
    required: u64,
    available: u64,
) -> Result<(), GpuResidentError> {
    if required > available {
        Err(GpuResidentError::DeviceLimit {
            limit,
            required,
            available,
        })
    } else {
        Ok(())
    }
}

fn validate_capacity(capacity: usize, limits: &wgpu::Limits) -> Result<(), GpuResidentError> {
    let count = u32::try_from(capacity).map_err(|_| GpuResidentError::ArithmeticOverflow)?;
    let slots = u64::from(count.max(1));
    let largest = slots
        .checked_mul(size_of::<GpuRecord>() as u64)
        .ok_or(GpuResidentError::ArithmeticOverflow)?;
    require_limit(
        "max_buffer_size",
        largest.max(size_of::<GpuParams>() as u64),
        limits.max_buffer_size,
    )?;
    require_limit(
        "max_storage_buffer_binding_size",
        largest,
        limits.max_storage_buffer_binding_size,
    )?;
    require_limit(
        "max_uniform_buffer_binding_size",
        size_of::<GpuParams>() as u64,
        limits.max_uniform_buffer_binding_size,
    )?;
    require_limit(
        "max_storage_buffers_per_shader_stage",
        4,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )?;
    require_limit(
        "max_uniform_buffers_per_shader_stage",
        1,
        u64::from(limits.max_uniform_buffers_per_shader_stage),
    )?;
    require_limit(
        "max_bindings_per_bind_group",
        5,
        u64::from(limits.max_bindings_per_bind_group),
    )?;
    require_limit("max_bind_groups", 1, u64::from(limits.max_bind_groups))?;
    require_limit(
        "max_compute_workgroup_size_x",
        u64::from(WORKGROUP_SIZE),
        u64::from(limits.max_compute_workgroup_size_x),
    )?;
    require_limit(
        "max_compute_workgroup_size_y",
        1,
        u64::from(limits.max_compute_workgroup_size_y),
    )?;
    require_limit(
        "max_compute_workgroup_size_z",
        1,
        u64::from(limits.max_compute_workgroup_size_z),
    )?;
    require_limit(
        "max_compute_invocations_per_workgroup",
        u64::from(WORKGROUP_SIZE),
        u64::from(limits.max_compute_invocations_per_workgroup),
    )?;
    require_limit(
        "max_compute_workgroup_storage_size",
        8,
        u64::from(limits.max_compute_workgroup_storage_size),
    )?;
    require_limit(
        "max_compute_workgroups_per_dimension",
        u64::from(count.div_ceil(WORKGROUP_SIZE).max(1)),
        u64::from(limits.max_compute_workgroups_per_dimension),
    )
}

fn checked_end(capacity: usize, offset: usize, count: usize) -> Result<usize, GpuResidentError> {
    let end = offset
        .checked_add(count)
        .ok_or(GpuResidentError::ArithmeticOverflow)?;
    if end > capacity {
        return Err(GpuResidentError::CapacityExceeded { end, capacity });
    }
    Ok(end)
}

fn validate_write(
    capacity: usize,
    offset: usize,
    soa: &ProxySoa,
) -> Result<usize, GpuResidentError> {
    if soa.positions.len() != soa.errors.len() || soa.positions.len() != soa.metadata.len() {
        return Err(GpuResidentError::ColumnLengthMismatch);
    }
    let end = checked_end(capacity, offset, soa.len())?;
    for (index, ((position, error), metadata)) in soa
        .positions
        .iter()
        .zip(&soa.errors)
        .zip(&soa.metadata)
        .enumerate()
    {
        let proxy = ProxyInstance {
            proxy_id: metadata[0],
            cluster_id: metadata[1],
            flags: metadata[2],
            position: [position[0], position[1], position[2]],
            radius: position[3],
            geometric_error: error[0],
            max_distance: error[1],
        };
        if !proxy.is_finite() || !error.iter().all(|v| v.is_finite()) {
            return Err(GpuResidentError::InvalidProxy { index });
        }
    }
    Ok(end)
}

fn validate_camera(camera: VisibilityInput) -> Result<(), GpuResidentError> {
    if !camera.camera_position.iter().all(|v| v.is_finite())
        || !camera.projection_scale.is_finite()
        || camera.projection_scale < 0.0
        || !camera.max_distance.is_finite()
        || camera.max_distance < 0.0
        || camera
            .frustum
            .iter()
            .any(|p| !p.distance.is_finite() || !p.normal.iter().all(|v| v.is_finite()))
    {
        return Err(GpuResidentError::InvalidCamera);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_device_limits_fail_before_allocation_or_upload() {
        assert_eq!(
            checked_end(129, usize::MAX, 1),
            Err(GpuResidentError::ArithmeticOverflow)
        );
        assert_eq!(checked_end(129, 129, 0), Ok(129));
        assert!(matches!(
            checked_end(129, 128, 2),
            Err(GpuResidentError::CapacityExceeded { .. })
        ));
        let limits = wgpu::Limits::downlevel_defaults();
        assert!(validate_capacity(0, &limits).is_ok());
        assert!(validate_capacity(129, &limits).is_ok());
        assert_eq!(
            validate_capacity(usize::MAX, &limits),
            Err(GpuResidentError::ArithmeticOverflow)
        );
        let mut small = limits.clone();
        small.max_storage_buffer_binding_size = 128 * size_of::<GpuRecord>() as u64;
        assert!(matches!(
            validate_capacity(129, &small),
            Err(GpuResidentError::DeviceLimit {
                limit: "max_storage_buffer_binding_size",
                ..
            })
        ));
        small = limits.clone();
        small.max_buffer_size = 128;
        assert!(matches!(
            validate_capacity(129, &small),
            Err(GpuResidentError::DeviceLimit {
                limit: "max_buffer_size",
                ..
            })
        ));
        small = limits;
        small.max_compute_workgroups_per_dimension = 2;
        assert!(matches!(
            validate_capacity(129, &small),
            Err(GpuResidentError::DeviceLimit {
                limit: "max_compute_workgroups_per_dimension",
                ..
            })
        ));
    }

    #[test]
    fn public_soa_and_camera_values_are_checked_without_a_device() {
        let mut soa = ProxySoa::from_proxies(&[
            ProxyInstance::new(1, 0, [0.0; 3], 1.0, 1.0, 10.0),
            ProxyInstance::new(2, 0, [0.0; 3], 1.0, 1.0, 10.0),
        ])
        .unwrap();
        assert_eq!(validate_write(129, 127, &soa), Ok(129));
        soa.errors[1][1] = -1.0;
        assert_eq!(
            validate_write(129, 0, &soa),
            Err(GpuResidentError::InvalidProxy { index: 1 })
        );
        soa.errors[1][1] = 10.0;
        soa.positions[1][0] = f32::INFINITY;
        assert_eq!(
            validate_write(129, 0, &soa),
            Err(GpuResidentError::InvalidProxy { index: 1 })
        );
        soa.metadata.pop();
        assert_eq!(
            validate_write(129, 0, &soa),
            Err(GpuResidentError::ColumnLengthMismatch)
        );
        let input = VisibilityInput::from_projection_scale(
            [0.0; 3],
            [crate::FrustumPlane::new([1.0, 0.0, 0.0], 10.0); 6],
            1.0,
            10.0,
        );
        assert_eq!(validate_camera(input), Ok(()));
        let mut invalid = input;
        invalid.frustum[5].distance = f32::NAN;
        assert_eq!(
            validate_camera(invalid),
            Err(GpuResidentError::InvalidCamera)
        );
        invalid = input;
        invalid.max_distance = -1.0;
        assert_eq!(
            validate_camera(invalid),
            Err(GpuResidentError::InvalidCamera)
        );
    }
}
