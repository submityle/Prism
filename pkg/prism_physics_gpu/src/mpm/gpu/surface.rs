//! Bounded, deterministic GPU particle-density isosurface publication.
//!
//! A fixed authored sampling domain avoids CPU particle/bounds readback. The
//! domain clips the field (it does not automatically follow particles). Six
//! Freudenthal tetrahedra per cell are classified, prefix-scanned using the
//! existing GPU scan, and emitted in cell/tetrahedron/table order. Output is a
//! non-indexed triangle list; shared edge vertices are deliberately duplicated.
//! No geometry, counts or error flags are mapped by the publication path.
//!
//! Each lease owns fresh buffers which this producer never writes again.
//! Consumers must retain a lease through GPU completion, not just recording,
//! and must not explicitly destroy or write its borrowed native buffers.
//! A renderer must use the exact same host context/queue and import the
//! buffers as external resources, not as renderer-owned allocations.

use std::sync::{Arc, Weak};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{BindGroupLayout, Buffer, BufferBindingType, BufferUsages, ComputePipeline};

use super::layout::{buffer_entry, entry};
use crate::{buffer, context::GpuContext, scan::GpuScan};

/// ABI version of the vertex, indirect and status layouts below.
pub const RESIDENT_SURFACE_ABI: u32 = 1;

/// Interleaved Float32x3 position at offset 0 and normal at offset 16; stride 32.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SurfaceVertex {
    /// World position; padding is zero.
    pub position: [f32; 4],
    /// Unit outward normal; padding is zero.
    pub normal: [f32; 4],
}

/// Exact 16-byte non-indexed draw-indirect ABI. One instance, first vertex 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SurfaceDrawIndirect {
    /// Number of triangle-list vertices; zero for an empty or rejected surface.
    pub vertex_count: u32,
    /// Number of instances, always one for a submitted publication.
    pub instance_count: u32,
    /// First vertex in the publication buffer, always zero.
    pub first_vertex: u32,
    /// First instance, always zero.
    pub first_instance: u32,
}

/// GPU status word. A nonzero word suppresses the entire indirect draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SurfaceStatus {
    /// Bit 0: invalid particle/field; bit 1: output capacity exceeded.
    pub flags: u32,
}

/// Authored bounded sampling domain and work/allocation ceilings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentSurfaceConfig {
    /// World-space position of sampling node `[0, 0, 0]`.
    pub origin: Vec3,
    /// Node counts, at least two on each axis. Cells = product(nodes - 1).
    pub nodes: [u32; 3],
    /// Finite, strictly positive spacing between adjacent sampling nodes.
    /// Domains unresolved at float32 precision are rejected before publication.
    pub dx: f32,
    /// Finite, strictly positive support radius of each particle's density kernel.
    pub radius: f32,
    /// Strictly positive density threshold. Density kernel is (1-q*q)^3.
    pub iso: f32,
    /// Upper bound on the product of the three node counts.
    pub max_field_nodes: u32,
    /// Upper bound on `(nodes[0] - 1) * (nodes[1] - 1) * (nodes[2] - 1)`.
    pub max_cells: u32,
    /// All node/particle pairs are visited, including zero-weight pairs.
    pub max_sample_visits: u64,
    /// Triangle-list vertices, not unique vertices; must be a multiple of 3.
    pub max_vertices: u32,
    /// Maximum outstanding leases for this producer, including consumer pins.
    pub max_live_publications: usize,
}

/// Rejection before publication allocation or submission. GPU data-dependent
/// failures instead set SurfaceStatus and produce a zero-vertex indirect draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentSurfaceError {
    /// Domain, kernel, threshold or publication capacity parameters are invalid.
    InvalidConfig,
    /// Authored work ceilings or representable node, cell or scan totals are exceeded.
    BudgetExceeded,
    /// Required buffers, bindings or compute dispatches exceed native device limits.
    DeviceLimit,
    /// Outstanding publication leases reached the configured limit; drop leases and retry.
    InFlightLimit,
    /// The supplied context does not share the resident solver's host identity.
    WrongDevice,
    /// The runtime is not using a GPU backend capable of native publication.
    BackendUnavailable,
    /// The producer's monotonically increasing publication sequence cannot advance.
    SequenceExhausted,
}

#[derive(Clone, Copy)]
struct Plan {
    nodes: u32,
    cells: u32,
    vertex_bytes: u64,
}

impl ResidentSurfaceConfig {
    fn plan(self, particles: usize) -> Result<Plan, ResidentSurfaceError> {
        use ResidentSurfaceError::{BudgetExceeded, InvalidConfig};
        let upper = self.origin
            + Vec3::new(
                self.nodes[0].saturating_sub(1) as f32,
                self.nodes[1].saturating_sub(1) as f32,
                self.nodes[2].saturating_sub(1) as f32,
            ) * self.dx;
        if !self.origin.is_finite()
            || !upper.is_finite()
            || !(self.origin - Vec3::splat(self.dx)).is_finite()
            || !(upper + Vec3::splat(self.dx)).is_finite()
            || self.nodes.iter().any(|&n| n < 2 || n > (1 << 24))
            || !self.dx.is_finite()
            || self.dx <= 0.0
            || !self.dx.recip().is_finite()
            || !self.radius.is_finite()
            || self.radius <= 0.0
            || !self.radius.recip().is_finite()
            || !self.iso.is_finite()
            || self.iso <= 0.0
            || self.max_vertices == 0
            || self.max_vertices % 3 != 0
            || self.max_live_publications == 0
            || (self.origin + Vec3::splat(self.dx))
                .cmpeq(self.origin)
                .any()
            || (upper - Vec3::splat(self.dx)).cmpeq(upper).any()
        {
            return Err(InvalidConfig);
        }
        // Endpoint additions alone do not detect collapsed interior nodes:
        // origin=2^24, dx=1.5, nodes=4 passes them, but nodes 2 and 3 coincide.
        // Reserve rounding headroom for both i*dx + origin and the half-cell
        // normal samples. Include the span even when origin cancels it. Use
        // f64 only for this conservative O(1) preflight, never CPU field work.
        for (axis, &count) in self.nodes.iter().enumerate() {
            let origin = f64::from(self.origin[axis]);
            let span = f64::from(count - 1) * f64::from(self.dx);
            let magnitude = origin.abs().max((origin + span).abs()).max(span);
            if f64::from(self.dx) < 4.0 * f64::from(f32::EPSILON) * magnitude {
                return Err(InvalidConfig);
            }
        }
        let nodes = self
            .nodes
            .iter()
            .try_fold(1u32, |n, &v| n.checked_mul(v))
            .ok_or(BudgetExceeded)?;
        let cells = self
            .nodes
            .iter()
            .try_fold(1u32, |n, &v| n.checked_mul(v - 1))
            .ok_or(BudgetExceeded)?;
        // The scan sums at most 36 vertices per cell. Never permit u32 wrap.
        if cells.checked_mul(36).is_none()
            || nodes > self.max_field_nodes
            || cells > self.max_cells
            || particles > u32::MAX as usize
            || u64::from(nodes)
                .checked_mul(particles as u64)
                .is_none_or(|n| n > self.max_sample_visits)
        {
            return Err(BudgetExceeded);
        }
        Ok(Plan {
            nodes,
            cells,
            vertex_bytes: u64::from(self.max_vertices) * 32,
        })
    }

    /// Validates authored work ceilings without allocating or acquiring a GPU.
    pub fn validate(self, particles: usize) -> Result<(), ResidentSurfaceError> {
        self.plan(particles).map(|_| ())
    }

    /// Conservative world-space bounds of every emitted vertex.
    #[must_use]
    pub fn bounds(self) -> (Vec3, Vec3) {
        (
            self.origin,
            self.origin
                + Vec3::new(
                    self.nodes[0].saturating_sub(1) as f32,
                    self.nodes[1].saturating_sub(1) as f32,
                    self.nodes[2].saturating_sub(1) as f32,
                ) * self.dx,
        )
    }
}

#[derive(Debug)]
struct SurfaceBuffers {
    vertices: Buffer,
    indirect: Buffer,
    status: Buffer,
}

/// Submitted immutable surface publication. Clones share ownership, not copies.
/// Readiness is established by submission order on the shared host queue; this
/// is not a CPU completion fence. Device replacement requires a new context
/// identity and consumer invalidation of every old registration.
#[derive(Clone, Debug)]
pub struct ResidentSurface {
    buffers: Arc<SurfaceBuffers>,
    device_identity: Arc<()>,
    sequence: u64,
    config: ResidentSurfaceConfig,
}

impl ResidentSurface {
    /// Tests shared host adoption identity, not merely matching adapter/device properties.
    #[must_use]
    pub fn is_for_context(&self, ctx: &GpuContext) -> bool {
        Arc::ptr_eq(&self.device_identity, ctx.device_identity())
    }
    /// One-based successful publication sequence within this producer's lifetime.
    /// Upload replacement preserves it; a newly created producer starts a new sequence.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }
    /// Returns the authored domain and budgets used to produce this lease.
    #[must_use]
    pub const fn config(&self) -> ResidentSurfaceConfig {
        self.config
    }
    /// Borrow for registration only. Never write or explicitly destroy it.
    #[must_use]
    pub fn vertices(&self) -> &Buffer {
        &self.buffers.vertices
    }
    /// STORAGE | INDIRECT | COPY_SRC. Consume at offset 0, count 1, stride 16.
    #[must_use]
    pub fn indirect(&self) -> &Buffer {
        &self.buffers.indirect
    }
    /// Optional explicit diagnostic/checkpoint readback, not needed to draw.
    #[must_use]
    pub fn status(&self) -> &Buffer {
        &self.buffers.status
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin_dx: [f32; 4],
    dims_particles: [u32; 4],
    sizes: [u32; 4],  // nodes, cells, vertex capacity, unused
    kernel: [f32; 4], // radius, iso, unused, unused
}

pub(super) struct SurfaceProducer {
    layout: BindGroupLayout,
    sample: ComputePipeline,
    classify: ComputePipeline,
    finalize: ComputePipeline,
    emit: ComputePipeline,
    scan: GpuScan,
    live: Vec<Weak<SurfaceBuffers>>,
    sequence: u64,
}

impl SurfaceProducer {
    pub(super) fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prism_mpm_surface"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mpm_surface.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prism_mpm_surface_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prism_mpm_surface_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            sample: pipeline("sample_field"),
            classify: pipeline("classify"),
            finalize: pipeline("finalize"),
            emit: pipeline("emit"),
            layout,
            scan: GpuScan::new(ctx),
            live: Vec::new(),
            sequence: 0,
        }
    }

    pub(super) fn preflight(
        ctx: &GpuContext,
        config: ResidentSurfaceConfig,
        particles: usize,
    ) -> Result<(), ResidentSurfaceError> {
        let p = config.plan(particles)?;
        let limits = ctx.device().limits();
        let max = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let scan_padded = (u64::from(p.cells) + 1).div_ceil(crate::scan::BLOCK as u64)
            * crate::scan::BLOCK as u64
            * 4;
        if [
            p.vertex_bytes,
            u64::from(p.nodes) * 4,
            scan_padded,
            (particles as u64) * 16,
        ]
        .into_iter()
        .any(|size| size > max)
            || limits.max_storage_buffers_per_shader_stage < 6
            || limits.max_compute_invocations_per_workgroup < crate::scan::WORKGROUP
            || limits.max_compute_workgroup_size_x < crate::scan::WORKGROUP
            || limits.max_compute_workgroup_storage_size < crate::scan::BLOCK * 4
            || limits.max_bindings_per_bind_group < 7
            || p.nodes.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || p.cells.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(ResidentSurfaceError::DeviceLimit);
        }
        Ok(())
    }

    pub(super) fn publish(
        &mut self,
        ctx: &GpuContext,
        positions: &Buffer,
        particles: usize,
        config: ResidentSurfaceConfig,
    ) -> Result<ResidentSurface, ResidentSurfaceError> {
        Self::preflight(ctx, config, particles)?;
        self.live.retain(|lease| lease.strong_count() != 0);
        if self.live.len() >= config.max_live_publications {
            return Err(ResidentSurfaceError::InFlightLimit);
        }
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or(ResidentSurfaceError::SequenceExhausted)?;
        let p = config.plan(particles)?;
        let device = ctx.device();
        let allocate = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let buffers = Arc::new(SurfaceBuffers {
            vertices: allocate(
                "prism_mpm_surface_vertices",
                p.vertex_bytes,
                BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
            ),
            indirect: allocate(
                "prism_mpm_surface_indirect",
                16,
                BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            ),
            status: buffer::storage_rw_zeroed(device, "prism_mpm_surface_status", 4),
        });
        let field =
            buffer::storage_rw_zeroed(device, "prism_mpm_surface_field", u64::from(p.nodes) * 4);
        // Last zero sentinel becomes the total after exclusive scan.
        let counts = buffer::storage_rw_zeroed(
            device,
            "prism_mpm_surface_counts",
            (u64::from(p.cells) + 1) * 4,
        );
        let params = buffer::uniform(
            device,
            "prism_mpm_surface_params",
            &Params {
                origin_dx: [config.origin.x, config.origin.y, config.origin.z, config.dx],
                dims_particles: [
                    config.nodes[0],
                    config.nodes[1],
                    config.nodes[2],
                    particles as u32,
                ],
                sizes: [p.nodes, p.cells, config.max_vertices, 0],
                kernel: [config.radius, config.iso, 0.0, 0.0],
            },
        );
        let bind = |offsets: &Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism_mpm_surface_bind"),
                layout: &self.layout,
                entries: &[
                    entry(0, &params),
                    entry(1, positions),
                    entry(2, &field),
                    entry(3, offsets),
                    entry(4, &buffers.vertices),
                    entry(5, &buffers.indirect),
                    entry(6, &buffers.status),
                ],
            })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prism_mpm_surface_encoder"),
        });
        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                        pipeline: &ComputePipeline,
                        group: &wgpu::BindGroup,
                        groups| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prism_mpm_surface_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        };
        let input_bind = bind(&counts);
        dispatch(
            &mut encoder,
            &self.sample,
            &input_bind,
            p.nodes.div_ceil(64),
        );
        dispatch(
            &mut encoder,
            &self.classify,
            &input_bind,
            p.cells.div_ceil(64),
        );
        let levels = self
            .scan
            .record_scan(device, &mut encoder, counts, p.cells as usize + 1);
        let output_bind = bind(levels.result());
        dispatch(&mut encoder, &self.finalize, &output_bind, 1);
        dispatch(&mut encoder, &self.emit, &output_bind, p.cells.div_ceil(64));
        ctx.queue().submit([encoder.finish()]);
        self.sequence = sequence;
        self.live.push(Arc::downgrade(&buffers));
        Ok(ResidentSurface {
            buffers,
            device_identity: ctx.device_identity().clone(),
            sequence,
            config,
        })
    }
}
