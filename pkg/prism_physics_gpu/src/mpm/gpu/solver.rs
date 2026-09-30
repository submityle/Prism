//! Resident, GPU-driven MLS-MPM solver that keeps every particle and grid
//! buffer live on the device across many frames.
//!
//! [`GpuMpmStep::advance`](super::step::GpuMpmStep::advance) is a one-shot
//! convenience: it uploads the batch, advances a fixed number of steps, and
//! reads the result straight back, so every call pays a full upload and
//! read-back. A real-time loop instead wants to upload once, step every frame,
//! and only pull the state back when it actually needs to render or checkpoint.
//! [`GpuMpmResident`] provides that split:
//!
//! * [`GpuMpmResident::upload`] allocates the resident particle and grid buffers
//!   and copies the initial state up once.
//! * [`GpuMpmResident::step`] encodes and submits any number of full steps over
//!   the resident buffers without transferring anything back across the bus.
//! * [`GpuMpmResident::snapshot`] copies the current particle state back for
//!   rendering or checkpointing, leaving the resident buffers untouched so the
//!   simulation keeps advancing from where it left off.
//!
//! Because every stage reuses the exact fused kernels and uniform layout of
//! [`GpuMpmStep`], a resident run of `k` steps is numerically identical to a
//! single `advance` of `k` steps, and splitting those `k` steps across several
//! `step` calls (separate command submissions, i.e. frames) leaves the state
//! bit-for-bit unchanged — the resident-buffer parity test pins both invariants.
//!
//! # Grid residency
//!
//! The four fixed-point atomic accumulators and the finalised node-velocity
//! field are allocated once and reused. The `step_clear` pass zeroes the atomic
//! accumulators at the start of every step, so no stale grid data survives
//! between steps or between `step` calls; only the particle buffers carry state
//! forward, exactly as in the one-shot path.
//!
//! # Provenance
//!
//! The MLS-MPM transfer (Hu et al. 2018), the affine `P2G`/`G2P` conventions
//! (Jiang et al. 2015), and the fixed-corotated / snow plasticity model
//! (Stomakhin et al. 2013) are standard, publicly documented techniques. No
//! Unreal Engine source or derived code.

use glam::{Mat3, Vec3};
use wgpu::{BindGroup, BindGroupDescriptor, Buffer, CommandEncoderDescriptor};

use crate::buffer;
use crate::context::GpuContext;

use super::layout::entry;
use super::params::{cols_to_mat3, mat3_to_cols, vec3_to_vec4};
use super::step::{build_step_params, GpuMpmStep, StepConfig, StepInputs, StepParticles};

/// The device-resident buffer set backing an uploaded particle batch.
///
/// Read-write buffers (positions, velocities, affine `C`, deformation `F`, and
/// the plastic determinant `Jp`) are advected in place each step and copied
/// back on demand; masses and volumes are read-only inputs; the grid buffers
/// are scratch accumulators re-cleared every step.
struct Resident {
    /// Number of particles in the uploaded batch.
    count: usize,
    /// Workgroup count covering the grid nodes (one thread per node).
    node_groups: u32,
    /// Workgroup count covering the particles (one thread per particle).
    particle_groups: u32,
    /// Bound uniform block; kept alive for the lifetime of the bind group.
    _params: Buffer,
    /// Particle positions (read-write, copied back by `snapshot`).
    positions: Buffer,
    /// Particle velocities (read-write, copied back by `snapshot`).
    velocities: Buffer,
    /// Affine velocity matrices `C` (read-write, copied back by `snapshot`).
    affine: Buffer,
    /// Deformation gradients `F` (read-write, copied back by `snapshot`).
    deformation: Buffer,
    /// Per-particle masses (read-only input); kept alive for the bind group.
    _masses: Buffer,
    /// Per-particle volumes (read-only input); kept alive for the bind group.
    _volumes: Buffer,
    /// Plastic determinants `Jp` (read-write, copied back by `snapshot`).
    plastic_det: Buffer,
    /// Fixed-point atomic mass accumulator; kept alive for the bind group.
    _grid_mass: Buffer,
    /// Fixed-point atomic x-momentum accumulator; kept alive for the bind group.
    _grid_mom_x: Buffer,
    /// Fixed-point atomic y-momentum accumulator; kept alive for the bind group.
    _grid_mom_y: Buffer,
    /// Fixed-point atomic z-momentum accumulator; kept alive for the bind group.
    _grid_mom_z: Buffer,
    /// Finalised node velocity field; kept alive for the bind group.
    _grid_velocity: Buffer,
    /// The bind group wiring every buffer to the shared step layout.
    bind: BindGroup,
}

/// A resident, GPU-driven MLS-MPM solver.
///
/// Compile the kernels once with [`GpuMpmResident::new`], upload a batch with
/// [`GpuMpmResident::upload`], then alternate [`GpuMpmResident::step`] and
/// [`GpuMpmResident::snapshot`] to drive a real-time loop without re-uploading
/// the particle state each frame.
pub struct GpuMpmResident {
    /// The compiled full-step pipeline set (module, layout, four kernels).
    step: GpuMpmStep,
    /// The currently uploaded batch, if any.
    resident: Option<Resident>,
}

impl GpuMpmResident {
    /// Compiles every full-step kernel on `ctx`. No batch is resident yet.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMpmResident {
        GpuMpmResident {
            step: GpuMpmStep::new(ctx),
            resident: None,
        }
    }

    /// Uploads `inputs` and allocates the resident particle and grid buffers,
    /// replacing any previously uploaded batch.
    ///
    /// # Panics
    ///
    /// Panics if the [`StepInputs`] slices do not all have the same length.
    pub fn upload(&mut self, ctx: &GpuContext, inputs: &StepInputs, cfg: &StepConfig) {
        let count = inputs.positions.len();
        assert!(
            inputs.velocities.len() == count
                && inputs.affine.len() == count
                && inputs.deformation.len() == count
                && inputs.masses.len() == count
                && inputs.volumes.len() == count
                && inputs.plastic_det.len() == count,
            "StepInputs slices must all have the particle count length",
        );

        let device = ctx.device();
        let (nx, ny, nz) = cfg.dims;
        let node_count = nx * ny * nz;

        let params = build_step_params(cfg, count);
        let params_buf = buffer::uniform(device, "prism_mpm_resident_params", &params);

        let pos_packed: Vec<[f32; 4]> = inputs.positions.iter().map(vec3_to_vec4).collect();
        let vel_packed: Vec<[f32; 4]> = inputs.velocities.iter().map(vec3_to_vec4).collect();
        let affine_cols: Vec<[[f32; 4]; 3]> = inputs.affine.iter().map(mat3_to_cols).collect();
        let deform_cols: Vec<[[f32; 4]; 3]> = inputs.deformation.iter().map(mat3_to_cols).collect();

        let positions =
            buffer::storage_rw_init(device, "prism_mpm_resident_positions", &pos_packed);
        let velocities =
            buffer::storage_rw_init(device, "prism_mpm_resident_velocities", &vel_packed);
        let affine = buffer::storage_rw_init(device, "prism_mpm_resident_affine", &affine_cols);
        let deformation =
            buffer::storage_rw_init(device, "prism_mpm_resident_deformation", &deform_cols);
        let masses = buffer::storage_read(device, "prism_mpm_resident_masses", inputs.masses);
        let volumes = buffer::storage_read(device, "prism_mpm_resident_volumes", inputs.volumes);
        let plastic_det =
            buffer::storage_rw_init(device, "prism_mpm_resident_plastic_det", inputs.plastic_det);

        let atomic_bytes = (node_count * size_of::<i32>()) as u64;
        let node_vec4_bytes = (node_count * size_of::<[f32; 4]>()) as u64;
        let grid_mass =
            buffer::storage_rw_zeroed(device, "prism_mpm_resident_grid_mass", atomic_bytes);
        let grid_mom_x =
            buffer::storage_rw_zeroed(device, "prism_mpm_resident_grid_mom_x", atomic_bytes);
        let grid_mom_y =
            buffer::storage_rw_zeroed(device, "prism_mpm_resident_grid_mom_y", atomic_bytes);
        let grid_mom_z =
            buffer::storage_rw_zeroed(device, "prism_mpm_resident_grid_mom_z", atomic_bytes);
        let grid_velocity =
            buffer::storage_rw_zeroed(device, "prism_mpm_resident_grid_velocity", node_vec4_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_mpm_resident_bind"),
            layout: &self.step.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions),
                entry(2, &velocities),
                entry(3, &affine),
                entry(4, &deformation),
                entry(5, &masses),
                entry(6, &volumes),
                entry(7, &plastic_det),
                entry(8, &grid_mass),
                entry(9, &grid_mom_x),
                entry(10, &grid_mom_y),
                entry(11, &grid_mom_z),
                entry(12, &grid_velocity),
            ],
        });

        let node_groups = u32::try_from(node_count.div_ceil(64)).unwrap_or(u32::MAX);
        let particle_groups = u32::try_from(count.div_ceil(64)).unwrap_or(u32::MAX);

        self.resident = Some(Resident {
            count,
            node_groups,
            particle_groups,
            _params: params_buf,
            positions,
            velocities,
            affine,
            deformation,
            _masses: masses,
            _volumes: volumes,
            plastic_det,
            _grid_mass: grid_mass,
            _grid_mom_x: grid_mom_x,
            _grid_mom_y: grid_mom_y,
            _grid_mom_z: grid_mom_z,
            _grid_velocity: grid_velocity,
            bind,
        });
    }

    /// Returns the number of particles in the resident batch, or `0` if nothing
    /// has been uploaded.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.resident.as_ref().map_or(0, |r| r.count)
    }

    /// Advances the resident batch by `steps` full MLS-MPM steps in a single
    /// command submission, without transferring any state back across the bus.
    ///
    /// A `steps` of `0` submits no work. Call [`GpuMpmResident::snapshot`] to
    /// read the advanced state.
    ///
    /// # Panics
    ///
    /// Panics if no batch has been uploaded with [`GpuMpmResident::upload`].
    pub fn step(&self, ctx: &GpuContext, steps: usize) {
        let resident = self
            .resident
            .as_ref()
            .expect("GpuMpmResident::step called before upload");
        if steps == 0 {
            return;
        }
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_resident_encoder"),
        });
        for _ in 0..steps {
            self.step.pass(
                &mut encoder,
                &self.step.clear,
                &resident.bind,
                resident.node_groups,
            );
            self.step.pass(
                &mut encoder,
                &self.step.p2g,
                &resident.bind,
                resident.particle_groups,
            );
            self.step.pass(
                &mut encoder,
                &self.step.grid,
                &resident.bind,
                resident.node_groups,
            );
            self.step.pass(
                &mut encoder,
                &self.step.g2p,
                &resident.bind,
                resident.particle_groups,
            );
        }
        ctx.queue().submit([encoder.finish()]);
    }

    /// Copies the current resident particle state back for rendering or
    /// checkpointing. The resident buffers are left in place, so the simulation
    /// keeps advancing from the same state on the next [`GpuMpmResident::step`].
    ///
    /// # Panics
    ///
    /// Panics if no batch has been uploaded with [`GpuMpmResident::upload`].
    #[must_use]
    pub fn snapshot(&self, ctx: &GpuContext) -> StepParticles {
        let resident = self
            .resident
            .as_ref()
            .expect("GpuMpmResident::snapshot called before upload");
        let count = resident.count;
        if count == 0 {
            return StepParticles {
                positions: Vec::new(),
                velocities: Vec::new(),
                affine: Vec::new(),
                deformation: Vec::new(),
                plastic_det: Vec::new(),
            };
        }
        let device = ctx.device();
        let vec4_bytes = (count * size_of::<[f32; 4]>()) as u64;
        let mat3_bytes = (count * size_of::<[[f32; 4]; 3]>()) as u64;
        let scalar_bytes = (count * size_of::<f32>()) as u64;

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_resident_snapshot_encoder"),
        });
        let pos_stage = buffer::staging(device, "prism_mpm_resident_positions_stage", vec4_bytes);
        buffer::copy(&mut encoder, &resident.positions, &pos_stage, vec4_bytes);
        let vel_stage = buffer::staging(device, "prism_mpm_resident_velocities_stage", vec4_bytes);
        buffer::copy(&mut encoder, &resident.velocities, &vel_stage, vec4_bytes);
        let affine_stage = buffer::staging(device, "prism_mpm_resident_affine_stage", mat3_bytes);
        buffer::copy(&mut encoder, &resident.affine, &affine_stage, mat3_bytes);
        let deform_stage =
            buffer::staging(device, "prism_mpm_resident_deformation_stage", mat3_bytes);
        buffer::copy(
            &mut encoder,
            &resident.deformation,
            &deform_stage,
            mat3_bytes,
        );
        let jp_stage =
            buffer::staging(device, "prism_mpm_resident_plastic_det_stage", scalar_bytes);
        buffer::copy(&mut encoder, &resident.plastic_det, &jp_stage, scalar_bytes);

        ctx.queue().submit([encoder.finish()]);

        let pos_raw = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let vel_raw = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);
        let affine_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &affine_stage);
        let deform_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &deform_stage);
        let jp_raw = buffer::read_back::<f32>(ctx, &jp_stage);

        StepParticles {
            positions: pos_raw
                .iter()
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect(),
            velocities: vel_raw
                .iter()
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect(),
            affine: affine_raw.iter().map(cols_to_mat3).collect::<Vec<Mat3>>(),
            deformation: deform_raw.iter().map(cols_to_mat3).collect::<Vec<Mat3>>(),
            plastic_det: jp_raw,
        }
    }
}
