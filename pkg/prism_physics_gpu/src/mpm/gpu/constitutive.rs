//! Real-device `wgpu` pipeline that evaluates the MLS-MPM constitutive kernels
//! in isolation, so a parity test can pin the ported arithmetic before the
//! kernels are fused into the transfer pipeline.
//!
//! [`GpuMpmConstitutive`] compiles the shared pure-function library
//! `shaders/mpm_math.wgsl` concatenated with the probe entry point in
//! `shaders/mpm_constitutive_probe.wgsl`, and exposes
//! [`GpuMpmConstitutive::evaluate`], which runs the probe over a batch of
//! deformation gradients and returns, per particle:
//!
//! 1. the fixed-corotated stress `P Fᵀ` (with the hardening-scaled Lamé
//!    parameters when plasticity is enabled),
//! 2. the polar-decomposition rotation `R`,
//! 3. the snow return-mapping's corrected elastic deformation gradient, and
//! 4. the updated plastic determinant `Jp`.
//!
//! This is the device twin of the per-particle constitutive calls in the CPU
//! golden [`prism_physics_core::mpm`] path (`corotated_pf`, `polar_rotation`,
//! `hardening_factor`, and `snow_return_mapping`).
//!
//! # Provenance
//!
//! The fixed-corotated energy and snow return-mapping (Stomakhin et al. 2013)
//! and the affine MLS-MPM conventions (Hu et al. 2018; Jiang et al. 2015) are
//! standard, publicly documented techniques. No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use glam::Mat3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::layout::{buffer_entry, entry};
use super::params::{cols_to_mat3, mat3_to_cols};

/// Uniform parameter block. Layout matches `ProbeParams` in
/// `shaders/mpm_constitutive_probe.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProbeParams {
    /// `x` = particle count, `y` = plastic-enabled flag (`0`/`1`), `zw` pad.
    counts: [u32; 4],
    /// `x` = shear modulus `μ0`, `y` = first Lamé `λ0`, `z` = hardening `ξ`,
    /// `w` = padding.
    mat: [f32; 4],
    /// `x` = critical compression `θc`, `y` = critical stretch `θs`, `zw` pad.
    plast: [f32; 4],
}

/// The per-particle outputs of one constitutive-probe evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstitutiveOutput {
    /// Fixed-corotated stress `P Fᵀ` per particle.
    pub pf: Vec<Mat3>,
    /// Polar-decomposition rotation `R` per particle.
    pub polar: Vec<Mat3>,
    /// Snow return-mapping corrected elastic deformation gradient per particle.
    pub f_elastic: Vec<Mat3>,
    /// Updated plastic determinant `Jp` per particle.
    pub plastic_det: Vec<f32>,
}

/// A compiled, reusable constitutive-probe `GPU` pipeline.
pub struct GpuMpmConstitutive {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMpmConstitutive {
    /// Compiles the constitutive-probe kernel on `ctx`.
    ///
    /// The shared math library is concatenated ahead of the probe entry point
    /// because WGSL has no include directive.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMpmConstitutive {
        let device = ctx.device();
        let source = format!(
            "{}\n{}",
            include_str!("../../shaders/mpm_math.wgsl"),
            include_str!("../../shaders/mpm_constitutive_probe.wgsl"),
        );
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_mpm_constitutive_probe"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_mpm_constitutive_probe_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_mpm_constitutive_probe_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_mpm_constitutive_probe_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("constitutive_probe"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMpmConstitutive {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the constitutive kernels for every deformation gradient in
    /// `deformations`, using per-particle plastic determinants `plastic_dets`.
    ///
    /// `mu0` and `lambda0` are the material Lamé parameters, `hardening` is the
    /// snow hardening coefficient `ξ`, and `theta_c`/`theta_s` are the snow
    /// critical compression/stretch. When `plastic` is `true` the Lamé
    /// parameters are scaled by `hardening_factor(ξ, Jp)` (mirroring the CPU
    /// P2G path); otherwise they are used unscaled.
    ///
    /// # Panics
    ///
    /// Panics if `deformations` and `plastic_dets` have different lengths.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        deformations: &[Mat3],
        plastic_dets: &[f32],
        mu0: f32,
        lambda0: f32,
        hardening: f32,
        theta_c: f32,
        theta_s: f32,
        plastic: bool,
    ) -> ConstitutiveOutput {
        assert_eq!(
            deformations.len(),
            plastic_dets.len(),
            "deformations and plastic determinants must have equal length",
        );
        let count = deformations.len();
        if count == 0 {
            return ConstitutiveOutput {
                pf: Vec::new(),
                polar: Vec::new(),
                f_elastic: Vec::new(),
                plastic_det: Vec::new(),
            };
        }
        let device = ctx.device();

        let params = ProbeParams {
            counts: [
                u32::try_from(count).unwrap_or(u32::MAX),
                u32::from(plastic),
                0,
                0,
            ],
            mat: [mu0, lambda0, hardening, 0.0],
            plast: [theta_c, theta_s, 0.0, 0.0],
        };
        let params_buf = buffer::uniform(device, "prism_mpm_constitutive_params", &params);

        let f_cols: Vec<[[f32; 4]; 3]> = deformations.iter().map(mat3_to_cols).collect();
        let f_in = buffer::storage_read(device, "prism_mpm_constitutive_f_in", &f_cols);
        let jp_in = buffer::storage_read(device, "prism_mpm_constitutive_jp_in", plastic_dets);

        let mat_bytes = (count * size_of::<[[f32; 4]; 3]>()) as u64;
        let scalar_bytes = (count * size_of::<f32>()) as u64;
        let pf_out = buffer::storage_rw_zeroed(device, "prism_mpm_constitutive_pf_out", mat_bytes);
        let polar_out =
            buffer::storage_rw_zeroed(device, "prism_mpm_constitutive_polar_out", mat_bytes);
        let f_elastic_out =
            buffer::storage_rw_zeroed(device, "prism_mpm_constitutive_f_elastic_out", mat_bytes);
        let jp_out =
            buffer::storage_rw_zeroed(device, "prism_mpm_constitutive_jp_out", scalar_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_mpm_constitutive_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &f_in),
                entry(2, &jp_in),
                entry(3, &pf_out),
                entry(4, &polar_out),
                entry(5, &f_elastic_out),
                entry(6, &jp_out),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_constitutive_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_mpm_constitutive_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = u32::try_from(count.div_ceil(64)).unwrap_or(u32::MAX);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let pf_stage = buffer::staging(device, "prism_mpm_constitutive_pf_stage", mat_bytes);
        buffer::copy(&mut encoder, &pf_out, &pf_stage, mat_bytes);
        let polar_stage = buffer::staging(device, "prism_mpm_constitutive_polar_stage", mat_bytes);
        buffer::copy(&mut encoder, &polar_out, &polar_stage, mat_bytes);
        let f_elastic_stage =
            buffer::staging(device, "prism_mpm_constitutive_f_elastic_stage", mat_bytes);
        buffer::copy(&mut encoder, &f_elastic_out, &f_elastic_stage, mat_bytes);
        let jp_stage = buffer::staging(device, "prism_mpm_constitutive_jp_stage", scalar_bytes);
        buffer::copy(&mut encoder, &jp_out, &jp_stage, scalar_bytes);

        ctx.queue().submit([encoder.finish()]);

        let pf_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &pf_stage);
        let polar_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &polar_stage);
        let f_elastic_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &f_elastic_stage);
        let plastic_det = buffer::read_back::<f32>(ctx, &jp_stage);

        ConstitutiveOutput {
            pf: pf_raw.iter().map(cols_to_mat3).collect(),
            polar: polar_raw.iter().map(cols_to_mat3).collect(),
            f_elastic: f_elastic_raw.iter().map(cols_to_mat3).collect(),
            plastic_det,
        }
    }
}
