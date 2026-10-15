//! Production particle service: water spawn -> bounded simulation -> lagged foam.
//! All resources here belong to the Bevy scene device, never the separate RHI.
use std::collections::BTreeMap;

use bevy_asset::Assets;
use bevy_ecs::prelude::*;
use bevy_material::descriptor::BindGroupLayoutDescriptor;
use bevy_render::{render_resource::*, renderer::RenderDevice};
use bevy_shader::Shader;
use prism_render_architecture::{
    render_fabric::{
        LagOneHistory, PARTICLE_FEATURE_ID, SPRAY_FEEDBACK_FEATURE_ID, SPRAY_FEEDBACK_SERVICE,
    },
    water::{
        breaking::{BreakingCriteria, BreakingSample},
        Vec3,
    },
};

use super::{
    abi::{GpuSprayParams, GpuSpraySource, GpuWaterFoamParams},
    bind_groups::WaterBodyGpuBuffers,
    body::WaterBody,
};
use crate::gpu::fabric::SceneFabric;

const PACKET_CAPACITY: u32 = 256;

/// World-space foam receiving grid and bounded particle time horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SprayFeedbackGrid {
    pub dimensions: [u32; 2],
    pub origin: [f32; 3],
    pub cell_size: f32,
    pub dt: f32,
    pub lifetime: f32,
    pub foam_per_particle: f32,
}

impl SprayFeedbackGrid {
    fn valid(self) -> bool {
        self.dimensions.into_iter().all(|v| v > 0)
            && self.dimensions[0]
                .checked_mul(self.dimensions[1])
                .is_some_and(|cells| cells <= 65_536)
            && self.origin.into_iter().all(f32::is_finite)
            && [self.cell_size, self.dt, self.lifetime]
                .into_iter()
                .all(|v| v.is_finite() && v > 0.0)
            && self.dt <= 1.0
            && self.foam_per_particle.is_finite()
            && self.foam_per_particle >= 0.0
    }

    fn cells(self) -> u32 {
        self.dimensions[0] * self.dimensions[1]
    }
}

/// Authored crest samples consumed by the existing GPU breaking emitter.
#[derive(Clone, Copy, Debug)]
pub struct WaterSpraySource {
    pub position: Vec3,
    pub sample: BreakingSample,
    pub tangent: Vec3,
    pub normal: Vec3,
}

impl WaterBody {
    /// Enables the real GPU emitter and particle feedback service on this body.
    /// Admission is bounded; bad domains/non-finite samples fail at authoring.
    pub fn with_spray_feedback(
        mut self,
        sources: &[WaterSpraySource],
        criteria: BreakingCriteria,
        jet_speed: f32,
        max_spray_count: u32,
        grid: SprayFeedbackGrid,
    ) -> Result<Self, &'static str> {
        if !grid.valid() || sources.len() > PACKET_CAPACITY as usize {
            return Err("invalid or over-budget spray feedback domain");
        }
        if ![
            criteria.steepness_threshold,
            criteria.jacobian_fold_threshold,
            criteria.curvature_threshold,
            criteria.breaking_intensity,
            jet_speed,
        ]
        .into_iter()
        .all(f32::is_finite)
            || jet_speed < 0.0
            || max_spray_count > 65_535
        {
            return Err("invalid spray emitter tuning");
        }
        self.spray_sources = sources
            .iter()
            .map(|source| GpuSpraySource {
                position: [source.position.x, source.position.y, source.position.z],
                steepness: source.sample.steepness,
                tangent: [source.tangent.x, source.tangent.y, source.tangent.z],
                jacobian: source.sample.jacobian,
                normal: [source.normal.x, source.normal.y, source.normal.z],
                curvature: source.sample.curvature,
            })
            .collect();
        if !self.spray_sources.iter().all(|source| {
            bytemuck::cast_slice::<GpuSpraySource, f32>(std::slice::from_ref(source))
                .iter()
                .all(|v| v.is_finite())
        }) {
            return Err("non-finite spray sample");
        }
        self.spray_capacity = sources.len() as u32;
        self.spray_params = GpuSprayParams {
            steepness_threshold: criteria.steepness_threshold,
            jacobian_fold_threshold: criteria.jacobian_fold_threshold,
            curvature_threshold: criteria.curvature_threshold,
            breaking_intensity: criteria.breaking_intensity,
            jet_speed,
            source_count: self.spray_capacity,
            spawn_capacity: self.spray_capacity,
            max_spray_count,
        };
        self.particle_count = self.particle_count.max(self.spray_capacity);
        self.passes.spray = !sources.is_empty();
        self.passes.foam = true;
        self.foam_cells = grid.cells();
        self.counts.foam_cells = self.foam_cells;
        self.foam_sources.resize(self.foam_cells as usize, 0.0);
        self.foam_params = GpuWaterFoamParams {
            nx: grid.dimensions[0],
            nz: grid.dimensions[1],
            dx: grid.cell_size,
            dt: grid.dt,
            base_decay: 1.0,
            persistence_floor: 0.1,
            reference_speed: 1.0,
            _pad: 0,
        };
        self.grid2d_texels = self.grid2d_texels.max(self.foam_cells);
        self.spray_feedback = Some(grid);
        Ok(self)
    }
}

#[derive(Resource)]
pub(crate) struct SprayPipelines {
    layout: BindGroupLayout,
    pub(crate) apply: CachedComputePipelineId,
    pub(crate) simulate: CachedComputePipelineId,
    pub(crate) publish: CachedComputePipelineId,
}

pub(crate) fn init_spray_pipelines(
    mut commands: Commands,
    mut shaders: ResMut<Assets<Shader>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    fabric: Res<SceneFabric>,
) {
    let selected = fabric.0.selection();
    assert_eq!(
        selected.provider(SPRAY_FEEDBACK_SERVICE),
        Some(SPRAY_FEEDBACK_FEATURE_ID)
    );
    let entries: Vec<_> = (0..7)
        .map(|binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: if binding == 6 {
                    BufferBindingType::Uniform
                } else {
                    BufferBindingType::Storage {
                        read_only: matches!(binding, 0 | 1 | 3),
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let descriptor = BindGroupLayoutDescriptor::new("prism particle spray feedback", &entries);
    let layout = device.create_bind_group_layout("prism particle spray feedback", &entries);
    let shader = shaders.add(Shader::from_wgsl(
        include_str!("../shaders/particle_spray_feedback.wgsl"),
        "prism particle spray feedback",
    ));
    let queue = |entry: &'static str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("prism spray {entry}").into()),
            layout: vec![descriptor.clone()],
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..Default::default()
        })
    };
    commands.insert_resource(SprayPipelines {
        layout,
        apply: queue("apply_feedback"),
        simulate: queue("simulate"),
        publish: queue("publish_feedback"),
    });
}

struct SprayEntry {
    grid: SprayFeedbackGrid,
    particles: [Buffer; 2],
    feedback: [Buffer; 2],
    history: LagOneHistory,
}

#[derive(Resource, Default)]
pub(crate) struct SprayHistory {
    pub(crate) frame: u64,
    pub(crate) dispatched: Option<u64>,
    entries: BTreeMap<u64, SprayEntry>,
}

impl SprayHistory {
    pub(crate) fn begin_frame(&mut self, entities: &[Entity]) {
        self.frame = self
            .frame
            .checked_add(1)
            .expect("spray history frame overflow");
        self.dispatched = None;
        self.entries
            .retain(|key, _| entities.iter().any(|entity| entity.to_bits() == *key));
    }

    pub(crate) fn prepare(
        &mut self,
        entity: Entity,
        body: &WaterBody,
        buffers: &WaterBodyGpuBuffers,
        device: &RenderDevice,
        pipelines: &SprayPipelines,
        fabric: &SceneFabric,
    ) -> Option<PreparedSpray> {
        let Some(grid) = body.spray_feedback else {
            self.entries.remove(&entity.to_bits());
            return None;
        };
        fabric
            .0
            .admit(PARTICLE_FEATURE_ID)
            .expect("particle content outside frozen build roots");
        assert!(
            grid.valid(),
            "invalid spray grid reached device preparation"
        );
        let make_buffer = |label, size| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let entry = self
            .entries
            .entry(entity.to_bits())
            .or_insert_with(|| SprayEntry {
                grid,
                particles: std::array::from_fn(|_| {
                    make_buffer(
                        "prism particle spray state",
                        16 + u64::from(PACKET_CAPACITY) * 32,
                    )
                }),
                feedback: std::array::from_fn(|_| {
                    make_buffer("prism particle foam history", u64::from(grid.cells()) * 4)
                }),
                history: LagOneHistory::default(),
            });
        if entry.grid != grid {
            entry.grid = grid;
            entry.feedback = std::array::from_fn(|_| {
                make_buffer("prism particle foam history", u64::from(grid.cells()) * 4)
            });
            entry.history = LagOneHistory::default();
        }
        let write = LagOneHistory::write_slot(self.frame);
        let read = 1 - write;
        let params = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism particle spray domain"),
            contents: bytemuck::cast_slice(&[
                grid.dimensions[0],
                grid.dimensions[1],
                body.spray_params.source_count.min(body.spray_capacity),
                PACKET_CAPACITY,
                grid.origin[0].to_bits(),
                grid.origin[1].to_bits(),
                grid.origin[2].to_bits(),
                grid.cell_size.to_bits(),
                grid.dt.to_bits(),
                grid.lifetime.to_bits(),
                grid.foam_per_particle.to_bits(),
                0,
            ]),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(
            "prism particle spray feedback",
            &pipelines.layout,
            &BindGroupEntries::sequential((
                buffers.spray_spawn.as_entire_binding(),
                entry.particles[read].as_entire_binding(),
                entry.particles[write].as_entire_binding(),
                entry.feedback[read].as_entire_binding(),
                entry.feedback[write].as_entire_binding(),
                buffers.foam_sources.as_entire_binding(),
                params.as_entire_binding(),
            )),
        );
        Some(PreparedSpray {
            key: entity.to_bits(),
            bind_group,
            cells: grid.cells(),
            dimensions: grid.dimensions,
        })
    }

    /// Invalid/missed publication means an explicit empty input, never stale data.
    pub(crate) fn reset_missing(&self, encoder: &mut CommandEncoder, prepared: &PreparedSpray) {
        let entry = &self.entries[&prepared.key];
        if entry.history.read_slot(self.frame).is_none() {
            let read = 1 - LagOneHistory::write_slot(self.frame);
            encoder.clear_buffer(&entry.particles[read], 0, None);
            encoder.clear_buffer(&entry.feedback[read], 0, None);
        }
    }

    pub(crate) fn publish(&mut self, prepared: &PreparedSpray) {
        assert!(self
            .entries
            .get_mut(&prepared.key)
            .expect("prepared spray entity")
            .history
            .publish(self.frame));
    }
}

pub(crate) struct PreparedSpray {
    key: u64,
    pub(crate) bind_group: BindGroup,
    pub(crate) cells: u32,
    pub(crate) dimensions: [u32; 2],
}

#[cfg(test)]
mod tests {
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

    use super::*;

    fn keep_wgsl(
        _: &(),
        source: ShaderCacheSource,
        _: &ValidateShader,
    ) -> Result<String, ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("particle feedback is WGSL"),
        }
    }

    #[test]
    fn particle_feedback_wgsl_compiles_through_shader_cache() {
        let mut cache = ShaderCache::new((), keep_wgsl);
        let id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d5f_5350_5241_5900_0000_0001),
        };
        cache.set_shader(
            id,
            Shader::from_wgsl(
                include_str!("../shaders/particle_spray_feedback.wgsl"),
                "embedded://prism_render_scene/shaders/particle_spray_feedback.wgsl",
            ),
        );
        cache.get(0, id, &[]).unwrap_or_else(|error| {
            panic!("particle_spray_feedback.wgsl failed to compile: {error}")
        });
    }

    #[test]
    fn authoring_activates_real_emitter_and_full_foam_domain() {
        let grid = SprayFeedbackGrid {
            dimensions: [3, 5],
            origin: [0.0; 3],
            cell_size: 1.0,
            dt: 0.1,
            lifetime: 2.0,
            foam_per_particle: 0.1,
        };
        let criteria = BreakingCriteria {
            steepness_threshold: 1.0,
            jacobian_fold_threshold: 0.0,
            curvature_threshold: 1.0,
            breaking_intensity: 0.5,
        };
        let source = WaterSpraySource {
            position: Vec3::ZERO,
            sample: BreakingSample {
                steepness: 2.0,
                jacobian: -1.0,
                curvature: 2.0,
            },
            tangent: Vec3::ZERO,
            normal: Vec3::new(0.0, 1.0, 0.0),
        };
        let body = WaterBody::default()
            .with_spray_feedback(&[source], criteria, 1.0, 10, grid)
            .unwrap();
        let plan = prism_render_architecture::water::gpu::pipeline::prepare(&body.as_extract());
        assert!(
            plan.dispatches
                .iter()
                .any(|d| d.kernel
                    == prism_render_architecture::water::kernels::WaterKernel::SprayEmit)
        );
        assert!(plan.dispatches.iter().any(
            |d| d.kernel == prism_render_architecture::water::kernels::WaterKernel::FoamAdvect
        ));
        assert_eq!(body.foam_sources.len(), 15);
        assert!(WaterBody::default()
            .with_spray_feedback(&[source; 257], criteria, 1.0, 10, grid)
            .is_err());
        assert!(WaterBody::default()
            .with_spray_feedback(
                &[],
                criteria,
                1.0,
                10,
                SprayFeedbackGrid {
                    dt: f32::NAN,
                    ..grid
                }
            )
            .is_err());
    }
}
