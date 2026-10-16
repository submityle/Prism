//! Runtime execution of the declarative shading frame graph.
//!
//! The graph compiler owns pass order and culling.  This module only binds the
//! compiled pass table to ECS systems and runs the surviving systems in the
//! compiler-provided order.  ECS schedules remain phase boundaries around the
//! driver; they are not a second pass-order source.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use bevy_ecs::{
    prelude::{Mut, Resource, World},
    system::BoxedSystem,
};
use prism_render_architecture::render_fabric::{DeviceFabric, FeatureId, VariantId};
use prism_render_graph::{
    frame_graph::{AccessKind, GpuFrameGraphBuilder},
    ExecutionPlan,
};

/// These are existing Core3d schedule boundaries, not another pass order.
/// Water stays outside the four drivers, between resolve and response.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ShadingExecutionPhase {
    PreResolve,
    Resolve,
    WaterSurface,
    Response,
    Post,
}

impl ShadingExecutionPhase {
    fn for_declaration(declaration: &str) -> Option<Self> {
        match declaration {
            "visibility_raster"
            | "material_classification_count"
            | "material_classification_prefix"
            | "material_classification_scatter"
            | "texel_space_shade_cache" => Some(Self::PreResolve),
            "shading_resolve" => Some(Self::Resolve),
            "water_surface_raster" => Some(Self::WaterSurface),
            "response_composition" => Some(Self::Response),
            name if super::post_graph::POST_PROCESS_DECLARATIONS
                .iter()
                .any(|pass| pass.name == name) =>
            {
                Some(Self::Post)
            }
            _ => None,
        }
    }
}

type ShadingPhaseRanges = [Range<usize>; 5];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShadingResourceAccess {
    pub(crate) resource: String,
    pub(crate) kind: AccessKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SelectedVariantRequirement {
    pub(crate) feature: FeatureId,
    pub(crate) variant: VariantId,
}

pub(crate) struct ShadingPassBinding {
    pub(crate) declaration: &'static str,
    pub(crate) requires: &'static [SelectedVariantRequirement],
    pub(crate) executor: ShadingExecutorBinding,
}

pub(crate) enum ShadingExecutorBinding {
    /// The sole owner of a contiguous group of compiled declarations.
    Systems {
        group: &'static str,
        systems: Vec<BoxedSystem>,
    },
    /// A declaration executed by the named owner's systems, not an empty system.
    GroupMember {
        group: &'static str,
        owner: &'static str,
    },
    /// Admission only: the real executor remains in its existing ECS schedule.
    ExternalSchedule { site: ExternalExecutionSite },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ExternalExecutionSite {
    WaterSurface,
}

/// Populated by the plugins that install the corresponding real schedules.
#[derive(Resource, Default)]
pub(crate) struct ShadingExternalExecutors {
    sites: HashSet<ExternalExecutionSite>,
}

impl ShadingExternalExecutors {
    pub(crate) fn register(&mut self, site: ExternalExecutionSite) -> Result<(), String> {
        if !self.sites.insert(site) {
            return Err(format!(
                "shading external executor {site:?} is already registered"
            ));
        }
        Ok(())
    }

    pub(crate) fn contains(&self, site: ExternalExecutionSite) -> bool {
        self.sites.contains(&site)
    }
}

pub(crate) struct ShadingPassRuntime {
    pub(crate) declaration: String,
    pub(crate) position: Option<usize>,
    pub(crate) accesses: Vec<ShadingResourceAccess>,
    executor: ShadingExecutorBinding,
    initialized: bool,
}

#[derive(Resource)]
pub(crate) struct ShadingExecutionPlan {
    compiled: ExecutionPlan,
    pass_index_by_name: HashMap<String, usize>,
    passes: Vec<ShadingPassRuntime>,
    executed_groups: HashSet<&'static str>,
    driver_ranges: Option<ShadingPhaseRanges>,
}

impl ShadingExecutionPlan {
    /// Admit bindings against one device selection and the supplied compiler
    /// result. The caller must compile this exact declaration table; validation
    /// checks table consistency, not graph provenance or another pass schedule.
    pub(crate) fn new(
        fabric: &DeviceFabric,
        external: &ShadingExternalExecutors,
        graph: &GpuFrameGraphBuilder,
        compiled: ExecutionPlan,
        bindings: Vec<ShadingPassBinding>,
    ) -> Result<Self, String> {
        Self::validate_compiled(&compiled, graph.passes().len())?;

        let mut binding_by_name = HashMap::with_capacity(bindings.len());
        for binding in bindings {
            if binding_by_name
                .insert(binding.declaration, binding)
                .is_some()
            {
                return Err("shading execution ledger contains a duplicate declaration".into());
            }
        }

        let resource_names = graph
            .resources()
            .iter()
            .map(|resource| resource.name.to_string())
            .collect::<Vec<_>>();
        let mut pass_index_by_name = HashMap::with_capacity(graph.passes().len());
        let mut passes = Vec::with_capacity(graph.passes().len());

        for (index, pass) in graph.passes().iter().enumerate() {
            let declaration = pass.name.to_string();
            if pass_index_by_name
                .insert(declaration.clone(), index)
                .is_some()
            {
                return Err(format!(
                    "shading graph contains duplicate pass {declaration}"
                ));
            }
            let binding = binding_by_name
                .remove(declaration.as_str())
                .ok_or_else(|| {
                    format!("shading execution ledger is missing declaration {declaration}")
                })?;
            // Use the device's frozen selection, including root permission.
            // A dependency-closure service is not permission to execute a root.
            for requirement in binding.requires {
                let selected = fabric.admit(requirement.feature).map_err(|error| {
                    format!(
                        "shading pass {declaration} requires frozen root {:?}: {error}",
                        requirement.feature
                    )
                })?;
                if selected.variant_id() != requirement.variant {
                    return Err(format!(
                        "shading pass {declaration} requires feature {:?} variant {:?}, but frozen selection chose {:?}",
                        requirement.feature,
                        requirement.variant,
                        selected.variant_id()
                    ));
                }
            }
            if let ShadingExecutorBinding::ExternalSchedule { site } = &binding.executor
                && !external.contains(*site)
            {
                return Err(format!(
                    "shading pass {declaration} requires unregistered external executor {site:?}"
                ));
            }
            let accesses = pass
                .accesses
                .iter()
                .map(|access| {
                    let resource =
                        resource_names
                            .get(access.resource.0 as usize)
                            .ok_or_else(|| {
                                format!(
                                    "pass {declaration} references unknown resource {}",
                                    access.resource.0
                                )
                            })?;
                    Ok(ShadingResourceAccess {
                        resource: resource.clone(),
                        kind: access.kind,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            passes.push(ShadingPassRuntime {
                declaration,
                position: compiled.position(index),
                accesses,
                executor: binding.executor,
                initialized: false,
            });
        }

        if !binding_by_name.is_empty() {
            return Err("shading execution ledger contains an unknown declaration".into());
        }
        Self::validate_executor_groups(&passes, &pass_index_by_name)?;
        let driver_ranges = Self::validate_driver_phases(&passes, &compiled)?;

        Ok(Self {
            compiled,
            pass_index_by_name,
            passes,
            executed_groups: HashSet::new(),
            driver_ranges,
        })
    }

    /// Check the compiler's index tables, without deriving another schedule.
    fn validate_compiled(compiled: &ExecutionPlan, pass_count: usize) -> Result<(), String> {
        if compiled.declared_pass_count != pass_count {
            return Err(format!(
                "compiled shading plan declares {} passes but graph has {pass_count}",
                compiled.declared_pass_count
            ));
        }
        if compiled.alive.len() != pass_count || compiled.position_by_pass.len() != pass_count {
            return Err(format!(
                "compiled shading plan liveness/position table lengths must equal {pass_count}"
            ));
        }
        if compiled.barriers.len() != compiled.order.len()
            || compiled.attachments.len() != compiled.order.len()
        {
            return Err("compiled shading plan ordered metadata lengths disagree".into());
        }
        let mut seen = vec![false; pass_count];
        for (position, &index) in compiled.order.iter().enumerate() {
            let entry = seen
                .get_mut(index)
                .ok_or_else(|| format!("compiled shading plan references unknown pass {index}"))?;
            if *entry {
                return Err(format!("compiled shading plan repeats pass {index}"));
            }
            *entry = true;
            if !compiled.alive[index] || compiled.position_by_pass[index] != Some(position) {
                return Err(format!(
                    "compiled shading plan pass {index} has inconsistent liveness/position at {position}"
                ));
            }
        }
        for (index, &scheduled) in seen.iter().enumerate() {
            if compiled.alive[index] != scheduled
                || compiled.position_by_pass[index].is_some() != scheduled
            {
                return Err(format!(
                    "compiled shading plan pass {index} disagrees with survivor order"
                ));
            }
        }
        Ok(())
    }

    fn validate_executor_groups(
        passes: &[ShadingPassRuntime],
        pass_index_by_name: &HashMap<String, usize>,
    ) -> Result<(), String> {
        let mut groups = HashMap::new();
        for (index, pass) in passes.iter().enumerate() {
            if let ShadingExecutorBinding::Systems { group, systems } = &pass.executor {
                if systems.is_empty() {
                    return Err(format!(
                        "shading pass {} has empty Systems",
                        pass.declaration
                    ));
                }
                if groups.insert(*group, (index, vec![index])).is_some() {
                    return Err(format!(
                        "shading executor group {group} has multiple owners"
                    ));
                }
            }
        }
        for (index, pass) in passes.iter().enumerate() {
            if let ShadingExecutorBinding::GroupMember { group, owner } = &pass.executor {
                let (owner_index, members) = groups.get_mut(*group).ok_or_else(|| {
                    format!(
                        "shading executor group {group} has no Systems owner for {}",
                        pass.declaration
                    )
                })?;
                if pass_index_by_name.get(*owner) != Some(&*owner_index) {
                    return Err(format!(
                        "shading group member {} names owner {owner}, but group {group} is owned by {}",
                        pass.declaration, passes[*owner_index].declaration
                    ));
                }
                members.push(index);
            }
        }
        for (group, (owner_index, members)) in groups {
            let owner = &passes[owner_index];
            let phase = ShadingExecutionPhase::for_declaration(&owner.declaration);
            if members.iter().any(|&index| {
                ShadingExecutionPhase::for_declaration(&passes[index].declaration) != phase
            }) {
                return Err(format!(
                    "shading executor group {group} crosses driver phases"
                ));
            }
            if members
                .iter()
                .any(|&index| passes[index].position.is_some() != owner.position.is_some())
            {
                return Err(format!("shading executor group {group} has mixed liveness"));
            }
            let Some(owner_position) = owner.position else {
                continue;
            };
            let mut first = owner_position;
            let mut last = owner_position;
            for &index in &members {
                let position = passes[index].position.expect("validated group liveness");
                first = first.min(position);
                last = last.max(position);
            }
            if first != owner_position {
                return Err(format!(
                    "shading executor group {group} owner {} must be first",
                    owner.declaration
                ));
            }
            if last - first + 1 != members.len() {
                return Err(format!(
                    "shading executor group {group} is not contiguous in compiled order"
                ));
            }
        }
        Ok(())
    }

    /// Project the compiler's surviving order onto existing schedule phases.
    /// Reject an incompatible order instead of sorting or recompiling it.
    fn validate_driver_phases(
        passes: &[ShadingPassRuntime],
        compiled: &ExecutionPlan,
    ) -> Result<Option<ShadingPhaseRanges>, String> {
        let phases = passes
            .iter()
            .map(|pass| ShadingExecutionPhase::for_declaration(&pass.declaration))
            .collect::<Vec<_>>();
        for (pass, phase) in passes.iter().zip(&phases) {
            match (&pass.executor, phase) {
                (
                    ShadingExecutorBinding::ExternalSchedule {
                        site: ExternalExecutionSite::WaterSurface,
                    },
                    Some(ShadingExecutionPhase::WaterSurface),
                ) => {}
                (ShadingExecutorBinding::ExternalSchedule { site }, _) => {
                    return Err(format!(
                        "shading external executor {site:?} must bind water_surface_raster, not {}",
                        pass.declaration
                    ));
                }
                (_, Some(ShadingExecutionPhase::WaterSurface)) => {
                    return Err(
                        "shading pass water_surface_raster must use external executor WaterSurface"
                            .into(),
                    );
                }
                _ => {}
            }
        }
        // Direct range execution also supports small, non-production graphs.
        // They have no Core3d driver contract and cannot run via these drivers.
        if phases.iter().all(Option::is_none) {
            return Ok(None);
        }
        if let Some(index) = phases.iter().position(Option::is_none) {
            return Err(format!(
                "shading pass {} has no driver phase",
                passes[index].declaration
            ));
        }

        let mut ranges: ShadingPhaseRanges = std::array::from_fn(|_| 0..0);
        let mut previous = None;
        for (position, &index) in compiled.order.iter().enumerate() {
            let phase = phases[index].expect("validated driver phase");
            if previous.is_some_and(|previous| previous > phase) {
                return Err(format!(
                    "compiled shading order crosses driver phases backwards at {} ({phase:?})",
                    passes[index].declaration
                ));
            }
            let range = &mut ranges[phase as usize];
            if range.start >= range.end {
                range.start = position;
            }
            range.end = position + 1;
            previous = Some(phase);
        }
        Ok(Some(ranges))
    }

    fn run_driver_phase(
        &mut self,
        world: &mut World,
        phase: ShadingExecutionPhase,
    ) -> Result<(), String> {
        let ranges = self
            .driver_ranges
            .as_ref()
            .ok_or_else(|| "shading plan has no Core3d driver phases".to_string())?;
        let positions = ranges[phase as usize].clone();
        self.run_position_range(world, positions)
    }

    pub(crate) fn compiled(&self) -> &ExecutionPlan {
        &self.compiled
    }

    pub(crate) fn pass(&self, declaration: &str) -> Option<&ShadingPassRuntime> {
        self.pass_index_by_name
            .get(declaration)
            .and_then(|index| self.passes.get(*index))
    }

    pub(crate) fn is_alive(&self, declaration: &str) -> bool {
        self.pass(declaration)
            .is_some_and(|pass| pass.position.is_some())
    }

    /// Name lookup and position lookup are both constant-time: the first uses
    /// the prebuilt hash map and the second uses `ExecutionPlan::position`.
    pub(crate) fn position(&self, declaration: &str) -> Option<usize> {
        self.pass(declaration).and_then(|pass| pass.position)
    }

    pub(crate) fn begin_frame(&mut self) {
        self.executed_groups.clear();
    }

    pub(crate) fn run_position_range(
        &mut self,
        world: &mut World,
        positions: Range<usize>,
    ) -> Result<(), String> {
        let end = positions.end.min(self.compiled.order.len());
        let start = positions.start.min(end);
        for &pass_index in &self.compiled.order[start..end] {
            let runtime = self
                .passes
                .get_mut(pass_index)
                .ok_or_else(|| format!("compiled plan references unknown pass {pass_index}"))?;
            let (group, systems) = match &mut runtime.executor {
                ShadingExecutorBinding::Systems { group, systems } => (*group, systems),
                ShadingExecutorBinding::GroupMember { group, owner } => {
                    if !self.executed_groups.contains(*group) {
                        return Err(format!(
                            "shading group member {} reached before owner {owner} executed",
                            runtime.declaration
                        ));
                    }
                    continue;
                }
                // The schedule owner executes this site at its real phase
                // boundary. Traversing it here does not claim GPU execution.
                ShadingExecutorBinding::ExternalSchedule { .. } => continue,
            };
            if self.executed_groups.contains(group) {
                continue;
            }
            if !runtime.initialized {
                for executor in systems.iter_mut() {
                    executor.initialize(world);
                }
                runtime.initialized = true;
            }
            for executor in systems.iter_mut() {
                executor.run((), world).map_err(|error| {
                    format!("shading executor {} failed: {error}", executor.name())
                })?;
            }
            self.executed_groups.insert(group);
        }
        Ok(())
    }

    pub(crate) fn run_named_range(
        &mut self,
        world: &mut World,
        first: &str,
        last: &str,
    ) -> Result<(), String> {
        let start = self
            .position(first)
            .ok_or_else(|| format!("shading range starts at culled or unknown pass {first}"))?;
        let end = self
            .position(last)
            .ok_or_else(|| format!("shading range ends at culled or unknown pass {last}"))?;
        if start > end {
            return Err(format!("invalid shading range {first}..={last}"));
        }
        self.run_position_range(world, start..end + 1)
    }

    pub(crate) fn run_named_pass(
        &mut self,
        world: &mut World,
        declaration: &str,
    ) -> Result<(), String> {
        let position = self
            .position(declaration)
            .ok_or_else(|| format!("shading pass is culled or unknown: {declaration}"))?;
        self.run_position_range(world, position..position + 1)
    }

    pub(crate) fn with_resource<R>(
        world: &mut World,
        f: impl FnOnce(&mut World, Mut<Self>) -> R,
    ) -> R {
        world.resource_scope(f)
    }
}

pub(crate) fn run_shading_pre_resolve_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        // Reset even when this frame has no surviving pre-resolve passes.
        plan.begin_frame();
        if let Err(error) = plan.run_driver_phase(world, ShadingExecutionPhase::PreResolve) {
            panic!("shading pre-resolve execution failed: {error}");
        }
    });
}

pub(crate) fn run_shading_resolve_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        if let Err(error) = plan.run_driver_phase(world, ShadingExecutionPhase::Resolve) {
            panic!("shading resolve execution failed: {error}");
        }
    });
}

pub(crate) fn run_shading_response_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        if let Err(error) = plan.run_driver_phase(world, ShadingExecutionPhase::Response) {
            panic!("shading response execution failed: {error}");
        }
    });
}

pub(crate) fn run_shading_post_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        if let Err(error) = plan.run_driver_phase(world, ShadingExecutionPhase::Post) {
            panic!("shading post execution failed: {error}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{
        prelude::ResMut,
        schedule::{IntoScheduleConfigs, Schedule, SystemSet},
        system::IntoSystem,
    };
    use prism_render_architecture::{
        backend::capability::CapabilitySet,
        render_fabric::{
            builtin_catalog, FrozenRoots, RenderQualityTier, LIT_FEATURE_ID, LIT_VARIANT_ID,
            NPR_FEATURE_ID, SPRAY_FEEDBACK_FEATURE_ID, WATER_FEATURE_ID, WATER_VARIANT_ID,
        },
    };
    use prism_render_graph::frame_graph::{
        PassDescriptor, QueueClass, ResourceDescriptor, ResourceLifetime,
    };

    fn test_graph() -> GpuFrameGraphBuilder {
        let mut graph = GpuFrameGraphBuilder::default();
        let input = graph.add_resource(ResourceDescriptor {
            name: "input".into(),
            lifetime: ResourceLifetime::Imported,
            size: 0,
            alignment: 1,
        });
        let output = graph.add_resource(ResourceDescriptor {
            name: "output".into(),
            lifetime: ResourceLifetime::Imported,
            size: 0,
            alignment: 1,
        });
        let intermediate = graph.add_resource(ResourceDescriptor {
            name: "intermediate".into(),
            lifetime: ResourceLifetime::Transient,
            size: 0,
            alignment: 1,
        });
        graph.add_pass(PassDescriptor {
            name: "first".into(),
            queue: QueueClass::Compute,
            accesses: vec![
                prism_render_graph::frame_graph::ResourceAccess {
                    resource: input,
                    kind: AccessKind::SampledRead,
                },
                prism_render_graph::frame_graph::ResourceAccess {
                    resource: intermediate,
                    kind: AccessKind::StorageWrite,
                },
            ],
            depends_on: vec![],
        });
        graph.add_pass(PassDescriptor {
            name: "culled".into(),
            queue: QueueClass::Compute,
            accesses: vec![],
            depends_on: vec![],
        });
        graph.add_pass(PassDescriptor {
            name: "last".into(),
            queue: QueueClass::Compute,
            accesses: vec![
                prism_render_graph::frame_graph::ResourceAccess {
                    resource: intermediate,
                    kind: AccessKind::StorageRead,
                },
                prism_render_graph::frame_graph::ResourceAccess {
                    resource: output,
                    kind: AccessKind::StorageWrite,
                },
            ],
            depends_on: vec![prism_render_graph::frame_graph::PassId(0)],
        });
        graph
    }

    #[derive(Resource, Default)]
    struct ExecutionTrace(Vec<&'static str>);

    fn test_fabric() -> DeviceFabric {
        let catalog = builtin_catalog().unwrap();
        let roots =
            FrozenRoots::from_build_profile(&catalog, &[WATER_FEATURE_ID], &[LIT_FEATURE_ID])
                .unwrap();
        DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap()
    }

    fn systems_binding(declaration: &'static str, group: &'static str) -> ShadingPassBinding {
        ShadingPassBinding {
            declaration,
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group,
                systems: vec![Box::new(IntoSystem::into_system(
                    move |mut trace: ResMut<ExecutionTrace>| trace.0.push(declaration),
                ))],
            },
        }
    }

    fn member_binding(
        declaration: &'static str,
        group: &'static str,
        owner: &'static str,
    ) -> ShadingPassBinding {
        ShadingPassBinding {
            declaration,
            requires: &[],
            executor: ShadingExecutorBinding::GroupMember { group, owner },
        }
    }

    fn external_binding(declaration: &'static str) -> ShadingPassBinding {
        ShadingPassBinding {
            declaration,
            requires: &[],
            executor: ShadingExecutorBinding::ExternalSchedule {
                site: ExternalExecutionSite::WaterSurface,
            },
        }
    }

    fn independent_bindings() -> Vec<ShadingPassBinding> {
        ["first", "culled", "last"]
            .into_iter()
            .map(|declaration| systems_binding(declaration, declaration))
            .collect()
    }

    fn grouped_bindings() -> Vec<ShadingPassBinding> {
        vec![
            systems_binding("first", "group"),
            member_binding("culled", "group", "first"),
            member_binding("last", "group", "first"),
        ]
    }

    /// A coherent compiler-table fixture, not a modified real GPU plan with
    /// stale barrier/attachment indices. No backend resources are executed.
    fn mock_compiled(pass_count: usize, order: &[usize]) -> ExecutionPlan {
        let mut alive = vec![false; pass_count];
        let mut position_by_pass = vec![None; pass_count];
        for (position, &index) in order.iter().enumerate() {
            assert!(!alive[index]);
            alive[index] = true;
            position_by_pass[index] = Some(position);
        }
        ExecutionPlan {
            declared_pass_count: pass_count,
            order: order.to_vec(),
            alive,
            position_by_pass,
            barriers: vec![vec![]; order.len()],
            attachments: vec![None; order.len()],
            ..Default::default()
        }
    }

    fn independent_graph(names: &[&'static str]) -> GpuFrameGraphBuilder {
        let mut graph = GpuFrameGraphBuilder::default();
        for &name in names {
            graph.add_pass(PassDescriptor {
                name: name.into(),
                queue: QueueClass::Compute,
                accesses: vec![],
                depends_on: vec![],
            });
        }
        graph
    }

    fn admit(
        graph: &GpuFrameGraphBuilder,
        compiled: ExecutionPlan,
        bindings: Vec<ShadingPassBinding>,
    ) -> Result<ShadingExecutionPlan, String> {
        ShadingExecutionPlan::new(
            &test_fabric(),
            &ShadingExternalExecutors::default(),
            graph,
            compiled,
            bindings,
        )
    }

    fn assert_rejected(
        graph: &GpuFrameGraphBuilder,
        compiled: ExecutionPlan,
        bindings: Vec<ShadingPassBinding>,
        expected: &str,
    ) {
        let error = admit(graph, compiled, bindings)
            .err()
            .expect("invalid execution plan must be rejected");
        assert!(
            error.contains(expected),
            "{error:?} should contain {expected:?}"
        );
    }

    const DRIVER_DECLARATIONS: &[&str] = &[
        "visibility_raster",
        "material_classification_count",
        "material_classification_prefix",
        "material_classification_scatter",
        "texel_space_shade_cache",
        "shading_resolve",
        "water_surface_raster",
        "response_composition",
        "post_exposure",
        "post_bloom",
        "post_tonemap",
        "post_upscale",
    ];

    fn driver_bindings(names: &[&'static str]) -> Vec<ShadingPassBinding> {
        names
            .iter()
            .map(|&name| {
                if name == "water_surface_raster" {
                    external_binding(name)
                } else {
                    systems_binding(name, name)
                }
            })
            .collect()
    }

    fn admit_with_water(
        graph: &GpuFrameGraphBuilder,
        compiled: ExecutionPlan,
        bindings: Vec<ShadingPassBinding>,
    ) -> Result<ShadingExecutionPlan, String> {
        let mut external = ShadingExternalExecutors::default();
        external
            .register(ExternalExecutionSite::WaterSurface)
            .unwrap();
        ShadingExecutionPlan::new(&test_fabric(), &external, graph, compiled, bindings)
    }

    fn driver_fixture(order: &[usize]) -> ShadingExecutionPlan {
        let graph = independent_graph(DRIVER_DECLARATIONS);
        let mut bindings = driver_bindings(DRIVER_DECLARATIONS);
        if let ShadingExecutorBinding::Systems { systems, .. } = &mut bindings[1].executor {
            for &name in &DRIVER_DECLARATIONS[2..4] {
                systems.push(Box::new(IntoSystem::into_system(
                    move |mut trace: ResMut<ExecutionTrace>| trace.0.push(name),
                )));
            }
        }
        for index in 2..4 {
            bindings[index] = member_binding(
                DRIVER_DECLARATIONS[index],
                "material_classification_count",
                "material_classification_count",
            );
        }
        admit_with_water(
            &graph,
            mock_compiled(DRIVER_DECLARATIONS.len(), order),
            bindings,
        )
        .unwrap()
    }

    #[derive(Clone, Debug, Eq, Hash, PartialEq, SystemSet)]
    enum TestCore3dSystems {
        MainPass,
        WaterDraw,
    }

    /// CPU-only schedule fixture: the marker represents the external boundary,
    /// not a GPU recorder or proof that the real water draw submitted commands.
    fn trace_water_boundary(mut trace: ResMut<ExecutionTrace>) {
        trace.0.push("water_surface_raster");
    }

    fn driver_schedule() -> Schedule {
        let mut schedule = Schedule::default();
        schedule.add_systems((
            run_shading_pre_resolve_driver.before(TestCore3dSystems::MainPass),
            run_shading_resolve_driver
                .after(run_shading_pre_resolve_driver)
                .before(TestCore3dSystems::MainPass),
            trace_water_boundary
                .in_set(TestCore3dSystems::WaterDraw)
                .after(TestCore3dSystems::MainPass),
            run_shading_response_driver
                .after(run_shading_resolve_driver)
                .after(TestCore3dSystems::MainPass)
                .after(TestCore3dSystems::WaterDraw)
                .before(run_shading_post_driver),
            run_shading_post_driver.after(run_shading_response_driver),
        ));
        schedule
    }

    #[test]
    fn production_compiler_order_fits_existing_driver_and_external_boundaries() {
        let graph = super::super::graph::shading_frame_graph();
        let names = graph
            .passes()
            .iter()
            .map(|pass| pass.name.as_ref())
            .collect::<Vec<_>>();
        let bindings = DRIVER_DECLARATIONS[..8]
            .iter()
            .copied()
            .chain(
                super::super::post_graph::POST_PROCESS_DECLARATIONS
                    .iter()
                    .map(|pass| pass.name),
            )
            .collect::<Vec<_>>();
        assert_eq!(names, bindings);
        let compiled = graph.compile().unwrap();
        let plan = admit_with_water(&graph, compiled.clone(), driver_bindings(&bindings)).unwrap();
        assert_eq!(plan.compiled(), &compiled);
        let ranges = plan.driver_ranges.as_ref().unwrap();
        assert_eq!(ranges[ShadingExecutionPhase::PreResolve as usize], 0..5);
        assert_eq!(ranges[ShadingExecutionPhase::Resolve as usize], 5..6);
        assert_eq!(ranges[ShadingExecutionPhase::WaterSurface as usize], 6..7);
        assert_eq!(ranges[ShadingExecutionPhase::Response as usize], 7..8);
        assert_eq!(ranges[ShadingExecutionPhase::Post as usize], 8..names.len());
    }

    #[test]
    fn four_drivers_reset_groups_and_preserve_the_external_boundary_across_frames() {
        let order = (0..DRIVER_DECLARATIONS.len()).collect::<Vec<_>>();
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        world.insert_resource(driver_fixture(&order));
        let mut schedule = driver_schedule();
        for _ in 0..2 {
            schedule.run(&mut world);
        }
        assert_eq!(
            world.resource::<ExecutionTrace>().0,
            DRIVER_DECLARATIONS.repeat(2)
        );
    }

    #[test]
    fn culling_old_range_endpoints_does_not_skip_surviving_phase_work() {
        let order = (0..DRIVER_DECLARATIONS.len())
            .filter(|index| ![4, 8, 11].contains(index))
            .collect::<Vec<_>>();
        let expected = order
            .iter()
            .map(|&index| DRIVER_DECLARATIONS[index])
            .collect::<Vec<_>>();
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        world.insert_resource(driver_fixture(&order));
        let mut schedule = driver_schedule();
        for _ in 0..2 {
            schedule.run(&mut world);
        }
        assert_eq!(world.resource::<ExecutionTrace>().0, expected.repeat(2));
    }

    #[test]
    fn drivers_preserve_compiler_order_within_each_phase() {
        let names = [
            "visibility_raster",
            "texel_space_shade_cache",
            "shading_resolve",
            "response_composition",
            "post_exposure",
            "post_bloom",
            "post_upscale",
        ];
        let graph = independent_graph(&names);
        let order = [1, 0, 2, 3, 6, 5, 4];
        let expected = order.iter().map(|&index| names[index]).collect::<Vec<_>>();
        let plan = admit(
            &graph,
            mock_compiled(names.len(), &order),
            driver_bindings(&names),
        )
        .unwrap();
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        world.insert_resource(plan);
        run_shading_pre_resolve_driver(&mut world);
        run_shading_resolve_driver(&mut world);
        run_shading_response_driver(&mut world);
        run_shading_post_driver(&mut world);
        assert_eq!(world.resource::<ExecutionTrace>().0, expected);
    }

    #[test]
    fn empty_pre_phase_still_resets_later_groups_every_frame() {
        let order = (5..DRIVER_DECLARATIONS.len()).collect::<Vec<_>>();
        let mut plan = driver_fixture(&order);
        // A previous frame may have executed every later driver.
        for &name in &DRIVER_DECLARATIONS[5..] {
            plan.executed_groups.insert(name);
        }
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        world.insert_resource(plan);
        let mut schedule = driver_schedule();
        for _ in 0..2 {
            schedule.run(&mut world);
        }
        assert_eq!(
            world.resource::<ExecutionTrace>().0,
            DRIVER_DECLARATIONS[5..].repeat(2)
        );
    }

    #[test]
    fn culled_resolve_response_and_whole_phases_are_noops() {
        for order in [vec![0, 1, 2, 3, 4, 9, 10], vec![]] {
            let expected = order
                .iter()
                .map(|&index| DRIVER_DECLARATIONS[index])
                .collect::<Vec<_>>();
            let mut world = World::new();
            world.init_resource::<ExecutionTrace>();
            world.insert_resource(driver_fixture(&order));
            for _ in 0..2 {
                run_shading_pre_resolve_driver(&mut world);
                run_shading_resolve_driver(&mut world);
                run_shading_response_driver(&mut world);
                run_shading_post_driver(&mut world);
            }
            assert_eq!(world.resource::<ExecutionTrace>().0, expected.repeat(2));
        }
    }

    #[test]
    fn group_cannot_execute_work_from_a_later_driver_even_when_contiguous_or_culled() {
        for names in [
            ["texel_space_shade_cache", "shading_resolve"],
            ["shading_resolve", "response_composition"],
            ["response_composition", "post_exposure"],
        ] {
            let graph = independent_graph(&names);
            for order in [&[0, 1][..], &[][..]] {
                assert_rejected(
                    &graph,
                    mock_compiled(2, order),
                    vec![
                        systems_binding(names[0], "cross_phase"),
                        member_binding(names[1], "cross_phase", names[0]),
                    ],
                    "crosses driver phases",
                );
            }
        }
    }

    #[test]
    fn compiler_order_cannot_move_work_across_fixed_schedule_boundaries() {
        let names = [
            "texel_space_shade_cache",
            "shading_resolve",
            "water_surface_raster",
            "response_composition",
            "post_bloom",
        ];
        let graph = independent_graph(&names);
        for order in [
            [1, 0, 2, 3, 4],
            [0, 2, 1, 3, 4],
            [0, 1, 3, 2, 4],
            [0, 1, 2, 4, 3],
            [3, 0, 1, 2, 4],
        ] {
            let error = admit_with_water(&graph, mock_compiled(5, &order), driver_bindings(&names))
                .err()
                .expect("compiler order incompatible with Core3d must be rejected");
            assert!(error.contains("driver phases backwards"), "{error}");
        }
    }

    #[test]
    fn water_site_cannot_stand_in_for_another_declaration_or_use_driver_systems() {
        for name in ["shading_resolve", "response_composition", "post_bloom"] {
            let graph = independent_graph(&[name]);
            for order in [&[0][..], &[][..]] {
                let error = admit_with_water(
                    &graph,
                    mock_compiled(1, order),
                    vec![external_binding(name)],
                )
                .err()
                .expect("external site must match the scheduled declaration");
                assert!(error.contains("must bind water_surface_raster"), "{error}");
            }
        }
        let graph = independent_graph(&["water_surface_raster"]);
        for order in [&[0][..], &[][..]] {
            assert_rejected(
                &graph,
                mock_compiled(1, order),
                vec![systems_binding("water_surface_raster", "fake_water")],
                "must use external executor WaterSurface",
            );
        }
    }

    #[test]
    fn production_driver_plan_cannot_silently_omit_an_unassigned_declaration() {
        let names = ["shading_resolve", "unassigned"];
        let graph = independent_graph(&names);
        for order in [&[0, 1][..], &[0][..]] {
            assert_rejected(
                &graph,
                mock_compiled(2, order),
                driver_bindings(&names),
                "unassigned has no driver phase",
            );
        }
    }

    #[test]
    fn plan_survival_and_position_are_compiled_metadata() {
        let graph = test_graph();
        let compiled = mock_compiled(3, &[0, 2]);
        let plan = admit(&graph, compiled, independent_bindings()).unwrap();
        assert!(plan.is_alive("first"));
        assert!(!plan.is_alive("culled"));
        assert!(plan.is_alive("last"));
        assert_eq!(plan.position("first"), Some(0));
        assert_eq!(plan.position("culled"), None);
        assert_eq!(plan.position("last"), Some(1));
    }

    #[test]
    fn plan_resource_accesses_preserve_declaration_mapping() {
        let graph = test_graph();
        let compiled = graph.compile().unwrap();
        let plan = admit(&graph, compiled, independent_bindings()).unwrap();
        assert_eq!(
            plan.pass("first").unwrap().accesses,
            vec![
                ShadingResourceAccess {
                    resource: "input".into(),
                    kind: AccessKind::SampledRead,
                },
                ShadingResourceAccess {
                    resource: "intermediate".into(),
                    kind: AccessKind::StorageWrite,
                },
            ]
        );
        assert_eq!(
            plan.pass("last").unwrap().accesses,
            vec![
                ShadingResourceAccess {
                    resource: "intermediate".into(),
                    kind: AccessKind::StorageRead,
                },
                ShadingResourceAccess {
                    resource: "output".into(),
                    kind: AccessKind::StorageWrite,
                },
            ]
        );
    }

    #[test]
    fn plan_order_uses_survivor_order_and_not_declaration_iteration() {
        let graph = independent_graph(&["first", "culled", "last"]);
        let compiled = mock_compiled(3, &[2, 0]);
        let mut plan = admit(&graph, compiled, independent_bindings()).unwrap();
        assert_eq!(plan.position("last"), Some(0));
        assert_eq!(plan.position("first"), Some(1));
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        plan.run_position_range(&mut world, 0..2).unwrap();
        assert_eq!(world.resource::<ExecutionTrace>().0, ["last", "first"]);
    }

    #[test]
    fn position_queries_are_indexed_without_scanning_order() {
        let graph = test_graph();
        let compiled = mock_compiled(3, &[0, 2]);
        let plan = admit(&graph, compiled, independent_bindings()).unwrap();
        for _ in 0..10_000 {
            assert_eq!(plan.position("last"), Some(1));
            assert!(!plan.is_alive("culled"));
        }
        assert_eq!(plan.compiled().position_by_pass.len(), graph.passes().len());
    }

    #[test]
    fn required_roots_and_variants_use_the_unchanged_device_selection() {
        let fabric = test_fabric();
        let snapshot = fabric.selection().clone();
        let graph = test_graph();
        let mut bindings = independent_bindings();
        bindings[0].requires = &[
            SelectedVariantRequirement {
                feature: WATER_FEATURE_ID,
                variant: WATER_VARIANT_ID,
            },
            SelectedVariantRequirement {
                feature: LIT_FEATURE_ID,
                variant: LIT_VARIANT_ID,
            },
        ];
        ShadingExecutionPlan::new(
            &fabric,
            &ShadingExternalExecutors::default(),
            &graph,
            graph.compile().unwrap(),
            bindings,
        )
        .unwrap();
        assert_eq!(fabric.selection(), &snapshot);
    }

    #[test]
    fn missing_root_and_wrong_variant_are_rejected_even_for_culled_passes() {
        let graph = test_graph();
        let requirements: &[&'static [SelectedVariantRequirement]] = &[
            &[SelectedVariantRequirement {
                feature: NPR_FEATURE_ID,
                variant: VariantId(0),
            }],
            &[SelectedVariantRequirement {
                feature: WATER_FEATURE_ID,
                variant: VariantId(u32::MAX),
            }],
        ];
        for (requires, expected) in requirements
            .iter()
            .zip(["frozen root", "frozen selection chose"])
        {
            let mut bindings = independent_bindings();
            bindings[1].requires = requires;
            assert_rejected(&graph, mock_compiled(3, &[0, 2]), bindings, expected);
        }
    }

    #[test]
    fn selected_dependency_service_is_not_a_root_permission() {
        let fabric = test_fabric();
        assert!(fabric
            .selection()
            .feature(SPRAY_FEEDBACK_FEATURE_ID)
            .is_some());
        let snapshot = fabric.selection().clone();
        let graph = test_graph();
        let mut bindings = independent_bindings();
        bindings[0].requires = &[SelectedVariantRequirement {
            feature: SPRAY_FEEDBACK_FEATURE_ID,
            variant: VariantId(0),
        }];
        let error = ShadingExecutionPlan::new(
            &fabric,
            &ShadingExternalExecutors::default(),
            &graph,
            graph.compile().unwrap(),
            bindings,
        )
        .err()
        .expect("selected service is not an admitted root");
        assert!(error.contains("OutsideFrozenRoots"), "{error}");
        assert_eq!(fabric.selection(), &snapshot);
    }

    #[test]
    fn systems_must_not_be_empty_including_when_culled() {
        let graph = test_graph();
        for index in [0, 1] {
            let mut bindings = independent_bindings();
            bindings[index].executor = ShadingExecutorBinding::Systems {
                group: "empty",
                systems: vec![],
            };
            assert_rejected(&graph, mock_compiled(3, &[0, 2]), bindings, "empty Systems");
        }
    }

    #[test]
    fn a_group_has_exactly_one_systems_owner() {
        let graph = test_graph();
        let mut bindings = independent_bindings();
        bindings[2] = systems_binding("last", "first");
        assert_rejected(
            &graph,
            graph.compile().unwrap(),
            bindings,
            "multiple owners",
        );

        let mut bindings = independent_bindings();
        bindings[0] = member_binding("first", "missing", "last");
        assert_rejected(
            &graph,
            graph.compile().unwrap(),
            bindings,
            "no Systems owner",
        );
    }

    #[test]
    fn group_member_must_name_the_actual_owner_of_its_group() {
        let graph = test_graph();
        for owner in ["missing", "culled", "last"] {
            let mut bindings = grouped_bindings();
            bindings[2] = member_binding("last", "group", owner);
            assert_rejected(&graph, graph.compile().unwrap(), bindings, "names owner");
        }
        let mut bindings = independent_bindings();
        bindings[1] = member_binding("culled", "last", "first");
        assert_rejected(&graph, graph.compile().unwrap(), bindings, "names owner");
    }

    #[test]
    fn group_cannot_partially_survive_culling() {
        let graph = test_graph();
        for order in [&[0, 2][..], &[1, 2][..]] {
            assert_rejected(
                &graph,
                mock_compiled(3, order),
                grouped_bindings(),
                "mixed liveness",
            );
        }
        assert!(admit(&graph, mock_compiled(3, &[]), grouped_bindings()).is_ok());
    }

    #[test]
    fn surviving_group_must_be_contiguous_and_owner_first() {
        let graph = independent_graph(&["first", "culled", "last"]);
        let bindings = vec![
            systems_binding("first", "group"),
            systems_binding("culled", "unrelated"),
            member_binding("last", "group", "first"),
        ];
        assert_rejected(
            &graph,
            mock_compiled(3, &[0, 1, 2]),
            bindings,
            "not contiguous",
        );
        assert_rejected(
            &graph,
            mock_compiled(3, &[1, 0, 2]),
            grouped_bindings(),
            "must be first",
        );
    }

    #[test]
    fn contiguous_group_executes_all_owner_systems_once_per_frame() {
        let graph = test_graph();
        let mut bindings = grouped_bindings();
        if let ShadingExecutorBinding::Systems { systems, .. } = &mut bindings[0].executor {
            systems.push(Box::new(IntoSystem::into_system(
                |mut trace: ResMut<ExecutionTrace>| trace.0.push("second system"),
            )));
        }
        let mut plan = admit(&graph, graph.compile().unwrap(), bindings).unwrap();
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        for _ in 0..2 {
            plan.begin_frame();
            plan.run_named_pass(&mut world, "first").unwrap();
            plan.run_named_range(&mut world, "first", "last").unwrap();
        }
        assert_eq!(
            world.resource::<ExecutionTrace>().0,
            ["first", "second system", "first", "second system"]
        );
    }

    #[test]
    fn member_range_cannot_silently_skip_its_owner() {
        let graph = test_graph();
        let mut plan = admit(&graph, graph.compile().unwrap(), grouped_bindings()).unwrap();
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        let error = plan.run_named_pass(&mut world, "last").unwrap_err();
        assert!(error.contains("before owner first"), "{error}");
        assert!(world.resource::<ExecutionTrace>().0.is_empty());
        plan.run_named_pass(&mut world, "first").unwrap();
        plan.run_named_pass(&mut world, "last").unwrap();
        plan.begin_frame();
        assert!(plan.run_named_pass(&mut world, "last").is_err());
    }

    #[test]
    fn external_site_registration_is_unique_and_does_not_fake_execution() {
        let names = [
            "visibility_raster",
            "water_surface_raster",
            "response_composition",
        ];
        let graph = independent_graph(&names);
        let mut external = ShadingExternalExecutors::default();
        assert!(!external.contains(ExternalExecutionSite::WaterSurface));
        external
            .register(ExternalExecutionSite::WaterSurface)
            .unwrap();
        assert!(external.contains(ExternalExecutionSite::WaterSurface));
        assert!(external
            .register(ExternalExecutionSite::WaterSurface)
            .is_err());
        assert!(external.contains(ExternalExecutionSite::WaterSurface));
        let bindings = driver_bindings(&names);
        let mut plan = ShadingExecutionPlan::new(
            &test_fabric(),
            &external,
            &graph,
            graph.compile().unwrap(),
            bindings,
        )
        .unwrap();
        let mut world = World::new();
        world.init_resource::<ExecutionTrace>();
        plan.run_named_pass(&mut world, "water_surface_raster")
            .unwrap();
        assert!(world.resource::<ExecutionTrace>().0.is_empty());
        assert!(plan.executed_groups.is_empty());
        plan.run_position_range(&mut world, 0..3).unwrap();
        assert_eq!(
            world.resource::<ExecutionTrace>().0,
            ["visibility_raster", "response_composition"]
        );
    }

    #[test]
    fn unregistered_external_site_is_rejected_even_when_culled() {
        let graph = test_graph();
        for order in [&[0, 1, 2][..], &[0, 2][..]] {
            let mut bindings = independent_bindings();
            bindings[1] = external_binding("culled");
            assert_rejected(
                &graph,
                mock_compiled(3, order),
                bindings,
                "unregistered external executor",
            );
        }
    }

    #[test]
    fn inconsistent_compiler_indices_and_positions_are_rejected() {
        let graph = test_graph();
        let cases: &[(fn(&mut ExecutionPlan), &str)] = &[
            (|plan| plan.declared_pass_count = 2, "declares"),
            (
                |plan| {
                    plan.alive.pop();
                },
                "table lengths",
            ),
            (
                |plan| {
                    plan.position_by_pass.pop();
                },
                "table lengths",
            ),
            (
                |plan| {
                    plan.barriers.pop();
                },
                "ordered metadata",
            ),
            (
                |plan| {
                    plan.attachments.pop();
                },
                "ordered metadata",
            ),
            (|plan| plan.order[1] = 3, "unknown pass"),
            (|plan| plan.order[1] = 0, "repeats pass"),
            (
                |plan| plan.alive[0] = false,
                "inconsistent liveness/position",
            ),
            (
                |plan| plan.position_by_pass[0] = None,
                "inconsistent liveness/position",
            ),
            (
                |plan| plan.position_by_pass[0] = Some(1),
                "inconsistent liveness/position",
            ),
            (
                |plan| plan.position_by_pass[0] = Some(9),
                "inconsistent liveness/position",
            ),
            (|plan| plan.alive[1] = true, "disagrees with survivor order"),
            (
                |plan| plan.position_by_pass[1] = Some(0),
                "disagrees with survivor order",
            ),
        ];
        for &(mutate, expected) in cases {
            let mut compiled = mock_compiled(3, &[0, 2]);
            mutate(&mut compiled);
            assert_rejected(&graph, compiled, independent_bindings(), expected);
        }
    }

    #[test]
    fn declaration_ledger_must_be_a_bijection() {
        let graph = test_graph();
        let mut bindings = independent_bindings();
        bindings.push(systems_binding("first", "duplicate"));
        assert_rejected(
            &graph,
            graph.compile().unwrap(),
            bindings,
            "duplicate declaration",
        );
        let mut bindings = independent_bindings();
        bindings.pop();
        assert_rejected(
            &graph,
            graph.compile().unwrap(),
            bindings,
            "missing declaration last",
        );
        let mut bindings = independent_bindings();
        bindings.push(systems_binding("unknown", "unknown"));
        assert_rejected(
            &graph,
            graph.compile().unwrap(),
            bindings,
            "unknown declaration",
        );

        let graph = independent_graph(&["first", "first", "last"]);
        assert_rejected(
            &graph,
            mock_compiled(3, &[0, 1, 2]),
            independent_bindings(),
            "duplicate pass first",
        );
    }

    #[test]
    fn unknown_resource_mapping_is_rejected() {
        let mut graph = GpuFrameGraphBuilder::default();
        graph.add_pass(PassDescriptor {
            name: "first".into(),
            queue: QueueClass::Compute,
            accesses: vec![prism_render_graph::frame_graph::ResourceAccess {
                resource: prism_render_graph::frame_graph::ResourceId(0),
                kind: AccessKind::StorageRead,
            }],
            depends_on: vec![],
        });
        assert_rejected(
            &graph,
            mock_compiled(1, &[0]),
            vec![systems_binding("first", "first")],
            "unknown resource 0",
        );
    }
}
