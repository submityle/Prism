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
use prism_render_graph::{
    frame_graph::{AccessKind, GpuFrameGraphBuilder},
    ExecutionPlan,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShadingResourceAccess {
    pub(crate) resource: String,
    pub(crate) kind: AccessKind,
}

pub(crate) struct ShadingPassBinding {
    pub(crate) declaration: &'static str,
    pub(crate) executor_group: &'static str,
    pub(crate) executors: Vec<BoxedSystem>,
}

pub(crate) struct ShadingPassRuntime {
    pub(crate) declaration: String,
    pub(crate) declaration_index: usize,
    pub(crate) position: Option<usize>,
    pub(crate) accesses: Vec<ShadingResourceAccess>,
    pub(crate) executor_group: &'static str,
    executors: Vec<BoxedSystem>,
    initialized: bool,
}

#[derive(Resource)]
pub(crate) struct ShadingExecutionPlan {
    compiled: ExecutionPlan,
    pass_index_by_name: HashMap<String, usize>,
    passes: Vec<ShadingPassRuntime>,
    executed_groups: HashSet<&'static str>,
}

impl ShadingExecutionPlan {
    pub(crate) fn new(
        graph: &GpuFrameGraphBuilder,
        compiled: ExecutionPlan,
        bindings: Vec<ShadingPassBinding>,
    ) -> Result<Self, String> {
        if compiled.declared_pass_count != graph.passes().len() {
            return Err(format!(
                "compiled shading plan declares {} passes but graph has {}",
                compiled.declared_pass_count,
                graph.passes().len()
            ));
        }

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
                declaration_index: index,
                position: compiled.position(index),
                accesses,
                executor_group: binding.executor_group,
                executors: binding.executors,
                initialized: false,
            });
        }

        if !binding_by_name.is_empty() {
            return Err("shading execution ledger contains an unknown declaration".into());
        }

        Ok(Self {
            compiled,
            pass_index_by_name,
            passes,
            executed_groups: HashSet::new(),
        })
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
            if runtime.executors.is_empty() || self.executed_groups.contains(runtime.executor_group)
            {
                continue;
            }
            if !runtime.initialized {
                for executor in &mut runtime.executors {
                    executor.initialize(world);
                }
                runtime.initialized = true;
            }
            for executor in &mut runtime.executors {
                executor.run((), world).map_err(|error| {
                    format!("shading executor {} failed: {error}", executor.name())
                })?;
            }
            self.executed_groups.insert(runtime.executor_group);
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
        plan.begin_frame();
        let Some(last) = plan.position("texel_space_shade_cache") else {
            return;
        };
        if let Err(error) = plan.run_position_range(world, 0..last + 1) {
            panic!("shading pre-resolve execution failed: {error}");
        }
    });
}

pub(crate) fn run_shading_resolve_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        if let Err(error) = plan.run_named_pass(world, "shading_resolve") {
            panic!("shading resolve execution failed: {error}");
        }
    });
}

pub(crate) fn run_shading_response_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        if let Err(error) = plan.run_named_pass(world, "response_composition") {
            panic!("shading response execution failed: {error}");
        }
    });
}

pub(crate) fn run_shading_post_driver(world: &mut World) {
    ShadingExecutionPlan::with_resource(world, |world, mut plan| {
        let (Some(first), Some(last)) = (
            plan.position("post_exposure"),
            plan.position("post_upscale"),
        ) else {
            return;
        };
        if let Err(error) = plan.run_position_range(world, first..last + 1) {
            panic!("shading post execution failed: {error}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn empty_bindings() -> Vec<ShadingPassBinding> {
        ["first", "culled", "last"]
            .into_iter()
            .map(|declaration| ShadingPassBinding {
                declaration,
                executor_group: declaration,
                executors: Vec::new(),
            })
            .collect()
    }

    fn compiled_with_culled_middle_pass(graph: &GpuFrameGraphBuilder) -> ExecutionPlan {
        let mut compiled = graph.compile().unwrap();
        compiled.order = vec![0, 2];
        compiled.alive = vec![true, false, true];
        compiled.position_by_pass = vec![Some(0), None, Some(1)];
        compiled
    }

    #[test]
    fn plan_survival_and_position_are_compiled_metadata() {
        let graph = test_graph();
        let compiled = compiled_with_culled_middle_pass(&graph);
        let plan = ShadingExecutionPlan::new(&graph, compiled, empty_bindings()).unwrap();
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
        let plan = ShadingExecutionPlan::new(&graph, compiled, empty_bindings()).unwrap();
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
        let graph = test_graph();
        let mut compiled = graph.compile().unwrap();
        compiled.order.reverse();
        compiled.position_by_pass = vec![Some(1), None, Some(0)];
        let plan = ShadingExecutionPlan::new(&graph, compiled, empty_bindings()).unwrap();
        assert_eq!(plan.position("last"), Some(0));
        assert_eq!(plan.position("first"), Some(1));
    }

    #[test]
    fn position_queries_are_indexed_without_scanning_order() {
        let graph = test_graph();
        let compiled = compiled_with_culled_middle_pass(&graph);
        let plan = ShadingExecutionPlan::new(&graph, compiled, empty_bindings()).unwrap();
        for _ in 0..10_000 {
            assert_eq!(plan.position("last"), Some(1));
            assert!(!plan.is_alive("culled"));
        }
        assert_eq!(plan.compiled().position_by_pass.len(), graph.passes().len());
    }
}
