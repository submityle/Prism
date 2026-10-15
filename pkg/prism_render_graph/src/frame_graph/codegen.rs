//! Declarative pass-skeleton lowering for the backend-neutral frame graph.
//!
//! A pass declaration owns the four pieces of plumbing that used to be copied
//! into every post-processing module: ABI identity, bind-group shape, pipeline
//! entry metadata, and logical resource accesses. [`declare_passes`] resolves
//! names once at graph construction time and emits ordinary `PassDescriptor`
//! values, so the compiler still has one execution path and the generated
//! surface remains backend-neutral.

use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};

use super::{
    AccessKind, GpuFrameGraphBuilder, PassDescriptor, PassId, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceId, ResourceLifetime,
};

/// ABI identity emitted by the pass declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SkeletonAbi {
    /// Version of the host/shader parameter contract.
    pub version: u32,
    /// Stable contract digest used to invalidate pipeline data.
    pub hash: u64,
}

/// Bind-group shape emitted by the pass declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindGroupContract {
    /// Number of bind groups used by the pass.
    pub group_count: u8,
    /// Number of bindings in each group.
    pub bindings_per_group: &'static [u8],
}

/// Pipeline metadata emitted by the pass declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineContract {
    /// Shader entry point selected by the generated pipeline row.
    pub entry_point: &'static str,
    /// Compute workgroup size, or `[0, 0, 0]` for raster declarations.
    pub workgroup_size: [u32; 3],
}

/// A named resource declared by a pass skeleton.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceDeclaration {
    /// Stable logical resource name.
    pub name: &'static str,
    /// Ownership/lifetime policy.
    pub lifetime: ResourceLifetime,
    /// Logical byte footprint used by the topology compiler.
    pub size: u64,
    /// Required alignment for transient placement.
    pub alignment: u64,
}

/// A named resource access emitted by a pass skeleton.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccessDeclaration {
    /// Resource name resolved against the graph's declaration table.
    pub resource: &'static str,
    /// Logical access mode.
    pub kind: AccessKind,
}

/// Complete declarative description of one pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PassSkeletonDeclaration {
    /// Stable diagnostic and generated-table name.
    pub name: &'static str,
    /// Logical queue used by the frame-graph scheduler.
    pub queue: QueueClass,
    /// ABI contract for the host/shader interface.
    pub abi: SkeletonAbi,
    /// Bind-group contract.
    pub bind_groups: BindGroupContract,
    /// Pipeline contract.
    pub pipeline: PipelineContract,
    /// Resources introduced by this declaration.
    pub resources: &'static [ResourceDeclaration],
    /// Accesses introduced by this declaration.
    pub accesses: &'static [AccessDeclaration],
    /// Explicit prerequisites by stable pass name.
    pub depends_on: &'static [&'static str],
}

/// The generated result retained by callers that need the contracts after
/// graph construction (for diagnostics, pipeline specialization, or tests).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedPass {
    /// Original pass id in the graph declaration table.
    pub id: PassId,
    /// The four generated contracts.
    pub declaration: PassSkeletonDeclaration,
}

/// Errors found before a declaration mutates the graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodegenError {
    /// The generated table exceeds the graph's checked admission limits.
    Graph(super::CompileError),
    /// A pass name was repeated.
    DuplicatePass(String),
    /// A resource name was repeated with incompatible ownership/shape.
    ConflictingResource(String),
    /// A dependency named a pass that does not exist in this declaration set.
    UnknownDependency {
        /// Pass that declared the invalid edge.
        pass: String,
        /// Missing prerequisite name.
        dependency: String,
    },
    /// An access named a resource that is not present in the graph or set.
    UnknownResource {
        /// Pass that declared the invalid access.
        pass: String,
        /// Missing resource name.
        resource: String,
    },
    /// The bind-group count does not agree with its shape array.
    BindGroupShape {
        /// Pass whose bind-group declaration is malformed.
        pass: String,
        /// Declared group count.
        expected: usize,
        /// Number of supplied group-shape entries.
        actual: usize,
    },
}

/// Lowers a checked declaration set into the shared frame-graph builder.
///
/// Existing graph resources are accepted as inputs, which lets a generated
/// post chain consume the resident HDR target created by the shading graph.
/// Dependencies may name any pass in the set, regardless of declaration order.
/// Admission is transactional: validation or capacity errors leave the graph
/// unchanged. Cycles are diagnosed by the shared compiler at compile time.
pub fn declare_passes(
    graph: &mut GpuFrameGraphBuilder,
    declarations: &[PassSkeletonDeclaration],
) -> Result<Vec<GeneratedPass>, CodegenError> {
    crate::capacity::GraphLimits::check_count(
        "passes",
        graph.passes().len().saturating_add(declarations.len()),
        u32::MAX as usize,
    )
    .map_err(CodegenError::Graph)?;
    let base_pass = graph.passes().len() as u32;
    let mut resources = BTreeMap::<&'static str, ResourceId>::new();
    let mut pass_names = BTreeMap::<String, PassId>::new();
    for (index, pass) in graph.passes().iter().enumerate() {
        pass_names.insert(pass.name.to_string(), PassId(index as u32));
    }

    for (index, declaration) in declarations.iter().enumerate() {
        if pass_names
            .insert(
                declaration.name.to_string(),
                PassId(base_pass + index as u32),
            )
            .is_some()
        {
            return Err(CodegenError::DuplicatePass(declaration.name.into()));
        }
        if declaration.bind_groups.group_count as usize
            != declaration.bind_groups.bindings_per_group.len()
        {
            return Err(CodegenError::BindGroupShape {
                pass: declaration.name.into(),
                expected: declaration.bind_groups.group_count as usize,
                actual: declaration.bind_groups.bindings_per_group.len(),
            });
        }
    }
    // Resolve against the entire table, not just the prefix visited so far.
    for declaration in declarations {
        for dependency in declaration.depends_on {
            if !pass_names.contains_key(*dependency) {
                return Err(CodegenError::UnknownDependency {
                    pass: declaration.name.into(),
                    dependency: (*dependency).into(),
                });
            }
        }
    }

    // Validate the complete resource table before mutating the builder. This
    // keeps a failed generated table atomic even when the error is discovered
    // in a later declaration.
    for declaration in declarations {
        for resource in declaration.resources {
            if declarations
                .iter()
                .flat_map(|candidate| candidate.resources.iter())
                .any(|candidate| {
                    candidate.name == resource.name
                        && (candidate.lifetime != resource.lifetime
                            || candidate.size != resource.size
                            || candidate.alignment != resource.alignment)
                })
            {
                return Err(CodegenError::ConflictingResource(resource.name.into()));
            }
            if let Some(existing) = graph
                .resources()
                .iter()
                .find(|descriptor| descriptor.name == resource.name)
            {
                if existing.lifetime != resource.lifetime
                    || existing.size != resource.size
                    || existing.alignment != resource.alignment
                {
                    return Err(CodegenError::ConflictingResource(resource.name.into()));
                }
            }
        }
        for access in declaration.accesses {
            let in_graph = graph
                .resources()
                .iter()
                .any(|resource| resource.name == access.resource);
            let in_declarations = declarations
                .iter()
                .flat_map(|candidate| candidate.resources.iter())
                .any(|resource| resource.name == access.resource);
            if !in_graph && !in_declarations {
                return Err(CodegenError::UnknownResource {
                    pass: declaration.name.into(),
                    resource: access.resource.into(),
                });
            }
        }
    }

    let mut staged = graph.clone();
    for declaration in declarations {
        for resource in declaration.resources {
            if let Some(existing) = resource_id(&staged, &resources, resource.name) {
                resources.insert(resource.name, existing);
            } else {
                let id = staged
                    .try_add_resource(ResourceDescriptor {
                        name: resource.name.into(),
                        lifetime: resource.lifetime,
                        size: resource.size,
                        alignment: resource.alignment,
                    })
                    .map_err(CodegenError::Graph)?;
                resources.insert(resource.name, id);
            }
        }
    }

    for declaration in declarations {
        let descriptor = PassDescriptor {
            name: declaration.name.into(),
            queue: declaration.queue,
            accesses: declaration
                .accesses
                .iter()
                .map(|access| ResourceAccess {
                    resource: resource_id(&staged, &resources, access.resource)
                        .expect("validated generated resource"),
                    kind: access.kind,
                })
                .collect(),
            depends_on: declaration
                .depends_on
                .iter()
                .map(|name| {
                    pass_names
                        .get(*name)
                        .copied()
                        .expect("validated generated dependency")
                })
                .collect(),
        };
        staged
            .try_add_pass(descriptor)
            .map_err(CodegenError::Graph)?;
    }

    *graph = staged;
    Ok(declarations
        .iter()
        .enumerate()
        .map(|(index, declaration)| GeneratedPass {
            id: PassId(base_pass + index as u32),
            declaration: *declaration,
        })
        .collect())
}

fn resource_id(
    graph: &GpuFrameGraphBuilder,
    generated: &BTreeMap<&'static str, ResourceId>,
    name: &'static str,
) -> Option<ResourceId> {
    generated.get(name).copied().or_else(|| {
        graph
            .resources()
            .iter()
            .position(|resource| resource.name == name)
            .map(|index| ResourceId(index as u32))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESOURCES: &[ResourceDeclaration] = &[ResourceDeclaration {
        name: "generated_params",
        lifetime: ResourceLifetime::Persistent,
        size: 64,
        alignment: 16,
    }];
    const ACCESSES: &[AccessDeclaration] = &[AccessDeclaration {
        resource: "generated_params",
        kind: AccessKind::StorageRead,
    }];
    const GROUPS: &[u8] = &[2];

    const SPECS: &[PassSkeletonDeclaration] = &[
        PassSkeletonDeclaration {
            name: "generated_a",
            queue: QueueClass::Compute,
            abi: SkeletonAbi {
                version: 1,
                hash: 7,
            },
            bind_groups: BindGroupContract {
                group_count: 1,
                bindings_per_group: GROUPS,
            },
            pipeline: PipelineContract {
                entry_point: "generated_a",
                workgroup_size: [8, 8, 1],
            },
            resources: RESOURCES,
            accesses: ACCESSES,
            depends_on: &[],
        },
        PassSkeletonDeclaration {
            name: "generated_b",
            queue: QueueClass::Compute,
            abi: SkeletonAbi {
                version: 1,
                hash: 8,
            },
            bind_groups: BindGroupContract {
                group_count: 1,
                bindings_per_group: GROUPS,
            },
            pipeline: PipelineContract {
                entry_point: "generated_b",
                workgroup_size: [8, 8, 1],
            },
            resources: RESOURCES,
            accesses: ACCESSES,
            depends_on: &["generated_a"],
        },
    ];

    #[test]
    fn declaration_codegen_is_deterministic_and_reuses_resources() {
        let mut graph = GpuFrameGraphBuilder::default();
        let generated = declare_passes(&mut graph, SPECS).unwrap();
        assert_eq!(generated.len(), 2);
        assert_eq!(graph.resources().len(), 1);
        assert_eq!(graph.passes().len(), 2);
        assert_eq!(graph.passes()[1].depends_on, [PassId(0)]);
        assert_eq!(generated[1].declaration.pipeline.entry_point, "generated_b");
    }

    #[test]
    fn declaration_codegen_rejects_shape_and_dependency_drift() {
        let mut invalid = SPECS[0];
        invalid.bind_groups = BindGroupContract {
            group_count: 2,
            bindings_per_group: GROUPS,
        };
        assert!(matches!(
            declare_passes(&mut GpuFrameGraphBuilder::default(), &[invalid]),
            Err(CodegenError::BindGroupShape { .. })
        ));

        invalid = SPECS[0];
        invalid.depends_on = &["missing"];
        assert!(matches!(
            declare_passes(&mut GpuFrameGraphBuilder::default(), &[invalid]),
            Err(CodegenError::UnknownDependency { .. })
        ));
    }

    #[test]
    fn forward_dependencies_lower_into_real_hazards_and_queue_waits() {
        let mut consumer = SPECS[1];
        consumer.depends_on = &["generated_a"];
        consumer.queue = QueueClass::Graphics;
        let mut producer = SPECS[0];
        producer.accesses = &[AccessDeclaration {
            resource: "generated_params",
            kind: AccessKind::StorageWrite,
        }];
        let mut graph = GpuFrameGraphBuilder::default();
        let generated = declare_passes(&mut graph, &[consumer, producer]).unwrap();
        assert_eq!(generated[0].id, PassId(0));
        assert_eq!(generated[1].id, PassId(1));
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.execution_order, [PassId(1), PassId(0)]);
        assert_eq!(compiled.resource_versions.len(), 1);
        assert_eq!(compiled.resource_versions[0].writer, PassId(1));
        assert_eq!(compiled.resource_versions[0].version, 1);
        assert_eq!(compiled.barriers.len(), 1);
        let barrier = compiled.barriers[0];
        assert_eq!(
            (barrier.source, barrier.destination),
            (PassId(1), PassId(0))
        );
        assert_eq!(
            (barrier.before, barrier.after),
            (AccessKind::StorageWrite, AccessKind::StorageRead)
        );
        assert!(barrier.queue_transfer);
        assert_eq!(compiled.queue_batches[1].waits_for, [0]);
    }

    #[test]
    fn forward_dependency_cycles_reach_shared_compiler_diagnostics() {
        let mut first = SPECS[0];
        first.depends_on = &["generated_b"];
        let mut graph = GpuFrameGraphBuilder::default();
        declare_passes(&mut graph, &[first, SPECS[1]]).unwrap();
        let Err(super::super::CompileError::Cycle { cycle, names, .. }) = graph.compile() else {
            panic!("explicit cycle must be diagnosed by the shared compiler");
        };
        assert_eq!(cycle, [0, 1, 0]);
        assert_eq!(names, ["generated_a", "generated_b", "generated_a"]);
    }

    #[test]
    fn declaration_capacity_failure_preserves_existing_graph() {
        use crate::{capacity::GraphLimits, CompileError};

        for (limits, kind) in [
            (
                GraphLimits {
                    resources: 1,
                    ..GraphLimits::default()
                },
                "resources",
            ),
            (
                GraphLimits {
                    passes: 2,
                    ..GraphLimits::default()
                },
                "passes",
            ),
            (
                GraphLimits {
                    accesses_per_pass: 0,
                    ..GraphLimits::default()
                },
                "accesses",
            ),
        ] {
            let mut graph = GpuFrameGraphBuilder::with_limits(limits);
            graph.add_resource(ResourceDescriptor {
                name: "existing".into(),
                lifetime: ResourceLifetime::Imported,
                size: 0,
                alignment: 1,
            });
            graph.add_pass(PassDescriptor {
                name: "existing".into(),
                queue: QueueClass::Graphics,
                accesses: Vec::new(),
                depends_on: Vec::new(),
            });
            let resources = graph.resources().to_vec();
            let result = declare_passes(&mut graph, SPECS);
            assert!(
                matches!(result, Err(CodegenError::Graph(CompileError::Capacity { kind: actual, .. })) if actual == kind)
            );
            assert_eq!(graph.resources(), resources);
            assert_eq!(graph.passes().len(), 1);
            assert_eq!(graph.passes()[0].name, "existing");
            assert_eq!(graph.compile().unwrap().execution_order, [PassId(0)]);

            // Neither failed pass/resource admission nor a forward reference
            // may consume IDs or discard dependencies on existing passes.
            let mut retry = SPECS[0];
            retry.resources = &[];
            retry.accesses = &[];
            retry.depends_on = &["existing"];
            let generated = declare_passes(&mut graph, &[retry]).unwrap();
            assert_eq!(generated[0].id, PassId(1));
            assert_eq!(graph.passes()[1].depends_on, [PassId(0)]);
            assert_eq!(graph.resources(), resources);
            assert_eq!(
                graph.compile().unwrap().execution_order,
                [PassId(0), PassId(1)]
            );
        }
    }
}
