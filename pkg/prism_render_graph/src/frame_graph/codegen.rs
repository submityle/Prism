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
/// Validation completes before any pass is inserted, so a malformed generated
/// table cannot leave a partially registered graph behind.
pub fn declare_passes(
    graph: &mut GpuFrameGraphBuilder,
    declarations: &[PassSkeletonDeclaration],
) -> Result<Vec<GeneratedPass>, CodegenError> {
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

    for declaration in declarations {
        for resource in declaration.resources {
            if let Some(existing) = resource_id(graph, &resources, resource.name) {
                resources.insert(resource.name, existing);
            } else {
                let id = graph.add_resource(ResourceDescriptor {
                    name: resource.name.into(),
                    lifetime: resource.lifetime,
                    size: resource.size,
                    alignment: resource.alignment,
                });
                resources.insert(resource.name, id);
            }
        }
    }

    for declaration in declarations {
        graph.add_pass(PassDescriptor {
            name: declaration.name.into(),
            queue: declaration.queue,
            accesses: declaration
                .accesses
                .iter()
                .map(|access| ResourceAccess {
                    resource: resource_id(graph, &resources, access.resource)
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
        });
    }

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
}
