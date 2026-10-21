//! Declaration lowering only. Scheduling, SSA, barriers and aliasing delegate
//! to `GpuFrameGraphBuilder`, including the producer-first declaration ordering.
use super::*;
use crate::frame_graph::*;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    sync::Arc,
    vec::Vec,
};
use core::fmt;
pub use prism_render_graph::capacity::GraphLimits;
use prism_render_graph::{ExecutionPlan, LogicalResource};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FabricCompileError {
    DuplicateTemplate(ResourceKey),
    UndeclaredTemplate(ResourceKey),
    MissingTemplate(ResourceKey),
    InvalidLayout(ResourceKey),
    InvalidAccess {
        pass: PassKey,
        resource: ResourceKey,
        access: AccessKind,
    },
    HistoryNotPersistent(ResourceKey),
    ForeignSelection(FeatureId),
    MultipleWriters(ResourceKey),
    MissingSelectedPass(PassKey),
    DuplicateActivePass(PassKey),
    ForeignActivePass(PassKey),
    ConflictingAlternativePasses {
        feature: FeatureId,
        first: PassKey,
        second: PassKey,
    },
    DuplicateFulfilledDependency(PassKey),
    ForeignFulfilledDependency(PassKey),
    MissingActiveDependency {
        pass: PassKey,
        dependency: PassKey,
    },
    InvalidProducer {
        pass: PassKey,
        resource: ResourceKey,
        producer: PassKey,
    },
    HistoryProducer {
        pass: PassKey,
        resource: ResourceKey,
    },
    MissingProducer(ResourceKey),
    ReadWriteSamePass {
        pass: PassKey,
        resource: ResourceKey,
    },
    DuplicateBinding {
        resource: ResourceKey,
        frame: u64,
    },
    UnexpectedBinding {
        resource: ResourceKey,
        frame: u64,
    },
    MissingBinding {
        resource: ResourceKey,
        frame: u64,
    },
    InvalidBinding {
        resource: ResourceKey,
        frame: u64,
    },
    OverlappingBindings {
        first: ResourceKey,
        first_frame: u64,
        second: ResourceKey,
        second_frame: u64,
    },
    HistoryBeforeFrameZero {
        resource: ResourceKey,
        lag: u32,
    },
    PayloadNotReferenced(ResourceKey),
    PayloadLayoutMismatch(ResourceKey),
    PayloadDomainMismatch(ResourceKey),
    DuplicatePayloadRole {
        object: RenderObjectId,
        domain: RenderDomain,
        history_key: u32,
        role: PayloadRole,
    },
    DomainIdentityMismatch {
        object: RenderObjectId,
        declared: RenderDomain,
    },
    InvalidObjectAccess {
        pass: PassKey,
        ordinal: usize,
    },
    InconsistentObjectResource(ResourceKey),
    MissingObjectPayload {
        resource: ResourceKey,
        identity: ObjectResourceIdentity,
    },
    UnexpectedObjectBinding(ResourceAddress),
    DuplicateObjectBinding(ResourceAddress),
    MissingObjectBinding(ResourceAddress),
    InvalidObjectBinding(ResourceAddress),
    PayloadInstanceMismatch(ResourceKey),
    Graph(CompileError),
}

impl fmt::Display for FabricCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "render fabric compile error: {self:?}")
    }
}
impl std::error::Error for FabricCompileError {}
impl From<CompileError> for FabricCompileError {
    fn from(value: CompileError) -> Self {
        Self::Graph(value)
    }
}

/// Validates resource metadata against *all* variants, not only the selected one.
/// Separate from `FeatureCatalog`: the catalog still owns semantic keys only.
pub struct ResourceTemplates<'a> {
    catalog: &'a FeatureCatalog,
    templates: BTreeMap<ResourceKey, ResourceTemplate>,
    object_accesses: BTreeMap<(PassKey, usize), ObjectResourceAccess>,
    object_resources:
        BTreeMap<ResourceKey, (RenderDomain, ObjectLayoutPolicy, BTreeSet<PayloadRole>)>,
}

#[derive(Clone, Copy)]
struct ExpandedResourceUse {
    ordinal: usize,
    declaration: ResourceUse,
    address: ResourceAddress,
}

#[derive(Default)]
struct IndexedObjectPayloads {
    streams: BTreeMap<u32, BTreeSet<(RenderObjectId, u32)>>,
    payloads: BTreeMap<(ResourceKey, ObjectResourceIdentity), ResourcePayload>,
    item_indices: BTreeMap<ObjectResourceIdentity, usize>,
}

#[derive(Default)]
struct ExternalBindingIndex {
    buffers: BTreeMap<BufferId, BTreeMap<u64, IndexedBufferBinding>>,
    textures: BTreeMap<TextureId, FrameResourceBinding>,
}

struct IndexedBufferBinding {
    order: usize,
    binding: FrameResourceBinding,
}

impl ExternalBindingIndex {
    /// Only insert bindings after layout/range/version validation. Accepted
    /// ranges are disjoint, so successful queries inspect at most a predecessor
    /// plus an empty successor range. The first failure may visit O(n) ranges,
    /// but aborts compilation: total alias validation remains O(n log n).
    fn insert(
        &mut self,
        binding: FrameResourceBinding,
        order: usize,
    ) -> Result<(), FrameResourceBinding> {
        match binding.physical {
            PhysicalResource::Texture(id) => match self.textures.entry(id) {
                alloc::collections::btree_map::Entry::Occupied(previous) => {
                    return Err(*previous.get());
                }
                alloc::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(binding);
                }
            },
            PhysicalResource::Buffer { id, offset } => {
                let end = offset
                    .checked_add(binding.layout.size().expect("validated layout"))
                    .expect("validated binding range");
                let ranges = self.buffers.entry(id).or_default();
                let predecessor = ranges
                    .range(..offset)
                    .next_back()
                    .map(|(_, previous)| previous)
                    .filter(|previous| binding.overlaps(previous.binding));
                // Preserve the old first-prior-resource diagnostic, not the
                // lowest physical offset, if this range spans several entries.
                let collision = predecessor
                    .into_iter()
                    .chain(ranges.range(offset..end).map(|(_, previous)| previous))
                    .min_by_key(|previous| previous.order);
                if let Some(previous) = collision {
                    return Err(previous.binding);
                }
                ranges.insert(offset, IndexedBufferBinding { order, binding });
            }
        }
        Ok(())
    }
}

impl<'a> ResourceTemplates<'a> {
    pub fn new(
        catalog: &'a FeatureCatalog,
        templates: &[ResourceTemplate],
    ) -> Result<Self, FabricCompileError> {
        Self::new_with_object_accesses(catalog, templates, &[])
    }

    /// Accesses absent from this explicit table retain shared semantics. A key
    /// cannot mix shared and per-object uses, even across alternate variants.
    pub fn new_with_object_accesses(
        catalog: &'a FeatureCatalog,
        templates: &[ResourceTemplate],
        object_accesses: &[ObjectResourceAccess],
    ) -> Result<Self, FabricCompileError> {
        let mut result = Self {
            catalog,
            templates: BTreeMap::new(),
            object_accesses: BTreeMap::new(),
            object_resources: BTreeMap::new(),
        };
        let declared: BTreeSet<_> = catalog.resources().collect();
        for &template in templates {
            if !declared.contains(&template.key) {
                return Err(FabricCompileError::UndeclaredTemplate(template.key));
            }
            if template.layout.size().is_none() || !template.alignment.is_power_of_two() {
                return Err(FabricCompileError::InvalidLayout(template.key));
            }
            if result.templates.insert(template.key, template).is_some() {
                return Err(FabricCompileError::DuplicateTemplate(template.key));
            }
        }
        for key in declared {
            if !result.templates.contains_key(&key) {
                return Err(FabricCompileError::MissingTemplate(key));
            }
        }
        for feature in catalog.features() {
            for variant in feature.variants {
                for pass in variant.passes {
                    for usage in pass.accesses {
                        let template = *result
                            .templates
                            .get(&usage.key)
                            .ok_or(FabricCompileError::MissingTemplate(usage.key))?;
                        if !template.layout.supports(usage.access) {
                            return Err(FabricCompileError::InvalidAccess {
                                pass: pass.id,
                                resource: usage.key,
                                access: usage.access,
                            });
                        }
                        if usage.frame_lag.is_some()
                            && template.lifetime != ResourceLifetime::Persistent
                        {
                            return Err(FabricCompileError::HistoryNotPersistent(usage.key));
                        }
                    }
                }
            }
        }
        result.validate_object_accesses(object_accesses)?;
        Ok(result)
    }

    fn validate_object_accesses(
        &mut self,
        declarations: &[ObjectResourceAccess],
    ) -> Result<(), FabricCompileError> {
        // A catalog that validated role-separated writers binds that proof to
        // the exact scopes. Do not let callers erase or relabel it at lowering.
        if !self.catalog.object_accesses().is_empty() {
            let mut supplied = BTreeMap::new();
            for &scope in declarations {
                if self
                    .catalog
                    .object_accesses()
                    .get(&(scope.pass, scope.ordinal))
                    != Some(&scope)
                    || supplied
                        .insert((scope.pass, scope.ordinal), scope)
                        .is_some()
                {
                    return Err(FabricCompileError::InvalidObjectAccess {
                        pass: scope.pass,
                        ordinal: scope.ordinal,
                    });
                }
            }
            for &(pass, ordinal) in self.catalog.object_accesses().keys() {
                if !supplied.contains_key(&(pass, ordinal)) {
                    return Err(FabricCompileError::InvalidObjectAccess { pass, ordinal });
                }
            }
        }
        // Preserve the original first catalog occurrence for repeated pass
        // declarations across variants, and declaration-order diagnostics.
        let mut passes = BTreeMap::new();
        for feature in self.catalog.features() {
            for variant in feature.variants {
                for pass in variant.passes {
                    passes.entry(pass.id).or_insert((feature, pass));
                }
            }
        }
        for &declaration in declarations {
            let invalid = FabricCompileError::InvalidObjectAccess {
                pass: declaration.pass,
                ordinal: declaration.ordinal,
            };
            let &(feature, pass) = passes
                .get(&declaration.pass)
                .ok_or_else(|| invalid.clone())?;
            let domain = feature.domain.ok_or_else(|| invalid.clone())?;
            let usage = pass
                .accesses
                .get(declaration.ordinal)
                .ok_or_else(|| invalid.clone())?;
            // Object storage belongs to a subsystem, never a shared service output.
            let ResourceOwner::Feature(owner) = usage.key.owner() else {
                return Err(invalid);
            };
            if self.catalog.feature(owner).and_then(|owner| owner.domain) != Some(domain)
                || !declaration.layout.accepts(
                    self.templates[&usage.key].layout,
                    self.templates[&usage.key].layout,
                )
                || self
                    .object_accesses
                    .insert((declaration.pass, declaration.ordinal), declaration)
                    .is_some()
            {
                return Err(invalid);
            }
            let scope = self
                .object_resources
                .entry(usage.key)
                .or_insert_with(|| (domain, declaration.layout, BTreeSet::new()));
            if scope.0 != domain || scope.1 != declaration.layout {
                return Err(FabricCompileError::InconsistentObjectResource(usage.key));
            }
            scope.2.insert(declaration.role);
        }
        for feature in self.catalog.features() {
            for variant in feature.variants {
                for pass in variant.passes {
                    for (ordinal, usage) in pass.accesses.iter().enumerate() {
                        if self.object_resources.contains_key(&usage.key)
                            && !self.object_accesses.contains_key(&(pass.id, ordinal))
                        {
                            return Err(FabricCompileError::InconsistentObjectResource(usage.key));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compiles object resources alongside explicit shared bindings. Each
    /// selected pass remains one domain batch, regardless of instance count.
    pub fn compile_with_instances(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        shared: &[FrameResourceBinding],
        objects: &[ObjectResourceBinding],
        items: &[FrameRenderItem],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        self.compile_selected(
            selection,
            frame,
            shared,
            objects,
            items,
            None,
            &BTreeSet::new(),
            limits,
        )
    }

    pub fn compile(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        external: &[FrameResourceBinding],
        items: &[FrameRenderItem],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        self.compile_selected(
            selection,
            frame,
            external,
            &[],
            items,
            None,
            &BTreeSet::new(),
            limits,
        )
    }

    /// Compiles only the named passes from the immutable device selection.
    ///
    /// This is the execution-stage boundary used by frame/view recorders. An
    /// active pass cannot introduce a feature or variant outside `selection`,
    /// and every dependency omitted from this stage must be named explicitly
    /// in `fulfilled_dependencies`. Resource lowering is limited to accesses of
    /// the active passes, so a recorder supplies only its real imports rather
    /// than fabricating bindings for unrelated selected features.
    pub fn compile_active(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        external: &[FrameResourceBinding],
        items: &[FrameRenderItem],
        active_passes: &[PassKey],
        fulfilled_dependencies: &[PassKey],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        self.compile_active_with_instances(
            selection,
            frame,
            external,
            &[],
            items,
            active_passes,
            fulfilled_dependencies,
            limits,
        )
    }

    pub fn compile_active_with_instances(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        external: &[FrameResourceBinding],
        object_external: &[ObjectResourceBinding],
        items: &[FrameRenderItem],
        active_passes: &[PassKey],
        fulfilled_dependencies: &[PassKey],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        self.compile_active_instanced(
            selection,
            frame,
            external,
            object_external,
            items,
            active_passes,
            fulfilled_dependencies,
            limits,
        )
    }

    /// Instance-aware active slice; omitted dependencies must be explicitly
    /// fulfilled, exactly as for the shared-only active compiler entrypoint.
    pub fn compile_active_instanced(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        external: &[FrameResourceBinding],
        object_external: &[ObjectResourceBinding],
        items: &[FrameRenderItem],
        active_passes: &[PassKey],
        fulfilled_dependencies: &[PassKey],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        let mut active = BTreeSet::new();
        for &pass in active_passes {
            if !active.insert(pass) {
                return Err(FabricCompileError::DuplicateActivePass(pass));
            }
        }
        let selected_passes: BTreeSet<_> = selection
            .selected()
            .iter()
            .flat_map(|feature| feature.passes().iter().map(|pass| pass.id))
            .collect();
        if let Some(&pass) = active.iter().find(|pass| !selected_passes.contains(pass)) {
            return Err(FabricCompileError::ForeignActivePass(pass));
        }
        let mut fulfilled = BTreeSet::new();
        for &dependency in fulfilled_dependencies {
            if !fulfilled.insert(dependency) {
                return Err(FabricCompileError::DuplicateFulfilledDependency(dependency));
            }
            if !selected_passes.contains(&dependency) || active.contains(&dependency) {
                return Err(FabricCompileError::ForeignFulfilledDependency(dependency));
            }
        }
        self.compile_selected(
            selection,
            frame,
            external,
            object_external,
            items,
            Some(&active),
            &fulfilled,
            limits,
        )
    }

    fn compile_selected(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        external: &[FrameResourceBinding],
        object_external: &[ObjectResourceBinding],
        items: &[FrameRenderItem],
        active_passes: Option<&BTreeSet<PassKey>>,
        fulfilled_dependencies: &BTreeSet<PassKey>,
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        for selected in selection.selected() {
            let definition = self
                .catalog
                .feature(selected.feature_id())
                .filter(|feature| selected.matches_definition(feature))
                .ok_or(FabricCompileError::ForeignSelection(selected.feature_id()))?;
            let selected_passes: BTreeSet<_> =
                selected.passes().iter().map(|pass| pass.id).collect();
            for group in definition.pass_alternatives {
                // Catalog validation guarantees a variant contains the whole
                // group or none. Check membership once, not per group member.
                if !selected_passes.contains(&group[0]) {
                    continue;
                }
                let mut first = None;
                for &pass in *group {
                    let admitted = active_passes.is_none_or(|active| active.contains(&pass))
                        || fulfilled_dependencies.contains(&pass);
                    if admitted {
                        if let Some(first) = first {
                            return Err(FabricCompileError::ConflictingAlternativePasses {
                                feature: selected.feature_id(),
                                first,
                                second: pass,
                            });
                        }
                        first = Some(pass);
                    }
                }
            }
        }
        let IndexedObjectPayloads {
            streams,
            payloads: object_payloads,
            item_indices,
        } = self.index_object_payloads(items)?;
        let mut declarations = Vec::new();
        for feature in selection.selected() {
            for pass in feature.passes() {
                if active_passes.is_some_and(|active| !active.contains(&pass.id)) {
                    continue;
                }
                GraphLimits::check_count("passes", declarations.len() + 1, limits.passes)?;
                declarations.push((feature.feature_id(), feature.variant_id(), pass));
            }
        }
        let pass_indices: BTreeMap<_, _> = declarations
            .iter()
            .enumerate()
            .map(|(index, (_, _, pass))| (pass.id, index))
            .collect();
        let mut expanded = Vec::with_capacity(declarations.len());
        let mut writers = BTreeMap::<ResourceAddress, Vec<usize>>::new();
        let mut needed = BTreeSet::new();
        for (index, &(_, _, pass)) in declarations.iter().enumerate() {
            let mut accesses = Vec::new();
            for (ordinal, &usage) in pass.accesses.iter().enumerate() {
                let resource_frame = use_frame(frame, usage)?;
                let mut push = |identity| -> Result<(), FabricCompileError> {
                    GraphLimits::check_count(
                        "accesses",
                        accesses.len() + 1,
                        limits.accesses_per_pass,
                    )?;
                    accesses.push(ExpandedResourceUse {
                        ordinal,
                        declaration: usage,
                        address: ResourceAddress {
                            key: usage.key,
                            frame: resource_frame,
                            identity,
                        },
                    });
                    Ok(())
                };
                if let Some(scope) = self.object_accesses.get(&(pass.id, ordinal)) {
                    let domain = self.object_resources[&usage.key].0;
                    for &(object, history_key) in
                        streams.get(&(domain as u32)).into_iter().flatten()
                    {
                        let identity = ObjectResourceIdentity::new(object, history_key, scope.role);
                        if !object_payloads.contains_key(&(usage.key, identity)) {
                            return Err(FabricCompileError::MissingObjectPayload {
                                resource: usage.key,
                                identity,
                            });
                        }
                        push(Some(identity))?;
                    }
                } else {
                    push(None)?;
                }
            }
            for usage in &accesses {
                let address = usage.address;
                if !needed.contains(&address) {
                    GraphLimits::check_count("resources", needed.len() + 1, limits.resources)?;
                    needed.insert(address);
                }
                if usage.declaration.access.writes() {
                    writers.entry(address).or_default().push(index);
                }
            }
            expanded.push(accesses);
        }

        // Preserve the producer-first Fabric pass identity required by runtime
        // lowering. Ask the shared kernel only for control admission order;
        // no resource lowering or full compilation is needed at this stage.
        let mut admission = GpuFrameGraphBuilder::with_limits(limits);
        let mut sources = Vec::with_capacity(declarations.len());
        let mut prerequisites = Vec::with_capacity(declarations.len());
        let mut successors = BTreeSet::new();
        for (index, &(_, _, pass)) in declarations.iter().enumerate() {
            let mut dependencies = BTreeSet::new();
            for &dependency in pass.depends_on {
                if let Some(&dependency) = pass_indices.get(&dependency) {
                    dependencies.insert(dependency);
                } else if !fulfilled_dependencies.contains(&dependency) {
                    return Err(if active_passes.is_some() {
                        FabricCompileError::MissingActiveDependency {
                            pass: pass.id,
                            dependency,
                        }
                    } else {
                        FabricCompileError::MissingSelectedPass(dependency)
                    });
                }
            }
            let mut pass_sources = Vec::with_capacity(expanded[index].len());
            for expanded_usage in &expanded[index] {
                let usage = &expanded_usage.declaration;
                let resource_writers = writers
                    .get(&expanded_usage.address)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let source = if usage.frame_lag.is_some() {
                    if matches!(usage.input, ResourceInput::ProducedBy(_)) {
                        return Err(FabricCompileError::HistoryProducer {
                            pass: pass.id,
                            resource: usage.key,
                        });
                    }
                    None
                } else {
                    match usage.input {
                        ResourceInput::Initial => None,
                        ResourceInput::Automatic => {
                            if resource_writers.len() > 1 {
                                return Err(FabricCompileError::MultipleWriters(usage.key));
                            }
                            if usage.access.writes() {
                                None
                            } else {
                                resource_writers.first().copied()
                            }
                        }
                        ResourceInput::ProducedBy(producer) => {
                            let source = pass_indices
                                .get(&producer)
                                .copied()
                                .filter(|source| {
                                    // Writers are appended in declaration order.
                                    *source != index
                                        && resource_writers.binary_search(source).is_ok()
                                })
                                .ok_or(FabricCompileError::InvalidProducer {
                                    pass: pass.id,
                                    resource: usage.key,
                                    producer,
                                })?;
                            Some(source)
                        }
                    }
                };
                if let Some(source) = source {
                    if source == index {
                        return Err(FabricCompileError::ReadWriteSamePass {
                            pass: pass.id,
                            resource: usage.key,
                        });
                    }
                    dependencies.insert(source);
                } else if !usage.access.writes()
                    && self.templates[&usage.key].lifetime == ResourceLifetime::Transient
                {
                    return Err(FabricCompileError::MissingProducer(usage.key));
                }
                if usage.access.writes() && !successors.insert((expanded_usage.address, source)) {
                    return Err(FabricCompileError::MultipleWriters(usage.key));
                }
                pass_sources.push(source);
            }
            sources.push(pass_sources);
            admission.try_add_pass(PassDescriptor {
                name: format!("feature-pass:{}", pass.id.0).into(),
                queue: pass.queue,
                accesses: Vec::new(),
                depends_on: dependencies
                    .iter()
                    .copied()
                    .map(|id| PassId(id as u32))
                    .collect(),
            })?;
            prerequisites.push(dependencies);
        }
        let admission_order = admission
            .dependency_order()?
            .into_iter()
            .map(|id| id.0 as usize)
            .collect::<Vec<_>>();
        let mut supplied = BTreeMap::new();
        for binding in external {
            let Some(template) = self.templates.get(&binding.key) else {
                return Err(FabricCompileError::UnexpectedBinding {
                    resource: binding.key,
                    frame: binding.frame,
                });
            };
            if !needed.contains(&ResourceAddress::shared(binding.key, binding.frame))
                || template.lifetime == ResourceLifetime::Transient
            {
                return Err(FabricCompileError::UnexpectedBinding {
                    resource: binding.key,
                    frame: binding.frame,
                });
            }
        }
        for &binding in external {
            let address = ResourceAddress::shared(binding.key, binding.frame);
            if supplied.insert(address, binding).is_some() {
                return Err(FabricCompileError::DuplicateBinding {
                    resource: binding.key,
                    frame: binding.frame,
                });
            }
        }
        for &object in object_external {
            let address = object.address();
            if !needed.contains(&address)
                || self
                    .templates
                    .get(&address.key)
                    .is_none_or(|template| template.lifetime == ResourceLifetime::Transient)
            {
                return Err(FabricCompileError::UnexpectedObjectBinding(address));
            }
            if supplied.insert(address, object.binding).is_some() {
                return Err(FabricCompileError::DuplicateObjectBinding(address));
            }
        }
        let mut graph = GpuFrameGraphBuilder::with_limits(limits);
        let mut resources: Vec<BoundResource> = Vec::new();
        let mut ids = BTreeMap::new();
        let mut binding_index = ExternalBindingIndex::default();
        for address in needed {
            let ResourceAddress {
                key,
                frame: resource_frame,
                identity,
            } = address;
            let mut template = *self
                .templates
                .get(&key)
                .ok_or(FabricCompileError::MissingTemplate(key))?;
            let history = resource_frame != frame;
            if let Some(identity) = identity {
                let layout = if history {
                    supplied
                        .get(&address)
                        .ok_or(FabricCompileError::MissingObjectBinding(address))?
                        .layout
                } else {
                    object_payloads[&(key, identity)].layout
                };
                if !self.object_resources[&key]
                    .1
                    .accepts(template.layout, layout)
                {
                    return Err(FabricCompileError::InvalidObjectBinding(address));
                }
                template.layout = layout;
            }
            let binding = if template.lifetime == ResourceLifetime::Transient {
                None
            } else {
                let binding = supplied.get(&address).copied().ok_or_else(|| {
                    if identity.is_some() {
                        FabricCompileError::MissingObjectBinding(address)
                    } else {
                        FabricCompileError::MissingBinding {
                            resource: key,
                            frame: resource_frame,
                        }
                    }
                })?;
                if !binding.matches(template) {
                    if identity.is_some() {
                        return Err(FabricCompileError::InvalidObjectBinding(address));
                    }
                    return Err(FabricCompileError::InvalidBinding {
                        resource: key,
                        frame: resource_frame,
                    });
                }
                if binding.version > limits.versions {
                    return Err(CompileError::VersionOverflow {
                        version: binding.version,
                        limit: limits.versions,
                    }
                    .into());
                }
                if let Err(other) = binding_index.insert(binding, resources.len()) {
                    return Err(FabricCompileError::OverlappingBindings {
                        first: other.key,
                        first_frame: other.frame,
                        second: key,
                        second_frame: resource_frame,
                    });
                }
                Some(binding)
            };
            let descriptor = ResourceDescriptor {
                name: format!("{:?}", address).into(),
                lifetime: if history {
                    ResourceLifetime::Imported
                } else {
                    template.lifetime
                },
                size: template.layout.size().expect("validated layout"),
                alignment: template.alignment,
            };
            let id = graph.try_add_resource(descriptor.clone())?;
            ids.insert(address, id);
            resources.push(BoundResource {
                id,
                key,
                frame: resource_frame,
                identity,
                template,
                descriptor,
                external: binding,
            });
        }
        // The physical-range index is validation-only; release it before the
        // graph compiler allocates its SSA/hazard tables.
        drop(binding_index);
        let admitted_indices: BTreeMap<_, _> = admission_order
            .iter()
            .enumerate()
            .map(|(index, &original)| (original, PassId(index as u32)))
            .collect();
        let mut passes = Vec::new();
        for &admitted in &admission_order {
            let (feature, variant, template) = declarations[admitted];
            let accesses = expanded[admitted]
                .iter()
                .map(|usage| {
                    Ok(ResourceAccess {
                        resource: ids
                            .get(&usage.address)
                            .copied()
                            .ok_or(FabricCompileError::MissingTemplate(usage.declaration.key))?,
                        kind: usage.declaration.access,
                    })
                })
                .collect::<Result<Vec<_>, FabricCompileError>>()?;
            let descriptor = PassDescriptor {
                name: format!("feature-pass:{}", template.id.0).into(),
                queue: template.queue,
                accesses,
                depends_on: prerequisites[admitted]
                    .iter()
                    .map(|dependency| admitted_indices[dependency])
                    .collect(),
            };
            let id = graph.try_add_pass(descriptor.clone())?;
            passes.push(PassSpec {
                id,
                feature,
                variant,
                domain: self
                    .catalog
                    .feature(feature)
                    .expect("validated selection")
                    .domain,
                template,
                template_ordinals: expanded[admitted]
                    .iter()
                    .map(|usage| usage.ordinal)
                    .collect(),
                access_indices: BTreeMap::new(),
                descriptor,
                bound_accesses: Vec::new(),
            });
        }
        // Bind semantic producers to declaration access identities, not guessed
        // SSA versions. The single full compile resolves these references from
        // checked accessor outputs as it lowers the real graph.
        let access_orders = normalized_access_orders(&passes, resources.len());
        let mut writer_accesses = BTreeMap::new();
        for pass in &passes {
            for (ordinal, usage) in pass.descriptor.accesses.iter().enumerate() {
                if usage.kind.writes() {
                    writer_accesses.insert(
                        (pass.key(), resources[usage.resource.0 as usize].address()),
                        (pass.id, ordinal),
                    );
                }
            }
        }
        let inputs = admission_order
            .iter()
            .map(|&original| {
                sources[original]
                    .iter()
                    .zip(&expanded[original])
                    .map(|(source, usage)| {
                        let Some(source) = source else {
                            return Ok(AccessInput::Version(0));
                        };
                        let producer = declarations[*source].2.id;
                        let &(pass, access) = writer_accesses
                            .get(&(producer, usage.address))
                            .ok_or(FabricCompileError::InvalidProducer {
                                pass: declarations[original].2.id,
                                resource: usage.declaration.key,
                                producer,
                            })?;
                        Ok(AccessInput::ProducedBy { pass, access })
                    })
                    .collect::<Result<Vec<_>, FabricCompileError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = graph.compile_with_sources(&inputs)?;
        let mut versions = Vec::new();
        for &pass_index in &plan.order {
            let pass = &passes[pass_index];
            for (&ordinal, access) in access_orders[pass_index]
                .iter()
                .zip(&plan.accesses[pass_index])
            {
                let Some(ssa_version) = access.output_version else {
                    continue;
                };
                debug_assert!(
                    pass.descriptor.accesses[ordinal].kind.writes(),
                    "only writes may produce an SSA output version"
                );
                let resource = &resources[pass.descriptor.accesses[ordinal].resource.0 as usize];
                let base = resource.external.map_or(0, |binding| binding.version);
                let content_version = base
                    .checked_add(ssa_version)
                    .filter(|v| *v <= limits.versions)
                    .ok_or(CompileError::VersionOverflow {
                        version: base,
                        limit: limits.versions,
                    })?;
                versions.push(FrameResourceVersion {
                    key: resource.key,
                    frame: resource.frame,
                    identity: resource.identity,
                    resource: resource.id,
                    writer: passes[pass_index].key(),
                    ssa_version,
                    content_version,
                });
            }
        }
        // Retain every output; an older read must not be rebound to the final
        // write of the frame. History and initial contents stay at SSA zero.
        let outputs: BTreeMap<_, _> = versions
            .iter()
            .map(|version| ((version.resource.0, version.ssa_version), *version))
            .collect();
        for (index, pass) in passes.iter_mut().enumerate() {
            let mut authority = vec![None; pass.descriptor.accesses.len()];
            for (&ordinal, access) in access_orders[index].iter().zip(&plan.accesses[index]) {
                authority[ordinal] = Some(*access);
            }
            pass.bound_accesses = pass
                .template_ordinals
                .iter()
                .zip(&pass.descriptor.accesses)
                .zip(authority)
                .map(|((&template_ordinal, access), authority)| {
                    let declaration = pass.template.accesses[template_ordinal];
                    let authority = authority.expect("authority preserves admitted accesses");
                    let resource = &resources[access.resource.0 as usize];
                    let version = authority.output_version.unwrap_or(authority.input_version);
                    let output = outputs.get(&(access.resource.0, version));
                    let input = outputs.get(&(access.resource.0, authority.input_version));
                    BoundResourceUse {
                        declaration,
                        template_ordinal,
                        identity: resource.identity,
                        resource: resource.id,
                        frame: resource.frame,
                        input_ssa_version: authority.input_version,
                        input_content_version: input.map_or_else(
                            || resource.external.map_or(0, |binding| binding.version),
                            |version| version.content_version,
                        ),
                        input_producer: input.map(|version| version.writer),
                        ssa_version: version,
                        content_version: output.map_or_else(
                            || resource.external.map_or(0, |binding| binding.version),
                            |version| version.content_version,
                        ),
                        producer: output.map(|version| version.writer),
                    }
                })
                .collect();
            pass.access_indices = pass
                .bound_accesses
                .iter()
                .enumerate()
                .map(|(index, usage)| ((usage.identity, usage.template_ordinal), index))
                .collect();
        }
        let barriers = project_barriers(&plan, &passes, resources.len());
        let mut compiled = CompiledFeatureGraph {
            frame,
            limits,
            resources,
            ids,
            passes,
            versions,
            plan,
            barriers,
            barriers_by_destination: Vec::new(),
            items: Arc::from(items),
            item_indices,
        };
        compiled.barriers_by_destination = vec![Vec::new(); compiled.plan.declared_pass_count];
        for (index, barrier) in compiled.barriers.iter().enumerate() {
            compiled.barriers_by_destination[barrier.destination.0 as usize].push(index);
        }
        // Index current-frame domain accesses once. Frame items can number in
        // the millions; rescanning every selected pass per item is unnecessary.
        let payload_accesses: BTreeSet<_> = compiled
            .passes
            .iter()
            .filter_map(|pass| pass.domain().map(|domain| (pass, domain)))
            .flat_map(|(pass, domain)| {
                pass.accesses()
                    .iter()
                    .filter(|access| access.frame == frame)
                    .map(move |access| (access.address(), domain as u32))
            })
            .collect();
        for item in items {
            self.validate_payload(&compiled, &payload_accesses, item)?;
        }
        Ok(compiled)
    }

    fn index_object_payloads(
        &self,
        items: &[FrameRenderItem],
    ) -> Result<IndexedObjectPayloads, FabricCompileError> {
        let mut indexed = IndexedObjectPayloads::default();
        for (index, item) in items.iter().enumerate() {
            if item.object_id.domain() != item.domain {
                return Err(FabricCompileError::DomainIdentityMismatch {
                    object: item.object_id,
                    declared: item.domain,
                });
            }
            let identity =
                ObjectResourceIdentity::new(item.object_id, item.history_key, item.payload_role());
            if indexed.item_indices.insert(identity, index).is_some() {
                return Err(FabricCompileError::DuplicatePayloadRole {
                    object: item.object_id,
                    domain: item.domain,
                    history_key: item.history_key,
                    role: item.payload_role(),
                });
            }
            indexed
                .streams
                .entry(item.domain as u32)
                .or_default()
                .insert((item.object_id, item.history_key));
            let DomainPayload::Resource(payload) = item.payload else {
                continue;
            };
            let Some((domain, policy, roles)) = self.object_resources.get(&payload.resource) else {
                continue;
            };
            if *domain != item.domain || !roles.contains(&payload.role) {
                return Err(FabricCompileError::PayloadInstanceMismatch(
                    payload.resource,
                ));
            }
            let template = self.templates[&payload.resource];
            if payload.lifetime != template.lifetime
                || payload.alignment != template.alignment
                || payload.layout.size() != Some(payload.size)
                || !policy.accepts(template.layout, payload.layout)
            {
                return Err(FabricCompileError::PayloadLayoutMismatch(payload.resource));
            }
            indexed
                .payloads
                .insert((payload.resource, identity), payload);
        }
        Ok(indexed)
    }

    fn validate_payload(
        &self,
        graph: &CompiledFeatureGraph,
        accesses: &BTreeSet<(ResourceAddress, u32)>,
        item: &FrameRenderItem,
    ) -> Result<(), FabricCompileError> {
        if let DomainPayload::Resource(payload) = item.payload {
            // A current-frame domain payload must actually occur in a pass, not
            // merely be present in the resource declaration table/history map.
            let identity = self
                .object_resources
                .contains_key(&payload.resource)
                .then(|| {
                    ObjectResourceIdentity::new(item.object_id, item.history_key, payload.role)
                });
            let address = ResourceAddress {
                key: payload.resource,
                frame: graph.frame,
                identity,
            };
            let id = graph
                .resource_id_at(address)
                .ok_or(FabricCompileError::PayloadNotReferenced(payload.resource))?;
            let template = graph.resources[id.0 as usize].template;
            if payload.lifetime != template.lifetime
                || payload.layout != template.layout
                || payload.size != template.layout.size().expect("validated layout")
                || payload.alignment != template.alignment
            {
                return Err(FabricCompileError::PayloadLayoutMismatch(payload.resource));
            }
            if let ResourceOwner::Feature(owner) = payload.resource.owner()
                && let Some(domain) = self.catalog.feature(owner).and_then(|f| f.domain)
                && domain != item.domain
            {
                return Err(FabricCompileError::PayloadDomainMismatch(payload.resource));
            }
            if !accesses.contains(&(address, item.domain as u32)) {
                return Err(FabricCompileError::PayloadNotReferenced(payload.resource));
            }
        }
        Ok(())
    }
}

fn use_frame(frame: u64, usage: ResourceUse) -> Result<u64, FabricCompileError> {
    match usage.frame_lag {
        None => Ok(frame),
        Some(lag) => frame.checked_sub(u64::from(lag.frames())).ok_or(
            FabricCompileError::HistoryBeforeFrameZero {
                resource: usage.key,
                lag: lag.frames(),
            },
        ),
    }
}

/// Original access ordinals in the authority's texture-then-buffer order.
/// Resource namespace inference mirrors the declaration frontend; it does not
/// assign versions or derive dependencies.
fn texture_resources(passes: &[PassSpec], resource_count: usize) -> Vec<bool> {
    let mut textures = vec![false; resource_count];
    for access in passes.iter().flat_map(|pass| &pass.descriptor.accesses) {
        if access.kind.requires_texture() {
            textures[access.resource.0 as usize] = true;
        }
    }
    textures
}

fn normalized_access_orders(passes: &[PassSpec], resource_count: usize) -> Vec<Vec<usize>> {
    let textures = texture_resources(passes, resource_count);
    passes
        .iter()
        .map(|pass| {
            pass.descriptor
                .accesses
                .iter()
                .enumerate()
                .filter(|(_, access)| textures[access.resource.0 as usize])
                .chain(
                    pass.descriptor
                        .accesses
                        .iter()
                        .enumerate()
                        .filter(|(_, access)| !textures[access.resource.0 as usize]),
                )
                .map(|(ordinal, _)| ordinal)
                .collect()
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct BoundResource {
    pub id: ResourceId,
    pub key: ResourceKey,
    pub frame: u64,
    pub identity: Option<ObjectResourceIdentity>,
    pub template: ResourceTemplate,
    pub descriptor: ResourceDescriptor,
    pub external: Option<FrameResourceBinding>,
}
impl BoundResource {
    pub const fn address(&self) -> ResourceAddress {
        ResourceAddress {
            key: self.key,
            frame: self.frame,
            identity: self.identity,
        }
    }
}

/// The only bound pass representation; constructed from a selected template.
#[derive(Clone, Debug)]
pub struct PassSpec {
    id: PassId,
    feature: FeatureId,
    variant: VariantId,
    domain: Option<RenderDomain>,
    template: &'static PassTemplate,
    template_ordinals: Vec<usize>,
    access_indices: BTreeMap<(Option<ObjectResourceIdentity>, usize), usize>,
    descriptor: PassDescriptor,
    bound_accesses: Vec<BoundResourceUse>,
}
impl PassSpec {
    pub const fn id(&self) -> PassId {
        self.id
    }
    pub const fn key(&self) -> PassKey {
        self.template.id
    }
    pub const fn feature(&self) -> FeatureId {
        self.feature
    }
    pub const fn variant(&self) -> VariantId {
        self.variant
    }
    /// Declared subsystem domain; services and responses remain domainless.
    pub const fn domain(&self) -> Option<RenderDomain> {
        self.domain
    }
    pub const fn template(&self) -> &'static PassTemplate {
        self.template
    }
    pub const fn descriptor(&self) -> &PassDescriptor {
        &self.descriptor
    }
    pub fn accesses(&self) -> &[BoundResourceUse] {
        &self.bound_accesses
    }
    /// Lookup is by original template ordinal, never an expanded-vector index.
    /// `None` explicitly requests shared storage.
    pub fn access(
        &self,
        identity: Option<ObjectResourceIdentity>,
        ordinal: usize,
    ) -> Option<&BoundResourceUse> {
        self.bound_accesses
            .get(self.access_index(identity, ordinal)?)
    }
    pub(super) fn access_index(
        &self,
        identity: Option<ObjectResourceIdentity>,
        ordinal: usize,
    ) -> Option<usize> {
        self.access_indices.get(&(identity, ordinal)).copied()
    }
}

/// Read provenance or write output from the compiled authority, with the frame
/// and external content version retained for backend/history binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundResourceUse {
    pub declaration: ResourceUse,
    pub template_ordinal: usize,
    pub identity: Option<ObjectResourceIdentity>,
    pub resource: ResourceId,
    pub frame: u64,
    /// Input observed before this access; writes retain their predecessor too.
    pub input_ssa_version: u32,
    pub input_content_version: u32,
    pub input_producer: Option<PassKey>,
    pub ssa_version: u32,
    pub content_version: u32,
    /// None denotes imported initial/history contents, never an implicit pass.
    pub producer: Option<PassKey>,
}
impl BoundResourceUse {
    pub const fn address(self) -> ResourceAddress {
        ResourceAddress {
            key: self.declaration.key,
            frame: self.frame,
            identity: self.identity,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameResourceVersion {
    pub key: ResourceKey,
    pub frame: u64,
    pub identity: Option<ObjectResourceIdentity>,
    pub resource: ResourceId,
    pub writer: PassKey,
    pub ssa_version: u32,
    pub content_version: u32,
}
impl FrameResourceVersion {
    pub const fn address(self) -> ResourceAddress {
        ResourceAddress {
            key: self.key,
            frame: self.frame,
            identity: self.identity,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompiledFeatureGraph {
    frame: u64,
    limits: GraphLimits,
    resources: Vec<BoundResource>,
    ids: BTreeMap<ResourceAddress, ResourceId>,
    passes: Vec<PassSpec>,
    versions: Vec<FrameResourceVersion>,
    plan: ExecutionPlan,
    barriers: Vec<Barrier>,
    barriers_by_destination: Vec<Vec<usize>>,
    items: Arc<[FrameRenderItem]>,
    item_indices: BTreeMap<ObjectResourceIdentity, usize>,
}
impl CompiledFeatureGraph {
    pub const fn frame(&self) -> u64 {
        self.frame
    }
    pub(super) const fn limits(&self) -> GraphLimits {
        self.limits
    }
    pub fn resource_id(&self, key: ResourceKey, frame: u64) -> Option<ResourceId> {
        self.resource_id_at(ResourceAddress::shared(key, frame))
    }
    pub fn resource_id_at(&self, address: ResourceAddress) -> Option<ResourceId> {
        self.ids.get(&address).copied()
    }
    pub fn object_resource_id(
        &self,
        key: ResourceKey,
        frame: u64,
        identity: ObjectResourceIdentity,
    ) -> Option<ResourceId> {
        self.resource_id_at(ResourceAddress::object(key, frame, identity))
    }
    pub fn resources(&self) -> &[BoundResource] {
        &self.resources
    }
    pub fn passes(&self) -> &[PassSpec] {
        &self.passes
    }
    pub fn versions(&self) -> &[FrameResourceVersion] {
        &self.versions
    }
    pub fn items(&self) -> &[FrameRenderItem] {
        &self.items
    }
    pub(super) fn shared_items(&self) -> Arc<[FrameRenderItem]> {
        Arc::clone(&self.items)
    }
    /// O(log n) identity lookup followed by full validated item equality,
    /// including payload metadata/value and motion. Float equality retains the
    /// same semantics as `items().contains(item)`; payloads need not implement Ord.
    pub fn contains_item(&self, item: &FrameRenderItem) -> bool {
        let identity =
            ObjectResourceIdentity::new(item.object_id, item.history_key, item.payload_role());
        self.item_indices
            .get(&identity)
            .is_some_and(|&index| self.items[index] == *item)
    }
    pub const fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }
    pub fn barriers(&self) -> &[Barrier] {
        &self.barriers
    }

    /// Consume a compiled plan once. The sink realizes resources and records
    /// work; this is not a claim that a concrete GPU executor is integrated.
    /// No callback accepts a replacement access list or resolves variants.
    pub fn execute<S: FeatureGraphSink>(self, sink: &mut S) -> Result<(), S::Error> {
        sink.begin(&self.resources, &self.items, &self.plan)?;
        for &id in &self.plan.order {
            for &barrier in &self.barriers_by_destination[id] {
                sink.barrier(&self.barriers[barrier])?;
            }
            sink.pass(&self.passes[id])?;
        }
        sink.finish(&self.plan)
    }
}

/// Backend integration seam: all methods are required, with no no-op defaults.
pub trait FeatureGraphSink {
    type Error;
    fn begin(
        &mut self,
        resources: &[BoundResource],
        items: &[FrameRenderItem],
        plan: &ExecutionPlan,
    ) -> Result<(), Self::Error>;
    fn barrier(&mut self, barrier: &Barrier) -> Result<(), Self::Error>;
    fn pass(&mut self, pass: &PassSpec) -> Result<(), Self::Error>;
    fn finish(&mut self, plan: &ExecutionPlan) -> Result<(), Self::Error>;
}

/// Projects the unified graph's logical hazards into the semantic render
/// fabric barrier type. The graph remains the sole scheduler and hazard
/// solver; this function only restores the feature graph's resource IDs and
/// access vocabulary for its sink contract.
fn project_barriers(
    plan: &ExecutionPlan,
    passes: &[PassSpec],
    resource_count: usize,
) -> Vec<Barrier> {
    let resource_is_texture = texture_resources(passes, resource_count);
    // `ExecutionPlan` keeps texture and buffer resource indices in separate
    // namespaces. `PassSpec` still exposes the feature-graph's single
    // ResourceId namespace, so build dense namespace-local projections before
    // lowering logical barriers.
    let texture_resource_ids: Vec<_> = resource_is_texture
        .iter()
        .enumerate()
        .filter_map(|(index, &is_texture)| is_texture.then_some(ResourceId(index as u32)))
        .collect();
    let buffer_resource_ids: Vec<_> = resource_is_texture
        .iter()
        .enumerate()
        .filter_map(|(index, &is_texture)| (!is_texture).then_some(ResourceId(index as u32)))
        .collect();
    let normalized: Vec<Vec<AccessKind>> = passes
        .iter()
        .map(|pass| {
            pass.descriptor
                .accesses
                .iter()
                .filter(|access| resource_is_texture[access.resource.0 as usize])
                .map(|access| access.kind)
                .chain(
                    pass.descriptor
                        .accesses
                        .iter()
                        .filter(|access| !resource_is_texture[access.resource.0 as usize])
                        .map(|access| access.kind),
                )
                .collect()
        })
        .collect();

    plan.logical_barriers
        .iter()
        .map(|barrier| {
            let resource = match barrier.resource {
                LogicalResource::Texture(index) => texture_resource_ids[index.get() as usize],
                LogicalResource::Buffer(index) => buffer_resource_ids[index.get() as usize],
            };
            Barrier {
                resource,
                source: PassId(barrier.source as u32),
                destination: PassId(barrier.destination as u32),
                before: normalized[barrier.source][barrier.before],
                after: normalized[barrier.destination][barrier.after],
                queue_transfer: barrier.queue_transfer,
            }
        })
        .collect()
}

#[cfg(test)]
mod binding_index_tests {
    use super::*;

    #[test]
    fn indexed_alias_diagnostics_match_linear_reference() {
        for permutation in [1, 3, 17, 63] {
            let mut index = ExternalBindingIndex::default();
            let mut accepted: Vec<FrameResourceBinding> = Vec::new();
            // All permutations visit the same disjoint ranges, but in different
            // insertion orders. Later candidates cross several of those ranges.
            for ordinal in 0..384_u64 {
                let (offset, bytes) = if ordinal < 64 {
                    ((ordinal * permutation % 64) * 64, 32)
                } else {
                    ((ordinal * 113 % 270) * 16, (1 + ordinal % 23) * 16)
                };
                let binding = FrameResourceBinding {
                    key: ResourceKey::of(FeatureId(700), ordinal as u16),
                    frame: ordinal,
                    version: 0,
                    physical: PhysicalResource::Buffer {
                        id: BufferId::from_parts(1, 1),
                        offset,
                    },
                    layout: ResourceLayout::Buffer {
                        bytes,
                        usage: BufferUsages::STORAGE,
                    },
                    alignment: 16,
                };
                let expected = accepted
                    .iter()
                    .copied()
                    .find(|&other| binding.overlaps(other));
                assert_eq!(index.insert(binding, accepted.len()).err(), expected);
                if expected.is_none() {
                    accepted.push(binding);
                }
            }
        }
    }

    #[test]
    fn alias_index_keeps_touching_ranges_generations_and_resource_kinds_distinct() {
        let mut index = ExternalBindingIndex::default();
        let first = FrameResourceBinding {
            key: ResourceKey::of(FeatureId(700), 0),
            frame: 10,
            version: 0,
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(1, 1),
                offset: 0,
            },
            layout: ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE,
            },
            alignment: 16,
        };
        let touching = FrameResourceBinding {
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(1, 1),
                offset: 64,
            },
            ..first
        };
        let recycled = FrameResourceBinding {
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(1, 2),
                offset: 0,
            },
            ..first
        };
        let texture = FrameResourceBinding {
            physical: PhysicalResource::Texture(TextureId::from_parts(1, 1)),
            layout: ResourceLayout::Texture {
                extent: [4, 4, 1],
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsages::TEXTURE_BINDING,
            },
            ..first
        };
        for (order, binding) in [first, touching, recycled, texture].into_iter().enumerate() {
            assert_eq!(index.insert(binding, order), Ok(()));
        }
        assert_eq!(index.insert(first, 4), Err(first));
        assert_eq!(index.insert(texture, 4), Err(texture));
        assert_eq!(
            index.insert(
                FrameResourceBinding {
                    frame: 9,
                    ..texture
                },
                4
            ),
            Err(texture)
        );
        assert_eq!(
            index.insert(
                FrameResourceBinding {
                    physical: PhysicalResource::Texture(TextureId::from_parts(1, 2)),
                    ..texture
                },
                4
            ),
            Ok(())
        );
    }
}
