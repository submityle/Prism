//! Declaration lowering only. Scheduling, SSA, barriers and aliasing delegate
//! to `GpuFrameGraphBuilder`, including the producer-first declaration ordering.
use super::*;
use crate::frame_graph::*;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
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
}

impl<'a> ResourceTemplates<'a> {
    pub fn new(
        catalog: &'a FeatureCatalog,
        templates: &[ResourceTemplate],
    ) -> Result<Self, FabricCompileError> {
        let mut result = Self {
            catalog,
            templates: BTreeMap::new(),
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
        Ok(result)
    }

    pub fn compile(
        &self,
        selection: &FeatureSelection,
        frame: u64,
        external: &[FrameResourceBinding],
        items: &[FrameRenderItem],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        for selected in selection.selected() {
            let valid = self
                .catalog
                .feature(selected.feature_id())
                .is_some_and(|feature| selected.matches_definition(feature));
            if !valid {
                return Err(FabricCompileError::ForeignSelection(selected.feature_id()));
            }
        }
        let mut declarations = Vec::new();
        for feature in selection.selected() {
            for pass in feature.passes() {
                GraphLimits::check_count("passes", declarations.len() + 1, limits.passes)?;
                declarations.push((feature.feature_id(), feature.variant_id(), pass));
            }
        }
        let pass_indices: BTreeMap<_, _> = declarations
            .iter()
            .enumerate()
            .map(|(index, (_, _, pass))| (pass.id, index))
            .collect();
        let mut writers = BTreeMap::<ResourceKey, Vec<usize>>::new();
        let mut needed = BTreeSet::new();
        for (index, &(_, _, pass)) in declarations.iter().enumerate() {
            GraphLimits::check_count("accesses", pass.accesses.len(), limits.accesses_per_pass)?;
            for usage in pass.accesses {
                let resource_frame = use_frame(frame, *usage)?;
                let address = (usage.key, resource_frame);
                if !needed.contains(&address) {
                    GraphLimits::check_count("resources", needed.len() + 1, limits.resources)?;
                    needed.insert(address);
                }
                if usage.access.writes() {
                    writers.entry(usage.key).or_default().push(index);
                }
            }
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
                dependencies.insert(*pass_indices.get(&dependency)
                    .ok_or(FabricCompileError::MissingSelectedPass(dependency))?);
            }
            let mut pass_sources = Vec::with_capacity(pass.accesses.len());
            for usage in pass.accesses {
                let resource_writers = writers.get(&usage.key).map(Vec::as_slice).unwrap_or(&[]);
                let source = if usage.frame_lag.is_some() {
                    if matches!(usage.input, ResourceInput::ProducedBy(_)) {
                        return Err(FabricCompileError::HistoryProducer {
                            pass: pass.id, resource: usage.key,
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
                            if usage.access.writes() { None } else { resource_writers.first().copied() }
                        }
                        ResourceInput::ProducedBy(producer) => {
                            let source = pass_indices.get(&producer).copied()
                                .filter(|source| *source != index && resource_writers.contains(source))
                                .ok_or(FabricCompileError::InvalidProducer {
                                    pass: pass.id, resource: usage.key, producer,
                                })?;
                            Some(source)
                        }
                    }
                };
                if let Some(source) = source {
                    if source == index {
                        return Err(FabricCompileError::ReadWriteSamePass {
                            pass: pass.id, resource: usage.key,
                        });
                    }
                    dependencies.insert(source);
                } else if !usage.access.writes()
                    && self.templates[&usage.key].lifetime == ResourceLifetime::Transient
                {
                    return Err(FabricCompileError::MissingProducer(usage.key));
                }
                if usage.access.writes() && !successors.insert((usage.key, source)) {
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
                    .iter().copied()
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
            if !needed.contains(&(binding.key, binding.frame))
                || template.lifetime == ResourceLifetime::Transient
            {
                return Err(FabricCompileError::UnexpectedBinding {
                    resource: binding.key,
                    frame: binding.frame,
                });
            }
        }
        for &binding in external {
            let address = (binding.key, binding.frame);
            if supplied.insert(address, binding).is_some() {
                return Err(FabricCompileError::DuplicateBinding {
                    resource: binding.key,
                    frame: binding.frame,
                });
            }
        }
        let mut graph = GpuFrameGraphBuilder::with_limits(limits);
        let mut resources: Vec<BoundResource> = Vec::new();
        let mut ids = BTreeMap::new();
        for (key, resource_frame) in needed {
            let template = *self
                .templates
                .get(&key)
                .ok_or(FabricCompileError::MissingTemplate(key))?;
            let history = resource_frame != frame;
            let binding = if template.lifetime == ResourceLifetime::Transient {
                None
            } else {
                let binding = supplied.get(&(key, resource_frame)).copied().ok_or(
                    FabricCompileError::MissingBinding {
                        resource: key,
                        frame: resource_frame,
                    },
                )?;
                if !binding.matches(template) {
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
                for previous in &resources {
                    if let Some(other) = previous.external
                        && binding.overlaps(other)
                    {
                        return Err(FabricCompileError::OverlappingBindings {
                            first: other.key,
                            first_frame: other.frame,
                            second: key,
                            second_frame: resource_frame,
                        });
                    }
                }
                Some(binding)
            };
            let descriptor = ResourceDescriptor {
                name: format!("{:?}@{}", key, resource_frame).into(),
                lifetime: if history {
                    ResourceLifetime::Imported
                } else {
                    template.lifetime
                },
                size: template.layout.size().expect("validated layout"),
                alignment: template.alignment,
            };
            let id = graph.try_add_resource(descriptor.clone())?;
            ids.insert((key, resource_frame), id);
            resources.push(BoundResource {
                id,
                key,
                frame: resource_frame,
                template,
                descriptor,
                external: binding,
            });
        }
        let admitted_indices: BTreeMap<_, _> = admission_order.iter().enumerate()
            .map(|(index, &original)| (original, PassId(index as u32)))
            .collect();
        let mut passes = Vec::new();
        for &admitted in &admission_order {
            let (feature, variant, template) = declarations[admitted];
            let accesses = template
                .accesses
                .iter()
                .map(|usage| {
                    Ok(ResourceAccess {
                        resource: ids
                            .get(&(usage.key, use_frame(frame, *usage)?))
                            .copied()
                            .ok_or(FabricCompileError::MissingTemplate(usage.key))?,
                        kind: usage.access,
                    })
                })
                .collect::<Result<Vec<_>, FabricCompileError>>()?;
            let descriptor = PassDescriptor {
                name: format!("feature-pass:{}", template.id.0).into(),
                queue: template.queue,
                accesses,
                depends_on: prerequisites[admitted].iter()
                    .map(|dependency| admitted_indices[dependency]).collect(),
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
            for (ordinal, usage) in pass.template.accesses.iter().enumerate() {
                if usage.access.writes() {
                    writer_accesses.insert((pass.key(), usage.key), (pass.id, ordinal));
                }
            }
        }
        let inputs = admission_order
            .iter()
            .map(|&original| {
                sources[original]
                    .iter()
                    .zip(declarations[original].2.accesses)
                    .map(|(source, usage)| {
                        let Some(source) = source else {
                            return Ok(AccessInput::Version(0));
                        };
                        let producer = declarations[*source].2.id;
                        let &(pass, access) = writer_accesses.get(&(producer, usage.key)).ok_or(
                            FabricCompileError::InvalidProducer {
                                pass: declarations[original].2.id,
                                resource: usage.key,
                                producer,
                            },
                        )?;
                        Ok(AccessInput::ProducedBy { pass, access })
                    })
                    .collect::<Result<Vec<_>, FabricCompileError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = graph.compile_with_sources(&inputs)?;
        let mut versions = Vec::new();
        for &pass_index in &plan.order {
            let pass = &passes[pass_index];
            for (&ordinal, access) in access_orders[pass_index].iter().zip(&plan.accesses[pass_index]) {
                let Some(ssa_version) = access.output_version else { continue };
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
            let mut authority = vec![None; pass.template.accesses.len()];
            for (&ordinal, access) in access_orders[index].iter().zip(&plan.accesses[index]) {
                authority[ordinal] = Some(*access);
            }
            pass.bound_accesses = pass
                .template
                .accesses
                .iter()
                .zip(&pass.descriptor.accesses)
                .zip(authority)
                .map(|((&declaration, access), authority)| {
                    let authority = authority.expect("authority preserves admitted accesses");
                    let resource = &resources[access.resource.0 as usize];
                    let version = authority.output_version.unwrap_or(authority.input_version);
                    let output = outputs.get(&(access.resource.0, version));
                    let input = outputs.get(&(access.resource.0, authority.input_version));
                    BoundResourceUse {
                        declaration,
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
            items: Vec::new(),
        };
        compiled.barriers_by_destination = vec![Vec::new(); compiled.plan.declared_pass_count];
        for (index, barrier) in compiled.barriers.iter().enumerate() {
            compiled.barriers_by_destination[barrier.destination.0 as usize].push(index);
        }
        for item in items {
            self.validate_payload(&compiled, item)?;
        }
        compiled.items.extend_from_slice(items);
        Ok(compiled)
    }

    fn validate_payload(
        &self,
        graph: &CompiledFeatureGraph,
        item: &FrameRenderItem,
    ) -> Result<(), FabricCompileError> {
        if let DomainPayload::Resource(payload) = item.payload {
            // A current-frame domain payload must actually occur in a pass, not
            // merely be present in the resource declaration table/history map.
            let id = graph
                .resource_id(payload.resource, graph.frame)
                .ok_or(FabricCompileError::PayloadNotReferenced(payload.resource))?;
            if !graph.passes.iter().any(|pass| {
                pass.descriptor
                    .accesses
                    .iter()
                    .any(|access| access.resource == id)
            }) {
                return Err(FabricCompileError::PayloadNotReferenced(payload.resource));
            }
            let template = *self
                .templates
                .get(&payload.resource)
                .ok_or(FabricCompileError::MissingTemplate(payload.resource))?;
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
fn normalized_access_orders(passes: &[PassSpec], resource_count: usize) -> Vec<Vec<usize>> {
    let mut textures = vec![false; resource_count];
    for access in passes.iter().flat_map(|pass| &pass.descriptor.accesses) {
        if matches!(access.kind, AccessKind::SampledRead | AccessKind::ColorAttachment | AccessKind::DepthAttachment) {
            textures[access.resource.0 as usize] = true;
        }
    }
    passes.iter().map(|pass| {
        pass.descriptor.accesses.iter().enumerate()
            .filter(|(_, access)| textures[access.resource.0 as usize])
            .chain(pass.descriptor.accesses.iter().enumerate()
                .filter(|(_, access)| !textures[access.resource.0 as usize]))
            .map(|(ordinal, _)| ordinal).collect()
    }).collect()
}

#[derive(Clone, Debug)]
pub struct BoundResource {
    pub id: ResourceId,
    pub key: ResourceKey,
    pub frame: u64,
    pub template: ResourceTemplate,
    pub descriptor: ResourceDescriptor,
    pub external: Option<FrameResourceBinding>,
}

/// The only bound pass representation; constructed from a selected template.
#[derive(Clone, Debug)]
pub struct PassSpec {
    id: PassId,
    feature: FeatureId,
    variant: VariantId,
    domain: Option<RenderDomain>,
    template: &'static PassTemplate,
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
}

/// Read provenance or write output from the compiled authority, with the frame
/// and external content version retained for backend/history binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundResourceUse {
    pub declaration: ResourceUse,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameResourceVersion {
    pub key: ResourceKey,
    pub frame: u64,
    pub resource: ResourceId,
    pub writer: PassKey,
    pub ssa_version: u32,
    pub content_version: u32,
}

#[derive(Debug)]
pub struct CompiledFeatureGraph {
    frame: u64,
    limits: GraphLimits,
    resources: Vec<BoundResource>,
    ids: BTreeMap<(ResourceKey, u64), ResourceId>,
    passes: Vec<PassSpec>,
    versions: Vec<FrameResourceVersion>,
    plan: ExecutionPlan,
    barriers: Vec<Barrier>,
    barriers_by_destination: Vec<Vec<usize>>,
    items: Vec<FrameRenderItem>,
}
impl CompiledFeatureGraph {
    pub const fn frame(&self) -> u64 {
        self.frame
    }
    pub(super) const fn limits(&self) -> GraphLimits {
        self.limits
    }
    pub fn resource_id(&self, key: ResourceKey, frame: u64) -> Option<ResourceId> {
        self.ids.get(&(key, frame)).copied()
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
    let mut resource_is_texture = vec![false; resource_count];
    for access in passes
        .iter()
        .flat_map(|pass| pass.descriptor.accesses.iter())
    {
        if matches!(
            access.kind,
            AccessKind::SampledRead | AccessKind::ColorAttachment | AccessKind::DepthAttachment
        ) {
            resource_is_texture[access.resource.0 as usize] = true;
        }
    }
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
