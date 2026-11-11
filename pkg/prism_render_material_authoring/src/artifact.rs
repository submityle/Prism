//! The authoring-to-runtime material artifact seam.
//!
//! This module deliberately composes the existing material contracts instead
//! of defining a second surface or VT ABI. A successful artifact contains the
//! loss-aware OpenPBR import, the optional lowered graph, immutable VT source
//! snapshots and the exact `GpuMaterialTexture` rows that the renderer can
//! publish. No unresolved image or procedural input is turned into a default.

use std::collections::BTreeMap;
use std::fmt;

use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{
    fallback_material_record, ClosureKind, GpuMaterialTexture, MaterialFeatureFlags, MaterialGraph,
    MaterialRecord, MaterialRenderClass, MaterialValidationError, MaterialVtSourceSnapshot,
    NormalizedMaterial, ProceduralVtBakeConfig, ShadingPlan,
};

use crate::import::TextureNodeKey;
use crate::{
    author_node_identity, lower_procedural_graph, AuthorNodeIdentity, AuthoringError, Document,
    ProceduralGraphError, RuntimeSurface,
};

/// One immutable source and its ABI row for an authored MaterialX image.
#[derive(Clone, Debug)]
pub struct MaterialVtAuthoringBinding {
    /// MaterialX nodegraph that owns the image node.
    pub graph: String,
    /// MaterialX image node name used by surface inputs.
    pub node: String,
    /// Existing material texture semantic (`0` = base color, `1` = emissive,
    /// `2` = metallic/roughness, and so on as defined by the material shader).
    pub semantic: u32,
    /// Bindless image index assigned by the resource publisher.
    pub bindless_index: u32,
    /// Generation paired with `bindless_index`.
    pub generation: u32,
    /// Bindless sampler index paired with the image.
    pub sampler_index: u32,
    /// Complete immutable source pyramid consumed by the Material-VT producer.
    pub source: MaterialVtSourceSnapshot,
}

/// Explicit runtime publication metadata for a procedural VT source.
///
/// The bake produces immutable source bytes, but it cannot invent a bindless
/// slot, generation, sampler, or shader semantic. Those values therefore have
/// to be supplied by the asset/resource publisher before an artifact is
/// publishable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVtPublication {
    /// Stable material texture semantic consumed by the shader.
    pub semantic: u32,
    /// Bindless image index assigned by the resource publisher.
    pub bindless_index: u32,
    /// Generation paired with `bindless_index`.
    pub generation: u32,
    /// Bindless sampler index paired with the image.
    pub sampler_index: u32,
}

/// Authoring request for one material artifact.
#[derive(Clone, Debug)]
pub struct MaterialArtifactRequest {
    /// Nodegraph containing the OpenPBR or Standard Surface node.
    pub surface_graph: String,
    /// Surface node to import.
    pub surface_node: String,
    /// Optional graph to lower as a procedural VT source.
    pub procedural_graph: Option<String>,
    /// Optional bake configuration for `procedural_graph`.
    pub procedural_vt: Option<ProceduralVtBakeConfig>,
    /// Explicit runtime publication metadata for the procedural source.
    pub procedural_publication: Option<MaterialVtPublication>,
    /// Response/composition plan selected by authoring.
    pub plan: ShadingPlan,
    /// Blend/raster family selected by authoring.
    pub render_class: MaterialRenderClass,
    /// Runtime object domain selected by authoring.
    pub render_domain: prism_render_material::RenderDomain,
}

impl MaterialArtifactRequest {
    /// Creates a surface-only request with the canonical Lit response.
    #[must_use]
    pub fn surface(surface_graph: impl Into<String>, surface_node: impl Into<String>) -> Self {
        Self {
            surface_graph: surface_graph.into(),
            surface_node: surface_node.into(),
            procedural_graph: None,
            procedural_vt: None,
            procedural_publication: None,
            plan: ShadingPlan::default(),
            render_class: MaterialRenderClass::default(),
            render_domain: prism_render_material::RenderDomain::Surface,
        }
    }

    /// Adds a procedural graph and the exact source dimensions used to bake it.
    #[must_use]
    pub fn with_procedural_vt(
        mut self,
        graph: impl Into<String>,
        config: ProceduralVtBakeConfig,
        publication: MaterialVtPublication,
    ) -> Self {
        self.procedural_graph = Some(graph.into());
        self.procedural_vt = Some(config);
        self.procedural_publication = Some(publication);
        self
    }
}

/// Failure while assembling an authoring artifact.
#[derive(Clone, Debug, PartialEq)]
pub enum MaterialArtifactError {
    /// The surface or graph reference is invalid.
    Authoring(AuthoringError),
    /// The procedural graph cannot be lowered into the existing runtime graph.
    Procedural(ProceduralGraphError),
    /// An image binding names no MaterialX image node.
    InvalidTextureNode(String),
    /// Two bindings publish the same MaterialX node.
    DuplicateTextureNode(String),
    /// A texture binding uses a reserved or otherwise invalid source identity.
    InvalidTextureSource(String),
    /// The surface field and bound texture use different runtime semantics.
    TextureSemanticMismatch {
        /// Surface field that requested a texture.
        field: String,
        /// Image node supplying it.
        node: String,
        /// Semantic supplied by the binding.
        actual: u32,
        /// Semantic required by the production material shader.
        expected: u32,
    },
    /// A texture input was authored but no binding was supplied for its node.
    MissingTextureBinding(String),
    /// A procedural bake was requested without a graph name.
    MissingProceduralGraph,
    /// A procedural bake was requested without explicit runtime publication
    /// metadata.
    MissingProceduralPublication,
    /// The requested revision cannot be represented by the runtime ABI.
    RevisionOverflow(u64),
    /// The lowered procedural graph cannot be admitted by the runtime graph
    /// validator without changing its semantics.
    RuntimeValidation(MaterialValidationError),
}

impl fmt::Display for MaterialArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authoring(error) => error.fmt(formatter),
            Self::Procedural(error) => error.fmt(formatter),
            Self::InvalidTextureNode(node) => {
                write!(formatter, "material texture binding names invalid node '{node}'")
            }
            Self::DuplicateTextureNode(node) => {
                write!(formatter, "material texture node '{node}' is bound more than once")
            }
            Self::InvalidTextureSource(node) => {
                write!(formatter, "material texture source for '{node}' is invalid")
            }
            Self::TextureSemanticMismatch {
                field,
                node,
                actual,
                expected,
            } => write!(
                formatter,
                "texture node '{node}' supplies semantic {actual} for field '{field}', expected {expected}"
            ),
            Self::MissingTextureBinding(node) => {
                write!(formatter, "surface texture node '{node}' has no VT binding")
            }
            Self::MissingProceduralGraph => {
                formatter.write_str("a procedural VT bake requires a procedural graph")
            }
            Self::MissingProceduralPublication => formatter.write_str(
                "a procedural VT bake requires explicit semantic, bindless index, generation, and sampler publication metadata",
            ),
            Self::RevisionOverflow(revision) => write!(
                formatter,
                "material revision {revision} exceeds the runtime u32 ABI",
            ),
            Self::RuntimeValidation(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for MaterialArtifactError {}

impl From<AuthoringError> for MaterialArtifactError {
    fn from(error: AuthoringError) -> Self {
        Self::Authoring(error)
    }
}

impl From<ProceduralGraphError> for MaterialArtifactError {
    fn from(error: ProceduralGraphError) -> Self {
        Self::Procedural(error)
    }
}

/// A validated authoring output built on the existing Prism runtime contracts.
#[derive(Clone, Debug)]
pub struct MaterialArtifact {
    /// Loss-aware OpenPBR/Standard Surface result.
    pub surface: RuntimeSurface,
    /// Author-selected response and raster/domain state.
    pub plan: ShadingPlan,
    /// Optional lowered graph retained for offline inspection and bake parity.
    pub procedural_graph: Option<MaterialGraph>,
    /// Optional baked procedural VT source.
    pub procedural_source: Option<MaterialVtSourceSnapshot>,
    /// Explicit publication metadata for `procedural_source`.
    pub procedural_publication: Option<MaterialVtPublication>,
    /// Image sources and their exact GPU rows, ordered by node name.
    pub textures: Vec<MaterialVtAuthoringBinding>,
    /// Direct author-node identities present in the document.
    pub author_nodes: Vec<AuthorNodeIdentity>,
    /// Blend/raster family selected by authoring.
    pub render_class: MaterialRenderClass,
    /// Runtime object domain selected by authoring.
    pub render_domain: prism_render_material::RenderDomain,
}

/// The complete authoring-to-runtime response produced for one artifact.
///
/// `record` is the existing publication ABI. `procedural` is the validated,
/// live-pruned graph for callers that compile or inspect the procedural
/// response before publishing its baked VT row. Keeping both values together
/// prevents a caller from publishing a record while silently skipping graph
/// validation.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialRuntimeResponse {
    /// Existing runtime material record, including exact texture rows.
    pub record: MaterialRecord,
    /// Validated procedural response, when the artifact contains one.
    pub procedural: Option<NormalizedMaterial>,
}

impl MaterialArtifact {
    /// Build an artifact from a parsed MaterialX document and immutable source
    /// bindings. The source document is validated before any runtime output is
    /// returned, so callers never receive a partially publishable artifact.
    pub fn build(
        document: &Document,
        request: MaterialArtifactRequest,
        bindings: impl IntoIterator<Item = MaterialVtAuthoringBinding>,
    ) -> Result<Self, MaterialArtifactError> {
        let mut textures = BTreeMap::<TextureNodeKey, MaterialVtAuthoringBinding>::new();
        for binding in bindings {
            let key = TextureNodeKey {
                graph: binding.graph.clone(),
                node: binding.node.clone(),
            };
            if textures.contains_key(&key) {
                return Err(MaterialArtifactError::DuplicateTextureNode(
                    qualified_texture_node(&key),
                ));
            }
            let Some(graph) = document
                .nodegraphs
                .iter()
                .find(|graph| graph.name == binding.graph)
            else {
                return Err(MaterialArtifactError::InvalidTextureNode(
                    qualified_texture_node(&key),
                ));
            };
            let Some(node) = graph.nodes.iter().find(|node| node.name == binding.node) else {
                return Err(MaterialArtifactError::InvalidTextureNode(
                    qualified_texture_node(&key),
                ));
            };
            if node.category != "image" {
                return Err(MaterialArtifactError::InvalidTextureNode(
                    qualified_texture_node(&key),
                ));
            }
            let descriptor = binding.source.descriptor();
            if descriptor.texture_id() == prism_render_material::NO_VIRTUAL_TEXTURE
                || descriptor.width() == 0
                || descriptor.height() == 0
                || descriptor.page_size() == 0
            {
                return Err(MaterialArtifactError::InvalidTextureSource(
                    qualified_texture_node(&key),
                ));
            }
            textures.insert(key, binding);
        }

        // Admit every authored image to the surface adapter first. This lets
        // the adapter preserve the texture connection and lets this builder
        // report a missing binding explicitly instead of collapsing it into a
        // generic unsafe-connection diagnostic.
        let texture_semantics = document
            .nodegraph_images()
            .map(|(graph, node)| {
                (
                    TextureNodeKey {
                        graph: graph.name.clone(),
                        node: node.name.clone(),
                    },
                    textures
                        .get(&TextureNodeKey {
                            graph: graph.name.clone(),
                            node: node.name.clone(),
                        })
                        .map_or(u32::MAX, |binding| binding.semantic),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let surface = crate::import::adapt_openpbr_surface_with_textures(
            document,
            &request.surface_graph,
            &request.surface_node,
            &texture_semantics,
        )?;
        for input in &surface.texture_inputs {
            let key = TextureNodeKey {
                graph: input.graph.clone(),
                node: input.node.clone(),
            };
            let Some(binding) = textures.get(&key) else {
                return Err(MaterialArtifactError::MissingTextureBinding(
                    qualified_texture_node(&key),
                ));
            };
            let expected = expected_semantic(&input.field);
            if binding.semantic != expected {
                return Err(MaterialArtifactError::TextureSemanticMismatch {
                    field: input.field.clone(),
                    node: qualified_texture_node(&key),
                    actual: binding.semantic,
                    expected,
                });
            }
        }

        let procedural_graph = request
            .procedural_graph
            .as_deref()
            .map(|name| {
                let graph = document
                    .nodegraphs
                    .iter()
                    .find(|graph| graph.name == name)
                    .ok_or_else(|| {
                        AuthoringError::new(
                            crate::DiagnosticCode::InvalidReference,
                            format!("nodegraph '{name}' does not exist"),
                            None,
                        )
                    })?;
                lower_procedural_graph(document, graph).map_err(MaterialArtifactError::from)
            })
            .transpose()?;
        let procedural_source = match (
            procedural_graph.as_ref(),
            request.procedural_vt,
            request.procedural_publication,
        ) {
            (Some(graph), Some(config), Some(_publication)) => Some(
                prism_render_material::bake_procedural_vt(graph, config).map_err(|error| {
                    MaterialArtifactError::Procedural(ProceduralGraphError::Bake(error))
                })?,
            ),
            (None, Some(_), _) => return Err(MaterialArtifactError::MissingProceduralGraph),
            (Some(_), Some(_), None) => {
                return Err(MaterialArtifactError::MissingProceduralPublication)
            }
            (Some(_), None, Some(_)) => {
                return Err(MaterialArtifactError::MissingProceduralPublication)
            }
            (None, None, Some(_)) => return Err(MaterialArtifactError::MissingProceduralGraph),
            (Some(_), None, None) | (None, None, None) => None,
        };

        let mut author_nodes = document
            .nodegraphs
            .iter()
            .flat_map(|graph| graph.nodes.iter())
            .filter_map(|node| author_node_identity(document, node).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        author_nodes.sort_by(|left, right| {
            left.nodedef
                .cmp(&right.nodedef)
                .then(left.pcm_uid.cmp(&right.pcm_uid))
        });

        Ok(Self {
            surface,
            plan: request.plan,
            procedural_graph,
            procedural_source,
            procedural_publication: request.procedural_publication,
            textures: textures.into_values().collect(),
            author_nodes,
            render_class: request.render_class,
            render_domain: request.render_domain,
        })
    }

    /// Returns the exact VT rows to publish, in deterministic node order.
    #[must_use]
    pub fn texture_rows(&self) -> Vec<GpuMaterialTexture> {
        let mut rows = self
            .textures
            .iter()
            .map(|binding| {
                binding
                    .source
                    .texture_row(
                        binding.bindless_index,
                        binding.generation,
                        binding.semantic,
                        binding.sampler_index,
                    )
                    .gpu
            })
            .collect::<Vec<_>>();
        if let (Some(source), Some(publication)) =
            (self.procedural_source.as_ref(), self.procedural_publication)
        {
            rows.push(
                source
                    .texture_row(
                        publication.bindless_index,
                        publication.generation,
                        publication.semantic,
                        publication.sampler_index,
                    )
                    .gpu,
            );
        }
        rows
    }

    /// Materialize the existing runtime record ABI without inventing a second
    /// record layout. The caller owns handle allocation and revision policy.
    #[must_use]
    pub fn material_record(
        &self,
        handle: GenerationalHandle,
        revision: u64,
    ) -> Result<MaterialRecord, MaterialArtifactError> {
        let revision = u32::try_from(revision)
            .map_err(|_| MaterialArtifactError::RevisionOverflow(revision))?;
        let mut record = fallback_material_record(handle, u64::from(revision));
        record.response_plan = self.plan.clone();
        record.render_class = self.render_class;
        record.render_domain = self.render_domain;
        record.surface = self.surface.parameters;
        record.closure_mask = closure_mask(record.surface);
        if record.surface.emission_luminance != 0.0 {
            record.features |= MaterialFeatureFlags::EMISSIVE;
        }
        if record.surface.transmission_weight != 0.0 {
            record.features |= MaterialFeatureFlags::TRANSMISSION;
        }
        record.textures = self.texture_rows();
        Ok(record)
    }

    /// Compile the artifact into the existing runtime record plus an optional
    /// normalized procedural response. Every requested runtime-facing piece is
    /// validated before the response is returned.
    pub fn compile_runtime_response(
        &self,
        handle: GenerationalHandle,
        revision: u64,
    ) -> Result<MaterialRuntimeResponse, MaterialArtifactError> {
        let record = self.material_record(handle, revision)?;
        let procedural = self
            .procedural_graph
            .as_ref()
            .map(MaterialGraph::normalize)
            .transpose()
            .map_err(MaterialArtifactError::RuntimeValidation)?;
        Ok(MaterialRuntimeResponse { record, procedural })
    }

    /// Converts the adapted surface to the independent path-traced reference
    /// contract, preserving non-BRDF outputs beside the OpenPBR closure.
    pub fn reference_material(
        &self,
    ) -> Result<crate::ReferenceMaterial, crate::ReferenceMaterialError> {
        crate::reference_material(&self.surface)
    }

    /// Converts the artifact's surface with explicit shade-time glint data.
    pub fn reference_material_with_context(
        &self,
        context: crate::ReferenceShadeContext,
    ) -> Result<crate::ReferenceMaterial, crate::ReferenceMaterialError> {
        crate::reference_material_with_context(&self.surface, context)
    }
}

fn closure_mask(surface: prism_render_material::GpuSurfaceParameters) -> u32 {
    let mut mask = 1 << ClosureKind::Diffuse as u32;
    if surface.emission_luminance != 0.0 {
        mask |= 1 << ClosureKind::Emission as u32;
    }
    if surface.coat_weight != 0.0 {
        mask |= 1 << ClosureKind::ClearCoat as u32;
    }
    if surface.fuzz_weight != 0.0 {
        mask |= 1 << ClosureKind::Sheen as u32;
    }
    if surface.subsurface_weight != 0.0 {
        mask |= 1 << ClosureKind::Subsurface as u32;
    }
    if surface.transmission_weight != 0.0 {
        mask |= 1 << ClosureKind::Transmission as u32;
    }
    mask
}

fn qualified_texture_node(key: &TextureNodeKey) -> String {
    format!("{}::{}", key.graph, key.node)
}

fn expected_semantic(field: &str) -> u32 {
    match field {
        "base_color"
        | "specular_color"
        | "coat_color"
        | "fuzz_color"
        | "subsurface_color"
        | "subsurface_radius_scale"
        | "transmission_color"
        | "transmission_scatter" => 0,
        "emission_color" => 1,
        "base_metalness" | "specular_roughness" => 2,
        "normal_scale" => 3,
        "ambient_occlusion" => 4,
        "coat_weight" => 5,
        "coat_roughness" => 6,
        _ => u32::MAX,
    }
}
