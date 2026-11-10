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
    fallback_material_record, GpuMaterialTexture, MaterialGraph, MaterialRecord,
    MaterialRenderClass, MaterialVtSourceSnapshot, ProceduralVtBakeConfig, ShadingPlan,
};

use crate::{
    author_node_identity, lower_procedural_graph, AuthorNodeIdentity, AuthoringError, Document,
    ProceduralGraphError, RuntimeSurface,
};

/// One immutable source and its ABI row for an authored MaterialX image.
#[derive(Clone, Debug)]
pub struct MaterialVtAuthoringBinding {
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
    ) -> Self {
        self.procedural_graph = Some(graph.into());
        self.procedural_vt = Some(config);
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
    /// Image sources and their exact GPU rows, ordered by node name.
    pub textures: Vec<MaterialVtAuthoringBinding>,
    /// Direct author-node identities present in the document.
    pub author_nodes: Vec<AuthorNodeIdentity>,
    /// Blend/raster family selected by authoring.
    pub render_class: MaterialRenderClass,
    /// Runtime object domain selected by authoring.
    pub render_domain: prism_render_material::RenderDomain,
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
        let mut textures = BTreeMap::<String, MaterialVtAuthoringBinding>::new();
        for binding in bindings {
            if textures.contains_key(&binding.node) {
                return Err(MaterialArtifactError::DuplicateTextureNode(binding.node));
            }
            let Some(node) = document
                .nodegraphs
                .iter()
                .flat_map(|graph| graph.nodes.iter())
                .find(|node| node.name == binding.node)
            else {
                return Err(MaterialArtifactError::InvalidTextureNode(binding.node));
            };
            if node.category != "image" {
                return Err(MaterialArtifactError::InvalidTextureNode(binding.node));
            }
            let descriptor = binding.source.descriptor();
            if descriptor.texture_id() == prism_render_material::NO_VIRTUAL_TEXTURE
                || descriptor.width() == 0
                || descriptor.height() == 0
                || descriptor.page_size() == 0
            {
                return Err(MaterialArtifactError::InvalidTextureSource(binding.node));
            }
            textures.insert(binding.node.clone(), binding);
        }

        // Admit every authored image to the surface adapter first. This lets
        // the adapter preserve the texture connection and lets this builder
        // report a missing binding explicitly instead of collapsing it into a
        // generic unsafe-connection diagnostic.
        let texture_semantics = document
            .nodegraph_images()
            .map(|node| {
                (
                    node.name.clone(),
                    textures
                        .get(&node.name)
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
            let Some(binding) = textures.get(&input.node) else {
                return Err(MaterialArtifactError::MissingTextureBinding(
                    input.node.clone(),
                ));
            };
            let expected = expected_semantic(&input.field);
            if binding.semantic != expected {
                return Err(MaterialArtifactError::TextureSemanticMismatch {
                    field: input.field.clone(),
                    node: input.node.clone(),
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
        let procedural_source = match (procedural_graph.as_ref(), request.procedural_vt) {
            (Some(graph), Some(config)) => Some(
                prism_render_material::bake_procedural_vt(graph, config).map_err(|error| {
                    MaterialArtifactError::Procedural(ProceduralGraphError::Bake(error))
                })?,
            ),
            (None, Some(_)) => return Err(MaterialArtifactError::MissingProceduralGraph),
            _ => None,
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
            textures: textures.into_values().collect(),
            author_nodes,
            render_class: request.render_class,
            render_domain: request.render_domain,
        })
    }

    /// Returns the exact VT rows to publish, in deterministic node order.
    #[must_use]
    pub fn texture_rows(&self) -> Vec<GpuMaterialTexture> {
        self.textures
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
            .collect()
    }

    /// Materialize the existing runtime record ABI without inventing a second
    /// record layout. The caller owns handle allocation and revision policy.
    #[must_use]
    pub fn material_record(&self, handle: GenerationalHandle, revision: u64) -> MaterialRecord {
        let mut record = fallback_material_record(handle, revision);
        record.response_plan = self.plan.clone();
        record.render_class = self.render_class;
        record.render_domain = self.render_domain;
        record.surface = self.surface.parameters;
        record.textures = self.texture_rows();
        record
    }
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
