use std::collections::BTreeMap;

use prism_render_material::GpuSurfaceParameters;

use crate::{AuthoringError, Diagnostic, DiagnosticCode, Document, Severity, TypedValue};

/// Runtime parameters and an explicit account of conversion losses.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeSurface {
    /// Source node category (`open_pbr_surface` or `standard_surface`).
    pub source_category: String,
    /// Canonical Prism runtime parameter block.
    pub parameters: GpuSurfaceParameters,
    /// MaterialX image/normal-map inputs retained as explicit texture
    /// bindings. The numeric adapter never turns a texture into a default
    /// scalar; artifact construction resolves these names to VT rows.
    pub texture_inputs: Vec<RuntimeTextureInput>,
    /// Fields applied, fields not represented, and warnings.
    pub report: AdapterReport,
}

/// One authored surface field whose value is supplied by a material texture
/// row instead of a scalar parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeTextureInput {
    /// Canonical Prism surface field name.
    pub field: String,
    /// MaterialX image or normal-map node name.
    pub node: String,
}

/// Loss report for a successful MaterialX-to-runtime conversion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdapterReport {
    /// Canonical Prism parameter names recognized on this source node.
    pub supported_fields: Vec<String>,
    /// Canonical Prism parameter names with an explicit literal or supported
    /// constant connection. Supported but omitted inputs keep OpenPBR defaults.
    pub applied_fields: Vec<String>,
    /// Canonical fields supplied by an explicit texture row.
    pub textured_fields: Vec<String>,
    /// Texture bindings retained during an artifact-aware adaptation.
    pub texture_inputs: Vec<RuntimeTextureInput>,
    /// Authored source fields without a runtime mapping.
    pub unsupported_fields: Vec<String>,
    /// Non-fatal diagnostics for unsupported-but-preserved source semantics.
    pub diagnostics: Vec<Diagnostic>,
}

/// Adapt one OpenPBR or MaterialX Standard Surface node to Prism's canonical
/// runtime parameter block.
///
/// Literals, graph interface inputs, graph outputs, and the bounded
/// pure-expression subset implemented by [`crate::evaluate_connection`] are
/// evaluated. Texture and procedural nodes still fail with an explicit
/// diagnostic; callers must bake or evaluate those graphs rather than silently
/// accepting defaults.
pub fn adapt_openpbr_surface(
    document: &Document,
    nodegraph: &str,
    node_name: &str,
) -> Result<RuntimeSurface, AuthoringError> {
    adapt_surface(document, nodegraph, node_name, None)
}

/// Adapt a surface while retaining explicitly bound image inputs.
///
/// `texture_nodes` is an authoring-time binding map. Only image/normal-map
/// connections found in that map are accepted; every other unresolved
/// connection keeps the normal structured error from [`adapt_openpbr_surface`].
pub(crate) fn adapt_openpbr_surface_with_textures(
    document: &Document,
    nodegraph: &str,
    node_name: &str,
    texture_nodes: &BTreeMap<String, u32>,
) -> Result<RuntimeSurface, AuthoringError> {
    adapt_surface(document, nodegraph, node_name, Some(texture_nodes))
}

fn adapt_surface(
    document: &Document,
    nodegraph: &str,
    node_name: &str,
    texture_nodes: Option<&BTreeMap<String, u32>>,
) -> Result<RuntimeSurface, AuthoringError> {
    let graph = document
        .nodegraphs
        .iter()
        .find(|graph| graph.name == nodegraph)
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("nodegraph '{nodegraph}' does not exist"),
            )
        })?;
    let node = graph
        .nodes
        .iter()
        .find(|node| node.name == node_name)
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("node '{node_name}' does not exist in nodegraph '{nodegraph}'"),
            )
        })?;
    if !matches!(
        node.category.as_str(),
        "open_pbr_surface" | "standard_surface"
    ) {
        return Err(error(
            DiagnosticCode::UnsupportedMaterial,
            format!(
                "node '{node_name}' has unsupported material category '{}'",
                node.category
            ),
        ));
    }

    let aliases = field_aliases(&node.category);
    let evaluation_index = crate::evaluator::EvaluationIndex::new(document);
    let mut parameters = GpuSurfaceParameters::default();
    let mut report = AdapterReport::default();
    let mut supplied_fields = std::collections::BTreeSet::new();
    for input in &node.inputs {
        let Some(field) = aliases.get(input.name.as_str()).copied() else {
            report.unsupported_fields.push(input.name.clone());
            continue;
        };
        report.supported_fields.push(field.to_owned());
        let value = match resolve_input(&evaluation_index, document, graph, input) {
            Ok(value) => value,
            Err(error) => {
                let Some(texture_node) = texture_source_node(document, graph, input) else {
                    return Err(error);
                };
                let Some(texture_nodes) = texture_nodes else {
                    return Err(error);
                };
                if !texture_nodes.contains_key(&texture_node) {
                    return Err(error);
                }
                if !texture_field_accepts(field) {
                    return Err(error);
                }
                report.supported_fields.push(field.to_owned());
                report.textured_fields.push(field.to_owned());
                report.texture_inputs.push(RuntimeTextureInput {
                    field: field.to_owned(),
                    node: texture_node,
                });
                continue;
            }
        };
        if let Some(value) = value {
            if let (Some(input_type), Some(source_type)) =
                (input.type_name.as_deref(), value_type(&value))
                && source_type != input_type
            {
                return Err(error(
                    DiagnosticCode::InvalidValue,
                    format!(
                        "input '{}' declares type '{input_type}' but its value is '{source_type}'",
                        input.name
                    ),
                ));
            }
            validate_field_type(field, input.type_name.as_deref(), &value.value)?;
            if !supplied_fields.insert(field) {
                return Err(error(
                    DiagnosticCode::InvalidStructure,
                    format!("multiple inputs provide runtime field '{field}'"),
                ));
            }
            assign(&mut parameters, field, &value.value)?;
            report.applied_fields.push(field.to_owned());
        }
    }
    report.supported_fields.sort();
    report.supported_fields.dedup();
    report.applied_fields.sort();
    report.textured_fields.sort();
    report.textured_fields.dedup();
    report.unsupported_fields.sort();
    report.unsupported_fields.dedup();
    report.diagnostics = report
        .unsupported_fields
        .iter()
        .map(|field| Diagnostic {
            code: DiagnosticCode::UnsupportedField,
            severity: Severity::Warning,
            message: format!(
                "MaterialX field '{field}' has no Prism runtime mapping and was not applied"
            ),
            location: None,
        })
        .collect();
    Ok(RuntimeSurface {
        source_category: node.category.clone(),
        parameters,
        texture_inputs: report.texture_inputs.clone(),
        report,
    })
}

fn field_aliases(category: &str) -> BTreeMap<&'static str, &'static str> {
    let mut map = BTreeMap::new();
    for name in [
        "base_color",
        "base_metalness",
        "base_weight",
        "specular_roughness",
        "specular_weight",
        "ambient_occlusion",
        "normal_scale",
        "alpha_cutoff",
        "base_diffuse_roughness",
        "specular_color",
        "specular_ior",
        "emission_color",
        "emission_luminance",
        "coat_weight",
        "coat_roughness",
        "coat_ior",
        "coat_color",
        "coat_roughness_anisotropy",
        "coat_darkening",
        "anisotropy_rotation",
        "fuzz_weight",
        "fuzz_roughness",
        "fuzz_color",
        "subsurface_weight",
        "subsurface_color",
        "subsurface_radius",
        "subsurface_radius_scale",
        "subsurface_scatter_anisotropy",
        "index_of_refraction",
        "transmission_dispersion_scale",
        "transmission_color",
        "transmission_scatter",
        "transmission_scatter_anisotropy",
        "transmission_dispersion_abbe_number",
        "thin_film_weight",
        "thin_film_thickness",
        "thin_film_ior",
        "transmission_weight",
        "thickness",
        "geometry_opacity",
        "geometry_thin_walled",
    ] {
        map.insert(name, name);
    }
    map.insert("specular_roughness_anisotropy", "anisotropy");
    map.insert("transmission_depth", "thickness");
    if category == "standard_surface" {
        map.insert("base", "base_weight");
        map.insert("diffuse_roughness", "base_diffuse_roughness");
        map.insert("metalness", "base_metalness");
        map.insert("specular_IOR", "specular_ior");
        map.insert("specular", "specular_weight");
        map.insert("specular_anisotropy", "anisotropy");
        map.insert("specular_rotation", "anisotropy_rotation");
        map.insert("coat", "coat_weight");
        map.insert("subsurface", "subsurface_weight");
        // Standard Surface's emission input is a weight/intensity, while the
        // runtime field is OpenPBR luminance in nits. Without an explicit
        // unit/exposure conversion this is not a safe alias; leave it in the
        // unsupported-field report rather than silently changing semantics.
        map.insert("transmission", "transmission_weight");
        map.insert("thin_walled", "geometry_thin_walled");
    }
    map
}

fn resolve_input<'a>(
    index: &crate::evaluator::EvaluationIndex<'_>,
    document: &'a Document,
    graph: &'a crate::NodeGraph,
    input: &'a crate::Input,
) -> Result<Option<TypedValue>, AuthoringError> {
    crate::evaluator::evaluate_input_indexed(index, document, graph, input)
}

fn texture_field_accepts(field: &str) -> bool {
    matches!(
        field,
        "base_color"
            | "emission_color"
            | "specular_color"
            | "coat_color"
            | "fuzz_color"
            | "subsurface_color"
            | "subsurface_radius_scale"
            | "transmission_color"
            | "transmission_scatter"
            | "base_metalness"
            | "specular_roughness"
            | "ambient_occlusion"
            | "normal_scale"
            | "coat_weight"
            | "coat_roughness"
    )
}

fn texture_source_node(
    document: &Document,
    graph: &crate::NodeGraph,
    input: &crate::Input,
) -> Option<String> {
    let connection = input.connection.as_ref()?;
    let target = connection
        .nodegraph
        .as_deref()
        .and_then(|name| document.nodegraphs.iter().find(|item| item.name == name))
        .unwrap_or(graph);
    let node_name = connection.node.as_deref()?;
    let node = target.nodes.iter().find(|node| node.name == node_name)?;
    match node.category.as_str() {
        "image" => Some(node.name.clone()),
        "normalmap" => node
            .inputs
            .iter()
            .find(|input| matches!(input.name.as_str(), "in" | "input" | "normal"))
            .and_then(|input| texture_source_node(document, target, input)),
        _ => None,
    }
}

fn value_type(value: &TypedValue) -> Option<&str> {
    (value.type_name != "string").then_some(value.type_name.as_str())
}

fn validate_field_type(
    field: &str,
    declared_type: Option<&str>,
    raw: &str,
) -> Result<(), AuthoringError> {
    let Some(type_name) = declared_type else {
        return Ok(());
    };
    let valid = if field == "geometry_thin_walled" {
        matches!(type_name, "boolean" | "integer")
    } else if matches!(
        field,
        "base_color"
            | "emission_color"
            | "specular_color"
            | "coat_color"
            | "fuzz_color"
            | "subsurface_color"
            | "subsurface_radius_scale"
            | "transmission_color"
            | "transmission_scatter"
    ) {
        matches!(type_name, "color3" | "color4")
    } else {
        matches!(type_name, "float" | "integer")
    };
    if !valid {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("field '{field}' cannot consume MaterialX type '{type_name}' (value '{raw}')"),
        ));
    }
    Ok(())
}

fn assign(p: &mut GpuSurfaceParameters, field: &str, raw: &str) -> Result<(), AuthoringError> {
    match field {
        "base_color" => p.base_color = color(raw)?,
        "emission_color" => p.emission_color = color(raw)?,
        "specular_color" => p.specular_color = color(raw)?,
        "coat_color" => p.coat_color = color(raw)?,
        "fuzz_color" => p.fuzz_color = color(raw)?,
        "subsurface_color" => p.subsurface_color = color(raw)?,
        "subsurface_radius_scale" => p.subsurface_radius_scale = color(raw)?,
        "transmission_color" => p.transmission_color = color(raw)?,
        "transmission_scatter" => p.transmission_scatter = color(raw)?,
        "geometry_thin_walled" => p.thin_walled = f32::from(boolean(raw)?),
        "base_metalness" => p.base_metalness = scalar(raw, field)?,
        "specular_roughness" => p.specular_roughness = scalar(raw, field)?,
        "specular_weight" => p.specular_weight = scalar(raw, field)?,
        "ambient_occlusion" => p.ambient_occlusion = scalar(raw, field)?,
        "normal_scale" => p.normal_scale = scalar(raw, field)?,
        "alpha_cutoff" => p.alpha_cutoff = scalar(raw, field)?,
        "base_weight" => p.base_weight = scalar(raw, field)?,
        "base_diffuse_roughness" => p.base_diffuse_roughness = scalar(raw, field)?,
        "specular_ior" => p.specular_ior = scalar(raw, field)?,
        "emission_luminance" => p.emission_luminance = scalar(raw, field)?,
        "coat_weight" => p.coat_weight = scalar(raw, field)?,
        "coat_roughness" => p.coat_roughness = scalar(raw, field)?,
        "coat_ior" => p.coat_ior = scalar(raw, field)?,
        "coat_roughness_anisotropy" => p.coat_roughness_anisotropy = scalar(raw, field)?,
        "coat_darkening" => p.coat_darkening = scalar(raw, field)?,
        "anisotropy" => p.anisotropy = scalar(raw, field)?,
        "anisotropy_rotation" => p.anisotropy_rotation = scalar(raw, field)?,
        "fuzz_weight" => p.fuzz_weight = scalar(raw, field)?,
        "fuzz_roughness" => p.fuzz_roughness = scalar(raw, field)?,
        "subsurface_weight" => p.subsurface_weight = scalar(raw, field)?,
        "subsurface_radius" => p.subsurface_radius = scalar(raw, field)?,
        "subsurface_scatter_anisotropy" => p.subsurface_scatter_anisotropy = scalar(raw, field)?,
        "index_of_refraction" => p.index_of_refraction = scalar(raw, field)?,
        "transmission_dispersion_scale" => p.transmission_dispersion_scale = scalar(raw, field)?,
        "transmission_scatter_anisotropy" => {
            p.transmission_scatter_anisotropy = scalar(raw, field)?;
        }
        "transmission_dispersion_abbe_number" => {
            p.transmission_dispersion_abbe_number = scalar(raw, field)?;
        }
        "thin_film_weight" => p.thin_film_weight = scalar(raw, field)?,
        "thin_film_thickness" => p.thin_film_thickness = scalar(raw, field)?,
        "thin_film_ior" => p.thin_film_ior = scalar(raw, field)?,
        "transmission_weight" => p.transmission_weight = scalar(raw, field)?,
        "thickness" => p.thickness = scalar(raw, field)?,
        "geometry_opacity" => p.geometry_opacity = scalar(raw, field)?,
        _ => {
            return Err(error(
                DiagnosticCode::UnsupportedField,
                format!("unsupported OpenPBR field '{field}'"),
            ))
        }
    }
    Ok(())
}

fn scalar(raw: &str, field: &str) -> Result<f32, AuthoringError> {
    let values = vector(raw)?;
    if values.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("'{field}' expects one scalar, got '{raw}'"),
        ));
    }
    Ok(values[0])
}

fn color(raw: &str) -> Result<[f32; 4], AuthoringError> {
    let values = vector(raw)?;
    if !(3..=4).contains(&values.len()) {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("color expects 3 or 4 components, got '{raw}'"),
        ));
    }
    let mut result = [1.0; 4];
    result[..values.len()].copy_from_slice(&values);
    Ok(result)
}

fn vector(raw: &str) -> Result<Vec<f32>, AuthoringError> {
    raw.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let value = part.parse::<f32>().map_err(|_| {
                error(
                    DiagnosticCode::InvalidValue,
                    format!("expected numeric value, got '{raw}'"),
                )
            })?;
            if !value.is_finite() {
                return Err(error(
                    DiagnosticCode::InvalidValue,
                    format!("non-finite numeric value in '{raw}'"),
                ));
            }
            Ok(value)
        })
        .collect()
}

fn boolean(raw: &str) -> Result<bool, AuthoringError> {
    match raw.trim() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(error(
            DiagnosticCode::InvalidValue,
            format!("expected boolean, got '{raw}'"),
        )),
    }
}

fn error(code: DiagnosticCode, message: String) -> AuthoringError {
    AuthoringError::new(code, message, None)
}
