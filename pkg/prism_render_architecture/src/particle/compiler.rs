//! Deterministic `GraphIr` to executable `WGSL` assembly.
//!
//! [`EmberModule::emit_wesl`] currently emits source fragments, not semantic
//! kernels or a typed expression IR. This compiler therefore treats each
//! fragment as a `WGSL` function-body statement, wraps the ordered fragments in
//! generated compute entry points, emits declared resource bindings, and asks
//! `naga` to parse and validate the complete module. It also derives the live
//! SoA allocation plan and checks each module's required attributes against its
//! graph stage. It does not synthesize particle behavior or claim that a
//! syntactically valid module implements the declared attribute read/write
//! semantics. A fragment that refers to a missing function, resource, or local
//! is rejected by validation.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::attributes::{AttributeAccess, AttributeFormat, AttributeLayoutPlan, AttributeSemantic};
use super::authoring::{BindingSource, EmberAsset, ParamHandle, ParamType, ParamValue};
use super::graph::{self, CompileError as GraphError, GraphIr};
use super::modules::{
    DataInterface, EmberModule, ParameterBlockDecl, ParameterType, ParameterValue, ResourceAccess,
    ResourceDecl, ResourceKind,
};
use super::IterationDomain;

/// The fixed workgroup width in the generated compute-entry ABI.
pub const WORKGROUP_SIZE: u32 = 64;

/// Reserved read-only buffer carrying the current invocation count for each
/// stage. Module resources use group 2 and above.
pub const DISPATCH_COUNT_BINDING: ResourceDecl =
    ResourceDecl::buffer("ember_dispatch_counts", 1, 0);

/// A module assigned to one graph stage. The input order is significant and is
/// preserved inside that stage's entry point.
pub struct StageModule<'a> {
    /// Index into [`GraphIr::stages`].
    pub stage_index: usize,
    /// Module whose emitted function-body statements are appended to the stage.
    pub module: &'a dyn EmberModule,
}

/// A compiled, Naga-validated compute module and its deterministic schedule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledGraph {
    /// Complete WGSL source, including runtime resource bindings and
    /// one compute entry point for each graph stage.
    pub source: String,
    /// Graph stage indices in deterministic topological execution order.
    pub stage_order: Vec<usize>,
    /// Generated entry-point names in the same order as `stage_order`.
    pub entry_points: Vec<String>,
    /// Host submission records in deterministic topological order. Each record
    /// names one pipeline and gives the number of times to dispatch it.
    pub dispatches: Vec<CompiledDispatch>,
    /// The globally merged module resource declarations used by the module.
    pub bindings: Vec<ResourceDecl>,
    /// Typed, host-shareable uniform parameter layouts in binding order.
    pub parameter_blocks: Vec<CompiledParameterBlock>,
    /// The live, deterministic SoA allocation plan derived from the graph.
    pub attribute_layout: AttributeLayoutPlan,
    /// WGSL storage bindings for live graph attributes, in binding order.
    pub attribute_bindings: Vec<CompiledAttributeBinding>,
}

/// One parameter member's verified host/WGSL layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledParameterField {
    /// WGSL struct member name.
    pub name: String,
    /// Exact WGSL/host scalar or vector type.
    pub ty: ParameterType,
    /// Byte offset from the start of the uniform block.
    pub offset: u32,
    /// Default payload bytes for this field (without inter-field padding).
    pub default_bytes: Vec<u8>,
}

/// Runtime upload contract for one typed uniform resource.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledParameterBlock {
    /// Resource binding name.
    pub resource: String,
    /// Bind-group and binding slots after deterministic allocation.
    pub group: u32,
    /// Binding slot after deterministic allocation.
    pub binding: u32,
    /// Ordered field layouts.
    pub fields: Vec<CompiledParameterField>,
    /// Struct size rounded to the WGSL uniform-struct alignment.
    pub size: u32,
    /// Default payload suitable for initial GPU upload.
    pub default_bytes: Vec<u8>,
}

impl CompiledParameterBlock {
    /// Encodes named runtime values into the verified ABI layout.
    ///
    /// Values must contain exactly one entry for each declared field. The
    /// returned byte vector is zero-padded, deterministic, and little-endian.
    pub fn encode(
        &self,
        values: &[(&str, ParameterValue)],
    ) -> Result<Vec<u8>, ParameterEncodeError> {
        let mut output = alloc::vec![0u8; self.size as usize];
        for field in &self.fields {
            let Some((_, value)) = values.iter().find(|(name, _)| *name == field.name) else {
                return Err(ParameterEncodeError::MissingField {
                    name: field.name.clone(),
                });
            };
            if value.parameter_type() != field.ty {
                return Err(ParameterEncodeError::TypeMismatch {
                    name: field.name.clone(),
                    expected: field.ty,
                    actual: value.parameter_type(),
                });
            }
            let start = field.offset as usize;
            encode_parameter_value(*value, &mut output[start..start + field.ty.size() as usize]);
        }
        for (name, _) in values {
            if !self.fields.iter().any(|field| field.name == *name) {
                return Err(ParameterEncodeError::UnknownField {
                    name: String::from(*name),
                });
            }
        }
        if values.len() != self.fields.len() {
            return Err(ParameterEncodeError::FieldCount {
                expected: self.fields.len(),
                actual: values.len(),
            });
        }
        Ok(output)
    }
}

/// A fully encoded uniform payload ready for the render layer to upload and
/// bind at the compiled group/binding pair.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedParameterBlock {
    /// Uniform resource name from the compiled graph.
    pub resource: String,
    /// Compiled bind-group index.
    pub group: u32,
    /// Compiled binding index.
    pub binding: u32,
    /// Host bytes matching the compiled WGSL uniform layout.
    pub bytes: Vec<u8>,
}

impl CompiledGraph {
    /// Resolves `.ember` exposed parameters into the compiled uniform blocks.
    ///
    /// Asset parameter targets must identify exactly one compiled uniform
    /// resource/member. Constant bindings provide their authored defaults;
    /// externally bound values must be supplied by stable [`ParamHandle`].
    /// Runtime values for constants are rejected. Unexposed fields retain
    /// their compiled module defaults. This prepares bytes only; GPU resource
    /// creation, upload, and bind-group lifetime belong to the render layer.
    pub fn prepare_asset_parameters(
        &self,
        asset: &EmberAsset,
        runtime_values: &[(ParamHandle, ParamValue)],
    ) -> Result<Vec<PreparedParameterBlock>, AssetParameterError> {
        for (index, (handle, _)) in runtime_values.iter().enumerate() {
            if runtime_values[..index]
                .iter()
                .any(|(previous, _)| previous == handle)
            {
                return Err(AssetParameterError::DuplicateRuntimeValue(*handle));
            }
            let Some(param) = asset.param(*handle) else {
                return Err(AssetParameterError::UnknownRuntimeHandle(*handle));
            };
            if !param.is_externally_driven() {
                return Err(AssetParameterError::UnexpectedRuntimeValue(*handle));
            }
        }

        let mut prepared: Vec<PreparedParameterBlock> = self
            .parameter_blocks
            .iter()
            .map(|block| PreparedParameterBlock {
                resource: block.resource.clone(),
                group: block.group,
                binding: block.binding,
                bytes: block.default_bytes.clone(),
            })
            .collect();

        for param in &asset.params {
            let matches: Vec<(usize, usize)> = self
                .parameter_blocks
                .iter()
                .enumerate()
                .flat_map(|(block_index, block)| {
                    block
                        .fields
                        .iter()
                        .enumerate()
                        .filter(|(_, field)| {
                            block.resource == param.target.resource
                                && field.name == param.target.field
                        })
                        .map(move |(field_index, _)| (block_index, field_index))
                })
                .collect();
            let [(block_index, field_index)] = matches.as_slice() else {
                return Err(if matches.is_empty() {
                    AssetParameterError::UnmappedParameter {
                        handle: param.handle,
                        name: param.name.clone(),
                    }
                } else {
                    AssetParameterError::AmbiguousParameter {
                        handle: param.handle,
                        name: param.name.clone(),
                    }
                });
            };
            let field = &self.parameter_blocks[*block_index].fields[*field_index];
            let value = match runtime_values
                .iter()
                .find(|(handle, _)| *handle == param.handle)
            {
                Some((_, value)) => *value,
                None => match param.binding {
                    BindingSource::Constant(value) => value,
                    _ => return Err(AssetParameterError::MissingRuntimeValue(param.handle)),
                },
            };
            let value = authored_parameter_value(param.handle, param.ty, value)?;
            if value.parameter_type() != field.ty {
                return Err(AssetParameterError::CompiledTypeMismatch {
                    handle: param.handle,
                    expected: field.ty,
                    actual: value.parameter_type(),
                });
            }
            let start = field.offset as usize;
            encode_parameter_value(
                value,
                &mut prepared[*block_index].bytes[start..start + field.ty.size() as usize],
            );
        }
        Ok(prepared)
    }
}

/// Failure to resolve an authored `.ember` parameter into the compiled ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssetParameterError {
    /// A runtime value repeated a stable handle.
    DuplicateRuntimeValue(ParamHandle),
    /// A runtime value does not identify a parameter in the asset.
    UnknownRuntimeHandle(ParamHandle),
    /// An externally driven parameter has no value for this frame.
    MissingRuntimeValue(ParamHandle),
    /// A constant parameter was given a runtime override.
    UnexpectedRuntimeValue(ParamHandle),
    /// An authored parameter name is absent from all compiled blocks.
    UnmappedParameter { handle: ParamHandle, name: String },
    /// An authored parameter name occurs in multiple compiled blocks.
    AmbiguousParameter { handle: ParamHandle, name: String },
    /// The authored value does not match its declared `.ember` type.
    AuthoredTypeMismatch { handle: ParamHandle, ty: ParamType },
    /// The matched compiled field has a different ABI type.
    CompiledTypeMismatch {
        handle: ParamHandle,
        expected: ParameterType,
        actual: ParameterType,
    },
    /// A floating-point parameter contains NaN or infinity.
    NonFiniteValue(ParamHandle),
}

fn authored_parameter_value(
    handle: ParamHandle,
    param_type: ParamType,
    value: ParamValue,
) -> Result<ParameterValue, AssetParameterError> {
    match (param_type, value) {
        (ParamType::Scalar, ParamValue::Scalar(value)) if value.is_finite() => {
            Ok(ParameterValue::F32(value))
        }
        (ParamType::Vec3, ParamValue::Vec3(value)) if value.into_iter().all(f32::is_finite) => {
            Ok(ParameterValue::Vec3(value))
        }
        (ParamType::Color, ParamValue::Color(value)) if value.into_iter().all(f32::is_finite) => {
            Ok(ParameterValue::Vec4(value))
        }
        (ParamType::Int, ParamValue::Int(value)) => Ok(ParameterValue::I32(value)),
        // WGSL `bool` is not host-shareable in a uniform block. The asset ABI
        // therefore stores booleans as canonical 0/1 `u32` lanes; shader code
        // tests the field against `0u`.
        (ParamType::Bool, ParamValue::Bool(value)) => Ok(ParameterValue::U32(u32::from(value))),
        (ParamType::Scalar, ParamValue::Scalar(_))
        | (ParamType::Vec3, ParamValue::Vec3(_))
        | (ParamType::Color, ParamValue::Color(_)) => {
            Err(AssetParameterError::NonFiniteValue(handle))
        }
        _ => Err(AssetParameterError::AuthoredTypeMismatch {
            handle,
            ty: param_type,
        }),
    }
}

/// A host parameter payload did not match its compiled schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParameterEncodeError {
    /// The input did not provide exactly the schema's number of values.
    FieldCount { expected: usize, actual: usize },
    /// A declared field is absent.
    MissingField { name: String },
    /// A field's value has a different scalar/vector type than declared.
    TypeMismatch {
        name: String,
        expected: ParameterType,
        actual: ParameterType,
    },
    /// An extra value was supplied.
    UnknownField { name: String },
}

fn encode_parameter_value(value: ParameterValue, output: &mut [u8]) {
    fn write_f32(output: &mut [u8], index: usize, value: f32) {
        output[index * 4..index * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
    }
    match value {
        ParameterValue::F32(value) => write_f32(output, 0, value),
        ParameterValue::U32(value) => output[..4].copy_from_slice(&value.to_le_bytes()),
        ParameterValue::I32(value) => output[..4].copy_from_slice(&value.to_le_bytes()),
        ParameterValue::Vec2(values) => values
            .iter()
            .enumerate()
            .for_each(|(i, v)| write_f32(output, i, *v)),
        ParameterValue::Vec3(values) => values
            .iter()
            .enumerate()
            .for_each(|(i, v)| write_f32(output, i, *v)),
        ParameterValue::Vec4(values) => values
            .iter()
            .enumerate()
            .for_each(|(i, v)| write_f32(output, i, *v)),
    }
}

/// Which physical buffer copy backs a graph attribute binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeBufferRole {
    /// A single buffer for attributes without previous-frame ping-pong.
    Single,
    /// The previous-frame copy of a ping-ponged attribute.
    Previous,
    /// The current-frame copy of a ping-ponged attribute.
    Current,
}

/// One generated SoA storage binding in group zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledAttributeBinding {
    /// Attribute semantic stored by this binding.
    pub semantic: AttributeSemantic,
    /// WGSL storage element format.
    pub format: AttributeFormat,
    /// Physical copy represented by this binding.
    pub role: AttributeBufferRole,
    /// WGSL identifier available to generated modules.
    pub name: String,
    /// Bind-group index, always zero for graph attributes.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Storage access emitted into WGSL.
    pub access: ResourceAccess,
}

impl CompiledAttributeBinding {
    /// Returns the physical copy index to bind for one frame.
    ///
    /// A ping-ponged attribute uses copy `frame % 2` as the current destination
    /// and the other copy as the previous-frame source. Single-copy attributes
    /// always return zero. The result is an index into the attribute's planned
    /// region, not a bind-group slot.
    #[must_use]
    pub const fn physical_copy(&self, frame_index: u64) -> u32 {
        match self.role {
            AttributeBufferRole::Single => 0,
            AttributeBufferRole::Previous => 1 - (frame_index as u32 & 1),
            AttributeBufferRole::Current => frame_index as u32 & 1,
        }
    }
}

/// The runtime-selected physical copy for one generated attribute binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeAttributeBinding {
    /// Attribute semantic represented by the binding.
    pub semantic: AttributeSemantic,
    /// Logical role used by the generated shader.
    pub role: AttributeBufferRole,
    /// Group-zero binding slot in the generated shader.
    pub binding: u32,
    /// Physical copy within the planned attribute region.
    pub physical_copy: u32,
}

impl CompiledGraph {
    /// Resolves all graph-attribute bindings for a frame parity.
    ///
    /// The returned order is the same as `attribute_bindings`, making it safe
    /// for a backend to use as a deterministic bind-group update list.
    #[must_use]
    pub fn attribute_bindings_for_frame(&self, frame_index: u64) -> Vec<RuntimeAttributeBinding> {
        self.attribute_bindings
            .iter()
            .map(|binding| RuntimeAttributeBinding {
                semantic: binding.semantic,
                role: binding.role,
                binding: binding.binding,
                physical_copy: binding.physical_copy(frame_index),
            })
            .collect()
    }
}

/// One concrete compute dispatch emitted from a graph stage iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledDispatch {
    /// Index into [`GraphIr::stages`].
    pub stage_index: usize,
    /// Iteration domain used to obtain the live count from the runtime.
    pub domain: IterationDomain,
    /// Safe initial count for the runtime buffer; dynamic domains default to
    /// zero and must be populated before dispatch.
    pub default_count: u32,
    /// Number of consecutive dispatches to submit for this stage.
    pub iterations: u32,
    /// Entry point to submit on each dispatch.
    pub entry_point: String,
}

/// Which shader validation phase rejected generated source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderValidationPhase {
    /// WGSL grammar parsing failed.
    Parse,
    /// WGSL semantic/type validation failed.
    Validate,
}

/// A structured compiler failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    /// Existing graph semantic validation failed.
    InvalidGraph(GraphError),
    /// A module assignment refers to no graph stage.
    ModuleStageOutOfRange { module: usize, stage: usize },
    /// The module's declared stage kind cannot run in the assigned iteration
    /// domain.
    ModuleDomainMismatch {
        module: usize,
        stage: usize,
        expected: IterationDomain,
        actual: IterationDomain,
    },
    /// A module requires an attribute that its graph stage did not declare.
    ModuleAttributeUndeclared {
        module: usize,
        stage: usize,
        semantic: AttributeSemantic,
    },
    /// A module's generated kernel needs an access mode that the graph stage
    /// did not declare (for example `READ_PREV | WRITE` for a ping-pong read).
    ModuleAttributeAccessUndeclared {
        module: usize,
        stage: usize,
        semantic: AttributeSemantic,
        required: AttributeAccess,
        declared: AttributeAccess,
    },
    /// A module's `resources()` and `CodegenCtx` binding declarations disagree.
    ModuleBindingMismatch { module: usize, name: String },
    /// A uniform does not have exactly one typed schema, or a schema does not
    /// refer to a declared uniform resource.
    ParameterSchemaMismatch { resource: String },
    /// A parameter field name is not a legal WGSL identifier.
    InvalidParameterIdentifier { resource: String, field: String },
    /// A field default does not match the declared value type.
    ParameterDefaultTypeMismatch { resource: String, field: String },
    /// A data interface's `bindings()` and `CodegenCtx` declarations disagree.
    DataInterfaceBindingMismatch { interface: usize, name: String },
    /// Different declarations use one binding name.
    BindingNameConflict { name: String },
    /// Different declarations occupy one `(group, binding)` slot.
    BindingSlotConflict { group: u32, binding: u32 },
    /// Two modules emitted different definitions for one helper or kernel name.
    FunctionNameConflict { name: String },
    /// The compiler cannot allocate another module resource in this bind group.
    BindingIndexLimitExceeded { group: u32 },
    /// The resource declaration cannot be represented by the supported WGSL
    /// resource ABI (for example, a writable sampled texture).
    UnsupportedBinding { name: String },
    /// The graph declares more SoA buffer copies than fit the WGSL binding index.
    AttributeBindingLimitExceeded,
    /// Generated identifiers for two graph stages collide.
    StageEntryPointCollision {
        first: usize,
        second: usize,
        name: String,
    },
    /// A resource name is not a legal WGSL identifier.
    InvalidBindingIdentifier { name: String },
    /// Naga rejected the assembled WGSL.
    InvalidGeneratedWgsl {
        phase: ShaderValidationPhase,
        diagnostic: String,
    },
}

/// Compiles a validated graph and its explicitly assigned modules to WGSL.
///
/// The graph's topological order is stable: ties are resolved by graph stage
/// index by [`graph::schedule_order`]. Bindings are merged in stage/module
/// encounter order, while source entry points follow that schedule. Compatible
/// resources with the same name are shared. Module binding numbers are
/// allocation hints within groups 2 and above; collisions are resolved to the
/// next free binding deterministically. Groups zero and one are reserved for
/// graph attributes and dispatch counts.
///
/// Module fragments are function-body statements, not complete WGSL modules.
/// The result is Naga-validated before it is returned. This validates the WGSL
/// ABI and executable source subset, but does not prove that snippets implement
/// the attribute semantics declared by `GraphIr` or `EmberModule`.
pub fn compile(
    graph_ir: &GraphIr,
    modules: &[StageModule<'_>],
) -> Result<CompiledGraph, CompileError> {
    compile_with_data_interfaces(graph_ir, modules, &[])
}

/// Compiles a graph with explicitly selected data interfaces.
///
/// Data interfaces contribute only global resource declarations and reusable
/// sampling functions; modules still own the stage fragments that call them.
/// Keeping the interface list explicit makes asset compilation deterministic
/// and prevents unused scene bindings from entering a pipeline layout.
pub fn compile_with_data_interfaces(
    graph_ir: &GraphIr,
    modules: &[StageModule<'_>],
    data_interfaces: &[&dyn DataInterface],
) -> Result<CompiledGraph, CompileError> {
    graph_ir.validate().map_err(CompileError::InvalidGraph)?;
    let stage_order = graph::schedule_order(graph_ir).map_err(CompileError::InvalidGraph)?;
    let entry_names = entry_point_names(graph_ir)?;

    let mut stage_fragments: Vec<Vec<String>> = Vec::new();
    stage_fragments.resize_with(graph_ir.stages.len(), Vec::new);
    let mut bindings = Vec::new();
    let mut functions = Vec::new();
    let mut parameter_schemas = Vec::new();
    let attribute_layout = graph::plan_layout(graph_ir);
    let attribute_bindings =
        build_attribute_bindings(&attribute_layout, &graph::live_usages(graph_ir))?;

    for (interface_index, interface) in data_interfaces.iter().enumerate() {
        let mut context = super::modules::CodegenCtx::new();
        interface.emit_wesl_functions(&mut context);

        for emitted in context.bindings() {
            if !interface.bindings().iter().any(|decl| decl == emitted) {
                return Err(CompileError::DataInterfaceBindingMismatch {
                    interface: interface_index,
                    name: String::from(emitted.name),
                });
            }
        }
        for declared in interface.bindings() {
            if !context.bindings().iter().any(|emitted| emitted == declared) {
                return Err(CompileError::DataInterfaceBindingMismatch {
                    interface: interface_index,
                    name: String::from(declared.name),
                });
            }
            merge_module_binding(&mut bindings, *declared)?;
        }
        parameter_schemas.extend_from_slice(interface.parameter_blocks());
        for function in context.functions() {
            let resolved = resolve_parameter_identifiers(function, &parameter_schemas);
            merge_function(&mut functions, &resolved)?;
        }
    }

    for (module_index, assignment) in modules.iter().enumerate() {
        let Some(stage) = graph_ir.stages.get(assignment.stage_index) else {
            return Err(CompileError::ModuleStageOutOfRange {
                module: module_index,
                stage: assignment.stage_index,
            });
        };
        let expected = assignment.module.stage().default_domain();
        if expected != stage.domain {
            return Err(CompileError::ModuleDomainMismatch {
                module: module_index,
                stage: assignment.stage_index,
                expected,
                actual: stage.domain,
            });
        }
        let stage_demand = stage.demand();
        for &semantic in assignment.module.required_attributes() {
            if !stage_demand.demands(semantic) {
                return Err(CompileError::ModuleAttributeUndeclared {
                    module: module_index,
                    stage: assignment.stage_index,
                    semantic,
                });
            }
        }
        for required in assignment.module.attribute_accesses() {
            let declared = stage_demand
                .usages()
                .iter()
                .find(|usage| usage.semantic == required.semantic)
                .map_or(AttributeAccess::empty(), |usage| usage.access);
            if !declared.contains(required.access) {
                return Err(CompileError::ModuleAttributeAccessUndeclared {
                    module: module_index,
                    stage: assignment.stage_index,
                    semantic: required.semantic,
                    required: required.access,
                    declared,
                });
            }
        }

        let declared = assignment.module.resources();
        validate_module_bindings(declared)?;
        let mut context = super::modules::CodegenCtx::new();
        assignment.module.emit_wesl(&mut context);
        assignment.module.emit_wesl_functions(&mut context);

        for emitted in context.bindings() {
            if !declared.iter().any(|decl| decl == emitted) {
                return Err(CompileError::ModuleBindingMismatch {
                    module: module_index,
                    name: String::from(emitted.name),
                });
            }
        }
        for declared_binding in declared {
            if !context
                .bindings()
                .iter()
                .any(|emitted| emitted == declared_binding)
            {
                return Err(CompileError::ModuleBindingMismatch {
                    module: module_index,
                    name: String::from(declared_binding.name),
                });
            }
            merge_module_binding(&mut bindings, *declared_binding)?;
        }
        parameter_schemas.extend_from_slice(assignment.module.parameter_blocks());

        for function in context.functions() {
            let resolved = resolve_attribute_identifiers(function, &attribute_layout);
            let resolved = resolve_parameter_identifiers(&resolved, &parameter_schemas);
            merge_function(&mut functions, &resolved)?;
        }

        stage_fragments[assignment.stage_index].extend(context.fragments().iter().map(
            |fragment| {
                let resolved = resolve_attribute_identifiers(fragment, &attribute_layout);
                resolve_parameter_identifiers(&resolved, &parameter_schemas)
            },
        ));
    }

    // Resolve shorthand after collecting the complete module/interface schema
    // set so output meaning does not depend on encounter order.
    for function in &mut functions {
        *function = resolve_parameter_identifiers(function, &parameter_schemas);
    }
    for fragments in &mut stage_fragments {
        for fragment in fragments {
            *fragment = resolve_parameter_identifiers(fragment, &parameter_schemas);
        }
    }

    if !graph_ir.stages.is_empty() {
        merge_binding(&mut bindings, DISPATCH_COUNT_BINDING)?;
    }

    validate_attribute_bindings(&bindings, &attribute_bindings)?;
    let parameter_blocks = compile_parameter_blocks(&bindings, &parameter_schemas)?;

    let mut source = String::from("// Generated by Prism Ember graph compiler.\n");
    for binding in &attribute_bindings {
        emit_attribute_binding(&mut source, binding);
    }
    for binding in &bindings {
        if binding.kind != ResourceKind::Uniform {
            emit_binding(&mut source, *binding)?;
        }
    }
    for block in &parameter_blocks {
        emit_parameter_block(&mut source, block);
    }
    for binding in &bindings {
        if binding.kind == ResourceKind::Curve {
            emit_curve_sampler(&mut source, *binding);
        }
    }
    for function in &functions {
        source.push('\n');
        source.push_str(function);
        source.push('\n');
    }

    let mut entry_points = Vec::with_capacity(stage_order.len());
    let mut dispatches = Vec::with_capacity(stage_order.len());
    for &stage_index in &stage_order {
        let stage = &graph_ir.stages[stage_index];
        let index_name = if stage.domain == IterationDomain::PerParticle {
            "particle_index"
        } else {
            "invocation_index"
        };
        let entry_name = &entry_names[stage_index];
        entry_points.push(entry_name.clone());
        dispatches.push(CompiledDispatch {
            stage_index,
            domain: stage.domain,
            default_count: dispatch_count_default(stage.domain, graph_ir.capacity),
            iterations: stage.iterations,
            entry_point: entry_name.clone(),
        });
        source.push_str(&format!(
            "@compute @workgroup_size({WORKGROUP_SIZE})\nfn {entry_name}(\n    @builtin(global_invocation_id) invocation_id: vec3<u32>\n) {{\n    let {index_name}: u32 = invocation_id.x;\n    if ({index_name} >= ember_dispatch_counts[{stage_index}u]) {{ return; }}\n"
        ));
        for fragment in &stage_fragments[stage_index] {
            source.push_str("    ");
            source.push_str(fragment);
            if !fragment.ends_with('\n') {
                source.push('\n');
            }
        }
        source.push_str("}\n");
    }

    validate_wgsl(&source)?;
    Ok(CompiledGraph {
        source,
        stage_order,
        entry_points,
        dispatches,
        bindings,
        parameter_blocks,
        attribute_layout,
        attribute_bindings,
    })
}

fn build_attribute_bindings(
    layout: &AttributeLayoutPlan,
    usages: &[super::attributes::AttributeUsage],
) -> Result<Vec<CompiledAttributeBinding>, CompileError> {
    let mut output = Vec::new();
    for planned in &layout.attributes {
        let Some(usage) = usages
            .iter()
            .find(|usage| usage.semantic == planned.semantic)
        else {
            continue;
        };
        let base_name = attribute_identifier(planned.semantic);
        if planned.copies == 2 {
            push_attribute_binding(
                &mut output,
                planned.semantic,
                planned.format,
                AttributeBufferRole::Previous,
                format!("ember_attr_{base_name}_previous"),
                ResourceAccess::ReadOnly,
            )?;
            push_attribute_binding(
                &mut output,
                planned.semantic,
                planned.format,
                AttributeBufferRole::Current,
                format!("ember_attr_{base_name}_current"),
                ResourceAccess::ReadWrite,
            )?;
        } else {
            let access = if usage.access.contains(AttributeAccess::WRITE) {
                ResourceAccess::ReadWrite
            } else {
                ResourceAccess::ReadOnly
            };
            push_attribute_binding(
                &mut output,
                planned.semantic,
                planned.format,
                AttributeBufferRole::Single,
                format!("ember_attr_{base_name}"),
                access,
            )?;
        }
    }
    Ok(output)
}

fn push_attribute_binding(
    output: &mut Vec<CompiledAttributeBinding>,
    semantic: AttributeSemantic,
    format: AttributeFormat,
    role: AttributeBufferRole,
    name: String,
    access: ResourceAccess,
) -> Result<(), CompileError> {
    let binding =
        u32::try_from(output.len()).map_err(|_| CompileError::AttributeBindingLimitExceeded)?;
    output.push(CompiledAttributeBinding {
        semantic,
        format,
        role,
        name,
        group: 0,
        binding,
        access,
    });
    Ok(())
}

fn attribute_identifier(semantic: AttributeSemantic) -> String {
    let name = match semantic {
        AttributeSemantic::Position => "position".into(),
        AttributeSemantic::Velocity => "velocity".into(),
        AttributeSemantic::Age => "age".into(),
        AttributeSemantic::Lifetime => "lifetime".into(),
        AttributeSemantic::Color => "color".into(),
        AttributeSemantic::Size => "size".into(),
        AttributeSemantic::Rotation => "rotation".into(),
        AttributeSemantic::Scale => "scale".into(),
        AttributeSemantic::Alive => "alive".into(),
        AttributeSemantic::ParticleId => "particle_id".into(),
        AttributeSemantic::RibbonId => "ribbon_id".into(),
        AttributeSemantic::SortKey => "sort_key".into(),
        AttributeSemantic::Normal => "normal".into(),
        AttributeSemantic::Tangent => "tangent".into(),
        AttributeSemantic::Emissive => "emissive".into(),
        AttributeSemantic::Roughness => "roughness".into(),
        AttributeSemantic::Metallic => "metallic".into(),
        AttributeSemantic::MaterialId => "material_id".into(),
        AttributeSemantic::ShadingParams => "shading_params".into(),
        AttributeSemantic::Custom(id) => format!("custom_{id}"),
    };
    name
}

/// Resolves the logical attribute identifiers emitted by modules to the
/// physical storage names in this graph's ABI.
///
/// Built-in kernels use `ember_attr_<semantic>` for current-frame accesses and
/// may use explicit `_previous`/`_current` suffixes when a module needs both
/// sides. A non-ping-ponged channel has one physical binding, so both explicit
/// sides collapse to that binding. Identifier-token replacement keeps names
/// such as `velocity_current` from being rewritten as a substring of another
/// identifier and also makes custom snippets deterministic.
fn resolve_attribute_identifiers(source: &str, layout: &AttributeLayoutPlan) -> String {
    let mut replacements = Vec::new();
    for planned in &layout.attributes {
        let base = format!("ember_attr_{}", attribute_identifier(planned.semantic));
        let previous = format!("{base}_previous");
        let current = format!("{base}_current");
        let (resolved_previous, resolved_current, resolved_base) = if planned.copies == 2 {
            (previous.clone(), current.clone(), current.clone())
        } else {
            (base.clone(), base.clone(), base.clone())
        };
        replacements.push((previous, resolved_previous));
        replacements.push((current, resolved_current));
        replacements.push((base, resolved_base));
    }

    let mut output = String::with_capacity(source.len());
    let mut token = String::new();
    let flush = |output: &mut String, token: &mut String| {
        if token.is_empty() {
            return;
        }
        if let Some((_, replacement)) = replacements.iter().find(|(name, _)| name == token) {
            output.push_str(replacement);
        } else {
            output.push_str(token);
        }
        token.clear();
    };
    for ch in source.chars() {
        if ch == '_' || ch.is_ascii_alphanumeric() {
            token.push(ch);
        } else {
            flush(&mut output, &mut token);
            output.push(ch);
        }
    }
    flush(&mut output, &mut token);
    output
}

fn validate_attribute_bindings(
    module_bindings: &[ResourceDecl],
    attribute_bindings: &[CompiledAttributeBinding],
) -> Result<(), CompileError> {
    for attribute in attribute_bindings {
        if module_bindings
            .iter()
            .any(|module| module.name == attribute.name.as_str())
        {
            return Err(CompileError::BindingNameConflict {
                name: attribute.name.clone(),
            });
        }
        if module_bindings
            .iter()
            .any(|module| (module.group, module.binding) == (attribute.group, attribute.binding))
        {
            return Err(CompileError::BindingSlotConflict {
                group: attribute.group,
                binding: attribute.binding,
            });
        }
    }
    Ok(())
}

fn compile_parameter_blocks(
    bindings: &[ResourceDecl],
    schemas: &[ParameterBlockDecl],
) -> Result<Vec<CompiledParameterBlock>, CompileError> {
    let mut output = Vec::new();
    for schema in schemas {
        if let Some(existing) = output
            .iter()
            .find(|existing: &&CompiledParameterBlock| existing.resource == schema.resource)
        {
            let same_schema = existing.fields.len() == schema.fields.len()
                && existing
                    .fields
                    .iter()
                    .zip(schema.fields)
                    .all(|(compiled, declared)| {
                        compiled.name == declared.name
                            && compiled.ty == declared.ty
                            && compiled.default_bytes == parameter_value_bytes(declared.default)
                    });
            if !same_schema {
                return Err(CompileError::ParameterSchemaMismatch {
                    resource: String::from(schema.resource),
                });
            }
            continue;
        }
        let Some(binding) = bindings
            .iter()
            .find(|binding| binding.name == schema.resource)
        else {
            return Err(CompileError::ParameterSchemaMismatch {
                resource: String::from(schema.resource),
            });
        };
        if binding.kind != ResourceKind::Uniform || schema.fields.is_empty() {
            return Err(CompileError::ParameterSchemaMismatch {
                resource: String::from(schema.resource),
            });
        }
        let mut fields = Vec::with_capacity(schema.fields.len());
        let mut offset = 0u32;
        let mut block_alignment = 16u32;
        let mut default_bytes = Vec::new();
        for field in schema.fields {
            if !is_wgsl_identifier(field.name) {
                return Err(CompileError::InvalidParameterIdentifier {
                    resource: String::from(schema.resource),
                    field: String::from(field.name),
                });
            }
            if field.default.parameter_type() != field.ty {
                return Err(CompileError::ParameterDefaultTypeMismatch {
                    resource: String::from(schema.resource),
                    field: String::from(field.name),
                });
            }
            if fields
                .iter()
                .any(|previous: &CompiledParameterField| previous.name == field.name)
            {
                return Err(CompileError::ParameterSchemaMismatch {
                    resource: String::from(schema.resource),
                });
            }
            let alignment = field.ty.alignment();
            block_alignment = block_alignment.max(alignment);
            offset = align_up(offset, alignment).ok_or_else(|| {
                CompileError::ParameterSchemaMismatch {
                    resource: String::from(schema.resource),
                }
            })?;
            let bytes = parameter_value_bytes(field.default);
            let start = offset as usize;
            if default_bytes.len() < start + bytes.len() {
                default_bytes.resize(start + bytes.len(), 0);
            }
            default_bytes[start..start + bytes.len()].copy_from_slice(&bytes);
            fields.push(CompiledParameterField {
                name: String::from(field.name),
                ty: field.ty,
                offset,
                default_bytes: bytes,
            });
            offset = offset.checked_add(field.ty.size()).ok_or_else(|| {
                CompileError::ParameterSchemaMismatch {
                    resource: String::from(schema.resource),
                }
            })?;
        }
        let size = align_up(offset, block_alignment).ok_or_else(|| {
            CompileError::ParameterSchemaMismatch {
                resource: String::from(schema.resource),
            }
        })?;
        default_bytes.resize(size as usize, 0);
        let resource = String::from(schema.resource);
        output.push(CompiledParameterBlock {
            resource,
            group: binding.group,
            binding: binding.binding,
            fields,
            size,
            default_bytes,
        });
    }
    for binding in bindings {
        if binding.kind == ResourceKind::Uniform
            && !output.iter().any(|block| block.resource == binding.name)
        {
            return Err(CompileError::ParameterSchemaMismatch {
                resource: String::from(binding.name),
            });
        }
    }
    output.sort_by_key(|block| {
        bindings
            .iter()
            .position(|binding| binding.name == block.resource)
            .unwrap_or(usize::MAX)
    });
    Ok(output)
}

fn align_up(value: u32, alignment: u32) -> Option<u32> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

fn parameter_value_bytes(value: ParameterValue) -> Vec<u8> {
    let size = value.parameter_type().size() as usize;
    let mut bytes = alloc::vec![0u8; size];
    encode_parameter_value(value, &mut bytes);
    bytes
}

/// Lowers shorthand references for one-field blocks; multi-field schemas use
/// explicit `resource.field` access in their generated source.
fn resolve_parameter_identifiers(source: &str, schemas: &[ParameterBlockDecl]) -> String {
    let mut output = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let identifier = &source[start..index];
            if bytes.get(index) == Some(&b'.') {
                let member_start = index + 1;
                let member_end = member_start
                    + bytes[member_start..]
                        .iter()
                        .position(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_')
                        .unwrap_or(bytes.len() - member_start);
                let member = &source[member_start..member_end];
                if let Some(schema) = schemas.iter().find(|schema| {
                    schema.resource == identifier
                        && schema.fields.len() == 1
                        && !schema.fields.iter().any(|field| field.name == member)
                }) {
                    output.push_str(identifier);
                    output.push('.');
                    output.push_str(schema.fields[0].name);
                } else {
                    output.push_str(identifier);
                }
            } else if let Some(schema) = schemas
                .iter()
                .find(|schema| schema.resource == identifier && schema.fields.len() == 1)
            {
                output.push_str(identifier);
                output.push('.');
                output.push_str(schema.fields[0].name);
            } else {
                output.push_str(identifier);
            }
        } else {
            let ch = source[index..].chars().next().expect("valid source index");
            output.push(ch);
            index += ch.len_utf8();
        }
    }
    output
}

fn emit_parameter_block(source: &mut String, block: &CompiledParameterBlock) {
    source.push_str(&format!("\nstruct ember_params_{} {{\n", block.resource));
    for field in &block.fields {
        source.push_str(&format!("    {}: {},\n", field.name, field.ty.wgsl()));
    }
    source.push_str(&format!(
        "}}\n@group({}) @binding({}) var<uniform> {}: ember_params_{};\n",
        block.group, block.binding, block.resource, block.resource
    ));
}

fn emit_attribute_binding(source: &mut String, binding: &CompiledAttributeBinding) {
    let access = match binding.access {
        ResourceAccess::ReadOnly => "read",
        ResourceAccess::ReadWrite => "read_write",
    };
    source.push_str(&format!(
        "@group({}) @binding({}) var<storage, {access}> {}: array<{}>;\n",
        binding.group,
        binding.binding,
        binding.name,
        wgsl_type(binding.format),
    ));
}

fn entry_point_names(graph_ir: &GraphIr) -> Result<Vec<String>, CompileError> {
    let mut names = Vec::with_capacity(graph_ir.stages.len());
    for (index, stage) in graph_ir.stages.iter().enumerate() {
        let mut name = String::from("ember_stage_");
        name.push_str(&index.to_string());
        name.push('_');
        for (position, ch) in stage.name.chars().enumerate() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                if position == 0 && ch.is_ascii_digit() {
                    name.push('_');
                }
                name.push(ch);
            } else {
                name.push('_');
            }
        }
        if names.iter().any(|existing| existing == &name) {
            let first = names
                .iter()
                .position(|existing| existing == &name)
                .unwrap_or(0);
            return Err(CompileError::StageEntryPointCollision {
                first,
                second: index,
                name,
            });
        }
        names.push(name);
    }
    Ok(names)
}

fn validate_module_bindings(bindings: &[ResourceDecl]) -> Result<(), CompileError> {
    for (index, binding) in bindings.iter().enumerate() {
        if !is_wgsl_identifier(binding.name) {
            return Err(CompileError::InvalidBindingIdentifier {
                name: String::from(binding.name),
            });
        }
        for previous in &bindings[..index] {
            if previous.name == binding.name && previous != binding {
                return Err(CompileError::BindingNameConflict {
                    name: String::from(binding.name),
                });
            }
            if (previous.group, previous.binding) == (binding.group, binding.binding)
                && previous != binding
            {
                return Err(CompileError::BindingSlotConflict {
                    group: binding.group,
                    binding: binding.binding,
                });
            }
        }
    }
    Ok(())
}

fn dispatch_count_default(domain: IterationDomain, capacity: u32) -> u32 {
    match domain {
        IterationDomain::PerParticle => capacity,
        IterationDomain::Custom(count) => count,
        IterationDomain::PerNeighborCell
        | IterationDomain::PerGridVoxel
        | IterationDomain::PerConstraint
        | IterationDomain::PerEvent => 0,
    }
}

fn merge_binding(
    bindings: &mut Vec<ResourceDecl>,
    binding: ResourceDecl,
) -> Result<(), CompileError> {
    for existing in bindings.iter() {
        if existing.name == binding.name {
            if existing == &binding {
                return Ok(());
            }
            return Err(CompileError::BindingNameConflict {
                name: String::from(binding.name),
            });
        }
        if (existing.group, existing.binding) == (binding.group, binding.binding) {
            return Err(CompileError::BindingSlotConflict {
                group: binding.group,
                binding: binding.binding,
            });
        }
    }
    bindings.push(binding);
    Ok(())
}

fn merge_module_binding(
    bindings: &mut Vec<ResourceDecl>,
    mut binding: ResourceDecl,
) -> Result<(), CompileError> {
    if binding.group < 2 {
        return Err(CompileError::BindingSlotConflict {
            group: binding.group,
            binding: binding.binding,
        });
    }

    if let Some(existing) = bindings
        .iter()
        .find(|existing| existing.name == binding.name)
    {
        if existing.kind == binding.kind
            && existing.access == binding.access
            && existing.format == binding.format
            && existing.group == binding.group
        {
            return Ok(());
        }
        return Err(CompileError::BindingNameConflict {
            name: String::from(binding.name),
        });
    }

    while bindings
        .iter()
        .any(|existing| existing.group == binding.group && existing.binding == binding.binding)
    {
        binding.binding =
            binding
                .binding
                .checked_add(1)
                .ok_or(CompileError::BindingIndexLimitExceeded {
                    group: binding.group,
                })?;
    }
    bindings.push(binding);
    Ok(())
}

fn merge_function(functions: &mut Vec<String>, function: &str) -> Result<(), CompileError> {
    let name = function
        .strip_prefix("fn ")
        .and_then(|rest| rest.split_once('(').map(|(name, _)| name.trim()))
        .filter(|name| is_wgsl_identifier(name))
        .ok_or_else(|| CompileError::InvalidGeneratedWgsl {
            phase: ShaderValidationPhase::Parse,
            diagnostic: String::from("generated function is missing a valid WGSL name"),
        })?;
    if let Some(existing) = functions.iter().find(|existing| {
        existing
            .strip_prefix("fn ")
            .and_then(|rest| rest.split_once('(').map(|(name, _)| name.trim()))
            == Some(name)
    }) {
        if existing == function {
            return Ok(());
        }
        return Err(CompileError::FunctionNameConflict {
            name: String::from(name),
        });
    }
    functions.push(String::from(function));
    Ok(())
}

fn emit_binding(source: &mut String, binding: ResourceDecl) -> Result<(), CompileError> {
    if !is_wgsl_identifier(binding.name) {
        return Err(CompileError::InvalidBindingIdentifier {
            name: String::from(binding.name),
        });
    }
    source.push_str(&format!(
        "@group({}) @binding({}) ",
        binding.group, binding.binding
    ));
    match binding.kind {
        ResourceKind::Uniform if binding.access == ResourceAccess::ReadOnly => {
            source.push_str(&format!(
                "var<uniform> {}: {} ;\n",
                binding.name,
                wgsl_type(binding.format)
            ));
        }
        ResourceKind::Curve
            if binding.access == ResourceAccess::ReadOnly
                && binding.format == AttributeFormat::Vec4 =>
        {
            source.push_str(&format!(
                "var<storage, read> {}: array<vec4<f32>>;\n",
                binding.name
            ));
        }
        ResourceKind::StorageBuffer => {
            let access = match binding.access {
                ResourceAccess::ReadOnly => "read",
                ResourceAccess::ReadWrite => "read_write",
            };
            source.push_str(&format!(
                "var<storage, {access}> {}: array<{}>;\n",
                binding.name,
                wgsl_type(binding.format)
            ));
        }
        ResourceKind::Texture2d | ResourceKind::Texture3d
            if binding.access == ResourceAccess::ReadOnly =>
        {
            let dimension = match binding.kind {
                ResourceKind::Texture2d => "2d",
                ResourceKind::Texture3d => "3d",
                _ => unreachable!(),
            };
            source.push_str(&format!(
                "var {}: texture_{dimension}<f32>;\n",
                binding.name
            ));
        }
        _ => {
            return Err(CompileError::UnsupportedBinding {
                name: String::from(binding.name),
            });
        }
    }
    Ok(())
}

/// Emits a normalized-age, linearly filtered sampler for a curve LUT binding.
/// Curve resources use `array<vec4<f32>>` so the same ABI supports scalar LUTs
/// (read from `.x`) and color ramps (all four channels). The runtime array length
/// is queried from the bound buffer, so resolution remains an asset property.
fn emit_curve_sampler(source: &mut String, binding: ResourceDecl) {
    source.push_str(&format!(
        "\nfn ember_sample_{}(age: f32) -> vec4<f32> {{\n    let count = arrayLength(&{});\n    if (count == 0u) {{ return vec4<f32>(0.0); }}\n    let position = clamp(age, 0.0, 1.0) * f32(count - 1u);\n    let lower = min(u32(floor(position)), count - 1u);\n    let upper = min(lower + 1u, count - 1u);\n    return mix({}[lower], {}[upper], fract(position));\n}}\n",
        binding.name, binding.name, binding.name, binding.name
    ));
}

fn wgsl_type(format: AttributeFormat) -> &'static str {
    match format {
        AttributeFormat::F32 => "f32",
        AttributeFormat::U32 => "u32",
        AttributeFormat::Vec2 => "vec2<f32>",
        AttributeFormat::Vec3 => "vec3<f32>",
        AttributeFormat::Vec4 => "vec4<f32>",
    }
}

fn is_wgsl_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn validate_wgsl(source: &str) -> Result<(), CompileError> {
    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        CompileError::InvalidGeneratedWgsl {
            phase: ShaderValidationPhase::Parse,
            diagnostic: error.emit_to_string(source),
        }
    })?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| CompileError::InvalidGeneratedWgsl {
        phase: ShaderValidationPhase::Validate,
        diagnostic: format!("{error:?}"),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::attributes::AttributeSemantic;
    use crate::particle::graph::{read, read_prev, write, PlatformProfile, StageIr};
    use crate::particle::modules::{
        BuiltinDataInterface, BuiltinModule, CodegenCtx, DataInterface, StageKind,
    };
    use crate::particle::EmberShadingModel;

    struct SnippetModule {
        stage: StageKind,
        resources: Vec<ResourceDecl>,
        fragments: Vec<&'static str>,
    }

    const TEST_PARAMS: &[ParameterBlockDecl] = &[ParameterBlockDecl::new(
        "params",
        &[
            super::super::modules::ParameterField::new(
                "weight",
                ParameterType::F32,
                ParameterValue::F32(0.5),
            ),
            super::super::modules::ParameterField::new(
                "direction",
                ParameterType::Vec3,
                ParameterValue::Vec3([1.0, 2.0, 3.0]),
            ),
        ],
    )];

    const TEST_CONTROL_PARAMS: &[ParameterBlockDecl] = &[ParameterBlockDecl::new(
        "controls",
        &[
            super::super::modules::ParameterField::new(
                "steps",
                ParameterType::I32,
                ParameterValue::I32(0),
            ),
            super::super::modules::ParameterField::new(
                "enabled_bits",
                ParameterType::U32,
                ParameterValue::U32(0),
            ),
        ],
    )];

    impl EmberModule for SnippetModule {
        fn stage(&self) -> StageKind {
            self.stage
        }

        fn required_attributes(&self) -> &[AttributeSemantic] {
            &[]
        }

        fn resources(&self) -> &[ResourceDecl] {
            &self.resources
        }

        fn parameter_blocks(&self) -> &[ParameterBlockDecl] {
            if self
                .resources
                .iter()
                .any(|resource| resource.name == "params")
            {
                TEST_PARAMS
            } else if self
                .resources
                .iter()
                .any(|resource| resource.name == "controls")
            {
                TEST_CONTROL_PARAMS
            } else {
                &[]
            }
        }

        fn emit_wesl(&self, context: &mut CodegenCtx) {
            for resource in &self.resources {
                context.declare_binding(*resource);
            }
            for fragment in &self.fragments {
                context.push_fragment(*fragment);
            }
        }
    }

    fn graph() -> GraphIr {
        GraphIr::new(128, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("advance", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Position)),
            )
            .with_stage(
                StageIr::new("spawn", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Position))
                    .depends_on(0),
            )
    }

    #[test]
    fn output_is_deterministic_and_entry_points_follow_topological_order() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: Vec::new(),
            fragments: vec!["let observed_index: u32 = particle_index;"],
        };
        let assignments = [StageModule {
            stage_index: 0,
            module: &module,
        }];
        let first = compile(&graph(), &assignments).unwrap();
        let second = compile(&graph(), &assignments).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.stage_order, [0, 1]);
        assert_eq!(
            first.entry_points,
            ["ember_stage_0_advance", "ember_stage_1_spawn"]
        );
        assert_eq!(first.dispatches.len(), 2);
        assert_eq!(first.dispatches[0].stage_index, 0);
        assert_eq!(first.dispatches[0].iterations, 1);
        assert_eq!(first.dispatches[0].domain, IterationDomain::PerParticle);
        assert_eq!(first.dispatches[0].default_count, 128);
        assert_eq!(first.dispatches[0].entry_point, first.entry_points[0]);
        assert!(first.source.contains("@compute @workgroup_size(64)"));
        assert!(first
            .source
            .contains("@builtin(global_invocation_id) invocation_id"));
        assert!(first
            .source
            .contains("let particle_index: u32 = invocation_id.x;"));
        assert!(first
            .source
            .contains("let observed_index: u32 = particle_index;"));
    }

    #[test]
    fn non_particle_domains_require_host_dispatch_extent() {
        let grid_graph = GraphIr::new(128, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("grid_solve", IterationDomain::PerGridVoxel, 3));
        let compiled = compile(&grid_graph, &[]).unwrap();
        assert!(compiled.source.contains("ember_dispatch_counts[0u]"));
        assert_eq!(compiled.dispatches[0].domain, IterationDomain::PerGridVoxel);
        assert_eq!(compiled.dispatches[0].default_count, 0);
        assert!(compiled
            .source
            .contains("let invocation_index: u32 = invocation_id.x;"));
        assert!(!compiled.source.contains("let particle_index:"));
    }

    #[test]
    fn repeated_solver_iterations_share_one_kernel_and_expand_the_dispatch_plan() {
        let repeated = GraphIr::new(128, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("jacobi", IterationDomain::PerParticle, 3))
            .with_stage(StageIr::new("finalize", IterationDomain::PerParticle, 2).depends_on(0));
        let compiled = compile(&repeated, &[]).unwrap();
        assert_eq!(compiled.entry_points.len(), 2);
        assert_eq!(compiled.dispatches.len(), 2);
        assert_eq!(compiled.dispatches[0].iterations, 3);
        assert_eq!(compiled.dispatches[1].iterations, 2);
        assert_eq!(compiled.dispatches[0].entry_point, compiled.entry_points[0]);
        assert_eq!(compiled.dispatches[1].entry_point, compiled.entry_points[1]);
        assert_eq!(
            compiled
                .source
                .matches("@compute @workgroup_size(64)")
                .count(),
            2,
            "the shader binary is emitted once per stage, not once per solver iteration"
        );
    }

    #[test]
    fn graph_validation_errors_are_preserved_structurally() {
        let invalid = GraphIr::new(32, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("bad", IterationDomain::PerParticle, 0));
        assert_eq!(
            compile(&invalid, &[]),
            Err(CompileError::InvalidGraph(GraphError::ZeroIterations {
                stage: 0
            }))
        );
    }

    #[test]
    fn invalid_module_stage_and_domain_are_reported() {
        let module = SnippetModule {
            stage: StageKind::Constraint,
            resources: Vec::new(),
            fragments: Vec::new(),
        };
        assert!(matches!(
            compile(
                &graph(),
                &[StageModule {
                    stage_index: 99,
                    module: &module
                }]
            ),
            Err(CompileError::ModuleStageOutOfRange {
                module: 0,
                stage: 99
            })
        ));
        assert!(matches!(
            compile(
                &graph(),
                &[StageModule {
                    stage_index: 0,
                    module: &module
                }]
            ),
            Err(CompileError::ModuleDomainMismatch {
                module: 0,
                stage: 0,
                ..
            })
        ));
    }

    #[test]
    fn module_attribute_contracts_must_be_declared_and_live_layout_is_returned() {
        let graph_ir = GraphIr::new(128, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("force", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_output(AttributeSemantic::Velocity);
        let set_position = BuiltinModule::SetPosition;
        assert!(matches!(
            compile(
                &graph_ir,
                &[StageModule {
                    stage_index: 0,
                    module: &set_position,
                }]
            ),
            Err(CompileError::ModuleAttributeUndeclared {
                module: 0,
                stage: 0,
                semantic: AttributeSemantic::Position,
            })
        ));

        let fragment = SnippetModule {
            stage: StageKind::Update,
            resources: Vec::new(),
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph_ir,
            &[StageModule {
                stage_index: 0,
                module: &fragment,
            }],
        )
        .unwrap();
        assert_eq!(compiled.attribute_layout.capacity, 128);
        assert_eq!(compiled.attribute_layout.attributes.len(), 1);
        assert_eq!(
            compiled.attribute_layout.attributes[0].semantic,
            AttributeSemantic::Velocity
        );
    }

    #[test]
    fn attribute_abi_emits_typed_storage_and_ping_pong_bindings() {
        let writer = SnippetModule {
            stage: StageKind::Update,
            resources: Vec::new(),
            fragments: vec!["ember_attr_velocity[particle_index] = vec3<f32>(0.0);"],
        };
        let current_graph = GraphIr::new(32, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("write_velocity", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_output(AttributeSemantic::Velocity);
        let compiled = compile(
            &current_graph,
            &[StageModule {
                stage_index: 0,
                module: &writer,
            }],
        )
        .unwrap();
        assert_eq!(compiled.attribute_bindings.len(), 1);
        assert_eq!(
            compiled.attribute_bindings[0].role,
            AttributeBufferRole::Single
        );
        assert_eq!(
            compiled.attribute_bindings[0].access,
            ResourceAccess::ReadWrite
        );
        assert!(compiled.source.contains(
            "@group(0) @binding(0) var<storage, read_write> ember_attr_velocity: array<vec3<f32>>;"
        ));

        let ping_pong_graph = GraphIr::new(32, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("history", IterationDomain::PerParticle, 1)
                    .reading(read_prev(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_output(AttributeSemantic::Velocity);
        let compiled = compile(&ping_pong_graph, &[]).unwrap();
        assert_eq!(compiled.attribute_layout.attributes[0].copies, 2);
        assert_eq!(compiled.attribute_bindings.len(), 2);
        assert_eq!(
            compiled.attribute_bindings[0].role,
            AttributeBufferRole::Previous
        );
        assert_eq!(
            compiled.attribute_bindings[0].access,
            ResourceAccess::ReadOnly
        );
        assert_eq!(
            compiled.attribute_bindings[1].role,
            AttributeBufferRole::Current
        );
        assert_eq!(
            compiled.attribute_bindings[1].access,
            ResourceAccess::ReadWrite
        );
        assert_eq!(
            compiled.attribute_bindings_for_frame(0),
            vec![
                RuntimeAttributeBinding {
                    semantic: AttributeSemantic::Velocity,
                    role: AttributeBufferRole::Previous,
                    binding: 0,
                    physical_copy: 1,
                },
                RuntimeAttributeBinding {
                    semantic: AttributeSemantic::Velocity,
                    role: AttributeBufferRole::Current,
                    binding: 1,
                    physical_copy: 0,
                },
            ]
        );
        assert_eq!(compiled.attribute_bindings_for_frame(1)[0].physical_copy, 0);
        assert_eq!(compiled.attribute_bindings_for_frame(1)[1].physical_copy, 1);
        validate_wgsl(&compiled.source).unwrap();
    }

    #[test]
    fn module_access_contract_selects_previous_and_current_ping_pong_sides() {
        let module = BuiltinModule::InheritVelocity;
        let graph = GraphIr::new(32, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("inherit", IterationDomain::PerParticle, 1)
                    .reading(read_prev(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_output(AttributeSemantic::Velocity);
        let compiled = compile(
            &graph,
            &[StageModule {
                stage_index: 0,
                module: &module,
            }],
        )
        .unwrap();
        assert!(compiled
            .source
            .contains("ember_attr_velocity_current[index] = emitter_velocity.value.xyz + ember_attr_velocity_previous[index]"));
        assert!(compiled
            .source
            .contains("var<storage, read> ember_attr_velocity_previous"));
        assert!(compiled
            .source
            .contains("var<storage, read_write> ember_attr_velocity_current"));
        validate_wgsl(&compiled.source).unwrap();

        let bad_graph = GraphIr::new(32, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("inherit", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_output(AttributeSemantic::Velocity);
        assert!(matches!(
            compile(
                &bad_graph,
                &[StageModule {
                    stage_index: 0,
                    module: &module,
                }]
            ),
            Err(CompileError::ModuleAttributeAccessUndeclared {
                semantic: AttributeSemantic::Velocity,
                ..
            })
        ));
    }

    #[test]
    fn every_builtin_module_emits_a_naga_validated_kernel() {
        for module in BuiltinModule::ALL {
            let mut stage = StageIr::new(module.kernel_name(), module.stage().default_domain(), 1);
            for usage in module.attribute_accesses() {
                if usage.access.contains(AttributeAccess::READ)
                    || usage.access.contains(AttributeAccess::READ_PREV)
                {
                    stage = stage.reading(usage);
                }
                if usage.access.contains(AttributeAccess::WRITE) {
                    stage = stage.writing(usage);
                }
            }
            let mut graph = GraphIr::new(64, EmberShadingModel::Unlit, PlatformProfile::Desktop)
                .with_stage(stage);
            for &semantic in module.required_attributes() {
                graph = graph.with_output(semantic);
            }
            let compiled = compile(
                &graph,
                &[StageModule {
                    stage_index: 0,
                    module: &module,
                }],
            )
            .unwrap_or_else(|error| {
                panic!(
                    "builtin {} failed to compile: {error:?}",
                    module.kernel_name()
                )
            });
            assert!(
                compiled
                    .source
                    .contains(&format!("fn {}(index: u32)", module.kernel_function_name())),
                "missing kernel definition for {}",
                module.kernel_name()
            );
        }
    }

    #[test]
    fn every_builtin_data_interface_emits_naga_validated_functions() {
        for interface in BuiltinDataInterface::ALL {
            let graph =
                GraphIr::new(64, EmberShadingModel::Unlit, PlatformProfile::Desktop).with_stage(
                    StageIr::new("data_interface_validation", IterationDomain::PerParticle, 1),
                );
            let interfaces: [&dyn DataInterface; 1] = [&interface];
            let compiled =
                compile_with_data_interfaces(&graph, &[], &interfaces).unwrap_or_else(|error| {
                    panic!(
                        "data interface {:?} failed to compile: {error:?}",
                        interface
                    )
                });
            for &function in interface.sample_functions() {
                assert!(
                    compiled.source.contains(&format!("fn {function}(")),
                    "missing data-interface function {function} for {interface:?}"
                );
            }
        }
    }

    #[test]
    fn conflicting_names_fail_and_colliding_module_slots_are_reallocated() {
        let first = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("params", 2, 0)],
            fragments: Vec::new(),
        };
        let conflicting_name = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::texture2d("params", 2, 1)],
            fragments: Vec::new(),
        };
        assert!(matches!(
            compile(
                &graph(),
                &[
                    StageModule {
                        stage_index: 0,
                        module: &first
                    },
                    StageModule {
                        stage_index: 0,
                        module: &conflicting_name
                    },
                ]
            ),
            Err(CompileError::BindingNameConflict { .. })
        ));

        let conflicting_slot = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::texture2d("image", 2, 0)],
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph(),
            &[
                StageModule {
                    stage_index: 0,
                    module: &first,
                },
                StageModule {
                    stage_index: 0,
                    module: &conflicting_slot,
                },
            ],
        )
        .unwrap();
        assert_eq!(compiled.bindings[0].binding, 0);
        assert_eq!(compiled.bindings[1].binding, 1);
        assert_eq!(compiled.bindings[1].name, "image");
        validate_wgsl(&compiled.source).unwrap();
    }

    #[test]
    fn runtime_dispatch_buffer_slot_is_reserved() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("module_params", 1, 0)],
            fragments: Vec::new(),
        };
        assert!(matches!(
            compile(
                &graph(),
                &[StageModule {
                    stage_index: 0,
                    module: &module,
                }]
            ),
            Err(CompileError::BindingSlotConflict {
                group: 1,
                binding: 0,
            })
        ));
    }

    #[test]
    fn module_binding_contract_mismatch_is_reported() {
        struct MissingEmission;
        impl EmberModule for MissingEmission {
            fn stage(&self) -> StageKind {
                StageKind::Update
            }
            fn required_attributes(&self) -> &[AttributeSemantic] {
                &[]
            }
            fn resources(&self) -> &[ResourceDecl] {
                static RESOURCES: [ResourceDecl; 1] = [ResourceDecl::uniform("params", 2, 0)];
                &RESOURCES
            }
            fn emit_wesl(&self, _: &mut CodegenCtx) {}
        }
        assert!(matches!(
            compile(
                &graph(),
                &[StageModule {
                    stage_index: 0,
                    module: &MissingEmission
                }]
            ),
            Err(CompileError::ModuleBindingMismatch { module: 0, .. })
        ));
    }

    #[test]
    fn naga_rejects_incomplete_or_invalid_module_fragments() {
        let unresolved = SnippetModule {
            stage: StageKind::Update,
            resources: Vec::new(),
            fragments: vec!["missing_kernel(particle_index);"],
        };
        assert!(matches!(
            compile(
                &graph(),
                &[StageModule {
                    stage_index: 0,
                    module: &unresolved
                }]
            ),
            Err(CompileError::InvalidGeneratedWgsl { .. })
        ));

        let malformed = SnippetModule {
            stage: StageKind::Update,
            resources: Vec::new(),
            fragments: vec!["let broken = ;"],
        };
        assert!(matches!(
            compile(
                &graph(),
                &[StageModule {
                    stage_index: 0,
                    module: &malformed
                }]
            ),
            Err(CompileError::InvalidGeneratedWgsl {
                phase: ShaderValidationPhase::Parse,
                ..
            })
        ));
    }

    #[test]
    fn valid_resource_abi_and_generated_entry_points_validate() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![
                ResourceDecl::uniform("params", 2, 0),
                ResourceDecl::buffer("indices", 2, 1),
                ResourceDecl::curve("curve_lut", 2, 2),
                ResourceDecl::texture2d("surface", 2, 3),
                ResourceDecl::texture3d("volume", 2, 4),
            ],
            fragments: vec![
                "let value: f32 = params.weight + params.direction.x + f32(indices[particle_index]) + curve_lut[0].x;",
            ],
        };
        let compiled = compile(
            &graph(),
            &[StageModule {
                stage_index: 0,
                module: &module,
            }],
        )
        .unwrap();
        assert_eq!(compiled.bindings.len(), 6);
        assert!(compiled.source.contains(
            "@group(1) @binding(0) var<storage, read> ember_dispatch_counts: array<u32>"
        ));
        assert!(compiled.source.contains("struct ember_params_params {"));
        assert!(compiled
            .source
            .contains("@group(2) @binding(0) var<uniform> params: ember_params_params;"));
        assert!(compiled
            .source
            .contains("var<storage, read> indices: array<u32>"));
        assert!(compiled
            .source
            .contains("var<storage, read> curve_lut: array<vec4<f32>>"));
        assert!(compiled.source.contains("var surface: texture_2d<f32>"));
        assert!(compiled.source.contains("var volume: texture_3d<f32>"));
        assert!(compiled
            .source
            .contains("fn ember_sample_curve_lut(age: f32) -> vec4<f32>"));
        assert!(compiled.source.contains("arrayLength(&curve_lut)"));
        assert!(compiled.source.contains("ember_dispatch_counts[0u]"));
        validate_wgsl(&compiled.source).unwrap();
    }

    #[test]
    fn parameter_block_layout_defaults_and_runtime_encoding_match_wgsl_abi() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("params", 2, 0)],
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph(),
            &[StageModule {
                stage_index: 0,
                module: &module,
            }],
        )
        .unwrap();
        let block = &compiled.parameter_blocks[0];
        assert_eq!(block.size, 32);
        assert_eq!(block.fields[0].offset, 0);
        assert_eq!(block.fields[1].offset, 16);
        assert_eq!(block.default_bytes.len(), 32);
        assert_eq!(block.default_bytes[0..4], 0.5f32.to_bits().to_le_bytes());
        assert!(block.default_bytes[4..16].iter().all(|byte| *byte == 0));
        assert_eq!(block.default_bytes[16..20], 1.0f32.to_bits().to_le_bytes());
        assert_eq!(block.default_bytes[20..24], 2.0f32.to_bits().to_le_bytes());
        assert_eq!(block.default_bytes[24..28], 3.0f32.to_bits().to_le_bytes());
        assert!(block.default_bytes[28..32].iter().all(|byte| *byte == 0));

        let encoded = block
            .encode(&[
                ("weight", ParameterValue::F32(1.5)),
                ("direction", ParameterValue::Vec3([4.0, 5.0, 6.0])),
            ])
            .unwrap();
        assert_eq!(encoded[0..4], 1.5f32.to_bits().to_le_bytes());
        assert!(encoded[4..16].iter().all(|byte| *byte == 0));
        assert_eq!(encoded[16..20], 4.0f32.to_bits().to_le_bytes());
        assert_eq!(encoded[20..24], 5.0f32.to_bits().to_le_bytes());
        assert_eq!(encoded[24..28], 6.0f32.to_bits().to_le_bytes());
        assert!(encoded[28..32].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn authored_asset_parameters_resolve_to_compiled_uniform_bytes() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("params", 2, 0)],
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph(),
            &[StageModule {
                stage_index: 0,
                module: &module,
            }],
        )
        .unwrap();
        let asset = EmberAsset {
            format_version: 1,
            graph: super::super::authoring::GraphAssetRef(7),
            params: alloc::vec![
                super::super::authoring::ExposedParam {
                    handle: ParamHandle(1),
                    name: String::from("Authored intensity"),
                    target: super::super::authoring::ParameterTarget {
                        resource: String::from("params"),
                        field: String::from("weight"),
                    },
                    ty: ParamType::Scalar,
                    binding: BindingSource::Constant(ParamValue::Scalar(0.75)),
                },
                super::super::authoring::ExposedParam {
                    handle: ParamHandle(2),
                    name: String::from("wind / direction"),
                    target: super::super::authoring::ParameterTarget {
                        resource: String::from("params"),
                        field: String::from("direction"),
                    },
                    ty: ParamType::Vec3,
                    binding: BindingSource::Gameplay(3),
                },
            ],
            renderer_config: super::super::authoring::RendererConfig {
                id: super::super::authoring::RendererConfigId(0),
                kind: super::super::authoring::RendererKind::Sprite,
                writes_motion_vectors: false,
            },
            shading: super::super::authoring::ShadingSettings {
                model: EmberShadingModel::Unlit,
                receive_shadows: false,
                cast_shadows: false,
            },
            format: super::super::authoring::SerializationFormat::Binary,
        };

        let missing = compiled.prepare_asset_parameters(&asset, &[]);
        assert_eq!(
            missing,
            Err(AssetParameterError::MissingRuntimeValue(ParamHandle(2)))
        );
        let constant_override =
            compiled.prepare_asset_parameters(&asset, &[(ParamHandle(1), ParamValue::Scalar(0.5))]);
        assert_eq!(
            constant_override,
            Err(AssetParameterError::UnexpectedRuntimeValue(ParamHandle(1)))
        );

        let prepared = compiled
            .prepare_asset_parameters(
                &asset,
                &[(ParamHandle(2), ParamValue::Vec3([4.0, 5.0, 6.0]))],
            )
            .unwrap();
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].resource, "params");
        assert_eq!((prepared[0].group, prepared[0].binding), (2, 0));
        assert_eq!(prepared[0].bytes[0..4], 0.75f32.to_bits().to_le_bytes());
        assert!(prepared[0].bytes[4..16].iter().all(|byte| *byte == 0));
        assert_eq!(prepared[0].bytes[16..20], 4.0f32.to_bits().to_le_bytes());
        assert_eq!(prepared[0].bytes[20..24], 5.0f32.to_bits().to_le_bytes());
        assert_eq!(prepared[0].bytes[24..28], 6.0f32.to_bits().to_le_bytes());
        assert!(prepared[0].bytes[28..32].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn authored_parameter_validation_distinguishes_mismatch_unsupported_and_non_finite() {
        assert_eq!(
            authored_parameter_value(ParamHandle(4), ParamType::Bool, ParamValue::Scalar(1.0)),
            Err(AssetParameterError::AuthoredTypeMismatch {
                handle: ParamHandle(4),
                ty: ParamType::Bool,
            })
        );
        assert_eq!(
            authored_parameter_value(ParamHandle(5), ParamType::Bool, ParamValue::Bool(true)),
            Ok(ParameterValue::U32(1))
        );
        assert_eq!(
            authored_parameter_value(ParamHandle(7), ParamType::Int, ParamValue::Int(-17)),
            Ok(ParameterValue::I32(-17))
        );
        assert_eq!(
            authored_parameter_value(
                ParamHandle(6),
                ParamType::Scalar,
                ParamValue::Scalar(f32::INFINITY),
            ),
            Err(AssetParameterError::NonFiniteValue(ParamHandle(6)))
        );
    }

    #[test]
    fn authored_integer_and_boolean_values_encode_to_host_shareable_wgsl_lanes() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("controls", 2, 0)],
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph(),
            &[StageModule {
                stage_index: 0,
                module: &module,
            }],
        )
        .unwrap();
        assert!(compiled.source.contains("steps: i32"));
        assert!(compiled.source.contains("enabled_bits: u32"));
        let asset = EmberAsset {
            format_version: 1,
            graph: super::super::authoring::GraphAssetRef(8),
            params: alloc::vec![
                super::super::authoring::ExposedParam {
                    handle: ParamHandle(3),
                    name: String::from("Iteration count"),
                    target: super::super::authoring::ParameterTarget {
                        resource: String::from("controls"),
                        field: String::from("steps"),
                    },
                    ty: ParamType::Int,
                    binding: BindingSource::Constant(ParamValue::Int(-17)),
                },
                super::super::authoring::ExposedParam {
                    handle: ParamHandle(4),
                    name: String::from("Use turbulence"),
                    target: super::super::authoring::ParameterTarget {
                        resource: String::from("controls"),
                        field: String::from("enabled_bits"),
                    },
                    ty: ParamType::Bool,
                    binding: BindingSource::Constant(ParamValue::Bool(true)),
                },
            ],
            renderer_config: super::super::authoring::RendererConfig {
                id: super::super::authoring::RendererConfigId(0),
                kind: super::super::authoring::RendererKind::Sprite,
                writes_motion_vectors: false,
            },
            shading: super::super::authoring::ShadingSettings {
                model: EmberShadingModel::Unlit,
                receive_shadows: false,
                cast_shadows: false,
            },
            format: super::super::authoring::SerializationFormat::Binary,
        };
        let payload = compiled
            .prepare_asset_parameters(&asset, &[])
            .unwrap()
            .remove(0)
            .bytes;
        assert_eq!(payload[0..4], (-17i32).to_le_bytes());
        assert_eq!(payload[4..8], 1u32.to_le_bytes());
        assert!(payload[8..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn one_field_parameter_lowering_preserves_vector_swizzles() {
        static FIELDS: &[super::super::modules::ParameterField] =
            &[super::super::modules::ParameterField::new(
                "value",
                ParameterType::Vec4,
                ParameterValue::Vec4([0.0; 4]),
            )];
        let schemas = [ParameterBlockDecl::new("params", FIELDS)];

        assert_eq!(
            resolve_parameter_identifiers("params.xyz + params.y + params.value.x", &schemas,),
            "params.value.xyz + params.value.y + params.value.x"
        );
    }

    #[test]
    fn parameter_block_encoding_rejects_missing_extra_and_mistyped_values() {
        let module = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("params", 2, 0)],
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph(),
            &[StageModule {
                stage_index: 0,
                module: &module,
            }],
        )
        .unwrap();
        let block = &compiled.parameter_blocks[0];
        assert_eq!(
            block.encode(&[("weight", ParameterValue::F32(1.0))]),
            Err(ParameterEncodeError::MissingField {
                name: String::from("direction"),
            })
        );
        assert_eq!(
            block.encode(&[
                ("weight", ParameterValue::F32(1.0)),
                ("direction", ParameterValue::Vec3([0.0; 3])),
                ("extra", ParameterValue::U32(9)),
            ]),
            Err(ParameterEncodeError::UnknownField {
                name: String::from("extra"),
            })
        );
        assert_eq!(
            block.encode(&[
                ("weight", ParameterValue::U32(1)),
                ("direction", ParameterValue::Vec3([0.0; 3])),
            ]),
            Err(ParameterEncodeError::TypeMismatch {
                name: String::from("weight"),
                expected: ParameterType::F32,
                actual: ParameterType::U32,
            })
        );
    }

    #[test]
    fn compatible_uniform_schemas_are_shared_across_modules() {
        let first = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("params", 2, 0)],
            fragments: Vec::new(),
        };
        let second = SnippetModule {
            stage: StageKind::Update,
            resources: alloc::vec![ResourceDecl::uniform("params", 2, 0)],
            fragments: Vec::new(),
        };
        let compiled = compile(
            &graph(),
            &[
                StageModule {
                    stage_index: 0,
                    module: &first,
                },
                StageModule {
                    stage_index: 0,
                    module: &second,
                },
            ],
        )
        .unwrap();
        assert_eq!(compiled.parameter_blocks.len(), 1);
        assert_eq!(
            compiled
                .bindings
                .iter()
                .filter(|item| item.name == "params")
                .count(),
            1
        );
        validate_wgsl(&compiled.source).unwrap();
    }

    #[test]
    fn curve_resource_rejects_non_vec4_or_writable_abi() {
        for curve in [
            ResourceDecl::new(
                "bad_curve_format",
                ResourceKind::Curve,
                ResourceAccess::ReadOnly,
                AttributeFormat::F32,
                2,
                0,
            ),
            ResourceDecl::new(
                "writable_curve",
                ResourceKind::Curve,
                ResourceAccess::ReadWrite,
                AttributeFormat::Vec4,
                2,
                0,
            ),
        ] {
            let module = SnippetModule {
                stage: StageKind::Update,
                resources: alloc::vec![curve],
                fragments: Vec::new(),
            };
            assert!(matches!(
                compile(
                    &graph(),
                    &[StageModule {
                        stage_index: 0,
                        module: &module
                    }]
                ),
                Err(CompileError::UnsupportedBinding { .. })
            ));
        }
    }

    #[test]
    fn curve_sampler_is_clamped_and_handles_empty_and_single_sample_luts() {
        let curve = SnippetModule {
            stage: StageKind::Lifecycle,
            resources: alloc::vec![ResourceDecl::curve("age_curve", 2, 0)],
            fragments: vec!["let sample: f32 = ember_sample_age_curve(0.5).x;"],
        };
        let curve_graph = GraphIr::new(16, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("age", IterationDomain::PerParticle, 1));
        let compiled = compile(
            &curve_graph,
            &[StageModule {
                stage_index: 0,
                module: &curve,
            }],
        )
        .unwrap();
        assert!(compiled.source.contains("if (count == 0u)"));
        assert!(compiled.source.contains("clamp(age, 0.0, 1.0)"));
        assert!(compiled.source.contains("min(lower + 1u, count - 1u)"));
        validate_wgsl(&compiled.source).unwrap();
    }
}
