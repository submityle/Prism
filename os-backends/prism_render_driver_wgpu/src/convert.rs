//! Pure, GPU-free translation between the frozen `prism_render_driver` RHI
//! types and their `wgpu` 30 equivalents.
//!
//! Every function here is a deterministic value mapping with no device state,
//! which keeps them unit-testable without a GPU. Id-resolving translations
//! (bind groups, pipelines, command replay) live in [`crate::device`] and
//! [`crate::queue`] because they need the device's slot maps.

use core::num::{NonZeroU32, NonZeroU64};

use prism_render_driver as rhi;

/// Reinterprets a generational id from one resource kind to another.
///
/// The RHI's `*Kind` marker types are not re-exported, so the backend stores
/// resources under its own private markers. Ids are structurally identical
/// `(index, generation)` pairs, so this rebuilds the id under a different tag.
pub(crate) fn convert_id<A: ?Sized, B: ?Sized>(id: rhi::ResourceId<A>) -> rhi::ResourceId<B> {
    rhi::ResourceId::from_parts(id.index(), id.generation())
}

/// Translates a texture format.
pub(crate) fn texture_format(format: rhi::TextureFormat) -> wgpu::TextureFormat {
    use rhi::TextureFormat as F;
    use wgpu::TextureFormat as W;
    match format {
        F::R8Unorm => W::R8Unorm,
        F::R8Uint => W::R8Uint,
        F::R16Float => W::R16Float,
        F::R32Float => W::R32Float,
        F::R32Uint => W::R32Uint,
        F::Rg8Unorm => W::Rg8Unorm,
        F::Rg16Float => W::Rg16Float,
        F::Rg32Float => W::Rg32Float,
        F::Rgba8Unorm => W::Rgba8Unorm,
        F::Rgba8UnormSrgb => W::Rgba8UnormSrgb,
        F::Bgra8Unorm => W::Bgra8Unorm,
        F::Bgra8UnormSrgb => W::Bgra8UnormSrgb,
        F::Rgb10a2Unorm => W::Rgb10a2Unorm,
        F::Rg11b10Float => W::Rg11b10Ufloat,
        F::Rgba16Float => W::Rgba16Float,
        F::Rgba32Float => W::Rgba32Float,
        F::Depth32Float => W::Depth32Float,
        F::Depth24PlusStencil8 => W::Depth24PlusStencil8,
        // The RHI format set is `#[non_exhaustive]`; every variant known at
        // this crate's pin is handled above. A future RHI format with no wgpu
        // analogue is a hard configuration error rather than a silent default.
        other => panic!("unsupported RHI texture format for wgpu backend: {other:?}"),
    }
}

/// Translates a vertex attribute format.
pub(crate) fn vertex_format(format: rhi::VertexFormat) -> wgpu::VertexFormat {
    use rhi::VertexFormat as F;
    use wgpu::VertexFormat as W;
    match format {
        F::Float32 => W::Float32,
        F::Float32x2 => W::Float32x2,
        F::Float32x3 => W::Float32x3,
        F::Float32x4 => W::Float32x4,
        F::Float16x2 => W::Float16x2,
        F::Float16x4 => W::Float16x4,
        F::Uint32 => W::Uint32,
        F::Uint32x2 => W::Uint32x2,
        F::Uint32x4 => W::Uint32x4,
        F::Sint32 => W::Sint32,
        F::Unorm8x4 => W::Unorm8x4,
        F::Snorm8x4 => W::Snorm8x4,
        other => panic!("unsupported RHI vertex format for wgpu backend: {other:?}"),
    }
}

/// Translates buffer usage flags.
pub(crate) fn buffer_usages(usage: rhi::BufferUsages) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::empty();
    if usage.contains(rhi::BufferUsages::COPY_SRC) {
        out |= wgpu::BufferUsages::COPY_SRC;
    }
    if usage.contains(rhi::BufferUsages::COPY_DST) {
        out |= wgpu::BufferUsages::COPY_DST;
    }
    if usage.contains(rhi::BufferUsages::INDEX) {
        out |= wgpu::BufferUsages::INDEX;
    }
    if usage.contains(rhi::BufferUsages::VERTEX) {
        out |= wgpu::BufferUsages::VERTEX;
    }
    if usage.contains(rhi::BufferUsages::UNIFORM) {
        out |= wgpu::BufferUsages::UNIFORM;
    }
    if usage.contains(rhi::BufferUsages::STORAGE) {
        out |= wgpu::BufferUsages::STORAGE;
    }
    if usage.contains(rhi::BufferUsages::INDIRECT) {
        out |= wgpu::BufferUsages::INDIRECT;
    }
    out
}

/// Translates texture usage flags.
pub(crate) fn texture_usages(usage: rhi::TextureUsages) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    if usage.contains(rhi::TextureUsages::COPY_SRC) {
        out |= wgpu::TextureUsages::COPY_SRC;
    }
    if usage.contains(rhi::TextureUsages::COPY_DST) {
        out |= wgpu::TextureUsages::COPY_DST;
    }
    if usage.contains(rhi::TextureUsages::TEXTURE_BINDING) {
        out |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(rhi::TextureUsages::STORAGE_BINDING) {
        out |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if usage.contains(rhi::TextureUsages::RENDER_ATTACHMENT) {
        out |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    out
}

/// Translates shader stage visibility flags.
pub(crate) fn shader_stages(stages: rhi::ShaderStages) -> wgpu::ShaderStages {
    let mut out = wgpu::ShaderStages::empty();
    if stages.contains(rhi::ShaderStages::VERTEX) {
        out |= wgpu::ShaderStages::VERTEX;
    }
    if stages.contains(rhi::ShaderStages::FRAGMENT) {
        out |= wgpu::ShaderStages::FRAGMENT;
    }
    if stages.contains(rhi::ShaderStages::COMPUTE) {
        out |= wgpu::ShaderStages::COMPUTE;
    }
    out
}

/// Translates color write-mask flags.
pub(crate) fn color_writes(mask: rhi::ColorWrites) -> wgpu::ColorWrites {
    let mut out = wgpu::ColorWrites::empty();
    if mask.contains(rhi::ColorWrites::RED) {
        out |= wgpu::ColorWrites::RED;
    }
    if mask.contains(rhi::ColorWrites::GREEN) {
        out |= wgpu::ColorWrites::GREEN;
    }
    if mask.contains(rhi::ColorWrites::BLUE) {
        out |= wgpu::ColorWrites::BLUE;
    }
    if mask.contains(rhi::ColorWrites::ALPHA) {
        out |= wgpu::ColorWrites::ALPHA;
    }
    out
}

/// Translates a texture storage dimension.
pub(crate) fn texture_dimension(dim: rhi::TextureDimension) -> wgpu::TextureDimension {
    match dim {
        rhi::TextureDimension::D1 => wgpu::TextureDimension::D1,
        rhi::TextureDimension::D2 => wgpu::TextureDimension::D2,
        rhi::TextureDimension::D3 => wgpu::TextureDimension::D3,
    }
}

/// Translates a texture view dimension.
pub(crate) fn texture_view_dimension(dim: rhi::TextureViewDimension) -> wgpu::TextureViewDimension {
    match dim {
        rhi::TextureViewDimension::D1 => wgpu::TextureViewDimension::D1,
        rhi::TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
        rhi::TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
        rhi::TextureViewDimension::Cube => wgpu::TextureViewDimension::Cube,
        rhi::TextureViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        rhi::TextureViewDimension::D3 => wgpu::TextureViewDimension::D3,
    }
}

/// Translates a texture aspect selector.
pub(crate) fn texture_aspect(aspect: rhi::TextureAspect) -> wgpu::TextureAspect {
    match aspect {
        rhi::TextureAspect::All => wgpu::TextureAspect::All,
        rhi::TextureAspect::DepthOnly => wgpu::TextureAspect::DepthOnly,
        rhi::TextureAspect::StencilOnly => wgpu::TextureAspect::StencilOnly,
    }
}

/// Translates a texel extent.
pub(crate) fn extent3d(extent: rhi::Extent3d) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: extent.depth_or_array_layers,
    }
}

/// Translates a sampler addressing mode.
pub(crate) fn address_mode(mode: rhi::AddressMode) -> wgpu::AddressMode {
    match mode {
        rhi::AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        rhi::AddressMode::Repeat => wgpu::AddressMode::Repeat,
        rhi::AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        rhi::AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

/// Translates a min/mag filter mode.
pub(crate) fn filter_mode(mode: rhi::FilterMode) -> wgpu::FilterMode {
    match mode {
        rhi::FilterMode::Nearest => wgpu::FilterMode::Nearest,
        rhi::FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

/// Translates a mip filter mode to wgpu's distinct `MipmapFilterMode`.
pub(crate) fn mipmap_filter_mode(mode: rhi::FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        rhi::FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        rhi::FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

/// Translates a sampler border color.
pub(crate) fn sampler_border_color(color: rhi::SamplerBorderColor) -> wgpu::SamplerBorderColor {
    match color {
        rhi::SamplerBorderColor::TransparentBlack => wgpu::SamplerBorderColor::TransparentBlack,
        rhi::SamplerBorderColor::OpaqueBlack => wgpu::SamplerBorderColor::OpaqueBlack,
        rhi::SamplerBorderColor::OpaqueWhite => wgpu::SamplerBorderColor::OpaqueWhite,
    }
}

/// Translates a comparison predicate.
pub(crate) fn compare_function(func: rhi::CompareFunction) -> wgpu::CompareFunction {
    match func {
        rhi::CompareFunction::Never => wgpu::CompareFunction::Never,
        rhi::CompareFunction::Less => wgpu::CompareFunction::Less,
        rhi::CompareFunction::Equal => wgpu::CompareFunction::Equal,
        rhi::CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        rhi::CompareFunction::Greater => wgpu::CompareFunction::Greater,
        rhi::CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        rhi::CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        rhi::CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

/// Translates a texture sample type.
pub(crate) fn texture_sample_type(ty: rhi::TextureSampleType) -> wgpu::TextureSampleType {
    match ty {
        rhi::TextureSampleType::Float => wgpu::TextureSampleType::Float { filterable: true },
        rhi::TextureSampleType::UnfilterableFloat => {
            wgpu::TextureSampleType::Float { filterable: false }
        }
        rhi::TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
        rhi::TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
        rhi::TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
    }
}

/// Translates a buffer binding type.
pub(crate) fn buffer_binding_type(ty: rhi::BufferBindingType) -> wgpu::BufferBindingType {
    match ty {
        rhi::BufferBindingType::Uniform => wgpu::BufferBindingType::Uniform,
        rhi::BufferBindingType::Storage { read_only } => {
            wgpu::BufferBindingType::Storage { read_only }
        }
    }
}

/// Translates a sampler binding type.
pub(crate) fn sampler_binding_type(ty: rhi::SamplerBindingType) -> wgpu::SamplerBindingType {
    match ty {
        rhi::SamplerBindingType::Filtering => wgpu::SamplerBindingType::Filtering,
        rhi::SamplerBindingType::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
        rhi::SamplerBindingType::Comparison => wgpu::SamplerBindingType::Comparison,
    }
}

/// Translates a storage texture access mode.
pub(crate) fn storage_texture_access(
    access: rhi::StorageTextureAccess,
) -> wgpu::StorageTextureAccess {
    match access {
        rhi::StorageTextureAccess::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
        rhi::StorageTextureAccess::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
        rhi::StorageTextureAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
    }
}

/// Translates a single binding slot type.
pub(crate) fn binding_type(ty: rhi::BindingType) -> wgpu::BindingType {
    match ty {
        rhi::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => wgpu::BindingType::Buffer {
            ty: buffer_binding_type(ty),
            has_dynamic_offset,
            min_binding_size: min_binding_size.and_then(NonZeroU64::new),
        },
        rhi::BindingType::Sampler(ty) => wgpu::BindingType::Sampler(sampler_binding_type(ty)),
        rhi::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => wgpu::BindingType::Texture {
            sample_type: texture_sample_type(sample_type),
            view_dimension: texture_view_dimension(view_dimension),
            multisampled,
        },
        rhi::BindingType::StorageTexture {
            access,
            format,
            view_dimension,
        } => wgpu::BindingType::StorageTexture {
            access: storage_texture_access(access),
            format: texture_format(format),
            view_dimension: texture_view_dimension(view_dimension),
        },
    }
}

/// Translates a bind group layout entry.
pub(crate) fn bind_group_layout_entry(
    entry: &rhi::BindGroupLayoutEntry,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: shader_stages(entry.visibility),
        ty: binding_type(entry.ty),
        count: entry.count.and_then(NonZeroU32::new),
    }
}

/// Translates a blend factor.
pub(crate) fn blend_factor(factor: rhi::BlendFactor) -> wgpu::BlendFactor {
    match factor {
        rhi::BlendFactor::Zero => wgpu::BlendFactor::Zero,
        rhi::BlendFactor::One => wgpu::BlendFactor::One,
        rhi::BlendFactor::Src => wgpu::BlendFactor::Src,
        rhi::BlendFactor::OneMinusSrc => wgpu::BlendFactor::OneMinusSrc,
        rhi::BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        rhi::BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        rhi::BlendFactor::Dst => wgpu::BlendFactor::Dst,
        rhi::BlendFactor::OneMinusDst => wgpu::BlendFactor::OneMinusDst,
        rhi::BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        rhi::BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        rhi::BlendFactor::Constant => wgpu::BlendFactor::Constant,
        rhi::BlendFactor::OneMinusConstant => wgpu::BlendFactor::OneMinusConstant,
    }
}

/// Translates a blend operation.
pub(crate) fn blend_operation(op: rhi::BlendOperation) -> wgpu::BlendOperation {
    match op {
        rhi::BlendOperation::Add => wgpu::BlendOperation::Add,
        rhi::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        rhi::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        rhi::BlendOperation::Min => wgpu::BlendOperation::Min,
        rhi::BlendOperation::Max => wgpu::BlendOperation::Max,
    }
}

/// Translates a per-channel blend equation.
pub(crate) fn blend_component(component: rhi::BlendComponent) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: blend_factor(component.src_factor),
        dst_factor: blend_factor(component.dst_factor),
        operation: blend_operation(component.operation),
    }
}

/// Translates a full color/alpha blend state.
pub(crate) fn blend_state(state: rhi::BlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: blend_component(state.color),
        alpha: blend_component(state.alpha),
    }
}

/// Translates a primitive topology.
pub(crate) fn primitive_topology(topology: rhi::PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match topology {
        rhi::PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        rhi::PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        rhi::PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        rhi::PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        rhi::PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

/// Translates a front-face winding order.
pub(crate) fn front_face(front: rhi::FrontFace) -> wgpu::FrontFace {
    match front {
        rhi::FrontFace::Ccw => wgpu::FrontFace::Ccw,
        rhi::FrontFace::Cw => wgpu::FrontFace::Cw,
    }
}

/// Translates a cull face.
pub(crate) fn face(face: rhi::Face) -> wgpu::Face {
    match face {
        rhi::Face::Front => wgpu::Face::Front,
        rhi::Face::Back => wgpu::Face::Back,
    }
}

/// Translates a polygon fill mode.
pub(crate) fn polygon_mode(mode: rhi::PolygonMode) -> wgpu::PolygonMode {
    match mode {
        rhi::PolygonMode::Fill => wgpu::PolygonMode::Fill,
        rhi::PolygonMode::Line => wgpu::PolygonMode::Line,
        rhi::PolygonMode::Point => wgpu::PolygonMode::Point,
    }
}

/// Translates an index format.
pub(crate) fn index_format(format: rhi::IndexFormat) -> wgpu::IndexFormat {
    match format {
        rhi::IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        rhi::IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

/// Translates a vertex step mode.
pub(crate) fn vertex_step_mode(mode: rhi::VertexStepMode) -> wgpu::VertexStepMode {
    match mode {
        rhi::VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        rhi::VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    }
}

/// Translates full primitive assembly/rasterization state.
pub(crate) fn primitive_state(state: rhi::PrimitiveState) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: primitive_topology(state.topology),
        strip_index_format: state.strip_index_format.map(index_format),
        front_face: front_face(state.front_face),
        cull_mode: state.cull_mode.map(face),
        unclipped_depth: state.unclipped_depth,
        polygon_mode: polygon_mode(state.polygon_mode),
        conservative: false,
    }
}

/// Translates a stencil operation.
pub(crate) fn stencil_operation(op: rhi::StencilOperation) -> wgpu::StencilOperation {
    match op {
        rhi::StencilOperation::Keep => wgpu::StencilOperation::Keep,
        rhi::StencilOperation::Zero => wgpu::StencilOperation::Zero,
        rhi::StencilOperation::Replace => wgpu::StencilOperation::Replace,
        rhi::StencilOperation::Invert => wgpu::StencilOperation::Invert,
        rhi::StencilOperation::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        rhi::StencilOperation::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        rhi::StencilOperation::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        rhi::StencilOperation::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
    }
}

/// Translates a single-face stencil state.
pub(crate) fn stencil_face_state(state: rhi::StencilFaceState) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: compare_function(state.compare),
        fail_op: stencil_operation(state.fail_op),
        depth_fail_op: stencil_operation(state.depth_fail_op),
        pass_op: stencil_operation(state.pass_op),
    }
}

/// Translates full stencil state.
pub(crate) fn stencil_state(state: rhi::StencilState) -> wgpu::StencilState {
    wgpu::StencilState {
        front: stencil_face_state(state.front),
        back: stencil_face_state(state.back),
        read_mask: state.read_mask,
        write_mask: state.write_mask,
    }
}

/// Translates depth-bias (polygon offset) parameters.
pub(crate) fn depth_bias_state(state: rhi::DepthBiasState) -> wgpu::DepthBiasState {
    wgpu::DepthBiasState {
        constant: state.constant,
        slope_scale: state.slope_scale,
        clamp: state.clamp,
    }
}

/// Translates depth/stencil attachment state.
pub(crate) fn depth_stencil_state(state: rhi::DepthStencilState) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: texture_format(state.format),
        depth_write_enabled: Some(state.depth_write_enabled),
        depth_compare: Some(compare_function(state.depth_compare)),
        stencil: stencil_state(state.stencil),
        bias: depth_bias_state(state.bias),
    }
}

/// Translates multisample resolve state.
pub(crate) fn multisample_state(state: rhi::MultisampleState) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: state.count,
        mask: state.mask,
        alpha_to_coverage_enabled: state.alpha_to_coverage_enabled,
    }
}

/// Translates a clear/blend color.
pub(crate) fn color(color: rhi::Color) -> wgpu::Color {
    wgpu::Color {
        r: color.r,
        g: color.g,
        b: color.b,
        a: color.a,
    }
}

/// Translates a store operation.
pub(crate) fn store_op(op: rhi::StoreOp) -> wgpu::StoreOp {
    match op {
        rhi::StoreOp::Store => wgpu::StoreOp::Store,
        rhi::StoreOp::Discard => wgpu::StoreOp::Discard,
    }
}

/// Builds color-attachment load/store operations.
pub(crate) fn color_operations(
    load: rhi::LoadOp,
    store: rhi::StoreOp,
) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match load {
            rhi::LoadOp::Clear(c) => wgpu::LoadOp::Clear(color(c)),
            rhi::LoadOp::Load => wgpu::LoadOp::Load,
        },
        store: store_op(store),
    }
}

/// Builds depth-plane load/store operations.
pub(crate) fn depth_operations(ops: rhi::DepthOperations) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            rhi::DepthLoadOp::Clear(value) => wgpu::LoadOp::Clear(value),
            rhi::DepthLoadOp::Load => wgpu::LoadOp::Load,
        },
        store: store_op(ops.store),
    }
}

/// Builds stencil-plane load/store operations.
pub(crate) fn stencil_operations(ops: rhi::StencilOperations) -> wgpu::Operations<u32> {
    wgpu::Operations {
        load: match ops.load {
            rhi::StencilLoadOp::Clear(value) => wgpu::LoadOp::Clear(value),
            rhi::StencilLoadOp::Load => wgpu::LoadOp::Load,
        },
        store: store_op(ops.store),
    }
}

/// Maps a wgpu backend to the RHI's coarser backend enum.
pub(crate) fn backend_from_wgpu(backend: wgpu::Backend) -> rhi::Backend {
    match backend {
        wgpu::Backend::Noop => rhi::Backend::Noop,
        wgpu::Backend::Vulkan => rhi::Backend::Vulkan,
        wgpu::Backend::Metal => rhi::Backend::Metal,
        wgpu::Backend::Dx12 => rhi::Backend::Dx12,
        // The RHI has no distinct GL variant; both GL and browser WebGPU report
        // as the WebGPU-family backend.
        wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu => rhi::Backend::WebGpu,
    }
}

/// Maps the RHI's requested features onto the wgpu features to enable at device
/// creation. `INDIRECT_DRAW` is core in wgpu and carries no flag.
pub(crate) fn features_to_wgpu(features: rhi::Features) -> wgpu::Features {
    let mut out = wgpu::Features::empty();
    if features.contains(rhi::Features::MULTI_DRAW_INDIRECT) {
        out |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
    }
    if features.contains(rhi::Features::RAY_TRACING) {
        out |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    }
    if features.contains(rhi::Features::BINDLESS) {
        out |= wgpu::Features::BUFFER_BINDING_ARRAY;
    }
    if features.contains(rhi::Features::SHADER_INT64) {
        out |= wgpu::Features::SHADER_INT64;
    }
    if features.contains(rhi::Features::SHADER_FLOAT16) {
        out |= wgpu::Features::SHADER_F16;
    }
    if features.contains(rhi::Features::MESH_SHADER) {
        out |= wgpu::Features::EXPERIMENTAL_MESH_SHADER;
    }
    if features.contains(rhi::Features::TIMESTAMP_QUERY) {
        out |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if features.contains(rhi::Features::PIPELINE_STATISTICS_QUERY) {
        out |= wgpu::Features::PIPELINE_STATISTICS_QUERY;
    }
    if features.contains(rhi::Features::DEPTH_CLAMP) {
        out |= wgpu::Features::DEPTH_CLIP_CONTROL;
    }
    if features.contains(rhi::Features::TEXTURE_COMPRESSION_BC) {
        out |= wgpu::Features::TEXTURE_COMPRESSION_BC;
    }
    if features.contains(rhi::Features::DUAL_SOURCE_BLENDING) {
        out |= wgpu::Features::DUAL_SOURCE_BLENDING;
    }
    out
}

/// Maps a wgpu adapter/device feature set back onto the RHI's feature flags.
/// `INDIRECT_DRAW` is always reported because wgpu supports it unconditionally.
pub(crate) fn features_from_wgpu(features: wgpu::Features) -> rhi::Features {
    let mut out = rhi::Features::INDIRECT_DRAW;
    if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
        out |= rhi::Features::MULTI_DRAW_INDIRECT;
    }
    if features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
        out |= rhi::Features::RAY_TRACING;
    }
    if features.contains(wgpu::Features::BUFFER_BINDING_ARRAY) {
        out |= rhi::Features::BINDLESS;
    }
    if features.contains(wgpu::Features::SHADER_INT64) {
        out |= rhi::Features::SHADER_INT64;
    }
    if features.contains(wgpu::Features::SHADER_F16) {
        out |= rhi::Features::SHADER_FLOAT16;
    }
    if features.contains(wgpu::Features::EXPERIMENTAL_MESH_SHADER) {
        out |= rhi::Features::MESH_SHADER;
    }
    if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        out |= rhi::Features::TIMESTAMP_QUERY;
    }
    if features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY) {
        out |= rhi::Features::PIPELINE_STATISTICS_QUERY;
    }
    if features.contains(wgpu::Features::DEPTH_CLIP_CONTROL) {
        out |= rhi::Features::DEPTH_CLAMP;
    }
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        out |= rhi::Features::TEXTURE_COMPRESSION_BC;
    }
    if features.contains(wgpu::Features::DUAL_SOURCE_BLENDING) {
        out |= rhi::Features::DUAL_SOURCE_BLENDING;
    }
    out
}

/// Translates RHI limits into wgpu limits, starting from wgpu's defaults so
/// unmapped fields keep sensible values.
pub(crate) fn limits_to_wgpu(limits: rhi::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_texture_dimension_3d: limits.max_texture_dimension_3d,
        max_texture_array_layers: limits.max_texture_array_layers,
        max_bind_groups: limits.max_bind_groups,
        max_bindings_per_bind_group: limits.max_bindings_per_bind_group,
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_vertex_buffers: limits.max_vertex_buffers,
        max_vertex_attributes: limits.max_vertex_attributes,
        max_color_attachments: limits.max_color_attachments,
        max_immediate_size: limits.max_push_constant_size,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        ..wgpu::Limits::default()
    }
}

/// Reads a wgpu limit set back into the RHI's limit struct.
pub(crate) fn limits_from_wgpu(limits: &wgpu::Limits) -> rhi::Limits {
    rhi::Limits {
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_texture_dimension_3d: limits.max_texture_dimension_3d,
        max_texture_array_layers: limits.max_texture_array_layers,
        max_bind_groups: limits.max_bind_groups,
        max_bindings_per_bind_group: limits.max_bindings_per_bind_group,
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_vertex_buffers: limits.max_vertex_buffers,
        max_vertex_attributes: limits.max_vertex_attributes,
        max_color_attachments: limits.max_color_attachments,
        max_push_constant_size: limits.max_immediate_size,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    }
}

/// Computes the wgpu `immediate_size` from the RHI push-constant ranges: the
/// highest end offset across all ranges. Non-zero sizes require the
/// `IMMEDIATES` feature at device creation.
pub(crate) fn immediate_size(ranges: &[rhi::PushConstantRange]) -> u32 {
    ranges.iter().map(|range| range.end).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_feature_round_trips() {
        // Every individual RHI feature should survive a translation to wgpu and
        // back (INDIRECT_DRAW is always reported because wgpu supports it
        // unconditionally).
        const FEATURES: &[rhi::Features] = &[
            rhi::Features::INDIRECT_DRAW,
            rhi::Features::MULTI_DRAW_INDIRECT,
            rhi::Features::RAY_TRACING,
            rhi::Features::BINDLESS,
            rhi::Features::SHADER_INT64,
            rhi::Features::SHADER_FLOAT16,
            rhi::Features::MESH_SHADER,
            rhi::Features::TIMESTAMP_QUERY,
            rhi::Features::PIPELINE_STATISTICS_QUERY,
            rhi::Features::DEPTH_CLAMP,
            rhi::Features::TEXTURE_COMPRESSION_BC,
            rhi::Features::DUAL_SOURCE_BLENDING,
        ];
        for &bit in FEATURES {
            let wgpu_features = features_to_wgpu(bit);
            let back = features_from_wgpu(wgpu_features);
            assert!(
                back.contains(bit),
                "feature {bit:?} was lost in translation"
            );
            assert!(
                back.contains(rhi::Features::INDIRECT_DRAW),
                "INDIRECT_DRAW should always be reported"
            );
        }
    }

    #[test]
    fn full_feature_set_maps_both_ways() {
        let all = rhi::Features::all();
        let back = features_from_wgpu(features_to_wgpu(all));
        assert_eq!(back, all);
    }

    #[test]
    fn baseline_limits_round_trip() {
        let baseline = rhi::Limits::baseline();
        assert_eq!(limits_from_wgpu(&limits_to_wgpu(baseline)), baseline);
    }

    #[test]
    fn desktop_limits_round_trip() {
        let desktop = rhi::Limits::desktop();
        assert_eq!(limits_from_wgpu(&limits_to_wgpu(desktop)), desktop);
    }

    #[test]
    fn backend_mapping() {
        assert_eq!(backend_from_wgpu(wgpu::Backend::Noop), rhi::Backend::Noop);
        assert_eq!(
            backend_from_wgpu(wgpu::Backend::Vulkan),
            rhi::Backend::Vulkan
        );
        assert_eq!(backend_from_wgpu(wgpu::Backend::Metal), rhi::Backend::Metal);
        assert_eq!(backend_from_wgpu(wgpu::Backend::Dx12), rhi::Backend::Dx12);
        assert_eq!(backend_from_wgpu(wgpu::Backend::Gl), rhi::Backend::WebGpu);
        assert_eq!(
            backend_from_wgpu(wgpu::Backend::BrowserWebGpu),
            rhi::Backend::WebGpu
        );
    }

    #[test]
    fn texture_format_mapping_is_exhaustive() {
        // Translating every RHI texture format must not panic, and a few
        // representative formats must land on the expected wgpu format.
        let cases = [
            (rhi::TextureFormat::R8Unorm, wgpu::TextureFormat::R8Unorm),
            (
                rhi::TextureFormat::Rgba8UnormSrgb,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ),
            (
                rhi::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Bgra8Unorm,
            ),
            (
                rhi::TextureFormat::Rgba16Float,
                wgpu::TextureFormat::Rgba16Float,
            ),
            (
                rhi::TextureFormat::Depth32Float,
                wgpu::TextureFormat::Depth32Float,
            ),
            (
                rhi::TextureFormat::Depth24PlusStencil8,
                wgpu::TextureFormat::Depth24PlusStencil8,
            ),
            // The RHI's packed 11/11/10 float is wgpu's `Rg11b10Ufloat`.
            (
                rhi::TextureFormat::Rg11b10Float,
                wgpu::TextureFormat::Rg11b10Ufloat,
            ),
        ];
        for (rhi_format, wgpu_format) in cases {
            assert_eq!(texture_format(rhi_format), wgpu_format);
        }
    }

    #[test]
    fn vertex_format_maps_exhaustively() {
        // Every RHI vertex format must map to the matching wgpu vertex format.
        // Byte sizes are intentionally *not* compared: the frozen RHI defines
        // its own `VertexFormat::size()` table (e.g. `Float16x2` is reported as
        // 8 bytes), which drives RHI-side offset/stride math and need not equal
        // wgpu's physical element size. The backend passes RHI-supplied offsets
        // and strides through verbatim, so only the format mapping matters.
        use rhi::VertexFormat as F;
        use wgpu::VertexFormat as W;
        let cases = [
            (F::Float32, W::Float32),
            (F::Float32x2, W::Float32x2),
            (F::Float32x3, W::Float32x3),
            (F::Float32x4, W::Float32x4),
            (F::Float16x2, W::Float16x2),
            (F::Float16x4, W::Float16x4),
            (F::Uint32, W::Uint32),
            (F::Uint32x2, W::Uint32x2),
            (F::Uint32x4, W::Uint32x4),
            (F::Sint32, W::Sint32),
            (F::Unorm8x4, W::Unorm8x4),
            (F::Snorm8x4, W::Snorm8x4),
        ];
        for (rhi_format, wgpu_format) in cases {
            assert_eq!(vertex_format(rhi_format), wgpu_format);
        }
    }

    #[test]
    fn buffer_usages_or_together() {
        let usage = rhi::BufferUsages::VERTEX | rhi::BufferUsages::COPY_DST;
        let translated = buffer_usages(usage);
        assert!(translated.contains(wgpu::BufferUsages::VERTEX));
        assert!(translated.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!translated.contains(wgpu::BufferUsages::INDEX));
    }

    #[test]
    fn texture_usages_or_together() {
        let usage = rhi::TextureUsages::TEXTURE_BINDING | rhi::TextureUsages::RENDER_ATTACHMENT;
        let translated = texture_usages(usage);
        assert!(translated.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(translated.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(!translated.contains(wgpu::TextureUsages::STORAGE_BINDING));
    }

    #[test]
    fn shader_stages_or_together() {
        let stages = rhi::ShaderStages::VERTEX | rhi::ShaderStages::FRAGMENT;
        let translated = shader_stages(stages);
        assert!(translated.contains(wgpu::ShaderStages::VERTEX));
        assert!(translated.contains(wgpu::ShaderStages::FRAGMENT));
        assert!(!translated.contains(wgpu::ShaderStages::COMPUTE));
    }

    #[test]
    fn immediate_size_takes_max_end() {
        let ranges = [
            rhi::PushConstantRange {
                stages: rhi::ShaderStages::VERTEX,
                start: 0,
                end: 16,
            },
            rhi::PushConstantRange {
                stages: rhi::ShaderStages::FRAGMENT,
                start: 16,
                end: 48,
            },
        ];
        assert_eq!(immediate_size(&ranges), 48);
        assert_eq!(immediate_size(&[]), 0);
    }
}
