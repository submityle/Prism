//! Unit tests for the RHI type system invariants.

use crate::prelude::*;
use crate::{
    Backend, BindGroupLayoutDescriptor, BindGroupLayoutEntry, DepthStencilState, ResourceId,
    SamplerBorderColor, ShaderModuleId, ShaderStages, TextureDimension, VertexStepMode,
};
use alloc::vec;

#[test]
fn flag_set_algebra() {
    let rw = BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST);
    assert!(rw.contains(BufferUsages::COPY_SRC));
    assert!(rw.contains(BufferUsages::COPY_DST));
    assert!(!rw.contains(BufferUsages::VERTEX));
    assert_eq!(
        rw.intersection(BufferUsages::COPY_SRC),
        BufferUsages::COPY_SRC
    );
    assert_eq!(
        rw.difference(BufferUsages::COPY_SRC),
        BufferUsages::COPY_DST
    );
    assert!(rw.intersects(BufferUsages::COPY_DST));

    let mut m = BufferUsages::NONE;
    assert!(m.is_empty());
    m.insert(BufferUsages::INDEX);
    assert!(m.contains(BufferUsages::INDEX));
    m.toggle(BufferUsages::INDEX);
    assert!(m.is_empty());
    m.insert(BufferUsages::VERTEX);
    m.remove(BufferUsages::VERTEX);
    assert!(m.is_empty());

    // Operator forms agree with method forms.
    assert_eq!(
        BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST)
    );
    assert_eq!(
        BufferUsages::COPY_SRC & rw,
        BufferUsages::COPY_SRC.intersection(rw)
    );
}

#[test]
fn flag_debug_lists_named_flags() {
    extern crate std;
    use std::format;
    let s = format!("{:?}", ShaderStages::vertex_fragment());
    assert!(s.contains("VERTEX"));
    assert!(s.contains("FRAGMENT"));
    assert_eq!(format!("{:?}", ShaderStages::NONE), "ShaderStages(NONE)");
}

#[test]
fn color_writes_color_excludes_alpha() {
    let c = ColorWrites::color();
    assert!(c.contains(ColorWrites::RED));
    assert!(c.contains(ColorWrites::GREEN));
    assert!(c.contains(ColorWrites::BLUE));
    assert!(!c.contains(ColorWrites::ALPHA));
    assert_eq!(ColorWrites::all().bits(), 0b1111);
}

#[test]
fn texture_format_byte_sizes() {
    assert_eq!(TextureFormat::R8Unorm.bytes_per_texel(), 1);
    assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_texel(), 4);
    assert_eq!(TextureFormat::Rgba16Float.bytes_per_texel(), 8);
    assert_eq!(TextureFormat::Rgba32Float.bytes_per_texel(), 16);
    assert!(TextureFormat::Rgba8UnormSrgb.is_srgb());
    assert!(!TextureFormat::Rgba8Unorm.is_srgb());
    assert!(TextureFormat::Depth24PlusStencil8.has_depth());
    assert!(TextureFormat::Depth24PlusStencil8.has_stencil());
    assert!(TextureFormat::Depth32Float.has_depth());
    assert!(!TextureFormat::Depth32Float.has_stencil());
}

#[test]
fn vertex_format_size_and_components() {
    assert_eq!(VertexFormat::Float32x3.size(), 12);
    assert_eq!(VertexFormat::Float32x3.components(), 3);
    assert_eq!(VertexFormat::Float32x4.size(), 16);
    assert_eq!(VertexFormat::Float32x2.size(), 8);
}

#[test]
fn extent_mip_math() {
    let e = Extent3d::new_2d(256, 128);
    assert_eq!(e.depth_or_array_layers, 1);
    assert_eq!(e.max_mip_levels(TextureDimension::D2), 9); // log2(256) + 1
    let m1 = e.mip_level_size(1, TextureDimension::D2);
    assert_eq!((m1.width, m1.height), (128, 64));
    let big = e.mip_level_size(100, TextureDimension::D2);
    assert_eq!((big.width, big.height), (1, 1));
}

#[test]
fn vertex_layout_packed_offsets() {
    let layout = VertexBufferLayout::packed(
        VertexStepMode::Vertex,
        0,
        &[VertexFormat::Float32x3, VertexFormat::Float32x2],
    );
    assert_eq!(layout.array_stride, 12 + 8);
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[0].shader_location, 0);
    assert_eq!(layout.attributes[1].offset, 12);
    assert_eq!(layout.attributes[1].shader_location, 1);
    assert!(layout.is_within_stride());

    let mut bad = layout;
    bad.array_stride = 10;
    assert!(!bad.is_within_stride());
}

#[test]
fn blend_presets() {
    assert_eq!(BlendState::REPLACE, BlendState::REPLACE);
    assert!(!BlendState::REPLACE.uses_constant());
    // Distinct presets differ.
    assert_ne!(BlendState::ALPHA_BLENDING, BlendState::REPLACE);
    assert_ne!(BlendState::ADDITIVE, BlendState::PREMULTIPLIED_ALPHA);
}

#[test]
fn primitive_state_strip_validity() {
    let list = PrimitiveState::default();
    assert!(list.is_valid()); // triangle list, no strip index

    let mut strip = PrimitiveState {
        topology: PrimitiveTopology::TriangleStrip,
        ..PrimitiveState::default()
    };
    assert!(!strip.is_valid()); // strip needs an index format
    strip.strip_index_format = Some(IndexFormat::Uint32);
    assert!(strip.is_valid());
}

#[test]
fn stencil_state_enabled() {
    let depth = DepthStencilState::reverse_z(TextureFormat::Depth32Float);
    assert!(!depth.stencil.is_enabled());
}

#[test]
fn limits_satisfies_and_tiers() {
    let base = Limits::baseline();
    let desk = Limits::desktop();
    assert!(desk.satisfies(&base));
    assert!(!base.satisfies(&desk));
    assert!(base.satisfies(&base));
    assert_eq!(Limits::default(), Limits::baseline());
}

#[test]
fn device_capabilities_supports() {
    let caps = DeviceCapabilities {
        backend: Backend::Vulkan,
        features: Features::INDIRECT_DRAW.union(Features::RAY_TRACING),
        limits: Limits::desktop(),
    };
    assert!(caps.supports(Features::INDIRECT_DRAW, &Limits::baseline()));
    assert!(!caps.supports(Features::MESH_SHADER, &Limits::baseline()));
    assert!(!caps.supports(Features::INDIRECT_DRAW, &{
        let mut l = Limits::desktop();
        l.max_texture_dimension_2d += 1;
        l
    }));
}

#[test]
fn resource_id_generational_type_safety() {
    let a = BufferId::from_parts(3, 1);
    let b = BufferId::from_parts(3, 2);
    assert_ne!(a, b); // same slot, different generation
    assert!(a < b);
    assert_eq!(a, BufferId::from_parts(3, 1));
    assert_eq!(a.index(), 3);
    assert_eq!(b.generation(), 2);

    // Different kinds are distinct types sharing the generational layout.
    let t: ResourceId<_> = TextureId::from_parts(3, 1);
    assert_eq!(t.index(), 3);
}

#[test]
fn bind_group_layout_unique_bindings() {
    let good = BindGroupLayoutDescriptor {
        label: None,
        entries: vec![
            BindGroupLayoutEntry::uniform(0, ShaderStages::VERTEX),
            BindGroupLayoutEntry::texture_2d(1, ShaderStages::FRAGMENT),
            BindGroupLayoutEntry::sampler(2, ShaderStages::FRAGMENT),
        ],
    };
    assert!(good.has_unique_bindings());

    let dup = BindGroupLayoutDescriptor {
        label: None,
        entries: vec![
            BindGroupLayoutEntry::uniform(0, ShaderStages::VERTEX),
            BindGroupLayoutEntry::uniform(0, ShaderStages::FRAGMENT),
        ],
    };
    assert!(!dup.has_unique_bindings());
}

#[test]
fn color_target_presets() {
    let opaque = ColorTargetState::opaque(TextureFormat::Rgba8Unorm);
    assert!(opaque.blend.is_none());
    assert_eq!(opaque.write_mask, ColorWrites::all());
    let blended = ColorTargetState::alpha_blended(TextureFormat::Rgba8Unorm);
    assert_eq!(blended.blend, Some(BlendState::ALPHA_BLENDING));
}

#[test]
fn render_pipeline_target_count_and_validity() {
    let frag = FragmentState {
        module: ShaderModuleId::from_parts(0, 1),
        entry_point: "fs".into(),
        targets: vec![
            Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm)),
            None,
            Some(ColorTargetState::opaque(TextureFormat::Rgba16Float)),
        ],
    };
    let desc = RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: VertexState {
            module: ShaderModuleId::from_parts(0, 1),
            entry_point: "vs".into(),
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(frag),
    };
    assert_eq!(desc.color_target_count(), 2);
    assert!(desc.is_valid());
}

#[test]
fn sampler_comparison_detection() {
    let s = SamplerDescriptor::linear_repeat();
    assert!(!s.is_comparison());
    let mut shadow = SamplerDescriptor::linear_repeat();
    shadow.compare = Some(CompareFunction::LessEqual);
    assert!(shadow.is_comparison());
    let _ = SamplerBorderColor::OpaqueBlack;
}

#[test]
fn command_encoder_records_passes() {
    let mut enc = CommandEncoder::labeled("frame");
    enc.push_render_pass(
        RenderPassDescriptor::default(),
        vec![RenderCommand::Draw {
            vertices: 0..3,
            instances: 0..1,
        }],
    );
    enc.push_compute_pass(None, vec![]);
    assert_eq!(enc.pass_count(), 2);
    let cb = enc.finish();
    assert_eq!(cb.label.as_deref(), Some("frame"));
    assert_eq!(cb.passes.len(), 2);
}
