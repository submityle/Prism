use super::*;
use crate::pipeline_cache::{PipelineCache, PipelineCompiler, PipelineKey, PipelineResolution};
use prism_render_architecture::shader_package::{ExecutionFamily, ModuleId, PermutationIndex};
use prism_render_driver::{Instance, RenderDevice};
use prism_render_driver_null::{NullDevice, NullInstance};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

fn descriptor() -> rhi::RenderPipelineDescriptor {
    let module = rhi::ShaderModuleId::from_parts(0, 0);
    rhi::RenderPipelineDescriptor {
        label: Some("pipeline-state-fixture".into()),
        layout: None,
        vertex: rhi::VertexState {
            module,
            entry_point: "vs".into(),
            buffers: vec![rhi::VertexBufferLayout::packed(
                rhi::VertexStepMode::Vertex,
                0,
                &[rhi::VertexFormat::Float32x3],
            )],
        },
        primitive: rhi::PrimitiveState {
            topology: rhi::PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(rhi::IndexFormat::Uint16),
            ..rhi::PrimitiveState::default()
        },
        depth_stencil: Some(rhi::DepthStencilState::reverse_z(
            rhi::TextureFormat::Depth24PlusStencil8,
        )),
        multisample: rhi::MultisampleState::default(),
        fragment: Some(rhi::FragmentState {
            module,
            entry_point: "fs".into(),
            targets: vec![
                Some(rhi::ColorTargetState {
                    format: rhi::TextureFormat::Rgba8Unorm,
                    blend: Some(rhi::BlendState::ALPHA_BLENDING),
                    write_mask: rhi::ColorWrites::all(),
                }),
                None,
                Some(rhi::ColorTargetState::opaque(
                    rhi::TextureFormat::Rgba16Float,
                )),
            ],
        }),
    }
}

/// One mutation for every RHI fixed-function leaf and each optional/container
/// boundary. Exhaustive state encoders also fail to compile on an unhandled new
/// named field, so this list cannot mask newly added descriptor/state fields.
fn mutations() -> Vec<(&'static str, rhi::RenderPipelineDescriptor)> {
    let mut cases = Vec::new();
    macro_rules! change {
        ($name:literal, $d:ident, $change:expr) => {{
            let mut $d = descriptor();
            $change;
            cases.push(($name, $d));
        }};
    }
    change!(
        "vertex buffer count",
        d,
        d.vertex.buffers.push(d.vertex.buffers[0].clone())
    );
    change!("vertex stride", d, d.vertex.buffers[0].array_stride += 4);
    change!(
        "vertex step mode",
        d,
        d.vertex.buffers[0].step_mode = rhi::VertexStepMode::Instance
    );
    change!("attribute count", d, {
        let attribute = d.vertex.buffers[0].attributes[0];
        d.vertex.buffers[0].attributes.push(attribute);
    });
    change!(
        "attribute format",
        d,
        d.vertex.buffers[0].attributes[0].format = rhi::VertexFormat::Uint32
    );
    change!(
        "attribute offset",
        d,
        d.vertex.buffers[0].attributes[0].offset = 4
    );
    change!(
        "attribute location",
        d,
        d.vertex.buffers[0].attributes[0].shader_location = 1
    );
    change!(
        "topology",
        d,
        d.primitive.topology = rhi::PrimitiveTopology::LineStrip
    );
    change!(
        "strip index format",
        d,
        d.primitive.strip_index_format = Some(rhi::IndexFormat::Uint32)
    );
    change!(
        "strip index absent",
        d,
        d.primitive.strip_index_format = None
    );
    change!("front face", d, d.primitive.front_face = rhi::FrontFace::Cw);
    change!(
        "cull face",
        d,
        d.primitive.cull_mode = Some(rhi::Face::Front)
    );
    change!("cull absent", d, d.primitive.cull_mode = None);
    change!(
        "polygon mode",
        d,
        d.primitive.polygon_mode = rhi::PolygonMode::Line
    );
    change!("unclipped depth", d, d.primitive.unclipped_depth = true);
    change!("depth absent", d, d.depth_stencil = None);
    change!(
        "depth format",
        d,
        d.depth_stencil.as_mut().unwrap().format = rhi::TextureFormat::Depth32Float
    );
    change!(
        "depth write",
        d,
        d.depth_stencil.as_mut().unwrap().depth_write_enabled = false
    );
    change!(
        "depth compare",
        d,
        d.depth_stencil.as_mut().unwrap().depth_compare = rhi::CompareFunction::Less
    );
    change!(
        "stencil front compare",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.front.compare = rhi::CompareFunction::Equal
    );
    change!(
        "stencil front fail",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.front.fail_op = rhi::StencilOperation::Replace
    );
    change!(
        "stencil front depth fail",
        d,
        d.depth_stencil
            .as_mut()
            .unwrap()
            .stencil
            .front
            .depth_fail_op = rhi::StencilOperation::Replace
    );
    change!(
        "stencil front pass",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.front.pass_op = rhi::StencilOperation::Replace
    );
    change!(
        "stencil back compare",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.back.compare = rhi::CompareFunction::Equal
    );
    change!(
        "stencil back fail",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.back.fail_op = rhi::StencilOperation::Replace
    );
    change!(
        "stencil back depth fail",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.back.depth_fail_op =
            rhi::StencilOperation::Replace
    );
    change!(
        "stencil back pass",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.back.pass_op = rhi::StencilOperation::Replace
    );
    change!(
        "stencil read mask",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.read_mask = 0xff
    );
    change!(
        "stencil write mask",
        d,
        d.depth_stencil.as_mut().unwrap().stencil.write_mask = 0xff
    );
    change!(
        "bias constant",
        d,
        d.depth_stencil.as_mut().unwrap().bias.constant = -1
    );
    change!(
        "bias slope",
        d,
        d.depth_stencil.as_mut().unwrap().bias.slope_scale = 1.25
    );
    change!(
        "bias clamp",
        d,
        d.depth_stencil.as_mut().unwrap().bias.clamp = 0.5
    );
    change!("msaa count", d, d.multisample.count = 4);
    change!("msaa mask", d, d.multisample.mask ^= 1);
    change!("msaa mask high bit", d, d.multisample.mask ^= 1 << 63);
    change!(
        "alpha coverage",
        d,
        d.multisample.alpha_to_coverage_enabled = true
    );
    change!("fragment absent", d, d.fragment = None);
    change!(
        "color target count",
        d,
        d.fragment.as_mut().unwrap().targets.push(None)
    );
    change!(
        "color target empty",
        d,
        d.fragment.as_mut().unwrap().targets.clear()
    );
    change!(
        "disabled color slot",
        d,
        d.fragment.as_mut().unwrap().targets[0] = None
    );
    change!(
        "enable color slot",
        d,
        d.fragment.as_mut().unwrap().targets[1] = Some(rhi::ColorTargetState::opaque(
            rhi::TextureFormat::Rgba8Unorm
        ))
    );
    change!(
        "color target order",
        d,
        d.fragment.as_mut().unwrap().targets.swap(0, 2)
    );
    change!(
        "color format",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .format = rhi::TextureFormat::Bgra8Unorm
    );
    change!(
        "color write mask",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .write_mask = rhi::ColorWrites::RED
    );
    change!(
        "blend absent",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend = None
    );
    change!(
        "blend color source",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend
            .as_mut()
            .unwrap()
            .color
            .src_factor = rhi::BlendFactor::Constant
    );
    change!(
        "blend color destination",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend
            .as_mut()
            .unwrap()
            .color
            .dst_factor = rhi::BlendFactor::Constant
    );
    change!(
        "blend color operation",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend
            .as_mut()
            .unwrap()
            .color
            .operation = rhi::BlendOperation::Max
    );
    change!(
        "blend alpha source",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend
            .as_mut()
            .unwrap()
            .alpha
            .src_factor = rhi::BlendFactor::Constant
    );
    change!(
        "blend alpha destination",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend
            .as_mut()
            .unwrap()
            .alpha
            .dst_factor = rhi::BlendFactor::Constant
    );
    change!(
        "blend alpha operation",
        d,
        d.fragment.as_mut().unwrap().targets[0]
            .as_mut()
            .unwrap()
            .blend
            .as_mut()
            .unwrap()
            .alpha
            .operation = rhi::BlendOperation::Max
    );
    change!(
        "second color target format",
        d,
        d.fragment.as_mut().unwrap().targets[2]
            .as_mut()
            .unwrap()
            .format = rhi::TextureFormat::Rgba32Float
    );
    cases
}

#[test]
fn canonical_state_bytes_match_hash_and_cover_every_fixed_function_field() {
    fn hash_bytes(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
    let baseline = PipelineState::from_descriptor(&descriptor());
    let bytes = baseline.canonical_bytes();
    assert_eq!(&bytes[..6], b"PSTATE");
    assert_eq!(&bytes[6..14], &1u64.to_le_bytes());
    assert_eq!(hash_bytes(&bytes), baseline.to_hash().0);
    assert_eq!(bytes, baseline.clone().canonical_bytes());
    for (name, descriptor) in mutations() {
        let state = PipelineState::from_descriptor(&descriptor);
        let changed = state.canonical_bytes();
        assert_ne!(changed, bytes, "canonical encoding omitted {name}");
        assert_eq!(hash_bytes(&changed), state.to_hash().0, "{name}");
    }
}

#[test]
fn state_struct_fields_are_all_hashed() {
    let baseline = PipelineState::from_descriptor(&descriptor());
    let baseline_key = PipelineKey::new(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        baseline.clone(),
    );
    let mut hashes = BTreeSet::from([baseline.to_hash()]);
    for (name, descriptor) in mutations() {
        let state = PipelineState::from_descriptor(&descriptor);
        assert_ne!(state, baseline, "state forgot {name}");
        assert_ne!(
            PipelineKey::new(
                ExecutionFamily::FusedBuiltin,
                ModuleId(42),
                PermutationIndex::ZERO,
                state.clone()
            ),
            baseline_key,
            "hot key forgot {name}"
        );
        assert_ne!(state.to_hash(), baseline.to_hash(), "hash forgot {name}");
        assert!(
            hashes.insert(state.to_hash()),
            "fixture hash collision for {name}"
        );
    }
}

#[test]
fn stable_pipeline_state_hash_golden_and_float_bits() {
    let state = PipelineState::from_descriptor(&descriptor());
    // Frozen v1 value is calculated independently from the documented wire
    // words, not from derived Rust Hash or enum discriminants.
    assert_eq!(state.to_hash().0, 0xc5dc_594b_5a5a_8cb1);
    assert_eq!(state.to_hash(), state.clone().to_hash());
    let mut negative_zero = descriptor();
    negative_zero
        .depth_stencil
        .as_mut()
        .unwrap()
        .bias
        .slope_scale = -0.0;
    let negative_zero = PipelineState::from_descriptor(&negative_zero);
    assert_ne!(negative_zero, state);
    assert_ne!(negative_zero.to_hash(), state.to_hash());
    let mut nan = descriptor();
    nan.depth_stencil.as_mut().unwrap().bias.clamp = f32::from_bits(0x7fc0_0001);
    let nan_state = PipelineState::from_descriptor(&nan);
    assert_eq!(nan_state, nan_state.clone());
    assert_eq!(
        nan_state.to_hash(),
        PipelineState::from_descriptor(&nan).to_hash()
    );
    nan.depth_stencil.as_mut().unwrap().bias.clamp = f32::from_bits(0x7fc0_0002);
    assert_ne!(
        nan_state.to_hash(),
        PipelineState::from_descriptor(&nan).to_hash()
    );
}

#[test]
fn pipeline_state_preserves_layout_order_and_optional_stage_boundaries() {
    let mut original = descriptor();
    original.vertex.buffers = vec![
        rhi::VertexBufferLayout::packed(
            rhi::VertexStepMode::Vertex,
            0,
            &[rhi::VertexFormat::Float32x3, rhi::VertexFormat::Float32x2],
        ),
        rhi::VertexBufferLayout::packed(
            rhi::VertexStepMode::Instance,
            2,
            &[rhi::VertexFormat::Float32x4],
        ),
    ];
    let original_key = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &original,
    );
    let mut changed = original.clone();
    changed.vertex.buffers.swap(0, 1);
    let buffer_order = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &changed,
    );
    changed = original.clone();
    changed.vertex.buffers[0].attributes.swap(0, 1);
    let attribute_order = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &changed,
    );
    for key in [buffer_order, attribute_order] {
        assert_ne!(original_key, key);
        assert_ne!(
            original_key.typed_pipeline_state.to_hash(),
            key.typed_pipeline_state.to_hash()
        );
    }
    let mut empty_targets = descriptor();
    empty_targets.fragment.as_mut().unwrap().targets.clear();
    let mut no_fragment = empty_targets.clone();
    no_fragment.fragment = None;
    let empty = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &empty_targets,
    );
    let absent = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &no_fragment,
    );
    assert_ne!(empty, absent);
    assert_ne!(
        empty.typed_pipeline_state.to_hash(),
        absent.typed_pipeline_state.to_hash()
    );
}

#[test]
fn descriptor_runtime_handles_and_labels_are_not_fixed_function_state() {
    let original = descriptor();
    let mut changed = original.clone();
    changed.label = Some("different label".into());
    changed.layout = Some(rhi::PipelineLayoutId::from_parts(3, 4));
    changed.vertex.module = rhi::ShaderModuleId::from_parts(5, 6);
    changed.vertex.entry_point = "other_vs".into();
    changed.fragment.as_mut().unwrap().module = rhi::ShaderModuleId::from_parts(7, 8);
    changed.fragment.as_mut().unwrap().entry_point = "other_fs".into();
    let state = PipelineState::from_descriptor(&original);
    let changed_state = PipelineState::from_descriptor(&changed);
    assert_eq!(state, changed_state);
    assert_eq!(state.to_hash(), changed_state.to_hash());
    assert_ne!(
        PipelineKey::new(
            ExecutionFamily::FusedBuiltin,
            ModuleId(11),
            PermutationIndex::ZERO,
            state
        ),
        PipelineKey::new(
            ExecutionFamily::FusedBuiltin,
            ModuleId(12),
            PermutationIndex::ZERO,
            changed_state
        )
    );
}

fn device() -> NullDevice {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&rhi::RequestAdapterOptions::default())
        .unwrap();
    instance
        .create_device(&adapter, &rhi::DeviceDescriptor::default())
        .unwrap()
        .0
}

#[test]
fn pipeline_cache_uses_complete_descriptor_state_for_lookup() {
    let device = device();
    let mut cache = PipelineCache::new();
    let descriptor = descriptor();
    let key = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &descriptor,
    );
    let first = cache.get_or_create(&device, key.clone(), &descriptor);
    assert_eq!(
        first,
        cache.get_or_create(&device, key.clone(), &descriptor)
    );
    let mut ids = BTreeSet::from([first]);
    for (name, descriptor) in mutations() {
        let other_key = PipelineKey::from_descriptor(
            ExecutionFamily::FusedBuiltin,
            ModuleId(42),
            PermutationIndex::ZERO,
            &descriptor,
        );
        let id = cache.get_or_create(&device, other_key.clone(), &descriptor);
        assert!(ids.insert(id), "aliased fixed-function override {name}");
        assert!(
            cache.release(&device, &other_key),
            "full key release {name}"
        );
    }
    assert_eq!(cache.len(), 1);
    assert!(!cache.release(&device, &key));
    assert!(cache.release(&device, &key));
    assert!(cache.is_empty());
}

#[test]
fn pipeline_key_is_used_for_lookup_and_not_for_writeback() {
    let device = device();
    let descriptor = descriptor();
    let key = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &descriptor,
    );
    let unchanged = key.clone();
    let mut cache = PipelineCache::new();
    cache.get_or_create(&device, key.clone(), &descriptor);
    assert_eq!(key, unchanged);
    let mut mismatched = descriptor.clone();
    mismatched.depth_stencil.as_mut().unwrap().bias.constant = 1;
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.get_or_create(&device, key.clone(), &mismatched);
    }));
    assert!(error.is_err(), "cache hit must reject a stale state key");
    assert_eq!(key, unchanged);
    assert_eq!(cache.len(), 1);
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.resolve_or_request(
            &device,
            key.clone(),
            descriptor.clone(),
            key.clone(),
            mismatched,
        );
    }));
    assert!(
        error.is_err(),
        "fallback key must also agree before cache lookup"
    );
    assert_eq!(key, unchanged);
}

#[test]
fn pipeline_cache_rejects_stale_state_before_cold_requests_and_prewarm() {
    let device = Arc::new(device());
    let mut cache = PipelineCache::with_worker(device.clone());
    let descriptor = descriptor();
    let key = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(42),
        PermutationIndex::ZERO,
        &descriptor,
    );
    let mut mismatched = descriptor.clone();
    mismatched.multisample.alpha_to_coverage_enabled = true;
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.get_or_create(device.as_ref(), key.clone(), &mismatched);
    }));
    assert!(error.is_err());
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.resolve_or_request(
            device.as_ref(),
            key.clone(),
            mismatched.clone(),
            key.clone(),
            descriptor.clone(),
        );
    }));
    assert!(error.is_err());
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.prewarm(
            device.as_ref(),
            [(key.clone(), descriptor.clone(), key, mismatched)],
        );
    }));
    assert!(error.is_err());
    assert!(cache.is_empty());
    assert_eq!(cache.pending_len(), 0);
    assert_eq!(cache.waiting_len(), 0);
}

struct RecordingCompiler {
    device: Arc<NullDevice>,
    states: Mutex<Vec<PipelineKey>>,
}
impl PipelineCompiler for RecordingCompiler {
    fn compile(
        &self,
        key: PipelineKey,
        descriptor: &rhi::RenderPipelineDescriptor,
    ) -> rhi::RenderPipelineId {
        assert_eq!(
            key.typed_pipeline_state,
            PipelineState::from_descriptor(descriptor)
        );
        self.states.lock().unwrap().push(key);
        self.device.create_render_pipeline(descriptor)
    }
}

#[test]
fn async_pipeline_cache_tracks_fixed_state_in_pending_and_fallback() {
    let device = Arc::new(device());
    let compiler = Arc::new(RecordingCompiler {
        device: device.clone(),
        states: Mutex::new(Vec::new()),
    });
    let mut cache = PipelineCache::with_compiler(compiler.clone());
    let fallback_descriptor = descriptor();
    let fallback_key = PipelineKey::from_descriptor(
        ExecutionFamily::FusedBuiltin,
        ModuleId(100),
        PermutationIndex::ZERO,
        &fallback_descriptor,
    );
    let mut requests = Vec::new();
    for bias in [1, 2] {
        let mut descriptor = fallback_descriptor.clone();
        descriptor.depth_stencil.as_mut().unwrap().bias.constant = bias;
        let key = PipelineKey::from_descriptor(
            ExecutionFamily::FusedBuiltin,
            ModuleId(42),
            PermutationIndex::ZERO,
            &descriptor,
        );
        assert!(matches!(
            cache.resolve_or_request(
                device.as_ref(),
                key.clone(),
                descriptor.clone(),
                fallback_key.clone(),
                fallback_descriptor.clone(),
            ),
            PipelineResolution::Fallback(_)
        ));
        requests.push((key, descriptor));
    }
    assert_eq!(
        cache.pending_len(),
        2,
        "same class/shader with different bias must not merge jobs"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while cache.pending_len() != 0 {
        cache.pump(device.as_ref());
        assert!(
            std::time::Instant::now() < deadline,
            "background compile timed out"
        );
        std::thread::yield_now();
    }
    assert_eq!(compiler.states.lock().unwrap().len(), 2);
    let mut ready = BTreeSet::new();
    for (key, descriptor) in requests {
        let unchanged = key.clone();
        let result = cache.resolve_or_request(
            device.as_ref(),
            key.clone(),
            descriptor,
            fallback_key.clone(),
            fallback_descriptor.clone(),
        );
        assert!(result.is_ready());
        assert!(ready.insert(result.id()));
        assert_eq!(key, unchanged);
    }
    assert_eq!(
        cache.len(),
        3,
        "two specialized states plus resident fallback"
    );
    cache.clear(device.as_ref());
}
