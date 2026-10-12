//! Unit tests for the render graph: compilation and execution invariants.
//!
//! Every test drives the real compiler and (where relevant) a mock
//! [`RenderDevice`]/[`RenderQueue`] pair, so the full pipeline — culling, SSA
//! scheduling, lifetime/alias analysis, barrier planning, attachment
//! derivation, and command recording — is exercised with no real GPU.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use prism_render_driver::{
    Backend, BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferDescriptor, BufferId, BufferState, BufferUsages, Color, CommandBuffer,
    ComputePipelineDescriptor, ComputePipelineId, CopyCommand, DeviceCapabilities, Extent3d,
    Features, ImageDataLayout, Limits, LoadOp, Pass, PipelineLayoutDescriptor, PipelineLayoutId,
    RenderDevice, RenderPipelineDescriptor, RenderPipelineId, RenderQueue, SamplerDescriptor,
    SamplerId, ShaderModuleDescriptor, ShaderModuleId, StoreOp, TextureDescriptor, TextureFormat,
    TextureId, TextureState, TextureUsages, TextureViewDescriptor, TextureViewId, TextureWrite,
};

use crate::desc::{BufferDesc, TextureDesc};
use crate::graph::RenderGraph;
use crate::handle::TextureHandle;
use crate::pass::PassFlags;
use crate::resource::{ImportedBuffer, ImportedTexture};

// --- mock backend ---------------------------------------------------------

/// A record-only [`RenderDevice`]: mints fresh generational ids and captures
/// every descriptor it is asked to create, performing no real GPU work.
struct MockDevice {
    caps: DeviceCapabilities,
    next: Cell<u32>,
    textures: RefCell<Vec<TextureDescriptor>>,
    buffers: RefCell<Vec<BufferDescriptor>>,
}

impl MockDevice {
    fn new() -> Self {
        Self {
            caps: DeviceCapabilities {
                backend: Backend::Noop,
                features: Features::NONE,
                limits: Limits::default(),
            },
            next: Cell::new(1),
            textures: RefCell::new(Vec::new()),
            buffers: RefCell::new(Vec::new()),
        }
    }

    fn mint(&self) -> u32 {
        let n = self.next.get();
        self.next.set(n + 1);
        n
    }
}

impl RenderDevice for MockDevice {
    fn capabilities(&self) -> &DeviceCapabilities {
        &self.caps
    }

    fn create_buffer(&self, descriptor: &BufferDescriptor) -> BufferId {
        self.buffers.borrow_mut().push(descriptor.clone());
        BufferId::from_parts(self.mint(), 0)
    }
    fn create_texture(&self, descriptor: &TextureDescriptor) -> TextureId {
        self.textures.borrow_mut().push(descriptor.clone());
        TextureId::from_parts(self.mint(), 0)
    }
    fn create_texture_view(
        &self,
        _texture: TextureId,
        _descriptor: &TextureViewDescriptor,
    ) -> TextureViewId {
        TextureViewId::from_parts(self.mint(), 0)
    }
    fn create_sampler(&self, _descriptor: &SamplerDescriptor) -> SamplerId {
        SamplerId::from_parts(self.mint(), 0)
    }
    fn create_shader_module(&self, _descriptor: &ShaderModuleDescriptor) -> ShaderModuleId {
        ShaderModuleId::from_parts(self.mint(), 0)
    }
    fn create_bind_group_layout(
        &self,
        _descriptor: &BindGroupLayoutDescriptor,
    ) -> BindGroupLayoutId {
        BindGroupLayoutId::from_parts(self.mint(), 0)
    }
    fn create_bind_group(&self, _descriptor: &BindGroupDescriptor) -> BindGroupId {
        BindGroupId::from_parts(self.mint(), 0)
    }
    fn create_pipeline_layout(&self, _descriptor: &PipelineLayoutDescriptor) -> PipelineLayoutId {
        PipelineLayoutId::from_parts(self.mint(), 0)
    }
    fn create_render_pipeline(&self, _descriptor: &RenderPipelineDescriptor) -> RenderPipelineId {
        RenderPipelineId::from_parts(self.mint(), 0)
    }
    fn create_compute_pipeline(
        &self,
        _descriptor: &ComputePipelineDescriptor,
    ) -> ComputePipelineId {
        ComputePipelineId::from_parts(self.mint(), 0)
    }

    fn destroy_buffer(&self, _id: BufferId) {}
    fn destroy_texture(&self, _id: TextureId) {}
    fn destroy_texture_view(&self, _id: TextureViewId) {}
    fn destroy_sampler(&self, _id: SamplerId) {}
    fn destroy_shader_module(&self, _id: ShaderModuleId) {}
    fn destroy_bind_group_layout(&self, _id: BindGroupLayoutId) {}
    fn destroy_bind_group(&self, _id: BindGroupId) {}
    fn destroy_pipeline_layout(&self, _id: PipelineLayoutId) {}
    fn destroy_render_pipeline(&self, _id: RenderPipelineId) {}
    fn destroy_compute_pipeline(&self, _id: ComputePipelineId) {}
}

/// A record-only [`RenderQueue`] capturing submitted command buffers.
struct MockQueue {
    submitted: RefCell<Vec<CommandBuffer>>,
}

impl MockQueue {
    fn new() -> Self {
        Self {
            submitted: RefCell::new(Vec::new()),
        }
    }
}

impl RenderQueue for MockQueue {
    fn write_buffer(&self, _buffer: BufferId, _offset: u64, _data: &[u8]) {}
    fn write_texture(&self, _destination: TextureWrite, _data: &[u8], _layout: ImageDataLayout) {}
    fn submit(&self, command_buffers: &[CommandBuffer]) {
        self.submitted
            .borrow_mut()
            .extend_from_slice(command_buffers);
    }
}

// --- fixtures -------------------------------------------------------------

fn swap() -> Extent3d {
    Extent3d {
        width: 1920,
        height: 1080,
        depth_or_array_layers: 1,
    }
}

/// A fresh imported swapchain image seeded with distinctive ids.
fn imported_swapchain() -> ImportedTexture {
    ImportedTexture {
        texture: TextureId::from_parts(1000, 0),
        default_view: TextureViewId::from_parts(1000, 0),
        entry_state: TextureState::initial(),
        exit_state: TextureState::present(),
        mip_levels: 1,
        array_layers: 1,
    }
}

fn import_sc(g: &mut RenderGraph) -> TextureHandle {
    g.import_texture(
        "swapchain",
        TextureDesc::color(TextureFormat::Bgra8Unorm),
        imported_swapchain(),
    )
}

/// Builds the canonical two-pass chain: pass `A` clears an offscreen color
/// target, pass `B` samples it and writes the imported swapchain. `B` is a
/// root (writes an imported resource) and `A` feeds `B`, so both survive.
fn build_chain() -> RenderGraph {
    let mut g = RenderGraph::new(swap());
    let sc = import_sc(&mut g);

    g.add_raster_pass("A", PassFlags::EMPTY, |b| {
        let off = b.create_texture("offscreen", TextureDesc::color(TextureFormat::Rgba8Unorm));
        let off = b.color_attachment(off);
        b.blackboard_mut().set(off);
        |_ctx| {}
    });

    g.add_raster_pass("B", PassFlags::EMPTY, move |b| {
        let off = *b
            .blackboard()
            .get::<TextureHandle>()
            .expect("pass A publishes the offscreen handle");
        b.sample(off);
        b.color_attachment(sc);
        |_ctx| {}
    });

    g
}

// --- culling --------------------------------------------------------------

#[test]
fn culls_pass_whose_output_is_never_observed() {
    let mut g = RenderGraph::new(swap());
    let sc = import_sc(&mut g);

    // Root: writes the imported swapchain.
    g.add_raster_pass("main", PassFlags::EMPTY, move |b| {
        b.color_attachment(sc);
        |_ctx| {}
    });
    // Dead: writes a transient nobody reads and has no side effect.
    g.add_raster_pass("dead", PassFlags::EMPTY, |b| {
        let t = b.create_texture("dead_tex", TextureDesc::color(TextureFormat::Rgba8Unorm));
        b.color_attachment(t);
        |_ctx| {}
    });

    let plan = g.compile().expect("acyclic graph compiles");
    assert!(plan.alive[0], "root pass survives culling");
    assert!(!plan.alive[1], "unobserved pass is culled");
    assert_eq!(plan.order, vec![0], "only the root is scheduled");
    // swapchain is textures[0]; dead_tex is textures[1].
    assert!(!plan.used_textures[1], "a culled pass's resource is unused");
}

// --- scheduling (SSA ordering) --------------------------------------------

#[test]
fn schedules_producer_before_consumer() {
    let g = build_chain();
    let plan = g.compile().expect("acyclic graph compiles");

    let pos_a = plan
        .order
        .iter()
        .position(|&i| i == 0)
        .expect("A scheduled");
    let pos_b = plan
        .order
        .iter()
        .position(|&i| i == 1)
        .expect("B scheduled");
    assert!(pos_a < pos_b, "producer A runs before consumer B");
}

// --- barriers -------------------------------------------------------------

#[test]
fn plans_barriers_for_layout_transitions() {
    let g = build_chain();
    let plan = g.compile().expect("acyclic graph compiles");
    // The offscreen target transitions color-attachment -> sampled between A
    // and B, which must be synchronized.
    assert!(
        plan.barrier_count() > 0,
        "a color->sampled transition needs at least one barrier"
    );
}

// --- transient aliasing ---------------------------------------------------

#[test]
fn aliases_disjoint_transients_into_shared_memory() {
    let mut g = RenderGraph::new(swap());
    let sc = import_sc(&mut g);

    // Two independent produce/consume pairs that never coexist: `t0` is born in
    // `A` and dies in `B`; only afterwards is `t1` born in `C` and consumed in
    // `D`. Both passes blit into the imported swapchain, so every pass is a
    // live root and nothing is culled. Because `t0`'s last use (`B`) strictly
    // precedes `t1`'s first use (`C`), their lifetimes are disjoint and the
    // aliaser is free to place both at the same byte offset.
    g.add_raster_pass("A", PassFlags::EMPTY, |b| {
        let t0 = b.create_texture("t0", TextureDesc::color(TextureFormat::Rgba8Unorm));
        let t0 = b.color_attachment(t0);
        b.blackboard_mut().set((t0,));
        |_ctx| {}
    });
    g.add_raster_pass("B", PassFlags::EMPTY, move |b| {
        let (t0,) = *b.blackboard().get::<(TextureHandle,)>().expect("A set t0");
        b.sample(t0);
        b.color_attachment(sc);
        |_ctx| {}
    });
    g.add_raster_pass("C", PassFlags::EMPTY, |b| {
        let t1 = b.create_texture("t1", TextureDesc::color(TextureFormat::Rgba8Unorm));
        let t1 = b.color_attachment(t1);
        b.blackboard_mut().set(t1);
        |_ctx| {}
    });
    g.add_raster_pass("D", PassFlags::EMPTY, move |b| {
        let t1 = *b.blackboard().get::<TextureHandle>().expect("C set t1");
        b.sample(t1);
        b.color_attachment(sc);
        |_ctx| {}
    });

    let plan = g.compile().expect("acyclic graph compiles");
    // textures: [swapchain=0, t0=1, t1=2].
    let slot_t0 = plan.alias.texture_slots[1].expect("t0 is an aliasable transient");
    let slot_t1 = plan.alias.texture_slots[2].expect("t1 is an aliasable transient");
    assert_eq!(
        slot_t0.offset, slot_t1.offset,
        "non-overlapping transients of equal size reuse the same memory"
    );
    assert!(
        plan.alias.heap_size < slot_t0.size + slot_t1.size,
        "aliasing shrinks the heap below the naive sum"
    );
}

// --- attachment derivation ------------------------------------------------

#[test]
fn clears_fresh_target_and_stores_only_consumed_output() {
    let g = build_chain();
    let plan = g.compile().expect("acyclic graph compiles");

    let pos_a = plan
        .order
        .iter()
        .position(|&i| i == 0)
        .expect("A scheduled");
    let att = plan.attachments[pos_a]
        .as_ref()
        .expect("raster pass A has derived attachments");
    let color = &att.colors[0];
    assert_eq!(
        color.load,
        LoadOp::Clear(Color::BLACK),
        "a freshly written version-0 target is cleared"
    );
    assert_eq!(
        color.store,
        StoreOp::Store,
        "an output consumed by a later pass is stored"
    );
}

#[test]
fn discards_unconsumed_side_effect_output() {
    let mut g = RenderGraph::new(swap());
    // A side-effecting pass is kept even though nothing reads its output.
    g.add_raster_pass("probe", PassFlags::SIDE_EFFECT, |b| {
        let t = b.create_texture("scratch", TextureDesc::color(TextureFormat::Rgba8Unorm));
        b.color_attachment(t);
        |_ctx| {}
    });

    let plan = g.compile().expect("acyclic graph compiles");
    assert_eq!(plan.order, vec![0], "the side-effect pass survives");
    let att = plan.attachments[0]
        .as_ref()
        .expect("raster pass has attachments");
    assert_eq!(
        att.colors[0].store,
        StoreOp::Discard,
        "an output nobody consumes is discarded"
    );
}

// --- usage inference / lowering -------------------------------------------

#[test]
fn folds_inferred_usage_into_realized_descriptor() {
    let g = build_chain();
    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("chain executes");

    // The offscreen target is both a color attachment (pass A) and sampled
    // (pass B); its realized descriptor must carry both usages.
    let textures = device.textures.borrow();
    let offscreen = textures
        .iter()
        .find(|d| d.usage.contains(TextureUsages::TEXTURE_BINDING))
        .expect("a sampled transient was created");
    assert!(
        offscreen.usage.contains(TextureUsages::RENDER_ATTACHMENT),
        "attachment usage is unioned with sampled usage"
    );
}

// --- execution ------------------------------------------------------------

#[test]
fn execute_records_one_render_pass_per_surviving_raster_pass() {
    let g = build_chain();
    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("chain executes");

    let submitted = queue.submitted.borrow();
    assert_eq!(submitted.len(), 1, "execution submits exactly one buffer");
    let cb = &submitted[0];
    assert_eq!(
        cb.passes.len(),
        2,
        "two raster passes become two GPU passes"
    );
    assert!(
        cb.passes.iter().all(|p| matches!(p, Pass::Render { .. })),
        "both recorded passes are render passes"
    );
}

#[test]
fn present_pass_is_a_root_but_records_no_gpu_pass() {
    let mut g = RenderGraph::new(swap());
    let sc = import_sc(&mut g);
    g.add_raster_pass("main", PassFlags::EMPTY, move |b| {
        b.color_attachment(sc);
        |_ctx| {}
    });
    g.add_present_pass("present", PassFlags::EMPTY, move |b| {
        b.present_texture(sc);
        |_ctx| {}
    });

    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("present chain executes");

    let submitted = queue.submitted.borrow();
    assert_eq!(submitted.len(), 1);
    let cb = &submitted[0];
    assert_eq!(
        cb.passes.len(),
        1,
        "only the raster pass records a GPU pass; present is state-only"
    );
    assert!(matches!(cb.passes[0], Pass::Render { .. }));
}

// --- label plumbing -------------------------------------------------------

#[test]
fn render_pass_carries_its_debug_label() {
    let g = build_chain();
    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("chain executes");

    let submitted = queue.submitted.borrow();
    let cb = &submitted[0];
    let labels: Vec<Option<String>> = cb
        .passes
        .iter()
        .map(|p| match p {
            Pass::Render { descriptor, .. } => descriptor.label.clone(),
            Pass::Compute { label, .. } | Pass::Transfer { label, .. } => label.clone(),
        })
        .collect();
    assert!(
        labels.contains(&Some(String::from("A"))) && labels.contains(&Some(String::from("B"))),
        "pass names flow through to driver pass labels: {labels:?}"
    );
}

// --- transfer recording ---------------------------------------------------

/// A transfer pass over imported source/destination buffers must record exactly
/// one driver `Pass::Transfer` whose `CopyCommand` matches what the pass
/// declared, with virtual handles resolved to the imported driver ids.
#[test]
fn transfer_pass_records_copy_commands() {
    let mut g = RenderGraph::new(swap());
    let src = g.import_buffer(
        "src",
        BufferDesc::new(256, BufferUsages::COPY_SRC),
        ImportedBuffer {
            buffer: BufferId::from_parts(2000, 0),
            entry_state: BufferState::initial(),
        },
    );
    let dst = g.import_buffer(
        "dst",
        BufferDesc::new(256, BufferUsages::COPY_DST),
        ImportedBuffer {
            buffer: BufferId::from_parts(2001, 0),
            entry_state: BufferState::initial(),
        },
    );

    g.add_transfer_pass("copy", PassFlags::EMPTY, move |b| {
        let src = b.copy_src_buffer(src);
        let dst = b.copy_dst_buffer(dst);
        move |ctx| {
            ctx.copy_buffer_to_buffer(src, 0, dst, 0, 256);
        }
    });

    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("transfer graph executes");

    let submitted = queue.submitted.borrow();
    assert_eq!(submitted.len(), 1, "execution submits exactly one buffer");
    let cb = &submitted[0];
    assert_eq!(cb.passes.len(), 1, "one transfer pass becomes one GPU pass");
    let Pass::Transfer { label, commands } = &cb.passes[0] else {
        panic!("a transfer pass must record a Pass::Transfer");
    };
    assert_eq!(
        label.as_deref(),
        Some("copy"),
        "the pass name flows through to the driver label"
    );
    assert_eq!(
        commands,
        &vec![CopyCommand::CopyBufferToBuffer {
            src: BufferId::from_parts(2000, 0),
            src_offset: 0,
            dst: BufferId::from_parts(2001, 0),
            dst_offset: 0,
            size: 256,
        }],
        "the recorded copy matches the declared handles and offsets"
    );
}

/// A transfer pass with an unconsumed output survives culling when flagged
/// [`PassFlags::SIDE_EFFECT`] and is recorded; one with no flag and no consumer
/// is culled and records nothing — mirroring the raster culling tests.
#[test]
fn culls_unconsumed_transfer_pass_unless_side_effecting() {
    let mut g = RenderGraph::new(swap());

    // Dead: copies into transients nobody reads, with no side-effect flag.
    g.add_transfer_pass("dead", PassFlags::EMPTY, |b| {
        let src = b.create_buffer("dead_src", BufferDesc::new(64, BufferUsages::COPY_SRC));
        let dst = b.create_buffer("dead_dst", BufferDesc::new(64, BufferUsages::COPY_DST));
        let src = b.copy_src_buffer(src);
        let dst = b.copy_dst_buffer(dst);
        move |ctx| {
            ctx.copy_buffer_to_buffer(src, 0, dst, 0, 64);
        }
    });

    // Kept: declared side-effecting, so it is a root even though nothing reads
    // its output.
    g.add_transfer_pass("kept", PassFlags::SIDE_EFFECT, |b| {
        let src = b.create_buffer("kept_src", BufferDesc::new(64, BufferUsages::COPY_SRC));
        let dst = b.create_buffer("kept_dst", BufferDesc::new(64, BufferUsages::COPY_DST));
        let src = b.copy_src_buffer(src);
        let dst = b.copy_dst_buffer(dst);
        move |ctx| {
            ctx.copy_buffer_to_buffer(src, 0, dst, 0, 64);
        }
    });

    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("transfer graph executes");

    let submitted = queue.submitted.borrow();
    assert_eq!(submitted.len(), 1);
    let cb = &submitted[0];
    assert_eq!(
        cb.passes.len(),
        1,
        "only the side-effecting transfer is recorded"
    );
    let Pass::Transfer { label, commands } = &cb.passes[0] else {
        panic!("expected a single recorded Pass::Transfer");
    };
    assert_eq!(label.as_deref(), Some("kept"));
    assert_eq!(commands.len(), 1, "the surviving pass recorded its copy");
}

/// Every transfer variant resolves its virtual handles to the realized driver
/// ids, including the texture forms, and records them in declaration order.
#[test]
fn transfer_pass_records_every_copy_variant() {
    fn imported_color(texture: u32, view: u32) -> ImportedTexture {
        ImportedTexture {
            texture: TextureId::from_parts(texture, 0),
            default_view: TextureViewId::from_parts(view, 0),
            entry_state: TextureState::initial(),
            exit_state: TextureState::present(),
            mip_levels: 1,
            array_layers: 1,
        }
    }

    let extent = Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
    };

    let mut g = RenderGraph::new(swap());
    let src_tex = g.import_texture(
        "src_tex",
        TextureDesc::color(TextureFormat::Rgba8Unorm),
        imported_color(3000, 3001),
    );
    let dst_tex = g.import_texture(
        "dst_tex",
        TextureDesc::color(TextureFormat::Rgba8Unorm),
        imported_color(3002, 3003),
    );
    let buf = g.import_buffer(
        "staging",
        BufferDesc::new(64, BufferUsages::COPY_SRC.union(BufferUsages::COPY_DST)),
        ImportedBuffer {
            buffer: BufferId::from_parts(2002, 0),
            entry_state: BufferState::initial(),
        },
    );

    g.add_transfer_pass("blit", PassFlags::EMPTY, move |b| {
        let src_tex = b.copy_src_texture(src_tex);
        let dst_tex = b.copy_dst_texture(dst_tex);
        let copy_src_buf = b.copy_src_buffer(buf);
        b.copy_dst_buffer(buf);
        move |ctx| {
            ctx.copy_buffer_to_texture(buf, 0, None, None, dst_tex, 0, [0, 0, 0], extent)
                .copy_texture_to_buffer(src_tex, 0, [0, 0, 0], copy_src_buf, 8, None, None, extent)
                .copy_texture_to_texture(src_tex, 0, [0, 0, 0], dst_tex, 0, [0, 0, 0], extent);
        }
    });

    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("transfer graph executes");

    let submitted = queue.submitted.borrow();
    let cb = &submitted[0];
    let Pass::Transfer { commands, .. } = &cb.passes[0] else {
        panic!("expected a recorded Pass::Transfer");
    };
    assert_eq!(
        commands,
        &vec![
            CopyCommand::CopyBufferToTexture {
                src: BufferId::from_parts(2002, 0),
                src_offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
                dst: TextureId::from_parts(3002, 0),
                dst_mip_level: 0,
                dst_origin: [0, 0, 0],
                size: extent,
            },
            CopyCommand::CopyTextureToBuffer {
                src: TextureId::from_parts(3000, 0),
                src_mip_level: 0,
                src_origin: [0, 0, 0],
                dst: BufferId::from_parts(2002, 0),
                dst_offset: 8,
                bytes_per_row: None,
                rows_per_image: None,
                size: extent,
            },
            CopyCommand::CopyTextureToTexture {
                src: TextureId::from_parts(3000, 0),
                src_mip_level: 0,
                src_origin: [0, 0, 0],
                dst: TextureId::from_parts(3002, 0),
                dst_mip_level: 0,
                dst_origin: [0, 0, 0],
                size: extent,
            },
        ],
        "all three texture-touching variants resolve to the imported driver ids"
    );
}
