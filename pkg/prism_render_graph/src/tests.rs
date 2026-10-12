//! Unit tests for the render graph: compilation and execution invariants.
//!
//! Every test drives the real compiler and (where relevant) a mock
//! [`RenderDevice`]/[`RenderQueue`] pair, so the full pipeline — culling, SSA
//! scheduling, lifetime/alias analysis, barrier planning, attachment
//! derivation, and command recording — is exercised with no real GPU.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use prism_render_driver::{
    Backend, BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferDescriptor, BufferId, BufferState, BufferUsages, Color, CommandBuffer, ComputeCommand,
    ComputePipelineDescriptor, ComputePipelineId, CopyCommand, DeviceCapabilities, Extent3d,
    Features, ImageDataLayout, Limits, LoadOp, Pass, PipelineLayoutDescriptor, PipelineLayoutId,
    RenderCommand, RenderDevice, RenderPipelineDescriptor, RenderPipelineId, RenderQueue,
    SamplerDescriptor, SamplerId, ShaderModuleDescriptor, ShaderModuleId, StoreOp,
    TextureDescriptor, TextureFormat, TextureId, TextureState, TextureUsages,
    TextureViewDescriptor, TextureViewId, TextureWrite, Viewport, DRAW_INDEXED_INDIRECT_ARGS_SIZE,
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
    imported_texture(1000, 1000)
}

/// An imported texture seeded with distinctive ids.
fn imported_texture(texture: u32, view: u32) -> ImportedTexture {
    ImportedTexture {
        texture: TextureId::from_parts(texture, 0),
        default_view: TextureViewId::from_parts(view, 0),
        entry_state: TextureState::initial(),
        exit_state: TextureState::present(),
        mip_levels: 1,
        array_layers: 1,
    }
}

/// The debug label of every recorded GPU pass, in submission order.
fn labels(buffer: &CommandBuffer) -> Vec<Option<String>> {
    buffer
        .passes
        .iter()
        .map(|p| match p {
            Pass::Render { descriptor, .. } => descriptor.label.clone(),
            Pass::Compute { label, .. } | Pass::Transfer { label, .. } => label.clone(),
        })
        .collect()
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
    let recorded = labels(cb);
    assert!(
        recorded.contains(&Some(String::from("A"))) && recorded.contains(&Some(String::from("B"))),
        "pass names flow through to driver pass labels: {recorded:?}"
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

// --- parallel recording ---------------------------------------------------

/// Builds a graph of `passes` mutually independent passes cycling through every
/// pass kind — raster, compute, transfer, present. Each pass owns its own
/// imported resources, so the compiler derives no cross-pass dependencies and
/// the schedule is exactly declaration order; this is the fixture the two
/// executors must agree on byte for byte.
fn mixed_fixture(passes: usize) -> RenderGraph {
    let mut g = RenderGraph::new(swap());
    for i in 0..passes {
        let id = i as u32;
        match i % 4 {
            0 => {
                let target = g.import_texture(
                    format!("raster-target-{i}"),
                    TextureDesc::color(TextureFormat::Rgba8Unorm),
                    imported_texture(7_000 + id, 8_000 + id),
                );
                g.add_raster_pass(format!("raster-{i}"), PassFlags::EMPTY, move |b| {
                    b.color_attachment(target);
                    move |ctx| {
                        ctx.set_viewport(Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: 32.0,
                            height: 32.0,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        });
                        ctx.set_blend_constant(Color::WHITE);
                        ctx.draw(0..3, 0..1);
                    }
                });
            }
            1 => {
                g.add_compute_pass(format!("compute-{i}"), PassFlags::SIDE_EFFECT, move |_b| {
                    move |ctx| {
                        ctx.dispatch(id + 1, 2, 3);
                    }
                });
            }
            2 => {
                let src = g.import_buffer(
                    format!("transfer-src-{i}"),
                    BufferDesc::new(256, BufferUsages::COPY_SRC),
                    ImportedBuffer {
                        buffer: BufferId::from_parts(9_000 + id, 0),
                        entry_state: BufferState::initial(),
                    },
                );
                let dst = g.import_buffer(
                    format!("transfer-dst-{i}"),
                    BufferDesc::new(256, BufferUsages::COPY_DST),
                    ImportedBuffer {
                        buffer: BufferId::from_parts(10_000 + id, 0),
                        entry_state: BufferState::initial(),
                    },
                );
                g.add_transfer_pass(format!("transfer-{i}"), PassFlags::EMPTY, move |b| {
                    let src = b.copy_src_buffer(src);
                    let dst = b.copy_dst_buffer(dst);
                    move |ctx| {
                        ctx.copy_buffer_to_buffer(src, 0, dst, 8, 128);
                    }
                });
            }
            _ => {
                let target = g.import_texture(
                    format!("present-target-{i}"),
                    TextureDesc::color(TextureFormat::Bgra8Unorm),
                    imported_texture(11_000 + id, 12_000 + id),
                );
                g.add_present_pass(format!("present-{i}"), PassFlags::EMPTY, move |b| {
                    b.present_texture(target);
                    |_ctx| {}
                });
            }
        }
    }
    g
}

/// Runs `graph` through the feature-selected executor and returns what it
/// submitted.
fn submitted_by_active(graph: RenderGraph) -> Vec<CommandBuffer> {
    let device = MockDevice::new();
    let queue = MockQueue::new();
    graph
        .execute(&device, &queue)
        .expect("the active executor runs the graph");
    queue.submitted.borrow().clone()
}

#[cfg(feature = "parallel")]
fn submitted_sequentially(graph: RenderGraph) -> Vec<CommandBuffer> {
    let device = MockDevice::new();
    let queue = MockQueue::new();
    graph
        .execute_sequential(&device, &queue)
        .expect("the sequential executor runs the graph");
    queue.submitted.borrow().clone()
}

#[cfg(feature = "parallel")]
fn submitted_in_parallel(graph: RenderGraph) -> Vec<CommandBuffer> {
    let device = MockDevice::new();
    let queue = MockQueue::new();
    graph
        .execute_parallel(&device, &queue)
        .expect("the parallel executor runs the graph");
    queue.submitted.borrow().clone()
}

/// The feature-selected executor is deterministic: recording the same graph
/// twice yields identical command buffers. Compiles and runs with `parallel`
/// both on and off.
#[test]
fn repeated_execution_is_deterministic() {
    let first = submitted_by_active(mixed_fixture(9));
    let second = submitted_by_active(mixed_fixture(9));
    assert_eq!(
        first, second,
        "recording the same graph twice must be deterministic"
    );
    assert_eq!(first.len(), 1, "one command buffer is submitted per frame");
}

/// The worker pool scales to the machine's parallelism without ever spawning
/// more workers than there are units, and its chunking covers every unit — the
/// policy the determinism guarantee rests on.
#[cfg(feature = "parallel")]
#[test]
fn worker_pool_partitions_cover_all_units() {
    let available = std::thread::available_parallelism().map_or(1, core::num::NonZeroUsize::get);
    assert_eq!(
        crate::parallel::worker_count(0),
        1,
        "the pool never reports zero workers"
    );
    for total in [1usize, 2, 7, 64, 1000] {
        let workers = crate::parallel::worker_count(total);
        assert!(
            (1..=total).contains(&workers),
            "workers stay within 1..=units for {total} units"
        );
        assert!(
            workers <= available,
            "the pool never exceeds the machine's parallelism"
        );
        let chunk_size = total.div_ceil(workers);
        assert!(
            chunk_size * (workers - 1) < total,
            "chunking with {workers} workers covers all {total} units"
        );
    }
}

/// The parallel executor must reproduce the sequential executor's command
/// buffer exactly for a graph of independent raster/compute/transfer/present
/// passes, preserving plan order.
#[cfg(feature = "parallel")]
#[test]
fn parallel_recording_matches_sequential() {
    let parallel = submitted_in_parallel(mixed_fixture(8));
    let sequential = submitted_sequentially(mixed_fixture(8));
    assert_eq!(
        parallel, sequential,
        "parallel assembly must equal sequential recording byte for byte"
    );
    assert_eq!(
        parallel.len(),
        1,
        "one command buffer is submitted per frame"
    );
    assert_eq!(
        parallel[0].passes.len(),
        6,
        "8 passes, two of each kind; present records no GPU pass"
    );
    assert_eq!(
        labels(&parallel[0]),
        vec![
            Some(String::from("raster-0")),
            Some(String::from("compute-1")),
            Some(String::from("transfer-2")),
            Some(String::from("raster-4")),
            Some(String::from("compute-5")),
            Some(String::from("transfer-6")),
        ],
        "assembly follows the plan's deterministic declaration order"
    );
}

/// A stress-sized graph (well past 50 passes) must also assemble identically,
/// proving the worker-pool chunking preserves determinism at scale.
#[cfg(feature = "parallel")]
#[test]
fn parallel_recording_matches_sequential_for_large_graph() {
    const PASSES: usize = 64;
    let parallel = submitted_in_parallel(mixed_fixture(PASSES));
    let sequential = submitted_sequentially(mixed_fixture(PASSES));
    assert_eq!(
        parallel, sequential,
        "the stress graph must assemble identically in parallel"
    );
    assert_eq!(
        parallel.len(),
        1,
        "one command buffer is submitted per frame"
    );
    assert_eq!(
        parallel[0].passes.len(),
        PASSES / 4 * 3,
        "64 passes, three quarters record a GPU pass"
    );
    let recorded = labels(&parallel[0]);
    assert_eq!(
        recorded.first().and_then(Clone::clone),
        Some(String::from("raster-0")),
        "the first recorded pass is the first declared pass"
    );
    assert_eq!(
        recorded.last().and_then(Clone::clone),
        Some(String::from("transfer-62")),
        "the last recorded pass is the last declared recording pass"
    );
}

// --- indirect recording ---------------------------------------------------

/// An imported indirect-argument buffer, as a GPU-authored cull would leave it.
fn imported_args(buffer: u32) -> (BufferDesc, ImportedBuffer) {
    (
        BufferDesc::new(
            64,
            BufferUsages::STORAGE
                .union(BufferUsages::INDIRECT)
                .union(BufferUsages::COPY_DST),
        ),
        ImportedBuffer {
            buffer: BufferId::from_parts(buffer, 0),
            entry_state: BufferState::storage_write(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
        },
    )
}

/// A raster pass records both indexed-indirect primitives, resolving the
/// declared argument handle to the imported driver buffer.
#[test]
fn raster_pass_records_indexed_and_multi_indirect_draws() {
    let mut g = RenderGraph::new(swap());
    let sc = import_sc(&mut g);
    let (args_desc, args_import) = imported_args(4000);
    let args = g.import_buffer("args", args_desc, args_import);

    g.add_raster_pass("indirect", PassFlags::EMPTY, move |b| {
        b.read_indirect(args);
        b.color_attachment(sc);
        move |ctx| {
            ctx.draw_indexed_indirect(args, 8);
            ctx.multi_draw_indirect(args, 0, 4, DRAW_INDEXED_INDIRECT_ARGS_SIZE);
        }
    });

    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue).expect("indirect graph executes");

    let submitted = queue.submitted.borrow();
    let cb = &submitted[0];
    let Pass::Render { commands, .. } = &cb.passes[0] else {
        panic!("a raster pass must record a Pass::Render");
    };
    assert_eq!(
        commands,
        &vec![
            RenderCommand::DrawIndexedIndirect {
                buffer: BufferId::from_parts(4000, 0),
                offset: 8,
            },
            RenderCommand::MultiDrawIndirect {
                buffer: BufferId::from_parts(4000, 0),
                offset: 0,
                count: 4,
                stride: DRAW_INDEXED_INDIRECT_ARGS_SIZE,
            },
        ],
        "both indirect recorders resolve the declared handle to the imported buffer"
    );
}

/// A compute pass records an indirect dispatch from a declared argument buffer,
/// so a GPU-authored workgroup count reaches the command stream through the
/// graph.
#[test]
fn compute_pass_records_indirect_dispatch() {
    let mut g = RenderGraph::new(swap());
    let (args_desc, args_import) = imported_args(4100);
    let args = g.import_buffer("args", args_desc, args_import);

    g.add_compute_pass("indirect-dispatch", PassFlags::SIDE_EFFECT, move |b| {
        b.read_indirect(args);
        move |ctx| {
            ctx.dispatch_indirect(args, 16);
        }
    });

    let device = MockDevice::new();
    let queue = MockQueue::new();
    g.execute(&device, &queue)
        .expect("indirect dispatch executes");

    let submitted = queue.submitted.borrow();
    let cb = &submitted[0];
    let Pass::Compute { commands, .. } = &cb.passes[0] else {
        panic!("a compute pass must record a Pass::Compute");
    };
    assert_eq!(
        commands,
        &vec![ComputeCommand::DispatchIndirect {
            buffer: BufferId::from_parts(4100, 0),
            offset: 16,
        }],
        "the indirect dispatch resolves the declared handle to the imported buffer"
    );
}
/// A depth attachment nobody has written is cleared to the engine's *far* plane:
/// reversed-Z puts the near plane at 1 and the far plane at 0, so the clear
/// value is `0.0` (with `GreaterEqual` testing, anything else would clear the
/// buffer to "in front of every fragment" and reject the whole frame).
#[test]
fn a_fresh_depth_attachment_is_cleared_to_the_reversed_z_far_plane() {
    let mut g = RenderGraph::new(swap());
    let color = g.import_texture(
        "color",
        TextureDesc::color(TextureFormat::Rgba8Unorm),
        imported_texture(1, 2),
    );
    let depth = g.import_texture(
        "depth",
        TextureDesc::color(TextureFormat::Depth32Float),
        imported_texture(3, 4),
    );
    g.add_raster_pass("with-depth", PassFlags::SIDE_EFFECT, move |b| {
        b.color_attachment(color);
        b.depth_attachment(depth);
        |_ctx| {}
    });

    let plan = g.compile().expect("acyclic graph compiles");
    let index = plan
        .order
        .iter()
        .position(|&i| i == 0)
        .expect("the pass is scheduled");
    let depth = plan.attachments[index]
        .as_ref()
        .expect("the pass has attachments")
        .depth
        .as_ref()
        .expect("the pass has a depth attachment");
    assert!(depth.writes, "the pass writes depth");
    assert_eq!(
        depth.depth_load,
        prism_render_driver::DepthLoadOp::Clear(0.0),
        "a fresh depth attachment clears to the reversed-Z far plane"
    );
}
