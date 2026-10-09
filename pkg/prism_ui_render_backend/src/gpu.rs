//! Optional `wgpu`/Metal GPU backend — the parity twin of [`crate::raster`].
//!
//! This module mirrors the headless CPU rasteriser one-to-one. Every pixel is
//! shaded by a single compute invocation that loops the flattened draw stream
//! in paint order and evaluates the *exact same* signed-distance math and
//! straight-alpha-over compositing as [`crate::raster::rasterize`]. Keeping the
//! algorithm identical is what lets a parity test assert the two backends agree
//! to within GPU floating-point rounding (see `tests/gpu_parity.rs`).
//!
//! The design follows the "CPU golden twin" pattern already used by
//! `prism_physics_gpu`: acquisition is best-effort ([`GpuRasterizer::try_new`]
//! returns [`None`] on a host with no usable adapter) so a test can skip rather
//! than fail, while a real device such as an Apple M-series GPU exercises the
//! full dispatch and read-back.
//!
//! No `unsafe` is used: `wgpu`'s API is safe and `bytemuck` is only invoked
//! through its safe `cast_slice` helper over `f32`/`u32` slices.
//!
//! Provenance: standard `wgpu` compute initialisation. Contains no Unreal
//! Engine source or derived code.

use alloc::vec;
use alloc::vec::Vec;

use prism_ui_style::Color;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Adapter, AddressMode, BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Extent3d, FilterMode, Instance,
    InstanceDescriptor, InstanceFlags, MapMode, Origin3d, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PollType, Queue, RequestAdapterOptions, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};

use crate::draw::{DrawCommand, DrawList, GlyphCmd};
use crate::layer::LayerTree;
use crate::raster::Framebuffer;

/// Number of `f32` lanes per encoded primitive (seven `vec4<f32>` = 112 bytes).
const PRIM_STRIDE: usize = 28;
/// Compute workgroup edge; the shader uses an `8x8` tile.
const WORKGROUP: u32 = 8;

/// WGSL compute shader. The distance functions and blend are a line-for-line
/// mirror of [`crate::sdf`] and [`crate::raster::Framebuffer::blend`].
const SHADER: &str = r#"
struct Prim {
    rect:   vec4<f32>,  // x, y, w, h   (top-left, size)
    fill:   vec4<f32>,  // r, g, b, a
    border: vec4<f32>,  // r, g, b, a
    params: vec4<f32>,  // radius, border_width, blur, opacity
    offset: vec4<f32>,  // offset.x, offset.y, screen_px_range(glyph), _
    flags:  vec4<f32>,  // kind(0=shadow,1=rect,2=glyph), has_fill, has_border, _
    glyph:  vec4<f32>,  // su0, sv0, sstep, tstep  (normalised atlas uv ramp)
};

@group(0) @binding(0) var<storage, read>       prims:   array<Prim>;
@group(0) @binding(1) var<storage, read_write> out_px:  array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             dims:    vec4<u32>; // w, h, count, _
@group(0) @binding(3) var                      atlas:   texture_2d<f32>;
@group(0) @binding(4) var                      atlas_s: sampler;

fn sd_rounded_box(px: f32, py: f32, half_w: f32, half_h: f32, radius: f32) -> f32 {
    let r = clamp(radius, 0.0, min(half_w, half_h));
    let qx = abs(px) - (half_w - r);
    let qy = abs(py) - (half_h - r);
    let outside = length(vec2<f32>(max(qx, 0.0), max(qy, 0.0)));
    let inside = min(max(qx, qy), 0.0);
    return outside + inside - r;
}

fn coverage(dist: f32) -> f32 {
    return clamp(0.5 - dist, 0.0, 1.0);
}

fn border_coverage(dist: f32, width: f32) -> f32 {
    if (width <= 0.0) {
        return 0.0;
    }
    let outer = coverage(dist);
    let inner = coverage(dist + width);
    return clamp(outer - inner, 0.0, 1.0);
}

fn shadow_alpha(dist: f32, blur: f32) -> f32 {
    if (blur <= 0.0) {
        if (dist <= 0.0) { return 1.0; }
        return 0.0;
    }
    let t = clamp(dist / blur, 0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    return 1.0 - s;
}

// Straight (non-premultiplied) alpha-over, identical to the CPU reference.
fn over(dst: vec4<f32>, src: vec4<f32>) -> vec4<f32> {
    if (src.a <= 0.0) {
        return dst;
    }
    let sa = clamp(src.a, 0.0, 1.0);
    let out_a = sa + dst.a * (1.0 - sa);
    if (out_a <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, out_a);
    }
    let rgb = (src.rgb * sa + dst.rgb * dst.a * (1.0 - sa)) / out_a;
    return vec4<f32>(rgb, out_a);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.x;
    let h = dims.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let count = dims.z;
    let sx = f32(gid.x) + 0.5;
    let sy = f32(gid.y) + 0.5;
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let p = prims[i];
        let hw = p.rect.z * 0.5;
        let hh = p.rect.w * 0.5;
        let kind = p.flags.x;
        let opacity = p.params.w;
        if (kind < 0.5) {
            // Shadow: centre shifted by offset.
            let cx = p.rect.x + hw + p.offset.x;
            let cy = p.rect.y + hh + p.offset.y;
            let dist = sd_rounded_box(sx - cx, sy - cy, hw, hh, p.params.x);
            let a = shadow_alpha(dist, p.params.z) * opacity * p.fill.a;
            acc = over(acc, vec4<f32>(p.fill.rgb, a));
        } else if (kind < 1.5) {
            // Rect: fill then border ring.
            let cx = p.rect.x + hw;
            let cy = p.rect.y + hh;
            let dist = sd_rounded_box(sx - cx, sy - cy, hw, hh, p.params.x);
            if (p.flags.y > 0.5) {
                let cov = coverage(dist) * opacity * p.fill.a;
                acc = over(acc, vec4<f32>(p.fill.rgb, cov));
            }
            if (p.params.y > 0.0 && p.flags.z > 0.5) {
                let cov = border_coverage(dist, p.params.y) * opacity * p.border.a;
                acc = over(acc, vec4<f32>(p.border.rgb, cov));
            }
        } else {
            // Glyph: sample the single-channel SDF atlas and apply the
            // screenPxRange coverage rule. This is a line-for-line mirror of
            // `prism_ui_font::sdf::coverage` and the CPU raster_glyph_sdf ramp:
            // only pixels inside this glyph's device cell are touched.
            let lx = sx - p.rect.x;
            let ly = sy - p.rect.y;
            if (lx >= 0.0 && ly >= 0.0 && lx <= p.rect.z && ly <= p.rect.w) {
                let su = p.glyph.x + lx * p.glyph.z;
                let sv = p.glyph.y + ly * p.glyph.w;
                let d = textureSampleLevel(atlas, atlas_s, vec2<f32>(su, sv), 0.0).r;
                let spr = max(p.offset.z, 1.0);
                let cov = clamp(spr * (d - 0.5) + 0.5, 0.0, 1.0) * opacity * p.fill.a;
                acc = over(acc, vec4<f32>(p.fill.rgb, cov));
            }
        }
    }

    out_px[gid.y * w + gid.x] = acc;
}
"#;

/// Blocks the current thread until `future` resolves.
fn block_on<F: Future>(future: F) -> F::Output {
    futures_lite::future::block_on(future)
}

/// A reusable GPU rasteriser: holds the device/queue and the compiled pipeline.
///
/// Construct once with [`GpuRasterizer::try_new`] and call
/// [`GpuRasterizer::rasterize`] per frame. The `_instance`/`_adapter` fields are
/// retained only to keep the device alive for the rasteriser's lifetime.
pub struct GpuRasterizer {
    _instance: Instance,
    _adapter: Adapter,
    device: Device,
    queue: Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    atlas_view: TextureView,
    atlas_sampler: Sampler,
}

impl GpuRasterizer {
    /// Attempts to acquire a headless compute device and build the pipeline.
    ///
    /// Returns [`None`] when no adapter or device is available, so callers can
    /// skip GPU work on hosts without a usable adapter instead of failing.
    #[must_use]
    pub fn try_new() -> Option<GpuRasterizer> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_ui_gpu_raster"),
            source: ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_ui_gpu_raster_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                uniform_entry(2),
                texture_entry(3),
                sampler_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_ui_gpu_raster_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_ui_gpu_raster_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let (atlas_view, atlas_sampler) = build_atlas_texture(&device, &queue);

        Some(GpuRasterizer {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            pipeline,
            layout,
            atlas_view,
            atlas_sampler,
        })
    }

    /// Rasterises `list` into a fresh [`Framebuffer`] on the GPU.
    ///
    /// The draw list is flattened by layer opacity (identical to
    /// [`crate::raster::rasterize`]) before upload, so nested group opacity is
    /// honoured the same way on both backends.
    #[must_use]
    pub fn rasterize(&self, list: &DrawList, width: u32, height: u32) -> Framebuffer {
        let flat = LayerTree::parse(list).flatten();
        let prims = encode_prims(&flat);
        let count = (prims.len() / PRIM_STRIDE) as u32;
        let pixel_count = (width as usize) * (height as usize);

        // A zero-length storage buffer is invalid; pad to one dummy primitive
        // that the shader skips because `count == 0`.
        let prim_data = if prims.is_empty() {
            vec![0.0f32; PRIM_STRIDE]
        } else {
            prims
        };

        let prim_buf = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_ui_gpu_prims"),
            contents: bytemuck::cast_slice(&prim_data),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (pixel_count.max(1) * 16) as u64;
        let out_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("prism_ui_gpu_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims = [width, height, count, 0u32];
        let dims_buf = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_ui_gpu_dims"),
            contents: bytemuck::cast_slice(&dims),
            usage: BufferUsages::UNIFORM,
        });
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("prism_ui_gpu_staging"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_ui_gpu_bg"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: prim_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dims_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_ui_gpu_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_ui_gpu_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = width.div_ceil(WORKGROUP).max(1);
            let gy = height.div_ceil(WORKGROUP).max(1);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(core::iter::once(encoder.finish()));

        staging.slice(..).map_async(MapMode::Read, |_| {});
        self.wait();
        let raw: Vec<f32> = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .expect("mapped readback range should be available after poll");
            let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
            drop(view);
            staging.unmap();
            values
        };

        let mut pixels = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let b = i * 4;
            pixels.push([raw[b], raw[b + 1], raw[b + 2], raw[b + 3]]);
        }
        Framebuffer::from_pixels(width, height, pixels)
    }

    fn wait(&self) {
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the submitted work");
    }
}

/// A single primitive in the shader's `array<Prim>` layout, prior to byte
/// encoding. `kind` is `0.0` for a shadow and `1.0` for a rect/glyph fill.
struct PrimRaw {
    rect: [f32; 4],
    fill: [f32; 4],
    border: [f32; 4],
    params: [f32; 4],
    offset: [f32; 2],
    kind: f32,
    has_fill: bool,
    has_border: bool,
    /// Per-pixel screenPxRange for the glyph tier (0 for non-glyphs).
    screen_px_range: f32,
    /// Normalised atlas uv ramp `(su0, sv0, sstep, tstep)` for the glyph tier.
    glyph: [f32; 4],
}

impl PrimRaw {
    fn write_to(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.rect);
        out.extend_from_slice(&self.fill);
        out.extend_from_slice(&self.border);
        out.extend_from_slice(&self.params);
        out.extend_from_slice(&[self.offset[0], self.offset[1], self.screen_px_range, 0.0]);
        out.extend_from_slice(&[
            self.kind,
            if self.has_fill { 1.0 } else { 0.0 },
            if self.has_border { 1.0 } else { 0.0 },
            0.0,
        ]);
        out.extend_from_slice(&self.glyph);
    }
}

/// Encodes the flattened draw list into the shader's `array<Prim>` layout.
///
/// Layer markers must already be removed by [`LayerTree::flatten`]; any that
/// remain are skipped. Text runs expand into one SDF-atlas glyph primitive per
/// inked character (`kind == 2`), laid out identically to the CPU reference's
/// `raster_glyph_sdf`, so the shader can sample [`atlas_sdf`] and apply the
/// same `screenPxRange` coverage. When the atlas is unavailable the run falls
/// back to a single opaque rect fill so text never silently vanishes.
fn encode_prims(flat: &DrawList) -> Vec<f32> {
    let transparent = Color::rgba(0.0, 0.0, 0.0, 0.0);
    let mut out = Vec::new();
    for cmd in flat.commands() {
        match *cmd {
            DrawCommand::Shadow(s) => PrimRaw {
                rect: [
                    s.rect.left(),
                    s.rect.top(),
                    s.rect.size.width,
                    s.rect.size.height,
                ],
                fill: [s.color.r, s.color.g, s.color.b, s.color.a],
                border: [0.0; 4],
                params: [s.radius, 0.0, s.blur, s.opacity],
                offset: [s.offset.x, s.offset.y],
                kind: 0.0,
                has_fill: true,
                has_border: false,
                screen_px_range: 0.0,
                glyph: [0.0; 4],
            }
            .write_to(&mut out),
            DrawCommand::Rect(r) => {
                let fill = r.fill.unwrap_or(transparent);
                let border = r.border_color.unwrap_or(transparent);
                PrimRaw {
                    rect: [
                        r.rect.left(),
                        r.rect.top(),
                        r.rect.size.width,
                        r.rect.size.height,
                    ],
                    fill: [fill.r, fill.g, fill.b, fill.a],
                    border: [border.r, border.g, border.b, border.a],
                    params: [r.radius, r.border_width, 0.0, r.opacity],
                    offset: [0.0, 0.0],
                    kind: 1.0,
                    has_fill: r.fill.is_some(),
                    has_border: r.border_color.is_some(),
                    screen_px_range: 0.0,
                    glyph: [0.0; 4],
                }
                .write_to(&mut out);
            }
            DrawCommand::Glyph(g) => {
                let text = flat.text(g.text);
                for prim in glyph_prims(g, text) {
                    prim.write_to(&mut out);
                }
            }
            DrawCommand::PushLayer(_) | DrawCommand::PopLayer => continue,
        }
    }
    out
}

/// Process-wide ASCII SDF atlas, baked once with the same parameters the CPU
/// reference uses (`bake_ascii(48.0, 6, 12.0)`). `None` when the embedded
/// vector face is unavailable.
fn atlas_sdf() -> Option<&'static prism_ui_font::sdf::SdfAtlas> {
    use std::sync::OnceLock;
    static ATLAS: OnceLock<Option<prism_ui_font::sdf::SdfAtlas>> = OnceLock::new();
    ATLAS
        .get_or_init(|| prism_ui_font::sdf::bake_ascii(48.0, 6, 12.0))
        .as_ref()
}

/// Expands a text run into per-glyph SDF primitives, mirroring the CPU
/// reference `raster_glyph_sdf` layout exactly (monospace at `cmd.size`,
/// uniform fit into the box, vertically centred). Each primitive carries the
/// device-space cell rect plus the normalised atlas uv ramp so the shader can
/// sample the field per pixel. Falls back to a single opaque rect when the
/// atlas or metrics are unavailable, and emits a solid block for empty runs so
/// non-text callers keep their legacy appearance.
fn glyph_prims(g: GlyphCmd, text: &str) -> Vec<PrimRaw> {
    use prism_ui_font::vector;

    let block = |g: GlyphCmd| PrimRaw {
        rect: [
            g.rect.left(),
            g.rect.top(),
            g.rect.size.width,
            g.rect.size.height,
        ],
        fill: [g.color.r, g.color.g, g.color.b, g.color.a],
        border: [0.0; 4],
        params: [0.0, 0.0, 0.0, g.opacity],
        offset: [0.0, 0.0],
        kind: 1.0,
        has_fill: true,
        has_border: false,
        screen_px_range: 0.0,
        glyph: [0.0; 4],
    };

    if text.is_empty() {
        return alloc::vec![block(g)];
    }
    let Some(atlas) = atlas_sdf() else {
        return alloc::vec![block(g)];
    };
    let box_w = g.rect.size.width.max(0.0);
    let box_h = g.rect.size.height.max(0.0);
    if box_w <= 0.0 || box_h <= 0.0 {
        return Vec::new();
    }
    let base_px = g.size.max(1.0);
    let Some(m) = vector::metrics(base_px) else {
        return alloc::vec![block(g)];
    };

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len() as f32;
    let adv = m.advance.max(0.0);
    let line_h = m.line_height().max(1.0);

    let mut scale = 1.0f32;
    if line_h > box_h {
        scale = scale.min(box_h / line_h);
    }
    let run_w = n * adv;
    if run_w * scale > box_w && run_w > 0.0 {
        scale = scale.min(box_w / run_w);
    }
    if scale <= 0.0 {
        return Vec::new();
    }

    let px = base_px * scale;
    let adv_s = adv * scale;
    let ascent_s = m.ascent * scale;
    let text_w = n * adv_s;
    let text_h = line_h * scale;
    let origin_x = g.rect.left() + ((box_w - text_w) * 0.5).max(0.0);
    let top_y = g.rect.top() + ((box_h - text_h) * 0.5).max(0.0);
    let baseline_y = top_y + ascent_s;

    let em = atlas.bake_em.max(1.0);
    let dev_per_texel = px / em;
    if dev_per_texel <= 0.0 {
        return Vec::new();
    }
    let screen_px_range = atlas.px_range * (px / em);
    let aw = atlas.width.max(1) as f32;
    let ah = atlas.height.max(1) as f32;
    let sstep = 1.0 / (dev_per_texel * aw);
    let tstep = 1.0 / (dev_per_texel * ah);

    let mut prims = Vec::with_capacity(chars.len());
    for (i, ch) in chars.iter().enumerate() {
        let pen_x = origin_x + i as f32 * adv_s;
        let Some(gl) = atlas.glyph(*ch) else {
            continue; // space / unbaked: no ink
        };
        let cell_left = pen_x + gl.bearing_em.0 * px;
        let cell_top = baseline_y + gl.bearing_em.1 * px;
        let cell_w_dev = gl.size_em.0 * px;
        let cell_h_dev = gl.size_em.1 * px;
        if cell_w_dev <= 0.0 || cell_h_dev <= 0.0 {
            continue;
        }
        let su0 = (gl.px_min.0 as f32 + 0.5) / aw;
        let sv0 = (gl.px_min.1 as f32 + 0.5) / ah;
        prims.push(PrimRaw {
            rect: [cell_left, cell_top, cell_w_dev, cell_h_dev],
            fill: [g.color.r, g.color.g, g.color.b, g.color.a],
            border: [0.0; 4],
            params: [0.0, 0.0, 0.0, g.opacity],
            offset: [0.0, 0.0],
            kind: 2.0,
            has_fill: true,
            has_border: false,
            screen_px_range,
            glyph: [su0, sv0, sstep, tstep],
        });
    }
    prims
}

fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Bind-group layout entry for the SDF atlas as a filterable 2-D texture.
fn texture_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Bind-group layout entry for the atlas's linear (filtering) sampler.
fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

/// Uploads the process-wide SDF atlas as an `R8Unorm` texture and returns a
/// view plus a linear, clamp-to-edge sampler. Hardware normalises `u8 -> f32`
/// in `0.0..=1.0` and filters linearly, exactly matching the CPU
/// [`crate::sdf::SdfAtlas::sample_distance`] bilinear read. When the atlas is
/// unavailable a `1x1` mid-grey (edge) texel is uploaded so the binding stays
/// valid and glyph fallback blocks still render.
fn build_atlas_texture(device: &Device, queue: &Queue) -> (TextureView, Sampler) {
    let (w, h, data) = match atlas_sdf() {
        Some(a) => (a.width.max(1), a.height.max(1), a.data.clone()),
        None => (1u32, 1u32, alloc::vec![128u8]),
    };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("prism_ui_gpu_sdf_atlas"),
        size: Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &data,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w),
            rows_per_image: Some(h),
        },
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&TextureViewDescriptor::default());
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism_ui_gpu_sdf_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::TextRef;
    use prism_ui_layout::{Point, Rect, Size};
    use prism_ui_style::Color;

    /// Builds a glyph command whose box fits a short run at `size` px. The text
    /// pool handle is unused by `glyph_prims` (the run is passed separately), so
    /// an empty ref is fine for the encoding math.
    fn glyph_cmd(w: f32, h: f32, size: f32) -> GlyphCmd {
        GlyphCmd {
            rect: Rect::new(Point::new(10.0, 20.0), Size::new(w, h)),
            color: Color::rgba(0.1, 0.2, 0.3, 1.0),
            size,
            opacity: 1.0,
            text: TextRef::EMPTY,
        }
    }

    #[test]
    fn empty_run_emits_single_block() {
        let prims = glyph_prims(glyph_cmd(40.0, 16.0, 14.0), "");
        assert_eq!(prims.len(), 1);
        assert_eq!(prims[0].kind, 1.0, "empty run must stay a solid block");
    }

    #[test]
    fn text_run_expands_to_sdf_glyphs() {
        // Requires the embedded vector face (present under the gpu->std->vector
        // feature chain). If the atlas is somehow unavailable the run falls back
        // to a single block, which we treat as a skip rather than a failure.
        let prims = glyph_prims(glyph_cmd(60.0, 24.0, 18.0), "Hi");
        if prims.len() == 1 && prims[0].kind == 1.0 {
            return; // atlas unavailable in this build
        }
        // 'H' and 'i' are both inked ASCII glyphs.
        assert_eq!(prims.len(), 2, "two inked glyphs expected");
        for p in &prims {
            assert_eq!(p.kind, 2.0, "glyph prims use the SDF kind");
            assert!(p.screen_px_range > 0.0, "screenPxRange must be positive");
            assert!(p.rect[2] > 0.0 && p.rect[3] > 0.0, "cell must have area");
            let (su0, sv0, sstep, tstep) = (p.glyph[0], p.glyph[1], p.glyph[2], p.glyph[3]);
            assert!((0.0..=1.0).contains(&su0), "su0 in [0,1], got {su0}");
            assert!((0.0..=1.0).contains(&sv0), "sv0 in [0,1], got {sv0}");
            assert!(sstep > 0.0 && tstep > 0.0, "uv ramp steps must be positive");
            // The far corner of the cell must still land inside the atlas.
            let su1 = su0 + p.rect[2] * sstep;
            let sv1 = sv0 + p.rect[3] * tstep;
            assert!(su1 <= 1.0 + 1e-3, "u ramp overshoots atlas: {su1}");
            assert!(sv1 <= 1.0 + 1e-3, "v ramp overshoots atlas: {sv1}");
        }
    }
}
