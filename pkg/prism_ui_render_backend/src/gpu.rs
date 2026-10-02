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
    Adapter, BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::draw::{DrawCommand, DrawList};
use crate::layer::LayerTree;
use crate::raster::Framebuffer;

/// Number of `f32` lanes per encoded primitive (six `vec4<f32>` = 96 bytes).
const PRIM_STRIDE: usize = 24;
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
    offset: vec4<f32>,  // offset.x, offset.y, _, _
    flags:  vec4<f32>,  // kind(0=shadow,1=rect), has_fill, has_border, _
};

@group(0) @binding(0) var<storage, read>       prims:  array<Prim>;
@group(0) @binding(1) var<storage, read_write> out_px: array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             dims:   vec4<u32>; // w, h, count, _

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
        } else {
            // Rect (and glyph, lowered to a rect fill): fill then border ring.
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

        Some(GpuRasterizer {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            pipeline,
            layout,
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
}

impl PrimRaw {
    fn write_to(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.rect);
        out.extend_from_slice(&self.fill);
        out.extend_from_slice(&self.border);
        out.extend_from_slice(&self.params);
        out.extend_from_slice(&[self.offset[0], self.offset[1], 0.0, 0.0]);
        out.extend_from_slice(&[
            self.kind,
            if self.has_fill { 1.0 } else { 0.0 },
            if self.has_border { 1.0 } else { 0.0 },
            0.0,
        ]);
    }
}

/// Encodes the flattened draw list into the shader's `array<Prim>` layout.
///
/// Layer markers must already be removed by [`LayerTree::flatten`]; any that
/// remain are skipped. Glyph runs are lowered to an opaque rect fill, exactly
/// as [`crate::raster`] does, so the two backends shade them identically.
fn encode_prims(flat: &DrawList) -> Vec<f32> {
    let transparent = Color::rgba(0.0, 0.0, 0.0, 0.0);
    let mut out = Vec::new();
    for cmd in flat.commands() {
        let prim = match *cmd {
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
            },
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
                }
            }
            DrawCommand::Glyph(g) => PrimRaw {
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
            },
            DrawCommand::PushLayer(_) | DrawCommand::PopLayer => continue,
        };
        prim.write_to(&mut out);
    }
    out
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
