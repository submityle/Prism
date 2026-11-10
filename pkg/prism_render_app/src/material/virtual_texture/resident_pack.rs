//! Bounded, byte-exact resident pages -> copy-aligned VT atlas tile.
//!
//! No source payload is mapped or read back. The caller must submit consumers
//! on the context's queue after `pack`; device recovery requires a new packer.
//! This helper neither publishes residency nor owns a global live-VRAM budget.

use std::panic::{catch_unwind, AssertUnwindSafe};

use bytemuck::{Pod, Zeroable};
use prism_gdeflate_gpu::{GpuContext, ResidentDeflateTile};

const WORKGROUP_SIZE: u32 = 64;
pub(super) const PARAM_BYTES: u64 = size_of::<Params>() as u64;

/// Byte offset and tight cropped dimensions in the assembled resident input.
/// Entries are 3x3 row-major: index 4 is the mandatory center page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ResidentPageRegion {
    pub offset: u32,
    pub width: u32,
    pub height: u32,
}

/// A fresh square staging tile; `extent` is its edge length in texels.
#[derive(Debug)]
pub(super) struct PackedResidentTile {
    pub buffer: wgpu::Buffer,
    pub bytes_per_row: u32,
    pub extent: u32,
}

/// Admission/resource failures never return a partially usable output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResidentPackError {
    InvalidLayout,
    MissingCenter,
    InvalidRegion { index: usize },
    DeviceMismatch,
    SizeOverflow,
    DeviceLimit,
    AllocationBudgetExceeded { required: u64, limit: u64 },
    DeviceResource,
}

impl core::fmt::Display for ResidentPackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident VT pack: {self:?}")
    }
}

impl std::error::Error for ResidentPackError {}

// WGSL: eight scalar words followed by nine vec4<u32> entries. No host heap
// allocation is needed to prepare the uniform; absent entries are all zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    page_size: u32,
    gutter: u32,
    bytes_per_pixel: u32,
    extent: u32,
    bytes_per_row: u32,
    word_count: u32,
    padding: [u32; 2],
    regions: [[u32; 4]; 9],
}

const _: () = assert!(size_of::<Params>() == 176);

pub(super) struct Admission {
    params: Params,
    output_bytes: u64,
    workgroups: u32,
}

/// A cached pipeline bound to the exact original instance/logical device.
pub(super) struct ResidentAtlasPacker {
    instance: wgpu::Instance,
    device: wgpu::Device,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl ResidentAtlasPacker {
    pub(super) fn new(ctx: &GpuContext) -> Result<Self, ResidentPackError> {
        validate_compute_limits(&ctx.device().limits())?;
        device_scope(ctx.device(), || {
            let device = ctx.device();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("prism_vt_resident_pack"),
                source: wgpu::ShaderSource::Wgsl(include_str!("resident_pack.wgsl").into()),
            });
            let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("prism_vt_resident_pack_layout"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("prism_vt_resident_pack_pipeline_layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prism_vt_resident_pack_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            Ok(Self {
                instance: ctx.instance().clone(),
                device: device.clone(),
                layout,
                pipeline,
            })
        })
    }

    /// Packs tight cropped source pages without float/sRGB conversions.
    ///
    /// `max_device_bytes` bounds newly allocated output + 176-byte uniform
    /// descriptor sizes, excluding retained input, cached pipeline and driver
    /// overhead. All admission checks precede device allocation/submission.
    /// Input offsets may be unaligned; bytes outside logical input are never
    /// sampled. Output rows are 256-aligned and every padding byte is zero.
    ///
    /// No GPU completion wait or readback occurs. Scoped wgpu errors (including
    /// submit errors) are consumed before success; later asynchronous loss is
    /// the caller's recovery concern. A failed submission returns no resource,
    /// and a fresh allocation on every call prevents failed-range reuse.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pack(
        &self,
        ctx: &GpuContext,
        input: &ResidentDeflateTile,
        regions: &[Option<ResidentPageRegion>; 9],
        page_size: u32,
        gutter: u32,
        bytes_per_pixel: u32,
        max_device_bytes: u64,
    ) -> Result<PackedResidentTile, ResidentPackError> {
        if ctx.instance() != &self.instance
            || ctx.device() != &self.device
            || !input.is_on_context(ctx)
        {
            return Err(ResidentPackError::DeviceMismatch);
        }
        let admission = admit(
            &ctx.device().limits(),
            input.byte_len(),
            input.buffer().size(),
            regions,
            page_size,
            gutter,
            bytes_per_pixel,
            max_device_bytes,
        )?;
        if !input.buffer().usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(ResidentPackError::InvalidLayout);
        }
        device_scope(ctx.device(), || {
            let device = ctx.device();
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("prism_vt_resident_pack_output"),
                size: admission.output_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // Only metadata is mapped, never input/output payload. Device
            // allocation/mapping failures are contained by device_scope.
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("prism_vt_resident_pack_params"),
                size: PARAM_BYTES,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            });
            {
                let mut mapped = uniform
                    .slice(..)
                    .get_mapped_range_mut()
                    .map_err(|_| ResidentPackError::DeviceResource)?;
                mapped.copy_from_slice(bytemuck::bytes_of(&admission.params));
            }
            uniform.unmap();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prism_vt_resident_pack_bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prism_vt_resident_pack_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prism_vt_resident_pack_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(admission.workgroups, 1, 1);
            }
            ctx.queue().submit([encoder.finish()]);
            Ok(PackedResidentTile {
                buffer: output,
                bytes_per_row: admission.params.bytes_per_row,
                extent: admission.params.extent,
            })
        })
    }
}

#[allow(clippy::too_many_arguments)]
// Also used by the uploader's whole-stage preflight, before any source decode.
pub(super) fn admit(
    limits: &wgpu::Limits,
    input_bytes: u64,
    input_buffer_bytes: u64,
    regions: &[Option<ResidentPageRegion>; 9],
    page_size: u32,
    gutter: u32,
    bytes_per_pixel: u32,
    max_device_bytes: u64,
) -> Result<Admission, ResidentPackError> {
    use ResidentPackError::{DeviceLimit, InvalidLayout, SizeOverflow};
    validate_compute_limits(limits)?;
    if page_size == 0 || !matches!(bytes_per_pixel, 1 | 2 | 4 | 8) {
        return Err(InvalidLayout);
    }
    if regions[4].is_none() {
        return Err(ResidentPackError::MissingCenter);
    }
    let extent = gutter
        .checked_mul(2)
        .and_then(|border| page_size.checked_add(border))
        .ok_or(SizeOverflow)?;
    // Shader uses signed centered coordinates and u32 byte addresses.
    if extent > i32::MAX as u32 || extent > limits.max_texture_dimension_2d {
        return Err(DeviceLimit);
    }
    let row_bytes = extent.checked_mul(bytes_per_pixel).ok_or(SizeOverflow)?;
    let bytes_per_row = row_bytes.checked_add(255).ok_or(SizeOverflow)? & !255;
    let output_bytes = u64::from(bytes_per_row)
        .checked_mul(u64::from(extent))
        .ok_or(SizeOverflow)?;
    let aligned_input = input_bytes.checked_add(3).ok_or(SizeOverflow)? & !3;
    if input_bytes == 0
        || input_bytes > u64::from(u32::MAX)
        || input_buffer_bytes < aligned_input
        || input_buffer_bytes % 4 != 0
    {
        return Err(InvalidLayout);
    }
    if input_buffer_bytes > limits.max_buffer_size
        || input_buffer_bytes > limits.max_storage_buffer_binding_size
        || output_bytes > limits.max_buffer_size
        || output_bytes > limits.max_storage_buffer_binding_size
        || output_bytes > u64::from(u32::MAX)
    {
        return Err(DeviceLimit);
    }
    let mut encoded_regions = [[0; 4]; 9];
    for (index, region) in regions.iter().enumerate() {
        let Some(region) = region else { continue };
        if region.width == 0
            || region.height == 0
            || region.width > page_size
            || region.height > page_size
        {
            return Err(ResidentPackError::InvalidRegion { index });
        }
        let end = u64::from(region.width)
            .checked_mul(u64::from(region.height))
            .and_then(|bytes| bytes.checked_mul(u64::from(bytes_per_pixel)))
            .and_then(|bytes| bytes.checked_add(u64::from(region.offset)))
            .ok_or(SizeOverflow)?;
        if end > input_bytes {
            return Err(ResidentPackError::InvalidRegion { index });
        }
        encoded_regions[index] = [region.offset, region.width, region.height, 1];
    }
    let word_count = u32::try_from(output_bytes / 4).map_err(|_| DeviceLimit)?;
    let workgroups = word_count.div_ceil(WORKGROUP_SIZE);
    if workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(DeviceLimit);
    }
    let required = output_bytes.checked_add(PARAM_BYTES).ok_or(SizeOverflow)?;
    if required > max_device_bytes {
        return Err(ResidentPackError::AllocationBudgetExceeded {
            required,
            limit: max_device_bytes,
        });
    }
    Ok(Admission {
        params: Params {
            page_size,
            gutter,
            bytes_per_pixel,
            extent,
            bytes_per_row,
            word_count,
            padding: [0; 2],
            regions: encoded_regions,
        },
        output_bytes,
        workgroups,
    })
}

fn validate_compute_limits(limits: &wgpu::Limits) -> Result<(), ResidentPackError> {
    if limits.max_compute_workgroup_size_x < WORKGROUP_SIZE
        || limits.max_compute_workgroup_size_y == 0
        || limits.max_compute_workgroup_size_z == 0
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE
        || limits.max_compute_workgroups_per_dimension == 0
        || limits.max_storage_buffers_per_shader_stage < 2
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_bind_groups < 1
        || limits.max_bindings_per_bind_group < 3
        || limits.max_storage_buffer_binding_size < 4
        || limits.max_uniform_buffer_binding_size < PARAM_BYTES
        || limits.max_buffer_size < PARAM_BYTES
    {
        return Err(ResidentPackError::DeviceLimit);
    }
    Ok(())
}

// Matches page_resident.rs: always consume all scope handles, even on panic.
// Resolving these error futures is not polling/waiting for GPU completion.
fn device_scope<T>(
    device: &wgpu::Device,
    operation: impl FnOnce() -> Result<T, ResidentPackError>,
) -> Result<T, ResidentPackError> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result =
        catch_unwind(AssertUnwindSafe(operation)).unwrap_or(Err(ResidentPackError::DeviceResource));
    let validation = futures_lite::future::block_on(validation.pop());
    let internal = futures_lite::future::block_on(internal.pop());
    let oom = futures_lite::future::block_on(oom.pop());
    if validation.is_some() || internal.is_some() || oom.is_some() {
        Err(ResidentPackError::DeviceResource)
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use prism_gdeflate_gpu::{
        buffer, DeflateBatchLimits, DeflateTile, GpuDeflate, GpuProbeError, GROUP, LANES, WORD,
    };
    use prism_render_architecture::texture_streaming::TexturePageKey;
    use prism_render_material::{
        build_material_vt_atlas_tile, MaterialVtAtlasConfig, MaterialVtAtlasTile, MaterialVtFormat,
        MaterialVtSourceDescriptor, MaterialVtSourceLayout, MaterialVtSourceSnapshot,
    };

    use super::*;

    #[expect(
        clippy::print_stderr,
        reason = "record hardware evidence or an explicit skip"
    )]
    fn parity_gpu_context() -> Option<GpuContext> {
        let required = std::env::var_os("PRISM_REQUIRE_GPU").is_some()
            || std::env::var_os("REQUIRE_GPU").is_some();
        let ctx = match GpuContext::probe() {
            Ok(ctx) => ctx,
            Err(GpuProbeError::NoAdapter(error)) => {
                assert!(!required, "resident pack requires a GPU: {error}");
                eprintln!("SKIPPED resident pack parity: no adapter: {error}");
                return None;
            }
            Err(error) => panic!("resident pack GPU acquisition failed: {error}"),
        };
        let info = ctx.adapter_info();
        let hardware = matches!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        );
        assert!(!required || hardware, "physical GPU required: {info:?}");
        if !hardware {
            eprintln!("SKIPPED resident pack parity: non-hardware adapter: {info:?}");
            return None;
        }
        eprintln!("resident pack REAL HARDWARE GPU: {info:?}");
        Some(ctx)
    }

    fn parity_source(format: MaterialVtFormat) -> MaterialVtSourceSnapshot {
        // A non-power-of-two logical page and source: the interior center has
        // eight neighbors, including 3-wide / 2-high cropped edge pages.
        let descriptor = MaterialVtSourceDescriptor::new(71, 13, 12, 1, 5, 4, format).unwrap();
        let len =
            usize::try_from(MaterialVtSourceLayout::new(descriptor).unwrap().byte_len()).unwrap();
        // Include signed zero, subnormals, infinities and NaN payload bits in
        // the half-float case. These are bytes, never numerical conversions.
        let half_bits = [
            0x0000u16, 0x8000, 0x0001, 0x03ff, 0x7c00, 0xfc00, 0x7e35, 0x7c01,
        ];
        let bytes: Vec<u8> = (0..len)
            .map(|index| {
                if format == MaterialVtFormat::Rgba16Float {
                    half_bits[index / 2 % half_bits.len()].to_le_bytes()[index % 2]
                        ^ u8::try_from((index / 16) & 0x3f).unwrap()
                } else {
                    u8::try_from(index & 255).unwrap().wrapping_mul(37)
                        ^ u8::try_from((index >> 8) & 255).unwrap().wrapping_add(91)
                }
            })
            .collect();
        MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap()
    }

    fn parity_pages(
        source: &MaterialVtSourceSnapshot,
        key: TexturePageKey,
        mask: u16,
    ) -> (
        Vec<u8>,
        [Option<ResidentPageRegion>; 9],
        BTreeSet<TexturePageKey>,
    ) {
        // A guard prefix makes even 4/8-byte pixels start at unaligned byte
        // offsets. Pages themselves are concatenated tight, without row gaps.
        let mut bytes = vec![0xd3, 0x79, 0xa6];
        let mut regions = [None; 9];
        let mut resident = BTreeSet::new();
        for (index, region) in regions.iter_mut().enumerate() {
            if index != 4 && mask & (1 << index) == 0 {
                continue;
            }
            let x = i32::from(key.x) + i32::try_from(index % 3).unwrap() - 1;
            let y = i32::from(key.y) + i32::try_from(index / 3).unwrap() - 1;
            let (Ok(x), Ok(y)) = (u16::try_from(x), u16::try_from(y)) else {
                continue;
            };
            let neighbor = TexturePageKey { x, y, ..key };
            let Ok(page) = source.page(neighbor) else {
                continue;
            };
            let metadata = page.metadata();
            *region = Some(ResidentPageRegion {
                offset: u32::try_from(bytes.len()).unwrap(),
                width: metadata.extent_width,
                height: metadata.extent_height,
            });
            for row in page.rows() {
                bytes.extend_from_slice(row.bytes);
            }
            if index != 4 {
                resident.insert(neighbor);
            }
        }
        assert!(regions[4].is_some());
        bytes.extend_from_slice(&[0x51, 0xe9, 0x23]);
        (bytes, regions, resident)
    }

    fn parity_decode(ctx: &GpuContext, decoder: &GpuDeflate, bytes: &[u8]) -> ResidentDeflateTile {
        // Independent RFC 1951 stored block, not a CPU inflate or fallback.
        // The real GPU decoder creates the authenticated resident input; only
        // its status is read back, never the decoded source payload.
        let len = u16::try_from(bytes.len()).unwrap();
        let mut raw = vec![1]; // BFINAL=1, BTYPE=00, then byte alignment.
        raw.extend_from_slice(&len.to_le_bytes());
        raw.extend_from_slice(&(!len).to_le_bytes());
        raw.extend_from_slice(bytes);
        let mut interleaved = vec![0; raw.len().div_ceil(GROUP) * GROUP];
        let rounds = interleaved.len() / GROUP;
        for (index, &byte) in raw.iter().enumerate() {
            let word = index / WORD;
            let destination =
                (word % LANES as usize * rounds + word / LANES as usize) * WORD + index % WORD;
            interleaved[destination] = byte;
        }
        let mut tiles = decoder
            .decompress_batch_resident(
                ctx,
                &[DeflateTile {
                    interleaved: &interleaved,
                    compressed_size: raw.len(),
                    expected_output_size: bytes.len(),
                }],
                DeflateBatchLimits {
                    max_tiles: 1,
                    max_device_bytes: 4 << 20,
                },
            )
            .expect("real GPU stored-block decode, without CPU fallback");
        assert_eq!(tiles.len(), 1);
        tiles.pop().unwrap()
    }

    fn parity_readback(
        ctx: &GpuContext,
        outputs: Vec<(PackedResidentTile, MaterialVtAtlasTile)>,
        format: MaterialVtFormat,
        mask: u16,
    ) {
        let total: u64 = outputs.iter().map(|(gpu, _)| gpu.buffer.size()).sum();
        let staging = buffer::staging(ctx.device(), "resident_pack_parity_readback", total);
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        let mut offset = 0;
        for (gpu, golden) in &outputs {
            assert_eq!(gpu.extent, golden.layout.stored_tile_extent_px());
            assert_eq!(gpu.bytes_per_row, golden.bytes_per_row);
            assert_eq!(gpu.bytes_per_row % 256, 0);
            assert_eq!(gpu.buffer.size(), golden.staging_bytes.len() as u64);
            assert_eq!(
                gpu.buffer.usage(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            );
            encoder.copy_buffer_to_buffer(&gpu.buffer, 0, &staging, offset, gpu.buffer.size());
            offset += gpu.buffer.size();
        }
        // A downstream consumer on the SAME queue, after pack's submissions.
        // Mapping and completion waits exist only in this test consumer.
        let submission = ctx.queue().submit([encoder.finish()]);
        let actual: Vec<u8> = buffer::read_back::<u32>(ctx, &staging, submission)
            .expect("resident pack test-only readback")
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(actual.len() as u64, total);
        let mut offset = 0;
        for (gpu, golden) in outputs {
            let size = golden.staging_bytes.len();
            let tile = &actual[offset..offset + size];
            let pitch = gpu.bytes_per_row as usize;
            let row_bytes = (gpu.extent * format.bytes_per_texel()) as usize;
            for (y, (row, expected)) in tile
                .chunks_exact(pitch)
                .zip(golden.staging_bytes.chunks_exact(pitch))
                .enumerate()
            {
                // Compare padding too, with a compact first-difference report.
                if let Some(x) = row.iter().zip(expected).position(|(a, b)| a != b) {
                    panic!(
                        "pack mismatch: format={format:?} key={:?} mask={mask:#05x} gutter={} row={y} byte={x}: GPU={} CPU={}",
                        golden.key, golden.layout.gutter_px(), row[x], expected[x],
                    );
                }
                assert!(
                    row[row_bytes..].iter().all(|&byte| byte == 0),
                    "nonzero row padding"
                );
            }
            offset += size;
        }
    }

    /// Real-device byte parity, including every byte of copy-row padding.
    /// PRISM_REQUIRE_GPU=1 (or REQUIRE_GPU=1) forbids hardware/adapter skips.
    #[test]
    fn resident_pack_gpu_byte_exact_matches_cpu_atlas() {
        let Some(ctx) = parity_gpu_context() else {
            return;
        };
        let decoder = GpuDeflate::new(&ctx);
        let packer = ResidentAtlasPacker::new(&ctx).expect("cached GPU pack pipeline");
        let mut comparisons = 0;
        for format in [
            MaterialVtFormat::R8Unorm,
            MaterialVtFormat::Rg8Unorm,
            MaterialVtFormat::Rgba8Unorm,
            MaterialVtFormat::Rgba8Srgb,
            MaterialVtFormat::Rgba16Float,
        ] {
            let source = parity_source(format);
            // Interior with all eight valid neighbors, cropped bottom-right
            // center, and top-left center with out-of-bounds negative neighbors.
            for (x, y) in [(1, 1), (2, 2), (0, 0)] {
                let key = TexturePageKey {
                    texture: 71,
                    mip: 0,
                    layer: 0,
                    x,
                    y,
                };
                // None, all, and complementary mixtures of axial/diagonal
                // neighbors. Source pages exist even when deliberately absent.
                for mask in [0, 0x1ff, 0x155, 0x0aa] {
                    let (bytes, regions, resident) = parity_pages(&source, key, mask);
                    if (x, y) == (1, 1) && mask == 0x1ff {
                        assert!(regions.iter().all(Option::is_some));
                        assert_eq!(regions[8].unwrap().width, 3);
                        assert_eq!(regions[8].unwrap().height, 2);
                    }
                    if (x, y) == (2, 2) {
                        assert_eq!(
                            (regions[4].unwrap().width, regions[4].unwrap().height),
                            (3, 2)
                        );
                    }
                    let input = parity_decode(&ctx, &decoder, &bytes);
                    let mut outputs = Vec::new();
                    // 17 > 2 * page_size exercises negative remainders and
                    // immediate-neighbor selection far beyond a single page.
                    // For bpp=8 it also crosses the 256 -> 512 row-pitch step.
                    for gutter in [0, 2, 17] {
                        let golden = build_material_vt_atlas_tile(
                            &source,
                            key,
                            0,
                            &resident,
                            MaterialVtAtlasConfig::new(gutter, 1).unwrap(),
                        )
                        .expect("production CPU atlas golden");
                        let required = golden.staging_bytes.len() as u64 + PARAM_BYTES;
                        let output = packer
                            .pack(
                                &ctx,
                                &input,
                                &regions,
                                5,
                                gutter,
                                format.bytes_per_texel(),
                                required,
                            )
                            .expect("bounded GPU byte pack");
                        outputs.push((output, golden));
                        comparisons += 1;
                    }
                    // Prove submitted consumers do not borrow the input tile.
                    drop(input);
                    parity_readback(&ctx, outputs, format, mask);
                }
            }
        }
        assert_eq!(comparisons, 180);
    }

    fn center(width: u32, height: u32, offset: u32) -> [Option<ResidentPageRegion>; 9] {
        let mut regions = [None; 9];
        regions[4] = Some(ResidentPageRegion {
            offset,
            width,
            height,
        });
        regions
    }

    #[test]
    fn all_supported_byte_widths_have_exact_aligned_budget() {
        for bpp in [1, 2, 4, 8] {
            let regions = center(3, 2, 1);
            let input_len = 1 + 3 * 2 * u64::from(bpp);
            let required = 256 * 7 + PARAM_BYTES;
            let plan = admit(
                &wgpu::Limits::default(),
                input_len,
                input_len.div_ceil(4) * 4,
                &regions,
                3,
                2,
                bpp,
                required,
            )
            .unwrap();
            assert_eq!(plan.params.extent, 7);
            assert_eq!(plan.params.bytes_per_row, 256);
            assert_eq!(plan.output_bytes, 256 * 7);
            assert_eq!(plan.params.regions[4], [1, 3, 2, 1]);
            assert_eq!(plan.params.regions[0], [0; 4]);
            assert_eq!(plan.workgroups, 7);
            assert_eq!(
                admit(
                    &wgpu::Limits::default(),
                    input_len,
                    input_len.div_ceil(4) * 4,
                    &regions,
                    3,
                    2,
                    bpp,
                    required - 1
                )
                .err(),
                Some(ResidentPackError::AllocationBudgetExceeded {
                    required,
                    limit: required - 1
                })
            );
        }
    }

    #[test]
    fn missing_center_and_invalid_formats_are_rejected() {
        assert_eq!(
            admit(
                &wgpu::Limits::default(),
                4,
                4,
                &[None; 9],
                1,
                0,
                1,
                u64::MAX
            )
            .err(),
            Some(ResidentPackError::MissingCenter)
        );
        for (page, bpp) in [(0, 1), (1, 0), (1, 3), (1, 16)] {
            assert_eq!(
                admit(
                    &wgpu::Limits::default(),
                    4,
                    4,
                    &center(1, 1, 0),
                    page,
                    0,
                    bpp,
                    u64::MAX
                )
                .err(),
                Some(ResidentPackError::InvalidLayout)
            );
        }
    }

    #[test]
    fn every_present_region_is_validated_against_logical_not_padded_input() {
        for index in 0..9 {
            for region in [
                ResidentPageRegion {
                    offset: 3,
                    width: 1,
                    height: 1,
                },
                ResidentPageRegion {
                    offset: 0,
                    width: 0,
                    height: 1,
                },
                ResidentPageRegion {
                    offset: 0,
                    width: 3,
                    height: 1,
                },
                ResidentPageRegion {
                    offset: 0,
                    width: 1,
                    height: 3,
                },
            ] {
                let mut regions = center(1, 1, 0);
                regions[index] = Some(region);
                assert_eq!(
                    admit(&wgpu::Limits::default(), 3, 4, &regions, 2, 0, 1, u64::MAX).err(),
                    Some(ResidentPackError::InvalidRegion { index })
                );
            }
        }
    }

    #[test]
    fn large_gutters_keep_the_same_nine_region_abi() {
        let plan = admit(
            &wgpu::Limits::default(),
            1,
            4,
            &center(1, 1, 0),
            1,
            17,
            1,
            u64::MAX,
        )
        .unwrap();
        assert_eq!(plan.params.extent, 35);
        assert_eq!(plan.params.regions[4], [0, 1, 1, 1]);
        assert_eq!(plan.params.regions.len(), 9);
    }

    #[test]
    fn arithmetic_input_binding_and_dispatch_limits_are_checked() {
        let regions = center(1, 1, 0);
        let limits = wgpu::Limits::default();
        assert_eq!(
            admit(&limits, 1, 4, &regions, 1, u32::MAX, 1, u64::MAX).err(),
            Some(ResidentPackError::SizeOverflow)
        );
        for (logical, allocated) in [(0, 4), (5, 4), (1, 3)] {
            assert_eq!(
                admit(&limits, logical, allocated, &regions, 1, 0, 1, u64::MAX).err(),
                Some(ResidentPackError::InvalidLayout)
            );
        }
        let small_binding = wgpu::Limits {
            max_storage_buffer_binding_size: 255,
            ..limits.clone()
        };
        assert_eq!(
            admit(&small_binding, 1, 4, &regions, 1, 0, 1, u64::MAX).err(),
            Some(ResidentPackError::DeviceLimit)
        );
        let one_group = wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..limits.clone()
        };
        assert_eq!(
            admit(&one_group, 1, 4, &regions, 1, 1, 1, u64::MAX).err(),
            Some(ResidentPackError::DeviceLimit)
        );
        let small_uniform = wgpu::Limits {
            max_uniform_buffer_binding_size: PARAM_BYTES - 1,
            ..limits
        };
        assert_eq!(
            admit(&small_uniform, 1, 4, &regions, 1, 0, 1, u64::MAX).err(),
            Some(ResidentPackError::DeviceLimit)
        );
    }
}
