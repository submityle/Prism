//! Compressed Manifest pages -> GPU gather -> atlas -> qualified publication.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    panic::{catch_unwind, AssertUnwindSafe},
};

use prism_asset_container::ManifestReadError;
use prism_gdeflate_gpu::{
    DeflateBatchLimits, FixedDecodeError, GpuResidentAssembler, ResidentAssemblyError,
    ResidentAssemblyLimits, ResidentByteRange,
};
use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_driver_wgpu::WgpuDevice;
use prism_render_material::{
    build_material_vt_atlas_tile, MaterialVtAtlasConfig, MaterialVtAtlasTile,
    MaterialVtSourceLayout,
};
use prism_render_runtime::RenderRuntime;

use super::{
    gpu::{check_owner_runtime, GpuSlice},
    publication::{prepare_upload, Pending, PreparedUpload},
    resident_pack::{self, ResidentAtlasPacker, ResidentPackError, ResidentPageRegion},
    NativeMaterialVt, NativeVtConfig, NativeVtError, NativeVtResidency, NativeVtSources,
};
use crate::{material::TextureSlotHeap, ResidentRenderDevice, ResidentWgpuInstance};

/// Explicit admission for a complete stage. These are not global live-VRAM
/// limits: atlas slices, retained decoder inputs, pipelines and driver memory
/// are excluded. Decode and assembly descriptor limits apply per source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVtResidentLimits {
    pub decode: DeflateBatchLimits,
    pub assembly: ResidentAssemblyLimits,
    /// Maximum merged source row ranges for any one texture.
    pub max_source_ranges: usize,
    /// Aggregate CPU staging + padded gather outputs + packed tile outputs
    /// and their 176-byte uniforms. Admission precedes any decode or atlas write.
    /// Assembly uniforms are separately covered by `assembly`.
    pub max_upload_bytes: u64,
}

impl Default for NativeVtResidentLimits {
    fn default() -> Self {
        Self {
            decode: DeflateBatchLimits {
                max_tiles: 4096,
                max_device_bytes: 256 << 20,
            },
            assembly: ResidentAssemblyLimits {
                max_slices: 65536,
                max_device_bytes: 64 << 20,
            },
            max_source_ranges: 16384,
            max_upload_bytes: 128 << 20,
        }
    }
}

/// No failure publishes a new slice or changes material rows. A failure after
/// touching the inactive atlas quarantines it until `recreate_for_device`.
#[derive(Debug)]
pub enum NativeVtResidentError {
    Vt(NativeVtError),
    Manifest(ManifestReadError),
    Decoder(FixedDecodeError),
    Assembly(ResidentAssemblyError),
    SourceRangeLimit { requested: usize, limit: usize },
    UploadBudget { required: u64, limit: u64 },
    SizeOverflow,
    AllocationFailed,
    DeviceResource,
}

impl core::fmt::Display for NativeVtResidentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident VT upload: {self:?}")
    }
}
impl std::error::Error for NativeVtResidentError {}
impl From<NativeVtError> for NativeVtResidentError {
    fn from(error: NativeVtError) -> Self {
        Self::Vt(error)
    }
}

impl From<ResidentPackError> for NativeVtResidentError {
    fn from(error: ResidentPackError) -> Self {
        match error {
            ResidentPackError::DeviceMismatch => NativeVtError::DeviceMismatch.into(),
            ResidentPackError::SizeOverflow => Self::SizeOverflow,
            ResidentPackError::AllocationBudgetExceeded { required, limit } => {
                Self::UploadBudget { required, limit }
            }
            _ => Self::DeviceResource,
        }
    }
}

type AtlasPage = (TexturePageKey, u32);

struct GatherPlan {
    source_index: usize,
    pages: Vec<AtlasPage>,
    ranges: Vec<Range<u64>>,
    regions: BTreeMap<TexturePageKey, ResidentPageRegion>,
    byte_len: u64,
}

struct StagePlan {
    gathers: Vec<GatherPlan>,
    cpu: Vec<MaterialVtAtlasTile>,
}

/// Cached GPU pipelines authenticated to the renderer's original native
/// instance/device/queue. Reconstruct after a successful device recovery.
/// There is no decoded-payload readback or CPU fallback. Decoder status checks
/// remain synchronous; this is not a preemptible EDL or direct-storage queue.
pub struct NativeVtResidentUploader {
    shared: ResidentRenderDevice,
    assembler: GpuResidentAssembler,
    packer: ResidentAtlasPacker,
}

impl NativeVtResidentUploader {
    pub fn new(shared: ResidentRenderDevice) -> Result<Self, NativeVtResidentError> {
        let assembler =
            GpuResidentAssembler::new(shared.context()).map_err(NativeVtResidentError::Decoder)?;
        let packer =
            ResidentAtlasPacker::new(shared.context()).map_err(NativeVtResidentError::from)?;
        Ok(Self {
            shared,
            assembler,
            packer,
        })
    }

    /// Stage the complete desired set (CPU and resident sources may coexist).
    /// Use the ordinary `try_publish` and binding-lease API afterwards: only
    /// real queue completion plus unchanged source/heap tokens may publish.
    pub fn stage(
        &self,
        vt: &mut NativeMaterialVt<WgpuDevice>,
        runtime: &RenderRuntime<ResidentWgpuInstance>,
        sources: &NativeVtSources,
        heap: &TextureSlotHeap,
        residency: &NativeVtResidency,
        limits: NativeVtResidentLimits,
    ) -> Result<(), NativeVtResidentError> {
        check_owner_runtime(&vt.slices, runtime)?;
        let current =
            ResidentRenderDevice::from_runtime(runtime).ok_or(NativeVtError::DeviceMismatch)?;
        if current.context().instance() != self.shared.context().instance()
            || current.context().device() != self.shared.context().device()
            || current.context().queue() != self.shared.context().queue()
        {
            return Err(NativeVtError::DeviceMismatch.into());
        }
        if vt.pending.is_some() {
            return Err(NativeVtError::PendingUpload.into());
        }
        let index = vt.active.map_or(0, |i| 1 - i);
        if !GpuSlice::idle(&vt.slices[index]) {
            return Err(NativeVtError::SliceBusy.into());
        }
        let prepared = prepare_upload(vt.config, sources, heap, residency)?;
        let plan = plan_stage(vt.config, &prepared, limits, &self.shared)?;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(prepared.pages.len())
            .map_err(|_| NativeVtResidentError::AllocationFailed)?;
        // Build every input and packed tile before the first inactive write.
        // Failed decode/assembly/packing therefore leaves that slice reusable.
        for gather in &plan.gathers {
            let source = &prepared.sources[gather.source_index].source;
            let (asset, _) = source.resident().ok_or(NativeVtError::DeviceMismatch)?;
            let ranges = asset
                .read_ranges_resident(&gather.ranges, limits.decode)
                .map_err(NativeVtResidentError::Manifest)?;
            let slice_count = ranges.iter().try_fold(0usize, |n, range| {
                n.checked_add(range.slices().len())
                    .ok_or(NativeVtResidentError::SizeOverflow)
            })?;
            if slice_count > limits.assembly.max_slices {
                return Err(NativeVtResidentError::Assembly(
                    ResidentAssemblyError::TooManySlices {
                        requested: slice_count,
                        limit: limits.assembly.max_slices,
                    },
                ));
            }
            let mut slices = Vec::new();
            slices
                .try_reserve_exact(slice_count)
                .map_err(|_| NativeVtResidentError::AllocationFailed)?;
            for range in &ranges {
                for slice in range.slices() {
                    slices.push(ResidentByteRange {
                        tile: slice.tile(),
                        range: slice.buffer_range(),
                    });
                }
            }
            let input = self
                .assembler
                .assemble(self.shared.context(), &slices, limits.assembly)
                .map_err(NativeVtResidentError::Assembly)?;
            if input.byte_len() != gather.byte_len {
                return Err(NativeVtResidentError::SizeOverflow);
            }
            for &(key, slot) in &gather.pages {
                let regions = neighbor_regions(key, &gather.regions, &prepared.keys);
                let tile = self
                    .packer
                    .pack(
                        self.shared.context(),
                        &input,
                        &regions,
                        vt.config.page_size,
                        vt.config.gutter_px,
                        vt.config.format.bytes_per_texel(),
                        limits.max_upload_bytes,
                    )
                    .map_err(NativeVtResidentError::from)?;
                packed.push((slot, tile));
            }
        }
        let slice = &vt.slices[index];
        let device = self.shared.context().device();
        // Obtain RHI-owned native handles before acquiring the work obligation.
        let texture = self.shared.device().wgpu_texture(slice.texture());
        let table = self.shared.device().wgpu_buffer(slice.table());
        let config = self.shared.device().wgpu_buffer(slice.config());
        let geometry = vt.config.geometry()?;
        device_scope(device, || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prism_vt_resident_upload"),
            });
            for (slot, tile) in &packed {
                let placement = geometry.slot_placement(*slot);
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &tile.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(tile.bytes_per_row),
                            rows_per_image: Some(tile.extent),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: placement.origin_x_px,
                            y: placement.origin_y_px,
                            z: placement.layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: tile.extent,
                        height: tile.extent,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let commands = encoder.finish();
            // Nothing after this point may reuse the inactive slice on error:
            // an unwinding queue call is not proof that no work was enqueued.
            slice.begin_work();
            let queue = self.shared.context().queue();
            for tile in &plan.cpu {
                let placement = geometry.slot_placement(tile.slot);
                let extent = tile.layout.stored_tile_extent_px();
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: placement.origin_x_px,
                            y: placement.origin_y_px,
                            z: placement.layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tile.staging_bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(tile.bytes_per_row),
                        rows_per_image: Some(extent),
                    },
                    wgpu::Extent3d {
                        width: extent,
                        height: extent,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let empty = [0u32; 4];
            let words = if prepared.table.words().is_empty() {
                &empty[..]
            } else {
                prepared.table.words()
            };
            queue.write_buffer(&table, 0, bytemuck::cast_slice(words));
            queue.write_buffer(&config, 0, bytemuck::bytes_of(&prepared.config));
            queue.submit([commands]);
            Ok(())
        })?;
        let complete = slice.on_completion(self.shared.queue());
        vt.pending = Some(Pending {
            slice: index,
            sources: prepared.sources,
            complete,
        });
        Ok(())
    }
}

fn plan_stage(
    config: NativeVtConfig,
    prepared: &PreparedUpload,
    limits: NativeVtResidentLimits,
    shared: &ResidentRenderDevice,
) -> Result<StagePlan, NativeVtResidentError> {
    let mut gathers = Vec::new();
    gathers
        .try_reserve_exact(prepared.sources.len())
        .map_err(|_| NativeVtResidentError::AllocationFailed)?;
    let mut upload_bytes = 0u64;
    let row = u64::from(config.geometry()?.tile_extent_px())
        .checked_mul(u64::from(config.format.bytes_per_texel()))
        .ok_or(NativeVtResidentError::SizeOverflow)?;
    let pitch = row
        .checked_add(255)
        .ok_or(NativeVtResidentError::SizeOverflow)?
        / 256
        * 256;
    let tile_bytes = pitch
        .checked_mul(u64::from(config.geometry()?.tile_extent_px()))
        .ok_or(NativeVtResidentError::SizeOverflow)?;
    // Index only the requested pages, never atlas capacity. Preserve per-source
    // order: the gather regions and assembled byte stream must agree exactly.
    let mut pages_by_texture = group_pages(&prepared.pages)?;
    let mut cpu_pages = Vec::new();
    cpu_pages
        .try_reserve_exact(prepared.sources.len())
        .map_err(|_| NativeVtResidentError::AllocationFailed)?;
    for (source_index, resident) in prepared.sources.iter().enumerate() {
        let source = &resident.source;
        let texture = source.layout().descriptor().texture_id();
        let pages = pages_by_texture.remove(&texture).unwrap_or_default();
        let Some((asset, offset)) = source.resident() else {
            let bytes = tile_bytes
                .checked_mul(pages.len() as u64)
                .ok_or(NativeVtResidentError::SizeOverflow)?;
            add_budget(&mut upload_bytes, bytes, limits.max_upload_bytes)?;
            cpu_pages.push((source_index, pages));
            continue;
        };
        if asset.context().instance() != shared.context().instance()
            || asset.context().device() != shared.context().device()
            || asset.context().queue() != shared.context().queue()
        {
            return Err(NativeVtError::DeviceMismatch.into());
        }
        let gather = plan_gather(
            source_index,
            source.layout(),
            offset,
            asset.entry().raw_len,
            pages,
            limits.max_source_ranges,
        )?;
        let padded = admit_assembly(&gather, limits.assembly)?;
        for &(key, _) in &gather.pages {
            // Reuse the shipping packer's exact layout/device admission, not
            // a second estimate that could disagree after decode has begun.
            resident_pack::admit(
                &shared.context().device().limits(),
                gather.byte_len,
                padded,
                &neighbor_regions(key, &gather.regions, &prepared.keys),
                config.page_size,
                config.gutter_px,
                config.format.bytes_per_texel(),
                limits.max_upload_bytes,
            )?;
            add_budget(
                &mut upload_bytes,
                tile_bytes
                    .checked_add(resident_pack::PARAM_BYTES)
                    .ok_or(NativeVtResidentError::SizeOverflow)?,
                limits.max_upload_bytes,
            )?;
        }
        add_budget(&mut upload_bytes, padded, limits.max_upload_bytes)?;
        gathers.push(gather);
    }
    let mut cpu = Vec::new();
    cpu.try_reserve_exact(prepared.pages.len())
        .map_err(|_| NativeVtResidentError::AllocationFailed)?;
    let atlas = MaterialVtAtlasConfig::new(config.gutter_px, config.tiles_per_row)
        .map_err(NativeVtError::Upload)?;
    for (source_index, pages) in cpu_pages {
        let snapshot = prepared.sources[source_index]
            .source
            .cpu_snapshot()
            .ok_or(NativeVtError::DeviceMismatch)?;
        for (key, slot) in pages {
            cpu.push(
                build_material_vt_atlas_tile(snapshot, key, slot, &prepared.keys, atlas)
                    .map_err(NativeVtError::Upload)?,
            );
        }
    }
    Ok(StagePlan { gathers, cpu })
}

fn group_pages(
    pages: &[AtlasPage],
) -> Result<BTreeMap<u32, Vec<AtlasPage>>, NativeVtResidentError> {
    let mut groups = BTreeMap::<u32, Vec<AtlasPage>>::new();
    for &(key, slot) in pages {
        let group = groups.entry(key.texture).or_default();
        group
            .try_reserve(1)
            .map_err(|_| NativeVtResidentError::AllocationFailed)?;
        group.push((key, slot));
    }
    Ok(groups)
}

fn plan_gather(
    source_index: usize,
    layout: &MaterialVtSourceLayout,
    offset: u64,
    source_len: u64,
    pages: Vec<AtlasPage>,
    max_ranges: usize,
) -> Result<GatherPlan, NativeVtResidentError> {
    let mut gather = GatherPlan {
        source_index,
        pages,
        ranges: Vec::new(),
        regions: BTreeMap::new(),
        byte_len: 0,
    };
    for &(key, _) in &gather.pages {
        let metadata = layout.page_metadata(key).map_err(NativeVtError::Source)?;
        let region = ResidentPageRegion {
            offset: u32::try_from(gather.byte_len)
                .map_err(|_| NativeVtResidentError::SizeOverflow)?,
            width: metadata.extent_width,
            height: metadata.extent_height,
        };
        gather.byte_len = gather
            .byte_len
            .checked_add(metadata.extracted_byte_len)
            .filter(|n| *n <= u64::from(u32::MAX))
            .ok_or(NativeVtResidentError::SizeOverflow)?;
        gather.regions.insert(key, region);
        for row in 0..metadata.row_count {
            let start = offset
                .checked_add(metadata.first_row_offset_bytes)
                .and_then(|n| {
                    n.checked_add(u64::from(row) * u64::from(metadata.source_row_pitch_bytes))
                })
                .ok_or(NativeVtResidentError::SizeOverflow)?;
            let end = start
                .checked_add(u64::from(metadata.row_bytes))
                .filter(|end| *end <= source_len)
                .ok_or(NativeVtResidentError::SizeOverflow)?;
            if let Some(last) = gather.ranges.last_mut().filter(|r| r.end == start) {
                last.end = end;
            } else {
                let requested = gather
                    .ranges
                    .len()
                    .checked_add(1)
                    .ok_or(NativeVtResidentError::SizeOverflow)?;
                if requested > max_ranges {
                    return Err(NativeVtResidentError::SourceRangeLimit {
                        requested,
                        limit: max_ranges,
                    });
                }
                gather
                    .ranges
                    .try_reserve(1)
                    .map_err(|_| NativeVtResidentError::AllocationFailed)?;
                gather.ranges.push(start..end);
            }
        }
    }
    Ok(gather)
}

fn admit_assembly(
    gather: &GatherPlan,
    limits: ResidentAssemblyLimits,
) -> Result<u64, NativeVtResidentError> {
    let padded = gather
        .byte_len
        .checked_add(3)
        .ok_or(NativeVtResidentError::SizeOverflow)?
        & !3;
    let padded = padded.max(4);
    // A merged range may span multiple compressed blocks. This is a lower
    // bound only; the assembler checks the actual split count after decode.
    let minimum = padded
        .checked_add(
            (gather.ranges.len() as u64)
                .checked_mul(16)
                .ok_or(NativeVtResidentError::SizeOverflow)?,
        )
        .ok_or(NativeVtResidentError::SizeOverflow)?;
    if gather.ranges.len() > limits.max_slices {
        return Err(NativeVtResidentError::Assembly(
            ResidentAssemblyError::TooManySlices {
                requested: gather.ranges.len(),
                limit: limits.max_slices,
            },
        ));
    }
    if minimum > limits.max_device_bytes {
        return Err(NativeVtResidentError::Assembly(
            ResidentAssemblyError::AllocationBudgetExceeded {
                required: minimum,
                limit: limits.max_device_bytes,
            },
        ));
    }
    Ok(padded)
}

fn neighbor_regions(
    key: TexturePageKey,
    regions: &BTreeMap<TexturePageKey, ResidentPageRegion>,
    keys: &BTreeSet<TexturePageKey>,
) -> [Option<ResidentPageRegion>; 9] {
    let mut result = [None; 9];
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let (Ok(x), Ok(y)) = (
                u16::try_from(i32::from(key.x) + dx),
                u16::try_from(i32::from(key.y) + dy),
            ) else {
                continue;
            };
            let neighbor = TexturePageKey { x, y, ..key };
            if (dx == 0 && dy == 0) || keys.contains(&neighbor) {
                result[((dy + 1) * 3 + dx + 1) as usize] = regions.get(&neighbor).copied();
            }
        }
    }
    result
}

fn add_budget(used: &mut u64, amount: u64, limit: u64) -> Result<(), NativeVtResidentError> {
    *used = used
        .checked_add(amount)
        .ok_or(NativeVtResidentError::SizeOverflow)?;
    if *used > limit {
        return Err(NativeVtResidentError::UploadBudget {
            required: *used,
            limit,
        });
    }
    Ok(())
}

fn device_scope<T>(
    device: &wgpu::Device,
    operation: impl FnOnce() -> Result<T, NativeVtResidentError>,
) -> Result<T, NativeVtResidentError> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(Err(NativeVtResidentError::DeviceResource));
    let validation = futures_lite::future::block_on(validation.pop());
    let internal = futures_lite::future::block_on(internal.pop());
    let oom = futures_lite::future::block_on(oom.pop());
    if validation.is_some() || internal.is_some() || oom.is_some() {
        Err(NativeVtResidentError::DeviceResource)
    } else {
        result
    }
}

#[cfg(test)]
#[path = "resident_upload_tests.rs"]
mod tests;
