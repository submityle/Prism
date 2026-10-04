//! Per-view resident resources backing the glossy-specular ReSTIR *reuse* pass.
//!
//! The reuse pass is a screen-space layer that sits on top of the SSR trace: it
//! reconstructs each pixel's view-space glossy point from the SSR geometry
//! prepass's reverse-Z `scene_depth` and packed `normal_roughness`, streams the
//! current-frame screen-space GGX candidate into a per-pixel reservoir,
//! temporally merges the reprojected prior-frame reservoir under a
//! roughness-tightened confidence cap, finalises an unbiased contribution weight
//! and resolves a low-variance specular estimate (+ confidence) the
//! `spec_denoise` passes and the energy-conserving composite consume.
//!
//! Because the merge reads *last* frame's reservoir and writes *this* frame's,
//! the subsystem owns a **ping-pong** pair of per-pixel reservoir storage
//! buffers and flips which is the read source each frame (mirroring
//! [`super::super::world_restir::resources`]'s world-space table). Alongside the
//! pair it owns one viewport-sized resolved target the kernel writes the merged
//! `rgb` specular estimate + `a` confidence into — a distinct texture because
//! `rgba16float` is not read-write storage-capable, so the resolve cannot run in
//! place on the candidate it reads.
//!
//! Unlike the world-space `ReSTIR` table, these buffers are **viewport-sized**
//! (one reservoir slot per pixel), so the reallocation trigger is a framebuffer
//! size change, exactly like [`super::super::ssgi::resources`]'s targets. The
//! subsystem is gated identically to SSGI — on `enable_spec_gi`, `enable_ssr`
//! and `enable_visibility_buffer`, a resident visibility buffer, and a
//! single-sample view — because it consumes SSR's rebuilt prepass inputs and is
//! meaningless without them.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;
use super::abi::GpuSpecrReservoir;

/// Resolved glossy-specular estimate written by the reuse kernel: `rgb` = merged
/// low-variance specular radiance, `a` = ReSTIR confidence the `spec_denoise`
/// history clamp and the energy-conserving composite consume. Wide HDR so the
/// resolved reflection keeps its range up to the composite, matching
/// [`super::super::ssr::resources`]' reflection output and the SSGI gather.
pub(crate) const SPEC_GI_RESOLVED_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Byte stride of one resident reservoir slot. One slot per framebuffer texel;
/// the ping-pong buffers are sized `width * height` slots of this stride.
const SPEC_GI_RESERVOIR_STRIDE: u64 = size_of::<GpuSpecrReservoir>() as u64;

/// Compile-time guard that the resident buffer stride the allocator sizes with
/// is byte-for-byte the frozen 64-byte reservoir ABI the kernel indexes, so the
/// host allocation and the WESL `array<GpuSpecrReservoir>` never drift.
const _: () = assert!(SPEC_GI_RESERVOIR_STRIDE == 64);

/// Ping-pong read-source buffer index for a monotonic frame: even frames read
/// slot `0`, odd frames read slot `1`.
const fn reservoir_src_index(frame: u32) -> usize {
    (frame & 1) as usize
}

/// Ping-pong write-target buffer index for a monotonic frame: the complement of
/// [`reservoir_src_index`], so the reuse kernel never reads and writes the same
/// buffer in one dispatch.
const fn reservoir_dst_index(frame: u32) -> usize {
    ((frame & 1) ^ 1) as usize
}

/// Per-pixel reservoir-slot count for a viewport, clamped to at least one so a
/// zero-area or not-yet-sized view still allocates a valid (if trivial) buffer.
const fn reservoir_slot_count(size: UVec2) -> u64 {
    let texels = (size.x as u64) * (size.y as u64);
    if texels == 0 {
        1
    } else {
        texels
    }
}

/// Per-view resident glossy-specular reuse resources, present only while the
/// reuse pass is enabled and the viewport size is known.
#[derive(Component)]
pub(crate) struct ViewSpecGiReuse {
    /// Ping-pong pair of per-pixel reservoir storage buffers. Each is
    /// `width * height` slots of [`SPEC_GI_RESERVOIR_STRIDE`] bytes;
    /// [`frame`](Self::frame) selects which is this frame's read source and
    /// which is the write target. `wgpu` zero-initialises a freshly created
    /// storage buffer before its first shader read, so both start as all-empty
    /// (`has_sample == 0`) reservoirs without an explicit clear.
    reservoirs: [Buffer; 2],
    /// Resolved specular+confidence target the kernel writes (`@binding(6)`,
    /// storage write) and the denoise/composite read (`textureLoad`). Single
    /// mip, full resolution.
    resolved: CachedTexture,
    /// Viewport extent this view's buffers + target were allocated for; the only
    /// reallocation trigger.
    pub(crate) size: UVec2,
    /// Monotonic frame counter driving the ping-pong flip (`frame & 1`). The
    /// crate does not link Bevy's `FrameCount`, so the counter lives here and
    /// advances in [`prepare_spec_gi_reuse_resources`] (mirroring
    /// [`super::super::world_restir::resources`]).
    frame: u32,
}

impl ViewSpecGiReuse {
    /// This frame's read-only reservoir buffer (last frame's finalised output),
    /// bound at `@binding(1)` (`prior_reservoirs`).
    pub(crate) fn src_buffer(&self) -> &Buffer {
        &self.reservoirs[reservoir_src_index(self.frame)]
    }

    /// This frame's read-write reservoir buffer (the reuse kernel's output),
    /// bound at `@binding(2)` (`out_reservoirs`). Next frame the pair swaps, so
    /// this becomes the following dispatch's `src`.
    pub(crate) fn dst_buffer(&self) -> &Buffer {
        &self.reservoirs[reservoir_dst_index(self.frame)]
    }

    /// Storage/sampling view of the resolved specular+confidence target. The
    /// reuse kernel writes it (`@binding(6)`, storage) and the denoise/composite
    /// read it (`textureLoad`).
    pub(crate) fn resolved_view(&self) -> &TextureView {
        &self.resolved.default_view
    }

    /// This view's monotonic frame counter (advanced in
    /// [`prepare_spec_gi_reuse_resources`]). The spatial reuse pass salts its
    /// per-frame Fibonacci-spiral angular jitter with it so the neighbour
    /// pattern decorrelates across frames (the temporal reuse + `spec_denoise`
    /// filter resolve the residual).
    pub(crate) fn frame(&self) -> u32 {
        self.frame
    }
}

/// (Re)allocates [`ViewSpecGiReuse`] for every view that has a resident
/// visibility buffer while the reuse pass is enabled, and removes it otherwise.
///
/// Gated on `enable_spec_gi`, `enable_ssr` and `enable_visibility_buffer`: the
/// reuse pass reconstructs its glossy point from SSR's rebuilt reverse-Z
/// `scene_depth` and packed `normal_roughness`, so it is meaningless without
/// them, and on single-sample views because the visibility buffer those inputs
/// decode is itself single-sample. The buffers + target are recreated whenever
/// the viewport size changes; the steady-state per-frame work is the cheap
/// ping-pong flip (advancing `frame`).
pub(crate) fn prepare_spec_gi_reuse_resources(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
        Option<&mut ViewSpecGiReuse>,
    )>,
) {
    for (entity, camera, msaa, visibility, existing) in &mut views {
        let enabled = settings.enable_spec_gi
            && settings.enable_ssr
            && settings.enable_visibility_buffer
            && visibility.is_some()
            && msaa.is_none_or(|value| value.samples() == 1);
        if !enabled {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewSpecGiReuse>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };

        // Steady state: an existing allocation already at the live viewport only
        // flips the ping-pong and advances the frame counter. Every other case
        // (no allocation yet, or the viewport was resized) falls through to
        // (re)allocate the ping-pong pair + resolved target below.
        if let Some(mut view) = existing {
            if view.size == size {
                view.frame = view.frame.wrapping_add(1);
                continue;
            }
        }

        let buffer_size = reservoir_slot_count(size) * SPEC_GI_RESERVOIR_STRIDE;
        // STORAGE: bound read-only as `prior_reservoirs`, read-write as
        // `out_reservoirs`. COPY_DST so a host-side zero clear can be scheduled
        // if ever needed; `wgpu` already zero-initialises before the first read.
        let make = |label: &'static str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let reservoirs = [
            make("prism spec_gi reservoirs A"),
            make("prism spec_gi reservoirs B"),
        ];

        let resolved = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism spec_gi resolved"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SPEC_GI_RESOLVED_FORMAT,
                // Written by the reuse kernel (storage), sampled by the denoise
                // and the energy-conserving composite.
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(ViewSpecGiReuse {
            reservoirs,
            resolved,
            size,
            frame: 0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_format_is_wide_hdr_with_confidence_alpha() {
        // rgb specular + a confidence; shares the SSR reflection / SSGI gather
        // wide-HDR encoding so the composite reads every GI source uniformly.
        assert_eq!(SPEC_GI_RESOLVED_FORMAT, TextureFormat::Rgba16Float);
    }

    #[test]
    fn reservoir_stride_matches_the_slot_abi() {
        // The compile-time guard above already enforces this; assert at runtime
        // too so the intent is visible in the test report.
        assert_eq!(SPEC_GI_RESERVOIR_STRIDE, 64);
        assert_eq!(
            size_of::<GpuSpecrReservoir>() as u64,
            SPEC_GI_RESERVOIR_STRIDE
        );
    }

    #[test]
    fn ping_pong_indices_alternate_and_never_alias() {
        for frame in 0u32..8 {
            let src = reservoir_src_index(frame);
            let dst = reservoir_dst_index(frame);
            assert!(src < 2 && dst < 2);
            assert_ne!(src, dst, "reuse must never read and write the same buffer");
        }
        assert_eq!(reservoir_src_index(0), 0);
        assert_eq!(reservoir_dst_index(0), 1);
        assert_eq!(reservoir_src_index(1), 1);
        assert_eq!(reservoir_dst_index(1), 0);
        // This frame's dst is next frame's src (true ping-pong).
        assert_eq!(reservoir_dst_index(0), reservoir_src_index(1));
        assert_eq!(reservoir_dst_index(1), reservoir_src_index(2));
    }

    #[test]
    fn slot_count_is_viewport_area_clamped_to_one() {
        assert_eq!(reservoir_slot_count(UVec2::new(1920, 1080)), 1920 * 1080);
        assert_eq!(reservoir_slot_count(UVec2::new(0, 0)), 1);
        assert_eq!(reservoir_slot_count(UVec2::new(0, 1080)), 1);
        assert_eq!(reservoir_slot_count(UVec2::new(7, 3)), 21);
    }
}
