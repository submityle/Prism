//! Material-facing virtual page lookup: residency, mip-tail fallback and demand.
//!
//! LOD selection alone is not a virtual-texture sample. This module consumes
//! the two pages emitted by [`VirtualTexture::residency`] against the shared
//! streaming page table and produces physical references that a fetch backend
//! can use, while preserving every original demand for feedback. Missing pages
//! never fabricate a slot: they carry `None` and remain in `feedback`.

use alloc::vec::Vec;

use prism_render_architecture::texture_streaming::{GpuPageTable, PageResolution, TexturePageKey};

use super::{PageRequest, VirtualTexture};

/// One requested virtual page and its resident physical-page resolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialPageReference {
    /// Original finest page requested by the material sample.
    ///
    /// `None` means that the request cannot be represented by the packed
    /// streaming key ABI. Such a request is rejected and is never saturated
    /// into a fabricated key.
    pub requested: Option<TexturePageKey>,
    /// Resident page that backs the lookup, including a coarse mip fallback.
    pub resolved: Option<PageResolution>,
}

/// Material sample lookup result with explicit missing-page feedback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialPageLookup {
    /// Fine then coarse trilinear references. Duplicate top-mip requests remain
    /// duplicated to preserve the sample's exact blend pair.
    pub pages: [MaterialPageReference; 2],
    /// Original page demands requiring streamer attention. A coarse fallback is
    /// still reported so the streamer can refine quality; a total miss is also
    /// reported and must use the caller's non-VT fallback texel.
    pub feedback: Vec<TexturePageKey>,
}

impl VirtualTexture {
    /// Resolves a sample's trilinear page pair through the live streaming
    /// indirection table, retaining demands when a finer page is not resident.
    ///
    /// `mip_tail_floor` is the coarsest fallback mip pinned by the streamer. If
    /// absent, only exact pages are accepted. Page indices outside the packed
    /// streaming key range are treated as misses rather than truncated.
    #[must_use]
    pub fn lookup_pages(
        self,
        texture: u32,
        layer: u16,
        uv: [f32; 2],
        lod: f32,
        page_table: &GpuPageTable,
        mip_tail_floor: Option<u8>,
    ) -> MaterialPageLookup {
        let requested = self.residency(uv, lod);
        let first = resolve_page(texture, layer, requested[0], page_table, mip_tail_floor);
        let second = resolve_page(texture, layer, requested[1], page_table, mip_tail_floor);
        let mut feedback = Vec::with_capacity(2);
        for reference in [first, second] {
            if let Some(requested) = reference.requested
                && reference
                    .resolved
                    .is_none_or(|resolved| resolved.mip_bias != 0)
                && !feedback.contains(&requested)
            {
                feedback.push(requested);
            }
        }
        MaterialPageLookup {
            pages: [first, second],
            feedback,
        }
    }
}

fn resolve_page(
    texture: u32,
    layer: u16,
    page: PageRequest,
    table: &GpuPageTable,
    mip_tail_floor: Option<u8>,
) -> MaterialPageReference {
    let Some(mip) = u8::try_from(page.mip).ok() else {
        return MaterialPageReference {
            requested: None,
            resolved: None,
        };
    };
    let Some(x) = u16::try_from(page.page_x).ok() else {
        return MaterialPageReference {
            requested: None,
            resolved: None,
        };
    };
    let Some(y) = u16::try_from(page.page_y).ok() else {
        return MaterialPageReference {
            requested: None,
            resolved: None,
        };
    };
    let requested = TexturePageKey {
        texture,
        mip,
        layer,
        x,
        y,
    };
    let resolved = table.resolve(requested, mip_tail_floor.unwrap_or(requested.mip));
    MaterialPageReference {
        requested: Some(requested),
        resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::texture_streaming::GpuPageTable;

    fn key(mip: u8, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 7,
            mip,
            layer: 2,
            x,
            y,
        }
    }

    #[test]
    fn material_lookup_resolves_fine_pages_and_emits_no_feedback() {
        let table = GpuPageTable::from_bindings(&[(key(0, 1, 1), 13), (key(1, 0, 0), 21)]);
        let vt = VirtualTexture::new(256, 256, 128);
        let result = vt.lookup_pages(7, 2, [0.5, 0.5], 0.25, &table, Some(1));
        assert_eq!(result.pages[0].resolved.unwrap().slot, 13);
        assert_eq!(result.pages[1].resolved.unwrap().slot, 21);
        assert!(result.feedback.is_empty());
    }

    #[test]
    fn material_lookup_uses_tail_and_keeps_fine_demand_for_feedback() {
        let table = GpuPageTable::from_bindings(&[(key(1, 0, 0), 21)]);
        let vt = VirtualTexture::new(256, 256, 128);
        let result = vt.lookup_pages(7, 2, [0.5, 0.5], 0.25, &table, Some(1));
        assert_eq!(result.pages[0].resolved.unwrap().mip_bias, 1);
        assert_eq!(result.pages[0].resolved.unwrap().slot, 21);
        assert_eq!(result.feedback, vec![key(0, 1, 1)]);
    }

    #[test]
    fn absent_tail_is_a_real_miss_not_a_fabricated_slot() {
        let table = GpuPageTable::new();
        let vt = VirtualTexture::new(256, 256, 128);
        let result = vt.lookup_pages(7, 2, [0.5, 0.5], 0.25, &table, Some(1));
        assert!(result.pages.iter().all(|page| page.resolved.is_none()));
        assert_eq!(result.feedback, vec![key(0, 1, 1), key(1, 0, 0)]);
    }

    #[test]
    fn texture_layer_identity_is_preserved_during_fallback() {
        let table = GpuPageTable::from_bindings(&[(key(1, 0, 0), 21)]);
        let vt = VirtualTexture::new(256, 256, 128);
        let result = vt.lookup_pages(7, 3, [0.5, 0.5], 0.25, &table, Some(1));
        assert!(result.pages.iter().all(|page| page.resolved.is_none()));
    }

    #[test]
    fn unrepresentable_page_indices_are_rejected_without_saturation() {
        let table = GpuPageTable::from_bindings(&[(key(u8::MAX, u16::MAX, u16::MAX), 99)]);
        let max_valid = resolve_page(
            7,
            2,
            PageRequest {
                mip: u32::from(u8::MAX),
                page_x: u32::from(u16::MAX),
                page_y: u32::from(u16::MAX),
            },
            &table,
            Some(u8::MAX),
        );
        assert_eq!(max_valid.requested, Some(key(u8::MAX, u16::MAX, u16::MAX)));
        assert_eq!(max_valid.resolved.unwrap().slot, 99);

        let requests = [
            PageRequest {
                mip: u32::from(u8::MAX) + 1,
                page_x: 0,
                page_y: 0,
            },
            PageRequest {
                mip: 0,
                page_x: u32::from(u16::MAX) + 1,
                page_y: 0,
            },
            PageRequest {
                mip: 0,
                page_x: 0,
                page_y: u32::from(u16::MAX) + 1,
            },
        ];

        for request in requests {
            let reference = resolve_page(7, 2, request, &table, Some(0));
            assert_eq!(reference.requested, None);
            assert_eq!(reference.resolved, None);
        }
    }

    #[test]
    fn unrepresentable_xy_requests_do_not_enter_feedback() {
        let table = GpuPageTable::new();
        let vt = VirtualTexture::new(u32::MAX, u32::MAX, 1);
        let result = vt.lookup_pages(7, 2, [1.0, 1.0], 0.0, &table, Some(0));

        assert!(result
            .pages
            .iter()
            .all(|page| { page.requested.is_none() && page.resolved.is_none() }));
        assert!(result.feedback.is_empty());
    }
}
