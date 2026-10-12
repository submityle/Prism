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
    pub requested: TexturePageKey,
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
            if reference
                .resolved
                .is_none_or(|resolved| resolved.mip_bias != 0)
                && !feedback.contains(&reference.requested)
            {
                feedback.push(reference.requested);
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
    let requested = TexturePageKey {
        texture,
        mip: u8::try_from(page.mip).unwrap_or(u8::MAX),
        layer,
        x: u16::try_from(page.page_x).unwrap_or(u16::MAX),
        y: u16::try_from(page.page_y).unwrap_or(u16::MAX),
    };
    let representable = page.mip <= u8::MAX.into()
        && page.page_x <= u16::MAX.into()
        && page.page_y <= u16::MAX.into();
    let resolved = representable
        .then(|| table.resolve(requested, mip_tail_floor.unwrap_or(requested.mip)))
        .flatten();
    MaterialPageReference {
        requested,
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
}
