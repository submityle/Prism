//! Pure, bounded planning for the unified native VT atlas.
//!
//! Source IDs follow the native extractor's contract: `texture_id = slot.index`.
//! Feedback order is priority order. Valid feedback sources are admitted first,
//! then other referenced sources in texture-ID order. Every admission reserves
//! the complete coarsest mip across all source layers before any refinement.
//!
//! This module neither builds upload tiles nor caches plans. The scheduler must
//! cache the complete input identity (including source revisions and slot
//! generations), retry staging under backpressure, and invalidate its uploaded
//! identity on device recovery. Equal page lists alone do not identify equal
//! source bytes. Device-specific limits remain the GPU owner's responsibility.
//! Feedback keys contain no slot generation: the scheduler must discard stale
//! feedback when a slot is recycled. This planner checks the current slot index
//! and full image asset identity, not the generation that produced feedback.

use std::collections::{BTreeMap, BTreeSet};

use prism_asset::AssetId;
use prism_image::Image;
use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_material::{MaterialVtError, MaterialVtFormat, MaterialVtSourceLayout};

use super::{NativeVtConfig, NativeVtError, NativeVtResidency, NativeVtSources};
use crate::material::BindlessSlot;

/// A complete desired resident set and diagnostics; never a residency delta.
#[derive(Clone, Debug)]
pub struct NativeVtPlan {
    /// Compact, unique physical slots, with no portable-atlas entries.
    pub residency: NativeVtResidency,
    /// Counts and hard capacity bounds for this plan.
    pub stats: NativeVtPlanStats,
    /// Referenced sources whose entire coarsest-layer group was not admitted.
    pub rejected_sources: Vec<NativeVtRejectedSource>,
    /// Unique feedback keys not exactly resident, in first-occurrence order.
    pub unmet_demand: Vec<NativeVtUnmetDemand>,
}

/// Bounded residency accounting. Counts describe unique sources and pages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeVtPlanStats {
    /// Physical slots across every atlas layer.
    pub atlas_capacity: u32,
    /// Minimum of physical capacity and page-table capacity; may be zero.
    pub page_budget: u32,
    /// Referenced images, including missing or incompatible sources.
    pub referenced_sources: usize,
    /// Sources with all their coarsest-layer pages pinned.
    pub admitted_sources: usize,
    /// Pinned coarsest pages across all admitted source layers.
    pub tail_pages: u32,
    /// Additional pages on explicitly demanded ancestor chains.
    pub detail_pages: u32,
    /// Feedback list length before deduplication.
    pub demand_count: usize,
    /// Feedback keys after stable deduplication.
    pub unique_demand: usize,
    /// Repeated feedback keys ignored after their first occurrence.
    pub duplicate_demand: usize,
    /// Unique feedback keys exactly resident in the final plan.
    pub satisfied_demand: usize,
}

/// A recoverable admission or feedback failure, not a global planner failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVtPlanIssue {
    /// Feedback may not bring an image with no live material reference online.
    UnreferencedSource,
    /// No captured source exists at the referenced native texture ID.
    MissingSource,
    /// The source at this slot belongs to a different image identity.
    SourceImageMismatch,
    /// Source page size or texel format differs from the current atlas.
    AtlasMismatch,
    /// Shifted page coordinates cannot reach the valid coarsest tail for every
    /// source page. Admitting only tails would leave some pages without fallback.
    UnresolvableTail,
    /// Feedback addresses a nonexistent mip, layer, or page coordinate.
    InvalidPage(MaterialVtError),
    /// The entire coarsest-layer group could not fit; nothing was admitted.
    TailBudget {
        /// Number of coarsest pages required by this source.
        required: u32,
        /// Unused slots/entries at this source's admission attempt.
        available: u32,
    },
    /// The exact requested page did not fit after pinning and refinement.
    PageBudget,
}

/// One omitted referenced image, including sources without explicit feedback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVtRejectedSource {
    /// Referenced image identity.
    pub image: AssetId<Image>,
    /// Native texture ID, equal to its referenced bindless slot index.
    pub texture: u32,
    /// Why this complete source group was omitted.
    pub reason: NativeVtPlanIssue,
}

/// One unsatisfied unique feedback request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVtUnmetDemand {
    /// Original requested key.
    pub key: TexturePageKey,
    /// Why the exact key is absent.
    pub reason: NativeVtPlanIssue,
    /// Nearest resident covering ancestor, only for valid admitted requests.
    pub fallback: Option<TexturePageKey>,
}

/// Plan without touching GPU resources, source payloads, or material records.
///
/// All referenced-source tail groups have priority over detail pages. Admission
/// is greedy in deterministic source priority: an oversized group is skipped
/// whole, so a later smaller group can still fit. Detail allocation then walks
/// each unique valid demand from coarsest toward the requested mip, retaining
/// every admitted ancestor. No-feedback plans contain only pinned tail pages.
/// Those tails resolve every valid page of each admitted source, including all
/// layers. A source rejected as a whole has no VT fallback in this plan; callers
/// must observe `rejected_sources` rather than publish it as VT-resident.
/// Sources whose edge coordinates cannot shift into the coarsest tail are
/// rejected as `UnresolvableTail`. In particular, one-texel pages with an NPOT
/// largest dimension are incompatible with the current shift-only CPU/GPU
/// resolver. No invalid tail keys or unrequested fine pages are synthesized.
///
/// Invalid global geometry/format and duplicate referenced slot indices return
/// an error. Missing/incompatible sources, malformed feedback and insufficient
/// budgets instead produce diagnostics while other sources continue planning.
/// Source-format equality includes the sRGB distinction. Capacity arithmetic is
/// checked before allocating anything proportional to the output; planning never
/// enumerates the full source pyramid or creates staging/upload bytes.
/// Working storage scales with references, unique feedback and admitted pages,
/// not configured capacity. Physical slots are assigned in page-key order, not
/// admission order; coarsest-before-detail describes budget reservation.
pub fn plan_residency(
    config: NativeVtConfig,
    sources: &NativeVtSources,
    references: &BTreeMap<AssetId<Image>, BindlessSlot>,
    demand: &[TexturePageKey],
) -> Result<NativeVtPlan, NativeVtError> {
    config.geometry()?;
    let atlas_capacity = config.capacity()?;
    if config.format == MaterialVtFormat::Rgba32Float {
        return Err(NativeVtError::UnsupportedFormat);
    }
    let budget = atlas_capacity.min(config.max_entries);
    let mut available = BTreeMap::new();
    let mut images = BTreeMap::new();
    let mut issues = BTreeMap::new();
    for (&image, slot) in references {
        let texture = slot.index;
        if images.insert(texture, image).is_some() {
            return Err(NativeVtError::DuplicateSlot(texture));
        }
        let issue = match sources.get(texture) {
            None => Some(NativeVtPlanIssue::MissingSource),
            Some(source) if source.image() != image => Some(NativeVtPlanIssue::SourceImageMismatch),
            Some(source)
                if source.layout().descriptor().page_size() != config.page_size
                    || source.layout().descriptor().format() != config.format =>
            {
                Some(NativeVtPlanIssue::AtlasMismatch)
            }
            Some(source) if !coarsest_covers_source(source.layout()) => {
                Some(NativeVtPlanIssue::UnresolvableTail)
            }
            Some(source) => {
                available.insert(texture, source.as_ref());
                None
            }
        };
        if let Some(issue) = issue {
            issues.insert(texture, issue);
        }
    }

    let mut seen = BTreeSet::new();
    let unique: Vec<_> = demand
        .iter()
        .copied()
        .filter(|key| seen.insert(*key))
        .map(|key| {
            let issue = if !images.contains_key(&key.texture) {
                Some(NativeVtPlanIssue::UnreferencedSource)
            } else if let Some(&issue) = issues.get(&key.texture) {
                Some(issue)
            } else {
                available[&key.texture]
                    .layout()
                    .page_metadata(key)
                    .err()
                    .map(NativeVtPlanIssue::InvalidPage)
            };
            (key, issue)
        })
        .collect();
    drop(seen);

    let mut ordered = BTreeSet::new();
    let priority: Vec<_> = unique
        .iter()
        .filter(|(_, issue)| issue.is_none())
        .map(|(key, _)| key.texture)
        .chain(available.keys().copied())
        .filter(|texture| ordered.insert(*texture))
        .collect();
    drop(ordered);
    let mut pages = BTreeSet::new();
    let mut admitted = BTreeSet::new();
    let mut used = 0u32;
    for texture in priority {
        let descriptor = available[&texture].layout().descriptor();
        // A validated complete chain ends at 1x1, but has one tail per layer.
        let required = u32::from(descriptor.layers());
        let remaining = budget - used;
        if required > remaining {
            issues.insert(
                texture,
                NativeVtPlanIssue::TailBudget {
                    required,
                    available: remaining,
                },
            );
            continue;
        }
        for layer in 0..descriptor.layers() {
            pages.insert(TexturePageKey {
                texture,
                mip: descriptor.mip_count() - 1,
                layer,
                x: 0,
                y: 0,
            });
        }
        used += required;
        admitted.insert(texture);
    }
    let tail_pages = used;

    for &(key, issue) in &unique {
        if issue.is_some() || !admitted.contains(&key.texture) {
            continue;
        }
        let source = available[&key.texture].layout();
        let coarsest = source.descriptor().mip_count() - 1;
        for mip in (key.mip..coarsest).rev() {
            let page = ancestor(key, mip);
            if pages.contains(&page) {
                continue;
            }
            if used == budget {
                break;
            }
            // Truncated odd dimensions can have no page at a shifted parent
            // coordinate. Never synthesize it or clamp to a different footprint.
            if source.page_metadata(page).is_err() {
                continue;
            }
            pages.insert(page);
            used += 1;
        }
    }

    let mut unmet_demand = Vec::new();
    for &(key, issue) in &unique {
        if let Some(reason) = issue.or_else(|| issues.get(&key.texture).copied()) {
            unmet_demand.push(NativeVtUnmetDemand {
                key,
                reason,
                fallback: None,
            });
        } else if !pages.contains(&key) {
            let coarsest = available[&key.texture].layout().descriptor().mip_count() - 1;
            let fallback = (key.mip..=coarsest)
                .map(|mip| ancestor(key, mip))
                .find(|page| pages.contains(page));
            unmet_demand.push(NativeVtUnmetDemand {
                key,
                reason: NativeVtPlanIssue::PageBudget,
                fallback,
            });
        }
    }
    let rejected_sources = issues
        .into_iter()
        .map(|(texture, reason)| NativeVtRejectedSource {
            image: images[&texture],
            texture,
            reason,
        })
        .collect();
    let stats = NativeVtPlanStats {
        atlas_capacity,
        page_budget: budget,
        referenced_sources: references.len(),
        admitted_sources: admitted.len(),
        tail_pages,
        detail_pages: used - tail_pages,
        demand_count: demand.len(),
        unique_demand: unique.len(),
        duplicate_demand: demand.len() - unique.len(),
        satisfied_demand: unique.len() - unmet_demand.len(),
    };
    Ok(NativeVtPlan {
        residency: NativeVtResidency {
            pages: pages
                .into_iter()
                .enumerate()
                .map(|(slot, key)| (key, slot as u32))
                .collect(),
            portable: Vec::new(),
        },
        stats,
        rejected_sources,
        unmet_demand,
    })
}

fn coarsest_covers_source(source: &MaterialVtSourceLayout) -> bool {
    let coarsest = source.descriptor().mip_count() - 1;
    // The largest coordinate bounds every page in a mip, independently of the
    // number of layers. Check at most 32 metadata rows, never source texels or
    // the page pyramid. This mirrors the shift-only CPU/shader resolver rather
    // than assuming the final 1x1 mip is reachable from every NPOT edge page.
    source.mips().iter().all(|mip| {
        let shift = u32::from(coarsest - mip.mip);
        (mip.pages_x - 1).checked_shr(shift).unwrap_or(0) == 0
            && (mip.pages_y - 1).checked_shr(shift).unwrap_or(0) == 0
    })
}

fn ancestor(key: TexturePageKey, mip: u8) -> TexturePageKey {
    let shift = u32::from(mip - key.mip);
    TexturePageKey {
        mip,
        x: key.x.checked_shr(shift).unwrap_or(0),
        y: key.y.checked_shr(shift).unwrap_or(0),
        ..key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_asset::AssetIndex;
    use prism_render_architecture::texture_streaming::GpuPageTable;
    use prism_render_material::{MaterialVtSourceDescriptor, MaterialVtSourceSnapshot};

    fn config(entries: u32) -> NativeVtConfig {
        NativeVtConfig {
            page_size: 2,
            gutter_px: 1,
            tiles_per_row: 2,
            atlas_layers: 2,
            max_entries: entries,
            format: MaterialVtFormat::Rgba8Unorm,
        }
    }

    fn image(index: u32) -> AssetId<Image> {
        AssetId::new(AssetIndex::from_parts(index, 1))
    }

    fn reference(texture: u32) -> (AssetId<Image>, BindlessSlot) {
        (
            image(texture),
            BindlessSlot {
                index: texture,
                generation: 7,
            },
        )
    }

    fn snapshot(
        texture: u32,
        width: u32,
        height: u32,
        layers: u16,
        page_size: u32,
        format: MaterialVtFormat,
    ) -> MaterialVtSourceSnapshot {
        let mip_count = width.max(height).ilog2() + 1;
        let descriptor = MaterialVtSourceDescriptor::new(
            texture,
            width,
            height,
            layers,
            page_size,
            mip_count as u8,
            format,
        )
        .unwrap();
        let bytes: usize = (0..mip_count)
            .map(|mip| {
                (width >> mip).max(1) as usize
                    * (height >> mip).max(1) as usize
                    * usize::from(layers)
                    * format.bytes_per_texel() as usize
            })
            .sum();
        MaterialVtSourceSnapshot::new(descriptor, vec![0; bytes]).unwrap()
    }

    fn add(sources: &mut NativeVtSources, texture: u32, width: u32, height: u32, layers: u16) {
        sources
            .insert(
                image(texture),
                snapshot(texture, width, height, layers, 2, config(8).format),
            )
            .unwrap();
    }

    fn key(texture: u32, mip: u8, layer: u16, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture,
            mip,
            layer,
            x,
            y,
        }
    }

    fn check(plan: &NativeVtPlan, config: NativeVtConfig, sources: &NativeVtSources) {
        assert!(plan.residency.portable.is_empty());
        assert!(plan.residency.pages.len() <= config.max_entries as usize);
        assert!(plan.residency.pages.len() <= config.capacity().unwrap() as usize);
        assert_eq!(
            plan.residency.pages.len(),
            (plan.stats.tail_pages + plan.stats.detail_pages) as usize
        );
        let mut pages = BTreeSet::new();
        for (index, &(page, slot)) in plan.residency.pages.iter().enumerate() {
            assert_eq!(slot, index as u32);
            assert!(config.geometry().unwrap().slot_placement(slot).layer < config.atlas_layers);
            assert!(pages.insert(page));
            sources
                .get(page.texture)
                .unwrap()
                .layout()
                .page_metadata(page)
                .unwrap();
        }
        let textures: BTreeSet<_> = pages.iter().map(|page| page.texture).collect();
        assert_eq!(textures.len(), plan.stats.admitted_sources);
        let table = GpuPageTable::from_bindings(&plan.residency.pages);
        let mut tails = 0;
        for texture in textures {
            let source = sources.get(texture).unwrap().layout();
            let descriptor = source.descriptor();
            tails += u32::from(descriptor.layers());
            for layer in 0..descriptor.layers() {
                assert!(pages.contains(&key(texture, descriptor.mip_count() - 1, layer, 0, 0)));
            }
            // Enumerate only these tiny test sources, never in the planner.
            // Even a no-feedback plan must resolve every valid source page.
            for page in source.page_keys() {
                let resolved = table.resolve(page, descriptor.mip_count() - 1).unwrap();
                assert!(pages.contains(&resolved.key));
                assert_eq!(resolved.key.texture, page.texture);
                assert_eq!(resolved.key.layer, page.layer);
            }
        }
        assert_eq!(tails, plan.stats.tail_pages);
        for unmet in &plan.unmet_demand {
            assert_eq!(
                unmet.fallback.is_some(),
                unmet.reason == NativeVtPlanIssue::PageBudget
            );
            if let Some(fallback) = unmet.fallback {
                let coarsest = sources
                    .get(unmet.key.texture)
                    .unwrap()
                    .layout()
                    .descriptor()
                    .mip_count()
                    - 1;
                let resolved = table.resolve(unmet.key, coarsest).unwrap();
                assert_eq!(resolved.key, fallback);
            }
        }
    }

    #[test]
    fn no_feedback_pins_all_layers_and_no_invented_detail() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 16, 4, 3);
        add(&mut sources, 20, 1, 1, 2);
        let references = BTreeMap::from([reference(20), reference(10)]);
        let plan = plan_residency(config(8), &sources, &references, &[]).unwrap();
        check(&plan, config(8), &sources);
        assert_eq!(plan.stats.admitted_sources, 2);
        assert_eq!((plan.stats.tail_pages, plan.stats.detail_pages), (5, 0));
        assert!(plan.rejected_sources.is_empty());
        assert_eq!(plan.residency.pages.last().unwrap().1, 4);
        // Slot 4 is in atlas layer 1, not restricted to one physical layer.
        assert_eq!(config(8).geometry().unwrap().slot_placement(4).layer, 1);
    }

    #[test]
    fn oversized_tail_group_is_skipped_without_partial_admission() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 4, 8, 3);
        add(&mut sources, 20, 8, 4, 2);
        let references = BTreeMap::from([reference(10), reference(20)]);
        let plan = plan_residency(config(2), &sources, &references, &[]).unwrap();
        check(&plan, config(2), &sources);
        assert_eq!(plan.stats.tail_pages, 2);
        assert!(plan
            .residency
            .pages
            .iter()
            .all(|(key, _)| key.texture == 20));
        assert_eq!(
            plan.rejected_sources[0].reason,
            NativeVtPlanIssue::TailBudget {
                required: 3,
                available: 2
            }
        );
    }

    #[test]
    fn feedback_order_prioritizes_whole_sources_but_never_evicts_tails() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 8, 8, 1);
        add(&mut sources, 20, 8, 8, 2);
        let references = BTreeMap::from([reference(10), reference(20)]);
        let wanted = key(20, 0, 1, 3, 3);
        let plan = plan_residency(config(2), &sources, &references, &[wanted]).unwrap();
        check(&plan, config(2), &sources);
        assert_eq!((plan.stats.tail_pages, plan.stats.detail_pages), (2, 0));
        assert!(plan
            .residency
            .pages
            .iter()
            .all(|(key, _)| key.texture == 20));
        assert_eq!(plan.unmet_demand[0].fallback, Some(key(20, 3, 1, 0, 0)));
        assert_eq!(plan.rejected_sources[0].texture, 10);
    }

    #[test]
    fn increasing_budget_refines_demand_while_preserving_ancestors() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 16, 8, 1);
        let references = BTreeMap::from([reference(10)]);
        let wanted = key(10, 0, 0, 7, 3);
        let mut previous = BTreeSet::new();
        for budget in 1..=5 {
            let plan = plan_residency(config(budget), &sources, &references, &[wanted]).unwrap();
            check(&plan, config(budget), &sources);
            let pages: BTreeSet<_> = plan.residency.pages.iter().map(|(key, _)| *key).collect();
            assert!(previous.is_subset(&pages));
            assert_eq!(plan.stats.detail_pages, budget - 1);
            if budget < 5 {
                assert_eq!(
                    plan.unmet_demand[0].fallback,
                    Some(ancestor(wanted, (5 - budget) as u8))
                );
            } else {
                assert!(plan.unmet_demand.is_empty());
                assert_eq!(plan.stats.satisfied_demand, 1);
            }
            previous = pages;
        }
    }

    #[test]
    fn first_unique_demand_wins_and_repeated_feedback_is_free() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 4, 4, 1);
        let references = BTreeMap::from([reference(10)]);
        let first = key(10, 0, 0, 0, 0);
        let second = key(10, 0, 0, 1, 1);
        let plan =
            plan_residency(config(3), &sources, &references, &[first, first, second]).unwrap();
        let reverse = plan_residency(config(3), &sources, &references, &[second, first]).unwrap();
        check(&plan, config(3), &sources);
        check(&reverse, config(3), &sources);
        assert_eq!(
            (plan.stats.unique_demand, plan.stats.duplicate_demand),
            (2, 1)
        );
        assert_eq!(plan.unmet_demand[0].key, second);
        assert_eq!(reverse.unmet_demand[0].key, first);
        let once = plan_residency(config(3), &sources, &references, &[first, second]).unwrap();
        assert_eq!(plan.residency.pages, once.residency.pages);
    }

    #[test]
    fn every_sources_tails_precede_any_fine_page() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 8, 8, 1);
        add(&mut sources, 20, 8, 8, 1);
        let references = BTreeMap::from([reference(10), reference(20)]);
        let plan =
            plan_residency(config(2), &sources, &references, &[key(20, 0, 0, 0, 0)]).unwrap();
        check(&plan, config(2), &sources);
        assert_eq!(
            (
                plan.stats.admitted_sources,
                plan.stats.tail_pages,
                plan.stats.detail_pages
            ),
            (2, 2, 0)
        );
    }

    #[test]
    fn multilayer_tails_are_reserved_before_any_feedback_layer_is_refined() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 8, 4, 2);
        add(&mut sources, 20, 4, 8, 3);
        let references = BTreeMap::from([reference(10), reference(20)]);
        let wanted = key(20, 0, 2, 1, 3);
        for budget in [5, 6] {
            let plan = plan_residency(config(budget), &sources, &references, &[wanted]).unwrap();
            check(&plan, config(budget), &sources);
            assert!(plan.rejected_sources.is_empty());
            assert_eq!(plan.stats.tail_pages, 5);
            assert_eq!(plan.stats.detail_pages, budget - 5);
            assert_eq!(
                plan.unmet_demand[0].fallback,
                Some(ancestor(wanted, if budget == 5 { 3 } else { 2 }))
            );
            assert!(plan.residency.pages.iter().all(|(page, _)| {
                page.mip == 3 || (page.texture == 20 && page.layer == 2 && page.mip == 2)
            }));
        }
    }

    #[test]
    fn no_feedback_admits_by_slot_index_not_image_order_or_identity() {
        let mut sources = NativeVtSources::default();
        sources
            .insert(image(10), snapshot(90, 8, 4, 1, 2, config(8).format))
            .unwrap();
        sources
            .insert(image(90), snapshot(10, 4, 8, 1, 2, config(8).format))
            .unwrap();
        let references =
            BTreeMap::from([(image(10), reference(90).1), (image(90), reference(10).1)]);
        let plan = plan_residency(config(1), &sources, &references, &[]).unwrap();
        check(&plan, config(1), &sources);
        assert_eq!(plan.residency.pages, [(key(10, 3, 0, 0, 0), 0)]);
        assert_eq!(plan.rejected_sources[0].image, image(10));
        assert_eq!(plan.rejected_sources[0].texture, 90);
        let wanted = key(90, 0, 0, 3, 1);
        let prioritized =
            plan_residency(config(1), &sources, &references, &[wanted, wanted]).unwrap();
        check(&prioritized, config(1), &sources);
        assert_eq!(prioritized.residency.pages, [(key(90, 3, 0, 0, 0), 0)]);
        assert_eq!(
            prioritized.unmet_demand[0].fallback,
            Some(key(90, 3, 0, 0, 0))
        );
    }

    #[test]
    fn recycled_slot_cannot_admit_the_previous_image_asset_generation() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 4, 4, 1);
        let current_image = AssetId::new(AssetIndex::from_parts(10, 2));
        let current_slot = BindlessSlot {
            index: 10,
            generation: 8,
        };
        let references = BTreeMap::from([(current_image, current_slot)]);
        let demand = [key(10, 0, 0, 1, 1)];
        let stale = plan_residency(config(2), &sources, &references, &demand).unwrap();
        check(&stale, config(2), &sources);
        assert!(stale.residency.pages.is_empty());
        assert_eq!(stale.rejected_sources[0].image, current_image);
        assert_eq!(
            stale.rejected_sources[0].reason,
            NativeVtPlanIssue::SourceImageMismatch
        );
        assert_eq!(stale.unmet_demand[0].fallback, None);

        sources
            .insert(current_image, snapshot(10, 8, 4, 2, 2, config(8).format))
            .unwrap();
        let current = plan_residency(config(2), &sources, &references, &demand).unwrap();
        check(&current, config(2), &sources);
        assert!(current.rejected_sources.is_empty());
        assert_eq!(
            (current.stats.tail_pages, current.stats.detail_pages),
            (2, 0)
        );
        assert_eq!(current.unmet_demand[0].fallback, Some(key(10, 3, 0, 0, 0)));
    }

    #[test]
    fn remapped_image_needs_a_source_at_its_current_slot_index() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 8, 4, 2);
        let references = BTreeMap::from([(image(10), reference(20).1)]);
        let demand = [key(10, 0, 0, 0, 0), key(20, 0, 0, 0, 0)];
        let plan = plan_residency(config(8), &sources, &references, &demand).unwrap();
        check(&plan, config(8), &sources);
        assert!(plan.residency.pages.is_empty());
        assert_eq!(plan.rejected_sources[0].texture, 20);
        assert_eq!(
            plan.rejected_sources[0].reason,
            NativeVtPlanIssue::MissingSource
        );
        assert_eq!(
            plan.unmet_demand[0].reason,
            NativeVtPlanIssue::UnreferencedSource
        );
        assert_eq!(
            plan.unmet_demand[1].reason,
            NativeVtPlanIssue::MissingSource
        );
    }

    #[test]
    fn invalid_feedback_does_not_promote_or_allocate_sources() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 4, 4, 1);
        add(&mut sources, 20, 4, 4, 1);
        add(&mut sources, 30, 4, 4, 1);
        let references = BTreeMap::from([reference(10), reference(20)]);
        let demand = [
            key(20, 0, 2, 0, 0),
            key(20, 8, 0, 0, 0),
            key(20, 0, 0, 2, 0),
            key(20, 0, 0, 0, 2),
            key(30, 0, 0, 0, 0),
        ];
        let plan = plan_residency(config(1), &sources, &references, &demand).unwrap();
        check(&plan, config(1), &sources);
        assert_eq!(plan.residency.pages[0].0.texture, 10);
        assert_eq!(plan.unmet_demand.len(), demand.len());
        assert!(plan.unmet_demand[..4]
            .iter()
            .all(|unmet| matches!(unmet.reason, NativeVtPlanIssue::InvalidPage(_))));
        assert_eq!(
            plan.unmet_demand[4].reason,
            NativeVtPlanIssue::UnreferencedSource
        );
        assert!(plan
            .unmet_demand
            .iter()
            .all(|unmet| unmet.fallback.is_none()));
    }

    #[test]
    fn missing_mismatched_and_wrong_image_sources_are_diagnosed() {
        let mut sources = NativeVtSources::default();
        for (texture, page_size, format) in [
            (20, 4, MaterialVtFormat::Rgba8Unorm),
            (30, 2, MaterialVtFormat::Rgba8Srgb),
        ] {
            sources
                .insert(
                    image(texture),
                    snapshot(texture, 8, 4, 1, page_size, format),
                )
                .unwrap();
        }
        sources
            .insert(image(400), snapshot(40, 4, 4, 1, 2, config(8).format))
            .unwrap();
        add(&mut sources, 50, 4, 4, 1);
        let references = BTreeMap::from([
            reference(10),
            reference(20),
            reference(30),
            reference(40),
            reference(50),
        ]);
        let demand: Vec<_> = [10, 20, 30, 40]
            .into_iter()
            .map(|id| key(id, 0, 0, 0, 0))
            .collect();
        let plan = plan_residency(config(8), &sources, &references, &demand).unwrap();
        check(&plan, config(8), &sources);
        assert_eq!(plan.stats.admitted_sources, 1);
        let reasons: Vec<_> = plan
            .rejected_sources
            .iter()
            .map(|source| source.reason)
            .collect();
        assert_eq!(
            reasons,
            [
                NativeVtPlanIssue::MissingSource,
                NativeVtPlanIssue::AtlasMismatch,
                NativeVtPlanIssue::AtlasMismatch,
                NativeVtPlanIssue::SourceImageMismatch
            ]
        );
        for (unmet, reason) in plan.unmet_demand.iter().zip(reasons) {
            assert_eq!(unmet.reason, reason);
        }
    }

    #[test]
    fn odd_rectangular_mips_never_create_nonexistent_shifted_parents() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 5, 3, 2);
        let references = BTreeMap::from([reference(10)]);
        let wanted = key(10, 0, 1, 2, 1);
        let plan = plan_residency(config(3), &sources, &references, &[wanted]).unwrap();
        check(&plan, config(3), &sources);
        assert_eq!((plan.stats.tail_pages, plan.stats.detail_pages), (2, 1));
        assert_eq!(plan.stats.satisfied_demand, 1);
        assert!(!plan.residency.pages.iter().any(|(key, _)| key.mip == 1));
    }

    #[test]
    fn unresolvable_tail_sources_are_rejected_without_starving_compatible_sources() {
        let config = NativeVtConfig {
            page_size: 1,
            ..config(8)
        };
        let mut sources = NativeVtSources::default();
        for (texture, width, height) in [(10, 5, 3), (20, 3, 5), (30, 8, 3)] {
            sources
                .insert(
                    image(texture),
                    snapshot(texture, width, height, 2, 1, config.format),
                )
                .unwrap();
        }
        let references = BTreeMap::from([reference(10), reference(20), reference(30)]);
        let edge = key(10, 0, 1, 4, 2);
        // The source accepts this page, but the shared lookup cannot reach its
        // tail by shifting: x=4 >> 2 = 1, while the tail's only valid x is zero.
        sources
            .get(10)
            .unwrap()
            .layout()
            .page_metadata(edge)
            .unwrap();
        let tail_table = GpuPageTable::from_bindings(&[(key(10, 2, 1, 0, 0), 0)]);
        assert!(tail_table.resolve(edge, 2).is_none());
        let demand = [edge, edge, key(20, 0, 0, 2, 4)];
        for feedback in [&[][..], &demand[..]] {
            for budget in [2, 8] {
                let config = NativeVtConfig {
                    max_entries: budget,
                    ..config
                };
                let plan = plan_residency(config, &sources, &references, feedback).unwrap();
                check(&plan, config, &sources);
                assert_eq!(plan.stats.admitted_sources, 1);
                assert_eq!((plan.stats.tail_pages, plan.stats.detail_pages), (2, 0));
                assert!(plan
                    .residency
                    .pages
                    .iter()
                    .all(|(page, _)| page.texture == 30));
                assert_eq!(plan.rejected_sources.len(), 2);
                for (rejected, texture) in plan.rejected_sources.iter().zip([10, 20]) {
                    assert_eq!(rejected.texture, texture);
                    assert_eq!(rejected.reason, NativeVtPlanIssue::UnresolvableTail);
                }
                assert!(plan.unmet_demand.iter().all(|unmet| {
                    unmet.reason == NativeVtPlanIssue::UnresolvableTail && unmet.fallback.is_none()
                }));
                assert_eq!(
                    plan.stats.unique_demand,
                    if feedback.is_empty() { 0 } else { 2 }
                );
            }
        }
    }

    #[test]
    fn tail_coverage_preflight_matches_shared_resolver_for_small_geometries() {
        // Cover both NPOT axes, non-square sources, multi-layer tails and page
        // sizes including one. Metadata preflight must not reject compatible
        // geometries merely because an axis or the page size is not POT.
        for page_size in 1..=4 {
            let config = NativeVtConfig {
                page_size,
                ..config(2)
            };
            for width in 1..=9 {
                for height in 1..=9 {
                    let snapshot = snapshot(10, width, height, 2, page_size, config.format);
                    let coarsest = snapshot.descriptor().mip_count() - 1;
                    let tail_table = GpuPageTable::from_bindings(&[
                        (key(10, coarsest, 0, 0, 0), 0),
                        (key(10, coarsest, 1, 0, 0), 1),
                    ]);
                    let fully_covered = snapshot
                        .page_keys()
                        .all(|page| tail_table.resolve(page, coarsest).is_some());
                    let mut sources = NativeVtSources::default();
                    sources.insert(image(10), snapshot).unwrap();
                    let references = BTreeMap::from([reference(10)]);
                    let plan = plan_residency(config, &sources, &references, &[]).unwrap();
                    assert_eq!(
                        plan.stats.admitted_sources == 1,
                        fully_covered,
                        "{width}x{height}, page_size={page_size}"
                    );
                    check(&plan, config, &sources);
                    if fully_covered {
                        assert!(plan.rejected_sources.is_empty());
                        assert_eq!((plan.stats.tail_pages, plan.stats.detail_pages), (2, 0));
                    } else {
                        assert!(plan.residency.pages.is_empty());
                        assert_eq!(
                            plan.rejected_sources[0].reason,
                            NativeVtPlanIssue::UnresolvableTail
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matching_filterable_formats_and_non_power_of_two_pages_are_supported() {
        for format in [
            MaterialVtFormat::R8Unorm,
            MaterialVtFormat::Rg8Unorm,
            MaterialVtFormat::Rgba8Unorm,
            MaterialVtFormat::Rgba8Srgb,
            MaterialVtFormat::Rgba16Float,
        ] {
            let config = NativeVtConfig {
                page_size: 3,
                format,
                ..config(8)
            };
            let mut sources = NativeVtSources::default();
            sources
                .insert(image(10), snapshot(10, 11, 7, 2, 3, format))
                .unwrap();
            let references = BTreeMap::from([reference(10)]);
            let wanted = key(10, 0, 1, 3, 2);
            let plan = plan_residency(config, &sources, &references, &[wanted]).unwrap();
            check(&plan, config, &sources);
            assert_eq!(plan.stats.tail_pages, 2);
            assert_eq!(plan.stats.satisfied_demand, 1);
        }
    }

    #[test]
    fn zero_table_budget_and_atlas_capacity_are_both_hard_bounds() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 16, 16, 1);
        let references = BTreeMap::from([reference(10)]);
        let demand: Vec<_> = (0..8).map(|x| key(10, 0, 0, x, 0)).collect();
        for entries in [0, 1, 3, 8, u32::MAX] {
            let plan = plan_residency(config(entries), &sources, &references, &demand).unwrap();
            check(&plan, config(entries), &sources);
            assert_eq!(plan.stats.page_budget, entries.min(8));
            if entries == 0 {
                assert!(plan.residency.pages.is_empty());
                assert_eq!(plan.unmet_demand.len(), demand.len());
                assert_eq!(
                    plan.rejected_sources[0].reason,
                    NativeVtPlanIssue::TailBudget {
                        required: 1,
                        available: 0
                    }
                );
            }
        }
    }

    #[test]
    fn maximum_configured_budget_does_not_preallocate_or_overflow_page_accounting() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 4, 4, 2);
        let references = BTreeMap::from([reference(10)]);
        let config = NativeVtConfig {
            tiles_per_row: 1,
            atlas_layers: u32::MAX,
            max_entries: u32::MAX,
            ..config(8)
        };
        // No device is allocated here; real device limits are validated by its owner.
        let plan = plan_residency(config, &sources, &references, &[key(10, 0, 1, 1, 1)]).unwrap();
        check(&plan, config, &sources);
        assert_eq!(plan.stats.page_budget, u32::MAX);
        assert_eq!((plan.stats.tail_pages, plan.stats.detail_pages), (2, 2));
        assert_eq!(plan.residency.pages.len(), 4);
        assert!(plan.unmet_demand.is_empty());
    }

    #[test]
    fn invalid_global_geometry_and_unfilterable_formats_fail_before_planning() {
        let sources = NativeVtSources::default();
        let references = BTreeMap::new();
        let base = config(8);
        let invalid = [
            NativeVtConfig {
                page_size: 0,
                ..base
            },
            NativeVtConfig {
                gutter_px: u32::MAX,
                ..base
            },
            NativeVtConfig {
                tiles_per_row: 0,
                ..base
            },
            NativeVtConfig {
                tiles_per_row: u32::MAX,
                ..base
            },
            NativeVtConfig {
                atlas_layers: 0,
                ..base
            },
            NativeVtConfig {
                atlas_layers: u32::MAX,
                ..base
            },
        ];
        for config in invalid {
            assert!(matches!(
                plan_residency(config, &sources, &references, &[]),
                Err(NativeVtError::InvalidConfig)
            ));
        }
        assert!(matches!(
            plan_residency(
                NativeVtConfig {
                    format: MaterialVtFormat::Rgba32Float,
                    ..base
                },
                &sources,
                &references,
                &[]
            ),
            Err(NativeVtError::UnsupportedFormat)
        ));
    }

    #[test]
    fn duplicate_reference_slots_are_not_ambiguous_admissions() {
        let sources = NativeVtSources::default();
        let slot = reference(10).1;
        // Different generations cannot make one physical index two live sources.
        let references = BTreeMap::from([
            (image(10), slot),
            (
                image(20),
                BindlessSlot {
                    generation: slot.generation + 1,
                    ..slot
                },
            ),
        ]);
        assert!(matches!(
            plan_residency(config(8), &sources, &references, &[]),
            Err(NativeVtError::DuplicateSlot(10))
        ));
    }

    #[test]
    fn repeated_inputs_produce_identical_plans_without_mutating_sources() {
        let mut sources = NativeVtSources::default();
        add(&mut sources, 10, 8, 4, 2);
        let references = BTreeMap::from([reference(10)]);
        let before = sources.get(10).unwrap().revision();
        let demand = [key(10, 0, 1, 3, 1), key(10, 0, 0, 0, 0)];
        let first = plan_residency(config(6), &sources, &references, &demand).unwrap();
        let second = plan_residency(config(6), &sources, &references, &demand).unwrap();
        assert_eq!(first.residency.pages, second.residency.pages);
        assert_eq!(first.stats, second.stats);
        assert_eq!(first.rejected_sources, second.rejected_sources);
        assert_eq!(first.unmet_demand, second.unmet_demand);
        assert_eq!(sources.get(10).unwrap().revision(), before);
        let empty = plan_residency(config(6), &sources, &BTreeMap::new(), &demand).unwrap();
        assert!(empty.residency.pages.is_empty());
        assert_eq!(empty.unmet_demand.len(), demand.len());
    }
}
