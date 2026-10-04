//! Deferred material binning: group visibility-buffer pixels by material so the
//! shading pass dispatches one tight batch per material.
//!
//! After the geometry pass fills a [`VisibilityBuffer`](crate::VisibilityBuffer)
//! with triangle ids, the deferred shading pass must run each material's shader
//! only over the pixels that use it. Binning performs a stable counting sort of
//! the covered pixels keyed by material id, yielding a prefix-sum `offsets`
//! table and a flat `pixels` list. On the GPU this maps onto one indirect
//! dispatch (or draw) per material whose thread range is
//! `offsets[m]..offsets[m + 1]`, eliminating per-pixel material branching and
//! keeping each wavefront coherent.

use crate::VisibilityBuffer;
use alloc::vec;
use alloc::vec::Vec;

/// Covered pixels grouped by material id.
///
/// `pixels` holds linear pixel indices (`y * width + x`) sorted by material.
/// `offsets` is a prefix sum of length `material_count + 1`; material `m` owns
/// `pixels[offsets[m]..offsets[m + 1]]`. `material_ids[m]` records the material
/// id each bin corresponds to (bins are indexed `0..material_count`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialBinning {
    /// Material id of each bin, in bin order.
    pub material_ids: Vec<u32>,
    /// Prefix-sum offsets into `pixels`; length `material_ids.len() + 1`.
    pub offsets: Vec<u32>,
    /// Linear pixel indices, grouped by material and stable within a bin.
    pub pixels: Vec<u32>,
}

impl MaterialBinning {
    /// Number of distinct materials with at least one covered pixel.
    #[must_use]
    pub fn material_count(&self) -> usize {
        self.material_ids.len()
    }

    /// Total number of binned (covered) pixels.
    #[must_use]
    pub fn covered_pixels(&self) -> usize {
        self.pixels.len()
    }

    /// Pixel index slice owned by bin `bin`, or `None` if out of range.
    #[must_use]
    pub fn bin_pixels(&self, bin: usize) -> Option<&[u32]> {
        let start = *self.offsets.get(bin)? as usize;
        let end = *self.offsets.get(bin + 1)? as usize;
        Some(&self.pixels[start..end])
    }

    /// Looks up the bin index for a material id via binary search over the
    /// sorted `material_ids`.
    #[must_use]
    pub fn bin_of_material(&self, material_id: u32) -> Option<usize> {
        self.material_ids.binary_search(&material_id).ok()
    }
}

/// Bins every covered pixel of `vb` by the material returned for its instance.
///
/// `material_of` maps an `instance_id` to its material id. The result lists
/// materials in ascending id order (so GPU dispatch order is deterministic),
/// and pixels within each bin preserve ascending linear-index order (stable),
/// which keeps memory access coherent during shading.
///
/// Runs in `O(n + k log k)` for `n` pixels and `k` distinct materials: two
/// linear passes over the buffer plus a sort of the small material table.
#[must_use]
pub fn bin_pixels_by_material<F>(vb: &VisibilityBuffer, material_of: F) -> MaterialBinning
where
    F: Fn(u32) -> u32,
{
    // First pass: tally pixels per material id, in a sorted map so bin order is
    // deterministic and binary-searchable.
    let samples = vb.samples();
    let mut counts: alloc::collections::BTreeMap<u32, u32> = alloc::collections::BTreeMap::new();
    for &packed in samples {
        if let Some(sample) = crate::VisibilitySample::unpack(packed) {
            let material = material_of(sample.instance_id);
            *counts.entry(material).or_insert(0) += 1;
        }
    }

    // Build the bin table and prefix-sum offsets.
    let material_count = counts.len();
    let mut material_ids = Vec::with_capacity(material_count);
    let mut offsets = Vec::with_capacity(material_count + 1);
    // Map material id -> bin index for the scatter pass.
    let mut bin_of: alloc::collections::BTreeMap<u32, usize> = alloc::collections::BTreeMap::new();
    let mut running = 0u32;
    offsets.push(0);
    for (bin, (&material, &count)) in counts.iter().enumerate() {
        material_ids.push(material);
        bin_of.insert(material, bin);
        running = running.saturating_add(count);
        offsets.push(running);
    }

    // Second pass: scatter pixels into their bins. A per-bin write cursor keeps
    // the counting sort stable (ascending linear index within each bin).
    let total = running as usize;
    let mut pixels = vec![0u32; total];
    let mut cursor: Vec<u32> = offsets[..material_count].to_vec();
    for (linear, &packed) in samples.iter().enumerate() {
        if let Some(sample) = crate::VisibilitySample::unpack(packed) {
            let material = material_of(sample.instance_id);
            let bin = bin_of[&material];
            let slot = cursor[bin] as usize;
            pixels[slot] = u32::try_from(linear).unwrap_or(u32::MAX);
            cursor[bin] += 1;
        }
    }

    MaterialBinning {
        material_ids,
        offsets,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VisibilityBuffer, VisibilitySample};

    /// Builds a buffer where pixel `i` is covered by instance `instances[i]`
    /// (or uncovered when the entry is `None`).
    fn buffer_from(width: u32, height: u32, instances: &[Option<u32>]) -> VisibilityBuffer {
        let mut vb = VisibilityBuffer::new(width, height);
        for (i, inst) in instances.iter().enumerate() {
            if let Some(id) = inst {
                let x = u32::try_from(i).unwrap() % width;
                let y = u32::try_from(i).unwrap() / width;
                vb.set(x, y, VisibilitySample::new(*id, 0));
            }
        }
        vb
    }

    #[test]
    fn empty_buffer_bins_to_nothing() {
        let vb = VisibilityBuffer::new(4, 4);
        let bins = bin_pixels_by_material(&vb, |_| 0);
        assert_eq!(bins.material_count(), 0);
        assert_eq!(bins.covered_pixels(), 0);
        assert_eq!(bins.offsets, vec![0]);
    }

    #[test]
    fn single_material_groups_all_covered_pixels() {
        let vb = buffer_from(2, 2, &[Some(0), Some(1), None, Some(2)]);
        // All instances share material 7.
        let bins = bin_pixels_by_material(&vb, |_| 7);
        assert_eq!(bins.material_ids, vec![7]);
        assert_eq!(bins.offsets, vec![0, 3]);
        // Linear indices 0, 1, 3 covered; stable ascending order.
        assert_eq!(bins.pixels, vec![0, 1, 3]);
    }

    #[test]
    fn multiple_materials_sorted_and_disjoint() {
        // instance->material: 0->5, 1->2, 2->5, 3->2, 4->9.
        let vb = buffer_from(
            3,
            2,
            &[Some(0), Some(1), Some(2), Some(3), Some(4), None],
        );
        let material_of = |inst: u32| match inst {
            0 | 2 => 5,
            1 | 3 => 2,
            _ => 9,
        };
        let bins = bin_pixels_by_material(&vb, material_of);

        // Materials appear in ascending id order.
        assert_eq!(bins.material_ids, vec![2, 5, 9]);
        // Offsets are a monotone prefix sum covering every covered pixel.
        assert_eq!(bins.offsets, vec![0, 2, 4, 5]);
        assert_eq!(bins.covered_pixels(), 5);

        // Bin for material 2 = instances 1,3 at linear indices 1,3.
        assert_eq!(bins.bin_pixels(0), Some(&[1u32, 3][..]));
        // Bin for material 5 = instances 0,2 at linear indices 0,2.
        assert_eq!(bins.bin_pixels(1), Some(&[0u32, 2][..]));
        // Bin for material 9 = instance 4 at linear index 4.
        assert_eq!(bins.bin_pixels(2), Some(&[4u32][..]));
    }

    #[test]
    fn offsets_are_monotone_nondecreasing() {
        let vb = buffer_from(4, 4, &[Some(0); 16]);
        let bins = bin_pixels_by_material(&vb, |inst| inst % 3);
        for w in bins.offsets.windows(2) {
            assert!(w[1] >= w[0]);
        }
        assert_eq!(*bins.offsets.last().unwrap() as usize, bins.covered_pixels());
    }

    #[test]
    fn bin_lookup_by_material_id() {
        let vb = buffer_from(2, 2, &[Some(10), Some(20), Some(30), Some(20)]);
        let bins = bin_pixels_by_material(&vb, |inst| inst);
        assert_eq!(bins.bin_of_material(10), Some(0));
        assert_eq!(bins.bin_of_material(20), Some(1));
        assert_eq!(bins.bin_of_material(30), Some(2));
        assert_eq!(bins.bin_of_material(99), None);
    }

    #[test]
    fn pixels_within_bin_are_stable_ascending() {
        // Many pixels of the same material: order must stay ascending linear.
        let vb = buffer_from(4, 2, &[Some(0); 8]);
        let bins = bin_pixels_by_material(&vb, |_| 1);
        let expected: Vec<u32> = (0..8).collect();
        assert_eq!(bins.pixels, expected);
    }

    #[test]
    fn every_covered_pixel_appears_exactly_once() {
        let instances: Vec<Option<u32>> = (0..16)
            .map(|i| if i % 3 == 0 { None } else { Some(i % 4) })
            .collect();
        let vb = buffer_from(4, 4, &instances);
        let bins = bin_pixels_by_material(&vb, |inst| inst % 2);

        let mut seen = bins.pixels.clone();
        seen.sort_unstable();
        let covered_before: Vec<u32> = (0..16u32)
            .filter(|&i| instances[i as usize].is_some())
            .collect();
        assert_eq!(seen, covered_before);
    }
}
