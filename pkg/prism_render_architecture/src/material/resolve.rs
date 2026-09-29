//! Deferred material resolve bucketing for the visibility buffer.
//!
//! Prism draws geometry once into a visibility buffer that stores, per pixel,
//! which [`MaterialRecord`] the surface belongs to. Shading then runs as a
//! screen-space *resolve* pass rather than inside the geometry raster, so the
//! cost of a material is paid only for the pixels it actually covers. Because
//! PBR, NPR, and closure-table (custom) materials evaluate incompatible BRDF
//! and lighting math, they cannot share a single resolve shader; instead each
//! [`MaterialExecutionPath`] gets its own resolve pass over just the materials
//! that need it.
//!
//! This module is the classification layer that makes that split possible. It
//! fans the set of materials referenced by the current frame out into one
//! bucket per execution path, in a single deterministic pass, mirroring the
//! geometry raster's [`RasterBins`](crate::virtual_geometry). The backend then
//! walks the buckets in a fixed order, emitting one resolve pass per non-empty
//! path, each writing into the shared shading targets.
//!
//! # Why this makes "hybrid" a material property, not a separate pipeline
//!
//! A single mesh may carry several sub-materials — some PBR, some NPR, some
//! custom. Each sub-material is its own [`MaterialRecord`], so a hybrid mesh
//! simply contributes several record references that land in different buckets.
//! The resolve passes still write into the same G-buffer and lighting targets,
//! so the halves composite naturally. There is no dedicated "hybrid pipeline":
//! hybrid is what falls out of per-material classification once resolve is
//! deferred and bucketed. PBR, NPR, custom, and hybrid are therefore all
//! first-class citizens of the same architecture.

use alloc::vec::Vec;

use super::{MaterialExecutionPath, MaterialRecord};

/// The materials referenced this frame, partitioned by resolve path.
///
/// Each bucket holds indices into the frame's [`MaterialRecord`] slice, in the
/// order they were first observed, so resolve submission stays deterministic.
/// The backend consumes one bucket at a time and dispatches a single resolve
/// pass per non-empty path: one for fixed PBR, one for fixed NPR, one for the
/// closure-table (custom) evaluator, and one for the diagnostic fallback.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaterialResolveBins {
    /// Records resolved by the fixed physically-based shading path.
    pub fixed_pbr: Vec<u32>,
    /// Records resolved by the fixed non-photorealistic shading path.
    pub fixed_npr: Vec<u32>,
    /// Records resolved by the data-driven closure-table (custom) evaluator.
    pub closure_table: Vec<u32>,
    /// Records routed to the diagnostic fallback (e.g. a missing shader).
    pub diagnostic_fallback: Vec<u32>,
}

impl MaterialResolveBins {
    /// Total number of record references across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.fixed_pbr.len()
            + self.fixed_npr.len()
            + self.closure_table.len()
            + self.diagnostic_fallback.len()
    }

    /// Returns `true` when no material landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fixed_pbr.is_empty()
            && self.fixed_npr.is_empty()
            && self.closure_table.is_empty()
            && self.diagnostic_fallback.is_empty()
    }

    /// Immutable view of the bucket backing a given execution path.
    #[must_use]
    pub fn bucket(&self, path: MaterialExecutionPath) -> &[u32] {
        match path {
            MaterialExecutionPath::FixedPbr => &self.fixed_pbr,
            MaterialExecutionPath::FixedNpr => &self.fixed_npr,
            MaterialExecutionPath::ClosureTable => &self.closure_table,
            MaterialExecutionPath::DiagnosticFallback => &self.diagnostic_fallback,
        }
    }

    /// Appends a record reference to the bucket backing `path`.
    ///
    /// Callers that already know a record's [`MaterialExecutionPath`] — such as
    /// the [`MaterialRegistry`](super::registry::MaterialRegistry) classifier —
    /// use this to route directly without re-inspecting the record.
    pub fn push(&mut self, path: MaterialExecutionPath, record: u32) {
        match path {
            MaterialExecutionPath::FixedPbr => self.fixed_pbr.push(record),
            MaterialExecutionPath::FixedNpr => self.fixed_npr.push(record),
            MaterialExecutionPath::ClosureTable => self.closure_table.push(record),
            MaterialExecutionPath::DiagnosticFallback => self.diagnostic_fallback.push(record),
        }
    }
}

/// The fixed order in which resolve passes are submitted.
///
/// Fixed PBR runs first because it dominates opaque coverage in most scenes,
/// then NPR, then the custom closure evaluator, and finally the diagnostic
/// fallback so any missing-material pixels are painted last and stay visible.
pub const RESOLVE_PASS_ORDER: [MaterialExecutionPath; 4] = [
    MaterialExecutionPath::FixedPbr,
    MaterialExecutionPath::FixedNpr,
    MaterialExecutionPath::ClosureTable,
    MaterialExecutionPath::DiagnosticFallback,
];

/// Partitions the frame's visible material records into per-path resolve
/// buckets.
///
/// `visible` lists indices into `records`, typically the unique material
/// records referenced by the visibility buffer this frame. Each index is routed
/// to the bucket for its record's [`MaterialExecutionPath`]. An index that
/// falls outside `records` is skipped rather than panicking, so a stale
/// visibility list cannot crash resolve submission; the input order is
/// preserved within each bucket.
#[must_use]
pub fn bin_visible_materials(visible: &[u32], records: &[MaterialRecord]) -> MaterialResolveBins {
    let mut bins = MaterialResolveBins::default();
    for &index in visible {
        let Some(record) = records.get(index as usize) else {
            continue;
        };
        bins.push(record.execution, index);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialDomain;
    use crate::shader_package::ShaderPackageId;
    use alloc::string::String;
    use alloc::vec;

    fn record(execution: MaterialExecutionPath) -> MaterialRecord {
        MaterialRecord {
            domain: MaterialDomain::Surface,
            execution,
            parameter_offset: 0,
            shader: None,
        }
    }

    #[test]
    fn routes_each_material_to_its_path() {
        let records = [
            record(MaterialExecutionPath::FixedPbr),
            record(MaterialExecutionPath::FixedNpr),
            record(MaterialExecutionPath::ClosureTable),
            record(MaterialExecutionPath::DiagnosticFallback),
        ];
        let visible = [0, 1, 2, 3];
        let bins = bin_visible_materials(&visible, &records);
        assert_eq!(bins.fixed_pbr, vec![0]);
        assert_eq!(bins.fixed_npr, vec![1]);
        assert_eq!(bins.closure_table, vec![2]);
        assert_eq!(bins.diagnostic_fallback, vec![3]);
        assert_eq!(bins.total(), 4);
    }

    #[test]
    fn hybrid_mesh_fans_out_across_buckets() {
        // One mesh, three sub-materials of different paths, plus a shared PBR.
        let records = [
            record(MaterialExecutionPath::FixedPbr),
            record(MaterialExecutionPath::FixedNpr),
            record(MaterialExecutionPath::ClosureTable),
        ];
        let visible = [0, 1, 2, 0];
        let bins = bin_visible_materials(&visible, &records);
        assert_eq!(bins.fixed_pbr, vec![0, 0]);
        assert_eq!(bins.fixed_npr, vec![1]);
        assert_eq!(bins.closure_table, vec![2]);
        assert_eq!(bins.total(), 4);
    }

    #[test]
    fn preserves_visible_order_within_a_bucket() {
        let records = [record(MaterialExecutionPath::FixedPbr)];
        let visible = [0, 0, 0];
        let bins = bin_visible_materials(&visible, &records);
        assert_eq!(bins.fixed_pbr, vec![0, 0, 0]);
    }

    #[test]
    fn skips_out_of_range_indices() {
        let records = [record(MaterialExecutionPath::FixedPbr)];
        let visible = [0, 7];
        let bins = bin_visible_materials(&visible, &records);
        assert_eq!(bins.total(), 1);
        assert_eq!(bins.fixed_pbr, vec![0]);
    }

    #[test]
    fn empty_visible_set_is_empty() {
        let bins = bin_visible_materials(&[], &[]);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    #[test]
    fn bucket_matches_pass_order_lookup() {
        let records = [
            record(MaterialExecutionPath::FixedNpr),
            record(MaterialExecutionPath::FixedPbr),
        ];
        let bins = bin_visible_materials(&[0, 1], &records);
        assert_eq!(bins.bucket(MaterialExecutionPath::FixedPbr), &[1]);
        assert_eq!(bins.bucket(MaterialExecutionPath::FixedNpr), &[0]);
        assert!(bins.bucket(MaterialExecutionPath::ClosureTable).is_empty());
        // Pass order is stable and covers every path exactly once.
        let mut seen = 0usize;
        for path in RESOLVE_PASS_ORDER {
            seen += bins.bucket(path).len();
        }
        assert_eq!(seen, bins.total());
    }

    #[test]
    fn a_shader_backed_custom_material_routes_to_closure_table() {
        let mut rec = record(MaterialExecutionPath::ClosureTable);
        rec.shader = Some(ShaderPackageId(String::from("custom.pbr")));
        let records = [rec];
        let bins = bin_visible_materials(&[0], &records);
        assert_eq!(bins.closure_table, vec![0]);
    }
}
