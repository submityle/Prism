//! Shader Execution Reordering (`SER`) / ray coherence sorting — `CPU` golden.
//!
//! Path tracing diverges: adjacent threads in a `GPU` wavefront bounce toward
//! unrelated materials and directions, serialising shading branches and
//! scattering memory accesses. Hardware Shader-Execution-Reordering (NVIDIA
//! `SER` on Ada / `OptiX`) and the classic *ray sorting* technique both fix
//! this by **reordering** the ray/hit stream so threads shaded together are
//! coherent — same material, similar direction, nearby in space.
//!
//! This module is the deterministic, device-free contract for that reorder:
//!
//! * [`sort_key`] folds the three coherence axes (material → octahedral
//!   direction → spatial `Morton`) into a single [`CoherenceKey`] whose numeric
//!   ordering *is* the coherence ordering.
//! * [`plan`] turns a slice of keys into a stable [`ReorderPlan`]: a permutation
//!   grouping coherent rays into contiguous [`CoherentBatch`] runs, plus
//!   [`ReorderStats`] telemetry.
//!
//! Everything here is integer-deterministic (the only floating-point work is the
//! clamped octahedral map + per-axis normalise), so a `GPU` radix sort can be
//! validated element-by-element against the plan. Per the crate contract, this
//! is `CPU` golden only; the `GPU` twin lives in a dedicated `*_gpu` crate.
//!
//! ```
//! use prism_render_architecture::ray_scene::reorder::{plan_reorder, CoherenceKeyLayout};
//!
//! let layout = CoherenceKeyLayout::balanced([-1.0; 3], [1.0; 3]).unwrap();
//! let keys = [
//!     layout.encode(2, [0.0, 1.0, 0.0], [0.1, 0.1, 0.1]),
//!     layout.encode(1, [1.0, 0.0, 0.0], [0.9, 0.2, 0.3]),
//!     layout.encode(2, [0.0, -1.0, 0.0], [0.8, 0.8, 0.8]),
//! ];
//! // Group purely by material / hit-group id.
//! let plan = plan_reorder(&keys, layout.material_shift());
//! assert_eq!(plan.batches.len(), 2); // material 1 and material 2
//! assert_eq!(plan.order.len(), 3);
//! ```
//!
//! # References
//! - Cigolle et al., *A Survey of Efficient Representations for Independent Unit
//!   Vectors*, JCGT 2014 (octahedral direction encode).
//! - NVIDIA, *Shader Execution Reordering* (Ada / `OptiX`) white paper.
//! - Meister et al., *A Survey on Bounding Volume Hierarchies for Ray Tracing*,
//!   EG 2021 (ray sorting for coherence).

pub mod plan;
pub mod sort_key;

pub use plan::{plan_reorder, CoherentBatch, ReorderPlan, ReorderStats};
pub use sort_key::{
    CoherenceKey, CoherenceKeyLayout, LayoutError, SpatialBounds, KEY_BIT_BUDGET,
    MAX_DIR_BITS_PER_AXIS, MAX_SPATIAL_BITS_PER_AXIS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_materials_group_into_coherent_batches() {
        let layout = CoherenceKeyLayout::balanced([-1.0; 3], [1.0; 3]).unwrap();
        // Interleave two materials; the plan must separate them regardless of
        // input order, and keep each material's rays contiguous.
        let keys: Vec<CoherenceKey> = [
            (3u32, [0.0f32, 1.0, 0.0], [0.1f32, 0.1, 0.1]),
            (1, [1.0, 0.0, 0.0], [0.5, 0.2, 0.9]),
            (3, [0.0, 0.0, 1.0], [0.3, 0.7, 0.2]),
            (1, [-1.0, 0.0, 0.0], [0.8, 0.1, 0.4]),
            (3, [0.0, -1.0, 0.0], [0.6, 0.6, 0.6]),
        ]
        .iter()
        .map(|&(m, d, o)| layout.encode(m, d, o))
        .collect();

        let plan = plan_reorder(&keys, layout.material_shift());
        assert_eq!(plan.batches.len(), 2);
        assert_eq!(plan.batches[0].key_prefix, 1);
        assert_eq!(plan.batches[0].len, 2);
        assert_eq!(plan.batches[1].key_prefix, 3);
        assert_eq!(plan.batches[1].len, 3);
        for b in &plan.batches {
            for &i in plan.batch_indices(b) {
                assert_eq!(keys[i as usize].raw() >> layout.material_shift(), b.key_prefix);
            }
        }
    }
}
