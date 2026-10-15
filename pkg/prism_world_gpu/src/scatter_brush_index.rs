//! Private, canonical-order brush hierarchy for the GPU upload ABI.

use prism_world_residency::DensityBrush;

use crate::GpuScatterError;

/// Full binary-tree size, checked before reserving the combined upload.
pub(super) fn node_count(brushes: usize) -> Result<u32, GpuScatterError> {
    if brushes == 0 {
        return Ok(0);
    }
    brushes
        .checked_add(brushes - 1)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or(GpuScatterError::ArithmeticOverflow)
}

/// Append a midpoint tree without allocating or changing the brush prefix.
///
/// The caller supplies exactly B canonical 64-byte brush records and capacity
/// for B + `node_count(B)` records, after successful count/size preflight.
/// Nodes are preorder, with closed signed-i64 XYZ bounds encoded as low/high
/// u32 pairs: min in words 0..6, max in 6..12. Word 12 is the exclusive
/// subtree end relative to the beginning of the NODE BLOCK, not the upload
/// or the current node. Word 13 is the canonical brush index for leaves and
/// u32::MAX for internal nodes; words 14 and 15 are zero.
///
/// Contiguous midpoint splits preserve canonical leaf order, including after
/// stackless AABB pruning. Noncommutative brush operations therefore need no
/// per-point scratch or sorting. Bounds are united bottom-up in O(B) time;
/// balanced recursion uses O(log B) stack and no auxiliary heap storage.
pub(super) fn append_nodes(brushes: &[DensityBrush], upload: &mut Vec<[u32; 16]>) {
    debug_assert_eq!(upload.len(), brushes.len());
    debug_assert!(node_count(brushes.len())
        .ok()
        .and_then(|count| brushes.len().checked_add(count as usize))
        .is_some_and(|required| required <= upload.capacity()));
    if !brushes.is_empty() {
        let node_start = upload.len();
        append_subtree(brushes, 0, node_start, upload);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bounds {
    min: [i64; 3],
    max: [i64; 3],
}

impl Bounds {
    fn brush(brush: &DensityBrush) -> Self {
        let radius = i128::from(brush.outer_radius_mm);
        let clamp = |value: i128| value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        Self {
            min: brush.center_mm.map(|c| clamp(i128::from(c) - radius)),
            max: brush.center_mm.map(|c| clamp(i128::from(c) + radius)),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: std::array::from_fn(|axis| self.min[axis].min(other.min[axis])),
            max: std::array::from_fn(|axis| self.max[axis].max(other.max[axis])),
        }
    }
}

fn append_subtree(
    brushes: &[DensityBrush],
    first_brush: usize,
    node_start: usize,
    upload: &mut Vec<[u32; 16]>,
) -> Bounds {
    let slot = upload.len();
    // Reserve this preorder position before visiting children. The caller has
    // already reserved the entire tree, so even these pushes cannot grow it.
    upload.push([0; 16]);
    let (bounds, brush_index) = if brushes.len() == 1 {
        (Bounds::brush(&brushes[0]), first_brush as u32)
    } else {
        let midpoint = brushes.len() / 2;
        let (left, right) = brushes.split_at(midpoint);
        let left_bounds = append_subtree(left, first_brush, node_start, upload);
        let right_bounds = append_subtree(right, first_brush + midpoint, node_start, upload);
        (left_bounds.union(right_bounds), u32::MAX)
    };
    let escape = (upload.len() - node_start) as u32;
    let words = &mut upload[slot];
    for axis in 0..3 {
        let min = bounds.min[axis] as u64;
        let max = bounds.max[axis] as u64;
        words[axis * 2] = min as u32;
        words[axis * 2 + 1] = (min >> 32) as u32;
        words[6 + axis * 2] = max as u32;
        words[7 + axis * 2] = (max >> 32) as u32;
    }
    words[12] = escape;
    words[13] = brush_index;
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_residency::DensityBrushOp;

    fn brush(id: u64, center_mm: [i64; 3], outer_radius_mm: u32) -> DensityBrush {
        DensityBrush {
            id,
            center_mm,
            inner_radius_mm: outer_radius_mm,
            outer_radius_mm,
            opacity: u16::MAX,
            density: DensityBrushOp::Replace(12345),
            species: None,
        }
    }

    fn upload(brushes: &[DensityBrush]) -> Vec<[u32; 16]> {
        let mut records =
            Vec::with_capacity(brushes.len() + node_count(brushes.len()).unwrap() as usize);
        // Distinct opaque brush records make prefix corruption visible.
        records.extend((0..brushes.len()).map(|index| [0xa500_0000 | index as u32; 16]));
        append_nodes(brushes, &mut records);
        records
    }

    fn decode_bounds(words: &[u32; 16]) -> Bounds {
        let signed = |offset: usize| {
            (u64::from(words[offset]) | (u64::from(words[offset + 1]) << 32)) as i64
        };
        Bounds {
            min: std::array::from_fn(|axis| signed(axis * 2)),
            max: std::array::from_fn(|axis| signed(6 + axis * 2)),
        }
    }

    // Independent reference uses saturating i64 arithmetic, not the builder's
    // widened arithmetic, and scans the full interval rather than child unions.
    fn reference_bounds(brushes: &[DensityBrush]) -> Bounds {
        let mut bounds = Bounds {
            min: [i64::MAX; 3],
            max: [i64::MIN; 3],
        };
        for brush in brushes {
            let radius = i64::from(brush.outer_radius_mm);
            for axis in 0..3 {
                bounds.min[axis] =
                    bounds.min[axis].min(brush.center_mm[axis].saturating_sub(radius));
                bounds.max[axis] =
                    bounds.max[axis].max(brush.center_mm[axis].saturating_add(radius));
            }
        }
        bounds
    }

    fn check_subtree(
        nodes: &[[u32; 16]],
        slot: usize,
        brushes: &[DensityBrush],
        first: usize,
    ) -> (usize, usize) {
        let words = &nodes[slot];
        assert_eq!(decode_bounds(words), reference_bounds(brushes));
        assert_eq!(&words[14..], &[0, 0]);
        let (end, depth) = if brushes.len() == 1 {
            assert_eq!(words[13], first as u32);
            (slot + 1, 1)
        } else {
            assert_eq!(words[13], u32::MAX);
            let midpoint = brushes.len() / 2;
            let (right, left_depth) = check_subtree(nodes, slot + 1, &brushes[..midpoint], first);
            let (end, right_depth) =
                check_subtree(nodes, right, &brushes[midpoint..], first + midpoint);
            assert_eq!(
                decode_bounds(words),
                decode_bounds(&nodes[slot + 1]).union(decode_bounds(&nodes[right]))
            );
            (end, 1 + left_depth.max(right_depth))
        };
        assert_eq!(words[12] as usize, end);
        (end, depth)
    }

    fn contains(bounds: Bounds, point: [i64; 3]) -> bool {
        (0..3).all(|axis| bounds.min[axis] <= point[axis] && point[axis] <= bounds.max[axis])
    }

    fn visit(nodes: &[[u32; 16]], point: [i64; 3]) -> Vec<usize> {
        let mut visited = Vec::new();
        let mut slot = 0;
        let mut steps = 0;
        while slot < nodes.len() {
            steps += 1;
            assert!(steps <= nodes.len(), "escape must always advance");
            let words = &nodes[slot];
            if !contains(decode_bounds(words), point) {
                slot = words[12] as usize;
            } else {
                if words[13] != u32::MAX {
                    visited.push(words[13] as usize);
                }
                slot += 1;
            }
        }
        assert_eq!(slot, nodes.len());
        visited
    }

    #[test]
    fn count_checks_empty_normal_gpu_limit_and_usize_overflow() {
        for (brushes, expected) in [(0, 0), (1, 1), (2, 3), (3, 5), (17, 33)] {
            assert_eq!(node_count(brushes), Ok(expected));
        }
        let largest = u64::from(u32::MAX).div_ceil(2);
        if let Ok(largest) = usize::try_from(largest) {
            assert_eq!(node_count(largest), Ok(u32::MAX));
            assert_eq!(
                node_count(largest + 1),
                Err(GpuScatterError::ArithmeticOverflow)
            );
        }
        assert_eq!(
            node_count(usize::MAX),
            Err(GpuScatterError::ArithmeticOverflow)
        );
        assert_eq!(
            node_count(usize::MAX / 2 + 2),
            Err(GpuScatterError::ArithmeticOverflow)
        );
    }

    #[test]
    fn empty_input_is_a_noop_even_without_reserved_storage() {
        let mut records = Vec::new();
        let pointer = records.as_ptr();
        append_nodes(&[], &mut records);
        assert!(records.is_empty());
        assert_eq!(records.capacity(), 0);
        assert_eq!(records.as_ptr(), pointer);
    }

    #[test]
    fn singleton_has_exact_signed_low_high_layout_and_closed_bounds() {
        let records = upload(&[brush(91, [-1, 0x1_0000_0000, -0x1_0000_0000], 1)]);
        assert_eq!(
            records[1],
            [
                0xffff_fffe,
                0xffff_ffff,
                0xffff_ffff,
                0,
                0xffff_ffff,
                0xffff_fffe,
                0,
                0,
                1,
                1,
                1,
                0xffff_ffff,
                1,
                0,
                0,
                0,
            ]
        );
        let nodes = &records[1..];
        assert_eq!(visit(nodes, [-2, 0xffff_ffff, -0x1_0000_0001]), [0]);
        assert_eq!(visit(nodes, [0, 0x1_0000_0001, -0xffff_ffff]), [0]);
        assert!(visit(nodes, [1, 0x1_0000_0000, -0x1_0000_0000]).is_empty());
    }

    #[test]
    fn odd_midpoint_shape_and_escapes_are_node_block_relative() {
        let brushes: Vec<_> = (0..5).map(|i| brush(i, [i as i64; 3], 1)).collect();
        let records = upload(&brushes);
        let nodes = &records[brushes.len()..];
        assert_eq!(
            nodes.iter().map(|words| words[12]).collect::<Vec<_>>(),
            [9, 4, 3, 4, 9, 6, 9, 8, 9]
        );
        assert_eq!(
            nodes.iter().map(|words| words[13]).collect::<Vec<_>>(),
            [u32::MAX, u32::MAX, 0, 1, u32::MAX, 2, u32::MAX, 3, 4]
        );
    }

    #[test]
    fn all_subtrees_have_exact_unions_midpoint_shape_and_canonical_leaves() {
        for count in 1..=257 {
            // Deliberately interleaved spatial positions: no spatial sort is
            // allowed, even when it would make the hierarchy tighter.
            let brushes: Vec<_> = (0..count)
                .map(|i| {
                    brush(
                        i as u64,
                        [((i * 97) % 257) as i64 - 128, -(i as i64), i as i64 * 11],
                        (i % 19) as u32,
                    )
                })
                .collect();
            let records = upload(&brushes);
            let nodes = &records[count..];
            let (end, depth) = check_subtree(nodes, 0, &brushes, 0);
            assert_eq!(end, node_count(count).unwrap() as usize);
            let max_depth = (usize::BITS - (count - 1).leading_zeros()) as usize + 1;
            assert_eq!(depth, max_depth);
            assert_eq!(
                nodes
                    .iter()
                    .filter(|words| words[13] != u32::MAX)
                    .map(|words| words[13] as usize)
                    .collect::<Vec<_>>(),
                (0..count).collect::<Vec<_>>()
            );
            assert_eq!(
                records,
                upload(&brushes),
                "construction must be deterministic"
            );
        }
    }

    #[test]
    fn world_edges_clamp_before_encoding_without_wrapping() {
        let radius = i64::from(u32::MAX);
        let brushes = [
            brush(1, [i64::MIN, i64::MAX, -1], u32::MAX),
            brush(2, [i64::MAX, i64::MIN, 0], u32::MAX),
        ];
        let records = upload(&brushes);
        let nodes = &records[2..];
        assert_eq!(
            decode_bounds(&nodes[1]),
            Bounds {
                min: [i64::MIN, i64::MAX - radius, -1 - radius],
                max: [i64::MIN + radius, i64::MAX, -1 + radius],
            }
        );
        assert_eq!(
            decode_bounds(&nodes[2]),
            Bounds {
                min: [i64::MAX - radius, i64::MIN, -radius],
                max: [i64::MAX, i64::MIN + radius, radius],
            }
        );
        check_subtree(nodes, 0, &brushes, 0);
        assert_eq!(visit(nodes, [i64::MIN, i64::MAX, 0]), [0]);
        assert_eq!(visit(nodes, [i64::MAX, i64::MIN, 0]), [1]);
        let zero = brush(3, [i64::MIN, i64::MAX, -1], 0);
        assert_eq!(
            decode_bounds(&upload(&[zero])[1]),
            Bounds {
                min: zero.center_mm,
                max: zero.center_mm
            }
        );
    }

    #[test]
    fn pruned_stackless_walk_matches_linear_closed_aabb_order_and_composition() {
        let brushes: Vec<_> = (0..37)
            .map(|i| DensityBrush {
                density: if i % 3 == 0 {
                    DensityBrushOp::Replace(i * 1000)
                } else {
                    DensityBrushOp::Add(7919)
                },
                ..brush(
                    i as u64,
                    [((i * 17) % 37) as i64 - 18, 0, 0],
                    (i % 7) as u32,
                )
            })
            .collect();
        let records = upload(&brushes);
        let nodes = &records[brushes.len()..];
        let compose = |order: &[usize]| {
            order
                .iter()
                .fold(123_u16, |density, &index| match brushes[index].density {
                    DensityBrushOp::Replace(value) => value,
                    DensityBrushOp::Add(delta) => {
                        (i32::from(density) + delta).clamp(0, 65535) as u16
                    }
                })
        };
        let mut order_matters = false;
        for x in -26..=26 {
            for y in [-8, -1, 0, 1, 8] {
                let point = [x, y, 0];
                let expected: Vec<_> = brushes
                    .iter()
                    .enumerate()
                    .filter(|(_, brush)| {
                        contains(reference_bounds(std::slice::from_ref(brush)), point)
                    })
                    .map(|(i, _)| i)
                    .collect();
                let visited = visit(nodes, point);
                assert_eq!(visited, expected, "point {point:?}");
                assert_eq!(compose(&visited), compose(&expected));
                let reversed: Vec<_> = expected.iter().copied().rev().collect();
                order_matters |= compose(&expected) != compose(&reversed);
            }
        }
        assert!(
            order_matters,
            "fixture must exercise noncommutative operations"
        );
    }

    #[test]
    fn append_preserves_prefix_pointer_and_capacity_of_reserved_vector() {
        for count in [0, 1, 2, 3, 31, 128, 1025] {
            let brushes: Vec<_> = (0..count)
                .map(|i| brush(i as u64, [i as i64; 3], 7))
                .collect();
            let required = count + node_count(count).unwrap() as usize;
            for spare in [0, 17] {
                let mut records = Vec::with_capacity(required + spare);
                records.extend((0..count).map(|i| [i as u32 ^ 0xdead_beef; 16]));
                let prefix = records.clone();
                let pointer = records.as_ptr();
                let capacity = records.capacity();
                append_nodes(&brushes, &mut records);
                assert_eq!(records.len(), required);
                assert_eq!(records.capacity(), capacity);
                assert_eq!(records.as_ptr(), pointer);
                assert_eq!(&records[..count], prefix.as_slice());
            }
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn debug_rejects_incorrect_brush_prefix_length() {
        append_nodes(&[brush(0, [0; 3], 1)], &mut Vec::with_capacity(2));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn debug_rejects_missing_node_reservation() {
        let mut records = vec![[0; 16]];
        append_nodes(&[brush(0, [0; 3], 1)], &mut records);
    }
}
