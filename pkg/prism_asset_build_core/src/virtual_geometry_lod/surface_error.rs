//! Deterministic nearest-triangle BVH for the one-sided authored-vertex metric.

use super::StaticVirtualGeometryLodError as Error;

type Point = [f64; 3];

struct Triangle {
    vertices: [Point; 3],
    min: Point,
    max: Point,
    centroid: Point,
    id: usize,
}

struct Node {
    min: Point,
    max: Point,
    kind: Kind,
}

enum Kind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

/// Largest nearest-surface distance over the source vertices. This samples
/// vertices only; it is deliberately not a continuous Hausdorff-error claim.
pub(super) fn measure(
    source: &[[f32; 3]],
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<f32, Error> {
    if source.is_empty() || positions.is_empty() || indices.is_empty() || indices.len() % 3 != 0 {
        return Err(Error::EmptySurface);
    }
    if source
        .iter()
        .chain(positions)
        .flatten()
        .any(|v| !v.is_finite())
        || indices.iter().any(|i| *i as usize >= positions.len())
    {
        return Err(Error::InvalidMetric);
    }
    let mut triangles: Vec<_> = indices
        .chunks_exact(3)
        .enumerate()
        .map(|(id, t)| {
            let vertices = [
                positions[t[0] as usize].map(f64::from),
                positions[t[1] as usize].map(f64::from),
                positions[t[2] as usize].map(f64::from),
            ];
            Triangle {
                min: core::array::from_fn(|axis| {
                    vertices
                        .iter()
                        .map(|v| v[axis])
                        .fold(f64::INFINITY, f64::min)
                }),
                max: core::array::from_fn(|axis| {
                    vertices
                        .iter()
                        .map(|v| v[axis])
                        .fold(f64::NEG_INFINITY, f64::max)
                }),
                centroid: core::array::from_fn(|axis| {
                    (vertices[0][axis] + vertices[1][axis] + vertices[2][axis]) / 3.0
                }),
                vertices,
                id,
            }
        })
        .collect();
    let mut nodes = Vec::with_capacity(triangles.len().saturating_mul(2));
    build(&mut triangles, 0, &mut nodes);
    let mut stack = Vec::with_capacity(64);
    let mut maximum: f64 = 0.0;
    for p in source {
        let p = p.map(f64::from);
        let mut best = f64::INFINITY;
        stack.clear();
        stack.push((0, box_distance(p, nodes[0].min, nodes[0].max)));
        while let Some((index, lower_bound)) = stack.pop() {
            if lower_bound > best {
                continue;
            }
            match nodes[index].kind {
                Kind::Leaf { start, end } => {
                    for triangle in &triangles[start..end] {
                        let [a, b, c] = triangle.vertices;
                        best = best.min(triangle_distance(p, a, b, c));
                    }
                }
                Kind::Branch { left, right } => {
                    let dl = box_distance(p, nodes[left].min, nodes[left].max);
                    let dr = box_distance(p, nodes[right].min, nodes[right].max);
                    let (near, far) = if dl <= dr {
                        ((left, dl), (right, dr))
                    } else {
                        ((right, dr), (left, dl))
                    };
                    // LIFO visits the nearer bound first, tightening pruning.
                    if far.1 <= best {
                        stack.push(far);
                    }
                    if near.1 <= best {
                        stack.push(near);
                    }
                }
            }
        }
        maximum = maximum.max(best);
    }
    upper_f32(maximum.sqrt()).ok_or(Error::InvalidMetric)
}

/// Rounds nonnegative finite distances upward, without hiding overflow.
pub(super) fn upper_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < 0.0 || value > f64::from(f32::MAX) {
        return None;
    }
    let rounded = value as f32;
    Some(if f64::from(rounded) < value {
        f32::from_bits(rounded.to_bits() + 1)
    } else {
        rounded
    })
}

fn build(triangles: &mut [Triangle], start: usize, nodes: &mut Vec<Node>) -> usize {
    let min = core::array::from_fn(|axis| {
        triangles
            .iter()
            .map(|t| t.min[axis])
            .fold(f64::INFINITY, f64::min)
    });
    let max = core::array::from_fn(|axis| {
        triangles
            .iter()
            .map(|t| t.max[axis])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let index = nodes.len();
    nodes.push(Node {
        min,
        max,
        kind: Kind::Leaf {
            start,
            end: start + triangles.len(),
        },
    });
    if triangles.len() > 8 {
        let axis = (0..3)
            .max_by(|a, b| {
                (max[*a] - min[*a])
                    .total_cmp(&(max[*b] - min[*b]))
                    .then_with(|| b.cmp(a))
            })
            .unwrap();
        let midpoint = triangles.len() / 2;
        triangles.select_nth_unstable_by(midpoint, |a, b| {
            a.centroid[axis]
                .total_cmp(&b.centroid[axis])
                .then_with(|| a.id.cmp(&b.id))
        });
        let (left, right) = triangles.split_at_mut(midpoint);
        let left = build(left, start, nodes);
        let right = build(right, start + midpoint, nodes);
        nodes[index].kind = Kind::Branch { left, right };
    }
    index
}

fn box_distance(p: Point, min: Point, max: Point) -> f64 {
    (0..3)
        .map(|axis| {
            (min[axis] - p[axis])
                .max(0.0)
                .max(p[axis] - max[axis])
                .powi(2)
        })
        .sum()
}

fn triangle_distance(p: Point, a: Point, b: Point, c: Point) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let normal = cross(ab, ac);
    let area = dot(normal, normal);
    if area == 0.0 {
        return segment_distance(p, a, b)
            .min(segment_distance(p, a, c))
            .min(segment_distance(p, b, c));
    }
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return segment_distance(p, a, b);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return segment_distance(p, a, c);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return segment_distance(p, b, c);
    }
    dot(ap, normal).powi(2) / area
}

fn segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let edge = sub(b, a);
    let length = dot(edge, edge);
    let t = if length > 0.0 {
        (dot(sub(p, a), edge) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let delta = sub(p, core::array::from_fn(|axis| a[axis] + edge[axis] * t));
    dot(delta, delta)
}

fn sub(a: Point, b: Point) -> Point {
    core::array::from_fn(|axis| a[axis] - b[axis])
}
fn dot(a: Point, b: Point) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}
fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvh_matches_brute_force_and_is_repeatable() {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for i in 0..137 {
            let x = (i % 13) as f32;
            let y = (i / 13) as f32;
            let base = positions.len() as u32;
            positions.extend([[x, y, 0.0], [x + 0.8, y, 0.1], [x, y + 0.7, -0.1]]);
            indices.extend([base, base + 1, base + 2]);
        }
        let source: Vec<_> = (0..71)
            .map(|i| {
                [
                    (i % 17) as f32 * 0.75 - 1.0,
                    (i % 11) as f32 * 1.2,
                    i as f32 * 0.013,
                ]
            })
            .collect();
        let brute = source
            .iter()
            .map(|p| {
                indices
                    .chunks_exact(3)
                    .map(|t| {
                        triangle_distance(
                            p.map(f64::from),
                            positions[t[0] as usize].map(f64::from),
                            positions[t[1] as usize].map(f64::from),
                            positions[t[2] as usize].map(f64::from),
                        )
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
            .sqrt();
        let expected = upper_f32(brute).unwrap();
        for _ in 0..4 {
            assert_eq!(measure(&source, &positions, &indices), Ok(expected));
        }
    }

    #[test]
    fn distances_do_not_disappear_at_tiny_scales_or_overflow_at_large_scales() {
        for scale in [1.0e-30_f32, 1.0, 1.0e30] {
            let positions = [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0]];
            let p = [[scale * 0.25, scale * 0.25, scale]];
            let error = measure(&p, &positions, &[0, 1, 2]).unwrap();
            assert!(error >= scale && error <= scale * (1.0 + f32::EPSILON));
        }
        let tiny = f32::from_bits(1);
        assert_eq!(
            measure(&[[0.0, 0.0, tiny]], &[[0.0; 3]], &[0, 0, 0]),
            Ok(tiny)
        );
    }

    #[test]
    fn degenerate_faces_use_segments_and_invalid_metrics_are_rejected() {
        assert_eq!(
            measure(&[[0.5, 1.0, 0.0]], &[[0.0; 3], [1.0, 0.0, 0.0]], &[0, 1, 1]),
            Ok(1.0)
        );
        assert_eq!(measure(&[[0.0; 3]], &[], &[]), Err(Error::EmptySurface));
        assert_eq!(
            measure(&[[0.0; 3]], &[[0.0; 3]], &[0, 1, 0]),
            Err(Error::InvalidMetric)
        );
        assert_eq!(
            measure(&[[f32::NAN; 3]], &[[0.0; 3]], &[0, 0, 0]),
            Err(Error::InvalidMetric)
        );
        assert_eq!(upper_f32(f64::from(f32::MAX) * 2.0), None);
        assert_eq!(upper_f32(-1.0), None);
    }
}
