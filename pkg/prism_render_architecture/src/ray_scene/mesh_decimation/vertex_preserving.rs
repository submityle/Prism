//! Endpoint QEM with exact attribute provenance and local topology guards.

use alloc::collections::{BTreeMap, BTreeSet, BinaryHeap};
use core::cmp::Reverse;

use super::{plane_of, Collapse, DecimationError, Decimator, TriangleMesh};

/// A continuous attribute's contribution to collapse ordering.
/// Values are never interpolated: output attributes are gathered using the
/// returned source indices. This channel only penalizes removing detail.
#[derive(Clone, Debug, PartialEq)]
pub struct DecimationChannel {
    /// One finite vector per source vertex; inactive lanes must also be finite.
    pub values: Vec<[f64; 4]>,
    /// Number of active lanes, in `1..=4`.
    pub components: u8,
    /// Nonnegative, finite cost multiplier in object-space squared units.
    pub weight: f64,
}

/// Semantic-neutral restrictions for vertex-subset simplification.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VertexDecimationConstraints {
    /// Empty, or one flag per vertex. A locked vertex cannot be removed or moved.
    pub locked_vertices: Vec<bool>,
    /// Empty, or one equivalence-class ID per vertex. Different classes may
    /// never collapse; callers use this for categorical integer attributes.
    pub discrete_groups: Vec<u32>,
    /// Continuous detail penalties. Does not claim an attribute-error bound.
    pub channels: Vec<DecimationChannel>,
    /// Pin all initial open/nonmanifold edge vertices. Useful for independently
    /// simplified material partitions and split UV/normal charts.
    pub preserve_boundary: bool,
}

/// Simplified geometry plus exact provenance for every output vertex.
#[derive(Clone, Debug, PartialEq)]
pub struct VertexPreservingDecimation {
    /// Geometry containing only referenced, surviving source vertices.
    pub mesh: TriangleMesh,
    /// Output-to-input remap. All authored attribute bytes can be gathered
    /// through this map, including integer IDs and tangent handedness.
    pub source_vertices: Vec<u32>,
}

/// Reduces geometry using source endpoints only, with no nearest-neighbour
/// attribute reconstruction. Link-condition, duplicate-face and orientation
/// checks protect the local surface. Locks and categorical restrictions can
/// stop reduction above the requested target; the actual count is in `mesh`.
/// The algorithm never deliberately undershoots the target. A degenerate-only
/// input produces an empty surface rather than fabricating geometry.
pub fn decimate_preserving_vertices(
    mesh: &TriangleMesh,
    target_triangles: usize,
    constraints: &VertexDecimationConstraints,
) -> Result<VertexPreservingDecimation, DecimationError> {
    if target_triangles == 0 {
        return Err(DecimationError::ZeroTarget);
    }
    validate(mesh, constraints)?;
    let mut state = Decimator::new(mesh);
    // Repeated-index faces are already retired. Remove geometric degenerates
    // too, before constructing boundary locks or the candidate heap.
    for face in 0..state.faces.len() {
        if !state.removed_face[face] && face_normal(&state, state.faces[face]).is_none() {
            state.removed_face[face] = true;
            state.live_faces -= 1;
        }
    }
    let mut locked = if constraints.locked_vertices.is_empty() {
        vec![false; mesh.vertex_count()]
    } else {
        constraints.locked_vertices.clone()
    };
    let mut edges = BTreeMap::<(u32, u32), usize>::new();
    for (face, triangle) in state.faces.iter().enumerate() {
        if state.removed_face[face] {
            continue;
        }
        for (a, b) in triangle_edges(*triangle) {
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    for (&(a, b), &count) in &edges {
        if count > 2 || (count == 1 && constraints.preserve_boundary) {
            locked[a as usize] = true;
            locked[b as usize] = true;
        }
    }
    let mut masses = vec![1.0; mesh.vertex_count()];
    let mut heap = BinaryHeap::<Reverse<Collapse>>::new();
    for &(a, b) in edges.keys() {
        for (keep, drop) in [(a, b), (b, a)] {
            if let Some(candidate) = score(&state, keep, drop, &locked, &masses, constraints) {
                heap.push(Reverse(candidate));
            }
        }
    }
    while state.live_faces > target_triangles {
        let Some(Reverse(candidate)) = heap.pop() else {
            break;
        };
        if !state.is_fresh(&candidate) || !legal(&state, &candidate, target_triangles) {
            continue;
        }
        // Topology validity changes for the complete one-ring, not just the
        // surviving endpoint. Revisit candidates previously rejected there.
        let mut affected = state.neighbours(candidate.keep);
        affected.extend(state.neighbours(candidate.drop));
        affected.push(candidate.keep);
        affected.sort_unstable();
        affected.dedup();
        state.apply(candidate.keep, candidate.drop, candidate.target);
        masses[candidate.keep as usize] += masses[candidate.drop as usize];
        let mut updated_edges = BTreeSet::new();
        for &vertex in &affected {
            if state.removed_vertex[vertex as usize] {
                continue;
            }
            state.version[vertex as usize] += 1;
            // Retired face IDs otherwise accumulate over long simplifications.
            state.incident[vertex as usize].retain(|face| !state.removed_face[*face as usize]);
            for neighbour in state.neighbours(vertex) {
                updated_edges.insert((vertex.min(neighbour), vertex.max(neighbour)));
            }
        }
        for (a, b) in updated_edges {
            for (keep, drop) in [(a, b), (b, a)] {
                if let Some(candidate) = score(&state, keep, drop, &locked, &masses, constraints) {
                    heap.push(Reverse(candidate));
                }
            }
        }
        // Lazy invalidation has bounded storage even for high-valence inputs.
        if heap.len() > state.live_faces.saturating_mul(24).max(1024) {
            heap.retain(|entry| state.is_fresh(&entry.0));
        }
    }
    finish(&state, mesh)
}

fn validate(mesh: &TriangleMesh, c: &VertexDecimationConstraints) -> Result<(), DecimationError> {
    let vertices = mesh.vertex_count();
    if vertices > u32::MAX as usize
        || mesh.triangle_count() > u32::MAX as usize
        || mesh.positions().iter().flatten().any(|v| !v.is_finite())
        || mesh.normals().iter().flatten().any(|v| !v.is_finite())
        || mesh.uvs().iter().flatten().any(|v| !v.is_finite())
    {
        return Err(DecimationError::InvalidConstraints("positions/counts"));
    }
    if (!c.locked_vertices.is_empty() && c.locked_vertices.len() != vertices)
        || (!c.discrete_groups.is_empty() && c.discrete_groups.len() != vertices)
    {
        return Err(DecimationError::InvalidConstraints("vertex count"));
    }
    let mut faces = BTreeSet::new();
    for triangle in mesh.indices() {
        let mut key = *triangle;
        key.sort_unstable();
        if key[0] != key[1] && key[1] != key[2] && !faces.insert(key) {
            return Err(DecimationError::InvalidConstraints("duplicate face"));
        }
    }
    for channel in &c.channels {
        if channel.values.len() != vertices || !(1..=4).contains(&channel.components)
            || !channel.weight.is_finite() || channel.weight < 0.0
            // Bounding the magnitude prevents a finite input from overflowing
            // squared penalties. Source f32/u32 channels fit comfortably.
            || channel.values.iter().flatten().any(|v| !v.is_finite() || v.abs() > f64::from(f32::MAX))
            || channel.weight > f64::from(f32::MAX).powi(2)
        {
            return Err(DecimationError::InvalidConstraints("attribute channel"));
        }
    }
    Ok(())
}

fn score(
    state: &Decimator,
    keep: u32,
    drop: u32,
    locked: &[bool],
    masses: &[f64],
    c: &VertexDecimationConstraints,
) -> Option<Collapse> {
    if locked[drop as usize]
        || (!c.discrete_groups.is_empty()
            && c.discrete_groups[keep as usize] != c.discrete_groups[drop as usize])
    {
        return None;
    }
    let mut q = state.quadrics[keep as usize];
    q.add_assign(&state.quadrics[drop as usize]);
    let penalty: f64 = c
        .channels
        .iter()
        .map(|channel| {
            let va = channel.values[keep as usize];
            let vb = channel.values[drop as usize];
            (0..channel.components as usize)
                .map(|lane| (va[lane] - vb[lane]).powi(2))
                .sum::<f64>()
                * channel.weight
        })
        .sum();
    let cost =
        q.error_at(state.positions[keep as usize]).max(0.0) + penalty * masses[drop as usize];
    cost.is_finite().then_some(Collapse {
        cost,
        keep,
        drop,
        keep_version: state.version[keep as usize],
        drop_version: state.version[drop as usize],
        target: state.positions[keep as usize],
    })
}

fn legal(state: &Decimator, c: &Collapse, target: usize) -> bool {
    let a = state.neighbours(c.keep);
    let b = state.neighbours(c.drop);
    let shared: BTreeSet<_> = a
        .iter()
        .copied()
        .filter(|v| b.binary_search(v).is_ok())
        .collect();
    let faces: BTreeSet<_> = state.incident[c.keep as usize]
        .iter()
        .chain(&state.incident[c.drop as usize])
        .copied()
        .filter(|face| !state.removed_face[*face as usize])
        .collect();
    let mut opposite = BTreeSet::new();
    let mut removed = 0;
    let mut unique = BTreeSet::new();
    for &face in &faces {
        let old = state.faces[face as usize];
        if old.contains(&c.keep) && old.contains(&c.drop) {
            removed += 1;
            opposite.extend(old.into_iter().filter(|v| *v != c.keep && *v != c.drop));
            continue;
        }
        let new = old.map(|v| if v == c.drop { c.keep } else { v });
        let mut canonical = new;
        canonical.sort_unstable();
        if !unique.insert(canonical) {
            return false;
        }
        let (Some(before), Some(after)) = (face_normal(state, old), face_normal(state, new)) else {
            return false;
        };
        // Reject zero area and flips, including near-perpendicular collapses.
        if dot(before, after) <= 1.0e-8 {
            return false;
        }
    }
    (1..=2).contains(&removed)
        && state.live_faces.saturating_sub(removed) >= target
        && shared == opposite
}

fn finish(
    state: &Decimator,
    source: &TriangleMesh,
) -> Result<VertexPreservingDecimation, DecimationError> {
    let mut used = vec![false; state.positions.len()];
    for (face, triangle) in state.faces.iter().enumerate() {
        if !state.removed_face[face] {
            for &vertex in triangle {
                used[vertex as usize] = true;
            }
        }
    }
    let source_vertices: Vec<_> = used
        .iter()
        .enumerate()
        .filter_map(|(i, used)| used.then_some(i as u32))
        .collect();
    let mut remap = vec![u32::MAX; used.len()];
    for (local, &original) in source_vertices.iter().enumerate() {
        remap[original as usize] = local as u32;
    }
    let positions = source_vertices
        .iter()
        .map(|v| source.positions()[*v as usize])
        .collect();
    let normals = if source.has_normals() {
        source_vertices
            .iter()
            .map(|v| source.normals()[*v as usize])
            .collect()
    } else {
        Vec::new()
    };
    let uvs = if source.has_uvs() {
        source_vertices
            .iter()
            .map(|v| source.uvs()[*v as usize])
            .collect()
    } else {
        Vec::new()
    };
    let indices = state
        .faces
        .iter()
        .enumerate()
        .filter(|(f, _)| !state.removed_face[*f])
        .map(|(_, triangle)| triangle.map(|v| remap[v as usize]))
        .collect();
    Ok(VertexPreservingDecimation {
        mesh: TriangleMesh::new(positions, normals, uvs, indices)
            .map_err(DecimationError::Rebuild)?,
        source_vertices,
    })
}

fn triangle_edges([a, b, c]: [u32; 3]) -> [(u32, u32); 3] {
    [(a, b), (b, c), (c, a)]
}
fn face_normal(state: &Decimator, [a, b, c]: [u32; 3]) -> Option<[f64; 3]> {
    let p = state.positions[a as usize];
    let q = state.positions[b as usize];
    let r = state.positions[c as usize];
    let (x, y, z, _) = plane_of(p, q, r)?;
    Some([x, y, z])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests;
