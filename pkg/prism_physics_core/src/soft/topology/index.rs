//! Stable root-edge index and incremental physical one-ring connectivity.

use super::{BTreeMap, BTreeSet, TearableMesh, TopologyEdge};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct RootEdge {
    pub edge: TopologyEdge,
    pub triangles: Vec<usize>,
}

impl TearableMesh {
    pub(super) fn build_index(&mut self) {
        self.vertex_triangles = vec![Vec::new(); self.vertex_count()];
        let mut edges: BTreeMap<TopologyEdge, Vec<usize>> = BTreeMap::new();
        for (index, &[a, b, c]) in self.triangles.iter().enumerate() {
            for vertex in [a, b, c] {
                self.vertex_triangles[vertex as usize].push(index);
            }
            for (a, b) in [(a, b), (b, c), (c, a)] {
                edges.entry(self.root_edge(a, b)).or_default().push(index);
            }
        }
        self.root_edges = edges
            .into_iter()
            .map(|(edge, triangles)| RootEdge { edge, triangles })
            .collect();
        self.root_neighbors = vec![Vec::new(); self.vertex_count()];
        for (index, edge) in self.root_edges.iter().enumerate() {
            self.root_neighbors[edge.edge.a as usize].push(index);
            self.root_neighbors[edge.edge.b as usize].push(index);
        }
    }

    pub(super) fn open_seam(&mut self, edge: TopologyEdge) -> bool {
        if !self.seams.insert(edge) {
            return false;
        }
        for vertex in [edge.a, edge.b] {
            let degree = &mut self.seam_degrees[vertex as usize];
            *degree += 1;
            if *degree == 1 {
                self.front_roots.insert(vertex);
            } else {
                self.front_roots.remove(&vertex);
            }
        }
        true
    }

    /// Returns only tip-adjacent edges while a crack is propagating. Finished
    /// seam interiors are never scanned. If all tips are exhausted, a bounded
    /// round-robin initiation pass can discover a separate crack elsewhere.
    pub(super) fn candidate_edges(&mut self, initiation_budget: usize) -> (Vec<usize>, bool) {
        let mut candidates = BTreeSet::new();
        for &root in &self.front_roots {
            for &index in &self.root_neighbors[root as usize] {
                let edge = &self.root_edges[index];
                if edge.triangles.len() == 2 && !self.seams.contains(&edge.edge) {
                    candidates.insert(index);
                }
            }
        }
        if !candidates.is_empty() {
            return (candidates.into_iter().collect(), true);
        }
        let count = self.root_edges.len();
        let budget = initiation_budget.min(count);
        let indices = (0..budget)
            .map(|offset| (self.initiation_cursor + offset) % count)
            .collect();
        if count != 0 {
            self.initiation_cursor = (self.initiation_cursor + budget) % count;
        }
        (indices, false)
    }

    pub(super) fn physical_endpoints(&self, edge_index: usize) -> Option<(u32, u32)> {
        let edge = &self.root_edges[edge_index];
        let triangle = self.triangles[*edge.triangles.first()?];
        let a = *triangle
            .iter()
            .find(|&&v| self.vertex_roots[v as usize] == edge.edge.a)?;
        let b = *triangle
            .iter()
            .find(|&&v| self.vertex_roots[v as usize] == edge.edge.b)?;
        Some((a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initiation_is_bounded_and_round_robin_visits_every_edge() {
        let mut mesh = super::super::tests::square();
        let mut visited = BTreeSet::new();
        for _ in 0..mesh.root_edges.len() {
            let (edges, front) = mesh.candidate_edges(1);
            assert!(!front);
            assert_eq!(edges.len(), 1);
            visited.extend(edges);
        }
        assert_eq!(visited.len(), mesh.root_edges.len());
    }

    #[test]
    fn splitting_updates_only_incident_lists_without_stale_vertex_links() {
        let mut mesh = super::super::tests::square();
        mesh.advance(Default::default());
        assert!(mesh.vertex_count() > 4);
        for (vertex, incident) in mesh.vertex_triangles.iter().enumerate() {
            let expected: Vec<_> = mesh
                .triangles
                .iter()
                .enumerate()
                .filter_map(|(i, triangle)| triangle.contains(&(vertex as u32)).then_some(i))
                .collect();
            assert_eq!(*incident, expected);
        }
    }
}
