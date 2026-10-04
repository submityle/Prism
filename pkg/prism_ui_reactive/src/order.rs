//! Deterministic evaluation ordering over a reactive [`GraphSnapshot`].
//!
//! The reactive runtime pulls values lazily, but tooling frequently needs an
//! explicit, stable order in which nodes *would* be evaluated: every node after
//! all of its dependencies. This module derives two such views purely from a
//! captured snapshot, without ever touching or driving the runtime.
//!
//! * [`GraphSnapshot::topological_order`] — a single linear order in which every
//!   node follows all of its `sources`.
//! * [`GraphSnapshot::update_layers`] — the same ordering grouped into depth
//!   layers, where a node in layer `k` depends only on nodes in earlier layers,
//!   so each layer could in principle be recomputed as one wavefront.
//!
//! Both follow the `sources` edges (a dependency precedes its dependents),
//! ignore dangling edges to absent nodes exactly as [`GraphSnapshot::has_cycle`]
//! does, and return `None` for a cyclic graph. Output is deterministic: ties are
//! broken by ascending [`NodeId`], so the same graph always yields identical
//! results.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::introspect::GraphSnapshot;
use crate::NodeId;

impl GraphSnapshot {
    /// Returns a deterministic topological order of the graph's live nodes:
    /// every node appears after all of its `sources`.
    ///
    /// Returns `None` when the `sources` edges contain a directed cycle —
    /// exactly the graphs for which [`GraphSnapshot::has_cycle`] is `true`.
    ///
    /// Ties are broken by ascending [`NodeId`], so the same graph always yields
    /// the same order. Dangling edges to absent nodes are ignored, mirroring
    /// [`GraphSnapshot::has_cycle`]. The cost is linear in the number of nodes
    /// plus edges.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_ui_reactive::Runtime;
    ///
    /// let rt = Runtime::new();
    /// let a = rt.signal(1i32);
    /// let m = rt.memo({
    ///     let a = a.clone();
    ///     move || a.get() + 1
    /// });
    /// let _ = m.get();
    ///
    /// let graph = rt.graph_snapshot();
    /// let order = graph.topological_order().unwrap();
    /// let a_id = graph.signals()[0].id;
    /// let m_id = graph.memos()[0].id;
    /// let pos = |id| order.iter().position(|&x| x == id).unwrap();
    /// // The signal is evaluated before the memo that reads it.
    /// assert!(pos(a_id) < pos(m_id));
    /// ```
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<NodeId>> {
        let present: BTreeSet<NodeId> = self.nodes.iter().map(|node| node.id).collect();

        // In-degree over distinct, present dependency edges, plus the reverse
        // adjacency used to relax successors. A self-edge is counted so that a
        // self-dependent node is correctly reported as cyclic.
        let mut in_degree: BTreeMap<NodeId, usize> =
            present.iter().map(|&id| (id, 0usize)).collect();
        let mut dependents: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for node in &self.nodes {
            let mut seen = BTreeSet::new();
            for &source in &node.sources {
                if !present.contains(&source) || !seen.insert(source) {
                    continue;
                }
                *in_degree.entry(node.id).or_insert(0) += 1;
                dependents.entry(source).or_default().push(node.id);
            }
        }

        // Kahn's algorithm; always take the smallest ready id for determinism.
        let mut ready: BTreeSet<NodeId> = in_degree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::with_capacity(present.len());
        while let Some(&id) = ready.iter().next() {
            ready.remove(&id);
            order.push(id);
            if let Some(children) = dependents.get(&id) {
                for &child in children {
                    if let Some(degree) = in_degree.get_mut(&child) {
                        *degree -= 1;
                        if *degree == 0 {
                            ready.insert(child);
                        }
                    }
                }
            }
        }

        if order.len() == present.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Returns the graph's nodes grouped into dependency depth layers.
    ///
    /// Layer `0` holds every node with no (present) `sources`; a node sits in
    /// layer `1 + max(layer(source))` over its present sources. Every node in a
    /// layer therefore depends only on nodes in earlier layers, so a layer forms
    /// a wavefront that could be recomputed together. Within each layer, ids are
    /// sorted ascending.
    ///
    /// Returns `None` for a cyclic graph (the same condition as
    /// [`GraphSnapshot::topological_order`]). An empty graph yields an empty
    /// list of layers.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_ui_reactive::Runtime;
    ///
    /// let rt = Runtime::new();
    /// let a = rt.signal(1i32);
    /// let m = rt.memo({
    ///     let a = a.clone();
    ///     move || a.get() + 1
    /// });
    /// let _ = m.get();
    ///
    /// let graph = rt.graph_snapshot();
    /// let a_id = graph.signals()[0].id;
    /// let m_id = graph.memos()[0].id;
    /// let layers = graph.update_layers().unwrap();
    /// assert_eq!(layers, vec![vec![a_id], vec![m_id]]);
    /// ```
    #[must_use]
    pub fn update_layers(&self) -> Option<Vec<Vec<NodeId>>> {
        let order = self.topological_order()?;
        if order.is_empty() {
            return Some(Vec::new());
        }
        let present: BTreeSet<NodeId> = order.iter().copied().collect();

        // Distinct, present dependency edges per node. The graph is acyclic here
        // (otherwise `topological_order` returned `None`), so no self-edges.
        let sources_of: BTreeMap<NodeId, Vec<NodeId>> = self
            .nodes
            .iter()
            .map(|node| {
                let mut sources: Vec<NodeId> = node
                    .sources
                    .iter()
                    .copied()
                    .filter(|source| present.contains(source))
                    .collect();
                sources.sort_unstable();
                sources.dedup();
                (node.id, sources)
            })
            .collect();

        let mut layer: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut max_layer = 0usize;
        for &id in &order {
            let depth = sources_of
                .get(&id)
                .map(|sources| sources.iter().map(|source| layer[source] + 1).max().unwrap_or(0))
                .unwrap_or(0);
            layer.insert(id, depth);
            max_layer = max_layer.max(depth);
        }

        let mut layers: Vec<Vec<NodeId>> = (0..=max_layer).map(|_| Vec::new()).collect();
        for (&id, &depth) in &layer {
            layers[depth].push(id);
        }
        for level in &mut layers {
            level.sort_unstable();
        }
        Some(layers)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::collections::{BTreeMap, BTreeSet};
    use alloc::vec::Vec;

    use crate::introspect::{GraphSnapshot, NodeInfo, NodeKindInfo};
    use crate::NodeId;

    fn node(id: NodeId, sources: &[NodeId]) -> NodeInfo {
        let mut sources = sources.to_vec();
        sources.sort_unstable();
        NodeInfo {
            id,
            kind: NodeKindInfo::Signal,
            sources,
            observers: Vec::new(),
        }
    }

    fn graph(nodes: Vec<NodeInfo>) -> GraphSnapshot {
        GraphSnapshot { nodes }
    }

    /// Independent oracle: `order` lists every live node exactly once, and every
    /// present dependency of a node appears strictly before it.
    fn is_valid_topo(g: &GraphSnapshot, order: &[NodeId]) -> bool {
        let present: BTreeSet<NodeId> = g.nodes.iter().map(|n| n.id).collect();
        if order.len() != present.len() {
            return false;
        }
        let mut pos: BTreeMap<NodeId, usize> = BTreeMap::new();
        for (i, &id) in order.iter().enumerate() {
            if pos.insert(id, i).is_some() {
                return false; // duplicate id in the order
            }
        }
        if !present.iter().all(|id| pos.contains_key(id)) {
            return false;
        }
        for n in &g.nodes {
            for &s in &n.sources {
                if !present.contains(&s) {
                    continue;
                }
                if pos[&s] >= pos[&n.id] {
                    return false;
                }
            }
        }
        true
    }

    /// Independent oracle for the layering: flattening recovers exactly the live
    /// node set, every layer is non-empty and ascending, and each node's layer
    /// equals `1 + max` over the layers of its distinct present sources (`0` with
    /// no such sources).
    fn check_layers(g: &GraphSnapshot, layers: &[Vec<NodeId>]) {
        let present: BTreeSet<NodeId> = g.nodes.iter().map(|n| n.id).collect();
        let mut layer_of: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut flat: BTreeSet<NodeId> = BTreeSet::new();
        for (lvl, ids) in layers.iter().enumerate() {
            assert!(!ids.is_empty(), "layer {lvl} is empty");
            assert!(ids.windows(2).all(|w| w[0] < w[1]), "layer not sorted/unique");
            for &id in ids {
                layer_of.insert(id, lvl);
                assert!(flat.insert(id), "id {id} appeared in two layers");
            }
        }
        assert_eq!(flat, present, "layers must cover exactly the live nodes");
        for n in &g.nodes {
            let mut sources: Vec<NodeId> = n
                .sources
                .iter()
                .copied()
                .filter(|s| present.contains(s))
                .collect();
            sources.sort_unstable();
            sources.dedup();
            let expected = sources
                .iter()
                .map(|s| layer_of[s] + 1)
                .max()
                .unwrap_or(0);
            assert_eq!(layer_of[&n.id], expected, "wrong depth for node {}", n.id);
        }
    }

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, hi: usize) -> usize {
            (self.next_u64() % hi as u64) as usize
        }
    }

    #[test]
    fn empty_graph_is_empty_order_and_layers() {
        let g = graph(Vec::new());
        assert_eq!(g.topological_order(), Some(Vec::new()));
        assert_eq!(g.update_layers(), Some(Vec::new()));
    }

    #[test]
    fn single_root_node() {
        let g = graph([node(0, &[])].into());
        assert_eq!(g.topological_order(), Some([0].into()));
        assert_eq!(g.update_layers(), Some([[0].into()].into()));
    }

    #[test]
    fn chain_orders_dependencies_first() {
        // 2 -> 1 -> 0 (child lists its parent as a source).
        let g = graph([node(0, &[]), node(1, &[0]), node(2, &[1])].into());
        assert_eq!(g.topological_order(), Some([0, 1, 2].into()));
        assert_eq!(
            g.update_layers(),
            Some([[0].into(), [1].into(), [2].into()].into())
        );
    }

    #[test]
    fn diamond_layers() {
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let g = graph([node(0, &[]), node(1, &[0]), node(2, &[0]), node(3, &[1, 2])].into());
        let order = g.topological_order().unwrap();
        assert!(is_valid_topo(&g, &order));
        let layers = g.update_layers().unwrap();
        assert_eq!(layers, vec![vec![0], vec![1, 2], vec![3]]);
        check_layers(&g, &layers);
    }

    #[test]
    fn non_contiguous_ids() {
        let g = graph([node(9, &[5]), node(2, &[]), node(5, &[2])].into());
        assert_eq!(g.topological_order(), Some([2, 5, 9].into()));
        assert_eq!(
            g.update_layers(),
            Some([[2].into(), [5].into(), [9].into()].into())
        );
    }

    #[test]
    fn self_loop_is_cyclic() {
        let g = graph([node(0, &[0])].into());
        assert!(g.has_cycle());
        assert_eq!(g.topological_order(), None);
        assert_eq!(g.update_layers(), None);
    }

    #[test]
    fn two_cycle_is_rejected() {
        let g = graph([node(0, &[1]), node(1, &[0])].into());
        assert!(g.has_cycle());
        assert_eq!(g.topological_order(), None);
        assert_eq!(g.update_layers(), None);
    }

    #[test]
    fn dangling_source_is_ignored() {
        // 99 is not a node in the graph, so node 0 is treated as a root.
        let g = graph([node(0, &[99])].into());
        assert!(!g.has_cycle());
        assert_eq!(g.topological_order(), Some([0].into()));
        assert_eq!(g.update_layers(), Some([[0].into()].into()));
    }

    #[test]
    fn deterministic_tie_break_by_id() {
        // Three independent roots must come out in ascending id order.
        let g = graph([node(7, &[]), node(3, &[]), node(5, &[])].into());
        assert_eq!(g.topological_order(), Some([3, 5, 7].into()));
        assert_eq!(g.update_layers(), Some([[3, 5, 7].into()].into()));
    }

    #[test]
    fn randomized_acyclic_graphs_satisfy_oracles() {
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..1000 {
            let n = rng.below(13); // 0..=12 nodes with ids 0..n
            let mut nodes = Vec::new();
            for id in 0..n {
                // Only depend on strictly smaller ids -> guaranteed acyclic.
                let mut sources = Vec::new();
                for candidate in 0..id {
                    if rng.below(3) == 0 {
                        sources.push(candidate);
                    }
                }
                nodes.push(node(id, &sources));
            }
            let g = graph(nodes);
            assert!(!g.has_cycle());
            let order = g.topological_order().expect("acyclic graph must order");
            assert!(is_valid_topo(&g, &order));
            let layers = g.update_layers().expect("acyclic graph must layer");
            check_layers(&g, &layers);
            // The flattened layer order is itself a valid topological order.
            let flat: Vec<NodeId> = layers.iter().flatten().copied().collect();
            assert!(is_valid_topo(&g, &flat));
        }
    }

    #[test]
    fn randomized_graphs_match_has_cycle() {
        let mut rng = SplitMix64(0x0FED_CBA9_8765_4321);
        for _ in 0..1000 {
            let n = 1 + rng.below(8); // 1..=8 nodes
            let mut nodes = Vec::new();
            for id in 0..n {
                // Arbitrary edges among all nodes -> may form cycles.
                let mut sources = Vec::new();
                for candidate in 0..n {
                    if candidate != id && rng.below(4) == 0 {
                        sources.push(candidate);
                    }
                }
                nodes.push(node(id, &sources));
            }
            let g = graph(nodes);
            let acyclic = !g.has_cycle();
            let order = g.topological_order();
            assert_eq!(order.is_some(), acyclic);
            assert_eq!(g.update_layers().is_some(), acyclic);
            if let Some(order) = order {
                assert!(is_valid_topo(&g, &order));
                check_layers(&g, &g.update_layers().unwrap());
            }
        }
    }
}
