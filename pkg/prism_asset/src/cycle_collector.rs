//! Server-owned cycle collection for promoted labeled-asset closures.

use crate::handle::{HandleId, UntypedHandle, UntypedWeakHandle};
use crate::id::UntypedAssetId;
use crate::stores::AssetStores;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// Breaks unreachable cycles that contain a promoted label/source pair.
///
/// The retained table is the server's ownership graph. A weak owner index is
/// supplied so an owner that has no incoming retained edge (for example a live
/// external label consumer) can still be recognized as an external root.
pub(crate) fn collect_promoted_cycles(
    retained: &mut BTreeMap<UntypedAssetId, Vec<UntypedHandle>>,
    promoted_labels: &BTreeMap<UntypedAssetId, UntypedAssetId>,
    owner_weaks: &BTreeMap<UntypedAssetId, UntypedWeakHandle>,
    stores: &AssetStores,
) -> bool {
    let graph = retained_graph(retained, stores);
    let mut cycle_nodes = BTreeSet::new();
    for (label, source) in promoted_labels {
        if !reaches(&graph, *source, *label) || !reaches(&graph, *label, *source) {
            continue;
        }
        for node in graph.keys().copied() {
            if reaches(&graph, *source, node) && reaches(&graph, node, *source) {
                cycle_nodes.insert(node);
            }
        }
    }
    if cycle_nodes.is_empty() {
        return false;
    }

    let mut observers = BTreeMap::new();
    for node in graph.keys().copied() {
        let Some(weak) = owner_weaks.get(&node) else {
            continue;
        };
        if let Some(handle) = weak.upgrade() {
            observers.insert(node, handle);
        }
    }

    let retained_counts = retained_counts(retained);
    let observer_counts = observer_counts(&observers);
    let initial_external_root_nodes =
        external_roots(&observers, &retained_counts, &observer_counts);
    if initial_external_root_nodes
        .iter()
        .any(|root| cycle_nodes.iter().any(|node| reaches(&graph, *root, *node)))
    {
        return false;
    }

    // Tombstone every unrooted owner that can reach the candidate cycle. This
    // closes the race where a weak reference upgrades through an owner outside
    // the SCC while the collector is cutting the incoming retained edge.
    let guarded_nodes: BTreeSet<_> = graph
        .keys()
        .copied()
        .filter(|node| {
            cycle_nodes
                .iter()
                .any(|cycle| reaches(&graph, *node, *cycle))
        })
        .collect();
    let mut marked = Vec::new();
    for node in &guarded_nodes {
        if let Some(handle) = observers.get(node) {
            handle.set_retired_for_collection(true);
            marked.push(*node);
        }
    }

    let external_after = external_roots(&observers, &retained_counts, &observer_counts);
    let raced_root = external_after
        .iter()
        .any(|root| guarded_nodes.contains(root));
    if raced_root {
        for node in marked {
            if let Some(handle) = observers.get(&node) {
                handle.set_retired_for_collection(false);
            }
        }
        return false;
    }

    let mut changed = false;
    for owner in &guarded_nodes {
        let Some(edges) = retained.get_mut(owner) else {
            continue;
        };
        if cycle_nodes.contains(owner) {
            changed |= !edges.is_empty();
            edges.clear();
        } else {
            let before = edges.len();
            edges.retain(|handle| !cycle_nodes.contains(&handle.id()));
            changed |= before != edges.len();
        }
    }

    // Non-cycle guards were only needed to close the upgrade window. The
    // candidate handles stay tombstoned until their arena slots are reclaimed.
    for node in marked {
        if !cycle_nodes.contains(&node) {
            if let Some(handle) = observers.get(&node) {
                handle.set_retired_for_collection(false);
            }
        }
    }
    drop(observers);
    changed
}

fn retained_graph(
    retained: &BTreeMap<UntypedAssetId, Vec<UntypedHandle>>,
    stores: &AssetStores,
) -> BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>> {
    let mut graph = BTreeMap::new();
    for (owner, edges) in retained {
        if !stores.contains(*owner) {
            continue;
        }
        let mut outgoing_ids = BTreeSet::new();
        for handle in edges {
            if stores.contains(handle.id()) {
                outgoing_ids.insert(handle.id());
            }
        }
        graph
            .entry(*owner)
            .or_insert_with(BTreeSet::new)
            .extend(outgoing_ids.iter().copied());
        for target in outgoing_ids {
            graph.entry(target).or_insert_with(BTreeSet::new);
        }
    }
    graph
}

fn reaches(
    graph: &BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>>,
    from: UntypedAssetId,
    target: UntypedAssetId,
) -> bool {
    if from == target {
        return true;
    }
    let mut seen = BTreeSet::new();
    let mut pending = vec![from];
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(edges) = graph.get(&node) {
            for next in edges {
                if *next == target {
                    return true;
                }
                pending.push(*next);
            }
        }
    }
    false
}

fn retained_counts(
    retained: &BTreeMap<UntypedAssetId, Vec<UntypedHandle>>,
) -> BTreeMap<HandleId, usize> {
    let mut counts = BTreeMap::new();
    for edges in retained.values() {
        for handle in edges {
            *counts.entry(handle.handle_id()).or_default() += 1;
        }
    }
    counts
}

fn observer_counts(
    observers: &BTreeMap<UntypedAssetId, UntypedHandle>,
) -> BTreeMap<HandleId, usize> {
    let mut counts = BTreeMap::new();
    for handle in observers.values() {
        *counts.entry(handle.handle_id()).or_default() += 1;
    }
    counts
}

fn external_roots(
    observers: &BTreeMap<UntypedAssetId, UntypedHandle>,
    retained_counts: &BTreeMap<HandleId, usize>,
    observer_counts: &BTreeMap<HandleId, usize>,
) -> BTreeSet<UntypedAssetId> {
    observers
        .iter()
        .filter_map(|(node, handle)| {
            let internal = retained_counts
                .get(&handle.handle_id())
                .copied()
                .unwrap_or(0);
            let observed = observer_counts
                .get(&handle.handle_id())
                .copied()
                .unwrap_or(0);
            (handle.strong_count() > internal + observed).then_some(*node)
        })
        .collect()
}
