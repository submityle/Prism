//! Output-buffer liveness analysis and linear-scan slot allocation.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Every output port
//! produces one logical buffer whose live interval runs from the producer's
//! topological position to the position of its last consumer. A node-ordered
//! linear scan assigns each buffer to a reusable physical slot, and in-place
//! aliasing lets an in-place-capable node overwrite an input buffer whose final
//! consumer is that same node.
//!
//! # Relationship
//!
//! This is the audio analogue of the transient-resource aliasing a modern render
//! graph performs: instead of allocating one buffer per port, the engine
//! allocates a small pool of slots and routes production into them. The pool size
//! and per-buffer slot map feed [`crate::exec_plan::ExecPlan`], which hands the
//! runtime a compact buffer arena.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::graph_desc::{GraphDesc, NodeId};
use crate::topo::position_map;

/// Identifier of a logical output buffer (one per node output port).
pub type BufferId = usize;

/// A logical output buffer and its computed live interval and physical slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferLiveness {
    /// Node that produces this buffer.
    pub producer: NodeId,
    /// Output port on the producer.
    pub port: u32,
    /// Topological position at which the buffer is written (its producer's).
    pub birth: usize,
    /// Topological position of the last consumer, or `birth` if unconsumed.
    pub death: usize,
    /// Physical slot assigned to the buffer by the linear scan.
    pub slot: usize,
}

/// Result of liveness analysis and slot allocation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SlotAllocation {
    buffers: Vec<BufferLiveness>,
    buffer_index: Vec<Vec<BufferId>>,
    pool_size: usize,
    inplace_pairs: Vec<(BufferId, BufferId)>,
}

impl SlotAllocation {
    /// Returns every logical buffer with its interval and assigned slot.
    #[must_use]
    pub fn buffers(&self) -> &[BufferLiveness] {
        &self.buffers
    }

    /// Returns the number of physical slots the runtime must allocate.
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Returns the buffer produced by `node` on `port`, if that port exists.
    #[must_use]
    pub fn buffer_of(&self, node: NodeId, port: u32) -> Option<&BufferLiveness> {
        let id = *self.buffer_index.get(node.0)?.get(port as usize)?;
        self.buffers.get(id)
    }

    /// Returns `(input_buffer, output_buffer)` pairs that were aliased in place.
    ///
    /// For each pair the two buffers share a physical slot: the output overwrites
    /// the input because the producing node was its input's final consumer and is
    /// flagged in-place capable.
    #[must_use]
    pub fn inplace_pairs(&self) -> &[(BufferId, BufferId)] {
        &self.inplace_pairs
    }
}

/// Computes buffer liveness and assigns physical slots.
///
/// `order` must be a topological order of `desc`. The allocator walks nodes in
/// that order; before producing a node's outputs it frees slots whose buffers
/// have already died, optionally aliases a dying input slot for an in-place
/// node, and otherwise draws from the free pool (growing it only when empty).
#[must_use]
pub fn allocate_slots(desc: &GraphDesc, order: &[NodeId]) -> SlotAllocation {
    let n = desc.node_count();
    let pos = position_map(order, n);

    // Enumerate one buffer per output port, in (node, port) order.
    let mut buffers: Vec<BufferLiveness> = Vec::new();
    let mut buffer_index: Vec<Vec<BufferId>> = Vec::with_capacity(n);
    for (node_idx, node_desc) in desc.nodes().iter().enumerate() {
        let mut ports = Vec::with_capacity(node_desc.output_ports as usize);
        for port in 0..node_desc.output_ports {
            let id = buffers.len();
            buffers.push(BufferLiveness {
                producer: NodeId(node_idx),
                port,
                birth: pos[node_idx],
                death: pos[node_idx],
                slot: usize::MAX,
            });
            ports.push(id);
        }
        buffer_index.push(ports);
    }

    // Extend death to the latest consumer; collect the input buffers each node
    // consumes (for in-place aliasing and end-of-step freeing).
    let mut inputs_of: Vec<Vec<BufferId>> = (0..n).map(|_| Vec::new()).collect();
    for edge in desc.edges() {
        let buf = buffer_index[edge.from_node.0][edge.from_port as usize];
        let consumer_pos = pos[edge.to_node.0];
        if consumer_pos > buffers[buf].death {
            buffers[buf].death = consumer_pos;
        }
        inputs_of[edge.to_node.0].push(buf);
    }

    // Node-ordered linear scan.
    let mut slot_death: Vec<usize> = Vec::new();
    let mut slot_free: Vec<bool> = Vec::new();
    let mut inplace_pairs: Vec<(BufferId, BufferId)> = Vec::new();

    for &node in order {
        let p = pos[node.0];

        // Free slots whose buffers died strictly before this step. Inputs whose
        // last consumer is this node (death == p) stay occupied so a non-in-place
        // node never overwrites data it is still reading.
        for s in 0..slot_free.len() {
            if !slot_free[s] && slot_death[s] < p {
                slot_free[s] = true;
            }
        }

        // Candidate input buffers for in-place reuse: those dying exactly now.
        let mut inplace_candidates: Vec<BufferId> = Vec::new();
        if desc.nodes()[node.0].can_process_in_place {
            for &buf in &inputs_of[node.0] {
                if buffers[buf].death == p {
                    inplace_candidates.push(buf);
                }
            }
        }

        let outputs = &buffer_index[node.0];
        let mut used_candidate = 0_usize;
        for &out in outputs {
            let out_death = buffers[out].death;
            if used_candidate < inplace_candidates.len() {
                let in_buf = inplace_candidates[used_candidate];
                used_candidate += 1;
                let s = buffers[in_buf].slot;
                buffers[out].slot = s;
                slot_death[s] = out_death;
                slot_free[s] = false;
                inplace_pairs.push((in_buf, out));
            } else if let Some(s) = (0..slot_free.len()).find(|&i| slot_free[i]) {
                buffers[out].slot = s;
                slot_death[s] = out_death;
                slot_free[s] = false;
            } else {
                let s = slot_free.len();
                slot_free.push(false);
                slot_death.push(out_death);
                buffers[out].slot = s;
            }
        }

        // Release dying-input slots that were not aliased in place this step.
        for &buf in &inputs_of[node.0] {
            if buffers[buf].death == p && !inplace_candidates[..used_candidate].contains(&buf) {
                slot_free[buffers[buf].slot] = true;
            }
        }
    }

    SlotAllocation {
        buffers,
        buffer_index,
        pool_size: slot_free.len(),
        inplace_pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};
    use crate::topo::topological_order;

    fn no_overlap_distinct_slots(alloc: &SlotAllocation) {
        // Any two buffers whose live intervals overlap must use distinct slots,
        // except for an explicit in-place pair that shares a slot by design.
        let bufs = alloc.buffers();
        for i in 0..bufs.len() {
            for j in (i + 1)..bufs.len() {
                let a = &bufs[i];
                let b = &bufs[j];
                let overlap = a.birth <= b.death && b.birth <= a.death;
                if overlap && a.slot == b.slot {
                    let paired = alloc.inplace_pairs().contains(&(i, j))
                        || alloc.inplace_pairs().contains(&(j, i));
                    assert!(paired, "overlapping buffers {i} and {j} share slot {}", a.slot);
                }
            }
        }
    }

    #[test]
    fn chain_without_inplace_uses_two_slots() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, c, 0));
        g.set_master(c);
        let order = topological_order(&g).unwrap();
        let alloc = allocate_slots(&g, &order);
        assert_eq!(alloc.pool_size(), 2);
        no_overlap_distinct_slots(&alloc);
    }

    #[test]
    fn chain_with_inplace_collapses_to_one_slot() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
        let c = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, c, 0));
        g.set_master(c);
        let order = topological_order(&g).unwrap();
        let alloc = allocate_slots(&g, &order);
        assert_eq!(alloc.pool_size(), 1);
        assert_eq!(alloc.inplace_pairs().len(), 2);
        no_overlap_distinct_slots(&alloc);
    }

    #[test]
    fn parallel_branches_need_more_slots() {
        // Two sources both read by a single mixer: the two source buffers are
        // simultaneously live and must occupy separate slots.
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(0, 1));
        let mix = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(a, 0, mix, 0));
        g.connect(EdgeDesc::new(b, 0, mix, 1));
        g.set_master(mix);
        let order = topological_order(&g).unwrap();
        let alloc = allocate_slots(&g, &order);
        assert!(alloc.pool_size() >= 2);
        no_overlap_distinct_slots(&alloc);
    }

    #[test]
    fn fan_out_keeps_producer_live_until_last_consumer() {
        // a feeds b and c; a's buffer must stay live through c (the later node).
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(a, 0, c, 0));
        g.connect(EdgeDesc::new(b, 0, c, 1));
        g.set_master(c);
        let order = topological_order(&g).unwrap();
        let alloc = allocate_slots(&g, &order);
        let a_buf = alloc.buffer_of(a, 0).unwrap();
        let c_pos = order.iter().position(|&x| x == c).unwrap();
        assert_eq!(a_buf.death, c_pos);
        no_overlap_distinct_slots(&alloc);
    }
}
