//! Lower 期共享上下文：图索引、slot 分配、block 表。

use crate::error::{VsError, VsResult};
use crate::ir::{Block, BlockId, DefId, SlotId, SlotProgram, VarId};
use crate::model::{Endpoint, Graph, NodeId, PinIndex};
use crate::registry::NodeDefinition;
use crate::registry::NodeRegistry;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub(super) struct LowerCtx<'a> {
    pub graph: &'a Graph,
    pub defs_by_node: HashMap<NodeId, Arc<NodeDefinition>>,
    pub def_table: Vec<Arc<NodeDefinition>>,
    pub def_index: HashMap<String, DefId>,
    /// 入边索引（按目标端点 (node, pin) 唯一）。
    pub incoming_by_endpoint: HashMap<(NodeId, PinIndex), Endpoint>,
    /// 出边按 (源节点, 源 pin) 分组。
    pub outgoing_by_endpoint: HashMap<(NodeId, PinIndex), Vec<Endpoint>>,
    /// impure 节点的 data 输出 slot 起始位置。
    pub impure_out_slots: HashMap<NodeId, SlotId>,
    pub next_slot: u32,
    pub blocks: Vec<Block>,
    pub block_for_impure: HashMap<NodeId, BlockId>,
    pub vars: Vec<String>,
    pub entries: BTreeMap<u32, BlockId>,
    pub registry_digest: [u8; 32],
}

impl<'a> LowerCtx<'a> {
    pub fn new(graph: &'a Graph, lib: &dyn NodeRegistry) -> VsResult<Self> {
        let mut defs_by_node = HashMap::new();
        let mut def_table: Vec<Arc<NodeDefinition>> = Vec::new();
        let mut def_index: HashMap<String, DefId> = HashMap::new();

        for n in &graph.nodes {
            if defs_by_node.contains_key(&n.id) {
                return Err(VsError::Validate(format!("duplicate node id {}", n.id)));
            }
            let def = lib.resolve(n)?;
            if !def_index.contains_key(&def.id) {
                def_index.insert(def.id.clone(), def_table.len() as DefId);
                def_table.push(def.clone());
            }
            defs_by_node.insert(n.id, def);
        }

        let mut incoming_by_endpoint: HashMap<(NodeId, PinIndex), Endpoint> = HashMap::new();
        let mut outgoing_by_endpoint: HashMap<(NodeId, PinIndex), Vec<Endpoint>> = HashMap::new();
        for e in &graph.edges {
            if let Some(prev) = incoming_by_endpoint.insert((e.to.node, e.to.pin), e.from) {
                return Err(VsError::Validate(format!(
                    "pin ({}, {}) has multiple incoming edges (from {:?} and {:?})",
                    e.to.node, e.to.pin, prev, e.from
                )));
            }
            outgoing_by_endpoint
                .entry((e.from.node, e.from.pin))
                .or_default()
                .push(e.to);
        }

        let vars: Vec<String> = graph.variables.iter().map(|v| v.name.clone()).collect();

        Ok(Self {
            graph,
            defs_by_node,
            def_table,
            def_index,
            incoming_by_endpoint,
            outgoing_by_endpoint,
            impure_out_slots: HashMap::new(),
            next_slot: 0,
            blocks: Vec::new(),
            block_for_impure: HashMap::new(),
            vars,
            entries: BTreeMap::new(),
            registry_digest: lib.registry_digest(),
        })
    }

    pub fn allocate_impure_outputs(&mut self) -> VsResult<()> {
        let nodes: Vec<NodeId> = self.graph.nodes.iter().map(|n| n.id).collect();
        for nid in nodes {
            if self.defs_by_node[&nid].purity == crate::registry::Purity::Impure {
                let n_outs = u32::try_from(self.defs_by_node[&nid].data_outputs().len())
                    .map_err(|_| VsError::Validate("slot count exceeds u32::MAX".into()))?;
                let start = self.alloc_slots(n_outs)?;
                self.impure_out_slots.insert(nid, start);
            }
        }
        Ok(())
    }

    pub fn alloc_slots(&mut self, n: u32) -> VsResult<SlotId> {
        let start = self.next_slot;
        self.next_slot = start
            .checked_add(n)
            .ok_or_else(|| VsError::Validate("slot count exceeds u32::MAX".into()))?;
        Ok(start)
    }

    pub fn finish(self) -> SlotProgram {
        SlotProgram {
            slots_count: self.next_slot,
            blocks: self.blocks,
            defs: self.def_table,
            vars: self.vars,
            entries: self.entries,
            registry_digest: self.registry_digest,
        }
    }

    /// 让 var_id 列表稳定下来（lower 期不需要重建索引，留给未来 Get/SetVar 节点）。
    #[allow(dead_code)]
    pub fn var_id(&self, name: &str) -> Option<VarId> {
        self.vars.iter().position(|v| v == name).map(|p| p as VarId)
    }
}
