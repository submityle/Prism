//! Lower：把 Graph 翻译为 [`SlotProgram`]。
//!
//! 策略：每个 impure 节点一个 Block；进入 block 时反向递归 emit 上游 pure 节点。

use super::context::LowerCtx;
use crate::error::VsResult;
use crate::ir::{Block, BlockId, Instr, SlotId, SlotProgram};
use crate::model::{ControlNode, Endpoint, NodeId, Pin, PinIndex};
use crate::registry::Purity;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(super) fn run(cx: &mut LowerCtx<'_>) -> VsResult<()> {
    let impure: Vec<NodeId> = cx
        .graph
        .nodes
        .iter()
        .filter(|n| cx.defs_by_node[&n.id].purity == Purity::Impure)
        .map(|n| n.id)
        .collect();

    // 每个 impure 节点分配一个 block
    for nid in &impure {
        let bid = cx.blocks.len() as BlockId;
        cx.blocks.push(Block::default());
        cx.block_for_impure.insert(*nid, bid);
    }

    register_entries(cx, &impure);

    for nid in impure {
        lower_impure_block(cx, nid)?;
    }
    Ok(())
}

fn register_entries(cx: &mut LowerCtx<'_>, impure: &[NodeId]) {
    let mut entries: BTreeSet<NodeId> = cx.graph.entries.iter().copied().collect();
    // 自动识别：是 event 节点且 exec_in 没有连入边 → 视为入口。
    for &nid in impure {
        let def = &cx.defs_by_node[&nid];
        if !def.is_event() {
            continue;
        }
        let has_exec_in = def
            .inputs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_exec())
            .any(|(i, _)| cx.incoming_by_endpoint.contains_key(&(nid, i as PinIndex)));
        if !has_exec_in {
            entries.insert(nid);
        }
    }
    for nid in entries {
        if let Some(&bid) = cx.block_for_impure.get(&nid) {
            cx.entries.insert(nid, bid);
        }
    }
}

fn lower_impure_block(cx: &mut LowerCtx<'_>, node_id: NodeId) -> VsResult<()> {
    let bid = cx.block_for_impure[&node_id];
    let mut block = std::mem::take(&mut cx.blocks[bid as usize]);
    let mut pure_emitted: HashMap<NodeId, SlotId> = HashMap::new();

    let in_range = emit_inputs(cx, node_id, &mut block, &mut pure_emitted)?;
    let out_start = cx.impure_out_slots[&node_id];
    let out_count = u32::try_from(cx.defs_by_node[&node_id].data_outputs().len())
        .map_err(|_| crate::error::VsError::Validate("slot count exceeds u32::MAX".into()))?;
    let out_end = out_start
        .checked_add(out_count)
        .ok_or_else(|| crate::error::VsError::Validate("slot count exceeds u32::MAX".into()))?;
    let out_range = out_start..out_end;

    let def = cx.defs_by_node[&node_id].clone();
    let mut targets: Vec<(String, BlockId)> = Vec::new();
    let mut on_end: BlockId = SlotProgram::HALT;
    let exec_out_count = def.outputs.iter().filter(|p| p.is_exec()).count();
    for (i, p) in def.outputs.iter().enumerate() {
        if !p.is_exec() {
            continue;
        }
        if let Some(eps) = cx.outgoing_by_endpoint.get(&(node_id, i as PinIndex))
            && let Some(ep) = eps.first()
        {
            let tgt = cx
                .block_for_impure
                .get(&ep.node)
                .copied()
                .unwrap_or(SlotProgram::HALT);
            targets.push((p.name.clone(), tgt));
        }
    }
    if exec_out_count == 1
        && let Some((_, t)) = targets.first()
    {
        on_end = *t;
    }
    let error_target = def
        .outputs
        .iter()
        .position(|pin| pin.is_exec() && pin.name == "error")
        .and_then(|pin_index| {
            cx.outgoing_by_endpoint
                .get(&(node_id, pin_index as PinIndex))
                .and_then(|eps| eps.first())
        })
        .and_then(|ep| cx.block_for_impure.get(&ep.node).copied());
    let config = canonical_config(cx, node_id);

    let target = targets
        .iter()
        .find(|(name, _)| name == "then")
        .map(|(_, target)| *target)
        .unwrap_or(on_end);
    match cx.graph.controls.get(&node_id) {
        Some(ControlNode::FrameEntry) => block.instrs.push(Instr::FrameEntry {
            node_id,
            outputs: out_range,
            target,
        }),
        Some(ControlNode::Return) => block.instrs.push(Instr::Return {
            node_id,
            inputs: in_range,
        }),
        Some(ControlNode::Call { target: graph }) => block.instrs.push(Instr::CallGraph {
            node_id,
            target_graph: graph.clone(),
            inputs: in_range,
            outputs: out_range,
            target,
            error_target,
        }),
        Some(ControlNode::Fork {
            branches,
            policy,
            max_concurrency,
        }) => block.instrs.push(Instr::ForkGraphs {
            node_id,
            branches: branches.clone(),
            policy: *policy,
            max_concurrency: *max_concurrency,
            inputs: in_range,
            outputs: out_range,
            target,
            error_target,
        }),
        Some(ControlNode::Transition { state }) => block.instrs.push(Instr::Transition {
            node_id,
            state: state.clone(),
        }),
        None => {
            let def_id = cx.def_index[&def.id];
            block.instrs.push(Instr::CallImpure {
                def: def_id,
                node_id,
                inputs: in_range,
                outputs: out_range,
                config,
                targets,
                on_end,
                error_target,
            });
        }
    }
    cx.blocks[bid as usize] = block;
    Ok(())
}

/// 为指定节点的 data 输入构造 slot range，写入填充指令。
fn emit_inputs(
    cx: &mut LowerCtx<'_>,
    node_id: NodeId,
    block: &mut Block,
    pure_emitted: &mut HashMap<NodeId, SlotId>,
) -> VsResult<std::ops::Range<SlotId>> {
    let def = cx.defs_by_node[&node_id].clone();
    let data_inputs = def.data_inputs();
    let n = u32::try_from(data_inputs.len())
        .map_err(|_| crate::error::VsError::Validate("slot count exceeds u32::MAX".into()))?;
    let start = cx.alloc_slots(n)?;
    for (k, &pin_idx) in data_inputs.iter().enumerate() {
        let slot = start
            .checked_add(u32::try_from(k).map_err(|_| {
                crate::error::VsError::Validate("slot count exceeds u32::MAX".into())
            })?)
            .ok_or_else(|| crate::error::VsError::Validate("slot count exceeds u32::MAX".into()))?;
        let pin = &def.inputs[pin_idx];
        let pin_idx16 = pin_idx as PinIndex;
        if let Some(src) = cx.incoming_by_endpoint.get(&(node_id, pin_idx16)).copied() {
            let src_slot = ensure_value(cx, src, block, pure_emitted)?;
            block.instrs.push(Instr::Move {
                dst: slot,
                src: src_slot,
            });
        } else {
            let v = pin
                .default
                .clone()
                .unwrap_or_else(|| pin.ty.default_value());
            block.instrs.push(Instr::LoadConst {
                dst: slot,
                value: v,
            });
        }
    }
    let end = start
        .checked_add(n)
        .ok_or_else(|| crate::error::VsError::Validate("slot count exceeds u32::MAX".into()))?;
    Ok(start..end)
}

/// 确保 endpoint 上有可用值的 slot；如必要 emit 上游 pure 节点。
fn ensure_value(
    cx: &mut LowerCtx<'_>,
    ep: Endpoint,
    block: &mut Block,
    pure_emitted: &mut HashMap<NodeId, SlotId>,
) -> VsResult<SlotId> {
    let def = cx.defs_by_node[&ep.node].clone();
    let data_pos = u32::try_from(data_pin_position(&def.outputs, ep.pin))
        .map_err(|_| crate::error::VsError::Validate("slot count exceeds u32::MAX".into()))?;
    match def.purity {
        Purity::Impure => {
            let base = cx.impure_out_slots[&ep.node];
            base.checked_add(data_pos).ok_or_else(|| {
                crate::error::VsError::Validate("slot count exceeds u32::MAX".into())
            })
        }
        Purity::Pure => {
            if let Some(&base) = pure_emitted.get(&ep.node) {
                return base.checked_add(data_pos).ok_or_else(|| {
                    crate::error::VsError::Validate("slot count exceeds u32::MAX".into())
                });
            }
            let in_range = emit_inputs(cx, ep.node, block, pure_emitted)?;
            let n_outs = u32::try_from(def.data_outputs().len()).map_err(|_| {
                crate::error::VsError::Validate("slot count exceeds u32::MAX".into())
            })?;
            let out_start = cx.alloc_slots(n_outs)?;
            let out_end = out_start.checked_add(n_outs).ok_or_else(|| {
                crate::error::VsError::Validate("slot count exceeds u32::MAX".into())
            })?;
            let def_id = cx.def_index[&def.id];
            block.instrs.push(Instr::CallPure {
                def: def_id,
                node_id: ep.node,
                inputs: in_range,
                outputs: out_start..out_end,
                config: canonical_config(cx, ep.node),
            });
            pure_emitted.insert(ep.node, out_start);
            Ok(out_start.checked_add(data_pos).ok_or_else(|| {
                crate::error::VsError::Validate("slot count exceeds u32::MAX".into())
            })?)
        }
    }
}

fn canonical_config(cx: &LowerCtx<'_>, node_id: NodeId) -> BTreeMap<String, crate::value::Value> {
    cx.graph
        .node(node_id)
        .map(|node| {
            node.config
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// data 输出针 idx 在 data-only 序列中的位置（exec 不计入）。
fn data_pin_position(pins: &[Pin], idx: PinIndex) -> usize {
    let mut pos = 0usize;
    for (i, p) in pins.iter().enumerate() {
        if i == idx as usize {
            return pos;
        }
        if !p.is_exec() {
            pos += 1;
        }
    }
    pos
}
