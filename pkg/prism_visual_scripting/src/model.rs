//! Graph 模型：Pin / Node / Edge / Variable / Graph。
//!
//! 这是编辑器与磁盘序列化的形态，运行期会进一步 lower 到 `ir::SlotProgram`。

use crate::value::{Value, ValueType};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// 节点 id（图内唯一）。
pub type NodeId = u32;
/// 节点上 pin 的本地下标。
pub type PinIndex = u16;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCall {
    pub graph: String,
    pub entry: NodeId,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinPolicy {
    All,
    Race,
}

/// 由多图 Workflow VM 执行的结构化控制语义。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlNode {
    FrameEntry,
    Return,
    Call {
        target: GraphCall,
    },
    Fork {
        branches: Vec<GraphCall>,
        policy: JoinPolicy,
        max_concurrency: u16,
    },
    Transition {
        state: String,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PinDir {
    In,
    Out,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pin {
    pub name: String,
    pub dir: PinDir,
    pub ty: ValueType,
    /// 仅 In + Data pin 有意义：未连线时使用的字面量。
    #[serde(default)]
    pub default: Option<Value>,
}

impl Pin {
    pub fn is_exec(&self) -> bool {
        matches!(self.ty, ValueType::Exec)
    }
}

/// 节点引用（指向 registry 中的 NodeDefinition）。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeDefRef {
    pub id: String,
    /// 可选语义化版本。MVP 不强制。
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub def: NodeDefRef,
    #[serde(default)]
    pub title: Option<String>,
    /// 节点的字面量配置（不通过 pin 传入的常量）。
    #[serde(default)]
    pub config: HashMap<String, Value>,
}

/// 端点：(节点, pin 下标)。
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Endpoint {
    pub node: NodeId,
    pub pin: PinIndex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub from: Endpoint,
    pub to: Endpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub ty: ValueType,
    #[serde(default)]
    pub default: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    #[serde(default)]
    pub variables: Vec<Variable>,
    /// 事件入口节点 id。
    #[serde(default)]
    pub entries: Vec<NodeId>,
    /// node id -> 结构化控制语义。普通图保持为空。
    #[serde(default)]
    pub controls: BTreeMap<NodeId, ControlNode>,
}

impl Graph {
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn nodes_by_id(&self) -> HashMap<NodeId, &Node> {
        self.nodes.iter().map(|n| (n.id, n)).collect()
    }

    /// 入边索引：to.node -> Vec<Edge>。
    pub fn incoming(&self) -> HashMap<NodeId, Vec<&Edge>> {
        let mut m: HashMap<NodeId, Vec<&Edge>> = HashMap::new();
        for e in &self.edges {
            m.entry(e.to.node).or_default().push(e);
        }
        m
    }

    /// 出边索引。
    pub fn outgoing(&self) -> HashMap<NodeId, Vec<&Edge>> {
        let mut m: HashMap<NodeId, Vec<&Edge>> = HashMap::new();
        for e in &self.edges {
            m.entry(e.from.node).or_default().push(e);
        }
        m
    }
}
