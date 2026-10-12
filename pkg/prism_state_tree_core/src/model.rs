//! The immutable TreeDef-to-Plan model.
use alloc::{vec, vec::Vec};

/// Stable numeric identity of a state node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StateId(pub u32);

/// Stable numeric identity of an input event kind.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EventId(pub u32);

/// Stable numeric identity of a pure guard input.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GuardId(pub u32);

/// History depth captured for a compound or parallel state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryMode {
    /// Restore the remembered direct child or region roots and initialize below them.
    Shallow,
    /// Restore the complete remembered active configuration.
    Deep,
}

/// Structural kind of a state node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NodeKind {
    /// A leaf state with no children.
    Atomic,
    /// Exactly one child is active; `initial` is selected on ordinary entry.
    Compound {
        /// Direct children in declaration order.
        children: Vec<StateId>,
        /// Initial direct child.
        initial: StateId,
        /// Whether this node records history when it is exited.
        history: Option<HistoryMode>,
    },
    /// All region roots are active concurrently.
    Parallel {
        /// Direct children in declaration order.
        children: Vec<StateId>,
        /// Direct region roots in deterministic entry order.
        regions: Vec<StateId>,
        /// Whether this node records history when it is exited.
        history: Option<HistoryMode>,
    },
}

impl NodeKind {
    pub(crate) fn children(&self) -> &[StateId] {
        match self {
            Self::Atomic => &[],
            Self::Compound { children, .. } | Self::Parallel { children, .. } => children,
        }
    }

    pub(crate) fn history(&self) -> Option<HistoryMode> {
        match self {
            Self::Compound { history, .. } | Self::Parallel { history, .. } => *history,
            Self::Atomic => None,
        }
    }
}

/// A state node in an authoring definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateNode {
    /// Stable identity of this node.
    pub id: StateId,
    /// Parent identity, or `None` for the root.
    pub parent: Option<StateId>,
    /// Structural state kind.
    pub kind: NodeKind,
    /// Transitions declared by this node, in authoring order.
    pub transitions: Vec<Transition>,
}

impl StateNode {
    /// Creates an atomic node.
    pub fn atomic(id: StateId, parent: Option<StateId>) -> Self {
        Self {
            id,
            parent,
            kind: NodeKind::Atomic,
            transitions: Vec::new(),
        }
    }

    /// Creates a compound node.
    pub fn compound(
        id: StateId,
        parent: Option<StateId>,
        children: Vec<StateId>,
        initial: StateId,
        history: Option<HistoryMode>,
    ) -> Self {
        Self {
            id,
            parent,
            kind: NodeKind::Compound {
                children,
                initial,
                history,
            },
            transitions: Vec::new(),
        }
    }

    /// Creates a parallel node with explicitly ordered region roots.
    pub fn parallel(
        id: StateId,
        parent: Option<StateId>,
        children: Vec<StateId>,
        regions: Vec<StateId>,
        history: Option<HistoryMode>,
    ) -> Self {
        Self {
            id,
            parent,
            kind: NodeKind::Parallel {
                children,
                regions,
                history,
            },
            transitions: Vec::new(),
        }
    }

    /// Appends a transition in declaration order.
    pub fn with_transition(mut self, transition: Transition) -> Self {
        self.transitions.push(transition);
        self
    }
}

/// Trigger used to select a transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Trigger {
    /// Select when the queued event has this identity.
    Event(EventId),
    /// Select when the active configuration is stabilized after an event or entry.
    Eventless,
}

/// Transition lifecycle behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransitionKind {
    /// Exit and enter according to the least common ancestor.
    External,
    /// Keep the active configuration and only report the transition.
    Internal,
}

/// Transition destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransitionTarget {
    /// Enter this state normally.
    State(StateId),
    /// Re-enter the owner and restore its remembered configuration.
    History {
        /// Compound or parallel owner whose history is requested.
        owner: StateId,
        /// Requested restore depth.
        depth: HistoryMode,
    },
    /// The transition's source state; external self transitions restart it.
    SelfState,
}

/// A pure-data transition declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transition {
    /// State that owns this transition.
    pub source: StateId,
    /// Destination.
    pub target: TransitionTarget,
    /// Trigger.
    pub trigger: Trigger,
    /// Optional pure guard identity.
    pub guard: Option<GuardId>,
    /// Lower values win before declaration order.
    pub priority: u16,
    /// External or internal lifecycle semantics.
    pub kind: TransitionKind,
    pub(crate) declaration_order: u32,
}

impl Transition {
    /// Creates a transition; its declaration order is assigned during compilation.
    pub fn new(
        source: StateId,
        target: TransitionTarget,
        trigger: Trigger,
        guard: Option<GuardId>,
        priority: u16,
        kind: TransitionKind,
    ) -> Self {
        Self {
            source,
            target,
            trigger,
            guard,
            priority,
            kind,
            declaration_order: 0,
        }
    }

    /// Returns the deterministic order assigned by [`TreeDef::compile`].
    pub fn declaration_order(&self) -> u32 {
        self.declaration_order
    }
}

/// Authoring input for a compiled state-tree plan.
#[derive(Clone, Debug)]
pub struct TreeDef {
    /// Root identity.
    pub root: StateId,
    /// All nodes. Their `id` values must be unique and dense from zero.
    pub nodes: Vec<StateNode>,
}

impl TreeDef {
    /// Validates and freezes this definition into a runtime plan.
    pub fn compile(self) -> Result<Plan, PlanError> {
        Plan::compile(self)
    }
}

/// An immutable, validated, runtime-ready state-tree plan.
#[derive(Clone, Debug)]
pub struct Plan {
    pub(crate) root: StateId,
    pub(crate) nodes: Vec<StateNode>,
    pub(crate) parents: Vec<Option<StateId>>,
    pub(crate) depths: Vec<u32>,
}

impl Plan {
    /// Validates and compiles a [`TreeDef`].
    pub fn compile(def: TreeDef) -> Result<Self, PlanError> {
        let count = def.nodes.len();
        if count == 0 {
            return Err(PlanError::Empty);
        }
        if def.root.0 as usize >= count {
            return Err(PlanError::UnknownState(def.root));
        }
        let mut nodes = def.nodes;
        let mut parents = vec![None; count];
        for (index, node) in nodes.iter().enumerate() {
            if node.id.0 as usize != index {
                return Err(PlanError::NonDenseStateId(node.id));
            }
            if let Some(parent) = node.parent {
                if parent.0 as usize >= count {
                    return Err(PlanError::UnknownState(parent));
                }
                if parents[index].replace(parent).is_some() {
                    return Err(PlanError::MultipleParents(node.id));
                }
            }
            if node.id == def.root && node.parent.is_some() {
                return Err(PlanError::RootHasParent(node.id));
            }
            for &child in node.kind.children() {
                if child.0 as usize >= count {
                    return Err(PlanError::UnknownState(child));
                }
            }
            match &node.kind {
                NodeKind::Atomic if !node.kind.children().is_empty() => {
                    return Err(PlanError::AtomicHasChildren(node.id))
                }
                NodeKind::Atomic => {}
                NodeKind::Compound {
                    children, initial, ..
                } => {
                    if children.is_empty() || !children.contains(initial) {
                        return Err(PlanError::InvalidInitial {
                            owner: node.id,
                            child: *initial,
                        });
                    }
                    for (index, child) in children.iter().enumerate() {
                        if children[..index].contains(child) {
                            return Err(PlanError::DuplicateChild {
                                owner: node.id,
                                child: *child,
                            });
                        }
                    }
                }
                NodeKind::Parallel {
                    children, regions, ..
                } => {
                    if regions.is_empty()
                        || children.is_empty()
                        || regions.len() != children.len()
                        || regions.iter().any(|region| !children.contains(region))
                        || children.iter().any(|child| !regions.contains(child))
                    {
                        return Err(PlanError::InvalidRegions(node.id));
                    }
                    for (index, child) in children.iter().enumerate() {
                        if children[..index].contains(child) {
                            return Err(PlanError::DuplicateChild {
                                owner: node.id,
                                child: *child,
                            });
                        }
                    }
                    for (index, region) in regions.iter().enumerate() {
                        if regions[..index].contains(region) {
                            return Err(PlanError::DuplicateRegion {
                                owner: node.id,
                                region: *region,
                            });
                        }
                    }
                }
            }
            for transition in &node.transitions {
                if transition.source != node.id {
                    return Err(PlanError::TransitionSourceMismatch(node.id));
                }
            }
        }
        for node in &nodes {
            for &child in node.kind.children() {
                match parents[child.0 as usize] {
                    Some(parent) if parent != node.id => {
                        return Err(PlanError::MultipleParents(child))
                    }
                    Some(_) => {}
                    None => parents[child.0 as usize] = Some(node.id),
                }
            }
        }
        for node in &nodes {
            if node.id != def.root
                && !parents[node.id.0 as usize]
                    .and_then(|parent| nodes.get(parent.0 as usize))
                    .map(|parent| parent.kind.children().contains(&node.id))
                    .unwrap_or(false)
            {
                return Err(PlanError::ParentMismatch(node.id));
            }
        }
        if parents[def.root.0 as usize].is_some() {
            return Err(PlanError::RootHasParent(def.root));
        }
        for (index, parent) in parents.iter().enumerate() {
            if StateId(index as u32) != def.root && parent.is_none() {
                return Err(PlanError::Disconnected(StateId(index as u32)));
            }
        }
        let mut depths = vec![0; count];
        for (index, depth) in depths.iter_mut().enumerate() {
            let mut cursor = StateId(index as u32);
            let mut steps = 0;
            while let Some(parent) = parents[cursor.0 as usize] {
                steps += 1;
                if steps > count as u32 {
                    return Err(PlanError::Cycle(cursor));
                }
                cursor = parent;
            }
            *depth = steps;
        }
        let mut order = 0;
        for node in &mut nodes {
            for transition in &mut node.transitions {
                transition.declaration_order = order;
                order += 1;
            }
        }
        for node in &nodes {
            for transition in &node.transitions {
                match transition.target {
                    TransitionTarget::State(id) if id.0 as usize >= count => {
                        return Err(PlanError::UnknownState(id));
                    }
                    TransitionTarget::History { owner, .. } => {
                        if owner.0 as usize >= count {
                            return Err(PlanError::UnknownState(owner));
                        }
                        if !matches!(
                            nodes[owner.0 as usize].kind,
                            NodeKind::Compound {
                                history: Some(_),
                                ..
                            } | NodeKind::Parallel {
                                history: Some(_),
                                ..
                            }
                        ) {
                            return Err(PlanError::InvalidHistoryOwner(owner));
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(Self {
            root: def.root,
            nodes,
            parents,
            depths,
        })
    }

    /// Returns the root state.
    pub fn root(&self) -> StateId {
        self.root
    }
    /// Returns a state node.
    pub fn node(&self, id: StateId) -> Option<&StateNode> {
        self.nodes.get(id.0 as usize)
    }
    /// Returns the parent, if any.
    pub fn parent(&self, id: StateId) -> Option<StateId> {
        self.parents.get(id.0 as usize).copied().flatten()
    }
    /// Returns the plan's state count.
    pub fn state_count(&self) -> usize {
        self.nodes.len()
    }
    /// Returns a state's precomputed depth.
    pub fn depth(&self, id: StateId) -> Option<u32> {
        self.depths.get(id.0 as usize).copied()
    }
    pub(crate) fn contains(&self, id: StateId) -> bool {
        (id.0 as usize) < self.nodes.len()
    }
    pub(crate) fn is_descendant(&self, mut id: StateId, ancestor: StateId) -> bool {
        loop {
            if id == ancestor {
                return true;
            }
            match self.parent(id) {
                Some(parent) => id = parent,
                None => return false,
            }
        }
    }
    pub(crate) fn lca(&self, mut a: StateId, mut b: StateId) -> StateId {
        let mut da = self.depths[a.0 as usize];
        let mut db = self.depths[b.0 as usize];
        while da > db {
            a = self.parent(a).expect("validated parent");
            da -= 1;
        }
        while db > da {
            b = self.parent(b).expect("validated parent");
            db -= 1;
        }
        while a != b {
            a = self.parent(a).expect("validated tree");
            b = self.parent(b).expect("validated tree");
        }
        a
    }
}

/// Errors found while compiling a tree definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// No nodes were supplied.
    Empty,
    /// A state identity was not in the dense node range.
    UnknownState(StateId),
    /// IDs must be zero-based and dense for stable compact indexing.
    NonDenseStateId(StateId),
    /// A root was assigned a parent.
    RootHasParent(StateId),
    /// Two nodes claimed the same child.
    MultipleParents(StateId),
    /// A node's parent field and its parent's child list disagree.
    ParentMismatch(StateId),
    /// An atomic node has children.
    AtomicHasChildren(StateId),
    /// A compound initial child is absent.
    InvalidInitial {
        /// Compound state whose initial declaration is invalid.
        owner: StateId,
        /// Requested initial child.
        child: StateId,
    },
    /// A parallel node has no valid region roots.
    InvalidRegions(StateId),
    /// A compound or parallel node repeats a direct child.
    DuplicateChild {
        /// Node declaring the repeated child.
        owner: StateId,
        /// Repeated child identity.
        child: StateId,
    },
    /// A parallel node repeats a region root.
    DuplicateRegion {
        /// Parallel node declaring the repeated region.
        owner: StateId,
        /// Repeated region root.
        region: StateId,
    },
    /// A history transition targets a node without configured history.
    InvalidHistoryOwner(StateId),
    /// A transition source differs from the node containing it.
    TransitionSourceMismatch(StateId),
    /// A node is not reachable from the root.
    Disconnected(StateId),
    /// A parent cycle was detected.
    Cycle(StateId),
}
