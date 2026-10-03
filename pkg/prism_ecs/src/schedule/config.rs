//! System configuration tree: the composable description of *what* a schedule
//! runs and *under which run conditions*.
//!
//! The core type is [`SystemConfigs`], a recursive node that is either a single
//! boxed system (a leaf) or an ordered group of child configs. Each node can
//! additionally carry a list of [`BoxedCondition`]s and a `chained` flag. This
//! recursive shape lets `add_systems((a, b, c).run_if(cond))` build an
//! arbitrarily nested configuration with no cloning and no stub placeholders —
//! grouping, chaining, and gating all compose uniformly.
//!
//! Conversion into this tree is driven by [`IntoSystemConfigs`], implemented for
//! every system-like function and for tuples of configurable things.

use crate::schedule::condition::BoxedCondition;
use crate::system::{BoxedSystem, IntoSystem};
use crate::world::World;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

/// Whether a node is a single system or an ordered group of child nodes.
enum ConfigKind {
    /// A single runnable system.
    Leaf(BoxedSystem),
    /// An ordered collection of child configurations.
    Group(Vec<SystemConfigs>),
}

/// A node in the system-configuration tree.
///
/// Build one from a system function, or from a tuple of systems, via
/// [`IntoSystemConfigs`]. Attach run conditions with
/// [`run_if`](IntoSystemConfigs::run_if) and request chained ordering with
/// [`chain`](IntoSystemConfigs::chain).
pub struct SystemConfigs {
    kind: ConfigKind,
    conditions: Vec<BoxedCondition>,
    chained: bool,
}

impl SystemConfigs {
    /// Build a leaf node wrapping a single boxed system.
    fn leaf(system: BoxedSystem) -> Self {
        Self {
            kind: ConfigKind::Leaf(system),
            conditions: Vec::new(),
            chained: false,
        }
    }

    /// Build a group node wrapping an ordered list of child configs.
    fn group(children: Vec<SystemConfigs>) -> Self {
        Self {
            kind: ConfigKind::Group(children),
            conditions: Vec::new(),
            chained: false,
        }
    }

    /// Whether this node requests chained (strictly-ordered) execution of its
    /// children.
    ///
    /// This is a public getter rather than a private field so the flag stays
    /// reachable. The current [`SingleThreadedExecutor`] already runs children
    /// in insertion order, so chaining is implicit and the flag carries no
    /// extra behaviour yet; it exists for the future parallel executor, which
    /// will use it to force sequential ordering across otherwise-independent
    /// systems.
    ///
    /// [`SingleThreadedExecutor`]: crate::schedule::executor::SingleThreadedExecutor
    pub fn chained(&self) -> bool {
        self.chained
    }

    /// Recursively initialize every system in this subtree against `world`.
    ///
    /// Must be called before [`run`](Self::run) so each system can build its
    /// parameter state (query state, resource ids, …).
    pub(crate) fn initialize(&mut self, world: &mut World) {
        match &mut self.kind {
            ConfigKind::Leaf(system) => system.initialize(world),
            ConfigKind::Group(children) => {
                for child in children.iter_mut() {
                    child.initialize(world);
                }
            }
        }
    }

    /// Evaluate this node's run conditions, short-circuiting on the first that
    /// returns `false`.
    fn conditions_pass(&mut self, world: &World) -> bool {
        self.conditions.iter_mut().all(|condition| condition(world))
    }

    /// Recursively run this subtree against `world`, honouring run conditions.
    ///
    /// If this node's conditions do not all pass, the whole subtree is skipped.
    pub(crate) fn run(&mut self, world: &mut World) {
        if !self.conditions_pass(world) {
            return;
        }
        match &mut self.kind {
            ConfigKind::Leaf(system) => {
                system.run(world);
            }
            ConfigKind::Group(children) => {
                for child in children.iter_mut() {
                    child.run(world);
                }
            }
        }
    }
}

/// Conversion into a [`SystemConfigs`] node, with ergonomic combinators.
///
/// `Marker` is an unconstrained type parameter used only to disambiguate the
/// blanket impls (single system vs. tuple of configs) at the type level; it
/// never appears in a value.
pub trait IntoSystemConfigs<Marker>: Sized {
    /// Convert `self` into a configuration node.
    fn into_configs(self) -> SystemConfigs;

    /// Request chained (strictly-ordered) execution of the resulting node's
    /// children. See [`SystemConfigs::chained`] for the current semantics.
    fn chain(self) -> SystemConfigs {
        let mut configs = self.into_configs();
        configs.chained = true;
        configs
    }

    /// Gate the resulting node behind a run condition. The node (and, if it is
    /// a group, its whole subtree) only runs when `condition` returns `true`.
    fn run_if(
        self,
        condition: impl FnMut(&World) -> bool + Send + Sync + 'static,
    ) -> SystemConfigs {
        let mut configs = self.into_configs();
        configs.conditions.push(Box::new(condition));
        configs
    }
}

/// Identity marker: a [`SystemConfigs`] is already configured and converts to
/// itself. Keeping this on a distinct marker type avoids overlapping with the
/// system/tuple blanket impls below.
pub struct AlreadyConfigured;

impl IntoSystemConfigs<AlreadyConfigured> for SystemConfigs {
    fn into_configs(self) -> SystemConfigs {
        self
    }
}

impl<Marker, S> IntoSystemConfigs<Marker> for S
where
    S: IntoSystem<(), Marker>,
{
    fn into_configs(self) -> SystemConfigs {
        SystemConfigs::leaf(Box::new(self.into_system()))
    }
}

/// Marker distinguishing the tuple impls from the single-system impl above.
///
/// Placing a distinct, private marker type as the first element of the tuple
/// impls' `Marker` keeps them from overlapping with the `S: IntoSystem` impl:
/// a tuple can never satisfy `IntoSystem`, and the distinct first marker makes
/// the two blanket impls provably disjoint to the coherence checker.
pub struct SystemConfigTupleMarker;

macro_rules! impl_into_system_configs_tuple {
    ($(($S:ident, $M:ident)),+) => {
        impl<$($S, $M),+> IntoSystemConfigs<(SystemConfigTupleMarker, $($M,)+)> for ($($S,)+)
        where
            $($S: IntoSystemConfigs<$M>,)+
        {
            #[allow(non_snake_case)]
            fn into_configs(self) -> SystemConfigs {
                let ($($S,)+) = self;
                SystemConfigs::group(vec![$($S.into_configs(),)+])
            }
        }
    };
}

impl_into_system_configs_tuple!((S0, M0));
impl_into_system_configs_tuple!((S0, M0), (S1, M1));
impl_into_system_configs_tuple!((S0, M0), (S1, M1), (S2, M2));
impl_into_system_configs_tuple!((S0, M0), (S1, M1), (S2, M2), (S3, M3));
impl_into_system_configs_tuple!((S0, M0), (S1, M1), (S2, M2), (S3, M3), (S4, M4));
impl_into_system_configs_tuple!((S0, M0), (S1, M1), (S2, M2), (S3, M3), (S4, M4), (S5, M5));
impl_into_system_configs_tuple!(
    (S0, M0), (S1, M1), (S2, M2), (S3, M3), (S4, M4), (S5, M5), (S6, M6)
);
impl_into_system_configs_tuple!(
    (S0, M0), (S1, M1), (S2, M2), (S3, M3), (S4, M4), (S5, M5), (S6, M6), (S7, M7)
);
