//! System and set *configuration*: ordering edges, set membership, phase, and
//! run-conditions attached to systems before they are added to a
//! [`Schedule`](crate::schedule::Schedule) (design §8.2).
//!
//! The ergonomic surface is a small recursive tree, [`SystemConfigs`]:
//!
//! * a bare system (or a pre-built [`SystemConfig`]) is a **leaf**;
//! * a tuple of configs is a **group**;
//! * fluent combinators refine either: [`chain`](SystemConfigs::chain) orders a
//!   group's children end-to-end, [`run_if`](SystemConfigs::run_if) gates a leaf
//!   (per-system) or a whole group (one shared condition), and
//!   [`in_set`](SystemConfigs::in_set) / [`before`](SystemConfigs::before) /
//!   [`after`](SystemConfigs::after) / [`in_phase`](SystemConfigs::in_phase)
//!   distribute down to every leaf.
//!
//! Anything that converts into this tree implements [`IntoSystemConfigs`], so
//! the same fluent methods work directly on a bare `fn`, a tuple, or an
//! already-built [`SystemConfigs`].

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::schedule::condition::{BoxedCondition, Condition};
use crate::schedule::phase::Phase;
use crate::schedule::set::{SystemSet, SystemSetId};
use crate::system::function::{BoxedSystem, IntoSystem};

/// A single system plus its ordering metadata: the leaf of a [`SystemConfigs`]
/// tree.
pub struct SystemConfig {
    pub(crate) system: BoxedSystem,
    pub(crate) sets: Vec<SystemSetId>,
    pub(crate) before: Vec<SystemSetId>,
    pub(crate) after: Vec<SystemSetId>,
    pub(crate) conditions: Vec<BoxedCondition>,
    pub(crate) phase: Phase,
}

impl SystemConfig {
    pub(crate) fn from_system(system: BoxedSystem) -> Self {
        Self {
            system,
            sets: Vec::new(),
            before: Vec::new(),
            after: Vec::new(),
            conditions: Vec::new(),
            phase: Phase::Update,
        }
    }
}

/// A tree of configured systems: either a single [`SystemConfig`] leaf or a
/// group of child trees.
///
/// Build one with [`IntoSystemConfigs::into_configs`] (usually implicitly, by
/// passing a system or tuple to
/// [`Schedule::add_systems`](crate::schedule::Schedule::add_systems)) and refine
/// it with the fluent methods.
pub enum SystemConfigs {
    /// A single configured system.
    Node(SystemConfig),
    /// A group of child configs, optionally chained end-to-end, optionally
    /// gated by shared run-conditions.
    Group {
        /// The child configs, in declaration order.
        configs: Vec<SystemConfigs>,
        /// When `true`, every system contributed by child *i* is ordered before
        /// every system contributed by child *i + 1*.
        chained: bool,
        /// Conditions gating the group as a whole; evaluated once per run and
        /// applied to every leaf (see [`run_if`](SystemConfigs::run_if)).
        collective_conditions: Vec<BoxedCondition>,
    },
}

impl SystemConfigs {
    /// Order this group's children end-to-end (a *chain*): all systems of child
    /// *i* run before all systems of child *i + 1*. A no-op on a single-system
    /// leaf (nothing to order).
    #[must_use]
    pub fn chain(mut self) -> Self {
        if let SystemConfigs::Group { chained, .. } = &mut self {
            *chained = true;
        }
        self
    }

    /// Whether this is a chained group.
    #[inline]
    #[must_use]
    pub fn chained(&self) -> bool {
        matches!(
            self,
            SystemConfigs::Group {
                chained: true,
                ..
            }
        )
    }

    /// Gate on `condition`. On a leaf this is a per-system condition; on a group
    /// it is a shared condition evaluated once per run that gates every member.
    #[must_use]
    pub fn run_if<M>(mut self, condition: impl Condition<M>) -> Self {
        match &mut self {
            SystemConfigs::Node(config) => config.conditions.push(condition.into_boxed_condition()),
            SystemConfigs::Group {
                collective_conditions,
                ..
            } => collective_conditions.push(condition.into_boxed_condition()),
        }
        self
    }

    /// Add every leaf to `set`, inheriting the set's ordering edges and
    /// run-conditions.
    #[must_use]
    pub fn in_set(mut self, set: impl SystemSet) -> Self {
        let id = set.set_id();
        self.for_each_leaf(&mut |config| config.sets.push(id));
        self
    }

    /// Order every leaf before every member of `set`.
    #[must_use]
    pub fn before(mut self, set: impl SystemSet) -> Self {
        let id = set.set_id();
        self.for_each_leaf(&mut |config| config.before.push(id));
        self
    }

    /// Order every leaf after every member of `set`.
    #[must_use]
    pub fn after(mut self, set: impl SystemSet) -> Self {
        let id = set.set_id();
        self.for_each_leaf(&mut |config| config.after.push(id));
        self
    }

    /// Place every leaf in `phase` (overriding the default [`Phase::Update`]).
    #[must_use]
    pub fn in_phase(mut self, phase: Phase) -> Self {
        self.for_each_leaf(&mut |config| config.phase = phase);
        self
    }

    pub(crate) fn for_each_leaf(&mut self, f: &mut impl FnMut(&mut SystemConfig)) {
        match self {
            SystemConfigs::Node(config) => f(config),
            SystemConfigs::Group { configs, .. } => {
                for child in configs {
                    child.for_each_leaf(f);
                }
            }
        }
    }
}

/// Conversion into a [`SystemConfigs`] tree, with fluent defaults so the
/// combinators work directly on bare systems and tuples.
pub trait IntoSystemConfigs<Marker>: Sized {
    /// Perform the conversion.
    fn into_configs(self) -> SystemConfigs;

    /// Chain the resulting group (see [`SystemConfigs::chain`]).
    #[must_use]
    fn chain(self) -> SystemConfigs {
        self.into_configs().chain()
    }

    /// Gate behind `condition` (see [`SystemConfigs::run_if`]).
    #[must_use]
    fn run_if<M>(self, condition: impl Condition<M>) -> SystemConfigs {
        self.into_configs().run_if(condition)
    }

    /// Add to `set` (see [`SystemConfigs::in_set`]).
    #[must_use]
    fn in_set(self, set: impl SystemSet) -> SystemConfigs {
        self.into_configs().in_set(set)
    }

    /// Order before `set` (see [`SystemConfigs::before`]).
    #[must_use]
    fn before(self, set: impl SystemSet) -> SystemConfigs {
        self.into_configs().before(set)
    }

    /// Order after `set` (see [`SystemConfigs::after`]).
    #[must_use]
    fn after(self, set: impl SystemSet) -> SystemConfigs {
        self.into_configs().after(set)
    }

    /// Place in `phase` (see [`SystemConfigs::in_phase`]).
    #[must_use]
    fn in_phase(self, phase: Phase) -> SystemConfigs {
        self.into_configs().in_phase(phase)
    }
}

/// Identity conversion for an already-built tree.
impl IntoSystemConfigs<()> for SystemConfigs {
    #[inline]
    fn into_configs(self) -> SystemConfigs {
        self
    }
}

/// Marker for the [`SystemConfig`] leaf conversion.
pub struct IsSystemConfig;

impl IntoSystemConfigs<IsSystemConfig> for SystemConfig {
    #[inline]
    fn into_configs(self) -> SystemConfigs {
        SystemConfigs::Node(self)
    }
}

/// Marker for the function/system blanket conversion.
pub struct IsSystem;

impl<Marker, S> IntoSystemConfigs<(IsSystem, Marker)> for S
where
    S: IntoSystem<(), Marker>,
{
    #[inline]
    fn into_configs(self) -> SystemConfigs {
        let system: BoxedSystem = Box::new(IntoSystem::into_system(self));
        SystemConfigs::Node(SystemConfig::from_system(system))
    }
}

/// Marker for the tuple (group) conversions.
pub struct IsGroup;

macro_rules! impl_into_configs_tuple {
    ($(($type:ident, $marker:ident, $value:ident)),+) => {
        impl<$($type, $marker),+> IntoSystemConfigs<(IsGroup, $($marker,)+)> for ($($type,)+)
        where
            $($type: IntoSystemConfigs<$marker>,)+
        {
            #[inline]
            fn into_configs(self) -> SystemConfigs {
                let ($($value,)+) = self;
                SystemConfigs::Group {
                    configs: alloc::vec![$($value.into_configs(),)+],
                    chained: false,
                    collective_conditions: Vec::new(),
                }
            }
        }
    };
}

impl_into_configs_tuple!((C0, M0, v0));
impl_into_configs_tuple!((C0, M0, v0), (C1, M1, v1));
impl_into_configs_tuple!((C0, M0, v0), (C1, M1, v1), (C2, M2, v2));
impl_into_configs_tuple!((C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3));
impl_into_configs_tuple!((C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4));
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5)
);
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5),
    (C6, M6, v6)
);
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5),
    (C6, M6, v6), (C7, M7, v7)
);
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5),
    (C6, M6, v6), (C7, M7, v7), (C8, M8, v8)
);
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5),
    (C6, M6, v6), (C7, M7, v7), (C8, M8, v8), (C9, M9, v9)
);
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5),
    (C6, M6, v6), (C7, M7, v7), (C8, M8, v8), (C9, M9, v9), (C10, M10, v10)
);
impl_into_configs_tuple!(
    (C0, M0, v0), (C1, M1, v1), (C2, M2, v2), (C3, M3, v3), (C4, M4, v4), (C5, M5, v5),
    (C6, M6, v6), (C7, M7, v7), (C8, M8, v8), (C9, M9, v9), (C10, M10, v10), (C11, M11, v11)
);

/// Shared ordering/condition metadata for a [`SystemSet`], registered via
/// [`Schedule::configure_set`](crate::schedule::Schedule::configure_set).
#[derive(Default)]
pub struct SetConfig {
    pub(crate) before: Vec<SystemSetId>,
    pub(crate) after: Vec<SystemSetId>,
    pub(crate) conditions: Vec<BoxedCondition>,
}

impl SetConfig {
    /// A fresh, empty configuration.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Order every member of this set before every member of `set`.
    #[must_use]
    pub fn before(mut self, set: impl SystemSet) -> Self {
        self.before.push(set.set_id());
        self
    }

    /// Order every member of this set after every member of `set`.
    #[must_use]
    pub fn after(mut self, set: impl SystemSet) -> Self {
        self.after.push(set.set_id());
        self
    }

    /// Gate every member of this set behind `condition` (evaluated once per run
    /// the first time a member is reached).
    #[must_use]
    pub fn run_if<M>(mut self, condition: impl Condition<M>) -> Self {
        self.conditions.push(condition.into_boxed_condition());
        self
    }
}
