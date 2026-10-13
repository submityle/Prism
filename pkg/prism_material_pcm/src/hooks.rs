//! The typed hook ABI: what each of the nine extension-point kinds may
//! contribute, and where that contribution runs (§5.4 / §5.4.1).
//!
//! A manifest declares *what* an extension point is — identity, namespace,
//! kind, capability needs. This module declares what a `kind` **can do**: its
//! hook set, the stage each hook runs in, and whether the hook owes a
//! dual-line (CPU golden against WESL twin) pair. Two design rules are encoded
//! here rather than left to prose:
//!
//! * **§5.2 red line — bake-time monomorphisation, never runtime dispatch.**
//!   A [`Stage::Bake`] hook runs in the baker/loader and is a constant by the
//!   time the runtime sees the material. A [`Stage::Runtime`] hook runs inside
//!   the material's *already specialised* WESL. Registering a kind never gives
//!   the engine a runtime interpreter.
//! * **§11 dual-line determinism.** Every [`Stage::Runtime`] hook owes a byte
//!   comparison against the CPU gold standard ([`Twin::ByteCompared`]),
//!   because a runtime hook is exactly where a CPU/GPU divergence turns into a
//!   picture bug. A bake hook is exempt ([`Twin::NotRequired`]) — §5.4.1's
//!   offline exception: it never reaches the GPU, so there is nothing for it
//!   to diverge from. §5.4.1's `golden="auto"` / `golden="none"` split is the
//!   authoring-tier view of the same fact (a restricted-Rust lobe gets its twin
//!   for free; a raw-WESL escape hatch must hand-write one).
//!
//! # The two questions the header digest answers
//!
//! [`Kind::header_resident`] is the other half of the same contract. The
//! material header carries a fixed 64-slot `used`/`required` digest (§5.4), and
//! that address space is scarce: it is the only place a *per-material* fact can
//! live without violating §5.2. So a kind earns a slot only when the runtime
//! genuinely reads one, i.e. only when it changes (a) the shader the baker must
//! specialise, or (b) the state the runtime tracks per material:
//!
//! * **resident** — `lobe` (`closure_mask`), `channel` (the §17.1 dynamic
//!   segment), `response_mode` (the §5.2 classification input), `cache_aux`
//!   (§14.3: the cache stores a different aux set per material);
//! * **identity-only** — `transfer`, `provider`, `downgrade`, `author_node`,
//!   `composite`. Each leaves its trace in the material's *parameters* or in the
//!   frame/pass graph (§5.2: folded into the bake-time permutation key or the
//!   pass graph), so a per-material bit would be state with no reader — the
//!   exact thing §5.2 forbids. They are still registered with identity,
//!   namespace, stability and capability needs, and are still negotiated, just
//!   at bake/pass level rather than per material.

use crate::Kind;

/// When a hook runs (see the module docs for the §5.2 red line).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Stage {
    /// Runs in the baker/loader; its result is a constant for the runtime.
    Bake,
    /// Runs inside the material's monomorphised WESL.
    Runtime,
}

impl Stage {
    /// Both stages, bake first (the order the pipeline runs them in).
    pub const ALL: [Self; 2] = [Self::Bake, Self::Runtime];

    /// The canonical lowercase spelling used in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bake => "bake",
            Self::Runtime => "runtime",
        }
    }

    /// The dual-line obligation a hook at this stage carries (§11).
    pub const fn twin(self) -> Twin {
        match self {
            Self::Bake => Twin::NotRequired,
            Self::Runtime => Twin::ByteCompared,
        }
    }
}

impl core::fmt::Display for Stage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The CPU against GPU dual-line obligation of a hook (§11).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Twin {
    /// The hook never reaches the GPU: no twin exists to drift (§5.4.1).
    NotRequired,
    /// The hook must be byte-compared against the CPU gold standard.
    ByteCompared,
}

impl Twin {
    /// Whether a twin comparison is owed.
    pub const fn is_required(self) -> bool {
        matches!(self, Self::ByteCompared)
    }

    /// The canonical lowercase spelling used in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::ByteCompared => "byte_compared",
        }
    }
}

impl core::fmt::Display for Twin {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One hook of a kind's typed ABI.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Hook {
    name: &'static str,
    stage: Stage,
}

impl Hook {
    /// Declare a hook. Its dual-line obligation follows from its stage.
    pub const fn new(name: &'static str, stage: Stage) -> Self {
        Self { name, stage }
    }

    /// The hook's canonical name (`unpack`, `produce_page`, ...).
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// When the hook runs.
    pub const fn stage(self) -> Stage {
        self.stage
    }

    /// Whether the hook owes a CPU/WESL twin comparison.
    pub const fn twin(self) -> Twin {
        self.stage.twin()
    }
}

impl core::fmt::Display for Hook {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}::{} ({})", self.name, self.stage, self.twin())
    }
}

/// The widest hook set any kind declares.
pub const MAX_HOOKS: usize = 4;

const LOBE_HOOKS: [Hook; 4] = [
    Hook::new("unpack", Stage::Bake),
    Hook::new("classify", Stage::Bake),
    Hook::new("sample", Stage::Runtime),
    Hook::new("respond", Stage::Runtime),
];
const CHANNEL_HOOKS: [Hook; 2] = [
    Hook::new("declare", Stage::Bake),
    Hook::new("apply", Stage::Runtime),
];
const RESPONSE_MODE_HOOKS: [Hook; 2] = [
    Hook::new("class_entry", Stage::Bake),
    Hook::new("respond", Stage::Runtime),
];
const TRANSFER_HOOKS: [Hook; 1] = [Hook::new("remap", Stage::Bake)];
const CACHE_AUX_HOOKS: [Hook; 1] = [Hook::new("produce", Stage::Runtime)];
const PROVIDER_HOOKS: [Hook; 1] = [Hook::new("produce_page", Stage::Bake)];
const DOWNGRADE_HOOKS: [Hook; 1] = [Hook::new("plan", Stage::Bake)];
const AUTHOR_NODE_HOOKS: [Hook; 1] = [Hook::new("evaluate", Stage::Bake)];
const COMPOSITE_HOOKS: [Hook; 1] = [Hook::new("record", Stage::Bake)];

impl Kind {
    /// This kind's typed hook ABI (§5.4).
    pub const fn hooks(self) -> &'static [Hook] {
        match self {
            Self::Lobe => &LOBE_HOOKS,
            Self::Channel => &CHANNEL_HOOKS,
            Self::ResponseMode => &RESPONSE_MODE_HOOKS,
            Self::Transfer => &TRANSFER_HOOKS,
            Self::CacheAux => &CACHE_AUX_HOOKS,
            Self::Provider => &PROVIDER_HOOKS,
            Self::Downgrade => &DOWNGRADE_HOOKS,
            Self::AuthorNode => &AUTHOR_NODE_HOOKS,
            Self::Composite => &COMPOSITE_HOOKS,
        }
    }

    /// Whether this kind occupies a slot in the material header digest.
    ///
    /// See the module docs: a kind is resident only when the runtime reads a
    /// per-material fact about it. Everything else resolves at bake/pass time
    /// and is registered identity-only, so the 64-slot digest is spent on
    /// addresses that are actually read.
    pub const fn header_resident(self) -> bool {
        match self {
            Self::Lobe | Self::Channel | Self::ResponseMode | Self::CacheAux => true,
            Self::Transfer
            | Self::Provider
            | Self::Downgrade
            | Self::AuthorNode
            | Self::Composite => false,
        }
    }

    /// Whether this kind's registration can change the §5.2 step-2 class table.
    ///
    /// Exactly one kind can: a `response_mode` makes the *number of classes*
    /// registration-driven, which is the single hot-path structural change this
    /// whole capability model admits (§5.2, roadmap P2-j). The predicate exists
    /// so a reviewer — and the CI gate — can point at the one exception
    /// instead of re-deriving it from prose.
    pub const fn touches_class_table(self) -> bool {
        matches!(self, Self::ResponseMode)
    }

    /// Number of runtime hooks this kind declares — the hooks that owe a
    /// dual-line twin (§11) and that the WESL specialiser must emit.
    pub const fn runtime_hook_count(self) -> usize {
        let hooks = self.hooks();
        let mut count = 0;
        let mut index = 0;
        while index < hooks.len() {
            if matches!(hooks[index].stage(), Stage::Runtime) {
                count += 1;
            }
            index += 1;
        }
        count
    }

    /// Whether this kind declares a hook named `name`.
    pub const fn declares_hook(self, name: &str) -> bool {
        let hooks = self.hooks();
        let mut index = 0;
        while index < hooks.len() {
            if eq_str(hooks[index].name(), name) {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// `str` equality in a `const fn` (`==` is not const-callable on `&str`).
const fn eq_str(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Kind;
    use alloc::vec::Vec;

    #[test]
    fn every_kind_declares_a_non_empty_hook_set_within_the_abi_width() {
        for kind in Kind::ALL {
            let hooks = kind.hooks();
            assert!(!hooks.is_empty(), "{kind} declares no hook");
            assert!(
                hooks.len() <= MAX_HOOKS,
                "{kind} declares more hooks than the ABI width"
            );
            // Names are unique inside a kind: the hook table is a set.
            for (index, hook) in hooks.iter().enumerate() {
                assert!(
                    !hooks[..index]
                        .iter()
                        .any(|earlier| earlier.name() == hook.name()),
                    "{kind} declares {} twice",
                    hook.name()
                );
                assert!(!hook.name().is_empty());
            }
        }
    }

    #[test]
    fn the_lobe_hook_set_matches_the_design_documents_spelling() {
        // §5.4: "lobe: unpack/classify/sample/respond".
        let names: [&str; 4] = ["unpack", "classify", "sample", "respond"];
        let hooks = Kind::Lobe.hooks();
        assert_eq!(hooks.len(), names.len());
        for (hook, name) in hooks.iter().zip(names) {
            assert_eq!(hook.name(), name);
        }
        // The single-named kinds, verbatim from the same sentence.
        assert!(Kind::Provider.declares_hook("produce_page"));
        assert!(Kind::Transfer.declares_hook("remap"));
        assert!(Kind::Composite.declares_hook("record"));
    }

    #[test]
    fn runtime_hooks_owe_a_twin_and_bake_hooks_do_not() {
        for kind in Kind::ALL {
            for hook in kind.hooks() {
                assert_eq!(
                    hook.twin().is_required(),
                    hook.stage() == Stage::Runtime,
                    "{kind}::{} twin policy",
                    hook.name()
                );
            }
        }
        // §11/§5.4.1: a bake-time-only kind has no dual-line obligation at all,
        // which is what makes the offline author-node exception sound.
        for kind in [
            Kind::Transfer,
            Kind::Provider,
            Kind::Downgrade,
            Kind::AuthorNode,
            Kind::Composite,
        ] {
            assert_eq!(kind.runtime_hook_count(), 0, "{kind} is bake-time only");
        }
        // A lobe's behaviour is half bake, half runtime: the layout is frozen by
        // the baker, the response is per pixel.
        assert_eq!(Kind::Lobe.runtime_hook_count(), 2);
        assert_eq!(Kind::Channel.runtime_hook_count(), 1);
        assert_eq!(Kind::ResponseMode.runtime_hook_count(), 1);
        assert_eq!(Kind::CacheAux.runtime_hook_count(), 1);
    }

    #[test]
    fn the_header_digest_holds_only_kinds_the_runtime_reads() {
        for kind in Kind::ALL {
            let expected = matches!(
                kind,
                Kind::Lobe | Kind::Channel | Kind::ResponseMode | Kind::CacheAux
            );
            assert_eq!(kind.header_resident(), expected, "{kind} residency");
        }
        // A provider produces a *page* at bake time; the lobe that consumes it
        // already carries the per-material bit, so the provider needs none.
        assert!(!Kind::Provider.header_resident());
    }

    #[test]
    fn exactly_one_kind_may_restructure_the_class_table() {
        let hot: Vec<Kind> = Kind::ALL
            .into_iter()
            .filter(|kind| kind.touches_class_table())
            .collect();
        assert_eq!(hot, [Kind::ResponseMode], "§5.2's single red-line change");
    }

    #[test]
    fn stage_and_twin_spellings_round_trip_as_constants() {
        assert_eq!(Stage::ALL.len(), 2);
        assert_eq!(Stage::Bake.as_str(), "bake");
        assert_eq!(Stage::Runtime.as_str(), "runtime");
        assert_eq!(Stage::Bake.twin(), Twin::NotRequired);
        assert_eq!(Stage::Runtime.twin(), Twin::ByteCompared);
        assert_eq!(Twin::ByteCompared.as_str(), "byte_compared");
        assert_eq!(Twin::NotRequired.as_str(), "not_required");
    }
}
