//! PCM capability negotiation — the load-time half of §5.3 (roadmap P1-h).
//!
//! The registry itself lives in [`prism_material_pcm`]: one table for all nine
//! extension-point kinds, the namespace/bit-region partition, the promotion
//! state machine and the `used`/`required` digest. This module is the **bridge**
//! that turns the schema-derived lobe catalogue
//! ([`crate::surface_layout::SURFACE_LOBES`]) into that registry and derives the
//! material header's digest — it deliberately owns no vocabulary of its own, so
//! there is exactly one `Capability` enum and one set of spellings in the tree.
//!
//! # The handshake (§5.3)
//!
//! Each manifest declares the **platform capabilities** it needs beyond the
//! shared über path (`LobeLayout::requires_caps`, generated from the schema's
//! `requires_caps`). A material declares which extensions it **uses**
//! ([`GpuMaterialHeader::extensions_used_low`]) and which it **requires**
//! ([`GpuMaterialHeader::extensions_required_low`]). Negotiation follows the
//! Vulkan/glTF split:
//!
//! * a **required** capability whose needs are unmet is a **hard error** — the
//!   material cannot be represented faithfully, so it fails loudly rather than
//!   rendering something else;
//! * a merely **used** capability whose needs are unmet is **gracefully
//!   ignored**: it is dropped from the resolved digest (the material projects to
//!   a variant without it) and reported, exactly like an unsupported glTF
//!   extension.
//!
//! Nothing here runs per pixel: the result is a digest and a mask the baker
//! specializes against, so §5.2's "bake-time monomorphisation, zero runtime
//! dispatch" holds.
//!
//! # What "used" means in the header
//!
//! `extensions_used` is the **non-`KHR`** subset of `lobe_mask`: the `KHR`
//! lobes are the specification core, not extensions, so a pure-`OpenPBR`
//! material carries a zero digest and never enters negotiation at all. That is
//! also what makes the downward projection of §4.5 constraint 3 (drop non-`KHR`
//! items, keep the pure-`OpenPBR` response) a single bitmask operation.

use alloc::vec::Vec;

use prism_material_pcm::negotiate::negotiate;
use prism_material_pcm::{
    CapabilitySet, ExtensionManifest, ExtensionRegistry, ExtensionSet, Kind, Namespace,
    Negotiation, NegotiationError, Stability,
};

use crate::surface_layout::SURFACE_LOBES;
use crate::LobeMask;

pub use prism_material_pcm::{Capability, UnsupportedExtension};

/// The namespace spelling that marks the `OpenPBR` specification core.
pub const SPECIFICATION_NAMESPACE: &str = "KHR";

/// The schema's `stability` spelling that makes an extension part of the
/// engine's contract: present ⇒ required.
pub const REQUIRED_STABILITY: &str = "stable";

/// Parse a schema `requires_caps` entry into the PCM capability set.
///
/// A name the crate does not know is treated as **unsatisfiable**, never as
/// "assume present": silently dropping a requirement would turn a missing
/// capability into a rendering difference instead of a diagnosable error.
///
/// # Errors
///
/// Returns the offending name when it is not a known capability.
pub fn parse_capabilities<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<CapabilitySet, &'a str> {
    let mut set = CapabilitySet::EMPTY;
    for name in names {
        let capability = Capability::parse(name).ok_or(name)?;
        set.insert(capability);
    }
    Ok(set)
}

/// The lobe catalogue as a PCM registry: one `kind = Lobe` manifest per row of
/// [`crate::surface_layout::SURFACE_LOBES`].
///
/// This is a *projection*, not a second source of truth — it reads `uid`,
/// `registry_slot`, `namespace`, `stability` and `requires_caps` straight out of
/// the generated table, and `tests/pcm_parity.rs` asserts the committed PCM
/// catalogue agrees with it row for row.
///
/// # Panics
///
/// Only if the compiled-in schema violates a §5.4 predicate (a slot outside its
/// namespace's region, a `KHR` lobe declaring a platform requirement, a
/// duplicate `uid`). That is a build-time programming error, caught by the
/// schema tests and the parity gate, so panicking here is strictly better than
/// silently dropping a lobe from negotiation.
#[must_use]
pub fn lobe_registry() -> ExtensionRegistry {
    let mut registry = ExtensionRegistry::new();
    for lobe in &SURFACE_LOBES {
        let origin = Namespace::parse(lobe.namespace).unwrap_or_else(|| {
            panic!(
                "lobe {:?} declares unknown PCM namespace {:?}",
                lobe.id, lobe.namespace
            )
        });
        let stability = Stability::parse(lobe.stability).unwrap_or_else(|| {
            panic!(
                "lobe {:?} declares unknown PCM stability {:?}",
                lobe.id, lobe.stability
            )
        });
        let requires_caps =
            parse_capabilities(lobe.requires_caps.iter().copied()).unwrap_or_else(|unknown| {
                panic!("lobe {:?} requires unknown capability {unknown:?}", lobe.id)
            });
        let manifest = ExtensionManifest::new(lobe.uid, lobe.registry_slot, origin, Kind::Lobe)
            .with_stability(stability)
            .with_requires_caps(requires_caps);
        registry
            .register(manifest)
            .unwrap_or_else(|error| panic!("lobe {:?} is not registrable: {error}", lobe.id));
    }
    registry
}

/// The §5.3 digest for a material carrying `lobe_mask`.
///
/// `used` is the non-`KHR` subset of the mask; `required` is the subset of that
/// which the schema marks `stable` — a stable extension is part of the engine's
/// contract, so a material carrying one *requires* it, while an experimental one
/// degrades gracefully. Slots outside the digest cannot occur: `lobe_mask` is
/// derived from this very table, and every slot it can name is ≤
/// [`prism_material_pcm::HEADER_DIGEST_SLOTS`].
#[must_use]
pub fn digest_for(lobe_mask: LobeMask) -> ExtensionSet {
    let mut set = ExtensionSet::EMPTY;
    for lobe in &SURFACE_LOBES {
        if lobe_mask.bits() & (1u32 << lobe.registry_slot) == 0 {
            continue;
        }
        if lobe.namespace == SPECIFICATION_NAMESPACE {
            continue;
        }
        set.insert_used(lobe.registry_slot)
            .expect("a schema lobe slot is inside the fixed-width digest");
        if lobe.stability == REQUIRED_STABILITY {
            set.insert_required(lobe.registry_slot)
                .expect("a schema lobe slot is inside the fixed-width digest");
        }
    }
    set
}

/// Negotiate a material digest against a target's capability set.
///
/// # Errors
///
/// Returns [`NegotiationError`] when a **required** extension's capabilities are
/// missing. Used-only extensions that cannot be honoured come back through
/// [`Negotiation::ignored`] instead, with the resolved digest already projected.
pub fn negotiate_digest(
    digest: ExtensionSet,
    available: CapabilitySet,
) -> Result<Negotiation, NegotiationError> {
    negotiate(&lobe_registry(), digest, available)
}

/// The non-`KHR` lobes of `lobe_mask`, in registry order — the ids an
/// exporter/tool reports when an extension is projected away.
#[must_use]
pub fn extension_ids(lobe_mask: LobeMask) -> Vec<&'static str> {
    SURFACE_LOBES
        .iter()
        .filter(|lobe| lobe.namespace != SPECIFICATION_NAMESPACE)
        .filter(|lobe| lobe_mask.bits() & (1u32 << lobe.registry_slot) != 0)
        .map(|lobe| lobe.id)
        .collect()
}

#[cfg(all(test, not(feature = "bevy")))]
mod tests {
    use super::*;
    use prism_material_pcm::negotiate::negotiate;

    /// A synthetic registry entry with a real capability requirement: every
    /// *shipping* lobe is pure ALU today, so the mechanism is exercised with an
    /// explicit need rather than a fake one being added to the schema.
    fn synthetic(requires_caps: &[&str], stability: Stability) -> ExtensionRegistry {
        let mut registry = ExtensionRegistry::new();
        registry
            .register(
                ExtensionManifest::new("studio.prism.test_lobe", 16, Namespace::Prism, Kind::Lobe)
                    .with_stability(stability)
                    .with_requires_caps(parse_capabilities(requires_caps.iter().copied()).unwrap()),
            )
            .unwrap();
        registry
    }

    #[test]
    fn the_shipping_catalogue_registers_every_lobe_as_a_pcm_item() {
        let registry = lobe_registry();
        assert_eq!(registry.len(), SURFACE_LOBES.len());
        for lobe in &SURFACE_LOBES {
            let manifest = registry
                .manifest_at(lobe.registry_slot)
                .expect("every schema lobe is registered");
            assert_eq!(manifest.uid(), lobe.uid);
            assert_eq!(manifest.namespace().as_str(), lobe.namespace);
            assert_eq!(manifest.kind(), Kind::Lobe);
            assert_eq!(manifest.stability().as_str(), lobe.stability);
        }
    }

    #[test]
    fn baseline_path_needs_no_capabilities() {
        // Every shipping lobe is pure ALU: the table proves the mechanism is
        // *available* without pretending the engine already needs it.
        assert!(SURFACE_LOBES
            .iter()
            .all(|lobe| lobe.requires_caps.is_empty()));
        let digest = digest_for(LobeMask::all());
        let out = negotiate_digest(digest, CapabilitySet::EMPTY)
            .expect("a capability-free catalogue negotiates on any target");
        assert_eq!(out.extensions().used(), digest.used());
        assert!(out.is_lossless());
    }

    #[test]
    fn the_digest_is_the_non_khr_subset_of_the_lobe_mask() {
        // A pure-`OpenPBR` material carries no extension digest at all.
        assert_eq!(digest_for(LobeMask(0)).used(), 0);
        // The `KHR` lobes stay out of it, so the pure-core projection is a
        // no-op.
        for lobe in SURFACE_LOBES
            .iter()
            .filter(|lobe| lobe.namespace == SPECIFICATION_NAMESPACE)
        {
            let digest = digest_for(LobeMask(1 << lobe.registry_slot));
            assert_eq!(digest.used(), 0, "{} is specification core", lobe.id);
        }
        // A non-`KHR` lobe lands in `used`, and — since NPR_FACE is
        // experimental — not in `required`.
        let face = SURFACE_LOBES
            .iter()
            .find(|lobe| lobe.namespace != SPECIFICATION_NAMESPACE)
            .expect("the schema carries at least one extension lobe");
        let digest = digest_for(LobeMask(1u32 << face.registry_slot));
        assert!(digest.contains_used(face.registry_slot));
        assert_eq!(
            digest.contains_required(face.registry_slot),
            face.stability == REQUIRED_STABILITY
        );
    }

    #[test]
    fn a_required_capability_that_is_missing_is_a_hard_error() {
        let registry = synthetic(&["compute_scatter"], Stability::Stable);
        let mut digest = ExtensionSet::EMPTY;
        digest.insert_required(16).unwrap();
        let error = negotiate(&registry, digest, CapabilitySet::EMPTY)
            .expect_err("a required extension with an unmet need must fail");
        assert_eq!(error.unsupported.len(), 1);
        assert!(error.unsupported[0]
            .missing
            .contains(Capability::ComputeScatter));
        // Supplying the capability resolves it.
        let resolved = negotiate(
            &registry,
            digest,
            CapabilitySet::of(Capability::ComputeScatter),
        )
        .expect("capability present");
        assert_eq!(resolved.extensions().required(), digest.required());
    }

    #[test]
    fn a_used_only_capability_is_gracefully_ignored() {
        let registry = synthetic(&["virtual_texture"], Stability::Experimental);
        let digest = ExtensionSet::from_used_slot(16).unwrap();
        let out = negotiate(&registry, digest, CapabilitySet::EMPTY)
            .expect("an optional extension degrades instead of failing");
        assert_eq!(out.extensions().used(), 0);
        assert_eq!(out.ignored().len(), 1);
        assert_eq!(out.ignored()[0].slot, 16);
        assert!(out.ignored()[0]
            .missing
            .contains(Capability::VirtualTexture));
    }

    #[test]
    fn capability_names_round_trip_and_unknown_names_are_rejected() {
        for capability in Capability::ALL {
            assert_eq!(Capability::parse(capability.as_str()), Some(capability));
        }
        assert!(parse_capabilities(["object_space_cache", "storage_f16"]).is_ok());
        assert_eq!(
            parse_capabilities(["no_such_cap"]),
            Err("no_such_cap"),
            "an unknown requirement is a schema bug, not a silent pass"
        );
    }
}
