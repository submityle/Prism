//! P1-h parity gate: the committed PCM catalogue and the schema-derived lobe
//! table are **one** truth, not two.
//!
//! `prism_material_pcm::builtin` has to declare the built-in catalogue as
//! `&'static str` constants — it is the loader's dependency and deliberately
//! does not pull in the codegen toolchain. That makes drift *possible*; this
//! gate makes it impossible to merge. It is the §5.4 "single source of truth"
//! claim reduced to an assertion: slot, `uid`, namespace, stability and
//! `requires_caps` must agree row for row, in both directions.

use prism_material_pcm::builtin::builtin_registry;
use prism_material_pcm::{Capability, ExtensionSet, Kind, Namespace, Stability};
use prism_render_material::capability::{
    lobe_registry, parse_capabilities, SPECIFICATION_NAMESPACE,
};
use prism_render_material::{LobeMask, SURFACE_LOBES};

/// Every schema lobe is registered in the committed catalogue with the same
/// identity and governance fields, and the bridge agrees with both.
#[test]
fn the_committed_catalogue_agrees_with_the_schema_row_for_row() {
    let bridged = lobe_registry();
    let committed = builtin_registry();

    assert_eq!(
        committed.len(),
        SURFACE_LOBES.len() + 4,
        "7 KHR lobes + 1 PRISM lobe + the 4 dynamic channels"
    );

    for lobe in &SURFACE_LOBES {
        let schema_row = bridged
            .manifest_at(lobe.registry_slot)
            .unwrap_or_else(|| panic!("schema lobe {} is not in its own registry", lobe.id));
        let committed_row = committed
            .manifest_at(lobe.registry_slot)
            .unwrap_or_else(|| panic!("committed catalogue is missing lobe {}", lobe.id));

        assert_eq!(committed_row.uid(), lobe.uid, "{} uid", lobe.id);
        assert_eq!(schema_row.uid(), lobe.uid, "{} uid", lobe.id);
        assert_eq!(committed_row.kind(), Kind::Lobe, "{} kind", lobe.id);
        assert_eq!(
            committed_row.namespace(),
            Namespace::parse(lobe.namespace).expect("schema namespace spelling"),
            "{} namespace",
            lobe.id
        );
        assert_eq!(
            committed_row.stability(),
            Stability::parse(lobe.stability).expect("schema stability spelling"),
            "{} stability",
            lobe.id
        );
        let requires = parse_capabilities(lobe.requires_caps.iter().copied())
            .unwrap_or_else(|unknown| panic!("{} requires unknown {unknown}", lobe.id));
        assert_eq!(committed_row.requires_caps(), requires, "{} caps", lobe.id);
        assert_eq!(schema_row.requires_caps(), requires, "{} caps", lobe.id);

        // §4.5 constraint 2: the slot is pinned to the *origin* region, so a
        // promotion can never move a bit and never alias the core.
        assert_eq!(
            committed_row.origin(),
            committed_row.namespace(),
            "{} origin",
            lobe.id
        );
        assert_eq!(
            committed_row.registry_slot(),
            lobe.registry_slot,
            "{} slot",
            lobe.id
        );
        assert_eq!(
            schema_row.registry_slot(),
            committed_row.registry_slot(),
            "{} slot",
            lobe.id
        );
    }

    // And nothing in the committed catalogue claims a lobe slot the schema does
    // not define — the direction that would hide an unreviewed extension.
    for manifest in committed.iter().filter(|m| m.kind() == Kind::Lobe) {
        assert!(
            SURFACE_LOBES
                .iter()
                .any(|lobe| lobe.registry_slot == manifest.registry_slot()
                    && lobe.uid == manifest.uid()),
            "committed lobe {} at slot {} has no schema row",
            manifest.uid(),
            manifest.registry_slot()
        );
    }

    // The extension region is **one** slot space (§5.4), so the kinds must not
    // interleave: the schema's `[[lobe]]` table owns the dense lobe sub-range
    // from `EXTENSION_SLOT_BASE`, and every other kind allocates after the last
    // lobe. Without this the next lobe would silently alias a dynamic channel.
    let last_lobe_slot = SURFACE_LOBES
        .iter()
        .filter(|lobe| lobe.namespace != "KHR")
        .map(|lobe| lobe.registry_slot)
        .max()
        .expect("the schema declares at least one extension lobe");
    for manifest in committed.iter().filter(|m| m.kind() != Kind::Lobe) {
        assert!(
            manifest.registry_slot() > last_lobe_slot,
            "{} (slot {}) is not a lobe but sits inside the schema's lobe sub-range 16..={last_lobe_slot}",
            manifest.uid(),
            manifest.registry_slot()
        );
    }
    // …and the lobes are exactly that dense sub-range, not merely a subset.
    let extension_slots: Vec<u32> = SURFACE_LOBES
        .iter()
        .filter(|lobe| lobe.namespace != "KHR")
        .map(|lobe| lobe.registry_slot)
        .collect();
    assert_eq!(
        extension_slots,
        (16..16 + extension_slots.len() as u32).collect::<Vec<u32>>(),
        "extension lobes are dense from the region base (§4.5 constraint 2)"
    );
}

/// The four header words and the derivation are the same function, so a
/// material's digest survives a write/read round trip byte for byte.
#[test]
fn the_header_digest_round_trips_through_its_four_words() {
    for mask in [LobeMask(0), LobeMask::all(), LobeMask(1 << 16), LobeMask(1)] {
        let digest = prism_render_material::capability::digest_for(mask);
        let round_tripped = ExtensionSet::from_words(
            digest.used_low(),
            digest.used_high(),
            digest.required_low(),
            digest.required_high(),
        );
        assert_eq!(round_tripped, digest, "{mask:?} round trip");
        assert!(
            round_tripped.required() & !round_tripped.used() == 0,
            "required must stay a subset of used"
        );
    }
    // A pure-`OpenPBR` mask is a pure-core surface: no extension digest at all,
    // which is what makes the §4.5 constraint-3 projection a no-op on it.
    let pure_core = LobeMask::all().canonical_region();
    let digest = prism_render_material::capability::digest_for(pure_core);
    assert_eq!(digest.used(), 0);
    assert!(pure_core.is_pure_openpbr());
}

/// The specification core is portable: no `KHR` lobe may require a platform
/// capability (§5.3 — otherwise `OpenPBR` would not run everywhere), and every
/// exposed capability name is one an author can actually ask for.
#[test]
fn the_specification_core_requires_no_platform_capability() {
    for lobe in SURFACE_LOBES
        .iter()
        .filter(|lobe| lobe.namespace == SPECIFICATION_NAMESPACE)
    {
        assert!(
            lobe.requires_caps.is_empty(),
            "{} must stay portable",
            lobe.id
        );
    }
    for capability in Capability::ALL {
        assert_eq!(Capability::parse(capability.as_str()), Some(capability));
    }
}
