//! Behavioural gates for the PCM foundation (§5).

use alloc::boxed::Box;

use crate::builtin::{
    builtin_registry, full_capabilities, register_prism_extension, UID_CHANNEL_SNOW,
    UID_CHANNEL_WETNESS, UID_LOBE_CLEARCOAT, UID_LOBE_FACE,
};
use crate::namespace::Region;
use crate::negotiate::negotiate;
use crate::registry::is_specification_region;
use crate::{
    Capability, CapabilitySet, ExtensionRegistry, ExtensionSet, Kind, Namespace, RegistryError,
    Stability, HEADER_DIGEST_SLOTS,
};

fn caps(list: impl IntoIterator<Item = Capability>) -> CapabilitySet {
    list.into_iter().collect()
}

#[test]
fn the_builtin_catalogue_registers_every_core_lobe_as_a_khr_item() {
    let registry = builtin_registry();
    assert_eq!(
        registry.len(),
        7 + 6,
        "7 KHR lobes + 2 Prism lobes + 4 Prism channels"
    );
    for (index, slot) in [0u32, 1, 2, 3, 4, 5, 6].into_iter().enumerate() {
        let manifest = registry.manifest_at(slot).expect("core lobe slot");
        assert_eq!(manifest.namespace(), Namespace::Khr);
        assert_eq!(manifest.kind(), Kind::Lobe);
        assert_eq!(manifest.stability(), Stability::Stable);
        assert_eq!(manifest.registry_slot(), slot);
        assert_eq!(manifest.origin(), Namespace::Khr);
        assert_eq!(manifest.region(), Region::Specification);
        assert_eq!(manifest.version(), 1, "slot {index} version");
        assert_eq!(
            manifest.id().as_str(),
            match slot {
                0 => "KHR_lobe_emission",
                1 => "KHR_lobe_coat",
                2 => "KHR_lobe_specular_anisotropy",
                3 => "KHR_lobe_fuzz",
                4 => "KHR_lobe_subsurface",
                5 => "KHR_lobe_transmission",
                _ => "KHR_lobe_thin_film",
            }
        );
    }
    // The specification core is portable: no KHR item may require a capability.
    for manifest in registry.iter().filter(|m| m.namespace() == Namespace::Khr) {
        assert!(manifest.requires_caps().is_empty());
    }
}

#[test]
fn the_prism_face_lobe_sits_at_the_extension_region_base() {
    let registry = builtin_registry();
    let face = registry
        .manifest_by_uid(UID_LOBE_FACE)
        .expect("PRISM face lobe");
    assert_eq!(face.registry_slot(), 16, "extension region base (§4.5)");
    assert_eq!(face.region(), Region::Prism);
    assert_eq!(face.namespace(), Namespace::Prism);
    assert_eq!(face.kind(), Kind::Lobe);
    // Pure-OpenPBR projection uses the bit-region predicate, so the face lobe
    // is outside the specification region and is projected away.
    assert!(!is_specification_region(face.registry_slot()));
    assert!(is_specification_region(0));
}

#[test]
fn view_dependence_matches_the_caching_split() {
    let registry = builtin_registry();
    // §14.3: emission and subsurface are view independent and therefore
    // cacheable; every specular/coat/transmission/thin-film lobe is not.
    assert!(!registry.manifest_at(0).unwrap().view_dependent());
    assert!(!registry.manifest_at(4).unwrap().view_dependent());
    for slot in [1u32, 2, 3, 5, 6] {
        assert!(registry.manifest_at(slot).unwrap().view_dependent());
    }
}

#[test]
fn channel_and_lobe_kinds_share_one_namespace_partitioned_slot_space() {
    let registry = builtin_registry();
    let wetness = registry
        .manifest_by_uid(UID_CHANNEL_WETNESS)
        .expect("wetness channel");
    assert_eq!(wetness.kind(), Kind::Channel);
    assert_eq!(wetness.registry_slot(), 18);
    assert_eq!(wetness.region(), Region::Prism);
    let snow = registry
        .manifest_by_uid(UID_CHANNEL_SNOW)
        .expect("snow channel");
    assert_eq!(snow.kind(), Kind::Channel);
    // Kinds share one slot space; only the namespace partitions bits.
    assert_ne!(snow.registry_slot(), wetness.registry_slot());
}

#[test]
fn register_allocates_from_the_namespace_region_and_never_recycles() {
    let mut registry = builtin_registry();
    // The Prism region already holds slots 16..21 (2 lobes + 4 channels), so the
    // next fresh Prism extension lands at 22 and the one after at 23.
    let first = register_prism_extension(&mut registry, "org.prism.lobe.watercolor", Kind::Lobe)
        .expect("prism region has room");
    assert_eq!(first.registry_slot(), 22);
    let second = register_prism_extension(&mut registry, "org.prism.pass.outline", Kind::Composite)
        .expect("prism region has room");
    assert_eq!(second.registry_slot(), 23);
    // A studio extension allocates from its own region, not the Prism one.
    let studio = registry
        .register_new("com.studio.lobe.watercolor", Namespace::Studio, Kind::Lobe)
        .expect("studio region has room");
    assert_eq!(studio.registry_slot(), 48);
    assert_eq!(studio.region(), Region::Studio);
}

#[test]
fn a_full_region_is_an_explicit_pool_exhaustion_not_a_silent_fold_in() {
    let mut registry = ExtensionRegistry::new();
    // Region capacity is 16 slots; the 17th KHR-origin item must fail loudly.
    for index in 0..Region::SLOTS_PER_REGION {
        let uid = alloc::format!("org.test.namespace_partition_fill.{index}");
        let uid: &'static str = Box::leak(uid.into_boxed_str());
        registry
            .register_new(uid, Namespace::Ext, Kind::Transfer)
            .expect("region has room");
    }
    let error = registry
        .register_new("org.test.one_too_many", Namespace::Ext, Kind::Transfer)
        .expect_err("the cross-vendor region is full");
    match error {
        RegistryError::PoolExhausted(exhausted) => {
            assert_eq!(exhausted.origin, Namespace::Ext);
            assert_eq!(exhausted.region, Region::CrossVendor);
            assert_eq!(exhausted.capacity, Region::SLOTS_PER_REGION);
        }
        other => panic!("expected a pool exhaustion, got {other:?}"),
    }
    // The Prism region is untouched by the Ext exhaustion.
    assert_eq!(registry.allocate_slot(Namespace::Prism).unwrap(), 16);
}

#[test]
fn promotion_flips_attribution_and_keeps_identity_bits() {
    let mut registry = builtin_registry();
    let before = registry
        .manifest_by_uid(UID_LOBE_FACE)
        .expect("face lobe")
        .to_owned();
    let promoted = registry
        .promote(UID_LOBE_FACE, Namespace::Khr, "prism", Stability::Stable)
        .expect("PRISM -> KHR is on the promotion chain");

    // Identity is pinned: uid, slot and origin region never move.
    assert_eq!(promoted.uid(), before.uid());
    assert_eq!(promoted.registry_slot(), before.registry_slot());
    assert_eq!(promoted.origin(), Namespace::Prism);
    assert_eq!(promoted.region(), Region::Prism);
    // Attribution flipped.
    assert_eq!(promoted.namespace(), Namespace::Khr);
    assert_eq!(promoted.stability(), Stability::Stable);
    // The display id is derived from the uid's semantic segment. The lobe's uid
    // is pinned to its original spelling (`studio.prism.face_shadow`) because
    // identity is not rewritten by a rename, so the derived name follows the
    // uid, not the schema's human-facing `id`.
    assert_eq!(promoted.id().as_str(), "KHR_lobe_face_shadow");
    // The registry now agrees, and no cross-reference broke.
    assert_eq!(registry.slot_of_uid(UID_LOBE_FACE), Some(16));
    assert!(registry.is_registered(16));
}

#[test]
fn illegal_and_regressing_promotions_are_rejected() {
    let mut registry = builtin_registry();
    // EXT -> PRISM is not on the Studio -> Ext -> KHR chain.
    let error = registry
        .promote(UID_LOBE_FACE, Namespace::Ext, "prism", Stability::Stable)
        .expect_err("PRISM -> EXT is not a legal promotion");
    assert!(matches!(error, RegistryError::Manifest(_)));
    // Promotion may not lower stability.
    let error = registry
        .promote(
            UID_LOBE_CLEARCOAT,
            Namespace::Khr,
            "khronos",
            Stability::Experimental,
        )
        .expect_err("KHR core is already Stable");
    assert!(matches!(error, RegistryError::Manifest(_)));
    // A KHR item may not demote.
    let error = registry
        .promote(
            UID_LOBE_CLEARCOAT,
            Namespace::Prism,
            "prism",
            Stability::Stable,
        )
        .expect_err("KHR never demotes");
    assert!(matches!(error, RegistryError::Manifest(_)));
    assert!(registry
        .promote("org.nobody.here", Namespace::Khr, "x", Stability::Stable)
        .is_err());
}

#[test]
fn manifest_predicates_reject_region_and_capability_violations() {
    use crate::ExtensionManifest;
    // A Prism-origin manifest placed in the specification region is rejected:
    // the bit-region partition is a machine-checked predicate, not a convention.
    let misplaced = ExtensionManifest::new("org.prism.lobe.oops", 3, Namespace::Prism, Kind::Lobe);
    assert!(misplaced.validate().is_err());

    // A specification-core item may not require a platform capability.
    let core_needs_caps =
        ExtensionManifest::new("org.khronos.openpbr.glint", 7, Namespace::Khr, Kind::Lobe)
            .with_requires_caps(CapabilitySet::of(Capability::HeavySampling));
    assert!(core_needs_caps.validate().is_err());

    // A slot outside the fixed-width digest is rejected explicitly.
    let out_of_digest = ExtensionManifest::new(
        "org.prism.lobe.far",
        HEADER_DIGEST_SLOTS,
        Namespace::Prism,
        Kind::Lobe,
    );
    assert!(out_of_digest.validate().is_err());

    // A malformed uid is rejected.
    let bad_uid = ExtensionManifest::new("not a uid", 17, Namespace::Prism, Kind::Lobe);
    assert!(bad_uid.validate().is_err());
}

#[test]
fn negotiation_errors_on_a_missing_required_capability() {
    let registry = builtin_registry();
    let mut requested = ExtensionSet::EMPTY;
    // `snow` needs virtual texturing.
    requested.insert_used(19).unwrap();
    requested.insert_required(19).unwrap();

    let weak = caps([Capability::BindlessTextures]);
    let error = negotiate(&registry, requested, weak).expect_err("snow is unsatisfiable");
    assert_eq!(error.unsupported.len(), 1);
    assert_eq!(error.unsupported[0].slot, 19);
    assert_eq!(
        error.unsupported[0].uid.as_deref(),
        Some("org.prism.channel.snow")
    );
    assert!(error.unsupported[0]
        .missing
        .contains(Capability::VirtualTexture));
    assert_eq!(error.available, weak);

    // With the capability, the same request negotiates losslessly.
    let strong = caps([Capability::BindlessTextures, Capability::VirtualTexture]);
    let negotiation = negotiate(&registry, requested, strong).expect("snow is satisfiable");
    assert!(negotiation.is_lossless());
    assert_eq!(negotiation.extensions(), requested);
}

#[test]
fn negotiation_gracefully_ignores_an_unsatisfiable_used_only_extension() {
    let registry = builtin_registry();
    let mut requested = ExtensionSet::EMPTY;
    requested.insert_used(16).unwrap(); // face lobe: no caps needed
    requested.insert_used(19).unwrap(); // snow: needs VT
    requested.insert_used(20).unwrap(); // damage: no caps needed

    let weak = caps([Capability::BindlessTextures]);
    let negotiation = negotiate(&registry, requested, weak).expect("nothing is required");
    assert!(!negotiation.is_lossless());
    assert_eq!(negotiation.ignored().len(), 1);
    assert_eq!(negotiation.ignored()[0].slot, 19);
    let kept = negotiation.extensions();
    assert!(kept.contains_used(16));
    assert!(kept.contains_used(20));
    assert!(
        !kept.contains_used(19),
        "the unsatisfiable extension is dropped"
    );
    assert!(!kept.contains_required(19));
}

#[test]
fn required_is_always_a_subset_of_used_in_the_digest() {
    let mut set = ExtensionSet::EMPTY;
    set.insert_required(19).unwrap();
    assert!(set.contains_used(19) && set.contains_required(19));
    // A caller cannot construct required without used through the safe API.
    assert!(set.required_is_subset_of(set));
    // A slot outside the digest is rejected by the fallible mutators.
    assert!(set.insert_used(HEADER_DIGEST_SLOTS).is_err());
    assert!(ExtensionSet::from_used_slot(9999).is_err());
}

#[test]
fn the_digest_round_trips_through_its_four_header_words() {
    let mut set = ExtensionSet::EMPTY;
    for slot in [0u32, 1, 16, 47, 48, 63] {
        set.insert_used(slot).unwrap();
    }
    set.insert_required(1).unwrap();
    set.insert_required(63).unwrap();

    let rebuilt = ExtensionSet::from_words(
        set.used_low(),
        set.used_high(),
        set.required_low(),
        set.required_high(),
    );
    assert_eq!(rebuilt, set);
    assert!(rebuilt.contains_used(0));
    assert!(!rebuilt.contains_used(2));
    assert!(rebuilt.contains_required(63));
    assert!(!rebuilt.contains_required(48));
    assert_eq!(rebuilt.used_count(), 6);
}

#[test]
fn intersect_used_prunes_required_too_so_the_invariant_holds() {
    let mut requested = ExtensionSet::EMPTY;
    requested.insert_required(1).unwrap();
    requested.insert_used(16).unwrap();
    // Keeping only the specification region drops both the Prism lobe and its
    // (unset) required bit stays consistent.
    let pruned = requested.intersect_used(0x0000_ffff);
    assert!(pruned.contains_used(1));
    assert!(!pruned.contains_used(16));
    assert!(pruned.required_is_subset_of(pruned));
}

#[test]
fn unknown_slots_negotiate_as_absent_rather_than_breaking_the_digest() {
    let registry = builtin_registry();
    let mut requested = ExtensionSet::EMPTY;
    requested.insert_used(40).unwrap(); // not registered in this build
    let negotiation = negotiate(&registry, requested, full_capabilities())
        .expect("an unknown used-only slot is not required");
    assert_eq!(negotiation.ignored().len(), 1);
    assert!(negotiation.ignored()[0].uid.is_none());
    assert_eq!(negotiation.extensions(), ExtensionSet::EMPTY);

    // The same slot as *required* is a hard error.
    let mut required = ExtensionSet::EMPTY;
    required.insert_required(40).unwrap();
    let error = negotiate(&registry, required, full_capabilities())
        .expect_err("an unregistered required slot cannot be shaded");
    assert_eq!(error.unsupported[0].slot, 40);
    assert!(error.unsupported[0].uid.is_none());
}
