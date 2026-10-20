//! Behavioural gates for the PCM foundation (§5).

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::builtin::{
    builtin_author_node, builtin_energy_policy, builtin_registry, full_capabilities,
    register_prism_extension, BUILTIN_AUTHOR_NODES, UID_CACHE_AUX_SURFEL, UID_CHANNEL_DAMAGE,
    UID_CHANNEL_DISSOLVE, UID_CHANNEL_SNOW, UID_CHANNEL_WETNESS, UID_COMPOSITE_OUTLINE,
    UID_DOWNGRADE_SAMPLE_BUDGET, UID_LOBE_ANISOTROPY, UID_LOBE_CLEARCOAT, UID_LOBE_EMISSION,
    UID_LOBE_FACE, UID_LOBE_GLINT, UID_LOBE_SHEEN, UID_LOBE_SUBSURFACE, UID_LOBE_THIN_FILM,
    UID_LOBE_TRANSMISSION, UID_MODE_LIT, UID_MODE_NPR, UID_MODE_UNLIT,
    UID_PROVIDER_FACE_SHADOW_SDF, UID_PROVIDER_MEASURED_BRDF, UID_TRANSFER_SRGB,
};
use crate::namespace::Region;
use crate::negotiate::negotiate;
use crate::registry::is_specification_region;
use crate::{
    Capability, CapabilitySet, EnergyPolicy, ExtensionManifest, ExtensionRegistry, ExtensionSet,
    Kind, Namespace, RegistryError, Stability, Stage, HEADER_DIGEST_SLOTS,
};

fn caps(list: impl IntoIterator<Item = Capability>) -> CapabilitySet {
    list.into_iter().collect()
}

#[test]
fn the_builtin_catalogue_registers_every_core_lobe_as_a_khr_item() {
    let registry = builtin_registry();
    assert_eq!(
        registry.len(),
        7 + 7 + 4 + 5,
        "7 KHR lobes + 3 Prism lobes + 4 channels + 3 response modes + 1 cache aux \
         + 5 bake/pass-time rows"
    );
    for (index, slot) in [0u32, 1, 2, 3, 4, 5, 6].into_iter().enumerate() {
        let manifest = registry.manifest_at(slot).expect("core lobe slot");
        assert_eq!(manifest.namespace(), Namespace::Khr);
        assert_eq!(manifest.kind(), Kind::Lobe);
        assert_eq!(manifest.stability(), Stability::Stable);
        assert_eq!(manifest.registry_slot(), Some(slot));
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
    assert_eq!(
        face.registry_slot(),
        Some(16),
        "extension region base (§4.5)"
    );
    assert_eq!(face.region(), Region::Prism);
    assert_eq!(face.namespace(), Namespace::Prism);
    assert_eq!(face.kind(), Kind::Lobe);
    // Pure-OpenPBR projection uses the bit-region predicate, so the face lobe
    // is outside the specification region and is projected away.
    assert!(!is_specification_region(
        face.registry_slot()
            .expect("the face lobe is header resident")
    ));
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
    assert_eq!(wetness.registry_slot(), Some(19));
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
    // The Prism region already holds 16..26 (3 lobes + 4 channels + 3 response
    // modes + the cache aux), so the next fresh Prism extension lands at 27 and
    // the one after at 28.
    let first = register_prism_extension(&mut registry, "org.prism.lobe.watercolor", Kind::Lobe)
        .expect("prism region has room");
    assert_eq!(first.registry_slot(), Some(27));
    let second = register_prism_extension(
        &mut registry,
        "org.prism.mode.watercolor",
        Kind::ResponseMode,
    )
    .expect("prism region has room");
    assert_eq!(second.registry_slot(), Some(28));
    // A bake/pass-time kind needs no address at all, so it cannot exhaust the
    // region and does not move the allocator (§5.2).
    let outline = register_prism_extension(
        &mut registry,
        "org.prism.composite.watercolor_edges",
        Kind::Composite,
    )
    .expect("an identity-only registration never runs out of slots");
    assert_eq!(outline.registry_slot(), None);
    assert_eq!(registry.allocate_slot(Namespace::Prism).unwrap(), 29);
    assert_eq!(
        registry
            .manifest_by_uid("org.prism.composite.watercolor_edges")
            .expect("identity-only rows are registered, just unaddressed")
            .id()
            .as_str(),
        "PRISM_composite_watercolor_edges"
    );
    // A studio extension allocates from its own region, not the Prism one.
    let studio = registry
        .register_new("com.studio.lobe.watercolor", Namespace::Studio, Kind::Lobe)
        .expect("studio region has room");
    assert_eq!(studio.registry_slot(), Some(48));
    assert_eq!(studio.region(), Region::Studio);
}

#[test]
fn a_full_region_is_an_explicit_pool_exhaustion_not_a_silent_fold_in() {
    let mut registry = ExtensionRegistry::new();
    // Region capacity is 16 slots; the 17th slot-resident Ext-origin item must
    // fail loudly. (`Kind::Lobe` is header resident, so it needs a slot.)
    for index in 0..Region::SLOTS_PER_REGION {
        let uid = alloc::format!("org.test.namespace_partition_fill.{index}");
        let uid: &'static str = Box::leak(uid.into_boxed_str());
        registry
            .register_new(uid, Namespace::Ext, Kind::Lobe)
            .expect("region has room");
    }
    let error = registry
        .register_new("org.test.one_too_many", Namespace::Ext, Kind::Lobe)
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
fn identity_only_rows_promote_in_place_without_gaining_a_slot() {
    let mut registry = builtin_registry();
    let before = registry
        .manifest_by_uid(UID_TRANSFER_SRGB)
        .expect("the srgb transfer is registered")
        .to_owned();
    assert_eq!(
        before.registry_slot(),
        None,
        "a transfer resolves at bake/pass time and holds no digest slot"
    );

    // A registered identity-only row promotes — it must not be reported as an
    // unknown uid, which would hide a real extension behind a false diagnostic
    // and make every bake/pass-time kind un-promotable.
    let promoted = registry
        .promote(
            UID_TRANSFER_SRGB,
            Namespace::Khr,
            "khronos",
            Stability::Stable,
        )
        .expect("a registered identity-only row is promotable");

    // Identity and residency are pinned: promotion flips attribution only.
    assert_eq!(promoted.uid(), before.uid());
    assert_eq!(promoted.origin(), Namespace::Prism);
    assert_eq!(
        promoted.registry_slot(),
        None,
        "promotion never mints a slot"
    );
    assert_eq!(promoted.kind(), Kind::Transfer);
    // Attribution flipped.
    assert_eq!(promoted.namespace(), Namespace::Khr);
    assert_eq!(promoted.stability(), Stability::Stable);
    // The identity-only store was rewritten in place, not shadowed: the row is
    // still discoverable by uid, the count is unchanged, and no slot appeared.
    assert_eq!(
        registry
            .manifest_by_uid(UID_TRANSFER_SRGB)
            .map(ExtensionManifest::namespace),
        Some(Namespace::Khr)
    );
    assert_eq!(registry.identity_only().len(), 5);
    assert_eq!(registry.len(), 23);
    assert_eq!(registry.slot_of_uid(UID_TRANSFER_SRGB), None);
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
    requested.insert_used(20).unwrap();
    requested.insert_required(20).unwrap();

    let weak = caps([Capability::BindlessTextures]);
    let error = negotiate(&registry, requested, weak).expect_err("snow is unsatisfiable");
    assert_eq!(error.unsupported.len(), 1);
    assert_eq!(error.unsupported[0].slot, 20);
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
    requested.insert_used(20).unwrap(); // snow: needs VT
    requested.insert_used(21).unwrap(); // damage: no caps needed

    let weak = caps([Capability::BindlessTextures]);
    let negotiation = negotiate(&registry, requested, weak).expect("nothing is required");
    assert!(!negotiation.is_lossless());
    assert_eq!(negotiation.ignored().len(), 1);
    assert_eq!(negotiation.ignored()[0].slot, 20);
    let kept = negotiation.extensions();
    assert!(kept.contains_used(16));
    assert!(kept.contains_used(21));
    assert!(
        !kept.contains_used(20),
        "the unsatisfiable extension is dropped"
    );
    assert!(!kept.contains_required(20));
}

#[test]
fn required_is_always_a_subset_of_used_in_the_digest() {
    let mut set = ExtensionSet::EMPTY;
    set.insert_required(20).unwrap();
    assert!(set.contains_used(20) && set.contains_required(20));
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

/// §5.2: the header digest is per-material runtime state, so only a kind the
/// runtime reads may hold a slot — and every such kind must hold one.
#[test]
fn residency_is_a_machine_checked_predicate_in_both_directions() {
    // A bake/pass-time kind with a slot is rejected…
    let addressed = ExtensionManifest::new(
        "org.prism.composite.oops",
        17,
        Namespace::Prism,
        Kind::Composite,
    );
    assert!(matches!(
        addressed.validate(),
        Err(crate::ManifestError::SlotForIdentityOnlyKind {
            kind: Kind::Composite
        })
    ));
    // …and a runtime-read kind without one is rejected too: silently addressing
    // nothing would make the material's digest claim a fact it cannot carry.
    let unaddressed = ExtensionManifest::new_identity_only(
        "org.prism.mode.oops",
        Namespace::Prism,
        Kind::ResponseMode,
    );
    assert!(matches!(
        unaddressed.validate(),
        Err(crate::ManifestError::MissingSlotForHeaderResidentKind {
            kind: Kind::ResponseMode
        })
    ));
    // The two legal shapes validate.
    assert!(addressed_slot_manifest().validate().is_ok());
    assert!(identity_only_manifest().validate().is_ok());
}

fn addressed_slot_manifest() -> ExtensionManifest {
    // A header-resident kind with a slot in its origin's own region.
    ExtensionManifest::new(
        "org.prism.cache_aux.surfel",
        Region::Prism.base(),
        Namespace::Prism,
        Kind::CacheAux,
    )
}

fn identity_only_manifest() -> ExtensionManifest {
    ExtensionManifest::new_identity_only(
        "org.prism.composite.outline",
        Namespace::Prism,
        Kind::Composite,
    )
}

/// §17.2: only a lobe carries energy, and the lobe-level policy is the rollup of
/// the schema's field-level column.
#[test]
fn energy_policies_are_lobe_only_and_roll_up_by_priority() {
    // The rollup is order independent and `coupled` dominates.
    assert_eq!(
        EnergyPolicy::roll_up([
            EnergyPolicy::None,
            EnergyPolicy::Independent,
            EnergyPolicy::Coupled
        ]),
        EnergyPolicy::Coupled
    );
    assert_eq!(
        EnergyPolicy::roll_up([EnergyPolicy::Independent, EnergyPolicy::None]),
        EnergyPolicy::Independent
    );
    assert_eq!(EnergyPolicy::roll_up([]), EnergyPolicy::None);
    assert_eq!(
        EnergyPolicy::roll_up([EnergyPolicy::None, EnergyPolicy::None]),
        EnergyPolicy::None
    );

    // A non-lobe kind may not declare energy: the budget is a lobe concern.
    let composite = ExtensionManifest::new_identity_only(
        "org.prism.composite.oops",
        Namespace::Prism,
        Kind::Composite,
    )
    .with_energy_policy(EnergyPolicy::Coupled);
    assert!(matches!(
        composite.validate(),
        Err(crate::ManifestError::EnergyPolicyOutsideLobe {
            kind: Kind::Composite,
            policy: EnergyPolicy::Coupled
        })
    ));

    // The built-in column, row for row: emission adds its own radiance, the
    // stylized face terminator carries none, and every physical lobe that layers
    // over base shares the coupled budget.
    let independent = [UID_LOBE_EMISSION];
    let none = [UID_LOBE_FACE];
    let coupled = [
        UID_LOBE_CLEARCOAT,
        UID_LOBE_ANISOTROPY,
        UID_LOBE_SHEEN,
        UID_LOBE_SUBSURFACE,
        UID_LOBE_TRANSMISSION,
        UID_LOBE_THIN_FILM,
        UID_LOBE_GLINT,
    ];
    for uid in independent {
        assert_eq!(
            builtin_energy_policy(Kind::Lobe, uid),
            EnergyPolicy::Independent,
            "{uid}"
        );
    }
    for uid in none {
        assert_eq!(
            builtin_energy_policy(Kind::Lobe, uid),
            EnergyPolicy::None,
            "{uid}"
        );
    }
    for uid in coupled {
        assert_eq!(
            builtin_energy_policy(Kind::Lobe, uid),
            EnergyPolicy::Coupled,
            "{uid}"
        );
    }
    // A non-lobe kind is `None` by construction, so a caller cannot accidentally
    // hand a channel an energy budget.
    for kind in Kind::ALL.into_iter().filter(|kind| *kind != Kind::Lobe) {
        assert_eq!(
            builtin_energy_policy(kind, UID_LOBE_CLEARCOAT),
            EnergyPolicy::None,
            "{kind}"
        );
    }

    // …and the registry carries that column, so a consumer never re-derives it.
    let registry = builtin_registry();
    for lobe in registry.iter().filter(|m| m.kind() == Kind::Lobe) {
        assert_eq!(
            lobe.energy_policy(),
            builtin_energy_policy(lobe.kind(), lobe.uid()),
            "{} energy policy",
            lobe.uid()
        );
        assert_eq!(
            lobe.energy_policy() != EnergyPolicy::None,
            lobe.energy_policy().is_energy_carrying()
        );
    }
    // A non-lobe row never carries energy.
    for other in registry.iter().filter(|m| m.kind() != Kind::Lobe) {
        assert_eq!(other.energy_policy(), EnergyPolicy::None, "{}", other.uid());
    }
}

/// Every extension-point kind is represented by at least one real extension.
/// Surface/runtime rows live in `builtin_registry`; author nodes live in their
/// bake-only catalogue because they never become surface-header identities.
#[test]
fn the_catalogue_covers_every_kind_that_has_an_implementation() {
    let registry = builtin_registry();
    for kind in Kind::ALL {
        let rows: Vec<&str> = registry
            .iter()
            .filter(|m| m.kind() == kind)
            .map(ExtensionManifest::uid)
            .collect();
        if kind == Kind::AuthorNode {
            assert!(
                rows.is_empty(),
                "author nodes must not enter the surface catalogue"
            );
            assert!(!BUILTIN_AUTHOR_NODES.is_empty());
            continue;
        }
        assert!(!rows.is_empty(), "{kind} has no registered extension");
    }

    // The residency split, kind by kind: an identity-only row is registered but
    // holds no header address.
    for uid in [
        UID_PROVIDER_FACE_SHADOW_SDF,
        UID_PROVIDER_MEASURED_BRDF,
        UID_TRANSFER_SRGB,
        UID_DOWNGRADE_SAMPLE_BUDGET,
        UID_COMPOSITE_OUTLINE,
    ] {
        let manifest = registry.manifest_by_uid(uid).expect("registered");
        assert_eq!(manifest.registry_slot(), None, "{uid} must be unaddressed");
        assert!(!manifest.is_header_resident());
        assert!(registry.slot_of_uid(uid).is_none());
    }
    for uid in [
        UID_MODE_LIT,
        UID_MODE_NPR,
        UID_MODE_UNLIT,
        UID_CACHE_AUX_SURFEL,
        UID_CHANNEL_WETNESS,
        UID_CHANNEL_SNOW,
        UID_CHANNEL_DAMAGE,
        UID_CHANNEL_DISSOLVE,
    ] {
        let manifest = registry.manifest_by_uid(uid).expect("registered");
        assert!(manifest.is_header_resident(), "{uid} must own a slot");
        assert!(manifest.registry_slot().is_some_and(|slot| slot < 32));
    }

    // The response modes mirror the runtime's `Illumination` classes one for
    // one: `prism_render_material`'s parity gate asserts the other direction.
    let modes: Vec<&str> = registry
        .iter()
        .filter(|m| m.kind() == Kind::ResponseMode)
        .map(ExtensionManifest::uid)
        .collect();
    assert_eq!(modes, [UID_MODE_LIT, UID_MODE_NPR, UID_MODE_UNLIT]);
}

#[test]
fn materialx_author_nodes_have_unique_identity_only_bake_manifests() {
    let mut previous_category = None;
    let mut uids = alloc::collections::BTreeSet::new();
    for entry in &BUILTIN_AUTHOR_NODES {
        if let Some(previous) = previous_category {
            assert!(previous < entry.category(), "catalogue must stay sorted");
        }
        previous_category = Some(entry.category());
        assert!(
            uids.insert(entry.manifest().uid()),
            "duplicate author-node uid"
        );
        assert_eq!(builtin_author_node(entry.category()), Some(entry));

        let manifest = entry.manifest();
        manifest.validate().expect("author-node manifest is valid");
        assert_eq!(manifest.kind(), Kind::AuthorNode);
        assert_eq!(manifest.namespace(), Namespace::Prism);
        assert_eq!(manifest.stability(), Stability::Candidate);
        assert_eq!(manifest.registry_slot(), None);
        assert!(!manifest.is_header_resident());
        assert_eq!(manifest.requires_caps(), CapabilitySet::EMPTY);
        assert!(!manifest.view_dependent());
    }
    assert!(builtin_author_node("noise").is_none());

    let hooks = Kind::AuthorNode.hooks();
    assert_eq!(hooks.len(), 1);
    assert_eq!(hooks[0].name(), "evaluate");
    assert_eq!(hooks[0].stage(), Stage::Bake);
    assert!(!hooks[0].twin().is_required());
    assert_eq!(Kind::AuthorNode.runtime_hook_count(), 0);
}

/// A bake/pass-time row still negotiates its capabilities — just by name,
/// because it has no slot to be requested by (§5.2/§5.3).
#[test]
fn bake_time_rows_negotiate_capabilities_by_uid() {
    let registry = builtin_registry();
    let needs = registry.required_caps_of_uids([UID_PROVIDER_MEASURED_BRDF]);
    assert!(needs.contains(Capability::HeavySampling));
    assert!(needs.is_subset_of(full_capabilities()));
    // An unknown name contributes nothing rather than panicking: the caller that
    // asks for a uid it cannot find has a bug, and the registry's job is to make
    // the *requirement* visible, not to guess.
    assert!(registry
        .required_caps_of_uids(["org.prism.provider.nobody"])
        .is_empty());
}

/// The extension region still has room after the whole catalogue lands — the
/// identity-only split exists precisely so that a kind cannot eat the digest.
#[test]
fn the_catalogue_leaves_header_digest_headroom_for_growth() {
    let mut registry = builtin_registry();
    let prism_used = registry
        .iter()
        .filter_map(ExtensionManifest::registry_slot)
        .filter(|slot| Region::Prism.contains(*slot))
        .count();
    assert!(
        prism_used < Region::SLOTS_PER_REGION as usize,
        "the Prism region must keep at least one address free"
    );
    // Prove it: a fresh header-resident Prism extension still registers.
    let fresh = register_prism_extension(&mut registry, "org.prism.lobe.reserve_probe", Kind::Lobe)
        .expect("the Prism region has headroom");
    assert!(fresh.is_header_resident());
}

/// The hook ABI and the residency predicate come from the same table: a kind
/// with a runtime hook the runtime never reads would be a contradiction.
#[test]
fn runtime_hooks_and_header_residency_agree_for_every_kind() {
    for kind in Kind::ALL {
        for hook in kind.hooks() {
            if hook.stage() == Stage::Runtime {
                assert!(
                    kind.header_resident() || kind == Kind::Provider,
                    "{kind}::{} runs at runtime but the runtime never reads it",
                    hook.name()
                );
            }
        }
    }
    // `transfer`/`downgrade`/`author_node`/`composite` are bake-time through and
    // through, which is exactly why they need no header address.
    for kind in [
        Kind::Transfer,
        Kind::Downgrade,
        Kind::AuthorNode,
        Kind::Composite,
    ] {
        assert_eq!(kind.runtime_hook_count(), 0);
        assert!(!kind.header_resident());
    }
}
