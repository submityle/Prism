//! P1-h parity gate: the committed PCM catalogue and the schema-derived lobe
//! table are **one** truth, not two.
//!
//! `prism_material_pcm::builtin` has to declare the built-in catalogue as
//! `&'static str` constants — it is the loader's dependency and deliberately
//! does not pull in the codegen toolchain. That makes drift *possible*; this
//! gate makes it impossible to merge. It is the §5.4 "single source of truth"
//! claim reduced to an assertion: slot, `uid`, namespace, stability and
//! `requires_caps` — and the §17.2 `energy_policy`, the rollup of the schema's
//! field-level column — must agree row for row, in both directions.

use prism_material_pcm::builtin::builtin_registry;
use prism_material_pcm::{Capability, EnergyPolicy, ExtensionSet, Kind, Namespace, Stability};
use prism_render_material::capability::{
    lobe_registry, parse_capabilities, SPECIFICATION_NAMESPACE,
};
use prism_render_material::{
    BuiltinProgram, DynamicChannels, LobeMask, SURFACE_CACHE_AUX, SURFACE_CHANNELS,
    SURFACE_IDENTITY_EXTENSIONS, SURFACE_LOBES, SURFACE_RESPONSE_PROGRAMS,
};

/// Every schema lobe is registered in the committed catalogue with the same
/// identity and governance fields, and the bridge agrees with both.
#[test]
fn the_committed_catalogue_agrees_with_the_schema_row_for_row() {
    let bridged = lobe_registry();
    let committed = builtin_registry();

    // The lobe direction is exhaustive (asserted below); the non-lobe rows are
    // counted per kind so this gate does not have to be re-typed every time a
    // response mode, provider or pass is added to the catalogue.
    let count_of = |kind: Kind| committed.iter().filter(|m| m.kind() == kind).count();
    assert_eq!(count_of(Kind::Lobe), SURFACE_LOBES.len(), "schema lobes");
    assert_eq!(count_of(Kind::Channel), 4, "dynamic channels");
    assert_eq!(
        count_of(Kind::ResponseMode),
        SURFACE_RESPONSE_PROGRAMS.len(),
        "response-mode extensions"
    );
    assert_eq!(
        count_of(Kind::CacheAux),
        SURFACE_CACHE_AUX.len(),
        "object-space cache auxiliaries"
    );
    let count_identity = |kind: Kind| {
        SURFACE_IDENTITY_EXTENSIONS
            .iter()
            .filter(|extension| extension.kind == kind.as_str())
            .count()
    };
    assert_eq!(
        count_of(Kind::Provider),
        count_identity(Kind::Provider),
        "bake-time page providers"
    );
    assert_eq!(
        count_of(Kind::Transfer),
        count_identity(Kind::Transfer),
        "transfer remaps"
    );
    assert_eq!(
        count_of(Kind::Downgrade),
        count_identity(Kind::Downgrade),
        "downgrade policies"
    );
    assert_eq!(
        count_of(Kind::AuthorNode),
        count_identity(Kind::AuthorNode),
        "§4's authoring front-end does not exist yet, so no author node can be registered"
    );
    assert_eq!(
        count_of(Kind::Composite),
        count_identity(Kind::Composite),
        "composite passes"
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

        // §17.2: the lobe-level energy policy is generated from the schema's
        // field-level column (codegen `lobe_energy_policy`), so the hand-copied
        // built-in catalogue cannot drift from it.
        let energy_policy = EnergyPolicy::parse(lobe.energy_policy)
            .unwrap_or_else(|| panic!("{} energy-policy spelling", lobe.id));
        assert_eq!(
            committed_row.energy_policy(),
            energy_policy,
            "{} energy policy",
            lobe.id
        );
        assert_eq!(
            schema_row.energy_policy(),
            energy_policy,
            "{} energy policy",
            lobe.id
        );

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
            Some(lobe.registry_slot),
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
        let slot = manifest
            .registry_slot()
            .expect("a lobe is header resident and carries a slot");
        assert!(
            SURFACE_LOBES
                .iter()
                .any(|lobe| lobe.registry_slot == slot && lobe.uid == manifest.uid()),
            "committed lobe {} at slot {} has no schema row",
            manifest.uid(),
            slot
        );
    }

    // The extension region is **one** slot space (§5.4), so the kinds must not
    // interleave: the schema's `[[lobe]]` table owns the dense lobe sub-range
    // from `EXTENSION_SLOT_BASE`, and every other kind allocates after the last
    // lobe. Without this the next lobe would silently alias a dynamic channel.
    // Only header-resident rows hold an address at all; the bake/pass-time rows
    // (`provider`, `transfer`, `downgrade`, `composite`) resolve before the
    // runtime and are registered identity-only (§5.2).
    let last_lobe_slot = SURFACE_LOBES
        .iter()
        .filter(|lobe| lobe.namespace != "KHR")
        .map(|lobe| lobe.registry_slot)
        .max()
        .expect("the schema declares at least one extension lobe");
    for manifest in committed
        .iter()
        .filter(|m| m.kind() != Kind::Lobe && m.is_header_resident())
    {
        let slot = manifest
            .registry_slot()
            .expect("header-resident rows carry a slot");
        assert!(
            slot > last_lobe_slot,
            "{} (slot {}) is not a lobe but sits inside the schema's lobe sub-range 16..={last_lobe_slot}",
            manifest.uid(),
            slot
        );
    }
    // …and a bake/pass-time row must not hold one.
    for manifest in committed.iter().filter(|m| !m.kind().header_resident()) {
        assert_eq!(
            manifest.registry_slot(),
            None,
            "{} resolves at bake/pass time and must not consume a digest slot",
            manifest.uid()
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

/// The generated response-program rows are the runtime's closed built-in
/// response set. They no longer alias PCM response-mode extensions: ABI8
/// carries a verified dense program index, while PCM keeps authoring/catalogue
/// extension identities separate.
#[test]
fn generated_response_programs_mirror_runtime_builtins() {
    let mut declared = Vec::new();
    for mode in &SURFACE_RESPONSE_PROGRAMS {
        let builtin = match mode.builtin {
            "lit" => BuiltinProgram::Lit,
            "npr" => BuiltinProgram::Npr,
            "unlit" => BuiltinProgram::Unlit,
            other => panic!("{} names unknown built-in response {other}", mode.id),
        };
        assert_eq!(
            mode.id, mode.builtin,
            "{} id must name its built-in",
            mode.id
        );
        assert_eq!(mode.registry_slot, 23 + declared.len() as u32);
        assert!(mode.uid.starts_with("org.prism.response_program."));
        assert_eq!(mode.namespace, "PRISM");
        assert_eq!(mode.vendor, "prism");
        assert_eq!(mode.stability, "candidate");
        assert!(mode.requires_caps.is_empty());
        declared.push(builtin);
    }
    assert_eq!(
        declared,
        vec![
            BuiltinProgram::Lit,
            BuiltinProgram::Npr,
            BuiltinProgram::Unlit
        ]
    );

    // The response table is the identity source; no material-side bit axis is
    // involved in selecting a built-in program.
    assert_eq!(declared.len(), 3);
}

/// P2-i: the runtime dynamic channels are declared by the *schema* and
/// registered by the PCM catalogue. This gate ties the two together (uid, order
/// and inert default), and ties both to the header words the ABI ships — the
/// three lists that used to be hand-maintained separately.
#[test]
fn the_schema_channels_match_the_catalogue_and_the_header_words() {
    let committed = builtin_registry();
    for (index, channel) in SURFACE_CHANNELS.iter().enumerate() {
        assert_eq!(channel.header_word, index as u32, "{} word", channel.id);
        let row = committed
            .manifest_by_uid(channel.uid)
            .unwrap_or_else(|| panic!("catalogue is missing channel {}", channel.uid));
        assert_eq!(row.kind(), Kind::Channel, "{} kind", channel.id);
        assert_eq!(
            row.registry_slot(),
            Some(channel.registry_slot),
            "{} slot",
            channel.id
        );
        assert_eq!(
            row.namespace(),
            Namespace::parse(channel.namespace).expect("schema namespace spelling"),
            "{} namespace",
            channel.id
        );
        assert_eq!(row.vendor(), channel.vendor, "{} vendor", channel.id);
        assert_eq!(
            row.stability(),
            Stability::parse(channel.stability).expect("schema stability spelling"),
            "{} stability",
            channel.id
        );
        let requires = parse_capabilities(channel.requires_caps.iter().copied())
            .unwrap_or_else(|unknown| panic!("{} requires unknown {unknown}", channel.id));
        assert_eq!(row.requires_caps(), requires, "{} caps", channel.id);
        assert_eq!(
            row.view_dependent(),
            channel.view_dependent,
            "{} view dependence",
            channel.id
        );
        // A channel is inert at its schema default; every declared default is 0,
        // which is what `DynamicChannels::NONE` packs.
        assert_eq!(channel.default, 0.0, "{} default", channel.id);
    }
    // The header's four dynamic words are exactly the declared channels, in
    // declaration order.
    assert_eq!(SURFACE_CHANNELS.len(), 4);
    assert_eq!(
        SURFACE_CHANNELS.map(|channel| channel.id),
        ["wetness", "snow", "damage", "dissolve"]
    );
    assert_eq!(
        DynamicChannels::from_words(DynamicChannels::NONE.to_words()),
        DynamicChannels::NONE
    );
}

/// P2-i: the object-space cache auxiliaries are declared by the schema and
/// mirrored by the PCM catalogue. Unlike channels they hold no per-material
/// payload, so identity/governance/view-dependence is the whole contract.
#[test]
fn the_schema_cache_aux_matches_the_catalogue() {
    let committed = builtin_registry();
    let mut slots = Vec::new();
    for aux in &SURFACE_CACHE_AUX {
        let row = committed
            .manifest_by_uid(aux.uid)
            .unwrap_or_else(|| panic!("catalogue is missing cache aux {}", aux.uid));
        assert_eq!(row.kind(), Kind::CacheAux, "{} kind", aux.id);
        assert_eq!(
            row.registry_slot(),
            Some(aux.registry_slot),
            "{} slot",
            aux.id
        );
        assert_eq!(
            row.namespace(),
            Namespace::parse(aux.namespace).expect("schema namespace spelling"),
            "{} namespace",
            aux.id
        );
        assert_eq!(row.vendor(), aux.vendor, "{} vendor", aux.id);
        assert_eq!(
            row.stability(),
            Stability::parse(aux.stability).expect("schema stability spelling"),
            "{} stability",
            aux.id
        );
        let requires = parse_capabilities(aux.requires_caps.iter().copied())
            .unwrap_or_else(|unknown| panic!("{} requires unknown {unknown}", aux.id));
        assert_eq!(row.requires_caps(), requires, "{} caps", aux.id);
        assert_eq!(
            row.view_dependent(),
            aux.view_dependent,
            "{} view dependence",
            aux.id
        );
        slots.push(aux.registry_slot);
    }
    // The schema's header-resident non-lobe rows tile the slot range after the
    // last extension lobe: channels, then response modes, then cache auxiliaries.
    let last_lobe_slot = SURFACE_LOBES
        .iter()
        .filter(|lobe| lobe.namespace != "KHR")
        .map(|lobe| lobe.registry_slot)
        .max()
        .expect("at least one extension lobe");
    let mut expected: Vec<u32> = SURFACE_CHANNELS
        .iter()
        .map(|channel| channel.registry_slot)
        .chain(
            SURFACE_RESPONSE_PROGRAMS
                .iter()
                .map(|mode| mode.registry_slot),
        )
        .chain(slots)
        .collect();
    expected.sort_unstable();
    assert_eq!(
        expected,
        (last_lobe_slot + 1..last_lobe_slot + 1 + expected.len() as u32).collect::<Vec<_>>(),
        "header-resident non-lobe slots must be dense after the extension lobes"
    );
}

/// P2-i: the bake/pass-time identity rows are declared by the schema and
/// mirrored by the PCM catalogue. They share the registry's identity and
/// governance space but must never consume a header-digest slot.
#[test]
fn the_schema_identity_extensions_match_the_catalogue() {
    let committed = builtin_registry();
    for extension in &SURFACE_IDENTITY_EXTENSIONS {
        let kind = Kind::parse(extension.kind).expect("schema identity kind spelling");
        assert!(
            matches!(
                kind,
                Kind::Provider
                    | Kind::Transfer
                    | Kind::Downgrade
                    | Kind::AuthorNode
                    | Kind::Composite
            ),
            "{} must be a bake/pass-time kind",
            extension.id
        );
        let row = committed
            .manifest_by_uid(extension.uid)
            .unwrap_or_else(|| panic!("catalogue is missing identity extension {}", extension.uid));
        assert_eq!(row.kind(), kind, "{} kind", extension.id);
        assert_eq!(
            row.registry_slot(),
            None,
            "{} resolves before the runtime and must not hold a slot",
            extension.id
        );
        assert_eq!(
            row.namespace(),
            Namespace::parse(extension.namespace).expect("schema namespace spelling"),
            "{} namespace",
            extension.id
        );
        assert_eq!(row.vendor(), extension.vendor, "{} vendor", extension.id);
        assert_eq!(
            row.stability(),
            Stability::parse(extension.stability).expect("schema stability spelling"),
            "{} stability",
            extension.id
        );
        let requires = parse_capabilities(extension.requires_caps.iter().copied())
            .unwrap_or_else(|unknown| panic!("{} requires unknown {unknown}", extension.id));
        assert_eq!(row.requires_caps(), requires, "{} caps", extension.id);
        assert_eq!(
            row.view_dependent(),
            extension.view_dependent,
            "{} view dependence",
            extension.id
        );
    }
    // The other direction: every identity-only catalogue row has a schema row.
    for manifest in committed.iter().filter(|manifest| {
        matches!(
            manifest.kind(),
            Kind::Provider | Kind::Transfer | Kind::Downgrade | Kind::AuthorNode | Kind::Composite
        )
    }) {
        assert!(
            SURFACE_IDENTITY_EXTENSIONS
                .iter()
                .any(|extension| extension.uid == manifest.uid()),
            "{} has no schema identity-extension row",
            manifest.uid()
        );
    }
}
