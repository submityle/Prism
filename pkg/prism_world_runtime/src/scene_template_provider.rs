//! Bounded loading of published scene templates through the runtime asset
//! manifest and mounted container overlay.

use core::fmt;

use prism_asset::StableGuid;
use prism_asset_container::{ManifestReadError, ManifestReader, OverlaySet, RuntimeManifest};
use prism_scene::{
    decode_template, SceneTemplate, SchemaMigrationError, SchemaMigrationRegistry,
    TemplateCodecError, TemplateDecodeLimits, SCENE_TEMPLATE_KIND, SCENE_TEMPLATE_SCHEMA,
};

/// Errors returned while resolving a published scene-template route.
#[derive(Debug)]
pub enum SceneTemplateLoadError {
    /// The route does not exist or cannot be bound to the mounted overlay.
    Manifest(ManifestReadError),
    /// The route schema does not match the supported PRST schema.
    Schema {
        /// Required PRST schema.
        expected: u32,
        /// Schema found on the route.
        actual: u32,
    },
    /// A known source schema could not be migrated to the current runtime schema.
    Migration(SchemaMigrationError),
    /// The route schema did not match the payload's declared source schema.
    PayloadSchemaMismatch {
        /// Content schema recorded by the runtime route.
        route: u32,
        /// Content schema declared in the PRST payload.
        payload: u32,
    },
    /// The route kind is not the compiled scene-template artifact kind.
    Kind {
        /// Required scene-template artifact kind.
        expected: u128,
        /// Artifact kind found on the route.
        actual: u128,
    },
    /// The container payload exceeds the caller's bound.
    PayloadTooLarge {
        /// Actual uncompressed payload size.
        actual: u64,
        /// Caller-provided size bound.
        limit: u64,
    },
    /// The bounded payload could not be represented by this platform.
    PayloadLengthOverflow(u64),
    /// The route payload is not a valid bounded PRST template.
    Template(TemplateCodecError),
    /// The decoded template identity differs from the manifest route.
    GuidMismatch {
        /// GUID used to resolve the manifest route.
        route: StableGuid,
        /// GUID decoded from the PRST payload.
        template: StableGuid,
    },
    /// The route dependency closure differs from the dependencies embedded in
    /// the validated scene template.
    DependencyMismatch,
}

impl fmt::Display for SceneTemplateLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(f, "scene template manifest lookup failed: {error}"),
            Self::Schema { expected, actual } => write!(
                f,
                "scene template route schema {actual} is unsupported; expected {expected}"
            ),
            Self::Migration(error) => {
                write!(f, "scene template schema migration failed: {error:?}")
            }
            Self::PayloadSchemaMismatch { route, payload } => write!(
                f,
                "scene template route schema {route} disagrees with payload schema {payload}"
            ),
            Self::Kind { expected, actual } => write!(
                f,
                "scene template route kind {actual:#034x} is unsupported; expected {expected:#034x}"
            ),
            Self::PayloadTooLarge { actual, limit } => write!(
                f,
                "scene template payload is {actual} bytes, exceeding the {limit}-byte limit"
            ),
            Self::PayloadLengthOverflow(length) => write!(
                f,
                "scene template payload length {length} cannot be represented on this platform"
            ),
            Self::Template(error) => write!(f, "invalid scene template payload: {error:?}"),
            Self::GuidMismatch { route, template } => write!(
                f,
                "scene template GUID {template:?} does not match route GUID {route:?}"
            ),
            Self::DependencyMismatch => {
                f.write_str("scene template dependencies disagree with runtime manifest routes")
            }
        }
    }
}

impl std::error::Error for SceneTemplateLoadError {}

/// Loads one published PRST template using a validated runtime manifest.
///
/// The route schema and payload size are checked before allocating the payload
/// buffer. After bounded decoding, the scene GUID and canonical dependency
/// list are cross-checked against the runtime route metadata. Container mmap
/// can avoid copying the container backing, but this API currently decodes
/// PRST into owned scene columns; it does not claim zero-copy scene loading.
pub fn load_scene_template(
    manifest: &RuntimeManifest,
    overlays: &OverlaySet,
    guid: StableGuid,
    max_payload_bytes: u64,
    decode_limits: TemplateDecodeLimits,
) -> Result<SceneTemplate, SceneTemplateLoadError> {
    load_scene_template_with_migrations(
        manifest,
        overlays,
        guid,
        max_payload_bytes,
        decode_limits,
        &SchemaMigrationRegistry::default(),
    )
}

/// Loads a scene template with application-provided semantic migrations.
///
/// The default [`load_scene_template`] includes only the built-in v1-to-v2
/// identity migration. Applications that change component semantics should
/// pass a registry containing their typed conversions here.
pub fn load_scene_template_with_migrations(
    manifest: &RuntimeManifest,
    overlays: &OverlaySet,
    guid: StableGuid,
    max_payload_bytes: u64,
    decode_limits: TemplateDecodeLimits,
    migrations: &SchemaMigrationRegistry,
) -> Result<SceneTemplate, SceneTemplateLoadError> {
    let reader = manifest
        .bind(overlays)
        .map_err(SceneTemplateLoadError::Manifest)?;
    decode_routed_template(
        &reader,
        manifest,
        guid,
        max_payload_bytes,
        decode_limits,
        migrations,
    )
}

fn decode_routed_template(
    reader: &ManifestReader<'_, '_>,
    manifest: &RuntimeManifest,
    guid: StableGuid,
    max_payload_bytes: u64,
    decode_limits: TemplateDecodeLimits,
    migrations: &SchemaMigrationRegistry,
) -> Result<SceneTemplate, SceneTemplateLoadError> {
    let metadata = manifest
        .metadata(guid)
        .ok_or_else(|| SceneTemplateLoadError::Manifest(ManifestReadError::MissingRoute(guid)))?;
    if metadata.kind != SCENE_TEMPLATE_KIND {
        return Err(SceneTemplateLoadError::Kind {
            expected: SCENE_TEMPLATE_KIND,
            actual: metadata.kind,
        });
    }
    if metadata.schema == 0 || metadata.schema > SCENE_TEMPLATE_SCHEMA {
        return Err(SceneTemplateLoadError::Schema {
            expected: SCENE_TEMPLATE_SCHEMA,
            actual: metadata.schema,
        });
    }
    let entry = reader
        .entry_info(guid)
        .map_err(SceneTemplateLoadError::Manifest)?;
    if entry.raw_len > max_payload_bytes {
        return Err(SceneTemplateLoadError::PayloadTooLarge {
            actual: entry.raw_len,
            limit: max_payload_bytes,
        });
    }
    let payload_len = usize::try_from(entry.raw_len)
        .map_err(|_| SceneTemplateLoadError::PayloadLengthOverflow(entry.raw_len))?;
    let payload = reader
        .read_all(guid)
        .map_err(SceneTemplateLoadError::Manifest)?;
    if payload.len() != payload_len {
        return Err(SceneTemplateLoadError::Template(
            TemplateCodecError::Truncated,
        ));
    }
    let template =
        decode_template(&payload, decode_limits).map_err(SceneTemplateLoadError::Template)?;
    if template.schema != metadata.schema {
        return Err(SceneTemplateLoadError::PayloadSchemaMismatch {
            route: metadata.schema,
            payload: template.schema,
        });
    }
    if template.guid != guid {
        return Err(SceneTemplateLoadError::GuidMismatch {
            route: guid,
            template: template.guid,
        });
    }

    let mut routed_dependencies = metadata.hard_dependencies.clone();
    routed_dependencies.extend_from_slice(&metadata.optional_dependencies);
    routed_dependencies.sort_unstable();
    routed_dependencies.dedup();
    if template.deps.as_ref() != routed_dependencies.as_slice() {
        return Err(SceneTemplateLoadError::DependencyMismatch);
    }
    migrations
        .migrate(template, SCENE_TEMPLATE_SCHEMA)
        .map_err(SceneTemplateLoadError::Migration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use prism_asset_container::{ContainerReader, ContainerWriter, ManifestMetadata, OverlaySet};
    use prism_scene::{
        encode_template, ArchetypeGroup, InstantiationPlan, ResidencyPlan, SceneTemplate,
    };

    fn fixture(
        template: &SceneTemplate,
        payload_override: Option<Vec<u8>>,
        schema: u32,
        hard_dependencies: Vec<StableGuid>,
    ) -> (RuntimeManifest, OverlaySet, StableGuid, Vec<u8>) {
        let payload = payload_override.unwrap_or_else(|| encode_template(template).unwrap());
        let guid = template.guid;
        let mut writer = ContainerWriter::new();
        writer.add(guid, payload.clone()).unwrap();
        let bytes = writer.finish().unwrap();
        let reader = ContainerReader::new(bytes).unwrap();
        let mut overlays = OverlaySet::new();
        overlays.mount("scene", 10, reader);
        let mut manifest = overlays.runtime_manifest().unwrap();
        manifest
            .set_metadata(
                guid,
                ManifestMetadata {
                    kind: SCENE_TEMPLATE_KIND,
                    hard_dependencies,
                    schema,
                    ..ManifestMetadata::default()
                },
            )
            .unwrap();
        (manifest, overlays, guid, payload)
    }

    fn template(deps: Vec<StableGuid>) -> SceneTemplate {
        SceneTemplate {
            guid: StableGuid::from_u128(0x5151),
            schema: SCENE_TEMPLATE_SCHEMA,
            plan: InstantiationPlan {
                entity_count: 1,
                groups: vec![ArchetypeGroup::new(
                    prism_scene::ArchetypeSignature::default(),
                    vec![prism_scene::LocalEntityId::new(0)].into_boxed_slice(),
                    Vec::new().into_boxed_slice(),
                )]
                .into_boxed_slice(),
                hierarchy: Vec::new().into_boxed_slice(),
                entity_valued_fields: Vec::new().into_boxed_slice(),
                reserved_local_base: prism_scene::LocalEntityId::RESERVED_BASE,
            },
            names: Default::default(),
            nested: Vec::new().into_boxed_slice(),
            deps: deps.into_boxed_slice(),
            tiers: ResidencyPlan::new(),
        }
    }

    #[test]
    fn loads_valid_template_and_checks_route_contract() {
        let dependency = StableGuid::from_u128(0x7171);
        let template = template(vec![dependency]);
        let (manifest, overlays, guid, _) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA, vec![dependency]);
        let loaded = load_scene_template(
            &manifest,
            &overlays,
            guid,
            1024 * 1024,
            TemplateDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(loaded, template);
    }

    #[test]
    fn rejects_schema_oversize_and_dependency_mismatch_before_acceptance() {
        let template = template(Vec::new());
        let (mut wrong_kind, overlays, guid, _) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA, Vec::new());
        let mut metadata = wrong_kind.metadata(guid).unwrap().clone();
        metadata.kind ^= 1;
        wrong_kind.set_metadata(guid, metadata).unwrap();
        assert!(matches!(
            load_scene_template(
                &wrong_kind,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::Kind { .. })
        ));

        let (manifest, overlays, guid, payload) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA + 1, Vec::new());
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::Schema { .. })
        ));

        let (manifest, overlays, guid, _) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA, Vec::new());
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                payload.len().saturating_sub(1) as u64,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::PayloadTooLarge { .. })
        ));

        let (manifest, overlays, guid, _) = fixture(
            &template,
            None,
            SCENE_TEMPLATE_SCHEMA,
            vec![StableGuid::from_u128(0x8181)],
        );
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::DependencyMismatch)
        ));
    }

    #[test]
    fn rejects_corrupt_prst_and_guid_mismatch() {
        let template = template(Vec::new());
        let (manifest, overlays, guid, _) = fixture(
            &template,
            Some(b"NOPE".to_vec()),
            SCENE_TEMPLATE_SCHEMA,
            Vec::new(),
        );
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::Template(_))
        ));

        let other = SceneTemplate {
            guid: StableGuid::from_u128(0x9999),
            ..template
        };
        let (_, _, _, payload) = fixture(&other, None, SCENE_TEMPLATE_SCHEMA, Vec::new());
        let route_guid = StableGuid::from_u128(0x5151);
        let mut writer = ContainerWriter::new();
        writer.add(route_guid, payload).unwrap();
        let reader = ContainerReader::new(writer.finish().unwrap()).unwrap();
        let mut mismatched_overlays = OverlaySet::new();
        mismatched_overlays.mount("scene", 10, reader);
        let mut mismatched = mismatched_overlays.runtime_manifest().unwrap();
        mismatched
            .set_metadata(
                route_guid,
                ManifestMetadata {
                    kind: SCENE_TEMPLATE_KIND,
                    schema: SCENE_TEMPLATE_SCHEMA,
                    ..ManifestMetadata::default()
                },
            )
            .unwrap();
        let reader = mismatched.bind(&mismatched_overlays).unwrap();
        assert!(matches!(
            decode_routed_template(
                &reader,
                &mismatched,
                route_guid,
                u64::MAX,
                TemplateDecodeLimits::default(),
                &SchemaMigrationRegistry::default()
            ),
            Err(SceneTemplateLoadError::GuidMismatch { .. })
        ));
    }

    #[test]
    fn accepts_application_registered_content_migration() {
        fn migrate_v1_to_v2(
            mut template: SceneTemplate,
        ) -> Result<SceneTemplate, SchemaMigrationError> {
            template.schema = 2;
            Ok(template)
        }

        let mut old = template(Vec::new());
        old.schema = 1;
        let (manifest, overlays, guid, _) = fixture(&old, None, 1, Vec::new());
        let mut migrations = SchemaMigrationRegistry::new();
        migrations
            .register(prism_scene::SchemaMigration {
                from: 1,
                to: 2,
                apply: migrate_v1_to_v2,
            })
            .unwrap();
        let loaded = load_scene_template_with_migrations(
            &manifest,
            &overlays,
            guid,
            u64::MAX,
            TemplateDecodeLimits::default(),
            &migrations,
        )
        .unwrap();
        assert_eq!(loaded.schema, SCENE_TEMPLATE_SCHEMA);
    }
}
