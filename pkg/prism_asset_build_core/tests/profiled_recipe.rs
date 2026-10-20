use prism_asset_build_core::profile::{
    EffectiveProfile, ProfileError, ProfileField, ProfileFieldType, ProfileLayer,
    ProfileLayerValues, ProfileMergePolicy, ProfileSchema, ProfileValue,
};
use prism_asset_build_core::{
    ArtifactSpec, AssetId, BuildExecutor, BuildGraph, BuildInputSnapshot, BuildRevision,
    DerivedDataCache, ImportProcessor, ImportRecipe, InvalidationReason, JobId, Processor,
    ProcessorError, RevisionStore, SourceSnapshot,
};
use std::collections::BTreeMap;

struct Configured(u8);
impl Processor for Configured {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        let mut output = inputs[0].to_vec();
        output.push(self.0);
        Ok(output)
    }
}
impl ImportProcessor for Configured {
    const IMPORTER_ID: u128 = 617;
    const IMPORTER_VERSION: u32 = 1;
}

fn recipe() -> ImportRecipe {
    ImportRecipe {
        importer_id: Configured::IMPORTER_ID,
        importer_version: Configured::IMPORTER_VERSION,
        canonical_settings: b"configured-v1".to_vec(),
        // The planning API replaces this with the actual recorded projection.
        profile_hash: [99; 32],
        toolchain_fingerprint: b"test-toolchain".to_vec(),
        environment_fingerprint: Vec::new(),
    }
}

fn profile(enabled: bool, quality: u64, unrelated: u64, optional: Option<u64>) -> EffectiveProfile {
    let numeric = || {
        ProfileField::new(
            ProfileFieldType::Unsigned { min: 0, max: 255 },
            ProfileMergePolicy::ScalarOverride,
        )
    };
    let schema = ProfileSchema::new(
        "recipe-profile",
        1,
        [
            (
                "enabled",
                ProfileField::new(ProfileFieldType::Bool, ProfileMergePolicy::ScalarOverride),
            ),
            ("quality", numeric()),
            ("unrelated", numeric()),
            ("optional", numeric()),
        ],
    )
    .unwrap();
    let mut values = ProfileLayerValues::new(ProfileLayer::EngineBaseline);
    values.set("enabled", ProfileValue::Bool(enabled)).unwrap();
    values
        .set("quality", ProfileValue::Unsigned(quality))
        .unwrap();
    values
        .set("unrelated", ProfileValue::Unsigned(unrelated))
        .unwrap();
    if let Some(value) = optional {
        values
            .set("optional", ProfileValue::Unsigned(value))
            .unwrap();
    }
    schema.resolve(&[values]).unwrap()
}

fn configured(
    reader: &mut prism_asset_build_core::profile::ProfileReader<'_>,
) -> Result<Configured, ProfileError> {
    let enabled = matches!(reader.get("enabled")?, Some(ProfileValue::Bool(true)));
    let field = if enabled { "quality" } else { "optional" };
    Ok(Configured(match reader.get(field)? {
        Some(ProfileValue::Unsigned(value)) => *value as u8,
        _ => 0,
    }))
}

fn graph(
    profile: &EffectiveProfile,
) -> (
    BuildGraph,
    BTreeMap<JobId, ArtifactSpec>,
    BuildInputSnapshot,
) {
    let source = SourceSnapshot::new("source.bin", b"source".to_vec()).unwrap();
    let mut graph = BuildGraph::new();
    let mut specs = BTreeMap::new();
    let mut inputs = BuildInputSnapshot::new().with_source(JobId(1), &source);
    specs.insert(JobId(1), source.artifact_spec(AssetId(1)));
    graph.add_job(source.build_job(JobId(1))).unwrap();
    let plan = recipe()
        .plan_with_profile(
            AssetId(2),
            JobId(2),
            JobId(1),
            profile,
            &mut inputs,
            configured,
            42,
            1,
            1,
        )
        .unwrap();
    specs.insert(JobId(2), plan.spec);
    graph.add_job(plan.job).unwrap();
    for (id, dependency) in [(3, 2), (4, 1)] {
        let plan = recipe()
            .plan(
                AssetId(id),
                JobId(id),
                JobId(dependency),
                Configured(17),
                42,
                1,
                1,
            )
            .unwrap();
        specs.insert(JobId(id), plan.spec);
        graph.add_job(plan.job).unwrap();
    }
    (graph, specs, inputs)
}

#[test]
fn recorded_reads_drive_cache_keys_committed_explanations_and_successor_rebuilds() {
    let temp = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(temp.path()).unwrap();
    let mut ddc = DerivedDataCache::open(temp.path().join("ddc")).unwrap();
    let executor = BuildExecutor::default();
    let (first_graph, first_specs, first_inputs) = graph(&profile(true, 2, 1, None));
    executor
        .execute_explained_with_limit(
            &first_graph,
            &first_specs,
            &first_inputs,
            &store,
            &mut ddc,
            BuildRevision(0),
            4,
            false,
        )
        .unwrap();
    let (warm_graph, warm_specs, warm_inputs) = graph(&profile(true, 2, 99, Some(77)));
    let warm = executor
        .execute_explained_with_limit(
            &warm_graph,
            &warm_specs,
            &warm_inputs,
            &store,
            &mut ddc,
            BuildRevision(1),
            4,
            false,
        )
        .unwrap();
    assert_eq!(warm.cache_hits.len(), 4);
    assert!(executor
        .load_committed_build_inputs(temp.path(), BuildRevision(1))
        .unwrap()
        .plan
        .is_empty());

    let (next_graph, next_specs, next_inputs) = graph(&profile(true, 9, 99, Some(77)));
    let next = executor
        .execute_explained_with_limit(
            &next_graph,
            &next_specs,
            &next_inputs,
            &store,
            &mut ddc,
            BuildRevision(2),
            4,
            false,
        )
        .unwrap();
    assert_eq!(
        next.cache_hits.iter().copied().collect::<Vec<_>>(),
        [JobId(1), JobId(4)]
    );
    assert_ne!(warm.artifact_keys[&JobId(2)], next.artifact_keys[&JobId(2)]);
    assert_eq!(
        store
            .cas()
            .read(next.published.manifest.entries[&AssetId(3)].content_hash)
            .unwrap(),
        b"source\x09\x11"
    );
    let committed = executor
        .load_committed_build_inputs(temp.path(), BuildRevision(2))
        .unwrap();
    assert_eq!(
        committed.plan.dirty_set().direct_jobs().collect::<Vec<_>>(),
        [JobId(2)]
    );
    assert_eq!(
        committed.plan.dirty_set().jobs().collect::<Vec<_>>(),
        [JobId(2), JobId(3)]
    );
    assert!(
        matches!(&committed.plan.changes()[0], InvalidationReason::Profile { fields, .. } if fields.len() == 1 && fields[0].field == "quality")
    );
}

#[test]
fn conditional_and_absent_reads_are_dependencies_but_unread_branch_values_are_not() {
    let (before_graph, _, before) = graph(&profile(false, 2, 0, None));
    let (_, _, unrelated) = graph(&profile(false, 9, 0, None));
    assert!(before
        .invalidation_plan(&unrelated, &before_graph)
        .unwrap()
        .is_empty());
    let (_, _, present) = graph(&profile(false, 9, 0, Some(5)));
    let changes = before.invalidation_plan(&present, &before_graph).unwrap();
    assert!(
        matches!(&changes.changes()[0], InvalidationReason::Profile { fields, .. } if fields.len() == 1 && fields[0].field == "optional")
    );
    let (_, _, switched) = graph(&profile(true, 9, 0, None));
    let changes = before.invalidation_plan(&switched, &before_graph).unwrap();
    assert!(
        matches!(&changes.changes()[0], InvalidationReason::Profile { fields, .. } if fields.iter().map(|field| field.field.as_str()).collect::<Vec<_>>() == ["enabled", "optional", "quality"])
    );
}

#[test]
fn failed_planning_never_records_partial_profile_or_contract_inputs() {
    let profile = profile(true, 2, 0, None);
    let mut inputs = BuildInputSnapshot::new().with_settings(JobId(2), b"original".to_vec());
    let before = inputs.clone();
    let result = recipe().plan_with_profile(
        AssetId(2),
        JobId(2),
        JobId(1),
        &profile,
        &mut inputs,
        |reader| {
            reader.get("enabled")?;
            reader.get("unknown")?;
            Ok(Configured(1))
        },
        42,
        1,
        1,
    );
    assert!(result.is_err());
    assert_eq!(inputs, before);
    let mut wrong = recipe();
    wrong.importer_version += 1;
    assert!(wrong
        .plan_with_profile(
            AssetId(2),
            JobId(2),
            JobId(1),
            &profile,
            &mut inputs,
            configured,
            42,
            1,
            1
        )
        .is_err());
    assert_eq!(inputs, before);
}

#[test]
fn contract_changes_have_settings_evidence_independent_from_profile_evidence() {
    let profile = profile(true, 2, 0, None);
    let (graph, _, before) = graph(&profile);
    for change in 0..7 {
        let mut recipe = recipe();
        let mut asset = AssetId(2);
        let (mut kind, mut schema, mut alignment) = (42, 1, 1);
        match change {
            0 => recipe.canonical_settings.push(3),
            1 => recipe.toolchain_fingerprint.push(3),
            2 => recipe.environment_fingerprint.push(3),
            3 => asset = AssetId(77),
            4 => kind = 43,
            5 => schema = 2,
            _ => alignment = 4,
        }
        let mut after = before.clone();
        recipe
            .plan_with_profile(
                asset,
                JobId(2),
                JobId(1),
                &profile,
                &mut after,
                configured,
                kind,
                schema,
                alignment,
            )
            .unwrap();
        let plan = before.invalidation_plan(&after, &graph).unwrap();
        assert_eq!(plan.changes().len(), 1);
        assert!(matches!(
            &plan.changes()[0],
            InvalidationReason::Settings { job: JobId(2), .. }
        ));
    }
}
