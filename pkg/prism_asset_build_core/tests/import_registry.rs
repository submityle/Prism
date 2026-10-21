use prism_asset_build_core::processor::{
    prepare_process, AnalyzeContext, ArtifactType, ArtifactWriter, ExecuteContext, InputCatalog,
    ProcessCancellation, ProcessLimits, ProcessPlan, ProcessReport, ProcessRequest,
    Processor as SdkProcessor, ProcessorCategory, ProcessorDescriptor,
};
use prism_asset_build_core::{
    import::CanonicalPngProcessor, ArtifactDescriptor, AssetId, BuildExecutor, BuildGraph,
    BuildResources, BuildRevision, DerivedDataCache, ImportProcessor, ImportSelector,
    ImporterPlugin, ImporterRegistry, ImporterRegistryError, ImporterSelectionError, JobId,
    ProcessorError, ProcessorIdentity, RevisionStore, SourceSnapshot,
};
use std::collections::BTreeMap;
use std::sync::OnceLock;

const COPY_OUTPUT: ArtifactType = ArtifactType {
    kind: 42,
    schema: 1,
};

fn copy_descriptor(identity: ProcessorIdentity) -> ProcessorDescriptor {
    ProcessorDescriptor {
        identity,
        name: "registry copy importer".into(),
        category: ProcessorCategory::Importer,
        input_types: vec![ArtifactType {
            kind: prism_asset_build_core::source::SOURCE_ARTIFACT_KIND,
            schema: prism_asset_build_core::source::SOURCE_ARTIFACT_SCHEMA,
        }],
        output_types: vec![COPY_OUTPUT],
        settings_schema: 1,
        deterministic: true,
        supports_cancellation: true,
    }
}

fn copy_plan() -> ProcessPlan {
    ProcessPlan {
        output: COPY_OUTPUT,
        alignment: 1,
        canonical_settings: vec![],
        resources: BuildResources::default(),
        cacheable: true,
    }
}

fn copy_inputs(
    ctx: &mut ExecuteContext<'_, '_>,
    output: &mut ArtifactWriter<'_>,
) -> Result<ProcessReport, ProcessorError> {
    for name in ctx.input_names().to_vec() {
        let mut input = ctx.input(&name)?;
        std::io::copy(&mut input, output)
            .map_err(|error| ProcessorError::new(error.to_string()))?;
    }
    Ok(ProcessReport {
        output_bytes: output.bytes_written(),
    })
}

#[derive(Clone, Copy)]
struct WebpProcessor;

impl SdkProcessor for WebpProcessor {
    fn descriptor(&self) -> &ProcessorDescriptor {
        static DESCRIPTOR: OnceLock<ProcessorDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| copy_descriptor(Self::identity()))
    }

    fn analyze(
        &self,
        _: &mut AnalyzeContext<'_>,
        _: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessorError> {
        Ok(copy_plan())
    }

    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, '_>,
        _: &ProcessPlan,
        output: &mut ArtifactWriter<'_>,
    ) -> Result<ProcessReport, ProcessorError> {
        copy_inputs(ctx, output)
    }
}

impl prism_asset_build_core::ImportProcessor for WebpProcessor {
    const IMPORTER_ID: u128 = 0x5745_4250;
    const IMPORTER_VERSION: u32 = 1;
}

impl ImporterPlugin for WebpProcessor {
    const EXTENSIONS: &'static [&'static str] = &[".WEBP"];
    const MIME_TYPES: &'static [&'static str] = &["image/webp; charset=binary"];
}

#[derive(Clone, Copy)]
struct AlternatePngProcessor;

impl SdkProcessor for AlternatePngProcessor {
    fn descriptor(&self) -> &ProcessorDescriptor {
        static DESCRIPTOR: OnceLock<ProcessorDescriptor> = OnceLock::new();
        DESCRIPTOR.get_or_init(|| copy_descriptor(Self::identity()))
    }

    fn analyze(
        &self,
        _: &mut AnalyzeContext<'_>,
        _: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessorError> {
        Ok(copy_plan())
    }

    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, '_>,
        _: &ProcessPlan,
        output: &mut ArtifactWriter<'_>,
    ) -> Result<ProcessReport, ProcessorError> {
        copy_inputs(ctx, output)
    }
}

impl prism_asset_build_core::ImportProcessor for AlternatePngProcessor {
    const IMPORTER_ID: u128 = 0xA11CE;
    const IMPORTER_VERSION: u32 = 1;
}

impl ImporterPlugin for AlternatePngProcessor {
    const EXTENSIONS: &'static [&'static str] = &["png"];
    const MIME_TYPES: &'static [&'static str] = &["image/png"];
}

struct InvalidSdkDescriptor {
    descriptor: ProcessorDescriptor,
    panics: bool,
}

impl ImportProcessor for InvalidSdkDescriptor {
    const IMPORTER_ID: u128 = 0xBAD;
    const IMPORTER_VERSION: u32 = 1;
}

impl ImporterPlugin for InvalidSdkDescriptor {
    const EXTENSIONS: &'static [&'static str] = &["invalid"];
    const MIME_TYPES: &'static [&'static str] = &[];
}

impl SdkProcessor for InvalidSdkDescriptor {
    fn descriptor(&self) -> &ProcessorDescriptor {
        assert!(!self.panics, "bad SDK descriptor");
        &self.descriptor
    }

    fn analyze(
        &self,
        _: &mut AnalyzeContext<'_>,
        _: &ProcessRequest,
    ) -> Result<ProcessPlan, ProcessorError> {
        Ok(copy_plan())
    }

    fn execute(
        &self,
        ctx: &mut ExecuteContext<'_, '_>,
        _: &ProcessPlan,
        output: &mut ArtifactWriter<'_>,
    ) -> Result<ProcessReport, ProcessorError> {
        copy_inputs(ctx, output)
    }
}

#[test]
fn built_in_registry_normalizes_and_selects_by_extension_or_mime() {
    let registry = ImporterRegistry::with_builtin().unwrap();
    let by_extension = registry.select_for_path("textures/HERO.PnG", None).unwrap();
    assert_eq!(
        by_extension.identity(),
        ProcessorIdentity::new(
            CanonicalPngProcessor::IMPORTER_ID,
            CanonicalPngProcessor::IMPORTER_VERSION
        )
    );
    let by_mime = registry
        .select(&ImportSelector::new(None, Some("IMAGE/PNG; charset=utf-8")))
        .unwrap();
    assert_eq!(by_mime.identity(), by_extension.identity());
    assert_eq!(
        by_extension.processor().descriptor().identity,
        by_extension.descriptor().identity()
    );
}

#[test]
fn selection_has_explicit_missing_and_no_match_errors() {
    let registry = ImporterRegistry::with_builtin().unwrap();
    assert!(matches!(
        registry.select(&ImportSelector::new(None, None)),
        Err(ImporterSelectionError::MissingSelector)
    ));
    assert!(matches!(
        registry.select_for_path("textures/hero.tga", None),
        Err(ImporterSelectionError::NoMatch { .. })
    ));
}

#[test]
fn registration_and_selection_are_deterministic_and_ambiguity_is_explicit() {
    let mut registry = ImporterRegistry::with_builtin().unwrap();
    registry.register(WebpProcessor).unwrap();
    let identities = registry
        .descriptors()
        .map(|descriptor| descriptor.identity())
        .collect::<Vec<_>>();
    assert_eq!(
        identities,
        vec![
            ProcessorIdentity::new(
                CanonicalPngProcessor::IMPORTER_ID,
                CanonicalPngProcessor::IMPORTER_VERSION
            ),
            ProcessorIdentity::new(0x5745_4250, 1),
        ]
    );

    registry.register(AlternatePngProcessor).unwrap();
    assert!(matches!(
        registry.select_for_path("textures/hero.png", None),
        Err(ImporterSelectionError::Ambiguous { candidates, .. })
            if candidates == vec![ProcessorIdentity::new(0xA11CE, 1), CanonicalPngProcessor::identity()]
    ));
    assert_eq!(
        registry.register(AlternatePngProcessor),
        Err(ImporterRegistryError::DuplicateProcessorIdentity(
            ProcessorIdentity::new(0xA11CE, 1)
        ))
    );
}

#[test]
fn sdk_identity_id_and_version_mismatches_reject_registration() {
    let expected = InvalidSdkDescriptor::identity();
    for actual in [
        ProcessorIdentity::new(expected.id + 1, expected.version),
        ProcessorIdentity::new(expected.id, expected.version + 1),
    ] {
        let mut registry = ImporterRegistry::new();
        let error = registry
            .register(InvalidSdkDescriptor {
                descriptor: copy_descriptor(actual),
                panics: false,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            ImporterRegistryError::InvalidDescriptor { identity, reason }
                if identity == expected && reason.contains("differs from static importer identity")
        ));
        assert_eq!(registry.descriptors().count(), 0);
        assert!(matches!(
            registry.select_for_path("source.invalid", None),
            Err(ImporterSelectionError::NoMatch { .. })
        ));
        registry.register(WebpProcessor).unwrap();
    }
}

#[test]
fn sdk_descriptor_panic_is_isolated_and_leaves_registry_unchanged() {
    let mut registry = ImporterRegistry::new();
    registry.register(WebpProcessor).unwrap();
    let before = registry.descriptors().cloned().collect::<Vec<_>>();
    let error = registry
        .register(InvalidSdkDescriptor {
            descriptor: copy_descriptor(InvalidSdkDescriptor::identity()),
            panics: true,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        ImporterRegistryError::InvalidDescriptor { identity, reason }
            if identity == InvalidSdkDescriptor::identity() && reason.contains("panicked")
    ));
    assert_eq!(registry.descriptors().cloned().collect::<Vec<_>>(), before);
    assert_eq!(
        registry
            .select_for_path("source.webp", None)
            .unwrap()
            .identity(),
        WebpProcessor::identity()
    );
}

#[test]
fn selected_handle_runs_recorded_sdk_analysis_and_copy_execution() {
    let mut registry = ImporterRegistry::new();
    registry.register(WebpProcessor).unwrap();
    let handle = registry.select_for_path("source.webp", None).unwrap();
    let processor = handle.processor();
    assert!(std::sync::Arc::ptr_eq(
        &processor,
        &handle.clone().processor()
    ));

    let root = tempfile::tempdir().unwrap();
    let store = RevisionStore::open(root.path().join("store")).unwrap();
    let mut ddc = DerivedDataCache::open(root.path().join("ddc")).unwrap();
    let source = SourceSnapshot::new("source.webp", b"registry SDK copy".to_vec()).unwrap();
    let source_job = JobId(1);
    let source_spec = source.artifact_spec(AssetId(1));
    let source_key = source_spec.artifact_key(1, &[]).unwrap();
    store.cas().put(source.bytes()).unwrap();
    let mut catalog = InputCatalog::new(&store).unwrap();
    catalog
        .insert(
            "source.webp",
            source_job,
            ArtifactDescriptor::for_bytes(
                source_key,
                source_spec.kind,
                source_spec.schema,
                source_spec.alignment,
                vec![],
                source.bytes(),
            ),
        )
        .unwrap();
    let prepared = prepare_process(
        processor,
        &catalog,
        ProcessRequest {
            job: JobId(2),
            asset: AssetId(2),
            primary_inputs: vec!["source.webp".into()],
            canonical_settings: vec![],
            profile_hash: [0; 32],
            toolchain_fingerprint: vec![],
            environment_fingerprint: vec![],
        },
        ProcessLimits::default(),
        ProcessCancellation::default(),
    )
    .unwrap();
    assert_eq!(prepared.job.dependencies(), &[source_job]);
    assert_eq!(prepared.spec.processor_id, handle.identity().id);
    assert_eq!(prepared.spec.processor_version, handle.identity().version);
    drop(catalog);

    let mut graph = BuildGraph::new();
    graph.add_job(source.build_job(source_job)).unwrap();
    graph.add_job(prepared.job).unwrap();
    let specs = BTreeMap::from([(source_job, source_spec), (JobId(2), prepared.spec)]);
    let result = BuildExecutor {
        protocol_version: 1,
    }
    .execute(&graph, &specs, &store, &mut ddc, BuildRevision(0))
    .unwrap();
    let output = &result.published.manifest.entries[&AssetId(2)];
    assert_eq!(
        store.cas().read(output.content_hash).unwrap(),
        source.bytes()
    );
    assert_eq!(output.dependencies, [source_key]);
}
