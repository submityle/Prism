use prism_asset_build_core::{
    ImportProcessor, ImportSelector, ImporterPlugin, ImporterRegistry, ImporterRegistryError,
    ImporterSelectionError, Processor, ProcessorError, ProcessorIdentity,
};

#[derive(Clone, Copy)]
struct WebpProcessor;

impl Processor for WebpProcessor {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        Ok(inputs.first().copied().unwrap_or_default().to_vec())
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

impl Processor for AlternatePngProcessor {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        Ok(inputs.first().copied().unwrap_or_default().to_vec())
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

#[test]
fn built_in_registry_normalizes_and_selects_by_extension_or_mime() {
    let registry = ImporterRegistry::with_builtin().unwrap();
    let by_extension = registry.select_for_path("textures/HERO.PnG", None).unwrap();
    assert_eq!(
        by_extension.identity(),
        ProcessorIdentity::new(
            prism_asset_build_core::import::CanonicalPngProcessor::IMPORTER_ID,
            2
        )
    );
    let by_mime = registry
        .select(&ImportSelector::new(None, Some("IMAGE/PNG; charset=utf-8")))
        .unwrap();
    assert_eq!(by_mime.identity(), by_extension.identity());
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
                prism_asset_build_core::import::CanonicalPngProcessor::IMPORTER_ID,
                2
            ),
            ProcessorIdentity::new(0x5745_4250, 1),
        ]
    );

    registry.register(AlternatePngProcessor).unwrap();
    assert!(matches!(
        registry.select_for_path("textures/hero.png", None),
        Err(ImporterSelectionError::Ambiguous { candidates, .. })
            if candidates == vec![ProcessorIdentity::new(0xA11CE, 1), ProcessorIdentity::new(prism_asset_build_core::import::CanonicalPngProcessor::IMPORTER_ID, 2)]
    ));
    assert_eq!(
        registry.register(AlternatePngProcessor),
        Err(ImporterRegistryError::DuplicateProcessorIdentity(
            ProcessorIdentity::new(0xA11CE, 1)
        ))
    );
}
