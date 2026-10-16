//! The device-lifetime owner. Frame consumers cannot replace its selection.
use super::*;
use crate::backend::capability::CapabilitySet;

#[derive(Debug)]
pub struct DeviceFabric {
    catalog: FeatureCatalog,
    selection: FeatureSelection,
}

impl DeviceFabric {
    pub fn new(
        catalog: FeatureCatalog,
        roots: FrozenRoots,
        caps: CapabilitySet,
        quality: RenderQualityTier,
    ) -> Result<Self, SelectionError> {
        let selection = FeatureSelection::resolve(&catalog, &roots, caps, quality)?;
        Ok(Self { catalog, selection })
    }

    pub fn selection(&self) -> &FeatureSelection {
        &self.selection
    }

    /// Admission is a root check, not a dependency-closure check.
    pub fn admit(&self, feature: FeatureId) -> Result<&SelectedFeature, RootError> {
        self.selection.roots().require_root(feature)?;
        Ok(self.selection.feature(feature).expect("resolved root"))
    }

    /// Graph consumers use exactly the selection used by pipeline/queue admission.
    /// Resource templates are copied during compilation, so per-frame/view layouts
    /// need not outlive the call or change the device's frozen selection.
    pub fn compile(
        &self,
        templates: &[ResourceTemplate],
        frame: u64,
        bindings: &[FrameResourceBinding],
        items: &[FrameRenderItem],
        limits: GraphLimits,
    ) -> Result<CompiledFeatureGraph, FabricCompileError> {
        ResourceTemplates::new(&self.catalog, templates)?.compile(
            &self.selection,
            frame,
            bindings,
            items,
            limits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::ResourceLifetime;

    const FRAME_FEATURE: FeatureId = FeatureId(800);
    const FRAME_VARIANT: VariantId = VariantId(800);
    const FRAME_PASS: PassKey = PassKey(800);
    const FRAME_BUFFER: ResourceKey = ResourceKey::of(FRAME_FEATURE, 0);
    const FRAME_TEXTURE: ResourceKey = ResourceKey::of(FRAME_FEATURE, 1);

    // Only semantic declarations are static; concrete view layouts below are local.
    static FRAME_ACCESSES: [ResourceUse; 2] = [
        ResourceUse {
            input: ResourceInput::Automatic,
            key: FRAME_BUFFER,
            access: AccessKind::StorageWrite,
            scope: ResourceScope::Whole,
            frame_lag: None,
        },
        ResourceUse {
            input: ResourceInput::Automatic,
            key: FRAME_TEXTURE,
            access: AccessKind::StorageWrite,
            scope: ResourceScope::Whole,
            frame_lag: None,
        },
    ];
    static FRAME_PASSES: [PassTemplate; 1] = [PassTemplate {
        depends_on: &[],
        id: FRAME_PASS,
        queue: QueueClass::Compute,
        accesses: &FRAME_ACCESSES,
    }];
    static FRAME_VARIANTS: [FeatureVariant; 1] = [FeatureVariant {
        id: FRAME_VARIANT,
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes: &FRAME_PASSES,
        shader_permutation: None,
    }];
    static FRAME_DEFINITIONS: [FeatureDefinition; 1] = [FeatureDefinition::subsystem(
        FRAME_FEATURE,
        RenderDomain::Surface,
        ServiceSet::EMPTY,
        &FRAME_VARIANTS,
    )];
    static FRAME_RESOURCES: [ResourceKey; 2] = [FRAME_BUFFER, FRAME_TEXTURE];
    static FRAME_ROOTS: [FeatureId; 1] = [FRAME_FEATURE];

    fn frame_device() -> DeviceFabric {
        let catalog = FeatureCatalog::new(&FRAME_DEFINITIONS, &FRAME_RESOURCES).unwrap();
        let roots = FrozenRoots::from_build_profile(&catalog, &FRAME_ROOTS, &[]).unwrap();
        DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap()
    }

    fn frame_templates(extent: [u32; 2], bytes: u64) -> [ResourceTemplate; 2] {
        [
            ResourceTemplate {
                key: FRAME_BUFFER,
                lifetime: ResourceLifetime::Transient,
                layout: ResourceLayout::Buffer {
                    bytes,
                    usage: BufferUsages::STORAGE,
                },
                alignment: 16,
            },
            ResourceTemplate {
                key: FRAME_TEXTURE,
                lifetime: ResourceLifetime::Transient,
                layout: ResourceLayout::Texture {
                    extent: [extent[0], extent[1], 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::STORAGE_BINDING,
                },
                alignment: 16,
            },
        ]
    }

    fn assert_frame_layouts(graph: &CompiledFeatureGraph, expected: [ResourceTemplate; 2]) {
        assert_eq!(graph.resources().len(), expected.len());
        for template in expected {
            let id = graph.resource_id(template.key, graph.frame()).unwrap();
            let resource = &graph.resources()[id.0 as usize];
            assert_eq!(resource.template, template);
            assert_eq!(resource.descriptor.size, template.layout.size().unwrap());
            assert_eq!(resource.descriptor.alignment, template.alignment);
        }
        assert_eq!(graph.passes().len(), 1);
        assert_eq!(graph.passes()[0].key(), FRAME_PASS);
        assert_eq!(graph.passes()[0].feature(), FRAME_FEATURE);
        assert_eq!(graph.passes()[0].variant(), FRAME_VARIANT);
    }

    #[test]
    fn compile_owns_frame_local_vec_templates_across_resize() {
        let device = frame_device();
        let selection = device.selection().clone();
        let (original, resized) = {
            let mut templates = frame_templates([320, 180], 4096).to_vec();
            let original = device
                .compile(&templates, 7, &[], &[], GraphLimits::default())
                .unwrap();
            assert_eq!(device.selection(), &selection);

            // Reuse the allocation for the next view size and buffer capacity.
            templates.copy_from_slice(&frame_templates([640, 360], 16384));
            let resized = device
                .compile(&templates, 8, &[], &[], GraphLimits::default())
                .unwrap();
            (original, resized)
        }; // The Vec is dropped before either compiled result is inspected.

        assert_eq!(original.frame(), 7);
        assert_eq!(resized.frame(), 8);
        assert_frame_layouts(&original, frame_templates([320, 180], 4096));
        assert_frame_layouts(&resized, frame_templates([640, 360], 16384));
        assert_eq!(device.selection(), &selection);
    }

    #[test]
    fn compile_owns_frame_local_stack_templates() {
        let device = frame_device();
        let selection = device.selection().clone();
        let compiled = {
            let templates = frame_templates([800, 450], 8192);
            device
                .compile(&templates, 9, &[], &[], GraphLimits::default())
                .unwrap()
        }; // The stack array is out of scope; no leaked or static layouts.

        assert_eq!(compiled.frame(), 9);
        assert_frame_layouts(&compiled, frame_templates([800, 450], 8192));
        assert_eq!(device.selection(), &selection);
    }

    #[test]
    fn frame_local_templates_preserve_layout_and_access_validation() {
        let device = frame_device();
        let selection = device.selection().clone();
        let zero_buffer = frame_templates([320, 180], 0);
        let zero_texture = frame_templates([0, 180], 4096);
        let mut bad_alignment = frame_templates([320, 180], 4096);
        bad_alignment[0].alignment = 3;
        let mut bad_usage = frame_templates([320, 180], 4096);
        bad_usage[0].layout = ResourceLayout::Buffer {
            bytes: 4096,
            usage: BufferUsages::COPY_DST,
        };
        for (templates, expected) in [
            (zero_buffer, FabricCompileError::InvalidLayout(FRAME_BUFFER)),
            (
                zero_texture,
                FabricCompileError::InvalidLayout(FRAME_TEXTURE),
            ),
            (
                bad_alignment,
                FabricCompileError::InvalidLayout(FRAME_BUFFER),
            ),
            (
                bad_usage,
                FabricCompileError::InvalidAccess {
                    pass: FRAME_PASS,
                    resource: FRAME_BUFFER,
                    access: AccessKind::StorageWrite,
                },
            ),
        ] {
            assert_eq!(
                device
                    .compile(&templates, 10, &[], &[], GraphLimits::default())
                    .unwrap_err(),
                expected
            );
            assert_eq!(device.selection(), &selection);
        }

        let compiled = device
            .compile(
                &frame_templates([320, 180], 4096),
                10,
                &[],
                &[],
                GraphLimits::default(),
            )
            .unwrap();
        assert_frame_layouts(&compiled, frame_templates([320, 180], 4096));
        assert_eq!(device.selection(), &selection);
    }

    #[test]
    fn device_admission_does_not_expand_roots_or_reselect() {
        let catalog = builtin_catalog().unwrap();
        let roots =
            FrozenRoots::from_build_profile(&catalog, &[WATER_FEATURE_ID], &[LIT_FEATURE_ID])
                .unwrap();
        let device = DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap();
        let snapshot = device.selection().clone();
        assert!(device.admit(WATER_FEATURE_ID).is_ok());
        assert_eq!(
            device.admit(PARTICLE_FEATURE_ID).unwrap_err(),
            RootError::OutsideFrozenRoots(PARTICLE_FEATURE_ID)
        );
        assert_eq!(
            device.admit(NPR_FEATURE_ID).unwrap_err(),
            RootError::OutsideFrozenRoots(NPR_FEATURE_ID)
        );
        // Selected service != root permission.
        assert!(device
            .selection()
            .feature(SPRAY_FEEDBACK_FEATURE_ID)
            .is_some());
        assert!(device.admit(SPRAY_FEEDBACK_FEATURE_ID).is_err());
        assert_eq!(device.selection(), &snapshot);
    }
}
