use std::mem::size_of;

use prism_render_architecture::{
    abi::{AbiHash, AbiVersion},
    shader_package::{
        ArtifactDigest, ManifestExpectation, ModuleId, PermutationSpace, ShaderPackageId,
        ShaderPackageManifest, ShaderPackageRegistry, ShaderPackageVariantManifest,
    },
};
use prism_render_material::{
    fallback_material_header, BlendDomain, BuiltinProgram, ChannelMask, CompiledEntry,
    ExecutionFamily, GpuMaterialHeader, MaterialExecutionTables, MaterialFeatureFlags, ProgramId,
    ProgramRef, ResponseChannel, ResponseMaskSource, ResponseProgramIndex, ShadingPlan,
};
use prism_render_shading::{
    ResponseBucketTable, ShadingWorkItem, ShadingWorkPlan, VisibilityPixel,
};

pub const STANDARD_WIDTH: u32 = 1_280;
pub const STANDARD_HEIGHT: u32 = 720;
pub const STANDARD_AUXILIARY_RATIO_NUMERATOR: u32 = 1;
pub const STANDARD_AUXILIARY_RATIO_DENOMINATOR: u32 = 8;

// These are the production ABI payload strides used by the resident response
// compositor/classification resources. This is a CPU accounting ledger; it is
// not a claim that a device allocated these bytes during this run.
const GPU_RESPONSE_STRIDE: u64 = 64;
const GPU_COMPOSITION_CONTROL_STRIDE: u64 = 16;
const GPU_AUXILIARY_STRIDE: u64 = 64;
const GPU_WATER_COVERAGE_STRIDE: u64 = 4;
const GPU_PIXEL_CLASS_STRIDE: u64 = 20;
const GPU_DISPATCH_ARGS_STRIDE: u64 = 12;
const GPU_BUCKET_STRIDE: u64 = 16;
const GPU_DIAGNOSTICS_BYTES: u64 = 16;

#[derive(Clone, Debug)]
pub struct ResponseCompositionWorkload {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
    pub materials: Vec<GpuMaterialHeader>,
    pub pixels: Vec<(u32, VisibilityPixel, u32)>,
    pub execution: MaterialExecutionTables,
    pub buckets: ResponseBucketTable,
    pub auxiliary_jobs: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkloadLedger {
    pub workload_pixels: u32,
    pub bucket_count: u32,
    pub auxiliary_jobs: u32,
    pub composition_payload_bytes: u64,
    pub classification_payload_bytes: u64,
    pub known_gpu_payload_bytes: u64,
    pub cpu_input_payload_bytes: u64,
    pub work_item_payload_bytes: u64,
}

impl WorkloadLedger {
    #[must_use]
    pub fn for_workload(workload: &ResponseCompositionWorkload) -> Self {
        let pixels = workload.width as u64 * workload.height as u64;
        let buckets = workload.buckets.len() as u64;
        let auxiliary_jobs = workload.auxiliary_jobs as u64;
        let composition_payload_bytes = pixels
            * (GPU_RESPONSE_STRIDE * 2
                + GPU_COMPOSITION_CONTROL_STRIDE
                + GPU_WATER_COVERAGE_STRIDE)
            + auxiliary_jobs * GPU_AUXILIARY_STRIDE;
        let classification_payload_bytes = pixels
            * (GPU_PIXEL_CLASS_STRIDE + size_of::<ShadingWorkItem>() as u64)
            + buckets
                * (4 * size_of::<u32>() as u64 + GPU_DISPATCH_ARGS_STRIDE + GPU_BUCKET_STRIDE)
            + GPU_DIAGNOSTICS_BYTES;
        let cpu_input_payload_bytes = workload.pixels.capacity()
            * size_of::<(u32, VisibilityPixel, u32)>()
            + workload.materials.capacity() * size_of::<GpuMaterialHeader>();
        let work_item_payload_bytes = pixels * size_of::<ShadingWorkItem>() as u64;
        Self {
            workload_pixels: pixels as u32,
            bucket_count: buckets as u32,
            auxiliary_jobs: workload.auxiliary_jobs,
            composition_payload_bytes,
            classification_payload_bytes,
            known_gpu_payload_bytes: composition_payload_bytes + classification_payload_bytes,
            cpu_input_payload_bytes: cpu_input_payload_bytes as u64,
            work_item_payload_bytes,
        }
    }
}

pub fn standard_workload() -> ResponseCompositionWorkload {
    let (mut execution, mut materials) = execution_fixture();
    let (continuous_header, continuous_index) = register_continuous_plan(&mut execution);
    materials.push(continuous_header);
    let buckets = ResponseBucketTable::from_execution_tables(&execution)
        .expect("fixture execution identities must produce a bucket table");

    let width = STANDARD_WIDTH;
    let height = STANDARD_HEIGHT;
    let pixel_count = width * height;
    let mut pixels = Vec::with_capacity(pixel_count as usize);
    for index in 0..pixel_count {
        let material_index = match index % 20 {
            0..=8 => 0,
            9..=11 => 1,
            12..=13 => 2,
            14..=15 => 3,
            _ => continuous_index,
        };
        let mut pixel = VisibilityPixel {
            scene_index: 1,
            scene_generation: 1,
            primitive_id: index,
            geometry_lod_or_cluster: index % 4,
            material_index,
            material_generation: materials[material_index as usize].generation,
            barycentrics_unorm16: 0,
            coverage_and_flags: 255,
        };
        if material_index == continuous_index {
            pixel.coverage_and_flags = ((index.wrapping_mul(29_791) % 65_536) as u16) as u32;
        }
        pixels.push((index, pixel, u32::MAX));
    }

    ResponseCompositionWorkload {
        name: "response-composition-1280x720-mixed-20pct-continuous",
        width,
        height,
        materials,
        pixels,
        execution,
        buckets,
        auxiliary_jobs: pixel_count / STANDARD_AUXILIARY_RATIO_DENOMINATOR
            * STANDARD_AUXILIARY_RATIO_NUMERATOR,
    }
}

pub fn build_plan(workload: &ResponseCompositionWorkload) -> ShadingWorkPlan {
    ShadingWorkPlan::build_with_region_and_buckets_and_execution(
        workload.pixels.iter().copied(),
        &workload.materials,
        workload.width * workload.height,
        &workload.buckets,
        Some(&workload.execution),
    )
    .expect("fixed response-composition workload must classify")
}

pub fn continuous_item_count(plan: &ShadingWorkPlan) -> usize {
    plan.work
        .iter()
        .filter(|item| item.right_shading_class != u32::MAX)
        .count()
}

pub fn plan_digest(plan: &ShadingWorkPlan) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for value in plan
        .offsets
        .iter()
        .copied()
        .chain(plan.counts.iter().copied())
        .chain(plan.work.iter().flat_map(|item| {
            [
                item.pixel_index,
                item.material_index,
                item.shading_class,
                item.response_kind,
                item.flags,
                item.right_shading_class,
                item.right_response_kind,
                item.response_meta,
            ]
        }))
    {
        digest ^= u64::from(value);
        digest = digest.wrapping_mul(0x1000_0000_01b3);
    }
    digest
}

fn execution_fixture() -> (MaterialExecutionTables, Vec<GpuMaterialHeader>) {
    let manifest = ShaderPackageManifest {
        id: ShaderPackageId::new("response-composition-perf"),
        source_revision: "fixed-workload-v1".into(),
        entry_point: "main".into(),
        abi_version: AbiVersion::from_parts(1, 0),
        abi_hash: AbiHash::from_bytes(b"response-composition-perf"),
        variants: ShaderPackageVariantManifest::from_digests(
            &PermutationSpace::new([2]),
            [ArtifactDigest([1; 32]), ArtifactDigest([2; 32])],
        )
        .unwrap(),
    };
    let binary = vec![
        0x0723_0203,
        0x0001_0000,
        0,
        5,
        0,
        (2 << 16) | 17,
        1,
        (3 << 16) | 14,
        0,
        1,
        (5 << 16) | 15,
        5,
        3,
        0x6e69_616d,
        0,
        (6 << 16) | 16,
        3,
        17,
        1,
        1,
        1,
        (2 << 16) | 19,
        1,
        (3 << 16) | 33,
        2,
        1,
        (5 << 16) | 54,
        1,
        3,
        0,
        2,
        (2 << 16) | 248,
        4,
        (1 << 16) | 253,
        (1 << 16) | 56,
    ];
    let entry = CompiledEntry::from_spirv(manifest.clone(), binary).unwrap();
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(entry.artifact().clone())
        .unwrap();
    packages
        .validate(
            &manifest.id,
            &ManifestExpectation::new(manifest.abi_version, manifest.abi_hash),
        )
        .unwrap();

    let mut tables = MaterialExecutionTables::new();
    let package = tables.register_package(&entry, &packages).unwrap();
    let programs = [
        (
            ProgramRef::Builtin(BuiltinProgram::Lit),
            ExecutionFamily::FusedBuiltin,
        ),
        (
            ProgramRef::Builtin(BuiltinProgram::Npr),
            ExecutionFamily::SplitResponse,
        ),
        (
            ProgramRef::Builtin(BuiltinProgram::Unlit),
            ExecutionFamily::SplitResponse,
        ),
        (
            ProgramRef::Registered(ProgramId(3)),
            ExecutionFamily::SplitResponse,
        ),
    ];
    let mut materials = Vec::with_capacity(programs.len());
    for (program, family) in programs {
        let program_index = tables
            .register_program(
                program,
                &entry,
                package,
                ModuleId(7),
                PermutationSpace::try_new([2]).unwrap(),
            )
            .unwrap();
        let plan = tables
            .register_plan(
                ShadingPlan::Single { program },
                program_index,
                None,
                family,
                ChannelMask::default(),
            )
            .unwrap();
        let words = tables
            .bind_raw(plan.get(), program_index.get(), package.get(), 1)
            .unwrap()
            .to_words();
        let mut header = fallback_material_header(1);
        header.execution_plan_index = words[0];
        header.execution_program_index = words[1];
        header.execution_package_index = words[2];
        header.execution_permutation_low = words[3];
        header.execution_permutation_high = words[4];
        materials.push(header);
    }
    (tables, materials)
}

fn register_continuous_plan(tables: &mut MaterialExecutionTables) -> (GpuMaterialHeader, u32) {
    let left = ResponseProgramIndex::from_raw(0, tables.programs()).unwrap();
    let right = ResponseProgramIndex::from_raw(1, tables.programs()).unwrap();
    let plan = tables
        .register_plan(
            ShadingPlan::MaskBlend {
                left: ProgramRef::Builtin(BuiltinProgram::Lit),
                right: ProgramRef::Builtin(BuiltinProgram::Npr),
                mask: ResponseMaskSource::RegionMask,
                domain: BlendDomain::DirectOnly,
            },
            left,
            Some(right),
            ExecutionFamily::Composition,
            ChannelMask::one(ResponseChannel::Direct),
        )
        .unwrap();
    let words = tables
        .bind_raw(plan.get(), left.get(), 0, 1)
        .unwrap()
        .to_words();
    let mut header = fallback_material_header(1);
    header.execution_plan_index = words[0];
    header.execution_program_index = words[1];
    header.execution_package_index = words[2];
    header.execution_permutation_low = words[3];
    header.execution_permutation_high = words[4];
    header.feature_flags |= MaterialFeatureFlags::ILLUMINATION_BLEND.0;
    (header, 4)
}
