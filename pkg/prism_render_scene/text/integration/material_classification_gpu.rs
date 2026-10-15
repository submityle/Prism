//! Bounded real-device regression for the production classify -> prefix -> scatter.
//! No shader translation, CPU dispatch emulation, or runtime pipeline is substituted.

use bevy_platform::future::block_on;
use prism_render_architecture::{
    abi::{AbiHash, AbiVersion},
    shader_package::{
        ArtifactDigest, ManifestExpectation, ModuleId, PermutationSpace, ShaderPackageId,
        ShaderPackageManifest, ShaderPackageRegistry, ShaderPackageVariantManifest,
    },
};
use prism_render_material::{
    fallback_material_header, BlendDomain, BuiltinProgram, ChannelMask, ClosureKind, CompiledEntry,
    ExecutionFamily, GpuMaterialHeader, LobeMask, MaterialExecutionTables, MaterialFeatureFlags,
    ProgramId, ProgramRef, ResponseMaskSource, ShadingPlan,
};
use prism_render_shading::{
    ClassificationError, ResponseBucketTable, ShadingWorkPlan, ShadingWorkPlanError,
    VisibilityPixel,
};
use wgpu::util::DeviceExt;

const INVALID: u32 = u32::MAX;
const PIXEL_CLASS_WORDS: usize = 5;
const SHADING_WORK_ITEM_WORDS: usize = 8;
const WIDTH: u32 = 17;
const HEIGHT: u32 = 9;
const PIXELS: u32 = WIDTH * HEIGHT;
const PARAM_BYTES: u32 = 6 * 4;
const STORAGE_BINDINGS: u32 = 12;

fn unavailable(reason: impl std::fmt::Display) -> Option<(wgpu::Device, wgpu::Queue)> {
    assert!(
        std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU set: material classification requires a GPU: {reason}"
    );
    eprintln!("material classification skipped: {reason}");
    None
}

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => return unavailable(error),
    };
    eprintln!("material classification adapter: {:?}", adapter.get_info());
    let limits = adapter.limits();
    if !adapter.features().contains(wgpu::Features::IMMEDIATES)
        || limits.max_immediate_size < PARAM_BYTES
        || limits.max_storage_buffers_per_shader_stage < STORAGE_BINDINGS
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
    {
        return unavailable(format!(
            "missing immediates/compute/storage limits: {limits:?}"
        ));
    }
    match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("material classification parity"),
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: PARAM_BYTES,
            max_storage_buffers_per_shader_stage: STORAGE_BINDINGS,
            ..Default::default()
        },
        ..Default::default()
    })) {
        Ok(device_queue) => Some(device_queue),
        Err(error) => unavailable(error),
    }
}

struct Fixture {
    materials: Vec<GpuMaterialHeader>,
    execution: MaterialExecutionTables,
    buckets: ResponseBucketTable,
    visibility: Vec<(u32, VisibilityPixel, u32)>,
    // Each rejected pixel's exact production diagnostic flag.
    rejected: Vec<(usize, u32)>,
}

fn fixture() -> Fixture {
    // This verified no-op package supplies real checked execution-table indices;
    // the package is not the shader being executed by this classification test.
    let manifest = ShaderPackageManifest {
        id: ShaderPackageId::new("classification-gpu-parity"),
        source_revision: "fixture-1".into(),
        entry_point: "main".into(),
        abi_version: AbiVersion::from_parts(1, 0),
        abi_hash: AbiHash::from_bytes(b"classification-gpu-parity"),
        variants: ShaderPackageVariantManifest::from_digests(
            &PermutationSpace::new([2]),
            [ArtifactDigest([1; 32]), ArtifactDigest([2; 32])],
        )
        .unwrap(),
    };
    let compiled = CompiledEntry::from_spirv(
        manifest.clone(),
        vec![
            0x0723_0203,
            0x0001_0000,
            0,
            5,
            0,
            (2 << 16) | 17,
            1, // OpCapability Shader
            (3 << 16) | 14,
            0,
            1, // OpMemoryModel Logical GLSL450
            (5 << 16) | 15,
            5,
            3,
            0x6e69_616d,
            0, // OpEntryPoint GLCompute %3 main
            (6 << 16) | 16,
            3,
            17,
            1,
            1,
            1, // LocalSize 1 1 1
            (2 << 16) | 19,
            1, // OpTypeVoid
            (3 << 16) | 33,
            2,
            1, // OpTypeFunction
            (5 << 16) | 54,
            1,
            3,
            0,
            2, // OpFunction
            (2 << 16) | 248,
            4,               // OpLabel
            (1 << 16) | 253, // OpReturn
            (1 << 16) | 56,  // OpFunctionEnd
        ],
    )
    .expect("verified fixture package");
    let mut packages = ShaderPackageRegistry::new();
    packages
        .register_verified(compiled.artifact().clone())
        .unwrap();
    packages
        .validate(
            &manifest.id,
            &ManifestExpectation::new(manifest.abi_version, manifest.abi_hash),
        )
        .unwrap();
    let mut execution = MaterialExecutionTables::new();
    let package = execution.register_package(&compiled, &packages).unwrap();
    let mut other_manifest = manifest.clone();
    other_manifest.id = ShaderPackageId::new("classification-gpu-foreign-owner");
    let other_compiled =
        CompiledEntry::from_spirv(other_manifest.clone(), compiled.spirv().to_vec())
            .expect("verified foreign-owner fixture package");
    packages
        .register_verified(other_compiled.artifact().clone())
        .unwrap();
    packages
        .validate(
            &other_manifest.id,
            &ManifestExpectation::new(other_manifest.abi_version, other_manifest.abi_hash),
        )
        .unwrap();
    let other_package = execution
        .register_package(&other_compiled, &packages)
        .unwrap();
    let base = fallback_material_header(7);
    let mut mixed = base;
    mixed.closure_mask |= (1 << ClosureKind::ClearCoat as u32)
        | (1 << ClosureKind::Sheen as u32)
        | (1 << ClosureKind::Subsurface as u32);
    mixed.lobe_mask = LobeMask::all().bits();
    mixed.feature_flags = 1; // Payload must survive both classification and scatter.
    let mut materials = vec![base, mixed];
    let mut programs = Vec::new();
    let refs = [
        ProgramRef::Builtin(BuiltinProgram::Lit),
        ProgramRef::Builtin(BuiltinProgram::Npr),
        ProgramRef::Builtin(BuiltinProgram::Unlit),
    ]
    .into_iter()
    // Deliberately register out of key order across small and large ids. The
    // builtin identities 0..=2 are excluded here because a same-family
    // collision is intentionally rejected as an ambiguous response bucket;
    // that negative contract is covered by the classification unit tests.
    .chain(
        [65535, 4096, 255, 64, 21, 16, 11, 8, 7, 6, 5, 4, 3]
            .into_iter()
            .map(|id| ProgramRef::Registered(ProgramId(id))),
    );
    for (i, program) in refs.enumerate() {
        let index = execution
            .register_program(
                program,
                &compiled,
                package,
                ModuleId(i as u64),
                PermutationSpace::try_new([2]).unwrap(),
            )
            .unwrap();
        let family = match program {
            // Only Lit advertises the fused builtin execution contract. NPR and
            // Unlit are builtin programs too, but their manifests intentionally
            // require the split response path.
            ProgramRef::Builtin(BuiltinProgram::Lit) => ExecutionFamily::FusedBuiltin,
            ProgramRef::Builtin(_) | ProgramRef::Registered(_) => ExecutionFamily::SplitResponse,
        };
        let plan = execution
            .register_plan(
                ShadingPlan::Single { program },
                index,
                None,
                family,
                ChannelMask::default(),
            )
            .unwrap();
        let words = execution
            .bind_raw(plan.get(), index.get(), package.get(), 1)
            .unwrap()
            .to_words();
        materials.push(bound_header(base, words));
        programs.push(index);
    }
    // Exercise family 2 and both validated plan arms as well as families 0/1.
    let plan = execution
        .register_plan(
            ShadingPlan::MaskBlend {
                left: ProgramRef::Builtin(BuiltinProgram::Lit),
                right: ProgramRef::Builtin(BuiltinProgram::Npr),
                mask: ResponseMaskSource::RegionMask,
                domain: BlendDomain::FullRadiance,
            },
            programs[0],
            Some(programs[1]),
            ExecutionFamily::Composition,
            ChannelMask::default(),
        )
        .unwrap();
    for program in &programs[..2] {
        let mut header = bound_header(
            mixed,
            execution
                .bind_raw(plan.get(), program.get(), package.get(), 0)
                .unwrap()
                .to_words(),
        );
        header.feature_flags |= MaterialFeatureFlags::ILLUMINATION_BLEND.0;
        materials.push(header);
    }
    let good_materials = materials.len();
    let mut visibility: Vec<_> = (0..PIXELS)
        .map(|pixel| {
            let material = pixel as usize % good_materials;
            let is_continuous =
                materials[material].feature_flags & MaterialFeatureFlags::ILLUMINATION_BLEND.0 != 0;
            let region = if is_continuous {
                [0, 32_768, u16::MAX as u32][pixel as usize / good_materials % 3]
            } else {
                [INVALID, 0, 1][pixel as usize / good_materials % 3]
            };
            (
                pixel,
                VisibilityPixel {
                    scene_index: pixel,
                    scene_generation: 1,
                    primitive_id: 0,
                    geometry_lod_or_cluster: 0,
                    material_index: material as u32,
                    material_generation: materials[material].generation,
                    barycentrics_unorm16: 0,
                    coverage_and_flags: if is_continuous { region } else { 255 },
                },
                region,
            )
        })
        .collect();
    let mut rejected = Vec::new();
    // Tail pixels reject background, stale/inactive slots, malformed ABI8
    // bindings, foreign plan/program ownership, and permutation overflow.
    let mut tail = visibility.len();
    let mut reject = |header: Option<GpuMaterialHeader>, flag: u32| {
        tail -= 1;
        let pixel = &mut visibility[tail].1;
        if let Some(header) = header {
            pixel.material_index = materials.len() as u32;
            pixel.material_generation = header.generation;
            materials.push(header);
        }
        rejected.push((tail, flag));
        tail
    };
    let background = reject(None, 1);
    let stale = reject(None, 4);
    let out_of_range = reject(None, 2);
    let mut bad = base;
    bad.is_active = 0;
    reject(Some(bad), 4);
    bad = base;
    bad.execution_plan_index = 0; // Only one of the five words bound.
    reject(Some(bad), 8);
    for field in [0, 1, 2, 3, 4] {
        let mut words = [0, 0, 0, 1, 0];
        words[field] = if field == 3 { 2 } else { 1000 };
        reject(Some(bound_header(base, words)), 8);
    }
    // In-range but not a member of plan 0.
    reject(Some(bound_header(base, [0, 1, 0, 0, 0])), 8);
    // All indices are in range, but program 0 belongs to package 0.
    reject(
        Some(bound_header(base, [0, 0, other_package.get(), 0, 0])),
        8,
    );
    bad = base;
    bad.render_domain = 1; // Non-surface domain.
    reject(Some(bad), 8);
    drop(reject);
    visibility[background].1.scene_index = INVALID;
    visibility[stale].1.material_generation += 1;
    visibility[out_of_range].1.material_index = materials.len() as u32 + 1;
    let buckets = ResponseBucketTable::from_execution_tables(&execution)
        .expect("unambiguous verified response bucket keys");
    assert!(buckets.len() > 9);
    assert!(buckets
        .buckets()
        .windows(2)
        .all(|rows| rows[0].key() < rows[1].key()));
    Fixture {
        materials,
        execution,
        buckets,
        visibility,
        rejected,
    }
}

fn bound_header(mut header: GpuMaterialHeader, words: [u32; 5]) -> GpuMaterialHeader {
    [
        header.execution_plan_index,
        header.execution_program_index,
        header.execution_package_index,
        header.execution_permutation_low,
        header.execution_permutation_high,
    ] = words;
    header
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layouts: [wgpu::BindGroupLayout; 3],
    pipelines: [wgpu::ComputePipeline; 3],
}

impl Harness {
    fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let texture_entries: Vec<_> = (0..2)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            })
            .collect();
        let material_entries: Vec<_> = [0, 10, 11, 12]
            .into_iter()
            .map(|binding| storage_entry(binding, true))
            .collect();
        let output_entries: Vec<_> = (0..8)
            .map(|binding| storage_entry(binding, binding == 7))
            .collect();
        let layouts = [texture_entries, material_entries, output_entries].map(|entries| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("classification parity layout"),
                entries: &entries,
            })
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("classification parity"),
            bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1]), Some(&layouts[2])],
            immediate_size: PARAM_BYTES,
        });
        // No imports: this production WESL file is WGSL with immediates. Strip
        // WESL module docs only; preserve every declaration and entry point.
        let source = include_str!("../../src/shaders/material_classification.wesl")
            .lines()
            .filter(|line| !line.trim_start().starts_with("//!"))
            .collect::<Vec<_>>()
            .join("\n");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production material_classification.wesl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = ["classify_count", "prefix_classes", "scatter_work"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self {
            device,
            queue,
            layouts,
            pipelines,
        }
    }

    fn run(&self, fixture: &Fixture, work_capacity: u32, physical_capacity: u32) -> Vec<Vec<u32>> {
        let Self { device, queue, .. } = self;
        let mut views = Vec::new();
        for metadata in [false, true] {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("classification parity visibility"),
                size: wgpu::Extent3d {
                    width: WIDTH,
                    height: HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let data: Vec<[u32; 4]> = fixture
                .visibility
                .iter()
                .map(|(_, pixel, region)| {
                    if metadata {
                        [pixel.material_index, pixel.material_generation, *region, 0]
                    } else {
                        [
                            pixel.scene_index,
                            pixel.scene_generation,
                            pixel.primitive_id,
                            pixel.geometry_lod_or_cluster,
                        ]
                    }
                })
                .collect();
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(WIDTH * 16),
                    rows_per_image: Some(HEIGHT),
                },
                texture.size(),
            );
            views.push(texture.create_view(&Default::default()));
        }
        let visibility_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("classification parity visibility"),
            layout: &self.layouts[0],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
            ],
        });
        let materials = [
            storage(device, bytemuck::cast_slice(&fixture.materials)),
            storage(
                device,
                bytemuck::cast_slice(&fixture.execution.gpu_plan_rows()),
            ),
            storage(
                device,
                bytemuck::cast_slice(&fixture.execution.gpu_program_rows()),
            ),
            storage(
                device,
                &vec![0; fixture.execution.gpu_package_count() as usize],
            ),
        ];
        let entries: Vec<_> = [0, 10, 11, 12]
            .into_iter()
            .zip(&materials)
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let material_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("classification parity execution"),
            layout: &self.layouts[1],
            entries: &entries,
        });
        let bucket_words: Vec<u32> = fixture
            .buckets
            .buckets()
            .iter()
            .flat_map(|bucket| {
                [
                    bucket.execution_family as u32,
                    bucket.program_id.0,
                    bucket.render_phase as u32,
                    0,
                ]
            })
            .collect();
        let classes = fixture.buckets.len();
        let output = [
            storage(device, &vec![INVALID; PIXELS as usize * PIXEL_CLASS_WORDS]),
            storage(
                device,
                &vec![INVALID; physical_capacity.max(1) as usize * SHADING_WORK_ITEM_WORDS],
            ),
            storage(device, &vec![0; classes]),
            storage(device, &vec![INVALID; classes]),
            storage(device, &vec![INVALID; classes]),
            storage(device, &vec![INVALID; classes * 3]),
            storage(device, &[0; 4]),
            storage(device, &bucket_words),
        ];
        let entries: Vec<_> = output
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let output_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("classification parity output"),
            layout: &self.layouts[2],
            entries: &entries,
        });
        let params = [
            WIDTH,
            HEIGHT,
            PIXELS,
            fixture.materials.len() as u32,
            work_capacity,
            classes as u32,
        ];
        let mut encoder = device.create_command_encoder(&Default::default());
        for (pipeline, groups) in
            self.pipelines
                .iter()
                .zip([PIXELS.div_ceil(64), 1, PIXELS.div_ceil(64)])
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &visibility_group, &[]);
            pass.set_bind_group(1, &material_group, &[]);
            pass.set_bind_group(2, &output_group, &[]);
            pass.set_immediates(0, bytemuck::cast_slice(&params));
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let readbacks: Vec<_> = output
            .iter()
            .map(|buffer| {
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("classification parity readback"),
                    size: buffer.size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
                readback
            })
            .collect();
        queue.submit([encoder.finish()]);
        readbacks
            .iter()
            .map(|buffer| {
                buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
                    result.expect("map classification readback")
                });
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("complete classification commands");
                let mapped = buffer
                    .slice(..)
                    .get_mapped_range()
                    .expect("mapped classification data");
                let values = bytemuck::cast_slice(&mapped).to_vec();
                drop(mapped);
                buffer.unmap();
                values
            })
            .collect()
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("classification parity storage"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn assert_parity(fixture: &Fixture, cpu: &ShadingWorkPlan, result: &[Vec<u32>], capacity: u32) {
    let [pixels, work, counts, offsets, cursors, dispatch, diagnostics, buckets] = result else {
        panic!("eight output planes")
    };
    assert_eq!(counts, &cpu.counts, "classify counts");
    assert_eq!(offsets, &cpu.offsets, "prefix offsets");
    assert_eq!(cursors, &cpu.counts, "scatter cursors include overflow");
    let expected_dispatch: Vec<_> = cpu
        .counts
        .iter()
        .flat_map(|count| [count.div_ceil(64), 1, 1])
        .collect();
    assert_eq!(dispatch, &expected_dispatch, "indirect dispatch");
    let expected_buckets: Vec<_> = cpu
        .buckets
        .buckets()
        .iter()
        .flat_map(|bucket| {
            [
                bucket.execution_family as u32,
                bucket.program_id.0,
                bucket.render_phase as u32,
                0,
            ]
        })
        .collect();
    assert_eq!(buckets, &expected_buckets, "uploaded CPU bucket keys");
    for item in cpu.work.iter().filter(|item| item.response_meta != INVALID) {
        assert_ne!(item.right_shading_class, INVALID);
        assert_eq!(
            item.right_response_kind, 1,
            "continuous blend right arm is NPR"
        );
        assert_eq!(
            item.response_meta & u16::MAX as u32,
            fixture.visibility[item.pixel_index as usize]
                .1
                .coverage_and_flags,
            "packed continuous mask matches visibility metadata"
        );
    }
    let mut expected_diagnostics = [0u32; 4];
    for &(pixel, flag) in &fixture.rejected {
        expected_diagnostics[match flag {
            1 => 0,
            2 | 4 => 1,
            8 => 2,
            _ => unreachable!(),
        }] += 1;
        let material = if flag == 1 {
            INVALID
        } else {
            fixture.visibility[pixel].1.material_index
        };
        assert_eq!(
            &pixels[pixel * PIXEL_CLASS_WORDS..pixel * PIXEL_CLASS_WORDS + PIXEL_CLASS_WORDS],
            &[INVALID, INVALID, material, INVALID, flag],
            "rejected pixel {pixel}"
        );
    }
    let written = capacity.min(cpu.work.len() as u32) as usize;
    expected_diagnostics[3] = cpu.work.len() as u32 - written as u32;
    assert_eq!(diagnostics.as_slice(), expected_diagnostics, "diagnostics");
    let mut seen = vec![false; PIXELS as usize];
    let mut actual_counts = vec![0; cpu.buckets.len()];
    for (destination, item) in work
        .chunks_exact(SHADING_WORK_ITEM_WORDS)
        .take(written)
        .enumerate()
    {
        let pixel = item[0] as usize;
        assert!(
            pixel < seen.len() && !seen[pixel],
            "unique in-bounds work pixel"
        );
        seen[pixel] = true;
        let expected = cpu
            .work
            .iter()
            .find(|entry| entry.pixel_index == item[0])
            .expect("GPU work has a CPU counterpart");
        assert_eq!(
            item,
            [
                expected.pixel_index,
                expected.material_index,
                expected.shading_class,
                expected.response_kind,
                expected.flags,
                expected.right_shading_class,
                expected.right_response_kind,
                expected.response_meta,
            ]
        );
        let bucket = item[2] as usize;
        assert!(
            (offsets[bucket] as usize..(offsets[bucket] + counts[bucket]) as usize)
                .contains(&destination),
            "work inside its prefix range"
        );
        actual_counts[bucket] += 1;
        assert_eq!(
            &pixels[pixel * PIXEL_CLASS_WORDS..pixel * PIXEL_CLASS_WORDS + PIXEL_CLASS_WORDS],
            &[item[2], destination as u32, item[1], item[3], item[4]],
            "pixel/work back-reference"
        );
    }
    // Atomic ordering within a bucket is intentionally unspecified. Overflow
    // must nevertheless retain exactly the clipped prefix span of each bucket.
    for (bucket, count) in actual_counts.into_iter().enumerate() {
        assert_eq!(
            count,
            cpu.counts[bucket].min(capacity.saturating_sub(cpu.offsets[bucket]))
        );
    }
    assert!(
        work[written * SHADING_WORK_ITEM_WORDS..]
            .iter()
            .all(|word| *word == INVALID),
        "unwritten work/guard records unchanged"
    );
    for expected in &cpu.work {
        let pixel = expected.pixel_index as usize;
        let record =
            &pixels[pixel * PIXEL_CLASS_WORDS..pixel * PIXEL_CLASS_WORDS + PIXEL_CLASS_WORDS];
        assert_eq!(
            [record[0], record[2], record[3], record[4]],
            [
                expected.shading_class,
                expected.material_index,
                expected.response_kind,
                expected.flags
            ]
        );
        if !seen[pixel] {
            assert_eq!(record[1], INVALID, "overflow leaves no work index");
        }
    }
    if written == cpu.work.len() {
        let mut actual: Vec<[u32; SHADING_WORK_ITEM_WORDS]> = work
            .chunks_exact(SHADING_WORK_ITEM_WORDS)
            .take(written)
            .map(|words| words.try_into().unwrap())
            .collect();
        let mut expected: Vec<_> = cpu
            .work
            .iter()
            .map(|item| {
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
            })
            .collect();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected, "complete work-item multiset");
    }
}

fn cpu_work_plan(fixture: &Fixture) -> ShadingWorkPlan {
    ShadingWorkPlan::build_with_region_and_buckets_and_execution(
        fixture.visibility.iter().copied().filter(|(pixel, _, _)| {
            !fixture
                .rejected
                .iter()
                .any(|(rejected, _)| *rejected == *pixel as usize)
        }),
        &fixture.materials,
        PIXELS,
        &fixture.buckets,
        Some(&fixture.execution),
    )
    .expect("CPU accepted work plan")
}

fn assert_cpu_rejections(fixture: &Fixture) {
    let build = |pixels: Vec<(u32, VisibilityPixel, u32)>| {
        ShadingWorkPlan::build_with_region_and_buckets_and_execution(
            pixels,
            &fixture.materials,
            PIXELS,
            &fixture.buckets,
            Some(&fixture.execution),
        )
    };
    // CPU replay hard-errors on malformed live materials; GPU records the same
    // rejections diagnostically and continues. Verify each CPU rejection before
    // removing it from the accepted-pixel oracle.
    for &(pixel, flag) in &fixture.rejected {
        let result = build(vec![fixture.visibility[pixel]]);
        let inactive = fixture
            .materials
            .get(fixture.visibility[pixel].1.material_index as usize)
            .is_some_and(|header| header.is_active == 0);
        if flag == 8 || inactive {
            let header = &fixture.materials[fixture.visibility[pixel].1.material_index as usize];
            let error = if inactive {
                ClassificationError::Inactive
            } else if header.render_class > 3 || !matches!(header.render_domain, 0 | 3) {
                ClassificationError::NonSurfaceClass
            } else {
                ClassificationError::InvalidExecutionBinding
            };
            assert_eq!(
                result,
                Err(ShadingWorkPlanError::Classification {
                    pixel_index: pixel as u32,
                    error,
                }),
                "CPU rejects malformed pixel {pixel}"
            );
        } else {
            assert!(result
                .expect("CPU filters invalid visibility")
                .work
                .is_empty());
        }
    }
}

#[test]
fn real_gpu_classification_matches_program_ref_cpu_plan() {
    assert_eq!(
        size_of::<GpuMaterialHeader>(),
        29 * 4,
        "material ABI8 stride"
    );
    let fixture = fixture();
    assert_cpu_rejections(&fixture);
    let cpu = cpu_work_plan(&fixture);
    let continuous_items: Vec<_> = cpu
        .work
        .iter()
        .filter(|item| item.response_meta != INVALID)
        .collect();
    assert!(
        !continuous_items.is_empty(),
        "continuous MaskBlend fixture coverage"
    );
    for item in continuous_items {
        assert_ne!(item.right_shading_class, INVALID);
        assert_eq!(
            item.right_response_kind, 1,
            "continuous blend right arm is NPR"
        );
        assert_eq!(
            item.response_meta & u16::MAX as u32,
            fixture.visibility[item.pixel_index as usize]
                .1
                .coverage_and_flags,
            "packed continuous mask matches visibility metadata"
        );
    }
    assert!(
        cpu.counts.iter().filter(|count| **count != 0).count() > 9,
        "more than nine occupied buckets"
    );
    assert_eq!(
        cpu.work.len() + fixture.rejected.len(),
        PIXELS as usize,
        "every fixture pixel accounted for"
    );
    // Pin expected behavior independently of CPU/GPU agreement.
    for &(pixel, ref visibility, region) in &fixture.visibility {
        if fixture
            .rejected
            .iter()
            .any(|(index, _)| *index == pixel as usize)
        {
            continue;
        }
        let item = cpu
            .work
            .iter()
            .find(|item| item.pixel_index == pixel)
            .unwrap();
        let kind = item.response_kind;
        match visibility.material_index {
            0 | 1 | 2 if region == INVALID || region == 0 => {
                assert_eq!(kind, 0, "mixed-lobe Lit stays Lit")
            }
            0..=3 if region == 1 => assert_eq!(kind, 1, "binary endpoint 1"),
            3 if region == 0 => assert_eq!(kind, 0, "binary endpoint 0 overrides Npr"),
            4 => assert_eq!(kind, 2, "Unlit ignores region"),
            5..=16 => assert_eq!(kind, 3, "registered programs including id0 ignore region"),
            _ => {}
        }
    }
    let Some((device, queue)) = gpu() else {
        return;
    };
    let harness = Harness::new(device, queue);
    // Full parity, logical overflow with guard rows, physical buffer overflow,
    // and zero logical capacity. All runs execute the three production kernels.
    for (logical, physical) in [(PIXELS, PIXELS + 2), (23, PIXELS + 2), (PIXELS, 23), (0, 2)] {
        eprintln!("classification parity: logical={logical}, physical={physical}, buckets={}, accepted={}", cpu.buckets.len(), cpu.work.len());
        let result = harness.run(&fixture, logical, physical);
        assert_parity(&fixture, &cpu, &result, logical.min(physical));
    }
}

#[test]
fn real_gpu_opaque_classifier_eligibility_matches_cpu() {
    let mut fixture = fixture();
    fixture.materials.clear();
    fixture.visibility.clear();
    fixture.rejected.clear();
    // Surface classes 4/5/6 belong to the visibility transparent path, not
    // this opaque/masked classifier. Cover all seven classes plus unknown
    // values, against Surface/Hair, other domains, and unknown domains.
    for domain in [0, 1, 2, 3, 4, 5, INVALID] {
        for class in [0, 1, 2, 3, 4, 5, 6, 7, INVALID] {
            let mut header = fallback_material_header(7);
            header.render_class = class;
            header.render_domain = domain;
            fixture.materials.push(header);
        }
    }
    for pixel in 0..PIXELS {
        let material = pixel as usize % fixture.materials.len();
        let header = fixture.materials[material];
        fixture.visibility.push((
            pixel,
            VisibilityPixel {
                scene_index: pixel,
                scene_generation: 1,
                primitive_id: 0,
                geometry_lod_or_cluster: 0,
                material_index: material as u32,
                material_generation: header.generation,
                barycentrics_unorm16: 0,
                coverage_and_flags: 255,
            },
            INVALID,
        ));
        if header.render_class > 3 || !matches!(header.render_domain, 0 | 3) {
            fixture.rejected.push((pixel as usize, 8));
        }
    }
    let cpu = cpu_work_plan(&fixture);
    assert!(!cpu.work.is_empty());
    assert_eq!(cpu.work.len() + fixture.rejected.len(), PIXELS as usize);
    if let Some((device, queue)) = gpu() {
        let result = Harness::new(device, queue).run(&fixture, PIXELS, PIXELS + 2);
        assert_parity(&fixture, &cpu, &result, PIXELS);
    }
    // Check CPU eligibility even when an optional real-device run is skipped.
    assert_cpu_rejections(&fixture);
}
