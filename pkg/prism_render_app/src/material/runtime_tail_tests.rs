//! Regression tests for the production registry -> native header upload path.

use alloc::{vec, vec::Vec};
use core::{cell::RefCell, mem::size_of};

use prism_asset::{AssetId, AssetIndex};
use prism_image::Image;
use prism_render_architecture::{
    abi::{AbiHash, AbiVersion, GenerationalHandle},
    gpu_scene::GpuCompletionValue,
    shader_package::{
        ArtifactDigest, ManifestExpectation, ModuleId, PermutationSpace, ShaderPackageId,
        ShaderPackageManifest, ShaderPackageRegistry, ShaderPackageVariantManifest,
    },
};
use prism_render_driver::{
    BufferId, BufferUsages, CommandBuffer, DeviceDescriptor, ImageDataLayout, Instance,
    RenderQueue, RequestAdapterOptions, TextureWrite,
};
use prism_render_driver_null::{NullDevice, NullInstance};
use prism_render_material::{
    fallback_material_record, inactive_material_header, BuiltinProgram, ChannelMask, ClosureKind,
    CompiledEntry, DynamicChannels, ExecutionFamily, GpuMaterialHeader, GpuMaterialTexture,
    MaterialExecutionBinding, MaterialExecutionTables, ProgramRef, ShadingPlan,
};

use super::{
    sync_dirty_material_buffers, MaterialParameterBuffers, MaterialSyncReport, RenderMaterials,
    TextureRequest,
};

struct BufferWrite {
    buffer: BufferId,
    offset: u64,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct RecordingQueue {
    writes: RefCell<Vec<BufferWrite>>,
}

impl RenderQueue for RecordingQueue {
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
        self.writes.borrow_mut().push(BufferWrite {
            buffer,
            offset,
            bytes: data.to_vec(),
        });
    }

    fn write_texture(&self, _: TextureWrite, _: &[u8], _: ImageDataLayout) {
        panic!("material buffer upload must not write textures");
    }

    fn submit(&self, _: &[CommandBuffer]) {
        panic!("material buffer upload must not submit commands");
    }
}

impl RecordingQueue {
    fn clear(&self) {
        self.writes.borrow_mut().clear();
    }

    fn assert_quiet(&self) {
        assert!(self.writes.borrow().is_empty());
    }

    fn assert_header_only(&self, buffers: &MaterialParameterBuffers, index: u32) {
        let writes = self.writes.borrow();
        assert_eq!(
            writes.len(),
            1,
            "runtime-only changes must not upload heaps"
        );
        assert_eq!(writes[0].buffer, buffers.headers().unwrap());
        assert_eq!(
            writes[0].offset,
            u64::from(index) * size_of::<GpuMaterialHeader>() as u64
        );
        assert_eq!(writes[0].bytes.len(), size_of::<GpuMaterialHeader>());
        drop(writes);
        assert_eq!(self.header(buffers, index), *buffers.header(index).unwrap());
    }

    fn header(&self, buffers: &MaterialParameterBuffers, index: u32) -> GpuMaterialHeader {
        let start = u64::from(index) * size_of::<GpuMaterialHeader>() as u64;
        let size = size_of::<GpuMaterialHeader>();
        let writes = self.writes.borrow();
        let write = writes
            .iter()
            .rev()
            .find(|write| {
                write.buffer == buffers.headers().unwrap()
                    && write.offset <= start
                    && start + size as u64 <= write.offset + write.bytes.len() as u64
            })
            .expect("the header must have been uploaded, not just mirrored");
        let relative = (start - write.offset) as usize;
        bytemuck::pod_read_unaligned(&write.bytes[relative..relative + size])
    }
}

fn null_device() -> NullDevice {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap()
        .0
}

struct Fixture {
    device: NullDevice,
    queue: RecordingQueue,
    buffers: MaterialParameterBuffers,
    materials: RenderMaterials,
}

impl Fixture {
    fn new() -> Self {
        let device = null_device();
        let buffers = MaterialParameterBuffers::new(&device, 16, 2048, 128, BufferUsages::NONE);
        Self {
            device,
            queue: RecordingQueue::default(),
            buffers,
            materials: RenderMaterials::new(32),
        }
    }

    fn publish(&mut self, revision: u64) -> GenerationalHandle {
        let handle = self.materials.registry_mut().allocate().unwrap();
        let mut record = fallback_material_record(handle, revision);
        record.response_plan = ShadingPlan::single_builtin(BuiltinProgram::Unlit);
        self.materials.publish(record).unwrap();
        handle
    }

    fn sync(&mut self) -> MaterialSyncReport {
        self.queue.clear();
        sync_dirty_material_buffers(
            &mut self.buffers,
            &mut self.materials,
            &self.device,
            &self.queue,
        )
        .unwrap_or_default()
    }

    fn set_runtime(&mut self, handle: GenerationalHandle, binding: MaterialExecutionBinding) {
        self.materials
            .registry_mut()
            .set_dynamic_channels(handle, channels())
            .unwrap();
        self.materials
            .registry_mut()
            .set_execution_binding(handle, binding)
            .unwrap();
    }
}

fn channels() -> DynamicChannels {
    DynamicChannels {
        wetness: 0.25,
        snow: 0.5,
        damage: 0.75,
        dissolve: 1.0,
    }
}

/// Use the real package verification, dense-table and registry validation APIs;
/// no synthetic execution identities or alternate runtime state are installed.
fn install_execution(materials: &mut RenderMaterials) -> [MaterialExecutionBinding; 2] {
    let permutations = PermutationSpace::new([2]);
    let manifest = ShaderPackageManifest {
        id: ShaderPackageId::new("native-header-runtime-test"),
        source_revision: "1".into(),
        entry_point: "main".into(),
        abi_version: AbiVersion::from_parts(1, 0),
        abi_hash: AbiHash::from_bytes(
            prism_render_material::response_source::RESPONSE_FUNCTION_ABI.as_bytes(),
        ),
        variants: ShaderPackageVariantManifest::from_digests(
            &permutations,
            [ArtifactDigest([1; 32]), ArtifactDigest([2; 32])],
        )
        .unwrap(),
    };
    let compiled = CompiledEntry::from_response_wgsl(
        manifest.clone(),
        vec![
            "var response: RegisteredResponse; return response;".into(),
            "var response: RegisteredResponse; return response;".into(),
        ],
    )
    .unwrap();
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
    let mut tables = MaterialExecutionTables::new();
    let package = tables.register_package(&compiled, &packages).unwrap();
    // Register a preceding program/plan so the tested indices aren't all zero.
    let mut bindings = None;
    for (index, builtin) in [BuiltinProgram::Lit, BuiltinProgram::Unlit]
        .into_iter()
        .enumerate()
    {
        let program = tables
            .register_program(
                ProgramRef::Builtin(builtin),
                &compiled,
                package,
                ModuleId(index as u64),
                permutations.clone(),
            )
            .unwrap();
        let plan = tables
            .register_plan(
                ShadingPlan::single_builtin(builtin),
                program,
                None,
                match builtin {
                    BuiltinProgram::Lit => ExecutionFamily::FusedBuiltin,
                    BuiltinProgram::Unlit => ExecutionFamily::SplitResponse,
                    _ => unreachable!("the fixture only registers Lit and Unlit"),
                },
                ChannelMask::default(),
            )
            .unwrap();
        bindings = Some([
            tables
                .bind_raw(plan.get(), program.get(), package.get(), 0)
                .unwrap(),
            tables
                .bind_raw(plan.get(), program.get(), package.get(), 1)
                .unwrap(),
        ]);
    }
    materials
        .registry_mut()
        .install_execution_tables(&tables)
        .unwrap();
    bindings.unwrap()
}

fn assert_tail(
    row: &GpuMaterialHeader,
    dynamic: DynamicChannels,
    binding: Option<MaterialExecutionBinding>,
) {
    assert_eq!(
        [
            row.dynamic_wetness,
            row.dynamic_snow,
            row.dynamic_damage,
            row.dynamic_dissolve
        ],
        dynamic.to_words()
    );
    assert_eq!(
        [
            row.execution_plan_index,
            row.execution_program_index,
            row.execution_package_index,
            row.execution_permutation_low,
            row.execution_permutation_high,
        ],
        binding
            .map(MaterialExecutionBinding::to_words)
            .unwrap_or([u32::MAX; 5])
    );
}

#[test]
fn registry_upload_preserves_native_offsets_identity_and_runtime_tail() {
    let mut f = Fixture::new();
    let first = f.materials.registry_mut().allocate().unwrap();
    let handle = f.materials.registry_mut().allocate().unwrap();
    let mut prefix = fallback_material_record(first, 3);
    prefix.textures = vec![GpuMaterialTexture::non_virtual(3, 2, 0, 3); 2];
    let mut record = fallback_material_record(handle, 17);
    record.closure_mask |= 1 << ClosureKind::ClearCoat as u32;
    record.textures = vec![GpuMaterialTexture::non_virtual(4, 7, 1, 4)];
    record.response_plan = ShadingPlan::single_builtin(BuiltinProgram::Unlit);
    let prefix_words = prefix.packed_parameters().pack().len() as u32;
    f.materials.publish(prefix).unwrap();
    f.materials.publish(record.clone()).unwrap();
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, binding);
    let report = f.sync();
    assert!(!report.grew);
    let row = *f.buffers.header(handle.index).unwrap();
    let fallback_words = f.buffers.header(0).unwrap().parameter_size / 4;
    assert_eq!(row.parameter_offset, fallback_words + prefix_words);
    assert_ne!(
        row.parameter_offset, handle.index,
        "native offsets are word offsets, not registry row indices"
    );
    assert_eq!(row.texture_offset, 2);
    assert_eq!(
        row.parameter_size as usize,
        record.packed_parameters().packed_size_bytes()
    );
    assert_eq!(
        (row.generation, row.revision, row.is_active),
        (handle.generation, 17, 1)
    );
    let mut expected = record.header_with_execution(
        row.parameter_offset,
        2,
        f.materials.snapshot().epoch,
        Some(binding),
    );
    [
        expected.dynamic_wetness,
        expected.dynamic_snow,
        expected.dynamic_damage,
        expected.dynamic_dissolve,
    ] = channels().to_words();
    assert_eq!(
        row, expected,
        "only the runtime tail augments the canonically placed header"
    );
    assert_eq!(f.queue.header(&f.buffers, handle.index), expected);
    assert_eq!(
        f.buffers.parameters_of(handle.index).unwrap(),
        record.packed_parameters().pack().as_slice()
    );
    assert_eq!(
        f.buffers.textures_of(handle.index).unwrap(),
        record.textures.as_slice()
    );
    assert_tail(f.buffers.header(0).unwrap(), DynamicChannels::NONE, None);
}

#[test]
fn execution_table_reinstall_reuploads_live_rows_before_snapshot_version_advances() {
    let mut f = Fixture::new();
    let handle = f.publish(17);
    let first_binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, first_binding);
    f.sync();
    let first_version = f.materials.registry().execution_table_version();
    let first_row = *f.buffers.header(handle.index).unwrap();

    // Reinstalling tables is a new registry publication and marks every row
    // dirty. The synchronizer still compares the rebuilt header against its
    // mirror: when the dense indices are byte-identical, no redundant header
    // write is needed, but the material snapshot must advance to the new
    // execution-table version so the two GPU snapshots remain paired.
    let replacement_binding = install_execution(&mut f.materials)[1];
    assert_eq!(replacement_binding, first_binding);
    f.queue.clear();
    let report = sync_dirty_material_buffers(&mut f.buffers, &mut f.materials, &f.device, &f.queue)
        .expect("table reinstall must consume the dirty material batch");
    assert_eq!(report, MaterialSyncReport::default());
    f.queue.assert_quiet();
    let second_version = f.materials.registry().execution_table_version();
    assert_ne!(first_version, second_version);
    assert_eq!(f.buffers.execution_table_version(), second_version);
    assert_eq!(
        second_version,
        f.materials.registry().execution_table_version()
    );
    let second_row = *f.buffers.header(handle.index).unwrap();
    assert_eq!(
        second_row.execution_binding_words(),
        first_row.execution_binding_words()
    );
    assert_eq!(second_row, first_row);
}

#[test]
fn dynamic_only_changes_upload_one_header_without_republishing_or_rewriting_heaps() {
    let mut f = Fixture::new();
    let handle = f.publish(9);
    f.sync();
    let before = *f.buffers.header(handle.index).unwrap();
    let words = f.buffers.parameters_of(handle.index).unwrap().to_vec();
    let snapshot = f.materials.snapshot();
    let ids = (
        f.buffers.headers(),
        f.buffers.parameters(),
        f.buffers.textures(),
    );
    // Exercise the registry's sanitization as well as all four live channels.
    let dirty = DynamicChannels {
        wetness: f32::NAN,
        snow: 1.5,
        damage: -0.5,
        dissolve: 0.4,
    };
    f.materials
        .registry_mut()
        .set_dynamic_channels(handle, dirty)
        .unwrap();
    let report = f.sync();
    assert_eq!(
        report,
        MaterialSyncReport {
            headers_written: 1,
            ..Default::default()
        }
    );
    f.queue.assert_header_only(&f.buffers, handle.index);
    let row = f.buffers.header(handle.index).unwrap();
    let mut expected = before;
    [
        expected.dynamic_wetness,
        expected.dynamic_snow,
        expected.dynamic_damage,
        expected.dynamic_dissolve,
    ] = dirty.to_words();
    assert_eq!(*row, expected);
    assert_eq!(f.materials.snapshot(), snapshot);
    assert_eq!(
        f.buffers.parameters_of(handle.index).unwrap(),
        words.as_slice()
    );
    assert_eq!(
        (
            f.buffers.headers(),
            f.buffers.parameters(),
            f.buffers.textures()
        ),
        ids
    );
    f.materials
        .registry_mut()
        .set_dynamic_channels(handle, dirty)
        .unwrap();
    assert_eq!(f.sync(), MaterialSyncReport::default());
    f.queue.assert_quiet();
}

#[test]
fn execution_only_set_change_and_clear_are_dirty_and_never_rewrite_parameters() {
    let mut f = Fixture::new();
    let handle = f.publish(11);
    let bindings = install_execution(&mut f.materials);
    f.sync();
    let before = *f.buffers.header(handle.index).unwrap();
    let snapshot = f.materials.snapshot();
    for binding in [Some(bindings[0]), Some(bindings[1]), None] {
        if let Some(binding) = binding {
            f.materials
                .registry_mut()
                .set_execution_binding(handle, binding)
                .unwrap();
        } else {
            f.materials
                .registry_mut()
                .clear_execution_binding(handle)
                .unwrap();
        }
        assert_eq!(
            f.sync(),
            MaterialSyncReport {
                headers_written: 1,
                ..Default::default()
            }
        );
        f.queue.assert_header_only(&f.buffers, handle.index);
        let row = f.buffers.header(handle.index).unwrap();
        assert_tail(row, DynamicChannels::NONE, binding);
        assert_eq!(
            (
                row.parameter_offset,
                row.texture_offset,
                row.generation,
                row.revision
            ),
            (
                before.parameter_offset,
                before.texture_offset,
                before.generation,
                before.revision
            )
        );
        assert_eq!(f.materials.snapshot(), snapshot);
        // Reasserting an identical binding (or clearing twice) must stay quiet.
        if let Some(binding) = binding {
            f.materials
                .registry_mut()
                .set_execution_binding(handle, binding)
                .unwrap();
        } else {
            f.materials
                .registry_mut()
                .clear_execution_binding(handle)
                .unwrap();
        }
        assert_eq!(f.sync(), MaterialSyncReport::default());
        f.queue.assert_quiet();
    }
}

#[test]
fn registry_rejects_uninstalled_execution_and_stale_handles_without_uploads() {
    let mut f = Fixture::new();
    let handle = f.publish(5);
    let mut other = RenderMaterials::new(4);
    let binding = install_execution(&mut other)[0];
    f.sync();
    let before = *f.buffers.header(handle.index).unwrap();
    assert!(f
        .materials
        .registry_mut()
        .set_execution_binding(handle, binding)
        .is_err());
    let stale = GenerationalHandle::new(handle.index, handle.generation + 1);
    assert!(f
        .materials
        .registry_mut()
        .set_dynamic_channels(stale, channels())
        .is_err());
    assert!(f
        .materials
        .registry_mut()
        .set_execution_binding(stale, binding)
        .is_err());
    assert_eq!(f.sync(), MaterialSyncReport::default());
    f.queue.assert_quiet();
    assert_eq!(*f.buffers.header(handle.index).unwrap(), before);
}

#[test]
fn republish_relocates_authored_data_and_uses_registry_reset_runtime_tail() {
    let mut f = Fixture::new();
    let handle = f.publish(8);
    f.publish(1); // Occupy the following run so a wider block really relocates.
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, binding);
    f.sync();
    let before = *f.buffers.header(handle.index).unwrap();
    let mut record = fallback_material_record(handle, 9);
    record.closure_mask |= 1 << ClosureKind::ClearCoat as u32;
    f.materials.publish(record.clone()).unwrap();
    let report = f.sync();
    assert_eq!(report.relocated, 1);
    assert_eq!(report.parameters_written, 1);
    let row = f.buffers.header(handle.index).unwrap();
    assert_ne!(row.parameter_offset, before.parameter_offset);
    assert_eq!((row.generation, row.revision), (handle.generation, 9));
    assert_tail(row, DynamicChannels::NONE, None);
    assert_eq!(f.queue.header(&f.buffers, handle.index), *row);
    assert_eq!(
        f.buffers.parameters_of(handle.index).unwrap(),
        record.packed_parameters().pack().as_slice()
    );
}

#[test]
fn revision_only_republish_resets_runtime_tail_without_rewriting_heaps() {
    let mut f = Fixture::new();
    let handle = f.publish(12);
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, binding);
    f.sync();
    let before = *f.buffers.header(handle.index).unwrap();
    let mut record = f.materials.registry().get(handle).unwrap().clone();
    record.revision = 13;
    f.materials.publish(record).unwrap();
    assert_eq!(
        f.sync(),
        MaterialSyncReport {
            headers_written: 1,
            ..Default::default()
        }
    );
    f.queue.assert_header_only(&f.buffers, handle.index);
    let row = f.queue.header(&f.buffers, handle.index);
    assert_eq!((row.generation, row.revision), (handle.generation, 13));
    assert_eq!(
        (row.parameter_offset, row.texture_offset),
        (before.parameter_offset, before.texture_offset)
    );
    assert_tail(&row, DynamicChannels::NONE, None);
}

#[test]
fn replacement_device_replays_complete_tail_then_runtime_updates_are_header_only() {
    let mut f = Fixture::new();
    let handle = f.publish(7);
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, binding);
    f.sync();
    let expected = *f.buffers.header(handle.index).unwrap();
    let words = f.buffers.parameters_of(handle.index).unwrap().to_vec();
    let replacement = null_device();
    let queue = RecordingQueue::default();
    f.buffers.recover(&replacement, &queue);
    assert_eq!(
        queue.writes.borrow().len(),
        3,
        "all mirrors replay on device replacement"
    );
    assert_eq!(queue.header(&f.buffers, handle.index), expected);
    let writes = queue.writes.borrow();
    let parameters = writes
        .iter()
        .find(|write| write.buffer == f.buffers.parameters().unwrap())
        .unwrap();
    let offset = expected.parameter_offset as usize * 4;
    let bytes: &[u8] = bytemuck::cast_slice(words.as_slice());
    assert_eq!(&parameters.bytes[offset..offset + bytes.len()], bytes);
    drop(writes);
    // No registry re-publication or secondary runtime state is needed.
    queue.clear();
    let report =
        sync_dirty_material_buffers(&mut f.buffers, &mut f.materials, &replacement, &queue)
            .unwrap_or_default();
    assert_eq!(report, MaterialSyncReport::default());
    queue.assert_quiet();
    f.materials
        .registry_mut()
        .set_dynamic_channels(handle, DynamicChannels::NONE)
        .unwrap();
    let report =
        sync_dirty_material_buffers(&mut f.buffers, &mut f.materials, &replacement, &queue)
            .unwrap_or_default();
    assert_eq!(
        report,
        MaterialSyncReport {
            headers_written: 1,
            ..Default::default()
        }
    );
    queue.assert_header_only(&f.buffers, handle.index);
    assert_tail(
        &queue.header(&f.buffers, handle.index),
        DynamicChannels::NONE,
        Some(binding),
    );
}

#[test]
fn retire_and_reuse_clear_tail_and_preserve_generation_rejection() {
    let mut f = Fixture::new();
    let handle = f.publish(6);
    let image = AssetId::<Image>::new(AssetIndex::from_parts(10, 1));
    let textures = f
        .materials
        .resolve_textures(handle.index, [(0, TextureRequest::Image(image))])
        .expect("fixture image fits the texture heap");
    let mut record = f.materials.registry().get(handle).unwrap().clone();
    record.revision = 7;
    record.textures = textures;
    f.materials.publish(record).unwrap();
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, binding);
    f.sync();
    let slot = f.materials.texture_slot(image).unwrap();
    f.materials
        .registry_mut()
        .retire(handle, GpuCompletionValue(3))
        .unwrap();
    assert_eq!(
        f.sync(),
        MaterialSyncReport {
            headers_written: 1,
            ..Default::default()
        }
    );
    f.queue.assert_header_only(&f.buffers, handle.index);
    assert_eq!(
        f.queue.header(&f.buffers, handle.index),
        inactive_material_header(handle.generation)
    );
    assert!(f.buffers.parameters_of(handle.index).is_none());
    assert!(f.buffers.textures_of(handle.index).is_none());
    assert!(f.materials.texture_slot(image).is_none());
    let replacement = AssetId::<Image>::new(AssetIndex::from_parts(11, 1));
    let recycled = f.materials.textures_mut().acquire(replacement).unwrap();
    assert_eq!(recycled.index, slot.index);
    assert_ne!(recycled.generation, slot.generation);
    assert!(f.materials.textures_mut().release(replacement));
    assert_eq!(
        f.materials
            .registry_mut()
            .reclaim_completed(GpuCompletionValue(3)),
        1
    );
    let reused = f.publish(1);
    assert_eq!(reused.index, handle.index);
    assert_ne!(reused.generation, handle.generation);
    assert!(f
        .materials
        .registry_mut()
        .set_dynamic_channels(handle, channels())
        .is_err());
    assert!(f
        .materials
        .registry_mut()
        .set_execution_binding(handle, binding)
        .is_err());
    f.sync();
    let row = f.queue.header(&f.buffers, reused.index);
    assert_eq!(
        (row.generation, row.revision, row.is_active),
        (reused.generation, 1, 1)
    );
    assert_tail(&row, DynamicChannels::NONE, None);
}

#[test]
fn buffer_growth_replays_previously_uploaded_runtime_tail() {
    let mut f = Fixture::new();
    let handle = f.publish(4);
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(handle, binding);
    f.sync();
    let expected = *f.buffers.header(handle.index).unwrap();
    for _ in 0..16 {
        f.publish(1);
    }
    assert!(f.sync().grew);
    assert_eq!(f.queue.header(&f.buffers, handle.index), expected);
    assert_eq!(*f.buffers.header(handle.index).unwrap(), expected);
}

#[test]
fn registry_resync_uses_current_generation_and_preserves_runtime_tail() {
    let mut f = Fixture::new();
    let old = f.publish(9);
    let bindings = install_execution(&mut f.materials);
    f.set_runtime(old, bindings[0]);
    f.sync();
    f.materials
        .registry_mut()
        .retire(old, GpuCompletionValue(1))
        .unwrap();
    assert_eq!(
        f.materials
            .registry_mut()
            .reclaim_completed(GpuCompletionValue(1)),
        1
    );
    let current = f.publish(1);
    assert_eq!(current.index, old.index);
    assert_ne!(current.generation, old.generation);
    f.set_runtime(current, bindings[1]);
    f.queue.clear();
    let report =
        f.buffers
            .sync_registry(&f.device, &f.queue, f.materials.registry(), [current.index]);
    assert_eq!(report.headers_written, 1);
    assert_eq!(report.parameters_written, 0);
    f.queue.assert_header_only(&f.buffers, current.index);
    let expected = f.queue.header(&f.buffers, current.index);
    assert_eq!(
        (expected.generation, expected.revision),
        (current.generation, 1)
    );
    assert_tail(&expected, channels(), Some(bindings[1]));

    f.queue.clear();
    assert_eq!(
        f.buffers
            .sync_registry(&f.device, &f.queue, f.materials.registry(), [current.index],),
        MaterialSyncReport::default()
    );
    f.queue.assert_quiet();
    assert_eq!(*f.buffers.header(current.index).unwrap(), expected);
}

#[test]
fn stale_remove_cannot_free_reused_heaps_or_reset_live_runtime_tail() {
    let mut f = Fixture::new();
    let old = f.publish(3);
    f.sync();
    f.materials
        .registry_mut()
        .retire(old, GpuCompletionValue(2))
        .unwrap();
    f.materials
        .registry_mut()
        .reclaim_completed(GpuCompletionValue(2));
    let current = f.publish(1);
    assert_eq!(current.index, old.index);
    assert_ne!(current.generation, old.generation);
    let image = AssetId::<Image>::new(AssetIndex::from_parts(21, 1));
    let textures = f
        .materials
        .resolve_textures(current.index, [(0, TextureRequest::Image(image))])
        .expect("fixture image fits the texture heap");
    let mut record = f.materials.registry().get(current).unwrap().clone();
    record.revision += 1;
    record.textures = textures;
    f.materials.publish(record).unwrap();
    let binding = install_execution(&mut f.materials)[1];
    f.set_runtime(current, binding);
    f.sync();
    let expected = f.queue.header(&f.buffers, current.index);
    let words = f.buffers.parameters_of(current.index).unwrap().to_vec();
    let textures = f.buffers.textures_of(current.index).unwrap().to_vec();
    let capacities = (f.buffers.used_words(), f.buffers.used_texture_rows());
    f.queue.clear();
    assert!(!f.buffers.remove(&f.queue, old.index, old.generation));
    assert!(!f
        .buffers
        .remove(&f.queue, current.index, current.generation + 1));
    assert!(!f.buffers.remove(&f.queue, u32::MAX, current.generation));
    f.queue.assert_quiet();
    assert_eq!(*f.buffers.header(current.index).unwrap(), expected);
    assert_eq!(f.buffers.parameters_of(current.index).unwrap(), words);
    assert_eq!(f.buffers.textures_of(current.index).unwrap(), textures);
    assert_eq!(
        (f.buffers.used_words(), f.buffers.used_texture_rows()),
        capacities
    );
    assert_eq!(f.materials.textures().refcount(image), 1);
    assert_tail(&expected, channels(), Some(binding));

    assert!(f
        .buffers
        .remove(&f.queue, current.index, current.generation));
    f.queue.assert_header_only(&f.buffers, current.index);
    assert_eq!(
        f.queue.header(&f.buffers, current.index),
        inactive_material_header(current.generation)
    );
    assert!(f.buffers.parameters_of(current.index).is_none());
    assert!(f.buffers.textures_of(current.index).is_none());
    f.queue.clear();
    assert!(!f
        .buffers
        .remove(&f.queue, current.index, current.generation));
    f.queue.assert_quiet();
}

#[test]
fn coalesced_retire_reuse_releases_unreferenced_textures_and_resets_tail() {
    let mut f = Fixture::new();
    let old = f.publish(4);
    let image = AssetId::<Image>::new(AssetIndex::from_parts(22, 1));
    let shared = AssetId::<Image>::new(AssetIndex::from_parts(23, 1));
    let shared_slot = f.materials.textures_mut().acquire(shared).unwrap();
    let textures = f
        .materials
        .resolve_textures(
            old.index,
            [
                (0, TextureRequest::Image(image)),
                (1, TextureRequest::Image(image)),
                (2, TextureRequest::Image(shared)),
            ],
        )
        .expect("fixture images fit the texture heap");
    let slot = f.materials.texture_slot(image).unwrap();
    let mut record = f.materials.registry().get(old).unwrap().clone();
    record.revision += 1;
    record.textures = textures;
    f.materials.publish(record).unwrap();
    let binding = install_execution(&mut f.materials)[0];
    f.set_runtime(old, binding);
    f.sync();
    assert_eq!(f.materials.textures().refcount(image), 2);
    assert_eq!(f.materials.textures().refcount(shared), 2);
    f.materials
        .registry_mut()
        .retire(old, GpuCompletionValue(3))
        .unwrap();
    f.materials
        .registry_mut()
        .reclaim_completed(GpuCompletionValue(3));
    let current = f.publish(1);
    assert_eq!(current.index, old.index);
    assert_ne!(current.generation, old.generation);

    // No prepare between retirement and publication, and no resolve for the
    // new texture-free record: the old ownership must still be released.
    assert_eq!(
        f.sync(),
        MaterialSyncReport {
            headers_written: 1,
            ..Default::default()
        }
    );
    f.queue.assert_header_only(&f.buffers, current.index);
    let row = f.queue.header(&f.buffers, current.index);
    assert_eq!(
        (row.generation, row.revision, row.texture_count),
        (current.generation, 1, 0)
    );
    assert_tail(&row, DynamicChannels::NONE, None);
    assert!(f.buffers.textures_of(current.index).unwrap().is_empty());
    assert!(f.materials.texture_slot(image).is_none());
    assert!(!f.materials.textures().validates(slot));
    assert_eq!(f.materials.textures().refcount(shared), 1);
    assert_eq!(f.materials.texture_slot(shared), Some(shared_slot));
    assert_eq!(f.materials.release_textures(current.index), 0);
    assert_eq!(f.sync(), MaterialSyncReport::default());
    f.queue.assert_quiet();
}

#[test]
fn coalesced_reuse_retains_only_live_texture_reference_multiplicity() {
    let mut f = Fixture::new();
    let old = f.publish(4);
    let kept = AssetId::<Image>::new(AssetIndex::from_parts(24, 1));
    let dropped = AssetId::<Image>::new(AssetIndex::from_parts(25, 1));
    let textures = f
        .materials
        .resolve_textures(
            old.index,
            [
                (0, TextureRequest::Image(kept)),
                (1, TextureRequest::Image(kept)),
                (2, TextureRequest::Image(dropped)),
            ],
        )
        .expect("fixture images fit the texture heap");
    let kept_slot = f.materials.texture_slot(kept).unwrap();
    let mut record = f.materials.registry().get(old).unwrap().clone();
    record.revision += 1;
    record.textures = textures.clone();
    f.materials.publish(record).unwrap();
    f.sync();
    f.materials
        .registry_mut()
        .retire(old, GpuCompletionValue(4))
        .unwrap();
    f.materials
        .registry_mut()
        .reclaim_completed(GpuCompletionValue(4));
    let current = f.materials.registry_mut().allocate().unwrap();
    assert_eq!(current.index, old.index);
    assert_ne!(current.generation, old.generation);
    let mut record = fallback_material_record(current, 1);
    record.textures = vec![textures[0]];
    // Exercise direct registry publication too: cleanup cannot depend solely
    // on callers going through RenderMaterials::publish or resolve_textures.
    f.materials.registry_mut().publish(record).unwrap();
    f.sync();
    let row = f.queue.header(&f.buffers, current.index);
    assert_eq!(row.texture_count, 1);
    assert_eq!(row.generation, current.generation);
    assert_eq!(
        f.buffers.textures_of(current.index).unwrap(),
        &textures[..1]
    );
    assert_eq!(f.materials.textures().refcount(kept), 1);
    assert_eq!(f.materials.texture_slot(kept), Some(kept_slot));
    assert_eq!(f.materials.textures().refcount(dropped), 0);

    f.materials
        .registry_mut()
        .retire(current, GpuCompletionValue(5))
        .unwrap();
    f.sync();
    assert_eq!(
        f.queue.header(&f.buffers, current.index),
        inactive_material_header(current.generation)
    );
    assert!(f.materials.texture_slot(kept).is_none());
    assert_eq!(f.materials.release_textures(current.index), 0);
}

#[test]
fn coalesced_reuse_then_retire_removes_the_last_uploaded_generation() {
    let mut f = Fixture::new();
    let old = f.publish(2);
    let image = AssetId::<Image>::new(AssetIndex::from_parts(29, 1));
    let textures = f
        .materials
        .resolve_textures(old.index, [(0, TextureRequest::Image(image))])
        .expect("fixture image fits the texture heap");
    let mut record = f.materials.registry().get(old).unwrap().clone();
    record.revision += 1;
    record.textures = textures;
    f.materials.publish(record).unwrap();
    let binding = install_execution(&mut f.materials)[0];
    f.set_runtime(old, binding);
    f.sync();
    f.materials
        .registry_mut()
        .retire(old, GpuCompletionValue(8))
        .unwrap();
    f.materials
        .registry_mut()
        .reclaim_completed(GpuCompletionValue(8));
    let current = f.publish(1);
    assert_eq!(current.index, old.index);
    assert_ne!(current.generation, old.generation);
    f.materials
        .registry_mut()
        .retire(current, GpuCompletionValue(9))
        .unwrap();
    assert_eq!(
        f.materials.registry().generation_at(current.index),
        Some(current.generation)
    );
    assert_eq!(
        f.sync(),
        MaterialSyncReport {
            headers_written: 1,
            ..Default::default()
        }
    );
    f.queue.assert_header_only(&f.buffers, old.index);
    assert_eq!(
        f.queue.header(&f.buffers, old.index),
        inactive_material_header(old.generation)
    );
    assert!(f.buffers.parameters_of(old.index).is_none());
    assert!(f.buffers.textures_of(old.index).is_none());
    assert!(f.materials.texture_slot(image).is_none());
    assert_eq!(f.sync(), MaterialSyncReport::default());
    f.queue.assert_quiet();
}

#[test]
fn retirement_cleanup_cannot_release_a_replacement_texture_generation() {
    let mut f = Fixture::new();
    let handle = f.publish(1);
    let image = AssetId::<Image>::new(AssetIndex::from_parts(26, 1));
    let textures = f
        .materials
        .resolve_textures(handle.index, [(0, TextureRequest::Image(image))])
        .expect("fixture image fits the texture heap");
    let mut record = f.materials.registry().get(handle).unwrap().clone();
    record.revision += 1;
    record.textures = textures;
    f.materials.publish(record).unwrap();
    f.sync();
    let old_slot = f.materials.texture_slot(image).unwrap();
    assert!(f.materials.textures_mut().release(image));
    let replacement = f.materials.textures_mut().acquire(image).unwrap();
    assert_eq!(replacement.index, old_slot.index);
    assert_ne!(replacement.generation, old_slot.generation);
    f.materials
        .registry_mut()
        .retire(handle, GpuCompletionValue(6))
        .unwrap();
    f.sync();
    assert_eq!(
        f.queue.header(&f.buffers, handle.index),
        inactive_material_header(handle.generation)
    );
    assert_eq!(f.materials.texture_slot(image), Some(replacement));
    assert_eq!(f.materials.textures().refcount(image), 1);
    assert_eq!(f.materials.release_textures(handle.index), 0);
}

#[test]
fn texture_resolution_overflow_rolls_back_and_preserves_published_material() {
    let mut f = Fixture::new();
    f.materials
        .set_texture_capacity(super::super::texture_heap::FIRST_DYNAMIC_SLOT + 2);
    let original = AssetId::<Image>::new(AssetIndex::from_parts(27, 1));
    let first_new = AssetId::<Image>::new(AssetIndex::from_parts(28, 1));
    let overflowing = AssetId::<Image>::new(AssetIndex::from_parts(29, 1));
    let handle = f.publish(1);
    let old_rows = f
        .materials
        .resolve_textures(handle.index, [(0, TextureRequest::Image(original))])
        .expect("original image fits");
    let mut record = f.materials.registry().get(handle).unwrap().clone();
    record.revision += 1;
    record.textures = old_rows.clone();
    f.materials.publish(record).unwrap();
    f.sync();

    let failure = f.materials.resolve_textures(
        handle.index,
        [
            (0, TextureRequest::Image(first_new)),
            (1, TextureRequest::Image(overflowing)),
        ],
    );
    assert_eq!(
        failure,
        Err(super::TextureResolutionError::CapacityExhausted {
            material_index: handle.index,
            semantic: 1,
        })
    );
    assert_eq!(
        f.materials.registry().get(handle).unwrap().textures,
        old_rows
    );
    assert_eq!(f.materials.textures().refcount(original), 1);
    assert_eq!(f.materials.texture_slot(first_new), None);
    assert_eq!(f.materials.textures().refcount(first_new), 0);
    assert_eq!(f.materials.texture_slot(overflowing), None);
    assert_eq!(f.materials.textures().refcount(overflowing), 0);
    assert_eq!(f.queue.header(&f.buffers, handle.index).texture_count, 1);

    f.materials.release_textures(handle.index);
    let retry = f
        .materials
        .resolve_textures(handle.index, [(0, TextureRequest::Image(first_new))])
        .expect("resolution succeeds after capacity is released");
    assert_eq!(retry.len(), 1);
    let slot = f.materials.texture_slot(first_new).unwrap();
    assert_eq!(f.materials.textures().refcount(first_new), 1);
    f.materials
        .registry_mut()
        .retire(handle, GpuCompletionValue(7))
        .unwrap();
    f.sync();
    assert_eq!(f.materials.texture_slot(first_new), None);
    assert_eq!(f.materials.textures().refcount(first_new), 0);
    assert!(slot.index >= super::super::texture_heap::FIRST_DYNAMIC_SLOT);
}
