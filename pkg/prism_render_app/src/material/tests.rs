//! Behavioural tests for the material ABI buffers.

use prism_render_driver::{
    BufferUsages, DeviceDescriptor, Instance, RequestAdapterOptions,
};
use prism_render_driver_null::NullInstance;
use prism_render_material::{
    ClosureKind, Illumination, MaterialRecord, fallback_material_record,
};

use prism_render_architecture::abi::GenerationalHandle;

use crate::material::MaterialParameterBuffers;

fn null_device() -> (prism_render_driver_null::NullDevice, impl prism_render_driver::RenderQueue)
{
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up")
}

/// A record whose packed block is wider than a plain dielectric's, so a test can
/// make a material change block size.
fn record_with_lobes(index: u32, lobes: u32) -> MaterialRecord {
    let handle = GenerationalHandle::new(index, 1);
    let mut record = fallback_material_record(handle, 1);
    record.closure_mask = lobes;
    record
}

#[test]
fn syncing_materials_places_their_blocks_and_writes_their_rows() {
    let (device, queue) = null_device();
    // Sized generously so the assertions below are about *placement*, not about
    // the growth path (which its own test covers).
    let mut buffers = MaterialParameterBuffers::new(&device, 4, 512, BufferUsages::NONE);
    assert_eq!(buffers.header_capacity(), 4);

    let first = record_with_lobes(0, 1);
    let second = record_with_lobes(1, 1);
    let first_words = first.packed_parameters().pack().len() as u32;
    let second_words = second.packed_parameters().pack().len() as u32;
    let report = buffers.sync(
        &device,
        &queue,
        [(0, first.clone()), (1, second.clone())],
        7,
    );

    assert_eq!(report.headers_written, 2, "both rows are new");
    assert_eq!(report.parameters_written, 2, "both blocks are new");
    assert!(!report.grew);
    assert_eq!(
        buffers.used_words(),
        first_words + second_words,
        "the heap covers both blocks"
    );

    let row = buffers.header(0).expect("row 0");
    assert_eq!(row.is_active, 1);
    assert_eq!(row.material_epoch_low, 7, "the epoch rides the row");
    assert_eq!(
        row.parameter_offset, 0,
        "the first block starts at the heap's origin"
    );
    assert_eq!(
        row.parameter_size as usize,
        first.packed_parameters().packed_size_bytes()
    );
    let words = buffers.parameters_of(0).expect("block 0");
    assert_eq!(
        words,
        first.packed_parameters().pack().as_slice(),
        "the words are the ABI's packed block"
    );

    let row = buffers.header(1).expect("row 1");
    assert_eq!(
        row.parameter_offset, first_words,
        "the second block follows the first"
    );
    assert_eq!(
        buffers.parameters_of(1).expect("block 1"),
        second.packed_parameters().pack().as_slice()
    );
}

/// A quiet frame writes nothing, and a re-sync of unchanged records is a no-op.
#[test]
fn an_unchanged_sync_writes_nothing() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 32, BufferUsages::NONE);
    let record = record_with_lobes(0, 1);
    buffers.sync(&device, &queue, [(0, record.clone())], 7);

    let report = buffers.sync(&device, &queue, [(0, record)], 7);

    assert_eq!(report.headers_written, 0);
    assert_eq!(report.parameters_written, 0);
    assert_eq!(report.relocated, 0);
}

/// A material that gains a lobe needs a wider block, so its old run goes back to
/// the heap and the new one is placed (and the row follows it).
#[test]
fn a_wider_block_relocates_and_frees_the_old_run() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 64, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);
    let narrow_words = buffers.used_words();
    assert_eq!(buffers.header(0).expect("row").parameter_offset, 0);

    // A second closure kind makes the derived lobe set larger, so the packed
    // block grows.
    let wider = record_with_lobes(0, 1 | (1 << ClosureKind::ClearCoat as u32));
    assert!(
        wider.packed_parameters().pack().len()
            > record_with_lobes(0, 1).packed_parameters().pack().len(),
        "the fixture really changes the block width"
    );
    let report = buffers.sync(&device, &queue, [(0, wider.clone())], 2);

    assert_eq!(report.relocated, 1, "the block moved");
    assert!(buffers.used_words() > narrow_words);
    assert_eq!(
        buffers.parameters_of(0).expect("block 0"),
        wider.packed_parameters().pack().as_slice()
    );
}

/// A material index past the current capacity grows the buffers, and the mirrors
/// refill the new allocation.
#[test]
fn a_higher_index_grows_and_refills() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 8, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);

    let report = buffers.sync(
        &device,
        &queue,
        [(0, record_with_lobes(0, 1)), (9, record_with_lobes(9, 1))],
        2,
    );

    assert!(report.grew, "index 9 does not fit in two rows");
    assert!(buffers.header_capacity() >= 10);
    assert_eq!(
        buffers.header(0).expect("row 0").is_active,
        1,
        "the mirror refilled the existing material"
    );
    assert_eq!(buffers.header(9).expect("row 9").is_active, 1);
    assert_eq!(
        buffers.header(5).expect("row 5").is_active,
        0,
        "rows nobody claimed stay explicitly inactive"
    );
}

/// Retiring a material frees its run and leaves a generation-tagged inactive
/// row, so a stale handle cannot read freed words.
#[test]
fn removing_a_material_frees_its_run_and_retires_its_row() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 64, BufferUsages::NONE);
    buffers.sync(
        &device,
        &queue,
        [(0, record_with_lobes(0, 1)), (1, record_with_lobes(1, 1))],
        1,
    );

    buffers.remove(&queue, 0, 1);

    let row = buffers.header(0).expect("row 0");
    assert_eq!(row.is_active, 0, "the row is retired");
    assert_eq!(row.generation, 1, "and tagged with the slot's generation");
    assert!(buffers.parameters_of(0).is_none(), "its run is gone");

    // Its words are reusable: the next material placed gets the freed run, which
    // is what "returned to the heap" means for a first-fit free list (the
    // high-water mark only falls when the *tail* is freed).
    let report = buffers.sync(&device, &queue, [(2, record_with_lobes(2, 1))], 2);
    assert_eq!(report.relocated, 0, "material 2 is new");
    assert_eq!(
        buffers.header(2).expect("row 2").parameter_offset,
        0,
        "the freed run at the origin was reused"
    );
    assert_eq!(report.relocated, 0, "material 1 keeps its own run");
}

/// The buffers are released together, and releasing twice is not an error.
#[test]
fn releasing_frees_both_buffers() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 2, 32, BufferUsages::NONE);
    buffers.sync(&device, &queue, [(0, record_with_lobes(0, 1))], 1);
    assert_eq!(device.live_resource_count(), 2, "headers + parameters");

    buffers.release(&device);
    assert_eq!(device.live_resource_count(), 0);
    buffers.release(&device);
    assert_eq!(device.live_resource_count(), 0, "releasing twice is a no-op");
}

#[test]
fn a_material_with_a_different_illumination_still_lands() {
    let (device, queue) = null_device();
    let mut buffers = MaterialParameterBuffers::new(&device, 1, 32, BufferUsages::NONE);
    let mut record = fallback_material_record(GenerationalHandle::new(0, 1), 1);
    record.illumination = Illumination::Stylized;

    buffers.sync(&device, &queue, [(0, record.clone())], 3);

    let row = buffers.header(0).expect("row");
    assert_eq!(row.illumination, Illumination::Stylized as u32);
    assert_eq!(
        buffers.parameters_of(0).expect("block"),
        record.packed_parameters().pack().as_slice(),
        "the packed block matches the record"
    );
}

