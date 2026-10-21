//! Actual compiler, file revision, and opt-in physical GPU replacement tests.
use prism_editor_domains::{reload::*, reload_shader::*};
use prism_render_driver::ShaderSource;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "prism-reload-shader-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, text: &str) {
        fs::write(self.0.join(name), text).unwrap();
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn ready<A>(mut request: ReloadRequest<A>) -> Result<PreparedReload<A>, ReloadError> {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = request.poll()? {
            return Ok(result);
        }
        assert!(Instant::now() < end, "reload worker did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn start(
    lane: &mut ReloadPipeline<ShaderReloadProvider>,
) -> Result<PreparedReload<CompiledReloadShader>, ReloadError> {
    ready(lane.start(Instant::now(), true)?.unwrap())
}
fn wgsl(files: &Files, limits: ReloadLimits) -> ReloadPipeline<ShaderReloadProvider> {
    ReloadPipeline::new(
        &files.0,
        ["shader.wgsl".into()],
        WGSL_SCHEMA.to_vec(),
        ShaderReloadProvider::wgsl("shader.wgsl".into()),
        limits,
    )
    .unwrap()
}
fn text(shader: &CompiledReloadShader) -> &str {
    match &shader.descriptor().source {
        ShaderSource::Wgsl(text) => text,
        _ => panic!("expected WGSL"),
    }
}
fn compute(value: u32) -> String {
    format!("@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn main() {{ output[0] = {value}u; }}")
}

#[test]
fn wgsl_real_validation_reflection_failures_and_host_swap() {
    let files = Files::new();
    files.write("shader.wgsl", &compute(1));
    let mut lane = wgsl(&files, ReloadLimits::default());
    let mut slot = lane.slot(String::from("last-good"));
    let prepared = start(&mut lane).unwrap();
    assert_eq!(
        prepared.artifact().entries(),
        &[ShaderEntry {
            name: "main".into(),
            stage: naga::ShaderStage::Compute,
            workgroup_size: [1, 1, 1]
        }]
    );
    assert_eq!(
        publish_shader(&lane, prepared, &mut slot, |shader| Ok(
            text(shader).to_owned()
        ))
        .unwrap(),
        "last-good"
    );
    let good = slot.get().clone();
    for invalid in [
        "not wgsl",
        "@compute @workgroup_size(1) fn main() { let x: u32 = true; }",
    ] {
        files.write("shader.wgsl", invalid);
        lane.notify("shader.wgsl", Instant::now()).unwrap();
        assert!(matches!(start(&mut lane), Err(ReloadError::Provider(_))));
        assert_eq!(slot.get(), &good);
    }
    files.write("shader.wgsl", &compute(2));
    lane.notify("shader.wgsl", Instant::now()).unwrap();
    let prepared = start(&mut lane).unwrap();
    assert_eq!(
        publish_shader(&lane, prepared, &mut slot, |_| Err(ReloadError::provider(
            "device layout rejected"
        ))),
        Err(ReloadError::provider("device layout rejected"))
    );
    assert_eq!(slot.get(), &good);
}

#[test]
fn wesl_links_registered_real_modules_and_dependency_edit_invalidates() {
    let files = Files::new();
    files.write(
        "main.wesl",
        "import package::helper::value; @compute @workgroup_size(1) fn main() { let x = value(); }",
    );
    files.write("helper.wesl", "fn value() -> u32 { return 7u; }");
    let modules = BTreeMap::from([
        ("package::main".into(), "main.wesl".into()),
        ("package::helper".into(), "helper.wesl".into()),
    ]);
    let mut lane = ReloadPipeline::new(
        &files.0,
        ["main.wesl".into(), "helper.wesl".into()],
        WESL_SCHEMA.to_vec(),
        ShaderReloadProvider::wesl("package::main".into(), modules),
        ReloadLimits::default(),
    )
    .unwrap();
    let mut slot = lane.slot(String::from("old"));
    let candidate = start(&mut lane).unwrap();
    assert!(!text(candidate.artifact()).contains("import "));
    assert!(text(candidate.artifact()).contains("7u"));
    files.write("helper.wesl", "fn value() -> u32 { return 9u; }");
    assert_eq!(
        publish_shader(&lane, candidate, &mut slot, |shader| Ok(
            text(shader).to_owned()
        )),
        Err(ReloadError::Stale)
    );
    lane.notify("helper.wesl", Instant::now()).unwrap();
    let candidate = start(&mut lane).unwrap();
    publish_shader(&lane, candidate, &mut slot, |shader| {
        Ok(text(shader).to_owned())
    })
    .unwrap();
    assert!(slot.get().contains("9u"));
    files.write("main.wesl", "import package::outside::value; @compute @workgroup_size(1) fn main() { let x = value(); }");
    lane.notify("main.wesl", Instant::now()).unwrap();
    assert!(matches!(start(&mut lane), Err(ReloadError::Provider(_))));
    assert!(slot.get().contains("9u"));
}

#[test]
fn shader_schema_and_output_budgets_are_checked_before_publication() {
    let files = Files::new();
    files.write("shader.wgsl", &compute(1));
    let mut lane = wgsl(
        &files,
        ReloadLimits {
            output_bytes: 8,
            ..ReloadLimits::default()
        },
    );
    assert!(matches!(start(&mut lane), Err(ReloadError::Budget(_))));
    let mut lane = wgsl(&files, ReloadLimits::default());
    lane.set_schema(WESL_SCHEMA.to_vec(), Instant::now())
        .unwrap();
    assert!(matches!(start(&mut lane), Err(ReloadError::Provider(_))));
}

fn gpu_build(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &CompiledReloadShader,
) -> Result<wgpu::ComputePipeline, ReloadError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("real reloaded shader"),
        source: wgpu::ShaderSource::Wgsl(text(shader).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("real reload pipeline"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = futures_lite::future::block_on(scope.pop()) {
        return Err(ReloadError::provider(error));
    }
    Ok(pipeline)
}
fn gpu_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    layout: &wgpu::BindGroupLayout,
) -> u32 {
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let bytes = readback.slice(..).get_mapped_range().unwrap();
    let value = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    drop(bytes);
    readback.unmap();
    value
}

#[test]
#[ignore = "requires a physical GPU; run explicitly with --ignored --nocapture"]
fn physical_gpu_dispatch_changes_and_layout_failure_preserves_last_good() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = futures_lite::future::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .expect("physical GPU required; no skip");
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    eprintln!("physical reload adapter: {:?}", adapter.get_info());
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .unwrap();
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let files = Files::new();
    files.write("shader.wgsl", &compute(1));
    let mut lane = wgsl(&files, ReloadLimits::default());
    let initial = start(&mut lane).unwrap();
    let mut slot = lane.slot(gpu_build(&device, &layout, initial.artifact()).unwrap());
    drop(initial);
    assert_eq!(gpu_read(&device, &queue, slot.get(), &bind_layout), 1);
    files.write("shader.wgsl", &compute(2));
    lane.notify("shader.wgsl", Instant::now()).unwrap();
    let candidate = start(&mut lane).unwrap();
    let old = publish_shader(&lane, candidate, &mut slot, |shader| {
        gpu_build(&device, &layout, shader)
    })
    .unwrap();
    drop(old);
    assert_eq!(gpu_read(&device, &queue, slot.get(), &bind_layout), 2);
    // Valid WGSL and Naga module; actual device pipeline layout rejects binding 1.
    files.write(
        "shader.wgsl",
        &compute(3).replace("@binding(0)", "@binding(1)"),
    );
    lane.notify("shader.wgsl", Instant::now()).unwrap();
    let candidate = start(&mut lane).unwrap();
    assert!(
        publish_shader(&lane, candidate, &mut slot, |shader| gpu_build(
            &device, &layout, shader
        ))
        .is_err()
    );
    assert_eq!(gpu_read(&device, &queue, slot.get(), &bind_layout), 2);
    files.write("shader.wgsl", "invalid wgsl");
    lane.notify("shader.wgsl", Instant::now()).unwrap();
    assert!(start(&mut lane).is_err());
    assert_eq!(gpu_read(&device, &queue, slot.get(), &bind_layout), 2);
}
