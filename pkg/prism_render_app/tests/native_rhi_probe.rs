//! Capability-gated native RHI timestamp evidence.
//!
//! The strict path deliberately records a timestamp through Prism's RHI:
//! query-set creation, a timestamped compute pass, RHI query resolve, queue
//! readback, and a real device submission. CPU wall time is never substituted
//! for GPU timestamp data.
#![forbid(unsafe_code)]

use prism_render_driver::{
    Adapter, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferReadbackRange,
    BufferUsages, CommandEncoder, ComputeCommand, DeviceDescriptor, Features, Instance,
    PipelineLayoutDescriptor, RenderDevice, RenderQueue, ShaderStages, TimestampQuerySetDescriptor,
    TimestampWrites,
};
use prism_render_driver_wgpu::{AdapterEvidence, WgpuAdapter, WgpuInstance};

fn require_hardware_gpu() -> bool {
    matches!(std::env::var("PRISM_REQUIRE_GPU"), Ok(value) if value == "1")
}

fn run_timestamp_probe(
    instance: &WgpuInstance,
    adapter: &WgpuAdapter,
) -> Result<(u64, u64, f32), String> {
    let descriptor = DeviceDescriptor {
        label: Some("native-rhi-timestamp-probe".into()),
        required_features: Features::TIMESTAMP_QUERY,
        ..DeviceDescriptor::default()
    };
    let (device, queue) = <WgpuInstance as Instance>::create_device(instance, adapter, &descriptor)
        .map_err(|error| format!("RHI device creation failed: {error}"))?;
    let period_ns = queue
        .timestamp_period_ns()
        .filter(|period| period.is_finite() && *period > 0.0)
        .ok_or_else(|| "RHI queue did not expose a valid timestamp period".to_owned())?;

    let data = device.create_buffer(&BufferDescriptor {
        label: Some("native-rhi-timestamp-work".into()),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(data, 0, &1_u32.to_ne_bytes());

    let shader = device.create_shader_module(&prism_render_driver::ShaderModuleDescriptor::wgsl(
        "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
         @compute @workgroup_size(1)\n\
         fn main() { data[0] = data[0] + 1u; }\n",
    ));
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("native-rhi-timestamp-layout".into()),
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("native-rhi-timestamp-pipeline-layout".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline =
        device.create_compute_pipeline(&prism_render_driver::ComputePipelineDescriptor {
            label: Some("native-rhi-timestamp-pipeline".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "main".into(),
        });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("native-rhi-timestamp-bind-group".into()),
        layout: Some(layout),
        entries: vec![BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: data,
                offset: 0,
                size: Some(4),
            },
        }],
    });

    let queries = device.create_timestamp_query_set(&TimestampQuerySetDescriptor {
        label: Some("native-rhi-timestamps".into()),
        count: 2,
    });
    let resolved = device.create_buffer(&BufferDescriptor {
        label: Some("native-rhi-timestamp-resolve".into()),
        size: 16,
        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = CommandEncoder::labeled("native-rhi-timestamp-command-buffer");
    encoder.push_timestamped_compute_pass(
        Some("native-rhi-timestamp-pass".into()),
        TimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        },
        vec![
            ComputeCommand::SetPipeline(pipeline),
            ComputeCommand::SetBindGroup {
                index: 0,
                bind_group,
                dynamic_offsets: Vec::new(),
            },
            ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
        ],
    );
    encoder.push_transfer_pass(
        Some("native-rhi-timestamp-resolve".into()),
        vec![prism_render_driver::CopyCommand::ResolveTimestampQueries {
            query_set: queries,
            first_query: 0,
            query_count: 2,
            destination: resolved,
            destination_offset: 0,
        }],
    );
    queue.submit(&[encoder.finish()]);
    if !device.poll_wait() {
        return Err("RHI device lost while waiting for timestamp submission".into());
    }

    let mut readback = queue
        .request_buffer_readback(
            &[BufferReadbackRange {
                buffer: resolved,
                offset: 0,
                size: 16,
            }],
            16,
        )
        .map_err(|error| format!("timestamp resolve readback request failed: {error:?}"))?;
    let bytes = loop {
        if let Some(mut ranges) = readback
            .try_complete()
            .map_err(|error| format!("timestamp readback failed: {error:?}"))?
        {
            break ranges
                .pop()
                .ok_or_else(|| "timestamp readback returned no range".to_owned())?;
        }
        if !device.poll_wait() {
            return Err("RHI device lost while mapping timestamp readback".into());
        }
    };
    let mut chunks = bytes.chunks_exact(8);
    let begin = u64::from_ne_bytes(
        chunks
            .next()
            .ok_or_else(|| "timestamp readback is shorter than two queries".to_owned())?
            .try_into()
            .expect("eight-byte timestamp chunk"),
    );
    let end = u64::from_ne_bytes(
        chunks
            .next()
            .ok_or_else(|| "timestamp readback is shorter than two queries".to_owned())?
            .try_into()
            .expect("eight-byte timestamp chunk"),
    );
    if begin == 0 || end <= begin {
        return Err(format!(
            "invalid timestamp pair returned by RHI: begin={begin} end={end}"
        ));
    }

    device.destroy_buffer(resolved);
    device.destroy_timestamp_query_set(queries);
    device.destroy_bind_group(bind_group);
    device.destroy_compute_pipeline(pipeline);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_bind_group_layout(layout);
    device.destroy_shader_module(shader);
    device.destroy_buffer(data);
    Ok((begin, end, period_ns))
}

#[test]
fn cross_adapter_timestamp_probe_is_strict_or_explicitly_skipped() {
    let instance = WgpuInstance::new();
    let adapters = instance
        .enumerate_adapters()
        .into_iter()
        .map(WgpuAdapter::new)
        .collect::<Vec<_>>();
    if adapters.is_empty() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1: no adapter was available for the RHI probe"
        );
        eprintln!("SKIPPED native RHI probe: no adapter");
        return;
    }

    let mut hardware_adapters = 0;
    let mut timestamp_capable_hardware = 0;
    let mut actual_timestamp_evidence = 0;
    for adapter in adapters {
        let evidence = AdapterEvidence::from_info(adapter.info());
        if !evidence.is_hardware() {
            eprintln!("SKIPPED native RHI probe: non-hardware adapter: {evidence}");
            continue;
        }
        hardware_adapters += 1;

        if !adapter
            .capabilities()
            .features
            .contains(Features::TIMESTAMP_QUERY)
        {
            eprintln!(
                "SKIPPED native RHI timestamp probe: adapter lacks TIMESTAMP_QUERY: {evidence}"
            );
            continue;
        }
        timestamp_capable_hardware += 1;

        match run_timestamp_probe(&instance, &adapter) {
            Ok((begin, end, period_ns)) => {
                actual_timestamp_evidence += 1;
                let gpu_ms = (end - begin) as f64 * f64::from(period_ns) / 1_000_000.0;
                eprintln!(
                    "REAL RHI GPU TIMESTAMP evidence: {evidence}; begin={begin} end={end} period_ns={period_ns} elapsed_ms={gpu_ms:.6}"
                );
            }
            Err(error) => {
                eprintln!("FAILED native RHI timestamp probe: {evidence}: {error}");
            }
        }
    }

    if require_hardware_gpu() {
        assert!(
            hardware_adapters > 0,
            "PRISM_REQUIRE_GPU=1: no physical GPU adapter was found"
        );
        assert!(
            timestamp_capable_hardware > 0,
            "PRISM_REQUIRE_GPU=1: no physical GPU adapter advertises TIMESTAMP_QUERY"
        );
        assert!(
            actual_timestamp_evidence > 0,
            "PRISM_REQUIRE_GPU=1: no physical GPU completed the RHI timestamp query path"
        );
    } else if actual_timestamp_evidence == 0 {
        eprintln!(
            "SKIPPED native RHI timestamp probe: no timestamp-capable hardware completed a real RHI submission"
        );
    }
}
