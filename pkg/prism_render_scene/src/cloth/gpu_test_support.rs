//! 共享的 cloth 真机 GPU 测试脚手架：把 `WESL` 源编译回 `Wgsl`、尽力取一个原生
//! compute 设备、以及裸 `wgpu` 缓冲小工具。
//!
//! 多个 parity 模块（body-collision / backstop 共用 `cloth_collision.wesl`，
//! skin-embed 用 `cloth_embed.wesl`）都要做同一批与内核无关的准备：编译着色器、
//! 在编译产物里定位真实入口符号、取设备、建只读存储缓冲。这些集中在此，避免每个
//! 测试文件各抄一份。内核相关的绑定布局、replay 与断言仍留在各自模块里，保持每个
//! 内核的对拍逻辑自解释。
//!
//! 全部 `#[cfg(test)]`：这是测试专用支撑，不进 shipping 二进制。

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BufferUsages, DeviceDescriptor, Instance, InstanceDescriptor,
    InstanceFlags, RequestAdapterOptions,
};

/// `GPU`-对-`CPU` 逐分量绝对容差，供各 parity 模块共用。
///
/// 这些内核都是单 pass 纯几何，CPU 与 GPU 两条路径跑同一份 `float32` 算术，唯一自由度
/// 是归一化里 CPU 的 `1.0 / sqrt` 与 WESL 的 `inverseSqrt`（多为原生 `rsqrt`）之间的
/// 几个 ULP 之差。位置量级 `O(1)`，`1e-4` 既能吸收该 ULP 差，又远紧于任何真实内核 bug
/// 会产生的 `O(0.1)` 级发散。
pub(super) const PARITY_EPS: f32 = 1.0e-4;

/// 编译 `cloth_collision.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_COLLISION_WESL_UUID: u128 = 0x434c_4f54_485f_434f_4c4c_4244_5f42_4f01;

/// 编译 `cloth_embed.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_EMBED_WESL_UUID: u128 = 0x434c_4f54_485f_454d_4245_445f_5f5f_4501;

/// 编译 `cloth_self_collision_virtual.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_VIRTUAL_WESL_UUID: u128 = 0x434c_4f54_485f_5654_5f5f_5f5f_5f5f_5601;

/// 编译 `cloth_self_ccd.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_SELF_CCD_WESL_UUID: u128 = 0x434c_4f54_485f_5343_4344_5f5f_5f5f_5701;

/// 编译 `cloth_pressure.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_PRESSURE_WESL_UUID: u128 = 0x434c_4f54_485f_5052_4553_5f5f_5f5f_5801;

/// 编译 `cloth_plasticity.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_PLASTICITY_WESL_UUID: u128 = 0x434c_4f54_485f_504c_4153_5f5f_5f5f_5901;

/// 编译 `cloth_ccd.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_CCD_WESL_UUID: u128 = 0x434c_4f54_485f_4343_445f_5f5f_5f5f_5a01;

/// 编译 `cloth_vbd.wesl` 用的一次性 `AssetId`，只需在本次编译内唯一。
const CLOTH_VBD_WESL_UUID: u128 = 0x434c_4f54_485f_5642_445f_5f5f_5f5f_5b01;

/// 把 `WESL` 源经 render-world 的 [`ShaderCache`] 编译回 `Wgsl` 字符串（不建
/// 设备），供各模块自建的裸 `wgpu` 设备使用。镜像 `sim_gpu_tests` 的编译闭包。
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("cloth shaders are WESL"),
    }
}

/// 经 `ShaderCache` 把一份嵌入式 cloth `WESL` 源编译成 `Wgsl`。`uuid` 只需在本次编译内
/// 唯一，`path` 仅用于错误信息里的定位。
fn compile_cloth_wgsl(source: &'static str, path: &'static str, uuid: u128) -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(uuid),
    };
    cache.set_shader(id, Shader::from_wesl(source, path));
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{path} failed to compile: {error}"));
    (*module).clone()
}

/// 经 `ShaderCache` 把嵌入式 `cloth_collision.wesl` 编译成 `Wgsl`。
pub(super) fn compile_collision_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_collision.wesl"),
        "shaders/cloth_collision.wesl",
        CLOTH_COLLISION_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_embed.wesl` 编译成 `Wgsl`。
pub(super) fn compile_embed_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_embed.wesl"),
        "shaders/cloth_embed.wesl",
        CLOTH_EMBED_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_self_collision_virtual.wesl` 编译成 `Wgsl`。
pub(super) fn compile_virtual_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_self_collision_virtual.wesl"),
        "shaders/cloth_self_collision_virtual.wesl",
        CLOTH_VIRTUAL_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_self_ccd.wesl` 编译成 `Wgsl`。
pub(super) fn compile_self_ccd_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_self_ccd.wesl"),
        "shaders/cloth_self_ccd.wesl",
        CLOTH_SELF_CCD_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_pressure.wesl` 编译成 `Wgsl`。
pub(super) fn compile_pressure_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_pressure.wesl"),
        "shaders/cloth_pressure.wesl",
        CLOTH_PRESSURE_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_plasticity.wesl` 编译成 `Wgsl`。
pub(super) fn compile_plasticity_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_plasticity.wesl"),
        "shaders/cloth_plasticity.wesl",
        CLOTH_PLASTICITY_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_ccd.wesl` 编译成 `Wgsl`。
pub(super) fn compile_ccd_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_ccd.wesl"),
        "shaders/cloth_ccd.wesl",
        CLOTH_CCD_WESL_UUID,
    )
}

/// 经 `ShaderCache` 把嵌入式 `cloth_vbd.wesl` 编译成 `Wgsl`。
pub(super) fn compile_vbd_wgsl() -> String {
    compile_cloth_wgsl(
        include_str!("../shaders/cloth_vbd.wesl"),
        "shaders/cloth_vbd.wesl",
        CLOTH_VBD_WESL_UUID,
    )
}

/// 在编译后的 `Wgsl` 里按子串定位 compute 入口的真实符号名（`WESL` 可能给模块内
/// 名字加前缀，故按子串而非固定符号查找）。
pub(super) fn find_entry_point(wgsl: &str, needle: &str) -> String {
    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains(needle) {
                return name.to_string();
            }
        }
    }
    panic!("no compute entry point containing `{needle}` in compiled Wgsl");
}

/// 尽力获取一个原生 compute 设备与队列。
///
/// 这些 parity 内核都不用 `immediate`（push-constant），故只需一个默认能力的 compute
/// 设备——比求解核宽松，能在更多机器上跑满。无 adapter 时返回 `None`（不 panic），让
/// 无头机保持绿。
pub(super) fn try_compute_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default())).ok()?;
    Some((device, queue))
}

/// 从一份 `Pod` 切片建只读存储缓冲；空输入回退到一个零元素，避免 runtime-sized
/// array 绑定非法（`0` 字节绑定被拒）。
pub(super) fn storage_from_slice<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    fallback: T,
) -> wgpu::Buffer {
    if data.is_empty() {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&fallback),
            usage: BufferUsages::STORAGE,
        })
    } else {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE,
        })
    }
}
