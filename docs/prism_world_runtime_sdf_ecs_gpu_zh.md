# Prism World Runtime：ECS SDF GPU Sampling

本文记录 `prism_world_runtime` 中 ECS 请求到真实 GPU SDF sampling 的已实现契约。该能力通过 `sdf_ecs_gpu` feature 启用，不修改 `prism_ecs` 的源码，也不提供 CPU fallback。

## 已实现路径

生产调用链如下：

```text
WorldSdfEcsRequest component
    -> Entity 排序、容量与 mip 校验
    -> GpuSdfQuery 上传到 GPU query buffer
    -> SDF clipmap lookup compute dispatch
    -> 16-byte result buffer
    -> staging buffer map/readback
    -> ABI 校验与解码
    -> WorldSdfEcsSample component
```

`WorldSdfEcsGpuBridge::sample_world` 只有在整个 readback 和 ABI 解码成功后才写入结果组件。结果写入前如果实体已经不可用，会返回 `EntityUnavailable`，不会把结果写入另一个实体。

请求实体按 `Entity` 排序后提交，因此 GPU 结果和 ECS 实体之间的对应关系是确定的。`request_id` 是调用方的版本/关联字段，会原样复制到结果；它不会自动去重、合并或丢弃旧请求。

## ECS 组件

```rust
WorldSdfEcsRequest {
    voxel: [i32; 3],
    mip: u8,
    request_id: u64,
}
```

结果为：

```rust
WorldSdfEcsSample {
    request_id: u64,
    value: WorldSdfEcsSampleValue,
}
```

`WorldSdfEcsSampleValue::Value(i16)` 表示请求的 brick 已驻留并返回量化距离；`Unknown` 表示 brick 不在当前可查询的驻留窗口中。`Unknown` 不等价于空空间，也不等价于距离值零。

## GPU ABI

每条结果记录固定 16 字节，按 little-endian 解码：

| 字节范围 | 字段 | 合法值 |
| --- | --- | --- |
| `0..4` | `status: u32` | `0` 或 `1` |
| `4..8` | `value: i32` | `status=1` 时必须在 `i16` 范围内 |
| `8..16` | reserved | 必须全为零 |

`status=0,value=0` 解码为 `Unknown`；`status=1` 解码为 `Value(i16)`。其余组合、越界值、非零 reserved 字段、错误记录长度都会使整个 batch 失败，结果组件不会部分发布。

## Focus、dirty 与数据权威

`WorldSdfGpuFocus` 负责把主机提供的世界毫米坐标映射到当前 clipmap ring。首次 `update_focus` 会从 `SparseSdfBrickGrid` 上传进入窗口的 brick；focus 保留时可通过 `mark_dirty` 标记已修改的 resident brick，再用 `flush_dirty` 在预算内上传最新 CPU 数据。

存在 pending dirty 数据时，查询会 fail closed：`sample_world` 在任何 query 上传或 dispatch 前返回 `PendingDirty`。调用方必须完成 `flush_dirty`，然后才可继续采样。

当前 CPU `SparseSdfBrickGrid` 仍是权威数据源；GPU ring 是查询镜像。只支持当前 focus/clipmap ring 的查询，窗口外或缺失 brick 返回 `Unknown`。

## GPU 可用性策略

`WorldSdfEcsGpuBridge::from_headless` 接受显式策略：

- `SkipUnavailable`：没有物理 GPU、适配器不存在或适配器仅为软件/非硬件证据时返回 `Ok(None)`。
- `Strict`：上述情况返回 `NoPhysicalGpu`。
- 设备创建失败、能力不足和运行时上传/dispatch 错误不会被静默转换为 skip，会返回明确错误。

真实硬件测试必须检查 `GpuContext::evidence().is_hardware()`。当前集成测试在物理 GPU 上验证了 Metal Apple M2 的真实 dispatch/readback；软件或 mock backend 不能作为 GPU 完成证据。

## 当前边界

- readback 是同步 map/poll 路径，单个 batch 最长等待 30 秒。
- 每次 batch 使用固定 `max_queries` 和上传字节预算。
- 不提供 CPU fallback；无硬件时由调用方决定 skip 或 strict fail。
- 结果组件在完整 batch 成功后统一发布，不提供部分成功结果。
- 当前 focus/clipmap 仍是单 Y slice 的既有 runtime 几何契约。
- `request_id` 不承担去重、排序或取消语义。

## 验证

CPU 单元测试覆盖请求排序、容量/mip 校验、ABI 极值与非法记录、配置校验。真实 GPU 集成测试覆盖：

- ECS 请求到 compute dispatch、readback、结果组件；
- `Unknown`、多 mip、`i16::MIN`；
- dirty mark/flush 前后的 fail-closed 行为；
- 空 batch、非法 mip；
- ECS entity generation 防止旧实体结果复用；
- 物理 GPU evidence 与 skip/strict 策略。
