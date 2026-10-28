# Prism Decal SceneFabric 写集实现

本文记录当前 Decal vertical slice 的真实写集边界与验证结果。本文只描述 Decal；统一 `fabric_runtime`、Particle、Hair 和全局 plugin 不属于本写集。

## 目标与边界

Decal 写集把一个稳定的 `DomainFrameIdentity` 降低为一个独立的 SceneFabric publication。publication 同时携带：

- 当前 frame、extent、dispatch workgroup 和 `RenderObjectId::Decal` identity；
- 已通过 Fabric semantic validation 的 projector、material、receiver 描述；
- 六个独立的、真实创建并写入 ABI 数据的 storage buffer lease；
- 一个经 `DomainRecordReadyView` 认证的 producer-to-consumer 资源视图。

缺少任何一项时，consumer 不会创建替代 buffer，也不会用静态 ledger、表内 ID 或 no-op dispatch 伪造工作。

## 六类 ABI buffer

每个 Decal object 都独立拥有以下六类 buffer，顺序由 Decal resource contract 固定，bind group 与 WESL binding 采用同一顺序：

| Fabric role | ABI 内容 | consumer 用途 |
| --- | --- | --- |
| `DECAL_INSTANCE_RESOURCE_ROLE` | `DecalInstance` | projector transform、projection direction、fade、material index |
| `DECAL_MATERIAL_RESOURCE_ROLE` | `DecalMaterial` | albedo、normal、scalar blend |
| `DECAL_RECEIVER_RESOURCE_ROLE` | `DecalReceiver` | G-buffer receiver position/normal/material |
| `DECAL_CLUSTER_RANGE_RESOURCE_ROLE` | `DecalClusterRange` | 当前 receiver 的 candidate range |
| `DECAL_CLUSTER_INDEX_RESOURCE_ROLE` | `u32` | range 指向的 projector instance index |
| `DECAL_RESOLVE_RESOURCE_ROLE` | `DecalResolvedReceiver` | resolve 输出及 miss/back-face 保留值 |

producer 按 contract 顺序创建并写入这些 buffer，再通过 `DomainGpuBinding` 和 `DomainRecordReadyView` 交给 consumer。`create_decal_bind_group` 只接受已认证的 record-ready buffer，并按 role 查找 binding。

## 多对象 fan-out

`DecalRenderWorldOwner` 保存按 object 排序的对象快照。object handle、history key 和 motion source 在 owner registration 时固定；更新 projector/material/receiver 不改变 temporal identity，remove 后才能重新注册新的 history stream。

一个 source frame 的多个 owner object 会分别生成 publication，但由 `prepare_decal_resolves` 组合为一次 SceneFabric request：

```text
owner/object A -> six buffers -> DomainRecordReadyView -> bind group A -> dispatch A
owner/object B -> six buffers -> DomainRecordReadyView -> bind group B -> dispatch B
                                      \              one DECAL_RESOLVE_PASS
```

因此 fan-out 保留 object/history identity、六组物理 buffer 和 dispatch 边界，同时只生成一个 SceneFabric graph/pass。任何对象的语义 item、frame、extent 或 record-ready view 不匹配时，整次对象 publication 不进入 recording。

## 生命周期

- 错帧：`DecalPreparedResources::take_for` 先按 identity 找 publication，再强制检查 frame；错帧返回错误并不消费资源。
- 移除：owner 快照在 extraction 中整体替换；对象移除后下一帧不再产生 semantic items 或 physical publication，空 owner 会清空 mailbox，避免旧对象重放。
- resize：extent 进入 publication 和 SceneFabric request；新的 extent 使用新的六组物理 buffer generation，不复用仍可能在 GPU 上使用的旧 generation。
- completion：recording 后将同一对象批次的 `RecordedGraph`、六类 buffer lease 和 publication 放入同一个 in-flight entry；只有 host queue 的 completion value 到达后才 retire。recording、submit 或 callback 注册本身都不视为完成。

## 验证

当前 Decal 定向命令：

```text
cargo test -p prism_render_scene --features legacy-bevy --lib decal --no-fail-fast
```

该命令覆盖 CPU clustered reference、semantic item count、WESL standalone compile、单对象 GPU/CPU parity、多对象 GPU/CPU parity、六 buffer non-alias、错帧、remove/re-register identity 和 resize generation 生命周期。当前结果为 10 passed、0 failed。

GPU parity 测试只接受硬件 adapter；设置 `PRISM_REQUIRE_GPU=1` 时，无硬件 adapter 或软件 adapter 会直接失败而不是静默伪造 GPU 证据。
