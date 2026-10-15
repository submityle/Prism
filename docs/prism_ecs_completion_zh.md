# ECS 系统完成与验收记录

本记录对应 `prism_ecs_design_zh.md` 全部设计要求。实现、跨 crate 接线、设备验证与性能验收分别记录；单元测试通过不代替整个系统完成。

## 2026-10-15 实施批次（历史基线）

| 要求 | 当前实施内容 | 完成所需证据 |
|---|---|---|
| §7 ParamSet | 补齐顺序借用、访问并集和参数人体工学 | 冲突初始化、编译失败借用测试、并行调度与延期命令测试通过 |
| §9 命令合并 | 同实体连续结构命令归并，闭包和可观察生命周期为语义屏障 | 最终数据与顺序回放一致，实际原型迁移次数下降，hook/observer 顺序不变 |
| §10 World 反应图 | 组件观察者自动标脏，统一收集后拓扑执行；脏节点队列、panic 后可重试 | 菱形依赖、重复写、无变化、稀疏组件、回调写入延后、跨 World 拒绝测试；UI 接线另验 |
| §16.2 动态组件热重载 | 动态注册已有，布局升级和活数据迁移需补齐 | 迁移成功保值、失败不破坏原值、布局/析构/快照一致性测试 |
| §16.3 字段级 prefab | 补齐字段继承与覆盖状态 | 父值更新、字段覆盖/撤销、祖先链与环检测、反射序列化往返 |
| §16.6 / M6 编辑器工具 | 内核诊断已有，UI/远程消费需补齐 | 实际选择实体、反射读写、撤销与时间旅行、profiler 消费集成测试 |
| §15 GPU 常驻列 | CPU dirty-span/column extraction、render-side resident-column readback/recovery 入口已有；physical-GPU evidence gate 与 virtual-texture benchmark harness 已有 | 真实 physical adapter 运行、GPU 映射/同步生命周期、上传收益与性能仍待验证；缺设备必须保留 `SKIPPED`/strict failure |
| §17 NUMA | VM/大页、OS topology/affinity readback、worker binding outcome 与 arena fallback acceptance gate 已有 | `numa_acceptance` 默认缺能力 `SKIPPED`、`PRISM_REQUIRE_NUMA=1` 严格失败；Linux strict success、ECS local-page proof 与 NUMA 性能仍待验证 |
| §18 / M6 迁移 | UI ECS 已使用 Prism；`prism_audio_ecs` 已使用原生 Prism ECS，`prism_audio_bevy` 仅是兼容 facade；已验证 30 个 unit tests、22 个 integration tests、4 个 facade integration tests | 依赖闭包检查与行为对拍；完整应用、旧场景资产及编辑工作流迁移验收；场景与音频联合工作流对拍；音频资产导入/解码、播放头绑定和场景生命周期接线；公共 API 变动同步迁移调用方 |
| §23.5 渲染提取 | 独立 render app 与跨线程流水线已有 | 保持现有提取/线程所有权测试，验证 ECS 常驻列真实渲染消费 |
| §24.1 tasks/time | lane 优先级和预算执行已有 | App 接入 time 帧预算；关键任务必达、后台顺延、确定性固定序测试 |
| §24.2 Transform | 补齐空间传播到 ECS 观察总线 | 重复变更合并、稳定实体序、空间增量更新、删除和空闲帧行为 |
| §24.3 reflect | 动态访问/字段 patch/存档/schema 迁移已有 | 结合动态热重载与 prefab 覆盖的集成验证 |
| §24.4 确定性 | 四环单场景夹具已有 | 扩充结构变更、回滚、关系、分区与多线程组合；跨平台定点证据 |
| §13 / M5 大世界 | 分区、LOD、休眠与浮动原点已有 | 持续流入流出、跨 cell 弱引用、恢复与内存压力长测 |
| §21 性能回归门禁 | 有规模测试和测量输出 | 可执行回归阻断、变化量复杂度门禁、独占机多核曲线 |

## 验收约束

所有实现保持核心 `no_std + alloc`；可选功能走 feature。保持现有用户改动，不把并行资产/地形开发混入 ECS 提交。每批实现完成后运行对应测试并记录实际结果，再提交与更新设计文档。独占机、异构平台和 NUMA 硬件证据不足时继续保留未验收状态。

## 2026-10-16 基于当前 `main` 历史的同步

当前基线为 `main @ 6b9964234`。本次只同步本文档与 `docs/prism_ecs_design_zh.md`，未修改源码、未重跑测试；下表是基于提交历史与现有验收入口的边界，不把历史入口写成当前已通过。

依据：UDP/ACK/rebind：`39bd3806a`、`0bf59f55a`、`d6000586b`、`c7201323e`；NUMA acceptance：`9390b7a5e`、`07a3d5b9c`；audio/source-route 与 event-log：`ea2ed8026`、`cd03b14c7`、`3e503dd38`；GPU resident/wgpu gate 与 benchmark：`22aebe26`、`a288cf12e`、`c7201323e`、`3a8120874`、`0e7e67259`；CPU SDF planner：`7e99fa30f`、`90ff7a89a`。

| 范围 | 已实现 | 已验证或已有验收入口 | 仍未验证/不可宣称 |
|---|---|---|---|
| UDP / ACK / rebind | 有限可靠 UDP、ACK 重传、候选端点与稳定 peer rebind | 真实 loopback fault-proxy 覆盖首个 data/ACK 丢包、顺序/exactly-once、旧 socket 关闭后的 rebind | 跨进程/跨机、公网、NAT/ICE/relay、生产迁移与本轮重跑结果 |
| NUMA / affinity | OS topology/affinity readback、worker outcome、arena fallback 语义；策略层拓扑模型 | `pkg/prism_tasks/tests/numa_acceptance.rs`；默认 `SKIPPED`，`PRISM_REQUIRE_NUMA=1` 严格失败；`Bound`/`Fallback` placement 有明确断言 | Linux strict pass、ECS arena node-local pages、拓扑感知调度收益与 NUMA 性能 |
| audio / event-log | audio source-route/scene adapter/full-chain；bounded memory/file `EventLogBackend` 与 replay bridge | 既有局部 audio unit/integration/full-chain 与 event-log provider/replay 入口 | 真实设备回调、完整应用/旧资产/editor 联合流程、durable plugin recovery、跨进程/跨机持久化 |
| GPU resident / benchmark / SDF | CPU resident-column extraction；physical-GPU gate；virtual-texture benchmark harness（`3a8120874`、`0e7e67259`）；CPU deterministic SDF planner | resident-column readback、wgpu hardware-evidence、benchmark 入口存在；无设备明确 skip/strict fail | 当前真实 adapter 通过、GPU mapping/sync/upload benefit、性能数字、GPU SDF sampling/ECS 集成 |

网络边界：ECS structural-equality 测试的 `in-process transport` 只证明单进程测试 World 的限定行为；真实 loopback UDP 验收是独立入口，不能互相替代。任何跨机/NAT/真实 GPU/性能完成结论都仍需对应环境中的实际证据。
