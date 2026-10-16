# Prism `pkg/` 各模块 对照设计文档 详细完成度审计

> 审计对象：`pkg/` 下 141 个目录 / **140 个真实 crate**（`pkg/prism_bevy/` 为空目录）；对照对象：`docs/` 下 117 篇设计与实现规范（5.4 MB）。
> 结论口径：**静态证据**（读码 + grep），未运行 `cargo build` / `cargo test`；所有百分比均由计数的「要求单元」推出，分母随表给出。
> 审计时间：2025-10（基线 `main @ c3e0de176`）。

---

## 0. 口径与方法

本审计采用与仓库既有审计（`docs/prism_ecs_completion_zh.md`、`docs/refactoring-dev/` 六篇）一致的双轨口径：

| 轨道 | 定义 | 为什么必须分开 |
|---|---|---|
| **骨架/类型层** | 文档要求的数据结构、trait、模块、函数签名在代码中存在 | 本仓多数模块类型层近乎完备，只看这一轨会得出「接近完成」的错误结论 |
| **端到端接通层** | 该能力有**非测试**消费者（被 `pkg/*/Cargo.toml` 依赖，或被 `pkg/` 之外的 manifest 依赖/调用），且文档要求的证据（验收/门禁）存在 | 本仓最普遍的失败形态是「类型齐备但没有生产调用者」 |

> 关于「消费者」的实测边界：仓库根 `Cargo.toml` 仍是上游 Bevy 门面包（`name = "bevy"`、`0.20.0-dev`），它对 `pkg/` 的依赖数为 **0**；`src/`、`examples/`、`tests-integration/` 属于该老栈，**并不消费 pkg crate**。`pkg/` 之外真正的消费者只有 4 个 manifest（`benches/`、两个 `os-backends/prism_render_driver_*`、`tools/render_triangle`），合计只拉到 8 个 crate。因此本审计另做了一份**客观机器判定**的全仓反向依赖图（§3.1、§3.2、附录 D）：**140 个 crate 中 59 个（42%）在全仓没有任何消费者**（该计数已把 dev-dependency 计为消费者）。

每个「要求单元」的判定分四档：**落地**（有生产调用者）/ **骨架**（类型或模块存在，无生产调用者）/ **部分**（子情形缺失）/ **未落地**（文档要求但代码没有）。

**明确不宣称的事项**：单元测试通过不等于系统完成；本审计未执行任何测试、未在真实 GPU / NUMA / 音频设备上运行，因此「已验收」一律不采信文档自述，只记录代码侧是否存在对应验收入口。

### 0.1 审计基线与工作树状态（可复现性声明）

- 代码基线：`main @ c3e0de176`（`feat(world-terrain): add incremental planet heightfield projection`，2026-10-16 18:52:51）。
- **读取的是工作树，而非该提交**：审计开始时工作树并不干净，有 **10 个已跟踪文件带未提交改动**（合计 `+877 / −155`）：

  | 文件 | 改动量 |
  |---|---:|
  | `pkg/prism_render_app/src/feature_mapping.rs` | +529 |
  | `pkg/prism_render_scene/src/gpu/fabric.rs` | +305 / −155 |
  | `pkg/prism_render_scene/src/gpu/rhi.rs` | +94 |
  | `pkg/prism_render_app/src/scene.rs` | +61 |
  | `os-backends/prism_render_driver_wgpu/src/device.rs` | +23 |
  | `pkg/prism_render_app/src/recovery.rs` | +7 |
  | `pkg/prism_render_architecture/src/render_fabric/compile.rs` | ±2 |
  | `docs/frame_graph_implementation_zh.md`、`docs/prism_ecs_design_zh.md`、`docs/render_fabric_selection_implementation_zh.md` | 文字改动 |

- 这些改动**早于本次审计**（mtime 18:43–19:33，本次审计自 19:35 起），**非本次审计所为**；本次审计仅新增本文件，未改动任何既有文件。
- 影响面：上述文件落在 §2.2（`render_fabric/compile.rs`）、§2.3（`prism_render_app/*`、`prism_render_scene/gpu/fabric.rs`、`gpu/rhi.rs`）的对照范围内，因此这两节反映的是**工作树状态**；若需以 `c3e0de176` 为准复核，需先 `git stash`。其余 §2.1、§2.4–§2.12 不受影响。
- 另注：`docs/refactoring-dev/`、`.mnemon/`、`.idea/` 在 `git status` 中为未跟踪目录。

---

## 1. 全局盘点

### 1.1 规模

| 项目 | 数量 |
|---|---:|
| `pkg/` 目录数 / 真实 crate 数 | 141 个目录，**140 个是真实 crate**（均有 `Cargo.toml`，且均为 workspace 成员）；`pkg/prism_bevy/` 是**空目录**（`src/`、`tests/` 皆空，0 个 `.rs`，无 `Cargo.toml`） |
| `pkg/` `.rs` 行数 | 2,670,701 |
| 全仓 `.rs` 行数（不含 `target/`） | 3,327,586（`pkg/` 占 80.3%，`crates/bevy_*` 老栈 532,949） |
| `pkg/` `#[test]` 数 | 42,436 |
| `docs/` 文档数 / 体量 | 117 篇 / 5.4 MB |
| 显式「验收/门禁」表格行 | 143 行（`refactoring-dev/05` 独占 53 行） |

### 1.2 文档与代码的覆盖关系

- **有 crate 无文档**：6 个 crate 在 `docs/` 中零字面命中 —— `prism_ray_reorder_gpu`、`prism_radiance_cascades_gpu`、`prism_motion_gpu`、`prism_dmm_gpu`、`prism_dmm`、`prism_audio_codec`。其中 `prism_audio_codec` 已确认是**孤儿**：主文档 0 次提及、§39 表不列、且无任何 crate 依赖它。
- **有文档无同名 crate ≠ 缺口**：`docs/` 中大量文档名是**机制名**而非 crate 名（如 `prism_gi_lumen_design_zh.md` 落在 `prism_render_scene`）。按文件名机械匹配会严重高估缺口，故本审计按「文档章节 → 机制 → crate」语义归位。
- **文档自述状态**：`docs/` 中 ✅ 475 处、🟡 124 处、❌ 46 处、`PLANNED` 69 行。这些是**文档侧自述**，本审计逐条回代码核验，并把「文档标 ✅ 但代码查无实现」单列为反例。

### 1.3 与 Bevy 老栈的耦合（完成度的独立维度）

30 / 140 个真实 crate 仍依赖仓库内 `crates/bevy_*` fork（该目录是上游原版 Bevy crate，如 `crates/bevy_math` 自述 `version "0.20.0-dev"`、`repository github.com/bevyengine/bevy`，并传递依赖 `bevy_reflect`）：

| 类别 | crate 数 | 说明 |
|---|---:|---|
| 仅依赖 `bevy_math`（纯数学类型，机械替换） | 27 | 音频系 24 个 + `prism_render_shading`、`prism_physics_gpu`、`prism_asset` 等；典型用法是 `use bevy_math::ops` 取确定性 f32 超越函数 |
| 依赖 2 个及以上 `bevy_*`（需换地基重接） | 3 | `prism_render_scene`（19 个：app/asset/camera/color/core_pipeline/derive/ecs/image/light/log/material/math/mesh/pbr/platform/render/shader/tasks/transform）、`prism_render_material`（6 个：asset/color/image/material/math/pbr）、`prism_render_visibility`（4 个：camera/math/shape/transform） |

（计数由 `pkg/*/Cargo.toml` 中 `crates/bevy_*` path 依赖解析得出，已排除 `description` 字符串等假阳性，二者合计 30，无第三类。）

`prism_audio_bevy` 已退化为纯重导出 facade（187 行，仅依赖 `prism_audio_ecs`，无 Bevy 运行时依赖）；而 `prism_bevy` **已被清空成空目录**（`pkg/prism_bevy/` 下 `src/`、`tests/` 皆空，无 `Cargo.toml`）。需注意这不是文档与代码矛盾——`docs/prism_bevy_refactor_plan_zh.md:30,184,241-242` 明确记载旧桥接「✅ 完成并退役」，空目录是退役残留；**真正的缺口是同文档 §6（`:170`）规定的替代集成 crate `prism_bevy`(渲染) / `prism_bevy_ui` / `prism_bevy_audio` / `prism_bevy_physics` 四个全部不存在**（「退役了旧的、没建新的」）。

---

## 2. 逐模块完成度

（本节按子系统分列；每个子系统给出「要求单元 → 状态 → file:line 证据 → 缺口」明细表与汇总表。）

---

## 2.1 ECS 与核心运行时（18 crate）

审计范围：18 个 pkg/ crate + 12 篇 docs（含完成台账）。方法：read/grep/glob。**未运行 cargo build/test**，所有"落地"判定仅基于源码与测试入口的存在性，**不等于已验证运行**。

#### (a) ECS 完成台账核对（docs/prism_ecs_completion_zh.md:36-41）
| 要求(§) | 声称状态 | 代码/测试证据(file:line) | 是否可宣称完成 |
|---|---|---|---|
| §7 ParamSet | 补齐顺序借用/访问并集 | pkg/prism_ecs/src/system/param_set.rs:131；pkg/prism_ecs/tests/param_set.rs | 机制可宣称；并发调度收益未实测 |
| §9 命令合并 | 结构命令归并 | pkg/prism_ecs/src/command/coalescing.rs、queue.rs | 可宣称（原型迁移次数下降未量化） |
| §10 World 反应图 | 自动标脏+拓扑执行 | pkg/prism_ecs/src/world_reactions.rs、change_observers.rs；tests/observer_replacement.rs | 可宣称；UI 接线另验 |
| §16.2 动态组件热重载 | **布局升级/活数据迁移需补齐** | 仅 component.rs:421 replace_dynamic_layout、:516 动态注册；**全 src 无 hot_reload 符号** | **不可宣称** |
| §16.3 字段级 prefab | 字段继承/覆盖 | pkg/prism_ecs/src/prefab.rs:95,141,173；tests/prefab_{inheritance,reflect,archive}.rs | 机制可宣称；旧资产兼容未验 |
| §16.6/M6 编辑器工具 | **UI/远程消费需补齐** | 有 pkg/prism_ui_inspector/src/ecs.rs、pkg/prism_ui_timetravel/src/ecs.rs | **部分**：profiler 消费与联合流程无 |
| §15 GPU 常驻列 | physical-GPU evidence gate 已有 | pkg/prism_ecs/src/gpu_resident.rs:31,47；pkg/prism_render_app/src/ecs_columns.rs:169 | **不可宣称**：真实 adapter/上传收益未验 |
| §17 NUMA | acceptance gate 已有 | pkg/prism_ecs/src/storage/arena.rs:156；pkg/prism_tasks/tests/numa_acceptance.rs:25,32（默认 SKIPPED） | **不可宣称**：Linux strict、ECS node-local pages 未验 |
| §18/M6 迁移 | audio_ecs 原生、bevy 为 facade | 15 个非测试 Cargo 消费者 | 部分；完整应用/旧资产迁移未验 |
| §23.5 渲染提取 | 独立 render app + 流水线 | pkg/prism_app/src/pipelined.rs、pkg/prism_render_app/src/plugin.rs | 机制可宣称；默认串行 |
| §24.1 tasks/time | lane 优先级/预算已有 | **pkg/prism_app/src/budget.rs:15 已 use prism_time::{Duration, FrameBudget}** | **代码超前于台账** |
| §24.2 Transform | 补齐空间传播到 ECS 观察总线 | pkg/prism_transform/src/observer.rs:124；tests/ecs_observer.rs；prism_transform_ecs/src/bvh.rs:9、grid.rs:8 | 机制可宣称；**ECS 设计 §22(:545) 自述该 crate 未改，不能声称完成** |
| §24.3 reflect | 动态访问/patch/存档/迁移 | pkg/prism_reflect/src/{diff.rs,apply.rs,net_delta,reflect_archive.rs} | 机制可宣称；热重载+prefab 集成未验 |
| §24.4 确定性 | 四环单场景夹具 | pkg/prism_ecs/tests/determinism_link.rs:1,152 | 部分：结构性变更/回滚/跨平台定点未扩 |
| §13/M5 大世界 | 分区/LOD/休眠/浮动原点 | pkg/prism_ecs/src/partition/{cell,lod,dormant,floating_origin,hlod}.rs | 机制可宣称；长测/内存压力未验 |
| §21 性能回归门禁 | 有规模测试和测量输出 | pkg/prism_ecs/tests/scale.rs:13 **明示"deliberately do not assert a wall-clock threshold"** | **不可宣称**：无回归阻断 |

台账引用的 15 个 commit 全部真实存在于本 checkout；HEAD=c3e0de176，台账基线 6b9964234 存在；设计文档(:20) 自述 HEAD a1e33a798 **未核实**。

#### (b) Per-crate 汇总
声明单元数口径 = 该文档 `###/####` 编号小节数（实测）；scene/state_tree 另计路线图行。骨架% = 类型/机制层；接通% = 非测试消费者数（Cargo.toml 依赖边，排除自身）。
| crate | 对应文档 | 声明单元数 | 落地/骨架/部分/未落地 | 骨架% | 接通% | 最大缺口 |
|---|---|---|---|---|---|---|
| prism_ecs | ecs_design(33) | 33 | 24/3/5/1 | ~92% | ~75%（15 消费者） | §16.2 活数据布局迁移缺失 |
| prism_ecs_macros | 同上 | 9 derive | 8/1/0/0 | ~90% | ~75% | 无独立验收 |
| prism_app | app_design(15) | 15 | 13/0/2/0 | ~90% | ~80% | 真实 Winit/移动/跨机 |
| prism_tasks | tasks_design(10) | 10 | 9/0/1/0 | ~95% | ~70%（8） | Linux NUMA strict、扩展曲线 |
| prism_time | time_design(9) | 9 | 8/0/1/0 | ~92% | ~60%（2） | 24.7 OS 挂起信号接线 PLANNED |
| prism_reflect | reflect_design(9) | 9 | 9/0/0/0 | ~95% | ~75%（6） | 热重载/prefab 集成 |
| prism_transform | transform_design(9) | 9 | 8/0/1/0 | ~95% | ~85%（6） | §24.7 免回读渲染接线 PLANNED |
| prism_transform_ecs | transform_design §24.5 | 3 文件 | 1/2/0/0 | ~70% | **0%（无消费者）** | 仅 tests/spatial_indexes.rs 自测 |
| prism_transform_gpu | transform_design §24.7 | 4 文件 | 1/2/0/0 | ~75% | **0%** | **无 strict GPU 失败门** |
| prism_math | math_design(10) | 10 | 9/0/1/0 | ~95% | ~90%（12） | 当前硬件 parity 无日志 |
| prism_math_gpu | math_design §24.1 | 46 文件/43 tests | 1/1/0/0 | ~80% | **0%** | 无消费者；仅 wgsl+metal |
| prism_utils | utils_design(8) | 8 | 7/0/0/1 | ~92% | **~20%（2）** | **prism_ecs 不依赖它** |
| prism_diagnostic | diagnostic_design(8) | 8 | 8/0/0/0 | ~95% | **~30%（1）** | 无 UI/远程消费 |
| prism_scene | scene_design(77+14) | 77 | 6 里程碑"contract 完成" | ~80% | ~40%（3） | 无 editor/asset-bake 联合，性能未测 |
| prism_state_tree_core | state_tree_design(67+9) | 67 | ST0/1/2/3/5/6/8 有切片 | ~90% | **~5%** | 家族外零消费者 |
| prism_state_tree_ecs | 同上 ST1 | 12 文件 | 1/2/0/0 | ~85% | **~5%** | 无 gameplay/AI 宿主 |
| prism_state_tree_gpu | 同上 ST6 | 4 文件/2 tests | 1/1/0/0 | ~80% | **0%** | 无生产消费者 |

关键依赖边实测：prism_ecs 15 个消费者；prism_state_tree_* 仅家族内互引（全 pkg/ 无外部 .rs 引用）；prism_transform_ecs/_gpu/prism_math_gpu 在 Cargo.toml 中仅出现在根 workspace members（Cargo.toml:24,32,33），无任何 crate 依赖；prism_utils 仅 prism_diagnostic、prism_reflect；prism_diagnostic 仅 prism_app（通过可选 diagnostic feature，pkg/prism_app/src/crash.rs:34）。

#### (c) 未验证清单（不可宣称）
1. **物理 GPU**：§15 常驻列、prism_math_gpu 43 个 parity、prism_state_tree_gpu 真机 parity——均有 SKIPPED/严格门，但**无当前主机 adapter 通过与性能数字**。
2. **NUMA/Linux**：PRISM_REQUIRE_NUMA=1 strict success、ECS arena node-local pages、拓扑感知调度收益未验。
3. **网络**：跨进程/跨机、公网、NAT/ICE/relay、生产迁移未验（loopback fault-proxy 仅单进程）。
4. **应用/资产**：完整应用工作流、旧场景资产、editor 联合流程、音频真实 device callback 未验。
5. **性能**：pkg/prism_ecs/tests/scale.rs:13 无墙钟阈值；独占机多核近线性曲线仍 PLANNED；scene/state_tree §13/§25 微秒目标为 design target。
6. **§16.2 动态组件活数据迁移**、**§16.6 profiler 消费**、**§21 可执行回归阻断**：源码无对应入口。
7. **本次未重跑任何测试**；文档中的 "860/1209/477/231 passed" 等为历史记录，**未核实**。

#### (d) 关键结论
- **台账高度诚实但偏保守**：§24.1"App 接入 time 帧预算"已在 pkg/prism_app/src/budget.rs:15 落地，台账仍列为待补证据——属**代码超前于文档**。§22（ecs_design:545）明确拒绝把 transform 部分声称完成，与代码一致。
- **最大系统性风险不是"未实现"，而是"零消费者孤岛"**：prism_state_tree_core/ecs/gpu、prism_transform_ecs/gpu、prism_math_gpu 共 6 个 crate 接通度为 0（仅自测）。state_tree 声称 ST0–ST8 全切片有实现，却没有任何 gameplay/AI crate 装配它——文档 §17 的"与既有子系统装配"未被生产代码消费。
- **三套 GPU twin 的严格门不对称**：pkg/prism_state_tree_gpu/tests/real_device_parity.rs:13 与 pkg/prism_math_gpu/src/context/mod.rs:25 都有 strict-fail 环境变量，而 pkg/prism_transform_gpu/tests/hierarchy_parity.rs:28 的 with_gpu() 在无 adapter 时仅打印 "skipping" 并**静默通过**——设计文档 §24.7 声称的"5 真机 parity 全绿"在本机静态证据下无法与"全部跳过"区分。这是**声称-证据缺口**。
- **prism_utils 未被 ECS 消费**：pkg/prism_ecs/Cargo.toml 无 prism_utils 依赖，尽管 utils 文档 §15 宣称与 ECS 集成；ECS 自带 storage/、util 重复实现，接通率仅 ~20%。
- **prism_diagnostic 已完整实现（§24.1–24.7 全部有模块+测试）但接通仅 ~30%**：唯一消费者是 prism_app 的可选 feature，UI/远程/编辑器消费缺失，与台账 §16.6 一致。
- **scene/state_tree 文档自评克制**：docs/prism_scene_design_zh.md:885 用"功能完成/codec contract 完成/调度 contract 完成"分级，docs/prism_state_tree_design_zh.md:888 用 IMPLEMENTED/DESIGNED/VERIFIED 三分标签，明确"只有实现+测试+基准+版本记录才标 VERIFIED"——口径正确，但 scene 的 docs/version/v0.4.0.md:18 仅有 135 单测，无基准。

---

## 2.2 渲染核心：帧图 / RHI 驱动 / 运行时 / 渲染架构（4 crate）

全部结论为**静态证据**（未实测运行、未 cargo build/test）。状态定义：**落地**=实现且有生产调用者；**骨架**=类型/模块存在但无生产调用者；**部分**=主体实现但子项缺失；**未落地**=文档要求而代码为空。`骨架%=(落地+部分+骨架)/N`，`接通%=落地/N`。

#### 2.2.1 prism_render_graph ← docs/prism_render_graph_design_zh.md（N=34）｜落地11 部分11 骨架0 未落地12｜骨架% 65% 接通% 32%
| 要求单元 | 状态 | 证据(file:line) | 缺口 |
|---|---|---|---|
| §4.1 setup/execute 两阶段 + PassData 回灌 | 落地 | pkg/prism_render_graph/src/pass.rs:22,31；graph.rs:352,390,428 | — |
| §4.2 PSO 反射自动推导 access | 未落地 | 仅用途位并集 resource.rs:81-83,111-112 | 无 reflect/bind_pipeline 入口 |
| §4.3 强类型句柄/Desc 构造器 | 落地 | builder.rs:92-457；handle.rs | — |
| §4.3/§18.11 子图 instantiate | 未落地 | grep subgraph/instantiate 零命中 | 整模块缺失 |
| §5.1 瞬态/导入/持久三态 | 落地 | builder.rs:92,167；graph.rs:267,312；resource.rs:30 | — |
| §5.2 尺寸推导 Screen/½/Fixed | 落地 | compile/alias.rs:53；compile/barrier.rs:64 | — |
| §6.1 SSA 写版本依赖 | 落地 | handle.rs:12；plan.rs:40-46 | — |
| §6.2 死 pass 剔除 | 落地 | compile/cull.rs:96 | — |
| §6.3 细粒度依赖(子区间/mip) | 部分 | 仅 mip/layer 状态 | 无子区间 WAW 细分 |
| §7.1/§7.3 生命周期+别名分配 | 部分 | compile/lifetime.rs；compile/alias.rs；plan.rs:168-187 | 只有 first-fit 一档，无区间图着色档 |
| §7.2 兼容性分类(memory_type/heap tier) | 部分 | AliasPlan 只分 texture/buffer 两池 plan.rs:183-185 | 无 heap tier/RT 专用键 |
| §7.4 aliasing barrier | 未落地 | 无 | — |
| §7.5/§7.7 AliasPolicy 旋钮/一键关 | 未落地 | grep AliasPolicy 零命中 | — |
| §8.1 三元组状态模型 | 落地 | pkg/prism_render_driver/src/sync.rs:334 | — |
| §8.3/§8.4 批合并/split barrier | 部分 | driver tracker.rs:328,357 存在 | graph 未调用 |
| §8.6 队列族所有权转移 | 部分 | plan.rs:63 queue_transfer 字段 | 无生成逻辑 |
| §9.1 队列分批 | 落地 | compile/schedule.rs:66-89；frame_graph/types.rs:46 | — |
| §9.2 timeline semaphore 跨队列 | 未落地 | — | — |
| §9.3 关键路径列表调度 | 部分 | Kahn 确定性排序 | 非关键路径优先级 |
| §10 并行录制确定性归并 | 落地 | parallel.rs；graph.rs:1029-1087 | — |
| §11 subpass/NRP 合并 | 未落地 | 仅 access.rs:39 InputAttachment、pass.rs:103 注释 | 无 subpass 编译器 |
| §11.3 load/store/resolve 推导 | 落地 | plan.rs:192-224 | resolve_target 恒 None graph.rs:1215 |
| §12 不可变 ExecutionPlan + 执行模型 | 落地 | plan.rs:229；graph.rs:52,72,152,1024 | — |
| §13 Blackboard | 落地 | blackboard.rs:20 | — |
| §14.1 DOT/JSON 导出 | 未落地 | — | — |
| §14.2/§14.3 GPU 计时/别名时间轴 | 未落地 | — | — |
| §16 history 环 | 未落地 | grep create_history 零命中 | — |
| §17.1/§17.2 结构哈希+编译缓存/增量 | 未落地 | grep struct_hash/CompileCache 零命中 | — |
| §17.3 零分配竞技场 | 未落地 | src/arena.rs 缺失 | — |
| §19.3 validation 层 | 未落地 | src/validation.rs 缺失 | — |
| §15.1 backend_bridge 适配层 | 部分 | graph 直接调 driver trait | 无独立 bridge |
| §20.4 CompilerPass trait | 部分 | compile/{cull,schedule,lifetime,alias,barrier} 模块化 | 无统一 trait（mod.rs 无 pub trait） |
| §22 GpuCaps 能力门控 | 未落地 | graph 不查 capabilities | — |
| §21 feature 门原则 | 未落地 | Cargo.toml 仅 default/std/parallel | 缺 alias/validation/serde 等 |

#### 2.2.2 prism_render_driver ← docs/prism_render_driver_design_zh.md（N=29）｜落地17 部分4 未落地8｜骨架% 72% 接通% 59%
| 要求单元 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| §4 Instance/Adapter/Surface trait | 落地 | instance.rs:122,135,178；surface.rs:140-183 | — |
| §4 Device/Queue trait | 落地 | device.rs:66,159 | — |
| §6 资源类型+generational 句柄 | 落地 | resource.rs；slotmap.rs:101 | — |
| §7 bindless 数组/绑定 | 落地 | binding.rs:85-167；capabilities.rs:24,43-56 | — |
| §12 PSO/布局 | 落地 | pipeline.rs；state.rs | — |
| §24.7 PSO 二进制导入导出 | 落地 | device.rs:100,112 | 缺管线库拆分层 |
| §9/§10 命令编码 | 落地 | command.rs:448-515 | 缺 bundle.rs/debug.rs |
| §10 状态追踪+自动屏障 | 落地 | tracker.rs:90,162,196 | — |
| §10 split barrier/批优化 | 落地 | tracker.rs:328,357 | — |
| §14 延迟回收 | 落地 | reclaim.rs | — |
| §21 帧飞行不变量 | 落地 | frame_sync.rs:32 | — |
| §14 子分配 TLSF/Buddy/Ring | 落地 | suballoc.rs:136,432,609 | — |
| §14 显存预算 | 落地 | budget.rs:82,88 | — |
| §3 能力/limits 门控 | 落地 | capabilities.rs:164-176 | — |
| §24.3 设备丢失 | 落地 | device.rs:129；loss.rs | 原生后端恢复未做 |
| §20 prelude | 落地 | lib.rs:116 | — |
| §20 swapchain/帧节奏 | 部分 | surface.rs:19-109 | 无 swapchain.rs，帧节奏在 runtime |
| §20§22 backend wgpu/null | 部分 | os-backends/prism_render_driver_{wgpu,null} | 不在本 crate；backend/ 目录缺失 |
| §24.1 GPU-driven indirect | 部分 | capabilities.rs:12；command.rs:654 | 无 multi-draw-indirect-count 原语 |
| §24.5 多队列/异步计算 | 部分 | sync.rs:171 QueueKind | 无 queue_graph.rs 调度 |
| §20§22 阶段二 Vulkan/D3D12/Metal | 未落地 | 无 | — |
| §20§18.4 raytracing.rs 原语 | 未落地 | 仅 Features::RAY_TRACING capabilities.rs:14；sync.rs:47,92 | 无 AS 构建/追踪 API |
| §20§18.3 mesh_shader.rs | 未落地 | 仅 Features::MESH_SHADER capabilities.rs:30 | — |
| §20 feature 列表 backend-*/raytracing/validation… | 未落地 | Cargo.toml 仅 default/std | — |
| §24.4 稀疏/PRT | 未落地 | — | — |
| §24.6 ROV/采样器反馈/保守光栅 | 未落地 | — | — |
| §24.2 多 GPU | 未落地 | — | — |
| §27 跨后端黄金图像门禁 | 未落地 | 仅单元 tests.rs | 无对拍设施 |

#### 2.2.3 prism_render_runtime ← 无专属文档（N=12）｜落地10 部分2｜骨架% 100% 接通% 83%
来源：prism_rhi_m0_m2_plan M0–M2、frame_graph_implementation、refactoring-dev/04-frame-graph_refactor §1.1、prism_render_driver_design_zh.md:511。
| 要求单元 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 经 Instance 建 device/queue | 落地 | src/lib.rs:39；RenderRuntime::<WgpuInstance> | — |
| frames-in-flight 有界 | 落地 | frame_pacing.rs:45,58 | — |
| graph 帧驱动 record/submit | 落地 | submission.rs:20；lib.rs:41 | — |
| 提交所有权 CompletionToken | 落地 | lib.rs:57；submission.rs | — |
| 窗口 surface 创建/配置 | 落地 | lib.rs:230-283 | — |
| acquire/present 循环 | 落地 | lib.rs:149-156 | — |
| 设备丢失恢复+surface 重建 | 落地 | lib.rs:141-146,204-206 | — |
| replay 句柄 | 落地 | replay.rs:91,106 | — |
| 生产调用者 | 落地 | pkg/prism_render_app/src/{lib,device_loss,recovery,config}.rs | — |
| 资源回读 | 部分 | src/tests.rs:6；submission_gpu_tests.rs:53 | 仅测试路径，无生产 API |
| 文档口径 vs 代码 | 部分 | lib.rs:17-19 自称 headless 待接 app loop | 与已有 create_surface/render_frame 矛盾（文档滞后） |
| Null/wgpu 双后端回归 | 落地 | Cargo.toml dev-deps prism_render_driver_null | — |

#### 2.2.4 prism_render_architecture ← prism_rendering_architecture_zh.md + 05-render-fabric + 03/04（N=38）｜落地21 部分9 骨架8｜骨架% 100% 接通% 55%
| 机制组 | 落地/部分/骨架 | 代表证据 | 最大缺口 |
|---|---|---|---|
| §5 统一 GPU Scene | 4/0/0 | gpu_scene/mod.rs:73,84,103,122；upload.rs | — |
| §6 虚拟几何 | 3/1/0 | virtual_geometry/{lod,page_table,page_request,stream_adapter}.rs；外部引用 250 | visibility buffer 未证 |
| §7 材质 PBR/NPR | 3/1/0 | material/mod.rs:11-25；shader_package/execution.rs:8-12 | MaterialExecutionPath::ClosureTable 消费未核实 |
| §8 虚拟资源 | 0/0/3 | virtual_resource/{mod,registry,scheduler,budget}.rs:51,66,86 | **0 外部引用**，纯骨架 |
| §9 虚拟阴影 | 1/2/0 | mod.rs:30,38；prism_render_shading/shadow/virtual_sm/request.rs:24 | clipmap/coverage 无外部证据 |
| §10 GI/光追/离线 | 3/1/0 | ray_scene/(620 引用)、reference_pt/(119)、lighting/(148) | denoise/reservoir 未证 |
| §11 Render Graph 2.0 | 1/1/1 | frame_graph/mod.rs:3（re-export）；work_graph/*(1213 LOC) | work_graph 0 调用者 |
| §12 离线渲染预留 | 0/1/0 | reference_pt/42 文件 | 无 AOV/批处理/checkpoint |
| §13 跨平台 profile | 0/0/2 | backend/capability.rs:77 CapabilitySet | 仅 2 处外部引用 |
| §14 性能/诊断 | 0/0/2 | diagnostics/(1790 LOC)、capture/(1274 LOC) | 0 外部引用 |
| 05 §3–§9 render_fabric | 4/2/0 | render_fabric/{types,pipeline.rs:203-250,selection.rs:40,builtins.rs:527}；生产闭环 fabric.rs:299 已调 prepare_runtime | 仅 dispatch(1,1,1) 探针 |
| 其它(view_family/history/paging/water/quality) | 2/0/0 | pkg/prism_render_app/src/views.rs:27；history::InvalidationMask；fabric.rs:90 | — |

#### 关键结论
- **prism_render_graph** 最大未落地项 = **§17 编译缓存/增量重编译 + §17.3 帧竞技场**（src/arena.rs、src/cache.rs 均缺失），直接违背 §1.1 原则5「成本随变化不随规模」；次之 §11 subpass 合并与 §7.4 aliasing barrier——两者已在 access.rs:39、plan.rs:63 留下类型钩子却无编译器。
- **prism_render_driver** 最大未落地项 = **§24.5 多队列/时间线信号量**（无 queue_graph.rs），sync.rs:171 只有 QueueKind 枚举，故 §21「多队列经 timeline semaphore」契约无法成立；其次 §24.4/§24.6 稀疏与 ROV 全空。
- **prism_render_architecture** 最大未落地项 = **§8 虚拟资源系统的生产接通**：virtual_resource 四文件齐全但**外部引用为 0**。
- **prism_render_runtime** 最薄弱项 = 资源回读仅在 src/tests.rs:6、submission_gpu_tests.rs:53 的测试路径，且 lib.rs:17-19 仍自称 headless——文档与实际能力口径不一致（文档滞后，非代码缺失）。
- 修正 05 文档 §15.3 旧结论：`prepare_runtime` 的生产调用**已存在**于 pkg/prism_render_scene/src/gpu/fabric.rs:299（缺口应为「真实全域 payload 提取」而非「无生产调用」）。
- **无对应 crate 的文档**：prism_rhi_port_surface_zh.md、prism_render_scene_rhi_port_zh.md（目标 prism_render_scene 不在本组）；gpu_cross_sandbox_implementation_zh.md；prism_rhi_m0_m2_plan_zh.md（跨三 crate）；wgpu_native_interop_implementation_zh.md（指向 os-backends/）。**无专属文档的 crate**：prism_render_runtime。

---

## 2.3 渲染应用层：app / scene / visibility / material / shading（7 crate）

**口径声明**：全程只读，**未执行 cargo build/test，未实测运行**。状态分类：落地=有生产调用者；骨架=类型/函数存在但仅测试或 crate 内调用；部分=仅一半机制；未落地=文档声明而代码无；未核实=本轮未取到证据。行号相对仓库根。

#### (a) 要求单元核验（关键单元）

| crate | 要求单元 | 状态 | 证据(file:line) | 缺口 |
|---|---|---|---|---|
| app | PipelineKey/PipelineState 具名状态+from_descriptor/to_hash | 落地 | `pkg/prism_render_app/src/pipeline_cache.rs:286-291`、`:167` | — |
| app | 有界等待队列+多 worker 编译池+背压 | 落地 | `pkg/prism_render_app/src/pipeline_cache.rs:351`、`:403-415`、`:1201` | 文档命名常量 `MAX_IN_FLIGHT_COMPILES` 不存在 |
| app | ResponseModuleCache.sync + register_module_package | 落地 | `pkg/prism_render_app/src/mesh_pipeline.rs:256`、`:264`；调用者 `queue_stage.rs:60` | — |
| app | 冷键只编码 typed state | 落地 | `pkg/prism_render_app/src/pipeline_cache.rs:177`；`lib.rs:142` | — |
| app | LRU 驻留 with_lru_budget/record_resident_size | 骨架 | `pkg/prism_render_app/src/pipeline_cache.rs:690`、`:822` | 调用者仅 `pipeline_worker_tests.rs:721` |
| app | RHI binary import 返回 Unsupported | 骨架 | `pkg/prism_render_app/src/pipeline_cache.rs:991` | 调用者仅 `pipeline_worker_tests.rs:733` |
| app | ecs_columns/recovery/window 忽略门禁 | 落地 | `pkg/prism_render_app/tests/ecs_columns.rs`；`src/recovery/tests.rs:361`；`src/recovery/hardware_tests.rs:109`；`src/presentation/tests.rs:104`(#[ignore]) | 通过数（16/231）未实测 |
| scene | 从 response_program+permutation 派生分类/桶键 | 落地 | `pkg/prism_render_scene/src/shading/runtime.rs:5`、`:439` | — |
| scene | ShadingWorkPlan 经 execution tables 接生产 | **部分** | `pkg/prism_render_scene/src/shading/resources.rs:434` 分支不可达 → `:443` builtin | 无 execution tables 生产者 |
| scene | ResponseOutput 组合生产使用 | 落地 | `pkg/prism_render_scene/src/shading/composition.rs`、`src/shading/response_gpu.rs` | — |
| scene | 两阶段 HZB late stream 消费 | 未核实 | `pkg/prism_render_scene/src/opaque/indirect.rs:247` 仅测试名 | — |
| visibility | cull_view 生产消费 | 落地 | `pkg/prism_render_app/src/visibility/systems.rs:10` | — |
| visibility | HZB 全链模块（footprint/projection/pyramid/query/build/cull_hzb） | 骨架 | `pkg/prism_render_visibility/src/lib.rs:47-71` | 仅 crate 内调用 |
| visibility | two_phase_resolve 遮挡解析闭环 | 骨架 | `pkg/prism_render_visibility/src/two_phase_resolve.rs:83`；`cull_two_phase.rs:71` | `cull_view_two_phase`(lib.rs:47) 无 crate 外调用者 |
| visibility | cull_view_with_hzb | 骨架 | `pkg/prism_render_visibility/src/lib.rs:47` | 无 crate 外调用者 |
| visibility | vbuffer_material_bins | 骨架 | `pkg/prism_render_visibility/src/lib.rs:71` | 无 crate 外调用者 |
| material | ABI 版本一致性（文档称 v6） | 部分/失效 | `pkg/prism_render_material/src/lib.rs:112`；`surface_layout.rs:13`(=8)；`schema/surface.toml:18`(=8) | 文档 `:90` 引 `record.rs:18` 实为 `NO_VIRTUAL_TEXTURE` |
| material | 删硬顶+纹理变长堆 | 落地 | `pkg/prism_render_scene/src/material/buffers.rs:46`、`:114` | — |
| material | PCM 九类 kind hooks | 落地 | `pkg/prism_material_pcm/src/hooks.rs:173`、`:269`、`:320-324` | — |
| material | P2-j response-mode 注册化 | **未落地** | `from_registry`/`UnhandledResponseMode` 全仓 0 命中；`classification.rs` 仅 709 行 | 文档 ✅(`verification:102`) 与代码矛盾 |
| material | P2-c glint/measured 瓣 | 落地 | `pkg/prism_render_shading/src/surface.rs:751-752`、`:807` | `LobeMask::COUNT` 未核实 |
| material | ShadingPlan 三态生产默认 | 落地 | `src/ir.rs:149`；`src/bevy_bridge.rs:224`；`src/registry.rs:412`；`src/execution.rs:226` | — |
| material | MaterialExecutionTables 注册/install | 骨架 | `pkg/prism_render_material/src/execution.rs:282-288`；`src/registry.rs:288` | install 全仓 0 调用者 |
| material | P2-a/P2-b 作者前端→runtime | 落地(跨 crate) | `pkg/prism_render_material_authoring/src/import.rs:146`；`src/procedural.rs:96-126` | 无 crate 消费 |
| material | P1-f2 OpenPBR 默认值八项 / P2-g 动态通道 | 未核实 | — | — |
| authoring | MaterialX 子集 IR+纯数值求值 | 落地 | `pkg/prism_render_material_authoring/src/ir.rs`、`parse.rs`、`evaluator.rs:11`、`tests/materialx.rs` | — |
| authoring | procedural→MaterialGraph→VtSourceSnapshot | 落地 | `pkg/prism_render_material_authoring/src/procedural.rs:11-12`、`:96`、`:126` | 无外部消费者 |
| authoring | nodedef 往返 | 部分 | `pkg/prism_render_material_authoring/src/export.rs` | 映射/实例未落地（`verification:31` 自述） |
| material_gpu | BlockOracle::try_new + 7 项 parity 测试 | 落地 | `pkg/prism_render_material_gpu/src/lib.rs:52-65`；`tests/{bc_parity,albedo_table_gpu_parity,glint_gpu_parity,texture_lod_gpu_parity,…}.rs` | 按文档定位无生产调用者 |
| shading | classify_material_header* + ResponseBucketTable | 落地 | `pkg/prism_render_shading/src/classification.rs:662/665/667/673/678`、`:152`；消费方 `pkg/prism_render_scene/src/shading/runtime.rs:439` | — |
| shading | ResponseOutput/组合契约 | 落地 | `pkg/prism_render_shading/src/response_output.rs:52/74/143/202/233/255`；`lib.rs:63` | — |
| shading | respond_custom / apply_custom_variant | 骨架 | `pkg/prism_render_shading/src/custom_response.rs:97`、`:153` | 唯一非 src 引用是测试注释 |
| shading | CpuResponseRegistry 生产构造 | 部分 | `pkg/prism_render_shading/src/cpu_response.rs:57`；`resolve.rs:404/417` | 构造者仅测试 |
| shading | LobeSampleBudget 贵瓣预算 | 部分 | `pkg/prism_render_shading/src/quality.rs:9/29/91` | GPU 侧未落地（P1-e 自述） |
| shading | GI 63 子系统 | 落地 | `pkg/prism_render_shading/src/gi/{area_light,path_reuse,caustics,light,nee,world_space,specular_aa,irradiance_volume,ies_profile,micro,denoise,temporal,sample,env_brdf,refraction}` 全部存在 | — |

#### (b) 逐 crate 汇总（分母=本轮抽取的可核验声明单元数）

| crate | 对应文档 | 单元数 | 落地/骨架/部分/未落地(含未核实) | 骨架%(类型层) | 接通% | 最大缺口 |
|---|---|---|---|---|---|---|
| prism_render_app | pipeline_runtime / pipeline_state / pipeline_worker_pool_implementation / prism_app_render_acceptance | 7 | 5/2/0/0 | 100% | 71% | LRU 驻留与 driver binary 仅测试调用者 |
| prism_render_scene | prism_aaa(:10,:32) / pipeline_runtime / response_gpu_implementation | 5 | 3/1/1/0 | 100% | 60% | execution_tables 分支不可达 |
| prism_render_visibility | prism_aaa(:722,:830,:917,:1006) | 5 | 1/4/0/0 | 100% | 20% | 两阶段 HZB 仅导出，无 crate 外调用者 |
| prism_render_material | material_pipeline_refactor(+verification) / material_pipeline_design / openpbr_pipeline_function | 13 | 6/1/3/3 | 77% | 46% | P2-j 缺失 + 无 tables 生产者 |
| prism_render_material_authoring | material_pipeline_design(:125) / refactor_zh(:1165,:1190,:1191) | 7 | 4/0/1/2 | 71% | **0%** | 无任何 crate 依赖它 |
| prism_render_material_gpu | material_pipeline_design(:126,:637) / refactor_zh(:27,:774) | 3 | 3/0/0/0 | 100% | N/A（设计即测试 oracle） | 无生产调用者（符合文档定位） |
| prism_render_shading | shading_plan_contract / response_output / response_classification_runtime / refactor_verification / gi_lumen_design | 8 | 3/1/2/2 | 75% | 38% | custom response 生产入口未接通；P2-j 缺失 |

#### (c) 关键结论

- **文档 ✅ 但代码无（最重要）**：`docs/prism_material_pipeline_refactor_verification_zh.md:102` 的 P2-j 证据链（`classification.rs:453 from_registry`、`:509 class_count()`、`:1206` 穷举门禁）整条不存在——`pkg/prism_render_shading/src/classification.rs` 只有 709 行，`from_registry` 与 `UnhandledResponseMode` 全仓 0 命中。
- **文档 ✅ 但证据行失效**：同文档 `:90` 称 ABI v6，实测 ABI=8（`pkg/prism_render_material/src/surface_layout.rs:13`，`prism_material_schema/schema/surface.toml:18`；`lib.rs:112` 才是别名定义处）；`:94` 称 `pipeline_cache.rs:289 MAX_IN_FLIGHT_COMPILES=4`，全仓 0 命中，等价物是 `PipelineCompileConfig{worker_count:4,queue_capacity:64}`（`pkg/prism_render_app/src/pipeline_cache.rs:403-415`）。
- **execution-family 全链骨架化**：`MaterialExecutionTables::new()` 只出现在测试与 `text/integration/` 测试目标，`install_execution_tables`（`pkg/prism_render_material/src/registry.rs:288`）唯一命中即其定义 ⇒ `pkg/prism_render_scene/src/shading/resources.rs:434` 永远走不到，生产恒为 `ResponseBucketTable::builtin()`（`:443`）。
- **引擎侧 custom response 未接通**：`register_custom_spirv` / `ResponseRegistry` 仅 `pkg/prism_render_shading/tests/shading_plan.rs` 调用；`respond_custom`（`pkg/prism_render_shading/src/custom_response.rs:97`）除定义外唯一引用是测试注释 `pkg/prism_render_material/tests/shading_plan.rs:249`。
- **两个孤岛 crate**：`prism_render_material_authoring`、`prism_render_material_gpu` 在 `pkg/*/Cargo.toml` 中零依赖 ⇒ refactor_zh:1190-1191 的 P2-a/P2-b「runtime adapter / bounded lowering」只有 crate 内实现与自测（接通 0%）；AAA `:622/:734` 称 `virtual_texture`/`gdeflate` 为 0 文件也已过时（`pkg/prism_render_scene/src/material/virtual_texture/`、`pkg/prism_gdeflate_gpu/` 等已存在），规模计数同样滞后（shading 344→372、visibility 22→27）。
- **文档覆盖缺口**：`docs/prism_virtual_texture_feedback_zh.md`(39 行) 只规格化 `prism_virtual_texture_gpu`/VT 落点，不对应本次 7 crate；`docs/prism_gi_lumen_design_zh.md`(551 行) 仅 2 处提及 `prism_render_shading`，其余落在 scene；`prism_render_visibility` 在分配清单中无专门设计文档，仅 AAA 规模表（`:722/:830/:917/:1006`）与 `material_pipeline_refactor` 提及；`gpu_cross_sandbox_implementation` / `response_gpu_implementation`(18 行) / `pipeline_worker_pool_implementation`(29 行) / `prism_derived_data_contract` 为片段级，未逐条核实。

#### (d) 本次审计的独立复核
captain 亲自复验了 (c) 中前三条（全仓 `grep -rn --include='*.rs'`，排除 `target/`、`.git/`）：
- `pkg/prism_render_shading/src/classification.rs` 实测 **709 行**；`from_registry` / `class_count` / `UnhandledResponseMode` 在该文件均 **0 命中**（仓库中同名符号仅为无关的 `plan_from_registry`、`begin_shade_writes_from_registry`、`class_counts` 缓冲名）。
- `pkg/prism_render_material/src/lib.rs:112` = `pub use surface_layout::SURFACE_ABI_VERSION as MATERIAL_ABI_VERSION;`，`surface_layout.rs:13` = `pub const SURFACE_ABI_VERSION: u32 = 8;` ⇒ **ABI = 8**，文档 `:90` 的 v6 失效。
- `install_execution_tables` 全仓唯一命中 = `pkg/prism_render_material/src/registry.rs:288` 定义处；`MAX_IN_FLIGHT_COMPILES` 全仓 **0 命中**。

---

## 2.4 几何・特性・派生 GPU 层（16 crate）

**口径**：只读 read/grep/glob，未运行 cargo build/test；分母为逐条核对的「要求单元」。「接通%」= 该 crate 的 GPU/功能路径有**非测试的真实调用者或真实 dispatch 入口**的比例。

#### (a) 关键要求单元核验（跨 crate）
| 要求单元 | 状态 | 证据(file:line) | 缺口 |
|---|---|---|---|
| prism_mesh 设备无关数据模型（属性/索引/拓扑/包围体/流布局/打包/校验/哈希） | 落地 | attribute.rs、indices.rs、topology.rs、bounds.rs、streams.rs、hash.rs | — |
| prism_mesh 独立 validate.rs 模块 | 未落地 | mesh_design §5.2 规定 validate.rs；实测 ls pkg/prism_mesh/src/ 无此文件 | 校验收敛进 mesh.rs/limits.rs/error.rs，文档布局与实测不符 |
| prism_mesh 内置图元（立方/球/胶囊/网格/地形片） | 落地 | pkg/prism_mesh/src/primitives/{cube,sphere,capsule,terrain,plane}.rs | 「网格」图元名未单列 |
| prism_mesh 生产消费 | 落地 | pkg/prism_render_scene/Cargo.toml、prism_render_app/Cargo.toml、prism_world_terrain/Cargo.toml、prism_asset_build_core/Cargo.toml | — |
| OMM 烘焙契约 A8（掩码哈希+细分等级+格式+内核版本，Byte Reproducible） | 落地 | docs/prism_derived_data_contract_zh.md:54 指派 prism_micromap_gpu::bake；pkg/prism_micromap/src（9 文件）、prism_micromap_gpu/src（4 文件） | 无任何 crate 依赖 prism_micromap*，仅孪生自测 |
| DMM 位移微网格 11-bit unorm 烘焙 | 落地 | pkg/prism_dmm/src/lib.rs:1-13、prism_dmm_gpu/src/lib.rs:1-16 | 无设计文档章节规定其落点；无外部调用者 |
| VT feedback decode / page lookup / page table / upload_resizing / real-adapter gate | 落地 | pkg/prism_virtual_texture_gpu/src（12 文件）；规格 docs/prism_virtual_texture_feedback_zh.md | 唯一依赖者 benches/Cargo.toml，非生产路径 |
| 虚拟几何 vis-buffer 软光栅 10 子系统 + 真机 render pass | 落地 | pkg/prism_virtual_geometry_gpu/src（实测 52 文件；AAA:367 记 49） | 无 crate 依赖它 |
| Volumetric 逐 kernel GPU 实现 | 落地 | pkg/prism_volumetric_gpu/src（1277 文件）；真实 ABI+WESL 落 pkg/prism_render_architecture/src/volumetric/gpu(5)、prism_render_scene/src/shaders/volumetric* | prism_volumetric_gpu 本身无调用者 |
| Hair LSS/BSDF GPU 内核 | 落地 | pkg/prism_hair_gpu/src（176 文件）；真实 ABI 落 pkg/prism_render_architecture/src/hair | 无 crate 依赖它 |
| GDeflate「有设备走 GPU 去交错 + CPU Huffman，否则显式回退，返回 DecodeReport」 | 部分 | pkg/prism_gdeflate_gpu/src/decoder.rs:31 GpuDeinterleaveCpuHuffman、:50 DecodeReport、:94 GDeflateDecoder、:124 decode_auto | GPU Huffman 未落地；与 docs/prism_asset_design_zh.md:615 自述一致，**未过度宣称** |
| Radiance Cascades CPU golden + 3 kernel 孪生 | 落地 | prism_render_architecture/src/lighting/radiance_cascades/{mod,cascade,hierarchy,interval,merge,resolve}.rs；prism_radiance_cascades_gpu/src/lib.rs:1 | 无调用者；但 AAA:492 仍标 ⬜待建 |
| SER ray reorder CPU golden + LSD 基数排序孪生 | 落地 | pkg/prism_ray_reorder_gpu/src/lib.rs:1-9、radix.rs；golden 在 prism_render_architecture::ray_scene::reorder | docs 无落点路径；无调用者 |
| motion vector GPU 内核（closest-depth dilation） | 骨架→落地 | pkg/prism_motion_gpu/src（11 文件）；golden 在 prism_render_architecture/src/motion/（tiles.rs/dilation.rs） | **无设计章节**；唯一文档命中 docs/prism_material_pipeline_design_zh.md:438 计数表 |
| prism_image data-half 资产 + 生产消费 | 落地 | pkg/prism_render_app/src/image.rs:10、material/texture_heap.rs:34、material/binding.rs:16、material/stage.rs:9 | — |
| prism_shader WESL 组合/置换/预处理 | 落地 | pkg/prism_shader/src/{compose,def,expr,module,permutation,preprocess}.rs | 无 crate 依赖它，未接入生产 shader 编译链 |
| prism_water_render 真机 wgpu 水面光栅 + PNG 回读 | 落地（自证） | pkg/prism_water_render/{Cargo.toml, src/ocean.rs, src/surface.rs, src/png.rs, examples/{fft_ocean,water_surface}.rs} | **无任何设计文档为其规定机制/里程碑/验收**；无 crate 依赖 |

#### (b) 逐 crate 汇总
| crate | 对应文档 | 声明单元数 | 落地/骨架/部分/未落地 | 骨架% | 接通% | 最大缺口 |
|---|---|---|---|---|---|---|
| prism_mesh | mesh_design §5.2/§5.3/§6/§7/§9/§10 | 11 模块 | 10/0/0/1 | 100% | 91% | validate.rs 模块名不符；4 个真实消费者 |
| prism_micromap | 派生数据契约 A8(:54) + mesh §5.2/§12 + AAA §14.2.3 | 3 | 3/0/0/0 | 100% | 0% | 无外部调用者 |
| prism_micromap_gpu | 同上 | 3 | 3/0/0/0 | 100% | 0% | 无外部调用者 |
| prism_dmm | AAA §14.2.3（:754）仅机制，无落点文件名 | 2 | 2/0/0/0 | 100% | 0% | 无设计文档章节、无调用者 |
| prism_dmm_gpu | AAA §14.2.3 | 2 | 2/0/0/0 | 100% | 0% | 同上 |
| prism_virtual_geometry_gpu | mesh §5.2 + AAA §11.5/§14.3 | 10 子系统 | 10/0/0/0 | 100% | 20%（仅真机 parity 测试触发） | 无生产 dispatch 入口 |
| prism_virtual_texture_gpu | prism_virtual_texture_feedback_zh.md（全文 39 行） | 5 | 5/0/0/0 | 100% | 10%（仅 benches） | 无生产调用者 |
| prism_volumetric_gpu | prism_volumetric_engine_design_zh.md §14 落点明细 | 8 | 8/0/0/0 | 100% | 0% | 真机 adapter 门禁未通过（文档自述） |
| prism_hair_gpu | prism_hair_engine_design_zh.md §9.1 | 7 | 7/0/0/0 | 100% | 0% | 无调用者 |
| prism_water_render | **无规格文档** | 0 | — | n/a | 0% | 无 spec、无依赖者，仅 examples 驱动 |
| prism_radiance_cascades_gpu | gi_lumen §6.3(:179) + AAA §11.5(:492 标⬜待建) | 3 kernel | 3/0/0/0 | 100% | 0% | 文档陈旧标「待建」，且无调用者 |
| prism_ray_reorder_gpu | gi_lumen :44/:382（SER 段，无落点路径） | 2 | 2/0/0/0 | 100% | 0% | 无落点文档、无调用者 |
| prism_motion_gpu | **无设计章节**（仅 material_pipeline:438 计数表） | 1 | 1/0/0/0 | 100% | 0% | 无 spec、无调用者 |
| prism_gdeflate_gpu | asset_design :39/:615/§11.2(:631) | 4 | 3/0/1/0 | 100% | 0% | GPU Huffman 未落地；无调用者 |
| prism_image | asset_design + asset_build_design + rhi_port_surface | 6 | 6/0/0/0 | 100% | 100% | 无 |
| prism_shader | rhi_port_surface / render_driver 面 | 7 | 7/0/0/0 | 100% | 0% | 未接入生产编译链 |

#### (c) 文档声明与代码冲突
| 文档声明 | 实测 | 判定 |
|---|---|---|
| docs/prism_aaa_advanced_features_zh.md:734、:622、:12：opacity_micromap / micromap / gdeflate / virtual_texture / nanite_tessellation **当前均 0 文件** | prism_micromap=9、prism_micromap_gpu=4、prism_dmm=8、prism_dmm_gpu=4、prism_gdeflate_gpu=9、prism_virtual_texture_gpu=12 | **文档陈旧低估（v7 未复算）** |
| :492 辐射级联 ⬜ 待建，落点 gi/radiance_cascades/（待建） | pkg/prism_render_architecture/src/lighting/radiance_cascades/ 6 文件 + RC GPU 孪生 5 文件 | **文档陈旧低估** |
| :812 微几何 RT（DMM+OMM）⬜ micromap 0 文件 | 4 个 crate 共 25 文件 | **文档陈旧低估** |
| :367 虚拟几何 prism_virtual_geometry_gpu（49 文件） | 实测 52 | 计数偏旧 |
| docs/water_render_fabric_implementation_zh.md「严格 PRISM_REQUIRE_GPU=1 为 PASS」+ water_hybrid_implementation_verification_zh.md「严格 GPU 门禁已通过，PASS」 | 两文档同段自述：新增 water_hybrid_gpu target「当前开发机本轮未拿到可用硬件 adapter…尚无本轮真实 dispatch/readback 通过证据」 | **口径冲突**：旧 water raster 门禁 PASS，新 hybrid 路径未验 |
| 两份 water 实现文档的全部落点 | 均为 prism_render_scene（water_surface_raster.wesl 等），**零处提及 pkg/prism_water_render** | 文档与受审 crate 不对应 |

#### (d) 关键结论
- **架构真相**：本仓 pkg/*_gpu 绝大多数是「**真机 wgpu parity 孪生**」，设备无关黄金实现在 pkg/prism_render_architecture/，生产 GPU 路径是 pkg/prism_render_scene/src/shaders/*.wesl + dispatch 接线。因此这些 crate「骨架% 高、接通% 0」是**设计使然而非缺陷**；真正落地度量应看其 golden 宿主是否被生产消费。
- **仅有 3 个 crate 有真实生产消费者**：prism_mesh（4 个 crate 依赖）、prism_image（prism_render_app 多处）、prism_dmm（被 prism_dmm_gpu 依赖）。其余 13 个 crate 在本仓**零 Cargo 依赖者**（prism_virtual_geometry_gpu / prism_volumetric_gpu / prism_hair_gpu / prism_water_render / prism_radiance_cascades_gpu / prism_ray_reorder_gpu / prism_motion_gpu / prism_gdeflate_gpu / prism_shader 全部 0；prism_virtual_texture_gpu 仅 benches）。
- **无规格文档的 crate**：prism_water_render（水体引擎设计 §11 的 18 模块落点全在 prism_render_architecture/src/water/，未列该 crate）、prism_motion_gpu（仅 material_pipeline:438 计数表提及）、prism_ray_reorder_gpu（只有 SER 机制段落，无落点路径）。prism_dmm/prism_dmm_gpu 有机制段落（AAA §14.2.3:754）但**无落点文件名**。
- **有 spec 但文档显著滞后于代码**：prism_aaa_advanced_features_zh.md 在 :12/:622/:734/:812 四处声称 micromap/DMM/gdeflate/virtual_texture **0 文件**，prism_radiance_cascades_gpu 被标 ⬜ 待建——实测共 38+ 文件、含 3 个 GPU 孪生。这是**反向失真**（低估），会误导后续排期。
- **唯一可核实的「文档宣称 ✅ 而代码保守」项**：docs/prism_asset_design_zh.md:39/:615 主动写明 prism_gdeflate_gpu「仍不是 GPU Huffman、DirectStorage 队列」，与 decoder.rs:30-31 的 GpuDeinterleaveCpuHuffman **完全一致，无过度宣称**；prism_mesh_design_zh.md §5.3 甚至主动披露 pkg/prism_render_app/src/ecs_columns.rs:451 读取不存在 ChunkInfo::incarnation 导致主树 RenderApp 编译受阻。
- **GPU 真机证据密度不均**：有明确真机 gate 数字的只有 VT（18 passed，Apple M2）与虚拟几何（cluster_vertex_pull_parity 1 项通过）；volumetric 与 water-hybrid 文档则自述**本轮无可用 adapter、未取得真机通过证据**。
- 所有结论为静态对照，未实测运行；各 crate 内 todo!()/unimplemented!() 计数均为 0，故「骨架」判定基于**模块存在性与是否为纯类型/契约层**，非 stub 宏。
- **在役文档中未被任何受审 crate 对应者**：docs/prism_rhi_port_surface_zh.md、water_render_fabric_implementation_zh.md、water_hybrid_implementation_verification_zh.md、prism_derived_data_contract_zh.md（除 A8 一行外）、prism_asset_build_design_zh.md（gdeflate/image 仅旁及）。

---

## 2.5 材质 PCM 与 Schema

#### 口径

- **审计对象**（均存在）：`pkg/prism_material_pcm`（src 合计 3315 行，其中 722 行为 `src/tests.rs`）、`pkg/prism_material_schema`（`src` + `src/bin` 合计 3529 行，另 `schema/surface.toml` 1406 行）。
- **状态码四档**：
  - **落地** = 代码实现完整，且存在 crate 之外的非测试生产调用方；
  - **骨架** = 类型/API 齐备，但生产侧无调用方（仅测试、门禁或桥接层引用）；
  - **部分** = 已实现，但文档要求的子情形缺失；
  - **未落地** = 文档有明确要求，代码中不存在对应机制。
- **骨架/类型层 % = (落地 + 骨架 + 部分) / 单元总数**（分母为该 crate 从文档抽出的要求单元数，逐一列出）；含义是"结构上至少有对应物"的覆盖率。
- **端到端接通/生产调用层 % = 落地 / 单元总数**；接通判据严格采用题给口径：**被 crate 之外的非测试代码调用**。`tests/*.rs`、`#[cfg(test)]` 模块、dev-dependency 一律不计。凡调用链的入口结构体只在测试中构造，另在「缺口」列显式标注，不因此上调档位。
- 本次为**静态证据审计**：按题目要求未执行 `cargo build`/`cargo test`。文档中的 "N 项通过" 均**未实测**；仅在与静态 `#[test]` 计数可交叉验证处标注"静态一致/不一致"。
- 路径均相对仓库根。无法静态判定处一律写 **未核实**，不做推测。

---

#### 2.5.1 prism_material_pcm（Prism Capability Model）

##### 表 A：design doc `prism_material_pipeline_design_zh.md`（§2.5 决策 / §3.6 概念表 / 附录 C）

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| A1 Kind 九值枚举 `Lobe/Channel/ResponseMode/Transfer/CacheAux/Provider/Downgrade/AuthorNode/Composite` | 骨架 | `pkg/prism_material_pcm/src/namespace.rs:213-233`（枚举）、`:237-247`（`ALL`）、`:250-262`（`as_str`，含 `response_mode`/`cache_aux`/`author_node`） | 类型完整；无生产调用方（唯一构造/使用点是 `pkg/prism_render_material/src/capability.rs:121` 的 `Kind::Lobe`，而该函数无生产调用方） |
| A2 四区 × 16 槽 = 64 槽摘要 | 骨架 | `namespace.rs:89-98`（`Region` 四值）、`:112`（`SLOTS_PER_REGION = 16`）、`lib.rs:75,79`（`HEADER_DIGEST_SLOTS = 64`/`HEADER_DIGEST_WORDS = 4`） | 常量为生产可见，但分配器无生产调用方 |
| A3 `Stability` 三级 + 单向晋升门 | 骨架 | `namespace.rs:170-177`、`:181`（`ALL`）、`:200-202`（`can_promote_to`） | 仅被 `manifest.rs:526-552` 与桥接层引用 |
| A4 `ExtensionManifest` 身份/可变分组 + 派生展示名 | 骨架 | `manifest.rs:238-251`（字段划分）、`:257,280`（两种形态构造）、`:300-355`（builder）、`:437-450`（`id()` 派生 `{NAMESPACE}_{kind}_{semantic}`） | 无生产调用方 |
| A5 `validate()` 字段谓词（10 类具名错误） | 骨架 | `manifest.rs:33-90`（`ManifestError` 十项）、`:460-513`（校验体：uid 形态/版本非零/常驻↔槽一致/位区一致/能量策略仅 lobe/KHR 不得要能力） | 只被 `registry.rs:116` 与桥接层 `pkg/prism_render_architecture/src/render_fabric/pcm.rs` 调用，两者均无生产调用方 |
| A6 一张注册表 + 分区槽分配 + `PoolExhausted` | 骨架 | `registry.rs:79-89`（`slots: [Option<ExtensionManifest>; 64]` + `identity_only` + 每区游标）、`:191-208`（first-fit 分配）、`:44-55`（`RegistryError`） | **生产调用方 0**：全仓 `ExtensionRegistry` 实例化点仅 `prism_render_material/src/capability.rs:98`、`render_fabric/pcm.rs:105`、两处测试文件 |
| A7 `promote()` 晋升 = 翻字段不改身份 | 骨架 | `registry.rs:224-252`（原位改写两种常驻形态）、`manifest.rs:526-552` | 无生产调用方 |
| A8 identity-only 形态 + `header_resident()` 谓词 | 骨架 | `hooks.rs:193-202`、`manifest.rs:280`（`new_identity_only`）、`registry.rs:174`（`register_identity_only`）、`:367`（`registry_slot()==None`） | 语义完整（5 类无槽），无生产调用方 |
| A9 按 uid 协商 `required_caps_of_uids` | 骨架 | `registry.rs:297`（`required_caps_of`）、`:315`（`required_caps_of_uids`） | 无生产调用方 |
| A10 性能红线：治理在烘焙/载入期完成、运行时零分发 | 部分 | 唯一生产触点 `pkg/prism_render_material/src/record.rs:810`（CPU 侧 header 构建时 `digest_for`）；GPU 热路径未见注册表/协商调用点 | 无运行期门禁或测量证明"零分发"；**未核实**（本轮不运行，无法实测） |
| A11 边界约束：PCM 不得成为总注册表 | 落地 | `manifest.rs:238-251`（`ExtensionManifest` 无 `reads/writes/passes/services` 字段）；`docs/refactoring-dev/05-render-fabric_refactor_zh.md:39,1077` 明文禁止 | — |

##### 表 B：`prism_material_pipeline_refactor_zh.md` §5（含 §5.1–§5.5）+ §13.a P1-h/P2-i 行

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| B1 能力位集 `Capability`（6 值）+ `CapabilitySet` | 落地 | `capability.rs:14-27`、`:31`、`:84-146`；生产侧经 `pkg/prism_render_material/src/capability.rs:51` re-export，用于 `pkg/prism_render_material/src/shading_plan.rs:666`（`PlanValidationContext.capabilities`）与 `:1272-1277`（`Capability::parse` + `contains`） | 注意：`PlanValidationContext` 仅在 `pkg/prism_render_material/tests/shading_plan.rs:386-387` 构造，`validate_plan` 无 crate 外生产调用方 → 该类型虽在非测试源码中被引用，生产实际未走通 |
| B2 `ExtensionSet` 64 槽 `used/required` 不变式（`required ⊆ used`） | 落地 | `extension_set.rs:38-41`、`:190-199`（`bit()` 拒绝越界槽）、`:134-167`；生产链：`pkg/prism_render_material/src/capability.rs:142-166`（`digest_for`）← `pkg/prism_render_material/src/record.rs:810`（`header_with_execution`）← `pkg/prism_render_scene/src/material/buffers.rs:160`、`pkg/prism_render_app/src/material/buffers.rs:308`、`pkg/prism_render_scene/src/shading/runtime.rs:414`（三处均在各自 `#[cfg(test)]` 之前，属生产路径） | — |
| B3 4 字摘要常量面（`HEADER_DIGEST_SLOTS`/`HEADER_DIGEST_WORDS`） | 骨架 | `lib.rs:75,79`；crate 之外**仅两处文档注释**引用（`pkg/prism_render_material/src/record.rs:211`、`src/capability.rs:140`），无非测试代码读取该常量 | 常量不参与生产计算（摘要实宽由 `ExtensionSet` 的 `u64` 硬编码决定），亦无门禁断言其与实际宽度一致 |
| B4 `hooks` 表：九类 kind 有序 hook + `Stage`/`Twin` 由 stage 派生 | 骨架 | `hooks.rs:48-53`（`Stage`）、`:68-73`（`twin()`）、`:148-169`（九张 hook 常量表）、`:173-231`（`hooks()/header_resident()/runtime_hook_count()/declares_hook()`） | 无生产调用方；仅 `hooks.rs:261-368` 内 5 条测试（静态计数 `hooks.rs` 6 个 `#[test]`） |
| B5 `Kind::touches_class_table()` 收敛唯一热路径改动 | 骨架 | `hooks.rs:211-213`；唯一使用点 `hooks.rs:354`（同文件 `#[cfg(test)]`） | 无生产调用方 |
| B6 `energy_policy` + `roll_up`（coupled>independent>none）+ 非 lobe 拒绝 | 骨架 | `manifest.rs:170-213`（枚举 + `roll_up`）、`:498-503`（`EnergyPolicyOutsideLobe`）、`builtin.rs:321-334`（内置策略轮转） | 无生产调用方 |
| B7 内置目录（22 行：7 KHR 瓣 + 7 PRISM 瓣/通道/模式 + 5 identity-only） | 骨架 | `builtin.rs:98`（KHR 槽表）、`:106-114`、`:139-196`、`:205-238`、`:269-305`、`:347-391`（`builtin_registry()`） | 唯一调用方 `pkg/prism_render_material/tests/pcm_parity.rs:12,27,303,360,428`，全为测试 |
| B8 §5.1 核心自举 dogfood：OpenPBR 核心也注册为 `kind=lobe` | 骨架 | `builtin.rs:106-114`（KHR 七瓣以 `Kind::Lobe` 走同一注册路径）；无"核心瓣硬编码 / 扩展瓣走注册"二分 | 结构上成立，但注册表不在生产路径上（见 A6） |
| B9 §5.3 `stability` 是 required/optional 唯一策略源 + digest 由 schema 派生 | 落地 | `pkg/prism_render_material/src/capability.rs:142-166`（`REQUIRED_STABILITY` ⇒ `insert_required`，其余仅 `insert_used`）；生产链同 B2；双向平价门禁 `pkg/prism_render_material/tests/pcm_parity.rs:215,231` | 协商本体（`negotiate`）本身仍未接通（见 B10） |
| B10 载入期握手 `negotiate()`：required 缺失硬错 / used 缺失 graceful ignore | 骨架 | `negotiate.rs:134-167`、`:38-58`（错误类型）、`:60-70`（`is_lossless`）；桥 `pkg/prism_render_material/src/capability.rs:168-172`（`negotiate_digest`） | `negotiate_digest` 的引用仅为文档注释 `pkg/prism_render_material/src/record.rs:223` 与测试；**生产路径无任何协商调用** |
| B11 §5.4 "每条扩展 = 一份 TOML manifest（codeless）" | 未落地（本 crate） | `pkg/prism_material_pcm/Cargo.toml` `[dependencies]` 为空；`builtin.rs` 为 Rust `const` 行（`grep` 无 `std::fs`/TOML/loader 命中） | PCM 自身无 TOML 解析与加载路径；TOML 真源在 `pkg/prism_material_schema/schema/surface.toml`，两侧靠 `tests/pcm_parity.rs` 平价而非加载 |
| B12 §5.4.1 受限 Rust 瓣 `proc-macro → WESL` + `golden="auto"/"none"` 逃生口 | 未落地 | 全仓 `proc-macro = true` 仅有 `pkg/prism_ecs_macros/Cargo.toml`、`pkg/prism_reflect_macros/Cargo.toml`、`pkg/prism_ui_macro/Cargo.toml`；无材质瓣宏/`golden` 属性 | 机制整体缺失 |
| B13 §5.4.2 编辑器渲染插件 + 打包即烘焙 + 插件信任边界（签名 / wasm 沙箱） | 未落地 | 无 PCM 包格式、loader、沙箱代码 | 机制整体缺失 |
| B14 §5.5 `cost_hint` 进 CI | 未落地 | `cost_hint` 仅出现在文档：`docs/prism_material_pipeline_refactor_zh.md:320,424`；`ExtensionManifest` 无该字段（`manifest.rs:238-251`） | 声明字段与 CI 门禁均缺 |
| B15 §13.a P1-h 记录"九类 kind 全部迁入 + 28 项门禁" | 部分（数目） | `pkg/prism_material_pcm/src/tests.rs` 22 个 `#[test]` + `src/hooks.rs` 6 个 `#[test]` = **28**，与 `docs/prism_material_pipeline_refactor_zh.md` §13.a.1「`prism_material_pcm` 28」**静态一致** | 仅静态计数一致；"通过"未实测。P1-h 描述的功能面全部只到骨架档 |
| B16 §13.a P2-i「8 个有实例的 kind 全部声明化；author_node 为第 9 类」 | 落地（声明面） | schema 侧 8 类记录见 2.x.2 表 A/B；PCM 侧 `builtin.rs:269-305` 5 条 identity-only + `hooks.rs:168` `AUTHOR_NODE_HOOKS` | 声明面成立；author_node 实例缺失（见 B17） |

##### 表 C：`prism_material_pipeline_refactor_verification_zh.md` §3.1/§4 与跨 crate 缺口

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| C1 第 9 类 kind `author_node` 有实例（词表/hook 就位，缺实例） | 未落地 | `builtin.rs:258-274`（注释显式说明故意缺席，`PRISM_IDENTITY_ONLY` 仅 5 行）、`src/tests.rs:599-613`（显式断言其行为空） | 无实例、无 schema 记录（`schema/surface.toml` 无 author_node 段） |
| C2 design §3.6 缺口：「composite 无运行时 pass 注册（帧图作为 pass 来源）」 | 未落地 | 无 pass 注册 API（`grep` 无 `pass` 注册表/帧图来源桥） | 机制缺失 |
| C3 design §3.6 声称「PCM 目前只被 `prism_render_scene/.../classification_gpu.rs:42` 与 `prism_render_shading/src/classification.rs` 消费」 | 与代码不符（文档低估） | `pkg/prism_render_shading/Cargo.toml` **不含** `prism_material_pcm`；`pkg/prism_render_scene/src/shading/classification_gpu.rs:42` 前后为 `#[cfg(test)]` 区（`:38`）+ `GpuShadingClassificationParams` 定义，非 PCM 消费点；实际生产触点见 B1/B2 | **文档-代码矛盾**：文档描述的消费点已过期，真实触点已迁到 `prism_render_material/src/{record.rs:810, shading_plan.rs:666,1272}` |

##### PCM 汇总表

| crate | 对应文档 | 声明单元数 | 落地/骨架/部分/未落地 | 骨架% | 接通% | 最大缺口 |
|---|---|---|---|---|---|---|
| `prism_material_pcm` | design §2.5/§3.6；refactor §5（§5.1–5.5）+ §13.a P1-h/P2-i；refactor_verification §3.1/§4；refactoring-dev/05 §0.2/§5.2；render_fabric_selection_implementation | 29 | 5 / 16 / 2 / 6 | **79.3%**（23/29） | **10.3%**（3/29：B1、B2、B9） | 整张注册表（`ExtensionRegistry`/`ExtensionManifest`/`negotiate`/内置目录）**没有任何生产调用方**；生产唯一真正接通的只有 `ExtensionSet` 摘要类型与 `stability→required` 派生，`negotiate` 从未被调用 |

> 计数说明：表 A 11 项 + 表 B 16 项 + 表 C 计入 2 项（C1、C2；C3 是"文档错报"发现，不作为要求单元计入）＝ **29** 个互不重复单元。
> 分档明细：落地 5 = A11、B1、B2、B9、B16；骨架 16 = A1,A2,A3,A4,A5,A6,A7,A8,A9,B3,B4,B5,B6,B7,B8,B10；部分 2 = A10、B15；未落地 6 = B11,B12,B13,B14,C1,C2。
> 落地 ≠ 接通的 2 项：A11（负向边界约束，`ExtensionManifest` 无 `reads/writes` 等字段，无调用方可言）与 B16（声明面完备，仅由测试核验）计入"实现"档但不计入接通分子；B1 的调用点虽在非测试源码 `pkg/prism_render_material/src/shading_plan.rs:666,1272`，但其入口结构体 `PlanValidationContext` 只在测试中构造，已在表 A/B 缺口列标注。

---

#### 2.5.2 prism_material_schema（自描述 surface schema + codegen）

##### 表 A：`prism_material_pipeline_refactor_zh.md` §17.5.1/§17.5.2（声明模型）

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| A1 §17.5.1 schema 作为单一声明源、字段/掩码/默认值不再多处手写（S1） | 落地 | `pkg/prism_material_schema/schema/surface.toml`（1406 行，14 条 `[[core.field]]`、10 条 `[[lobe]]`、63 条 `[[lobe.field]]`、4 条 `[[channel]]`、3 条 `[[response_program]]`、1 条 `[[cache_aux]]`、5 条 `[[identity_extension]]`）；`pkg/prism_material_schema/src/lib.rs:43,53-61`（`include_str!` + `surface_schema()`） | S1 的逐字节对拍门禁不在本 crate，而在 `pkg/prism_render_material/tests/schema_parity.rs:21,155,489`（属测试） |
| A2 字段维度完整性（名/类型/位宽/位区/打包偏移/`spec_subfield`/`consumed_by`/默认值/合法谓词/能量策略/视角/调试视图/金标准绑定） | 部分 | `lib.rs:324-364`（`Field`：`name/ty/bit_region/pack_slot/spec_subfield/consumed_by/responses/default/valid/energy_policy/view/debug_view/surface_field`）、`:378-407`（`FieldType::words()` 派生位宽）、`:521-529`（`SpecSubfield` 三值）、`:534-555`（`ConsumedStep` 十值，含 `Golden`/`Debug`/`Doc`/`Compliance`） | **缺"金标准绑定"字段**（CPU golden 绑定名无声明位）；`ConsumedStep::Golden/Debug/Doc/Compliance` 只声明了消费方，无对应生成物（见表 B） |
| A3 §17.5.2 九类 kind 一视同仁（provider 声明页格式、composite 声明 pass I/O、response-mode 声明 class 条目） | 部分 | `lib.rs:101-132`（`Channel`：`header_word/default/valid/requires_caps/view_dependent/usable_as_mask`）、`:218-236`（`CacheAux`）、`:247-265`（`IdentityExtension`：`kind/namespace/vendor/stability/requires_caps/view_dependent`）、`:270-276`（`ResponseProgram`）；`schema/surface.toml:1359-1402`（5 条 identity_extension：provider×2/transfer/downgrade/composite） | **8/9 kind 有记录，`author_node` 无**；且 `provider` 未声明"页格式"、`composite` 未声明"pass I/O"——这两类目前只有身份/治理/能力列，与文档 §17.5.2 的差异化声明要求不符 |
| A4 lobe 变长词足迹 + `SURFACE_LOBE_WORDS_BY_SLOT` | 落地 | `lib.rs:310-318`（`words` 逐瓣）、`:649-694`（`packed_len_words`/`placements`/`lobe_field_offset`）；`codegen.rs:509-556`；产物 `pkg/prism_render_material/src/surface_layout.rs` 生成 `[8,16,4,8,16,16,4,0×9,4,4,4]`（`codegen.rs:1340` 断言同值） | — |
| A5 位区隔离与槽致密（§4.5 约束 2）+ 全 kind 共享槽空间机检 | 落地 | `lib.rs:775-800`（canonical `0..k` 与 extension `EXTENSION_SLOT_BASE..` 双段致密，`NonContiguousRegistrySlots`）、`:757-770`（`DuplicateRegistrySlot`）、`:818-840`（`DuplicateHeaderSlot` 跨 lobe/response_program/channel/cache_aux）、`:841-880`（通道字序致密 + 默认值在域内） | 致密性的"瓣区 vs 非瓣区不得交错"断言在 `pkg/prism_render_material/tests/pcm_parity.rs:437-456`（测试），本 crate 只查区内致密 |
| A6 `closure` 关系 + `spec_subfield='closure_mask'` | 落地 | `lib.rs:300-306`（`Lobe.closure` 可选）、`:521-529`（`SpecSubfield::ClosureMask`）、`:775-781`（空 closure 名拒绝 `EmptyClosureName`）；`codegen.rs:726-730`（`SURFACE_LOBE_CLOSURES` 类型化引用） | — |
| A7 运行时身份与 header word 映射（channel 等） | 落地 | `lib.rs:101-132`（`header_word` 与 `registry_slot` 语义分离的注释 + 字段）、`:673-695`（`lobe_field_offset`/`core_field_offset`）；`codegen.rs:759-790`（`ChannelLayout`）、`:815-860`（`CacheAuxLayout`）、`:941-1010`（response program 的五个消费契约） | — |
| A8 不变量守恒（无 `MAX_*`、确定性单源、每材质静态特化） | 落地 | `lib.rs:775-800`（位区按 namespace 增长而非硬顶）；`codegen.rs:70-97,556`（`SURFACE_MAX_WORDS` 由 schema 派生）；`schema/surface.toml` 内无 `MAX_*` 字面量 | — |
| A9 §17.5.5 设计边界：schema 不表达行为数学 | 落地 | `lib.rs:27-33`（边界声明注释）；`Field`/`Lobe` 结构体无行为/表达式字段（行为在 `responses` 语义表内，见 `pkg/prism_material_schema/src/response.rs:11-258`） | — |

##### 表 B：`prism_material_pipeline_refactor_zh.md` §17.5.3（六类 codegen 生成物）与 §17.5.7/17.5.8

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| B1 生成物 ② WESL `material_unpack` | 落地 | `codegen.rs:99-286`（`emit_wesl_unpack`）；产物写出 `pkg/prism_material_schema/src/bin/prism-material-codegen.rs:51-54` → `pkg/prism_render_scene/src/shaders/material_unpack.wesl`；被 4 个生产着色器 import（`grep -rl`：`pkg/prism_render_scene/src/shaders/opaque.wesl`、`shading_resolve.wesl`、`ssr_repack.wesl`、`transparent.wesl`） | 生成头注释（`codegen.rs:134-136`）声称"all five consumers (opaque, transparent, visibility, shading_resolve, ssr_repack)"，实际 `visibility.wesl` 未 import：**生成物注释与代码轻微不一致（4/5）** |
| B2 `01-surface-abi_refactor_zh.md:7` 要求 codegen 同时派生 WESL "mirror struct" | 部分 | `codegen.rs:99-286` 只发射常量 + `fn`，并 `import ...PrismSurfaceParameters`（`codegen.rs:135`）；无 WESL 结构体发射代码 | WESL mirror struct 仍为手写，非生成物 |
| B3 生成物 ① Rust layout + 字段目录（并被生产消费） | 落地 | `codegen.rs:411-1237`：`SURFACE_LOBES:684`、`LobeMask` 常量 `:573`、`SURFACE_LOBE_WORDS_BY_SLOT:542`、`SURFACE_CHANNELS:785`、`SURFACE_CACHE_AUX:842`、`SURFACE_IDENTITY_EXTENSIONS:914`、`SURFACE_RESPONSE_PROGRAMS:1070`、`SURFACE_FIELDS:1200`；产物 `pkg/prism_render_material/src/surface_layout.rs` 被生产代码消费：`pkg/prism_render_material/src/lib.rs:25,112-113`、`src/surface.rs:26`、`src/openpbr.rs:33`、`src/shading_plan.rs:318,1272`、`src/capability.rs:48` | — |
| B4 生成物 ② WESL `material_classification` | 未落地 | 无对应 emit 函数（`codegen.rs` 仅两处 `emit_`：`:99`、`:411`）；`pkg/prism_material_pipeline_refactor_verification_zh.md:114` 亦记"`material_classification.wesl` 仍手写" | 手写漂移面仍在 |
| B5 生成物 ② WESL `material_sample` | 未落地 | 同上；`pkg/prism_render_scene/src/shaders/material_sample.wesl` 存在但非生成 | 手写漂移面仍在 |
| B6 生成物 ③ CPU golden 布局（`prism_render_shading`） | 未落地 | 无生成物；`ConsumedStep::Golden` 仅声明（`lib.rs:538`） | 真值侧仍是第二份手写 |
| B7 生成物 ④ debug view 字段表 | 未落地 | `Field.debug_view` 声明于 `lib.rs:356`；全仓 `debug_view` 的其他命中均为 `prism_render_scene` 的 GPU 调试视图无关字段（`src/opaque/queue.rs:36,126,189,228`、`src/opaque/pipeline.rs:72-251`）；codegen 不发射该列 | 声明了但**零消费者、零生成物** |
| B8 生成物 ⑤ 文档字段表 + 合规枚举 | 未落地 | 无生成物；doc §4.4/§3.5 字段表仍手写 | 文档表仍需人工同步 |
| B9 生成物 ⑥ 校验谓词（§4.5 三约束可机检） | 部分 | `lib.rs:730-899` 机检：槽碰撞/位区一致/区内致密/通道字序/默认值在域内/合法谓词可解析（`MissingValidityPredicate`/`InvalidValidityPredicate`/`DefaultOutOfRange`）；`lib.rs:152-197`（`ValidRange::parse/contains`） | 约束 1「合规对拍 = 只用 KHR 项时响应等于 OpenPBR 参考」**无代码**（`grep openpbr` 在本 crate 仅命中注释与 uid 字面量，无参考枚举/对拍表） |
| B10 §17.5.7 codegen 增量生成 + 缓存 + CI 新鲜度 | 部分 | 新鲜度门禁已落地：`bin/prism-material-codegen.rs:76-100`（`check` 逐产物比对、列出 stale）+ `pkg/prism_render_material/tests/codegen_freshness.rs:18-47` | **无增量/内容哈希缓存**：`bin/prism-material-codegen.rs:47-58` 每次全量重新发射 2 个产物 |
| B11 §17.5.7 生成物可读 + 源映射回 schema | 部分 | `pkg/prism_render_material/src/surface_layout.rs:1-2`（`// DO NOT EDIT` + 再生成命令）；`codegen.rs:111-113`（WESL 同款 banner） | 无 schema 行号映射（`grep surface.toml` 在产物中仅 3 处注释引用，无 `file:line` 溯源） |
| B12 §17.5.8 阶段 S1 ✅ / S2 ✅ 与阶段门禁 | 落地 | S1 对拍：`pkg/prism_render_material/tests/schema_parity.rs:21,155,489`；S2 生成 + 新鲜度：`bin/prism-material-codegen.rs`、`tests/codegen_freshness.rs`；本 crate 静态测试计数 `src/lib.rs` 18 + `src/codegen.rs` 8 = **26** 个 `#[test]` | 文档所记门禁数（verification doc:47「19 vs 20」、§13.a.1 各行 13/15/18/19）与静态 26 **不一致**，属文档快照过期（未实测运行，不判定通过/失败） |
| B13 `abi_version` 与运行时 ABI 单一来源 | 落地 | `schema/surface.toml:18`（`abi_version = 8`）→ `src/lib.rs:75`（`SurfaceSchema.abi_version`）→ `codegen.rs:495`（`SURFACE_ABI_VERSION`）→ `pkg/prism_render_material/src/surface_layout.rs:13`（`= 8`）→ `pkg/prism_render_material/src/lib.rs:112`（re-export 为 `MATERIAL_ABI_VERSION`）；断言 `tests/schema_parity.rs:21` | 版本号在文档侧全面过期，见「关键结论」 |
| B14 schema 作为 PCM 目录的单向真源（双向平价） | 落地 | `pkg/prism_render_material/tests/pcm_parity.rs:26-27`（schema 派生表 ↔ `builtin_registry()` 双向逐行）、`:215,231`（digest）、`:303,360,428`（identity-only/channel/cache_aux） | 平价是**测试**而非 schema crate 内门禁；schem 侧无自我校验 API |

##### schema 汇总表

| crate | 对应文档 | 声明单元数 | 落地/骨架/部分/未落地 | 骨架% | 接通% | 最大缺口 |
|---|---|---|---|---|---|---|
| `prism_material_schema` | refactor §17.5（§17.5.1–17.5.8）；refactoring-dev/01-surface-abi；material_artifact_implementation；prism_openpbr_pipeline_function（§12 位预算、字段真源）；design §2.4/§3.5；refactor_verification §2.1/§3.2/§5.1 | 23 | 12 / 0 / 6 / 5 | **78.3%**（18/23） | **0%**（0/23，严格 API 口径）<br>生成物被生产消费口径 **≈26%**（6/23） | 六类生成物只做出 ①②（且 ② 只有 unpack）：`material_classification`/`material_sample`/CPU 金标准/调试视图表/文档字段表/合规枚举全部未生成；crate API 本身**零生产调用方**（`surface_schema()` 只被测试与自带的开发 CLI 调用） |

> 计数说明：表 A 9 项 + 表 B 14 项 = **23** 个互不重复单元（A4 记声明侧、B3 记生成物侧，二者不合并）。
> 分档明细：落地 12 = A1,A4,A5,A6,A7,A8,A9,B1,B3,B12,B13,B14；骨架 0；部分 6 = A2,A3,B2,B9,B10,B11；未落地 5 = B4,B5,B6,B7,B8。
> 接通口径特别说明：`prism_material_schema` 的公开 API（`surface_schema()`、`load_surface_schema()`、`emit_*`）在全仓的非测试调用点**只有**它自带的 `src/bin/prism-material-codegen.rs`（同 crate 的开发/CI CLI）——`pkg/prism_render_material/src/surface_layout.rs:2,6` 仅有注释提及，`tests/{schema_parity,shading_plan,surface_abi_mirror,codegen_freshness}.rs` 全为测试。因此严格 API 接通 = **0/23**。但其**生成物**进入生产：`surface_layout.rs` 编入 `prism_render_material`（`pkg/prism_render_material/src/lib.rs:25,112-113`）并被 `src/surface.rs:26`、`src/openpbr.rs:33`、`src/shading_plan.rs:318,1272`、`src/capability.rs:48` 使用（`LobeMask::*` 见 `src/record.rs:772-777`），`material_unpack.wesl` 被 4 个生产着色器 import，`MATERIAL_ABI_VERSION` 用于 `pkg/prism_render_app/src/pipeline_cache.rs:147`。按"生成物进入生产"口径计入 A4、A5、A7、B1、B3、B13 → **6/23 ≈ 26%**。

---

#### 覆盖矩阵：文档 ↔ crate

| 文档 | 是否指定本审计 crate | 说明 |
|---|---|---|
| `docs/prism_material_pipeline_design_zh.md` | 两者 | pcm 8 处 / schema 9 处 |
| `docs/prism_material_pipeline_refactor_zh.md` | 两者 | pcm 18 处 / schema 39 处（主规格） |
| `docs/prism_material_pipeline_refactor_verification_zh.md` | 两者 | pcm 2 处 / schema 6 处（doc-native 核验报告） |
| `docs/prism_openpbr_pipeline_function_zh.md` | 仅 schema（4 处） | 字段真源 + §12 位预算风险；不提 pcm |
| `docs/refactoring-dev/01-surface-abi_refactor_zh.md` | 仅 schema | Surface ABI 真源与生成物清单（`docs/refactoring-dev/01-surface-abi_refactor_zh.md:7,107-108`） |
| `docs/material_artifact_implementation_zh.md` | 仅 schema（3 处） | codegen 产物记录 |
| `docs/refactoring-dev/05-render-fabric_refactor_zh.md` | 仅 pcm 边界 | §0.2/§5.2/§8 约束"PCM 不得成为总注册表" |
| `docs/render_fabric_selection_implementation_zh.md` | 仅 pcm 桥 | `:9` 声称 render_fabric `mod.rs` 已公开 PCM 桥 |
| `docs/prism_derived_data_contract_zh.md` | **无**（0/0） | 未提及任一 crate → 本次不产生要求单元 |
| `docs/prism_asset_design_zh.md` | **无**（0/0） | 未提及任一 crate → 本次不产生要求单元 |
| `docs/material_execution_family_validation_zh.md` | **无**（0/0） | 未提及任一 crate → 本次不产生要求单元 |

- **docs 有要求但无对应 crate**：无（范围内每份含 crate 名的文档都能落到上述两个 crate）。
- **crate 无文档指定**：无（`prism_material_pcm`、`prism_material_schema` 均至少被 3 份文档具名规定）。

---

#### 关键结论

- **`prism_material_pcm`：骨架 79.3% / 接通 10.3%（29 单元；落地 5，其中有生产调用方者 3）**。最大未落地项 = **整张注册表没有生产调用方**：`ExtensionRegistry` 的全仓实例化点只有 `pkg/prism_render_material/src/capability.rs:98` 与 `pkg/prism_render_architecture/src/render_fabric/pcm.rs:105`，二者均无生产调用方（唯一引用来自 `pkg/prism_render_material/tests/pcm_parity.rs` 与 `pkg/prism_render_architecture/src/render_fabric/pcm_tests.rs`）；`negotiate`/`negotiate_digest` 在生产路径上**一次都没被调用**，`builtin_registry()` 只被测试调用。PCM 真正接到生产上的只有 `ExtensionSet` 摘要类型 + `stability→required` 派生（`pkg/prism_render_material/src/capability.rs:142-166` ← `src/record.rs:810` → `pkg/prism_render_scene/src/material/buffers.rs:160`）与 `Capability`/`CapabilitySet`（`src/shading_plan.rs:666,1272`，但其入口结构体仅由测试构造）。
- **`prism_material_pcm` 的第二大缺口是"扩展点即声明"的落地形态**：文档 §5.4 要求"每条扩展 = 一份 TOML manifest（codeless）"，而 crate 的 `Cargo.toml` `[dependencies]` 为空、无任何 TOML/文件加载路径，22 行目录硬写在 `pkg/prism_material_pcm/src/builtin.rs`；TOML 真源只存在于 `pkg/prism_material_schema/schema/surface.toml`，两侧靠测试平价而非真加载。§5.4.1（proc-macro → WESL、`golden=auto/none`）、§5.4.2（编辑器插件 / 打包即烘焙 / wasm 沙箱）、§5.5（`cost_hint` 进 CI）在代码中**完全不存在**。
- **`prism_material_schema`：骨架 78.3% / 接通 0%（严格 API 口径；23 单元，落地 12）；生成物进入生产口径 ≈26%（6/23）**。最大未落地项 = **§17.5.3 六类生成物只完成 ① 与 ② 的一半**：`codegen.rs` 仅有两个发射函数（`emit_wesl_unpack:99`、`emit_rust_layout:411`），产物只有 `material_unpack.wesl` 与 `surface_layout.rs`（`src/bin/prism-material-codegen.rs:47-58`），`material_classification`/`material_sample`/CPU 金标准/调试视图表/文档字段表/合规枚举（`ConsumedStep::{Classify,Sample,Golden,Debug,Doc,Compliance}`，`lib.rs:534-555`）**全部无生成物**；`Field.debug_view`（`lib.rs:356`）零消费者；§4.5 约束 1「合规对拍」无任何代码。
- **两处文档-代码矛盾（方向相反）**：① **文档低估代码**——`docs/prism_material_pipeline_design_zh.md:123` 记 `prism_material_schema (2431)`，实际 src+bin 3529 行；`:269` 记 `surface.toml（1255 行）… schema_version = 1、abi_version = 6`，实际 `pkg/prism_material_schema/schema/surface.toml:13,18` 为 `schema_version = 2`、`abi_version = 8`（生成物 `pkg/prism_render_material/src/surface_layout.rs:11,13` 同值），而 `refactor_verification` 仍以 v6 立论、`refactoring-dev/01-surface-abi` 仍以 "abi7" 命名门禁（`:100`）——**三份文档的 ABI 版本号（6/7/8）互不一致**。② **文档错报消费点**——`design §3.6` 称 PCM 被 `prism_render_scene/.../classification_gpu.rs:42` 与 `prism_render_shading/src/classification.rs` 消费，但 `pkg/prism_render_shading/Cargo.toml` **不依赖** `prism_material_pcm`，`classification_gpu.rs:42` 位于 `#[cfg(test)]` 区（`:38`）之后；`docs/render_fabric_selection_implementation_zh.md:9` 称 PCM 桥"不再是仅靠独立 harness 加载的源码"，然而该桥（`pub use pcm::*`，`pkg/prism_render_architecture/src/render_fabric/mod.rs:55`）**没有任何生产调用方**。
- **门禁计数只有一处可静态交叉验证**：`prism_material_pcm` 静态 `#[test]` = `src/tests.rs` 22 + `src/hooks.rs` 6 = **28**，与 `refactor §13.a.1` 记的 28 **静态一致**；`prism_material_schema` 静态 `#[test]` = `src/lib.rs` 18 + `src/codegen.rs` 8 = **26**，与文档记录的 13/15/18/19/20 **静态不一致**（文档快照过期）。全部"通过/全绿"结论本轮**未实测运行**，按题目要求不作为证据。
- **跨 crate 待核风险（本审计范围外的 doc 断言，未核实）**：`docs/prism_openpbr_pipeline_function_zh.md:616` 称 Prism 瓣被 schema 强制从 slot 16 起严格连续（`prism_material_schema/src/lib.rs:730-735`，与代码位置一致），下一个 Prism 瓣必须为 slot 19，而 `bin_class` 的 19 位瓣窗口会静默掩掉它，需要 `BIN_CLASS_BITS 29 → 32` 与一条位预算断言。当前工作树中 `BIN_CLASS_BITS`/`to_bin_class`/`0x7_FFFF` **均无命中**，该机制似已重构（现为 `prism_render_architecture/src/render_fabric/pipeline.rs:269` 的 `PipelineBinClass`），因此该风险是否仍成立 **未核实**。

---

## 2.6 UI 核心（prism_ui*，18 crate）

对照 `docs/prism_ui_loom_design_zh.md`（下称 loom）、`docs/prism_ui_component_kit_design_zh.md`（下称 kit）、`docs/prism_loom_runtime_framework_design_zh.md`、`docs/prism_loom_text_rendering_design_zh.md`，以及 crate 内文档 `pkg/prism_ui/docs/{architecture.md,layers.md,roadmap.md,advanced-features.md}`。

| crate / 机制 | 文档来源 | 声明单元 | 状态 | 关键证据(file:line) | 缺口 |
|---|---|---:|---|---|---|
| `prism_ui` Element 构建器 / Ui 运行时 / Backend / 最小化协调 | loom §3–§6 | 6 | 落地 | `src/element.rs:59`、`src/ui.rs:80,152,163,171-172,413`、`src/backend.rs:35,90` | 无引擎级消费者 |
| `prism_ui_tree` 分代 Arena / LIS keyed diff | roadmap.md:13 | 4 | 落地 | `src/arena.rs:50`、`src/reconcile.rs:75`、`src/tree.rs:192` | — |
| `prism_ui_reactive` Signal·Memo·Effect / GraphSnapshot | loom §10、roadmap.md:79 | 5 | 落地 | `signal.rs:16`、`memo.rs:15`、`effect.rs:14`、`runtime.rs:16`、`introspect.rs:52` | — |
| `prism_ui_macro` `loom!` / `$` / `bind!` | roadmap.md:41,47 | 3 | 落地 | `lib.rs:96,117,171`、`lower.rs:56` | — |
| `prism_ui_macro` hoist 静态提升 | roadmap.md:42 | 1 | **骨架** | `lib.rs:61`（`mod hoist;` **私有**）、`hoist.rs:42`（`pub(crate)`） | 文档写 `prism_ui_macro::hoist`，crate 外不可达，全仓无外部调用 |
| `prism_ui_macro` dualmode | §9.1、roadmap.md:82 | 1 | **骨架** | `lib.rs:60`（私有）、`dualmode.rs:28`（`pub(crate) enum Mode`） | 仅内部语义分类，无对外入口 |
| `prism_ui_store` Store / selector / middleware | roadmap.md:25,67 | 3 | 落地 | `store.rs:24`、`selector.rs:20`、`middleware.rs:18` | **零依赖者** |
| `prism_ui_layout` 5 求解器 | loom §10 | 5 | 落地 | `lib.rs:58-66`、`flex.rs:141` | — |
| `prism_ui_layout` LayoutProtocol + Modifier | §9.4 | 2 | 落地 | `protocol.rs:154`、`modifier.rs:63,76` | — |
| `prism_ui_layout` RelayoutBoundary 增量 | §9.3 | 1 | **部分** | `dirty.rs:22,96`、`incremental.rs:71` | 仅 `tests/composition.rs:33,53` 调用，**无生产调用者** |
| `prism_ui_style` token/class/级联/环检测 | roadmap.md:14 | 4 | 落地 | `token.rs:16,122`、`class.rs:23,230`、`cascade.rs:79,118`、`error.rs:17` | — |
| `prism_ui_style` 选择器 | roadmap.md:14 | 1 | **部分** | `selector.rs:18,135` | 无选择器解析/匹配引擎，实为 `InteractionState`+`Breakpoint`+`MatchContext` |
| `prism_ui_scoped` ScopeId + `@media` | roadmap.md:54-55 | 2 | 落地 | `media.rs:56,62,144` | **零依赖者** |
| `prism_ui_theme` 编译/语义令牌/环检测/Signal 主题/RTL | §9.12 | 5 | 落地 | `compile.rs:114,209`、`semantic.rs`、`reactive.rs:44,56`、`logical.rs:1` | — |
| `prism_ui_text` 分词·字素·UAX#14 | §9.6 | 3 | 落地 | `segmentation.rs:38,50,56`、`Cargo.toml`（unicode-linebreak） | — |
| `prism_ui_text` 可插拔整形 | §9.6 | 1 | 落地 | `lib.rs:79,82`、feature `shaping=["dep:swash"]` | — |
| `prism_ui_text` 缓存·富文本·光标·BiDi | §9.6 | 4 | 落地 | `cache.rs:49,110`、`rich_text.rs:241`、`cursor.rs:138`、`src/bidi/`（8 文件） | **零依赖者，未接渲染后端** |
| `prism_ui_font` 位图+矢量+SDF 图集 | 文本 §7,§9 | 3 | 落地 | `lib.rs:1-3,53,64`、`vector.rs:56,88`、`sdf.rs:44,115` | — |
| `prism_ui_font` swash 字形源 | 文本 §4.1 | 1 | **未落地** | `Cargo.toml:10` 仅 `ab_glyph` | 无 swash 依赖 |
| `prism_ui_font` glyph cache·逐出·GPU MSDF | 文本 §11,§5 | 2 | **未落地** | `lib.rs:13` 自述 "GPU MSDF atlas path … future work" | 无 cache/逐出/GPU 路径 |
| `prism_ui_input` 裁剪命中·三阶段分发·手势竞技场·焦点环 | §9.7 | 4 | 落地 | `hit_test.rs:46`、`dispatch.rs:92-103`、`gesture.rs:40,448`、`focus.rs:29` | 仅 `prism_ui_dock` 使用 |
| `prism_ui_render_backend` 保留场景·合批·分层合成·CPU 栅格 | §9.8 | 4 | 落地 | `scene.rs:52,92`、`batch.rs:18,35,52,87`、`layer.rs:18-22`、`raster.rs:20,116` | — |
| `prism_ui_render_backend` wgpu/Metal GPU + 真机 parity | §9.8 | 1 | **部分** | `gpu.rs:208`（820 行）、`Cargo.toml:21`（`gpu=["dep:wgpu",…]`）、`tests/gpu_parity.rs:1` | **全仓无任何 `Cargo.toml` 启用 `gpu` feature** ⇒ 默认/CI 构建根本不编译该后端，"真机 parity 全绿" 在默认路径不可复现 |
| `prism_ui_ecs` 字段级双向绑定 | roadmap.md:47 | 1 | 落地 | `binding.rs:5-9`、`bridge.rs:10,193` | — |
| `prism_ui_ecs` 调度器集成 | roadmap.md:48 | 1 | 落地（改名） | `schedule.rs:73` 用 `prism_ecs::prelude::{…,Schedule,SystemSet}`、`:179` `add_loom_sync_systems` | 文档写 "Bevy 调度器"，实为 `prism_ecs`；**零依赖者** |
| `prism_ui_ecs` Show/For 结构绑定 | roadmap.md:49 | 1 | 落地 | `structural.rs:74,127,203,331,360,367` | — |
| `prism_ui_component` 组件模型 / 生命周期钩子 | kit §1.1、roadmap.md:66 | 6 | 落地 | `props.rs:26,78,91`、`context.rs:27`、`slots.rs:20`、`lifecycle.rs:60,78,88,100` | 文档 roadmap.md:66 称钩子"规划中" ⇒ 文档过期 |
| `prism_ui_component_kit` P0–P5 控件 | kit §5,§15 | 8 | 落地 | `src/` 12 目录 ≈143 模块、316 `pub struct`、593 tests | — |
| `prism_ui_component_kit` §11 v0.4 A–E | kit §11 | 1 | **部分** | 全仓无 `ActionSheet`/`PullToRefresh`/`NoticeBar`/`SwipeAction`/`IndexBar`/`FocusScope` | 13 项未补齐 |
| `prism_ui_component_kit` feature gate | kit §16 | 1 | **未落地** | `Cargo.toml:20-30` 仅 `default/gallery/std` | 全部控件无条件编译，与文档矛盾 |
| `prism_ui_component_kit` inputs 接 `prism_ui_form` | kit §5.2 | 1 | **未落地** | `Cargo.toml` 无 `prism_ui_form`；`inputs/search_field.rs:8` 等仅注释提及 | 控件与表单/校验未接线 |
| `prism_ui_form` 校验器族 | kit §1.1 | 4 | 落地 | `validator.rs:149,154,159,165,175,183,194`、`email.rs`、`luhn.rs`、`iban.rs`、`form.rs:23,198` | 本应被 kit inputs 消费，实际**零依赖者** |
| `prism_ui_scheduler` 车道·预算·可见性 | §9.5 | 6 | 落地 | `lane.rs:20`、`budget.rs:29`、`apportion.rs:47,101`、`visibility.rs`、`work.rs`、`reconcile.rs` | **零依赖者** ⇒ 整 crate 仅骨架级集成 |

**汇总**

| crate | 声明单元 | 落地/骨架/部分/未落地 | 骨架% | 接通% | bevy 耦合 | 最大缺口 |
|---|---:|---|---:|---:|---|---|
| `prism_ui` | 6 | 6/0/0/0 | 100 | 100 | 无 | 无引擎级消费者 |
| `prism_ui_tree` | 4 | 4/0/0/0 | 100 | 100 | 无 | — |
| `prism_ui_reactive` | 5 | 5/0/0/0 | 100 | 100 | 无 | — |
| `prism_ui_macro` | 6 | 4/2/0/0 | 100 | 67 | 无 | hoist/dualmode 私有不可达 |
| `prism_ui_layout` | 9 | 8/0/1/0 | 100 | 89 | 无 | 增量布局仅测试调用 |
| `prism_ui_style` | 6 | 5/0/1/0 | 100 | 83 | 无 | 无选择器引擎 |
| `prism_ui_scoped` | 2 | 2/0/0/0 | 100 | **0** | 无 | 零依赖者 |
| `prism_ui_theme` | 5 | 5/0/0/0 | 100 | 100 | 无 | — |
| `prism_ui_text` | 8 | 8/0/0/0 | 100 | **0** | 无 | 零依赖者，未接渲染后端 |
| `prism_ui_font` | 6 | 3/0/0/3 | **50** | 50 | 无 | 无 swash / glyph cache / GPU MSDF |
| `prism_ui_input` | 4 | 4/0/0/0 | 100 | 100 | 无 | 仅 `prism_ui_dock` 使用 |
| `prism_ui_render_backend` | 10 | 9/0/1/0 | 100 | **10** | 无 | `gpu` feature 全仓未启用 |
| `prism_ui_ecs` | 5 | 5/0/0/0 | 100 | **0** | 无（依赖 `prism_ecs`） | 零依赖者 |
| `prism_ui_component` | 6 | 6/0/0/0 | 100 | 100 | 无 | — |
| `prism_ui_component_kit` | 11 | 7/0/2/2 | **82** | 55 | 无 | feature gate 未落地、未接 form |
| `prism_ui_form` | 4 | 4/0/0/0 | 100 | **0** | 无 | 零依赖者 |
| `prism_ui_scheduler` | 6 | 6/0/0/0 | 100 | **0** | 无 | 零依赖者 |
| `prism_ui_store` | 3 | 3/0/0/0 | 100 | **0** | 无 | 零依赖者 |

**关键结论**

1. **Bevy 解耦成立**：18 个 crate 的 `Cargo.toml` 均无 `bevy_*` 依赖。唯一涉及 Bevy 语义的 `prism_ui_ecs::schedule` 实际基于 **`prism_ecs`**（`schedule.rs:73`），`roadmap.md:48` 的 "Bevy 调度器集成" 属**命名滞后**而非耦合。
2. **6 个 crate 零工作区消费者**：`prism_ui_ecs` / `prism_ui_text` / `prism_ui_store` / `prism_ui_form` / `prism_ui_scoped` / `prism_ui_scheduler` 无任何 `Cargo.toml` 依赖它们。骨架完整但集成度为 0，与 loom §10「✅ 已交付」的表述落差最大。
3. **文档自称 ✅ 但代码不可达/缺失**：`prism_ui_macro::hoist`（`lib.rs:61` 私有）、`dualmode`（`lib.rs:60` 私有）；`prism_ui_render_backend` 的 wgpu/Metal 后端虽在 `gpu.rs` 存在，但**全仓无任何 `Cargo.toml` 启用 `gpu` feature**，「真机 parity 7/7 全绿」在默认构建路径**不可复现**；kit §16 的 `node_editor/editor/media/qr` feature gate 在 `Cargo.toml:20-30` 中根本不存在。
4. **`pkg/prism_ui/docs` 内部自相矛盾**：`layers.md:83` 称 `@media`/热重载"规划中"、`architecture.md:99` 称 ECS 绑定"规划中"、`layers.md:139` 称编排"规划中"，而 `roadmap.md:54-56` 与代码（`prism_ui_scoped` 28 tests、`prism_ui_ecs/schedule.rs`）均为已交付 ⇒ 同目录较早文档已过期；`roadmap.md:66` 称生命周期钩子"规划中"，但 `lifecycle.rs:88,100` 已实现。
5. **文本栈与渲染栈断链**：`prism_ui_text`（整形/BiDi/缓存，81 tests）**无任何消费者**；`prism_ui_render_backend` 只依赖 `prism_ui_font`（5×7 位图 + ab_glyph 矢量 + SDF 图集），**未接** `prism_ui_text` 的 `ShapedRun`/`GlyphCmd` 字形序列；`prism_ui_font` 未采用文本设计 §4.1 的 swash，且自述 GPU MSDF 为 future work（`lib.rs:13`）。
6. **无引擎级消费者**：除 `pkg/prism_ui*` 自身，全仓 `Cargo.toml` 只在根 workspace members 命中 `prism_ui`；`prism_app` / `examples` 均未依赖任一 UI crate ⇒ **端到端接通止于 UI 内部**。
7. **文档在范围内但无对应 crate**：`docs/prism_loom_runtime_framework_design_zh.md` 的 10 个 `prism_runtime_*`（§20/§22 R1–R12）在 `pkg/` 与根 `Cargo.toml` 中**完全不存在**（`ls pkg | grep -c prism_runtime` = 0）；该文仅以 §20 末行「复用地基：prism_ui* 全家桶 ✅」回指本次审计对象。

---

## 2.7 UI 高级层（16 crate）

**范围/方法**：只读静态核查（read/grep/glob），未执行 cargo build/test。分母＝从文档抽出的 **87 个机制单元**。状态码：**L**落地（有真实实现＋crate 外生产调用方）/ **S**骨架（有真实实现、零生产调用方）/ **P**部分（实现或接入偏离文档）/ **N**未落地。**测试/dev-dependencies 不计入生产接通**。
**文档缩写**：L=docs/prism_ui_loom_design_zh.md；R=docs/prism_loom_runtime_framework_design_zh.md；E=docs/prism_editor_framework_design_zh.md；AF=pkg/prism_ui/docs/advanced-features.md；RM=pkg/prism_ui/docs/roadmap.md；D=docs/prism_ui_dock_design_zh.md；D2=pkg/prism_ui_dock/docs/version/v0.2.0.md；ER=docs/prism_editor_remote_design_zh.md。
**stub 扫描**：16 个 crate 的 src 内 `todo!/unimplemented!/unreachable!/TODO/FIXME` **命中数全为 0**；pub 项 26–366 个/crate。
**Bevy 耦合**：**16/16 均无任何 bevy_* 或 crates/bevy_* 路径依赖**（逐 crate 读 Cargo.toml）。doc 称"复用/保留 Bevy 机制底座"（R:317、E:382-383），但本批 crate 代码层与其零耦合。

#### (a) 机制级核对表
| crate/机制 | 来源 | 要求单元 | 状态 | 证据(file:line) | 缺口 |
|---|---|---|---|---|---|
| anim 缓动/弹簧/时间线/过渡/编排 | L:481,222；RM M4 | Easing、Spring、Timeline、Tween、Choreography | S | pkg/prism_ui_anim/src/easing.rs:36、spring.rs:18、timeline.rs:45、driver.rs:11、choreography.rs:126 | 仅伞 crate 再导出 pkg/prism_ui/src/lib.rs:97 |
| motion 隐式过渡/FLIP/共享元素 | L:506；AF§11 | TransitionTracker、FlipAnimation、SharedElementTransition::from_frames | S | pkg/prism_ui_motion/src/transition.rs:116、flip.rs:24、shared_element.rs:97/121 | 零调用方 |
| async 资源/挂起/边界/竞态/SWR | L:444；AF§9；R:227 | AsyncState、Resource、suspense(_all)、error_boundary/guarded、RaceController、swr | S | pkg/prism_ui_async/src/state.rs:16、resource.rs:21、suspense.rs:33、boundary.rs:69、race.rs:28、swr.rs | 无消费者（sdui/src/decode.rs:24 仅注释） |
| router 路由/守卫/深链接/:param | L:444；AF§4；RM:27 | Location、RoutePattern、RouteTable、Router::navigate、Guard/GuardPoll、DeepLinkState | S | pkg/prism_ui_router/src/path.rs:24、route.rs:32、matcher.rs:42、router.rs:109、guard.rs:64/202、router.rs:31 | 零调用方 |
| virtual 定高/变高/回收池 | L:504；AF§8 | FixedList、VariableList、RecyclePool、virtualize_fixed/variable、Viewport | S | pkg/prism_ui_virtual/src/fixed.rs:13、variable.rs:17、pool.rs:22、virtualize.rs:65/100 | scheduler 仅注释(lib.rs:29)，无依赖 |
| workbench 控件/Story/隔离预览 | L:516；AF§14 | ControlValue/ArgSet、Story/Context、Workbench、render_story | S | pkg/prism_ui_workbench/src/controls.rs:58、story.rs:114、registry.rs:22、harness.rs:80 | 仅 dev-dep＋tests（component_kit/Cargo.toml:17） |
| a11y 角色/语义树/焦点/键盘/直播区 | L:492；AF§10；R:317 | Role、A11yNode/Tree、FocusOrder、KeyboardNav、LiveRegion、setpos/sr | L | pkg/prism_ui_a11y/src/role.rs:18、node.rs:15、tree.rs:20、focus.rs:21、keyboard.rs:78、live.rs:30、setpos.rs:119 | 无（唯一真接通者） |
| i18n 目录/插值/CLDR 复数 | L:500；AF§3；R:241 | LocaleId、Catalog、PluralRules/Category、Args、I18n::translation；t!宏/RTL 自标 PLANNED | S×5＋N×2 | pkg/prism_ui_i18n/src/locale.rs:22/130/235、catalog.rs:26、plural/mod.rs:71、format.rs:70 | t!宏/RTL 无实现；零调用方 |
| devtools 快照/树 dump/OpTrace | L:512；AF§5；R:307 | snapshot、render_tree、OpTrace/summary；时间旅行 UI PLANNED | L×3＋N×1 | pkg/prism_ui_devtools/src/snapshot.rs:106、format.rs:39、trace.rs:19/98 | 消费方全是孤岛 |
| dock 模型/拖拽/RenderPlan/持久化/迁移/实例/a11y/motion/指标 | D§1-16；D2；L:100 | 12 子系统 | L×11＋P×1 | pkg/prism_ui_dock/src/display.rs:49/69、registry.rs:15、workspace_store.rs:274、lib.rs:69-79、motion.rs:13 | tab 窗口自实现(tab_window.rs:6)，非 D:182 所述 prism_ui_virtual |
| hotreload 身份/计划/状态裁剪/样式 diff/监视 | L:491；AF§17 | NodePath、ReloadPlan、StateStore::apply_plan、diff_classes、HotReloader、LoomWatcher/Coalescer | S | pkg/prism_ui_hotreload/src/plan.rs:70、statestore.rs:30/105、style_reload.rs:165、reload.rs:16、watch.rs:112/188 | 零调用方；watch 受 feature 门控(lib.rs:82) |
| inspector 路径/查询/性能/依赖图/ECS | L:513；AF§16 | NodePath、Query、PerfReport/TreeMetrics、DependencyGraph/topo/to_dot、EcsInspectorSession | S | pkg/prism_ui_inspector/src/path.rs:20、query.rs:22、perf.rs:22/130、graph.rs:59/151、ecs.rs:170 | 零调用方（prism_ecs 仅注释） |
| overlay 管理器/z 序/遮罩/FocusTrap | L:503；AF§6；E:195 | OverlayManager/Entry/Kind、on_escape/on_scrim_click、FocusTrap | S | pkg/prism_ui_overlay/src/manager.rs:64/174/192、kind.rs:18、focus.rs:17 | 零调用方 |
| snapshot 序列化/比对/diff/布局 | L:515；AF§13；R:277 | serialize/parse_tree、Snapshot/Comparison、capture/serialize_layout | L | pkg/prism_ui_snapshot/src/serialize.rs:41/142、compare.rs:22、layout_snapshot.rs:84 | 仅 timetravel（自身孤岛） |
| timetravel 时间线/回放/帧/二分 | L:514；AF§15；R:278 | Timeline、ReplayStep/Replayer/diff_between、Frame、bisect | S | pkg/prism_ui_timetravel/src/timeline.rs:25、replay.rs:30/68/115、frame.rs:22、bisect.rs | 零调用方 |
| sdui 白名单/沙箱/版本协商/解码 | L:430,526；R:243；ER:460 | CapabilitySet、Sandbox::sanitize、Version 协商、decode；wire 解码 N | S×5＋N×1 | pkg/prism_ui_sdui/src/capability.rs:19、sandbox.rs:124/197、version.rs:18/56、decode.rs:60 | 自认不做 I/O(lib.rs:26-29)→端到端缺第一跳 |

#### (b) 逐 crate 汇总（类型层% =(87−N)/87；接通%=L/总数）
| crate | 声明单元 | L/S/P/N | 类型层% | 接通% | bevy 耦合 | 最大缺口 |
|---|---|---|---|---|---|---|
| anim | 5 | 0/5/0/0 | 100 | 0 | 无 | 仅伞 re-export |
| motion | 3 | 0/3/0/0 | 100 | 0 | 无 | 零调用方 |
| async | 7 | 0/7/0/0 | 100 | 0 | 无 | 零调用方 |
| router | 6 | 0/6/0/0 | 100 | 0 | 无 | 零调用方 |
| virtual | 4 | 0/4/0/0 | 100 | 0 | 无 | scheduler 未真接 |
| workbench | 4 | 0/4/0/0 | 100 | 0 | 无 | 仅 dev-dep |
| a11y | 6 | 6/0/0/0 | 100 | 100 | 无 | 仅 component_kit |
| i18n | 7 | 0/5/0/2 | 71 | 0 | 无 | t!/RTL 未实现 |
| devtools | 4 | 3/0/0/1 | 75 | 75 | 无 | 孤岛依赖 |
| dock | 12 | 11/0/1/0 | 100 | 92 | 无 | tab 窗口未用 virtual |
| hotreload | 6 | 0/6/0/0 | 100 | 0 | 无 | 零调用方 |
| inspector | 5 | 0/5/0/0 | 100 | 0 | 无 | 零调用方 |
| overlay | 4 | 0/4/0/0 | 100 | 0 | 无 | 零调用方 |
| snapshot | 4 | 4/0/0/0 | 100 | 100 | 无 | 依赖自身孤岛 |
| timetravel | 4 | 0/4/0/0 | 100 | 0 | 无 | 零调用方 |
| sdui | 6 | 0/5/0/1 | 83 | 0 | 无 | 无 wire 解码 |
| **合计** | **87** | **24/58/1/4** | **95.4** | **27.6** | **0** | 12/16 crate 零生产调用方 |

#### (c) 关键结论
1. **类型层接近交付、接通层严重脱节**：无任何 todo/unimplemented，87 单元中 83 个有真实实现（95.4%），但 58 个零生产调用方——**12/16 crate**（anim/motion/async/router/virtual/workbench/i18n/hotreload/inspector/overlay/timetravel/sdui）仅被根 Cargo.toml 列为 workspace 成员。唯一接通：component_kit/Cargo.toml:14＋src 内 84 处引用 a11y、render_backend/Cargo.toml:28 的 dock render feature、devtools/snapshot（依赖它们的 4 个 crate 自身也是孤岛）。
2. **应用层完全未接**：`src/` 与 `pkg/prism_app` 对 `prism_ui*` grep **0 命中**；伞 crate 只依赖 reactive/tree/style/layout/anim/macro（pkg/prism_ui/Cargo.toml:9-15），连 motion/router/i18n/overlay/virtual/async 都未纳入。文档"✅ 已交付"只等于 crate 自测，不等于端到端可用。
3. **doc-claims-✅ 与代码不符**：AF:164 称"守卫与深链接为 roadmap M5 后续增强(PLANNED)"，而 RM:27 与 L:444-450 称已交付；代码侧完整（router/guard.rs:64/202、router.rs:31）→ AF 过时低估。反向：D:182 与 D2 称 tab header"可使用 prism_ui_virtual"，dock 无该依赖、自实现 tab_window.rs:6。
4. **未落地 4 项均有文档自认**：i18n t!宏/RTL（AF:820-830）、devtools 时间旅行 UI、sdui 的"字节→RemoteDocument"wire 层（pkg/prism_ui_sdui/src/lib.rs:26-29），后者使服务端驱动 UI 链路缺第一跳。
5. **本批文档无对应 crate 者**：docs/prism_loom_text_rendering_design_zh.md（其 prism_ui_text/prism_ui_font 不在 16 个内）、docs/prism_editor_remote_design_zh.md（ER:461 明确 prism_editor_remote 尚未创建）。
6. **结论均为静态代码证据，未实测运行**；文档所有测试计数（router 53、dock 120/149/152、sdui 43 等）**未核实**。

---

## 2.8 音频 part 1：core / DSP / 实时路径（16 crate）

**静态审计（read/grep/glob，未运行 cargo build/test，未实测运行）**。基准 docs/prism_audio_engine_design_zh.md（§号已核）。16 个 crate 目录全部存在。计数口径：以文档 §5–§53 中的机制/验收行为「要求单元」。

#### (a) 要求单元逐项核对
| crate | 文档章节 | 要求单元（状态） | 证据 file:line | 缺口 |
|---|---|---|---|---|
| prism_audio_core | §5–13,§46 | 图编译+块渲染 落地／planar buffer+add_scaled 落地／Smoothed+Ramp 落地／Transport+EventScheduler+NamedClock 落地／节点矩阵(140 文件) 落地／声源族(~50 文件) 落地／母带 LUFS+TruePeak+HDR 落地／语音池+虚拟化 落地／算法档 denormal 落地／§12 调制(ControlBus/EnvelopeFollower/Adsr/Curve/Modulator trait) **骨架**／§46.1 UPOLS-NUPOLS 零延迟分区卷积 **未落地**／§46.2 DspKernel+SIMD **未落地**／§46.7 反馈安全图 **未落地**／§46.8 声明式自动混音 **未落地** | pkg/prism_audio_core/src/graph.rs:363,438；buffer.rs:21,255；param.rs:16,51；time.rs:35；scheduler.rs:37,56,208；voice.rs:79,119,299；nodes/mastering/loudness_normalizer.rs:185；true_peak_limiter.rs:404；hdr.rs:284；math.rs:84；modulation.rs:17,34（仅 Lfo）；nodes/reverb/convolver.rs:12（仅时域） | 卷积仍为时域分区，非 §53.1 要求的 FFT 分块 64→2048；无 SIMD 内核抽象；graph.rs 无 feedback/FeedbackDelayNode；无 automix 类目轴 |
| prism_audio_rt | §21,§21.2 | process_block 单一执行序 落地／命令块粒度分层 落地／GraphHandoff capacity-1 落地／RetireQueue 背压+epoch 回收 落地／MPSC+SPSC ArrayQueue 有界 落地／零 unsafe 落地（6/6） | pkg/prism_audio_rt/src/runtime.rs:272；epoch.rs:66,98,126,133；epoch.rs:19 | 无 |
| prism_audio_exec | §29,§30 | topo/levels/islands/liveness+别名/PDC/ExecPlan 纯数据可序列化/wavefront 计划 落地(8)／热交换经命令环交付 RT 未落地／§30(b)(d) Job 化 worker 池+原子完成计数 未落地／NUMA 亲和 未落地／wasm 串行回退计划无 RT 消费方 未落地 | pkg/prism_audio_exec/src/lib.rs:27-35（模块表）；schedule.rs:62；islands.rs；exec_plan.rs；Cargo.toml（依赖仅 serde） | **exec 不依赖 core/rt**，是纯整数图算法；全仓 0 个外部 .rs 引用 |
| prism_audio_output | §48,§53.3 | 48.1 DownmixMatrix(BS.775) 落地／48.2 BassManager(LR 交叉+默认 80 Hz+LFE 校准) 落地／48.3 四档 OutputProfile 落地／53.3 档位默认值 **部分(与代码不符)**／48.4 对白锚定+合规计量遥测回传 **部分(仅 profile.target_lufs)**／48.5 设备协商优雅降级 **未落地** | pkg/prism_audio_output/src/downmix.rs:MINUS_3DB；bass_management.rs:DEFAULT_CROSSOVER_HZ,DEFAULT_LFE_GAIN_DB；profiles.rs:50-104 | §53.3 规定 Home Theater −24 / TV −23 LUFS，代码为 HomeTheater −18（profiles.rs:73）、Tv −16（:81）；§53.3 的 Game 档在代码中不存在；无 device 依赖故无协商 |
| prism_audio_device | §22,§46.3,§53.7 | cpal 后端(feature 门控+默认开) 落地／layout_for_channels+8 种 SampleFormat 分派 落地／BlockRenderer 共享渲染路径 落地／file_sink render_to_wav 32f 确定性 落地／CaptureSink 无锁环形+溢出计数 落地／DeviceState 状态机 落地(6)／AudioWorklet 未落地／DriftResampler(§53.7 ±0.5%/±2 ms) 未落地／§46.3 硬件档 FTZ/DAZ 回调边界 未落地 | pkg/prism_audio_device/src/cpal_backend.rs:78；render.rs:32；file_sink.rs:36；capture.rs:21；backend.rs:18；Cargo.toml:24-25 | 全 crate 无 denormal/FTZ/MXCSR/FPCR 命中；无 DriftResampler；**没有任何 crate 依赖 device**，ecs/lib.rs:5 自述 device-less |
| prism_audio_resample | §34,§53.7 | Resampler 分级 trait+多相 sinc 落地／TimeStretcher 两档(WSOLA+相位声码器) 落地／分数延迟内核 落地／确定性 落地(4)／DriftResampler 未落地 | pkg/prism_audio_resample/src/resampler.rs:101；polyphase_sinc.rs:62；time_stretcher.rs:98；phase_vocoder.rs:74 | 仅 conditioning 依赖，而 conditioning 自身无生产调用者 |
| prism_audio_wave | §43,§53.5 | ARD 求解+烘焙 落地／感知参数编码 落地／探针场+三线性插值(RT 无分配) 落地／动态开口多状态插值 落地／混合权重 落地／WaveBackend 可插拔 落地(6/6) | pkg/prism_audio_wave/src/solver/、bake.rs:120；encoding/；field.rs:33,230；lookup.rs:28；openings.rs；hybrid.rs:29,122；backend.rs:42 | §53.5「与 §31 GPU 射线共用渲染器几何真相」未证实：Cargo.toml 仅依赖 core+spatial+governor |
| prism_audio_governor | §32,§33 | 预算闭环+滞回 落地／LOD 五维(过采样/混响/空间化/调制率/传播档) 落地／优先级重要度 落地／平台功耗档 落地／可观测报告 落地／掩蔽感知虚拟化 落地／HDR 联动 落地／聚类(N 空间化→K) 落地／确定性 落地(10)／对象床预算归并 部分 | pkg/prism_audio_governor/src/governor/mod.rs:105,145；governor/lod.rs:57,90,141,167,199；masking/masking_model.rs:118；masking/hdr_gate.rs；clustering/cluster.rs:80,120 | 仅 wave 依赖 governor，链条末端无生产消费者 |
| prism_audio_spatial | §14,§15,§16 | geometry/attenuation/cone/doppler/air/panner/ambisonics/occlusion/propagation(RT 零分配)/early_reflections/rooms/reverb_zones/spread/multi_position 落地／HOA 编解码+旋转+beamform+nfc 落地(14)／§16 对象音频(床+对象/Atmos 元数据流) 未落地／平台空间后端探测+桥接 未落地 | pkg/prism_audio_spatial/src/propagation.rs:297(trait)；portal_graph.rs:344；nfc.rs；hoa_rotation.rs | 无 atmos/object 模块命中；文档自述「对象/Atmos 与平台后端桥仍在规划」与 §39 ✅ 并存，属文档内部张力 |
| prism_audio_hrtf | §16,§44.3,§46.5 | dataset/interpolation(ITD 对齐)/binaural(overlap-save)/nearfield(分耳视差)/headtracked/hoa_binaural/transaural(串音消除)/**SOFA 二进制解码 落地(8/8)** | pkg/prism_audio_hrtf/src/binaural.rs:55；headtracked.rs:181；transaural.rs:264；sofa/decode.rs:131；nearfield.rs；calibration.rs；personalization.rs | **代码超出文档**：文档反复称「SOFA 二进制解码为后续档」，实际 decode.rs:131 已实现；接通仍为 0 |
| prism_audio_geometry | §31 | PropagationBackend 实现 落地／BVH 加速直达+镜面反射+Maekawa 衍射+高阶反射/衍射 落地／材质表 落地／优雅降级配置 落地(6)／§31(a) 与渲染管线共享同一 BVH/meshlet 部分 | pkg/prism_audio_geometry/src/backend.rs:48,87；lib.rs:5,83,115；direct_path.rs；diffraction_path.rs；higher_order_reflection.rs；material_map.rs | 复用 prism_physics_geometry 的 BVH（Cargo.toml:12），**非** §31 所写的 pkg/prism_render_* 渲染器几何真相 |
| prism_audio_geometry_gpu | §31 | wgpu compute 直达+镜面反射 落地／每 kernel CPU golden parity 落地／场景上传+buffer/layout/params 落地(6)／异步无 RT 阻塞 部分 | pkg/prism_audio_geometry_gpu/src/lib.rs:7；direct.rs；reflection.rs；scene_upload.rs；Cargo.toml:17(wgpu 30) | 同样不复用渲染器 BVH，自建场景上传 |
| prism_audio_physics | §47,§53.8 | §47.5 声学材质耦合 落地(1)／§47.1 接触事件总线 部分(translator.rs:65，但无 ContactEventSource trait、无 §21 命令环下发)／§47.2 模态合成 部分(核心仅有 ModalResonatorNode，无 ModalSynth/ModalBank 模态表资产)／§47.3 摩擦滚滑 部分(roughness.rs)／§47.4 颗粒 部分(核心 GranularNode，无预分配颗粒池/§53.8 512 grains 档)／§47.6 预算档 部分 | pkg/prism_audio_physics/src/translator.rs:65；lib.rs:14-34；cluster.rs；material.rs；pkg/prism_audio_core/src/nodes/effects/modal_resonator.rs:192；granular.rs:257 | 合成侧落在 core 的通用 effects 节点，未形成 §47 的模态表/颗粒池资产与 §32 分档映射 |
| prism_audio_conditioning | §51 | 解码矩阵/响度离线/循环点/瞬态+节拍+标记/离线重采样/编解码分级+delay_trim/HRTF 条件化/Bank 作者化/口型字幕/内容哈希确定性/边界契约 落地(11/11) | pkg/prism_audio_conditioning/src/content_hash.rs:30；hrtf_condition.rs:262；decode.rs；loop_point.rs；transient.rs；tempo.rs；marker.rs；bank.rs；lipsync.rs；delay_trim.rs | §42 备注的默认阈值/交叉淡化窗长待定项在代码中亦未见集中默认档表 |
| prism_audio_profiler | §26 | frame/session/meters/spectrum/voice_monitor/graph_inspect/golden_diff/insights 落地(8/8) | pkg/prism_audio_profiler/src/lib.rs:14-30（逐模块表）；frame.rs；golden_diff.rs；insights.rs | 文档 §26 标「规划」而 §39 标 ✅，属文档内部张力；crate 自述刻意不依赖 rt，故 RT 遥测→帧的映射器尚无（全仓 0 外部引用） |
| prism_audio_comm | §45,§53.6 | 上行链(高通+AEC NLMS+双讲冻结+残余回声抑制+NS+AGC+VAD) 落地／编解码插入点+抖动缓冲+PLC 落地／位置语音参数 落地(2)／FEC 带内抗丢包 未落地／§45.5 VoiceSource/VoiceCapture ECS 组件 未落地／隐私合规授权+录音指示 未落地 | pkg/prism_audio_comm/src/uplink/aec.rs:37,97,152；uplink/{noise_suppress,agc,vad}.rs；transport/{codec,jitter_buffer,plc,packet}.rs；pipeline.rs:78,140 | 无 FEC 命中；comm 与 ecs 中均无 VoiceSource/VoiceCapture 命中 |

#### (b) per-crate 汇总（分母=上述计数单元）
| crate | 声明单元数 | 落地/骨架/部分/未落地 | 骨架% | 接通% | 最大缺口 |
|---|---|---|---|---|---|
| prism_audio_core | 14 | 9/1/0/4 | 71% | 57%(8/14) | §46.1/46.2/46.7/46.8 四项内核深化全缺 |
| prism_audio_rt | 6 | 6/0/0/0 | 100% | 100% | 无 |
| prism_audio_exec | 12 | 8/0/0/4 | 67% | **0%**(12) | 零依赖孤岛，编译计划无任何消费者 |
| prism_audio_output | 6 | 3/0/2/1 | 83% | **0%** | 无 device 协商；默认 LUFS 与 §53.3 冲突 |
| prism_audio_device | 9 | 6/0/0/3 | 67% | **0%**(9) | 无任何 crate 依赖；DriftResampler/硬件 denormal/Worklet 全缺 |
| prism_audio_resample | 5 | 4/0/0/1 | 80% | 0% | 仅被同样孤立的 conditioning 依赖 |
| prism_audio_wave | 7 | 7/0/0/0 | 100% | **0%**(7) | 未与渲染器/GPU 几何共享真相 |
| prism_audio_governor | 11 | 10/0/1/0 | 100% | 0% | 只被 wave 依赖，闭环无真实遥测输入 |
| prism_audio_spatial | 16 | 14/0/0/2 | 88% | **0%**(16) | 对象/Atmos 与平台后端桥缺失；无 RT 宿主 |
| prism_audio_hrtf | 8 | 8/0/0/0 | 100% | **0%**(8) | 无消费者（文档还低估了其完成度） |
| prism_audio_geometry | 7 | 6/0/1/0 | 100% | **0%**(7) | 复用 physics BVH 而非 §31 的渲染器 BVH |
| prism_audio_geometry_gpu | 7 | 6/0/1/0 | 100% | **0%**(7) | 同上；无真机 GPU 调用方 |
| prism_audio_physics | 6 | 1/0/5/0 | 100% | **0%**(6) | 只有「桥」，无模态表/颗粒池资产层 |
| prism_audio_conditioning | 11 | 11/0/0/0 | 100% | **0%**(11) | 完全未接入资产导入管线 |
| prism_audio_profiler | 8 | 8/0/0/0 | 100% | **0%**(8) | RT 遥测→帧映射器缺失 |
| prism_audio_comm | 5 | 2/0/1/2 | 60% | **0%**(5) | ECS 组件与 FEC 缺失 |
| **合计** | **138** | **109/1/11/17** | **88%** | **10%(14/138)** | — |

「骨架」列按「类型存在但无生产调用者」计；接通% 分母为该 crate 单元数，仅 core(8/14)+rt(6/6) 处于 `prism_audio_bevy→prism_audio_ecs→prism_audio_rt→prism_audio_core` 生产链上（pkg/prism_audio_bevy/Cargo.toml:10,14,15；pkg/prism_audio_ecs/Cargo.toml:14,15,16）。

#### (c) 关键结论
1. **类型层近满、接通层近乎断裂**：138 个要求单元中 109 个有真实实现（88% 类型层），但仅 14 个（10%）处在唯一的生产调用链 core→rt→ecs→bevy 上。其余 13 个 crate 的外部 `.rs` 引用数为：exec 0、output 0、device 0、wave 0、physics 0、conditioning 0、profiler 0、comm 0、geometry_gpu 2、hrtf 2、resample 4、governor 4、geometry 12 —— 且这些「外部」引用全部指向同样无消费端的兄弟 crate（geometry_gpu→geometry、wave→governor、conditioning→resample/hrtf）。**exec/output/device/wave/physics/conditioning/profiler/comm 八个 crate 的零非测试调用方是最硬的缺口**。
2. **文档 ✅ 与代码存在 4 处实质背离**（测试数普遍高于文档声称：core 2257 vs 文档 M1+M2≈474、spatial 766 vs 434、conditioning 116 vs 54、hrtf 126 vs 81）：(a) §46.2 DspKernel/SIMD 与 §46.1 UPOLS/NUPOLS 分区卷积在 core 中零命中，convolver.rs:12 自述为「时域分区」；(b) §28 要求的无分配守卫 `#[cfg(feature="rt-alloc-guard")]` 全仓零命中；(c) §22/§53.7 的 DriftResampler 在 device/resample 中零命中；(d) §46.3 把 FTZ/DAZ 明确划给 device 回调边界，device 中零命中（core 侧只有 math.rs:84 的算法档 flush_denormal，已广泛调用）。
3. **反向背离一例**：docs 反复称「SOFA 二进制解码为诚实后续档」，实际 pkg/prism_audio_hrtf/src/sofa/decode.rs:131 `decode_sofa` 与 hdf5.rs 已实现——审计应对 hrtf 的完成度上修而非下修。
4. **§31 的架构承诺未兑现**：文档要求几何 GPU 后端「复用渲染管线 BVH/meshlet 场景，声学与视觉/物理共享同一几何真相」，代码实际复用 prism_physics_geometry 的三角形网格与 BVH（pkg/prism_audio_geometry/Cargo.toml:12），geometry_gpu 则自建 scene_upload.rs，**无任何 prism_render_* 依赖**。
5. **数值默认档与 §53 冲突**：§53.3 要求 Home Theater −24 LUFS、TV −23 LUFS 且含 Game 默认档；代码 profiles.rs:72-103 只有四档且 HomeTheater=−18、Tv=−16，无 Game 档。§53.7/§53.8 的颗粒池 512、模态 8–16 等默认档在 governor 侧也无对应常量表。
6. 结论为静态推断（未实测运行、未执行 cargo）：「接通 0%」由 Cargo.toml 依赖图与全仓文本引用双向确认，但运行时是否经反射/动态注册间接调用未核实；各 crate 测试通过性同样未核实。

#### (d) 本次审计的独立复核
对「八个零消费者 crate」这一结论做了双向复核，结果确认：
- **`.rs` 引用**：以 `grep -rIl --include='*.rs' '\b<crate>\b'`（排除自身目录、`target/`、`.git/`）扫描全仓，`prism_audio_exec` / `device` / `profiler` / `comm` / `wave` / `conditioning` / `output` / `physics` 八个 crate **命中文件数均为 0**。
- **`Cargo.toml` 依赖**：同名扫描（排除自身 `Cargo.toml`）在八个 crate 上唯一命中的是**根 `Cargo.toml` 的 `members` 列举**（`:135,:143,:149,:161,:165,:172,:173,:179`），**没有任何 `[dependencies]` 条目**。
即：这八个 crate 仅被「编译」，不在任何生产调用链上。

---

## 2.9 音频 part 2：内容・创作・资产・沉浸（15 crate）

对照 `docs/prism_audio_engine_design_zh.md`（728 KB，主文档）§11/§12/§13/§18/§19/§20/§21/§23/§36/§37/§38/§39/§42/§44/§46.8/§47/§49/§51/§52 与 `docs/prism_ecs_completion_zh.md`、`docs/prism_derived_data_contract_zh.md`。

| crate | 文档章节 | 声明单元 | 状态 | 关键证据(file:line) | 缺口 |
|---|---|---:|---|---|---|
| `prism_audio_content` | §18(design:615)、§39(design:967) | 9 | 落地（数据层） | `container.rs:138`（Random/Sequence/Blend/Switch/Scatter）、`event.rs:29`、`system.rs:62` | 无 §21 命令环下发 |
| `prism_audio_authoring` | §11(design:510)、§12、§39:964、§42、§46.8 | 4 | 落地 | `patch/compiled.rs:142`（`impl AudioNode for PatchNode`）、`patch/builder.rs:27` | `builder.rs:96` 只产 `PatchDescription`，无命令环/epoch 热切换 |
| `prism_audio_assets` | §20(design:646)、§44.1、§35 | 7 | 部分（6/7） | `bank/registry.rs:52`、`codec/decoder.rs:52`、`streaming/stream_source.rs:117`（真 `std::fs`） | 无热重载；§39:969「待第三方 crate」已过时 |
| `prism_audio_music` | §19(design:632) | 7 | 落地 | `lib.rs:49-58` 全模块齐 | 零非测试消费者 |
| `prism_audio_midi` | §52(design:1517) | 4 | 落地 | `ump/decoder.rs:30`、`mpe/allocator.rs:29`、`mapping/router.rs:159` | 未接 §8 EventScheduler |
| `prism_audio_procedural` | §37、§47 | 6 | 落地 | `modal/synth.rs:45`、`granular/engine.rs:68`、`soundscape/palette.rs:27` | 仅被 `prism_audio_physics` 消费 |
| `prism_audio_immersive` | §49(design:1359) | 3 | 落地 | `adm/bw64.rs:108`、`array/vbap.rs:54`、`eif/import.rs:79` | 无 app 消费 |
| `prism_audio_object` | §44.2(design:1139) | 10 | 落地 | `object.rs:56`、`fold.rs:99`、`platform/spatial_backend.rs:31` | 仅 immersive 消费；无平台真机后端 |
| `prism_audio_haptics` | §36(design:915) | 7 | 落地（类型层） | `backend.rs:48/73/142/200`（三后端）、`bus.rs:45` | 无设备后端实现（无 cpal/hid）；bus 非 §5 图节点 |
| `prism_audio_accessibility` | §23(design:717) | 6 | 落地（类型层） | `caption.rs:129`、`visual_cue.rs:44`、`compression.rs:47` | 遥测环仅注释（`caption.rs:15`） |
| `prism_audio_snapshot` | §13 | 8 | 落地 | `snapshot.rs:59`、`registry.rs:24` | 零消费者 |
| `prism_audio_remote` | §38(design:944) | 7 | 部分（6/7） | `auth.rs:91`、`transport.rs:49/77` | 仅 `InProcessTransport`，无 socket；无 `debug_assertions` 编译剔除 |
| `prism_audio_ecs` | §21/§21.1/§21.2 | 16 | 落地 | `assets.rs:50`、`client_resource.rs:169`（`pump_block`）、`scene_adapter.rs:40`、`player.rs:39-45` | 仅被 facade 依赖，无 app 接入 |
| `prism_audio_bevy` | §39:990、`ecs_completion:17` | 1 | 落地（facade） | `src/lib.rs:11`（`pub use prism_audio_ecs::*`）、`:16-19` | 无任何 crate 依赖它 |
| `prism_audio_codec` | 主文档 **0 次字面提及**；语义见 §44.1/§51 | 5 | 部分（4/5） | `decoder.rs:206`（`impl SourceDecoder`）、`:307-315` | 无 Opus；零依赖方 = 孤儿 |

**汇总**

| crate | 骨架% | 接通% | 最大缺口 |
|---|---:|---:|---|
| `prism_audio_content` | 100% | ~11%（→snapshot） | 无命令环 |
| `prism_audio_authoring` | 100% | 0% | 无热切换 |
| `prism_audio_assets` | 86% | ~43% | 无热重载 |
| `prism_audio_music` | 100% | 0% | 无消费者 |
| `prism_audio_midi` | 100% | 0% | 未接调度器 |
| `prism_audio_procedural` | 100% | 33% | 无 app 链路 |
| `prism_audio_immersive` | 100% | 0% | 无 app 链路 |
| `prism_audio_object` | 100% | 10% | 无平台真机 |
| `prism_audio_haptics` | 100% | 0% | 无设备后端 |
| `prism_audio_accessibility` | 100% | 0% | 无遥测环 |
| `prism_audio_snapshot` | 100% | 0% | 无消费者 |
| `prism_audio_remote` | 86% | 0% | 无 socket |
| `prism_audio_ecs` | 100% | ~20% | 无 app 接入 |
| `prism_audio_bevy` | 100% | 0% | 无人依赖 |
| `prism_audio_codec` | 80% | 0% | 孤儿构建 |

**关键结论**

1. 骨架层近乎完备（15 个中 13 个达 100%，模块名与主文档 §39 表逐项对齐），但**接通层整体 ≈ 0**。全仓 `Cargo.toml` 只有 4 条内部链：content→snapshot（`Cargo.toml:11`）、object→immersive（`Cargo.toml:12`）、procedural→physics、assets→{ecs,conditioning,codec}；`src/`、`examples/`、`tests-integration/` 无任何 `prism_audio*` 引用；`prism_audio_bevy` 自身也无人依赖。
2. **`prism_audio_codec` 是文档-代码错位 finding**：主文档 0 次提及、§39 表不列该行，其 `src/lib.rs:5` 却自述实现 §20/§44.1。且主文档 §39:969 / design:1023「三方 codec 接口已备、待第三方 crate」与代码不符 —— Symphonia 已在仓内（`pkg/prism_audio_codec/Cargo.toml:18-27`），FLAC/Vorbis/MP3/AAC 已接，Opus 明确不支持（`decoder.rs:307` 注释）。
3. **`prism_audio_bevy` 已完全退化为重导出 facade**：`pkg/prism_audio_bevy/src/lib.rs:11` + `:16-19`，全 crate 2 文件 / 187 行，`Cargo.toml` 仅依赖 `prism_audio_ecs`，无 Bevy 运行时。与 `ecs_completion:17`、`ecs_design:490` 一致；风险在于它自身无消费者 ⇒ ECS 音频前端无应用接线。
4. **文档内部自相矛盾**：`ecs_completion:17` 称「30 unit tests」，主文档 `design:163` 称 37，实测 `#[test]` 37。
5. **文档低估而非夸大**（少数反向情形）：`prism_audio_authoring` 声明 83 测试、实测 120（新增 automix §46.8 / pages §42，§39 表未列这两组）；`prism_audio_object` 声明 72、实测 101。其余一致（content 79、music 54、procedural 104、snapshot 57、accessibility 26、haptics 53、remote 37、midi 60、immersive 87；assets 实测 31 vs 文档 30）。
6. **命名/归属错位**：§51 资产条件化实际由 `prism_audio_conditioning` 实现（`lib.rs:56`）；`derived_data_contract:52` 引用 `prism_bake_audio`（`pkg/` 无此 crate）；`bevy_refactor_plan:166` 写 `prism_bevy_audio` 与 `prism_audio_bevy` 不一致；`bevy_refactor_plan:58` 把 `bevy_audio` 映射到 `prism_audio_core`+`device`，不含本批任何 crate。

---

## 2.10 平台・窗口・输入・相机

> 章节号 `2.4` 为占位，合入总报告时按母报告编号调整。
> 审计范围：`pkg/prism_platform`、`pkg/prism_platform_os`、`pkg/prism_window`、`pkg/prism_window_winit`、`pkg/prism_input`、`pkg/prism_camera`、`pkg/prism_camera_ecs`、`pkg/prism_camera_gpu`、`pkg/prism_camera_rig`（9 个 crate，全部存在于根 workspace `Cargo.toml:17+`）。9 个 crate 全部落在仓根 Bevy fork（`Cargo.toml:1-9`，`name="bevy"`、`version="0.20.0-dev"`）的同一 workspace 内。

#### 口径

- **审计性质**：静态证据审计（`read` / `grep` / `glob` / shell grep）。**未实测运行** —— 全程未执行 `cargo build` / `cargo test` / `cargo bench`。文档自述的测试通过数、基准 ns 值、真机验证结论一律标为「文档自述」，本节不重复验证、也不否认。
- **状态码**：
  - **落地** = 文档要求的类型/机制在代码中存在，**且被本 crate 之外的非测试代码调用**（存在生产调用者）。
  - **骨架** = 类型/机制在代码中存在（含 crate 内单测），但本 crate 之外**无非测试调用者**。
  - **部分** = 已实现但子项缺失（下表逐条列出缺什么）。
  - **未落地** = 文档要求，代码中无对应物（已 grep 确认）。
  - **PLANNED** = 文档自己声明为 PLANNED 的单元（仍计入分母，缺口列标明）。
- **接通定义**：被**本 crate 之外的非测试代码**调用。`#[cfg(test)] mod tests`、`tests/`、`benches/` 与 dev-dependencies **一律不算**。
- **骨架%** = (落地 + 骨架 + 0.5 × 部分) ÷ 声明单元数。
- **接通%** = (有已验证生产调用者的单元 + 0.5 × 仅部分接通的单元) ÷ 声明单元数。
- **单元抽取规则**：只取文档内**有明确机制规定**或**明确验收行**的条目（路线图 `验收`/`门禁` 列、`✅/🟡/❌` 状态标记、具名类型清单、`必须` 句式）。散文性论证、参考产品综述不计为单元。
- **未核实**：凡本审计未能静态确认的，行内显式写「未核实」，不作推测。

---

#### 2.10.1 `prism_platform` — 对应 `docs/prism_platform_design_zh.md`（438 行，v0.2）

声明单元数 **22**（口径：§3 档位化 1 + §20 模块布局 1 + §6/§7/§8/§9/§10/§11/§12/§13 机制 8 + §6 mmap·watch 1 + §16 崩溃 1 + §21 契约 1 + §22 基准 1 + §24.1/24.2/24.3/24.5/24.6 五项 5 + §24.4/24.7 PLANNED 2 = 22。§22 M0–M6 里程碑与上述机制单元一一对应，不重复计数）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §3 档位化：`PlatformCaps` 运行时位标志 + 13 个编译期 feature + 锁档 | 部分 | `pkg/prism_platform/src/platform.rs:44`（`pub struct PlatformCaps`）；`pkg/prism_platform/Cargo.toml:12-26` | 文档 `docs/prism_platform_design_zh.md:302` 要求 `["std","backend-win","backend-apple","backend-posix","backend-android","backend-web","crash","dynlib","affinity","hugepages","parking-lot","mock"]` 13 个；实际只有 `std`/`dynlib`/`watch`/`mock`(+`default=["std"]`)。缺 9 个：`backend-*`×5、`crash`、`affinity`、`hugepages`、`parking-lot`；`watch` 文档未列（多出 1） |
| §20 crate 分层与模块布局（含 `backend/{win,apple,posix,android,web,mock}.rs`、`crash/{handler,minidump}.rs`） | 部分 | 文档 `docs/prism_platform_design_zh.md:276-302`；实际 `ls pkg/prism_platform/src/` → `aio atomics.rs capability clock.rs cpu.rs crash dynlib fs lib.rs platform.rs prelude.rs process security stdio.rs suspend.rs tests*.rs thread topology vm wallclock.rs` | **`src/backend/` 目录完全不存在**（OS 代码迁到 `pkg/prism_platform_os/`）；`crash/` 下只有 `context.rs mock.rs mod.rs posix.rs unsupported.rs`，**无 `handler.rs`、无 `minidump.rs`**；计划里的 `vmem.rs`→实际 `vm/`、`process.rs`→实际 `process/`（命名不同但存在） |
| §6 文件系统/VFS：file/dir/path/标准目录 | 骨架 | `pkg/prism_platform/src/fs/file.rs`(162 行)、`fs/dir.rs`(81)、`fs/path.rs`(85)、`fs/dirs.rs`(144)、`fs/tests.rs`(153) | 全 workspace grep `prism_platform::fs::{file,mmap,dirs}` / `prism_platform::Mmap` **零生产调用者**；`pkg/prism_utils/src/reloc/mod.rs:41`、`pkg/prism_utils/src/alloc_/guard.rs:37` 只在注释里指向本模块 |
| §7 高精度时钟：单调 Instant + TSC 快速路径 + 标定 + 回退 + 墙钟 + 高精度 sleep | 落地 | `pkg/prism_platform/src/clock.rs`、`wallclock.rs`(197)；生产调用 `pkg/prism_app/src/time.rs`、`pkg/prism_diagnostic/src/span.rs:14`、`pkg/prism_diagnostic/src/metrics/frame.rs:19`、`pkg/prism_window_winit/src/runner.rs:326` | 墙钟 `wallclock.rs` 无 crate 外调用者（仅单调时钟被用） |
| §8 线程与同步：spawn/tls/affinity/sync/park | 部分 | `pkg/prism_platform/src/thread/{spawn.rs,tls.rs,affinity.rs,sync.rs,park.rs}`；生产调用 `pkg/prism_tasks/src/affinity.rs:22-37`、`pkg/prism_tasks/src/scheduler.rs`（`current_thread_affinity`） | `spawn`/`tls`/`sync`/`park` 无 crate 外调用者；仅 `affinity` 接通 |
| §9 虚拟内存：reserve/commit/decommit/release/大页/对齐/保护页/内存信息 | 落地 | `pkg/prism_platform/src/vm/{mod.rs 497,unix.rs 816,windows.rs 707}.rs`；生产调用 `pkg/prism_ecs/src/storage/arena.rs:40,153-157`（`Reservation`/`huge_pages_supported`/`large_page_size`/`page_size`）、`pkg/prism_tasks/src/arena.rs`（`NumaAllocation`） | — |
| §10 动态库：load/get_symbol/unload + 热重载 + `dynlib` feature | 骨架 | `pkg/prism_platform/src/dynlib/{mod.rs 335,unix.rs 135,windows.rs 106,fallback.rs 34,tests.rs 208}` | 全 workspace grep `prism_platform::Dynlib` / `prism_platform::dynlib` **零生产调用者**；热重载闭环（文档 §10「接 asset/script」）未接通 |
| §6 mmap 零拷贝 + §8 文件监视原生后端（inotify/FSEvents/ReadDirectoryChangesW） | 部分 | `pkg/prism_platform/src/fs/mmap/{mod.rs 342,unix.rs 183,windows.rs 220,fallback.rs 76}.rs`；`fs/watch/{inotify.rs 385,kqueue.rs 355,poll.rs 181,windows.rs 568,mod.rs 328}.rs`；watch 生产调用 `pkg/prism_ui_hotreload/src/watch.rs`（`use prism_platform::{Event, EventKind, WatchBackend, WatchError, Watcher}`） | `mmap` 无 crate 外调用者（仅骨架）；`watch` 已接通 |
| §11 进程/环境/CLI/标准流：argv/env/子进程/stdout/stderr/退出钩子 | 骨架 | `pkg/prism_platform/src/process/{args.rs 59,child.rs 350,env.rs 117,mod.rs 26,tests.rs 169}`、`stdio.rs`(185) | 全 workspace grep `prism_platform::process` / `prism_platform::stdio` **零生产调用者** |
| §12 CPU 探测与拓扑：指令集/缓存行/页大小/拓扑/TSC 频率 | 落地 | `pkg/prism_platform/src/cpu.rs`(137)；生产调用 `pkg/prism_math/src/lib.rs:191`（`CpuInfo::detect()`）、`pkg/prism_tasks/src/lib.rs`（`CpuInfo::detect().logical_cores`） | — |
| §13 原子与屏障：atomic facade + fence + CPU pause/yield（no_std） | 骨架 | `pkg/prism_platform/src/atomics.rs`（21 行） | 无 crate 外调用者 |
| §16 崩溃捕获与 minidump（sigaction/SEH/mach + 线程栈+寄存器+模块+内存转储 + 交 `prism_diagnostic`） | 部分 | 处理器已实现：`pkg/prism_platform/src/crash/mod.rs:234`（`pub fn install`）、`crash/posix.rs:287`（`sigaltstack`）、`crash/posix.rs:304`（`act.sa_sigaction`）、`crash/mock.rs` | **`minidump.rs` 不存在**；`pkg/prism_platform/src/lib.rs:67-68` 代码注释自述「minidump serialization … remain future work」，与文档 `:334`「M0–M6 核心路线图已全部落地并通过验证」、`:326` M6「minidump（三桌面）」**直接矛盾**。另：全 workspace 无非测试代码调用 `crash::install`（grep `install_crash_handler|CrashHandler|install_handler` 在 `pkg/*/src` 排除本 crate 后零命中），交付给 `prism_diagnostic` 的闭环未接通 |
| §20 M6 Web/Android/iOS 后端 + mock 后端 | 未落地 | `pkg/prism_platform/src/crash/unsupported.rs`（仅诚实 `Unsupported` 路径）；`src/backend/` 不存在 | 文档 §14「平台差异矩阵」的 Web/主机列（`:202-214`）无任何实现载体 |
| §21 契约不变量（单实例 Platform / 单调 Instant / caps 只读稳定 / `suspend_resume` 三态 / reserve-commit 配对 / dynlib 卸载安全 / 版本化契约） | 落地 | `pkg/prism_platform/src/suspend.rs`(151)；生产调用 `pkg/prism_app/src/lifecycle.rs`（`SuspendEvent`/`SuspendEventKind`）、`pkg/prism_app/src/runner/winit.rs:142,470`、`pkg/prism_window_winit/src/runner.rs:260-261` | — |
| §22 基准即规格（8 项基准：取时成本/单调性/锁争用/绑核/页分配/mmap 零拷贝/CPU 探测/minidump 可符号化） | 部分 | `pkg/prism_platform/Cargo.toml:28-30`（`[[bench]] name = "vm_commit"`） | 仅 1 个 bench；`minidump 可符号化` 基准不可能存在（无 minidump） |
| §24.1 高级异步 I/O（`AioQueue`/`ReadOp`/`IoPriority`；Apple lio_listio / Linux io_uring / Windows IOCP）🟡 | 落地 | `pkg/prism_platform/src/aio/{mod.rs 335,posix.rs 405,uring.rs 777,windows.rs 509,tests.rs 779}`；生产调用 `pkg/prism_tasks/src/async_exec/io_bridge.rs`（`use prism_platform::aio::{AioError, AioQueue, IoPriority, ReadOp, WriteOp}`）、`pkg/prism_world_io/src/reactor.rs:32` | 文档自述「零拷贝/对齐直读 + I/O 带宽预算」仍 PLANNED |
| §24.2 虚拟内存进阶 `MirroredRing`（magic ring buffer + `VmError::Unsupported` + `is_supported()==false`）🟡 | 骨架 | `pkg/prism_platform/src/vm/ring.rs`(141) | 无 crate 外调用者；文档自述 Linux `memfd_create`、稀疏虚拟堆、GPU 共享内存仍 PLANNED |
| §24.3 混合核/NUMA/能耗（`CpuTopology`/`TopologyBuilder`/`CoreKind`/`qos_hint`/`PowerState::policy` + 兄弟 crate `prism_platform_os::probe_topology()`）✅ | 落地 | `pkg/prism_platform/src/topology/{mod.rs 478,qos.rs 116,power.rs 120}.rs`；生产调用 `pkg/prism_tasks/src/numa.rs`（`prism_platform::topology::{CoreKind,CpuTopology}`）、`pkg/prism_tasks/src/topology/descriptor.rs:34,218` | 真实 `pthread_set_qos_class_self_np`/`SetThreadInformation`/`setpriority` 调用文档自述 PLANNED（代码侧 `qos.rs` 只产出策略数值） |
| §24.5 安全加固（`Mitigation` 10 项 + `SigningStatus` + `IntegrityPolicy` + `SandboxModel` + `SecurityPosture::detect()`）🟡 | 骨架 | `pkg/prism_platform/src/security/{mitigation.rs 197,signing.rs 164,sandbox.rs 160,mod.rs 239}.rs`；真实探针 `pkg/prism_platform_os/src/security/mod.rs`(227) | `SecurityPosture::detect()` 无 crate 外调用者；仅 crate 内 `tests_security.rs`(296) |
| §24.6 能力数据库（`CapabilityDatabase::from_platform` / `require().select()` / 字节稳定 `report()`）✅ | 骨架 | `pkg/prism_platform/src/capability/{catalog.rs 129,database.rs 255,require.rs 130,report.rs 74,support.rs 92}.rs` | 无 crate 外调用者 |
| §24.4 跨进程 Crashpad + 稳定堆栈哈希 + 可控转储粒度 + 离线符号化 | 未落地（PLANNED） | `pkg/prism_platform/src/lib.rs:67-68` 自述 future work | 文档 `:390-398` 自述 PLANNED；代码零对应物 |
| §24.7 Web/主机后端进阶（SAB+Web Worker pool / IndexedDB·fetch / console 内存域 / console IO） | 未落地（PLANNED） | 无 | 文档 `:429-434` 自述 PLANNED；代码零对应物 |

**小计**：落地 6 / 骨架 7 / 部分 6 / 未落地 3（其中 2 为文档自述 PLANNED）。
**骨架% = (6 + 7 + 0.5×6) ÷ 22 = 16.0/22 = 73%**；**接通% = (6 + 0.5×2) ÷ 22 = 7.0/22 = 32%**（接通单元：§7 时钟、§8 affinity、§6 watch、§9 虚拟内存、§12 CPU、§21 契约、§24.1 aio、§24.3 topology）。

---

#### 2.10.2 `prism_platform_os` — 对应文档片段（无独立设计文档）

**先答一个范围问题**：`docs/` 下**没有** `prism_platform_os_design_zh.md`。本 crate 的规格分散在 4 份文档里：`docs/prism_platform_design_zh.md:377-386`（§24.3 兄弟 crate 段）、`:413-417`（§24.5 真机探针）、`:438`（§24.8 诚实边界）、`docs/prism_window_refactor_zh.md:60`（OS 能力层复用边界）、`docs/prism_tasks_design_zh.md:472`（要求 `probe_topology()` 且 `is_probed()==true`）、`docs/prism_ecs_design_zh.md:429`（拓扑/绑核回读验收条目）。

声明单元数 **3**。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §24.3 真机拓扑探针：`probe_topology()`、`is_probed()==true`、Apple M2 8 逻辑核 = 4P+4E、非支持平台返回 `ProbeError::Unsupported` | 落地 | `pkg/prism_platform_os/src/topology/mod.rs:157`（`pub fn probe_topology`）、`:43`（`probe_numa_nodes`）、`:92`（`bind_current_thread_to_node`）、`:173`（`mod linux`）、`:321`/`:373`（两处 `probe()` 实现）；`pkg/prism_platform_os/src/sysctl/mod.rs`(114)、`src/tests_topology.rs`(162)；**生产调用者** `pkg/prism_tasks/src/numa.rs:105`（`prism_platform_os::probe_topology().unwrap_or_else(|_| prism_platform::topology::CpuTopology::detect())`，位于 `impl Topology { pub fn detect() }`；`mod numa` 见 `pkg/prism_tasks/src/lib.rs:108`，未 feature-gate） | — |
| §24.5 真机安全探针（`_dyld_get_image_header` MH_PIE / arm64e / `csops(CS_OPS_STATUS)` CS_VALID / CS_RUNTIME） | 骨架 | `pkg/prism_platform_os/src/security/mod.rs`(227)、`src/tests_security.rs`(86)、`src/lib.rs:46`（`pub use security::probe_security`） | 无 crate 外调用者 |
| §24.3 其余 OS 后端（Linux `/sys`、Windows `GetLogicalProcessorInformationEx`）+ 真实 qos/power syscall 调用 | 部分 | Linux 已具备：`pkg/prism_platform_os/src/topology/mod.rs:44,93,158`（`#[cfg(target_os = "linux")]` 下 NUMA 节点、`allowed_cpus`、`probe`）；`pkg/prism_platform_os/Cargo.toml:6` 描述自述 Linux sysfs + 绑核 | **Windows 后端零实现**；`qos`/`power` 的 OS 调用零实现（`pkg/prism_platform_os/src/` 只有 `error/lib/security/sysctl/topology`）。**文档滞后于代码**：`docs/prism_platform_design_zh.md:383-385` 称「Linux /sys + Windows … backend PLANNED」「only verified aarch64 macOS else `ProbeError::Unsupported`」，与 `topology/mod.rs:44,93,158` 的 Linux 实现矛盾 |

**小计**：落地 1 / 骨架 1 / 部分 1。
**骨架% = 83%**；**接通% = 1/3 = 33%**（唯一生产调用点 `pkg/prism_tasks/src/numa.rs:105`）。
**附注**：`pkg/prism_platform_os/Cargo.toml` 无任何 `bevy_*` 依赖；`libc` 仅 `[target.'cfg(target_os = "macos")'.dependencies]` 下（`Cargo.toml:13-14`）。

---

#### 2.10.3 `prism_window` — 对应 `docs/prism_window_refactor_zh.md`（484 行，Draft 架构提案）

声明单元数 **17**（§9 路线图 P0/P0.5/P1/P2/P3 共 5 单元 + §0 硬约束 1 + §4.2/4.3/4.4/4.5/4.6 共 5 + §5.1/5.2/5.3/5.4 共 4 + §6 低延迟 1 + §7 golden harness 1 = 17）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §9 P0 契约奠基：破坏性改 `apply(&mut self, WindowEventEnvelope) -> bool`、`WindowRegistry`、事件信封 | 落地 | `pkg/prism_window/src/envelope.rs:103`（`PlatformEventStamp`）、`:141`（`WindowEventEnvelope`）、`:87`（`EventSource`）、`:26`（`MonotonicTimestamp`）、`:60`（`PlatformEventSequence`）；`src/registry.rs:23`（`WindowRegistry`）、`:124`（`apply`）、`:141`（`apply_sorted`）；生产调用 `pkg/prism_app/src/runner/winit.rs:33-38`（`use prism_window::{envelope, event, geometry, WindowId}`） | 未做到文档 §7.1 的验收（见 §7 golden 行） |
| §9 P0.5 首个后端 → `prism_window_winit`，真实开窗/多窗口路由/resize+scale 回灌 | 落地 | crate 存在（`pkg/prism_window_winit/`，1893 行）；生产调用 `pkg/prism_app/src/runner/winit.rs:33-38`（feature `winit`）、`pkg/prism_render_app/src/presentation.rs:27,788,791`（`pkg/prism_render_app/Cargo.toml:27` 对 `prism_window` 为非 optional） | 仅 feature `winit` 下接通（`prism_app/Cargo.toml` 的 `winit` feature 默认关闭）；真实开窗未核实（未运行） |
| §9 P1 多窗口 + 显示器：`WindowRegistry` 多条目、`parent`、`Monitor` 扩展 | 部分 | `pkg/prism_window/src/registry.rs:23-114`（多条目 BTreeMap 语义）；`src/monitor.rs:33-49`（`Monitor`） | `Monitor` 缺 `work_area`（grep `work_area\|WorkArea` 于 `pkg/prism_window/src/` → 零命中） |
| §9 P2 次世代显示：`HdrMetadata`/`HdrConfig`/`VrrMode`/`LowLatencyHint`/`FramePacer` | 未落地 | 仅 `HdrConfig` 落地（`pkg/prism_window/src/capabilities.rs:51`）；`HdrMetadata`、`VrrMode`、`LowLatencyHint` grep 于 `pkg/prism_window/src/` → **零命中** | 内核缺 3 个词汇表类型；`FramePacer` 不在内核，而在 `pkg/prism_app/src/pacing.rs`（跨 crate 分裂） |
| §9 P3 远期：SDL3/原生/主机/无头后端并存 + 独占全屏 + `VideoMode` 绑定 | 部分 | 无头/重放 crate 不存在（`ls -d pkg/prism_window_headless pkg/prism_window_replay` → No such file or directory）；独占全屏降级为无边框：`pkg/prism_window_winit/src/command.rs`（`Fullscreen::Borderless(None)`，单测名 `windowed_has_no_fullscreen_and_fullscreen_is_borderless`）；文档自述 `docs/prism_window_refactor_zh.md:484`「exclusive fullscreen video-mode enumeration」为下一步 | 4 类后端 0 实现；`VideoMode` 未与全屏绑定 |
| §0 硬约束：纯/确定性/`no_std + alloc`/**零依赖**/无 `unsafe` | 落地 | `pkg/prism_window/Cargo.toml`（无 `[dependencies]` 段）；根 `Cargo.toml` `[workspace.lints.rust] unsafe_code = "deny"`；`pkg/prism_window` 各源文件无 `unsafe` | — |
| §4.2 破坏性 ABI 词汇表（`WindowCommand`/`WindowCommandEnvelope`/`WindowCommandResult`/`CommandStatus`/`WindowCapabilities`/`WindowAttributesDelta`） | 部分 | `src/command.rs:193-247`（`WindowCommand` 26 变体，含 `SetIme` `:226`、`SetHdr` `:227`）、`:265`（`WindowCommandEnvelope`）；`src/result.rs:95`（`CommandStatus`）、`:183`（`WindowCommandResult`）；`src/capabilities.rs:83`（`WindowCapabilities`） | **`WindowAttributesDelta` 不存在**（grep 于 `pkg/prism_window/src/` → 零命中，文档 `:193` 明确要求 changed-fields-only 类型）；`WindowCommandEnvelope`/`WindowCommandResult`/`CommandStatus`/`RealizedWindowDelta` **在 `pkg/`+`os-backends/` 全域无非测试消费者**（grep 排除 `pkg/prism_window/src/` → 零命中；`pkg/prism_window/src/tests.rs` 内亦零命中），即「命令出/结果回」这条 ABI 只有类型没有使用者；`WindowCapabilities` 无生产者也无消费者 |
| §4.3 三态 Desired/Applied/Realized | 落地 | `pkg/prism_window/src/window.rs:15-46`（`WindowAttributes` = desired）；`src/registry.rs:23-114`；后端侧 `pkg/prism_window_winit/src/sync.rs`（`WindowSync::{desired,applied,realized}` + `diff()`）；生产消费 `pkg/prism_render_app/src/presentation.rs:791`（`backend.realized(id)`） | — |
| §4.4 内核不依赖 `raw_window_handle`；后端实现 `HasWindowHandle`/`HasDisplayHandle` 并暴露 `WindowId`→handle | 部分 | 内核侧满足：`pkg/prism_window/src/` grep `raw_window_handle` → 零命中（零依赖即证）；后端侧 `pkg/prism_window_winit/src/runner.rs:137`（`pub fn window(&self, id) -> Option<Arc<WinitWindow>>`） | 暴露的是 `Arc<winit::Window>` 裸句柄而非内核化的「handle 租约」；**无 retire 状态机**（文档 §28 风险行「handle lease + retire」）；`WindowId`→handle 无类型化映射类型 |
| §4.5 单属主 `WindowRegistry`（无 `Arc<Mutex>/RwLock`）+ 两条 SPSC ring + frame-fence drain-to-batch + 背压合并 | 部分 | `pkg/prism_window/src/registry.rs` 直接持有（无内部锁，满足单属主）；`pkg/prism_window_winit/src/runner.rs:150`（`drain_events`） | **无任何 SPSC 环**（`pkg/prism_window/src/`、`pkg/prism_window_winit/src/` grep `Ring\|Spsc\|spsc` 无环实现）；后端用 `HashMap` + `Vec` 缓冲；无背压/容量上限策略 |
| §4.6 事件合并表：Resized/Moved/ScaleFactorChanged/CursorMoved last-wins；CloseRequested/Destroyed 永不合并 | 落地 | `pkg/prism_window_winit/src/batch.rs:19-30`（`CoalesceKind::{Resized,Moved,CursorMoved,ScaleFactor}` + 映射）、`batch.rs:105-115`（单测断言 `CloseRequested` 存活） | 合并实现在后端而非内核（文档把合并列为内核/后端契约，位置可争议） |
| §5.1 `WindowAttributes.parent: Option<WindowId>` | 未落地 | `pkg/prism_window/src/window.rs:21-46` 字段清单无 `parent`；grep `parent` 于 `pkg/prism_window/src/` → 零命中 | 父子窗口/工具窗整条能力缺失（文档 `:255`） |
| §5.2 `Monitor` 增 `work_area`（排除任务栏）、`position`（虚拟桌面）、`scale_factor_milli`（整数缩放） | 部分 | `pkg/prism_window/src/monitor.rs:33-49`（`Monitor` 已有 `position`；`scale_factor` 为 `f32`） | `work_area` 缺；`scale_factor` 仍是 `f32` 而非 `scale_factor_milli`——文档 `:259` 要求**全部缩放走整数 milli 以消除 `f32` 非确定性**，内核未满足 |
| §5.3 HDR/色域：`HdrMetadata{max_luminance_nits,min_luminance_milli_nits,max_full_frame_nits,color_gamut,eotf}`、`HdrConfig`、`Monitor.hdr` | 部分 | `pkg/prism_window/src/capabilities.rs:51`（`HdrConfig`）、`:83`（`WindowCapabilities`）；`pkg/prism_window/src/lib.rs:44,67`（`HdrMode`/`HdrSupport` 导出） | `HdrMetadata`、`ColorGamut`、`Eotf` grep 全域 → **零命中**；`Monitor.hdr` 缺；后端 `SetHdr` 为显式 no-op（`pkg/prism_window_winit/src/command.rs` 的 `SetPresentMode\|SetHdr\|SetIme\|Create\|Destroy` 空分支） |
| §5.4 VRR：`VideoMode.refresh_rate_range_millihertz`、`WindowAttributes.vrr: VrrMode` | 未落地 | `pkg/prism_window/src/monitor.rs:14-21`（`VideoMode{size,bit_depth,refresh_rate_millihertz}` 无 range）；grep `VrrMode\|refresh_rate_range_millihertz` 全域 → 零命中 | VRR 整条能力（文档 `:283-286`）无类型、无命令、无后端路径（`SetVrr` 亦不存在） |
| §6 低延迟 present：`LowLatencyHint`、`FramePacer` 依刷新率/VRR range 算 submit 截止、`t_sample…t_present` photon 延迟直方图 | 部分 | `FramePacer` 存在但在 `pkg/prism_app/src/pacing.rs`（`pkg/prism_app/src/runner/winit.rs:45,115,225,296`：`pacer` 字段 + `next_wakeup` 返回 `Instant`） | `LowLatencyHint` grep 全域 → 零命中；无 `t_*` 时间戳链、无 photon 延迟直方图；`refresh_rate_millihertz`→deadline 的纯策略不在内核 |
| §7 确定性 golden harness（录制→双重重放字节级一致、x86/ARM 跨架构一致、5 场景语料库、性质测试、fuzz、CI 红线） | 未落地 | 仅 `pkg/prism_window/src/tests.rs`(265 行)；全 workspace 的 golden 产物在别的 crate（`pkg/prism_physics_core/src/cache/golden.rs`、`pkg/prism_render_scene/tests/water_hybrid_golden.rs`、`pkg/prism_render_architecture/src/water/frame_golden.rs`、`pkg/prism_audio_profiler/src/golden_diff.rs`），窗口侧零个 | 无录制器、无重放器、无 golden 语料、无跨架构对拍门禁；这是文档 §9 P0 的**验收条件本身**（`:374`） |

**小计**：落地 5 / 部分 8 / 未落地 4 / 骨架 0。
**骨架% = (5 + 0.5×8) ÷ 17 = 9.0/17 = 53%**；**接通% = 3/17 = 18%**（接通：P0 信封/事件/几何、P0.5 后端被消费、§4.6 合并作用于 `WindowEventEnvelope`。**`WindowRegistry` 与全部 命令/结果 信封均无 crate 外消费者**）。

---

#### 2.10.4 `prism_window_winit` — 对应 `docs/prism_window_winit_design_zh.md`（1906 行，纯设计文档，**全文无 `✅/🟡/❌` 状态标记**）

声明单元数 **24**（§26 路线 M0–M5 共 6 单元 + §4.1/4.2/5/6/7/9/10/11/12/13/14/15/16/17/18/19/23/27 共 18 机制单元）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §26 M0 最小桌面闭环 | 落地 | crate + 1893 行：`pkg/prism_window_winit/src/{lib,runner,convert,id,translate,sync,batch,command,error}.rs`；`runner.rs:108`（`request_window`）、`:117`（`request_destroy`）、`:354`（`WinitRunner::run`）、`:300`（`WinitApp`）；生产消费 `pkg/prism_app/src/runner/winit.rs`、`pkg/prism_render_app/src/presentation.rs:791` | 验收行里的「keyboard+mouse button/motion/wheel basic translation」未达成（见 §11 行）；真机 Windows/macOS 未核实（未运行） |
| §26 M1 命令/事件 ABI + 多窗口 + Fake backend + Golden | 部分 | 多窗口：`runner.rs:156`（`window_count`）、`:162`（`is_empty`）；事件信封：`runner.rs:144,150` | **命令结果 ABI 未接**：`pkg/prism_window_winit/src/` grep `WindowCommandEnvelope\|WindowCommandResult\|CommandStatus` → 零命中；无 Fake 后端、无记录/重放；`WindowBackend` 抽象不存在（只有具体 `WinitRunner`） |
| §26 M2 完整输入与桌面体验（raw mouse/IME/touch/gesture/drag-drop/自定义光标/AccessKit/X11·Wayland 差异/Backend Inspector） | 未落地 | `pkg/prism_window_winit/src/translate.rs:71-75,99-103` —— `KeyboardInput`、`ModifiersChanged`、`Ime`、`MouseWheel`、`MouseInput` 全部落在**丢弃分支**；`translate.rs:95`（`is_input_device_event`，注释 `:67` 称「routed by prism_input」）但 **全 crate 无调用者** | 键盘/鼠标按键/滚轮/IME/触摸/手势/拖放 全部无翻译；无 `accesskit` 依赖、无 Inspector、无 X11/Wayland 差异诊断 |
| §26 M3 AAA 呈现协作（RHI realized present/HDR/VRR handshake、frame pacing、resize storm、混合 DPI/HDR/VRR 真机矩阵） | 未落地 | `command.rs` 中 `SetPresentMode\|SetHdr` 为 no-op；无 VRR/HDR 类型可传（内核亦缺） | 整层缺失；无真机矩阵 |
| §26 M4 Web 与移动生命周期（CanvasBinding/ResizeObserver/pointer lock+fullscreen gesture/iOS·Android 生命周期/安全区/虚拟键盘） | 未落地 | 无任何 web/ios/android 源文件；`pkg/prism_window_winit/Cargo.toml` 无对应依赖 | 整层缺失 |
| §26 M5 规模化与替代后端验证（Pipelined runner / 外部宿主窗口 / headless·replay ABI / SDL·fake adapter / 空闲功耗·8kHz·64 窗口 stress gate） | 未落地 | 无 pipelined 模式、无 `prism_window_headless`/`prism_window_replay` crate（`ls` 均不存在）、无 SDL adapter | 整层缺失；「证明 winit 语义不外泄」的门禁无从执行 |
| §4.1 模块计划（`plugin.rs`/`state.rs`/`registry.rs`/`event.rs`/`convert/`/`monitor.rs`/`handle.rs`/`pacing.rs`/`accessibility.rs`/`diagnostics.rs`/`recording.rs`/`platform/{macos,windows,linux,web,ios,android}.rs`，文档 `:260-296`） | 未落地 | 实际模块：`batch command convert error id lib runner sync translate` | 文档计划的 12+ 文件/目录**一个都不存在**；`accessibility.rs`、`diagnostics.rs`、`recording.rs`、`pacing.rs`、`platform/` 全缺 |
| §4.2 Feature 建议（`x11`/`wayland`/`web`/`accessibility`/`serde`/`recording`/`trace`/`custom_cursor`/平台扩展，文档 `:304-308`） | 未落地 | `pkg/prism_window_winit/Cargo.toml` **完全没有 `[features]` 段** | 零 feature 门控；平台裁剪/无障碍/录制均不可编译期选择 |
| §5 命令/事件 ABI（`WindowCommandResult` 回执、批次、ABI 版本化） | 部分 | 后端自有三态：`src/sync.rs`（`WindowConfig` desired / `RealizedState` applied / `WindowSync::diff()`）；`src/command.rs`（`pub fn apply(window, &WindowCommand)`）+ `runner.rs:228-240`（`flush_commands` 逐条 `command::apply` + `mark_applied`） | 不产出 `WindowCommandResult`/`CommandStatus`（批次回执 ABI 未实现）；无 ABI 版本号 |
| §6 所有权与线程模型（主线程独占 EventLoop/原生窗口、默认集成模式、可选流水线模式、唤醒、无全局可变状态） | 部分 | 单线程主属主：`runner.rs`（`WinitBackend` 无非 `Send` 共享态、无全局静态）；唤醒/循环：`runner.rs:429,431`（`ControlFlow::WaitUntil(deadline)` / `Wait`）；`WinitApp` 有 `resumed/suspended/low_memory` 钩子（`runner.rs:248+`） | 无可选流水线模式；无显式 wakeup proxy 类型（依赖 winit 自带） |
| §7 EventLoop 与 App 生命周期（winit 0.30 状态机、`resumed`/`suspended`/`about_to_wait`/退出/panic·致命错误） | 部分 | `runner.rs:260-261`（`SuspendEventKind::{Resumed,Suspended}` 分派）、`runner.rs:412`（`about_to_wait`）、`src/error.rs`(57)、`FrameOutcome::{Continue,Exit}`（`runner.rs:44`） | panic/致命错误的宿主可见路径未核实；无 `run_on_demand` 类 API |
| §9 Desired/Applied/Realized 三态与差量应用顺序 | 落地 | `src/sync.rs`（`WindowSync` 三态 + `diff()`/`is_dirty`/`mark_applied`/`observe`）；生产消费 `pkg/prism_render_app/src/presentation.rs:791`（读 `realized(id).occluded`） | 文档 §9.2 的 7 步 realize 顺序归本 crate，代码未显式分步（合并进 `diff()`） |
| §10 事件转换与批处理（不可合并/可合并、队列过载、**每帧零分配**） | 部分 | `src/batch.rs:19-30`（4 类可合并）、`runner.rs:412`（每帧 `coalesce` 后再 `app.frame`） | `batch::coalesce` 用 `HashMap` + `Vec` 建每帧临时结构 → 与 §10.5「每帧零分配」、§27「内存稳态 30 分钟无增长」冲突；无队列过载/背压策略 |
| §11 输入/IME/指针/手势（键盘/文本 IME/指针/滚轮触摸板/触摸与笔/文件拖放） | 部分 | 仅指针位置：`src/translate.rs:57`（`CursorMoved`）、`translate.rs:60-64`（`CursorEntered`/`CursorLeft`） | 键盘、IME、滚轮、触摸、笔、文件拖放 **全部为丢弃分支**（`translate.rs:71-75,99-103`）；`is_input_device_event`（`:95`）**无调用者** → M0 的输入验收不成立，且 `prism_input` 没有任何 `winit→InputEventEnvelope` 生产者 |
| §12 DPI/坐标/多显示器（坐标空间、ScaleFactorChanged、用户缩放 vs 后端缩放、显示器身份、热插拔、视频模式选择） | 部分 | `translate.rs`（`ScaleFactorChanged` 映射）、`sync.rs`（`RealizedState.scale_factor_milli`）、`runner.rs:137` | 无显示器枚举/身份/热插拔 API（`backend.window(id)` 之外无 monitor 查询）；视频模式选择缺（全屏降级无边框） |
| §13 光标与捕获（状态模型/自定义光标/捕获丢失/策略） | 部分 | `src/command.rs`（`apply_cursor` + cursor 相关命令分支） | 无捕获丢失事件、无自定义光标 feature、无游戏/编辑器策略 |
| §14 Surface 句柄与渲染握手（raw handle 非长期资源、Surface 状态机、resize storm、HDR、VRR+Present） | 部分 | `runner.rs:137`（`window()` 返回 `Arc<WinitWindow>`，winit 类型实现 `HasWindowHandle`；注释即写明「for building a render surface via `raw-window-handle`」）；消费方 `pkg/prism_render_app/src/presentation.rs:791`（`Arc::clone(&window)` 挂到渲染请求） | 无 Surface 状态机/epoch/`Surface lost` 建模；resize storm 合并由渲染侧自处理；HDR/VRR 握手无对应内核类型 |
| §15 Frame Pacing/低延迟/功耗（Update Policy、ControlFlow 选择、多窗口 redraw、低延迟标记、Reflex、时间预算） | 部分 | `runner.rs:429`（`ControlFlow::WaitUntil`）、`pkg/prism_app/src/runner/winit.rs:225`（`next_wakeup` → `crate::presentation::next_window_wakeup(&self.app)`） | 无 `WindowLoopPolicy`（文档 `:1673` 具名方案）——grep `WindowLoopPolicy` 于 `pkg/prism_app`、`pkg/prism_window`、`pkg/prism_window_winit` → **零命中**；无 `LowLatencyPolicy`（文档 `:1098`）；无 Reflex；无 `RedrawRequested` 处理（grep `RedrawRequested` 于生产路径仅见于 `translate.rs` 丢弃分支） |
| §16 多窗口与编辑器级能力（窗口角色、退出策略、布局持久化、数量治理、嵌入与外部宿主） | 部分 | 多窗口已具备：`runner.rs:108,117,156,162`（多窗口注册 + 独立创建/销毁 + 引用计数句柄） | 窗口角色/布局持久化/数量治理/嵌入宿主全缺 |
| §17 无障碍（分层、多窗口语义树、性能、正确性） | 未落地 | `pkg/prism_window_winit/Cargo.toml`、`pkg/prism_window/Cargo.toml` grep `accesskit` → 零命中 | 整层缺失（文档 §17 与 §26 M2 均要求 AccessKit 多窗口桥） |
| §18 平台能力矩阵（Windows/macOS/X11/Wayland/Web/iOS·Android/支持级别） | 未落地 | 无 `src/platform/` 目录；无 feature 门控（见 §4.2 行）；无能力上报类型 | 矩阵无实现载体，`WindowCapabilities`（内核）无生产者 |
| §19 错误/诊断/可观测性（错误分类、日志降噪、指标、调试面板、事件录制） | 未落地 | `src/error.rs`(57 行) 仅有基础错误类型；无 `recording.rs`、无 `diagnostics.rs`、无指标 | 文档 §19.3/19.5（指标 + 事件录制）零实现；`prism_diagnostic` 未接入窗口路径 |
| §23 测试与验证（单元/属性/模型/Fake 后端/事件 Golden/真机矩阵/性能基准/故障注入） | 部分 | crate 内单测：`pkg/prism_window_winit/src/runner.rs:449`（`struct CountingApp`）、`:500`（`struct RecordingApp`），均在 `#[cfg(test)] mod tests` 内 | 无 `tests/` 目录、无 Fake 后端、无 Golden、无 fuzz、无故障注入、无基准 |
| §27 性能与质量验收表（15 行门槛：p99 <0.1ms 批转换、8kHz 输入零丢失、resize 1000 ev/s 合并、稳态 30 分钟无增长、Golden 100% 等） | 未落地 | 无 `benches/`（`find pkg/prism_window_winit -name '*.rs' -not -path '*/src/*'` → 零结果）；`batch::coalesce` 每帧分配 | 15 项门槛全部无自动化载体 |

**小计**：落地 2 / 部分 12 / 未落地 10 / 骨架 0。
**骨架% = (2 + 0.5×12) ÷ 24 = 8.0/24 = 33%**；**接通% = 2/24 = 8%**（仅 M0 与 §9 三态有 crate 外生产消费者；且该消费者本身由 feature `winit` 门控、**默认关闭**——`pkg/prism_app/Cargo.toml` 的 `winit` feature 与 `pkg/prism_render_app/Cargo.toml` 的 `winit` feature 均不在 `default` 内）。

---

#### 2.10.5 `prism_input` — 对应 `docs/prism_input_refactor_zh.md`（515 行）

声明单元数 **17**（§13 路线 P0/P1/P1.2/P1.5/P2.5/P3 共 6 单元 + §4.1/4.3/4.6/5.2/5.5/5.6/7/8/11 共 9 机制单元 + §15 交付的 ABI 信封层 1 + 硬约束 1）。

> **本节最重要的事实**：`pkg/prism_input` **在生产代码里零调用者**。全 workspace grep `prism_input` 只命中根 `Cargo.toml:56`（workspace member）与三处**注释**（`pkg/prism_app/src/determinism.rs:628`、`pkg/prism_window/src/event.rs:5`、`pkg/prism_window_winit/src/{lib.rs:14,translate.rs:11,42,67,91,runner.rs:403}`）。因此按口径，本 crate **没有任何单元可以判为「落地」**。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §13 P0 契约奠基（信封 + 时间轴 tick + `DeviceId` + SPSC 环 + 合并 + golden harness） | 部分 | 信封已交付：`src/envelope.rs:360` 行（`MonotonicTimestamp`/`InputEventSequence`/`InputDeviceId`/`PlatformInputStamp`/`InputEventEnvelope`，见 `src/lib.rs:39-42`）；`src/registry.rs:28,40,88,101` | `InputState::tick`、`InputRing`、SPSC 环、事件合并、golden harness **全缺**。文档 `docs/prism_input_refactor_zh.md:456` 自述「尚未落地完整 `InputState::tick`」、`:461` 自述「`InputRing` / 快照回滚（§5.5）、`GamepadOutput`（§4.1/§8）、IME/Text 变体（§5.4）尚未实现」 |
| §13 P1 Action 层：新 crate `prism_input_map`（Action/Context/Modifier/Trigger + 优先级栈 + 折叠表） | 未落地 | `ls -d pkg/prism_input_map` → **No such file or directory**；文档 `:238-294`（§7）要求为新 crate，文档 `:402` 列为 P1 产出 | 整个 crate 不存在；Action 语义层零实现 |
| §13 P1.2 回滚地基（快照/回滚/固定步长累加器 §5.5/§5.6） | 未落地 | grep `snapshot\|InputRing\|Accumulator\|accumulator\|FixedStep` 于 `pkg/prism_input/src/` → **零命中** | 无 `snapshot`/`restore`、无固定步长累加器 |
| §13 P1.5 手柄双向（`GamepadOutput`、热插拔保槽、设备档案） | 未落地 | grep `GamepadOutput\|AdaptiveTrigger\|HapticPriority\|GamepadInfo` 于 `pkg/prism_input/src/` → 零命中；`src/gamepad.rs`(293) 仅有输入侧（按钮/轴/死区） | 输出命令类型不存在；文档 `:456` 自述未实现 |
| §13 P2.5 文本/IME（`InputEvent::Text`/`Ime*`） | 未落地 | `src/event.rs:34-70`（`InputEvent` 8 变体：`Keyboard`/`MouseButton`/`MouseMotion`/`MouseWheel`/`Touch`/`GamepadButton`/`GamepadAxis`/`GamepadConnection`）——**无 `Text`/`ImePreedit`/`ImeCommit`** | 逻辑字符层缺失；中日韩 IME 用例不可能通过 |
| §13 P3 远期（Pointer 统一 + SDL3·原生·主机·无头后端 + Steam Input glyph） | 未落地 | 无后端 crate；`pkg/prism_input/Cargo.toml` 零 dependencies | 多后端并存零实现 |
| §4.1 `InputBackendCapabilities` 能力探测 + 降级不降模型 | 骨架 | `src/capabilities.rs:145` 行（字段 `keyboard`/`mouse`/`touch`/`gamepad`/`raw_motion`/`max_polling_hz`/`max_gamepads`/`rumble`/`adaptive_trigger`/`gyro`/`touchpad`；`NONE`/`none()`；`supports_force_feedback`/`has_gamepad_output`）；导出 `src/lib.rs:38` | 无生产者（无后端上报）、无消费者 |
| §4.3 帧栅栏 drain-to-batch（与窗口侧 `WindowRegistry::apply_sorted` 对称） | 骨架 | `src/registry.rs:28`（`sort_envelopes`，稳定排序）、`:40`（`InputEnvelopeQueue`）、`:88`（`drain_sorted`）、`:101`（`apply_sorted`）；导出 `src/lib.rs:50` | 无 SPSC 环（文档 §4.3 要求「后端口↔模拟线程唯一通信方式」）；无调用者 |
| §4.6 事件合并 coalescing（相对 delta 累加、绝对值取最后、不可合并永不丢） | 未落地 | grep `coalesc` 于 `pkg/prism_input/src/` → **零命中** | 内核侧无合并实现（合并被文档划给后端，但 `prism_window_winit` 也只合并窗口事件、不产出输入信封） |
| §5.2 时间轴 + `tick` 纯函数 | 部分 | `src/envelope.rs`（`MonotonicTimestamp(pub u64)` 整数纳秒，文档 `:443` 已解释该设计改动）；`PlatformInputStamp` 仅按 `(timestamp, sequence)` 定序 | `InputState` 与 `tick` 完全不存在 |
| §5.5 `InputRing` / 快照 / 回滚（迟到输入 restore+replay） | 未落地 | 上述 grep 零命中 | 整条回滚地基缺失 |
| §5.6 固定步长累加器（整数 ns、帧率无关） | 未落地 | 上述 grep 零命中 | — |
| §7 Action 映射层（`prism_input_map`） | 未落地 | 同 P1 行 | — |
| §8 手柄双向输出（`GamepadOutput`/`HapticPriority`/自适应扳机） | 未落地 | 同 P1.5 行 | — |
| §11 确定性 golden harness（录制双重重放字节级一致 + x86/ARM + 5 场景语料 + 性质测试 + fuzz + 模型对拍） | 未落地 | 无 `tests/`、无 `benches/`、无 golden 产物（全 workspace golden 产物均在物理/渲染/音频 crate）；仅 `src/tests.rs`(299) | CI 红线无载体 |
| §15 已交付的 ABI 信封层（作者自述 P0 已落地） | 骨架 | `src/lib.rs:39-42`、`src/envelope.rs`、`src/capabilities.rs`、`src/registry.rs`；文档 `:442-455` 自述 46 个 lib 测试全绿、`no_std` 构建通过（**文档自述，未复核**） | 无跨 crate 消费者 |
| 硬约束：纯/确定性/`no_std + alloc`/**零依赖**/无 `unsafe` | 骨架 | `pkg/prism_input/Cargo.toml`（无 `[dependencies]`）；根 `Cargo.toml` `unsafe_code = "deny"`；`src/lib.rs` 无 `unsafe` | 满足，但无调用者 |

**小计**：落地 0 / 骨架 4 / 部分 2 / 未落地 11。
**骨架% = (4 + 0.5×2) ÷ 17 = 5.0/17 = 29%**；**接通% = 0/17 = 0%**。
**文档-代码矛盾**：文档 `:400` 要求「winit 后端（P0.5）同时产出窗口与输入信封」，但 `pkg/prism_window_winit/src/translate.rs:71-75,99-103` 把 `KeyboardInput`/`MouseInput`/`MouseWheel`/`Ime` 全部丢弃，`:95` 的 `is_input_device_event` 无调用者 → 两端从未接上。

---

#### 2.10.6 相机家族 — 对应 `docs/prism_camera_design_zh.md`（1288 行，v1.5，2026-10-15）

**唯一文档源**：`grep -rl prism_camera docs/` 命中 `prism_camera_design_zh.md`、`prism_gameplay_design_zh.md`、`prism_render_scene_rhi_port_zh.md`、`prism_game_creator_design_zh.md`；而 `prism_camera_ecs` / `prism_camera_gpu` / `prism_camera_rig` **只被 `prism_camera_design_zh.md` 提及**，无各自独立设计文档。`docs/` 下无 `prism_camera_ecs_design_zh.md` / `prism_camera_gpu_design_zh.md` / `prism_camera_rig_design_zh.md`。

**先给决定性事实**：**四个相机 crate 在生产代码里零外部消费者**。
- `pkg/prism_camera` 的消费者只有三个兄弟相机 crate：`pkg/prism_camera_ecs/Cargo.toml:12`、`pkg/prism_camera_gpu/Cargo.toml:11`、`pkg/prism_camera_rig/Cargo.toml:13`。
- `pkg/prism_camera_ecs` **无任何 dependents**；`pkg/prism_camera_gpu` **无任何 dependents**；`pkg/prism_camera_rig` 只被 `pkg/prism_camera_ecs/Cargo.toml:16`（optional）依赖。
- 排除相机家族后，全 `pkg/` grep `CameraBundle|CameraIntent` → **零命中**；`examples/`、`pkg/prism_app`、`pkg/prism_render_app`、`pkg/prism_render_runtime` grep `CameraPlugin|camera_ecs|camera_rig|camera_gpu` → 只命中 Bevy 自带的 `bevy::camera_controller::free_camera::FreeCameraPlugin`（如 `examples/3d/solari.rs:56`、`examples/3d/rect_light.rs:3`），与本家族无关。
- → 整个相机家族是**传递性死端**：内部互相引用，外部无入口。文档 `:1215` 自己承认「本文中涉及的渲染后处理算法、物理后端、XR 设备驱动和具体渲染 backend 由对应所有者消费这些稳定契约」——即消费方尚未接线。

##### `prism_camera`（`pkg/prism_camera`，4521 行，13 个模块 + 17 个 feature）

声明单元数 **19**（§26 交付矩阵 M0/M1/M2/M3/M6 共 5 行 + §17 输入与拾取 1 + §19 确定性/回放 1 + §22 可观测性 `trace` 1 + §25 契约与不变量 1 + §28.1–§28.10 十项高级契约 10）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §26 M0 代数核（Perspective/Orthographic、reversed-Z、无限远、斜裁剪、SubView、5/6 面 frustum） | 骨架 | `src/projection.rs`(361)：`FovAxis :9`、`ObliquePlane :17`、`SubView :37`、`PerspectiveProjection :56`、`OrthographicProjection :80`、`Projection :102`、`ProjectionError :114`；`src/frustum.rs`(128) | 无 crate 外消费者；文档 `:1173` 的门禁（端点深度/FOV/逆变换/5/6 面剔除）已有 crate 内单测（文档 `:1198` 自述 `prism_camera 32 passed`，**未复核**） |
| §26 M1 ViewData（视图派生、历史槽、`ViewRegistry`、viewport/order、RenderTarget） | 骨架 | `src/view.rs`(447)：`CameraPose :14`、`CameraIntent :32`、`ViewData :72`、`ViewStatus :100`、`ViewHistory :107`；`src/target.rs`(122)；`src/lib.rs:86`（`derive_view`/`publish_view`）；`src/view.rs:2`（`use prism_render_architecture::view_family::{ViewHandle,ViewImportance,ViewKind,ViewRegistry}`，`registry` 参数 `view.rs:281`） | 无外部消费者；`ViewRegistry` 真相在 `prism_render_architecture`（跨 crate 依赖已接通但方向相反） |
| §26 M2 时域（Halton/Sobol、稳定矩阵、teleport、运动矢量输入） | 骨架 | `src/jitter.rs`(104) | 无外部消费者 |
| §26 M3 镜头与后处理（Filmback/Lens/EV100/Auto/DOF/PostStack 固定物理序） | 骨架 | `src/lens.rs`(215)：`Filmback :7`、`Lens :26`、`Exposure :110`；`src/post.rs`(313)：`PostStage :12`、`DofSettings :37`、`Tonemapper :50`、`PostStack :186`、`PostStackError :176` | 无外部消费者（后处理真实执行归渲染 crate，文档 `:1215` 已声明） |
| §26 M6 大世界/XR/GPU/确定性（相对渲染原点、per-eye、late-latch、View ABI、回放 digest、外部姿态校验、遮挡滞回、前瞻预测、历史稳定化） | 骨架 | `src/stereo.rs`(250)：`Eye :6`、`StereoConfig :12`、`LateLatchPose :29`、`StereoPair :36`；`src/gpu.rs`(267)：`GpuViewRecord :26`；`src/advanced.rs`(917) + `src/advanced_runtime.rs`(1117) | 无外部消费者；相对渲染 GPU 侧在 `prism_camera_gpu`（内部闭环） |
| §17 输入与拾取（viewport↔world 互转、picking、多相机命中仲裁） | 骨架 | `src/pick.rs`(135) | 无外部消费者；与 `prism_input` 的协作链不存在（`prism_input` 零消费者，见 2.4.5） |
| §19 确定性与回放对接（相机不进复制单元、纯函数、预测只跟随不和解） | 骨架 | `src/lib.rs:107`；feature `determinism` | 无外部消费者；`prism_network` 侧未见消费（未核实） |
| §22 可观测性与诊断（feature `trace`） | 骨架 | `pkg/prism_camera/Cargo.toml` feature 列表（文档 `:17` 与 `:1041` 要求 `trace`）；诊断资源落在 `pkg/prism_camera_ecs/src/lib.rs:465`（`CameraDiagnostics`） | crate 内 `trace` feature 的实际观测出口未核实 |
| §25 契约、不变量与版本化 | 骨架 | `src/error.rs:17`（`CameraError`）；`src/lib.rs:7`（`#![forbid(unsafe_code)]`）；`src/gpu.rs`（`VIEW_SCHEMA_VERSION`） | 无外部消费者 |
| §28.1–§28.10 十项 AAA 高级契约（`cutscene`/`autofocus`/`occlusion_fade`/`shutter`/`view_batch`/`lookahead_stream`/`edge_stabilize`/`virtual_production`/`view_lod`/`composition_aid`） | 骨架 ×10 | 文档 `:1246-1286` 逐项定义 feature；`pkg/prism_camera/Cargo.toml` 全部 10 个 feature 存在；文档 `:1211` 自述 advanced 契约提供 `CameraTrack`/自动对焦/快门/foveation/批量 key/LOD/构图/确定性 digest/`ExternalPoseValidator`/`OcclusionFadeState`/`LookaheadPredictor`/`HistoryStabilizer` | 10 项全部无 crate 外消费者；`composition_aid` 标为 editor-only |

**小计**：落地 0 / 骨架 19 / 部分 0 / 未落地 0。
**骨架% = 19/19 = 100%**；**接通% = 0/19 = 0%**。
（注：crate 内部实现密度高——文档 `:1211` 具名类型与我的 grep 逐一对应，未发现具名类型缺失。骨架% 高、接通% 0 是本家族的核心结论。）

##### `prism_camera_rig`（`pkg/prism_camera_rig`，2050 行单文件 + `benches/camera_contracts.rs`）

声明单元数 **3**（§26 M4 Rig 与控制器、§26 M5 导演、§18 控制器取代 `bevy_camera_controller`）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §26 M4 Rig 与控制器（Body/Aim、SpringArm、批量求交、半衰期阻尼、输入归一化、LOD 调度） | 骨架 | 文档 `:1177`、`:1211`；`pkg/prism_camera_rig/src/lib.rs`（2050 行，`mod tests :1670`）；`src/lib.rs:12,14`（`use prism_camera::CameraPose` / `Lens`）；feature `director`/`rig`/`determinism`/`trace`（`Cargo.toml`）；文档 `:1198` 自述 15 passed（**未复核**） | 无 crate 外消费者（唯一 dependents 是 optional 的 `prism_camera_ecs`）；`SceneQuery` 的 `prism_physics` 实现不存在（文档 `:670` 声明「无物理时降级直伸臂」，属设计内） |
| §26 M5 导演（VCam、Brain、稳定选择、混合、确定性噪声、ImpulseBuffer） | 骨架 | 文档 `:1178`、`:1211`；`prism_camera_rig::CameraBrain`/`VirtualCamera`（文档 `:1099-1101` 引用） | 同上 |
| §18 控制器预设（取代 `bevy_camera_controller`，文档 `:900`、`:1134`「升为一等」） | 骨架 | `src/lib.rs`；`src/lib.rs:3`（`#![forbid(unsafe_code)]`） | 无 crate 外消费者；`examples/` 仍在用 Bevy 的 `bevy::camera_controller::free_camera::FreeCameraPlugin`（`examples/3d/solari.rs:56` 等）→ 「取代」未发生 |

**小计**：落地 0 / 骨架 3。**骨架% = 100%**；**接通% = 0/3 = 0%**。

##### `prism_camera_gpu`（`pkg/prism_camera_gpu`，1123 行单文件 + `tests/rhi_gpu.rs`）

声明单元数 **2**（§20 GPU 视图常驻/View ABI/增量上传/相机相对还原、§26.1 GPU ABI 验收 + §26 M6 的 CPU/GPU parity 门禁）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §20.1–§20.4 View UBO 布局 / 多视图数组与 GPU-driven / 增量上传 / shader 还原 | 骨架 | `src/lib.rs`：`FrameSlot :159`、`UploadPlan :250`、`RhiViewBuffer :320`（`new :329`/`destroy :409`）、`RhiUploadBackend :422`、`ViewBufferRing :640`（`begin_frame :726`/`plan_uploads :757`/`submit :829`/`commit :840`/`reclaim :877`）、私有 `ViewRecord :38` + WGSL `views: array<ViewRecord>` `:86`；`src/lib.rs:28`（`use prism_render_driver::{BufferDescriptor,BufferId,BufferUsages,RenderDevice,RenderQueue}`）；`src/lib.rs:20`（`#![forbid(unsafe_code)]`） | 无 crate 外消费者——GPU 侧「写真实 RHI buffer」是**出向**调用（`prism_camera_gpu`→`prism_render_driver`），不是被别处调用；`tests/rhi_gpu.rs` 是**测试**，按口径不计接通 |
| §26.1 GPU ABI 验收 + CPU/GPU parity（字段偏移/schema/flags/矩阵列序逐项断言） | 骨架 | 文档 `:1190`、`:1202`（自述 `PRISM_REQUIRE_GPU=1 … 1 passed`，**未复核**）；`tests/rhi_gpu.rs`；dev-deps `prism_render_driver_wgpu` + `wgpu 30`（`Cargo.toml`） | 验收载体存在但是测试；无生产渲染管线消费 |

**小计**：落地 0 / 骨架 2。**骨架% = 100%**；**接通% = 0/2 = 0%**。

##### `prism_camera_ecs`（`pkg/prism_camera_ecs`，1042 行单文件 + `benches/multiview_contracts.rs`）

声明单元数 **3**（§21 与 Transform/App/Time/Window 集成、§14 视图注册与合成（`PostUpdate` 发布链）、§22 `CameraDiagnostics`）。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §21.1–§21.4 系统集与顺序 / App 装配 / Window 尺寸缩放 / Time 协作 | 骨架 | `src/lib.rs`：`CameraSystemSet :37`、`CameraFrame :236`（`effective_dt :270`）、`CameraPlugin :695`、systems `apply_camera_directors :487`/`apply_camera_controls :521`/`apply_camera_rigs :551`/`prepare_camera_frame :616`/`publish_camera_views :623`；`src/lib.rs:16`（`use prism_app::{App,EngineClocks,Plugin,PostUpdate}`）、`:21`（`prism_ecs::prelude`）；`src/lib.rs:9`（`#![forbid(unsafe_code)]`） | 无 crate 外消费者；文档 `:1004` 明确「以下名称是跨 crate 的装配顺序约定，不是当前 `prism_camera` 导出的枚举」（诚实标注）；`CameraPlugin` 仅在本 crate 的 `mod tests :769` 里被装配 |
| §14 ViewRegistry 与合成规则（实体→稳定 `ViewHandle`、相机写/渲染读单向解耦） | 骨架 | `src/lib.rs:291`（`CameraViewRegistry`）、`:380`（`PublishedView`）、`:389`（`PublishedViews`）、`:456`（`CameraViewError`） | 渲染侧读端未接线（`prism_render_app` grep 相机家族 → 零命中） |
| §22 `CameraDiagnostics` | 骨架 | `src/lib.rs:465`（`CameraDiagnostics`） | 无外部消费者 |

**小计**：落地 0 / 骨架 3。**骨架% = 100%**；**接通% = 0/3 = 0%**。

---

#### 2.10.7 每 crate 汇总表

| crate | 对应文档 | 声明单元数 | 落地 / 骨架 / 部分 / 未落地 | 骨架% | 接通% | bevy 耦合 | 最大缺口 |
|---|---|---|---|---|---|---|---|
| `prism_platform` | `docs/prism_platform_design_zh.md` | 22 | 6 / 7 / 6 / 3 | **73%** | **32%** | 无 | M6 崩溃捕获的 **minidump 不存在**（`pkg/prism_platform/src/lib.rs:67-68` 自述 future work，与文档 `:334` 声称 M0–M6 全落地矛盾）；`src/backend/` 与 9/13 个 feature 缺失 |
| `prism_platform_os` | 散见 4 份文档（无独立设计文档） | 3 | 1 / 1 / 1 / 0 | **83%** | **33%** | 无 | Windows 后端与真实 qos/power syscall 零实现；安全探针无调用者 |
| `prism_window` | `docs/prism_window_refactor_zh.md` | 17 | 5 / 0 / 8 / 4 | **53%** | **18%** | 无 | §7 golden 确定性对拍门禁（P0 的**验收条件本身**）零实现；HDR/VRR/`parent`/`WindowAttributesDelta` 词汇表缺失；命令/结果 ABI 只有类型没有使用者 |
| `prism_window_winit` | `docs/prism_window_winit_design_zh.md` | 24 | 2 / 0 / 12 / 10 | **33%** | **8%** | 无 | 键盘/鼠标/滚轮/IME/触摸/拖放输入**全部被丢弃**（`src/translate.rs:71-75,99-103`）；无 `[features]` 段、模块计划 12+ 文件全缺；接通被 `winit` feature 门控且默认关闭 |
| `prism_input` | `docs/prism_input_refactor_zh.md` | 17 | 0 / 4 / 2 / 11 | **29%** | **0%** | 无 | 全 workspace **零生产调用者**；`InputState::tick`/`InputRing`/快照回滚/`GamepadOutput`/IME·Text 变体全缺；`prism_input_map` crate 不存在 |
| `prism_camera` | `docs/prism_camera_design_zh.md` | 19 | 0 / 19 / 0 / 0 | **100%** | **0%** | 无 | 实现密度高但**零外部消费者**；消费方（渲染/物理/XR）未接线（文档 `:1215` 自认） |
| `prism_camera_rig` | 同上（§26 M4/M5、§18） | 3 | 0 / 3 / 0 / 0 | **100%** | **0%** | 无 | 「取代 `bevy_camera_controller`」未发生：`examples/` 仍用 Bevy 自带 `FreeCameraPlugin` |
| `prism_camera_gpu` | 同上（§20、§26.1） | 2 | 0 / 2 / 0 / 0 | **100%** | **0%** | 无 | 唯一验证载体是测试（`tests/rhi_gpu.rs`），无生产渲染管线消费 |
| `prism_camera_ecs` | 同上（§14、§21、§22） | 3 | 0 / 3 / 0 / 0 | **100%** | **0%** | 无 | `CameraPlugin` 只在自身 `mod tests` 内被装配；渲染读端未接线 |

> 汇总表口径提示：相机四 crate 的「骨架% 100%」不等于「完成度高」——它只说明**文档具名的类型全部存在**；其**接通% = 0%** 才是本组真正的完成度结论。

---

#### 2.10.8 Bevy 耦合实测

**实测方法**：逐个读 9 个 crate 的 `pkg/<crate>/Cargo.toml`（`[dependencies]` / `[dev-dependencies]` / `[target.*.dependencies]` / `[features]` 全段），并对 9 个 crate 的 `src/` 做 `grep -n bevy`。

**结论：9 个 crate 中 `bevy_*` 依赖与 `crates/bevy_*` path 依赖均为 0 条。本组是**完全解耦**的。**

| crate | `bevy_*` 依赖行 | `crates/bevy_*` path 依赖 | 唯一的 "bevy" 文本出现处（非依赖） |
|---|---|---|---|
| `prism_platform` | 无 | 无 | `pkg/prism_platform/Cargo.toml:6`（description：「depends on no `bevy_*` crate」）；`pkg/prism_platform/src/lib.rs:71`（注释：「depends on no `bevy_*` crate」） |
| `prism_platform_os` | 无 | 无 | 无 |
| `prism_window` | 无（`Cargo.toml` 连 `[dependencies]` 段都没有） | 无 | `pkg/prism_window/Cargo.toml:6`（description 提及后端形态） |
| `prism_window_winit` | 无（deps 仅 `prism_window` / `prism_platform` / `winit 0.30` / `raw-window-handle 0.6`） | 无 | `pkg/prism_window_winit/Cargo.toml`（无 bevy 字样） |
| `prism_input` | 无（零 dependencies） | 无 | 无 |
| `prism_camera` | 无 | 无 | 无（`Cargo.toml:6` 描述「depends on no `bevy_*` crate」；文档 `docs/prism_camera_design_zh.md:17` 约束「不依赖任何 `bevy_*` crate」） |
| `prism_camera_rig` | 无（deps：`libm`/`prism_math`/`prism_camera`） | 无 | 无 |
| `prism_camera_gpu` | 无（deps：`prism_camera`/`prism_render_driver`） | 无 | 无 |
| `prism_camera_ecs` | 无（deps：`prism_app`/`prism_camera`/`prism_ecs`/`prism_math`/`prism_render_architecture`/`prism_camera_rig`(opt)） | 无 | 无 |

**对照事实**（解释为什么「Bevy 耦合」在本组是零）：本仓是 Bevy fork——仓根 `Cargo.toml:1-9` 的 `[package] name = "bevy"`、`version = "0.20.0-dev"`，上游 `bevy_*` crate 全在 `crates/`（`bevy_a11y`、`bevy_app`、`bevy_camera`、`bevy_input`、`bevy_camera_controller` …）。本组 9 个 Prism crate 位于 `pkg/`，与 `crates/` 是**平行的两套体系**；9 个 crate 无一引用 `crates/` 下的任何东西。唯一的 Bevy 残留在 `examples/`（如 `examples/3d/solari.rs:56` 用 `bevy::camera_controller::free_camera::FreeCameraPlugin`），那是 fork 自带示例，不在本组审计范围。

**`unsafe` 与依赖方向的交叉核对**（佐证「解耦 + 纯内核」）：根 `Cargo.toml` `[workspace.lints.rust] unsafe_code = "deny"`，9 个 crate 全部 `[lints] workspace = true`；其中 `pkg/prism_camera/src/lib.rs:7`、`pkg/prism_camera_rig/src/lib.rs:3`、`pkg/prism_camera_gpu/src/lib.rs:20`、`pkg/prism_camera_ecs/src/lib.rs:9` 另有显式 `#![forbid(unsafe_code)]`；`prism_platform` 的 `unsafe` 只出现在 `thread::affinity` / `vm` / `fs::mmap` / `fs::watch` / `dynlib` / `process::env` / `crash` 的逐点 `#[expect(unsafe_code, reason)]`（`pkg/prism_platform/src/lib.rs:73-81` 自述）。

---

#### 2.10.9 文档无 crate / crate 无文档

**（A）文档要求但 crate 不存在（本节范围内）**

| 文档要求的 crate / 产物 | 文档出处 | 实测 | 影响 |
|---|---|---|---|
| `prism_input_map`（Action 映射层，新 crate） | `docs/prism_input_refactor_zh.md:238-294`（§7）、`:402`（§13 P1 产出「新 crate」） | `ls -d pkg/prism_input_map` → **No such file or directory** | 输入语义层整体缺失；P1 阶段无法开始 |
| `prism_window_headless`（headless/replay 后端，走同一 ABI） | `docs/prism_window_winit_design_zh.md` M5 行（「full ABI via prism_window_headless/replay」） | `ls -d pkg/prism_window_headless pkg/prism_window_replay` → 均不存在 | winit 语义不外泄的门禁无载体；CI 无法脱离真实窗口验证 ABI |
| `prism_window_winit` 的 `platform/{macos,windows,linux,web,ios,android}.rs`、`accessibility.rs`、`diagnostics.rs`、`recording.rs`、`pacing.rs`、`plugin.rs`、`state.rs`、`handle.rs` | `docs/prism_window_winit_design_zh.md:260-296`（§4.1 模块计划） | 实际只有 `batch command convert error id lib runner sync translate` | 设计文档的模块级计划与实际实现几乎不重合（12+ 计划文件零存在） |
| 窗口/输入两侧的 golden 语料与对拍门禁 | `docs/prism_window_refactor_zh.md:322-347`（§7）、`docs/prism_input_refactor_zh.md:356-384`（§11），两处均标为 **CI 红线** | 无任何录制器/重放器/golden 语料；全 workspace 的 golden 产物只在 `pkg/prism_physics_core/src/cache/golden.rs`、`pkg/prism_render_scene/tests/water_hybrid_golden.rs`、`pkg/prism_render_architecture/src/water/frame_golden.rs`、`pkg/prism_audio_profiler/src/golden_diff.rs` | 两条 CI 红线均无实现 |
| `prism_platform` 的 `src/backend/{win,apple,posix,android,web,mock}.rs` | `docs/prism_platform_design_zh.md:300`（§20） | `ls pkg/prism_platform/src/backend/` → No such file or directory | 后端分层被 `pkg/prism_platform_os` 部分替代，但 mock/web/android 无家可归 |
| `prism_platform` 的 `crash/{handler.rs,minidump.rs}` | `docs/prism_platform_design_zh.md:294`（§20） | 实际 `crash/{context,mock,mod,posix,unsupported}.rs` | `minidump.rs` 缺失见 §2.4.1 结论 |

**（B）crate 存在但无文档指定它（本节范围内）**

| crate | 状态 |
|---|---|
| `prism_platform_os` | **无独立设计文档**（`docs/` 下无 `prism_platform_os_design_zh.md`）。其规格只以片段散落在 `docs/prism_platform_design_zh.md:377-386,413-417,438`、`docs/prism_window_refactor_zh.md:60`、`docs/prism_tasks_design_zh.md:472`、`docs/prism_ecs_design_zh.md:429`。**后果已实测**：代码已有 Linux 后端（`pkg/prism_platform_os/src/topology/mod.rs:44,93,158`）而文档 `:383-385` 仍写 PLANNED——文档滞后于代码，无法作为该 crate 的验收依据 |
| `prism_camera_rig` | 无专属文档；只被 `docs/prism_camera_design_zh.md` 在 `:13,101-102,128,144,173,670,902,1099-1101,1134,1158` 等处以段落形式规定 |
| `prism_camera_gpu` | 无专属文档；只被 `docs/prism_camera_design_zh.md:13,1190,1202,1211` 规定（§20 + §26.1） |
| `prism_camera_ecs` | 无专属文档；只被 `docs/prism_camera_design_zh.md:13,67,168,174,234,237,1004,1023,1028,1084,1089-1090,1140` 规定（§21 + §14 + §22） |

**（C）文档全部 9 个 crate 都有某种文档覆盖**——不存在「完全无文档的 crate」。（A）类才是真正「有文档无实现」的部分。

---

#### 2.10.10 关键结论

1. **`prism_platform` 的最大未落地项是 M6 的 minidump，且存在文档-代码直接矛盾。** `docs/prism_platform_design_zh.md:326`（§22 M6）与 `:334`（§23 诚实边界「M0–M6 核心路线图**已全部落地并通过验证**」）声称 minidump 已交付；但 `pkg/prism_platform/src/crash/` 无 `minidump.rs`（文档 `:294` 的 §20 布局明确要求该文件），`pkg/prism_platform/src/lib.rs:67-68` 代码注释自述「minidump serialization and cross-process handlers … remain future work」，且 `grep minidump` 在整个 `src/` 只命中 `lib.rs` 的这句注释。同时 `crash::install`（`pkg/prism_platform/src/crash/mod.rs:234`）在 workspace 内**无任何非测试调用者**，交付 `prism_diagnostic` 的崩溃闭环实际未接通。
2. **`prism_window_winit` 是「窗口能开、输入全丢」。** `src/translate.rs:71-75,99-103` 把 `KeyboardInput`/`ModifiersChanged`/`Ime`/`MouseWheel`/`MouseInput` 全部列入丢弃分支，只有 `CursorMoved`/`CursorEntered`/`CursorLeft`（`:57-64`）被翻译；`:95` 的 `is_input_device_event` 无任何调用者（注释 `:67` 声称「routed by `prism_input`」）。因此文档 §26 M0 的验收行「keyboard+mouse button/motion/wheel basic translation」不成立，且 `prism_input` 与窗口后端之间**从未接上**。叠加 `pkg/prism_window_winit/Cargo.toml` **完全没有 `[features]` 段**（文档 §4.2 要求 9 个）、§4.1 计划的 12+ 模块文件零存在、§27 的 15 项性能门槛无 `benches/` 载体，其骨架% 仅 33%、接通% 仅 8%，是本组**实现-文档偏离最大**的 crate。
3. **窗口/输入两侧的确定性 golden 对拍门禁（两份文档都标为「CI 红线」）均零实现，而这恰是窗口 P0 的验收条件本身。** `docs/prism_window_refactor_zh.md:322-347` 要求「录制 → 双重重放 → snapshot 字节级一致 + x86/ARM 一致」；`docs/prism_input_refactor_zh.md:356-384` 要求同一形态。实测无录制器、无重放器、无 golden 语料（全 workspace 的 4 个 golden 产物都在物理/渲染/音频 crate），窗口侧只有 `pkg/prism_window/src/tests.rs`(265 行) 单元测试。这直接导致 `prism_window` 的 P0 单元只能记「落地类型、未达验收」。
4. **`prism_window` 内核的命令/结果 ABI 是一条「只有类型、没有使用者」的断头路。** `WindowCommandEnvelope`（`pkg/prism_window/src/command.rs:265`）、`WindowCommandResult`（`src/result.rs:183`）、`CommandStatus`（`src/result.rs:95`）、`RealizedWindowDelta`（`src/result.rs:139-148`）在 `pkg/` + `os-backends/` 全域**零非测试消费者**，`pkg/prism_window/src/tests.rs` 内也零命中；后端走的是自己的 `WindowSync`（`pkg/prism_window_winit/src/sync.rs`）+ `flush_commands`（`runner.rs:228-240`，逐条 `command::apply` 后 `mark_applied`，不产出回执）。同时 §4.2 要求的 `WindowAttributesDelta`、§5.1 的 `WindowAttributes.parent`、§5.3 的 `HdrMetadata`/`ColorGamut`/`Eotf`、§5.4 的 `VrrMode`/`refresh_rate_range_millihertz`、§6 的 `LowLatencyHint`、§5.2 的 `Monitor.work_area` 与 `scale_factor_milli` 全部 grep 零命中 → P2 次世代显示整包未落地。
5. **`prism_input` 是全组唯一「零生产调用者」的输入内核，且其设计文档自述的 P0 只完成了信封层。** 全 workspace grep `prism_input` 只命中 workspace member 声明与 3 处注释。文档 `:461` 自述 `InputState::tick` / `InputRing` / 快照回滚（§5.5）/ `GamepadOutput`（§8）/ IME·Text 变体（§5.4）尚未实现——我逐一 grep 确认：`snapshot`/`Accumulator`/`coalesc`/`GamepadOutput`/`Text` 在 `pkg/prism_input/src/` 全部零命中；文档 §7/§13-P1 要求的新 crate `prism_input_map` 在 `pkg/` 下不存在。骨架% 29%、接通% 0%。
6. **相机家族的完成度结构是「内实外虚」：四个 crate 的文档具名类型 100% 存在，但对外接通 0%。** `prism_camera`(4521 行)/`prism_camera_rig`(2050)/`prism_camera_gpu`(1123)/`prism_camera_ecs`(1042) 合计 8736 行，`prism_camera` 的 `CameraPose/CameraIntent/ViewData/ViewHistory/Projection/PostStack/StereoPair/LateLatchPose/GpuViewRecord`、`rig` 的 `VirtualCamera/CameraBrain/SceneQuery`、`gpu` 的 `ViewBufferRing/RhiViewBuffer/RhiUploadBackend`、`ecs` 的 `CameraBundle/CameraPlugin/CameraViewRegistry/CameraDiagnostics` 与文档 `:1211` 逐一对应；但 `pkg/prism_camera_ecs`、`pkg/prism_camera_gpu` **无任何 dependents**，`pkg/prism_camera_rig` 只被 optional 依赖，`pkg/prism_camera` 只被这三个兄弟 crate 消费 → 传递性死端。典型反证：文档 `:900,1134` 声称用 `prism_camera_rig` 控制器「取代 `bevy_camera_controller`」，而 `examples/3d/*.rs`（如 `examples/3d/solari.rs:56`、`examples/3d/rect_light.rs:3`）仍在使用 Bevy 自带的 `FreeCameraPlugin`。文档 `:1215` 已自认「由对应所有者消费这些稳定契约」，即消费方尚未接线。
7. **Bevy 耦合判定：本组 9 个 crate 完全解耦，`bevy_*` 依赖 0 条、`crates/bevy_*` path 依赖 0 条。** 9 个 `Cargo.toml` 逐段核对（含 `[dev-dependencies]` 与 `[target.*.dependencies]`），`bevy` 字样只出现在 3 处**文字/注释**（`pkg/prism_platform/Cargo.toml:6`、`pkg/prism_platform/src/lib.rs:71`、`prism_camera` 文档 `docs/prism_camera_design_zh.md:17`）。本仓虽是 Bevy fork（仓根 `Cargo.toml:1-9`，上游 crate 在 `crates/`），但 `pkg/` 与 `crates/` 是两套平行体系，本组 9 个 crate 无一引用 `crates/` 下任何 crate。

---

**审计范围与局限声明**

- 本轮**未执行任何 `cargo build`/`cargo test`/`cargo bench`**；所有「测试通过」「基准 ns」「真机验证」均为**文档自述**，本审计只做存在性/接线性的静态核对。
- 百分比的分母 = 我从文档抽出的具名单元数（逐 crate 已在上文声明），**不是文档行数或代码行数**；换一套单元抽取口径，数字会变。相机四 crate 的「骨架% 100%」尤其依赖「文档 `:1211` 具名类型逐一存在」这一判断。
- 「接通」只按**静态导入/调用点**判定，不代表运行期行为正确。
- **未核实**的项已逐条在表内标注，主要是：`prism_window_winit` 真机 Windows/macOS 行为、`prism_camera` 的 `trace` feature 实际观测出口、`prism_camera` 与 `prism_network` 的回放消费、`prism_platform` 各 TSC/大页/崩溃路径的运行期正确性。

---

## 2.11 世界系统（world + replication，11 crate）

静态审计（只读，未运行 cargo）。

#### (a) 机制级核对表
**prism_world_stream**（规范 §5/§6/§24.2）
| 要求单元 | 状态 | 证据(file:line) | 缺口 |
|---|---|---|---|
| i64 定点 + Euclid 除法 + GlobalPos/LocalPos | 落地 | pkg/prism_world_stream/src/lib.rs:109,127 | — |
| sector 树 + Morton 编码 | 落地 | pkg/prism_world_stream/src/lib.rs:49,64,138 | — |
| Hilbert 编码（§5 与 §24.2 明列） | 未落地 | 全仓 grep 仅命中无关 pkg/prism_audio_core/src/nodes/effects/frequency_shifter.rs:15 | 无实现；implement 文档自认未做 |
| cell manifest 解析 | 落地 | pkg/prism_world_stream/src/lib.rs:224 | 无磁盘 manifest 解析器/容器解码 |
| 流送 VM：需求集/预算/优先级/驱逐 | 落地 | pkg/prism_world_stream/src/lib.rs:241,284,312,332,361 | 接 scene spawn/despawn 由宿主实现 |
| VM trace | 落地 | pkg/prism_world_stream/src/lib.rs:562 | — |
| 异步 IO 触发 + GPU 直传触发 | **未落地** | `gpu_stream` feature 全工作区无 | 无任何直传代码 |

**prism_world_residency**（§9/§10/§11.2/§14/§15/§17/§21/§23）
| 要求单元 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 三档常驻状态机 + 迟滞 | 落地 | pkg/prism_world_residency/src/residency.rs:121 | — |
| proxy⇄ECS 升降级预算 | 落地 | pkg/prism_world_residency/src/residency.rs:121 | 需宿主 runtime |
| HLOD 簇/分层代理 | 落地 | pkg/prism_world_residency/src/hlod.rs:62 | — |
| QoS 预算治理 | 骨架 | pkg/prism_world_residency/src/qos.rs:125 | 无端到端 QoS 实测 |
| 稀疏 SDF brick 场 | 骨架 | pkg/prism_world_residency/src/sdf.rs:301 | 无生产消费者 |
| SDF clipmap 环滚动 | 骨架 | pkg/prism_world_residency/src/clipmap.rs:208 | 同上 |
| SDF 一场多用（§17.2 五个消费者） | **未落地** | Cargo.toml 无 prism_gi/physics/render 依赖 | 消费者未接 |
| PCG 图→IR→求值 | 骨架 | pkg/prism_world_residency/src/pcg.rs:372 | 无 GPU 求值（prism_world_gpu 无 pcg 模块） |
| CPU 散布/最小间距/笔刷索引 | 落地 | scatter.rs:315、scatter_brush_index.rs:394 | — |
| 散布/placement 删除持久化 | 落地 | 同 runtime 行 | — |

**prism_world_persist**（§11.1–§11.5）
| 持久 ID + override delta | 落地 | types.rs:7,353 | — |
| append-only 事务日志 | 落地 | durable.rs:335 | — |
| 快照/时间轴 | 落地 | timeline.rs:59 | — |
| CRDT 空间合并（§11.3） | 落地（含边界） | docs/prism_world_system_design_zh.md:1085 自认深层结构冲突未解 | 冲突策略缺失 |

**prism_world_gpu**（15 个 WGSL）
| proxy 可见性 + 固定容量 + indirect | 落地 | resident.rs:95、registry.rs:162 | 几何为 f32，非 CPU i64 逐位对齐（prism_world_gpu_planet_placement_zh.md 自认） |
| sector proxies | 部分 | sector_proxies.rs:86 | 未接产品 scene decoder |
| density 距离淡出 | 落地 | density.rs:16、shaders/density_dither.wgsl | — |
| GPU 散布（field/heightfield/brush/删除） | 落地 | scatter.rs:110、scatter_field.rs:92、scatter_heightfield.rs:151、scatter_deletion.rs:40 | — |
| 高度场 placement + 删除 | 落地 | heightfield_placement.rs:132、heightfield_placement_deletion.rs:38 | — |
| 行星 placement + 删除 | 落地（边界外） | planet_placement.rs:106、planet_placement_deletion.rs、shaders/planet_placement{,_deletion}.wgsl | 真机 readback 无通过证据（文档自报 Metal 无 adapter） |
| SDF brick 查找 | 骨架 | sdf.rs:172 | 无消费方 |
| placement runtime / 单元格宿主与调度 | **未落地** | docs/prism_world_gpu_placement_runtime_zh.md 仅约 9 行占位 | 无文档、无代码 |

**prism_world_runtime**
| cell 帧协调 + reload/替换 | 落地 | src/lib.rs:904 | — |
| 数据层 Loaded/Activated | 落地 | src/data_layers.rs:32,316,466 | — |
| IO 调度适配 | 落地 | src/io_adapter.rs:31,146 | — |
| 资产容器 cell provider | 落地 | src/asset_container_provider.rs:291 | — |
| proxy cell host | 落地 | src/proxy_cell_host.rs:97 | 依赖宿主提供 decoder |
| 线程/任务/reactor cell IO | 落地 | src/task_cell_io.rs:30、src/priority_cell_io.rs:28 | — |
| 能力门禁 | 落地 | src/world_partition_capability.rs:38,58 | — |
| GPU 直传适配器 | 未落地 | 全工作区无 gpu_stream | 与 design §24.3 矛盾 |

**prism_world_io**
| 请求/调度/预算/背压 | 落地 | src/lib.rs:14,31,423 |
| 线程/队列/反应堆提供者 | 落地 | src/threaded_file.rs:49,77、src/reactor.rs:102 |
| 校验（§9.4） | 落地 | IoScheduler + benches |

**prism_world_network**
| Session/状态机/传输 | 落地 | src/session.rs:18,29,172,333,109 |
| AOI 订阅 + 移交账本 | 落地 | src/lib.rs:51,262,378 | AoiSubscription 为 AABB，非 AoiGrid |
| ServerMesh | 落地 | src/server_mesh.rs:382 |
| 与 prism_replication 对接 | **未落地** | grep -rl prism_world_network pkg/*/Cargo.toml 仅命中自身；prism_replication/src/aoi.rs:35 的 AoiGrid 只被 pkg/prism_transform_ecs/src/grid.rs:18 用 | 无下游消费者 |

**prism_world_terrain**
| TileKey/四叉树 + 邻接 | 落地 | src/tile.rs:63 |
| 高度场网格 + 增量重烘 | 落地 | src/bake.rs:305、src/rebake.rs:172 |
| 行星增量网格 | 落地 | src/planet_mesh.rs:131 | 自认不接 renderer graph |
| LOD 接缝规划 | 部分 | src/lod_seam.rs:304 | 无渲染接入 |
| 平面高度场 + 流适配 | 骨架 | src/heightfield_surface.rs:96、src/stream_adapter.rs:82 | 无消费方 |
| SDF 体积烘焙 → 曲面网格 | 骨架 | src/volume_bake.rs:182、src/surface.rs:114 | 无生产调用 |
| VHM/体素/Dual Contouring/Transvoxel | 未落地 | 全仓无 VoxelGrid |
| 碰撞几何导出 / 渲染图 / WESL | 未落地 | 无 consumer |

**prism_world_sim**
| 统一预算 tick | 落地 | src/sim.rs:13,101 |
| 侵蚀/破碎/生态/水文 | 骨架 | src/hydrology.rs:88,209、src/ecology.rs:91,10 | 无产品调用；§16.5 实时规模未验 |
| 仿真 delta → 事务落盘 | 落地 | src/persist.rs:46、pkg/prism_world/Cargo.toml `simulation` |

**prism_replication**
| AOI/兴趣增量/预算 | 落地 | src/aoi.rs:35,229,314 | 只接 prism_transform_ecs |
| ECS 复制 + scene 接缝 | 落地 | src/scene_world.rs:102,131；Cargo.toml:45 `scene_world` | feature 无外部启用者 |
| 测试面 | 落地 | 252 个 #[test] |

#### (b) 汇总（骨架% = (落地+部分+骨架)/声明单元；接通% = 落地/声明单元）
| crate | 相关文档数 | 声明单元 | 落地/骨架/部分/未落地 | 骨架% | 接通% | 最大缺口 |
|---|---|---|---|---|---|---|
| prism_world（门面） | 2 | 4 | 2/0/0/2 | 50% | 50% | §24.3 十项 feature 中仅 2 项存在 |
| prism_world_stream | 9 | 7 | 5/0/0/2 | 71% | 71% | Hilbert、GPU 直传 |
| prism_world_residency | 16 | 10 | 5/4/0/1 | 90% | 50% | §17.2 五个 SDF 消费者全未接 |
| prism_world_persist | 8 | 4 | 4/0/0/0 | 100% | 100% | §11.3 深层结构冲突 |
| prism_world_gpu | 13 | 8 | 6/1/1/0 | 100% | 75% | 无产品 scene/ECS/renderer 接入 |
| prism_world_runtime | 8 | 8 | 7/0/0/1 | 88% | 88% | 无 GPU 直传适配器 |
| prism_world_io | 3 | 3 | 3/0/0/0 | 100% | 100% | GPU 直传（同上前置） |
| prism_world_network | 2 | 4 | 3/0/0/1 | 75% | 75% | 无下游消费者（零集成） |
| prism_world_terrain | 20 | 8 | 3/3/1/1 | 88% | 38% | 体素/VHM/碰撞导出/渲染接入 |
| prism_world_sim | 6 | 3 | 2/1/0/0 | 100% | 67% | 无产品调用 |
| prism_replication | 1 | 3 | 3/0/0/0 | 100% | 100% | 只接 transform_ecs |

#### (c) 文档与代码错位清单
**文档超前于代码**
- §24.3 十项 feature（gpu_stream/async_io/hlod/pcg/sdf/planet/vgeo/vtex/bake_farm）在 pkg/prism_world/Cargo.toml 全缺失，实际只有 scene_host、simulation。
- §24.1 crate 图基本一致，但 §24.2 声明的 io/、manifest/、terrain/、scatter/、sdf/、sim/、pcg/ 目录在子 crate 内不存在。
- §27.1 M9 状态与各切片文档「未扩大边界」不一致（M9 文本未逐字核对，标**未核实**）。
- M4/M5/M7 交付声明为 🟡，端到端验收未提供。
- §17.2 表格声明五个 SDF 消费者——代码中无任何消费者依赖。
**代码超前于文档**
- pkg/prism_world_gpu/src/registry_tests.rs(+registry.rs) 的 proxy 注册/预算测试面（1019+477 行）在 13 篇 GPU 文档中无对应规范文本。
- pkg/prism_world_runtime/src/data_layers.rs（1264 行，DataLayerState:32、LayeredCellFrameCoordinator:466）在 docs/prism_world_system_design_zh.md 内无独立章节引用。
**有文档无 crate / 有 crate 无文档**
- docs/prism_world_gpu_placement_runtime_zh.md 仅约 9 行占位，无对应生产接线代码。
- prism_world_residency 的 clipmap.rs、pcg.rs、qos.rs、sdf.rs 为骨架级孤立实现：有类型与单测，无消费方。

#### (d) 关键结论
1. **骨架远好于接通**。11 crate 骨架率 ≥50%，接通率 ≥88% 的只有 persist/io/replication；terrain 接通率仅 38%（8 项中 5 项为骨架/未落地）。
2. **世界子系统事实上与引擎其余部分隔离**。prism_world_network 是零消费者 crate；GPU/terrain/sim 的「未扩大边界」全部自认为不接 renderer graph / ECS / physics / GI。
3. **§24.3 feature 矩阵是最大的文档-代码裂口**：十项中九项无 Cargo.toml 体现，其中 gpu_stream（§8 GPU 直传）在 runtime/io/gpu 三处均无落地（io_adapter.rs:31,146 仅有 CPU 路径）。
4. **Hilbert 编码与 pcg/sdf/planet 三档 feature 属「文档有、代码零」**，prism_world_stream/src/lib.rs 只实现 Morton（:49,:64）。
5. **精度契约存在已知裂缝**：GPU 侧 f32 近似，与 CPU i64 毫米坐标不逐位对齐（文档自认）；「GPU-first 零回读」尚未达到可验收状态。
6. 静态审计；docs/prism_world_system_implementation_zh.md 内 passed 数字为文档自报，本报告仅用符号存在性 + feature 门控交叉印证。

---

## 2.12 物理・资产・Bevy 桥

#### 口径

- **状态码**：`落地` = 文档要求单元在代码中实现，且存在**被测 crate 之外的、非测试**生产调用者；`骨架` = 类型/模块存在但无 crate 外生产调用者（含仅测试/dev-dependency 调用）；`部分` = 已实现但文档要求的子项/命名/接线缺失，或实现位置与文档不符；`未落地` = 文档明确要求而代码中无对应实现。
- **骨架%/接通%**：`骨架% = (落地+骨架+部分)/声明单元数`（表示"类型/结构层已存在的比例"）；`接通% = 落地/声明单元数`（表示"已接入生产调用链的比例"）。**接通只计 crate 之外的非测试调用者**；测试、`[dev-dependencies]`、bench、同 crate 内调用一律不计。每个 crate 的分母在对应表格的"声明单元数"列中给出。
- **未实测运行**：本次审计为静态审计，**未执行 `cargo build`/`cargo test`**。所有"有回归测试"类结论均来自代码与 `tests/` 文件存在性，不代表测试实际通过；文档中给出的性能数字（§24 基准目标、§21 性能目标）**均未核实**。
- 基准 revision：工作区 `/Users/wangkailong/Documents/GitHub/Prism` 静态快照；文档状态日期多为 2026-10-16。
- `pkg/prism_bevy/` **经实测为空目录**（`ls -la pkg/prism_bevy` 返回仅 `src/`、`tests/` 两个空目录，无 `Cargo.toml`、无任何 `.rs`），故 `pkg/` 实际存在 140 个 crate 而非 141 个目录。

---

#### 表 A：prism_physics_core — docs/prism_physics_design_zh.md（v0.6, 1263 行）

| 要求单元 | 里程碑 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|---|
| §16 离线模式最小真实切片：`SimulationDriver` + Offline/Playback | M0–M8 | 骨架 | `pkg/prism_physics_core/src/driver/mod.rs:245`（`pub struct SimulationDriver`）、`:42-59`（`enum DriveMode { Realtime, Offline{substeps,iterations,target_frames}, Playback(CacheHandle) }`） | 文档写 `prism_physics_core::SimulationDriver`（`docs/prism_physics_design_zh.md:257`），实测文件为 `driver/mod.rs` 而非文档暗示的 `driver.rs`；无 crate 外生产调用者 |
| §16 + derived-data B7：`cache::bake` 离线轨迹烘焙/回放 | M0–M8 | 骨架 | `pkg/prism_physics_core/src/cache/bake.rs:35`（`pub struct BakeConfig`）、`:68`（`pub struct Baker`）、`:118`（`pub fn bake_body`）；同目录 `asset.rs/golden.rs/playback.rs/quantize.rs/track.rs` | `prism_derived_data_contract_zh.md:71` 把 B7 归属 `prism_physics_core::cache::bake`，但无任何 crate 依赖该路径 |
| §28.2 布料边级撕裂（已落地，勿重做）：`tear_flags`/`apply_tearing`/`TearReport`/`TearingParams::sanitized()` | M15 前置 | 骨架 | `pkg/prism_physics_core/src/soft/damage/tearing.rs:82/91/97` 使用 `params.break_strain`；`soft/damage/tearing.rs` 内 `pub fn tear_flag(rest_length, length, break_strain) -> bool`；文档自述见 `docs/prism_physics_design_zh.md:1130-1135` | 仅被 GPU 孪生/测试调用；无 crate 外生产消费者 |
| §28.2 塑性蠕变 `soft::damage::plasticity` | M15 前置 | 骨架 | `pkg/prism_physics_core/src/soft/damage/`（`plasticity` 模块存在，见 `docs/prism_physics_design_zh.md:1134`） | 同上，无外部调用者 |
| §7 CPU golden twin：每个 GPU 内核与 CPU 金标准逐胞元比对 | M5 | 骨架 | `pkg/prism_physics_core/src/cache/golden.rs` | parity 仅由本 crate/GPU crate 测试断言，无生产接线 |
| §21 Crate 拆分与落地形态：core / geometry / gpu 三分 | — | 骨架 | `pkg/prism_physics_core/Cargo.toml`、`pkg/prism_physics_geometry/Cargo.toml`、`pkg/prism_physics_gpu/` 三 crate 均存在 | **文档 §21 代码块（`docs/prism_physics_design_zh.md:406-413`）给出的 7 个 crate 名为 `bevy_prism_core`/`bevy_prism_geometry`/`bevy_prism_backend_cpu`/`bevy_prism_backend_gpu`/`bevy_prism_cache`/`bevy_prism`/`bevy_prism_debug`，与实际的 `prism_physics_*` 命名完全不符** —— 文档与代码命名矛盾（文档代码块未随重命名更新） |
| §21 `TearingParams` 随 crate 导出且阈值来自 params | — | 部分 | `soft/damage/tearing.rs:97` 取 `break_strain: Real` 形参；`TearingParams` 带 `sanitized()`（`docs/prism_physics_design_zh.md:1132`） | 文档暗示统一 `TearingParams` 驱动；实测有 `pub fn tear_flag(rest_length, length, break_strain)` 这种裸标量签名并存，参数来源不唯一 |
| §7/§17 软体内核被下游生产调用（aero / collision） | — | **落地** | `pkg/prism_render_architecture/src/cloth/wind.rs:159,167,200,205,209,270`（`prism_physics_core::soft::aero::{triangle_aero_force, apply_aero_forces, WindField, AeroParams, turbulence_offset}`）；`pkg/prism_render_architecture/src/cloth/collision.rs:140,158,176`（`soft::collision::project_out_of_sphere/half_space/obb`）；`pkg/prism_render_architecture/src/cloth/virtual_particles_jacobi.rs:28,40,71`（`soft::collision as physics_collision`） | 唯一真实生产接通点；但 `cloth/wind.rs:262` 有一处 `#[cfg] reason = "...; exercised by tests and reserved for the wind solver"`，即部分风求解路径仍为"预留"状态 |
| §22 路线图 M0–M8 内核已落地（含 GPU 真机验证） | M0–M8 | 骨架 | 自述 `docs/prism_physics_design_zh.md:7`、`:1263`；模块树 broadphase/bvh/ccd/collide/collider/constraint/dynamics/fluid/fracture/island/joint/lod/mpm/soft/xpbd 等齐备 | "已落地/已上真机验证"为文档单方声明；本次静态审计**未核实**真机验证 |
| §24 M10 动态装配 | M10 | 未落地 | grep `snap_align|AssemblyFeature|snap_point|SnapPose` 于 `pkg/prism_physics_core/src/` → **无命中** | 整个 M10 无代码 |
| §24.5 吸附装配几何原语 | M10/M11 | 未落地 | 同上 grep 无命中；`pkg/prism_physics_geometry/src/` 亦无装配原语模块 | 无代码 |
| §26 M13 关节锥限与布偶（`RagdollBuilder`、`.reduced()`、主动↔被动布偶） | M13 | 未落地 | grep `RagdollBuilder|pub struct Articulation|fn reduced` 于 `pkg/prism_physics_core/src/` → **无命中**（唯一 `reduced_radius` 命中在 `collider/capillary_bridge.rs:107`，与布偶无关） | M13 无代码 |
| §27 M14 降阶坐标铰接（ABA / Featherstone） | M14 | 未落地 | grep `featherstone|articulated_body|aba_` 于 `pkg/prism_physics_core/src/` → **无命中** | M14 全节无代码 |
| §28.3–§28.11 M15 布料拓扑级撕裂（顶点分裂/应力场/裂纹前沿/穿刺/渲染缝合/GPU 拓扑变更） | M15 | 未落地 | `docs/prism_physics_design_zh.md:1145-1196` 要求；代码仅有边级 `tearing`（§28.2 同款），无顶点分裂/半边结构/应力张量分解 | M15 除边断裂保底外全部未落地 |
| §17 集成层 API（Layer 4） | — | 部分 | 章节标题 `docs/prism_physics_design_zh.md:316`；生产接通见 `prism_render_architecture/src/cloth/*` | 无独立的物理 Plugin/SystemSet 装配 crate；文档 §6（bevy 方案）预期的集成 crate 不存在 |

**prism_physics_core 汇总**：声明单元数 **15**｜落地 1 / 骨架 7 / 部分 2 / 未落地 5 → **骨架% = 10/15 = 67%**，**接通% = 1/15 = 7%**

---

#### 表 B：prism_physics_geometry

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §21 独立几何 crate（Layer 1：几何、BVH、GJK/EPA、SDF） | **落地** | `pkg/prism_physics_geometry/Cargo.toml`；生产调用 `pkg/prism_transform_ecs/src/bvh.rs:5`（`use prism_physics_geometry::{Aabb, DynamicBvh, ProxyId}`）、`pkg/prism_audio_geometry/src/scene.rs:28`（`use prism_physics_geometry::{Ray, TriangleMesh}`）、`pkg/prism_audio_geometry/src/lib.rs:16,116` | 无 |
| §21 主类型 Aabb / DynamicBvh / ProxyId / Ray / TriangleMesh 被外部使用 | **落地** | 同上一行 | 无 |
| §24.5 吸附装配几何原语 | 未落地 | grep `snap_align|AssemblyFeature|SnapPose` 无命中 | 无代码 |
| §13/§15 碰撞体与几何查询子集（collider 形状族、GJK/EPA、SDF/程序化几何等） | 骨架 | `pkg/prism_physics_geometry/src/` 子模块齐备（见 `pkg/prism_physics_core/src/collider/*` 对其的复用） | 多数子模块无 crate 外调用者 |
| §21 建议命名 `bevy_prism_geometry` | 部分（矛盾） | 文档 `docs/prism_physics_design_zh.md:407` 写 `bevy_prism_geometry`；实际 `prism_physics_geometry` | 文档代码块未随重命名更新 |

**prism_physics_geometry 汇总**：声明单元数 **5**｜落地 2 / 骨架 2 / 部分 1 / 未落地 1（其中 1 项并入"部分"计数）→ 落地 2、骨架 2、部分 1、未落地 1 = **骨架% = 4/5 = 80%**，**接通% = 2/5 = 40%**

---

#### 表 C：prism_physics_gpu

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §21 wgpu compute 后端（M5：GPU broad-phase / XPBD 投影 / 流体 / MPM） | 骨架 | `pkg/prism_physics_gpu/src/` 顶层 21 个模块：broadphase, buffer, bvh, cfl, cloth, collider, contacts, context, fluid, fracture, grid, indirect, island, mpm, narrowphase, radix, rigid, scan, shaders, vbd, xpbd | **无任何 crate 依赖它**（grep `prism_physics_gpu` 于 `pkg/*/Cargo.toml`、`crates/*/Cargo.toml`、`apps/*/Cargo.toml` 仅命中自身 `Cargo.toml:2` 与自 dev-dep `:20`）→ 接通为 0 |
| §7 CPU golden + 真机 GPU parity 双重验证 | 骨架 | `pkg/prism_physics_gpu/src/rigid/joint/*_cpu.rs` 系列（d6_cpu.rs:104、universal_cpu.rs:70、hinge_limit_cpu.rs:56、revolute_servo_cpu.rs:100、elliptical_cone_twist_cpu.rs:103、swing_twist_cpu.rs:85、cylindrical_cpu.rs:285、d6_drive_cpu.rs:106 等） | parity 仅测试层 |
| §17 Bevy 集成层：GPU 后端接入下游装配 | 未落地 | 无依赖者；无 Plugin/SystemSet | 完全无集成 |
| §17/§21 隐含"无 Bevy 依赖的纯物理内核" | 部分（矛盾） | `pkg/prism_physics_gpu/Cargo.toml:12-14` 依赖 `bevy_math`（path 指向 vendored 上游 Bevy crate） | 见"Bevy 耦合实测" |
| §21 M5 后端 feature 门控（`bevy_prism_backend_gpu // wgpu compute 后端 (feature)`） | 骨架 | `pkg/prism_physics_gpu/Cargo.toml` `[features] default = []` | 与文档命名不符，且默认关闭 |

**prism_physics_gpu 汇总**：声明单元数 **5**｜落地 0 / 骨架 3 / 部分 1 / 未落地 1 → **骨架% = 4/5 = 80%**，**接通% = 0/5 = 0%**

---

#### 表 D：prism_asset — docs/prism_asset_design_zh.md（v0.2, 1104 行）

Denominator 构成：§19 模块布局 9 + §19 feature 分层 8 + §19 依赖清单 6 + §20 契约不变量 7 + §23 AAA 增补 8 + §26 API 人体工学 7 + §21 路线图 M0–M6 7 = **52**

| D1. §19 规定的 `src/` 模块（9） | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| `id.rs` AssetIndex/AssetId/UntypedAssetId/AssetTypeId | 落地 | `pkg/prism_asset/src/id.rs` | — |
| `guid.rs` StableGuid | 落地 | `pkg/prism_asset/src/guid.rs` | — |
| `path.rs` AssetPath + source:// + #label | 落地 | `pkg/prism_asset/src/path.rs:1,7,14,22-23,30,35,41`（`label: Option<String>`） | — |
| `handle.rs` Handle/WeakHandle/SoftHandle/UntypedHandle/ReleaseQueue | 落地 | `pkg/prism_asset/src/handle.rs` | — |
| `storage.rs` Assets\<A\>（reserve/fulfill/fail/collect_releases） | 落地 | `pkg/prism_asset/src/storage.rs` | — |
| `types.rs` AssetTypes 注册 + `Asset` trait | 部分 | 功能被拆分到 `pkg/prism_asset/src/asset.rs`、`type_id.rs`、`stores.rs` | 文档规定的单文件 `types.rs` 不存在（功能等价，位置不符） |
| `dependency.rs` DependencyGraph（增量就绪 + 失效传播 + 环检测） | 落地 | `pkg/prism_asset/src/dependency.rs` | — |
| `load_state.rs` LoadState/RecursiveDependencyLoadState/ErrorRegistry | 落地 | `pkg/prism_asset/src/load_state.rs` | — |
| `event.rs` AssetEvent（含 Failed/Unused） | 落地 | `pkg/prism_asset/src/event.rs` | — |

| D2. §19 规定的 feature 分层（8） | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| `std` | 落地 | `pkg/prism_asset/Cargo.toml` `[features] std = ["dep:prism_tasks", "dep:rustix"]`；被 `prism_image`/`prism_scene` 等生产 crate 启用 | — |
| `ecs`（Assets\<A\> 实现 `prism_ecs::Resource`） | 落地 | `pkg/prism_asset/Cargo.toml` `ecs = ["dep:prism_ecs"]`；生产启用方 `pkg/prism_render_app/Cargo.toml:12`（`prism_asset = { path = "../prism_asset", features = ["ecs"] }`） | — |
| `async_io` | 部分 | 文档 `docs/prism_asset_design_zh.md:119`（能力表 `std` + `async_io`）；实际无同名 feature，异步能力置于 `std` 下（`src/streaming/async_io.rs`） | 命名/分层与文档不符 |
| `hot_reload` | 部分 | 文档 `docs/prism_asset_design_zh.md:120` 规定同名 feature；实际 feature 名为 `file_watcher`（`Cargo.toml` `file_watcher = ["std", "dep:notify-debouncer-full"]`），实现于 `src/watch.rs`/`reload.rs`/`reload_plan.rs` | feature 名与文档不符；文档 `docs/prism_asset_design_zh.md:20` 自认"当前实际 feature 仅提供 `std`、`ecs`、`file_watcher`" |
| `process`（AssetProcessor + 内容哈希缓存 + 单文件资产包） | 未落地 | 文档 `docs/prism_asset_design_zh.md:121`、`:923`（src 布局注释 `process # AssetProcessor + 内容哈希缓存 + 单文件资产包`）；`Cargo.toml` 无 `process` feature | 该能力被移到 `prism_asset_build_core`，本 crate 未落地 |
| `pack`（ContainerIo + 挂载优先级） | 部分 | 文档 `docs/prism_asset_design_zh.md:154`、`:563` 规定 `feature = "pack"`；实现于独立 crate `pkg/prism_asset_container`（见 `pkg/prism_asset_container/src/overlay.rs:46`），非 `prism_asset` 的 feature | 分层位置改变，`prism_asset` 侧无 `pack` feature |
| `gpu_stream`（DirectStorage 形态队列 + GDeflate） | 部分 | 文档 `docs/prism_asset_design_zh.md:123`、`:633`；`prism_asset` 侧有 `src/streaming/page_residency.rs`、`range_assembly.rs`；GDeflate 在 `pkg/prism_gdeflate_gpu`（文档自认 `docs/prism_asset_design_zh.md:640` "目标规格，尚未完整实现"） | 完整 GPU 流送未闭环 |
| `retain_cache`（KeepAlive/LRU/Pinned 卸载策略） | 未落地 | 文档 `docs/prism_asset_design_zh.md:264`、`:786`；`Cargo.toml` 无 `retain_cache` feature | 无代码 |

| D3. §19 规定的依赖（6） | 状态 | 证据 | 缺口 |
|---|---|---|---|
| `prism_tasks` | 落地 | `pkg/prism_asset/Cargo.toml`（`std` feature 内 `dep:prism_tasks`） | — |
| `prism_utils`（哈希/位打包/MPSC 队列） | 未落地 | `pkg/prism_asset/Cargo.toml` `[dependencies]` 实测仅 `prism_ecs`(optional)、`prism_tasks`(optional)、`notify-debouncer-full`(optional)、`rustix`(optional) | 文档 `docs/prism_asset_design_zh.md:938` 要求，实际无此依赖 |
| `prism_reflect`（settings/meta） | 未落地 | 同上，`prism_asset` 不依赖 `prism_reflect` | 直接导致 §26"编辑器/脚本暴露"无法落地 |
| `prism_diagnostic`（计数/trace） | 未落地 | 同上 | 无 |
| `prism_gdeflate_gpu`（可选） | 未落地 | 同上 | 无 |
| `prism_platform`（FS watch，可选） | 未落地 | 同上；改用 `rustix` + `notify-debouncer-full` | 依赖选择与文档不符 |

| D4. §20 契约与不变量（7） | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| AssetIndex 代际：槽位回收 generation+1，旧 id 永不别名新占用 | 落地 | `pkg/prism_asset/src/id.rs`、`src/storage.rs`；下游生产使用方 `prism_image`/`prism_scene`/`prism_render_app` | — |
| StableGuid 稳定：跨构建/平台一致，规则冻结 | 落地 | `pkg/prism_asset/src/guid.rs:11,93` | — |
| 句柄身份不变：热重载复用同 AssetIndex，只发 `Modified` | 落地 | `pkg/prism_asset/src/reload_plan.rs`、`src/reload.rs` | — |
| 回收单点：只发生在 `collect_releases`；`Unused` 先、`Removed` 后 | 落地 | `pkg/prism_asset/src/storage.rs`、`src/event.rs` | — |
| 拓扑/烘焙确定：双跑一致；处理器版本号单调递增 | 骨架 | `pkg/prism_asset/src/hash.rs` | 双跑纪律的唯一真相在 build 侧（`docs/prism_asset_design_zh.md:1089` 指向 build §18.1），本 crate 无门禁 |
| 类型安全：`UntypedAssetId` 带 `AssetTypeId`，错类型 `typed::<B>()` 不静默命中 | 落地 | `pkg/prism_asset/src/type_id.rs:13`、`src/id.rs` | — |
| 版本化契约：StableGuid 规则/容器 ToC/内容目录/资产包 schema/AssetEvent 变体 | 骨架 | `src/guid.rs`、`src/event.rs` | 无 schema 版本迁移机制接线 |

| D5. §23 AAA 高级功能增补（8） | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §23.1 子资产与标签图（`parentGuid + label` 派生 StableGuid） | 落地 | `pkg/prism_asset/src/path.rs:7,22-30,35`；下游生产使用 `prism_image`/`prism_render_app` | — |
| §23.2 软引用与引用闭包预算（`load_soft` 每帧 N 个分批） | 部分 | `pkg/prism_asset/src/server.rs:867`（`pub fn load_soft<A: Asset>(&self, soft: &SoftHandle<A>) -> Handle<A>`）、`src/server.rs:137-140`（`catalog: BTreeMap<StableGuid, AssetPath>`）；测试 `src/tests.rs:2086,2096,2107` | `load_soft` 存在，但**无"每帧 N 个分批预算"API**（未发现 soft-load budget 类型）；只有测试调用者 |
| §23.3 占位/兜底资产（placeholder，`get` 返回类型化兜底而非 `None`） | 未落地 | grep `placeholder` 于 `pkg/prism_asset/src/*.rs` → **无命中**；`src/loader.rs:30` 注释明确写 "separate fallback layered on top by the execution crate; this module models ..."，即本 crate 声明不做 | 文档要求，代码显式排除 |
| §23.4 内容目录与远程分发（`StableGuid → ChunkId → 容器`） | 部分 | 本 crate 有 `guid → path` 目录（`src/server.rs:137-140,242,358`）；`ChunkId`/容器侧在 `pkg/prism_asset_container/src/format.rs` | 缺 `StableGuid→ChunkId` 的内容目录与远程 CDN/本地缓存混合层 |
| §23.5 patch / DLC 叠挂（多容器优先级覆盖） | 落地 | `pkg/prism_asset_container/src/overlay.rs:46`；`pkg/prism_asset_build_core/src/runtime_patch.rs`、`patch.rs`；生产使用方 `pkg/prism_asset_build_cli/src/patch.rs` | — |
| §23.6 虚拟纹理/虚拟几何页流送对接（`read_range` 按 page 拉取，GPU 直通解压直灌页缓存） | 部分 | `pkg/prism_asset/src/streaming/page_residency.rs`、`src/streaming/range_assembly.rs`；文档自认 GPU 直通未完成（`docs/prism_asset_design_zh.md:640`） | GPU 直灌页缓存未落地 |
| §23.7 不可信容器/网络字节的安全边界（长度/偏移界内、哈希、解压炸弹上限、返 `AssetErrorId` 不 panic/OOM） | 部分 | 容器侧严格校验：`pkg/prism_asset_container/src/format.rs:30` 起（ToC/Chunk Table/范围/哈希/对齐/零填充校验）；错误类型 `pkg/prism_asset_container/src/manifest.rs`（`ManifestReadError`，见 `pkg/prism_image/src/tests.rs:437,488`） | 文档 §25（`docs/prism_asset_design_zh.md:1053`）要求的 **fuzz 常驻语料库 + 崩溃回归目标不存在**（工作区无 `fuzz/` 目录）；"解压后尺寸上限"防解压炸弹**未核实** |
| §23.8 加载预算与首帧墙（`Immediate` vs `Streaming/Prefetch`） | 未落地 | grep `LoadPriority` / `Immediate` / `Prefetch` 于 `pkg/prism_asset/src/scheduler.rs` → **无命中** | 文档 §9.2（`docs/prism_asset_design_zh.md:470`）规定的 `LoadPriority` 四级枚举在代码中不存在 |

| D6. §26 API 人体工学（7, `docs/prism_asset_design_zh.md:1073-1100`） | 状态 | 证据 | 缺口 |
|---|---|---|---|
| `#[derive(Asset)]` 派生宏（自动 `asset_type()` + `visit_dependencies`） | 未落地 | 全工作区无 `prism_asset_macros` crate；`#[derive(Asset)]` 仅作为文档注释文本出现，无 proc-macro 实现 | 整个派生宏缺失；这是"消除忘登记依赖"的关键 ergonomics 项 |
| 类型化系统参数（`Res<AssetServer>` / `ResMut<Assets<Mesh>>` / `EventReader`） | 骨架 | `ecs` feature 使 `Assets<A>` 实现 `prism_ecs::Resource`（`pkg/prism_asset/Cargo.toml`；启用方 `pkg/prism_render_app/Cargo.toml:12`） | 无 `AssetServer` 作为 `Resource` 的注入式参数层 |
| 就绪即回调（`server.when_loaded(handle, ..)` / `handle.loaded().await`） | 未落地 | grep `when_loaded` / `fn loaded` 于 `pkg/prism_asset/src/` → **无命中** | 只能走事件轮询 |
| 类型安全防呆（`AssetId<A>` 类型标签、`SoftHandle::resolve` 保留期望类型） | 落地 | `pkg/prism_asset/src/id.rs`、`src/type_id.rs:13`、`src/handle.rs` | — |
| `prism_asset::prelude`（一处导出含**派生宏**） | 部分 | `pkg/prism_asset/src/lib.rs` 有 `prelude`（从 b9 记录：`prism_asset::prelude` 单点导出）；文档要求的"派生宏"项无法导出（见上一行） | prelude 存在但缺派生宏；`AssetLoader`/`LoadContext` 导出**未核实** |
| 清晰错误（`AssetErrorId → {path, source, reason, dependents}`，按根因聚合） | 落地 | `pkg/prism_asset/src/error.rs`（`AssetErrorId` + `ErrorRegistry`，见 `src/load_state.rs`） | 面板聚合属编辑器侧，未接线 |
| 编辑器/脚本暴露（经 `prism_reflect` 暴露句柄/加载态/依赖关系） | 未落地 | `prism_asset` **不依赖 `prism_reflect`**（`pkg/prism_asset/Cargo.toml` 实测） | 依赖缺失导致该单元无法成立 |

| D7. §21 路线图 M0–M6（7, `docs/prism_asset_design_zh.md:958-964`） | 状态 | 文档自述 | 缺口 |
|---|---|---|---|
| M0 身份/存储/依赖核（已交付） | 落地 | `:958` | 被 `prism_image`/`prism_scene`/`prism_render_app` 生产使用 |
| M1 内核重铸（已交付） | 落地 | `:959` | 同上 |
| M2 加载管线（部分交付） | 部分 | `:960`："EDL 作业图、range-backed 装配、真正可抢占阶段仍待交付" | — |
| M3 热重载（部分交付） | 部分 | `:961`："完整 labeled revision 生命周期、可抢占阶段、跨平台安全打开语义仍待交付" | — |
| M4 流送（部分交付） | 部分 | `:962`："GPU feedback 真实生成/读回/解码、StreamingManager 预取、GPU page-table、块级解码、DirectStorage 队列、完整 GDeflate 解压仍待交付" | — |
| M5 资产包消费与容器挂载（部分交付） | 部分 | `:963`："完整 block-level Chunk wire contract、压缩块索引/加密/签名、GPU 直解、Cook/发行闭环仍待交付" | — |
| M6 集成（目标） | 未落地 | `:964`："ECS、渲染驻留、场景/编辑器与 Asset Build 制品的端到端接线" | 文档自标"目标"，即未交付 |

**prism_asset 汇总**：声明单元数 **52**｜落地 22 / 骨架 3 / 部分 14 / 未落地 13 → **骨架% = 39/52 = 75%**，**接通% = 22/52 = 42%**
（生产调用者证据：`pkg/prism_render_app/Cargo.toml:12`、`pkg/prism_image/src/runtime_loader.rs:11` 等）

---

#### 表 E：prism_asset_container

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| v1 raw / per-payload Deflate 容器 reader/writer | 落地 | `pkg/prism_asset_container/src/reader.rs`、`src/writer.rs`；被 `prism_asset_build_core`、`prism_asset_build_cli`、`prism_image` 生产依赖（各自 `Cargo.toml`） | — |
| v2 独立 Chunk Table + 每 Chunk 独立 Raw/Deflate | 落地 | `pkg/prism_asset_container/src/format.rs`、`src/chunk.rs`；下游 `pkg/prism_image/Cargo.toml:19` | — |
| 严格 ToC/Chunk Table/范围/哈希/对齐/零填充校验 | 落地 | `pkg/prism_asset_container/src/format.rs:30` 起；错误路径 `ManifestReadError`（`pkg/prism_image/src/tests.rs:437,488`） | — |
| `RangeSource`（positional I/O、短读/截断门禁、并发 range read） | 落地 | `pkg/prism_asset_container/src/range_source.rs:51`；生产使用 `pkg/prism_image/src/runtime_loader.rs:11` | — |
| `MmapRangeSource`（可选 `mmap` feature 只读） | 骨架 | `pkg/prism_asset_container` 的 `mmap` feature 路径 | 无 crate 外生产启用者 |
| overlay / 挂载优先级（同名容器 remount 替换） | 落地 | `pkg/prism_asset_container/src/overlay.rs:46`；`pkg/prism_image/src/runtime_loader.rs:11`（`use prism_asset_container::{ManifestReadError, OverlaySet, RuntimeManifest}`） | — |
| Runtime Manifest v3（容器身份 + route 级硬/可选依赖 + 初始驻留/streaming 层级 + codec/schema + 安装能力位图） | 落地 | `pkg/prism_asset_container/src/manifest.rs:41`；生产使用见 `pkg/prism_image/src/runtime_loader.rs:11` | — |
| `ManifestReader` / `ManifestAssetReader` + `RangeBatchPolicy` 确定性批量 range 调度 | 落地 | `pkg/prism_asset_container/src/manifest_reader.rs:45`、`src/range_batch.rs:48` | — |
| Patch / `RuntimePatchDescriptor` | 骨架 | `pkg/prism_asset_container/src/patch.rs:47` | 生产 patch 路径走 `prism_asset_build_cli`，容器侧仅类型 |
| install / `RuntimeInstallPolicy` / rollback / recovery record | 骨架 | `pkg/prism_asset_container/src/install.rs:34` | 未接入生产安装流程 |
| block-level 压缩索引 | 未落地 | 文档 `docs/prism_asset_build_design_zh.md:54`"尚未满足"列明 | 无代码 |
| 加密 / 签名信任根 | 未落地 | 同上 | 无代码 |
| GPU 直解（DirectStorage / GPU inflate） | 未落地 | 文档 `docs/prism_asset_design_zh.md:640` 自认未完整实现 | 无代码 |
| 生产接通（runtime loader / world runtime） | 落地 | `pkg/prism_image/src/runtime_loader.rs:11`；`pkg/prism_world_runtime/Cargo.toml:17`（`prism_asset_container = { path = "../prism_asset_container", optional = true }`）、`:27`（`asset_container = ["std", "dep:prism_asset", "dep:prism_asset_container"]`） | — |

**prism_asset_container 汇总**：声明单元数 **14**｜落地 8 / 骨架 3 / 部分 0 / 未落地 3 → **骨架% = 11/14 = 79%**，**接通% = 8/14 = 57%**

---

#### 表 F：prism_asset_build_core — docs/prism_asset_build_design_zh.md（1478 行）+ prism_asset_build_core_coordination_zh.md + prism_derived_data_contract_zh.md

Denominator：可执行验收矩阵 15 行（`docs/prism_asset_build_design_zh.md:42-56`）+ 协调切片 3 单元（`docs/prism_asset_build_core_coordination_zh.md`）+ 派生数据登记表中的 `prism_bake_*` 归属 1 单元 = **19**

| 要求单元（验收矩阵，line） | 文档状态 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|---|
| Artifact 身份、确定性 DAG 规划与处理器执行 (`:42`) | 已实现 | 落地 | `pkg/prism_asset_build_core/src/hash.rs`、`src/graph.rs` | 完整导入插件协议、字段级失效分析、百万资产规模证据未完成（文档自述） |
| DAG wave 并行 (`:43`) | 已实现 | 落地 | `src/graph.rs`（`execute_parallel`/`execute_parallel_with_limit`） | 该 API 不含 DDC/CAS/Revision 发布（文档自述边界） |
| BuildExecutor 并行执行 (`:44`) | 已实现 | 落地 | `src/executor.rs`、`src/executor/wave_results.rs` | 不代表跨进程/远程调度 |
| RuntimeDerived 进程内 `BakeService` (`:45`) | 已实现 | 落地 | `src/bake.rs` | 未接入 ECS/渲染/编辑器消费（文档自述） |
| DirtySet 增量 BuildExecutor (`:46`) | 已实现 | 落地 | `src/executor.rs`、`src/dirty.rs` | — |
| 双跑确定性发布门禁 (`:47`) | 已实现 | 落地 | `src/executor/determinism.rs`；`BuildExecutor::execute_verified_with_limit` | 仅本机同进程 |
| 本地 CAS、DDC 与 BuildExecutor 缓存校验 (`:48`) | 部分实现 | 部分 | `src/cas.rs`、`src/ddc.rs`、`src/executor.rs` | 缺完整字段级失效、远程 CAS/worker、非 Unix 跨进程协调、大规模性能证据 |
| Artifact 与 Manifest 字节预算 (`:49`) | 已实现 | 落地 | `src/validation.rs`（`max_artifact_bytes`/`max_total_manifest_bytes`） | — |
| SourceSnapshot / Recipe / PNG 导入垂直链 (`:50`) | 部分实现 | 部分 | `src/source.rs`、`src/recipe.rs`、`src/import/image.rs`、`src/import/mod.rs` | 仅 PNG 垂直切片，非通用 Cook |
| ImporterPlugin / ImporterRegistry (`:51`) | 已实现 | 落地 | `src/import/mod.rs`、`src/import/image.rs` | 非完整 importer 生态/动态加载/隔离 worker |
| 本地 Revision 事务与恢复 (`:52`) | 部分实现 | 部分 | `src/transaction.rs`、`src/journal.rs`、`src/publication.rs` | 缺完整磁盘故障注入、非 Unix 等价锁、远程幂等重试、事务级持久化 registry |
| 本地发布完整性与单调 Revision gate (`:53`) | 部分实现 | 部分 | `src/release.rs`、`src/provenance.rs` | 缺密钥托管/轮换、跨机器信任根、完整 anti-rollback 服务 |
| Runtime Manifest / raw 容器打包桥接 (`:54`) | 部分实现 | 部分 | `src/package.rs`、`src/runtime_patch.rs`、`src/provenance.rs` | 缺 block-level 压缩索引、项目级策略、加密、DLC 闭包、远程发布/回滚 |
| CLI (`:55`) | 部分实现 | 部分 | 见"表 G" | CLI 本身仅覆盖 PNG 工作流 |
| GPU 直通/解压、远程 Build Farm、完整 Cook/发行链 (`:56`) | **未实现** | 未落地 | 文档自标"未实现"；`prism_gdeflate_gpu::GDeflateDecoder` 仅有 de-interleave 子集 | 完整 GPU Huffman、远程 Build Farm、完整发行链无代码 |
| 协调切片：`transaction.rs` 用 Unix 稳定锁文件 + `flock` 跨进程排他 | 已完成 | 落地 | `src/transaction.rs:127`（"Cross-process publication lease. On Unix this is an advisory `flock`"）、`:141`（`rustix::fs::flock(..LockExclusive)`）、`:152`（Unlock） | 非 Unix 无验证后端（fail-closed） |
| 协调切片：`ddc.rs` 非 Unix `CoordinationUnavailable` fail-closed | 已完成 | 落地 | `src/ddc.rs:32`（`CoordinationUnavailable(&'static str)`）、`:107`、`:356`（flock）、`:375`、`:60`（错误转文本） | 同上 |
| 协调切片：`#[cfg(not(unix))]` 单测验证明确错误 | 已完成 | 落地 | `src/transaction.rs:2267`（`Err(PublishError::CoordinationUnavailable(_))`）、`src/ddc.rs:956`（`Err(DdcError::CoordinationUnavailable(_))`） | — |
| derived-data 登记表要求的 `prism_bake_*` 内核 crate 归属 | — | 未落地 | `docs/prism_derived_data_contract_zh.md:47-58` 归属 `prism_bake_texture`/`prism_bake_mesh`/`prism_bake_shader`/`prism_bake_animation`/`prism_bake_audio`；实测 `ls -d pkg/prism_bake_*` → **No such file or directory** | 5 个规定 crate 全部不存在 |

**prism_asset_build_core 汇总**：声明单元数 **19**｜落地 11 / 骨架 0 / 部分 6 / 未落地 2 → **骨架% = 17/19 = 89%**，**接通% = 11/19 = 58%**
（唯一 crate 外生产调用者：`pkg/prism_asset_build_cli/Cargo.toml:13`）

---

#### 表 G：prism_asset_build_cli

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| `plan` 子命令 | 落地 | `pkg/prism_asset_build_cli/src/args.rs:16`（`Plan(Plan)`）、`:25`（`#[argh(subcommand, name = "plan")]`） | — |
| `import` 子命令 | 落地 | `src/args.rs:17`、`:39`（`name = "import"`）、`:54`（`verify_determinism` 开关） | — |
| `build` 子命令 | 落地 | `src/args.rs:18`、`:61`（`name = "build"`） | — |
| `package` 子命令 | 落地 | `src/args.rs:19`；`src/pipeline.rs`、`src/mapping.rs` | 文档 §16.4 要求的 `cook`/`diff` 未覆盖 |
| `validate` 子命令 | 落地 | `src/args.rs:20`；`src/patch.rs` | — |
| `patch` 子命令 | 落地 | `src/args.rs:21`；`src/patch.rs` | — |
| `inspect` 子命令 | 落地 | `src/args.rs:22` | — |
| `cook` 子命令 | 未落地 | 文档 `docs/prism_asset_build_design_zh.md:217` 规定 `# import/build/cook/package/inspect/diff`；`Command` 枚举（`src/args.rs:15-23`）无 `Cook` | 无 |
| `diff` 子命令 | 未落地 | 同上 | 无 |

**prism_asset_build_cli 汇总**：声明单元数 **9**｜落地 7 / 骨架 0 / 部分 0 / 未落地 2 → **骨架% = 7/9 = 78%**，**接通% = 0/9 = 0%**
> 口径说明：`prism_asset_build_cli` 是**二进制生产入口**（`src/main.rs`），无上游 crate 调用它，因此按"crate 外非测试调用者"口径 接通% 为 0 —— 这不代表它未在生产路径上，仅表示它位于调用链顶端。

---

#### 表 H：prism_audio_bevy

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| gap 文档要求"音频世界的 ECS 对接层，重接到 `prism_ecs`"（`docs/prism_engine_component_gap_zh.md:57`） | 骨架 | `pkg/prism_audio_bevy/src/lib.rs:11`（`pub use prism_audio_ecs::*;`）；`:16-18` 显式重导出 `budget, client_resource, commands, emitter, listener, master_gain, playback, player, player_systems, plugin, systems, telemetry, voice_registry, volume`；被桥接的 `prism_audio_ecs` 依赖全为原生 crate（`pkg/prism_audio_ecs/Cargo.toml`：`prism_app/prism_ecs/prism_math/prism_transform/prism_audio_core/prism_audio_rt/prism_audio_assets/prism_scene`） | 实现已迁出本 crate；本 crate 仅 19 行 re-export，无自身逻辑；**无任何 crate 依赖 `prism_audio_bevy`** |
| gap 文档称其依赖 `app/ecs/transform`（`:57`） | 部分（文档过时） | 实测 `pkg/prism_audio_bevy/Cargo.toml`：`[dependencies]` 仅 `prism_audio_ecs`；`prism_app`/`prism_ecs`/`prism_math`/`prism_transform`/`prism_audio_core`/`prism_audio_rt` 全部位于 `[dev-dependencies]` | 文档描述的依赖关系已不成立；`lib.rs:5-7` 注释自述"the audio control plane uses `prism_app`, `prism_ecs`, `prism_math`, and `prism_transform` instead of Bevy runtime crates" |
| 旧公开路径保持稳定（兼容外观，不做第二实现） | 骨架 | `pkg/prism_audio_bevy/src/lib.rs:1-2,11-18`（注释："This is a path-level compatibility bridge, not a second owner of any audio state or systems"） | 无外部消费者，兼容性目标无证据 |
| 生产接通（下游使用该 facade） | 未落地 | grep `prism_audio_bevy` 于 `pkg/*/Cargo.toml`、`apps/*/Cargo.toml`（排除自身）→ **无命中** | 0 个生产依赖者 |
| 唯一所有权边界（音频 ECS 状态单点） | 骨架 | `pkg/prism_audio_ecs/` 持有实现（由 `lib.rs:5` 指明） | 不在本次分配范围内，未深审 |

**prism_audio_bevy 汇总**：声明单元数 **5**｜落地 0 / 骨架 3 / 部分 1 / 未落地 1 → **骨架% = 4/5 = 80%**，**接通% = 0/5 = 0%**

---

#### 表 I：prism_bevy（空目录）— docs/prism_bevy_refactor_plan_zh.md（245 行）

**实测确认**：`ls -la pkg/prism_bevy` → 仅 `src/`、`tests/` 两个**空**目录，无 `Cargo.toml`、无 `.rs` 文件。

| 要求单元 | 状态 | 证据（file:line） | 缺口 |
|---|---|---|---|
| §6 下游集成 crate `prism_bevy`（渲染）作为装配层 | 未落地 | `docs/prism_bevy_refactor_plan_zh.md:170` 架构图规定 `下游集成 crate：prism_bevy(渲染) · prism_bevy_ui · prism_bevy_audio · prism_bevy_physics`；实测 `ls -d pkg/*bevy*` → 仅 `pkg/prism_audio_bevy`、`pkg/prism_bevy`（空） | 规定 crate 不存在于代码 |
| §6 `prism_bevy_ui` | 未落地 | `ls -d pkg/prism_bevy_ui` → `No such file or directory` | 无 |
| §6 `prism_bevy_audio` | 未落地 | `ls -d pkg/prism_bevy_audio` → `No such file or directory`（注：`prism_audio_bevy` 命名不同且为空 facade） | 无 |
| §6 `prism_bevy_physics` | 未落地 | `ls -d pkg/prism_bevy_physics` → `No such file or directory` | 无 |
| §2/§12 旧 `prism_bevy` 可见性插件"完成迁移审计并退役" | 落地 | 自述 `docs/prism_bevy_refactor_plan_zh.md:30`（"原下游集成 crate `pkg/prism_bevy` 已完成迁移审计并退役；可见性能力由 native Prism crate 承载"）、`:184`（S2 "✅ 完成并退役"）、`:241-242` | **空目录是退役后的残留**，与文档一致（非矛盾）；但残留空目录应清理 |
| §7 S1：`prism_render_visibility` 接 `bevy` 桥接 | 落地 | `pkg/prism_render_visibility/src/bevy_bridge.rs`；`pkg/prism_render_visibility/Cargo.toml:16`（`bevy = ["dep:bevy_camera", "dep:bevy_transform"]`） | — |
| §7 S2：可见性插件退役（能力迁 native Prism crate） | 落地 | 空目录 `pkg/prism_bevy/`；文档 `:184`、`:242` | — |
| §2 "已接 `bevy` 桥接的 pkg 仅 2 个：`prism_render_material`、`prism_render_visibility`（均含 `src/bevy_bridge.rs`）" | 落地（已核实） | `find pkg -name 'bevy_bridge.rs'` → 恰好 2 个：`pkg/prism_render_material/src/bevy_bridge.rs`、`pkg/prism_render_visibility/src/bevy_bridge.rs` | — |
| §7 S3–S14（12 个切片，材质/场景提取/着色光照/虚拟几何/体积头发/Render Graph/音频/物理/UI×4） | 未落地 | 文档 `:185-198` 状态列全为"计划" | 除 S1/S2 外全部未开始 |
| §6 "不能让 `crates/bevy_*` 直接依赖 `pkg/prism_*`"（禁止依赖环） | 未核实 | 文档 §2 `:29` 自称"已核实 `rg prism_ crates/*/Cargo.toml` 为空" | 本次**未复核** `crates/*/Cargo.toml` |
| gap 文档：`prism_bevy` "脱 Bevy 后整体被 `prism_app` 取代/废弃"（`docs/prism_engine_component_gap_zh.md:56`） | 部分 | 空目录 + `pkg/prism_app` 存在 | "被 prism_app 取代"的映射关系无显式证据 |

**prism_bevy 汇总**：声明单元数 **11**｜落地 4 / 骨架 0 / 部分 1 / 未落地 5 / 未核实 1 → **骨架% = 5/11 = 45%**，**接通% = 4/11 = 36%**

---

#### Bevy 耦合实测

**§1 分配范围内 8 个存在 crate 的 bevy 依赖实测（`grep -n "bevy" pkg/<crate>/Cargo.toml`）**

| crate | 文件:行 | 原文（依赖行） | 判定 |
|---|---|---|---|
| prism_physics_core | `pkg/prism_physics_core/Cargo.toml` | 无 `bevy` 匹配（deps: `glam 0.33.2`, `prism_physics_geometry`, `serde`(opt), `rayon`(opt)） | **解耦** |
| prism_physics_geometry | `pkg/prism_physics_geometry/Cargo.toml` | 无 `bevy` 匹配 | **解耦** |
| **prism_physics_gpu** | **`pkg/prism_physics_gpu/Cargo.toml:12`** | `bevy_math = { path = "../../crates/bevy_math", default-features = false, features = [`<br>`  "nostd-libm",`<br>`] }`（占 :12-14） | **耦合（唯一）** |
| prism_asset | `pkg/prism_asset/Cargo.toml:6` | 命中仅为 `description` 字符串内 "no `bevy_*` crate and no Unreal/Unity source or derived code"（**非依赖**） | **解耦** |
| prism_asset_build_cli | `pkg/prism_asset_build_cli/Cargo.toml` | 无 `bevy` 匹配 | **解耦** |
| prism_asset_build_core | `pkg/prism_asset_build_core/Cargo.toml` | 无 `bevy` 匹配 | **解耦** |
| prism_asset_container | `pkg/prism_asset_container/Cargo.toml` | 无 `bevy` 匹配 | **解耦** |
| prism_audio_bevy | `pkg/prism_audio_bevy/Cargo.toml:2` | 命中仅为 `name = "prism_audio_bevy"`（**非依赖**） | **解耦** |
| prism_bevy | 无 `Cargo.toml`（空目录） | — | 不适用 |

**§2 `crates/bevy_*` path 依赖（分配范围内仅 1 条）**

| 依赖方 | 依赖行 | 指向 | 性质 |
|---|---|---|---|
| `pkg/prism_physics_gpu/Cargo.toml:12-14` | `bevy_math = { path = "../../crates/bevy_math", ... }` | `crates/bevy_math` | **上游 Bevy 原版 crate**：`crates/bevy_math/Cargo.toml` 头部为 `name = "bevy_math"`, `version = "0.20.0-dev"`, `description = "Provides math functionality for Bevy Engine"`, `homepage = "https://bevy.org"`, `repository = "https://github.com/bevyengine/bevy"`，并**传递依赖 `bevy_reflect`（path ../bevy_reflect）、`bevy_macro_utils`、`variadics_please`** → 该依赖把整个 Bevy crate 子树拉入编译图 |

**§3 该依赖的实际使用面**：`pkg/prism_physics_gpu/src/` 内 **11 个文件** `use bevy_math::ops`（共 368 个 `.rs` 文件），集中在 `rigid/joint/*_cpu.rs`：`d6_drive_cpu.rs:106`、`cylindrical_cpu.rs:285`、`cylindrical_drive_cpu.rs:457`、`universal_cpu.rs:70`、`hinge_limit_cpu.rs:56`、`revolute_servo_cpu.rs:100`、`elliptical_cone_twist_cpu.rs:103`、`cylindrical_limit_cpu.rs:391`、`d6_cpu.rs:104`、`swing_twist_cpu.rs:85`、`revolute_drive_cpu.rs:96`。使用理由是**确定性 `f32` 超越函数**（如 `d6_drive_cpu.rs:81` 注释："`bevy_math::ops::atan2` rather than `f32::atan2` so the CPU reference and …"）——即为与 GPU 位等对齐而借用 `bevy_math::ops`。

**§4 仓库级上下文（超出分配范围，仅作背景）**：`bevy_math` path 依赖在 `pkg/` 内被约 **35 个 crate** 引用（含全部 `prism_audio_*`、`prism_render_material:14`、`prism_render_scene:16`、`prism_render_shading:11`、`prism_render_visibility:19`、`prism_physics_gpu:12`）。`pkg/prism_math/Cargo.toml:23` 注释自称 "M6 migration shim: glam/bevy_math-shaped type aliases"，`pkg/prism_utils`/`prism_time`/`prism_tasks`/`prism_transform`/`prism_reflect`/`prism_math` 提供 `compat-bevy` feature（如 `pkg/prism_math/Cargo.toml:25` `compat-bevy = []`）。这与 `docs/prism_engine_component_gap_zh.md:163` 的"M4：… ~50 个 `bevy_math` 用户批量换 `prism_math`"尚未完成一致。

**§5 与文档声明的矛盾**

| 文档声明 | 位置 | 实测 | 判定 |
|---|---|---|---|
| "**不依赖任何 `bevy_*` crate**"（针对 `prism_asset`） | `docs/prism_asset_design_zh.md:20`（另 `:938` "**不碰任何 `bevy_*`。**"） | `prism_asset` 确无 bevy 依赖 | **相符** |
| §17 "Bevy 集成层 API（Layer 4）" | `docs/prism_physics_design_zh.md:316` | 无对应集成 crate/Plugin；文档 `:406-413` 的 crate 代码块给出 `bevy_prism_core` 等 7 个命名，与 `prism_physics_*` 不符 | **文档与代码命名矛盾** |
| 物理内核"无 Bevy 依赖"（Layer 0-2 隐含；§21 代码块注释 `bevy_prism_core // Layer 0-2: 无 Bevy 依赖的纯物理内核`） | `docs/prism_physics_design_zh.md:406-413`（注释在 `:407`） | `prism_physics_core`/`prism_physics_geometry` 确无 bevy；**但同属物理体系的 `prism_physics_gpu` 依赖上游 `bevy_math` 并传递引入 `bevy_reflect`** | **部分矛盾** |
| gap 文档状态表："资产内核 `prism_asset` ❌" | `docs/prism_engine_component_gap_zh.md:98` | `pkg/prism_asset` 存在（29 个 src 模块，M0/M1 已交付，被 `prism_image`/`prism_scene`/`prism_render_app` 生产依赖） | **文档严重过期/自相矛盾**：同文档 `:25` 又把它列为 P1 组件，`:160`「M1 反射与资产」把它作为推进项；`:163` 的 M4 亦已部分完成 |
| gap 文档："`prism_bevy` \| app/camera/ecs/shape/transform"、":57 `prism_audio_bevy` \| app/ecs/transform" | `docs/prism_engine_component_gap_zh.md:56-57` | `pkg/prism_bevy` 为空目录；`prism_audio_bevy` 依赖仅 `prism_audio_ecs`，bevy 相关 crate 全在 dev-deps | **文档描述的依赖关系已不成立** |
| refactor plan §2："原下游集成 crate `pkg/prism_bevy` 已完成迁移审计并退役" | `docs/prism_bevy_refactor_plan_zh.md:30,184,241-242` | `pkg/prism_bevy` 为空目录、无 Cargo.toml | **相符**（退役已执行；空目录为残留） |

**Bevy 耦合裁决**：分配范围内 **8 个存在 crate 中 7 个完全解耦**（`prism_physics_core`、`prism_physics_geometry`、`prism_asset`、`prism_asset_build_cli`、`prism_asset_build_core`、`prism_asset_container`、`prism_audio_bevy`），**唯一耦合点是 `pkg/prism_physics_gpu/Cargo.toml:12-14` 对上游 Bevy crate `crates/bevy_math` 的 path 依赖**，其使用面仅限 `bevy_math::ops` 的确定性超越函数（11 个文件）。`prism_bevy` 已按文档退役，仅剩空目录。

---

#### 文档 ↔ crate 对应关系缺口

**A. 规定了但代码中不存在的 crate（文档 prescribe、pkg/ 无目录）**

| 规定它的文档 | 位置 | 规定 crate | 实测 |
|---|---|---|---|
| prism_asset_build_design_zh.md §3.1 | `:206-227`（代码块 `:206-228`） | `prism_asset_registry` | `pkg/prism_asset_registry` 不存在（目录缺失） |
| 同上 | `:209` | `prism_asset_graph` | 不存在 |
| 同上 | `:210` | `prism_asset_processor` | 不存在 |
| 同上 | `:211` | `prism_asset_build` | 不存在 |
| 同上 | `:212` | `prism_asset_cas` | 不存在 |
| 同上 | `:213` | `prism_asset_ddc` | 不存在 |
| 同上 | `:214` | `prism_asset_profile` | 不存在 |
| 同上 | `:215` | `prism_asset_validation` | 不存在 |
| 同上 | `:216` | `prism_asset_package` | 不存在 |
| 同上 | `:218` | `prism_asset_build_editor` | 不存在 |
| 同上 | `:220` | `prism_import_image` | 不存在 |
| 同上 | `:221` | `prism_import_gltf` | 不存在 |
| 同上 | `:222` | `prism_import_usd` | 不存在 |
| 同上 | `:223` | `prism_bake_texture` | 不存在 |
| 同上 | `:224` | `prism_bake_mesh` | 不存在 |
| 同上 | `:225` | `prism_bake_animation` | 不存在 |
| 同上 | `:226` | `prism_bake_audio` | 不存在 |
| 同上 | `:227` | `prism_bake_shader` | 不存在 |
| prism_asset_design_zh.md §19 | `:928` | `prism_asset_import` | 不存在 |
| prism_asset_design_zh.md §19 | `:935` | `prism_vfs` | 不存在（其职责被 `prism_asset_container` 承担） |
| prism_derived_data_contract_zh.md §2.1 | `:47-58` | `prism_bake_texture`/`prism_bake_mesh`/`prism_bake_shader`/`prism_bake_animation`/`prism_bake_audio`/`prism_micromap_gpu` | `ls -d pkg/prism_bake_*` → 无匹配；`prism_micromap_gpu` 未核实 |
| prism_bevy_refactor_plan_zh.md §6 | `:170` | `prism_bevy`(渲染集成)、`prism_bevy_ui`、`prism_bevy_audio`、`prism_bevy_physics` | 4 个均不存在（`prism_bevy` 仅为空目录） |
| prism_physics_design_zh.md §21 | `:406-413` | `bevy_prism_core`/`bevy_prism_geometry`/`bevy_prism_backend_cpu`/`bevy_prism_backend_gpu`/`bevy_prism_cache` | 均不存在；实际为 `prism_physics_core`/`prism_physics_geometry`/`prism_physics_gpu`（命名矛盾而非缺失） |

> 结论：`prism_asset_build_design_zh.md` §3.1 的"建议 crate 划分"是**建议（"建议"二字为文档原文）**，`prism_asset_build_core` 单 crate 已把其中 graph/cas/ddc/validation/package/processor 职责内聚实现（文件 `graph.rs`/`cas.rs`/`ddc.rs`/`validation.rs`/`package.rs`/`import/mod.rs`）——因此这些"缺失 crate"应理解为**分层未落地**，而非功能全缺。**但 `prism_bake_*` 与 `prism_import_*` 是功能缺失**（无任何等价实现位置）。

**B. 有 crate 但无专门设计文档指定它**

| crate | 文档覆盖情况 | 判定 |
|---|---|---|
| `prism_physics_geometry` | 仅在 `prism_physics_design_zh.md` §21（`:404-413`）作为代码块一行出现，及 §13 能力描述；无独立设计文档 | **无专门文档** |
| `prism_physics_gpu` | 同上（§7/§21）；另有 `prism_aaa_advanced_features_zh.md`、`prism_ui_loom_design_zh.md`、`prism_material_pipeline_design_zh.md` 提及但非规格主体 | **无专门文档** |
| `prism_asset_container` | 无独立文档；规格散落在 `prism_asset_design_zh.md` §10/§11/§19（`:928-935`）与 `prism_asset_build_design_zh.md` §14/§15 | **无专门文档（规格分散）** |
| `prism_asset_build_cli` | 无独立文档；规格散落在 `prism_asset_build_design_zh.md:217`（§3.1）与 `:55`（验收矩阵 CLI 行） | **无专门文档（规格分散）** |
| `prism_bevy` | 有 `docs/prism_bevy_refactor_plan_zh.md` 多节规定其退役与替代架构；另 `prism_app_design_zh.md`/`prism_editor_framework_design_zh.md`/`prism_editor_remote_design_zh.md`/`prism_engine_component_gap_zh.md` 提及 | 有文档，但**代码不存在**（反向缺口） |
| `prism_audio_bevy` | `prism_engine_component_gap_zh.md:57`、`prism_ecs_design_zh.md`、`prism_ecs_completion_zh.md`、`prism_audio_engine_design_zh.md` | 有文档 |

---

#### 关键结论

1. **`pkg/prism_bevy` 经实测为空目录**（`ls -la pkg/prism_bevy` 仅有空 `src/`、`tests/`，无 `Cargo.toml`、无 `.rs`），故 `pkg/` 实际 crate 数为 **140**（141 个目录 − 1 空目录）。**但它不是"文档描述了不存在的 crate"那么简单**：`docs/prism_bevy_refactor_plan_zh.md:30,184,241-242` 明确记载旧 `prism_bevy` 桥接插件"✅ 完成并退役"，空目录是**退役后的残留物**——文档与代码在此**一致**，真正的问题是同一文档 §6（`:170`）规定的替代集成 crate `prism_bevy`(渲染)/`prism_bevy_ui`/`prism_bevy_audio`/`prism_bevy_physics` **四个全部不存在**（`ls -d` 均报 No such file or directory），即"退役了旧的、没建新的"。

2. **Bevy 耦合裁决：8 个存在 crate 中 7 个完全解耦，唯一耦合点是 `pkg/prism_physics_gpu/Cargo.toml:12-14` 对上游原版 Bevy crate `crates/bevy_math`（`version = "0.20.0-dev"`, `repository = github.com/bevyengine/bevy`）的 path 依赖，且会传递引入 `bevy_reflect`**。使用面仅 `bevy_math::ops` 的确定性 `f32` 超越函数（11 个文件，如 `pkg/prism_physics_gpu/src/rigid/joint/d6_drive_cpu.rs:106`），用于 CPU/GPU 位等对齐——因此这是"为了确定性借用 Bevy 数学库"，而非渲染栈耦合，但**它确实使 `prism_physics_gpu` 违反"pkg 不依赖 Bevy"的架构约束**，且替换成本极低（换成 `libm` 或 `prism_math` 的 ops 即可）。仓库级还有约 35 个 `pkg/` crate 依赖同一 path，与 `docs/prism_engine_component_gap_zh.md:163` 的"M4 批量换 `prism_math`"未完成相符。

3. **`prism_physics_gpu` 的最大缺口是全无接通（接通% = 0/5）**：368 个 `.rs`、21 个顶层内核模块（broadphase/bvh/fluid/mpm/xpbd/cloth/fracture…）齐备，但 grep `prism_physics_gpu` 于全部 `Cargo.toml` **只命中自身**（`pkg/prism_physics_gpu/Cargo.toml:2` 与自 dev-dep `:20`），无任何生产依赖者；而 `prism_physics_core` 的接通也只靠单一窄桥 `pkg/prism_render_architecture/src/cloth/*`（`wind.rs:159,200`、`collision.rs:140,158,176`），且 `wind.rs:262` 一处仍标注"reserved for the wind solver"。物理体系整体呈"内核宽、接通极窄"形态。

4. **`prism_physics_core` 的物理设计文档存在严重"文档↔代码命名矛盾"**：`docs/prism_physics_design_zh.md:406-413` 的 §21 crate 代码块规定 **7 个 `bevy_` 前缀** crate（`bevy_prism_core`/`bevy_prism_geometry`/`bevy_prism_backend_cpu`/`bevy_prism_backend_gpu`/`bevy_prism_cache`/`bevy_prism`/`bevy_prism_debug`），与实际 `prism_physics_core`/`prism_physics_geometry`/`prism_physics_gpu` 完全不符，且 §17（`:316`）"Bevy 集成层 API（Layer 4）"在代码中无对应集成 crate/Plugin。同时 M10（动态装配，`§24`）、M13（布偶/锥限，`§26`）、M14（降阶坐标 ABA，`§27`）、M15（拓扑撕裂，`§28.3–28.11`）四个里程碑 grep 全部零命中——`RagdollBuilder`/`Articulation`/`reduced`/`featherstone`/`snap_align`/`SnapPose` 均不存在，占 15 个声明单元中的 5 个未落地。

5. **`prism_asset` 的最大单体缺口是 §26 的 `#[derive(Asset)]` 派生宏完全缺失（未落地），并连带压垮 prelude 与编辑器暴露**：全工作区无 `prism_asset_macros` crate，`#[derive(Asset)]` 仅以文档注释文本存在；同时 `pkg/prism_asset/Cargo.toml` 实测**不依赖 `prism_reflect`**，而 `docs/prism_asset_design_zh.md:938` 明确把 `prism_reflect`（settings/meta）列为依赖、`:1100` 要求"经 `prism_reflect` 把句柄/加载态/依赖关系暴露给编辑器与脚本层"——依赖缺失使该单元无法成立。同类高价值缺口还有：§23.3 占位/兜底资产（grep `placeholder` 零命中，`src/loader.rs:30` 注释显式声明本 crate 不做兜底）、§23.8 / §9.2 的 `LoadPriority`（`scheduler.rs` 无 `Immediate`/`Prefetch`）、§25（`docs/prism_asset_design_zh.md:1050,1051,1053`）要求的 `loom` 并发穷举与 fuzz 常驻语料库（`prism_asset` 无 `loom`/`proptest` 依赖，工作区无 `fuzz/` 目录）。

6. **`prism_asset_build_core` 是全部分配 crate 中落地最扎实、且文档最诚实的**（骨架% 89%、接通% 58%）：其验收矩阵 15 行中 8 行"已实现"、6 行"部分实现"、1 行"未实现"（`docs/prism_asset_build_design_zh.md:42-56`），且本次抽验的每一项都能落到具体 file:line——包括"已实现"行（如 `src/validation.rs` 字节预算、`src/graph.rs` wave 并行、`src/bake.rs` BakeService）与协调切片的 `flock`（`src/transaction.rs:141`）、非 Unix fail-closed（`src/ddc.rs:32,375`、`src/transaction.rs:2267`）。其最大缺口是**规格化 crate 全部缺位的 `prism_bake_*` 烘焙内核族**（`pkg/prism_bake_*` 无匹配）与**GPU 直通/远程 Build Farm/完整 Cook**（文档自标"未实现"）。另注：`prism_asset_build_cli` 接通% 为 0 仅因其为调用链顶端的二进制入口（`src/main.rs`，`Command` 枚举 `src/args.rs:15-23` 含 Plan/Import/Build/Package/Patch/Validate/Inspect 共 7 个，缺文档 §3.1 `:217` 规定的 `cook`/`diff`）。

---

## 3. 全局归纳

### 3.1 子系统完成度总表

下表右三列中，**「有消费者/零消费者」是本报告唯一的客观机器判定列**：遍历仓库内全部 `Cargo.toml`（排除 `target/`、`.git/`、`crates/`），把 `path = "…"` 依赖解析到 `pkg/<crate>`，同 crate 自引不计；它统计的是「全仓是否还有别的 crate 依赖它」，**已把 dev-dependency 计为消费者**，故严格「仅非测试生产消费者」口径下的零接通数只会 ≥ 此列。

| § | 子系统 | crate | 有消费者 | 零消费者 | 零率 | 骨架/类型层% | 端到端接通% | 判定 |
|---|---|---:|---:|---:|---:|---|---|---|
| 2.1 | ECS 与核心运行时 | 18 | 14 | 4 | 22% | 70–95% | 0–90%（多数 60–85%） | 骨架基本齐备；4 个 GPU/ECS twin 全孤岛 |
| 2.2 | 渲染核心（帧图/RHI/架构） | 4 | 4 | 0 | 0% | 65–100% | 32–83% | 唯一整组都被消费的子系统 |
| 2.3 | 渲染应用层 | 7 | 3 | 4 | 57% | 0–100% | 0–71% | 应用层本体缺外部消费者 |
| 2.4 | GPU 几何・特性・派生 | 16 | 5 | 11 | 69% | 多数 100% | 多数 0% | **设计使然**（真机 parity 孪生），但 13 个无任何 Cargo 依赖者 |
| 2.5 | 材质 PCM / Schema | 2 | 2 | 0 | 0% | 78–79% | 0–10% | 唯一真接通的生产链是 `digest_for ← header_with_execution ← material/buffers.rs` |
| 2.6 | UI 核心 | 18 | 12 | 6 | 33% | 约 50–100% | 断链为主 | 6 个零消费者；文本栈与渲染后端断链 |
| 2.7 | UI 高级层 | 16 | 7 | 9 | 56% | 95.4%（87 单元） | 27.6% | 12/16 无生产调用方 |
| 2.8 | 音频 part 1 | 16 | 7 | 9 | 56% | 88%（138 单元） | 10%（14/138） | 唯一生产链是 bevy→ecs→rt→core |
| 2.9 | 音频 part 2 | 15 | 5 | 9 | 60% | 100% | ≈0 | 全为类型层；`prism_audio_bevy` 因分组重复计入 2.12 |
| 2.10 | 平台・窗口・输入・相机 | 9 | 6 | 3 | 33% | 29–100% | 0–33% | winit 后端把键盘/滚轮/鼠标/IME 输入全丢弃 |
| 2.11 | 世界系统 | 11 | 10 | 1 | 9% | 50–100% | 38–100% | 接通率最高；terrain 仅 38% |
| 2.12 | 物理・资产・Bevy 桥 | 8 | 5 | 3 | 38% | 45–89% | 0–58% | 文档最诚实的一组（build_core 89%/58%） |
| — | **合计** | **140** | **81** | **59** | **42%** | — | — | 见 §3.2 |

> 口径警告：各节的骨架%/接通% **分母由各自子审计从文档抽出的「具名要求单元数」定义，彼此不可比**（同一 crate 在不同分节口径下分母可能不同，例如 `prism_math_gpu` 在 2.1 以 46 个文件计、`prism_audio_bevy` 同时在 2.9 与 2.12 出现）。因此本表**不做跨子系统的百分比加总**，跨子系统比较只使用「有消费者/零消费者」这一客观列。表中百分比仅表示各节自述区间。

### 3.2 五条全局结论

1. **仓库是两套互不消费的平行世界。** 仓库根 `Cargo.toml:1-14` 的 `[package]` 仍是上游 Bevy 老栈门面（`name = "bevy"`、`version = "0.20.0-dev"`、`repository github.com/bevyengine/bevy`）；根依赖表对 `pkg/` 的引用数为 **0**（`grep -c 'path = "pkg/' Cargo.toml` = 0），根 Cargo.toml 中出现的 `pkg/prism_*` 全部来自 `[workspace] members` 列举（如 `:27`、`:50`、`:56`、`:78`）。`pkg/` 之外**仅 4 个 manifest** 消费 pkg crate——`benches/Cargo.toml`、`os-backends/prism_render_driver_wgpu/Cargo.toml`、`os-backends/prism_render_driver_null/Cargo.toml`、`tools/render_triangle/Cargo.toml`——合计只拉到 **8 个**不同 crate（prism_render_driver 最多）。**没有任何应用把 pkg/ 装配成可运行产品。**

2. **42%（59/140）的 crate 在全仓没有任何消费者。** 明细见附表 D；重灾子系统是 GPU 特性层（11/16）、音频 part 2（9/15）、UI 高级层（9/16）、音频 part 1（9/16）。这些 crate 的「骨架% 高」并不等于进度，它们连编译期依赖边都没有。

3. **`pkg/*_gpu` 的「高骨架 / 零接通」多数是设计使然，不应记为缺陷。** 本仓 GPU 层的架构是「真机 wgpu parity 孪生」：设备无关的黄金实现在 `pkg/prism_render_architecture/`，生产 GPU 路径在 `pkg/prism_render_scene/src/shaders/*.wesl` + dispatch 接线。因此判断这些 crate 的进度**必须看其 golden 宿主是否被生产消费**，而不是看孪生 crate 自身的依赖者数。此判据同样适用于 ECS 侧的三套 GPU twin（`prism_transform_gpu` / `prism_math_gpu` / `prism_state_tree_gpu`）。

4. **文档体系本身是分层的：三份台账（`docs/prism_ecs_completion_zh.md`、`prism_asset_build_core` 的验收矩阵、`prism_scene`/`prism_state_tree` 的 IMPLEMENTED/DESIGNED/VERIFIED 三分标签）口径诚实且可逐条回落到 `file:line`；而面向特性的大文档（`prism_aaa_advanced_features_zh.md`、`prism_audio_engine_design_zh.md`、`prism_material_pipeline_refactor_*`、`prism_bevy_refactor_plan_zh.md`）存在大量过期快照、错报消费点与「✅ 而已」。** 后者是本报告 §3.3/§3.4 反例的主要来源。

5. **最普遍的单点失败形态不是「未实现」，而是「零消费者孤岛」与「私有/未启用入口」。** 典型三例：`prism_ui_macro` 的 `hoist`/`dualmode` 因 `lib.rs:60-61` 私有而在 crate 外不可达；`prism_ui_render_backend` 的 wgpu/Metal 后端（`gpu.rs`，820 行）全仓无任何 `Cargo.toml` 启用 `gpu` feature；`prism_render_material::install_execution_tables` 全仓唯一命中是其定义处，导致 `prism_render_scene/src/shading/resources.rs:434` 的 `from_execution_tables` 分支不可达、生产恒走内置表。

### 3.3 反例汇总（文档标 ✅／「已验收」而代码查无实现）

| # | 文档声明 | 实测证据 | 性质 |
|---|---|---|---|
| 1 | `prism_material_pipeline_refactor_verification_zh.md:102` P2-j 已注册化，证据 `classification.rs:453 from_registry`、`:509 class_count()`、`:1206` 穷举门禁 | `pkg/prism_render_shading/src/classification.rs` 实测 **709 行**；`from_registry`/`class_count`/`UnhandledResponseMode` 全仓 `.rs` **0 命中** | 整条证据链不存在 |
| 2 | 同文档 `:90` `MATERIAL_ABI_VERSION = 6`（`record.rs:18`） | `pkg/prism_render_material/src/lib.rs:112` 是 `SURFACE_ABI_VERSION` 别名；`surface_layout.rs:13` **= 8**；`schema/surface.toml:18` 亦为 8 | 数值失真 |
| 3 | 同文档 `:94` `pipeline_cache.rs:289 MAX_IN_FLIGHT_COMPILES = 4` | 该常量全仓 **0 命中**；等价物是 `pkg/prism_render_app/src/pipeline_cache.rs:403-415` 的 `PipelineCompileConfig { worker_count: 4, queue_capacity: 64 }` | 符号与位置均错 |
| 4 | `docs/prism_platform_design_zh.md:334`「M0–M6 核心路线图已全部落地并通过验证」 | `pkg/prism_platform/src/crash/` 无 `minidump.rs`（§20 布局明确要求）；`src/lib.rs:67-68` 自述 minidump 与跨进程处理仍为 future work；`crash::install`（`crash/mod.rs:234`）零生产调用者 | 里程碑虚报 |
| 5 | 同文档 §4.1/§4.2/§27 的 12+ 模块、9 个 feature、15 项性能门槛 | `pkg/prism_window_winit/Cargo.toml` **无 `[features]` 段**；12+ 计划模块文件零存在；无 `benches/` | 计划被写成已完成 |
| 6 | 窗口 M0 验收「keyboard + mouse button/motion/wheel basic translation」 | `pkg/prism_window_winit/src/translate.rs:71-75,99-103` 把 `KeyboardInput`/`ModifiersChanged`/`Ime`/`MouseWheel`/`MouseInput` **全部丢弃**，只翻译 `CursorMoved`/`CursorEntered`/`CursorLeft`（`:57-64`） | 验收行不成立 |
| 7 | `docs/prism_audio_engine_design_zh.md` §46.1/§46.2/§28/§22/§53.7/§46.3 | `UPOLS`/`NUPOLS`/`DspKernel`/SIMD 内核/rt-alloc-guard/`DriftResampler`/device 侧 FTZ·DAZ 在 `pkg/` 全部 **零命中** | 多处 ✅ 无代码 |
| 8 | `prism_aaa_advanced_features_zh.md` 隐含 micromap/DMM 实现度与 `docs/prism_material_pipeline_refactor_zh.md` §5.4/§5.4.1/§5.4.2/§5.5 | PCM 的 TOML manifest 加载、`proc-macro → WESL` 瓣宏、编辑器插件与 wasm 沙箱、`cost_hint` 进 CI 全部未落地；`prism_material_pcm/Cargo.toml` 依赖为空 | 机制整体缺失 |
| 9 | `docs/prism_material_pipeline_design_zh.md` §3.6 称 PCM 被 `classification_gpu.rs:42` 与 `prism_render_shading/src/classification.rs` 消费 | 前者位于 `#[cfg(test)]` 区（`:38`）之后；`pkg/prism_render_shading/Cargo.toml` **不依赖** `prism_material_pcm`；真实触点在 `prism_render_material/src/{record.rs:810, shading_plan.rs:666,1272}` | 错报消费点 |
| 10 | `docs/prism_asset_design_zh.md:938/:1100` 要求 `prism_reflect` 集成 + §26 `#[derive(Asset)]` 派生宏 | `pkg/prism_asset/Cargo.toml` 对 `prism_reflect` **0 命中**；无 `prism_asset_macros` crate，`#[derive(Asset)]` 仅存在于文档注释 | 依赖与宏双缺 |
| 11 | `docs/prism_physics_design_zh.md` §24/§26/§27/§28.3–28.11（M10/M13/M14/M15） | `RagdollBuilder`/`featherstone`/`snap_align`/`SnapPose`/`reduced_coordinates` 在 `pkg/` 下 `.rs` **全仓 0 命中**（`Articulation` 的命中实为声学 `ArticulationLoss`，无关） | 4 个里程碑未落地 |
| 12 | `docs/prism_bevy_refactor_plan_zh.md:170` 规定的替代集成 crate | `prism_bevy`(渲染)/`prism_bevy_ui`/`prism_bevy_audio`/`prism_bevy_physics` **四个全部不存在**（旧桥接确已按 `:30,184,241-242` 退役，`pkg/prism_bevy/` 只剩空 `src/`、`tests/`） | 「退役旧的、没建新的」 |
| 13 | `docs/prism_loom_runtime_framework_design_zh.md` 的 10 个 `prism_runtime_*` | `pkg/` 与根 `Cargo.toml` 中**完全不存在** | 整套设计无代码 |
| 14 | `docs/prism_asset_build_design_zh.md:208-227`、`prism_derived_data_contract_zh.md:47-58` | `prism_asset_registry/graph/processor/build/cas/ddc/profile/validation/package/build_editor`、`prism_import_*`、`prism_bake_*`（含 `prism_bake_audio`）全部无目录 | 功能层缺失 |
| 15 | `docs/prism_ui_component_kit_design_zh.md` §16 的 node_editor/editor/media/qr | `pkg/prism_ui_component_kit/Cargo.toml:20-30` 无这些 feature gate | 设计未落 feature |
| 16 | `prism_transform_gpu` 文档 §24.7「5 项真机 parity 全绿」 | `pkg/prism_transform_gpu/tests/hierarchy_parity.rs:28` 的 `with_gpu()` 无 adapter 时仅 `eprintln!("skipping: no usable GPU adapter on this host")` 后返回 `None`，**测试照常通过** ⇒ 「全绿」与「全跳过」不可区分 | 声称无法与事实区分 |

### 3.4 反向失真（文档低估代码）

| # | 文档声明 | 实测 | 影响 |
|---|---|---|---|
| 1 | `prism_aaa_advanced_features_zh.md:12/:622/:734/:812` 四处称 micromap / DMM / gdeflate / virtual_texture **当前 0 文件**；`:492` 辐射级联标 ⬜ 待建 | `prism_micromap` 9 文件、`prism_micromap_gpu` 4、`prism_dmm` 8、`prism_dmm_gpu` 4、`prism_gdeflate_gpu` 9、`prism_virtual_texture_gpu` 12、`prism_render_architecture/src/lighting/radiance_cascades/` 6 + RC GPU 5 | **会误导排期**：把已存在的孪生读成待建 |
| 2 | `docs/prism_engine_component_gap_zh.md:98` 标「资产内核 prism_asset ❌」 | `pkg/prism_asset` 存在、M0/M1 已交付，且被 `prism_image`/`prism_scene`/`prism_render_app` 生产依赖 | 该文档整体已过时（其称 pkg 仅 59 crate） |
| 3 | `prism_material_pipeline_design_zh.md:123` 记 schema crate 2431 行、`:269` 记 `schema_version=1`/`abi_version=6` | 实测 `src`+`src/bin` **3529 行**；`schema/surface.toml:13,18` = **2 / 8** | 三份文档 ABI 版本 6/7/8 互不一致 |
| 4 | `prism_audio_codec` 在主音频文档中 **0 次字面提及**、§39 表不列该行 | 该 crate 存在且已接 Symphonia（`Cargo.toml:18-27`），FLAC/Vorbis/MP3/AAC 均实现（`decoder.rs:206`、`:307-315`），Opus 明确不支持；`src/lib.rs:5` 自述实现 §20/§44.1 | 孤儿 crate：实现存在但无文档归属，也无消费者 |
| 5 | `docs/prism_platform_os` 无独立设计文档，规格散在 4 份文档，`:383-385` 仍写 PLANNED | 代码已有 Linux 后端：`pkg/prism_platform_os/src/topology/mod.rs:44,93,158` 的 `probe_numa_nodes`/`bind_current_thread_to_node`/`probe_topology`，且被 `pkg/prism_tasks/src/numa.rs:105` 生产调用 | 文档滞后于代码 |
| 6 | 音频文档记测试数：`prism_audio_authoring` 83、`prism_audio_object` 72；`docs/prism_ecs_completion_zh.md:17` 记 audio_ecs 30 | 实测 120、101；audio_ecs 实测 **37**（主文档 `design:163` 亦记 37） | 文档内部自相矛盾 |
| 7 | `docs/prism_ecs_completion_zh.md` §24.1 把「App 接入 time 帧预算」列为待补证据 | 已在 `pkg/prism_app/src/budget.rs:15` 落地 | 台账保守 |
| 8 | `pkg/prism_audio_hrtf` 文档未列 SOFA 二进制解码 | `pkg/prism_audio_hrtf/src/sofa/decode.rs:131` 已实现 | 低估 |

### 3.5 命名与归属错位（按文件名机械匹配文档会严重误判）

- **文档名多为机制名而非 crate 名**：`prism_gi_lumen_design_zh.md` → `prism_render_scene`/`prism_render_architecture`，`prism_ui_loom_design_zh.md` → `prism_ui*` 全家。按文件名匹配会得出完全错误的缺口。
- **6 个 crate 在 `docs/` 中零字面命中**：`prism_ray_reorder_gpu`、`prism_radiance_cascades_gpu`、`prism_motion_gpu`、`prism_dmm_gpu`、`prism_dmm`、`prism_audio_codec`。其中 `prism_water_render` 虽有水体文档但**两份实现文档的全部落点都在 `prism_render_scene`，零处提及该 crate**。
- **文档规定的 crate 名与代码不符**：`prism_physics_design_zh.md:406-413` 规定 7 个 `bevy_prism_*` 前缀 crate，实际是 `prism_physics_core/geometry/gpu`；`docs/prism_bevy_refactor_plan_zh.md:166` 写 `prism_bevy_audio`、`:58` 把 `bevy_audio` 映射到 `prism_audio_core`+`device`，与实际的 `prism_audio_bevy` 不一致。
- **文档引用了不存在的 crate**：`docs/prism_derived_data_contract_zh.md:52` 引用 `prism_bake_audio`；`prism_asset_design_zh.md:928,935` 引用 `prism_asset_import`/`prism_vfs`。
- **文档–crate 覆盖不对称**：`prism_platform_os` 无独立设计文档；反向地，`docs/prism_rhi_port_surface_zh.md`、`water_render_fabric_implementation_zh.md`、`water_hybrid_implementation_verification_zh.md` 等文档在受审 crate 中没有对应实现落点。

### 3.6 不可宣称清单（本轮未验证）

依 §0 口径，以下事项本报告**一律不作证据**，仅记录文档自述：

1. **任何测试/门禁的「通过/全绿」**：本轮未执行 `cargo build`/`cargo test`/`cargo bench`，也未运行真机 GPU、音频设备、NUMA 或多机网络。
2. **真机 GPU 结论**：`prism_transform_gpu` 的 parity 门禁在无 adapter 时静默跳过（§3.3 第 16 条）；volumetric 与 water-hybrid 文档自述本轮未取得可用 adapter；有明确真机数字的仅虚拟纹理（18 passed，Apple M2）与虚拟几何（1 项 cluster_vertex_pull_parity）。
3. **性能数字**：`pkg/prism_ecs/tests/scale.rs:13` 明示不设墙钟阈值；各文档的微秒/近线性曲线为 design target 或历史记录。
4. **§16.2 活数据布局迁移、§15 GPU 常驻列、§17 NUMA 收益、§21 可执行性能回归**：源码中无对应入口，未核实。
5. **跨进程/跨机/公网/NAT·ICE·relay 与音频真实 device callback**：未验证。
6. **`crates/*` 反向依赖环**、fuzz/loom 效果、崩溃恢复细节：未核实。

### 3.7 本报告自身的复核（详见附录 C）

各分节的结论由分头静态审计得出，其中**最关键或最反直觉的 30 余条由本报告统一复验**（附录 C）。复核中修正了子审计的两处计数错误（Bevy 耦合 30/27/3 而非「约 35」或「28/2」）并排除了两处误判（`Articulation` 命中实为声学 `ArticulationLoss`；`pkg/prism_bevy` 的 `bevy` 文本命中来自 `description` 字符串）。凡未能复验者，正文已标「未核实」。

---

### 附录 A：pkg/ 各 crate 规模、测试数与文档引用数

| crate | .rs 文件 | 行数 | #[test] | 被文档引用次数 | bevy 依赖 |
|---|---:|---:|---:|---:|---|
| `prism_volumetric_gpu` | 1277 | 572,597 | 4,839 | 6 | — |
| `prism_render_architecture` | 861 | 415,765 | 11,372 | 29 | — |
| `prism_render_scene` | 653 | 170,057 | 1,894 | 39 | bevy_app,bevy_asset,bevy_camera,bevy_color,bevy_core_pipeline,bevy_derive,bevy_ecs,bevy_image,bevy_light,bevy_log,bevy_material,bevy_math,bevy_mesh,bevy_pbr,bevy_platform,bevy_render,bevy_shader,bevy_tasks,bevy_transform |
| `prism_render_shading` | 376 | 134,862 | 3,424 | 16 | bevy_math |
| `prism_physics_core` | 431 | 133,017 | 2,661 | 11 | — |
| `prism_physics_gpu` | 489 | 127,669 | 1,483 | 6 | bevy_math |
| `prism_audio_core` | 149 | 96,236 | 2,267 | 3 | bevy_math |
| `prism_ecs` | 215 | 91,883 | 1,374 | 29 | — |
| `prism_hair_gpu` | 176 | 51,516 | 602 | 5 | — |
| `prism_render_material` | 140 | 48,584 | 950 | 20 | bevy_asset,bevy_color,bevy_image,bevy_material,bevy_math,bevy_pbr |
| `prism_app` | 67 | 35,392 | 497 | 28 | — |
| `prism_render_app` | 83 | 34,413 | 274 | 22 | — |
| `prism_ui_component_kit` | 172 | 31,816 | 595 | 1 | — |
| `prism_audio_spatial` | 60 | 30,028 | 778 | 3 | bevy_math |
| `prism_world_gpu` | 49 | 28,711 | 215 | 22 | — |
| `prism_math` | 86 | 25,736 | 267 | 27 | — |
| `prism_replication` | 37 | 24,572 | 256 | 12 | — |
| `prism_asset_build_core` | 45 | 22,579 | 211 | 5 | — |
| `prism_tasks` | 70 | 21,286 | 318 | 34 | — |
| `prism_math_gpu` | 89 | 20,826 | 215 | 7 | — |
| `prism_platform` | 80 | 18,618 | 178 | 17 | — |
| `prism_world_terrain` | 32 | 18,589 | 197 | 11 | — |
| `prism_asset` | 36 | 18,534 | 235 | 33 | — |
| `prism_reflect` | 74 | 18,135 | 208 | 19 | — |
| `prism_diagnostic` | 80 | 17,973 | 257 | 29 | — |
| `prism_state_tree_core` | 24 | 16,783 | 125 | 1 | — |
| `prism_world_runtime` | 34 | 15,799 | 174 | 12 | — |
| `prism_utils` | 65 | 15,062 | 209 | 9 | — |
| `prism_ui_dock` | 32 | 14,889 | 156 | 1 | — |
| `prism_transform` | 46 | 14,451 | 256 | 14 | — |
| `prism_ui_render_backend` | 43 | 13,887 | 310 | 8 | — |
| `prism_virtual_geometry_gpu` | 52 | 13,051 | 102 | 8 | — |
| `prism_scene` | 30 | 12,049 | 158 | 14 | — |
| `prism_world_residency` | 17 | 11,147 | 125 | 10 | — |
| `prism_time` | 51 | 10,862 | 222 | 20 | — |
| `prism_render_material_gpu` | 8 | 10,222 | 74 | 5 | — |
| `prism_render_graph` | 31 | 9,602 | 78 | 12 | — |
| `prism_physics_geometry` | 34 | 9,148 | 226 | 5 | — |
| `prism_audio_geometry` | 20 | 8,253 | 147 | 1 | bevy_math |
| `prism_asset_container` | 13 | 7,729 | 75 | 3 | — |
| `prism_render_driver` | 30 | 7,538 | 88 | 21 | — |
| `prism_audio_authoring` | 47 | 7,239 | 134 | 1 | bevy_math |
| `prism_audio_hrtf` | 14 | 7,218 | 139 | 3 | bevy_math |
| `prism_render_visibility` | 27 | 6,682 | 131 | 10 | bevy_camera,bevy_math,bevy_shape,bevy_transform |
| `prism_world_io` | 11 | 6,440 | 78 | 4 | — |
| `prism_audio_immersive` | 22 | 6,403 | 93 | 1 | bevy_math |
| `prism_audio_conditioning` | 22 | 6,354 | 132 | 1 | bevy_math |
| `prism_audio_procedural` | 30 | 5,884 | 113 | 1 | bevy_math |
| `prism_audio_governor` | 21 | 5,869 | 172 | 1 | bevy_math |
| `prism_ui_layout` | 20 | 5,808 | 94 | 2 | — |
| `prism_audio_assets` | 26 | 5,463 | 42 | 1 | bevy_math |
| `prism_state_tree_ecs` | 15 | 5,222 | 47 | 1 | — |
| `prism_audio_ecs` | 20 | 5,138 | 61 | 3 | — |
| `prism_mesh` | 26 | 5,012 | 48 | 4 | — |
| `prism_audio_wave` | 20 | 4,985 | 85 | 2 | bevy_math |
| `prism_camera` | 14 | 4,521 | 40 | 4 | — |
| `prism_render_runtime` | 8 | 4,500 | 44 | 7 | — |
| `prism_ui_text` | 19 | 4,425 | 89 | 5 | — |
| `prism_audio_comm` | 17 | 4,424 | 67 | 1 | bevy_math |
| `prism_audio_object` | 23 | 4,348 | 110 | 1 | bevy_math |
| `prism_audio_midi` | 17 | 4,241 | 72 | 1 | bevy_math |
| `prism_audio_content` | 14 | 3,975 | 86 | 1 | bevy_math |
| `prism_world_network` | 3 | 3,735 | 24 | 1 | — |
| `prism_world_persist` | 15 | 3,587 | 22 | 4 | — |
| `prism_ui_ecs` | 11 | 3,542 | 88 | 5 | — |
| `prism_material_schema` | 4 | 3,529 | 26 | 6 | — |
| `prism_render_gpudriven` | 15 | 3,447 | 18 | 9 | — |
| `prism_audio_music` | 13 | 3,431 | 61 | 1 | bevy_math |
| `prism_material_pcm` | 10 | 3,315 | 28 | 5 | — |
| `prism_render_material_authoring` | 13 | 3,285 | 28 | 4 | — |
| `prism_ui_i18n` | 16 | 3,270 | 61 | 4 | — |
| `prism_world_sim` | 8 | 3,194 | 27 | 3 | — |
| `prism_audio_resample` | 10 | 3,109 | 57 | 1 | bevy_math |
| `prism_ui_style` | 12 | 3,037 | 48 | 4 | — |
| `prism_ui_input` | 11 | 3,013 | 65 | 7 | — |
| `prism_virtual_texture_gpu` | 12 | 2,967 | 32 | 3 | — |
| `prism_audio_geometry_gpu` | 13 | 2,828 | 18 | 1 | bevy_math |
| `prism_ui_inspector` | 9 | 2,736 | 69 | 6 | — |
| `prism_window` | 14 | 2,604 | 44 | 11 | — |
| `prism_motion_gpu` | 11 | 2,578 | 25 | 0 | — |
| `prism_ui` | 11 | 2,572 | 40 | 27 | — |
| `prism_audio_physics` | 14 | 2,513 | 65 | 1 | bevy_math |
| `prism_ui_reactive` | 12 | 2,496 | 44 | 4 | — |
| `prism_ui_theme` | 8 | 2,488 | 53 | 4 | — |
| `prism_ui_timetravel` | 9 | 2,466 | 47 | 6 | — |
| `prism_asset_build_cli` | 10 | 2,422 | 21 | 2 | — |
| `prism_ui_a11y` | 13 | 2,379 | 57 | 5 | — |
| `prism_ecs_macros` | 10 | 2,378 | 62 | 1 | — |
| `prism_ui_snapshot` | 9 | 2,366 | 44 | 2 | — |
| `prism_audio_snapshot` | 13 | 2,359 | 69 | 1 | bevy_math |
| `prism_ui_anim` | 10 | 2,332 | 42 | 3 | — |
| `prism_audio_profiler` | 10 | 2,294 | 35 | 1 | bevy_math |
| `prism_ui_sdui` | 10 | 2,212 | 53 | 3 | — |
| `prism_camera_rig` | 2 | 2,203 | 15 | 1 | — |
| `prism_audio_remote` | 9 | 2,179 | 49 | 1 | bevy_math |
| `prism_audio_haptics` | 9 | 2,177 | 61 | 1 | bevy_math |
| `prism_ui_hotreload` | 9 | 2,177 | 52 | 4 | — |
| `prism_ui_motion` | 8 | 2,167 | 47 | 3 | — |
| `prism_input` | 12 | 2,122 | 46 | 9 | — |
| `prism_ui_macro` | 9 | 2,110 | 56 | 1 | — |
| `prism_ui_router` | 9 | 2,110 | 54 | 4 | — |
| `prism_audio_device` | 9 | 1,997 | 29 | 4 | — |
| `prism_ui_workbench` | 7 | 1,959 | 40 | 3 | — |
| `prism_image` | 8 | 1,952 | 18 | 3 | — |
| `prism_audio_exec` | 10 | 1,951 | 39 | 1 | — |
| `prism_ui_form` | 9 | 1,940 | 45 | 4 | — |
| `prism_gdeflate_gpu` | 9 | 1,901 | 22 | 3 | — |
| `prism_window_winit` | 9 | 1,893 | 43 | 6 | — |
| `prism_ui_overlay` | 8 | 1,885 | 35 | 4 | — |
| `prism_ui_async` | 10 | 1,871 | 44 | 4 | — |
| `prism_audio_accessibility` | 8 | 1,862 | 38 | 1 | bevy_math |
| `prism_world` | 6 | 1,727 | 18 | 42 | — |
| `prism_water_render` | 11 | 1,710 | 2 | 1 | — |
| `prism_audio_output` | 5 | 1,633 | 37 | 2 | bevy_math |
| `prism_shader` | 8 | 1,517 | 18 | 3 | — |
| `prism_micromap` | 9 | 1,423 | 19 | 3 | — |
| `prism_camera_gpu` | 2 | 1,394 | 7 | 1 | — |
| `prism_audio_rt` | 7 | 1,324 | 11 | 3 | — |
| `prism_ui_scheduler` | 9 | 1,324 | 17 | 4 | — |
| `prism_camera_ecs` | 2 | 1,323 | 9 | 1 | — |
| `prism_ui_scoped` | 6 | 1,306 | 31 | 2 | — |
| `prism_ui_virtual` | 8 | 1,294 | 31 | 5 | — |
| `prism_ui_tree` | 7 | 1,136 | 12 | 3 | — |
| `prism_platform_os` | 7 | 1,114 | 8 | 4 | — |
| `prism_reflect_macros` | 2 | 1,092 | 2 | 1 | — |
| `prism_ui_component` | 6 | 1,080 | 22 | 4 | — |
| `prism_transform_ecs` | 4 | 1,015 | 12 | 1 | — |
| `prism_dmm` | 8 | 1,004 | 14 | 0 | — |
| `prism_state_tree_gpu` | 6 | 983 | 7 | 1 | — |
| `prism_world_stream` | 2 | 967 | 5 | 3 | — |
| `prism_radiance_cascades_gpu` | 5 | 929 | 6 | 0 | — |
| `prism_dmm_gpu` | 5 | 901 | 5 | 0 | — |
| `prism_ui_font` | 3 | 875 | 11 | 2 | — |
| `prism_transform_gpu` | 5 | 843 | 5 | 1 | — |
| `prism_micromap_gpu` | 5 | 830 | 5 | 3 | — |
| `prism_ray_reorder_gpu` | 8 | 793 | 6 | 0 | — |
| `prism_ui_devtools` | 5 | 736 | 15 | 6 | — |
| `prism_audio_codec` | 4 | 717 | 11 | 0 | — |
| `prism_ui_store` | 4 | 567 | 11 | 3 | — |
| `prism_audio_bevy` | 2 | 187 | 4 | 4 | — |
| `prism_bevy` | 0 | 0 | 0 | 5 | — |
| **合计 141 crate** | | **2,670,701** | **42,436** | | 30 个耦合 |

### 附录 B：docs/ 自述状态标记与验收/门禁行数

| 文档 | 行数 | ✅ | 🟡 | ❌ | PLANNED 行 | 验收/门禁表行 |
|---|---:|---:|---:|---:|---:|---:|
| `docs/refactoring-dev/05-render-fabric_refactor_zh.md` | 1248 | 3 | 0 | 6 | 0 | 53 |
| `docs/prism_aaa_advanced_features_zh.md` | 1195 | 34 | 93 | 0 | 0 | 24 |
| `docs/prism_material_pipeline_refactor_verification_zh.md` | 207 | 33 | 0 | 0 | 0 | 14 |
| `docs/prism_material_pipeline_refactor_zh.md` | 1785 | 41 | 1 | 0 | 0 | 13 |
| `docs/prism_asset_build_design_zh.md` | 1479 | 0 | 0 | 0 | 0 | 5 |
| `docs/prism_material_pipeline_design_zh.md` | 650 | 14 | 0 | 0 | 0 | 4 |
| `docs/prism_mesh_design_zh.md` | 1112 | 11 | 0 | 0 | 0 | 3 |
| `docs/prism_openpbr_pipeline_function_zh.md` | 721 | 13 | 0 | 0 | 0 | 3 |
| `docs/prism_window_refactor_zh.md` | 485 | 0 | 0 | 0 | 0 | 3 |
| `docs/prism_ecs_completion_zh.md` | 44 | 0 | 0 | 0 | 0 | 3 |
| `docs/prism_rendering_architecture_zh.md` | 1473 | 0 | 0 | 0 | 0 | 2 |
| `docs/prism_state_tree_design_zh.md` | 1128 | 0 | 0 | 0 | 0 | 2 |
| `docs/prism_particle_engine_design_zh.md` | 696 | 0 | 0 | 0 | 0 | 2 |
| `docs/prism_input_refactor_zh.md` | 516 | 0 | 0 | 0 | 0 | 2 |
| `docs/prism_camera_design_zh.md` | 1289 | 0 | 0 | 0 | 0 | 1 |
| `docs/prism_scene_design_zh.md` | 1113 | 0 | 0 | 0 | 0 | 1 |
| `docs/prism_world_system_design_zh.md` | 1099 | 0 | 0 | 0 | 0 | 1 |
| `docs/prism_ecs_design_zh.md` | 695 | 0 | 0 | 0 | 5 | 1 |
| `docs/prism_animation_engine_design_zh.md` | 516 | 5 | 0 | 0 | 0 | 1 |
| `docs/prism_hair_engine_design_zh.md` | 461 | 0 | 0 | 0 | 0 | 1 |
| `docs/refactoring-dev/02-response-composition_refactor_zh.md` | 178 | 0 | 0 | 0 | 0 | 1 |
| `docs/refactoring-dev/01-surface-abi_refactor_zh.md` | 123 | 0 | 0 | 0 | 0 | 1 |
| `docs/gpu_cross_sandbox_implementation_zh.md` | 102 | 0 | 0 | 0 | 0 | 1 |
| `docs/prism_rhi_m0_m2_plan_zh.md` | 58 | 0 | 0 | 0 | 0 | 1 |
| `docs/prism_window_winit_design_zh.md` | 1907 | 0 | 0 | 0 | 0 | 0 |
| `docs/ue_vulkan_internal_port_plan_zh.md` | 1797 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_audio_engine_design_zh.md` | 1616 | 67 | 0 | 0 | 0 | 0 |
| `docs/prism_physics_design_zh.md` | 1264 | 14 | 1 | 3 | 0 | 0 |
| `docs/prism_gameplay_design_zh.md` | 1116 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_asset_design_zh.md` | 1105 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_render_graph_design_zh.md` | 830 | 0 | 0 | 0 | 3 | 0 |
| `docs/prism_visual_scripting_design_zh.md` | 754 | 4 | 0 | 2 | 0 | 0 |
| `docs/prism_game_creator_design_zh.md` | 745 | 1 | 1 | 0 | 5 | 0 |
| `docs/prism_character_controller_design_zh.md` | 740 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_render_driver_design_zh.md` | 670 | 20 | 5 | 3 | 4 | 0 |
| `docs/prism_elemental_reactions_design_zh.md` | 661 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_script_design_zh.md` | 658 | 3 | 0 | 1 | 0 | 0 |
| `docs/prism_network_design_zh.md` | 613 | 9 | 0 | 0 | 0 | 0 |
| `docs/prism_ui_loom_design_zh.md` | 604 | 63 | 0 | 0 | 1 | 0 |
| `docs/prism_ui_component_kit_design_zh.md` | 580 | 0 | 0 | 0 | 2 | 0 |
| `docs/prism_transform_design_zh.md` | 577 | 17 | 0 | 0 | 5 | 0 |
| `docs/prism_anticheat_design_zh.md` | 557 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_gi_lumen_design_zh.md` | 552 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_app_design_zh.md` | 525 | 7 | 0 | 0 | 1 | 0 |
| `docs/prism_tasks_design_zh.md` | 507 | 7 | 1 | 0 | 1 | 0 |
| `docs/prism_math_design_zh.md` | 496 | 10 | 0 | 0 | 0 | 0 |
| `docs/prism_editor_remote_design_zh.md` | 480 | 2 | 0 | 0 | 4 | 0 |
| `docs/prism_reflect_design_zh.md` | 476 | 15 | 0 | 0 | 1 | 0 |
| `docs/prism_time_design_zh.md` | 466 | 8 | 0 | 0 | 3 | 0 |
| `docs/prism_editor_framework_design_zh.md` | 460 | 2 | 0 | 0 | 2 | 0 |
| `docs/prism_volumetric_engine_design_zh.md` | 458 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_diagnostic_design_zh.md` | 457 | 7 | 0 | 0 | 1 | 0 |
| `docs/prism_terrain_world_design_zh.md` | 441 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_platform_design_zh.md` | 439 | 34 | 12 | 7 | 7 | 0 |
| `docs/prism_loom_text_rendering_design_zh.md` | 421 | 0 | 0 | 0 | 3 | 0 |
| `docs/prism_utils_design_zh.md` | 421 | 26 | 0 | 0 | 3 | 0 |
| `docs/prism_water_engine_design_zh.md` | 415 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_loom_runtime_framework_design_zh.md` | 407 | 2 | 0 | 0 | 1 | 0 |
| `docs/prism_world_system_implementation_zh.md` | 327 | 0 | 0 | 0 | 0 | 0 |
| `docs/linux_dependencies.md` | 288 | 0 | 0 | 0 | 0 | 0 |
| `docs/CODE_RULES.md` | 276 | 0 | 0 | 0 | 4 | 0 |
| `docs/prism_ui_dock_design_zh.md` | 272 | 0 | 0 | 0 | 0 | 0 |
| `docs/profiling.md` | 256 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_bevy_refactor_plan_zh.md` | 246 | 4 | 0 | 0 | 0 | 0 |
| `docs/prism_app_render_acceptance_zh.md` | 236 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_heightfield_placement_zh.md` | 222 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_cloth_engine_design_zh.md` | 220 | 0 | 0 | 0 | 0 | 0 |
| `docs/cargo_features.md` | 218 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_placement_deletions_zh.md` | 198 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_scatter_deletions_zh.md` | 198 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_heightfield_scatter_zh.md` | 195 | 0 | 0 | 0 | 0 | 0 |
| `docs/frame_graph_implementation_zh.md` | 188 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_engine_component_gap_zh.md` | 178 | 2 | 10 | 30 | 3 | 0 |
| `docs/prism_world_gpu_scatter_deletions_zh.md` | 178 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_heightfield_scatter_zh.md` | 174 | 0 | 0 | 0 | 0 | 0 |
| `docs/response_output_implementation_zh.md` | 147 | 0 | 0 | 0 | 0 | 0 |
| `docs/refactoring-dev/04-frame-graph_refactor_zh.md` | 136 | 0 | 0 | 0 | 0 | 0 |
| `docs/refactoring-dev/00-overview_refactor_zh.md` | 131 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_heightfield_placement_deletions_zh.md` | 122 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_derived_data_contract_zh.md` | 119 | 0 | 0 | 0 | 0 | 0 |
| `docs/refactoring-dev/03-shader-package_refactor_zh.md` | 119 | 0 | 0 | 0 | 0 | 0 |
| `docs/pipeline_state_implementation_zh.md` | 118 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_render_scene_rhi_port_zh.md` | 109 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_scatter_field_zh.md` | 108 | 0 | 0 | 0 | 0 | 0 |
| `docs/water_render_fabric_implementation_zh.md` | 107 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_scatter_brush_index_zh.md` | 101 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_rhi_port_surface_zh.md` | 100 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_density_zh.md` | 100 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_heightfield_placement_zh.md` | 98 | 0 | 0 | 0 | 0 | 0 |
| `docs/render_fabric_selection_implementation_zh.md` | 94 | 0 | 0 | 0 | 0 | 0 |
| `docs/pipeline_runtime_implementation_zh.md` | 92 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_sector_proxies_zh.md` | 91 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_scatter_brush_index_zh.md` | 89 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_scatter_zh.md` | 83 | 0 | 0 | 0 | 0 | 0 |
| `docs/water_hybrid_implementation_verification_zh.md` | 83 | 0 | 0 | 0 | 0 | 0 |
| `docs/frame_graph_declaration_validation_zh.md` | 77 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_scatter_field_zh.md` | 73 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_async_runtime_zh.md` | 69 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_native_io_zh.md` | 66 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_terrain_lod_seam_zh.md` | 66 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_resident_zh.md` | 65 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_gpu_sdf_zh.md` | 65 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_planet_placement_zh.md` | 64 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_shared_priority_zh.md` | 60 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_sim_replay_zh.md` | 60 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_io_validation_zh.md` | 52 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_runtime_range_stream_zh.md` | 50 | 0 | 0 | 0 | 0 | 0 |
| `docs/wgpu_native_interop_implementation_zh.md` | 49 | 0 | 0 | 0 | 0 | 0 |
| `docs/linters.md` | 48 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_world_planet_mesh_incremental_zh.md` | 41 | 0 | 0 | 0 | 0 | 0 |
| `docs/prism_virtual_texture_feedback_zh.md` | 40 | 0 | 0 | 0 | 0 | 0 |
| `docs/shading_plan_contract_implementation_zh.md` | 40 | 0 | 0 | 0 | 0 | 0 |

---

### 附录 C：独立复核记录

# 独立复核记录（captain 亲自执行，非子代理转述）

#### V1 `prism_ui_macro` 公开入口不存在
- `pkg/prism_ui_macro/src/lib.rs:61` = `mod hoist;`（私有，无 `pub`）；`hoist.rs:20` 为 `pub(crate) enum StaticClass`。
- `pkg/prism_ui_macro/src/lib.rs:60` = `mod dualmode;`（私有）；`dualmode.rs:28` 为 `pub(crate) enum Mode`。
- 结论：文档所述 crate 外可达的 hoist / dualmode 入口不存在。

#### V2 `prism_ui_render_backend` 的 `gpu` feature 全仓从未启用
- 唯一依赖者：`pkg/prism_ui_component_kit/Cargo.toml:18`，未开启任何 feature。
- 根 `Cargo.toml:215` 仅为 workspace members 列举，非依赖。
- 结论：wgpu/Metal 后端不在任何默认构建路径上。

#### V3 `prism_transform_gpu` 真机门禁静默通过
- `pkg/prism_transform_gpu/tests/hierarchy_parity.rs:28` `with_gpu()` 无 adapter 时仅 `eprintln!("skipping: no usable GPU adapter on this host")` 后 `return None` → 测试照常通过。
- 结论：「5 项真机 parity 全绿」与「全部被跳过」不可区分。

#### V4 音频八个 crate 零消费者（双向确认）
- `.rs`：`prism_audio_exec / device / profiler / comm / wave / conditioning / output / physics` 全仓命中 **0 文件**。
- `Cargo.toml`：唯一命中是根 `Cargo.toml` `members`（:135,:143,:149,:161,:165,:172,:173,:179），**无 `[dependencies]`**。

#### V5 材质/着色文档 ✅ 与代码背离（4 条全部实证）
- `docs/prism_material_pipeline_refactor_verification_zh.md:102` 称 P2-j 分类注册化 ✅，证据 `classification.rs:453 from_registry`。实测 `pkg/prism_render_shading/src/classification.rs` 仅 **709 行**；`from_registry` 在该文件 0 命中；`class_count` / `UnhandledResponseMode` 全仓 `.rs` 0 命中。
- `:90` 称 `MATERIAL_ABI_VERSION = 6`（record.rs:18）。实测 `pkg/prism_render_material/src/lib.rs:112` 为 `pub use surface_layout::SURFACE_ABI_VERSION as MATERIAL_ABI_VERSION;`，`surface_layout.rs:13` **= 8**。
- `:94` 称 `pipeline_cache.rs:289 MAX_IN_FLIGHT_COMPILES=4`。实测该常量全仓 0 命中。
- `pkg/prism_render_material/src/registry.rs:288 install_execution_tables` 全仓唯一命中即定义本身 ⇒ `pkg/prism_render_scene/src/shading/resources.rs:434` 的 `from_execution_tables` 分支不可达。

#### V6 Bevy 耦合精确计数（修正 §1.3 与附表 A）
`pkg/*/Cargo.toml` 中 `crates/bevy_*` path 依赖解析（已排除 `description` 等假阳性）：
- 依赖 `crates/bevy_*` 的 pkg crate 共 **30** 个。
- **仅依赖 `bevy_math`：27 个**。
- 深层耦合 **3** 个：`prism_render_scene`（19 个 bevy_*）、`prism_render_material`（6：asset/color/image/material/math/pbr）、`prism_render_visibility`（4：camera/math/shape/transform）。
- `pkg/prism_physics_gpu/Cargo.toml:12-14` 确认 `bevy_math = { path = "../../crates/bevy_math", default-features = false, features = ["nostd-libm"] }`。
- 修正：子代理报告的「约 35 个」与附表原写的「28 仅 bevy_math / 2 深层」均不准确。

#### V7 物理/资产 头条结论复核
- `RagdollBuilder` / `featherstone` / `snap_align` / `SnapPose` / `reduced_coordinates` 在 `pkg/` 下 `*.rs` **全仓 0 命中**；`Articulation` 的 2 个文件命中实为 `pkg/prism_audio_spatial/src/articulation_loss.rs` 的 **`ArticulationLoss`（Peutz-Klein 辅音清晰度）**，与 Featherstone 关节树无关 ⇒ M10/M13/M14/M15 里程碑确实未落地。
- `pkg/prism_asset/Cargo.toml` 对 `prism_reflect` **0 命中**（与文档 `:938`/`:1100` 要求不符）；`pkg/prism_asset_macros` 目录不存在。
- `pkg/prism_bevy/` 实测仅 `src/`、`tests/` 两个空目录（无 `Cargo.toml`、无 `.rs`）；`pkg/prism_bevy_ui`、`pkg/prism_bevy_audio`、`pkg/prism_bevy_physics` 均**不存在**。

#### V8 全仓反向依赖图（客观机器判定，独立于子代理自述）
方法：遍历仓库内**所有** `Cargo.toml`（排除 `target/`、`.git/`、`crates/`），解析 `path = "…"` 依赖并把目标解析为 `pkg/<name>`，源记为所属 crate 名（`pkg/*` 用目录名，其余用其目录路径）。同 crate 自引用（dev-dependency 自引）不计。

结果：
- `pkg/` 真实 crate **140** 个；指向 `pkg/*` 的内部 path 依赖边 **323** 条。
- **零消费者 crate 共 59 个**（占 140 的 **42%**）——这些 crate 在全仓任何 `Cargo.toml` 中都没有被别的 crate 依赖过一次。
- 注意口径：本计数**把 dev-dependency 也算作消费者**，因此严格「仅非测试生产消费者」口径下的零接通数 **≥ 59**（只会更多，不会更少）。
- 完整名单与消费者数已存 `/tmp/prism_audit/reverse_deps.tsv`。名单（59）：prism_asset_build_cli, prism_audio_accessibility, prism_audio_authoring, prism_audio_bevy, prism_audio_codec, prism_audio_comm, prism_audio_conditioning, prism_audio_device, prism_audio_exec, prism_audio_geometry_gpu, prism_audio_haptics, prism_audio_immersive, prism_audio_midi, prism_audio_music, prism_audio_output, prism_audio_physics, prism_audio_profiler, prism_audio_remote, prism_audio_snapshot, prism_audio_wave, prism_camera_ecs, prism_camera_gpu, prism_dmm_gpu, prism_gdeflate_gpu, prism_hair_gpu, prism_input, prism_math_gpu, prism_micromap_gpu, prism_motion_gpu, prism_physics_gpu, prism_radiance_cascades_gpu, prism_ray_reorder_gpu, prism_render_app, prism_render_material_authoring, prism_render_material_gpu, prism_render_scene, prism_shader, prism_state_tree_gpu, prism_transform_ecs, prism_transform_gpu, prism_ui_async, prism_ui_ecs, prism_ui_form, prism_ui_hotreload, prism_ui_i18n, prism_ui_inspector, prism_ui_motion, prism_ui_overlay, prism_ui_router, prism_ui_scheduler, prism_ui_scoped, prism_ui_sdui, prism_ui_store, prism_ui_text, prism_ui_virtual, prism_virtual_geometry_gpu, prism_volumetric_gpu, prism_water_render, prism_world_network。
- 被依赖最多的 crate：prism_audio_core 29、prism_ui 22、prism_render_architecture 19、prism_render_driver 16、prism_ecs 15、prism_math 13、prism_ui_reactive 13、prism_asset 10、prism_platform 9。

### 关键结构事实：pkg/ 之外**几乎没有任何消费者**
仓库根 `Cargo.toml` 的 `[package]` 是 **`name = "bevy"` / `version = "0.20.0-dev"` / `repository github.com/bevyengine/bevy`**——即仓库根仍是上游 Bevy 老栈的门面包，`[workspace] members` 里虽然列了 `pkg/*`，但根的 `[dependencies]`/`[dev-dependencies]` 对 `pkg/` crate **0 引用**（`grep -c 'path = "pkg/' Cargo.toml` = 0；第 27/50/56/78 行的 `pkg/prism_*` 全是 workspace members 列举）。
`pkg/` 之外仅 **4 个** manifest 依赖 pkg crate，合计只拉到 **8 个**不同 crate：
- `benches/Cargo.toml` → prism_render_architecture、prism_virtual_texture_gpu、prism_render_driver、prism_render_gpudriven、prism_physics_core、prism_physics_geometry
- `os-backends/prism_render_driver_wgpu/Cargo.toml`、`os-backends/prism_render_driver_null/Cargo.toml` → prism_render_driver
- `tools/render_triangle/Cargo.toml` → prism_render_driver、prism_render_graph、prism_render_runtime

⇒ 仓库存在**两套互不消费的平行世界**：`crates/bevy_*` + 根 package + `examples/`/`tests/`（老栈）与 `pkg/`（新栈 140 crate）。**没有任何应用把 pkg/ 装配起来**。

---

### 附录 D：全仓反向依赖图（客观机器判定）

方法：遍历仓库内全部 `Cargo.toml`（排除 `target/`、`.git/`、`crates/`），把 `path = "…"` 依赖解析到 `pkg/<crate>`，源记为所属 crate 目录名（`pkg/*` 用目录名，其余用其相对路径），同 crate 自引不计。**dev-dependency 亦计为消费者**，故严格「仅非测试生产消费者」口径下的零接通数只会 ≥ 本表。

- `pkg/` 真实 crate：**140**（`pkg/` 下 141 个目录，其中 `pkg/prism_bevy/` 为空目录）
- 去重后的 crate→crate 依赖边（不同消费者 crate 计数）：**320**
- 零消费者 crate：**59**（占 42%）

#### D.1 零消费者 crate 清单（59 个）

| crate | 所属子系统 |
|---|---|
| `prism_asset_build_cli` | 2.12 物理・资产・Bevy 桥 |
| `prism_audio_accessibility` | 2.9 音频 part 2 |
| `prism_audio_authoring` | 2.9 音频 part 2 |
| `prism_audio_bevy` | 2.12 物理・资产・Bevy 桥 |
| `prism_audio_codec` | 2.9 音频 part 2 |
| `prism_audio_comm` | 2.8 音频 part 1 |
| `prism_audio_conditioning` | 2.8 音频 part 1 |
| `prism_audio_device` | 2.8 音频 part 1 |
| `prism_audio_exec` | 2.8 音频 part 1 |
| `prism_audio_geometry_gpu` | 2.8 音频 part 1 |
| `prism_audio_haptics` | 2.9 音频 part 2 |
| `prism_audio_immersive` | 2.9 音频 part 2 |
| `prism_audio_midi` | 2.9 音频 part 2 |
| `prism_audio_music` | 2.9 音频 part 2 |
| `prism_audio_output` | 2.8 音频 part 1 |
| `prism_audio_physics` | 2.8 音频 part 1 |
| `prism_audio_profiler` | 2.8 音频 part 1 |
| `prism_audio_remote` | 2.9 音频 part 2 |
| `prism_audio_snapshot` | 2.9 音频 part 2 |
| `prism_audio_wave` | 2.8 音频 part 1 |
| `prism_camera_ecs` | 2.10 平台・窗口・输入・相机 |
| `prism_camera_gpu` | 2.10 平台・窗口・输入・相机 |
| `prism_dmm_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_gdeflate_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_hair_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_input` | 2.10 平台・窗口・输入・相机 |
| `prism_math_gpu` | 2.1 ECS/核心运行时 |
| `prism_micromap_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_motion_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_physics_gpu` | 2.12 物理・资产・Bevy 桥 |
| `prism_radiance_cascades_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_ray_reorder_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_render_app` | 2.3 渲染应用层 |
| `prism_render_material_authoring` | 2.3 渲染应用层 |
| `prism_render_material_gpu` | 2.3 渲染应用层 |
| `prism_render_scene` | 2.3 渲染应用层 |
| `prism_shader` | 2.4 GPU 几何・特性・派生 |
| `prism_state_tree_gpu` | 2.1 ECS/核心运行时 |
| `prism_transform_ecs` | 2.1 ECS/核心运行时 |
| `prism_transform_gpu` | 2.1 ECS/核心运行时 |
| `prism_ui_async` | 2.7 UI 高级层 |
| `prism_ui_ecs` | 2.6 UI 核心 |
| `prism_ui_form` | 2.6 UI 核心 |
| `prism_ui_hotreload` | 2.7 UI 高级层 |
| `prism_ui_i18n` | 2.7 UI 高级层 |
| `prism_ui_inspector` | 2.7 UI 高级层 |
| `prism_ui_motion` | 2.7 UI 高级层 |
| `prism_ui_overlay` | 2.7 UI 高级层 |
| `prism_ui_router` | 2.7 UI 高级层 |
| `prism_ui_scheduler` | 2.6 UI 核心 |
| `prism_ui_scoped` | 2.6 UI 核心 |
| `prism_ui_sdui` | 2.7 UI 高级层 |
| `prism_ui_store` | 2.6 UI 核心 |
| `prism_ui_text` | 2.6 UI 核心 |
| `prism_ui_virtual` | 2.7 UI 高级层 |
| `prism_virtual_geometry_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_volumetric_gpu` | 2.4 GPU 几何・特性・派生 |
| `prism_water_render` | 2.4 GPU 几何・特性・派生 |
| `prism_world_network` | 2.11 世界系统 |

#### D.2 被依赖最多的 25 个 crate

| crate | 消费者数 | 主要消费者 |
|---|---:|---|
| `prism_audio_core` | 29 | `prism_audio_accessibility`, `prism_audio_assets`, `prism_audio_authoring`, `prism_audio_bevy`, `prism_audio_codec`, `prism_audio_comm` |
| `prism_ui` | 22 | `prism_ui_a11y`, `prism_ui_async`, `prism_ui_component`, `prism_ui_component_kit`, `prism_ui_devtools`, `prism_ui_dock` |
| `prism_render_architecture` | 19 | `benches`, `prism_asset_build_core`, `prism_camera`, `prism_camera_ecs`, `prism_gdeflate_gpu`, `prism_hair_gpu` |
| `prism_render_driver` | 16 | `benches`, `os-backends/prism_render_driver_null`, `os-backends/prism_render_driver_wgpu`, `prism_asset_build_core`, `prism_camera_gpu`, `prism_image` |
| `prism_ecs` | 15 | `prism_app`, `prism_asset`, `prism_audio_bevy`, `prism_audio_ecs`, `prism_camera_ecs`, `prism_render_app` |
| `prism_math` | 13 | `prism_audio_bevy`, `prism_audio_ecs`, `prism_camera`, `prism_camera_ecs`, `prism_camera_rig`, `prism_ecs` |
| `prism_ui_reactive` | 13 | `prism_ui`, `prism_ui_async`, `prism_ui_component`, `prism_ui_ecs`, `prism_ui_form`, `prism_ui_i18n` |
| `prism_asset` | 10 | `prism_asset_build_cli`, `prism_asset_build_core`, `prism_asset_container`, `prism_audio_ecs`, `prism_image`, `prism_mesh` |
| `prism_platform` | 9 | `prism_app`, `prism_diagnostic`, `prism_ecs`, `prism_math`, `prism_platform_os`, `prism_tasks` |
| `prism_tasks` | 8 | `prism_app`, `prism_asset`, `prism_asset_build_core`, `prism_ecs`, `prism_render_app`, `prism_transform` |
| `prism_ui_style` | 8 | `prism_ui`, `prism_ui_component_kit`, `prism_ui_hotreload`, `prism_ui_motion`, `prism_ui_render_backend`, `prism_ui_scoped` |
| `prism_app` | 6 | `prism_audio_bevy`, `prism_audio_ecs`, `prism_camera_ecs`, `prism_render_app`, `prism_replication`, `prism_state_tree_ecs` |
| `prism_audio_spatial` | 6 | `prism_audio_geometry`, `prism_audio_geometry_gpu`, `prism_audio_hrtf`, `prism_audio_immersive`, `prism_audio_object`, `prism_audio_wave` |
| `prism_reflect` | 6 | `prism_app`, `prism_ecs`, `prism_replication`, `prism_scene`, `prism_state_tree_ecs`, `prism_ui_inspector` |
| `prism_render_graph` | 6 | `prism_render_app`, `prism_render_architecture`, `prism_render_gpudriven`, `prism_render_runtime`, `prism_render_scene`, `tools/render_triangle` |
| `prism_render_material` | 6 | `prism_render_app`, `prism_render_material_authoring`, `prism_render_material_gpu`, `prism_render_scene`, `prism_render_shading`, `prism_render_visibility` |
| `prism_transform` | 6 | `prism_audio_bevy`, `prism_audio_ecs`, `prism_ecs`, `prism_render_app`, `prism_transform_ecs`, `prism_transform_gpu` |
| `prism_ui_layout` | 5 | `prism_ui`, `prism_ui_overlay`, `prism_ui_render_backend`, `prism_ui_snapshot`, `prism_ui_virtual` |
| `prism_world_stream` | 5 | `prism_world`, `prism_world_gpu`, `prism_world_network`, `prism_world_runtime`, `prism_world_terrain` |
| `prism_asset_container` | 4 | `prism_asset_build_cli`, `prism_asset_build_core`, `prism_image`, `prism_world_runtime` |
| `prism_mesh` | 4 | `prism_asset_build_core`, `prism_render_app`, `prism_render_scene`, `prism_world_terrain` |
| `prism_physics_core` | 4 | `benches`, `prism_audio_physics`, `prism_physics_gpu`, `prism_render_architecture` |
| `prism_physics_geometry` | 4 | `benches`, `prism_audio_geometry`, `prism_physics_core`, `prism_transform_ecs` |
| `prism_ui_devtools` | 4 | `prism_ui_inspector`, `prism_ui_snapshot`, `prism_ui_timetravel`, `prism_ui_workbench` |
| `prism_world_persist` | 4 | `prism_world`, `prism_world_network`, `prism_world_runtime`, `prism_world_sim` |

#### D.3 `pkg/` 之外的消费者（全部 4 个 manifest）

| manifest | 消费的 pkg crate |
|---|---|
| `benches/Cargo.toml` | prism_render_architecture, prism_virtual_texture_gpu, prism_render_driver, prism_render_gpudriven, prism_physics_core, prism_physics_geometry |
| `os-backends/prism_render_driver_wgpu/Cargo.toml` | prism_render_driver |
| `os-backends/prism_render_driver_null/Cargo.toml` | prism_render_driver |
| `tools/render_triangle/Cargo.toml` | prism_render_driver, prism_render_graph, prism_render_runtime |

> 仓库根 `Cargo.toml` 是上游 Bevy 门面包（`name = "bevy"`、`0.20.0-dev`），其对 `pkg/` 的依赖数为 0；`pkg/prism_*` 仅出现在 `[workspace] members` 列举中。

