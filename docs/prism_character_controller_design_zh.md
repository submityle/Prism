# Prism Character Controller 次世代角色控制器设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的数据导向、运动学/动力学混合、可预测可回滚的角色控制器（CCT, Character Controller）设计。
> 借鉴 UE5（CharacterMovementComponent / Mover 2.0 / Motion Warping）、Jolt（CharacterVirtual）、PhysX/Unity（CharacterController）、Rapier/Avian KCC、bevy_tnua（浮空射线 CCT）、Godot（CharacterBody），并吸收 Naughty Dog（Uncharted/TLOU）、Insomniac（Spider-Man）、Respawn（Titanfall/Apex）、Doom Eternal、Celeste、Super Mario Odyssey 的手感工程经验。
> 本文档为设计规格，不含任何 AI/ML 内容；采用纯经典数值几何 + 状态机路线。

- 版本: v0.1（设计阶段）
- 适用引擎: Prism / prism_ecs 生态（脱离 Bevy 后的独立引擎，不依赖任何 bevy_* crate）
- 关键依赖（均为 `pkg/` 下的 prism 原生 crate，非 Bevy）: `prism_ecs`（数据导向 ECS：组件/查询/关系/反应，脱离 Bevy 的 greenfield 内核）、`prism_math`（glam SIMD）、`prism_math_gpu`、`prism_transform`、`prism_time`（固定 tick / 时间线，§4D）、`prism_input`、`prism_tasks`（fiber job graph 并行）、`prism_reflect`（反射 / 序列化 / 脏标）、`prism_asset`（Profile 资产 / 热重载）、`prism_app`（插件与调度）、`prism_physics_core` / `prism_physics_geometry`（物理内核：Sweep/Overlap/Raycast/GJK-EPA/BVH/SDF）、`prism_diagnostic`（计数器 / trace）；动画耦合见 `prism_animation_engine_design_zh.md`（Root Motion / Motion Matching / IK）
- 相关文档: `prism_physics_design_zh.md`（物理内核与查询）、`prism_gameplay_design_zh.md`（§7 Pawn/Controller/Character、§10 增强输入、§16 网络预测、§37 手感层、§41 动画）、`prism_animation_engine_design_zh.md`、`prism_network_design_zh.md`

---

## 目录
1. 设计哲学
2. 现状基线与差距
3. 对标与借鉴（取长补短总表）
4. 分层架构
5. 核心内核：运动求解器（Collide-and-Slide）
6. 碰撞体与形状探测
7. 地面检测与地面接触模型
8. 移动模式状态机（MovementMode）
9. 运动学 / 动力学 / 混合三档
10. 手感工程层（Game Feel）
11. 高级移动能力（Mantle / Vault / Climb / Wallrun / Slide / Dash）
12. 动画耦合：Root Motion / Motion Warping / Motion Matching / IK
13. 物理耦合：移动平台 / 推力 / 被推 / 载具
14. 相机耦合与视角
15. 输入接入（Enhanced Input 对标）
16. 与网络框架的接缝（预测/回滚归网络文档）
17. 确定性
18. 性能工程与帧预算
19. 数据驱动与可配置（Movement Profile）
20. 调试与可观测性
21. 易用性分层与默认体验
22. 公共 API 草案
23. Crate 拆分与落地形态
24. 一帧内的数据流与时序
25. 路线图
26. 关键扩展点清单
27. 落地与验收清单
28. 术语表

---

## 1. 设计哲学

对标商用旗舰引擎的角色移动层，确立六条铁律：

- **几何求解优先，物理耦合可控**：角色移动的"硬需求"是精确的碰撞响应（不穿墙、不卡缝、台阶/斜坡可预测），这天然属于几何扫掠（Sweep）+ 滑动（Slide）问题，而非把角色丢给刚体求解器听天由命。核心内核采用**运动学 Collide-and-Slide**，动力学只作为可选耦合层。这是对标 UE CMC / Jolt CharacterVirtual 的主路线。
- **固定步长，可预测即可回滚**：移动求解跑在固定时间步（固定 tick，接 `prism_time` §4D）上，输入→状态推进是纯函数式的 `(state, input, dt) -> state`，天然满足客户端预测 + 服务器回滚（对标 UE Network Prediction / Mover 2.0）。
- **手感是一等公民**：土狼时间、跳跃缓冲、加速度曲线、空中控制、转向插值这些"玄学"参数不是补丁，而是内核显式的数据字段，可被 Movement Profile 序列化、热重载、按角色/状态切换。对标 Celeste/Doom 的工程化手感。
- **数据驱动，分层下钻**：默认一个 `CharacterController` bundle 即可走跑跳；进阶用户可逐层替换地面探测策略、移动模式集合、求解迭代参数，直到自定义约束。分层模块化、可逐层下钻（借鉴 Bevy 的 modular 理念，但不依赖 Bevy，构建于 prism 原生 crate 之上）。
- **引擎无关内核**：求解器（Layer 1-2）只依赖几何查询 trait（`SweepProvider`/`OverlapProvider`），不依赖 prism_ecs，可被服务器逻辑/其它引擎复用。引擎集成层（`prism_app`/`prism_ecs`）只做组件同步与调度桥接。对标 Rapier 的"内核 + 胶水"分层。
- **确定性可选**：提供 f32 高性能路径 + 定点/有序确定性路径，编译期切换，服务联机同步与录像回放，零运行时代价。

设计取舍总表：

| 维度 | Prism 选择 | 对标与理由 |
|---|---|---|
| 主求解范式 | 运动学 Collide-and-Slide + 深度解穿 | UE CMC / Jolt CharacterVirtual：精确、可预测、易调 |
| 动力学 | 可选浮空射线（float ride）/ 力耦合档 | bevy_tnua / 动力学档，处理动态场景交互 |
| 时间步 | 固定 tick（prism_time §4D）+ 插值渲染 | 预测回滚、确定性前提 |
| 碰撞体 | 胶囊为主，支持多胶囊/球扫掠 | 工业标准，台阶/斜坡表现稳定 |
| 地面检测 | 扫掠探地 + 接触法线分类 | 比单射线鲁棒，解决边缘/台阶 |
| 内存 | SoA 热数据 + ECS 组件 | 并行/SIMD 对齐 |
| 并行 | 无耦合角色批量并行；接触体分岛 | prism_tasks（fiber job graph）高并行 |
| 网络 | 固定步长预测 + 快照回滚 + Lag Comp | UE Network Prediction / GGPO 思路 |
| 手感 | 显式数据字段 + 曲线资产 | Celeste/Doom 工程化手感 |
| API | 默认零配置 bundle，分层下钻 | 分层模块化（借鉴 Bevy 理念，非依赖） |

---

## 2. 现状基线与差距

Prism 现状（基于仓库）：
- `prism_gameplay_design_zh.md` §7.4 定义了 `Character = Pawn + CharacterMovement`（胶囊 + 地面检测 + 状态机），并显式写明"与 Prism Physics 的角色控制器对接（见物理设计文档）"——**但物理文档当前未落地独立 CCT 章节，形成设计缺口**。本文档即填补该缺口，并成为 §7.4 的落地规格。
- `prism_physics_design_zh.md` 提供统一 XPBD 内核、GJK/EPA、BVH、SDF、Sweep/Overlap 查询能力——这是 CCT 内核所需的几何查询底座。
- Prism 相机模块（`prism_render_scene` 相机 / 相机控制器，设计推进中）作为第三/第一人称相机耦合起点。

差距清单：
- 缺少**运动学扫掠求解器**（Collide-and-Slide + depenetration）与其几何查询适配层。
- 缺少**地面接触模型**（斜坡限制、台阶上行、边缘滑落、地面吸附）。
- 缺少**移动模式状态机**（Walking/Falling/Swimming/Flying 可插拔）。
- 缺少**手感工程层**（coyote/buffer/加速曲线/空中控制）。
- 缺少**动画耦合**（Root Motion 驱动位移、Motion Warping 对齐、足部 IK）。
- 缺少**网络预测回滚**与**确定性**路径在移动层的具体规格。
- 缺少**数据驱动 Movement Profile**与调试可视化。

---

## 3. 对标与借鉴（取长补短总表）

| 来源 | 可借鉴的核心能力 | Prism 取舍 |
|---|---|---|
| **UE5 CharacterMovementComponent (CMC)** | MovementMode 枚举、台阶上行、斜坡滑落、移动平台基座、网络预测保存移动 | 吸收模式机与网络模型；重构为组件 + System 分派，去掉巨石类 |
| **UE5 Mover 2.0** | 纯函数式 `SimulationTick`、无状态求解、模块化 MovementMode、原生网络预测 | 作为架构主蓝本：固定步长、函数式推进、可回滚 |
| **UE5 Motion Warping** | 根运动按目标点动态变形（翻越/攀爬对齐） | 作为高级移动能力的位移矫正层 |
| **Jolt CharacterVirtual** | 虚拟角色（不进刚体世界）、精确 collide-and-slide、推动动态体、支持被动态体承载 | 内核求解法主要参考；接触回调驱动双向耦合 |
| **PhysX / Unity CharacterController** | 胶囊扫掠、skinWidth（接触皮肤）、slopeLimit/stepOffset 经典参数 | 采用经典参数语义，降低迁移成本 |
| **Rapier / Avian KinematicCharacterController** | Rust 生态成熟 KCC：平移求解、自动台阶、吸附地面、斜坡滑动、推盒子 | 作为 Rust 实现参考与 API 对齐基准 |
| **bevy_tnua** | 浮空射线（float ride）动力学 CCT，天然处理动态地面与弹性手感 | 作为"动力学档"可选后端 |
| **Godot CharacterBody2D/3D** | `move_and_slide`/`move_and_collide` 极简易用 API、floor_snap、platform 继承 | 吸收其"零配置能跑"的 API 直觉 |
| **Naughty Dog（Uncharted/TLOU）** | 高保真 traversal、程序化攀爬、Motion Matching 落地 | 攀爬/翻越能力 + 动画耦合范式 |
| **Insomniac（Spider-Man）** | 高速移动 + 摆荡 + 墙面吸附的连续运动求解 | 高速 CCA（子步 + CCD）需求来源 |
| **Respawn（Titanfall/Apex）** | 滑铲/蹬墙跑/冲刺的动量保持与链式连招手感 | Wallrun/Slide/Dash 能力集与动量模型 |
| **Doom Eternal** | 快节奏空中机动、冲刺、精确跳跃平台 | 空中控制 + dash 的工程化参数 |
| **Celeste** | 教科书级手感：coyote time、jump buffer、变高跳、dash 缓冲 | 手感层参数与状态机范式直接吸收 |
| **Super Mario Odyssey** | 丰富状态机（翻滚/长跳/墙跳）、相机-移动协同 | 状态机可组合性 + 相机耦合 |
| **Overgrowth** | 程序化 IK 落地、动量驱动动画 | 足部 IK + 倾斜对齐 |

**差异化定位**：Prism 把 CMC 的"巨石组件"拆成 ECS 组件 + 可插拔 System，把 Mover 2.0 的"函数式可回滚"与 Jolt 的"精确 collide-and-slide"合并，把 Celeste 级手感做成**显式数据字段**而非硬编码补丁，并提供**运动学/动力学/混合三档**由同一套 API 覆盖。

---

## 4. 分层架构

```
Layer 5  作者层 Authoring   (Prefab/编辑器/Movement Profile 资产/热重载)
Layer 4  引擎集成层         (prism_app/prism_ecs：Plugin/Component/System/事件/Query；输入→意图桥接)
Layer 3  能力与手感层        (移动模式机、手感、Mantle/Vault/Climb/Slide/Dash、动画耦合)
Layer 2  运动求解内核        (Collide-and-Slide、解穿、地面模型、动量积分)
Layer 1  几何查询适配        (SweepProvider/OverlapProvider/RaycastProvider trait)
Layer 0  物理后端           (prism_physics_core/prism_physics_geometry BVH/GJK-EPA/SDF；或任意实现查询 trait 的后端)
```

要点：
- **边界只在 Layer 1 与 Layer 4**。Layer 1-3 组成"引擎无关角色内核"，不依赖 prism_ecs（纯数据 + trait，`no_std + alloc`），可被专用服务器复用。Layer 0 可换成任意满足查询 trait 的物理后端（`prism_physics_*` / Rapier / 自研）。
- **求解内核无副作用**：`tick(state, input, env, dt) -> (new_state, events)`，不直接写 ECS，使其可在预测/回滚中反复调用。
- 引擎集成层（`prism_ecs`/`prism_app`）负责：从组件读入状态、调用内核、写回 `Transform`（`prism_transform`）/速度组件、广播事件（`Landed`/`StepUp`/`WallHit`）。
- no_std / wasm 友好：关闭并行时退化为单线程；查询 trait 可对接 wasm 侧的 CPU BVH。

数据分布（SoA 热区，便于批量并行求解）：
```
CcKinematics { position, velocity, up, radius, half_height }  // 热：每帧写
CcGroundState { grounded, ground_normal, ground_entity, ground_point, slope_deg }
CcMode { current: MovementMode, time_in_mode, prev }
CcFeel { coyote_timer, jump_buffer_timer, air_time, last_jump_tick }
CcTuning(Handle<MovementProfile>)                              // 冷：共享只读
CcInputIntent { move_dir, want_jump, want_crouch, want_sprint, look } // 每帧由输入写
```

---

## 5. 核心内核：运动求解器（Collide-and-Slide）

角色移动的本质是：给定期望位移 `motion`，在不穿透场景的前提下求出实际位移，并沿碰撞面滑动以保留切向动量。这是对标 UE CMC `SafeMoveUpdatedComponent` + Jolt `CharacterVirtual::ExtendedUpdate` 的核心。

### 5.1 Collide-and-Slide 主循环
```
fn collide_and_slide(pos, motion, max_bounces=4) -> (pos', residual_vel):
  remaining = motion
  for bounce in 0..max_bounces:
      if |remaining| < EPSILON: break
      hit = sweep(shape, pos, dir=normalize(remaining), dist=|remaining| + skin_width)
      if no hit:
          pos += remaining; break
      // 走到接触点前 skin_width 处
      travel = max(hit.distance - skin_width, 0)
      pos += dir * travel
      remaining -= dir * travel
      // 投影到碰撞面切平面（滑动）
      remaining = project_on_plane(remaining, hit.normal)
      // 速度同样投影，保留切向动量
      velocity = project_on_plane(velocity, hit.normal)
  return pos, velocity
```
- **skin_width（接触皮肤）**：始终在表面前留 `skin_width`（PhysX/Unity 语义），避免浮点穿插与抖动；对标 Unity `CharacterController.skinWidth`。
- **max_bounces**：限制单帧滑动投影次数（典型 4），防止角缝死循环；用尽后残余位移丢弃。
- **投影策略**：墙+墙形成的"犄角"用双平面裁剪（crease），把位移投影到两面的交线方向，避免卡角（对标 Quake 经典 `ClipVelocity` + 犄角处理）。

### 5.2 深度解穿（Depenetration / Overlap Recovery）
扫掠无法处理"初始已穿插"（传送、生成、动态体压入）。每子步开头做一次 overlap 恢复：
```
fn depenetrate(pos):
  for contact in overlap(shape, pos):     // GJK/EPA 返回穿透深度 + 法线（MTV）
      pos += contact.normal * (contact.depth + skin_width)
  return pos
```
- 用 EPA 的最小平移向量（MTV）把角色推出；多接触取加权/迭代（最多 K 次）。
- 对动态可推体：解穿力按逆质量分配，角色与盒子互相退让（见 §13）。

### 5.3 子步与连续碰撞（高速角色）
- 单帧位移超过 `radius * 0.5` 时拆子步（substep），每子步独立 collide-and-slide，避免隧穿（对标 Spider-Man 高速移动需求）。
- 极端高速（投射类）启用 **swept CCD**：对首个 TOI（Time of Impact）命中截断，剩余速度在命中帧处理。
- 子步数 = `ceil(|motion| / (radius * k))`，`k` 可调（默认 0.5），上限保护。

### 5.4 速度与位置分离的积分
```
每固定子步 dt:
  1. 加速/减速：velocity = apply_acceleration(velocity, intent, dt)  // 手感曲线，见 §10
  2. 重力/浮力：velocity += mode.external_accel(dt)
  3. 期望位移：motion = velocity * dt
  4. 水平/垂直分解（相对 up）：horizontal + vertical 分别走 slide（台阶处理需分离）
  5. depenetrate -> collide_and_slide -> 更新 pos/velocity
  6. 地面检测（§7）、模式转移（§8）、着陆事件
```
- **水平/垂直分离**是台阶上行与贴地的关键：水平位移先行，再尝试台阶抬升，最后处理垂直（下落/贴地）。对标 UE `MoveAlongFloor` + `StepUp`。

---

## 6. 碰撞体与形状探测

- **主形状：胶囊（Capsule）**。工业标准：圆底平滑过台阶/斜坡，无角点卡顿。参数 `radius` / `half_height`，`up` 轴可配（支持任意重力方向/球面行走）。
- **可选形状**：球（简单 AI）、多胶囊（人形+背包/武器）、垂直胶囊堆叠（蹲下切换半高）。
- **蹲下/站起**：切半高前做"头顶 overlap 测试"，被挡则保持蹲姿（对标 CMC `CanUncrouch`）。
- **形状探测抽象（Layer 1 trait）**：
```rust
trait ShapeQuery {
    fn sweep(&self, shape: &Shape, origin: Isometry, dir: Vec3, max_dist: f32, filter: QueryFilter) -> Option<SweepHit>;
    fn overlap(&self, shape: &Shape, iso: Isometry, filter: QueryFilter, out: &mut Vec<Contact>);
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: QueryFilter) -> Option<RayHit>;
}
```
- `QueryFilter`：层掩码（LayerMask）、排除自身、排除触发器、动态/静态筛选。
- 由 `prism_physics_core` / `prism_physics_geometry` 的 BVH + GJK/EPA + SDF 实现；也可由测试用解析几何实现（单元测试不依赖物理后端）。

---

## 7. 地面检测与地面接触模型

比"单射线判地"鲁棒得多，是手感与台阶稳定的关键。

### 7.1 探地（Ground Probe）
- 在角色脚下做**向下短扫掠**（胶囊/球扫，不是单射线），距离 `probe_dist = snap_dist + skin_width`。
- 命中后按**接触法线与 up 的夹角**分类：
  - `angle <= slope_limit`：**可行走地面（Walkable）**→ grounded = true。
  - `slope_limit < angle < wall_limit`：**陡坡**→ 视为不可站立，沿坡下滑。
  - `angle >= wall_limit`：**墙**→ 不参与地面判定。
- 记录 `ground_normal / ground_point / ground_entity / slope_deg`，供移动投影、动画倾斜、移动平台继承使用。

### 7.2 地面吸附（Ground Snapping）
- 走下斜坡/台阶时，若脚下 `snap_dist` 内有可行走面，则**吸附贴地**（下拉到地面），避免"下坡起飞/抖动"。对标 Godot `floor_snap_length`、Rapier `snap_to_ground`。
- 吸附仅在"刚才在地面且未主动跳跃"时生效；跳跃当帧禁用吸附（否则跳不起来）。

### 7.3 台阶上行（Step Up）
- 水平被矮台阶挡住时：尝试抬升 `step_offset`（如 0.4m）→ 前移 → 下探回落到台面。三步成功则接受，否则回滚（对标 CMC `StepUp`、Unity `stepOffset`）。
- 下台阶靠地面吸附（§7.2）完成，无需显式逻辑。

### 7.4 边缘与悬崖（Ledge）
- 脚下部分悬空（接触点集中在胶囊一侧）→ 判定 `on_ledge`，可触发：边缘滑落、边缘抓取（Mantle 预备，见 §11）、或平衡动画。
- 用多点探地（胶囊底环采样）判断支撑面覆盖率。

### 7.5 斜坡滑落
- 陡坡（> slope_limit）施加沿坡向下的加速度，水平输入在陡坡上被削减切向分量，模拟站不稳（对标 CMC `bMaintainHorizontalGroundVelocity` 的反面）。

---

## 8. 移动模式状态机（MovementMode）

对标 UE CMC `EMovementMode` / Mover 2.0 可插拔 `MovementMode`，但用"枚举 + trait 对象/System 分派"实现，可数据驱动扩展。

### 8.1 内置模式
| 模式 | 行为要点 | 进入/退出条件 |
|---|---|---|
| `Walking` | 贴地移动、台阶、吸附、斜坡 | grounded 且非游泳 |
| `Falling` | 重力积分、空中控制、落地检测 | 失去地面或主动跳 |
| `Swimming` | 浮力、水阻、三维移动、出入水面 | 进入水体体积 |
| `Flying` | 无重力自由移动（创意/载具/能力） | 能力/debug 开启 |
| `Crouching`(修饰) | 半高切换，叠加于 Walking/Falling | 蹲下输入 + 可站起检测 |
| `Custom(id)` | 用户自定义（攀爬/蹬墙/滑铲，见 §11） | 由能力驱动进入 |

### 8.2 模式接口（可插拔）
```rust
trait MovementModeLogic {
    fn tick(&self, ctx: &mut MoveContext) -> ModeTransition; // 推进一子步，返回是否转移
    fn on_enter(&self, ctx: &mut MoveContext);
    fn on_exit(&self, ctx: &mut MoveContext);
}
enum ModeTransition { Stay, To(MovementMode) }
```
- `MoveContext` 聚合：运动学状态、地面状态、输入意图、tuning、查询接口、事件缓冲、`dt`。
- 转移由当前模式返回，集中在状态机驱动（避免模式间互相耦合）。观察者广播 `OnMovementModeChanged`。

### 8.3 模式注册表
- `MovementModeRegistry`：`id -> Box<dyn MovementModeLogic>`，支持 Game Feature 运行时插拔自定义模式（对标 gameplay §26 模块化玩法）。

---

## 9. 运动学 / 动力学 / 混合三档

同一套 CCT API 覆盖三种后端策略，按项目/角色需求切换（编译 feature + 运行时组件选择）：

### 9.1 运动学档（Kinematic，默认）
- 角色不进刚体求解；用 Collide-and-Slide 直接驱动 `Transform`。精确、可预测、易回滚。
- 与动态体交互：主动施加推力（§13.1），自身不被物理"随机"推动。
- 适用：主角、精确平台跳跃、竞技射击。对标 UE CMC / Jolt CharacterVirtual。

### 9.2 动力学档（Dynamic / Float-Ride）
- 角色是动力学刚体，脚下用**向下射线 + 弹簧-阻尼悬挂**（float ride）悬浮在地面之上，水平移动用速度/力驱动。对标 **bevy_tnua**。
- 优点：天然处理动态地面、斜坡弹性、被物理世界自然推挤、载具交互真实。
- 缺点：手感更"软"，预测回滚更难（需同步刚体状态）。
- 适用：物理玩法重的游戏、布娃娃过渡自然的场景。

### 9.3 混合档（Hybrid）
- 运动学求解主控位置，但在**与动态体接触**时切入冲量交换：角色对盒子施加冲量，盒子对角色的反作用按配置的"可被推动系数"回馈（0=纯运动学，1=接近动力学）。
- 布娃娃过渡：受击/死亡时从 CCT 平滑混合到 ragdoll（物理接管），对标 Euphoria 式过渡的轻量版（纯程序化混合，无 ML）。

档位对比：

| 维度 | 运动学 | 动力学(float) | 混合 |
|---|---|---|---|
| 精确可控 | ★★★ | ★★ | ★★★ |
| 动态交互真实 | ★ | ★★★ | ★★ |
| 预测回滚难度 | 低 | 高 | 中 |
| 手感可调性 | ★★★ | ★★ | ★★★ |
| 推荐用途 | 主角/竞技 | 物理玩法 | 混合体验 |

---

## 10. 手感工程层（Game Feel）

把"玄学手感"显式化为可序列化、可热重载、可按角色/状态切换的数据字段。对标 Celeste/Doom 的工程化手感。所有参数属于 `MovementProfile`（§19）。

### 10.1 加速度与减速（地面/空中分离）
- 不直接设速度，而是朝目标速度**插值/加力**：
```
target = move_dir * max_speed * analog_magnitude
accel  = if grounded { ground_accel } else { air_accel }
decel  = if grounded { ground_decel } else { air_decel }
velocity = move_towards(velocity, target, (if accelerating {accel} else {decel}) * dt)
```
- 支持曲线资产（`Curve<f32>`）描述"输入强度→目标速度"（手柄摇杆死区与非线性）。
- **转向插值**：高速反向时限制转向角速度（惯性/漂移感），或立即转向（街机感），由 `turn_rate` 控制。

### 10.2 跳跃手感
- **变高跳（Variable Jump Height）**：按住跳跃键期间减弱重力/维持上升，松开立即进入正常重力（短按矮跳、长按高跳）。对标 Mario/Celeste。
- **土狼时间（Coyote Time）**：离开地面后 `coyote_time`（如 0.1s）内仍可起跳，容错边缘误差。
- **跳跃缓冲（Jump Buffer）**：落地前 `jump_buffer`（如 0.1s）内按下的跳跃在落地瞬间触发，容错提前输入。
- **二段跳/多段跳**：`max_jumps`，空中剩余次数计数；可配置每段高度衰减。
- **顶头检测**：上升中头顶撞墙立即清零垂直速度（避免"粘天花板"）。

### 10.3 空中控制
- `air_control`（0..1）缩放空中水平加速；`air_control_boost` 低速时增强（起跳初段更灵活）。对标 CMC `AirControl`。
- 保留动量：空中不强制清零水平速度，支持 bunny-hop/连跳动量链（可配置是否允许，竞技/休闲不同取向）。

### 10.4 冲刺与瞬发（Dash / Impulse）
- `dash`：短时间定向高速（可无视重力），带冷却、缓冲、落地刷新次数（对标 Celeste dash）。
- `impulse`：外部一次性冲量（爆炸击退、弹簧板、技能位移），与移动速度按规则合并（叠加/覆盖/衰减）。

### 10.5 微手感细节
- **着陆缓冲**：落地按下落速度触发压缩动画/相机下沉/减速帧（对标 Doom/Overgrowth）。
- **贴墙微调（Wall Snap）**：贴墙移动时轻微吸附，避免抖动。
- **斜坡增速**：下坡保留/增益速度（滑行感），上坡轻微减速。
- **停止摩擦**：无输入时按 `ground_decel` 平滑停住，可配"急停帧"（街机）或"滑步"（写实）。

手感参数一览（节选，均在 Profile 中）：
```
max_walk_speed, max_sprint_speed, max_crouch_speed, max_air_speed
ground_accel, ground_decel, air_accel, air_decel, turn_rate
gravity_scale, jump_impulse, jump_hold_gravity_scale, max_jumps
coyote_time, jump_buffer, air_control, air_control_boost
dash_speed, dash_duration, dash_cooldown, dash_count
slope_limit_deg, step_offset, snap_dist, skin_width
```

---

## 11. 高级移动能力（Mantle / Vault / Climb / Wallrun / Slide / Dash）

对标 Naughty Dog traversal、Respawn 机动、Assassin's Creed 跑酷。能力以 `Custom(MovementMode)` + 动画耦合实现，由感知探测触发。

### 11.1 Mantle（翻越边缘/爬上去）
- 失足或面向矮墙时，向前上方探测可站立台面：找到后进入 `Mantle` 模式，用 **Motion Warping**（§12.2）把攀爬动画的根运动对齐到实际边缘点，位置由动画驱动，期间禁用常规碰撞响应。
- 分级：低矮（跨步）、中（撑手翻越）、高（跳抓+引体）——按高度选择动画集。

### 11.2 Vault（翻越障碍穿过去）
- 对薄障碍（栏杆/箱子）做"前方有障碍、障碍后方有落脚点"探测，播放翻越并以 warping 对齐，保留部分前向动量（Titanfall 式流畅）。

### 11.3 Climb（攀爬墙面/岩点）
- 自由攀爬：在可攀爬表面（标签/材质）上，输入映射为沿墙面切平面移动，用贴墙扫掠约束在表面；边缘/拐角做转移探测。
- 岩点攀爬：离散抓点图（Point graph），在点间插值移动 + IK 对齐手脚（对标 Uncharted）。

### 11.4 Wallrun / Wall-jump（蹬墙跑/墙跳）
- 侧向贴墙且速度足够时进入 `Wallrun`：沿墙切向移动，重力按曲线衰减，限时；墙跳给出离墙 + 向上冲量，保留动量链（对标 Titanfall/Apex）。

### 11.5 Slide（滑铲）
- 冲刺中蹲下进入 `Slide`：切半高、低摩擦、沿地面动量滑行，下坡加速、平地衰减；可衔接 slide-jump 保留速度（Apex 式）。

### 11.6 能力统一框架
```
trait TraversalAbility {
    fn detect(&self, ctx: &MoveContext) -> Option<TraversalPlan>; // 探测可行性 + 目标
    fn drive(&self, ctx: &mut MoveContext, plan: &TraversalPlan) -> AbilityState; // 位移/动画驱动
}
```
- 探测与驱动分离；探测可降频（非每帧），驱动在能力激活期每子步执行。
- 与 Gameplay Ability System（gameplay §9）打通：能力可消耗资源、受 Tag 阻断、带冷却。

---

## 12. 动画耦合：Root Motion / Motion Warping / Motion Matching / IK

角色控制器与动画是双向耦合：移动态驱动动画选择，动画（根运动）也可反向驱动位移。对标 `prism_animation_engine_design_zh.md`。

### 12.1 Root Motion 驱动位移
- 动画根骨骼位移/旋转作为**期望位移**喂给 collide-and-slide（而非直接改 Transform），保证根运动也走碰撞响应、不穿墙。对标 CMC `bUseRootMotion`。
- 模式切换：`AnimationDriven`（根运动主导，近战/过场精确）vs `CapsuleDriven`（输入主导，自由移动）。可按状态混合（根运动的水平 + 输入的转向）。

### 12.2 Motion Warping（运动变形）
- 把动画根运动按运行时目标（边缘点/敌人位置/落点）做**线性/旋转变形**，使通用动画精确对齐任意几何。用于 Mantle/Vault/攀爬/近战位移。对标 UE Motion Warping。
- Warp 窗口（动画时间区间）内分配位置/朝向误差，窗口外保持原始根运动。

### 12.3 Motion Matching 协同
- 若动画层启用 Motion Matching，CCT 提供"未来轨迹（trajectory）"预测：由当前速度 + 输入意图 + 手感曲线外推未来 N 帧位置/朝向，供 MM 选帧。对标 Naughty Dog/UE5 MM。
- CCT 的实际位移与 MM 选中的动画根运动做"误差消解"（warp 或软跟随），避免滑步（foot sliding）。

### 12.4 足部 IK 与倾斜对齐
- 地面法线/高度采样 → 两足 IK 贴合台阶/斜坡，髋部按低足下沉；身体按地面法线轻微倾斜（对标 Overgrowth）。
- 落地缓冲、重心偏移由 IK 层叠加，CCT 仅提供地面数据（法线/高度/脚下探测结果）。

### 12.5 滑步消除（Foot Sliding）
- 速度与动画播放速率匹配（距离-速度映射），或用 MM；最后用足锁定 IK（foot lock）在支撑相锁住接触点。

---

## 13. 物理耦合：移动平台 / 推力 / 被推 / 载具

### 13.1 推动动态体
- 运动学角色扫掠命中动态刚体时，按相对速度与可配置 `push_force` 施加冲量；重物推不动（按质量阈值），对标 Jolt `OnContactAdded` 推力回调、Rapier KCC `apply_impulses_to_dynamic_bodies`。
- 推力沿接触法线，避免"铲飞"；可选只推不被推（运动学档）。

### 13.2 被动态体承载（移动平台 / 电梯 / 载具顶）
- 站在运动学/动画平台上：记录 `ground_entity` 作为**基座（base）**，每帧继承基座的位移与旋转（含绕轴自转带动角色公转），对标 CMC `MovementBase` / Godot platform floor。
- 平滑进出：离开平台时继承基座速度作为初速（跳下电梯带惯性）。

### 13.3 被推挤与解穿退让
- 动态体压入角色时，解穿（§5.2）按逆质量分配：轻角色被重物推开，混合档下角色也可对物体退让。
- 防挤压死亡：被两个几何夹击且无处可去时，触发 `Crush` 事件交给 gameplay 处理（扣血/传送），而非无限解穿抖动。

### 13.4 水与流体
- 进入水体体积切 `Swimming`：浮力（按浸没比例）、水阻、出入水面检测、水下重力缩放；与物理流体（physics §8 全物态耦合）可选对接，默认用轻量水体体积近似。

### 13.5 载具附着
- 角色可"附身/附着"到载具 Pawn：CCT 挂起，位置由载具座位 socket 驱动；下车时恢复 CCT 并继承载具速度。

---

## 14. 相机耦合与视角

- 复用 Prism 相机模块（`prism_render_scene` 相机 / 相机控制器，设计推进中）作为相机后端，CCT 提供移动态供相机策略消费。
- **第三人称**：轨道相机跟随，带碰撞回弹（相机被墙遮挡时拉近，spring arm，对标 UE Spring Arm）、速度相关 FOV、落地下沉、转向滞后。
- **第一人称**：相机锚定头骨/眼点，含头部摆动（head bob，可关）、着陆冲击、瞄准时降速。
- **相机-移动协同**：相对相机的移动方向映射（WASD 相对相机）、冲刺拉 FOV、滑铲压低机位（对标 Apex）、墙跑倾斜（camera roll）。
- 相机独立于固定步长，在渲染帧插值平滑（见 §16.4 插值），避免固定步长抖动传入画面。

---

## 15. 输入接入（Enhanced Input 对标）

对标 gameplay §10 增强输入系统。CCT 只消费**抽象意图**，不直接读硬件。
- 输入动作（InputAction）→ `CcInputIntent`：`move_dir`(Vec2/相对相机)、`look`、`want_jump`(含按下/松开边沿)、`want_crouch`、`want_sprint`、`want_dash`、能力触发。
- 支持 InputContext 栈：UI 打开时屏蔽移动、载具/攀爬切换不同映射。
- 玩家与 AI 共用意图接口：AIController 产出相同 `MovementIntent`（gameplay §7.3），复用同一 CCT 执行路径（无分叉逻辑）。
- 跳跃等边沿事件缓冲到固定步长消费（避免固定步长丢输入），与 jump buffer（§10.2）配合。

---

## 16. 与网络框架的接缝（预测/回滚由网络文档负责）

> 网络复制、客户端预测、权威和解（rollback）、延迟补偿、确定性对账的**完整机制归 `prism_network_design_zh.md`**（§7 复制模型、§9 命令帧定序、§10 预测与和解、§11 延迟补偿、§13/§21 确定性）。本节不重复规定网络协议或和解算法，只定义**角色控制器为满足那套机制必须暴露的接缝**。CCT 不自带网络栈、不发明网络状态格式。

### 16.1 CCT 为网络层提供的契约
- **纯函数、可重放的推进**：`tick(state, input, env, dt)` 无副作用、确定性，使网络层的"回滚到权威态 → 重放本地输入序列"可反复调用（对标网络文档 §10 reconciliation、§13 确定性重放）。这是 Layer 2 函数式内核的核心收益。
- **瞬时态即快照字段**：CCT 的 `position/velocity/mode/feel/ground` 属于**高频瞬时仿真态**，走网络文档 §7 的"状态快照通道"，本身不进 persist 事务日志（与网络文档"瞬时态与持久态分离"立场一致）。
- **输入即命令帧负载**：`CcInputIntent`（含跳跃边沿）按固定步长打包进网络文档 §9 的 command frame；CCT 消费的是已定序的输入，不关心传输。
- **历史快照供延迟补偿**：CCT 维护带 tick 的历史胶囊（position + half_height），供网络文档 §11 lag compensation 回溯命中判定调用;CCT 只提供数据,回溯策略归网络层。
- **合法性包络供反作弊**：CCT 可由同一份 Profile 推导"单 tick 最大位移/最大速度/可达状态"包络,供网络权威侧与 `prism_anticheat_design_zh.md` 做约束校验;确定性求解使该包络可精确计算。

### 16.2 CCT 自身只负责的部分（不属于网络层）
- **固定步长与渲染插值**：求解跑在固定 tick（`prism_time` §4D），渲染帧对上一/当前 fixed 态做位置/朝向插值(§24)。这是本地平滑,与网络无关,单机也需要。
- **数值后端选择**：f32 / 定点路径由 CCT 编译期切换(§17);网络层据此决定是否启用严格确定性对账。

### 16.3 边界约定
- CCT **不**实现:传输、snapshot delta 编解码、AOI、权威移交、和解触发时机——这些全在网络文档。
- 网络层**不**侵入:CCT 的求解算法、手感、模式机——它只调用 `tick` 并读写状态快照。
- 关闭网络 feature 时,CCT 完全不含一行网络代码(对标网络文档"零侵入热路径")。

---


## 17. 确定性

- 求解器纯函数：`tick(state, input, env, dt)` 不读全局可变状态、不依赖迭代顺序外的随机性。
- **数值路径**：默认 f32 高性能（单机/宽松联机）；可选定点/软浮点（`fixed`）确定性路径，编译期切换，服务严格联机与跨平台回放。
- 查询确定性：几何查询（sweep/overlap）结果需稳定排序（命中按距离+实体 id 定序），避免同输入不同序。
- 固定步长 + 有序系统调度：移动相关 System 固定执行序，避免并行非确定性影响和解。
- 录像回放：记录每 tick 输入即可完整重演（对标 gameplay §34）。
- **分工边界**：CCT 只保证「求解器本身确定性 + 数值后端可切换」；严格联机的跨机对账、回放日志、CRDT/定序由 `prism_network_design_zh.md`（§13/§21）负责，本节不重复。

---

## 18. 性能工程与帧预算

### 18.1 批量并行
- 无接触耦合的角色之间互相独立 → 按实体批量并行跑 collide-and-slide（`prism_tasks` fiber job / `prism_ecs` 列式并行迭代）。
- 有互推接触的角色分岛（island）串内并行、岛间并行，保确定性可控（对标 physics §Island）。

### 18.2 查询成本控制
- **广相预筛**：对角色周围做一次 AABB 查询收集候选碰撞体，后续子步/bounce 复用候选集，避免每次全局 BVH 查询。
- **接触缓存 / warm start**：跨子步缓存接触面，减少重复扫掠。
- **探地降频**：稳定站立时探地可降频（每 N tick 全查，其间轻量校验）。

### 18.3 LOD 与人群
- 远处/非关键角色降 CCT 保真：简化为"射线判地 + 简单滑动"，甚至纯动画位移（对标 gameplay §27 Mass 人群）。
- 分级：L0 全保真（主角）、L1 简化扫掠、L2 射线+胶囊近似、L3 纯运动学脚本/位置流。
- 休眠：静止且无输入的角色挂起求解（sleep），被扰动唤醒。

### 18.4 帧预算（对标 gameplay §47）
- 典型目标：单主角 CCT < 0.1ms；百级 NPC（L1/L2）合计 < 1ms（并行后）。
- 子步/bounce 上限保护最坏情况；高速角色的子步数设上限并降级到 CCD。
- 内存：热状态 SoA、冷 tuning 共享 `Handle<MovementProfile>`，减少 cache miss。

### 18.5 SIMD
- 批量扫掠的 AABB 预筛、法线投影、向量运算走 glam SIMD；候选集按 SoA 布局便于向量化。

---

## 19. 数据驱动与可配置（Movement Profile）

- `MovementProfile`（`Asset` + `Reflect`）：聚合全部手感/几何/模式参数（§10 列表 + slope/step/skin/各模式开关），可序列化（RON/JSON）、热重载、编辑器可视化调参。
- 分层覆盖：全局默认 → 角色类型 → 实例 → 状态临时覆盖（如受击减速 buff 改 profile 字段）。
- 运行时切换：不同角色（重装/轻甲/载具）引用不同 profile；同角色不同态（陆/水/攀爬）切 profile 子集。
- 作者层：Prefab/Scene 中以组件引用 profile；编辑器提供曲线编辑（加速曲线、FOV 曲线）与实时预览。
- 校验：加载时校验参数合法区间（slope_limit∈[0,90]、skin_width>0 等），非法回退默认并告警。

---

## 20. 调试与可观测性

对标 gameplay §38 Gameplay Debugger / Visual Logger。
- **形状可视化**：绘制胶囊、探地扫掠、命中法线、台阶抬升尝试、滑动投影向量、犄角裁剪平面（gizmos）。
- **状态 HUD**：当前模式、速度（水平/垂直）、grounded、slope、coyote/buffer 计时、空中时间、子步/bounce 次数。
- **时间轴记录器**：录制每 tick 的输入与状态，支持逐帧回看、导出（排查手感/网络和解问题）。
- **网络调试**：预测与权威状态偏差曲线、回滚次数/重放帧数、和解纠偏可视化。
- **实时调参**：运行中热改 Movement Profile 字段并即时生效（inspector/控制台）。
- **自动化测试探针**：导出求解器输入/输出做回归（§27）。

---

## 21. 易用性分层与默认体验

借鉴 Bevy modular 理念、Godot `move_and_slide` 的"零配置能跑"（均为外部设计参考，非依赖）。

- **L0 一行可用**：插入 `CharacterControllerBundle::default()` → 立即获得走/跑/跳/斜坡/台阶/贴地（默认 Profile + 运动学档 + Walking/Falling 模式）。
```rust
commands.spawn((
    CharacterControllerBundle::default(),
    Transform::from_xyz(0.0, 1.0, 0.0),
));
```
- **L1 换 Profile**：`CcTuning(assets.load("tuning/hero.ccprofile.ron"))` 调手感，无需碰代码。
- **L2 加能力**：附加 `Mantle`/`Slide`/`Wallrun` 组件启用对应 traversal 能力。
- **L3 换档/换模式集**：选择动力学/混合档，或注册自定义 `MovementMode`。
- **L4 自定义内核**：实现 `ShapeQuery` 换物理后端，或替换 collide-and-slide 策略（trait 注入）。

默认体验基线（开箱即得）：相对相机移动、跳跃含 coyote/buffer/变高、斜坡滑落、台阶上行、地面吸附、移动平台继承、落地事件——无需任何调参即达到可玩手感。

---

## 22. 公共 API 草案

```rust
// ---- Bundle（L0 默认体验）----
#[derive(Bundle)]
pub struct CharacterControllerBundle {
    pub kinematics: CcKinematics,     // 胶囊尺寸 + 速度 + up
    pub ground: CcGroundState,
    pub mode: CcMode,                 // 默认 Walking/Falling
    pub feel: CcFeel,
    pub intent: CcInputIntent,
    pub tuning: CcTuning,             // Handle<MovementProfile>（默认内置）
    pub backend: CcBackend,           // Kinematic | Dynamic | Hybrid
}

// ---- 组件 ----
#[derive(Component)]
pub struct CcKinematics { pub velocity: Vec3, pub up: Dir3, pub radius: f32, pub half_height: f32 }

#[derive(Component)]
pub struct CcGroundState {
    pub grounded: bool, pub ground_normal: Vec3, pub ground_point: Vec3,
    pub ground_entity: Option<Entity>, pub slope_deg: f32, pub on_ledge: bool,
}

#[derive(Component)]
pub struct CcInputIntent {
    pub move_dir: Vec2, pub look: Vec2,
    pub jump: ButtonEdge, pub crouch: bool, pub sprint: bool, pub dash: ButtonEdge,
}

// ---- 事件（观察者广播）----
#[derive(Event)] pub struct Landed { pub entity: Entity, pub impact_speed: f32 }
#[derive(Event)] pub struct SteppedUp { pub entity: Entity, pub height: f32 }
#[derive(Event)] pub struct WallHit { pub entity: Entity, pub normal: Vec3 }
#[derive(Event)] pub struct MovementModeChanged { pub entity: Entity, pub from: MovementMode, pub to: MovementMode }
#[derive(Event)] pub struct Crushed { pub entity: Entity }

// ---- 引擎无关内核（Layer 2，不依赖 prism_ecs，no_std + alloc）----
pub struct MoveState { pub position: Vec3, pub velocity: Vec3, pub mode: MovementMode, pub feel: FeelState, pub ground: GroundSample }
pub struct MoveInput { pub move_dir: Vec2, pub jump: bool, pub jump_held: bool, pub crouch: bool, pub sprint: bool, pub dash: bool }
pub struct MoveEnv<'a> { pub query: &'a dyn ShapeQuery, pub gravity: Vec3, pub tuning: &'a MovementProfile }

/// 纯函数式单子步推进：可在预测/回滚中反复调用。
pub fn tick(state: &MoveState, input: &MoveInput, env: &MoveEnv, dt: f32) -> (MoveState, SmallVec<MoveEvent>);

// ---- 查询适配 trait（Layer 1）----
pub trait ShapeQuery {
    fn sweep(&self, shape: &Shape, iso: Isometry3d, dir: Dir3, max_dist: f32, filter: QueryFilter) -> Option<SweepHit>;
    fn overlap(&self, shape: &Shape, iso: Isometry3d, filter: QueryFilter, out: &mut Vec<Contact>);
    fn raycast(&self, origin: Vec3, dir: Dir3, max_dist: f32, filter: QueryFilter) -> Option<RayHit>;
}

// ---- 插件 ----
pub struct PrismCharacterControllerPlugin;
// 注册组件/事件/固定步长 System（intent 采集 -> tick -> 写回 Transform -> 事件广播 -> 渲染插值）
```

---

## 23. Crate 拆分与落地形态

```
pkg/
  prism_cct_core    // Layer 1-3: 引擎无关角色内核（求解器/模式机/手感/能力），纯数据+trait，no_std+alloc，不依赖 prism_ecs，可独立复用
  prism_cct         // Layer 4-5: 引擎集成（prism_app/prism_ecs 插件、组件、System、事件、Profile 资产、输入桥接）
  prism_cct_debug   // 可视化 & 时间轴记录器 & 网络和解调试
```
- `prism_cct_core` 不依赖 prism_ecs（`no_std + alloc`，不依赖任何 bevy_* crate）：便于在专用服务器/其它引擎复用，并保证确定性求解可被独立测试。
- 几何查询由 `prism_physics_core` / `prism_physics_geometry`（物理）实现 `ShapeQuery`；也可由 Rapier 适配或测试桩实现。
- 与 gameplay crate 对接：`Character` bundle（gameplay §7.4）内部组合本 crate 的 `CharacterControllerBundle`。

---

## 24. 一帧内的数据流与时序

```
[输入阶段]      输入采集：硬件 -> InputAction -> CcInputIntent（含跳跃边沿缓冲）
[固定步长阶段 FixedTick / prism_time §4D]（可执行 0..N 次，取决于累积时间）
   1. 读 CcInputIntent + CcKinematics + CcGroundState + Profile
   2. 能力探测（Mantle/Vault/...，可降频）
   3. cct_core::tick(state, input, env, dt)：
        加速/重力 -> 期望位移 -> depenetrate -> collide&slide（水平/台阶/垂直）
        -> 探地/吸附 -> 模式转移 -> 手感计时推进 -> 产出事件
   4. 写回 CcKinematics/CcGroundState/CcMode；记录历史缓冲（网络）
   5. 物理耦合：对动态体施加推力/继承移动平台基座
   6. 广播事件（Landed/SteppedUp/WallHit/ModeChanged/Crushed）-> 观察者（音效/VFX/动画触发）
[帧更新阶段]    动画：读移动态选帧/Root Motion/Motion Matching 轨迹；足部 IK 采样地面
[呈现前阶段]    渲染插值：对 fixed 态插值到渲染帧；相机跟随与碰撞回弹
[网络]       发送带 tick 输入；收权威态 -> 和解（回滚+重放 tick 序列）
```
- 关键顺序约束：探地在滑动之后、模式转移之前；移动平台基座继承在求解之前读、位移写回之后应用；渲染插值永远读"上一+当前"两帧 fixed 态。

---

## 25. 路线图

- **M0 内核骨架**：胶囊 collide-and-slide + 解穿 + 探地；Walking/Falling；运动学档。单机走跑跳可玩。
- **M1 手感层**：加速曲线、coyote/buffer、变高跳、空中控制、斜坡/台阶/吸附、移动平台基座。
- **M2 物理耦合**：推动动态体、被推退让、水体 Swimming、载具附着；混合档。
- **M3 动画耦合**：Root Motion 位移、Motion Warping、足部 IK、滑步消除；Motion Matching 轨迹接口。
- **M4 高级能力**：Mantle/Vault/Climb/Wallrun/Slide/Dash 能力框架 + 与 GAS 打通。
- **M5 网络接缝**：对接 `prism_network_design_zh.md`——暴露可重放 `tick`、瞬时态快照字段、带 tick 历史胶囊、合法性包络；预测/回滚/Lag Comp 机制本身在网络文档落地。
- **M6 性能与人群**：批量并行、广相预筛、CCT LOD、休眠、SIMD。
- **M7 确定性 & 工具**：定点路径、录像回放、调试器/时间轴记录器、编辑器调参与曲线编辑。
- **M8 动力学档完善**：float-ride 后端、ragdoll 平滑过渡。

---

## 26. 关键扩展点清单

- `ShapeQuery`：接入任意物理后端（`prism_physics_*` / Rapier / 自研）。
- `MovementModeLogic` + `MovementModeRegistry`：注册自定义移动模式（飞行/磁吸/低重力）。
- `TraversalAbility`：注册自定义高级移动能力。
- `MovementProfile`：数据驱动手感；可被 Game Feature 热插拔覆盖。
- `tick` 的子阶段 hook：加速/重力/滑动/探地各阶段可注入自定义策略（trait 对象/回调）。
- 事件观察者：Landed/WallHit/ModeChanged 挂接音效/VFX/相机/动画，无侵入扩展。
- 数值后端：f32 / 定点编译期切换。

---

## 27. 落地与验收清单

功能验收：
- [ ] 斜坡：≤slope_limit 可行走，>slope_limit 下滑，不抖动。
- [ ] 台阶：≤step_offset 平滑上行，下台阶吸附不起飞。
- [ ] 边缘：悬崖边缘不穿、不卡；coyote 容错起跳成立。
- [ ] 跳跃：变高跳、jump buffer、coyote、二段跳、顶头清零均符合预期。
- [ ] 移动平台：站立继承平移/旋转，离开继承速度。
- [ ] 动态体：可推轻物、推不动重物、被重物挤压不抖且触发 Crush。
- [ ] 高速：高速移动不隧穿（子步/CCD）。
- [ ] 水体：进出水面切模式，浮力/水阻合理。
- [ ] 能力：Mantle/Vault/Slide/Wallrun 探测与驱动正确，warping 对齐无滑步。

性能验收：
- [ ] 单主角 CCT < 0.1ms；百级 NPC 合计 < 1ms（并行后，L1/L2 LOD）。
- [ ] 最坏情况子步/bounce 有上限，无死循环。

网络验收（机制归 `prism_network_design_zh.md`，此处只验 CCT 接缝）：
- [ ] 和解：200ms RTT 下预测平滑、纠偏不可感知抖动。
- [ ] 回滚重放确定性：相同输入序列重放得到相同状态。
- [ ] Lag Comp：移动目标命中判定公平（favor-the-shooter 可配）。

质量验收：
- [ ] 内核单元测试覆盖滑动/台阶/斜坡/解穿边界（用解析几何桩，不依赖物理后端）。
- [ ] 确定性测试：f32/定点路径回放一致性。
- [ ] Demo：平台跳跃关 + 竞技移动场 + traversal 跑酷场。

---

## 28. 术语表

| 术语 | 含义 |
|---|---|
| CCT / CCA | Character Controller / Character Controller Agent，角色控制器 |
| Collide-and-Slide | 扫掠命中后沿碰撞面切向滑动的移动求解范式 |
| Depenetration / MTV | 解穿；最小平移向量，把重叠形状推开 |
| Skin Width | 接触皮肤，表面前预留的微小间隙，防穿插抖动 |
| Sweep | 形状扫掠查询（胶囊/球沿方向推进求首个命中） |
| Ground Snapping | 地面吸附，下坡/下台阶时下拉贴地防起飞 |
| Step Offset | 可自动上行的最大台阶高度 |
| Slope Limit | 可行走的最大坡度 |
| Coyote Time | 离地后仍可起跳的容错时间 |
| Jump Buffer | 落地前预输入跳跃的缓冲时间 |
| Variable Jump | 变高跳，按住更高、短按更矮 |
| Air Control | 空中水平操控强度 |
| Root Motion | 由动画根骨骼位移驱动角色移动 |
| Motion Warping | 按运行时目标变形根运动以对齐几何 |
| Motion Matching | 按未来轨迹匹配动画帧的播放技术 |
| Float Ride | 动力学 CCT：射线+悬挂弹簧使刚体悬浮行走 |
| Base / MovementBase | 移动平台基座，被站立角色继承其运动 |
| Reconciliation | 网络和解：按权威状态回滚并重放本地输入 |
| Lag Compensation | 延迟补偿：按攻击者 RTT 回退目标历史位置判定 |
| CCD / TOI | 连续碰撞检测 / 碰撞时刻，防高速隧穿 |
| Mantle / Vault | 翻越上去 / 翻越穿过 的 traversal 能力 |

---

> 本设计与 `prism_physics_design_zh.md`（几何查询底座）、`prism_gameplay_design_zh.md`（§7 Character、§10 输入、§16/§33 网络、§37 手感、§41 动画）、`prism_animation_engine_design_zh.md`（Root Motion/Motion Matching/IK）协同落地；是 gameplay §7.4 "与 Prism Physics 角色控制器对接"的具体规格实现。
