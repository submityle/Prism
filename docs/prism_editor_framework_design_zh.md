# Prism Loom 编辑器框架设计方案（Loom Studio）
> v2 / 顶级次世代 AAA 级编辑器 — 高级特性增补 + 大世界编辑 + GPU 驱动视口 + 多人协同 + 分析诊断 + 域编辑器全景 + 模块化收口

> 面向 Prism(Bevy fork)的**模块化、数据驱动、运行时解耦、AAA 规模**的编辑器框架。
> 工作名 **Loom Studio**。以 Loom 声明式 UI/响应式内核为唯一 UI 地基。
> 借形态不抄码,系统性对标:
> **Unreal Editor 5**(World Partition / One-File-Per-Actor / Multi-User / Sequencer / Niagara / Insights / Live Coding / Gameplay Debugger)、
> **Unity**(反射 Inspector / Timeline / Shader Graph / Addressables / Profiler / Frame Debugger / DOTS 编辑)、
> **Frostbite·Anvil·Decima**(超大世界流送编辑)、
> **Omniverse**(USD 分层 / 实时协同)、**Houdini**(过程化节点图 / 非破坏式)、
> **Blender**(Operator 命令 / 非阻塞 modal / 几何节点)、
> **VS Code·Fleet·Rider**(贡献点 / 命令面板 / 前后端分离 / Tool Window)、
> **Figma·Zed**(GPU 驱动画布 / CRDT 协同 / 瓦片重绘)、
> **RenderDoc·PIX·Tracy**(帧捕获 / GPU 计时 / 性能回放)、**Perforce·Git**(大二进制资产版本控制)。
>
> 本文为**设计规格**。地基层(Loom 25 crate + 相关 Bevy crate)为**已交付**(SHIPPED);
> 编辑器专有层(E 系列)为**规划项**(PLANNED);**域编辑器**的成熟度受对应引擎子系统成熟度约束,文中显式标注。
> 全文严格区分「已实现并通过测试」与「规划中」,不把未落地能力描述为已落地。
>
> **非目标(与引擎文档一致)**:纯经典数值与确定性工具链,**不含 AI / ML / 神经网络 / LLM** 功能;
> 不以替代专业 DCC(Maya/Houdini/Substance)为目标,而以**无缝往返 + 引擎内权威编辑**为目标。

- 版本: v2.0(AAA 高级特性设计阶段;v1 内核/面板设计已立)
- 适用引擎: Prism / Bevy ECS 生态
- 关键地基(SHIPPED): `prism_ui*`(响应/树/布局/样式/后端/调度/虚拟化/输入)、`prism_ui_inspector`、
  `prism_ui_timetravel`、`prism_ui_workbench`、`prism_ui_hotreload`、`prism_ui_ecs`、`prism_ui_overlay`、
  `prism_ui_form`、`prism_ui_router`、`prism_ui_i18n`、`prism_ui_a11y`、`prism_ui_sdui`、
  `prism_ui_render_backend`;`bevy_reflect`、`bevy_remote`(BRP)、`bevy_picking`、`bevy_gizmos`、
  `bevy_scene`、`bevy_asset`、`bevy_state`、`bevy_diagnostic`、`bevy_dev_tools`、`bevy_input_focus`
- 域子系统(编辑器挂接对象,成熟度各异): 材质/WESL 管线、`Ember` 粒子、`Animation` 动画、
  `Resonance` 音频、Lumen GI、物理、地形/世界系统、虚拟几何/体积/毛发 GPU 孪生
- 核心契约: 继承 Loom「**成本 ∝ 变化量**」;编辑器一切编辑走**命令**,一切状态可**内省/回放**,一切长任务**异步可取消**。

---

## 目录
1. 设计哲学与核心契约(AAA 加固)
2. 顶级产品对标:借形态、取什么、不抄什么
3. 分层架构总览(含大世界 / 协同 / 诊断)
4. 编辑器内核 Editor Kernel
5. 编辑器外壳 Shell
6. 大世界编辑:World Partition / 数据层 / HLOD / OFPA / 层级实例
7. GPU 驱动视口:多视口 / GPU 拾取 / 虚拟几何预览 / 实时 GI 预览 / 高级 Gizmo
8. 域编辑器全景(挂接引擎子系统)
9. 分析与诊断:帧分析器 / 帧图调试 / 内存预算 / Gameplay Debugger / 可视日志
10. 实时迭代:热重载 / 实时着色器 / PIE 实时编辑 / Live Coding
11. 内容与资产管线:内容浏览器 / 依赖图 / 校验 / 构建 / 版本控制
12. Prefab / 变体 / 覆盖传播
13. 多人协同编辑(Multi-User)
14. 性能工程:成本 ∝ 变化量在 AAA 编辑器中的落地
15. 自动化 / 确定性 / 无头编辑器(CI)
16. 可达性 / DPI / HDR / 色彩管理
17. Crate 全景与状态矩阵
18. 与 Bevy / BSN / Loom / 子系统的关系
19. 路线图(E1–E12)
20. 风险与取舍
21. 术语表

---

## 1. 设计哲学与核心契约(AAA 加固)

在 v1 七条铁律基础上,为 AAA 规模追加三条:

- **D1 一切编辑皆命令**:任何文档修改都是可执行、可撤销、可序列化的 `Command`。UI 不直接改模型。
  撤销/重做、宏录制、协同同步、脚本自动化、CI 回放全部复用这一通路。
- **D2 反射驱动 UI**:面板由 `bevy_reflect` 类型信息**自动生成**检查器,自定义仅作覆盖。
- **D3 贡献点而非硬编码**:面板/命令/菜单/键位/Gizmo/Inspector 控件/importer/profiler 轨道都经注册表贡献。
- **D4 运行时解耦**:编辑器与运行时边界清晰、可跨进程(BRP);编辑态与播放态(PIE)互不污染。
- **D5 成本 ∝ 变化量**:继承 Loom 契约;大场景按脏传播,长列表虚拟化,编辑器**可测试性能**。
- **D6 可内省 / 可回放**:编辑器自身状态可快照、可时间旅行(`prism_ui_timetravel`)。
- **D7 失败可见 + 非阻塞**:错误走错误边界;导入/烘焙/编译/Cook 异步可取消,永不冻结 UI。
- **D8 规模无关(AAA 新增)**:十万级实体、GB 级资产、公里级世界下,编辑器交互延迟不随规模线性恶化;
  依赖流送、虚拟化、GPU 拾取、增量差分、层级实例。
- **D9 非破坏式 + 过程化(AAA 新增)**:优先非破坏式工作流(修改器栈 / 过程化节点图 / 覆盖层),
  源数据可回溯,借鉴 Houdini / Blender 修改器。
- **D10 往返无损(AAA 新增)**:与 DCC / 外部工具经稳定中间格式(glTF/USD 风格、`.bsn`/`.scn`)无损往返,
  引擎内编辑为权威,外部变更可增量合并。

### 核心契约

> **编辑器 = 对 Document 的命令流 + 对 Document 的响应式视图 + 对 Runtime 的解耦协议。**
> 命令改模型,模型发信号,信号驱动 Loom 视图产出最小增量;运行时经 BRP 单独演进。
> 撤销=命令逆重放;协同=命令广播;自动化=命令脚本;CI=命令回放+黄金断言。四者同机制。

---

## 2. 顶级产品对标:借形态、取什么、不抄什么

| 产品 | 值得学的高级能力 | Loom Studio 的取法 | 落点 |
|---|---|---|---|
| **Unreal 5** | World Partition 网格流送、One-File-Per-Actor、HLOD、Multi-User、Sequencer、Niagara、Insights、Live Coding、Gameplay Debugger、Nanite/Lumen 视口 | 大世界编辑(§6)、协同(§13)、序列器/VFX 域编辑(§8)、帧分析器(§9)、热重载/Live(§10) | §6–§10 |
| **Unity** | 反射 Inspector、Timeline、Shader Graph、Addressables、Profiler、Frame Debugger、Prefab 变体、DOTS 编辑 | 反射 Inspector(§4)、材质图/时间线(§8)、内容管线+打包(§11)、Prefab 变体(§12) | §4,§8,§11,§12 |
| **Frostbite / Anvil / Decima** | 超大开放世界流送编辑、数据层、分块协作 | World Partition 网格 + 数据层 + OFPA(§6) | §6 |
| **Omniverse** | USD 分层 / 组合、实时多人、Live Sync | 分层覆盖 + 协同(§12,§13);`.bsn`/`.scn` 分层 | §12,§13 |
| **Houdini** | 过程化节点图、非破坏式、属性驱动 | 过程化图编辑器 + 修改器栈(D9,§8) | §8 |
| **Blender** | Operator 命令、非阻塞 modal、几何节点、工作区 | 命令模型(§4)、工作区预设(§5)、几何节点(§8) | §4,§5,§8 |
| **VS Code / Fleet / Rider** | 贡献点、命令面板、前后端分离、Tool Window、Action | 贡献点+命令面板(§4,§5)、BRP 远端后端(§4-运行时) | §4,§5 |
| **Figma / Zed** | GPU 画布、CRDT 协同、瓦片重绘、无限画布 | GPU 保留绘制 + 瓦片/脏矩形(§7,§14)、CRDT(§13) | §7,§13,§14 |
| **RenderDoc / PIX / Tracy** | 帧捕获、GPU 计时、调用树、性能回放 | 帧图调试器 + Tracy 对接 + 帧分析器(§9) | §9 |
| **Perforce / Git-LFS** | 大二进制版本控制、细粒度 checkout、合并 | OFPA 细粒度 + 版本控制贡献点(§11) | §11 |

**不抄的教训**:即时模式 Inspector(状态难留)→ 保留模式 + 稳定 ID;单体耦合(改一处牵全身)→ 贡献点;
UI 与模型混写 → 命令强制单向;整文档撤销快照(大世界爆内存)→ 命令逆 + 分块;把 DCC 功能全塞进引擎 → 往返无损(D10)。

---

## 3. 分层架构总览

```
┌────────────────────────────────────────────────────────────────────────┐
│ L4 域编辑器 Domain Editors(贡献点,挂接引擎子系统)                          │
│   材质图 · VFX(Ember) · 动画/状态机/重定向 · 序列器 · 地形/世界分区 ·         │
│   音频混音(Resonance) · 物理编辑 · 水/布/毛发/体积 · 过程化几何节点            │
├────────────────────────────────────────────────────────────────────────┤
│ L3 通用面板 Panels(贡献点)                                                 │
│   Inspector · 场景大纲 · 内容浏览器 · 视口+Gizmo · 控制台 · 帧分析器 · 依赖图    │
├────────────────────────────────────────────────────────────────────────┤
│ L2 外壳 Shell:Dock 停靠 · 命令面板 · 菜单/工具栏/状态栏/活动栏 · 键位上下文      │
├────────────────────────────────────────────────────────────────────────┤
│ L1 内核 Kernel:Document/World · Command/Transaction/Undo · Selection ·      │
│   Reflection桥 · Service/Contribution 注册表 · 异步任务/诊断 · 分层/覆盖        │
├────────────────────────────────────────────────────────────────────────┤
│ L0 地基 Loom(SHIPPED,复用不改):reactive/tree/layout/style/render_backend/    │
│   scheduler/virtual/input/inspector/timetravel/workbench/hotreload/overlay/   │
│   form/router/i18n/a11y/sdui/ecs · bevy_reflect/remote/picking/gizmos/scene/   │
│   asset/state/diagnostic/dev_tools/input_focus                               │
└────────────────────────────────────────────────────────────────────────┘
        ↕ BRP(进程内快路径 / 跨进程 / PIE)        ↕ 协同通道(命令广播 / CRDT)
┌────────────────────────────────────────────────────────────────────────┐
│ 被编辑运行时 Runtime(Prism/Bevy app):编辑态 & 播放态(PIE) & 远端真机         │
└────────────────────────────────────────────────────────────────────────┘
```

每层一个或多个独立 crate,可单独编译/测试/演进。L0 完全复用,L1–L4 新建。

---

## 4. 编辑器内核 Editor Kernel

> crate: `prism_editor_core`(模型/命令/选择/服务/贡献/分层)、`prism_editor_reflect`(反射↔控件)

### 4.1 文档模型 Document / World / 分层
- `EditorDocument`:当前被编辑对象的统一句柄,封装一个 ECS `World`(或其 BRP 代理,§4.6)+ 编辑元数据。
- **分层组合(D10,借鉴 USD)**:文档是若干 `Layer` 的有序组合——基底层 + 覆盖层(关卡变体 / 用户本地覆盖 /
  运行时注入)。读为「组合结果」,写落到「当前编辑目标层」,支持覆盖高亮与「还原到基底」。
- 文档以 `Signal`/`Memo` 暴露派生视图,接入 Loom 同一张无毛刺依赖图;面板「只在相关数据变化时重算」。

### 4.2 命令与事务 Command / Transaction / Undo
```rust
pub trait Command: Send {
    fn apply(&mut self, cx: &mut EditCx) -> Result<(), EditError>;
    fn undo(&mut self, cx: &mut EditCx);
    fn label(&self) -> &str;
    fn merge(&mut self, next: &dyn Command) -> MergeResult; // 连续拖拽合并
    fn affected(&self) -> AffectSet;                        // 影响域→协同冲突检测/脏传播
}
```
- `CommandStack`:undo/redo 双栈 + 合并窗口;`Transaction` 原子打包(借鉴 Unreal `ScopedTransaction`)。
- **大世界撤销(D8)**:存「命令的逆」而非整快照,成本 ∝ 变化量;按**分块(cell)**作用域裁剪撤销影响。
- 与 `prism_ui_timetravel`(SHIPPED)协同:历史可视化、相邻帧 diff、分支。
- 命令是数据 → 协同广播(§13)、宏录制(Blender 式)、CI 回放(§15)共用。

### 4.3 选择集 Selection
- 有序去重的实体/资产/子对象(顶点/边/面/关键帧/节点端口)集合,`Signal` 暴露;主选 + 多选 + 软选择(§7)。
- Hierarchy ↔ Viewport ↔ 域编辑器三向高亮同步;选择可选入 undo(配置项)。

### 4.4 反射驱动 Inspector(D2)
> crate: `prism_editor_reflect`
- 输入 `bevy_reflect` `TypeInfo`/`ReflectRef`,输出 Loom `Element` 控件树;`WidgetResolver` 按类型/标注选控件。
- 字段读写统一经**命令**(`SetReflectField`,可撤销可合并);多选显示交集/差异。
- 覆盖:贡献 `InspectorOverride<T>`(Godot `EditorInspectorPlugin` / Unity `PropertyDrawer`)。
- 协作:`prism_ui_form`(校验/错误)、`prism_ui_ecs`(字段级双向绑定,空闲实体零成本)。

### 4.5 服务与贡献点(模块化心脏,D3)
- `ServiceRegistry`:类型键依赖注入(复用 `prism_ui_component::ContextMap` 思想),提供文档/命令/选择/资产/BRP/
  profiler/version-control 等服务,模块按需取用,互不硬依赖。
- `ContributionRegistry`:运行时可注册/注销(`Disposable`,VS Code 式),贡献点含:
  `commands`(id/标题/默认键位/`when` 上下文)、`panels`、`menus`/`toolbars`、`inspectors`、`gizmos`、
  `importers`/`exporters`、`profiler_tracks`、`viewport_overlays`、`graph_nodes`(过程化节点)。
- `EditorModule` trait:一个模块 = 一组贡献 + 生命周期;卸载时 `Disposables` 自动回收,防泄漏。

### 4.6 运行时协议(D4,详见随文各处)
- `prism_editor_remote` 封装 BRP(`bevy_remote`,SHIPPED,JSON-RPC 2.0),三拓扑同协议:**进程内快路径 / 跨进程 / PIE**。
- Inspector 直接吃 BRP 的反射数据,**本地无需链接游戏逻辑 crate**;变更回传经命令→BRP。

---

## 5. 编辑器外壳 Shell
> crate: `prism_editor_dock`、`prism_editor_shell`
- **Dock 停靠**:嵌套 split/tab/浮动窗 + 拖拽重排 + 布局持久化(`.loomlayout`)+ 多工作区预设
  (Blender/Unity workspaces)。Flexbox 排布(SHIPPED)、手势拖拽(`prism_ui_input`)、浮动(`prism_ui_overlay`)。
- **命令面板**:全局模糊搜索所有注册命令并执行,显示键位(贡献点红利)。
- **菜单/工具栏/状态栏/活动栏**:全部贡献点聚合,`when` 上下文控制可见/启用,不硬编码。
- **键位系统**:keymap + 上下文(焦点面板/编辑模式),冲突检测,多套预设;焦点复用 `bevy_input_focus`。
- **主题/i18n/a11y**:复用 Loom 样式 token/`@media`、`prism_ui_i18n`、`prism_ui_a11y`(SHIPPED)。

---

## 6. 大世界编辑:World Partition / 数据层 / HLOD / OFPA(D8)
> crate: `prism_editor_world`(规划);挂接地形/世界子系统

借鉴 Unreal World Partition / Frostbite 大世界:

- **网格流送编辑**:世界切成 cell 网格,编辑器按相机/编辑焦点**按需加载/卸载** cell,公里级世界不整载。
  视口显示已加载区 + 代理占位;对接世界系统的流送契约。
- **数据层(Data Layers)**:同一空间叠加逻辑层(昼/夜、剧情阶段、LOD 变体),可见性/编辑性独立开关。
- **HLOD**:编辑器触发/预览分层 LOD 代理烘焙,远景用代理、近景用全量。
- **One-File-Per-Actor(OFPA 等价)**:实体以**细粒度文件**落盘(或 `.bsn`/`.scn` 分片),
  使多人并行编辑/版本控制冲突面最小(§11、§13 的前提)。
- **层级实例(Level Instance / Prefab 关卡)**:子关卡作为可实例化、可就地编辑的单元;与 §12 Prefab 统一。
- **编辑作用域裁剪**:命令撤销/脏传播/协同冲突均按 cell 作用域裁剪(§4.2),保证规模无关延迟。

---

## 7. GPU 驱动视口(D5/D8)
> crate: `prism_editor_viewport`;挂接 `bevy_picking`/`bevy_gizmos`/虚拟几何 GPU

- **多视口**:透视 + 正交(顶/前/侧)+ 可同步相机 + 独立显示模式(线框/光照/光照复杂度/overdraw/LOD 着色)。
- **GPU 拾取(D8)**:十万实体下走 GPU id-buffer 拾取而非 CPU 射线逐个测,`bevy_picking`(SHIPPED)为基线,
  大场景扩展 GPU 路径;框选/刷选同理。
- **虚拟几何预览**:对接 `prism_virtual_geometry_gpu`(SHIPPED 实验孪生),编辑态预览细分几何不爆显存。
- **实时 GI 预览**:对接 Lumen GI 子系统,编辑移动光源/物体实时反馈间接光(可切「预览质量/最终质量」)。
- **高级 Gizmo(借鉴 Unreal/Blender)**:移动/旋转/缩放 + **吸附**(网格/顶点/表面/角度/增量)、
  **轴心模式**(中心/各自/游标)、**坐标系**(世界/局部/视图/自定义)、就地数值输入、测量尺、
  多物体公共轴心。交互产出变换**命令**(可撤销、拖拽合并)。绘制走 `bevy_gizmos`(SHIPPED)。
- **软选择(soft/proportional)**:衰减半径内的加权变换(Blender proportional edit)。
- **视口叠加贡献点**:导航网格、碰撞体、光照探针、流送网格、调试箭头等由模块贡献,可独立开关。
- **瓦片/脏矩形重绘(§14)**:画布只重绘变化区域(Figma 式),走 `prism_ui_render_backend` 保留绘制 + 批合并。

---

## 8. 域编辑器全景(挂接引擎子系统,D9)
> crate: `prism_editor_graph`(通用节点图)、`prism_editor_timeline`(通用时间线)、`prism_editor_domains`(各域装配)
> 说明:**域编辑器成熟度受对应子系统成熟度约束**;下表「挂接子系统」指编辑器编辑/预览的目标。

| 域编辑器 | 对标 | 挂接子系统 | 关键能力 |
|---|---|---|---|
| **材质/着色图** | Unity Shader Graph / UE Material | 材质·WESL 管线 | 节点图 + 实时预览球 + WESL 代码视图往返 + 参数化实例 |
| **VFX / 粒子** | UE Niagara / Unity VFX Graph | `Ember` 粒子 | 发射器/模块栈 + 曲线 + GPU 预览 + 时间轴洗刷 |
| **动画** | UE AnimBP / Unity Mecanim | `Animation` 动画 | 状态机 + 混合空间 + 重定向 + 曲线/通知轨 + 预览骨架 |
| **序列器 / 过场** | UE Sequencer / Unity Timeline | 动画 + 相机 + 音频 | 多轨 + 子序列 + 相机切换 + 关键帧曲线编辑 |
| **地形 / 世界分区** | UE Landscape / World Partition | 地形/世界系统 | 高度/法线雕刻 + 图层绘制 + 植被散布 + 分区网格(§6) |
| **音频混音** | Wwise / UE MetaSounds | `Resonance` 音频 | 总线/发送/效果链 + 空间/HRTF 预览 + 事件触发矩阵 |
| **物理编辑** | UE PhAT / Unity Physics | 物理 | 碰撞体/关节编辑 + 约束调试 + 实时模拟预览 + 可视化 |
| **水/布/毛发/体积** | 各家流体/布料 | 水/布/毛发/体积 GPU | 参数编辑 + 实时孪生预览 + 边界条件可视化 |
| **过程化几何节点** | Houdini / Blender 几何节点 | 网格/场景 | 非破坏式节点栈 + 属性驱动 + 可缓存重算 |

**通用节点图内核**(`prism_editor_graph`):端口/连线/框选/对齐/折叠分组/子图;大图**虚拟化 + 视口裁剪**;
连线即命令;渲染走保留绘制流。材质/VFX/动画状态机/行为树/几何节点**共用**此内核,只换节点贡献集(D3)。
**通用时间线内核**(`prism_editor_timeline`):多轨 + 关键帧 + 曲线编辑器(复用 `prism_ui_anim` 曲线/弹簧,SHIPPED)+
轨道虚拟化;序列器/动画/VFX 时间轴共用。

---

## 9. 分析与诊断(D7)
> crate: `prism_editor_profiler`;挂接 `bevy_diagnostic`/`bevy_dev_tools`/Tracy/帧图

借鉴 Unreal Insights / Unity Profiler / RenderDoc / Tracy:

- **帧分析器**:CPU/GPU 时间轴火焰图 + 调用树 + 统计数值(FPS/draw calls/三角数/内存),数据源 `bevy_diagnostic`
  (SHIPPED)+ Tracy span(见 `docs/profiling.md`,SHIPPED);轨道可由模块**贡献**(§4.5)。
- **帧图调试器(Frame Graph / 渲染调试,RenderDoc 式)**:对接 `prism_render_architecture`(SHIPPED)的帧图,
  列出 pass / 资源 / 依赖 / 读写,逐 pass 预览中间缓冲(颜色/深度/GBuffer/阴影);离线捕获可回放。
- **内存与预算(D8)**:分类内存(纹理/网格/音频/世界 cell)+ **预算阈值**,超预算在编辑器内高亮告警
  (借鉴主机平台 memory budget 守门)。
- **Gameplay Debugger**:运行态按实体叠加逻辑可视化(AI 状态/感知/路径/黑板),贡献点扩展类目。
- **可视日志(Visual Logger)**:时间轴记录「某时刻某实体画了什么调试图元」,可回放定位偶发 bug(UE 式)。
- **Stat HUD / 诊断叠层**:视口角标实时统计,复用 `bevy_dev_tools`(SHIPPED)。

---

## 10. 实时迭代(D7)
- **保状态热重载**:`.loom` 视图/样式、关卡数据、资产变更**就地热重载**不丢瞬态(滚动位/输入态/动画进度),
  复用 `prism_ui_hotreload`(SHIPPED,稳定 NodePath + 保状态 plan)。
- **实时着色器/材质**:WESL 着色器改动即重编译热替换,视口实时反馈(Unity/UE 式)。
- **PIE 实时编辑**:播放态下改组件/数值即时生效(经 BRP),停止回退编辑态(Unreal PIE)。
- **Live Coding(系统热更)**:系统逻辑的热替换交由引擎/构建侧(dylib 热加载,`bevy_dylib` 方向),编辑器提供触发与状态保全界面;**不含 AI 代码生成**。

---

## 11. 内容与资产管线(D7/D8/D10)
> crate: `prism_editor_content`;挂接 `bevy_asset`(SHIPPED)
- **内容浏览器**:缩略图网格(虚拟化,`prism_ui_virtual`)+ 集合(Collections)+ 标签/过滤/搜索 + 收藏/固定。
- **资产依赖/引用图**:谁引用谁、反向引用、循环检测;复用通用节点图内核(§8)+ `prism_ui_inspector` 的
  依赖图分析(SHIPPED)渲染。删除前提示引用者,避免悬空引用。
- **资产校验(Validation)**:可贡献校验规则(命名/预算/缺失引用/非法配置),保存/提交时门禁,报告可点击跳转。
- **导入/导出(importer/exporter 贡献点)**:glTF/USD 风格无损往返(D10),`.bsn`/`.scn` 场景格式;增量重导。
- **Cook / 构建管线 UI**:平台目标选择 + 增量 cook + 进度/日志 + 产物浏览;长任务异步可取消(D7)。
- **版本控制集成(贡献点)**:Git / Git-LFS / Perforce 适配;依赖 §6 的 OFPA 细粒度实现**最小冲突面**、
  per-asset checkout/lock、差异/合并入口。大二进制走 LFS/Perforce,文本化资产走正常 diff。

---

## 12. Prefab / 变体 / 覆盖传播(D9/D10)
> 挂接 §4.1 分层 + §6 层级实例
- **嵌套 Prefab + 变体**:Prefab 可嵌套;变体继承基底并记录覆盖(Unity Prefab Variant / UE Child Actor)。
- **覆盖追踪**:实例对基底的每处修改被显式记录,Inspector 高亮「已覆盖字段」,支持「还原」「应用到基底」「向下传播」。
- **分层组合**:复用 §4.1 Layer 组合语义,Prefab 覆盖 = 覆盖层;与 USD 组合/Omniverse 分层思路一致。

---

## 13. 多人协同编辑 Multi-User(可选层,PLANNED)
> crate: `prism_editor_collab`(可选,默认不启用)
借鉴 Unreal Multi-User Editor / Omniverse Live:
- **命令广播(基线)**:命令已是可序列化数据(D1)+ 携带影响域(`affected()`),天然适合广播/重放式协同;
  OFPA(§6)把冲突面降到单实体/单 cell。
- **CRDT 收敛(进阶)**:树结构与属性各用合适 CRDT,命令作意图、CRDT 作收敛层(Figma 式),解决自由并发。
- **在场与锁(presence/lock)**:显示他人选择/相机/编辑焦点;可选 per-cell / per-asset 软锁。
- **路线**:先交付「单人 + 命令录制回放」,协同作为后续增量,不阻塞主线。

---

## 14. 性能工程:成本 ∝ 变化量在 AAA 编辑器中落地(D5/D8)
钉成**可测试契约**(`RecordingBackend` 断言):

| 机制 | 作用 | 复用/新增 |
|---|---|---|
| 面板响应式派生 | 面板只在相关信号变化时重算,空闲面板零成本 | `prism_ui_reactive`(SHIPPED) |
| 命令式撤销 + cell 裁剪 | 存命令逆而非整快照,大世界撤销廉价 | 新增 `prism_editor_core` |
| 大纲/内容/日志/轨道虚拟化 | 十万节点只构建可见窗口 + overscan | `prism_ui_virtual`(SHIPPED) |
| keyed 最小协调(LIS) | 展开/重排/过滤最小 BackendOp | `prism_ui_tree`(SHIPPED) |
| 帧预算时间切片 | 大树重建不卡输入/动画,逼近预算让出 | `prism_ui_scheduler`(SHIPPED) |
| 增量布局 RelayoutBoundary | 局部变化不触发全局重排 | `prism_ui_layout`(SHIPPED) |
| 字段相等性守卫 | ECS 回写值未变不置脏,掐断振荡 | `prism_ui_ecs`(SHIPPED) |
| World Partition 流送 | 公里级世界不整载,延迟与规模解耦 | 新增 `prism_editor_world` |
| GPU 拾取 | 十万实体拾取不随数量线性恶化 | `bevy_picking` + GPU 扩展 |
| 视口瓦片/脏矩形重绘 | 画布只重绘变化区域 | 新增 `prism_editor_viewport` |
| GPU 保留绘制 + 批合并 | 面板与画布走保留绘制流 | `prism_ui_render_backend`(SHIPPED,GPU parity 待验证) |
| BRP 增量同步 | 只拉变化组件,不整表轮询 | 新增 `prism_editor_remote` |
| 异步任务编排 | 导入/烘焙/cook 不阻塞 UI | 新增 `prism_editor_core`(任务系统) |

**钉死的回归契约(规划)**:
- 「切换选择只重算 Inspector 子树,其余面板 0 新建操作」
- 「大纲滚动:仅可见窗口 ± overscan 被构建」
- 「连续拖拽 Gizmo:undo 栈仅增长 1(合并),BackendOp 流 ∝ 实际变换」
- 「空闲帧(无命令/无运行时变更):0 BackendOp」
- 「加载/卸载一个 cell:脏传播与撤销影响被裁剪在该 cell 作用域内」

**效果层**:面板转场/选中高亮/FLIP/停靠动画复用 `prism_ui_motion`/`anim`(SHIPPED),声明式一等能力,不手搓补间。

---

## 15. 自动化 / 确定性 / 无头编辑器(D6)
- **命令脚本/宏录制**:录制命令序列为可重放脚本(Blender 式),用于批处理与教学。
- **无头编辑器(CI)**:不开窗装配内核 + 面板的无渲染后端(Loom `RecordingBackend`),在 CI 跑
  「打开资产→执行命令→断言文档/操作流」的黄金测试;确定性(整数/排序,无浮点非确定)保证可复现。
- **自动化测试面板**:发现/运行/报告编辑器内自动化用例(UE Automation 式)。

---

## 16. 可达性 / DPI / HDR / 色彩管理
- **可达性**:复用 `prism_ui_a11y`(SHIPPED),焦点序 / 读屏语义 / 键盘全操作。
- **DPI / 多显示器**:每显示器缩放,拖拽跨屏;布局以逻辑像素描述。
- **HDR 显示 + 色彩管理**:视口支持 HDR 输出与显示变换(OCIO 风格配置),保证编辑所见=目标色域所得;
  纯数值配置,不含 AI 色彩处理。

---

## 17. Crate 全景与状态矩阵

| crate | 职责 | 状态 |
|---|---|---|
| `prism_editor_core` | Document/Layer、Command/Transaction/Undo、Selection、Service/Contribution、异步任务/诊断 | 🔜 规划 |
| `prism_editor_reflect` | `bevy_reflect` ↔ Loom 控件;WidgetResolver;字段命令 | 🔜 规划 |
| `prism_editor_dock` | 嵌套 split/tab/浮动停靠 + 持久化 + 工作区 | 🔜 规划 |
| `prism_editor_shell` | 外壳装配:活动栏/侧栏/命令面板/菜单/工具栏/状态栏/键位/主题 | 🔜 规划 |
| `prism_editor_panels` | Inspector / Outliner / Content / Console 等通用面板 | 🔜 规划 |
| `prism_editor_viewport` | 多视口、GPU 拾取、虚拟几何/GI 预览、高级 Gizmo、瓦片重绘 | 🔜 规划 |
| `prism_editor_world` | World Partition 流送、数据层、HLOD、OFPA、层级实例 | 🔜 规划 |
| `prism_editor_graph` | 通用节点图内核(材质/VFX/动画/行为树/几何节点) | 🔜 规划 |
| `prism_editor_timeline` | 通用时间线/序列器/关键帧曲线 | 🔜 规划 |
| `prism_editor_domains` | 各域编辑器装配(挂接子系统) | 🔜 规划 |
| `prism_editor_profiler` | 帧分析器、帧图调试、内存预算、Gameplay Debugger、可视日志 | 🔜 规划 |
| `prism_editor_content` | 内容浏览器、依赖图、校验、导入导出、cook/构建、版本控制 | 🔜 规划 |
| `prism_editor_remote` | BRP 客户端、PIE、增量同步、安全 | 🔜 规划 |
| `prism_editor_collab` | 命令广播 / CRDT 协同 / 在场锁(可选) | 🔜 规划(可选) |
| `prism_editor_app` | 二进制:装配模块、窗口、与 `bevy_app` 集成 | 🔜 规划 |
| — 复用地基 — | `prism_ui*` 全家桶 + `bevy_reflect/remote/picking/gizmos/scene/asset/state/diagnostic/dev_tools/input_focus` | ✅ 已交付 |

> 原则:每个新 crate 以「crate + 测试 + 文档」三件套闭环交付,不堆半成品;域编辑器成熟度受子系统约束;
> 未落地并本地提交前不计入「已胜出」。

---

## 18. 与 Bevy / BSN / Loom / 子系统的关系
- **Loom 是唯一 UI 地基**:全部界面用 `loom!` 构建,享受成本契约与工具链。
- **与 BSN 共存**:`.bsn`/`.scn` 作为场景资产格式(数据驱动、可分层),编辑器 UI 用 Loom;定位不冲突。
- **与 ECS 解耦**:经 BRP + 反射访问运行时 World,本地不链接游戏逻辑,崩溃隔离。
- **与子系统分工**:编辑器**不实现**渲染/物理/动画/音频算法,只提供其**权威编辑 + 预览 + 调试**界面;
  域编辑器随子系统成熟逐步点亮。
- **渐进采用**:可先做嵌入现有 app 的调试面板模块,逐步长成完整编辑器。
- **与运行时框架共生**:编辑产出的 `.bsn`/`.scn` 与 DataAsset 由 **Loom Runtime**(`prism_loom_runtime_framework_design_zh.md`)同格式加载,编辑—运行往返无损;运行态经 BRP 回连编辑器实现 PIE/远程调参。

---

## 19. 路线图(E1–E12)
- **E1 内核地基**:`prism_editor_core`——Document/Layer、Command/Undo(+cell 裁剪)、Selection、Service/Contribution、异步任务、Disposable。配套命令回归测试。
- **E2 反射 Inspector**:`prism_editor_reflect`——反射→控件、WidgetResolver、字段命令、覆盖机制、多选差异。
- **E3 外壳 + 停靠**:`prism_editor_dock` + `prism_editor_shell`——停靠/命令面板/菜单键位/主题/工作区持久化。
- **E4 核心面板**:`prism_editor_panels`——Inspector + Outliner(虚拟化)+ Console。打通「选择→检查→改字段→撤销」。
- **E5 运行时桥 + PIE**:`prism_editor_remote`——BRP 进程内快路径 + 增量同步 + PIE 隔离;文档接 World。
- **E6 GPU 视口 + Gizmo**:`prism_editor_viewport`——多视口、`bevy_picking` 拾取、高级 Gizmo(命令化)、相机/网格。
- **E7 内容与管线**:`prism_editor_content`——内容浏览器 + 依赖图 + 校验 + 导入导出 + 版本控制贡献点。
- **E8 大世界编辑**:`prism_editor_world`——World Partition 流送 + 数据层 + OFPA + 层级实例 + HLOD 预览。
- **E9 分析诊断**:`prism_editor_profiler`——帧分析器 + 帧图调试 + 内存预算 + Gameplay Debugger + 可视日志。
- **E10 通用图 + 时间线**:`prism_editor_graph` + `prism_editor_timeline`——材质图 + 序列器先行。
- **E11 域编辑器**:`prism_editor_domains`——VFX(Ember)/动画状态机/地形/音频混音/物理,随子系统点亮。
- **E12 协同 + 无头 CI**:`prism_editor_collab`(可选)+ 无头编辑器自动化/黄金测试。

**建议优先级**:E1→E2→E4 = 可用属性编辑器最短路径;E3 外壳并行;E5/E6 = 能动场景的分水岭;
E7/E8 = AAA 规模前提;E9/E10/E11 = 高级生产力;E12 为增量。

---

## 20. 风险与取舍
1. **反射覆盖面**:自定义/泛型/句柄类型覆盖决定 Inspector 自动化度;默认结构体回退 + 可贡献覆盖,保证「未知也能编辑」。
2. **命令逆运算完备性**:破坏性操作回退快照式撤销;回归钉死「apply→undo→redo == 原状」。
3. **BRP 延迟与一致性**:进程内快路径 + 增量订阅 + 乐观本地回显;跨进程仅在真机/隔离场景。
4. **大世界编辑复杂度**:流送/数据层/OFPA/HLOD 是独立大工程;先网格流送 + OFPA,HLOD/数据层增量。
5. **GPU 后端 parity**:`prism_ui_render_backend` wgpu 后端需真机对拍 CPU 参考后端后方可承诺;视口 GPU 拾取同理。
6. **域编辑器与子系统耦合度**:编辑器只做界面,算法在子系统;子系统未成熟则域编辑器为空壳,显式标注不冒进。
7. **协同正确性**:CRDT 收敛与命令意图的一致性是难点;先命令广播 + OFPA 降冲突,CRDT 作后续。
8. **版本控制大二进制**:LFS/Perforce 适配面大;先 Git-LFS + OFPA 文本化分片,Perforce 增量。
9. **范围蔓延**:覆盖面极大;严格按 E1→E12 分批,每批三件套闭环,不堆半成品。
10. **确定性守恒**:全链路整数/排序优先,避免浮点非确定,保证 CI 黄金测试可复现(D6)。

---

## 21. 术语表
- **贡献点(contribution point)**:模块向编辑器注册命令/面板/菜单/控件/轨道的声明式扩展位。
- **命令 / 事务 / 命令逆**:对文档的最小可撤销修改 / 原子打包 / 撤销所重放的反操作。
- **分层组合(layer composition)**:基底 + 覆盖层按序组合为编辑结果(USD 式),Prefab 覆盖是其特例。
- **World Partition**:世界按 cell 网格流送编辑,延迟与世界规模解耦。
- **数据层(Data Layers)**:同一空间叠加可独立开关的逻辑层。
- **HLOD**:分层 LOD 代理,远景用代理近景用全量。
- **OFPA(One-File-Per-Actor)**:实体细粒度落盘,最小化多人编辑/版本控制冲突面。
- **PIE(Play-In-Editor)**:编辑器内隔离运行游戏逻辑,编辑态不被污染。
- **BRP(Bevy Remote Protocol)**:基于 JSON-RPC 2.0 的运行时远程检查/变更协议(`bevy_remote`)。
- **GPU 拾取**:用 GPU id-buffer 做大规模场景拾取,延迟与实体数解耦。
- **帧图调试器**:列出渲染 pass/资源/依赖并逐 pass 预览中间缓冲(RenderDoc 式)。
- **Gameplay Debugger / 可视日志**:运行态逻辑可视化 / 时间轴记录调试图元可回放。
- **非破坏式(non-destructive)**:源数据保留、以修改器栈/节点图/覆盖层表达变换,可回溯(Houdini 式)。
- **Disposable**:贡献/订阅的生命周期句柄,卸载自动回收防泄漏(VS Code 式)。

---

> 本文为设计规格。L0 地基项均已实现并通过测试(Loom 25 crate / 600+ 测试,Clippy 零告警;
> 相关 Bevy crate 随引擎交付);编辑器专有层(E1–E12)与域编辑器在落地并本地提交前不计入「已胜出」,
> 且域编辑器成熟度显式受对应子系统成熟度约束。本框架为**纯经典数值 / 确定性**工具链,不含 AI/ML/LLM 功能。
