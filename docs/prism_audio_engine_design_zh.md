# Prism Resonance 次世代音频引擎设计方案

> 面向 Prism（Bevy fork）的数据导向、RT 安全、图编译式高性能音频引擎设计。
> 借鉴 UE5（MetaSounds / Submix / Quartz / Audio Modulation / HDR-Attenuation）、Unity（AudioMixer / DSP Graph / Spatializer & Ambisonic Decoder SDK / Snapshots）、Godot（AudioServer Bus/Effect/Stream / 频谱分析 / 麦克风捕获 / 程序化 Generator）、Wwise（Event/Container/State/Switch/RTPC / Interactive Music / HDR / Occlusion·Obstruction / Aux Sends / Profiler / SoundBank）、FMOD Studio（Event/Parameter/Snapshot / Bank 流式 / Transceiver）、Steam Audio（遮挡·衍射·透射·反射·烘焙·HRTF·Ambisonics）、Web Audio API（AudioNode 图）、微软 Windows Sonic / 索尼 Tempest 3D / 杜比 Atmos / Meta XR Audio（平台空间后端），取长补短。
> **空间与内容一等公民**：几何驱动空间化与 Event 驱动内容模型同为一等公民；程序化合成（Patch）、调制（Modulation）与母带合规（LUFS/True-Peak/HDR）三者贯通，可达顶级次世代 AAA 质量。
> 本文档为设计规格与落地实现的权威规范；采用纯经典 DSP 路线，不含任何 AI/ML 内容，不含任何 UE/Unity/Godot/Wwise/FMOD 源码或衍生代码。

- 版本: v0.97（**本版新增**：M1 效果族 `prism_audio_core` 新增 **音频变压器建模节点 `TransformerNode`/`TransformerParams`/`Transformer`（`nodes/effects/transformer`）**（借鉴音频变压器/铁芯输出变压器这一广为公开记录的电声与磁学物理思想[频率相关磁芯饱和(低频先饱和)+ 偶次谐波不对称 + 绕组/漏感高频谐振 + 变压器不传 DC]，但不抄任何源码或衍生代码、无任何 AI/ML）：**变压器染色节点**——对每通道施加一条逐样本交错的"频率加权饱和"链：`pre=低频搁架(+emphasis_db)` 先把低频抬入磁芯 -> `core=asym_tanh(drive*pre+bias)` 不对称 tanh 磁芯饱和 -> `post=低频搁架(-emphasis_db)` 逆搁架恢复频谱平衡 -> `wind=peaking(+winding_db@winding_hz)` 绕组/漏感谐振峰 -> `out=highpass(dc_block_hz)` 隔直高通 -> `y=(1-mix)*x+mix*(trim*out)`；**频率相关饱和的立足点**——进核前的低频搁架提升使等幅低频比高频以更高电平抵达非线性，故低频削顶更狠、谐波更多(铁芯特性)，核后逆搁架恢复音色平衡；**偶次谐波**——非对称 bias 使转移曲线偏置(`asym_tanh(pre)=(tanh(drive*pre+bias)-tanh(bias))/(drive*(1-tanh(bias)^2))`，归一到单位小信号增益且过原点，bias!=0 注入变压器铁芯特征的偶次谐波)；**磁滞记忆**——可选 hysteresis 深度把上一样本磁芯输出的一小部分回灌入核输入(钳于 `[0,MAX_HYSTERESIS=0.9]` 以保有界 tanh 稳定)；**绕组谐振**——顶端一倍频程的轻阻尼 peaking 峰("绕组环鸣")；**隔直**——变压器两绕组间不传静态电压，故输出端 ~10Hz 高通；**复用而非重复**(`# Relationship`)——四个二阶节(两低频搁架/peaking 绕组/隔直高通)全部复用本 crate 自有的 `BiquadCoeffs::design`(RBJ cookbook)产系数而不重述公式，但因搁架->饱和->搁架->峰->隔直链必须逐样本交错非线性，无法用整块 `Biquad::process_inplace`，故自写私有 Direct Form I (`DF1`) 逐通道状态 `Df1{x1,x2,y1,y2}` 逐样本步进；区别于 memoryless 定曲线 `saturation`(无频率加权)、区别于磁带 `tape`(memoryless tanh+wow/flutter+一极点高频衰减，无频率相关磁芯饱和/绕组谐振/隔直)、区别于加高次谐波的 `exciter`(谱向相反)、区别于反射折叠的 `wavefolder`——这四者无一同时提供"低频先饱和 + 偶次谐波 + 绕组谐振峰 + 隔直"；RT 契约——四节逐通道 `DF1` 状态 + 逐通道 hysteresis 一极点记忆均在 `new` 预分配，`voice` 零堆分配/无锁/不 panic、逐发射样本 `flush_denormal`、非有限输入置 0、`latency_frames=0`；参数 `TransformerParams{drive,bias,emphasis_db,emphasis_hz,hysteresis,winding_hz,winding_q,winding_db,dc_block_hz,output_trim_db,mix}`(Copy+Default 2.0/0.2/9dB/150Hz/0.2/8kHz/Q2/4dB/10Hz/0dB/mix1、serde `serialize` 门控、全字段钳入合法域 + 非有限归安全默认)；导出 `MAX_DRIVE=64`/`MAX_BIAS=3`/`MAX_EMPHASIS_DB=36`/`MAX_WINDING_DB=24`/`MAX_WINDING_Q=16`/`MAX_HYSTERESIS=0.9`/`MAX_TRIM_DB=24`/`MIN_DC_BLOCK_HZ=1`/`MAX_DC_BLOCK_HZ=60`(模块私有)、公开导出 `Transformer`/`TransformerNode`/`TransformerParams`、`Transformer::{new,channels,drive,bias,hysteresis,emphasis_db,voice,reset,set_*}`、`TransformerNode::{new(params,sample_rate,channels),channels,set_drive,set_bias,set_hysteresis,set_emphasis_db,set_emphasis_hz,set_winding_hz,set_winding_q,set_winding_db,set_dc_block_hz,set_output_trim_db,set_mix}`+impl `AudioNode`；effects/mod.rs 字母序 `pub mod transformer;` 置于 `tilt_eq` 后 `tremolo` 前 + 同序 re-export + catalogue 一条用 "--"；17 单测(latency=0、silence->silence、tone 全有限、非有限输入保持有限、零帧安全、mix=0 逐样本透传、极端参数不 panic 不泄漏、非有限参数归安全默认、setters 钳位拒非有限、**DC 输入被阻断(settled tail 均值趋零)**、**低频比高频更早饱和(80Hz 谐波/基波比 > 2000Hz 的 2 倍，goertzel 测)**、**bias 产生偶次谐波(biased 二次谐波 > symmetric 的 4 倍)**、**绕组谐振抬升其频带(+12dB 峰使 8kHz 电平 >1.5 倍)**、立体声对同输入相位相干、reset 清尾、+6dB trim 约翻倍电平、Vec 收集不 panic)+1 doctest 全绿；core 累计 **814 单测 + 40 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit(`transformer.rs` 1108 行 + effects `mod.rs` +11，commit `84354283f`，经全绿验证 + 逐行 CR)；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 源码或衍生代码）
- 版本历史: v0.96（**本版新增**：M1 效果族 `prism_audio_core` 新增 **旋转喇叭音箱节点 `LeslieNode`/`LeslieParams`（`nodes/effects/leslie`）**（借鉴旋转喇叭音箱这一广为公开记录的电声物理与经典键盘音色处理思想[Doppler 移频 + directional horn 幅度 tremolo + 双话筒反相成像 + 机械惯性 spin-up/spin-down]，但不抄任何源码或衍生代码，且 "Leslie" 仅用于指代该通用声学现象而非任何品牌实现）：**旋转喇叭音箱模拟节点**——把输入下混为单声道后经复用的 `LinkwitzRileyCrossover`(`crate::nodes::crossover`，Mono，默认 `DEFAULT_CROSSOVER_HZ=800`，构造时固定) 分成低频带（驱动沉重缓慢的低音转子 drum）与高频带（驱动轻快迅捷的高音号 horn）两路，各自经一只 `Rotor` 施加由单一转子相位累加器驱动的三耦合周期效应：**Doppler 音高抖动**（转子交替朝向/背离话筒，建模为被 `sin(phase)` 调制的分数延迟线，左右话筒反相 `delay_l=base+depth*sin`、`delay_r=base-depth*sin`，`read_frac` 镜像 `chorus` 的 floor+frac 线性插值 + `rem_euclid` 环绕、先读后写）、**幅度 tremolo**（号口交替正对/偏离话筒，建模为周期增益 `am_l=1+am*cos`、`am_r=1-am*cos`，Doppler 用 sin / AM 用 cos 自然构成 90° 相位关系）、**反相立体声成像**（两只虚拟话筒以相反相位听到旋转）；horn/drum 以相反方向旋转（`direction=+1`/`-1`）、各具不同标称转速（`HORN_SLOW_HZ=0.8`/`HORN_FAST_HZ=6.9`、`DRUM_SLOW_HZ=0.7`/`DRUM_FAST_HZ=6.5`）；**机械惯性**——每只转子经一极点平滑 `inertia_coeff(sec,sr)=1-exp(-1/(tau*sr))` 向目标转速逼近，轻号加速快(`HORN_ACCEL_SECONDS=1.0`/减速 `HORN_DECEL_SECONDS=0.6`)、重转子迟缓(`DRUM_ACCEL_SECONDS=3.2`/`DRUM_DECEL_SECONDS=4.0`)，使 Brake/Chorale/Tremolo 三档切换靠真实 spin-up/coast-down 斜坡而非瞬变；合成 L/R 后经 `stereo_spread` 向 mid 收拢混合(`wet=mid+spread*(leg-mid)`，spread=0 塌缩为单声道、spread=1 最宽)，再 `wet/dry` 混；**纯确定性**——每个调制皆由显式转子相位的确定性正弦驱动，相位 wrap 用 `next-TAU*floor(next/TAU)` 规避 std-only `f32::rem_euclid` 以保 no_std；**复用而非重复**（`# Relationship`）——频带拆分复用共享 `LinkwitzRileyCrossover` 而非私带滤波器组，Doppler 复用 `chorus`/`flanger` 的调制分数延迟思想但耦合相位锁定的 AM tremolo 与反相立体成像（二者皆无），区别于纯幅度域 `tremolo`（本节点亦移动音高与立体像）与纯音高 `vibrato`，亦区别于静态 panner（运动由含惯性的模拟机械旋转生成）；RT 契约——crossover、mono downmix 暂存、两 band 缓冲、两转子延迟线均在 `new` 预分配，`process` 仅读输入/推进转子相位累加器/写输出，逐存储与发射样本 `flush_denormal`、非有限输入置 0、零堆分配/无锁/不 panic、`latency_frames=0`（干声直达，调制短延迟作为效果一部分而非上报延迟）；参数 `LeslieParams{speed,crossover_hz,doppler_depth,am_depth,stereo_spread,wet,dry}`(Copy+Default Chorale/800/0.5/0.5/1.0/wet1/dry0、serde `serialize` 门控、`sanitised` 非有限归默认 + 全字段钳入合法域)、`LeslieSpeed`∈{Brake,Chorale(默认),Tremolo}；导出 `DEFAULT_CROSSOVER_HZ`/`HORN_SLOW_HZ`/`HORN_FAST_HZ`/`DRUM_SLOW_HZ`/`DRUM_FAST_HZ`/`HORN_ACCEL_SECONDS`/`HORN_DECEL_SECONDS`/`DRUM_ACCEL_SECONDS`/`DRUM_DECEL_SECONDS`/`MAX_AM_DEPTH=1.0`/`MAX_DOPPLER_DEPTH=1.0`、`LeslieNode::{new(sample_rate,max_block_frames,params),max_block_frames,speed,doppler_depth,am_depth,stereo_spread,horn_rate_hz,drum_rate_hz,set_speed,set_doppler_depth,set_am_depth,set_stereo_spread,set_wet,set_dry}`+impl `AudioNode`（process 内析构 disjoint 借用调 `crossover.process_block`/reset 清 crossover+mono+bands 并把两转子 park 到当前 speed target + 重置 wet/dry Smoothed/latency=0）；effects/mod.rs 字母序 `pub mod leslie;` 置于 `haas_widener` 后 `mid_side_matrix` 前 + 同序 re-export（`MAX_*` 等名独特无冲突）+ catalogue 一条用 "--"；21 单测（latency=0、max_block_frames getter、零 max_block 钳为 1、默认 Chorale、默认转速起于 slow、params sanitise 钳位+非有限归默认、setters 钳位拒非有限、set_speed 更新状态、silence→silence、非有限输入保持有限、零帧安全、wet0/dry1 逐样本透传、wet-only 旋转有能量、**Tremolo 喂 6s 后 horn/drum 转速逼近 fast(>6.0/>4.0)**、**Brake→Tremolo spin-up 渐进且重转子滞后轻号(horn>drum 均未达 fast)**、**stereo_spread=0 使 L==R、spread=1 去相关 L/R**、mono 输出等于两腿均值(与 stereo 节点逐样本比对)、surplus(>2) 通道透传、reset 清尾、wet/dry setter 钳位后仍渲染有限)+1 doctest（零延迟构造）全绿；收尾 CR 删除 `Rotor` 未读的 `slow_hz`/`fast_hz` 字段与对应 `new` 参数（消 dead_code 警告，连带删去不再需要的 `too_many_arguments` expect）、修复 `f32::rem_euclid` 在 no_std 不可用（改 floor 环绕）；core 累计 **797 单测 + 39 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`leslie.rs` 989 行 + effects `mod.rs` +12，commit `ac2dd3573`，经全绿验证 + 逐行 CR）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 源码或衍生代码）
- 版本历史: v0.95（**本版新增**：M1 效果族 `prism_audio_core` 新增 **频域频谱冻结节点 `SpectralFreezeNode`/`SpectralFreezeParams`（`nodes/effects/spectral_freeze`）**（借鉴 GRM Tools Freeze、Max/MSP `freeze~`、SoundHack spectral、以及现代音乐制作中广为记录的 spectral-freeze / 频谱无限延音这一经典思想，但不抄任何源码或衍生代码）：**频域无限延音节点**——对输入跑一条 Hann 窗加权叠加（`WOLA` / `STFT`，`OVERLAP_FACTOR=4`、hop=`fft_size/OVERLAP_FACTOR`、`COLA` 归一），未冻结时透明重建输入（纯 `latency=fft_size` 延迟、单位增益），冻结时锁存当前幅度谱并让每个 bin 相位按其中心频率增量 `k*expct` 逐 hop 推进，使被捕捉的音色化为稳定、无限延续的 pad（而非可闻的块循环）；信号流：每帧写入 `in_fifo`→以 rover 从 `fifo_latency` 起、满 `fft_size` 时 `process_frame` 并发布一 hop→滑动 fifo/accum；`process_frame` 内对每 bin 在频域对 live 复数谱与 held 复数谱逐 bin 线性插值（`re[k]=live_re+mix*(held_re-live_re)`，im 同），`freeze_target`∈{0,1}、每 hop 按 `freeze_step=hop/(FREEZE_RAMP_SECONDS*sr)` 步进实现 click-free 冻结/解冻；未冻结时持续跟踪 `frozen_mag=live_mag`/`frozen_phase=atan2(im,re)` 以保证恒等重建并让后续冻结瞬时锁存；**可选 diffusion**——冻结时每 hop 对每 bin 相位加 `diffusion*rng.next_bipolar()*PI`（私有 `FreezeRng`，`xorshift64`-star，`state=seed|1`，与 granular `GrainRng` 同算法但独立定义），破坏完美相干冻结的金属静态感、赋予延音轻微 shimmer 的演化感而不改频谱包络，无任何 AI/ML；**复用而非重复**（`# Relationship`）——`STFT` 框架与共享 radix-2 `Fft`(`crate::fft`) 复用兄弟模块 `pitch_shifter`/`spectral_gate` 的结构，但不做瞬时频率估计/bin 重映射（区别于 pitch_shifter）亦不衰减 bin（区别于 spectral_gate），且区别于时域 `granular` 的重触发式延音；RT 契约——窗、共享 `Fft` plan、每通道 fifo/accum/频谱锁存均在 `new` 预分配，`process` 零堆分配/无锁/不 panic、逐输出样本 `flush_denormal`、非有限输入置 0 保持输出有限、`latency_frames=fft_size`；参数 `SpectralFreezeParams{frozen,diffusion,seed}`(Copy+Default false/0/golden-seed、serde `serialize` 门控、`sanitised` 非有限归默认 + diffusion 钳 [0,`MAX_DIFFUSION=1.0`])；导出 `OVERLAP_FACTOR=4`/`DEFAULT_FFT_SIZE=2048`/`MIN_FREEZE_FFT_SIZE=64`/`FREEZE_RAMP_SECONDS=0.05`/`MAX_DIFFUSION=1.0`/`DEFAULT_SEED`、`SpectralFreezeNode::{new(sample_rate,channels,requested_size,params),fft_size,hop,channels,is_frozen,diffusion,set_frozen,set_diffusion}`+impl `AudioNode`（process/reset 清所有 fifo/accum/frozen_mag/frozen_phase + 重播种 rng + freeze_mix=freeze_target/latency=fft_size）；effects/mod.rs 字母序 `pub mod spectral_freeze;` 置于 `spectral_gate` 前 + 同序 re-export（**省略 `DEFAULT_FFT_SIZE`/`OVERLAP_FACTOR`/`DEFAULT_SEED` 以避与 spectral_gate/granular 已有 re-export 的 E0252 命名冲突**，仅导 `FREEZE_RAMP_SECONDS`/`MAX_DIFFUSION`/`MIN_FREEZE_FFT_SIZE`/`SpectralFreezeNode`/`SpectralFreezeParams`，其余常量经 `spectral_freeze::` 路径访问）+ catalogue 一条用 "--"；18 单测（requested_size 向上取 2 的幂、小请求命中 `MIN_FREEZE_FFT_SIZE`、hop=size/OVERLAP_FACTOR、latency=size、params sanitise 钳位、set_diffusion 钳位拒非有限、set_frozen 切换状态、silence→silence、非有限输入保持有限、零帧安全、未冻结单位增益重建且基频主导(goertzel)、**冻结后停止输入仍持续输出能量(核心 freeze 行为)**、冻结单频输入持续产生该频能量(goertzel)、diffusion>0 使冻结输出去相关于 diffusion=0、冻结切换 click-free(逐样本跳变 <0.5 且有界)、立体声双通道一致、surplus 通道透传、reset 清尾)+1 doctest（零延迟外的 latency=1024 构造）全绿；core 累计 **776 单测 + 38 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`spectral_freeze.rs` 767 行 + effects `mod.rs` +10，commit `b0bba8cc7`，经全绿验证 + 逐行 CR + 收尾时删除未使用的 `use crate::param::{Ramp, Smoothed};` import）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 源码或衍生代码）
- 版本历史: v0.94（**本版新增**：M1 效果族 `prism_audio_core` 新增 **实时粒子云纹理节点 `GranularNode`/`GranularParams`（`nodes/effects/granular`）**（借鉴 GRM Tools、Ableton Granulator、Reaktor 以及现代游戏音频中间件所公开记录的经典 granular synthesis / grain-cloud 手法这一思想，但不抄任何源码或衍生代码）：**实时颗粒化处理节点**而非静态素材采样器——把输入连续下混为单声道录入一条 `DEFAULT_CAPTURE_SECONDS=4.0s`（外加一个最大 grain 余量）的环形 capture buffer，按 `density_hz` 调度大量短促、Hann 窗加权的 grain 从近期历史回放，每个 grain 独立随机 position/pitch/pan，使稳态输入溶解为不断演化的叠加微事件云（闪烁 pad、时间涂抹氛围、音高云、glitch 结巴皆由控件决定）；信号流：每帧 downmix→写 capture（`flush_denormal`）→`write_total+=1`→以 `interval=sr/density` 调度 spawn（带 `MAX_GRAINS` 守卫）→渲染所有 active grain（窗相位 `env=0.5-0.5*cos(TAU*age/length)`×overlap 归一增益×equal-power pan，linear interp 读 capture，`pos+=pitch`）→`out=dry*input+wet*cloud`；**读取有界安全**——`read_capture` 把读位 clamp 到有效录制窗 `[write_total-len, write_total-1]` 并经 `ring_index`(`abs % len`) 取模，`write_total==0` 返回 0，永不 panic；**响度恒定**——`overlap=density*grain_seconds`，`gain_norm=1/sqrt(max(overlap,1))` 使密度/粒长变化时响度大致恒定（`recompute_gain` 在 `new` 及 `set_grain_size_ms`/`set_density` 调用）；**复用而非重复**（`# Relationship`）——grain detune 复用兄弟模块 `pitch_shifter` 的 `semitones_to_ratio` 并钳入其 `MIN_PITCH_RATIO`/`MAX_PITCH_RATIO`，与单比率 `PitchShifterNode`、静态素材 `sources::sample_player` 明确区分；**确定性**——私有 `GrainRng`(`xorshift64`-star，`state=seed|1`) 产生可复现 jitter，`reset` 重播种，无任何 AI/ML；RT 契约——capture 环与 `MAX_GRAINS=64` grain 池在 `new` 预分配，`process` 零堆分配/无锁/不 panic，capture 写入与每输出样本 `flush_denormal`，`wet`/`dry` 走 `Smoothed`(`MIX_RAMP_SECONDS=0.01`) click-free，塑形参数于下次 spawn 生效，`latency_frames=0`（干声直达）；通道——stereo 用 wet_l/wet_r、mono 求和两腿、>2 偶/奇复用，干声保原通道；参数 `GranularParams{grain_size_ms,density_hz,position_seconds,position_jitter_ms,pitch_ratio,pitch_jitter_semitones,spread,wet,dry,seed}`(Copy+Default 80ms/20Hz/0.25s/50ms/1.0/0/0.6/wet1/dry0/golden-seed、serde `serialize` 门控、`sanitised` 非有限归默认 + 全字段钳入合法域)；导出 `MAX_GRAINS=64`/`DEFAULT_CAPTURE_SECONDS=4.0`/`MIN_GRAIN_MS=2.0`/`MAX_GRAIN_MS=2000.0`/`MIN_DENSITY_HZ=0.1`/`MAX_DENSITY_HZ=200.0`/`MAX_SPREAD=1.0`/`DEFAULT_SEED`、`GranularNode::{new(sample_rate,channels,params),channels,capture_frames,pitch_ratio,density,active_grains,set_grain_size_ms,set_density,set_position_seconds,set_position_jitter_ms,set_pitch_ratio,set_pitch_semitones,set_pitch_jitter_semitones,set_spread,set_wet,set_dry}`+impl `AudioNode`（process/reset/latency 0）；effects/mod.rs 字母序 `pub mod granular;` 置于 `frequency_shifter` 后 `graphic_eq` 前 + 同序 re-export（仅导本模块自定义符号，不重导复用自 `pitch_shifter` 的 ratio 常量/转换）+ catalogue 一条用 "--"；16 单测（capture 环覆盖默认窗、latency=0、silence→silence、零帧安全、wet0/dry1 逐样本透传、wet-only 云有能量、满密度输出有限有界(<16)、**pitch=2 八度能量 goertzel 判别移至高频**、同 seed 逐位相等、异 seed 去相关(back-half)、reset 可复现、active_grains 有界(0<x<=64)、立体声满 spread 去相关 L/R、mono 输出求和两腿、sanitised 钳位、setters 钳位拒非有限）+1 doctest（零延迟构造 + 处理一块有限）全绿；core 累计 **758 单测 + 37 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`granular.rs` 979 行 + effects `mod.rs` +13，commit `1c8d4e18e`，经全绿验证 + 逐行 CR + 两处测试缺陷修复：零帧测试改用 capacity>0+set_active_frames(0) 绕开 `AudioBuffer::new` 非零断言、异 seed 测试改用 1s 输入分析 back-half 以保证 grain 读到有效近期历史）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 源码或衍生代码）
- 版本历史: v0.93（**本版新增**：M1 reverb 族 `prism_audio_core` 新增 **shimmer 混响 `ShimmerReverb`/`ShimmerReverbParams`（`nodes/reverb/shimmer`）**（借鉴 Eventide/Lexicon 式 "shimmer" 补丁与其众多软件衍生所公开记录的经典录音室手法——反馈环内嵌八度音高变换——这一思想但不抄任何源码/衍生代码）：**组合型节点**，严格复用而非重复 DSP——内部持有一个被强制全湿（`wet=1`/`dry=0`，输出纯混响尾）的 `FdnReverb` tank + 一个驱动反馈移调的 `PitchShifterNode`（`# Relationship` 已澄清与二者各自单独使用的区别）；信号流：`reverb_in = dry + shimmer_feedback*前一块的移调尾`→送入全湿 FDN tank 得 `reverb_out`→本节点做 `out = dry*dry_gain + wet*reverb_out` 的干湿混→再把 `reverb_out` 经 `PitchShifterNode`（默认 +12 半音即八度上移）变换成**下一块**的反馈源（故反馈环含一块延迟 + 声码器自身延迟，形成音乐性的 pre-bloom 预泛音；但干声与首次湿声直达输出零延迟，`latency_frames=0`）；**上行自限**——移调能量终将迁移到 tank 被阻尼的高频区被吸收，故 sub-unity 反馈增益 `MAX_SHIMMER_FEEDBACK=0.85` 叠加混响自身衰减即保证环路有界，无需任何限幅器；逐样本 `flush_denormal` 反馈拷贝防递归路径 denormal 累积，热路径全部 scratch（`reverb_in`/`reverb_out`/`shifted`/`feedback_buf`）在 `new` 预分配、零堆分配/无锁/不 panic；`shimmer_feedback`/`wet`/`dry` 全走 `Smoothed` 每帧推进 click-free，`pitch_semitones` 经 `set_pitch_semitones` 在下次声码器帧生效不打断正响尾音，`set_decay`/`set_damping` 转发 tank；参数 `ShimmerReverbParams{room_size,decay_rt60_seconds,damping,pitch_semitones,shimmer_feedback,wet,dry}`(Copy+Default room1/rt60 2.8s/damp0.4/+12 半音/fb0.5/wet0.3/dry1、serde `serialize` 门控、`sanitised` 非有限归默认 + 全字段钳入合法域)；导出 `MAX_SHIMMER_FEEDBACK=0.85`、`DEFAULT_SHIMMER_SEMITONES=12`、`DEFAULT_SHIMMER_FFT_SIZE=2048`、`ShimmerReverb::{new(sample_rate,layout,max_block_frames,params),layout,channels,max_block_frames,fft_size,shimmer_feedback,pitch_ratio,set_shimmer_feedback,set_pitch_semitones,set_decay,set_damping,set_wet,set_dry}`+impl `AudioNode`（process/reset 转发内部两节点并清反馈缓冲/latency_frames=0）；reverb/mod.rs 字母序 `pub mod shimmer;` 置于 `plate` 后 + 同序 re-export `MAX_SHIMMER_FEEDBACK`/`ShimmerReverb`/`ShimmerReverbParams` + catalogue 一条用 "--"；14 单测（零延迟、默认音高比≈2、silence→silence、wet=0/fb=0 纯干逐样本透传、输入停后尾音仍有能量且有限、**shimmer 开启使 1kHz 输入在 2kHz 八度分量能量 >4x 于关闭时（goertzel 单频 DFT 判别）**、满反馈下 1s 全幅激励输出有限且峰值 <50 环路有界、reset 清尾、零帧安全、非有限参数 sanitise、立体声双通道有限、sanitised 钳位、set_shimmer_feedback 钳 [0,0.85] 含非有限、getters）+1 doctest（零延迟构造+处理）全绿；core 累计 **742 单测 + 36 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`shimmer.rs` 746 行 + reverb `mod.rs` +8，commit `b47867e26`，经全绿验证 + 逐行 CR：反馈环一块延迟定序、全湿 tank + 本节点干湿混的职责划分、上行自限有界性论证、per-frame 平滑器推进一致性、组合子节点 disjoint 字段借用安全均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 源码或衍生代码）
- 版本历史: v0.92（**本版新增**：M1 频域族 `prism_audio_core` 抽出 **共享 radix-2 FFT 原语 `Fft`（crate 根模块 `fft`）** 并新增 **相位声码器 pitch shifter `PitchShifterNode`/`PitchShifterParams`（`nodes/effects/pitch_shifter`）**（借鉴 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 的频域重合成/变调思想但不抄任何源码）：① **`Fft` 原语**——把原本散落在 `spectrum.rs` 与 `spectral_gate.rs` 各自私有的 radix-2 Cooley-Tukey 蝶形实现统一成一份经充分测试的共享原语，供所有频域节点复用而非各自复制；`new(size)` 用 `power_of_two_at_least` 把请求长度向上取整为 2 的幂并预计算 bit-reversal 置换表与 twiddle 旋转因子表（`tw_re[k]=cos(-TAU*k/N)`/`tw_im[k]=sin(-TAU*k/N)` 长度 N/2，全走 `bevy_math::ops` 确定性数学），`forward(&self,re,im)` 原地无缩放变换、`inverse(&self,re,im)` 共享同一 twiddle 表走共轭路径并除以 N 还原单位增益，`size()` 暴露实际 2 的幂长度；twiddle 约定与既有 `spectrum`/`spectral_gate` 逐位一致，为将来迁移消除重复 FFT 铺路；导出 `Fft::{new,size,forward,inverse}`、`power_of_two_at_least(usize)->usize`、`MIN_FFT_SIZE=2`；9 单测（2 的幂取整、往返恒等、线性性、单位冲激→常数谱、Parseval 能量守恒、已知单频 bin 精确、…）+1 doctest 全绿；字母序 crate 根 `pub mod fft;` 置于 `buffer` 后 `graph` 前；② **`PitchShifterNode` 相位声码器**——等比 pitch shift（**保时长、保谐波比**，区别于加 Hz 偏移导致非谐的 `FrequencyShifterNode` 与基于循环分数延迟的时域 `VibratoNode`）：Hann 加权 OLA STFT（`OVERLAP_FACTOR=4` 即 hop=N/4）复用共享 `Fft`，分析阶段每帧 forward FFT 后逐 bin 用相位解缠估计**瞬时频率**（`delta=phase-last_phase; delta-=k*expct; delta=princ_arg(delta); true_freq=(k+osamp*delta/TAU)*freq_per_bin`，`princ_arg` 用 `ops::floor` 折到 (-pi,pi]），移调阶段把 bin `k` 幅度累加进 `round(k*ratio)` 并把该合成 bin 真频标记为 `ana_freq[k]*ratio`，重合成阶段 `sum_phase[j]+=TAU*((syn_freq[j]-j*freq_per_bin)/freq_per_bin)/osamp+j*expct` 积累相位、`re=mag*cos`/`im=mag*sin` 并做 Hermitian 共轭对称镜像后 inverse FFT，Hann 合成窗 + COLA `ola_norm`（平方窗分母倒数，ratio=1 时单位增益精确重建）OLA 叠加；常量 `freq_per_bin=sr/N`、`expct=TAU*hop/N`、`osamp=N/hop`；流式骨架照搬 `spectral_gate`（in_fifo/out_fifo/out_accum/rover 延迟对齐），per-channel `last_phase`/`sum_phase` 状态、per-frame `ana_mag`/`ana_freq`/`syn_mag`/`syn_freq` scratch 跨通道复用，surplus 通道透传，`latency_frames=N`；参数 `PitchShifterParams{pitch_ratio}`（Default 1.0、serde `serialize` 门控、`sanitised` 钳 [0.25,4.0] 非有限→1）；导出 `MAX_PITCH_RATIO=4.0`/`MIN_PITCH_RATIO=0.25`、`semitones_to_ratio(st)=ops::powf(2,st/12)`、`PitchShifterNode::{new(sample_rate,channels,requested_size,params),fft_size,hop,pitch_ratio,set_pitch_ratio,set_semitones}`+impl `AudioNode`（含 reset 清尾/latency_frames=N）；**故意不 re-export `OVERLAP_FACTOR`（与 spectral_gate 同名常量冲突 E0252），经 `pitch_shifter::OVERLAP_FACTOR` 访问**；16 单测（几何 fft_size/hop/latency、semitones<->ratio、clamp/sanitised、silence→silence、非有限安全、零帧安全、unity ratio 保音保电平、八度上↑2f 主导、八度下↑500Hz 主导、立体声同输入逐位相等、reset 清尾、surplus 通道透传，tests 用 goertzel 单频 DFT 检频）+1 doctest（latency=1024）全绿；字母序 effects `pub mod pitch_shifter;` 置于 `phaser` 后 `ring_modulator` 前（effects/mod.rs 同序 re-export + catalogue 用 "--"）；core 累计 **728 单测 + 35 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`Fft` 原语 `fft.rs` 397 行 commit `831bc0c21`；`PitchShifterNode` `pitch_shifter.rs` 687 行 + effects `mod.rs` +10 commit `5155efdab`，均经全绿验证 + 逐行 CR）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio/Web Audio 源码或衍生代码）
- 版本历史: v0.91（**本版新增**：M1 效果族 `prism_audio_core` 抽出 **共享多相过采样原语 `Oversampler`/`OversamplerState`/`DryDelay`（crate 根模块 `oversampler`）** 并新增 **west-coast 波折叠 `WavefolderNode`/`WavefolderParams`/`FoldShape`（`nodes/effects/wavefolder`）**，同时把既有 `WaveshaperNode` 迁移至该共享原语以消除重复过采样代码（借鉴 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 的抗混叠非线性处理思想但不抄任何源码）：① **`Oversampler` 原语**——把多相上采样→逐样本整型倍率非线性 shaping→抽取的教科书多速率机器从单个效果里抽成一份共享、经充分测试的实现，供所有非线性节点复用而非各自复制；`new(factor,taps_per_phase)`（factor/taps 各 `.max(1)`，`factor==1` 为零延迟恒等旁路，`process_sample` 直接 `flush_denormal(shape(x))` 故调用方无需特判）、`with_default_taps(factor)`、`factor`/`filter_taps`(=`taps_per_phase*factor`，恒等时 0)/`latency_frames`(=`round((taps-1)/factor)`)、`make_state`→每通道 `OversamplerState`、`process_sample(state,x,shape:FnMut(Sample)->Sample)`（环形缓冲多相插值→对每个过采样值跑非线性闭包→同一 Hann 窗 sinc 原型 lowpass 抽取，统一 DC 增益、线性相位固定群延迟、denormal-flush）；配套 `DryDelay{new(frames),push,reset,len,is_empty}` 定长延迟线对齐干声相位（长度 0 恒等）；原型滤波 `design_lowpass`（Hann 窗 sinc、cutoff=1/factor、归一化单位 DC 增益）由 waveshaper 原样搬入并成为唯一副本；② **`WavefolderNode` 折叠效果**——与 `waveshaper` 软削波本质正交互补：削波在阈值处**压缩**信号趋近天花板，折叠在阈值处把信号**反射**回来（`saturation` 注释亦自述"not a wave folder"），折叠谐波极丰富故支持到 X8 过采样；两种闭式折叠传递函数 `FoldShape::Triangle`（period-4 三角波 `t(v)=|rem_4(v-1)-2|-1`，在 [-1,1] 恒等、越界镜像反射，用 `ops::floor` 实现非负取模保确定性）与 `FoldShape::Sine`（`sin(v*pi/2)` 原点单位斜率、天然有界到 [-1,1] 的 Buchla 式平滑折叠）；信号路径 `fold(drive*v+offset)` 经共享 `Oversampler` 求值、×`output_gain`、再与 `DryDelay` 对齐的干声按 `wet`/`dry` 混；参数 `WavefolderParams{drive,offset,output_gain,wet,dry}`(Copy+Default drive1/offset0/gain1/wet1/dry0、serde `serialize` 门控、`sanitised` 非有限归默认+mix 钳 [0,1])，`offset` 引入偶次谐波破坏对称；`drive`/`offset`/`output_gain`/`wet`/`dry` 全走 `Smoothed` click-free、per-channel 快照回放模式（末通道 commit 回写）；导出 `Oversample{X1,X2,X4,X8}`(re-export 别名 `FolderOversample` 避与 waveshaper 的 `Oversample` 冲突)、`FoldShape{Triangle,Sine}`、`WavefolderNode::{new(sample_rate,channels,oversample,fold_shape,params),set_drive,set_offset,set_output_gain,set_wet,set_dry,oversample,fold_shape,filter_taps}`+impl `AudioNode`（process/reset/latency_frames 转发 oversampler）；③ **`WaveshaperNode` 迁移**——删除其私有 `ChannelState`/`shape_oversampled`/`dry_delayed`/`design_lowpass`/`zeroed`/`TAPS_PER_PHASE`，改持一个共享 `Oversampler`+per-channel `OversamplerState`+`DryDelay`，`tanh` 作为 shaping 闭包 `|v| ops::tanh(drive*v)`，净删 176 行重复代码且全部 7 个既有单测（含 `latency_matches_filter_design`/`oversampling_reduces_aliasing` alias 阈值测试）逐位保持通过，并补齐 `# Provenance`/`# Relationship` rustdoc；`oversampler` 原语单测 10 个（恒等 factor==1 零延迟逐位旁路、退化参数钳位、`filter_taps=taps*factor`、latency 非负且随滤波增长、DC 过采样后≈原样、低频 pass-band 能量比≈1、reset 后静音逐位、`DryDelay` 延迟/0 长恒等/reset 清零）、`wavefolder` 单测 14 个+1 doctest（三角折叠手算精确值、正弦折叠单位斜率且绝对有界、两形 X8 极端 drive 有界、X1 直流折叠精确、offset 引入二次谐波>3×、X1 vs X8 7kHz 第 7 谐波折回 1kHz alias 抑制>2×、wet=0 纯干逐位、dry=0 纯湿有能量、latency 转发 oversampler、stereo 逐通道独立、非有限参数 sanitise、零帧安全、reset 后与新实例逐位一致、Default 往返）；字母序 crate 根 `pub mod oversampler;` 置于 `nodes` 后 `param` 前，effects 字母序 `pub mod wavefolder;` 置于 `vocoder` 后 `waveshaper` 前（effects/mod.rs 同序 re-export+catalogue 用 "--"）；core 累计 **703 单测 + 33 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，分两次精确 commit（`oversampler.rs` 504 行 + `wavefolder.rs` 798 行 + effects `mod.rs` +8 + `lib.rs` +1，commit 1442e3f84；waveshaper 迁移 +64/-240，commit a1434a83b，均经逐行 CR）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.90（**本版新增**：M1 分析族 `prism_audio_core` 的 `analysis/` 新增 **谱特征分析 `SpectralFeatures`/`SpectralFeatureSet`/`SpectralFeaturesNode`（`nodes/analysis/spectral_features`）**（借鉴 `MPEG-7` Audio 低层描述符与标准音乐信息检索 `MIR` 教科书公式，但不抄任何源码——仅在既有 `SpectrumAnalyzer` 之上做纯统计降维，把一帧单边幅度谱压缩成一组标量"音色"描述符，与 `spectrum` 暴露原始 bin、`loudness` 测响度、`correlation`/`goniometer` 测立体声像、`pitch_detector` 测基频四者正交互补、不重复，仅复用 `SpectrumAnalyzer` 公共 API 组合（`# Relationship` 已澄清，非重复实现 FFT）：由单边谱 `M[k]`(k=0..K) 与采样率，反推分析长度 `size=(K-1)*2`、bin 中心频率 `f[k]=k*sample_rate/size`(Hz)，输出 10 个描述符——**centroid**(亮度/质心, Hz)=Σf·M/ΣM、**spread**(带宽/二阶矩, Hz)=sqrt(Σ(f-C)^2·M/ΣM)、**skewness**=Σ(f-C)^3·M/ΣM/spread^3(关于质心的偏度)、**kurtosis**=Σ(f-C)^4·M/ΣM/spread^4(原始四阶标准矩, 非减 3 的 excess, 高斯约 3)、**flatness**(0..=1, Wiener 熵/平坦度)=功率谱 P=M^2 的几何均值/算术均值(白噪→1、纯音→0, 几何均值前加微小功率地板避免 ln(0))、**crest**=max(P)/mean(P)(谱峰度, 峰谱大/平坦谱≈1)、**rolloff**(Hz)=使累计幅度达到可配 `rolloff_fraction`(默认 0.85)的最低频、**flux**=sqrt(Σ(M_t-M_{t-1})^2)(相邻帧 `L2` 距离, onset/瞬态大、稳态 0, 首帧对全零前帧)、**slope**(幅度/Hz)=M 对 f 最小二乘回归斜率(常为负下滑)、**decrease**=`MPEG-7` 谱衰减 Σ_{k>=1}(M[k]-M[0])/k/Σ_{k>=1}M[k]；静音帧(ΣM<地板)或<2 bin 全返回 0 杜绝除零，非有限 bin 当 0 处理；实时契约：`SpectralFeatures` 仅持 flux 所需的上一帧幅度一个堆缓冲、首次 `analyze` 定长后复用、稳态零分配/无锁/不 panic，`SpectralFeaturesNode` 另持一个 `SpectrumAnalyzer`(其缓冲构造期一次分配)，全部超越函数经 `bevy_math::ops`(ln/exp/sqrt) 位级可复现，f64 累加读回 `as Sample`；节点为 pass-through meter：process 透传信号+喂第一通道，`SpectrumAnalyzer` 每完成一次变换即把新幅度 bin 降维成一帧 `SpectralFeatureSet` 存 `latest`、`frames_analyzed` 递增；导出 `DEFAULT_ROLLOFF_FRACTION=0.85`、`SpectralFeatureSet{centroid,spread,skewness,kurtosis,flatness,crest,rolloff,flux,slope,decrease}`(Copy+Default+PartialEq、serde `serialize` 门控)、`SpectralFeatures::{new(rolloff_fraction),default,rolloff_fraction,set_rolloff_fraction,reset,analyze(magnitudes,sample_rate)}`、`SpectralFeaturesNode::{new(requested_size,hop,window,rolloff_fraction),latest,frames_analyzed,analyzer,analyzer_mut,features,features_mut}` 并 impl `AudioNode`；字母序 `pub mod spectral_features;` 置于 `pitch_detector` 之后、`spectrum` 之前(analysis/mod.rs 同序 re-export+catalogue 用 "--")；core 累计 **679 单测 + 32 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`spectral_features.rs` 807 行 + analysis `mod.rs` +10，commit 5a1db0326，经逐行 CR）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.89（**本版新增**：M1 混响族 `prism_audio_core` 新增 **Dattorro 图八板式混响 `PlateReverb`/`PlateReverbParams`（`nodes/reverb/plate`）**（借鉴 Lexicon 224/EMT 140 机械板混响音色与 J. Dattorro「Effect Design, Part 1: Reverberator and Other Filters」JAES 1997 图八「tank」拓扑及其调音表但不抄任何源码——单一交叉耦合 allpass 反馈环的 insert 效果节点，1 输入 1 输出，与 `algorithmic` 的 Freeverb 并联 comb/早反射房间模型、`fdn` 的 Jot 正交 Hadamard 反馈网络、`convolver` 的实测脉冲响应卷积三者正交互补、不重复）：信号链三段——① 输入调理：pre-delay → 一极点 `bandwidth` 低通（`y+=bandwidth*(x-y)` 入环前削顶）→ 四级串联扩散 allpass（延迟 142/107 用 `input_diffusion_1`=0.750、379/277 用 `input_diffusion_2`=0.625，参考 29761Hz）；② tank：左右两半交叉耦合（`left_in=diffused+right_out`、`right_in=diffused+left_out`，1 样本寄存器引入对称反馈），每半 = 调制 allpass（左 base 672/右 908，~1Hz 正弦 LFO 调制分数延迟、线性内插、左右正交相位 0 与 pi/2 去相关、避免尾音驻留固定本征频率）→ 长延迟 1（左 4453/右 4217）→ 一极点 `damping` 低通（`out=(1-damping)*x+damping*prev`，高频随尾音衰减仿空气吸收）→ ×`decay` → 固定 allpass 2（左 1800/右 2656，`decay_diffusion_2`=0.50）→ 长延迟 2（左 3720/右 3163）→ ×`decay` 反馈到对侧；③ 多抽头输出：左右各 7 个带符号抽头从两半内部延迟线读取（yl 从 `delay_r1`[266,2974]/`ap_r2`[1913]/`delay_r2`[1996]/`delay_l1`[1990]/`ap_l2`[187]/`delay_l2`[1066]，yr 对称取另一半），由单声道激励生成宽立体声去相关尾音；全部延迟/抽头偏移以 29761Hz 参考值在构造期按 `rate_scale=sr/29761` 重缩放（仿 `algorithmic` 从 44100 缩放）、`.max(1)` 并钳到各线长；allpass 为真格型 `w=x+g*d; store; y=d-g*w` → `H(z)=(z^-M-g)/(1-g z^-M)`、`|H|=1`、`|g|<1` 稳定（区别于 `algorithmic` 的 Freeverb 近似 `output=-input+buffered`）；`decay<1` 且各 allpass `|g|<1` 保证环路增益 <1、脉冲响应衰减、输出有界；实时契约：所有延迟线/allpass/一极点在 `new` 构造期一次分配，`process` 仅读写缓冲、推进整数索引、每样本一次正弦，零分配/无锁/不 panic，denormal 在每次延迟写入/滤波状态/反馈寄存器处 flush；多通道：输入下混单声道激励，立体声按 yl/yr 两抽头、单声道按 (yl+yr)/2、>2 通道偶/奇通道复用 yl/yr；导出 `PlateReverbParams{pre_delay_ms,bandwidth,decay,decay_diffusion_1,decay_diffusion_2,input_diffusion_1,input_diffusion_2,damping,mod_depth,wet,dry}`（Copy+Default、serde `serialize` 门控、`sanitised()` 全钳位且非有限替换为默认）、`PlateReverb::{new(sample_rate,channels,params),channels,decay,damping,bandwidth,wet,dry,set_decay,set_damping,set_bandwidth,set_decay_diffusion_1,set_decay_diffusion_2,set_mod_depth,set_wet,set_dry}`，impl `AudioNode`（process 下混+tick+干湿混 / reset 清所有延迟线/滤波/反馈寄存器/LFO 相位）；wet(0.4)/dry(1.0) 走 `Smoothed` click-free、decay/damping/bandwidth/diffusion/mod_depth 即时 setter 全钳位且拒非有限；13 golden 对拍（脉冲响应衰减且有界、decay 越大尾音越长、damping 越大高频一阶差分能量越低、立体声左右去相关、wet=0 纯干、dry=0 纯湿、同配置逐位复现、reset 后与新实例逐位一致、pre_delay 前段无湿能量、单声道有限且有能量、非有限参数构造期拒绝、setter 钳位与拒非有限、零帧安全）+既有 doctest；字母序 `pub mod plate;` 置于 `fdn` 之后（reverb/mod.rs 同序 re-export+catalogue 用 "--"、"four complementary algorithms"）；core 累计 **659 单测 + 30 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`plate.rs` 1077 行 + reverb `mod.rs` +8/-1，commit 22b804569，经逐行 CR）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.88（**本版新增**：M1 源层 `prism_audio_core` 新增 **带限 mipmap wavetable 振荡器 source 节点 `WavetableOscillatorNode`/`WavetableOscillatorParams`（`nodes/sources/wavetable_oscillator`）**（借鉴 Serum/Vital 式 wavetable 合成与 Nigel Redmon "earlevel engineering" 公开文章所述的"每八度一张带限表 mipmap"经典加性合成/离散傅里叶级数思想但不抄任何源码——**纯查表** source 节点（0 输入 1 输出，区别于几何波形 `PolyBLEP` 的 `oscillator`、随机的 `noise`、回放 PCM 的 `sample_player`、自激励拨弦的 `karplus_strong`）：构造期按调用方给定的**谐波幅度数组** `harmonic_amplitudes[k-1]`=第 k 次正弦谐波线性幅度用**加性合成**建一组每八度一张的单周期表 `table_m[i]=sum_{k=1..=m} a[k]*sin(2*pi*k*i/TABLE_SIZE)`，表 m（含 1..=m 次谐波）仅在归一化基频 `nf=f0/sr` 满足 `m*nf<=0.5` 时启用故其所有谐波精确位于 Nyquist 以下、**无混叠**（非 FFT、全自带、确定性）；表按 Nyquist 上限频率 `top_freq=0.5/m` 索引、每块一次 `select(nf)` 取首个 `top_freq>=nf` 的表（否则末张最带限），因各表共享谐波幅度一致、且**全体表乘同一全局归一化因子**（取自最丰富表的峰值）故音高扫动切表时电平无跳变；读表走**周期环绕四点 Catmull-Rom** 内插（与 `sample_player` 共用同一内插核但此处作用于合成单周期而非录音 PCM）兼得平滑与无线性内插的高频损失，`TABLE_SIZE=2048`、谐波数按 `min(max_h, sr/2/MIN_FREQUENCY_HZ=20, TABLE_SIZE/2=1024)` 封顶、从 `top_h` 起每次 `/2` 建到 1（saw 48kHz 得 11 张表 1024..1）、相位累加器 `phase in[0,1)`、`inc=f0/sr<0.5` 故单次减法 wrap；精确整数相位归约 `(k*i)%TABLE_SIZE` 保 f32 下谐波严格周期；便捷构造 `saw`（`a[k]=1/k`）/`square`（奇次 `1/k`）/`triangle`（奇次 `(-1)^((k-1)/2)/k^2`）暴露经典几何波形谱、`new`/`from_params` 接任意谱、空/全零谱退化为单张静默表；与几何 `OscillatorNode` 严格区分不重复——后者以 `PolyBLEP` 阶跃/斜率修正带限**固定解析波形**、本节点以八度 mipmap 加性表带限**任意谐波谱**，二者不共享滤波数学，仅复用本 crate 自有 `Sample`/`Smoothed`/Catmull-Rom 核（`# Relationship` 已澄清）；实时契约：整套 mipmap 在构造期一次建成，`process` 零分配/无锁/不 panic——按索引选表、有界内插读表、推进 wrap 相位，表索引恒在界内、denormal-flush、amplitude 走 `Smoothed` click-free、非有限参数构造期/setter 全拒；导出 `TABLE_SIZE=2048`、`WavetableOscillatorParams{frequency_hz,amplitude}`（Copy+Default 220Hz/1、serde `serialize` 门控）、`WavetableOscillatorNode::{new(sample_rate,harmonic_amplitudes,frequency_hz,amplitude),saw,square,triangle,from_params,set_frequency_hz,set_amplitude,frequency_hz,amplitude,phase,table_count}`，impl `AudioNode`（process 选表+读表+复制到多通道 / reset 清相位与 amplitude）；`MIN_FREQUENCY_HZ=20` 保持模块私有避免与 `karplus_strong` 同层 re-export 冲突；15 golden 对拍（saw 自相关周期≈sr/f0=480、13kHz 仅基频入带→与纯正弦归一化互相关>0.995 验带限、amplitude 线性缩放、同配置逐位复现、空/全零谱静默、stereo 双通道逐位相等、非有限频率钳位、零帧安全、set_frequency 改音高且有界、三波形输出有界<=1.1、mipmap 11 张表、选表随频率单调且端点正确、Default 往返、reset 清相位）+1 doctest；字母序 `pub mod wavetable_oscillator;` 置于 `sample_player` 之后（sources/mod.rs 同序 re-export+catalogue 用 "--"）；core 累计 **646 单测 + 30 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`wavetable_oscillator.rs` 781 行 + sources `mod.rs` +6，commit 9b9dc21fb，经逐行 CR 并修复选表测试对 index-0 的错误期望——最丰富表仅服务 nf<=0.5/1024 的最低音高）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.87（**本版新增**：M1 效果/源层 `prism_audio_core` 新增 **Extended Karplus-Strong 拨弦物理建模 source 节点 `KarplusStrongNode`/`KarplusStrongParams`（`nodes/sources/karplus_strong`）**（借鉴 Karplus-Strong 1983《Digital Synthesis of Plucked-String and Drum Timbres》与 Jaffe-Smith 1983《Extensions of the Karplus-Strong Plucked-String Algorithm》及 J.O.Smith《Physical Audio Signal Processing》分数延迟 allpass 内插这一教科书级物理建模/经典 DSP 思想但不抄任何源码——**自激励**拨弦物理模型 source 节点（0 输入 1 输出，区别于几何波形的 `oscillator`、随机的 `noise`、回放 PCM 的 `sample_player`）：仅在拨弦瞬间注入一段**噪声突发激励**填入延迟线，音色完全由环路如何滤波并再循环该能量涌现；递推 `delayed[n]=line[n-N]`（整数延迟线）、`lp[n]=(1-S)*delayed[n]+S*delayed[n-1]`（一零阻尼环路滤波器）、`ap[n]=C*lp[n]+lp[n-1]-C*ap[n-1]`（一阶 allpass 调音）、`line[n]=g*ap[n]`（环路增益写回）；基频 `f0=sr/D`，总环路延迟 `D=N+S+eps` 三段相加：整数延迟线贡献 `N`、一零滤波器在低频相位延迟 `S` 样本、allpass 低频相位延迟 `eps=(1-C)/(1+C)` 样本，故由 `delta=D-S`、`N=floor(delta)`、`eps=delta-N in[0,1)` 解 `C=(1-eps)/(1+eps)`（钳到 `MAX_ALLPASS_COEFF=0.9995` 避极点贴单位圆，detune<1e-3 样本不可闻），把音高拆成整数延迟线 + allpass 亚样本两段实现**连续精确调音**而非量化到整样本音高；`brightness in[0,1]` 映射阻尼系数 `S=0.5*(1-brightness) in[0,0.5]`——`brightness=1` 时 `S=0` 环路滤波器为恒等（亮/金属、HF 衰减慢），`brightness=0` 时 `S=0.5` 为经典二抽头均值 `0.5+0.5 z^-1`（Nyquist 增益归零、HF 快速衰减变暗，契合真实琴弦先失高频）；因阻尼滤波器 DC 增益恒 1，基频按纯环路增益 `g` 每环路周期衰减，为命中 `60 dB`（因子 `10^-3`）衰减时间 `decay_seconds`（期间环路走 `decay_seconds*f0` 个周期）取 `g=10^(-3/(decay_seconds*f0))`（钳到 `MAX_LOOP_GAIN=0.9999` 保证恒衰减），全走 `bevy_math::ops::{floor,powf,round}`；激励为自带 xorshift64/SplitMix64（魔数 `0x9E3779B97F4A7C15` 等）确定性 PRNG 的白噪突发，同 `seed` 同拨弦逐位复现；`trigger()` 置 pending、在 `process` 块首沿块边界原子施加拨弦（重算并 latch 系数 `n`/`damping`/`allpass_coeff`/`loop_gain` 再填 `line[0..n]` 噪声并清滤波器记忆）故 retrigger 采样精确且控制线程从不触碰延迟线、`set_frequency_hz`/`set_decay_seconds`/`set_brightness` 更新用户参数并在下次 `trigger` latch（正发声的弦不被打断、retune 不爆音）；与 input 驱动的 `effects::CombResonatorNode`（线性插值、lowpass-damped 反馈梳状、暴露原始 feedback 系数）严格区分——本节点自激励、以 allpass 做亚样本调音、暴露显式 T60 衰减时间、以 Jaffe-Smith 一零滤波器控亮度，`# Relationship` 已澄清；延迟线按最低音高 `MIN_FREQUENCY_HZ=20` 预分配（`line_len=round(sr/20)+2`），热路径零堆分配/无锁/不 panic，逐样本 denormal-flush、环路增益钳 <1、setter 拒非有限、输出幅度走 `Smoothed` 实现 click-free 自动化；导出 `MIN_FREQUENCY_HZ=20`/`MAX_LOOP_GAIN=0.9999`、`KarplusStrongParams{frequency_hz,decay_seconds,brightness,excitation,amplitude}`（Copy+Default 220Hz/2s/0.5/1/1、serde `serialize` 门控）、`KarplusStrongNode::{new(sample_rate,seed,params),trigger,set_frequency_hz,set_decay_seconds,set_brightness,set_excitation,set_amplitude,frequency_hz,decay_seconds,brightness,excitation,loop_gain,delay_len,is_ringing}`+impl `AudioNode`（含 `reset` 清零静默）；14 golden 对拍（触发前静默/拨弦注入能量/441Hz@44100 延迟线精确 100 样本且自相关峰在周期 100+-2/逐块能量单调不增/亮弦比暗弦 HF 衰减慢（一阶差分能量晚窗/早窗之比）/同 seed 逐位复现/retrigger 能量超衰减尾部/长衰减尾部能量超短衰减/非有限参数输出仍有限/零帧安全/立体声双通道逐位相等/reset 后静默/频率钳到 `[20, sr/2]`/环路增益 <1 且 >0）+1 doctest（默认拨弦后块非静默）；字母序 `pub mod karplus_strong;` 插在 `noise` 之前（sources/mod.rs 同序 re-export `KarplusStrongNode/KarplusStrongParams`，catalogue 一条用 "--"，**本模块 `MIN_FREQUENCY_HZ` 与 effects::comb_resonator 的同名常量不冲突因不在同层 re-export**）；core 累计 **631 单测 + 29 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`karplus_strong.rs` 833 行 + `sources/mod.rs` +11-3，commit 2bfc8e563，经全绿验证 + 逐行 CR：D=N+S+eps 调音拆分、allpass 系数与钳位、brightness->S 映射、T60->g 公式、trigger 块边界 latch 与噪声填充、与 CombResonatorNode 的自激励/调音/T60 区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.86（**本版新增**：M2 空间层 `prism_audio_spatial` 新增 **双耳时间差 ITD 预测模型 `InterauralTimeDifference`（`interaural_time_difference`）**（借鉴双耳定位 duplex 理论与经典球形头颅声学这一教科书级心理声学/经典 DSP 思想但不抄任何源码——以刚性球半径 `a=head_radius_m`、介质声速 `c=speed_of_sound`、源方位角 `phi`（0 正前、正向一侧耳）解析预测近远耳到达时差（秒），方位先 `wrap_pi` 归一到 `[-PI,PI]`；提供两族闭式预测：**Woodworth 射线模型**（频率无关、几何）前区 `|phi|<=PI/2` 取 `ITD=a/c*(phi+sin phi)`，后区 `|phi|>PI/2` 用对 `+-90` 度耳轴的镜像对称把后半球折到前半球 `ITD=sign(phi)*a/c*((PI-|phi|)+sin(PI-|phi|))`，使 ITD 由正前 0 升至 `+-90` 度峰值再在正后 `+-180` 度回落到 0（契合对称球的前后混淆），峰值 `max_itd=a/c*(PI/2+1)`；**Kuhn 渐近模型**（频率相关）低频（约 <500 Hz）相位延迟 `itd_low=3*a*sin(phi)/c`、高频（约 >3 kHz）群延迟 `itd_high=2*a*sin(phi)/c`，二者在 `sin(phi)` 上为奇函数故天然前后对称且正前/正后为零；常量 `DEFAULT_HEAD_RADIUS_M=0.0875`（8.75 cm 平均头半径）/`DEFAULT_SPEED_OF_SOUND=343`/`KUHN_LOW_FACTOR=3`/`KUHN_HIGH_FACTOR=2`；私有 `radius_over_speed()` 在 `a<=0||c<=0` 或非有限时返回 0、`finite` 守卫非有限方位、`wrap_pi(phi)=phi-TAU*round(phi/TAU)`（全走 `bevy_math::ops::sin/round`）；与既有 `spatial_impression::interaural_cross_correlation`（从一对双耳脉冲响应**实测** IACC 相似度而非预测时差）、`panner`（**幅度**平移律而非时差）严格区分不重复，`# Relationship` 已澄清——本 ITD 预测可驱动分数样本延迟线做双耳渲染、与别处幅度线索互补；结构体 `InterauralTimeDifference{head_radius_m,speed_of_sound}`（私有字段 + `new`/`default`/`head_radius_m()`/`speed_of_sound()` 访问器，Debug/Clone/Copy/PartialEq+serde `serialize` 门控）+方法 `woodworth_itd(azimuth_rad)`/`kuhn_itd_low(azimuth_rad)`/`kuhn_itd_high(azimuth_rad)`/`max_itd()`；控制率/音频率均安全（纯标量、零堆分配/无锁/不 panic，可逐源逐块乃至逐样本调用）；17 golden 对拍（正前三族皆 0/Woodworth 在 `+-90` 度达 `max_itd`/Woodworth 奇函数/正后 `+-180` 度回零/前后对称 60 度与 120 度同幅/前象限单调增/全域幅值被峰值界住/Kuhn 低高频 90 度闭式精确/Kuhn 低/高比值恒 1.5/Kuhn 奇函数/Kuhn 正后回零/`max_itd` 闭式且落在 0.6-0.8 ms 人类区间/方位整圈 TAU 环绕一致/非有限输入安全归零/退化几何 c<=0 或 a<=0 或负半径归零/访问器与 Default）+1 doctest（默认头正前 0、`+-90` 度达 `max_itd` 且两侧互为相反数）；字母序 `pub mod interaural_time_difference;` 插在 `initial_time_delay_gap` 与 `late_lateral_sound_level` 之间，lib.rs re-export `DEFAULT_HEAD_RADIUS_M/DEFAULT_SPEED_OF_SOUND/InterauralTimeDifference/KUHN_HIGH_FACTOR/KUHN_LOW_FACTOR`、re-export 前核对确认与 `doppler::SPEED_OF_SOUND_MPS` 等无 E0252；spatial 累计 **677 单测 + 52 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`interaural_time_difference.rs` 395 行 + `lib.rs` +5，commit b179b4814，经全绿验证 + 逐行 CR：Woodworth 前区/后区折叠与符号、`max_itd` 峰值、Kuhn 低高频因子、`wrap_pi` 环绕、`radius_over_speed` 退化守卫、与 spatial_impression/panner 的实测/幅度区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.85（**本版新增**：M1 效果层 `prism_audio_core` 新增 **Haas 立体声展宽器 `HaasWidenerNode`/`HaasWidenerParams`（`nodes/effects/haas_widener`）**（借鉴 Helmut Haas 1949 年量化的优先效应/先至波阵面定位这一教科书级心理声学与经典 DSP 思想但不抄任何源码——不同于「抬高侧信号幅度」的展宽，本节点仅对 Mid-Side 分解后的**侧信号 S 施加几毫秒的短时延**，令左右声道在**时间上去相关**从而拓宽声像而不改变频谱平衡：信号在 M/S 域处理 `M=(L+R)/2`、`S=(L-R)/2`、`S'=side_level*lerp(S, delay(S,t), width)`、`L=M+S'`、`R=M-S'`——因延迟只作用于侧信号故中信号与单声道和 `L+R=2M` 完美时间对齐、**单声道兼容**（下混无梳状滤波或抵消），`width=0` 逐位复现输入（纯直达侧信号）、增大 `width` 渐入时延侧信号获更强去相关；侧信号延迟经共享单声道 `ring`（长度 `ring_len=max_delay+2`）实现，分数延迟由线性插值 `lerp` 完成（读位 `read_pos=w-delay`、`i0=floor(read_pos).rem_euclid(len)`、`i1=i0+1`、全走 `bevy_math::ops::floor`），超出前置立体声对的任何声道原样透传；每参数均走 `Smoothed` 实现 click-free 自动化（Haas 延迟为**固定**值仅平滑避 zipper，绝不像 LFO 周期扫动），`finite` 守卫非有限参数、`flush_denormal` 守卫反规格化；与既有 `stereo_width`（缩放侧**幅度**的静态增益、从不引入时差）、`mid_side_matrix`（纯 M/S 编解码+trim 增益、无延迟）、`chorus`/`flanger`/`vibrato`（抽头被 LFO 周期扫动做移调）严格区分不重复，`# Relationship` 已澄清展宽/延迟族各节点意图之别；构造期按 `max_delay_frames` 预分配单条 mono ring，热路径零堆分配/无锁/不 panic（逐帧 M/S 分解、侧信号分数读插值、写回、重建 L/R），`set_params` 保留 ring 与平滑状态实现 click-free 重设、`reset` 清零 ring+写头并以当前 target 重建全部 `Smoothed`；导出常量 `MAX_WIDTH=1`/`MAX_SIDE_LEVEL=2`/`DEFAULT_DELAY_MS=12`/`DEFAULT_WIDTH=0.5`/`DEFAULT_SIDE_LEVEL=1`、`HaasWidenerParams{delay_ms,width,side_level}`（Copy+Default+serde `serialize` 门控）、图节点 `HaasWidenerNode::{new(sample_rate,max_delay_frames,params),max_delay_frames,params,set_params,reset}`+impl `AudioNode`；14 golden 对拍 + 1 doctest（硬左瞬态获时延右声道回声且单声道和守恒）；字母序 `pub mod haas_widener;` 插在 `graphic_eq` 与 `mid_side_matrix` 之间（effects/mod.rs 同序只 re-export `HaasWidenerNode/HaasWidenerParams`，**因 `MAX_WIDTH` 已被 `stereo_width` re-export 故本模块的 `MAX_WIDTH`/`MAX_SIDE_LEVEL` 均不再 re-export 以避 E0252**，catalogue 一条用 "--"），re-export 前核对确认命名冲突；core 累计 **617 单测 + 28 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`haas_widener.rs` 555 行 + `effects/mod.rs` +7，commit d8d19c0bf，经全绿验证 + 逐行 CR：M/S 分解与重建、侧信号分数延迟插值、单声道兼容性、`width`/`side_level` 钳位、`finite`/`flush_denormal` 守卫、与 stereo_width/mid_side_matrix/chorus 族的意图区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.84（**本版新增**：M1 效果层 `prism_audio_core` 新增 **多抽头延迟 `MultiTapDelayNode`/`MultiTapDelayParams`/`TapSpec`（`nodes/effects/multi_tap_delay`）**（借鉴多抽头延迟这一教科书级经典 DSP 思想但不抄任何源码——同一条延迟线被多个独立定时的抽头同时读取：延迟线本身为**单声道**（输入各通道求和为一条共享历史缓冲 `ring`，长度 `ring_len=max_delay+2`），最多 `MAX_TAPS=8` 枚抽头各以独立的 `delay_seconds`/`gain`/`pan` 读取同一历史、分数延迟由线性插值 `lerp` 实现（读位 `read_pos=w-delay`、`i0=floor(read_pos).rem_euclid(len)`、`i1=i0+1`、`frac=read_pos-floor`，全走 `bevy_math::ops::floor`），每抽头经 `equal_power_pan(pan)` 等功率平移烙入立体声左右，单枚全局反馈系数 `feedback`（钳 `[0, MAX_FEEDBACK=0.999]`）把抽头求和 `tap_sum` 重注入写头 `ring[w]=flush_denormal(mono_in+feedback*tap_sum)` 以维持节奏图案；干声按通道 `dry*input[ch]` 保留故未处理像不塌陷，湿声多抽头形成相干立体声图案（这是 send/pattern 延迟的标准架构）；每参数均走 `Smoothed` 实现 click-free 自动化，`finite` 守卫非有限参数以防污染反馈环；与既有 `delay`（单条**逐通道** ring、单枚分数抽头+反馈的回声/slap-back 基元）、`chorus`/`flanger`（抽头被 LFO 周期扫动做移调加厚）严格区分不重复——本模块抽头时刻固定（仅平滑避 zipper），`# Relationship` 已澄清延迟族各节点意图之别；构造期按 `max_delay_frames` 预分配单条 mono ring，热路径零堆分配/无锁/不 panic（逐帧 mono-sum 输入、逐抽头读插值累加、单次写回、立体声左右或单声道混干湿），`set_params` 保留 ring 与平滑状态实现 click-free 重设、`reset` 清零 ring+写头并以当前 target 重建全部 `Smoothed`；导出常量 `MAX_TAPS=8`/`MAX_FEEDBACK=0.999`/`DEFAULT_TAP_COUNT=3`/`DEFAULT_WET=0.5`/`DEFAULT_DRY=1`/`DEFAULT_FEEDBACK=0`、`TapSpec{delay_seconds,gain,pan}`（Copy+Default+serde `serialize` 门控）、`MultiTapDelayParams{taps:[TapSpec;MAX_TAPS],active_taps,feedback,wet,dry}`（Default 为跨立体声场渐衰三抽头 125/250/375 ms）、图节点 `MultiTapDelayNode::{new(sample_rate,max_delay_frames,params),max_delay_frames,active_taps,params,set_params,reset}`+impl `AudioNode`；16 golden 对拍（wet=0 干声逐位透传/单整数抽头把脉冲搬到第 5 帧且他处静默/双抽头产两回声（3 帧 1.0 与 7 帧 0.5）/4.5 帧分数抽头半分到第 4、5 帧/硬左平移右声道静默且左声道可闻/中心平移左右各 `FRAC_1_SQRT_2` 等功率/反馈以 0.5 逐 4 帧递归衰减回声（4/8/12 帧 1.0/0.5/0.25）/反馈钳到 0.999/`active_taps` 钳到 `MAX_TAPS`/`reset` 后静默/`reset` 后同输入逐位复现/参数往返/零帧安全/非有限参数输出仍有限/`set_params` 更新 target/静默输入恒静默）+1 doctest（单抽头 5 帧回搬脉冲）；字母序 `pub mod multi_tap_delay;` 插在 `mid_side_matrix` 与 `parametric_eq` 之间（effects/mod.rs 同序 re-export `MAX_TAPS/MultiTapDelayNode/MultiTapDelayParams/TapSpec`，**因 `MAX_FEEDBACK` 已被 `comb_resonator` re-export 故本模块的 `MAX_FEEDBACK` 不再 re-export 以避 E0252**，catalogue 一条用 "--"），re-export 前 grep 确认 `MAX_TAPS/TapSpec/MultiTapDelay*` 均无 E0252；core 累计 **603 单测 + 27 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`multi_tap_delay.rs` 686 行 + `effects/mod.rs` +9，commit 7f6953901，经全绿验证 + 逐行 CR：分数抽头插值、等功率平移、单声道共享延迟线架构、反馈重注入、`finite` 守卫、与 delay/chorus/flanger 的区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.83（**本版新增**：M1 效果层 `prism_audio_core` 新增 **通道声码器 `Vocoder`/`VocoderNode`（`nodes/effects/vocoder`）**（借鉴 Dudley 1930 年代通道声码器这一教科书级经典 DSP 思想但不抄任何源码——交叉合成：以 `VOCODER_BANDS=16` 枚对数间隔的恒定 Q 带通谐振器复用共享 `Biquad` 核分析调制信号（modulator，典型为人声）的各频带能量，每带配一枚线性域包络跟随器量测该带能量包络；同一组带通把载波（carrier，典型为合成器）切分进相同频带，各带以调制信号对应带的包络缩放后求和，故载波「说出」调制信号移动的共振峰；带中心按 `f[i]=low*ratio^i`（`ratio=(high/low)^(1/(N-1))`）对数排布、恒定 Q 由几何带距闭式 `Q=sqrt(ratio)/(ratio-1)` 给定（几何带缘宽度 `f*(ratio-1)/sqrt(ratio)`），范围经 `sanitize_range` 钳到 `MIN_LOW_HZ=20<=low<high<=0.49*sr` 且 `high>=2*low` 保设计良态；包络跟随器为线性域 attack/release 一支单极点——rect=`|x|`、`coef=rect>env?attack:release`、`env=coef*env+(1-coef)*rect`，attack/release 一支单极点系数复用 `dynamics::detector::time_to_coef`（`exp(-1/(t*sr))`，全走 `bevy_math::ops`）；调制信号读 input 0、载波读 input 1（镜像 `dynamics::ducking` 的旁链约定，用 `io.split()`），**未接载波时输出静音**；与既有 `formant_filter`（并联带通但烙印**固定预设**元音共振峰、带电平为静态元音表常量、单输入）、`ring_modulator`（单枚双极性乘法无滤波）、`auto_wah`（单枚扫动带）、`dynamics::multiband`（把**单**信号切带各自压缩、从不交叉合成两路信号）严格区分不重复，`# Relationship` 已澄清意图与信号流之别：声码器烙印的是实时**量测的时变**调制包络、带电平随调制信号移动且需两路输入；构造期按 `max_frames` 预分配两组带通 bank（各 `channels` 宽）、每带每通道包络记忆 `Vec<Sample>`（`band*channels+ch` 索引）、两枚求和用 scratch `AudioBuffer`（layout 随通道数），`process_into` 热路径零堆分配/无锁/不 panic（逐带 copy 调制/载波入 scratch、`process_inplace` 带通、逐样本包络跟随并以包络缩放载波带累加入 output，末端施加输出增益），`set_params` 保留滤波器与包络状态实现 click-free 重设计、`reset` 清零两组 bank + 全部包络 + scratch；导出常量 `VOCODER_BANDS=16`/`DEFAULT_LOW_HZ=80`/`DEFAULT_HIGH_HZ=12000`/`DEFAULT_ATTACK_MS=2`/`DEFAULT_RELEASE_MS=15`/`DEFAULT_OUTPUT_GAIN_DB=0`、`VocoderParams{low_hz,high_hz,attack_ms,release_ms,output_gain_db}`（Copy+Default+serde `serialize` 门控）、可嵌入 DSP 核 `Vocoder::{new(sr,channels,max_frames,params),params,bands,set_params,process_into,reset}`、图节点 `VocoderNode::{new,params,bands,set_params}`+impl `AudioNode`；14 golden 对拍（静音调制信号出静音/静音载波出静音/缺载波输入出静音/1 kHz 调制信号把载波 1 kHz 区段（测试内 Goertzel 跨 24 子带积分抑噪）能量放行显著强于 8 kHz 静区（>2 倍）/低频 vs 高频调制信号在 300 Hz 响应有别（频率跟踪）/输出增益逐样本线性缩放（最大误差 <1e-3）/立体声两通道逐位一致/`reset` 后同输入逐位复现/`set_params` 更新参数/零帧安全/参数往返/嵌入核与节点逐位一致/带数上报 16/带中心单调递增且落在范围内）+1 doctest（噪声/斜坡载波被方波调制信号整形、输出有限）；字母序 `pub mod vocoder;` 插在 `vibrato` 与 `waveshaper` 之间（effects/mod.rs 同序 re-export `Vocoder/VocoderNode/VocoderParams` + catalogue 一条用 "--"），re-export 前 grep 确认 `Vocoder/VocoderNode/VocoderParams/VOCODER_BANDS` 均无 E0252；CR 期修复 `imprints_modulator_band_onto_carrier` 单 bin 噪声方差过大问题（改为跨 850-1150 Hz 与 7000-9000 Hz 各 24 子带 Goertzel 积分能量比较）；core 累计 **587 单测 + 26 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`vocoder.rs` 692 行 + `effects/mod.rs` +7，commit b35379c7c，经全绿验证 + 逐行 CR：对数带中心/恒定 Q 闭式、线性域 attack/release 包络跟随、旁链双输入 `io.split()` 约定、缺载波静音、与 formant_filter/ring_modulator/auto_wah/multiband 的区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.82（**本版新增**：M1 效果层 `prism_audio_core` 新增 **元音共振峰滤波器 `FormantFilter`/`FormantFilterNode`（`nodes/effects/formant_filter`）**（借鉴 talk-box／声码器邻域「以数枚并联带通谐振器把元音共振峰烙印到任意激励信号上」这一经典 DSP 思想但不抄任何源码：人声元音由声道的若干共振峰频率区分，本节点以 `FORMANT_COUNT=5` 枚**并联** RBJ `BiquadKind::BandPass` 谐振器复用共享 `Biquad` 核，每枚锚定一个共振峰的中心频率并以其带宽决定 `Q=(freq/bandwidth)*resonance`、以其电平线性缩放（`db_to_linear(gain_db)`）后**求和**（非串联），末端乘输出补偿增益 `output_gain_db`；支持两元音连续形变——单一 `morph∈[0,1]` 对 `vowel_a` 与 `vowel_b` 每个共振峰的 freq/bandwidth/gain_db 做线性插值，`morph==0` 恰为 `vowel_a`、`morph==1` 恰为 `vowel_b`；五枚基本元音 `Vowel{A,E,I,O,U}`（Default=A）各自 `const fn formants()->[FormantSpec;5]` 内嵌已发表的（典型男低音）声学语音学共振峰频率/带宽/电平数据表——这些是数值常量而非代码；与既有 `parametric_eq`／`graphic_eq`（皆**串联**同一 `Biquad` 构件以塑形整体响应、带由用户自由选定或固定 ISO 栅格）、`auto_wah`（以包络/LFO **扫动单一**带通共振峰）严格区分不重复：本节点是**并联求和**且各带由元音预设驱动、唯一的运动是元音到元音的 `morph`，`# Relationship` 已澄清三者拓扑与意图之别；构造期按 `max_frames` 预分配求和用 scratch `AudioBuffer`（layout 随通道数，多声道共用，仿 multiband 范式）与全部滤波器状态，`process_into` 热路径零堆分配/无锁/不 panic（`output.clear()` 后对每枚共振峰 copy 干信号入 scratch、`process_inplace` 带通、`dst[f]+=src[f]*level` 累加，末端施加输出增益），`set_params` 保留滤波器状态实现 click-free 重设计、`reset` 清零全部谐振器记忆与 scratch；导出常量 `FORMANT_COUNT=5`/`DEFAULT_RESONANCE=1`/`DEFAULT_MORPH=0`/`DEFAULT_OUTPUT_GAIN_DB=0`、`FormantSpec{freq_hz,bandwidth_hz,gain_db}`、`Vowel{A,E,I,O,U}`（Default=A，`const fn formants()`）、`FormantFilterParams{vowel_a,vowel_b,morph,resonance,output_gain_db}`（Copy+Default+serde `serialize` 门控）、可嵌入 DSP 核 `FormantFilter::{new(sr,channels,max_frames,params),params,set_params,process_into,reset}`、图节点 `FormantFilterNode::{new,params,set_params}`+impl `AudioNode`；14 golden 对拍（静音入静音出/元音 A 在 F1≈600 Hz 处能量显著高于非共振区 5 kHz（测试内 Goertzel 探能）/不同元音（A vs I）在 600 Hz 响应有别/`morph==0` 逐位匹配 vowel_a/`morph==1` 逐位匹配 vowel_b/`morph` 中间值响应非零/立体声两通道逐位一致/输出增益线性缩放/`reset` 后同脉冲逐位复现/`set_params` 保状态/零帧安全/参数往返/嵌入核与节点逐位一致/Default 元音为 A）+1 doctest（把元音 A 烙印到宽带脉冲、响应非平凡）；字母序 `pub mod formant_filter;` 插在 `flanger` 与 `frequency_shifter` 之间（effects/mod.rs 同序 re-export `FormantFilter/FormantFilterNode/FormantFilterParams/FormantSpec/Vowel` + catalogue 一条用 "--"），re-export 前 grep 确认 `FormantFilter/FormantSpec/Vowel/FORMANT_COUNT/DEFAULT_RESONANCE/DEFAULT_MORPH` 均无 E0252；CR 修复构造期冗余 scratch 二次赋值（折叠为单次 `layout_for(channels)` 构造）与 `formant_coeffs` 重复调用（`core::array::from_fn` 内一次拿 coeffs+level 顺带填 levels 数组），`layout_for` 合并 `2=>Stereo` 与 `_=>Stereo` 规避 match_same_arms；core 累计 **573 单测 + 25 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`formant_filter.rs` 694 行 + `effects/mod.rs` +8，commit abb2cfdbf，经全绿验证 + 逐行 CR：并联带通求和拓扑、元音共振峰数据表、morph 线性插值端点逐位匹配、click-free 保状态、与 parametric_eq/graphic_eq/auto_wah 的拓扑与意图区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.81（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **Peutz 辅音清晰度损失 `ArticulationLoss`（`%ALcons`，`articulation_loss`）**（借鉴 Peutz-Klein 公开发表的「从房间几何统计预测语言可懂度」这一经典声学思想但不抄任何源码：令听者距离 `r` 米、房间体积 `V` 立方米、中频混响时间 `RT` 秒（通常取 500 Hz 与 1000 Hz 倍频带平均）、声源指向性因子 `Q`（全指向=1），混响半径（临界距离）`r_c=PEUTZ_CRITICAL_DISTANCE_CONSTANT*sqrt(Q*V/RT)`，复用与 `room_acoustics::critical_distance` 相同的 `0.057` 系数但**扩展加入指向性因子 Q**；Peutz 经验公式以 `R_LIMIT=3.16` 临界距离边界分两段——距离主导区 `r<=R_LIMIT*r_c` 时 `%ALcons=200*r^2*RT^2/(V*Q)`、混响饱和区 `r>R_LIMIT*r_c` 时平台 `%ALcons=9*RT`，结果钳 `[0,MAX_ALCONS=100]`；可选便捷映射 `alcons_to_sti` 以公开经验关系 `STI=0.9482-0.1845*ln(%ALcons)`（无 log10 用 `ops::ln` 自然对数）钳 `[0,1]`、非正或非有限损失返回 1（零损失即完美可懂）；与既有 `speech_transmission_index`（STI/RASTI）严格区分不重复：后者是从录制脉冲响应的**调制传递函数 MTF** 路径量测可懂度，本模块是从 `V/RT/r/Q` 几何统计**解析预测**，二者是对同一感知概念的两条独立建模路径、输入与公式皆不同、互不复制，`alcons_to_sti` 仅为公开经验换算与 MTF 计算无关；`# Relationship` 并澄清 `room_acoustics` 提供 Sabine 混响时间与无 Q 临界距离作为输入来源但不复用其缓存状态——含 Q 的临界距离作**私有 helper `peutz_critical_distance_m`（不导出）**以免与公开 `room_acoustics::critical_distance`（无 Q）命名/DRY 冲突；控制速率离线解析估计器（非音频线程、零堆分配、不 panic），退化输入（`V<=MIN_DIVISOR`/`RT<=0`/`Q<=MIN_DIVISOR`/`r<0`/非有限）→安全哨兵 `MAX_ALCONS`（最差可懂度），中间量 f64 累加读回 `as Sample`、`sqrt`/`ln` 全走 `bevy_math::ops`；导出常量 `R_LIMIT=3.16`/`MAX_ALCONS=100`/`PEUTZ_CRITICAL_DISTANCE_CONSTANT=0.057`、`articulation_loss_percent(distance_m,volume_m3,reverberation_time_s,directivity_q)->Sample`、`alcons_to_sti(alcons_percent)->Sample`、`ArticulationLoss{alcons_percent,equivalent_sti,is_distance_limited}`+`from_room(...)`（Debug/Clone/Copy/PartialEq/Default+serde `serialize` 门控）；16 golden 对拍（已知手算 `%ALcons=5`/近场低损/远场饱和平台 `9*RT`/距离加倍损失四倍/边界 `is_distance_limited` 翻转/非正体积哨兵/非正 RT 哨兵/非正 Q 哨兵/负距离哨兵/非有限输入安全/`alcons_to_sti` 单调递减且钳位/临界距离随 V 增 RT 减/`from_room` 与自由函数一致/Default 为零/常量稳定/结果钳位区间）+1 doctest（5 m 全指向 1000 m^3 1 s 房间 `%ALcons=5` 且距离受限、等效 STI 落在开区间）；字母序 `pub mod articulation_loss;` 插在 `ambisonics` 之后、`attenuation` 之前（第 53 行），lib.rs re-export `ArticulationLoss/MAX_ALCONS/PEUTZ_CRITICAL_DISTANCE_CONSTANT/R_LIMIT/alcons_to_sti/articulation_loss_percent`、第二次 grep 确认无 E0252；spatial 累计 **660 单测 + 51 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，由 Bernoulli 子 agent 自提精确 commit（`articulation_loss.rs` 367 行 + `lib.rs` +5，commit eb8c13d61，经主 agent 独立全绿验证 + 逐行 CR：两段式 Peutz 公式与边界、含 Q 临界距离、各退化哨兵、`alcons_to_sti` 自然对数映射、与 STI 的 MTF 路径区分、私有临界距离 helper 规避与 room_acoustics DRY 冲突均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.80（**本版新增**：M1 效果层 `prism_audio_core` 新增 **单旋钮频谱倾斜均衡器 `TiltEq`/`TiltEqNode`（`nodes/effects/tilt_eq`）**（借鉴母带/音色塑形中「以单一控制量围绕支点频率旋转整条频谱」这一经典 DSP 思想但不抄任何源码：以 pivot 频率为支点，低端与高端反向移动——实现为两枚级联的 RBJ 架子滤波器复用共享 `Biquad` 核：低架子 `low-shelf` 增益取 `-tilt_db`、高架子 `high-shelf` 增益取 `+tilt_db`，故正 `tilt_db` 提亮（高频升、低频降）、负 `tilt_db` 变暖（低频升、高频降）、`tilt_db==0` 为平直透传；斜率陡度由共享架子品质因数 `slope_q` 决定（默认 `DEFAULT_SLOPE_Q=FRAC_1_SQRT_2≈0.707` 即 Butterworth S=1）；**关键数值细节：`tilt_db==0` 时折叠为手构单位系数 `{b0:1.0, 其余:0}` 以保 bit-exact 透传**——因 RBJ 架子设计末尾乘 `inv_a0=1.0/a0` 的倒数舍入使 0 dB 架子并非 bit-exact 单位（与 graphic_eq 同类坑），私有 helper `shelf_coeffs` 在 `gain_db==0.0` 时返回 `unit_coeffs()`、否则 `BiquadCoeffs::design(LowShelf/HighShelf,...)`；与既有 `parametric_eq`（任意数量独立可调带：钟形/架子/通滤波，各自频率/Q/增益）、`graphic_eq`（固定 ISO 栅格峰值带）严格区分不重复：三者皆级联同一 `Biquad` 构件但建模不同用户意图——本节点只暴露唯一一个倾斜量作用于锚定共享 pivot 的匹配低/高架子对，从不单独暴露两枚架子增益，`# Relationship` 已澄清；构造期预分配全部滤波器状态，`process` 热路径零堆分配/无锁/不 panic（`output.copy_from(input)` 后低架子、高架子依次 `process_inplace`），`set_params` 保留滤波器状态实现 click-free 重设计，`reset` 清零两枚架子记忆；导出常量 `DEFAULT_PIVOT_HZ=1000`/`DEFAULT_SLOPE_Q=FRAC_1_SQRT_2`/`DEFAULT_TILT_DB=0`、`TiltEqParams{pivot_hz,tilt_db,slope_q}`（Copy+Default+serde `serialize` 门控）、可嵌入 DSP 核 `TiltEq::{new,params,channels,set_params,process_inplace,reset}`、图节点 `TiltEqNode::{new,params,set_params}`+impl `AudioNode`；12 golden 对拍（静音入静音出/`tilt_db==0` bit-exact 透传/默认参数平直/正 tilt 低端 DC 增益降至单位以下/负 tilt 低端 DC 增益升至单位以上/正负 tilt 跨单位相反/立体声两通道逐位一致/`set_params` 保状态/`reset` 后同脉冲逐位复现/零帧安全/参数往返/嵌入核与节点逐位一致）+1 doctest（提亮母带 tilt 的非平凡响应）；字母序 `pub mod tilt_eq;` 插在 `tape` 与 `tremolo` 之间（effects/mod.rs 同序 re-export `TiltEq/TiltEqNode/TiltEqParams` + catalogue 一条用 "--"），re-export 前 grep 确认 `TiltEq/DEFAULT_PIVOT/DEFAULT_SLOPE/DEFAULT_TILT` 均无 E0252；core 累计 **559 单测 + 24 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`tilt_eq.rs` 497 行 + `effects/mod.rs` +7，commit 31ec91330，经全绿验证 + 逐行 CR：匹配低/高架子增益符号、0 dB 折叠单位系数 bit-exact 透传、正负 tilt 的 DC 增益方向、click-free 保状态、与 parametric_eq/graphic_eq 的意图区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.79（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **Abel-Huang 归一化回声密度剖面 `EchoDensityProfile`（`NEDP`，`echo_density`）与实测混合时间**（借鉴 Abel-Huang 公开发表的归一化回声密度剖面这一经典 DSP 思想但不抄任何源码：令 `h[n]` 为采样率 `sample_rate` Hz 的宽带房间冲激响应，围绕每个样本索引 `n` 取半宽 `w=round(ECHO_DENSITY_WINDOW_MS/1000*sample_rate)`（钳 `>=1`，非有限 sr 回退 1，`ECHO_DENSITY_WINDOW_MS=10` 即约 20 ms 窗）的对称窗 `[n-w,n+w]`（saturating 边界 + `.min(len)`），零均值假设下窗内标准差 `sigma=sqrt((1/(2w+1))*sum h[m]^2)`，归一化回声密度 `eta[n]=(1/(C*(2w+1)))*#{|h[m]|>sigma}`，其中高斯参考比例 `C=erfc(1/sqrt(2))=0.317_310_5`（标准正态样本落在正负一倍标准差之外的概率，以字面量常量 `GAUSSIAN_EXCEEDANCE` 供给、**绝不运行时计算 erf**）；稀疏早期尾 `eta` 近 0、充分扩散的高斯尾 `eta` 趋近 1，实测混合时间为 `eta` 首达阈值 `MIXING_THRESHOLD=1.0` 的样本时刻折算 ms；与既有 `diffusion_field` 严格区分不重复：后者是**从房间体积出发的理论**预测（`N(t)=(4/3)*PI*c^3*t^3/V` 与理论混合时间，持有 `MAX_ECHO_DENSITY` 常量及 `echo_density()`/`mixing_time_ms()` 方法），本模块是**从实测 RIR 逐窗量测的经验值**——此「实测 vs 理论」关系与 `direct_to_reverberant_ratio`（实测）之于 `reverberant_field`（统计理论）完全平行，`# Relationship` 已澄清、二者永不互相复制、第二次 grep 排除 diffusion_field 后对所有新名字返回空确认无 E0252；控制速率离线估计器（非音频线程、剖面变体写入调用方 `&mut [Sample]` 缓冲故零堆分配、不 panic），空/全零/非有限样本/sr 非有限或 `<=0` 等退化输入→安全哨兵 `NO_MIXING_TIME_MS=-1` + 全零剖面 + `converged=false`，非有限样本经 `finite`/`finite_abs` 归零、能量用 f64 累加器、`sigma<=0` 与越界守卫均加注释规避 `neg_cmp_op_on_partial_ord`、`normalized_echo_density` 用 `iter_mut().enumerate()`、`mixing_time_ms` 的 `for n in 0..len` 中 `n` 仅作中心参数与时间计算不索引 ir 故不触发 `needless_range_loop`；导出常量 `ECHO_DENSITY_WINDOW_MS=10`/`GAUSSIAN_EXCEEDANCE=0.317_310_5`/`MIXING_THRESHOLD=1`/`NO_MIXING_TIME_MS=-1`、`normalized_echo_density(ir,sample_rate,out:&mut [Sample])`（越界槽零填、退化全零、零堆分配）、`mixing_time_ms(ir,sample_rate)->Sample`（单遍扫描、收敛即早返）、`EchoDensityProfile{mixing_time_ms,final_density,converged}`+`from_impulse_response(ir,sample_rate)`（Debug/Clone/Copy/PartialEq/Default+serde `serialize` 门控）；15 golden 对拍（稠密尾收敛且混合时间非负/高斯尾密度近 1/稀疏反射低密度/空响应哨兵/全零响应哨兵/零采样率哨兵/NaN 采样率哨兵/非有限样本安全/`from_impulse_response` 与自由函数一致/剖面值有限非负/越界槽零填/退化剖面全零/早到的稠密尾混合更早/Default 为零/常量稳定）+1 doctest（伪随机稠密尾收敛且终值 `>0`）；字母序 `pub mod echo_density;` 置于第 63 行（`echo_criterion` 之后、`geometry` 之前，`echo_c`<`echo_d`），lib.rs 同序 re-export；spatial 累计 **644 单测 + 50 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`echo_density.rs` 443 行 + spatial `lib.rs` +5，commit e5b3ed010，经主 agent 独立全绿验证 + 逐行 CR：窗内 sigma/超越计数/高斯参考比例归一、各退化哨兵、`sigma<=0` 与能量 `<=` 守卫规避 neg_cmp、与理论 diffusion_field 的实测/理论区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.78（**本版新增**：M1 分析层 `prism_audio_core` 新增 **YIN 单音基频估计器 `PitchDetector`/`PitchDetectorNode`（`nodes/analysis/pitch_detector`）**（借鉴 de Cheveigne & Kawahara 2002 公开发表的 `YIN` 时域差分函数法「以累积均值归一化差分规避原始自相关峰拾取的八度误差」这一经典 DSP 思想但不抄任何源码：对长度 `window+tau_max` 的时序帧计算平方差分函数 `d(tau)=sum_{j<window}(x[j]-x[j+tau])^2`（随 `tau` 等于信号周期时趋零），再除以运行均值得累积均值归一化差分 `d'(0)=1`、`d'(tau)=d(tau)*tau/sum_{k<=tau}d(k)` 以消除对 `tau=0` 的偏置；从 `tau_min` 起步取首个 `d'<threshold` 且处局部极小的滞后，无合格者回退到带内 `d'` 全局最小；对所选滞后三点作抛物线插值细化到亚样本分辨率得 `refined`，`f0=sample_rate/refined`、周期置信度 `confidence=1-d'`、`is_voiced=d'<=threshold`；频带 `[min_hz,max_hz]` 换算为滞后界 `tau_min=round(sr/max_hz)`（钳 `>=MIN_TAU=2`）、`tau_max=round(sr/min_hz)`（钳 `>=tau_min+2`），`window=tau_max`、`ring_len=window+tau_max`；仿既有 `analysis/spectrum` 的 ring+hop 范式——`feed_sample(x)` 非有限归 0 入环形缓冲、每 `hop` 样本载时序帧跑一次完整 `YIN` 全程，静音帧（逐样本均方 `<=SILENCE_ENERGY=1e-10`）判 unvoiced；与既有 `spectrum`（窗口 FFT 全谱幅度，报能量在各频率的分布）严格区分不重复：本模块把单音信号凝练为**单一 `f0`+置信度**服务「这是什么音？」而非全谱视图，`# Relationship` 已澄清二者皆透传信号且预分配工作缓冲；所有 scratch（时序帧副本、两个 `tau_max+1` 长 f64 差分数组、环形缓冲）均在 `new` 一次性分配，热路径 `feed_sample`/`process` 无分配/无锁/不 panic、单次全程 `O(tau_max*window)`、差分与归一化用 f64 累加器、非有限输入作静音故环形缓冲永不被 `NaN` 毒化；`run_yin` 8 参自由函数 `#[expect(too_many_arguments)]`、差分内积用 `.iter().zip(&frame[tau..tau+window])` 规避 needless_range_loop、全局最小回退用 `.iter().enumerate().take().skip()`、`is_voiced=d'<=threshold` 的 `<=` 守卫规避 neg_cmp_op_on_partial_ord；导出常量 `DEFAULT_YIN_THRESHOLD=0.15`/`DEFAULT_MIN_HZ=50`/`DEFAULT_MAX_HZ=2000`/`DEFAULT_HOP=256`、`PitchEstimate{frequency_hz,confidence,is_voiced}`（Copy+Default+serde 门控）、`PitchDetector`（`new(sample_rate,min_hz,max_hz,hop)`+`sample_rate/hop/min_frequency_hz/max_frequency_hz/threshold/set_threshold/latest/frames_computed/reset/feed_sample`）、`PitchDetectorNode`（透传 tap，首通道喂检测器、`new/detector/detector_mut`+impl AudioNode）；14 golden 对拍（440 Hz 正弦判 voiced 且 `|f0-440|<5` 置信度 `>0.8`/低音 80 Hz 与高音 1500 Hz 均正确/静音 unvoiced 且 `f0=0`/白噪置信度 `<0.95`/220 Hz 加二三次谐波仍锁基频而非八度/频带越界重排与钳位/`hop>=1`/阈值钳 `(0,1)`/`frames_computed` 随 hop 计数/reset 清态/非有限输入安全/节点逐通道透传/Default 为 unvoiced/抛物线细化不越一样本）+1 doctest（八块 512 帧 440 Hz 正弦读得接近 440 Hz）；re-export 仅 `PitchDetector/PitchDetectorNode/PitchEstimate`（**刻意不 re-export `DEFAULT_HOP` 以避与 `spectrum::DEFAULT_HOP` 的 E0252 撞名**，常量经模块完整路径访问）；字母序 `pub mod pitch_detector;` 插在 `loudness` 与 `spectrum` 之间（analysis/mod.rs 同序 re-export + catalogue 一条，新增行用 `--` 保 ASCII）；core 累计 **547 单测 + 23 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`nodes/analysis/pitch_detector.rs` 688 行 + `nodes/analysis/mod.rs` +7，commit e21f81690，经全绿验证 + 逐行 CR：差分/CMND 归一化正确性、绝对阈值+局部极小搜索与全局最小回退、抛物线亚样本细化钳位、静音/噪声/谐波/退化守卫、int_plus_one 与 needless_range_loop 两处 clippy 修复均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.77（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **Zahorik 直达-混响声能比 `DirectToReverberantRatio`（`DRR`，`direct_to_reverberant_ratio`）**（借鉴听觉距离感知研究对「直达声能与其后全部到达声能之比是首要距离线索」的工程共识但不抄任何源码：从实测房间冲激响应 `p[n]` 以 `|p|` 全局峰值首个索引 `n_d` 定位直达声，围绕 `n_d` 取半宽 `w=round(DIRECT_WINDOW_MS/1000*sample_rate)` 的窄对称窗 `[n_d-w, n_d+w]`（闭区间，`DIRECT_WINDOW_MS=2.5`）隔离直达路径——`E_direct=窗内平方和`、`E_total=全体平方和`、`E_reverberant=E_total-E_direct`，`DRR=10*log10(E_direct/E_reverberant)` dB 钳 `[-MAX_DRR_DB,MAX_DRR_DB]`；无 `log10` 以 `10*ops::ln/LN_10` 实现、能量用 f64 累加器逐样本平方和、非有限样本经 `finite`/`finite_abs` 归零、窗边界用 `saturating_sub/add`+`.min(len)` 防越界；与既有参数严格区分不重复：`room_clarity` 的 `C50`/`C80` 与 `useful_to_detrimental_ratio` 的 `U50`/`U80` 以**时间零点起算的固定 50/80 ms 边界**切分早晚声量化清晰度/可懂度，本模块以**检测到的直达峰为中心的极窄窗**隔离直达路径服务距离感知，窗位置与用途根本不同、能量积分不共享；`center_time`（能量重心）、`initial_time_delay_gap`（首反射间隙）为各自独立单值参数；`reverberant_field::ReverberantField` 上基于距离+指向性的 Sabine 统计理论 DRR（`Q*R/(16*PI*r^2)`，是方法而非自由函数）是**理论预测**，本模块是**从录制 IR 实测**——`# Relationship` 已澄清、无 E0252、无 DRY（rustdoc 含 `# Model`/`# Relationship`/`# Real-time contract`/`# Provenance` 四段）；控制速率离线估计器（非音频线程、无堆分配、不 panic），空/sr 非有限或<=0/全零/`E_total<=1e-20`/结果非有限→安全哨兵 `NO_DRR_DB=-100`、无混响能量（`E_reverberant<=1e-20`）→钳 `MAX_DRR_DB=100`；导出 `DIRECT_WINDOW_MS=2.5`/`NO_DRR_DB=-100`/`MAX_DRR_DB=100`、`direct_to_reverberant_ratio_db(ir,sample_rate:Sample)->Sample`、`DirectToReverberantRatio{drr_db,direct_arrival_index}`+`from_impulse_response(ir,sample_rate)`（Debug/Clone/Copy/PartialEq/Default+serde 门控）；15 golden 对拍（已知比值 6.0206 dB、直达峰索引定位、窗宽随采样率缩放、全在窗内钳 max、混响能量增则比值降、刚好窗内的反射计入直达、空/全零/零采样率/NaN 采样率哨兵、非有限样本安全、自由函数与 `from_impulse_response` 一致、Default 为零、常量稳定、结果钳位区间）+1 doctest；字母序 `pub mod direct_to_reverberant_ratio;` 插在 `diffusion_field` 与 `doppler` 之间（lib.rs 同序 re-export，插入前 grep 查重确认无 E0252）；spatial 累计 **629 单测 + 49 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`direct_to_reverberant_ratio.rs` 364 行 + spatial `lib.rs` +5，commit 5778b2aeb，经主 agent 独立全绿验证 + 逐行 CR：峰值检测/窄窗隔离/`E_reverberant=E_total-E_direct`/各退化守卫/`<=` 能量守卫规避 `neg_cmp_op_on_partial_ord`/与统计理论 DRR 区分均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.76（**本版新增**：M1 效果层 `prism_audio_core` 新增 **ISO 栅格图示均衡器 `GraphicEqNode`（`graphic_eq`，恒定 Q 峰值滤波器组）**（借鉴硬件调音台/现场扩声图示 EQ「固定 ISO 频点推子、只调增益」的工程范式但不抄任何源码：以 ISO 266 优选频率为中心、复用本 crate `Biquad` 的 RBJ 峰值截面级联，`GraphicEqSpacing{Octave(10 带,31.25 Hz~16 kHz)/ThirdOctave(31 带,约 20 Hz~20 kHz)}`，各带中心频率 `1000*2^e`（octave: `e=index-5`；third-octave: `e=(index-17)/3`，均锚定 1 kHz 参考 `REFERENCE_FREQUENCY_HZ`），全带共享恒定品质因数 `Q=1/(2^(1/(2*fraction))-2^(-1/(2*fraction)))`（octave=1.4142、third-octave=4.3187 教科书值），超 Nyquist 的高频带由 `BiquadCoeffs::design` 内部钳位保持稳定；**关键设计：某带恰为 0 dB 时折叠为透传单位系数（`b0=1`、其余为 0）而非 RBJ 峰值形式**——RBJ 峰值在 0 dB 仅在 `1/a0` 倒数舍入意义下近似单位，单位系数则**既 bit-exact 透传又保持延迟状态随输入「预热」**，故 flat 带（及 flat 整机）为逐位透传且后续从 0 dB 推拉保持 click-free；与既有 `parametric_eq`（任意频率/Q/shape 的可调参数 EQ）严格区分不重复：图示 EQ 固定 ISO 频点与 Q、只暴露增益，二者级联同一 `Biquad` 峰值截面故不重复实现滤波数学（rustdoc 含 `# The model`/`# Real-time contract`/`# Relationship`/`# Provenance` 四段）；实时契约：全部每带每通道滤波状态在 `new` 预分配，`process` 零分配/无锁/不 panic，`set_gain` 原位重设计单带系数并保留状态；导出 `REFERENCE_FREQUENCY_HZ=1000`/`OCTAVE_BAND_COUNT=10`/`THIRD_OCTAVE_BAND_COUNT=31`/`MAX_BAND_GAIN_DB=24`、`GraphicEqSpacing`（`band_count()`/`fraction()`/`band_frequency_hz(index)->Option<Sample>`，Copy+Eq+serde 门控）、`GraphicEqNode::new(sample_rate,channels,spacing)`+`spacing()`+`band_count()`+`band_frequency_hz()`+`gain_db(index)->Option`+`set_gain(index,gain_db)`（钳 ±24 dB、重设计保留状态、越界 no-op），impl `AudioNode`（process copy+级联 / reset）；14 golden 对拍（flat 带逐位透传、带数与 spacing 匹配、octave/third-octave 中心频率合 ISO、Q 对拍教科书值、单带与独立 `BiquadNode` 逐样本一致、1 kHz 推/拉升降幅、远端带不影响 1 kHz、增益钳位、越界 no-op、交替 ±6 dB 冲激响应有限且衰减、reset 清状态、访问器、低采样率高频带钳位稳定）+1 doctest；字母序 `pub mod graphic_eq;` 插在 `frequency_shifter` 与 `mid_side_matrix` 之间（effects/mod.rs 同序 re-export+catalogue）；core 累计 **533 单测 + 22 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`graphic_eq.rs` 515 行 + effects `mod.rs` +7，commit df7eb178e，经逐行 CR 并修复「0 dB 峰值并非 bit-exact 单位」问题改用透传单位系数后 flat 透传测试通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.75（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **Bradley 有用-有害声能比 `UsefulToDetrimental`（`U50`/`U80`，`useful_to_detrimental_ratio`）**（借鉴音乐厅/语言声学评价对「清晰度在有背景噪声时须把噪声计入有害分母」的工程需求但不抄任何源码：以 Bradley 1986 把 `C50`/`C80` 清晰度扩展一个背景噪声项——早到声视为「有用」、晚到声连同稳态背景噪声视为「有害」，`U=10*log10(E_early/(E_late+E_noise))`，其中 `E_noise=E_total*10^(-snr_db/10)`、分割时间 `t_s` 取 50 ms（U50，语言）或 80 ms（U80，音乐）；无 `log10` 以 `ops::ln`/`LN_10` 实现、`10^x` 以 `ops::powf(10.0,·)`，能量用 f64 累加器逐样本平方和、`finite()` 把非有限样本归零；与既有参数严格区分不重复：`room_clarity` 的 `C50`/`C80` 是**无噪声**早晚能量比，本模块在有害分母**加噪声项 `E_noise`**，当 `snr_db→+inf` 时 `E_noise→0`、`U_t` 收敛到**同样钳位的 `C_t`**（单测对拍 `room_clarity::clarity_db`），早晚分割常量 `EARLY_LATE_SPLIT_50_MS`/`EARLY_LATE_SPLIT_80_MS` 与钳位 `MAX_CLARITY_DB` 直接复用、无噪声能量积分不重实现（rustdoc 含 `# Model`/`# Relationship`/`# Real-time contract`/`# Provenance` 四段）；控制速率离线估计器（非音频线程、无堆分配、不 panic），`snr_db` 非有限→当作无穷信噪比（`E_noise=0` 退回纯 `C_t`）、空/sr 非有限或<=0/`E_total<=1e-20`/结果非有限→安全哨兵 `NO_USEFUL_RATIO_DB=-100`、无有害能量（分母<=floor）→钳 `MAX_CLARITY_DB`；导出 `NO_USEFUL_RATIO_DB`、`useful_to_detrimental_ratio_db(ir,split_ms,snr_db,sample_rate:Sample)->Sample`、`UsefulToDetrimental{u50_db,u80_db}`+`from_impulse_response(ir,snr_db,sample_rate)`（Copy+Default+serde 门控）；14 golden 对拍（高 snr 的 U50/U80 分别对拍 C50/C80、非有限 snr 等价无噪声、snr 降低 U 单调降、50 ms 与 80 ms 窗区分一次 65 ms 反射、已知比值 6.0206 dB、无晚期能量钳 max、空/全零/零采样率/NaN 采样率哨兵、短 IR 全在早期窗仍有限钳位、非有限样本安全、自由函数与 from_impulse_response 一致、Default 为零）+1 doctest；字母序 `pub mod useful_to_detrimental_ratio;` 插在 `stage_support` 之后（lib.rs 同序 re-export，插入前 grep 查重确认无 E0252）；spatial 累计 **614 单测 + 48 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`useful_to_detrimental_ratio.rs` 364 行 + spatial `lib.rs` +4，commit ff661cd6b，经主 agent 独立全绿验证 + 逐行 CR：早晚能量积分/噪声项扩展/高 snr 对拍 clarity/各退化守卫/`neg_cmp_op_on_partial_ord` 规避改写等价均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.74（**本版新增**：M1 效果层 `prism_audio_core` 新增 **单边带频移节点 `FrequencyShifterNode`（`nodes/effects/frequency_shifter`）**（借鉴 Bode/Moog 频移器与通信理论的单边带（SSB）频移思想但不抄任何源码：对实信号频谱做**刚性 Hz 平移**——与音高移位「乘以常数比率、谐波仍成谐波」不同，频移是「对每个分音加同一 `shift_hz` 偏移」，令谐波列 `f,2f,3f,...` 变为 `f+s,2f+s,3f+s,...` 失去整数关系，产生金属质感的钟鸣/clangorous 音色；实现以 Hilbert 变换构造解析信号 `x_r+j*x_i` 再乘复指数 `exp(j*2*pi*s*t)` 取实部：Hilbert 用奇数长 `HILBERT_TAPS=127` 的 Type-III 反对称 FIR（理想核 `2/(pi*m)` 仅奇 `m` 非零、Blackman 窗抑制纹波，群延迟 `GROUP_DELAY=(TAPS-1)/2=63`），实路径延迟同样群延迟与正交路径对齐，`y[n]=x_r*cos(theta)-x_i*sin(theta)`、`theta` 每样本前进 `2*pi*shift_hz/sr` 并 wrap 到 `[-pi,pi)`；复振荡器全通道共享以保立体声/环绕像相位一致、正 `shift_hz` 上移负则下移（下边带）；与既有节点严格区分不重复——`ring_modulator` 乘实载波产生 `f+/-f_c` **对称双边带**（镜像频谱），本节点用解析信号只保单边带故**平移而非镜像**，`parametric_eq`/dynamics 是改**各频率幅度**而本节点**搬移频率**（rustdoc 含 `# Relationship`/`# Model`/`# Real-time contract` 逐一澄清）；Hilbert 核与各通道延迟线构造期一次性分配、`process` 无锁/不 panic/零分配、通道失配与零帧优雅退化、经 `latency_frames()->GROUP_DELAY` 报告群延迟、wet/dry 的 dry 用**同样延迟后的实路径**以避免与 wet 梳状干涉；导出常量 `HILBERT_TAPS=127`/`DEFAULT_SHIFT_HZ=100`/`DEFAULT_MIX=1`、`FrequencyShifterParams{shift_hz,mix}`（Copy+Default+serde 门控、mix 构造即钳 `[0,1]`）、`FrequencyShifterNode::new(sample_rate,channels,params)`+`set_params`（重调频移与 mix 但保留相位与延迟态故无咔哒），impl `AudioNode`（process/reset/latency_frames）；12 golden 对拍（静音透传/正频移能量落在 f+s 且镜像 f-s 被压制 4x+/负频移能量落在 f-s/零频移为纯延迟/mix=0 为延迟后 dry/立体声两通道逐样本一致/报告群延迟延迟/核反对称且中心抽头为零偶抽头消失/零帧安全/reset 清零/mix 钳位/默认参数合理）+1 doctest；字母序 `pub mod frequency_shifter;` 插在 `flanger` 与 `mid_side_matrix` 之间（effects/mod.rs 同序 re-export + catalogue 一条，新增行用 `--` 保 ASCII）；core 累计 **519 单测 + 21 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`nodes/effects/frequency_shifter.rs` 463 行 + `nodes/effects/mod.rs` +8，commit e722b791b，经全绿验证 + 逐行 CR：Hilbert 核反对称性/群延迟对齐/SSB 单边带选择性/复振荡器相位 wrap/wet-dry 延迟对齐/退化透传均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.73（**本版新增**：M1 动态层 `prism_audio_core` 新增 **单频带动态均衡节点 `DynamicEqNode`（`nodes/dynamics/dynamic_eq`）**（借鉴母带/混音「动态 EQ」对「只在某频段越阈时才介入的频率选择性动态处理」的工程需求但不抄任何源码：采用**固定滤波器 + depth 交叉淡化**的实时安全表述——shaping 用一枚恒设在满 `range_db` 的 RBJ 峰值滤波器 `peaking(x)`，取 `delta=peaking(x)-x`、`out=x+depth*delta`，`depth∈[0,1]`（0=平直旁通、1=满 `range_db`），系数永不逐样本重算故无条件稳定、无 zipper 噪声、热路径零分配；检测路径对各活动通道的**逐样本单声道平均**过一枚同 `freq/q` 的带通后喂 `LevelDetector`（RMS 10 ms 窗）得频段电平 dB，`DynamicEqMode::Above` 在越阈上升介入（动态切谐振或动态推）、`Below` 在跌破阈介入（仅安静段抬升某频段），`drive=max(0,±(level-threshold))`、`depth_target=clamp(drive/width_db,0,1)`，再用 `attack/release` 单极平滑（复用 `detector::time_to_coef`，升用 attack 系数、降用 release）；全通道共用同一 `depth` 以保立体声像一致）：与既有动态/EQ 节点严格区分不重复——`compressor` 是全带单一增益、`multiband` 是 Linkwitz-Riley 分频带各自压缩再求和、`de_esser` 是专用高频 sibilance 分频带压缩、`parametric_eq` 是**静态**峰值带，本节点是**单枚可任意调谐、双向、由频段电平交叉淡化**的峰值钟形（rustdoc 含 `# Relationship` 逐一澄清）；因 `Biquad` 仅暴露 buffer 级处理，自实现 DF-I 逐样本 `DirectFormBiquad`（存 `[x1,x2,y1,y2]`、`y0=b0*x+b1*x1+b2*x2-a1*y1-a2*y2`、`flush_denormal` 收尾），检测带通 1 份 mono 状态、shaping 峰值每通道 1 份状态，构造期一次性分配、`process` 无锁/不 panic、通道/帧数失配与零帧优雅退化；导出常量 `DEFAULT_FREQUENCY_HZ=1000`/`DEFAULT_Q=2`/`DEFAULT_RANGE_DB=-6`/`DEFAULT_THRESHOLD_DB=-24`/`DEFAULT_WIDTH_DB=12`/`DEFAULT_ATTACK_MS=10`/`DEFAULT_RELEASE_MS=120`/`DETECTION_RMS_WINDOW_MS=10`/`MIN_WIDTH_DB=0.1`、`DynamicEqMode{Above(default)/Below}`、`DynamicEqParams{frequency_hz/q/range_db/threshold_db/width_db/attack_ms/release_ms/mode}`（Copy+Default+serde 门控）、`DynamicEqNode::new(sample_rate,channels,params)`+`set_params`（重设计滤波器与系数但保留运行态故无咔哒）+`mode()`+`depth()`，impl `AudioNode`（process/reset，reset 清滤波器状态/检测器/depth）；12 golden 对拍（静音平直旁通/阈下不介入近似透传/Above 越阈切该带能量下降/正 range 越阈推升/频带外 8 kHz 经 1 kHz 钟形基本不受影响/Below 模式安静时抬升/立体声共用 depth 两通道逐样本一致/零帧安全/reset 清零/mode 存取/set_params 保留运行态/默认参数合理）+1 doctest；字母序 `pub mod dynamic_eq;` 插在 `ducking` 与 `gate` 之间（dynamics/mod.rs 同序 re-export + catalogue 一条，新增行用 `--` 保 ASCII）；core 累计 **507 单测 + 20 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`nodes/dynamics/dynamic_eq.rs` 643 行 + `nodes/dynamics/mod.rs` +4，commit 94ca69dce，经全绿验证 + 逐行 CR：固定峰值+depth 交叉淡化恒等性、检测带通/电平律/attack-release 平滑、DF-I 逐样本正确性、频率选择性与退化透传均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.72（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **ISO 3382-1 晚期侧向声级 `late_lateral_sound_level`（`LJ` 听者包围感）**（借鉴音乐厅声学评价对「晚期侧向声能决定被混响场包围感 LEV」的工程需求但不抄任何源码：以图八字 lateral 脉冲响应 `pL` 的晚期能量比同源 10 m 自由场参考能量，取 `LJ=10*log10(sum_{80ms<t<=end} pL^2 / E_ref)`——无 `log10` 以 `ops::ln`/`LN_10` 实现；与既有参数严格区分不重复：`spatial_impression` 的 LF/LFC 是早期 0-80ms 侧向能量**分数**(无量纲、关联表观声源宽度 ASW)、`sound_strength` 的 G 是**全指向**声级 dB(无方向性)、本模块是**晚期侧向**能量**级** dB(关联听者包围感 LEV)，窗/量纲/物理意义各异)；晚期窗起点复用 `room_clarity::EARLY_LATE_SPLIT_80_MS`(=80ms)、参考能量复用 `sound_strength` 的 10m 自由场约定；f64 能量累加器逐样本平方和、`finite()` 把非有限样本映射为 0、`ms_to_index` 钳 `[0,len]` 且 round 前验有限正、`energy_to_db` 的 `!(late_energy>ENERGY_FLOOR)` 为 NaN 安全分支(带 `#[expect(...)]`)、参考能量非正/非有限与晚期窗静默/空/全零/非有限/零采样率/短 IR 全退化回退哨兵 `NO_LATE_LATERAL_DB=-100`；控制速率离线估计器(非音频线程、无堆分配、不 panic)；导出 `LATE_LATERAL_START_MS`(=80)/`NO_LATE_LATERAL_DB`(=-100)/`late_lateral_sound_level_db(figure_eight,reference_energy,sample_rate)->Sample`/`LateLateralSoundLevel{lj_db}`+`from_responses(...)`(Copy+Default+serde 门控)；15 golden 对拍(已知能量比 -6.0206/-12.0412dB、参考缩放 +3.0103dB、更强晚期侧向抬升 LJ、更大参考降低 LJ、80ms 窗边界含起点样本/排除前一样本、早期能量不入晚期窗、各退化回退哨兵)+1 doctest；字母序 `pub mod late_lateral_sound_level;` 插在 `initial_time_delay_gap` 与 `material_library` 之间(L69)、crate root re-export(L129)无命名冲突；spatial 累计 **600 单测 + 47 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，由 Bernoulli 子 agent 自提精确 commit(`late_lateral_sound_level.rs` 357 行 + `lib.rs` +2，commit 46d5de3cf，经主 agent 独立全绿验证+逐行 CR 通过：log10 实现/f64 能量累加/80ms 窗切片语义/NaN 安全分支/全退化回退/与 ASW·G 的职责区分均核验通过；1 处合理偏离：自由函数按 `sound_strength::late_sound_strength_db` 先例补 `sample_rate` 参数——80ms 窗无采样率不可计算，功能无损)；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.71（**本版新增**：M1 效果层 `prism_audio_core` 新增 **Mid/Side 矩阵编解码节点 `MidSideMatrixNode`（`nodes/effects/mid_side_matrix`）**（借鉴母带/混音流程对「中置与两侧分别处理」的插入点需求但不抄任何源码：仅做无损线性和差变换 `M=(L+R)/2`、`S=(L-R)/2`，逆变换 `L=M+S`、`R=M-S` 完美重构；`MidSideMode::Encode` 吃 L/R 在 ch0/ch1 输出 M/S 供下游独立处理，`MidSideMode::Decode` 吃 M/S 重建 L/R，单位增益下 encode→decode 为精确恒等——职责与 `stereo_width` 严格区分：后者永远内部整段往返、只对 side 施单一 width 缩放加 bass-mono 分频、从不在端口暴露 M/S，本节点则暴露原始 M/S 或由其重建 L/R 以便在两级之间插入任意处理）：独立 M/S 微调增益 `mid_gain_db`/`side_gain_db`（`Smoothed` 10 ms 斜坡去 zipper 噪声，编码模式缩放产出的 M/S、解码模式缩放入站 M/S），构造期零分配、`process` 热路径无锁/不 panic、通道<2 整体透传、超出立体声对的通道逐一透传、`flush_denormal` 收尾；导出常量 `GAIN_SMOOTH_SECONDS=0.01`、`MidSideMode{Encode/Decode}`（Copy+Default=Encode+serde 门控）、`MidSideMatrixParams{mode/mid_gain_db/side_gain_db}`（Copy+Default+serde 门控）、`MidSideMatrixNode::new(params)`+`set_params(params,sample_rate)`+`mode()`，impl `AudioNode`（process/reset，reset 把增益瞬时吸附到目标）；10 golden 对拍（编码产出 M/S/解码逆编码恒等/中置单声道零 side/side 增益仅缩放 side/单声道透传/超出通道透传/零帧安全/reset 吸附/默认参数单位编码/增益斜坡收敛）+ 1 doctest；字母序 `pub mod mid_side_matrix;` 插在 `flanger` 与 `parametric_eq` 之间（effects/mod.rs 同序 re-export + catalogue 一条）；core 累计 **495 单测 + 19 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，精确 commit（`nodes/effects/mid_side_matrix.rs` 417 行 + `nodes/effects/mod.rs` +8，commit 82f53dc67，经全绿验证 + 逐行 CR：和差矩阵与逆矩阵正确性、恒等往返、增益平滑/吸附语义、退化透传均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.70（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **ISO 3382-1 舞台支持度参数 `stage_support`（`ST_early`/`ST_late`）**（借鉴音乐厅/排练厅声学评价对「演奏者在舞台上听到自身与乐队回声支持」的工程需求，但不抄任何源码：描述舞台上一位乐手处 1 m 全指向声源经舞台围蔽/反射板返回同点的早/晚期能量支持，与 §听者侧的 `room_clarity`/`center_time`/`sound_strength` 互补——后者评价观众席听感，`stage_support` 评价演奏者侧的合奏便利度与自我监听，职责不重复）：对单点脉冲响应按 ISO 3382-1 Annex C 三能量窗积分——直达窗 0–10 ms（含直达声能量基准 `E_direct`）、早期窗 20–100 ms、晚期窗 100–1000 ms（10–20 ms 间隙为规范刻意排除区，避免直达声尾部污染早期支持）；`ST = 10·log10(E_window / E_direct)`（无 `log10` 以 `ln`/`LN_10` 实现）、能量用 f64 累加器逐样本平方求和、窗边界 `ms_to_index` 钳位 `[0, len]` 且 round 前已验有限正；空脉冲响应/全零/非有限/零采样率/过短脉冲响应/静默直达窗等退化场景全安全回退 `NO_SUPPORT_DB = 0`，`ratio_to_db` 的 `!(direct > ENERGY_FLOOR)` 分支为 NaN 安全（带 `#[expect(...)]`）；导出 `StageSupport{st_early_db, st_late_db}`、自由函数 `stage_support_early_db`/`stage_support_late_db(ir, sr: u32) -> Sample`、`NO_SUPPORT_DB`（crate root re-export），窗边界常量 `DIRECT_WINDOW_END_MS=10`/`EARLY_WINDOW_START_MS=20`/`EARLY_WINDOW_END_MS=100`/`LATE_WINDOW_START_MS=100`/`LATE_WINDOW_END_MS=1000` 为模块级 pub（经 `stage_support::` 路径访问，因 `EARLY_WINDOW_END_MS` 与既有 `spatial_impression::EARLY_WINDOW_END_MS` 冲突故不升顶，功能无损）；14 单测（含已知能量比 `-6.0206`/`-12.0412 dB`、直达更响 `-6.0206 dB`、各退化回退）+ 1 doctest；字母序 `pub mod stage_support;` 插在 `spread` 之后、re-export 接其后；spatial 累计 **585 单测 + 46 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净，由 Bernoulli 子 agent 自提精确 commit（`stage_support.rs` 389 行 + `lib.rs` +4，commit 83e3452c0，经主 agent 独立全绿验证 + 逐行 CR 通过：三能量窗边界/ISO 10–20 ms 排除/`ST=10log10` 实现/f64 能量累加/`ms_to_index` 钳位/NaN 安全分支/全退化回退均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.69（**本版新增**：M1 效果层 `prism_audio_core` 新增 **频域谱门/下行谱扩展节点 `SpectralGateNode`（`nodes/effects/spectral_gate`）**（借鉴 UE/Unity/Godot 对降噪/底噪抑制的插件需求但不抄任何源码：时域 `GateNode` 按宽带电平整体开合，无法在音符之间保留音符本身而只压抑嘶声/交流声——本模块对每个频点独立门控，是时域门做不到的能力，职责与时域 `GateNode` 互补不重复）：加权重叠相加短时傅里叶变换 `STFT`——每帧 `fft_size` 样本经 Hann 分析窗、`radix-2` 时域抽取 `DIT` 快速傅里叶变换 `FFT`、逐 bin 门控、逆变换、Hann 合成窗、跳距 `hop=fft_size/OVERLAP_FACTOR`（75% 重叠）重叠相加；平方 Hann 窗在该跳距满足恒重叠相加 `COLA` 条件，全门开时精确重构输入（经验实测重构误差 ~1e-12）；每 bin 单边幅度经窗相干增益归一为 `dBFS`（满刻度正弦落在某 bin 读回 `0 dBFS`），≥`threshold_db` 目标单位增益、否则目标线性 `reduction_db` 底，每跳一极平滑（开用 `attack_ms`、关用 `release_ms` 时间常数，抑制瞬时 bin 门控的「音乐噪声」），增益施于 bin 及其 Hermitian 镜像保逆变换为实；DAFX 风格 rover 流式算法（每通道 `in_fifo`/`out_fifo`/`out_accum`，共享 `rover`），构造期一次性预分配全部环形/重叠/旋转因子/窗/scratch 缓冲，`process` 热路径零分配/无锁/不 panic、非有限输入当静音；上报处理延迟 `latency_frames()`=`fft_size`（经多尺寸经验实测恒等于一整帧，内部 `fifo_latency`=`fft_size-hop` 仅为 FIFO 读偏移、与上报延迟区分）；导出常量 `MIN_FFT_SIZE=64`/`DEFAULT_FFT_SIZE=1024`/`OVERLAP_FACTOR=4`/`DEFAULT_THRESHOLD_DB=-60`/`DEFAULT_REDUCTION_DB=-80`/`DEFAULT_ATTACK_MS=2`/`DEFAULT_RELEASE_MS=50`、`SpectralGateParams{threshold_db/reduction_db/attack_ms/release_ms}`（Copy+Default+serde 门控）、`SpectralGateNode::new(sample_rate,channels,requested_size,params)`+`fft_size()/hop()/set_params()`，impl `AudioNode`（process/reset/latency_frames）；私有基建 `reverse_low_bits`/`power_of_two_at_least`/`transform`（迭代 `radix-2` `DIT`，正逆合一）/`frame_coeff`（跳距帧率一极系数），FFT 风格借鉴自既有 `nodes/analysis/spectrum.rs`；15 golden 对拍（2 的幂取整/最小尺寸/hop=四分之一/延迟=一整帧/全开门延迟重构输入/静音被衰减/响音通过/非有限安全/零帧安全/多通道独立门控/surplus 通道透传/reset 等价新实例/默认参数/frame_coeff 边界/set_params）+ 1 doctest；字母序 `pub mod spectral_gate;` 插在 `saturation` 与 `stereo_width` 之间（effects/mod.rs 同序 re-export + catalogue 一条）；core 累计 **485 单测 + 18 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过、ASCII 全净（两文件纯 ASCII），精确 commit（`nodes/effects/spectral_gate.rs` 744 行 + `nodes/effects/mod.rs` +9，commit ec98fddd2，经主 agent 全绿验证+逐行 CR：COLA 归一正确性经实测重构误差佐证、Hermitian 镜像、rover 流式索引不越界、延迟定义经多尺寸实测校正为 `fft_size`、门控平滑语义均核验通过）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.68（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **`IEC 60268-16` 语音传输指数 `SpeechTransmissionIndex`/`speech_transmission_index`（`speech_transmission_index`）**（借鉴 UE/Unity/Godot/Wwise/FMOD 对语音可懂度客观量化的需求但不抄任何源码：混响与噪声会抹平承载语音的慢速幅度调制、降低可懂度——本模块按公开发布的 `IEC 60268-16` 标准用 Schroeder 1981 调制传递函数 MTF 法，从单条房间脉冲响应 RIR 把该损失压缩为 `[0,1]` 单值，与 `room_clarity`（`C50`/`D50` 能量比）、`center_time`（能量重心 Ts）、`echo_criterion`（回声斜率）互补、互不重复实现——那三者报能量比/重心/回声斜率，本模块报基于调制传递的语音可懂度指数）：7 倍频带 `OCTAVE_CENTERS_HZ`（125Hz~8kHz）各以两级相同 `RBJ` 恒峰增益带通 biquad 级联（四阶、中心单位增益）滤波，平方包络 `h^2(t)` 为能量流，对 `MODULATION_COUNT`=14 个 1/3 倍频调制频 `MODULATION_FREQS_HZ`（0.63~12.5Hz）算调制传递 `m(F)=|Σ_t h^2(t)e^{-j2πFt}|/Σ_t h^2(t)`（f64 复数积分累加器），可选噪声项按 `1/(1+10^{-SNR/10})` 缩放每个 m；每个 m 化为表观信噪比 `SNR_app=10·log10(m/(1-m))` 钳 `[-15,+15]`dB，带内调制传递指数 `MTI_k` 为 14 个 `(SNR_app+15)/30` 之均值 ∈`[0,1]`；总指数施加男声权重 `STI=Σ_k α_k·MTI_k − Σ_k β_k·sqrt(MTI_k·MTI_{k+1})` 钳 `[0,1]`，具名公开权重 `MALE_ALPHA`（7 个，Σ=1.381）/`MALE_BETA`（6 个，Σ=0.381，满足 Σα−Σβ=1 使纯直达 STI=1 不变式）；定性 `StiRating{Bad<0.3/Poor<0.45/Fair<0.6/Good<0.75/Excellent≥0.75}`（`from_sti`、serde 门控）；RBJ 带通系数经 `bandpass_coeffs`（非有限/非正/≥Nyquist 退化返回 None 跳过该带）、带限滤波 `filter_band`（两级转置直接 II 型、非有限输入当 0）、能量底 `ENERGY_FLOOR`=1e-20 守卫静默带、`MTF_LIMIT`=1−1e-4 保 `m/(1-m)` 有限；导出常量 `OCTAVE_COUNT`=7/`MODULATION_COUNT`=14/`OCTAVE_CENTERS_HZ`/`MODULATION_FREQS_HZ`/`MALE_ALPHA`/`MALE_BETA`/`APPARENT_SNR_LIMIT_DB`=15，自由函数 `speech_transmission_index(ir,sample_rate,snr_db)->Sample` + `SpeechTransmissionIndex{sti/mti_per_band:[7]/rating}` + `Default`（全 0/Bad）+ `from_ir(...)`；控制率离线估计器，热路径外仅一次有界堆分配（长度=响应长的 scratch，跨 7 带复用），绝不跑音频线程、永不 panic——空/全零/非有限/非正采样率均安全回退 `0`；14 golden 对拍（纯直达 Excellent（sti>0.9）/ 混响降低 STI / 极端混响 Poor 以下（sti<0.45）/ 低 SNR 降低 STI / 空/全零/非有限/零采样率安全 / 定性评级边界 / 调制指数 ∈`[0,MTF_LIMIT]` / MTI ∈`[0,1]` / 权重数组长度与 Σα−Σβ=1 不变式 / from_ir 等价自由函数 / default 全零）+ 1 doctest；字母序插在 `spatializer` 与 `spread` 之间（接手提示的「sound_strength 后/source_directivity 前」按严格 ASCII 排序有误，`sp-e`>`sp-a`，已按严格字母序落位）；spatial 累计 **571 单测 + 45 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，ASCII 全净（两文件纯 ASCII），精确 commit（`speech_transmission_index.rs` 518 行 + spatial `lib.rs` +5 行，commit 4fe530578，由并行 agent 提交并经主 agent 独立复核全绿+逐行 CR）；并修正接手夹具缺陷——原平滑指数衰减尾经带通后带内几无 AC（MTF≈1 致 STI≈0.9999 两测失败），改为物理正确的带限噪声×指数包络（确定性 xorshift）使尾部各带载能、平方包络正确平滑调制，算法未改仅修正测试夹具；`# Model`（7 带×14 调制频 MTF + 表观 SNR + MTI 均值 + 男声 α/β 权重）+ `# Relationship`（与 room_clarity/center_time/echo_criterion 互补、不重复实现、共享 Sample 标量）+ `# Real-time contract`（控制率离线、单次有界 scratch 跨带复用、不跑音频线程、永不 panic）+ `# Provenance`（IEC 60268-16 与 Schroeder 1981、RBJ cookbook 均为公开文献/公式，明确无 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）段齐全；恪守纯经典 DSP、无任何 AI/ML）
- 版本历史: v0.67（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **Dietsch-Kraak 回声判据 `EchoCriterion`/`echo_criterion`（`echo_criterion`）**（借鉴 Dietsch & Kraak 1986 回声可闻性判据但不抄任何源码：一个强而孤立的晚期反射会被听成离散回声而非有用混响——本模块从单条宽带房间脉冲响应 RIR 预测该可闻性，度量「建立函数」能量重心在短感知窗内移动的最大速率，与 `center_time`（整条响应单一能量重心 Ts）、`room_clarity`（清晰/混响能量比）、`reverberation_spectrum`（倍频带混响谱）互补、互不重复实现——center_time 给单一重心，本模块复用其运行建立函数仅作中间量、报告其最大窗斜率这一回声判据）：建立函数 `ts(tau)=∫_0^tau |p|^n·t dt / ∫_0^tau |p|^n dt`，离散为 f64 前缀矩/权累加 `ts[k]=Σ(t_i·|p_i|^n)/Σ|p_i|^n`（秒，采样间隔因子在比值约掉）；回声判据 `EK=max_tau (ts(tau)-ts(tau-Δtau_E))/Δtau_E` 即建立函数在一个宽度 Δtau_E 窗上的最大正向上升——平滑衰减重心移动慢（EK 小）、孤立晚反射重心突移（EK 大）；两种听音模式 `EchoMode{Speech,Music}`（Default=Speech）：语音 n=2/3、Δtau_E=9ms、阈值 EK>1.0；音乐 n=1、Δtau_E=14ms、阈值 EK>1.5（约半数听者报告回声的水平），`window_ms()/threshold()/exponent()` 访问器 + 具名常量 `SPEECH/MUSIC_WINDOW_MS`、`SPEECH/MUSIC_THRESHOLD`、`SPEECH/MUSIC_EXPONENT`；窗样本数 `round(Δtau_E·sr).max(1)`、窗≥响应长返回 0；控制率离线估计器，仅一次有界堆分配（长度=响应长的 f64 scratch），绝不跑音频线程；空/全零/非有限/非正采样率均安全回退 0（不可闻）；`echo_criterion(ir,sample_rate,mode)->Sample` 自由函数 + `EchoCriterion{ek/perceptible/mode}` + `from_ir(...)`、serde 门控、Default 全 0；15 golden 对拍（纯直达 EK≈0 不可闻 / 强晚反射可闻 / 回声越强 EK 越大 / 回声越晚 EK 越大 / 指数作用语音≠音乐 / 模式参数值 / 指数衰减不可闻 / 空 / 全零 / 零采样率 / 非有限安全 / 窗>响应长归零 / perceptible 对齐阈值 / from_ir 等价自由函数 / default 全零）+ 1 doctest；spatial 累计 **557 单测 + 44 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，ASCII 全净，精确 commit（`echo_criterion.rs` + spatial `lib.rs`，字母序插在 `early_reflections` 与 `geometry` 之间，commit df33b00a6）；`# Model`（建立函数矩/权比 + 最大窗斜率 + 双模式参数）+ `# Relationship`（复用 center_time 建立函数仅作中间量、报告最大窗斜率回声判据、与 room_clarity/reverberation_spectrum 互补、不重复实现）+ `# Real-time contract`（控制率离线、单次有界分配、不跑音频线程、永不 panic）+ `# Provenance`（Dietsch & Kraak 1986 为公开文献，明确无 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）段齐全；恪守纯经典 DSP、无任何 AI/ML）
- 版本历史: v0.66（**本版新增**：M6 母带层 `prism_audio_core` 新增 **目标响度归一化节点 `LoudnessNormalizerNode`/`LoudnessNormalizerParams`（`nodes/mastering/loudness_normalizer`）**（借鉴 UE/Unity/Godot/Wwise/FMOD 与流媒体/广播交付的「测量-应用」两遍响度归一化工作流思想但不抄任何源码：现代交付按固定节目响度而非固定峰值发行，使不同来源内容在回放时感知电平一致——本节点实现该工作流的「应用」半边：它不自测响度（单遍流式无法预知未播音频的整合响度，故绝不假实现），而是把由 `LoudnessMeter`（analysis）量得的整合响度 LUFS 与真峰 dBTP 作为参数传入，给定交付目标后算出精确静态补偿增益并无咔哒施加）：核心纯函数 `normalization_gain_db(params)` 实现 `ITU-R BS.1770-4`/`EBU R128` 约定——理想补偿 `g = T - M` dB（LUFS 差即 dB 差），再加两道守卫：**真峰天花板**（施加 g 后峰值升至 `measured_peak+g`，须 ≤ 天花板 C（默认 -1 dBTP），故 `g = min(g, C - measured_peak)`，仅向下削正增益、源已越顶则强制衰减）与**增益上限**（钳到 ±`max_gain_db`，默认 24dB，防把近静默源抬进噪声底）；源响度 ≤ `BS.1770` 绝对门 -70 LUFS（`SILENCE_GATE_LUFS`）或非有限测量/目标→判为不可测、单位增益直通；节点 `LoudnessNormalizerNode` 复用 `GainNode` 式 `Smoothed` 无咔哒滑行（重设目标/测量时平滑过渡）、同一增益施于全通道保持声像相干、热路径零分配/无锁/不 panic；`LoudnessNormalizerParams{measured_lufs/target_lufs/measured_true_peak_dbtp/max_true_peak_dbtp/max_gain_db}`（Copy、serde 门控、Default 为单位直通：measured 置于静默门）、常量 `DEFAULT_TARGET_LUFS(-14)`/`DEFAULT_MAX_TRUE_PEAK_DBTP(-1)`/`DEFAULT_MAX_GAIN_DB(24)`/`SILENCE_GATE_LUFS(-70)`/`DEFAULT_RAMP_SECONDS(0.05)`、`new(sample_rate,&params)/params/set_params/set_measured_lufs/set_target_lufs/set_measured_true_peak_dbtp/set_max_true_peak_dbtp/set_max_gain_db/set_ramp/current_gain_linear/applied_gain_db` + impl `AudioNode`；16 golden 对拍（测=目标单位增益 / 响于目标则衰减 / 弱于目标则增益 / 真峰天花板封顶增益 / 真峰越顶强制衰减 / max_gain 钳位 / 静默门禁用 / 非有限测量单位 / 非有限目标单位 / Default 单位 / 节点施加稳态增益 / 重设无咔哒单调滑行 / reset 吸附目标 / 全通道相干 / applied_gain_db 读数 / 零帧 no-op）+ 1 doctest；core 累计 **470 单测 + 17 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，ASCII 全净，精确 commit（`loudness_normalizer.rs` + mastering `mod.rs` + nodes `mod.rs`，字母序 `dither` 之后 `mastering_chain` 之前，commit dd67ef0b3）；`# Model`（T-M 补偿 + 真峰天花板 + 增益上限 + 静默门）+ `# Relationship`（analysis `LoudnessMeter` 的信号施加对偶、复用 `GainNode` 式 `Smoothed` 滑行、不重复实现计量/限幅/滤波、是 `MasteringChainNode` 天然前级）+ `# Real-time contract`（增益计算标量级零分配、热路径无锁不 panic、重设为控制率）+ `# Provenance`（BS.1770-4/R128 为公开标准，明确无 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）段齐全；恪守纯经典 DSP、无任何 AI/ML）
- 版本历史: v0.65（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **ISO 3382-1 倍频带混响时间谱 `ReverberationSpectrum`（`reverberation_spectrum`）**（借鉴 ISO 3382-1 倍频带混响时间与 Beranek 低音比/高音比（warmth/brilliance）判据但不抄任何源码：真实房间各频段混响不等——软装吸收高频使混响时间随频率下降——本模块从单条宽带房间脉冲响应 RIR 量测逐倍频带混响时间，并凝练出 Beranek 低音比 BR（温暖感）与高音比 TR（明亮感），与 `room_clarity`（宽带 T30/EDT）、`center_time`（能量重心）、`sound_strength`（能量级 G）、`initial_time_delay_gap`（亲密感间隙）互补、互不重复实现）：把响应带通滤波进 8 个 ISO 倍频带（中心 `OCTAVE_BAND_CENTERS` 63Hz~8kHz），每带用两级相同 RBJ constant-peak-gain 带通双二阶级联（四阶、中心单位增益）；带内信号经 Schroeder 后向积分成能量衰减曲线（dB，起点归一化 0dB、下限约 -100dB），在 -5dB~-25dB 段最小二乘拟合直线并外推至 -60dB 跌落得逐带混响时间（即 T60，等价把 20dB 拟合跨度乘三）；由带时 T(f) 得 BR=(T(125)+T(250))/(T(500)+T(1000))、TR=(T(2000)+T(4000))/(T(500)+T(1000))；Nyquist 守卫（中心×2≥采样率返回 None）、非有限/全零/空/非正采样率/退化拟合均安全回退 0；控制率离线估计器，带循环外仅一次有界堆分配（一块与响应等长的 scratch，跨带复用），绝不跑音频线程；`ReverberationSpectrum{t30_per_band:[8]/bass_ratio/treble_ratio + from_ir/default}`、serde 门控；14 golden 对拍（带通中心单位增益 / 远端衰减 / 指数音 T60 对拍 / 带隔离能量比 / 全零 / 空 / 零采样率 / 非有限 / BR·TR 定义 / 零分母 / from_ir 一致 / default 全零 / 越 Nyquist None）+ 1 doctest；spatial 累计 **542 单测 + 43 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，ASCII 全净，精确 commit（`reverberation_spectrum.rs` + spatial `lib.rs`，字母序插在 `reverberant_field` 与 `room_acoustics` 之间，commit 6688bc9cc）；`# Model`（倍频带带通→Schroeder EDC→最小二乘外推 T60→BR/TR 定义）+ `# Relationship`（扩展 room_clarity 宽带 T30/EDT 为逐带谱并加 BR/TR、不复用其宽带值、与 center_time/sound_strength/initial_time_delay_gap 互补、共享 material_library 倍频带中心）+ `# Real-time contract`（控制率离线、单次有界分配、不跑音频线程、永不 panic）+ `# Provenance`（ISO 3382-1 与 Beranek BR/TR 及 RBJ cookbook 为公开标准/公式，明确无 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）段齐全；恪守纯经典 DSP、无任何 AI/ML）
- 版本历史: v0.64（**本版新增**：M6 母带层 `prism_audio_core` 新增 **固定序母带链编排节点 `MasteringChainNode`/`MasteringChainParams`（`nodes/mastering/mastering_chain`）**（借鉴 UE/Unity/Godot/Wwise/FMOD 的 Submix/Bus 母带处理链思想但不抄任何源码：按业界公认母带信号序 **参量 EQ -> 动态压缩 -> 砖墙限幅 -> 抖动量化** 把现有四个处理节点串成单一交付级节点；本节点本身不实现任何 DSP，仅负责按序编排、缓冲管理与时延累计，是对已有节点的纯组合）：内部各持一份 `ParametricEqNode`（effects）、`CompressorNode`/`LimiterNode`（dynamics）、`DitherNode`（mastering/dither）各阶段，均以 `layout` 推得的通道数构造；每阶段带独立 bypass 开关（`bypass_eq`/`bypass_compressor`/`bypass_limiter`/`bypass_dither` + 对应 `set_bypass_*` 控制率切换），全 bypass 时退化为逐比特直通；由于节点不得同一缓冲既作输入又作输出，构造期按 `max_block_frames`×`layout` 预分配两块 scratch `AudioBuffer`，`process` 内 `split_at_mut` 做乒乓接力（`run_stage` 读 `scratch[src]` 写 `scratch[1-src]` 保证不别名），首阶段从输入拷入 scratch（窄输入零填充）、末阶段拷至输出共享通道；超过 scratch 容量的块被钳到容量、零帧/零通道早返回，热路径零分配/无锁/不 panic（非有限输入由各子节点自身的钳位/限幅处理）；`latency_frames` 累加启用的压缩+限幅前瞻时延（EQ/dither 为 0），`reset` 级联各子节点并清空 scratch；`MasteringChainParams{eq_bands/compressor/limiter/dither + 四个 bypass + Default（EQ 空 unity、dither 默认 bypass）}`、`MasteringChainNode{new(sample_rate,layout,max_block_frames,&params)/layout/max_block_frames/set_bypass_*/eq_mut/compressor_mut/limiter_mut/dither_mut}` + impl `AudioNode`；13 golden 对拍（全 bypass 逐比特直通 / 仅 EQ 等价独立 EQ / 三级链等价手动串联 / 限幅压低电平 / bypass 切换改变输出 / reset 清空 scratch 与各级 / 零帧早返回不加宽输出 / 超容量块钳位不 panic / 单声道无 panic / 非有限输入经限幅保持有限 / 时延等于启用动态级之和 / bypass 动态级时延归零 / dither 级重量化改变输出）+ 1 doctest；core 累计 **454 单测 + 16 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，ASCII 全净，精确 commit（`mastering_chain.rs` + mastering `mod.rs` + nodes `mod.rs`，字母序 `dither` 之后，commit 5b6b2d7a4）；`# Provenance`（固定 EQ/压缩/限幅/dither 母带序为公开音频工程常识，明确无 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）+ `# Relationship`（纯组合复用四处理节点、固定母带信号序、不重复实现任何滤波/检测/量化 DSP）段齐全；恪守纯经典 DSP、无任何 AI/ML）
- 版本历史: v0.63（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **Beranek 初始时间延迟间隙 `InitialTimeDelayGap`（`initial_time_delay_gap`）**（借鉴 Beranek 音乐厅亲密感（intimacy）判据但不抄任何源码：从单条实测房间脉冲响应 RIR 量测「初始时间延迟间隙 ITDG」——直达声到达与第一个显著反射到达之间的毫秒间隙，经验上小于 20ms 显亲密、过大显疏远，与 `early_reflections`（镜像源法定位全部早期反射）、`reflection_clustering`（按方向聚类反射抽头）、`room_clarity`（清晰/混响能量比）、`center_time`（能量重心）互补、互不重复实现）：直达索引取 `|p[n]|` 全局峰值的首个样本索引；首反射取自直达之后首个「局部极大」（幅度大于等于左右相邻、末样本单侧比较）且幅度大于等于 `peak * db_to_linear(threshold_db)` 的样本；间隙 `= (reflection_index - direct_index) / sample_rate * 1000` ms；复用 `prism_audio_core::math::db_to_linear`，确定性数学无额外超越需求；导出常量 `DEFAULT_REFLECTION_THRESHOLD_DB = -10.0`、自由函数 `initial_time_delay_gap_ms`、`InitialTimeDelayGap{gap_ms/direct_index/reflection_index + Default（全 0）+ from_impulse_response + serde serialize 门控}`；空 / 采样率 0 / 峰值非正 / 无过门限反射 返回安全哨兵 `0`，非有限样本经 `finite_abs` 视作 0，离线控制率估计器（零堆分配、无 panic、rustdoc 注明不得跑音频线程）；14 golden 对拍 + 1 doctest；字母序插在 `hoa_rotation` 与 `material_library` 之间；spatial 累计 **528 单测 + 42 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`initial_time_delay_gap.rs` + spatial `lib.rs`，commit cff111135，由并行 agent 提交并经主 agent 独立复核全绿）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.62（**本版新增**：M6 分析/量测层 `prism_audio_core` 新增 **窗加权 radix-2 DIT FFT 幅度谱分析器 `SpectrumAnalyzer`/`SpectrumNode`（`nodes/analysis/spectrum`）**（借鉴 UE/Unity/Godot 实时频谱显示与经典频谱估计思想但不抄任何源码：把运行音频流转为单边幅度谱序列，服务频谱显示/声谱图/调音工具，与 `loudness`（响度）、`correlation`/`goniometer`（立体声相关/矢量）互补——本模块回答能量在哪个频率）：输入环形缓冲每满 `hop` 样本对最近 `size` 样本（构造时向上取整到 2 的幂、下限 2）施加窗后做 decimation-in-time radix-2 Cooley-Tukey FFT；`Window{Rectangular/Hann/Hamming/Blackman}` 四种标准窗（主瓣宽度与旁瓣抑制折中），`coefficient`/`coherent_gain` 各就位；单边幅度以相干增益（窗系数和）归一——内部 bin 乘 `2/window_sum`、DC(bin0) 与 Nyquist bin 乘 `1/window_sum`，使整 bin 正弦读回其线性幅度；构造期预算 twiddle 表（`tw_re`/`tw_im`）、bit-reversal 置换表、窗系数、幅度输出（长 `size/2+1`）全部一次分配，热路径 `feed_sample`/`process` 零分配/无锁/不 panic（非有限输入当静音），全部超越函数走 `bevy_math::ops` 保跨平台可复现；导出常量 `MIN_FFT_SIZE`/`DEFAULT_FFT_SIZE`/`DEFAULT_HOP`、`Window`、`SpectrumAnalyzer{new/size/hop/window/magnitudes/bin_frequency/frames_computed/reset/feed_sample}`、`SpectrumNode`（impl `AudioNode` 透传音频并喂通道 0）；16 golden 对拍（2 的幂取整/幅度谱长等于 size/2+1/DC 集中 bin0/整 bin 正弦读回幅度且邻 bin 无泄漏/Nyquist bin/bin_frequency 公式/各窗相干增益/Hann 端点 0/hop 下 frames_computed 递增/节点透传/reset 清空/非有限安全/多通道取 ch0/零帧安全/最小尺寸/hop 下限 1）+ 1 doctest；字母序插在 analysis `loudness` 之后；core 累计 **441 单测 + 15 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/analysis/spectrum.rs` + `nodes/analysis/mod.rs` + `nodes/mod.rs`，commit 9ccd04ff2）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.61（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **`ISO 3382-1` 中心时间 `CenterTime`（`center_time`）**（对齐公开发布的 `ISO 3382-1` 客观房间声学标准：从单条房间脉冲响应 RIR 量测厅堂「中心时间 Ts」——即能量重心（平方脉冲响应的一阶时间矩）出现的时刻，小 Ts 表清晰直达主导、大 Ts 表混响充盈，与 `room_clarity`（硬性早/晚分窗的 `C50`/`C80`/`EDT` 清晰度比）互补地给出**无需硬切分界**的单值重心度量，并与 `sound_strength`（能量级 G）、`spatial_impression`（侧向能量/双耳相关）、`room_acoustics`（几何预测侧）互补、互不重复实现）：核心公式 `Ts = sum_n (t_n * p[n]^2) / sum_n p[n]^2` 秒，`t_n = n / sample_rate`，分子 `sum t_n*p^2` 与分母 `sum p^2` 均以 f64 双精度单遍累加后读回 `Sample` 保长响应精度；导出自由函数 `center_time_seconds`/`center_time_ms`（毫秒即秒乘 1000）、`CenterTime{ts_seconds/ts_ms + Default（全 0）+ from_impulse_response(response,sample_rate) + serde "serialize" 门控}`；空响应 / 采样率 0 / 总能量 <=0 / 非有限结果 → 返回安全哨兵 `0`，非有限样本按 0 累计，绝不 `-inf`/`NaN`/panic；离线控制率估计器（非逐样本回调、零堆分配、rustdoc 注明不得跑音频线程），确定性数学仅走 `bevy_math::ops`；13 golden 对拍 + 1 doctest；字母序插在 `attenuation` 与 `cone` 之间；spatial 累计 **514 单测 + 41 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`center_time.rs` + spatial `lib.rs`，commit 9fe8d1853，由并行 agent 提交并经主 agent 独立复核全绿）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.60（**本版新增**：M1 效果层 `prism_audio_core` 新增 **多曲线非对称 bias 饱和节点 `SaturationNode`（`nodes/effects/saturation`）**（借鉴 UE/Unity/Godot/模拟饱和建模思想但不抄任何源码：提供可选 `tanh`/`arctan`/cubic/reciprocal/sine 五种闭式软削波传递函数，各归一化到 ±1——`arctan(x)*FRAC_2_PI`、cubic=`1.5c-0.5c^3` clamp 后硬限 ±1、reciprocal=`x/(1+|x|)`、sine=`sin(c*FRAC_PI_2)` clamp）：整形 `f(drive*x+bias)-f(bias)` 施加 `drive` 增益与非对称直流 `bias`（偏置引入偶次谐波、破奇对称，静音时 `f(bias)-f(bias)=0` 恒不漏 DC）；**复用 `waveshaper` 的 `Oversample` 配置枚举（X1/X2/X4）与同款多相 windowed-sinc 抗混叠过采样**（自带独立 `design_lowpass`/`zeroed`，与 `waveshaper` 固定对称 `tanh` 削波明确区分：本模块多曲线 + 非对称 bias + DC blocker），dry 路径同步延迟对齐过采样 `latency_frames` 保干湿相位相干；输出接一极点 **DC blocker** `y = x - x[-1] + R*y[-1]`（`R = DEFAULT_DC_BLOCK_COEFF = 0.9995`）去除偏置整形产生的直流；非有限输入按静音守卫；逐通道 `Smoothed` 快照/重放保多通道同轨，`drive`/`bias`/`output_gain`/`mix` 皆可平滑自动化；导出 `DEFAULT_DC_BLOCK_COEFF`/`SaturationCurve`/`SaturationNode`/`SaturationParams`；24 golden 对拍（默认中性/tanh 参考/reciprocal 公式/cubic 硬限/arctan 归一/sine 峰值/热输入有界于 ±1/bias 静音恒 0/bias 破奇对称/DC blocker 去 DC/mix 混干湿/output_gain 缩放/drive 增饱和/过采样报 latency/X1 零 latency/过采样有界有限/reset 确定性/多通道独立/非有限安全/零帧安全/set_curve 切换）+ 1 doctest；字母序插在 effects `ring_modulator` 与 `stereo_width` 之间；core 累计 **425 单测 + 14 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/saturation.rs` + `nodes/effects/mod.rs` + `nodes/mod.rs`，commit 9ef835157）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.59（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **`ISO 3382-1` 声强/相对声级指标 `SoundStrength`（`sound_strength`）**（对齐公开发布的 `ISO 3382-1` 客观房间声学参数：从单条宽带房间脉冲响应 RIR 量测厅堂「声强 / 相对声级 G」——即某测点接收总能量相对自由场 10m 参考能量 `E_ref` 的分贝比，表征空间对声能的放大/支持程度，与 `room_clarity`（清晰/混响侧 `C50`/`C80`/`EDT`）、`spatial_impression`（空间感 `LF`/`IACC`）、`room_acoustics`（几何预测侧）互补、互不重复实现）：核心公式 `G = 10*log10(sum p^2 / E_ref)` dB，早期分量 `G_early`（0-80ms）、晚期分量 `G_late`（80ms-末）同理分窗累计，`10*log10(.)` 由 `ops::ln(.)/LN_10` 实现（与 `room_clarity` 风格一致）；**复用 `room_clarity::EARLY_LATE_SPLIT_80_MS` 作早/晚分界（DRY，Relationship 伙伴）**，不另造常量；窗口按 `[..split]` 早 / `[split..]` 晚半开切分，`from_impulse_response` 单遍累计早/晚能量后相加得总能量、三值一致；能量 0 / 非正 `reference_energy` / 非正或非有限 `sample_rate` / 空响应 → 返回安全哨兵 `MIN_STRENGTH_DB=-100.0`，绝不 `-inf`/`NaN`/panic，非有限样本按 0 累计；离线控制率估计器（非逐样本回调、零堆分配、rustdoc 注明不得跑音频线程），确定性数学仅走 `bevy_math::ops`；导出常量 `MIN_STRENGTH_DB`、自由函数 `sound_strength_db`/`early_sound_strength_db`/`late_sound_strength_db`、`SoundStrength{g_db/g_early_db/g_late_db + Default（全 0）+ from_impulse_response(response,reference_energy,sample_rate) + serde "serialize" 门控}`；15 golden 对拍 + 1 doctest；字母序插在 `seat_dip_effect` 与 `source_directivity` 之间；spatial 累计 **502 单测 + 40 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`sound_strength.rs` + spatial `lib.rs`，commit 160fe01e2）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.58（**本版新增**：M6 母带/分析层 `prism_audio_core` 新增 **矢量示波器坐标生成器 `Goniometer`（`nodes/analysis/goniometer`）**（对齐公开发布的立体声矢量示波器/Lissajous 测角仪原理：把左右声道实时投影成旋转后的 mid/side XY 点云供立体声声场可视化，与 `correlation` 标量相位相关计量互补——后者报标量相关/宽度/平衡，本模块产可视化坐标点云，二者互不重复实现）：能量守恒旋转 `x = (L-R)*FRAC_1_SQRT_2`（side，水平轴）、`y = (L+R)*FRAC_1_SQRT_2`（mid，垂直轴），单声道（L=R）落为垂直线、反相（L=-R）落为水平线；定长环形缓冲一次分配（容量钳 `MIN_POINT_CAPACITY=1`、默认 `DEFAULT_POINT_CAPACITY=1024`），`decimation` 抽点降密（默认 `DEFAULT_DECIMATION=1`，`set_decimation` 复位相位），满后覆盖最旧点，`points_ordered(&mut [GoniometerPoint])->usize` 按时序回填并截断到切片容量；`peak_radius` 峰值半径、`GoniometerStats{peak_radius/rms_radius/sample_count:u64}` 以 f64 累加 `sum_radius_sq` 保精度后读回 `Sample`；非有限样本当静音、`reset` 清空点与相位；RT 热路径 `feed_sample(l,r)` 零分配/无锁/不 panic；`GoniometerNode{new/scope/scope_mut}` impl `AudioNode`——透传信号并喂每样本对（mono 当居中 L=R、多声道取前两路）；导出 `MIN_POINT_CAPACITY`/`DEFAULT_POINT_CAPACITY`/`DEFAULT_DECIMATION`、`GoniometerPoint{x,y}`+Default+serde、`GoniometerStats`+Default+serde、`Goniometer{new/capacity/decimation/set_decimation/len/is_empty/peak_radius/stats/feed_sample/points_ordered/reset}`、`GoniometerNode`；17 golden 对拍（mono 垂直/反相水平/旋转守能/decimation 抽点/ring 覆盖最旧/points_ordered 截断/peak_radius/stats rms+count/非有限静音/reset 清空/cap+decimation 钳位/set_decimation 复位相位/空输出切片/node 透传/node mono 居中/node reset/node 零帧）；字母序插在 analysis `correlation` 与 `loudness` 之间；core 累计 **404 单测 + 13 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/analysis/goniometer.rs` + `nodes/analysis/mod.rs` + `nodes/mod.rs`，commit 165582ae2）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.57（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **`ISO 3382-1` 空间感指标 `SpatialImpression`（`spatial_impression`）**（对齐公开发布的 `ISO 3382-1` 客观房间声学标准中的空间感参数：从实测脉冲响应量测厅堂「表观声源宽度 ASW」与「听者包围感 LEV」相关的客观指标，与 `room_clarity`（清晰/混响侧）、`room_acoustics`（几何预测侧）互补、互不重复实现）：表观声源宽度用早期侧向能量分数 `LF`（即 `JLF`）与余弦加权变体 `LFC`——`LF = sum_{5ms<=t<=80ms} pL^2 / sum_{0<=t<=80ms} p^2`、`LFC = sum_{5ms<=t<=80ms} |pL*p| / sum_{0<=t<=80ms} p^2`，由同点全向响应 `p` 与侧向八字形响应 `pL` 算得、钳 [0,1]；听者包围感用双耳互相关系数 `IACC`——归一化互相关函数 `IACF(tau)=sum l[t]r[t+tau]/sqrt(sum l^2 · sum r^2)`、`IACC=max_{|tau|<=1ms} |IACF(tau)|`，报早(0-80ms)/晚(80ms-末)/全三窗口；离线控制率估计器（非逐样本回调、单次有界堆分配、rustdoc 注明不得跑音频线程），空/全零/非有限/非正采样率输入返回安全默认 0 绝不 panic，确定性数学仅走 `bevy_math::ops`；导出常量 `LF_EARLY_START_MS=5`/`EARLY_WINDOW_END_MS=80`/`IACC_MAX_LAG_MS=1`、自由函数 `lateral_energy_fraction`/`lateral_energy_fraction_cosine`/`interaural_cross_correlation`、`SpatialImpression{lf/lfc/iacc_early/iacc_late/iacc_all + from_responses(omni,figure_eight,left,right,sample_rate) + Default + serde 门控}`；16 golden 对拍 + 1 doctest；spatial 累计 **487 单测 + 39 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`spatial_impression.rs` + spatial `lib.rs`，commit 7cc89d75f）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.56（**本版新增**：M6 母带层 `prism_audio_core` 新增 **母带级位深缩减/再量化器 `Dither`（`nodes/mastering/dither`，新建 `mastering` 子模块）**（对齐公开发布的量化抖动与噪声整形理论——Lipshitz/Vanderkooy/Wannamaker《Quantization and Dither》JAES 1992、Zolzer《DAFX》：把 32-bit 浮点混音透明渲染到交付位深，用抖动把量化误差从信号相关失真解耦成稳定本底噪声，区别于 lo-fi 的 `BitcrusherNode`——后者为音色劣化、不抖动不整形）：量化到 `2^bits` 码的 `[-1,1)` 栅格、步长 `q=2/2^bits`、`round(x/q)*q` 并把码钳到有符二补 `[-2^(bits-1),2^(bits-1)-1]`；抖动按 LSB 单位加——`RPDF` 一次均匀抽样 ±0.5 LSB、`TPDF` 两次独立抽样之和（三角密度、消除噪声调制）；噪声整形用误差反馈 FIR 拓扑 `u=x-sum h[k]e[n-k]`、`y=Q(u+dither)`、`e=y-u`，一阶 `h=[1]`（噪声传递函数 `1-z^-1`）/二阶 `h=[2,-1]`（`(1-z^-1)^2`），FIR 反馈无条件稳定；RT 热路径 `process_sample` 零分配/无锁/不 panic（非有限输入当纯净静音、绕过抖动与整形、不污染反馈态，通道越界与零帧安全退化），内置自包含 `xorshift64`（`SplitMix64` 播种）RNG 保证同参 bit 级可复现、golden 可对拍；导出常量 `MIN_DITHER_BITS=1`/`MAX_DITHER_BITS=24`/`DEFAULT_DITHER_BITS=16`、`DitherType{None,Rectangular,Triangular}`、`NoiseShaping{None,FirstOrder,SecondOrder}`、`DitherParams{bits,dither,shaping,seed}`+Default(16/TPDF/None)、`Dither{new/bits/quantization_step/dither_type/noise_shaping/channels/set_*/reset/process_sample}`、`DitherNode{new/engine/engine_mut}` impl `AudioNode`（透传量化）；18 golden 对拍（栅格步长/位深钳位/无抖动落格/有抖动落格/同种子位同/异种子分叉/reset 复现/抖动去相关安静音/噪声整形有界落格/非有限安全/越界通道返 0/满刻度不溢格/setters/node 透传量化/node 零帧/node reset 复现/RNG 属 [0,1)/1-bit 格）；core 累计 **387 单测 + 13 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/mastering/dither.rs` + `nodes/mastering/mod.rs` + `nodes/mod.rs`，commit 782825eed）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.55（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **`ISO 3382-1` 房间清晰度/能量比计量 `RoomClarity`（`room_clarity`）**（对齐公开发布的 `ISO 3382-1` 客观房间声学参数与 M. R. Schroeder 1965 反向积分法：从单条宽带房间脉冲响应 RIR（`&[Sample]`+采样率）量测厅堂主观听感相关的客观指标，区别于 `room_acoustics` 从几何/吸声预测宽带混响时间的 Sabine/Eyring 预测侧——本模块为实测验证侧，二者互补、互不重复实现）：核心 `energy_decay_curve` 用 Schroeder 反向积分 `EDC[n]=sum_{m>=n} h[m]^2`（从尾部向前累加平方能量，天然非增、远比原始平方响应平滑）归一到起点 0 dB、floor 约 -100 dB；`C50`/`C80` 清晰度 `10*log10(early/late)`（早/晚界分别在 50 ms 语音 / 80 ms 音乐）；`D50` 明晰度（Deutlichkeit）= 早/总能量比属 [0,1]；`Ts` 重心时间（能量一阶矩 `sum(t*h^2)/sum(h^2)`，秒）；`EDT`/`T20`/`T30` 混响时间用对 EDC 分别在 0..-10 / -5..-25 / -5..-35 dB 段做普通最小二乘直线拟合、斜率外推到 -60 dB 全程（10/20/30 dB 窗分别 x6/x3/x2），`from_impulse_response` 一次性复用同一条 EDC 供三条 RT 拟合、单遍扫描同时累计总/早(50/80ms)能量与重心矩；离线控制率估计器（非逐样本回调、仅一次有界堆分配存 EDC，rustdoc 注明不得跑音频线程）、空/全零/非有限/非正采样率输入返回安全默认（0/空曲线/钳位哨兵）绝不 panic，确定性数学仅走 `bevy_math::ops`（`ln`/`round`/`exp`）+ `LN_10`；导出常量 `EARLY_LATE_SPLIT_50_MS/EARLY_LATE_SPLIT_80_MS/EDT_UPPER_DB/EDT_LOWER_DB/T20_UPPER_DB/T20_LOWER_DB/T30_UPPER_DB/T30_LOWER_DB/MAX_CLARITY_DB`、自由函数 `energy_decay_curve/clarity_db/definition/center_time_s/early_decay_time_s/reverberation_time_t20_s/reverberation_time_t30_s`、`RoomClarity{c50_db/c80_db/d50/center_time_s/edt_s/t20_s/t30_s + from_impulse_response/definition_percent + Default + serde 门控}`；18 golden 对拍（EDC 首样本 0 dB / EDC 单调非增 / 空+全零+非有限+零采样率安全 / Dirac 单位明晰度与饱和清晰度 / 指数衰减 `T30` 对解析解 `3*ln(10)*tau/sr` / `EDT`~`T30`~`T20` 指数衰减一致 / 衰减越长 EDT 越大 / 衰减越快清晰度越高 / `C80` 独立求和校验 / 重心时间随衰减增大 / `D50` 属 [0,1] / `C50`=`10log10(D50/(1-D50))` 清晰度-明晰度恒等 / `from_impulse_response` 与自由函数一致 / `definition_percent`）+ 1 doctest；spatial 累计 **471 单测 + 38 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`room_clarity.rs` + spatial `lib.rs`，commit 68027ff55）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.54（**本版新增**：M6 母带/量测层 `prism_audio_core` 新增 **立体声相位相关/测角计量 `CorrelationMeter`（`nodes/analysis/correlation`）**（对齐公开发布的广播/母带相位相关表（phase correlation meter）与 goniometer 测角原理：量测立体声左右相干性与声场宽度，服务单声道兼容性检查与母带相位监看，区别于 `dynamics` 的信号整形——本模块只读旁路量测、不改信号）：以能量守恒的中侧变换 `M=(L+R)/sqrt(2)`、`S=(L-R)/sqrt(2)` 为基础，用指数滑动平均（复用本 crate `dynamics::detector::time_to_coef` EMA 弹道、默认积分 `DEFAULT_INTEGRATION_MS=300`）在线累计 L/R 自相关与互相关，输出 Pearson 相关系数 `correlation` 属 [-1,1]（+1 单声道同相、0 不相干、-1 反相）、中侧能量比导出的宽度 `width` 属 [0,1]、左右平衡 `balance` 属 [-1,1]、以及中/侧 RMS dBFS 电平；RT 热路径 `process_sample(l,r)` 零分配/无锁、非有限输入当静音防 NaN、全部状态构造期分配；导出 `CorrelationMeter{new/with_default_ballistic/coefficient/set_integration_ms/reset/process_sample/correlation/width/balance/mid_level_db/side_level_db/measurement}`、`CorrelationMeasurement{correlation/width/balance/mid_level_db/side_level_db}`、`CorrelationMeterNode`（impl `AudioNode` 透传）、`DEFAULT_INTEGRATION_MS`；20 golden 对拍 + 1 doctest；core 累计 **369 单测 + 13 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/analysis/correlation.rs` + `nodes/analysis/mod.rs` + `nodes/mod.rs`，commit 15541e9a1）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.53（**本版新增**：M6 母带合规层 `prism_audio_core` 新增 **`ITU-R BS.1770-4` / `EBU R128` 响度与真峰计量 `LoudnessMeter` / `LoudnessMeterNode`（`nodes/analysis/loudness`）**（对齐公开发布的广播响度标准 `ITU-R BS.1770-4`、`EBU R128`、`EBU Tech 3341/3342`，实现"按感知响度而非峰值归一"的次世代母带/合规工作流的量测侧，区别于 `dynamics` 的信号整形——本模块只读旁路量测、不改信号）：信号链 `K-weighting`（两级：高架头部预滤波 `fc=1681.9745Hz/Q=0.70717525/gain=3.9998439dB` + `RLB` 高通 `fc=38.13547Hz/Q=0.50032705`，复用本 crate `BiquadCoeffs::design` RBJ cookbook，不另造）→ 逐声道均方按 `BS.1770` 声道权重（前 L/R/C=1.0、环绕=`SURROUND_WEIGHT=1.41`、LFE 排除=0、FOA 仅 W 分量）加权求和 → `L=-0.691+10*log10(sum_i G_i*z_i)`（`-0.691 LU` 偏置校准 0 dBFS 997Hz 正弦读 0 LKFS）；**momentary**（滑动 400ms）/**short-term**（滑动 3s）LUFS 用重叠 100ms 子块环形窗；**integrated**（节目响度）双门控（绝对门 `-70 LUFS` + 相对门 `-10 LU` 低于未门控均值）经定长直方图（`-70..=+5 LUFS` @0.1LU、`HIST_BINS=751`）累计；**LRA**（`EBU Tech 3342`）取门控（相对门 `-20 LU`）short-term 分布 10/95 百分位之差；**true-peak** `dBTP` 用自设计 Hann 窗 sinc 4x 多相（`OVERSAMPLE_FACTOR=4`、每相 12 taps/proto 48、每相归一化单位 DC）过采样插值捕获采样间峰；RT 契约：全部状态（滤波记忆/滑窗环/双门控直方图/真峰插值器）构造期 `LoudnessMeter::new` 一次分配，热路径 `feed_sample`/`advance_frame`/`LoudnessMeterNode::process` 零分配/无锁/不 panic，报告查询 `integrated_lufs`/`loudness_range_lu` 各做一次有界直方图扫描（O(751)，rustdoc 注明仅控制率轮询）；导出常量 `LUFS_OFFSET/MOMENTARY_SUBBLOCKS/SHORT_TERM_SUBBLOCKS/ABSOLUTE_GATE_LUFS/RELATIVE_GATE_LU/LRA_RELATIVE_GATE_LU/SURROUND_WEIGHT/OVERSAMPLE_FACTOR`、`KWeighting{new/channels/reset/tick}`、`TruePeakMeter{new/channels/reset/process}`、`LoudnessMeasurement{momentary_lufs/short_term_lufs/integrated_lufs/loudness_range_lu/true_peak_dbtp}`、`LoudnessMeter{new/channels/feed_sample/advance_frame/reset/momentary_lufs/short_term_lufs/integrated_lufs/loudness_range_lu/true_peak_dbtp/measurement}`、`LoudnessMeterNode`（impl `AudioNode` 透传）；非有限/低于绝对门输入被 `bin_index` 拒绝（含 `NaN`）、退化输入返回 `-inf`/0 不 panic；确定性数学仅走 `bevy_math::ops`；22 golden 单测（含参考音近 -23 LUFS / +10dB 抬升约 +10 LU / 真峰检采样间过冲 / 满刻度真峰近 0 dBTP / 稳态 LRA 小 / 直方图门控 / K-weighting 频响 等）+ 1 doctest；core 累计 **349 单测 + 12 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/analysis/loudness.rs` + `nodes/analysis/mod.rs` + `nodes/mod.rs`，commit 4670b7a81）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.52（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **座椅掠射衰减 `SeatDipEffect`（`seat_dip_effect`）**（对齐 Schultz & Watters 1964、Sessler & West、Davies/Lovetri 公开文献记载的音乐厅"座椅陷波"Seat-Dip Effect：声波近掠射掠过成排座椅时在低频（文献记载约 100-300Hz）产生梳状插入损失，区别于户外 `ground_effect` 的地面反射干涉——本模块专刻厅堂座椅几何的掠射陷波）：掠射角用 `sin(theta)=mean_h/sqrt(mean_h^2+horizontal^2)` 直接构造（`bevy_math::ops` 无 atan/asin），掠射权重 `(1-sin).clamp(0,1)`（近掠射满陷波、近垂直无陷波）；陷波基频用座椅间隙四分之一波长共振 `f0=c/(4*seat_height)`（0.45m/343m·s⁻¹≈190Hz，落文献 100-300Hz 区间，座椅越高 f0 越低）；梳状形状=在 log 频率上对奇次谐波 `(2k+1)*f0`（k=0,1,2）叠加高斯陷波（权重 `1/(2k+1)`、`NOTCH_SIGMA=0.6 oct`）；行权重 `1-exp(-rows_crossed/6)`（`rows_crossed=horizontal/row_spacing`，随掠过排数饱和）；三权重相乘 ×`MAX_SEAT_DIP_DB=20` 钳 [0,20]、退化/非有限输入（非正座椅高/排距/距离/声速）返回全零谱不 panic；输出复用 `material_library` 的 `OCTAVE_BAND_COUNT=8`/`OCTAVE_BAND_CENTERS`、查询面镜像 `ground_effect`/`diffraction`（per-band dB + 线性增益 + log 频率插值端点钳位）；导出 `MAX_SEAT_DIP_DB`、`SeatDipGeometry{排间距/座椅高/源高/受高/水平距离 + Default/new，serde 门控}`、`SeatDipEffect{new/geometry/sound_speed/attenuation_db/band_attenuations_db/band_gains/broadband_attenuation_db/broadband_gain}`、自由函数 `seat_dip_attenuation_db(&SeatDipGeometry,sound_speed)->[Sample;8]` 为主入口；19 golden 单测 + 1 doctest；spatial 累计 **453 单测 + 37 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`seat_dip_effect.rs` + spatial `lib.rs`，字母序 `scattering` 与 `source_directivity` 之间，commit 7355ccdaf）；`# Provenance`（Schultz & Watters 1964 等文献，明确无 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）+ `# Relationship`（作为室内厅堂几何的加性 per-octave 项，与 ISO 9613-2 户外项同一 dB 预算约定协同）段齐全；恪守纯经典 DSP、无任何 AI/ML）
- 版本历史: v0.51（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **凸多面体房间镜像源早期反射 `ConvexRoom`（`convex_room`）**（把 `early_reflections` 的矩形 shoebox 整数网格镜像法推广到任意凸多面体房间——斜顶厅堂、切角、任意由若干平面围成的凸空间（半空间交集），对齐 Allen & Berkley 1979 JASA 镜像法向任意平面边界的公开推广；区别于 `early_reflections` 的闭式整数网格，本模块逐平面做镜像 + 线段求交 + 内侧剔除）：内向半空间约定 `dot(n,p)>=offset`、`offset=dot(n,point_on_plane)`、normal 单位化、零/非有限法线退化跳过；镜像源 `S'=S-2*(dot(n,S)-offset)*n`、线段 `L->S'` 与平面求交 `t=-dl/(di-dl)` 仅接受 `t in (0,1)`、交点需满足所有有效平面内侧（`contains`，`INSIDE_EPSILON=1e-4`）剔除房外/无效路径；总路径长 `|S'-L|`、延迟 `path/sound_speed` 经整数步进转整样本（避 f32->int cast）、增益 `reflection/max(path,0.1)`、到达方向=反射点→听者单位向量经 `listener.orientation.inverse()` 转局部帧（`-Z` 前向，与 `geometry` 一致）；`order` 钳 1（`order==0` 仅直达，更高阶按 1 处理，`# Scope` 注明）；`ConvexRoom::shoebox()` 桥接 `ShoeboxRoom` 面序、数学验证凸立方体一阶等价 Allen-Berkley shoebox 镜像（`cube_matches_shoebox_first_order_count`/`cube_matches_shoebox_image_distances` 对拍）；复用 `ReflectionTap`（`delay_samples:usize`）/`MAX_EARLY_REFLECTIONS=32` 溢出 top-k 保最强、全走 `bevy_math::ops::sqrt`；`MAX_PLANES=12`、`ReflectionPlane{normal/offset/reflection + new/from_absorption(sqrt(1-alpha))/is_valid}`、`ConvexRoom{new/shoebox/planes/count/contains}`、`ConvexReflections{taps/count/is_empty}`、自由函数 `compute_convex_reflections(room,listener,source,order,sample_rate,sound_speed)` 为主入口，全栈定长数组无堆分配/锁/panic、退化安全（非有限听者/声源→空集、听者=声源→局部前向回退）、确定性数学仅走 `bevy_math::ops`、serde 门控、`# Provenance` + `# Relationship` + `# Scope` 段齐全、全 ASCII、no_std；20 golden 对拍（空房仅直达 / 直达指向声源 / order=0 仅直达 / 单面反射几何 / 反射点落在平面上 / 增益随距离衰减 / 反射系数缩放增益 / 面外声源被剔除 / 立方体一阶数等价 shoebox / 立方体镜像距离等价 shoebox / 听者=声源安全 / 非有限听者空集 / 非有限声源空集 / 零法线平面跳过 / order>1 钳到 1 / 输出数有界 / 方向单位长 / 局部帧旋转方向 / 喂 cluster_taps 不 panic / from_absorption 等价 sqrt 规则）+ 1 doctest；spatial 累计 **434 单测 + 36 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`convex_room.rs` + spatial `lib.rs`，commit ec1b24e63）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.50（**本版新增**：M1 核心层 `prism_audio_core` 新增 **磁带机模拟效果 `TapeNode`（`tape`）**（对齐 Udo Zolzer《DAFX: Digital Audio Effects》公开记载的模拟磁带着色思想：磁介质软饱和 + 走带速度抖动 wow/flutter + 磁头/缝隙高频损耗三要素合一，区别于 `waveshaper` 的纯整形与 `vibrato` 的纯音高调制——本模块叠加三者得到完整磁带音色）：逐通道逐样本信号链 `saturated=saturate(x)`→`filtered=one_pole_lp(saturated)`→`wet=delay_tap(filtered,d[n])`→节点层 `y=(1-mix)*x+mix*wet`；饱和曲线 `saturate(x)=(tanh(drive*x+bias)-tanh(bias))/(drive*(1-tanh(bias)^2))`（单位小信号增益归一 + 过原点 DC 校正，非零 bias 产生偶次谐波，`norm<=0` 退化恒等保 zero-drive 安全）；瞬时读延迟 `d[n]=center+wow_depth*wow[n]+flutter_depth*flutter[n]` 钳 `[1,max_delay]`，`center=wow_depth+flutter_depth` 保读头不超写头，wow/flutter 为两枚本 crate `Lfo`（Sine）实例；一极点低通 `y+=a*(x-y)`、`a=1-exp(-TAU*fc/sr)` 钳 [0,1] 做磁头高频滚降；`MAX_DRIVE=64`/`MAX_BIAS=3`（保 `1-tanh^2(bias)` 不近零）/`MAX_DELAY_MS=100`（钳 wow/flutter 深度，防病态参数请求无界环形缓冲分配）、自由函数 `clamp_drive/clamp_bias/clamp_depth_ms/normalising_gain/one_pole_coef`；`TapeParams{drive,bias,wow_hz,wow_depth_ms,flutter_hz,flutter_depth_ms,hf_rolloff_hz,mix}` derive Default（drive=2/bias=0.3/wow=0.7Hz/wow_depth=2ms/flutter=7Hz/flutter_depth=0.4ms/HF=12kHz/mix=1）+ 可选 serde；`Tape` DSP 核心 `{new/channels/drive/bias/set_drive/set_bias/set_hf_rolloff_hz/set_wow_hz/set_flutter_hz/saturate/advance->delay/voice(ch,x,delay)->wet/commit_frame/reset}` 全状态构造期预分配、每帧调用顺序 advance 一次→逐通道 voice→commit_frame 一次、非有限入参当静音防 NaN 中毒、`flush_denormal` 写回 LP 态、复用 vibrato 分数延迟 idiom（`base as isize`+`rem_euclid`+`lerp`）；`TapeNode`（impl `AudioNode`）干湿经 `Smoothed` mix 无咔哒混音、通道独立共享 coeffs 与一对 wow/flutter 振荡器保相位相干；23 golden 对拍（饱和有界有限 / 大峰压缩 / 小信号近单位增益 / 过原点 / bias 破对称 / 零 bias 对称 / 零 drive 直通 / drive 钳 [0,64] / bias 钳 [-3,3] / HF rolloff 衰高频 / wow/flutter 移动信号 / 深 wow 展宽延迟摆幅 / 静默→静默 / 非有限安全 / dry mix 透明 / reset 可复现 / 通道独立 / 立体声相位相干 / node reset 清尾 / 极端参数不 panic / 零帧 no-op / clamp_depth_ms 有界 / one_pole_coef 有界 / collect into Vec）+ 1 doctest；core 累计 **327 单测 + 11 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/tape.rs` + `nodes/effects/mod.rs`，commit 50c532860）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.49（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **早期反射方向聚类器 `ReflectionClusters`（`reflection_clustering`）**（把镜像源早期反射抽头列表按到达方向聚类成固定 6 个方向簇、能量守恒聚合，对齐游戏音频中间件 baked-reflection 管线把众多反射折叠成少量方向送出的通行做法；区别于 `reflection_directivity`（按源指向性加权同一批抽头）——本模块是其姊妹归约层，二者可组合「先加权后聚类」或独立使用，均不重复实现镜像源几何）：6 个轴对齐固定方向簇（`+X,-X,+Y,-Y,+Z,-Z`，`-Z` 为前向），每抽头按 `argmax(dot(dir,canonical))` 归入最近簇；能量守恒的非相干聚合——簇能量 `E=sum(gain^2)`、报告增益 `sqrt(E)`，全簇能量和恒等于全抽头能量和；代表方向=能量加权平均方向（归一化，退化回退簇规范方向）、`delay_samples`=能量加权平均延迟（`ops::round`）；非有限/零长方向、非有限增益、零能量抽头安全跳过；`CLUSTER_COUNT=6`、`ReflectionCluster{direction/gain/delay_samples/tap_count + is_active}`、`ReflectionClusters{from_taps/clusters/cluster/count/active_count/total_energy}`、自由函数 `cluster_taps` 为主入口，全栈定长 6 元数组无堆分配/锁/panic，用 `count()` 规避 `len_without_is_empty`、确定性数学仅走 `bevy_math::ops`（sqrt/round）、serde 门控、`# Provenance` + `# Relationship` 段齐全、全 ASCII、no_std；17 golden 对拍（簇数恒为 6 / 空集零能量无活跃 / 空簇保规范方向 / 单方向归一簇 / 聚合增益=能量和 / 跨簇能量守恒 / 代表方向=能量加权均值 / 代表方向单位长 / 延迟=能量加权均值 / 延迟偏向更响抽头 / 六轴均匀均分 / 非有限方向跳过 / 非有限增益跳过 / 零方向跳过 / 零增益无贡献 / 越界簇返回 None / from_taps 等价自由函数）+ 1 doctest；spatial 累计 **414 单测 + 35 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`reflection_clustering.rs` + spatial `lib.rs`，commit f13aebbcd）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.48（**本版新增**：M1 核心层 `prism_audio_core` 新增 **谐波激励器/听感增强节点 `ExciterNode`（`exciter`）**（对齐 Udo Zolzer《DAFX: Digital Audio Effects》公开记载的 Aphex Aural Exciter 式听感激励器思想：分频→生成高次谐波→混回，区别于 `parametric_eq` 的减法重塑频谱——本模块是**生成式**地添加全新谐波内容）：逐样本信号链 `band=highpass(x,fc)`→`shaped=shape(drive*band)`→`harmonics=highpass(shaped,fc)`→节点层 `y=x+amount*harmonics`，用**两级高通 `Svf`**（Butterworth Q=1/√2 TPT 核，复用本 crate `svf` 原语，crossover 调制相位相干）夹一个 `tanh` 系波形整形器——后置高通只保留整形新生成的高次谐波、丢弃易混叠的低次积；`HarmonicMode{Odd,Even,Mix}`：Odd=对称 `tanh` 奇次谐波（阀门式明亮）、Even=DC 校正偏置 `tanh`（`biased_tanh=tanh(driven+bias)-tanh(bias)`，`EVEN_BIAS=0.5`）偶次谐波（八度味甜音）、Mix=二者均值；`ExciterParams{frequency_hz,mode,drive,amount}` derive Default（3000Hz/Even/drive=2/amount=0.25）+ 可选 serde；`Exciter` DSP 核心 `{new/channels/frequency_hz/drive/mode/set_frequency_hz(重设两级 coeffs)/set_drive(钳 [0,64])/set_mode/harmonics(ch,x)->只返回谐波/reset}` 全状态构造期预分配、非有限入参当静音防 NaN 中毒、`flush_denormal` 写回防非正规数、`clamp_drive` 钳 `MAX_DRIVE=64`，`harmonics` 实时安全（无分配/锁/panic）；`ExciterNode`（impl `AudioNode`）干声直通 + `amount` 经 `Smoothed` 无咔哒混入谐波、逐帧逐通道 `x+amount*harmonics`，通道独立共享 coeffs；19 golden 对拍（低频被拒 / 高频生谐波 / drive 越大谐波越多 / 静默→静默 / 非有限安全 / odd 对称 / even 破对称 / even 零点无 DC / shaper 有界 ≤2 / drive 钳 [0,64] / freq 钳正 / reset 可复现 / 通道独立 / node amount=0 直通 / node 激励改变信号 / node reset 清尾 / mix 模式有限 / 极端参数不 panic / 零帧 no-op）+ 1 doctest；core 累计 **304 单测 + 10 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/exciter.rs` + `nodes/effects/mod.rs`，commit 70b3f47db）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.47（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **方向感知早期反射加权组合器 `DirectionalEarlyReflections`（`reflection_directivity`）**（把源指向性按每倍频程辐射增益施加到镜像源早期反射抽头上：区别于 `source_directivity`（只算辐射方向图）与 `early_reflections`（只算镜像几何），本模块是二者之上的 DRY 组合层，不重复实现任一方物理）：控制率无 per-sample DSP，消费 `SourceDirectivity` 的辐射方向图与 `early_reflections::ReflectionTap` 的抽头方向/增益并相乘——对每抽头以「(单位)源前向轴 · (单位)抽头到达方向」求离轴余弦 `cos`（`emission_cos`，钳 [-1,1]；直达路径镜像与源重合故到达方向即发射方向精确，高阶反射以到达方向作发射方向的远场代理）、`tap_band_gains=tap.gain*directivity.band_gains(cos)` 求每带加权幅度、`tap_broadband_gain=tap.gain*directivity.broadband_gain(cos,freq)`（log 频率插值不外推）求宽带、`total_band_energy` 非相干平方和求每带总能量、`total_send_gain=rms(每带能量)` 求早反射送出总线单标量增益（空集=0）；`DirectionalEarlyReflections{new(directivity,source_forward)/from_preset(preset,forward)/directivity/source_forward/emission_cos/tap_band_gains/tap_broadband_gain/total_band_energy/total_send_gain}`，前向轴归一化（零/非有限退化为 -Z）、抽头非有限方向当 on-axis、tap.gain 钳非负有限，全栈标量 + 定长 8 元数组无堆分配/锁/panic，倍频程对齐 `OCTAVE_BAND_CENTERS`、`bevy_math::ops` 确定性数学、serde 门控、`# Provenance` + `# Relationship` 段齐全、全 ASCII、no_std；14 golden 对拍（全向各带=tap.gain / 全向总能均匀 / 心形后向抽头静默 / 心形前向满增益 / 高带比低带离轴衰更多 / emission_cos 几何 / omni 送出总增益手算 / broadband 带心=带值 / 带间插值 / 端点钳位 / 零前向退化 -Z / 非有限入参安全 / 空集零能量 / 直达用精确发射方向）+ 1 doctest；spatial 累计 **397 单测 + 34 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`reflection_directivity.rs` + spatial `lib.rs`，commit 060ec5ca0）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.46（**本版新增**：M1 核心层 `prism_audio_core` 新增 **包络滤波自动哇音节点 `AutoWahNode`**（对齐 Udo Zolzer《DAFX: Digital Audio Effects》与 Will Pirkle《Designing Audio Effect Plugins in C++》公开记载的 envelope-follower 驱动谐振滤波器/自动哇音理论，区别于 `tremolo`/`phaser` 的 LFO 驱动——本模块由**振幅包络**驱动扫频）：单路 mono 侧链（每帧跨通道 peak）驱动一个整流 attack/release 一极点包络跟随器 `env`（复用 dynamics 家族 `time_to_coef` 的弹道系数，`coef = if rectified>env {attack} else {release}` 解耦快升慢落），灵敏度 drive 经软膝 `amount=1-exp(-drive)`（`drive=(env*sensitivity).clamp(0,8)`）映射到 `[0,1)`，再指数扫频 `cutoff = base_hz * 2^(sweep_octaves*amount*direction)`（`direction=±1` 上/下扫，`clamp` 到 `[1, sr*0.499]` 开奈奎斯特带）驱动共享 **`Svf`** 核心（TPT 梯形积分，快扫截止有界无咔哒，正是包络滤波所需，Direct Form I biquad 在此会咔哒）；`WahMode{BandPass,LowPass,Peak}`（`const fn svf_kind` 映射 `SvfKind`）+ `SweepDirection{Up,Down}`（`const fn sign->±1`），`AutoWahParams{mode,direction,base_freq_hz,sweep_octaves,q,sensitivity,attack_ms,release_ms,wet,dry}` derive Default（BandPass/Up/300Hz/3 oct/Q=3/sens=8/atk 5ms/rel 120ms/wet 1/dry 0）+ 可选 serde；`AutoWah` DSP 核心 `{new/channels/envelope/cutoff_hz/advance(sidechain)->cutoff/filter(ch,x)/reset}` 全状态构造期预分配、非有限侧链当静音防 NaN 中毒、`flush_denormal` 写回包络防非正规数，`advance`/`filter` 实时安全（无分配/锁/panic）；`AutoWahNode`（impl `AudioNode`）wet/dry 用 `Smoothed` 无咔哒、逐帧算通道 peak 侧链再逐通道 `dry*x+wet*filter`；20 golden 对拍（静默停 base / 变响上扫 / down 方向下扫 / cutoff 钳奈奎斯特 / 零 octaves 钉住 / 包络逼近输入 / attack 快于 release / 非有限侧链安全 / reset 复原 / reset 可复现 / 通道独立 / bandpass 拒 DC / lowpass 过 DC / node 有限且混音 / node 全 dry 直通 / node reset 清尾 / 极端参数不 panic / 零帧 no-op）+ 1 doctest；core 累计 **285 单测 + 9 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/auto_wah.rs` + `nodes/effects/mod.rs`，commit a9b178263）；至此 svf 铺路的调制滤波族首个成员就位；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.45（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **ISO 9613-2 户外传播总预算聚合模块 `outdoor_propagation`**（对齐 ISO 9613-2:1996《户外声传播衰减》第二部分总衰减预算 `A=A_div+A_atm+A_gr+A_bar`，区别于四个分量模块各自只算一项）：把此前已就位的四项加性衰减分量组合成单一每倍频程总预算的 DRY 组合器——几何散度 `A_div` 委托 `Attenuation::gain` 求 `-linear_to_db(gain)`（`MAX_DIVERGENCE_DB=200` 封顶防 `-inf`）、大气吸收 `A_atm` 委托 `air::absorption_db_per_metre(&conditions, centre)*distance`、地面效应 `A_gr` 委托 `ground_effect::attenuation_db(band)`、屏障绕射 `A_bar` 委托 `diffraction::insertion_loss_db(band)`（`Option<Diffraction>`，`None`=视线直达 0 损失），`total=A_div+A_atm+A_gr+A_bar` 每带求和、无任何物理公式重复实现（各分量真源唯一，本模块只做聚合）；`OutdoorPropagation{new(distance_m,attenuation,conditions,ground,barrier:Option<Diffraction>)/line_of_sight(...), 分项 divergence_db/atmospheric_db(band)/barrier_db(band)、总量 total_attenuation_db(band)/band_attenuations_db()->[8]/band_gains()->[8]、宽带 broadband_attenuation_db(freq)（log 频率插值 + 端点钳位不外推）/broadband_gain(freq), getters distance/has_barrier/ground_effect/barrier}`；`clamp_distance`（非有限→0/钳非负）、`MIN_DIVISOR=1e-9`、复用 `OCTAVE_BAND_CENTERS/OCTAVE_BAND_COUNT`、仅导出 `OutdoorPropagation`、serde 门控、`# Provenance` + `# Relationship` 段齐全（定位为 `attenuation`/`air`/`ground_effect`/`diffraction` 四分量之上的顶层聚合），确定性数学走 `bevy_math::ops`（仅 `ln`）、全 ASCII、no_std；14 golden 对拍（分量求和对拍 / 带屏障求和 / 无屏障屏障项=0 / 屏障加正损 / band_gains 与 dB 一致 / 散度随距增 / 散度静默仍有限 / 大气随频升 / 大气随距 x3 / 零距零散度大气 / broadband 带心=带值 / 带间插值 / 端点钳位 / 非有限退化安全）+ 1 doctest；spatial 累计 **383 单测 + 33 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`outdoor_propagation.rs` + spatial `lib.rs`，commit d6766a2db）；至此 ISO 9613-2 户外传播家族的几何扩散/大气吸收/地面效应/屏障绕射四项分量 + 总预算聚合已全部就位；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.44（**本版新增**：M1 核心层 `prism_audio_core` 新增 **拓扑保持状态变量滤波节点 `SvfNode`**（对齐 Andrew Simper/Cytomic「梯形积分求解连续 SVF」2013 技术论文与 Vadim Zavalishin《The Art of VA Filter Design》公开虚拟模拟滤波理论，区别于 `biquad` 的 RBJ cookbook Direct Form I）：单次双线性预扭 `g=tan(PI*fc/fs)` 与阻尼 `k=1/Q` 驱动两积分器，逐样本更新 `v3=v0-ic2eq; v1=a1*ic1eq+a2*v3; v2=ic2eq+a2*ic1eq+a3*v3; ic1eq=2*v1-ic1eq; ic2eq=2*v2-ic2eq`（`a1=1/(1+g*(g+k))`、`a2=g*a1`、`a3=g*a2`），band=v1/low=v2，一次输出混合 `y=m0*v0+m1*v1+m2*v2` 即可从同一对内部信号读出全部响应；`SvfKind` 九种响应 LowPass/HighPass/BandPass/Notch/Peak/AllPass/Bell/LowShelf/HighShelf（Bell 用 `k=1/(Q*A)`、LowShelf `g/=sqrt(A)`、HighShelf `g*=sqrt(A)`，`A=10^(gain_db/40)` 由私有 `pow10=exp(x*LN_10)` 求），各响应的 `m0/m1/m2` 混合权重在 `SvfCoeffs::design(kind,sample_rate,freq_hz,q,gain_db)` 设计期烘焙（freq 钳开奈奎斯特带 [1,sr*0.499]、q 钳 [1e-4,∞) 恒数值有效）；该拓扑核心卖点是**快速扫截止频率时状态有界无咔哒**（自动哇音/包络滤波器/滤波 LFO 所需，Direct Form I biquad 在快扫时会咔哒或发散），`SvfNode{new(params,sample_rate,channels), set_freq_hz(freq,Ramp), set_kind, set_q, set_gain_db}` 用 `Smoothed` 平滑截止频率——smoother settled 时每块设计一次定系数走 `process_inplace`，扫动时逐样本重设计并跨通道从同一平滑值 `tick`（TPT 稳定）；`SvfParams{kind,freq_hz,q,gain_db}` derive Default（LowPass/1kHz/Q=1/√2/0dB）+ 可选 serde，`Svf{new/from_params/set_coeffs/coeffs/channels/tick/process_inplace/reset}` 分配一次每通道 `[ic1eq,ic2eq]`、非有限入参当静音回退防递归状态中毒、`flush_denormal` 写回积分器与输出防非正规数堆积；17 golden 对拍（低通过 DC 拒高频 / 高通拒 DC 过高频 / 带通中心峰 3x 于旁瓣 / 陷波中心拒>16dB DC 过 / 全通幅频平坦±0.1 / 峰值有限谐振 / bell ±12dB 中心 x3.98/x0.25 且旁频≈1 / 低架 DC≈3.98 高频回≈1 / 高架高频>3 DC≈1 / 静音进静音出 / 非有限进有限出 / reset 复现 / 通道独立 / 极端参数不 panic / 零帧 no-op / 节点平滑扫频有限 / 节点 reset 清尾）+ 1 doctest；core 累计 **267 单测 + 8 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/svf.rs` + `nodes/mod.rs`，commit 455cccb84）；为后续 auto_wah/envelope-filter 等调制滤波铺路；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.43（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **ISO 9613-2 地面效应衰减模块 `ground_effect`**（对齐 ISO 9613-2:1996 §7.3.1 户外传播地面项 `A_gr`）：地面反射波与直达波干涉，硬地（G=0，混凝土/水面）建设性增益、软地（G=1，草地/农田/雪）低-中频衰减（经典 ground dip）；控制率估计器（不做逐样本 DSP），三区地面因子 `G∈[0,1]`（源区 `G_s`/中区 `G_m`/受区 `G_r`），总衰减 `A_gr=A_s(G_s,h_s)+A_r(G_r,h_r)+A_m(G_m,q)` 三区加性预算；源/受区按 Table 3 每倍频程逐带公式：63Hz 常数 `-1.5`、125/250/500/1kHz 用高度函数 `-1.5+G*{a,b,c,d}(h,d_p)`（`a=1.5+3*exp(-0.12*(h-5)^2)*(1-exp(-d_p/50))+5.7*exp(-0.09*h^2)*(1-exp(-2.8e-6*d_p^2))`、`b=1.5+8.6*exp(-0.09*h^2)*(1-exp(-d_p/50))`、`c=1.5+14*exp(-0.46*h^2)*(1-exp(-d_p/50))`、`d=1.5+5*exp(-0.9*h^2)*(1-exp(-d_p/50))`）、2k/4k/8kHz 用 `-1.5*(1-G)`；中间区权重 `q=1-30*(h_s+h_r)/d_p`（`d_p<=30*(h_s+h_r)` 时 `q=0`），63Hz `-3*q`、≥125Hz `-3*q*(1-G_m)`；忠实实现 Table 3 精确公式无近似占位、有意允许负衰减（硬地增益>1 物理正确）；复用 `material_library::{OCTAVE_BAND_CENTERS,OCTAVE_BAND_COUNT}`，`# Relationship` 段定位为与 `attenuation`（几何扩散）/`air`（大气吸收）/`diffraction`（屏障绕射）并列的 ISO 9613-2 加性预算项；`GroundEffect{from_geometry(hs,hr,dp,G)/from_regions(gs,gm,gr,hs,hr,dp), 6 getters, attenuation_db(band)/band_attenuations_db()->[8]/band_gains()->[8], broadband_attenuation_db(f)（log 频率插值端点钳位）/broadband_gain(f)}` serde 门控、仅导出结构体、地面因子钳 [0,1]/高度距离钳非负/非有限入参安全回退、确定性数学走 `bevy_math::ops`（`exp`/`ln`）；14 golden 对拍（软地 250Hz=10.376dB/500Hz=3.845dB 手算对拍 Table 3 / 硬地全带 -3dB / 软地高频归零 / 中区权重随距增长 q(200)=0.4 q(400)=0.7 / 短程中区失活 / band_gains 与 dB 一致 / 硬地增益>1 / broadband 带心对齐 + 带间插值 + 端点钳位 / 分区因子存储与钳位 / 非有限与退化安全）；spatial 累计 **369 单测 + 32 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`ground_effect.rs` + spatial `lib.rs`，commit 5294d8fdf）；至此 ISO 9613-2 户外传播家族的几何扩散/大气吸收/屏障绕射/地面效应四项加性衰减已就位；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.42（**本版新增**：M1 核心层 `prism_audio_core` 新增 **单 LFO 扫频分数延迟颤音 `VibratoNode`**（对齐 U. Zoelzer DAFX 调制延迟线颤音）：读指针距离由低频振荡器扫动 `d[n]=depth*(1+lfo[n])`（lfo∈[-1,1] 扫 [0,2*depth]、钳 [1,max_delay] 保分数读跨两有效样本），线性插值抽头与干信号按 mix 混合 `y[n]=(1-mix)*x[n]+mix*tap(d[n])`（默认 mix=1 纯 wet）；单 LFO 每帧推进一次跨通道共享相位保相干（mono 源喂多通道仍相位一致），区别于 chorus 多抽头 wet+dry 合奏、flanger 反馈梳染色（vibrato 无反馈只做音高 movement）；`VibratoParams{rate_hz(默认5), depth_ms(默认2), mix(默认1), waveform(默认Sine)}` derive Default + 可选 serde，`VibratoNode{new(sample_rate,channels,params), channels, set_rate_hz, set_waveform, set_depth_ms(钳到 0.5*max_delay,ramp), set_mix(钳[0,1],ramp)}` impl AudioNode、depth/mix 用 Smoothed 平滑防咔哒、`reset` 清环与 write_pos/lfo/重建 Smoothed、非有限入参安全回退、分数延迟照 delay/chorus idiom（floor+rem_euclid+lerp、flush_denormal 写回）；12 golden 对拍（mix=0 bypass / 有界有限 / rate=0 静态 LFO=固定 depth 延迟冲激重现 / half mix 半干半湿 / 调制偏离干信号 / stereo 通道相位相干 / 负 depth 钳零 / mix 钳位 / 非有限安全 / reset 复现 / 零帧不 panic / 深 depth 展宽扫幅）；core 累计 **250 单测 + 7 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/vibrato.rs` + `nodes/effects/mod.rs` + `nodes/mod.rs`，commit 14aa4a8a7）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.41（**本版新增**：M4 空间层 `prism_audio_spatial` 新增 **边缘/屏障绕射插入损失模块 `diffraction`**（对齐 Z. Maekawa 1968 屏障降噪 / Kurze-Anderson / ISO 9613-2 屏障项）：控制率估计器（不做逐样本 DSP），Fresnel 数 `N=2*delta*f/c`（delta 为绕边程差、f 频率、c 可配置声速；N 随程差与频率增大故高频阴影更强），几何阴影区（N>0）Maekawa 插入损失 `IL_dB=5+20*log10(x/tanh(x))`（`x=sqrt(2*PI*N)`）钳到 `MAX_DIFFRACTION_DB=24`、视线内（delta<=0）损失 0；**DRY 复用 crate 内唯一真源 `propagation::maekawa_attenuation_db`**（不重复实现 Maekawa 公式，也回避 bevy_math::ops 无 tanh/log10 的问题，统一阴影边界地板与 24dB 上限），本模块只负责用可配置声速算 N（区别于 propagation 的自由函数硬编码声速、构建滤波截止），并按八倍频程（复用 `material_library::{OCTAVE_BAND_CENTERS,OCTAVE_BAND_COUNT}`）报告结构化插入损失/增益，与 `source_directivity`/`reverberant_field` 频谱对齐；`Diffraction{from_path_difference(delta,c)/from_geometry(source,edge,receiver,c)（delta=|s-e|+|e-r|-|s-r| 三角不等式非负）, path_difference/sound_speed, fresnel_number(f), insertion_loss_db_at(f)/insertion_loss_db(band)/band_losses_db()->[8]/band_gains()->[8], broadband_loss_db(f)（log 频率插值端点钳位）/broadband_gain(f)}` serde 门控、仅导出结构体避与 propagation 同名自由函数冲突、退化几何与非有限入参安全回退、`length` 用 `ops::sqrt(v.dot v)` 避 f32 intrinsic；12 golden 对拍（视线零损失 / 高频阴影更强单调且跨带>5dB / N=1->13.097dB Maekawa 对拍 / 上限钳位到 24dB / band_gains 与 dB 一致 / broadband 带心对齐 + 带间插值 + 端点钳位 / 几何 delta=2sqrt(2)-2 / 边在直线上零绕射 / 非有限与退化安全 / 声速升高降低 Fresnel 数）；spatial 累计 **355 单测 + 31 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`diffraction.rs` + spatial `lib.rs`，commit 00174b655）；至此 ISO 9613-2 户外传播家族的几何扩散（attenuation）/大气吸收（air）/屏障绕射（diffraction）三项加性衰减已就位；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.40（**本版新增**：M1 核心层 `prism_audio_core` 新增 **调谐反馈梳状谐振器 `CombResonatorNode`**（对齐 Karplus-Strong 1983 / Jaffe-Smith / Schroeder-Moorer 低通反馈梳）：分数延迟反馈环 `filtered[n]=(1-damping)*y[n-D]+damping*filtered[n-1]`、`y[n]=x[n]+feedback*filtered[n]`，`D=sample_rate/frequency_hz`（线性插值分数延迟，连续调谐）；环内一极点低通做阻尼（damping 越高高频衰减越快、音色越暗），feedback 钳 `[0,0.999]` 保证衰减，弦体/共鸣音色（区别于 delay 回声、flanger 短反馈扫频、chorus 无反馈合奏）；`CombResonatorParams{frequency_hz(默认220), feedback(默认0.9), damping(默认0.2), mix(默认1)}` derive Default + 可选 serde，`CombResonatorNode{new(sample_rate,layout,params), channels/delay_frames/feedback/damping/mix getter, set_frequency_hz(sr,钳[20,Nyquist]), set_feedback(钳[0,0.999]), set_damping(钳[0,1]), set_mix(钳[0,1])}` impl AudioNode、`reset` 清环与滤波状态、每通道独立环缓冲 + 一极点状态、非有限入参安全回退；14 golden 对拍（mix=0 bypass / 静音入静音出 / 无阻尼冲激按环周期重复且第 k 次=feedback^k / 频率设定环周期 / 重调谐改周期 / 阻尼衰减重复峰 / 高 feedback 尾能量更长 / 满阻尼杀谐振 / 参数钳位 / 非有限回退 / 通道独立 / 满 feedback 长块有界有限 / reset 复现 / 零帧不 panic）；core 累计 **238 单测 + 6 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/comb_resonator.rs` + `nodes/effects/mod.rs` + `nodes/mod.rs`，commit 23fb6479b）；同时 M4 空间层 `prism_audio_spatial` 新增 **稳态混响场能量与直混比模块 `reverberant_field`**（对齐 L. Beranek Acoustics / H. Kuttruff Room Acoustics）：房间常数 `R=S*a_bar/(1-a_bar)`（a_bar 钳 (0,0.999999) 保 R 有限）、直达能量 `Q/(4*PI*r^2)` + 混响（扩散、距离无关）能量 `4/R`、直混比 `DRR(r)=Q*R/(16*PI*r^2)` 及 dB 版 `10*log10(DRR)`、临界距离 `r_c=sqrt(Q*R/(16*PI))`（DRR=1 处）；声源功率 W 与声速在所有比值中相消故不作输入；`ReverberantField{from_surface_absorption(S,a_bar)/from_room_constant(R), room_constant, direct_energy(q,r), reverberant_energy, total_energy(q,r), direct_to_reverberant_ratio(q,r)+_db, critical_distance(q)}` serde 门控，与 `source_directivity` 的 `directivity_factor` 的 Q 天然协同、`# Relationship` 段交叉引用 `room_acoustics::critical_distance`（Sabine V/RT60 版，语义等价）；11 golden 对拍（R 闭式 / a→1 大而有限 / rc 处 DRR=1 / DRR∝1/r² / DRR_dB=10log10 / 直达 6dB/倍距 / 混响距离无关 / 高 Q 抬 DRR 并外推 rc / total=direct+reverb / 退化与非有限安全 / from_room_constant 往返）；spatial 累计 **343 单测 + 30 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`reverberant_field.rs` + spatial `lib.rs`，commit 651ccd5f4）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.39（**本版新增**：M1 核心层 `prism_audio_core` 新增 **环形调制器 `RingModulatorNode`**（对齐经典模拟环调拓扑 Zoelzer DAFX）：`output = input * ((1 - mix) + mix * carrier)`，载波为共享的**双极** `Lfo`（区别于 tremolo 的单极 AM：双极载波在 `[-1,1]` 每周期两次反相，抑制载波本身并产生 `f_in +/- f_c` 和差边带，得到非谐的铃/金属/机器人音色）；单个音频率载波跨通道共享保相位相干（每帧一个 `carrier=next_sample()`、逐通道 `out=flush_denormal(in*(dry+mix*carrier))`） + wet/dry `mix` 混合；`RingModulatorParams{carrier_hz(默认440), waveform(默认Sine), mix(默认1)}` derive Default + 可选 serde，`RingModulatorNode{new(sample_rate,params)（无 layout 入参，单载波共享）, set_carrier_hz(sr,钳>=0), set_waveform, set_mix(钳[0,1])}` impl AudioNode、`reset` 重置载波相位；10 golden 对拍（full-wet 输出=载波（首帧≈0、峰>0.9、在[-1,1]）/ mix=0 bypass / square 载波极性翻转 / 幅度不超输入 / 载波跨通道共享 / 干湿线性混合 / 参数钳位 / reset 复现 / 零帧不 panic / 四波形输出有限）；core 累计 **224 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/ring_modulator.rs` + `nodes/effects/mod.rs` + `nodes/mod.rs`，commit 93f104dc5）；同时 M4 空间层 `prism_audio_spatial` 新增 **频变声源指向性模块 `source_directivity`**（对齐经典电声学 L. Beranek Acoustics / H. Olson Acoustical Engineering）：控制率描述声源辐射指向（不做逐样本 DSP），加权一阶模型 `d(band,cos)=((1-s_b)+s_b*cos).max(0)`（后半球截断）、`s_b` 为每倒频程锐度随频率递增；指向性因子 `Q_b=1/((1-s_b)^2+s_b^2/3)`（s=0->Q=1 DI=0dB、s=1->Q=3 DI≈4.77dB）、`DI_b=10*log10(Q_b)=10*ln(Q)/LN_10`；`DirectivityPreset{Omni,Cardioid,Voice,Trumpet}::sharpness()->[8]`（Voice/Trumpet 频谱单调非减、Trumpet 每带>=Voice）、`SourceDirectivity{from_sharpness([8]每带钳[0,1]、非有限->0), from_preset, sharpness(), gain_at(band,cos), band_gains(cos)->[8], broadband_gain(cos,freq)（log 频率插值 sharpness）, directivity_factor(band), directivity_index_db(band)}` serde 门控；复用 `material_library::{OCTAVE_BAND_CENTERS,OCTAVE_BAND_COUNT}`；11 golden 对拍（omni 全角度=1 / cardioid 后向与侧向（s=1 实为 d=cos 故侧向=0）零 / 各预设轴向=1 / 锐度越高侧向越低 / 预设随频单调 / Q 闭式 / DI 知名值 / broadband 带中心对齐 / broadband 带间插值 / 越界钳末带 / 非有限安全）；spatial 累计 **332 单测 + 29 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`source_directivity.rs` + spatial `lib.rs`，commit 60cbf3976）；恕守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或行生代码）
- 版本历史: v0.38（**本版新增**：M1 核心层 `prism_audio_core` 新增 **bit-crusher/decimator lo-fi 效果 `BitcrusherNode`**（对齐经典 lo-fi 降质拓扑 Zoelzer DAFX，落地 §255/§39/§40）：位深量化（`levels=2^bit_depth`、`step=2/levels`、`round(x/step)*step` 并把整数码钳到有符号补码范围 `[-2^(bit_depth-1), 2^(bit_depth-1)-1]` 保证恰好 `2^bit_depth` 个码、如实模拟真实转换器硬件、`bit_depth` 允许分数平滑变形） + 采样率降采样（零阶 sample-and-hold decimator：相位累加器每帧 `+1/downsample`、跨越 1 时捕获并量化一帧、其余帧保持上次捕获、`downsample>=1` 允许分数） + wet/dry `mix` 混合（`out=in*(1-mix)+held*mix`）；每通道 hold 状态在 `new` 分配、process 零分配/锁/panic、通道不匹配与零长块优雅退化、`reset` 清 held 并令首帧即捕获；`BitcrusherParams{bit_depth(默认8),downsample(默认1),mix(默认1)}` derive Default + 可选 serde，`BitcrusherNode{new(layout,params), set_bit_depth(钳[MIN_BIT_DEPTH=1,MAX_BIT_DEPTH=24]), set_downsample(钳>=1), set_mix(钳[0,1])}` impl AudioNode、公开常量 `MIN_BIT_DEPTH`/`MAX_BIT_DEPTH`；10 golden 对拍（64bit+ds1+wet1 透明 / mix=0 干声原样 / 2bit 塔到 <=4 码 / 3bit 值落网格 / ds=4 每 4 帧成组保持 / stereo 同步捕获 / 参数钳位后仍有限 / 多次设置输出有限 / reset 复现 / 零帧不 panic）；确定性数学走 `bevy_math::ops`（`powf`/`round`），core 累计 **214 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/bitcrusher.rs` + `nodes/effects/mod.rs` + `nodes/mod.rs`，commit a7736cb62）；同时 M4 空间层 `prism_audio_spatial` 新增 **房间模态混响模块 `room_modes`**（对齐经典波动声学 Rayleigh 驻波本征频率 / H. Kuttruff Room Acoustics 轴向/切向/斜向分类 / Maa-Kuttruff 模态密度，落地 §14/§39/§40）：矩形房间本征频率闭式 `f(nx,ny,nz)=(c/2)*sqrt((nx/Lx)^2+(ny/Ly)^2+(nz/Lz)^2)`（三元组非全零），按非零索引数分类 Axial/Tangential/Oblique 并赋经典 4:2:1 归一权重 1.0/0.5/0.25；每轴独立以纯轴向频率上界枚举（`axis_index_limit` 封顶 32、整数 while 递推规避 f32->int cast）、定长栈缓冲升序插入 + top-k 淘汰最高频保留最低 `MAX_ROOM_MODES=64`、退化维（非正尺寸）不出该轴模态、非有限 ceiling/零体积返回空集；`modal_density(f)=4*PI*V*f^2/c^3`（Maa/Kuttruff）、`schroeder_frequency` 委托 `room_acoustics`、`band_coloration` 把模态权重按几何 `*sqrt(2)` 上边界分入八倍频程 `OCTAVE_BAND_CENTERS` 守恒总权重；`ModeKind{Axial,Tangential,Oblique}::weight()`、`RoomMode{frequency_hz,kind,weight}`、`RoomModes{from_shoebox(&ShoeboxRoom,max_frequency_hz,sound_speed), modes/count/volume/fundamental_hz/schroeder_frequency/modal_density/band_coloration}` serde 门控（`RoomModes` 内含 [RoomMode;64] 超 serde 32 上限故非 serde），`DEFAULT_SOUND_SPEED` 因与 `early_reflections` 同名私有未 re-export；11 golden 对拍（立方体轴向 = 闭式解 / 升序 / 三类齐全且切向 sqrt(2)、斜向 sqrt(3) 倍 / 权重序 / 零维安全 / 模态密度随 f^2 翻四倍且大房更密 / top-k 保最低 / 基频最低 / Schroeder 活室>死室且退化安全 / 非有限输入安全 / band coloration 非负且守恒总权重）；确定性数学走 `bevy_math::ops`（`sqrt`），spatial 累计 **320 单测 + 28 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`room_modes.rs` + spatial `lib.rs`，commit 4ed68122a）；恺守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或行生代码）
- 版本历史: v0.37（**本版新增**：M4 空间层 `prism_audio_spatial` 新增**晚期扩散场模块 `diffusion_field`**（对齐经典统计声学 M. R. Schroeder 回声密度增长律 / H. Kuttruff Room Acoustics 扩散场能量平衡 / J.-D. Polack 与 Jot-Gardner FDN 混响文献混合时间估计，落地 §14/§39/§40）：`scattering` 把每次反射拆出的 diffuse 能量汇聚成随时间/空间平滑的晚期扩散混响能量场描述（纯控制率、不做逐样本 DSP，实际延迟线滤波在 core reverb）；对入射带能量 `E` 吸声 `alpha` 散射系数 `s`，反射能量 `E*(1-alpha)`、其 diffuse 份额 `E*(1-alpha)*s`，逐面 `add_surface` 累加成每倍频程 diffuse 能量密度（与 `OCTAVE_BAND_CENTERS` 对齐）；镜像源计数 `N(t)=(4/3)*PI*c^3*t^3/V` 求导得 Schroeder 回声密度 `dN/dt=4*PI*c^3*t^2/V`（随 t^2 上升、大房更低、钳 `MAX_ECHO_DENSITY`），混合时间取 Polack/Jot `t_mix~sqrt(V)` 毫秒，diffusion coefficient=diffuse/reflected 每带比均值 [0,1]（除零守卫）；`late_send_gains` 峰值归一每带晚期湿声发送、`fdn_late_gains(&OctaveReverb,delay)` = 每带 decay 增益 x send 归一（钳 [0,0.999999)）驱动频变 FDN；控制率零分配/锁/panic、退化输入（零体积/空面集/非有限时间）返回安全有限值不出 NaN/Inf、超越函数全走 `bevy_math::ops`（`sqrt`）无 f32 内建无 f32->int cast；`DiffusionField{new(&ShoeboxRoom), add_surface(&mut,incident&[8],&ScatteringSpectrum,&MaterialAbsorption), from_uniform_shoebox(...), band_energy/broadband_energy/volume/surface_area/diffusion_coefficient/mixing_time_ms/echo_density(t,c)/late_send_gains/fdn_late_gains}` serde 门控私有字段，公开常量 `MAX_ECHO_DENSITY` re-export（`DEFAULT_SOUND_SPEED` 因与 `early_reflections` 同名未 re-export、仍可经 `diffusion_field::` 访问避免 E0252）；11 golden 对拍（大房混合时间更长且 sqrt(V) 精确 / 回声密度随时间上升且 t^2 律翻四倍并钳顶 / 大房回声密度更低 / 退化时间与零体积安全 / 高吸声降 diffuse 能量（混凝土>地毯）/ 高散射升 diffuse 能量（diffuser>flat）/ 带能量非负 / 累加+坏输入钳零 / diffusion 系数 [0,1] 且随散射上升+空场零 / late send 峰值归一+空场零 / fdn 晚期增益结合 decay 与 send+零延迟塌缩）；确定性数学走 `bevy_math::ops`，spatial 累计 **309 单测 + 27 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`diffusion_field.rs` + spatial `lib.rs`，commit 31a993cbd）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.36（**本版新增**：M1 核心层 `prism_audio_core` 新增**颤音/自动声像节点 `TremoloNode`**（对齐经典调制效果拓扑 Zoelzer DAFX / Reiss and McPherson Audio Effects，落地 §254/§336/§39/§40）：单个双极控制率 `Lfo` 每样本取值 `m` 属于 [-1,1]，`TremoloMode::Amplitude`（默认）映射标量增益 `gain=1-depth*(0.5-0.5*m)`（在波谷 `1-depth`、波峰 `1` 之间摆动）应用到每通道，可选 `stereo_phase`（周期分数 [0,1)）让通道 c 起始相位 `c*stereo_phase` 交错产生旋转立体声闪烁（0.5 使立体声对反相去相关）；`TremoloMode::AutoPan` 把 `m` 当声像位置喂 `equal_power_pan` 出等功率左右增益对、把声像在立体声场左右移动而不改感知响度、只作用前两通道（其余透传、非立体声回退 Amplitude）；每通道一个 LFO 在 new 分配、process 零分配/锁/panic、通道不匹配与零长块优雅退化、reset 复原各通道初相；`TremoloParams{rate_hz(默认5),depth(默认0.5钳[0,1]),waveform,mode,stereo_phase}` derive Default + 可选 serde，`TremoloMode{Amplitude(默认),AutoPan}` derive + serde，`TremoloNode{new(sample_rate,layout,params), set_depth/set_rate/set_waveform/set_mode}` impl AudioNode；9 golden 对拍（depth=0 bypass / amplitude 输出落在 [1-depth,1] 包络且触及两端 / 深 depth 摆幅更大 / auto_pan 保功率 l^2+r^2=1 且左右移动镜像 / mono auto_pan 回退 amplitude / stereo_phase=0.5 通道去相关 / reset 复原初相 / 四波形输出有限 / 零帧不 panic）；确定性数学走 `bevy_math::ops`（`floor`），core 累计 **204 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/effects/tremolo.rs` + `nodes/effects/mod.rs` + `nodes/mod.rs`，commit 9617a8d69）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.35（**本版新增两项**：（1）M1 核心层 `prism_audio_core` 新增**分频去齿音器 `DeEsserNode`**（对齐经典录音室分频动态拓扑 Reiss and McPherson Audio Effects / Zoelzer DAFX，落地 §254/§336/§39/§40）：二路 Linkwitz-Riley 分频在 `crossover_hz`（默认 6000）拆低/高（齿音）带且幅频重组平坦，stereo-linked 每帧取最响高带样本整流做 side-chain 喂 `LevelDetector.level_db` -> 标准软拐点 `compressor_reduction_db` -> 钳 `max_reduction_db` -> `GainBallistics` attack/release 平滑 -> `gain=db_to_linear(-reduction)`，`DeEsserMode::SplitBand`（只缩高带再与低带重组、保留人声本体，默认最透明）/ `Wideband`（缩全信号）；不同于宽带压缩器整轨抽吸，去齿音器只在高频齿音超阈时短促下压 5-9 kHz；`DeEsserParams{crossover_hz,threshold_db,ratio,knee_db,attack_ms,release_ms,max_reduction_db,detection,rms_window_ms,mode}` derive Default（典型人声）+ 可选 serde，`DeEsserNode{new(sample_rate,layout,params,max_frames), set_threshold_db/set_ratio/set_max_reduction_db/set_mode, reduction_db}` impl AudioNode（state 全在 new 分配、process 零分配/锁/panic、通道不匹配与零长块优雅退化、reset 清 crossover/detector/ballistics/last_reduction + bands）；9 golden 对拍（低频 200 Hz 透过不衰减 / 响亮 8 kHz 齿音被缩 / 阈下安静齿音不动 / Wideband 缩全信号 / reduction 钳到 max_reduction_db / 输出有限+reset 清态 / 静音安全 / 零帧不 panic / stereo linked 保持左右比例）；确定性数学走 `bevy_math::ops`，core 累计 **195 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/dynamics/de_esser.rs` + `nodes/dynamics/mod.rs` + `nodes/mod.rs`，commit 3f82838a5）。（2）M4 空间层 `prism_audio_spatial` 新增**表面散射/扩散模块 `scattering`**（对齐 ISO 17497-1 散射系数 / Kuttruff Room Acoustics Lambert 余弦律 / Cox and DAntonio Acoustic Absorbers and Diffusers，落地 §14/§39/§40）：真实表面非完美镜面，粗糙起伏把部分反射能量散离镜像方向；`ScatteringSpectrum` 每倍频程散射系数谱（与 `OCTAVE_BAND_CENTERS` 对齐、复用 material_library 的 log 频率插值端点钳位不外推、随频率上升），`SurfaceScatter{Flat,PaintedBrick,RoughBrick,Bookshelf,Diffuser,Curtain}::scattering()` 六材质取 ISO 17497-1 典型测量范围，`specular_fraction(s)=1-s`/`diffuse_fraction(s)=s` 能量守恒拆分（specular 保镜像方向喂 `early_reflections`、diffuse 喂晚期扩散场），`lambert_weight(cos_theta)=cos/PI`（前向半球积分为 1 的能量守恒每球面度权重，背向/掠射/NaN 退化 0）/ `lambert_directivity(normal,dir)`（内部归一化、零向量退化 0）；控制率纯标量、零分配/锁/panic、退化输入钳有限值不出 NaN/Inf，12 golden 对拍（全材质带内 [0,1] / 随频率上升 / 粗糙>光滑（diffuser>bookshelf>flat）/ new 钳越界 / 带心返回该带 / 端点钳位+退化频率不 panic / 倍频程对数插值 / broadband 均值 / specular+diffuse 恒等 1（含越界钳位）/ Lambert 法向峰值掠射归零 / 用法向夹角 / 零向量安全）；确定性数学走 `bevy_math::ops`（`ln`/`sqrt`），spatial 累计 **298 单测 + 26 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`scattering.rs` + spatial `lib.rs`，commit e681495e0）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio 源码或衍生代码）
- 版本历史: v0.34（**本版新增两项**：（1）M1 核心层 `prism_audio_core` 新增**差分包络瞬态整形器 `TransientShaperNode`**（对齐经典录音室瞬态设计拓扑 fast/slow 双包络跟随器差分驱动增益，Zoelzer DAFX，落地 §254/§336/§39/§40）：两个 AR 单极包络跟随器（fast 抢先 onset、slow 滞后）复用 `detector::time_to_coef` 弹道，stereo-linked 每帧取最响通道整流电平做 side-chain，`ratio=(fast-slow)/max(fast,slow)` 属于 (-1,1)，`ratio>=0`（onset）应用 `gain=1+attack*ratio`、`ratio<0`（decay）应用 `gain=1+sustain*(-ratio)`，用峰值包络归一化使增益天然有界、电平无关、无 zipper 噪声，gain 钳位 `[db_to_linear(-max_db), db_to_linear(max_db)]`，输出 `dry*x+wet*(gain*x)` 并联湿干；不同于压缩器依赖绝对阈值，瞬态整形器响应包络移动速度、无阈值随素材跟随任意电平（更紧的鼓/更长的房间尾/更干净的拨弦）；`TransientShaperParams{attack,sustain,fast_attack_ms,fast_release_ms,slow_attack_ms,slow_release_ms,max_gain_db,wet,dry}` derive Default（中性透明）+ 可选 serde，`TransientShaperNode{new, set_attack/set_sustain（钳 [-1,1]）, set_wet/set_dry(v,Ramp), gain_db}` impl AudioNode（state 全在 new 分配、process 零分配/锁/panic、通道不匹配与零长块优雅退化、latency 0、reset 清跟随器与 last_gain）；11 golden 对拍（中性近恒等 / +attack 抬 onset 峰 / -attack 削峰 / +sustain 增尾能量 / -sustain 减尾能量 / gain 钳到 max_gain_db 范围 / 输出有限+reset 清态 / 静音安全恒等 / 零帧不 panic / stereo linked 保持左右比例）；确定性数学走 `bevy_math::ops`，core 累计 **186 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（`nodes/dynamics/transient_shaper.rs` + `nodes/dynamics/mod.rs` + `nodes/mod.rs`，commit 4195330cd）。（2）M4 空间层 `prism_audio_spatial` 新增**倍频程混响时间模块 `OctaveReverb`**（对齐经典统计声学 per-band RT60 频谱，Eyring 1930 / Kuttruff Room Acoustics / Beranek Acoustics，落地 §14/§39/§40）：真实房间无单一混响时间——表面吸声随频率上升故高频尾衰减更快（"暖"晚期声场），本模块保留频率依赖，对每个倍频程用 `material_library` 每面每带吸声 + `early_reflections::ShoeboxRoom` 几何算 Eyring RT60（`RT60(band)=0.161*V/(-S*ln(1-alpha_bar(band)))`，高吸声下比 Sabine 更物理，复用 `room_acoustics::eyring_rt60`），输出与 `OCTAVE_BAND_CENTERS` 对齐的八带 RT60 谱，可任意频率查询（log 频率插值端点钳位不外推、NaN/非正回退最低带）并转 per-band FDN 反馈增益 `g=10^(-3*delay/RT60)`（60 dB 衰减 = 1e-3 因子，Schroeder/Jot 时间常数关系），`OctaveReverb{from_shoebox_material(&ShoeboxRoom,&[MaterialAbsorption;6]), uniform(&ShoeboxRoom,Material), from_bands([Sample;8])（钳非负）, rt60_bands, rt60_at(freq), broadband_rt60（500+1000 Hz 均值 T_mid）, fdn_decay_gains(delay)->[Sample;8]（钳 [0,0.999999]、delay/rt60<=0 回退 0、无 NaN/Inf）}`；控制率纯标量、零分配/锁/panic、退化几何（零体积/满吸声/非正 RT60/非正 delay）返回有限值，13 golden 对拍（硬房 RT60>软房各带 / 地毯高频衰减快于低频 / 全材质有限非负 / 满吸声近零 / 零体积安全 / 带心返回该带 / 端点外钳位+退化频率不 panic / 倍频程几何中点对数插值 / from_bands 钳负 / FDN 增益随 RT60 单调 / 单 RT60 延迟 -60 dB / FDN 增益优雅退化 / broadband 中带均值），spatial 累计 **286 单测 + 25 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `octave_reverb.rs` + spatial `lib.rs`，commit a1cc4d98d）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.33（**本版新增两项**：（1）M1 核心层 `prism_audio_core` 新增**Linkwitz-Riley 多频带压缩器 `MultibandCompressorNode`**（对齐经典录音室/母带多频带动态处理拓扑 band-split -> 每带独立压缩 -> 求和重建，Zoelzer DAFX，落地 §254/§336/§39/§40）：复用 `crossover` 的 LR4 全通相位补偿分频把频谱切成 M+1 带（幅频平坦），每带一个完整 `CompressorNode`（软拐点/峰值+RMS 检测/前瞻/补偿增益/并联湿干）独立动态，再逐样本求和重建；使低频事件（如底鼓）不再泵动无关高频（如镲片），是母带级 glue/de-ess/响度控制优于全频段单元的核心；`MultibandCompressorNode{new(sample_rate,layout,crossover_freqs,band_params,max_frames), num_bands, band_gain_reduction_db(band)}`，state 全在 new 分配（crossover + 每带 CompressorNode + 分频 scratch + 每带输出 scratch）、process 零分配/锁/panic（通道不匹配与零长块优雅退化）、latency_frames=各带前瞻最大值；9 golden 对拍：单带近恒等 / 三带透明重建幅频平坦 / 低带压低频 / 高频穿过低带压缩不受影响 / reset 清尾 / 输出有限 / 零帧不 panic / 带数=交叉数+1 / 越界 band reduction=0；确定性数学走 `bevy_math::ops`，core 累计 **176 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过（注册 `nodes/dynamics/multiband.rs` + `nodes/dynamics/mod.rs` + `nodes/mod.rs`，随并发提交 c22dcfc6a 落库）。（2）M4 空间层 `prism_audio_spatial` 新增**建筑声学吸声材料库 `material_library`**（对齐经典建筑声学公开吸声数据 Kuttruff Room Acoustics / Beranek Acoustics / Vorlander Auralization，落地 §14/§39/§40）：为 `early_reflections`/`room_acoustics` 提供权威具名材料吸声、替代手调魔数；`OCTAVE_BAND_CENTERS=[63,125,250,500,1000,2000,4000,8000]` Hz 八倍频程中心，`MaterialAbsorption{new（钳 [0,1]）, bands, at(freq)（log 频率线性插值 + 端点钳位不外推 + 非有限/非正回退最低带 + 钳 [0,1]）, broadband_mean（八带算术均值）, uniform_walls/uniform_walls_at}` + `enum Material{Concrete,PaintedConcrete,Brick,Plaster,Wood,Glass,Carpet,HeavyCurtain,AcousticTile,Water}` + `Material::{absorption,broadband_mean,uniform_walls}`；吸声表为材料类的典型教科书值（非厂商数据表）：硬密表面（混凝土/玻璃/水）全带低吸声、多孔/垂帘（地毯/厚幕/吸声吊顶）高频强吸声；控制率纯标量、零分配/锁/panic、确定性数学走 `bevy_math::ops`、no_std，13 golden 对拍（每带在 [0,1] / 带心返回该带 / 倍频程几何中点近对数线性中值 / 端点外钳位 / 吸声体>反射体宽带均值 / 玻璃与水低频吸声小 / 宽带均值等于手工平均 / 构造钳越界 / 退化频率安全 / 硬房 RT60>软房 等），spatial 累计 **273 单测 + 24 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `material_library.rs` + spatial `lib.rs`，commit `92d8b4f9c`）；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.32（**本版新增两项**：（1）M1 核心层 `prism_audio_core` 新增**Linkwitz-Riley 多频带分频器 `crossover`**（对齐 Linkwitz 1976 有源分频网络理论 + RBJ cookbook，落地 §254/§336/§39/§40）：M 个交叉频产出 M+1 频带，每交叉点四阶 Linkwitz-Riley（LR4，24 dB/oct）= 两级级联 Butterworth 二阶（Q=1/sqrt(2)）低通 + 两级高通，**复用现有 `nodes::biquad`** 递推；串行拓扑 carry 逐级高通、**全通相位补偿**（band[k] 过所有更高交叉点的二阶全通等效）使各频带和恒为全通级联->幅频平坦（LP4+HP4=二阶全通、同相、交叉点各 -6dB 相加 0dB）；频率入构造即钳 [1, sr*0.499] + 升序排序 + take(MAX_CROSSOVERS=7)，常量 MAX_BANDS=8；`LinkwitzRileyCrossover{new(sample_rate,layout,crossover_freqs,max_frames), num_bands, channels, process_block(&input,&mut[AudioBuffer bands]), reset}`，state 全在 new 分配、`process_block` 零分配/锁/panic（自写 `copy_active` 避 layout mismatch panic）；11 golden 对拍：单频带恒等 / 2-way 与 3-way 幅频平坦（sum RMS≈in RMS）/ 低带拒高频 / 高带拒低频 / 交叉点两带各 -6dB / 3-way 中带隔离 / DC 落低带 / 输出有限+reset 清尾 / 退化不 panic / 交叉频数钳 MAX_BANDS；确定性数学走 `bevy_math::ops`，core 累计 **167 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `nodes/crossover.rs` + `nodes/mod.rs`，commit `48ad5ccdd`）。（2）M4 空间层 `prism_audio_spatial` 新增**房间统计混响声学估计 `room_acoustics`**（对齐 Sabine 1900 / Eyring 1930 / Millington-Sette 1932 RT60 + Schroeder 1962 频率 + Kosten 平均自由程 + 临界距离，落地 §14/§39/§40）：从 `ShoeboxRoom`（六面独立吸声）或 `Room`（单一墙材、alpha=1-reflection）求体积/表面积/总吸声（metric sabins），统计估计房间晚期扩散场以驱动上游 FDN，补 `early_reflections`（早期离散镜像）->统计晚期混响空档；自由函数 `mean_free_path(V,S)=4V/S`/`sabine_rt60(V,A)=k*V/A`（k=0.161）/`eyring_rt60(V,S,alpha_bar)=k*V/(-S*ln(1-alpha_bar))`/`millington_sette_rt60(V,faces)`（逐面对数）/`schroeder_frequency(rt60,V)=2000*sqrt(rt60/V)`/`critical_distance(V,rt60)=0.057*sqrt(V/rt60)` + `RoomAcoustics{from_aggregates,from_shoebox,from_room, volume/surface_area/total_absorption/mean_absorption, mean_free_path/rt60_sabine/rt60_eyring/schroeder_freq/critical_distance}`；退化守卫 MIN_DIVISOR=1e-9 地板 + MAX_MEAN_ABSORPTION=0.9999 钳位保满吸声 RT60->近零不 Inf/NaN，Eyring<=Sabine（均匀房 -ln(1-x)>=x）、Millington 对均匀每面退化为 Eyring；13 golden 对拍（Sabine 解析立方 / Eyring<=Sabine / 满吸声 RT60 有限 / Schroeder 随 RT60 升 / 平均自由程精确等）；控制率纯标量、零分配/锁/panic、确定性数学走 `bevy_math::ops`，spatial 累计 **260 单测 + 23 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `room_acoustics.rs` + spatial `lib.rs`，commit `5cb0a4d56`）；刷新 §14/§254/§336/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.31（**本版新增两项**：（1）M1 核心层 `prism_audio_core` 新增**立体声展宽器 `stereo_width`**（对齐 Blumlein 1931 和差立体声原理，落地 §254/§336/§39/§40）：Mid-Side 展宽（M=(L+R)/2、S=(L-R)/2，L=M+width*S / R=M-width*S），`width` 钳 [0, MAX_WIDTH=4.0]（0=mono/1=原样/>1=展宽）经 `Smoothed` 逐样本平滑；可选 bass-mono 交叉分频（`bass_mono_hz>0` 启用）复用 `BiquadCoeffs::design(LowPass, Q=FRAC_1_SQRT_2)` 取 side 低频带、内联 DF1（自持 [Sample;4] 状态跨块保留）从 side 相减->交叉频以下折回中置，不足双声道直通、超立体声对通道原样拷贝；`StereoWidthParams{width, bass_mono_hz}`/`StereoWidthNode{new, set_width(w,Ramp), width, set_bass_mono_hz, bass_mono_enabled, process, reset}`；golden 对拍：width=1 恒等 / width=0 折叠 mono / width=2 侧能量>3x / width 钳 MAX / mono 直通 / bass-mono 移除低频侧(40Hz 尾 RMS<0.2) / 保留高频侧(6kHz RMS>0.6) / 超额通道直通 / 输出有限+reset 清尾；确定性数学走 `bevy_math::ops`、热路径零分配/锁/panic，core 累计 **156 单测 + 5 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `nodes/effects/stereo_width.rs` + `nodes/effects/mod.rs`，commit `cfcd4fa13`）。（2）M4 空间层 `prism_audio_spatial` 新增**多跳门户路由 `portal_graph`**（对齐 Fresnel-Kirchhoff 孔径耦合 + 经典图搜索，落地 §14/§470/§39/§40）：房间=节点、门户=边的**深度有界无环 DFS**（深度<=min(max_hops, MAX_PORTAL_HOPS=4)），`visited` 定长栈数组去环（源房间预置）、门户按索引升序枚举保确定性；同房间->单 0-hop 直达；几何链 源->各门户 `closest_point` 孔径->听者累积距离，增益=各 `portal_coupling_gain` 积 x 1/max(dist,0.1)，`seconds_to_samples` 整数步进避 f32->int cast，溢出流式 top-k 保最强、退化输入安全返回；`route_portals(rooms, portals, listener, emitter, max_hops, sample_rate, sound_speed, out)->usize`/`RoutedPath{hop_count, hops:[PortalHop;4], total_gain, total_distance, delay_samples}`/`PortalHop{portal_index, aperture}`，常量 MAX_ROOMS=64/MAX_PORTALS=128/MAX_ROUTED_PATHS=32；确定性数学走 `bevy_math::ops`、热路径零分配/锁/panic（栈数组 + 递归深度<=4），spatial 累计 **247 单测 + 22 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `portal_graph.rs` + spatial `lib.rs`，commit `de1ccf6c9`）；刷新 §14/§254/§336/§470/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.30（**本版新增**：M4 空间层 `prism_audio_spatial` 新增**镜像源法早期反射 `early_reflections`**（对齐 Allen & Berkley 1979 JASA 矩形房间镜像法，落地 §14 反射 + §39/§40）：矩形（shoebox/AABB）房间早期反射用整数索引 `(nx,ny,nz)` 枚举镜像源（折叠闭式 n 偶 `n*W+sr`、n 奇 `(n+1)*W-sr`，非递归几何、位可复现），每镜像→听者局部到达方向（复用 `listener.orientation.inverse()`，与 `geometry::localize` 一致）/欧氏距离/延迟（整数步进累加避 f32->int cast、钳 `MAX_DELAY_SAMPLES=1<<18`）/增益（`1/max(distance,0.1)` × 各面幅度反射系数 `beta=sqrt(1-alpha)` 之积，穿越计数 `div_ceil`/整除），总阶 `|nx|+|ny|+|nz|` 钳 `MAX_REFLECTION_ORDER=4`、退化轴（span<=1e-6）仅出 n=0、`is_direct` 标零阶直达；`compute_early_reflections(room,listener,source,order,sample_rate,sound_speed,out)->usize` 栈上零分配、溢出保留最强（流式 top-k，上限 `MAX_EARLY_REFLECTIONS=32`）；额外实现 RT 渲染器 `EarlyReflectionRenderer{new,set_taps,process_block,reset}`——单共享延迟线 + 多抽头读 + `equal_power_pan` L/R，`new` 一次性分配、`process_block` 零分配/锁/panic；`ShoeboxRoom{new/rigid/size, wall_absorption:[Sample;6]}`/`ReflectionTap{delay_samples,gain,direction,order,is_direct}`；golden 对拍：立方体中心一阶六镜像共延迟共增益、直达延迟=distance/speed、增益随 1/r、延迟随房间增大、全反射=1/distance、满吸收面静默、阶数计数、90degree 偏航正前落本地 +X、退化不 panic、溢出保最强、seconds_to_samples 就近取整、渲染器延迟/声像/reset 清尾；确定性数学走 `bevy_math::ops`、热路径零分配/锁/panic，spatial 累计 **235 单测 + 21 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `early_reflections.rs` + spatial `lib.rs`，commit `523158e6d`）；刷新 §14/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.29（**本版新增**：M4 空间层 `prism_audio_spatial` 新增**HOA 近场补偿滤波 `nfc`**（对齐 Daniel 2001 / Daniel 2004 稳定化 NFC，落地 §16 近场补偿 + §46.5）：点声源有限距离辐射的球面波在高阶球谐分量上带来低频强调（近场效应），远场编码器缺失的逐阶传函 `F_m(kr)` 在 DC 发散不可实现，故用**参考距离稳定化**——`H_m(s)=theta_m(s*r_src/c)/theta_m(s*r_ref/c)`（theta_m=m 阶反向 Bessel 多项式），分子分母同为首一多项式，高频增益抵消为 1、DC 增益有限 `(r_ref/r_src)^m`；**反向 Bessel 根硬编码闭式**（theta_1: x=-1；theta_2: -1.5+-j*sqrt(3)/2；theta_3: -2.3221854 与 -1.8389073+-j*1.7543810，非数值求根、位可复现），每阶经**双线性变换**映射为一阶节 + biquad 级联（order 1 仅一阶节、order 2 仅 biquad、order 3 两者），控制率设计在 f64 内做避 `k=2fs` 大项对小极零点的灾难抵消、系数存 f32；`NfcCoeffs::design(order, source_distance, reference_distance, sample_rate, sound_speed)` + `NfcFilter{new, set_coeffs, process_channel, process_block}`（每 ACN 通道独立 IIR 状态、同阶 (2m+1) 通道共系数各自独立、退化输入钳位不 panic）；golden 对拍：全极点模<1（稳定）、0 阶恒等、r_src==r_ref 冲激=delta、DC 增益=`(r_ref/r_src)^m`、近源低频抬升/远源衰减随阶增长、稳态匹配解析 DC 增益；确定性数学走 `bevy_math::ops`、热路径零分配/锁/panic，spatial 累计 **222 单测 + 20 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `nfc.rs` + spatial `lib.rs`，commit `0b608eac6`）；刷新 §16/§39/§40/§46.5；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio/Google Resonance Audio 源码或衍生代码）
- 版本历史: v0.28（**本版新增**：M4 空间层 `prism_audio_hrtf` 新增**跨听/串音消除渲染 `transaural`**（对齐 Bauer 1961 / Schroeder-Atal 1963 / Cooper-Bauck 1989 / Gardner 1998，落地 §16 跨听渲染 + §49 术语）：扬声器回放双耳信号时消除对侧扬声器到耳的声学串音，`CrosstalkCanceller::process_block(in_l,in_r,out_l,out_r)` 递归消除器 `s_l[n]=b_l[n]-beta*s_r[n]; s_r[n]=b_r[n]-beta*s_l[n]` 精确逆对称串音矩阵 `C=[[1,beta],[beta,1]]`（beta=对侧/同侧比：延迟 d>=1 断代数环 + 头影增益 g<1 + 一阶低通头影，环路增益 g^2<1 无条件稳定），`CrosstalkParams::from_geometry` 由对称扬声器半角 Woodworth ITD `(a/c)(phi+sin phi)` + 头影截止 `c/(2 pi a)` 推导路径；确定性数学走 `bevy_math::ops`、控制率构造、热路径零分配/锁/panic，gain 钳 [0,0.999]/delay 钳 >=1 防越界，hrtf 累计 **81 单测 + 4 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，精确 commit（仅 `transaural.rs` + hrtf `lib.rs`，commit `4c21001e4`）；刷新 §16/§39/§40/§49；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio 源码或衍生代码）
- 版本历史: v0.27（**本版新增**：M4 空间层完成 **HOA 三件套一等公民**——`prism_audio_spatial` 新增**双频段能量优化解码 `hoa_decode`**（对齐 Daniel 2001 / Google Resonance Audio 双频段解码，落地 §16 双频段解码 + §46.4）：`DualBandDecoder{new(order), decode_low/decode_high/decode(band,...), decode_to_speakers}` 把 HOA 场解到任意 `SpeakerLayout`（栈上 32 扬声器），低频段 basic/in-phase 保 ITD/相位、高频段 **max-rE** 保能量矢量；`max_re_gains(order)`=经典 `g_n=P_n(r_E)` 归一 g_0=1、`max_re_radius(order)=cos(137.9度/(order+1.51))`；**关键数值修正**：`hoa.rs` 用 SN3D-scaled 投影解码（每阶自能量=1、加法定理 `Sigma_m Y*Y=P_n`，缺 (2n+1) 模态计数），故高频段每通道增益改为 **`(2n+1)*g_n`**（`high_norm=Sigma(2n+1)*g_n`）修正能量向量，修正后 sphere/ring 各阶 high>low（order1: 0.577>0.5 命中经典 r_E），`max_re_gains` 公开 API 仍返回经典 `g_n`、(2n+1) 内含于解码器，分频交叉（Linkwitz-Riley）明确留调用方（11 单测 + 1 doctest，commit `dd3e00e73`）。**`prism_audio_hrtf` 新增虚拟扬声器双耳 `hoa_binaural`**（落地 §16 虚拟扬声器双耳 + §46.4）：`HoaBinauralDecoder{new(dataset,order,layout,max_block), process_block(hoa, out_l, out_r)}` 每 ACN 通道预烘焙一对滤波器 `filter[c]=Sigma_s D[s][c]*HRIR(s)`（`D[s][c]=encode_hoa(dir_s)[c]/(order+1)` 等价 decode_hoa），运行时每通道一个 `BinauralRenderer` 吃 `hoa[c]` 累加 L/R，**数学等价全虚拟扬声器解码但更省**、零分配；`VirtualSpeakerLayout::cube26`（6 面+12 边+8 角 26 点，满足 order 3）、`MAX_VIRTUAL_SPEAKERS=32`（serde 数组上限），golden `impulse_matches_virtual_speaker_sum` 对拍显式虚拟扬声器和（8 单测 + 1 doctest，commit `0db3ff069`）。**`prism_audio_spatial` 新增可转向虚拟传声器/模态波束成形 `hoa_beamform`**（对齐 Zotter-Frank 2019 / Daniel 2001，落地 §16 + §46.4）：`Beamformer::beam(coeffs, look_dir) -> Sample` 把 HOA 场收成单路可转向 mono 虚拟传声器，波束 `B(gamma)=(Sigma_n g_n*P_n(cos gamma))/(Sigma_n g_n)`、归一 `Sigma_n g_n` 使 look 轴响应恰=1；四族模态波束 `BeamPattern{Basic(g_n=1), MaxDi(g_n=2n+1，DI 精确=(order+1)^2), MaxRe(g_n=P_n(r_E) 复用 max_re_gains), InPhase(g_n=(L!)^2/((L+n)!(L-n)!) 无负旁瓣)}`，`beam_gains(pattern,order)` 控制率预烘焙每通道增益、热路径三切片点积 + 除零守卫，InPhase 0-180 度采样 minB>=0、MaxDi 必有负旁瓣、Basic 与 decode_hoa 逐样本对拍相等（14 单测 + 1 doctest，commit `b4fae9761`）；三模块均确定性数学走 `bevy_math::ops`、控制率预烘焙、热路径零分配/锁/panic，spatial 累计 **211 单测 + 19 doctest 全绿**、hrtf **68 单测 + 3 doctest 全绿**，clippy 全特性/无默认特性零告警、no_std 双构建通过，各自精确 commit（仅模块文件 + 各自 crate lib.rs）；刷新 §16/§39/§40/§46.4；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio 源码或衍生代码）
- 版本历史: v0.26（**本版新增**：M4 空间层 `prism_audio_spatial` 新增**高阶 Ambisonics 声场旋转模块 `hoa_rotation`**（对齐 Google Resonance Audio / Steam Audio 声场旋转与头追贯通，落地 §16 声场旋转 + §46 工程内核）——实球谐（SN3D/ACN，与 `hoa` 同约定）在任意 `Quat` 下的**分块对角旋转**，支持到 `MAX_HOA_ORDER=3`（16 通道，阶 l 占 (2l+1)×(2l+1) 块）；一阶块 `R^1 = A·Q·A`（`A=diag(-1,+1,-1)`、`Q=Mat3::from_quat`，由 ACN 1/2/3=left/up/front=-x/+y/-z 轴约定推导，与 `hoa.rs` 完全一致）、二/三阶经 **Ivanic–Ruedenberg 递推**（1996 论文 + 1998 勘误的 U/V/W + P 系数递推）逐阶从前一块构造，越界 prev-block 访问返回 0（恰对应系数为 0 项、数学正确且零 panic）；公开 API `rotate_hoa(coeffs, order, rotation)`（原地、栈上临时矩阵零堆分配）与 `HoaRotationMatrix{from_quat, apply, order, channels}`（控制率算一次矩阵、多帧 `apply` 复用，非有限/近零四元数回退恒等无 NaN）；确定性数学仅走 `bevy_math::ops`（`ops::sqrt`）、热路径 `apply` 零分配/锁/panic（无 unwrap/expect/panic，仅 `unwrap_or`）、两处 `#[expect(needless_range_loop, reason=...)]` 收敛且无 pedantic cast 告警（`usize::try_from(..).unwrap_or(0)` + match 访问器）；**核心 golden 对拍**：`rotate_hoa(encode(d)) == encode(q·d)`（覆盖 order 1–3、绕 X/Y/Z 各轴 90°/180°/任意角、复合旋转）、0 阶 W 不变、逐阶能量守恒、order 钳位、短缓冲不 panic、零四元数恒等；新增 10 单测 + 1 doctest（crate 累计 **186 单测 + 17 doctest 全绿**）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit（`8b75d7ae7`，仅 `hoa_rotation.rs` + spatial `lib.rs` 两文件）；刷新 §16/§39/§40/§46；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio 源码或衍生代码）
- 版本历史: v0.25（**本版新增**：M4 空间层 `prism_audio_hrtf` 新增**头部追踪双耳模块 `headtracked`**（对齐 Meta XR Audio / Steam Audio 头追与 Resonance Audio 姿态旋转，落地 §16 头部追踪双耳）——`HeadPose{orientation:Quat, angular_velocity:Vec3}` + `predict(seconds)` 以四元数**轴角指数映射**做一阶姿态外推（`delta=exp(0.5·omega·dt)` 经 `ops::sin_cos`、`predicted=delta*orientation`、seconds 钳 `[0, MAX_PREDICTION_SECONDS=0.1]` 抑制过冲）；`HeadTracker{smoothed, prediction_seconds, smoothing_time_constant}` 的 `update(pose, dt)` 先外推再 slerp 平滑（`alpha=1-exp(-dt/tau)` 一阶时间常数）避免抖动；`world_to_local_direction=orientation.inverse()*单位化(dir)`（退化回退 -Z 前向）、`local_azimuth=atan2(x,-z)`/`local_elevation=atan2(y,hypot(x,z))`、一步式 `predicted_local_angles` 直接喂既有 `interpolation::interpolate`（复用 binaural/interpolation 的 ITD 群延迟对齐 + Shepard 反距加权，**不新造卷积**、无重复实现）；低延迟短前瞻使头动到声像更新走旋转/重选 HRIR 方位路径、避免声像黏头；确定性数学全走 `bevy_math::ops`、热路径无 unwrap/expect/panic（`expect` 仅在 `#[cfg(test)]`）、Provenance 齐；新增 13 单测 + 1 doctest（crate 累计 **60 单测 + 2 doctest 全绿**）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit（`7a7d2d477`，仅 `headtracked.rs` + hrtf `lib.rs` 两文件）；刷新 §16/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio 源码或衍生代码）
- 版本历史: v0.24（**本版新增**：M4 空间层新增**高阶 Ambisonics(HOA) 编解码模块 `hoa`**（对齐 UE MetaSounds Ambisonics / Unity Ambisonic Decoder SDK / Steam Audio / Google Resonance Audio，落地 §16 HOA 与 §44 对象/HOA 传输）——纯经典实球谐 DSP，`MAX_HOA_ORDER=3`/`MAX_HOA_CHANNELS=16`、`hoa_channel_count(order)=(order+1)^2`、AmbiX 通道序 `acn_index(n,m)=n*n+n+m`、SN3D 归一化（无 Condon-Shortley）；核心 `fill_hoa_coeffs` 走**纯 Cartesian 递推**（声学轴 front=-z/left=-x/up=y、方位三角用 Chebyshev 递推、关联勒让德按对角/次对角/升阶三递推构造、`sn3d_norm` 用阶乘比），**零向量退化为仅 W 全向**（避免塌到 front 轴产生伪高阶项）；`encode_hoa(dir, order, &mut [Sample]) -> usize` 栈缓冲编码后截断拷贝、`decode_hoa(coeffs, speaker_dir, order)` 把扬声器方向编码后与系数点积并**除以 `(order+1)`**（SN3D 每阶自能量=1、加法定理在源方向求和=N+1，与 FOA `DECODE_WEIGHT=0.5=1/(N+1)` 一致，源方向解回 1.0）；RT 编码节点 `HoaEncoderNode`（镜像 `FoaEncoderNode`：`gains:[Smoothed; 16]` 逐样本平滑、`set_direction(dir, ramp)`/`set_direction_immediate`/`order`/`active_channels`/`current_gains`、`impl AudioNode` 逐通道 process 缺通道也 advance 保相位、reset settle 到 target）；确定性数学全走 `bevy_math::ops`（`ops::sqrt`）、RT 热路径零分配/锁/panic、阶数钳位、短缓冲不 panic；新增 14 单测 + 1 doctest（crate 累计 **176 单测 + 16 doctest 全绿**）、clippy 全特性/无默认特性零告警（循环体多处 `needless_range_loop` 用 fn 级 `#[expect+reason]` 收敛）、no_std 双构建通过，已精确 commit（`61a4d4a9e`，仅 `hoa.rs` + spatial `lib.rs` 两文件）；刷新 §16/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何 UE/Unity/Godot/Wwise/FMOD/Steam Audio 源码或衍生代码）
- 版本历史: v0.23（**本版新增**：M4 空间层新增**独立 HRTF 双耳渲染 crate `prism_audio_hrtf`**（对齐 Steam Audio / Meta XR Audio 双耳，落地 §16 HRTF 双耳渲染 + §44.3 SOFA/AES69 装配）——8 文件 2103 行，职责单一分文件：`dataset`（`HrtfDataset` measurement-major 扁平 L/R 缓冲 + `Measurement`，越界返回空切片/`None` 不 panic）、`sofa`（`HrirSource` trait + `build_dataset` + `SofaRecords/SofaRecord/SofaConvention` + `aes69_to_local` AES69 角度转本地帧 + `HrirRecord`；**SOFA 二进制 HDF5/netCDF-4 解码诚实标注后续档**，改以 `HrirSource`+已解码 `SofaRecords`→dataset 真实路径，非假实现）、`interpolation`（`interpolate` = **ITD 群延迟对齐**（避免梳状抵消）+ **Shepard 反距加权**，`MAX_NEIGHBORS=4` 栈数组零分配 + `direction_from_angles`/`angular_distance`/`estimate_onset_delay`）、`binaural`（`BinauralRenderer` **overlap-save 分区卷积** + 持久历史延迟线 + HRIR 更新线性交叉淡入零咔哒，RT 零分配/锁/panic）、`nearfield`（`resolve` = 双耳视差几何 + 每耳精确 1/r 增益 + 球形头近场遮蔽近似（文档标注传输近似）+ `HeadGeometry`/`DEFAULT_HEAD_RADIUS`）；deps `prism_audio_core`/`prism_audio_spatial`（复用几何）+ `bevy_math`（`nostd-libm`）+ 可选 serde，features `default=["std"]`/`std`/`serialize`；`#![forbid(unsafe_code)]`、确定性经 clippy `disallowed-methods` 强制走 `bevy_math::ops`、热路径无 unwrap/expect/panic；golden 对拍 `exact_match_reproduces_measurement`/`alignment_avoids_comb_cancellation`/`near_source_has_larger_ild_than_far`/`shadow_attenuates_contralateral_ear` 等；**47 单测 + 1 doctest 全绿**、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit（`6c60cff2e`，仅本 crate 8 文件 + 根 Cargo.toml 一行 member）；刷新 §16/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.22（**本版新增**：M4 空间层新增**混响分区与游戏定义辅助发送 `reverb_zones`**（对齐 Wwise Game-Defined Aux Sends / UE Submix Sends / FMOD Snapshot 区域，落地 §17 辅助发送/Reverb Zones）——`ReverbZone{bus:AuxBusId, shape:ZoneShape, send_level, blend_distance}` 把世界空间区域绑定到一个辅助返回总线；`ZoneShape` 支持 `Box`（AABB）/`Sphere` 两形，统一以 `signed_distance`（内负外正，Box 用标准外部 SDF、Sphere 用欧氏距离）+ `contains` 表达；`weight(point)` 以 **cubic smoothstep** 在区域外缘 `blend_distance` 过渡带内由 1 平滑降到 0（两端导数为零，无可听拐点，进洞→出洞交叉淡化），`blend_distance=0` 退化硬边、`effective_send=send_level×weight`；`ReverbZoneField{zones}` 对**单一听者位置**一次 `resolve` 出活跃 `AuxSend{bus, level}` 集合写入定容缓冲——**同总线取最强**（主导环境胜，不重复发送）、不同总线各占一路（上限 `MAX_AUX_SENDS=4`）、超容淘汰最弱、低于 `MIN_AUDIBLE_GAIN` 忽略；**环境发送由听者驱动**（每控制块一次环境查询喂所有声源），每声源再以 `source_send_gain(zone_send, wet_gain)=zone_send×wet_gain`（钳[0,1]）叠加自身由距离/遮挡决定的湿声缩放（复用 `spatializer` 的 `SpatialParams::wet_gain`），把环境与声源两层职责解耦、每声源仅一次乘法；纯控制率函数零分配/锁/panic、距离数学走 `bevy_math::ops` 确定性可 golden 对拍；新增 13 单测 + 2 doctest（crate 累计 **160 单测 + 15 doctest** 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §17/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.21（**本版新增**：M4 空间层新增**房间与门户 `rooms`**（对齐 Wwise Rooms & Portals / Steam Audio）——`RoomNetwork{rooms, portals}` 实现 `PropagationBackend`，把**跨房间传播**接入既有 `propagation`→`occlusion`→`spatializer` 管线，复用 `AcousticMaterial` 与几何坐标约定；`Room{id, center, half_extents, wall}` AABB 声学体积（`contains`/`volume`），`room_of(point, rooms)` 取**最内层（最小体积）**含点房间、外部为 `None`；`Portal{center, normal, up, half_width, half_height, front/back, openness, material}` 矩形门户——`basis()` 正交基、`area()`、`connects`/`other_side`、`closest_point(p)` 投影钳到孔径矩形取最近点、`transmission_gain()`=`lerp(闭门材质透射, 1.0, openness)` 开合线性混合；`obliquity_factor(cos)=0.5·(1+|cos|)` **Fresnel-Kirchhoff 斜度因子**[0.5,1]、`portal_coupling_gain`=门户透射 × 声源侧斜度 × 听者侧斜度；`query`：同房间/皆在外→单直达全带 gain1；跨房间→(1) 直达穿墙 `Transmission` 路径（两房间墙透射增益乘积、`direct` 遮挡因子 block=1-wall_gain）+(2) 每个连通门户一条经孔径次级到达（方向偏向门户、延迟按绕行程长 (emitter→孔径+孔径→listener)/c、gain=coupling），上限 `MAX_PROPAGATION_PATHS=8` 且受缓冲界限；穿墙谱着色交由透射模型不臆造墙低通；纯几何 helper（`distance`/`normalize_or`/`any_perpendicular`）零分配/锁/panic、确定性走 `bevy_math::ops`；新增 18 单测 + 3 doctest（crate 累计 **147 单测 + 13 doctest** 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §17/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.20（**本版新增**：M4 空间层新增**几何声学传播后端 `propagation`**（对齐 Steam Audio 级、可插拔）——引入比 `OcclusionQuery` 更丰富的 `trait PropagationBackend`（物理层实现，本 crate **不硬依赖 `prism_physics`**），`query(listener, emitter, &mut [PropagationPath]) -> PropagationSummary` 于**控制率**枚举一个声源经**直达/透射/衍射/反射**四类机制（`PathKind`）到达听者的全部路径（各带 `delay_seconds`/`gain`/`cutoff_hz`/局部单位 `direction`，`delay_samples(sr)` 便捷、`MAX_PROPAGATION_PATHS=8` 有界），`PropagationSummary{direct: OcclusionFactors, path_count}` 同时喂 `occlusion` 系统；内置 `FreeFieldBackend` 自由场默认（单直达路径、`direct=OPEN`、全带 `FULL_BAND_CUTOFF_HZ`，consumers 钳 Nyquist）。**经典 DSP 可对拍**：`fresnel_number(δ,f)=2δf/c` Fresnel 数；**Maekawa 1968 屏障衍射** `maekawa_attenuation_db(N)`（N<-0.2→0dB、[-0.2,0) 线性 ramp 0→5dB、N≥0 → `5+20·log10(x/tanh(x))`（x=`sqrt(2πN)`，x→0 比值→1 避 0/0）钳 `MAX_DIFFRACTION_DB=24`；对拍 N=0→5dB/N=1→≈13.1dB/N=10→≈23dB/N=-0.1→2.5dB）、`diffraction_gain(δ,f)` 频率相关衍射增益、`diffraction_cutoff_hz(δ,sr)` 仿 `air` 32 点频率网格求衍射低通截止（阈值 8dB=5dB 阴影底 + 3dB 高频滚降，随程差单调不增、钳 Nyquist）、`transmission_gain(loss_db)=db_to_linear(-max(loss,0))` 透射损失、`edge_path_difference(L,E,S)` 绕边程差几何、`AcousticMaterial{transmission_loss_db, reflection}` 材质（`transmission_gain`/`reflection_gain` 钳位，仅 `OPEN` 默认不臆造材料表）；`bevy_math::ops` 无 `tanh`/`log10`，故 `tanh` 由 `exp` 构建、`log10` 由 `ln/LN_10` 构建，全路径确定性可 golden 对拍；纯 helper **零分配/锁/panic**、backend 明确为控制率（非 RT，可 trace 几何）；新增 14 单测 + 5 doctest（crate 累计 **129 单测 + 10 doctest** 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §14/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.19（**本版新增**：M4 空间层新增**多位置声源 `multi_position`**（对齐 Wwise Multi-Position）——一个逻辑声源映射到多个世界空间点（`PositionInput{emitter,factors}`，上限 `MAX_POSITIONS=16`），纯控制率函数 `resolve_multi(listener, positions, descriptor, sample_rate, mode) -> SpatialParams` 把逐点 `resolve` 结果折叠成单一图像，支持三种合成模式 `MultiPositionMode`：**Nearest**（取最近点）/**Blend**（按增益加权合成方向、取最响单点响度——加权不增益）/**Envelop**（全部点同时贡献：能量功率求和响度并钳到 1、方向取增益加权质心、按各点相对合成方向的角展宽扩散实现环绕）；**方向合成一律在监听者局部单位方向向量上做**（避免方位角环绕/正负相消误差），完全相对时退化到最近点方向保持良定义；pitch/直达低通取增益加权均值、湿声取最强；全路径栈上固定缓冲**零分配/锁/panic**、确定性数学走 `bevy_math::ops`，已接入 `spatializer` 生态；新增 8 单测 + 1 doctest（crate 累计 **115 单测 + 5 doctest** 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §15/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.18（**本版新增**：M4 空间层新增**扩散/聚焦塑形 `spread`**——以**多点虚拟子源**实现声像宽度：真实声源被替换为围绕其真实方位、半宽 `spread·PI` 的对称虚拟子源扇形，每个子源经任意 `Panner` 独立声像化后按归一化权重求和；权重由 **focus 控制的升余弦窗** `cos(u·PI/2)^(focus·MAX_FOCUS_POWER)` 给定（focus=0 均匀最散/focus=1 中心集中），spread=0 时扇形塌缩为精确点声源。`Spread` 携**距离曲线**（`near_spread`→`far_spread` 线性插值、范围外保持），`resolve(distance)->SpreadParams` 每控制块求值；`spread_taps` 生成对称扇形（tap 数截 `MAX_SPREAD_TAPS=9` 并强制奇数以精确采样中心）、`compute_spread_gains` 用栈上固定缓冲逐 tap 加权累加增益——全路径**零分配/锁/panic**、确定性数学走 `bevy_math::ops`（libm）；并**接入 `spatializer`**（`SourceDescriptor` 增 `spread` 字段、`SpatialParams` 携解析后的 `spread`）；新增单测（crate 累计 **107 单测 + 4 doctest** 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §15/§16/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.17（**本版新增**：M4 空间层新增**空间化编排层 `spatializer`**——引入 `SourceDescriptor`（authoring 数据：`attenuation`/`cone`/`doppler`/`occlusion`/`conditions`，可 serde）与纯控制率函数 `resolve(listener, emitter, descriptor, factors, sample_rate) -> SpatialParams`，把已落地的叶子模型组合成 per-source 单一解析结果：**直达增益** = 距离衰减 × 锥增益 × 遮挡直达增益（三个独立 [0,1] 线性因子相乘）、**多普勒频移比**（由 `localize` 的径向速度求得）、**方位/仰角**（监听者局部帧，供 `PannerNode`）、**直达低通截止**（取空气吸收与遮挡两条截止中更紧的一条——串联两级低通由更低者主导，避免级联双滤波）、**湿声发送缩放**（仅取遮挡值，obstruction 不动混响）；纯函数无音频状态/无分配/无锁/无 panic，可在 RT 线程调用，不新增 RT 节点（复用 PannerNode/AirAbsorptionNode/OcclusionNode + 变调重采样）；新增 12 单测（crate 累计 95 单测 + 4 doctest 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §14/§15/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.16（**本版新增**：M4 空间层新增**遮挡/障碍模型 `occlusion`**——可插拔 `trait OcclusionQuery`（物理层实现，本 crate 不依赖 `prism_physics`）+ `NullOcclusionQuery` 默认全开；`Occlusion` 配置按业界 obstruction/occlusion 语义把两个归一化因子映射为**直达增益**（dB 线性衰减）、**直达低通截止**（对数域频率插值、端点精确吸附）与**湿声发送缩放**（仅由 occlusion 驱动，obstruction 不动混响路径）；RT `OcclusionNode` 复用 core `Biquad`，热路径零分配/锁/panic、低通仅在非 RT 侧重设计；新增 10 单测（crate 累计 83 单测 + 3 doctest 全绿）、clippy 全特性/无默认特性零告警、no_std 双构建通过，已精确 commit；刷新 §14/§39/§40；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.15（**本版新增**：**M4 空间层 `prism_audio_spatial` 首批落地并通过验证**——在已 commit 的几何基座（`geometry`：`Listener`/`Emitter`/`LocalSource` + `localize`，右手系 +X 右/+Y 上/-Z 前，方位/仰角）之上新增 6 个职责单一的经典 DSP 模块并逐个精确 commit：`attenuation`（OpenAL 1.1 clamped 距离模型 Inverse/Linear/Exponential）、`cone`（内/外锥角 + 外锥增益，按前向与朝向夹角插值）、`doppler`（径向速度→频移比，`SPEED_OF_SOUND=343`、强度系数与最大比钳制）、`air`（真实 ISO 9613-1:1993 空气吸收系数 + 32 点频率网格求截止频率 + 复用 core `Biquad` 低通的 `AirAbsorptionNode`）、`panner`（`trait Panner` + `VbapPanner` 各布局扬声器方位环 pairwise 等功率 + `PannerNode` 逐通道 `Smoothed`）、`ambisonics`（AmbiX ACN/SN3D FOA 编码 `encode_foa_*`/场旋转 `rotate_foa`/解码 `decode_foa` + `FoaEncoderNode`）；全部确定性数学走 `bevy_math::ops`（libm）、RT 热路径零分配/锁/panic、72 单测 + 3 doctest 全绿、clippy 全特性/无默认特性零告警、`std` + `--no-default-features` no_std 双构建通过；据此刷新 §14/§15/§16 落地标注、§39 crate 表 spatial 行、§40 M4 状态；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.14（**本版新增**：据实借鉴 OS 音频栈（CoreAudio/WASAPI/ALSA·JACK）与 Rust 实时音频工程实践，补齐三处产品级韧性/可信度缺口——**§22 设备韧性**（xrun/欠载识别与 PLC 式淡出隐藏 + 遥测上报、默认设备变更/热插拔在非 RT 侧重开流续跑与优雅降级、引擎与设备时钟漂移的有界异步重采样 `DriftResampler` 且与“开流即拒绝不匹配率”正交）、**§10 采样精确 seek/scrub**（内存源保分数相位、流式源重填预取、与循环点/事件同一样本网格对齐，对齐 Wwise/FMOD `setPosition`）、**§28 可验证 RT 安全**（无分配守卫把“热路径零分配”变机器可验证、无锁/无阻塞检查、图编译器与无锁环 `cargo-fuzz` 目标、ASan/TSan + 双构建 + golden 的 CI 门禁矩阵）；§2.1 追踪表 +1 行、§42 开放问题 +3、术语表 +5 并保持 §49 编号；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.13（据实校准设计与已落地代码——**M3 设备层 `prism_audio_device` 已实现并通过验证**：cpal 原生输出后端（默认特性门控、`--no-default-features` 可离线构建、采样率不匹配即报错不静默重采样、F32/F64/I16/U16/I32/I8/U8 全格式回调分派、回调零分配/零锁）、`BlockRenderer` 固定块→可变交错缓冲拉取适配器（设备与离线共用同一渲染路径、逐样本一致）、`render_to_wav` 确定性离线 32-bit float WAV、无锁环形捕获（麦克风/总线回读、整帧溢出计数）；11 单测全绿、clippy 零告警、双特性配置构建通过、已 commit。据此刷新 §22 设备后端、§39 crate 表、§40 M3 路线图状态，使方案与仓库实现保持一致；恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本历史: v0.12（§47 物理耦合程序化音频（接触/模态/颗粒合成 · 直接联动本仓库 `prism_physics` 与 §31 GPU 场景同一几何真相）——借鉴 Wwise Impacter 与经典模态合成/物理建模，把「碰撞→查表播 wav」升级为「冲量→采样精确激励模态/颗粒合成」，滚动/摩擦/滑动为随相对速度演化的连续合成；§48 输出渲染链与母带交付档（ITU-R BS.775 下混矩阵、低频管理 LFE 交叉、Home Theater/TV/Night/耳机 动态范围交付档、平台响度目标与对白锚定响度）——补齐「内部规范格式→具体收听设备」最后一公里与主机认证硬指标；§2.1 追踪表补「主机认证响度 / 杜比输出模式 / BS.775 下混」一行；§41 扩展点 +6（`ContactEventSource`/`ModalSynth`/`GranularEngine`/`DownmixMatrix`/`OutputProfile`/`BassManager`）；§42 开放问题把 Impacter 项落为 §47 并新增输出档默认值决策；术语表 +14 并顺延为 §49；均恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或行生代码）
- 版本: v0.11（**本版新增**：§46 次世代 DSP 与工程内核深化（借鉴超越）——零延迟分区卷积（UPOLS 非均匀分区）、planar SIMD 内核抽象、反规格化数保护（FTZ/DAZ + denormal-safe DSP）、Ambisonic 声场旋转与双频段解码（max-rE / in-phase + 虚拟扬声器双耳）、近场 HRTF 与视差补偿、频变空气吸收（ISO 9613-1 式）、反馈安全图（单样本延迟破代数环）、声明式自动混音与类别响度治理（CRIWARE REACT / Category 式）；§2 逐引擎深读补「日系 / 底层中间件参照（CRIWARE ADX2 / RAD Miles）」；§2.1 追踪表补 CRIWARE ADX2 与 Wwise 2024 Reflect·卷积·Auto-Ducking 两行；术语表 +12 并顺延为 §47；均恪守纯经典 DSP、无任何 AI/ML、无任何引擎源码或衍生代码）
- 版本: v0.10（**本版新增**：§45 实时通信音频（位置语音/AEC 回声消除/经典降噪 NS/AGC/VAD/自适应抖动缓冲/丢包隐藏 PLC，**全经典 DSP、无任何 AI/ML**）——把实时语音纳入统一渲染图作为一等声源，复用 §13 母带/§14 遮挡/§15 距离/§16 空间化/§31 GPU 声学/§32 LOD；§2.1 追踪表补 WebRTC 经典 DSP 与 Wwise Communication·平台语音两行；§41 扩展点 +3（`VoiceCommPipeline`/`EchoCanceller`/`VoiceTransport`）；§42 开放问题补通信抖动缓冲档位与「是否放开纯经典约束纳入神经音频」决策项；术语表 +9，术语表顺延为 §46）
- 版本历史（v0.9 及更早，**当版新增**：§43 波动声学与混合精算传播（Project Acoustics/Triton/ARD 式离线波动场 + 几何/GPU 射线混合，作为可插拔 `PropagationBackend` 的波动档）、§44 编解码/对象音频/个性化空间输出（Opus/Vorbis/ADPCM 编解码矩阵、杜比 Atmos/MPEG-H 床+对象、SOFA/AES69 测量 HRTF 个性化选型——**均恪守本文档纯经典 DSP 路线、无任何 AI/ML**）；§2.1 追踪表补齐 Project Acoustics/索尼360RA·苹果空间音频/杜比Atmos·MPEG-H/Opus 四行采纳映射，§41 扩展点与 §42 开放问题、术语表相应扩充。前序 v0.8：**M1 全族已完成**：effects/dynamics/reverb 三族节点全部落地，103 单测 + 4 doctest 全绿、clippy 零告警、no_std 双构建通过，已 commit；现进入 **M2 声源与调度**（源节点/采样精确调度器/命名时钟/语音池）。本版据实校准 §3/§9/§39/§40 状态，并新增 §2.1「较新版本特性追踪（2023–2025）」把各引擎最新一代能力（Steam Audio UTD 衍射与探针烘焙、Wwise Impacter/Strata、UE5.4/5.5 MetaSound Pages、Unity 6 DOTS Audio）纳入采纳映射，深化 §8 调度器与 §10 声源的 M2 落地细节。v0.7 前序：基础层已落地编码：`pkg/prism_audio_core`，M1 效果族已落地首个节点 `ParametricEqNode`（级联复用 `Biquad`，35 测试绿/clippy 零告警/no_std 双构建通过）；本版聚焦「逐引擎深读」精修方案：扩展 §2 为业界参考+采纳映射+逐引擎深读（UE5/Unity/Godot 借鉴·覆盖·超越三段式），并补齐各引擎较新特性——UE5 MetaSound Builder API 运行时构图（§11）、Audio Gameplay Volumes（§17）、Audio Insights 检视（§26）；Godot AudioStreamInteractive 片段式交互流与过渡类型（§19）、AudioStreamPolyphonic 多voice复用（§25）、延迟补偿播放头查询（§8）；Unity Audio Random Container 原生随机容器（§18）、Timeline 音频轨（§19）；相应增补 §41 扩展点 `PatchBuilder` 与 §42 开放问题。v0.6 前序：新增内容生产与跨模态层：对白与本地化（程序化对白/语言 Bank/字幕同步/viseme 口型）、触感与跨模态输出（音频同源触感/触感总线/DualSense·双马达后端）、程序化环境音景（Soundscape 调色板与散布）、实时授权与远程工具 API（WAAPI 式远程遥测+白名单写命令/live tuning/授权热重载），并为 §16 增补头部追踪双耳、§28 增补属性/模糊测试。v0.5 前序：编译图执行模型（拓扑计划/缓冲活跃度分配/就地别名/PDC）、并行 DSP 图调度（Job 化/确定性并行/岛屿划分）、GPU 加速几何声学（共享渲染器 BVH 的声线与路径追踪）、性能自适应治理与音频 LOD、心理声学虚拟化与声源聚类、时间伸缩变调与重采样质量分级；扩展了参考映射、扩展点、路线图与术语表。v0.4 前序：程序化内容图 Patch、调制系统、多普勒/锥形/Spread/Focus/多位置、遮挡与障碍区分、Aux 发送与环境、HDR 音频、Bank/流式/内存、输入捕获、平台空间后端、虚拟语音行为、Profiler 与可视化调试）
- 范围: 一步到位（统一渲染图 / 采样精确调度 / 程序化 Patch / 调制 / 几何声学 / Event+RTPC 编排 / 交互音乐 / LUFS+HDR 母带 / 平台空间输出）
- 适用引擎: Prism / Bevy ECS 生态
- 关键依赖: bevy_ecs（并行 ECS）、bevy_math（glam SIMD + `ops` 确定性标量数学）、bevy_tasks（资产解码/烘焙任务）、bevy_asset（音频资产与 Bank）、bevy_transform（听者/声源位姿）、bevy_a11y（无障碍）、cpal/AudioWorklet（设备后端，前端 crate）
- 复用的现有设施: bevy_asset 加载与热重载、bevy_tasks 异步解码/烘焙、bevy_math::ops（跨平台确定性 sin/cos/exp/log）、bevy_transform 空间层级、bevy_diagnostic 诊断面板、prism_physics 射线/几何查询（遮挡与反射复用）、prism_material_pipeline 表面属性协同
- 关联文档: `prism_physics_design_zh.md`（几何/射线查询原语，供遮挡与反射复用）、`prism_material_pipeline_design_zh.md`（声学材质与视觉材质的表面属性协同）

---

## 目录
1. 设计哲学与目标
2. 业界参考、采纳映射与逐引擎深读（UE5 / Unity / Godot）
3. 分层架构
4. 概念模型：Graph / Node / Bus / Voice / Patch
5. 统一实时渲染图
6. 数据布局与缓冲池
7. 参数系统与采样精确自动化
8. Transport 与采样精确调度器（多时钟）
9. 节点库（DSP 原语矩阵）
10. 声源与合成
11. 程序化内容图（Patch / MetaSounds 式可编译子图）
12. 音频调制系统（Modulation：控制总线 / LFO / 包络 / 曲线）
13. 母带与动态处理（LUFS / True-Peak / Sidechain / HDR）
14. 空间音频：几何声学传播（遮挡 / 障碍 / 透射 / 衍射 / 反射）
15. 距离与方向塑形（衰减曲线 / 锥形 / Spread / Focus / 多普勒 / 多位置）
16. HRTF / Ambisonics / 对象音频 / 平台空间后端
17. 环境与辅助发送（Aux Sends / Reverb Zones / Rooms & Portals）
18. 内容模型：Event / Container / State / Switch / RTPC
19. 交互音乐系统
20. 资产、Bank 与流式媒体
21. 无锁 ECS 集成层与线程/内存模型
22. 设备后端、离线渲染与输入捕获
23. 无障碍（Accessibility）
24. 确定性与网络
25. 语音管理与虚拟化
26. 剖析、遥测与可视化调试
27. 性能预算与验收
28. 质量与可信度基础设施
29. 编译图执行模型（拓扑计划 / 缓冲活跃度分配 / 就地与别名优化）
30. 并行 DSP 图调度（Job 化渲染 / 确定性并行 / 岛屿划分）
31. GPU 加速几何声学（共享渲染器 BVH 的光线与路径追踪）
32. 性能自适应治理与音频 LOD（CPU 预算驱动质量缩放）
33. 心理声学虚拟化与声源聚类（掩蔽感知剔除 / 对象床限制）
34. 时间伸缩与变调 / 重采样质量分级（与多普勒解耦）
35. 对白与本地化（Dialogue / Localization / 字幕 / 口型）
36. 触感与跨模态输出（Haptics / Motion / 手柄反馈）
37. 程序化环境音景（Soundscape / 程序化 Ambience）
38. 实时授权与远程工具 API（Live Authoring / WAAPI 式 / 热调）
39. Crate 拆分与落地形态
40. 路线图
41. 关键扩展点清单
42. 开放问题
43. 波动声学与混合精算传播（预计算波动场 + 几何/GPU 射线混合）
44. 编解码、对象音频与个性化空间输出（Codec / Object-based / MPEG-H / 测量 HRTF）
45. 实时通信音频（Communication / 位置语音 / 回声消除 / 抖动缓冲）
46. 次世代 DSP 与工程内核深化（借鉴超越）
47. 物理耦合程序化音频（接触 / 模态 / 颗粒合成 · 物理引擎联动）
48. 输出渲染链与母带交付档（下混矩阵 / 低频管理 / 动态范围与响度交付）
49. 术语表

---

## 1. 设计哲学与目标

对标商用旗舰音频中间件与引擎，确立八条铁律：

- **单一统一渲染图**：全引擎所有声源/效果/总线/空间化器都是同一张有向无环图（DAG）中的 `AudioNode`，而不是"每个声源一个独立 Sink"。融合 Web Audio 的 `AudioNode` 图、UE Submix 树、Godot Bus 链三者优点：一次离线编译成确定性、零分配的处理计划。
- **实时铁律（RT-safety）**：音频回调线程上的一切（`AudioGraph::process` 可达代码）**零分配、无锁、不 panic、不阻塞**。所有状态（滤波器记忆、延迟线、平滑参数）构造期预分配。图变更与编译只在音频线程之外发生。
- **内容与代码解耦（Event 驱动）**：游戏逻辑只触发 `Event`（"脚步"、"爆炸"），永不直接播放文件、永不硬编码总线/音量。声音设计师在数据侧决定随机化、容器、状态与实时参数（RTPC）——这是 Wwise/FMOD 的核心生产力，也是与"直接 `play(sound.wav)`" 玩具引擎的关键差异。
- **内容即图（程序化 Patch）**：单个声音本身也可以是一张可编译的 DSP 子图（Patch，对齐 UE5 MetaSounds）——用振荡器/包络/滤波/采样器程序化合成，而不仅是回放固定波形。Patch 编译后作为一个 `AudioNode` 嵌入运行时图，采样精确、零分配。
- **几何驱动空间化**：空间音频不是"距离衰减 + 声像"，而是遮挡→障碍→透射→衍射、实时/烘焙反射、Rooms & Portals、HRTF/Ambisonics 的完整链路（Steam Audio 级），且传播后端可插拔。
- **采样精确 + 多时钟**：一次性声、无缝循环、节拍量化的音乐过渡、stinger，全部对齐到样本，由样本计数驱动的 `Transport` 与并发命名时钟（对齐 UE Quartz），而非帧率抖动的游戏时钟。
- **确定性可回放**：可注入种子 RNG；相同输入产出相同样本。联机时音频作为本地表现层由确定性事件触发，天然可做 golden/parity 测试。
- **可扩展 + 可剖析**：Node / Patch / PropagationBackend / Panner / Modulator / SourceDecoder / DeviceBackend 均为 trait 插件点，第三方无需改内核即可注册；全链路可经遥测环导出到 Profiler（对齐 Wwise Profiler）。

设计取舍总表：

| 维度 | Resonance 选择 | 理由 |
|---|---|---|
| 图模型 | 单一统一 DAG（编译后处理） | 融合 Web Audio/Submix/Bus，零解释、确定性 |
| 内容模型 | Event 驱动 + 可编译 Patch | 生产力（Wwise/FMOD）+ 程序化合成（MetaSounds） |
| 处理粒度 | 定长块（block）planar 缓冲 | 对 per-channel DSP 与 SIMD 友好 |
| RT 分配 | 编译期预分配、热路径零分配 | 音频线程无 GC/malloc 抖动 |
| 内部格式 | f32 planar | 144dB 动态范围、总线可超 0dBFS 不裁切 |
| 数学 | bevy_math::ops（libm） | 跨平台位一致，可回放 |
| 参数 | 逐样本 Smoothed + 调制栈 | 无 zipper noise，可叠加 LFO/包络 |
| 空间后端 | 可插拔（几何/平台 SDK） | 兼容 Windows Sonic/Tempest/Atmos/XR |
| 调度 | 样本计数 Transport + 命名时钟 | 采样精确、多并发节拍网格 |
| 执行模型 | 编译期 ExecPlan + 缓冲活跃度分配 | 零解释、内存最优、可逐字节对拍 |
| 图调度 | 编译期岛屿划分 + Job 化并行 | 高语音密度可扩展，确定性不变 |
| 声学加速 | 复用渲染器 GPU BVH 做声线/路径 | 声学与视觉同一几何真相，异步无 RT 阻塞 |
| 质量伸缩 | CPU 预算闭环 + 音频 LOD | 稳帧不爆音，感知优先分配算力 |

---

## 2. 业界参考、采纳映射与逐引擎深读（UE5 / Unity / Godot）

| 引擎/标准 | 我们采纳的核心思想 | 落地位置 |
|---|---|---|
| **Web Audio API** | `AudioNode` 有向图、参数自动化、离线渲染上下文 | §5 图 / §7 参数 / §22 离线 |
| **UE5 MetaSounds** | 采样精确程序化 DSP 图作为"内容"，可编译成节点 | §11 Patch |
| **UE5 Submix / Source Effect Chain** | 总线树 + 声源效果链 + 发送 | §5 图 / §9 节点 / §17 发送 |
| **UE5 Quartz** | 样本精确、量化的音乐/事件时钟，多并发时钟 | §8 调度器 |
| **UE5 Audio Modulation** | 调制控制总线、LFO/包络/曲线、参数目标、叠加规则 | §12 调制 |
| **UE5 HDR-Attenuation / Wwise HDR** | 动态响度窗口，突出前景声、压抑背景声 | §13 HDR |
| **Unity AudioMixer / Snapshots** | 总线组 + 快照插值 + sidechain ducking | §13 / §18 States |
| **Unity Spatializer & Ambisonic Decoder SDK** | 可插拔空间化器 / Ambisonic 解码接口 | §16 Panner |
| **Unity DSP Graph** | 数据导向、无 GC 的 DSP 图执行 | §5 图 / §6 缓冲 |
| **Godot AudioServer（Bus/Effect/Stream）** | 总线链、效果实例、Stream 抽象 | §5 / §9 / §10 |
| **Godot SpectrumAnalyzer / AudioEffectCapture / Generator** | 频谱分析、捕获总线、程序化推流 | §26 剖析 / §22 捕获 / §10 |
| **Godot AudioStreamPlayer3D + Microphone** | 3D 声源属性 / 麦克风输入 | §15 / §22 |
| **Wwise Event/Container/State/Switch/RTPC** | 数据驱动内容与实时参数控制 | §18 内容 |
| **Wwise Interactive Music** | 段/播放列表/量化过渡/stinger/垂直分层 | §19 音乐 |
| **Wwise Occlusion vs Obstruction** | 遮挡（直达+混响）与障碍（仅直达）区分 | §14 空间 |
| **Wwise Aux Sends / Game-Defined Aux** | 环境混响发送由游戏体积驱动 | §17 发送 |
| **Wwise Virtual Voices / Playback Limit** | 虚拟语音行为、优先级、实例上限 | §25 语音 |
| **Wwise Profiler / Meters** | 实时捕获、语音监视、计量、事件时间线 | §26 剖析 |
| **Wwise SoundBank / FMOD Bank** | 资产打包、内存/流式媒体、预取 | §20 资产 |
| **FMOD Studio Parameter/Snapshot/Transceiver** | 全局/本地参数、快照、无线发送 | §12 / §17 / §18 |
| **Steam Audio** | 遮挡·衍射·透射·反射（实时+烘焙）·HRTF·Ambisonics·探针 | §14 / §16 |
| **ITU-R BS.1770 / EBU R128** | LUFS 响度测量、门限、true-peak | §13 母带 |
| **RBJ Audio EQ Cookbook** | biquad 系数公式（已实现于 `nodes/biquad.rs`） | §9 节点 |
| **AmbiX / ACN-SN3D 约定** | Ambisonics 通道序与归一化标准 | §16 Ambisonics |
| **平台空间音频（Windows Sonic / Tempest 3D / Atmos / Meta XR）** | 对象音频床、平台原生双耳/多声道解码 | §16 平台后端 |
| **现代渲染图（render graph）transient 资源别名** | 编译期缓冲活跃度分析 + 图着色复用中间缓冲 | §29 执行模型 |
| **Unity DSP Graph（数据导向并行执行）** | 岛屿划分 + Job 化确定性并行渲染 | §30 图调度 |
| **Steam Audio 光线/路径 + Prism GPU BVH** | 复用渲染器加速结构做 GPU 声学传播 | §31 GPU 声学 |
| **游戏引擎 LOD / Wwise 语音上限（闭环化）** | CPU 预算驱动质量档与音频 LOD 自适应 | §32 治理器 |
| **心理声学掩蔽 / Atmos 对象床上限** | 掩蔽感知虚拟化 + 邻近声源聚类归并 | §33 感知层 |
| **相位声码器 / WSOLA / 多相 sinc** | 变调不变速 / 变速不变调 / 重采样质量分级 | §34 时间伸缩 |

| **FMOD Programmer Instrument / Wwise External Sources·Dialogue Event** | 运行时程序化选择对白媒体、语言变体、决策树命中 | §35 对白 |
| **本地化字幕轨 / viseme 口型时间线** | 字幕同步与口型/表情驱动数据（不占 RT） | §35 对白 |
| **Wwise Motion / PS5 DualSense·Tempest Haptics** | 音频同源触感、触感总线、宽频/双马达可插拔后端 | §36 触感 |
| **UE5 Soundscape** | 环境状态调色板 + 程序化 one-shot 散布，低重复环境床 | §37 音景 |
| **Wwise Authoring API (WAAPI) / FMOD Live Update** | 远程只读遥测 + 白名单写命令 + 实时调参与授权热重载 | §38 授权 |
| **UE5 MetaSound Builder API（运行时构图）** | 运行时以代码/数据增量拼装并热切换 Patch，而非仅离线烘焙 | §11 Patch |
| **UE5 Audio Gameplay Volumes（5.3+）** | 体积驱动的室内外/混响/衰减覆盖与门户连通，取代旧混响体 | §17 发送 |
| **UE5 Audio Insights（5.4+）** | 引擎内实时声源/总线/参数/虚拟化检视面板 | §26 剖析 |
| **Godot 4.x AudioStreamInteractive / Playlist / Synchronized** | 片段式交互音乐（immediate/next-beat/next-bar/marker 过渡）与多流同步 | §19 音乐 |
| **Godot 4.x AudioStreamPolyphonic** | 单播放器动态多 voice 复用（脚步/连发免手动建声源） | §25 语音 |
| **Godot 延迟补偿播放头查询** | playback_position + time_since_last_mix − output_latency 对齐画面 | §8 调度器 |
| **Unity Audio Random Container（2023.2+）** | 引擎原生随机容器（音高/音量/顺序/避免重复） | §18 内容 |
| **Unity Timeline 音频轨** | 时间线上采样精确编排音频剪辑/事件 | §19 音乐 |

**次世代差异化**（相对单一中间件的组合优势）：
- 空间传播复用 `prism_physics` 的射线/几何查询做遮挡与反射，**声学与物理共享同一场景表示**，避免重复维护碰撞体。
- 声学材质与视觉材质在资产层协同（`prism_material_pipeline`），一个表面同时携带吸收/散射/透射系数与视觉 BRDF。
- 程序化 Patch（MetaSounds 级）+ Event 驱动内容（Wwise 级）+ 几何声学（Steam Audio 级）**三线合一**，而非只取其一。
- 全链路 `bevy_math::ops` 确定性数学 + 种子 RNG，使音频可做 **golden 逐样本对拍测试**，与仓库"不造假 parity"的工程 ethos 一致。
- 无锁命令环 + ECS 原生，声源即 `Entity`，天然融入 Prism 的 `Transform` 层级与并行调度。
- **声学复用渲染器 GPU 加速结构**（§31）：作为"渲染器 + 音频引擎"合体，声学传播直接跑在渲染管线的 BVH/meshlet 上，独立中间件（各自维护声学场景）无法做到，这是最强次世代差异。
- **编译图执行 + 确定性并行 + 质量治理**（§29/§30/§32）：把图当作可编译、可并行、可逐字节对拍的确定性计划，并在 CPU 预算内闭环缩放质量——兼得高密度、稳帧与可回放。

**逐引擎深读（借鉴 / 我们覆盖在哪 / 我们如何超越）**

*UE Series（UE4 SoundCue → UE5 MetaSounds 时代）*
- **借鉴**：MetaSounds 把“单个声音”变成采样精确、可编译的程序化 DSP 图；Submix 树 + Source/Submix Effect Chain 做分层路由；Quartz 提供样本精确、多并发的音乐/事件时钟；Audio Modulation 用控制总线做参数联动；HDR-Attenuation 做动态响度窗口；较新版还有 **MetaSound Builder API**（运行时构图）、**Audio Gameplay Volumes**（体积驱动室内外/混响/门户，5.3+）、**Audio Insights**（引擎内检视，5.4+）、Convolution Reverb Submix。
- **覆盖**：Patch=§11、Submix/效果链=§5/§9/§17、Quartz=§8、Modulation=§12、HDR=§13；本版补齐 Builder API=§11、AGV=§17、Audio Insights=§26。
- **超越**：MetaSounds 的几何空间化仍依赖外部插件（Steam Audio），我们把几何声学（§14）与 **GPU BVH 声学**（§31）内建并与渲染器共享同一几何真相；全链路 `bevy_math::ops` 确定性数学使 Patch 与母带可做 **golden 逐样本对拍**（UE 不保证跨平台位一致）。

*Unity（AudioMixer / DSP Graph / DOTS Audio 时代）*
- **借鉴**：AudioMixer 组 + Snapshot 插值 + 暴露参数 + sidechain ducking；Native Audio Plugin SDK 的可插拔 **Spatializer / Ambisonic Decoder** 接口；DSP Graph（DOTS Audio）的数据导向、无 GC 并行执行；较新的 **Audio Random Container**（2023.2+，引擎原生随机容器）与 Timeline 音频轨。
- **覆盖**：Mixer/Snapshot/ducking=§13/§18、可插拔 Panner/Ambisonic=§16、DSP Graph 数据导向=§5/§6 与岛屿并行=§30；本版补齐 Random Container=§18、Timeline 轨=§19。
- **超越**：Unity 的 Snapshot 是控制率插值、AudioMixer 图为运行时解释；我们是**编译期 ExecPlan + 缓冲活跃度分配**（§29）零解释执行，参数逐样本 `Smoothed`（§7）无 zipper，且 §30 岛屿并行是**确定性可对拍**的（DOTS Audio 不保证跨平台样本一致）。

*Godot（AudioServer：Bus / Effect / Stream 时代）*
- **借鉴**：AudioServer 的 Bus 链 + 效果实例 + Stream 抽象；SpectrumAnalyzer / AudioEffectCapture / AudioStreamGenerator（频谱/捕获/程序化推流）；Area 驱动的混响总线覆盖；麦克风捕获；较新的 **AudioStreamInteractive**（片段图 + 过渡类型 immediate/next-beat/next-bar/marker）、**AudioStreamPolyphonic**（单播放器多 voice）、**AudioStreamSynchronized/Playlist**，以及**延迟补偿播放头查询**。
- **覆盖**：Bus/Effect/Stream=§5/§9/§10、频谱/捕获/Generator=§26/§22/§10、Area 混响=§17、麦克风=§22；本版补齐交互流=§19、Polyphonic=§25、延迟补偿播放头=§8。
- **超越**：Godot 的空间化仅距离衰减 + 简单混响，无遮挡/衍射/HRTF/Ambisonics 完整链路；我们提供 Steam Audio 级几何传播（§14/§16）；Godot 混音为运行时逐总线处理，我们是编译图 + Job 化并行（§29/§30）+ CPU 预算治理（§32），密度与稳帧维度代差领先。

*专业中间件参照（Wwise / FMOD / Steam Audio）*：内容生产力（Event/Container/State/Switch/RTPC=§18、交互音乐=§19、Bank/流式=§20、Profiler=§26、WAAPI/Live Update=§38）与几何声学（§14/§16）已系统性采纳；差异化在于把中间件的“内容生产 + 几何声学”与引擎内建的“渲染器共享 GPU 声学 + 编译图确定性并行 + 可回放对拍”合一，而非以外挂中间件形式并存。

*日系 / 底层中间件参照（CRIWARE ADX2 / RAD Miles）*
- **借鉴**：CRIWARE ADX2（大量日系 AAA 与开放世界项目采用）贡献四个独到范式——**AISAC**（多维交互控制曲线：一个游戏量经曲线簇同时驱动一组音量/滤波/选层，比单一 RTPC 更接近「控制面」）、**Block 播放**（以「播放块 + 块间转移」为单位的采样精确音序，天然支持无缝分段与量化跳转）、**REACT**（类别间自动闪避/让路的声明式自动混音）、**Category 音量树**（正交于总线树的类别增益/上限层）；RAD Miles Sound System 则是流式解码 + 低层混音 + 平台抽象的老牌工程范本。
- **覆盖**：AISAC=§12 调制栈的「多目标控制总线 + 曲线」（一个控制源扇出多参数）叠加 §18 RTPC；Block 播放=§8 采样精确调度器的块量化转移 + §19 交互音乐段落；REACT=§13 sidechain/ducking 的**声明式类别规则**扩展（详见 §46.8）；Category 音量树=正交于总线树的「类别增益层」（并入 §13 母带与 §18 States 之外的独立增益轴）。
- **超越**：ADX2 的 AISAC/REACT 是运行时解释的控制层；我们把它降为 §12 调制图的**编译期静态连线 + 逐样本 `Smoothed`**（无 zipper、可 golden 对拍），并把 REACT 式类别让路规则并入 §32 `QualityGovernor` 的 CPU 预算闭环，使「自动混音」与「质量治理」共享同一决策面，而非两套割裂系统。

### 2.1 较新版本特性追踪（2023–2025）与采纳映射

上表覆盖各引擎的经典能力；本节补齐**最新一代**（2023–2025）出现、值得借鉴的特性，并给出 Resonance 的落地/超越判断。原则不变：**只借思想，不含任何源码或衍生代码**。

| 引擎/版本 | 较新特性 | 我们的采纳 / 落地位置 |
|---|---|---|
| **Steam Audio 4.x** | 频率相关**透射**（多频段材质损失）、基于 **UTD（均匀衍射理论）** 的边缘衍射、**探针批次（Probe Batch）烘焙**反射并运行时插值、**TrueAudio Next** GPU 卷积加速 | 3 频段透射曲线并入 §14 声学材质；UTD 衍射作为 `PropagationBackend` 的边缘路径求解档位（§14）；探针烘焙 = §14 反射「烘焙」路径的具体数据结构；GPU 卷积对齐 §31 复用渲染器算力 |
| **UE 5.4 / 5.5 MetaSounds** | **MetaSound Pages**（按平台/质量分档选择子图实现）、**Wavetable** 合成节点族、内联空间化与 MIDI/trigger 类型扩展、Builder API 成熟化 | Pages 直接映射 §32 音频 LOD 的「按预算选 Patch 变体」，落到 §11 `PatchBuilder` 的**分档编译**；Wavetable 合成并入 §10 `WavetableNode`/§11 Patch 原语 |
| **Wwise 2023.1 / 2024.1** | **Impacter**（物理冲量驱动的程序化撞击/材质合成）、**Strata** 分层多变体音效库、Motion（触感统一）、Spatial Audio 几何驱动衍射/房间门户成熟化 | Impacter = 物理碰撞冲量→Patch（§11）合成参数的**跨系统联动**（新增 §42 开放问题）；Strata 分层变体并入 §18 容器/§20 Bank 分层预取；Motion=§36 触感；几何衍射/门户=§14/§17 已覆盖 |
| **FMOD Studio 2.02+** | Programmer Instrument（运行时选媒体）、Spatializer + Resonance Audio 后端、Bank 部分加载 | Programmer=§35 对白 `DialogueResolver`；Resonance Audio=§16 可插拔 `Panner` 后端；部分加载=§20 Bank 粒度预取 |
| **Unity 6 / DOTS Audio** | DOTS Audio（Burst 编译、无 GC 的数据导向 DSP）、Audio Random Container 原生化、新一代 Spatializer | DOTS 数据导向执行=§5/§6 + §30 岛屿并行的既有取向；Random Container=§18；Spatializer 接口=§16 `Panner` trait |
| **Godot 4.3 / 4.4** | 实时 MIDI 输入、AudioStreamInteractive/Synchronized/Playlist、播放统计（playback stats）、`AudioStreamPolyphonic` | 交互流=§19、Polyphonic=§25、延迟补偿播放头=§8 已补；实时 MIDI 作为 §8 调度器的采样精确 note 事件源（新增扩展点） |
| **微软 Project Acoustics（Triton/ARD 波动声学）** | 离线**波动仿真**烘焙 + 运行时**感知参数**（遮挡/衰减/到达方向/混响）查表插值，物理正确处理低频、绕射、房间耦合与动态开口 | 新增 §43 波动档：作为 `PropagationBackend` 的一档，与几何档（§14）、GPU 档（§31）**混合**，复用渲染器同一几何真相并统一到同一空间参数总线 |
| **索尼 360RA / 苹果个性化空间音频** | 个性化 HRTF（人体测量/耳廓扫描选型）提升双耳定位精度 | 新增 §44.3 走**经典数据驱动**路径：SOFA(AES69) 测量集加载 + 人体测量选型/运行时校准，**无 ML 推理**（恪守本文档纯经典 DSP 路线） |
| **杜比 Atmos / MPEG-H 3D Audio** | 床+对象（Atmos ≤128 对象）/ 对象+声道+HOA 沉浸式标准（ATSC 3.0 广播） | 新增 §44.2：空间总线原生产出**床+对象**元数据流，平台后端直收或折算进床/双耳，超限对象走 §33 聚类 |
| **Opus / Vorbis / ADPCM 编解码** | 低延迟语音/对白、流式音乐、大量并发短音效的编码分级 | 新增 §44.1 `SourceDecoder` 编解码矩阵：PCM/ADPCM/Vorbis/Opus/FLAC 按用途分档，后台线程解码、RT 只读 |
| **WebRTC 经典 DSP（AEC3 / NS / AGC / VAD）** | 回声消除、噪声抑制、自动增益、语音活动检测的**经典自适应滤波/谱域**实现（**非 ML**） | 新增 §45.2 采集前处理链，纳入 `VoiceCommPipeline` 上行链 |
| **Wwise Communication / 平台 Voice Chat（PS/Xbox/Meta）** | 实时语音采集/编码/传输/空间化的通信子系统 | 新增 §45：通信语音作为统一图**一等声源**，复用 §14/§15/§16 空间化、§13 母带、§32 LOD |
| **CRIWARE ADX2（2023+）** | AISAC 多维交互控制、Block 播放音序、REACT 自动混音、Category 音量树、ACB/AWB 分包流式 | AISAC=§12 多目标控制总线；Block=§8 块量化转移 + §19 段落；REACT=§13 声明式让路并入 §32/§46.8 治理；Category=独立类别增益层（§13）；ACB/AWB=§20 Bank 粒度预取 |
| **Wwise 2024.1 / Reflect·卷积·Auto-Ducking** | Reflect 实时镜像源反射、卷积混响 Submix、Auto-Ducking 带恢复曲线、Meter 作 RTPC 源 | Reflect=§14 实时反射档；卷积=§9 `convolver` + §46.1 零延迟分区卷积；Auto-Ducking 恢复曲线=§13/§46.8；Meter→RTPC=§26 + §18 |
| **主机认证响度 / 杜比输出模式 / ITU-R BS.775** | 主机出货强制的响度与下混合规、客厅影院/电视/夜间/耳机动态范围模式、标准下混系数（-3/-4.5/-6 dB）与低频管理（LFE 交叉/校准） | 新增 §48 输出渲染链：BS.775 下混矩阵 + 低频管理 + 交付动态范围档（Home Theater/TV/Night/耳机）+ 平台响度目标与对白锚定，作为 §13 母带之后、设备之前的显式确定性末级 |
| **OS 音频栈（CoreAudio / WASAPI / ALSA·JACK）与 Rust 实时音频工程实践** | 设备欠载（xrun）语义、默认设备变更/热插拔事件、设备与引擎时钟漂移的异步重采样、以及“进入回调即禁止堆分配”的可验证 RT 安全约束（分配守卫/无锁环模糊/sanitizer） | §22 设备韧性与时钟漂移；§10 采样精确 seek；§28 可验证 RT 安全（无分配守卫/`cargo-fuzz`/CI 矩阵） |

**超越判断（保持代际差异）**：这些较新特性多为「单点能力」升级，而 Resonance 的差异化在于把它们**统一进同一条编译图 + 确定性并行 + 渲染器共享 GPU 声学 + 逐样本可对拍**的骨架里——例如 MetaSound Pages 的分档只解决内容分档，我们让分档编译直接受 §32 `QualityGovernor` 的 CPU 预算闭环驱动；Steam Audio 的探针烘焙是独立中间件维护自己的声学场景，我们复用渲染器 BVH（§31）使声学与视觉同一几何真相。

---

## 3. 分层架构

自底向上四层，每层一个 crate，下层不依赖上层：

```
┌─────────────────────────────────────────────────────────────┐
│  L4  bevy_audio 前端（ECS 组件/系统、AudioPlayer 兼容 API）   │  → crates/bevy_audio（改接命令通道）
├─────────────────────────────────────────────────────────────┤
│  L3  prism_audio_authoring（Event/Container/State/RTPC/音乐/  │  → pkg/prism_audio_authoring
│      Patch 编译/Modulation/Bank）                             │
│      prism_audio_spatial（几何传播/HRTF/Ambisonics/panner）   │  → pkg/prism_audio_spatial
│      prism_audio_device（cpal/worklet/离线 FileSink/捕获）    │  → pkg/prism_audio_device
├─────────────────────────────────────────────────────────────┤
│  L2  prism_audio_core::nodes（gain/biquad/pan/mix/…效果/动态）│  → pkg/prism_audio_core（已落地）
├─────────────────────────────────────────────────────────────┤
│  L1  prism_audio_core（math/buffer/param/time/graph）         │  → pkg/prism_audio_core（已落地）
└─────────────────────────────────────────────────────────────┘
```

- **L1 内核（已实现）**：`math`（Sample/dB/denormal/等功率声像）、`buffer`（planar `AudioBuffer` + `ChannelLayout`）、`param`（`Smoothed`/`Ramp`）、`time`（`Transport`/`TimeSignature`）、`graph`（`AudioNode`/`AudioGraph`/编译/块渲染）。
- **L2 节点库（M1 完成，M2 进行中）**：全部实现 `AudioNode` trait 的具体处理单元。基础 `GainNode`/`BiquadNode`/`StereoPanNode`/`SumNode`/`LinkwitzRileyCrossover`（LR4 多频带分频）+ effects（parametric_eq/delay/waveshaper/chorus/flanger/phaser/stereo_width）+ dynamics（detector/compressor/limiter/gate/ducking）+ reverb（fdn/convolver/algorithmic）已全部落地；M2 扩展声源与采样精确调度见 §8/§9/§10。
- **L3 子系统**：空间、编排/事件（含 Patch 编译与调制）、设备/捕获。互相独立、写集不相交，适合并行开发。
- **L4 前端**：`bevy_audio` 保留现有 `AudioPlayer`/`PlaybackSettings`/`Volume` API 兼容，内部改接无锁命令通道。

---

## 4. 概念模型：Graph / Node / Bus / Voice / Patch

- **AudioNode**（trait）：单个处理单元。契约——`process(&mut self, ctx, io)` 必须 RT 安全（不分配/锁/阻塞/panic），内部状态构造期预分配；`reset()` 清零；`latency_frames()` 报告延迟以供延迟补偿。
- **AudioGraph**：`AudioNode` 与其连接的容器。生命周期：`add_node`/`connect`（可分配，线程外）→ `compile`（Kahn 拓扑排序 + 预分配全部中间缓冲）→ `process`（音频线程，零分配）。
- **Port / 连接**：节点有若干输入/输出 port，每个 port 有 `ChannelLayout`。同一输入 port 的多条入边自动求和；一条输出可扇出多个输入。`connect_with_gain` 提供 send 风格的增益连接。层不匹配在连接期即被拒绝。
- **Bus（总线）**：约定意义上的"汇聚节点"（如 `SumNode` 或带效果链的子图），对应 Godot Bus / UE Submix。总线本身也是图中的节点，无特殊类型。
- **Voice（语音）**：一个正在发声的声源实例（一次 Event 触发可产生多个）。语音由语音池（§25）管理，虚拟化/优先级/限量，超限时按响度与优先级淘汰。
- **Patch（内容子图）**：由内容侧编排的一张小 DSP 图（振荡器/采样器/包络/滤波/数学节点），离线编译成单个 `AudioNode`（`PatchNode`）后嵌入运行时图（§11）。这是"声音即程序"的载体。

`master`：图指定某个输出 port 为母带输出，`process` 把它拷入调用者缓冲。

---

## 5. 统一实时渲染图

**核心差异化**。已落地于 `pkg/prism_audio_core/src/graph.rs`：

- **编译**：`compile()` 用 Kahn 算法做节点级拓扑排序；检测到环返回 `GraphError::Cycle`。为每个 port 预分配 `max_block` 容量的 `AudioBuffer`；对所有节点调用 `reset()`。
- **块渲染**：`process(frames, playhead, master_out)` 按拓扑序遍历——先清空本节点输入 port，再把上游输出按边增益累加进来，调用节点 `process`，最后把 master port 拷给调用者。全程零分配。
- **ProcessIo**：向节点暴露 `input(port)`/`output(port)`/`io(ip,op)`（效果原地变换的常见形态）。也可 `ProcessIo::new` 脱离图独立驱动一个节点（用于池化语音或单元测试）。
- **错误模型**：`GraphError::{UnknownNode, PortOutOfRange, LayoutMismatch, Cycle, NoMaster}`，连接/编译期充分校验，运行期不再校验（RT 铁律）。

图变更策略（线程外）：采用**三缓冲/epoch 图交换**——在任务线程构建/编译新图，通过命令环把"就绪的编译图"原子交给音频线程；音频线程在块边界切换指针，旧图经 epoch 回收队列在无引用后于任务线程 drop（避免在 RT 线程 drop 分配）。详见 §21。

---

## 6. 数据布局与缓冲池

已落地于 `buffer.rs`：

- **planar 存储**：`data[ch * capacity_frames + frame]`，每通道连续，利于 per-channel DSP 与自动向量化。
- **固定容量**：通道数与最大帧数构造期固定，RT 线程永不重分配；`active_frames` 可缩小用于流末尾的部分块。
- **通道布局**：`ChannelLayout::{Mono, Stereo, Quad, Surround5_1, Surround7_1, AmbisonicFoa}`（`#[non_exhaustive]`，可扩展 7.1.4 等 Atmos 床与 HOA 阶）。
- **混音原语**：`add_scaled`（图求和的基石）、`copy_from`、`channel_pair_mut`（无借用检查器摩擦的立体声处理）、`clear`。

规划：块内存来自图编译期分配的 arena；跨块的延迟线/卷积 tail 由各节点自持（构造期分配）。上/下混只在显式转换节点发生。

---

## 7. 参数系统与采样精确自动化

已落地于 `param.rs`：

- **Smoothed**：`Copy`、无堆数据，可直接内嵌进 RT 节点。`set_target(value, ramp)` 设置目标；`next_sample()` 在最内层 DSP 循环逐样本推进。
- **Ramp**：`Immediate`（瞬跳，仅用于非逐样本量如模式切换）、`Linear{samples}`（线性）、`Exponential{tau_samples}`（一极点，带吸附阈值确保收敛）。`Ramp::linear_seconds(s, sr)` 由秒构造。
- **无 zipper noise**：所有可听参数（增益/截止/声像）走 `Smoothed`，避免块边界阶跃爆音。
- **自动化事件**：命令环可携带带样本偏移的参数事件（"在本块第 N 帧把增益设为 X"），实现块内采样精确的参数变化（对齐 Web Audio `setValueAtTime`）。

调制叠加见 §12——最终参数值 = 基值（Smoothed）经调制栈（LFO/包络/控制总线）按叠加规则合成。

---

## 8. Transport 与采样精确调度器（多时钟）

已落地 `time.rs`（`Transport`/`TimeSignature`），规划调度器：

- **Transport**：样本计数驱动的播放头，`samples_per_beat`/`samples_per_bar`/`next_bar_boundary`/`seconds_to_samples`/`set_tempo_bpm`。这是音乐与量化事件的时间基准，不随帧率抖动。
- **命名时钟（Quartz 式）**：允许多个并发命名时钟（如"音乐时钟 128 BPM"与"环境脉冲时钟"），各自维护拍/小节网格，事件可量化到指定时钟的边界。
- **采样精确调度器**：维护按触发样本排序的事件堆；每块开始把落入本块的事件按帧偏移插入，语音/参数在精确帧启停。跨块事件保留到后续块。
- **量化过渡**：过渡对齐拍/小节/段边界（用 `next_bar_boundary`），保证无缝（§19）。
- **前瞻窗口**：调度器提前一个块预调度，配合设备缓冲吸收抖动。
- **延迟补偿播放头查询（对齐 Godot）**：向 gameplay 暴露“画面对齐”的播放位置 `pos = raw_playhead + time_since_last_mix − output_latency`，供画面/字幕/节奏玩法精确同步，而非直接用抖动的游戏帧时钟读原始播放头。
- **实时 MIDI / note 事件源（对齐 Godot 4.3 实时 MIDI + UE Quartz）**：外部 MIDI/序列作为调度器的采样精确事件源，note-on/off 经命令环下发并落到指定时钟的量化边界，与 Patch（§11）触发端口贯通。

**M2 落地契约（RT 铁律）**：
- 事件堆为**构造期预分配的有界最小堆**（按触发样本排序），块内插入/弹出零分配；溢出走可配置策略（丢最旧/记遥测），绝不在音频线程分配。
- 命名时钟集合固定容量（构造期定），每时钟维护 `(bpm, time_sig, sample_origin)`，量化 API 提供 `quantize(target_sample, Grid)`（Grid=Beat/Bar/Nth/Marker），返回对齐后的绝对样本；跨块事件保留至后续块，保证块边界不吞事件。
- 前瞻窗口 = 一个块，量化过渡（§19）在前瞻内决议，避免边界抖动。

---

## 9. 节点库（DSP 原语矩阵）

已实现（`pkg/prism_audio_core/src/nodes/`，103 单测 + 4 doctest 全绿、clippy 零告警、no_std 双构建）：

- **基础**：`GainNode`、`BiquadNode`（RBJ 7 型）、`StereoPanNode`（等功率）、`SumNode`（N 输入求和）、`LinkwitzRileyCrossover`（Linkwitz-Riley LR4 多频带分频，全通相位补偿）。
- **effects**：`parametric_eq`（biquad 级联）、`delay`（分数延迟 + 反馈 + 湿干）、`waveshaper`（过采样防混叠）、`chorus`/`flanger`/`phaser`（调制延迟族）、`stereo_width`（Mid-Side 立体声展宽 + 可选 bass-mono 交叉分频）。
- **dynamics**：`detector`（峰值/RMS 检波）、`compressor`（软/硬拐点 + 前瞻）、`limiter`（前瞻）、`gate`（扩展门）、`ducking`（侧链闪避）。
- **reverb**：`fdn`（反馈延迟网络）、`convolver`（分块卷积）、`algorithmic`（Freeverb 式 pre-delay + 早反射 + 并联 comb + 串联 allpass + 立体声宽度）。

仍规划（M2 及以后）：

规划扩展（每个：完整实现、构造期预分配、带 impulse/golden 稳定性测试）：

- **effects**：`ParametricEqNode`（biquad 级联）、`DelayNode`（分数延迟 + 反馈 + 湿干）、`WaveshaperNode`（过采样防混叠）、`ChorusNode`/`FlangerNode`/`PhaserNode`（调制延迟）、`FilterSweepNode`。
- **dynamics**：`CompressorNode`（软/硬拐点、前瞻）、`LimiterNode`（前瞻 true-peak）、`ExpanderGateNode`、`DuckingNode`（sidechain 输入）、`MultibandCompressorNode`。
- **reverb**：`FdnReverbNode`（反馈延迟网络，色散扩散）、`ConvolverNode`（分块 FFT 卷积，实测 IR）、`AlgorithmicRoomNode`（早反射 + 尾混）。
- **spatial**（L3 crate）：`VbapPannerNode`、`AttenuationNode`（距离衰减 + 空气吸收 + 锥形）、`HrtfNode`、`AmbisonicEncodeNode`/`AmbisonicDecodeNode`、`DopplerNode`。
- **routing**：`VcaNode`（增益控制总线）、`SendReturnNode`、`ChannelConverterNode`（上/下混）、`TransceiverNode`（无线发送，对齐 FMOD Transceiver）。
- **modulation**（见 §12）：`LfoNode`、`EnvelopeFollowerNode`、`ControlBusNode`。
- **analysis**（见 §26）：`MeterNode`（峰值/RMS/LUFS）、`SpectrumNode`（FFT）、`CaptureNode`（回读缓冲）。
- **sources**：见 §10。

---

## 10. 声源与合成

规划（L2/L3）：

- **SamplePlayerNode**：解码后 PCM 的播放头，支持循环点、分数重采样（线性/Catmull-Rom）、变速播放、start/stop 采样精确。解码在 `bevy_tasks` 任务线程，RT 线程只读环形/预载缓冲。
- **StreamingSource**：长音频流式，双缓冲预取，欠载保护（输出静音而非阻塞）。见 §20 流式媒体。
- **OscillatorNode / WavetableNode**：带限振荡器（sine/saw/square/triangle）用 **PolyBLEP** 在跳变点做多项式带限校正抑制混叠；相位累加走 `bevy_math::ops` 保证跨平台位一致；波表合成（对齐 UE5.4 MetaSound Wavetable）用相位读表 + 线性/Catmull-Rom 插值，mip 化波表按基频选层进一步抗混叠。
- **NoiseNode**：white（种子 PRNG）/ pink（Voss-McCartney 或 Paul Kellet 一阶滤波器组，恒 -3dB/oct）/ brown（积分白噪 + 泄漏防漂移）；种子确定性，逐样本可对拍。
- **SamplePlayerNode（变调重采样）**：播放头 + 分数重采样（线性/Catmull-Rom）+ 循环点（forward/ping-pong）+ 播放速率；高变调倍率并入抗混叠（过采样或波表 mip 思路），start/stop 采样精确对齐 §8 事件。
- **GeneratorSource**：外部程序按块推流（对齐 Godot `AudioStreamGenerator`），供 gameplay 生成的 PCM；RT 侧只读**预分配环形缓冲**，欠载输出静音不阻塞。
- **采样精确 seek / scrub（对齐 Wwise/FMOD `setPosition`）**：`SamplePlayerNode` 与 `StreamingSource` 支持把播放头跳到任意样本位置——命中内存的采样源直接重置分数相位（seek 落在两样本间时保留分数部分，避免咔哒）；流式源经命令环通知解码任务**重填预取缓冲**（丢弃旧预取、从目标样本重新解码，编解码器有预滚/填充时按 §44.1 校准裁剪），重填未就绪期间 RT 侧输出静音而非旧数据。seek 与循环点/事件调度（§8）对齐同一样本网格，供过场跳转、检查点续播、对白快进采样精确复现。
- **SilenceNode**：确定性静音源（占位/测试）。

资产：经 `bevy_asset` 加载 wav/ogg/flac；解码格式插件化（`SourceDecoder` trait）。程序化声音优先走 Patch（§11）而非固定波形。

---

## 11. 程序化内容图（Patch / MetaSounds 式可编译子图）

规划 crate `prism_audio_authoring`（对齐 UE5 MetaSounds，"声音即程序"）：

- **Patch 定义**：一张小型 DSP 图，节点为振荡器/采样器/包络/滤波/数学/逻辑原语，输入为 Patch 参数（频率/触发/衰减…），输出为若干音频通道。定义为数据资产（可编辑、可热重载）。
- **编译为节点**：Patch 经与运行时图相同的编译器（Kahn 拓扑 + 预分配）离线编译成一个 `PatchNode`，实现 `AudioNode`。运行时图看到的只是一个普通节点，采样精确、零分配。
- **输入/触发**：Patch 暴露命名输入（对齐 RTPC/参数）与触发端口（如 note-on）。触发经 §8 调度器采样精确注入。
- **确定性合成**：所有振荡/随机走 `bevy_math::ops` 与种子 RNG，Patch 输出逐样本可对拍。
- **复用与嵌套**：Patch 可作为节点被更大的 Patch/图引用（有界嵌套深度，编译期展开）。
- **运行时增量构图（Builder API 式，对齐 UE5 MetaSound Builder API）**：除离线烘焙外，提供 `PatchBuilder` 在任务线程以代码/数据增量拼装或改写 Patch，编译成新 `PatchNode` 后经无锁命令环与 §21 epoch 原子热切换（旧节点确认无 RT 引用后延迟回收），实现运行时程序化音色演化——超越“只能离线固化内容”的传统管线，同时不破坏 RT 铁律（构图/编译永不在音频线程）。

价值：脚步、UI、武器、程序化环境声可完全由 Patch 合成，减少波形资产、天然随机化、内存友好——这是 MetaSounds 相对纯采样回放引擎的代际优势。

---

## 12. 音频调制系统（Modulation：控制总线 / LFO / 包络 / 曲线）

规划（对齐 UE5 Audio Modulation + FMOD 调制器）：

- **调制控制总线（Control Bus）**：一条命名的标量控制信号（如"紧张度"、"水下程度"），可被多个参数订阅。控制总线本身可被 RTPC、LFO、包络、其他总线驱动。
- **调制器（Modulator，trait）**：
  - `LfoModulator`（正弦/三角/方波/S&H，速率可同步到 §8 时钟）
  - `EnvelopeFollowerModulator`（跟随某总线电平，做自适应闪避/泵感）
  - `AdsrModulator`（触发型包络）
  - `CurveModulator`（RTPC 值经曲线映射）
- **参数目标与叠加规则**：一个参数（如某总线增益）可被多个调制器叠加，叠加规则可选 `Mix`（求和）/`Multiply`/`Max`/`Min`（对齐 UE Modulation Mixing）。最终值再进 §7 `Smoothed` 平滑。
- **RT 求值**：调制图在音频线程按块（或按控制率）求值，写入各节点的参数目标；求值零分配、拓扑有序、无环（编译期校验）。
- **与 RTPC 的关系**：RTPC 是"游戏量→参数"的直接映射；调制系统是"参数间与时变信号"的组合层。二者可级联（RTPC → 控制总线 → 多目标）。

---

## 13. 母带与动态处理（LUFS / True-Peak / Sidechain / HDR）

规划（对齐 ITU-R BS.1770 / EBU R128 + Wwise/UE HDR）：

- **响度测量**：K-weighting 预滤波 + 门限积分，输出 Integrated/Short-term/Momentary LUFS 与 Loudness Range。
- **响度归一化**：目标 LUFS（如 -16 游戏、-23 广播）自动增益（可按资产/总线归一）。
- **True-Peak Limiter**：4× 过采样峰值检测 + 前瞻，防止 inter-sample peak 削波，母带最后一级。
- **Sidechain Ducking**：`DuckingNode` 以对白/音乐总线为 sidechain 键，压低环境总线（对齐 Unity AudioMixer ducking）。
- **HDR 音频窗口**（对齐 Wwise HDR / UE HDR-Attenuation）：以场景内最响声源为参考，动态调整"响度窗口"，突出前景声（如近处枪声）、压抑背景声（远处环境），在有限扬声器动态范围内表达巨大声压差。窗口用 §7 平滑避免抽吸感。
- **Snapshot（快照）**：混音状态快照与插值切换（战斗/探索/过场），对齐 Unity Snapshot / Wwise States / FMOD Snapshot。快照由 §18 States 驱动。

---

## 14. 空间音频：几何声学传播（遮挡 / 障碍 / 透射 / 衍射 / 反射）

crate `pkg/prism_audio_spatial`（Steam Audio 级，可插拔 `PropagationBackend`）。**已落地（M4 首批，已 commit）**：`geometry` 几何基座（`Listener`/`Emitter`/`LocalSource` + `localize`，右手系 +X 右/+Y 上/-Z 前）、6 个叶子 DSP 模块（`attenuation`/`cone`/`doppler`/`air`/`panner`/`ambisonics`，见 §15/§16），以及**本节的遮挡/障碍模型**（`occlusion`：可插拔 `trait OcclusionQuery` + `NullOcclusionQuery` 默认全开、`Occlusion` 配置把 obstruction/occlusion 两因子映射为直达增益/对数域低通截止 + 湿声发送缩放、RT `OcclusionNode`）。以及**本节的透射/衍射/反射传播后端**（`propagation`：可插拔 `trait PropagationBackend` 解耦物理层 + `FreeFieldBackend` 自由场默认，`query` 控制率枚举直达/透射/衍射/反射四类 `PathKind` 路径为 `PropagationPath` 数组 + `PropagationSummary`；经典 DSP：Fresnel 数 + **Maekawa 1968 屏障衍射** `maekawa_attenuation_db`、`diffraction_gain`/`diffraction_cutoff_hz` 频率相关衍射低通、`transmission_gain` 透射损失、`edge_path_difference` 绕边程差、`AcousticMaterial` 材质）。**真实几何因子仍依赖上层 `prism_physics` 射线实现 backend 后接入（见下）**：

- **遮挡（Occlusion）vs 障碍（Obstruction）**（对齐 Wwise 语义区分）：
  - **障碍（Obstruction）**：仅**直达路径**被挡（听者与声源在同一混响空间，但中间有物体），只衰减/低通直达声，混响仍完整。
  - **遮挡（Occlusion）**：直达**与**混响路径均被挡（声源在另一空间），直达与湿声一起衰减。
  - 二者由 `prism_physics` 射线 + 房间归属判定区分，分别驱动直达增益/低通与 aux 发送量。
- **透射（Transmission）**：穿过材质的频变衰减，材质携带透射损失曲线。
- **衍射（Diffraction）**：绕过边缘的路径，路径查询给出衍射角与附加衰减。
- **反射（早期反射已落地 `prism_audio_spatial::early_reflections`）**：实时早期反射用 **Allen-Berkley 1979 镜像源法**（矩形房间整数索引枚举镜像，`compute_early_reflections` 产出延迟/增益/到达方向 tap 列表 + `EarlyReflectionRenderer` 单延迟线多抽头渲染），后续可扩几何射线镜像/光线追踪反射；烘焙（离线预计算响应，运行时插值，探针网格）见 §43。
- **声学材质**：表面携带吸收/散射/透射系数，与 `prism_material_pipeline` 视觉材质在资产层协同。

后端可插拔：默认几何后端；可注册更高精度（波动/BEM）或第三方后端。遮挡/障碍查询复用物理射线，避免重复维护碰撞体。**波动档的离线预计算（Project Acoustics/ARD 式）与几何/GPU/波动的混合传播详见 §43，后端选择矩阵同列于该节。**

---

## 15. 距离与方向塑形（衰减曲线 / 锥形 / Spread / Focus / 多普勒 / 多位置）

对齐 Wwise/FMOD/Unity 3D 声源属性。**已落地（`prism_audio_spatial`，已 commit）**：距离衰减（`attenuation`：OpenAL 1.1 clamped 的 Inverse/Linear/Exponential）、锥形（`cone`：内/外锥角 + 外锥增益按夹角插值）、多普勒（`doppler`：径向速度→频移比，`SPEED_OF_SOUND=343`、强度系数与最大比钳制）、空气吸收（`air`：ISO 9613-1:1993 系数 + 频率网格求截止 + 复用 core `Biquad` 的 `AirAbsorptionNode`）、**扩散/聚焦（`spread`：多点虚拟子源扇形 + focus 升余弦窗 + 距离曲线，`spread_taps`/`compute_spread_gains` RT 零分配，已接入 `spatializer`）**、**多位置声源（`multi_position`：一逻辑源映射多点，`resolve_multi` 三模式 nearest/blend/envelop 折叠为单一 `SpatialParams`，方向在单位向量上加权合成、envelop 能量求和 + 角展宽扩散）**。至此 §15 距离与方向塑形全族落地：

- **距离衰减曲线**：可配置形状（线性/对数/自定义曲线 + 最小/最大距离），驱动增益、低通（空气吸收）、混响发送量、Spread 等多条曲线（对齐 Wwise Attenuation ShareSets）。
- **锥形衰减（Cone）**：声源朝向 + 内/外锥角 + 外锥增益与低通，模拟指向性声源（喇叭/人声）。
- **Spread（扩散）**（**已落地**）：随距离控制声像宽度——远处点声源收窄，近处可环绕，避免"点声源贴脸"失真；多点虚拟子源扇形 + 距离曲线线性插值。
- **Focus（聚焦）**（**已落地**）：控制能量集中程度，与 Spread 配合塑造宽/窄声像；升余弦窗指数由 focus∈[0,1] 缩放到 [0,MAX_FOCUS_POWER]。
- **多普勒（Doppler）**：由听者/声源相对径向速度计算频移，`DopplerNode` 用分数延迟线实现连续变调（无爆音），可配置多普勒强度系数。速度取自 `Transform` 帧间差分或显式速度组件。
- **多位置声源（Multi-Position）**（**已落地**）：一个逻辑声源映射到多个空间位置（对齐 Wwise Multi-Position），用于大型/分布式声源（河流、人群、机器），按 Nearest（最近）/Blend（加权）/Envelop（全部）模式合成空间参数——方向在监听者局部单位向量上按增益加权、Envelop 能量功率求和并按角展宽扩散实现环绕。

上述所有塑形量经 §7 `Smoothed` 平滑，随听者/声源运动逐块更新。

---

## 16. HRTF / Ambisonics / 对象音频 / 平台空间后端

**已落地（`prism_audio_spatial`，已 commit）**：多布局 `panner`（`trait Panner` + `VbapPanner` 各布局扬声器方位环 pairwise 等功率 + `PannerNode` 逐通道 `Smoothed` 平滑）与 FOA `ambisonics`（AmbiX ACN/SN3D 编码 `encode_foa_*`、场旋转 `rotate_foa`、解码 `decode_foa` + `FoaEncoderNode`），以及**独立 HRTF 双耳渲染 crate `prism_audio_hrtf`**（见下条 ✅），并新增**高阶 Ambisonics(HOA) 编解码 `hoa`**（✅ 已落地，见下条）。头部追踪双耳（✅ 已落地 `prism_audio_hrtf::headtracked`，见下条）。HOA 声场旋转（✅ 已落地 `hoa_rotation`）。**对象/Atmos 与平台原生空间后端桥仍在规划**：

- **HRTF 双耳渲染（✅ 已落地 `prism_audio_hrtf`）**：`BinauralRenderer` overlap-save 分区卷积 HRIR（`interpolate` 按方位/仰角以 ITD 群延迟对齐 + Shepard 反距加权 4 邻插值，避免梳状抵消），近场 `nearfield::resolve` 双耳视差 + 每耳 1/r 增益 + 球形头遮蔽近似（ITD/ILD），可经 `HrirSource` trait + `SofaRecords`/`aes69_to_local` 装配自定义 HRTF 数据集（SOFA 二进制解码为诚实后续档，个性化选型见 §44.3）；HRIR 更新线性交叉淡入零咔哒、RT 零分配/锁/panic、确定性走 `bevy_math::ops`（47 单测 + 1 doctest，clippy 零告警，no_std 双构建，已 commit）。
- **头部追踪双耳（Head-tracked Binaural）（✅ 已落地 `prism_audio_hrtf::headtracked`）**（对齐 Meta XR Audio / Steam Audio 头追）：`HeadPose`+`predict` 四元数轴角指数映射一阶外推（seconds 钳 `[0, 0.1]`）、`HeadTracker::update` 先外推再 slerp 平滑（`alpha=1-exp(-dt/tau)`）、`world_to_local_direction`/`local_azimuth`/`local_elevation`/一步式 `predicted_local_angles` 喂既有 `interpolation::interpolate` 重选 HRIR 方位；XR/VR 下低延迟短前瞻使头动到声像更新，避免"声像黏在头上"；姿态更新经命令环下发、RT 侧插值平滑，确定性走 `bevy_math::ops`（13 单测 + 1 doctest，clippy 零告警，no_std 双构建，已 commit）。
- **Ambisonics（FOA + HOA ✅ 编解码已落地 `prism_audio_spatial`）**：FOA/HOA 场景总线，声源编码进 Ambisonic 域，最终按输出布局解码（双耳/多声道）。约定采用 **AmbiX（ACN 通道序 + SN3D 归一化）**，与主流工具链兼容。FOA 见 `ambisonics`；HOA 见 `hoa`（至三阶 16 通道，`encode_hoa`/`decode_hoa`（解码除以 `order+1`）/`HoaEncoderNode`，纯 Cartesian 实球谐递推、零向量退化全向）。**声场旋转（✅ 已落地 `hoa_rotation`）**：实球谐分块对角旋转，一阶块 `R^1=A·Q·A` + 二/三阶 Ivanic–Ruedenberg 递推，`rotate_hoa`/`HoaRotationMatrix`（控制率算一次、多帧复用），golden `rotate_hoa(encode(d))==encode(q·d)` 对拍（见 §46）。**近场补偿（NFC，✅ 已落地 `nfc`）**：有限距离点声源的近场效应用参考距离稳定化的逐阶补偿滤波修正——`H_m(s)=theta_m(s*r_src/c)/theta_m(s*r_ref/c)`（反向 Bessel 多项式根硬编码 + 双线性变换到一阶/biquad 级联，高频归一、DC 增益 `(r_ref/r_src)^m` 有限、全极点在单位圆内稳定），`NfcCoeffs::design`/`NfcFilter`（见 §46.5）。
- **对象音频 / Atmos**：对象元数据（位置/大小）输出到支持的床（7.1.4）或下混到扬声器/耳机（床+对象模型、MPEG-H 与对象预算/聚类见 §44.2）。
- **平台空间后端**（`Panner`/输出适配可插拔）：耳机（内建 HRTF）、立体声、5.1/7.1；并可桥接平台原生空间 API——**Windows Sonic / Spatial Sound**、**索尼 Tempest 3D**、**杜比 Atmos**、**Meta XR Audio**——由 `prism_audio_device` 侦测并选择解码路径。
- **输出适配**：自动按设备与用户偏好选择 HRTF / 多声道 / 对象床路径。
- **跨听/串音消除渲染（Transaural，✅ 已落地 `prism_audio_hrtf::transaural`）**：用一对立体声扬声器回放双耳信号时，用递归串音消除器抵消对侧扬声器到耳的声学串音（`CrosstalkCanceller`：`s_l=b_l-beta*s_r; s_r=b_r-beta*s_l` 精确逆对称串音矩阵 `C=[[1,beta],[beta,1]]`，beta=延迟 d>=1+头影增益 g<1+一阶低通、环路增益 g^2<1 无条件稳定；`CrosstalkParams::from_geometry` 由扬声器半角 Woodworth ITD 推导路径），使双耳空间线索在扬声器上重建。

Panner 可插拔（`Panner` trait），对齐 Unity Spatializer / Ambisonic Decoder SDK。

---

## 17. 环境与辅助发送（Aux Sends / Reverb Zones / Rooms & Portals）

对齐 Wwise Aux Sends / UE Submix Sends / FMOD Snapshot 区域（辅助发送与 Reverb Zones ✅ 已落地 `prism_audio_spatial::reverb_zones`）：

- **辅助发送（Aux Send，✅ 已落地）**：声源除干路外，可按可变增益发送到一个或多个混响/效果返回总线（`AuxBusId`/`AuxSend`）。每声源发送量随距离曲线（§15）与遮挡/障碍状态（§14）经 `spatializer` 的 `wet_gain` 动态调整，再由 `source_send_gain` 与所处 Reverb Zone 的环境发送相乘。
- **游戏定义发送（Game-Defined Aux，✅ 已落地 `reverb_zones`）**：由听者所处的**混响体积（`ReverbZone`，Box/Sphere 形）**自动决定发往哪个环境总线与发送量（进洞穴→洞穴混响，出洞→户外）；`ReverbZoneField::resolve` 每控制块一次由听者位置解析活跃 `AuxSend` 集合（同总线取最强、上限 `MAX_AUX_SENDS`），过渡用区域外缘 `blend_distance` 内的 cubic smoothstep 平滑（呼应 §7）。
- **Rooms & Portals（✅ 已落地 `prism_audio_spatial::rooms`）**：房间体积 + 门户连接。`RoomNetwork` 实现 `PropagationBackend`：`Room` AABB 声学体积 + `wall` 透射材质、`room_of` 取最内层房间归属；`Portal` 矩形门户以 `openness` 混合闭门材质透射与全开、`closest_point` 求最近孔径点、`portal_coupling_gain`=透射×Fresnel-Kirchhoff 斜度因子；`query` 同房间→单直达，跨房间→直达穿墙透射路径 + 每个连通门户一条经孔径的次级到达（方向/延迟/增益随几何），与 §14 遮挡/障碍/传播联动。**单跳门户由 `rooms` 覆盖；多跳门户路由已由 `portal_graph` 落地（房间图有界无环 DFS 深度<=4、去环、累积增益/延迟、top-k）；真实几何 backend 为后续档。**
- **返回总线**：混响/延迟返回作为普通节点存在于统一图（§5），可再被母带链（§13）处理。

---

## 18. 内容模型：Event / Container / State / Switch / RTPC

规划 crate `prism_audio_authoring`（Wwise/FMOD 生产力核心）：

- **Event**：游戏触发的最小单位（"footstep"、"explosion"）。Event 携带动作（play/stop/set-param/set-switch/set-state）。游戏代码只发 Event。
- **Container**：
  - `Random`（随机选一，带避免重复窗口）
  - `Sequence`（顺序播放）
  - `Blend`（按参数交叉淡化多层，如引擎转速）
  - `Switch`（按 Switch 状态选分支）
  - `Scatter`（空间散布，如群鸟）
- **States**（全局）：游戏状态（战斗/潜行）驱动混音快照（§13）与 Event 变体。
- **Switches**（每对象）：如"地表材质"决定脚步音色。
- **RTPC（实时参数控制）**：连续游戏量（速度/血量/紧张度）经映射曲线驱动任意参数（音量/滤波/音高），可级联到调制控制总线（§12）。

编排层把这些解析成对 L1 图、§11 Patch 与 §8 调度器的命令，经 §21 命令环下发。

---

## 19. 交互音乐系统

规划（对齐 Wwise Interactive Music / FMOD Transition）：

- **Segment / 播放列表**：音乐段带入点/出点/前后余量；播放列表定义段的顺序/循环/随机。
- **量化过渡**：过渡在拍/小节/段边界发生（用 §8 `next_bar_boundary` 与命名时钟），采样精确无缝。可配过渡段（transition segment）桥接。
- **Stinger**：叠加短乐句（如命中提示），精确对齐到量化点。
- **垂直分层**：多轨按 State/RTPC 增减层（战斗强度）。
- **水平重排**：按玩法分支切换段。
- **片段式交互流（对齐 Godot AudioStreamInteractive）**：以“剪辑图”描述交互音乐——节点为剪辑，边为带触发条件与**过渡类型**（immediate / next-beat / next-bar / segment-end / marker）的转移，采样精确切换，可挂过渡剪辑与淡入淡出；与上面的 Segment/播放列表模型互补，覆盖从“轻量剪辑跳转”到“专业段/垂直分层”的完整谱系。
- **时间线编排（对齐 Unity Timeline 音频轨）**：离线在时间线上按样本位置编排剪辑/stinger/事件，经 §8 调度器采样精确回放，用于过场与脚本化演出。

---

## 20. 资产、Bank 与流式媒体

规划（对齐 Wwise SoundBank / FMOD Bank）：

- **Bank 打包**：把一组 Event/Container/Patch/媒体打包为可加载/卸载单元，按关卡或情景加载，控制内存占用。经 `bevy_asset` 加载与热重载。
- **内存 vs 流式媒体**：短音效常驻内存池；长音乐/环境走**流式**（磁盘→解码任务→环形预取缓冲→RT 只读），欠载输出静音不阻塞。
- **预取（Prefetch）**：流式声的首段常驻内存，保证零延迟起播，其余边播边取。
- **内存池**：解码缓冲、语音状态、延迟线来自构造期分配的池，RT 线程零 malloc。Bank 卸载在任务线程回收（epoch，§21）。
- **解码任务**：`bevy_tasks` 后台解码，格式插件化（`SourceDecoder` trait：wav/ogg/flac/自定义）。编解码矩阵（PCM/ADPCM/Vorbis/Opus/FLAC）与按用途分档见 §44.1。

---

## 21. 无锁 ECS 集成层与线程/内存模型

规划：

- **线程模型**：
  - **游戏/ECS 线程**：产生 Event 与命令，写入命令环；从遥测环读回状态。
  - **音频回调线程（RT）**：唯一执行 `AudioGraph::process` 的线程，零分配/锁/panic；每块吸收命令、渲染、写遥测。
  - **任务线程（bevy_tasks）**：解码、Bank 加载、图/Patch 编译、烘焙；产出的编译图/缓冲经命令环交付。
- **命令环（Command Ring）**：ECS/gameplay → 音频线程的 MPSC 无锁环。命令包括 play/stop/set-param/set-transform/set-switch/set-state/swap-graph。RT 线程每块开始批量吸收，带样本偏移的命令交 §8 调度器。
- **遥测环（Telemetry Ring）**：音频线程 → ECS 的 SPSC 环，回传峰值/RMS/LUFS/语音数/事件完成回调/Profiler 帧，用于 UI 表头、gameplay 反馈与 §26 剖析。
- **图/资源交换与回收**：编译好的新图/新缓冲经命令环原子交付，块边界切换指针；旧对象进 **epoch 回收队列**，确认无 RT 引用后在任务线程 drop（RT 线程从不 drop 分配）。
- **零锁**：主线程与音频线程只经环通信，无互斥锁，无优先级反转。

前端 `bevy_audio` 保留 `AudioPlayer`/`PlaybackSettings`/`Volume` API 兼容，内部翻译为命令。

### 21.1 运行时桥落地契约（`prism_audio_rt`，已实现）

以下为 `prism_audio_rt` 的具体实现契约，是上述线程/内存模型的可运行落地，与 AAA 实践对齐并逐条闭环验证：

- **RT 唯一入口 `process_block`**：单一执行序为 `swap_graph → drain_commands → render → apply_master_gain（逐样本）→ voices.advance → build_telemetry → push_telemetry`。全链路**零分配/零锁/零 panic**（禁 `unwrap/expect/vec!`），`cpu_load` 用 `Instant` 测量并随遥测回传，对齐 Wwise/FMOD 音频回调的硬实时约束。
- **命令粒度分层**（借鉴 UE5 Quartz 的"块量化 vs 采样精确"分工）：语音生命周期与池容量类命令在**块起点批量应用**（块粒度足够且避免每样本分支开销）；只有母线增益 `SetMasterGain{at_frame, ramp_frames}` 被**逐样本兑现**为采样精确 ramp（`Ramp::Immediate` 或 `Ramp::Linear`）；需要采样精确的音乐/事件时刻交给 §8 `EventScheduler` 的最小堆，避免命令枚举堆积语义无用的时间字段。
- **图交换 = capacity-1 最新胜**（`GraphHandoff`）：任务线程 `publish_graph` 时，若上一张待接图未被 RT 取走则被挤下并**退回任务线程**处理；RT 在块起点 `take` 最新图并原子换指针，**被换下的旧图绝不在 RT 侧 drop**，而是投入 epoch 回收队列。对齐"渲染图双缓冲 + 编译产物热切换"实践。
- **epoch 回收带背压**（`RetireQueue`）：RT 侧 `retire(Box<dyn Any + Send>)` 有界推入；队满时**返回资源给 RT 保留**（下一轮再试），绝不阻塞或 drop-in-RT；收集线程 `collect()` 在 gameplay/任务线程 drain 并析构。这实现了 §21"RT 从不 drop 分配"的铁律。
- **通道形态**：命令环为 MPSC 就绪（生产端 `Clone` 共享 `Arc<ArrayQueue>`，多 ECS/gameplay 系统可并发投递）；遥测环为 SPSC（RT 单产、UI 单耗）。有界、满则 `push` 返回 `Err(item)` 不阻塞不分配。
- **零 unsafe**：本 crate 不含任何 `unsafe`，无锁底座复用经审计的 `crossbeam-queue::ArrayQueue`，把无锁正确性下沉到成熟依赖，符合 §28 可信度基础设施取向。


---

## 22. 设备后端、离线渲染与输入捕获

crate `prism_audio_device`（✅ M3 输出/离线/捕获三条链已落地并单测通过；worklet/触感/远程授权后续档）：

- **cpal 后端**（✅ 已落地，`cpal_backend.rs`）：桌面/移动原生输出。以 `feature = "cpal-backend"`（默认开）门控，离线/CI/无声卡主机可 `--no-default-features` 不链平台 SDK。`open_default_output` 取默认 host/device，**按 `AudioRuntime` 采样率建流；设备无法满足该率时上报 `BuildStream` 错误而非静默重采样**（避免隐式质量损失），设备声道数经 `layout_for_channels` 映射到引擎 `ChannelLayout`（1→Mono/2→Stereo/4→Quad/6→5.1/8→7.1）。数据回调经 `SampleFormat` 分派到泛型 `build_typed::<T>`（覆盖 F32/F64/I16/U16/I32/I8/U8），用开流时一次性预分配的 scratch 在引擎固定块与主机可变缓冲间搬运、逐样本 `from_sample` 转格式，**回调路径零分配/零锁**。
- **`BlockRenderer` 拉取适配器**（✅ `render.rs`）：把 `AudioRuntime` 产出的固定引擎块喂进任意大小的交错缓冲，稳态零分配；设备与离线**共用同一渲染路径**，因此离线 WAV 与实时播放对同一图/命令流逐样本一致。
- **FileSink（离线）**（✅ `file_sink.rs`）：`render_to_wav` 以任意块大小确定性离线渲染到 32-bit float WAV，用于 golden 对拍与过场预渲染。
- **输入捕获（麦克风 / 回读）**（✅ `capture.rs`，对齐 Godot Microphone / AudioEffectCapture）：`CaptureSink::push_interleaved` 从设备回调把交错输入以整帧写入无锁环形缓冲，`CaptureConsumer` 在非 RT 侧取平面帧，供录制、语音（§45）、频谱 UI（§26）；缓冲溢出按整帧丢弃并计数，不阻塞回调。
- **AudioWorklet 后端**（规划）：Web 平台（wasm），在 worklet 线程跑图。
- **设备韧性与故障隔离**：
  - **欠载（xrun/dropout）识别与隐藏**：设备回调取不到整块时，由平台错误回调 + 环形缓冲空读计数（`CpalOutput::error_count` 及饥饿计数）判定欠载，输出侧不吐垃圾/旧数据——短欠载对上一块做快速淡出并保持静音、恢复供给后淡入（PLC 式隐藏，避免爆裂），欠载事件与时长经遥测环（§26）上报供 Profiler 定位；回调始终非阻塞返回。
  - **默认设备变更与热插拔跟随**：插拔耳机、拔出声卡、系统默认设备切换时，平台事件在**非 RT 侧**触发重开流——按新设备采样率/声道重建 `BlockRenderer` 与 `layout_for_channels` 映射，从命令环/遥测的当前状态无缝续跑，重开窗口期输出静音，绝不在 RT 线程做设备枚举或分配。设备不可用时优雅降级到 null/离线 sink 持续推进图（保持确定性与播放头连续）。
  - **引擎时钟 vs 设备时钟漂移**：开流时采样率不匹配仍**报错不静默重采样**（避免隐式质量损失）；但对**长期运行的时钟漂移**（设备晶振与引擎标称率的 ppm 级偏差、多设备/回环协同）提供可选的**有界异步重采样档**（`DriftResampler`，多相 FIR，§34 质量分级），由缓冲填充水位闭环微调比率。这与“开流即拒绝不匹配率”是两个正交机制：后者拒绝配置错误、前者纠正稳态漂移。漂移校正默认关闭，仅在检测到持续水位偏移时启用，校正量逐块平滑无爆音。

---

## 23. 无障碍（Accessibility）

规划（对齐 `bevy_a11y`）：

- **字幕/描述**：Event 可携带字幕元数据，触发时经遥测环上报给 UI 层。
- **单声道下混**：单耳听力用户一键下混。
- **语音优先/闪避增强**：对白优先级最高，可强化 ducking（§13）保证可懂度。
- **视觉声音提示**：关键音效可触发视觉指示（方向/类型）。
- **动态范围压缩档**：夜间/听障档启用更强压缩（§13 HDR 窗口收窄）。

---

## 24. 确定性与网络

- **种子 RNG**：Container 随机、Scatter 散布、Patch 随机源均用可注入种子 RNG，回放/测试可复现。
- **确定性数学**：全链路 `bevy_math::ops`（libm），跨平台位一致。
- **网络模型**：音频是本地表现层，由确定性游戏事件触发，不同步音频样本；避免网络抖动进入 RT 路径。
- **固定块**：离线与测试用固定块大小，产出逐样本可对拍的 golden。

---

## 25. 语音管理与虚拟化

规划（对齐 Wwise Virtual Voices / Playback Limit）：

- **语音池**：预分配固定数量语音，避免 RT 分配。
- **优先级 + 响度淘汰**：超限时按 (优先级, 估计响度, 距离) 淘汰最不重要者。
- **虚拟语音行为**（对齐 Wwise Virtual Voice Behavior）：被淘汰/超阈值的语音可配置行为——
  - `ContinueVirtual`（继续推进播放头不出声，资源空出可复活，保证循环声位置连续）
  - `Kill`（直接停止释放）
  - `RestartFromBeginning`（复活时从头播）
  - `PlayFromElapsedTime`（复活时从应到达位置播）
- **限量（Playback Limit）**：同类声源实例上限（如最多 8 个同时脚步），超限按策略拒绝或替换最旧/最弱者，对齐 Wwise Playback Limit。
- **进入/离开阈值**：以估计响度的进出阈值（带滞回）决定何时虚拟化，避免边界抖动。
- **多voice复用声源（对齐 Godot AudioStreamPolyphonic）**：单个逻辑声源可动态复用池内多条 voice 播放重叠实例（脚步/连发/碰撞），免去为每次触发手动新建声源；复用与上限受同一语音池与 Playback Limit 约束。

---

## 26. 剖析、遥测与可视化调试

规划（对齐 Wwise Profiler / Godot SpectrumAnalyzer / Unity Audio Profiler）：

- **实时捕获**：遥测环（§21）导出每块的语音清单、总线电平、CPU 占用、事件时间线，供 UI/Profiler 面板回放（可录制会话）。
- **计量（Meters）**：`MeterNode` 在任意总线插入，回读峰值/RMS/LUFS/相位/相关度。
- **频谱分析**：`SpectrumNode`（FFT）输出频带能量，供 gameplay 反应（音乐可视化、节拍触发）与调试。
- **语音监视**：列出活动/虚拟语音及其优先级/响度/衰减状态，定位"为什么这个声音不响"。
- **图检视**：导出当前编译图（节点/连接/延迟）为可视化，配合 `bevy_diagnostic` 面板。
- **golden 差异**：离线渲染与参考波形的逐样本 diff 可视化，回归定位。
- **引擎内实时检视（对齐 UE5 Audio Insights）**：无需外部工具即可在编辑器/运行时面板检视活动声源、总线电平、参数/RTPC 当前值、虚拟化状态与每块 CPU 占用，数据源自遥测环（§21），只读、不占 RT。

---

## 27. 性能预算与验收

| 指标 | 目标 |
|---|---|
| RT 回调分配 | 0（热路径零 malloc/lock） |
| 512 语音 @48k | 单核 < 30% 预算（桌面档） |
| 图/Patch 编译 | 线程外，< 数 ms（百节点级） |
| 延迟 | 设备块 + 前瞻，可配置（默认 ~10ms 桌面） |
| LUFS 归一误差 | < 0.5 LU |
| True-peak | ≤ -1 dBTP（母带后） |
| 多普勒/衰减更新 | 逐块平滑，无 zipper/爆音 |
| golden 对拍 | 逐样本 ULP 级一致（固定种子/块） |

---

## 28. 质量与可信度基础设施

- **单元测试**：每节点带 impulse/稳定性/golden 测试（已有：graph 求和/send/环拒绝/层校验、biquad 频响/稳定、pan 功率守恒、param 收敛，共 31 项通过：27 单测 + 4 doctest）。
- **no_std + std 双构建**：内核与节点在 `--no-default-features` 下亦编译（libm 后端），保证嵌入式/wasm 可移植。
- **clippy 零告警**：遵守工作区严格 lints（`missing_docs`、`disallowed-methods` 确定性数学、`allow_attributes_without_reason` 等）。
- **离线 golden 渲染**：FileSink 渲染参考波形，回归对拍。
- **provenance**：无 UE/Unity/Godot/Wwise/FMOD 源码或衍生代码，所有 DSP 出自公开标准知识（RBJ cookbook、BS.1770、等功率声像、AmbiX/ACN-SN3D、FDN 等）。
- **属性/模糊测试**：对节点输入（NaN/inf/denormal、极端块长、参数边界）做属性与模糊测试，验证 RT 契约（不 panic、输出有界、denormal 被 flush），发布构建保持 panic-free（错误经 `Result` 上抛而非 unwrap）。
- **可验证 RT 安全（不止于声明）**：把“零分配/无锁/不 panic”从口头契约变成**可自动验证**的机器约束——
  - **无分配守卫**：测试与 debug 构建下用 feature 门控的全局分配器包裹（`#[cfg(feature = "rt-alloc-guard")]`），进入 `process`/`process_block` 期间任何堆分配立即 panic 并定位调用栈；CI 对全部节点与整图跑一遍稳态渲染，确保热路径真的零 malloc（含第三方节点）。
  - **无锁/无阻塞检查**：RT 路径禁用 `std::sync::Mutex`/阻塞 IO（`disallowed-methods` lint + 审查），命令环/遥测环/回收队列均为有界无锁结构（§21），压测其在满/空/竞争下不阻塞、不丢确定性。
  - **编译器与无锁环模糊目标**：对图编译器（§29 拓扑排序/缓冲活跃度分配/别名校验）与 §21 三条无锁环建 `cargo-fuzz` 目标——随机图（含非法环、宽扇入扇出、深嵌套 Patch）不得 panic、必须产出合法 `ExecPlan` 或明确 `Cycle`/容量错误；随机跨线程命令/图交换序列不得违反交付/回收不变量。
  - **CI 门禁矩阵**：默认 std + `--no-default-features`(no_std/libm) 双构建 × clippy 零告警 × golden 逐样本对拍 × 无分配守卫 × sanitizer（ASan/TSan 验证无数据竞争）在提交门禁执行，构成“可信度”的量化验收面（§27）。

---

## 29. 编译图执行模型（拓扑计划 / 缓冲活跃度分配 / 就地与别名优化）

图在音频线程之外编译成一份**确定性执行计划（ExecPlan）**——借鉴 Web Audio 的图模型与现代渲染图（render graph）的"编译期资源分配"思想，把"图是什么"与"如何跑"彻底分离：

- **拓扑排序计划**：编译期做 DAG 拓扑排序，产出定序的 `Vec<PlanStep>`（每步 = 节点索引 + 输入缓冲槽 + 输出缓冲槽 + 增益/求和指令）。RT 线程只顺序执行这份扁平计划，无递归、无 `HashMap` 查找、无虚分发以外的间接。
- **缓冲活跃度分配（Buffer Liveness Allocation）**：把中间缓冲当作"寄存器"，用**活跃区间分析 + 图着色**为每条边分配缓冲槽，生命周期不重叠的边复用同一物理缓冲（类似编译器寄存器分配 / 渲染图的 transient 资源别名）。相较"每条边一块缓冲"，峰值内存与 cache 足迹显著下降；分配在编译期完成，RT 期零分配。
- **就地处理（In-place）与别名优化**：当某节点的某输出端仅被单一消费者读取、且节点声明 `can_process_in_place()`，编译器让其输入/输出复用同一缓冲槽，省去一次 copy。别名安全性在编译期校验（禁止把仍被其他步读取的缓冲就地覆写）。
- **求和与发送内联**：同一输入端的多入边求和、以及 Aux 发送的加权累加，被编译成计划内的 `AddScaled` 指令序列（复用 `AudioBuffer::add_scaled`），而非运行时遍历边表。
- **延迟对齐（PDC）**：编译期沿计划累计各节点 `latency_frames()`，对并行路径插入整数样本延迟补偿（Plugin Delay Compensation），保证多路汇合相位对齐——这是母带链与并行发送正确性的前提。
- **热交换**：新计划在任务线程编译完成后经命令环交付，RT 在块边界原子切换计划指针；旧计划连同其缓冲池进 epoch 回收（§21 无锁模型的回收策略），RT 线程从不 drop。

`ExecPlan` 是纯数据（`Copy`/`Send` 的索引与指令），可序列化用于离线 golden 对拍：相同图编译出**逐字节一致**的计划，是确定性验收（§28）的基石。

---

## 30. 并行 DSP 图调度（Job 化渲染 / 确定性并行 / 岛屿划分）

单线程 RT 图在语音密集（数百并发语音 + 多总线 + 卷积混响）场景会成为瓶颈。借鉴 **Unity DSP Graph 的数据导向无 GC 执行**并结合 Prism 已有的 `bevy_tasks` 工作窃取，Resonance 支持**块内并行**的图执行，同时保持确定性：

- **岛屿划分（Island Partitioning）**：编译期把 DAG 按连通性与拓扑层级切成可并行的"岛屿/层"。同一拓扑层、无数据依赖的节点可并发执行；跨层用屏障（barrier）同步。
- **Job 化渲染**：每个可并行节点或子链封装为一个 job，提交到**固定规模的音频 worker 池**（独立于 gameplay 任务池，绑定优先级，避免与渲染/物理抢占）。worker 数按设备核数与 CPU 预算（§32）配置，可降到 1 = 纯串行回退。
- **确定性并行**：并行**不改变数值结果**——求和顺序在编译期固定（计划里的 `AddScaled` 定序），浮点累加按固定次序执行；job 只并行"互不依赖"的节点，不并行"同一求和的多个加数"。因此并行/串行两条路径产出**逐样本一致**，可交叉对拍验证。
- **RT 安全的 job 系统**：worker 不分配、不加锁竞争（每 worker 私有 scratch 缓冲，来自构造期预分配的池），完成计数用原子；主 RT 线程作为协调者提交并等待层屏障，欠时（overrun）时记录并降级（§32）。
- **NUMA/亲和**：缓冲槽分配（§29）尽量让同岛屿的读写落在同一 worker 的私有缓冲，减少跨核共享脏行。
- **回退保证**：若平台不支持多线程音频（wasm 单线程 worklet），编译器输出串行计划，行为等价。

并行是**编译期决策 + RT 期执行**，与 §29 的 ExecPlan 同源：并行计划只是给每个 PlanStep 附加"岛屿 id + 层号"。

---

## 31. GPU 加速几何声学（共享渲染器 BVH 的光线与路径追踪）

这是 Resonance 相对独立音频中间件（Wwise/FMOD/Steam Audio 各自维护声学场景）的**核心次世代差异**：Prism 既是渲染器又是音频引擎，声学传播可**直接复用渲染器的 GPU 场景表示与加速结构**，而非重建一套声学几何。

- **共享加速结构**：遮挡/衍射/反射查询复用 `prism_physics` 与渲染管线已有的 **BVH/meshlet 场景**（同一 `pkg/prism_render_*` 世界表示），声学与视觉/物理**共享同一几何真相**，杜绝"碰撞体与声学体不一致"的经典 bug。
- **GPU 声线/路径追踪（可插拔 `PropagationBackend` 的 GPU 实现）**：
  - **反射**：在 GPU 上从声源发射声线，蒙特卡洛追踪镜面 + 漫散射反射，累积能量到听者，估计**早反射 + 后期混响衰减（RT60/EDC）**，输出给 §14 的反射/混响参数与 FDN/卷积混响的房间响应。
  - **遮挡/障碍**：批量射线（声源→听者、声源→探针）在 GPU 上并行求交，返回直达可见性与穿越材质，驱动 §14 的直达增益/低通与 aux 发送。
  - **衍射**：沿几何边缘的最短路径（GPU 上的路径查询/边缘图），给出绕射角与附加衰减。
- **异步、无 RT 阻塞**：声学 GPU 作业在**渲染帧率节奏**（非音频块节奏）异步派发，结果（衰减/低通/发送量/房间响应系数）经遥测/命令环喂给音频线程，RT 侧只做**平滑插值**（§7）——GPU 延迟对听感透明。
- **烘焙与实时混合**：静态几何离线烘焙探针网格（GPU 加速预计算），运行时对动态遮挡做**少量实时声线**修正，二者插值。烘焙数据经 `bevy_asset` 加载。
- **声学材质协同**：表面从 `prism_material_pipeline` 同一资产读取吸收/散射/透射系数（与视觉 BRDF 并存于一个材质），GPU 追踪时按频带加权。
- **优雅降级**：无合适 GPU 或预算不足时，退回 `prism_physics` CPU 射线后端（§14 默认几何后端），接口不变。

安全边界：GPU 声学**只产出控制参数**（标量/低维系数），不产出音频样本流回 RT，避免 GPU→音频的实时同步风险；样本级 DSP（卷积、FDN、双耳）始终在 CPU 音频线程确定性执行。

---

## 32. 性能自适应治理与音频 LOD（CPU 预算驱动质量缩放）

对标主机/移动"稳帧"诉求，Resonance 内建**运行时质量治理器（QualityGovernor）**，在 CPU 预算内动态缩放质量，避免爆音/欠载——借鉴 Wwise 的语音上限/虚拟语音与游戏引擎的 LOD 思想，但做成**闭环自适应**：

- **CPU 预算闭环**：遥测环（§21）回传每块的渲染耗时占预算比。治理器据此在**块边界**（非 RT 热路径内决策）升/降质量档，滞回（hysteresis）避免抖动。
- **音频 LOD 维度**（按声源优先级 + 距离 + 感知重要度分级）：
  - **过采样档**：波形整形/非线性效果（§9 waveshaper）的抗混叠过采样从 4x→2x→1x 随预算下调。
  - **混响质量**：卷积分块 FFT 尺寸、FDN 延迟线数、早反射条数分级；远处/次要声源用更廉价的混响或共享混响返回。
  - **空间化档**：近处/重要声源用 HRTF 双耳卷积，远处降为廉价 VBAP/立体声声像；HOA 阶数按预算降阶。
  - **调制/自动化速率**：次要声源的调制与自动化从逐样本降为逐块控制率。
  - **语音数与虚拟化**：超预算时提高虚拟化阈值（§25），把更多低感知贡献语音转虚拟。
- **优先级模型**：每语音的有效重要度 = 显式优先级 × 距离衰减 × 感知响度 ×（是否被掩蔽，§33）。治理器优先降级低重要度语音。
- **平台档位**：移动端可叠加全局降采样率/降语音上限的功耗档（§32 呼应移动功耗开放问题）。
- **可观测**：当前档位、降级原因、各维度节省经 Profiler（§26）可视化，便于声音设计师与工程师调优。

治理器**只调参数、不改图拓扑**（拓扑热交换代价高、留给显式重编译），保证平滑无爆音。

---

## 33. 心理声学虚拟化与声源聚类（掩蔽感知剔除 / 对象床限制）

在"数百声源、有限扬声器/对象床/CPU"的次世代场景下，纯"最响的 N 个"淘汰会漏掉感知细节。Resonance 引入**心理声学感知层**（纯经典信号处理，无 ML），把有限资源花在**听得见**的声音上：

- **掩蔽感知虚拟化（Masking-aware Virtualization）**：在响度基础上估计**频域掩蔽**——用简化的临界频带（Bark/ERB 近似）能量比较，判断某语音是否被更响的邻近语音在其频带内掩蔽。被掩蔽且低贡献的语音**优先转虚拟**（§25：继续推进播放位置但不渲染），资源释放给可闻语音。掩蔽阈值随治理器（§32）预算收紧/放宽。
- **HDR 联动**：与 §13 HDR 窗口共享响度估计——落在动态窗口下沿之外的语音直接虚拟化。
- **声源聚类（Source Clustering）**：借鉴对象音频床（Atmos/平台后端）的对象数上限与 Wwise 多位置思想，把空间上邻近、音色相近的多个语音**动态聚类**为少量"代表性虚拟源"：
  - 聚类质心 = 成员按响度加权的空间位置；聚类信号 = 成员下混（能量守恒）。
  - 用于"人群/雨/弹雨/森林"等海量点声源，把 N 个空间化开销降为 K 个（K ≪ N），听感上保持空间分布。
  - 聚类在**块边界**重算（增量维护），成员进出用 §7 平滑淡入淡出避免爆音。
- **对象床预算**：输出到平台对象床（§16）时，若对象数超平台上限，用同一聚类机制归并到床容量内，其余下混到环境声道。
- **确定性**：掩蔽与聚类判定基于确定性能量/位置输入，可回放、可 golden 对拍（判定日志经遥测导出）。

感知层**只决定"谁渲染/如何归并"**，被判定的语音状态推进不变，恢复可闻时无缝复活（§25 虚拟语音行为）。

---

## 34. 时间伸缩与变调 / 重采样质量分级（与多普勒解耦）

音高、时值、播放速率与多普勒是四个**可独立控制**的量；朴素引擎把它们耦合在"改变采样读取步进"上，导致变速必变调、多普勒有爆音。Resonance 提供解耦的高质量原语：

- **重采样质量分级（`Resampler` trait）**：
  - 设备/资产采样率转换与任意比率播放走可插拔重采样器，质量分级：**线性**（LOD 低档/远处）→ **多相 FIR / windowed-sinc**（默认）→ **高阶 sinc**（母带/近场）。
  - 系数构造期预计算，RT 期零分配；比率随多普勒/变速连续变化时相位连续（无爆音）。
- **变调不变速 / 变速不变调（`TimeStretcher` trait）**：
  - 提供 **WSOLA/SOLA**（低开销、语音/音效友好）与**相位声码器**（音乐、大幅拉伸更平滑）两档实现，供交互音乐（§19）做节拍匹配、音效做音高随机化而不改时长。
  - 与 §11 Patch 的采样器原语集成：采样器可指定"保持音高的变速"或"保持时长的变调"。
- **多普勒解耦**：§15 的 `DopplerNode` 用**分数延迟线**（连续变长）实现物理多普勒频移，与"内容侧变调/变速"分开——多普勒是传播效应，变调是创作意图，二者可叠加而互不污染。
- **确定性**：所有重采样/伸缩系数与相位状态由 `bevy_math::ops`/`libm` 确定性计算，构造期预分配，逐样本可回放。

三条原语（重采样 / 时间伸缩 / 多普勒延迟线）共享"分数延迟 + 插值"内核，代码复用且各有独立质量档，随治理器（§32）自适应。

---

## 35. 对白与本地化（Dialogue / Localization / 字幕 / 口型）

规划 crate `prism_audio_authoring`（对齐 FMOD Programmer Instrument / Wwise External Sources + Dialogue Event / 通用本地化管线）：

- **程序化对白（运行时选择媒体）**：对白不预烘进逻辑，而由**对白解析器**在运行时按键值（角色 / 情绪 / 语言 / 变体 id）选择媒体后注入语音（对齐 FMOD Programmer Sound 与 Wwise External Sources）。游戏只发"说这句台词"的语义 Event，解析→取流→播放全在链路下游完成。
- **对白决策树（Dialogue Event）**：Wwise 式——按一组 State/Switch（谁在说、在哪、什么心情）沿决策树命中具体台词或随机变体，支持"通用回退"路径，避免缺变体时静默。
- **语言 Bank 热切换**：本地化媒体按语言分包（`voice_en`/`voice_zh`/…），切语言只换语音 Bank，逻辑与时间线不变；未加载语言回退默认并遥测告警。
- **字幕/说明文字同步（Caption Sync）**：媒体携带时间码字幕轨（或外部字幕资产），播放头驱动字幕事件经遥测环（§21）上抛给 UI，支持逐句/逐词高亮与无障碍全字幕（联动 §23）。
- **口型/表情驱动（Viseme / Lip-sync）**：viseme/音素时间线与能量包络随资产附带或离线分析导出，运行时经遥测环喂给动画系统，**不占 RT 预算**；无预烘数据时用 §26 `SpectrumNode`/包络跟随做粗略张口降级。
- **对白优先级与闪避**：对白总线作为 §13 HDR/ducking 的 sidechain 键，说话自动压低音乐/环境；对白语音高优先级，§25 虚拟化最后淘汰。
- **RT 边界**：解析/查表/取流在任务线程完成，RT 只播已就绪语音；缺失媒体输出静音并告警，不阻塞、不 panic。

---

## 36. 触感与跨模态输出（Haptics / Motion / 手柄反馈）

规划（归入 `prism_audio_device` 输出适配层，对齐 Wwise Motion / PS5 DualSense·Tempest Haptics / 通用手柄 rumble）：

- **音频同源触感**：把音频信号（或其低频/包络）转成触感波形——声音与触感**同源**，天然逐样本同步（对齐 Wwise Motion 把声音渲染到"运动设备"），免去另做一套振动曲线。
- **触感总线（Haptic Bus）**：统一图（§5）里一条并行输出总线，声源可像 Aux 发送（§17）一样按增益发往触感总线；总线走独立带通/整流/包络链后交 `HapticBackend`。
- **`HapticBackend`（可插拔 trait）**：
  - **宽频高保真**（DualSense / Tempest 式）：直接吃触感波形（低采样率重采样），表达细腻纹理。
  - **双马达 rumble**（通用手柄）：信号分低/高频包络分别驱动左右（低/高频）马达。
  - **无设备**：静默回退。
- **跨模态对齐**：触感与声音共享 §8 播放头与 §29 PDC，保证"看到—听到—摸到"同一样本时刻对齐；设备固有延迟由后端上报并补偿。
- **空间触感**：按声源方向/距离（§15）加权左右强度，做方向性冲击（左侧爆炸→左马达更强）。
- **预算与降级**：触感受 §32 治理器预算约束，低档旁路；触感生成**纯旁路**，不回灌音频路径。

---

## 37. 程序化环境音景（Soundscape / 程序化 Ambience）

规划（对齐 UE5 Soundscape / 通用程序化环境系统）：

- **音景状态（Soundscape State）**：由环境上下文（生物群系 / 天气 / 时段 / 室内外，来自 gameplay 与 §17 Reverb Zone）激活一组**调色板（Palette）**。
- **调色板 / 元素（Palette / Color Point）**：每个元素定义一个环境声（鸟鸣/风/滴水/远处交通）及其**播放规则**——触发概率、间隔分布、随机音高/增益、空间散布半径、并发上限、昼夜权重。
- **程序化调度**：调度器复用 §8 时钟 + 种子 RNG，在听者周围**程序化散布 one-shot**，形成永不循环、低重复感的环境床，替代"一段环境 loop 干听"。
- **几何/遮挡联动**：散布点经 §14 遮挡/障碍与 §17 房间归属过滤（室内不放室外鸟鸣），发送量随 §15 距离曲线。
- **确定性**：调度用种子 RNG（§24），同种子 + 同状态序列可复现，便于 golden 对拍与网络一致。
- **预算内自适应**：并发环境元素数受 §32/§33 治理（聚类/掩蔽剔除），远处密集元素聚合为床。

---

## 38. 实时授权与远程工具 API（Live Authoring / WAAPI 式 / 热调）

规划（对齐 Wwise Authoring API (WAAPI) / FMOD Studio Live Update / 通用远程调试）：

- **远程工具通道（`AuthoringTransport`）**：编辑器/外部工具经本地 socket（或进程内通道）连到运行引擎，**只读遥测**（语音清单/总线电平/CPU/事件时间线，来自 §26 遥测环）+ **白名单写命令**（改参数/触发 Event/切 State/换 snapshot），命令经 §21 命令环下发，绝不直接触碰 RT 内存。
- **实时调参（Live Tuning）**：运行时调 RTPC/总线增益/衰减曲线/混响参数并**即时听到**（对齐 FMOD Live Update），满意后回写授权资产；改动经 §7 平滑，无爆音。
- **授权数据热重载**：Event/Container/Bank/Patch 经 `bevy_asset` 热重载——任务线程重编译受影响子图/Patch 成新 `ExecPlan`（§29），RT 块边界原子热交换（§30），无需重启。
- **远程 Profiler 连接**：Profiler 面板（§26）可连本地或远端设备（主机/移动真机）会话，录制/回放；能力探测决定带宽与采样率。
- **安全边界**：远程通道鉴权 + 命令白名单，默认仅开发构建启用，发布构建整体编译剔除，收敛攻击面。
- **契约**：远程写与游戏代码走同一命令环，故**远程改动与代码改动语义一致、可确定性回放**。

---

## 39. Crate 拆分与落地形态

| Crate | 层 | 内容 | 状态 |
|---|---|---|---|
| `pkg/prism_audio_core` | L1+L2 | math/buffer/param/time/graph + nodes | ✅ L1 内核 + L2 effects（含 stereo_width）/dynamics/reverb 全族 + crossover（LR4 分频）+ multiband（LR4 多频带压缩）+ transient_shaper（差分包络瞬态整形）+ de_esser（分频去齿音）+ tremolo（颤音/自动声像）+ bitcrusher（位深量化/降采样 lo-fi）+ ring_modulator（双极载波环形调制）+ comb_resonator（调谐反馈梳谐振）+ vibrato（单 LFO 扫频分数延迟颤音）+ svf（Cytomic TPT 状态变量滤波，9 种响应）+ auto_wah（包络滤波自动哇音，振幅驱动扫频 Svf）+ exciter（谐波激励器，两级高通 Svf 夹 tanh 整形生成高次谐波）+ tape（磁带机模拟，软饱和 + wow/flutter 调制延迟 + HF rolloff）已落地（327 单测 + 11 doctest）；M2 sources/scheduler 进行中 |
| `pkg/prism_audio_spatial` | L3 | 几何传播/HRTF/Ambisonics/panner/多普勒/平台后端桥 | 🚧 M4 进行中：`geometry` 基座 + `attenuation`/`cone`/`doppler`/`air`/`panner`/`ambisonics` 6 模块 + `occlusion`（遮挡/障碍：可插拔 `OcclusionQuery`）+ `spatializer`（per-source 控制率编排：`SourceDescriptor`/`resolve`->`SpatialParams`）+ `spread`（扩散/聚焦：多点虚拟子源扇形 + focus 升余弦窗 + 距离曲线，已接入 spatializer）+ `multi_position`（多位置声源：一逻辑源映射多点，`resolve_multi` nearest/blend/envelop 三模式）+ `propagation`（几何声学传播后端：可插拔 `trait PropagationBackend` + `FreeFieldBackend`，Maekawa 屏障衍射/Fresnel 数/`transmission_gain` 透射/`edge_path_difference` 绕边程差/`AcousticMaterial` 材质）+ `rooms`（房间与门户：`RoomNetwork` 实现 `PropagationBackend`、`Room` AABB + wall 透射、`room_of` 最内层归属、`Portal` 矩形门户 `closest_point`/openness 混合 `transmission_gain`、`portal_coupling_gain`=透射×Fresnel-Kirchhoff 斜度因子、query 同房间单直达 vs 跨房间穿墙+门户次级路径）+ `reverb_zones`（混响分区与游戏定义辅助发送：`ReverbZone` Box/Sphere 区域绑定 `AuxBusId`、cubic smoothstep 过渡带、`ReverbZoneField::resolve` 由听者位置解析活跃 `AuxSend` 集合（同总线取最强、上限 `MAX_AUX_SENDS`）、`source_send_gain` 叠加每声源 `wet_gain`） + `hoa`（高阶 Ambisonics 编解码：SN3D/ACN 实球谐至三阶 16 通道，`fill_hoa_coeffs` 纯 Cartesian 递推（Chebyshev 方位 + 关联勒让德三递推 + `sn3d_norm` 阶乘比），零向量退化全向、`encode_hoa`/`decode_hoa`（解码除以 `order+1`）、RT `HoaEncoderNode` 逐样本平滑）+ `hoa_rotation`（HOA 声场旋转：实球谐分块对角旋转至三阶，一阶块 `R^1=A·Q·A` + 二/三阶 Ivanic–Ruedenberg 递推，`rotate_hoa`/`HoaRotationMatrix`（控制率算一次、多帧复用），golden `rotate_hoa(encode(d))==encode(q·d)` 对拍）+ `hoa_decode`（双频段能量优化解码：低频 basic/in-phase、高频 max-rE 加 `(2n+1)*g_n` 模态计数修正、`DualBandDecoder`/`max_re_gains`/`max_re_radius`）+ `hoa_beamform`（可转向虚拟传声器模态波束：`BeamPattern` basic/max-DI/max-rE/in-phase、`Beamformer::beam`）+ `nfc`（HOA 近场补偿：参考距离稳定化的逐阶反向 Bessel 极点 + 双线性变换 IIR 级联、DC 增益 `(r_ref/r_src)^m`、`NfcCoeffs::design`/`NfcFilter`）+ `early_reflections`（镜像源法早期反射：Allen-Berkley 1979 矩形房间整数索引镜像枚举，`compute_early_reflections` 产出延迟/增益/听者局部到达方向/order tap 列表（栈上零分配、溢出保最强、`MAX_EARLY_REFLECTIONS=32`/`MAX_REFLECTION_ORDER=4`）+ RT `EarlyReflectionRenderer` 单共享延迟线多抽头 `equal_power_pan` 渲染，`ShoeboxRoom`/`ReflectionTap`）+ `portal_graph`（多跳门户路由）+ `room_acoustics`（统计混响 RT60/Schroeder/临界距离）+ `material_library`（建筑声学吸声材料库：八倍频程系数 + log 插值 + 具名材料）+ `octave_reverb`（倍频程混响时间：per-band Eyring RT60 谱 + FDN 衰减增益）+ `scattering`（表面散射/扩散：ISO 17497 散射系数谱 + Lambert 拆分）+ `diffusion_field`（晚期扩散场：diffuse 能量汇聚 + Schroeder 回声密度 + 混合时间/FDN 晚期增益） + `room_modes`（矩形房间驻波 eigenmode：Rayleigh 本征频率 + 轴/切/斜向分类 + 模态密度/Schroeder 过渡/倍频程染色）+ `source_directivity`（频变声源指向性：加权一阶 `(1-s)+s*cos` 每倒频程辐射图 + Q=1/((1-s)^2+s^2/3) 指向因子 + DI 指数 + Omni/Cardioid/Voice/Trumpet 预设）+ `reverberant_field`（稳态混响场：房间常数 R + 直达/混响能量 + DRR/临界距离）+ `diffraction`（Maekawa 屏障/边缘绕射插入损失：每倍频程 IL/增益）+ ground_effect（ISO 9613-2 地面效应 A_gr：三区地面因子 + Table 3 每倍频程衰减/增益，补齐户外传播地面项）+ `outdoor_propagation`（ISO 9613-2 户外传播总预算聚合 `A_div+A_atm+A_gr+A_bar` 每倍频程组合，各分量委托真源无重复实现）+ `reflection_directivity`（方向感知早期反射加权：源指向性 x 镜像抽头每倍频程加权组合）+ `reflection_clustering`（早期反射方向聚类：镜像抽头按到达方向归入 6 轴对齐方向簇、能量守恒聚合为紧凑方向送出）+ `convex_room`（凸多面体房间镜像源早期反射：任意凸房间一阶镜像反射，产出同 `ReflectionTap` 列表喂下游）（434 单测 + 36 doctest，clippy 零告警，no_std 双构建，已 commit）；HRTF 已拆至独立 crate `prism_audio_hrtf`；对象/Atmos 与平台后端桥规划 |
| `pkg/prism_audio_hrtf` | L3 | HRTF 双耳渲染：数据集/SOFA 装配/插值/分区卷积/近场 | ✅ M4 落地：`dataset`（`HrtfDataset` measurement-major 缓冲）+ `sofa`（`HrirSource` trait + `aes69_to_local` + `SofaRecords`→dataset；SOFA 二进制解码后续档）+ `interpolation`（ITD 群延迟对齐 + Shepard 反距加权 4 邻）+ `binaural`（`BinauralRenderer` overlap-save 分区卷积 + HRIR 交叉淡入零咔哒）+ `nearfield`（双耳视差 + 1/r 增益 + 球形头遮蔽近似）+ `headtracked`（头部追踪双耳：`HeadPose`/`predict` 四元数轴角指数外推 + `HeadTracker` slerp 平滑 + `predicted_local_angles` 喂既有插值重选 HRIR 方位）+ `hoa_binaural`（虚拟扬声器双耳：每 ACN 通道预烘焙 `filter[c]=Sigma_s D[s][c]*HRIR(s)`、运行时每通道一 `BinauralRenderer` 累加 L/R、`HoaBinauralDecoder`/`VirtualSpeakerLayout::cube26`）+ `transaural`（跨听/串音消除：递归 crosstalk 消除器精确逆对称串音矩阵、`CrosstalkCanceller`/`CrosstalkParams::from_geometry` 对称扬声器几何推导）（81 单测 + 4 doctest，clippy 零告警，no_std 双构建，已 commit）；SOFA 二进制解码后续档 |
| `pkg/prism_audio_authoring` | L3 | Event/Container/State/Switch/RTPC/Patch 编译/Modulation/交互音乐/Bank/对白与本地化/音景 | 规划 |
| `pkg/prism_audio_device` | L3 | cpal 输出/离线 FileSink/输入捕获（已落地）；worklet/触感后端/远程授权通道（规划） | ✅ M3 输出+离线+捕获落地（11 测试，cpal 特性门控，`--no-default-features` 离线构建通过）；余项规划 |
| `crates/bevy_audio` | L4 | ECS 前端（改接命令通道，保留兼容 API） | 规划改造 |

**并行开发拆分**（写集不相交，可 fan-out 给并行 agent）：
- effects（biquad 级联/delay/waveshaper/调制延迟）
- dynamics（compressor/limiter/gate/ducking/multiband/transient_shaper）
- reverb（FDN/convolver/早反射）
- spatial（panner/attenuation/cone/spread/doppler/HRTF/ambisonics）
- sources（sample player/oscillator/noise/streaming/generator）
- routing（bus/send/VCA/converter/transceiver）
- patch（内容子图编译器 + 合成原语）
- modulation（LFO/包络/控制总线/叠加）
- scheduler（采样精确调度器 + 命名时钟）
- 无锁环（command/telemetry ring、voice pool、epoch 回收）
- 剖析（meter/spectrum/capture + 面板）
- dialogue（对白解析/本地化/语言 Bank/字幕/viseme）
- haptics（触感总线/`HapticBackend`/双马达/宽频）
- soundscape（程序化音景调色板与散布调度）
- tooling（远程授权 API/live update/远程 profiler）

---

## 40. 路线图

- **M0 内核（已完成）**：math/buffer/param/time/graph + 首发 4 节点，31 测试（27 单测 + 4 doctest），双构建，零告警，已 commit。
- **M1 效果与动态（✅ 已完成）**：effects（parametric_eq/delay/waveshaper/chorus/flanger/phaser/stereo_width）+ dynamics（detector/compressor/limiter/gate/ducking）+ reverb（fdn/convolver/algorithmic 三族全落地）+ `crossover`（Linkwitz-Riley LR4 多频带分频，全通相位补偿保幅频平坦）+ `multiband`（LR4 多频带压缩器：每带独立 `CompressorNode` 求和重建）+ `transient_shaper`（差分包络瞬态整形器：fast/slow 双包络差分驱动无阈值 attack/sustain 塑形）+ `de_esser`（分频去齿音器：LR 分频高带 side-chain 压缩、SplitBand/Wideband 两模式）+ `tremolo`（颤音/自动声像：控制率 LFO 幅度调制、等功率自动声像、per-channel stereo_phase 交错） + `bitcrusher`（bit-crusher/decimator lo-fi：位深量化码钳补码范围 + sample-and-hold 降采样 + wet/dry）+ `ring_modulator`（环形调制：双极载波乘法产生和差边带金属/铃音，单载波跨通道共享 + wet/dry）+ `comb_resonator`（调谐反馈梳状谐振器：分数延迟反馈环 + 环内一极点低通阻尼，Karplus-Strong/Schroeder-Moorer 弦体共鸣音色）+ `vibrato`（单 LFO 扫频分数延迟颤音：调制延迟线纯 wet 音高调制，Zoelzer DAFX vibrato）+ `svf`（拓扑保持状态变量滤波：Cytomic/Zavalishin TPT 梯形积分，LowPass/HighPass/BandPass/Notch/Peak/AllPass/Bell/LowShelf/HighShelf 九种响应共享同一对积分器、快速扫截止稳定）+ `auto_wah`（包络滤波自动哇音：mono 侧链整流 attack/release 包络驱动 `Svf` 指数扫频，`WahMode{BandPass,LowPass,Peak}` × `SweepDirection{Up,Down}`，复用 dynamics `time_to_coef` 弹道，svf 铺路的振幅驱动调制滤波首成员）+ `exciter`（谐波激励器/听感增强：两级高通 `Svf` 隔离上频带、`tanh` 系整形器生成奇/偶/混合高次谐波再混回干声，Aphex Aural Exciter/DAFX 听感激励思想，生成式补充 `parametric_eq` 的减法重塑）+ `tape`（磁带机模拟：drive/bias `tanh` 软饱和 + wow/flutter 双 `Lfo` 调制分数延迟 + 一极点高频滚降三要素合一，DAFX 模拟磁带着色思想，`MAX_DELAY_MS=100` 钳深度防无界分配）。327 单测 + 11 doctest 全绿，clippy 零告警，no_std 双构建通过，已分多次 commit。
- **M2 声源与调度（✅ 已完成）**：`nodes/sources/`（oscillator 带限 PolyBLEP / noise white·pink·brown / sample_player 变调重采样+循环点）+ `scheduler.rs`（采样精确事件最小堆 `EventScheduler` + Quartz 式命名多时钟 `NamedClock` 量化）+ `voice.rs` 语音池（固定容量、优先级窃取、per-group Playback Limit 三策略、Wwise 式虚拟语音行为 `ContinueVirtual`/`Kill`/`RestartFromBeginning`/`PlayFromElapsedTime` + 迟滞进出阈值 revoice）。147 单测 + 5 doctest 全绿，clippy 零告警，no_std 双构建通过，已分三次 commit。streaming/解码依赖 std+file IO，归 M3 device 层，M2 不做假实现。
- **M3 ECS 桥与设备**：
  - **运行时桥（✅ 已落地）**：`prism_audio_rt` crate——有界无锁命令环（`AudioCommand`：SpawnVoice/StopVoice/SetVoiceImportance/SetMasterGain/SetMaxPhysicalVoices）+ 遥测环（`TelemetryFrame`：块序号/播放头/物理·虚拟语音数/主峰值·RMS/CPU 负载）+ **epoch 回收队列**（RT `retire` 推 `Box<dyn Any+Send>`，收集线程 drain·drop，队满时 RT 保留不 drop 形成背压）+ **capacity-1 最新胜图交换**（`GraphHandoff`，被挤下的旧图退回任务线程处理，绝不在 RT 侧 drop）。`process_block` 全链路零分配/锁/panic；语音命令块起点应用（块粒度），仅 `SetMasterGain` 的 `at_frame`+`ramp_frames` 逐样本兑现（采样精确增益 ramp），采样精确音乐事件调度交 §8 `EventScheduler`。真 std 多线程集成测试验证跨线程交付/图交换/回收/遥测。零 unsafe（底座复用 `crossbeam-queue` 的 `ArrayQueue`）。
  - **设备后端（✅ 已落地）**：`prism_audio_device` crate——`cpal_backend`（默认特性门控的原生输出，采样率不匹配即报错不静默重采样，F32/F64/I16/U16/I32/I8/U8 全格式分派，回调零分配/零锁）+ `render::BlockRenderer`（固定块→可变交错缓冲的拉取适配器，设备/离线同路径）+ `file_sink::render_to_wav`（确定性离线 32-bit float WAV，golden 对拍）+ `capture`（无锁环形缓冲的麦克风/总线回读，整帧溢出计数）。11 单测全绿，clippy 零告警，默认与 `--no-default-features` 双配置构建通过，已 commit。worklet/触感/远程授权通道后续档。
  - **ECS 前端（进行中）**：`bevy_audio` 前端改造（`AudioPlayer`/`PlaybackSettings`/`Volume` 兼容 API 翻译为命令环）。
- **M4 空间（进行中）**：
  - **首批已落地（已 commit）**：`prism_audio_spatial` = `geometry` 几何基座（`Listener`/`Emitter`/`LocalSource` + `localize`）+ 6 个叶子 DSP 模块——`attenuation`（OpenAL clamped 距离模型）/`cone`（锥形）/`doppler`（径向速度频移）/`air`（ISO 9613-1 空气吸收 + Biquad 低通节点）/`panner`（VBAP/pairwise 等功率多布局 + `PannerNode`）/`ambisonics`（AmbiX ACN/SN3D FOA 编码/旋转/解码 + `FoaEncoderNode`）+ `occlusion`（遮挡/障碍：可插拔 `trait OcclusionQuery` + `NullOcclusionQuery`，`Occlusion` 把 obstruction/occlusion 两因子映射为直达增益/对数域低通 + 湿声发送缩放，RT `OcclusionNode`）+ `spatializer`（空间化编排层：`SourceDescriptor` authoring 数据 + 纯控制率 `resolve` 产出 `SpatialParams`——直达增益=距离×锥×遮挡、多普勒频移、方位/仰角、直达低通取空气与遮挡更紧者、湿声发送；不新增 RT 节点、可 RT 调用）+ `spread`（扩散/聚焦塑形：多点虚拟子源扇形 + focus 升余弦窗 `cos(u·PI/2)^(focus·6)` + 距离曲线线性插值，`spread_taps`/`compute_spread_gains` 栈上固定缓冲零分配，已接入 `spatializer`）+ `multi_position`（多位置声源：`PositionInput` 集合 + `resolve_multi` 三模式 nearest/blend/envelop 折叠为单一 `SpatialParams`，方向在局部单位向量上加权合成、envelop 能量功率求和 + 角展宽扩散、栈上固定缓冲零分配）+ `propagation`（几何声学传播后端：可插拔 `trait PropagationBackend`——比 `OcclusionQuery` 更丰富、控制率枚举直达/透射/衍射/反射四类 `PathKind` 路径 + `FreeFieldBackend` 自由场默认；经典 DSP `fresnel_number`/**Maekawa 1968 屏障衍射** `maekawa_attenuation_db`/`diffraction_gain`/`diffraction_cutoff_hz` 衍射低通/`transmission_gain` 透射/`edge_path_difference` 绕边程差/`AcousticMaterial` 材质，`tanh`/`log10` 由 `exp`/`ln` 构建）+ `rooms`（房间与门户：`RoomNetwork` 实现 `PropagationBackend` 把跨房间传播接入 propagation→occlusion→spatializer 管线；`Room` AABB 声学体积 + wall `AcousticMaterial` 透射、`room_of` 取最内层房间归属、`Portal` 矩形门户 `closest_point`/openness 混合 `transmission_gain`、`portal_coupling_gain`=门户透射×声源/听者侧 Fresnel-Kirchhoff 斜度因子 `obliquity_factor`、query 同房间单直达 vs 跨房间穿墙透射+门户次级路径，`MAX_PROPAGATION_PATHS` 有界，单跳门户）+ `reverb_zones`（混响分区与游戏定义辅助发送：`ReverbZone{bus, shape:Box/Sphere, send_level, blend_distance}` 区域绑定辅助返回总线、`weight` cubic smoothstep 过渡带、`ReverbZoneField::resolve` 由听者位置解析活跃 `AuxSend` 集合（同总线取最强、不同总线各一路、超 `MAX_AUX_SENDS` 淘汰最弱）、环境发送由听者驱动、每声源 `source_send_gain`=zone_send×wet_gain 叠加自身湿声缩放） + `hoa`（高阶 Ambisonics 编解码：SN3D/ACN 实球谐至三阶 16 通道、`fill_hoa_coeffs` 纯 Cartesian 递推（Chebyshev 方位 + 关联勒让德三递推 + `sn3d_norm` 阶乘比）、零向量退化全向、`encode_hoa`/`decode_hoa`（解码除以 `order+1`）、RT `HoaEncoderNode` 逐样本平滑） + `hoa_rotation`（HOA 声场旋转：实球谐（SN3D/ACN）分块对角旋转至三阶，一阶块 `R^1=A·Q·A`（A=diag(-1,+1,-1)、Q=Mat3::from_quat）+ 二/三阶 **Ivanic–Ruedenberg 递推** U/V/W+P 系数逐阶构造，`rotate_hoa`（原地栈上临时矩阵）/`HoaRotationMatrix`（控制率算一次、多帧 `apply` 复用），非有限/近零四元数回退恒等，golden `rotate_hoa(encode(d))==encode(q·d)` 对拍 + 逐阶能量守恒） + `hoa_decode`（HOA 双频段能量优化解码：低频段 basic/in-phase 保相位、高频段 max-rE 保能量矢量，高频每通道增益 `(2n+1)*g_n` 修正 SN3D-scaled 投影解码的模态计数缺失、`max_re_gains`=`P_n(r_E)`/`max_re_radius`=`cos(137.9度/(order+1.51))`、`DualBandDecoder`/`SpeakerLayout` 栈 32，分频交叉留调用方） + `hoa_beamform`（可转向虚拟传声器/模态波束成形：`BeamPattern{Basic, MaxDi(g_n=2n+1，DI=(order+1)^2), MaxRe(P_n(r_E)), InPhase((L!)^2/((L+n)!(L-n)!) 无负旁瓣)}`、归一 `Sigma g_n` 使 look 轴响应=1、`Beamformer::beam(coeffs, look_dir)` 收 mono 虚拟传声器） + `nfc`（HOA 近场补偿滤波：参考距离稳定化 `H_m(s)=theta_m(s*r_src/c)/theta_m(s*r_ref/c)`、反向 Bessel 根硬编码 + 双线性变换到一阶/biquad 级联、高频归一/DC 增益 `(r_ref/r_src)^m` 有限、全极点单位圆内稳定，`NfcCoeffs::design`/`NfcFilter{process_channel,process_block}`）+ `early_reflections`（镜像源法早期反射：Allen-Berkley 1979 矩形房间整数索引 `(nx,ny,nz)` 镜像枚举（折叠闭式、位可复现），每镜像→听者局部到达方向/欧氏距离/延迟（整数步进避 f32->int cast、钳 `MAX_DELAY_SAMPLES`）/增益（`1/max(distance,0.1)` × 各面 `beta=sqrt(1-alpha)` 反射系数积），总阶钳 `MAX_REFLECTION_ORDER=4`、退化轴仅出 n=0、溢出保最强 top-k（`MAX_EARLY_REFLECTIONS=32`），`compute_early_reflections` 栈上零分配 + RT `EarlyReflectionRenderer` 单共享延迟线多抽头 `equal_power_pan` 渲染，`ShoeboxRoom{new/rigid}`/`ReflectionTap`）+ `portal_graph`（多跳门户路由：房间图有界无环 DFS 寻路（深度<=MAX_PORTAL_HOPS=4）、visited 定长栈去环、门户升序枚举保确定性，几何链 源->门户 closest_point 孔径->听者累积距离/延迟、增益=各 portal_coupling_gain 积 x 1/max(dist,0.1)、溢出流式 top-k 保最强，`route_portals`/`RoutedPath`/`PortalHop`）+ `room_acoustics`（房间统计混响声学：Sabine/Eyring/Millington-Sette RT60 + Schroeder 频率 + 平均自由程/临界距离，驱动上游 FDN）+ `material_library`（建筑声学吸声材料库：八倍频程吸声系数表 + log 频率插值 + 宽带均值 + 具名材料 Concrete/Carpet/AcousticTile 等，喂 `early_reflections`/`room_acoustics`）+ `octave_reverb`（倍频程混响时间：per-band Eyring RT60 频谱（每面每带吸声 + ShoeboxRoom 几何）+ log 频率插值 + broadband T_mid + `fdn_decay_gains` per-band FDN 反馈增益 g=10^(-3*delay/RT60)，驱动频变 FDN 混响）+ `scattering`（表面散射/扩散：ISO 17497-1 每倍频程散射系数谱 + specular/diffuse 能量守恒拆分 + Lambert 余弦律 directivity，specular 喂 `early_reflections`、diffuse 喂晚期扩散场）+ `diffusion_field`（晚期扩散场：`scattering` 拆出的 diffuse 能量逐面汇聚成每倍频程扩散能量密度 + Schroeder 回声密度 `4*PI*c^3*t^2/V` + Polack/Jot `sqrt(V)` 混合时间 + diffusion coefficient + `late_send_gains`/`fdn_late_gains` 驱动频变 FDN 晚期） + `room_modes`（矩形房间驻波本征模态：Rayleigh 频率闭式 + 轴/切/斜向分类及 4:2:1 权重 + 模态密度 + 倍频程染色） + `source_directivity`（频变声源指向性：加权一阶心形辐射 `d=((1-s_b)+s_b*cos).max(0)` 每倒频程 sharpness 随频递增 + 指向因子 `Q_b=1/((1-s_b)^2+s_b^2/3)` 与指数 `DI=10*log10(Q)` + `DirectivityPreset{Omni,Cardioid,Voice,Trumpet}` 预设 + `broadband_gain` 的 log 频率 sharpness 插值，供上游混音 direct 声源辐射加色）+ `reverberant_field`（稳态混响场能量与直混比：房间常数 R=S*a_bar/(1-a_bar) + 直达 Q/(4πr²)/混响 4/R 能量分解 + DRR 及 dB + 临界距离 r_c=sqrt(QR/16π)，与 source_directivity 的 Q 协同）+ `diffraction`（边缘/屏障绕射插入损失：Fresnel 数 N=2*delta*f/c + Maekawa 曲线 IL=5+20log10(x/tanh x) 每倍频程衰减/增益，复用 propagation 的 Maekawa 单一真源、可配置声速，补齐 ISO 9613-2 屏障项）+ `ground_effect`（ISO 9613-2:1996 §7.3.1 地面效应衰减 `A_gr`：三区地面因子 `G∈[0,1]`（源/中/受区）+ Table 3 每倍频程高度函数（63Hz 常数 `-1.5`、125/250/500/1kHz 用 `-1.5+G*{a,b,c,d}(h,d_p)`、2k/4k/8kHz 用 `-1.5*(1-G)`）+ 中间区权重 `q=1-30*(h_s+h_r)/d_p`，正值=衰减/负值=增益（硬地建设性反射），补齐户外传播加性预算 `A_div+A_atm+A_gr+A_bar` 的地面项）+ `outdoor_propagation`（ISO 9613-2 户外传播总预算聚合：`A_total=A_div+A_atm+A_gr+A_bar` 每倍频程组合器，散度 `A_div` 委托 `Attenuation`、大气 `A_atm` 委托 `air`、地面 `A_gr` 委托 `ground_effect`、屏障 `A_bar` 委托 `diffraction`，DRY 无物理重复）+ `reflection_directivity`（方向感知早期反射加权组合器：源指向性按每倍频程辐射增益施加到镜像源抽头，`source_directivity` x `early_reflections` 之上的 DRY 组合层，控制率无 per-sample DSP）+ `reflection_clustering`（早期反射方向聚类：把镜像源抽头按到达方向归入 6 个轴对齐方向簇、能量守恒非相干聚合（簇能量=Σgain²、代表方向/延迟=能量加权平均），折叠众多反射为紧凑方向送出，`reflection_directivity` 的姊妹归约层，可组合「先加权后聚类」）+ `convex_room`（凸多面体房间镜像源早期反射：把 shoebox 镜像法推广到任意凸房间（半空间交集），镜像 `S'=S-2*(dot(n,S)-offset)*n`、线段 `L->S'` 与各面求交 + 其余面内侧剔除房外路径，`order` 钳 1（直达 + 一阶）、整数步进延迟避 f32->int cast、溢出 top-k 保最强，产出同 `ReflectionTap` 列表无缝喂下游 `reflection_directivity`/`reflection_clustering`/`EarlyReflectionRenderer`，`ReflectionPlane{normal/offset/reflection + new/from_absorption}`/`ConvexRoom{new/shoebox/planes/contains}`/`ConvexReflections`/`compute_convex_reflections`，Allen-Berkley 镜像法向任意凸平面边界的公开推广）。434 单测 + 36 doctest 全绿。**独立 crate `prism_audio_hrtf`（HRTF 双耳渲染，已 commit）**：`dataset`/`sofa`（`HrirSource` trait + `aes69_to_local`，SOFA 二进制解码后续档）/`interpolation`（ITD 群延迟对齐 + Shepard 反距加权）/`binaural`（overlap-save 分区卷积 + HRIR 交叉淡入）/`nearfield`（双耳视差 + 1/r + 球形头遮蔽近似）/`headtracked`（头部追踪双耳：四元数轴角指数外推 + slerp 平滑 + `predicted_local_angles` 喂既有插值重选 HRIR 方位）+ `hoa_binaural`（虚拟扬声器双耳：每 ACN 通道预烘焙 `filter[c]=Sigma_s D[s][c]*HRIR(s)` 对、运行时每通道一 `BinauralRenderer` 累加 L/R、数学等价全虚拟扬声器解码但更省、`HoaBinauralDecoder`/`VirtualSpeakerLayout::cube26`）+ `transaural`（跨听/串音消除渲染：递归 crosstalk 消除器 + 对称扬声器 Woodworth 几何推导），81 单测 + 4 doctest 全绿，clippy 全特性/无默认特性零告警，no_std 双构建通过，确定性数学走 `bevy_math::ops`。
  - **规划**：为 `propagation` 提供**真实几何 backend**（复用 `prism_physics`/GPU BVH 射线，追踪边/门户/反射面，为 `occlusion` 提供真实几何因子）（HOA 编解码 `hoa`、HOA 声场旋转 `hoa_rotation`、HRTF 双耳与头部追踪双耳 `prism_audio_hrtf::{binaural,headtracked}` 均已落地；距离与方向塑形全族 spread/focus/多位置 + 传播后端接口已落地；余真实几何 backend（多跳门户路由 `portal_graph` 已落地）、对象/Atmos 与平台原生空间后端桥）（`rooms` 单跳门户 + `reverb_zones` 混响分区/辅助发送已落地，多跳门户路由 `portal_graph` 已落地，剩真实几何 backend）+ 平台空间后端桥。
- **M5 编排、Patch 与音乐**：Event/Container/State/Switch/RTPC + Patch 编译器与合成原语 + Modulation（LFO/包络/控制总线）+ 交互音乐（段/过渡/stinger）+ Bank/流式 + 对白与本地化解析（§35）+ 程序化音景（§37） + 物理耦合程序化音频（接触事件总线/模态/颗粒合成，§47，联动 `prism_physics`）。
- **M6 母带、合规、剖析与工具**：LUFS 归一 + true-peak limiter + HDR 窗口 + snapshot + 无障碍 + Profiler/频谱/计量面板 + 触感与跨模态输出（§36）+ 实时授权与远程工具 API（§38） + 输出渲染链（下混矩阵/低频管理/交付动态范围档，§48）。
- **M7 次世代执行与声学**（横切增强，随 M1-M6 演进落地）：编译图 ExecPlan + 缓冲活跃度分配 + PDC（§29）；岛屿划分与 Job 化确定性并行调度（§30）；复用渲染器 GPU BVH 的声学声线/路径后端与烘焙（§31）；CPU 预算闭环治理器与音频 LOD（§32）；掩蔽感知虚拟化与声源聚类（§33）；`Resampler`/`TimeStretcher` 质量分级（§34）。每项均带确定性/golden 对拍验收。

---

## 41. 关键扩展点清单

| 扩展点 | trait | 用途 |
|---|---|---|
| 处理单元 | `AudioNode` | 任意 DSP/总线/空间化 |
| 内容子图 | `PatchNode`（编译产物） | 程序化合成声音 |
| 运行时构图 | `PatchBuilder` | 任务线程增量拼装/改写 Patch 后热切换 |
| 传播后端 | `PropagationBackend` | 几何/GPU/波动烘焙/第三方，可混合（§14/§31/§43） |
| 声像/空间化 | `Panner` | VBAP/HRTF/Ambisonic/平台 SDK |
| 调制器 | `Modulator` | LFO/包络/曲线/自定义调制 |
| 解码器 | `SourceDecoder` | wav/ogg/flac/自定义 |
| 设备后端 | `DeviceBackend` | cpal/worklet/离线/捕获 |
| 参数源 | 控制总线/RTPC 映射 | 参数联动与调制 |
| 图调度器 | `GraphScheduler` | 串行/Job 化并行图执行策略 |
| 重采样器 | `Resampler` | 线性/多相 sinc/高阶 sinc 质量分级 |
| 时间伸缩 | `TimeStretcher` | WSOLA/相位声码器（变调不变速） |
| 质量治理 | `QualityGovernor` | CPU 预算闭环的音频 LOD 策略 |
| 对白解析 | `DialogueResolver` | 运行时按键值/语言/决策树选媒体 |
| 触感后端 | `HapticBackend` | 宽频（DualSense）/双马达/运动设备 |
| 音景调色板 | `SoundscapePalette` | 程序化环境散布规则与调度 |
| 授权通道 | `AuthoringTransport` | 远程只读遥测 + 白名单写命令 |
| 波动烘焙 | `WaveAcousticsBaker` | 离线波动仿真 → 感知参数场（§43） |
| 声学探针场 | `AcousticProbeField` | 运行时感知参数查表/插值/流式加载（§43） |
| HRTF 数据集 | `HrtfDataset`（SOFA/AES69） | 测量 HRIR 加载与个性化选型（§44.3） |
| 对象渲染 | `ObjectRenderer` | 床+对象/MPEG-H 元数据渲染与折算（§44.2） |
| 通信管线 | `VoiceCommPipeline` | 上行前处理/下行解码隐藏的可插拔通信链（§45） |
| 回声消除 | `EchoCanceller` | AEC 自适应滤波 + 双讲检测 + 残余抑制（§45.2，经典、无 ML） |
| 语音传输 | `VoiceTransport` | 对接宿主/平台网络语音包收发与抖动缓冲（§45.3） |
| 接触事件源 | `ContactEventSource` | 物理碰撞/接触流→采样精确合成激励（§47） |
| 模态合成 | `ModalSynth`/`ModalBank` | 冲量激励的谐振器组撞击/材质合成（§47） |
| 颗粒合成 | `GranularEngine` | 确定性颗粒云（碎裂/群体/材质细节，§47） |
| 下混矩阵 | `DownmixMatrix` | BS.775 多声道折降与双耳（§48） |
| 低频管理 | `BassManager` | LFE 交叉/校准/耳机折回（§48） |
| 输出交付档 | `OutputProfile` | Home Theater/TV/Night/耳机 动态范围与响度档（§48） |

---

## 42. 开放问题

- 图/资源交换的旧对象回收策略：延迟队列 vs 引用计数 vs epoch（当前倾向 epoch）。
- HOA 阶数与 CPU 预算的默认档位。
- 卷积混响的分块 FFT 大小与延迟/CPU 折中。
- 移动端功耗档：降采样率/降语音数的自适应策略。
- 声学材质与视觉材质资产的字段合并范围。
- Patch 嵌套深度上限与编译展开的内存上界。
- 平台空间后端的能力探测与优雅降级策略。
- 触感设备能力差异（宽频 vs 双马达）的统一波形抽象与降级映射。
- 对白媒体运行时选择的查表命中与流式预取延迟预算。
- 远程授权通道在发布构建的启用/鉴权策略与命令白名单粒度。
- 程序化音景元素密度与 §33 聚类阈值的默认档位。
- 运行时 `PatchBuilder` 增量构图的编译预算与热切换频率上限（防任务线程构图风暴与 epoch 回收积压）。
- 片段式交互流与段/播放列表两种交互音乐模型的统一数据表示与作者取舍。
- 物理耦合程序化音频（§47，Impacter/模态/颗粒）：模态表规模与每物体模态数默认档、每块撞击事件上限与去重/聚类阈值、颗粒池容量，与 §32 预算/§33 聚类的默认映射。
- 输出交付档（§48）默认值：低频管理交叉频率（80 Hz？）与 LFE 校准增益、Home Theater/TV/Night 各档的压缩/HDR 窗口默认参数、各平台默认响度目标（游戏 -16 vs 主机认证）与对白锚定的默认锚点选择。
- MetaSound Pages 式分档编译与 §32 `QualityGovernor` 预算联动：分档切换滞回阈值与热切换频率上限，避免档位抖动引发编译风暴。
- 变调 SamplePlayer 抗混叠档位（过采样倍率 vs 波表 mip）与 §32 音频 LOD 的默认映射。
- 延迟补偿播放头查询在不同设备后端（cpal/worklet）下 output_latency 的可得性与估计精度。
- 波动声学探针网格分辨率、感知参数量化位深、压缩与内存·质量折中，以及波动档与几何/GPU 档的混合权重与交叉淡化策略。
- 动态开口（门/可破坏墙）用多状态预烘焙插值 vs 几何实时增量修正的选择阈值与烘焙组合爆炸控制。
- 编解码分档默认映射（音效 PCM/ADPCM vs 音乐/对白 Opus）与编码器延迟/填充在无缝循环下的裁剪校准。
- 硬件对象数上限下的对象→床折算与 §33 聚类阈值默认档；MPEG-H/Atmos 后端能力探测与优雅降级。
- 个性化 HRTF 的选型输入（人体测量 vs 运行时校准）与 SOFA 数据集打包/流式加载预算。
- 通信语音自适应抖动缓冲的默认目标深度与「延迟 vs 卡顿」权衡档位（移动/主机/PC 分档）。
- 位置语音是否默认关多普勒、团队频道 2D 与世界语音 3D 的并存路由默认策略。
- 时钟漂移校正 `DriftResampler`（§22）的默认水位阈值与滞回、有界重采样比率范围，以及“稳态漂移纠正 vs 开流即拒绝不匹配率”两机制的边界划定；多设备/回环协同下的主时钟选择。
- 采样精确 seek（§10）在流式源下的预取重填延迟预算、编解码器预滚/填充在 seek 点的裁剪校准，以及 seek 未就绪静音窗口的可接受时长档位。
- 无分配守卫（§28）对第三方 `AudioNode` 的强制范围与 opt-out 策略，以及守卫在 debug/test 之外是否提供轻量运行期采样式检测。
- **是否放开「纯经典 DSP、无 AI/ML」约束纳入神经音频**（神经降噪 RNNoise/DeepFilterNet、神经编解码 Encodec/SoundStream、神经 HRTF 个性化、神经源分离与上混）——当前默认**关闭**，待用户决策；若放开须严格限定「训练/推理只在离线·资产期或独立线程，RT 侧只查表/取已算结果或旁路，绝不在音频回调内跑网络推理」的安全边界，并保留经典路径为默认回退。

---

## 43. 波动声学与混合精算传播（预计算波动场 + 几何/GPU 射线混合）

几何声学（§14 射线遮挡/障碍、§31 GPU 声线/路径）在**低频、复杂绕射、耦合空间、真实混响衰减**上会失真——射线模型假设波长远小于几何尺度，对门缝绕射、房间共振、软遮挡并不物理正确。次世代旗舰转向**离线波动仿真 + 运行时感知参数查表**：微软 **Project Acoustics**（基于 Triton / **ARD 自适应矩形分解**波动求解，用于《战争机器5》《盗贼之海》）离线求解波动方程，把物理正确的遮挡、衰减、到达方向、混响编码为紧凑参数场，运行时轻量查表即得。Resonance 将其纳为 `PropagationBackend` 的**波动档**，与几何档、GPU 档**并列且可混合**，编译期/资产期选择，运行时统一喂入同一套 §14 空间参数与 §17 发送——不新增第二套声学真相。

- **离线波动烘焙（Wave Bake）**：对静态几何做时域波动仿真。采用 ARD（把场景划成矩形子域，域内解析求解 + 界面数值耦合，远比朴素 FDTD 高效、数值色散低），在**探针网格（Probe Grid）**上对听者位置、多个声源采样位置求解脉冲响应。烘焙经 `bevy_tasks` 后台或离线工具执行，产物经 `bevy_asset` 加载与热重载。
- **感知参数编码（Perceptual Encoding）**：不存原始 IR（体积巨大），而从仿真 IR 提取**感知参数场**——直达/初期能量（→遮挡增益 + 低通截止）、衰减时间 RT60/EDC（→混响湿量与衰减）、初期到达方向与其能量（→空间化方位与早反射塑形）、湿/干比。每探针一组紧凑量化系数，运行时对听者位置做三线性插值。对齐 Project Acoustics 的「Baked perceptual parameters」，但字段与 §14/§16/§17 的既有参数总线一一对齐，无中间语义层。
- **动态开口（Dynamic Openings）**：门/窗/可破坏墙等运行时可变连通，用**多状态预烘焙**（开/关及中间档）参数插值，或以几何档实时修正**叠加到波动基线**——静态波动给全局物理正确基线，几何/GPU 射线补动态增量。
- **混合传播（Hybrid Propagation，本引擎默认目标）**：波动档负责**低频、绕射、软遮挡、房间耦合与真实混响尾**（波动物理强项）；几何/GPU 档（§14/§31）负责**高频镜面反射、动态遮挡增量、多普勒**（射线强项）。两档产出的同名参数在同一 §7 `Smoothed` 上叠加或交叉淡化，`PropagationBackend` 抽象对上层完全透明；场景/预算决定单帧内某源走哪档。
- **数据规模治理**：探针网格分辨率、参数量化位深随 Bank（§20）分档；远场/次要区域稀疏探针，交互热点加密；压缩场 + 流式加载（§20）控制内存，卸载走 epoch 回收（§21）。
- **确定性与 RT 安全**：波动求解**只在离线/任务线程**发生；RT 线程只做**查表 + 插值 + 平滑**（标量参数，零分配、无锁、不 panic），绝不在音频回调跑求解——与 §31 GPU 声学「只产控制参数、不回样本流」同构，样本级 DSP（卷积/FDN/双耳）始终在 CPU 音频线程确定性执行。

后端选择矩阵（补充 §14 的可插拔后端说明）：

| 后端档 | 强项 | 代价 | 触发条件 |
|---|---|---|---|
| 几何 CPU（`prism_physics` 射线） | 动态、低延迟、通用 | 低频/绕射不准 | 默认；动态源、无烘焙区域 |
| GPU 声线/路径（§31） | 高密度反射/遮挡并行 | 需 GPU、异步延迟 | 高语音密度且有 GPU |
| 波动烘焙（本节） | 低频/绕射/耦合物理正确、混响真实 | 需离线烘焙 + 内存 | 静态复杂空间、旗舰质量 |
| 混合 | 各取所长、代际质量 | 编排/权重复杂 | 次世代默认目标 |

**超越判断（保持代际差异）**：Project Acoustics 是独立中间件，维护自有场景与探针；Resonance 复用渲染器/物理**同一几何真相**（§31），并把波动档、几何档、GPU 档**统一在同一可插拔后端与同一空间参数总线**下，由 §32 `QualityGovernor` 按 CPU 预算与场景特征**自动选档与混合**——近处动态源走几何/GPU、整体房间声学走波动烘焙，二者在参数域无缝叠加。

---

## 44. 编解码、对象音频与个性化空间输出（Codec / Object-based / MPEG-H / 测量 HRTF）

本节补齐从「资产字节」到「沉浸式输出」两端的工程能力。**恪守本文档纯经典 DSP 路线：以下全部为经典编解码与数据驱动方法，不含任何 AI/ML 推理。**

### 44.1 编解码策略（Codec）

- **`SourceDecoder` 矩阵化**（§10/§20 已引 trait）：PCM（无损、常驻内存音效）、**ADPCM**（低解码开销、支持海量并发短音效，对齐主机常用）、Vorbis/Ogg、**Opus**（低延迟，语音/对白与流式音乐优选，CBR/VBR）、FLAC（无损归档）。格式插件化，第三方可注册自定义解码器。
- **质量/内存/CPU 三角**：Bank（§20）按用途选编码——短高频音效走 PCM/ADPCM（解码近零成本、可高并发），音乐/环境走 Opus/Vorbis（压缩比优先、流式），对白走 Opus（低码率高可懂度）。
- **解码位置**：一律在 `bevy_tasks` 后台线程解码到预分配缓冲/环形（§20），RT 线程只读——**编解码永不在音频回调发生**。
- **无缝循环与采样精确**：Opus/Vorbis 的编码器延迟/填充在资产期记录，解码后精确裁剪，保证 §8 循环点与节拍量化对齐不随重编码漂移。

### 44.2 对象音频与沉浸格式（Object-based / Atmos / MPEG-H）

- **床 + 对象（Bed + Objects）模型**：混音同时输出**声道床**（7.1.4 等）与**动态对象**（携位置/大小/增益元数据），对齐杜比 **Atmos**（最多 128 对象）与 **MPEG-H 3D Audio**（对象 + 声道 + HOA，ATSC 3.0 广播标准）。§16 空间总线原生产出对象元数据流。
- **渲染分叉**：支持对象的平台后端（Atmos/Tempest/Windows Sonic）直接接收对象元数据；不支持的设备由内建渲染器把对象**折算进床或双耳**（§16 输出适配），能力探测 + 优雅降级。
- **对象预算与聚类**：硬件对象数有限，超限时用 §33 声源聚类把弱贡献对象归并为床或代表对象，感知优先保留前景对象。
- **HOA 传输**：场景型环境声用高阶 Ambisonics（AmbiX：ACN + SN3D）承载，末端解码到床/对象/双耳（§16）。

### 44.3 测量与个性化 HRTF（Personalized / SOFA）

- **SOFA 数据集加载**（AES69 标准）：`HrtfDataset` 加载测量 HRIR（多方位/仰角/距离），对齐 Steam Audio 自定义 HRTF 与学术数据集（SADIE/CIPIC 等）。
- **个性化选择（经典、无 ML）**：借鉴索尼 360 Reality Audio / 苹果个性化空间音频的**目标**，但走经典数据驱动路径——按用户人体测量（耳廓尺寸/头围）或从若干标准数据集中选最匹配者，或运行时校准（用户微调仰角提示直至定位准确）后锁定对应 HRIR 集。**无神经网络推理**，纯查选 + 插值，恪守本文档纯经典 DSP 路线。
- **近场与距离**：近场声源做 ITD/ILD 增强与近场增益补偿（§16 已引），随距离在测量集内插值。
- **头追双耳贯通**：与 §16 头部追踪一致——姿态旋转 Ambisonic 场或重选 HRIR 方位，低延迟短前瞻，避免声像黏头。

---

## 45. 实时通信音频（Communication / 位置语音 / 回声消除 / 抖动缓冲）

现代 AAA、多人、社交与 VR 场景把**实时语音通信**当作一等音频子系统（Wwise Communication、平台 Voice Chat：PlayStation/Xbox/Meta，以及 WebRTC 的经典 DSP 前处理链）。Resonance 将通信语音纳入**统一渲染图**——远端语音是图里的**一等声源**，与游戏音效共享空间化（§16）、距离塑形（§15）、遮挡/衍射（§14/§43）、母带 HDR（§13）与 LOD（§32），而非旁路混音。**恪守本文档纯经典 DSP 路线：以下 AEC/降噪/AGC/VAD/PLC 全为经典自适应滤波与谱域方法，不含任何 AI/ML；神经降噪等作为 §42 开放决策项，默认关闭。**

### 45.1 通信管线总览

上行（本地麦克风）与下行（远端语音）两条链，均可插拔（`VoiceCommPipeline` trait）：

- **上行**：采集 → DC/高通 → 回声消除（AEC）→ 噪声抑制（NS）→ 自动增益（AGC）→ 语音活动检测（VAD）门控 → 编码（Opus）→ 交宿主/平台传输。
- **下行**：网络包 → 自适应抖动缓冲 → 解码 → 丢包隐藏（PLC）→ 作为源节点入图空间化 → 混入总线。

网络 I/O 与前处理在 std/任务线程；RT 线程只从无锁环（§21）取已解码 PCM——**编解码与网络永不在音频回调内发生**。

### 45.2 采集前处理链（Uplink，全经典 DSP）

- **DC 去除与高通**：去直流与低频隆隆，为后续自适应滤波提供稳态输入。
- **声学回声消除（AEC）**：以本地扬声器输出为参考信号，用 **NLMS / 分块频域自适应滤波**估计并抵消经房间反馈进入麦克风的回声；含**双讲检测（Double-Talk Detector）**在本地与远端同时说话时冻结自适应、**残余回声抑制**做尾处理。对齐 WebRTC AEC3 的经典结构（**非 ML**）。
- **噪声抑制（NS）**：经典**谱减法 / 维纳滤波** + 噪声底估计（最小值跟踪 / MCRA），抑制稳态背景噪声；**不使用 RNNoise / DeepFilterNet 等 ML 降噪**（恪守本文档路线；放开与否见 §42 神经音频开放项）。
- **自动增益（AGC）**：目标响度归一（§13 LUFS 思想的轻量在线版）+ 前瞻限幅防削波。
- **语音活动检测（VAD）**：以能量 / 过零率 / 谱平坦度做经典判决，驱动**静音门控**（不传静音包省带宽）并联动 §33 感知层与 §25 语音管理。

### 45.3 编解码、网络与抗丢包

- **编解码**：复用 §44.1 `SourceDecoder` 与编码器矩阵，语音默认 **Opus**（低延迟、窄带至全带自适应、内建 FEC/PLC），码率随网络自适应（CBR/VBR）。
- **前向纠错与丢包隐藏**：Opus **带内 FEC** + 解码侧 **PLC**（用前一帧激励外推填补丢包），避免爆裂与断续。
- **自适应抖动缓冲（Jitter Buffer）**：按网络抖动统计动态调整目标缓冲深度，在延迟与卡顿间权衡；通信走**墙钟低延迟路径**，与 §8 采样精确节拍时钟**解耦**（语音不做节拍量化）。
- **传输解耦**：引擎不内置网络栈，实际传输由宿主/平台提供，`VoiceTransport` 抽象对接语音包收发；引擎侧只负责前处理、编解码、缓冲与空间化。

### 45.4 位置语音与空间化

- **一等声源接入**：解码后的远端语音作为源节点入图，复用 §16 HRTF/Ambisonic/平台后端做 3D 定位、§15 距离衰减（语音通常关多普勒）、§14 遮挡/障碍与 §43 波动衍射（隔墙说话闷、门后绕射），随场景与预算享 §31 GPU 声学与 §32 LOD。
- **邻近语音（Proximity Chat）**：随距离渐入渐出、可分频道；团队频道走 2D 非空间化直达路径，世界语音走空间化路径，二者可并存并分别配增益。
- **侧音（Side-tone）**：可选把本地处理后语音以低增益回授本人耳机，减轻佩戴闭耳耳机时的"闷耳"失真感（经典可选项）。

### 45.5 引擎集成、隐私与合规

- **ECS 组件**：`VoiceSource`（远端说话者→实体绑定，驱动空间化位置）与 `VoiceCapture`（本地麦克风），经命令环（§21）下发路由/增益/静音，遥测（§26）上抛说话者电平、丢包率、抖动、AEC 收敛度。
- **推送说话（PTT）/ 开放麦 / 静音**：门控策略与本地/全局静音、屏蔽名单，RT 侧零分配原子切换。
- **隐私与平台合规**：麦克风采集需显式授权与录音指示（联动 §23 无障碍/合规）；平台语音策略（家长控制、屏蔽、地区法规）由宿主策略层裁决，引擎只执行不越权。

### 45.6 超越判断（保持代际差异）

Wwise Communication 与平台语音多为**独立旁路**混音，位置语音往往退化为固定 2D 直达；Resonance 把通信语音**统一进同一条编译图 + 确定性并行**，因而位置语音天然享有与游戏声一致的几何/波动声学（§14/§43）、GPU 声学（§31）、母带 HDR（§13）与预算 LOD（§32）——远处队友语音可被 §33 感知层虚拟化、被 §32 按 CPU 预算降质，遮挡随波动场物理正确变化，而非旁路里一套割裂的空间化。

---

## 46. 次世代 DSP 与工程内核深化（借鉴超越）

前述章节确立了架构与内容/空间/编排的完整版图；本节补齐把设计**落到代际领先实现**所需的底层 DSP 与工程内核细节——这些是区分「能出声的玩具」与「AAA 次世代内核」的关键，均恪守纯经典 DSP、RT 铁律与 golden 逐样本对拍。

### 46.1 零延迟分区卷积（Partitioned Convolution）

`convolver`（§9）与卷积混响、测量 HRIR（§16）、Reflect 式反射（§14）都需要与**长冲激响应（IR，数千至数十万抽头）**做实时卷积。朴素时域卷积 O(N) 每样本不可行，单块 FFT 卷积则引入等于 IR 长度的延迟——对交互音频不可接受。采纳业界标准 **UPOLS（Uniformly-Partitioned Overlap-Save）+ 非均匀分区（Gardner/García）**：

- **均匀分区**：IR 切成等长块，每块预算 FFT（构造期完成，RT 只做频域复乘 + 累加频域延迟线），单块延迟 = 一个音频块，等同直路延迟，故「零额外延迟」。
- **非均匀分区（NUPOLS）**：头部用小块（低延迟）、尾部用逐级增大的块（低 CPU），在延迟与吞吐间取最优；块尺寸阶梯与调度对齐 §8 Transport 块边界。
- **RT 安全**：所有 FFT plan、频域缓冲、延迟线槽位构造期预分配；RT 侧仅正/逆变换与复数 MAC，零分配零锁。FFT 走确定性定点/`bevy_math::ops` 兼容实现以保证跨平台位一致（§24）。
- **落地**：作为 `convolver` 的内部策略，IR 更换（换混响预设/HRIR 集）经 §21 命令环在线程外重算分区并 epoch 热切换，不打断 RT。

### 46.2 SIMD 向量化与内核抽象

planar 布局（§6）的核心收益是**每通道连续 → 自动/显式向量化**。设计一层薄内核抽象（`DspKernel`），把 gain/mix/biquad/pan/MAC 等热原语写成对 `&[Sample]` 的批处理，运行期按 CPU 特性选择实现：

- **可移植优先**：默认走标量循环，交由 LLVM 自动向量化（形状友好：无别名、定长、连续），保证任意目标可编译。
- **显式加速档**：在 x86 提供 SSE/AVX、在 aarch64 提供 NEON 的窄向量内核（经 `std::arch` 或 `wide`/`portable_simd` 之类的可移植 SIMD 封装），**须与标量档逐样本一致**（golden 对拍校验，容差为 ULP 级）——否则回退标量，绝不牺牲可回放性。
- **RT 铁律**：内核零分配、无分支热路径（用掩码/select 替代 per-sample 分支）；反规格化保护见 §46.3。
- **与并行的关系**：SIMD 是**块内**数据并行，§30 岛屿 Job 化是**图级**任务并行，二者正交叠加。

### 46.3 反规格化数保护（Denormal / FTZ·DAZ）

递归结构（biquad、FDN、delay 反馈、包络）在信号衰减到近零时会滑入**次正规浮点**，在部分 CPU 上引发 10–100× 变慢的隐性爆音风险。双重防护：

- **硬件档**：在音频回调进入时对本线程置 **FTZ（Flush-To-Zero）/DAZ（Denormals-Are-Zero）** MXCSR/FPCR 标志，退出时恢复；此为平台特定、隔离在 `prism_audio_device` 回调边界，核心 DSP crate 不依赖它。
- **算法档（可回放）**：对反馈路径注入极小 **DC 偏置 / 抖动**或做「denormal 挤零」（低于阈值直接置 0），保证即使硬件档不可用也不劣化，且行为确定性、纳入 golden 对拍。核心 crate 走此档以维持 no_std 位一致。

### 46.4 Ambisonic 声场旋转与双频段解码

§16 的 Ambisonics 与头追双耳需要两项工程内核：

- **声场旋转（Soundfield Rotation）（✅ 已落地 `prism_audio_spatial::hoa_rotation`）**：听者头部转动时，对 HOA 场直接施加**球谐旋转矩阵**（`hoa_rotation`：一阶块 `R^1=A·Q·A` + 二/三阶 **Ivanic–Ruedenberg 递推**逐阶构造，`rotate_hoa`/`HoaRotationMatrix`），O(阶²) 一次旋转即可，远比「每源重选 HRIR 方位」廉价且无插值缝；姿态经 §21 命令环下发、RT 侧对旋转矩阵做逐块 `Smoothed` 插值防跳变（矩阵逐块插值为后续工程档）。
- **双频段解码（Dual-Band Decode）（✅ 已落地 `prism_audio_spatial::hoa_decode`）**：低频用 **basic/in-phase**（保 ITD/相位）、高频用 **max-rE**（保能量矢量、优化响度定位），分频后合并，显著改善多声道/双耳的定位稳健性（`DualBandDecoder`：高频每通道增益 `(2n+1)*g_n` 修正 SN3D-scaled 投影解码的模态计数缺失，`max_re_gains`/`max_re_radius`，分频交叉 Linkwitz-Riley 留调用方）。
- **虚拟扬声器双耳（✅ 已落地 `prism_audio_hrtf::hoa_binaural`）**：Ambisonic → 一组虚拟扬声器方位 → 各方位 HRIR 卷积求和，得到与布局无关的双耳输出；HRIR 卷积复用 §46.1 分区卷积（`HoaBinauralDecoder`：每 ACN 通道预烘焙 `filter[c]=Sigma_s D[s][c]*HRIR(s)` 对，运行时每通道一个 `BinauralRenderer` 累加 L/R，数学等价全虚拟扬声器解码但更省；`VirtualSpeakerLayout::cube26`）。
- **可转向虚拟传声器/模态波束成形（✅ 已落地 `prism_audio_spatial::hoa_beamform`）**：把 HOA 场收成单路可转向 mono 虚拟传声器（`Beamformer::beam(coeffs, look_dir)`），四族模态波束 `BeamPattern{Basic, MaxDi(g_n=2n+1，DI=(order+1)^2), MaxRe(P_n(r_E)), InPhase((L!)^2/((L+n)!(L-n)!) 无负旁瓣)}`，归一 `Sigma g_n` 使 look 轴响应=1；用于点采样隔离/上混分析/到达方向探测/定向 mono 发送。
- 全部系数（旋转矩阵、解码矩阵、虚拟扬声器 HRIR）构造期预算，RT 侧仅矩阵-向量乘与卷积，零分配。

### 46.5 近场 HRTF 与视差补偿

远场 HRTF 假设平面波，对**贴近听者（<1m）**的声源会丢失近场效应与双耳视差：

- **近场增益/ILD 补偿**：随距离对低频施加近场增益抬升与增强的 ILD（对齐球形头模型的近场传输函数修正）。
- **双耳视差（Parallax）**：近距离时左右耳「看向」声源的方位角显著不同，须**分耳计算入射方位**再各取 HRIR，而非共用一个方位——这是「贴脸音」定位可信度的关键。
- 与 §15 Spread/衰减曲线、§46.4 解码链协同，随距离在近场/远场模型间 `Smoothed` 交叉淡化。

### 46.6 频变空气吸收（ISO 9613-1 式）

§15 的「低通=空气吸收」升级为**频率相关**衰减：随距离与介质（温湿度、气压可作场景参数）按 **ISO 9613-1 式**大气吸收系数对高频施加逐频段损失，用一阶/二阶低通级联或多频段增益近似，系数随距离 `Smoothed`。这让远处声源自然「发闷」，而非全频等量衰减；与 §14 透射（材质频变损失）共用多频段塑形基础设施。

### 46.7 反馈安全图（单样本延迟破代数环）

统一图（§5）编译为 DAG，禁止环；但真实 DSP（MetaSounds 式 Patch、FDN、部分混响拓扑）需要**反馈**。解决办法对齐主流做法：

- 反馈连接必须经**显式单样本延迟节点（`z⁻¹` / `FeedbackDelayNode`）**，把「代数环（algebraic loop）」打断成合法 DAG——延迟节点的输出取自上一块/上一样本的历史，编译期即可拓扑排序。
- 编译期检测未经延迟的真环仍报 `GraphError::Cycle`（§5），并给出「需插入反馈延迟」的诊断，指导内容侧修正。
- 块粒度反馈（延迟≥一个块）零成本；样本粒度反馈（如物理建模弦/管）在节点内部以**样本级内循环**实现，仍封装为单个 `AudioNode`（对外无环），保持图级零分配与确定性。

### 46.8 声明式自动混音与类别响度治理

借鉴 CRIWARE **REACT** 与 Wwise **Auto-Ducking（带恢复曲线）**，把「谁让路给谁」从散落的手连 sidechain 升级为**声明式规则**：

- **类别（Category）轴**：正交于总线树的一层增益/上限（对白 > UI > 武器 > 环境 > 音乐…），规则声明「当类别 A 有活动声源时，类别 B 衰减 x dB，起攻/恢复用曲线」。
- **编译为调制连线**：规则在线程外编译成 §12 调制图的 sidechain 连接与 §13 ducking 参数，RT 侧仍是逐样本 `Smoothed` 增益，无运行时规则解释、可 golden 对拍。
- **与治理统一**：类别让路与 §32 `QualityGovernor` 的 CPU 预算/响度预算共享同一决策面——响度拥挤时既可让路也可虚拟化（§33），由统一策略选择，而非两套系统打架。
- **响度合规联动**：类别母线接 §13 LUFS/True-Peak 计量，自动混音的目标窗口对齐 HDR 音频（§13），确保让路后整体响度仍满足 BS.1770/R128 目标。

**超越判断（保持代际差异）**：以上多为业界成熟单点技术；Resonance 的差异在于把它们**全部收敛进同一条编译图 + 确定性数学 + golden 对拍**的骨架——分区卷积/SIMD/denormal 走同一 RT 预分配契约，声场旋转/双频段/近场同挂一条可插拔 `Panner`（§16），自动混音与质量治理共享同一决策面（§32）。没有任何一档是「外挂中间件各自维护状态」，因此跨平台位一致、可逐样本回放，这是与拼装式管线的代际分野。

---

## 47. 物理耦合程序化音频（接触 / 模态 / 颗粒合成 · 物理引擎联动）

次世代音效正从「碰撞→查表播 wav」转向**由物理真相直接合成声音**：撞击音色随冲量、材质对、接触点与物体几何连续变化，滚动/摩擦是随相对速度演化的连续过程，而非离散样本。业界标杆是 Wwise **Impacter**（冲量驱动的撞击/材质程序化合成）与学界成熟的**模态合成（modal synthesis）/物理建模**。Resonance 把这条链路做成一等公民，且**直接耦合本仓库 `prism_physics`（与 §31 GPU 场景同一几何真相）**——碰撞求解器已算出的冲量/接触点/相对速度即是合成激励，无需第二套「声学碰撞体」。全程经典 DSP、确定性、可 golden 对拍，**无任何 AI/ML**。

### 47.1 接触事件总线（Contact Event Bus，`ContactEventSource`）

- **来源**：`prism_physics` 每步产生的碰撞/接触流——首次撞击（impulse 冲量、法向/切向分量、接触点、材质对 ID）、持续接触（相对切向速度、法向压力、粗糙度）、分离。经 §21 命令环下发，携**样本偏移**交 §8 `EventScheduler`，使撞击落在物理发生的确切样本（避免帧率量化的「哒哒」感）。
- **预算与合并（RT 前，线程外/块起点）**：每块撞击事件设上限；近同时/近共点撞击**去重合并**（能量相加、取最强接触点），远处密集撞击经 §33 聚类归并为「群体撞击」，避免语音风暴。合并策略确定性、可回放。
- **能量映射**：冲量幅度→激励能量与初始增益；法向/切向比→激励谱形（硬碰亮、擦碰暗）；接触点→模态激励权重（见 §47.2）。映射为纯函数曲线（§12 曲线原语），无运行时解释。

### 47.2 模态合成（Modal Synthesis，`ModalSynth` / `ModalBank`）

- **模型**：每个「可发声物体/材质对」拥有一组**模态**——每模态为一个衰减正弦（等价于一个高 Q 双二阶谐振器 `Biquad`），参数为 `(频率 f, 衰减/半衰期 τ, 增益 g)`。撞击冲量作为**激励脉冲**注入并联谐振器组，输出叠加即撞击音色；τ 决定余韵长度，随尺寸/材质变化。
- **模态数据来源（离线/资产期）**：可由几何特征模态分析（模态特征频率）、测量分解或作者手调得到，烘焙为紧凑模态表（`bevy_asset` 加载/热重载）。运行时**只查表 + 激励**，不做特征分解。
- **激励塑形**：接触点决定各模态激励权重（敲边缘 vs 敲中心谱形不同）；冲量能量缩放激励幅度；切向占比经一阶低通/高通塑形激励谱；可叠加极短噪声簇模拟接触瞬态（attack）。
- **RT 契约**：谐振器状态与模态表构造期预分配；激励为**写入历史缓冲的单次脉冲**，块内零分配、无 panic；模态数上限受 §32 `QualityGovernor` 分档（近处满模态、远处削减高频模态）。全部走 `bevy_math::ops` + 种子 RNG，逐样本可对拍。

### 47.3 连续接触：摩擦 / 滚动 / 滑动

- **摩擦/滑动**：以**滤波噪声**为源——相对切向速度→噪声增益 + 谱质心（快擦更亮），表面粗糙度→带宽；接触材质经 §47.5 决定共振着色（把噪声送入物体模态谐振器，得「被物体染色的摩擦声」）。速度归零则 §7 `Smoothed` 淡出，无爆音。
- **滚动**：粗糙表面滚动 = **接触脉冲序列**（脉冲率 ∝ 速度 × 表面颗粒密度）激励模态谐振器；速度变化连续改变脉冲率（音高感）与能量。脉冲相位由确定性种子驱动，可回放。
- **状态机**：接触总线维护 per-contact 状态（撞击→滚动→滑动→分离），连续量随物理量逐块更新，离散撞击经 §8 采样精确注入；连续声与撞击声共享同一模态谐振器组，避免二次维护。

### 47.4 颗粒合成（Granular，`GranularEngine`）

- **用途**：碎裂/沙砾/群体/材质细节等「云状」音色——从源材质（短波形或合成粒）以**颗粒云**渲染，颗粒率/长度/音高/声像/包络由 RTPC 与物理量（如碎裂能量）驱动。
- **确定性调度**：颗粒起点/参数由**种子 RNG**决定，颗粒经 §8 调度器采样精确排布；颗粒声音走**预分配颗粒语音池**（固定容量、超限窃取，复用 §25 语音管理），块内零分配。
- **与 Patch 复用**：颗粒引擎作为 §11 Patch 的一类可编译节点，编译后作为普通 `AudioNode` 嵌入运行时图；无独立解释器。

### 47.5 声学材质耦合（复用 `prism_material_pipeline`）

- 撞击/摩擦音色由**材质对**决定：材质携带模态表引用、摩擦谱模板、透射/吸收（与 §14 空间声学材质**同一字段总线**，见 §14 与关联文档 `prism_material_pipeline_design_zh.md`），避免视觉/物理/声学三套材质各自维护。
- 材质对（如「金属—石头」）经查表/插值得到合成参数；未命名材质对回退到类目默认，保证无洞。

### 47.6 编译、预算与确定性

- **预分配**：模态谐振器组、颗粒语音池、噪声源状态均构造期分配；RT 侧仅「激励 + 推进」，零分配/锁/panic，恪守 §21 RT 铁律。
- **预算联动**：同时发声的模态物体数、每物体模态数、颗粒密度受 §32 `QualityGovernor` CPU 预算闭环分档缩放；密集撞击经 §33 感知剔除/聚类；分档切换带滞回，避免抖动。
- **确定性**：激励能量映射、颗粒/滚动随机全走种子 RNG 与 `bevy_math::ops`，与物理确定性（§24）一致 → 相同碰撞序列产出相同样本，纳入 golden 对拍。

**借鉴 · 覆盖 · 超越**：借鉴 Wwise Impacter 的「冲量驱动撞击/材质合成」范式与经典模态/颗粒合成；覆盖为 §11 Patch 的可编译节点族（`ModalSynth`/`GranularEngine`）+ §8 采样精确注入 + §25 语音池；**超越**在于激励直接取自 `prism_physics`/§31 GPU 的**同一几何与碰撞真相**（而非中间件另建声学场景），并纳入统一编译图 + 确定性数学 + golden 对拍——跨系统一致、可逐样本回放。

---

## 48. 输出渲染链与母带交付档（下混矩阵 / 低频管理 / 动态范围与响度交付）

§13 解决「内容侧响度与动态」，§16 解决「空间化到床/对象」；但**从内部规范格式（planar f32，最高 7.1.4/HOA）到具体收听设备**这最后一公里——下混系数、低频管理、以及每台主机游戏都必带的「客厅/电视/夜间/耳机」动态范围档——需一条显式、可配置、确定性的输出渲染链。这是 AAA 出货硬指标（主机认证含响度与下混要求），也是「装样机 vs 真出货」的分野。本链位于 §13 母带之后、`prism_audio_device`（§22）设备回调之前。

### 48.1 输出格式协商与下混矩阵（`DownmixMatrix`，ITU-R BS.775）

- **规范内部格式**：图内部按最高目标布局（如 7.1.4）渲染；到设备布局经**显式下混矩阵**折算，而非隐式声道丢弃。
- **BS.775 系数**：标准折算系数（中置/环绕按 -3 / -4.5 / -6 dB 混入前置左右），能量守恒、相位一致；折降链 `7.1.4 → 7.1 → 5.1 → 立体声 → 单声道` 逐级可组合。
- **双耳路径**：立体声/多声道→双耳走 §16 HRTF/虚拟扬声器（§46.4），与扬声器下混同为 `DownmixMatrix` 的一档。
- **RT 契约**：下混矩阵构造期预算，RT 侧仅矩阵-向量乘，零分配；布局变更（拔插耳机）在块边界原子换矩阵（§21 图交换机制）。

### 48.2 低频管理（Bass Management / LFE 交叉，`BassManager`）

- **交叉分频**：小音箱档把全频声道 < 交叉频率（默认 80 Hz，可配）的低频分离并汇入 LFE/超低音；LFE 播放校准（影院 +10 dB 惯例，可配）。
- **耳机/立体声档**：无独立 LFE，LFE 内容按增益折回主声道，避免低频丢失。
- **相位安全**：分频用线性相位或匹配的 Linkwitz–Riley 交叉，避免交叉点相位抵消；系数构造期预算。

### 48.3 交付动态范围档（Output Presets：Home Theater / TV / Night / 耳机）

对齐所有主机游戏必带的收听模式（杜比/主机 UI 常见 Full/Standard/Night 三档思路）：

- **Home Theater（全动态）**：不额外压缩，保留 §13 HDR 全窗口，供安静环境 + 好设备。
- **TV / Standard（中度）**：适度压缩 + HDR 窗口收窄，抑制过大峰值、提亮对白，适配电视扬声器。
- **Night / Midnight（强压缩）**：强上抬安静段、强压过响段（枪爆），供夜间/公寓；配合对白锚定保清晰。
- **耳机档**：叠加双耳（§16）+ 适度限幅保护听力。
- **落地**：每档是 §13 `HDR 窗口 + 压缩器 + True-Peak 限幅` 的**参数化预设链**，运行时经命令环切换、§7 逐样本 `Smoothed` 过渡无爆音，可 golden 对拍；不新增 DSP 类型，只是参数档。

### 48.4 平台响度交付目标与对白锚定

- **分目标 LUFS**：游戏典型 -16 ~ -18、流媒体 -14、广播 -23/-24（EBU R128）、主机认证按平台要求；交付档选目标，§13 LUFS 积分器驱动归一。
- **对白锚定响度（Dialog-Anchored）**：以对白总线响度为锚做整体响度基准（对齐「对白清晰度优先」的出货实践），配合 §46.8 类别让路与 §13 sidechain 确保对白始终可懂。
- **合规计量回传**：Integrated/Short-term/Momentary LUFS 与 True-Peak 经 §26 遥测回传，供 QA 与 §38 远程工具核验是否满足目标。

### 48.5 设备协商与优雅降级

- `prism_audio_device`（§22，cpal/worklet）上报设备布局/采样率/空间能力（§16），输出链据此**自动选择**下混矩阵 + 低频管理 + 交付档；采样率不符走 §34 `Resampler` 分档重采样；欠载输出静音计数（§22），不阻塞。
- 平台原生空间后端（Windows Sonic/Tempest/Atmos/XR，§16）可用时，下混/双耳交给平台解码，本链仅做响度/动态交付档与合规计量。

**借鉴 · 覆盖 · 超越**：借鉴 BS.775 下混、影院低频管理与主机「客厅/电视/夜间」动态范围档；覆盖为 §13 母带之后的显式确定性末级（矩阵 + 交叉 + 参数化档）；**超越**在于全部走同一编译图预分配 + 逐样本 `Smoothed` + 确定性数学 + golden 对拍，且交付档与 §46.8 自动混音、§32 治理、§13 HDR 共享同一响度/动态决策面，而非散落在设备层的隐式硬编码。

---

## 49. 术语表


- **RT-safe**：实时安全，指音频回调线程可执行（零分配/锁/阻塞/panic）。
- **block / 块**：一次处理的定长样本帧数。
- **planar**：每通道连续存储（对比 interleaved 交织）。
- **Patch**：可编译成单个节点的内容侧程序化 DSP 子图（MetaSounds 式）。
- **zipper noise**：参数突变导致的爆音。
- **denormal**：接近零的次正规浮点，某些 CPU 上运算极慢，需 flush。
- **LUFS**：响度单位（BS.1770），感知响度测量。
- **True-Peak**：过采样后的采样间峰值。
- **HDR 音频**：动态响度窗口，突出前景声、压抑背景声。
- **RTPC**：实时参数控制（游戏量→音频参数）。
- **Modulation**：参数间与时变信号（LFO/包络/控制总线）的组合调制层。
- **Occlusion / Obstruction**：遮挡（直达+混响均挡）/ 障碍（仅直达挡）。
- **Aux Send**：向混响/效果返回总线的辅助发送。
- **HRTF**：头相关传输函数，双耳空间化基础。
- **Ambisonics / AmbiX**：全景声场表示（W/X/Y/Z…），AmbiX = ACN 通道序 + SN3D 归一化。
- **Doppler**：相对运动导致的频移。
- **FDN**：反馈延迟网络（混响结构）。
- **VBAP**：基于矢量基的幅度声像（多声道定位）。
- **Virtual Voice**：被淘汰但可复活的语音（继续推进/杀死/重启/按时长续播）。
- **Bank / SoundBank**：可加载/卸载的资产打包单元。
- **epoch 回收**：确认无 RT 引用后在任务线程释放旧对象的无锁回收策略。
- **ExecPlan / 执行计划**：编译期从图产出的扁平、定序、零分配的处理指令序列，RT 顺序执行。
- **缓冲活跃度分配**：用活跃区间分析 + 图着色为图的中间缓冲复用物理槽位（类比寄存器分配）。
- **就地处理（In-place）**：节点输入输出复用同一缓冲以省 copy，别名安全由编译期校验。
- **PDC（Plugin Delay Compensation）**：对并行路径插入整数样本延迟以对齐相位。
- **岛屿（Island）**：图中可并发执行的、无相互数据依赖的节点子集。
- **音频 LOD**：按重要度对声源缩放过采样/混响/空间化/控制率的分级质量策略。
- **QualityGovernor**：按 CPU 预算闭环调整音频 LOD 与语音数的运行时治理器。
- **掩蔽（Masking）**：频域上响声掩盖弱声的心理声学现象，用于感知剔除。
- **声源聚类（Source Clustering）**：把邻近相近的多个声源动态归并为少量代表性虚拟源。
- **WSOLA / 相位声码器**：变速不变调 / 变调不变速的时域/频域时间伸缩算法。
- **多相 FIR（Polyphase）**：高效任意比率重采样的滤波器组结构。
- **Programmer Sound / 程序化对白**：运行时按键值选择媒体注入语音的对白机制（FMOD/Wwise 式）。
- **Viseme / 口型**：与音素对应的口型帧，用于唇形/表情动画驱动。
- **Caption Sync / 字幕同步**：由播放头驱动、随媒体时间码上抛的字幕事件。
- **Haptics / 触感**：与音频同源生成的振动/力反馈输出（手柄/运动设备）。
- **Soundscape / 音景**：由环境状态驱动、程序化散布 one-shot 形成的低重复环境声床。
- **WAAPI / 授权 API**：远程连接运行引擎做只读遥测与白名单写命令的工具协议范式。
- **Builder API / 运行时构图**：在任务线程增量拼装或改写内容 DSP 图（Patch），编译后原子热切换到运行时图（UE5 MetaSound Builder API 式）。
- **Audio Gameplay Volume**：体积驱动的室内外/混响/衰减覆盖与门户连通模型（UE5 5.3+，取代旧混响体）。
- **片段式交互流（Interactive Stream）**：以剪辑为节点、以带过渡类型（next-beat/next-bar/marker 等）的转移为边的交互音乐图（Godot AudioStreamInteractive 式）。
- **延迟补偿播放头**：向 gameplay 暴露的、扣除输出延迟并加回自上次混音以来时间的“画面对齐”播放位置。
- **Polyphonic 声源**：单逻辑声源动态复用池内多条 voice 播放重叠实例（Godot AudioStreamPolyphonic 式）。
- **波动声学（Wave Acoustics）**：以波动方程数值求解声传播，物理正确处理低频/绕射/耦合，对比几何射线的高频近似。
- **ARD（自适应矩形分解）**：把场景划为矩形子域内解析求解 + 界面耦合的高效波动求解法，Project Acoustics/Triton 采用。
- **感知参数场（Perceptual Parameter Field）**：从烘焙 IR 提取的紧凑声学参数（遮挡/衰减 RT60/到达方向/湿干比）按探针网格存储、运行时插值。
- **混合传播（Hybrid Propagation）**：波动档负责低频/绕射/混响、几何或 GPU 档负责高频/动态增量，二者参数域叠加交叉淡化。
- **Codec / 编解码**：PCM/ADPCM/Vorbis/Opus/FLAC 等音频编码，按用途（音效/音乐/对白/归档）分档。
- **对象音频（Object-based / Bed+Objects）**：声道床 + 携位置元数据的动态对象混音模型（杜比 Atmos / MPEG-H）。
- **MPEG-H 3D Audio**：对象 + 声道 + HOA 的沉浸式音频标准（ATSC 3.0 广播）。
- **SOFA（AES69）**：空间定向声学数据格式，承载测量 HRIR/HRTF 数据集。
- **个性化 HRTF**：按用户人体测量/校准选取最匹配的测量 HRTF 集以提升双耳定位（本文档走经典数据驱动、无 ML）。
- **VOIP / 实时语音**：网络实时语音通信；本引擎作为统一图一等声源统一空间化（§45）。
- **AEC（回声消除）**：以扬声器输出为参考、经典自适应滤波（NLMS/频域）抵消麦克风回声，含双讲检测与残余抑制（无 ML）。
- **NS（噪声抑制）**：谱减 / 维纳滤波 + 噪声底估计的经典稳态噪声抑制（无 ML）。
- **AGC（自动增益）**：目标响度归一 + 前瞻限幅保护的在线增益控制。
- **VAD（语音活动检测）**：能量/过零率/谱平坦度的经典判决，驱动静音门控与感知剔除。
- **PLC（丢包隐藏）**：解码侧用前一帧激励外推填补丢包，避免爆裂断续。
- **抖动缓冲（Jitter Buffer）**：按网络抖动自适应调整深度、在延迟与卡顿间权衡的接收缓冲。
- **邻近语音（Proximity Chat）**：随距离渐入渐出的世界空间位置语音。
- **侧音（Side-tone）**：低增益回授本地处理后语音以减轻闭耳耳机闷耳感。
- **分区卷积 / UPOLS**：把长 IR 切块、频域重叠保存的实时卷积法；非均匀分区（NUPOLS）头小块低延迟、尾大块低 CPU，实现零额外延迟长卷积。
- **SIMD 内核 / `DspKernel`**：对 planar `&[Sample]` 批处理的热原语抽象，标量默认档 + SSE/AVX/NEON 显式档，须逐样本一致方可启用。
- **FTZ / DAZ**：Flush-To-Zero / Denormals-Are-Zero，CPU 浮点标志，回调边界置位以避免次正规数拖慢递归 DSP；核心 crate 另走可回放的算法挤零档。
- **声场旋转（Soundfield Rotation）**：对 HOA 场施加球谐旋转矩阵实现听者转头，O(阶²) 且无插值缝，优于逐源重选 HRIR。
- **双频段解码（Dual-Band Decode）（✅ `hoa_decode`）**：Ambisonics 低频 in-phase/basic 保相位、高频 max-rE 保能量矢量，分频合并提升定位稳健性。
- **虚拟扬声器双耳（✅ `hoa_binaural`）**：Ambisonic → 虚拟扬声器方位 → 各方位 HRIR 卷积求和的布局无关双耳渲染。
- **跨听渲染 / 串音消除（Transaural / Crosstalk Cancellation）（✅ `transaural`）**：用扬声器回放双耳信号时，递归消除对侧扬声器到耳的声学串音，使双耳空间线索在扬声器上重建；`CrosstalkCanceller` 精确逆对称串音矩阵、`CrosstalkParams::from_geometry` 由扬声器半角 Woodworth ITD 推导路径。
- **可转向虚拟传声器/模态波束成形（✅ `hoa_beamform`）**：把 HOA 场收成单路可转向 mono 虚拟传声器，`BeamPattern` basic/max-DI/max-rE/in-phase 四族，look 轴响应归一为 1，用于点采样隔离/到达方向探测/定向发送。
- **近场补偿 / 双耳视差**：<1m 声源的近场增益/ILD 修正与分耳入射方位计算，改善贴脸音定位可信度。
- **空气吸收（ISO 9613-1 式）**：随距离/介质对高频施加频变大气衰减，使远处声源自然发闷。
- **代数环 / 单样本延迟（`z⁻¹`）**：图内反馈须经显式单样本延迟节点打断成合法 DAG；未经延迟的真环编译期报 `Cycle`。
- **AISAC**：CRIWARE ADX2 的多维交互控制曲线簇，一个游戏量扇出驱动多参数，映射到本引擎 §12 多目标控制总线。
- **REACT / 声明式自动混音**：类别间「谁让路给谁」的声明式规则，编译为 §12 sidechain 连线 + §13 ducking，与 §32 治理共享决策面。
- **Category 音量树**：正交于总线树的类别增益/上限层（对白/UI/武器/环境/音乐…），承载类别级让路与响度合规。
- **模态合成（Modal Synthesis）**：以一组衰减正弦（高 Q 谐振器）叠加合成撞击/材质音色，冲量作为激励脉冲。
- **接触事件总线（Contact Event Bus）**：`prism_physics` 碰撞/接触流携样本偏移注入合成激励的采样精确通道。
- **颗粒合成（Granular Synthesis）**：以大量短颗粒云合成碎裂/群体/材质细节，颗粒参数由种子 RNG 与物理量驱动。
- **冲量激励（Impulse Excitation）**：碰撞冲量映射为激励能量/谱形，注入模态谐振器组。
- **摩擦/滚动合成**：滤波噪声（摩擦）与接触脉冲序列（滚动）随相对速度连续演化的接触声。
- **下混矩阵（Downmix Matrix / BS.775）**：多声道折降到目标布局的显式能量守恒系数矩阵（中置/环绕 -3/-4.5/-6 dB）。
- **低频管理（Bass Management / LFE 交叉）**：把全频声道低频分频汇入 LFE/超低音，耳机档则折回主声道。
- **交付动态范围档（Night / TV / Home Theater）**：主机游戏必带的收听模式参数化预设链（HDR 窗口+压缩+限幅）。
- **对白锚定响度（Dialog-Anchored Loudness）**：以对白总线响度为整体响度基准锈，保对白清晰度优先。
- **Linkwitz–Riley 交叉**：低频管理分频用的相位匹配分频器，避免交叉点相位抵消。
- **声学材质对（Material Pair）**：撞击/摩擦双方材质的组合，查表/插值得合成参数，与视觉/物理材质同一字段总线。
- **Wwise Impacter（参照）**：冲量驱动的撞击/材质程序化合成中间件，本引擎只借范式、无源码衍生。
- **感知瞬态（attack transient）**：撞击起始的极短噪声簇，提升撞击“硬度”辨识度。
- **折降链（Fold-down）**：`7.1.4→7.1→5.1→立体声→单声道` 的逐级可组合下混序列。
- **xrun / 欠载**：设备回调未能按时取得整块样本导致的断流；隐藏策略为淡出保持静音、恢复后淡入（PLC 式），并计数上报。
- **设备热插拔跟随**：默认设备变更/拔插时在非 RT 侧重开流并按新率/布局续跑，RT 侧从不做设备枚举或分配。
- **时钟漂移 / `DriftResampler`**：设备晶振与引擎标称采样率的 ppm 级长期偏差，由缓冲水位闭环驱动的有界异步多相重采样微调纠正（区别于开流即拒绝不匹配率）。
- **采样精确 seek / scrub**：把播放头跳到任意样本位置，内存源保留分数相位、流式源重填预取缓冲，与循环点/事件同一样本网格对齐（Wwise/FMOD `setPosition` 式）。
- **无分配守卫（no-alloc guard）**：测试/debug 下包裹全局分配器，进入 `process` 期间任何堆分配即 panic，把“热路径零分配”从声明变为机器可验证约束。
