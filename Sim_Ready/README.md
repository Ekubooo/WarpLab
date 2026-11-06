# Sim_Ready

独立维护的离线资产工具。当前预设只生产正式 PBF 的 `static` 和
`stairs-rigids`，不导入 demo、归档或运行应用。PBF 专用 preset
可以读取 PBF 配置及调用其 VM 预计算；底层工具不依赖使用方。

## 使用

在仓库根目录、已激活 WarpSim 环境时运行：

```powershell
python -m pip install -r Sim_Ready/requirements.txt
python -m Sim_Ready.pipelines.preset.pbf --scene static
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --particle-radius 0.0092
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --config PBF/config/default_para.json
```

### 新电脑首次还原

Git 只提供代码、代码中的场景配方、依赖版本和记录，不提供模型或发布资产。
先安装 `PBF/requirements.txt` 和本包 `requirements.txt` 中的依赖，再手动创建
`Sim_Ready/Assets`，从自己的备份／网盘放入 **`bunny_flat_bottom.obj`**。
需要的是现有处理后的兔子，不是任意同名或其他版本的 Stanford Bunny。

源 OBJ 的 SHA-256：
`5c2fc17c5e851b02dd5503ac50bd6456dbb09959d90492a784c3481cef22d529`。
可用 PowerShell `Get-FileHash Sim_Ready/Assets/bunny_flat_bottom.obj -Algorithm SHA256`
核对。来源信息见 `PROVENANCE.md`。当前场景不需要 Spot，也不需要手动提供盒子、球或地形。

```powershell
python -m pip install -r PBF/requirements.txt -r Sim_Ready/requirements.txt
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids
python -m PBF.app --scene stairs-rigids --cuda-graph
```

空资产目录、空缓存也能用普通准备命令完整生成；`--rebuild` 则明确跳过复用。
默认参数与实例继续由代码描述，不新增场景 JSON，不需要修改代码或读取流体参数 JSON
就能还原默认场景。已有流体 JSON 保留为副本／记录，只有显式 `--config` 才读取。

| 参数 | 默认值／含义 |
| --- | --- |
| `--scene` | 必填，`static` 或 `stairs-rigids` |
| `--config` | 使用现有 PBF JSON；与 `--particle-radius` 互斥 |
| `--particle-radius` | 未指定时采用 PBF 默认半径 0.0125 m |
| `--device` | `cuda:0`；SDF/VM 生成需要 CUDA |
| `--output-dir` | 仓库内 `PBF/Assets` |
| `--cache-dir` | 仓库内 `.cache/sim-ready/pbf` |
| `--rebuild` | 跳过缓存及已发布资源复用，从源几何完整生成 |

首次迁移优先验证、复用现有正式资产，保持原 OBJ、SDF、凸块和元数据。
缺少完整资源且没有缓存时自动离线生成。损坏或摘要不一致的资源报错，
可用 `--rebuild` 明确重新生成。生产、缓存检查与发布均发生在此离线命令中。

```powershell
# 在独立目录检查完整重建，不覆盖当前基准。
python -m Sim_Ready.pipelines.preset.pbf --scene stairs-rigids --rebuild --output-dir outputs/pbf-newton/rebuilt-assets
```

## 调用流程与职责

```text
pipelines/preset/pbf.py：场景配方、缓存选择与整组发布
  ├─ pipelines/container.py
  │    prepare：地形、兔子放置、可见网格和对齐后的碰撞 boxes
  │    build：容器布尔组合 → 空腔校验 → 模型凸块 → SDF 与暂存 OBJ
  └─ pipelines/rigid.py
       prepare：按尺寸创建局部基础形状
       build：完整实体校验 → 按需凸块 → 局部 SDF 与暂存 OBJ
  → preset 生成 JSON/碰撞附件、调用 PBF 的 VM 预计算
  → 验证全部产物 → publish 整组发布与失败回滚
```

两条下层 pipeline 只接收数据、参数、暂存路径和设备，返回 OBJ/SDF 路径与凸块；
不导入 PBF、不选择消费者、不独立发布。基础形状无需凸分解；`rigid.build`
仅在调用者提供 `collision_cache` 时生成代理。底层几何、检查、烘焙、NVDB 和
发布功能继续共用。旧 `pipelines/pbf.py` 已移除，没有兼容转发或新增下层 CLI。

`config.py` 只含几何和采样参数；`asset_io.py` 提供内容摘要，
`nvdb_io.py` 负责轻量 NVDB 读写。`PBF/scene_data.py` 保存两边共用的
基础刚体尺寸，实例、位置和密度仍由 PBF 运行场景持有。

地形直接调用公开 `newton.Mesh.create_terrain()`；不复制 Newton 生成器。
当前台阶、轴向转换、float32 接缝对齐、兔子组合和容器参数保持原样。
兔子碰撞保持 CoACD 1.0.14、seed=0、实际尺度阈值 0.005 m、最多 32 块、
每块最多 64 顶点及原简化参数。块数上限可能使近似误差超过目标阈值，
不把阈值当成误差保证；新生成诊断留在离线缓存。

## 产物与运行边界

- `stairs_scene.obj/.nvdb/.json/.collision.npz`：可见阶梯兔子、容器 SDF/VM、
  24 个静态 boxes 与兔子凸块。静态发布也保留完整静态碰撞附件。
- `Rigid/stairs_box.*`、`Rigid/stairs_sphere.*`：完整网格、局部 SDF/VM 和元数据；
  基础碰撞形状和质量惯量仍由运行时 Newton 原生 shape 产生。
- 源模型在本包 `Assets`；产物默认在 `PBF/Assets`；缓存不是运行资源。
- 两个资产目录均从 Git 忽略，包括源 OBJ、生成 OBJ/NVDB、JSON、NPZ；
  `PBF/Assets/.gitignore` 自身保留。忽略不等于删除，已跟踪文件仍需用户自行处理。
  资产目录外的代码、依赖、来源记录和 `PBF/config` 中的流体 JSON 不受影响。
- 缓存包含源内容、几何、采样、生成版本和依赖版本。密度、时间步、压力迭代
  不改变几何缓存。小于已有支持范围的半径变化只影响 VM。
- 默认静态体素 0.0075 m、动态体素 0.005 m，支持半径至少 0.05 m。
  超过时离线重建；运行时仍只允许 VM 失配重算并写回同一个 NVDB。
- SDF 内容身份排除 NVDB 多 grid 包装字段，不以整个可变 NVDB 的摘要判断几何。
- 发布前停止消费者；全部附件预备好后替换，处理到的 I/O 失败恢复旧文件。
  这不是跨文件原子可见或崩溃恢复协议，不支持并发发布。

首次迁移与重新烘焙分开验收：迁移产物要求一致；重烘焙可能因稀疏布局或
浮点计算产生差异，使用独立输出检查采样误差，不自动替换数值回归基准。

## 验证

```powershell
python -m unittest tests.sim_ready.test_pbf_assets -v
python -m tests.sim_ready.verify_build --rebuild
python -m tests.sim_ready.verify_clean_workspace
python -m unittest tests.pbf.test_static tests.pbf.test_scene_assets tests.pbf.test_pressure tests.pbf.test_rigid_collisions tests.pbf.test_coupled_lifecycle
```

`verify_clean_workspace` 只复制生产源码、现有共享 billboard 源码、验收脚本和
手动提供的兔子到独立目录，不复制旧发布资源、资产缓存或 demo/归档资产。
它检查隔离后的导入路径，完整生成后运行耦合、Graph/重置、实际 OGL 与 USD。

来源和许可信息见 `PROVENANCE.md`；当前阶段不提供机器人、关节或 MimicKit 导入。
