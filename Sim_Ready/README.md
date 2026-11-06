# Sim_Ready

独立维护的离线资产工具。当前只生产正式 PBF 的 `static` 和
`stairs-rigids`，不导入 demo、归档或运行应用。PBF 专用 pipeline
可以读取 PBF 配置及调用其 VM 预计算；底层工具不依赖使用方。

## 使用

在仓库根目录、已激活 WarpSim 环境时运行：

```powershell
python -m pip install -r Sim_Ready/requirements.txt
python -m Sim_Ready.pipelines.pbf --scene static
python -m Sim_Ready.pipelines.pbf --scene stairs-rigids
python -m Sim_Ready.pipelines.pbf --scene stairs-rigids --particle-radius 0.0092
python -m Sim_Ready.pipelines.pbf --scene stairs-rigids --config PBF/config/default_para.json
```

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
python -m Sim_Ready.pipelines.pbf --scene stairs-rigids --rebuild --output-dir outputs/pbf-newton/rebuilt-assets
```

## 调用流程与职责

```text
pipelines/pbf.py
  配置、源摘要和生成版本 → 缓存命中／验证复用／完整生成
  geometry：Newton 地形 → 坐标与接缝对齐 → 兔子、容器布尔组合
  validation：实体、空腔及几何检查
  collision：兔子 CoACD 凸块；台阶和墙使用同源 boxes
  sdf：静态容器／局部刚体的 SDF 烘焙
  PBF.boundary_model：预计算 VM，写入暂存 NVDB
  验证 OBJ/SDF/碰撞附件 → publish：整组发布与失败回滚
```

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
python -m unittest tests.pbf.test_static tests.pbf.test_scene_assets tests.pbf.test_pressure tests.pbf.test_rigid_collisions tests.pbf.test_coupled_lifecycle
```

来源和许可信息见 `PROVENANCE.md`；当前阶段不提供机器人、关节或 MimicKit 导入。
