# sim_Ready 离线资产管线

将输入模型放置在程序化阶梯中央，输出一套同名文件：`NAME.obj` 是无墙景观，`NAME.nvdb` 是含四墙与顶盖的碰撞距离场。两者使用相同的 Y-up 世界坐标，单位为米。生成程序与仿真程序之间只交换文件，无相互导入或调用。

## 使用

在仓库根目录、配置了依赖的 Python 环境中运行：

```powershell
python -m demos.sim_Ready
python -m demos.sim_Ready --obj bunny_flat_bottom.obj --name stairs_scene --device cuda:0
python -m demos.sim_Ready --obj spot_triangulated_good.obj --name stairs_spot
python demos/sim_Ready/__main__.py --help
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--obj PATH` | `bunny_flat_bottom.obj` | 裸文件名从本目录的 `Assets` 读取；含目录的相对路径按工作目录解释，支持绝对路径 |
| `--name NAME` | `stairs_scene` | 两种输出共用的文件名主体，不含扩展名、目录或盘符 |
| `--obj-up-axis Y\|Z` | `Y` | 输入模型的向上轴，OBJ 自身通常不记录该信息 |
| `--sdf-voxel-size` | `0.0075` | 体素间距，米 |
| `--device` | `cuda:0` | 烘焙 CUDA 设备 |

路径根据脚本位置确定；从其他工作目录运行模块时，需要让仓库根目录处于 Python 模块搜索路径。帮助与参数解析不加载模型或初始化 CUDA。

依赖：`numpy`、`trimesh`、支持 `Mesh64` 的 `manifold3d` 和支持 CUDA NanoVDB 的 `warp`。运行测试另需 `scipy`。输入模型必须是封闭、朝向一致的正体积实体；不会自动补洞或裁平模型。

## 阶段与职责

```text
__main__.py：CLI 参数
└─ pipeline.py：流程与发布
   ├─ config.py：配置与静态目的地
   ├─ geometry.py：地形、模型定位、布尔和简化
   │  └─ terrain_generator.py：原样复制的地形源码
   ├─ validation.py：内存网格检查
   └─ sdf.py：CUDA 计算与原子保存

生成地形 → 统一 Y-up → 读取/缩放/放置模型
├─ 地形＋模型 → 布尔并集 → 简化 → output/NAME.obj
└─ 地形＋模型＋墙/盖/底座扩展 → 独立布尔并集 → 简化
   └─ 检查源网格及 float32 网格 → CUDA 烘焙 → output/NAME.nvdb
      └─ 发布同名 OBJ/NVDB 到其他配置目的地
```

景观先输出，随后构造和检查容器。两条布尔分支都使用未简化的基础几何；容器不使用简化景观作为输入。容器仅在内存中存在，不输出容器 OBJ、拼接对照或未简化副本。

## 几何规则

全部参数集中在 `config.GeometryOptions`，默认值为：

- 地形长宽 `3.1×1.6 m`，台阶宽 `0.15 m`、高度增量 `0.1 m`，中央平台 `0.7×0.7 m`。
- 底层地形为 Z-up，返回后以 `(x,y,z) → (x,z,W-y)` 转换；`W` 是原地形 Y 最大值。后续全部使用 Y-up。
- 模型先转换输入轴、绕 Y 旋转，再按 `0.95 × 平台短边 / 模型水平最大跨度` 等比缩放。默认 yaw 为零。支撑中心对齐平台中央，下沉 `0.003 m`；有平底时使用面积重心，否则使用下沉截面重心，最后退回最低接触点范围中心。
- 墙厚 `w=0.25 m`、内嵌 `δ=0.075 m`，内顶面 `Y=4.5 m`。长墙覆盖完整外长度，短墙长度为 `min(地形宽+δ, 外宽)`；墙顶进入顶盖 `δ`，墙、盖、底座扩展共用外轮廓。底座向下扩展 `max(w-0.15,0)`，上部进入原底座。
- 两条分支始终使用 Manifold 简化，容差 `0.001 m`，不平滑或重网格。

默认外轮廓约为 `(-0.175,-0.2,-0.175)` 至 `(3.275,4.75,1.775)`，空腔极值约为 `(0.075,0,0.075)` 至 `(3.025,4.5,1.525)`；空腔范围内部仍有台阶和模型。

检查包含有限坐标、面索引、退化与重复面、闭合边、顶点流形性、绕序、一个正向外壳和一个反向内腔，以及长方体统一外轮廓；float32 转换后再次检查。检查不构成通用自交证明，瘦长但非退化三角形仅告警，不自动调整内腔法线。

## 输出与失败处理

`config.OUTPUT_DIRECTORIES` 默认包含本目录 `output` 和静态路径 `PBF_ASSETS`。以后新增目的地，直接在该配置元组中追加路径；不需要新增 CLI 参数。默认根输出目录必须保留在元组中。

- 每次运行重新生成、检查和烘焙，同名覆盖；没有文件存在即复用的缓存逻辑。
- 本地景观在首阶段原子替换。后续检查或烘焙失败时，这份新景观保留，既有 NVDB 不变，消费者目录不发布新文件。
- NVDB 完整写入同目录临时文件后原子替换；生成成功才准备消费者目录的 OBJ/NVDB 副本与备份。
- 发布时发生可处理的 I/O 错误则回滚已替换文件。若回滚本身失败，保留恢复副本并报告位置。多个文件与目录之间无法保证同时可见，应在仿真停止时发布。
- 拒绝输出覆盖输入模型，包括符号链接或硬链接指向同一文件的情况。临时文件正常结束时清理。

Python API 支持独立配置对象：

```python
from demos.sim_Ready.config import GeometryOptions, BakeOptions
from demos.sim_Ready.pipeline import build_assets

result = build_assets(
    "bunny_flat_bottom.obj", name="stairs_scene",
    geometry_options=GeometryOptions(),
    bake_options=BakeOptions(voxel_size=0.0075), device="cuda:0",
)
```

## PBF 与归档

```powershell
python -m demos.fluid.PBF.render_opengl --device cuda:0
# 使用 Spot 时，先在 PBF.py 中设置 SDF_FILENAME = "stairs_spot.nvdb"
python -m demos.fluid.PBF.render_opengl --device cuda:0
```

PBF 仅从自身 `Assets` 根目录读取资源。由 `PBF.py` 模块级常量 `SDF_FILENAME = "stairs_scene.nvdb"` 选择资源；OpenGL 使用同一常量加载同名 OBJ。PBF 启动 CLI 不接受 SDF 文件名，加载仍在仿真构造函数内执行。USD 入口只要求 NVDB。渲染保持默认灰色、平面法线、单位变换；粒子初始化和物理配置保持原状。

旧 Boundary 目录和 PBF 的 `Assets/Boundary`、`Assets/Container` 保留为归档，不回退读取。新管线不导入任何归档模块；只有 `terrain_generator.py` 原样复制，保留其许可证与版权信息。

运行测试：

```powershell
python -m unittest discover -s tests/sim_ready -t . -v
python -m unittest discover -s tests/fluid/pbf -t . -v
```
