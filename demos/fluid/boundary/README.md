# Boundary 场景生成与合成

`creator1.py` 生成并导出七种程序化地形；`scene_creator.py` 自动生成阶梯、放置 OBJ 模型，并导出布尔并集，可选简化副本。两套命令可独立运行，场景合成器会自行生成阶梯。

以下命令均在项目根目录运行，使用已配置的 Conda 环境：

```powershell
conda activate warplab
```

地形生成器依赖 `numpy`、`trimesh`。场景合成还需要 `manifold3d`、`pillow`；简化统计需要 `scipy` 或 `networkx`。当前 Conda `warplab` 环境已满足运行需求。

输入模型放在 `Assets/origin/`；程序化地形输出到 `Assets/procedural/`，合成场景输出到 `Assets/output/`。

## 默认运行命令

生成全部七种程序化地形（推荐命令，显式选择 `all`）：

```powershell
python demos/fluid/boundary/creator1.py --terrain all
```

导出 Y-up 的 `flat.obj`、`pyramid_stairs.obj`、`random_grid.obj`、`wave.obj`、`box.obj`、`gap.obj`、`heightfield.obj` 到 `Assets/procedural/`。其中高度场使用内置示例，随机地形每次可不同。无需输入 OBJ。

`all` 是 `--terrain` 的取值。省略 `--terrain` 时程序实际默认生成 `wave`。也可通过包入口生成全部地形：

```powershell
python -m demos.fluid.boundary.creator1 --terrain all
```

生成默认阶梯与奶牛合成场景：

```powershell
python -m demos.fluid.boundary.scene_creator
```

默认阶梯区域为 6×4 米，台阶宽 0.25 米、高度增量 0.05 米，中心平台宽 1 米。奶牛缩放为 0.5，自动放置在中央平台并下沉 0.02 米。默认导出 Y-up 场景：

- `Assets/output/stairs_scene.obj`：原始拼接对照，保留内部面。
- `Assets/output/stairs_scene_bool.obj`：实体布尔并集。

启用简化并另存 `stairs_scene_bool_simplified.obj`：

```powershell
python -m demos.fluid.boundary.scene_creator --simplify
```

直接运行脚本也支持：

```powershell
python demos/fluid/boundary/scene_creator.py
```

## 地形生成 CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--terrain TYPE` | `wave` | 选择 `flat`、`pyramid_stairs`、`random_grid`、`wave`、`box`、`gap`、`heightfield`；`all` 分别导出七种地形。 |
| `--up-axis Y\|Z` | `Y` | 输出的向上轴；`Z` 保留原始地形坐标。 |
| `--seed INT` | 不固定 | 随机地形种子；用于 `random_grid` 或 `all`，固定整数可复现。 |
| `--heightfield FILE.npy` | 内置隆起和凹坑示例 | 高度场二维数组；用于 `heightfield` 或 `all`。第一维对应 X，第二维对应 Y，数值单位米。 |
| `--ground-z FLOAT` | 最低地表高度减 `0.3` 米 | 高度场底面高度，采用轴转换前的 Z-up 坐标；须严格低于最低地表。用于 `heightfield` 或 `all`。 |
| `-h` / `--help` | — | 查看完整帮助。 |

可复现的全部地形，导出为 Z-up：

```powershell
python demos/fluid/boundary/creator1.py --terrain all --seed 42 --up-axis Z
```

只生成随机地形：

```powershell
python demos/fluid/boundary/creator1.py --terrain random_grid --seed 42
```

使用自定义高度场（文件路径相对于当前工作目录）：

```powershell
python demos/fluid/boundary/creator1.py --terrain heightfield --heightfield height.npy
```

高度数组必须为有限数值二维数组，每个方向至少有 2 个采样点。当前高度场重复生成 6×6 块，每块 6×4 米，总范围为 36×24 米；其他类型各生成一块 6×4 米地形。尺寸、台阶和波浪等参数在 `creator1.py` 对应的 `generate_*()` 函数中修改。

输出目录固定为 `Assets/procedural/`，同名文件会覆盖。`flat` 和 `wave` 是开放表面；箱体类地形和多块高度场为拼接网格，导出过程不执行布尔并集，不能将所有输出直接视为单一封闭实体。

## 场景合成 CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--obj PATH` | `Assets/origin/spot_triangulated_good.obj` | 输入 OBJ 模型。 |
| `--output PATH` | `Assets/output/stairs_scene_bool.obj` | 布尔结果 OBJ 路径。 |
| `--raw-output PATH` | `Assets/output/stairs_scene.obj` | 原始拼接对照 OBJ 路径。 |
| `--simplify` / `--no-simplify` | 关闭 | 开启或显式关闭简化副本导出。 |
| `--simplify-tolerance FLOAT` | `0.001` | 简化表面容差，单位米，仅启用简化时生效。 |
| `--simplified-output PATH` | 与布尔输出同目录，文件名为 `<stem>_simplified.obj` | 简化副本路径，须与 `--simplify` 一起使用。 |
| `--scale FLOAT` | `0.5` | 输入模型的尺寸倍率，须大于零。 |
| `--yaw FLOAT` | `0.0` | 绕场景竖直 Z 轴旋转的角度，单位度。 |
| `--obj-up-axis Y\|Z` | `Y` | 输入 OBJ 的向上轴。 |
| `--up-axis Y\|Z` | `Y` | 输出场景的向上轴。 |
| `--position X Y Z` | 自动放置 | 指定模型包围盒底部中心，使用 Z-up 坐标，单位米；指定后不再叠加 `--sink`。 |
| `--sink FLOAT` | `0.02` | 自动放置时下沉的深度，单位米，须非负。 |
| `-h` / `--help` | — | 查看完整帮助。 |

表中的默认资产路径相对于本目录；用户传入的相对路径相对于运行命令时的工作目录。输出会覆盖同名文件，输入和各输出路径必须互不相同。

自定义模型和输出的示例：

```powershell
python -m demos.fluid.boundary.scene_creator --obj demos/fluid/boundary/Assets/origin/bunny_repaired.obj --scale 5 --yaw 30 --up-axis Z --simplify --output outputs/boundary/bunny_scene.obj --raw-output outputs/boundary/bunny_raw.obj
```

输入模型必须是封闭、朝向一致的正体积实体；程序会焊接重复顶点，但不会自动补洞。默认奶牛资产需在本地存在，也可用 `--obj` 指定其他模型。

阶梯尺寸和台阶参数在 `scene_creator.py` 的 `generate_stairs()` 中设置，当前没有对应 CLI 参数。放置计算统一使用 Z-up 坐标，导出时按 `--up-axis` 转换整个场景。
