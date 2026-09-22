# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、**带航道/海缆走廊/生态缓冲区等禁建多边形及各自安全净距的可行域模型**、间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束（含禁建区校验与不可行场景）、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 禁建区与安全净距配置

租赁边界内部可以配置多个**允许凹陷**的禁建多边形，每个区域单独指定
风轮边缘安全净距。在 JSON 配置中通过 `exclusion_zones` 给出：

```json
{
  "boundary_type": "rectangular",
  "boundary_params": {"width": 4000, "height": 4000},
  "lease_setback": 0.0,
  "exclusion_zones": [
    {
      "name": "主航道",
      "kind": "shipping_lane",
      "setback": 200.0,
      "vertices": [[-2000, -300], [2000, -300], [2000, 100], [-2000, 100]]
    },
    {
      "name": "35kV海缆走廊",
      "kind": "cable_corridor",
      "setback": 100.0,
      "vertices": [[900, -2000], [1100, -2000], [1100, 2000], [900, 2000]]
    },
    {
      "name": "近岸生态缓冲区",
      "kind": "ecological_buffer",
      "setback": 150.0,
      "vertices": [[-2000, 1200], [-600, 1050], [-400, 2000], [-2000, 2000]]
    }
  ]
}
```

- `kind` 可选 `shipping_lane`（航道）、`cable_corridor`（海缆走廊）、
  `ecological_buffer`（生态缓冲区）或 `other`，决定图例配色。
- `setback` 是**风轮边缘**到该区边缘的安全净距（米），各区可以不同；
  程序按最大风轮半径把净距换算到塔位约束。
- `vertices` 为 `(N, 2)` 坐标数组，允许凹陷但不得自相交、重合或共线；
  非法几何、负净距、重名、与租赁边界完全不相交等情况会在加载时
  抛出明确的 `GeometryValidationError`。

可行域规则（租赁边界 − 全部禁建区及净距）贯穿随机采样、规则基线、
间距修复以及 GA/PSO 的惩罚与修复全过程。优化器最终只返回通过
完整校验的布局；当机位过密或可行域被禁建区切碎到无法容纳时，
程序在有限时间（采样/修复均有次数与时间预算）内失败，退出码为 `2`，
并在控制台和输出目录的 `failure.json` 中给出**具体约束原因**
（已布置台数、最小间距、可行域面积、涉及的禁建区及净距等），
而不会卡住或返回违规布局。

图形与 `results.json` 会区分租赁边界（绿）、各类禁建区（红色系
斜线填充 + 虚线净距轮廓）与最终机位（蓝色圆或尾流损失色阶）。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件（示例已含三类禁建区），再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。
