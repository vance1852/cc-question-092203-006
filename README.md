# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

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

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 禁建区与可行域

最新勘测图中的航道、海缆走廊和生态缓冲区通过配置文件的 `exclusions` 字段描述。每个禁建区是一个**可为凹陷**的多边形，并可配置各自独立的安全净距（风轮外缘至禁建区边缘的距离，米）：

```json
{
  "boundary_type": "rectangular",
  "boundary_params": {"width": 3500, "height": 3500},
  "exclusions": [
    {
      "name": "航道",
      "setback": 200.0,
      "vertices": [[-1750, -400], [1750, -300], [1750, 0], [-1750, -100]]
    },
    {
      "name": "生态缓冲区",
      "setback": 150.0,
      "vertices": [[-1600, 700], [-900, 700], [-900, 1100], [-500, 1100], [-500, 1600], [-1600, 1600]]
    }
  ]
}
```

可行域 = 租赁多边形内部 − 每个禁建区向外缓冲 `setback + 风轮半径` 的范围。该规则贯穿：

- 随机采样（可行域拒绝采样，总尝试次数有上限）；
- 规则网格/交错基线（网格点与随机补点都必须可行）；
- 间距修复（先投影回可行域，再推开、重定位，预算有界）；
- GA 与 PSO 的初始种群、惩罚（按违反深度计）、变异后修复；
- 图形输出（租赁边界=绿色实线，禁建区=红色系斜纹多边形+虚线净距线，机位=带编号圆）与 `results.json`（含各禁建区顶点、净距与可行域摘要）。

几何数据在加载时严格校验：顶点数、NaN/Inf、重复点、退化（零面积）、自相交，以及禁建顶点必须落在租赁边界内；非法数据直接报错退出（退出码 3），不会带病运行。

当方案过于密集或根本不可行时，程序在有限时间内失败并给出**具体约束原因**（哪台风机、违反哪个禁建区/租赁边界/最小间距、违反深度多少米、可行域命中率），退出码 2，而不会卡住或返回违规布局。

