"""生成校准核对所需的三套合成资料（确定性、无第三方依赖）。

资料仅用于在明确的抽样假设下核对算法：含相关边际的有界熵校准、权重触界、
满样本可行但 delete-one-PSU 副本不可行、以及边际总量不一致。所有厂区、
设备与结果均为合成，不代表任何真实机器人总体。

样本为 2 层 × 3 PSU × 2 行共 12 行，east/low 格分散在 4 行（两个层各两行）。
紧界行 H1P10 的无约束期望权重低于其比例下界，熵解把它压到下界并在格内
重新分配；由于该格跨多个 PSU，删任一 PSU 后比例界（施加在放大后的基础
权重上）仍留有缓冲，所有 jackknife 副本可行。把紧界继续抬高使该格下界
之和超过格目标时，同一目标即变为带界不可行。
"""
import json

SOLVER = {"tolerance": 1e-08, "max_iterations": 10000}
Z = 1.95996398454

# (id, stratum, psu, base_weight, outcome, region, load, type,
#  可行资料下界比例, 不可行资料下界比例, 可行上界比例, 不可行上界比例)
SAMPLE_ROWS = [
    # 紧界行：可行资料 lower=1.1（权重下界 2.2，无约束期望约 1.25）→ 贴下界；
    # 删同层其他 PSU 时该行基础权重放大 1.5 倍，下界 3.3 加同格其余下界 1.5
    # 仍不超过格目标 5，所有副本可行。
    ("H1P10", "H1", "H1P1", 2.0, 1, "east", "low", "A", 1.1, 0.25, 4.0, 0.9),
    ("H1P11", "H1", "H1P1", 2.0, 0, "east", "high", "A", 0.25, 0.25, 4.0, 0.9),
    ("H1P20", "H1", "H1P2", 2.0, 1, "west", "low", "A", 0.25, 0.25, 4.0, 0.9),
    ("H1P21", "H1", "H1P2", 2.0, 0, "west", "high", "B", 0.25, 0.25, 4.0, 0.9),
    ("H1P30", "H1", "H1P3", 2.0, 0, "east", "low", "A", 0.25, 0.25, 4.0, 0.9),
    ("H1P31", "H1", "H1P3", 2.0, 1, "east", "high", "A", 0.25, 0.25, 4.0, 0.9),
    ("H2P10", "H2", "H2P1", 2.0, 0, "east", "low", "A", 0.25, 0.25, 4.0, 0.9),
    ("H2P11", "H2", "H2P1", 2.0, 1, "east", "high", "A", 0.25, 0.25, 4.0, 0.9),
    ("H2P20", "H2", "H2P2", 2.0, 0, "west", "low", "A", 0.25, 0.25, 4.0, 0.9),
    ("H2P21", "H2", "H2P2", 2.0, 0, "west", "high", "B", 0.25, 0.25, 4.0, 0.9),
    ("H2P30", "H2", "H2P3", 2.0, 0, "east", "low", "A", 0.25, 0.25, 4.0, 0.9),
    ("H2P31", "H2", "H2P3", 2.0, 1, "west", "high", "B", 0.25, 0.25, 4.0, 0.9),
]

# total=24。格目标：east/low=6、east/high=8、west/low=4、west/high=6。
FEASIBLE_MARGINS = {
    "region": {"east": 14.0, "west": 10.0},
    "load": {"low": 10.0, "high": 14.0},
    "type": {"A": 18.0, "B": 6.0},
}
# 边际自身能求和但与 total 不一致（load 合计 23 ≠ total 24）。
INCONSISTENT_MARGINS = {
    "region": {"east": 14.0, "west": 10.0},
    "load": {"low": 10.0, "high": 13.0},
    "type": {"A": 18.0, "B": 6.0},
}


def _rows(variant):
    rows = []
    for (rid, stratum, psu, base_weight, outcome, region, load, kind,
         lower_feasible, lower_infeasible, upper_feasible,
         upper_infeasible) in SAMPLE_ROWS:
        if variant == "feasible":
            lower, upper = lower_feasible, upper_feasible
        else:
            # 不可行资料：上界比例统一压到 0.9（权重上界 1.8）。east 有 7 行，
            # 上界之和 12.6 < east 边际目标 14；无界条件下原目标仍可满足，
            # 因此这是纯粹的“带界不可行”。
            lower, upper = lower_infeasible, upper_infeasible
        rows.append(
            {
                "id": rid,
                "stratum": stratum,
                "psu": psu,
                "base_weight": base_weight,
                "outcome": outcome,
                "features": {"region": region, "load": load, "type": kind},
                "lower": lower,
                "upper": upper,
            }
        )
    return rows


def _material(margins, variant):
    return {
        "sample": _rows(variant),
        "target": {"total": 24, "margins": margins},
        "solver": dict(SOLVER),
        "confidence_z": Z,
    }


MATERIALS = {
    "bound-active.json": _material(FEASIBLE_MARGINS, "feasible"),
    "bounded-infeasible.json": _material(FEASIBLE_MARGINS, "infeasible"),
    "inconsistent-totals.json": _material(INCONSISTENT_MARGINS, "feasible"),
}


def write(directory):
    paths = []
    for name, material in MATERIALS.items():
        path = directory / name
        path.write_text(
            json.dumps(material, ensure_ascii=False, indent=2) + "\n",
            encoding="utf8",
        )
        paths.append(path)
    return paths
