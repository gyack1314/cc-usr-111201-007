"""有界熵校准（raking）与分层 delete-one-PSU jackknife。

数学口径（与 docs/data-contract.md 一致）：

    最小化   f(w) = sum_i [ w_i*log(w_i/d_i) - w_i + d_i ]
    约束     M^T w = t        （总量 + 去冗余分类指示列）
             L_i <= w_i <= U_i，其中 L_i=lower_i*d_i，U_i=upper_i*d_i

对偶变量 lambda 下逐行闭式解为 w_i = clip(d_i*exp((M lambda)_i), L_i, U_i)，
因此在对偶空间最大化凹分片二次可微函数即可，牛顿步的 Hessian 为
-M^T diag(v) M（界内行 v_i=w_i，触界行 v_i=0）。模块只依赖标准库。

任何无法满足原目标与原比例界的情形都按独立状态上报，不会通过放宽边际
或事后截尾"凑出"一个解。
"""
import math
from collections import defaultdict

from .io import finite, unique

# 求解状态。前两类为成功，其余互斥的失败/诊断状态。
OPTIMAL = "optimal"
OPTIMAL_REDUNDANT = "optimal_redundant_constraints"
TARGET_INCONSISTENT = "target_inconsistent"
UNSUPPORTED_TARGET = "unsupported_target"
INFEASIBLE_DEPENDENT = "infeasible_dependent_constraints"
BOUNDED_INFEASIBLE = "bounded_infeasible"
NOT_CONVERGED = "not_converged"

SOLVED_STATUSES = frozenset((OPTIMAL, OPTIMAL_REDUNDANT))
# 残差判定之外的固定数值保护。
_EXP_LIMIT = 700.0
_ACTIVE_TOL = 1e-9


class CalibrationError(ValueError):
    """输入资料不满足校准前提（区别于可行域层面的诊断状态）。"""


class _SingularMatrix(Exception):
    """高斯消元发现矩阵奇异。"""


# --------------------------------------------------------------------------
# 小型线性代数原语（矩阵按行存储，M[i][j] 表示第 i 行第 j 列）
# --------------------------------------------------------------------------
def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(M, v):
    """M (n×m) 乘 v (m) -> n。"""
    return [_dot(row, v) for row in M]


def _vec_mat(v, M):
    """v (n) 乘 M (n×m) -> m，即 M^T v。"""
    m = len(M[0])
    out = [0.0] * m
    for vi, row in zip(v, M):
        if vi == 0.0:
            continue
        for j in range(m):
            out[j] += vi * row[j]
    return out


def _gaussian_solve(A, b, pivot_tol=1e-12):
    """带部分主元的高斯消元解 A x = b，奇异时抛 _SingularMatrix。"""
    n = len(A)
    a = [row[:] for row in A]
    x = b[:]
    scale = max((abs(v) for row in a for v in row), default=1.0)
    tol = pivot_tol * max(1.0, scale)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= tol:
            raise _SingularMatrix
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            x[col], x[pivot] = x[pivot], x[col]
        pivot_value = a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / pivot_value
            if factor == 0.0:
                continue
            a[r][col] = 0.0
            for k in range(col + 1, n):
                a[r][k] -= factor * a[col][k]
            x[r] -= factor * x[col]
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        rest = sum(a[i][j] * result[j] for j in range(i + 1, n))
        result[i] = (x[i] - rest) / a[i][i]
    return result


def _gram(M, diag=None):
    """返回 M^T diag(diag) M；diag 为 None 时单位权。"""
    m = len(M[0])
    H = [[0.0] * m for _ in range(m)]
    for i, row in enumerate(M):
        scale = 1.0 if diag is None else diag[i]
        if scale == 0.0:
            continue
        for j in range(m):
            if row[j] == 0.0:
                continue
            base = scale * row[j]
            for k in range(j, m):
                if row[k] != 0.0:
                    value = base * row[k]
                    H[j][k] += value
                    if k != j:
                        H[k][j] += value
    return H


# --------------------------------------------------------------------------
# 资料校验与设计矩阵
# --------------------------------------------------------------------------
def _validate(value):
    if not isinstance(value, dict):
        raise CalibrationError("资料顶层必须是对象")
    rows = value.get("sample")
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("sample 必须是非空列表")
    unique(rows, "id")
    target = value.get("target")
    if not isinstance(target, dict):
        raise CalibrationError("target 必须是对象")
    total = finite(target.get("total"))
    if total <= 0:
        raise CalibrationError("target.total 必须为正")
    margins = target.get("margins")
    if not isinstance(margins, dict) or not margins:
        raise CalibrationError("target.margins 必须是非空对象")
    feature_keys = None
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise CalibrationError("每行必须是对象")
        d = finite(row.get("base_weight"))
        if d <= 0:
            raise CalibrationError("base_weight 必须为正")
        lower = finite(row.get("lower"))
        upper = finite(row.get("upper"))
        if lower <= 0 or not math.isfinite(upper) or lower > upper:
            raise CalibrationError("比例界必须满足 0<lower<=upper 且 upper 有限")
        outcome = row.get("outcome")
        if type(outcome) is not int or outcome not in (0, 1):
            raise CalibrationError("outcome 必须是取值 0/1 的整数")
        for key in ("stratum", "psu"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise CalibrationError(f"{key} 必须是非空字符串")
        features = row.get("features")
        if not isinstance(features, dict) or not features:
            raise CalibrationError("features 必须是非空对象")
        if feature_keys is None:
            feature_keys = set(features)
        elif set(features) != feature_keys:
            raise CalibrationError("各行分类特征键集合必须一致")
        for name, category in features.items():
            if not isinstance(category, str) or not category:
                raise CalibrationError(f"特征 {name} 的取值必须是非空字符串")
        normalized.append(
            {
                "id": row["id"],
                "stratum": row["stratum"],
                "psu": row["psu"],
                "base_weight": float(d),
                "lower_ratio": float(lower),
                "upper_ratio": float(upper),
                "outcome": outcome,
                "features": dict(features),
            }
        )

    if set(margins) != feature_keys:
        raise CalibrationError("边际特征必须与样本特征键完全一致")
    target_tables = {}
    for name, table in margins.items():
        if not isinstance(table, dict) or not table:
            raise CalibrationError(f"边际 {name} 必须是非空对象")
        categories = sorted({r["features"][name] for r in normalized})
        if set(table) != set(categories):
            raise CalibrationError(f"边际 {name} 的类别集合必须与样本一致")
        cleaned = {}
        for category in categories:
            amount = finite(table[category])
            if amount < 0:
                raise CalibrationError(f"边际 {name}.{category} 不能为负")
            cleaned[category] = float(amount)
        target_tables[name] = cleaned

    margin_sum_issues = []
    for name, cleaned in target_tables.items():
        amount_sum = sum(cleaned.values())
        if abs(amount_sum - total) > 1e-12 * max(1.0, total):
            margin_sum_issues.append(
                {"margin": name, "sum": amount_sum, "total": total,
                 "difference": amount_sum - total}
            )

    solver = value.get("solver", {})
    if not isinstance(solver, dict):
        raise CalibrationError("solver 必须是对象")
    tolerance = finite(solver.get("tolerance", 1e-8))
    max_iterations = solver.get("max_iterations", 10000)
    if tolerance <= 0 or type(max_iterations) is not int or max_iterations <= 0:
        raise CalibrationError("solver.tolerance 与 max_iterations 必须为正")
    z = finite(value.get("confidence_z", 1.95996398454))
    if z <= 0:
        raise CalibrationError("confidence_z 必须为正")
    settings = {"tolerance": float(tolerance), "max_iterations": max_iterations, "z": float(z)}
    return normalized, float(total), target_tables, settings, margin_sum_issues


def build_spec(rows, target_tables):
    """确定特征顺序、类别顺序与每特征参考类（确定性排序，可被副本复用）。"""
    spec = {}
    for name in sorted(target_tables):
        categories = sorted(target_tables[name])
        spec[name] = {"categories": categories, "reference": categories[0]}
    return spec


def build_matrix(rows, total, target_tables, spec):
    """构造总量列 + 每特征去参考类指示列；返回 M、列名、目标向量。"""
    names = ["total"]
    targets = [total]
    for name in sorted(spec):
        info = spec[name]
        reference = info["reference"]
        for category in info["categories"]:
            if category == reference:
                continue
            names.append(f"{name}:{category}")
            targets.append(target_tables[name][category])
    M = []
    for row in rows:
        vector = [1.0]
        for name in sorted(spec):
            info = spec[name]
            value = row["features"][name]
            for category in info["categories"]:
                if category == info["reference"]:
                    continue
                vector.append(1.0 if value == category else 0.0)
        M.append(vector)
    return M, names, targets


def _independent_basis(M, targets, names, tol=1e-11):
    """分两步处理列相关性，避免正交化过程中的系数换算错误。

    1) 经典修正 Gram-Schmidt（未归一化正交向量）只用于稳定识别主元列与
       冗余列；2) 每个冗余列对主元列的表示系数单独解 Gram 系统
       M_P^T M_P a = M_P^T c_j，再用残差范数核对表示是否成立。目标一致性
       即 t_c = Σ a_p t_p；不成立则相关约束矛盾，原系统无可行解。
    """
    n = len(M)
    m = len(names)
    basis = []
    pivots = []
    redundant_indices = []
    for j in range(m):
        column = [M[i][j] for i in range(n)]
        norm_original = math.sqrt(_dot(column, column))
        residual = column[:]
        for _, u in basis:
            projection = _dot(residual, u) / _dot(u, u)
            if projection != 0.0:
                residual = [x - projection * y for x, y in zip(residual, u)]
        norm_residual = math.sqrt(_dot(residual, residual))
        if norm_original == 0.0 or norm_residual <= tol * max(1.0, norm_original):
            redundant_indices.append(j)
        else:
            basis.append((j, residual))
            pivots.append(j)

    pivot_M = [[M[i][p] for p in pivots] for i in range(n)]
    gram = _gram(pivot_M)
    redundant = []
    for j in redundant_indices:
        column = [M[i][j] for i in range(n)]
        rhs = _vec_mat(column, pivot_M)
        try:
            coefficients = _gaussian_solve(gram, rhs)
        except _SingularMatrix:
            # 主元列本应满秩；走到这里说明数值退化，加岭项后求最小范数表示。
            ridge = [row[:] for row in gram]
            scale = max(abs(ridge[i][i]) for i in range(len(ridge))) or 1.0
            for i in range(len(ridge)):
                ridge[i][i] += 1e-12 * scale
            coefficients = _gaussian_solve(ridge, rhs)
        fitted = _mat_vec(pivot_M, coefficients)
        fit_error = math.sqrt(_dot([f - c for f, c in zip(fitted, column)],
                                   [f - c for f, c in zip(fitted, column)]))
        implied = sum(coef * targets[p] for coef, p in zip(coefficients, pivots))
        redundant.append(
            {
                "column": names[j],
                "representation": {names[p]: coef for coef, p in zip(coefficients, pivots)},
                "target": targets[j],
                "implied_target": implied,
                "mismatch": abs(targets[j] - implied),
                "fit_residual": fit_error,
            }
        )
    return pivots, redundant


def _reduce_basis(M, targets, names, tolerance):
    pivots, redundant = _independent_basis(M, targets, names)
    conflicting = []
    for r in redundant:
        scale = max(1.0, abs(r["target"]), abs(r["implied_target"]))
        if r["mismatch"] > tolerance * scale:
            conflicting.append(r)
    if conflicting:
        return None, redundant, conflicting
    reduced_M = [[row[j] for j in pivots] for row in M]
    reduced_t = [targets[j] for j in pivots]
    reduced_names = [names[j] for j in pivots]
    return (reduced_M, reduced_t, reduced_names, pivots), redundant, []


# --------------------------------------------------------------------------
# 对偶熵求解
# --------------------------------------------------------------------------
def _primal_from_lambda(lam, M, d, lower, upper):
    """w = clip(d exp(M lambda), L, U)，同时返回 z 与界内标记。"""
    z = _mat_vec(M, lam)
    w = []
    interior = []
    for zi, di, Li, Ui in zip(z, d, lower, upper):
        log_l = math.log(Li / di)
        log_u = math.log(Ui / di)
        if zi <= log_l:
            w.append(Li)
            interior.append(False)
        elif zi >= log_u:
            w.append(Ui)
            interior.append(False)
        else:
            w.append(di * math.exp(min(zi, _EXP_LIMIT)))
            interior.append(True)
    return z, w, interior


def _dual_objective(lam, targets, z, w, d, lower, upper, interior):
    """对偶目标 g(lambda)（待最大化），分片闭式表达式。"""
    value = _dot(lam, targets)
    for zi, wi, di, Li, Ui, inside in zip(z, w, d, lower, upper, interior):
        if inside:
            # d(1 - exp(z)) = d - w
            value += di - wi
        elif wi == Li:
            l_ratio = Li / di
            value += di * (l_ratio * math.log(l_ratio) - l_ratio + 1.0) - zi * Li
        else:
            u_ratio = Ui / di
            value += di * (u_ratio * math.log(u_ratio) - u_ratio + 1.0) - zi * Ui
    return value


def _scaled_residual(w, M, targets):
    achieved = _vec_mat(w, M)
    differences = [a - t for a, t in zip(achieved, targets)]
    scaled = max(
        (abs(diff) / max(1.0, abs(t)) for diff, t in zip(differences, targets)),
        default=0.0,
    )
    return achieved, differences, scaled


def _newton_step(grad, M, w, interior, step_cap=20.0):
    """解 M^T diag(v) M step = grad；奇异时岭项兜底，再失败退回最速方向。

    触界行使 Hessian 半正定且可能奇异，岭项过大时牛顿步会把 lambda 推入
    极端区域；因此岭项超过迹的 1e-4 仍失败就改用最速上升方向，并对步长
    分量做信赖域截断，方向保持不变。
    """
    v = [wi if inside else 0.0 for wi, inside in zip(w, interior)]
    H = _gram(M, v)
    trace_scale = max(abs(H[i][i]) for i in range(len(H))) or 1.0
    tau = 1e-12 * trace_scale
    direction = None
    for _ in range(9):
        try:
            direction = _gaussian_solve(
                [[H[i][j] + (tau if i == j else 0.0) for j in range(len(H))]
                 for i in range(len(H))], grad)
            if tau > 1e-4 * trace_scale:
                direction = grad[:]
            break
        except _SingularMatrix:
            tau *= 10.0
    if direction is None:
        direction = grad[:]
    largest = max((abs(x) for x in direction), default=0.0)
    if largest > step_cap:
        scale = step_cap / largest
        direction = [x * scale for x in direction]
    return direction


def solve_entropy(M, targets, d, lower, upper, tolerance, max_iterations):
    """对偶牛顿最大化，返回完整求解诊断。失败时不带任何边际放宽。"""
    m = len(M[0])
    lam = [0.0] * m
    z, w, interior = _primal_from_lambda(lam, M, d, lower, upper)
    g_old = _dual_objective(lam, targets, z, w, d, lower, upper, interior)
    best = {"scaled": math.inf, "w": w, "lam": lam, "z": z, "interior": interior}
    stagnant_rounds = 0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        achieved, differences, scaled = _scaled_residual(w, M, targets)
        if scaled < best["scaled"]:
            if best["scaled"] - scaled <= 1e-15 * max(1.0, best["scaled"]):
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            best = {"scaled": scaled, "w": w[:], "lam": lam[:], "z": z[:],
                    "interior": interior[:]}
        else:
            stagnant_rounds += 1
        if scaled <= tolerance:
            return {
                "converged": True,
                "status": OPTIMAL,
                "iterations": iterations,
                "lambda": lam,
                "z": z,
                "w": w,
                "interior": interior,
                "dual_objective": g_old,
                "scaled_residual": scaled,
                "achieved": achieved,
                "differences": differences,
            }
        # 残差长期无法改善：继续迭代只会让无界对偶发散，交给可行性投影判定。
        if stagnant_rounds >= 200:
            break
        grad = [-diff for diff in differences]  # 最大化方向：t - M^T w
        direction = _newton_step(grad, M, w, interior)
        step_size = 1.0
        accepted = False
        armijo = 1e-4 * _dot(grad, direction)
        for _ in range(80):
            candidate_lam = [lam[j] + step_size * direction[j] for j in range(m)]
            cz, cw, cinside = _primal_from_lambda(candidate_lam, M, d, lower, upper)
            g_new = _dual_objective(candidate_lam, targets, cz, cw, d, lower, upper, cinside)
            if g_new >= g_old + step_size * armijo - 1e-14 * abs(g_old):
                lam, z, w, interior, g_old = candidate_lam, cz, cw, cinside, g_new
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            # 牛顿方向失效：最速上升 + 小步长，仍保持原界与原目标。
            direction = grad[:]
            step_size = 1.0 / (1.0 + math.sqrt(_dot(grad, grad)))
            candidate_lam = [lam[j] + step_size * direction[j] for j in range(m)]
            cz, cw, cinside = _primal_from_lambda(candidate_lam, M, d, lower, upper)
            g_new = _dual_objective(candidate_lam, targets, cz, cw, d, lower, upper, cinside)
            if g_new > g_old:
                lam, z, w, interior, g_old = candidate_lam, cz, cw, cinside, g_new
    achieved, differences, scaled = _scaled_residual(best["w"], M, targets)
    return {
        "converged": False,
        "status": NOT_CONVERGED,
        "iterations": iterations,
        "lambda": best["lam"],
        "z": best["z"],
        "w": best["w"],
        "interior": best["interior"],
        "dual_objective": g_old,
        "scaled_residual": best["scaled"],
        "achieved": achieved,
        "differences": differences,
    }


def _project_box_affine(M, targets, lower, upper, tolerance, max_iterations=50000):
    """Dykstra 交替投影：盒 [L,U] 与仿射集 {M^T w = t} 的最近交点。

    标准两集合 Dykstra 迭代（p、q 为两个集合各自的增量变量）：
        y = P_box(x + p);  p = x + p - y
        x = P_affine(y + q); q = y + q - x
    极限点同时落在两个集合中。收敛后若仿射残差仍显著大于容差，则带界
    可行域为空；返回的最佳点带最小残差证据，不改动任何目标或比例界。
    """
    m = len(M[0])
    gram = _gram(M)
    x = [min(max(0.5 * (Li + Ui), Li), Ui) for Li, Ui in zip(lower, upper)]
    p = [0.0] * len(x)
    q = [0.0] * len(x)
    best_scaled = math.inf
    best_point = x[:]
    plateau = 0.0
    stable_rounds = 0
    for _ in range(max_iterations):
        box_input = [x[i] + p[i] for i in range(len(x))]
        y = [min(max(value, Li), Ui) for value, Li, Ui in zip(box_input, lower, upper)]
        p = [box_input[i] - y[i] for i in range(len(x))]
        affine_input = [y[i] + q[i] for i in range(len(x))]
        mismatch = _vec_mat(affine_input, M)
        mismatch = [mismatch[j] - targets[j] for j in range(m)]
        correction = _gaussian_solve(gram, mismatch)
        x = [affine_input[i] - _dot(M[i], correction) for i in range(len(x))]
        q = [affine_input[i] - x[i] for i in range(len(x))]
        # Dykstra 的极限点属于盒；用 y（盒投影结果）评估联合残差。
        achieved, differences, scaled = _scaled_residual(y, M, targets)
        box_gap = max(
            max(Li - yi, 0.0, yi - Ui)
            for yi, Li, Ui in zip(y, lower, upper)
        )
        joint = max(scaled, box_gap / max(1.0, max(upper)))
        if joint < best_scaled:
            best_scaled, best_point = joint, y[:]
        if joint <= tolerance:
            return True, y, achieved, differences, joint
        if abs(plateau - joint) <= 1e-14 * max(1.0, joint):
            stable_rounds += 1
            if stable_rounds >= 1000:
                break
        else:
            stable_rounds = 0
            plateau = joint
    achieved, differences, scaled = _scaled_residual(best_point, M, targets)
    return False, best_point, achieved, differences, best_scaled


def solve_calibration(M, names, targets, d, lower, upper, tolerance, max_iterations):
    """矩阵层面的完整校准流程：支撑检查 -> 去冗余 -> 有界熵求解 -> 可行性诊断。"""
    # 正目标类别必须在样本中有正质量支撑。
    unsupported = []
    for j, name in enumerate(names):
        if targets[j] > 0.0 and all(M[i][j] == 0.0 for i in range(len(M))):
            unsupported.append({"column": name, "target": targets[j]})
    if unsupported:
        return {"status": UNSUPPORTED_TARGET, "unsupported": unsupported}

    reduced, redundant, conflicting = _reduce_basis(M, targets, names, tolerance)
    audit = {
        "columns": names[:],
        "rank": (len(names) - len(redundant)) if reduced is not None else None,
        "redundant_removed": [r["column"] for r in redundant],
        "redundancy_detail": redundant,
    }
    if conflicting:
        audit["conflicting"] = conflicting
        return {"status": INFEASIBLE_DEPENDENT, "audit": audit}

    reduced_M, reduced_t, reduced_names, pivots = reduced
    result = solve_entropy(reduced_M, reduced_t, d, lower, upper, tolerance, max_iterations)
    if result["converged"]:
        result["status"] = OPTIMAL_REDUNDANT if redundant else OPTIMAL
        result["audit"] = audit
        result["column_names"] = reduced_names
        result["pivot_columns"] = pivots
        return result

    # 未收敛：用投影判定是带界不可行还是确实只是未收敛。
    feasible, point, achieved, differences, scaled = _project_box_affine(
        reduced_M, reduced_t, lower, upper, tolerance
    )
    result["audit"] = audit
    result["column_names"] = reduced_names
    result["feasibility_projection"] = {
        "feasible": feasible,
        "scaled_residual": scaled,
        "point": point,
        "achieved": achieved,
        "differences": differences,
        "targets": reduced_t,
        "names": reduced_names,
    }
    result["status"] = NOT_CONVERGED if feasible else BOUNDED_INFEASIBLE
    return result


# --------------------------------------------------------------------------
# 估计量与 KKT 核对
# --------------------------------------------------------------------------
def _primal_objective(w, d):
    return sum(wi * math.log(wi / di) - wi + di for wi, di in zip(w, d))


def _kkt_check(result, d, lower, upper, tolerance):
    """逐项核对对偶/KKT 条件，返回可核验的数值证据。"""
    w = result["w"]
    z = result["z"]
    interior = result["interior"]
    stationarity = 0.0
    clipping_violations = []
    at_lower, at_upper = [], []
    for i, (wi, zi, di, Li, Ui, inside) in enumerate(
        zip(w, z, d, lower, upper, interior)
    ):
        log_l, log_u = math.log(Li / di), math.log(Ui / di)
        at_l = abs(wi - Li) <= _ACTIVE_TOL * max(1.0, Li)
        at_u = abs(wi - Ui) <= _ACTIVE_TOL * max(1.0, Ui)
        if at_l:
            at_lower.append(i)
            if zi > log_l + max(tolerance, 1e-10):
                clipping_violations.append({"row": i, "bound": "lower", "z": zi, "limit": log_l})
        elif at_u:
            at_upper.append(i)
            if zi < log_u - max(tolerance, 1e-10):
                clipping_violations.append({"row": i, "bound": "upper", "z": zi, "limit": log_u})
        else:
            # 严格界内解析解 wi = di*exp(zi)，平稳性即对数残差。
            stationarity = max(stationarity, abs(math.log(wi / di) - zi))
    primal_objective = _primal_objective(w, d)
    gap = primal_objective - result["dual_objective"]
    return {
        "stationarity_max": stationarity,
        "clipping_violations": clipping_violations,
        "primal_objective": primal_objective,
        "dual_objective": result["dual_objective"],
        "duality_gap": gap,
        "at_lower": at_lower,
        "at_upper": at_upper,
    }


def _estimate_section(rows, w, base_ratio):
    total_w = sum(w)
    numerator = sum(wi * r["outcome"] for wi, r in zip(w, rows))
    ratio = numerator / total_w
    ess = total_w * total_w / sum(wi * wi for wi in w)
    max_share = max(w) / total_w
    return {
        "weighted_ratio": ratio,
        "weight_total": total_w,
        "effective_sample_size": ess,
        "max_weight_share": max_share,
        "base_weight_ratio": base_ratio,
        "records": len(rows),
    }


# --------------------------------------------------------------------------
# 分层 delete-one-PSU jackknife
# --------------------------------------------------------------------------
def _jackknife(rows, total, target_tables, spec, settings, point_estimate):
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[row["stratum"]][row["psu"]].append(row)
    strata_table = {h: sorted(psus) for h, psus in sorted(strata.items())}
    m_counts = {h: len(psus) for h, psus in strata_table.items()}
    reasons = []
    singleton = sorted(h for h, m in m_counts.items() if m == 1)
    if singleton:
        reasons.append({"reason": "single_psu_stratum", "strata": singleton})

    replicate_estimates = defaultdict(list)
    replicate_detail = []
    failures = []
    for h, psus in strata_table.items():
        m_h = m_counts[h]
        if m_h == 1:
            continue
        factor = m_h / (m_h - 1)
        for psu in psus:
            sub_rows = [r for r in rows if not (r["stratum"] == h and r["psu"] == psu)]
            d = [r["base_weight"] * (factor if r["stratum"] == h else 1.0) for r in sub_rows]
            lower = [r["lower_ratio"] * di for r, di in zip(sub_rows, d)]
            upper = [r["upper_ratio"] * di for r, di in zip(sub_rows, d)]
            M, names, t = build_matrix(sub_rows, total, target_tables, spec)
            result = solve_calibration(
                M, names, t, d, lower, upper,
                settings["tolerance"], settings["max_iterations"],
            )
            detail = {"stratum": h, "psu": psu, "status": result["status"]}
            if result["status"] in SOLVED_STATUSES:
                w = result["w"]
                theta = sum(wi * r["outcome"] for wi, r in zip(w, sub_rows)) / sum(w)
                replicate_estimates[h].append(theta)
                detail["estimate"] = theta
                detail["iterations"] = result["iterations"]
            else:
                failures.append({"stratum": h, "psu": psu, "status": result["status"]})
                detail["failure"] = result.get("unsupported") or result.get(
                    "feasibility_projection") or result.get("audit")
            replicate_detail.append(detail)
    if failures:
        reasons.append({"reason": "replicate_failure", "failures": failures})

    section = {
        "strata_psu_counts": m_counts,
        "replicate_count": len(replicate_detail),
        "replicates": replicate_detail,
        "suppressed": bool(reasons),
        "reasons": reasons,
    }
    if reasons:
        section.update({"variance": None, "standard_error": None,
                        "confidence_interval": None, "clipped_interval": None})
        return section

    variance = 0.0
    stratum_contributions = {}
    for h, estimates in replicate_estimates.items():
        m_h = m_counts[h]
        mean_h = sum(estimates) / m_h
        contribution = (m_h - 1) / m_h * sum((theta - mean_h) ** 2 for theta in estimates)
        stratum_contributions[h] = contribution
        variance += contribution
    se = math.sqrt(variance)
    point = point_estimate
    z = settings["z"]
    low, high = point - z * se, point + z * se
    section.update({
        "variance": variance,
        "standard_error": se,
        "stratum_variance_contributions": stratum_contributions,
        "confidence_interval": {"low": low, "high": high, "z": z},
        "clipped_interval": {"low": max(0.0, low), "high": min(1.0, high)},
    })
    return section


# --------------------------------------------------------------------------
# 顶层分析
# --------------------------------------------------------------------------
CAVEAT = (
    "校准只强制样本加权边际等于给定总体边际；若参与上传的机器人在未观测维度上"
    "存在自选择，边际校准不能保证消除自选择偏差，区间也只反映抽样方差，不覆盖"
    "目标总体本身的覆盖性偏差。"
)


def analyze(value):
    """对一份符合资料约定的 JSON 对象执行完整校准分析，返回结构化报告。"""
    rows, total, target_tables, settings, margin_issues = _validate(value)
    spec = build_spec(rows, target_tables)
    M, names, targets = build_matrix(rows, total, target_tables, spec)
    d = [r["base_weight"] for r in rows]
    lower = [r["lower_ratio"] * di for r, di in zip(rows, d)]
    upper = [r["upper_ratio"] * di for r, di in zip(rows, d)]
    base_ratio = sum(di * r["outcome"] for di, r in zip(d, rows)) / sum(d)

    report = {
        "settings": settings,
        "caveat": CAVEAT,
        "base_weight_ratio": base_ratio,
    }
    if margin_issues:
        report["status"] = TARGET_INCONSISTENT
        report["margin_sum_issues"] = margin_issues
        report["estimate"] = None
        report["jackknife"] = None
        return report

    result = solve_calibration(
        M, names, targets, d, lower, upper,
        settings["tolerance"], settings["max_iterations"],
    )
    report["status"] = result["status"]

    if result["status"] == UNSUPPORTED_TARGET:
        report["unsupported_targets"] = result["unsupported"]
        report["estimate"] = None
        report["jackknife"] = None
        return report
    if result["status"] == INFEASIBLE_DEPENDENT:
        report["constraint_audit"] = result["audit"]
        report["estimate"] = None
        report["jackknife"] = None
        return report
    if result["status"] == BOUNDED_INFEASIBLE:
        report["constraint_audit"] = result["audit"]
        report["feasibility_projection"] = result["feasibility_projection"]
        report["estimate"] = None
        report["jackknife"] = None
        return report
    if result["status"] == NOT_CONVERGED:
        report["constraint_audit"] = result["audit"]
        report["best_effort"] = {
            "scaled_residual": result["scaled_residual"],
            "feasibility_projection": result["feasibility_projection"],
        }
        report["estimate"] = None
        report["jackknife"] = None
        return report

    # 成功路径：补全残差表、KKT、估计量与 jackknife。
    kkt = _kkt_check(result, d, lower, upper, settings["tolerance"])
    achieved_full = _vec_mat(result["w"], M)
    residual_table = []
    for j, name in enumerate(names):
        residual_table.append({
            "target": name,
            "target_value": targets[j],
            "achieved": achieved_full[j],
            "difference": achieved_full[j] - targets[j],
        })
    multipliers = {name: None for name in names}
    for name, value in zip(result["column_names"], result["lambda"]):
        multipliers[name] = value
    weights = []
    for i, row in enumerate(rows):
        weights.append({
            "id": row["id"],
            "stratum": row["stratum"],
            "psu": row["psu"],
            "base_weight": d[i],
            "weight": result["w"][i],
            "ratio": result["w"][i] / d[i],
            "at_lower": i in kkt["at_lower"],
            "at_upper": i in kkt["at_upper"],
        })
    report["constraint_audit"] = result["audit"]
    report["weights"] = weights
    report["calibration"] = {
        "iterations": result["iterations"],
        "scaled_residual": result["scaled_residual"],
        "residuals": residual_table,
        "objective": kkt["primal_objective"],
        "dual_objective": kkt["dual_objective"],
        "duality_gap": kkt["duality_gap"],
        "stationarity_max": kkt["stationarity_max"],
        "clipping_violations": kkt["clipping_violations"],
        "dual_multipliers": multipliers,
        "bound_active": {
            "lower": [rows[i]["id"] for i in kkt["at_lower"]],
            "upper": [rows[i]["id"] for i in kkt["at_upper"]],
        },
    }
    report["estimate"] = _estimate_section(rows, result["w"], base_ratio)
    report["jackknife"] = _jackknife(
        rows, total, target_tables, spec, settings,
        report["estimate"]["weighted_ratio"],
    )
    return report
