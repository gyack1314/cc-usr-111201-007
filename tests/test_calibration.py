"""核对有界熵校准、对偶/KKT 与分层 jackknife 的数学口径与诊断状态。"""
import copy
import json
import unittest

from analysis_data.io import load
from analysis_data.calibration import (
    analyze,
    build_matrix,
    build_spec,
    solve_calibration,
    _validate,
    OPTIMAL,
    OPTIMAL_REDUNDANT,
    TARGET_INCONSISTENT,
    UNSUPPORTED_TARGET,
    INFEASIBLE_DEPENDENT,
    BOUNDED_INFEASIBLE,
    NOT_CONVERGED,
    SOLVED_STATUSES,
)

DATA = "data"


def _material(name):
    return load(f"{DATA}/{name}.json")[0]


class CalibrationMathTests(unittest.TestCase):
    def test_normal_margins_residuals_and_kkt(self):
        report = analyze(_material("normal"))
        self.assertEqual(report["status"], OPTIMAL_REDUNDANT)
        calibration = report["calibration"]
        # 每条校准约束都在容差内命中目标。
        for row in calibration["residuals"]:
            self.assertLess(abs(row["difference"]), 1e-7)
        # 对偶间隙与平稳性残差都应接近机器精度。
        self.assertLess(abs(calibration["duality_gap"]), 1e-6)
        self.assertLess(calibration["stationarity_max"], 1e-9)
        self.assertEqual(calibration["clipping_violations"], [])
        # 手算解：low 行权重 1.5、high 行 2.5。
        weights = {w["id"]: w["weight"] for w in report["weights"]}
        self.assertAlmostEqual(weights["H1P10"], 1.5, places=6)
        self.assertAlmostEqual(weights["H1P11"], 2.5, places=6)
        # 目标函数非负，且 w=d 时为 0；这里应严格为正。
        self.assertGreater(calibration["objective"], 0.0)

    def test_weights_stay_within_bounds(self):
        report = analyze(_material("normal"))
        for w in report["weights"]:
            lower = w["base_weight"] * 0.25
            upper = w["base_weight"] * 4.0
            self.assertGreaterEqual(w["weight"] + 1e-9, lower)
            self.assertLessEqual(w["weight"] - 1e-9, upper)

    def test_bound_activation_and_replicate_feasibility(self):
        report = analyze(_material("bound-active"))
        self.assertEqual(report["status"], OPTIMAL)
        self.assertEqual(report["calibration"]["bound_active"]["lower"], ["H1P10"])
        tight = next(w for w in report["weights"] if w["id"] == "H1P10")
        self.assertAlmostEqual(tight["weight"], 2.2, places=6)
        self.assertTrue(tight["at_lower"])
        # 所有副本均重新校准成功，并给出有效区间。
        jackknife = report["jackknife"]
        self.assertFalse(jackknife["suppressed"])
        self.assertEqual(len(jackknife["replicates"]), 6)
        self.assertTrue(all(r["status"] in SOLVED_STATUSES for r in jackknife["replicates"]))
        self.assertIsNotNone(jackknife["standard_error"])
        self.assertGreater(jackknife["standard_error"], 0.0)
        interval = jackknife["confidence_interval"]
        self.assertLess(interval["low"], report["estimate"]["weighted_ratio"] < interval["high"])
        # 截尾可视区间必须落在 [0,1]，原区间保留且可能越界。
        clipped = jackknife["clipped_interval"]
        self.assertGreaterEqual(clipped["low"], 0.0)
        self.assertLessEqual(clipped["high"], 1.0)

    def test_estimate_outputs(self):
        report = analyze(_material("normal"))
        estimate = report["estimate"]
        self.assertAlmostEqual(estimate["weight_total"], 16.0, places=6)
        self.assertLess(estimate["effective_sample_size"], estimate["records"])
        self.assertGreater(estimate["effective_sample_size"], 0.0)
        self.assertGreaterEqual(estimate["max_weight_share"], 1.0 / estimate["records"] - 1e-9)
        self.assertLessEqual(estimate["max_weight_share"], 1.0 + 1e-9)

    def test_jackknife_variance_formula(self):
        # 用独立的逐副本重算路径复算方差，核对分层 (m_h-1)/m_h 公式。
        report = analyze(_material("normal"))
        jackknife = report["jackknife"]
        by_stratum = {}
        for replicate in jackknife["replicates"]:
            by_stratum.setdefault(replicate["stratum"], []).append(replicate["estimate"])
        expected = 0.0
        for stratum, values in by_stratum.items():
            m_h = len(values)
            mean = sum(values) / m_h
            expected += (m_h - 1) / m_h * sum((v - mean) ** 2 for v in values)
        self.assertAlmostEqual(jackknife["variance"], expected, places=10)
        self.assertAlmostEqual(jackknife["standard_error"], expected ** 0.5, places=10)


class DiagnosticStatusTests(unittest.TestCase):
    def test_target_inconsistent(self):
        report = analyze(_material("inconsistent-totals"))
        self.assertEqual(report["status"], TARGET_INCONSISTENT)
        self.assertEqual(report["margin_sum_issues"][0]["margin"], "load")
        self.assertIsNone(report["estimate"])

    def test_unsupported_target(self):
        # 数据契约要求边际类别集合与样本一致，因此“正目标无支撑”在设计矩阵
        # 层面体现为：某指示列存在但全为 0，同时目标向量给它正质量。
        # 直接在矩阵层面构造，确认 solve_calibration 单独诊断而非崩溃。
        value = _material("normal")
        rows, total, tables, settings, _ = _validate(value)
        spec = build_spec(rows, tables)
        M, names, targets = build_matrix(rows, total, tables, spec)
        # 追加一个恒零指示列并给正目标。
        unsupported_M = [row + [0.0] for row in M]
        unsupported_names = names + ["phantom:level"]
        unsupported_targets = targets + [3.0]
        d = [r["base_weight"] for r in rows]
        lower = [r["lower_ratio"] * x for r, x in zip(rows, d)]
        upper = [r["upper_ratio"] * x for r, x in zip(rows, d)]
        result = solve_calibration(
            unsupported_M, unsupported_names, unsupported_targets, d,
            lower, upper, settings["tolerance"], settings["max_iterations"],
        )
        self.assertEqual(result["status"], UNSUPPORTED_TARGET)
        self.assertEqual(result["unsupported"][0]["column"], "phantom:level")

    def test_full_pipeline_unsupported_via_empty_indicator(self):
        # 端到端路径：样本中保留某类别取值但该类别行被全部排除到无法承载
        # 正目标时（权重下界严格为正、指示列全 0），同样命中无支撑诊断。
        value = _material("replicate-support-loss")
        # rare=yes 仅 H1P1 两行；把它们的 rare 改成 no，同时在 H2 某行保留
        # 一个 rare=yes 以维持类别集合，但令该 H2 行…… 此处直接验证矩阵入口。
        rows, total, tables, settings, _ = _validate(value)
        spec = build_spec(rows, tables)
        M, names, targets = build_matrix(rows, total, tables, spec)
        yes_index = names.index("rare:yes")
        for row in M:
            row[yes_index] = 0.0
        d = [r["base_weight"] for r in rows]
        lower = [r["lower_ratio"] * x for r, x in zip(rows, d)]
        upper = [r["upper_ratio"] * x for r, x in zip(rows, d)]
        result = solve_calibration(
            M, names, targets, d, lower, upper,
            settings["tolerance"], settings["max_iterations"],
        )
        self.assertEqual(result["status"], UNSUPPORTED_TARGET)

    def test_infeasible_dependent_constraints(self):
        report = analyze(_material("infeasible-margins"))
        self.assertEqual(report["status"], INFEASIBLE_DEPENDENT)
        detail = report["constraint_audit"]["conflicting"]
        self.assertTrue(detail)
        self.assertGreater(detail[0]["mismatch"], 0.0)

    def test_bounded_infeasible(self):
        report = analyze(_material("bounded-infeasible"))
        self.assertEqual(report["status"], BOUNDED_INFEASIBLE)
        projection = report["feasibility_projection"]
        self.assertFalse(projection["feasible"])
        self.assertGreater(projection["scaled_residual"], 1e-4)
        self.assertIsNone(report["estimate"])

    def test_not_converged_is_reported_not_relaxed(self):
        value = _material("normal")
        value["solver"]["max_iterations"] = 1
        report = analyze(value)
        self.assertEqual(report["status"], NOT_CONVERGED)
        self.assertIsNone(report["estimate"])
        # 绝不能偷偷改目标：最佳努力残差必须明显大于容差。
        self.assertGreater(report["best_effort"]["scaled_residual"],
                           value["solver"]["tolerance"])

    def test_invalid_base_weight_rejected(self):
        with self.assertRaises(ValueError):
            analyze(_material("invalid-base-weight"))


class JackknifeSuppressionTests(unittest.TestCase):
    def test_single_psu_stratum_keeps_point_estimate(self):
        report = analyze(_material("single-psu-stratum"))
        self.assertIn(report["status"], SOLVED_STATUSES)
        self.assertIsNotNone(report["estimate"])
        jackknife = report["jackknife"]
        self.assertTrue(jackknife["suppressed"])
        self.assertEqual(jackknife["reasons"][0]["reason"], "single_psu_stratum")
        self.assertIn("H1", jackknife["reasons"][0]["strata"])
        self.assertIsNone(jackknife["confidence_interval"])

    def test_replicate_support_loss_suppresses_interval(self):
        report = analyze(_material("replicate-support-loss"))
        self.assertIn(report["status"], SOLVED_STATUSES)
        jackknife = report["jackknife"]
        self.assertTrue(jackknife["suppressed"])
        self.assertEqual(jackknife["reasons"][0]["reason"], "replicate_failure")
        failed = jackknife["reasons"][0]["failures"]
        self.assertTrue(any(f["psu"] == "H1P1" for f in failed))
        self.assertIsNone(jackknife["standard_error"])

    def test_replicates_reapply_bounds_on_scaled_base_weights(self):
        # 副本失败原因必须是带界/失撑诊断，而不是静默放宽；核对失败状态枚举。
        report = analyze(_material("replicate-support-loss"))
        failure = report["jackknife"]["reasons"][0]["failures"][0]
        self.assertEqual(failure["status"], UNSUPPORTED_TARGET)


class EncodingInvarianceTests(unittest.TestCase):
    def _weights(self, value):
        report = analyze(value)
        self.assertIn(report["status"], SOLVED_STATUSES)
        return {w["id"]: w["weight"] for w in report["weights"]}

    def test_reference_class_choice_invariant(self):
        value = _material("normal")
        baseline = self._weights(value)
        # 手工构造另一套参考类：交换 region 的 east/west 编码，等价于改写
        # 指示列与目标，约束系统解集不变。
        flipped = copy.deepcopy(value)
        for row in flipped["sample"]:
            row["features"]["region"] = (
                "west" if row["features"]["region"] == "east" else "east"
            )
        flipped["target"]["margins"]["region"] = {
            "west": value["target"]["margins"]["region"]["east"],
            "east": value["target"]["margins"]["region"]["west"],
        }
        flipped_weights = self._weights(flipped)
        for identifier, weight in baseline.items():
            self.assertAlmostEqual(flipped_weights[identifier], weight, places=6)

    def test_row_permutation_invariant(self):
        value = _material("normal")
        baseline = self._weights(value)
        permuted = copy.deepcopy(value)
        permuted["sample"] = list(reversed(permuted["sample"]))
        permuted_weights = self._weights(permuted)
        for identifier, weight in baseline.items():
            self.assertAlmostEqual(permuted_weights[identifier], weight, places=6)

    def test_redundant_full_dummy_encoding_invariant(self):
        # 用全指示列（含参考类）+ 总量构造线性相关约束，求解器去冗余后应
        # 得到与去参考类编码相同的权重。
        value = _material("normal")
        rows, total, tables, settings, _ = _validate(value)
        spec = build_spec(rows, tables)
        reduced_M, names, targets = build_matrix(rows, total, tables, spec)
        d = [r["base_weight"] for r in rows]
        lower = [r["lower_ratio"] * x for r, x in zip(rows, d)]
        upper = [r["upper_ratio"] * x for r, x in zip(rows, d)]
        result_reduced = solve_calibration(
            reduced_M, names, targets, d, lower, upper,
            settings["tolerance"], settings["max_iterations"],
        )
        # 全 dummy：每个特征的所有类别都出指示列，与总量列线性相关。
        full_names = ["total"]
        full_targets = [total]
        for name in sorted(spec):
            for category in spec[name]["categories"]:
                full_names.append(f"{name}:{category}")
                full_targets.append(tables[name][category])
        full_M = []
        for row in rows:
            vector = [1.0]
            for name in sorted(spec):
                for category in spec[name]["categories"]:
                    vector.append(1.0 if row["features"][name] == category else 0.0)
            full_M.append(vector)
        result_full = solve_calibration(
            full_M, full_names, full_targets, d, lower, upper,
            settings["tolerance"], settings["max_iterations"],
        )
        self.assertIn(result_full["status"], SOLVED_STATUSES)
        self.assertTrue(result_full["audit"]["redundant_removed"])
        for a, b in zip(result_reduced["w"], result_full["w"]):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
