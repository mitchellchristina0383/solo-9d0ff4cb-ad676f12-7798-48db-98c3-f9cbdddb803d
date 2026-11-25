"""根证书更换顺序求解器。

状态建模
========
* 每个服务副本有两种链状态: OLD(旧链) / NEW(新链)，一次只更换一个副本；
* 每批终端有三种根包状态: OLD(旧包) / TRANSITION(新旧并集) / NEW(新包)。
  OLD→NEW 可直跳，也可经 TRANSITION 过渡；处于 TRANSITION 的批次数不得超过
  页面设定的“同时过渡批次上限”。

核验约束
========
任何核验时刻（初始态、每一步动作之后），每批终端都必须能用“当前持有的根
证书包”按域名验证每个副本“当前出示的证书链”。核验委托给
:mod:`app.certutil` 中的 openssl verify。

搜索
====
状态空间至多 2^4 × 3^4 = 1296，直接 BFS 求步数最少的可行顺序；无解时区分
“材料本身无法通过验证（硬无解）”与“过渡名额不足（调大上限即可）”。

核验调用上限（预算模式）
========================
页面可设置可选的 openssl 调用上限 ``verify_budget``：

* 每个“副本 × 链状态 × 批次 × 根包状态”核验格在整次请求内最多调用一次
  openssl，结果写入缓存后全程复用——即所有判定都按请求开始时刻的材料
  快照作出，同一格不会出现两种结论；
* 预算用尽后，未核验格返回 :data:`PENDING`（不入缓存），依赖这些格子的
  状态判为“未知”，BFS 不把未知状态入队；
* 能确定当前名额下的最短方案、名额不足时的最小同时过渡批次数、或放开名额
  仍然无解时即停止核验；预算用尽且未验格仍可能改判时，结论标记为待定，
  页面展示“未验”。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .certutil import PemError, merge_pem_bundles, verify_chain

OLD = 0
NEW = 1
TRANSITION = 2

# 核验格三态：通过 / 失败 / 未验（预算用尽，尚未调用 openssl）
PASS = "pass"
FAIL = "fail"
PENDING = "pending"

STATE_LABEL = {OLD: "旧", TRANSITION: "过渡(并集)", NEW: "新"}
BUNDLE_MODE_LABEL = {OLD: "旧根包", NEW: "新根包", TRANSITION: "并集根包"}
CHAIN_LABEL = {OLD: "旧证书链", NEW: "新证书链"}


@dataclass
class Replica:
    name: str
    domain: str
    chains: dict[int, str]  # OLD / NEW 两份 PEM 链


@dataclass
class Batch:
    name: str
    bundles: dict[int, str]  # OLD / NEW / TRANSITION(并集) 三份 PEM 包


@dataclass
class CellResult:
    ok: bool
    reason: str = ""
    state: str = PASS  # PASS | FAIL | PENDING


@dataclass
class SolveResult:
    # "plan" | "plan_unconfirmed" | "hard_infeasible"
    # | "insufficient_quota" | "pending"
    status: str
    plan: list[dict] = field(default_factory=list)
    steps_count: int = 0
    matrix: list[list[dict]] = field(default_factory=list)
    initial_violations: list[dict] = field(default_factory=list)
    target_violations: list[dict] = field(default_factory=list)
    blocker_pairs: list[dict] = field(default_factory=list)
    max_transition: int = 0
    min_required: int | None = None
    message: str = ""
    verify_budget: int | None = None
    verify_calls: int = 0
    pending_cells: int = 0
    shortest_confirmed: bool = True


class Planner:
    def __init__(
        self,
        replicas: list[Replica],
        batches: list[Batch],
        max_transition: int,
        verify_budget: int | None = None,
    ):
        self.replicas = replicas
        self.batches = batches
        self.n = len(replicas)
        self.m = len(batches)
        self.max_transition = max_transition
        # None 表示不启用调用上限（原行为：验完全部核验格）
        self.verify_budget = verify_budget
        # 实际发起的 openssl 子进程调用次数（缓存命中不计数）
        self.verify_calls = 0
        # verify_cache[(replica_idx, chain_state, bundle_mode, batch_idx)]
        self._cache: dict[tuple[int, int, int, int], CellResult] = {}

    # ------------------------------------------------------------------ #
    # 核验矩阵
    # ------------------------------------------------------------------ #
    def _cell(self, i: int, chain_state: int, j: int, bundle_mode: int) -> CellResult:
        """核验单个“批次 × 副本”格。

        同一格在整次请求内最多调用一次 openssl（结果缓存复用，等价于按
        请求开始时刻核验）；启用预算且调用次数已用尽时返回未验格
        :data:`PENDING`，未验结果不入缓存，后续请求仍可补验。
        """
        key = (i, chain_state, bundle_mode, j)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.verify_budget is not None and self.verify_calls >= self.verify_budget:
            return CellResult(False, "未验：openssl 调用次数已达上限", PENDING)
        replica = self.replicas[i]
        bundle = self.batches[j].bundles[bundle_mode]
        result = verify_chain(replica.chains[chain_state], bundle, replica.domain)
        self.verify_calls += 1
        cell = CellResult(
            result.ok,
            result.reason,
            PASS if result.ok else FAIL,
        )
        self._cache[key] = cell
        return cell

    def _state_check(
        self, chains: tuple[int, ...], bundles: tuple[int, ...]
    ) -> str:
        """整体状态核验，三态返回 PASS / FAIL / PENDING。

        任一已验格失败即 FAIL（不继续消耗预算）；存在未验格且无失败格时
        返回 PENDING；全部已验且通过才是 PASS。
        """
        saw_pending = False
        for i, chain_state in enumerate(chains):
            for j, bundle_mode in enumerate(bundles):
                cell = self._cell(i, chain_state, j, bundle_mode)
                if cell.state == PENDING:
                    saw_pending = True
                elif not cell.ok:
                    return FAIL
        return PENDING if saw_pending else PASS

    # ------------------------------------------------------------------ #
    # BFS
    # ------------------------------------------------------------------ #
    def _neighbors(
        self,
        chains: tuple[int, ...],
        bundles: tuple[int, ...],
        limit: int,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...], dict]]:
        neighbors: list[tuple[tuple[int, ...], tuple[int, ...], dict]] = []
        transitioning = sum(1 for b in bundles if b == TRANSITION)

        # 顺序固定，保证方案可复现：先开过渡 → 再换副本链 → 最后收过渡/直跳
        for j in range(self.m):
            if bundles[j] == OLD and transitioning < limit:
                new_bundles = list(bundles)
                new_bundles[j] = TRANSITION
                move = {
                    "kind": "batch_transition",
                    "batch_index": j,
                    "text": (
                        f"终端批次「{self.batches[j].name}」: 旧根包 → 过渡根包"
                        f"（新旧并集）"
                    ),
                }
                neighbors.append((chains, tuple(new_bundles), move))

        for i in range(self.n):
            if chains[i] == OLD:
                new_chains = list(chains)
                new_chains[i] = NEW
                move = {
                    "kind": "replica",
                    "replica_index": i,
                    "text": (
                        f"服务副本「{self.replicas[i].name}」"
                        f"({self.replicas[i].domain}): 旧证书链 → 新证书链"
                    ),
                }
                neighbors.append((tuple(new_chains), bundles, move))

        for j in range(self.m):
            if bundles[j] == TRANSITION:
                new_bundles = list(bundles)
                new_bundles[j] = NEW
                move = {
                    "kind": "batch_finish",
                    "batch_index": j,
                    "text": (
                        f"终端批次「{self.batches[j].name}」: 过渡根包 → 新根包"
                    ),
                }
                neighbors.append((chains, tuple(new_bundles), move))
            elif bundles[j] == OLD:
                # 直跳: OLD → NEW（仅当目标态核验通过才可能被走到）
                new_bundles = list(bundles)
                new_bundles[j] = NEW
                move = {
                    "kind": "batch_direct",
                    "batch_index": j,
                    "text": (
                        f"终端批次「{self.batches[j].name}」: 旧根包 → 新根包"
                        f"（直接更换）"
                    ),
                }
                neighbors.append((chains, tuple(new_bundles), move))
        return neighbors

    def _bfs(
        self, limit: int
    ) -> tuple[list[dict] | None, bool]:
        """按过渡名额 ``limit`` 做 BFS。

        返回 ``(path, hit_unknown)``：

        * 找到目标态时 ``path`` 为动作序列（BFS 保证是已核验状态空间内的
          最短路径；未核验状态从未入队）；走不通时 ``path`` 为 ``None``；
        * ``hit_unknown`` 表示搜索过程中是否遇到过未验格（预算用尽）。
          为 ``False`` 且 ``path is None`` 时，才能断言该名额下确实无解；
          为 ``True`` 时未验格可能改判，结论只能是待定。
        """
        start_chains = tuple([OLD] * self.n)
        start_bundles = tuple([OLD] * self.m)
        target_chains = tuple([NEW] * self.n)
        target_bundles = tuple([NEW] * self.m)

        start_check = self._state_check(start_chains, start_bundles)
        if start_check == FAIL:
            return [], False  # 空列表表示初始态即不合法，由上层诊断
        hit_unknown = start_check == PENDING
        goal = (target_chains, target_bundles)
        queue: deque[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = deque([(start_chains, start_bundles)])
        prev: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[int, ...], tuple[int, ...], dict] | None,
        ] = {(start_chains, start_bundles): None}

        while queue:
            chains, bundles = queue.popleft()
            if (chains, bundles) == goal:
                # 回溯动作序列
                path: list[dict] = []
                node = goal
                while prev[node] is not None:
                    prev_chains, prev_bundles, move = prev[node]
                    move = dict(move)
                    move["chains_after"] = list(node[0])
                    move["bundles_after"] = list(node[1])
                    path.append(move)
                    node = (prev_chains, prev_bundles)
                path.reverse()
                return path, hit_unknown
            for next_chains, next_bundles, move in self._neighbors(
                chains, bundles, limit
            ):
                if (next_chains, next_bundles) in prev:
                    continue
                check = self._state_check(next_chains, next_bundles)
                if check == FAIL:
                    continue
                if check == PENDING:
                    # 预算用尽：未验格可能通过也可能失败，不入队也不下结论
                    hit_unknown = True
                    continue
                prev[(next_chains, next_bundles)] = (chains, bundles, move)
                queue.append((next_chains, next_bundles))
        return None, hit_unknown

    # ------------------------------------------------------------------ #
    # 诊断
    # ------------------------------------------------------------------ #
    def _build_matrix(self) -> list[list[dict]]:
        """每个 (副本, 批次) 在三种根包 × 两种链下的核验结果。

        只读取已核验缓存，未验格标记为 :data:`PENDING`（不再额外调用
        openssl），保证矩阵调用次数与求解过程合计不超过预算。
        """
        matrix: list[list[dict]] = []
        for i in range(self.n):
            row: list[dict] = []
            for j in range(self.m):
                cells = {}
                for chain_state in (OLD, NEW):
                    for bundle_mode in (OLD, TRANSITION, NEW):
                        cell = self._cache.get(
                            (i, chain_state, bundle_mode, j)
                        )
                        if cell is None:
                            cells[f"{chain_state}-{bundle_mode}"] = {
                                "ok": False,
                                "state": PENDING,
                                "reason": "未验：openssl 调用次数已达上限",
                            }
                        else:
                            cells[f"{chain_state}-{bundle_mode}"] = {
                                "ok": cell.ok,
                                "state": cell.state,
                                "reason": cell.reason,
                            }
                row.append(cells)
            matrix.append(row)
        return matrix

    def _violations(self, chain_state: int, bundle_mode: int) -> tuple[list[dict], int]:
        """返回 (已验失败项, 未验格数量)。未验格不计为失败。"""
        out: list[dict] = []
        pending = 0
        for i in range(self.n):
            for j in range(self.m):
                cell = self._cell(i, chain_state, j, bundle_mode)
                if cell.state == PENDING:
                    pending += 1
                elif not cell.ok:
                    out.append(
                        {
                            "replica_index": i,
                            "replica_name": self.replicas[i].name,
                            "domain": self.replicas[i].domain,
                            "batch_index": j,
                            "batch_name": self.batches[j].name,
                            "reason": cell.reason,
                            "chain_state": chain_state,
                            "bundle_mode": bundle_mode,
                        }
                    )
        return out, pending

    def solve(self) -> SolveResult:
        result = SolveResult(
            status="plan",
            max_transition=self.max_transition,
            verify_budget=self.verify_budget,
        )

        # 1) 现状与目标态核验：已验失败即硬无解；含未验格则结论待定
        initial_violations, initial_pending = self._violations(OLD, OLD)
        target_violations, target_pending = self._violations(NEW, NEW)
        result.initial_violations = initial_violations
        result.target_violations = target_violations

        if initial_violations or target_violations:
            result.status = "hard_infeasible"
            result.blocker_pairs = self._blocker_pairs()
            result.matrix = self._build_matrix()
            self._fill_counters(result)
            if initial_violations and target_violations:
                result.message = (
                    "现状与目标态均存在无法通过的核验，证书材料本身不成立。"
                )
            elif initial_violations:
                result.message = "现状（旧链 + 旧根包）已无法通过核验，无法开始变更。"
            else:
                result.message = (
                    "目标态（新链 + 新根包）无法通过核验，再怎么安排顺序也无解。"
                )
            return result

        if initial_pending or target_pending:
            # 预算在现状/目标态阶段就已用尽，无法给出任何确定结论
            return self._finish_pending(
                result,
                "openssl 调用次数已用尽：现状或目标态仍有未验格，"
                "能否开始变更尚无法判断，请调大调用上限后重新求解。",
            )

        # 2) 按页面上限搜索（BFS 找到目标即提前停止，不再多验）
        path, hit_unknown = self._bfs(self.max_transition)
        if path is not None:
            result.plan = self._decorate(path)
            result.steps_count = len(path)
            if hit_unknown:
                result.status = "plan_unconfirmed"
                result.shortest_confirmed = False
                result.message = (
                    f"已找到一条每一步均通过核验的变更顺序，共 {len(path)} 步；"
                    "但 openssl 调用次数已用尽、仍有未验格，是否存在更短方案"
                    "待定，请调大调用上限后重新求解确认最短步数。"
                )
            else:
                result.status = "plan"
                result.shortest_confirmed = True
                result.message = f"找到可行变更顺序，共 {len(path)} 步（已确认为最短步数）。"
            result.matrix = self._build_matrix()
            self._fill_counters(result)
            return result

        # 3) 当前名额走不通：放开上限到全部批次，区分硬无解 / 名额不足 / 待定
        relaxed, relaxed_unknown = self._bfs(self.m)
        if relaxed == []:
            # 放开后初始态反而不合法（理论上与第 1 步重复，防御性处理）
            result.status = "hard_infeasible"
            result.blocker_pairs = self._blocker_pairs()
            result.message = "现状核验无法通过，无法开始变更。"
            result.matrix = self._build_matrix()
            self._fill_counters(result)
            return result
        if relaxed is None and not relaxed_unknown:
            result.status = "hard_infeasible"
            result.blocker_pairs = self._blocker_pairs()
            result.message = (
                "即使所有批次同时过渡也无解：存在副本链与任何可用根包都无法"
                "完成的核验组合，见下方阻塞明细。"
            )
            result.matrix = self._build_matrix()
            self._fill_counters(result)
            return result
        if relaxed is None:
            return self._finish_pending(
                result,
                "openssl 调用次数已用尽：放开过渡名额后仍有未验状态，"
                "究竟是硬无解还是名额不足尚无法判断，请调大调用上限后重新求解。",
            )

        # 放开上限可解：当前名额不足。二分最小名额；若二分过程中预算用尽，
        # 已核验结果仍能给出“至少需要”的下界，但精确最小值待定。
        min_required, min_confirmed = self._min_limit()
        if min_confirmed:
            result.status = "insufficient_quota"
            result.min_required = min_required
            result.shortest_confirmed = True
            result.message = (
                f"过渡名额不足：当前上限为 {self.max_transition}，"
                f"至少需要 {min_required} 个批次同时处于过渡态才能排通。"
            )
        else:
            result.status = "pending"
            result.min_required = min_required  # 已核验出的下界
            result.shortest_confirmed = False
            result.message = (
                "openssl 调用次数已用尽：已确认当前过渡名额不足，且至少需要 "
                f"{min_required} 个批次同时过渡；精确的最小名额仍可能更大，"
                "请调大调用上限后重新求解。"
            )
        result.matrix = self._build_matrix()
        self._fill_counters(result)
        return result

    def _finish_pending(self, result: SolveResult, message: str) -> SolveResult:
        result.status = "pending"
        result.shortest_confirmed = False
        result.message = message
        result.matrix = self._build_matrix()
        self._fill_counters(result)
        return result

    def _fill_counters(self, result: SolveResult) -> None:
        result.verify_calls = self.verify_calls
        pending_cells = 0
        for row in result.matrix:
            for cells in row:
                pending_cells += sum(
                    1 for cell in cells.values() if cell["state"] == PENDING
                )
        result.pending_cells = pending_cells

    def _min_limit(self) -> tuple[int, int]:
        """求最小同时过渡名额，返回 (名额下界, 是否确认精确最小值)。

        前置保证：调用前已确认名额 ``m``（全部批次同时过渡）可解，因此真实
        最小值一定落在 ``[1, m]``。二分中：

        * 探测名额确认可解 → 真实最小值 ≤ mid，收紧上界；
        * 确认不可解（无未知格）→ 真实最小值 ≥ mid+1，收紧下界；
        * 探测遇到未验格 → mid 是否可解未知；此时预算已用尽，后续探测不会
          再获得新信息，立即返回当前已证实的下界与 ``confirmed=False``。
        """
        lo, hi = 1, self.m
        while lo < hi:
            mid = (lo + hi) // 2
            path, hit_unknown = self._bfs(mid)
            if path is not None:
                hi = mid
            elif not hit_unknown:
                lo = mid + 1
            else:
                # path 为 None（搜索途中遇未验格）或 []（起始态即含未验格）：
                # 该名额是否可解未知，立即返回已证实的稳妥下界
                return lo, False
        return lo, True

    def _blocker_pairs(self) -> list[dict]:
        """列出存在失败核验组合的 (副本, 批次) 对及全部失败原因。

        仅依据已核验缓存构造；未验格不参与失败明细（矩阵中单独显示未验）。
        """
        blockers: list[dict] = []
        for i in range(self.n):
            for j in range(self.m):
                failed_cells: list[dict] = []
                any_ok = False
                for chain_state in (OLD, NEW):
                    for bundle_mode in (OLD, TRANSITION, NEW):
                        cell = self._cache.get(
                            (i, chain_state, bundle_mode, j)
                        )
                        if cell is None or cell.state == PENDING:
                            continue
                        if cell.ok:
                            any_ok = True
                        else:
                            failed_cells.append(
                                {
                                    "chain_state": chain_state,
                                    "bundle_mode": bundle_mode,
                                    "chain_label": CHAIN_LABEL[chain_state],
                                    "bundle_label": BUNDLE_MODE_LABEL[bundle_mode],
                                    "reason": cell.reason,
                                }
                            )
                if failed_cells:
                    blockers.append(
                        {
                            "replica_index": i,
                            "replica_name": self.replicas[i].name,
                            "domain": self.replicas[i].domain,
                            "batch_index": j,
                            "batch_name": self.batches[j].name,
                            "any_ok": any_ok,
                            "failed_cells": failed_cells,
                        }
                    )
        return blockers

    def _decorate(self, path: list[dict]) -> list[dict]:
        """补充分步的状态快照与本步核验摘要，便于页面展示。"""
        decorated: list[dict] = []
        for step_no, move in enumerate(path, start=1):
            info = dict(move)
            info["step_no"] = step_no
            info["chains_label"] = [STATE_LABEL[s] for s in move["chains_after"]]
            info["bundles_label"] = [STATE_LABEL[s] for s in move["bundles_after"]]
            # 本步动作后仍逐对核验通过，附上核验计数作为证据
            checks: list[dict] = []
            for i, chain_state in enumerate(move["chains_after"]):
                for j, bundle_mode in enumerate(move["bundles_after"]):
                    cell = self._cell(i, chain_state, j, bundle_mode)
                    checks.append(
                        {
                            "ok": cell.ok,
                            "state": cell.state,
                            "reason": cell.reason,
                        }
                    )
            info["checks_total"] = len(checks)
            info["checks_passed"] = sum(1 for c in checks if c["ok"])
            decorated.append(info)
        return decorated


def build_planner(
    replica_rows: list[dict],
    batch_rows: list[dict],
    max_transition: int,
    verify_budget: int | None = None,
) -> Planner:
    """从页面表单数据构造 :class:`Planner`，并计算并集根包。

    每个 replica_rows 元素: {name, domain, old_chain, new_chain}
    每个 batch_rows 元素:   {name, old_bundle, new_bundle}

    ``verify_budget`` 为 ``None`` 时不限制 openssl 调用次数；否则须为正整数，
    核验格按请求开始时刻的材料快照逐格最多调用一次 openssl。
    """
    if not (1 <= len(replica_rows) <= 4):
        raise ValueError("服务副本数量须在 1~4 之间")
    if not (1 <= len(batch_rows) <= 4):
        raise ValueError("终端批次数量须在 1~4 之间")
    if not (0 <= max_transition <= len(batch_rows)):
        raise ValueError("同时过渡批次上限须在 0 与批次数之间")
    if verify_budget is not None and (
        isinstance(verify_budget, bool)
        or not isinstance(verify_budget, int)
        or verify_budget < 1
    ):
        raise ValueError("openssl 调用上限须为不小于 1 的整数，或留空表示不限")

    replicas: list[Replica] = []
    for idx, row in enumerate(replica_rows, start=1):
        domain = (row.get("domain") or "").strip()
        if not domain:
            raise ValueError(f"第 {idx} 个副本缺少域名")
        try:
            old_chain = row["old_chain"]
            new_chain = row["new_chain"]
            merge_pem_bundles(old_chain, new_chain)  # 仅用于校验两份 PEM 均可解析
        except PemError as exc:
            raise PemError(f"第 {idx} 个副本证书链解析失败: {exc}") from exc
        replicas.append(
            Replica(
                name=(row.get("name") or f"副本{idx}").strip() or f"副本{idx}",
                domain=domain,
                chains={
                    OLD: old_chain,
                    NEW: new_chain,
                },
            )
        )

    batches: list[Batch] = []
    for idx, row in enumerate(batch_rows, start=1):
        name = (row.get("name") or f"批次{idx}").strip() or f"批次{idx}"
        old_b = row["old_bundle"]
        new_b = row["new_bundle"]
        try:
            union_b = merge_pem_bundles(old_b, new_b)
        except PemError as exc:
            raise PemError(f"批次「{name}」根证书包解析失败: {exc}") from exc
        batches.append(
            Batch(
                name=name,
                bundles={OLD: old_b, NEW: new_b, TRANSITION: union_b},
            )
        )
    return Planner(replicas, batches, max_transition, verify_budget)
