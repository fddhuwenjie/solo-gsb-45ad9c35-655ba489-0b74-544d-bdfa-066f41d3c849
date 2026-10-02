"""无中断改线演练（seamless rewire drill）。

操作员提交**当前配置**（设备 / 插孔对 / 当前路由）、**目标路由列表**与
**必须始终保持通声的当前路由序号**。系统先按当前配置用 ``plan_all`` 重建
已安装跳线，再把同一份设备 / 插孔对配上目标路由做一次全新规划，得到目标
物理状态（插孔占用决定全常态 / 半常态 / 直通是否生效），然后在「当前接线
状态 → 目标接线状态」之间搜索一条逐步操作序列：

  remove 拆线：一根跳线两端全部拔下；
  move   换接：跳线一端保持，另一端拔下改插到另一个空闲插孔（含为解开
               插孔占用死锁的“临时让位”）；
  add    补线：在两个空闲插孔间插上一根跳线（优先复用此前拆下 / 多出的
               兼容跳线，其次才动用新跳线）；
  temp-* 临时跳线：只在过渡中存在、目标态并不需要的旁路跳线，单独编号、
               单独列出。其端点只能取**目标态也空闲**的插孔，因此永远
               不会占用目标补线需要的插孔。

硬约束：

  1. 物理可行性——插头只能插向当前空闲插孔（T 对输出、B 对输入）；
  2. 每一步完成后，所有**保护路由**在当前插头状态推导出的物理图上仍有
     source → target 完整通路（设备内部 + 生效常态 + 在役跳线）。全常态
     任一侧插线断常态、半常态仅下孔插线断常态（上孔插线并送）、直通无
     常态——全部由插头状态直接推导；
  3. 目标配置非法（含目标路由无解）时直接 400，不产生任何步骤；
  4. 找不到满足约束的顺序时 409，返回结构化阻塞原因（占用死锁插孔、
     必断常态 / 必拆跳线、无空闲旁路插孔等），同样不给出半成品步骤。

优化目标（字典序）：步骤数最少（每步恰为 2 个插 / 拔动作，即拆装次数
最少）→ 优先做直接完工 / 补线 / 拆线（驱动兼容跳线复用，少动用新跳线）
→ 最后才动用临时旁路跳线。搜索为确定性 Dijkstra，节点上限
SEARCH_NODE_CAP，超限返回带 ``inconclusive`` 标记的冲突（不伪称证明无解）。
"""

from heapq import heappop, heappush

from solver import (THROUGH, MODE_LABELS, ConfigError, _pid, plan_all)

SEARCH_NODE_CAP = 300_000
TEMP_ID_BASE = 9001                 # 临时跳线展示编号从 9001 起


class RewireError(ConfigError):
    """无中断改线请求非法。"""

    def __init__(self, message, code="REWIRE_INVALID", details=None):
        super().__init__(message, code=code, details=details or {})


class SeamlessConflict(ConfigError):
    """不存在能保持保护路由连续通声的操作顺序。"""

    def __init__(self, message, payload):
        super().__init__(message, code="NO_SEAMLESS_ORDER", details=payload)
        self.payload = payload


# ---- 物理状态仿真 ---------------------------------------------------------


class PhysModel:
    """按插头占用直接推导常态与信号通路的有向物理图。"""

    def __init__(self, devices, pairs):
        self.devices = devices
        self.pairs = pairs
        self.port_dir = {}
        for name, d in devices.items():
            for p in d["inputs"]:
                self.port_dir[f"{name}.{p}"] = (name, "in")
            for p in d["outputs"]:
                self.port_dir[f"{name}.{p}"] = (name, "out")
        self.pair_of = {}
        self.normal_from = {}              # 上端口 -> (下端口, pair)
        for p in pairs:
            self.pair_of[f"{p['id']}T"] = p
            self.pair_of[f"{p['id']}B"] = p
            if p["mode"] != THROUGH and p["top"] and p["bottom"]:
                self.normal_from[_pid(p["top"])] = (_pid(p["bottom"]), p)
        self.jack_of_port = {}
        self.port_of_jack = {}

    def bind_jacks(self, port_jack):
        self.jack_of_port = dict(port_jack)
        self.port_of_jack = {j: p for p, j in port_jack.items()}

    @staticmethod
    def plug_set(conns):
        plugs = set()
        for t, b in conns:
            plugs.add(t)
            plugs.add(b)
        return plugs

    def active_normal_ids(self, plugs):
        ids = set()
        for p in self.pairs:
            if p["mode"] == THROUGH or not (p["top"] and p["bottom"]):
                continue
            t, b = f"{p['id']}T", f"{p['id']}B"
            if p["mode"] == "full":
                ok = t not in plugs and b not in plugs
            else:  # half：仅下插孔插线断常态
                ok = b not in plugs
            if ok:
                ids.add(p["id"])
        return ids

    def trace(self, conns, src, dst, allow_edge=None):
        """求 src→dst 有向最短通路。

        conns:      frozenset((tJack, bJack)) 当前在役跳线；
        allow_edge: 额外允许的一条候选跳线 (tJack, bJack)，两端须当前空闲；
        返回 (端口列表, 边列表) 或 None。边形如：
          ("device", name) / ("normal", pair_id) /
          ("cable", tJack, bJack, is_extra)
        Dijkstra 代价 (候选边使用次数, 总跳数)，尽量不动用候选边。
        注意：候选边插上后可能断开其插孔所在对的常态，此处按“未插”估算，
        是否真的成立由调用方在完整状态上用 audible() 复核。
        """
        plugs = self.plug_set(conns)
        active = self.active_normal_ids(plugs)
        tmap = {t: b for t, b in conns}
        extra = None
        if allow_edge:
            et, eb = allow_edge
            if et not in plugs and eb not in plugs and et != eb:
                extra = (et, eb)

        dist = {src: (0, 0)}
        prev = {}
        pq = [((0, 0), src)]
        INF = (99, 99)
        while pq:
            d, u = heappop(pq)
            if d > dist.get(u, INF):
                continue
            if u == dst:
                break
            n_extra, hops = d
            dev, direction = self.port_dir[u]
            cands = []
            if direction == "in":
                for o in self.devices[dev]["outputs"]:
                    cands.append((f"{dev}.{o}", ("device", dev)))
            else:
                if u in self.normal_from:
                    bport, pair = self.normal_from[u]
                    if pair["id"] in active:
                        cands.append((bport, ("normal", pair["id"])))
                ju = self.jack_of_port.get(u)
                if ju and ju in tmap:
                    jv = tmap[ju]
                    cands.append((self.port_of_jack[jv], ("cable", ju, jv, False)))
                if extra and ju == extra[0] and n_extra == 0:
                    jv = extra[1]
                    cands.append((self.port_of_jack[jv], ("cable", ju, jv, True)))
            for v, edge in cands:
                nd = (n_extra + (1 if len(edge) == 4 and edge[3] else 0), hops + 1)
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = (u, edge)
                    heappush(pq, (nd, v))
        if dst not in dist or dist[dst][0] > (1 if allow_edge else 0):
            return None
        ports, edges = [dst], []
        u = dst
        while u != src:
            pu, edge = prev[u]
            edges.append(edge)
            ports.append(pu)
            u = pu
        ports.reverse()
        edges.reverse()
        return ports, edges

    def audible(self, conns, src, dst):
        return self.trace(conns, src, dst) is not None

    def bypass_candidates(self, conns, src, dst, goal_free_jacks):
        """枚举“目标态空闲、当前也空闲”插孔组成的临时旁路候选。

        返回 [(tJack, bJack)]，按路径总跳数与插孔编号排序，至多 3 条。
        """
        plugs = self.plug_set(conns)
        free_t = sorted(j for j in goal_free_jacks
                        if j.endswith("T") and j not in plugs)
        free_b = sorted(j for j in goal_free_jacks
                        if j.endswith("B") and j not in plugs)
        found = []
        for jt in free_t:
            for jb in free_b:
                r = self.trace(conns, src, dst, allow_edge=(jt, jb))
                if r is not None:
                    found.append((len(r[0]), jt, jb))
        found.sort(key=lambda x: (x[0], x[1], x[2]))
        return [(jt, jb) for _, jt, jb in found[:3]]


# ---- 请求解析 -------------------------------------------------------------


def _parse_route_list(raw, devices, ctx="目标路由"):
    """在既有设备表上校验目标路由列表（插孔对不变，故无需重建背板）。"""
    if not isinstance(raw, list) or not raw:
        raise RewireError(f"{ctx}必须是非空数组，每项形如 {{\"source\":..., \"target\":...}}")
    out = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise RewireError(f"{ctx}第 {i + 1} 项必须是 JSON 对象")

        def ref(field, want):
            v = r.get(field)
            if not isinstance(v, str) or "." not in v:
                raise RewireError(f"{ctx}第 {i + 1} 项：{field} 必须是 “设备.端口” 引用")
            dev, port = v.split(".", 1)
            if dev not in devices:
                raise RewireError(f"{ctx}第 {i + 1} 项：未知设备 “{dev}”")
            pool = devices[dev]["outputs" if want == "out" else "inputs"]
            if port not in pool:
                kind = "输出" if want == "out" else "输入"
                raise RewireError(f"{ctx}第 {i + 1} 项：设备 {dev} 没有{kind}端口 “{port}”")
            return v

        out.append({"source": ref("source", "out"), "target": ref("target", "in")})
    return out


def seamless_rewire(config, target_routes_raw, protected_raw):
    """生成无中断改线方案。成功返回结果 dict；失败抛 RewireError/SeamlessConflict。"""
    # ---- 1. 当前配置：重建已安装跳线 -----------------------------------
    try:
        cur_result, cur_plan = plan_all(config)
    except ConfigError as exc:
        raise RewireError(f"当前配置非法：{exc}", code="CONFIG_INVALID",
                          details=exc.details)
    devices = cur_plan["devices"]
    pairs = cur_plan["pairs"]
    cur_routes = cur_plan["routes"]
    s0 = cur_plan["solver"]
    phys = PhysModel(devices, pairs)
    phys.bind_jacks(cur_plan["port_jack"])

    # ---- 2. 目标路由：同设备 / 插孔对，替换路由后全新规划 --------------
    target_routes = _parse_route_list(target_routes_raw, devices)
    tgt_config = {"devices": config.get("devices") or [],
                  "pairs": config.get("pairs") or [],
                  "routes": target_routes}
    try:
        tgt_result, tgt_plan = plan_all(tgt_config)
    except ConfigError as exc:
        raise RewireError(f"目标配置非法：{exc}", code="TARGET_INVALID",
                          details=exc.details)
    if tgt_plan["conflicts"]:
        raise RewireError(
            f"目标路由列表存在 {len(tgt_plan['conflicts'])} 条无解路由，"
            "不得生成半成品改线方案",
            code="TARGET_INVALID",
            details={"conflicts": [
                {"route": c["route"], "source": c["source"], "target": c["target"],
                 "reasons": c["reasons"], "ports": c["ports"]}
                for c in tgt_plan["conflicts"]]})
    st = tgt_plan["solver"]

    # ---- 3. 保护路由 ----------------------------------------------------
    if protected_raw is None:
        protected_raw = []
    if not isinstance(protected_raw, list):
        raise RewireError("protectedRoutes 必须是路由序号数组（从 0 开始）",
                          code="PROTECTED_INVALID")
    protected, seen = [], set()
    for x in protected_raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise RewireError(f"保护路由序号必须是整数，得到 {x!r}",
                              code="PROTECTED_INVALID")
        if not 0 <= x < len(cur_routes):
            raise RewireError(f"保护路由序号越界：{x}（当前共 {len(cur_routes)} 条路由）",
                              code="PROTECTED_INVALID")
        if x in seen:
            raise RewireError(f"保护路由序号重复：{x}", code="PROTECTED_INVALID")
        seen.add(x)
        protected.append(x)

    tgt_keys = {(_pid(r["source"]), _pid(r["target"])) for r in tgt_plan["routes"]}
    protected_info = []
    for idx in protected:
        r = cur_routes[idx]
        key = (_pid(r["source"]), _pid(r["target"]))
        if key not in tgt_keys:
            raise RewireError(
                f"保护路由 {idx + 1}（{key[0]} → {key[1]}）不在目标路由列表中："
                "需要始终通声的路由在目标态也必须存在",
                code="PROTECTED_INVALID",
                details={"index": idx, "source": key[0], "target": key[1]})
        protected_info.append({"index": idx, "source": key[0], "target": key[1]})

    # ---- 4. 起止物理状态 ------------------------------------------------
    init_conns = frozenset((c["fromJack"], c["toJack"]) for c in s0.cables)
    goal_conns = frozenset((c["fromJack"], c["toJack"]) for c in st.cables)
    pr_keys = [(p["source"], p["target"]) for p in protected_info]

    # 保护路由在现状就不通声 → 无从“保持”
    silent = [p for p in protected_info
              if not phys.audible(init_conns, p["source"], p["target"])]
    if silent:
        raise SeamlessConflict(
            "保护路由在当前配置下本身就未通声，无法保持连续通声",
            _conflict_payload(protected_info, cur_result, tgt_result,
                              [{"type": "protected-silent-baseline",
                                "message": f"路由 {p['index'] + 1}（{p['source']} → "
                                           f"{p['target']}）在现状下没有完整通路",
                                "route": p["index"], "source": p["source"],
                                "target": p["target"]} for p in silent],
                              inconclusive=False, note=None))

    goal_plugs = PhysModel.plug_set(goal_conns)
    goal_free = frozenset(j for j in set(phys.pair_of) if j not in goal_plugs)

    # 终态健全性：目标跳线全部就位后，每条目标路由都必须在按插头推导的物理图上
    # 真实通声（防止贪心规划给出逐边检查通过、整体却不可达的连接集）
    goal_silent = []
    for r in tgt_plan["routes"]:
        src, dst = _pid(r["source"]), _pid(r["target"])
        if not phys.audible(goal_conns, src, dst):
            goal_silent.append({"source": src, "target": dst})
    if goal_silent:
        raise RewireError(
            "目标配置的跳线全部就位后仍有路由不通声（目标连接集与物理可达性不一致）",
            code="TARGET_INVALID", details={"silentRoutes": goal_silent})

    # ---- 5. 搜索 --------------------------------------------------------
    tgt_keys_all = [(_pid(r["source"]), _pid(r["target"])) for r in tgt_plan["routes"]]
    actions, inconclusive, note, blockers = _search(
        phys, init_conns, goal_conns, protected_info, goal_free, tgt_keys_all)
    if actions is None:
        raise SeamlessConflict(
            "不存在能保持保护路由连续通声的改线顺序",
            _conflict_payload(protected_info, cur_result, tgt_result,
                              list(blockers.values()),
                              inconclusive=inconclusive, note=note))

    # ---- 6. 回放并构造报告 ----------------------------------------------
    return _replay(phys, cur_plan, tgt_plan, init_conns, goal_conns, actions,
                   protected_info, goal_free, note)


# ---- 搜索 -----------------------------------------------------------------

# 动作的确定性次序（同代价时影响探索顺序，也倾向于先推进、少用临时跳线）
PRIORITY = {"complete": 0, "add": 1, "remove": 2, "park": 3,
            "temp-add": 4, "temp-remove": 5}


def _apply(conns, temps, action):
    """施加动作，返回 (新连接集, 新临时连接集)。"""
    kind, c, end, j = action
    new = set(conns)
    nt = set(temps)
    if kind in ("complete", "park"):
        new.remove(c)
        formed = (c[0], j) if end == "B" else (j, c[1])
        new.add(formed)
        if c in nt:                       # 兼容：临时跳线被移动（一般不会出现）
            nt.discard(c)
            nt.add(formed)
        return frozenset(new), frozenset(nt)
    if kind in ("add", "temp-add"):
        new.add(c)
        if kind == "temp-add":
            nt.add(c)
        return frozenset(new), frozenset(nt)
    new.discard(c)                        # remove / temp-remove
    nt.discard(c)
    return frozenset(new), frozenset(nt)


def _safe(phys, conns, pr_keys):
    return all(phys.audible(conns, src, dst) for src, dst in pr_keys)


def _search(phys, init_conns, goal_conns, protected_info, goal_free, tgt_keys):
    """确定性 Dijkstra。返回 (actions, inconclusive, note, blockers)。"""
    pr_keys = [(p["source"], p["target"]) for p in protected_info]
    blockers = {}

    def record(rec, key):
        if key not in blockers and len(blockers) < 40:
            rec.pop("key", None)
            blockers[key] = rec

    def gen(conns, temps, allow_temp):
        plugs = PhysModel.plug_set(conns)
        t_owner = {t: c for t, c in ((c[0], c) for c in conns)}
        b_owner = {b: c for b, c in ((c[1], c) for c in conns)}
        free_t = sorted(j for j in goal_free if j.endswith("T") and j not in plugs)
        free_b = sorted(j for j in goal_free if j.endswith("B") and j not in plugs)
        acts = []
        trace_cache = {}

        # 保护路由当前通路上的电缆集合（拆除 / 让位这些电缆需要先有替代通路）
        protected_conns = set()
        if pr_keys:
            for pi, (src, dst) in enumerate(pr_keys):
                tr = phys.trace(conns, src, dst)
                trace_cache[pi] = tr
                if tr:
                    for e in tr[1]:
                        if e[0] == "cable":
                            protected_conns.add((e[1], e[2]))

        for g in sorted(goal_conns - conns):
            t, b = g
            ct, cb = t_owner.get(t), b_owner.get(b)
            if ct is not None and cb is not None:
                # 两端都被占：若占用者不承载任何保护路由，直接拆即可，无需让位；
                # 只有承载保护路由、不能拆的电缆才临时让到 goal-free 同型插孔。
                parked = False
                if cb in protected_conns and free_b:
                    acts.append(("park", cb, "B", free_b[0]))
                    parked = True
                if ct in protected_conns and free_t:
                    acts.append(("park", ct, "T", free_t[0]))
                    parked = True
                record({
                    "type": "jack-occupied",
                    "message": f"目标插孔 {t} 被跳线 {ct[0]}→{ct[1]} 占用、"
                               f"{b} 被跳线 {cb[0]}→{cb[1]} 占用，需先临时让位才能补成 "
                               f"{t}→{b}",
                    "jacks": [t, b],
                    "occupants": [{"cable": [ct[0], ct[1]]}, {"cable": [cb[0], cb[1]]}],
                }, ("jack-occupied", t, b, ct, cb))
                if not parked:
                    # 未生成 park：拆除占用者由下方 remove 分支统一产生
                    pass
            elif ct is not None and b not in plugs:
                acts.append(("complete", ct, "B", b))
            elif cb is not None and t not in plugs:
                acts.append(("complete", cb, "T", t))
            else:
                acts.append(("add", g, None, None))

        for c in sorted(conns - goal_conns):
            acts.append(("temp-remove" if c in temps else "remove", c, None, None))

        # 只在没有安全的推进 / 让位动作时，才为“受威胁”的保护路由寻找临时旁路
        if allow_temp:
            base_acts = [a for a in acts if a[0] in ("complete", "add", "remove", "park")]
            safe_exists = any(
                _safe(phys, _apply(conns, temps, a)[0], pr_keys) for a in base_acts)
            if not safe_exists:
                threatened = []
                for p in protected_info:
                    src, dst = p["source"], p["target"]
                    if not phys.audible(conns, src, dst):
                        continue
                    if any(not _safe(phys, _apply(conns, temps, a)[0], [(src, dst)])
                           for a in base_acts):
                        threatened.append(p)
                if not threatened and base_acts:
                    # 所有推进动作都安全时不需要旁路；若无 base_acts（极端），仍尝试
                    threatened = [p for p in protected_info
                                  if phys.audible(conns, p["source"], p["target"])]
                for p in threatened[:2]:
                    cands = phys.bypass_candidates(conns, p["source"], p["target"],
                                                   goal_free)
                    if not cands:
                        record({
                            "type": "no-bypass-jack",
                            "message": f"保护路由 {p['index'] + 1}（{p['source']} → "
                                       f"{p['target']}）一旦拆线就会失声，而目标态空闲"
                                       "插孔不足以搭一条临时旁路",
                            "route": p["index"], "source": p["source"],
                            "target": p["target"],
                        }, ("no-bypass", p["index"]))
                    for edge in cands:
                        if edge not in conns:
                            acts.append(("temp-add", edge, None, None))
        return acts

    def record_unsafe(conns, temps, action):
        new_conns, _ = _apply(conns, temps, action)
        inserted = []
        if action[0] in ("complete", "park"):
            inserted = [action[3]]
        elif action[0] in ("add", "temp-add"):
            inserted = [action[1][0], action[1][1]]
        for p in protected_info:
            src, dst = p["source"], p["target"]
            if phys.audible(new_conns, src, dst):
                continue
            old = phys.trace(conns, src, dst)
            if old is None:
                continue
            for e in old[1]:
                if e[0] == "cable" and (e[1], e[2]) not in new_conns:
                    is_temp = (e[1], e[2]) in temps
                    record({
                        "type": "must-remove-cable",
                        "message": f"保护路由 {p['index'] + 1}（{src} → {dst}）当前经由跳线 "
                                   f"{e[1]}→{e[2]}" + ("（临时旁路）" if is_temp else "") +
                                   "，但该跳线必须在替代通路建立前拆动",
                        "route": p["index"], "source": src, "target": dst,
                        "cable": [e[1], e[2]], "temporary": is_temp,
                    }, ("cable", p["index"], e[1], e[2]))
                elif e[0] == "normal":
                    pid = e[1]
                    hit = [x for x in inserted if int(x[:-1]) == pid]
                    if hit:
                        pair = phys.pair_of[f"{pid}T"]
                        record({
                            "type": "must-break-normal",
                            "message": f"保护路由 {p['index'] + 1}（{src} → {dst}）依赖第 {pid} 对"
                                       f"（{MODE_LABELS[pair['mode']]}）常态，向 {hit[0]} 插线会"
                                       "断开它，且该步没有能保持通声的替代通路 / 旁路",
                            "route": p["index"], "source": src, "target": dst,
                            "pair": pid, "plugJack": hit[0],
                        }, ("normal", p["index"], pid, hit[0]))

    dist = {init_conns: 0}
    temps_at = {init_conns: frozenset()}
    parent = {init_conns: None}
    pq = [(0, 0, init_conns)]
    counter = 0
    explored = 0
    goal_node = None

    while pq:
        d, _, conns = heappop(pq)
        if d != dist.get(conns):
            continue
        explored += 1
        if conns == goal_conns:
            # 终态除连接集一致外，所有目标路由都必须真实通声（防止规划器的
            # 贪心连接集与逐边可达性不一致——例如设备回环造成的伪路径）
            if all(phys.audible(conns, src, dst) for src, dst in tgt_keys):
                goal_node = conns
                break
            else:
                record({
                    "type": "goal-not-audible",
                    "message": "目标接线全部就位，但仍有目标路由在该物理状态下不通声"
                               "（规划器给出的目标连接集与逐边可达性不一致）",
                }, ("goal-not-audible",))
                continue
        if explored >= SEARCH_NODE_CAP:
            return None, True, ("搜索节点超过上限：已列出搜索中发现的阻塞点，"
                               "但未能在时限内证明不存在可行顺序"), blockers
        acts = gen(conns, temps_at[conns], allow_temp=True)
        acts.sort(key=lambda a: (PRIORITY[a[0]], _action_key(a)))
        for a in acts:
            new_conns, new_temps = _apply(conns, temps_at[conns], a)
            if not _safe(phys, new_conns, pr_keys):
                record_unsafe(conns, temps_at[conns], a)
                continue
            nd = d + 1
            if nd < dist.get(new_conns, 1 << 30):
                dist[new_conns] = nd
                temps_at[new_conns] = new_temps
                parent[new_conns] = (conns, a)
                counter += 1
                heappush(pq, (nd, counter, new_conns))

    if goal_node is None:
        return None, False, None, blockers

    chain, c = [], goal_node
    while parent[c] is not None:
        prev_c, a = parent[c]
        chain.append(a)
        c = prev_c
    chain.reverse()
    return chain, False, None, blockers


def _action_key(a):
    kind, c, end, j = a
    if kind in ("complete", "park"):
        return (c[0], c[1], end or "", j or "")
    return (c[0] if c else "", c[1] if c else "", "", "")


# ---- 回放与报告 -----------------------------------------------------------


def _union_routes(cur_plan, tgt_plan, protected):
    out = {}
    for i, r in enumerate(cur_plan["routes"]):
        key = (_pid(r["source"]), _pid(r["target"]))
        e = out.setdefault(key, {"source": key[0], "target": key[1],
                                 "protected": False})
        e["cur"] = i
        if i in protected:
            e["protected"] = True
    for i, r in enumerate(tgt_plan["routes"]):
        key = (_pid(r["source"]), _pid(r["target"]))
        e = out.setdefault(key, {"source": key[0], "target": key[1],
                                 "protected": False})
        e["tgt"] = i
    return out


def _audible_map(phys, conns, routes):
    return {key: phys.audible(conns, v["source"], v["target"])
            for key, v in routes.items()}


def _trace_map(phys, conns, routes):
    return {key: phys.trace(conns, v["source"], v["target"])
            for key, v in routes.items()}


def _path_touches(trace, touched_pairs, touched_conns):
    if not trace:
        return False
    for e in trace[1]:
        if e[0] == "normal" and e[1] in touched_pairs:
            return True
        if e[0] == "cable" and (e[1], e[2]) in touched_conns:
            return True
    return False


def _snapshot(pairs, conns, label_of, jack_port):
    plugs = PhysModel.plug_set(conns)
    pair_out = []
    for p in pairs:
        t, b = f"{p['id']}T", f"{p['id']}B"
        if p["mode"] == THROUGH or not (p["top"] and p["bottom"]):
            active = False
        elif p["mode"] == "full":
            active = t not in plugs and b not in plugs
        else:
            active = b not in plugs
        pair_out.append({
            "id": p["id"], "mode": p["mode"], "modeLabel": MODE_LABELS[p["mode"]],
            "topJack": t, "bottomJack": b,
            "topPort": _pid(p["top"]) if p["top"] else None,
            "bottomPort": _pid(p["bottom"]) if p["bottom"] else None,
            "normalActive": active,
        })
    cables = []
    for conn in sorted(conns):
        lab = label_of.get(conn)
        if lab is None:
            continue
        kind, cid = lab
        cables.append({"id": cid, "fromJack": conn[0], "toJack": conn[1],
                       "fromPort": jack_port.get(conn[0]),
                       "toPort": jack_port.get(conn[1]),
                       "temporary": kind == "temp"})
    cables.sort(key=lambda c: c["id"])
    return {"pairs": pair_out, "cables": cables}


def _label_text(lab):
    kind, cid = lab
    if kind == "temp":
        return f"临时跳线 T{cid}"
    if kind == "new":
        return f"新跳线 #{cid}"
    return f"跳线 #{cid}"


def _replay(phys, cur_plan, tgt_plan, init_conns, goal_conns, actions,
            protected_info, goal_free, note):
    s0, st = cur_plan["solver"], tgt_plan["solver"]
    jack_port = dict(s0.jack_port)
    pairs = cur_plan["pairs"]

    origin = {c["id"]: (c["fromJack"], c["toJack"]) for c in s0.cables}
    label_of = {ep: ("orig", cid) for cid, ep in origin.items()}
    spare = []                                    # 已拆下可复用的原跳线 id
    next_new = max([c["id"] for c in s0.cables]
                   + [c["id"] for c in st.cables] + [0]) + 1
    next_temp = 1
    temp_meta = {}                                # temp 序号 -> 元信息
    temp_use = {}                                 # temp 序号 -> 保护路由序号集合

    routes = _union_routes(cur_plan, tgt_plan,
                           {p["index"] for p in protected_info})

    conns = init_conns
    prev_aud = _audible_map(phys, conns, routes)
    prev_traces = _trace_map(phys, conns, routes)
    steps = []

    for seq, action in enumerate(actions, 1):
        kind, c, end, j = action
        old_conns = conns
        jack_changes, touched_conns, formed = [], {c}, None

        if kind in ("complete", "park"):
            lab = label_of.pop(c)
            formed = (c[0], j) if end == "B" else (j, c[1])
            label_of[formed] = lab
            moved = c[1] if end == "B" else c[0]
            jack_changes = [
                {"jack": moved, "port": jack_port.get(moved), "change": "unplug"},
                {"jack": j, "port": jack_port.get(j), "change": "plug"},
            ]
            touched_conns.add(formed)
            end_cn = "下端 B" if end == "B" else "上端 T"
            summary = (f"{_label_text(lab)} 保持插接"
                       + ("（临时让位）" if kind == "park" else "")
                       + f"，将{end_cn}从 {moved}（{jack_port.get(moved)}）"
                         f"换插到 {j}（{jack_port.get(j)}）")
            action_label = "临时让位换接" if kind == "park" else "换接"
        elif kind in ("add", "temp-add"):
            if kind == "temp-add":
                tid = next_temp
                next_temp += 1
                lab = ("temp", tid)
                disp_id = TEMP_ID_BASE + tid - 1
                temp_meta[tid] = {"tempNo": tid, "displayId": disp_id,
                                  "label": f"临时跳线 T{tid}",
                                  "fromJack": c[0], "toJack": c[1],
                                  "fromPort": jack_port.get(c[0]),
                                  "toPort": jack_port.get(c[1]),
                                  "addStep": seq, "removeStep": None}
            elif spare:
                lab = ("orig", spare.pop(0))
            else:
                lab = ("new", next_new)
                next_new += 1
            label_of[c] = lab
            jack_changes = [
                {"jack": c[0], "port": jack_port.get(c[0]), "change": "plug"},
                {"jack": c[1], "port": jack_port.get(c[1]), "change": "plug"},
            ]
            formed = c
            summary = (f"{_label_text(lab)}：{c[0]}（{jack_port.get(c[0])}） ↔ "
                       f"{c[1]}（{jack_port.get(c[1])}）"
                       + ("（临时旁路，替代通路建立后拆除）" if kind == "temp-add"
                          else ("（复用拆下的兼容跳线）" if lab[0] == "orig"
                                and kind == "add" else "")))
            action_label = "接入临时旁路跳线" if kind == "temp-add" else "补线"
        else:  # remove / temp-remove
            lab = label_of.pop(c)
            is_temp = lab[0] == "temp"
            if is_temp:
                temp_meta[lab[1]]["removeStep"] = seq
            else:
                spare.append(lab[1])
                spare.sort()
            jack_changes = [
                {"jack": c[0], "port": jack_port.get(c[0]), "change": "unplug"},
                {"jack": c[1], "port": jack_port.get(c[1]), "change": "unplug"},
            ]
            summary = (f"拆除{_label_text(lab)}：拔下 {c[0]}（{jack_port.get(c[0])}）"
                       f"与 {c[1]}（{jack_port.get(c[1])}）")
            action_label = "拆除临时旁路跳线" if is_temp else "拆线"

        conns, _temps = _apply(old_conns, frozenset(), action)
        touched_pairs = {int(jc["jack"][:-1]) for jc in jack_changes}

        new_aud = _audible_map(phys, conns, routes)
        new_traces = _trace_map(phys, conns, routes)
        affected = []
        for key, info in routes.items():
            if (prev_aud[key] != new_aud[key]
                    or _path_touches(prev_traces[key], touched_pairs, touched_conns)
                    or _path_touches(new_traces[key], touched_pairs, touched_conns)):
                affected.append({
                    "source": info["source"], "target": info["target"],
                    "currentIndex": info.get("cur"),
                    "targetIndex": info.get("tgt"),
                    "protected": info["protected"],
                    "audibleBefore": prev_aud[key],
                    "audibleAfter": new_aud[key],
                })
        affected.sort(key=lambda x: (x["currentIndex"] is None,
                                     x["currentIndex"] if x["currentIndex"] is not None else 0,
                                     x["targetIndex"] if x["targetIndex"] is not None else 0))

        protected_status = []
        for p in protected_info:
            tr = phys.trace(conns, p["source"], p["target"])
            uses_temp = []
            if tr:
                for e in tr[1]:
                    if e[0] == "cable":
                        lab_e = label_of.get((e[1], e[2]))
                        if lab_e and lab_e[0] == "temp":
                            uses_temp.append(lab_e[1])
                            temp_use.setdefault(lab_e[1], set()).add(p["index"])
            protected_status.append({
                "index": p["index"], "routeNo": p["index"] + 1,
                "source": p["source"], "target": p["target"],
                "audible": tr is not None,
                "viaTemporary": sorted(set(uses_temp)),
                "path": None if tr is None else [
                    {"port": pp, "jack": phys.jack_of_port.get(pp)} for pp in tr[0]],
            })

        steps.append({
            "seq": seq, "action": kind, "actionLabel": action_label,
            "summary": summary,
            "cable": {"label": _label_text(lab), "kind": lab[0],
                      "temporary": lab[0] == "temp",
                      "fromJack": c[0], "toJack": c[1]},
            "formedCable": None if formed is None else {
                "fromJack": formed[0], "toJack": formed[1],
                "label": _label_text(label_of.get(formed, lab)),
            },
            "jackChanges": jack_changes,
            "affectedRoutes": affected,
            "protected": protected_status,
            "protectedAllAudible": all(x["audible"] for x in protected_status),
            "snapshot": _snapshot(pairs, conns, label_of, jack_port),
        })
        prev_aud, prev_traces = new_aud, new_traces

    # ---- 跳线复用归类 ----------------------------------------------------
    def reuse_entry(gc):
        t, b = gc
        target_id = next((c["id"] for c in st.cables
                          if (c["fromJack"], c["toJack"]) == gc), None)
        base = {"fromJack": t, "toJack": b,
                "fromPort": jack_port.get(t), "toPort": jack_port.get(b),
                "targetCableId": target_id}
        lab = label_of.get(gc)
        if lab is None:
            return {**base, "reuse": "unknown", "label": "?"}
        kind, cid = lab
        if kind == "new":
            return {**base, "reuse": "new", "sourceCableId": None,
                    "displayId": cid, "label": f"新跳线 #{cid}"}
        if kind == "temp":
            return {**base, "reuse": "temp", "label": f"临时跳线 T{cid}"}
        ot, ob = origin[cid]
        if (ot, ob) == (t, b):
            reuse = "unchanged"
        elif ot == t or ob == b:
            reuse = "move"
        else:
            reuse = "recycle"
        return {**base, "reuse": reuse, "sourceCableId": cid,
                "label": f"跳线 #{cid}",
                "originalFromJack": ot, "originalToJack": ob}

    goal_entries = [reuse_entry(gc) for gc in sorted(goal_conns)]
    unchanged = [e for e in goal_entries if e["reuse"] == "unchanged"]
    moved = [e for e in goal_entries if e["reuse"] == "move"]
    recycled = [e for e in goal_entries if e["reuse"] == "recycle"]
    new_cables = [e for e in goal_entries if e["reuse"] == "new"]
    used_orig = {e["sourceCableId"] for e in goal_entries
                 if e["reuse"] in ("unchanged", "move", "recycle")}
    removed = [{
        "sourceCableId": cid, "label": f"跳线 #{cid}",
        "fromJack": origin[cid][0], "toJack": origin[cid][1],
        "fromPort": jack_port.get(origin[cid][0]),
        "toPort": jack_port.get(origin[cid][1]),
    } for cid in sorted(set(origin) - used_orig)]

    temporary = []
    for tid in sorted(temp_meta):
        m = temp_meta[tid]
        temporary.append({**m, "protects": sorted(temp_use.get(tid, set())),
                          "note": "仅过渡使用，目标态已拆除"
                                  if m["removeStep"] is not None else "目标态仍在（异常）"})

    return {
        "ok": True,
        "protectedRoutes": protected_info,
        "steps": steps,
        "stepCount": len(steps),
        "metrics": {
            "plugOps": 2 * len(steps),
            "removeCount": sum(1 for a in actions if a[0] == "remove"),
            "moveCount": sum(1 for a in actions if a[0] in ("complete", "park")),
            "parkCount": sum(1 for a in actions if a[0] == "park"),
            "addCount": sum(1 for a in actions if a[0] == "add"),
            "tempAddCount": sum(1 for a in actions if a[0] == "temp-add"),
            "tempCableCount": len(temporary),
            "newCableCount": len(new_cables),
            "reusedCableCount": len(unchanged) + len(moved) + len(recycled),
        },
        "reuse": {"unchanged": unchanged, "moved": moved,
                  "recycled": recycled, "new": new_cables,
                  "removedOriginal": removed},
        "temporaryCables": temporary,
        "baseline": {
            "cableCount": len(s0.cables),
            "pairs": cur_plan["pairs_out"], "cables": [dict(c) for c in s0.cables],
            "ok": not cur_plan["conflicts"],
            "conflicts": [info["index"] for info in cur_plan["planned"]
                          if not info["ok"]],
        },
        "target": {
            "cableCount": len(st.cables),
            "pairs": tgt_plan["pairs_out"], "cables": [dict(c) for c in st.cables],
            "routes": [{"index": i, "source": _pid(r["source"]),
                        "target": _pid(r["target"])}
                       for i, r in enumerate(tgt_plan["routes"])],
            "ok": not tgt_plan["conflicts"],
        },
        "feedbackLoops": tgt_plan["loops"],
        "solverNote": note,
    }


def _conflict_payload(protected_info, cur_result, tgt_result, blockers,
                      inconclusive, note):
    return {
        "protectedRoutes": protected_info,
        "blockers": blockers,
        "inconclusive": bool(inconclusive),
        "solverNote": note,
        "baseline": {
            "cableCount": cur_result["cableCount"],
            "pairs": cur_result["pairs"], "cables": cur_result["cables"],
            "ok": cur_result["ok"],
            "conflicts": [r["index"] for r in cur_result["routes"]
                          if r["status"] == "conflict"],
        },
        "target": {
            "cableCount": tgt_result["cableCount"],
            "pairs": tgt_result["pairs"], "cables": tgt_result["cables"],
            "ok": tgt_result["ok"],
        },
    }
