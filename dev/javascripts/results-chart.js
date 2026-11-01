// The results chart on docs/baseline-results.md: each task's oracle and
// ceiling against its expert, or both controllers against the ceiling. The
// rows come from the page itself (#tg-results-data), which
// scripts/generate_baseline_table.py writes from the recorded results.
// document$ re-runs it on every page Material's instant navigation loads.
document$.subscribe(() => {
  const host = document.querySelector(".tg-results-chart");
  const source = document.getElementById("tg-results-data");
  if (!host || !source) return;
  const rows = JSON.parse(source.textContent);
  const L10 = (a, b) => (a > 0 ? Math.log10(a / b) : -Infinity);
  const fmt = (x) =>
    x === 0 ? "0"
      : Math.abs(x) >= 1000 || Math.abs(x) < 0.01 ? x.toExponential(2).replace("e+", "e")
      : String(+x.toPrecision(3));
  const times = (d) => {
    if (!isFinite(d)) return "at zero cost";
    const f = Math.pow(10, Math.abs(d));
    const n = f >= 100 ? Math.round(f).toLocaleString("en") : f.toFixed(f >= 10 ? 0 : 1);
    return d <= 0 ? `${n}× cheaper` : `${n}× costlier`;
  };
  const groups = [...new Set(rows.map((r) => r.group))];
  const order = [];
  groups.forEach((g) => {
    order.push({ h: g });
    rows.filter((r) => r.group === g)
      .sort((a, b) => L10(b.expert, b.ceiling) - L10(a.expert, a.ceiling))
      .forEach((r) => order.push(r));
  });
  const MODES = {
    expert: {
      lo: -5, hi: 1.5, shade: false,
      at: (r) => ({ e: 0, o: L10(r.oracle, r.expert), c: L10(r.ceiling, r.expert) }),
      ticks: [[-5, "10⁵×"], [-4, "10⁴×"], [-3, "1000×"], [-2, "100×"], [-1, "10×"], [0, "expert"], [1, "10× costlier"]],
      title: "Hold cost relative to the expert (log scale; left is cheaper)",
    },
    ceiling: {
      lo: -3, hi: 4.5, shade: true,
      at: (r) => ({ e: L10(r.expert, r.ceiling), o: L10(r.oracle, r.ceiling), c: 0 }),
      ticks: [[-3, "1/1000"], [-2, "1/100"], [-1, "1/10"], [0, "ceiling"], [1, "10×"], [2, "100×"], [3, "1000×"], [4, "10⁴×"]],
      title: "Hold cost relative to the ceiling (log scale; left is better)",
    },
  };
  host.innerHTML =
    '<div class="tg-bar"><span>Anchor</span>' +
    '<button type="button" data-mode="expert" aria-pressed="true">Expert</button>' +
    '<button type="button" data-mode="ceiling" aria-pressed="false">Ceiling</button>' +
    '<button type="button" class="tg-table-toggle">Show table</button></div>' +
    '<div class="tg-legend"><span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="var(--tg-s1)"/></svg>Expert</span>' +
    '<span><svg width="12" height="12" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill="var(--tg-s2)"/></svg>Oracle (full-state MPC)</span>' +
    '<span><svg width="8" height="14" aria-hidden="true"><rect x="3" width="2" height="14" rx="1" fill="var(--tg-cl)"/></svg>Ceiling (instrument resolution)</span></div>' +
    '<div class="tg-plot"></div><div class="tg-tip" role="tooltip"></div><div class="tg-table"></div>';
  const plot = host.querySelector(".tg-plot");
  const tip = host.querySelector(".tg-tip");
  const W = 680, L = 170, R = 664, T = 8, RH = 19, HH = 26, AX = 44, NS = "http://www.w3.org/2000/svg";
  let mode = "expert";
  function render() {
    const M = MODES[mode];
    const X = (v) => L + ((Math.max(M.lo, Math.min(M.hi, v)) - M.lo) / (M.hi - M.lo)) * (R - L);
    let y = T;
    order.forEach((r) => { r.y = y + (r.h ? HH : RH) / 2; y += r.h ? HH : RH; });
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${W} ${y + AX}`);
    s.setAttribute("width", "100%");
    s.setAttribute("role", "img");
    s.setAttribute("aria-label", mode === "expert"
      ? "Each task's oracle hold cost and ceiling relative to its expert, in decades."
      : "Each task's expert and oracle hold costs relative to its ceiling, in decades.");
    const el = (t, a, p = s) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
    const label = (p, x, yy, txt, cls, anchor = "start") => { const e = el("text", { x, y: yy, "text-anchor": anchor, class: cls }, p); e.textContent = txt; return e; };
    if (M.shade) el("rect", { x: L, y: T, width: X(0) - L, height: y - T, class: "tg-shade" });
    M.ticks.forEach(([v, l]) => {
      el("line", { x1: X(v), x2: X(v), y1: T, y2: y, class: v === 0 ? "tg-ref" : "tg-grid" });
      label(s, X(v), y + 16, l, "tg-muted", "middle");
    });
    label(s, (L + R) / 2, y + 36, M.title, "tg-sub", "middle");
    order.forEach((r) => {
      if (r.h) {
        label(s, 0, r.y + 5, r.h, "tg-head");
        el("line", { x1: 0, x2: R, y1: r.y + 11, y2: r.y + 11, class: "tg-grid" });
        return;
      }
      const p = M.at(r);
      const g = el("g", { tabindex: "0", "aria-label": `${r.name}: oracle ${times(L10(r.oracle, r.expert))} than the expert` });
      const hit = el("rect", { x: 0, y: r.y - RH / 2, width: R, height: RH, rx: 4, class: "tg-hit" }, g);
      label(g, L - 24, r.y + 4, r.name, "tg-name", "end");
      el("line", { x1: X(p.e), x2: X(p.o), y1: r.y, y2: r.y, class: "tg-conn" }, g);
      [p.e, p.o, p.c].forEach((v) => {
        if (v < M.lo) label(g, X(M.lo) - 11, r.y + 4, "‹", "tg-muted", "middle");
        if (v > M.hi) label(g, X(M.hi) + 9, r.y + 4, "›", "tg-muted", "middle");
      });
      el("rect", { x: X(p.c) - 1, y: r.y - 7, width: 2, height: 14, rx: 1, class: "tg-ceiling" }, g);
      el("rect", { x: X(p.o) - 5, y: r.y - 5, width: 10, height: 10, rx: 1.5, class: "tg-oracle" }, g);
      el("circle", { cx: X(p.e), cy: r.y, r: 5.5, class: "tg-expert" }, g);
      const show = () => {
        hit.classList.add("tg-on");
        tip.innerHTML =
          `<div class="tg-tip-name">${r.name}</div>` +
          `<div>Oracle, holding: <b>${times(L10(r.oracle, r.expert))}</b> than the expert</div>` +
          `<div>Oracle, whole episode: <b>${times(L10(r.oracle_gain, r.expert_gain))}</b></div>` +
          `<div>Ceiling: ${times(L10(r.ceiling, r.expert))} than the expert</div>` +
          `<div class="tg-tip-raw">Hold costs: expert ${fmt(r.expert)}, oracle ${fmt(r.oracle)}, ceiling ${fmt(r.ceiling)}</div>`;
        const w = plot.getBoundingClientRect().width, sc = w / W;
        tip.style.top = `${(r.y + 12) * sc + plot.offsetTop}px`;
        tip.style.left = `${Math.min(Math.max(0, Math.min(X(p.e), X(p.o)) * sc - 60), w - 240)}px`;
        tip.style.opacity = 1;
      };
      const hide = () => { hit.classList.remove("tg-on"); tip.style.opacity = 0; };
      g.addEventListener("mouseenter", show); g.addEventListener("mouseleave", hide);
      g.addEventListener("focus", show); g.addEventListener("blur", hide);
    });
    plot.replaceChildren(s);
  }
  host.querySelectorAll("button[data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.dataset.mode;
      host.querySelectorAll("button[data-mode]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      render();
    }));
  const toggle = host.querySelector(".tg-table-toggle"), table = host.querySelector(".tg-table");
  toggle.addEventListener("click", () => {
    if (table.innerHTML) { table.innerHTML = ""; toggle.textContent = "Show table"; return; }
    table.innerHTML =
      "<table><thead><tr><th>Task</th><th>Oracle vs expert, hold</th><th>Oracle vs expert, episode</th><th>Ceiling vs expert</th></tr></thead><tbody>" +
      order.filter((r) => !r.h).map((r) =>
        `<tr><td><code>${r.name}</code></td><td>${times(L10(r.oracle, r.expert))}</td><td>${times(L10(r.oracle_gain, r.expert_gain))}</td><td>${times(L10(r.ceiling, r.expert))}</td></tr>`).join("") +
      "</tbody></table>";
    toggle.textContent = "Hide table";
  });
  render();
});
