"use strict";

const API = "/api";
const HUB = "/hub"; // the alert hub (Alertmanager webhook receiver), see src/alerts/hub.py
const HISTORY_KEY = "churn.history";
const HISTORY_MAX = 100;
const YES_NO = [["Yes", "Có"], ["No", "Không"]];
const ADDON = [["Yes", "Có"], ["No", "Không"], ["No internet service", "Không có internet"]];

// One entry per model input. `kind` is select or number; values are the raw Telco categories the API expects.
const GROUPS = [
  {
    title: "Hồ sơ khách hàng",
    fields: [
      { name: "gender", label: "Giới tính", kind: "select", options: [["Female", "Nữ"], ["Male", "Nam"]] },
      { name: "SeniorCitizen", label: "Người cao tuổi", kind: "select", options: [["0", "Không"], ["1", "Có"]], int: true },
      { name: "Partner", label: "Có bạn đời", kind: "select", options: YES_NO },
      { name: "Dependents", label: "Có người phụ thuộc", kind: "select", options: YES_NO },
      { name: "tenure", label: "Thời gian gắn bó (tháng)", kind: "number", min: 0, max: 120, step: 1, required: true },
    ],
  },
  {
    title: "Dịch vụ đang dùng",
    fields: [
      { name: "PhoneService", label: "Dịch vụ điện thoại", kind: "select", options: YES_NO },
      { name: "MultipleLines", label: "Nhiều đường dây", kind: "select", options: [["Yes", "Có"], ["No", "Không"], ["No phone service", "Không có điện thoại"]] },
      { name: "InternetService", label: "Internet", kind: "select", options: [["DSL", "DSL"], ["Fiber optic", "Cáp quang"], ["No", "Không dùng"]] },
      { name: "OnlineSecurity", label: "Bảo mật trực tuyến", kind: "select", options: ADDON },
      { name: "OnlineBackup", label: "Sao lưu trực tuyến", kind: "select", options: ADDON },
      { name: "DeviceProtection", label: "Bảo vệ thiết bị", kind: "select", options: ADDON },
      { name: "TechSupport", label: "Hỗ trợ kỹ thuật", kind: "select", options: ADDON },
      { name: "StreamingTV", label: "Truyền hình", kind: "select", options: ADDON },
      { name: "StreamingMovies", label: "Phim", kind: "select", options: ADDON },
    ],
  },
  {
    title: "Hợp đồng và thanh toán",
    fields: [
      { name: "Contract", label: "Loại hợp đồng", kind: "select", options: [["Month-to-month", "Theo tháng"], ["One year", "1 năm"], ["Two year", "2 năm"]] },
      { name: "PaperlessBilling", label: "Hóa đơn điện tử", kind: "select", options: YES_NO },
      { name: "PaymentMethod", label: "Phương thức thanh toán", kind: "select", options: [["Electronic check", "Séc điện tử"], ["Mailed check", "Séc gửi thư"], ["Bank transfer (automatic)", "Chuyển khoản tự động"], ["Credit card (automatic)", "Thẻ tín dụng tự động"]] },
      { name: "MonthlyCharges", label: "Cước hàng tháng ($)", kind: "number", min: 0, max: 500, step: 0.05, required: true },
      { name: "TotalCharges", label: "Tổng cước đã trả ($, bỏ trống nếu khách mới)", kind: "number", min: 0, step: 0.05 },
    ],
  },
];
const FIELDS = GROUPS.flatMap((g) => g.fields);
const ADDON_NAMES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"];

const SAMPLES = {
  high: {
    gender: "Female", SeniorCitizen: "0", Partner: "No", Dependents: "No", tenure: 3,
    PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
    OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "No", StreamingMovies: "No",
    Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check", MonthlyCharges: 85.5, TotalCharges: 256.5,
  },
  low: {
    gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes", tenure: 60,
    PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL",
    OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No",
    Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Credit card (automatic)", MonthlyCharges: 62.4, TotalCharges: 3744,
  },
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const money = (x) => `${x < 0 ? "-" : ""}$${Math.abs(Math.round(x)).toLocaleString("en-US")}`;
const labelOf = (field, value) => (field.options?.find(([v]) => v === String(value)) ?? [value, value])[1];

let modelInfo = null; // GET /v1/model, null until loaded or when no model
let challenger = null; // GET /v1/model/challenger, null when nothing awaits approval
let versions = []; // GET /v1/model/versions, newest first
let alertState = null; // GET /hub/alerts, null when the hub cannot be reached

// ---------- tabs ----------
function showTab(name) {
  $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  if (name === "model") renderModel();
  if (name === "history") renderHistory();
  if (name === "alerts") renderAlerts();
  history.replaceState(null, "", `#${name}`);
}
$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]");
  if (b) showTab(b.dataset.tab);
});

// ---------- form ----------
function buildForm() {
  $("#form-groups").innerHTML = GROUPS.map((g) => `
    <fieldset><legend>${esc(g.title)}</legend><div class="fields">
      ${g.fields.map((f) => `<label>${esc(f.label)}${f.kind === "select"
        ? `<select name="${f.name}">${f.options.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("")}</select>`
        : `<input name="${f.name}" type="number" min="${f.min ?? ""}" max="${f.max ?? ""}" step="${f.step}" inputmode="decimal" ${f.required ? "required" : ""}>`}</label>`).join("")}
    </div></fieldset>`).join("");
  fillForm(SAMPLES.high);
  $$("#form select").forEach((s) => s.addEventListener("change", syncDependencies));
  syncDependencies();
}

function fillForm(values) {
  FIELDS.forEach((f) => {
    const el = $(`[name="${f.name}"]`);
    el.value = values[f.name] ?? "";
  });
  syncDependencies();
}

// Telco data only has these combinations: no internet -> add-ons are "No internet service", no phone -> no lines.
function syncDependencies() {
  const noNet = $('[name="InternetService"]').value === "No";
  ADDON_NAMES.forEach((n) => {
    const el = $(`[name="${n}"]`);
    if (noNet) el.value = "No internet service";
    else if (el.value === "No internet service") el.value = "No";
    el.disabled = noNet;
    $$("option", el).forEach((o) => { o.hidden = o.value === "No internet service" && !noNet; });
  });
  const noPhone = $('[name="PhoneService"]').value === "No";
  const lines = $('[name="MultipleLines"]');
  if (noPhone) lines.value = "No phone service";
  else if (lines.value === "No phone service") lines.value = "No";
  lines.disabled = noPhone;
  $$("option", lines).forEach((o) => { o.hidden = o.value === "No phone service" && !noPhone; });
}

function readForm() {
  const out = {};
  for (const f of FIELDS) {
    const raw = $(`[name="${f.name}"]`).value.trim();
    if (f.kind === "number") {
      if (raw === "") {
        if (f.required) throw new Error(`Hãy nhập "${f.label}".`);
        out[f.name] = null;
        continue;
      }
      const n = Number(raw);
      if (!Number.isFinite(n) || n < (f.min ?? -Infinity)) throw new Error(`"${f.label}" không hợp lệ.`);
      out[f.name] = n;
    } else {
      out[f.name] = f.int ? Number(raw) : raw;
    }
  }
  return out;
}

// ---------- risk ----------
function economics() {
  const e = modelInfo?.economics ?? {};
  return {
    months: e.value_months ?? 12,
    cost: e.retention_cost_rate ?? 0.1,
    success: e.retention_success ?? 0.3,
  };
}
const breakEven = () => economics().cost / economics().success;

function riskLevel(p) {
  const b = breakEven();
  if (p >= b) return { key: "high", label: "Cao" };
  if (p >= 0.6 * b) return { key: "mid", label: "Trung bình" };
  return { key: "low", label: "Thấp" };
}

function expectedProfit(p, monthly) {
  const { months, cost, success } = economics();
  const value = monthly * months;
  return p * success * value - cost * value;
}

function renderResult(entry) {
  const { probability: p, inputs } = entry;
  const level = riskLevel(p);
  const b = breakEven();
  const profit = expectedProfit(p, inputs.MonthlyCharges);
  const e = economics();
  const worth = profit > 0;
  $("#result").innerHTML = `
    <div class="pct">${pct(p)}</div>
    <p class="muted" style="margin:4px 0 10px">xác suất khách hàng rời bỏ dịch vụ</p>
    <span class="badge ${level.key}">Rủi ro ${level.label.toLowerCase()}</span>
    <div class="meter" role="img" aria-label="Xác suất ${pct(p)}, ngưỡng hòa vốn ${pct(b, 0)}">
      <i style="width:${Math.min(100, p * 100)}%"></i><b style="left:${Math.min(100, b * 100)}%"></b>
    </div>
    <div class="meter-labels"><span>0%</span><span>Ngưỡng hòa vốn ${pct(b, 0)}</span><span>100%</span></div>
    <div class="advice">
      <b>${worth ? "Nên liên hệ giữ chân" : "Chưa cần ưu đãi"}</b>
      <p style="margin:6px 0 0">Lợi nhuận kỳ vọng nếu liên hệ: <b>${money(profit)}</b>
      (cước ${money(inputs.MonthlyCharges)}/tháng, giá trị ${e.months} tháng; ưu đãi ${pct(e.cost, 0)}, giữ được ${pct(e.success, 0)}).</p>
      <p class="muted" style="margin:6px 0 0;font-size:13px">Dựa trên giả định chi phí, không phải số liệu thật.</p>
    </div>
    ${(entry.warnings ?? []).length ? `<div class="advice" style="background:var(--mid-bg)"><b>Lưu ý về dữ liệu nhập</b>${entry.warnings.map((w) => `<p style="margin:4px 0 0">${esc(w)}</p>`).join("")}<p class="muted" style="margin:6px 0 0;font-size:13px">Giá trị chưa từng xuất hiện khi huấn luyện, kết quả có thể kém tin cậy.</p></div>` : ""}
    <p class="muted" style="margin:12px 0 0;font-size:12px">${entry.modelVersion
      ? `Model ${esc(modelInfo?.name ?? "churn-model")} v${esc(entry.modelVersion)}`
      : (modelInfo ? `Model ${esc(modelInfo.name)} v${esc(modelInfo.version)} (${esc(modelInfo.alias)})` : "Model đang chạy")}
      ${entry.requestId ? `<br>Mã yêu cầu: <span class="mono">${esc(entry.requestId)}</span>` : ""}</p>`;
}

// FastAPI returns 422 with one {loc, msg} per invalid field; show them by field label.
function apiErrorMessage(status, body) {
  if (status === 503) return "Chưa có model được duyệt (champion). Hãy huấn luyện và duyệt một model trước.";
  if (status === 422 && Array.isArray(body.detail)) {
    return body.detail.map((d) => {
      const name = d.loc[d.loc.length - 1];
      return `${FIELDS.find((f) => f.name === name)?.label ?? name}: ${d.msg}`;
    }).join(" · ");
  }
  return typeof body.detail === "string" ? body.detail : `Lỗi ${status}`;
}

async function onSubmit(ev) {
  ev.preventDefault();
  const btn = $("#submit");
  const box = $("#result");
  let inputs;
  try {
    inputs = readForm();
  } catch (err) {
    box.innerHTML = `<div class="error">${esc(err.message)}</div>`;
    return;
  }
  btn.disabled = true;
  btn.textContent = "Đang dự đoán…";
  try {
    const res = await fetch(`${API}/v1/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inputs),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(apiErrorMessage(res.status, body));
    const entry = {
      time: Date.now(),
      probability: body.churn_probability,
      modelVersion: body.model_version ?? null,
      requestId: body.request_id ?? res.headers.get("X-Request-ID") ?? null,
      inputs,
      warnings: body.warnings ?? [],
    };
    saveHistory(entry);
    renderResult(entry);
  } catch (err) {
    box.innerHTML = `<div class="error"><b>Không dự đoán được.</b> ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Dự đoán";
  }
}

// ---------- history ----------
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) ?? []; } catch { return []; }
}
function saveHistory(entry) {
  const items = [entry, ...loadHistory()].slice(0, HISTORY_MAX);
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch { /* storage may be unavailable */ }
  updateHistoryCount();
}
function updateHistoryCount() {
  const n = loadHistory().length;
  $("#history-count").textContent = n ? `(${n})` : "";
}
function renderHistory() {
  const items = loadHistory();
  const contract = FIELDS.find((f) => f.name === "Contract");
  $("#history-empty").hidden = items.length > 0;
  $("#history-body").innerHTML = items.map((it, i) => {
    const lv = riskLevel(it.probability);
    return `<tr class="clickable" data-i="${i}" tabindex="0">
      <td>${new Date(it.time).toLocaleString("vi-VN")}</td>
      <td class="num">${pct(it.probability)}</td>
      <td><span class="badge ${lv.key}">${lv.label}</span></td>
      <td class="mono">${it.modelVersion ? `v${esc(it.modelVersion)}` : "—"}</td>
      <td>${esc(labelOf(contract, it.inputs.Contract))}</td>
      <td class="num">${esc(it.inputs.tenure)}</td>
      <td class="num">${money(it.inputs.MonthlyCharges)}</td></tr>`;
  }).join("");
}
function openHistoryRow(row) {
  const it = loadHistory()[Number(row.dataset.i)];
  if (!it) return;
  fillForm(Object.fromEntries(Object.entries(it.inputs).map(([k, v]) => [k, v ?? ""])));
  showTab("predict");
  renderResult(it);
}
$("#history-body").addEventListener("click", (e) => { const r = e.target.closest("tr[data-i]"); if (r) openHistoryRow(r); });
$("#history-body").addEventListener("keydown", (e) => { if (e.key === "Enter") { const r = e.target.closest("tr[data-i]"); if (r) openHistoryRow(r); } });
$("#clear").addEventListener("click", () => {
  try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
  updateHistoryCount();
  renderHistory();
});
$("#export").addEventListener("click", () => {
  const items = loadHistory();
  if (!items.length) return;
  const cols = ["time", "churn_probability", "model_version", "request_id", ...FIELDS.map((f) => f.name)];
  const csv = [cols.join(","), ...items.map((it) => [
    new Date(it.time).toISOString(), it.probability, it.modelVersion ?? "", it.requestId ?? "",
    ...FIELDS.map((f) => it.inputs[f.name] ?? ""),
  ].join(","))].join("\n");
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "churn-predictions.csv" });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ---------- model page ----------
function profitChart(curve) {
  const W = 560, H = 240, m = { l: 56, r: 16, t: 14, b: 34 };
  const xs = curve.map((r) => r.k * 100), ys = curve.map((r) => r.profit);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(0, ...ys), y1 = Math.max(...ys) * 1.08;
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const best = curve.reduce((a, r) => (r.profit > a.profit ? r : a), curve[0]);
  const ticks = [y0, y0 + (y1 - y0) / 2, y1].map((v) => `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${money(v)}</text>`).join("");
  const xt = curve.filter((_, i) => i % 2 === 1 || i === curve.length - 1).map((r) => `<text x="${X(r.k * 100)}" y="${H - 12}" text-anchor="middle">${Math.round(r.k * 100)}%</text>`).join("");
  const path = curve.map((r, i) => `${i ? "L" : "M"}${X(r.k * 100)},${Y(r.profit)}`).join(" ");
  const pts = curve.map((r) => `<circle class="${r === best ? "best" : "pt"}" cx="${X(r.k * 100)}" cy="${Y(r.profit)}" r="${r === best ? 6 : 3.5}"><title>${Math.round(r.k * 100)}%: ${money(r.profit)}</title></circle>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Lợi nhuận theo tỷ lệ khách được liên hệ">${ticks}${xt}<path class="line" d="${path}"/>${pts}</svg>`;
}

function candidateRows(all) {
  // The four dummy configurations score identically; show one.
  const cands = all.filter((c, i) => !c.name.startsWith("dummy") || all.findIndex((x) => x.name.startsWith("dummy")) === i)
    .map((c) => (c.name.startsWith("dummy") ? { ...c, name: "dummy (đoán theo tỷ lệ churn)" } : c));
  const max = Math.max(...cands.map((c) => c.cv_pr_auc));
  return cands.map((c) => `<tr><td class="mono">${esc(c.name)}</td><td class="num">${c.cv_pr_auc.toFixed(4)}</td>
    <td><div class="bar"><i style="width:${(c.cv_pr_auc / max) * 100}%"></i></div></td></tr>`).join("");
}

async function loadModel() {
  try {
    const res = await fetch(`${API}/v1/model`);
    modelInfo = res.ok ? await res.json() : null;
  } catch { modelInfo = null; }
  return modelInfo;
}

async function loadChallenger() {
  try {
    const res = await fetch(`${API}/v1/model/challenger`);
    challenger = res.ok ? await res.json() : null;
  } catch { challenger = null; }
  $("#pending-dot").hidden = challenger === null;
  return challenger;
}

async function loadVersions() {
  try {
    const res = await fetch(`${API}/v1/model/versions`);
    versions = res.ok ? await res.json() : [];
  } catch { versions = []; }
  return versions;
}

function historyCard(list, serving) {
  if (!list.length) return "";
  const canChange = !serving || serving.alias !== null; // an instance pinned to one version cannot switch
  const when = (t) => (t ? new Date(t).toLocaleString("vi-VN") : "—");
  const rows = list.map((v) => {
    const isChampion = v.aliases.includes("champion"), isChallenger = v.aliases.includes("challenger");
    const status = isChampion ? '<span class="badge low">đang phục vụ</span>' : isChallenger ? '<span class="badge mid">chờ duyệt</span>' : '<span class="muted">đã lưu</span>';
    const action = isChampion ? "" : isChallenger ? '<span class="muted">duyệt ở trên</span>'
      : (canChange ? `<button class="btn" data-rollback="${esc(v.version)}">Khôi phục</button>` : "");
    return `<tr><td><b>v${esc(v.version)}</b></td><td>${status}</td>
      <td class="num">${v.pr_auc != null ? v.pr_auc.toFixed(4) : "—"}</td><td class="num">${v.brier != null ? v.brier.toFixed(4) : "—"}</td>
      <td class="num">${v.best_profit != null ? money(v.best_profit) : "—"}</td>
      <td>${when(v.created_at)}</td><td>${v.restored_at ? `khôi phục ${when(v.restored_at)}` : (v.approved_at ? `duyệt ${when(v.approved_at)}` : "—")}</td>
      <td>${action}</td></tr>`;
  }).join("");
  return `<div class="card">
    <div class="spread"><h2 style="margin:0">Lịch sử phiên bản</h2><span class="muted">Đang lưu ${list.length} phiên bản</span></div>
    <p class="muted" style="margin:6px 0 14px">Mọi phiên bản đã đăng ký đều được giữ lại. Khôi phục một bản cũ sẽ phục vụ nó ngay (cần khóa quản trị); model được nạp và chạy thử trước khi đổi.</p>
    <div class="table-wrap"><table><thead><tr><th>Phiên bản</th><th>Trạng thái</th><th class="num">PR-AUC</th><th class="num">Brier</th><th class="num">Lợi nhuận</th><th>Tạo lúc</th><th>Duyệt / khôi phục</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table></div></div>`;
}

// metric, label, formatter, whether a higher value is better
const COMPARE = [
  ["pr_auc", "PR-AUC (chỉ số chính)", (v) => v.toFixed(4), true],
  ["roc_auc", "ROC-AUC", (v) => v.toFixed(4), true],
  ["brier", "Brier (thấp hơn là tốt hơn)", (v) => v.toFixed(4), false],
  ["recall", "Recall tại mức liên hệ tối ưu", (v) => pct(v, 1), true],
  ["precision", "Precision tại mức đó", (v) => pct(v, 1), true],
  ["best_profit", "Lợi nhuận thực tế tối đa", (v) => money(v), true],
];

function compareRows(cur, cand) {
  return COMPARE.map(([key, label, fmt, higher]) => {
    const a = cur?.metrics[key], b = cand.metrics[key];
    let delta = "—", cls = "";
    if (a != null && b != null && a !== 0) {
      const diff = b - a;
      cls = Math.abs(diff / a) < 0.001 ? "" : (diff > 0) === higher ? "good" : "bad";
      delta = `${diff > 0 ? "+" : ""}${key === "best_profit" ? money(diff) : diff.toFixed(4)}`;
    }
    return `<tr><td>${label}</td><td class="num">${a != null ? fmt(a) : "—"}</td><td class="num">${fmt(b)}</td><td class="num delta ${cls}">${delta}</td></tr>`;
  }).join("");
}

function pendingCard(cur, cand) {
  const names = { xgb: "XGBoost", logreg: "Logistic Regression", rf: "Random Forest", dummy: "Dummy" };
  const sameData = !cur || cur.provenance.data_md5 === cand.provenance.data_md5;
  return `<div class="card pending">
    <div class="spread"><h2 style="margin:0">Model chờ duyệt <span class="badge mid">${esc(cand.alias)} · v${esc(cand.version)}</span></h2>
    <span class="muted">${new Date(cand.created_at).toLocaleString("vi-VN")}</span></div>
    <p class="muted" style="margin:6px 0 14px">Model này đã qua quality gate nhưng <b>chưa phục vụ</b>. Chỉ khi bạn duyệt, API mới bắt đầu dùng nó.</p>
    ${sameData ? "" : '<div class="error" style="margin:0 0 12px">Dữ liệu huấn luyện của hai model khác nhau, các chỉ số không so sánh trực tiếp được.</div>'}
    <div class="table-wrap"><table>
      <thead><tr><th>Chỉ số (tập kiểm tra)</th><th class="num">${cur ? `Đang chạy v${esc(cur.version)}` : "Đang chạy"}</th><th class="num">Chờ duyệt v${esc(cand.version)}</th><th class="num">Chênh lệch</th></tr></thead>
      <tbody>${compareRows(cur, cand)}
        <tr><td>Thuật toán</td><td class="num">${cur ? esc(names[cur.selected.model] ?? cur.selected.model) : "—"}</td><td class="num">${esc(names[cand.selected.model] ?? cand.selected.model)}</td><td></td></tr>
        <tr><td>Git commit</td><td class="num mono">${cur ? esc(cur.provenance.git_commit.slice(0, 8)) : "—"}</td><td class="num mono">${esc(cand.provenance.git_commit.slice(0, 8))}</td><td></td></tr>
      </tbody></table></div>
    <div class="row" style="margin-top:16px"><button class="btn primary" id="approve-open" data-version="${esc(cand.version)}">Duyệt và đưa vào phục vụ</button>
    <span class="muted" style="font-size:13px">Cần khóa quản trị.</span></div>
  </div>`;
}

async function renderModel() {
  const box = $("#model-content");
  const info = modelInfo ?? await loadModel();
  await Promise.all([loadChallenger(), loadVersions()]);
  const pending = challenger ? pendingCard(info, challenger) : "";
  const history = historyCard(versions, info);
  if (!info) {
    box.innerHTML = `${pending}${history}<div class="card placeholder">Chưa có thông tin mô hình. Hãy huấn luyện (<span class="mono">dvc repro</span>), duyệt model (<span class="mono">python -m src.training.promote</span>) rồi khởi động lại API.</div>`;
    return;
  }
  const m = info.metrics, sel = info.selected, p = info.provenance;
  const stat = (v, l) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const modelNames = { xgb: "XGBoost", logreg: "Logistic Regression", rf: "Random Forest", dummy: "Dummy" };
  box.innerHTML = `${pending}
    <div class="card">
      <div class="spread"><h2 style="margin:0">${esc(info.name)} <span class="badge low">${esc(info.alias)} · v${esc(info.version)}</span></h2>
      <span class="muted">${new Date(info.created_at).toLocaleString("vi-VN")}</span></div>
      <p class="muted" style="margin:6px 0 16px">Chỉ số đo trên tập kiểm tra (20% dữ liệu, không dùng khi chọn mô hình).</p>
      <div class="stats">
        ${stat(m.pr_auc.toFixed(3), "PR-AUC (chỉ số chính)")}
        ${stat(m.roc_auc.toFixed(3), "ROC-AUC")}
        ${stat(m.brier.toFixed(3), `Brier (trước hiệu chuẩn ${m.brier_uncalibrated.toFixed(3)})`)}
        ${stat(pct(m.recall, 0), `Recall khi liên hệ ${pct(m.best_k, 0)} khách`)}
        ${stat(pct(m.precision, 0), "Precision tại mức đó")}
        ${stat(money(m.best_profit), "Lợi nhuận thực tế tối đa")}
      </div>
    </div>
    ${history}
    <div class="grid-2">
      <div class="card">
        <h3>Lợi nhuận theo tỷ lệ khách được liên hệ</h3>
        <p class="muted">Liên hệ những khách có điểm cao nhất trước. Điểm tròn rỗng là mức cho lợi nhuận cao nhất (${pct(m.best_k, 0)}).</p>
        ${profitChart(info.profit_curve)}
      </div>
      <div class="card">
        <h3>Độ tin cậy của xác suất</h3>
        <p class="muted">Gần đường chéo nghĩa là xác suất dự đoán khớp tần suất thực tế.</p>
        <img class="fig" alt="Biểu đồ hiệu chuẩn xác suất trước và sau hiệu chuẩn" src="${API}/v1/model/figures/calibration.png">
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3>So sánh các cấu hình (CV PR-AUC)</h3>
        <p class="muted">Chọn mô hình chỉ dựa trên cross-validation. Chênh lệch giữa các mô hình khác Dummy rất nhỏ.</p>
        <div class="table-wrap"><table><thead><tr><th>Cấu hình</th><th class="num">PR-AUC</th><th></th></tr></thead>
        <tbody>${candidateRows(info.candidates)}</tbody></table></div>
        <p class="muted" style="font-size:12px;margin-top:8px">tên = mô hình · fe1/fe0 có/không feature mới · nosens/sens bỏ/giữ giới tính và độ tuổi</p>
      </div>
      <div class="card">
        <h3>Mô hình được chọn</h3>
        <dl>
          <dt>Thuật toán</dt><dd>${esc(modelNames[sel.model] ?? sel.model)}</dd>
          <dt>Feature mới</dt><dd>${sel.add_features ? "Có" : "Không"}</dd>
          <dt>Giới tính, độ tuổi</dt><dd>${sel.drop_sensitive ? "Không dùng làm đầu vào" : "Dùng làm đầu vào"}</dd>
          <dt>CV PR-AUC</dt><dd>${m.selected_cv_pr_auc?.toFixed(4) ?? "—"}</dd>
          ${m.baseline_pr_auc != null ? `<dt>Baseline (Logistic Regression, test)</dt><dd>${m.baseline_pr_auc.toFixed(4)}</dd>` : ""}
          <dt>Chênh lệch chọn khách theo giới tính / tuổi (top 10%)</dt><dd>${pct(m.dpd_gender, 1)} / ${pct(m.dpd_senior, 1)}</dd>
        </dl>
        <h3 style="margin-top:18px">Nguồn gốc (để tái lập)</h3>
        <dl>
          <dt>Git commit</dt><dd class="mono">${esc(p.git_commit.slice(0, 12))}</dd>
          <dt>Hash dữ liệu</dt><dd class="mono">${esc(p.data_md5)}</dd>
          <dt>MLflow run</dt><dd class="mono">${esc(info.run_id)}</dd>
        </dl>
        <h3 style="margin-top:18px">Giả định kinh tế</h3>
        <dl>
          <dt>Giá trị khách</dt><dd>cước tháng × ${esc(info.economics.value_months)}</dd>
          <dt>Chi phí ưu đãi</dt><dd>${pct(info.economics.retention_cost_rate, 0)} doanh thu năm</dd>
          <dt>Tỷ lệ giữ chân thành công</dt><dd>${pct(info.economics.retention_success, 0)}</dd>
        </dl>
      </div>
    </div>`;
}

// ---------- approval ----------
const dialog = $("#approve-dialog");
function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  setTimeout(() => { t.hidden = true; }, 4000);
}
function approveError(res, body) {
  if (res.status === 401) return "Khóa quản trị không đúng.";
  if (res.status === 503) return "Server chưa cấu hình ADMIN_KEY nên chức năng duyệt đang tắt.";
  if (res.status === 404) return "Không tìm thấy phiên bản hoặc model chờ duyệt. Hãy tải lại trang.";
  if (res.status === 422) return typeof body.detail === "string" ? body.detail : "Phiên bản này không tương thích với API hiện tại.";
  return typeof body.detail === "string" ? body.detail : `Lỗi ${res.status}`;
}
function openDialog(action, version) {
  const from = modelInfo ? `v${modelInfo.version}` : "chưa có model";
  const restoring = action === "rollback";
  $("#approve-title").textContent = restoring ? "Khôi phục model cũ" : "Duyệt model đưa vào phục vụ";
  $("#approve-confirm").textContent = restoring ? "Xác nhận khôi phục" : "Xác nhận duyệt";
  $("#approve-summary").innerHTML = restoring
    ? `Bạn sắp quay về <b>v${esc(version)}</b> thay cho model đang phục vụ (<b>${esc(from)}</b>). Từ lúc xác nhận, mọi dự đoán sẽ dùng bản cũ này. Model đang chờ duyệt (nếu có) không bị ảnh hưởng.`
    : `Bạn sắp thay model đang phục vụ (<b>${esc(from)}</b>) bằng <b>v${esc(version)}</b>. Từ lúc xác nhận, mọi dự đoán sẽ dùng model mới.`;
  $("#approve-error").hidden = true;
  $("#approve-key").value = "";
  Object.assign(dialog.dataset, { action, version });
  dialog.showModal();
  $("#approve-key").focus();
}
$("#model-content").addEventListener("click", (e) => {
  const approve = e.target.closest("#approve-open");
  const rollback = e.target.closest("[data-rollback]");
  if (approve) openDialog("promote", approve.dataset.version);
  else if (rollback) openDialog("rollback", rollback.dataset.rollback);
});
$("#approve-cancel").addEventListener("click", () => dialog.close());
$("#approve-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const confirm = $("#approve-confirm"), errBox = $("#approve-error");
  confirm.disabled = true;
  confirm.textContent = "Đang xử lý…";
  errBox.hidden = true;
  try {
    const res = await fetch(`${API}/v1/model/${dialog.dataset.action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Key": $("#approve-key").value },
      body: JSON.stringify({ version: dialog.dataset.version }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(approveError(res, body));
    modelInfo = body;
    dialog.close();
    await renderModel();
    refreshStatus();
    toast(`${dialog.dataset.action === "rollback" ? "Đã khôi phục" : "Đã duyệt"}: model v${body.version} đang phục vụ`);
  } catch (err) {
    errBox.textContent = err.message;
    errBox.hidden = false;
  } finally {
    $("#approve-key").value = "";
    confirm.disabled = false;
    confirm.textContent = dialog.dataset.action === "rollback" ? "Xác nhận khôi phục" : "Xác nhận duyệt";
  }
});

// ---------- alerts ----------
const SEVERITY = { critical: ["high", "Nghiêm trọng"], warning: ["mid", "Cảnh báo"], info: ["info", "Thông tin"] };

async function loadAlerts() {
  try {
    const res = await fetch(`${HUB}/alerts?limit=30`);
    alertState = res.ok ? await res.json() : null;
  } catch { alertState = null; }
  const badge = $("#alerts-badge"), firing = alertState?.counts.firing ?? 0;
  badge.hidden = firing === 0;
  badge.textContent = firing;
  badge.classList.toggle("critical", (alertState?.counts.critical ?? 0) > 0);
  return alertState;
}

function alertRow(a, resolved = false) {
  const [cls, label] = SEVERITY[a.severity] ?? SEVERITY.warning;
  const where = Object.entries(a.labels).filter(([k]) => ["pod", "track", "field"].includes(k)).map(([k, v]) => `${k}=${v}`).join(" · ");
  const when = resolved ? `đã xử lý ${new Date(a.ended_at || a.received_at).toLocaleString("vi-VN")}` : `từ ${new Date(a.started_at).toLocaleString("vi-VN")}`;
  return `<div class="alert-row"><span class="badge ${cls}">${label}</span>
    <div><b>${esc(a.name)}</b> ${a.status === "resolved" ? '<span class="badge low">đã xử lý</span>' : ""}<div>${esc(a.summary)}</div>
    ${a.description ? `<div class="meta">${esc(a.description)}</div>` : ""}${where ? `<div class="meta mono">${esc(where)}</div>` : ""}</div>
    <div class="meta">${esc(when)}</div></div>`;
}

async function renderAlerts() {
  const box = $("#alerts-content");
  const state = await loadAlerts();
  const host = location.origin;
  const tools = `<div class="links"><a class="btn" href="${host}/grafana/dashboards" target="_blank" rel="noopener">Dashboard Grafana</a>
    <a class="btn" href="${host}/alertmanager/" target="_blank" rel="noopener">Alertmanager</a>
    <a class="btn" href="${host}/prometheus/alerts" target="_blank" rel="noopener">Luật trong Prometheus</a></div>`;
  if (!state) {
    box.innerHTML = `<div class="card"><h2>Cảnh báo</h2><div class="error">Không kết nối được hub cảnh báo (<span class="mono">${HUB}/alerts</span>). Giám sát có thể chưa được triển khai.</div></div>`;
    return;
  }
  const active = state.active.length
    ? state.active.map((a) => alertRow(a)).join("")
    : '<div class="ok-banner">Không có cảnh báo nào đang bắn.</div>';
  const past = state.history.filter((e) => e.status === "resolved" || !state.active.some((a) => a.id === e.id));
  box.innerHTML = `<div class="card"><div class="spread"><h2 style="margin:0">Đang bắn (${state.counts.firing})</h2>${tools}</div>
      <p class="muted" style="margin:6px 0 4px">Alertmanager gửi cảnh báo tới hub cảnh báo của hệ thống bằng webhook, trang này đọc lại từ đó mỗi 20 giây.</p>${active}</div>
    <div class="card"><h2>Sự kiện gần đây</h2>${past.length ? past.map((e) => alertRow(e, e.status === "resolved")).join("") : '<p class="muted">Chưa có sự kiện nào. Hub chỉ nhớ các sự kiện từ lần khởi động gần nhất.</p>'}</div>`;
}

// ---------- status + links ----------
async function refreshStatus() {
  const dot = $("#status-dot"), text = $("#status-text");
  try {
    const res = await fetch(`${API}/health`);
    const h = await res.json();
    dot.className = `dot ${h.model_loaded ? "ok" : "bad"}`;
    if (h.model_loaded && !modelInfo) await loadModel();
    await loadChallenger();
    loadAlerts();
    text.textContent = h.model_loaded ? `API sẵn sàng${modelInfo ? ` · model v${modelInfo.version}` : ""}` : "API chạy nhưng chưa có model";
  } catch {
    dot.className = "dot bad";
    text.textContent = "Không kết nối được API";
  }
}

function buildLinks() {
  const host = location.hostname;
  const links = [
    ["Swagger / OpenAPI", `${location.origin}${API}/docs`],
    ["MLflow", `http://${host}:5001`],
    ["Grafana", `${location.origin}/grafana/dashboards`],
    ["Prometheus", `${location.origin}/prometheus/`],
    ["Alertmanager", `${location.origin}/alertmanager/`],
  ];
  $("#doc-links").innerHTML = links.map(([l, u]) => `<a class="btn" href="${u}" target="_blank" rel="noopener">${l}</a>`).join("");
}

// ---------- init ----------
buildForm();
buildLinks();
updateHistoryCount();
$("#form").addEventListener("submit", onSubmit);
$("#form").addEventListener("reset", () => { setTimeout(() => { fillForm(SAMPLES.high); $("#result").innerHTML = '<div class="placeholder">Chưa có dự đoán. Điền thông tin và bấm Dự đoán.</div>'; }, 0); });
$$("[data-sample]").forEach((b) => b.addEventListener("click", () => fillForm(SAMPLES[b.dataset.sample])));
refreshStatus();
setInterval(refreshStatus, 20000);
const initial = location.hash.slice(1);
if (["predict", "history", "model", "alerts", "docs"].includes(initial)) showTab(initial);
