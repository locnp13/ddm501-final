"use strict";

const API = "/api";
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

// ---------- tabs ----------
function showTab(name) {
  $$("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  if (name === "model") renderModel();
  if (name === "history") renderHistory();
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
    <p class="muted" style="margin:12px 0 0;font-size:12px">${modelInfo
      ? `Model ${esc(modelInfo.name)} v${esc(modelInfo.version)} (${esc(modelInfo.alias)})` : "Model đang chạy"}</p>`;
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
    const entry = { time: Date.now(), probability: body.churn_probability, inputs, warnings: body.warnings ?? [] };
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
  const cols = ["time", "churn_probability", ...FIELDS.map((f) => f.name)];
  const csv = [cols.join(","), ...items.map((it) => [new Date(it.time).toISOString(), it.probability, ...FIELDS.map((f) => it.inputs[f.name] ?? "")].join(","))].join("\n");
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

async function renderModel() {
  const box = $("#model-content");
  const info = modelInfo ?? await loadModel();
  if (!info) {
    box.innerHTML = `<div class="card placeholder">Chưa có thông tin mô hình. Hãy huấn luyện (<span class="mono">dvc repro</span>), duyệt model (<span class="mono">python -m src.training.promote</span>) rồi khởi động lại API.</div>`;
    return;
  }
  const m = info.metrics, sel = info.selected, p = info.provenance;
  const stat = (v, l) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const modelNames = { xgb: "XGBoost", logreg: "Logistic Regression", rf: "Random Forest", dummy: "Dummy" };
  box.innerHTML = `
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

// ---------- status + links ----------
async function refreshStatus() {
  const dot = $("#status-dot"), text = $("#status-text");
  try {
    const res = await fetch(`${API}/health`);
    const h = await res.json();
    dot.className = `dot ${h.model_loaded ? "ok" : "bad"}`;
    if (h.model_loaded && !modelInfo) await loadModel();
    text.textContent = h.model_loaded ? `API sẵn sàng${modelInfo ? ` · model v${modelInfo.version}` : ""}` : "API chạy nhưng chưa có model";
  } catch {
    dot.className = "dot bad";
    text.textContent = "Không kết nối được API";
  }
}

function buildLinks() {
  const host = location.hostname;
  const links = [
    ["Swagger / OpenAPI", `http://${host}:8000/docs`],
    ["MLflow", `http://${host}:5001`],
    ["Grafana", `http://${host}:3000`],
    ["Prometheus", `http://${host}:9090`],
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
if (["predict", "history", "model", "docs"].includes(initial)) showTab(initial);
