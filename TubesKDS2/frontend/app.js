// ═══════════════════════════════════════════════════════════
// Cell Cycle Intelligence System — Dashboard + Inference
// ═══════════════════════════════════════════════════════════

const PHASE_COLORS = {
    "G1": "#00E676", "S": "#448AFF", "G2": "#E040FB",
    "Prophase": "#FF9100", "Metaphase": "#FF1744",
    "Anaphase": "#F50057", "Telophase": "#00E5FF"
};
const SEVERITY_COLORS = {
    "low": "#FFD600", "medium": "#FF9100", "high": "#FF1744", "critical": "#D500F9"
};

// Biological interpretation per phase (static knowledge)
const PHASE_BIO_INFO = {
    "G1": {
        description: "Fase pertumbuhan sel. Sel mensintesis protein dan organel, meningkatkan ukuran. Checkpoint G1/S (Rb/E2F) memastikan kondisi siap untuk replikasi DNA sebelum berkomitmen ke S fase.",
        visual_cue: "🔬 Grad-CAM: perhatian pada nukleus spherical kecil-menengah dengan kromatin yang terdistribusi merata (2N DNA)."
    },
    "S": {
        description: "Fase sintesis DNA. Replikasi DNA aktif berlangsung — konten DNA meningkat dari 2N ke 4N. Intensitas Hoechst 33342 meningkat progresif selama S fase.",
        visual_cue: "🔬 Grad-CAM: intensitas nukleus lebih tinggi dan heterogen dibanding G1; batas nukleus masih utuh."
    },
    "G2": {
        description: "Fase persiapan mitosis. Sel memverifikasi replikasi DNA selesai (checkpoint G2/M via ATM/ATR→Chk2→Wee1). CDK1/Cyclin B mulai terakumulasi.",
        visual_cue: "🔬 Grad-CAM: nukleus lebih besar (4N DNA), intensitas Hoechst tinggi; sel mulai bulat."
    },
    "Prophase": {
        description: "Mitosis dimulai. Kromosom mulai terkondensasi menjadi struktur kompak yang terlihat. CDK1/Cyclin B aktif menginduksi kondensasi. Membran nukleus mulai terurai.",
        visual_cue: "🔬 Grad-CAM: struktur kromosom kondensasi mulai terlihat; nukleus terlihat 'berbintik' dan memadat."
    },
    "Metaphase": {
        description: "Kromosom teralignment di bidang ekuatorial (metaphase plate). Spindle Assembly Checkpoint (SAC/Mad2) memastikan semua kinetochore terikat spindle sebelum anafase.",
        visual_cue: "🔬 Grad-CAM: pola 'pelat' linear kromosom di tengah sel — aktivasi tinggi pada metaphase plate."
    },
    "Anaphase": {
        description: "APC/C mengaktivasi separase → kohesi terpotong → kromatid sister terpisah menuju kutub berlawanan. CDK1 mulai diinaktivasi melalui degradasi Cyclin B.",
        visual_cue: "🔬 Grad-CAM: dua cluster kromosom bergerak ke arah berlawanan — pola 'V' atau dua titik terpisah."
    },
    "Telophase": {
        description: "Kromosom mencapai kutub. Membran nukleus terbentuk kembali di sekitar setiap set kromosom. Cyclin B terdegradasi hampir sempurna; APC/C aktif.",
        visual_cue: "🔬 Grad-CAM: dua nukleus terpisah terbentuk; kromatin mulai dekondensasi; cleavage furrow terlihat."
    }
};

function show(id) { const el = document.getElementById(id); if (el) el.style.display = "block"; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = "none"; }

function showCardError(errorId, message) {
    const el = document.getElementById(errorId);
    if (!el) return;
    el.textContent = message;
    el.style.display = "flex";
}
function hideCardError(errorId) {
    const el = document.getElementById(errorId);
    if (el) el.style.display = "none";
}

// ═══ STATIC DASHBOARD FALLBACK ═══════════════════════════

const STATIC_MANIFEST_URL = "/assets/plots/manifest.json";
let _staticManifest = null;

async function getStaticManifest() {
    if (_staticManifest) return _staticManifest;
    const res = await fetch(STATIC_MANIFEST_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`Static manifest not available (HTTP ${res.status})`);
    _staticManifest = await res.json();
    return _staticManifest;
}

function setImgSrc(id, src) {
    const el = document.getElementById(id);
    if (el) el.src = src;
}

// ═══ DASHBOARD: Run Full Simulation (SEQUENTIAL) ══════════

document.getElementById("btn-run-all").addEventListener("click", runFullSimulation);

async function runFullSimulation() {
    const btn = document.getElementById("btn-run-all");
    btn.disabled = true;

    const steps = [
        { fn: runCNN,        label: "1/6: Evaluasi CNN…" },
        { fn: runGradCAM,    label: "2/6: Grad-CAM…" },
        { fn: runODE,        label: "3/6: Solving ODE…" },
        { fn: runHMM,        label: "4/6: Koreksi HMM…" },
        { fn: runCheckpoint, label: "5/6: Deteksi Anomali…" },
        { fn: runPopulation, label: "6/6: Analisis Populasi…" },
    ];

    for (const step of steps) {
        btn.textContent = `⏳ ${step.label}`;
        await step.fn();
    }

    btn.disabled = false;
    btn.innerHTML = "&#9654; Run Full Simulation";
}

// ── Layer 1: CNN Results ─────────────────────────────────

async function runCNN() {
    show("loading-cnn"); hide("results-cnn"); hideCardError("error-cnn");
    try {
        const res = await fetch("/api/dashboard/cnn");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        document.getElementById("cnn-metrics").innerHTML = `
            <div class="metric-card"><div class="value" style="color:#00E676">${(data.test_accuracy*100).toFixed(1)}%</div><div class="label">Test Accuracy</div><div class="sublabel">real test set</div></div>
            <div class="metric-card"><div class="value">${data.total_samples}</div><div class="label">Test Samples</div><div class="sublabel">BBBC048</div></div>
            <div class="metric-card"><div class="value">${data.train_epochs}</div><div class="label">Epochs</div></div>
            <div class="metric-card"><div class="value" style="color:#448AFF">${data.model_params}</div><div class="label">Parameters</div></div>
        `;

        // Key insight from confusion pairs
        if (data.most_confused_pairs && data.most_confused_pairs.length > 0) {
            const pairs = data.most_confused_pairs
                .map(p => `<strong>${p.true_class} → ${p.predicted_class}</strong> (${p.count}×)`)
                .join(" &nbsp;|&nbsp; ");
            const hard = data.hard_classes ? data.hard_classes.join(", ") : "";
            const insightEl = document.getElementById("cnn-insight");
            insightEl.innerHTML = `
                💡 <strong>Top confusions:</strong> ${pairs}<br>
                ${hard ? `⚠ <strong>Kelas tersulit (F1 terendah):</strong> ${hard} — konsisten dengan overlap visual antar fase interphase.` : ""}
            `;
            insightEl.style.display = "block";
        }

        document.getElementById("cnn-confusion").src = "data:image/png;base64," + data.confusion_matrix;
        document.getElementById("cnn-curves").src = "data:image/png;base64," + data.training_curves;
        show("results-cnn");
    } catch(e) {
        // Fallback: static plots + metrics (inference-only deployment)
        try {
            const m = await getStaticManifest();
            const cnn = m.cnn;
            document.getElementById("cnn-metrics").innerHTML = `
                <div class="metric-card"><div class="value" style="color:#00E676">${(cnn.test_accuracy*100).toFixed(1)}%</div><div class="label">Test Accuracy</div><div class="sublabel">snapshot</div></div>
                <div class="metric-card"><div class="value">${cnn.total_samples}</div><div class="label">Test Samples</div><div class="sublabel">snapshot</div></div>
                <div class="metric-card"><div class="value">${cnn.train_epochs}</div><div class="label">Epochs</div></div>
                <div class="metric-card"><div class="value" style="color:#448AFF">${cnn.model_params}</div><div class="label">Parameters</div></div>
            `;

            const insightEl = document.getElementById("cnn-insight");
            insightEl.innerHTML = `📌 Snapshot plot statis (tanpa evaluasi live). Generated: <code>${m.generated_at || "-"}</code>`;
            insightEl.style.display = "block";

            setImgSrc("cnn-confusion", `/assets/plots/${cnn.images.confusion}`);
            setImgSrc("cnn-curves", `/assets/plots/${cnn.images.curves}`);
            show("results-cnn");
        } catch (fallbackErr) {
            showCardError("error-cnn", `Gagal memuat CNN: ${e.message}`);
        }
    } finally { hide("loading-cnn"); }
}

// ── Grad-CAM batch ───────────────────────────────────────

async function runGradCAM() {
    show("loading-gradcam"); hide("results-gradcam"); hideCardError("error-gradcam");
    try {
        const res = await fetch("/api/dashboard/gradcam");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        document.getElementById("gradcam-batch").src = "data:image/png;base64," + data.gradcam_plot;
        show("results-gradcam");
    } catch(e) {
        try {
            const m = await getStaticManifest();
            setImgSrc("gradcam-batch", `/assets/plots/${m.gradcam.images.batch}`);
            show("results-gradcam");
        } catch {
            showCardError("error-gradcam", `Gagal memuat Grad-CAM: ${e.message}`);
        }
    } finally { hide("loading-gradcam"); }
}

// ── ODE ──────────────────────────────────────────────────

async function runODE() {
    show("loading-ode"); hide("results-ode"); hideCardError("error-ode");
    try {
        const res = await fetch("/api/dashboard/ode");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        document.getElementById("ode-plot").src = "data:image/png;base64," + data.ode_plot;

        let html = "<table><tr><th></th>";
        data.phases.forEach(p => html += `<th>${p}</th>`);
        html += "</tr>";
        data.transition_matrix.forEach((row, i) => {
            html += `<tr><th>${data.phases[i]}</th>`;
            row.forEach(val => {
                const bg = `rgba(68,138,255,${Math.min(val*3,1)*0.4})`;
                html += `<td style="background:${bg}">${val.toFixed(3)}</td>`;
            });
            html += "</tr>";
        });
        html += "</table>";
        document.getElementById("transition-matrix").innerHTML = html;
        show("results-ode");
    } catch(e) {
        showCardError("error-ode", `Gagal memuat ODE: ${e.message}`);
    } finally { hide("loading-ode"); }
}

// ── HMM ──────────────────────────────────────────────────

async function runHMM() {
    show("loading-hmm"); hide("results-hmm"); hideCardError("error-hmm");
    try {
        const res = await fetch("/api/dashboard/hmm");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        document.getElementById("hmm-metrics").innerHTML = `
            <div class="metric-card">
                <div class="value" style="color:#FF9100">${(data.cnn_accuracy*100).toFixed(1)}%</div>
                <div class="label">CNN-Only</div>
                <div class="sublabel">error_rate=0.18</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color:#00E676">${(data.hmm_accuracy*100).toFixed(1)}%</div>
                <div class="label">HMM-Corrected</div>
                <div class="sublabel">Viterbi decoding</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color:#448AFF">+${(data.improvement*100).toFixed(1)}%</div>
                <div class="label">Improvement</div>
                <div class="sublabel">simulasi</div>
            </div>
            <div class="metric-card">
                <div class="value">${data.n_frames}</div>
                <div class="label">Frames</div>
                <div class="sublabel">${data.total_hours}h sim</div>
            </div>
        `;

        const insightEl = document.getElementById("hmm-insight");
        insightEl.innerHTML = `
            💡 Matriks transisi HMM <strong>diturunkan dari ODE Tyson-Novak</strong>, bukan dari data training.
            Ini memungkinkan koreksi sekuens tanpa data time-lapse berlabel.
            Transisi tidak valid (G1→Anaphase) memiliki probabilitas ≈10⁻⁴ dalam matriks ODE → Viterbi menolaknya secara otomatis.
        `;
        insightEl.style.display = "block";

        document.getElementById("hmm-plot").src = "data:image/png;base64," + data.hmm_plot;
        show("results-hmm");
    } catch(e) {
        showCardError("error-hmm", `Gagal memuat HMM: ${e.message}`);
    } finally { hide("loading-hmm"); }
}

// ── Checkpoint ───────────────────────────────────────────

async function runCheckpoint() {
    show("loading-checkpoint"); hide("results-checkpoint"); hideCardError("error-checkpoint");
    try {
        const res = await fetch("/api/dashboard/checkpoint");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        const reduction = data.n_cnn_anomalies > 0
            ? Math.round((1 - data.n_hmm_anomalies / data.n_cnn_anomalies) * 100)
            : 0;

        document.getElementById("checkpoint-metrics").innerHTML = `
            <div class="metric-card">
                <div class="value" style="color:#FF1744">${data.n_cnn_anomalies}</div>
                <div class="label">CNN-Only Anomalies</div>
                <div class="sublabel">termasuk artefak</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color:#00E676">${data.n_hmm_anomalies}</div>
                <div class="label">HMM-Corrected</div>
                <div class="sublabel">genuine anomalies</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color:#448AFF">-${reduction}%</div>
                <div class="label">False Anomalies</div>
                <div class="sublabel">dieliminasi HMM</div>
            </div>
        `;

        const insightEl = document.getElementById("checkpoint-insight");
        insightEl.innerHTML = `
            💡 <strong>Mayoritas anomali CNN-only adalah artefak klasifikasi</strong> — transisi tidak valid (G1→Anaphase) bukan anomali biologis.
            HMM mengeliminasi artefak ini; sisa <strong>${data.n_hmm_anomalies} anomali genuine</strong> merepresentasikan potensi disregulasi checkpoint biologis.
        `;
        insightEl.style.display = "block";

        document.getElementById("checkpoint-plot").src = "data:image/png;base64," + data.checkpoint_plot;

        const list = document.getElementById("anomaly-list");
        if (data.anomalies_hmm.length === 0) {
            list.innerHTML = '<p style="color:#666;font-size:0.78rem;">Tidak ada anomali checkpoint terdeteksi pada sekuens HMM-corrected.</p>';
        } else {
            list.innerHTML = data.anomalies_hmm.map(a => `
                <div class="anomaly-item severity-${a.severity}">
                    <div class="anomaly-header">
                        <span class="anomaly-badge" style="background:${SEVERITY_COLORS[a.severity]}30;color:${SEVERITY_COLORS[a.severity]}">${a.severity}</span>
                        ${a.checkpoint} (${a.phase}) &mdash; frame ${a.frame_start}–${a.frame_end}
                    </div>
                    <div class="anomaly-body">${a.bio_interpretation}</div>
                </div>
            `).join("");
        }
        show("results-checkpoint");
    } catch(e) {
        showCardError("error-checkpoint", `Gagal memuat checkpoint: ${e.message}`);
    } finally { hide("loading-checkpoint"); }
}

// ── Population ───────────────────────────────────────────

async function runPopulation() {
    show("loading-population"); hide("results-population"); hideCardError("error-population");
    try {
        const res = await fetch("/api/dashboard/population");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

        document.getElementById("pop-metrics").innerHTML = `
            <div class="metric-card"><div class="value">${data.total_cells}</div><div class="label">Cells</div><div class="sublabel">test set</div></div>
            <div class="metric-card"><div class="value" style="color:#FF1744">${(data.mitotic_index*100).toFixed(1)}%</div><div class="label">Mitotic Index</div><div class="sublabel">lit: 5–10%</div></div>
            <div class="metric-card"><div class="value" style="color:#448AFF">${(data.growth_fraction*100).toFixed(1)}%</div><div class="label">Growth Fraction</div></div>
            <div class="metric-card"><div class="value" style="color:#E040FB">${data.doubling_time_hours.toFixed(1)}h</div><div class="label">Doubling Time</div><div class="sublabel">lit: ~24h</div></div>
        `;
        document.getElementById("population-plot").src = "data:image/png;base64," + data.population_plot;
        document.getElementById("pop-clinical").innerHTML = `
            <h4>${data.status}</h4>
            <p>${data.clinical_significance}</p>
        `;
        show("results-population");
    } catch(e) {
        try {
            const m = await getStaticManifest();
            const p = m.population;
            document.getElementById("pop-metrics").innerHTML = `
                <div class="metric-card"><div class="value">${p.total_cells}</div><div class="label">Cells</div><div class="sublabel">snapshot</div></div>
                <div class="metric-card"><div class="value" style="color:#FF1744">${(p.mitotic_index*100).toFixed(1)}%</div><div class="label">Mitotic Index</div><div class="sublabel">lit: 5–10%</div></div>
                <div class="metric-card"><div class="value" style="color:#448AFF">${(p.growth_fraction*100).toFixed(1)}%</div><div class="label">Growth Fraction</div></div>
                <div class="metric-card"><div class="value" style="color:#E040FB">${p.doubling_time_hours.toFixed(1)}h</div><div class="label">Doubling Time</div><div class="sublabel">lit: ~24h</div></div>
            `;
            setImgSrc("population-plot", `/assets/plots/${p.images.plot}`);
            document.getElementById("pop-clinical").innerHTML = `<h4>${p.status}</h4><p>${p.clinical_significance}</p>`;
            show("results-population");
        } catch {
            showCardError("error-population", `Gagal memuat populasi: ${e.message}`);
        }
    } finally { hide("loading-population"); }
}

// ═══ SIDEBAR: Live Inference ═════════════════════════════

const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const btnClassify = document.getElementById("btn-classify");
const btnSample = document.getElementById("btn-sample");
let selectedFile = null;

uploadArea.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("dragover"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", e => {
    e.preventDefault(); uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length > 0) handleFile(fileInput.files[0]); });

function handleFile(file) {
    selectedFile = file;
    document.getElementById("preview-img").src = URL.createObjectURL(file);
    document.getElementById("filename").textContent = file.name;
    document.getElementById("preview").style.display = "block";
    btnClassify.style.display = "block";
    hide("results-classify"); hide("error-msg"); hide("bio-interp");
}

// Try Sample Image
btnSample.addEventListener("click", async () => {
    btnSample.disabled = true;
    btnSample.textContent = "Loading sample…";
    try {
        const res = await fetch("/api/sample_image");
        if (!res.ok) throw new Error("Sample not available");
        const blob = await res.blob();
        const contentDisp = res.headers.get("Content-Disposition") || "";
        const nameMatch = contentDisp.match(/filename="?([^"]+)"?/);
        const fname = nameMatch ? nameMatch[1] : "sample_cell.png";
        const file = new File([blob], fname, { type: blob.type });
        handleFile(file);
    } catch(e) {
        document.getElementById("error-msg").textContent = `Sample image: ${e.message}`;
        show("error-msg");
    } finally {
        btnSample.disabled = false;
        btnSample.textContent = "🔬 Try Sample Image";
    }
});

btnClassify.addEventListener("click", async () => {
    if (!selectedFile) return;
    btnClassify.disabled = true;
    show("loading-classify"); hide("results-classify"); hide("error-msg"); hide("bio-interp");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const res = await fetch("/api/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed");

        const phase = data.predicted_phase;
        document.getElementById("predicted-phase").textContent = phase;
        document.getElementById("predicted-phase").style.color = PHASE_COLORS[phase] || "#fff";
        document.getElementById("confidence-text").textContent = `${(data.confidence*100).toFixed(1)}% confidence`;

        const info = data.phase_info;
        document.getElementById("phase-info").innerHTML = `
            <span class="chip">${info.category}</span>
            <span class="chip">${info.duration_hours}h avg</span>
            <span class="chip">${(info.fraction_of_cycle*100).toFixed(1)}% of cycle</span>
        `;

        let bars = "";
        for (const [p, prob] of Object.entries(data.probabilities)) {
            const pct = (prob*100).toFixed(1);
            bars += `<div class="prob-row"><div class="prob-label">${p}</div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${pct}%;background:${PHASE_COLORS[p]}"></div></div><div class="prob-value">${pct}%</div></div>`;
        }
        document.getElementById("prob-bars-container").innerHTML = bars;
        document.getElementById("gradcam-img").src = "data:image/png;base64," + data.gradcam_image;

        // Biological interpretation
        const bioInfo = PHASE_BIO_INFO[phase];
        if (bioInfo) {
            document.getElementById("bio-interp-text").textContent = bioInfo.description;
            document.getElementById("bio-visual-cue").textContent = bioInfo.visual_cue;
            show("bio-interp");
        }

        show("results-classify");
    } catch(err) {
        document.getElementById("error-msg").textContent = err.message;
        show("error-msg");
    } finally { hide("loading-classify"); btnClassify.disabled = false; }
});

// ═══ STATUS ══════════════════════════════════════════════

async function checkStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        const dot = document.getElementById("status-dot");
        const text = document.getElementById("status-text");
        const banner = document.getElementById("setup-banner");

        const inferenceReady = data.model_loaded;
        const dashboardReady = data.model_loaded && data.ode_ready && data.hmm_ready && data.data_loaded;

        if (dashboardReady) {
            dot.className = "status-dot online";
            text.textContent = `Ready (${data.device})`;
            if (banner) banner.style.display = "none";
        } else if (inferenceReady) {
            dot.className = "status-dot online";
            const missing = [];
            if (!data.data_loaded) missing.push("dataset");
            if (!data.ode_ready || !data.hmm_ready) missing.push("ODE/HMM");
            text.textContent = missing.length ? `Inference ready (missing: ${missing.join(", ")})` : `Inference ready (${data.device})`;
            if (banner) banner.style.display = "none";
        } else {
            dot.className = "status-dot offline";
            const missing = [];
            if (!data.model_loaded) missing.push("model");
            if (!data.data_loaded) missing.push("data");
            if (!data.ode_ready || !data.hmm_ready) missing.push("ODE/HMM");
            text.textContent = `Not ready: ${missing.join(", ")}`;
            if (banner) banner.style.display = "block";
        }
    } catch {
        document.getElementById("status-dot").className = "status-dot offline";
        document.getElementById("status-text").textContent = "Server offline";
        const banner = document.getElementById("setup-banner");
        if (banner) banner.style.display = "block";
    }
}
checkStatus();
setInterval(checkStatus, 10000);
