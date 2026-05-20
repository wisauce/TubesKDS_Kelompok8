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

function show(id) { document.getElementById(id).style.display = "block"; }
function hide(id) { document.getElementById(id).style.display = "none"; }

// ═══ DASHBOARD: Run Full Simulation ══════════════════════

document.getElementById("btn-run-all").addEventListener("click", runFullSimulation);

async function runFullSimulation() {
    const btn = document.getElementById("btn-run-all");
    btn.disabled = true; btn.textContent = "Running...";

    // Run all in parallel
    await Promise.all([
        runCNN(),
        runGradCAM(),
        runODE(),
        runHMM(),
        runCheckpoint(),
        runPopulation()
    ]);

    btn.disabled = false; btn.textContent = "\u25B6 Run Full Simulation";
}

// ── Layer 1: CNN Results ─────────────────────────────────

async function runCNN() {
    show("loading-cnn"); hide("results-cnn");
    try {
        const res = await fetch("/api/dashboard/cnn");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);

        document.getElementById("cnn-metrics").innerHTML = `
            <div class="metric-card"><div class="value" style="color:#00E676">${(data.test_accuracy*100).toFixed(1)}%</div><div class="label">Test Accuracy</div></div>
            <div class="metric-card"><div class="value">${data.total_samples}</div><div class="label">Test Samples</div></div>
            <div class="metric-card"><div class="value">${data.train_epochs}</div><div class="label">Epochs</div></div>
            <div class="metric-card"><div class="value" style="color:#448AFF">${data.model_params}</div><div class="label">Parameters</div></div>
        `;
        document.getElementById("cnn-confusion").src = "data:image/png;base64," + data.confusion_matrix;
        document.getElementById("cnn-curves").src = "data:image/png;base64," + data.training_curves;
        show("results-cnn");
    } catch(e) { console.error("CNN:", e); }
    finally { hide("loading-cnn"); }
}

// ── Grad-CAM batch ───────────────────────────────────────

async function runGradCAM() {
    show("loading-gradcam"); hide("results-gradcam");
    try {
        const res = await fetch("/api/dashboard/gradcam");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        document.getElementById("gradcam-batch").src = "data:image/png;base64," + data.gradcam_plot;
        show("results-gradcam");
    } catch(e) { console.error("GradCAM:", e); }
    finally { hide("loading-gradcam"); }
}

// ── ODE ──────────────────────────────────────────────────

async function runODE() {
    show("loading-ode"); hide("results-ode");
    try {
        const res = await fetch("/api/dashboard/ode");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);

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
    } catch(e) { console.error("ODE:", e); }
    finally { hide("loading-ode"); }
}

// ── HMM ──────────────────────────────────────────────────

async function runHMM() {
    show("loading-hmm"); hide("results-hmm");
    try {
        const res = await fetch("/api/dashboard/hmm");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);

        document.getElementById("hmm-metrics").innerHTML = `
            <div class="metric-card"><div class="value" style="color:#FF9100">${(data.cnn_accuracy*100).toFixed(1)}%</div><div class="label">CNN-Only</div></div>
            <div class="metric-card"><div class="value" style="color:#00E676">${(data.hmm_accuracy*100).toFixed(1)}%</div><div class="label">HMM-Corrected</div></div>
            <div class="metric-card"><div class="value" style="color:#448AFF">+${(data.improvement*100).toFixed(1)}%</div><div class="label">Improvement</div></div>
            <div class="metric-card"><div class="value">${data.n_frames}</div><div class="label">Frames</div></div>
        `;
        document.getElementById("hmm-plot").src = "data:image/png;base64," + data.hmm_plot;
        show("results-hmm");
    } catch(e) { console.error("HMM:", e); }
    finally { hide("loading-hmm"); }
}

// ── Checkpoint ───────────────────────────────────────────

async function runCheckpoint() {
    show("loading-checkpoint"); hide("results-checkpoint");
    try {
        const res = await fetch("/api/dashboard/checkpoint");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);

        document.getElementById("checkpoint-metrics").innerHTML = `
            <div class="metric-card"><div class="value" style="color:#FF1744">${data.n_cnn_anomalies}</div><div class="label">CNN-Only Anomalies</div></div>
            <div class="metric-card"><div class="value" style="color:#00E676">${data.n_hmm_anomalies}</div><div class="label">HMM-Corrected</div></div>
        `;
        document.getElementById("checkpoint-plot").src = "data:image/png;base64," + data.checkpoint_plot;

        const list = document.getElementById("anomaly-list");
        if (data.anomalies_hmm.length === 0) {
            list.innerHTML = '<p style="color:#666;font-size:0.78rem;">No anomalies detected.</p>';
        } else {
            list.innerHTML = data.anomalies_hmm.map(a => `
                <div class="anomaly-item severity-${a.severity}">
                    <div class="anomaly-header">
                        <span class="anomaly-badge" style="background:${SEVERITY_COLORS[a.severity]}30;color:${SEVERITY_COLORS[a.severity]}">${a.severity}</span>
                        ${a.checkpoint} (${a.phase})
                    </div>
                    <div class="anomaly-body">${a.bio_interpretation}</div>
                </div>
            `).join("");
        }
        show("results-checkpoint");
    } catch(e) { console.error("Checkpoint:", e); }
    finally { hide("loading-checkpoint"); }
}

// ── Population ───────────────────────────────────────────

async function runPopulation() {
    show("loading-population"); hide("results-population");
    try {
        const res = await fetch("/api/dashboard/population");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);

        document.getElementById("pop-metrics").innerHTML = `
            <div class="metric-card"><div class="value">${data.total_cells}</div><div class="label">Cells</div></div>
            <div class="metric-card"><div class="value" style="color:#FF1744">${(data.mitotic_index*100).toFixed(1)}%</div><div class="label">Mitotic Index</div></div>
            <div class="metric-card"><div class="value" style="color:#448AFF">${(data.growth_fraction*100).toFixed(1)}%</div><div class="label">Growth Fraction</div></div>
            <div class="metric-card"><div class="value" style="color:#E040FB">${data.doubling_time_hours.toFixed(1)}h</div><div class="label">Doubling Time</div></div>
        `;
        document.getElementById("population-plot").src = "data:image/png;base64," + data.population_plot;
        document.getElementById("pop-clinical").innerHTML = `
            <h4>${data.status}</h4>
            <p>${data.clinical_significance}</p>
        `;
        show("results-population");
    } catch(e) { console.error("Population:", e); }
    finally { hide("loading-population"); }
}

// ═══ SIDEBAR: Live Inference ═════════════════════════════

const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const btnClassify = document.getElementById("btn-classify");
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
    hide("results-classify"); hide("error-msg");
}

btnClassify.addEventListener("click", async () => {
    if (!selectedFile) return;
    btnClassify.disabled = true;
    show("loading-classify"); hide("results-classify"); hide("error-msg");

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
            <span class="chip">${info.duration_hours}h</span>
            <span class="chip">${(info.fraction_of_cycle*100).toFixed(1)}% of cycle</span>
        `;

        let bars = "";
        for (const [p, prob] of Object.entries(data.probabilities)) {
            const pct = (prob*100).toFixed(1);
            bars += `<div class="prob-row"><div class="prob-label">${p}</div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${pct}%;background:${PHASE_COLORS[p]}"></div></div><div class="prob-value">${pct}%</div></div>`;
        }
        document.getElementById("prob-bars-container").innerHTML = bars;
        document.getElementById("gradcam-img").src = "data:image/png;base64," + data.gradcam_image;
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
        if (data.model_loaded && data.ode_ready && data.hmm_ready) {
            dot.className = "status-dot online";
            text.textContent = `Ready (${data.device})`;
        } else {
            dot.className = "status-dot offline";
            text.textContent = "Model not loaded";
        }
    } catch {
        document.getElementById("status-dot").className = "status-dot offline";
        document.getElementById("status-text").textContent = "Offline";
    }
}
checkStatus(); setInterval(checkStatus, 10000);
