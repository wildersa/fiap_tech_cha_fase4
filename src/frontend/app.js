Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";

let chart = null;
let resourceChart = null;
let latencyChart = null;
let currentModelCardType = 'single';
let currentInferenceModel = 'single';

function buildSinglePayload(requiredRows) {
  const count = Math.max(requiredRows, 65);
  const closes = Array.from({ length: count }, (_, i) => {
    const base = 43.0 + i * 0.035 + Math.sin(i / 4) * 0.45;
    return Number((base + Math.sin(i / 3) * 0.18).toFixed(2));
  });
  return JSON.stringify({ symbol: "PETR4.SA", closes }, null, 2);
}

function buildMultiPayload(requiredRows) {
  const count = Math.max(requiredRows, 65);
  const start = new Date("2024-01-02T00:00:00");
  const rows = [];
  for (let i = 0; rows.length < count; i++) {
    const date = new Date(start);
    date.setDate(start.getDate() + i);
    const day = date.getDay();
    if (day === 0 || day === 6) continue;

    const close = 43.1 + rows.length * 0.08 + Math.sin(rows.length / 4) * 0.25;
    rows.push({
      date: date.toISOString().slice(0, 10),
      open: Number((close - 0.12).toFixed(2)),
      high: Number((close + 0.42).toFixed(2)),
      low: Number((close - 0.36).toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: 12000000 + rows.length * 85000
    });
  }
  return JSON.stringify({ symbol: "PETR4.SA", rows }, null, 2);
}

async function changeInferenceModel(type) {
  currentInferenceModel = type;
  const btnSingle = document.getElementById('btn-inf-single');
  const btnMulti = document.getElementById('btn-inf-multi');
  const infoEl = document.getElementById('inference-model-info');
  const payloadTextarea = document.getElementById('payload');

  // Fetch dynamic window size
  let windowSize = 60; // default
  try {
    const res = await fetch(`/model-card?type=${type}`);
    if(res.ok) {
       const data = await res.json();
       if(data.training_details && data.training_details.window_size) {
          windowSize = data.training_details.window_size;
       }
    }
  } catch (e) {
    console.warn("Failed to fetch window_size", e);
  }

  const requirementMsg = `<br/><span style="color:#fcd34d; font-weight:600; margin-top:4px; display:inline-block;">⚠️ O modelo atual exige um json contendo pelo menos ${windowSize + 1} fechamentos/dias para obter as sequências passadas necessárias.</span>`;

  if (type === 'single') {
    btnSingle.style.background = 'var(--accent)';
    btnSingle.style.color = '#fff';
    btnMulti.style.background = 'transparent';
    btnMulti.style.color = '#a0aec0';
    
    infoEl.innerHTML = `
      <strong style="color: var(--accent-hover); display: block; margin-bottom: 2px;">🏆 Melhor Modelo Univariado (Single)</strong>
      <span>Esta opção executa inferência usando o melhor modelo univariado promovido pelo MLflow (armazenado em <code>models/lstm_petr4</code>). Requer apenas o histórico de fechamentos recentes.${requirementMsg}</span>
    `;
    payloadTextarea.value = buildSinglePayload(windowSize + 1);
  } else {
    btnSingle.style.background = 'transparent';
    btnSingle.style.color = '#a0aec0';
    btnMulti.style.background = 'var(--accent)';
    btnMulti.style.color = '#fff';

    infoEl.innerHTML = `
      <strong style="color: var(--accent-hover); display: block; margin-bottom: 2px;">🏆 Melhor Modelo Multivariado (Multi)</strong>
      <span>Esta opção executa inferência usando o melhor modelo multivariado promovido pelo MLflow (armazenado em <code>models/lstm_petr4_multi</code>). Requer a série histórica de dados OHLCV completos.${requirementMsg}</span>
    `;
    payloadTextarea.value = buildMultiPayload(windowSize + 1);
  }
}

const ENABLE_TRAINING_API = Boolean(window.DASHBOARD_CONFIG?.enableTrainingApi);
if (!ENABLE_TRAINING_API) {
  const trainTab = document.getElementById('tab-train');
  if (trainTab) trainTab.style.display = 'none';
}

function setModelCardType(type) {
  currentModelCardType = type;

  const btnSingle = document.getElementById('btn-mc-single');
  const btnMulti = document.getElementById('btn-mc-multi');

  if (!btnSingle || !btnMulti) return;
  if (type === 'single') {
    btnSingle.classList.add('active');
    btnMulti.classList.remove('active');
  } else {
    btnMulti.classList.add('active');
    btnSingle.classList.remove('active');
  }
}

async function syncChampionSelection({ updateInference = false, updateCard = true } = {}) {
  try {
    const res = await fetch('/model-champion');
    if (!res.ok) return null;

    const champion = await res.json();
    const selectedType = champion.has_champion ? champion.selected_model_type : null;
    if (!selectedType) return champion;

    setModelCardType(selectedType);

    if (updateInference && currentInferenceModel !== selectedType) {
      await changeInferenceModel(selectedType);
    }

    if (updateCard) {
      await loadModelCard();
    }

    return champion;
  } catch (e) {
    console.warn('Falha ao sincronizar champion:', e);
    return null;
  }
}

function showTab(name) {
  for (const tab of ['home', 'inference', 'telemetry', 'train']) {
    const el = document.getElementById(tab);
    const btn = document.getElementById('tab-' + tab);
    if (el) el.classList.toggle('hidden', name !== tab);
    if (btn) btn.classList.toggle('active', name === tab);
  }
  if (name === 'home') loadModelCard();
  if (name === 'telemetry') loadTelemetry();
  if (name === 'train') loadRuns();
  localStorage.setItem('activeTab', name);
}

function money(value) {
  return Number(value).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
}

async function runPrediction() {
  const errEl = document.getElementById('error');
  errEl.textContent = '';
  errEl.classList.add('hidden');

  let payload;
  try {
    payload = JSON.parse(document.getElementById('payload').value);
  } catch (err) {
    errEl.textContent = 'JSON inválido: ' + err.message;
    errEl.classList.remove('hidden');
    return;
  }

  const start = performance.now();
  try {
    const endpoint = currentInferenceModel === 'multi' ? '/predict/ohlcv' : '/predict';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const elapsed = performance.now() - start;
    const data = await response.json();

    if (!response.ok) {
      errEl.textContent = data.detail || JSON.stringify(data, null, 2);
      errEl.classList.remove('hidden');
      return;
    }

    document.getElementById('lastClose').textContent = money(data.last_close);
    document.getElementById('predictedClose').textContent = money(data.predicted_close);

    const dirEl = document.getElementById('direction');
    dirEl.textContent = data.predicted_direction.toUpperCase();
    dirEl.style.color = data.predicted_direction === 'alta' ? 'var(--success)' : 'var(--danger)';

    const retPct = Number(data.predicted_return_pct);
    const retEl = document.getElementById('returnPct');
    retEl.textContent = (retPct > 0 ? '+' : '') + retPct.toFixed(3) + '%';
    retEl.style.color = retPct > 0 ? 'var(--success)' : 'var(--danger)';

    document.getElementById('diffAbs').textContent = (data.predicted_change_abs > 0 ? '+' : '') + money(data.predicted_change_abs);
    document.getElementById('latency').textContent = elapsed.toFixed(1) + ' ms';

    let closes = [];
    if (currentInferenceModel === 'multi') {
      closes = (payload.rows || []).map(r => r.close);
    } else {
      closes = payload.closes || [];
    }
    renderChart(closes, data.predicted_close);
    loadTelemetry();
    await syncChampionSelection({ updateInference: true, updateCard: true });
  } catch (e) {
    errEl.textContent = 'Erro de rede: ' + e.message;
    errEl.classList.remove('hidden');
  }
}

function renderChart(closes, predicted) {
  const recent = closes.slice(-30).map(Number);
  const labels = recent.map((_, i) => String(i - recent.length + 1)).concat(['Previsão']);
  const closeSeries = recent.concat(null);
  const predictedSeries = new Array(recent.length).fill(null).concat(Number(predicted));
  const ctx = document.getElementById('priceChart');

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Fechamentos Históricos',
          data: closeSeries,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.3,
          borderWidth: 2,
          fill: true
        },
        {
          label: 'Projeção LSTM',
          data: predictedSeries,
          borderColor: '#a78bfa',
          backgroundColor: '#a78bfa',
          pointBackgroundColor: '#a78bfa',
          pointBorderColor: '#fff',
          pointRadius: 6,
          pointHoverRadius: 8,
          borderWidth: 2,
          borderDash: [5, 5]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.9)', titleFont: { size: 13 }, bodyFont: { size: 14 }, padding: 12, cornerRadius: 8, displayColors: false }
      },
      scales: {
        x: { grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false } },
        y: { grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false }, ticks: { callback: (val) => 'R$ ' + val } }
      }
    }
  });
}

async function loadTelemetry() {
  try {
    const response = await fetch('/telemetry');
    const data = await response.json();

    document.getElementById('tel-reqs').textContent = data.api.prediction_requests ?? 0;
    const errRate = data.api.total_requests > 0 ? (data.api.total_errors / data.api.total_requests * 100) : 0;
    const errEl = document.getElementById('tel-errs');
    errEl.textContent = errRate.toFixed(1) + '%';
    errEl.style.color = errRate > 0 ? 'var(--danger)' : 'var(--success)';

    document.getElementById('tel-uptime').textContent = data.uptime_seconds + 's';
    document.getElementById('tel-lat').textContent = data.inference.average_time_ms.toFixed(1) + ' ms';

    const trainEl = document.getElementById('tel-train-time');
    if (trainEl) {
      if (!data.training.enabled) {
        trainEl.textContent = 'N/A';
        trainEl.style.color = 'var(--muted)';
      } else if (data.training.last_time_sec === null || data.training.last_time_sec === undefined) {
        trainEl.textContent = 'Aguardando';
        trainEl.style.color = 'var(--muted)';
      } else {
        trainEl.textContent = data.training.last_time_sec + 's';
        trainEl.style.color = '';
      }
    }

    const history = data.history || [];
    const labels = history.map(h => h.timestamp);
    const cpuData = history.map(h => h.cpu_percent);
    const memData = history.map(h => h.memory_mb);
    const latData = history.map(h => h.latency_ms);

    if (resourceChart) resourceChart.destroy();
    resourceChart = new Chart(document.getElementById('resourceChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'CPU (%)',
            data: cpuData,
            yAxisID: 'cpu',
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.08)',
            tension: 0.2,
            pointRadius: 1,
            borderWidth: 2
          },
          {
            label: 'RAM (MB)',
            data: memData,
            yAxisID: 'memory',
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.08)',
            tension: 0.2,
            pointRadius: 1,
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { grid: { display: false } },
          cpu: {
            type: 'linear',
            position: 'left',
            min: 0,
            max: 100,
            title: { display: true, text: 'CPU (%)' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { callback: (value) => value + '%' }
          },
          memory: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'RAM (MB)' },
            grid: { drawOnChartArea: false },
            ticks: { callback: (value) => value + ' MB' }
          }
        }
      }
    });

    if (latencyChart) latencyChart.destroy();
    latencyChart = new Chart(document.getElementById('latencyChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Latência (ms)', data: latData, borderColor: '#a78bfa', backgroundColor: 'rgba(167, 139, 250, 0.1)', fill: true, tension: 0.2, pointRadius: 1, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        scales: { x: { grid: { display: false } }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } }
      }
    });

    // Carrega e processa as métricas do Prometheus
    await loadPrometheusMetrics();
  } catch (e) {
    console.error('Falha ao carregar telemetria:', e);
  }
}

async function loadPrometheusMetrics() {
  try {
    const response = await fetch('/metrics');
    if (!response.ok) return;
    const text = await response.text();
    if (!text) return;

    const lines = text.split('\n');
    let status2xx = 0;
    let status4xx = 0;
    let status5xx = 0;
    let endpointCounts = {};
    let endpointSums = {};

    for (let line of lines) {
      if (line.startsWith('#') || !line.trim()) continue;
      
      const matchValue = line.match(/\s+([\d.e+-]+)(?:\s*|$)/);
      if (!matchValue) continue;
      const value = parseFloat(matchValue[1]);

      if (line.includes('requests_total') || line.includes('requests_created') || line.includes('duration_seconds_count')) {
        const statusMatch = line.match(/(?:status|status_code|http_status|code)="(\d+)"/);
        if (statusMatch) {
          const status = parseInt(statusMatch[1]);
          if (status >= 200 && status < 300) status2xx += value;
          else if (status >= 400 && status < 500) status4xx += value;
          else if (status >= 500 && status < 600) status5xx += value;
        }
      }

      const handlerMatch = line.match(/(?:handler|path|endpoint|route)="([^"]+)"/);
      if (handlerMatch) {
        const handler = handlerMatch[1];
        if (handler !== '/metrics' && handler !== '/favicon.ico') {
          if (line.includes('duration_seconds_sum') || line.includes('duration_ms_sum')) {
            endpointSums[handler] = (endpointSums[handler] || 0) + value;
          }
          if (line.includes('duration_seconds_count') || line.includes('requests_total') || line.includes('duration_ms_count')) {
            endpointCounts[handler] = (endpointCounts[handler] || 0) + value;
          }
        }
      }
    }

    document.getElementById('metrics-2xx').textContent = Math.round(status2xx);
    document.getElementById('metrics-4xx').textContent = Math.round(status4xx);
    document.getElementById('metrics-5xx').textContent = Math.round(status5xx);

    const listEl = document.getElementById('metrics-endpoints-list');
    listEl.innerHTML = '';
    const sortedEndpoints = Object.keys(endpointCounts).sort((a, b) => endpointCounts[b] - endpointCounts[a]);
    if (sortedEndpoints.length === 0) {
      listEl.innerHTML = '<div class="muted" style="text-align: center; padding-top: 10px;">Nenhuma requisição registrada.</div>';
    } else {
      for (let handler of sortedEndpoints) {
        const count = endpointCounts[handler];
        const sum = endpointSums[handler] || 0;
        const avgMs = count > 0 ? (sum / count) * 1000 : 0;
        const avgText = avgMs > 0 ? `${avgMs.toFixed(1)} ms` : 'N/A';
        
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.justify = 'space-between';
        row.style.alignItems = 'center';
        row.style.borderBottom = '1px solid rgba(255,255,255,0.03)';
        row.style.padding = '4px 0';
        row.innerHTML = `<span style="font-family: monospace; color: #a78bfa;">${handler}</span><span><strong>${Math.round(count)} reqs</strong> <span class="muted" style="font-size:10px; margin-left:8px;">(${avgText})</span></span>`;
        listEl.appendChild(row);
      }
    }
  } catch (e) {
    console.error('Falha ao processar /metrics do Prometheus:', e);
  }
}

async function changeModelCardView(type) {
  setModelCardType(type);
  await loadModelCard();
}

async function loadModelCard() {
  try {
    const res = await fetch(`/model-card?type=${currentModelCardType}`);
    const data = await res.json();

    const overview = data.model_overview || {};
    const training = data.training_details || {};
    const evaluation = data.evaluation_details || {};
    const deployment = data.deployment_details || {};
    const additional = data.additional_information || {};

    const art = evaluation.artifacts_available || {};
    const isLoaded = art.model_onnx && art.preprocessor_joblib;
    const statusBadge = document.getElementById('mc-status');
    if (statusBadge) {
      if (isLoaded) {
        statusBadge.textContent = 'ONLINE (' + (overview.model_status || 'Approved') + ')';
        statusBadge.style.background = 'rgba(16, 185, 129, 0.2)';
        statusBadge.style.color = '#10b981';
        statusBadge.style.border = '1px solid rgba(16, 185, 129, 0.3)';
      } else {
        statusBadge.textContent = 'OFFLINE (Faltam Artefatos)';
        statusBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        statusBadge.style.color = '#ef4444';
        statusBadge.style.border = '1px solid rgba(239, 68, 68, 0.3)';
      }
    }

    const updateArtBadge = (id, exists) => {
      const badge = document.getElementById(id);
      if (badge) {
        if (exists) {
          badge.style.background = 'rgba(16, 185, 129, 0.15)';
          badge.style.color = '#10b981';
          badge.style.border = '1px solid rgba(16, 185, 129, 0.3)';
        } else {
          badge.style.background = 'rgba(239, 68, 68, 0.15)';
          badge.style.color = '#ef4444';
          badge.style.border = '1px solid rgba(239, 68, 68, 0.3)';
        }
      }
    };
    updateArtBadge('mc-art-onnx', art.model_onnx);
    updateArtBadge('mc-art-safetensors', art.model_safetensors);
    updateArtBadge('mc-art-preproc', art.preprocessor_joblib);
    updateArtBadge('mc-art-meta', art.metadata_json);

    document.getElementById('mc-name').textContent = overview.model_name || '-';
    document.getElementById('mc-version').textContent = overview.model_version || '-';
    document.getElementById('mc-description').textContent = overview.model_description || '-';
    document.getElementById('mc-intended-uses').textContent = overview.intended_uses || '-';

    const riskRating = document.getElementById('mc-risk-rating');
    if (riskRating && overview.risk_rating) {
      riskRating.textContent = overview.risk_rating + ' Risk';
      if (overview.risk_rating.toLowerCase() === 'high') {
        riskRating.style.background = 'rgba(239, 68, 68, 0.15)';
        riskRating.style.color = '#ef4444';
        riskRating.style.border = '1px solid rgba(239, 68, 68, 0.3)';
      } else if (overview.risk_rating.toLowerCase() === 'medium') {
        riskRating.style.background = 'rgba(245, 158, 11, 0.15)';
        riskRating.style.color = '#f59e0b';
        riskRating.style.border = '1px solid rgba(245, 158, 11, 0.3)';
      } else {
        riskRating.style.background = 'rgba(16, 185, 129, 0.15)';
        riskRating.style.color = '#10b981';
        riskRating.style.border = '1px solid rgba(16, 185, 129, 0.3)';
      }
    }

    document.getElementById('mc-symbol').textContent = training.symbol || '-';
    document.getElementById('mc-window').textContent = training.window_size ? training.window_size + ' dias' : '-';
    document.getElementById('mc-target').textContent = training.target_mode || '-';
    document.getElementById('mc-feature-mode').textContent = training.feature_mode || '-';
    document.getElementById('mc-feature-scaler').textContent = training.feature_scaler_type || '-';
    document.getElementById('mc-target-scaler').textContent = training.target_scaler_type || '-';
    document.getElementById('mc-data-source').textContent = training.data_source || '-';
    document.getElementById('mc-run-id').textContent = training.mlflow_run_id || '-';
    document.getElementById('mc-runtime-mode').textContent = deployment.runtime_mode_label || '-';
    document.getElementById('mc-model-source').textContent = deployment.model_source_policy || '-';

    const featureCols = training.feature_cols || [];
    const featureContainer = document.getElementById('mc-features-container');
    if (featureContainer && featureCols.length > 1) {
      featureContainer.style.display = 'block';
      const featureCountEl = document.getElementById('mc-feature-count');
      const featureColsEl = document.getElementById('mc-feature-cols');
      if (featureCountEl) featureCountEl.textContent = training.feature_count || featureCols.length;
      if (featureColsEl) featureColsEl.textContent = featureCols.join(', ');
    } else if (featureContainer) {
      featureContainer.style.display = 'none';
    }

    document.getElementById('mc-observations').textContent = training.training_observations || '-';

    document.getElementById('mc-eval-dataset').textContent = evaluation.evaluation_dataset || '-';

    const evalMetrics = evaluation.metrics || {};
    const lstmTest = evalMetrics.lstm_test || {};
    const baselineTest = evalMetrics.baseline_test || {};
    const relativeGain = evalMetrics.relative_gain_vs_baseline_pct || {};
    const metricNumber = (value) => {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const formatNumber = (value, digits = 2, suffix = '') => {
      const numeric = metricNumber(value);
      return numeric === null ? '-' : numeric.toFixed(digits) + suffix;
    };
    const formatGain = (value) => {
      const numeric = metricNumber(value);
      if (numeric === null) return '-';
      return (numeric > 0 ? '+' : '') + numeric.toFixed(1) + '%';
    };
    const gainMapeValue = metricNumber(relativeGain.mape_pct);

    const setText = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
      return el;
    };

    setText('mc-mape', formatNumber(lstmTest.mape_pct, 2, '%'));
    setText('mc-baseline-mape', formatNumber(baselineTest.mape_pct, 2, '%'));
    const gainMapeEl = document.getElementById('mc-gain-mape');
    if (gainMapeEl) {
      gainMapeEl.textContent = formatGain(gainMapeValue);
      gainMapeEl.style.color = gainMapeValue > 0 ? 'var(--success)' : (gainMapeValue < 0 ? 'var(--danger)' : 'inherit');
    }
    setText('mc-mae', formatNumber(lstmTest.mae, 2));
    setText('mc-rmse', formatNumber(lstmTest.rmse, 2));
    setText('mc-dir', formatNumber(evalMetrics.directional_accuracy_test_lstm_pct, 1, '%'));

    document.getElementById('mc-ethical').textContent = additional.ethical_considerations || '-';
    document.getElementById('mc-caveats').textContent = additional.caveats_and_recommendations || '-';

    const imgEl = document.getElementById('mc-image');
    const zoomBtn = document.getElementById('btn-zoom-image');
    if (imgEl) {
      imgEl.onload = () => {
        imgEl.style.display = 'block';
        if (zoomBtn) zoomBtn.style.display = 'block';
      };
      imgEl.onerror = () => {
        imgEl.style.display = 'none';
        if (zoomBtn) zoomBtn.style.display = 'none';
      };
      imgEl.src = `/model-image?type=${currentModelCardType}&t=${new Date().getTime()}`;
    }

  } catch (e) {
    console.error('Falha ao carregar Model Card:', e);
  }
}

function clearLineage() {
  const el = document.getElementById('train-parent-run-id');
  if (el) el.value = "";
  const badge = document.getElementById('lineage-badge');
  if (badge) badge.classList.add('hidden');
}

const FEATURE_REGISTRY = [
  { name: "Return", label: "Retorno Simples", group: "Retornos", desc: "Variação percentual de fechamento dia a dia.", requires: ["Close"] },
  { name: "Log_Return", label: "Retorno Logarítmico", group: "Retornos", desc: "Log-retorno de fechamento diário.", requires: ["Close"] },
  { name: "Log_Return_Lag1", label: "Lag 1 de Log-Retorno", group: "Retornos", desc: "Log-retorno do pregão anterior.", requires: ["Close"] },
  { name: "Log_Return_Lag2", label: "Lag 2 de Log-Retorno", group: "Retornos", desc: "Log-retorno de 2 pregões atrás.", requires: ["Close"] },
  { name: "Log_Return_Lag3", label: "Lag 3 de Log-Retorno", group: "Retornos", desc: "Log-retorno de 3 pregões atrás.", requires: ["Close"] },
  { name: "Log_Return_Lag5", label: "Lag 5 de Log-Retorno", group: "Retornos", desc: "Log-retorno de 5 pregões atrás.", requires: ["Close"] },
  { name: "Rolling_Return_5", label: "Retorno Acumulado (5d)", group: "Retornos", desc: "Retorno acumulado nos últimos 5 dias.", requires: ["Close"] },
  { name: "Rolling_Return_20", label: "Retorno Acumulado (20d)", group: "Retornos", desc: "Retorno acumulado nos últimos 20 dias.", requires: ["Close"] },

  { name: "SMA_7", label: "Média Móvel Simples 7d", group: "Tendência", desc: "Média dos preços de fechamento nos últimos 7 dias.", requires: ["Close"] },
  { name: "SMA_21", label: "Média Móvel Simples 21d", group: "Tendência", desc: "Média dos preços de fechamento nos últimos 21 dias.", requires: ["Close"] },
  { name: "Momentum_5", label: "Momentum 5d", group: "Tendência", desc: "Razão do fechamento de hoje contra 5 dias atrás.", requires: ["Close"] },
  { name: "RSI_14", label: "Índice de Força Relativa (14d)", group: "Tendência", desc: "Mede velocidade e mudança de movimentos de preço.", requires: ["Close"] },
  { name: "MACD", label: "Linha MACD", group: "Tendência", desc: "Diferença entre EMAs de 12 e 26 dias.", requires: ["Close"] },
  { name: "MACD_Signal", label: "Sinal do MACD", group: "Tendência", desc: "EMA de 9 dias da linha MACD.", requires: ["Close"] },
  { name: "MACD_Hist", label: "Histograma MACD", group: "Tendência", desc: "Diferença entre linha MACD e sinal.", requires: ["Close"] },

  { name: "Volatility_21", label: "Volatilidade 21d", group: "Volatilidade", desc: "Desvio padrão dos log-retornos nos últimos 21 dias.", requires: ["Close"] },
  { name: "BB_Width", label: "Largura das Bandas de Bollinger", group: "Volatilidade", desc: "Mede a dispersão das bandas de volatilidade.", requires: ["Close"] },
  { name: "ATR_14", label: "Average True Range (14d)", group: "Volatilidade", desc: "Mede a amplitude média real de negociação.", requires: ["High", "Low", "Close"] },
  { name: "Range_Pct", label: "Range Diário %", group: "Volatilidade", desc: "Razão entre Máxima/Mínima e o Fechamento.", requires: ["High", "Low", "Close"] },

  { name: "Volume_Z", label: "Volume Normalizado (Z-score)", group: "Volume", desc: "Z-score do volume transacionado em janela de 21 dias.", requires: ["Volume"] },
  { name: "Log_Volume", label: "Log do Volume", group: "Volume", desc: "Logaritmo natural do volume transacionado.", requires: ["Volume"] },

  { name: "Day_Of_Week", label: "Dia da Semana", group: "Calendário", desc: "Representação numérica do dia da semana (0-4).", requires: ["Date"] }
];

const FEATURE_PRESETS = {
  returns_basic: ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5"],
  returns_trend: ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5", "Momentum_5", "Rolling_Return_5", "Rolling_Return_20", "SMA_7", "SMA_21", "MACD", "MACD_Signal", "MACD_Hist"],
  returns_volatility: ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5", "Volatility_21", "BB_Width", "ATR_14", "Range_Pct"],
  technical_complete: ["Log_Return", "SMA_7", "SMA_21", "Volatility_21", "Momentum_5", "Range_Pct", "Volume_Z", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Width", "ATR_14", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5", "Rolling_Return_5", "Rolling_Return_20", "Day_Of_Week", "Log_Volume"]
};

function renderCalculatedFeatures() {
  const container = document.getElementById("groups-container");
  container.innerHTML = "";

  const groups = {};
  FEATURE_REGISTRY.forEach(f => {
    if (!groups[f.group]) groups[f.group] = [];
    groups[f.group].push(f);
  });

  for (const [groupName, features] of Object.entries(groups)) {
    const groupDiv = document.createElement("div");
    groupDiv.style.borderBottom = "1px solid var(--panel-border)";
    groupDiv.style.paddingBottom = "12px";

    const headerDiv = document.createElement("div");
    headerDiv.style.display = "flex";
    headerDiv.style.alignItems = "center";
    headerDiv.style.gap = "8px";
    headerDiv.style.marginBottom = "8px";

    const groupCbId = "group-" + groupName.replace(/\s+/g, "-");
    headerDiv.innerHTML = `
      <input type="checkbox" id="${groupCbId}" onchange="toggleGroup('${groupName}', this.checked)" style="width: 14px !important; height: 14px; margin: 0 !important; cursor: pointer;">
      <label for="${groupCbId}" style="cursor: pointer; font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase; margin: 0;">${groupName}</label>
    `;
    groupDiv.appendChild(headerDiv);

    const gridDiv = document.createElement("div");
    gridDiv.style.display = "grid";
    gridDiv.style.gridTemplateColumns = "repeat(auto-fill, minmax(280px, 1fr))";
    gridDiv.style.gap = "10px";

    features.forEach(f => {
      const itemDiv = document.createElement("div");
      itemDiv.style.background = "rgba(255,255,255,0.02)";
      itemDiv.style.border = "1px solid var(--panel-border)";
      itemDiv.style.borderRadius = "6px";
      itemDiv.style.padding = "8px";
      itemDiv.style.display = "flex";
      itemDiv.style.alignItems = "flex-start";
      itemDiv.style.gap = "8px";

      itemDiv.innerHTML = `
        <input type="checkbox" id="feat-${f.name}" value="${f.name}" onchange="updateCalculatedFeatures(true)" style="width: 14px !important; height: 14px; margin-top: 2px !important; cursor: pointer;">
        <div style="display: flex; flex-direction: column; gap: 2px;">
          <label for="feat-${f.name}" style="cursor: pointer; font-weight: 600; color: #fff; font-size: 12px; margin: 0;">${f.label} <code>(${f.name})</code></label>
          <span style="font-size: 10px; color: var(--muted); line-height: 1.3;">${f.desc}</span>
        </div>
      `;
      gridDiv.appendChild(itemDiv);
    });

    groupDiv.appendChild(gridDiv);
    container.appendChild(groupDiv);
  }
}

function toggleGroup(groupName, checked) {
  FEATURE_REGISTRY.forEach(f => {
    if (f.group === groupName) {
      const cb = document.getElementById("feat-" + f.name);
      if (cb) cb.checked = checked;
    }
  });
  updateCalculatedFeatures(true);
}

function toggleCalcFeaturesSection() {
  const checked = document.getElementById("use-calc-features").checked;
  const panel = document.getElementById("calc-features-panel");
  const featSelect = document.getElementById("train-feature");

  if (checked) {
    panel.classList.remove("hidden");
    featSelect.value = "technical_features";
    applyPreset();
  } else {
    panel.classList.add("hidden");
    featSelect.value = "single";
  }
}

function onFeatureModeChange() {
  const val = document.getElementById("train-feature").value;
  const useCalcCb = document.getElementById("use-calc-features");
  const panel = document.getElementById("calc-features-panel");

  if (val === "technical_features" || val === "custom") {
    useCalcCb.checked = true;
    panel.classList.remove("hidden");
    if (val === "custom") {
      document.getElementById("feature-preset-select").value = "custom";
    } else {
      if (document.getElementById("feature-preset-select").value === "custom") {
        document.getElementById("feature-preset-select").value = "technical_complete";
      }
      applyPreset();
    }
  } else {
    useCalcCb.checked = false;
    panel.classList.add("hidden");
  }
}

function applyPreset() {
  const preset = document.getElementById("feature-preset-select").value;
  if (preset === "custom") {
    document.getElementById("train-feature").value = "custom";
    return;
  }

  document.getElementById("train-feature").value = "technical_features";
  const presetList = FEATURE_PRESETS[preset] || [];

  FEATURE_REGISTRY.forEach(f => {
    const cb = document.getElementById("feat-" + f.name);
    if (cb) {
      cb.checked = presetList.includes(f.name);
    }
  });

  updateCalculatedFeatures(false);
}

function updateCalculatedFeatures(triggerSuggestions = true) {
  if (triggerSuggestions) {
    const checkAndSelect = (triggerName, targets) => {
      const triggerCb = document.getElementById("feat-" + triggerName);
      if (triggerCb && triggerCb.checked) {
        targets.forEach(target => {
          const targetCb = document.getElementById("feat-" + target);
          if (targetCb && !targetCb.checked) {
            targetCb.checked = true;
            const parent = targetCb.parentElement;
            if (parent) {
              parent.style.borderColor = "var(--primary)";
              setTimeout(() => { parent.style.borderColor = "var(--panel-border)"; }, 1500);
            }
          }
        });
      }
    };

    checkAndSelect("MACD", ["MACD_Signal", "MACD_Hist"]);
    checkAndSelect("SMA_7", ["SMA_21"]);
    checkAndSelect("Rolling_Return_5", ["Rolling_Return_20"]);
    checkAndSelect("Volume_Z", ["Log_Volume"]);
  }

  const groups = {};
  FEATURE_REGISTRY.forEach(f => {
    if (!groups[f.group]) groups[f.group] = [];
    groups[f.group].push(f);
  });

  for (const [groupName, features] of Object.entries(groups)) {
    const groupCb = document.getElementById("group-" + groupName.replace(/\s+/g, "-"));
    if (groupCb) {
      const allChecked = features.every(f => {
        const cb = document.getElementById("feat-" + f.name);
        return cb && cb.checked;
      });
      groupCb.checked = allChecked;
    }
  }

  let currentSelection = Array.from(document.querySelectorAll('#groups-container input[type="checkbox"][id^="feat-"]:checked'))
    .map(cb => cb.value);

  let matchedPreset = "custom";
  for (const [presetName, list] of Object.entries(FEATURE_PRESETS)) {
    if (list.length === currentSelection.length && list.every(item => currentSelection.includes(item))) {
      matchedPreset = presetName;
      break;
    }
  }

  const presetSelect = document.getElementById("feature-preset-select");
  presetSelect.value = matchedPreset;

  const featSelect = document.getElementById("train-feature");
  if (matchedPreset === "custom") {
    featSelect.value = "custom";
  } else {
    featSelect.value = "technical_features";
  }

  let needsOhlcv = false;
  FEATURE_REGISTRY.forEach(f => {
    const cb = document.getElementById("feat-" + f.name);
    if (cb && cb.checked) {
      if (f.requires.includes("High") || f.requires.includes("Low") || f.requires.includes("Volume")) {
        needsOhlcv = true;
      }
    }
  });

  const warnEl = document.getElementById("warning-multivariate");
  if (needsOhlcv) {
    warnEl.classList.remove("hidden");
  } else {
    warnEl.classList.add("hidden");
  }
}

function populateFormFromRunId(runId, buttonEl) {
  const run = window.allRunsData.find(x => x.run_id === runId);
  if (run && run.params) {
    populateTrainingForm(runId, run.params);
  }
  if (buttonEl) {
    buttonEl.textContent = "Preenchido!";
    setTimeout(() => buttonEl.textContent = "Preencher Formulário", 2000);
  }
}

async function deleteRun(runId, buttonEl) {
  if (!confirm("Tem certeza que deseja deletar esta run do MLflow?")) {
    return;
  }
  const originalText = buttonEl.textContent;
  buttonEl.textContent = "Deletando...";
  buttonEl.disabled = true;
  try {
    const response = await fetch('/runs/' + runId, {
      method: 'DELETE'
    });
    const data = await response.json();
    if (!response.ok) {
      alert('Erro ao deletar run: ' + (data.detail || JSON.stringify(data)));
      buttonEl.textContent = originalText;
      buttonEl.disabled = false;
    } else {
      loadRuns();
    }
  } catch (e) {
    alert('Erro de rede: ' + e.message);
    buttonEl.textContent = originalText;
    buttonEl.disabled = false;
  }
}

function populateTrainingForm(run_id, p) {
  const paramsForForm = { ...(p || {}) };
  const copiedEndDate = String(paramsForForm.end_date || '').trim().toLowerCase();
  if (!copiedEndDate || copiedEndDate === 'none' || copiedEndDate === 'null' || copiedEndDate === 'nan') {
    paramsForForm.end_date = paramsForForm.dataset_end_real || '';
  }

  const map = {
    symbol: 'train-symbol',
    start_date: 'train-start',
    end_date: 'train-end',
    target_mode: 'train-target',
    feature_mode: 'train-feature',
    device: 'train-device',
    hidden_size: 'train-hidden',
    num_layers: 'train-layers',
    window_size: 'train-window',
    batch_size: 'train-batch',
    max_epochs: 'train-epochs',
    patience: 'train-patience',
    learning_rate: 'train-lr',
    weight_decay: 'train-wd',
    dropout: 'train-dropout',
    grad_clip: 'train-clip',
    feature_scaler_type: 'train-feature-scaler',
    target_scaler_type: 'train-target-scaler'
  };
  for (const [key, id] of Object.entries(map)) {
    if (paramsForForm[key] !== undefined && document.getElementById(id)) {
      document.getElementById(id).value = paramsForForm[key];
    }
  }

  document.querySelectorAll('#groups-container input[type="checkbox"]').forEach(cb => cb.checked = false);

  const featureMode = paramsForForm.feature_mode || 'single';
  if (featureMode === 'technical_features' || featureMode === 'custom') {
    document.getElementById('use-calc-features').checked = true;
    document.getElementById('calc-features-panel').classList.remove('hidden');

    let preset = paramsForForm.feature_preset || 'custom';
    document.getElementById('feature-preset-select').value = preset;

    let selected = [];
    if (paramsForForm.selected_features) {
      try {
        if (paramsForForm.selected_features.trim().startsWith('[')) {
          selected = JSON.parse(paramsForForm.selected_features.replace(/'/g, '"'));
        } else {
          selected = paramsForForm.selected_features.split(',').map(x => x.trim()).filter(Boolean);
        }
      } catch (e) {
        // Fallback para representações em string de listas Python que podem falhar no JSON.parse
        selected = paramsForForm.selected_features.replace(/^\[|\]$/g, '').split(',').map(x => x.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean);
      }
    }

    if (selected.length === 0 && preset !== "custom" && FEATURE_PRESETS[preset]) {
      selected = FEATURE_PRESETS[preset];
    }

    selected.forEach(featName => {
      const cb = document.getElementById('feat-' + featName);
      if (cb) cb.checked = true;
    });

    updateCalculatedFeatures(false);
  } else {
    document.getElementById('use-calc-features').checked = false;
    document.getElementById('calc-features-panel').classList.add('hidden');
  }

  document.getElementById('train-feature').value = featureMode;

  const parentInput = document.getElementById('train-parent-run-id');
  if (parentInput) parentInput.value = run_id;
  const lineageId = document.getElementById('lineage-run-id');
  if (lineageId) lineageId.textContent = run_id.substring(0, 8);
  const lineageBadge = document.getElementById('lineage-badge');
  if (lineageBadge) lineageBadge.classList.remove('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function loadRuns() {
  const tbody = document.getElementById('runs-table-body');
  tbody.innerHTML = '<tr><td colspan="10" class="muted" style="text-align: center; padding: 32px;">Buscando métricas no MLflow...</td></tr>';
  try {
    const response = await fetch('/runs');
    const data = await response.json();
    window.allRunsData = data.runs || [];
    renderRunsTable();
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="10" class="error" style="margin: 0; border: none; text-align: center; padding: 32px;">Erro ao carregar MLflow: ${e.message}</td></tr>`;
  }
}

function numberOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function runMetric(run, key) {
  const metrics = run.metrics || {};
  if (key === 'baseline_gain_pct') return numberOrNull(metrics.baseline_gain_pct ?? metrics.gain_mape_pct);
  if (key === 'directional_accuracy_pct') return numberOrNull(metrics.directional_accuracy_test_lstm_pct ?? metrics.directional_accuracy_pct);
  return numberOrNull(metrics[key]);
}

function runRequiredRows(run) {
  const params = run.params || {};
  const windowSize = Number.parseInt(params.window_size || '60', 10);
  const targetMode = params.target_mode || 'log_returns';
  const featureMode = params.feature_mode || 'single';
  if (featureMode === 'single') {
    return windowSize + (targetMode === 'log_returns' || targetMode === 'returns' ? 1 : 0);
  }
  return windowSize + 21;
}

function runGroup(run) {
  return (run.params || {}).feature_mode === 'single' ? 'single' : 'multi';
}

function selectChampionRun(runs) {
  const eligible = runs
    .filter(run => run.status === 'FINISHED')
    .map(run => ({
      run,
      gain: runMetric(run, 'baseline_gain_pct'),
      mape: runMetric(run, 'test_lstm_mape_pct'),
      baseline: runMetric(run, 'test_baseline_mape_pct'),
      directional: runMetric(run, 'directional_accuracy_pct'),
      rows: runRequiredRows(run),
      windowSize: Number.parseInt((run.params || {}).window_size || '60', 10),
    }))
    .filter(item => item.gain !== null && item.gain > 0 && item.mape !== null && item.baseline !== null && item.mape < item.baseline);

  if (!eligible.length) return null;

  const maxGain = Math.max(...eligible.map(item => item.gain));
  return eligible
    .filter(item => maxGain - item.gain <= 0.3 + 1e-9)
    .sort((a, b) =>
      (b.directional ?? -Infinity) - (a.directional ?? -Infinity) ||
      a.mape - b.mape ||
      a.rows - b.rows ||
      a.windowSize - b.windowSize ||
      b.gain - a.gain
    )[0].run;
}

function championRunIds(runs) {
  const ids = new Map();
  for (const group of ['single', 'multi']) {
    const champion = selectChampionRun(runs.filter(run => runGroup(run) === group));
    if (champion) ids.set(champion.run_id, group);
  }
  return ids;
}

function formatRunDate(timestamp) {
  const numeric = Number(timestamp);
  if (!Number.isFinite(numeric)) return '-';
  return new Date(numeric).toLocaleString('pt-BR', {
    day: '2-digit',
    month: '2-digit',
    year: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function sortRuns(key) {
  const current = window.runsSort || { key: 'start_time', direction: 'desc' };
  window.runsSort = {
    key,
    direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc',
  };
  renderRunsTable();
}

function runSortValue(run, key) {
  const params = run.params || {};
  if (key === 'target_mode') return params.target_mode || '';
  if (key === 'feature_mode') return params.feature_mode || '';
  if (key === 'run_id') return run.run_id || '';
  if (key === 'status') return run.status || '';
  if (key === 'start_time') return Number(run.start_time || 0);
  return runMetric(run, key);
}

function runMatchesSearch(run, query) {
  if (!query) return true;
  const params = run.params || {};
  const metrics = run.metrics || {};
  const tags = run.tags || {};
  const text = [
    run.run_id,
    run.status,
    formatRunDate(run.start_time),
    params.target_mode,
    params.feature_mode,
    params.feature_preset,
    params.selected_features,
    tags.derived_from,
    metrics.test_lstm_mape_pct,
    metrics.test_baseline_mape_pct,
    metrics.baseline_gain_pct ?? metrics.gain_mape_pct,
    metrics.directional_accuracy_test_lstm_pct ?? metrics.directional_accuracy_pct,
  ].filter(value => value !== undefined && value !== null).join(' ').toLowerCase();
  return text.includes(query.toLowerCase().trim());
}

function updateSortIndicators() {
  const sort = window.runsSort || { key: 'start_time', direction: 'desc' };
  for (const id of ['run_id', 'status', 'start_time', 'target_mode', 'test_lstm_mape_pct', 'test_baseline_mape_pct', 'baseline_gain_pct', 'test_lstm_rmse', 'test_lstm_mae', 'directional_accuracy_pct']) {
    const el = document.getElementById('sort-' + id);
    if (el) el.textContent = sort.key === id ? (sort.direction === 'asc' ? '↑' : '↓') : '';
  }
}

function renderRunsTable() {
  const tbody = document.getElementById('runs-table-body');
  const runs = window.allRunsData || [];
  const searchEl = document.getElementById('runs-search');
  const query = searchEl ? searchEl.value : '';
  const sort = window.runsSort || { key: 'start_time', direction: 'desc' };
  window.runsSort = sort;

  updateSortIndicators();
  tbody.innerHTML = '';

  if (!runs.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="muted" style="text-align: center; padding: 32px;">Nenhum treinamento encontrado neste experimento.</td></tr>';
    return;
  }

  const filteredRuns = runs
    .filter(run => runMatchesSearch(run, query))
    .sort((a, b) => {
      const aValue = runSortValue(a, sort.key);
      const bValue = runSortValue(b, sort.key);
      const direction = sort.direction === 'asc' ? 1 : -1;
      if (typeof aValue === 'number' || typeof bValue === 'number') {
        return direction * ((aValue ?? -Infinity) - (bValue ?? -Infinity));
      }
      return direction * String(aValue ?? '').localeCompare(String(bValue ?? ''), 'pt-BR');
    });

  if (!filteredRuns.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="muted" style="text-align: center; padding: 32px;">Nenhuma run encontrada para a pesquisa.</td></tr>';
    return;
  }

  const champions = championRunIds(runs);

  for (const r of filteredRuns) {
    const tr = document.createElement('tr');
    const p = r.params || {};
    const m = r.metrics || {};

    const mapeValue = runMetric(r, 'test_lstm_mape_pct');
    const baselineMapeValue = runMetric(r, 'test_baseline_mape_pct');
    const gainMapeValue = runMetric(r, 'baseline_gain_pct');
    const rmseValue = runMetric(r, 'test_lstm_rmse');
    const maeValue = runMetric(r, 'test_lstm_mae');
    const dirAccValue = runMetric(r, 'directional_accuracy_pct');
    const mape = mapeValue !== null ? mapeValue.toFixed(2) + '%' : '-';
    const baselineMape = baselineMapeValue !== null ? baselineMapeValue.toFixed(2) + '%' : '-';
    const gainMape = gainMapeValue !== null ? (gainMapeValue > 0 ? '+' : '') + gainMapeValue.toFixed(1) + '%' : '-';
    const rmse = rmseValue !== null ? rmseValue.toFixed(4) : '-';
    const mae = maeValue !== null ? maeValue.toFixed(4) : '-';
    const dirAcc = dirAccValue !== null ? dirAccValue.toFixed(1) + '%' : '-';
    const tMode = p.target_mode || '-';
    const championGroup = champions.get(r.run_id);

    tr.style.cursor = 'pointer';
    tr.title = "Clique para ver os parâmetros";

    const isFinished = r.status === 'FINISHED';
    const statusHtml = `<span class="status-badge ${isFinished ? 'status-finished' : 'status-failed'}">
      ${isFinished ? '✓' : '✗'} ${r.status}
    </span>`;

    let runIdHtml = `<code>${r.run_id.substring(0, 8)}</code>`;
    if (championGroup) {
      const label = championGroup === 'single' ? 'Champion Univariado' : 'Champion Multivariado/Custom';
      runIdHtml = `<span title="${label}" style="margin-right: 4px;">🏆</span>${runIdHtml}`;
    }
    if (r.tags && r.tags.derived_from) {
      runIdHtml += `<br><span style="font-size: 9px; color: var(--muted); border: 1px solid var(--panel-border); border-radius: 4px; padding: 1px 4px; display: inline-block; margin-top: 4px; background: rgba(0,0,0,0.2);" title="Derivado do run ${r.tags.derived_from}">↳ de ${r.tags.derived_from.substring(0, 8)}</span>`;
    }

    tr.innerHTML = `
      <td>${runIdHtml}</td>
      <td>${statusHtml}</td>
      <td style="white-space: nowrap;">${formatRunDate(r.start_time)}</td>
      <td>${tMode}</td>
      <td style="font-weight: 500; color: ${mapeValue !== null && mapeValue < 2.0 ? 'var(--success)' : 'inherit'}">${mape}</td>
      <td style="color: var(--muted);">${baselineMape}</td>
      <td style="font-weight: 600; color: ${gainMapeValue > 0 ? 'var(--success)' : (gainMapeValue < 0 ? 'var(--danger)' : 'inherit')}">${gainMape}</td>
      <td>${rmse}</td>
      <td>${mae}</td>
      <td style="color: ${dirAccValue > 50 ? 'var(--success)' : 'inherit'}">${dirAcc}</td>
    `;

    tr.onclick = () => {
      const el = document.getElementById('details-' + r.run_id);
      if (el) el.classList.toggle('hidden');
    };

    tbody.appendChild(tr);

    const detailsTr = document.createElement('tr');
    detailsTr.id = 'details-' + r.run_id;
    detailsTr.className = 'hidden';

    let paramsHtml = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';
    for (const [k, v] of Object.entries(p)) {
      const displayVal = String(v).replace(/,/g, ', ');
      paramsHtml += `<div style="background: rgba(255,255,255,0.03); border: 1px solid var(--panel-border); border-radius: 4px; padding: 4px 8px; font-size: 12px; max-width: 100%; word-break: break-word;"><span style="color: var(--muted); margin-right: 4px;">${k}:</span><span style="color: var(--text); font-weight: 500;">${displayVal}</span></div>`;
    }
    paramsHtml += '</div>';

    let compareBtn = '';
    if (r.tags && r.tags.derived_from) {
      compareBtn = `<button class="secondary" style="padding: 4px 10px; font-size: 11px; margin-right: 8px; background: rgba(59, 130, 246, 0.15); border: 1px solid var(--primary); color: #93c5fd;" onclick='compareLineage("${r.run_id}", "${r.tags.derived_from}")'>📊 Comparar com Pai</button>`;
    }

    detailsTr.innerHTML = `
      <td colspan="10" style="padding: 16px; background: rgba(0,0,0,0.3); border-bottom: 1px solid rgba(255,255,255,0.05); max-width: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
          <div style="font-size: 13px; font-weight: 600; color: var(--accent-hover);">Hiperparâmetros do Run:</div>
          <div>
            ${compareBtn}
            <button class="primary" style="padding: 4px 10px; font-size: 11px; margin-right: 8px;" onclick='populateFormFromRunId("${r.run_id}", this)'>Preencher Formulário</button>
            <button class="danger" style="padding: 4px 10px; font-size: 11px; background: rgba(239, 68, 68, 0.15); border: 1px solid var(--danger); color: #fca5a5;" onclick='deleteRun("${r.run_id}", this)'>🗑️ Deletar Run</button>
          </div>
        </div>
        ${paramsHtml}
      </td>
    `;
    tbody.appendChild(detailsTr);
  }
}

async function runTraining() {
  const btn = document.getElementById('train-btn');
  const errEl = document.getElementById('train-error');
  const succEl = document.getElementById('train-success');

  errEl.classList.add('hidden');
  succEl.classList.add('hidden');

  btn.innerHTML = '<span style="display: inline-block; animation: pulse 1.5s infinite;">Executando Pipeline... (Isso pode levar minutos)</span>';
  btn.disabled = true;

  const pipeline = document.getElementById('train-pipeline');
  pipeline.classList.remove('hidden');
  for (let i = 1; i <= 4; i++) {
    document.getElementById('step-' + i).className = 'pipeline-node';
  }

  let currentStep = 1;
  // Fake progress indicator to give visual feedback on long poll
  const progressInterval = setInterval(() => {
    if (currentStep > 1 && currentStep <= 3) document.getElementById('step-' + (currentStep - 1)).className = 'pipeline-node done';
    if (currentStep <= 3) {
      document.getElementById('step-' + currentStep).className = 'pipeline-node active';
      currentStep++;
    }
  }, 4000);

  const useCalc = document.getElementById('use-calc-features').checked;
  let featureMode = document.getElementById('train-feature').value;
  let selectedFeatures = null;
  let featurePreset = null;

  if (useCalc) {
    const preset = document.getElementById('feature-preset-select').value;
    if (preset === 'custom') {
      featureMode = 'custom';
      featurePreset = 'custom';
      selectedFeatures = Array.from(document.querySelectorAll('#groups-container input[type="checkbox"][id^="feat-"]:checked'))
        .map(cb => cb.value);
    } else {
      featureMode = 'technical_features';
      featurePreset = preset;
      selectedFeatures = FEATURE_PRESETS[preset] || [];
    }
  } else {
    if (featureMode === 'technical_features' || featureMode === 'custom') {
      featureMode = 'single';
    }
  }

  const payload = {
    symbol: document.getElementById('train-symbol').value,
    start_date: document.getElementById('train-start').value,
    end_date: document.getElementById('train-end').value || null,
    target_mode: document.getElementById('train-target').value,
    feature_mode: featureMode,
    selected_features: selectedFeatures,
    feature_preset: featurePreset,
    device: document.getElementById('train-device').value,
    hidden_size: parseInt(document.getElementById('train-hidden').value, 10),
    num_layers: parseInt(document.getElementById('train-layers').value, 10),
    window_size: parseInt(document.getElementById('train-window').value, 10),
    batch_size: parseInt(document.getElementById('train-batch').value, 10),
    max_epochs: parseInt(document.getElementById('train-epochs').value, 10),
    patience: parseInt(document.getElementById('train-patience').value, 10),
    learning_rate: parseFloat(document.getElementById('train-lr').value),
    weight_decay: parseFloat(document.getElementById('train-wd').value),
    dropout: parseFloat(document.getElementById('train-dropout').value),
    grad_clip: parseFloat(document.getElementById('train-clip').value),
    feature_scaler_type: document.getElementById('train-feature-scaler').value,
    target_scaler_type: document.getElementById('train-target-scaler').value
  };

  const parentId = document.getElementById('train-parent-run-id');
  if (parentId && parentId.value) {
    payload.parent_run_id = parentId.value;
  }
  clearLineage();

  try {
    const response = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();

    if (!response.ok) {
      errEl.textContent = data.detail || JSON.stringify(data, null, 2);
      errEl.classList.remove('hidden');
    } else {
      const metrics = data.metrics || {};
      const testSummary = {
        lstm_test: metrics.lstm_test || {},
        baseline_test: metrics.baseline_test || {},
        relative_gain_vs_baseline_pct: metrics.relative_gain_vs_baseline_pct || {}
      };
      succEl.textContent = data.message + '\n\nMétricas de Validação e Teste:\n' + JSON.stringify(testSummary, null, 2);
      succEl.classList.remove('hidden');
      loadRuns();
      await syncChampionSelection({ updateInference: true, updateCard: true });
    }
  } catch (e) {
    errEl.textContent = 'Erro de rede: O servidor pode ter demorado demais para responder ou caiu. ' + e.message;
    errEl.classList.remove('hidden');
  } finally {
    clearInterval(progressInterval);
    for (let i = 1; i <= 4; i++) {
      const st = document.getElementById('step-' + i);
      if (!errEl.classList.contains('hidden')) st.className = 'pipeline-node error';
      else st.className = 'pipeline-node done';
    }
    btn.textContent = 'Inicializar Pipeline de Treinamento';
    btn.disabled = false;
    if (errEl.classList.contains('hidden')) {
      setTimeout(() => pipeline.classList.add('hidden'), 8000);
    }
  }
}

function compareLineage(childId, parentId) {
  const child = window.allRunsData.find(x => x.run_id === childId);
  const parent = window.allRunsData.find(x => x.run_id === parentId);

  if (!child || !parent) {
    alert('O run pai não está nos 100 resultados mais recentes para comparação.');
    return;
  }

  let paramDiffs = '';
  for (const [k, v] of Object.entries(child.params)) {
    if (parent.params[k] !== v) {
      const formattedParent = String(parent.params[k] || 'N/A').replace(/,/g, ', ');
      const formattedChild = String(v).replace(/,/g, ', ');
      paramDiffs += `<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed rgba(255,255,255,0.1); gap: 16px; align-items: flex-start;">
        <span style="color: var(--muted); min-width: 120px; word-break: break-word;">${k}</span>
        <span style="word-break: break-word; text-align: right; max-width: 70%;"><del style="color: var(--danger); margin-right: 8px;">${formattedParent}</del> <strong style="color: var(--success);">➔ ${formattedChild}</strong></span>
      </div>`;
    }
  }
  if (!paramDiffs) paramDiffs = '<span class="muted">Nenhuma alteração de hiperparâmetro.</span>';

  const mChild = child.metrics || {};
  const mParent = parent.metrics || {};

  function metricDiff(name, key, isLowerBetter) {
    const vc = mChild[key];
    const vp = mParent[key];
    if (vc === undefined || vp === undefined) return '';
    const diff = vc - vp;
    if (Math.abs(diff) < 0.0001) return '';
    let color = 'inherit';
    let arrow = '';
    if (diff < 0) { color = isLowerBetter ? 'var(--success)' : 'var(--danger)'; arrow = '↓'; }
    if (diff > 0) { color = isLowerBetter ? 'var(--danger)' : 'var(--success)'; arrow = '↑'; }

    return `<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px dashed rgba(255,255,255,0.1);">
      <span style="color: var(--muted);">${name}</span>
      <span>${vp.toFixed(4)} ➔ <strong style="color: ${color};">${vc.toFixed(4)} ${arrow} (${diff > 0 ? '+' : ''}${diff.toFixed(4)})</strong></span>
    </div>`;
  }

  let metricsHtml = metricDiff('MAPE LSTM (%)', 'test_lstm_mape_pct', true);
  metricsHtml += metricDiff('RMSE Teste', 'test_lstm_rmse', true);
  metricsHtml += metricDiff('MAE Teste', 'test_lstm_mae', true);
  metricsHtml += metricDiff('Acurácia Dir. (%)', 'directional_accuracy_pct', false);
  if (!metricsHtml) metricsHtml = '<span class="muted">Métricas muito similares ou ausentes.</span>';

  let insights = [];
  let score = 0;

  const diffMAPE = (mChild.test_lstm_mape_pct || 0) - (mParent.test_lstm_mape_pct || 0);
  const diffRMSE = (mChild.test_lstm_rmse || 0) - (mParent.test_lstm_rmse || 0);
  const diffMAE = (mChild.test_lstm_mae || 0) - (mParent.test_lstm_mae || 0);
  const diffDIR = (mChild.directional_accuracy_pct || 0) - (mParent.directional_accuracy_pct || 0);

  const vParentMAPE = mParent.test_lstm_mape_pct !== undefined ? mParent.test_lstm_mape_pct.toFixed(2) + '%' : 'N/A';
  const vChildMAPE = mChild.test_lstm_mape_pct !== undefined ? mChild.test_lstm_mape_pct.toFixed(2) + '%' : 'N/A';

  const vParentRMSE = mParent.test_lstm_rmse !== undefined ? mParent.test_lstm_rmse.toFixed(4) : 'N/A';
  const vChildRMSE = mChild.test_lstm_rmse !== undefined ? mChild.test_lstm_rmse.toFixed(4) : 'N/A';

  const vParentMAE = mParent.test_lstm_mae !== undefined ? mParent.test_lstm_mae.toFixed(4) : 'N/A';
  const vChildMAE = mChild.test_lstm_mae !== undefined ? mChild.test_lstm_mae.toFixed(4) : 'N/A';

  const vParentDIR = mParent.directional_accuracy_pct !== undefined ? mParent.directional_accuracy_pct.toFixed(1) + '%' : 'N/A';
  const vChildDIR = mChild.directional_accuracy_pct !== undefined ? mChild.directional_accuracy_pct.toFixed(1) + '%' : 'N/A';

  // 1. Métricas Insights
  if (Math.abs(diffMAPE) > 0.001) {
    if (diffMAPE < 0) {
      insights.push(`O erro percentual médio (<strong>MAPE</strong>) melhorou de <strong>${vParentMAPE}</strong> para <strong>${vChildMAPE}</strong>, indicando maior precisão percentual geral.`);
      score++;
    } else {
      insights.push(`O erro percentual médio (<strong>MAPE</strong>) subiu de <strong>${vParentMAPE}</strong> para <strong>${vChildMAPE}</strong>, revelando perda de acurácia no preço previsto.`);
      score--;
    }
  }

  if (Math.abs(diffRMSE) > 0.0001) {
    if (diffRMSE < 0) {
      insights.push(`O <strong>RMSE</strong> reduziu de <strong>${vParentRMSE}</strong> para <strong>${vChildRMSE}</strong>, indicando menor ocorrência de grandes erros (outliers/desvios de pico).`);
      score++;
    } else {
      insights.push(`O <strong>RMSE</strong> aumentou de <strong>${vParentRMSE}</strong> para <strong>${vChildRMSE}</strong>, apontando que o modelo cometeu desvios mais graves em dias atípicos.`);
      score--;
    }
  }

  if (Math.abs(diffMAE) > 0.0001) {
    if (diffMAE < 0) {
      insights.push(`O desvio absoluto médio (<strong>MAE</strong>) caiu de <strong>${vParentMAE}</strong> para <strong>${vChildMAE}</strong>, mostrando que a distância absoluta média em R$ diminuiu.`);
      score++;
    } else {
      insights.push(`O desvio absoluto médio (<strong>MAE</strong>) subiu de <strong>${vParentMAE}</strong> para <strong>${vChildMAE}</strong>, sinalizando desvio financeiro médio maior.`);
      score--;
    }
  }

  if (Math.abs(diffDIR) > 0.1) {
    if (diffDIR > 0) {
      insights.push(`A <strong>Acurácia Direcional</strong> aumentou de <strong>${vParentDIR}</strong> para <strong>${vChildDIR}</strong>, otimizando o acerto de tendência diária (alta/baixa).`);
      score++;
    } else {
      insights.push(`A <strong>Acurácia Direcional</strong> caiu de <strong>${vParentDIR}</strong> para <strong>${vChildDIR}</strong>, reduzindo a assertividade na previsão do sentido do movimento.`);
      score--;
    }
  }

  // 2. Parâmetros Mutados Insights
  const pParent = parent.params || {};
  const pChild = child.params || {};

  for (const [k, v] of Object.entries(pChild)) {
    const pValParent = pParent[k];
    if (pValParent !== undefined && pValParent !== v) {
      if (k === 'learning_rate') {
        const numP = parseFloat(pValParent);
        const numC = parseFloat(v);
        if (numC < numP) {
          insights.push(`A redução do <code>learning_rate</code> (${pValParent} ➔ ${v}) reduziu a velocidade do passo do gradiente, buscando evitar oscilações em torno do mínimo de erro.`);
        } else {
          insights.push(`O aumento do <code>learning_rate</code> (${pValParent} ➔ ${v}) acelerou a convergência, mas pode ter gerado instabilidade no mínimo global.`);
        }
      }
      if (k === 'hidden_size') {
        const numP = parseInt(pValParent, 10);
        const numC = parseInt(v, 10);
        if (numC > numP) {
          insights.push(`A ampliação do <code>hidden_size</code> (${pValParent} ➔ ${v}) aumentou a capacidade de memória interna da LSTM, embora aumente a complexidade e risco de overfitting.`);
        } else {
          insights.push(`A redução do <code>hidden_size</code> (${pValParent} ➔ ${v}) simplificou a capacidade da rede, atuando como um regularizador estrutural.`);
        }
      }
      if (k === 'window_size') {
        const numP = parseInt(pValParent, 10);
        const numC = parseInt(v, 10);
        if (numC > numP) {
          insights.push(`O aumento do lookback (<code>window_size</code>: ${pValParent} ➔ ${v}) deu mais contexto temporal à LSTM, mas expôs a rede neural a comportamentos de mercado mais antigos.`);
        } else {
          insights.push(`A diminuição do lookback (<code>window_size</code>: ${pValParent} ➔ ${v}) focou a LSTM em padrões recentes de preço, mas reduziu a memória de médio prazo.`);
        }
      }
      if (k === 'dropout') {
        const numP = parseFloat(pValParent);
        const numC = parseFloat(v);
        if (numC > numP) {
          insights.push(`O aumento do <code>dropout</code> (${pValParent} ➔ ${v}) impôs regularização mais agressiva para mitigar o overfitting dos dados de treino.`);
        } else {
          insights.push(`A redução do <code>dropout</code> (${pValParent} ➔ ${v}) permitiu maior ajuste das conexões, mas diminuiu a proteção contra memorização da série.`);
        }
      }
      if (k === 'target_mode') {
        insights.push(`A mudança de <code>target_mode</code> de <code>${pValParent}</code> para <code>${v}</code> modificou a escala do problema. Projeções sobre retornos costumam ser estacionárias e estatisticamente superiores a preços brutos.`);
      }
      if (k === 'feature_mode') {
        insights.push(`A mudança de <code>feature_mode</code> de <code>${pValParent}</code> para <code>${v}</code> alterou a dimensionalidade de entrada da rede.`);
      }
    }
  }

  let insightHtml = '';
  if (insights.length > 0) {
    insightHtml = '<ul style="margin: 0 0 10px 16px; padding: 0; list-style-type: disc;">' +
      insights.map(item => `<li style="margin-bottom: 6px; color: var(--text);">${item}</li>`).join('') +
      '</ul>';
  } else {
    insightHtml = '<p style="margin: 0 0 8px 0; color: var(--text);">A diferença entre os modelos foi muito pequena para gerar um insight significativo.</p>';
  }

  let conclusion = '';
  if (score >= 2) conclusion = "<span style='color: var(--success); font-weight: bold;'>🏆 Conclusão: A mutação foi um SUCESSO. A performance geral melhorou substancialmente!</span>";
  else if (score === 1) conclusion = "<span style='color: var(--success); font-weight: bold;'>✅ Conclusão: Saldo positivo leve. Vale a pena manter as alterações.</span>";
  else if (score === 0) conclusion = "<span style='color: var(--muted); font-weight: bold;'>⚖️ Conclusão: Empate técnico ou mudanças irrelevantes. Impacto não conclusivo.</span>";
  else if (score === -1) conclusion = "<span style='color: var(--danger); font-weight: bold;'>⚠️ Conclusão: A mutação prejudicou o modelo em métricas chaves.</span>";
  else conclusion = "<span style='color: var(--danger); font-weight: bold;'>❌ Conclusão: Regressão severa. Recomendado reverter hiperparâmetros.</span>";

  document.getElementById('lineage-modal-content').innerHTML = `
    <div style="margin-bottom: 16px; text-align: center; font-size: 13px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 6px;">
      <span style="color: var(--muted);">Modelo Base (Pai):</span> <code>${parentId.substring(0, 8)}</code> 
      <strong style="margin: 0 10px;">➔</strong> 
      <span style="color: var(--muted);">Modelo Derivado (Filho):</span> <code style="color: var(--accent-hover);">${childId.substring(0, 8)}</code>
    </div>
    
    <h4 style="margin: 0 0 10px 0; color: var(--text); border-bottom: 1px solid var(--panel-border); padding-bottom: 4px;">🛠️ O que mudou?</h4>
    <div style="margin-bottom: 20px; font-size: 13px;">${paramDiffs}</div>
 
    <h4 style="margin: 0 0 10px 0; color: var(--text); border-bottom: 1px solid var(--panel-border); padding-bottom: 4px;">📈 Impacto nas Métricas</h4>
    <div style="font-size: 13px; margin-bottom: 16px;">${metricsHtml}</div>
    
    <div style="background: rgba(59, 130, 246, 0.1); border-left: 3px solid var(--primary); padding: 10px 12px; font-size: 12px; line-height: 1.5; color: var(--text);">
      <strong style="display: block; margin-bottom: 6px; color: var(--accent-hover);">🧠 Análise da MLOps Engine:</strong>
      ${insightHtml}
      <div style="margin-top: 8px;">${conclusion}</div>
    </div>
  `;

  const modal = document.getElementById('lineage-modal');
  modal.classList.remove('hidden');
  modal.style.display = 'flex';
}

function openImageModal() {
  const srcImg = document.getElementById('mc-image').src;
  if (!srcImg) return;
  const modal = document.getElementById('image-modal');
  const modalImg = document.getElementById('modal-img-content');
  modalImg.src = srcImg;
  modal.classList.remove('hidden');
  modal.style.display = 'flex';
  setTimeout(() => {
    modal.style.opacity = '1';
  }, 10);
}

function closeImageModal() {
  const modal = document.getElementById('image-modal');
  modal.style.opacity = '0';
  setTimeout(() => {
    modal.classList.add('hidden');
    modal.style.display = 'none';
  }, 300);
}

async function initializeDashboard() {
  renderCalculatedFeatures();

  // Reseta qualquer run pai persistida pelo cache de formulário do navegador
  clearLineage();

  await syncChampionSelection({ updateInference: true, updateCard: true });

  let initialTab = localStorage.getItem('activeTab') || 'home';
  if (!ENABLE_TRAINING_API && initialTab === 'train') {
    initialTab = 'home';
  }
  showTab(initialTab);
}

initializeDashboard();

// Add simple pulse animation for the training button and pipeline steps
const style = document.createElement('style');
style.innerHTML = `
  @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
`;
document.head.appendChild(style);

// Atualiza automaticamente a telemetria quando qualquer botão do portal é clicado
document.addEventListener('click', (event) => {
  if (event.target.closest('button')) {
    setTimeout(loadTelemetry, 150);
  }
});
