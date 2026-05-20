import { state } from './state.js';
import { money } from './utils.js';
import { loadTelemetry } from './telemetry.js';
import { syncChampionSelection } from './modelCard.js';

export function buildSinglePayload(requiredRows) {
  const count = Math.max(requiredRows, 65);
  const closes = Array.from({ length: count }, (_, i) => {
    const base = 43.0 + i * 0.035 + Math.sin(i / 4) * 0.45;
    return Number((base + Math.sin(i / 3) * 0.18).toFixed(2));
  });
  return JSON.stringify({ symbol: "PETR4.SA", closes }, null, 2);
}

export function buildMultiPayload(requiredRows) {
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

export async function changeInferenceModel(type) {
  state.currentInferenceModel = type;
  const btnSingle = document.getElementById('btn-inf-single');
  const btnMulti = document.getElementById('btn-inf-multi');
  const infoEl = document.getElementById('inference-model-info');
  const payloadTextarea = document.getElementById('payload');

  let windowSize = 60; 
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

export async function runPrediction() {
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
    const endpoint = state.currentInferenceModel === 'multi' ? '/predict/ohlcv' : '/predict';
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
    if (state.currentInferenceModel === 'multi') {
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

export function renderChart(closes, predicted) {
  const recent = closes.slice(-30).map(Number);
  const labels = recent.map((_, i) => String(i - recent.length + 1)).concat(['Previsão']);
  const closeSeries = recent.concat(null);
  const predictedSeries = new Array(recent.length).fill(null).concat(Number(predicted));
  const ctx = document.getElementById('priceChart');

  if (state.chart) state.chart.destroy();

  state.chart = new Chart(ctx, {
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
