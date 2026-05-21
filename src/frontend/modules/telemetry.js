import { state } from './state.js';

let localTelemetryHistory = [];

export async function loadTelemetry() {
  try {
    const response = await fetch('/telemetry');
    const data = await response.json();

    document.getElementById('tel-reqs').textContent = data.api.prediction_requests ?? 0;
    const errRate = data.api.total_requests > 0 ? (data.api.total_errors / data.api.total_requests * 100) : 0;
    const errEl = document.getElementById('tel-errs');
    errEl.textContent = errRate.toFixed(1) + '%';
    errEl.style.color = errRate > 0 ? 'var(--danger)' : 'var(--success)';

    document.getElementById('tel-uptime').textContent = data.uptime_seconds + 's';
    document.getElementById('tel-lat').textContent = data.inference.last_time_ms.toFixed(1) + ' ms';

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

    // Mantém a história localmente
    const snapshot = {
      timestamp: new Date().toLocaleTimeString('pt-BR'),
      cpu_percent: data.resources.cpu_percent,
      memory_mb: data.resources.process_memory_mb,
      predict_latency_ms: data.inference.last_time_ms,
      predict_multi_latency_ms: data.inference.last_time_ms // Usando o último registrado
    };
    
    localTelemetryHistory.push(snapshot);
    if (localTelemetryHistory.length > 100) {
      localTelemetryHistory.shift();
    }

    const history = localTelemetryHistory;
    const labels = history.map(h => h.timestamp);
    const cpuData = history.map(h => h.cpu_percent);
    const memData = history.map(h => h.memory_mb);
    const predictLatencyData = history.map(h => h.predict_latency_ms ?? null);
    const predictMultiLatencyData = history.map(h => h.predict_multi_latency_ms ?? null);

    if (state.resourceChart) state.resourceChart.destroy();
    state.resourceChart = new Chart(document.getElementById('resourceChart'), {
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

    if (state.latencyChart) state.latencyChart.destroy();
    state.latencyChart = new Chart(document.getElementById('latencyChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Predict Uni (ms)',
            data: predictLatencyData,
            borderColor: '#a78bfa',
            backgroundColor: 'rgba(167, 139, 250, 0.10)',
            fill: true,
            tension: 0.2,
            pointRadius: 1,
            borderWidth: 2,
            spanGaps: true
          },
          {
            label: 'Predict Multi (ms)',
            data: predictMultiLatencyData,
            borderColor: '#22d3ee',
            backgroundColor: 'rgba(34, 211, 238, 0.08)',
            fill: false,
            tension: 0.2,
            pointRadius: 1,
            borderWidth: 2,
            spanGaps: true
          }
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

export async function loadPrometheusMetrics() {
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
        const statusMatch = line.match(/(?:status|status_code|http_status|code)="(\d+xx|\d+)"/);
        if (statusMatch) {
          const statusStr = statusMatch[1];
          if (statusStr.startsWith('2')) status2xx += value;
          else if (statusStr.startsWith('4')) status4xx += value;
          else if (statusStr.startsWith('5')) status5xx += value;
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
