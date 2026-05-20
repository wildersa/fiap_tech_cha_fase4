import { formatRunDate, numberOrNull } from './utils.js';
import { syncChampionSelection } from './modelCard.js';
import { FEATURE_PRESETS } from './features.js';

export function clearLineage() {
  const el = document.getElementById('train-parent-run-id');
  if (el) el.value = "";
  const badge = document.getElementById('lineage-badge');
  if (badge) badge.classList.add('hidden');
}

export function populateFormFromRunId(runId, buttonEl) {
  const run = window.allRunsData.find(x => x.run_id === runId);
  if (run && run.params) {
    populateTrainingForm(runId, run.params);
  }
  if (buttonEl) {
    buttonEl.textContent = "Preenchido!";
    setTimeout(() => buttonEl.textContent = "Preencher Formulário", 2000);
  }
}

export async function deleteRun(runId, buttonEl) {
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

export function populateTrainingForm(run_id, p) {
  const paramsForForm = { ...(p || {}) };
  const copiedEndDate = String(paramsForForm.end_date || '').trim().toLowerCase();
  if (!copiedEndDate || copiedEndDate === 'none' || copiedEndDate === 'null' || copiedEndDate === 'nan') {
    paramsForForm.end_date = paramsForForm.end_date_requested || '';
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

    if (window.updateCalculatedFeatures) {
      window.updateCalculatedFeatures(false);
    }
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

export async function loadRuns() {
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

export function runMetric(run, key) {
  const metrics = run.metrics || {};
  if (key === 'baseline_gain_pct') return numberOrNull(metrics.baseline_gain_pct ?? metrics.gain_mape_pct);
  if (key === 'directional_accuracy_pct') return numberOrNull(metrics.directional_accuracy_test_lstm_pct ?? metrics.directional_accuracy_pct);
  return numberOrNull(metrics[key]);
}

export function runRequiredRows(run) {
  const params = run.params || {};
  const windowSize = Number.parseInt(params.window_size || '60', 10);
  const targetMode = params.target_mode || 'log_returns';
  const featureMode = params.feature_mode || 'single';
  if (featureMode === 'single') {
    return windowSize + (targetMode === 'log_returns' || targetMode === 'returns' ? 1 : 0);
  }
  return windowSize + 21;
}

export function runGroup(run) {
  return (run.params || {}).feature_mode === 'single' ? 'single' : 'multi';
}

export function selectChampionRun(runs) {
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

export function championRunIds(runs) {
  const ids = new Map();
  for (const group of ['single', 'multi']) {
    const champion = selectChampionRun(runs.filter(run => runGroup(run) === group));
    if (champion) ids.set(champion.run_id, group);
  }
  return ids;
}

export function sortRuns(key) {
  const current = window.runsSort || { key: 'start_time', direction: 'desc' };
  window.runsSort = {
    key,
    direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc',
  };
  renderRunsTable();
}

export function runSortValue(run, key) {
  const params = run.params || {};
  if (key === 'target_mode') return params.target_mode || '';
  if (key === 'feature_mode') return params.feature_mode || '';
  if (key === 'run_id') return run.run_id || '';
  if (key === 'status') return run.status || '';
  if (key === 'start_time') return Number(run.start_time || 0);
  return runMetric(run, key);
}

export function runMatchesSearch(run, query) {
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

export function updateSortIndicators() {
  const sort = window.runsSort || { key: 'start_time', direction: 'desc' };
  for (const id of ['run_id', 'status', 'start_time', 'target_mode', 'test_lstm_mape_pct', 'test_baseline_mape_pct', 'baseline_gain_pct', 'test_lstm_rmse', 'test_lstm_mae', 'directional_accuracy_pct']) {
    const el = document.getElementById('sort-' + id);
    if (el) el.textContent = sort.key === id ? (sort.direction === 'asc' ? '↑' : '↓') : '';
  }
}

export function renderRunsTable() {
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
    const mape = mapeValue !== null ? mapeValue.toFixed(3) + '%' : '-';
    const baselineMape = baselineMapeValue !== null ? baselineMapeValue.toFixed(3) + '%' : '-';
    const gainMape = gainMapeValue !== null ? (gainMapeValue > 0 ? '+' : '') + gainMapeValue.toFixed(3) + '%' : '-';
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

export async function runTraining() {
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
      selectedFeatures = window.FEATURE_PRESETS ? window.FEATURE_PRESETS[preset] : [];
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

export function compareLineage(childId, parentId) {
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
