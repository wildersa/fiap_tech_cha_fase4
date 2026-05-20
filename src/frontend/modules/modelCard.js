import { state } from './state.js';
import { changeInferenceModel } from './inference.js';

export function setModelCardType(type) {
  state.currentModelCardType = type;

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

export async function syncChampionSelection({ updateInference = false, updateCard = true } = {}) {
  try {
    const res = await fetch('/model-champion');
    if (!res.ok) return null;

    const champion = await res.json();
    const selectedType = champion.has_champion ? champion.selected_model_type : null;
    if (!selectedType) return champion;

    setModelCardType(selectedType);

    if (updateInference && state.currentInferenceModel !== selectedType) {
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

export async function changeModelCardView(type) {
  setModelCardType(type);
  await loadModelCard();
}

export async function loadModelCard() {
  try {
    const res = await fetch(`/model-card?type=${state.currentModelCardType}`);
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
      imgEl.src = `/model-image?type=${state.currentModelCardType}&t=${new Date().getTime()}`;
    }

  } catch (e) {
    console.error('Falha ao carregar Model Card:', e);
  }
}

export function openImageModal() {
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

export function closeImageModal() {
  const modal = document.getElementById('image-modal');
  modal.style.opacity = '0';
  setTimeout(() => {
    modal.classList.add('hidden');
    modal.style.display = 'none';
  }, 300);
}
