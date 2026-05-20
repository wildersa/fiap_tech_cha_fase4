import { showTab } from './modules/navigation.js';
import { changeInferenceModel, runPrediction } from './modules/inference.js';
import { changeModelCardView, openImageModal, closeImageModal, syncChampionSelection } from './modules/modelCard.js';
import { loadTelemetry } from './modules/telemetry.js';
import { toggleGroup, toggleCalcFeaturesSection, applyPreset, onFeatureModeChange, updateCalculatedFeatures, renderCalculatedFeatures, FEATURE_PRESETS } from './modules/features.js';
import { clearLineage, populateFormFromRunId, deleteRun, loadRuns, sortRuns, renderRunsTable, runTraining, compareLineage } from './modules/training.js';

// Define globals needed by inline event handlers in index.html
window.showTab = showTab;
window.changeInferenceModel = changeInferenceModel;
window.runPrediction = runPrediction;
window.changeModelCardView = changeModelCardView;
window.openImageModal = openImageModal;
window.closeImageModal = closeImageModal;
window.loadTelemetry = loadTelemetry;

window.toggleGroup = toggleGroup;
window.toggleCalcFeaturesSection = toggleCalcFeaturesSection;
window.applyPreset = applyPreset;
window.onFeatureModeChange = onFeatureModeChange;
window.updateCalculatedFeatures = updateCalculatedFeatures;
window.FEATURE_PRESETS = FEATURE_PRESETS;

window.clearLineage = clearLineage;
window.populateFormFromRunId = populateFormFromRunId;
window.deleteRun = deleteRun;
window.loadRuns = loadRuns;
window.sortRuns = sortRuns;
window.renderRunsTable = renderRunsTable;
window.runTraining = runTraining;
window.compareLineage = compareLineage;

// Base Chart setup
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";

const ENABLE_TRAINING_API = Boolean(window.DASHBOARD_CONFIG?.enableTrainingApi);
if (!ENABLE_TRAINING_API) {
  const trainTab = document.getElementById('tab-train');
  if (trainTab) trainTab.style.display = 'none';
}

async function initializeDashboard() {
  renderCalculatedFeatures();
  clearLineage();
  await syncChampionSelection({ updateInference: true, updateCard: true });

  let initialTab = localStorage.getItem('activeTab') || 'home';
  if (!ENABLE_TRAINING_API && initialTab === 'train') {
    initialTab = 'home';
  }
  showTab(initialTab);
}

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

initializeDashboard();
