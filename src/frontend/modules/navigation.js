import { loadModelCard } from './modelCard.js';
import { loadTelemetry } from './telemetry.js';
import { loadRuns } from './training.js';

export function showTab(name) {
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
