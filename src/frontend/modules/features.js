export const FEATURE_REGISTRY = [
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

export const FEATURE_PRESETS = {
  returns_basic: ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5"],
  returns_trend: ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5", "Momentum_5", "Rolling_Return_5", "Rolling_Return_20", "SMA_7", "SMA_21", "MACD", "MACD_Signal", "MACD_Hist"],
  returns_volatility: ["Log_Return", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5", "Volatility_21", "BB_Width", "ATR_14", "Range_Pct"],
  technical_complete: ["Log_Return", "SMA_7", "SMA_21", "Volatility_21", "Momentum_5", "Range_Pct", "Volume_Z", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Width", "ATR_14", "Log_Return_Lag1", "Log_Return_Lag2", "Log_Return_Lag3", "Log_Return_Lag5", "Rolling_Return_5", "Rolling_Return_20", "Day_Of_Week", "Log_Volume"]
};

export function renderCalculatedFeatures() {
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

export function toggleGroup(groupName, checked) {
  FEATURE_REGISTRY.forEach(f => {
    if (f.group === groupName) {
      const cb = document.getElementById("feat-" + f.name);
      if (cb) cb.checked = checked;
    }
  });
  updateCalculatedFeatures(true);
}

export function toggleCalcFeaturesSection() {
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

export function onFeatureModeChange() {
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

export function applyPreset() {
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

export function updateCalculatedFeatures(triggerSuggestions = true) {
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
