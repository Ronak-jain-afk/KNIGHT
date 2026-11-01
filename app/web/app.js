// SPDX-License-Identifier: Apache-2.0
// ============================================================
// Knight Web Client - ES Module
// ============================================================

// State
let currentConversationId = null;
let currentAbortController = null;
let currentSources = [];
let autoScroll = true;
let loadedConversations = [];
let allLoadedMemories = [];
let activeMemoryKind = "all";
let lastUserPrompt = "";

// DOM Elements
const sidebar = document.getElementById("sidebar");
const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const searchConversationsInput = document.getElementById("search-conversations-input");
const conversationsList = document.getElementById("conversations-list");
const conversationTitle = document.getElementById("conversation-title");
const forkChatBtn = document.getElementById("fork-chat-btn");
const exportChatBtn = document.getElementById("export-chat-btn");
const openSettingsBtn = document.getElementById("open-settings-btn");
const openShortcutsBtn = document.getElementById("open-shortcuts-btn");
const modelBadge = document.getElementById("model-badge");
const healthIndicator = document.getElementById("health-indicator");
const healthBanner = document.getElementById("health-banner");
const retryHealthBtn = document.getElementById("retry-health-btn");
const messagesContainer = document.getElementById("messages-container");
const composerForm = document.getElementById("composer-form");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const stopBtn = document.getElementById("stop-btn");
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const scrollToBottomBtn = document.getElementById("scroll-to-bottom-btn");
const scrollUnreadBadge = document.getElementById("scroll-unread-badge");
const citationTooltip = document.getElementById("citation-preview-tooltip");
const citationTooltipIdx = document.getElementById("citation-tooltip-idx");
const citationTooltipDomain = document.getElementById("citation-tooltip-domain");
const citationTooltipTitle = document.getElementById("citation-tooltip-title");
const citationTooltipSnippet = document.getElementById("citation-tooltip-snippet");

// Feature 2: Model Switcher & Hardware HUD Elements
const modelSelect = document.getElementById("model-select");
const hardwareHudBtn = document.getElementById("hardware-hud-btn");
const hudGpuText = document.getElementById("hud-gpu-text");
const hudGpuDot = document.getElementById("hud-gpu-dot");
const hudRamText = document.getElementById("hud-ram-text");
const hudLatText = document.getElementById("hud-lat-text");

// Hardware Dialog Elements
const hardwareDialog = document.getElementById("hardware-dialog");
const closeHardwareBtn = document.getElementById("close-hardware-btn");
const refreshHardwareBtn = document.getElementById("refresh-hardware-btn");
const hwGpuStatus = document.getElementById("hw-gpu-status");
const hwGpuName = document.getElementById("hw-gpu-name");
const hwGpuVram = document.getElementById("hw-gpu-vram");
const hwGpuLoad = document.getElementById("hw-gpu-load");
const hwGpuTemp = document.getElementById("hw-gpu-temp");
const hwGpuProgress = document.getElementById("hw-gpu-progress");
const hwRamUsage = document.getElementById("hw-ram-usage");
const hwRamProgress = document.getElementById("hw-ram-progress");
const hwServerStatus = document.getElementById("hw-server-status");
const hwActiveModel = document.getElementById("hw-active-model");
const hwMaxCtx = document.getElementById("hw-max-ctx");
const hwServerLat = document.getElementById("hw-server-lat");
const hwServerSlots = document.getElementById("hw-server-slots");

// File Attachment Elements & State
let attachedFiles = [];
const SERVER_LIMITS = {
  max_message_chars: 32_000,
  max_attachment_bytes: 2_000_000,
};
const attachFileBtn = document.getElementById("attach-file-btn");
const fileInput = document.getElementById("file-input");
const attachmentsTray = document.getElementById("attachments-tray");

// Memories Dialog Elements
const memoriesDialog = document.getElementById("memories-dialog");
const openMemoriesBtn = document.getElementById("open-memories-btn");
const closeMemoriesBtn = document.getElementById("close-memories-btn");
const clearAllMemoriesBtn = document.getElementById("clear-all-memories-btn");
const addMemoryForm = document.getElementById("add-memory-form");
const newMemoryText = document.getElementById("new-memory-text");
const newMemoryKind = document.getElementById("new-memory-kind");
const newMemoryImportance = document.getElementById("new-memory-importance");
const memorySearchInput = document.getElementById("memory-search-input");
const memoryFilterPills = document.getElementById("memory-filter-pills");
const memoriesList = document.getElementById("memories-list");
const memoryFormFeedback = document.getElementById("memory-form-feedback");

// Edit Memory Dialog
const editMemoryDialog = document.getElementById("edit-memory-dialog");
const closeEditMemoryBtn = document.getElementById("close-edit-memory-btn");
const cancelEditMemoryBtn = document.getElementById("cancel-edit-memory-btn");
const editMemoryForm = document.getElementById("edit-memory-form");
const editMemoryId = document.getElementById("edit-memory-id");
const editMemoryText = document.getElementById("edit-memory-text");
const editMemoryImportance = document.getElementById("edit-memory-importance");

// Settings Dialog & Model Hyperparameters
const settingsDialog = document.getElementById("settings-dialog");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const cancelSettingsBtn = document.getElementById("cancel-settings-btn");
const settingsForm = document.getElementById("settings-form");
const customSystemPrompt = document.getElementById("custom-system-prompt");
const temperatureSlider = document.getElementById("temperature-slider");
const temperatureVal = document.getElementById("temperature-val");
const topPSlider = document.getElementById("top-p-slider");
const topPVal = document.getElementById("top-p-val");
const topKSlider = document.getElementById("top-k-slider");
const topKVal = document.getElementById("top-k-val");
const maxTokensSlider = document.getElementById("max-tokens-slider");
const maxTokensVal = document.getElementById("max-tokens-val");
const presencePenaltySlider = document.getElementById("presence-penalty-slider");
const presencePenaltyVal = document.getElementById("presence-penalty-val");
const frequencyPenaltySlider = document.getElementById("frequency-penalty-slider");
const frequencyPenaltyVal = document.getElementById("frequency-penalty-val");
const enableThinkingToggle = document.getElementById("enable-thinking-toggle");
const settingsSummaryBadge = document.getElementById("settings-summary-badge");
const settingsActiveModelName = document.getElementById("settings-active-model-name");
const presetsChipBar = document.getElementById("presets-chip-bar");
const resetSettingsBtn = document.getElementById("reset-settings-btn");
const settingsMemoryCountBadge = document.getElementById("settings-memory-count-badge");
const settingsOpenMemoriesBtn = document.getElementById("settings-open-memories-btn");
const settingsClearMemoriesBtn = document.getElementById("settings-clear-memories-btn");

// Shortcuts Dialog
const shortcutsDialog = document.getElementById("shortcuts-dialog");
const closeShortcutsBtn = document.getElementById("close-shortcuts-btn");

// Custom Confirmation Dialog Elements
const confirmDialog = document.getElementById("confirm-dialog");
const closeConfirmDialogBtn = document.getElementById("close-confirm-dialog-btn");
const cancelConfirmDialogBtn = document.getElementById("cancel-confirm-dialog-btn");
const confirmDialogActionBtn = document.getElementById("confirm-dialog-action-btn");
const confirmDialogTitle = document.getElementById("confirm-dialog-title");
const confirmDialogMessage = document.getElementById("confirm-dialog-message");
const confirmDialogItemBox = document.getElementById("confirm-dialog-item-box");
const confirmDialogItemName = document.getElementById("confirm-dialog-item-name");
const confirmDialogSubtext = document.getElementById("confirm-dialog-subtext");
const confirmIconWrap = document.getElementById("confirm-icon-wrap");
const confirmIconDanger = document.getElementById("confirm-icon-danger");
const confirmIconDefault = document.getElementById("confirm-icon-default");

// Custom Export Dialog Elements
const exportDialog = document.getElementById("export-dialog");
const closeExportDialogBtn = document.getElementById("close-export-dialog-btn");
const cancelExportDialogBtn = document.getElementById("cancel-export-dialog-btn");
const submitExportDialogBtn = document.getElementById("submit-export-dialog-btn");
const exportEmptyState = document.getElementById("export-empty-state");
const exportContentState = document.getElementById("export-content-state");
const exportDialogConvTitle = document.getElementById("export-dialog-conv-title");
const exportDialogTurnsBadge = document.getElementById("export-dialog-turns-badge");
const exportIncludeSourcesCheck = document.getElementById("export-include-sources-check");

// Voice ASR Microphone Elements
const micBtn = document.getElementById("mic-btn");
const voiceRecordingTray = document.getElementById("voice-recording-tray");
const voiceRecordingTimer = document.getElementById("voice-recording-timer");
const voiceRecordingStatus = document.getElementById("voice-recording-status");
const voiceCancelBtn = document.getElementById("voice-cancel-btn");
const settingsAsrBadge = document.getElementById("settings-asr-badge");
const settingsAsrDownloadBtn = document.getElementById("settings-asr-download-btn");
const settingsAsrLanguage = document.getElementById("settings-asr-language");
const settingsAsrAutoSend = document.getElementById("settings-asr-autosend");

// QoE Elements: Sound Effects, In-Chat Search & Slash Commands
const soundEffectsToggle = document.getElementById("sound-effects-toggle");
const inchatSearchToggleBtn = document.getElementById("inchat-search-toggle-btn");
const inchatSearchBar = document.getElementById("inchat-search-bar");
const inchatSearchInput = document.getElementById("inchat-search-input");
const inchatSearchCount = document.getElementById("inchat-search-count");
const inchatSearchPrevBtn = document.getElementById("inchat-search-prev-btn");
const inchatSearchNextBtn = document.getElementById("inchat-search-next-btn");
const inchatSearchCloseBtn = document.getElementById("inchat-search-close-btn");
const slashMenu = document.getElementById("slash-commands-menu");
const slashList = document.getElementById("slash-commands-list");

// Feature 4: Persona Profiles Elements & State
let loadedProfiles = [];
let activeProfileId = localStorage.getItem("knight-active-profile") || "general";

const profileSelect = document.getElementById("profile-select");
const openProfilesBtn = document.getElementById("open-profiles-btn");
const profilesDialog = document.getElementById("profiles-dialog");
const closeProfilesBtn = document.getElementById("close-profiles-btn");
const profilesGrid = document.getElementById("profiles-grid");
const newProfileBtn = document.getElementById("new-profile-btn");
const profileEditorForm = document.getElementById("profile-editor-form");
const profileEditorTitle = document.getElementById("profile-editor-title");
const profileEditId = document.getElementById("profile-edit-id");
const profileNameInput = document.getElementById("profile-name-input");
const profileIconInput = document.getElementById("profile-icon-input");
const profileModeInput = document.getElementById("profile-mode-input");
const profileDescInput = document.getElementById("profile-desc-input");
const profilePromptInput = document.getElementById("profile-prompt-input");
const cancelProfileEditBtn = document.getElementById("cancel-profile-edit-btn");

// Feature 5: GGUFBook Elements & State
let currentGgufbookRepo = null;
let downloadPollInterval = null;
let activeDownloads = [];
const previousCompletedTasks = new Set();

const openGgufbookBtn = document.getElementById("open-ggufbook-btn");
const ggufbookDialog = document.getElementById("ggufbook-dialog");
const closeGgufbookBtn = document.getElementById("close-ggufbook-btn");
const ggufbookRigVram = document.getElementById("ggufbook-rig-vram");
const ggufbookRigRam = document.getElementById("ggufbook-rig-ram");
const ggufbookSearchInput = document.getElementById("ggufbook-search-input");
const ggufbookSearchBtn = document.getElementById("ggufbook-search-btn");
const ggufbookFilterChips = document.getElementById("ggufbook-filter-chips");
const ggufbookReposView = document.getElementById("ggufbook-repos-view");
const ggufbookReposContainer = document.getElementById("ggufbook-repos-container");
const ggufbookVariantsView = document.getElementById("ggufbook-variants-view");
const backToReposBtn = document.getElementById("back-to-repos-btn");
const variantRepoTitle = document.getElementById("variant-repo-title");
const variantRepoMeta = document.getElementById("variant-repo-meta");
const ggufbookVariantsList = document.getElementById("ggufbook-variants-list");
const ggufbookDownloadsDrawer = document.getElementById("ggufbook-downloads-drawer");
const ggufbookDownloadsCount = document.getElementById("ggufbook-downloads-count");
const ggufbookDownloadsList = document.getElementById("ggufbook-downloads-list");

// ============================================================
// Initialization & Theme Handling
// ============================================================

function initTheme() {
  const saved = localStorage.getItem("knight-theme") || "dark";
  applyTheme(saved);
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("knight-theme", theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
}

// ============================================================
// ============================================================
// Model & Session Settings Studio
// ============================================================

const SAMPLING_PRESETS = {
  balanced: {
    name: "Balanced",
    temperature: 0.70,
    top_p: 0.90,
    top_k: 40,
    max_output_tokens: 4096,
    presence_penalty: 0.00,
    frequency_penalty: 0.00,
    enable_thinking: false,
  },
  precise: {
    name: "Precise / Code",
    temperature: 0.20,
    top_p: 0.80,
    top_k: 20,
    max_output_tokens: 4096,
    presence_penalty: 0.00,
    frequency_penalty: 0.00,
    enable_thinking: false,
  },
  creative: {
    name: "Creative",
    temperature: 0.95,
    top_p: 0.95,
    top_k: 50,
    max_output_tokens: 4096,
    presence_penalty: 0.30,
    frequency_penalty: 0.20,
    enable_thinking: false,
  },
  thinking: {
    name: "Deep Thinker",
    temperature: 0.60,
    top_p: 0.95,
    top_k: 40,
    max_output_tokens: 4096,
    presence_penalty: 0.00,
    frequency_penalty: 0.00,
    enable_thinking: true,
  },
  fast: {
    name: "Fast & Concise",
    temperature: 0.50,
    top_p: 0.85,
    top_k: 30,
    max_output_tokens: 1024,
    presence_penalty: 0.00,
    frequency_penalty: 0.00,
    enable_thinking: false,
  },
};

const TONE_SNIPPETS = {
  "Concise": "Be concise, direct, and omit fluff or unnecessary filler.",
  "Code-First": "Respond code-first with copy-pasteable snippets and clean explanations.",
  "Step-by-Step": "Explain systematically step-by-step with clear numbered points.",
  "Socratic Tutor": "Act as an interactive tutor, asking clarifying questions and guiding me through problems.",
};

function getActiveModelCustomization() {
  const temp = parseFloat(localStorage.getItem("knight-temperature") ?? "0.7");
  const topP = parseFloat(localStorage.getItem("knight-top-p") ?? "0.9");
  const topK = parseInt(localStorage.getItem("knight-top-k") ?? "40", 10);
  const maxTokens = parseInt(localStorage.getItem("knight-max-tokens") ?? "4096", 10);
  const presencePenalty = parseFloat(localStorage.getItem("knight-presence-penalty") ?? "0.0");
  const frequencyPenalty = parseFloat(localStorage.getItem("knight-frequency-penalty") ?? "0.0");
  const enableThinking = localStorage.getItem("knight-enable-thinking") === "true";

  return {
    temperature: isNaN(temp) ? 0.7 : temp,
    top_p: isNaN(topP) ? 0.9 : topP,
    top_k: isNaN(topK) ? 40 : topK,
    max_output_tokens: isNaN(maxTokens) ? 4096 : maxTokens,
    presence_penalty: isNaN(presencePenalty) ? 0.0 : presencePenalty,
    frequency_penalty: isNaN(frequencyPenalty) ? 0.0 : frequencyPenalty,
    enable_thinking: enableThinking,
  };
}

function updateCustomPresetState() {
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.classList.remove("active");
  });
  if (settingsSummaryBadge) {
    settingsSummaryBadge.textContent = "Preset: Custom";
  }
  localStorage.setItem("knight-active-preset", "custom");
}

function applyPreset(key, syncUiOnly = false) {
  const preset = SAMPLING_PRESETS[key];
  if (!preset) return;

  if (temperatureSlider) temperatureSlider.value = preset.temperature;
  if (temperatureVal) temperatureVal.textContent = Number(preset.temperature).toFixed(2);

  if (topPSlider) topPSlider.value = preset.top_p;
  if (topPVal) topPVal.textContent = Number(preset.top_p).toFixed(2);

  if (topKSlider) topKSlider.value = preset.top_k;
  if (topKVal) topKVal.textContent = preset.top_k;

  if (maxTokensSlider) maxTokensSlider.value = preset.max_output_tokens;
  if (maxTokensVal) maxTokensVal.textContent = preset.max_output_tokens;

  if (presencePenaltySlider) presencePenaltySlider.value = preset.presence_penalty;
  if (presencePenaltyVal) presencePenaltyVal.textContent = Number(preset.presence_penalty).toFixed(2);

  if (frequencyPenaltySlider) frequencyPenaltySlider.value = preset.frequency_penalty;
  if (frequencyPenaltyVal) frequencyPenaltyVal.textContent = Number(preset.frequency_penalty).toFixed(2);

  if (enableThinkingToggle) enableThinkingToggle.checked = !!preset.enable_thinking;

  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.preset === key);
  });

  if (settingsSummaryBadge) {
    settingsSummaryBadge.textContent = `Preset: ${preset.name}`;
  }

  if (!syncUiOnly) {
    localStorage.setItem("knight-active-preset", key);
    localStorage.setItem("knight-temperature", preset.temperature);
    localStorage.setItem("knight-top-p", preset.top_p);
    localStorage.setItem("knight-top-k", preset.top_k);
    localStorage.setItem("knight-max-tokens", preset.max_output_tokens);
    localStorage.setItem("knight-presence-penalty", preset.presence_penalty);
    localStorage.setItem("knight-frequency-penalty", preset.frequency_penalty);
    localStorage.setItem("knight-enable-thinking", String(preset.enable_thinking));
  }
}

function updateToneChipsActive() {
  const current = customSystemPrompt ? customSystemPrompt.value : "";
  document.querySelectorAll(".tone-chip").forEach((chip) => {
    const tone = chip.dataset.tone;
    const snippet = TONE_SNIPPETS[tone];
    chip.classList.toggle("active", Boolean(snippet && current.includes(snippet)));
  });
}

function initSettings() {
  const savedPrompt = localStorage.getItem("knight-custom-instructions") || "";
  if (customSystemPrompt) customSystemPrompt.value = savedPrompt;
  updateToneChipsActive();

  const activePreset = localStorage.getItem("knight-active-preset") || "balanced";

  const savedTemp = localStorage.getItem("knight-temperature");
  const savedTopP = localStorage.getItem("knight-top-p");
  const savedTopK = localStorage.getItem("knight-top-k");
  const savedMaxTokens = localStorage.getItem("knight-max-tokens");
  const savedPresence = localStorage.getItem("knight-presence-penalty");
  const savedFreq = localStorage.getItem("knight-frequency-penalty");
  const savedThinking = localStorage.getItem("knight-enable-thinking");

  if (activePreset !== "custom" && SAMPLING_PRESETS[activePreset] && savedTemp === null) {
    applyPreset(activePreset, true);
  } else {
    if (temperatureSlider) {
      temperatureSlider.value = savedTemp !== null ? savedTemp : "0.70";
      if (temperatureVal) temperatureVal.textContent = Number(temperatureSlider.value).toFixed(2);
    }
    if (topPSlider) {
      topPSlider.value = savedTopP !== null ? savedTopP : "0.90";
      if (topPVal) topPVal.textContent = Number(topPSlider.value).toFixed(2);
    }
    if (topKSlider) {
      topKSlider.value = savedTopK !== null ? savedTopK : "40";
      if (topKVal) topKVal.textContent = topKSlider.value;
    }
    if (maxTokensSlider) {
      maxTokensSlider.value = savedMaxTokens !== null ? savedMaxTokens : "4096";
      if (maxTokensVal) maxTokensVal.textContent = maxTokensSlider.value;
    }
    if (presencePenaltySlider) {
      presencePenaltySlider.value = savedPresence !== null ? savedPresence : "0.00";
      if (presencePenaltyVal) presencePenaltyVal.textContent = Number(presencePenaltySlider.value).toFixed(2);
    }
    if (frequencyPenaltySlider) {
      frequencyPenaltySlider.value = savedFreq !== null ? savedFreq : "0.00";
      if (frequencyPenaltyVal) frequencyPenaltyVal.textContent = Number(frequencyPenaltySlider.value).toFixed(2);
    }
    if (enableThinkingToggle) {
      enableThinkingToggle.checked = savedThinking === "true";
    }

    if (activePreset === "custom") {
      updateCustomPresetState();
    } else if (SAMPLING_PRESETS[activePreset]) {
      document.querySelectorAll(".preset-chip").forEach((chip) => {
        chip.classList.toggle("active", chip.dataset.preset === activePreset);
      });
      if (settingsSummaryBadge) {
        settingsSummaryBadge.textContent = `Preset: ${SAMPLING_PRESETS[activePreset].name}`;
      }
    }
  }

  // Update active model name in settings banner
  if (settingsActiveModelName) {
    const modelSelector = document.getElementById("model-select");
    const activeText = modelSelector && modelSelector.options && modelSelector.selectedIndex >= 0
      ? modelSelector.options[modelSelector.selectedIndex].text
      : "Knight Local Model";
    settingsActiveModelName.textContent = activeText || "Knight Local Model";
  }

  // Update ASR status and options in settings
  if (settingsAsrLanguage) {
    settingsAsrLanguage.value = localStorage.getItem("knight-asr-language") || "";
  }
  if (settingsAsrAutoSend) {
    settingsAsrAutoSend.checked = localStorage.getItem("knight-asr-autosend") === "true";
  }
  if (soundEffectsToggle) {
    soundEffectsToggle.checked = localStorage.getItem("knight-sound-effects") !== "false";
  }
  checkASRStatus();
  updateSettingsMemoryCount();
}

function openSettings() {
  initSettings();
  const modalBody = settingsDialog.querySelector(".settings-modal-body");
  if (modalBody) modalBody.scrollTop = 0;
  settingsDialog.showModal();
}

function closeSettings() {
  settingsDialog.close();
}

settingsForm.addEventListener("submit", (e) => {
  e.preventDefault();
  localStorage.setItem("knight-custom-instructions", customSystemPrompt.value.trim());
  if (temperatureSlider) localStorage.setItem("knight-temperature", temperatureSlider.value);
  if (topPSlider) localStorage.setItem("knight-top-p", topPSlider.value);
  if (topKSlider) localStorage.setItem("knight-top-k", topKSlider.value);
  if (maxTokensSlider) localStorage.setItem("knight-max-tokens", maxTokensSlider.value);
  if (presencePenaltySlider) localStorage.setItem("knight-presence-penalty", presencePenaltySlider.value);
  if (frequencyPenaltySlider) localStorage.setItem("knight-frequency-penalty", frequencyPenaltySlider.value);
  if (enableThinkingToggle) localStorage.setItem("knight-enable-thinking", String(enableThinkingToggle.checked));
  if (settingsAsrLanguage) localStorage.setItem("knight-asr-language", settingsAsrLanguage.value);
  if (settingsAsrAutoSend) localStorage.setItem("knight-asr-autosend", String(settingsAsrAutoSend.checked));
  if (soundEffectsToggle) localStorage.setItem("knight-sound-effects", String(soundEffectsToggle.checked));
  closeSettings();
});

if (cancelSettingsBtn) {
  cancelSettingsBtn.addEventListener("click", () => {
    closeSettings();
  });
}

// Sliders live-value and custom state tracking
[
  [temperatureSlider, temperatureVal, 2],
  [topPSlider, topPVal, 2],
  [topKSlider, topKVal, 0],
  [maxTokensSlider, maxTokensVal, 0],
  [presencePenaltySlider, presencePenaltyVal, 2],
  [frequencyPenaltySlider, frequencyPenaltyVal, 2],
].forEach(([slider, valEl, decimals]) => {
  if (!slider || !valEl) return;
  slider.addEventListener("input", (e) => {
    valEl.textContent = decimals > 0 ? Number(e.target.value).toFixed(decimals) : e.target.value;
    updateCustomPresetState();
  });
});

if (enableThinkingToggle) {
  enableThinkingToggle.addEventListener("change", () => {
    updateCustomPresetState();
  });
}

// Preset chips click listener
if (presetsChipBar) {
  presetsChipBar.addEventListener("click", (e) => {
    const chip = e.target.closest(".preset-chip");
    if (!chip) return;
    const presetKey = chip.dataset.preset;
    if (presetKey && SAMPLING_PRESETS[presetKey]) {
      applyPreset(presetKey);
    }
  });
}

// Tone modifier chips click listener
document.querySelectorAll(".tone-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const tone = chip.dataset.tone;
    const snippet = TONE_SNIPPETS[tone];
    if (!snippet || !customSystemPrompt) return;

    let current = customSystemPrompt.value.trim();
    if (current.includes(snippet)) {
      current = current.replace(snippet, "").trim();
      chip.classList.remove("active");
    } else {
      current = current ? `${current}\n${snippet}` : snippet;
      chip.classList.add("active");
    }
    customSystemPrompt.value = current;
  });
});

if (customSystemPrompt) {
  customSystemPrompt.addEventListener("input", () => {
    updateToneChipsActive();
  });
}

resetSettingsBtn.addEventListener("click", () => {
  if (customSystemPrompt) customSystemPrompt.value = "";
  localStorage.removeItem("knight-custom-instructions");
  localStorage.removeItem("knight-temperature");
  localStorage.removeItem("knight-top-p");
  localStorage.removeItem("knight-top-k");
  localStorage.removeItem("knight-max-tokens");
  localStorage.removeItem("knight-presence-penalty");
  localStorage.removeItem("knight-frequency-penalty");
  localStorage.removeItem("knight-enable-thinking");
  localStorage.removeItem("knight-active-preset");
  localStorage.removeItem("knight-asr-language");
  localStorage.removeItem("knight-asr-autosend");
  localStorage.removeItem("knight-sound-effects");
  if (settingsAsrLanguage) settingsAsrLanguage.value = "";
  if (settingsAsrAutoSend) settingsAsrAutoSend.checked = false;
  if (soundEffectsToggle) soundEffectsToggle.checked = true;
  applyPreset("balanced");
  updateToneChipsActive();
});

// ============================================================
// Synthetic Web Audio Soundscape (0 External Audio Assets)
// ============================================================

class SoundEffects {
  constructor() {
    this.ctx = null;
  }

  isEnabled() {
    return localStorage.getItem("knight-sound-effects") !== "false";
  }

  _getCtx() {
    if (!this.isEnabled()) return null;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!this.ctx || this.ctx.state === "closed") {
        this.ctx = new AudioCtx();
      }
      if (this.ctx.state === "suspended") {
        this.ctx.resume();
      }
      return this.ctx;
    } catch (_) {
      return null;
    }
  }

  playSendSound() {
    const ctx = this._getCtx();
    if (!ctx) return;
    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const now = ctx.currentTime;
      osc.type = "sine";
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(880, now + 0.08);
      gain.gain.setValueAtTime(0.04, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.08);
    } catch (_) {}
  }

  playDoneSound() {
    const ctx = this._getCtx();
    if (!ctx) return;
    try {
      const now = ctx.currentTime;
      [587.33, 880.0].forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, now + idx * 0.04);
        gain.gain.setValueAtTime(0.03, now + idx * 0.04);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.04 + 0.22);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now + idx * 0.04);
        osc.stop(now + idx * 0.04 + 0.22);
      });
    } catch (_) {}
  }

  playMicToggleSound() {
    const ctx = this._getCtx();
    if (!ctx) return;
    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const now = ctx.currentTime;
      osc.type = "triangle";
      osc.frequency.setValueAtTime(320, now);
      osc.frequency.exponentialRampToValueAtTime(160, now + 0.05);
      gain.gain.setValueAtTime(0.03, now);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.05);
    } catch (_) {}
  }
}

const soundEffects = new SoundEffects();

// ============================================================
// Auth: bearer token for /api/* (KNIGHT_AUTH_TOKEN)
// ============================================================

const AUTH_TOKEN_KEY = "knight-auth-token";

function getAuthToken() {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY) || "";
  } catch (_) {
    return "";
  }
}

function setAuthToken(token) {
  try {
    if (token) {
      localStorage.setItem(AUTH_TOKEN_KEY, token);
    } else {
      localStorage.removeItem(AUTH_TOKEN_KEY);
    }
  } catch (_) {
    /* storage may be unavailable (private mode) */
  }
}

function promptForAuthToken() {
  const entered = window.prompt(
    "Knight requires an access token (see KNIGHT_AUTH_TOKEN). Paste it here:",
    ""
  );
  if (entered && entered.trim()) {
    setAuthToken(entered.trim());
    return true;
  }
  return false;
}

// Allow opening the UI once with ?token=...; store it and strip it from the URL.
(function bootstrapAuthTokenFromUrl() {
  try {
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get("token");
    if (!urlToken) return;
    setAuthToken(urlToken.trim());
    params.delete("token");
    const query = params.toString();
    const cleanUrl =
      window.location.pathname + (query ? `?${query}` : "") + window.location.hash;
    window.history.replaceState({}, document.title, cleanUrl);
  } catch (_) {
    /* ignore malformed URLs */
  }
})();

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await window.fetch(url, { ...options, headers });
  if (response.status !== 401) {
    return response;
  }

  // A body stream/FormData cannot be replayed; prompt once and let the caller retry.
  const replayable =
    options.body === undefined || options.body === null || typeof options.body === "string";
  if (!replayable || !promptForAuthToken()) {
    return response;
  }

  headers.set("Authorization", `Bearer ${getAuthToken()}`);
  return window.fetch(url, { ...options, headers });
}


// ============================================================
// Health & Config Check
// ============================================================

async function checkHealth() {
  try {
    const res = await apiFetch("/api/health");
    if (!res.ok) throw new Error(`Health status ${res.status}`);
    const data = await res.json();

    const dot = healthIndicator.querySelector(".health-dot");
    if (data.llama && data.llama.reachable) {
      dot.className = "health-dot health-ok";
      dot.title = `Model: ${data.llama.model || "ready"} (ctx: ${data.llama.n_ctx || "16k"})`;
      healthBanner.classList.add("hidden");
    } else {
      dot.className = "health-dot health-down";
      dot.title = "llama-server unreachable";
      healthBanner.classList.remove("hidden");
    }
  } catch (err) {
    const dot = healthIndicator.querySelector(".health-dot");
    dot.className = "health-dot health-down";
    healthBanner.classList.remove("hidden");
  }
}

async function loadConfig() {
  try {
    const res = await apiFetch("/api/config");
    if (res.ok) {
      const data = await res.json();
      if (data.model_name && modelBadge) {
        modelBadge.textContent = data.model_name;
      }
      if (Number.isFinite(data.max_message_chars)) {
        SERVER_LIMITS.max_message_chars = data.max_message_chars;
      }
      if (Number.isFinite(data.max_attachment_bytes)) {
        SERVER_LIMITS.max_attachment_bytes = data.max_attachment_bytes;
      }
    }
  } catch (err) {
    console.debug("Config fetch error:", err);
  }
}

async function checkASRStatus() {
  if (!settingsAsrBadge) return;
  try {
    const res = await apiFetch("/api/asr/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.enabled) {
      settingsAsrBadge.textContent = "Disabled in Config";
      settingsAsrBadge.className = "settings-card-badge status-unavailable";
      if (settingsAsrDownloadBtn) settingsAsrDownloadBtn.classList.add("hidden");
      if (micBtn) micBtn.classList.add("hidden");
      return;
    }

    if (data.model_downloaded) {
      settingsAsrBadge.textContent = "Whistle Ready";
      settingsAsrBadge.className = "settings-card-badge status-ready";
      if (settingsAsrDownloadBtn) settingsAsrDownloadBtn.classList.add("hidden");
      if (micBtn) {
        micBtn.classList.remove("hidden");
        micBtn.title = "Voice Input (Click to speak)";
      }
    } else {
      settingsAsrBadge.textContent = "Model Not Downloaded";
      settingsAsrBadge.className = "settings-card-badge status-downloading";
      if (settingsAsrDownloadBtn) {
        settingsAsrDownloadBtn.classList.remove("hidden");
        settingsAsrDownloadBtn.textContent = "Download Whistle (16.9 MB)";
        settingsAsrDownloadBtn.disabled = false;
      }
      if (micBtn) {
        micBtn.classList.remove("hidden");
        micBtn.title = "Voice Input (Whistle model download required in Settings)";
      }
    }
  } catch (err) {
    console.debug("ASR status check failed:", err);
    settingsAsrBadge.textContent = "Engine Offline";
    settingsAsrBadge.className = "settings-card-badge status-unavailable";
    if (settingsAsrDownloadBtn) settingsAsrDownloadBtn.classList.add("hidden");
  }
}

if (settingsAsrDownloadBtn) {
  settingsAsrDownloadBtn.addEventListener("click", async () => {
    settingsAsrDownloadBtn.disabled = true;
    settingsAsrDownloadBtn.textContent = "Downloading Whistle...";
    try {
      const res = await apiFetch("/api/asr/download", { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Download failed");
      }
      showToast("Whistle model downloaded successfully!");
      checkASRStatus();
    } catch (err) {
      showToast(`Whistle download failed: ${err.message}`, true);
      settingsAsrDownloadBtn.disabled = false;
      settingsAsrDownloadBtn.textContent = "Download Whistle (16.9 MB)";
    }
  });
}

let activeModelName = "";

function showToast(message, isError = false) {
  let toast = document.getElementById("toast-notification");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast-notification";
    toast.className = "toast-notification";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.toggle("toast-error", isError);
  toast.classList.add("show");
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 4000);
}

async function loadModels() {
  if (!modelSelect) return;
  try {
    const res = await apiFetch("/api/models");
    if (!res.ok) return;
    const data = await res.json();
    activeModelName = data.active_model || "";

    modelSelect.innerHTML = "";
    if (!data.models || data.models.length === 0) {
      const opt = document.createElement("option");
      opt.value = activeModelName || "qwen3.5-4b";
      opt.textContent = activeModelName || "Default model";
      modelSelect.appendChild(opt);
    } else {
      let hasSelected = false;
      for (const m of data.models) {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = `${m.stem} (${m.size_formatted})`;
        if (m.is_active || m.name === activeModelName || m.stem === activeModelName) {
          opt.selected = true;
          activeModelName = m.name;
          hasSelected = true;
        }
        modelSelect.appendChild(opt);
      }
      if (!hasSelected && data.models.length > 0) {
        modelSelect.selectedIndex = 0;
        activeModelName = data.models[0].name;
      }
    }
    if (modelBadge) {
      modelBadge.textContent = activeModelName || "qwen3.5-4b";
    }
    if (settingsActiveModelName) {
      const activeText = modelSelect && modelSelect.options && modelSelect.selectedIndex >= 0
        ? modelSelect.options[modelSelect.selectedIndex].text
        : activeModelName || "Knight Local Model";
      settingsActiveModelName.textContent = activeText;
    }
  } catch (err) {
    console.debug("Failed to load models list:", err);
  }
}

async function handleModelSwitch(newModelName) {
  if (!newModelName || newModelName === activeModelName) return;

  const confirmed = await showConfirmDialog({
    title: "Switch Active Model",
    message: "The inference server will restart with the new model weights.",
    itemTitle: newModelName,
    confirmText: "Switch Model",
    cancelText: "Cancel",
    subtext: "Inference server restart may take up to 30 seconds.",
    isDanger: false,
  });
  if (!confirmed) {
    await loadModels();
    return;
  }

  modelSelect.disabled = true;
  showToast(`Switching to ${newModelName}... this may take up to 30s.`);

  try {
    const res = await apiFetch("/api/models/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: newModelName }),
    });

    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Failed to switch model", true);
      alert(`Model Switch Error:\n${data.detail || "Unknown error"}`);
      await loadModels();
    } else {
      showToast(`Active model changed to: ${data.model_name}`);
      await loadModels();
      await fetchHardwareTelemetry();
      await checkHealth();
    }
  } catch (err) {
    showToast(`Network error switching model: ${err.message}`, true);
    await loadModels();
  } finally {
    modelSelect.disabled = false;
  }
}

async function fetchHardwareTelemetry() {
  try {
    const res = await apiFetch("/api/system/hardware");
    if (!res.ok) return;
    const data = await res.json();

    const { gpu, ram, context, server } = data;

    // GPU HUD & Modal
    if (gpu && gpu.available) {
      if (hudGpuText) hudGpuText.textContent = `GPU: ${gpu.vram_percent}%`;
      if (hudGpuDot) {
        hudGpuDot.className = "hud-dot gpu-active";
        hudGpuDot.title = `VRAM: ${gpu.vram_used_mb}/${gpu.vram_total_mb} MB (${gpu.temperature_c}°C)`;
      }

      if (hwGpuStatus) {
        hwGpuStatus.textContent = "Active";
        hwGpuStatus.className = "badge";
        hwGpuName.textContent = gpu.name || "NVIDIA GPU";
        hwGpuVram.textContent = `${gpu.vram_used_mb} / ${gpu.vram_total_mb} MB (${gpu.vram_percent}%)`;
        hwGpuLoad.textContent = `${gpu.utilization_percent}%`;
        hwGpuTemp.textContent = `${gpu.temperature_c}°C`;
        hwGpuProgress.style.width = `${Math.min(gpu.vram_percent, 100)}%`;
        hwGpuProgress.classList.toggle("warn", gpu.vram_percent > 85);
      }
    } else {
      if (hudGpuText) hudGpuText.textContent = "GPU: --";
      if (hudGpuDot) {
        hudGpuDot.className = "hud-dot";
        hudGpuDot.title = "No GPU query returned";
      }

      if (hwGpuStatus) {
        hwGpuStatus.textContent = "Inactive / Host Only";
        hwGpuStatus.className = "badge";
        hwGpuName.textContent = "No dedicated NVIDIA GPU detected";
        hwGpuVram.textContent = "N/A";
        hwGpuLoad.textContent = "N/A";
        hwGpuTemp.textContent = "N/A";
        hwGpuProgress.style.width = "0%";
      }
    }

    // RAM HUD & Modal
    if (ram && ram.available) {
      if (hudRamText) hudRamText.textContent = `RAM: ${ram.percent}%`;
      if (hwRamUsage) {
        hwRamUsage.textContent = `${ram.used_mb} / ${ram.total_mb} MB (${ram.percent}%)`;
        hwRamProgress.style.width = `${Math.min(ram.percent, 100)}%`;
        hwRamProgress.classList.toggle("warn", ram.percent > 85);
      }
    } else {
      if (hudRamText) hudRamText.textContent = "RAM: --";
      if (hwRamUsage) {
        hwRamUsage.textContent = "Unavailable";
        hwRamProgress.style.width = "0%";
      }
    }

    // Context & Server
    if (context) {
      if (hwActiveModel) hwActiveModel.textContent = context.model_name || "--";
      if (hwMaxCtx) hwMaxCtx.textContent = `${Number(context.max_tokens).toLocaleString()} tokens`;
    }

    if (server && server.reachable) {
      if (hudLatText) hudLatText.textContent = `${server.latency_ms}ms`;
      if (hwServerStatus) {
        hwServerStatus.textContent = "Online";
        hwServerLat.textContent = `${server.latency_ms} ms`;
        hwServerSlots.textContent = `${server.slots_busy || 0} busy`;
      }
    } else {
      if (hudLatText) hudLatText.textContent = "Offline";
      if (hwServerStatus) {
        hwServerStatus.textContent = "Offline";
        hwServerLat.textContent = "--";
        hwServerSlots.textContent = "0";
      }
    }

    // Sidebar Rig Card
    const sidebarRigTitle = document.getElementById("sidebar-rig-title");
    const sidebarRigSub = document.getElementById("sidebar-rig-sub");
    if (sidebarRigTitle && sidebarRigSub) {
      if (gpu && gpu.available) {
        sidebarRigTitle.textContent = gpu.name || "NVIDIA GPU";
        sidebarRigSub.textContent = `VRAM: ${gpu.vram_used_mb} / ${gpu.vram_total_mb} MB (${gpu.vram_percent}%)`;
      } else if (ram && ram.available) {
        sidebarRigTitle.textContent = "Host Inference";
        sidebarRigSub.textContent = `RAM: ${ram.used_mb} / ${ram.total_mb} MB (${ram.percent}%)`;
      }
    }
  } catch (err) {
    console.debug("Hardware telemetry error:", err);
  }
}

// ============================================================
// Feature 4: Persona Profiles Management
// ============================================================

async function loadProfiles() {
  try {
    const res = await apiFetch("/api/profiles");
    if (!res.ok) return;
    loadedProfiles = await res.json();

    if (profileSelect) {
      profileSelect.innerHTML = "";
      for (const p of loadedProfiles) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.name;
        if (p.id === activeProfileId) {
          opt.selected = true;
        }
        profileSelect.appendChild(opt);
      }
    }
  } catch (err) {
    console.debug("Failed to load profiles:", err);
  }
}

function selectProfile(profileId) {
  const prof = loadedProfiles.find((p) => p.id === profileId);
  if (!prof) return;

  activeProfileId = profileId;
  localStorage.setItem("knight-active-profile", profileId);

  if (profileSelect) {
    profileSelect.value = profileId;
  }

  // Set preferred mode radio button
  if (prof.mode) {
    const modeRadio = document.querySelector(`input[name="chat-mode"][value="${prof.mode}"]`);
    if (modeRadio) {
      modeRadio.checked = true;
    }
  }

  showToast(`Persona: ${prof.name} active`);
  renderProfilesGrid();
}

function renderProfilesGrid() {
  if (!profilesGrid) return;
  profilesGrid.innerHTML = "";

  for (const p of loadedProfiles) {
    const card = document.createElement("div");
    card.className = `profile-card ${p.id === activeProfileId ? "active-profile" : ""}`;

    const header = document.createElement("div");
    header.className = "profile-card-header";

    const titleGroup = document.createElement("div");
    titleGroup.className = "profile-card-title-group";

    const nameSpan = document.createElement("span");
    nameSpan.className = "profile-card-name";
    nameSpan.textContent = p.name;

    titleGroup.appendChild(nameSpan);

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = p.mode;

    header.appendChild(titleGroup);
    header.appendChild(badge);

    const desc = document.createElement("p");
    desc.className = "profile-card-desc";
    desc.textContent = p.description || "(No description)";

    const promptPreview = document.createElement("div");
    promptPreview.className = "profile-card-prompt-preview";
    promptPreview.textContent = p.system_prompt;

    const actions = document.createElement("div");
    actions.className = "profile-card-actions";

    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = `btn btn-sm ${p.id === activeProfileId ? "btn-primary" : "btn-outline"}`;
    useBtn.textContent = p.id === activeProfileId ? "Active Persona" : "Use Profile";
    useBtn.addEventListener("click", () => {
      selectProfile(p.id);
    });
    actions.appendChild(useBtn);

    if (!p.is_builtin) {
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "btn btn-sm btn-ghost";
      delBtn.title = "Delete custom profile";
      delBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
      delBtn.addEventListener("click", async () => {
        const ok = await showConfirmDialog({
          title: "Delete Custom Profile",
          message: "Are you sure you want to delete this custom persona profile?",
          itemTitle: p.name,
          confirmText: "Delete",
          cancelText: "Cancel",
          isDanger: true,
        });
        if (!ok) return;
        try {
          const res = await apiFetch(`/api/profiles/${p.id}`, { method: "DELETE" });
          if (res.ok) {
            showToast(`Deleted profile "${p.name}"`);
            if (activeProfileId === p.id) {
              selectProfile("general");
            }
            await loadProfiles();
            renderProfilesGrid();
          }
        } catch (e) {
          alert(`Failed to delete profile: ${e.message}`);
        }
      });
      actions.appendChild(delBtn);
    }

    card.appendChild(header);
    card.appendChild(desc);
    card.appendChild(promptPreview);
    card.appendChild(actions);

    profilesGrid.appendChild(card);
  }
}

// ============================================================
// Conversations List & Filtering
// ============================================================

async function loadConversations() {
  try {
    const res = await apiFetch("/api/conversations?limit=100");
    if (!res.ok) return;
    loadedConversations = await res.json();
    filterAndRenderConversations();
  } catch (err) {
    console.error("Failed to load conversations:", err);
  }
}

function filterAndRenderConversations() {
  const query = (searchConversationsInput.value || "").trim().toLowerCase();
  const filtered = query
    ? loadedConversations.filter((c) => (c.title || "").toLowerCase().includes(query))
    : loadedConversations;
  renderConversations(filtered);
}

searchConversationsInput.addEventListener("input", filterAndRenderConversations);

function getConversationGroup(isoDateStr) {
  if (!isoDateStr) return "Older";
  const date = new Date(isoDateStr);
  if (isNaN(date.getTime())) return "Older";

  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfYesterday = new Date(startOfToday.getTime() - 24 * 60 * 60 * 1000);
  const startOf7Days = new Date(startOfToday.getTime() - 7 * 24 * 60 * 60 * 1000);
  const startOf30Days = new Date(startOfToday.getTime() - 30 * 24 * 60 * 60 * 1000);

  if (date >= startOfToday) {
    return "Today";
  } else if (date >= startOfYesterday) {
    return "Yesterday";
  } else if (date >= startOf7Days) {
    return "Previous 7 Days";
  } else if (date >= startOf30Days) {
    return "Previous 30 Days";
  } else {
    return "Older";
  }
}

function updateWelcomeHeroVisibility() {
  const hasMessages = messagesContainer && messagesContainer.children.length > 0;
  const welcomeHero = document.getElementById("welcome-hero");
  const quickCards = document.getElementById("quick-suggestion-cards");
  if (welcomeHero) {
    welcomeHero.classList.toggle("hidden", hasMessages);
  }
  if (quickCards) {
    quickCards.classList.toggle("hidden", hasMessages);
  }
}

async function togglePinConversation(id, pinned) {
  try {
    const res = await apiFetch(`/api/conversations/${id}/pin`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pinned }),
    });
    if (!res.ok) throw new Error("Failed to update pin state");
    const target = loadedConversations.find((c) => c.id === id);
    if (target) {
      target.pinned = pinned;
    }
    renderConversations(loadedConversations);
    showToast(pinned ? "Conversation pinned to top" : "Conversation unpinned");
  } catch (err) {
    showToast(`Error: ${err.message}`, true);
  }
}

function createConversationItemElement(item) {
  const isBranch = Boolean(item.parent_conversation_id);
  const div = document.createElement("div");
  div.className = `conversation-item ${item.id === currentConversationId ? "active" : ""} ${isBranch ? "conversation-branch" : ""} ${item.pinned ? "is-pinned" : ""}`;
  div.dataset.id = item.id;

  if (isBranch) {
    const branchIcon = document.createElement("span");
    branchIcon.className = "branch-icon";
    branchIcon.title = "Branch conversation";
    branchIcon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path></svg>`;
    div.appendChild(branchIcon);
  }

  const span = document.createElement("span");
  span.className = "conversation-title";
  span.textContent = item.title;
  span.title = item.title;

  span.addEventListener("dblclick", (e) => {
    e.stopPropagation();
    startRenameConversation(item.id, span);
  });

  const actions = document.createElement("div");
  actions.className = "conversation-actions";

  const pinBtn = document.createElement("button");
  pinBtn.className = `item-action-btn pin-chat-btn ${item.pinned ? "active" : ""}`;
  pinBtn.title = item.pinned ? "Unpin conversation" : "Pin conversation";
  pinBtn.innerHTML = item.pinned
    ? `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>`
    : `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>`;

  pinBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    togglePinConversation(item.id, !item.pinned);
  });

  const forkBtn = document.createElement("button");
  forkBtn.className = "item-action-btn";
  forkBtn.title = "Fork / Branch conversation";
  forkBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path></svg>`;
  forkBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    forkConversationById(item.id, item.title);
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.className = "item-action-btn";
  deleteBtn.title = "Delete conversation";
  deleteBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
  deleteBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    deleteConversation(item.id, item.title);
  });

  actions.appendChild(pinBtn);
  actions.appendChild(forkBtn);
  actions.appendChild(deleteBtn);
  div.appendChild(span);
  div.appendChild(actions);

  div.addEventListener("click", () => {
    selectConversation(item.id);
  });

  return div;
}

function renderConversations(items) {
  conversationsList.innerHTML = "";
  if (!items || items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-hint";
    empty.style.padding = "16px 12px";
    empty.style.color = "var(--text-muted)";
    empty.style.fontSize = "0.85rem";
    empty.style.textAlign = "center";
    empty.textContent = loadedConversations.length > 0 ? "No matches found" : "No conversations yet";
    conversationsList.appendChild(empty);
    return;
  }

  // 1. Pinned conversations at top
  const pinnedItems = items.filter((item) => Boolean(item.pinned));
  const unpinnedItems = items.filter((item) => !item.pinned);

  if (pinnedItems.length > 0) {
    const pinnedHeader = document.createElement("div");
    pinnedHeader.className = "conversation-group-header pinned";
    pinnedHeader.innerHTML = `
      <svg class="pinned-star-icon" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
      <span>Pinned</span>
    `;
    conversationsList.appendChild(pinnedHeader);
    for (const item of pinnedItems) {
      conversationsList.appendChild(createConversationItemElement(item));
    }
  }

  const groupOrder = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"];
  const groups = {
    "Today": [],
    "Yesterday": [],
    "Previous 7 Days": [],
    "Previous 30 Days": [],
    "Older": [],
  };

  for (const item of unpinnedItems) {
    const groupName = getConversationGroup(item.updated_at || item.created_at);
    if (!groups[groupName]) {
      groups[groupName] = [];
    }
    groups[groupName].push(item);
  }

  for (const gName of groupOrder) {
    const groupItems = groups[gName];
    if (!groupItems || groupItems.length === 0) continue;

    const groupHeader = document.createElement("div");
    groupHeader.className = "conversation-group-header";
    groupHeader.textContent = gName;
    conversationsList.appendChild(groupHeader);

    for (const item of groupItems) {
      conversationsList.appendChild(createConversationItemElement(item));
    }
  }
}

async function selectConversation(id) {
  if (currentConversationId === id) return;
  if (typeof closeInChatSearch === "function") closeInChatSearch();
  if (typeof closeSlashMenu === "function") closeSlashMenu();
  currentConversationId = id;
  sidebar.classList.remove("open");

  // Highlight in list
  for (const el of conversationsList.querySelectorAll(".conversation-item")) {
    el.classList.toggle("active", el.dataset.id === id);
  }

  messagesContainer.innerHTML = "";
  try {
    const res = await apiFetch(`/api/conversations/${id}`);
    if (!res.ok) return;
    const detail = await res.json();
    conversationTitle.textContent = detail.title || "Knight";

    if (detail.messages && detail.messages.length > 0) {
      for (const msg of detail.messages) {
        renderMessageBubble(msg.role, msg.content, msg.sources || [], msg.id);
      }
      scrollToBottom(true);
    }
    updateWelcomeHeroVisibility();
  } catch (err) {
    console.error("Failed to load conversation details:", err);
    updateWelcomeHeroVisibility();
  }
}

function startRenameConversation(id, titleSpan) {
  const currentTitle = titleSpan.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "input";
  input.value = currentTitle;
  input.style.width = "100%";
  input.style.padding = "2px 6px";
  input.style.fontSize = "0.85rem";

  async function finishRename() {
    const newTitle = input.value.trim();
    if (newTitle && newTitle !== currentTitle) {
      try {
        const res = await apiFetch(`/api/conversations/${id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: newTitle }),
        });
        if (res.ok) {
          titleSpan.textContent = newTitle;
          if (id === currentConversationId) {
            conversationTitle.textContent = newTitle;
          }
          const item = loadedConversations.find((c) => c.id === id);
          if (item) item.title = newTitle;
        }
      } catch (err) {
        titleSpan.textContent = currentTitle;
      }
    } else {
      titleSpan.textContent = currentTitle;
    }
  }

  input.addEventListener("blur", finishRename);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      input.blur();
    } else if (e.key === "Escape") {
      input.value = currentTitle;
      input.blur();
    }
  });

  titleSpan.textContent = "";
  titleSpan.appendChild(input);
  input.focus();
  input.select();
}

// ============================================================
// Custom Confirmation Dialog Service
// ============================================================

let confirmDialogResolver = null;

function showConfirmDialog({
  title = "Confirm Action",
  message = "Are you sure you want to proceed?",
  itemTitle = "",
  confirmText = "Confirm",
  cancelText = "Cancel",
  subtext = "This action cannot be undone.",
  isDanger = false,
}) {
  return new Promise((resolve) => {
    confirmDialogResolver = resolve;

    if (confirmDialogTitle) confirmDialogTitle.textContent = title;
    if (confirmDialogMessage) confirmDialogMessage.textContent = message;

    if (confirmDialogItemBox && confirmDialogItemName) {
      if (itemTitle && itemTitle.trim()) {
        confirmDialogItemName.textContent = itemTitle.trim();
        confirmDialogItemBox.classList.remove("hidden");
      } else {
        confirmDialogItemBox.classList.add("hidden");
      }
    }

    if (confirmDialogSubtext) {
      confirmDialogSubtext.textContent = subtext || "";
      confirmDialogSubtext.style.display = subtext ? "block" : "none";
    }

    if (confirmDialogActionBtn) {
      confirmDialogActionBtn.textContent = confirmText;
      confirmDialogActionBtn.className = isDanger ? "btn btn-danger" : "btn btn-primary";
    }

    if (cancelConfirmDialogBtn) {
      cancelConfirmDialogBtn.textContent = cancelText;
    }

    if (confirmIconWrap) {
      if (isDanger) {
        confirmIconWrap.className = "confirm-icon-wrap";
        if (confirmIconDanger) confirmIconDanger.classList.remove("hidden");
        if (confirmIconDefault) confirmIconDefault.classList.add("hidden");
      } else {
        confirmIconWrap.className = "confirm-icon-wrap default";
        if (confirmIconDanger) confirmIconDanger.classList.add("hidden");
        if (confirmIconDefault) confirmIconDefault.classList.remove("hidden");
      }
    }

    if (confirmDialog) {
      confirmDialog.showModal();
    } else {
      resolve(false);
    }
  });
}

function closeConfirmDialog(result = false) {
  if (confirmDialog && confirmDialog.open) {
    confirmDialog.close();
  }
  if (confirmDialogResolver) {
    confirmDialogResolver(result);
    confirmDialogResolver = null;
  }
}

if (confirmDialogActionBtn) {
  confirmDialogActionBtn.addEventListener("click", () => closeConfirmDialog(true));
}
if (cancelConfirmDialogBtn) {
  cancelConfirmDialogBtn.addEventListener("click", () => closeConfirmDialog(false));
}
if (closeConfirmDialogBtn) {
  closeConfirmDialogBtn.addEventListener("click", () => closeConfirmDialog(false));
}

async function deleteConversation(id, title) {
  const confirmed = await showConfirmDialog({
    title: "Delete Conversation",
    message: "Are you sure you want to permanently delete this conversation? All messages, turns, and citations will be removed.",
    itemTitle: title || "Untitled Conversation",
    confirmText: "Delete",
    cancelText: "Cancel",
    isDanger: true,
  });
  if (!confirmed) return;

  try {
    const res = await apiFetch(`/api/conversations/${id}`, { method: "DELETE" });
    if (res.ok) {
      if (currentConversationId === id) {
        startNewChat();
      }
      loadConversations();
      showToast("Conversation deleted");
    } else {
      showToast("Failed to delete conversation");
    }
  } catch (err) {
    console.error("Failed to delete conversation:", err);
    showToast("Failed to delete conversation");
  }
}

async function forkConversationById(convId, baseTitle = "Conversation") {
  if (!convId) return;
  const suggestedTitle = `${baseTitle} (Branch)`;
  const branchTitle = prompt("Create branch / fork of conversation:\nEnter branch title:", suggestedTitle);
  if (branchTitle === null) return;

  showToast("Creating conversation branch...");
  try {
    const res = await apiFetch(`/api/conversations/${convId}/fork`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: branchTitle.trim() || suggestedTitle }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(`Fork failed: ${err.detail || "Server error"}`);
      return;
    }
    const newConv = await res.json();
    showToast(`Branch "${newConv.title}" created!`);
    await loadConversations();
    await selectConversation(newConv.id);
  } catch (err) {
    showToast(`Failed to fork conversation: ${err.message}`);
  }
}

async function forkConversationAtMessage(messageId) {
  if (!currentConversationId) return;
  const currentTitle = conversationTitle.textContent || "Conversation";
  const suggestedTitle = `${currentTitle} (Branch)`;
  const branchTitle = prompt("Branch from this turn (Time Travel):\nEnter branch title:", suggestedTitle);
  if (branchTitle === null) return;

  showToast("Forking conversation from selected turn...");
  try {
    const res = await apiFetch(`/api/conversations/${currentConversationId}/fork`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message_id: messageId,
        title: branchTitle.trim() || suggestedTitle,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      showToast(`Fork failed: ${err.detail || "Server error"}`);
      return;
    }
    const newConv = await res.json();
    showToast(`Branch "${newConv.title}" created!`);
    await loadConversations();
    await selectConversation(newConv.id);
    messageInput.focus();
  } catch (err) {
    showToast(`Failed to fork conversation: ${err.message}`);
  }
}

function startNewChat() {
  if (typeof closeInChatSearch === "function") closeInChatSearch();
  if (typeof closeSlashMenu === "function") closeSlashMenu();
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentConversationId = null;
  conversationTitle.textContent = "New Session";
  messagesContainer.innerHTML = "";
  for (const el of conversationsList.querySelectorAll(".conversation-item")) {
    el.classList.remove("active");
  }
  messageInput.value = "";
  updateWelcomeHeroVisibility();
  messageInput.focus();
}

// ============================================================
// Custom Export Conversation Dialog & Service
// ============================================================

function openExportDialog() {
  if (!exportDialog) return;

  if (!currentConversationId) {
    if (exportEmptyState) exportEmptyState.classList.remove("hidden");
    if (exportContentState) exportContentState.classList.add("hidden");
    if (submitExportDialogBtn) submitExportDialogBtn.disabled = true;
  } else {
    if (exportEmptyState) exportEmptyState.classList.add("hidden");
    if (exportContentState) exportContentState.classList.remove("hidden");
    if (submitExportDialogBtn) submitExportDialogBtn.disabled = false;

    const title = conversationTitle ? conversationTitle.textContent.trim() : "Current Chat";
    if (exportDialogConvTitle) exportDialogConvTitle.textContent = title || "Current Chat";

    const msgCount = messagesContainer ? messagesContainer.querySelectorAll(".message-row").length : 0;
    if (exportDialogTurnsBadge) exportDialogTurnsBadge.textContent = `${msgCount} message${msgCount === 1 ? "" : "s"}`;
  }

  exportDialog.showModal();
}

function closeExportDialog() {
  if (exportDialog && exportDialog.open) {
    exportDialog.close();
  }
}

// Export format card radio toggling
document.querySelectorAll(".export-format-card").forEach((card) => {
  card.addEventListener("click", () => {
    document.querySelectorAll(".export-format-card").forEach((c) => c.classList.remove("active"));
    card.classList.add("active");
    const radio = card.querySelector('input[type="radio"]');
    if (radio) radio.checked = true;
  });
});

if (closeExportDialogBtn) {
  closeExportDialogBtn.addEventListener("click", closeExportDialog);
}
if (cancelExportDialogBtn) {
  cancelExportDialogBtn.addEventListener("click", closeExportDialog);
}

if (submitExportDialogBtn) {
  submitExportDialogBtn.addEventListener("click", async () => {
    if (!currentConversationId) return;
    const selectedFormat = document.querySelector('input[name="export-file-format"]:checked')?.value || "markdown";
    const includeSources = exportIncludeSourcesCheck ? exportIncludeSourcesCheck.checked : true;
    closeExportDialog();
    await exportCurrentConversation(selectedFormat, includeSources);
  });
}

if (exportChatBtn) {
  exportChatBtn.addEventListener("click", openExportDialog);
}

async function exportCurrentConversation(format = "markdown", includeSources = true) {
  if (!currentConversationId) {
    showToast("No active conversation to export");
    return;
  }
  try {
    const res = await apiFetch(`/api/conversations/${currentConversationId}`);
    if (!res.ok) throw new Error("Failed to load conversation details");
    const data = await res.json();
    const title = data.title || "Knight_Conversation";

    let blob, ext;
    if (format === "json") {
      const exportData = { ...data };
      if (!includeSources && exportData.messages) {
        exportData.messages = exportData.messages.map((m) => {
          const { sources, ...rest } = m;
          return rest;
        });
      }
      blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
      ext = "json";
    } else {
      let md = `# ${title}\n\n`;
      md += `*Exported from Knight on ${new Date().toLocaleString()}*\n\n---\n\n`;
      if (data.messages) {
        for (const m of data.messages) {
          const roleLabel = m.role === "user" ? "### User" : "### Knight";
          md += `${roleLabel}\n\n${m.content}\n\n`;
          if (includeSources && m.sources && m.sources.length > 0) {
            md += `**Sources:**\n`;
            for (const s of m.sources) {
              md += `- [${s.idx}] [${s.title || s.url}](${s.url})\n`;
            }
            md += "\n";
          }
          md += "---\n\n";
        }
      }
      blob = new Blob([md], { type: "text/markdown" });
      ext = "md";
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeTitle = title.replace(/[^a-zA-Z0-9_\-]/g, "_").toLowerCase();
    a.download = `${safeTitle}.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Exported "${safeTitle}.${ext}"`);
  } catch (err) {
    console.error("Export failed:", err);
    showToast("Export failed: " + err.message);
  }
}

// ============================================================
// Markdown, Code Cards & Citation Rendering
// ============================================================

function extractThinkingBlocks(rawMarkdown) {
  if (!rawMarkdown) return { thinking: null, main: "", isLive: false };

  // Check for complete <think>...</think>
  const closedRegex = /<think>([\s\S]*?)<\/think>/i;
  const closedMatch = rawMarkdown.match(closedRegex);
  if (closedMatch) {
    const thinking = closedMatch[1].trim();
    const main = rawMarkdown.replace(closedRegex, "").trim();
    return { thinking, main, isLive: false };
  }

  // Check for open <think>... (streaming in progress)
  const openRegex = /<think>([\s\S]*)$/i;
  const openMatch = rawMarkdown.match(openRegex);
  if (openMatch) {
    const prefix = rawMarkdown.substring(0, openMatch.index).trim();
    const thinking = openMatch[1].trim();
    return { thinking, main: prefix, isLive: true };
  }

  return { thinking: null, main: rawMarkdown, isLive: false };
}

function sanitizeAndRenderMarkdown(rawMarkdown, sources = []) {
  if (!rawMarkdown) return "";

  const { thinking, main, isLive } = extractThinkingBlocks(rawMarkdown);

  let thinkingHtml = "";
  if (thinking !== null && thinking.length > 0) {
    let parsedThinking = "";
    if (window.marked && typeof window.marked.parse === "function") {
      parsedThinking = window.marked.parse(thinking);
    } else {
      parsedThinking = `<p>${escapeHTML(thinking)}</p>`;
    }

    const openAttr = isLive ? " open" : "";
    const statusText = isLive ? "Thinking..." : "Thought process";
    const statusClass = isLive ? "thinking-live" : "thinking-done";

    thinkingHtml = `
      <details class="thinking-block ${statusClass}"${openAttr}>
        <summary class="thinking-summary">
          <div class="thinking-summary-left">
            <svg class="thinking-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2a8 8 0 0 0-8 8c0 3 2 5.5 4 7v3h8v-3c2-1.5 4-4 4-7a8 8 0 0 0-8-8z"></path>
              <line x1="9" y1="22" x2="15" y2="22"></line>
            </svg>
            <span class="thinking-title">Thinking Process</span>
          </div>
          <span class="thinking-badge">${escapeHTML(statusText)}</span>
        </summary>
        <div class="thinking-body">
          ${parsedThinking}
        </div>
      </details>
    `;
  }

  // 1. Parse main markdown with marked.js
  let mainHtml = "";
  if (main) {
    if (window.marked && typeof window.marked.parse === "function") {
      mainHtml = window.marked.parse(main);
    } else {
      mainHtml = `<p>${escapeHTML(main)}</p>`;
    }
  }

  let html = thinkingHtml + mainHtml;

  // 2. Transform [n] citation markers into links BEFORE sanitization (U-07)
  // Building citations prior to sanitization ensures DOMPurify processes the complete
  // markup and strips any injected quotes, attributes, or malicious tags.
  if (sources && sources.length > 0) {
    const sourceMap = new Map();
    for (const s of sources) {
      sourceMap.set(s.idx, s);
    }

    html = html.replace(/\[(\d+)\]/g, (match, idxStr) => {
      const idx = parseInt(idxStr, 10);
      const s = sourceMap.get(idx);
      if (s && isValidHttpUrl(s.url)) {
        let domain = "";
        try {
          domain = new URL(s.url).hostname;
        } catch (_) {}
        const titleEscaped = escapeHTML(s.title || domain || s.url);
        const snippetEscaped = escapeHTML(s.snippet || "");
        return `<a href="${encodeURI(s.url)}" target="_blank" rel="noopener noreferrer" class="citation-link" data-idx="${idx}" data-domain="${escapeHTML(domain)}" data-title="${titleEscaped}" data-snippet="${snippetEscaped}" title="${titleEscaped}">[${idx}]</a>`;
      }
      return match;
    });
  }

  // 3. Sanitize HTML strictly with DOMPurify as the final step
  if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
    html = window.DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true },
      ADD_TAGS: ["details", "summary"],
      ADD_ATTR: ["target", "rel", "class", "title", "open", "data-idx", "data-domain", "data-title", "data-snippet"],
    });
  }

  return html;
}

const CODE_EXT_MAP = {
  python: "py", py: "py",
  javascript: "js", js: "js",
  typescript: "ts", ts: "ts",
  html: "html", htm: "html",
  css: "css",
  json: "json",
  sql: "sql",
  bash: "sh", sh: "sh", shell: "sh",
  rust: "rs", rs: "rs",
  go: "go",
  c: "c", cpp: "cpp", "c++": "cpp",
  markdown: "md", md: "md",
  yaml: "yaml", yml: "yaml",
  toml: "toml",
  xml: "xml",
};

function enhanceCodeBlocks(container) {
  if (!container) return;
  const preElements = container.querySelectorAll("pre:not(.enhanced)");
  preElements.forEach((pre) => {
    pre.classList.add("enhanced");
    const code = pre.querySelector("code");
    const codeText = code ? code.innerText : pre.innerText;

    let lang = "text";
    if (code && code.className) {
      const match = code.className.match(/language-([a-zA-Z0-9_-]+)/);
      if (match) lang = match[1].toLowerCase();
    }

    const card = document.createElement("div");
    card.className = "code-block-card";

    const header = document.createElement("div");
    header.className = "code-card-header";

    const langLabel = document.createElement("span");
    langLabel.className = "code-lang-label";
    langLabel.textContent = lang;

    const actions = document.createElement("div");
    actions.className = "code-card-actions";

    // 1. Line numbers toggle
    const linesBtn = document.createElement("button");
    linesBtn.type = "button";
    linesBtn.className = "code-tool-btn";
    linesBtn.title = "Toggle line numbers";
    linesBtn.innerHTML = `<span>#</span>`;
    linesBtn.addEventListener("click", () => {
      let gutter = pre.querySelector(".line-numbers-gutter");
      if (gutter) {
        gutter.remove();
        pre.classList.remove("with-line-numbers");
        linesBtn.classList.remove("active");
      } else {
        const lineCount = codeText.split("\n").length;
        gutter = document.createElement("div");
        gutter.className = "line-numbers-gutter";
        gutter.innerHTML = Array.from({ length: lineCount }, (_, i) => i + 1).join("<br>");
        pre.classList.add("with-line-numbers");
        pre.insertBefore(gutter, pre.firstChild);
        linesBtn.classList.add("active");
      }
    });

    // 2. Wrap lines toggle
    const wrapBtn = document.createElement("button");
    wrapBtn.type = "button";
    wrapBtn.className = "code-tool-btn";
    wrapBtn.title = "Toggle line wrapping";
    wrapBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 10 4 15 9 20"></polyline><path d="M20 4v7a4 4 0 0 1-4 4H4"></path></svg>`;
    wrapBtn.addEventListener("click", () => {
      const isWrapped = pre.classList.toggle("code-soft-wrap");
      wrapBtn.classList.toggle("active", isWrapped);
    });

    // 3. Download file
    const downloadBtn = document.createElement("button");
    downloadBtn.type = "button";
    downloadBtn.className = "code-tool-btn";
    downloadBtn.title = "Download code file";
    downloadBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`;
    downloadBtn.addEventListener("click", () => {
      const ext = CODE_EXT_MAP[lang] || "txt";
      const blob = new Blob([codeText], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `snippet.${ext}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast(`Downloaded snippet.${ext}`);
    });

    // 4. Copy button
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "code-tool-btn";
    copyBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      <span>Copy</span>
    `;

    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(codeText);
        copyBtn.classList.add("copied");
        copyBtn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
          <span>Copied!</span>
        `;
        setTimeout(() => {
          copyBtn.classList.remove("copied");
          copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>Copy</span>
          `;
        }, 2000);
      } catch (e) {
        console.error("Clipboard copy failed", e);
      }
    });

    actions.appendChild(linesBtn);
    actions.appendChild(wrapBtn);
    actions.appendChild(downloadBtn);
    actions.appendChild(copyBtn);

    header.appendChild(langLabel);
    header.appendChild(actions);

    pre.parentNode.insertBefore(card, pre);
    card.appendChild(header);
    card.appendChild(pre);
  });
}

function isValidHttpUrl(string) {
  try {
    const url = new URL(string);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch (_) {
    return false;
  }
}

function escapeHTML(val) {
  const str = String(val ?? "");
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

// ============================================================
// Message Bubbles, Actions & Tool Chips
// ============================================================

function createMessageActionsBar(role, text, messageId = null, telemetry = null) {
  const bar = document.createElement("div");
  bar.className = "message-actions-bar";

  // 1. Copy message
  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "msg-action-btn";
  copyBtn.title = "Copy message";
  copyBtn.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
    <span>Copy</span>
  `;
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.classList.add("copied");
      copyBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
        <span>Copied</span>
      `;
      setTimeout(() => {
        copyBtn.classList.remove("copied");
        copyBtn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          <span>Copy</span>
        `;
      }, 2000);
    } catch (e) {
      console.error("Clipboard copy failed", e);
    }
  });
  bar.appendChild(copyBtn);

  if (role === "user") {
    // 2. Edit & Resend
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "msg-action-btn";
    editBtn.title = "Edit & resend message";
    editBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
      <span>Edit</span>
    `;
    editBtn.addEventListener("click", () => {
      messageInput.value = text;
      autoResizeTextarea();
      messageInput.focus();
    });
    bar.appendChild(editBtn);
  } else if (role === "assistant") {
    // 2. Regenerate
    const regenBtn = document.createElement("button");
    regenBtn.type = "button";
    regenBtn.className = "msg-action-btn";
    regenBtn.title = "Regenerate response";
    regenBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"></polyline><polyline points="23 20 23 14 17 14"></polyline><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path></svg>
      <span>Regenerate</span>
    `;
    regenBtn.addEventListener("click", () => {
      if (lastUserPrompt) {
        sendMessage(lastUserPrompt);
      }
    });
    bar.appendChild(regenBtn);
  }

  // 3. Fork from this turn (Time Travel Branch)
  if (messageId) {
    const forkBtn = document.createElement("button");
    forkBtn.type = "button";
    forkBtn.className = "msg-action-btn fork-msg-btn";
    forkBtn.title = "Fork conversation from this turn (Time Travel)";
    forkBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path></svg>
      <span>Fork</span>
    `;
    forkBtn.addEventListener("click", () => {
      forkConversationAtMessage(messageId);
    });
    bar.appendChild(forkBtn);
  }

  // 4. Telemetry Pill (Tokens/sec, TTFT, Token count)
  if (role === "assistant" && telemetry) {
    const telemetryBadge = document.createElement("span");
    telemetryBadge.className = "msg-telemetry-badge";
    const durationSec = (telemetry.elapsedMs / 1000).toFixed(1);
    const throughput = telemetry.tps === null ? "n/a" : `${telemetry.tps} t/s`;
    const count =
      telemetry.tokens === null
        ? `${telemetry.chunks} chunks`
        : `${telemetry.tokens} tok`;
    telemetryBadge.title =
      `Throughput: ${throughput} | Generated: ${count} | ` +
      `TTFT: ${telemetry.ttftMs}ms | Latency: ${durationSec}s`;
    telemetryBadge.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
      <span>${throughput} • ${count} • ${telemetry.ttftMs}ms TTFT</span>
    `;
    bar.appendChild(telemetryBadge);
  }

  return bar;
}

function renderMessageBubble(role, text, sources = [], messageId = null) {
  const wrapper = document.createElement("div");
  wrapper.className = `message-wrapper ${role}`;
  if (messageId) {
    wrapper.dataset.messageId = messageId;
  }

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (role === "user") {
    bubble.textContent = text;
    wrapper.appendChild(bubble);
    const actionsBar = createMessageActionsBar("user", text, messageId);
    wrapper.appendChild(actionsBar);
  } else {
    bubble.className += " markdown-body";
    bubble.innerHTML = sanitizeAndRenderMarkdown(text, sources);
    enhanceCodeBlocks(bubble);

    // Render sources card if present
    if (sources && sources.length > 0) {
      const sourcesEl = createSourcesElement(sources);
      bubble.appendChild(sourcesEl);
    }
    wrapper.appendChild(bubble);
    const actionsBar = createMessageActionsBar("assistant", text, messageId);
    wrapper.appendChild(actionsBar);
  }

  messagesContainer.appendChild(wrapper);
  updateWelcomeHeroVisibility();
  return wrapper;
}

function createSourcesElement(sources) {
  const card = document.createElement("div");
  card.className = "sources-container";

  const header = document.createElement("div");
  header.className = "sources-header";
  header.textContent = "Sources";
  card.appendChild(header);

  const list = document.createElement("ul");
  list.className = "sources-list";

  for (const s of sources) {
    if (!isValidHttpUrl(s.url)) continue;
    const li = document.createElement("li");
    li.className = "source-item";

    const a = document.createElement("a");
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";

    let domain = s.url;
    try { domain = new URL(s.url).hostname; } catch (_) {}

    a.innerHTML = `<span class="badge">[${s.idx}]</span> <span>${escapeHTML(s.title || domain)}</span>`;
    li.appendChild(a);
    list.appendChild(li);
  }

  card.appendChild(list);
  return card;
}

function createStreamingAssistantBubble() {
  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper assistant";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble markdown-body";

  const toolChipsContainer = document.createElement("div");
  toolChipsContainer.className = "tool-chips-container";

  const liveContent = document.createElement("div");
  liveContent.setAttribute("aria-live", "polite");
  liveContent.className = "live-markdown-content";

  const cursor = document.createElement("span");
  cursor.className = "streaming-cursor";

  const statsFooter = document.createElement("div");
  statsFooter.className = "message-stats-footer hidden";

  bubble.appendChild(toolChipsContainer);
  bubble.appendChild(liveContent);
  bubble.appendChild(cursor);
  bubble.appendChild(statsFooter);
  wrapper.appendChild(bubble);

  messagesContainer.appendChild(wrapper);
  updateWelcomeHeroVisibility();
  scrollToBottom(true);

  return { wrapper, bubble, toolChipsContainer, liveContent, cursor, statsFooter };
}

// ============================================================
// Chat Streaming via SSE
// ============================================================

async function sendMessage(text) {
  const modeRadio = document.querySelector('input[name="chat-mode"]:checked');
  const mode = modeRadio ? modeRadio.value : "agent";
  lastUserPrompt = text;

  // Sound effect on send
  soundEffects.playSendSound();
  const requestStartTime = performance.now();
  let firstTokenTime = null;

  // 1. Render user message bubble
  renderMessageBubble("user", text);
  messageInput.value = "";
  autoResizeTextarea();
  scrollToBottom(true);

  // 2. Set streaming UI state
  setStreamingState(true);
  currentAbortController = new AbortController();
  currentSources = [];

  const streamElements = createStreamingAssistantBubble();
  let accumulatedText = "";

  const customInstructions = localStorage.getItem("knight-custom-instructions") || "";

  try {
    const payload = {
      conversation_id: currentConversationId,
      message: text,
      mode: mode,
      profile_id: activeProfileId,
      model_customization: getActiveModelCustomization(),
    };
    if (customInstructions.trim()) {
      payload.system_prompt_override = customInstructions.trim();
    }

    const response = await apiFetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal,
    });

    if (!response.ok) {
      let errMsg = `Server error ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) errMsg = errJson.detail;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      for (const frame of frames) {
        if (!frame.trim() || frame.startsWith(":")) continue;

        const lines = frame.split("\n");
        let eventType = "message";
        let dataStr = "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataStr = line.slice(5).trim();
          }
        }

        if (!dataStr) continue;
        let data = {};
        try {
          data = JSON.parse(dataStr);
        } catch (_) {
          continue;
        }

        // Handle SSE events
        if (eventType === "meta") {
          if (data.conversation_id && !currentConversationId) {
            currentConversationId = data.conversation_id;
            loadConversations();
          }
        } else if (eventType === "token") {
          if (!firstTokenTime) {
            firstTokenTime = performance.now();
          }
          accumulatedText += data.text || "";

          // Preserve user-toggled open/close state on thinking accordion if present
          const existingThink = streamElements.liveContent.querySelector(".thinking-block");
          const wasOpen = existingThink ? existingThink.open : null;

          streamElements.liveContent.innerHTML = sanitizeAndRenderMarkdown(
            accumulatedText,
            currentSources
          );

          if (wasOpen !== null) {
            const newThink = streamElements.liveContent.querySelector(".thinking-block");
            if (newThink) newThink.open = wasOpen;
          }

          if (autoScroll) {
            scrollToBottom(false);
          } else {
            notifyUnreadUpdate();
          }
        } else if (eventType === "tool_start") {
          renderToolStart(streamElements.toolChipsContainer, data);
          if (autoScroll) {
            scrollToBottom(false);
          } else {
            notifyUnreadUpdate();
          }
        } else if (eventType === "tool_end") {
          renderToolEnd(streamElements.toolChipsContainer, data);
        } else if (eventType === "sources") {
          currentSources = data.items || [];
        } else if (eventType === "done") {
          // Finalise assistant message
          streamElements.cursor.remove();
          streamElements.liveContent.innerHTML = sanitizeAndRenderMarkdown(
            accumulatedText,
            currentSources
          );
          enhanceCodeBlocks(streamElements.bubble);

          if (currentSources.length > 0) {
            const sourcesEl = createSourcesElement(currentSources);
            streamElements.bubble.appendChild(sourcesEl);
          }

          const finishTime = performance.now();
          const elapsedMs = data.elapsed_ms || Math.round(finishTime - requestStartTime);
          const ttftMs = firstTokenTime ? Math.round(firstTokenTime - requestStartTime) : 0;
          const usage = data.usage || {};
          const tokens = Number.isFinite(usage.completion_tokens)
            ? usage.completion_tokens
            : null;
          const chunks = Number.isFinite(usage.chunks) ? usage.chunks : 0;
          const sec = elapsedMs / 1000;
          const tok_s =
            tokens !== null && tokens > 0 && sec > 0
              ? (tokens / sec).toFixed(1)
              : null;

          const telemetry = {
            tps: tok_s,
            tokens: tokens,
            chunks: chunks,
            ttftMs: ttftMs,
            elapsedMs: elapsedMs,
          };

          // Render Performance Stats Footer. Chunk counts are labelled as
          // chunks; tokens/sec is only shown when the server gave real usage.
          if (tok_s !== null) {
            streamElements.statsFooter.textContent = `${tok_s} t/s · ${tokens} tokens · ${sec.toFixed(1)}s`;
            streamElements.statsFooter.classList.remove("hidden");
          } else if (chunks > 0) {
            streamElements.statsFooter.textContent = `${chunks} chunks · ${sec.toFixed(1)}s`;
            streamElements.statsFooter.classList.remove("hidden");
          }

          // Append Message Actions Bar with Telemetry
          const actionsBar = createMessageActionsBar("assistant", accumulatedText, data.message_id, telemetry);
          streamElements.wrapper.appendChild(actionsBar);

          // Play peaceful completion chime
          soundEffects.playDoneSound();

          loadConversations();
        } else if (eventType === "error") {
          throw new Error(data.message || data.code || "Generation failed");
        }
      }
    }
  } catch (err) {
    streamElements.cursor.remove();
    if (err.name === "AbortError") {
      const cancelHint = document.createElement("div");
      cancelHint.style.color = "var(--text-muted)";
      cancelHint.style.fontSize = "0.85rem";
      cancelHint.style.marginTop = "8px";
      cancelHint.textContent = "(Generation stopped)";
      streamElements.bubble.appendChild(cancelHint);
    } else {
      renderInlineError(streamElements.bubble, err.message, text);
    }
  } finally {
    setStreamingState(false);
    currentAbortController = null;
  }
}

function renderToolStart(container, data) {
  const chipGroup = document.createElement("div");
  chipGroup.className = "tool-chip-group";
  chipGroup.style.display = "flex";
  chipGroup.style.flexDirection = "column";
  chipGroup.style.width = "100%";

  const chip = document.createElement("div");
  chip.className = "tool-chip";
  chip.id = `tool-${data.id}`;
  chip.title = "Click to inspect tool execution";

  const iconName =
    data.name === "web_search"
      ? "Search"
      : data.name === "fetch_page"
      ? "Fetch Page"
      : data.name === "read_local_file"
      ? "Read File"
      : data.name === "list_directory"
      ? "List Folder"
      : data.name === "calculate"
      ? "Calculate"
      : data.name === "current_datetime"
      ? "DateTime"
      : data.name;
  chip.innerHTML = `
    <span class="spinner"></span>
    <span class="tool-name">${escapeHTML(iconName)}</span>
  `;

  const details = document.createElement("div");
  details.className = "tool-details-panel hidden";
  details.id = `tool-details-${data.id}`;

  const argsFormatted = data.args ? JSON.stringify(data.args, null, 2) : "{}";
  details.innerHTML = `
    <div class="tool-detail-row"><span class="tool-detail-label">Arguments:</span> ${escapeHTML(argsFormatted)}</div>
    <div class="tool-detail-row"><span class="tool-detail-label">Status:</span> Running...</div>
  `;

  chip.addEventListener("click", () => {
    details.classList.toggle("hidden");
  });

  chipGroup.appendChild(chip);
  chipGroup.appendChild(details);
  container.appendChild(chipGroup);
  if (autoScroll) scrollToBottom(false);
}

function renderToolEnd(container, data) {
  const chip = container.querySelector(`#tool-${data.id}`);
  const details = container.querySelector(`#tool-details-${data.id}`);
  if (!chip) return;

  const statusIcon = data.ok ? "OK" : "Error";
  chip.className = `tool-chip ${data.ok ? "success" : "error"}`;
  const durationSec = data.duration_ms ? `${(data.duration_ms / 1000).toFixed(1)}s` : "";
  const nameSpan = chip.querySelector(".tool-name");
  const nameText = nameSpan ? nameSpan.textContent : (data.name || "tool");

  chip.innerHTML = `
    <span class="tool-status-tag">${statusIcon}</span>
    <span>${escapeHTML(nameText)}</span>
    ${durationSec ? `<span style="opacity:0.6">${durationSec}</span>` : ""}
    <span style="font-size:0.65rem;opacity:0.6;margin-left:4px">▼</span>
  `;

  if (details) {
    const summaryText = data.summary || (data.ok ? "Completed successfully" : "Execution failed");
    details.innerHTML = `
      <div class="tool-detail-row"><span class="tool-detail-label">Duration:</span> ${data.duration_ms || 0}ms</div>
      <div class="tool-detail-row"><span class="tool-detail-label">Status:</span> ${data.ok ? "Success" : "Failed"}</div>
      <div class="tool-detail-row"><span class="tool-detail-label">Output summary:</span> ${escapeHTML(summaryText)}</div>
    `;
  }
}

function renderInlineError(container, errorMessage, originalText) {
  const errBox = document.createElement("div");
  errBox.className = "system-error-bubble";
  errBox.style.marginTop = "10px";

  const span = document.createElement("span");
  span.textContent = `Error: ${errorMessage}`;

  const retryBtn = document.createElement("button");
  retryBtn.className = "btn btn-outline btn-sm";
  retryBtn.textContent = "Retry";
  retryBtn.addEventListener("click", () => {
    errBox.remove();
    sendMessage(originalText);
  });

  errBox.appendChild(span);
  errBox.appendChild(retryBtn);
  container.appendChild(errBox);
}

function setStreamingState(isStreaming) {
  if (isStreaming) {
    sendBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    messageInput.disabled = true;
  } else {
    sendBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    messageInput.disabled = false;
    messageInput.focus();
  }
}

let unreadUpdatesCount = 0;

function scrollToBottom(force = false) {
  const threshold = 80;
  const isNearBottom =
    messagesContainer.scrollHeight -
      messagesContainer.scrollTop -
      messagesContainer.clientHeight <=
    threshold;

  if (force || isNearBottom) {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    resetUnreadScrollBadge();
  }
}

function updateScrollButtonVisibility() {
  if (!messagesContainer || !scrollToBottomBtn) return;
  const distanceFromBottom =
    messagesContainer.scrollHeight -
    messagesContainer.scrollTop -
    messagesContainer.clientHeight;

  if (distanceFromBottom > 120) {
    scrollToBottomBtn.classList.remove("hidden");
  } else {
    scrollToBottomBtn.classList.add("hidden");
    resetUnreadScrollBadge();
  }
}

function resetUnreadScrollBadge() {
  unreadUpdatesCount = 0;
  if (scrollUnreadBadge) {
    scrollUnreadBadge.classList.add("hidden");
  }
}

function notifyUnreadUpdate() {
  unreadUpdatesCount++;
  if (scrollUnreadBadge) {
    scrollUnreadBadge.classList.remove("hidden");
  }
}

messagesContainer.addEventListener("scroll", () => {
  const threshold = 80;
  autoScroll =
    messagesContainer.scrollHeight -
      messagesContainer.scrollTop -
      messagesContainer.clientHeight <=
    threshold;
  updateScrollButtonVisibility();
});

if (scrollToBottomBtn) {
  scrollToBottomBtn.addEventListener("click", () => {
    scrollToBottom(true);
    scrollToBottomBtn.classList.add("hidden");
  });
}

// Citation Hover Preview Tooltip
let activeCitationTarget = null;

if (messagesContainer && citationTooltip) {
  messagesContainer.addEventListener("mouseover", (e) => {
    const link = e.target.closest(".citation-link");
    if (!link || link === activeCitationTarget) return;

    activeCitationTarget = link;
    const idx = link.dataset.idx || "";
    const domain = link.dataset.domain || "";
    const title = link.dataset.title || "";
    const snippet = link.dataset.snippet || "";

    if (!title && !domain && !snippet) return;

    if (citationTooltipIdx) citationTooltipIdx.textContent = `[${idx}]`;
    if (citationTooltipDomain) citationTooltipDomain.textContent = domain;
    if (citationTooltipTitle) citationTooltipTitle.textContent = title;
    if (citationTooltipSnippet) {
      citationTooltipSnippet.textContent = snippet;
      citationTooltipSnippet.style.display = snippet ? "block" : "none";
    }

    const rect = link.getBoundingClientRect();
    const tooltipWidth = 320;
    let left = rect.left + rect.width / 2 - tooltipWidth / 2;
    if (left < 16) left = 16;
    if (left + tooltipWidth > window.innerWidth - 16) {
      left = window.innerWidth - tooltipWidth - 16;
    }

    let top = rect.top - 8;
    citationTooltip.style.left = `${Math.round(left)}px`;
    citationTooltip.style.top = `${Math.round(top)}px`;
    citationTooltip.style.transform = "translateY(-100%)";
    citationTooltip.classList.remove("hidden");
  });

  messagesContainer.addEventListener("mouseout", (e) => {
    const link = e.target.closest(".citation-link");
    if (!link) return;
    if (activeCitationTarget === link) {
      activeCitationTarget = null;
      if (citationTooltip) citationTooltip.classList.add("hidden");
    }
  });
}

// ============================================================
// Memories Drawer, Search, Filtering & In-Place Edit
// ============================================================

async function openMemories() {
  memoriesDialog.showModal();
  loadMemories();
}

function closeMemories() {
  memoriesDialog.close();
}

async function loadMemories() {
  memoriesList.innerHTML = `<div class="loading-state" style="padding:16px;text-align:center;color:var(--text-muted)">Loading memories...</div>`;
  try {
    const res = await apiFetch("/api/memories?limit=100");
    if (!res.ok) throw new Error("Failed to load");
    allLoadedMemories = await res.json();
    filterAndRenderMemories();
    updateMemoryCountBadges(allLoadedMemories.length);
  } catch (err) {
    memoriesList.innerHTML = `<div class="error-state" style="padding:16px;color:var(--danger-primary)">Error loading memories.</div>`;
  }
}

function filterAndRenderMemories() {
  const query = (memorySearchInput.value || "").trim().toLowerCase();
  let filtered = allLoadedMemories;

  if (activeMemoryKind !== "all") {
    filtered = filtered.filter((m) => m.kind === activeMemoryKind);
  }

  if (query) {
    filtered = filtered.filter((m) => (m.text || "").toLowerCase().includes(query));
  }

  renderMemoriesList(filtered);
}

memorySearchInput.addEventListener("input", filterAndRenderMemories);

memoryFilterPills.addEventListener("click", (e) => {
  const btn = e.target.closest(".pill-btn");
  if (!btn) return;
  for (const b of memoryFilterPills.querySelectorAll(".pill-btn")) {
    b.classList.remove("active");
  }
  btn.classList.add("active");
  activeMemoryKind = btn.dataset.kind || "all";
  filterAndRenderMemories();
});

function renderMemoriesList(items) {
  memoriesList.innerHTML = "";
  if (!items || items.length === 0) {
    memoriesList.innerHTML = `<div style="padding:20px;text-align:center;color:var(--text-muted);font-size:0.9rem">No matching memories found.</div>`;
    return;
  }

  for (const m of items) {
    const card = document.createElement("div");
    card.className = "memory-card";

    const content = document.createElement("div");
    content.className = "memory-content";

    const text = document.createElement("div");
    text.className = "memory-text";
    text.textContent = m.text;

    const meta = document.createElement("div");
    meta.className = "memory-meta";
    meta.innerHTML = `
      <span class="badge" style="background:var(--bg-elevated);color:var(--text-secondary)">${escapeHTML(m.kind)}</span>
      <span class="badge" style="background:var(--accent-surface);color:var(--accent-primary)">Importance ${m.importance}</span>
      ${m.use_count > 0 ? `<span class="badge" style="background:var(--bg-elevated);color:var(--text-muted)">used ${m.use_count}x</span>` : ""}
    `;

    content.appendChild(text);
    content.appendChild(meta);

    const actions = document.createElement("div");
    actions.className = "memory-actions";
    actions.style.display = "flex";
    actions.style.gap = "4px";

    // Edit memory button
    const editBtn = document.createElement("button");
    editBtn.className = "item-action-btn";
    editBtn.title = "Edit memory";
    editBtn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>`;
    editBtn.addEventListener("click", () => {
      openEditMemory(m);
    });

    // Delete memory button
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "item-action-btn";
    deleteBtn.title = "Delete memory";
    deleteBtn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;

    deleteBtn.addEventListener("click", async () => {
      const ok = await showConfirmDialog({
        title: "Delete Memory",
        message: "Are you sure you want to permanently delete this memory item?",
        itemTitle: m.text,
        confirmText: "Delete",
        cancelText: "Cancel",
        isDanger: true,
      });
      if (!ok) return;
      try {
        const res = await apiFetch(`/api/memories/${m.id}`, { method: "DELETE" });
        if (res.ok) {
          card.remove();
          allLoadedMemories = allLoadedMemories.filter((item) => item.id !== m.id);
          updateMemoryCountBadges(allLoadedMemories.length);
        }
      } catch (err) {
        alert("Failed to delete memory.");
      }
    });

    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);

    card.appendChild(content);
    card.appendChild(actions);
    memoriesList.appendChild(card);
  }
}

// In-place memory edit
function openEditMemory(m) {
  editMemoryId.value = m.id;
  editMemoryText.value = m.text;
  editMemoryImportance.value = String(m.importance);
  editMemoryDialog.showModal();
}

function closeEditMemory() {
  editMemoryDialog.close();
}

editMemoryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = editMemoryId.value;
  const text = editMemoryText.value.trim();
  const importance = parseInt(editMemoryImportance.value, 10);

  if (!id || !text) return;

  try {
    const res = await apiFetch(`/api/memories/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, importance }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Update failed");
    }
    closeEditMemory();
    loadMemories();
  } catch (err) {
    alert("Error updating memory: " + err.message);
  }
});

closeEditMemoryBtn.addEventListener("click", closeEditMemory);
cancelEditMemoryBtn.addEventListener("click", closeEditMemory);

addMemoryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = newMemoryText.value.trim();
  const kind = newMemoryKind.value;
  const importance = parseInt(newMemoryImportance.value, 10);

  if (!text) return;
  memoryFormFeedback.className = "form-feedback hidden";

  try {
    const res = await apiFetch("/api/memories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, kind, importance }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to add memory");
    }

    newMemoryText.value = "";
    memoryFormFeedback.className = "form-feedback success";
    memoryFormFeedback.textContent = data.status || "Saved successfully.";
    loadMemories();
  } catch (err) {
    memoryFormFeedback.className = "form-feedback error";
    memoryFormFeedback.textContent = err.message;
  }
});

function updateMemoryCountBadges(count) {
  if (settingsMemoryCountBadge) {
    settingsMemoryCountBadge.textContent = `${count} ${count === 1 ? "memory" : "memories"} stored`;
  }
  if (settingsClearMemoriesBtn) {
    settingsClearMemoriesBtn.disabled = count === 0;
  }
  if (clearAllMemoriesBtn) {
    clearAllMemoriesBtn.disabled = count === 0;
  }
}

async function updateSettingsMemoryCount() {
  if (!settingsMemoryCountBadge && !clearAllMemoriesBtn && !settingsClearMemoriesBtn) return;
  try {
    const res = await apiFetch("/api/memories?limit=1000");
    if (!res.ok) throw new Error("Failed to load");
    const items = await res.json();
    const count = Array.isArray(items) ? items.length : 0;
    updateMemoryCountBadges(count);
  } catch (err) {
    if (settingsMemoryCountBadge) {
      settingsMemoryCountBadge.textContent = "Memory active";
    }
  }
}

async function clearAllMemories() {
  const confirmed = await showConfirmDialog({
    title: "Clear All Memories",
    message: "Are you sure you want to permanently clear all stored long-term memories? This removes all personalized facts, preferences, and background knowledge across conversations.",
    itemTitle: "All Long-term Memories",
    confirmText: "Clear All",
    cancelText: "Cancel",
    subtext: "This action cannot be undone.",
    isDanger: true,
  });
  if (!confirmed) return;

  try {
    const res = await apiFetch("/api/memories", { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to clear memories");
    }
    const data = await res.json();
    const count = data.deleted_count || 0;
    allLoadedMemories = [];
    filterAndRenderMemories();
    updateMemoryCountBadges(0);
    showToast(`Cleared ${count} ${count === 1 ? "memory" : "memories"}`);
  } catch (err) {
    showToast("Error clearing memories: " + err.message, true);
  }
}

if (clearAllMemoriesBtn) {
  clearAllMemoriesBtn.addEventListener("click", clearAllMemories);
}

if (settingsClearMemoriesBtn) {
  settingsClearMemoriesBtn.addEventListener("click", clearAllMemories);
}

if (settingsOpenMemoriesBtn) {
  settingsOpenMemoriesBtn.addEventListener("click", () => {
    if (settingsDialog.open) settingsDialog.close();
    openMemories();
  });
}

// ============================================================
// Event Listeners, Keyboard Shortcuts & Auto-resize
// ============================================================

// ============================================================
// File Attachments & Drag-and-Drop
// ============================================================

function addAttachedFile(file) {
  if (file.size > SERVER_LIMITS.max_attachment_bytes) {
    alert(
      `File "${file.name}" exceeds the ` +
        `${(SERVER_LIMITS.max_attachment_bytes / 1_000_000).toFixed(1)} MB limit.`
    );
    return;
  }
  const reader = new FileReader();
  reader.onload = async () => {
    const content = String(reader.result ?? "");
    const existing = attachedFiles.reduce((sum, f) => sum + String(f.content).length, 0);
    const pendingText = messageInput.value.trim();
    if (
      existing + content.length + pendingText.length >
      SERVER_LIMITS.max_message_chars
    ) {
      alert(
        `Attaching "${file.name}" would exceed the ` +
          `${SERVER_LIMITS.max_message_chars.toLocaleString()}-character limit.`
      );
      return;
    }
    attachedFiles.push({
      name: file.name,
      content,
      size: file.size,
    });
    renderAttachmentsTray();

    // Also persist to server documents store in background
    try {
      await apiFetch(`/api/documents/upload?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        body: reader.result,
      });
    } catch (_) {}
  };
  reader.readAsText(file);
}

function renderAttachmentsTray() {
  attachmentsTray.innerHTML = "";
  if (attachedFiles.length === 0) {
    attachmentsTray.classList.add("hidden");
    return;
  }
  attachmentsTray.classList.remove("hidden");
  attachedFiles.forEach((file, idx) => {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    chip.innerHTML = `
      <span>${escapeHTML(file.name)}</span>
      <button type="button" class="chip-remove" aria-label="Remove attachment">&times;</button>
    `;
    chip.querySelector(".chip-remove").addEventListener("click", () => {
      attachedFiles.splice(idx, 1);
      renderAttachmentsTray();
    });
    attachmentsTray.appendChild(chip);
  });
}

attachFileBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files) {
    for (const f of e.target.files) {
      addAttachedFile(f);
    }
    fileInput.value = "";
  }
});

composerForm.addEventListener("dragover", (e) => {
  e.preventDefault();
  composerForm.classList.add("drag-over");
});

composerForm.addEventListener("dragleave", (e) => {
  if (!composerForm.contains(e.relatedTarget)) {
    composerForm.classList.remove("drag-over");
  }
});

composerForm.addEventListener("drop", (e) => {
  e.preventDefault();
  composerForm.classList.remove("drag-over");
  if (e.dataTransfer && e.dataTransfer.files) {
    for (const f of e.dataTransfer.files) {
      addAttachedFile(f);
    }
  }
});

function autoResizeTextarea() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 200)}px`;
}

// ============================================================
// Prompt Library & Inline Slash Commands (/) Logic
// ============================================================

const SLASH_COMMANDS = [
  {
    command: "/summarize",
    prompt: "Summarize the key takeaways, decisions, and action items in bullet points.",
    description: "Summarize key takeaways, decisions & action items in bullets"
  },
  {
    command: "/fix",
    prompt: "Review the following code for bugs, edge cases, and performance issues, then provide corrected code:\n\n",
    description: "Review code for bugs, edge cases & performance issues"
  },
  {
    command: "/explain",
    prompt: "Explain this concept in plain English with a simple analogy and practical example:\n\n",
    description: "Explain in plain English with simple analogy & example"
  },
  {
    command: "/critique",
    prompt: "Critique this idea constructively: highlight potential flaws, unstated assumptions, and improvements:\n\n",
    description: "Constructive critique highlighting flaws & improvements"
  },
  {
    command: "/tests",
    prompt: "Write comprehensive unit tests for this code covering standard behavior and edge cases:\n\n",
    description: "Write unit tests covering happy path and edge cases"
  },
  {
    command: "/translate",
    prompt: "Translate the following text into English maintaining original nuance and tone:\n\n",
    description: "Translate text maintaining nuance and tone"
  },
  {
    command: "/rewrite",
    prompt: "Rewrite this text to be more concise, clear, and professional:\n\n",
    description: "Rewrite text to be concise, clear, and professional"
  }
];

let filteredSlashCommands = [];
let activeSlashIndex = 0;

function renderSlashCommands(matches) {
  if (!slashList) return;
  slashList.innerHTML = "";
  matches.forEach((cmd, idx) => {
    const item = document.createElement("div");
    item.className = `slash-item ${idx === activeSlashIndex ? "active" : ""}`;
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", idx === activeSlashIndex ? "true" : "false");
    item.dataset.index = String(idx);
    item.innerHTML = `
      <span class="slash-keyword">${cmd.command}</span>
      <span class="slash-desc">${cmd.description}</span>
    `;
    item.addEventListener("click", () => {
      applySlashCommand(cmd);
    });
    item.addEventListener("mouseenter", () => {
      activeSlashIndex = idx;
      updateSlashActiveHighlight();
    });
    slashList.appendChild(item);
  });
}

function updateSlashActiveHighlight() {
  if (!slashList) return;
  const items = slashList.querySelectorAll(".slash-item");
  items.forEach((item, idx) => {
    const isActive = idx === activeSlashIndex;
    item.classList.toggle("active", isActive);
    item.setAttribute("aria-selected", isActive ? "true" : "false");
    if (isActive) {
      item.scrollIntoView({ block: "nearest" });
    }
  });
}

function openSlashMenu(matches) {
  if (!slashMenu) return;
  filteredSlashCommands = matches;
  activeSlashIndex = 0;
  renderSlashCommands(filteredSlashCommands);
  slashMenu.classList.remove("hidden");
}

function closeSlashMenu() {
  if (!slashMenu) return;
  slashMenu.classList.add("hidden");
  filteredSlashCommands = [];
  activeSlashIndex = 0;
}

function applySlashCommand(cmd) {
  if (!cmd || !messageInput) return;
  messageInput.value = cmd.prompt;
  closeSlashMenu();
  messageInput.focus();
  autoResizeTextarea();
}

function handleSlashInput() {
  if (!messageInput) return;
  const val = messageInput.value;
  if (val.startsWith("/")) {
    const spaceIdx = val.indexOf(" ");
    const newlineIdx = val.indexOf("\n");
    if (spaceIdx === -1 && newlineIdx === -1) {
      const query = val.slice(1).toLowerCase();
      const matches = SLASH_COMMANDS.filter((c) =>
        c.command.slice(1).startsWith(query)
      );
      if (matches.length > 0) {
        openSlashMenu(matches);
        return;
      }
    }
  }
  closeSlashMenu();
}

messageInput.addEventListener("input", () => {
  autoResizeTextarea();
  handleSlashInput();
});

messageInput.addEventListener("keydown", (e) => {
  if (slashMenu && !slashMenu.classList.contains("hidden")) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (filteredSlashCommands.length > 0) {
        activeSlashIndex = (activeSlashIndex + 1) % filteredSlashCommands.length;
        updateSlashActiveHighlight();
      }
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (filteredSlashCommands.length > 0) {
        activeSlashIndex = (activeSlashIndex - 1 + filteredSlashCommands.length) % filteredSlashCommands.length;
        updateSlashActiveHighlight();
      }
      return;
    }
    if (e.key === "Enter" || e.key === "Tab") {
      e.preventDefault();
      if (filteredSlashCommands[activeSlashIndex]) {
        applySlashCommand(filteredSlashCommands[activeSlashIndex]);
      }
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      closeSlashMenu();
      return;
    }
  }

  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const val = messageInput.value.trim();
    if ((val || attachedFiles.length > 0) && !sendBtn.disabled && stopBtn.classList.contains("hidden")) {
      composerForm.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  }
});

document.addEventListener("click", (e) => {
  if (slashMenu && !slashMenu.classList.contains("hidden")) {
    if (!slashMenu.contains(e.target) && e.target !== messageInput) {
      closeSlashMenu();
    }
  }
});

composerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text && attachedFiles.length === 0) return;

  let fullPrompt = text;
  if (attachedFiles.length > 0) {
    let attachmentsText = "";
    for (const file of attachedFiles) {
      attachmentsText += `\n\n<attached_file name="${file.name}">\n${file.content}\n</attached_file>`;
    }
    fullPrompt = text
      ? `${text}\n${attachmentsText}`
      : `Please analyze the attached file(s):${attachmentsText}`;
    attachedFiles = [];
    renderAttachmentsTray();
  }

  if (fullPrompt.length > SERVER_LIMITS.max_message_chars) {
    alert(
      `This message is ${fullPrompt.length.toLocaleString()} characters; the ` +
        `limit is ${SERVER_LIMITS.max_message_chars.toLocaleString()}.`
    );
    return;
  }

  sendMessage(fullPrompt);
});

stopBtn.addEventListener("click", () => {
  if (currentAbortController) {
    currentAbortController.abort();
  }
});

newChatBtn.addEventListener("click", startNewChat);

sidebarToggleBtn.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

themeToggleBtn.addEventListener("click", toggleTheme);
openMemoriesBtn.addEventListener("click", openMemories);
closeMemoriesBtn.addEventListener("click", closeMemories);
retryHealthBtn.addEventListener("click", checkHealth);

openSettingsBtn.addEventListener("click", openSettings);
closeSettingsBtn.addEventListener("click", closeSettings);

openShortcutsBtn.addEventListener("click", () => shortcutsDialog.showModal());
closeShortcutsBtn.addEventListener("click", () => shortcutsDialog.close());

if (forkChatBtn) {
  forkChatBtn.addEventListener("click", () => {
    if (currentConversationId) {
      forkConversationById(currentConversationId, conversationTitle.textContent || "Conversation");
    } else {
      showToast("No active conversation to fork.", true);
    }
  });
}

// Feature 2: Model Switcher & Hardware HUD Listeners
if (modelSelect) {
  modelSelect.addEventListener("change", (e) => {
    handleModelSwitch(e.target.value);
  });
}

if (hardwareHudBtn) {
  hardwareHudBtn.addEventListener("click", () => {
    fetchHardwareTelemetry();
    if (hardwareDialog) hardwareDialog.showModal();
  });
}

if (closeHardwareBtn) {
  closeHardwareBtn.addEventListener("click", () => {
    if (hardwareDialog) hardwareDialog.close();
  });
}

if (refreshHardwareBtn) {
  refreshHardwareBtn.addEventListener("click", () => {
    fetchHardwareTelemetry();
  });
}

// Feature 4: Persona Profiles Listeners
if (profileSelect) {
  profileSelect.addEventListener("change", (e) => {
    selectProfile(e.target.value);
  });
}

if (openProfilesBtn) {
  openProfilesBtn.addEventListener("click", () => {
    renderProfilesGrid();
    if (profilesDialog) profilesDialog.showModal();
  });
}

if (closeProfilesBtn) {
  closeProfilesBtn.addEventListener("click", () => {
    if (profilesDialog) profilesDialog.close();
  });
}

if (newProfileBtn) {
  newProfileBtn.addEventListener("click", () => {
    profileEditorForm.reset();
    profileEditId.value = "";
    profileEditorTitle.textContent = "Create Custom Persona";
    profileEditorForm.classList.remove("hidden");
    profileNameInput.focus();
  });
}

if (cancelProfileEditBtn) {
  cancelProfileEditBtn.addEventListener("click", () => {
    profileEditorForm.classList.add("hidden");
  });
}

if (profileEditorForm) {
  profileEditorForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = profileNameInput.value.trim();
    const icon = profileIconInput.value.trim() || "";
    const mode = profileModeInput.value || "agent";
    const desc = profileDescInput.value.trim();
    const prompt = profilePromptInput.value.trim();

    if (!name || !prompt) {
      alert("Name and System Instructions are required.");
      return;
    }

    try {
      const res = await apiFetch("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          icon,
          mode,
          description: desc,
          system_prompt: prompt,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Failed to save profile: ${err.detail || "Server error"}`);
        return;
      }

      const created = await res.json();
      showToast(`Profile "${created.name}" created!`);
      profileEditorForm.classList.add("hidden");
      await loadProfiles();
      selectProfile(created.id);
    } catch (err) {
      alert(`Error saving profile: ${err.message}`);
    }
  });
}

// ============================================================
// Feature 5: GGUFBook Model Explorer & Downloader
// ============================================================

function formatBytes(bytes) {
  if (bytes == null || isNaN(bytes)) return "Unknown";
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

async function updateGGUFRigBar() {
  try {
    const res = await apiFetch("/api/system/hardware");
    if (!res.ok) return;
    const { gpu, ram } = await res.json();
    if (ggufbookRigVram) {
      if (gpu && gpu.available) {
        ggufbookRigVram.textContent = `${gpu.name || "NVIDIA GPU"} (${gpu.vram_total_mb} MB)`;
      } else {
        ggufbookRigVram.textContent = "No GPU (Host Only)";
      }
    }
    if (ggufbookRigRam) {
      if (ram && ram.available) {
        ggufbookRigRam.textContent = `${ram.total_mb} MB (${ram.used_mb} MB used)`;
      } else {
        ggufbookRigRam.textContent = "Host RAM";
      }
    }
  } catch (err) {
    console.debug("Error updating rig bar:", err);
  }
}

function showGGUFReposView() {
  if (ggufbookReposView) ggufbookReposView.classList.remove("hidden");
  if (ggufbookVariantsView) ggufbookVariantsView.classList.add("hidden");
  currentGgufbookRepo = null;
}

function showGGUFVariantsView() {
  if (ggufbookReposView) ggufbookReposView.classList.add("hidden");
  if (ggufbookVariantsView) ggufbookVariantsView.classList.remove("hidden");
}

async function searchGGUFBook(query = "") {
  if (!ggufbookReposContainer) return;
  ggufbookReposContainer.innerHTML = `<div class="loading-state">Searching Hugging Face for "${escapeHTML(query || "curated")}" GGUFs...</div>`;

  try {
    const res = await apiFetch(`/api/ggufbook/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) {
      const err = await res.json();
      ggufbookReposContainer.innerHTML = `<div class="loading-state" style="color:var(--danger-primary)">Failed to search models: ${escapeHTML(err.detail || "Server error")}</div>`;
      return;
    }

    const repos = await res.json();
    if (!repos || repos.length === 0) {
      ggufbookReposContainer.innerHTML = `<div class="loading-state">No GGUF models found matching "${escapeHTML(query)}". Try another search keyword.</div>`;
      return;
    }

    ggufbookReposContainer.innerHTML = "";
    for (const repo of repos) {
      const card = document.createElement("div");
      card.className = "repo-card";
      card.innerHTML = `
        <div class="repo-card-header">
          <span class="repo-author">${escapeHTML(repo.author || "Community")}</span>
          <span class="repo-name">${escapeHTML(repo.repo_name || repo.id)}</span>
          <p class="repo-desc">${escapeHTML(repo.description || "GGUF quantized repository for local inference.")}</p>
        </div>
        <div class="repo-card-footer">
          <div class="repo-stats">
            ${repo.downloads != null ? `<span>${repo.downloads.toLocaleString()} downloads</span>` : `<span class="curated-badge" style="font-size:0.75rem;padding:2px 8px;border-radius:12px;background:var(--bg-tertiary);color:var(--text-secondary);border:1px solid var(--border-color);">Curated Model</span>`}
            ${repo.likes != null ? `<span>${repo.likes.toLocaleString()} likes</span>` : ""}
          </div>
          <button class="btn btn-outline btn-sm view-variants-btn">Explore</button>
        </div>
      `;

      card.addEventListener("click", () => {
        inspectGGUFRepo(repo.id);
      });

      ggufbookReposContainer.appendChild(card);
    }
  } catch (err) {
    ggufbookReposContainer.innerHTML = `<div class="loading-state" style="color:var(--danger-primary)">Network error: ${escapeHTML(err.message)}</div>`;
  }
}

async function inspectGGUFRepo(repoId) {
  currentGgufbookRepo = repoId;
  showGGUFVariantsView();
  if (variantRepoTitle) variantRepoTitle.textContent = repoId;
  if (variantRepoMeta) variantRepoMeta.textContent = "Scanning Hugging Face LFS files & calculating hardware fit...";
  if (ggufbookVariantsList) {
    ggufbookVariantsList.innerHTML = `<div class="loading-state">Inspecting .gguf files in ${escapeHTML(repoId)}...</div>`;
  }

  try {
    const res = await apiFetch(`/api/ggufbook/models/${encodeURIComponent(repoId)}`);
    if (!res.ok) {
      const err = await res.json();
      ggufbookVariantsList.innerHTML = `<div class="loading-state" style="color:var(--danger-primary)">Failed to inspect model: ${escapeHTML(err.detail || "Server error")}</div>`;
      return;
    }

    const data = await res.json();
    const variants = data.variants || [];
    if (variantRepoMeta) {
      variantRepoMeta.textContent = `${variants.length} GGUF quantization variants detected`;
    }

    if (variants.length === 0) {
      ggufbookVariantsList.innerHTML = `<div class="loading-state">No .gguf files found in this repository.</div>`;
      return;
    }

    renderGGUFVariants(variants, repoId);
  } catch (err) {
    ggufbookVariantsList.innerHTML = `<div class="loading-state" style="color:var(--danger-primary)">Network error: ${escapeHTML(err.message)}</div>`;
  }
}

function renderGGUFVariants(variants, repoId) {
  if (!ggufbookVariantsList) return;
  ggufbookVariantsList.innerHTML = "";

  for (const v of variants) {
    const row = document.createElement("div");
    row.className = "variant-row";

    const fit = v.hardware_fit || {};
    const fitStatus = fit.status || "optimal";
    const fitLabel = fit.label || fit.badge || "Optimal";
    const fitExplanation = fit.explanation || fit.detail || fit.warning || "";

    const isDownloading = activeDownloads.some(
      (task) => task.filename === v.filename && (task.status === "pending" || task.status === "queued" || task.status === "downloading")
    );

    const isDownloaded = Boolean(v.already_downloaded ?? v.is_downloaded);

    let actionButtonHtml = "";
    if (isDownloaded) {
      actionButtonHtml = `<button class="btn btn-sm btn-downloaded" disabled title="Model file already exists in data/models/">Downloaded</button>`;
    } else if (isDownloading) {
      actionButtonHtml = `<button class="btn btn-sm btn-downloading" disabled>Downloading...</button>`;
    } else {
      actionButtonHtml = `<button class="btn btn-primary btn-sm btn-download">Download</button>`;
    }

    row.innerHTML = `
      <div class="variant-left">
        <span class="quant-badge">${escapeHTML(v.quant_type || "GGUF")}</span>
        <div class="variant-info">
          <span class="variant-filename" title="${escapeHTML(v.filename)}">${escapeHTML(v.filename)}</span>
          <span class="variant-size">${v.size_bytes ? formatBytes(v.size_bytes) : "Unknown size"}</span>
        </div>
      </div>
      <div class="variant-right">
        <span class="compat-badge compat-${escapeHTML(fitStatus)}" title="${escapeHTML(fitExplanation)}">
          <span>${escapeHTML(fitLabel)}</span>
        </span>
        ${actionButtonHtml}
      </div>
    `;

    const downloadBtn = row.querySelector(".btn-download");
    if (downloadBtn) {
      downloadBtn.addEventListener("click", () => {
        dispatchGGUFDownload(repoId, v);
      });
    }

    ggufbookVariantsList.appendChild(row);
  }
}

async function dispatchGGUFDownload(repoId, variant) {
  const fit = variant.hardware_fit || {};
  const status = fit.status;

  if (status === "oom_danger") {
    const ok = await showConfirmDialog({
      title: "Hardware Warning: Out of Memory Danger",
      message: fit.explanation || "This model exceeds total host RAM and will likely crash your system or freeze inference.",
      itemTitle: `${variant.filename} (${formatBytes(variant.size_bytes)})`,
      confirmText: "Download Anyway",
      cancelText: "Cancel",
      subtext: "System instability or inference crash may occur if loaded.",
      isDanger: true,
    });
    if (!ok) return;
  } else if (status === "partial_cpu") {
    const ok = await showConfirmDialog({
      title: "Hardware Notice: Partial CPU Offload",
      message: fit.explanation || "Model does not fit fully in GPU VRAM and will partially run on CPU and Host RAM, which is significantly slower.",
      itemTitle: `${variant.filename} (${formatBytes(variant.size_bytes)})`,
      confirmText: "Proceed",
      cancelText: "Cancel",
      subtext: "Inference speeds will be lower compared to full GPU acceleration.",
      isDanger: false,
    });
    if (!ok) return;
  }

  try {
    const res = await apiFetch("/api/ggufbook/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_id: repoId,
        filename: variant.filename,
        download_url: variant.download_url,
        size_bytes: variant.size_bytes,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || "Failed to start download", true);
      alert(`Download Error:\n${data.detail || "Server error"}`);
      return;
    }

    showToast(`Download started for ${variant.filename}`);
    if (currentGgufbookRepo === repoId) {
      await inspectGGUFRepo(repoId);
    }
    await pollGGUFDownloads();
    if (!downloadPollInterval) {
      downloadPollInterval = setInterval(pollGGUFDownloads, 1500);
    }
  } catch (err) {
    showToast(`Download network error: ${err.message}`, true);
  }
}

async function pollGGUFDownloads() {
  try {
    const res = await apiFetch("/api/ggufbook/downloads");
    if (!res.ok) return;
    const tasks = await res.json();
    activeDownloads = tasks;

    const runningTasks = tasks.filter(
      (t) => t.status === "pending" || t.status === "queued" || t.status === "downloading"
    );

    for (const t of tasks) {
      const taskId = t.task_id || t.id;
      if (t.status === "completed" && taskId && !previousCompletedTasks.has(taskId)) {
        previousCompletedTasks.add(taskId);
        showToast(`Model ready: ${t.filename}!`);
        await loadModels();
        if (currentGgufbookRepo) {
          await inspectGGUFRepo(currentGgufbookRepo);
        }
      }
    }

    if (runningTasks.length === 0) {
      if (ggufbookDownloadsDrawer) ggufbookDownloadsDrawer.classList.add("hidden");
      if (ggufbookDownloadsCount) ggufbookDownloadsCount.textContent = "0";
      return;
    }

    if (ggufbookDownloadsDrawer) ggufbookDownloadsDrawer.classList.remove("hidden");
    if (ggufbookDownloadsCount) ggufbookDownloadsCount.textContent = runningTasks.length.toString();

    if (ggufbookDownloadsList) {
      ggufbookDownloadsList.innerHTML = "";
      for (const t of runningTasks) {
        const item = document.createElement("div");
        item.className = "download-item";

        const taskId = t.task_id || t.id;
        const pct = Math.round(t.progress_pct ?? t.percentage ?? 0);
        const downloadedFmt = formatBytes(t.downloaded_bytes || 0);
        const totalBytes = t.total_bytes ?? t.size_bytes ?? 0;
        const totalFmt = formatBytes(totalBytes);
        const speed = t.speed_mb_s ?? t.speed_mbps ?? 0;
        const speedFmt = speed > 0 ? `${Number(speed).toFixed(1)} MB/s` : "Calculating...";
        const etaFmt = t.eta_seconds ? `${t.eta_seconds}s left` : "--";

        item.innerHTML = `
          <div class="download-item-header">
            <span class="download-filename" title="${escapeHTML(t.filename)}">${escapeHTML(t.filename)}</span>
            <button class="cancel-download-btn" data-task-id="${escapeHTML(taskId || "")}" ${!taskId ? "disabled" : ""}>Cancel</button>
          </div>
          <div class="download-bar-wrap">
            <div class="download-bar-fill" style="width: ${pct}%"></div>
          </div>
          <div class="download-meta">
            <span>${pct}% • ${downloadedFmt} / ${totalFmt}</span>
            <span>${speedFmt} • ${etaFmt}</span>
          </div>
        `;

        const cancelBtn = item.querySelector(".cancel-download-btn");
        if (cancelBtn) {
          cancelBtn.addEventListener("click", async () => {
            const rawId = taskId || cancelBtn.dataset.taskId;
            if (!rawId || rawId === "undefined") {
              showToast("Cannot cancel: missing task ID", true);
              return;
            }
            const ok = await showConfirmDialog({
              title: "Cancel Download",
              message: "Are you sure you want to abort this in-progress model download?",
              itemTitle: t.filename,
              confirmText: "Cancel Download",
              cancelText: "Keep Downloading",
              isDanger: true,
            });
            if (!ok) return;
            try {
              const cRes = await apiFetch(`/api/ggufbook/downloads/${encodeURIComponent(rawId)}/cancel`, { method: "POST" });
              if (cRes.ok) {
                showToast(`Download cancelled: ${t.filename}`);
                await pollGGUFDownloads();
                if (currentGgufbookRepo) {
                  await inspectGGUFRepo(currentGgufbookRepo);
                }
              }
            } catch (cErr) {
              alert(`Cancel failed: ${cErr.message}`);
            }
          });
        }

        ggufbookDownloadsList.appendChild(item);
      }
    }
  } catch (err) {
    console.debug("Failed polling downloads:", err);
  }
}

async function openGGUFBook() {
  if (!ggufbookDialog) return;
  ggufbookDialog.showModal();
  showGGUFReposView();
  await updateGGUFRigBar();
  await pollGGUFDownloads();
  if (!downloadPollInterval) {
    downloadPollInterval = setInterval(pollGGUFDownloads, 1500);
  }
  if (ggufbookReposContainer && ggufbookReposContainer.children.length <= 1) {
    await searchGGUFBook("");
  }
}

function closeGGUFBook() {
  if (ggufbookDialog && ggufbookDialog.open) {
    ggufbookDialog.close();
  }
  if (activeDownloads.length === 0 && downloadPollInterval) {
    clearInterval(downloadPollInterval);
    downloadPollInterval = null;
  }
}

if (openGgufbookBtn) {
  openGgufbookBtn.addEventListener("click", openGGUFBook);
}

if (closeGgufbookBtn) {
  closeGgufbookBtn.addEventListener("click", closeGGUFBook);
}

if (backToReposBtn) {
  backToReposBtn.addEventListener("click", showGGUFReposView);
}

if (ggufbookSearchBtn) {
  ggufbookSearchBtn.addEventListener("click", () => {
    if (ggufbookSearchInput) {
      searchGGUFBook(ggufbookSearchInput.value.trim());
    }
  });
}

if (ggufbookSearchInput) {
  ggufbookSearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      searchGGUFBook(ggufbookSearchInput.value.trim());
    }
  });
}

if (ggufbookFilterChips) {
  ggufbookFilterChips.addEventListener("click", (e) => {
    const btn = e.target.closest(".chip-btn");
    if (!btn) return;
    for (const b of ggufbookFilterChips.querySelectorAll(".chip-btn")) {
      b.classList.remove("active");
    }
    btn.classList.add("active");
    const q = btn.dataset.query || "";
    if (ggufbookSearchInput) ggufbookSearchInput.value = q;
    searchGGUFBook(q);
  });
}

// ============================================================
// Voice ASR Microphone Recording & Whistle Integration
// ============================================================

let isRecordingVoice = false;
let audioContext = null;
let audioStream = null;
let audioProcessor = null;
let audioInputSource = null;
let recordedAudioBuffers = [];
let recordingTimerInterval = null;
let recordingStartTimestamp = 0;

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  // RIFF identifier
  writeString(view, 0, "RIFF");
  // file length
  view.setUint32(4, 36 + samples.length * 2, true);
  // RIFF type
  writeString(view, 8, "WAVE");
  // format chunk identifier
  writeString(view, 12, "fmt ");
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw PCM)
  view.setUint16(20, 1, true);
  // channel count (mono)
  view.setUint16(22, 1, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, sampleRate * 2, true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, 2, true);
  // bits per sample
  view.setUint16(34, 16, true);
  // data chunk identifier
  writeString(view, 36, "data");
  // data chunk length
  view.setUint32(40, samples.length * 2, true);

  // Write PCM 16-bit signed integer samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

function cleanupAudioNodes() {
  isRecordingVoice = false;
  if (recordingTimerInterval) {
    clearInterval(recordingTimerInterval);
    recordingTimerInterval = null;
  }
  if (audioProcessor) {
    audioProcessor.disconnect();
    audioProcessor = null;
  }
  if (audioInputSource) {
    audioInputSource.disconnect();
    audioInputSource = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  if (micBtn) {
    micBtn.classList.remove("recording");
    micBtn.title = "Voice Input";
  }
  if (voiceRecordingTray) {
    voiceRecordingTray.classList.add("hidden");
  }
}

async function startVoiceRecording() {
  if (isRecordingVoice) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Microphone recording is not supported in this browser.", true);
    return;
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtx({ sampleRate: 16000 });
    audioInputSource = audioContext.createMediaStreamSource(audioStream);

    audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    recordedAudioBuffers = [];

    audioProcessor.onaudioprocess = (e) => {
      if (!isRecordingVoice) return;
      const inputData = e.inputBuffer.getChannelData(0);
      recordedAudioBuffers.push(new Float32Array(inputData));
    };

    audioInputSource.connect(audioProcessor);
    audioProcessor.connect(audioContext.destination);

    isRecordingVoice = true;
    recordingStartTimestamp = Date.now();

    if (micBtn) {
      micBtn.classList.add("recording");
      micBtn.title = "Stop recording & transcribe";
    }
    if (voiceRecordingTray) {
      voiceRecordingTray.classList.remove("hidden");
    }
    if (voiceRecordingStatus) {
      voiceRecordingStatus.textContent = "Listening...";
    }
    if (voiceRecordingTimer) {
      voiceRecordingTimer.textContent = "00:00";
    }

    recordingTimerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - recordingStartTimestamp) / 1000);
      const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
      const secs = String(elapsedSec % 60).padStart(2, "0");
      if (voiceRecordingTimer) {
        voiceRecordingTimer.textContent = `${mins}:${secs}`;
      }
      if (elapsedSec >= 120) {
        stopVoiceRecording(true);
      }
    }, 500);

  } catch (err) {
    console.error("Microphone access error:", err);
    showToast(`Microphone access error: ${err.message}`, true);
    cleanupAudioNodes();
  }
}

async function stopVoiceRecording(shouldTranscribe = true) {
  if (!isRecordingVoice) return;

  const chunks = recordedAudioBuffers.slice();
  cleanupAudioNodes();

  if (!shouldTranscribe) {
    showToast("Voice recording cancelled.");
    return;
  }

  if (!chunks || chunks.length === 0) {
    showToast("No audio captured.", true);
    return;
  }

  let totalLength = 0;
  for (const chunk of chunks) {
    totalLength += chunk.length;
  }
  const mergedSamples = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    mergedSamples.set(chunk, offset);
    offset += chunk.length;
  }

  if (totalLength < 3200) {
    showToast("Audio clip too short.", true);
    return;
  }

  const wavBlob = encodeWAV(mergedSamples, 16000);
  const lang = localStorage.getItem("knight-asr-language") || "";
  const transcribeUrl = lang ? `/api/asr/transcribe?language=${encodeURIComponent(lang)}` : "/api/asr/transcribe";

  showToast("Transcribing voice input...");

  try {
    const res = await apiFetch(transcribeUrl, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: wavBlob,
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Transcription failed (${res.status})`);
    }

    const data = await res.json();
    const transcript = (data.transcript || "").trim();

    if (!transcript) {
      showToast("No speech recognized.", true);
      return;
    }

    if (messageInput) {
      const existingText = messageInput.value.trim();
      if (existingText) {
        messageInput.value = `${existingText} ${transcript}`;
      } else {
        messageInput.value = transcript;
      }
      messageInput.focus();
      autoResizeTextarea();
    }

    showToast(`Transcribed: "${transcript.length > 40 ? transcript.slice(0, 37) + '...' : transcript}"`);

    const autoSend = localStorage.getItem("knight-asr-autosend") === "true";
    if (autoSend && messageInput && messageInput.value.trim()) {
      composerForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }

  } catch (err) {
    console.error("Transcription error:", err);
    showToast(`Transcription error: ${err.message}`, true);
  }
}

if (micBtn) {
  micBtn.addEventListener("click", () => {
    if (isRecordingVoice) {
      stopVoiceRecording(true);
    } else {
      startVoiceRecording();
    }
  });
}

if (voiceCancelBtn) {
  voiceCancelBtn.addEventListener("click", () => {
    stopVoiceRecording(false);
  });
}

// ============================================================
// In-Chat Search (Ctrl+F) Logic & Controller
// ============================================================

let currentSearchMatches = [];
let currentSearchIndex = -1;

function clearSearchHighlights() {
  if (!messagesContainer) return;
  const highlights = messagesContainer.querySelectorAll("mark.search-highlight");
  for (const mark of highlights) {
    const parent = mark.parentNode;
    if (parent) {
      parent.replaceChild(document.createTextNode(mark.textContent || ""), mark);
      parent.normalize();
    }
  }
  currentSearchMatches = [];
  currentSearchIndex = -1;
  if (inchatSearchCount) {
    inchatSearchCount.textContent = "0 of 0";
  }
}

function updateActiveSearchMatch() {
  for (let i = 0; i < currentSearchMatches.length; i++) {
    currentSearchMatches[i].classList.toggle("active", i === currentSearchIndex);
  }
  if (currentSearchIndex >= 0 && currentSearchIndex < currentSearchMatches.length) {
    const active = currentSearchMatches[currentSearchIndex];
    active.scrollIntoView({ behavior: "smooth", block: "center" });
    if (inchatSearchCount) {
      inchatSearchCount.textContent = `${currentSearchIndex + 1} of ${currentSearchMatches.length}`;
    }
  } else if (inchatSearchCount) {
    inchatSearchCount.textContent = `0 of ${currentSearchMatches.length}`;
  }
}

function nextSearchMatch() {
  if (currentSearchMatches.length === 0) return;
  currentSearchIndex = (currentSearchIndex + 1) % currentSearchMatches.length;
  updateActiveSearchMatch();
}

function prevSearchMatch() {
  if (currentSearchMatches.length === 0) return;
  currentSearchIndex = (currentSearchIndex - 1 + currentSearchMatches.length) % currentSearchMatches.length;
  updateActiveSearchMatch();
}

function performInChatSearch(query) {
  clearSearchHighlights();
  const q = (query || "").trim();
  if (!q || !messagesContainer) {
    return;
  }

  const messageBodies = messagesContainer.querySelectorAll(".message-bubble");
  const lowerQ = q.toLowerCase();
  const matched = [];

  for (const body of messageBodies) {
    const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) => {
        if (!node.nodeValue || !node.nodeValue.toLowerCase().includes(lowerQ)) {
          return NodeFilter.FILTER_SKIP;
        }
        if (node.parentElement && node.parentElement.tagName === "MARK") {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });

    const textNodes = [];
    let textNode;
    while ((textNode = walker.nextNode())) {
      textNodes.push(textNode);
    }

    for (const tn of textNodes) {
      const originalText = tn.nodeValue;
      let startIdx = originalText.toLowerCase().indexOf(lowerQ);
      if (startIdx === -1) continue;

      const frag = document.createDocumentFragment();
      let lastPos = 0;
      while (startIdx !== -1) {
        if (startIdx > lastPos) {
          frag.appendChild(document.createTextNode(originalText.slice(lastPos, startIdx)));
        }
        const mark = document.createElement("mark");
        mark.className = "search-highlight";
        mark.textContent = originalText.slice(startIdx, startIdx + q.length);
        frag.appendChild(mark);
        matched.push(mark);

        lastPos = startIdx + q.length;
        startIdx = originalText.toLowerCase().indexOf(lowerQ, lastPos);
      }
      if (lastPos < originalText.length) {
        frag.appendChild(document.createTextNode(originalText.slice(lastPos)));
      }
      if (tn.parentNode) {
        tn.parentNode.replaceChild(frag, tn);
      }
    }
  }

  currentSearchMatches = matched;
  if (currentSearchMatches.length > 0) {
    currentSearchIndex = 0;
    updateActiveSearchMatch();
  } else if (inchatSearchCount) {
    inchatSearchCount.textContent = "0 of 0";
  }
}

function openInChatSearch() {
  if (!inchatSearchBar) return;
  inchatSearchBar.classList.remove("hidden");
  if (inchatSearchInput) {
    inchatSearchInput.focus();
    inchatSearchInput.select();
    if (inchatSearchInput.value.trim()) {
      performInChatSearch(inchatSearchInput.value);
    }
  }
}

function closeInChatSearch() {
  if (!inchatSearchBar) return;
  inchatSearchBar.classList.add("hidden");
  clearSearchHighlights();
  if (messageInput) {
    messageInput.focus();
  }
}

if (inchatSearchToggleBtn) {
  inchatSearchToggleBtn.addEventListener("click", () => {
    if (inchatSearchBar && !inchatSearchBar.classList.contains("hidden")) {
      closeInChatSearch();
    } else {
      openInChatSearch();
    }
  });
}

if (inchatSearchCloseBtn) {
  inchatSearchCloseBtn.addEventListener("click", closeInChatSearch);
}

if (inchatSearchNextBtn) {
  inchatSearchNextBtn.addEventListener("click", nextSearchMatch);
}

if (inchatSearchPrevBtn) {
  inchatSearchPrevBtn.addEventListener("click", prevSearchMatch);
}

if (inchatSearchInput) {
  let searchDebounceTimer = null;
  inchatSearchInput.addEventListener("input", () => {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
      performInChatSearch(inchatSearchInput.value);
    }, 150);
  });
  inchatSearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (e.shiftKey) {
        prevSearchMatch();
      } else {
        nextSearchMatch();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeInChatSearch();
    }
  });
}

// Global Keyboard Shortcuts
window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    startNewChat();
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "f") {
    e.preventDefault();
    openInChatSearch();
  } else if ((e.ctrlKey || e.metaKey) && e.key === "/") {
    e.preventDefault();
    shortcutsDialog.showModal();
  } else if (e.key === "Escape") {
    if (inchatSearchBar && !inchatSearchBar.classList.contains("hidden")) {
      closeInChatSearch();
      return;
    }
    if (slashMenu && !slashMenu.classList.contains("hidden")) {
      closeSlashMenu();
      return;
    }
    if (isRecordingVoice) {
      stopVoiceRecording(false);
    }
    if (currentAbortController) {
      currentAbortController.abort();
    }
    if (confirmDialog && confirmDialog.open) closeConfirmDialog(false);
    if (exportDialog && exportDialog.open) closeExportDialog();
    if (shortcutsDialog.open) shortcutsDialog.close();
    if (settingsDialog.open) settingsDialog.close();
    if (editMemoryDialog.open) editMemoryDialog.close();
    if (memoriesDialog.open) memoriesDialog.close();
    if (hardwareDialog && hardwareDialog.open) hardwareDialog.close();
    if (profilesDialog && profilesDialog.open) profilesDialog.close();
    if (ggufbookDialog && ggufbookDialog.open) closeGGUFBook();
  }
});

// Close dialogs when backdrop is clicked
for (const dialog of [
  memoriesDialog,
  editMemoryDialog,
  settingsDialog,
  shortcutsDialog,
  hardwareDialog,
  profilesDialog,
  ggufbookDialog,
  confirmDialog,
  exportDialog,
]) {
  if (!dialog) continue;
  dialog.addEventListener("click", (e) => {
    const rect = dialog.getBoundingClientRect();
    const isInDialog =
      rect.top <= e.clientY &&
      e.clientY <= rect.top + rect.height &&
      rect.left <= e.clientX &&
      e.clientX <= rect.left + rect.width;
    if (!isInDialog) {
      if (dialog === ggufbookDialog) {
        closeGGUFBook();
      } else if (dialog === confirmDialog) {
        closeConfirmDialog(false);
      } else if (dialog === exportDialog) {
        closeExportDialog();
      } else {
        dialog.close();
      }
    }
  });
}

// Sidebar close button on mobile
const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
if (sidebarCloseBtn) {
  sidebarCloseBtn.addEventListener("click", () => {
    sidebar.classList.remove("open");
  });
}

// Sidebar Rig Card Click -> Open Hardware Dialog
const sidebarRigCard = document.getElementById("sidebar-rig-card");
if (sidebarRigCard) {
  sidebarRigCard.addEventListener("click", () => {
    fetchHardwareTelemetry();
    if (hardwareDialog) hardwareDialog.showModal();
  });
}

// Quick action suggestion cards on Welcome Hero
for (const card of document.querySelectorAll(".suggestion-card[data-prompt]")) {
  card.addEventListener("click", () => {
    const prompt = card.dataset.prompt;
    if (prompt && messageInput) {
      messageInput.value = prompt;
      messageInput.focus();
      autoResizeTextarea();
    }
  });
}

const suggestionGgufbook = document.getElementById("suggestion-open-ggufbook");
if (suggestionGgufbook) {
  suggestionGgufbook.addEventListener("click", () => {
    openGGUFBook();
  });
}

// Start application
initTheme();
initSettings();
loadConfig();
loadModels();
loadProfiles();
fetchHardwareTelemetry();
checkHealth();
checkASRStatus();
loadConversations();
pollGGUFDownloads();
updateWelcomeHeroVisibility();
setInterval(checkHealth, 20000);
setInterval(fetchHardwareTelemetry, 8000);
