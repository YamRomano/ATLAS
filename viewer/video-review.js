(() => {
  "use strict";

  const video = document.querySelector("#review-video");
  const timeline = document.querySelector("#timeline");
  const currentTime = document.querySelector("#current-time");
  const durationTime = document.querySelector("#duration-time");
  const playToggle = document.querySelector("#play-toggle");
  const frameSurface = document.querySelector("#frame-surface");
  const frameMarker = document.querySelector("#frame-marker");
  const rangeOverlay = document.querySelector("#range-overlay");
  const startLabel = document.querySelector("#start-label");
  const endLabel = document.querySelector("#end-label");
  const markerLabel = document.querySelector("#marker-label");
  const noteInput = document.querySelector("#note");
  const categoryInput = document.querySelector("#category");
  const priorityInput = document.querySelector("#priority");
  const formError = document.querySelector("#form-error");
  const addButton = document.querySelector("#add-annotation");
  const formTitle = document.querySelector("#form-title");
  const cancelEdit = document.querySelector("#cancel-edit");
  const annotationList = document.querySelector("#annotation-list");
  const emptyNotes = document.querySelector("#empty-notes");
  const annotationCount = document.querySelector("#annotation-count");
  const template = document.querySelector("#annotation-template");
  const saveState = document.querySelector("#save-state");
  const saveLabel = document.querySelector("#save-label");
  const noteFilter = document.querySelector("#note-filter");
  const loopSelection = document.querySelector("#loop-selection");

  const state = {
    annotations: [],
    start: 0,
    end: null,
    point: null,
    editingId: null,
    selectedId: null,
    loopId: null,
    duration: 120,
    savingTimer: null,
  };

  const categoryLabels = {
    visual: "Visual",
    edit: "Edit & pacing",
    music: "Music / sound",
    text: "Text / message",
    technical: "Technical",
    other: "Other",
  };

  function formatTime(seconds) {
    const totalMs = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
    const minutes = Math.floor(totalMs / 60000);
    const wholeSeconds = Math.floor((totalMs % 60000) / 1000);
    const milliseconds = totalMs % 1000;
    return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
  }

  function setSaveStatus(kind, message) {
    saveState.dataset.state = kind;
    saveLabel.textContent = message;
  }

  function updatePlayhead() {
    const duration = Number.isFinite(video.duration) ? video.duration : state.duration;
    state.duration = duration || 120;
    timeline.max = String(state.duration);
    timeline.value = String(video.currentTime || 0);
    currentTime.textContent = formatTime(video.currentTime);
    durationTime.textContent = formatTime(state.duration);
    playToggle.textContent = video.paused ? "▶ Play" : "❚❚ Pause";
  }

  function renderMarker() {
    if (!state.point) {
      frameMarker.classList.add("hidden");
      markerLabel.textContent = "No frame position selected";
      document.querySelector("#clear-marker").disabled = true;
      return;
    }
    frameMarker.classList.remove("hidden");
    frameMarker.style.left = `${state.point.x * 100}%`;
    frameMarker.style.top = `${state.point.y * 100}%`;
    markerLabel.textContent = `Marked at ${Math.round(state.point.x * 100)}% × ${Math.round(state.point.y * 100)}% of frame`;
    document.querySelector("#clear-marker").disabled = false;
  }

  function renderTiming() {
    startLabel.textContent = formatTime(state.start);
    endLabel.textContent = state.end == null ? "Single moment" : formatTime(state.end);
    if (state.end == null || !state.duration) {
      rangeOverlay.style.display = "none";
      return;
    }
    const left = Math.min(100, Math.max(0, (state.start / state.duration) * 100));
    const right = Math.min(100, Math.max(left, (state.end / state.duration) * 100));
    rangeOverlay.style.display = "block";
    rangeOverlay.style.left = `${left}%`;
    rangeOverlay.style.width = `${Math.max(0.2, right - left)}%`;
  }

  function resetForm({ keepPlayhead = true } = {}) {
    state.start = keepPlayhead ? video.currentTime || 0 : 0;
    state.end = null;
    state.point = null;
    state.editingId = null;
    formTitle.textContent = "Add annotation";
    addButton.textContent = "Add Annotation";
    cancelEdit.classList.add("hidden");
    noteInput.value = "";
    categoryInput.value = "visual";
    priorityInput.value = "P2";
    formError.textContent = "";
    renderTiming();
    renderMarker();
  }

  function annotationTime(item) {
    return item.end == null ? formatTime(item.start) : `${formatTime(item.start)}–${formatTime(item.end)}`;
  }

  function selectAnnotation(item, { seek = true } = {}) {
    state.selectedId = item.id;
    if (seek) {
      video.currentTime = item.start;
      video.pause();
    }
    state.point = item.point ? { ...item.point } : null;
    renderMarker();
    loopSelection.disabled = item.end == null;
    loopSelection.textContent = state.loopId === item.id ? "Stop Loop" : "Loop Selected Range";
    renderList();
  }

  function editAnnotation(item) {
    selectAnnotation(item);
    state.editingId = item.id;
    state.start = item.start;
    state.end = item.end;
    state.point = item.point ? { ...item.point } : null;
    categoryInput.value = item.category;
    priorityInput.value = item.priority;
    noteInput.value = item.note;
    formTitle.textContent = "Edit annotation";
    addButton.textContent = "Save Annotation";
    cancelEdit.classList.remove("hidden");
    renderTiming();
    renderMarker();
    noteInput.focus();
  }

  function renderList() {
    const filter = noteFilter.value;
    annotationList.replaceChildren();
    const visible = state.annotations.filter((item) => filter === "all" || item.priority === filter);
    emptyNotes.classList.toggle("hidden", state.annotations.length > 0);
    annotationCount.textContent = String(state.annotations.length);
    for (const item of visible) {
      const fragment = template.content.cloneNode(true);
      const row = fragment.querySelector(".annotation-item");
      row.dataset.id = item.id;
      row.classList.toggle("selected", item.id === state.selectedId);
      fragment.querySelector(".note-time").textContent = annotationTime(item);
      const priority = fragment.querySelector(".note-priority");
      priority.textContent = item.priority;
      priority.dataset.priority = item.priority;
      fragment.querySelector(".note-category").textContent = categoryLabels[item.category] || "Other";
      fragment.querySelector(".note-copy").textContent = item.note;
      fragment.querySelector(".note-position").textContent = item.point
        ? `◎ ${Math.round(item.point.x * 100)}%, ${Math.round(item.point.y * 100)}%`
        : "";
      fragment.querySelector(".note-main").addEventListener("click", () => selectAnnotation(item));
      fragment.querySelector(".edit-note").addEventListener("click", () => editAnnotation(item));
      fragment.querySelector(".delete-note").addEventListener("click", async () => {
        state.annotations = state.annotations.filter((entry) => entry.id !== item.id);
        if (state.selectedId === item.id) state.selectedId = null;
        if (state.loopId === item.id) state.loopId = null;
        if (state.editingId === item.id) resetForm();
        renderList();
        await persistReview();
      });
      annotationList.append(fragment);
    }
  }

  function reviewPayload() {
    return {
      version: 1,
      video: "ATLAS_CINEMATIC_PRODUCT_STORY_2MIN.mp4",
      annotations: state.annotations,
    };
  }

  async function persistReview() {
    window.clearTimeout(state.savingTimer);
    setSaveStatus("saving", "Saving notes…");
    try {
      const response = await fetch("/api/video-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reviewPayload()),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Could not save review.");
      state.annotations = payload.review.annotations || [];
      localStorage.setItem("atlas-video-review", JSON.stringify(payload.review));
      renderList();
      setSaveStatus("saved", `Saved ${state.annotations.length} note${state.annotations.length === 1 ? "" : "s"} to project`);
      return payload.review;
    } catch (error) {
      localStorage.setItem("atlas-video-review", JSON.stringify(reviewPayload()));
      setSaveStatus("error", `Project save failed · kept in browser`);
      throw error;
    }
  }

  async function loadReview() {
    setSaveStatus("loading", "Loading notes…");
    try {
      const response = await fetch("/api/video-review");
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Could not load review.");
      state.annotations = payload.review.annotations || [];
      setSaveStatus("saved", `${state.annotations.length} saved note${state.annotations.length === 1 ? "" : "s"}`);
    } catch (error) {
      try {
        const cached = JSON.parse(localStorage.getItem("atlas-video-review") || "{}");
        state.annotations = Array.isArray(cached.annotations) ? cached.annotations : [];
        setSaveStatus("error", "Offline notes loaded from browser");
      } catch {
        state.annotations = [];
        setSaveStatus("error", "Could not load saved notes");
      }
    }
    renderList();
  }

  function buildBrief() {
    const lines = [
      "# ATLAS Cinematic Product Story — Adjustment Brief",
      "",
      `Video: ATLAS_CINEMATIC_PRODUCT_STORY_2MIN.mp4`,
      `Annotations: ${state.annotations.length}`,
      "",
    ];
    state.annotations.forEach((item, index) => {
      const point = item.point ? ` · frame position ${(item.point.x * 100).toFixed(1)}% x, ${(item.point.y * 100).toFixed(1)}% y` : "";
      lines.push(`## ${index + 1}. ${annotationTime(item)} · ${item.priority} · ${categoryLabels[item.category] || "Other"}`);
      lines.push("");
      lines.push(`${item.note}${point}`);
      lines.push("");
    });
    return lines.join("\n");
  }

  function seekBy(delta) {
    video.pause();
    video.currentTime = Math.min(state.duration, Math.max(0, video.currentTime + delta));
  }

  video.addEventListener("loadedmetadata", () => {
    state.duration = video.duration || 120;
    state.start = video.currentTime || 0;
    updatePlayhead();
    renderTiming();
  });
  video.addEventListener("timeupdate", () => {
    if (state.loopId) {
      const item = state.annotations.find((entry) => entry.id === state.loopId);
      if (!item || item.end == null) {
        state.loopId = null;
      } else if (video.currentTime >= item.end) {
        video.currentTime = item.start;
        video.play().catch(() => {});
      }
    }
    updatePlayhead();
  });
  video.addEventListener("play", updatePlayhead);
  video.addEventListener("pause", updatePlayhead);
  video.addEventListener("ended", updatePlayhead);

  frameSurface.addEventListener("click", (event) => {
    const rect = frameSurface.getBoundingClientRect();
    state.point = {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    };
    video.pause();
    renderMarker();
  });

  timeline.addEventListener("input", () => {
    video.currentTime = Number(timeline.value);
    video.pause();
    updatePlayhead();
  });
  playToggle.addEventListener("click", () => video.paused ? video.play() : video.pause());
  document.querySelector("#back-five").addEventListener("click", () => seekBy(-5));
  document.querySelector("#next-five").addEventListener("click", () => seekBy(5));
  document.querySelector("#back-frame").addEventListener("click", () => seekBy(-1 / 24));
  document.querySelector("#next-frame").addEventListener("click", () => seekBy(1 / 24));
  document.querySelectorAll("[data-speed]").forEach((button) => button.addEventListener("click", () => {
    video.playbackRate = Number(button.dataset.speed);
    document.querySelectorAll("[data-speed]").forEach((item) => item.classList.toggle("active", item === button));
  }));

  document.querySelector("#set-start").addEventListener("click", () => {
    state.start = video.currentTime;
    if (state.end != null && state.end < state.start) state.end = null;
    renderTiming();
  });
  document.querySelector("#set-end").addEventListener("click", () => {
    if (video.currentTime < state.start) {
      formError.textContent = "End time must be after the start time.";
      return;
    }
    state.end = video.currentTime;
    formError.textContent = "";
    renderTiming();
  });
  document.querySelector("#clear-end").addEventListener("click", () => { state.end = null; renderTiming(); });
  document.querySelector("#clear-marker").addEventListener("click", () => { state.point = null; renderMarker(); });
  cancelEdit.addEventListener("click", () => resetForm());

  addButton.addEventListener("click", async () => {
    const note = noteInput.value.trim();
    if (!note) {
      formError.textContent = "Describe the change before adding the annotation.";
      noteInput.focus();
      return;
    }
    const existing = state.editingId && state.annotations.find((item) => item.id === state.editingId);
    const annotation = {
      id: existing?.id || (crypto.randomUUID ? crypto.randomUUID() : `note-${Date.now()}`),
      start: Number(state.start.toFixed(3)),
      end: state.end == null ? null : Number(state.end.toFixed(3)),
      category: categoryInput.value,
      priority: priorityInput.value,
      note,
      point: state.point ? { x: Number(state.point.x.toFixed(5)), y: Number(state.point.y.toFixed(5)) } : null,
      created_at: existing?.created_at || new Date().toISOString(),
    };
    if (existing) {
      state.annotations = state.annotations.map((item) => item.id === existing.id ? annotation : item);
    } else {
      state.annotations.push(annotation);
    }
    state.annotations.sort((a, b) => a.start - b.start);
    state.selectedId = annotation.id;
    renderList();
    try {
      await persistReview();
      resetForm();
    } catch (error) {
      formError.textContent = error.message;
    }
  });

  document.querySelector("#save-review").addEventListener("click", async () => {
    try { await persistReview(); } catch (error) { formError.textContent = error.message; }
  });
  document.querySelector("#copy-brief").addEventListener("click", async (event) => {
    try {
      await navigator.clipboard.writeText(buildBrief());
      const old = event.currentTarget.textContent;
      event.currentTarget.textContent = "Brief Copied";
      window.setTimeout(() => { event.currentTarget.textContent = old; }, 1600);
    } catch {
      formError.textContent = "Clipboard access was blocked. Use Download JSON instead.";
    }
  });
  document.querySelector("#download-review").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(reviewPayload(), null, 2)], { type: "application/json" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = "ATLAS_CINEMATIC_PRODUCT_STORY_review.json";
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  });
  noteFilter.addEventListener("change", renderList);
  loopSelection.addEventListener("click", () => {
    const selected = state.annotations.find((item) => item.id === state.selectedId);
    if (!selected || selected.end == null) return;
    state.loopId = state.loopId === selected.id ? null : selected.id;
    loopSelection.textContent = state.loopId ? "Stop Loop" : "Loop Selected Range";
    if (state.loopId) {
      video.currentTime = selected.start;
      video.play().catch(() => {});
    }
  });

  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") {
      event.preventDefault();
      video.paused ? video.play() : video.pause();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      seekBy(event.shiftKey ? -5 : -1 / 24);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      seekBy(event.shiftKey ? 5 : 1 / 24);
    }
  });

  resetForm({ keepPlayhead: false });
  loadReview();
})();
