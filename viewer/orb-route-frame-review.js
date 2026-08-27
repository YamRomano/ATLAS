(() => {
  const state = { data: null, filtered: [], page: 0, pageSize: 96 };
  const $ = (selector) => document.querySelector(selector);
  const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 });

  function metric(value, label) {
    const node = document.createElement("div");
    node.className = "metric";
    node.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
    return node;
  }

  function applyFilters() {
    const source = $("#sourceFilter").value;
    const leg = $("#legFilter").value;
    const frame = $("#frameFilter").value.trim().replace(/^0+/, "");
    state.filtered = state.data.anchors.filter((anchor) => {
      const sourceOk = source === "all" || anchor.source_replay_id === source;
      const legOk = leg === "all" || anchor.leg === Number(leg);
      const frameOk = !frame || String(anchor.source_frame).includes(frame);
      return sourceOk && legOk && frameOk;
    });
    state.page = 0;
    render();
  }

  function setPage(nextPage) {
    const pageCount = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
    state.page = Math.max(0, Math.min(pageCount - 1, nextPage));
    render();
    window.scrollTo({ top: $("#resultCount").offsetTop - 80, behavior: "smooth" });
  }

  function render() {
    const gallery = $("#gallery");
    gallery.replaceChildren();
    const first = state.page * state.pageSize;
    const visible = state.filtered.slice(first, first + state.pageSize);
    const template = $("#cardTemplate");
    for (const anchor of visible) {
      const card = template.content.firstElementChild.cloneNode(true);
      const link = card.querySelector(".image-link");
      const image = card.querySelector("img");
      link.href = anchor.image_url;
      image.src = anchor.image_url;
      image.alt = `ORB anchor frame ${anchor.source_frame}, leg ${anchor.from_point} to ${anchor.to_point}`;
      card.querySelector(".card-title strong").textContent = `Frame ${String(anchor.source_frame).padStart(6, "0")}`;
      const chip = card.querySelector(".leg-chip");
      chip.textContent = anchor.live_reference_enabled
        ? `${anchor.from_point} → ${anchor.to_point} · LIVE`
        : `${anchor.from_point} → ${anchor.to_point} · STORED ONLY`;
      if (!anchor.live_reference_enabled) card.classList.add("stored-only");
      card.querySelector(".progress-track i").style.width = `${Math.max(0, Math.min(100, anchor.progress * 100))}%`;
      card.querySelector(".progress-value").textContent = `${(anchor.progress * 100).toFixed(2)}%`;
      card.querySelector(".descriptor-value").textContent = anchor.descriptor_count.toLocaleString();
      card.querySelector(".source-value").textContent = anchor.source_title;
      card.querySelector(".source-value").title = anchor.source_replay_id;
      card.querySelector(".index-value").textContent = anchor.anchor_index.toLocaleString();
      gallery.append(card);
    }
    const pageCount = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
    const label = `Page ${state.page + 1} of ${pageCount}`;
    $("#resultCount").textContent = `${state.filtered.length.toLocaleString()} exact ORB anchors · showing ${visible.length.toLocaleString()}`;
    $("#pageLabel").textContent = label;
    $("#pageLabelBottom").textContent = label;
    for (const id of ["#previousPage", "#previousPageBottom"]) $(id).disabled = state.page === 0;
    for (const id of ["#nextPage", "#nextPageBottom"]) $(id).disabled = state.page >= pageCount - 1;
  }

  async function boot() {
    const response = await fetch("/public/orb-route-frame-index.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Frame index request failed: ${response.status}`);
    state.data = await response.json();
    state.filtered = state.data.anchors;
    $("#summary").textContent = `${state.data.live_reference_anchor_count.toLocaleString()} live-enabled anchors + ${state.data.stored_audit_only_anchor_count.toLocaleString()} stored audit-only · ${state.data.descriptor_count.toLocaleString()} ORB descriptors · ${state.data.sources.length} source recordings · bank: ${state.data.bank.split("/").at(-1)}`;
    const metrics = $("#metrics");
    metrics.append(metric(state.data.anchor_count.toLocaleString(), "exact frames"));
    metrics.append(metric(compact.format(state.data.descriptor_count), "ORB descriptors"));
    for (let leg = 1; leg <= 4; leg += 1) metrics.append(metric(Number(state.data.per_leg[leg]).toLocaleString(), `leg ${leg} → ${(leg % 4) + 1}`));
    for (const source of state.data.sources) {
      const option = document.createElement("option");
      option.value = source.id;
      option.textContent = `${source.title} · ${Number(state.data.per_source[source.id]).toLocaleString()} frames`;
      $("#sourceFilter").append(option);
    }
    render();
  }

  $("#sourceFilter").addEventListener("change", applyFilters);
  $("#legFilter").addEventListener("change", applyFilters);
  $("#frameFilter").addEventListener("input", applyFilters);
  $("#pageSize").addEventListener("change", (event) => { state.pageSize = Number(event.target.value); state.page = 0; render(); });
  $("#previousPage").addEventListener("click", () => setPage(state.page - 1));
  $("#previousPageBottom").addEventListener("click", () => setPage(state.page - 1));
  $("#nextPage").addEventListener("click", () => setPage(state.page + 1));
  $("#nextPageBottom").addEventListener("click", () => setPage(state.page + 1));
  boot().catch((error) => { $("#summary").textContent = error.message; console.error(error); });
})();
