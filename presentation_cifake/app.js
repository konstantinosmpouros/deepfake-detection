(() => {
  "use strict";

  // ---------------- Data ----------------
  const sections = [
    { key: "problem",       name: "Πρόβλημα" },
    { key: "data",          name: "Δεδομένα & Features" },
    { key: "architectures", name: "Αρχιτεκτονικές" },
    { key: "results",       name: "Αποτελέσματα" },
    { key: "fewshot",       name: "Few-shot" },
    { key: "conclusions",   name: "Συμπεράσματα" },
  ];

  const architectures = [
    { id: "classical", name: "Κλασικοί αλγόριθμοι", meta: "k-NN · NB · LogReg · SVM · DT", score: "0.906",
      family: "Baseline · handcrafted & CNN features",
      subtitle: "Feature extraction (HOG/LBP/Color/CNN) → PCA→150 → κλασικός ταξινομητής με CV.",
      layers: "HOG / LBP / Color / ResNet18-features\n  → StandardScaler → PCA(150)\n  → {k-NN, NaiveBayes, LogReg, SVM-rbf, DecisionTree}",
      note: "Καλύτερος handcrafted: HOG+SVM 0.765. Καλύτερα CNN features: SVM+CNN 0.906 — η αναπαράσταση κάνει τη διαφορά." },
    { id: "mlp", name: "MLP (raw pixels)", meta: "1.71M params", score: "0.755",
      family: "Baseline · fully-connected",
      subtitle: "Πλήρως συνδεδεμένο δίκτυο πάνω σε flattened pixels — αγνοεί τη χωρική δομή.",
      layers: "flatten(3072)\n  → Linear(3072→512) → BN → ReLU → Dropout(0.3)\n  → Linear(512→256)  → BN → ReLU → Dropout(0.3)\n  → Linear(256→2)",
      note: "Ισοδύναμο με handcrafted classical· σαφώς κάτω από CNN. Η πολυπλοκότητα δεν δικαιολογείται σε raw pixels." },
    { id: "cnn", name: "Custom CNN", meta: "356k params", score: "0.933",
      family: "From scratch · convolutional",
      subtitle: "3 conv blocks σε 32×32 — locality, weight sharing, translation invariance.",
      layers: "[Conv→BN→ReLU→MaxPool] ×3  (32→64→128)\n  → Flatten → Dropout → Linear(→128) → ReLU → Linear(→2)",
      note: "0.933 με ~5× λιγότερες παραμέτρους από το MLP. Το data augmentation εδώ έβλαψε (αλλοιώνει τα artifacts)." },
    { id: "dual", name: "Dual-Branch Artifact-Aware CNN", meta: "169k params", score: "0.935",
      family: "Hybrid · RGB + frequency/residual",
      subtitle: "Spatial branch (RGB) + frequency branch (high-pass, SRM, FFT) με gated fusion.",
      layers: "Branch A (RGB):  3 conv blocks → 96-d\nBranch B (artifacts): high-pass + SRM×3 + FFT → 3 conv blocks → 96-d\nFusion: concat(192) → gated → Linear→128 → Linear→2",
      note: "Ισοφαρίζει το απλό CNN (0.935 vs 0.933) με τις μισές παραμέτρους — στα 32×32 τα frequency features έχουν λίγο σήμα να εκμεταλλευτούν." },
    { id: "transfer", name: "Transfer · ResNet18 / EfficientNet-B0", meta: "224px · ImageNet", score: "0.958",
      family: "Transfer learning",
      subtitle: "Pretrained backbones, full fine-tuning vs frozen. Αντικατάσταση μόνο του head.",
      layers: "ImageNet backbone (ResNet18 / EfficientNet-B0)\n  → replace head → Linear(→2)\n  full: 0.942 / 0.958   ·   frozen: 0.852 / 0.865",
      note: "Full fine-tuning κερδίζει ~9 μονάδες από το frozen. Το freezing μειώνει παραμέτρους, όχι όμως τον wall-clock χρόνο." },
    { id: "prototype", name: "Prototype classifier", meta: "frozen ResNet18", score: "few-shot",
      family: "Few-shot · metric-based",
      subtitle: "Μέσος όρος embeddings ανά κλάση· ταξινόμηση με πλησιέστερο prototype.",
      layers: "frozen ResNet18 (fc=Identity) → 512-d\n  prototype_c = mean(support embeddings)\n  argmin distance {Euclidean, Cosine}",
      note: "Euclidean ≈ Cosine. Έχασε από το fine-tuning στο few-shot — ο ImageNet encoder δεν είναι εξειδικευμένος στο real/fake." },
    { id: "simclr", name: "SimCLR", meta: "self-supervised", score: "0.909",
      family: "Self-supervised · contrastive",
      subtitle: "ResNet18 encoder εκπαιδευμένος με NT-Xent· linear-eval & fine-tune.",
      layers: "encoder ResNet18 → 512-d\n  projection head: 512→512→128\n  NT-Xent (τ=0.5), 2 augmented views",
      note: "Linear 0.868 · fine-tune 0.909. Ανταγωνιστικό στο 10-shot (0.703)· το 5-shot ασταθές (0.516)." },
    { id: "vit", name: "ViT-B/16 + LoRA", meta: "85.8M / 0.30M", score: "0.962 / 0.957",
      family: "Transfer + PEFT",
      subtitle: "Full fine-tuning έναντι LoRA (r=8 στις query/value projections).",
      layers: "ViT-B/16 (patch16, 12 layers, hidden 768)\n  full: 85.8M trainable (100%) → 0.962\n  LoRA r=8 (q/v): 0.30M trainable (0.34%) → 0.957",
      note: "Το LoRA χάνει μόλις 0.5 μονάδα εκπαιδεύοντας 0.34% των παραμέτρων — χαμηλό intrinsic rank της προσαρμογής." },
    { id: "probe", name: "CLIP / DINOv2 probe", meta: "frozen foundation", score: "0.947 / 0.945",
      family: "Foundation-model · frozen + head",
      subtitle: "Frozen CLIP ViT-B/32 (512-d) ή DINOv2 ViT-S/14 (384-d) + μικρό head.",
      layers: "frozen encoder (CLIP / DINOv2) → embedding\n  → StandardScaler → MLP head → 2 logits\n  (linear probe: LogisticRegression)",
      note: "Καλύτερο no-fine-tuning (0.947) και κορυφαίο few-shot (10-shot 0.743). Σχεδόν γραμμικά διαχωρίσιμα — η αναπαράσταση κωδικοποιεί ήδη το σήμα." },
  ];

  const resultsRows = [
    ["ViT-B/16 — full fine-tune", "Transfer + PEFT", "0.962", false],
    ["EfficientNet-B0 — full", "Transfer learning", "0.958", false],
    ["LoRA (ViT-B/16, r=8)", "Transfer + PEFT", "0.957", false],
    ["CLIP-probe (frozen + head)", "Foundation · frozen", "0.947", true],
    ["DINOv2-probe (frozen + head)", "Foundation · frozen", "0.945", true],
    ["ResNet18 — full", "Transfer learning", "0.942", false],
    ["Dual-Branch Artifact-Aware CNN", "Hybrid · RGB+freq", "0.935", true],
    ["Custom CNN (from scratch)", "From scratch", "0.933", false],
    ["Best classical (SVM + CNN feats)", "Baseline", "0.906", false],
    ["MLP on CNN features", "Baseline", "0.900", false],
    ["Best handcrafted (HOG + SVM)", "Baseline", "0.765", false],
    ["MLP (raw pixels)", "Baseline", "0.755", false],
  ];

  const fewshotRows = [
    ["CLIP-probe", "0.743", "0.675", true],
    ["DINOv2-probe", "0.718", "0.650", true],
    ["ResNet18 / full", "0.710", "0.619", false],
    ["SimCLR fine-tuned", "0.703", "0.516", false],
    ["EfficientNet / full", "0.668", "0.513", false],
    ["Prototype (ResNet18)", "0.610", "0.583", false],
  ];

  const findings = [
    ["1", "Η αναπαράσταση κυριαρχεί", "Handcrafted ~0.76 → ImageNet features ~0.91 → foundation embeddings ~0.945–0.947 → full fine-tune 0.962. Η βελτίωση έρχεται από καλύτερες αναπαραστάσεις, όχι πιο σύνθετους ταξινομητές."],
    ["2", "Foundation probes = αποδοτικότητα", "Frozen CLIP/DINOv2 + μικρό head: σχεδόν κορυφαία ακρίβεια full-data και τα καλύτερα few-shot, με μηδενική εκπαίδευση backbone. Το CLIP προηγείται και είναι πιο σταθερό."],
    ["3", "Το LoRA επιβεβαιώνει το low-rank", "0.957 με 0.34% των παραμέτρων — πρακτικά ισόπαλο με το full fine-tuning (0.962). Η προσαρμογή του ViT σε αυτό το task έχει χαμηλό intrinsic rank."],
    ["4", "Artifact-aware ισοφαρίζει, δεν ξεπερνά", "Το Dual-Branch πέτυχε 0.935 — ίσο με το CNN (0.933) αλλά με τις μισές παραμέτρους. Στα 32×32 τα explicit frequency features προσφέρουν περιορισμένο επιπλέον σήμα."],
    ["5", "Το few-shot είναι δύσκολο", "Όλες οι μέθοδοι πέφτουν στο 5-shot (0.51–0.68). Αναδεικνύονται οι data-efficient προσεγγίσεις — με πρώτο το CLIP-probe και μετά το ResNet18 (αντί του μεγαλύτερου EfficientNet)."],
    ["6", "Αντι-διαισθητικά & διακύμανση", "Το data augmentation έβλαψε το CNN (αλλοιώνει artifacts)· ο prototype έχασε από το fine-tuning· το SimCLR 5-shot κατέρρευσε — τα few-shot νούμερα θέλουν πολλαπλά seeds."],
  ];

  // ---------------- Rendering ----------------
  const $ = (sel, ctx = document) => ctx.querySelector(sel);

  function renderArchitectures() {
    const list = $("#arch-list");
    list.innerHTML = architectures.map((a, i) => `
      <button class="arch-item${i === 0 ? " active" : ""}" type="button" role="option" data-arch="${a.id}" aria-selected="${i === 0}">
        <span class="ai-name">${a.name}</span>
        <span class="ai-meta">${a.family}</span>
        <span class="ai-score">${a.score}</span>
      </button>`).join("");
    renderArchDetail(architectures[0].id);
    list.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-arch]");
      if (!btn) return;
      list.querySelectorAll(".arch-item").forEach((b) => {
        const on = b === btn; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on));
      });
      renderArchDetail(btn.dataset.arch);
    });
  }

  function renderArchDetail(id) {
    const a = architectures.find((x) => x.id === id);
    if (!a) return;
    $("#arch-detail").innerHTML = `
      <p class="eyebrow">${a.family}</p>
      <h1>${a.name}</h1>
      <p>${a.subtitle}</p>
      <div class="arch-meta-row">
        <div><span>Test accuracy</span><strong>${a.score}</strong></div>
        <div><span>Σημείωση</span><strong style="font-size:14px;font-family:Inter,sans-serif;font-weight:600">${a.meta}</strong></div>
      </div>
      <div class="arch-layers">${a.layers}</div>
      <div class="arch-note"><p>${a.note}</p></div>`;
  }

  function renderResults() {
    $("#results-table").innerHTML = `
      <table class="data">
        <thead><tr><th>Μέθοδος</th><th>Οικογένεια</th><th class="num">Full-data acc</th></tr></thead>
        <tbody>${resultsRows.map(([m, f, s, hi]) => `
          <tr class="${hi ? "highlight" : ""}"><td>${m}</td><td>${f}</td><td class="num">${s}</td></tr>`).join("")}
        </tbody>
      </table>`;
  }

  function renderFewshot() {
    $("#fewshot-table").innerHTML = `
      <table class="data">
        <thead><tr><th>Μέθοδος</th><th class="num">10-shot</th><th class="num">5-shot</th></tr></thead>
        <tbody>${fewshotRows.map(([m, a, b, hi]) => `
          <tr class="${hi ? "highlight" : ""}"><td>${m}</td><td class="num">${a}</td><td class="num">${b}</td></tr>`).join("")}
        </tbody>
      </table>`;
  }

  function renderFindings() {
    $("#findings").innerHTML = findings.map(([n, t, p]) => `
      <div class="finding"><div class="fnum">${n}</div><h3>${t}</h3><p>${p}</p></div>`).join("");
  }

  // ---------------- Navigation ----------------
  function setView(key, updateHash = true) {
    if (!sections.some((s) => s.key === key)) key = "problem";
    document.querySelectorAll(".view").forEach((view) => {
      const active = view.dataset.view === key;
      view.classList.toggle("active", active);
      view.setAttribute("aria-hidden", String(!active));
      if (active) { const sc = view.querySelector(".view-scroll"); if (sc) sc.scrollTop = 0; }
    });
    document.querySelectorAll(".nav-tab[data-view-target]").forEach((b) => {
      const active = b.dataset.viewTarget === key;
      b.classList.toggle("active", active);
      if (active) b.scrollIntoView({ block: "nearest", inline: "center" });
    });
    const idx = sections.findIndex((s) => s.key === key);
    $("#section-index").textContent = String(idx + 1).padStart(2, "0");
    $("#section-name").textContent = sections[idx].name;
    if (updateHash) history.replaceState(null, "", `#${key}`);
  }

  function step(delta) {
    const current = sections.findIndex((s) => s.key === (location.hash.slice(1) || "problem"));
    const next = Math.min(sections.length - 1, Math.max(0, current + delta));
    setView(sections[next].key);
  }

  // ---------------- Image zoom ----------------
  const imageDialog = $("#image-dialog");
  const dialogImage = $("#dialog-image");
  function openImage(img) {
    dialogImage.src = img.currentSrc || img.src;
    dialogImage.alt = img.alt || "";
    if (typeof imageDialog.showModal === "function") imageDialog.showModal();
  }
  $("#image-close").addEventListener("click", () => imageDialog.close());
  imageDialog.addEventListener("click", (e) => { if (e.target === imageDialog) imageDialog.close(); });
  document.querySelectorAll("img.zoomable").forEach((img) => {
    img.tabIndex = 0; img.setAttribute("role", "button");
  });

  // ---------------- Events ----------------
  document.addEventListener("click", (e) => {
    const viewBtn = e.target.closest("[data-view-target]");
    if (viewBtn) { setView(viewBtn.dataset.viewTarget); return; }
    const zoom = e.target.closest("img.zoomable");
    if (zoom) openImage(zoom);
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("img.zoomable") && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openImage(e.target); return; }
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key.toLowerCase() === "f") $("#fullscreen").click();
  });

  $("#previous-section").addEventListener("click", () => step(-1));
  $("#next-section").addEventListener("click", () => step(1));
  $("#fullscreen").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else document.documentElement.requestFullscreen?.();
  });
  document.addEventListener("fullscreenchange", () => {
    $("#fullscreen").title = document.fullscreenElement ? "Έξοδος από πλήρη οθόνη" : "Πλήρης οθόνη";
  });
  window.addEventListener("hashchange", () => setView(location.hash.slice(1) || "problem", false));

  // ---------------- Init ----------------
  renderArchitectures();
  renderResults();
  renderFewshot();
  renderFindings();
  setView(location.hash.slice(1) || "problem", false);
})();
