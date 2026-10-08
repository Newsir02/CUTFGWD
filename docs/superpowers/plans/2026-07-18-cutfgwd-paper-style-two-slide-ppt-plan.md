# CUTFGWD Paper-Style Two-Slide PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a two-slide CUTFGWD presentation that inherits the complete white academic style of `path.pptx` while reducing the mathematical content to three core losses and one total objective.

**Architecture:** Duplicate source slides 2 and 3 from the supplied reference deck, preserve their master/layout branding, and edit inherited objects in place. Reuse source-slide shapes for the teacher/student diagram and replace inherited source images with LaTeX-rendered vector equations. Export a distinct PPTX, render both pages, and run template-fidelity and overflow checks.

**Tech Stack:** `@oai/artifact-tool` JavaScript ES modules, PowerPoint PPTX, MiKTeX `latex.exe`, `dvisvgm.exe`, bundled presentation rendering and template-fidelity scripts.

## Global Constraints

- Source reference: `D:\文件\NormalFile\周汇报\path.pptx`.
- Preserve the source file without modification.
- Final output: `D:\文件\CodeFile\CUTFGWD\artifacts\cutfgwd_two_slide_paper_style.pptx`.
- Output exactly two 16:9 slides.
- Output slide 1 duplicates source slide 2; output slide 2 duplicates source slide 3.
- Preserve the white background, Southwest University logo, title-arrow language, gray divider, blue footer, typography, and spacing conventions.
- Slide 1 displays only the framework and `L_KD`, `L_W1`, and `L_CUT-FGW` labels.
- Slide 2 displays only the approved KD, W1, CUT-FGW, and total-loss equations.
- Do not show CE expansion, softmax definitions, CDF construction, FGW intermediate costs, Sinkhorn updates, diversity expansion, derivations, or explanatory paragraphs.
- No Git commit because `D:\文件\CodeFile\CUTFGWD` is not a Git repository.

---

### Task 1: Create the source-deck inspection and template map

**Files:**
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\template-audit.txt`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\template-inspect\template-inspect.ndjson`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\template-frame-map.json`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\deviation-log.txt`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\inspect-reference.mjs`

**Interfaces:**
- Consumes: `path.pptx` and source-slide XML evidence for slides 1–5.
- Produces: a complete rendered/layout inventory and a validated two-slide duplication map.

- [ ] **Step 1: Copy the reference to an ASCII temporary path**

Copy `D:\文件\NormalFile\周汇报\path.pptx` to `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\reference.pptx`. This avoids the Windows `unzip` helper's Unicode-path limitation while preserving the original.

- [ ] **Step 2: Inspect all five slides with artifact-tool**

Create `inspect-reference.mjs` that imports `reference.pptx`, iterates every slide, writes a PNG and layout JSON, and writes `presentation.inspect({ kind: "slide,textbox,shape,image,layout", maxChars: 100000 })` to `template-inspect.ndjson`. Assert the imported deck contains five slides.

- [ ] **Step 3: Record the source styling contract**

Write `template-audit.txt` with these verified rules: source slide 2 is the architecture pattern; source slide 3 is the equation pattern; the logo/footer are inherited through the source layout; white background and navy/cyan title language are preserved; source content objects are rewritten, replaced, or deleted explicitly.

- [ ] **Step 4: Write the frame map**

Create `template-frame-map.json` with exactly these mappings:

```json
{
  "outputSlides": [
    {
      "outputSlide": 1,
      "sourceSlide": 2,
      "narrativeRole": "framework content",
      "reuseMode": "duplicate-slide",
      "editTargets": [
        { "shapeId": "23", "action": "rewrite-and-reposition" },
        { "shapeId": "24", "action": "rewrite-and-reposition" },
        { "shapeIds": ["44", "45", "46", "47", "48", "49"], "action": "rewrite-and-reposition" },
        { "shapeIds": ["174", "175", "176", "177", "178", "179"], "action": "rewrite-and-reposition" },
        { "shapeId": "143", "action": "rewrite" }
      ]
    },
    {
      "outputSlide": 2,
      "sourceSlide": 3,
      "narrativeRole": "mathematical content",
      "reuseMode": "duplicate-slide",
      "editTargets": [
        { "shapeId": "3", "action": "delete" },
        { "shapeIds": ["78", "79"], "action": "rewrite-and-reposition" },
        { "shapeIds": ["4", "5", "13", "16"], "action": "replace" },
        { "shapeIds": ["30", "34", "40", "59"], "action": "rewrite-and-reposition" }
      ]
    }
  ],
  "omittedSourceSlides": [
    { "sourceSlide": 1, "reason": "introduction page is outside the two-slide scope" },
    { "sourceSlide": 4, "reason": "experiment page is outside the two-slide scope" },
    { "sourceSlide": 5, "reason": "code-results page is outside the two-slide scope" }
  ]
}
```

If artifact-tool reports stable element IDs different from the OOXML shape IDs above, rewrite the map to the reported stable IDs before validation; the mapping roles and source slides remain unchanged.

- [ ] **Step 5: Validate the map**

Run `validate_template_plan.mjs` with the custom inspection file. Expected result: status `pass` and zero unresolved edit targets.

### Task 2: Create the two-slide starter deck and formula assets

**Files:**
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\template-starter.pptx`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\formula-kd.svg`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\formula-w1.svg`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\formula-cutfgw.svg`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\formula-total.svg`

**Interfaces:**
- Consumes: the validated frame map and the four approved LaTeX equations.
- Produces: a two-slide inherited starter and four transparent black vector equations.

- [ ] **Step 1: Duplicate source slides 2 and 3**

Run `prepare_template_starter_deck.mjs` using `reference.pptx` and `template-frame-map.json`. Expected result: `template-starter.pptx` contains exactly two slides, with source logos, footer, page chrome, and layout preserved.

- [ ] **Step 2: Render the four approved equations**

Use MiKTeX `latex.exe` and `dvisvgm.exe --no-fonts` to render these exact equations as transparent SVG files:

```tex
\mathcal L_{\mathrm{KD}}=T^2D_{\mathrm{KL}}(p^T\Vert p^S)
\mathcal L_{\mathrm{W1}}=\sum_{k=1}^{C-1}w_k\left|F_k^T-F_k^S\right|
\mathcal L_{\mathrm{CUT-FGW}}=\frac{\langle\Pi^\star,C_{\mathrm{FGW}}(\Pi^\star)\rangle}{\lVert\Pi^\star\rVert_1}
\mathcal L_{\mathrm{Total}}=\mathcal L_{\mathrm{Task}}+\lambda_1\mathcal L_{\mathrm{KD}}+\lambda_2\mathcal L_{\mathrm{W1}}+\lambda_3\mathcal L_{\mathrm{CUT-FGW}}+\lambda_4\mathcal L_{\mathrm{div}}
```

Expected result: all four SVG files have transparent backgrounds, black glyphs, visible superscripts/subscripts, and no text braces.

### Task 3: Edit the inherited slides in place

**Files:**
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\build-paper-style.mjs`
- Create: `D:\文件\CodeFile\CUTFGWD\artifacts\cutfgwd_two_slide_paper_style.pptx`

**Interfaces:**
- Consumes: `template-starter.pptx`, formula SVGs, and resolved inherited object IDs.
- Produces: the final two-slide paper-style deck.

- [ ] **Step 1: Import the starter deck**

Use `PresentationFile.importPptx(await FileBlob.load(starterPath))`. Assert exactly two slides and resolve every mapped edit target before changing content.

- [ ] **Step 2: Rewrite slide 1**

Delete source-paper-specific content that is not mapped for reuse. Reposition inherited teacher and student shapes into a left-to-right layout, rewrite labels to `TGN Teacher` and `Graph-free Student`, and use three inherited middle shapes for `L_KD`, `L_W1`, and `L_CUT-FGW`. Keep the source slide's top `模型架构` ribbon, university logo, white background, footer, and thin gray rule.

- [ ] **Step 3: Rewrite slide 2**

Delete the source SmartArt title and reuse inherited navy/cyan shapes to form the `CUTFGWD 蒸馏目标` title ribbon. Replace four inherited source images with the KD, W1, CUT-FGW, and total-loss SVGs. Rewrite four inherited text boxes to `Logit KD`, `Rank-Wasserstein`, `CUT-FGW`, and `Total Loss`; use the source page's academic spacing and black formula treatment.

- [ ] **Step 4: Export final evidence**

Export the final PPTX, two PNG previews, two layout JSON files, an inspect NDJSON file, and a montage. Assert the final presentation contains exactly two slides.

### Task 4: Verify template fidelity and presentation quality

**Files:**
- Verify: `D:\文件\CodeFile\CUTFGWD\artifacts\cutfgwd_two_slide_paper_style.pptx`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\final-render\slide-1.png`
- Create: `%TEMP%\codex-presentations\manual-20260718\cutfgwd-paper-style\tmp\final-render\slide-2.png`

**Interfaces:**
- Consumes: final PPTX and starter deck.
- Produces: structural, visual, and fidelity evidence.

- [ ] **Step 1: Render and inspect each slide**

Run `render_slides.py` at 1920×1080. Inspect both slides individually for formula clarity, inherited logo/footer visibility, title hierarchy, unwanted source-paper remnants, clipping, and overlap.

- [ ] **Step 2: Run overflow checks**

Run `slides_test.py`. Expected result: exit code 0 and `Test passed. No overflow detected.`

- [ ] **Step 3: Verify page count and placeholder state**

Inspect the final PPTX archive and assert exactly two `ppt/slides/slide*.xml` files. Fail if any source-specific label such as `NCE`, `Dual-Grained`, `Share Sampling_path`, or `Ditisll Sapce` remains visible.

- [ ] **Step 4: Run template fidelity**

Run `check_template_fidelity.mjs` with the starter PPTX, final PPTX, frame map, starter layouts, and final layouts. Expected result: no unplanned deletion of the university logo, footer, title chrome, or inherited page furniture.

- [ ] **Step 5: Iterate until all checks pass**

If any check fails, patch `build-paper-style.mjs`, rebuild, rerender both pages, and repeat the full verification suite before delivery.
