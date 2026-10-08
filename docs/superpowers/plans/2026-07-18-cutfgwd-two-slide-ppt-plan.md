# CUTFGWD Mathematical Two-Slide PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `presentations:Presentations` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing two-slide deck with a projector-readable mathematical version that shows the complete project-aligned distillation objective without derivations.

**Architecture:** Rebuild both slides with `@oai/artifact-tool` in one JavaScript ES module. Slide 1 uses a left-to-right teacher/channel/student mechanism; slide 2 uses a full-width joint objective, two upper formula groups, one dominant CUT-FGW formula group, and a diversity footer. Export editable text and native shapes, then render and validate the PPTX.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, PowerPoint PPTX, bundled presentation render and QA scripts.

## Global Constraints

- Final path: `artifacts/cutfgwd_two_slide_presentation.pptx`.
- Exactly two 1280×720 slides.
- Dark-blue custom style matching the earlier deck.
- All formulas are editable text in Cambria Math; Chinese uses Microsoft YaHei.
- No derivations, explanatory paragraphs, experiment results, or unsupported claims.
- Minimum visible formula/body size: 16 pt; slide title: at least 35 pt.
- No Git commit because the workspace is not a Git repository.

---

### Task 1: Update the mathematical content model

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-cutfgwd-two-slide-design.md`
- Modify: `docs/superpowers/plans/2026-07-18-cutfgwd-two-slide-ppt-plan.md`

**Interfaces:**
- Consumes: formulas in `loss/distillation.py`, `loss/kd.py`, and `loss/cutfgw.py`.
- Produces: the exact two-slide formula inventory and layout requirements.

- [ ] Confirm slide 1 contains compact KD, W1, and CUT-FGW formulas.
- [ ] Confirm slide 2 contains joint, CE, KD, W1, FGW cost, Sinkhorn, and diversity formulas.
- [ ] Search both documents for `TBD|TODO|placeholder`; expected result: no matches.

### Task 2: Rebuild the presentation

**Files:**
- Modify: `%TEMP%/codex-presentations/manual-20260718/cutfgwd-two-slide/tmp/build.mjs`
- Create: `%TEMP%/codex-presentations/manual-20260718/cutfgwd-two-slide/tmp/source-notes.txt`
- Replace: `artifacts/cutfgwd_two_slide_presentation.pptx`

**Interfaces:**
- Consumes: exact formula strings and the existing navy palette.
- Produces: a two-slide editable `Presentation` object and final PPTX.

- [ ] Refactor the slide helper functions so labels and formulas can use separate typefaces and font sizes.
- [ ] Rebuild slide 1 with teacher and student modules plus three central formula bands.
- [ ] Rebuild slide 2 with the joint objective at the top; CE/KD and W1 above; CUT-FGW/Sinkhorn below; diversity in the footer.
- [ ] Export both slide PNG previews, layout JSON, inspect NDJSON, montage, and the final PPTX.
- [ ] Assert `presentation.slides.items.length === 2` before export.

### Task 3: Render and verify

**Files:**
- Verify: `artifacts/cutfgwd_two_slide_presentation.pptx`
- Create: `%TEMP%/codex-presentations/manual-20260718/cutfgwd-two-slide/tmp/independent-render/slide-1.png`
- Create: `%TEMP%/codex-presentations/manual-20260718/cutfgwd-two-slide/tmp/independent-render/slide-2.png`

**Interfaces:**
- Consumes: final PPTX.
- Produces: visual and structural evidence that the deck is readable and bounded.

- [ ] Run `render_slides.py` on the final PPTX; expected result: exactly two PNG files.
- [ ] Inspect both slides at full size for wrapping, clipping, contrast, and unintended overlaps.
- [ ] Run `slides_test.py`; expected result: exit code 0 and no overflow errors.
- [ ] Inspect the PPTX archive and assert exactly two `ppt/slides/slide*.xml` members.
- [ ] If any check fails, patch `build.mjs`, rebuild, and repeat all verification commands.
