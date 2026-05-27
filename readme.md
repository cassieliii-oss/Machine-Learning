# ML Final Project

This workspace contains the materials for our `Machine Learning 2` final project:
an end-to-end system that identifies which game appears in an image, both from
clean screenshots and from real-world photos of a screen.

## Final Entry Point

The **official final project repository** in this workspace is:

- `screenshot-to-game-classifier-main/`

This is the version to use for:

- the final presentation
- the GitHub showcase
- the live demo
- the final code walkthrough

## What Each Folder Is For

- `screenshot-to-game-classifier-main/`: final integrated repository with the
  17-class pipeline, trained model, results, assets, and demo.
- `pre_project/`: early reference prototype.
- `Final_project_Data/`: local reference copy related to earlier data-stage
  work; keep for comparison and backup, not as the final repo entry point.
- `Project Report.pdf`: project proposal/report context and original goals.
- `docs/`: integration, presentation, and demo materials prepared for the
  final submission.

## Project Summary

We are building a computer vision system that can:

1. classify a direct gameplay screenshot
2. detect a screen from a phone photo or monitor photo
3. correct the perspective of that detected screen
4. classify the corrected game image

The final integrated version currently centers on:

- a 17-class game classifier
- a reproducible data pipeline
- a Gradio demo with screenshot mode and photo-of-screen mode
- evaluation results and presentation-ready assets

## How To Run The Final Demo

From the final repo root:

```bash
cd "screenshot-to-game-classifier-main"
python demo/app.py
```

The demo opens locally at `http://localhost:7860`.

## Presentation Package

The final presentation is designed as a three-part package rather than slides
only:

1. `slides`: structured storytelling and speaker flow
2. `GitHub page`: project completeness, engineering depth, and reproducibility
3. `web demo`: live proof that the system works

Supporting materials live in:

- `docs/presentation_strategy.md`
- `docs/slides_prompt.md`
- `docs/demo_runbook.md`
- `docs/presentation_assets.md`
- `docs/risk_register.md`

## Notes For Final Presentation

- Present the project as an **end-to-end photo-of-screen recognition system**,
  not just as a classifier.
- Do **not** present knowledge distillation as a completed component unless the
  corresponding implementation is later added to the repo.
- Use `screenshot-to-game-classifier-main/README.md` as the main GitHub page to
  show during presentation.