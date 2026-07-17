# Variational Neural Inference Group Talk

This directory contains a 60-minute, English-language research-group presentation.
It begins with probability, latent variables, gradients, and optimization before
developing mixture models, HMMs, variational inference, VAEs, LFADS, and gpSLDS.

## Files

- `Variational_Neural_Inference_Group_Talk.pptx`: editable 16:9 PowerPoint deck
- `Variational_Neural_Inference_Group_Talk.pdf`: rendered presentation
- `Variational_Neural_Inference_Speaker_Notes.md`: English notes and timing cue for every slide

The 40 slide timing cues sum to 58 minutes and 40 seconds. No discussion block is
reserved. The remaining time covers natural transitions and brief pauses.

## Rebuild

Run the following command from the repository root:

```bash
python scripts/build_group_presentation.py
```

The build script extracts validated figures from the executed tutorial notebooks
and embeds the same English notes in the PowerPoint notes pane.
