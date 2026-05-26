# SigVamp Memory

## Current Working State

- `sigvamp_loop.py` is the main stable performance script.
- Held-note polyphonic looping works well.
- Auto-root detection is working nicely and makes chromatic playback feel musical.
- `sigvamp_grain.py` is the experimental time-stable granular pitch script.
- Granular audio works again after Pi stability tuning, but polyphony and tone still need refinement.

## Current Control Map

- `CC112`: loop start
- `CC114`: loop end
- `CC74`: gain
- `CC18`: attack
- `CC16` / `CC19`: release

## Future Ideas

### Granular Tone And Smoothness

- The granular script still sounds a little too cyclic.
- The start/end of the perceived grain or loop cycle can click or pulse strongly.
- Explore per-grain amplitude envelopes, similar to modern granular synths.
- Possible controls:
  - grain attack
  - grain release
  - grain envelope blend or shape
- Goal: make grains feel more natural, less obviously looped or mechanically periodic.

### Shorter Buffer And Finer Loop Control

- The current 3 second recording buffer is probably too long.
- Try reducing the buffer duration to around 1 second.
- Shorter buffer should make loop start/end controls feel more precise.
- Goal: support picking out tiny waveform regions or even near-single-cycle material.

### Zero-Crossing Or Endpoint Smoothing

- Investigate ways to smooth loop boundaries.
- Idea: force or interpolate loop endpoints toward zero before playback/grain generation.
- Another option: add a short loop crossfade at wrap points.
- Goal: reduce clicks when cycling tight waveform windows.

### More Gain Range

- Current gain max of `4.0` is still too soft.
- Increase max gain substantially.
- Consider whether gain should be linear or more musical/exponential.
- Keep clipping protection, but allow much hotter output when needed.

## Next Good Development Pass

1. Reduce capture duration from 3 seconds to 1 second in an experimental branch or script.
2. Increase gain ceiling.
3. Add endpoint smoothing or short loop crossfade to `sigvamp_loop.py`.
4. Add per-grain envelope shaping to `sigvamp_grain.py`.
5. Re-test on the Pi with the SP-404MKII as the audio interface.
