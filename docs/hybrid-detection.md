# Hybrid detection architecture

The hybrid implementation layers optional audio transitions, trusted local
fingerprints, on-screen ad-label/countdown OCR, and a lightweight trained
classifier on top of the existing visual detector. Legacy mode remains the
default and retains its original behavior.

## Existing pipeline and persistence

The control path is `FrameCapture` (OpenCV) → detector pipeline → `Engine`
(the single commercial-break state owner) → `PlaybackController` (Apple TV
mute/skip). FastAPI reads telemetry and invokes engine controls; it does not
independently change playback state.

Channel profiles are JSON plus NumPy/image assets under `profiles/`. Pairing,
the last active session, and crash-time mute ownership use local files. Manual
labels still tune visual thresholds and now also write grouped feature rows to
`data/features.sqlite3`. Runtime dashboard settings are not written back to
`config.py`.

Video uses V4L2 on Linux and AVFoundation on macOS. Hybrid audio probes
FFmpeg/AVFoundation and remains unavailable unless the configured input exists.

## Modes and detector contract

- `legacy` uses the original logo/color/combo classification, debounce,
  hysteresis, re-entry cooldown, and break decay.
- `hybrid` fuses enabled detectors, then uses the same engine transitions and
  Apple TV intervention path. There is no second mute/skip state machine.

Select a mode in the dashboard, set `DETECTION_MODE`, or pass
`detection_mode` to `POST /api/start`. Evaluate hybrid mode in Shadow first.

Every detector returns `DetectorResult`: detector name; `program`, `commercial`,
or `unknown`; detector-local confidence and kind; timestamp; availability;
diagnostics; native raw score; and optional calibrated `P(COMMERCIAL)`. Fusion
never directly compares raw scores. Disabled, missing, stale, and failed
detectors return `unknown`.

The visual adapter preserves the existing detector's temporal smoothing. Fusion
has separate enter/exit thresholds. Logo absence and audio transitions are
non-decisive and require corroboration. Color/combo evidence and validated
fingerprints can be decisive. ML is supporting evidence.

## macOS Monterey audio setup

Install FFmpeg and inspect inputs:

```bash
brew install ffmpeg
ffmpeg -hide_banner -f avfoundation -list_devices true -i ""
```

Do not assume an MS2130 exposes audio. Find its actual audio input name, then:

```bash
export DETECTION_MODE=hybrid
export AUDIO_TRANSITION_ENABLED=true
export AUDIO_DEVICE_NAME="USB Digital Audio"
python3 server.py
```

The reader and analyzer use background threads, a four-item bounded queue, and
a bounded PCM ring. Features are RMS, short-term silence, loudness change,
dynamic range, spectral centroid, spectral flux, and 13 MFCCs. Status reports
availability, capture latency, backlog, and dropped chunks. PCM is not written
to disk.

## Trusted fingerprints

Enable `AUDIO_FINGERPRINT_ENABLED=true`. Confirm a commercial, then use
**Enroll recent audio** or:

```json
POST /api/fingerprints/enroll
{"name":"Example spot","seconds":10,"metadata":{"campaign":"example"}}
```

Enrollment is always explicit. SQLite stores only metadata, compact spectral
tokens, and offsets. Listing, deletion, duplicate checks, match confidence,
advertisement ID, matched duration, and estimated position are supported.
Matching runs outside the video loop. `FingerprintAlgorithm` makes the default
matcher replaceable by Chromaprint, audfprint, or another implementation.

The implementation choice was intentionally conservative: [Chromaprint](https://acoustid.org/chromaprint)
is a maintained native fingerprint extractor but leaves application-specific
local matching/storage to the caller; [Dejavu](https://github.com/worldveil/dejavu)
ships recognition but is oriented around MySQL/Postgres; and
[audfprint](https://github.com/dpwe/audfprint) is a capable landmark system with
a larger specialized toolchain. The built-in NumPy/SQLite algorithm keeps the
initial deployment small while the interface preserves a migration path.

## On-screen ad labels and countdowns

Hybrid mode can use Tesseract OCR to flag stable player UI such as **Ad 1 of
2**, **Advertisement 0:15**, **Commercial break**, or **Skip ad**. Enable
**Ad labels & countdowns** in Advanced settings. The detector scans the upper
and lower screen bands once per second, requires repeated recognition, and
briefly holds a confirmed flag through an OCR dropout.

A timer by itself is never commercial evidence because sports, news, and game
shows contain legitimate clocks. Likewise, **Sponsored** alone is only a
diagnostic candidate. Even a confirmed explicit ad label is supporting—not
decisive—evidence, so the default Hybrid policy still requires another source
such as logo absence, a trusted fingerprint, audio transition, or ML result.

Install Tesseract manually when not using a deployment script:

```bash
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Debian/Ubuntu
```

## Collecting and training ML data

Install optional training dependencies:

```bash
python3 -m pip install -r requirements-ml.txt
```

Use **Mark Content** and **Mark Commercial** across multiple recordings and
channels. Each marking window gets a distinct session ID. The feature schema
contains visual scores, audio energy/dynamics/spectral/MFCC values, scene cuts,
and bounded temporal history.

Export and train:

```bash
python3 scripts/export_feature_dataset.py --output data/features.csv
python3 scripts/train_commercial_model.py \
  --dataset data/features.csv \
  --output data/commercial_model.joblib
```

Training uses standardized logistic regression with Platt calibration, suitable
for an older Intel CPU. Fit, calibration, and test partitions are separated by
source/session groups, preventing adjacent-frame leakage. The artifact includes
the schema, version, dataset digest, group assignments, precision, and recall.
No pretrained model or accuracy claim is shipped. Missing, corrupt, and
schema-incompatible models safely return `unknown`.

## Evaluation

Prepare `timestamp,label,legacy,hybrid` CSV data and explicitly identify it:

```bash
python3 scripts/evaluate_detection.py \
  --input evaluation.csv \
  --data-kind captured
```

Reports include precision/recall, false-muted programming seconds per hour,
commercial audio muted percentage, mute/unmute latency, transition counts, and
legacy-versus-hybrid performance. Simulated results are labeled as simulated.

## Dashboard and APIs

The dashboard provides mode selection, detector toggles, per-detector/fused
diagnostics, ad-overlay text/countdown flags, audio/fingerprint/model
availability, shadow comparison, recent actions and reasons, fingerprint
management, labels, and dataset export.

Relevant APIs include `/api/status`, `/api/settings`, `/api/fingerprints`,
`/api/fingerprints/enroll`, and `/api/ml/dataset/export`.

## Safety and recovery

Missing/blank/sync frames cannot start a hybrid commercial transition. Commands
are idempotent, controller calls reconnect once, maximum mute duration remains
enforced, Disarm starts a manual-override cooldown, and graceful shutdown
restores engine-owned volume. A persisted mute marker repairs an interrupted
mute on the next successful start without changing the existing configured
volume-step behavior.

## Limitations and next improvements

- Logo-only hybrid mode intentionally stays `unknown` on logo absence until a
  second trustworthy signal exists; legacy mode remains available.
- Ad-overlay OCR requires explicit ad wording and corroboration. Platform UI
  that uses only an unlabeled countdown remains diagnostic-only by design.
- The default fingerprint is optimized for repeated digital HDMI audio. Test
  transcoding/loudness-normalization robustness; replace it if field data needs
  a more mature algorithm.
- Audio capture currently implements macOS AVFoundation; Linux needs an
  ALSA/PulseAudio command builder.
- ML quality depends entirely on representative grouped user labels.
- Automated tests use synthetic audio/features; the MS2130 audio path and
  real-world precision/recall still require shadow-mode captured-data trials.
- Power loss while Apple TV is unreachable can still require manual recovery.
