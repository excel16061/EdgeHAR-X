# Dataset Mapping: UCF101 ↔ HMDB51 (12 classes)

Status: frozen for Phase 2B. The canonical machine-readable copy is
`TA3N_MAPPING` in `benchmark_manifest.py`.

## 1. Source

The mapping is transcribed from **Chen et al., "Temporal Attentive Alignment
for Large-Scale Video Domain Adaptation" (TA3N), ICCV 2019, arXiv:1907.12743,
§7.4.1 and Table 13**. It was read on 2026-11-01 via
https://ar5iv.labs.arxiv.org/html/1907.12743.

Verbatim from §7.4.1: *"We collect all of the relevant and overlapping
categories between UCF101 and HMDB51, which results in 12 categories: climb,
fencing, golf, kick_ball, pullup, punch, pushup, ride_bike, ride_horse,
shoot_ball, shoot_bow, and walk. Each category may correspond to multiple
categories in the original UCF101 or HMDB51 dataset, as shown in Table 13."*

The mapping was **not** chosen by this project. It is adopted unchanged from a
published protocol, so the class set and correctness rule are not
researcher-chosen.

## 2. Mapping

| Category = HMDB51 label | UCF101 label(s) (TA3N Table 13) | Strict label (secondary analysis) | UCF101 index (classInd, 1-based) |
|---|---|---|---|
| climb | RockClimbingIndoor, RopeClimbing | RockClimbingIndoor | 74, 75 |
| fencing | Fencing | Fencing | 28 |
| golf | GolfSwing | GolfSwing | 33 |
| kick_ball | SoccerPenalty | SoccerPenalty | 85 |
| pullup | PullUps | PullUps | 70 |
| punch | Punch, BoxingPunchingBag, BoxingSpeedBag | Punch | 71, 17, 18 |
| pushup | PushUps | PushUps | 72 |
| ride_bike | Biking | Biking | 11 |
| ride_horse | HorseRiding | HorseRiding | 42 |
| shoot_ball | Basketball | Basketball | 8 |
| shoot_bow | Archery | Archery | 3 |
| walk | WalkingWithDog | WalkingWithDog | 98 |

The indices are computed by `verify_mapping()` and stored in
`manifests/manifest_summary.json`.

**Independent verification** (`benchmark_manifest.verify_mapping()`, tested
in `test_mapping_verified_against_local_labels`):
- All 15 UCF101 labels exist in the local `classInd.txt`, which has exactly
  101 classes.
- All 12 HMDB51 labels exist in the local official split files (51 classes),
  each with splits 1–3.
- No UCF101 label is mapped to more than one category.

## 3. Correctness rules (fixed before any confirmatory prediction is seen)

X-CLIP predicts over all **101 UCF101 labels** (same candidate set as Phase 1
and the frozen Guardian models).

- **Primary — `correct_mapped`:** the predicted UCF101 label is in the
  category's TA3N label set.
- **Secondary — `correct_strict`:** the predicted label equals the strict
  label (first listed).

**Disclosure (knowledge contamination).** In the exploratory Phase 1 analysis
we saw that HMDB *punch* is often predicted as *BoxingPunchingBag*. TA3N's
mapping includes BoxingPunchingBag and BoxingSpeedBag under *punch*. We adopt
it because it is the external published definition, not because of what we
saw. To bound any bias, the strict rule is reported alongside the mapped rule
for every primary endpoint, and any conclusion that differs between the two is
flagged.

Known semantic caveats:
- `walk` ↔ WalkingWithDog only. HMDB *walk* clips usually contain no dog, so
  low accuracy is expected.
- `kick_ball` ↔ SoccerPenalty.
- `climb` merges indoor and rope climbing.

These are properties of the external mapping and are reported, not "fixed".

## 4. Which HMDB51 clips are used

From the official HMDB51 split files (`testTrainMulti_7030_splits`, split 1):

| Role | Definition | Clips |
|---|---|---|
| **confirmatory_primary** | Split-1 **train** partition (label 1) of all 12 classes, minus exclusions | 797 |
| **confirmatory_secondary** | Split-1 **test** partition (label 2) of the 6 classes *not* used in Phase 1 (climb, golf, kick_ball, shoot_ball, shoot_bow, walk), minus exclusions | 177 |
| excluded: Phase 1 clips | The 180 split-1 test clips of fencing, punch, pushup, pullup, ride_bike, ride_horse (exploratory; burned) | 180 |
| excluded: film-group leakage | Any clip whose conservative film/source group occurs among the 180 Phase 1 clips | 46 |
| not used | Label-0 clips (unused in split 1); split-1 test clips of the 6 burned classes | — |

Per-class primary counts: climb 69, fencing 65, golf 70, kick_ball 70,
pullup 70, punch 62, pushup 67, ride_bike 70, ride_horse 57, shoot_ball 70,
shoot_bow 60, walk 67. Secondary: 30 per class except shoot_bow (27).

**Why the split-1 train partition is legitimate evaluation data here:**
- Neither X-CLIP (zero-shot) nor Guardian (trained on UCF101) ever trains on
  HMDB51, so the HMDB51 "train" partition is unseen.
- The official splits are built so that train and test partitions do not
  share source videos. We verified this at film level: 0–2 shared groups per
  class under our conservative rule, and those are excluded where they
  touch Phase 1 clips.
- Using label 1 gives a balanced design of about 70 per class, including
  *walk*, which has 548 clips in total.
- TA3N's HMDB "training" set has 840 = 12×70 videos, *consistent with* split 1
  label 1. The paper text we read does not name the split (see the
  LITERATURE_REVIEW checklist).

## 5. Leakage and duplicate controls

1. **Exact Phase 1 clips** are excluded (asserted).
2. **Film-level source groups** (`film_group()`) — a conservative,
   class-agnostic key:
   - lowercase alphanumerics;
   - strip trailing digits, `part`/`teil`, and embedded extension / `hd`
     tokens;
   - merge franchises (Lord of the Rings: *Fellowship*, *Two Towers*, *Return
     of the King*, any capitalisation; *The Matrix* / *Matrix Revolutions*).

   Any group that occurs among the Phase 1 clips is excluded from the new
   manifest (46 clips). Over-merging only removes data and widens CIs.
3. **HMDB51 duplicate finding.** HMDB51 split 1 contains **the same clip
   twice under different names**:
   `Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_avi_fencing_*` (train
   partition) duplicates `Die_Another_Day_-_Fencing_Scene_Part_1_[HD]_fencing_*`
   (test partition, a Phase 1 clip). The perceptual hash distance was 0. This
   would have leaked Phase 1 clips into the confirmatory set. It is now caught
   both by the normaliser and by the perceptual check.
4. **Perceptual near-duplicate check** (`dhash_video`): a 192-bit
   difference-hash of 3 frames (25/50/75%). Any confirmatory clip within
   Hamming ≤10 of a Phase 1 clip is excluded.
   - Current result: 391 local confirmatory + 180 Phase 1 videos hashed; 0
     cross pairs remain after the film rule; 0 new↔new pairs across different
     film groups.
   - **Coverage gap:** the 6 classes not yet extracted (406 primary + 177
     secondary clips) are hashed only after extraction in Phase 2B step 1,
     *before* any inference.
5. **Exact duplicates:** SHA256 of all locally available confirmatory videos;
   0 duplicates.
6. **Cross-class film groups** inside the new manifest (e.g. *The Fugitive*
   in climb/punch/walk) are kept but handled by the unstratified film-level
   cluster bootstrap.
7. **UCF101 reference data** (Guardian development/calibration, 202 + 202
   videos) is asserted group-disjoint from the UCF101 evaluation split and
   from each other. It is the only data allowed for any fitting or selection.

## 6. Residual risks

- Filename-based grouping cannot detect shared actors or channels across
  differently named sources.
- K400 (X-CLIP pretraining) may contain some of the same YouTube source
  videos as HMDB51/UCF101. This is unknown and untestable locally.
- The near-duplicate check uses 3 frames; heavily edited re-uploads may
  evade it.
