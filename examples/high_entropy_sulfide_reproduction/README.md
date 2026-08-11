# High-entropy sulfide manuscript evidence replay

This example is a compact, read-only replay of numerical evidence already
reported in the manuscript and its supporting information.  It does **not**
run an MLIP, MD, AIMD, DFT, model training, or structure optimization, and it
does not claim an independent recalculation of the paper.

The original Word documents are deliberately not copied into this example.
Their basenames, byte sizes, SHA-256 digests, and the exact source locations of
each transcription are recorded in
[`evidence/transcription_provenance.json`](evidence/transcription_provenance.json).
The compact tables preserve the reported cells needed to reconstruct:

- the model/training-data inventory (Table S2);
- static PES test RMSE ranking (the table captioned "Table 3" in the supplied
  SI document);
- AIMD versus MLIP room-temperature conductivity comparisons (Table S3);
- DeePMD(se_atten_v2)-MD versus AIMD timings (Tables S8--S10); and
- voltage values for DFT and every evaluated MLIP (Table S11).

Three additional compact evidence records cover the broader manuscript
acceptance boundary:

- the exact top ten from an already-completed 247/247 large-supercell
  composition screening, with input-manifest and implementation fingerprints;
- an auditable identity link from rank-1 `Mn6_Fe3_Ni8_Cu4_Zn7` to historical
  folder `6_3_8_4_7` and exact read-only transport post-processing parity; and
- claim-level unseen/transfer statements pinned to the main-document SHA-256.

Historical transport parity is not AIMD/DFT high-fidelity validation.  That
comparison remains `EXTERNAL_VALIDATION_PENDING`, and the unseen/transfer
claim remains `NOT_TESTABLE_WITH_AVAILABLE_DATA` because paired raw values are
not supplied.

`reproduce.py` derives normalized benchmark metrics from those transcriptions,
uses the ordinary MLIPFlow routing implementation and the policies in
`project.yaml`, and writes the benchmark and report artifacts.  No winner is
encoded in the script.  The selected model is the result of the evidence,
metric direction, and policy weights.

From a source checkout:

```bash
PYTHONPATH=../../src python3 reproduce.py --check
```

Use `--write` to regenerate the committed normalized artifacts and reports.
Both modes read the compact evidence inside this example and use the local
MLIPFlow routing and benchmark-wrapper code; they do not fetch external data.
`--check` fails if a transcription or generated artifact has changed.

Important interpretation boundary: `REPLAY_VERIFIED` here means that the
machine-readable transcription and deterministic reductions are internally
consistent with the supplied source documents.  It is not a fresh scientific
prediction and is not a replacement for raw-data or production-scale parity
validation.
