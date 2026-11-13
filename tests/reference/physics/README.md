# Reference outputs for the physics layer

`tests/test_physics.py` checks bootlab's physics layer (reading the spectrum, its errors and warnings, the
coordinates) against the reference Hyperion-based program (bt r3.8): for each case, that program's own
parameter check was run once and its output recorded here. The tests themselves need no cluster.
`make_reference.py` is the script that made them.

The `bt-*.json`, `bootlab-*.json` and `*.result.json` files are the recorded input and output. The model-file
paths in them are the record of that run, not something the tests or bootlab read (tests/test_physics.py
RecordedPaths checks it). In this public copy their directory is replaced by the placeholder
`/cluster/work/bootlab-physics-check/`; nothing else in the recorded files is changed.

- `models/`: the model used (ising-oe, converted from simpleboot's AutoEqu_Ising_OE).
- `cases/<case>/`: bootlab's inputs; `points.json`; for each point k, `bt-k.json` (the reference program's
  form), `bootlab-k.json` made from bootlab's TOML (only where bootlab accepts the inputs), and
  `*.result.json`, what the reference program printed (exit code, stdout, stderr).
- `coords/<model>.json`: a feasibility point with no coordinates; the error names the coordinates in order.

| case | model, Lambda | what it covers |
|---|---|---|
| real-ising-oe | ising-oe | feasibility points of a real run |
| point-ising-oe | ising-oe, 11 | delta = "deltaSig + 2", deltaEpsP as a coordinate, finite-difference points |
