# Parallel oversampling validation

`run_oversampling_validation.py` runs independent target/seed shards in
parallel without changing model code, validation code, datasets, seeds, or the
experiment fingerprint. Existing successful JSON files are therefore reused.

The default is four workers. The graphical launcher uses the same default.
Change `DEFAULT_PARALLEL_WORKERS` near the top of
`launch_oversampling_validation.py` to adjust the GUI default.

Command-line examples:

```powershell
# Default: four independent workers.
python run_oversampling_validation.py run --profile full --protocol cpdp --variant tabddpm --variant full

# Explicit worker counts.
python run_oversampling_validation.py --workers 3 run --profile full --protocol cpdp --variant full
python run_oversampling_validation.py --workers 1 run --profile full --protocol cpdp --variant full
```

For CPDP, workers receive disjoint target-project shards. When only one target
is selected, workers receive disjoint seed shards. Source-CV uses the same rule
with dataset families. Protocol phases run sequentially, so the process count
never exceeds the configured worker count.

`--max-tasks` keeps the original serial behavior because a global task cap has
ordering semantics that cannot be preserved by independent shards. Successful
results remain resumable, and a final summary is rebuilt after every parallel
phase completes.
