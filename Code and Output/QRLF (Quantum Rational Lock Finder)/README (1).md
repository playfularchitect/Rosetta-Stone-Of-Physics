Important: The code explicitly downloads and processes three .xlsx files from the NIST (National Institute of Standards and Technology) Bell test data archive. Check QLRE V11.txt (code that shows files) and v11 output.md (output from the code)
# Megacell v13.13.9-onecell-megacell++++ — Run Artifacts
This folder contains a full audit trail: config & environment, ledger registry and hashes, and per-section CSV/JSON outputs.

## Contents
- `config.json`: run configuration, seeds, fingerprint, registry hash.
- `environment.json`: Python & NumPy versions, platform info.
- `ledger/`: registry fractions, prereg hash, file SHA256s.
- `sections/`: CSV/JSON exports for candidates, PB tails, FWER, Bayes, predictive holdout, E-values, nearest-dyadic z, cluster LOO, jackknife.

## Verification tips
- Re-run with the same seed to reproduce MC paths.
- Compare `registry_hash` and file SHA256s to validate the prereg ledger.
- Check that section CSVs match the corresponding rows in your console output.
