# Changelog

## 0.1.0

* Initial release.
* Added `maxtwin()` for deterministic train/test splitting with distribution
  preservation and train/test separation.
* Added `initialmaxtwin()` for the Step 1 split without the iterative separation
  improvement.
* Added tests covering split sizes, deterministic behavior, input validation,
  paired data, and consistency between the initial and full algorithms.
