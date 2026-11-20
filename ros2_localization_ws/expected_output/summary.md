# W2 localization benchmark

Machine: 2 CPUs (Linux-6.18.44-fc-v726-x86_64-with-glibc2.39); 600 scans at 5 Hz, kidnapped at t = 60.0 s

| method  | converged in s | tracking RMSE cm | heading RMSE deg | recovered after kidnap in s | ms / update (mean, p95) |
| ------- | -------------- | ---------------- | ---------------- | --------------------------- | ----------------------- |
| grid    | 0.0            | 5.9              | 2.1              | 0.4                         | 115.71, 139.41          |
| mcl     | 0.0            | 4.1              | 0.5              | 3.0                         | 3.07, 3.54              |
| mcl_ml  | 0.0            | 4.1              | 0.5              | 0.4                         | 3.15, 3.75              |
| ml_only | 0.0            | 39.9             | 9.5              | 1.2                         | 0.23, 0.27              |

Regressor: 117892 parameters, trained on 40000 scans in 40 s on CPU; validation median error 23 cm (p90 63 cm), inference 26 µs/scan.

Convergence = position error < 0.3 m and heading error < 15° held for 5 s.
