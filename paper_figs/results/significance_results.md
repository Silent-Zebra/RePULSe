Two-sided tests at the final evaluation step, across seeds. diff = mean(reference) - mean(compared).

| Claim | n | diff | Welch p | Bootstrap p | Bootstrap 95% CI of diff | Permutation p |
|---|---|---|---|---|---|---|
| Fig 4a: CTL tempering vs CTL, KL(s|q) | 10/10 | 6.475 | 0.010 | <0.001 | [4.030, 10.672] | <0.001 (exact) |
| Fig 4a: CTL tempering vs CTL, KL(q|s) | 10/10 | 1.540 | <0.001 | <0.001 | [1.172, 1.875] | <0.001 (exact) |
| Fig 4a: RLOO tempering vs RLOO, KL(s|q) | 10/10 | 6.228 | 0.005 | <0.001 | [3.940, 9.789] | <0.001 (exact) |
| Fig 4a: RLOO tempering vs RLOO, KL(q|s) | 10/10 | 1.509 | <0.001 | <0.001 | [1.153, 1.826] | <0.001 (exact) |
| Fig 4b: CTL tempering vs CTL, KL(s|q) | 10/10 | 4.895 | 0.012 | 0.005 | [1.531, 8.027] | 0.014 (exact) |
| Fig 4b: RLOO tempering vs RLOO, KL(s|q) | 10/10 | 3.302 | 0.342 | 0.305 | [-3.171, 9.315] | 0.335 (exact) |
| Fig 4c: CTL tempering vs CTL, KL(s|q) | 5/5 | 0.249 | 0.138 | 0.067 | [-0.016, 0.508] | 0.143 (exact) |
| Fig 4c: RLOO tempering vs RLOO, KL(s|q) | 5/5 | 8.301 | <0.001 | <0.001 | [6.344, 10.278] | 0.008 (exact) |
| Fig 5a: CTL exploration vs CTL, KL(s|q) | 10/10 | 4.856 | <0.001 | <0.001 | [3.135, 6.565] | <0.001 (exact) |
| Fig 5a: RLOO exploration vs RLOO, KL(s|q) | 10/10 | 5.674 | <0.001 | <0.001 | [4.530, 6.763] | <0.001 (exact) |
| Fig 5b: CTL exploration vs CTL, KL(s|q) | 10/10 | 0.377 | 0.202 | 0.160 | [-0.146, 0.884] | 0.193 (exact) |
| Fig 5b: RLOO exploration vs RLOO, KL(s|q) | 10/10 | 4.495 | 0.165 | 0.087 | [-0.490, 10.472] | 0.162 (exact) |
| Fig 5b: CTL exploration vs CTL, KL(q|s) | 10/10 | 12.277 | 0.006 | <0.001 | [5.704, 18.775] | 0.011 (exact) |
| Fig 5b: CTL tempering vs exploration, KL(s|q) | 10/10 | 9.595 | 0.002 | <0.001 | [5.392, 13.609] | <0.001 (exact) |
| Fig 5b: RLOO tempering vs exploration, KL(s|q) | 10/10 | 12.995 | 0.001 | <0.001 | [7.616, 18.171] | <0.001 (exact) |
| Fig 5c: CTL exploration vs CTL, KL(s|q) | 10/10 | 1.084 | <0.001 | <0.001 | [0.865, 1.299] | <0.001 (exact) |
| Fig 5c: RLOO exploration vs RLOO, KL(s|q) | 10/10 | 0.822 | <0.001 | <0.001 | [0.639, 1.006] | <0.001 (exact) |
| Fig 5c: CTL tempering vs CTL, KL(s|q) | 10/10 | 0.253 | 0.024 | 0.005 | [0.070, 0.445] | 0.019 (exact) |
| Fig 5c: RLOO tempering vs RLOO, KL(s|q) | 10/10 | 0.103 | 0.492 | 0.451 | [-0.172, 0.371] | 0.493 (exact) |
| Fig 6: entropy vs CTL, KL(s|q) | 10/10 | -0.063 | 0.975 | 0.964 | [-3.804, 3.691] | 0.975 (exact) |
| Fig 6: entropy vs CTL, ELBO | 10/10 | -0.013 | 0.994 | 0.999 | [-3.347, 3.260] | 0.994 (exact) |
| Fig 6: mixture vs CTL, KL(s|q) | 10/10 | 10.912 | <0.001 | <0.001 | [8.324, 13.274] | <0.001 (exact) |
| Fig 6: mixture vs CTL, ELBO | 10/10 | 3.794 | 0.026 | 0.012 | [0.844, 6.617] | 0.026 (exact) |
| Fig 6: tempering+exploration vs tempering, KL(s|q) | 10/10 | 0.566 | 0.763 | 0.757 | [-2.816, 4.030] | 0.761 (exact) |
| Fig 6: tempering+exploration vs tempering, ELBO | 10/10 | 0.649 | 0.720 | 0.704 | [-2.665, 3.939] | 0.717 (exact) |
| Fig 7: tempering+exploration vs exploration, KL(s|q) | 10/10 | 0.253 | 0.308 | 0.270 | [-0.230, 0.650] | 0.320 (exact) |
| Fig 7: tempering+exploration vs exploration, KL(q|s) | 10/10 | -1.041 | 0.101 | 0.067 | [-2.164, 0.070] | 0.101 (exact) |
| Fig 7: DPG exploration vs DPG, KL(s|q) | 10/10 | 1.251 | <0.001 | <0.001 | [0.941, 1.536] | <0.001 (exact) |
| Fig 7: DPG exploration vs DPG, KL(q|s) | 10/10 | 0.194 | 0.707 | 0.717 | [-0.685, 1.195] | 0.723 (exact) |
| Fig 7: DPG tempering vs DPG, KL(s|q) | 10/10 | 0.013 | 0.919 | 0.901 | [-0.219, 0.236] | 0.920 (exact) |
| Fig 7: DPG tempering vs DPG, KL(q|s) | 10/10 | -0.099 | 0.804 | 0.787 | [-0.828, 0.634] | 0.783 (exact) |
| Fig 7: CTL vs DPG, KL(s|q) | 10/10 | 1.006 | <0.001 | <0.001 | [0.863, 1.143] | <0.001 (exact) |
| Fig 7: CTL vs DPG, KL(q|s) | 10/10 | 0.496 | 0.201 | 0.162 | [-0.203, 1.178] | 0.204 (exact) |
| Fig 7: CTL tempering vs DPG tempering, KL(s|q) | 10/10 | 1.246 | <0.001 | <0.001 | [0.994, 1.510] | <0.001 (exact) |
| Fig 7: CTL tempering vs DPG tempering, KL(q|s) | 10/10 | 0.450 | 0.337 | 0.286 | [-0.328, 1.364] | 0.379 (exact) |
| Fig 7: CTL exploration vs DPG exploration, KL(s|q) | 10/10 | 0.839 | <0.001 | <0.001 | [0.504, 1.190] | <0.001 (exact) |
| Fig 7: CTL exploration vs DPG exploration, KL(q|s) | 10/10 | 0.443 | 0.452 | 0.414 | [-0.668, 1.471] | 0.465 (exact) |
| Fig 7: CTL(U) vs CTL, KL(s|q) | 10/10 | 0.021 | 0.792 | 0.791 | [-0.124, 0.173] | 0.790 (exact) |
| Fig 7: CTL(U) vs CTL, KL(q|s) | 10/10 | -0.145 | 0.640 | 0.649 | [-0.712, 0.420] | 0.799 (exact) |
| Fig 8: entropy vs CTL, KL(s|q) | 10/10 | 0.652 | 0.015 | 0.007 | [0.195, 1.056] | 0.013 (exact) |
| Fig 8: entropy vs CTL, KL(q|s) | 10/10 | 6.019 | 0.084 | 0.056 | [-0.027, 12.191] | 0.208 (exact) |
| Fig 8: entropy vs exploration, KL(s|q) | 10/10 | 0.275 | 0.422 | 0.382 | [-0.349, 0.889] | 0.418 (exact) |
| Fig 8: entropy vs exploration, KL(q|s) | 10/10 | -6.259 | 0.195 | 0.160 | [-14.680, 2.523] | 0.182 (exact) |
| Fig 8: mixture vs CTL, KL(s|q) | 10/10 | -0.007 | 0.951 | 0.958 | [-0.232, 0.208] | 0.950 (exact) |
| Fig 8: mixture vs CTL, KL(q|s) | 10/10 | 0.031 | 0.075 | 0.026 | [0.003, 0.063] | 0.075 (exact) |
