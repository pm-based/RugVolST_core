# RoughVolatility-ShortTimeApproximation (C++)

High‑performance building blocks for simulating fractional / rough volatility models (Rough Bergomi) via Volterra kernels. Clean C++20 API, tested with Catch2, and a simple CLI for demos and diagnostics.

---

## Features

- **Volterra kernels**
  - `VolterraKernelBrownian` — H=0.5 triangular (sanity/reference kernel)
  - `VolterraKernelPowerLaw` — H∈(0,1), normalized so VarB^H_T=T^{2H}
- **Noise generation** — VolterraNoise produces dB, dW and B^H levels; optional antithetic paths; configurable ρ = Corr(dW,dB)
- **Rough Bergomi**
  - `RoughBergomiFactor` — ξ_t = ξ_0(t) exp(η B^H_t - ½η² t^{2H})
  - `RoughBergomiAssetEuler` — log‑Euler asset evolution using variance factors
- **CLI demos and Catch2 tests**

---

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFBM_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## CLI examples

```bash
./build/fbm_cli kernel    --nsteps 64   --H 0.5 --quad 16
./build/fbm_cli rb-noise  --npaths 20000 --nsteps 256 --T 1.0 --H 0.3 --seed 7
./build/fbm_cli rb-factor --npaths 20000 --nsteps 256 --T 1.0 --H 0.5 --eta 1.5 --xi0 0.04 --seed 7
./build/fbm_cli rb-asset  --npaths 20000 --nsteps 256 --T 1.0 --H 0.5 --eta 1.5 --xi0 0.04 --S0 100 --seed 7
```

With ξ₀(t) curve:

```bash
./build/fbm_cli rb-factor --npaths 20000 --nsteps 256 --T 1 \
--H 0.5 --eta 1.3 --beta0 0.03 --beta1 0.02 --tau1 0.5 --beta2 0.01 --tau2 1.5
```

---

## Build configuration

- **C++ standard**: C++20
- **Options**:
  - `FBM_BUILD_TESTS` (ON/OFF) — build Catch2 tests
  - `FBM_USE_OPENMP` (OFF default) — enable OpenMP for per‑path loops
  - `FBM_USE_BLAS` (OFF default) — define BLAS path for matrix multiply (dgemm). Linking left to toolchain/user

### Install

```bash
cmake --install build --prefix /your/prefix
```

This installs `fbm_core` library, headers under `include/`, `fbm_cli`, and a minimal CMake package (`fbmTargets.cmake`).

---

## Project layout

```
include/fbm/core/
├── BrownianNoise.h
├── GBMEuler.h
├── IKernel.h
├── INoise.h
├── IEvolver.h
├── ISimulator.h
├── RoughBergomiAssetEuler.h
├── RoughBergomiFactor.h
├── Simulator.h
├── VolterraKernelBrownian.h
├── VolterraKernelPowerLaw.h
└── VolterraNoise.h

src/core/
├── BrownianNoise.cpp
├── GBMEuler.cpp
├── RoughBergomiAssetEuler.cpp
├── RoughBergomiFactor.cpp
├── Simulator.cpp
├── VolterraKernelBrownian.cpp
├── VolterraKernelPowerLaw.cpp
└── VolterraNoise.cpp

src/cli/
└── main.cpp
```

---

## Library overview

### Volterra kernels

```cpp
VolterraKernelPowerLaw K;                  // H in (0,1)
K.build(time, H, quad_points, K_small);    // K_small is N×N, row‑major
```

### Noise generation

```cpp
VolterraNoise noise(K_small, N, rho /*Corr(dW,dB)*/, use_antithetic);
noise.sample(dB, dW, BH, m, N, dt, seed);
```

- `dB`, `dW` ~ N(0, dt), size m·N (path‑major), `BH` are B^H levels at t₁..t_N

### Rough Bergomi factor

```cpp
RoughBergomiFactor f;
// curve variant (xi0(t_0..t_N))
f.compute(dBH, time, m, N, H, xi0t, eta, XI);
// constant variant
f.compute(dBH, time, m, N, H, xi0,   eta, XI);
```

- Inputs are increments `dBH`; the API reconstructs levels internally
- Output `XI` has size m·N and corresponds to t₁..t_N

### Asset evolution

```cpp
RoughBergomiAssetEuler evol;
evol.evolve(time, m, dB /*unused*/, dW, BH /*unused*/, XI, dt, S0, S);
```

---

## Reproducibility & tests

- All stochastic demos accept `--seed`
- Test suite checks: kernel shapes/variance, noise variance & correlations, RB factor mean vs ξ₀(t), asset drift sanity
- Run tests: `ctest --output-on-failure`

---

## Performance notes

- Building K is O(N²); generating B^H via matrix multiply is O(m N²)
- Memory: K is N×N; for large N prefer chunked GEMM or external BLAS
- Parallelism: enable OpenMP (`FBM_USE_OPENMP=ON`) for per‑path loops

---

## Citation

If you use this code in academic work, please cite the repository and the rough Bergomi literature as appropriate.