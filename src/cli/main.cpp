#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <fbm/core/INoise.h>
#include <fbm/core/IEvolver.h>
#include <fbm/core/ISimulator.h>
#include <fbm/core/Simulator.h>
#include <fbm/core/GBMEuler.h>
#include <fbm/core/BrownianNoise.h>
#include <fbm/core/IKernel.h>
#include <fbm/core/VolterraKernelBrownian.h>
#include <fbm/core/VolterraKernelPowerLaw.h>
#include <fbm/core/VolterraNoise.h>
#include <fbm/core/RoughBergomiFactor.h>
#include <fbm/core/RoughBergomiAssetEuler.h>

#if defined(FBM_USE_OPENMP)
#include <omp.h>
#endif

struct CLIArgs {
  std::string command;
  std::size_t n_paths = 10000;
  std::size_t n_steps = 1000;
  double S0 = 100.0;
  double mu = 0.0;
  double sigma = 0.2;
  double T = 1.0;
  std::uint64_t seed = 42;
  double H = 0.3;
  std::size_t quad_points = 64;
  bool show_help = false;
  bool use_antithetic = false;
  int threads = 0; // 0 = library default
  double rho = 0.0; // correlation between dW and dB
  double eta = 1.5; // vol-of-vol
  double xi0 = 0.04; // shortcut per beta0
  // xi0(t) = beta0 + beta1*exp(-t/tau1) + beta2*(t/tau2)*exp(-t/tau2)
  double beta0 = 0.04;
  double beta1 = 0.0;
  double beta2 = 0.0;
  double tau1 = 1.0;
  double tau2 = 1.0;
};

static void printUsage() {
  std::cout << "Usage: fbm_cli <command> [options]\n\n";
  std::cout << "Commands:\n";
  std::cout << "  bs            Run Black–Scholes simulation\n";
  std::cout << "  kernel        Build and summarize Volterra kernel\n";
  std::cout << "  rb-noise      Generate Volterra noise BH = dB @ K\n";
  std::cout << "  rb-factor     Generate Rough Bergomi variance factors\n";
  std::cout << "  rb-asset      Generate Rough Bergomi asset paths\n\n";
  std::cout << "Common options:\n";
  std::cout << "  --npaths N    Number of paths\n";
  std::cout << "  --nsteps N    Number of time steps\n";
  std::cout << "  --T VALUE     Time horizon (default: 1.0)\n";
  std::cout << "  --seed N      Random seed\n";
  std::cout << "  --help        Show this help message\n\n";

  std::cout << "BS options:\n";
  std::cout << "  --S0 VALUE    Initial price (default: 100.0)\n";
  std::cout << "  --mu VALUE    Drift (default: 0.0)\n";
  std::cout << "  --sigma VALUE Volatility (default: 0.2)\n\n";

  std::cout << "Kernel / RB-noise / RB-factor / RB-asset options:\n";
  std::cout << "  --H VALUE     Hurst parameter in (0,1)\n";
  std::cout << "  --quad N      Quadrature points (kernel build)\n";
  std::cout << "  --antithetic  Use antithetic variates\n";
  std::cout << "  --rho VALUE   Corr(dW,dB) in (-1,1)\n";
#if defined(FBM_USE_OPENMP)
  std::cout << "  --threads N   OpenMP threads (0 = lib default)\n";
#endif
  std::cout << "\nRB-factor / RB-asset specific options:\n";
  std::cout << "  --eta VALUE   Volatility of volatility (>=0)\n";
  std::cout << "  --xi0 VALUE   Constant xi0 (alias for --beta0)\n";
  std::cout << "  --beta0 VAL   xi0 base level\n";
  std::cout << "  --beta1 VAL   exp decay weight\n";
  std::cout << "  --beta2 VAL   hump weight\n";
  std::cout << "  --tau1 VAL    exp decay time\n";
  std::cout << "  --tau2 VAL    hump time\n";
}

static CLIArgs parseArgs(int argc, char* argv[]) {
  CLIArgs args;

  if (argc < 2) {
    args.show_help = true;
    return args;
  }

  const std::string command = argv[1];
  if (command != "bs" && command != "kernel" && command != "rb-noise" &&
      command != "rb-factor" && command != "rb-asset") {
    std::cerr << "Error: Unknown command '" << command << "'\n";
    args.show_help = true;
    return args;
  }
  args.command = command;

  // Command-specific defaults
  if (command == "kernel") {
    args.n_steps = 100;
  } else if (command == "rb-noise") {
    args.n_paths = 20000; args.n_steps = 256;
    args.H = 0.5; args.quad_points = 16; args.seed = 7;
  } else if (command == "rb-factor" || command == "rb-asset") {
    args.n_paths = 20000; args.n_steps = 256;
    args.H = 0.5; args.quad_points = 16;
    args.eta = 1.5; args.xi0 = 0.04; args.beta0 = args.xi0; args.seed = 7;
    if (command == "rb-asset") args.S0 = 100.0;
  }

  // Parse options
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto need = [&](int k=1){ return i + k < argc; };

    if (arg == "--help") { args.show_help = true; return args; }
    else if (arg == "--npaths" && need()) { args.n_paths = std::stoull(argv[++i]); }
    else if (arg == "--nsteps" && need()) { args.n_steps = std::stoull(argv[++i]); }
    else if (arg == "--S0" && need())     { args.S0 = std::stod(argv[++i]); }
    else if (arg == "--mu" && need())     { args.mu = std::stod(argv[++i]); }
    else if (arg == "--sigma" && need())  { args.sigma = std::stod(argv[++i]); }
    else if (arg == "--T" && need())      { args.T = std::stod(argv[++i]); }
    else if (arg == "--seed" && need())   { args.seed = std::stoull(argv[++i]); }
    else if (arg == "--H" && need())      { args.H = std::stod(argv[++i]); }
    else if (arg == "--quad" && need())   { args.quad_points = std::stoull(argv[++i]); }
    else if (arg == "--antithetic")       { args.use_antithetic = true; }
#if defined(FBM_USE_OPENMP)
    else if (arg == "--threads" && need()){ args.threads = std::stoi(argv[++i]); }
#endif
    else if (arg == "--rho" && need())    { args.rho = std::stod(argv[++i]); }
    else if (arg == "--eta" && need())    { args.eta = std::stod(argv[++i]); }
    else if (arg == "--xi0" && need())    { args.xi0 = args.beta0 = std::stod(argv[++i]); }
    else if (arg == "--beta0" && need())  { args.beta0 = std::stod(argv[++i]); }
    else if (arg == "--beta1" && need())  { args.beta1 = std::stod(argv[++i]); }
    else if (arg == "--beta2" && need())  { args.beta2 = std::stod(argv[++i]); }
    else if (arg == "--tau1" && need())   { args.tau1 = std::stod(argv[++i]); }
    else if (arg == "--tau2" && need())   { args.tau2 = std::stod(argv[++i]); }
    else {
      std::cerr << "Error: Unknown or incomplete argument '" << arg << "'\n";
      args.show_help = true;
      return args;
    }
  }

  return args;
}

static inline std::vector<double> make_uniform_time(std::size_t n_steps, double T) {
  std::vector<double> t(n_steps + 1);
  const double dt = T / static_cast<double>(n_steps);
  for (std::size_t i = 0; i <= n_steps; ++i) t[i] = dt * static_cast<double>(i);
  return t;
}

static void runBlackScholes(const CLIArgs& args) {
  try {
    if (args.n_steps < 2) throw std::invalid_argument("nsteps must be at least 2");
    if (args.sigma < 0.0) throw std::invalid_argument("sigma must be non-negative");
    if (args.T <= 0.0) throw std::invalid_argument("T must be positive");
    if (args.S0 <= 0.0) throw std::invalid_argument("S0 must be positive");

    auto time = make_uniform_time(args.n_steps, args.T);

    auto noise = std::make_shared<fbm::core::BrownianNoise>();
    auto evolver = std::make_shared<fbm::core::GBMEuler>(args.mu, args.sigma);
    auto simulator = std::make_shared<fbm::core::Simulator>(noise, evolver);

    std::vector<double> S_out(args.n_paths * (args.n_steps + 1));

    const auto t0 = std::chrono::high_resolution_clock::now();
    simulator->simulate(time, args.n_paths, args.seed, args.S0, S_out);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::vector<double> final_prices(args.n_paths);
    std::vector<double> log_returns(args.n_paths);
    for (std::size_t i = 0; i < args.n_paths; ++i) {
      const double ST = S_out[i * (args.n_steps + 1) + args.n_steps];
      final_prices[i] = ST;
      log_returns[i] = std::log(ST / args.S0);
    }
    const double mean_final =
        std::accumulate(final_prices.begin(), final_prices.end(), 0.0) / args.n_paths;
    const double mean_log =
        std::accumulate(log_returns.begin(), log_returns.end(), 0.0) / args.n_paths;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Black-Scholes Simulation Results\n";
    std::cout << "================================\n";
    std::cout << "Paths: " << args.n_paths << ", Steps: " << args.n_steps << "\n";
    std::cout << "S0: " << args.S0 << ", mu: " << args.mu << ", sigma: " << args.sigma
              << ", T: " << args.T << "\n";
    std::cout << "Seed: " << args.seed << "\n\n";
    std::cout << "Elapsed time: " << ms << " ms\n";
    std::cout << "Mean final price: " << mean_final << "\n";
    std::cout << "Mean log-return: " << mean_log << "\n";
    std::cout << "Expected final price: " << args.S0 * std::exp(args.mu * args.T) << "\n";
    std::cout << "Expected log-return: "
              << (args.mu - 0.5 * args.sigma * args.sigma) * args.T << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    std::exit(1);
  }
}

static void runKernelTest(const CLIArgs& args) {
  try {
    if (args.n_steps < 2) throw std::invalid_argument("nsteps must be at least 2");
    if (args.T <= 0.0) throw std::invalid_argument("T must be positive");
    if (args.H <= 0.0 || args.H >= 1.0) throw std::invalid_argument("H must be in (0,1)");
    if (args.quad_points < 1) throw std::invalid_argument("quad_points must be at least 1");

    auto time = make_uniform_time(args.n_steps, args.T);
    const std::size_t N = args.n_steps;
    const double dt = args.T / static_cast<double>(N);

    fbm::core::VolterraKernelPowerLaw kernel;
    std::vector<double> K_small(N * N, 0.0);

    const auto t0 = std::chrono::high_resolution_clock::now();
    kernel.build(time, args.H, args.quad_points, K_small);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    double sumK = std::accumulate(K_small.begin(), K_small.end(), 0.0);
    double frob = 0.0;
    for (double v : K_small) frob += v * v;
    frob = std::sqrt(frob);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Volterra Kernel Test Results\n";
    std::cout << "============================\n";
    std::cout << "Shape: (" << N << "," << N << ")\n";
    std::cout << "Steps: " << N << ", T: " << args.T << ", dt: " << dt << "\n";
    std::cout << "H: " << args.H << ", Quad points: " << args.quad_points << "\n\n";
    std::cout << "Elapsed time: " << ms << " ms\n";
    std::cout << "Sum(K): " << sumK << "\n";
    std::cout << "Frobenius norm: " << frob << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    std::exit(1);
  }
}

static void runRBNoiseTest(const CLIArgs& args) {
  try {
    if (args.n_steps < 2) throw std::invalid_argument("nsteps must be at least 2");
    if (args.T <= 0.0) throw std::invalid_argument("T must be positive");
    if (args.H <= 0.0 || args.H >= 1.0) throw std::invalid_argument("H must be in (0,1)");
    if (args.quad_points < 1) throw std::invalid_argument("quad_points must be at least 1");

    auto time = make_uniform_time(args.n_steps, args.T);
    const std::size_t N = args.n_steps;
    const double dt = args.T / static_cast<double>(N);

    // Build kernel
    fbm::core::VolterraKernelPowerLaw kernel;
    std::vector<double> K_small(N * N, 0.0);
    kernel.build(time, args.H, args.quad_points, K_small);

    // Volterra noise generator
    fbm::core::VolterraNoise noise_gen(std::vector<double>(K_small), N, args.rho, args.use_antithetic);

    const std::size_t m = args.n_paths;
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    const auto t0 = std::chrono::high_resolution_clock::now();
    noise_gen.sample(dB, dW, BH, m, N, dt, args.seed);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Sample variances at a few indices
    std::vector<std::size_t> idxs{N / 4, N / 2, N - 1};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Volterra Noise Generation Results\n";
    std::cout << "=================================\n";
    std::cout << "Paths: " << m << ", Steps: " << N << "\n";
    std::cout << "T: " << args.T << ", H: " << args.H << ", dt: " << dt << "\n";
    std::cout << "Seed: " << args.seed << "\n";
    std::cout << "Antithetic: " << (args.use_antithetic ? "ON" : "OFF") << "\n";
    std::cout << "Rho: " << args.rho << "\n";
#if defined(FBM_USE_OPENMP)
    std::cout << "Threads: " << omp_get_max_threads() << "\n";
#endif
    std::cout << "\nElapsed time: " << ms << " ms\n\n";

    for (std::size_t idx : idxs) {
      double sum = 0.0, sumsq = 0.0;
      for (std::size_t p = 0; p < m; ++p) {
        const double v = BH[p * N + idx];
        sum += v;
        sumsq += v * v;
      }
      const double mean = sum / static_cast<double>(m);
      const double var_hat = (sumsq - static_cast<double>(m) * mean * mean) / static_cast<double>(m - 1);
      const double t = time[idx + 1];
      const double theory = std::pow(t, 2.0 * args.H); // Var(B_H(t)) = t^(2H)
      const double abs_err = std::abs(var_hat - theory);
      const double rel_err = abs_err / std::max(theory, 1e-16);
      std::cout << "t = " << t
                << ", Var_hat = " << var_hat
                << ", Theory = " << theory
                << ", AbsErr = " << abs_err
                << ", RelErr = " << rel_err << "\n";
    }

    std::cout << "\nCorrelation Analysis (dW vs dB):\n";
    std::cout << "================================\n";
    for (std::size_t idx : idxs) {
      double mean_dB = 0.0, mean_dW = 0.0;
      for (std::size_t p = 0; p < m; ++p) {
        mean_dB += dB[p * N + idx];
        mean_dW += dW[p * N + idx];
      }
      mean_dB /= static_cast<double>(m);
      mean_dW /= static_cast<double>(m);

      double cov = 0.0, var_dB = 0.0, var_dW = 0.0;
      for (std::size_t p = 0; p < m; ++p) {
        const double dB_dev = dB[p * N + idx] - mean_dB;
        const double dW_dev = dW[p * N + idx] - mean_dW;
        cov += dB_dev * dW_dev;
        var_dB += dB_dev * dB_dev;
        var_dW += dW_dev * dW_dev;
      }
      cov /= static_cast<double>(m - 1);
      var_dB /= static_cast<double>(m - 1);
      var_dW /= static_cast<double>(m - 1);

      const double corr_hat = cov / std::sqrt(var_dB * var_dW);
      std::cout << "t = " << time[idx + 1] << ", corr_hat = " << corr_hat << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    std::exit(1);
  }
}

static void runRBFactor(const CLIArgs& args) {
  try {
    if (args.n_steps < 2) throw std::invalid_argument("nsteps must be at least 2");
    if (args.T <= 0.0) throw std::invalid_argument("T must be positive");
    if (args.H <= 0.0 || args.H >= 1.0) throw std::invalid_argument("H must be in (0,1)");
    if (args.quad_points < 1) throw std::invalid_argument("quad_points must be at least 1");

    auto time = make_uniform_time(args.n_steps, args.T);
    const std::size_t N = args.n_steps;
    const double dt = args.T / static_cast<double>(N);

    // Build kernel
    fbm::core::VolterraKernelPowerLaw kernel;
    std::vector<double> K_small(N * N, 0.0);
    kernel.build(time, args.H, args.quad_points, K_small);

    // Generate Volterra noise (BH levels)
    fbm::core::VolterraNoise noise_gen(std::vector<double>(K_small), N, args.rho, args.use_antithetic);
    const std::size_t m = args.n_paths;
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise_gen.sample(dB, dW, BH, m, N, dt, args.seed);

    // Convert BH levels -> increments dBH
    std::vector<double> dBH(m * N);
    for (std::size_t p = 0; p < m; ++p) {
      double prev = 0.0;
      for (std::size_t i = 0; i < N; ++i) {
        const double curr = BH[p * N + i];
        dBH[p * N + i] = curr - prev;
        prev = curr;
      }
    }

    // xi0(t) curve on grid t_0..t_N (N+1 values)
    std::vector<double> xi0t(N + 1, 0.0);
    for (std::size_t i = 0; i <= N; ++i) {
      const double t = time[i];
      xi0t[i] = args.beta0
              + args.beta1 * std::exp(-t / args.tau1)
              + args.beta2 * (t / args.tau2) * std::exp(-t / args.tau2);
    }

    // Decide constant vs curve
    const bool use_curve = (std::abs(args.beta1) > 0.0) || (std::abs(args.beta2) > 0.0) || (args.beta0 != args.xi0);

    // Compute Rough Bergomi variance factors
    fbm::core::RoughBergomiFactor rb_factor;
    std::vector<double> XI(m * N);

    const auto t0 = std::chrono::high_resolution_clock::now();
    if (use_curve) {
      rb_factor.compute(std::span<const double>(dBH), std::span<const double>(time),
                        m, N, args.H, std::span<const double>(xi0t),
                        args.eta, std::span<double>(XI));
    } else {
      rb_factor.compute(std::span<const double>(dBH), std::span<const double>(time),
                        m, N, args.H, args.xi0, args.eta,
                        std::span<double>(XI));
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Sample means at a few indices
    std::vector<std::size_t> idxs{N / 4, N / 2, N - 1};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Rough Bergomi Factor Generation Results\n";
    std::cout << "======================================\n";
    std::cout << "Paths: " << m << ", Steps: " << N << "\n";
    std::cout << "T: " << args.T << ", H: " << args.H << ", dt: " << dt << "\n";
    std::cout << "Seed: " << args.seed << "\n";
    std::cout << "Antithetic: " << (args.use_antithetic ? "ON" : "OFF") << "\n";
    std::cout << "Rho: " << args.rho << "\n";
    std::cout << "Eta: " << args.eta << "\n";
    std::cout << "xi0(t): beta0=" << args.beta0
              << ", beta1=" << args.beta1
              << ", beta2=" << args.beta2
              << ", tau1="  << args.tau1
              << ", tau2="  << args.tau2 << "\n";
    std::cout << "\nElapsed time: " << ms << " ms\n\n";

    for (std::size_t idx : idxs) {
      double sum = 0.0;
      for (std::size_t p = 0; p < m; ++p) sum += XI[p * N + idx];
      const double mean = sum / static_cast<double>(m);
      const double t = time[idx + 1];
      const double theory = use_curve ? (args.beta0 + args.beta1 * std::exp(-t / args.tau1)
                                        + args.beta2 * (t / args.tau2) * std::exp(-t / args.tau2))
                                      : args.xi0;
      const double err = std::abs(mean - theory);
      std::cout << "t = " << t << ", Mean = " << mean
                << ", Theory = " << theory << ", Error = " << err << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    std::exit(1);
  }
}

static void runRBAsset(const CLIArgs& args) {
  try {
    if (args.n_steps < 2) throw std::invalid_argument("nsteps must be at least 2");
    if (args.T <= 0.0) throw std::invalid_argument("T must be positive");
    if (args.H <= 0.0 || args.H >= 1.0) throw std::invalid_argument("H must be in (0,1)");
    if (args.quad_points < 1) throw std::invalid_argument("quad_points must be at least 1");
    if (args.S0 <= 0.0) throw std::invalid_argument("S0 must be positive");

    auto time = make_uniform_time(args.n_steps, args.T);
    const std::size_t N = args.n_steps;
    const double dt = args.T / static_cast<double>(N);

    // Build kernel
    fbm::core::VolterraKernelPowerLaw kernel;
    std::vector<double> K_small(N * N, 0.0);
    kernel.build(time, args.H, args.quad_points, K_small);

    // Volterra noise (BH levels)
    fbm::core::VolterraNoise noise_gen(std::vector<double>(K_small), N, args.rho, args.use_antithetic);
    const std::size_t m = args.n_paths;

    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise_gen.sample(dB, dW, BH, m, N, dt, args.seed);

    // dBH increments
    std::vector<double> dBH(m * N);
    for (std::size_t p = 0; p < m; ++p) {
      double prev = 0.0;
      for (std::size_t i = 0; i < N; ++i) {
        const double curr = BH[p * N + i];
        dBH[p * N + i] = curr - prev;
        prev = curr;
      }
    }

    // xi0(t) curve on t_0..t_N
    std::vector<double> xi0t(N + 1, 0.0);
    for (std::size_t i = 0; i <= N; ++i) {
      const double t = time[i];
      xi0t[i] = args.beta0
              + args.beta1 * std::exp(-t / args.tau1)
              + args.beta2 * (t / args.tau2) * std::exp(-t / args.tau2);
    }
    const bool use_curve = (std::abs(args.beta1) > 0.0) || (std::abs(args.beta2) > 0.0) || (args.beta0 != args.xi0);

    // RB variance factors
    fbm::core::RoughBergomiFactor rb_factor;
    std::vector<double> XI(m * N);
    if (use_curve) {
      rb_factor.compute(std::span<const double>(dBH), std::span<const double>(time),
                        m, N, args.H, std::span<const double>(xi0t),
                        args.eta, std::span<double>(XI));
    } else {
      rb_factor.compute(std::span<const double>(dBH), std::span<const double>(time),
                        m, N, args.H, args.xi0, args.eta, std::span<double>(XI));
    }

    // Evolve asset paths
    std::vector<double> S_out(m * (N + 1));
    const auto t0 = std::chrono::high_resolution_clock::now();
    fbm::core::evolve_rb_asset(std::span<const double>(XI), std::span<const double>(dW),
                               m, N, dt, args.S0, std::span<double>(S_out));
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Diagnostics
    std::vector<double> log_returns(m);
    std::size_t positive_count = 0;
    for (std::size_t p = 0; p < m; ++p) {
      const double ST = S_out[p * (N + 1) + N];
      log_returns[p] = std::log(ST / args.S0);
      if (ST > 0.0) ++positive_count;
    }

    const double sample_mean = std::accumulate(log_returns.begin(), log_returns.end(), 0.0) / static_cast<double>(m);
    // riferimento semplice (se xi0 costante)
    const double theoretical_mean = -0.5 * args.xi0 * args.T;
    const double abs_error = std::abs(sample_mean - theoretical_mean);
    const double percent_positive = 100.0 * static_cast<double>(positive_count) / static_cast<double>(m);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Rough Bergomi Asset Evolution Results\n";
    std::cout << "====================================\n";
    std::cout << "Paths: " << m << ", Steps: " << N << "\n";
    std::cout << "T: " << args.T << ", H: " << args.H << ", dt: " << dt << "\n";
    std::cout << "S0: " << args.S0 << ", eta: " << args.eta << ", xi0: " << args.xi0 << "\n";
    std::cout << "Seed: " << args.seed << "\n";
    std::cout << "Antithetic: " << (args.use_antithetic ? "ON" : "OFF") << "\n";
    std::cout << "Rho: " << args.rho << "\n\n";
    std::cout << "Elapsed time (factor+asset): " << ms << " ms\n";
    std::cout << "Mean log-return: " << sample_mean << "\n";
    std::cout << "Theoretical reference (const xi0): " << theoretical_mean << "\n";
    std::cout << "Absolute error: " << abs_error << "\n";
    std::cout << "Paths with S_T > 0: " << percent_positive << "%\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    std::exit(1);
  }
}

int main(int argc, char* argv[]) {
  const CLIArgs args = parseArgs(argc, argv);

  if (args.show_help) {
    printUsage();
    return 0;
  }

#if defined(FBM_USE_OPENMP)
  if (args.threads > 0) omp_set_num_threads(args.threads);
#endif

  if (args.command == "bs") {
    runBlackScholes(args);
  } else if (args.command == "kernel") {
    runKernelTest(args);
  } else if (args.command == "rb-noise") {
    runRBNoiseTest(args);
  } else if (args.command == "rb-factor") {
    runRBFactor(args);
  } else if (args.command == "rb-asset") {
    runRBAsset(args);
  }
  return 0;
}