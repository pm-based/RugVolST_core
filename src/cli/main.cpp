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
  double eta = 1.5; // volatility of volatility for rb-factor
  double xi0 = 0.04; // initial variance level for rb-factor
};

static void printUsage() {
  std::cout << "Usage: fbm_cli <command> [options]\n\n";
  std::cout << "Commands:\n";
  std::cout << "  bs            Run Black–Scholes simulation\n";
  std::cout << "  kernel        Build and summarize Volterra kernel\n";
  std::cout << "  rb-noise      Generate Volterra noise BH = dB @ K\n";
  std::cout << "  rb-factor     Generate Rough Bergomi variance factors\n";
  std::cout << "  rb-asset      Generate Rough Bergomi asset paths\n\n";
  std::cout << "  --npaths N     Number of paths (default varies by command)\n";
  std::cout << "  --nsteps N     Number of time steps (default varies by command)\n";
  std::cout << "  --T VALUE      Time horizon (default: 1.0)\n";
  std::cout << "  --seed N       Random seed (default varies by command)\n";
  std::cout << "  --help         Show this help message\n\n";

  std::cout << "BS options:\n";
  std::cout << "  --S0 VALUE     Initial price (default: 100.0)\n";
  std::cout << "  --mu VALUE     Drift (default: 0.0)\n";
  std::cout << "  --sigma VALUE  Volatility (default: 0.2)\n\n";

  std::cout << "Kernel / RB-noise / RB-factor / RB-asset options:\n";
  std::cout << "  --H VALUE      Hurst parameter in (0,1) (default: 0.3 for kernel, 0.5 for others)\n";
  std::cout << "  --quad N       Quadrature points (default: 64 for kernel, 16 for others)\n";
  std::cout << "  --antithetic   Use antithetic variates for variance reduction (default: off)\n";
  std::cout << "  --rho VALUE    Correlation between dW and dB in (-1,1) (default: 0.0)\n";
#if defined(FBM_USE_OPENMP)
  std::cout << "  --threads N    Number of OpenMP threads (default: 0 = library default)\n";
#endif
  std::cout << "\nRB-factor / RB-asset specific options:\n";
  std::cout << "  --eta VALUE    Volatility of volatility parameter (default: 1.5)\n";
  std::cout << "  --xi0 VALUE    Initial variance level (default: 0.04)\n";
}

static CLIArgs parseArgs(int argc, char* argv[]) {
  CLIArgs args;

  if (argc < 2) {
    args.show_help = true;
    return args;
  }

  const std::string command = argv[1];
  if (command != "bs" && command != "kernel" && command != "rb-noise" && command != "rb-factor" && command != "rb-asset") {
    std::cerr << "Error: Unknown command '" << command << "'\n";
    args.show_help = true;
    return args;
  }
  args.command = command;

  // Command-specific defaults
  if (command == "kernel") {
    args.n_steps = 100;
  } else if (command == "rb-noise") {
    args.n_paths = 20000;
    args.n_steps = 256;
    args.H = 0.5;
    args.quad_points = 16;
    args.seed = 7;
  } else if (command == "rb-factor") {
    args.n_paths = 20000;
    args.n_steps = 256;
    args.H = 0.5;
    args.quad_points = 16;
    args.seed = 7;
  } else if (command == "rb-asset") {
    args.n_paths = 20000;
    args.n_steps = 256;
    args.H = 0.5;
    args.quad_points = 16;
    args.eta = 1.5;
    args.xi0 = 0.04;
    args.S0 = 100.0;
    args.seed = 7;
  }

  // Parse options
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      args.show_help = true;
      return args;
    } else if (arg == "--npaths" && i + 1 < argc) {
      args.n_paths = std::stoull(argv[++i]);
    } else if (arg == "--nsteps" && i + 1 < argc) {
      args.n_steps = std::stoull(argv[++i]);
    } else if (arg == "--S0" && i + 1 < argc) {
      args.S0 = std::stod(argv[++i]);
    } else if (arg == "--mu" && i + 1 < argc) {
      args.mu = std::stod(argv[++i]);
    } else if (arg == "--sigma" && i + 1 < argc) {
      args.sigma = std::stod(argv[++i]);
    } else if (arg == "--T" && i + 1 < argc) {
      args.T = std::stod(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      args.seed = std::stoull(argv[++i]);
    } else if (arg == "--H" && i + 1 < argc) {
      args.H = std::stod(argv[++i]);
    } else if (arg == "--quad" && i + 1 < argc) {
      args.quad_points = std::stoull(argv[++i]);
    } else if (arg == "--antithetic") {
      args.use_antithetic = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      args.threads = std::stoi(argv[++i]);
    } else if (arg == "--rho" && i + 1 < argc) {
      args.rho = std::stod(argv[++i]);
    } else if (arg == "--eta" && i + 1 < argc) {
      args.eta = std::stod(argv[++i]);
    } else if (arg == "--xi0" && i + 1 < argc) {
      args.xi0 = std::stod(argv[++i]);
    } else {
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

    // Volterra noise generator with antithetic flag
    fbm::core::VolterraNoiseGEMM noise_gen(std::vector<double>(K_small), N, args.rho, args.use_antithetic);
    // Generate noise

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

    // If antithetic is enabled, compute variance reduction ratio at last index
    std::cout << "\nCorrelation Analysis (dW vs dB):\n";
    std::cout << "================================\n";
    for (std::size_t idx : idxs) {
      // Compute sample means
      double mean_dB = 0.0, mean_dW = 0.0;
      for (std::size_t p = 0; p < m; ++p) {
        mean_dB += dB[p * N + idx];
        mean_dW += dW[p * N + idx];
      }
      mean_dB /= static_cast<double>(m);
      mean_dW /= static_cast<double>(m);

      // Compute covariance and variances
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

      // Compute Pearson correlation
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

    // Generate Volterra noise (BH = dB @ K)
    fbm::core::VolterraNoiseGEMM noise_gen(std::vector<double>(K_small), N, args.rho, args.use_antithetic);
    const std::size_t m = args.n_paths;
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise_gen.sample(dB, dW, BH, m, N, dt, args.seed);

    // Compute Rough Bergomi variance factors
    fbm::core::RB_Factor rb_factor;
    std::vector<double> XI(m * N);

    const auto t0 = std::chrono::high_resolution_clock::now();
    rb_factor.compute(std::span<const double>(BH), std::span<const double>(time),
                      m, N, args.H, args.xi0, args.eta, std::span<double>(XI));
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
    std::cout << "Xi0: " << args.xi0 << "\n";
    std::cout << "\nElapsed time: " << ms << " ms\n\n";

    for (std::size_t idx : idxs) {
      double sum = 0.0;
      for (std::size_t p = 0; p < m; ++p) {
        sum += XI[p * N + idx];
      }
      const double mean = sum / static_cast<double>(m);
      const double theory = args.xi0; // E[xi_t] = xi0 due to drift adjustment
      const double err = std::abs(mean - theory);
      std::cout << "t = " << time[idx + 1] << ", Mean = " << mean << ", Error = " << err << "\n";
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

    // Generate Volterra noise (BH = dB @ K)
    fbm::core::VolterraNoiseGEMM noise_gen(std::vector<double>(K_small), N, args.rho, args.use_antithetic);
    const std::size_t m = args.n_paths;
    std::vector<double> dB(m * N), dW(m * N), BH(m * N);
    noise_gen.sample(dB, dW, BH, m, N, dt, args.seed);

    // Compute Rough Bergomi variance factors
    fbm::core::RB_Factor rb_factor;
    std::vector<double> XI(m * N);
    rb_factor.compute(std::span<const double>(BH), std::span<const double>(time),
                      m, N, args.H, args.xi0, args.eta, std::span<double>(XI));

    // Evolve asset paths
    std::vector<double> S_out(m * (N + 1));
    const auto t0 = std::chrono::high_resolution_clock::now();
    fbm::core::evolve_rb_asset(std::span<const double>(XI), std::span<const double>(dW),
                               m, N, dt, args.S0, std::span<double>(S_out));
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Compute diagnostics
    std::vector<double> log_returns(m);
    std::size_t positive_count = 0;
    for (std::size_t p = 0; p < m; ++p) {
      const double ST = S_out[p * (N + 1) + N];
      log_returns[p] = std::log(ST / args.S0);
      if (ST > 0.0) ++positive_count;
    }

    const double sample_mean = std::accumulate(log_returns.begin(), log_returns.end(), 0.0) / static_cast<double>(m);
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
    std::cout << "Theoretical reference: " << theoretical_mean << "\n";
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
  // Set OpenMP thread count if specified
  if (args.threads > 0) {
    omp_set_num_threads(args.threads);
  }
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
