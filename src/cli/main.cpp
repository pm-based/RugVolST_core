#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include <fbm/core/INoise.h>
#include <fbm/core/IEvolver.h>
#include <fbm/core/ISimulator.h>
#include <fbm/core/Simulator.h>
#include <fbm/core/GBM_Euler.h>
#include <fbm/core/BrownianNoise.h>

void printUsage() {
  std::cout << "Usage: fbm_cli bs [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --npaths N     Number of paths to simulate (default: 10000)\n";
  std::cout << "  --nsteps N     Number of time steps (default: 1000)\n";
  std::cout << "  --S0 VALUE     Initial stock price (default: 100.0)\n";
  std::cout << "  --mu VALUE     Drift parameter (default: 0.0)\n";
  std::cout << "  --sigma VALUE  Volatility parameter (default: 0.2)\n";
  std::cout << "  --T VALUE      Time horizon (default: 1.0)\n";
  std::cout << "  --seed N       Random seed (default: 42)\n";
  std::cout << "  --help         Show this help message\n";
}

struct CLIArgs {
  std::size_t n_paths = 10000;
  std::size_t n_steps = 1000;
  double S0 = 100.0;
  double mu = 0.0;
  double sigma = 0.2;
  double T = 1.0;
  std::uint64_t seed = 42;
  bool show_help = false;
};

CLIArgs parseArgs(int argc, char* argv[]) {
  CLIArgs args;

  if (argc < 2) {
    args.show_help = true;
    return args;
  }

  std::string command = argv[1];
  if (command != "bs") {
    std::cerr << "Error: Unknown command '" << command << "'\n";
    args.show_help = true;
    return args;
  }

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
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      args.show_help = true;
      return args;
    }
  }

  return args;
}

void runBlackScholes(const CLIArgs& args) {
  try {
    // Validate arguments
    if (args.n_steps < 2) {
      throw std::invalid_argument("nsteps must be at least 2");
    }
    if (args.sigma < 0.0) {
      throw std::invalid_argument("sigma must be non-negative");
    }
    if (args.T <= 0.0) {
      throw std::invalid_argument("T must be positive");
    }
    if (args.S0 <= 0.0) {
      throw std::invalid_argument("S0 must be positive");
    }

    // Create time grid
    std::vector<double> time(args.n_steps + 1);
    const double dt = args.T / args.n_steps;
    for (std::size_t i = 0; i <= args.n_steps; ++i) {
      time[i] = i * dt;
    }

    // Create components
    auto noise = std::make_shared<fbm::core::BrownianNoise>();
    auto evolver = std::make_shared<fbm::core::GBM_Euler>(args.mu, args.sigma);
    auto simulator = std::make_shared<fbm::core::Simulator>(noise, evolver);

    // Allocate output
    std::vector<double> S_out(args.n_paths * (args.n_steps + 1));

    // Run simulation with timing
    auto start = std::chrono::high_resolution_clock::now();
    simulator->simulate(time, args.n_paths, args.seed, args.S0, S_out);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Compute statistics
    std::vector<double> final_prices(args.n_paths);
    std::vector<double> log_returns(args.n_paths);

    for (std::size_t i = 0; i < args.n_paths; ++i) {
      final_prices[i] = S_out[i * (args.n_steps + 1) + args.n_steps];
      log_returns[i] = std::log(final_prices[i] / args.S0);
    }

    const double mean_final = std::accumulate(final_prices.begin(), final_prices.end(), 0.0) / args.n_paths;
    const double mean_log_return = std::accumulate(log_returns.begin(), log_returns.end(), 0.0) / args.n_paths;

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Black-Scholes Simulation Results\n";
    std::cout << "================================\n";
    std::cout << "Paths: " << args.n_paths << ", Steps: " << args.n_steps << "\n";
    std::cout << "S0: " << args.S0 << ", mu: " << args.mu << ", sigma: " << args.sigma << ", T: " << args.T << "\n";
    std::cout << "Seed: " << args.seed << "\n\n";
    std::cout << "Elapsed time: " << duration.count() << " ms\n";
    std::cout << "Mean final price: " << mean_final << "\n";
    std::cout << "Mean log-return: " << mean_log_return << "\n";
    std::cout << "Expected final price: " << args.S0 * std::exp(args.mu * args.T) << "\n";
    std::cout << "Expected log-return: " << (args.mu - 0.5 * args.sigma * args.sigma) * args.T << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::exit(1);
  }
}

int main(int argc, char* argv[]) {
  auto args = parseArgs(argc, argv);

  if (args.show_help) {
    printUsage();
    return 0;
  }

  runBlackScholes(args);
  return 0;
}
