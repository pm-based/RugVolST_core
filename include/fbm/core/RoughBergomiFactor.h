#pragma once

#include <span>
#include <cstddef>

namespace fbm::core {

    /// RoughBergomiFactor
    /// ------------------
    /// Calcola i fattori di varianza xi_t su griglia uniforme per il modello Rough Bergomi.
    ///
    /// Formula per i times t_{i+1}, i=0..N-1:
    ///   xi_t[i] = xi0(t_{i+1}) * exp( eta * BH_{t_{i+1}} - 0.5 * eta^2 * t_{i+1}^{2H} )
    /// dove BH_{t} è la fBM (livelli). L’API accetta dBH (incrementi) e ricostruisce i livelli.
    ///
    /// Convenzioni I/O (row-major per path):
    ///   - dBH:  (m_paths * N)  incrementi fBM per passo
    ///   - time: (N+1)          griglia uniforme con time[0] = 0
    ///   - XI:   (m_paths * N)  output ai tempi t_1..t_N
    class RoughBergomiFactor {
    public:
        /// Versione con curva xi0(t) fornita su t_0..t_N (N+1 valori).
        /// Usa soltanto t_1..t_N per costruire XI.
        void compute(std::span<const double> dBH,
                     std::span<const double> time,
                     std::size_t m_paths, std::size_t N,
                     double H, std::span<const double> xi0t, double eta,
                     std::span<double> XI) const;

        /// Overload con xi0 costante.
        void compute(std::span<const double> dBH,
                     std::span<const double> time,
                     std::size_t m_paths, std::size_t N,
                     double H, double xi0, double eta,
                     std::span<double> XI) const;
    };

} // namespace fbm::core