#ifndef FASTTSNE_MEASURE_HPP
#define FASTTSNE_MEASURE_HPP

#include <filesystem>
#include <fstream>
#include <string_view>

#include "tsc_x86.hpp"

namespace fasttsne::measure {

constexpr int NUM_TARGETS = 1;

enum MeasurementTarget {
  PERFORM_GRADIENT_DESCENT_ITERATION,
};

constexpr std::string_view names[NUM_TARGETS] = {
    "performGradientDescentIteration"
};

int num_measurements[NUM_TARGETS] = {};
uint64_t starts[NUM_TARGETS] = {};
uint64_t total_cycles[NUM_TARGETS] = {};

int NUM_SKIP_FIRST = 0;
void setNumSkip(int num_skip) {
  NUM_SKIP_FIRST = num_skip;
}

template <MeasurementTarget MT>
void start() {}

template <MeasurementTarget MT>
void stop() {}

#define REGISTER_TARGET(MT)                            \
  template <>                                          \
  void start<MT>() {                                   \
    if (num_measurements[MT] < NUM_SKIP_FIRST) {       \
      return;                                          \
    }                                                  \
    starts[MT] = tsc_x86::start_tsc();                 \
  }                                                    \
  template <>                                          \
  void stop<MT>() {                                    \
    if (num_measurements[MT]++ < NUM_SKIP_FIRST) {     \
      return;                                          \
    }                                                  \
    total_cycles[MT] += tsc_x86::stop_tsc(starts[MT]); \
  }

REGISTER_TARGET(PERFORM_GRADIENT_DESCENT_ITERATION)
#undef REGISTER_TARGET

void appendResults(const std::filesystem::path& path, int n) {
  std::ofstream file(path, std::ios_base::app);
  for (int i = 0; i < NUM_TARGETS; ++i) {
    if (num_measurements[i] - NUM_SKIP_FIRST > 0) {
      file << names[i] << "," << n << "," << total_cycles[i] << "," << num_measurements[i] - NUM_SKIP_FIRST << "\n";
    }
  }
  file.close();
}

void reset() {
  for (int i = 0; i < NUM_TARGETS; ++i) {
    num_measurements[i] = 0;
    total_cycles[i] = 0;
  }
}

}  // namespace fasttsne::measure

#endif  // FASTTSNE_MEASURE_HPP
