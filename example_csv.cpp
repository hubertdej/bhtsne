#include <cstdlib>
#include <fstream>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsne.h"

struct delimiter_ctype : std::ctype<char> {
  static const mask* make_table(const std::string& delims) {
    static std::vector<mask> v(classic_table(), classic_table() + table_size);
    for (mask& m : v) {
      m &= ~space;
    }
    for (char d : delims) {
      v[d] |= space;
    }
    return &v[0];
  }
  delimiter_ctype(std::string delims, size_t refs = 0) : ctype(make_table(delims), false, refs) {}
};

struct Dataset {
  std::vector<double> data;
  int n;
  int d;
};

Dataset readCsv(const std::string& filename, int n, bool header = false) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open data file!");
  }

  std::string line;
  std::getline(file, line);
  int d = std::count(line.begin(), line.end(), ',') + 1;

  if (!header) {
    file.seekg(0);
  }
  file.imbue(std::locale(file.getloc(), new delimiter_ctype(",\n")));

  std::vector<double> data(n * d);
  for (auto& value : data) {
    file >> value;
  }
  std::cout << "Read " << n << " x " << d << " data matrix\n";
  return {data, n, d};
}

struct Case {
  int n;
  int num_iters;
  int perplexity;
};

const std::vector<Case> cases = {
    {200, 2000, 40},
    {500, 1000, 40},
    {752, 1000, 40},
    {1000, 500, 40},
    {2000, 200, 40},
    {3000, 200, 40},
    {4000, 200, 40},
    {5000, 200, 40},
    {6000, 200, 40},
    {10000, 100, 40},
    {20000, 100, 40}
};

int main(int argc, char** argv) {
  Dataset dataset = readCsv("mnist_reduced_20k.csv", 20000, false);

  std::ofstream file("measurements.csv");
  file << "name,n,total_cycles,num_measurements\n";
  file.close();

  for (const auto& c : cases) {
    std::cout << "Running case: n=" << c.n << ", num_iters=" << c.num_iters << ", perplexity=" << c.perplexity << "\n";
    auto Y = new double[c.n * 2];
    TSNE::run(dataset.data.data(), c.n, dataset.d, Y, 2, c.perplexity, 0.0, 42, false, c.num_iters, 250, 250);
    delete[] Y;
  }

  return 0;
}
