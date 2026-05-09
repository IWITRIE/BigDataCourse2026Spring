#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kDampingFactor = 0.85;
constexpr double kTolerance = 1e-12;
constexpr int kMaxIterations = 1000;
constexpr int kTopK = 100;
constexpr int kBlockSize = 1024;

struct Graph {
    std::vector<std::int64_t> node_ids;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

std::ifstream OpenInputFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    return input;
}

Graph ReadGraph(const std::string& input_path) {
    std::ifstream input = OpenInputFile(input_path);

    std::vector<std::pair<std::int64_t, std::int64_t>> raw_edges;
    std::vector<std::int64_t> raw_nodes;
    raw_edges.reserve(200000);
    raw_nodes.reserve(400000);

    std::int64_t from = 0;
    std::int64_t to = 0;
    while (input >> from >> to) {
        raw_edges.emplace_back(from, to);
        raw_nodes.push_back(from);
        raw_nodes.push_back(to);
    }

    if (raw_edges.empty()) {
        throw std::runtime_error("input graph is empty");
    }

    std::sort(raw_nodes.begin(), raw_nodes.end());
    raw_nodes.erase(std::unique(raw_nodes.begin(), raw_nodes.end()), raw_nodes.end());

    std::unordered_map<std::int64_t, int> id_to_index;
    id_to_index.reserve(raw_nodes.size() * 2);
    for (int i = 0; i < static_cast<int>(raw_nodes.size()); ++i) {
        id_to_index.emplace(raw_nodes[i], i);
    }

    std::vector<std::pair<int, int>> edges;
    edges.reserve(raw_edges.size());
    for (const auto& edge : raw_edges) {
        edges.emplace_back(id_to_index.at(edge.first), id_to_index.at(edge.second));
    }
    raw_edges.clear();
    raw_edges.shrink_to_fit();

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    const int n = static_cast<int>(raw_nodes.size());
    std::vector<int> out_degree(n, 0);
    for (const auto& edge : edges) {
        ++out_degree[edge.first];
    }

    Graph graph;
    graph.node_ids = std::move(raw_nodes);
    graph.row_ptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        graph.row_ptr[i + 1] = graph.row_ptr[i] + out_degree[i];
    }

    graph.col_idx.assign(edges.size(), 0);
    std::vector<int> cursor = graph.row_ptr;
    for (const auto& edge : edges) {
        graph.col_idx[cursor[edge.first]++] = edge.second;
    }

    return graph;
}

std::vector<double> ComputePageRank(const Graph& graph, int* iterations, double* residual) {
    const int n = static_cast<int>(graph.node_ids.size());
    const double initial_rank = 1.0 / static_cast<double>(n);
    const double base_rank = (1.0 - kDampingFactor) / static_cast<double>(n);

    std::vector<double> rank(n, initial_rank);
    std::vector<double> next_rank(n, 0.0);

    double diff = 0.0;
    int iter = 0;
    for (iter = 1; iter <= kMaxIterations; ++iter) {
        double dead_rank_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            if (graph.row_ptr[i] == graph.row_ptr[i + 1]) {
                dead_rank_sum += rank[i];
            }
        }

        const double dead_contribution =
            kDampingFactor * dead_rank_sum / static_cast<double>(n);
        std::fill(next_rank.begin(), next_rank.end(), base_rank + dead_contribution);

        for (int block_start = 0; block_start < n; block_start += kBlockSize) {
            const int block_end = std::min(block_start + kBlockSize, n);
            for (int src = block_start; src < block_end; ++src) {
                const int begin = graph.row_ptr[src];
                const int end = graph.row_ptr[src + 1];
                const int degree = end - begin;
                if (degree == 0) {
                    continue;
                }

                const double contribution =
                    kDampingFactor * rank[src] / static_cast<double>(degree);
                for (int edge = begin; edge < end; ++edge) {
                    next_rank[graph.col_idx[edge]] += contribution;
                }
            }
        }

        const double total_rank =
            std::accumulate(next_rank.begin(), next_rank.end(), 0.0);
        if (total_rank > 0.0) {
            for (double& value : next_rank) {
                value /= total_rank;
            }
        }

        diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff += std::abs(next_rank[i] - rank[i]);
        }

        rank.swap(next_rank);
        if (diff < kTolerance) {
            break;
        }
    }

    if (iterations != nullptr) {
        *iterations = std::min(iter, kMaxIterations);
    }
    if (residual != nullptr) {
        *residual = diff;
    }
    return rank;
}

void WriteTopRanks(const std::string& output_path, const Graph& graph,
                   const std::vector<double>& rank) {
    std::vector<int> order(rank.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        if (rank[lhs] != rank[rhs]) {
            return rank[lhs] > rank[rhs];
        }
        return graph.node_ids[lhs] < graph.node_ids[rhs];
    });

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + output_path);
    }

    output << std::setprecision(12);
    const int limit = std::min(kTopK, static_cast<int>(order.size()));
    for (int i = 0; i < limit; ++i) {
        const int idx = order[i];
        output << graph.node_ids[idx] << ' ' << rank[idx] << '\n';
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string input_path = argc >= 2 ? argv[1] : "Data.txt";
    const std::string output_path = argc >= 3 ? argv[2] : "Res.txt";

    try {
        const Graph graph = ReadGraph(input_path);

        int iterations = 0;
        double residual = 0.0;
        const std::vector<double> rank = ComputePageRank(graph, &iterations, &residual);
        WriteTopRanks(output_path, graph, rank);

        std::cerr << "nodes: " << graph.node_ids.size()
                  << ", edges: " << graph.col_idx.size()
                  << ", iterations: " << iterations
                  << ", residual: " << std::scientific << residual << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
