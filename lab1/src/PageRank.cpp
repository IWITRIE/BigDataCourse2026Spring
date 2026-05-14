#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kDampingFactor = 0.85;
constexpr double kTolerance = 1e-12;
constexpr int kMaxIterations = 1000;
constexpr int kTopK = 100;
constexpr int kBlockSize = 1024;
constexpr std::uint64_t kMaxDenseIdRange = 1000000;

using PackedEdge = std::uint64_t;

struct Graph {
    std::vector<std::int64_t> node_ids;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> out_weight;
    std::vector<int> dead_nodes;
};

std::ifstream OpenInputFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    return input;
}

PackedEdge PackEdge(int from, int to) {
    return (static_cast<PackedEdge>(static_cast<std::uint32_t>(from)) << 32) |
           static_cast<std::uint32_t>(to);
}

int EdgeFrom(PackedEdge edge) {
    return static_cast<int>(edge >> 32);
}

int EdgeTo(PackedEdge edge) {
    return static_cast<int>(edge & 0xffffffffu);
}

class NodeIndexer {
public:
    explicit NodeIndexer(const std::vector<std::int64_t>& node_ids)
        : node_ids_(node_ids) {
        if (node_ids_.empty()) {
            return;
        }

        min_id_ = node_ids_.front();
        const std::int64_t max_id = node_ids_.back();
        if (min_id_ >= 0) {
            const std::uint64_t range =
                static_cast<std::uint64_t>(max_id) -
                static_cast<std::uint64_t>(min_id_) + 1;
            const std::uint64_t density_limit = node_ids_.size() * 4ull;
            if (range > kMaxDenseIdRange || range > density_limit) {
                return;
            }
            dense_index_.assign(static_cast<std::size_t>(range), -1);
            for (int i = 0; i < static_cast<int>(node_ids_.size()); ++i) {
                dense_index_[static_cast<std::size_t>(node_ids_[i] - min_id_)] = i;
            }
        }
    }

    int Find(std::int64_t node_id) const {
        if (!dense_index_.empty()) {
            if (node_id >= min_id_) {
                const std::uint64_t offset =
                    static_cast<std::uint64_t>(node_id) -
                    static_cast<std::uint64_t>(min_id_);
                if (offset < dense_index_.size()) {
                    const int index = dense_index_[static_cast<std::size_t>(offset)];
                    if (index >= 0) {
                        return index;
                    }
                }
            }
            throw std::runtime_error("node id not found while compressing graph");
        }

        const auto iter = std::lower_bound(node_ids_.begin(), node_ids_.end(), node_id);
        if (iter == node_ids_.end() || *iter != node_id) {
            throw std::runtime_error("node id not found while compressing graph");
        }
        return static_cast<int>(iter - node_ids_.begin());
    }

private:
    const std::vector<std::int64_t>& node_ids_;
    std::int64_t min_id_ = 0;
    std::vector<int> dense_index_;
};

std::vector<std::int64_t> ReadNodeIds(const std::string& input_path,
                                      std::size_t* edge_count) {
    std::ifstream input = OpenInputFile(input_path);

    std::vector<std::int64_t> node_ids;
    node_ids.reserve(400000);

    std::int64_t from = 0;
    std::int64_t to = 0;
    std::size_t count = 0;
    while (input >> from >> to) {
        node_ids.push_back(from);
        node_ids.push_back(to);
        ++count;
    }

    if (count == 0) {
        throw std::runtime_error("input graph is empty");
    }

    std::sort(node_ids.begin(), node_ids.end());
    node_ids.erase(std::unique(node_ids.begin(), node_ids.end()), node_ids.end());

    if (edge_count != nullptr) {
        *edge_count = count;
    }
    return node_ids;
}

std::vector<PackedEdge> ReadCompressedEdges(
    const std::string& input_path, const std::vector<std::int64_t>& node_ids,
    std::size_t edge_count) {
    std::ifstream input = OpenInputFile(input_path);
    const NodeIndexer indexer(node_ids);

    std::vector<PackedEdge> edges;
    edges.reserve(edge_count);

    std::int64_t from = 0;
    std::int64_t to = 0;
    while (input >> from >> to) {
        edges.push_back(PackEdge(indexer.Find(from), indexer.Find(to)));
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

Graph ReadGraph(const std::string& input_path) {
    std::size_t edge_count = 0;
    Graph graph;
    graph.node_ids = ReadNodeIds(input_path, &edge_count);

    std::vector<PackedEdge> edges =
        ReadCompressedEdges(input_path, graph.node_ids, edge_count);

    const int n = static_cast<int>(graph.node_ids.size());
    graph.row_ptr.assign(n + 1, 0);
    for (const auto& edge : edges) {
        ++graph.row_ptr[EdgeFrom(edge) + 1];
    }

    for (int i = 0; i < n; ++i) {
        graph.row_ptr[i + 1] += graph.row_ptr[i];
    }

    graph.col_idx.assign(edges.size(), 0);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        graph.col_idx[i] = EdgeTo(edges[i]);
    }

    graph.out_weight.assign(n, 0.0);
    for (int node = 0; node < n; ++node) {
        const int degree = graph.row_ptr[node + 1] - graph.row_ptr[node];
        if (degree == 0) {
            graph.dead_nodes.push_back(node);
        } else {
            graph.out_weight[node] =
                kDampingFactor / static_cast<double>(degree);
        }
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
        for (const int node : graph.dead_nodes) {
            dead_rank_sum += rank[node];
        }

        const double dead_contribution =
            kDampingFactor * dead_rank_sum / static_cast<double>(n);
        std::fill(next_rank.begin(), next_rank.end(), base_rank + dead_contribution);

        for (int block_start = 0; block_start < n; block_start += kBlockSize) {
            const int block_end = std::min(block_start + kBlockSize, n);
            for (int src = block_start; src < block_end; ++src) {
                const int begin = graph.row_ptr[src];
                const int end = graph.row_ptr[src + 1];
                if (begin == end) {
                    continue;
                }

                const double contribution = rank[src] * graph.out_weight[src];
                for (int edge = begin; edge < end; ++edge) {
                    next_rank[graph.col_idx[edge]] += contribution;
                }
            }
        }

        const double total_rank =
            std::accumulate(next_rank.begin(), next_rank.end(), 0.0);
        const double normalize_factor = total_rank > 0.0 ? 1.0 / total_rank : 1.0;
        diff = 0.0;
        for (int i = 0; i < n; ++i) {
            next_rank[i] *= normalize_factor;
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

    const auto better_rank = [&](int lhs, int rhs) {
        if (rank[lhs] != rank[rhs]) {
            return rank[lhs] > rank[rhs];
        }
        return graph.node_ids[lhs] < graph.node_ids[rhs];
    };

    const int limit = std::min(kTopK, static_cast<int>(order.size()));
    std::partial_sort(order.begin(), order.begin() + limit, order.end(),
                      better_rank);

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + output_path);
    }

    output << std::setprecision(12);
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
