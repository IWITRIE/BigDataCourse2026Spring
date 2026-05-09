#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr double kDampingFactor = 0.85;
constexpr double kTolerance = 1e-12;
constexpr int kMaxIterations = 1000;
constexpr int kTopK = 100;
constexpr int kBlockSize = 1024;
constexpr int kInvalidIndex = -1;
constexpr std::size_t kMaxOpenWriters = 128;

struct EdgeRecord {
    int source = 0;
    int target_local = 0;
};

struct BlockFile {
    int begin = 0;
    int end = 0;
    std::int64_t edge_count = 0;
    std::filesystem::path path;
};

struct TempDirectory {
    std::filesystem::path path;

    TempDirectory() = default;
    explicit TempDirectory(std::filesystem::path value) : path(std::move(value)) {}

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    TempDirectory(TempDirectory&& other) noexcept : path(std::move(other.path)) {
        other.path.clear();
    }

    TempDirectory& operator=(TempDirectory&& other) noexcept {
        if (this != &other) {
            Cleanup();
            path = std::move(other.path);
            other.path.clear();
        }
        return *this;
    }

    ~TempDirectory() {
        Cleanup();
    }

    void Cleanup() noexcept {
        if (path.empty()) {
            return;
        }
        std::error_code error;
        std::filesystem::remove_all(path, error);
        path.clear();
    }
};

struct Graph {
    std::vector<std::int64_t> node_ids;
    std::vector<int> out_degree;
    std::vector<double> inverse_out_degree;
    std::vector<int> dead_nodes;
    std::vector<BlockFile> blocks;
    TempDirectory temp_dir;
    std::int64_t edge_count = 0;
};

struct OpenWriter {
    std::size_t block_index = 0;
    std::uint64_t tick = 0;
    std::ofstream stream;
};

std::ifstream OpenInputFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    return input;
}

TempDirectory MakeTempDirectory() {
    const auto timestamp =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    const std::filesystem::path path =
        std::filesystem::current_path() / (".pagerank_block_cache_" + std::to_string(timestamp));
    std::filesystem::create_directories(path);
    return TempDirectory(path);
}

std::ofstream& AcquireWriter(std::vector<OpenWriter>& writers,
                             const std::vector<BlockFile>& blocks,
                             std::size_t block_index,
                             std::uint64_t tick) {
    for (OpenWriter& writer : writers) {
        if (writer.block_index == block_index) {
            writer.tick = tick;
            return writer.stream;
        }
    }

    if (writers.size() < std::min(kMaxOpenWriters, blocks.size())) {
        writers.push_back(OpenWriter{});
        OpenWriter& writer = writers.back();
        writer.block_index = block_index;
        writer.tick = tick;
        writer.stream.open(blocks[block_index].path, std::ios::binary | std::ios::app);
        if (!writer.stream) {
            throw std::runtime_error("failed to open block file: " + blocks[block_index].path.string());
        }
        return writer.stream;
    }

    std::size_t victim = 0;
    for (std::size_t i = 1; i < writers.size(); ++i) {
        if (writers[i].tick < writers[victim].tick) {
            victim = i;
        }
    }

    OpenWriter& writer = writers[victim];
    writer.stream.close();
    writer.block_index = block_index;
    writer.tick = tick;
    writer.stream.open(blocks[block_index].path, std::ios::binary | std::ios::app);
    if (!writer.stream) {
        throw std::runtime_error("failed to open block file: " + blocks[block_index].path.string());
    }
    return writer.stream;
}

Graph ReadGraph(const std::string& input_path) {
    std::ifstream input = OpenInputFile(input_path);

    std::int64_t from = 0;
    std::int64_t to = 0;
    std::int64_t edge_count = 0;
    std::int64_t max_node_id = 0;
    while (input >> from >> to) {
        max_node_id = std::max(max_node_id, std::max(from, to));
        ++edge_count;
    }

    if (edge_count == 0) {
        throw std::runtime_error("input graph is empty");
    }

    std::vector<std::uint8_t> seen(static_cast<std::size_t>(max_node_id) + 1, 0);
    input = OpenInputFile(input_path);
    while (input >> from >> to) {
        seen[static_cast<std::size_t>(from)] = 1;
        seen[static_cast<std::size_t>(to)] = 1;
    }

    std::vector<std::int64_t> node_ids;
    std::vector<int> id_to_index(static_cast<std::size_t>(max_node_id) + 1, kInvalidIndex);
    for (std::int64_t raw_id = 0; raw_id <= max_node_id; ++raw_id) {
        if (seen[static_cast<std::size_t>(raw_id)] == 0) {
            continue;
        }
        const int index = static_cast<int>(node_ids.size());
        id_to_index[static_cast<std::size_t>(raw_id)] = index;
        node_ids.push_back(raw_id);
    }

    const int n = static_cast<int>(node_ids.size());
    std::vector<int> out_degree(n, 0);
    input = OpenInputFile(input_path);
    while (input >> from >> to) {
        ++out_degree[id_to_index[static_cast<std::size_t>(from)]];
    }

    Graph graph;
    graph.node_ids = std::move(node_ids);
    graph.out_degree = std::move(out_degree);
    graph.inverse_out_degree.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        if (graph.out_degree[i] == 0) {
            graph.dead_nodes.push_back(i);
        } else {
            graph.inverse_out_degree[i] = 1.0 / static_cast<double>(graph.out_degree[i]);
        }
    }
    graph.edge_count = edge_count;
    graph.temp_dir = MakeTempDirectory();

    for (int begin = 0; begin < n; begin += kBlockSize) {
        BlockFile block;
        block.begin = begin;
        block.end = std::min(begin + kBlockSize, n);
        block.path = graph.temp_dir.path / ("block_" + std::to_string(graph.blocks.size()) + ".bin");
        graph.blocks.push_back(std::move(block));
    }

    for (const BlockFile& block : graph.blocks) {
        std::ofstream output(block.path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("failed to create block file: " + block.path.string());
        }
    }

    std::vector<OpenWriter> writers;
    writers.reserve(std::min(kMaxOpenWriters, graph.blocks.size()));
    std::uint64_t tick = 0;
    input = OpenInputFile(input_path);
    while (input >> from >> to) {
        const int source = id_to_index[static_cast<std::size_t>(from)];
        const int target = id_to_index[static_cast<std::size_t>(to)];
        const std::size_t block_index =
            std::min<std::size_t>(target / kBlockSize, graph.blocks.size() - 1);
        const EdgeRecord record{source, target - graph.blocks[block_index].begin};
        std::ofstream& writer = AcquireWriter(writers, graph.blocks, block_index, ++tick);
        writer.write(reinterpret_cast<const char*>(&record), sizeof(record));
        if (!writer) {
            throw std::runtime_error("failed to write block file: " + graph.blocks[block_index].path.string());
        }
        ++graph.blocks[block_index].edge_count;
    }

    return graph;
}

std::vector<double> ComputePageRank(const Graph& graph, int* iterations, double* residual) {
    const int n = static_cast<int>(graph.node_ids.size());
    const double initial_rank = 1.0 / static_cast<double>(n);
    const double base_rank = (1.0 - kDampingFactor) / static_cast<double>(n);

    std::vector<double> rank(n, initial_rank);
    std::vector<double> next_rank(n, 0.0);
    std::vector<EdgeRecord> buffer(65536);
    std::vector<double> block_accumulator(kBlockSize, 0.0);
    std::vector<std::ifstream> block_inputs;
    if (graph.blocks.size() <= kMaxOpenWriters) {
        block_inputs.resize(graph.blocks.size());
        for (std::size_t i = 0; i < graph.blocks.size(); ++i) {
            block_inputs[i].open(graph.blocks[i].path, std::ios::binary);
            if (!block_inputs[i]) {
                throw std::runtime_error("failed to open block file: " + graph.blocks[i].path.string());
            }
        }
    }

    double diff = 0.0;
    int iter = 0;
    for (iter = 1; iter <= kMaxIterations; ++iter) {
        double dead_rank_sum = 0.0;
        for (const int node : graph.dead_nodes) {
            dead_rank_sum += rank[node];
        }

        const double dead_contribution =
            kDampingFactor * dead_rank_sum / static_cast<double>(n);
        const double base_score = base_rank + dead_contribution;

        diff = 0.0;
        for (std::size_t block_index = 0; block_index < graph.blocks.size(); ++block_index) {
            const BlockFile& block = graph.blocks[block_index];
            const int block_length = block.end - block.begin;
            std::fill(block_accumulator.begin(),
                      block_accumulator.begin() + block_length,
                      0.0);

            std::ifstream temporary_input;
            std::istream* block_input = nullptr;
            if (!block_inputs.empty()) {
                block_inputs[block_index].clear();
                block_inputs[block_index].seekg(0, std::ios::beg);
                block_input = &block_inputs[block_index];
            } else {
                temporary_input.open(block.path, std::ios::binary);
                if (!temporary_input) {
                    throw std::runtime_error("failed to open block file: " + block.path.string());
                }
                block_input = &temporary_input;
            }

            while (true) {
                block_input->read(reinterpret_cast<char*>(buffer.data()),
                                  static_cast<std::streamsize>(buffer.size() * sizeof(EdgeRecord)));
                const std::streamsize bytes_read = block_input->gcount();
                const std::size_t record_count =
                    static_cast<std::size_t>(bytes_read / static_cast<std::streamsize>(sizeof(EdgeRecord)));

                for (std::size_t i = 0; i < record_count; ++i) {
                    const EdgeRecord& edge = buffer[i];
                    block_accumulator[edge.target_local] +=
                        rank[edge.source] * graph.inverse_out_degree[edge.source];
                }

                if (!(*block_input)) {
                    if (block_input->eof()) {
                        break;
                    }
                    throw std::runtime_error("failed to read block file: " + block.path.string());
                }
            }

            for (int local = 0; local < block_length; ++local) {
                const int node = block.begin + local;
                next_rank[node] = base_score + kDampingFactor * block_accumulator[local];
                diff += std::abs(next_rank[node] - rank[node]);
            }
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
    const double total_rank = std::accumulate(rank.begin(), rank.end(), 0.0);
    if (total_rank > 0.0) {
        for (double& value : rank) {
            value /= total_rank;
        }
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
                  << ", edges: " << graph.edge_count
                  << ", iterations: " << iterations
                  << ", residual: " << std::scientific << residual << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
