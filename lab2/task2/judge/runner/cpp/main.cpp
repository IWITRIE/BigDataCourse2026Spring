#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

#include "solution.cpp"

struct JudgeHeader {
    char magic[8];
    int32_t version;
    int32_t num_users;
    int32_t num_items;
    int32_t latent_dim;
    int32_t incremental_count;
    int32_t test_count;
    float global_mean;
    float reserved0;
    float reserved1;
};

static const char* MAGIC = "SVDJUDGE";

template <typename T>
void read_exact(std::ifstream& in, T* data, std::size_t count) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
        throw std::runtime_error("failed to read judge data");
    }
}

struct JudgeData {
    JudgeHeader header;
    std::vector<float> user_matrix;
    std::vector<float> item_matrix;
    std::vector<Rating> incremental;
    std::vector<Rating> test;
};

JudgeData load_data(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open judge data: " + path);
    }

    JudgeData data;
    read_exact(in, &data.header, 1);
    if (std::memcmp(data.header.magic, MAGIC, 8) != 0) {
        throw std::runtime_error("invalid judge data magic");
    }
    if (data.header.version != 1) {
        throw std::runtime_error("unsupported judge data version");
    }

    const std::size_t p_count = static_cast<std::size_t>(data.header.num_users) * data.header.latent_dim;
    const std::size_t q_count = static_cast<std::size_t>(data.header.num_items) * data.header.latent_dim;
    data.user_matrix.resize(p_count);
    data.item_matrix.resize(q_count);
    data.incremental.resize(data.header.incremental_count);
    data.test.resize(data.header.test_count);

    read_exact(in, data.user_matrix.data(), p_count);
    read_exact(in, data.item_matrix.data(), q_count);
    read_exact(in, data.incremental.data(), data.incremental.size());
    read_exact(in, data.test.data(), data.test.size());
    return data;
}

double compute_rmse(IncrementalSVD& model, const std::vector<Rating>& ratings) {
    if (ratings.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    long double sqerr = 0.0;
    for (const Rating& r : ratings) {
        const float pred = model.predict(r.user, r.item);
        if (!std::isfinite(pred)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const long double err = static_cast<long double>(r.rating) - pred;
        sqerr += err * err;
    }
    return std::sqrt(static_cast<double>(sqerr / ratings.size()));
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    const std::string data_path = argc >= 2 ? argv[1] : "/data/judge_data.bin";
    const double epsilon = argc >= 3 ? std::stod(argv[2]) : 0.001;
    const int benchmark_runs = argc >= 4 ? std::max(1, std::stoi(argv[3])) : 10;

    try {
        JudgeData data = load_data(data_path);
        std::vector<double> elapsed_runs;
        elapsed_runs.reserve(static_cast<std::size_t>(benchmark_runs));
        double rmse_base = 0.0;
        double rmse_new = 0.0;

        for (int run = 0; run < benchmark_runs; ++run) {
            std::vector<float> user_matrix = data.user_matrix;
            std::vector<float> item_matrix = data.item_matrix;
            IncrementalSVD model;
            model.load_base_model(
                user_matrix.data(),
                item_matrix.data(),
                data.header.num_users,
                data.header.num_items,
                data.header.latent_dim,
                data.header.global_mean
            );
            if (run == 0) {
                rmse_base = compute_rmse(model, data.test);
            }

            const auto start = std::chrono::steady_clock::now();
            model.update(data.incremental);
            const auto end = std::chrono::steady_clock::now();
            elapsed_runs.push_back(std::chrono::duration<double>(end - start).count());

            if (run == 0) {
                rmse_new = compute_rmse(model, data.test);
            }
        }

        const double elapsed = std::accumulate(elapsed_runs.begin(), elapsed_runs.end(), 0.0);
        const bool valid = std::isfinite(rmse_new) && rmse_new <= rmse_base - epsilon;

        std::cout
            << "{\"status\":\"success\","
            << "\"time_sec\":" << elapsed << ","
            << "\"benchmark_runs\":" << benchmark_runs << ","
            << "\"time_aggregation\":\"sum\","
            << "\"time_runs\":[";
        for (std::size_t i = 0; i < elapsed_runs.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << elapsed_runs[i];
        }
        std::cout
            << "],"
            << "\"rmse_base\":" << rmse_base << ","
            << "\"rmse\":" << rmse_new << ","
            << "\"valid\":" << (valid ? "true" : "false")
            << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cout
            << "{\"status\":\"failed\","
            << "\"error\":\"" << json_escape(exc.what()) << "\","
            << "\"time_sec\":null,"
            << "\"rmse_base\":null,"
            << "\"rmse\":null,"
            << "\"valid\":false"
            << "}" << std::endl;
        return 1;
    }
}
