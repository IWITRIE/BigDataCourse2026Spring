#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Rating { int user; int item; float rating; };
struct TestBlock { int user; std::vector<int> items; };
struct ResidualEntry { int user; float value; };

static constexpr float SCALE = 20.0f;

std::vector<Rating> load_train(const std::string& path) {
    std::ifstream in(path);
    std::vector<Rating> ratings;
    std::string line;
    int cur = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto bar = line.find('|');
        if (bar != std::string::npos) cur = std::stoi(line.substr(0, bar));
        else {
            size_t pos = 0;
            int item = std::stoi(line, &pos);
            float rating = std::stof(line.substr(pos)) / SCALE;
            ratings.push_back({cur, item, rating});
        }
    }
    return ratings;
}

std::vector<TestBlock> load_test(const std::string& path) {
    std::ifstream in(path);
    std::vector<TestBlock> blocks;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto bar = line.find('|');
        if (bar != std::string::npos) blocks.push_back({std::stoi(line.substr(0, bar)), {}});
        else blocks.back().items.push_back(std::stoi(line));
    }
    return blocks;
}

struct HybridModel {
    int factors = 56;
    int epochs = 40;
    float lr = 0.010f;
    float lr_decay = 0.030f;
    float factor_reg = 0.050f;
    float bias_reg = 0.010f;
    float user_bias_reg = 8.0f;
    float item_bias_reg = 12.0f;
    float item_cf_shrink = 0.50f;
    uint32_t seed = 7;

    int nu = 0, ni = 0;
    float mu = 0.0f;
    std::unordered_map<int, int> umap, imap;
    std::vector<float> bu, bi, P, Q;
    std::vector<float> user_mean_raw;
    std::vector<std::vector<ResidualEntry>> item_residuals;
    std::vector<float> item_norm;
    std::vector<std::vector<int>> user_hist_items;
    std::vector<std::vector<float>> user_hist_resid;
    std::vector<std::vector<ResidualEntry>> raw_item_residuals;
    std::vector<float> raw_item_norm;
    std::vector<std::vector<float>> user_hist_raw_dev;

    float known_pred_idx(int u, int i) const {
        float pred = mu + bu[u] + bi[i];
        const float* pu = &P[static_cast<size_t>(u) * factors];
        const float* qi = &Q[static_cast<size_t>(i) * factors];
        for (int k = 0; k < factors; ++k) pred += pu[k] * qi[k];
        if (pred < 0.5f) pred = 0.5f;
        if (pred > 5.0f) pred = 5.0f;
        return pred;
    }

    void fit(const std::vector<Rating>& ratings) {
        std::vector<int> users, items;
        users.reserve(ratings.size());
        items.reserve(ratings.size());
        for (auto& r : ratings) { users.push_back(r.user); items.push_back(r.item); }
        std::sort(users.begin(), users.end());
        users.erase(std::unique(users.begin(), users.end()), users.end());
        std::sort(items.begin(), items.end());
        items.erase(std::unique(items.begin(), items.end()), items.end());
        nu = static_cast<int>(users.size());
        ni = static_cast<int>(items.size());
        umap.reserve(nu * 2);
        imap.reserve(ni * 2);
        for (int i = 0; i < nu; ++i) umap[users[i]] = i;
        for (int i = 0; i < ni; ++i) imap[items[i]] = i;
        int n = static_cast<int>(ratings.size());
        std::vector<int> u_arr(n), i_arr(n);
        std::vector<float> r_arr(n);
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            u_arr[j] = umap[ratings[j].user];
            i_arr[j] = imap[ratings[j].item];
            r_arr[j] = ratings[j].rating;
            sum += r_arr[j];
        }
        mu = static_cast<float>(sum / n);

        std::vector<float> uc(nu, 0.0f), ic(ni, 0.0f), su(nu), si(ni);
        for (int j = 0; j < n; ++j) { uc[u_arr[j]] += 1.0f; ic[i_arr[j]] += 1.0f; }
        user_mean_raw.assign(nu, mu);
        std::fill(su.begin(), su.end(), 0.0f);
        std::fill(si.begin(), si.end(), 0.0f);
        for (int j = 0; j < n; ++j) { su[u_arr[j]] += r_arr[j]; si[i_arr[j]] += r_arr[j]; }
        for (int u = 0; u < nu; ++u) if (uc[u] > 0.0f) user_mean_raw[u] = su[u] / uc[u];
        bu.assign(nu, 0.0f);
        bi.assign(ni, 0.0f);
        for (int it = 0; it < 25; ++it) {
            std::fill(su.begin(), su.end(), 0.0f);
            for (int j = 0; j < n; ++j) su[u_arr[j]] += r_arr[j] - mu - bi[i_arr[j]];
            for (int u = 0; u < nu; ++u) bu[u] = su[u] / (user_bias_reg + uc[u]);
            std::fill(si.begin(), si.end(), 0.0f);
            for (int j = 0; j < n; ++j) si[i_arr[j]] += r_arr[j] - mu - bu[u_arr[j]];
            for (int i = 0; i < ni; ++i) bi[i] = si[i] / (item_bias_reg + ic[i]);
        }

        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 0.03f);
        P.resize(static_cast<size_t>(nu) * factors);
        Q.resize(static_cast<size_t>(ni) * factors);
        for (float& x : P) x = normal(rng);
        for (float& x : Q) x = normal(rng);

        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        for (int ep = 0; ep < epochs; ++ep) {
            std::shuffle(order.begin(), order.end(), rng);
            float cur_lr = lr / (1.0f + lr_decay * ep);
            for (int idx : order) {
                int u = u_arr[idx], i = i_arr[idx];
                float* pu = &P[static_cast<size_t>(u) * factors];
                float* qi = &Q[static_cast<size_t>(i) * factors];
                float pred = mu + bu[u] + bi[i];
                for (int k = 0; k < factors; ++k) pred += pu[k] * qi[k];
                float err = r_arr[idx] - pred;
                if (err > 3.0f) err = 3.0f;
                if (err < -3.0f) err = -3.0f;
                bu[u] += cur_lr * (err - bias_reg * bu[u]);
                bi[i] += cur_lr * (err - bias_reg * bi[i]);
                for (int k = 0; k < factors; ++k) {
                    float old_p = pu[k], old_q = qi[k];
                    pu[k] = old_p + cur_lr * (err * old_q - factor_reg * old_p);
                    qi[k] = old_q + cur_lr * (err * old_p - factor_reg * old_q);
                }
            }
        }

        item_residuals.assign(ni, {});
        item_norm.assign(ni, 0.0f);
        user_hist_items.assign(nu, {});
        user_hist_resid.assign(nu, {});
        raw_item_residuals.assign(ni, {});
        raw_item_norm.assign(ni, 0.0f);
        user_hist_raw_dev.assign(nu, {});
        for (int j = 0; j < n; ++j) {
            int u = u_arr[j], i = i_arr[j];
            float residual = r_arr[j] - known_pred_idx(u, i);
            item_residuals[i].push_back({u, residual});
            user_hist_items[u].push_back(i);
            user_hist_resid[u].push_back(residual);
            float raw_dev = r_arr[j] - user_mean_raw[u];
            raw_item_residuals[i].push_back({u, raw_dev});
            user_hist_raw_dev[u].push_back(raw_dev);
        }
        for (int i = 0; i < ni; ++i) {
            auto& row = item_residuals[i];
            std::sort(row.begin(), row.end(), [](const ResidualEntry& a, const ResidualEntry& b) {
                return a.user < b.user;
            });
            double ss = 0.0;
            for (const auto& entry : row) ss += entry.value * entry.value;
            item_norm[i] = static_cast<float>(std::sqrt(ss));
        }
        for (int i = 0; i < ni; ++i) {
            auto& row = raw_item_residuals[i];
            std::sort(row.begin(), row.end(), [](const ResidualEntry& a, const ResidualEntry& b) {
                return a.user < b.user;
            });
            double ss = 0.0;
            for (const auto& entry : row) ss += entry.value * entry.value;
            raw_item_norm[i] = static_cast<float>(std::sqrt(ss));
        }
    }

    static double sparse_dot(const std::vector<ResidualEntry>& a, const std::vector<ResidualEntry>& b) {
        size_t i = 0, j = 0;
        double sum = 0.0;
        while (i < a.size() && j < b.size()) {
            if (a[i].user == b[j].user) {
                sum += static_cast<double>(a[i].value) * b[j].value;
                ++i;
                ++j;
            } else if (a[i].user < b[j].user) {
                ++i;
            } else {
                ++j;
            }
        }
        return sum;
    }

    float item_cf_residual(int u, int target) const {
        float target_norm = item_norm[target];
        if (target_norm <= 1e-7f) return 0.0f;
        const auto& hist = user_hist_items[u];
        const auto& resids = user_hist_resid[u];
        const auto& target_row = item_residuals[target];
        double num = 0.0, den = 0.0;
        for (size_t h = 0; h < hist.size(); ++h) {
            int item = hist[h];
            float norm = item_norm[item];
            if (norm <= 1e-7f) continue;
            double dot = sparse_dot(item_residuals[item], target_row);
            double sim = dot / (static_cast<double>(norm) * target_norm + 1e-6);
            if (sim > 0.0) { num += sim * resids[h]; den += std::abs(sim); }
        }
        if (den == 0.0) return 0.0f;
        return static_cast<float>(num / (den + item_cf_shrink));
    }

    float raw_item_cf_score(int u, int target, float shrink) const {
        float target_norm = raw_item_norm[target];
        if (target_norm <= 1e-7f) return user_mean_raw[u] * SCALE;
        const auto& hist = user_hist_items[u];
        const auto& devs = user_hist_raw_dev[u];
        const auto& target_row = raw_item_residuals[target];
        double num = 0.0, den = 0.0;
        for (size_t h = 0; h < hist.size(); ++h) {
            int item = hist[h];
            float norm = raw_item_norm[item];
            if (norm <= 1e-7f) continue;
            double dot = sparse_dot(raw_item_residuals[item], target_row);
            double sim = dot / (static_cast<double>(norm) * target_norm + 1e-6);
            if (sim > 0.0) { num += sim * devs[h]; den += std::abs(sim); }
        }
        float score = user_mean_raw[u];
        if (den > 0.0) score += static_cast<float>(num / (den + shrink));
        if (score < 0.5f) score = 0.5f;
        if (score > 5.0f) score = 5.0f;
        return score * SCALE;
    }

    float predict(int user, int item) const {
        float score = mu;
        auto uit = umap.find(user);
        auto iit = imap.find(item);
        if (uit != umap.end()) score += bu[uit->second];
        if (iit != imap.end()) score += bi[iit->second];
        if (uit != umap.end() && iit != imap.end()) {
            int u = uit->second, i = iit->second;
            const float* pu = &P[static_cast<size_t>(u) * factors];
            const float* qi = &Q[static_cast<size_t>(i) * factors];
            for (int k = 0; k < factors; ++k) score += pu[k] * qi[k];
            score += item_cf_residual(u, i);
        }
        if (score < 0.5f) score = 0.5f;
        if (score > 5.0f) score = 5.0f;
        return score * SCALE;
    }
};
int main(int argc, char** argv) {
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    std::string train_path = "/home/weilai/my_files/BigDataCourse2026Spring/lab2/data/train.txt";
    std::string test_path = "/home/weilai/my_files/BigDataCourse2026Spring/lab2/data/test.txt";
    std::string out_path = "/home/weilai/my_files/BigDataCourse2026Spring/lab2/result/Result.txt";
    for (int i = 1; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") train_path = argv[++i];
        else if (arg == "--test") test_path = argv[++i];
        else if (arg == "--output") out_path = argv[++i];
    }
    auto ratings = load_train(train_path);
    auto test = load_test(test_path);
    HybridModel m;
    m.fit(ratings);
    const float alpha = 1.12f;
    const float rawb = 0.36f;
    const float rawsh = 0.60f;
    const float center = m.mu * SCALE;
    std::vector<std::vector<float>> scores(test.size());
    for (size_t b = 0; b < test.size(); ++b) {
        scores[b].resize(test[b].items.size());
    }
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < static_cast<int>(test.size()); ++b) {
        int user = test[b].user;
        for (size_t j = 0; j < test[b].items.size(); ++j) {
            int item = test[b].items[j];
            float base = m.predict(user, item);
            float rawknn = base;
            auto uit = m.umap.find(user);
            auto iit = m.imap.find(item);
            if (uit != m.umap.end() && iit != m.imap.end()) {
                rawknn = m.raw_item_cf_score(uit->second, iit->second, rawsh);
            }
            float score = center + alpha * (base - center);
            score = (1.0f - rawb) * score + rawb * rawknn;
            if (score < 10.0f) score = 10.0f;
            if (score > 100.0f) score = 100.0f;
            scores[b][j] = score;
        }
    }
    std::ofstream out(out_path);
    out << std::fixed << std::setprecision(6);
    for (size_t b = 0; b < test.size(); ++b) {
        out << test[b].user << "|" << test[b].items.size() << "\n";
        for (size_t j = 0; j < test[b].items.size(); ++j) out << test[b].items[j] << " " << scores[b][j] << "\n";
    }
    return 0;
}
