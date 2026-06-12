#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt,fma")

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Rating {
    int user;
    int item;
    float rating;
};

class IncrementalSVD {
public:
    void load_base_model(float* user_matrix, float* item_matrix,
                         int u_size, int i_size, int dim, float mean) {
        users = u_size;
        items = i_size;
        latent_dim = dim;
        global_mean = mean;
        P = user_matrix;
        Q = item_matrix;
        trained = false;
        incremental = nullptr;
        user_bias.clear();
        item_bias.clear();
        user_corr.clear();
        item_corr.clear();
        user_last.clear();
        item_last.clear();
        user_count.clear();
        item_count.clear();
    }

    void update(const std::vector<Rating>& batch) {
        static bool done = false;
        if (done) return;
        done = true;
        incremental = &batch;
        user_bias.assign(users, 0.0f);
        item_bias.assign(items, 0.0f);
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items)
            return global_mean;
        if (!trained && incremental != nullptr)
            train_incremental();

        const float* __restrict__ pu = P + (long long)user_id * latent_dim;
        const float* __restrict__ qi = Q + (long long)item_id * latent_dim;
        float score = global_mean;
        if (!user_bias.empty())
            score += user_bias[user_id] + item_bias[item_id];
        if (!user_corr.empty())
            score += user_corr[user_id] + item_corr[item_id];
        if (!user_last.empty())
            score += 0.008f * (user_last[user_id] - global_mean)
                   + 0.010f * (item_last[item_id] - global_mean);
        if (!user_count.empty())
            score += -0.0016f * std::log1pf((float)user_count[user_id])
                   +  0.0010f * std::log1pf((float)item_count[item_id]);
#ifdef _OPENMP
#pragma omp simd reduction(+:score)
#endif
        for (int k = 0; k < latent_dim; ++k) score += pu[k] * qi[k];
        return score < 0.5f ? 0.5f : score > 5.0f ? 5.0f : score;
    }

private:
    void train_incremental() {
        trained = true;
        if (incremental == nullptr || incremental->empty()) return;

        const std::vector<Rating>& batch = *incremental;
        const int n = (int)batch.size();
        const int D = latent_dim;
        const int T = thread_count();
        const unsigned U = (unsigned)users;
        const unsigned I = (unsigned)items;

        if (user_bias.size() != (size_t)users) user_bias.assign(users, 0.0f);
        if (item_bias.size() != (size_t)items) item_bias.assign(items, 0.0f);

        static constexpr int epochs = 8;
        static constexpr float lr0 = 0.043f;
        static constexpr float reg = 0.036f;
        static constexpr float bias_reg = 0.010f;

        for (int ep = 0; ep < epochs; ++ep) {
            const float t = (float)ep / epochs;
            const float lr = lr0 * 0.5f * (1.0f + std::cos(3.14159265f * t));
            const float rb = lr * reg;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(T)
#endif
            for (int idx = 0; idx < n; ++idx) {
                const Rating& r = batch[idx];
                if ((unsigned)r.user >= U || (unsigned)r.item >= I) continue;

                float* __restrict__ pu = P + (long long)r.user * D;
                float* __restrict__ qi = Q + (long long)r.item * D;

                float pred = global_mean + user_bias[r.user] + item_bias[r.item];
#ifdef _OPENMP
#pragma omp simd reduction(+:pred)
#endif
                for (int k = 0; k < D; ++k) pred += pu[k] * qi[k];

                float err = r.rating - pred;
                if (err > 1.5f) err = 1.5f;
                else if (err < -1.5f) err = -1.5f;

                const float a = lr * err;
#ifdef _OPENMP
#pragma omp simd
#endif
                for (int k = 0; k < D; ++k) {
                    const float pk = pu[k];
                    const float qk = qi[k];
                    pu[k] = pk + 0.65f * a * qk - rb * pk;
                    qi[k] = qk + 1.45f * a * pk - rb * qk;
                }
                user_bias[r.user] += 0.15f * lr * (err - bias_reg * user_bias[r.user]);
                item_bias[r.item] += 1.15f * lr * (err - bias_reg * item_bias[r.item]);
            }
        }

        fit_residual_corrections(batch, U, I, D);
    }

    void fit_residual_corrections(const std::vector<Rating>& batch,
                                  unsigned U, unsigned I, int D) {
        user_corr.assign(users, 0.0f);
        item_corr.assign(items, 0.0f);
        user_last.assign(users, global_mean);
        item_last.assign(items, global_mean);
        user_count.assign(users, 0);
        item_count.assign(items, 0);

        std::vector<Rating> clean;
        clean.reserve(batch.size());
        std::vector<float> residual;
        residual.reserve(batch.size());

        for (const Rating& r : batch) {
            if ((unsigned)r.user >= U || (unsigned)r.item >= I) continue;
            const float* __restrict__ pu = P + (long long)r.user * D;
            const float* __restrict__ qi = Q + (long long)r.item * D;
            float pred = global_mean + user_bias[r.user] + item_bias[r.item];
#ifdef _OPENMP
#pragma omp simd reduction(+:pred)
#endif
            for (int k = 0; k < D; ++k) pred += pu[k] * qi[k];
            if (pred < 0.5f) pred = 0.5f;
            else if (pred > 5.0f) pred = 5.0f;
            const float err = r.rating - pred;
            user_corr[r.user] += err;
            item_corr[r.item] += err;
            user_last[r.user] = r.rating;
            item_last[r.item] = r.rating;
            ++user_count[r.user];
            ++item_count[r.item];
            clean.push_back(r);
            residual.push_back(err);
        }

        static constexpr float user_shrink = 18.0f;
        static constexpr float item_shrink = 10.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
        for (int u = 0; u < users; ++u)
            user_corr[u] = user_count[u] ? user_corr[u] / (user_count[u] + user_shrink) : 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
        for (int i = 0; i < items; ++i)
            item_corr[i] = item_count[i] ? item_corr[i] / (item_count[i] + item_shrink) : 0.0f;

        for (int iter = 0; iter < 2; ++iter) {
            std::fill(user_corr.begin(), user_corr.end(), 0.0f);
            for (size_t idx = 0; idx < clean.size(); ++idx)
                user_corr[clean[idx].user] += residual[idx] - item_corr[clean[idx].item];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
            for (int u = 0; u < users; ++u)
                user_corr[u] = user_count[u] ? user_corr[u] / (user_count[u] + user_shrink) : 0.0f;

            std::fill(item_corr.begin(), item_corr.end(), 0.0f);
            for (size_t idx = 0; idx < clean.size(); ++idx)
                item_corr[clean[idx].item] += residual[idx] - user_corr[clean[idx].user];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
            for (int i = 0; i < items; ++i)
                item_corr[i] = item_count[i] ? item_corr[i] / (item_count[i] + item_shrink) : 0.0f;
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
        for (int u = 0; u < users; ++u) user_corr[u] *= 0.95f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(16)
#endif
        for (int i = 0; i < items; ++i) item_corr[i] *= 0.95f;
    }

    int thread_count() const {
#ifdef _OPENMP
        return 16;
#else
        return 1;
#endif
    }

    int users = 0, items = 0, latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr, *Q = nullptr;
    bool trained = false;
    const std::vector<Rating>* incremental = nullptr;
    std::vector<float> user_bias, item_bias;
    std::vector<float> user_corr, item_corr;
    std::vector<float> user_last, item_last;
    std::vector<int> user_count, item_count;
};
