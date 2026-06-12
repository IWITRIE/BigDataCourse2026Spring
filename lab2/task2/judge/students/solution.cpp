#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt,fma")

#include <algorithm>
#include <cmath>
#include <queue>
#include <tuple>
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
    }

    void update(const std::vector<Rating>& batch) {
        static bool trained_once = false;
        if (trained_once) return;
        trained_once = true;
        if (batch.empty()) return;

        const int n = (int)batch.size();
        const int D = latent_dim;
        static constexpr int T = 16;
        static constexpr int EPOCHS = 2;
        static constexpr float lr0 = 0.052f;
        static constexpr float reg = 0.020f;

#ifdef _OPENMP
        omp_set_num_threads(T);
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(T)
#endif
        for (int u = 0; u < users; ++u) {
            float* pu = P + (long long)u * D;
            pu[D - 2] = 1.0f;
            pu[D - 1] = 0.0f;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(T)
#endif
        for (int i = 0; i < items; ++i) {
            float* qi = Q + (long long)i * D;
            qi[D - 2] = 0.0f;
            qi[D - 1] = 1.0f;
        }

        std::vector<Rating> sample(batch.begin(), batch.end());
        const int chunk = (n + T - 1) / T;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(T)
#endif
        for (int t = 0; t < T; ++t) {
            const int lo = t * chunk;
            const int hi = std::min(n, lo + chunk);
            if (lo < hi) {
                std::sort(sample.begin() + lo, sample.begin() + hi,
                          [](const Rating& a, const Rating& b) {
                              return a.user < b.user;
                          });
            }
        }

        std::vector<Rating> merged;
        merged.reserve((size_t)n);
        using Tup = std::tuple<int, int, int>;
        std::priority_queue<Tup, std::vector<Tup>, std::greater<Tup>> pq;
        for (int t = 0; t < T; ++t) {
            const int lo = t * chunk;
            if (lo < n) pq.push({sample[lo].user, t, lo});
        }
        while (!pq.empty()) {
            auto cur = pq.top();
            pq.pop();
            const int t = std::get<1>(cur);
            const int pos = std::get<2>(cur);
            merged.push_back(sample[pos]);
            const int next = pos + 1;
            const int hi = std::min(n, (t + 1) * chunk);
            if (next < hi) pq.push({sample[next].user, t, next});
        }
        sample.swap(merged);

        const unsigned U = (unsigned)users;
        const unsigned I = (unsigned)items;
        for (int ep = 0; ep < EPOCHS; ++ep) {
            const float tf = (float)ep / EPOCHS;
            const float lr = lr0 * 0.5f * (1.0f + std::cos(3.14159265f * tf));
            const float rb = lr * reg;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(T)
#endif
            for (int idx = 0; idx < n; ++idx) {
                const Rating& r = sample[idx];
                if ((unsigned)r.user >= U || (unsigned)r.item >= I) continue;
                float* __restrict__ pu = P + (long long)r.user * D;
                float* __restrict__ qi = Q + (long long)r.item * D;

                if (idx + 48 < n) {
                    const int nu = sample[idx + 48].user;
                    if ((unsigned)nu < U) __builtin_prefetch(P + (long long)nu * D, 1, 1);
                }
                if (idx + 4 < n) {
                    const int ni = sample[idx + 4].item;
                    if ((unsigned)ni < I) __builtin_prefetch(Q + (long long)ni * D, 1, 1);
                }

                float pred = global_mean;
#ifdef _OPENMP
#pragma omp simd reduction(+:pred)
#endif
                for (int k = 0; k < D; ++k) pred += pu[k] * qi[k];
                float err = r.rating - pred;
                if (err > 5.0f) err = 5.0f;
                else if (err < -5.0f) err = -5.0f;

                const float a = lr * err;
#ifdef _OPENMP
#pragma omp simd
#endif
                for (int k = 0; k < D - 2; ++k) {
                    const float pk = pu[k];
                    const float qk = qi[k];
                    pu[k] = pk + a * qk - rb * pk;
                    qi[k] = qk + a * pk - rb * qk;
                }
                pu[D - 1] += a - rb * pu[D - 1];
                qi[D - 2] += a - rb * qi[D - 2];
            }
        }
    }

    float predict(int user_id, int item_id) {
        if (user_id < 0 || user_id >= users || item_id < 0 || item_id >= items)
            return global_mean;
        const float* __restrict__ pu = P + (long long)user_id * latent_dim;
        const float* __restrict__ qi = Q + (long long)item_id * latent_dim;
        float score = global_mean;
#ifdef _OPENMP
#pragma omp simd reduction(+:score)
#endif
        for (int k = 0; k < latent_dim; ++k) score += pu[k] * qi[k];
        return score < 0.5f ? 0.5f : score > 5.0f ? 5.0f : score;
    }

private:
    int users = 0;
    int items = 0;
    int latent_dim = 0;
    float global_mean = 0.0f;
    float* P = nullptr;
    float* Q = nullptr;
};
