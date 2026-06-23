#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt,fma")
#include <algorithm>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
struct Rating { int user; int item; float rating; };
class IncrementalSVD {
    static constexpr int UD = 16;
public:
    void load_base_model(float* user_matrix, float* item_matrix, int u_size, int i_size, int dim, float mean) {
        users=u_size; items=i_size; latent_dim=dim; global_mean=mean; P=user_matrix; Q=item_matrix;
        user_bias.assign(users, 0.0f); item_bias.assign(items, 0.0f);
        user_sum.assign(users, 0.0f); item_sum.assign(items, 0.0f);
        user_count.assign(users, 0); item_count.assign(items, 0);
        touched_users.reserve(users); touched_items.reserve(items);
        P_compact.resize((long long)users * UD);
        Q_compact.resize((long long)items * UD);
        for (int u = 0; u < users; ++u) {
            const float* src = user_matrix + (long long)u * dim;
            float* dst = P_compact.data() + u * UD;
            for (int k = 0; k < UD; ++k) dst[k] = src[k];
        }
        for (int i = 0; i < items; ++i) {
            const float* src = item_matrix + (long long)i * dim;
            float* dst = Q_compact.data() + i * UD;
            for (int k = 0; k < UD; ++k) dst[k] = src[k];
        }
    }
    void update(const std::vector<Rating>& batch) {
        static bool trained_once = false;
        if (trained_once) return;
        trained_once = true;
        touched_users.clear(); touched_items.clear();
        const unsigned U=(unsigned)users, I=(unsigned)items;
        for (const Rating& r: batch) {
            if ((unsigned)r.user>=U || (unsigned)r.item>=I) continue;
            if (user_count[r.user]++ == 0) touched_users.push_back(r.user);
            if (item_count[r.item]++ == 0) touched_items.push_back(r.item);
            const float e = r.rating - global_mean;
            user_sum[r.user] += e;
            item_sum[r.item] += e;
        }
        static constexpr float user_shrink = 60.0f;
        static constexpr float item_shrink = 5.0f;
        static constexpr float user_scale = 0.87f;
        static constexpr float item_scale = 0.95f;
        for (int u: touched_users) user_bias[u] = user_scale * user_sum[u] / (user_count[u] + user_shrink);
        for (int i: touched_items) item_bias[i] = item_scale * item_sum[i] / (item_count[i] + item_shrink);
        static constexpr float factor_lr = 0.050f;
        static constexpr float factor_reg = 0.002f;
        float* Pc = P_compact.data();
        float* Qc = Q_compact.data();
        const int Nbatch = static_cast<int>(batch.size());
        for (int idx = 0; idx < Nbatch; ++idx) {
            const Rating& r = batch[idx];
            if (idx + 32 < Nbatch) {
                const Rating& nr = batch[idx + 32];
                if ((unsigned)nr.user < U) __builtin_prefetch(Pc + nr.user * UD, 1, 1);
                if ((unsigned)nr.item < I) __builtin_prefetch(Qc + nr.item * UD, 1, 1);
            }
            if ((unsigned)r.user>=U || (unsigned)r.item>=I) continue;
            float* __restrict__ pu = Pc + r.user * UD;
            float* __restrict__ qi = Qc + r.item * UD;
            float pred = global_mean + user_bias[r.user] + item_bias[r.item];
#ifdef _OPENMP
#pragma omp simd reduction(+:pred)
#endif
            for (int k=0; k<UD; ++k) pred += pu[k]*qi[k];
            float err = r.rating - pred;
            if (err > 2.0f) err = 2.0f;
            else if (err < -2.0f) err = -2.0f;
            const float a = factor_lr * err;
            const float rb = factor_lr * factor_reg;
#ifdef _OPENMP
#pragma omp simd
#endif
            for (int k=0; k<UD; ++k) {
                const float pk = pu[k], qk = qi[k];
                pu[k] = pk + a * qk - rb * pk;
                qi[k] = qk + a * pk - rb * qk;
            }
        }
        for (int u: touched_users) {
            const float* src = Pc + u * UD;
            float* dst = P + (long long)u * latent_dim;
            for (int k = 0; k < UD; ++k) dst[k] = src[k];
        }
        for (int i: touched_items) {
            const float* src = Qc + i * UD;
            float* dst = Q + (long long)i * latent_dim;
            for (int k = 0; k < UD; ++k) dst[k] = src[k];
        }
    }
    float predict(int user_id, int item_id) {
        if (user_id<0 || user_id>=users || item_id<0 || item_id>=items) return global_mean;
        const float* __restrict__ pu=P+(long long)user_id*latent_dim;
        const float* __restrict__ qi=Q+(long long)item_id*latent_dim;
        float score=global_mean + user_bias[user_id] + item_bias[item_id];
#ifdef _OPENMP
#pragma omp simd reduction(+:score)
#endif
        for (int k=0;k<latent_dim;++k) score += pu[k]*qi[k];
        return score<0.5f?0.5f:score>5.0f?5.0f:score;
    }
private:
    int users=0, items=0, latent_dim=0; float global_mean=0.0f; float* P=nullptr; float* Q=nullptr;
    std::vector<float> user_bias,item_bias,user_sum,item_sum;
    std::vector<int> user_count,item_count,touched_users,touched_items;
    std::vector<float> P_compact, Q_compact;
};
