/**
 * Hierarchical softmax is an alternative to the softmax in which the probability
 * of any one outcome depends on a number of model parameters that is only logarithmic
 * in the total number of outcomes.
 *
 * @file skipgram.hh
 * @date 2017/12/20
 *
 */
#ifndef SKIPGRAM_HH
#define SKIPGRAM_HH

#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <boost/progress.hpp>

#include "hierarchical_softmax.hh"

namespace skipgram {

class skipgram {
    private:
        hsm::hierarchical_softmax hSm;

        hsm::mat2d v2;

        int V;

        int d;
        int c;
        int num_epoch;
        float alpha;

    public:
        skipgram();
        skipgram(int d=20, int c=5, int num_epoch=10, float alpha=0.1)
            : d(d), c(c), num_epoch(num_epoch), alpha(alpha) { }

        const void fit(const int V, const std::map<int, int> freqs, std::vector<std::vector<int>> contexts) {
            this->V = V;

            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<float> score(-1.0,1.0);

            v2.resize(V+100);

            for(int i=0; i<V+100; ++i) {
                for(int j=0; j<d; ++j)
                    v2[i].emplace_back(score(mt));
            }

            std::cout << "Encoding..." << std::endl;
            hSm = hsm::hierarchical_softmax(V, d);
            std::vector<hsm::code_info> code_table = hSm.encode(freqs);

            int epoch = 0;

            std::cout << "Training Skipgram..." << std::endl;

            int mini_batch = std::min(128, (int)contexts.size());

            const unsigned long expected_count = mini_batch * num_epoch;
            boost::progress_display show_progress(expected_count);

            while(epoch < num_epoch) {
                std::random_shuffle(contexts.begin(), contexts.end());
                for(int i=0; i<mini_batch; ++i) {
                    auto context = contexts[i];
                    // loss += train(context);
                    ++show_progress;
                }
                // std::cout << "Epoch: " << epoch << " / " << num_epoch <<  ", Loss: " << loss << std::endl;
                // std::cout << loss / contexts.size() << std::endl;

                ++epoch;
            }

        }

        void train(const std::vector<int> context) {
            int T = context.size();

            for(int t=0; t<T; ++t) {
                const int w2 = context[t];
                for(int j=-c; j<=c; ++j) {
                    if(t+j<=0) continue;
                    if(t+j>=T) break;

                    const int w1 = context[t+j];
                    // res += log(prob(w1, w2, hSm));
                    // const std::vector<float> grad_f = grad(w1, w2);
                    /**
                    for(int i=0; i<d; ++i) {
                        v2[w1][i] += alpha * grad_f[i];
                        res += grad_f[i];
                    }
                    **/
                    const float h = 0.1;
                    hsm::mat2d v_h;
                    for(int i=0; i<(this->d); ++i) {
                        v_h = v2;
                        v_h[w2][i] += h;
                        const float a = hSm.softmax(w2, w1, v_h);
                        const float b = hSm.softmax(w2, w1, v2);
                        v2[w1][i] = alpha * (a - b);
                    }

                }
            }
        }

        const std::vector<float> grad(int w1, int w2) {
            constexpr float h = 0.1;

            std::vector<float> res(this->d);

            hsm::mat2d v_h;
            for(int i=0; i<(this->d); ++i) {
                v_h = v2;
                v_h[w2][i] += h;
                const float a = hSm.softmax(w2, w1, v_h);
                const float b = hSm.softmax(w2, w1, v2);
                v2[w1][i] = alpha * (a - b);
            }

            return res;
        }

        const std::vector<std::vector<float>> get_vect() const {
            return this->v2;
        }
};

#endif

} //namespace skipgram
