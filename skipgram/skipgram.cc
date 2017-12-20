#ifndef SKIPGRAM_HH
#define SKIPGRAM_HH

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
# include <random>
#include <sstream>
#include <string>
#include <vector>

#include<omp.h>

#include "hierarchical_softmax.hh"

namespace skipgram {

class skipgram {
    private:
        hsm::mat3d v1;
        hsm::mat2d v2;

        int V;

        int d;
        int c;
        int num_epoch;
        float alpha;

    public:
        skipgram();
        skipgram(int d=20, int c=5, int num_epoch=1, float alpha=0.1)
            : d(d), c(c), num_epoch(num_epoch), alpha(alpha) { }

        const void fit(const int V, const std::map<int, int> freqs, const std::vector<std::vector<int>> contexts) {
            std::cout << "init..." << std::endl;
            this->V = V;

            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<float> score(0,1.0);

            for(int i=0; i<2*V; ++i) {
                std::vector<std::vector<float>> tmp2;
                for(int j=0; j<3*log2(V); ++j) {
                    std::vector<float> tmp1;
                    for(int k=0; k<d; ++k) tmp1.emplace_back(score(mt));
                    tmp2.emplace_back(tmp1);
                }
                v1.emplace_back(tmp2);
            }

            for(int i=0; i<V; ++i) {
                std::vector<float> tmp;
                for(int j=0; j<d; ++j) tmp.emplace_back(score(mt));
                v2.emplace_back(tmp);
            }

            hsm::hierarchical_softmax hSm(V);
            std::vector<hsm::code_info> code_table = hSm.encode(freqs);

            int epoch = 0;
            float loss = 0;
            while(epoch < num_epoch) {
                loss = 0.0;
                for(const auto& context : contexts) {
                    loss += train(context, hSm);
                }
                //std::cout << "Epoch: " << epoch << " / " << num_epoch <<  ", Loss: " << loss << std::endl;
                std::cout << loss / contexts.size() << std::endl;
                epoch++;
            }

        }

        const float train(const std::vector<int> context, hsm::hierarchical_softmax hSm) {
            float res = 0.0;
            const int T = context.size();

            for(int t=0; t<T; ++t) {
                const int w2 = context[t];
                for(int j=-c; j<=c; ++j) {
                    if(t+j<=0) continue;
                    if(t+j>=T) break;

                    const int w1 = context[t+j];
                    // res += log(prob(w1, w2, hSm));
                    const std::vector<float> grad_f = grad(w1, w2, hSm);
                    #pragma omp parallel for
                    for(int i=0; i<d; ++i) {
                        this->v2[w1][i] += alpha * grad_f[i];
                        res += grad_f[i];
                    }
                }
            }
            return res;
        }

        const float prob(const int w1, const int w2, hsm::hierarchical_softmax hSm) const {
            return hSm.softmax(w2, w1, v1, v2);
        }

        const float prob(const int w1, const int w2, const hsm::mat2d v, hsm::hierarchical_softmax hSm) const {
            return hSm.softmax(w2, w1, v1, v);
        }

        const std::vector<float> grad(int w1, int w2, hsm::hierarchical_softmax hSm) {
            constexpr float h = 0.1;

            std::vector<float> res(this->d);

            hsm::mat2d v_h;
            for(int i=0; i<(this->d); ++i) {
                v_h = v2;
                v_h[w2][i] += h;
                res[i] = (prob(w1, w2, v_h, hSm) - prob(w1, w2, hSm)) / h;
            }

            return res;
        }

        const std::vector<std::vector<float>> get_vect() const {
            return this->v2;
        }
};

#endif

} //namespace skipgram

std::vector<int> split(const std::string &input, char delimiter) {
    std::istringstream stream(input);
    std::string field;
    std::vector<int> result;

    while (getline(stream, field, delimiter))
        result.emplace_back(stoll(field));

    return result;
}

int main() {
    std::map<int, int> freqs;
    std::vector<std::vector<int>> contexts;

    std::string input_file = "./example/sequences.txt";

    int num_epoch = 30;
    int d = 10;
    int c = 5;
    float alpha = 0.1;

    std::ifstream ifs(input_file);
    if (!ifs) {
        std::cout << "File Not Foud..." << std::endl;
        return 1;
    }

    std::string str;
    while(getline(ifs, str)) {
        auto nodes = split(str, ' ');
        std::vector<int> sq;

        for(const auto node : nodes) {
            freqs[node]++;
            sq.emplace_back(node);
        }
        contexts.emplace_back(sq);
    }

    int V = freqs.size();
    std::cout << V << std::endl;

    skipgram::skipgram skg(d, c, num_epoch, alpha);
    skg.fit(V, freqs, contexts);

    auto vec = skg.get_vect();

    for(unsigned i=0; i<vec.size(); ++i) {
        for(unsigned j=0; j<vec[i].size(); ++j) std::cout << vec[i][j] << " ";
        std::cout << std::endl;
    }
}
