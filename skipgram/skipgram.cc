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

#include "hierarchical_softmax.hh"

namespace skipgram {

class skipgram {
    private:
        hsm::mat3d v1;
        hsm::mat2d v2;

        int V;

        int c;
        int d;
        int num_epoch;
        double alpha;

    public:
        skipgram();
        skipgram(int d=20, int c=5, int num_epoch=1, double alpha=0.1)
            : d(d), c(c), num_epoch(num_epoch), alpha(alpha) { }

        void fit(int V, std::map<int, int> freqs, std::vector<std::vector<int>> contexts) {
            std::cout << "init..." << std::endl;
            this->V = V;

            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> score(0,1.0);

            for(int i=0; i<2*V; ++i) {
                std::vector<std::vector<double>> tmp2;
                for(int j=0; j<4*log2(V); ++j) {
                    std::vector<double> tmp1;
                    for(int k=0; k<d; ++k) tmp1.push_back(score(mt));
                    tmp2.push_back(tmp1);
                }
                v1.push_back(tmp2);
            }

            for(int i=0; i<V; ++i) {
                std::vector<double> tmp;
                for(int j=0; j<d; ++j) tmp.push_back(score(mt));
                v2.push_back(tmp);
            }

            hsm::hierarchical_softmax hSm(V);
            std::vector<hsm::code_info> code_table = hSm.encode(freqs);

            int epoch = 0;
            while(epoch < num_epoch) {
                double loss = 0;
                for(auto context : contexts) {
                    loss += train(context, hSm);
                }
                //std::cout << "Epoch: " << epoch << " / " << num_epoch <<  ", Loss: " << loss << std::endl;
                std::cout << loss / contexts.size() << std::endl;
                epoch++;
            }

        }

        double train(std::vector<int> context, hsm::hierarchical_softmax hSm) {
            double res = 0.0;
            int T = context.size();

            for(int t=0; t<T; ++t) {
                for(int j=-c; j<=c; ++j) {
                    if(t+j<=0) continue;
                    if(t+j>=T) break;

                    int w1 = context[t+j];
                    int w2 = context[t];
                    // res += log(prob(w1, w2, hSm));
                    std::vector<double> grad_f = grad(w1, w2, hSm);
                    for(int i=0; i<d; ++i) {
                        this->v2[w1][i] += alpha * grad_f[i];
                        res += grad_f[i];
                    }
                }
            }
            return res;
        }

        double prob(int w1, int w2, hsm::hierarchical_softmax hSm) {
            return hSm.softmax(w2, w1, v1, v2);
        }

        double prob(int w1, int w2, hsm::mat2d v, hsm::hierarchical_softmax hSm) {
            return hSm.softmax(w2, w1, v1, v);
        }

        std::vector<double> grad(int w1, int w2, hsm::hierarchical_softmax hSm) {
            double h = 0.1;

            std::vector<double> res(this->d);

            for(int i=0; i<(this->d); ++i) {
                hsm::mat2d v_h = v2;
                // std::copy(v2.begin(), v2.end(), std::back_inserter(v_h));
                v_h[w2][i] += h;
                res[i] = (prob(w1, w2, v_h, hSm) - prob(w1, w2, hSm)) / h;
            }

            return res;
        }

        const std::vector<std::vector<double>> get_vect() const {
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
        result.push_back(stoll(field));

    return result;
}

int main() {
    std::map<int, int> freqs;
    std::vector<std::vector<int>> contexts;

    std::string input_file = "./sequences.txt";

    int num_epoch = 30;
    int d = 10;
    int c = 5;
    double alpha = 0.1;

    std::ifstream ifs(input_file);
    if (!ifs) {
        std::cout << "File Not Foud..." << std::endl;
        return 1;
    }
    std::string str;
    while(getline(ifs, str)) {
        auto nodes = split(str, ' ');
        std::vector<int> sq;
        for(auto node : nodes) {
            freqs[node]++;
            sq.push_back(node);
        }
        contexts.push_back(sq);
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
