#include <iostream>
#include <vector>

#include "hierarchical_softmax.hh"

int main() {
    std::map<int, int> freqs;
    freqs[0] = 5;
    freqs[1] = 3;
    freqs[2] = 2;
    freqs[3] = 1;
    freqs[4] = 1;
    freqs[5] = 1;

    hsm::hierarchical_softmax hSm(6);
    auto table = hSm.encode(freqs);
    for(auto c : table) std::cout << c << std::endl;

    hSm.print_paths();

    hsm::mat3d v1(10, std::vector<std::vector<double>>(6));
    for(int i=0; i<10; ++i) {
        for(int j=0; j<6; ++j) {
            for(int k=0; k<10; ++k) v1[i][j].push_back(0.3);
        }
    }
    hsm::mat2d v2(10);
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) v2[i].push_back(0.2);
    }

    double p = 0.0;
    for(int i=0; i<6; ++i) {
        p += hSm.softmax(0, i, v1, v2);
        std::cout << hSm.softmax(0, i, v1, v2) << std::endl;
    }
    std::cout << p << std::endl;
}
