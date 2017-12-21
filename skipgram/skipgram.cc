/**
 * Hierarchical softmax is an alternative to the softmax in which the probability
 * of any one outcome depends on a number of model parameters that is only logarithmic
 * in the total number of outcomes.
 *
 * @file skipgram.cc
 * @date 2017/12/20
 *
 */

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "skipgram.hh"

std::set<std::string> st;
std::map<std::string, int> mp;
std::map<int, std::string> mp2;

std::vector<int> split(const std::string &input, char delimiter) {
    std::istringstream stream(input);
    std::string field;
    std::vector<int> result;

    while (getline(stream, field, delimiter))
        result.emplace_back(stoll(field));

    return result;
}

void save_emb(const char* filename, std::vector<std::vector<float>> vecs) {
    std::cout << "Saving vector file..." << std::endl;
    FILE *fpw = fopen(filename, "wb");
    for(int j=0, m=vecs.size(); j<m; ++j) {
        auto vec = vecs[j];
        if(mp2.find(j)==mp2.end()) continue;
        fprintf(fpw, "%s,", mp2[j].c_str());
        for(int i=0, n=vec.size(); i<n; ++i) {
            if(i==n-1) fprintf(fpw, "%f\n", vec[i]);
            else fprintf(fpw, "%f,", vec[i]);
        }
    }
}

void arg_err(char *argv[]) {
    printf("Usage: %s [-i input_file] [-o output_file] [-d vector dim] [-c window_size] [-a alpha] [-e num_epoch] [-h] \n", argv[0]);
    exit(1);
}

int main(int argc, char *argv[]) {
    std::string input_file = "./example/sequences.txt";
    char* output_file = (char*)"./out.emb";

    int num_epoch = 10;
    int d = 10;
    int c = 5;
    float alpha = 0.1;

    int opt;
    while((opt = getopt(argc, argv, "i:o:d:c:a:e:h")) != -1) {
        switch(opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'd':
                d = atoi(optarg);
                break;
            case 'c':
                c = atoi(optarg);
                break;
            case 'a':
                alpha = atof(optarg);
                break;
            case 'e':
                num_epoch = atoi(optarg);
                break;
            case 'h':
                arg_err(argv);
                exit(1);
            default:
                arg_err(argv);
                exit(1);
        }
    }

    std::cout << "Model parameters: " << std::endl;
    std::cout << "input file: " << input_file << std::endl;
    std::cout << "output file: " << output_file << std::endl;
    std::cout << "vector dim: " << d << std::endl;
    std::cout << "window size: " << c << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "num epoch: " << num_epoch << std::endl;

    std::map<int, int> freqs;
    std::vector<std::vector<int>> contexts;

    std::ifstream ifs(input_file);
    if (!ifs) {
        std::cout << "File Not Foud..." << std::endl;
        return 1;
    }

    std::cout << "making vocab..." << std::endl;
    int k = 0;
    std::string str;
    while(getline(ifs, str)) {
        auto nodes = split(str, ' ');
        std::vector<int> sq;

        for(const auto node : nodes) {
            if(st.find(std::to_string(node))==st.end()) {
                st.insert(std::to_string(node));
                mp[std::to_string(node)] = k;
                mp2[k] = std::to_string(node);
                ++k;
            }
            freqs[mp[std::to_string(node)]]++;
            sq.emplace_back(mp[std::to_string(node)]);
        }
        contexts.emplace_back(sq);
    }

    int V = freqs.size();
    std::cout << "Number of vocab: " << V << std::endl;

    skipgram::skipgram skg(d, c, num_epoch, alpha);
    skg.fit(V, freqs, contexts);

    auto vecs = skg.get_vect();
    save_emb(output_file, vecs);
}
