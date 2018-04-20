#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

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

class unigram {
    private:
        double freq_table[300000] = {0};
        double denorm ;
        double a = 0.75;
    public:
        unigram(std::map<int, int> freq) {
            double den = 0;
            for(auto it=freq.begin(); it!=freq.end(); ++it) {
                freq_table[it->first] = it->second;
                den += ((double)(it->second)*a);
            }
            denorm = den;
        }
        double prob(int w) {
            return freq_table[w]*(a) / denorm;
        }
};


int main() {
    const std::string input_file = "./facebook_n2v.seqs";
    std::map<int, int> freq;

    std::ifstream ifs(input_file);
    if (!ifs) {
        std::cout << "File Not Foud..." << std::endl;
        return 1;
    }
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
            freq[mp[std::to_string(node)]]++;
        }
    }

    unigram uni(freq);
    int target = 0;
    double tmp = 0.0;
    int ave = 0;
    for(int i=0; i<freq.size(); ++i) {
        std::cout << i << ": " << uni.prob(i) << std::endl;
        ave += freq[i];
        if(tmp <= uni.prob(i)) {
            tmp = uni.prob(i);
            target = i;
        }
    }
    ave /= freq.size();
    std::cout << target << ": " << tmp << ", freq: " << freq[target] << std::endl;
    std::cout << "ave freq: " << ave << std::endl;
}
