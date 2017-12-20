/**
 * Hierarchical softmax is an alternative to the softmax in which the probability
 * of any one outcome depends on a number of model parameters that is only logarithmic
 * in the total number of outcomes.
 *
 * @file hierarchical_softmax_hh
 * @date 2017/12/20
 *
 */
#ifndef HIERARCHICAL_SOFTMAX_HH
#define HIERARCHICAL_SOFTMAX_HH

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include<omp.h>

namespace hsm {

using mat2d = std::vector<std::vector<float>>;
using mat3d = std::vector<std::vector<std::vector<float>>>;

/**
 * @struct
 * Struct of information on the code
 */
struct code_info {
    int code;
    int code_size;

    code_info() : code(0), code_size(0) { }

    code_info(int code, int code_size)
        : code(code), code_size(code_size) { }

    friend std::ostream& operator << (std::ostream& os, const code_info& c) {
        os << c.code << ", " << c.code_size;
        return os;
    }
};

/**
 * @struct
 * Struct of huffman tree node
 */
struct node {
    int value;
    int freq;
    node *left;
    node *right;

    node() : value(0), freq(0), left(nullptr), right(nullptr) { }
    node(int freq, node* left, node* right)
        : value(0), freq(freq), left(left), right(right) { }
};

bool operator <(const node& lhs, const node& rhs) { return lhs.freq < rhs.freq; }
bool operator ==(const node& lhs, const node& rhs) { return lhs.freq == rhs.freq; }
bool operator >(const node& lhs, const node& rhs) { return lhs.freq > rhs.freq; }


/**
 * @class
 * Hierarchical soft max class
 */
class hierarchical_softmax {
    private:
        int V;
        std::map<int, std::vector<std::pair<int, int>>> paths;

        int idx = -1;

        void dfs(std::vector<code_info> &table, node* nd, int code, int code_size, std::vector<std::pair<int, int>> pth) {
            idx++;
            int edge_count = 0;

            if(nd->left!=nullptr) edge_count++;
            if(nd->right!=nullptr) edge_count++;

            pth.emplace_back(std::make_pair(idx, edge_count));

            if(nd->left == nullptr && nd->right == nullptr) {
                table[nd->value] = code_info(code, code_size);
                paths[nd->value] = pth;
            } else {
                dfs(table, nd->left, (code<<1), code_size+1, pth);
                dfs(table, nd->right, (code<<1)+1, code_size+1, pth);
            }
        }

        const float sigmoid(const float x) const {
            if(x>6) return 1.0;
            else if(x<-6) return 0.0;
            else return 1 / (1+std::exp(-x));
        }

        const float dot(const std::vector<float> v1, const std::vector<float> v2) const {
            //std::cout << v1.size() << " " << v2.size() << std::endl;
            assert(v1.size() == v2.size());

            float res = 0.0;
            for(unsigned i=0, n=v1.size(); i<n; ++i) res += v1[i]*v2[i];

            return res;
        }

    public:
        hierarchical_softmax(int V) : V(V) { }

        std::vector<code_info> encode(std::map<int, int> freqs) {
            std::vector<node*> nodes(V);

            #pragma omp parallel for
            for(int i=0; i<V; ++i) {
                nodes[i] = new node();
                nodes[i]->value = i;
            }

            for(auto iter=freqs.begin(); iter!=freqs.end(); ++iter) {
                nodes[iter->first]->freq = iter->second;
            }

            std::priority_queue<node*> heap;
            for(const auto nd : nodes) heap.push(nd);

            while(heap.size() > 1) {
                node *n1 = heap.top(); heap.pop();
                node *n2 = heap.top(); heap.pop();
                heap.push(new node(n1->freq + n2->freq, n2, n1));
            }

            node *root = heap.top(); heap.pop();

            std::vector<code_info> code_table(V);
            dfs(code_table, root, 0, 0, std::vector<std::pair<int, int>>());

            return code_table;
        }

        const float softmax(const int w_i, const int w, const mat3d v1, const mat2d v2) {
            float res = 1.0;
            const unsigned n=paths[w].size()-1;

            for(unsigned j=0; j<n; ++j) {
                const int x = paths[w][j].first;
                const float edges = paths[w][j].second;

                // std::cout << j << " " << x << " " << edges << std::endl;
                if(edges==2)
                    res *= (sigmoid(0.5 * dot(v1[x][j], v2[w_i])));
                else if(edges==1)
                    res *= (sigmoid(dot(v1[x][j], v2[w_i])));
            }

            return res;
        }

        const void print_paths() const {
            for(auto iter=paths.begin(); iter!=paths.end(); ++iter) {
                std::cout << iter->first << ":" << std::endl;
                for(const auto& v : iter->second) {
                    std::cout << "->" << v.first;
                }
                std::cout << std::endl;
            }
        }

        const void print(std::vector<float> vec) {
            for(unsigned i=0; i<vec.size(); ++i) {
                std::cout << vec[i] << ",";
            }
            std::cout << std::endl;
        }
};

} // namespace hsm

#endif
