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

namespace hsm {

using mat2d = std::vector<std::vector<double>>;
using mat3d = std::vector<std::vector<std::vector<double>>>;

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

            pth.push_back(std::make_pair(idx, edge_count));

            if(nd->left == nullptr && nd->right == nullptr) {
                table[nd->value] = code_info(code, code_size);
                paths[nd->value] = pth;
            } else {
                dfs(table, nd->left, (code<<1), code_size+1, pth);
                dfs(table, nd->right, (code<<1)+1, code_size+1, pth);
            }
        }

        const double sigmoid(double x) const {
            if(x>6) return 1.0;
            else if(x<-6) return 0.0;
            else return 1 / (1+std::exp(-x));
        }

        const double prod(std::vector<double> v1, std::vector<double> v2) const {
            assert(v1.size() == v2.size());

            double res = 0.0;
            for(int i=0; i<v1.size(); ++i) res += v1[i]*v2[i];

            return res;
        }

    public:
        hierarchical_softmax(int V) : V(V) { }

        std::vector<code_info> encode(std::map<int, int> freqs) {
            std::vector<node*> nodes(V);
            for(int i=0; i<V; ++i) {
                nodes[i] = new node();
                nodes[i]->value = i;
            }

            for(auto iter=freqs.begin(); iter!=freqs.end(); ++iter) {
                nodes[iter->first]->freq = iter->second;
            }

            std::priority_queue<node*> heap;
            for(auto nd : nodes) heap.push(nd);

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

        double softmax(int id, mat3d v1, mat2d v2) {
            double res = 1.0;

            for(int j=0; j<paths[id].size()-1; ++j) {
                int w = paths[id][j].first;
                int edges = paths[id][j].second;

                res *= sigmoid((1 / edges) * prod(v1[w][j], v2[id]));
            }

            return res;
        }

        const void print_paths() const {
            for(auto iter=paths.begin(); iter!=paths.end(); ++iter) {
                std::cout << iter->first << ":" << std::endl;
                for(auto v : iter->second) {
                    std::cout << "->" << v.first;
                }
                std::cout << std::endl;
            }
        }
};

} // namespace hsm

#endif
