## Skipgram with Hierarchical Softmax
Skipgram is widely studied in the context of word variance representation learning.  

This is a C ++ implementation of Skip-gram with hierarchical softmax(hSm).  
hSm can speed up parameter update by utilizing the constructed binary tree.

The properties of hSm are as follows:
* The number of vectors that need to be updated is logarithmic order.
* By using the [Huffman tree](https://en.wikipedia.org/wiki/Huffman_coding), the data approaches balanced data at each node.

### Dependencies
* gcc 7.2.0
* boost 1.66.0

### Link
* [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
* [Hierarchical Probabilistic Neural Network Language Model](https://pdfs.semanticscholar.org/39eb/fbb53b041b97332cd351886749c0395037fb.pdf#page=255)
* [word2vec](https://github.com/dav/word2vec)
