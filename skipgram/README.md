### Skipgram with hSm

### Usage

```bash
$ ./skipgram -h
Usage: ./skipgram [-i input_file] [-o output_file] [-d vector dim] [-c window_size] [-a alpha] [-h]
```

### Instalation & Run

```bash
$ make
$ ./skipgram
$ ./skipgram -i ./example/sequences.txt -o ./out.emb -d 15 -c 5 -a 0.1
Model parameters:
input file: ./example/sequences.txt
output file: ./out.emb
vector dim: 15
window size: 5
alpha: 0
making vocab...
Number of vocab: 10
Encoding...
Training Skipgram...

0%   10   20   30   40   50   60   70   80   90   100%
|----|----|----|----|----|----|----|----|----|----|
***************************************************
Saving vector file...
```
