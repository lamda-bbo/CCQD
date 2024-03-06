# Sample-Efficient Quality-Diversity by Cooperative Coevolution

The official implementation of our ICLR'24 paper *Sample-Efficient Quality-Diversity by Cooperative Coevolution*.

## Requirements

The implementation is built in [conda](https://www.anaconda.com/).

The environment can be built with

```bash
conda env create -f environment.yml
```

## Running Experiments

Run the following commands to evaluate CCQD on *Humanoid Uni*:

```bash
conda activate ccqd
python -m src algo=CCQD env=humanoid_uni seed=1000
```

You can replace each of the three parameters (i.e., algo, env, and seed) to evaluate different methods on different environments with different seeds.

## License

The code is licensed under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{CCQD,
    author = {Ke Xue and Ren-Jian Wang and Pengyi Li and Dong Li and Jianye Hao and Chao Qian},
    title = {Sample-Efficient Quality-Diversity by Cooperative Coevolution},
    booktitle = {Proceedings of the 12th International Conference on Learning Representations (ICLR'24)},
    year = {2024},
    address = {Vienna, Austria},
}
```
