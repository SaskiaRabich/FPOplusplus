


<h1 align="center" id="heading">FPO++: Efficient Encoding and Rendering of Dynamic Neural Radiance Fields by Analyzing and Enhancing Fourier PlenOctrees</h1>

<p align="center">
    <p align="center">
		<b><a href="https://cg.cs.uni-bonn.de/person/m-sc-saskia-rabich">Saskia Rabich</a></b>
        &nbsp;·&nbsp;
        <b><a href="https://cg.cs.uni-bonn.de/person/dr-patrick-stotko">Patrick Stotko</a></b>
        &nbsp;·&nbsp;
        <b><a href="https://cg.cs.uni-bonn.de/person/prof-dr-reinhard-klein">Reinhard Klein</a></b>
    </p>
    <p align="center">
        University of Bonn
    </p>
    <h3 align="center">The Visual Computer &nbsp;·&nbsp; Presented at CGI 2024</h3>
    <h3 align="center">
        <a href="https://doi.org/10.1007/s00371-024-03475-3">Paper</a>
        &nbsp; | &nbsp;
        <a href="https://arxiv.org/abs/2310.20710">arXiv</a>
        &nbsp; | &nbsp;
        <a href="https://cg.cs.uni-bonn.de/publication/rabich-2024-fpo">Project Page</a>
		&nbsp; | &nbsp;
        <a href="https://huggingface.co/datasets/WestAI-SC/FPOplusplus">Data</a>
    </h3>
    <div align="center"></div>
</p>

<p align="left">
    This is the repository for the source code of our implementation for "FPO++: Efficient Encoding and Rendering of Dynamic Neural Radiance Fields by Analyzing and Enhancing Fourier PlenOctrees".
</p>


## Abstract

Fourier PlenOctrees have shown to be an efficient representation for real-time rendering of dynamic neural radiance fields (NeRF). Despite its many advantages, this method suffers from artifacts introduced by the involved compression when combining it with recent state-of-the-art techniques for training the static per-frame NeRF models. In this paper, we perform an in-depth analysis of these artifacts and leverage the resulting insights to propose an improved representation. In particular, we present a novel density encoding that adapts the Fourier-based compression to the characteristics of the transfer function used by the underlying volume rendering procedure and leads to a substantial reduction of artifacts in the dynamic model. We demonstrate the effectiveness of our enhanced Fourier PlenOctrees in the scope of quantitative and qualitative evaluations on synthetic and real-world scenes.

## Citation

If you find this code useful for your research, please cite FPO++ as follows:

```
@article{rabich2024FPOplusplus:,
	 title = {FPO++: efficient encoding and rendering of dynamic neural radiance fields by analyzing and enhancing {Fourier} {PlenOctrees}},
	 author = {Saskia Rabich and Patrick Stotko and Reinhard Klein},
	 journal = {The Visual Computer},
	 year = {2024},
	 issn = {1432-2315},
	 doi = {10.1007/s00371-024-03475-3},
	 url = {https://doi.org/10.1007/s00371-024-03475-3},
}
```

## Installation

Clone the repository using

```sh
git clone --recurse-submodule https://github.com/SaskiaRabich/FPOplusplus.git
```

to clone the repository including the relevant submodules for the real-time renderer.

Make sure, the following requirements are installed:

- python (3.8)
- pytorch (1.13)
- cuda (11.7)

### Creating the environment

We recommend using the provided `environment.yml` for installing all necessary dependencies via

```sh
conda env create -f environment.yml
```

Activate the environment with

```sh
conda activate fpoplusplus
```

### Installing svox

With the conda environment activated, run the following from the svox subdirectory to install the differentiable renderer (required for computing and fine-tuning FPOs).

```sh
python setup.py install
```

### Installing volrend

Run the following from the volrend subdirectory to install the real-time renderer (only required for interactive viewing).

```sh
mkdir build && cd build
cmake ..
make -j12
```

## Running the code

PlenOctrees for all timesteps {0, ..., T-1} and data should be provided using the following file structure.
The static PlenOctrees are assumed to have the same size and depth.

```shell
logs
└── example experiment
    ├── 0
    │   ├── octrees
    │   │   └── tree.npz
    │   └── ...
    ├── 1
    │   ├── octrees
    │   │   └── tree.npz
    │   └── ...
    ...
    └── T-1 
        ├── octrees
        │   └── tree.npz
        └── ...
    ...
data
├── example dataset 1
│   ├── example scene 1
│   │   └── ...
│   └── example scene 2
│   ...
└── example dataset 2
    └── ...
...
```

Activate the conda environment and run `./scripts/create_fpo.sh` to build an FPO from the given static PlenOctrees. It will create a subdirectory `octrees` within the experiment's directory and save the result there.

To fine-tune a given FPO, run `./scripts/finetune_fpo.sh`.
Results will be saved in the same subdirectory.

For evaluation, run `./scripts/evaluate_fpo.sh`.

Please **refer to the respective scripts** for scene-specific settings etc.



## Rendering

For interactive real-time rendering, run `./scripts/launch_volrend.sh`.

For offscreen rendering, run `./scripts/launch_volrend_offscreen.sh`.

Please **refer to the respective scripts** for scene-specific settings etc.

The offscreen rendering script requires the following additional input information:

- a path to an intrinsics file containing a 4x4 intrinsics matrix
- a paths to a directory containing pose files with 3x4 or 4x4 extrinsic matrices

## Datasets

We provide settings for a few datasets, which can be downloaded from here:

- [Lego and Walk Datasets](https://huggingface.co/datasets/WestAI-SC/FPOplusplus)
- [NHR Dataset](https://wuminye.github.io/NHR/datasets.html)

### Training static PlenOctrees

To provide static PlenOctrees, the [PlenOctree conversion code](https://github.com/sxyu/plenoctree) can be used. 
Make sure that all PlenOctrees have the same size and depth. 
The position in space does not need to be the same of all PlenOctrees.

## License

This project is provided under the MIT license.

This codebase is built upon the incredible prior work of [Yu et al.](https://alexyu.net/plenoctrees/). Please refer to the following links for the original implementation of [svox](https://github.com/sxyu/svox) and [volrend](https://github.com/sxyu/volrend).
Thank you to all contributors for providing this great project!

## Acknowledgements

This work has been funded by the Federal Ministry of Education and Research under grant no. 01IS22094E WEST-AI, by the Federal Ministry of Education and Research of Germany and the state of North-Rhine Westphalia as part of the Lamarr-Institute for Machine Learning and Artificial Intelligence, and additionally by the DFG project KL 1142/11-2 (DFG Research Unit FOR 2535 Anticipating Human Behavior).

