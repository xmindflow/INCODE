# INCODE: Implicit Neural Conditioning with Prior Knowledge Embeddings <br> <span style="float: rigth"><sub><sup>WACV 2024</sub></sup></span>

### [Project Page](https://xmindflow.github.io/incode) | [Paper](https://arxiv.org/abs/2310.18846) | [Data](https://drive.google.com/uc?export=download&id=1zOXY05K_E_mtlWLdFgb3N-X3yA2XmJ3J)

INCODE is a new method that improves Implicit Neural Representations (INRs) by dynamically adjusting activation functions using deep prior knowledge. Specifically, INCODE comprises a harmonizer network and a composer network, where the harmonizer network dynamically adjusts key parameters of the composer's activation function. It excels in signal representation, handles various tasks such as audio, image, and 3D reconstructions, and tackles complex challenges like neural radiance fields (NeRFs) and inverse problems (denoising, super-resolution, inpainting, CT reconstruction). 

> [*Amirhossein Kazerouni*](https://amirhossein-kz.github.io/), [*Reza Azad*](https://rezazad68.github.io/), [*Alireza Hosseini*](https://arhosseini77.github.io/), [*Dorit Merhof*](https://scholar.google.com/citations?user=0c0rMr0AAAAJ&hl=en), [*Ulas Bagci*](https://scholar.google.com/citations?user=9LUdPM4AAAAJ&hl=en)
>

<br>

<p align="center">
  <img src="https://github.com/xmindflow/INCODE/assets/61879630/3065d887-6f36-47b3-80ea-239a49a87cb4" width="950">
</p>

## 💥 News 💥
- **`30.10.2023`** | Code is released!
- **`24.10.2023`** | Accepted in WACV 2024! 🥳

## Get started

### Data
You can download the data utilized in the paper from this  [link](https://drive.google.com/uc?export=download&id=1zOXY05K_E_mtlWLdFgb3N-X3yA2XmJ3J).

### Requirements
Install the requirements with:
```bash
pip install -r requirements.txt
```


### Image Representation
The image experiment can be reproduced by running the `train_image.ipynb` notebook.

### Audio Representation
The audio experiment can be reproduced by running the `train_audio.ipynb` notebook.

### Shape Representation
The shape experiment can be reproduced by running the `train_sdf.ipynb` notebook. For your convenience, we have included the occupancy volume of Lucy with regular sampling in 512x512x512 cubes in the data file. 

> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/note.svg">
>   <img alt="Note" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/note.svg">
> </picture><br>
> To test the model with custom input data, you can run the <code>preprocess_sdf.ipynb</code> notebook, which will generate a pre-processed <code>.npy</code> file for your desired input.
>
> <br>
>  The output is a <code>.dae</code> file that can be visualized using software such as Meshlab (a cross-platform visualizer and editor for 3D models).

### Image Denoising
The denoising experiment can be reproduced by running the `train_denoising.ipynb` notebook.

### Image Super-resolution
The super-resolution experiment can be reproduced by running the `train_sr.ipynb` notebook.

### CT Reconstruction
The CT reconstruction experiment can be reproduced by running the `train_ct_reconstruction.ipynb` notebook.

### Image Inpainting
The inpainting experiment can be reproduced by running the `train_inpainting.ipynb` notebook.

## Documentation
If you would like to replace the INCODE with other methods, including `SIREN`, `FINER`, `Gauss`, `ReLU`, `SIREN`, `WIRE`, `WIRE2D`,`FFN`, `MFN`, please refer to the [Readme](https://github.com/xmindflow/INCODE/tree/main/documentation) in the documentation folder.


## Acknowledgement
We thank the authors of [WIRE](https://github.com/vishwa91/wire), [MINER_pl](https://github.com/kwea123/MINER_pl), [torch-ngp](https://github.com/ashawkey/torch-ngp), and [SIREN for inpainting](https://github.com/dalmia/siren/tree/master) for their code repositories.


## Citation
```bibtex
@inproceedings{kazerouni2024incode,
  title={INCODE: Implicit Neural Conditioning with Prior Knowledge Embeddings},
  author={Kazerouni, Amirhossein and Azad, Reza and Hosseini, Alireza and Merhof, Dorit and Bagci, Ulas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1298--1307},
  year={2024}
}
```
