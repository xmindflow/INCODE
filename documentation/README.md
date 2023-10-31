# Replacing INCODE with Other Methods

This README provides instructions for replacing the INCODE method with alternative methods in our project.

## Introduction

INCODE is the default method used in this project, but you may want to explore other methods for specific use cases or research purposes. Follow the instructions below to make this substitution.

## Encoding

This section provides an overview of the available encoding methods for positional information in the project. You can configure the encoding by setting the corresponding dictionary parameters as shown below:

### Frequency Encoding

```python
# Frequency Encoding
pos_encode_freq = {'type':'frequency', 'use_nyquist': True, 'mapping_input':512}
```

-  **`type:`** The type of encoding, which is "Frequency."
   
-  **`use_nyquist:`** A boolean parameter that determines whether to use the Nyquist frequency. Set to True for using Nyquist, and False for not using it.
    
- **`mmapping_input:`** An integer value for mapping input (image: `int(max(H, W))`, shape: `int(max(H, W, T)`, audio: `len(audio.data)`, CT: `Number of CT measurement)`.


### Gaussian Encoding

```python
# Gaussian Encoding
pos_encode_gaus = {'type':'gaussian', 'scale_B': 10, 'mapping_input': [Choose based on your input data])}
```

- **`type:`** The type of encoding, which is "Gaussian."
   
- **`scale_B:`** Scaling factor of Gaussian matrix.
   
- **`mmapping_input:`** An integer value for mapping input.

### No Encoding

```python
# No Encoding
pos_encode_no = {'type': None}
```

- **`type:`** The type of encoding, which is "None." 

Choose the encoding method that suits your needs by modifying the respective dictionary parameters.

## Shared Hyper-parameters

These parameters are common to all methods in the project.

- **`in_features:`** The number of input features (int).
- **`hidden_features:`** The number of hidden features (int).
- **`hidden_layers:`** The number of hidden layers (int).
- **`out_features:`** The number of output features (int).
- **`pos_encode_configs:`** The chosen encoding method, e.g., pos_encode_no.


## Methods

### INCODE
```python
MLP_configs = {
    'model': 'resnet34',
    'truncated_layer': 5,
    'in_channels': 64,
    'hidden_channels': [64, 32, 4],
    'mlp_bias': 0.3120,
    'activation_layer': nn.SiLU,
    'GT': [input image]
}

model = INR('incode').run(
    in_features=2,
    out_features=3,
    hidden_features=256,
    hidden_layers=3,
    first_omega_0=30,
    hidden_omega_0=30,
    pos_encode_configs=pos_encode_no,
    MLP_configs=MLP_configs
)
```
- **`first_omega_0:`** The value of $\omega$ for the first hidden layer.
- **`hidden_omega_0:`** The value of  $\omega$ for subsequent hidden layers.
- $\text{activation function} : a  \cdot \sin(b \cdot \omega \cdot x + c) + d$
> <picture>
>   <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/light-theme/note.svg">
>   <img alt="Note" src="https://raw.githubusercontent.com/Mqxx/GitHub-Markdown/main/blockquotes/badge/dark-theme/note.svg">
> </picture><br>
> MLP configurations are different for each task, follow the main notebook for your desired task.
---

### Gauss
[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930139.pdf)] [[arXiv](https://arxiv.org/abs/2111.15135)]
```python
model = INR('gauss').run(
    in_features=2,
    out_features=3,
    hidden_features=256,
    hidden_layers=3,
    sigma=10,
    pos_encode_configs=pos_encode_no
)
```

- **`sigma:`** A numerical value representing the sigma parameter for the Gaussian activation function.
- $\text{activation function} : e^{-(\text{sigma} \cdot \text{x}^2)}$

---

### ReLU
```python
model = INR('relu').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    pos_encode_configs=pos_encode_no
)
```

- $\text{activation function} :\text{ReLU}(x) = \max(0, x)$

---

### MFN
[[Paper](https://openreview.net/forum?id=OmtmcPkkhT)] [[GitHub](https://github.com/boschresearch/multiplicative-filter-networks)]
```python
model = INR('mfn').run(
    in_features=2,
    out_features=3,
    hidden_features=256,
    hidden_layers=3,
    pos_encode_configs=pos_encode_no
)
```

$z^{(1)} = g(x; \theta^{(1)})$

$\text{Output} : z^{(i+1)} = (W^{(i)} z^{(i)} + b^{(i)}) \circ g(x; \theta^{(i+1)}), \quad i=1,\ldots,k-1$

$g_j\left(x ; \theta^{(i)}\right)=\exp \left(-\frac{\gamma_j^{(i)}}{2}\left\|x-\mu_j^{(i)}\right\|_2^2\right) \sin \left(\omega_j^{(i)} x+\phi_j^{(i)}\right)$

---

### SIREN
[[Paper](https://arxiv.org/abs/2006.09661)] [[GitHub](https://github.com/vsitzmann/siren)]
```python
model = INR('siren').run(
    in_features=2,
    out_features=3,
    hidden_features=256,
    hidden_layers=3,
    first_omega_0=30,
    hidden_omega_0=30,
    pos_encode_configs=pos_encode_no)
```
- **`first_omega_0:`** The value of $\omega$ for the first hidden layer.
- **`hidden_omega_0:`** The value of  $\omega$ for subsequent hidden layers.
- $\text{activation function} : \sin(\omega \cdot x)$

---

### WIRE
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Saragadam_WIRE_Wavelet_Implicit_Neural_Representations_CVPR_2023_paper.html)] [[GitHub](https://github.com/vishwa91/wire)]
```python
model = INR('wire').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    first_omega_0=30,
    hidden_omega_0=30,
    sigma=10.0,
    wire_type='complex',
    pos_encode_configs=pos_encode_no)
```
- **`first_omega_0:`** The value of $\omega$ for the first hidden layer to control the frequency of the wavelet.
- **`hidden_omega_0:`** The value of  $\omega$ for subsequent hidden layers.
- **`sigma:`** The value of  $\sigma$ to control the spread or width of the wavelet.
- $\text{activation function} : e^{j \omega x} e^{-\left|\sigma x\right|^2}$

#### WIRE2D
```python
model = INR('wire2d').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    first_omega_0=30,
    hidden_omega_0=30,
    sigma=10.0,
    wire_type='complex',
    pos_encode_configs=pos_encode_no)
```
