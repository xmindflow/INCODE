# Documentation

This is a readme file for instructions to replace the INCODE with other methods.

## Encoding

- Frequency Encoding (pos_encode_freq)
    > **Type:** The type of encoding, which is "Frequency."
   
    > **Use Nyquist:** A boolean parameter that determines whether to use the Nyquist frequency. Set to True for using Nyquist, and False for not using it.
    
    > **Mapping Input:** An integer value representing the mapping input.

- Gaussian Encoding (pos_encode_gaus)
   
    > **Type:** The type of encoding, which is "Gaussian."
   
    > **Scale B:** A numerical value representing the scale parameter for the Gaussian encoding.
   
    > **Mapping Input:** An integer value representing the mapping input.

- No Encoding (pos_encode_no)
   
    > **Type:** The type of encoding, which is "None." There are no additional configuration parameters for this method.
   
## Parameters

- INR Type: "gauss"
- in_features: An integer representing the number of input features.
- hidden_features: An integer representing the number of hidden features.
- hidden_layers: An integer representing the number of hidden layers.
- out_features: An integer representing the number of output features.
- pos_encode_configs: The chosen encoding method, e.g., pos_encode_no.



### INCODE
```python
MLP_configs = {
    'model': 'resnet34',
    'truncated_layer': 5,
    'in_channels': 64,
    'hidden_channels': [64, 32, 4],
    'mlp_bias': 0.31,
    'activation_layer': nn.SiLU,
    'GT': torch.rand(1, 3, 224, 224)
}

model = INR('incode').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    first_omega_0=30,
    hidden_omega_0=30,
    pos_encode_configs=pos_encode_no,
    MLP_configs=MLP_configs
)
```

### Gauss

```python
model = INR('gauss').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    sigma=10,
    pos_encode_configs=pos_encode_no
)
```

- ``sigma: A numerical value representing the sigma parameter for Gaussian encoding.``


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

### MFN
```python
model = INR('mfn').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    pos_encode_configs=pos_encode_no
)
```

### SIREN
```python
model = INR('siren').run(
    in_features=2,
    hidden_features=256,
    hidden_layers=3,
    out_features=3,
    first_omega_0=30,
    hidden_omega_0=30,
    pos_encode_configs=pos_encode_no)
```

### WIRE
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

### WIRE2D
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
