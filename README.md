#### Torch-neuronx library installation steps

```
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install --upgrade neuronx-cc==2.* torch-neuronx torch
pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com
```