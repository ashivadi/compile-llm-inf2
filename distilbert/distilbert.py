import torch
import torch_neuronx
from transformers import DistilBertTokenizer, DistilBertModel

# Create the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()

# Get an example input
text = "Replace me by any text you'd like."

encoded_input = tokenizer(
    text,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

example = (
    encoded_input['input_ids'],
    encoded_input['attention_mask'],
)

# Run inference on CPU
output_cpu = model(*example)

# Compile the model
model_neuron = torch_neuronx.trace(model, example)

# Save the TorchScript for inference deployment
filename = 'bert_model.pt'
torch.jit.save(model_neuron, filename)
print(f'Model saved at {filename}')
