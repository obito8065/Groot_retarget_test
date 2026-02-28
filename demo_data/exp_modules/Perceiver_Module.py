from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverTextPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverClassificationDecoder,
)
import torch
import requests
from PIL import Image

# EXAMPLE 1: using the Perceiver to classify texts
# - we define a TextPreprocessor, which can be used to embed tokens
# - we define a ClassificationDecoder, which can be used to decode the
# final hidden states of the latents to classification logits
# using trainable position embeddings
config = PerceiverConfig()
preprocessor = PerceiverTextPreprocessor(config)
decoder = PerceiverClassificationDecoder(
    config,
    num_channels=config.d_latents,
    trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
    use_query_residual=True,
)
model = PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)

# you can then do a forward pass as follows:
tokenizer = PerceiverTokenizer()
text = "hello world"
inputs = tokenizer(text, return_tensors="pt").input_ids

with torch.no_grad():
    outputs = model(inputs=inputs)
logits = outputs.logits
list(logits.shape)

# to train, one can train the model using standard cross-entropy:
criterion = torch.nn.CrossEntropyLoss()

labels = torch.tensor([1])
loss = criterion(logits, labels)

# EXAMPLE 2: using the Perceiver to classify images
# - we define an ImagePreprocessor, which can be used to embed images
config = PerceiverConfig(image_size=224)
preprocessor = PerceiverImagePreprocessor(
    config,
    prep_type="conv1x1",
    spatial_downsample=1,
    out_channels=256,
    position_encoding_type="trainable",
    concat_or_add_pos="concat",
    project_pos_dim=256,
    trainable_position_encoding_kwargs=dict(
        num_channels=256,
        index_dims=config.image_size**2,
    ),
)

model = PerceiverModel(
    config,
    input_preprocessor=preprocessor,
    decoder=PerceiverClassificationDecoder(
        config,
        num_channels=config.d_latents,
        trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
        use_query_residual=True,
    ),
)

# you can then do a forward pass as follows:
image_processor = PerceiverImageProcessor()
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").pixel_values

with torch.no_grad():
    outputs = model(inputs=inputs)
logits = outputs.logits
list(logits.shape)

# to train, one can train the model using standard cross-entropy:
criterion = torch.nn.CrossEntropyLoss()

labels = torch.tensor([1])
loss = criterion(logits, labels)

print(loss.item())