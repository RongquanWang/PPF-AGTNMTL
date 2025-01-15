### A multimodal personality prediction framework based on adaptive graph transformer network and multi-task learning

![](./img/tmp4C59.png)

You can install the necessary toolkits by:

```shell
pip install -r requirements.txt
```



Before running the code, please download the extracted data and our trained model checkpoints from [this link](https://drive.google.com/drive/folders/1uHweaOKppG9-2LyOIxppfC-M9hQCw49Y?usp=sharing) and unzip it to the root directory.



You can train the model by:

```shell
python train_encoder.py -d CFIv2
# or
python train_encoder.py -d UDIVA
```

Similarly, you can also view the results of our trained model

```python
python test.py -d CFIv2
# or
python test.py -d UDIVA
```


We extract personality descriptors and transcript features through the following code:

```python
import torch
import clip
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)  #ViT-L/14@336px,ViT-B/32
# text = ['AAA', 'BBB', 'CCC']
text = clip.tokenize(text).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
```

We extract single frame video features through the following code:

```python
import torch
import clip
from PIL import Image
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)  #ViT-L/14@336px,ViT-B/32
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)  # [1,768]
    # image_features /= image_features.norm(dim=-1, keepdim=True)
```

We extract audio features through the following code:

```python
# ！pip install wav2clip
import wav2clip
import numpy as np
import librosa
audio, sr = librosa.load(path, sr=None)
model = wav2clip.get_model(frame_length=audio.shape[0] // 15, hop_length=audio.shape[0] // 15)
embeddings = wav2clip.embed_audio(audio, model)
embeddings = np.swapaxes(embeddings.squeeze(), 0, 1)
```

