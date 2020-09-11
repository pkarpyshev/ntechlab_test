# ntechlab_test
Test tasks for NTechLab internship

### Task 1: Max Subarray
The function for task 1 can be found in [max_subarary.py](./max_subarray.py)

### Task 2: Gender identification nn
The architecture used is **VGG16** based on [this](https://www.sciencedirect.com/science/article/pii/S1877050918307853) article.
Data augmentation was used to improve the quality of prediction:
* Random horizontal flip
* Random affine (rotate, translate, scale)
* Color jitter (contrast, hue, saturation)

All images were resized to 224x224 pixels.

Training was performed for 10 epochs using **SGD** optimizer with learning rate 0.001 and momentum 0.9.

Dataset was divided in 90:10 ratio and split into batches of 50 images.

Accuracy achieved: **96%** on validation dataset.


Usage:
```
python3 network_eval.py folder/with/images
```

The file uses **state.pth** file as *state_dict* and can download it automatically. The file itself can be found [here](https://drive.google.com/uc?id=1--wMPNZBgBV4pEcBEDtOajRotm1LWb5N&export=download)
