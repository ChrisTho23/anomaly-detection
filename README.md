# anomaly-detection

## Introduction

This project focuses on the challenge of anomaly detection within a modified EMNIST dataset. The EMNIST dataset, being a widely recognized benchmark in the machine learning community, especially in image recognition tasks, provides a complex yet structured ground for exploring anomaly detection techniques. The objective is to detect and analyze introduced in this dataset, showcasing the robustness and adaptability of machine learning models in handling corrupted data.

## Summary

- Altered EMNIST: The original EMNIST dataset has been deliberately corrupted in various ways to simulate anomalies. The modified dataset is made available in [data/corrupted_emnist](https://github.com/ChrisTho23/anomaly-detection/blob/main/data/corrupted_emnist).
- We started by loading the EMNIST dataset, which contains 28x28 grayscale images of letters and digits. Our first step was to familiarize ourselves with the dataset's characteristics. This can be found in the following [jupyter notebook](https://github.com/ChrisTho23/anomaly-detection/blob/main/src/helpter.ipynb)
- After considering various models, we decided to use a Variational Autoencoder (VAE). We chose the VAE for its capability to accurately capture the underlying distribution of each image and thus, reconstruct them with little to no anomalies.
- To detect anomalies, we subtracted the grayscale values of the VAE-reconstructed images from the original images.
We computed an initial anomaly score defined as the sum of the greyscale values of anomalous classified pixels divided by the total number of pixels. Then, we plotted a histogram of these scores and set a reasonable threshold to classify images as altered or unaltered
- We manually inspected all flagged images to determine the types of alterations, which included pixel inversion, noise addition, image interpolation, random overlaps, underscore addition, and dot addition.

Note that the [helper.ipynb](https://github.com/ChrisTho23/anomaly-detection/blob/main/src/helpter.ipynb) contains a comprehensive overview of the entire project that also explains the results and findings.

## Next steps
As part of the ongoing development of our project, we explored additional methods to enhance the accuracy of our Variational Autoencoder (VAE). Our idea was to build a classifier capable of detecting the specific character in each image. By training the VAE separately for each letter, we hypothesized that we could significantly increase the model's accuracy in anomaly detection and reconstruction.
(1) Attempt to Implement OCR for Character Detection: We explored the possibility of using pre-trained OCR models, like TrOCR, to train the VAE separately for each character. However, integrating these OCR models proved challenging, and we couldn't get them to work effectively.
(2) Exploring a K-Means Classifier: As an alternative, we tried using a k-means classifier with 36 classes on the predicted mean and log variance of the latent space for character detection. Unfortunately, this classifier performed poorly and was ultimately discarded.

## Requirements

This project uses poetry for dependency management. If you haven't installed poetry yet, you can do so by following the instructions on their official documentation: [Poetry Installation Guide](https://python-poetry.org/docs/#installation)

### Setting Up the Environment
1. Clone the repository
```bash
git clone https://github.com/ChrisTho23/anomaly-detection.git
cd anomaly-detection
```
2. Install the dependencies using poetry
```bash
poetry install
```
3. Run the project from the [/src](https://github.com/ChrisTho23/anomaly-detection/blob/main/src) folder of the environment
```bash
cd src
poetry run python main.py
```

## Contributors
- Maria Stoelben
- Joao Melo
- Christophe Thomassin