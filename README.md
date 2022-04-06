
# Malicious URL Detection

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Datasets](#datasets)
* [Models](#models)
* [Results](#results-4)
* [Future Improvements](#future-improvements)
* [References](#references)

## Overview

In the past, attempts at malicious URL detection have included feature extraction through lexical, or DNS analysis. These are extremely valuable techniques that can provide a well rounded approach to malicious URL analysis. The extracted features can be compared with IOCs, or analyzed through machine learning models / deep neural networks to form well-rounded detection. 

What differentiates this project from others is the development of a neural network capable of analyzing *raw* URLs. This raw analysis model does not leverage any feature extraction, but can be combined with models / IOCs that do in order to provide better coverage. 

As a result, this raw URL analysis model, in combination with n-gram (1, 2, 3), and lexical analysis models out performs many models previously developed.

## Project Structure

```
.
├── LICENSE
├── Makefile
├── PoC.py
├── README.md
├── main.py
├── preprocess
│   ├── __init__.py
│   └── format.py
├── requirements.txt
├── test
│   ├── __init__.py
│   └── test.py
├── train
│   ├── __init__.py
│   ├── train.py
│   ├── train_lexical.py
│   ├── train_ngram.py
│   └── models
│       ├── ResNet_v2_2DCNN.py
│       └── __init__.py
└── util
    ├── __init__.py
    └── util.py
```
## Installation

*`Please note that a pypi package will be published in the future. For now, please follow the instructions bellow.`*

```
git clone https://github.com/axelmierczuk/malicious-url-detection.git
cd malicious-url-detection
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
make all
```

## Usage

For information on training / evaluating models:
```
python3 main.py -h
```

The `PoC.py` file leverages the `API` class imported from `main.py` to analyze URLs with pre-trained models. 

To build a report from existing datasets:

```
python3 PoC.py --report
```

to simply run the models and see accuracy:

```
python3 PoC.py
```

Bellow is example usage for the API class:

```
from main import API

# This will take time to execute as API loads the models
api = API()

# Set the URLs
api.set_urls(['www.google.com'])

# Get model predictions
res_raw, res_ngram_1, res_ngram_2, res_ngram_3, res_lexical, final_arr = api.make_predictions()

# Build report - will output to data/google-analysis.csv
api.build_report("data", "google-analysis")
```

## Datasets

The following datasets were used to train and validate the models:

- https://www.sciencedirect.com/science/article/pii/S2352340920311987?via%3Dihub#ecom0001 **[4]**
- https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset **[5]**


These datasets were compiled into one dataset with an evenly distributed amount of benign and malicious URLs. This final
compiled dataset can be found [here](https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/lexical_dataset.csv).

A 80/10/10 split was used with this dataset to train, validate, and test, respectively.

---------

The following dataset was used to test the models:


- https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls **[2]**


## Models

### Unified URL Analysis

#### Overview

Unified URL analysis leverages the Raw, N-Gram (1, 2, 3), and Lexical URL analysis models prediction to generate a single,
unified score for URL maliciousness. The model works as follows:

```
let α = Raw URL analysis score
let β = N-Gram-1 URL analysis score
let γ = N-Gram-2 URL analysis score
let δ = N-Gram-3 URL analysis score
let ε = Lexical URL analysis score

Malicioussness Score = α + ε + [(β + γ + δ) / 3]
```

This scoring model averages the scores from N-Gram-1,2,3 and adds it to the Raw URL score, and Lexical Score to provide a 
single unified score.

#### Results

| Data Type |  Model   | Detection Rate [P(X) > 0.5] | False Negatives | Processing Time / URL (ms) |
|:---------:|:--------:|:---------------------------:|:---------------:|:--------------------------:|
|  Benign   | Unified  |             N/a             |       N/a       |            N/a             |
| Malicious | Unified  |          0.936210           |      6662       |           28.52            |


### Raw URL Analysis

#### Overview

This model aims to be able to detect malicious URLs from nothing but the URL itself. The model used for training is ResNet18 **[4]**,
a 2D CNN specifically build for image analysis. In order to leverage ResNet18, URLs are first pre-processed into batches of 2D arrays
of shape (1, 175, 175). Sample pre processing code can be seen bellow:

```python
def buildmatrix_raw(self, url):
    char_obj = get_characters()
    main = np.zeros(shape=(1, 4375, 7), dtype=np.byte)
    for cc, char in enumerate(url):
        if cc < 4375:
            main[0][cc] = np.array(char_obj.get(ord(char), [0, 0, 0, 0, 0, 0, 0]), dtype=np.byte)
    return np.reshape(main, (1, 175, 175))
```

By converting each character into bit-form and building the 2D array, URLs can then be interpreted by the model. Let it be known that arrays are 0-padded, and of fixed size meaning
that URLs longer than 4375 characters will be trimmed.

Bellow is a visual representation of ResNet18 and its layers, but feel free to take a look at the Github repo [here](https://github.com/Sakib1263/ResNet-ResNetV2-SEResNet-ResNeXt-SEResNeXt-1D-2D-Tensorflow-Keras).


![img](https://bitb-detection.s3.amazonaws.com/models/url-detection/model.png)

For more details on the training, pre-processing, and model, please take a look at the following files:

```
.
├── preprocess
│   └── format.py
└── train
    ├── train.py
    └── models
        └──  ResNet_v2_2DCNN.py

```

#### Results

| Data Type | Model | Detection Rate [P(X) > 0.5] | False Negatives |
|:---------:|:-----:|:---------------------------:|:---------------:|
|  Benign   |  Raw  |             N/a             |       N/a       |                        
| Malicious |  Raw  |          0.950823           |      5136       |


### N-Gram URL Analysis

#### Overview

The n-gram (1, 2, 3) URL analysis leverages decision trees to predict the class of URLs after having been split
into n-grams. The pipeline is build using sklearn as follows (where self.ngram is the count - 1, 2, or 3):

```python
def build_pipeline(self):
    print("Building pipeline.")
    return Pipeline([('vect', CountVectorizer(ngram_range=(self.ngram, self.ngram))), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier(max_depth=10000, random_state=0))])
```

#### Results

| Data Type |  Model   | Detection Rate [P(X) > 0.5] | False Negatives |
|:---------:|:--------:|:---------------------------:|:---------------:|
|  Benign   | N-Gram 1 |             N/a             |       N/a       |
|  Benign   | N-Gram 2 |             N/a             |       N/a       |
|  Benign   | N-Gram 3 |             N/a             |       N/a       |
| Malicious | N-Gram 1 |          0.992655           |       767       |
| Malicious | N-Gram 2 |          0.538769           |      48170      |
| Malicious | N-Gram 3 |          0.205413           |      82985      |

### Lexical URL Analysis

#### Overview

The Lexical URL analysis model leverages feature extraction, and decision trees to generate score prediction. The features
used are inspired by the research paper "Malicious URL Detection based on Machine Learning" **[1]** and include the following:


- Subdomain
- Len
- IsPort
- NumDigits
- NumChars
- PeriodChar
- DashChar
- AtChar
- TidelChar
- UnderscoreChar
- PercentChar
- AmpersandChar
- HashChar
- PathLen
- DoubleSlash
- QueryLen
- Entropy

These features are extracted as follows:

```python
def buildmatrix_lexical(self, url, label):
    main = np.zeros(shape=(self.tensor_root + 1), dtype=np.float32)
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'http://' + url
    try:
        parsed = urlparse(url)
    except:
        return None
    if parsed.netloc is None:
        return None
    try:
        p = int(parsed.port)
    except:
        p = None

    main[0] = 1 if tldextract.extract(parsed.netloc).subdomain != "" else 0
    main[1] = len(url)
    main[2] = p if p else 0
    main[3] = sum(c.isdigit() for c in url)
    main[4] = sum(c.isalpha() for c in url)
    main[5] = sum(c == "." for c in url)
    main[6] = sum(c == "-" for c in url)
    main[7] = 1 if "@" in url else 0
    main[8] = 1 if "~" in url else 0
    main[9] = sum(c == "_" for c in url)
    main[10] = sum(c == "%" for c in url)
    main[11] = sum(c == "&" for c in url)
    main[12] = sum(c == "#" for c in url)
    main[13] = len(parsed.path.split('/'))
    main[14] = 1 if "//" in parsed.path else 0
    main[15] = len(parsed.query)
    # Shannon Entropy
    main[16] = shannon(url)
    return main
```

Bellow are the weightings of each feature, from greatest to smallest:


![img](https://bitb-detection.s3.amazonaws.com/models/url-detection/lexical/importance-lexical.png)

#### Results

| Data Type |  Model  | Detection Rate [P(X) > 0.5] | False Negatives |
|:---------:|:-------:|:---------------------------:|:---------------:|
|  Benign   | Lexical  |             N/a            |       N/a      |
| Malicious | Lexical  |          0.540889          |      47957     |

## Results

By leveraging [this](https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls) **[2]** dataset, reports were generated for each of the CSVs which were *not* used to train the models:

| Data Type |  Model   | Detection Rate [P(X) > 0.5] | False Negatives | Processing Time / URL (ms) |
|:---------:|:--------:|:---------------------------:|:---------------:|:--------------------------:|
|  Benign   |   Raw    |             N/a             |       N/a       |            N/a             |
|  Benign   | N-Gram 1 |             N/a             |       N/a       |            N/a             |
|  Benign   | N-Gram 2 |             N/a             |       N/a       |            N/a             |
|  Benign   | N-Gram 3 |             N/a             |       N/a       |            N/a             |
|  Benign   | Lexical  |             N/a             |       N/a       |            N/a             |
|  Benign   | Unified  |             N/a             |       N/a       |            N/a             |
| Malicious |   Raw    |          0.950823           |      5136       |            N/a             |
| Malicious | N-Gram 1 |          0.992655           |       767       |            N/a             |
| Malicious | N-Gram 2 |          0.538769           |      48170      |            N/a             |
| Malicious | N-Gram 3 |          0.205413           |      82985      |            N/a             |
| Malicious | Lexical  |          0.540889           |      47957      |            N/a             |
| Malicious | Unified  |          0.936210           |      6662       |           28.52            |



## Future Improvements

1. Improvement of overall detection with better training datasets.
2. Implementation of DNS feature extraction to enhance model detections.

## References

**[1]** Cho Do Xuan, Hoa Dinh Nguyen and Tisenko Victor Nikolaevich, “Malicious URL Detection based on Machine Learning” International Journal of Advanced Computer Science and Applications(IJACSA), 11(1), 2020. http://dx.doi.org/10.14569/IJACSA.2020.0110119

**[2]** Kumar, S. (2019, May 31). Malicious and benign urls. Kaggle. Retrieved April 5, 2022, from https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls 

**[3]** Sakib1263, ResNet-ResNetV2-SEResNet-ResNeXt-SEResNeXt-1D-2D-Tensorflow-Keras, (2022), GitHub repository, https://github.com/Sakib1263/ResNet-ResNetV2-SEResNet-ResNeXt-SEResNeXt-1D-2D-Tensorflow-Keras

**[4]** Singh, A.K. “Malicious and Benign Webpages Dataset.” Data in Brief, vol. 32, 2020, p. 106304., https://doi.org/10.1016/j.dib.2020.106304. Accessed 5 Apr. 2022. 

**[5]** Siddhartha, M. (2021, July 23). Malicious urls dataset. Kaggle. Retrieved April 5, 2022, from https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

