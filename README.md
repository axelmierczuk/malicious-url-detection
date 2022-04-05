
# Malicious URL Detection

## Overview

In the past, attempts at malicious URL detection have included feature extraction through lexical, or DNS analysis. These are extremely valuable techniques that can provide a well rounded approach to malicious URL analysis. The extracted features can be compared with IOCs, or analyzed through machine learning models / deep neural networks to form well-rounded detection. 

What differentiates this project from others is the development of a neural network capable of analyzing *raw* URLs. This raw analysis model does not leverage any feature extraction, but can be combined with models / IOCs that do in order to provide fuller coverage. 

As a result, this raw URL analysis model, in combination with n-gram (1, 2, 3), and lexical analysis models out performs many models previously developed.

## Installation

*`Please note that a pypi package will be published in the future. For now, please follow the instructions bellow.`*

```
git clone <repo>
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
api.build_report("data/", "google-analysis")
```

## Models
### Raw URL Analysis

#### Overview

#### Results

### N-Gram URL Analysis

#### Overview

#### Results

### Lexical URL Analysis

#### Overview

#### Results

## Results

By leveraging the [ISCX-URL2016](https://www.unb.ca/cic/datasets/url-2016.html) [2] dataset, reports were generated for each of the CSVs which were *not* used to train the models:

| Data Type  |  Model   | Detection Rate (P(X) > 0.5) | False Negative Rate | Processing Time / URL (ms) |
|:----------:|:--------:|:---------------------------:|:-------------------:|:--------------------------:|
|   Benign   |   Raw    |          0.956243           |        1548         |            N/a             |
|   Benign   | N-Gram 1 |           0.97959           |         722         |            N/a             |
|   Benign   | N-Gram 2 |           0.99262           |         261         |            N/a             |
|   Benign   | N-Gram 3 |           0.99895           |         37          |            N/a             |
|   Benign   | Lexical  |           0.99658           |         121         |            N/a             |
|   Benign   | Overall  |           0.99726           |         96          |           24.83            |
| Defacement |   Raw    |           0.99660           |         328         |            N/a             |
| Defacement | N-Gram 1 |           0.99980           |         19          |            N/a             |
| Defacement | N-Gram 2 |           0.99928           |         69          |            N/a             |
| Defacement | N-Gram 3 |          0.993740           |         604         |            N/a             |
| Defacement | Lexical  |           0.96488           |        3388         |            N/a             |
| Defacement | Overall  |           0.99969           |         30          |           23.92            |
|  Malware   |   Raw    |           0.98539           |         169         |            N/a             |
|  Malware   | N-Gram 1 |           0.99965           |          4          |            N/a             |
|  Malware   | N-Gram 2 |           0.99533           |         54          |            N/a             |
|  Malware   | N-Gram 3 |           0.97579           |         280         |            N/a             |
|  Malware   | Lexical  |           0.99490           |         59          |            N/a             |
|  Malware   | Overall  |           0.99715           |         33          |           24.42            |
|  Phishing  |   Raw    |           0.98916           |         108         |            N/a             |
|  Phishing  | N-Gram 1 |           0.99970           |          3          |            N/a             |
|  Phishing  | N-Gram 2 |           0.95655           |         433         |            N/a             |
|  Phishing  | N-Gram 3 |           0.74782           |        2513         |            N/a             |
|  Phishing  | Lexical  |           0.96357           |         363         |            N/a             |
|  Phishing  | Overall  |           0.99809           |         19          |           27.97            |
|    Spam    |   Raw    |           0.83633           |        1964         |            N/a             |
|    Spam    | N-Gram 1 |           0.83175           |        2019         |            N/a             |
|    Spam    | N-Gram 2 |           0.17033           |        9956         |            N/a             |
|    Spam    | N-Gram 3 |           0.16025           |        10077        |            N/a             |
|    Spam    | Lexical  |           0.55917           |        5290         |            N/a             |
|    Spam    | Overall  |           0.83150           |        2022         |           31.99            |

A more visual example of the data can be seen bellow:

![img](https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/testing/results.png)

_This image describes the density plots of all the "overall" model results._

Each category tends to have a fairly high conviction rate when making predictions other than "phishing". This is also reflected
in the overall accuracy of the combined models. This can likely be associated with the training datasets. Having had
less exposure to phishing URLs, the models are less capable of properly identifying malicious phishing URLs.

## Future Improvements

1. Improvement of phishing URL / overall detection with better training datasets.
2. Implementation of DNS feature extraction to enhance model detections (potentially improve phishing URL detection).

## References

**[1]** Cho Do Xuan, Hoa Dinh Nguyen and Tisenko Victor Nikolaevich, “Malicious URL Detection based on Machine Learning” International Journal of Advanced Computer Science and Applications(IJACSA), 11(1), 2020. http://dx.doi.org/10.14569/IJACSA.2020.0110119

**[2]** Mohammad Saiful Islam Mamun, Mohammad Ahmad Rathore, Arash Habibi Lashkari, Natalia Stakhanova and Ali A. Ghorbani, "Detecting Malicious URLs Using Lexical Analysis", Network and System Security, Springer International Publishing, P467--482, 2016.

**[3]** Sakib1263, ResNet-ResNetV2-SEResNet-ResNeXt-SEResNeXt-1D-2D-Tensorflow-Keras, (2022), GitHub repository, https://github.com/Sakib1263/ResNet-ResNetV2-SEResNet-ResNeXt-SEResNeXt-1D-2D-Tensorflow-Keras

**[4]** Singh, A.K. “Malicious and Benign Webpages Dataset.” Data in Brief, vol. 32, 2020, p. 106304., https://doi.org/10.1016/j.dib.2020.106304. Accessed 5 Apr. 2022. 

**[5]** Siddhartha, M. (2021, July 23). Malicious urls dataset. Kaggle. Retrieved April 5, 2022, from https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

