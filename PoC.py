"""
This is a sample PoC to demonstrate the capabilities of the model. Either run the program using:

`python3 PoC.py`

Which would yield the following results:

Modeling executed in 839.8494541729888 seconds.
DATASET - data/test/benign.csv
URL Count - 35378
(Model - raw) Detected 95.68% accurately
(Model - ngram_1) Detected 97.72% accurately
(Model - ngram_2) Detected 99.25% accurately
(Model - ngram_3) Detected 99.89% accurately
(Model - lexical) Detected 99.66% accurately
(Model - final) Detected 99.72% accurately
Modeling executed in 2253.941033238996 seconds.
DATASET - data/test/defacement.csv
URL Count - 96457
(Model - raw) Detected 99.66% accurately
(Model - ngram_1) Detected 99.98% accurately
(Model - ngram_2) Detected 99.93% accurately
(Model - ngram_3) Detected 99.37% accurately
(Model - lexical) Detected 96.49% accurately
(Model - final) Detected 99.97% accurately
Modeling executed in 272.26699988599285 seconds.
DATASET - data/test/malware.csv
URL Count - 11566
(Model - raw) Detected 98.53% accurately
(Model - ngram_1) Detected 99.96% accurately
(Model - ngram_2) Detected 99.52% accurately
(Model - ngram_3) Detected 97.57% accurately
(Model - lexical) Detected 99.48% accurately
(Model - final) Detected 99.71% accurately
Modeling executed in 253.37019420499564 seconds.
DATASET - data/test/phishing.csv
URL Count - 9965
(Model - raw) Detected 98.91% accurately
(Model - ngram_1) Detected 99.96% accurately
(Model - ngram_2) Detected 95.64% accurately
(Model - ngram_3) Detected 74.77% accurately
(Model - lexical) Detected 96.35% accurately
(Model - final) Detected 99.80% accurately
Modeling executed in 331.1913627160102 seconds.
DATASET - data/test/spam.csv
URL Count - 12000
(Model - raw) Detected 83.62% accurately
(Model - ngram_1) Detected 82.57% accurately
(Model - ngram_2) Detected 17.03% accurately
(Model - ngram_3) Detected 16.02% accurately
(Model - lexical) Detected 55.91% accurately
(Model - final) Detected 83.14% accurately

Or use the `--report` argument to generate reports. The reports are stored under:

`data/test/report-*.csv`
"""
import argparse
import sys

import numpy as np
from main import API
import pandas as pd


def main(is_report):
    # MALICIOUS / BENIGN URLs (NOT used in training)
    # https://www.unb.ca/cic/datasets/url-2016.html
    colnames = ['urls']
    csv_list = ["data/test/urldata.csv"]

    # Load models
    api = API()
    for csv in csv_list:
        api.set_urls(pd.read_csv(csv, names=colnames, header=None)['urls'].tolist())
        if is_report:
            # Build reports
            print("Building data/test/" + csv[:-4].split("/")[-1] + ".csv")
            api.build_report("data/test", csv[:-4].split("/")[-1])
        else:
            # Generate Predictions
            res_raw, res_ngram_1, res_ngram_2, res_ngram_3, res_lexical, final_arr = api.make_predictions()
            if csv != csv_list[0]:
                print(f"DATASET - {csv}")
                print(f"URL Count - {len(res_raw)}")
                print(f"(Model - raw) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_raw) > 0.5)[0]) / len(res_raw) * 100)}% accurately")
                print(f"(Model - ngram_1) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_1) > 0.5)[0]) / len(res_ngram_1) * 100)}% accurately")
                print(f"(Model - ngram_2) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_2) > 0.5)[0]) / len(res_ngram_2) * 100)}% accurately")
                print(f"(Model - ngram_3) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_3) > 0.5)[0]) / len(res_ngram_3) * 100)}% accurately")
                print(f"(Model - lexical) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_lexical) > 0.5)[0]) / len(res_lexical) * 100)}% accurately")
                print(f"(Model - final) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(final_arr) > 0.5)[0]) / len(final_arr) * 100)}% accurately")
            else:
                print(f"DATASET - {csv}")
                print(f"URL Count - {len(res_raw)}")
                print(f"(Model - raw) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_raw) < 0.5)[0]) / len(res_raw) * 100)}% accurately")
                print(f"(Model - ngram_1) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_1) < 0.5)[0]) / len(res_ngram_1) * 100)}% accurately")
                print(f"(Model - ngram_2) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_2) < 0.5)[0]) / len(res_ngram_2) * 100)}% accurately")
                print(f"(Model - ngram_3) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_3) < 0.5)[0]) / len(res_ngram_3) * 100)}% accurately")
                print(f"(Model - lexical) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_lexical) < 0.5)[0]) / len(res_lexical) * 100)}% accurately")
                print(f"(Model - final) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(final_arr) < 0.5)[0]) / len(final_arr) * 100)}% accurately")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Malicious url detection PoC')
    parser.add_argument('--report', default=False, action='store_true', help="Build a report.")
    main(vars(parser.parse_args(sys.argv[1:]))['report'])
