"""
This is a sample PoC to demonstrate the capabilities of the model. Either run the program using:

`python3 PoC.py`

Which would yield the following results:

Modeling executed in 2514.361137835 seconds.
DATASET - data/test/urldata.csv
URL Count - 104438
(Model - raw) Detected 95.08% accurately
(Model - ngram_1) Detected 99.26% accurately
(Model - ngram_2) Detected 53.88% accurately
(Model - ngram_3) Detected 20.54% accurately
(Model - lexical) Detected 54.08% accurately
(Model - final) Detected 93.62% accurately
...

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
    csv_list = ["data/test/benign-urldata.csv", "data/test/malicious-urldata.csv"]

    # Load models
    api = API()
    for csv in csv_list:
        api.set_urls(pd.read_csv(csv)['url'].tolist())
        if is_report:
            # Build reports
            print("Building data/test/" + csv[:-4].split("/")[-1] + ".csv")
            api.build_report("data/test", csv[:-4].split("/")[-1])
        else:
            # Generate Predictions
            res_raw, res_ngram_1, res_ngram_2, res_ngram_3, res_lexical, final_arr = api.make_predictions()
            print(f"DATASET - {csv}")
            print(f"URL Count - {len(res_raw)}")
            if csv == csv_list[0]:
                print(f"(Model - raw) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_raw) <= 0.5)[0]) / len(res_raw) * 100)}% accurately")
                print(f"(Model - ngram_1) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_1) <= 0.5)[0]) / len(res_ngram_1) * 100)}% accurately")
                print(f"(Model - ngram_2) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_2) <= 0.5)[0]) / len(res_ngram_2) * 100)}% accurately")
                print(f"(Model - ngram_3) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_3) <= 0.5)[0]) / len(res_ngram_3) * 100)}% accurately")
                print(f"(Model - lexical) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_lexical) <= 0.5)[0]) / len(res_lexical) * 100)}% accurately")
                print(f"(Model - final) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(final_arr) <= 0.5)[0]) / len(final_arr) * 100)}% accurately")
            else:
                print(f"(Model - raw) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_raw) > 0.5)[0]) / len(res_raw) * 100)}% accurately")
                print(f"(Model - ngram_1) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_1) > 0.5)[0]) / len(res_ngram_1) * 100)}% accurately")
                print(f"(Model - ngram_2) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_2) > 0.5)[0]) / len(res_ngram_2) * 100)}% accurately")
                print(f"(Model - ngram_3) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_ngram_3) > 0.5)[0]) / len(res_ngram_3) * 100)}% accurately")
                print(f"(Model - lexical) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(res_lexical) > 0.5)[0]) / len(res_lexical) * 100)}% accurately")
                print(f"(Model - final) Detected {'{0:.2f}'.format(np.count_nonzero(np.where(np.array(final_arr) > 0.5)[0]) / len(final_arr) * 100)}% accurately")
           
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Malicious url detection PoC')
    parser.add_argument('--report', default=False, action='store_true', help="Build a report.")
    main(vars(parser.parse_args(sys.argv[1:]))['report'])
