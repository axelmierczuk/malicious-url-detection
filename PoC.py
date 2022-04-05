import numpy as np

from main import API
import pandas as pd


def main():
    # MALICIOUS / BENIGN URLs (NOT used in training)
    # https://www.unb.ca/cic/datasets/url-2016.html
    colnames = ['urls']
    csv_list = ["data/test/benign.csv", "data/test/defacement.csv", "data/test/malware.csv", "data/test/phishing.csv", "data/test/spam.csv"]

    # Load models
    api = API()
    for csv in csv_list:
        api.set_urls(pd.read_csv(csv, names=colnames, header=None)['urls'].tolist())

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
    main()
