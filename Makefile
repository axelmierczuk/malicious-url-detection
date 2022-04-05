setup-data:
	-mkdir data
	cd data && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/Webpages_Classification_test_data.csv.zip && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/Webpages_Classification_train_data.csv.zip && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/malicious_phish.csv.zip && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/lexical_dataset.csv && unzip '*.zip' && rm *.zip

setup-data-test:
	-mkdir data/test
	cd data/test && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/testing/benign.csv && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/testing/defacement.csv && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/testing/malware.csv && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/testing/phishing.csv && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/testing/spam.csv

setup-ngram:
	-mkdir models && mkdir models/ngram
	cd models/ngram && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/ngram/pipeline-1.joblib && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/ngram/pipeline-2.joblib && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/ngram/pipeline-3.joblib