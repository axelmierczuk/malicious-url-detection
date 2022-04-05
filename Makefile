all:
	make setup-data
	make setup-data-test
	make setup-ngram
	make setup-lexical
	make setup-raw

setup-data:
	-mkdir data
	cd data && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/Webpages_Classification_test_data.csv.zip && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/Webpages_Classification_train_data.csv.zip && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/malicious_phish.csv.zip && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/lexical_dataset.csv && unzip '*.zip' && rm *.zip

setup-data-test:
	-mkdir data/test
	cd data/test && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/dataset/urldata.csv

setup-ngram:
	-mkdir models && mkdir models/ngram
	cd models/ngram && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/ngram/ngram-1.joblib && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/ngram/ngram-2.joblib && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/ngram/ngram-3.joblib

setup-lexical:
	-mkdir models && mkdir models/lexical
	cd models/lexical && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/lexical/lexical-forest.joblib

setup-raw:
	-mkdir models
	cd models && wget https://bitb-detection.s3.amazonaws.com/models/url-detection/raw.zip && unzip '*.zip' && rm *.zip
