mkdir data
wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip stanford-corenlp-full-2017-06-09.zip
unzip glove.840B.300d.zip
rm stanford-corenlp-full-2017-06-09.zip
rm glove.840B.300d.zip
git clone https://github.com/brendano/stanford_corenlp_pywrapper
cd stanford_corenlp_pywrapper
pip install .
mv stanford_corenlp_pywrapper/* ./
cd ..
