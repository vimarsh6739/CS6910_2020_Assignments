#Create cleaned corpus
python nlp_preprocessing.py

# Train models
python train_embedding.py -m cbow -d 100 -e 5 -cs 2 -bs 512
python train_embedding.py -m cbow -d 100 -e 5 -cs 2 -bs 512
python train_embedding.py -m skipgram -d 100 -e 5 -cs 2 -bs 2048
python train_embedding.py -m skipgram -d 200 -e 5 -cs 2 -bs 2048


