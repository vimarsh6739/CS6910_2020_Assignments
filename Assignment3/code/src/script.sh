#Create cleaned corpus
python nlp_preprocessing.py

# Train models
python train_embedding.py -m skipgram -d 100 -e 5 -cs 2 -bs 4
python train_embedding.py -m skipgram -d 200 -e 5 -cs 2 -bs 4

#Delete cleaned corpus
rm -rf ../data
