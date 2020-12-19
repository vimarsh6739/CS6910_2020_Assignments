import gensim.downloader as api # package to download text corpus
import nltk # text processing
from nltk.corpus import stopwords
import string
from tqdm import tqdm

# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')

# collect all words to be removed
stop = stopwords.words('english') + list(string.punctuation)

actual_words = []
cleaned_words = []
unique_words = set()

# remove stop words
print('removing stop words from text corpus')
for words in tqdm(data):
    current_nonstop_words = [w for w in words if w not in stop]
    cleaned_words += current_nonstop_words
    actual_words += words

    for ns in current_nonstop_words:
        unique_words.add(ns)

# print statistics
print(len(actual_words), 'words BEFORE cleaning stop words and punctuations')
print(len(cleaned_words), 'words AFTER cleaning stop words and punctuations')
print('vocabulary size: ', len(unique_words))

# Create word dictionary mapping
word_to_ix = {word: i for i, word in enumerate(unique_words)}
ix_to_word = {i: word for i, word in enumerate(unique_words)}

import pickle

# Save cleaned words
with open('../data/cleaned_words.pickle', 'wb') as handle:
    pickle.dump(cleaned_words, handle)

# Check if saved properly
with open('../data/cleaned_words.pickle', 'rb') as handle:
    temp = pickle.load(handle)
assert temp == cleaned_words, "cleaned_words not saved properly"
print('Saved cleaned_words')

# Save word_to_ix dict
with open('../data/word_to_ix.pickle', 'wb') as handle:
    pickle.dump(word_to_ix, handle)

# Check if saved properly
with open('../data/word_to_ix.pickle', 'rb') as handle:
    temp = pickle.load(handle)
assert temp == word_to_ix, "word_to_ix not saved properly"

print('Saved word_to_ix')

# Save ix_to_word dict
with open('../data/ix_to_word.pickle', 'wb') as handle:
    pickle.dump(ix_to_word, handle)

# Check if saved properly
with open('../data/ix_to_word.pickle', 'rb') as handle:
    temp = pickle.load(handle)
assert temp == ix_to_word, "ix_to_word not saved properly"

print('Saved ix_to_word')
print('Saved successfully')