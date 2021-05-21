from tqdm import tqdm
from .data_loading import PartsVectorizer

vectorizer = PartsVectorizer('models/final_model.vec')
for line in tqdm(open('full_concatenation.txt', 'r')):
    vectorizer.add_string(line.split())
vectorizer.dump('models/tokens.txt', 'models/vectors.npy')
