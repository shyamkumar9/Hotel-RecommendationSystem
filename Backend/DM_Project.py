


import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer

from nltk import pos_tag

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud,STOPWORDS

import spacy 


from collections import Counter





import pandas as pd
import numpy as np
import string
import re
import time as tm
from tqdm import tqdm

import threading

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')





import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag





df=pd.read_csv('Hotel_Reviews_1.csv')





df.shape





df.head(2)





from collections import Counter
import pandas as pd

hotel_freq = Counter(df['Hotel_Name'])

hotel_freq_sorted = sorted(hotel_freq.items(), key=lambda x: x[1], reverse=True)

for hotel, freq in hotel_freq_sorted:
    print(f'{hotel}: {freq}')





selected_hotels = []
for hotel, freq in hotel_freq_sorted:
    if freq <= 300 and freq >= 150:
        selected_hotels.append(hotel)





len(selected_hotels)



df_filtered = df[df['Hotel_Name'].isin(selected_hotels)]    





df_filtered.head(2)





df_filtered = df_filtered.dropna()
df_filtered = df_filtered.loc[~((df['Positive_Review'] == 'No Positive'))]
df_filtered





df_filtered = df_filtered.reset_index()
df_filtered





df_filtered['Review'] = df_filtered['Positive_Review']





df_filtered['Review']=df_filtered['Review'].str.lower()
df_filtered['Review']



stop_words = stopwords.words('english')
# stop_words.append("it'd")
stop_words.sort()




def replace_quotes(text):
    return text.replace("’", "'")

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def remove_username_punct_numbers(text):
    
    # Replace @USERNAME to '<user>'.
    text = re.sub('@[^\s]+','<user>', text)
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text



df_filtered['Review']=df_filtered['Review'].apply(replace_quotes)
df_filtered['Review']=df_filtered['Review'].apply(remove_stopwords)
df_filtered['Review']=df_filtered['Review'].apply(remove_username_punct_numbers)
df_filtered['Review']





import pandas as pd

import nltk

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(sentence):

    word_list = nltk.word_tokenize(sentence)

    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])

    return lemmatized_output


df_filtered['Review'] = df_filtered['Review'].apply(lemmatize_sentence)





df_filtered['Review'][0]




df_filtered['Review'] = df_filtered['Review'].apply(lambda x: " ".join ([w for w in x.split() if len(w)>3]))





df_filtered['Review'][0]





df_filtered.to_csv('filtered_hotels_preprocessed.csv')





df_filtered.head(2)






import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def get_sentiment_score(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    return score['compound']


new_df =df_filtered['Review']


new_df['sentiment_score'] = new_df['reviews'].apply(get_sentiment_score)





def classify_sentiment(score):
    if score>=0.25 and score<=1:
        return 'positive'
    elif score>=-1 and score<=-0.25:
        return 'negative'
    else:
        return 'neutral'
new_df['sentiment_class'] = new_df['sentiment_score'].apply(classify_sentiment)






from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(new_df['reviews'], new_df['sentiment_class'], test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


lr = LogisticRegression()
lr.fit(X_train_vec, y_train)


y_pred = lr.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)


print('Accuracy:', accuracy)
print('Confusion matrix:')
print(confusion_mat)



from sklearn.metrics import precision_recall_fscore_support


LR_precision, Lr_recall, Lr_f1, Lr_support = precision_recall_fscore_support(y_test, y_pred, average='weighted')


print('Precision:', LR_precision)
print('Recall:', Lr_recall)
print('F1 Score:', Lr_f1)





from sklearn.naive_bayes import MultinomialNB



X_train, X_test, y_train, y_test = train_test_split(new_df['reviews'], new_df['sentiment_class'], test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)


y_pred = nb.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)


print('Accuracy:', accuracy)
print('Confusion matrix:')
print(confusion_mat)


from sklearn.metrics import precision_recall_fscore_support

NB_precision, NB_recall, NB_f1, NB_support = precision_recall_fscore_support(y_test, y_pred, average='weighted')


print('Precision:', NB_precision)
print('Recall:', NB_recall)
print('F1 Score:', NB_f1)





import spacy
import time

start = time.time()
nlp = spacy.load('en_core_web_sm')

#df_filtered.Review = df_filtered.Review.str.lower()
aspect_terms = []

for review in nlp.pipe(df_filtered.Review):
    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    aspect_terms.append(chunks)

df_filtered['aspect_terms1'] = aspect_terms

print(start-time.time())



df_filtered.head(2)





import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import ast
import time





module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
model = hub.load(module_url)



def get_aspect_confidence(aspect_terms, review_text):

    total_aspect_confidence = []

    # Combine aspect and review text

    text = aspect_terms + ' ' + review_text

    # Embed text using USE

    embeddings = model([text])

    # Get embedding for aspect terms

    aspect_embedding = embeddings[0][0:len(aspect_terms.split())]

    # Reshape aspect_embedding to match the shape of review_embedding

    aspect_embedding = tf.reshape(aspect_embedding, [1, -1])

    aspect_embedding = np.transpose(aspect_embedding)

    # Get embedding for review text

    review_embedding = embeddings[0][len(aspect_terms.split()):]

    # Calculate cosine similarity between aspect embedding and review embedding

    similarity_scores = tf.reduce_sum(tf.multiply(aspect_embedding, review_embedding), axis=1)

    confidence_scores = tf.nn.softmax(similarity_scores)

    for i, aspect in enumerate(aspect_terms.split()):

        confidence_score = confidence_scores[i].numpy()

        total_aspect_confidence.append(confidence_score)

    if len(aspect_terms) != 0:

        return sum(total_aspect_confidence)/(len(aspect_terms))

    else: return sum(total_aspect_confidence)/(len(aspect_terms) + 99999)

 



start = time.time()

df_filtered['confidence_score'] = df_filtered.apply(lambda x: get_aspect_confidence(' '.join((x['aspect_terms1'])), x['Review']), axis=1)

print(time.time() - start)





df_filtered.head(2)





df_filtered['confidence_score'].max()





df_filtered['confidence_score'].min()



df_filtered['confidence_score'].value_counts()




import itertools
all_aspect_words = list(itertools.chain.from_iterable(df_filtered['aspect_terms1']))




len(all_aspect_words)




unique_aspect_words = list(set(all_aspect_words))
unique_aspect_words




len(unique_aspect_words)




import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Tokenize the words and convert them to IDs
tokens = tokenizer(unique_aspect_words, padding=True, truncation=True, return_tensors='pt')

# Pass the tokens through the BERT model to get the embeddings
with torch.no_grad():
    embeddings = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])[0][:, 0, :].numpy()




embeddings = pd.DataFrame(embeddings)



embeddings.to_csv('embeddings.csv')




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42).fit(embeddings)




clusters_kmeans = {}
for i, word in enumerate(unique_aspect_words):
    cluster_label = kmeans.labels_[i]
    if cluster_label not in clusters_kmeans:
        clusters_kmeans[cluster_label] = [] 
    clusters_kmeans[cluster_label].append(word)
 # Print the clusters




clusters_kmeans




from sklearn.metrics import silhouette_score
labels = kmeans.labels_
score = silhouette_score(embeddings,labels)
score




print(len(embeddings))




from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5,random_state=42).fit(embeddings)




labels = gmm.predict(embeddings)

clusters_gmm = {}
for i, word in enumerate(unique_aspect_words):
    cluster_label = labels[i]
    if cluster_label not in clusters_gmm:
        clusters_gmm[cluster_label] = [] 
    clusters_gmm[cluster_label].append(word)
 # Print the clusters




score = silhouette_score(embeddings,labels)
score




clusters_gmm



count_dict = Counter(all_aspect_words)
count_dict = dict(count_dict)
count_dict




def word_cloud(clusters):
    for i in range(0,len(clusters)):
        freq_dict = {word: count_dict.get(word, 0) for word in clusters[i]}
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(freq_dict)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()




word_cloud(clusters_kmeans)




word_cloud(clusters_gmm)




df_filtered = df_filtered[df_filtered['aspect_terms1'].apply(len)>0]




# Create a lookup dictionary for each aspect to its cluster
aspect_to_cluster = {}
for cluster, aspects in clusters_kmeans.items():
    for aspect in aspects:
        aspect_to_cluster[aspect] = cluster
#print(aspect_to_cluster)

# Define a function to assign the cluster to a review row
def assign_cluster(row):
    aspect_counts = Counter(row['aspect_terms1'])
    #print(aspect_counts)
    cluster_counts = Counter(aspect_to_cluster[aspect] for aspect in aspect_counts)
    #print(cluster_counts)
    assigned_cluster = cluster_counts.most_common(1)[0][0]
    return assigned_cluster

# Assign the cluster to a new column using the apply method
df_filtered['assigned_cluster'] = df_filtered.apply(assign_cluster, axis=1)


df_final = df_filtered.groupby(['assigned_cluster','Hotel_Name']).agg({'confidence_score':'mean'}).sort_values(['assigned_cluster','confidence_score'],ascending=[True,False])


df_final.to_csv('df_final.csv')




df_final












