import json
import re

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from scipy.sparse import hstack

import xgboost as xgb

#from tensorflow.keras import layers
#import tensorflow as tf
#import keras
#import keras_nlp

# Read in the data
f = open("data/gutenberg-paragraphs.json")
data = json.load(f)
text = [i["text"] for i in data]
labels = np.array([i["austen"] for i in data])

# Preliminary analysis
print("Number of rows:", labels.size)
print("Positive-negative ratio: {r:.3f}".format(r = np.mean(labels)))

# Preprocessing

# make 'less' equal to '\nless'
text = [i.replace("\n", " ") for i in text]
text = [i.replace("\r", " ") for i in text]

# If this punctuation is removed as is, words will be incorrectly joined together
text = [i.replace("--", " -- ") for i in text]

text = [re.sub(' +', ' ', i) for i in text] # multiple spaces -> single space

text = [i.lower() for i in text] # lowercase

# Create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=.2)

#### Baseline model (XGBoost on a bag of words and characters) ####

# Tokenize
# Words
count_vect = TfidfVectorizer(ngram_range=(1,2), min_df=30, max_features=10000)
count_vect.fit(X_train)
X_train_word = count_vect.transform(X_train)
X_test_word = count_vect.transform(X_test)

# Characters - authors can be identified by their use of particular punctuatuion
char_vect = TfidfVectorizer(ngram_range=(1,2), min_df=30, max_features=10000, analyzer='char', preprocessor=None)
char_vect.fit(X_train)
X_train_char = char_vect.transform(X_train)
X_test_char = char_vect.transform(X_test)

X_train_counts = hstack([X_train_word, X_train_char])
X_test_counts = hstack([X_test_word, X_test_char])

# tfidf_vect = TfidfVectorizer(min_df=10)
# tfidf_vect.fit(X_train)
# X_train_counts = tfidf_vect.transform(X_train)
# X_test_counts = tfidf_vect.transform(X_test)

# Create and fit the model
model = xgb.XGBClassifier(n_estimators=128, max_depth=4, learning_rate=1, objective='binary:logistic')
model.fit(X_train_counts, y_train)

# Evaluate the baseline
print("\nBaseline: XGBoost on a bag of words and characters")

print("\nTrain set")
pred = model.predict(X_train_counts)
print("Accuracy: {a:.3f}".format(a = accuracy_score(y_train, pred)))
print("Precision: {p:.3f}".format(p = precision_score(y_train, pred)))
print("Recall: {r:.3f}".format(r = recall_score(y_train, pred)))
print("F1: {f:.3f}".format(f = f1_score(y_train, pred)))

print("\nTest set")
pred = model.predict(X_test_counts)
print("Accuracy: {a:.3f}".format(a = accuracy_score(y_test, pred)))
print("Precision: {p:.3f}".format(p = precision_score(y_test, pred)))
print("Recall: {r:.3f}".format(r = recall_score(y_test, pred)))
print("F1: {f:.3f}".format(f = f1_score(y_test, pred)))

#### Deep learning model ####

# import transformers

# strategy = tf.distribute.MirroredStrategy()

# max_length = 512
# n_classes = 2

# # Define the model
# with strategy.scope():
#     # Encoded token ids from BERT tokenizer.
#     input_ids = tf.keras.layers.Input(
#         shape=(max_length,), dtype=tf.int32, name="input_ids"
#     )
#     # Attention masks indicates to the model which tokens should be attended to.
#     attention_masks = tf.keras.layers.Input(
#         shape=(max_length,), dtype=tf.int32, name="attention_masks"
#     )
#     # Token type ids are binary masks identifying different sequences in the model.
#     token_type_ids = tf.keras.layers.Input(
#         shape=(max_length,), dtype=tf.int32, name="token_type_ids"
#     )
#     # Loading pretrained BERT model
#     # Freeze the BERT model to reuse the pretrained features without modifying them.
#     bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
#     bert_model.trainable = False

#     bert_output = bert_model.bert(
#         input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
#     )
#     sequence_output = bert_output.last_hidden_state

#     # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
#     bi_lstm = tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(64, return_sequences=True)
#     )(sequence_output)

#     # Applying hybrid pooling approach to bi_lstm sequence output.
#     avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
#     max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
#     concat = tf.keras.layers.concatenate([avg_pool, max_pool])
#     dropout = tf.keras.layers.Dropout(0.3)(concat)

#     output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

#     model = tf.keras.models.Model(
#         inputs=[input_ids, attention_masks, token_type_ids], outputs=output
#     )

#     # Most classes appear rarely so it is trivial to get very high accuracies
#     # Therefore we use precision and recall to properly show the predictive quality of the model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(),
#         #loss="binary_crossentropy",
#         loss="sparse_categorical_crossentropy",
#         metrics=["binary_accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
#     )

# batch_size = 128

# class DataGenerator(tf.keras.utils.Sequence):
#     """Generates batches of data.

#     Args:
#         X: Array of input sentences.
#         y: Array of labels.
#         batch_size: Integer batch size.
#         shuffle: boolean, whether to shuffle the data.
#         include_labels: boolean, whether to include the labels.

#     Returns:
#         Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
#         (or just `[input_ids, attention_mask, `token_type_ids]`
#          if `include_labels=False`)
#     """

#     def __init__(
#         self,
#         X,
#         y=None,
#         batch_size=batch_size,
#         shuffle=True,
#         include_labels=True,
#     ):
#         self.X = X
#         self.y = y
#         self.shuffle = shuffle
#         self.batch_size = batch_size
#         self.include_labels = include_labels

#         # Load our BERT Tokenizer to encode the text.
#         # We will use base-base-uncased pretrained model.
#         self.tokenizer = transformers.BertTokenizer.from_pretrained(
#             "bert-base-uncased", do_lower_case=True
#         )
#         self.indices = np.arange(len(self.X))
#         self.on_epoch_end()

#     def __len__(self):
#         # Denotes the number of batches per epoch.
#         return len(self.X) // self.batch_size

#     def __getitem__(self, idx):
#         # Get the indices for the batch
#         indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
#         X = [self.X[i] for i in indices]

#         # With BERT tokenizer's batch_encode_plus batch of both the sentences are
#         # encoded together and separated by [SEP] token.
#         encoded = self.tokenizer.batch_encode_plus(
#             X,
#             add_special_tokens=True,
#             max_length=max_length,
#             truncation=True,
#             return_attention_mask=True,
#             return_token_type_ids=True,
#             padding="max_length",
#             return_tensors="tf",
#         )

#         # Convert batch of encoded features to numpy array.
#         input_ids = np.array(encoded["input_ids"], dtype="int32")
#         attention_masks = np.array(encoded["attention_mask"], dtype="int32")
#         token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

#         # Set to true if data generator is used for training/validation.
#         if self.include_labels:
#             y = np.array(self.y[indices], dtype="int32")
#             #y = np.hstack([1 - y, y])
#             return [input_ids, attention_masks, token_type_ids], y
#         else:
#             return [input_ids, attention_masks, token_type_ids]

#     def on_epoch_end(self):
#         # Shuffle indexes after each epoch if shuffle is set to True.
#         if self.shuffle:
#             np.random.RandomState(42).shuffle(self.indices)

# train_data = DataGenerator(
#     X_train,
#     y_train,
#     batch_size=batch_size,
#     shuffle=True,
# )

# # valid_data = DataGenerator(
# #     X_valid,
# #     y_valid,
# #     batch_size=batch_size,
# #     shuffle=False,
# # )

# test_data = DataGenerator(
#     X_test,
#     y_test,
#     batch_size=batch_size,
#     shuffle=False,
# )

# history = model.fit(
#     train_data,
#     validation_data=test_data,
#     epochs=16,
#     use_multiprocessing=True,
#     workers=-1,
# )

# There are 9 examples with > 1000 words, including 2 of length > 100k
# Cut them all off at 1000 and pad the rest
# MAX_SEQUENCE_LENGTH = 1000
# VOCAB_SIZE = 15000
# BATCH_SIZE = 32

# # def train_word_piece(ds, vocab_size, reserved_tokens):
# #     word_piece_ds = ds.unbatch().map(lambda x, y: x)
# #     vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
# #         word_piece_ds.batch(1000).prefetch(2),
# #         vocabulary_size=vocab_size,
# #         reserved_tokens=reserved_tokens,
# #     )
# #     return vocab

# train_ds = tf.data.Dataset.from_tensor_slices(X_train)
# train_ds = train_ds.batch(BATCH_SIZE)

# test_ds = tf.data.Dataset.from_tensor_slices(X_test)

# preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
#     "bert_tiny_en_uncased",
#     sequence_length=MAX_SEQUENCE_LENGTH,
# )

# train_cached = (
#     train_ds.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
# )
# test_cached = (
#     test_ds.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
# )

# classifier = keras_nlp.models.BertClassifier.from_preset(
#     "bert_tiny_en_uncased",
#     num_classes=2,
#     preprocessor=None
# )

# # reserved_tokens = ["[PAD]", "[UNK]"]
# # train_sentences = [element[0] for element in train_ds]
# # vocab = train_word_piece(train_ds, VOCAB_SIZE, reserved_tokens)

# # tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
# #     vocabulary=vocab,
# #     lowercase=False,
# #     sequence_length=MAX_SEQUENCE_LENGTH,
# # )

# classifier.fit(
#     train_cached,
#     validation_data=test_cached,
#     epochs=1,
# )
