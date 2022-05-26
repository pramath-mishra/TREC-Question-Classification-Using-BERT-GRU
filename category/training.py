import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import transformers
import tensorflow as tf
from data_gen import BertSemanticDataGenerator

# config
epochs = 2
batch_size = 32
max_length = 128

# loading train data
train = pd.read_csv(
    "../preprocessing/train.csv",
    low_memory=False,
    usecols=[
        "Text",
        "Category"
    ]
)
train.dropna(axis=0, inplace=True)
train = train.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"train data loaded...\n -shape: {train.shape}")

# loading test data
test = pd.read_csv(
    "../preprocessing/test.csv",
    low_memory=False,
    usecols=[
        "Text",
        "Category"
    ]
)
test.dropna(axis=0, inplace=True)
test = test.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"test data loaded...\n -shape: {test.shape}")


# sample data
print(f"text : {train.loc[1, 'Text']}")
print(f"category : {train.loc[1, 'Category']}")

# one hot encoding labels
labels = list(set(train.Category.tolist()))
label2id = {value: i for i, value in enumerate(labels)}
id2label = {i: value for i, value in enumerate(labels)}
pickle.dump(labels, open("./labels.sav", "wb"))

train["label"] = train["Category"].apply(lambda x: label2id[x])
y_train = tf.keras.utils.to_categorical(train.label, num_classes=len(labels))

test["label"] = test["Category"].apply(lambda x: label2id[x])
y_test = tf.keras.utils.to_categorical(test.label, num_classes=len(labels))

# model under distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_masks")
    token_type_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids")
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # fine-tune BERT model
    bert_model.trainable = True
    bert_output = bert_model.bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

    # extracting embedding & passing to Bi-directional GRU
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(sequence_output)

    # hybrid pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_gru)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_gru)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(len(labels), activation="softmax")(dropout)
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])


print(f"Strategy: {strategy}")
print(model.summary())

# train and validation data generation
train_data = BertSemanticDataGenerator(
    train["Text"].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True
)
test_data = BertSemanticDataGenerator(
    test["Text"].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
print("train & test data generated...")

# model training
model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)

# model saving
model.save("./category_model", save_format='tf')
print("model saved...")

# model evaluation
result = model.evaluate(test_data, verbose=1)
print(f"Accuracy: {result[1]}, Loss: {result[0]}")
