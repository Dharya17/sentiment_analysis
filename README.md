
# Movie Review Sentiment Analysis

## Overview
This project performs sentiment analysis on IMDB movie reviews using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The model is built with a Bidirectional LSTM and Word2Vec embeddings to classify reviews as positive or negative.

## Dataset
- **Source**: IMDB movie reviews dataset
- **Size**: 50,000 reviews (balanced between positive and negative)

## Model
- **Architecture**: Bidirectional LSTM
- **Embeddings**: Word2Vec
- **Accuracy**: Improved from 87% to 89%

## Dependencies
- Python
- Keras & TensorFlow
- Gensim
- NLTK
- Pandas
- NumPy

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/movie-review-sentiment-analysis.git
   ```
   This command will create a local copy of the repository on your machine.

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   This command will install all the necessary dependencies listed in the `requirements.txt` file.

## Usage
1. **Preprocess the data and train the Word2Vec model**:
   ```python
   from gensim.models import Word2Vec
   from nltk.tokenize import word_tokenize
   import nltk
   nltk.download('punkt')

   # Tokenize the reviews
   reviews = [word_tokenize(review) for review in train_data['review']]

   # Train Word2Vec model
   word2vec_model = Word2Vec(sentences=reviews, vector_size=100, window=5, min_count=1, workers=4)

   # Create embedding matrix
   embedding_matrix = np.zeros((5000, 100))
   for word, i in tokenizer.word_index.items():
       if i < 5000:
           try:
               embedding_vector = word2vec_model.wv[word]
               embedding_matrix[i] = embedding_vector
           except KeyError:
               pass
   ```
   This step involves tokenizing the text data, training a Word2Vec model on the tokenized reviews, and creating an embedding matrix for the model.

2. **Train the Bidirectional LSTM model**:
   ```python
   from keras.layers import Bidirectional
   from keras.initializers import Constant

   model = Sequential()
   model.add(Embedding(input_dim=5000, output_dim=100, embeddings_initializer=Constant(embedding_matrix), input_length=200, trainable=False))
   model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
   model.add(Dense(1, activation='sigmoid'))

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   model.fit(x_train, y_train, epochs=5, validation_split=0.3)
   ```
   This step defines and trains the Bidirectional LSTM model using the pre-trained Word2Vec embeddings.

3. **Evaluate the model on the test set**:
   ```python
   loss, accuracy = model.evaluate(x_test, y_test)
   print(f"Test loss: {loss}")
   print(f"Test accuracy: {accuracy}")
   ```
   This step evaluates the trained model on the test dataset to determine its performance.

## Key Learnings
- **Importance of data preprocessing**: Properly cleaning and preparing the data significantly impacts model performance.
- **Benefits of using pre-trained embeddings**: Leveraging pre-trained embeddings like Word2Vec can improve model accuracy.
- **Impact of model architecture**: The choice of model architecture, such as using Bidirectional LSTM, plays a crucial role in achieving better results.

## Future Work
- **Implement attention mechanisms**: Adding attention layers to help the model focus on important parts of the text.
- **Explore transformer-based models**: Experimenting with advanced models like BERT or GPT for potentially better performance.
- **Experiment with data augmentation techniques**: Using techniques like synonym replacement and back translation to create more training data.
