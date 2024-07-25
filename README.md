### Running the code
To run the code first install the required libraries and then run `main.py`. Note that the first time the model is run it will automatically download a part of speech tagger.
```
pip install -r requirements.txt
python3 main.py
```

### Running tests
Two tests are included to check the preprocessing code works as intended. To run them:
```
python3 test.py
```

### Output
The raw output from running `main.py` can be found in `output.txt`. This contains some summary statistics of the dataset, performance metrics for each fold of cross-validation, the same metrics averaged over all the folds and randomly selected errors for manual evaluation - 10 false positives and 10 false negatives.

### Performance
- In order to evaluate the performance reliably I have used K-fold cross-validation with n=5. Averaging across the folds the model performed as follows on the test sets:
  - Accuracy: 0.974
  - Precision: 0.976
  - Recall: 0.954
  - **F1-score: 0.965**
- From the qualitative analysis (in `output.txt`) you can see that it makes a couple of errors that should be relatively easy to correct. False positives no. 8 mentions 'allah' and no. 10 mentions 'anarchists'. This is probably due to the model not having seen these words either enough or at all during training. This can be fixed by using a pre-trained deep learning model. See the final point in 'Potential improvements' for more on this.

### What went well
- The character, word and part of speech (POS) vectorizations of the dataset complement each other well, providing a good set of features for XGBoost. During development I started with just words, then added characters, followed last by the POS tags. At each stage I verified that the additional features improved the performance of the model.
  - The character features help to model when a particular author is more likely to use certain types of punctuation such as exclamation marks or semicolons.
  - The word level captures a variety of important information. Names such as 'Elizabeth' will be strong indicators of Jane Austen, as well as subject-matter-related words such as "invitation". Other words and phrases like "mortified" contribute to an understanding of Austen's style of writing.
  - The part of speech information allows the model to make generalisations about the sentence structure of Jane Austen versus other authors without caring about specific words.
- TFIDF is useful for selecting a subset of the features that is most informative - a necessary task since simply selecting all the features would lead to massive overfitting. This is especially important since not only does using all three (characters, words and POS tags) mean triple the potential features, but bigrams are being used as well as unigrams, adding even more.
- XGBoost is fast and reliable, making it a good choice for establishing a high-quality baseline. The speed allows for fast iterations and the hyperparameters are reasonably intuitive to tune. For example, I found that reducing the maximum depth from 6 to 4 is helpful to reduce overfitting.

### Potential improvements
- Lemmatization could be used to provide another level of features in-between words and POS tags that could improve generalisation.
- The model is currently allowed to use names in the training, which makes a number of examples trivially easy. It might perform better on unseen examples if these were removed from the training set, since the model would be forced to learn how to use other features.
- The current approach suffers from overfitting, getting perfect results on the training set even with the depth of the trees limited to 4. One way to improve generalisation would be to use a deep learning model that has been pre-trained on a general task. The standard way to do this would be to download a pre-trained model such as BERT and add one or two layers including a final softmax for the classification task. Then two variations can be tried: (1) keep the original weights fixed and train only the softamx layer, (2) train all of the weights, including fine-tuning the originals. There's a decent quantity of data so my expectation would be that (2) would perform best.

### Putting the model into production
- To put the model into production:
  - Save the trained model to disk and load it into memory on startup. Generally speaking, you will want to ensure the production environment has sufficient memory to do this, but that is far more relevant for potential improvements using large language models than XGBoost, which has far fewer parameters.
  - Caching could be used to improve latency if it is likely that the same input might be received more than once. However, the model is not slow to run so the gains might not be large.
- To handle potential errors and ensure performance in production does not deviate from performance during development the following steps could be taken:
  - Ensure the same preprocessing steps are done for queries issued in production. This can be handled via testing.
  - Put the predict function inside a try-except statement so unexpected/invalid inputs do not crash the server. If an error does occur, log the input that caused it and return an appropriate error message.
  - Consider whether it may be necessary to periodically retrain on an updated dataset, in the event that the distribution of inputs shifts over time.