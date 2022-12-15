# HateOMeter

## Introduction
Society has benefited greatly out from increased usage of social media and knowledge sharing. However, this has resulted in a number of issues, including the proliferation and dissemination of hate speech messages. Though much research has been carried out on recognizing hate speech in conversations and posts made on online platforms. But still it lacks in the thorough analysis in detecting hate speech and making proper datasets for analyzing and collecting proper information for existing machine learning models and deep learning models to properly recognize hate speech. To tackle this I have made a web application which can predict if the text provided is hate speech or not.

## Proposed model
![proposed-model.jpg](/assets/proposed-model.jpg)

## LSTM model
An artificial neural network called Long Short-Term Memory (LSTM) is employed in deep learning and artificial intelligence. LSTM features feedback connections as opposed to typical feedforward neural networks. Such a recurrent neural network (RNN) may analyse whole data sequences in addition to single data points (such as photos) (such as speech or video). For instance, LSTM can be used for applications like speech recognition, robot control, machine translation, networked, unsegmented handwriting recognition, video games, and healthcare.

## Model performance
I have used a LSTM algorithm to train the model. It is trained on the [twitter sentiment analysis](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) and [hate speech and offensive language](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) datasets available on kaggle by merging the two to improve performance. The model had an  ```f1 score of 0.93``` with ```support of 6736```

## Website Preview
![website-preview](/assets/website-preview.jpg)