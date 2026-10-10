# Financial Transaction Fraud Detection Using Deep Learning

A deep learning project that explores the detection of potentially fraudulent financial transactions using a Multi-Layer Perceptron (MLP) neural network built with TensorFlow and Keras.

The project uses synthetic financial transaction data from **IBM AMLSim** to investigate how neural networks handle highly imbalanced classification problems.

## Dataset

- **Source:** IBM AMLSim synthetic financial transaction dataset

https://github-wiki-see.page/m/IBM/AMLSim/wiki/Download-Example-Data-Set#10k-1m-10k-accounts-1m-transactions

- **Data used:** First 100,000 transactions
- **Target variable:** `IS_FRAUD`
- **Task:** Binary classification
- **Features:** Sender account ID, receiver account ID, transaction amount, and timestamp

The dataset is synthetic and is intended for experimentation and research rather than direct evaluation of real-world fraud detection.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow and Keras

## Model Architecture

The MLP consists of:
- An input layer accepting four features
- Two hidden dense layers using ReLU activation
- An output layer using sigmoid activation to produce a fraud probability

The output is converted into a binary prediction using a classification threshold.


## Disclaimer

This project is an educational exploration of deep learning for synthetic financial transaction fraud detection. It is not intended for production use or as a reliable real-world fraud detection system.

## Author

**Mohammad Hafees**
