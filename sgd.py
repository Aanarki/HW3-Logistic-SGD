from math import exp
import random


def logistic(x):
    sigmax = 1/(1+exp(-x))  # Logistic Equation
    return sigmax


def dot(x, y):
    s = 0
    limit = len(x)
    for i in range(limit):
        s += x[i]*y[i]
    return s


def predict(model, point):
    dot_product = dot(model, point['features'])
    prediction = logistic(dot_product)
    return prediction


def accuracy(data, predictions):
    correct = 0
    length = len(data)
    # Initialize model
    model = initialize_model(len(data[0]['features']))

    for i in range(length):
        prediction = predict(model, data[i])

        # Prediction is true if >= to .5 else false
        if prediction >= .5:
            predictions[i] = True
        else:
            predictions[i] = False

        guess = predictions[i]  # Algorithm Classification
        real = data[i]['label']  # Actual Classification
        # If guess is same as real, correct classification
        if guess == real:
            correct += 1

    return float(correct)/len(data)


# TODO: Update model using learning rate and L2 regularization
def update(model, point, delta, rate, lam):
    # rate -> learning rate | lam -> regularization parameter |
    pass


def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]


# TODO: Train model using training data
def train(data, epochs, rate, lam):
    # epoch just means examining N data points where N is the number of points in your training data

    # The model contains the weights of each measured feature
    model = initialize_model(len(data[0]['features']))

    # Train model using epochs number of random examples
    for i in range(epochs):
        # TODO Calculate error for each point and adjust model weights base on gradient error

        # Loss = (Prediction - Truth)^2 + lam * ||model||
        # (...)^2 -> squared error | lam*len(model) -> loss from regularization
        loss = (predict(model, data[i]) - data[i]['label'])^2 + lam*len(model)

    return model


def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        # TODO: Add more feature extraction rules here!
        point['features'] = features
        data.append(point)
    return data


# TODO: Tune your parameters for final submission
def submission(data):
    return train(data, 1, .01, 0)
