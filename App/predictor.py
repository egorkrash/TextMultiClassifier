from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

MAX_SEQ_LEN = 255
# load all
LE = pickle.load(open('label_encoder.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

Model = load_model('model1.h5')
Model.load_weights('goodmodel1_weights.h5')


def get_true_class(pred):
    index_of_max = pred.argmax()
    return index_of_max


def get_true_classes(pred):
    for row in pred:
        index_of_max = row.argmax()

        print LE.inverse_transform(index_of_max)


def predict_speaker(texts, more_than_one=False):
    if not more_than_one:
        texts = [texts]

    print 'predicting...'
    sequences = tokenizer.texts_to_sequences(texts)
    vector = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
    probabilities = Model.predict_proba(vector, verbose=0)

    print 'almost done'
    if not more_than_one:
        cls = get_true_class(probabilities)
        return LE.inverse_transform(cls)

    else:
        cls = get_true_classes(probabilities)
