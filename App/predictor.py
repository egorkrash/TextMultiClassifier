from conf import MAX_SEQ_LEN,tokenizer,LE
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

Model = load_model('model1.h5')
Model.load_weights('goodmodel1_weights.h5')

def get_true_class(pred):
    index_of_max = pred.argmax()
    return index_of_max

def get_true_classes(pred):

    true_classes = []

    for row in pred:

        index_of_max = row.argmax()

        print LE.inverse_transform(index_of_max)

def predict_speaker(texts,more_than_one = False):
    if not more_than_one:
        texts = [texts]
    
    print 'predicting...'
    sequences = tokenizer.texts_to_sequences(texts)
    vector = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
    probs = Model.predict_proba(vector,verbose=0)
    
    print 'almost done'
    if not more_than_one:
        cls = get_true_class(probs)
        return LE.inverse_transform(cls)

    else:
        cls = get_true_classes(probs)
        
    

    
    