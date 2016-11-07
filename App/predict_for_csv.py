from predictor import predict_speaker
import pandas as pd
import sys 

path = sys.argv[1]
col_to_predict = sys.argv[2]

data = pd.read_csv(path)

text = data[col_to_predict].values
print 'it can take a long time...'
predict_speaker(text, True)