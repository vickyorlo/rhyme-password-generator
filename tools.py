from datamuse import datamuse
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import to_categorical
import numpy
import argparse
import time

def downloadData():
    api = datamuse.Datamuse()
    try:
        words = open("words_alpha.txt", encoding='utf-8')
        rhymes = open("rhymes.txt", 'a', encoding='utf-8')
        progress = open("progress.txt", 'r', encoding='utf-8')

        line_no = int(progress.readline())
        progress.close()
        line_begin = line_no
        print(line_no)
        for i, line in enumerate(words):
            if (i == line_no):
                if (line_no < line_begin + 75000):
                    line = line.rstrip()
                    try:
                        api_rhymes = api.words(rel_rhy=line, max=5)
                        [rhymes.write(line+';'+rh["word"]+'\n')
                            for rh in api_rhymes]
                        print(line)
                    except:
                        print("json decode on " + line)
                    
                    if len(api_rhymes) < 5:
                        try:
                            near_rhymes = api.words(rel_nry=line, max=5)
                            [rhymes.write(line+';'+rh["word"]+'\n')
                                for rh in near_rhymes]
                            print(line)
                        except:
                            print("json decode on " + line)
                    rhymes.flush()
                    line_no = line_no+1
                else:
                    break
    finally:
        progress = open("progress.txt", 'w', encoding='utf-8')
        progress.write(str(line_no))
        words.close()
        rhymes.close()
        progress.close()

def split_exact(size):
    with open("rhymes.txt",'r') as rhymes, open(f"x_batch_{size}_exact.txt",'w') as x_batch, open(f"y_batch_{size}_exact.txt",'w') as y_batch:
        for line in rhymes:
            x, y = map(lambda x: x.strip(), line.split(';'))
            if len(x) != size or len(y) != size:
                continue
            x_batch.write(x.rjust(size,'`')+'\n')
            y_batch.write(y.rjust(size,'`')+'\n')

def split(size):
    with open("rhymes.txt",'r') as rhymes, open(f"x_batch_{size}.txt",'w') as x_batch, open(f"y_batch_{size}.txt",'w') as y_batch:
        for line in rhymes:
            x, y = map(lambda x: x.strip(), line.split(';'))
            if len(x) > size or len(y) > size:
                continue
            x_batch.write(x.rjust(size,'`')+'\n')
            y_batch.write(y.rjust(size,'`')+'\n')

def predict_from_file_to_file(path_to_model):
    with open("to_predict.txt",'r') as to_predict, open("predicted.txt",'w') as predicted:
        predicted.write('\n'.join(predict(path_to_model,to_predict)))
        
def saveGraphKeras(modelName):
    model = load_model(modelName)
    plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=False)

def predict_from_list(path_to_model, to_predict):
    return predict(path_to_model,to_predict)

def predict(path_to_model, to_predict):
    model = load_model(path_to_model)

    to_predict = [line.rstrip().lower().rjust(16,'`') for line in to_predict]
    for line in to_predict:
        if len(line)>16:
            raise Exception('input cannot be longer than 16 characters!')
    to_predict = [[ [ord(ch)-96] for ch in line] for line in to_predict]
    to_predict = to_categorical(numpy.array(to_predict),27)
    result = model.predict(to_predict)

    predicted = []
    for line in result:
        word =""
        for c in line:
            if numpy.argmax(c) == 0:
                c = 32
            else:
                c = numpy.argmax(c)+96
            word += chr(c)
        predicted.append(word.strip())
    return predicted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('-m', help='predict with the given model')
    parser.add_argument('-g', help='graph the given model')
    parser.add_argument('-d', help='download data')
    parser.add_argument("--split", type=int, help="split rhymes")
    parser.add_argument("--split_exact", type=int, help="split rhymes exact")
    parser.add_argument("--download", help="download data")

    args = parser.parse_args()

    if args.m:
        predict_from_file_to_file(args.m)

    if args.g:
        saveGraphKeras(args.g)

    if args.split:
        split(args.split)

    if args.split_exact:
        split(args.split_exact)

    if args.download:
        downloadData()