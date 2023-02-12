import pathlib
import pickle
import csv
import numpy as np
from pyknp import Juman
from absl import app, flags


def modify(word):
    if len(word) > 7 and word[:7] == 'SSSSUNK' :
        modified = ['SSSS', word[7:]]
    elif len(word) > 4 and word[:4] == 'SSSS' :
        modified = ['SSSS', word[4:]]
    elif word == 'UNKUNK' :
        modified = ['UNK']
    elif len(word) > 3 and word[:3] == 'UNK' :
        modified = ['UNK', word[3:]]
    else :
        modified = [word]
    return modified    


def decompose_corpus(jumanpp, filepath):
    filepath = pathlib.Path(filepath)
    with filepath.open('r') as f:
        data = csv.reader(f)
        data = list(data)

    parts = []
    for i, line in enumerate(data):
        if len(line[0].encode('utf-8')) > 4096:
            continue

        result = jumanpp.analysis(line[0])
        for mrph in result.mrph_list():
            parts += modify(mrph.midasi)

        if i % 5000 == 0:
            print(i)

    return parts




FLAGS = flags.FLAGS

flags.DEFINE_multi_string("corpus_filepath", None, help="relative path to corpus file from data directory", short_name="f")


def main(args):
    jumanpp = Juman()
    data_dir_path = pathlib.Path(__file__).resolve().parents[1] / 'data'
    filepaths = [data_dir_path / f for f in FLAGS.corpus_filepath]
    
    parts_list = []
    for fp in filepaths:
        parts_list += decompose_corpus(jumanpp, fp)

    (data_dir_path / 'parts_list.pickle').write_bytes(pickle.dumps(parts_list))




if __name__ == '__main__':
    app.run(main)

