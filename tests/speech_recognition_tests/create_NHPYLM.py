import pickle
import tempfile

import nhpylm
from tqdm import tqdm

from nt.io.data_dir import testing as data_dir
from nt.speech_recognition import fst
from nt.transcription_handling import transcription_handler


def text_to_splitted_words(text):
    splitted = list()
    for l in text:
        splitted.append([list(w) for w in l.split()])
    return splitted


def find_symbols(text):
    symbols = set()
    for line in text:
        for word in line:
            for symbol in word:
                symbols.add(symbol)
    return symbols


train_data = ['AA AB BA', 'AC AA BC']
train_data_splitted = text_to_splitted_words(train_data)

symbols = sorted(find_symbols(train_data_splitted))

lm = nhpylm.NHPYLM(list(symbols), 1, 2)

train_data_ids = lm.word_lists_to_id_lists(train_data_splitted)

for l in tqdm(train_data_ids):
    lm.add_id_sentence_to_lm(l)

lm.resample_hyperparameters()

G_fst = lm.to_fst_text_format()

with tempfile.NamedTemporaryFile() as g_txt:
    with open(g_txt.name, 'w') as fid:
        for line in G_fst:
            fid.write(line.decode('utf-8') + '\n')
    fst.build_from_txt(transducer_as_txt=g_txt.name,
                       output_file=data_dir('speech_recognition', 'NHPYLM',
                                            'G.fst'),
                       sort_type='ilabel', rmepsilon=False)

lexicon = {k: list(k) for k in lm.string_ids[
                               len(nhpylm.c_core.nhpylm.special_symbols) + len(
                                   symbols):] if k != 'EOS'}
th = transcription_handler.HPYLMTranscriptionHandler(lexicon, blank='<blank>',
                                                eps='<eps>', phi='<phi>',
                                                sow='<sow>', eow='</eow>',
                                                sos='<sos>', eos='</eos>')
with open(data_dir('speech_recognition', 'NHPYLM', 'label_handler.pkl'),
          'wb') as fid:
    pickle.dump(th, fid)
