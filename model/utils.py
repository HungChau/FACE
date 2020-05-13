import pandas as pd
import json
import re
import ast
from os import listdir, walk
from os.path import isfile, join
from sklearn.datasets import load_svmlight_file
from nltk import sent_tokenize


def report2dict(cr):
    """
    This function is to transform the free text result from classification_result to indexed format dataframe
    :param cr: the text result from classification_result()
    :return: dataframe format for the result
    """
    # Parse rows
    rows = cr.split("\n")
    df = pd.DataFrame(columns=['metric'] + rows[0].split())
    for r in rows[1:]:
        if r != '':
            print(r.split())
            df.loc[df.shape[0]] = r.split()[-5:]
    return df


def read_csv(input_file):
    """
    Deal with reading big csv files
    :param input_file: the path to the input file
    :return: <DataFrame>
    """
    with open(input_file, 'r') as f:
        a = f.readline()
    csv_list = a.split(',')
    tsv_list = a.split('\t')
    if len(csv_list) > len(tsv_list):
        sep = ','
    else:
        sep = '\t'

    reader = pd.read_csv(input_file, iterator=True, low_memory=False, delimiter=sep, encoding='ISO-8859-1')
    loop = True
    chunk_size = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    df = pd.concat(chunks, ignore_index=True)
    return df


def get_filenames(path, format_files, filtered_with=''):
    """
    Get file names in a folder
    :param path: folder path
    :param format_files: file extension
    :param filtered_with: filter with specific values
    :return: a list of file names with full path
    """
    return [join(path, f) for f in listdir(path) if
            isfile(join(path, f)) and f[-3:] == format_files and f.find(filtered_with) > -1]


def get_filenames_from_root(path, format_files, filtered_with=''):
    """
    Get all file names from a root path, including files in sub-directories
    :param path: root path
    :param format_files: file extension
    :param filtered_with: filter with specific values
    :return: a list of file names with full path
    """
    filenames = []
    for path, subdirs, files in walk(path):
        for name in files:
            if name[-3:] == format_files and name.find(filtered_with) > -1:
                filenames.append(join(path, name))

    return filenames


def get_regex_patterns(file_path):
    """

    :param file_path:
    :return: <string> - concatenation of all the patterns in the input file
    """
    file = open(file_path)
    patterns = ''
    for line in file:
        pa = line.replace('\n', '').split('\t')[0]
        if pa[0] != '#':
            patterns = patterns + line.replace('\n', '').split('\t')[0] + '|'

    return patterns[:-1]


def load_config(path):
    with open(path) as json_file:
        data = json.load(json_file)
        pii_entity = data['Feature']
        constant_variable = data['ConstantVariable']

        return constant_variable, pii_entity


def read_dataset_1(filename):
    """

    :param filename: the path to the input file which has the format of each row is a text document
    :return: list of sentences
    """
    with open(filename, "r") as f:
        for line in f:
            sents = sent_tokenize(line)
            yield [s for s in sents]


def read_dataset_2(filename):
    """

    :param filename: the path to the input file which has the format of each row is <doc_id><tab><document_text>
    :return: list of [doc_id, doc_text)
    """
    with open(filename, "r") as f:
        for line in f:
            yield line.split('\t')


def get_labels_1(filename):
    """
    Get labels from a file where each keyphrase is per row without document id
    :param filename:
    :return: a list of keyphrases
    """
    with open(filename, "r") as f:
        yield [line[:-1] for line in f]


def get_labels_2(filename):
    """
    Get labels from a file where each row contains [doc_id] and a list if keyphrases
    :param filename:
    :return: a dictionary of doc ids and keyphrases ('doc_id': 'list of keyphrases')
    """

    dic = dict()
    with open(filename, "r") as f:
        for line in f:
            k, v = line.split('\t')
            dic[k] = ast.literal_eval(v.lower())

    return dic


def get_data(file_name):
    """

    :param file_name: path to a svmlight-formatted file
    :return: <data, label>
    """
    data = load_svmlight_file(file_name)
    return data[0], data[1]


def update_pos_pattern(filename, docs, labels, nlp, N=4):
    """

    :param filename: path to the current pattern file
    :param docs: input texts
    :param labels: input labels
    :param nlp: spaCy nlp model
    :param N: maximum phrase length
    :return:
    """

    r = re.compile(get_regex_patterns(filename), flags=re.I)
    patterns = []
    pattern_exam = dict()
    for doc in docs:
        sents = sent_tokenize(doc[1])
        for sen in sents:
            s = nlp(sen)
            for n in range(N):
                for i in range(len(s) - n):
                    tags = [s[j].tag_ for j in range(i, i + n + 1)]
                    candidate = str(s[i:(i + n + 1)]).lower()
                    if r.match(' '.join(tags)) is None and candidate in labels[doc[0]]:
                        p = ' '.join(tags) + '$'
                        if p not in patterns:
                            pattern_exam[p] = candidate
                            patterns.append(p)

    with open(filename, 'a') as f:
        for pa in patterns:
            f.write(pa + '\t' + pattern_exam[pa])
            f.write('\n')


def plot_ROC(fpr, tpr, index, title):
    import matplotlib.pyplot as plt

    plt.figure(index)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.show()


def report2dict(cr):
    """
    This function is to transform the free text result from classification_result to indexed format dataframe
    :param cr: the text result from classification_result()
    :return: dataframe format for the result
    """
    # Parse rows
    rows = cr.split("\n")
    df = pd.DataFrame(columns=['label'] + rows[0].split())
    for r in rows[1:4]:
        if r != '':
            df.loc[df.shape[0]] = r.split()[-5:]
    return df
