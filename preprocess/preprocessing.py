import re
import pandas as pd
import collections
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def get_key_words(doc, n_gram=1):
    """
    :param doc: String input to find the keywords
    :param n_gram: Integer default is 1 otherwirse, set n-gram number
    :return: List of keywords
    """
    keywords=[]
    kw_model = KeyBERT()
    keyword_distance =(kw_model.extract_keywords(doc, keyphrase_ngram_range=(n_gram, n_gram), stop_words=None))

    for ele in keyword_distance:
        keywords.append(ele[0])

    return keywords


def get_encoding(all_strings, most_occur_words):
    """
    Encodes the data using K most frequent words:
    In a sentence if a word belongs to most frequent words list encode the word by 1 otherwise 0

    :param all_strings: Dataframe input text data
    :param most_occur_words: List of most frequent word
    :return an encoded dataframe
    """

    encode_output = []
    for st in all_strings[0]:
        st = st.lower()
        temp_vector = [0] * len(most_occur_words)
        for i in range(len(most_occur_words)):
            if st[i] in most_occur_words:
                temp_vector[i] = 1
        encode_output.append(temp_vector)
    encode_output = pd.DataFrame(encode_output)
    return encode_output

def n_gram_encoding(all_strings, ngram=1):
    """
    N-gram encoding

    :param all_strings: Dataframe input text data
    :param ngram: Integer Default 1, otherwise user can set the ngram value
    :return an encoded dataframe
    """

    ngram_output = []
    for text in all_strings:
        words = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
        print("Sentence after removing stopwords:", words)

        temp = zip(*[words[i:] for i in range(0, ngram)])
        ans = [' '.join(ngram) for ngram in temp]
        join_ans=' '.join(ans)
        join_ans = re.sub(r'[^a-zA-Z0-9\s]+', '', join_ans)
        ngram_output.append(join_ans)
    ngram_output = pd.DataFrame(ngram_output)

    return ngram_output


def data_preprocess(df, top=4):
    """
    Preprocesses the input sentences and returns an encoded vector.
    If a word in a sentence present in top K most frequent word list of the entire dataset then set 1, otherwise set 0

    :param df: Dataframe input text data
    :param top: Integer indicates the count of most frequent words
    :returns: an encoded dataframe, vocabulary
    """

    # Concatenating all strings in a column of a dataframe
    all_strings = ' '.join(df)

    # Remove special characters
    string_without_special_chars = re.sub(r'[^a-zA-Z0-9\s]+', '', all_strings)

    # Remove stop words
    word_tokens = word_tokenize(string_without_special_chars)
    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Get vocabulary
    vocabulary = set(filtered_sentence)

    # Convert all characters to lower case
    string_without_special_chars_lower = ' '.join(filtered_sentence).lower()

    # split() returns list of all the words in the string
    split_it = string_without_special_chars_lower.split()

    # Pass the split_it list to instance of Counter class.
    Counter = collections.Counter(split_it)

    # most_common() produces k frequently encountered
    most_occur = Counter.most_common(top)
    most_occur_words = [value[0] for value in most_occur]


    print(get_key_words(string_without_special_chars, 2))

    # Get binary encoding
    # encoded_data = get_encoding(df, most_occur_words)

    # Bi-gram
    ngram_encode=n_gram_encoding(df, 2)
    ngram_encoded_data = get_encoding(ngram_encode, most_occur_words)
    print(ngram_encoded_data)

    # print(filtered_sentence)
    # print(string_without_special_chars)
    # print(string_without_special_chars_lower)
    # print(most_occur)
    # print(set(vocabulary))
    # print(most_occur_words)
    # print(encoded_data)

    #return vocabulary, encoded_data

# Get dataset
df = pd.read_csv('../data/unlabeled_data/dataset/my_data_with_corrected_label.csv')
df = df['tweet_text']
data_preprocess(df)

