import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import re
import pandas as pd
import collections
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))


def get_encoding(all_strings, most_occur_words):
    """
    Encodes the data using K most frequent words:
    In an sentence if a word belongs to most frequent words list encode the word by 1 otherwise 0

    :param all_strings: Dataframe input text data
    :param most_occur_words: List of most frequent word
    :return an encoded dataframe
    """

    encode_output = []
    for st in all_strings:
        st = st.lower()
        temp_vector = [0] * len(most_occur_words)
        for i in range(len(most_occur_words)):
            if st[i] in most_occur_words:
                temp_vector[i] = 1
        encode_output.append(temp_vector)
    #encode_output = pd.DataFrame(encode_output)
    return encode_output


def data_preprocess(df, top=3):
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

    # Get binary encoding
    encoded_data = get_encoding(df, most_occur_words)

    # print(filtered_sentence)
    # print(string_without_special_chars)
    # print(string_without_special_chars_lower)
    # print(most_occur)
    # print(set(vocabulary))
    # print(most_occur_words)
    # print(encoded_data)

    return vocabulary, encoded_data

def load_reddit():
      
  df = pd.read_csv('/users/psen/preprocessed_v1.csv', delimiter=',')
  df_text = df['text']
  label_ids=df['class']
  vocab,df_text=data_preprocess(df_text,10)  
  print(type(df_text))
  x = df_text
  print(type(np.array(x)))
  labels=['0','1']

  # split into sub-datasets
  dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(x)), torch.from_numpy(np.array(label_ids)))
  train, val, test = torch.utils.data.random_split(
        dataset,
        [
            round(0.8 * len(dataset)),
            round(0.1 * len(dataset)),
            len(dataset) - round(0.8 * len(dataset)) - round(0.1 * len(dataset)),
        ],
        torch.Generator().manual_seed(42),  # Use same seed to split data
    )
  return train, val, test, vocab, labels
def load_mnist(root):
    transform = transforms.Compose([transforms.ToTensor(),  
                                    transforms.Lambda(lambda x: torch.where(x < 0.5, -1., 1.))])
    trainset = datasets.MNIST(root, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root, train=False, transform=transform, download=True)
    return trainset, testset


def load_news(root):
    # read data from pickle file
    with open(f"{root}/processed/cleaned_categories10.pkl", "rb") as f:
        data = pickle.load(f)
        x, y = data["x"].toarray(), data["y"]
        label_ids, vocab = data["label_ids"], data["vocab"]

    # binarize by thresholding 0
    x = np.where((x > 0), np.ones(x.shape), -np.ones(x.shape))
    x = np.float32(x)

    # split into sub-datasets
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train, val, test = torch.utils.data.random_split(
        dataset,
        [
            round(0.8 * len(dataset)),
            round(0.1 * len(dataset)),
            len(dataset) - round(0.8 * len(dataset)) - round(0.1 * len(dataset)),
        ],
        torch.Generator().manual_seed(42),  # Use same seed to split data
    )
    return train, val, test, vocab, list(label_ids)


class CUB200(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        root,
        image_dir='CUB_200_2011',
        split='train',
        transform=None,
):
        
        self.root = root
        self.image_dir = os.path.join(self.root, 'CUB', image_dir)
        self.transform = transform

        ## Image
        pkl_file_path = os.path.join(self.root, 'CUB', f'{split}class_level_all_features.pkl')
        self.data = []
        with open(pkl_file_path, "rb") as f:
            self.data.extend(pickle.load(f))
            
        ## Classes
        self.classes = pd.read_csv(os.path.join(self.image_dir, 'classes.txt'))['idx species'].values


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _dict = self.data[idx]

        # image
        img_path = _dict['img_path']
        _idx = img_path.split("/").index("CUB_200_2011")
        img_path = os.path.join(self.root, 'CUB/CUB_200_2011', *img_path.split("/")[_idx + 1 :])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # class label
        class_label = _dict["class_label"]
        return img, class_label


def load_cub(root):    
    transform = transforms.Compose(
        [
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ]
    )
    trainset = CUB200(root, image_dir='CUB_200_2011', split='train', transform=transform)
    testset = CUB200(root, image_dir='CUB_200_2011', split='test', transform=transform)
    valset = CUB200(root, image_dir='CUB_200_2011', split='val', transform=transform)
    return trainset, valset, testset
    
def load_cifar10(root):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    return trainset, testset
