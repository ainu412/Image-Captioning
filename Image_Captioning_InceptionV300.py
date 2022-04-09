#!/usr/bin/env python
# coding: utf-8

# In[49]:
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass

tensorflow_shutup()


######################### values you need to specify ############################
# images_folder = 'Flickr8k_Dataset/Flicker8k_Dataset/'
# # test_images = {'2157173498_2eea42ee38.jpg', '3275537015_74e04c0f3e.jpg'}
# test_images = os.listdir(images_folder)
# parser = argparse.ArgumentParser()
# parser.add_argument("--folder", type=str, default='Flickr8k_Dataset/Flicker8k_Dataset/',
#                     help="folder of the images")
# parser.add_argument("--images", type=str,
#                     help="name of the images, seperated by ;, e.g. 'hehe.jpg hihi.jpg'")
# parser.add_argument("--txt", type=str, default='The staff is good.',
#                     help="review text")
# args = parser.parse_args()
# images_folder = args.folder
# # test_images = {'2157173498_2eea42ee38.jpg', '3275537015_74e04c0f3e.jpg'}
# test_images = set((args.images).split(';')) if args.images else os.listdir(images_folder)
# text = args.txt

def absa(images_folder='Flickr8k_Dataset/Flicker8k_Dataset/', test_images=None, text='The staff is good.'):
    import os

    if test_images is None:
        test_images = os.listdir(images_folder)
    elif isinstance(test_images, str):
        test_images = test_images.split(';')
    ######################### values you need to specify ############################
    # images_folder = 'Flickr8k_Dataset/Flicker8k_Dataset/'

    #################################################################################
    print('images_folder:', images_folder)
    print('test_images:', test_images)
    print('text:', text)
    # ######################### values you need to specify ############################
    # images_folder = 'Flickr8k_Dataset/Flicker8k_Dataset/'
    # # test_images = {'2157173498_2eea42ee38.jpg', '3275537015_74e04c0f3e.jpg'}
    # test_images = os.listdir(images_folder)
    # # test_images = {'a.jpg', 'IMG_20211006_213803.jpg'}
    # #################################################################################

    ## or you can write the name of the images into the file called
    ## 'Flickr8k_text/Flickr_8k.testImages.txt'
    ## and uncomment the below two lines
    # test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'
    # test_images = set(open(test_images_file, 'r').read().strip().split('\n'))


    # # import

    # In[610]:


    from model.bert import bert_ATE, bert_ABSA
    from data.dataset import dataset_ATM, dataset_ABSA
    from torch.utils.data import DataLoader, ConcatDataset
    from transformers import BertTokenizer
    import torch
    from torch.nn.utils.rnn import pad_sequence
    import pandas as pd
    import time


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrain_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
    lr = 2e-5
    model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
    optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
    model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
    optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)


    # In[611]:


    def evl_time(t):
        min, sec= divmod(t, 60)
        hr, min = divmod(min, 60)
        return int(hr), int(min), int(sec)

    def load_model(model, path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        return model

    def save_model(model, name):
        torch.save(model.state_dict(), name)


    # # Acpect Term Extraction

    # In[612]:


    laptops_train_ds = dataset_ATM(pd.read_csv("data/laptops_train.csv"), tokenizer)
    laptops_test_ds = dataset_ATM(pd.read_csv("data/laptops_test.csv"), tokenizer)
    restaurants_train_ds = dataset_ATM(pd.read_csv("data/restaurants_train.csv"), tokenizer)
    restaurants_test_ds = dataset_ATM(pd.read_csv("data/restaurants_test.csv"), tokenizer)
    twitter_train_ds = dataset_ATM(pd.read_csv("data/twitter_train.csv"), tokenizer)
    twitter_test_ds = dataset_ATM(pd.read_csv("data/twitter_test.csv"), tokenizer)


    # In[613]:


    # w,x,y,z = laptops_train_ds.__getitem__(121)
    # print(w)
    # print(x)
    # print(x.size())
    # print(y)
    # print(y.size())
    # print(z)
    # print(z.size())


    # In[614]:


    train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])
    test_ds = ConcatDataset([laptops_test_ds, restaurants_test_ds, twitter_test_ds])


    # In[615]:


    def create_mini_batch(samples):
        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)

        tags_tensors = [s[2] for s in samples]
        tags_tensors = pad_sequence(tags_tensors, batch_first=True)

        pols_tensors = [s[3] for s in samples]
        pols_tensors = pad_sequence(pols_tensors, batch_first=True)

        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

        return ids_tensors, tags_tensors, pols_tensors, masks_tensors


    # In[616]:


    train_loader = DataLoader(train_ds, batch_size=5, collate_fn=create_mini_batch, shuffle = True)
    test_loader = DataLoader(test_ds, batch_size=50, collate_fn=create_mini_batch, shuffle = True)


    # In[617]:


    # for batch in train_loader:
    #     w,x,y,z = batch
    #     print(w)
    #     print(w.size())
    #     print(x)
    #     print(x.size())
    #     print(y)
    #     print(y.size())
    #     print(z)
    #     print(z.size())
    #     break


    # In[618]:


    def train_model_ATE(loader, epochs):
        all_data = len(loader)
        for epoch in range(epochs):
            finish_data = 0
            losses = []
            current_times = []
            correct_predictions = 0

            for data in loader:
                t0 = time.time()
                ids_tensors, tags_tensors, _, masks_tensors = data
                ids_tensors = ids_tensors.to(DEVICE)
                tags_tensors = tags_tensors.to(DEVICE)
                masks_tensors = masks_tensors.to(DEVICE)

                loss = model_ATE(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors)
                losses.append(loss.item())
                loss.backward()
                optimizer_ATE.step()
                optimizer_ATE.zero_grad()

                finish_data += 1
                current_times.append(round(time.time()-t0,3))
                current = np.mean(current_times)
                hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
                print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)

            save_model(model_ATE, 'bert_ATE.pkl')

    def test_model_ATE(loader):
        pred = []
        trueth = []
        with torch.no_grad():
            for data in loader:

                ids_tensors, tags_tensors, _, masks_tensors = data
                ids_tensors = ids_tensors.to(DEVICE)
                tags_tensors = tags_tensors.to(DEVICE)
                masks_tensors = masks_tensors.to(DEVICE)

                outputs = model_ATE(ids_tensors=ids_tensors, tags_tensors=None, masks_tensors=masks_tensors)

                _, predictions = torch.max(outputs, dim=2)

                pred += list([int(j) for i in predictions for j in i ])
                trueth += list([int(j) for i in tags_tensors for j in i ])

        return trueth, pred


    # In[619]:


    # get_ipython().run_line_magic('time', '')
    # train_model_ATE(train_loader, 3)


    # In[620]:


    model_ATE = load_model(model_ATE, 'bert_ATE.pkl')


    # # Aspect Based Sentiment Analysis

    # In[621]:


    laptops_train_ds = dataset_ABSA(pd.read_csv("data/laptops_train.csv"), tokenizer)
    laptops_test_ds = dataset_ABSA(pd.read_csv("data/laptops_test.csv"), tokenizer)
    restaurants_train_ds = dataset_ABSA(pd.read_csv("data/restaurants_train.csv"), tokenizer)
    restaurants_test_ds = dataset_ABSA(pd.read_csv("data/restaurants_test.csv"), tokenizer)
    twitter_train_ds = dataset_ABSA(pd.read_csv("data/twitter_train.csv"), tokenizer)
    twitter_test_ds = dataset_ABSA(pd.read_csv("data/twitter_test.csv"), tokenizer)


    # In[622]:


    w,x,y,z = laptops_train_ds.__getitem__(121)


    # In[623]:


    def create_mini_batch2(samples):
        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)

        segments_tensors = [s[2] for s in samples]
        segments_tensors = pad_sequence(segments_tensors, batch_first=True)

        label_ids = torch.stack([s[3] for s in samples])

        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

        return ids_tensors, segments_tensors, masks_tensors, label_ids


    # In[624]:


    train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])
    test_ds = ConcatDataset([laptops_test_ds, restaurants_test_ds, twitter_test_ds])

    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=create_mini_batch2, shuffle = True)
    test_loader = DataLoader(test_ds, batch_size=2, collate_fn=create_mini_batch2, shuffle = True)
    torch.cuda.empty_cache()


    # In[625]:


    # for batch in train_loader:
    #     w,x,y,z = batch
    #     print(w)
    #     print(w.size())
    #     print(x)
    #     print(x.size())
    #     print(y)
    #     print(y.size())
    #     print(z)
    #     print(z.size())
    #     break


    # In[626]:


    def train_model_ABSA(loader, epochs):
        all_data = len(loader)
        for epoch in range(epochs):
            finish_data = 0
            losses = []
            current_times = []
            correct_predictions = 0

            for data in loader:
                t0 = time.time()
                ids_tensors, segments_tensors, masks_tensors, label_ids = data
                ids_tensors = ids_tensors.to(DEVICE)
                segments_tensors = segments_tensors.to(DEVICE)
                label_ids = label_ids.to(DEVICE)
                masks_tensors = masks_tensors.to(DEVICE)

                loss = model_ABSA(ids_tensors=ids_tensors, lable_tensors=label_ids, masks_tensors=masks_tensors, segments_tensors=segments_tensors)
                losses.append(loss.item())
                loss.backward()
                optimizer_ABSA.step()
                optimizer_ABSA.zero_grad()

                finish_data += 1
                current_times.append(round(time.time()-t0,3))
                current = np.mean(current_times)
                hr, min, sec = evl_time(current*(all_data-finish_data) + current*all_data*(epochs-epoch-1))
                print('epoch:', epoch, " batch:", finish_data, "/" , all_data, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)

            save_model(model_ABSA, 'bert_ABSA2.pkl')

    def test_model_ABSA(loader):
        pred = []
        trueth = []
        with torch.no_grad():
            for data in loader:

                ids_tensors, segments_tensors, masks_tensors, label_ids = data
                ids_tensors = ids_tensors.to(DEVICE)
                segments_tensors = segments_tensors.to(DEVICE)
                masks_tensors = masks_tensors.to(DEVICE)

                outputs = model_ABSA(ids_tensors, None, masks_tensors=masks_tensors, segments_tensors=segments_tensors)

                _, predictions = torch.max(outputs, dim=1)

                pred += list([int(i) for i in predictions])
                trueth += list([int(i) for i in label_ids])

        return trueth, pred



    # In[627]:


    model_ABSA = load_model(model_ABSA, 'bert_ABSA2.pkl')


    # In[628]:


    # %time
    # x, y = test_model_ABSA(test_loader)
    # print(classification_report(x, y, target_names=[str(i) for i in range(3)]))


    # # ATE + ABSA

    # In[629]:


    def predict_model_ABSA(sentence, aspect, tokenizer):
        t1 = tokenizer.tokenize(sentence)
        t2 = tokenizer.tokenize(aspect)

        word_pieces = ['[cls]']
        word_pieces += t1
        word_pieces += ['[sep]']
        word_pieces += t2

        segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

        ids = tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(DEVICE)
        segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

        with torch.no_grad():
            outputs = model_ABSA(input_tensor, None, None, segments_tensors=segment_tensor)
            _, predictions = torch.max(outputs, dim=1)

        return word_pieces, predictions, outputs

    def predict_model_ATE(sentence, tokenizer):
        word_pieces = []
        tokens = tokenizer.tokenize(sentence)
        word_pieces += tokens

        ids = tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(DEVICE)

        with torch.no_grad():
            outputs = model_ATE(input_tensor, None, None)
            # print('outputs', outputs)
            _, predictions = torch.max(outputs, dim=2)
            # print('predictions', predictions)
        predictions = predictions[0].tolist()

        return word_pieces, predictions, outputs

    def ATE_ABSA(text):
        dic = {}
        terms = []
        word = ""
        sent = {0: 'negative', 1: 'neutral', 2: 'positive'}
        x, y, z = predict_model_ATE(text, tokenizer)
        # print('before y:', y)
        while sum(y) == 0:
            # print('I am in')
            for i, row in enumerate(z[0]):
                z[0][i] = torch.tensor([row[0], row[1] + 0.1, row[2] + 0.1])

            _, y = torch.max(z, dim=2)
            y = y[0]
            # print('in y:', y)


        # print('y', y)
        for i in range(len(y)):
            if y[i] == 1:
                if len(word) != 0:
                    terms.append(word.replace(" ##",""))
                word = x[i]

            if y[i] == 2:
                word += (" " + x[i])


        if len(word) != 0:
                terms.append(word.replace(" ##",""))
        # print("----------------------------------")
        # print("Text:", text)
        # print("tokens:", x)
        # print("ATE:", terms)

        if len(terms) != 0:
            for i in terms:
                _, c, p = predict_model_ABSA(text, i, tokenizer)
                dic[i] = sent[int(c)]
                # print("term:", [i], "class:", [sent[int(c)]])
        return dic

    # In[630]:


    model_ABSA = load_model(model_ABSA, 'bert_ABSA2.pkl')
    model_ATE = load_model(model_ATE, 'bert_ATE.pkl')


    # 0: negative
    # 1: neutral
    # 2: positive




    # In[632]:


    import glob
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    # get_ipython().run_line_magic('matplotlib', 'inline')
    import pickle
    from tqdm import tqdm
    import pandas as pd
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
    from keras.optimizers import Adam, RMSprop
    from keras.layers.wrappers import Bidirectional
    from keras.applications.inception_v3 import InceptionV3
    from keras.preprocessing import image
    import os
    DEBUG = True


    # In[633]:


    token = 'Flickr8k_text/Flickr8k.token.txt'
    captions = open(token, 'r').read().strip().split('\n')


    # ## Creating a dictionary containing all the captions of the images

    # In[634]:


    d = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0])-2]
        if row[0] in d:
            d[row[0]].append(row[1])
        else:
            d[row[0]] = [row[1]]


    # In[635]:


    train_images_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'


    # In[636]:


    train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
    # print(train_images)


    # In[637]:


    img = glob.glob(images_folder+'*.jpg')
    def split_data(l):
        temp = []
        for i in img:
            if i[len(images_folder):] in l:
                temp.append(i)
        return temp


    # In[638]:


    # Getting the training images from all the images
    train_img = split_data(train_images)
    len(train_img)


    # In[639]:


    val_images_file = 'Flickr8k_text/Flickr_8k.devImages.txt'
    val_images = set(open(val_images_file, 'r').read().strip().split('\n'))


    # In[640]:


    # Getting the validation images from all the images
    val_img = split_data(val_images)


    # In[641]:


    # Getting the testing images from all the images
    test_img = split_data(test_images)
    len(test_img)


    # We will feed these images to VGG-16 to get the encoded images. Hence we need to preprocess the images as the authors of VGG-16 did. The last layer of VGG-16 is the softmax classifier(FC layer with 1000 hidden neurons) which returns the probability of a class. This layer should be removed so as to get a feature representation of an image. We will use the last Dense layer(4096 hidden neurons) after popping the classifier layer. Hence the shape of the encoded image will be (1, 4096)

    # In[642]:


    def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x


    # In[643]:


    def preprocess(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)
        return x


    # In[644]:


    # plt.imshow(np.squeeze(preprocess(train_img[0])))


    # In[645]:


    model = InceptionV3(weights='imagenet')


    # In[646]:


    from keras.models import Model

    new_input = model.input
    hidden_layer = model.layers[-2].output

    model_new = Model(new_input, hidden_layer)


    # In[647]:


    # tryi = model_new.predict(preprocess(train_img[0]))


    # In[648]:


    # tryi.shape


    # In[648]:





    # In[649]:


    def encode(image):
        image = preprocess(image)
        temp_enc = model_new.predict(image)
        temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
        return temp_enc


    # In[650]:


    if not os.path.exists("encoded_images_inceptionV3.p"):
        encoding_train = {}
        for img in tqdm(train_img):
            encoding_train[img[len(images_folder):]] = encode(img)


        with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
            pickle.dump(encoding_train, encoded_pickle)


    # In[651]:


    encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))


    # In[652]:


    if not os.path.exists('encoded_images_test_inceptionV3.p') or DEBUG:
        encoding_test = {}
        for img in tqdm(test_img):
            encoding_test[img[len(images_folder):]] = encode(img)

        with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
            pickle.dump(encoding_test, encoded_pickle)


    # In[653]:


    encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))


    # In[654]:


    # encoding_test[test_img[0][len(images_folder):]].shape


    # In[655]:


    train_d = {}
    for i in train_img:
        if i[len(images_folder):] in d:
            train_d[i] = d[i[len(images_folder):]]

    len(train_d)

    caps = []
    for key, val in train_d.items():
        for i in val:
            caps.append('<start> ' + i + ' <end>')


    # In[656]:


    val_d = {}
    for i in val_img:
        if i[len(images_folder):] in d:
            val_d[i] = d[i[len(images_folder):]]

    len(val_d)


    # Calculating the unique words in the vocabulary.

    # In[657]:


    caps = []
    for key, val in train_d.items():
        for i in val:
            caps.append('<start> ' + i + ' <end>')


    # In[658]:


    words = [i.split() for i in caps]


    # In[659]:


    # unique = []
    # for i in words:
    #     unique.extend(i)


    # In[660]:


    # unique = list(set(unique))


    # In[661]:


    # with open("unique.p", "wb") as pickle_d:
    #     pickle.dump(unique, pickle_d)


    # In[662]:


    unique = pickle.load(open('unique.p', 'rb'))


    # In[663]:


    len(unique)


    # Mapping the unique words to indices and vice-versa

    # In[664]:


    word2idx = {val:index for index, val in enumerate(unique)}


    # In[665]:


    word2idx['<start>']


    # In[666]:


    idx2word = {index:val for index, val in enumerate(unique)}


    # In[667]:


    idx2word[5553]


    # Calculating the maximum length among all the captions

    # In[668]:


    max_len = 0
    for c in caps:
        c = c.split()
        if len(c) > max_len:
            max_len = len(c)
    max_len


    # In[669]:


    len(unique), max_len


    # In[670]:


    vocab_size = len(unique)
    vocab_size


    # Adding <start> and <end> to all the captions to indicate the starting and ending of a sentence. This will be used while we predict the caption of an image

    # In[671]:


    f = open('flickr8k_training_dataset.txt', 'w')
    f.write("image_id\tcaptions\n")


    # In[672]:


    for key, val in train_d.items():
        for i in val:
            f.write(key[len(images_folder):] + "\t" + "<start> " + i +" <end>" + "\n")

    f.close()


    # In[673]:


    df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')


    # In[674]:


    len(df)


    # In[675]:


    c = [i for i in df['captions']]
    len(c)


    # In[676]:


    imgs = [i for i in df['image_id']]


    # In[677]:


    # a = c[-1]
    # a, imgs[-1]


    # In[678]:


    # for i in a.split():
    #     print (i, "=>", word2idx[i])


    # In[679]:


    samples_per_epoch = 0
    for ca in caps:
        samples_per_epoch += len(ca.split())-1


    # In[680]:


    samples_per_epoch


    # ## Generator
    #
    # We will use the encoding of an image and use a start word to predict the next word.
    # After that, we will again use the same image and use the predicted word
    # to predict the next word.
    # So, the image will be used at every iteration for the entire caption.
    # This is how we will generate the caption for an image. Hence, we need to create
    # a custom generator for that.
    #
    # The CS231n lecture by Andrej Karpathy explains this concept very clearly and beautifully.
    # Link for the lecture:- https://youtu.be/cO0a0QYmFm8?t=32m25s

    # In[681]:


    def data_generator(batch_size = 32):
            partial_caps = []
            next_words = []
            images = []

            df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
            df = df.sample(frac=1)
            iter = df.iterrows()
            c = []
            imgs = []
            for i in range(df.shape[0]):
                x = next(iter)
                c.append(x[1][1])
                imgs.append(x[1][0])


            count = 0
            while True:
                for j, text in enumerate(c):
                    current_image = encoding_train[imgs[j]]
                    for i in range(len(text.split())-1):
                        count+=1

                        partial = [word2idx[txt] for txt in text.split()[:i+1]]
                        partial_caps.append(partial)

                        # Initializing with zeros to create a one-hot encoding matrix
                        # This is what we have to predict
                        # Hence initializing it with vocab_size length
                        n = np.zeros(vocab_size)
                        # Setting the next word to 1 in the one-hot encoded matrix
                        n[word2idx[text.split()[i+1]]] = 1
                        next_words.append(n)

                        images.append(current_image)

                        if count>=batch_size:
                            next_words = np.asarray(next_words)
                            images = np.asarray(images)
                            partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                            yield [[images, partial_caps], next_words]
                            partial_caps = []
                            next_words = []
                            images = []
                            count = 0


    # ## Let's create the model

    # In[682]:


    embedding_size = 300


    # Input dimension is 4096 since we will feed it the encoded version of the image.

    # In[683]:


    image_model = Sequential([
            Dense(embedding_size, input_shape=(2048,), activation='relu'),
            RepeatVector(max_len)
        ])


    # Since we are going to predict the next word using the previous words(length of previous words changes with every iteration over the caption), we have to set return_sequences = True.

    # In[684]:


    caption_model = Sequential([
            Embedding(vocab_size, embedding_size, input_length=max_len),
            LSTM(256, return_sequences=True),
            TimeDistributed(Dense(300))
        ])


    # Merging the models and creating a softmax classifier

    # In[685]:


    final_model = Sequential([
            Merge([image_model, caption_model], mode='concat', concat_axis=1),
            Bidirectional(LSTM(256, return_sequences=False)),
            Dense(vocab_size),
            Activation('softmax')
        ])


    # In[686]:


    final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])


    # In[687]:


    # final_model.summary()


    # In[688]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[689]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[690]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[691]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[692]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[693]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[694]:


    # final_model.optimizer.lr = 1e-4
    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[695]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[696]:


    # final_model.save_weights('time_inceptionV3_7_loss_3.2604.h5')
    #


    # In[697]:


    # final_model.load_weights('time_inceptionV3_7_loss_3.2604.h5')
    #


    # In[698]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[699]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[700]:


    # final_model.save_weights('time_inceptionV3_3.21_loss.h5')
    #


    # In[701]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[702]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[703]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)
    #


    # In[704]:


    # final_model.save_weights('time_inceptionV3_3.15_loss.h5')
    #


    # In[705]:


    # final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1,
    #                           verbose=2)


    # In[706]:


    final_model.load_weights('weights/time_inceptionV3_1.5987_loss.h5')


    # ## Predict function

    # In[707]:


    def predict_captions(image):
        start_word = ["<start>"]
        while True:
            par_caps = [word2idx[i] for i in start_word]
            par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images_folder):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            word_pred = idx2word[np.argmax(preds[0])]
            start_word.append(word_pred)

            if word_pred == "<end>" or len(start_word) > max_len:
                break

        return ' '.join(start_word[1:-1])


    # In[708]:



    ################### final output #####################
    # im = 'Flickr8k_Dataset/Flicker8k_Dataset/IMG_20211006_213803.jpg'
    for im in test_img:
        img_txt = predict_captions(im)
        res_dic = ATE_ABSA(img_txt)
        print("----------------------------------")
        print("image name:", im)
        for key, val in res_dic.items():
            print('term:', [key], ' class:', [val])


    # print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
    # print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
    # print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
    # Image.open(im)

    # In[631]:

    print("----------------------------------")
    print('text:', text)
    res_dic = ATE_ABSA(text)
    for key, val in res_dic.items():
        print('term:', [key], ' class:', [val])

if __name__ == '__main__':
    absa(test_images='a.jpg;b.jpg')
    absa(test_images=['a.jpg', 'b.jpg'])