import csv
import re
import pandas as pd
import jieba
import numpy as np
import gensim
from gensim import corpora
# from sklearn.svm import SVC
# from sklearn import preprocessing
# from keras.utils import np_utils


class dataProProcess:

    def read_data_from_csv(path):
        commentset, labelset = [], []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                commentset.append(row[0])
                labelset.append(row[1])
        return commentset, labelset

    def cut_special_char(commentset):
        reviews = []
        for val in commentset:
            # str = '肯定的 放心估计过不了多久 天天见//@Edward_C狐: 回复@cnakalai-来来:一定啊
            # //@cnakalai-来来:得类！咱怎么着也肯定挤出时间来回北京，得给您个机会招呼我们不是~[偷笑][嘻嘻]'
            p1 = re.compile(r'//@.*?:')  # ？表示贪婪模式，['//@Edward_C狐:', '//@cnakalai-来来:']
            p2 = re.compile(r'回复@.*?:')
            p3 = re.compile(r'#.*?#')
            p4 = re.compile(r'【.*?】')
            p5 = re.compile(r'http[s]?://[a-zA-Z0-9./]+')
            p6 = re.compile(r"@.*?[' ']")
            result1 = p1.findall(val)
            result2 = p2.findall(val)
            result3 = p3.findall(val)
            result4 = p4.findall(val)
            result5 = p5.findall(val)
            result6 = p6.findall(val)

            if len(result1):
                for i in range(len(result1)):
                    val = val.replace(result1[i], '')
            if len(result2):
                for i in range(len(result2)):
                    val = val.replace(result2[i], '')
            if len(result3):
                for i in range(len(result3)):
                    val = val.replace(result3[i], '')
            if len(result4):
                for i in range(len(result4)):
                    val = val.replace(result4[i], '')
            if len(result5):
                for i in range(len(result5)):
                    val = val.replace(result5[i], '')
            if len(result6):
                for i in range(len(result6)):
                    val = val.replace(result6[i], '')
            reviews.append(val)
        return reviews

    def write_trainset_testset(commentset, labelset):
        path_train = "C:\\Users\\Hygge\\Desktop\\数据挖掘\\情感分析\\data\\train87898.csv"
        path_test = "C:\\Users\\Hygge\\Desktop\\数据挖掘\\情感分析\\data\\test20000.csv"
        gap = 20000
        train = pd.DataFrame({'review': commentset[gap:-1], 'label': labelset[gap:-1]})
        train.to_csv(path_train, encoding='gbk')
        test = pd.DataFrame({'review': commentset[0:gap], 'label': labelset[0:gap]})
        test.to_csv(path_test, encoding='gbk')

    # 观察数据，截断长度，返回jieba分词结果
    def search_cutline(commentset, labelset):
        train_text = []
        for line in commentset:
            t = jieba.lcut(line)
            train_text.append(t)
        sentence_length = [len(x) for x in train_text]  # train_text是train.csv中每一行分词之后的数据 %matplotlib notebook
        # import matplotlib.pyplot as plt
        # plt.hist(sentence_length, bins=1000, density=False, cumulative=False, color='blue', alpha=0.8, align='left', rwidth=3)
        # plt.xlim(np.min(sentence_length), 100)
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.title("信息长度/次数 频数分布图")
        # plt.xlabel("length of msg")
        # plt.ylabel("times")
        # plt.show()
        return train_text

    def cut_punctuation(commentset):
        reviews = []
        punctuation = "!#$%&\'()*+,-./:;<=>?@[]^_`{|}~、。？…！，《》―；：『』＂＂￣～＜＞“”【】""「」"
        for i in range(len(commentset)):
            text = re.sub(r'[{}]+'.format(punctuation), "", commentset[i])
            reviews.append(text)
        return reviews

    # 加载停用词表
    def load_stopwords(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                data.append(line)
        return data

    # 砍掉停用词
    def cut_stopwords(commentset, cn_stopwords_list):
        data = []
        for i in range(0, len(commentset)):
            t = []
            for j in range(0, len(commentset[i])):
                if commentset[i][j] in cn_stopwords_list or commentset[i][j] == ' ':
                    pass
                else:
                    t.append(commentset[i][j])
            data.append(t)
        return data

    # 训练词向量
    def train_word2vec(commentset):
        sentences_seg = []
        gap = 20000
        for i in range(gap, len(commentset)):
            t = " ".join(commentset[i])
            sentences_seg.append(t.split())
        print("开始训练词向量")
        cnt = 0
        for i in range(20000, len(sentences_seg)):
            cnt += len(sentences_seg[i])
        print("cccc", cnt)
        model = gensim.models.Word2Vec(sentences=sentences_seg, min_count=1, size=100, workers=2,
                                       iter=8)  # min_count (int, optional) – 忽略词频小于此值的单词。
        model.save("C:\\Users\\Hygge\\Desktop\\model_vague")
        # print(model["你们"])
        model.wv.save_word2vec_format("C:\\Users\\Hygge\\Desktop\\model_available", binary=False)

    @staticmethod
    def generate_id2wec():
        new_model = gensim.models.Word2Vec.load("C:\\Users\\Hygge\\Desktop\\model_vague")
        gensim_dict = corpora.Dictionary()
        gensim_dict.doc2bow(new_model.wv.vocab.keys(), allow_update=True)
        w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
        # print(w2id)
        w2vec = {word: new_model[word] for word in w2id.keys()}  # 词语的词向量
        n_vocabs = len(w2id) + 1
        embedding_weights = np.zeros((n_vocabs, 100))
        for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
            embedding_weights[index, :] = w2vec[w]
        return w2id, embedding_weights

    # 文本转数字模式
    def text_to_array(w2index, senlist, sequence_length):  # 文本转为索引数字模式
        from keras.preprocessing.sequence import pad_sequences
        num_list = []
        max_len = sequence_length
        for sen in senlist:
            new_sen = [w2index.get(word, 0) for word in sen]  # 单词转索引数字
            num_list.append(new_sen)
        num_list = pad_sequences(num_list, maxlen=max_len, truncating="pre")
        print("pad_sequences", num_list)
        return np.array(num_list[20000:]), np.array(num_list[:20000])


class neuralNetwork:
    def __init__(self, Embedding_dim, embedding_weights, w2id, labels_category, maxlen):
        # (100, embedding_weights, w2id, sequence_length, 2)
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = maxlen
        self.model = self.create_neural_model()

    def create_neural_model(self):
        from keras import Sequential
        from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Activation
        model = Sequential()
        # input dim(140,100)
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, X_test, y_test, n_epoch=1):
        self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
                       validation_data=(X_test, y_test),callbacks=[history])
        self.model.save('C:\\Users\\Hygge\\Desktop\\sentiment.h5')

    def predict(self, model_path, new_sen):
        from keras.preprocessing.sequence import pad_sequences
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen, truncating="pre")
        res = model.predict(sen_input)[0]
        return np.argmax(res)


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


if __name__ == '__main__':
    path = 'C:\\Users\\Hygge\\Desktop\\数据挖掘\\情感分析\\train_utf8.csv'
    # path = 'C:\\Users\\Hygge\\Desktop\\数据挖掘\\情感分析\\data\\train87898.csv'

    # 加载停用词表
    path_cn_stopwords = 'C:\\Users\\Hygge\\Desktop\\数据挖掘\\情感分析\\cn_stopwords.txt'
    cn_stopwords_list = dataProProcess.load_stopwords(path=path_cn_stopwords)
    # print(len(cn_stopwords_list), cn_stopwords_list)

    reviewset, labelset = dataProProcess.read_data_from_csv(path=path)
    reviewset.remove('review')
    labelset.remove('label')

    # 对句子进行去掉http等连接形式的无用信息
    reviews = dataProProcess.cut_special_char(reviewset)
    # print("reviews", reviews[0],type(reviews))

    # 得到分词结果
    reviews = dataProProcess.search_cutline(reviews, labelset)
    print("reviews", len(reviews), reviews[1])

    # 删除停用词,今后还可以继续扩展停用词
    reviews = dataProProcess.cut_stopwords(reviews, cn_stopwords_list=cn_stopwords_list)
    print("reviews", len(reviews), reviews[1])

    # dataProProcess.write_trainset_testset(reviews, labelset)

    # 训练词向量得到model,训练一次得到模型即可
    # dataProProcess.train_word2vec(reviews)

    new_model = gensim.models.Word2Vec.load("C:\\Users\\Hygge\\Desktop\\model_vague")
    print("aaaaaaaaa", new_model.most_similar(positive=["泪"]))

    # 得到词向量对应的w2id和embedding_weights
    w2id, embedding_weights = dataProProcess.generate_id2wec()

    # 求最大文本长度
    num_list = [len(one) for one in reviews[20000:]]  # 类似[3, 2, 3, 3, 3, 2, 3, 3]
    sequence_length = max(num_list)  # 求取最大文本长度3

    # 比如，实现将“泪”取代数字1
    X_train, X_test = dataProProcess.text_to_array(w2id, reviews, sequence_length)
    print("X_train and X_test", X_train, X_test)

    history = LossHistory()

    # 神经网络
    neuralNetwork = neuralNetwork(100, embedding_weights, w2id, 2, sequence_length)
    neuralNetwork.train(X_train, np.array(np_utils.to_categorical(labelset[20000:])), X_test,
                        np.array(np_utils.to_categorical(labelset[:20000])))

    history.loss_plot('epoch')


    #给出数据集，进行验证
    with open("C:\\Users\\Hygge\\Desktop\\数据挖掘\\情感分析\\测试集输入.csv", "r") as f:
        reader = csv.reader(f)
        result = list(reader)

    predict_label = []
    ans = 0
    for i in range(len(result)):

        label_dic = {0: "消极的", 1: "积极的"}
        # sen_new = "现如今的公司能够做成这样已经很不错了，微订点单网站的信息更新很及时，内容来源很真实"
        sen_new = result[i][0]
        pre = neuralNetwork.predict("C:\\Users\\Hygge\\Desktop\\sentiment.h5", sen_new)
        predict_label.append(pre)
        print("pre", pre)
        # if pre == int(result[i][1]):
        #     ans += 1
        # print("'{}'的情感是:\n{}".format(sen_new, label_dic.get(pre)))
    print(predict_label)
    result_file = "result.csv"
    res = pd.DataFrame(predict_label)
    res.to_csv(result_file)
    # print(ans)
    # print("准确率为：",  ans/len(result))
    '''
    87776/87988 [============================>.] - ETA: 0s - loss: 0.0556 - acc: 0.9787
    87808/87988 [============================>.] - ETA: 0s - loss: 0.0556 - acc: 0.9786
    87840/87988 [============================>.] - ETA: 0s - loss: 0.0556 - acc: 0.9786
    87872/87988 [============================>.] - ETA: 0s - loss: 0.0556 - acc: 0.9786
    87904/87988 [============================>.] - ETA: 0s - loss: 0.0556 - acc: 0.9786
    87936/87988 [============================>.] - ETA: 0s - loss: 0.0555 - acc: 0.9786
    87968/87988 [============================>.] - ETA: 0s - loss: 0.0556 - acc: 0.9786
    87988/87988 [==============================] - 398s 5ms/step - loss: 0.0556 - acc: 0.9786 - val_loss: 0.1298 - val_acc: 0.9659
    '''
    '''
    448s 5ms/step - loss: 0.0923 - acc: 0.9740 - val_loss: 0.0779 - val_acc: 0.9780
    '''
    # model = dataProProcess.define_model(vocab_cnt + 1, sequence_length)
    # model.fit(inputs, outputs, epochs=5, verbose=2, batch_size=5)
    # model.save('C:\\Users\\Hygge\\Desktop\\词向量模型分类.h5')


    # w2id {'冯小刚': 5611, '牡丹': 20130}
    # print("w2id", w2id)
    # print("222", w2id["泪"])
    # embedding_weights[18740]类型为<class 'numpy.ndarray'>
    # print("111", embedding_weights[16160].tolist())
    # print("w2id and embedding_weights:", len(w2id), len(embedding_weights))

    # reviews_val_coup = dataProProcess.compute_score(w2id, embedding_weights)
    # print("reviews_val_coup", reviews_val_coup)

    # X_train, X_test, y_train, y_test = dataProProcess.prepareData(reviews, labelset, reviews_val_coup)
    # print("X_train", X_train, len(X_train))
    # print("X_test", X_test, len(X_test))



    # str = '我们只是无意闯进你世界的陌路，你却在瞬间深深将我吸引。----记泸沽湖之行 [鼓掌][鼓掌] 请戳>>> http://t.cn/8Fm8OGu 谢谢台前幕后的用心与专业，成就了这样活泼有趣的节目，也让我们看到大夏的各种美. 我怀念的[泪]'
    # p6 = re.compile(r"@.*?[' ']")
    # result = p6.findall(str)
    # print(result)

    # sentence = '我爱自然语言处理'
    # generator = jieba.lcut(sentence) # 返回的不是生成器，而是一个list
    # print(generator)

    # import matplotlib.pyplot as plt
    #
    # a = [1, 2, 3, 4, 5, 2, 3, 1, -1, -1]
    # plt.hist([1, 2, 3, 4, 5, 2, 3, 1, -1, -1], bins=20, density=False, cumulative=False, color='blue', alpha=0.8, align='left')
    # plt.xlim(np.min(a), np.max(a))
    # plt.show()

    # import re
    # punctuation = "。"
    # line = "测试。。去除标点。。"
    # print(re.sub("[{}]+".format(punctuation), " ", line))
