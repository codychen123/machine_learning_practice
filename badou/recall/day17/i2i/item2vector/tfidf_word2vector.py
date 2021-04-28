from gensim.models import word2vec
import pandas as pd

path= r"/badou/recall/day14/ua.base"

dt = pd.read_csv(path,sep="\t",header=None)
dt.columns=["uid","iid","score","ts"]
dt["iid"]=dt['iid'].astype("str")

# (943,xx)
item_list = dt.sort_values("ts").groupby("uid")["iid"].apply(list)

#
out_path = r'/badou/recall/day14/cut_ua.base'

with open(out_path,"a+",encoding="utf-8") as f:
    for line in item_list:
        if len(line)>2:
            f.write(" ".join(line)+"\n")

def gensim_word2vector(cut_path):
    # 调用gensim
    sentence = word2vec.LineSentence(cut_path)
    # hs:1,softmax 0,负采样，配合negative
    w2v = word2vec.Word2Vec(sentence,hs=1,min_count=1,window=20,size=8)
    return w2v

w2v = gensim_word2vector(out_path)
res = w2v.wv.similar_by_word("4")
print(res)
