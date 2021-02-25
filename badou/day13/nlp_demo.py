import jieba

list1= ["我喜欢学习",
        "她喜欢学习"
        "他不喜欢学习",
        "她喜欢大数据",
        "她喜欢算法"
        ]


word_dict={}

index=0
for sent in list1:
    words = jieba.cut(sent)
    for word in words:
        if word_dict.get(word,-1)==-1:
            word_dict[word]=index
            index+=1

print(word_dict)