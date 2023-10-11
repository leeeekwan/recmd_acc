import os
import re
import time
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

image_dir='/home/aimlab/outfit-curation-server-master/data/images/preprocessed' ## 이미지디렉토리
image=tf.keras.preprocessing.image
preprocess=tf.keras.applications.inception_v3.preprocess_input
myinception=tf.keras.applications.inception_v3.InceptionV3(include_top = False, pooling = 'max') ##사용할 모델

# 100개에 4분
def get_embedding_vector(item):
    try:
        img=Image.open(os.path.join(image_dir, item + '.jpg')).resize((229, 229))
        mat=image.img_to_array(img)
        #mat.shape는 (229,229,3)이어야함
        if len(mat.shape)==3:
            if mat.shape[2]!=3:
                if mat.shape[2]==4:
                    mat=mat[:, :, :3]
                else:
                    return False
            mat=np.expand_dims(mat, axis=0)
            mat_p=preprocess(mat)
            itemvector=np.array(myinception.predict(mat_p))
            return itemvector
        else:
            return False # mat len이 3이 아닐 경우
    except:
        return False

if __name__=="__main__":

    with open('/home/aimlab/jhkim/Acc/data/2022-05-04_dailySaleItem_acc.json', encoding='utf-8') as f:
        daily=json.load(f)['data']
    # with open('/home/aimlab/jhkim/Acc/data/item_attribute.json', encoding='utf-8') as f:
    #     item_attribute=json.load(f)
    # with open('/home/aimlab/jhkim/Acc/embedding_vector_dict.pickle', 'rb') as f: #기존의 벡터 
    #     embedding_vector_dict_yesterday=pickle.load(f)

    daily_dict={item['skuCode']:item for item in daily}
    
    embedding_vector_dict={}
    for item in tqdm(daily_dict):
        try:
            if daily_dict[item]['galCategory'][0]['galCate02'] in ['SCARF', 'HAT', 'SOCKS', 'GLASSES', 'C&S', 'SWEATER', 'SHIRTS', 'BLOUSE&SHIRTS', 'ONEPIECE', 'VEST', 'COAT', 'DOWNPADDING', 'JACKET&JUMPER', 'LEATHER', 'SUIT', 'PANTS', 'SKIRT', 'SHOES']:
                #try:
                    #embedding_vector_dict[item]=embedding_vector_dict_yesterday[item]
                #except:
                
                vec_return=get_embedding_vector(item)
                if type(vec_return) is bool:
                    pass
                else:
                    embedding_vector_dict[item]=vec_return
        except:
            pass

#     with open('/home/aimlab/jhkim/Acc/embedding_vector_dict.pickle', 'wb') as f:
#         pickle.dump(embedding_vector_dict, f)