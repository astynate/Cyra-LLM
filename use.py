# -*- coding: utf-8 -*-

# import tensorflow as tf
from cyra_model.tokenizer import CyraTokenizer
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras import mixed_precision
from cyra_model.model import Cyra, GradientClippingOptimizer
import os

mixed_precision.set_global_policy('mixed_float16')

cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/tokenizer.txt'
cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

cyra_model = Cyra(cyra_tokenizer, 12, 256, 12, 2048, path='D:/Exider Company/Cyra/trained-models/cyra.h5')
# checkpoint_path = "D:/Exider Company/Cyra/trained-models/cyra_check_point.ckpt"

# if os.path.exists(checkpoint_path):
#     print(f'Model weights are loded form: {checkpoint_path}')
#     cyra_model.model.load_weights(checkpoint_path)
# cyra_model.model.save('D:/Exider Company/Cyra/trained-models/cyra.h5')

# model = load_model('trained-models/cyra.h5')

data = input('Ask Cyra: ')

data = cyra_tokenizer.get_sequence(data)
data = cyra_tokenizer.get_text(data)

generated_text = ''

for i in range(20):
 
    generated_token = cyra_model(data)
    data += generated_token
    generated_text += generated_token
    
print(f'{generated_text}', end='')

# print(cyra_model.model.predict(np.array(data, dtype='float16')))
# print(cyra_tokenizer.get_full_sequence('микробиологи обнаружили необычную мутантную форму кишечной палочки клетки которой способны сотни раз превышать размеру нормальные клетки обычной толщине длина достигает 07 миллиметра описание мутанта опубликовано journal bacteriolog мутантный штамм удалось получить фактически случайно первоначальной задачей исследователей поиск факторов которые делают бактерий другого штамма зависимыми аминокислоты метионина этого ученые провели рандомизированный мутагенез среди бактерий которые могли расти без метионина обнаружили микроорганизмы необычной формой клеток дальнейшие исследования показали новый штамм получивший названии eel extrem elong английском eel угорь практически отличается обычных кишечных палочек удлиняется примерно той скорость какой растут нормальные escherichia coli переносит температуры выживает тех условиях фактически единственно отличие нового штамма заключается низком содержании белка ftsz который участвует делении бактериальных клеток именно этот белок время деления образует поперечный диск делящий бактерию две половины этом сам ген белка ftsz бактерий штамма eel содержит мутаций вероятно появились других генах которые связаны работой ftsz гены какие мутации появились штамма eel ученые пока установили мутанты гену ftsz которые повышении температуры перестают делится известны образовывали вытянутые клетки никогда вырастали настолько большими изза погибали течение нескольких часов словам авторов мутации нового штамма eel точные смысле направленности механизм деления другими словами затрагивают другие важные процессы жизнедеятельности бактерий ученые надеются подобные длинные бактериальные клетки можно будет использовать нап'))