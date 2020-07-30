# Ömer Gözüaçık

import pickle
import pandas as pd
import pre_processing_tr as pr
from sklearn.feature_extraction.text import CountVectorizer

# Sınıflandırılmak istenen metin girilir.
text = input("Sınıflandırmak istediğiniz haber metnini giriniz. \n\n")

# Metin ön işleyiciden geçirildikten sonra sonuç
text_processed = pr.pre_process(text)
print('\nÖn işleme sonucu metinin son hali: \n')
print(text_processed)

#Önceden eğitilmiş modellerin yüklenmesi
filename = '4-kategori_vocab.sav'
loaded_vocab = pickle.load(open(filename, 'rb'))

cv = CountVectorizer(vocabulary=loaded_vocab)
text_vector = cv.fit_transform([text_processed])

filename = '4-kategori.sav'
loaded_model = pickle.load(open(filename, 'rb'))

class_mapping = {0: 'Ekonomi', 1: 'Siyaset', 2: 'Spor', 3: 'Teknoloji_Bilim'}

#Tahmin aşaması
pred = loaded_model.predict(text_vector)
print(f"\nKategori tahmini: {class_mapping[pred[0]]}")