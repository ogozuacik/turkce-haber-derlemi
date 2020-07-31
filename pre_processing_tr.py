# Ömer Gözüaçık

import pandas as pd
import re
import time
import string
from TurkishStemmer import TurkishStemmer

# Türkçe kök ayırıcı (stemmer). Projede kullanılmamıştır. 
# İstek doğrultusunda bu metot pre_processing_tr paketinin import edilmesi ile kullanılabilir.
def turkish_stemmer(s):
    stemmer = TurkishStemmer()
    words=s.lower().split()
    stemmed_words=[stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Haberlerde genel olarak bulunan bir yazım hatasını düzeltmek için bir methot.
# Genellikle iki özel bir isim başka bir kelime ile birleşik yazılmış. Methot ile yapışık iki kelime ayırılıyor.
# Örn: güzelAnkara --> güzel Ankara
def replace_lower_upper(s):
    pattern = '[a-z][A-Z]'
    while True:
        m=re.search(pattern, s)
        if m==None:
            break
        rep= m.group(0)
        s=re.sub(rep, ' '.join([str(elem) for elem in list(rep)]), s, count=1)
    return s

# Kelimeler arası birden fazla boşluk olan yerlerin tek boşluk ile değiştirilmesi için metot
def compress_whitespace(s):
    return re.sub("\s+", " " ,s.strip())

# Noktalama işaretlerini boşluğa çeviren metot (öncelikle boşluğa çevirmenin avantajı olası yazım hatalarına karşı önlem)
# Birçok haberde noktadan, virgülden sonra boşluk unutulmuştur. Örn: ... karar verildi.Meclis bu hafta tekrar toplanacak.
def remove_punc(s):
    return re.sub(r'[^\w\s]',' ',s)

# Sayıları boşluğa çeviren metot
def remove_numbers(s):
    return re.sub('[0-9]+', ' ', s)

# diğer 4 ön işleme methodunu düzgün sıra ile kullanan metot
def pre_process(s):
    return compress_whitespace(remove_punc(remove_numbers(replace_lower_upper(s)))).lower()

if __name__ == '__main__':
    print('Bu kodu doğrudan çalıştırmayınız. \n from pre_processing_tr import method_name ile dilediğiniz methotu çekebilirsiniz.')
