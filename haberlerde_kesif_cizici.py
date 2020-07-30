import pandas as pd
import matplotlib.pyplot as plt

# veri okuma
data=pd.read_csv('derlemler/filtrelenmis_temizlenmis_derlem.csv.gz')

# grafik çizimi için yardımcı metot
def window_average(x,N):
    low_index = 0
    high_index = low_index + N
    w_avg = []
    while(high_index<len(x)):
        temp = sum(x[low_index:high_index])/N
        w_avg.append(temp)
        low_index = low_index + N
        high_index = high_index + N
    return w_avg

# toplam ay sayısı (23 yıl)
n_bins = 276 

# bir kelimenin bütün haberlerde toplam bulunma sayısı ve 1 aylık periyotlarda haberlerde bulunma durumu için metot
def kelime_frekansı(word):
    word = word.lower()
    word_freq = data.news.apply(lambda text: 1 if word in text else 0)
    sum_word = word_freq.rolling(300).sum()
    sum_word = sum_word[300:]
    sum_word = window_average(sum_word, int(len(sum_word)/(n_bins+1)))
    return sum(word_freq), sum_word

#kelimelerin kullanıcı tarafından girilmesi

print('Kelimeleri arasında birer boşluk olarak giriniz.')
print('Örneğin: Erdoğan Bahçeli Kılıçdaroğlu')
kelimeler = input()
kelimeler = kelimeler.strip().split()

dt = pd.date_range(start='1/1/1997', end='1/1/2020', freq='M', closed='left')
f = plt.figure()
f.set_size_inches(10, 5)

# Grafik çizimi

for i in range(len(kelimeler)):
    s_w, sum_w = kelime_frekansı(kelimeler[i])
    plt.plot(dt, sum_w[:276][::-1], label=(kelimeler[i] + ' (' + str(s_w) + ')'))
    
plt.xlabel('Tarih', fontsize=14)
plt.ylabel('Haber Sayısı', fontsize=14)
plt.legend(loc='upper right')
plt.show()




