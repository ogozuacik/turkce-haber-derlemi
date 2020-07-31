<a href=""><img src="https://raw.githubusercontent.com/ogozuacik/turkce-haber-derlemi/master/figurler/logo.png" width="250" height="250" align="right" /></a>
# Türkçe Haber Derlemi ve Haberlerin Kategorik Sınıflandırılması

[Açık Kaynak Hackathon Programı 2020](https://www.acikhack.com/) için geliştirilen projede ana hedef &quot;Haber başlıkları üzerinden konu (kategori) tahmini&quot; olarak tanımlanmıştır.
- Milliyet.com üzerinden 1997-2019 yılları arası çıkan haberler kullanılmıştır. Derlem açık kaynak olarak paylaşılmıştır.
- Haber başlıklarına ek olarak haber metni ve özeti üzerinden de konu (kategorinin) tahmin edilmesi sağlanmıştır.
- Sonuçlar üzerine analiz sunulmuştur.

## Türkçe Haber Derleminin Oluşturulması
- Milliyet.com adresinden 1997-2019 yılları arasında çıkan, haberler büyük oranda çekilmiştir.
- Her haber, internet sayfasında bulunduğu gibi kaydedilmiştir.
- Başlık, özet, haber metni, kategori ve link
	- Özet bazı haberlerde bulunmamaktadır.
- Temizlenmemiş hali ile 116.068 örnek bulunmaktadır. (sıkıştırılmamış hali ile 403.9Mb)

## Gerekli Paketler
Deneylerde kök bulma (stemming) işlemi yapılmış fakat sınıflandırma performansına katkısı gözlemlenmemiştir. Bu nedenle, projede TurkishStemmer [1]  paketi kullanılmamaktadır ama kullanıcı isterse pre_processing_tr.py paketindeki turkish_stemmer(s) methodunu kullanarak diledikleri cümlelerdeki kelimeleri köküne ulaşabilmektedir. 
```python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
#from TurkishStemmer import TurkishStemmer 		#opsiyonel
```
## Türkçe Haber Derlemine Erişim
Proje kapsamında 3 farklı derlem paylaşılmıştır.
**milliyet_derlem.csv.gz:** Haberler milliyet.com&apos;dan çekildiği gibi saklanmıştır.

**temizlenmis_derlerm.csv.gz:** Bazı kategorilerde bulunan haberler atılmıştır. Benzer kategorideki haber türleri birleştirilmiştir. Konu hakkında detaylı bilgi için yazının *&quot;Veri Temizliği ve Düzenlenmesi&quot;* kısmını okuyabilirsiniz.

**filtrelenmis_temizlenmis_derlem.csv.gz:** Ön işleme sonucu filtrelenmiş haberleri içerir. Ön işleme adımları hakkında detaylı bilgi için yazının *&quot;Haber Metinlerinde Ön İşleme&quot;* kısmını okuyabilirsiniz.

**derleme_erisim** adlı jupyter notebookta gösterildiği gibi dilediğiniz derlemi ismini vererek açabilirsiniz.
```python
data=pd.read_csv('derlemler/filtrelenmis_temizlenmis_derlem.csv.gz')
```
------------

**Derlem sütunlarının anlamları:**
- **news_title:** haber başlığı
- **summary:** haber özeti
- **category:**  haberin kategorisi (filtrelenmiş)
- **date:** haberin paylaşıldığı gün (bazı haberlerde bu tarihte sapmalar bulunmaktadır)
- **link:** haberin çekildiği bağlantı adresi
- **news:** haber metni
- **category_backup:** haberin orijinal (milliyet sitesinde bulunduğu) kategori

------------

**Filtrelenmiş-temizlenmiş derlemin ilk 5 örneği:**
<a href=""><img src="https://raw.githubusercontent.com/ogozuacik/turkce-haber-derlemi/master/figurler/dataset.png" /></a>

## Haber Metinlerinde Ön İşleme
Haber yazılarının makine öğrenmesi yöntemlerine uygun hale getirilmesi için gereken adımlar:
1.  **Genel yazım hatalarının düzeltilmesi**
	- Verilerde genel olarak bulunan yazım hataları düzeltilmiştir.
	- Örneğin: ... kazandıTürkiye... --> ...kazandı Türkiye... 

2. **Sayıların ve noktalama işaretlerinin yerine boşluk konulması**
	- Bütün sayılar kategori tahmininde etkisiz kalacağı düşünülerek boşluk ile değiştirilmiştir.
	- Noktalama işaretlerinin yerine boşluk getirilmiştir.
	- Doğrudan silmek yerine boşluk konulması haberlerde büyük oranda bulunan virgülden sonra atlanan boşluk hatası sonucu kelimelerin birleşmesini önüne geçmektir.

3. **Fazla boşluklardan kurtulma**
	- Sayıların, noktalama işaretlerinin temizlenmesi sırasında iki kelime arası birden fazla boşluk oluşmuştur. Bu boşluklar tek boşluk olucak şekilde değiştirilnmiştir.

4. **Türkçe dolgu sözcüklerinin (stop words) çıkarılması**
	- Türkçe Dolgu Sözcükleri (stop words) sık kullanılan, fakat iptal metinden çıkarıldıklarında cümlenin anlamında önemli değişiklikler oluşturmayan sözcüklerdir. Necmettin Çarkacı&apos;nın GitHub hesabında paylaşılan [2] dolgu sözcükleri listesine göre vektöre dönüştürülme esnasında metinlerden bu sözcükler çıkarılmıştır.

5. **Kök bulma (stemming)**
	- Projede aktif olarak kullanılmamıştır fakat kullanıcı isterse pre_processing_tr.py paketindeki turkish_stemmer(s) methodunu kullanarak diledikleri cümlelerdeki kelimeleri köküne ulaşabilmektedir.

------------

**Ön işleme adımlarının (1-3), istenen yazıda yapılması için gereken adımlar:**
```python
import pre_processing_tr as pr

yazi = "Ön işlemden geçirmek istenilen yazı"
islenmis_yazi = pr.pre_process(text)
```

------------

**Kök bulma metotunun kullanılması için gereken adımlar**
```python
import pre_processing_tr as pr

yazi = "stem edilmek istenen yazı"
stem_yazi = pr.turkish_stemmer(yazi)
```

## Haber Metinlerinin Sınıflandırılması
- Gündem, Dünya, Cumartesi ve Pazar kategorilerindeki haberler genel konular hakkında olduğu için, Ege kategorisindeki haberler yerel haberler oldukları için analizin dışında tutulmuşlardır.
- 4-kategori ve 5-kategori içeren iki tür veri üzerinden model oluşturulmuştur. Farklı modeller haber başlığı, özeti ve metni üzerine eğitilmiştir. (Örneğin başlık üzerine eğitilen modelde eğitim ve sınıflandırma esnasında sadece haber başlıkları kullanılmıştır) 
- 4-kategori: Ekonomi, Siyaset, Spor, Teknoloji-Bilim
- 5-kategori: Ekonomi, Siyaset, Spor, Teknoloji-Bilim, Diğer (Kültür-Sanat, Magazin, Yaşam)

------------

Kendi istediğiniz haberleri sınıflandırmak için bu iki modelden birini kullanabilirsiniz.
- 4-kategorili sınıflandırma için **4_kategori_tahmin.py**
- 5 kategorili sınıflandırm için **5_kategori_tahmin.py**

Hazır modelleri çalıştırmak için command promp&apos;ta proje dizinine gelinmeli ve **python 5_kategori_tahmin.py** yazarak çalıştırılmalıdır. Daha sonra sınıflandırmak istenen metin girilmelidir.

------------

Kullanım rahatlığı açısından **kategorik_sınıflandırma** adlı jupyter notebook&apos;u kullanmanız tavsiye edilir.

------------

**Sonuçlar**

|   |  Başlık | Özet  | Haber Metni  |
| :------------: | :------------: | :------------: | :------------: |
| 4-kategori  | %71.3 | %82.2  | %85.5   |
|  5-kategori | %66.1  | %79.3  | %80.9  |

**Tablo:** Veri türüne ve kategori sayısına göre sınıflandırma performansları (doğruluk)

Haber metinleri üzerinden oluşturduğumuz modeller daha başarılı olduğu, projede onlar kullanılmıştır.

------------

**Kategorilere göre en çok geçen kelimeler:**
- **Ekonomi:** 'önemli', 'türk', 'dolar', 'yıl', 'yüzde'
- **Siyaset:** 'parti', 'başbakan', 'başkanı', 'erdoğan', 'chp'
- **Spor:** 'takım', 'futbol', 'beşiktaş', 'galatasaray', 'fenerbahçe'
- **Teknoloji_Bilim:** 'akıllı', 'samsung', 'galaxy', 'apple', 'yeni'

## Veri Temizliği ve Düzenlenmesi
- Temizlenmemiş derlemde toplam X kategori bulunmaktadır. Bu kategorilerden bazıları köşe yazarlarıdır. Bazı kategorilerde 10'dan az haber bulunmaktadır.
- Kategorilerin birleştirilmesi ve silinmesi:
	- Futbol, Skorer, Basketbol, Fenerbahçe, Galatasaray, Beşiktaş… gibi spor ile alakalı haberlere genel bir kategori oluşturulup "Spor" adı verillmiştir.
	- Teknoloji_Bilim, Magazin, Yaşam ve Dünya kategorileri de yukarıdaki mantık ile birden fazla kategorinin birleştirilmesi ile oluşturulmuştur.
	- Kategorisi köşe yazarları olan örnekler silinmiştir.
- Haberlerin orijinal (milliyet sitesinde bulundukları) kategori derlemde kategori_yedek sütununda tutulmuştur.

------------

**Filtrelenmiş ve temizlenmiş derlemde bulunan toplam haber sayısı:** 81407

------------

**Haber türüne göre örnek sayıları:**
- **Gündem:** 21031
- **Ekonomi:** 11165
- **Spor:** 10699
- **Siyaset:** 14530
- **Dünya:** 4792
- **Yaşam:** 1696
- **Pazar:** 2090
- **Ege:** 6131
- **Magazin:** 3980
- **Kültür_Sanat:** 1265
- **Teknoloji_Bilim:** 3105
- **Cumartesi:** 923

## Sınıflandırma Modelinin Oluşturulması
- Sınıflandırma işlemi öncesi sklearn kütüphanesi üzerinden CountVectorizer fonksiyonu ile haber metinleri vektöre çevrilmiştir.
	- BoW (Bag of words) yöntemi kullanılmıştır. Vektörde her sütun bir kelimeyi, her satır ise haberde o kelimenin kaç defa geçtiğini temsil etmektedir. Tf–idf yöntemi de denenmiştir fakat performansı BoW’e göre geride kaldığı için analize eklenmemiştir.
	- Vektöre dönüştürülme esnasında metinlerden türkçe dolgu sözcükleri (stop words) *&quot;Haber Metinlerinde Ön İşleme&quot;*  kısmında belirtildiği gibi çıkarılmıştır.

------------

- Model olarak Çokterimli Naive Bayes (Multinomial Naive Bayes) sınıflandırıcı kullanılmıştır.
	- Çeşitli parametreler ile çapraz geçerleme (cross validation) yapılmış ve en uygun (optimize) şekle getirilmiştir. [%70 train (eğitim), %30 test (deney)]
	- SVM, Random Forests, XGBoost, Yapay sinir ağları gibi daha karmaşık algoritmalar da denenmiştir fakat performans olarak Çokterimli Naive Bayes’e göre geride kaldıkları için analize eklenmemişlerdir.

------------

Model oluşturma aşamaları **model_oluşturma** adlı jupyter notebook üzerinden incelenebilir.

## Veri Üzerinde Keşif
- Haberlerin paylaşıldığı tarih bilgisini kullanarak kelimelerin aya ve yıla göre haberlerde bulunma durumlarını inceleyebiliriz. Bu sayede kişilerin, kurumların, vs. 1997-2019 yılları arasında medyada ne kadar yer kapladıkları gözlemlenebilmekte ve bunun üzerine çıkarımlar yapılabilmektedir. 
- Aşağıdaki grafiklerde seçili kelimelerin tarihe göre haberlerde ne kadar bulunduğunu göstermektedir. Lejandda parantez içerisindeki değer yanında bulunduğu kelimenin toplam geçtiği haber sayısıdır.
- Örneğin ilk grafik Erdoğan, Kılıçdaroğlu, Bahçeli ve Baykal kelimeleri ile elde edilen sonuçları içermektedir. 1996-1998 yılları arası Bahçeli ve Baykal hakkında daha çok haber çıkarken Erdoğan&apos;ın haber sayısı onlara göre azdır. O dönemde siyasi olarak ön planda olmayan Kılıçdaroğlu hakkında haber neredeyse hiç yoktur. 1998 sonrası dönemde Erdoğan baskın bir biçimde medyada yer almaktadır.
- Bir diğer göze çarpan detay ise Baykal ve Kılıçdaroğlu ile ilgilidir. 2010 yılında Kılıçdaroğlu Baykal&apos;ın yerine CHP genel başkanlığına seçilmiştir. Öncesinde hakkında neredeyse hiç haber yokken Baykal&apos;ı zamanla geçmiştir. Baykal&apos;ı içeren haberler ise o yıl sonrasında azalmış ve 2013 yılı itibari ile sıfıra yaklaşmıştır.


<a href=""><img src="https://raw.githubusercontent.com/ogozuacik/turkce-haber-derlemi/master/figurler/plot1.png" align="left" /></a>

- Bir diğer analiz ise cumhurbaşkanı ve başbakan kelimeleri üzerinden yapılabilir. Aşağıdaki grafik bu iki kelime ile elde edilen sonuçları içermektedir. 2016 yılına kadar başbakan medyada daha ön planda iken 2016 yılında başkanlık sistemi referendumunun sinyalleri ile cumhurbaşkanı ön plana çıkmaya başlamıştır. 2017 yılında başkanlık sisteminin kabulü ile cumhurbaşkanı ile ilgili yapılan haberler başbakan hakkında yapılan haber sayısını geçmiştir.

<a href=""><img src="https://raw.githubusercontent.com/ogozuacik/turkce-haber-derlemi/master/figurler/plot2.png" align="left" /></a>

- Yukarıda incelenen örnekler gibi birçok durum benzer şekilde analiz edilebilir. 

------------

**haberlerde_kesif** adlı jupyter notebook ile kendi istediğiniz kelimeler ile grafik oluşturabilir ya da örnek grafikleri inceliyebilirsiniz.

------------

Kendi istediğiniz kelimeler ile alternatif olarak grafik oluşturmak için command promp&apos;ta proje dizinine gelinmeli ve **python haberlerde_kesif_cizici.py** yazarak çalıştırılmalıdır. Daha sonra istenilen kelimeler sistem uyarı verdiğinde (veritabanın büyüklüğü sebebi ile biraz bekletmektedir) aralarında bir boşluk olarak girilmelidir.



## Referanslar
[1] Turkish Stemmer, Osman Tunçelli, [link](https://github.com/otuncelli/turkish-stemmer-python "link")

[2] Türkçe Dolgu Sözcükleri, Necmettin Çarkacı, [link](https://github.com/ncarkaci/tr-preprocessing "link")

<a href=""><img src="https://raw.githubusercontent.com/ogozuacik/turkce-haber-derlemi/master/figurler/logolar.png" align="left" /></a>






