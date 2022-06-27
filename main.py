# Kullanacağımız modül ve kütüphaneler
import pandas as pd
import numpy as np
import seaborn as sns
# Verilerimizi istediğimiz ilgili veri setinden okuyalım
nlf_data = pd.read_csv("../input/car-sales/Car_sales.csv")

#Tekrar işlemleri için (np.random.seed(0)) kodunu ekliyoruz.
#Sözde rasgele sayı üretecini başlatmak için random.seed() kullanılır.
#Bir sözde rasgele sayı üreteci tarafından verilen verileri yeniden üretmek için tohum ayarlanır
#Her seferinde random.seed() kullanarak listeden aynı öğeleri rastgele seçilir
np.random.seed(0) 
#İlk olarak elimdeki veri setimizin ne olduğunu bakıyoruz, yani nasıl bir veri setimiz var öncelikle inceliyoruz. Bu işlem bize veri setimizin doğru birşekilde #yüklenip yüklenmediği bilgisini verecektir. Şimdi elimdeki veri setinde eksik (NaN yada None) olan değerleri kontrol ediyoruz.

nlf_data


# Veri setimizin (nlf_data) rastgele 5 satırına bakarak eksik değerleri kontrol edebiliriz.
nlf_data.sample(5)

#Şimdi veri setimizdeki eksik veri sayısına bakalım

#Evet, şimdi kaç tane eksik verimiz olduğuna bakalım ve her satırdaki eksi veri sayılarını tesbit edelim.

#Her kolonda kaç tane eksik değer var onlara bakalım
missing_values_count = nlf_data.isnull().sum()
#ilk 10 sütundaki eksik değerlerin kaç tane olduğuna bakalım
missing_values_count[0:10]

#Sütunlarda eksik veriler mevcut. Problemimizin çözümünde bize doğru verilerin hassaslığı konusunda bu bilgiler bize yardımcı olacaktır.

#Toplam kaç tane eksik verimiz var ona bakalım
total_cells = np.product(nlf_data.shape)
total_missing = missing_values_count.sum()
total_missing

#51 adet değer eksik görünüyor. Bir sonraki adımımıza bu verilerin % ne kadar yaptığına bakıcaz.

#Eksik verilerin yüzde(%) olarak ne kadar olduğuna bakalım
(total_missing/total_cells) * 100

#Yaklaşık % 2 eksik veri var. Bir sonraki adımda eksik verilere daha yakından bakacağız ve ndene kaynaklandığını bulmaya çalışacağız.

#Veriler neden eksik?

#Veri setini alıp direkt analizler yapabiliriz, bu bize bazı sonuçlarda verecektir. Ancak bu sonuçlar gerçeği ne kadar yansıtıyor ? Verilerin neden olmadığı konusu #bir veri setini incelerken ilk dikkat etmemeiz gereken konudur. Eksik veriyi direk de silebilirdik. Fakat bunları hemen silersek bu eksik değerlerin neden oluştuğu #ve kendinden başka neleri etkilediğini görememize sebep olacaktir. Bunun için ilk olarak yukarda gördüğünüz üzere veri setindeki eksik verileri tespit ettik.

###Şimdi kendimize sormamız gereken soru şu: Bu veriler eksik peki neden ? Acaba veriler kayıt edilmemiş mi yoksa bu veriler üretilmemiş mi ?

#1- Eğer eksik olan veriler gerçekten üretilmetiği için veri setinde yoksa " Mesela erkeklerin hamile olma durumu" gibi bir durumu inceleyen bir veri olabilir.Ve bu #mümkün olmadığı için bu hamilelik durumu erkeklerde NaN(Not a Number- Uygun olmayan değerler) olarak gelecektir.

#2- Bir diğer durum ise verilerin kayıt olmama durumudur. Burada yapılması gereken işlem diğer verilere bakarak yaklaşık olarak ne olabileceği hakkında yorum #yapmaktır. Bu durumda en çok kullanılan değer(string değerler için) ya da ortalama bir değer atayarak(nümerik değerler için) bu değerleri doldurabiliriz.

#şimdi ilk 10 satırdaki eksik değerlere bakalım

#Örneğin 'year_resale_value' da 36 değer ekik görünüyor.Bu değer ıl satış değerini gösterir. Bu değerin girilmemesi o değerin var kayıt edilmediği anlamına gelmiyor. #Çünkü yıl içinde satışı yapılmamış olabililir. Bu sütun için, ya boş bırakmak ya da "NA" gibi üçüncü bir değer eklemek daha mantıklı olacaktır Ve daha sonra bu #alanları NA'ları değiştirmek gerekmektedir. Bunun gibi diğer değerleri de incelemeliyiz.

#Eğer çok ciddi bir veri seti üzerinde çalışıyorsanız, bu durumlara çok dikkat etmemiz gerekiyor, veri setinin orjinaline bakarak eksik değerlerin neler ekleneceği #ve #nelerin yok sayılacağı hakkında iyi bir analiz yapmamız gerekmektedir, çünkü hassas hesaplamalarda her değer çok önemlidir bunun için en uygun veriler ile anliz #yapmamız gerekmektedir.

#Evet şimdi sıra veri setindeki eksik (NaN) değerlerini çıkarıyoruz.
#Eğer işleminiz acele ve eksik verilerin olup olmaması durumu çok önemli değil ise(ki hassas bir analiz yapmıyorsak) veri setimizin sonuçlarını etkilememesi için satır #ve sütunlarda bulunan eksik verileri (NaN-Not a Number) tablomuzdan çıkarabiliriz. Eğer bu durumdan eminsek yani eksik verilerin analizimizi etkilemeyeceğini #biliyorsak verileri tablodan çıkarmak için Python kütüphanelerinden 'pandas' kütüp hanesi bize bu konuda yadımcı olacak ve 'dropna()' (eksik verileri veri setinden #çıkar) komutunu kullanacağız.

#Evet şimdi tüm satırlardaki eksik(NaN) değerlerini siliyoruz
nlf_data.dropna()

#NaN içeren bütün satırlar yok oldu ve 157 satırdan 149 a düştü, acaba hepsi eksik(NaN) değerleri ile mi dolu idi ? Satırlarda en az bir tane eksik(NaN) değer mevcut olan satırları sildik. Şimdi bu durumu satırlar için değil de sütünlar için yapalım.

#Bir tane bile eksik değer olan tüm kolonları silelim
colomns_with_na_dropped = nlf_data.dropna(axis=1)
colomns_with_na_dropped.head()

#Şimdi bakalım ne kadar değer kayıbımız var :)
print("Orijinal veri setinin sütunları : %d \n" % nlf_data.shape[1])
print("Eksik verileri çıkarıldıktan sonraki sütun sayısı : %d" % colomns_with_na_dropped.shape[1])


#Eksik verileri çıkarıldıktan sonraki sütun sayısı : 5
#Görüldüğü üzere verilerimizin bütük kısmı silindi.Artık veri setimiz içirisinde hiç bir sütünda eksik değer kalmadı.Tab bunu kalıcı olarak yapmadık. (Yapmak istesek #nlf_data.drop(nlf_data, axis=1, inplace=True) yapmamız ya da yeni değeri nlf_data'a atamamız gerekecekti.Veri seti ile çalışırken bu kısma çok dikkat etmeniz #gerekemektedir.)

#Sıradaki aşamada boş değerler yerine otomatik olarak değerler ekleyeceğiz.
#Bu aşamada boş değerler yerine otomatik olarak değerler ekleyeceğiz. Ve bundan sonraki kısımlarda nlf_data veri setinden bazı örnekler alarak işlemlerimizi #gerçekleştireceğiz.

#NFL veri setinden bir alt küme veri seti alıyoruz
subset_nlf_data = nlf_data.head()
subset_nlf_data

#Burada'Pandas'ın 'fillna()' fonksiyonunu kullanarak veri setinde eksik olan veriler yerine yeni veriler ekleyeceğiz. Önce boş 'NaN' olan değerler yerine sıfır '0' #yazacağız.

#Tablodaki NaN değerlerinin yerine hepsine sıfır'0' yazıyoruz.
subset_nlf_data.fillna(0)

#Aynı zamanda daha fazla bilgi sahibi olabilirim ve aynı değerden sonra gelen değerlerle birlikte eksik değerleri alabilirim. (Bu, gözlemlerin bir çeşit mantıksal #düzenleri olduğu veri kümeleri için çok anlam ifade eder.)

#Aynı sütunlardaki tüm NaN değerlerini  sıfır(0) ile değiştiriyoruz 

#Aynı zamanda 0 ile doldurmak yerine en çok kullanılan değer ya da ortalama değeri ile de doldurabilirdik.Şimdi onu da nasıl yapabiliriz bakalım

#NFL veri setinden bir alt küme veri seti alıyoruz
subset_nlf_data = nlf_data.head()
subset_nlf_data

#şimdi tekrar eksik olan sütun değerlerime bakmak istiyorum 
missing_values_count[0:10]

#şimdi sadece __year_resale_value değeri sütununda ortalama değeri bulacağım
sbunset_ort=subset_nlf_data["__year_resale_value"].mean()
sbunset_ort
21.288
#Tablodaki NaN değerlerinin yerine hepsine sıfır'0' yazıyoruz.
subset_nlf_data.fillna(sbunset_ort)
