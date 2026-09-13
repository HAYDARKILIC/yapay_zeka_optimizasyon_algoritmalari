# Yapay Zeka Optimizasyon Algoritmaları

**Sıfırdan implementasyonla ders notebook'ları**

Yapay Zeka Mühendisliği öğrencileri için beş temel optimizasyon algoritmasını
matematiksel türetimden başlayarak NumPy ile sıfırdan inşa eden bir ders serisi.

```
SGD + Momentum  ·  RMSProp  ·  Adam  ·  AdamW  ·  L-BFGS
```

---

## Hızlı başlangıç

```bash
git clone <repo-url>
cd yapay_zeka_optimizasyon_algoritmalari
pip install numpy scipy scikit-learn matplotlib jupyterlab
jupyter lab
```

**Kurulum adımı yok.** Her notebook kendi kendine yeter: ilk hücre, ders boyunca
kullanılan `optim` paketini çalışma dizinine yazar ve içe aktarır. Bir kez
çalıştırıp geçmeniz yeterli.

Google Colab'de de doğrudan açıp çalıştırabilirsiniz — ek kurulum gerekmez.

> Notebook'ları çalıştırdığınızda yanlarında bir `optim/` klasörü oluşur.
> Bu klasör ilk hücre tarafından üretilir; silseniz bile yeniden yazılır.
> `.gitignore` içinde olduğu için depoya girmez.

---

## Ders içeriği

Notebook'lar sırayla okunacak şekilde tasarlanmıştır; her biri bir öncekinin
üzerine kurulur.

| # | Notebook | Konu |
|---|---|---|
| 00 | `00_giris_ve_temeller.ipynb` | Gradyan, en dik iniş, kararlılık sınırı $\eta<2/L$, koşul sayısı, eyer noktaları, mini-batch gürültüsü, gradyan doğrulama |
| 01 | `01_sgd_momentum.ipynb` | SGD, gürültü tabanı, momentum, etkin adım $\eta/(1-\mu)$, Nesterov, lr programları, doğrusal ölçekleme kuralı |
| 02 | `02_adagrad_rmsprop.ipynb` | AdaGrad, $1/\sqrt{t}$ sönmesi, seyrek veri, RMSProp, $\alpha$ ve $\varepsilon$, centered varyant |
| 03 | `03_adam.ipynb` | Momentum + RMSProp, bias düzeltmesi, ölçek bağımsızlığı, $\beta$ taramaları, warmup, AMSGrad, bellek maliyeti |
| 04 | `04_adamw.ipynb` | L2 ile ağırlık sönümü farkı, Adam'daki bozukluk, ayrıştırma, $\lambda$ taraması, parametre grupları |
| 05 | `05_lbfgs.ipynb` | Newton, sekant denklemi, BFGS, iki döngülü özyineleme, güçlü Wolfe çizgi araması, mini-batch çöküşü |
| 06 | `06_karsilastirma_ve_proje.ipynb` | Adil karşılaştırma protokolü, çoklu tohum, dayanıklılık analizi, karar rehberi, dönem projesi |

Her notebook şu yapıyı izler:
**teori → sıfırdan implementasyon → referansla doğrulama → görselleştirme →
gerçek veri deneyi → tuzaklar → alıştırmalar**

---

## Ders felsefesi

**1. From scratch.** Otomatik türev (autograd) kullanılmaz. Tüm gradyanlar elle
hesaplanmış ve merkezi sonlu farkla doğrulanmıştır — MLP'nin geri yayılımı dahil.

**2. Her iddia ölçülür.** "Momentum hızlandırır" demek yetmez; kaç adımda, hangi
koşul sayısında, hangi öğrenme oranında olduğunu ölçeriz.

**3. Beklentiyle uyuşmayan sonuç saklanmaz.** Warmup'ın fayda sağlamadığı,
AdamW'nin her $\lambda$ değerinde kazanmadığı ve doğrusal ölçekleme kuralının
çöktüğü durumlar olduğu gibi raporlanır ve nedeni tartışılır.

Notebook'lardaki **tüm kod blokları çalıştırılmıştır**; altlarındaki çıktılar
gerçek koşum sonuçlarıdır.

---

## `optim` paketi

İlk hücrenin yazdığı paketin içeriği:

| Modül | İçerik |
|---|---|
| `base.py` | Optimizer arayüzü, eğitim döngüsü, mini-batch üreteci |
| `sgd.py` | SGD, Momentum, Nesterov, weight decay |
| `rmsprop.py` | AdaGrad, RMSProp (momentum ve centered varyantlarıyla) |
| `adam.py` | Adam + AMSGrad |
| `adamw.py` | AdamW + parametre gruplu sürüm |
| `lbfgs.py` | Güçlü Wolfe çizgi araması, iki döngülü özyineleme, `LBFGS` sınıfı |
| `functions.py` | Test fonksiyonları (Rosenbrock, Beale, Rastrigin…) + gradyan doğrulama |
| `models.py` | LinearRegression, LogisticRegression, MLP (elle geri yayılım) |
| `data.py` | Veri üreteçleri + öğrenme oranı programları |
| `viz.py` | Eş yükselti, yörünge ve kayıp eğrisi çizimleri |

Tüm optimizer'lar PyTorch'un tasarım desenini taklit eder:

```python
from optim import AdamW, MLP, train
from optim.data import load_digits_data

X_tr, X_te, y_tr, y_te = load_digits_data(seed=0)

model = MLP([64, 128, 64, 10], activation="relu", seed=42)
opt = AdamW(model.params, lr=3e-3, weight_decay=0.01)

hist = train(model, X_tr, y_tr, opt, n_epochs=40, batch_size=64,
             X_val=X_te, y_val=y_te)
```

Model sözleşmesi `loss, grads = model.loss_and_grads(X, y)`, optimizer
sözleşmesi `opt.step(grads)`. Parametreler **yerinde** güncellenir; bu sayede
beş optimizer'ı tek satır değiştirerek karşılaştırabilirsiniz.

---

## Gereksinimler

Python 3.10+ ve şunlar:

```
numpy  scipy  scikit-learn  matplotlib  jupyterlab
```

**İnternet bağlantısı gerekmez** — tüm veri setleri scikit-learn ile gelir veya
sentetik üretilir.

---

## Alıştırmalar ve proje

Her notebook 6 alıştırmayla biter (2 kolay, 2 orta, 2 zor).

Dönem projesi 06 numaralı notebook'ta tanımlıdır: listeden bir optimizer seçip
(NAdam, AdaDelta, RAdam, Lion, Adafactor, LAMB, Shampoo) sıfırdan yazmak, test
etmek ve adil protokolle karşılaştırmak. Değerlendirmenin %20'si **sonuçların
dürüst yorumlanmasına** ayrılmıştır — seçtiğiniz yöntemin karşılaştırmada
kaybetmesi not kaybı değildir.

---

## Lisans

MIT. Eğitim amaçlı serbestçe kullanılabilir, uyarlanabilir ve dağıtılabilir.
