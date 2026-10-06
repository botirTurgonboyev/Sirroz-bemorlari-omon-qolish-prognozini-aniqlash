# Sirroz bemorlari omon qolish prognozini aniqlash

Jigar sirrozi bilan og'rigan bemorlarning klinik ko'rsatkichlari asosida har bir bemor uchun uchta natija ehtimolligini bashorat qiladigan model.

| Sinf | Ma'nosi |
|------|---------|
| `C`  | Tadqiqot oxirida tirik (senzuralangan) |
| `CL` | Jigar transplantatsiyasi tufayli senzuralangan (tirik) |
| `D`  | Vafot etgan |

**Metrika:** multi-class log loss (kichikroq bo'lgani yaxshi). Har bir test `id` uchun `Status_C`, `Status_CL`, `Status_D` ehtimolliklari yuboriladi, qatorlar baholashdan oldin yig'indisi 1 ga keltiriladi.

## Natija

Yakuniy model: **5-fold CV log loss = 0.3729** (faqat sinf chastotasini bashorat qiluvchi baseline: 0.7236).

| Model | CV log loss |
|-------|-------------|
| HistGradientBoosting, xom belgilar (3 konfiguratsiya) | 0.3741 - 0.3743 |
| HistGradientBoosting, muhandislik belgilari (3 konfiguratsiya) | 0.3755 - 0.3767 |
| ExtraTrees | 0.4124 |
| LogisticRegression | 0.4434 |
| Teng og'irlikli ansambl | 0.3774 |
| **Optimal og'irlikli ansambl (yakuniy)** | **0.3729** |

Ansambl og'irliklari bir xil OOF bashoratlarida tanlangan, shuning uchun 0.3729 haqiqiy test natijasidan biroz optimistik bo'lishi mumkin. Haqiqiy test natijasi farq qilishi mumkin.

## Fayllar

| Fayl | Vazifasi |
|------|----------|
| `train.csv`, `test.csv`, `sample_submission.csv` | Kirish ma'lumotlari |
| `Sirroz_bemorlari_omon_qolish_prognozini_aniqlash_final.ipynb` | Google Colab daftari: yuklash, EDA, kodlash, vizualizatsiya, HGB + XGBoost + ExtraTrees ansambli, `submission.csv` (XGBoost'siz sandbox'da sinalgan, CV 0.3730) |
| `train.py` | Belgilar yaratish, HistGradientBoosting konfiguratsiyalari va CV funksiyalari |
| `final.py` | Yakuniy ansambl: modellar, og'irliklar, kalibratsiya, `submission.csv` |
| `clean_data.py` | Ma'lumotni tozalash tajribasi (natijani yaxshilamadi, quyida) |
| `ablation.py` | Belgilar to'plamlarini solishtirish tajribasi |
| `submission.csv` | Yakuniy bashoratlar (`id, Status_C, Status_CL, Status_D`) |
| `metrics.json` | Yakuniy CV natijalari va ansambl og'irliklari |

## Ishga tushirish

```bash
pip install pandas numpy scikit-learn scipy
export DATA_DIR=/yo/ma'lumotlar/papkasi   # train.csv, test.csv, sample_submission.csv shu yerda
export OUT_DIR=.
python final.py
```

`final.py` `train.py` bilan bir papkada bo'lishi kerak. Ishlash vaqti taxminan 8 daqiqa, oraliq bashoratlar `cache/` papkasiga saqlanadi (qayta ishga tushirish tez).

## Ma'lumotlar haqida topilmalar

- **Sinf nomutanosibligi:** `C` ≈ 67%, `D` ≈ 30%, `CL` ≈ 2.5%. `CL` eng kam va bashorat qilish eng qiyin.
- **`Y` sinfi:** train'da faqat 1 ta qator `Status = Y`. U `sample_submission.csv` da yo'q, shuning uchun olib tashlandi.
- **Blok ko'rinishidagi yo'qolgan qiymatlar:** qatorlarning ≈ 43-45% ida `Drug`, `Ascites`, `Hepatomegaly`, `Spiders`, `Copper`, `Alk_Phos`, `SGOT`, `Cholesterol`, `Tryglicerides` birga bo'sh (test'da ham xuddi shunday ulush). Bu qatorlarda `D` ulushi 31%, to'liq qatorlarda 30%, ya'ni bo'shliqning o'zi kuchli belgi emas.
- **Xato qiymatlar:** `Age` kunlarda berilgan va 15 qatorda 18 yoshdan kichik yoki 90 dan katta (1.1 yoshdan 354 yoshgacha); 26 qatorda `N_Days` > 6000 (max 38 320); 26 qatorda `Age` va `N_Days` aynan teng (bir ustun ikkinchisiga ko'chirilgan). Train'da 6 ta aynan bir xil qator bor.
- **Eng kuchli belgilar (Spearman, `D` bilan):** `Bilirubin` +0.50, `Prothrombin` +0.45, `Hepatomegaly` +0.43, `Copper` +0.41, `Stage` +0.39, `SGOT` +0.36, `N_Days` -0.35. `Drug` -0.04, deyarli bog'liq emas.
- **`CL` signali kuchsiz, yo'nalishi boshqacha:** `Copper` +0.12, `Age` -0.12 (yoshroq bemorlar ko'proq transplantatsiyaga yuboriladi), `Bilirubin` +0.10.
- **`D` ulushi:** `Stage` 1 da 7%, `Stage` 4 da 58%; `Edema = Y` da 99%; `Ascites = Y` da 97%.
- Korrelyatsiya bo'sh bo'lmagan qatorlarda hisoblangan (`Copper`, `Hepatomegaly` va boshqalar uchun ≈ 57% qator).

## Ish ketma-ketligi (Colab)

1. Ma'lumotni yuklash, `info()`, `describe()`, `isnull().sum()`.
2. Kategorik ustunlarni ajratish (`Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`; `Status` nishon).
3. Qo'lda `map` bilan kodlash. `NaN` saqlanadi, `Edema` uchun `N`=0, `S`=1, `Y`=2 tartibi.
4. Vizualizatsiya: sinf taqsimoti, kategorik belgilar va `Status` (100% stacked bar), raqamli belgilar (box plot, logarifmik shkala), Spearman korrelyatsiyasi.
5. Modellash: gradient boosting, `Y` sinfisiz, barcha qatorlar saqlangan, bo'sh qiymatlar modelning o'ziga topshirilgan.
6. Ansambl va submission.

## Nima ishlamadi (va nima uchun)

| Urinish | Natija |
|---------|--------|
| Xato qiymatlarni `NaN` qilish, dublikatlarni o'chirish (`clean_data.py`) | 0.3752 → 0.3758, yaxshilamadi. Anomaliyalar atigi 41 qator, daraxt modellari ularga chidamli |
| `Drug`, `Cholesterol`, `Tryglicerides` ni tashlash | Ikkala belgi to'plamida ham ≈ 0.0015 yomonlashdi |
| Muhandislik belgilari (nisbatlar, log) | Xom belgilardan biroz yomon (0.3833 va 0.3813, bitta konfiguratsiya, 2 seed) |
| `dropna` (bo'sh qatorlarni tashlash) | Train'ning 46% i yo'qoladi, bo'sh qatorlarda log loss 0.39 dan 0.51 ga yomonlashadi |
| `id` ni belgi sifatida berish | 0.3723 → 0.3751, yomonlashdi |
| Harorat kalibratsiyasi | Yutuq yo'q (T = 1.0) |
| Kesish chegarasi 1e-15 / 1e-6 / 1e-4 / 1e-3 | Farq ≤ 0.0001, 1e-4 tanlandi |

### `dropna` tajribasi tafsiloti

`HistGradientBoosting` bilan, to'liq qatorlar (8162 ta, 54.4%) va blok qatorlar (6837 ta) bo'yicha alohida:

| O'qitish | To'liq qatorlarda | Blok qatorlarda |
|---|---|---|
| Hamma qatorlarda | 0.3674 | 0.3947 |
| Faqat to'liq qatorlarda | 0.3723 | 0.5067 |

Test'ning ≈ 44% qatorida shu ustunlar bo'sh, shuning uchun bo'sh qatorlarni ko'rmagan model ularda sezilarli yomon ishlaydi.

## XGBoost haqida (Colab katagi)

Colab'da `XGBClassifier` ishlatilgan bo'lsa, quyidagilarga rioya qiling:

- `Status` ni `LabelEncoder` bilan raqamga o'tkazing (`C`=0, `CL`=1, `D`=2), XGBoost matnli nishonni qabul qilmaydi.
- `id` ni belgilardan chiqaring va `Y` qatorini olib tashlang.
- `n_estimators` ni qat'iy emas, `early_stopping_rounds` bilan tanlang (kichik `learning_rate`, `max_depth=3`).
- Bo'sh qatorlarni tashlamang, XGBoost `NaN` ni o'zi qayta ishlaydi.
- `MinMaxScaler` daraxt modellari uchun kerak emas.

Bu qadamlar sandbox'da sinalmagan (XGBoost bu muhitda o'rnatilmadi), yakuniy `submission.csv` scikit-learn modellari bilan olingan.

## Kod bo'yicha batafsil tushuntirish

### 1. `train.py`: belgilar yaratish (`make_features`)

```python
X["n_missing"] = X.isna().sum(axis=1)
X["block_missing"] = (X["n_missing"] >= 8).astype(int)
```

- Har bir qatorda nechta qiymat yo'qligini sanaydi.
- 8 va undan ko'p bo'sh qiymat bo'lsa, qator "blok" qator deb belgilanadi.

```python
X["Age_years"] = (X["Age"] / 365.25).clip(18, 90)
X["N_Days_clip"] = X["N_Days"].clip(upper=X["N_Days"].quantile(0.995))
```

- Yoshni kundan yilga o'giradi va 18-90 oralig'iga kesadi.
- `N_Days` ning yuqori 0.5% qismini 99.5-persentil qiymatiga tenglashtiradi.

```python
X["bili_per_albumin"] = X["Bilirubin"] / X["Albumin"]
X["sgot_per_platelets"] = X["SGOT"] / X["Platelets"]
```

- Bir nechta laboratoriya ko'rsatkichini bitta belgida birlashtiruvchi nisbatlar. Yakuniy ansamblda bu belgilar to'plami xom belgilardan kam og'irlik oldi.

```python
for c in CAT_COLS:
    X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
```

- Matnli kategoriyalarni 0, 1, 2... kodlarga o'giradi, `cat.codes` bo'sh qiymatni `-1` qiladi, uni yana `NaN` ga qaytaradi.

### 2. `train.py`: modellar (`CONFIGS`, `run_config`)

```python
model = HistGradientBoostingClassifier(
    categorical_features=cat_mask, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=60, random_state=seed, **params)
```

- `categorical_features` kategorik ustunlarni maxsus qayta ishlaydi, `early_stopping` ichki validatsiyada yaxshilanish to'xtasa o'qitishni to'xtatadi, daraxtlar sonini qo'lda tanlash shart emas.

```python
oof[va_idx] += model.predict_proba(X.iloc[va_idx]) / len(SEEDS)
test_pred += model.predict_proba(X_te) / (len(SEEDS) * N_SPLITS)
```

- `oof` (out-of-fold) har bir qator uchun o'qitishda ko'rilmagan model bashorati, shuning uchun CV halol.
- Test bashorati 3 seed × 5 fold = 15 ta modelning o'rtachasi.

### 3. `final.py`: modellar to'plami

```python
for tag, cols in [("eng", list(X.columns)), ("raw", RAW)]:
    for cname, params in T.CONFIGS.items():
        ...
```

- Bir xil 3 konfiguratsiya ikki belgi to'plamida (muhandislik va xom 18 ustun) o'qitiladi, jami 6 ta HistGradientBoosting.

```python
oofs["extratrees"], tests["extratrees"] = cached("extratrees", sk_model(lambda: make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    ExtraTreesClassifier(...))))
```

- ExtraTrees `NaN` ni qabul qilmaydi, shuning uchun `SimpleImputer` median bilan to'ldiradi, `add_indicator=True` "bu qiymat bo'sh edi" ustunlarini qo'shadi (bo'shliq blok tuzilishiga ega bo'lgani uchun).
- LogisticRegression uchun qo'shimcha `QuantileTransformer` o'ng tomonga cho'zilgan ustunlarni (`Bilirubin`, `Copper`) normal taqsimotga keltiradi.

```python
def cached(name, fn): ...
```

- Har bir modelning OOF va test bashoratini `cache/` ga saqlaydi. Skript qayta ishga tushsa, o'qitilgan modellar qayta hisoblanmaydi.

### 4. `final.py`: og'irliklar va yakuniy bashorat

```python
res = minimize(lambda w: log_loss(y, mix(w, oofs)), np.ones(k) / k, method="SLSQP",
               bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
```

- Har bir modelga 0 dan 1 gacha og'irlik beradi (yig'indisi 1), shunday qilib OOF log loss eng kichik bo'ladi. Natija: xom belgilardagi 3 ta HGB (jami ≈ 0.86) va muhandislik belgilaridagi `hgb_shallow` (0.11) asosiy hissa qo'shdi, ExtraTrees kichik (0.03), LogisticRegression va qolgan ikkita muhandislik HGB nol.

```python
def temper(P, T_):
    Q = P ** (1.0 / T_); return Q / Q.sum(1, keepdims=True)
```

- Harorat kalibratsiyasi: ehtimolliklarni yumshatadi (`T` > 1) yoki keskinlashtiradi (`T` < 1). OOF bo'yicha foyda bermagani uchun qo'llanmadi (T = 1.0).

```python
final = np.clip(P_te, CLIP, 1 - CLIP); final /= final.sum(1, keepdims=True)
```

- Ishonchli, lekin xato bashoratlar log loss'ni keskin oshirmasligi uchun `[1e-4, 1 - 1e-4]` oralig'iga kesadi va qatorlarni qayta normallaydi.

```python
assert submission["id"].tolist() == sub["id"].tolist()
assert np.allclose(submission.iloc[:, 1:].sum(axis=1), 1)
```

- Format tekshiruvlari: `id` tartibi `sample_submission.csv` bilan bir xil, har bir qatorda ehtimolliklar yig'indisi 1.

## Cheklovlar va keyingi qadamlar

- LightGBM, CatBoost va XGBoost bu muhitda o'rnatilmadi (PyPI bloklangan). Ularni ansamblga qo'shish, ehtimol, log loss'ni yana biroz yaxshilaydi.
- Gipersozlash (Optuna kabi) qilinmagan, konfiguratsiyalar qo'lda tanlangan.
- Ansambl og'irliklari OOF bashoratlarda tanlangan (5 ta kichik hissa), shuning uchun CV bahosi biroz optimistik.
- `CL` sinfi (≈ 2.5%) uchun maxsus ishlov (masalan, sinf og'irliklari) sinab ko'rilmagan.
- Haqiqiy test natijasi lokal CV'dan farq qilishi mumkin.
