# 🛣️ Road Image Detection

تطبيق Streamlit للكشف عن أضرار الطرق (شروخ، حفر، تشققات...) باستخدام موديل YOLO مدرّب مسبقًا.

## الملفات المطلوبة في نفس المجلد
```
road-detection-app/
├── app.py              # كود التطبيق
├── best_model.pt        # أوزان الموديل المدرب
├── classes.json          # أسماء الكلاسات (7 أنواع أضرار)
├── requirements.txt      # المكتبات المطلوبة
└── README.md
```

---

## خطوات الرفع من على الموقع (بدون ترمينال)

مشكلة رفع الملفات من متصفح GitHub مباشرة (drag & drop) إنه بيرفض أي ملف أكبر من 25 ميجا،
وموديلك (`best_model.pt`) حجمه ~44 ميجا. الحل: نرفع كل الملفات العادية على الريبو عادي،
ونرفع الموديل بس عن طريق **GitHub Releases** اللي بتسمح بملفات لحد 2 جيجا من نفس الموقع.

### 1) إنشاء الريبو
1. افتح [github.com](https://github.com) وسجل دخول.
2. اضغط **New repository**.
3. اسم الريبو مثلاً `Road-Image-Detection`، خليه **Public**، واضغط **Create repository**.

### 2) رفع الملفات الصغيرة (app.py, classes.json, requirements.txt, README.md)
1. في صفحة الريبو، اضغط **Add file → Upload files** (أو استخدم "choose your files").
2. اسحب أو اختار: `app.py`, `classes.json`, `requirements.txt`, `README.md`.
   **متسحبش `best_model.pt` هنا** لأنه هيرفض الرفع بسبب حجمه.
3. اكتب رسالة commit واضغط **Commit changes**.

### 3) رفع الموديل عن طريق Release
1. في صفحة الريبو، من الشريط الجانبي اليمين، دوس على **Releases** (لو مش ظاهرة، روح لـ tab باسم "Releases" فوق قرب "Code").
2. اضغط **Create a new release** (أو **Draft a new release**).
3. في **Tag**، اكتب حاجة زي `v1.0` واضغط "Create new tag".
4. في **Release title** اكتب أي اسم زي `Model weights v1.0`.
5. تحت في خانة **Attach binaries**، اسحب أو اختار ملف `best_model.pt` (الرفع هنا بيسمح لحد 2 جيجا).
6. اضغط **Publish release**.
7. بعد النشر، دوس بزرار الماوس اليمين على اسم الملف `best_model.pt` في قائمة الـ assets واختار **Copy link address** — هيديك رابط تحميل مباشر شبه:
   `https://github.com/USERNAME/Road-Image-Detection/releases/download/v1.0/best_model.pt`

### 4) ربط الرابط بالتطبيق
1. افتح `app.py` في نفس الريبو (اضغط على الملف ثم **Edit** ✏️).
2. لاقي السطر ده فوق:
   ```python
   MODEL_URL = ""
   ```
3. الصق الرابط اللي نسخته جوا الأقواس:
   ```python
   MODEL_URL = "https://github.com/USERNAME/Road-Image-Detection/releases/download/v1.0/best_model.pt"
   ```
4. اضغط **Commit changes** لحفظ التعديل.

بكده التطبيق أول ما يشتغل هيلاقي إن `best_model.pt` مش موجود جنبه، وهيحمله تلقائي من رابط الـ Release ده.

### 4) النشر على Streamlit Community Cloud
1. روح على [share.streamlit.io](https://share.streamlit.io) وسجل دخول بحساب GitHub بتاعك.
2. اضغط **New app**.
3. اختار الريبو `road-image-detection` والـ branch `main`.
4. في خانة **Main file path** اكتب `app.py`.
5. اضغط **Deploy**.
6. استنى شوية لحد ما يثبت المكتبات من `requirements.txt` ويشغل التطبيق — هتاخد رابط زي:
   `https://road-image-detection-xxxx.streamlit.app`

### 5) تجربة محلية قبل الرفع (اختياري)
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## مميزات التطبيق
- رفع صورة واحدة أو أكتر مرة واحدة.
- تحكم في **Confidence threshold** و **IoU threshold** من الشريط الجانبي.
- فلترة أنواع الأضرار اللي عايز تظهرها بس.
- ألوان مختلفة لكل نوع ضرر مع legend واضح.
- عرض الصورة الأصلية جنب الصورة بعد الكشف.
- جدول تفاصيل كل اكتشاف (الكلاس، نسبة الثقة، الإحداثيات).
- زرار تحميل الصورة بعد الرسم عليها.
- ملخص إحصائي (عدد الاكتشافات، متوسط الوقت لكل صورة) وشارت لتوزيع الأضرار.
- تبويب "About" بيشرح كل نوع من أنواع الأضرار السبعة.

## أنواع الأضرار المدعومة
alligator, block, crack, edge, longitudinal, pothole, transverse
