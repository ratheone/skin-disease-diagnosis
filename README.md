# ==========================================
# CİLT HASTALIKLARI SINIFLANDIRMA PROJESİ
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from google.colab import files

# --- 1. KAGGLE AYARLARI ---
# Kaggle API anahtarınızı (kaggle.json) bir kez yüklemeniz yeterlidir.
if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("Lütfen Kaggle API anahtarınızı (kaggle.json) yükleyin...")
    uploaded = files.upload()
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

# --- 2. VERİ SETİNİ İNDİRME ---
if not os.path.exists("dermnet.zip"):
    print("Veri seti indiriliyor (DermNet)... Lütfen bekleyin, bu işlem birkaç dakika sürebilir.")
    !kaggle datasets download -d shubhamgoel27/dermnet
    !unzip -q dermnet.zip
    print("Veri seti hazır.")

# --- 3. VERİ HAZIRLAMA ---
TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Eğitim verileri için artırma (Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Test verileri için sadece ölçeklendirme
test_datagen = ImageDataGenerator(rescale=1./255)

print("\nVeriler yükleniyor...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
print(f"Toplam Sınıf Sayısı: {len(class_names)}")

# --- 4. MODEL OLUŞTURMA (Transfer Learning - MobileNetV2) ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Önceden eğitilmiş ağırlıkları dondur

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5. MODELİ EĞİTME ---
# Not: Eğitim süresini kısaltmak için 5 epoch ayarlandı. Daha iyi sonuç için 15-20 yapabilirsiniz.
print("\nEğitim başlıyor (Bu işlem zaman alabilir)...")
history = model.fit(train_generator, epochs=5, validation_data=test_generator)

# --- 6. SONSUZ DÖNGÜDE TAHMİN VE GÖRSELLEŞTİRME ---
def predict_and_show_loop():
    print("\n" + "="*50)
    print("ANALİZ MODÜLÜ AKTİF")
    print("="*50)
    
    while True:
        print("\nTahmin etmek istediğiniz cilt hastalığı fotoğrafını seçin.")
        print("(İptal etmek için yükleme penceresini kapatın veya 'Cancel'a basın)")
        
        uploaded_img = files.upload()

        # Eğer kullanıcı dosya seçmeden iptal ederse (boş sözlük dönerse) döngüden çık
        if not uploaded_img or len(uploaded_img) == 0:
            print("\nAnaliz işlemi kullanıcı tarafından sonlandırıldı.")
            break

        for fn in uploaded_img.keys():
            # Resmi yükle ve modele uygun hale getir
            img_path = fn
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Tahmin yap
            preds = model.predict(img_array, verbose=0)
            score = np.max(preds)
            class_idx = np.argmax(preds)
            result = class_names[class_idx]

            # Sonucu görselleştir
            plt.figure(figsize=(6, 4))
            plt.imshow(img)
            plt.title(f"Tahmin: {result}\nGüven: %{score*100:.2f}")
            plt.axis('off')
            plt.show()

            # Metin bazlı sonuç çıktısı
            print(f"\n[ANALİZ SONUCU]")
            print(f"Dosya Adı: {fn}")
            print(f"Tahmin Edilen Sınıf: {result}")
            print(f"Doğruluk/Güven Payı: %{score*100:.2f}")
            print("-" * 40)
            print("NOT: Bu bir yapay zeka aracıdır. Tıbbi teşhis yerine geçmez.")
            print("-" * 40)
        
        print("\n---> Bir sonraki resim için hazırlanıyor...")

# Sonsuz döngü fonksiyonunu çalıştır
predict_and_show_loop()
