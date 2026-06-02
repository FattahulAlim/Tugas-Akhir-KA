# Import Library
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Membaca Dataset
df = pd.read_csv('student_performance.csv')

# Cek 5 data pertama
print("5 data pertama:")
print(df.head())

# Menambahkan kolom baru performance_category dengan fungsi qcut untuk membagi data berdasarkan kuantil sehingga jumlah data relatif seimbang
df['performance_category'] = pd.qcut(
    df['productivity_score'],
    q=3,
    labels=['Poor', 'Average', 'Excellent']
)

print("\nJumlah data per kategori:")
print(df['performance_category'].value_counts())

# Kolom yang digunakan sebagai inputan
features = [
    'age',
    'study_hours_per_day',
    'sleep_hours',
    'phone_usage_hours',
    'social_media_hours',
    'youtube_hours',
    'gaming_hours',
    'breaks_per_day',
    'coffee_intake_mg',
    'exercise_minutes',
    'assignments_completed',
    'attendance_percentage',
    'stress_level',
    'focus_score',
    'final_grade'
]

# Membagi input output
X = df[features]
y = df['performance_category']

# Mengubah performance_category menjadi kode angka 
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nLabel kelas:")
print(encoder.classes_)

# membagi data test dan train dengan ukuran 80% dan 20%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Normalisasi data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nJumlah data training:", X_train_scaled.shape)
print("Jumlah data testing:", X_test_scaled.shape)



model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStruktur Model:")
model.summary()

# history model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluasi dan visualisasi
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nHasil Evaluasi Model:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training dan Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training dan Validation Loss')
plt.legend()
plt.show()