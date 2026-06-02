import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('student_performance.csv')

print("5 data pertama:")
print(df.head())

print("\nNama kolom dataset:")
print(df.columns.tolist())

df['performance_category'] = pd.qcut(
    df['productivity_score'],
    q=3,
    labels=['Poor', 'Average', 'Excellent']
)

print("\nJumlah data per kategori:")
print(df['performance_category'].value_counts())

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

X = df[features]
y = df['performance_category']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nLabel kelas:")
print(encoder.classes_)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nJumlah data training:", X_train_scaled.shape)
print("Jumlah data testing:", X_test_scaled.shape)