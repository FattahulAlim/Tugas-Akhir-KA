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
