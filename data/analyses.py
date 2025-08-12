import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./hand_gesture_data.csv')
label_counts = df['label'].value_counts()

# Create a pie chart to visualize the distribution
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
plt.title(f'Distribution of Hand Gestures {df.shape[0]}', fontsize=16)
plt.ylabel('')
plt.show()