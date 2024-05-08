import json
import matplotlib.pyplot as plt

# Read sentiment data from the JSON file
sentiment_data = []
with open("sentiment_data.json", "r") as json_file:
    for line in json_file:
        data = json.loads(line)
        sentiment_data.append(data["sentiment_scores"])

# Calculate the number of positive, negative, and neutral sentiments
num_positive = sum(1 for sentiment in sentiment_data if sentiment["compound"] > 0)
num_negative = sum(1 for sentiment in sentiment_data if sentiment["compound"] < 0)
num_neutral = len(sentiment_data) - num_positive - num_negative

# Create a pie chart
labels = ['Positive', 'Negative', 'Neutral']
sizes = [num_positive, num_negative, num_neutral]
colors = ['#00ff00', '#ff0000', '#0000ff']  # Green for positive, Red for negative, Blue for neutral
explode = (0.1, 0.1, 0.1) 

plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title("Sentiment Analysis")
plt.show()
plt.savefig("sentiment_pie_chart.png")
