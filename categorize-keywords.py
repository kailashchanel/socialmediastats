import os
import pandas as pd

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/archive/cnn.csv')
new_filename = os.path.join(dirname, 'data/categorized/cnn-categorized.csv')

# Define the categorization function
def categorize_tweet(text):
    entertainment_keywords = ['music', 'film', 'movie', 'celebrity', 'entertainment']
    sports_keywords = ['sports', 'game', 'match', 'tournament', 'athlete']
    legislation_keywords = ['bill', 'law', 'legislation', 'policy', 'act']
    politics_keywords = ['politics', 'election', 'government', 'senate', 'congress']
    science_tech_keywords = ['science', 'technology', 'tech', 'research', 'innovation']
    
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in entertainment_keywords):
        totals["A"] += 1
        return 'A'
    elif any(keyword in text_lower for keyword in sports_keywords):
        totals["B"] += 1
        return 'B'
    elif any(keyword in text_lower for keyword in legislation_keywords):
        totals["C"] += 1
        return 'C'
    elif any(keyword in text_lower for keyword in politics_keywords):
        totals["D"] += 1
        return 'D'
    elif any(keyword in text_lower for keyword in science_tech_keywords):
        totals["E"] += 1
        return 'E'
    else:
        totals["Uncategorized"] += 1
        return 'Uncategorized'

# Read the CSV file
df = pd.read_csv(filename)

totals = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "Uncategorized": 0}
print("Total: " + str(len(df)))

# Apply the categorization function to each tweet
df['Category'] = df['text'].apply(categorize_tweet)

# Save the modified DataFrame to a new CSV file
df.to_csv(new_filename, index=False)

print(totals)

print("Program complete.\n")