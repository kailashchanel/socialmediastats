import os
from openai import OpenAI
import pandas as pd
import re

# Define relative filepaths
dirname = os.path.dirname(__file__)
filenames = [os.path.join(dirname, 'data/archive/cnn.csv'), os.path.join(dirname, 'data/archive/al_jazeera.csv'), os.path.join(dirname, 'data/archive/bbc.csv'), os.path.join(dirname, 'data/archive/reuters.csv')]
new_filenames = [os.path.join(dirname, 'data/categorized/cnn-categorized-ai-35.csv'), os.path.join(dirname, 'data/categorized/al_jazeera-categorized-ai-35.csv'), os.path.join(dirname, 'data/categorized/bbc-categorized-ai-35.csv'), os.path.join(dirname, 'data/categorized/reuters-categorized-ai-35.csv')]

# Set up OpenAI API key
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

# Function to categorize a tweet using OpenAI API
def categorize_tweet_with_openai(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a classifier that assigns categories to tweets and returns only one or two letters."},
            {"role": "user", "content": f"Classify the following tweet into one of the following categories:\
             A (Breaking News), B (World News), C (National News), D (Local News), E (Politics), F (Business & Economy), G (Technology), \
             H (Science), I (Health), J (Entertainment), K (Movies), L (TV Shows), M (Music), N (Celebrities), O (Sports), P (Football), \
             Q (Basketball), R (Baseball), S (Soccer), T (Other Sports), U (Lifestyle), V (Travel), W (Food & Drink), X (Fashion), Y (Beauty), \
             Z (Environment), AA (Education), AB (Crime & Justice), AC (Opinion & Editorial), AD (Weather), AE (Culture), AF (Automotive), AG (Real Estate), \
             AH (Finance & Markets), AI (Social Issues), AJ (Human Rights), AK (Gender Issues), AL (Race & Ethnicity), AM (Science & Space), AN (Religion), \
             AO (History), AP (Human Interest), AQ (Accidents & Disasters), AR (Military & Defense), AS (Art & Design), AT (Books & Literature), \
             AU (Media & Journalism), AV (Economy & Jobs), AW (Public Safety), AX (Health & Wellness), AY (Agriculture).\
             Tweet: '{text}'"}
        ]
    )
    category = response.choices[0].message.content.strip()


    # Sometimes, the model returns text like "Category E: Politics" or "E (Politics)" despite explicitly asking for responses 
    # of only one or two letters. This regular expression was put in place to isolate the letter from the string. However, it's not
    # perfect - strings like "E - Politics" are not parsed properly, and the value placed in the table ends up being the full string.

    match = re.search(r"\b([A-Z]+)\b(?:\s*[:\(])", category)
    if (match):
        letter = match.group(1)
    else:
        letter = category

    return letter

# Function to count the number of characters in each of the requests to gpt-3.5-turbo
def char_count(text):
    return len(f"Classify the following tweet into one of the following categories:\
        A (Breaking News), B (World News), C (National News), D (Local News), E (Politics), F (Business & Economy), G (Technology), \
        H (Science), I (Health), J (Entertainment), K (Movies), L (TV Shows), M (Music), N (Celebrities), O (Sports), P (Football), \
        Q (Basketball), R (Baseball), S (Soccer), T (Other Sports), U (Lifestyle), V (Travel), W (Food & Drink), X (Fashion), Y (Beauty), \
        Z (Environment), AA (Education), AB (Crime & Justice), AC (Opinion & Editorial), AD (Weather), AE (Culture), AF (Automotive), AG (Real Estate), \
        AH (Finance & Markets), AI (Social Issues), AJ (Human Rights), AK (Gender Issues), AL (Race & Ethnicity), AM (Science & Space), AN (Religion), \
        AO (History), AP (Human Interest), AQ (Accidents & Disasters), AR (Military & Defense), AS (Art & Design), AT (Books & Literature), \
        AU (Media & Journalism), AV (Economy & Jobs), AW (Public Safety), AX (Health & Wellness), AY (Agriculture).\
        Tweet: '{text}'")

# array to hold each of the total character counts (one per media source: total 4)
total_chars = []

for i, filename in enumerate(filenames):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Apply the categorization function to each tweet
    df['Category'] = df['text'].apply(categorize_tweet_with_openai)

    # Apply the character count for OpenAI requests to predict cost of usage
    df["request_chars"] = df['text'].apply(char_count)

    # Input sum of request characters into total_chars array
    total_chars.append(df['request_chars'].sum())
                       
    # Save the modified DataFrame to a new CSV file
    df.to_csv(new_filenames[i], index=False)

    print(f"Categorization complete.\nThe results are saved at '${new_filenames[i]}'\n")

print(total_chars)
print(f"Total Tokens: {(sum(total_chars))/4}")
print("GPT-3.5-Turbo: $0.0005 / 1K tokens")
print(f"Total Cost: ${(((sum(total_chars))/4) / 1000) * 0.0005} + Output Cost\n")