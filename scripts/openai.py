import polars as pl
import pandas as pd
import numpy as np
import random
import openai
from openai import OpenAI
import time


# 1. Load review data
reviews_hc = pl.read_csv("data/reviews/reviews_sent_hc.csv")

reviews_hc_sample = pl.read_csv("data/reviews/reviews_sent_hc_sample_20240830.csv")


# 3. Create prompt message with system prompt, examples, and review to rate

# 3.1 Load the external stored system prompt
with open("data/prompt_sys.txt", "r") as f:
    prompt_sys = f.read()


# 3.2 Create list of messages with fewshot examples
fewshot_list = []

for row in reviews_hc_sample.rows(named=True):
    fewshot_list.append({"role": "user", "content": row["review"]})
    fewshot_list.append({"role": "assistant", "content": str(row["sent_hc"])})

message = [{"role": "system", "content": prompt_sys}]
message.extend(fewshot_list)


# 4. Connect to openAI API and run function to rate sentiment

# 4.1 Load and set API key for openAI client
with open("C:/Users/munnes/Documents/API_Keys/openai_20240613.txt", "r") as f:
    openai_key = f.read()

client = OpenAI(api_key=openai_key)


# 4.2 Define function for sentiment analysis API call
def sentiment_analysis(message):
    """
    Analyze the sentiment of a given message using the GPT-4o model.

    Args:
        message (list): A list of dictionaries with the role and content of the message.

    Returns:
        pandas.DataFrame: A DataFrame with the sentiment analysis results.
    """
    # Call the OpenAI API to generate a response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=3,
    )

    # Extract the sentiment from the response
    logprobs = response.choices[0].logprobs.content[0].top_logprobs

    rows = []

    # Loop over the top 3 log probabilities and extract the token, log probability, and linear probability
    for item in logprobs:
        rows.append(
            {
                "token": item.token,
                "logprob": item.logprob,
                "linprob": np.round(np.exp(item.logprob) * 100, 2),
            }
        )

    # Create a DataFrame with the results
    df = pd.DataFrame(rows).reset_index()

    return df


# 4.3 Call sentiment analysis function for each review and write to csv
file_path = "data/sentiments/sent_openai.csv"
col_names = ["position", "sent_openai", "logprob", "linprob", "rev_id"]

pd.DataFrame(columns=col_names).to_csv(file_path, index=False, mode="w")

time_start = time.time()

for row_rev in reviews_hc.rows(named=True):

    print(row_rev["rev_id"])

    message_loop = message + [{"role": "user", "content": row_rev["review"]}]

    sentiment_results = sentiment_analysis(message_loop)

    sentiment_results = sentiment_results.assign(rev_id=row_rev["rev_id"])

    sentiment_results.to_csv(file_path, index=False, header=False, mode="a")

time_end = time.time()

time_process = time_end - time_start
print("Execution time:", round(time_process / 60, 2), "minutes")  # ~ 56 Minutes
