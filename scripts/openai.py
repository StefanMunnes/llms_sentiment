import polars as pl
import pandas as pd
import numpy as np
import random
import openai
from openai import OpenAI


# 1. Load review data
reviews = pl.read_csv("data/reviews_sentiment_frontiers.csv")


# 2.1 Function to randomly sample example reviews for different ratings
def random_rating(df, ratings, n):
    """
    Function to randomly sample example reviews for different ratings.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with reviews and their human-coded sentiment ratings
    ratings : list
        List of unique sentiment ratings, e.g. [1, 3, 5, 7]
    n : int
        Number of reviews to sample per rating

    Returns
    -------
    pandas.DataFrame
        polarsDataFrame with text (review) and rating (sent_hc_7) column; random order
    """
    revs_examples = pl.DataFrame(schema=df.schema)

    for rating in ratings:
        # select reviews with the current rating
        revs = df.filter(pl.col("sent_hc_7") == rating)
        # sample n reviews
        revs = revs.sample(n=n)
        # add to the result dataframe
        revs_examples = pl.concat([revs_examples, revs])

    # shuffle the result dataframe
    revs_examples = revs_examples.sample(fraction=1)

    # convert the rating column to string
    # revs_examples = revs_examples.with_columns(pl.col("sent_hc_7"))

    return revs_examples


# 2.2 Randomly sample example reviews for different ratings
reviews_examples = random_rating(reviews, [1, 4, 7], 1)

# 2.3 Remove the examples from the original dataframe
reviews_noexamples = reviews.join(reviews_examples, on="rev_id", how="anti")


# 3. Create prompt message with system prompt, examples, and review to rate

# 3.1 Define the system prompt
system_prompt = "Du bist Experte für deutschsprachige Literatur. Deine Aufgabe ist es, zu bewerten, wie gut oder schlecht ein Buch in einer Buchrezension besprochen wurde. Der Text ist eine Zusammenfassung der Rezension, die ursprünglich in einer Zeitung erschienen ist. Der Text enthält eine kurze Zusammenfassung des Inhalts und die Bewertung des Rezensenten. Die Qualität kann auf verschiedene Aspekte bezogen werden, darunter den sprachlichen Stil und einen überzeugenden Inhalt. Reflektiere die einzelnen Aspekte der Buchqualität und bewerte abschließend den Gesamteindruck. Bewerte die Qualität des rezensierten Buches auf einer Skala von 1 (sehr schlecht) bis 7 (sehr gut). Antworte nur mit der richtigen Zahl."

# 3.2 Define the list with fewshot examples
fewshot_list = []  #  empty list to append for fewshot and empty if zeroshot

for row in reviews_examples.rows(named=True):
    fewshot_list.append({"role": "user", "content": row["review"]})
    fewshot_list.append({"role": "assistant", "content": str(row["sent_hc_7"])})


# 3.3 Define function to create message
def create_message(review):
    """
    Create a message for the openAI API call.

    This function combines the system prompt and the fewshot examples
    and adds the review text. The message is a list of dictionaries with
    "role" and "content" keys.

    Parameters:
        review (str): The review text to be evaluated.

    Returns:
        list: The message to be sent to the openAI API.
    """

    global system_prompt
    global fewshot_list

    # create the first system message
    message = [{"role": "system", "content": system_prompt}]
    # add the fewshot examples
    message.extend(fewshot_list)
    # add the review text as final user message
    message.append({"role": "user", "content": review})

    return message


# 4. Connect to openAI API and run function to rate sentiment

# 4.1 Load and set API key for openAI client
openai.api_key = open(
    "C:/Users/munnes/Documents/API_Keys/openai_20240613.txt", "r"
).read()

client = OpenAI(api_key=openai.api_key)


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
    df = pd.DataFrame(rows).reset_index().rename(columns={"index": "index_tok"})

    return df


# 4.3 Call sentiment analysis function for each review
sentiments_df = pl.DataFrame()

file_path = "data/sentiment_openai.csv"
col_names = ["position", "sentiment", "logprob", "linprob", "rev_id"]

pd.DataFrame(columns=col_names).to_csv(file_path, index=False, mode="w")


for row_rev in reviews_noexamples.head(20).rows(named=True):

    print(row_rev["rev_id"])

    message = create_message(row_rev["review"])

    sentiments_df_loop = sentiment_analysis(message)

    sentiments_df_loop = sentiments_df_loop.assign(rev_id=row_rev["rev_id"])

    sentiments_df_loop.to_csv(file_path, index=False, header=False, mode="a")
