library(dplyr)
library(ggplot2)

reviews_frontiers <- read.csv("data/reviews_sentiment_frontiers.csv")

openai_sentiment <- read.csv("data/sentiment_openai.csv") |> 
  rename(sent_oa_7 = sentiment)

reviews_sentiment <- openai_sentiment |> 
  filter(position == 0) |> 
  left_join(reviews_frontiers, by = "rev_id") |> 
  mutate(
    sent_oa_7 = as.numeric(sent_oa_7),
    sent_diff = sent_hc_7 - sent_oa_7
  )

ggplot(reviews_sentiment, aes(x = sent_diff, y = linprob)) +
  geom_jitter()

cor(reviews_sentiment$linprob, reviews_sentiment$sent_diff)

reviews_sentiment_count <- reviews_sentiment |> 
  count(sent_hc_7, sent_oa_7)

ggplot(reviews_sentiment_count, aes(x = sent_hc_7, sent_oa_7, size = n, color = n)) +
  geom_point(alpha = 0.9)


install.packages("irrNA")
library(irrNA)

reviews_sentiment |> 
  filter(position == 0) |> 
  select(sent_hc_7, sent_oa_7) |> 
  iccNA() 
# ICC(1) one-way random -> single measure
# ICC(A,1) two-way random -> single measure


cor(reviews_sentiment$sent_hc_7, reviews_sentiment$sent_oa_7)


reviews_sentiment_diff <- reviews_sentiment |>
  filter(sent_diff != 0) |> 
  arrange(sent_diff)

reviews_sentiment_diff$review |> tail(3)

reviews_sentiment_diff |> 
  filter(row_number() <= 10 | row_number() >= n() - 9) |> 
  select(rev_id, review, sent_hc_7, sent_oa_7) |> 
  openxlsx::write.xlsx("data/tmp_reviews_openai_diff.xlsx")
