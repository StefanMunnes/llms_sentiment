
revs_coded <- readRDS("../frontiers/data/frontiers_reviews_coded.RDS") |>
  mutate(rev_id = row_number()) |>
  select(rev_id, review, sent_hc_7) |> 
  mutate(review = stringr::str_replace_all(review, "\\n", " "))


write.csv(revs_coded, "data/reviews_sentiment_frontiers.csv", row.names = FALSE)