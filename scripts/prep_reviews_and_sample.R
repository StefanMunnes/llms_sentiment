# Load review data and store just text and sentiment rating (hand coding)
reviews_hc <- readRDS("../frontiers/data/frontiers_reviews_coded.RDS") |>
  rename(sent_hc = sent_hc_7) |>
  mutate(
    rev_id = row_number(),
    # linebreaks in review text will break csv rows
    review = stringr::str_replace_all(review, "\\n", " ")
  ) |>
  select(rev_id, review, sent_hc)

readr::write_csv(reviews_hc, "data/reviews/reviews_sent_hc.csv")


set.seed(16161)

# Randomly choose sample reviews for specific ratings
lapply(c(1, 3, 5, 7), function(rating) {
  filter(reviews_hc, sent_hc == rating) |>
    slice_sample(n = 1)
}) |>
  bind_rows() |>
  slice_sample(prop = 1) |>
  readr::write_csv(
    glue::glue(
      "data/reviews/reviews_sent_hc_sample_{date}.csv",
      date = format(Sys.Date(), "%Y%m%d")
    )
  )
