library(ollamar)

test_connection()
list_models()

# 1.1 Load all reviews with sentiments and sample reviews
reviews_hc <- read.csv("data/reviews/reviews_sent_hc.csv")
reviews_hc_sample <- read.csv("data/reviews/reviews_sent_hc_sample_20240830.csv")

# 1.2 Load system prompt from external file
prompt_system <- scan("data/prompt_sys.txt", what = "character", sep = "\n")


# 2. Create base message object whit system prompt and fewshot examples
messages_base <- list(list(role = "system", content = prompt_system))

for (row in seq_len(nrow(reviews_hc_sample))) {
  messages_base <- append(
    messages_base,
    list(list(role = "user", content = reviews_hc_sample$review[row]))
  )
  messages_base <- append(
    messages_base,
    list(list(role = "assistent", content = as.character(reviews_hc_sample$sent_hc[row])))
  )
}


# 3. Loop over all reviews and write model response to prepared csv file
file_path <- "data/sentiments/sent_ollama.csv"

readr::write_csv(
  data.frame(sent_ollama = character(0), rev_id = numeric(0)),
  file = file_path
)


time_start <- Sys.time()

for (rev_id in reviews_hc$rev_id) {
  message(rev_id)

  messages <- append(
    messages_base,
    list(list(role = "user", content = reviews_hc$review[reviews_hc$rev_id == rev_id]))
  )

  result_sent <- chat("llama3.1", messages, output = "text")

  readr::write_csv(
    data.frame(sent_ollama = result_sent, rev_id = rev_id),
    file = file_path, append = TRUE
  )
}

time_end <- Sys.time()

time_end - time_start # ~ 16 Minuten

# RAM: 64GB
# GPU: RTX 4080 16 GB
# CPU:
