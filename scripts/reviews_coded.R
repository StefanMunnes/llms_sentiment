revs_coded <- readRDS("../frontiers/data/frontiers_reviews_coded.RDS")

revs_coded$review[1]
revs_coded$sent_hc_7[1]


rev_test <- revs_coded[c("review", "sent_hc_7")]

write.csv(rev_test, "data/tmp_reviews.csv", row.names = FALSE)

a <- janitor::get_dupes(revs_coded, review)


a <- read.csv("scripts/reviews_sentiment.csv")


a

iccNA(dplyr::select(a, sent_hc_7, sent_openai_fewshot1)) # ICC(1) one-way random -> single measure


install.packages("irrNA")
library(irrNA)

# https://stackoverflow.com/questions/78347799/why-my-anaconda-keeps-showing-error-while-loading-conda-entry-point


library(reticulate)
use_python("C:/Users/munnes/Anaconda3")

reticulate::py_available()


transformers <- reticulate::import("transformers")
py_config()


# Install Python package into virtual environment
py_install("transformers")
py_list_packages()


conda_install("transformer_sentiment", "transformers")

conda_list()
conda_version()

py_list_packages()

virtualenv_list()

py_discover_config()
