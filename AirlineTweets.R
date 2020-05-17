##### Chapter 4: Classification using Naive Bayes --------------------

## Example: Clasifying airline tweets ----
## Step 2: Exploring and preparing the data ---- 

##install.packages("tm")
##install.packages("SnowballC")
##install.packages("wordcloud")
##install.packages("e1071")
##install.packages("gmodels")

# read the data into the airline data frame
data_raw <- read.csv("airline_tweets.csv", stringsAsFactors = FALSE)

# examine the structure of the data
str(data_raw)

# convert positive/negative to factor.
data_raw$type <- factor(data_raw$type)

# examine the type variable more carefully
str(data_raw$type)
table(data_raw$type)

# build a corpus using the text mining (tm) package
library(tm)
data_corpus <- VCorpus(VectorSource(data_raw$text))

# examine the data corpus
print(data_corpus)
inspect(data_corpus[1:2])

as.character(data_corpus[[1]])
lapply(data_corpus[1:2], as.character)

# clean up the corpus using tm_map()
data_corpus_clean <- tm_map(data_corpus, content_transformer(tolower))

# show the difference between data_corpus and corpus_clean
as.character(data_corpus[[1]])
as.character(data_corpus_clean[[1]])

data_corpus_clean <- tm_map(data_corpus_clean, removeNumbers) # remove numbers
data_corpus_clean <- tm_map(data_corpus_clean, removeWords, stopwords()) # remove stop words
data_corpus_clean <- tm_map(data_corpus_clean, removePunctuation) # remove punctuation

# tip: create a custom function to replace (rather than remove) punctuation
removePunctuation("hello...world")
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")

# illustration of word stemming
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

data_corpus_clean <- tm_map(data_corpus_clean, stemDocument)

data_corpus_clean <- tm_map(data_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

# examine the final clean corpus
lapply(data_corpus[1:3], as.character)
lapply(data_corpus_clean[1:3], as.character)

# create a document-term sparse matrix
data_dtm <- DocumentTermMatrix(data_corpus_clean)

# setting pre defined R functions as TRUE, so all the stopwords, punctation and numbors are removed; The text set to lower
data_dtm2 <- DocumentTermMatrix(data_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# using custom stop words function, so we can compare the result
data_dtm3 <- DocumentTermMatrix(data_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = function(x) { removeWords(c("american", "you","now","southwest"), stopwords()) },#In my answer I used my previous results from the function that actulay worked
  removePunctuation = TRUE,
  stemming = TRUE
))

# compare the result
data_dtm
data_dtm2
data_dtm3

# creating training and test datasets
#data_dtm can be replaced with data_dtm2 or data_dtm3 in order to compare the impact of defining our stop words or using the default stopwords() function
data_dtm_train <- data_dtm3[1:10980, ]
data_dtm_test  <- data_dtm3[10981:14640, ]


data_train_labels <- data_raw[1:10980, ]$type
data_test_labels  <- data_raw[10981:14640, ]$type


prop.table(table(data_train_labels))
prop.table(table(data_test_labels))

# word cloud visualization
library(wordcloud)
wordcloud(data_corpus_clean, min.freq = 50, random.order = FALSE)

# subset the training data into positive and negative
positive <- subset(data_raw, type == "positive")
npositive  <- subset(data_raw, type == "not positive")

wordcloud(positive$text, max.words = 40, scale = c(3, 0.5))
wordcloud(npositive$text, max.words = 40, scale = c(3, 0.5))

data_dtm_freq_train <- removeSparseTerms(data_dtm_train, 0.999)
data_dtm_freq_train

# indicator features for frequent words
findFreqTerms(data_dtm_train, 5)

# save frequently-appearing terms to a character vector
data_freq_words <- findFreqTerms(data_dtm_train, 5)
str(data_freq_words)

# create DTMs with only the frequent terms
data_dtm_freq_train <- data_dtm_train[ , data_freq_words]
data_dtm_freq_test <- data_dtm_test[ , data_freq_words]

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
data_train <- apply(data_dtm_freq_train, MARGIN = 2, convert_counts)
data_test  <- apply(data_dtm_freq_test, MARGIN = 2, convert_counts)

## Step 3: Training a model on the data ----
library(e1071)
data_classifier <- naiveBayes(data_train, data_train_labels)

## Step 4: Evaluating model performance ----
data_test_pred <- predict(data_classifier, data_test)

library(gmodels)
CrossTable(data_test_pred, data_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance by applying the laplace estimator =1
data_classifier2 <- naiveBayes(data_train, data_train_labels, laplace = 1)
data_test_pred2 <- predict(data_classifier2, data_test)
CrossTable(data_test_pred2, data_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
