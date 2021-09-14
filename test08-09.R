###############################################################################
#                                                                             #
#             GENERATIVE ADVERSARIAL NEURAL NETWORKS                          #
#                                                                             #
###############################################################################

install.packages("keras")
install.packages("tensorflow")


install.packages('Rcpp')
library(Rcpp)

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")

install.packages("caTools")


### opening libraries ----

library(tensorflow)
library(keras)
library(EBImage)

library(tidyverse)
library(ggplot2)

library(caTools)

### installing necessary dependencies ----

install_tensorflow()

install_keras()

### training with MNIST dataset ---- 

mnist <- dataset_mnist() 


## preparing the data 

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

# x - 3d array of images, width, height 
# to prepare the data we need to conver the 3d data into 
# matrices, by reshaping the width and height into a single dimension 
# ie. 28 x 28 flattened as vectors of length 784
# then we need to convert grayscale values of integers ranging between 0 to 255 into 
# floating point values between 0 and 1 

# reshape 
x_train <- array_reshape(x_train, c(nrow(x_train), 784)) # large matrix 
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale 
x_train <- x_train / 255
x_test <- x_test / 255

## y data is ingeter vector with values ranging 0 - 9
## one-hot encoding the vectors into binary class matrices 
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

### defining the model 
## the simplest is a Sequential model - a linear stack of layers 

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', 
              input_shape = c(784)) %>%  # shape of input data 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax') # outputs a length 10 numeric vector
                               # using softmax activation function 
summary(model)

### model compiling 
## with loss function, optimizer and metrics 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

### training and evaluation 
# training the model for 30 epochs using batches of 128 images 
history <- model %>% 
  fit(
    x_train, y_train, 
    epochs = 30, batch_size = 128,
    validation_split = 0.2
  )

## plotting the trained model (loss and accuracy metrics)
plot(history)


### evaluating model's performance on the test data 

model %>% 
  evaluate(x_test, y_test)

### generate predictions on new data

predicted <- model %>% 
  #predict_classes(x_test) -- deprecated 
  predict(x_test) %>% 
  k_argmax()
summary(predicted)


######################################################################
######################################################################


### training with MGP data ---- 

prices <- read.csv("data/MGP_prices.csv", sep=";")[,1:3] %>%
  rename(data = ï.....Data.Date..YYYYMMDD.,
         ora = Ora..Hour ) %>%
  mutate(ora = gsub(" ", "", ifelse(ora >= 1 & ora <= 9, paste("0",ora),ora)),
         anno = substr(data, 1,4),
         mese = substr(data, 5,6),
         giorno = substr(data, 7,8),
         ora = gsub(" ", "", paste(ora, ":00"))) %>%
  filter(!is.na(PUN)) %>%
  select(-data) %>%
  unite(data, giorno, mese, anno, sep="-") %>%
  unite(data_ora, data,ora, sep=" ")



str(prices)

## hourly changes in prices over 2021

(ts_prices <- ggplot(prices, aes(data_ora, PUN, group=1))+
    geom_line()+
    theme_classic())

## daily changes over 2021 ----


prices_daily <- prices %>% 
  separate(data_ora, into=c("data","ora"),sep= " ") %>%
  group_by(data) %>%
  summarise(daily = mean(PUN))

(ts_prices_d <- ggplot(prices_daily, aes(data, daily, group=1))+
    geom_line()+
    theme_classic())





### splitting training and testing dataset ---- 

split <- sample.split(prices$PUN, SplitRatio = 0.8)
training_set <- subset(prices, split == TRUE)
testing_set <- subset(prices, split == FALSE)







