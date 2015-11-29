setwd("B:/DS_lab/DigitRecognizer")
library(caret)
library(h2o)
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '6g')

submitCSV <- function(pred_vector, filename){
    predict_on_test <- data.frame(seq(1:28000), pred_vector)
    names(predict_on_test) <- c("ImageId", "Label")
    write.csv(predict_on_test, file = filename, row.names = FALSE, quote = FALSE)}


##### start analysis/ preprocess data ##############

train_data_path <- "data/train.csv"
test_data_path <- "data/test.csv"
train_hex <- h2o.importFile(train_data_path, destination_frame = "train.hex")
train_hex$label <- as.factor(train_hex$label)
test_hex <- h2o.importFile(test_data_path, destination_frame = "test.hex")

train.split <- h2o.splitFrame(train_hex, ratios = c(0.75, 0.25), destination_frames = c("training", "validation"))

####################  GBM model ######################

### Load if saved before
fit_gbm <- h2o.loadModel('models/h2o_gbm')

### compute it
fit_gbm <- h2o.gbm(x = c(names(train.split[[1]][-1])), y = names(train.split[[1]][1]), 
                   training_frame = train.split[[1]],
                   validation_frame = train.split[[2]],
                   model_id = "GBMmodel", 
                   distribution = "multinomial", 
                   ntrees = 1501)
# since validation_frame was given it displays also the confusion matrix
show(fit_gbm) 


# to see the prediction (returns a data frame)
show(p_gbm)
# if no validation_frame was used during training, we can see the confusion matrix using
h2o.confusionMatrix(fit_gbm, train.split[[2]])

# save model to disk
h2o.saveModel(fit_gbm, dir = "models", name = "h2o_gbm")

# Downloads the prediction data frame
h2o.downloadCSV(p_gbm, "gbmprediction.csv")

################  Deep learning model ##############

### Load if saved before
fit_deep <- h2o.loadModel('/home/gimin/DS_lab/Digit-Recognizer/go07/h2o_deep')

### compute it
fit_deep <- h2o.deeplearning(
    x = c(names(train.split[[1]][-1])), y = names(train.split[[1]][1]), 
    training_frame = train.split[[1]], validation_frame = train.split[[2]],
    model_id = "DeeplLearningModel",
    activation = "RectifierWithDropout",
    hidden = c(512, 512),
    input_dropout_ratio = 0.01,
    l1 = 1E-04)
show(fit_deep)
h2o.saveModel(fit_deep, dir = "/home/gimin/DS_lab/Digit-Recognizer/go07", name = "h2o_deep")

###### Random Forest #######

### Load if saved before
fit_rf <- h2o.loadModel('/home/gimin/DS_lab/Digit-Recognizer/go07/h2o_rf')

### compute it
fit_rf <- h2o.randomForest(x = c(names(train.split[[1]][-1])), y = names(train.split[[1]][1]),
                           training_frame = train.split[[1]], validation_frame = train.split[[2]],
                           model_id = "RandomForestModel",
                           ntrees = 1501)
show(fit_rf)
h2o.saveModel(fit_rf, dir = "/home/gimin/DS_lab/Digit-Recognizer/go07", name = "h2o_rf")
#########################################################
attributes(fit_gbm@model)

fit_gbm@model$validation_metrics
fit_deep@model$validation_metrics
fit_rf@model$validation_metrics

# run times
sprintf("GBM run time in hours = %f", (fit_gbm@model$run_time/1000)/3600)
sprintf("Deep Learning run time in hours = %f", (fit_deep@model$run_time/1000)/3600)
sprintf("Random Forest run time in hours = %f", (fit_rf@model$run_time/1000)/3600)

#####################
# to predict on validation set 
p_gbm <- h2o.predict(fit_gbm, train.split[[2]])
p_deep <- h2o.predict(fit_deep, train.split[[2]])
p_rf <- h2o.predict(fit_rf, train.split[[2]])

new_df <- h2o.cbind(p_gbm$predict, p_deep$predict, p_rf$predict, train.split[[2]]) 

# train on new_df; here the response (label) is column 4
fit_comb_gbm <- h2o.gbm(x = c(names(new_df[-4])), y = names(new_df[4]), 
                        training_frame = new_df,
                        model_id = "GBMmodelCombi", 
                        distribution = "multinomial", 
                        ntrees = 501)
h2o.saveModel(fit_comb_gbm, dir = "/home/gimin/DS_lab/Digit-Recognizer/go07", name = "h2o_combi1")
p_combi <- h2o.predict(fit_comb_gbm, train.split[[2]])


######################################################
######################################################
### Done in Windows#####
mnist_model <- h2o.loadModel("DeepLearning_model_R_1442588220927_5")


mnist_model <- h2o.deeplearning(x = 2:785, y = 1, training_frame = train.split[[1]], 
                                activation = "RectifierWithDropout",
                                hidden = c(1024, 1024, 2048),
                                input_dropout_ratio = .2, 
                                l1 = 1E-5,
                                validation_frame = train.split[[2]],
                                epochs = 1000)
show(mnist_model)
h2o.saveModel(mnist_model)
pp1 <- h2o.predict(mnist_model, test_hex) 

h2o.downloadCSV(pp1, "prediction")
v <- read.csv("prediction")
submitCSV(v$predict, "submit10.csv")
