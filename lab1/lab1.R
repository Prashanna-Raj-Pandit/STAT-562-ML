setwd("~/Documents/SIUE/STAT 562")
data("mtcars")
df<-mtcars
n_col<-ncol(df)
n_row<-nrow(df)
n_col;n_row

response_var<-c("am")
qualitative_vars<-c("vs")
quantitative_vars<-setdiff(names(df),c(response_var,qualitative_vars))
n_quant <- length(quantitative_vars)
n_qual  <- length(qualitative_vars)
cat("Quantitative variables:", n_quant, "\n")
cat("Qualitative variables:", n_qual, "\n")
round(prop.table(table(df$am))*100,digits=1)

# Convert the qualitative predictor to a factor variable in df
df$vs <- factor(df$vs, levels = c(0, 1), labels = c("V-shaped", "Straight"))

# Convert factor variable into dummy variables (one-hot encoding)
vs_encoded <- model.matrix(~ vs, data = df)[, -1]

# Convert response to factor with readable labels
df$am <- factor(df$am, levels = c(0, 1), labels = c("Automatic", "Manual"))

# Check structure
str(df)

# Standardize:
scaled_quant <- scale(df[, quantitative_vars])
# Display summary of one variable (e.g., 'hp') before standardization
summary(df$hp)
summary(scaled_quant[,"hp"])
data_ready<-cbind(as.data.frame(scaled_quant),vs_encoded)
str(data_ready)
head(data_ready)


library(rsample)
set.seed(123)

# Combine predictors and response temporarily for splitting
split_data <- cbind(data_ready, am = df$am)

# Perform 70/30 stratified split by 'am'
split_obj <- initial_split(split_data, prop = 0.70, strata = am)

# Extract training and test datasets
train_data <- training(split_obj)
test_data  <- testing(split_obj)

# Check proportions in train/test
prop.table(table(train_data$am))
prop.table(table(test_data$am))

train_labels<-train_data$am
test_labels<-test_data$am
# Remove the label column from the feature matrices
train_data <- subset(train_data, select = -am)
test_data  <- subset(test_data, select = -am)

# (e) k-NN with k = √n

library(class)
k=sqrt(nrow(train_data))
k
test_pred<-knn(train = train_data,test = test_data,cl=train_labels,k=5)

table(test_labels,test_pred)
accuracy<-mean(test_labels==test_pred)
accuracy

#(g)

library(caret)
set.seed(123)
ctrl<-trainControl(method = "cv",number=10)
knn_grid<-expand.grid(k=1:20)
train_knn<-cbind(train_data,am=train_labels)

# train kNN using cross validation
knn_cv<-train(am~.,data=train_knn,
              method="knn",
              trControl=ctrl,
              tuneGrid=knn_grid)
best_k<-knn_cv$bestTune$k
plot(knn_cv, main = "k-NN Accuracy vs. Number of Neighbors (k)")

best_k

# (h)
library(caret)
set.seed(123)
train_control<-trainControl(method = "cv", number = 10, classProbs = TRUE, savePredictions = "final")
# Fit final k-NN model with cross validation
final_model_cv<-train(am~.,data=train_knn,method="knn",tuneGrid=data.frame(k=best_k), trControl=train_control)
print(final_model_cv)

# predict on test dataset
final_prediction<-predict(final_model_cv,newdata = test_data)
final_prediction

final_accuracy<-mean(final_prediction==test_labels)
final_accuracy

conf_matrix <- confusionMatrix(final_prediction, test_labels)
conf_matrix

# Extract performance metrics
overall_accuracy  <- conf_matrix$overall["Accuracy"]
misclassification <- 1 - overall_accuracy
sensitivity       <- conf_matrix$byClass["Sensitivity"]   # True Positive Rate (Recall)
specificity       <- conf_matrix$byClass["Specificity"]   # True Negative Rate
precision         <- conf_matrix$byClass["Precision"]
recall            <- conf_matrix$byClass["Recall"]
f1_score          <- conf_matrix$byClass["F1"]

cat("Overall Accuracy:", round(overall_accuracy, 4), "\n")
cat("Misclassification Rate:", round(misclassification, 4), "\n")
cat("Sensitivity:", round(sensitivity, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1 Score:", round(f1_score, 4), "\n")


# --- ROC Curve and AUC evaluation ---
library(pROC)

# Get predicted probabilities for the positive class ("Manual")
prob_predictions <- predict(final_model_cv, newdata = test_data, type = "prob")

# Build ROC curve
roc_curve <- roc(
  response = test_labels,
  predictor = prob_predictions$Manual,
  levels = levels(test_labels)
)

# Plot ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve for Final k-NN Mode")
auc_value <- auc(roc_curve)
auc_value

