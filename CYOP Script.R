# Load packages
if(!require(tidyverse)) install.packages("tidyverse", repos="http://cran.us.r-project.org")
if(!require(R.utils)) install.packages("R.utils", repos="http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos="http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos="http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos="http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos="http://cran.us.r-project.org")
library(R.utils)
library(e1071)
library(caret)
library(class)
library(lubridate)

# Download data
if(!file.exists(traffic.csv)){
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz", "traffic.csv.gz")
  gunzip("traffic.csv.gz")
}
df<-read.csv("traffic.csv")

# Exploratory Data Analysis

# Analyze basic features
nrow(df)
head(df)

# Preliminary graph for data analysis
plot(df$temp, df$traffic_volume, main="Temperature vs Traffic Volume", xlab="Temperature", ylab="Traffic Volume", col="red", pch=20)
plot(df$rain_1h, df$traffic_volume, main="Rain vs Traffic Volume", xlab="Rain in the Hour", ylab="Traffic Volume", col="blue", pch=20)
plot(df$snow_1h, df$traffic_volume, main="Snow vs Traffic Volume", xlab="Snow in the Hour", ylab="Traffic Volume", col="#ADD8E6", pch=20)
plot(df$clouds_all, df$traffic_volume, main="Clouds vs Traffic Volume", xlab="Cloud Percentage", ylab="Traffic Volume", col="grey", pch=20)
ggplot(df, aes(x=weather_main, y=traffic_volume)) + labs(title="Weather vs Traffic Volume", x="Weather", y="Traffic Volume") + geom_boxplot(fill='yellow')

# Data cleaning
df2 <- subset(df, temp>200 & rain_1h<1000)  # remove outliers (kelvin <200 and rain >1000)
nrow(df2)
head(df2)

outliers <- subset(df, temp<200 | rain_1h>1000)  # examine outliers
nrow(outliers)
print(outliers)

# Graphs for data analysis
plot(df2$temp, df2$traffic_volume, main="Temperature vs Traffic Volume", xlab="Temperature", ylab="Traffic Volume", col="red", pch=20)
plot(df2$rain_1h, df2$traffic_volume, main="Rain vs Traffic Volume", xlab="Rain in the Hour", ylab="Traffic Volume", col="blue", pch=20)
plot(df2$snow_1h, df2$traffic_volume, main="Snow vs Traffic Volume", xlab="Snow in the Hour", ylab="Traffic Volume", col="#ADD8E6", pch=20)
plot(df2$clouds_all, df2$traffic_volume, main="Clouds vs Traffic Volume", xlab="Cloud Percentage", ylab="Traffic Volume", col="grey", pch=20)
ggplot(df2, aes(x=weather_main, y=traffic_volume)) + labs(title="Weather vs Traffic Volume", x="Weather", y="Traffic Volume") + geom_boxplot(fill='yellow')


# Temperature versus Traffic Volume
plot(df2$temp, df2$traffic_volume, main="Temperature vs Traffic Volume", xlab="Temperature", ylab="Traffic Volume", col="red", pch=20)
# Rain versus Traffic Volume
plot(df2$rain_1h, df2$traffic_volume, main="Rain vs Traffic Volume", xlab="Rain in the Hour", ylab="Traffic Volume", col="blue", pch=20)
# Snow versus Traffic Volume
plot(df2$snow_1h, df2$traffic_volume, main="Snow vs Traffic Volume", xlab="Snow in the Hour", ylab="Traffic Volume", col="#ADD8E6", pch=20)
# Clouds versus Traffic Volume
plot(df2$clouds_all, df2$traffic_volume, main="Clouds vs Traffic Volume", xlab="Cloud Percentage", ylab="Traffic Volume", col="grey", pch=20)
# Weather versus Traffic Volume
ggplot(df2, aes(x=weather_main, y=traffic_volume)) + labs(title="Weather vs Traffic Volume", x="Weather", y="Traffic Volume") + geom_boxplot(fill="yellow")

# Parse datetime object into year, month, day, and hour
df2$time <- parse_date_time(df2$date_time, "ymd HMS")
df2$year <- df2$time %>% year()
df2$month <- df2$time %>% month()
df2$day <- df2$time %>% day()
df2$hour <- df2$time %>% hour()

#Examine the year to year variation
ggplot(df2, aes(x=as.factor(year), y=traffic_volume)) + labs(title="Year vs Traffic Volume", x="Year", y="Traffic Volume") + geom_boxplot()
# Examine the month to month variation
ggplot(df2, aes(x=as.factor(month), y=traffic_volume)) + labs(title="Month vs Traffic Volume", x="Month", y="Traffic Volume") + geom_boxplot()
# Examine the traffic flow with respect to the day of the month
ggplot(df2, aes(x=as.factor(day), y=traffic_volume)) + labs(title="Day vs Traffic Volume", x="Day", y="Traffic Volume") + geom_boxplot()
# Examine hour to hour variation
ggplot(df2, aes(x=as.factor(hour), y=traffic_volume)) + labs(title="Hour vs Traffic Volume", x="Hour", y="Traffic Volume") + geom_boxplot()

# Hold hour constant, analyze relationship between temperature and traffic volume 
# Temperature versus Traffic Volume at 3am
df2_hour3 <- subset(df2, hour==3)
plot(df2_hour3$temp, df2_hour3$traffic_volume, main="Temperature vs Traffic Volume (3am)", xlab="Temperature", ylab="Traffic Volume", pch=20)
# Temperature versus traffic volume at 12pm
df2_hour12 <- subset(df2, hour==12)
plot(df2_hour12$temp, df2_hour12$traffic_volume, main="Temperature vs Traffic Volume (12pm)", xlab="Temperature", ylab="Traffic Volume", pch=20)

# Modeling Approach: Treat this as a classification problem
range(df2$traffic_volume) # traffic_volume ranges from 0 to 7280
# Generate barplot for classification categories
df2$traffic <- cut(df2$traffic_volume, c(0,2500,5000,7500), labels=c("light", "medium", "heavy"), right=FALSE)
barplot(prop.table(table(df2$traffic)))

# Preprocess the data (normalize)
df3 <- subset(df2, select=-c(holiday, weather_main, weather_description, year, month, day, time, date_time, traffic_volume, traffic))
normalize <- function(x){ # normalize using min-max scaling
  ((x-min(x))/(max(x)-min(x)))
}
processed_df <- as.data.frame(lapply(df3, normalize)) # normalize all numeric variables
processed_df$traffic <- df2$traffic # append the traffic categories


# Subset the data
# Create a train test split with 90% training set and a 10% testing set
set.seed(123, sample.kind="Rounding")
train_index <- sample(1:nrow(processed_df), 0.9*nrow(processed_df))
training_data <- processed_df[train_index,] # training data
testing_data <- processed_df[-train_index,] # testing data

# Model 1 - KNN 
# Train knn model using Caret and k-fold cross validaiton
model_1 <- train(traffic~., method="knn", trControl=trainControl(method="repeatedcv", number=5, repeats=10), data=training_data)
model_1 # Print model outcome

knn_pred <- predict(model_1, newdata=testing_data)
cm_1 <- confusionMatrix(knn_pred, testing_data$traffic) # Generate confusion matrix
cm_1 # Print confusion matrix
model_1_accuracy <- cm_1$overall["Accuracy"] # Save accuracy field for results table

# Model 2 - Decision Tree
# Train dt model using Caret, k-fold cv, and rpart2 to optimize dt depth
tunegrid <- expand.grid(maxdepth=c(1, 3, 5, 7, 9, 11))
model_2 <- train(traffic~., method="rpart2", trControl=trainControl(method="repeatedcv", number=5, repeats=10), tuneGrid=tunegrid, data=training_data)
model_2 # Print model outcome

dt_pred <- predict(model_2, newdata=testing_data)
cm_2 <- confusionMatrix(dt_pred, testing_data$traffic) # Generate confusion matrix
cm_2 # Print confusion matrix
model_2_accuracy <- cm_2$overall["Accuracy"] # Save accuracy field for results table

plot(model_2$finalModel, uniform=TRUE, main="Classification Tree") # Plot out decision tree splits
text(model_2$finalModel, all=TRUE, cex=.8) # Label decision tree

# Model 3 - Multinomial Regression
# Train multinomial logistic regression using caret and k-fold cv
model_3 <- train(traffic~., method="multinom", trControl=trainControl(method="repeatedcv", number=5, repeats=10), trace=FALSE, data=training_data)
model_3 # Print model outcome

mn_pred <- predict(model_3, newdata=testing_data)
cm_3 <- confusionMatrix(mn_pred, testing_data$traffic) # Generate confusion matrix
cm_3 # Print confusion matrix
model_3_accuracy <- cm_3$overall["Accuracy"] # Save accuracy field for results table

# Model 4 - Neural Network (WARNING: TAKES 30+ minutes)
# Use a simple feed-forward neural network with one hidden layer using the nnet method
model_4 <- train(traffic~., method="nnet", trControl=trainControl(method="repeatedcv", number=5, repeats=10), trace=FALSE, data=training_data)
model_4 # Print model outcome

nn_pred <- predict(model_4, newdata=testing_data)
cm_4 <- confusionMatrix(nn_pred, testing_data$traffic) # Generate confusion matrix
cm_4  # Print confusion matrix
model_4_accuracy <- cm_4$overall["Accuracy"] # Save accuracy field for results table

# Results
tab <- matrix(c(model_1_accuracy, model_2_accuracy, model_3_accuracy, model_4_accuracy), ncol = 1, byrow = TRUE)
colnames(tab) <- "Accuracy"
rownames(tab) <- c("K Nearest Neighbors", "Decision Tree", "Multinomial Regression", "Neural Network")
tab <- as.table(tab)  # Generate results table
tab # Print results table