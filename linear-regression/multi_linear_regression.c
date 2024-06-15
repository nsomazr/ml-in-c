
/*
In this example, we are creating a simple linear regression model to predict the price of a pizza based on its diameter and number of toppings. 
The relationship between the diameter of the pizza and its price is modeled using a linear equation: 
price = weight * diameter + bias, which can be written as [y = wx + b].

The objective of this model is to find the optimal values of 'weight' (w) and 'bias' (b) that minimize the prediction error.
To evaluate the performance of our model, we use the R-squared metric, which measures how well the model's predictions 
match the actual data. Additionally, we use the loss function (cost function) to quantify the prediction error 
during the training process.

In this example, we divide our dataset into training and test sets. The training set is used to train the model, 
while the test set is used to evaluate the model's performance and ensure that it generalizes well to unseen data.
*/


/* Define training set */
static double train_features[][] = {
                                    {7, 9, 13, 17.5, 18}, // diameter
                                    {6, 8, 10, 14, 18}  // topping
                                    };

static double train_labes[] = {4, 10, 15, 16.5, 10}; // price

/* Define test set */
static double test_features[][] = {
                                    {11, 8.5, 15, 18, 11}, // diameter
                                    {8, 9, 11, 16, 12} //topping
                                    };

static double test_labels[] = {14, 13.5, 15, 17, 14.5}; //price