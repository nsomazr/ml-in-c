/*
In this example, we are creating a simple linear regression model to predict the price of a pizza based on its diameter and number of toppings. 
The relationship between the diameter of the pizza and its price is modeled using a linear equation: 
price = (weight_one * diameter) + (weight_two * topping) + bias, which can be written as [y = w1x1 + w2x2 + b].

The objective of this model is to find the optimal values of 'weights' (w1, w2) and 'bias' (b) that minimize the prediction error.
To evaluate the performance of our model, we use the R-squared metric, which measures how well the model's predictions 
match the actual data. Additionally, we use the loss function (cost function) to quantify the prediction error 
during the training process.

In this example, we divide our dataset into training and test sets. The training set is used to train the model, 
while the test set is used to evaluate the model's performance and ensure that it generalizes well to unseen data.
*/

#include <stdio.h>
#include <stdlib.h>

/* Define training set */
static double train_features[5][2] = {
                                    {7, 6},   // {diameter, number of toppings}
                                    {9, 8},
                                    {13, 10},
                                    {17.5, 14},
                                    {18, 18}
                                    };

static double train_labels[5] = {4, 10, 15, 16.5, 10}; // price

/* Define test set */
static double test_features[5][2] = {
                                    {11, 8.5}, // {diameter, number of toppings}
                                    {8.5, 9},
                                    {15, 11},
                                    {18, 16},
                                    {11, 12}
                                    };

static double test_labels[5] = {14, 13.5, 15, 17, 14.5}; // price

/* Define trainable parameters */
static double weights[2] = {0, 0}; // Two weights for two features
static double bias = 0;

static double* predict(double X[][2], double weights[], double bias, int n_samples){
    double* y_predicted = (double*)malloc(n_samples * sizeof(double));
    for(int i = 0; i < n_samples; i++){
        y_predicted[i] = X[i][0] * weights[0] + X[i][1] * weights[1] + bias; // y = w1*x1 + w2*x2 + b
    }
    return y_predicted;
}

static double cost(double X[][2], double y[], double weights[], double bias, int n_samples){
    double sum_loss = 0;
    double* y_predicted = predict(X, weights, bias, n_samples);
    for (int i = 0; i < n_samples; i++) {
        double loss_value = (y[i] - y_predicted[i]) * (y[i] - y_predicted[i]);
        sum_loss += loss_value;
    }
    free(y_predicted);
    return sum_loss / (2 * n_samples);
}

/* Partial derivative of weight with respect to loss */
static void weights_grad(double X[][2], double y[], double weights[], double bias, int n_samples, double grad[]) {
    double* y_predicted = predict(X, weights, bias, n_samples);
    for(int j = 0; j < 2; j++){
        grad[j] = 0;
    }
    for (int i = 0; i < n_samples; i++) {
        for(int j = 0; j < 2; j++){
            grad[j] += (y_predicted[i] - y[i]) * X[i][j];
        }
    }
    for(int j = 0; j < 2; j++){
        grad[j] /= n_samples;
    }
    free(y_predicted);
}

/* Partial derivative of bias with respect to loss */
double bias_grad(double X[][2], double y[], double weights[], double bias, int n_samples) {
    double grad = 0;
    double* y_predicted = predict(X, weights, bias, n_samples);
    for (int i = 0; i < n_samples; i++) {
        grad += (y_predicted[i] - y[i]);
    }
    free(y_predicted);
    return grad / n_samples;
}

/* Evaluation metric r-squared score, measures how well the model fits the data */
double r_squared(double X[][2], double y[], double weights[], double bias, int n_samples) {
    double rss = 0;
    double tss = 0;
    double y_total = 0;
    double y_mean = 0;
    double* y_predicted = predict(X, weights, bias, n_samples);
    for (int i = 0; i < n_samples; i++) {
        rss += (y[i] - y_predicted[i]) * (y[i] - y_predicted[i]);
    }
    for (int i = 0; i < n_samples; i++) {
        y_total += y[i];
    }
    y_mean = y_total / n_samples;
    for (int i = 0; i < n_samples; i++) {
        tss += (y[i] - y_mean) * (y[i] - y_mean);
    }
    free(y_predicted);
    return 1 - (rss / tss);
}

void model_test(double X[][2], double y[], double weights[], double bias, int n_samples) {
    double* predictions = predict(X, weights, bias, n_samples);
    printf("-------------------------------------------------\n");
    printf("Actual price (USD)  :  Price predicted (USD)\n");
    printf("-------------------------------------------------\n");
    for (int i = 0; i < n_samples; i++) {
        printf("%lf        :       %lf  \n", y[i], predictions[i]);
    }
    printf("-------------------------------------------------\n");
    free(predictions);
}

int main() {
    int epoch = 30000;
    double learning_rate = 0.00005;  
    double loss = 0;
    double grad_w[2] = {0, 0};
    double grad_b = 0;
    int train_n_samples = sizeof(train_features) / sizeof(train_features[0]);
    int test_n_samples = sizeof(test_features) / sizeof(test_features[0]);

    for (int i = 1; i <= epoch; i++) {
        loss = cost(train_features, train_labels, weights, bias, train_n_samples);
        weights_grad(train_features, train_labels, weights, bias, train_n_samples, grad_w);
        grad_b = bias_grad(train_features, train_labels, weights, bias, train_n_samples);

        for(int j = 0; j < 2; j++){
            weights[j] = weights[j] - learning_rate * grad_w[j];
        }
        bias = bias - learning_rate * grad_b;

        if(i % 1000 == 0 || i == 1) {
            printf("Epoch %d ---- Loss: %lf \n", i, loss);
            for(int j = 0; j < 2; j++){
                printf("Weight %d : %lf, Grad_W %d : %lf\n", j+1, weights[j], j+1, grad_w[j]); 
            }
            printf("Bias: %lf,  Grad_B: %lf\n", bias, grad_b); 
        }
    }

    printf("\nOPTIMUM PARAMETERS\n\n");
    for(int i = 0; i < 2; i++){
        printf("Optimum Weight %d: %lf \n", i+1, weights[i]);
    }
    printf("Optimum Bias: %lf \n", bias);
    printf("\nMODEL EVALUATION\n\n");
    printf("Model Loss: %lf \n", loss);
    printf("Model R Squared Score: %lf \n", r_squared(test_features, test_labels, weights, bias, test_n_samples));

    printf("\nPREDICTIONS ON TEST SET\n\n");
    model_test(test_features, test_labels, weights, bias, test_n_samples);

    return 0;
}
