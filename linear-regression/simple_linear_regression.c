/*
In this example, we are creating a simple linear regression model to predict the price of a pizza based on its diameter. 
The relationship between the diameter of the pizza and its price is modeled using a linear equation: 
price = weight * diameter + bias, which can be written as [y = wx + b].

The objective of this model is to find the optimal values of 'weight' (w) and 'bias' (b) that minimize the prediction error.
To evaluate the performance of our model, we use the R-squared metric, which measures how well the model's predictions 
match the actual data. Additionally, we use the loss function (cost function) to quantify the prediction error 
during the training process.

In this example, we divide our dataset into training and test sets. The training set is used to train the model, 
while the test set is used to evaluate the model's performance and ensure that it generalizes well to unseen data.
*/



#include <stdio.h>
#include <stdlib.h>

/* Define training set */
static double train_diameters[] = {6, 8, 10, 14, 18};
static double train_prices[] = {7, 9, 13, 17.5, 18};

/* Define test set */
static double test_diameters[] = {8, 9, 11, 16, 12};
static double test_prices[] = {11, 8.5, 15, 18, 11};

/* Define trainable parameters */
static double weight = 0;
static double bias = 0;

/* Define prediction or forward function */
double *predict(double X[], double weight, double bias, int data_size) {
    double *y_predicted = (double*)malloc(data_size * sizeof(double));
    if (y_predicted == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < data_size; i++) {
        y_predicted[i] = X[i] * weight + bias; // y = wx + b
    }
    return y_predicted;
}

/* Define cost function */
double cost(double X[], double y[], double weight, double bias, int data_size) {
    double loss_value = 0;
    double sum_loss = 0;
    double *y_predicted = predict(X, weight, bias, data_size);

    for (int i = 0; i < data_size; i++) {
        loss_value = (y[i] - y_predicted[i]) * (y[i] - y_predicted[i]);
        sum_loss += loss_value;
    }
    free(y_predicted);
    return sum_loss / (2 * data_size);
}

/* Partial derivative of weight with respect to loss */
double weight_grad(double X[], double y[], double weight, double bias, int data_size) {
    double grad = 0;
    double *y_predicted = predict(X, weight, bias, data_size);

    for (int i = 0; i < data_size; i++) {
        grad += (y_predicted[i] - y[i]) * X[i];
    }
    free(y_predicted);
    return grad / data_size;
}

/* Partial derivative of bias with respect to loss */
double bias_grad(double X[], double y[], double weight, double bias, int data_size) {
    double grad = 0;
    double *y_predicted = predict(X, weight, bias, data_size);

    for (int i = 0; i < data_size; i++) {
        grad += (y_predicted[i] - y[i]);
    }
    free(y_predicted);
    return grad / data_size;
}

/* Evaluation metric r-squared score, measures how well the model fits the data */
double r_squared(double X[], double y[], double weight, double bias, int data_size) {
    double rss = 0;
    double tss = 0;
    double y_total = 0;
    double y_mean = 0;
    double *y_predicted = predict(X, weight, bias, data_size);

    for (int i = 0; i < data_size; i++) {
        rss += (y[i] - y_predicted[i]) * (y[i] - y_predicted[i]);
    }
    for (int i = 0; i < data_size; i++) {
        y_total += y[i];
    }
    y_mean = y_total / data_size;
    for (int i = 0; i < data_size; i++) {
        tss += (y[i] - y_mean) * (y[i] - y_mean);
    }
    free(y_predicted);
    return 1 - (rss / tss);
}

void model_test(double X[], double y[], double weight, double bias, int data_size) {
    double *predictions = predict(X, weight, bias, data_size);
    printf("--------------------------------------------------------------------\n");
    printf("Pizza diameter (cm)  :  Actual price (USD)  :  Price predicted (USD)\n");
    printf("--------------------------------------------------------------------\n");
    for (int i = 0; i < data_size; i++) {
        printf("%lf         :         %lf        :       %lf  \n", X[i], y[i], predictions[i]);
    }
    printf("---------------------------------------------------------------------\n");
    free(predictions);
}

int main() {
    int epoch = 10000;
    double learning_rate = 0.00002;  
    double loss = 0;
    double grad_w = 0;
    double grad_b = 0;
    int train_size = sizeof(train_diameters) / sizeof(train_diameters[0]);
    int test_size = sizeof(test_diameters) / sizeof(test_diameters[0]);

    for (int i = 1; i <= epoch; i++) {
        loss = cost(train_diameters, train_prices, weight, bias, train_size);
        grad_w = weight_grad(train_diameters, train_prices, weight, bias, train_size);
        grad_b = bias_grad(train_diameters, train_prices, weight, bias, train_size);

        weight = weight - learning_rate * grad_w;
        bias = bias - learning_rate * grad_b;

        printf("Epoch %d ---- Loss: %lf \n", i, loss);
        printf("Weight: %lf, Bias: %lf, Grad_W: %lf, Grad_B: %lf\n", weight, bias, grad_w, grad_b); 
    }

    printf("\nOPTIMUM PARAMETERS\n\n");
    printf("Optimum Weight: %lf \n", weight);
    printf("Optimum Bias: %lf \n", bias);
    printf("\nMODEL EVALUATION\n\n");
    printf("Model Loss: %lf \n", loss);
    printf("Model R Squared Score: %lf \n", r_squared(test_diameters, test_prices, weight, bias, test_size));

    printf("\nPREDICTIONS ON TEST SET\n\n");
    model_test(test_diameters, test_prices, weight, bias, test_size);

    return 0;
}
