/*
This example provides a basic understanding of linear models, represented by the equation 
y = mx + c or y = wx + b. In this context, we are training a model to learn the optimal value 
of the weight (w), which represents the slope of the line, and the bias (b), which represents 
the y-intercept.

In this example, we use imaginary sample data where the independent variable (x) is the number 
of hours a student studies, and the dependent variable (y) is the corresponding performance 
of the student. Our goal is to train the model using this data to learn the relationship between 
study hours and performance.

Once the model is trained, we test it by providing new study hours as input, and the model predicts 
the expected performance of the student. We evaluate the model using a single metric, the loss 
(cost) function, which serves as our objective function to quantify the prediction error 
during the training process.
*/


#include <stdio.h>
#include <stdlib.h>

static double X[4] = {2, 4, 6, 8}; // study hours
static double y[4] = {20, 40, 60, 80}; // marks (performance)

static double weight = 0;
static double bias = 0;

// define the prediction function - responsible for prediction y = wx+b
static double* predict(double inputs[], int size, double weight, double bias) {
    double* y_predicted = (double*)malloc(size * sizeof(double));
    if (y_predicted == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        y_predicted[i] = inputs[i] * weight + bias; // y = wx+b
    }

    return y_predicted;
}

static double cost(double inputs[], double labels[], int size, double weight, double bias) {
    double loss_value = 0;
    double sum_loss = 0;
    double* y_predicted = predict(inputs, size, weight, bias);

    for (int i = 0; i < size; i++) {
        loss_value = (labels[i] - y_predicted[i]) * (labels[i] - y_predicted[i]);
        sum_loss += loss_value;
    }

    free(y_predicted);
    return sum_loss / (2 * size);
}

static double weight_grad(double inputs[], double labels[], int size) {
    double grad = 0;
    double* y_predicted = predict(inputs, size, weight, bias);

    for (int i = 0; i < size; i++) {
        grad += (y_predicted[i] - labels[i]) * inputs[i];
    }

    free(y_predicted);
    return grad / size;
}

static double bias_grad(double inputs[], double labels[], int size) {
    double grad = 0;
    double* y_predicted = predict(inputs, size, weight, bias);

    for (int i = 0; i < size; i++) {
        grad += (y_predicted[i] - labels[i]);
    }

    free(y_predicted);
    return grad / size;
}

void test() {
    int size;

    printf("Enter the size of your data (Number of data points):\n");
    scanf("%d", &size);

    double inputs[size];
    for (int i = 0; i < size; i++) {
        printf("Enter number of hour(s) for data : %d \n", i + 1);
        scanf("%lf", &inputs[i]);
    }

    double* predictions = predict(inputs, size, weight, bias);
    printf("Prediction for inputs\n");

    for (int i = 0; i < size; i++) {
        printf("%lf hrs : %lf marks(performances)\n", inputs[i], predictions[i]);
    }

    free(predictions);
}

int main(void) {
    int epoch = 15000;
    double learning_rate = 0.0001;  
    int size = sizeof(X) / sizeof(X[0]);

    for (int i = 1; i <= epoch; i++) {
        double loss = cost(X, y, size, weight, bias);
        double grad_w = weight_grad(X, y, size);
        double grad_b = bias_grad(X, y, size);

        weight = weight - learning_rate * grad_w;
        bias = bias - learning_rate * grad_b;

        printf("Epoch %d ---- Loss: %lf \n", i, loss);
        printf("Weight: %lf, Bias: %lf, Grad_W: %lf, Grad_B: %lf\n", weight, bias, grad_w, grad_b); 
    }

    printf("Optimum Weight: %lf \n", weight);
    printf("Optimum Bias: %lf \n", bias);

    test();
}
