# Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling

This project involves analyzing the Titanic dataset and building machine learning models to predict the survival of passengers. The dataset used in this project is from [Kaggle](https://www.kaggle.com/c/titanic/data).

## Dataset

The dataset contains various features of Titanic passengers and their survival status. You can find more details about the dataset [here](https://www.kaggle.com/c/titanic/data).

## Project Structure

1. **Data Exploration and Preprocessing**
    - Initial data inspection
    - Handling missing values
    - Feature engineering
    - Data visualization

2. **Model Building**
    - Splitting the data into training and testing sets
    - Model selection and training
    - Hyperparameter tuning
    - Model evaluation

3. **Results and Visualizations**
    - Performance comparison of different models
    
## Data Exploration and Preprocessing

### Initial Data Inspection

Here, we load the dataset and inspect its structure, check for null values, duplicates, and understand the data types of different features.

#### Visualizing Missing Values using Heatmap
![Heatmap visulazing missing data](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/6db427d3-4b78-4a0b-bf4b-395980068cdf)

#### Percentage of Missing Values
![Percentage of missing data](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/e7bb2452-5791-4f8b-9d61-3490c05705b4)

### Handling Missing Values

We apply techniques such as imputation or removal of rows/columns with missing values.

#### Age Missing Values
![Age Missing Value Intuition](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/dad45de7-26ff-428e-92b6-abcc458d7366)

We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We will use these average age values to impute based on Pclass for Age.We use below custom function to fill missing values of age .

```python
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

```


### Feature Engineering

We create new features that could help improve the model performance, such as interaction terms or derived features.

### Data Visualization

We visualize the data to understand the distribution of features and their relationship with the target variable (survival status).

#### Passegeners Survival Based on Gender
![Passegeners Survival Based on Gender](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/f76eb87c-bbd1-4789-a5eb-2983961d6727)


#### Passegeners Survived Based on Passenger Class
![Passegeners Survived Based on Passenger Class](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/d0f71201-242e-4510-b462-cc49398d8e2c)


#### Correlation Heatmap of Numerical Features
![Correlation_Heatmap](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/b96728eb-3072-4e4c-a95f-5fc945e6089a)


## Model Building

### Splitting the Data

The dataset is split into training and testing sets using an 70-30 split.

### Models Used

We experimented with the following models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine**
- **K-Nearest Neighbors**
- **Naive Bayes**

### Hyperparameter Tuning

We used GridSearchCV to tune the hyperparameters of the models.

### Model Evaluation

The models were evaluated using metrics such as Accuracy, Precision, Recall, F1 Score

## Results 

### Performance Comparison
![Comparsions](https://github.com/QHaider4622/Deep-Dive-into-the-Titanic-Data-Analysis-and-Predictive-Modeling/assets/79516393/fee2c584-7073-4537-b3d6-dd53c2efa4cb)


### Best Performing Model

The best performing model was **Random Forest ** with an accuracy of ** 0.83 ** and an F1 Score of ** 0.822**.

## References
- [Kaggle Dataset](https://www.kaggle.com/c/titanic/data)
