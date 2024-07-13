
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
import pandas as pd

# Clipping Outliers
def clip(data):
    columns_to_clip = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment']
    for col in columns_to_clip:
        upper_limit = data[col].quantile(0.99)
        lower_limit = data[col].quantile(0.01)
        data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)
    return data

############################ Feature engineering ###################################
def create_feature(data):
    # Interaction Features
    data['Prev_Admission_Interaction'] = data['Previous qualification (grade)'] * data['Admission grade']

    # Aggregated Features
    data['Total_Curricular_Units_1st_Sem'] = (
        data['Curricular units 1st sem (credited)'] +
        data['Curricular units 1st sem (enrolled)'] +
        data['Curricular units 1st sem (evaluations)'] +
        data['Curricular units 1st sem (approved)']
    )

    data['Total_Curricular_Units_2nd_Sem'] = (
        data['Curricular units 2nd sem (credited)'] +
        data['Curricular units 2nd sem (enrolled)'] +
        data['Curricular units 2nd sem (evaluations)'] +
        data['Curricular units 2nd sem (approved)']
    )

    data['Total_Curricular_Units'] = data['Total_Curricular_Units_1st_Sem'] + data['Total_Curricular_Units_2nd_Sem']

    # Average Grades
    data['Average_Grade_1st_Sem'] = data['Curricular units 1st sem (grade)'] / (data['Curricular units 1st sem (enrolled)'] + 1)
    data['Average_Grade_2nd_Sem'] = data['Curricular units 2nd sem (grade)'] / (data['Curricular units 2nd sem (enrolled)'] + 1)

    # Normalize Grades by Credited Units
    data['Grade_Per_Credited_1st_Sem'] = data['Curricular units 1st sem (grade)'] / (data['Curricular units 1st sem (credited)'] + 1)
    data['Grade_Per_Credited_2nd_Sem'] = data['Curricular units 2nd sem (grade)'] / (data['Curricular units 2nd sem (credited)'] + 1)

    # Ensure no division by zero (fillna handles cases where the denominator is zero)
    data['Average_Grade_1st_Sem'] = data['Average_Grade_1st_Sem'].fillna(0)
    data['Average_Grade_2nd_Sem'] = data['Average_Grade_2nd_Sem'].fillna(0)
    data['Grade_Per_Credited_1st_Sem'] = data['Grade_Per_Credited_1st_Sem'].fillna(0)
    data['Grade_Per_Credited_2nd_Sem'] = data['Grade_Per_Credited_2nd_Sem'].fillna(0)

    return data


############################ encoding ###################################################

def label_encoding(data,train=None):
    label_encoders = {}
    # Encoding each categorical column
    if train==False:
        cat_cols= ['Marital status', 'Application mode', 'Daytime/evening attendance', 'Nacionality', 'Gender']
    elif train==True:
        cat_cols= ['Marital status', 'Application mode', 'Daytime/evening attendance', 'Nacionality', 'Gender','Target']

    for column in cat_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    if train==True:
        for column in cat_cols:
            print(f"Mapping for {column}:")
            for idx, class_ in enumerate(label_encoders[column].classes_):
                print(f" {class_} -> {idx}")
            print("\n")
    return data



########################################### extracting cols to remove#####################################

def cols_to_remove():
    cols=["Nacionality","Father's qualification","Educational special needs","International","Curricular units 1st sem (without evaluations)",
           "Unemployment rate","Inflation rate","Prev_Admission_Interaction","Curricular units 1st sem (approved)","Mother's occupation","Father's occupation","Curricular units 1st sem (credited)",
           "Curricular units 1st sem (enrolled)","Curricular units 2nd sem (credited)","Curricular units 1st sem (evaluations)","Total_Curricular_Units_1st_Sem",
           "Total_Curricular_Units_2nd_Sem","Average_Grade_1st_Sem","Average_Grade_2nd_Sem"]
    
    return cols




############################################## Scaling #################################################



def scaling(train_df=None, test_df=None, is_train=False):
    if is_train:
        numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ['id', 'Target']]
        scaler = StandardScaler()
        train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
        if test_df is not None:
            test_df[numerical_features] = scaler.transform(test_df[numerical_features])
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        return train_df, test_df
    else:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        numerical_features = test_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ['id', 'Target']]
        test_df[numerical_features] = scaler.transform(test_df[numerical_features])
        return test_df






##################################### feature selection ######################################
# def feature_selection(data,train=None):
#     # Ensure 'Target' is not in the features being processed
#     features = data.drop(columns=['Target'])

#     # Step 1: Select features with significant correlation with the target
#     correlation_matrix = data.corr()
#     target_corr = correlation_matrix['Target'].drop('Target').abs()
#     selected_features = target_corr[target_corr > 0.1].index.tolist()

#     # Create a correlation matrix of the selected features
#     selected_corr_matrix = data[selected_features].corr().abs()

#     # Step 2: Remove highly correlated features
#     # Upper triangle of the correlation matrix
#     upper_tri = selected_corr_matrix.where(np.triu(np.ones(selected_corr_matrix.shape), k=1).astype(bool))

#     # Find features with correlation greater than 0.8
#     to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]

#     # Drop these features from the selected features list
#     final_features = [feature for feature in selected_features if feature not in to_drop]
#     print("total features after filtering",len(final_features))
#     final_for_test=final_features.copy()
#     # Include the target column for the final dataset
#     final_features.append('Target')
    
#     # Total features selected
#     print(f"Selected Features without target: {len(final_features) - 1}")  

#     # Create the final dataset with selected features
#     data = data[final_features]
    
#     return data,final_for_test




##################################################################### Class mapping ##############################################
def class_mapping(inn):
    cls_map= {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    if isinstance(inn,np.ndarray):
        inn=np.array([cls_map.get(x, 'Unknown') for x in inn])
        return inn
    elif isinstance(inn,pd.DataFrame):
        inn["Target"]=inn["Target"].map(cls_map)
        return inn
