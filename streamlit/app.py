# importing necessary libraries
import pickle
import pandas as pd
import streamlit as st


import warnings

warnings.simplefilter("ignore")


# for prediction
def predict(
    Age,
    Sex,
    Job,
    Housing,
    Saving_Accounts,
    Checking_Account,
    Credit_Amount,
    Duration,
    Purpose
):
    # categorical data handling
    load_encoding = pickle.load(open('OneHotEncoding.pkl', 'rb'))

    categorical_data = pd.DataFrame(dict(Sex=[Sex],
                                         Housing=[Housing],
                                         Saving_Accounts=[Saving_Accounts],
                                         Checking_Account=[Checking_Account],
                                         Purpose=[Purpose]))

    categorical_encoding = load_encoding.transform(categorical_data)

    # making a categorical dataframe
    enc_data = pd.DataFrame(categorical_encoding.toarray(),
                            columns=load_encoding.get_feature_names())

    # numerical data handling
    if Job == "unskilled and non-resident":
        Job = 0
    elif Job == "unskilled and resident":
        Job = 1
    elif Job == "skilled":
        Job = 2
    else:
        Job = 3

    numeric_data = pd.DataFrame(dict(Age=[Age],
                                     Job=[Job],
                                     Credit_Amount=[Credit_Amount],
                                     Duration=[Duration]))

    for j in ['Age', 'Credit_Amount', 'Duration']:
        numeric_data[j] = numeric_data[j]**(1/5)

    scaled = pickle.load(
        open("final_trained_scaler_model.pkl", "rb"))  # scaling
    num_data = scaled.transform(numeric_data)

    numeric_data = pd.DataFrame(
        num_data, columns=['Age', 'Job', 'Credit_Amount', 'Duration'])

    # final combined dataframe
    data = enc_data.join(numeric_data)

    # loading the model
    rf_model = pickle.load(open("final_trained_model.pkl", "rb"))
    predict = rf_model.predict(data)
    predict = predict[0]

    print(predict)

    if predict == 1:
        pred = "The credit seems risky so not approved. Try again later."
    else:
        pred = "Congratulations 🎉🎉 The credit has been approved."

    return pred


def main():

    result = ""
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">German Credit Card Assessment</h2>
    </div>
    """
    st.markdown(
        html_temp, unsafe_allow_html=True)  # for displaying the html output

    # getting the inputs

    Age = st.number_input("Age")
    Job = st.selectbox("Job", ('Unskilled and Non-Resident',
                       'Unskilled and Resident', 'Skilled', 'Highly Skilled')).lower()
    Credit_Amount = st.number_input("Credit Amount (in DM - Deutsch Mark)")
    Duration = st.number_input("Duration (in Months)")
    Sex = st.selectbox("Sex", ('Male', 'Female')).lower()
    Housing = st.selectbox("Housing", ('Own', 'Rent', 'Free')).lower()
    Saving_Accounts = st.selectbox(
        "Saving Accounts (in DM - Deutsch Mark)", ('Little', 'Moderate', 'Rich', 'Quite Rich', 'Not Provided')).lower()
    Checking_Account = st.selectbox(
        "Checking Account (in DM - Deutsch Mark)", ('Little', 'Moderate', 'Rich', 'Not Provided')).lower()
    Purpose = st.selectbox("Purpose", ('Business', 'Car', 'Domestic Appliances', 'Education',
                                       'Furniture/Equipment', 'Radio/TV', 'Repairs', 'Vacation/Others')).lower()

    if st.button("Predict"):
        result = predict(Age, Sex, Job, Housing, Saving_Accounts,
                         Checking_Account, Credit_Amount, Duration, Purpose)

        st.success(result)


if __name__ == "__main__":
    main()
