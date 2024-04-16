import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('m1_model.pkl')

# Load label encoders for categorical columns
label_encoders = {
    'Country': LabelEncoder(),
    'Status': LabelEncoder()
}
label_encoders['Country'].fit(['Afghanistan', 'Albania', 'Algeria','Angola',
       'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia',
       'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
       'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
       'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria',
       'Burkina Faso', 'Burundi', "Côte d'Ivoire", 'Cabo Verde',
       'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
       'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
       'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
       'Czechia', "Democratic People's Republic of Korea",
       'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
       'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
       'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala',
       'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
       'Hungary', 'Iceland', 'India', 'Indonesia',
       'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy',
       'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
       'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
       'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
       'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
       'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
       'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
       'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
       'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 'Pakistan', 'Palau',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Qatar', 'Republic of Korea',
       'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda',
       'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
       'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
       'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
       'The former Yugoslav republic of Macedonia', 'Timor-Leste', 'Togo',
       'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
       'United Arab Emirates',
       'United Kingdom of Great Britain and Northern Ireland',
       'United Republic of Tanzania', 'United States of America',
       'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
       'Zambia', 'Zimbabwe'])
label_encoders['Status'].fit(['Developing','Developed'])
# Function to preprocess input data
def preprocess_input(country, year, status, adult_mortality, alcohol, expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years, income_composition, schooling):
    # Label encode categorical columns
    country_encoded = label_encoders['Country'].transform([country])
    status_encoded = label_encoders['Status'].transform([status])
    
    # Create a DataFrame from the input data
    data = pd.DataFrame({
        'Country': [country_encoded],
        'Year': [year],
        'Status': [status_encoded],
        'Adult Mortality': [adult_mortality],
        'Alcohol': [alcohol],
        'percentage expenditure': [expenditure],
        'Hepatitis B': [hepatitis_b],
        'Measles ': [measles],
        ' BMI ': [bmi],
        'under-five deaths ': [under_five_deaths],
        'Polio': [polio],
        'Total expenditure': [total_expenditure],
        'Diphtheria ': [diphtheria],
        ' HIV/AIDS': [hiv_aids],
        'GDP': [gdp],
        'Population': [population],
        ' thinness  1-19 years': [thinness_1_19_years],
        'Income composition of resources': [income_composition],
        'Schooling': [schooling]
    })
    return data

# Create the Streamlit web app
def main():
    st.title('Life Expectancy Prediction')
    
    # Input fields for user to provide data
    country = st.selectbox('Country', ['Afghanistan', 'Albania', 'Algeria','Angola',
       'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia',
       'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
       'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
       'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria',
       'Burkina Faso', 'Burundi', "Côte d'Ivoire", 'Cabo Verde',
       'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
       'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
       'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
       'Czechia', "Democratic People's Republic of Korea",
       'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
       'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
       'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala',
       'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
       'Hungary', 'Iceland', 'India', 'Indonesia',
       'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy',
       'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
       'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
       'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
       'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
       'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
       'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
       'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
       'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 'Pakistan', 'Palau',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Qatar', 'Republic of Korea',
       'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda',
       'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
       'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
       'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
       'The former Yugoslav republic of Macedonia', 'Timor-Leste', 'Togo',
       'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
       'United Arab Emirates',
       'United Kingdom of Great Britain and Northern Ireland',
       'United Republic of Tanzania', 'United States of America',
       'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
       'Zambia', 'Zimbabwe'])  # List all countries
    year = st.number_input('Year', min_value=0)
    status = st.selectbox('Status', ['Developing', 'Developed'])
    adult_mortality = st.number_input('Adult Mortality', min_value=0)
    alcohol = st.number_input('Alcohol', min_value=0.0)
    expenditure = st.number_input('Percentage Expenditure', min_value=0.0)
    hepatitis_b = st.number_input('Hepatitis B', min_value=0.0)
    measles = st.number_input('Measles', min_value=0.0)
    bmi = st.number_input('BMI', min_value=0.0)
    under_five_deaths = st.number_input('Under-five Deaths', min_value=0.0)
    polio = st.number_input('Polio', min_value=0.0)
    total_expenditure = st.number_input('Total Expenditure', min_value=0.0)
    diphtheria = st.number_input('Diphtheria', min_value=0.0)
    hiv_aids = st.number_input('HIV/AIDS', min_value=0.0)
    gdp = st.number_input('GDP', min_value=0.0)
    population = st.number_input('Population', min_value=0.0)
    thinness_1_19_years = st.number_input('Thinness 1-19 Years', min_value=0.0)
    income_composition = st.number_input('Income Composition', min_value=0.0)
    schooling = st.number_input('Schooling', min_value=0.0)
    
    # When the user clicks the predict button
    if st.button('Predict'):
        # Preprocess the input data
        input_data = preprocess_input(country, year, status, adult_mortality, alcohol, expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years, income_composition, schooling)
        input1 = input_data.values
        # Make prediction using the loaded model
        prediction = model.predict(input1)
        
        # Display the prediction result
        st.success(f'The predicted life expectancy is {prediction[0]} years.')

# Run the Streamlit app
if __name__ == '__main__':
    main()
