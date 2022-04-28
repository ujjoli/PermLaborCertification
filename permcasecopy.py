from io import BytesIO
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from tempfile import TemporaryFile
# import dash_html_components as html
from dash import html as html
# import dash_core_components as dcc
from dash import dcc
import dash_bootstrap_components as dbc
from datetime import datetime as dt
from dash import dash_table
from dash.dependencies import Input, Output, State
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import warnings
# ignore warnings
# import tensorflow as tf
# from tensorflow import keras
from google.cloud import storage
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='/Users/ujjwaloli/Desktop/ujjwal documents/Data Analytics job/capstoneproject_Service_key.json'


client = storage.Client()

bucket = client.get_bucket("capstone-data-perm")

blob = bucket.get_blob(f"final_data.xlsx")

bt = blob.download_as_bytes()

# from io import StringIO

# s = str(bt)
# s= StringIO(s)

df = pd.read_excel(bt)
print(df.head())
# df = pd.read_excel('final_data.xlsx')

import datetime

# df1['Year']= df1['DECISION_DATE'].split()
df['Year'] = pd.DatetimeIndex(df['DECISION_DATE']).year
df.head()

# finding months duration from given time
time_taken_list = list((df['DECISION_DATE'] - df['CASE_RECEIVED_DATE']) / np.timedelta64(1, 'M'))
value_time = []
for i in range(len(time_taken_list)):
    val = time_taken_list[i]
    if val < 3:
        value_time.append("at least 3 months")
    elif val < 6:
        value_time.append("at least 6 months")
    elif val < 9:
        value_time.append("at least 9 months")
    elif val < 12:
        value_time.append("at least 12 months")
    elif val < 15:
        value_time.append("at least 15 months")
    elif val < 24:
        value_time.append("at least 24 months")
    elif val < 36:
        value_time.append("at least 36 months")
    elif val < 60:
        value_time.append("at least 60 months")
    elif val < 84:
        value_time.append("at least 84 months")
    elif val < 120:
        value_time.append("at least 120 months")
    else:
        value_time.append("more than 120 months")

df['time_taken'] = value_time

dff = df.drop(
    ['DECISION_DATE', 'CASE_RECEIVED_DATE', 'CASE_NUMBER', 'REFILE', 'FW_OWNERSHIP_INTEREST', 'PW_Job_Title_9089',
     'FW_INFO_EDUCATION_OTHER', 'JOB_INFO_EDUCATION_OTHER'], axis=1)

# Replacing missing values with mode
dff['CLASS_OF_ADMISSION'] = dff['CLASS_OF_ADMISSION'].fillna((dff['CLASS_OF_ADMISSION'].mode()[0]))
dff['COUNTRY_OF_CITIZENSHIP'] = dff['COUNTRY_OF_CITIZENSHIP'].fillna((dff['COUNTRY_OF_CITIZENSHIP'].mode()[0]))
dff['JOB_INFO_MAJOR'] = dff['JOB_INFO_MAJOR'].fillna((dff['JOB_INFO_MAJOR'].mode()[0]))
dff['FOREIGN_WORKER_INFO_MAJOR'] = dff['FOREIGN_WORKER_INFO_MAJOR'].fillna((dff['FOREIGN_WORKER_INFO_MAJOR'].mode()[0]))
dff['FOREIGN_WORKER_INFO_EDUCATION'] = dff['FOREIGN_WORKER_INFO_EDUCATION'].fillna('Other')
dff['JOB_INFO_FOREIGN_LANG_REQ'] = dff['JOB_INFO_FOREIGN_LANG_REQ'].fillna('Other')
dff['PW_SOC_TITLE'] = dff['PW_SOC_TITLE'].fillna('Other')

dff["PW_AMOUNT_9089"].replace({"#############": dff['PW_AMOUNT_9089'].mode()[0]}, inplace=True)

dff['PW_AMOUNT_9089'] = dff['PW_AMOUNT_9089'].fillna((dff['PW_AMOUNT_9089'].mode()[0]))

wages_list = list(dff['PW_AMOUNT_9089'])
unit_list = list(dff['PW_UNIT_OF_PAY_9089'])

new_wages_list = []
for i in range(len(wages_list)):
    try:
        wages_list[i] = float(wages_list[i])
    except:
        wages_list[i] = float(wages_list[i].replace(',', ''))
    finally:
        if unit_list[i] == 'Hour':
            wages_list[i] = wages_list[i] * 40 * 52
        elif unit_list[i] == 'Week':
            wages_list[i] = wages_list[i] * 52
        elif unit_list[i] == 'Month':
            wages_list[i] = wages_list[i] * 12
        elif unit_list[i] == 'Bi-Weekly':
            wages_list[i] = wages_list[i] * 24
        else:
            wages_list[i] = wages_list[i]

    new_wages_list.append(wages_list[i])

dff['PW_AMOUNT_9089'] = new_wages_list

print(dff.columns)

dfff = dff.drop(['PW_UNIT_OF_PAY_9089', 'Year'], axis=1)
dfff = dfff.dropna()
print(dfff.shape)

state_abbrevs = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'Northern Mariana Islands': 'MP',
    'Palau': 'PW',
    'Puerto Rico': 'PR',
    'Virgin Islands': 'VI',
    'District of Columbia': 'DC'
}

# Capitalizing Keys
us_state_abbrev = {k.upper(): v for k, v in state_abbrevs.items()}
dfff['EMPLOYER_STATE'].replace(us_state_abbrev, inplace=True)
dfff.EMPLOYER_STATE = dfff.EMPLOYER_STATE.astype(str)
# dfff['EMPLOYER_STATE'].value_counts()


dfff['JOB_INFO_WORK_STATE'].replace(us_state_abbrev, inplace=True)
dfff.JOB_INFO_WORK_STATE = dfff.JOB_INFO_WORK_STATE.astype(str)
# dfff['EMPLOYER_STATE'].value_counts()

dfff = dfff.rename(columns={'PW_AMOUNT_9089': 'WAGES_OFFERED',
                            'PW_SOC_TITLE': 'SOC_TITLE',
                            'JOB_INFO_WORK_CITY': 'JOB_CITY',
                            'JOB_INFO_WORK_STATE': 'JOB_STATE',
                            'JOB_INFO_JOB_TITLE': 'JOB_TITLE',
                            'JOB_INFO_EDUCATION': 'REQD_EDUCATION',
                            'JOB_INFO_MAJOR': 'RELTD_MAJOR',
                            'JOB_INFO_TRAINING': 'TRAINING_REQD',
                            'JOB_INFO_FOREIGN_LANG_REQ': 'LANG_REQD',
                            'COUNTRY_OF_CITIZENSHIP': 'CITIZENSHIP',
                            'CLASS_OF_ADMISSION': 'ADMISSION_TYPE',
                            'FOREIGN_WORKER_INFO_EDUCATION': 'WORKER_EDUCATION',
                            'FOREIGN_WORKER_INFO_MAJOR': 'WORKER_MAJOR'
                            })

dfff = dfff[dfff['CASE_STATUS'] != 'Withdrawn']
print(dfff.shape)

# dfff['JOB_TITLE'].isna().sum()
dfff['JOB_TITLE'] = dfff['JOB_TITLE'].fillna((dfff['JOB_TITLE'].mode()[0]))
print(dfff['JOB_TITLE'].isna().sum())
# dfff['JOB_TITLE'].isnull().sum()
# dfff['SOC_TITLE'].isnull().sum()
# dfff['SOC_TITLE'].unique()
# dfff.dtypes


dfff['OCCUPATION'] = np.nan
# dfff.OCCUPATION
# dfff = dfff.astype({"OCCUPATION": 'str'})
dfff['SOC_TITLE'] = dfff['SOC_TITLE'].str.lower()
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('computer', 'programmer')] = 'computer occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('technical', 'designer')] = 'computer occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('software', 'web developer')] = 'computer occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('system engineer', 'system')] = 'computer occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('database', 'systems')] = 'computer occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('math', 'statistic')] = 'Mathematical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('predictive model', 'stats')] = 'Mathematical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('teacher', 'linguist')] = 'Education Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('lecturer', 'lecture')] = 'Education Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('professor', 'Teach')] = 'Education Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('school principal')] = 'Education Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('medical', 'doctor')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('physician', 'dentist')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('pharmacist', 'gastroenterologist')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('Health', 'Physical Therapists')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('psychiatrist')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('surgeon', 'nurse')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('clinical data', 'psychiatr')] = 'Medical Occupations'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('chemist', 'physicist')] = 'Advance Sciences'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('rehabilitation specialist', 'scientist')] = 'Advance Sciences'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('biology', 'chemist')] = 'Advance Sciences'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('biologi', 'clinical research')] = 'Advance Sciences'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('public relation', 'manage')] = 'Management Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('consultant', 'clerk')] = 'Management Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('management', 'operation')] = 'Management Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('mgr', 'integration')] = 'Management Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('chief', 'plan')] = 'Management Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('executive', 'project')] = 'Management Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('advertis', 'marketing')] = 'Marketing Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('promotion', 'market research')] = 'Marketing Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('business', 'business analyst')] = 'Business Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('business systems analyst')] = 'Business Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('accountant', 'finance')] = 'Financial Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('financial', 'audit')] = 'Financial Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('engineer', 'architect')] = 'Architecture & Engineering'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('surveyor', 'carto')] = 'Architecture & Engineering'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('technician', 'drafter')] = 'Architecture & Engineering'
dfff.OCCUPATION[
    dfff['SOC_TITLE'].str.contains('information security', 'information tech')] = 'Architecture & Engineering'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('food', 'cook', )] = 'Food Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('waitress', 'waiters')] = 'Food Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('restaurant')] = 'Food Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('attorney', 'law')] = 'Law Occupation'
dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('legal', 'court')] = 'Law Occupation'

dfff['OCCUPATION'] = dfff.OCCUPATION.replace(np.nan, 'Others', regex=True)

dfff['NEW_RELATED_MAJOR'] = np.nan
# dfff.NEW_RELATED_MAJOR
# dfff = dfff.astype({"NEW_RELATED_MAJOR": 'str'})
dfff['RELTD_MAJOR'] = dfff['RELTD_MAJOR'].str.lower()
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer', 'comp.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('biology', 'physics')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('information technology', 'information systems')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('science', 'sci.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer', 'engineering')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('eng.', 'engineering')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('info', 'tech.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('math', 'physics.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('data', 'chemistry.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('astr.', 'astronomy.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer', 'comp.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer', 'comp.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer', 'comp.')] = 'STEM Major'
dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer', 'engineering')] = 'STEM Major'

dfff['NEW_RELATED_MAJOR'] = dfff.NEW_RELATED_MAJOR.replace(np.nan, 'NON-STEM Major', regex=True)

dfff['NEW_WORKER_MAJOR'] = np.nan
# dfff.NEW_WORKER_MAJOR
# dfff = dfff.astype({"NEW_WORKER_MAJOR": 'str'})
dfff['WORKER_MAJOR'] = dfff['WORKER_MAJOR'].str.lower()
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('computer', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('comp.', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('biology', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('information technology', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('science', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('computer', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('eng.', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('info', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('math', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('data', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('astr.', na=False)] = 'STEM Major'

dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('stat', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('physics', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('information systems', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('sci.', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('engineering', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('engineering', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('tech.', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('physics.', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('chemistry.', na=False)] = 'STEM Major'
dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('astronomy.', na=False)] = 'STEM Major'

dfff['NEW_WORKER_MAJOR'] = dfff.NEW_WORKER_MAJOR.replace(np.nan, 'NON-STEM Major', regex=True)

dfff['NEW_EMPLOYER_NAME'] = np.nan
# dfff.NEW_EMPLOYER_NAME
# dfff = dfff.astype({"NEW_EMPLOYER_NAME": 'str'})
dfff['EMPLOYER_NAME'] = dfff['EMPLOYER_NAME'].str.upper()
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION',
                                                          'MICROSOFT CORPORATION')] = 'Top 10 Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('INTEL CORPORATION', 'FACEBOOK')] = 'Top 10 Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('INFOSYS', 'GOOGLE')] = 'Top 10 Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('AMAZON', 'AMAZON.COM')] = 'Top 10 Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('APPLE')] = 'Top 10 Employer'

dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('CISCO', 'ORACLE')] = 'TOP 10-2O Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('TATA', 'DELOITTE')] = 'TOP 10-2O Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('HCL', 'QUALCOMM')] = 'TOP 10-2O Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('CISCO', 'ORACLE')] = 'TOP 10-2O Employer'
dfff.NEW_EMPLOYER_NAME[
    dfff['EMPLOYER_NAME'].str.contains('ERNST & YOUNG', 'JP MORGAN CHASE & CO')] = 'TOP 10-2O Employer'
dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('SALESFORECE', 'WIPRO')] = 'TOP 10-2O Employer'

dfff['NEW_EMPLOYER_NAME'] = dfff.NEW_EMPLOYER_NAME.replace(np.nan, 'Other Employer', regex=True)

conditions = [

    (dfff['WAGES_OFFERED'] <= 30000),
    (dfff['WAGES_OFFERED'] > 30000) & (dfff['WAGES_OFFERED'] <= 60000),
    (dfff['WAGES_OFFERED'] > 60000) & (dfff['WAGES_OFFERED'] <= 100000),
    (dfff['WAGES_OFFERED'] > 100000) & (dfff['WAGES_OFFERED'] <= 150000),
    (dfff['WAGES_OFFERED'] > 150000) & (dfff['WAGES_OFFERED'] <= 200000),
    (dfff['WAGES_OFFERED'] > 200000)]

# create a list of the values we want to assign for each condition
values = ['Below 30000', 'Between 30000 -60000', 'Between 60000 -100000', 'Between 100000 -1500000',
          'Between 150000 -2000000', 'Above 200000']

dfff['WAGE_OFFERED'] = np.select(conditions, values)

dffff = dfff.copy()
dffff = dffff.drop(['JOB_TITLE', 'WORKER_MAJOR', 'RELTD_MAJOR', 'EMPLOYER_NAME', 'WAGES_OFFERED', 'EMPLOYER_CITY'],
                   axis=1)

# Also employer city is not that important because it is just information of employer to contact them.
# City information of the foreign worker's intended area of employment so decided to drop employer city.
dffff.info()

dffff.drop(['EMPLOYER_STATE'], axis=1, inplace=True)

# ---------------------------------------
# EDA
# ---------------------------------------

# 1. Case approved/Denial case visualization

approved_Case = []
denied_case = []

# Approved_case_each_year = df2[df2['Year']==2015][df['CASE_STATUS']=='Certified']['Year'].count()
year = [2014, 2015, 2016, 2017, 2018, 2019]

for i in year:
    Approved_case_each_year = df[df['Year'] == i][df['CASE_STATUS'] == 'Certified']['Year'].count()
    Denied_case_Each_year = df[df['Year'] == i][df['CASE_STATUS'] == 'Denied']['Year'].count()

    approved_Case.append(Approved_case_each_year)
    denied_case.append(Denied_case_Each_year)

cases_status_df = pd.DataFrame({'Year': year,
                                'Approved_Case': approved_Case,
                                'Denied_Case': denied_case})

# 2. By country

# Top 15 countries with applications
# .head()
top_15_countries = df.groupby(["COUNTRY_OF_CITIZENSHIP"], as_index=False)['CASE_STATUS'].count()
top_15_countries = top_15_countries.sort_values(by='CASE_STATUS', ascending=False).head(15)
top_15_countries = top_15_countries.rename(
    columns={'COUNTRY_OF_CITIZENSHIP': 'COUNTRY', "CASE_STATUS": 'Total applications'})

country_list = list(top_15_countries['COUNTRY'])
# country_list

# making

approved_count = []
for i in country_list:
    country_certified = df[df['COUNTRY_OF_CITIZENSHIP'] == i][df['CASE_STATUS'] == 'Certified']
    country_certified_count = country_certified.groupby(["COUNTRY_OF_CITIZENSHIP"], as_index=False)[
        'CASE_STATUS'].count()
    approved_count.append(country_certified_count.iloc[:, 1][0])

top_15_countries['Approved Case'] = approved_count

top_15_countries['Denied Case'] = top_15_countries['Total applications'] - top_15_countries['Approved Case']
top_15_countries['approved_perc'] = round(
    top_15_countries['Approved Case'] / top_15_countries['Total applications'] * 100, 2)
top_15_countries['denied_perc'] = round(top_15_countries['Denied Case'] / top_15_countries['Total applications'] * 100,
                                        2)
percent = list(top_15_countries['approved_perc'])
for x in list(top_15_countries['denied_perc']):
    percent.append(x)

top_15_countries_perc = top_15_countries.drop(['Total applications'], axis=1)
# multiple unpivot columns
top_15_countries_perc = pd.melt(top_15_countries_perc, id_vars=['COUNTRY'], value_vars=['Approved Case', 'Denied Case'])
top_15_countries_perc['value'] = percent

# --------------------------------------
# 3. By visa type
# --------------------------------------

# Top 10 countries with applications
# df.head()
top_10_visa = df.groupby(["CLASS_OF_ADMISSION"], as_index=False)['CASE_STATUS'].count()
top_10_visa = top_10_visa.sort_values(by='CASE_STATUS', ascending=False).head(10)
top_10_visa = top_10_visa.rename(columns={"CASE_STATUS": 'Total applications'})
# top_10_visa

visa_list = list(top_10_visa['CLASS_OF_ADMISSION'])
# visa_list

# making

approved_count = []
for i in visa_list:
    visa_certified = df[df['CLASS_OF_ADMISSION'] == i][df['CASE_STATUS'] == 'Certified']
    visa_certified_count = visa_certified.groupby(["CLASS_OF_ADMISSION"], as_index=False)['CASE_STATUS'].count()
    approved_count.append(visa_certified_count.iloc[:, 1][0])

top_10_visa['Approved Case'] = approved_count
top_10_visa['Denied Case'] = top_10_visa['Total applications'] - top_10_visa['Approved Case']
top_10_visa['approved_perc'] = round(top_10_visa['Approved Case'] / top_10_visa['Total applications'] * 100, 2)
top_10_visa['denied_perc'] = round(top_10_visa['Denied Case'] / top_10_visa['Total applications'] * 100, 2)
percent = list(top_10_visa['approved_perc'])
for x in list(top_10_visa['denied_perc']):
    percent.append(x)

top_10_visa_perc = top_10_visa.drop(['Total applications'], axis=1)
# multiple unpivot columns
top_10_visa_perc = pd.melt(top_10_visa_perc, id_vars=['CLASS_OF_ADMISSION'],
                           value_vars=['Approved Case', 'Denied Case'])
top_10_visa_perc['value'] = percent
# top_10_visa_perc

# -------------------------------------
# Top sponsoring employee overs different years
# --------------------------------

ax = df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:10]  # .plot.barh(width=0.9,color='#ffd700')
ax = ax.to_frame().reset_index().rename(columns={'index': 'Employer Name', 'EMPLOYER_NAME': 'Total_count'})
# ax

ax["Employer Name"] = ax["Employer Name"].replace("GOOGLE INC.", "GOOGLE LLC")

emp_year = df[df['EMPLOYER_NAME'].isin(df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:5].index)]
emp_year = emp_year.groupby(['EMPLOYER_NAME', 'Year'])['CASE_STATUS'].count().reset_index()
emp_year.pivot('Year', 'EMPLOYER_NAME', 'CASE_STATUS').plot.bar(width=0.7)


#-----------------------------
# Acceptance ratio for education level
#---------------------------------
dff_education = dfff[['REQD_EDUCATION','CASE_STATUS']]
dff_education_total = dff_education.groupby(["REQD_EDUCATION"],as_index =False)['CASE_STATUS'].count()
#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)
dff_education_total = dff_education_total.rename(columns={"CASE_STATUS":'Total applications'})
#dff_education_total



dff_education_certified = dff_education[dff_education['CASE_STATUS'] =='Certified']
dff_education_certified= dff_education_certified.groupby(["REQD_EDUCATION"],as_index =False)['CASE_STATUS'].count()
#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)
dff_education_certified = dff_education_certified.rename(columns={"CASE_STATUS":'Certified applications'})
dff_education_certified

dff_education_total['Approved_Case'] = list(dff_education_certified['Certified applications'])
dff_education_total['Denied_Case']=dff_education_total['Total applications'] - dff_education_total['Approved_Case']
dff_education_total['Approved_Ratio'] = round(dff_education_total['Approved_Case']/dff_education_total['Total applications'] *100 ,2)
dff_education_total['Denied_Ratio'] = round(dff_education_total['Denied_Case']/dff_education_total['Total applications'] *100 ,2)

#dff_education_total




#-----------------------------
# Acceptance ratio for wages category
#---------------------------------



dff_wages = dfff[['WAGE_OFFERED','CASE_STATUS']]
dff_wages_total = dff_wages.groupby(["WAGE_OFFERED"],as_index =False)['CASE_STATUS'].count()
#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)
dff_wages_total = dff_wages_total.rename(columns={"CASE_STATUS":'Total applications'})
#dff_wages_total



dff_wages_certified = dff_wages[dff_wages['CASE_STATUS'] =='Certified']
dff_wages_certified= dff_wages_certified.groupby(["WAGE_OFFERED"],as_index =False)['CASE_STATUS'].count()
#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)
dff_wages_certified = dff_wages_certified.rename(columns={"CASE_STATUS":'Certified applications'})
#dff_wages_certified

dff_wages_total['Approved_Case'] = list(dff_wages_certified['Certified applications'])
dff_wages_total['Denied_Case']=dff_wages_total['Total applications'] - dff_wages_total['Approved_Case']
dff_wages_total['Approved_Ratio'] = round(dff_wages_total['Approved_Case']/dff_wages_total['Total applications'] *100 ,2)
dff_wages_total['Denied_Ratio'] = round(dff_wages_total['Denied_Case']/dff_wages_total['Total applications'] *100 ,2)

#dff_wages_total


#-------------------------------
#Analysis of time taken for decesion
#---------------------
decesion_df = dfff[['time_taken','CASE_STATUS']]
decesion_df_total = decesion_df.groupby(["time_taken"],as_index =False)['CASE_STATUS'].count()
decesion_df_total = decesion_df_total.rename(columns={"CASE_STATUS":'Total decided cases', 'time_taken':'Time Taken'})
decesion_df_total = decesion_df_total.sort_values(by='Total decided cases', ascending= False).reset_index()
#decesion_df_total



## -----------------------------------
#### Machine Learning Implementation
##------------------------------------

dffff_down = dffff.copy()

# replace 'CERTIFIED' and 'DENIED' label of 'CASE_STATUS' respectively with '1' and '0'
dffff_down['CASE_STATUS'] = dffff_down['CASE_STATUS'].replace({'Certified': 1, 'Denied': 0})
dffff_down.CASE_STATUS.astype(int)

# type(dffff_down)
dffff_down[
    ['OCCUPATION', 'JOB_CITY', 'JOB_STATE', 'REQD_EDUCATION', 'CITIZENSHIP', 'ADMISSION_TYPE', 'WORKER_EDUCATION',
     'NEW_RELATED_MAJOR', 'NEW_WORKER_MAJOR', 'NEW_EMPLOYER_NAME', 'WAGE_OFFERED']] = dffff_down[
    ['OCCUPATION', 'JOB_CITY', 'JOB_STATE', 'REQD_EDUCATION', 'CITIZENSHIP', 'ADMISSION_TYPE', 'WORKER_EDUCATION',
     'NEW_RELATED_MAJOR', 'NEW_WORKER_MAJOR', 'NEW_EMPLOYER_NAME', 'WAGE_OFFERED']].apply(
    lambda x: x.astype('category'))

dffff_down['JOB_CITY'] = dffff_down['JOB_CITY'].str.lower()
dff_final = dffff_down.drop(['time_taken', 'SOC_TITLE', 'LANG_REQD'], axis=1)
df_final = dff_final.copy()

# label encoding
from sklearn.preprocessing import LabelEncoder

categorical_variables = {}

# Creating categories denoted by integers from column values
for col in df_final.columns:
    cat_var_name = "cat_" + col
    cat_var_name = LabelEncoder()
    print(col)
    cat_var_name.fit(df_final[col])
    df_final[col] = cat_var_name.transform(df_final[col])
    categorical_variables[col] = cat_var_name

# df_final.info()


x = df_final.drop('CASE_STATUS', axis=1)
y = df_final.CASE_STATUS
# df_final.shape


seed = 10
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=12, sampling_strategy=1)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

# making dictionary to map the label encoder from call back later
cols = list(df_final.columns)
# cols

dict_list = []
for i in range(len(cols)):
    keys = list(dff_final[cols[i]])
    values = list(df_final[cols[i]])
    # print(keys[2])
    # print(values[2])
    # print(len(keys))
    # print(len(values))

    dict_name = cols[i] + '_' + 'dict'
    # print(dict_name)

    dict_list.append(dict_name)
    dict_list[i] = {}

    for a in range(len(keys)):
        dict_list[i][keys[a]] = values[a]
# dict_list

# pickle_in = open(path+"best_model_file",'rb')
# pickle_in = open("best_model_file",'rb')
# model = pickle.load(pickle_in)

#Feature importance on basis of random forest

from sklearn.ensemble import RandomForestClassifier

final_forest = RandomForestClassifier(n_estimators = 100, max_depth = 35, max_features = 4, random_state = 123, n_jobs = -1)
final_forest.fit(x_train_res, y_train_res)

feature_scores = pd.Series(final_forest.feature_importances_, index=x_train_res.columns).sort_values(ascending=False)

fig  = go.Figure([go.Bar(y = feature_scores, x=feature_scores.index, marker_color = 'blue',orientation='v')])
                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots


fig.update_layout(title = 'Visualization of Feature Importance Score ',
                 yaxis_title = 'Feature importance score',
                 xaxis_title = 'Features')
                 #barmode = 'group')

fig.show()


# ---------------------
# Dashboard
# ---------------------------


# Navigation bar

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY], )
server = app.server

# "https://en.wikipedia.org/wiki/United_States_Department_of_Labor#/media/File:Seal_of_the_United_States_Department_of_Labor.svg" #reference: https://twitter.com/USDOL

# DOL_Logo = mpimg.imread('/Users/ujjwaloli/Desktop/Capstone Project/DashboardLogo.png')


blob2 = bucket.get_blob(f"DashboardLogo.png")

bt2 = blob2.download_as_bytes()

DOL_Logo = Image.open(BytesIO(bt2))

# img = Image.open(BytesIO(blob.download_as_bytes()))


total = len(df.index)
certified = df[df['CASE_STATUS'] == 'Certified']['CASE_STATUS'].count()
denied = df[df['CASE_STATUS'] == 'Denied']['CASE_STATUS'].count()
certified_perc = round(certified / total * 100, 2)
denied_perc = round(denied / total * 100, 2)


def line_plot(df):
    fig = go.Figure(data=[
        go.Scatter(y=cases_status_df['Approved_Case'], x=cases_status_df['Year'], line=dict(color='firebrick', width=4),
                   text=cases_status_df['Approved_Case'], name='Cases Approved'),
        go.Scatter(y=cases_status_df['Denied_Case'], x=cases_status_df['Year'], line=dict(color='blue', width=4),
                   text=cases_status_df['Denied_Case'], name='Cases Denied')])

    fig.update_layout(title='Analysis of Approved cases and Denied Cases over the years',
                      xaxis_title='Year',
                      yaxis_title='No of cases',
                      margin=dict(l=4, r=4, t=30, b=4))
    return fig


def lolipop_plot(decesion_df_total):
    fig1 = go.Figure()
    # Draw points
    fig1.add_trace(go.Scatter(x=decesion_df_total["Total decided cases"],
                              y=decesion_df_total["Time Taken"],
                              mode='markers',
                              marker_color='darkblue',
                              marker_size=10))
    # Draw lines
    for i in range(0, len(decesion_df_total)):
        print(decesion_df_total["Total decided cases"][i])
        fig1.add_shape(type='line', x0=0, y0=i,
                       x1=decesion_df_total["Total decided cases"][i],
                       y1=i,
                       line=dict(color='crimson', width=3))
    # Set title
    fig1.update_layout(title_text=
                       "Analysis of time taken for cases to decided",
                       title_font_size=20)
    # Set x-axes range
    fig1.update_xaxes(title='Number of decided applications', range=[0, 220000])
    fig1.update_yaxes(title='Time taken')

    return fig1


def country_bar_plot(top_15_countries):
    fig = go.Figure(
        [go.Bar(x=top_15_countries['COUNTRY'], y=top_15_countries['Total applications'], marker_color='indianred')])
    # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

    fig.update_layout(title='Top 15 countries with high perm applications',
                      xaxis_title='Countries',
                      yaxis_title='Number of perm applications',
                      margin=dict(l=4, r=4, t=30, b=4))
    # barmode = 'group')

    return fig


def visa_bar_plot(top_10_visa):
    fig = go.Figure(
        [go.Bar(x=top_10_visa['CLASS_OF_ADMISSION'], y=top_10_visa['Total applications'], marker_color='blue')])
    # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

    fig.update_layout(title='Top 10 class of admissions with high perm applications',
                      xaxis_title='Countries',
                      yaxis_title='Number of class of admissions',
                      margin=dict(l=4, r=4, t=30, b=4))
    # barmode = 'group')

    return fig


def top_employer(ax):
    # print(ax['Total_count'])

    fig = go.Figure([go.Bar(x=ax['Total_count'], y=ax['Employer Name'], marker_color='green', orientation='h')])
    # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

    fig.update_layout(title='Top sponsoring employers ',
                      xaxis_title='PERM applications',
                      yaxis_title='Name of employers')
    # barmode = 'group'

    return fig


def top_employer_by_year(emp_year):
    fig = px.histogram(emp_year, x="Year", y="CASE_STATUS",
                       color='EMPLOYER_NAME', barmode='group',
                       # histfunc='avg',
                       height=500, title="Top Employers(sponsors) over the years")

    return fig


def data_for_cases(header, total_cases, percent):
    card_content = [
        dbc.CardHeader(header),

        dbc.CardBody(
            [
                dcc.Markdown(dangerously_allow_html=True,
                             children=["{0}<br><sub>+{1}</sub></br>".format(total_cases, percent)]
                             )
            ]
        )

    ]
    return card_content


body_app = dbc.Container([

    dbc.Row(html.Marquee("India is the top conuntry in terms of certified cases"), style={'color': 'green'}),
    dbc.Row([
        dbc.Col(dbc.Card(data_for_cases("Certified", f'{certified:,}', f'{certified_perc:,}'), color='success',
                         style={'text-align': 'center'}, inverse=True)),
        dbc.Col(dbc.Card(data_for_cases("Denied", f'{denied:,}', f'{denied_perc:,}'), color='danger',
                         style={'text-align': 'center'}, inverse=True)),
        # dbc.Col(dbc.Card(card_content, color='secondary', outline=True))

    ]),

    html.Br(),

    dbc.Row([html.Div(html.H5('Analysis & Visualizations'),
                      style={'textAlign': 'center', 'fontWeight': 'bold', 'family': 'georgia'})]),

    html.Br(),

    #dbc.Row(dcc.Graph(id='line_plot', figure=line_plot(cases_status_df)), style={'height': '450px'}),
    dbc.Row([dbc.Col(dcc.Graph(id='line_plot', figure=line_plot(cases_status_df)), style={'height': '450px'}),
             # dbc.Col(html.Div(), style={'height':'450px'})
             dbc.Col(dcc.Graph(id='lolipop-plot', figure=lolipop_plot(decesion_df_total)), style={'height': '450px'})

             ]),

    html.Br(),

    html.Br(),

    dbc.Row([html.Div(html.H4('Case status analysis by country with -- %'))]),

    html.Br(),

    dbc.Row([dbc.Col(dcc.Graph(id='country_bar_plot', figure=country_bar_plot(top_15_countries)),
                     style={'height': '450px'}),
             # dbc.Col(html.Div(), style={'height':'450px'})
             dbc.Col([html.Div(id='dropdown_div', children=[
                 dcc.Dropdown(id='country-dropdown',  # this is an input
                              options=[{'label': i, 'value': i} for i in top_15_countries_perc['COUNTRY'].unique()],
                              value='INDIA',
                              placeholder='Select the country')], style={'width': '100%', 'display': 'inline-block'}),

                      # html.Div(id ='pie-chart', children=[ #this is an output
                      dcc.Graph(id='pie-plot')], style={'height': '450px', 'width': '300px'})

             ]),

    html.Br(),

    html.Br(),

    dbc.Row([html.Div(html.H4('Case status analysis by visa type with -- %'))]),

    html.Br(),

    dbc.Row([dbc.Col(dcc.Graph(id='visa_bar_plot', figure=visa_bar_plot(top_10_visa)), style={'height': '450px'}),
             dbc.Col([html.Div(id='dropdown_div2', children=[
                 dcc.Dropdown(id='visa-dropdown',  # this is an input
                              options=[{'label': i, 'value': i} for i in
                                       top_10_visa_perc['CLASS_OF_ADMISSION'].unique()],
                              value='H1B',
                              placeholder='Select the visa type')], style={'width': '100%', 'display': 'inline-block'}),

                      # html.Div(id ='pie-chart', children=[ #this is an output
                      dcc.Graph(id='visa-pie-plot')], style={'height': '450px', 'width': '300px'})

             ]),

    html.Br(),

    html.Br(),

    dbc.Row(dcc.Graph(id='top_employer', figure=top_employer(ax)), style={'height': '450px'}),

    html.Br(),

    dbc.Row(dcc.Graph(id='top_employer_by_year', figure=top_employer_by_year(emp_year)), style={'height': '450px'}),

    html.Br(),

    html.Br(),

    dbc.Row([html.Div(html.H4("Case status analysis by education level and wage level"))]),

    html.Br(),

    dbc.Row([dbc.Col([html.Div(id='education_div', children=[
                dcc.RadioItems(id='certification_edu_ratio',  # this is an input
                       options=[{'label': 'Certified', 'value': 'Certified'},
                                {'label': 'Denied', 'value': 'Denied'}],
                       value='Certified'),
        # html.Div(id ='pie-chart', children=[ #this is an output
                dcc.Graph(id='education_bar_plot')], style={'height': '450px', 'width': '800px'})])
                ]),

    html.Br(),

    dbc.Row([dbc.Col([html.Div(id='wages_div', children=[
                dcc.RadioItems(id='certification_wage_radio',  # this is an input
                                options=[{'label': 'Certified', 'value': 'Certified'},
                                         {'label': 'Denied', 'value': 'Denied'}],
                                value='Denied'),

                 # html.Div(id ='pie-chart', children=[ #this is an output
                 dcc.Graph(id='wage_bar_plot')], style={'height': '450px', 'width': '800px'})])

             ]),


    html.Br(),

    html.Br(),

    dbc.Row([html.Div(html.H4('Prediction based on model'))]),

    html.Br(),

    dbc.Row([html.Div(dcc.Markdown('''
    #### Note: For your employer check following condition:


    '1.If your employer are:    *COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION,   MICROSOFT CORPORATION,   INTEL CORPORATION,
                     AMAZON CORPORATE LLC,  GOOGLE LLC,   FACEBOOK INC.,  APPLE INC.,   INFOSYS LTD.,
                'GOOGLE LLC,  AMAZON.COM SERVICES INC.  * then select ,  **Top 10 Employer**


    2.If your employer are:  *CISCO SYSTEMS, INC.,
                             'ORACLE AMERICA, INC.,
                             'HCL AMERICA INC.,
                            ' TATA CONSULTANCY SERVICES LIMITED,
                            ' DELOITTE CONSULTING LLP,
                             'QUALCOMM TECHNOLOGIES INC.,
                            ' ERNST & YOUNG U.S. LLP,
                             'JP MORGAN CHASE & CO,
                             'WIPRO LIMITED,
                             'Defender Services, Inc.* then select **Top 20 Employer**



    3. If your employer are not in any of above list select   **Other Employer**'''

                                   ))]),

    html.Br(),

    html.Br(),
    ## try again
    dbc.Row([dbc.Col([
        dbc.Row([html.Div(html.H6('Select CITY where job is located'))]),
        dbc.Row([html.Div(id='job_city', children=[
            dcc.Dropdown(id='job_city_dropdown',  # this is an input
                         options=[{'label': i, 'value': i} for i in dff_final['JOB_CITY'].unique()],
                         value='new york',
                         placeholder='Select CITY where job is located')],
                          style={'width': '100%', 'display': 'inline-block'})])
    ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select STATE where job is located'))]),
            dbc.Row([html.Div(id='job_state', children=[
                dcc.Dropdown(id='job_state_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['JOB_STATE'].unique()],
                             value='NEW YO',
                             placeholder='Select STATE where job is located ')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select education requirement needed for the job'))]),
            dbc.Row([html.Div(id='required_education', children=[
                dcc.Dropdown(id='recquired_education_type_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['REQD_EDUCATION'].unique()],
                             value='High School',
                             placeholder='Select Required Education type for the job as input')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

    ]),

    html.Br(),

    html.Br(),

    dbc.Row([dbc.Col([
        dbc.Row([html.Div(html.H6('Select Country where you belong to!!'))]),
        dbc.Row([html.Div(id='citizenship_type', children=[
            dcc.Dropdown(id='citizenship_type_dropdown',  # this is an input
                         options=[{'label': i, 'value': i} for i in dff_final['CITIZENSHIP'].unique()],
                         value='INDIA',
                         placeholder='Select which country are you from?')],
                          style={'width': '100%', 'display': 'inline-block'})])
    ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select which visa/admission you have'))]),
            dbc.Row([html.Div(id='admission_type', children=[
                dcc.Dropdown(id='admission_type_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['ADMISSION_TYPE'].unique()],
                             value='H1B',
                             placeholder='Select your visa type as input')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select your highest education level'))]),
            dbc.Row([html.Div(id='worker_education', children=[
                dcc.Dropdown(id='worker_education_type_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['WORKER_EDUCATION'].unique()],
                             value='High School',
                             placeholder='Select employee education')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

    ]),

    html.Br(),

    html.Br(),

    dbc.Row([dbc.Col([
        dbc.Row([html.Div(html.H6('Select if the job requires training or not?'))]),
        dbc.Row([html.Div(id='training_reqd', children=[
            dcc.Dropdown(id='training_reqd_dropdown',  # this is an input
                         options=[{'label': i, 'value': i} for i in dff_final['TRAINING_REQD'].unique()],
                         value='N',
                         placeholder='Select Y for Yes and N for No')],
                          style={'width': '100%', 'display': 'inline-block'})])
    ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select occupation that job belongs to'))]),
            dbc.Row([html.Div(id='occupation', children=[
                dcc.Dropdown(id='occupation_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['OCCUPATION'].unique()],
                             value='computer occupations',
                             placeholder='Select job from dropdown')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select if the work major requires STEM or NON-STEM'))]),
            dbc.Row([html.Div(id='related_major', children=[
                dcc.Dropdown(id='related_major_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['NEW_RELATED_MAJOR'].unique()],
                             value='STEM Major',
                             placeholder='Select if your major is STEM ')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

    ]),

    html.Br(),

    html.Br(),

    dbc.Row([dbc.Col([
        dbc.Row([html.Div(html.H6('Select if your major is STEM or NON-STEM'))]),
        dbc.Row([html.Div(id='worker_major', children=[
            dcc.Dropdown(id='worker_major_type_dropdown',  # this is an input
                         options=[{'label': i, 'value': i} for i in dff_final['NEW_WORKER_MAJOR'].unique()],
                         value='STEM Major',
                         placeholder='Select if your major is STEM')],
                          style={'width': '100%', 'display': 'inline-block'})])
    ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select if your employer is Top-10, Top20 or Other. Refer to above!!'))]),
            dbc.Row([html.Div(id='employer_name', children=[
                dcc.Dropdown(id='employer_name_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['NEW_EMPLOYER_NAME'].unique()],
                             value='Top 10 Employer',
                             placeholder='Select your employer')], style={'width': '100%', 'display': 'inline-block'})])
        ]),

        dbc.Col([
            dbc.Row([html.Div(html.H6('Select the wage offered category'))]),
            dbc.Row([html.Div(id='wage_offered', children=[
                dcc.Dropdown(id='wage_offered_dropdown',  # this is an input
                             options=[{'label': i, 'value': i} for i in dff_final['WAGE_OFFERED'].unique()],
                             value='Below 30000',
                             placeholder='Select your wage category')],
                              style={'width': '100%', 'display': 'inline-block'})])
        ]),

    ]),

    ## try finish

    html.Br(),

    dbc.Row([
        dbc.Col(dbc.Button(id='generate_ans', children='Generate Prediction', color='dark', n_clicks=0),
                width={'size': 15, 'offset': 3}),

    ]),

    html.Br(),

    html.Br(),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id='Load-ans',
                type='default',
                children=html.Div(id='div_answer', style={'textAlign': 'center', 'color': 'black', 'height': '50px',
                                                          'fontWeight': 'bold'})
            )
        ),

    ])

], fluid=True)
navbar = dbc.Navbar(id='navbar', children=[

    html.A(
        dbc.Row([
            dbc.Col(html.Img(src=DOL_Logo, height="70px")),
            dbc.Col(
                dbc.NavbarBrand("Perm Cases Tracker",
                                style={'color': 'black', 'fontSize': '25px', 'fontFamily': 'Times New Roman'})

            )

        ], align="center"),  # aligns title to center
        # no_gutters=True),
        href='/'
    ),
    dbc.Button(id='button', children='Clicke Me!', color='primary', className='ml-auto', href='/')

])

app.layout = html.Div(id='parent', children=[navbar, body_app])


@app.callback(Output(component_id='div_answer', component_property='children'),
              [Input(component_id='generate_ans', component_property='n_clicks')],
              [State(component_id='job_city_dropdown', component_property='value'),
               State('job_state_dropdown', 'value'),
               State('recquired_education_type_dropdown', 'value'),
               State('citizenship_type_dropdown', 'value'),
               State('admission_type_dropdown', 'value'),
               State('worker_education_type_dropdown', 'value'),
               State('training_reqd_dropdown', 'value'),
               State('occupation_dropdown', 'value'),
               State('related_major_dropdown', 'value'),
               State('worker_major_type_dropdown', 'value'),
               State('employer_name_dropdown', 'value'),
               State('wage_offered_dropdown', 'value')])
def show_answer(clicks, job_city, job_state, required_education, citizenship_type, admission_type, worker_education,
                training_reqd, occupation, related_major, worker_major, employer_name, wage_offered):
    print("*****************############################")
    print("Job_city is", job_city)

    if clicks > 0:
        value_1 = dict_list[1][str(job_city)]
        value_2 = dict_list[2][str(job_state)]
        value_3 = dict_list[3][str(required_education)]
        value_4 = dict_list[4][str(training_reqd)]
        value_5 = dict_list[5][str(citizenship_type)]
        value_6 = dict_list[6][str(admission_type)]
        value_7 = dict_list[7][str(worker_education)]
        value_8 = dict_list[8][str(occupation)]
        value_9 = dict_list[9][str(related_major)]
        value_10 = dict_list[10][str(worker_major)]
        value_11 = dict_list[11][str(employer_name)]
        value_12 = dict_list[12][str(wage_offered)]

        test_series = pd.Series({'JOB_CITY': value_1,
                                 'JOB_STATE': value_2,
                                 'REQD_EDUCATION': value_3,
                                 'TRAINING_REQD': value_4,
                                 'CITIZENSHIP': value_5,
                                 'ADMISSION_TYPE': value_6,
                                 'WORKER_EDUCATION': value_7,
                                 'OCCUPATION': value_8,
                                 'NEW_RELATED_MAJOR': value_9,
                                 'NEW_WORKER_MAJOR': value_10,
                                 'NEW_EMPLOYER_NAME': value_11,
                                 'WAGE_OFFERED': value_12})

        blob3 = bucket.blob("best_model_file")
        # download that file and name it 'local.joblib'

        with TemporaryFile() as temp_file:
            # download blob into temp file
            blob3.download_to_file(temp_file)
            temp_file.seek(0)
            # load into joblib
            model = pickle.load(temp_file)

        # pickle_in = open(bt2, 'rb')
        # model = pickle.load(pickle_in)

        # model = RandomForestClassifier(n_estimators = 100, max_depth = 35, max_features = 4, random_state = 123, n_jobs = -1)

        prediction_model = model.predict([test_series])

        print(prediction_model)

        for i in range(len(prediction_model)):
            if prediction_model[i] == 1:
                ans = "Congratulations!! based on the criteria you entered, there is likelihood that your case will be Certified"
            else:
                ans = "Unfortunately based on the criteria you entered, there is likelihood that your case will be Denied"


        print(ans)
        return ans
    else:
        return ""


@app.callback(Output(component_id='education_bar_plot', component_property='figure'),
              [Input(component_id='certification_edu_ratio', component_property='value')])
def generate_education_bar(status):
    colors = ['blueviolet', 'blue', 'lightskyblue', 'lightsteelblue', 'mediumblue', 'aqua', 'midnightblue']

    if status == 'Certified':

        dff_education_approved = dff_education_total.sort_values(by='Approved_Ratio', ascending=False)

        fig = go.Figure([go.Bar(x=dff_education_approved['REQD_EDUCATION'], y=dff_education_approved['Approved_Ratio'],
                                marker_color=colors)])
        # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

        fig.update_layout(title='Certified cases ratio by education level',
                          xaxis_title='Education level',
                          yaxis_title='Approved ratio')
        # barmode = 'group')

        return fig

    elif status == 'Denied':
        dff_education_denied = dff_education_total.sort_values(by='Denied_Ratio', ascending=False)

        fig = go.Figure([go.Bar(x=dff_education_denied['REQD_EDUCATION'], y=dff_education_denied['Denied_Ratio'],
                                marker_color=colors)])
        # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

        fig.update_layout(title='Denied cases ratio by education level',
                          xaxis_title='Education level',
                          yaxis_title='Denied ratio')
        # barmode = 'group')

        return fig


@app.callback(Output(component_id='wage_bar_plot', component_property='figure'),
              [Input(component_id='certification_wage_radio', component_property='value')])
def generate_wage_plot(status_wage):
    colors = ['blueviolet', 'blue', 'lightskyblue', 'lightsteelblue', 'mediumblue', 'aqua', 'midnightblue']

    print(status_wage)
    if status_wage == 'Certified':

        dff_wages_approved = dff_wages_total.sort_values(by='Approved_Ratio', ascending=False)
        # dff_wages_approved

        fig = go.Figure(
            [go.Bar(x=dff_wages_approved['WAGE_OFFERED'], y=dff_wages_approved['Approved_Ratio'], marker_color=colors)])
        # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

        fig.update_layout(title='Approved cases ratio by wage category',
                          xaxis_title='Wages cateogry',
                          yaxis_title='Approved ratio')
        # barmode = 'group')

        return fig

    elif status_wage == 'Denied':

        dff_wages_denied = dff_wages_total.sort_values(by='Denied_Ratio', ascending=False)

        fig = go.Figure(
            [go.Bar(x=dff_wages_denied['WAGE_OFFERED'], y=dff_wages_denied['Denied_Ratio'], marker_color=colors)])
        # go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots

        fig.update_layout(title='Denied cases ratio by wage category',
                          xaxis_title='Wages cateogry',
                          yaxis_title='Denied ratio')
        # barmode = 'group')

        return fig


@app.callback(Output(component_id='pie-plot', component_property='figure'),
              [Input(component_id='country-dropdown', component_property='value')])
def generate_pie(country):
    print("Country name is ", country)

    df_final = top_15_countries_perc.loc[top_15_countries_perc['COUNTRY'] == '{}'.format(country)]

    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])

    return fig


@app.callback(Output(component_id='visa-pie-plot', component_property='figure'),
              [Input(component_id='visa-dropdown', component_property='value')])
def generate_visa_pie(visa_dropdown):
    df_final = top_10_visa_perc.loc[top_10_visa_perc['CLASS_OF_ADMISSION'] == '{}'.format(visa_dropdown)]

    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])

    return fig


if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=8080)

