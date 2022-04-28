{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1630ecc4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: imblearn in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (0.0)\n",
      "Requirement already satisfied: imbalanced-learn in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from imblearn) (0.9.0)\n",
      "Requirement already satisfied: joblib>=0.11 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from imbalanced-learn->imblearn) (1.1.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from imbalanced-learn->imblearn) (2.2.0)\n",
      "Requirement already satisfied: numpy>=1.14.6 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from imbalanced-learn->imblearn) (1.20.3)\n",
      "Requirement already satisfied: scipy>=1.1.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from imbalanced-learn->imblearn) (1.7.1)\n",
      "Requirement already satisfied: scikit-learn>=1.0.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from imbalanced-learn->imblearn) (1.0.2)\n",
      "Requirement already satisfied: pandas in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (1.3.4)\n",
      "Requirement already satisfied: python-dateutil>=2.7.3 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2017.3 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from pandas) (2021.3)\n",
      "Requirement already satisfied: numpy>=1.17.3 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from pandas) (1.20.3)\n",
      "Requirement already satisfied: six>=1.5 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)\n",
      "Requirement already satisfied: jupyter-dash in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (0.4.0)\n",
      "Requirement already satisfied: retrying in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (1.3.3)\n",
      "Requirement already satisfied: flask in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (1.1.2)\n",
      "Requirement already satisfied: ansi2html in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (1.7.0)\n",
      "Requirement already satisfied: ipython in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (7.29.0)\n",
      "Requirement already satisfied: ipykernel in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (6.4.1)\n",
      "Requirement already satisfied: dash in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (2.1.0)\n",
      "Requirement already satisfied: requests in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-dash) (2.26.0)\n",
      "Requirement already satisfied: dash-html-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->jupyter-dash) (2.0.0)\n",
      "Requirement already satisfied: dash-core-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->jupyter-dash) (2.0.0)\n",
      "Requirement already satisfied: plotly>=5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->jupyter-dash) (5.5.0)\n",
      "Requirement already satisfied: flask-compress in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->jupyter-dash) (1.10.1)\n",
      "Requirement already satisfied: dash-table==5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->jupyter-dash) (5.0.0)\n",
      "Requirement already satisfied: Jinja2>=2.10.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask->jupyter-dash) (2.11.3)\n",
      "Requirement already satisfied: Werkzeug>=0.15 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask->jupyter-dash) (2.0.2)\n",
      "Requirement already satisfied: click>=5.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask->jupyter-dash) (8.0.3)\n",
      "Requirement already satisfied: itsdangerous>=0.24 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask->jupyter-dash) (2.0.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Jinja2>=2.10.1->flask->jupyter-dash) (1.1.1)\n",
      "Requirement already satisfied: six in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash->jupyter-dash) (1.16.0)\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash->jupyter-dash) (8.0.1)\n",
      "Requirement already satisfied: brotli in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask-compress->dash->jupyter-dash) (1.0.9)\n",
      "Requirement already satisfied: ipython-genutils in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (0.2.0)\n",
      "Requirement already satisfied: tornado<7.0,>=4.2 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (6.1)\n",
      "Requirement already satisfied: traitlets<6.0,>=4.1.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (5.1.0)\n",
      "Requirement already satisfied: debugpy<2.0,>=1.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (1.4.1)\n",
      "Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (0.1.2)\n",
      "Requirement already satisfied: jupyter-client<8.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (6.1.12)\n",
      "Requirement already satisfied: appnope in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter-dash) (0.1.2)\n",
      "Requirement already satisfied: jedi>=0.16 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (0.18.0)\n",
      "Requirement already satisfied: pexpect>4.3 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (4.8.0)\n",
      "Requirement already satisfied: pygments in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (2.10.0)\n",
      "Requirement already satisfied: setuptools>=18.5 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (58.0.4)\n",
      "Requirement already satisfied: decorator in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (5.1.0)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (3.0.20)\n",
      "Requirement already satisfied: backcall in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (0.2.0)\n",
      "Requirement already satisfied: pickleshare in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from ipython->jupyter-dash) (0.7.5)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jedi>=0.16->ipython->jupyter-dash) (0.8.2)\n",
      "Requirement already satisfied: jupyter-core>=4.6.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-client<8.0->ipykernel->jupyter-dash) (4.8.1)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-client<8.0->ipykernel->jupyter-dash) (2.8.2)\n",
      "Requirement already satisfied: pyzmq>=13 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from jupyter-client<8.0->ipykernel->jupyter-dash) (22.2.1)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from pexpect>4.3->ipython->jupyter-dash) (0.7.0)\n",
      "Requirement already satisfied: wcwidth in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->jupyter-dash) (0.2.5)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from requests->jupyter-dash) (1.26.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from requests->jupyter-dash) (2021.10.8)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from requests->jupyter-dash) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from requests->jupyter-dash) (3.2)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: dash in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (2.1.0)\n",
      "Requirement already satisfied: dash-table==5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash) (5.0.0)\n",
      "Requirement already satisfied: plotly>=5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash) (5.5.0)\n",
      "Requirement already satisfied: dash-core-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash) (2.0.0)\n",
      "Requirement already satisfied: flask-compress in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash) (1.10.1)\n",
      "Requirement already satisfied: Flask>=1.0.4 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash) (1.1.2)\n",
      "Requirement already satisfied: dash-html-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash) (2.0.0)\n",
      "Requirement already satisfied: itsdangerous>=0.24 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash) (2.0.1)\n",
      "Requirement already satisfied: click>=5.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash) (8.0.3)\n",
      "Requirement already satisfied: Werkzeug>=0.15 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash) (2.0.2)\n",
      "Requirement already satisfied: Jinja2>=2.10.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash) (2.11.3)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Jinja2>=2.10.1->Flask>=1.0.4->dash) (1.1.1)\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash) (8.0.1)\n",
      "Requirement already satisfied: six in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash) (1.16.0)\n",
      "Requirement already satisfied: brotli in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask-compress->dash) (1.0.9)\n",
      "Requirement already satisfied: plotly in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (5.5.0)\n",
      "Requirement already satisfied: six in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly) (1.16.0)\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly) (8.0.1)\n",
      "Requirement already satisfied: dash-bootstrap-components in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (1.0.3)\n",
      "Requirement already satisfied: dash>=2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash-bootstrap-components) (2.1.0)\n",
      "Requirement already satisfied: dash-table==5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash>=2.0.0->dash-bootstrap-components) (5.0.0)\n",
      "Requirement already satisfied: flask-compress in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash>=2.0.0->dash-bootstrap-components) (1.10.1)\n",
      "Requirement already satisfied: dash-core-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash>=2.0.0->dash-bootstrap-components) (2.0.0)\n",
      "Requirement already satisfied: plotly>=5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash>=2.0.0->dash-bootstrap-components) (5.5.0)\n",
      "Requirement already satisfied: dash-html-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash>=2.0.0->dash-bootstrap-components) (2.0.0)\n",
      "Requirement already satisfied: Flask>=1.0.4 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash>=2.0.0->dash-bootstrap-components) (1.1.2)\n",
      "Requirement already satisfied: Werkzeug>=0.15 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash>=2.0.0->dash-bootstrap-components) (2.0.2)\n",
      "Requirement already satisfied: itsdangerous>=0.24 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash>=2.0.0->dash-bootstrap-components) (2.0.1)\n",
      "Requirement already satisfied: Jinja2>=2.10.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash>=2.0.0->dash-bootstrap-components) (2.11.3)\n",
      "Requirement already satisfied: click>=5.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash>=2.0.0->dash-bootstrap-components) (8.0.3)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Jinja2>=2.10.1->Flask>=1.0.4->dash>=2.0.0->dash-bootstrap-components) (1.1.1)\n",
      "Requirement already satisfied: six in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash>=2.0.0->dash-bootstrap-components) (1.16.0)\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash>=2.0.0->dash-bootstrap-components) (8.0.1)\n",
      "Requirement already satisfied: brotli in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask-compress->dash>=2.0.0->dash-bootstrap-components) (1.0.9)\n",
      "Requirement already satisfied: dash-cytoscape in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (0.3.0)\n",
      "Requirement already satisfied: dash in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash-cytoscape) (2.1.0)\n",
      "Requirement already satisfied: dash-html-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->dash-cytoscape) (2.0.0)\n",
      "Requirement already satisfied: Flask>=1.0.4 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->dash-cytoscape) (1.1.2)\n",
      "Requirement already satisfied: dash-core-components==2.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->dash-cytoscape) (2.0.0)\n",
      "Requirement already satisfied: dash-table==5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->dash-cytoscape) (5.0.0)\n",
      "Requirement already satisfied: plotly>=5.0.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->dash-cytoscape) (5.5.0)\n",
      "Requirement already satisfied: flask-compress in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from dash->dash-cytoscape) (1.10.1)\n",
      "Requirement already satisfied: Werkzeug>=0.15 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash->dash-cytoscape) (2.0.2)\n",
      "Requirement already satisfied: Jinja2>=2.10.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash->dash-cytoscape) (2.11.3)\n",
      "Requirement already satisfied: itsdangerous>=0.24 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash->dash-cytoscape) (2.0.1)\n",
      "Requirement already satisfied: click>=5.1 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Flask>=1.0.4->dash->dash-cytoscape) (8.0.3)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from Jinja2>=2.10.1->Flask>=1.0.4->dash->dash-cytoscape) (1.1.1)\n",
      "Requirement already satisfied: six in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash->dash-cytoscape) (1.16.0)\n",
      "Requirement already satisfied: tenacity>=6.2.0 in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from plotly>=5.0.0->dash->dash-cytoscape) (8.0.1)\n",
      "Requirement already satisfied: brotli in /Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages (from flask-compress->dash->dash-cytoscape) (1.0.9)\n"
     ]
    }
   ],
   "source": [
    "!pip install imblearn\n",
    "!pip install pandas\n",
    "!pip install jupyter-dash\n",
    "!pip install dash\n",
    "!pip install plotly\n",
    "!pip install dash-bootstrap-components\n",
    "!pip install dash-cytoscape\n",
    "!pip install pyngrok --quiet\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "429a107a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/2608882513.py:18: UserWarning: \n",
      "The dash_html_components package is deprecated. Please replace\n",
      "`import dash_html_components as html` with `from dash import html`\n",
      "  import dash_html_components as html\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/2608882513.py:19: UserWarning: \n",
      "The dash_core_components package is deprecated. Please replace\n",
      "`import dash_core_components as dcc` with `from dash import dcc`\n",
      "  import dash_core_components as dcc\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/2608882513.py:22: UserWarning: \n",
      "The dash_table package is deprecated. Please replace\n",
      "`import dash_table` with `from dash import dash_table`\n",
      "\n",
      "Also, if you're using any of the table format helpers (e.g. Group), replace \n",
      "`from dash_table.Format import Group` with \n",
      "`from dash.dash_table.Format import Group`\n",
      "  import dash_table\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split \n",
    "from sklearn.linear_model import LogisticRegression\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas      as pd\n",
    "import numpy       as np\n",
    "import seaborn as sns\n",
    "import re\n",
    "import scipy.stats as stats\n",
    "import statsmodels.api as sm\n",
    "import statsmodels.formula.api as smf\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.graph_objects as go\n",
    "import plotly.express as px\n",
    "import dash\n",
    "import dash_html_components as html\n",
    "import dash_core_components as dcc\n",
    "import dash_bootstrap_components as dbc\n",
    "from datetime import datetime as dt\n",
    "import dash_table\n",
    "from dash.dependencies import Input, Output, State\n",
    "import matplotlib.image as mpimg\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import warnings\n",
    "#%tensorflow_Version 2.x\n",
    "#import tensorflow as tf\n",
    "#from tensorflow import keras\n",
    "#from plotly.offline import init_notebook_mode, plot\n",
    "#init_notebook_mode()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c2d8dab2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from urllib.request import urlopen\n",
    "import requests\n",
    "#url ='https://www.kaggle.com/datasets/ujjoli/permcases/final_data'\n",
    "#data = requests.get(url)\n",
    "df = pd.read_excel('final_data.xlsx')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5c811074",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['CASE_NUMBER', 'DECISION_DATE', 'CASE_STATUS', 'CASE_RECEIVED_DATE',\n",
      "       'REFILE', 'EMPLOYER_NAME', 'EMPLOYER_CITY', 'EMPLOYER_STATE',\n",
      "       'FW_OWNERSHIP_INTEREST', 'PW_SOC_TITLE', 'PW_AMOUNT_9089',\n",
      "       'PW_UNIT_OF_PAY_9089', 'JOB_INFO_WORK_CITY', 'JOB_INFO_WORK_STATE',\n",
      "       'JOB_INFO_JOB_TITLE', 'JOB_INFO_EDUCATION', 'JOB_INFO_EDUCATION_OTHER',\n",
      "       'JOB_INFO_MAJOR', 'JOB_INFO_TRAINING', 'JOB_INFO_FOREIGN_LANG_REQ',\n",
      "       'COUNTRY_OF_CITIZENSHIP', 'CLASS_OF_ADMISSION',\n",
      "       'FOREIGN_WORKER_INFO_EDUCATION', 'FW_INFO_EDUCATION_OTHER',\n",
      "       'FOREIGN_WORKER_INFO_MAJOR', 'PW_Job_Title_9089'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "def abt_func():\n",
    "    k = df.head\n",
    "    z = df.columns\n",
    "    \n",
    "    return z\n",
    "\n",
    "z = abt_func()\n",
    "print(z)\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "5be4e65b",
   "metadata": {},
   "source": [
    "\n",
    "DOL Visualization Project\n",
    "1. First we create navbar\n",
    "2. Then we create card"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f108424",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4b89c64e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CASE_NUMBER</th>\n",
       "      <th>DECISION_DATE</th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>CASE_RECEIVED_DATE</th>\n",
       "      <th>REFILE</th>\n",
       "      <th>EMPLOYER_NAME</th>\n",
       "      <th>EMPLOYER_CITY</th>\n",
       "      <th>EMPLOYER_STATE</th>\n",
       "      <th>FW_OWNERSHIP_INTEREST</th>\n",
       "      <th>PW_SOC_TITLE</th>\n",
       "      <th>...</th>\n",
       "      <th>JOB_INFO_MAJOR</th>\n",
       "      <th>JOB_INFO_TRAINING</th>\n",
       "      <th>JOB_INFO_FOREIGN_LANG_REQ</th>\n",
       "      <th>COUNTRY_OF_CITIZENSHIP</th>\n",
       "      <th>CLASS_OF_ADMISSION</th>\n",
       "      <th>FOREIGN_WORKER_INFO_EDUCATION</th>\n",
       "      <th>FW_INFO_EDUCATION_OTHER</th>\n",
       "      <th>FOREIGN_WORKER_INFO_MAJOR</th>\n",
       "      <th>PW_Job_Title_9089</th>\n",
       "      <th>Year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>A-14220-96665</td>\n",
       "      <td>2015-02-03</td>\n",
       "      <td>Certified</td>\n",
       "      <td>2014-09-10</td>\n",
       "      <td>N</td>\n",
       "      <td>UNION PACIFIC RAILROAD</td>\n",
       "      <td>OMAHA</td>\n",
       "      <td>NEBRASKA</td>\n",
       "      <td>N</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>...</td>\n",
       "      <td>Computer Science/Engineering/Science</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>NaN</td>\n",
       "      <td>COMPUTER SCIENCE</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A-14220-96720</td>\n",
       "      <td>2015-01-12</td>\n",
       "      <td>Certified</td>\n",
       "      <td>2014-08-08</td>\n",
       "      <td>N</td>\n",
       "      <td>VST CONSULTING INC</td>\n",
       "      <td>ISELIN</td>\n",
       "      <td>NEW JERSEY</td>\n",
       "      <td>N</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>...</td>\n",
       "      <td>Business Admin., Mgmt. Info. Systems., Finance...</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>NaN</td>\n",
       "      <td>BUSINESS ADMINISTRATION</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>A-14203-91167</td>\n",
       "      <td>2014-12-18</td>\n",
       "      <td>Certified</td>\n",
       "      <td>2014-07-28</td>\n",
       "      <td>N</td>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>MOUNTAIN VIEW</td>\n",
       "      <td>CALIFORNIA</td>\n",
       "      <td>N</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>...</td>\n",
       "      <td>Comp. Sci., Elec. Eng., Comp. Eng.,Comp. Info....</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>L-1</td>\n",
       "      <td>Master's</td>\n",
       "      <td>NaN</td>\n",
       "      <td>COMPUTER SCIENCE</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>2014</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A-14206-92509</td>\n",
       "      <td>2015-03-10</td>\n",
       "      <td>Certified</td>\n",
       "      <td>2014-10-19</td>\n",
       "      <td>N</td>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>SANTA CLARA</td>\n",
       "      <td>CALIFORNIA</td>\n",
       "      <td>N</td>\n",
       "      <td>Electronics Engineers, Except Computer</td>\n",
       "      <td>...</td>\n",
       "      <td>Elec&amp;Comp/Elect/Comp Engr, or Scie, or reld Sc...</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>BANGLADESH</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>NaN</td>\n",
       "      <td>ELECTRICAL ENGINEERING</td>\n",
       "      <td>Electronics Engineers, Except Computer</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A-14202-90786</td>\n",
       "      <td>2015-01-07</td>\n",
       "      <td>Certified</td>\n",
       "      <td>2014-08-25</td>\n",
       "      <td>N</td>\n",
       "      <td>NET ESOLUTIONS CORPORATION</td>\n",
       "      <td>MCLEAN</td>\n",
       "      <td>VIRGINIA</td>\n",
       "      <td>N</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>...</td>\n",
       "      <td>Computer Science, Engineering, Math, Physics o...</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>NaN</td>\n",
       "      <td>COMPUTER INFORMATION SYSTEMS (US EVALUATION)</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>2015</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 27 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     CASE_NUMBER DECISION_DATE CASE_STATUS CASE_RECEIVED_DATE REFILE  \\\n",
       "0  A-14220-96665    2015-02-03   Certified         2014-09-10      N   \n",
       "1  A-14220-96720    2015-01-12   Certified         2014-08-08      N   \n",
       "2  A-14203-91167    2014-12-18   Certified         2014-07-28      N   \n",
       "3  A-14206-92509    2015-03-10   Certified         2014-10-19      N   \n",
       "4  A-14202-90786    2015-01-07   Certified         2014-08-25      N   \n",
       "\n",
       "                EMPLOYER_NAME  EMPLOYER_CITY EMPLOYER_STATE  \\\n",
       "0      UNION PACIFIC RAILROAD          OMAHA       NEBRASKA   \n",
       "1          VST CONSULTING INC         ISELIN     NEW JERSEY   \n",
       "2                 GOOGLE INC.  MOUNTAIN VIEW     CALIFORNIA   \n",
       "3           INTEL CORPORATION    SANTA CLARA     CALIFORNIA   \n",
       "4  NET ESOLUTIONS CORPORATION         MCLEAN       VIRGINIA   \n",
       "\n",
       "  FW_OWNERSHIP_INTEREST                            PW_SOC_TITLE  ...  \\\n",
       "0                     N       Software Developers, Applications  ...   \n",
       "1                     N       Software Developers, Applications  ...   \n",
       "2                     N       Software Developers, Applications  ...   \n",
       "3                     N  Electronics Engineers, Except Computer  ...   \n",
       "4                     N       Software Developers, Applications  ...   \n",
       "\n",
       "                                      JOB_INFO_MAJOR JOB_INFO_TRAINING  \\\n",
       "0               Computer Science/Engineering/Science                 N   \n",
       "1  Business Admin., Mgmt. Info. Systems., Finance...                 N   \n",
       "2  Comp. Sci., Elec. Eng., Comp. Eng.,Comp. Info....                 N   \n",
       "3  Elec&Comp/Elect/Comp Engr, or Scie, or reld Sc...                 N   \n",
       "4  Computer Science, Engineering, Math, Physics o...                 N   \n",
       "\n",
       "  JOB_INFO_FOREIGN_LANG_REQ COUNTRY_OF_CITIZENSHIP CLASS_OF_ADMISSION  \\\n",
       "0                         N                  INDIA               H-1B   \n",
       "1                         N                  INDIA               H-1B   \n",
       "2                         N            SOUTH KOREA                L-1   \n",
       "3                         N             BANGLADESH               H-1B   \n",
       "4                         N                  INDIA               H-1B   \n",
       "\n",
       "  FOREIGN_WORKER_INFO_EDUCATION FW_INFO_EDUCATION_OTHER  \\\n",
       "0                    Bachelor's                     NaN   \n",
       "1                      Master's                     NaN   \n",
       "2                      Master's                     NaN   \n",
       "3                      Master's                     NaN   \n",
       "4                      Master's                     NaN   \n",
       "\n",
       "                      FOREIGN_WORKER_INFO_MAJOR  \\\n",
       "0                              COMPUTER SCIENCE   \n",
       "1                       BUSINESS ADMINISTRATION   \n",
       "2                              COMPUTER SCIENCE   \n",
       "3                        ELECTRICAL ENGINEERING   \n",
       "4  COMPUTER INFORMATION SYSTEMS (US EVALUATION)   \n",
       "\n",
       "                        PW_Job_Title_9089  Year  \n",
       "0       Software Developers, Applications  2015  \n",
       "1       Software Developers, Applications  2015  \n",
       "2       Software Developers, Applications  2014  \n",
       "3  Electronics Engineers, Except Computer  2015  \n",
       "4       Software Developers, Applications  2015  \n",
       "\n",
       "[5 rows x 27 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import datetime\n",
    "#df1['Year']= df1['DECISION_DATE'].split()\n",
    "df['Year'] = pd.DatetimeIndex(df['DECISION_DATE']).year\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "093b98ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "# finding months duration from given time\n",
    "time_taken_list = list((df['DECISION_DATE'] - df['CASE_RECEIVED_DATE'])/np.timedelta64(1, 'M'))\n",
    "value_time = []\n",
    "for i in range(len(time_taken_list)):\n",
    "  val = time_taken_list[i]\n",
    "  if val < 3 :\n",
    "    value_time.append(\"at least 3 months\")\n",
    "  elif val < 6:\n",
    "    value_time.append(\"at least 6 months\")\n",
    "  elif val < 9:\n",
    "    value_time.append(\"at least 9 months\")\n",
    "  elif val < 12:\n",
    "    value_time.append(\"at least 12 months\")\n",
    "  elif val < 15:\n",
    "    value_time.append(\"at least 15 months\")\n",
    "  elif val < 24:\n",
    "    value_time.append(\"at least 24 months\")\n",
    "  elif val < 36:\n",
    "    value_time.append(\"at least 36 months\")\n",
    "  elif val < 60:\n",
    "    value_time.append(\"at least 60 months\")\n",
    "  elif val < 84:\n",
    "    value_time.append(\"at least 84 months\")\n",
    "  elif val < 120:\n",
    "    value_time.append(\"at least 120 months\")\n",
    "  else:\n",
    "    value_time.append(\"more than 120 months\")\n",
    "\n",
    "\n",
    "df['time_taken']  = value_time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "72f5d0df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CASE_NUMBER                           0\n",
       "DECISION_DATE                         0\n",
       "CASE_STATUS                           0\n",
       "CASE_RECEIVED_DATE                    2\n",
       "REFILE                           220904\n",
       "EMPLOYER_NAME                         8\n",
       "EMPLOYER_CITY                        12\n",
       "EMPLOYER_STATE                       46\n",
       "FW_OWNERSHIP_INTEREST               774\n",
       "PW_SOC_TITLE                       1005\n",
       "PW_AMOUNT_9089                      662\n",
       "PW_UNIT_OF_PAY_9089                1816\n",
       "JOB_INFO_WORK_CITY                   99\n",
       "JOB_INFO_WORK_STATE                 102\n",
       "JOB_INFO_JOB_TITLE                  131\n",
       "JOB_INFO_EDUCATION                   37\n",
       "JOB_INFO_EDUCATION_OTHER         519119\n",
       "JOB_INFO_MAJOR                    68630\n",
       "JOB_INFO_TRAINING                   460\n",
       "JOB_INFO_FOREIGN_LANG_REQ           465\n",
       "COUNTRY_OF_CITIZENSHIP               75\n",
       "CLASS_OF_ADMISSION                38211\n",
       "FOREIGN_WORKER_INFO_EDUCATION       383\n",
       "FW_INFO_EDUCATION_OTHER          514277\n",
       "FOREIGN_WORKER_INFO_MAJOR         54456\n",
       "PW_Job_Title_9089                   610\n",
       "Year                                  0\n",
       "time_taken                            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()\n",
    "df.isnull().sum()\n",
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b844bbee",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "dff  = df.drop(['DECISION_DATE','CASE_RECEIVED_DATE','CASE_NUMBER','REFILE', 'FW_OWNERSHIP_INTEREST','PW_Job_Title_9089','FW_INFO_EDUCATION_OTHER','JOB_INFO_EDUCATION_OTHER'], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0b2076ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Replacing missing values with mode\n",
    "dff['CLASS_OF_ADMISSION']=dff['CLASS_OF_ADMISSION'].fillna((dff['CLASS_OF_ADMISSION'].mode()[0]))\n",
    "dff['COUNTRY_OF_CITIZENSHIP']=dff['COUNTRY_OF_CITIZENSHIP'].fillna((dff['COUNTRY_OF_CITIZENSHIP'].mode()[0]))\n",
    "dff['JOB_INFO_MAJOR']=dff['JOB_INFO_MAJOR'].fillna((dff['JOB_INFO_MAJOR'].mode()[0]))\n",
    "dff['FOREIGN_WORKER_INFO_MAJOR']=dff['FOREIGN_WORKER_INFO_MAJOR'].fillna((dff['FOREIGN_WORKER_INFO_MAJOR'].mode()[0]))\n",
    "dff['FOREIGN_WORKER_INFO_EDUCATION'] = dff['FOREIGN_WORKER_INFO_EDUCATION'].fillna('Other')\n",
    "dff['JOB_INFO_FOREIGN_LANG_REQ'] = dff['JOB_INFO_FOREIGN_LANG_REQ'].fillna('Other')\n",
    "dff['PW_SOC_TITLE'] = dff['PW_SOC_TITLE'].fillna('Other')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "86468952",
   "metadata": {},
   "outputs": [],
   "source": [
    "dff[\"PW_AMOUNT_9089\"].replace({\"#############\": dff['PW_AMOUNT_9089'].mode()[0]}, inplace=True)\n",
    "\n",
    "dff['PW_AMOUNT_9089'] = dff['PW_AMOUNT_9089'].fillna((dff['PW_AMOUNT_9089'].mode()[0]))\n",
    "\n",
    "\n",
    "wages_list = list(dff['PW_AMOUNT_9089'])\n",
    "unit_list = list(dff['PW_UNIT_OF_PAY_9089'])\n",
    "\n",
    "new_wages_list = []\n",
    "for i in range(len(wages_list)):\n",
    "  try:\n",
    "    wages_list[i] = float(wages_list[i])\n",
    "  except:\n",
    "    wages_list[i] = float(wages_list[i].replace(',',''))\n",
    "  finally:\n",
    "    if unit_list[i] == 'Hour':\n",
    "      wages_list[i] = wages_list[i] * 40 *52\n",
    "    elif unit_list[i] == 'Week':\n",
    "      wages_list[i] = wages_list[i] * 52\n",
    "    elif unit_list[i] == 'Month':\n",
    "      wages_list[i] = wages_list[i] * 12\n",
    "    elif unit_list[i] == 'Bi-Weekly':\n",
    "      wages_list[i] = wages_list[i] * 24\n",
    "    else: \n",
    "      wages_list[i] = wages_list[i]\n",
    "  \n",
    "  new_wages_list.append(wages_list[i])\n",
    "\n",
    "dff['PW_AMOUNT_9089'] = new_wages_list\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d02d3e17",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['CASE_STATUS', 'EMPLOYER_NAME', 'EMPLOYER_CITY', 'EMPLOYER_STATE',\n",
       "       'PW_SOC_TITLE', 'PW_AMOUNT_9089', 'PW_UNIT_OF_PAY_9089',\n",
       "       'JOB_INFO_WORK_CITY', 'JOB_INFO_WORK_STATE', 'JOB_INFO_JOB_TITLE',\n",
       "       'JOB_INFO_EDUCATION', 'JOB_INFO_MAJOR', 'JOB_INFO_TRAINING',\n",
       "       'JOB_INFO_FOREIGN_LANG_REQ', 'COUNTRY_OF_CITIZENSHIP',\n",
       "       'CLASS_OF_ADMISSION', 'FOREIGN_WORKER_INFO_EDUCATION',\n",
       "       'FOREIGN_WORKER_INFO_MAJOR', 'Year', 'time_taken'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dff.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "cbc39cdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(534765, 18)\n"
     ]
    }
   ],
   "source": [
    "dfff = dff.drop(['PW_UNIT_OF_PAY_9089','Year'], axis =1)\n",
    "dfff = dfff.dropna()\n",
    "print(dfff.shape)\n",
    "#dfff.isna().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f3d4b456",
   "metadata": {},
   "outputs": [],
   "source": [
    "state_abbrevs = {\n",
    "    'Alabama': 'AL',\n",
    "    'Alaska': 'AK',\n",
    "    'Arizona': 'AZ',\n",
    "    'Arkansas': 'AR',\n",
    "    'California': 'CA',\n",
    "    'Colorado': 'CO',\n",
    "    'Connecticut': 'CT',\n",
    "    'Delaware': 'DE',\n",
    "    'Florida': 'FL',\n",
    "    'Georgia': 'GA',\n",
    "    'Hawaii': 'HI',\n",
    "    'Idaho': 'ID',\n",
    "    'Illinois': 'IL',\n",
    "    'Indiana': 'IN',\n",
    "    'Iowa': 'IA',\n",
    "    'Kansas': 'KS',\n",
    "    'Kentucky': 'KY',\n",
    "    'Louisiana': 'LA',\n",
    "    'Maine': 'ME',\n",
    "    'Maryland': 'MD',\n",
    "    'Massachusetts': 'MA',\n",
    "    'Michigan': 'MI',\n",
    "    'Minnesota': 'MN',\n",
    "    'Mississippi': 'MS',\n",
    "    'Missouri': 'MO',\n",
    "    'Montana': 'MT',\n",
    "    'Nebraska': 'NE',\n",
    "    'Nevada': 'NV',\n",
    "    'New Hampshire': 'NH',\n",
    "    'New Jersey': 'NJ',\n",
    "    'New Mexico': 'NM',\n",
    "    'New York': 'NY',\n",
    "    'North Carolina': 'NC',\n",
    "    'North Dakota': 'ND',\n",
    "    'Ohio': 'OH',\n",
    "    'Oklahoma': 'OK',\n",
    "    'Oregon': 'OR',\n",
    "    'Pennsylvania': 'PA',\n",
    "    'Rhode Island': 'RI',\n",
    "    'South Carolina': 'SC',\n",
    "    'South Dakota': 'SD',\n",
    "    'Tennessee': 'TN',\n",
    "    'Texas': 'TX',\n",
    "    'Utah': 'UT',\n",
    "    'Vermont': 'VT',\n",
    "    'Virginia': 'VA',\n",
    "    'Washington': 'WA',\n",
    "    'West Virginia': 'WV',\n",
    "    'Wisconsin': 'WI',\n",
    "    'Wyoming': 'WY',\n",
    "    'Northern Mariana Islands':'MP', \n",
    "    'Palau': 'PW', \n",
    "    'Puerto Rico': 'PR', \n",
    "    'Virgin Islands': 'VI', \n",
    "    'District of Columbia': 'DC'\n",
    "}\n",
    "\n",
    "#Capitalizing Keys\n",
    "us_state_abbrev = {k.upper(): v for k, v in state_abbrevs.items()}\n",
    "dfff['EMPLOYER_STATE'].replace(us_state_abbrev, inplace=True)\n",
    "dfff.EMPLOYER_STATE = dfff.EMPLOYER_STATE.astype(str)\n",
    "#dfff['EMPLOYER_STATE'].value_counts()\n",
    "\n",
    "\n",
    "dfff['JOB_INFO_WORK_STATE'].replace(us_state_abbrev, inplace=True)\n",
    "dfff.JOB_INFO_WORK_STATE = dfff.JOB_INFO_WORK_STATE.astype(str)\n",
    "#dfff['EMPLOYER_STATE'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "07de31fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CA                                126470\n",
       "TX                                 68519\n",
       "NY                                 42854\n",
       "NJ                                 38119\n",
       "WA                                 29649\n",
       "IL                                 20259\n",
       "FL                                 20125\n",
       "MA                                 17942\n",
       "GA                                 17779\n",
       "MI                                 16811\n",
       "VA                                 16290\n",
       "NC                                 12671\n",
       "PA                                 12216\n",
       "OH                                  8808\n",
       "MD                                  7160\n",
       "OR                                  6392\n",
       "AZ                                  5966\n",
       "MO                                  4824\n",
       "WI                                  4551\n",
       "MN                                  4525\n",
       "CO                                  4512\n",
       "CT                                  4483\n",
       "SC                                  4009\n",
       "IN                                  3798\n",
       "TN                                  3085\n",
       "AL                                  2664\n",
       "DE                                  2619\n",
       "AR                                  2504\n",
       "UT                                  2152\n",
       "DC                                  2131\n",
       "KY                                  2105\n",
       "LA                                  2016\n",
       "IA                                  1983\n",
       "KS                                  1779\n",
       "NE                                  1771\n",
       "OK                                  1321\n",
       "NH                                  1044\n",
       "NV                                  1032\n",
       "NM                                  1004\n",
       "RI                                   834\n",
       "MS                                   810\n",
       "ID                                   725\n",
       "SD                                   606\n",
       "ND                                   540\n",
       "HI                                   500\n",
       "ME                                   446\n",
       "VT                                   414\n",
       "WV                                   387\n",
       "GU                                   331\n",
       "MP                                   269\n",
       "PR                                   250\n",
       "MT                                   207\n",
       "AK                                   182\n",
       "WY                                   127\n",
       "GUAM                                 127\n",
       "VI                                    60\n",
       "MARSHALL ISLANDS                       5\n",
       "MH                                     2\n",
       "FEDERATED STATES OF MICRONESIA         1\n",
       "Name: JOB_INFO_WORK_STATE, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfff['JOB_INFO_WORK_STATE'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4fda6f22",
   "metadata": {},
   "outputs": [],
   "source": [
    "dfff= dfff.rename(columns={'PW_AMOUNT_9089':'WAGES_OFFERED',\n",
    "                           'PW_SOC_TITLE':'SOC_TITLE',\n",
    "                           'JOB_INFO_WORK_CITY': 'JOB_CITY',\n",
    "                           'JOB_INFO_WORK_STATE':'JOB_STATE',\n",
    "                           'JOB_INFO_JOB_TITLE':'JOB_TITLE',\n",
    "                           'JOB_INFO_EDUCATION':'REQD_EDUCATION',\n",
    "                           'JOB_INFO_MAJOR':'RELTD_MAJOR',\n",
    "                           'JOB_INFO_TRAINING':'TRAINING_REQD',\n",
    "                           'JOB_INFO_FOREIGN_LANG_REQ':'LANG_REQD',\n",
    "                           'COUNTRY_OF_CITIZENSHIP':'CITIZENSHIP',\n",
    "                           'CLASS_OF_ADMISSION':'ADMISSION_TYPE',\n",
    "                           'FOREIGN_WORKER_INFO_EDUCATION':'WORKER_EDUCATION',\n",
    "                           'FOREIGN_WORKER_INFO_MAJOR':'WORKER_MAJOR'\n",
    "                            })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "76514128",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(514964, 18)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>EMPLOYER_NAME</th>\n",
       "      <th>EMPLOYER_CITY</th>\n",
       "      <th>EMPLOYER_STATE</th>\n",
       "      <th>SOC_TITLE</th>\n",
       "      <th>WAGES_OFFERED</th>\n",
       "      <th>JOB_CITY</th>\n",
       "      <th>JOB_STATE</th>\n",
       "      <th>JOB_TITLE</th>\n",
       "      <th>REQD_EDUCATION</th>\n",
       "      <th>RELTD_MAJOR</th>\n",
       "      <th>TRAINING_REQD</th>\n",
       "      <th>LANG_REQD</th>\n",
       "      <th>CITIZENSHIP</th>\n",
       "      <th>ADMISSION_TYPE</th>\n",
       "      <th>WORKER_EDUCATION</th>\n",
       "      <th>WORKER_MAJOR</th>\n",
       "      <th>time_taken</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Certified</td>\n",
       "      <td>UNION PACIFIC RAILROAD</td>\n",
       "      <td>OMAHA</td>\n",
       "      <td>NE</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>76482.0</td>\n",
       "      <td>Omaha</td>\n",
       "      <td>NE</td>\n",
       "      <td>Associate Systems Engineer</td>\n",
       "      <td>Master's</td>\n",
       "      <td>Computer Science/Engineering/Science</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>COMPUTER SCIENCE</td>\n",
       "      <td>at least 6 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Certified</td>\n",
       "      <td>VST CONSULTING INC</td>\n",
       "      <td>ISELIN</td>\n",
       "      <td>NJ</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>90459.0</td>\n",
       "      <td>Iselin</td>\n",
       "      <td>NJ</td>\n",
       "      <td>Business Analyst</td>\n",
       "      <td>Master's</td>\n",
       "      <td>Business Admin., Mgmt. Info. Systems., Finance...</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>BUSINESS ADMINISTRATION</td>\n",
       "      <td>at least 6 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Certified</td>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>MOUNTAIN VIEW</td>\n",
       "      <td>CA</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>98675.0</td>\n",
       "      <td>Mountain View</td>\n",
       "      <td>CA</td>\n",
       "      <td>Software Engineer</td>\n",
       "      <td>Master's</td>\n",
       "      <td>Comp. Sci., Elec. Eng., Comp. Eng.,Comp. Info....</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>L-1</td>\n",
       "      <td>Master's</td>\n",
       "      <td>COMPUTER SCIENCE</td>\n",
       "      <td>at least 6 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Certified</td>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>SANTA CLARA</td>\n",
       "      <td>CA</td>\n",
       "      <td>Electronics Engineers, Except Computer</td>\n",
       "      <td>80617.0</td>\n",
       "      <td>Folsom</td>\n",
       "      <td>CA</td>\n",
       "      <td>Graphics Hardware Engineer</td>\n",
       "      <td>Master's</td>\n",
       "      <td>Elec&amp;Comp/Elect/Comp Engr, or Scie, or reld Sc...</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>BANGLADESH</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>ELECTRICAL ENGINEERING</td>\n",
       "      <td>at least 6 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Certified</td>\n",
       "      <td>NET ESOLUTIONS CORPORATION</td>\n",
       "      <td>MCLEAN</td>\n",
       "      <td>VA</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>87422.0</td>\n",
       "      <td>McLean</td>\n",
       "      <td>VA</td>\n",
       "      <td>Computer Programmer</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>Computer Science, Engineering, Math, Physics o...</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>COMPUTER INFORMATION SYSTEMS (US EVALUATION)</td>\n",
       "      <td>at least 6 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535053</th>\n",
       "      <td>Certified</td>\n",
       "      <td>GERARDO FROESE</td>\n",
       "      <td>SEMINOLE</td>\n",
       "      <td>TX</td>\n",
       "      <td>Farmworkers and Laborers, Crop, Nursery, and G...</td>\n",
       "      <td>19240.0</td>\n",
       "      <td>SEMINOLE</td>\n",
       "      <td>TX</td>\n",
       "      <td>FARM WORKER</td>\n",
       "      <td>None</td>\n",
       "      <td>Computer Science</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>CANADA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>None</td>\n",
       "      <td>COMPUTER SCIENCE</td>\n",
       "      <td>at least 3 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535054</th>\n",
       "      <td>Certified</td>\n",
       "      <td>ACTIONIQ</td>\n",
       "      <td>NY</td>\n",
       "      <td>NY</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>96366.0</td>\n",
       "      <td>New York</td>\n",
       "      <td>NY</td>\n",
       "      <td>Full Stack Engineer</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>See H.14.</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>ELECTRICAL AND COMPUTER ENGINEERING</td>\n",
       "      <td>at least 3 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535055</th>\n",
       "      <td>Certified</td>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>SANTA CLARA</td>\n",
       "      <td>CA</td>\n",
       "      <td>Architectural and Engineering Managers</td>\n",
       "      <td>173046.0</td>\n",
       "      <td>Santa Clara</td>\n",
       "      <td>CA</td>\n",
       "      <td>Engineering Manager - Component Design</td>\n",
       "      <td>Master's</td>\n",
       "      <td>See Section H14 of the 9089</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>ELECTRICAL ENGINEERING</td>\n",
       "      <td>at least 3 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535063</th>\n",
       "      <td>Denied</td>\n",
       "      <td>BKFS I SERVICES LLC</td>\n",
       "      <td>JACKSONVILLE</td>\n",
       "      <td>FL</td>\n",
       "      <td>Software Developers, Applications</td>\n",
       "      <td>118664.0</td>\n",
       "      <td>IRVINE</td>\n",
       "      <td>CA</td>\n",
       "      <td>APPLICATIONS PROGRAMMER III</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>COMPUTER SCIENCE</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>COMPUTER ENGINEERING</td>\n",
       "      <td>at least 15 months</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535083</th>\n",
       "      <td>Denied</td>\n",
       "      <td>TECH MAHINDRA (AMERICAS) INC., (FORMERLY MBT I...</td>\n",
       "      <td>PLANO</td>\n",
       "      <td>TX</td>\n",
       "      <td>Computer Occupations, All Other</td>\n",
       "      <td>98925.0</td>\n",
       "      <td>Plano</td>\n",
       "      <td>TX</td>\n",
       "      <td>Data Warehousing Specialist</td>\n",
       "      <td>Master's</td>\n",
       "      <td>Computer Science</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>ELECTRONICS ENGINEERING</td>\n",
       "      <td>at least 9 months</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>514964 rows × 18 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       CASE_STATUS                                      EMPLOYER_NAME  \\\n",
       "0        Certified                             UNION PACIFIC RAILROAD   \n",
       "1        Certified                                 VST CONSULTING INC   \n",
       "2        Certified                                        GOOGLE INC.   \n",
       "3        Certified                                  INTEL CORPORATION   \n",
       "4        Certified                         NET ESOLUTIONS CORPORATION   \n",
       "...            ...                                                ...   \n",
       "535053   Certified                                     GERARDO FROESE   \n",
       "535054   Certified                                           ACTIONIQ   \n",
       "535055   Certified                                  INTEL CORPORATION   \n",
       "535063      Denied                                BKFS I SERVICES LLC   \n",
       "535083      Denied  TECH MAHINDRA (AMERICAS) INC., (FORMERLY MBT I...   \n",
       "\n",
       "        EMPLOYER_CITY EMPLOYER_STATE  \\\n",
       "0               OMAHA             NE   \n",
       "1              ISELIN             NJ   \n",
       "2       MOUNTAIN VIEW             CA   \n",
       "3         SANTA CLARA             CA   \n",
       "4              MCLEAN             VA   \n",
       "...               ...            ...   \n",
       "535053       SEMINOLE             TX   \n",
       "535054             NY             NY   \n",
       "535055    SANTA CLARA             CA   \n",
       "535063   JACKSONVILLE             FL   \n",
       "535083          PLANO             TX   \n",
       "\n",
       "                                                SOC_TITLE  WAGES_OFFERED  \\\n",
       "0                       Software Developers, Applications        76482.0   \n",
       "1                       Software Developers, Applications        90459.0   \n",
       "2                       Software Developers, Applications        98675.0   \n",
       "3                  Electronics Engineers, Except Computer        80617.0   \n",
       "4                       Software Developers, Applications        87422.0   \n",
       "...                                                   ...            ...   \n",
       "535053  Farmworkers and Laborers, Crop, Nursery, and G...        19240.0   \n",
       "535054                  Software Developers, Applications        96366.0   \n",
       "535055             Architectural and Engineering Managers       173046.0   \n",
       "535063                  Software Developers, Applications       118664.0   \n",
       "535083                    Computer Occupations, All Other        98925.0   \n",
       "\n",
       "             JOB_CITY JOB_STATE                               JOB_TITLE  \\\n",
       "0               Omaha        NE              Associate Systems Engineer   \n",
       "1              Iselin        NJ                        Business Analyst   \n",
       "2       Mountain View        CA                       Software Engineer   \n",
       "3              Folsom        CA              Graphics Hardware Engineer   \n",
       "4              McLean        VA                     Computer Programmer   \n",
       "...               ...       ...                                     ...   \n",
       "535053       SEMINOLE        TX                             FARM WORKER   \n",
       "535054       New York        NY                     Full Stack Engineer   \n",
       "535055    Santa Clara        CA  Engineering Manager - Component Design   \n",
       "535063         IRVINE        CA             APPLICATIONS PROGRAMMER III   \n",
       "535083          Plano        TX             Data Warehousing Specialist   \n",
       "\n",
       "       REQD_EDUCATION                                        RELTD_MAJOR  \\\n",
       "0            Master's               Computer Science/Engineering/Science   \n",
       "1            Master's  Business Admin., Mgmt. Info. Systems., Finance...   \n",
       "2            Master's  Comp. Sci., Elec. Eng., Comp. Eng.,Comp. Info....   \n",
       "3            Master's  Elec&Comp/Elect/Comp Engr, or Scie, or reld Sc...   \n",
       "4          Bachelor's  Computer Science, Engineering, Math, Physics o...   \n",
       "...               ...                                                ...   \n",
       "535053           None                                   Computer Science   \n",
       "535054     Bachelor's                                          See H.14.   \n",
       "535055       Master's                        See Section H14 of the 9089   \n",
       "535063     Bachelor's                                   COMPUTER SCIENCE   \n",
       "535083       Master's                                   Computer Science   \n",
       "\n",
       "       TRAINING_REQD LANG_REQD  CITIZENSHIP ADMISSION_TYPE WORKER_EDUCATION  \\\n",
       "0                  N         N        INDIA           H-1B       Bachelor's   \n",
       "1                  N         N        INDIA           H-1B         Master's   \n",
       "2                  N         N  SOUTH KOREA            L-1         Master's   \n",
       "3                  N         N   BANGLADESH           H-1B         Master's   \n",
       "4                  N         N        INDIA           H-1B         Master's   \n",
       "...              ...       ...          ...            ...              ...   \n",
       "535053             N         N       CANADA           H-1B             None   \n",
       "535054             N         N        INDIA           H-1B       Bachelor's   \n",
       "535055             N         N        INDIA           H-1B         Master's   \n",
       "535063             N         N        INDIA           H-1B       Bachelor's   \n",
       "535083             N         N        INDIA           H-1B       Bachelor's   \n",
       "\n",
       "                                        WORKER_MAJOR          time_taken  \n",
       "0                                   COMPUTER SCIENCE   at least 6 months  \n",
       "1                            BUSINESS ADMINISTRATION   at least 6 months  \n",
       "2                                   COMPUTER SCIENCE   at least 6 months  \n",
       "3                             ELECTRICAL ENGINEERING   at least 6 months  \n",
       "4       COMPUTER INFORMATION SYSTEMS (US EVALUATION)   at least 6 months  \n",
       "...                                              ...                 ...  \n",
       "535053                              COMPUTER SCIENCE   at least 3 months  \n",
       "535054           ELECTRICAL AND COMPUTER ENGINEERING   at least 3 months  \n",
       "535055                        ELECTRICAL ENGINEERING   at least 3 months  \n",
       "535063                          COMPUTER ENGINEERING  at least 15 months  \n",
       "535083                       ELECTRONICS ENGINEERING   at least 9 months  \n",
       "\n",
       "[514964 rows x 18 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfff = dfff[dfff['CASE_STATUS'] != 'Withdrawn']\n",
    "print(dfff.shape)\n",
    "dfff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7f0305bb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfff['JOB_TITLE'].isna().sum()\n",
    "dfff['JOB_TITLE'] = dfff['JOB_TITLE'].fillna((dfff['JOB_TITLE'].mode()[0]))\n",
    "dfff['JOB_TITLE'].isna().sum()\n",
    "dfff['JOB_TITLE'].isnull().sum()\n",
    "dfff['SOC_TITLE'].isnull().sum()\n",
    "#dfff['SOC_TITLE'].unique()\n",
    "#dfff.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2e78863f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:5: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('computer','programmer')] = 'computer occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:6: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('technical','designer')] = 'computer occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:7: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('software','web developer')] = 'computer occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:8: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('system engineer','system')] = 'computer occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:9: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('database','systems')] = 'computer occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:10: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('math','statistic')] = 'Mathematical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:11: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('predictive model','stats')] = 'Mathematical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:12: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('teacher','linguist')] = 'Education Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:13: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('lecturer','lecture')] = 'Education Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:14: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('professor','Teach')] = 'Education Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:15: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('school principal')] = 'Education Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:16: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('medical','doctor')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:17: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('physician','dentist')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:18: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('pharmacist','gastroenterologist')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:19: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('Health','Physical Therapists')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:20: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('psychiatrist')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:21: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('surgeon','nurse')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:22: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('clinical data','psychiatr')] = 'Medical Occupations'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:23: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('chemist','physicist')] = 'Advance Sciences'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:24: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('rehabilitation specialist','scientist')] = 'Advance Sciences'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:25: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('biology','chemist')] = 'Advance Sciences'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:26: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('biologi','clinical research')] = 'Advance Sciences'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:27: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('public relation','manage')] = 'Management Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:28: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('consultant','clerk')] = 'Management Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:29: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('management','operation')] = 'Management Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:30: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('mgr','integration')] = 'Management Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:31: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('chief','plan')] = 'Management Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:32: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('executive','project')] = 'Management Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:33: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('advertis','marketing')] = 'Marketing Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:34: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('promotion','market research')] = 'Marketing Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:35: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('business','business analyst')] = 'Business Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:36: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('business systems analyst')] = 'Business Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:37: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('accountant','finance')] = 'Financial Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:38: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('financial','audit')] = 'Financial Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:39: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('engineer','architect')] = 'Architecture & Engineering'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:40: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('surveyor','carto')] = 'Architecture & Engineering'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:41: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('technician','drafter')] = 'Architecture & Engineering'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:42: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('information security','information tech')] = 'Architecture & Engineering'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:43: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('food','cook',)] = 'Food Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:44: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('waitress','waiters')] = 'Food Occupation'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:45: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('restaurant')] = 'Food Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:46: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('attorney','law')] = 'Law Occupation'\n",
      "/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3739290873.py:47: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('legal','court')] = 'Law Occupation'\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "dfff['OCCUPATION'] = np.nan\n",
    "dfff.OCCUPATION\n",
    "#dfff = dfff.astype({\"OCCUPATION\": 'str'})\n",
    "dfff['SOC_TITLE'] = dfff['SOC_TITLE'].str.lower()\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('computer','programmer')] = 'computer occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('technical','designer')] = 'computer occupations' \n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('software','web developer')] = 'computer occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('system engineer','system')] = 'computer occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('database','systems')] = 'computer occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('math','statistic')] = 'Mathematical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('predictive model','stats')] = 'Mathematical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('teacher','linguist')] = 'Education Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('lecturer','lecture')] = 'Education Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('professor','Teach')] = 'Education Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('school principal')] = 'Education Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('medical','doctor')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('physician','dentist')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('pharmacist','gastroenterologist')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('Health','Physical Therapists')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('psychiatrist')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('surgeon','nurse')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('clinical data','psychiatr')] = 'Medical Occupations'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('chemist','physicist')] = 'Advance Sciences'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('rehabilitation specialist','scientist')] = 'Advance Sciences'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('biology','chemist')] = 'Advance Sciences'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('biologi','clinical research')] = 'Advance Sciences'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('public relation','manage')] = 'Management Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('consultant','clerk')] = 'Management Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('management','operation')] = 'Management Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('mgr','integration')] = 'Management Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('chief','plan')] = 'Management Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('executive','project')] = 'Management Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('advertis','marketing')] = 'Marketing Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('promotion','market research')] = 'Marketing Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('business','business analyst')] = 'Business Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('business systems analyst')] = 'Business Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('accountant','finance')] = 'Financial Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('financial','audit')] = 'Financial Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('engineer','architect')] = 'Architecture & Engineering'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('surveyor','carto')] = 'Architecture & Engineering'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('technician','drafter')] = 'Architecture & Engineering'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('information security','information tech')] = 'Architecture & Engineering'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('food','cook',)] = 'Food Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('waitress','waiters')] = 'Food Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('restaurant')] = 'Food Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('attorney','law')] = 'Law Occupation'\n",
    "dfff.OCCUPATION[dfff['SOC_TITLE'].str.contains('legal','court')] = 'Law Occupation'\n",
    "\n",
    "dfff['OCCUPATION']= dfff.OCCUPATION.replace(np.nan, 'Others', regex=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "bb0a067d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "dfff['RELTD_MAJOR']\n",
    "\n",
    "dfff['NEW_RELATED_MAJOR'] = np.nan\n",
    "dfff.NEW_RELATED_MAJOR\n",
    "#dfff = dfff.astype({\"NEW_RELATED_MAJOR\": 'str'})\n",
    "dfff['RELTD_MAJOR'] = dfff['RELTD_MAJOR'].str.lower()\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer','comp.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('biology','physics')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('information technology','information systems')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('science','sci.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer','engineering')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('eng.','engineering')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('info','tech.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('math','physics.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('data','chemistry.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('astr.','astronomy.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer','comp.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer','comp.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer','comp.')] = 'STEM Major'\n",
    "dfff.NEW_RELATED_MAJOR[dfff['RELTD_MAJOR'].str.contains('computer','engineering')] = 'STEM Major'\n",
    "\n",
    "dfff['NEW_RELATED_MAJOR']= dfff.NEW_RELATED_MAJOR.replace(np.nan, 'NON-STEM Major', regex=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0021887d",
   "metadata": {},
   "outputs": [],
   "source": [
    "warnings.filterwarnings(\"ignore\")\n",
    "dfff['WORKER_MAJOR'].isnull().sum()\n",
    "\n",
    "\n",
    "dfff['NEW_WORKER_MAJOR'] = np.nan\n",
    "dfff.NEW_WORKER_MAJOR\n",
    "#dfff = dfff.astype({\"NEW_WORKER_MAJOR\": 'str'})\n",
    "dfff['WORKER_MAJOR'] = dfff['WORKER_MAJOR'].str.lower()\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('computer',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('comp.',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('biology',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('information technology',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('science',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('computer',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('eng.',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('info',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('math',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('data',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('astr.',na=False)] = 'STEM Major'\n",
    "\n",
    "\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('stat',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('physics',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('information systems',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('sci.',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('engineering',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('engineering',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('tech.',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('physics.',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('chemistry.',na=False)] = 'STEM Major'\n",
    "dfff.NEW_WORKER_MAJOR[dfff['WORKER_MAJOR'].str.contains('astronomy.',na=False)] = 'STEM Major'\n",
    "\n",
    "\n",
    "dfff['NEW_WORKER_MAJOR']= dfff.NEW_WORKER_MAJOR.replace(np.nan, 'NON-STEM Major', regex=True)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "23e4fb4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#EMPLOYER_NAME\n",
    "# TOP 10 Employer [COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION, MICROSOFT CORPORATION, INTEL CORPORATION. AMAZON, GOOGLE, FACEBOOK, INFOSYS\t]\n",
    "#TOP 10-2O Employer [CISCO, ORACLE, TATA, DELOITTE, HCL, QUALCOMM, ERNST & YOUNG, JP MORGAN CHASE & CO, SALESFORECE, WIPRO ]\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "dfff['EMPLOYER_NAME']\n",
    "\n",
    "dfff['NEW_EMPLOYER_NAME'] = np.nan\n",
    "dfff.NEW_EMPLOYER_NAME\n",
    "#dfff = dfff.astype({\"NEW_EMPLOYER_NAME\": 'str'})\n",
    "dfff['EMPLOYER_NAME'] = dfff['EMPLOYER_NAME'].str.upper()\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION','MICROSOFT CORPORATION')] = 'Top 10 Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('INTEL CORPORATION','FACEBOOK')] = 'Top 10 Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('INFOSYS','GOOGLE')] = 'Top 10 Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('AMAZON','AMAZON.COM')] = 'Top 10 Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('APPLE')] = 'Top 10 Employer'\n",
    "\n",
    "\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('CISCO','ORACLE')] = 'TOP 10-2O Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('TATA','DELOITTE')] = 'TOP 10-2O Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('HCL','QUALCOMM')] = 'TOP 10-2O Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('CISCO','ORACLE')] = 'TOP 10-2O Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('ERNST & YOUNG','JP MORGAN CHASE & CO')] = 'TOP 10-2O Employer'\n",
    "dfff.NEW_EMPLOYER_NAME[dfff['EMPLOYER_NAME'].str.contains('SALESFORECE','WIPRO')] = 'TOP 10-2O Employer'\n",
    "\n",
    "\n",
    "dfff['NEW_EMPLOYER_NAME']= dfff.NEW_EMPLOYER_NAME.replace(np.nan, 'Other Employer', regex=True)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "eb2b24e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "warnings.filterwarnings(\"ignore\")\n",
    "dfff['WAGES_OFFERED'].max() # 221,035,360.0\n",
    "dfff['WAGES_OFFERED'].min()\n",
    "\n",
    "\n",
    "conditions = [\n",
    "\n",
    "    (dfff['WAGES_OFFERED'] <= 30000),\n",
    "    (dfff['WAGES_OFFERED'] > 30000) & (dfff['WAGES_OFFERED']  <= 60000),\n",
    "    (dfff['WAGES_OFFERED'] > 60000) & (dfff['WAGES_OFFERED']  <= 100000),\n",
    "    (dfff['WAGES_OFFERED'] > 100000) & (dfff['WAGES_OFFERED']  <= 150000),\n",
    "    (dfff['WAGES_OFFERED'] > 150000)& (dfff['WAGES_OFFERED']  <= 200000),\n",
    "    (dfff['WAGES_OFFERED'] > 200000)]\n",
    "    \n",
    "\n",
    "# create a list of the values we want to assign for each condition\n",
    "values = ['Below 30000', 'Between 30000 -60000', 'Between 60000 -100000', 'Between 100000 -1500000','Between 150000 -2000000','Above 200000']\n",
    "\n",
    "dfff['WAGE_OFFERED'] = np.select(conditions, values)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b2399c1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "58b71ae1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 534765 entries, 0 to 535083\n",
      "Data columns (total 17 columns):\n",
      " #   Column             Non-Null Count   Dtype \n",
      "---  ------             --------------   ----- \n",
      " 0   CASE_STATUS        534765 non-null  object\n",
      " 1   EMPLOYER_STATE     534765 non-null  object\n",
      " 2   SOC_TITLE          534765 non-null  object\n",
      " 3   JOB_CITY           534765 non-null  object\n",
      " 4   JOB_STATE          534765 non-null  object\n",
      " 5   REQD_EDUCATION     534765 non-null  object\n",
      " 6   TRAINING_REQD      534765 non-null  object\n",
      " 7   LANG_REQD          534765 non-null  object\n",
      " 8   CITIZENSHIP        534765 non-null  object\n",
      " 9   ADMISSION_TYPE     534765 non-null  object\n",
      " 10  WORKER_EDUCATION   534765 non-null  object\n",
      " 11  time_taken         534765 non-null  object\n",
      " 12  OCCUPATION         534765 non-null  object\n",
      " 13  NEW_RELATED_MAJOR  534765 non-null  object\n",
      " 14  NEW_WORKER_MAJOR   534765 non-null  object\n",
      " 15  NEW_EMPLOYER_NAME  534765 non-null  object\n",
      " 16  WAGE_OFFERED       534765 non-null  object\n",
      "dtypes: object(17)\n",
      "memory usage: 73.4+ MB\n"
     ]
    }
   ],
   "source": [
    "dffff = dfff.copy()\n",
    "dffff = dffff.drop(['JOB_TITLE','WORKER_MAJOR','RELTD_MAJOR','EMPLOYER_NAME','WAGES_OFFERED','EMPLOYER_CITY'], axis =1)\n",
    "\n",
    "#Also employer city is not that important because it is just information of employer to contact them.\n",
    "# City information of the foreign worker's intended area of employment so decided to drop employer city.\n",
    "dffff.info()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "20f3c740",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>SOC_TITLE</th>\n",
       "      <th>JOB_CITY</th>\n",
       "      <th>JOB_STATE</th>\n",
       "      <th>REQD_EDUCATION</th>\n",
       "      <th>TRAINING_REQD</th>\n",
       "      <th>LANG_REQD</th>\n",
       "      <th>CITIZENSHIP</th>\n",
       "      <th>ADMISSION_TYPE</th>\n",
       "      <th>WORKER_EDUCATION</th>\n",
       "      <th>time_taken</th>\n",
       "      <th>OCCUPATION</th>\n",
       "      <th>NEW_RELATED_MAJOR</th>\n",
       "      <th>NEW_WORKER_MAJOR</th>\n",
       "      <th>NEW_EMPLOYER_NAME</th>\n",
       "      <th>WAGE_OFFERED</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Omaha</td>\n",
       "      <td>NE</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Iselin</td>\n",
       "      <td>NJ</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Mountain View</td>\n",
       "      <td>CA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>L-1</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Certified</td>\n",
       "      <td>electronics engineers, except computer</td>\n",
       "      <td>Folsom</td>\n",
       "      <td>CA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>BANGLADESH</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>Architecture &amp; Engineering</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Top 10 Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>McLean</td>\n",
       "      <td>VA</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535057</th>\n",
       "      <td>Withdrawn</td>\n",
       "      <td>stock clerks and order fillers</td>\n",
       "      <td>ASTORIA</td>\n",
       "      <td>NY</td>\n",
       "      <td>High School</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>POLAND</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>High School</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>Others</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 30000 -60000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535058</th>\n",
       "      <td>Withdrawn</td>\n",
       "      <td>architectural and engineering managers</td>\n",
       "      <td>Exeter</td>\n",
       "      <td>RI</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>Architecture &amp; Engineering</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 100000 -1500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535059</th>\n",
       "      <td>Withdrawn</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Moonachie</td>\n",
       "      <td>NJ</td>\n",
       "      <td>High School</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>E-2</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 100000 -1500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535063</th>\n",
       "      <td>Denied</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>IRVINE</td>\n",
       "      <td>CA</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 15 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 100000 -1500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535083</th>\n",
       "      <td>Denied</td>\n",
       "      <td>computer occupations, all other</td>\n",
       "      <td>Plano</td>\n",
       "      <td>TX</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 9 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>534765 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       CASE_STATUS                               SOC_TITLE       JOB_CITY  \\\n",
       "0        Certified       software developers, applications          Omaha   \n",
       "1        Certified       software developers, applications         Iselin   \n",
       "2        Certified       software developers, applications  Mountain View   \n",
       "3        Certified  electronics engineers, except computer         Folsom   \n",
       "4        Certified       software developers, applications         McLean   \n",
       "...            ...                                     ...            ...   \n",
       "535057   Withdrawn          stock clerks and order fillers        ASTORIA   \n",
       "535058   Withdrawn  architectural and engineering managers         Exeter   \n",
       "535059   Withdrawn       software developers, applications      Moonachie   \n",
       "535063      Denied       software developers, applications         IRVINE   \n",
       "535083      Denied         computer occupations, all other          Plano   \n",
       "\n",
       "       JOB_STATE REQD_EDUCATION TRAINING_REQD LANG_REQD  CITIZENSHIP  \\\n",
       "0             NE       Master's             N         N        INDIA   \n",
       "1             NJ       Master's             N         N        INDIA   \n",
       "2             CA       Master's             N         N  SOUTH KOREA   \n",
       "3             CA       Master's             N         N   BANGLADESH   \n",
       "4             VA     Bachelor's             N         N        INDIA   \n",
       "...          ...            ...           ...       ...          ...   \n",
       "535057        NY    High School             N         N       POLAND   \n",
       "535058        RI       Master's             N         N        INDIA   \n",
       "535059        NJ    High School             N         N  SOUTH KOREA   \n",
       "535063        CA     Bachelor's             N         N        INDIA   \n",
       "535083        TX       Master's             N         N        INDIA   \n",
       "\n",
       "       ADMISSION_TYPE WORKER_EDUCATION          time_taken  \\\n",
       "0                H-1B       Bachelor's   at least 6 months   \n",
       "1                H-1B         Master's   at least 6 months   \n",
       "2                 L-1         Master's   at least 6 months   \n",
       "3                H-1B         Master's   at least 6 months   \n",
       "4                H-1B         Master's   at least 6 months   \n",
       "...               ...              ...                 ...   \n",
       "535057           H-1B      High School   at least 3 months   \n",
       "535058           H-1B         Master's   at least 3 months   \n",
       "535059            E-2       Bachelor's   at least 3 months   \n",
       "535063           H-1B       Bachelor's  at least 15 months   \n",
       "535083           H-1B       Bachelor's   at least 9 months   \n",
       "\n",
       "                        OCCUPATION NEW_RELATED_MAJOR NEW_WORKER_MAJOR  \\\n",
       "0             computer occupations        STEM Major       STEM Major   \n",
       "1             computer occupations        STEM Major   NON-STEM Major   \n",
       "2             computer occupations        STEM Major       STEM Major   \n",
       "3       Architecture & Engineering        STEM Major       STEM Major   \n",
       "4             computer occupations        STEM Major       STEM Major   \n",
       "...                            ...               ...              ...   \n",
       "535057                      Others        STEM Major       STEM Major   \n",
       "535058  Architecture & Engineering        STEM Major       STEM Major   \n",
       "535059        computer occupations        STEM Major   NON-STEM Major   \n",
       "535063        computer occupations        STEM Major       STEM Major   \n",
       "535083        computer occupations        STEM Major       STEM Major   \n",
       "\n",
       "       NEW_EMPLOYER_NAME             WAGE_OFFERED  \n",
       "0         Other Employer    Between 60000 -100000  \n",
       "1         Other Employer    Between 60000 -100000  \n",
       "2         Other Employer    Between 60000 -100000  \n",
       "3        Top 10 Employer    Between 60000 -100000  \n",
       "4         Other Employer    Between 60000 -100000  \n",
       "...                  ...                      ...  \n",
       "535057    Other Employer     Between 30000 -60000  \n",
       "535058    Other Employer  Between 100000 -1500000  \n",
       "535059    Other Employer  Between 100000 -1500000  \n",
       "535063    Other Employer  Between 100000 -1500000  \n",
       "535083    Other Employer    Between 60000 -100000  \n",
       "\n",
       "[534765 rows x 16 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dffff.drop(['EMPLOYER_STATE'],axis =1,inplace = True)\n",
    "dffff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72def31c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bec6521",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "371dd513",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "513c2943",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48378226",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "6fe46537",
   "metadata": {},
   "source": [
    "### 1. Case approved/ Denial case visualization "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "dbde7e10",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "approved_Case = []\n",
    "denied_case=[]\n",
    "\n",
    "#Approved_case_each_year = df2[df2['Year']==2015][df['CASE_STATUS']=='Certified']['Year'].count()\n",
    "year=[2014,2015,2016,2017,2018,2019]\n",
    "\n",
    "for i in year:\n",
    "    Approved_case_each_year = df[df['Year']==i][df['CASE_STATUS']=='Certified']['Year'].count()\n",
    "    Denied_case_Each_year = df[df['Year']==i][df['CASE_STATUS']=='Denied']['Year'].count()\n",
    "    \n",
    "    approved_Case.append(Approved_case_each_year)\n",
    "    denied_case.append(Denied_case_Each_year)\n",
    "    \n",
    "cases_status_df = pd.DataFrame({'Year':year,\n",
    "                                'Approved_Case':approved_Case,\n",
    "                               'Denied_Case':denied_case})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "568adacd",
   "metadata": {},
   "source": [
    "##### This visualization becomes card content of dash  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9ebf7ae",
   "metadata": {},
   "source": [
    "### 2. By country "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "43a2cef9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Top 15 countries with applications\n",
    "df.head()\n",
    "top_15_countries =df.groupby([\"COUNTRY_OF_CITIZENSHIP\"],as_index =False)['CASE_STATUS'].count()\n",
    "top_15_countries = top_15_countries.sort_values(by='CASE_STATUS',ascending = False).head(15)\n",
    "top_15_countries = top_15_countries.rename(columns={'COUNTRY_OF_CITIZENSHIP': 'COUNTRY',\"CASE_STATUS\":'Total applications'})\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "5684e7ed",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": "indianred"
         },
         "type": "bar",
         "x": [
          "INDIA",
          "CHINA",
          "SOUTH KOREA",
          "CANADA",
          "MEXICO",
          "PHILIPPINES",
          "VIETNAM",
          "BRAZIL",
          "UNITED KINGDOM",
          "TAIWAN",
          "PAKISTAN",
          "VENEZUELA",
          "FRANCE",
          "NEPAL",
          "IRAN"
         ],
         "y": [
          281176,
          50455,
          32606,
          18933,
          15499,
          9303,
          7644,
          6308,
          6198,
          5641,
          5427,
          5006,
          4475,
          4403,
          4005
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Top 15 countries with high perm applications"
        },
        "xaxis": {
         "title": {
          "text": "Countries"
         }
        },
        "yaxis": {
         "title": {
          "text": "Number of perm applications"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"00bf8abf-5316-4433-ba29-36dc953df6b7\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"00bf8abf-5316-4433-ba29-36dc953df6b7\")) {                    Plotly.newPlot(                        \"00bf8abf-5316-4433-ba29-36dc953df6b7\",                        [{\"marker\":{\"color\":\"indianred\"},\"x\":[\"INDIA\",\"CHINA\",\"SOUTH KOREA\",\"CANADA\",\"MEXICO\",\"PHILIPPINES\",\"VIETNAM\",\"BRAZIL\",\"UNITED KINGDOM\",\"TAIWAN\",\"PAKISTAN\",\"VENEZUELA\",\"FRANCE\",\"NEPAL\",\"IRAN\"],\"y\":[281176,50455,32606,18933,15499,9303,7644,6308,6198,5641,5427,5006,4475,4403,4005],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Top 15 countries with high perm applications\"},\"xaxis\":{\"title\":{\"text\":\"Countries\"}},\"yaxis\":{\"title\":{\"text\":\"Number of perm applications\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('00bf8abf-5316-4433-ba29-36dc953df6b7');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "fig  = go.Figure([go.Bar(x = top_15_countries['COUNTRY'],y=top_15_countries['Total applications'], marker_color = 'indianred')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Top 15 countries with high perm applications',\n",
    "                 xaxis_title = 'Countries',\n",
    "                 yaxis_title = 'Number of perm applications')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "f61aaacc",
   "metadata": {},
   "outputs": [],
   "source": [
    "country_list = list(top_15_countries['COUNTRY'])\n",
    "country_list\n",
    "\n",
    "#making \n",
    "\n",
    "approved_count =[]\n",
    "for i in country_list:\n",
    "  country_certified = df[df['COUNTRY_OF_CITIZENSHIP']== i][df['CASE_STATUS']=='Certified']\n",
    "  country_certified_count = country_certified.groupby([\"COUNTRY_OF_CITIZENSHIP\"],as_index =False)['CASE_STATUS'].count()\n",
    "  approved_count.append(country_certified_count.iloc[:,1][0])\n",
    "  \n",
    "top_15_countries['Approved Case'] =approved_count\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "5079f341",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_15_countries['Denied Case'] = top_15_countries['Total applications'] - top_15_countries['Approved Case']\n",
    "top_15_countries['approved_perc'] = round(top_15_countries['Approved Case']/top_15_countries['Total applications'] *100,2)\n",
    "top_15_countries['denied_perc'] = round(top_15_countries['Denied Case']/top_15_countries['Total applications'] *100,2)\n",
    "percent= list(top_15_countries['approved_perc'])\n",
    "for x in list(top_15_countries['denied_perc']):\n",
    "  percent.append(x)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "82e81f66",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>COUNTRY</th>\n",
       "      <th>variable</th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>INDIA</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>94.30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CHINA</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>92.48</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>81.36</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>CANADA</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>91.91</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>MEXICO</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>80.30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>PHILIPPINES</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>80.86</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>VIETNAM</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>66.51</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>BRAZIL</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>87.48</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>UNITED KINGDOM</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>89.79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>TAIWAN</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>91.01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>PAKISTAN</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>88.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>VENEZUELA</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>79.92</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>FRANCE</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>91.93</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>NEPAL</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>91.62</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>IRAN</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>87.09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>INDIA</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>5.70</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>CHINA</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>7.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>18.64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>CANADA</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>8.09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>MEXICO</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>19.70</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>PHILIPPINES</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>19.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>VIETNAM</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>33.49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>BRAZIL</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>12.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>UNITED KINGDOM</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>10.21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>TAIWAN</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>8.99</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>PAKISTAN</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>11.48</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>VENEZUELA</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>20.08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>FRANCE</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>8.07</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>NEPAL</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>8.38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>IRAN</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>12.91</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           COUNTRY       variable  value\n",
       "0            INDIA  Approved Case  94.30\n",
       "1            CHINA  Approved Case  92.48\n",
       "2      SOUTH KOREA  Approved Case  81.36\n",
       "3           CANADA  Approved Case  91.91\n",
       "4           MEXICO  Approved Case  80.30\n",
       "5      PHILIPPINES  Approved Case  80.86\n",
       "6          VIETNAM  Approved Case  66.51\n",
       "7           BRAZIL  Approved Case  87.48\n",
       "8   UNITED KINGDOM  Approved Case  89.79\n",
       "9           TAIWAN  Approved Case  91.01\n",
       "10        PAKISTAN  Approved Case  88.52\n",
       "11       VENEZUELA  Approved Case  79.92\n",
       "12          FRANCE  Approved Case  91.93\n",
       "13           NEPAL  Approved Case  91.62\n",
       "14            IRAN  Approved Case  87.09\n",
       "15           INDIA    Denied Case   5.70\n",
       "16           CHINA    Denied Case   7.52\n",
       "17     SOUTH KOREA    Denied Case  18.64\n",
       "18          CANADA    Denied Case   8.09\n",
       "19          MEXICO    Denied Case  19.70\n",
       "20     PHILIPPINES    Denied Case  19.14\n",
       "21         VIETNAM    Denied Case  33.49\n",
       "22          BRAZIL    Denied Case  12.52\n",
       "23  UNITED KINGDOM    Denied Case  10.21\n",
       "24          TAIWAN    Denied Case   8.99\n",
       "25        PAKISTAN    Denied Case  11.48\n",
       "26       VENEZUELA    Denied Case  20.08\n",
       "27          FRANCE    Denied Case   8.07\n",
       "28           NEPAL    Denied Case   8.38\n",
       "29            IRAN    Denied Case  12.91"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "top_15_countries_perc = top_15_countries.drop(['Total applications'], axis =1)\n",
    "# multiple unpivot columns\n",
    "top_15_countries_perc = pd.melt(top_15_countries_perc, id_vars =['COUNTRY'], value_vars=['Approved Case','Denied Case'])\n",
    "top_15_countries_perc['value'] =percent\n",
    "top_15_countries_perc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "6175e1ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hole": 0.3,
         "labels": [
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case"
         ],
         "type": "pie",
         "values": [
          94.3,
          92.48,
          81.36,
          91.91,
          80.3,
          80.86,
          66.51,
          87.48,
          89.79,
          91.01,
          88.52,
          79.92,
          91.93,
          91.62,
          87.09,
          5.7,
          7.52,
          18.64,
          8.09,
          19.7,
          19.14,
          33.49,
          12.52,
          10.21,
          8.99,
          11.48,
          20.08,
          8.07,
          8.38,
          12.91
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"5508cc9e-32c8-4ff7-9231-939e7b782b35\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"5508cc9e-32c8-4ff7-9231-939e7b782b35\")) {                    Plotly.newPlot(                        \"5508cc9e-32c8-4ff7-9231-939e7b782b35\",                        [{\"hole\":0.3,\"labels\":[\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\"],\"values\":[94.3,92.48,81.36,91.91,80.3,80.86,66.51,87.48,89.79,91.01,88.52,79.92,91.93,91.62,87.09,5.7,7.52,18.64,8.09,19.7,19.14,33.49,12.52,10.21,8.99,11.48,20.08,8.07,8.38,12.91],\"type\":\"pie\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('5508cc9e-32c8-4ff7-9231-939e7b782b35');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Use `hole` to create a donut-like pie chart\n",
    "fig = go.Figure(data=[go.Pie(labels=top_15_countries_perc['variable'], values=top_15_countries_perc['value'], hole=.3)])\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1843035a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "625c125f",
   "metadata": {},
   "source": [
    "## 3. By Visa Type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "3d2c6fc5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CLASS_OF_ADMISSION</th>\n",
       "      <th>Total applications</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>H-1B</td>\n",
       "      <td>370433</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>L-1</td>\n",
       "      <td>34647</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>F-1</td>\n",
       "      <td>30647</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>Not in USA</td>\n",
       "      <td>19650</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>TN</td>\n",
       "      <td>8753</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>E-2</td>\n",
       "      <td>6897</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>B-2</td>\n",
       "      <td>5678</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>J-1</td>\n",
       "      <td>2075</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>EWI</td>\n",
       "      <td>2073</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>E-3</td>\n",
       "      <td>1687</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CLASS_OF_ADMISSION  Total applications\n",
       "18               H-1B              370433\n",
       "30                L-1               34647\n",
       "12                F-1               30647\n",
       "35         Not in USA               19650\n",
       "51                 TN                8753\n",
       "9                 E-2                6897\n",
       "3                 B-2                5678\n",
       "26                J-1                2075\n",
       "11                EWI                2073\n",
       "10                E-3                1687"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Top 10 countries with applications\n",
    "df.head()\n",
    "top_10_visa =df.groupby([\"CLASS_OF_ADMISSION\"],as_index =False)['CASE_STATUS'].count()\n",
    "top_10_visa = top_10_visa.sort_values(by='CASE_STATUS',ascending = False).head(10)\n",
    "top_10_visa = top_10_visa.rename(columns={\"CASE_STATUS\":'Total applications'})\n",
    "top_10_visa"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "2fb6317a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": "blue"
         },
         "type": "bar",
         "x": [
          "H-1B",
          "L-1",
          "F-1",
          "Not in USA",
          "TN",
          "E-2",
          "B-2",
          "J-1",
          "EWI",
          "E-3"
         ],
         "y": [
          370433,
          34647,
          30647,
          19650,
          8753,
          6897,
          5678,
          2075,
          2073,
          1687
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Top 10 class of admissions with high perm applications"
        },
        "xaxis": {
         "title": {
          "text": "Countries"
         }
        },
        "yaxis": {
         "title": {
          "text": "Number of class of admissions"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"7fbad415-c40e-4a6c-9953-ac421bfc5cbe\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"7fbad415-c40e-4a6c-9953-ac421bfc5cbe\")) {                    Plotly.newPlot(                        \"7fbad415-c40e-4a6c-9953-ac421bfc5cbe\",                        [{\"marker\":{\"color\":\"blue\"},\"x\":[\"H-1B\",\"L-1\",\"F-1\",\"Not in USA\",\"TN\",\"E-2\",\"B-2\",\"J-1\",\"EWI\",\"E-3\"],\"y\":[370433,34647,30647,19650,8753,6897,5678,2075,2073,1687],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Top 10 class of admissions with high perm applications\"},\"xaxis\":{\"title\":{\"text\":\"Countries\"}},\"yaxis\":{\"title\":{\"text\":\"Number of class of admissions\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('7fbad415-c40e-4a6c-9953-ac421bfc5cbe');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig  = go.Figure([go.Bar(x = top_10_visa['CLASS_OF_ADMISSION'],y=top_10_visa['Total applications'], marker_color = 'blue')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Top 10 class of admissions with high perm applications',\n",
    "                 xaxis_title = 'Countries',\n",
    "                 yaxis_title = 'Number of class of admissions')\n",
    "                 #barmode = 'group')\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "84842654",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CLASS_OF_ADMISSION</th>\n",
       "      <th>variable</th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>H-1B</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>93.65</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>L-1</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>95.99</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F-1</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>87.74</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Not in USA</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>76.75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>TN</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>93.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>E-2</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>83.17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>B-2</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>73.02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>J-1</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>86.12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>EWI</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>62.42</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>E-3</td>\n",
       "      <td>Approved Case</td>\n",
       "      <td>91.29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>H-1B</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>6.35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>L-1</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>4.01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>F-1</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>12.26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>Not in USA</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>23.25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>TN</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>6.50</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>E-2</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>16.83</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>B-2</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>26.98</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>J-1</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>13.88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>EWI</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>37.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>E-3</td>\n",
       "      <td>Denied Case</td>\n",
       "      <td>8.71</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CLASS_OF_ADMISSION       variable  value\n",
       "0                H-1B  Approved Case  93.65\n",
       "1                 L-1  Approved Case  95.99\n",
       "2                 F-1  Approved Case  87.74\n",
       "3          Not in USA  Approved Case  76.75\n",
       "4                  TN  Approved Case  93.50\n",
       "5                 E-2  Approved Case  83.17\n",
       "6                 B-2  Approved Case  73.02\n",
       "7                 J-1  Approved Case  86.12\n",
       "8                 EWI  Approved Case  62.42\n",
       "9                 E-3  Approved Case  91.29\n",
       "10               H-1B    Denied Case   6.35\n",
       "11                L-1    Denied Case   4.01\n",
       "12                F-1    Denied Case  12.26\n",
       "13         Not in USA    Denied Case  23.25\n",
       "14                 TN    Denied Case   6.50\n",
       "15                E-2    Denied Case  16.83\n",
       "16                B-2    Denied Case  26.98\n",
       "17                J-1    Denied Case  13.88\n",
       "18                EWI    Denied Case  37.58\n",
       "19                E-3    Denied Case   8.71"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "visa_list = list(top_10_visa['CLASS_OF_ADMISSION'])\n",
    "visa_list\n",
    "\n",
    "\n",
    "#making \n",
    "\n",
    "approved_count =[]\n",
    "for i in visa_list:\n",
    "  visa_certified = df[df['CLASS_OF_ADMISSION']== i][df['CASE_STATUS']=='Certified']\n",
    "  visa_certified_count = visa_certified.groupby([\"CLASS_OF_ADMISSION\"],as_index =False)['CASE_STATUS'].count()\n",
    "  approved_count.append(visa_certified_count.iloc[:,1][0])\n",
    "  \n",
    "top_10_visa['Approved Case'] =approved_count\n",
    "top_10_visa['Denied Case'] = top_10_visa['Total applications'] - top_10_visa['Approved Case']\n",
    "top_10_visa['approved_perc'] = round(top_10_visa['Approved Case']/top_10_visa['Total applications'] *100,2)\n",
    "top_10_visa['denied_perc'] = round(top_10_visa['Denied Case']/top_10_visa['Total applications'] *100,2)\n",
    "percent= list(top_10_visa['approved_perc'])\n",
    "for x in list(top_10_visa['denied_perc']):\n",
    "  percent.append(x)\n",
    "\n",
    "top_10_visa_perc = top_10_visa.drop(['Total applications'], axis =1)\n",
    "# multiple unpivot columns\n",
    "top_10_visa_perc = pd.melt(top_10_visa_perc, id_vars =['CLASS_OF_ADMISSION'], value_vars=['Approved Case','Denied Case'])\n",
    "top_10_visa_perc['value'] =percent\n",
    "top_10_visa_perc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "26d7dd55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hole": 0.3,
         "labels": [
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Approved Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case",
          "Denied Case"
         ],
         "type": "pie",
         "values": [
          94.3,
          92.48,
          81.36,
          91.91,
          80.3,
          80.86,
          66.51,
          87.48,
          89.79,
          91.01,
          88.52,
          79.92,
          91.93,
          91.62,
          87.09,
          5.7,
          7.52,
          18.64,
          8.09,
          19.7,
          19.14,
          33.49,
          12.52,
          10.21,
          8.99,
          11.48,
          20.08,
          8.07,
          8.38,
          12.91
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"33c757db-e5c8-4715-9c0d-bd0100fe45ea\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"33c757db-e5c8-4715-9c0d-bd0100fe45ea\")) {                    Plotly.newPlot(                        \"33c757db-e5c8-4715-9c0d-bd0100fe45ea\",                        [{\"hole\":0.3,\"labels\":[\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Approved Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\",\"Denied Case\"],\"values\":[94.3,92.48,81.36,91.91,80.3,80.86,66.51,87.48,89.79,91.01,88.52,79.92,91.93,91.62,87.09,5.7,7.52,18.64,8.09,19.7,19.14,33.49,12.52,10.21,8.99,11.48,20.08,8.07,8.38,12.91],\"type\":\"pie\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('33c757db-e5c8-4715-9c0d-bd0100fe45ea');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Use `hole` to create a donut-like pie chart\n",
    "fig = go.Figure(data=[go.Pie(labels=top_15_countries_perc['variable'], values=top_15_countries_perc['value'], hole=.3)])\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5cdeaae",
   "metadata": {},
   "source": [
    "### 4. Top sponsoring employers over different years\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "ab9e91c1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['SALESFORCE.COM',\n",
       " 'HOUSE OF RAEFORD FARMS, INC.',\n",
       " 'LINKEDIN CORPORATION',\n",
       " 'IBM CORPORATION',\n",
       " 'VMWARE, INC.',\n",
       " 'KFORCE INC.',\n",
       " 'WAYNE FARMS LLC',\n",
       " 'CAPGEMINI AMERICA, INC.',\n",
       " 'YAHOO! INC.',\n",
       " 'CAPITAL ONE SERVICES, LLC']"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ax=df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[20:30]#.plot.barh(width=0.9,color='#ffd700')\n",
    "ax = ax.to_frame().reset_index().rename(columns={'index':'Employer Name','EMPLOYER_NAME':'Total_count'})\n",
    "ax\n",
    "\n",
    "ax[\"Employer Name\"] = ax[\"Employer Name\"].replace(\"GOOGLE INC.\", \"GOOGLE LLC\")\n",
    "list(ax[\"Employer Name\"])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "f43bd321",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": "green"
         },
         "orientation": "h",
         "type": "bar",
         "x": [
          1625,
          1466,
          1316,
          1273,
          1273,
          1251,
          1217,
          1211,
          1152,
          1121
         ],
         "y": [
          "SALESFORCE.COM",
          "HOUSE OF RAEFORD FARMS, INC.",
          "LINKEDIN CORPORATION",
          "IBM CORPORATION",
          "VMWARE, INC.",
          "KFORCE INC.",
          "WAYNE FARMS LLC",
          "CAPGEMINI AMERICA, INC.",
          "YAHOO! INC.",
          "CAPITAL ONE SERVICES, LLC"
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Top sponsoring employers "
        },
        "xaxis": {
         "title": {
          "text": "Name of employers"
         }
        },
        "yaxis": {
         "title": {
          "text": "PERM applications"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"09483565-2831-4473-a122-147c791e39b1\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"09483565-2831-4473-a122-147c791e39b1\")) {                    Plotly.newPlot(                        \"09483565-2831-4473-a122-147c791e39b1\",                        [{\"marker\":{\"color\":\"green\"},\"orientation\":\"h\",\"x\":[1625,1466,1316,1273,1273,1251,1217,1211,1152,1121],\"y\":[\"SALESFORCE.COM\",\"HOUSE OF RAEFORD FARMS, INC.\",\"LINKEDIN CORPORATION\",\"IBM CORPORATION\",\"VMWARE, INC.\",\"KFORCE INC.\",\"WAYNE FARMS LLC\",\"CAPGEMINI AMERICA, INC.\",\"YAHOO! INC.\",\"CAPITAL ONE SERVICES, LLC\"],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Top sponsoring employers \"},\"yaxis\":{\"title\":{\"text\":\"PERM applications\"}},\"xaxis\":{\"title\":{\"text\":\"Name of employers\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('09483565-2831-4473-a122-147c791e39b1');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig  = go.Figure([go.Bar(x = ax['Total_count'], y =ax['Employer Name'], marker_color = 'green',orientation='h')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Top sponsoring employers ',\n",
    "                 yaxis_title = 'PERM applications',\n",
    "                 xaxis_title = 'Name of employers')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84d8ebef",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "cd17d444",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EMPLOYER_NAME</th>\n",
       "      <th>Year</th>\n",
       "      <th>CASE_STATUS</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>AMAZON CORPORATE LLC</td>\n",
       "      <td>2014</td>\n",
       "      <td>239</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>AMAZON CORPORATE LLC</td>\n",
       "      <td>2015</td>\n",
       "      <td>845</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>AMAZON CORPORATE LLC</td>\n",
       "      <td>2016</td>\n",
       "      <td>1650</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AMAZON CORPORATE LLC</td>\n",
       "      <td>2017</td>\n",
       "      <td>1915</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>AMAZON CORPORATE LLC</td>\n",
       "      <td>2018</td>\n",
       "      <td>1586</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>AMAZON CORPORATE LLC</td>\n",
       "      <td>2019</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION</td>\n",
       "      <td>2014</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION</td>\n",
       "      <td>2015</td>\n",
       "      <td>7992</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION</td>\n",
       "      <td>2016</td>\n",
       "      <td>3622</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION</td>\n",
       "      <td>2017</td>\n",
       "      <td>4072</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION</td>\n",
       "      <td>2018</td>\n",
       "      <td>399</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION</td>\n",
       "      <td>2019</td>\n",
       "      <td>2580</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>2014</td>\n",
       "      <td>273</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>2015</td>\n",
       "      <td>1722</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>2016</td>\n",
       "      <td>1793</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>2017</td>\n",
       "      <td>1477</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>GOOGLE INC.</td>\n",
       "      <td>2018</td>\n",
       "      <td>536</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>2014</td>\n",
       "      <td>79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>2015</td>\n",
       "      <td>1984</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>2016</td>\n",
       "      <td>2075</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>2017</td>\n",
       "      <td>1756</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>2018</td>\n",
       "      <td>1618</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>INTEL CORPORATION</td>\n",
       "      <td>2019</td>\n",
       "      <td>1520</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>MICROSOFT CORPORATION</td>\n",
       "      <td>2014</td>\n",
       "      <td>528</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>MICROSOFT CORPORATION</td>\n",
       "      <td>2015</td>\n",
       "      <td>602</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>MICROSOFT CORPORATION</td>\n",
       "      <td>2016</td>\n",
       "      <td>3513</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>MICROSOFT CORPORATION</td>\n",
       "      <td>2017</td>\n",
       "      <td>2033</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>MICROSOFT CORPORATION</td>\n",
       "      <td>2018</td>\n",
       "      <td>2228</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>MICROSOFT CORPORATION</td>\n",
       "      <td>2019</td>\n",
       "      <td>1045</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                    EMPLOYER_NAME  Year  CASE_STATUS\n",
       "0                            AMAZON CORPORATE LLC  2014          239\n",
       "1                            AMAZON CORPORATE LLC  2015          845\n",
       "2                            AMAZON CORPORATE LLC  2016         1650\n",
       "3                            AMAZON CORPORATE LLC  2017         1915\n",
       "4                            AMAZON CORPORATE LLC  2018         1586\n",
       "5                            AMAZON CORPORATE LLC  2019            1\n",
       "6   COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION  2014           20\n",
       "7   COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION  2015         7992\n",
       "8   COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION  2016         3622\n",
       "9   COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION  2017         4072\n",
       "10  COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION  2018          399\n",
       "11  COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION  2019         2580\n",
       "12                                    GOOGLE INC.  2014          273\n",
       "13                                    GOOGLE INC.  2015         1722\n",
       "14                                    GOOGLE INC.  2016         1793\n",
       "15                                    GOOGLE INC.  2017         1477\n",
       "16                                    GOOGLE INC.  2018          536\n",
       "17                              INTEL CORPORATION  2014           79\n",
       "18                              INTEL CORPORATION  2015         1984\n",
       "19                              INTEL CORPORATION  2016         2075\n",
       "20                              INTEL CORPORATION  2017         1756\n",
       "21                              INTEL CORPORATION  2018         1618\n",
       "22                              INTEL CORPORATION  2019         1520\n",
       "23                          MICROSOFT CORPORATION  2014          528\n",
       "24                          MICROSOFT CORPORATION  2015          602\n",
       "25                          MICROSOFT CORPORATION  2016         3513\n",
       "26                          MICROSOFT CORPORATION  2017         2033\n",
       "27                          MICROSOFT CORPORATION  2018         2228\n",
       "28                          MICROSOFT CORPORATION  2019         1045"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAEWCAYAAACKSkfIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAABCRElEQVR4nO3deXgUVdbA4d8hIgQEHRYRCBBIAoEkJEJEwY1lRBAFxI1lCIqKOoiAgLIoMCM7KIoDKoosI4ZFPxZxxEHABR1EwCiILBGDoICIC/sWzvdHV8rupJN0ICSEPu/z5Onue29V3dtJn67cqjolqooxxpjgUKywO2CMMabgWNA3xpggYkHfGGOCiAV9Y4wJIhb0jTEmiFjQN8aYIHJRYXcgNxUqVNDw8PDC7oYxxhQp69at+0VVK2YuP++Dfnh4OGvXri3sbhhjTJEiIjv8ldv0jjHGBBEL+sYYE0Qs6BtjTBA57+f08+LkyZPs2rWLY8eOFXZXjDGmQJQsWZKwsDCKFy8eUPuAgr6I9AUeABTYANwHlALmAuFAGnC3qv7mtB8E3A+kA4+p6vtOeUNgBhAK/AforfmY8W3Xrl2UKVOG8PBwRCS/VmuMMeclVWX//v3s2rWLmjVrBrRMrtM7IlIVeAxIVNVYIAToCAwElqtqFLDceY2I1HPqY4BWwBQRCXFW9xLQA4hyfloFPrzcHTt2jPLly1vAN8YEBRGhfPnyeZrdCHRO/yIgVEQuwrOH/xPQDpjp1M8E2jvP2wFzVPW4qn4PpAKNRKQyUFZV/+fs3c/yWibfWMA3xgSTvMa8XIO+qv4ITAB+AHYDf6jqf4FKqrrbabMbuNxZpCqw02sVu5yyqs7zzOXGGGMKSCDTO3/Bs/deE6gClBaRv+W0iJ8yzaHc3zZ7iMhaEVm7b9++3LpoMvz0ZfY/RVRISAgJCQnuz5gxYwBo2rQp1atXx/uQUPv27bnkkksASEtLIzQ0lISEBOrVq8fDDz/M6dOnSUtLIzY2Nst2du3aRbt27YiKiiIiIoLevXtz4sQJBg8ezJNPPum227FjB7Vq1eL333+nadOm1KlTx+3bnXfeCcDw4cOpWrWqu+3k5OQcx3jvvfdStWpVjh8/DsAvv/xC5qvQJ06cSMmSJfnjjz/csg8//BARYdq0aW7Zl19+iYgwYcIEd901a9Z0+9ikSZNc33NzYQtkeuevwPequk9VTwL/BzQB9jpTNjiPPzvtdwHVvJYPwzMdtMt5nrk8C1WdqqqJqppYsWKWq4hNEAkNDSUlJcX9GThwoFt32WWX8emnnwLw+++/s3v3bp9lIyIiSElJ4euvv2bTpk0sXLjQ7zZUlQ4dOtC+fXu2bdvG1q1bOXToEEOGDOHpp59m0aJFfPvttwD07t2bZ555hssuuwyA2bNnu31766233HX27duXlJQUFi1axEMPPcTJkydzHGdISAivv/56tvXJyclcddVVLFiwwKc8Li6OuXPnuq/nzJlDfHy8T5vx48e7ffzss89y7Ie58AUS9H8ArhGRUuKZPGoBfAssBro5bboBi5zni4GOIlJCRGriOWC7xpkCOigi1zjrSfJaxpg869ixI3PmzAHg//7v/+jQoYPfdhdddBFNmjQhNTXVb/2KFSsoWbIk9913H+AJwBMnTuT1119HVXnuuef4+9//znvvvcfBgwfp0qVLwH2MioqiVKlS/Pbbbzm269OnDxMnTuTUqVNZ6r777jsOHTrEiBEjsvzXUL16dY4dO8bevXtRVZYuXUrr1q0D7p8JPoHM6X8OvAWsx3O6ZjFgKjAGuElEtgE3Oa9R1W+AecAmYCnQU1XTndU9AryG5+Dud8B7+TkYc+E5evSoz/SO915tixYt+Pjjj0lPT2fOnDncc889ftdx5MgRli9fTlxcnN/6b775hoYNG/qUlS1blurVq5Oamsott9xCuXLlSEpKYsqUKT7tunTp4vZtwIABWda9fv16oqKiuPzyy7PUeatevTrXXXcd//73v7PUJScn06lTJ66//nq2bNnCzz//7FN/5513Mn/+fD777DMaNGhAiRIlfOoHDBjg9jEvX1jmwhTQefqqOgwYlqn4OJ69fn/tRwIj/ZSvBbJOqBqTjYzpHX9CQkK47rrrmDt3LkePHs0yD/7dd9+RkJCAiNCuXTtat25NWlpalvWoqt8zILzLe/bsydGjR6lTp45Pm9mzZ5OYmJhl2YkTJ/Lqq6+yfft2li5dGtBYBw8eTNu2bWnTpo1P+Zw5c1iwYAHFihWjQ4cOzJ8/n549e7r1d999N/fccw+bN2+mU6dOWaZwxo8f7x5vMMbSMJgirWPHjvTq1Yu77747S13GnP6XX37J8OHDs11HTExMlkyuBw4cYOfOnURERABQrFgxihUL/OPSt29ftmzZwty5c0lKSgroPOrIyEgSEhKYN2+eW/b111+zbds2brrpJsLDw5kzZ06WKZ4rrriC4sWLs2zZMlq08LsfZozLgr4p0q6//noGDRpEp06dzngdLVq04MiRI8yaNQuA9PR0+vXrx7333kupUqXOqn8dOnQgMTGRmTNn5t4YGDJkiHvmDXimdoYPH05aWhppaWn89NNP/Pjjj+zY4Zs195///Cdjx44lJCQk8yqN8WFB35zXMs/pe5+9A54LU/r370+FChUCXueWLVsICwtzf9566y0WLFjA/PnziYqKonbt2pQsWZJRo0blui7vOf2//vWvftsMHTqU5557jtOnT+e6vpiYGBo0aOC+njNnDrfffrtPm9tvv909gJ2hSZMmtG/f3u86vef0ExISOHHiRK79MBcuycfUN+dEYmKiBnoTlW+//Za6deue4x6dx3I6H7/KlQXXD2NMgfIX+0RknapmOeBke/rGGBNELqjUysacz3r27OleTJahd+/e7vUBxhQEC/rGFJDJkycXdheMsekdY4wJJhb0jTEmiFjQN8aYIGJB/xxYsGABIsLmzZvdsrS0NESEp59+2i375ZdfKF68OI8++qjP8vHx8T4XG6Wnp/ucZ52QkECFChXcXDMnTpygT58+RDRpS9S17Wh3X192/bTXXV6qNqBfv37u6wkTJmR7hep7771HYmIidevWJTo6mv79+7t1U6dOJTo6mujoaBo1asSqVavcuow0w/Hx8Vx11VU+qRPCw8OJi4sjPj6eli1bsmfPHgD++OMPkpKSiIiIICIigqSkJDd1cObUyElJST6ZKk+dOkWFChUYNGgQACNHjnTfG+90zJMmTfJJdZzx8/vvv/uMO7uUy/fee69P9swMW7du5ZZbbiEyMpK6dety9913s3fv3iztjDnfXNAHcsMHvpuv60sb0yb3RniuorzuuuuYM2eOT3CtVasWS5Ys4ZlnngFg/vz5xMTE+Cz77bffcvr0aT7++GMOHz5M6dKlCQkJ8Qmiu3fvplGjRu4XyODBgzl48CBbP1lASEgI0+cuosOD/fl8ySxEhBIlLub//u//GDRoUI4XMW3cuJFHH32Ud999l+joaE6dOsXUqVMBWLJkCa+88gqrVq2iQoUKrF+/nvbt27NmzRquuOIK4M88NNOnT2fAgAEsW7bMXffKlSupUKECgwcPZtSoUUyaNIn777+f2NhY90rYYcOG8cADDzB//nzgzzQK6enp3HTTTcybN89NGPbf//6XOnXqMG/ePEaNGsWQIUMYMmQIAJdcconP+zV8+HD69u3r8wV2No4dO0abNm147rnnuO2229zx7du3j0qVKuXLNow5V2xPP58dOnSITz/9lGnTpmW5ajI0NJS6deu6eV7mzp2bJWfMm2++SdeuXWnZsiWLFy/Osn5VpVu3bgwYMIDY2FiOHDnC9OnTmThxonsJ/n33tKPExcVZsWoNABeFhNCjRw8mTpyYY9/HjRvHkCFDiI6O9ix30UX8/e9/B2Ds2LGMHz/e/dJo0KAB3bp183tGSuPGjfnxxx/9buOGG24gNTWV1NRU1q1b5/Ofz9ChQ1m7di3fffedzzIhISE0atTIZ53Jycn07t2b6tWrs3r16hzHld/efPNNGjdu7AZ8gGbNmvn9T8GY840F/Xy2cOFCWrVqRe3atSlXrhzr16/3qc/IAb9r1y5CQkKoUqWKT/3cuXO555576NSpk987Lk2cOJGLLrqIXr16AZCamkr16tUpW7asT7vE+vX4Zut293XPnj2ZPXu2z52XMtu4cWOWFMMZ/KUfTkxM5JtvvsnSdunSpdmmBFiyZAlxcXFs2rTJnYrJkDEtk3mdx44d4/PPP6dVq1aAJzXD8uXLufXWW7N9nzKbOHGiO7XTrFmzXNvnJKf3yZjznQX9fJacnEzHjh0BT4DPHJBatWrFsmXLSE5OzpL//YsvvqBixYrUqFGDFi1asH79ep+bb3z11Vc8//zzTJ8+3U35G0haYPDkh09KSmLSpEn5NtbM2+jSpQthYWGMHTvW/VLK0KxZMxISEjhw4ACDBg0KqN8ZqZHLly9P9erVqV+/PuD54mjWrBmlSpXijjvuYMGCBaSnp2dZl7eMO1mlpKSwcuXKsx26MUWWBf18tH//flasWMEDDzxAeHg448ePZ+7cuT73cb344otp2LAhzz77LHfccYfP8snJyWzevJnw8HAiIiI4cOAAb7/9NuDZu+3SpQtTpkzxmTeOjIxkx44dHDx40Gdd6zdupl5UTZ+yPn36MG3aNA4fPuy3/zExMaxbt85vXb169bLUrV+/nnr16rmvZ8+ezffff0/nzp198r2DZ847JSWFWbNmcdlllxETE8OXX37pk4Ts9OnTfPXVV24OkYw5/dTUVFavXu1OdyUnJ/PBBx8QHh5Ow4YN2b9/f4EG8pzeJ2POdxb089Fbb71FUlISO3bsIC0tjZ07d1KzZk2fs1wA+vXrx9ixYylfvrxbdvr0aebPn8/XX3/tptFdtGiR+59C//79ufHGG7n11lt91lW6dGm6devG448/7u7tzpq/hCNHj9H8ukY+bcuVK8fdd9/tcyNtbwMGDGDUqFFs3brV7dNzzz0HwBNPPMGTTz7J/v37AUhJSWHGjBnunH+G4sWLM2LECFavXu3eV9afyMhIrrzySkaMGOGWjRgxggYNGhAZGenTtnLlyowZM4bRo0dz4MABVq1axQ8//OC+T5MnTw5oiie/dO7cmc8++4x33/3zRIGlS5eyYcOGAuuDMWcq16AvInVEJMXr54CI9BGRciKyTES2OY9/8VpmkIikisgWEbnZq7yhiGxw6iaJv//vi7Dk5OQsaXDvuOMO3nzzTZ+ymJgYunXr5lP28ccfU7VqVapWreqW3XDDDWzatIkff/yRKVOmsGLFCp/TDjPOZBk9ejQlS5ak9vW3E3VtO+YvWcaC1571O33Sr18/fvnlF7/9r1+/Ps8//zydOnWibt26xMbGujcbb9u2Ld27d6dJkyZER0fz4IMP8sYbb1C5cuUs6wkNDaVfv34+eeH9mTZtGlu3biUyMpKIiAi2bt2a7RdS+/btOXLkCC+88ALNmzf3uSVgu3btWLx4McePH892W95z+gkJCX7voJU55XLGWUQPPfSQW9a4cWNCQ0NZsmQJL774IlFRUdSrV48ZM2bkektEY84HeUqtLCIhwI/A1UBP4FdVHSMiA4G/qOqTIlIPSAYaAVWAD4DaqpouImuA3sBq4D/AJFXN8T65llo5Dyy1sjFB6VymVm4BfKeqO4B2QMbtgGYC7Z3n7YA5qnpcVb/HcxP0RiJSGSirqv9TzzfNLK9ljDHGFIC8Bv2OePbiASqp6m4A5zHjf9uqwE6vZXY5ZVWd55nLjTHGFJCAg76IXAy0Bebn1tRPmeZQ7m9bPURkrYis3bdvX6BdNMYYk4u87Om3BtarakaCkb3OlA3O489O+S6gmtdyYcBPTnmYn/IsVHWqqiaqamLFihXz0EVjjDE5yUvQ78SfUzsAi4GMU1C6AYu8yjuKSAkRqQlEAWucKaCDInKNc9ZOktcyxhhjCkBACddEpBRwE/CQV/EYYJ6I3A/8ANwFoKrfiMg8YBNwCuipqhmXSz4CzABCgfecH2OMMQVFVc/rn4YNG2qgNm3aFHDbc2X37t16zz33aK1atbRu3braunVr3bJli6qqbty4UZs1a6ZRUVEaGRmp//znP/X06dPusu+9955eddVVWqdOHY2Pj9e7775bd+zYoaqq3bp10ypVquixY8dUVXXfvn1ao0YNVVX9/vvvNSYmRvXH9Tq4V3eNr1fb/YmqWV2LFSumBw8edLfTtm1bveaaa3z6PWzYMA0NDdW9e/e6ZaVLl9ZffvlF4+PjNT4+XitVqqRVqlRxXx8/ftxt26hRI42Pj9dq1apphQoV3Dbff/+91qhRQ2NjY92yXr16ucuNHz9e69SpozExMVq/fn2dOXOmqqreeOON+sUXX7jt3DGq6sqVKxXQxYsXu/Vt2rTRlStXqqrq8ePHtXfv3lqrVi2NjIzUtm3b6s6dO33Gldnvv/+uXbt21Vq1ammtWrW0a9eu+vvvv7v1W7du1TZt2mitWrW0QYMG2rRpU/3oo4+0U6dOOmXKFLfd6tWrNS4uTk+ePOmz/nfeeUcTEhK0fv36WrduXX355ZfduldeeUXr1KmjderU0auuuko/+eQTty7z+6CqWqNGDd23b5/7euXKldqmTRt9/fXX3fe4ePHi7nv+5JNP6vTp07Vnz54Bb9P7c/fFF1/ojTfeqKqqhw8f1s6dO2tsbKzGxMTotdde6/O3ld177L39zZs364033qjx8fEaHR2tDz74YJblVVW3bNmirVu31oiICI2Ojta77rpL9+zZo6qqn3zyiftZqVOnjr7yyivucsOGDXP/TuvWratvvvmmW9etWzcNDw/X+Ph4vfLKK/Wzzz5TVdXTp0/rM888o5GRkRoVFaVNmzbVjRs3+rznsbGxGhcXpzfccIOmpaX59NX7M7V06VL391C6dGmtXbu2xsfHa9euXd3fVYYFCxZoXFyc1qlTR2NjY3XBggU+fc3uM5+Zv9gHrFU/MbXQg3puP2cV9IeVzd+fXJw+fVqvueYafemll9yyL7/8Uj/++GM9cuSI1qpVS99//31V9Xx4WrVqpf/6179UVXXDhg0aGRnpM4ZFixbpRx99pKqeP4Bq1aq5ASa7oJ/5p/PtrXXIY/e76/ztt980LCxMo6Ojdfv27X++VcOGabVq1fSJJ55wyzJ/cIcNG6bjx4/P8T3IHFxUswapDC+99JK2bNlS//jjD1X1BN4ZM2aoau5BPywsTK+++mq33jvo9+vXT7t3766nTp1SVdXXX39dr7rqKvcL1l/Qv+OOO3TYsGHu66FDh+qdd96pqqpHjx7VqKgoXbRokVu/YcMGnT59uu7Zs0dr1qypP//8s6anp2tiYqJPAFVVPXHihFauXNn94jl27Jhu3rxZVT1fBg0aNHDfn3Xr1mm1atV09+7dft8Hf+9n5kDir4337yWQbVarVk3/85//qKpv0B81apT27dvXXe/mzZvdoOQtp6DfsmVLXbhwoVv39ddfZ1n+6NGjGhkZ6fPFvmLFCt2wYYPu3r1bq1WrpuvWrVNVz2ehQYMGumTJElX1/TvdunWrlilTRk+cOKGqns/R/PnzVVX1/fff17i4OFVVffHFF7V169Z6+PBht65WrVp69OjRLO/n0KFD9YEHHnD7ld1nKuO99P79ef+uUlJSNCIiwl1m+/btGhERoV999ZXb1+w+85nlJehbGoZ8tHLlSooXL87DDz/sliUkJHD99dfz5ptvcu2119KyZUsASpUqxb/+9S/GjBkDeFIXDx482OcCi7Zt23LDDTe4r/v06cPEiRM5depUQP154+13SU3byfB+f87Kvf3229x2221utk9v3bt3Z+7cufz66695H/wZGDVqFFOmTHEzhF566aVZrlTOTnx8PJdeeqlPzn7Af6rp++6jRIkSrFixwu+6ckvzPHv2bBo3bkzbtm3d+tjYWO69914qVapE//79eeKJJ3j55ZepX78+1113nc/6Dx48yKlTp9y0GyVKlKBOnTpA3lJW55dAtjlgwACfFBkZdu/e7XPVeJ06dXyujg7E7t27CQv785yOuLi4LG1ySl89efJk7r33Xho0aABAhQoVGDdunPtZ8hYVFUWpUqV8EhdmyEjzDZ735MUXX6RUqVIAtGzZkiZNmjB79uwsy2VOHZ7TZyonEyZMYPDgwdSs6cmRVbNmTQYNGsT48ePdNnn9zAfCgn4+ymtq4oiICA4dOsSBAwf45ptv3D/i7FSvXp3rrruOf//737n2JW3nTwwc9SKzXxzBRRf9eegmOTmZTp06+U1JfMkll9C9e3deeOGFXNefVxlZNhMSEpg4cSIHDx7k4MGDREREZLtMly5d3GVuueWWLPVPPfVUlsCUbarpbNJAA7mmec7td/Pwww+zadMmxo8fz7hx47LUlytXjrZt21KjRg06derE7Nmz3URzeUlZnV8C2Wbjxo0pUaJElkR23bt3Z+zYsTRu3JinnnqKbdu25Xn7ffv2pXnz5rRu3ZqJEydmuYsZ5F+a7/Xr1xMVFeU3RcY777xDXFwcBw4c4PDhw1n+FgNNHZ7TZyongYwjL5/5QFnQLyCaTSphIEv5/v37SUhIoHbt2lny1wwePJjx48f7ZKfMLD09nb/1eopnnniEyJrV3fK9e/eSmprKddddR+3atbnooovYuHGjz7KPPfYYM2fO5MCBA3kdYo4ysmympKTQt2/fHN+PDLNnz3aX+c9//pOl/vrrrwfgk08+ccuyW29O28vrMrfffjuxsbF06NABgGLFivHQQw/RunVrnyR63l577TWWL19Oo0aNmDBhAt27d/fbLre+Qta/l+zK8sLfNv19qSYkJLB9+3YGDBjAr7/+ylVXXZVjYj1/fbzvvvv49ttvueuuu/jwww+55pprcsybFEhfvdcPnlxLderU4eqrr85ya9ABAwaQkJDA1KlTs8315G87zZo14/LLL+eDDz6gc+fOQGCfqbyMw19ZIJ/5vLCgn49ySrkbExND5hxC27dv55JLLqFMmTLExMS4N1wpX748KSkp9OjRg0OHDvksExkZSUJCAvPmzcu2HyNeeI3Kl1fgvnva+ZTPnTuX3377jZo1axIeHk5aWlqWf0cvu+wyOnfuzJQpUwIe95koW7YspUuXZvv27bk3zsGQIUMYOXKk+zrbVNOZ0kB7yy3Ns/fvBjz3QJ4xY4bPNFixYsUoViznj1NcXBx9+/Zl2bJlbsrsQFJWZ1a+fHmf6Ypff/01x9tgZhboNps3b86xY8ey3JnskksuoUOHDkyZMoW//e1vfr+QQ0NDOXHiRLZ9rFKlCt27d2fRokV+A2VeP0vr1q3z6X/fvn3ZsmULc+fOJSkpiWPHjrl148ePJyUlhWXLlhEbG5vt32Lm92TlypXs2LGDmJgYhg4dCgT2mcqOv3H4+z0E8pnPCwv6+ah58+YcP36cV1991S374osv+Oijj+jSpQurVq3igw8+ADz58R977DGeeOIJwJO6eOTIkT57TUeOHPG7nSFDhmSbwXL1uq+ZMe8dpo57KktdcnIyS5cudVMSr1u3zu8f6OOPP84rr7ySr/OI/gwaNIiePXu6/1UcOHDAvSdvoFq2bMlvv/3GV199BWSTanrWLI4cOULz5s39riO3NM+dO3fm008/9bl9ZXa/G38OHTrEhx9+6L5OSUmhRo0aQOApq701bdrU/Xc/PT2dN954I093A8vLNocMGeIzZfXpp5+6XzgnTpxg06ZN7li83XjjjbzxxhuA52993rx5bh+XLl3q3uR+z5497N+/3+c4AeScvrpnz57MmDHDvQ/y/v37efLJJ93PkrcOHTqQmJjIzJkzs9R5GzBgAI899hhHjx4F4IMPPmDVqlXuHn2G0NBQnn/+eWbNmsWvv/4a8GfKn/79+zN69Gg342taWhqjRo2iX79+Wdrm9JnPqwv6xugFTURYsGABffr0YcyYMZQsWZLw8HCef/55QkNDWbRoEb169aJnz56kp6fTtWtXHn30UcCzF/jCCy+QlJTEwYMH3btF/eMf/8iynZiYGBo0aJDlVowAw559mSNHj9Hsrh4+5W8v/g8//PAD11xzjVtWs2ZNypYty+eff+7TtkKFCtx+++253lM3L5o1a+bOmdevX59Zs2bxyCOPcOjQIa666iqKFy9O8eLF/f7B52bIkCG0a/fnfzWjR4+mf//+1K5dm2LFihEdHc2CBQvcf5uPHDnicyDx8ccfZ9q0afTq1YvIyEhUlcaNG7v/+mekUn788cfp06cPlSpVokyZMjz1VNYvVn9UlXHjxvHQQw8RGhpK6dKlmTFjBuA5WP/jjz/SpEkTRIQyZcpkSVndpk0bihcvDnjm2l977TUeeeQR4uPjUVVatWrF3/72t4Dfr0C2meGWW27B+6r47777jkceeQRV5fTp07Rp0ybLzYAAXnjhBR566CEmTZqEqpKUlOSelPDf//6X3r17U7JkScCz533FFVf4LJ/xnvfp04c+ffpQvHhx6tevzwsvvEClSpV44403ePDBBzl48CCqSp8+fXwO+nobOnQonTt35sEHH8z2PenVqxe//fYbcXFxhISEcMUVV7Bo0SJCQ0OztK1cuTKdOnVi8uTJOX6mrr766my3B56psrFjx3Lbbbdx8uRJihcvzrhx40hISMjSNqfPfF7lKbVyYbDUynlgqZWNCUrnMrWyMcaYIsyCvjHGBBEL+sYYE0Qs6BtjTBCxoG+MMUHEgr4xxgQRC/r5bO/evXTu3JlatWrRsGFDGjduzIIFC9z6VatW0ahRI6Kjo4mOjs5yMdLUqVPdukaNGrFq1Sq37tSpUwwePJioqCg3J4331aiXRF2bpT/Dn32Zqg1vdtsnJCRkyXWSlpZGbGwsAB9++CEiwjvvvOPW33rrre7FRSdPnmTgwIFERUURGxtLo0aNeO89uy2CMUXFBX1xVtzMrNn7zsaGbhtyrFdV2rdvT7du3XjzzTcB2LFjh3sl5549e+jcuTMLFy6kQYMG/PLLL9x8881UrVqVNm3asGTJEl555RVWrVpFhQoVWL9+Pe3bt2fNmjVcccUVPPXUU+zZs4cNGzZQsmRJDh48yLPPPptrv/s+2IX+/wz8QquwsDBGjhzp92KXp59+mt27d7Nx40ZKlCjB3r17+eijjwJetzGmcNmefj5asWIFF198sU9q5Ro1atCrVy+AXFPC5pTy9siRI7z66qu8+OKL7pWMZcqUyZJMKj/klLY4ow8Z6XQrVarE3Xffne99MMacGwEFfRG5TETeEpHNIvKtiDQWkXIiskxEtjmPf/FqP0hEUkVki4jc7FXeUEQ2OHWT5GxTA55nckvBm1sq1ZzqM1IGlylTJs/9mvjqbHdqJ9AcLXlJW2yMKToC3dN/AViqqtFAPPAtMBBYrqpRwHLnNSJSD+gIxACtgCkikpGo/CWgB56bpUc59Resnj17Eh8fz1VXXQUElhI2s+yWmT59OgkJCVSrVo2dO3fm2I++D3ZxUxRnzo+eHX9pi40xRV+uQV9EygI3ANMAVPWEqv4OtAMyUtfNBNo7z9sBc1T1uKp+D6QCjUSkMlBWVf/n3MprltcyF4TMKXgnT57M8uXL2bdvn1ufU0rYnFLeRkZG8sMPP7gpg++77z5SUlK49NJL3WyS+c1f2mLvPhhjip5A9vRrAfuA6SLypYi8JiKlgUqquhvAecy4NU1VwHvXc5dTVtV5nrn8gpGRf/yll15yy7xT8OaWEjanlLelSpXi/vvv59FHH3Vzg6enp/vkLM9vmdMWZ/Thsccec7e7e/duN4WuMeb8F8jZOxcBDYBeqvq5iLyAM5WTDX9zFZpDedYViPTAMw1E9erV/TU5L4kICxcupG/fvowbN46KFStSunRpxo4dC3hSsuaUEja3lLcjR47k6aefJjY2ljJlyhAaGkq3bt2oUqUKAEeOHiOs4Z8zZo/38KTbnfjqbN5Y/Oe0zsKFCwkPDw9oTJnTFo8YMYKnnnqKevXqUbJkSUqXLs0///lPAB544AEefvhhEhOzJPYzxpwnck2tLCJXAKtVNdx5fT2eoB8JNFXV3c7UzYeqWkdEBgGo6min/fvAcCANWOkcF0BEOjnLP0QOLLVyHlhqZWOCUr6mVlbVPcBOEanjFLUANgGLgW5OWTdgkfN8MdBRREqISE08B2zXOFNAB0XkGuesnSSvZYwxxhSAQC/O6gXMFpGLge3AfXi+MOaJyP3AD8BdAKr6jYjMw/PFcAroqaoZRxofAWYAocB7zo8xxpgCElDQV9UUwN9EbYts2o8ERvopXwvE5qF/xhhj8pFdkWuMMUHEgr4xxgQRC/rGGBNELOjns0suuQTwpCsWEV588UW37tFHH2XGjBn07NmThIQE6tWrR2hoqJsX56233uLee++lZs2ablmTJk0AmDFjBo8++miO2z558iQDR00i6tp2xDa/i0ZtuvLeik8B+OOPP0hKSiIiIoKIiAiSkpL4448/3L5m9KNevXokJSVx8uRJwJNq+dJLL+XKK6+kbt26/OMf/3C3l1Oa6OHDh1O1alV3ncnJyT59XbBgASLC5s2bAbj66qtJSEigevXqVKxY0R1/Wloa4eHh/PLLLwDs2rWLdu3aERUVRUREBL1793YvFMstLbQx5gJPrfxtdP6es19387d5an/55Zfzwgsv8NBDD3HxxRe75ZMnTwY8wfbWW291r9AFWLJkCePHj+fOO+/Mc/+eHv8Su/f+wsYV8ylR4mL27tvPR//zpHW4//77iY2NZdasWQAMGzaMBx54gPnz5wMQERFBSkoK6enp3HTTTcybN48uXboAnjw8S5Ys4fDhwyQkJHDrrbdStWrVHNNEA/Tt25f+/fuzbds2GjZsyJ133knx4sUBSE5O5rrrrmPOnDkMHz6czz//HPB8ua1du5Z//etfWcanqnTo0IFHHnmERYsWkZ6eTo8ePRgyZAjjx48Hck4LbYyxPf1zqmLFirRo0YKZM2fm3vgsHTlyhFdnL+DFEU9QooTnC6ZSxfLc3bYlqd//wLp163j66afd9kOHDmXt2rV89913PusJCQmhUaNG/Pjjj1m2Ubp0aRo2bMh3332Xa5pob1FRUZQqVYrffvsNgEOHDvHpp58ybdo05syZE/AYV6xYQcmSJbnvvvvcvk6cOJHXX3/dTXeRXVpoY4yHBf1zbODAgTz77LN5Soo2YMAAd3ojY287N6mpqVSvegVly1ySpW7Ttu9JSEggJCTELQsJCSEhIcFN65zh2LFjfP7557RqlTUB6v79+1m9ejUxMTG5pon2tn79eqKiorj8ck96poULF9KqVStq165NuXLlfJLU5cTfNsuWLUv16tVJTU11y/ylhTbGeFzQ0zvng5o1a9KoUSP3TlqBONPpnexkl57Zu/y7774jISGBbdu2ceedd1K/fn233SeffMKVV15JsWLFGDhwIDExMQGliZ44cSKvvvoq27dvZ+nSpW55cnIyffr0AaBjx44kJyfneB+CvIwDLC20MTmxPf0CMHjwYMaOHcvp06fP2TYiIyP54cc9HDx0OEtdTO1afPnllz7bP336NF999ZWbryNjTj81NZXVq1e7t3gETxD98ssvWbdunXtXsNzSRINnTn/Lli3MnTuXpKQkjh07xv79+1mxYgUPPPAA4eHhjB8/nrlz55JbDqjstnngwAF27txJRESET3nmtNDGGA8L+gUgOjqaevXqsWTJknO2jVKlSnF/p3Y89vQ4TpzwnHmze+8+3nj7XSJrVufKK6/0mfIYMWIEDRo0IDIy0mc9lStXZsyYMYwePTrH7eWWJtpbhw4dSExMZObMmbz11lskJSWxY8cO0tLS2LlzJzVr1vS5AXx2WrRowZEjR9yD0enp6fTr1497772XUqVK+bTNnBbaGONhQb+ADBkyhF27duXeEN85/YSEBPeUxBkzZhAWFub+ZF7fiCd6UrH8X6jX7A5im99F+/v7UbG85y6W06ZNY+vWrURGRhIREcHWrVuZNm2a3+23b9+eI0eO5Dg94p0mOjo6miZNmtC9e/dsz5oZOnQozz33HLNnz+b222/3qbvjjjsCmv4SERYsWMD8+fOJioqidu3alCxZklGjRvltn5f33JhgkWtq5cJmqZXzwFIrGxOU8jW1sjHGmAuHBX1jjAkiFvSNMSaIWNA3xpggYkHfGGOCSEBBX0TSRGSDiKSIyFqnrJyILBORbc7jX7zaDxKRVBHZIiI3e5U3dNaTKiKTxN/llcYYY86ZvOzpN1PVBK9TgAYCy1U1CljuvEZE6gEdgRigFTBFRDKSvrwE9MBzs/Qop/6CIiJ07drVfX3q1CkqVqzIrbfeCmRNkTxr1ixiY2OJiYmhXr16TJgwAcAnxXJ8fDzLly93lzlx4gR9+vQhIiKCqKgo2rVr53M++sgXXiOm2Z3U/+vdJNzUkc/XbwhouYx8PBk/06dPd59ffPHFxMXFkZCQwMCBA7OMe82aNdxwww3UqVOH6OhoHnjgATcJ2sKFC6lfvz7R0dHExcWxcOFCd7mcxtm0aVPq1KlDfHw81157LVu2bMnTOGJjY7ntttv4/fffffoaHx9Pp06dAHIcY+bf1dSpU9000o0aNfK5oKxp06YkJv55dtzatWtp2rRplvfJmMJ2Nrl32gFNneczgQ+BJ53yOap6HPheRFKBRiKSBpRV1f8BiMgsoD3n8Obokx9eka/r6/ly81zblC5dmo0bN3L06FFCQ0NZtmwZVatW9dv2vffe4/nnn+e///0vVapU4dixY/z73/926zNy8KxcuZIePXqwbds2wJPW4eDBg2zdupWQkBCmT59Ohw4d+HzBy6xe9zVLPviE9UvfpESJi/nl1984ceJUzst9/jkiQmhoqE+aZ8DNaBkeHs7KlSupUKFClnHs3buXu+66izlz5tC4cWNUlbfffpuDBw+ybds2+vfvz7Jly6hZsybff/89N910E7Vq1XLz+2Q3ToDZs2eTmJjI1KlTGTBgAIsXL87TOLp168bkyZMZMmQI4Dmf+fTp03z88cccPnyY++67L9sxzpgxw+3HkiVLeOWVV1i1ahUVKlRg/fr1tG/fnjVr1nDFFVcA8PPPP/Pee+/RunXrXP9OjCksge7pK/BfEVknIj2cskqquhvAebzcKa8K7PRadpdTVtV5nrn8gtO6dWveffddwJNcLGOvMrPRo0czYcIEqlSpAkDJkiV58MEHs7Rr3Lixm+r4yJEjTJ8+nYkTJ7pZM++77z5KlCjBilVr2P3zL1Qod5mbXrlCub9Q5YqKOS+34uy+HCdPnky3bt1o3Lgx4Plv584776RSpUpMmDCBwYMHU7NmTcCTgG7QoEFu/vvsxpnZDTfcQGpqap7HkXmdb775Jl27dqVly5Y++YVyM3bsWMaPH+9+ITRo0MD9QskwYMAAy+5pznuBBv1rVbUB0BroKSI35NDW3zy95lCedQUiPURkrYis3bdvX4BdPH907NiROXPmcOzYMb7++muuvvpqv+02btyYJVWwP0uXLqV9+/aAk0K5enXKli3r0yYxMZFvtm6n5Y2N2fnTXmpf156/Dxrt3kQlx+WcdMhHjx51pzoyp0rISU7jyEsKZu9xZvbOO+8QFxcX0DgypKens3z5ctq2beuWzZ07l3vuuYdOnTpluZtXTgIZR+PGjSlRogQrV64MeL3GFLSAgr6q/uQ8/gwsABoBe0WkMoDz+LPTfBdQzWvxMOAnpzzMT7m/7U1V1URVTaxYsWLgozlP1K9fn7S0NJKTk7nlllvOeD0DBgygVq1a/O1vf2Pw4MFA7umFLyldinVLZzN13FNULH8Z9zwykBlzFweUljhjWiQlJYUFCxaccb+zW392Zf7GmaFLly4kJCTw6aefMmHChIDGkfHlVb58eX799VduuukmAL744gsqVqxIjRo1aNGiBevXr3dv7JJfY7Nc/uZ8l2vQF5HSIlIm4znQEtgILAa6Oc26AYuc54uBjiJSQkRq4jlgu8aZAjooItc4Z+0keS1zwWnbti39+/fPdmoHPKmC161bl239+PHjSU1NZcSIEXTr5nmrIyMj2bFjBwcPHvRpu379eupFeaZQQkJCaNokkX/0f4R/jXiSt/+zPOflvNIhn4mcxuEvHXLmbfobZ4bZs2eTkpLCwoULqVatWkDjyPjy2rFjBydOnHCnYJKTk9m8eTPh4eFERERw4MAB3n777YDGWK9evSxj9PfeNW/enGPHjrF69eqA1mtMQQtkT78SsEpEvgLWAO+q6lJgDHCTiGwDbnJeo6rfAPOATcBSoKeqZtw26hHgNSAV+I5zeBC3sHXv3p2hQ4cSFxeXbZtBgwbxxBNPsGfPHgCOHz/OpEmTfNoUK1aM3r17c/r0ad5//31Kly5Nt27dePzxx927cc2aNYsjR47Q/LpGbElNY9v2H9zlU77ZQo2wyjkv1zz3A9Q5efTRR5k5c6Z7n1uAN954gz179tC/f39Gjx5NWloa4Lkv8KhRo+jXr1+O48xOXsZx6aWXMmnSJCZMmMDx48eZP38+X3/9NWlpaaSlpbFo0aKAp3ieeOIJnnzySfbv3w9ASkoKM2bM4O9//3uWtkOGDGHcuHEBrdeYgpbr2Tuquh2I91O+H2iRzTIjgSx3sFDVtUBs3rtZ9ISFhdG7d+8c29xyyy3s3buXv/71r+5UQffu3bO0ExGeeuopxo0bx80338zo0aPp378/tWvXplixYkRHR7NgwQJEfuHQkSP0emocvx84yEUXhRAZXo2p454CyGG5s7tcolKlSsyZM4f+/fvz888/U6xYMW644QY6dOjAFVdcwdixY7nttts4efIkxYsXZ9y4cSQkJOQ6zuzkZRxXXnkl8fHxzJs3j6pVq/qcSXXDDTewadMmdu/eTeXKlXMcY9u2bfnxxx9p0qQJIkKZMmV44403/C53yy23UBSnJU1wsNTKFxJLrWxMULLUysYYY/yyoG+MMUHEgr4xxgSRCy7on+/HKIwxJj/lNeZdUEG/ZMmS7N+/3wK/MSYoqCr79++nZMmSAS9zNgnXzjthYWHs2rWLopi6IV/8/nP2dX98W3D9MMYUmJIlSxIWFpZ7Q8cFFfSLFy/uJvYKSsOvyaHuj4LrhzHmvHVBTe8YY4zJmQV9Y4wJIhb0jTEmiFjQN8aYIGJB3xhjgogFfWOMCSIW9I0xJohY0DfGmCBiQd8YY4JIwEFfREJE5EsRWeK8Liciy0Rkm/P4F6+2g0QkVUS2iMjNXuUNRWSDUzdJzvaWTcYYY/IkL3v6vQHvBC4DgeWqGgUsd14jIvWAjkAM0AqYIiIhzjIvAT3w3Cw9yqk3xhhTQALKvSMiYUAbPPe9fdwpbgc0dZ7PBD4EnnTK56jqceB7EUkFGolIGlBWVf/nrHMW0J4L+OboxuRo+KU51FmuJHNuBLqn/zzwBHDaq6ySqu4GcB4vd8qrAju92u1yyqo6zzOXG2OMKSC5Bn0RuRX4WVXXBbhOf/P0mkO5v232EJG1IrI2aNMkG2PMORDInv61QFtnemYO0FxE3gD2ikhlAOcxI5n7LqCa1/JhwE9OeZif8ixUdaqqJqpqYsWKFfMwHGOMMTnJNeir6iBVDVPVcDwHaFeo6t+AxUA3p1k3YJHzfDHQUURKiEhNPAds1zhTQAdF5BrnrJ0kr2WMMcYUgLO5icoYYJ6I3A/8ANwFoKrfiMg8YBNwCuipqunOMo8AM4BQPAdw7SCuMcYUoDwFfVX9EM9ZOqjqfqBFNu1G4jnTJ3P5WiA2r500xhiTP+yKXGOMCSIW9I0xJohY0DfGmCBiQd8YY4LI2Zy9Y0zBsHQFxuQb29M3xpggYkHfGGOCiE3vmKA1+eEVOdb3fLl5AfXEmIJje/rGGBNELOgbY0wQsaBvjDFBxIK+McYEEQv6xhgTRCzoG2NMELGgb4wxQcSCvjHGBBEL+sYYE0RyDfoiUlJE1ojIVyLyjYj8wykvJyLLRGSb8/gXr2UGiUiqiGwRkZu9yhuKyAanbpJzr1xjjDEFJJA9/eNAc1WNBxKAViJyDTAQWK6qUcBy5zUiUg/PDdRjgFbAFBEJcdb1EtADz83So5x6Y4wxBSTXoK8eh5yXxZ0fBdoBM53ymUB753k7YI6qHlfV74FUoJGIVAbKqur/VFWBWV7LGGOMKQABzemLSIiIpAA/A8tU9XOgkqruBnAeL3eaVwV2ei2+yymr6jzPXG6MMaaABBT0VTVdVROAMDx77bE5NPc3T685lGddgUgPEVkrImv37dsXSBeNMcYEIE9n76jq78CHeObi9zpTNjiPPzvNdgHVvBYLA35yysP8lPvbzlRVTVTVxIoVK+ali8YYY3IQyNk7FUXkMud5KPBXYDOwGOjmNOsGLHKeLwY6ikgJEamJ54DtGmcK6KCIXOOctZPktYwxxpgCEMhNVCoDM50zcIoB81R1iYj8D5gnIvcDPwB3AajqNyIyD9gEnAJ6qmq6s65HgBlAKPCe82OMMaaA5Br0VfVr4Eo/5fuBFtksMxIY6ad8LZDT8QBjjDHnkF2Ra4wxQcSCvjHGBBEL+sYYE0Qs6BtjTBCxoG+MMUEkkFM2jTHGDL80h7o/Cq4fZ8n29I0xJohY0DfGmCBiQd8YY4KIBX1jjAkiFvSNMSaIWNA3xpggYkHfGGOCiAV9Y4wJIhb0jTEmiNgVucaYImHywytyrO/5cvMC6knRZnv6xhgTRAK5R241EVkpIt+KyDci0tspLyciy0Rkm/P4F69lBolIqohsEZGbvcobisgGp26Sc69cY4wxBSSQPf1TQD9VrQtcA/QUkXrAQGC5qkYBy53XOHUdgRigFTDFub8uwEtADzw3S49y6o0xxhSQQO6RuxvY7Tw/KCLfAlWBdkBTp9lM4EPgSad8jqoeB74XkVSgkYikAWVV9X8AIjILaI/dHN2cQ99G182+sunkguuIMeeJPB3IFZFwPDdJ/xyo5HwhoKq7ReRyp1lVYLXXYrucspPO88zl/rbTA89/BFSvXj0vXTQm6NkBT5OTgIO+iFwCvA30UdUDOUzH+6vQHMqzFqpOBaYCJCYm+m1j8ldOe8R1N39bgD0xxpxLAZ29IyLF8QT82ar6f07xXhGp7NRXBn52yncB1bwWDwN+csrD/JQbY4wpILnu6Ttn2EwDvlXV57yqFgPdgDHO4yKv8jdF5DmgCp4DtmtUNV1EDorINXimh5KAF/NtJMacZ8IHvptjfVrJAuqIMV4Cmd65FugKbBCRFKdsMJ5gP09E7gd+AO4CUNVvRGQesAnPmT89VTXdWe4RYAYQiucArh3ENcaYAhTI2Tur8D8fD9Aim2VGAiP9lK8FYvPSQWOMMfnH0jCYIi1uZlyO9fMKqB/GFBUW9I0pguz6A3OmLOgHCdsjNsaAJVwzxpigYnv6ptDZqY0mg01bnXsW9I0xeZLrl/SYNgXUE3MmbHrHGGOCiO3pG2PMOXY+5bayoG/MecjOtjLnik3vGGNMELGgb4wxQcSCvjHGBBEL+sYYE0Qs6BtjTBCxoG+MMUHEgr4xxgQRC/rGGBNEcg36IvK6iPwsIhu9ysqJyDIR2eY8/sWrbpCIpIrIFhG52au8oYhscOomOffeNcYYU4AC2dOfAbTKVDYQWK6qUcBy5zUiUg/oCMQ4y0wRkRBnmZeAHnhulB7lZ53GGGPOsUDukfuxiIRnKm4HNHWezwQ+BJ50yueo6nHgexFJBRqJSBpQVlX/ByAis4D22I3RjTFBbvLDK3Ks7/ly83zd3pnO6VdS1d0AzuPlTnlVYKdXu11OWVXneeZyv0Skh4isFZG1+/btO8MuGmOMySy/D+T6m6fXHMr9UtWpqpqoqokVK1bMt84ZY0ywO9Ogv1dEKgM4jz875buAal7twoCfnPIwP+XGGGMK0JkG/cVAN+d5N2CRV3lHESkhIjXxHLBd40wBHRSRa5yzdpK8ljHGGFNAcj2QKyLJeA7aVhCRXcAwYAwwT0TuB34A7gJQ1W9EZB6wCTgF9FTVdGdVj+A5EygUzwFcO4ibR3YvWWPM2Qrk7J1O2VS1yKb9SGCkn/K1QGyeemeMMSZf2RW5xhgTRCzoG2NMELGgb4wxQcSCvjHGBBEL+sYYE0RyPXvHmILODWKMOXcs6BtjCkzczLgc6+cVUD+CmQX9fGR7xMaY853N6RtjTBCxoG+MMUHEpneMMflr+KXZ19WsXnD9MH7Znr4xxgQRC/rGGBNEbHonk9xOKdvQbUMB9cQYY/LfBRn0c807P6ZNAfXEGGPOLza9Y4wxQcSCvjHGBJECD/oi0kpEtohIqogMLOjtG2NMMCvQOX0RCQEmAzcBu4AvRGSxqm4qyH6cjW+j62Zf2XRywXXEGGPOQEHv6TcCUlV1u6qeAOYA7Qq4D8YYE7REVQtuYyJ3Aq1U9QHndVfgalV9NFO7HkAP52UdYEuBdTKrCsAvhbj9c8XGVbTYuIqW82FcNVS1YubCgj5lU/yUZfnWUdWpwNRz353cichaVU0s7H7kNxtX0WLjKlrO53EV9PTOLqCa1+sw4KcC7oMxxgStgg76XwBRIlJTRC4GOgKLC7gPxhgTtAp0ekdVT4nIo8D7QAjwuqp+U5B9OAPnxTTTOWDjKlpsXEXLeTuuAj2Qa4wxpnDZFbnGGBNELOgbY0wQsaBvjDFBxIK+McYEEQv6uRCRFYXdh7MlIhUyvf6biEwSkR4i4u+CuSJBRG4XkXLO84oiMktENojIXBEJK+z+nSkReU5Eri3sfuQ3ESknIkNF5AHxGCIiS0RkvIj8pbD7dzZEpJmI/EtEFonI2yIyRkQiC7tf/tjZO15E5OvMRUBtnDQQqlq/wDuVD0Rkvao2cJ4/BVwPvAncCuxS1b6F2b8zJSKbVLWe83wusBqYD/wV6KKqNxVm/86UiOwDdgAVgblAsqp+Wbi9Onsi8h9gA1AWqOs8n4cnAWO8qhbJPFwiMgaoBCwH2gPfA1uBvwOjVHV+4fUuKwv6XkRkMXAAGAEcxRP0PwGuA1DVHYXXuzMnIl+q6pXO8/XA9ap6WESKA+tVNed7RJ6nRGSLqtZxnq9T1YZedSmqmlBonTsLGb8vEYnCcwFjRzzXtSTj+QLYWqgdPEMZvxPnv8tdqlo1c13h9e7MiciGjM+QiFwEfKSq1zr/vXyiqrGF20NfNr3jRVXbAm/jubAiXlXTgJOquqOoBnxHqIhcKSINgRBVPQygqieB9MLt2ln5UET+KSKhzvP24PlXG/ijUHt2dhRAVbep6jOqGgPcDZQE/lOoPTs7xZxAWA24RETCAUSkPHBxYXbsLJ3OmGYEquD5gkZVf8N/vrFCdUHeI/dsqOoCEfkv8IyIPEDR/mPMsBt4znn+q4hUVtXdzoftVCH262w9CgzhzyysfUXkMPAO0LXQenX2sgQKVf0a+BoYVPDdyTejgc3O8+7AayKiQD3gH4XWq7M3CvhSRLYA0cAj4DnOBHxVmB3zx6Z3ciAi8UBjVX25sPtyLjg3tSmhqkcKuy9nS0QuBS5S1f2F3ZezJSKXqOqhwu7HueD8zYmTkuUiIAH4UVV3F27Pzo6zp18Lz/1Cfi/k7uTIgn6ARCRaVTfn3rJosXEVLTauouV8HJcF/QCJyA+qWr2w+5HfbFxFi42raDkfx2Vz+l5EZFJ2VcBlBdiVfGXjKlpsXEVLURuX7el7EZGDQD/guJ/qZ1W1gp/y856Nq2ixcRUtRW1ctqfv6wtgo6p+lrlCRIYXfHfyjY2raLFxFS1Faly2p+/FOQJ/7EI4m8WbjatosXEVLUVtXBb0jTEmiNgVuV5E5FInUdJmEdnv/HzrlF1W2P07UzauosXGVbQUtXFZ0Pc1D/gNaKqq5VW1PNDMKTuvkiblkY2raLFxFS1Falw2vePFO4FXXurOdzauosXGVbQUtXHZnr6vHSLyhIhUyigQkUoi8iSwsxD7dbZsXEWLjatoKVLjsqDv6x6gPPCRiPwmIr8CHwLl8GQ5LKpsXEWLjatoKVLjsumdTEQkGggDVnsnvRKRVqq6tPB6dnZsXEWLjatoKUrjsj19LyLyGLAIT8rejSLifSefUYXTq7Nn4ypabFxFS1Ebl12R6+tBoKGqHhLPDR7eEpFwVX2B8/BmCHlg4ypabFxFS5EalwV9XyEZ/5qpapqINMXzC6zBefjLywMbV9Fi4ypaitS4bHrH1x4RSch44fwibwUqAEXyPrIOG1fRYuMqWorUuOxArhcRCQNOqeoeP3XXquqnhdCts2bjKlpsXEVLURuXBX1jjAkiNr1jjDFBxIK+McYEEQv6xmQiHqtEpLVX2d0icl5dZGPMmbA5fWP8EJFYPBkSrwRCgBSglap+dwbrClHV9PztoTFnxoK+MdkQkXHAYaC081gDzyl4FwHDVXWRczHOv502AI+q6mfOudrDgN1AgqrWK9jeG+OfBX1jsiEipYH1wAlgCfCNqr4hnhtjrMHzX4ACp1X1mIhEAcmqmugE/XeBWFX9vjD6b4w/dkWuMdlQ1cMiMhc4hCdb4m0i0t+pLglUB34C/uVcnJMO1PZaxRoL+OZ8Y0HfmJyddn4EuENVt3hXishwYC8Qj+fEiGNe1YcLqI/GBMzO3jEmMO8DvUREAETkSqf8UmC3qp4GuuI56GvMecuCvjGBeQYoDnwtIhud1wBTgG4ishrP1I7t3Zvzmh3INcaYIGJ7+sYYE0Qs6BtjTBCxoG+MMUHEgr4xxgQRC/rGGBNELOgbY0wQsaBvjDFBxIK+McYEkf8HGPe0h499f9wAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "emp_year=df[df['EMPLOYER_NAME'].isin(df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:5].index)]\n",
    "emp_year=emp_year.groupby(['EMPLOYER_NAME','Year'])['CASE_STATUS'].count().reset_index()\n",
    "emp_year.pivot('Year','EMPLOYER_NAME','CASE_STATUS').plot.bar(width=0.7)\n",
    "emp_year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "b2e35042",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "alignmentgroup": "True",
         "bingroup": "x",
         "histfunc": "sum",
         "hovertemplate": "EMPLOYER_NAME=AMAZON CORPORATE LLC<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>",
         "legendgroup": "AMAZON CORPORATE LLC",
         "marker": {
          "color": "#636efa",
          "pattern": {
           "shape": ""
          }
         },
         "name": "AMAZON CORPORATE LLC",
         "offsetgroup": "AMAZON CORPORATE LLC",
         "orientation": "v",
         "showlegend": true,
         "type": "histogram",
         "x": [
          2014,
          2015,
          2016,
          2017,
          2018,
          2019
         ],
         "xaxis": "x",
         "y": [
          239,
          845,
          1650,
          1915,
          1586,
          1
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "bingroup": "x",
         "histfunc": "sum",
         "hovertemplate": "EMPLOYER_NAME=COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>",
         "legendgroup": "COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION",
         "marker": {
          "color": "#EF553B",
          "pattern": {
           "shape": ""
          }
         },
         "name": "COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION",
         "offsetgroup": "COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION",
         "orientation": "v",
         "showlegend": true,
         "type": "histogram",
         "x": [
          2014,
          2015,
          2016,
          2017,
          2018,
          2019
         ],
         "xaxis": "x",
         "y": [
          20,
          7992,
          3622,
          4072,
          399,
          2580
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "bingroup": "x",
         "histfunc": "sum",
         "hovertemplate": "EMPLOYER_NAME=GOOGLE INC.<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>",
         "legendgroup": "GOOGLE INC.",
         "marker": {
          "color": "#00cc96",
          "pattern": {
           "shape": ""
          }
         },
         "name": "GOOGLE INC.",
         "offsetgroup": "GOOGLE INC.",
         "orientation": "v",
         "showlegend": true,
         "type": "histogram",
         "x": [
          2014,
          2015,
          2016,
          2017,
          2018
         ],
         "xaxis": "x",
         "y": [
          273,
          1722,
          1793,
          1477,
          536
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "bingroup": "x",
         "histfunc": "sum",
         "hovertemplate": "EMPLOYER_NAME=INTEL CORPORATION<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>",
         "legendgroup": "INTEL CORPORATION",
         "marker": {
          "color": "#ab63fa",
          "pattern": {
           "shape": ""
          }
         },
         "name": "INTEL CORPORATION",
         "offsetgroup": "INTEL CORPORATION",
         "orientation": "v",
         "showlegend": true,
         "type": "histogram",
         "x": [
          2014,
          2015,
          2016,
          2017,
          2018,
          2019
         ],
         "xaxis": "x",
         "y": [
          79,
          1984,
          2075,
          1756,
          1618,
          1520
         ],
         "yaxis": "y"
        },
        {
         "alignmentgroup": "True",
         "bingroup": "x",
         "histfunc": "sum",
         "hovertemplate": "EMPLOYER_NAME=MICROSOFT CORPORATION<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>",
         "legendgroup": "MICROSOFT CORPORATION",
         "marker": {
          "color": "#FFA15A",
          "pattern": {
           "shape": ""
          }
         },
         "name": "MICROSOFT CORPORATION",
         "offsetgroup": "MICROSOFT CORPORATION",
         "orientation": "v",
         "showlegend": true,
         "type": "histogram",
         "x": [
          2014,
          2015,
          2016,
          2017,
          2018,
          2019
         ],
         "xaxis": "x",
         "y": [
          528,
          602,
          3513,
          2033,
          2228,
          1045
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "barmode": "group",
        "height": 500,
        "legend": {
         "title": {
          "text": "EMPLOYER_NAME"
         },
         "tracegroupgap": 0
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Top Employers(sponsors) over the years"
        },
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "Year"
         }
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0,
          1
         ],
         "title": {
          "text": "sum of CASE_STATUS"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"57d2e8ff-f7c4-4e59-a7d2-54e79ac807ee\" class=\"plotly-graph-div\" style=\"height:500px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"57d2e8ff-f7c4-4e59-a7d2-54e79ac807ee\")) {                    Plotly.newPlot(                        \"57d2e8ff-f7c4-4e59-a7d2-54e79ac807ee\",                        [{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"histfunc\":\"sum\",\"hovertemplate\":\"EMPLOYER_NAME=AMAZON CORPORATE LLC<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>\",\"legendgroup\":\"AMAZON CORPORATE LLC\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"AMAZON CORPORATE LLC\",\"offsetgroup\":\"AMAZON CORPORATE LLC\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[2014,2015,2016,2017,2018,2019],\"xaxis\":\"x\",\"y\":[239,845,1650,1915,1586,1],\"yaxis\":\"y\",\"type\":\"histogram\"},{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"histfunc\":\"sum\",\"hovertemplate\":\"EMPLOYER_NAME=COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>\",\"legendgroup\":\"COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\"COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION\",\"offsetgroup\":\"COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[2014,2015,2016,2017,2018,2019],\"xaxis\":\"x\",\"y\":[20,7992,3622,4072,399,2580],\"yaxis\":\"y\",\"type\":\"histogram\"},{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"histfunc\":\"sum\",\"hovertemplate\":\"EMPLOYER_NAME=GOOGLE INC.<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>\",\"legendgroup\":\"GOOGLE INC.\",\"marker\":{\"color\":\"#00cc96\",\"pattern\":{\"shape\":\"\"}},\"name\":\"GOOGLE INC.\",\"offsetgroup\":\"GOOGLE INC.\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[2014,2015,2016,2017,2018],\"xaxis\":\"x\",\"y\":[273,1722,1793,1477,536],\"yaxis\":\"y\",\"type\":\"histogram\"},{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"histfunc\":\"sum\",\"hovertemplate\":\"EMPLOYER_NAME=INTEL CORPORATION<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>\",\"legendgroup\":\"INTEL CORPORATION\",\"marker\":{\"color\":\"#ab63fa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"INTEL CORPORATION\",\"offsetgroup\":\"INTEL CORPORATION\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[2014,2015,2016,2017,2018,2019],\"xaxis\":\"x\",\"y\":[79,1984,2075,1756,1618,1520],\"yaxis\":\"y\",\"type\":\"histogram\"},{\"alignmentgroup\":\"True\",\"bingroup\":\"x\",\"histfunc\":\"sum\",\"hovertemplate\":\"EMPLOYER_NAME=MICROSOFT CORPORATION<br>Year=%{x}<br>sum of CASE_STATUS=%{y}<extra></extra>\",\"legendgroup\":\"MICROSOFT CORPORATION\",\"marker\":{\"color\":\"#FFA15A\",\"pattern\":{\"shape\":\"\"}},\"name\":\"MICROSOFT CORPORATION\",\"offsetgroup\":\"MICROSOFT CORPORATION\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[2014,2015,2016,2017,2018,2019],\"xaxis\":\"x\",\"y\":[528,602,3513,2033,2228,1045],\"yaxis\":\"y\",\"type\":\"histogram\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Year\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"sum of CASE_STATUS\"}},\"legend\":{\"title\":{\"text\":\"EMPLOYER_NAME\"},\"tracegroupgap\":0},\"title\":{\"text\":\"Top Employers(sponsors) over the years\"},\"barmode\":\"group\",\"height\":500},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('57d2e8ff-f7c4-4e59-a7d2-54e79ac807ee');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = px.histogram(emp_year, x=\"Year\", y=\"CASE_STATUS\",\n",
    "             color='EMPLOYER_NAME', barmode='group',\n",
    "             #histfunc='avg',\n",
    "             height=500,title=\"Top Employers(sponsors) over the years\")\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2efe7789",
   "metadata": {},
   "source": [
    "### 5. Acceptance ratio and Denial ratio comparison by employer and education"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "3f0a25b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0            Master's\n",
       "1            Master's\n",
       "2            Master's\n",
       "3            Master's\n",
       "4          Bachelor's\n",
       "             ...     \n",
       "535471    High School\n",
       "535472     Bachelor's\n",
       "535473     Bachelor's\n",
       "535474           None\n",
       "535475     Bachelor's\n",
       "Name: JOB_INFO_EDUCATION, Length: 535476, dtype: object"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns\n",
    "\n",
    "df['JOB_INFO_EDUCATION']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "5bbea6f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0           Between 60000 -100000\n",
       "1           Between 60000 -100000\n",
       "2           Between 60000 -100000\n",
       "3           Between 60000 -100000\n",
       "4           Between 60000 -100000\n",
       "                   ...           \n",
       "535057       Between 30000 -60000\n",
       "535058    Between 100000 -1500000\n",
       "535059    Between 100000 -1500000\n",
       "535063    Between 100000 -1500000\n",
       "535083      Between 60000 -100000\n",
       "Name: WAGE_OFFERED, Length: 534765, dtype: object"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfff['WAGE_OFFERED']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "d27c7366",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "dff_education = dfff[['REQD_EDUCATION','CASE_STATUS']]\n",
    "dff_education_total = dff_education.groupby([\"REQD_EDUCATION\"],as_index =False)['CASE_STATUS'].count()\n",
    "#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)\n",
    "dff_education_total = dff_education_total.rename(columns={\"CASE_STATUS\":'Total applications'})\n",
    "#dff_education_total\n",
    "\n",
    "\n",
    "\n",
    "dff_education_certified = dff_education[dff_education['CASE_STATUS'] =='Certified']\n",
    "dff_education_certified= dff_education_certified.groupby([\"REQD_EDUCATION\"],as_index =False)['CASE_STATUS'].count()\n",
    "#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)\n",
    "dff_education_certified = dff_education_certified.rename(columns={\"CASE_STATUS\":'Certified applications'})\n",
    "dff_education_certified\n",
    "\n",
    "dff_education_total['Approved_Case'] = list(dff_education_certified['Certified applications'])\n",
    "dff_education_total['Denied_Case']=dff_education_total['Total applications'] - dff_education_total['Approved_Case']\n",
    "dff_education_total['Approved_Ratio'] = round(dff_education_total['Approved_Case']/dff_education_total['Total applications'] *100 ,2)\n",
    "dff_education_total['Denied_Ratio'] = round(dff_education_total['Denied_Case']/dff_education_total['Total applications'] *100 ,2)\n",
    "\n",
    "#dff_education_total\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "6c7682c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": [
           "blueviolet",
           "blue",
           "lightskyblue",
           "lightsteelblue",
           "mediumblue",
           "aqua",
           "midnightblue"
          ]
         },
         "type": "bar",
         "x": [
          "Doctorate",
          "Master's",
          "Bachelor's",
          "Other",
          "Associate's",
          "None",
          "High School"
         ],
         "y": [
          94.22,
          93.87,
          93.25,
          91.5,
          77.2,
          73.28,
          70.82
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Certified cases ratio by education level"
        },
        "xaxis": {
         "title": {
          "text": "Education level"
         }
        },
        "yaxis": {
         "title": {
          "text": "Approved ratio"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"2fa45202-1b49-433b-8dec-cfc906dd7b4a\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"2fa45202-1b49-433b-8dec-cfc906dd7b4a\")) {                    Plotly.newPlot(                        \"2fa45202-1b49-433b-8dec-cfc906dd7b4a\",                        [{\"marker\":{\"color\":[\"blueviolet\",\"blue\",\"lightskyblue\",\"lightsteelblue\",\"mediumblue\",\"aqua\",\"midnightblue\"]},\"x\":[\"Doctorate\",\"Master's\",\"Bachelor's\",\"Other\",\"Associate's\",\"None\",\"High School\"],\"y\":[94.22,93.87,93.25,91.5,77.2,73.28,70.82],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Certified cases ratio by education level\"},\"xaxis\":{\"title\":{\"text\":\"Education level\"}},\"yaxis\":{\"title\":{\"text\":\"Approved ratio\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('2fa45202-1b49-433b-8dec-cfc906dd7b4a');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "#Approved Ratio\n",
    "\n",
    "colors = ['blueviolet','blue','lightskyblue','lightsteelblue','mediumblue','aqua','midnightblue']\n",
    "dff_education_approved = dff_education_total.sort_values(by ='Approved_Ratio', ascending =False)\n",
    "\n",
    "\n",
    "fig  = go.Figure([go.Bar(x = dff_education_approved['REQD_EDUCATION'],y=dff_education_approved['Approved_Ratio'], marker_color =colors)])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Certified cases ratio by education level',\n",
    "                 xaxis_title = 'Education level',\n",
    "                 yaxis_title = 'Approved ratio')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "225320b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": [
           "blueviolet",
           "blue",
           "lightskyblue",
           "lightsteelblue",
           "mediumblue",
           "aqua",
           "midnightblue"
          ]
         },
         "type": "bar",
         "x": [
          "High School",
          "None",
          "Associate's",
          "Other",
          "Bachelor's",
          "Master's",
          "Doctorate"
         ],
         "y": [
          29.18,
          26.72,
          22.8,
          8.5,
          6.75,
          6.13,
          5.78
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Denied cases ratio by education level"
        },
        "xaxis": {
         "title": {
          "text": "Education level"
         }
        },
        "yaxis": {
         "title": {
          "text": "Denied ratio"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"f0951c43-1a64-43c9-9929-378c2f403e72\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"f0951c43-1a64-43c9-9929-378c2f403e72\")) {                    Plotly.newPlot(                        \"f0951c43-1a64-43c9-9929-378c2f403e72\",                        [{\"marker\":{\"color\":[\"blueviolet\",\"blue\",\"lightskyblue\",\"lightsteelblue\",\"mediumblue\",\"aqua\",\"midnightblue\"]},\"x\":[\"High School\",\"None\",\"Associate's\",\"Other\",\"Bachelor's\",\"Master's\",\"Doctorate\"],\"y\":[29.18,26.72,22.8,8.5,6.75,6.13,5.78],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Denied cases ratio by education level\"},\"xaxis\":{\"title\":{\"text\":\"Education level\"}},\"yaxis\":{\"title\":{\"text\":\"Denied ratio\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('f0951c43-1a64-43c9-9929-378c2f403e72');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "#Denied Ratio\n",
    "\n",
    "colors = ['blueviolet','blue','lightskyblue','lightsteelblue','mediumblue','aqua','midnightblue']\n",
    "dff_education_denied = dff_education_total.sort_values(by ='Denied_Ratio', ascending =False)\n",
    "\n",
    "\n",
    "fig  = go.Figure([go.Bar(x = dff_education_denied['REQD_EDUCATION'],y=dff_education_denied['Denied_Ratio'], marker_color = colors)])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Denied cases ratio by education level',\n",
    "                 xaxis_title = 'Education level',\n",
    "                 yaxis_title = 'Denied ratio')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e061c75e",
   "metadata": {},
   "source": [
    "### 6.  Wages Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "300d47c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>WAGE_OFFERED</th>\n",
       "      <th>Total applications</th>\n",
       "      <th>Approved_Case</th>\n",
       "      <th>Denied_Case</th>\n",
       "      <th>Approved_Ratio</th>\n",
       "      <th>Denied_Ratio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Above 200000</td>\n",
       "      <td>2874</td>\n",
       "      <td>2641</td>\n",
       "      <td>233</td>\n",
       "      <td>91.89</td>\n",
       "      <td>8.11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Below 30000</td>\n",
       "      <td>43888</td>\n",
       "      <td>31216</td>\n",
       "      <td>12672</td>\n",
       "      <td>71.13</td>\n",
       "      <td>28.87</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Between 100000 -1500000</td>\n",
       "      <td>166199</td>\n",
       "      <td>157436</td>\n",
       "      <td>8763</td>\n",
       "      <td>94.73</td>\n",
       "      <td>5.27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Between 150000 -2000000</td>\n",
       "      <td>17184</td>\n",
       "      <td>16062</td>\n",
       "      <td>1122</td>\n",
       "      <td>93.47</td>\n",
       "      <td>6.53</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Between 30000 -60000</td>\n",
       "      <td>69869</td>\n",
       "      <td>59531</td>\n",
       "      <td>10338</td>\n",
       "      <td>85.20</td>\n",
       "      <td>14.80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Between 60000 -100000</td>\n",
       "      <td>234751</td>\n",
       "      <td>218561</td>\n",
       "      <td>16190</td>\n",
       "      <td>93.10</td>\n",
       "      <td>6.90</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              WAGE_OFFERED  Total applications  Approved_Case  Denied_Case  \\\n",
       "0             Above 200000                2874           2641          233   \n",
       "1              Below 30000               43888          31216        12672   \n",
       "2  Between 100000 -1500000              166199         157436         8763   \n",
       "3  Between 150000 -2000000               17184          16062         1122   \n",
       "4     Between 30000 -60000               69869          59531        10338   \n",
       "5    Between 60000 -100000              234751         218561        16190   \n",
       "\n",
       "   Approved_Ratio  Denied_Ratio  \n",
       "0           91.89          8.11  \n",
       "1           71.13         28.87  \n",
       "2           94.73          5.27  \n",
       "3           93.47          6.53  \n",
       "4           85.20         14.80  \n",
       "5           93.10          6.90  "
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dff_wages = dfff[['WAGE_OFFERED','CASE_STATUS']]\n",
    "dff_wages_total = dff_wages.groupby([\"WAGE_OFFERED\"],as_index =False)['CASE_STATUS'].count()\n",
    "#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)\n",
    "dff_wages_total = dff_wages_total.rename(columns={\"CASE_STATUS\":'Total applications'})\n",
    "dff_wages_total\n",
    "\n",
    "\n",
    "\n",
    "dff_wages_certified = dff_wages[dff_wages['CASE_STATUS'] =='Certified']\n",
    "dff_wages_certified= dff_wages_certified.groupby([\"WAGE_OFFERED\"],as_index =False)['CASE_STATUS'].count()\n",
    "#dff_education_total = dff_education_total.sort_values(by='CASE_STATUS',ascending = False)\n",
    "dff_wages_certified = dff_wages_certified.rename(columns={\"CASE_STATUS\":'Certified applications'})\n",
    "dff_wages_certified\n",
    "\n",
    "dff_wages_total['Approved_Case'] = list(dff_wages_certified['Certified applications'])\n",
    "dff_wages_total['Denied_Case']=dff_wages_total['Total applications'] - dff_wages_total['Approved_Case']\n",
    "dff_wages_total['Approved_Ratio'] = round(dff_wages_total['Approved_Case']/dff_wages_total['Total applications'] *100 ,2)\n",
    "dff_wages_total['Denied_Ratio'] = round(dff_wages_total['Denied_Case']/dff_wages_total['Total applications'] *100 ,2)\n",
    "\n",
    "dff_wages_total\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "27512bb7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": [
           "blueviolet",
           "blue",
           "lightskyblue",
           "lightsteelblue",
           "mediumblue",
           "aqua",
           "midnightblue"
          ]
         },
         "type": "bar",
         "x": [
          "Between 100000 -1500000",
          "Between 150000 -2000000",
          "Between 60000 -100000",
          "Above 200000",
          "Between 30000 -60000",
          "Below 30000"
         ],
         "y": [
          94.73,
          93.47,
          93.1,
          91.89,
          85.2,
          71.13
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Approved cases ratio by wage category"
        },
        "xaxis": {
         "title": {
          "text": "Wages cateogry"
         }
        },
        "yaxis": {
         "title": {
          "text": "Approved ratio"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"d3764779-41c4-46b4-889b-8e0a0f54f013\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"d3764779-41c4-46b4-889b-8e0a0f54f013\")) {                    Plotly.newPlot(                        \"d3764779-41c4-46b4-889b-8e0a0f54f013\",                        [{\"marker\":{\"color\":[\"blueviolet\",\"blue\",\"lightskyblue\",\"lightsteelblue\",\"mediumblue\",\"aqua\",\"midnightblue\"]},\"x\":[\"Between 100000 -1500000\",\"Between 150000 -2000000\",\"Between 60000 -100000\",\"Above 200000\",\"Between 30000 -60000\",\"Below 30000\"],\"y\":[94.73,93.47,93.1,91.89,85.2,71.13],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Approved cases ratio by wage category\"},\"xaxis\":{\"title\":{\"text\":\"Wages cateogry\"}},\"yaxis\":{\"title\":{\"text\":\"Approved ratio\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('d3764779-41c4-46b4-889b-8e0a0f54f013');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Approved Ratio\n",
    "\n",
    "colors = ['blueviolet','blue','lightskyblue','lightsteelblue','mediumblue','aqua','midnightblue']\n",
    "dff_wages_approved = dff_wages_total.sort_values(by ='Approved_Ratio', ascending =False)\n",
    "dff_wages_approved\n",
    "\n",
    "fig  = go.Figure([go.Bar(x = dff_wages_approved['WAGE_OFFERED'],y=dff_wages_approved['Approved_Ratio'], marker_color = colors)])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Approved cases ratio by wage category',\n",
    "                 xaxis_title = 'Wages cateogry',\n",
    "                 yaxis_title = 'Approved ratio')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "98f9802a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": [
           "blueviolet",
           "blue",
           "lightskyblue",
           "lightsteelblue",
           "mediumblue",
           "aqua",
           "midnightblue"
          ]
         },
         "type": "bar",
         "x": [
          "Below 30000",
          "Between 30000 -60000",
          "Above 200000",
          "Between 60000 -100000",
          "Between 150000 -2000000",
          "Between 100000 -1500000"
         ],
         "y": [
          28.87,
          14.8,
          8.11,
          6.9,
          6.53,
          5.27
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Denied cases ratio by wage category"
        },
        "xaxis": {
         "title": {
          "text": "Wages cateogry"
         }
        },
        "yaxis": {
         "title": {
          "text": "Denied ratio"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"6e76a0f1-0e51-44d6-aae0-730a46b559ae\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"6e76a0f1-0e51-44d6-aae0-730a46b559ae\")) {                    Plotly.newPlot(                        \"6e76a0f1-0e51-44d6-aae0-730a46b559ae\",                        [{\"marker\":{\"color\":[\"blueviolet\",\"blue\",\"lightskyblue\",\"lightsteelblue\",\"mediumblue\",\"aqua\",\"midnightblue\"]},\"x\":[\"Below 30000\",\"Between 30000 -60000\",\"Above 200000\",\"Between 60000 -100000\",\"Between 150000 -2000000\",\"Between 100000 -1500000\"],\"y\":[28.87,14.8,8.11,6.9,6.53,5.27],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Denied cases ratio by wage category\"},\"xaxis\":{\"title\":{\"text\":\"Wages cateogry\"}},\"yaxis\":{\"title\":{\"text\":\"Denied ratio\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('6e76a0f1-0e51-44d6-aae0-730a46b559ae');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#DENIED Ratio\n",
    "\n",
    "colors = ['blueviolet','blue','lightskyblue','lightsteelblue','mediumblue','aqua','midnightblue']\n",
    "dff_wages_denied = dff_wages_total.sort_values(by ='Denied_Ratio', ascending =False)\n",
    "dff_wages_denied\n",
    "\n",
    "fig  = go.Figure([go.Bar(x = dff_wages_denied['WAGE_OFFERED'],y=dff_wages_denied['Denied_Ratio'], marker_color = colors)])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Denied cases ratio by wage category',\n",
    "                 xaxis_title = 'Wages cateogry',\n",
    "                 yaxis_title = 'Denied ratio')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c965a3c1",
   "metadata": {},
   "source": [
    "### 7. Decesion data analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "de5f6521",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['CASE_STATUS', 'EMPLOYER_NAME', 'EMPLOYER_CITY', 'EMPLOYER_STATE',\n",
       "       'SOC_TITLE', 'WAGES_OFFERED', 'JOB_CITY', 'JOB_STATE', 'JOB_TITLE',\n",
       "       'REQD_EDUCATION', 'RELTD_MAJOR', 'TRAINING_REQD', 'LANG_REQD',\n",
       "       'CITIZENSHIP', 'ADMISSION_TYPE', 'WORKER_EDUCATION', 'WORKER_MAJOR',\n",
       "       'time_taken', 'OCCUPATION', 'NEW_RELATED_MAJOR', 'NEW_WORKER_MAJOR',\n",
       "       'NEW_EMPLOYER_NAME', 'WAGE_OFFERED'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dfff.columns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "91e1150d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>Time Taken</th>\n",
       "      <th>Total decided cases</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>6</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>212316</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>180389</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>9</td>\n",
       "      <td>at least 9 months</td>\n",
       "      <td>94755</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>at least 24 months</td>\n",
       "      <td>20665</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>at least 15 months</td>\n",
       "      <td>10425</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>at least 12 months</td>\n",
       "      <td>8195</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>5</td>\n",
       "      <td>at least 36 months</td>\n",
       "      <td>2715</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>7</td>\n",
       "      <td>at least 60 months</td>\n",
       "      <td>2705</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8</td>\n",
       "      <td>at least 84 months</td>\n",
       "      <td>2395</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1</td>\n",
       "      <td>at least 120 months</td>\n",
       "      <td>192</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>10</td>\n",
       "      <td>more than 120 months</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    index            Time Taken  Total decided cases\n",
       "0       6     at least 6 months               212316\n",
       "1       4     at least 3 months               180389\n",
       "2       9     at least 9 months                94755\n",
       "3       3    at least 24 months                20665\n",
       "4       2    at least 15 months                10425\n",
       "5       0    at least 12 months                 8195\n",
       "6       5    at least 36 months                 2715\n",
       "7       7    at least 60 months                 2705\n",
       "8       8    at least 84 months                 2395\n",
       "9       1   at least 120 months                  192\n",
       "10     10  more than 120 months                   13"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "decesion_df = dfff[['time_taken','CASE_STATUS']]\n",
    "decesion_df_total = decesion_df.groupby([\"time_taken\"],as_index =False)['CASE_STATUS'].count()\n",
    "decesion_df_total = decesion_df_total.rename(columns={\"CASE_STATUS\":'Total decided cases', 'time_taken':'Time Taken'})\n",
    "decesion_df_total = decesion_df_total.sort_values(by='Total decided cases', ascending= False).reset_index()\n",
    "decesion_df_total\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "b497218f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "212316\n",
      "180389\n",
      "94755\n",
      "20665\n",
      "10425\n",
      "8195\n",
      "2715\n",
      "2705\n",
      "2395\n",
      "192\n",
      "13\n"
     ]
    },
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": "darkblue",
          "size": 10
         },
         "mode": "markers",
         "type": "scatter",
         "x": [
          212316,
          180389,
          94755,
          20665,
          10425,
          8195,
          2715,
          2705,
          2395,
          192,
          13
         ],
         "y": [
          "at least 6 months",
          "at least 3 months",
          "at least 9 months",
          "at least 24 months",
          "at least 15 months",
          "at least 12 months",
          "at least 36 months",
          "at least 60 months",
          "at least 84 months",
          "at least 120 months",
          "more than 120 months"
         ]
        }
       ],
       "layout": {
        "shapes": [
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 212316,
          "y0": 0,
          "y1": 0
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 180389,
          "y0": 1,
          "y1": 1
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 94755,
          "y0": 2,
          "y1": 2
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 20665,
          "y0": 3,
          "y1": 3
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 10425,
          "y0": 4,
          "y1": 4
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 8195,
          "y0": 5,
          "y1": 5
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 2715,
          "y0": 6,
          "y1": 6
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 2705,
          "y0": 7,
          "y1": 7
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 2395,
          "y0": 8,
          "y1": 8
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 192,
          "y0": 9,
          "y1": 9
         },
         {
          "line": {
           "color": "crimson",
           "width": 3
          },
          "type": "line",
          "x0": 0,
          "x1": 13,
          "y0": 10,
          "y1": 10
         }
        ],
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "font": {
          "size": 20
         },
         "text": "Analysis of time taken for cases to decided"
        },
        "xaxis": {
         "range": [
          0,
          220000
         ],
         "title": {
          "text": "Number of decided applications"
         }
        },
        "yaxis": {
         "title": {
          "text": "Time taken"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"22a5a263-0ebe-4caa-b2ef-ebc5aa8ed925\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"22a5a263-0ebe-4caa-b2ef-ebc5aa8ed925\")) {                    Plotly.newPlot(                        \"22a5a263-0ebe-4caa-b2ef-ebc5aa8ed925\",                        [{\"marker\":{\"color\":\"darkblue\",\"size\":10},\"mode\":\"markers\",\"x\":[212316,180389,94755,20665,10425,8195,2715,2705,2395,192,13],\"y\":[\"at least 6 months\",\"at least 3 months\",\"at least 9 months\",\"at least 24 months\",\"at least 15 months\",\"at least 12 months\",\"at least 36 months\",\"at least 60 months\",\"at least 84 months\",\"at least 120 months\",\"more than 120 months\"],\"type\":\"scatter\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"shapes\":[{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":212316,\"y0\":0,\"y1\":0},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":180389,\"y0\":1,\"y1\":1},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":94755,\"y0\":2,\"y1\":2},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":20665,\"y0\":3,\"y1\":3},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":10425,\"y0\":4,\"y1\":4},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":8195,\"y0\":5,\"y1\":5},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":2715,\"y0\":6,\"y1\":6},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":2705,\"y0\":7,\"y1\":7},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":2395,\"y0\":8,\"y1\":8},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":192,\"y0\":9,\"y1\":9},{\"line\":{\"color\":\"crimson\",\"width\":3},\"type\":\"line\",\"x0\":0,\"x1\":13,\"y0\":10,\"y1\":10}],\"title\":{\"font\":{\"size\":20},\"text\":\"Analysis of time taken for cases to decided\"},\"xaxis\":{\"title\":{\"text\":\"Number of decided applications\"},\"range\":[0,220000]},\"yaxis\":{\"title\":{\"text\":\"Time taken\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('22a5a263-0ebe-4caa-b2ef-ebc5aa8ed925');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig1 = go.Figure()\n",
    "# Draw points\n",
    "fig1.add_trace(go.Scatter(x = decesion_df_total[\"Total decided cases\"],\n",
    "                          y = decesion_df_total[\"Time Taken\"],\n",
    "                          mode = 'markers',\n",
    "                          marker_color ='darkblue',\n",
    "                          marker_size  = 10))\n",
    "# Draw lines\n",
    "for i in range(0, len(decesion_df_total)):\n",
    "    print(decesion_df_total[\"Total decided cases\"][i])\n",
    "    fig1.add_shape(type='line',x0 = 0, y0 = i,\n",
    "                               x1 = decesion_df_total[\"Total decided cases\"][i],\n",
    "                              y1 = i,\n",
    "                              line=dict(color='crimson', width = 3))\n",
    "# Set title\n",
    "fig1.update_layout(title_text = \n",
    "                   \"Analysis of time taken for cases to decided\",\n",
    "                   title_font_size = 20)\n",
    "# Set x-axes range\n",
    "fig1.update_xaxes(title = 'Number of decided applications' , range=[0, 220000])\n",
    "fig1.update_yaxes(title = 'Time taken')\n",
    "fig1.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "053e6a14",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "6d07d6a9",
   "metadata": {},
   "source": [
    "# Machine learning implementation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "5dd120ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['CASE_STATUS', 'SOC_TITLE', 'JOB_CITY', 'JOB_STATE', 'REQD_EDUCATION',\n",
       "       'TRAINING_REQD', 'LANG_REQD', 'CITIZENSHIP', 'ADMISSION_TYPE',\n",
       "       'WORKER_EDUCATION', 'time_taken', 'OCCUPATION', 'NEW_RELATED_MAJOR',\n",
       "       'NEW_WORKER_MAJOR', 'NEW_EMPLOYER_NAME', 'WAGE_OFFERED'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dffff_down = dffff.copy()\n",
    "dffff_down.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e55c4a81",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(514964, 16)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>SOC_TITLE</th>\n",
       "      <th>JOB_CITY</th>\n",
       "      <th>JOB_STATE</th>\n",
       "      <th>REQD_EDUCATION</th>\n",
       "      <th>TRAINING_REQD</th>\n",
       "      <th>LANG_REQD</th>\n",
       "      <th>CITIZENSHIP</th>\n",
       "      <th>ADMISSION_TYPE</th>\n",
       "      <th>WORKER_EDUCATION</th>\n",
       "      <th>time_taken</th>\n",
       "      <th>OCCUPATION</th>\n",
       "      <th>NEW_RELATED_MAJOR</th>\n",
       "      <th>NEW_WORKER_MAJOR</th>\n",
       "      <th>NEW_EMPLOYER_NAME</th>\n",
       "      <th>WAGE_OFFERED</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Omaha</td>\n",
       "      <td>NE</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Iselin</td>\n",
       "      <td>NJ</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>Mountain View</td>\n",
       "      <td>CA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>L-1</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Certified</td>\n",
       "      <td>electronics engineers, except computer</td>\n",
       "      <td>Folsom</td>\n",
       "      <td>CA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>BANGLADESH</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>Architecture &amp; Engineering</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Top 10 Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>McLean</td>\n",
       "      <td>VA</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535053</th>\n",
       "      <td>Certified</td>\n",
       "      <td>farmworkers and laborers, crop, nursery, and g...</td>\n",
       "      <td>SEMINOLE</td>\n",
       "      <td>TX</td>\n",
       "      <td>None</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>CANADA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>None</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>Others</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Below 30000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535054</th>\n",
       "      <td>Certified</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>New York</td>\n",
       "      <td>NY</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535055</th>\n",
       "      <td>Certified</td>\n",
       "      <td>architectural and engineering managers</td>\n",
       "      <td>Santa Clara</td>\n",
       "      <td>CA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>Architecture &amp; Engineering</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Top 10 Employer</td>\n",
       "      <td>Between 150000 -2000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535063</th>\n",
       "      <td>Denied</td>\n",
       "      <td>software developers, applications</td>\n",
       "      <td>IRVINE</td>\n",
       "      <td>CA</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 15 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 100000 -1500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535083</th>\n",
       "      <td>Denied</td>\n",
       "      <td>computer occupations, all other</td>\n",
       "      <td>Plano</td>\n",
       "      <td>TX</td>\n",
       "      <td>Master's</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 9 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>514964 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       CASE_STATUS                                          SOC_TITLE  \\\n",
       "0        Certified                  software developers, applications   \n",
       "1        Certified                  software developers, applications   \n",
       "2        Certified                  software developers, applications   \n",
       "3        Certified             electronics engineers, except computer   \n",
       "4        Certified                  software developers, applications   \n",
       "...            ...                                                ...   \n",
       "535053   Certified  farmworkers and laborers, crop, nursery, and g...   \n",
       "535054   Certified                  software developers, applications   \n",
       "535055   Certified             architectural and engineering managers   \n",
       "535063      Denied                  software developers, applications   \n",
       "535083      Denied                    computer occupations, all other   \n",
       "\n",
       "             JOB_CITY JOB_STATE REQD_EDUCATION TRAINING_REQD LANG_REQD  \\\n",
       "0               Omaha        NE       Master's             N         N   \n",
       "1              Iselin        NJ       Master's             N         N   \n",
       "2       Mountain View        CA       Master's             N         N   \n",
       "3              Folsom        CA       Master's             N         N   \n",
       "4              McLean        VA     Bachelor's             N         N   \n",
       "...               ...       ...            ...           ...       ...   \n",
       "535053       SEMINOLE        TX           None             N         N   \n",
       "535054       New York        NY     Bachelor's             N         N   \n",
       "535055    Santa Clara        CA       Master's             N         N   \n",
       "535063         IRVINE        CA     Bachelor's             N         N   \n",
       "535083          Plano        TX       Master's             N         N   \n",
       "\n",
       "        CITIZENSHIP ADMISSION_TYPE WORKER_EDUCATION          time_taken  \\\n",
       "0             INDIA           H-1B       Bachelor's   at least 6 months   \n",
       "1             INDIA           H-1B         Master's   at least 6 months   \n",
       "2       SOUTH KOREA            L-1         Master's   at least 6 months   \n",
       "3        BANGLADESH           H-1B         Master's   at least 6 months   \n",
       "4             INDIA           H-1B         Master's   at least 6 months   \n",
       "...             ...            ...              ...                 ...   \n",
       "535053       CANADA           H-1B             None   at least 3 months   \n",
       "535054        INDIA           H-1B       Bachelor's   at least 3 months   \n",
       "535055        INDIA           H-1B         Master's   at least 3 months   \n",
       "535063        INDIA           H-1B       Bachelor's  at least 15 months   \n",
       "535083        INDIA           H-1B       Bachelor's   at least 9 months   \n",
       "\n",
       "                        OCCUPATION NEW_RELATED_MAJOR NEW_WORKER_MAJOR  \\\n",
       "0             computer occupations        STEM Major       STEM Major   \n",
       "1             computer occupations        STEM Major   NON-STEM Major   \n",
       "2             computer occupations        STEM Major       STEM Major   \n",
       "3       Architecture & Engineering        STEM Major       STEM Major   \n",
       "4             computer occupations        STEM Major       STEM Major   \n",
       "...                            ...               ...              ...   \n",
       "535053                      Others        STEM Major       STEM Major   \n",
       "535054        computer occupations    NON-STEM Major       STEM Major   \n",
       "535055  Architecture & Engineering    NON-STEM Major       STEM Major   \n",
       "535063        computer occupations        STEM Major       STEM Major   \n",
       "535083        computer occupations        STEM Major       STEM Major   \n",
       "\n",
       "       NEW_EMPLOYER_NAME             WAGE_OFFERED  \n",
       "0         Other Employer    Between 60000 -100000  \n",
       "1         Other Employer    Between 60000 -100000  \n",
       "2         Other Employer    Between 60000 -100000  \n",
       "3        Top 10 Employer    Between 60000 -100000  \n",
       "4         Other Employer    Between 60000 -100000  \n",
       "...                  ...                      ...  \n",
       "535053    Other Employer              Below 30000  \n",
       "535054    Other Employer    Between 60000 -100000  \n",
       "535055   Top 10 Employer  Between 150000 -2000000  \n",
       "535063    Other Employer  Between 100000 -1500000  \n",
       "535083    Other Employer    Between 60000 -100000  \n",
       "\n",
       "[514964 rows x 16 columns]"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dffff_down = dffff_down[dffff_down['CASE_STATUS'] != 'Withdrawn']\n",
    "print(dffff_down.shape)\n",
    "dffff_down"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "29ed9312",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         Certified\n",
       "1         Certified\n",
       "2         Certified\n",
       "3         Certified\n",
       "4         Certified\n",
       "            ...    \n",
       "535053    Certified\n",
       "535054    Certified\n",
       "535055    Certified\n",
       "535063       Denied\n",
       "535083       Denied\n",
       "Name: CASE_STATUS, Length: 514964, dtype: object"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dffff_down['CASE_STATUS']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7c420c54",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0         1\n",
       "1         1\n",
       "2         1\n",
       "3         1\n",
       "4         1\n",
       "         ..\n",
       "535053    1\n",
       "535054    1\n",
       "535055    1\n",
       "535063    0\n",
       "535083    0\n",
       "Name: CASE_STATUS, Length: 514964, dtype: int64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#replace 'CERTIFIED' and 'DENIED' label of 'CASE_STATUS' respectively with '1' and '0'\n",
    "dffff_down['CASE_STATUS'] = dffff_down['CASE_STATUS'].replace({'Certified': 1,'Denied':0})\n",
    "dffff_down.CASE_STATUS.astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "cec8abed",
   "metadata": {},
   "outputs": [],
   "source": [
    "type(dffff_down)\n",
    "dffff_down[['OCCUPATION', 'JOB_CITY', 'JOB_STATE', 'REQD_EDUCATION', 'CITIZENSHIP', 'ADMISSION_TYPE', 'WORKER_EDUCATION','NEW_RELATED_MAJOR', 'NEW_WORKER_MAJOR', 'NEW_EMPLOYER_NAME', 'WAGE_OFFERED']] = dffff_down[['OCCUPATION', 'JOB_CITY', 'JOB_STATE', 'REQD_EDUCATION', 'CITIZENSHIP', 'ADMISSION_TYPE', 'WORKER_EDUCATION','NEW_RELATED_MAJOR', 'NEW_WORKER_MAJOR', 'NEW_EMPLOYER_NAME', 'WAGE_OFFERED']].apply(lambda x: x.astype('category'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f666597e",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "dffff_down['JOB_CITY'] = dffff_down['JOB_CITY'].str.lower()\n",
    "dff_final = dffff_down.drop(['time_taken', 'SOC_TITLE', 'LANG_REQD'], axis=1)\n",
    "df_final = dff_final.copy()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "056660d3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CASE_STATUS             int64\n",
       "JOB_CITY               object\n",
       "JOB_STATE            category\n",
       "REQD_EDUCATION       category\n",
       "TRAINING_REQD          object\n",
       "CITIZENSHIP          category\n",
       "ADMISSION_TYPE       category\n",
       "WORKER_EDUCATION     category\n",
       "OCCUPATION           category\n",
       "NEW_RELATED_MAJOR    category\n",
       "NEW_WORKER_MAJOR     category\n",
       "NEW_EMPLOYER_NAME    category\n",
       "WAGE_OFFERED         category\n",
       "dtype: object"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_final.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "efa76ffa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CASE_STATUS\n",
      "JOB_CITY\n",
      "JOB_STATE\n",
      "REQD_EDUCATION\n",
      "TRAINING_REQD\n",
      "CITIZENSHIP\n",
      "ADMISSION_TYPE\n",
      "WORKER_EDUCATION\n",
      "OCCUPATION\n",
      "NEW_RELATED_MAJOR\n",
      "NEW_WORKER_MAJOR\n",
      "NEW_EMPLOYER_NAME\n",
      "WAGE_OFFERED\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 514964 entries, 0 to 535083\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count   Dtype\n",
      "---  ------             --------------   -----\n",
      " 0   CASE_STATUS        514964 non-null  int64\n",
      " 1   JOB_CITY           514964 non-null  int64\n",
      " 2   JOB_STATE          514964 non-null  int64\n",
      " 3   REQD_EDUCATION     514964 non-null  int64\n",
      " 4   TRAINING_REQD      514964 non-null  int64\n",
      " 5   CITIZENSHIP        514964 non-null  int64\n",
      " 6   ADMISSION_TYPE     514964 non-null  int64\n",
      " 7   WORKER_EDUCATION   514964 non-null  int64\n",
      " 8   OCCUPATION         514964 non-null  int64\n",
      " 9   NEW_RELATED_MAJOR  514964 non-null  int64\n",
      " 10  NEW_WORKER_MAJOR   514964 non-null  int64\n",
      " 11  NEW_EMPLOYER_NAME  514964 non-null  int64\n",
      " 12  WAGE_OFFERED       514964 non-null  int64\n",
      "dtypes: int64(13)\n",
      "memory usage: 55.0 MB\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "categorical_variables = {}\n",
    "\n",
    "#Creating categories denoted by integers from column values\n",
    "for col in df_final.columns:\n",
    "    cat_var_name = \"cat_\"+ col\n",
    "    cat_var_name = LabelEncoder()\n",
    "    print(col)\n",
    "    cat_var_name.fit(df_final[col])\n",
    "    df_final[col] = cat_var_name.transform(df_final[col])\n",
    "    categorical_variables[col] = cat_var_name\n",
    "\n",
    "df_final.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "94ebad56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(514964, 13)\n",
      "(514964, 13)\n"
     ]
    }
   ],
   "source": [
    "dff_final['JOB_CITY']\n",
    "print(dff_final.shape)\n",
    "print(df_final.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "9c806028",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "514964\n",
      "514964\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'omaha': 4921,\n",
       " 'iselin': 3173,\n",
       " 'mountain view': 4405,\n",
       " 'folsom': 2264,\n",
       " 'mclean': 4056,\n",
       " 'san francisco': 5874,\n",
       " 'porterville': 5374,\n",
       " 'san diego': 5863,\n",
       " 'falls church': 2168,\n",
       " 'jersey city': 3234,\n",
       " 'new york': 4615,\n",
       " 'hillsboro': 2955,\n",
       " 'santa maria': 5965,\n",
       " 'salt lake city': 5837,\n",
       " 'savannah': 5996,\n",
       " 'fort wayne': 2325,\n",
       " 'west conshohocken': 7209,\n",
       " 'milpitas': 4219,\n",
       " 'east hampton': 1868,\n",
       " 'san jose': 5894,\n",
       " 'muncie': 4449,\n",
       " 'northville': 4796,\n",
       " 'cambridge': 985,\n",
       " 'east moline': 1886,\n",
       " 'fort lauderdale': 2298,\n",
       " 'rosemead': 5707,\n",
       " 'southborough': 6317,\n",
       " 'college station': 1362,\n",
       " 'pittsburgh': 5258,\n",
       " 'bothell': 755,\n",
       " 'philadelphia': 5191,\n",
       " 'las vegas': 3574,\n",
       " 'cranbury': 1500,\n",
       " 'oklahoma city': 4888,\n",
       " 'piscataway': 5247,\n",
       " 'north wales': 4776,\n",
       " 'elkhart': 1998,\n",
       " 'iowa city': 3157,\n",
       " 'hanover': 2774,\n",
       " 'houston': 3063,\n",
       " 'westborough': 7286,\n",
       " 'birmingham': 644,\n",
       " 'danbury': 1602,\n",
       " 'washington': 7099,\n",
       " 'southlake': 6324,\n",
       " 'cleveland': 1310,\n",
       " 'montgomery': 4313,\n",
       " 'gaithersburg': 2442,\n",
       " 'apopka': 243,\n",
       " 'gallina': 2452,\n",
       " 'tumon': 6854,\n",
       " 'naperville': 4501,\n",
       " 'carol stream': 1050,\n",
       " 'schaumburg': 6008,\n",
       " 'meriden': 4112,\n",
       " 'fremont': 2387,\n",
       " 'annandale': 224,\n",
       " 'charlotte': 1176,\n",
       " 'fort collins': 2293,\n",
       " 'verona': 6979,\n",
       " 'denver': 1690,\n",
       " 'atlanta': 320,\n",
       " 'monmouth jct': 4286,\n",
       " 'sheffield': 6111,\n",
       " 'brooklyn': 862,\n",
       " 'austin': 350,\n",
       " 'san ramon': 5917,\n",
       " 'columbus': 1383,\n",
       " 'joliet': 3259,\n",
       " 'colorado springs': 1377,\n",
       " 'miami': 4135,\n",
       " 'chicago': 1225,\n",
       " 'troy': 6837,\n",
       " 'teaneck': 6686,\n",
       " 'cedar park': 1110,\n",
       " 'boulder': 758,\n",
       " 'monmouth junction': 4288,\n",
       " 'coral gables': 1432,\n",
       " 'tampa': 6662,\n",
       " 'dayton': 1625,\n",
       " 'norman': 4692,\n",
       " 'ellicott city': 2011,\n",
       " 'jamaica': 3204,\n",
       " 'winona': 7442,\n",
       " 'herndon': 2904,\n",
       " 'chantilly': 1160,\n",
       " 'dallas': 1592,\n",
       " 'titusville': 6774,\n",
       " 'shiprock': 6132,\n",
       " 'edison': 1952,\n",
       " 'san bernardino': 5858,\n",
       " 'sunnyvale': 6603,\n",
       " 'louisville': 3805,\n",
       " 'durham': 1820,\n",
       " 'burke': 922,\n",
       " 'crownpoint': 1544,\n",
       " 'bellevue': 539,\n",
       " 'san antonio': 5852,\n",
       " 'fairfax': 2137,\n",
       " 'little rock': 3726,\n",
       " 'castle rock': 1084,\n",
       " 'norwood': 4807,\n",
       " 'novato': 4812,\n",
       " 'boston': 751,\n",
       " 'katy': 3301,\n",
       " 'saginaw': 5799,\n",
       " 'boxborough': 769,\n",
       " 'doral': 1757,\n",
       " 'union city': 6890,\n",
       " 'rochester': 5643,\n",
       " 'pleasant prairie': 5299,\n",
       " 'aguadilla': 92,\n",
       " 'westchester': 7290,\n",
       " 'columbia': 1380,\n",
       " 'rosemont': 5709,\n",
       " 'canoga park': 1006,\n",
       " 'wainscott': 7031,\n",
       " 'fort lee': 2299,\n",
       " 'aliso viejo': 140,\n",
       " 'nashua': 4507,\n",
       " 'petersburg': 5179,\n",
       " 'apex': 242,\n",
       " 'san pedro': 5915,\n",
       " 'culver city': 1562,\n",
       " 'plano': 5280,\n",
       " 'manteca': 3923,\n",
       " 'escalon': 2080,\n",
       " 'irving': 3166,\n",
       " 'santa clara': 5955,\n",
       " 'hauppauge': 2833,\n",
       " 'los angeles': 3785,\n",
       " 'sterling': 6469,\n",
       " 'germantown': 2499,\n",
       " 'portland': 5375,\n",
       " 'bethesda': 603,\n",
       " 'seattle': 6052,\n",
       " 'menlo park': 4098,\n",
       " 'somers': 6219,\n",
       " 'greenwood village': 2678,\n",
       " 'provo': 5429,\n",
       " 'lauderhill': 3585,\n",
       " 'north brunswick': 4712,\n",
       " 'tallahassee': 6656,\n",
       " 'framingham': 2351,\n",
       " 'orangeburg': 4943,\n",
       " 'tempe': 6696,\n",
       " 'worcester': 7515,\n",
       " 'arlington heights': 273,\n",
       " 'el dorado hills': 1974,\n",
       " 'palo alto': 5049,\n",
       " 'winesburg': 7431,\n",
       " 'milwaukee': 4222,\n",
       " 'minnetonka': 4237,\n",
       " 'harbor beach': 2783,\n",
       " 'springfield': 6364,\n",
       " 'beltsville': 553,\n",
       " 'south windsor': 6313,\n",
       " 'symmes township': 6641,\n",
       " 'stamford': 6441,\n",
       " 'redmond': 5519,\n",
       " 'melville': 4088,\n",
       " 'peoria': 5161,\n",
       " 'palm beach gardens': 5036,\n",
       " 'high point': 2931,\n",
       " 'south burlington': 6246,\n",
       " 'rockville': 5664,\n",
       " 'saint louis': 5815,\n",
       " 'redwood shores': 5527,\n",
       " 'kirkland': 3403,\n",
       " 'baltimore': 403,\n",
       " 'flourtown': 2253,\n",
       " 'baton rouge': 455,\n",
       " 'acton': 71,\n",
       " 'alpharetta': 158,\n",
       " 'pasadena': 5099,\n",
       " 'la mirada': 3447,\n",
       " 'torrance': 6802,\n",
       " 'stillwater': 6482,\n",
       " 'norcross': 4684,\n",
       " 'glastonbury': 2529,\n",
       " 'hermosa beach': 2901,\n",
       " 'monee': 4282,\n",
       " 'ashburn': 291,\n",
       " 'west point': 7262,\n",
       " 'juno beach': 3275,\n",
       " 'rowland heights': 5740,\n",
       " 'moon township': 4324,\n",
       " 'deerfield': 1651,\n",
       " 'vernon': 6975,\n",
       " 'newark': 4628,\n",
       " 'centreville': 1138,\n",
       " 'cedar rapids': 1111,\n",
       " 'duluth': 1798,\n",
       " 'littleton': 3732,\n",
       " 'lake forest': 3497,\n",
       " 'lorton': 3780,\n",
       " 'moonachie': 4329,\n",
       " 'fairview': 2150,\n",
       " 'beaverton': 499,\n",
       " 'enola': 2068,\n",
       " 'palo alo': 5048,\n",
       " 'darien': 1611,\n",
       " 'richardson': 5569,\n",
       " 'madison': 3864,\n",
       " 'waterford': 7122,\n",
       " 'ann arbor': 221,\n",
       " 'pleasanton': 5301,\n",
       " 'clovis': 1328,\n",
       " 'moline': 4275,\n",
       " 'north bay village': 4703,\n",
       " 'orange': 4938,\n",
       " 'jupiter': 3276,\n",
       " 'southfield': 6320,\n",
       " 'morristown': 4359,\n",
       " 'oakland': 4842,\n",
       " 'libertyville': 3678,\n",
       " 'fresno': 2395,\n",
       " 'la vista': 3459,\n",
       " 'avon': 363,\n",
       " 'bala cynwyd': 391,\n",
       " 'reston': 5552,\n",
       " 'albany': 112,\n",
       " 'northridge': 4793,\n",
       " 'belleville': 538,\n",
       " 'ramsey': 5467,\n",
       " 'burlington': 925,\n",
       " 'farmington hills': 2191,\n",
       " 'ossining': 4980,\n",
       " 'encino': 2053,\n",
       " 'itasca': 3186,\n",
       " 'siloam springs': 6154,\n",
       " 'jacksonville': 3200,\n",
       " 'richmond': 5580,\n",
       " 'mettawa': 4130,\n",
       " 'bronx': 853,\n",
       " 'glen allen': 2530,\n",
       " 'wilmington': 7409,\n",
       " 'charleston': 1172,\n",
       " 'foster city': 2339,\n",
       " 'gardena': 2465,\n",
       " 'newton': 4650,\n",
       " 'king of prussia': 3381,\n",
       " 'carpentersville': 1053,\n",
       " 'piedmont': 5210,\n",
       " 'buffalo': 906,\n",
       " 'lexington': 3670,\n",
       " 'radnor': 5459,\n",
       " 'lantana': 3558,\n",
       " 'westminster': 7305,\n",
       " 'carrollton': 1061,\n",
       " 'san mateo': 5911,\n",
       " 'campbell': 994,\n",
       " 'englewood': 2057,\n",
       " 'glendale': 2544,\n",
       " 'arcadia': 255,\n",
       " 'watsonville': 7134,\n",
       " 'broomfield': 877,\n",
       " 'waxhaw': 7147,\n",
       " 'raleigh': 5463,\n",
       " 'east brunswick': 1852,\n",
       " 'cherry hill': 1202,\n",
       " 'irvine': 3165,\n",
       " 'st. martinville': 6411,\n",
       " 'sugar land': 6533,\n",
       " 'douglas': 1766,\n",
       " 'carson': 1065,\n",
       " 'rancho palos verdes': 5477,\n",
       " 'clearwater': 1304,\n",
       " 'brooklyn park': 867,\n",
       " 'signal hill': 6150,\n",
       " 'seaside': 6047,\n",
       " 'san bruno': 5859,\n",
       " 'braintree': 787,\n",
       " 'el segundo': 1979,\n",
       " 'manassas': 3893,\n",
       " 'alhambra': 137,\n",
       " 'mendota heights': 4096,\n",
       " 'east wareham': 1917,\n",
       " 'fullerton': 2422,\n",
       " 'brisbane': 841,\n",
       " 'warren': 7080,\n",
       " 'whittier': 7361,\n",
       " 'lehi': 3632,\n",
       " 'south plainfield': 6289,\n",
       " 'san carlos': 5860,\n",
       " 'carrington': 1056,\n",
       " 'hagerstown': 2732,\n",
       " 'duncan': 1804,\n",
       " 'newport beach': 4646,\n",
       " 'bensenville': 565,\n",
       " 'suite: 202': 6571,\n",
       " 'st. clair shores': 6392,\n",
       " 'york': 7556,\n",
       " 'st. paul': 6416,\n",
       " 'minot': 4241,\n",
       " 'chicopee': 1234,\n",
       " 'van nuys': 6953,\n",
       " 'brentwood': 813,\n",
       " 'elmhurst': 2020,\n",
       " 'somerset': 6222,\n",
       " 'syracuse': 6644,\n",
       " 'amelia': 189,\n",
       " 'pennsauken': 5158,\n",
       " 'escondido': 2081,\n",
       " 'keasbey': 3310,\n",
       " 'shattuck': 6103,\n",
       " 'huntersville': 3087,\n",
       " 'rocklin': 5661,\n",
       " 'north kansas city': 4739,\n",
       " 'chelmsford': 1193,\n",
       " 'cary': 1072,\n",
       " 'east windsor': 1920,\n",
       " 'fort myers': 2311,\n",
       " 'forest hills': 2277,\n",
       " 'pawcatuck': 5118,\n",
       " 'wichita': 7362,\n",
       " 'annapolis': 227,\n",
       " 'shreveport': 6142,\n",
       " 'sturgeon bay': 6521,\n",
       " 'cupertino': 1570,\n",
       " 'norristown': 4694,\n",
       " 'wakefield': 7035,\n",
       " 'delray beach': 1678,\n",
       " 'leawood': 3618,\n",
       " 'detroit': 1713,\n",
       " 'owings mills': 5001,\n",
       " 'waller': 7047,\n",
       " 'tamuning': 6664,\n",
       " 'morgantown': 4349,\n",
       " 'lawrence': 3600,\n",
       " 'bloomfield': 675,\n",
       " 'morganton': 4348,\n",
       " 'pinon': 5242,\n",
       " 'bethlehem': 605,\n",
       " 'wexford': 7320,\n",
       " 'richardson,': 5570,\n",
       " 'college point': 1361,\n",
       " 'plantation': 5282,\n",
       " 'brevard': 815,\n",
       " 'huntington': 3090,\n",
       " 'mechanicsburg': 4073,\n",
       " 'livonia': 3736,\n",
       " 'delano': 1667,\n",
       " 'kealakekua': 3305,\n",
       " 'latrobe': 3581,\n",
       " 'redwood city': 5523,\n",
       " 'livingston': 3735,\n",
       " 'minneapolis': 4234,\n",
       " 'chicago heights': 1226,\n",
       " 'lambertville': 3534,\n",
       " 'scottsdale': 6029,\n",
       " 'pecos': 5137,\n",
       " 'lagrange': 3476,\n",
       " 'indianapolis': 3137,\n",
       " 'port wentworth': 5367,\n",
       " 'georgetown': 2496,\n",
       " 'espanola': 2083,\n",
       " 'panama city': 5061,\n",
       " 'wellesley': 7170,\n",
       " 'pharr': 5187,\n",
       " 'hodgkins': 2985,\n",
       " 'ny': 4817,\n",
       " 'marlborough': 3961,\n",
       " 'dearborn': 1635,\n",
       " 'san clemente': 5861,\n",
       " 'malta': 3884,\n",
       " 'chesapeake': 1206,\n",
       " 'alcoa center': 122,\n",
       " 'solana beach': 6214,\n",
       " 'lawrenceville': 3604,\n",
       " 'long island city': 3768,\n",
       " 'fowlerville': 2347,\n",
       " 'greenville': 2674,\n",
       " 'franklin': 2358,\n",
       " 'lanham': 3553,\n",
       " 'camarillo': 980,\n",
       " 'draper': 1780,\n",
       " 'blue bell': 689,\n",
       " 'west palm beach': 7258,\n",
       " 'kingwood': 3397,\n",
       " 'valencia': 6929,\n",
       " \"downer's grove\": 1773,\n",
       " 'rockleigh': 5660,\n",
       " 'lincolnshire': 3694,\n",
       " 'erlanger': 2078,\n",
       " 'santa monica': 5966,\n",
       " 'woodbridge': 7480,\n",
       " 'new bern': 4553,\n",
       " 'alexandria': 131,\n",
       " 'virginia beach': 7005,\n",
       " 'topeka': 6799,\n",
       " 'johns creek': 3249,\n",
       " 'north franklin': 4731,\n",
       " 'greensboro': 2671,\n",
       " 'flushing': 2259,\n",
       " 'sayreville': 6001,\n",
       " 'gowanda': 2592,\n",
       " 'highland hills': 2938,\n",
       " 'ellisville': 2013,\n",
       " 'dulles': 1797,\n",
       " 'sugar mountain': 6534,\n",
       " 'frisco': 2400,\n",
       " 'south riding': 6299,\n",
       " 'big stone gap': 625,\n",
       " 'spartanburg': 6336,\n",
       " 'eden prairie': 1940,\n",
       " 'skokie': 6183,\n",
       " 'kensington': 3338,\n",
       " 'evanston': 2109,\n",
       " 'fairfield': 2140,\n",
       " 'west valley': 7275,\n",
       " 'foothill ranch': 2268,\n",
       " 'plainfield': 5272,\n",
       " 'tucson': 6847,\n",
       " 'knoxville': 3418,\n",
       " 'langhorne': 3551,\n",
       " 'henderson': 2887,\n",
       " 'yuma': 7569,\n",
       " 'orem': 4954,\n",
       " 'fredericksburg': 2374,\n",
       " 'fridley': 2397,\n",
       " 'coppell': 1427,\n",
       " 'sidney': 6147,\n",
       " 'canonsburg': 1008,\n",
       " 'del mar': 1659,\n",
       " 'phoenix': 5201,\n",
       " 'woonsocket': 7513,\n",
       " 'secaucus': 6057,\n",
       " 'malvern': 3885,\n",
       " 'converse': 1408,\n",
       " 'moorestown': 4334,\n",
       " 'statesboro': 6455,\n",
       " 'bloomington': 681,\n",
       " 'richfield': 5574,\n",
       " 'rancho cordova': 5470,\n",
       " 'anaheim': 205,\n",
       " 'paramus': 5072,\n",
       " 'fond du lac': 2266,\n",
       " 'mcallen': 4030,\n",
       " 'progreso': 5419,\n",
       " 'huntington beach': 3091,\n",
       " 'poplar bluff': 5340,\n",
       " 'shrewsbury': 6144,\n",
       " 'honolulu': 3024,\n",
       " 'vienna': 6990,\n",
       " 'durant': 1819,\n",
       " 'rancho dominguez': 5474,\n",
       " 'albuquerque': 118,\n",
       " 'riverside': 5626,\n",
       " 'gillette': 2517,\n",
       " 'east lansing': 1877,\n",
       " 'metuchen': 4132,\n",
       " 'auburn hills': 338,\n",
       " 'middletown': 4165,\n",
       " 'claxton': 1292,\n",
       " 'parlin': 5091,\n",
       " 'channelview': 1159,\n",
       " 'chandler': 1155,\n",
       " 'modesto': 4267,\n",
       " 'englewood cliffs': 2060,\n",
       " 'montvale': 4320,\n",
       " 'belmont': 549,\n",
       " 'silver spring': 6158,\n",
       " 'newtown': 4655,\n",
       " 'novi': 4814,\n",
       " 'burbank': 917,\n",
       " 'west chester': 7205,\n",
       " 'woodcliff lake': 7483,\n",
       " 'allentown': 147,\n",
       " 'research triangle park': 5549,\n",
       " 'lake worth': 3519,\n",
       " 'morrisville': 4360,\n",
       " 'newtown square': 4656,\n",
       " 'basking ridge': 443,\n",
       " 'ada': 73,\n",
       " 'danville': 1607,\n",
       " 'longmont': 3772,\n",
       " 'hidalgo': 2924,\n",
       " 'catonsville': 1094,\n",
       " 'arlington': 272,\n",
       " 'hudson': 3075,\n",
       " 'aurora': 346,\n",
       " 'natick': 4514,\n",
       " 'norfolk': 4688,\n",
       " 'cheshire': 1209,\n",
       " 'jasper': 3214,\n",
       " 'princeton': 5416,\n",
       " 'parsippany': 5096,\n",
       " 'eunice': 2102,\n",
       " 'champaign': 1151,\n",
       " 'beeville': 510,\n",
       " 'lansing': 3557,\n",
       " 'cusseta': 1574,\n",
       " 'jamesburg': 3208,\n",
       " 'university park': 6905,\n",
       " 'west columbia': 7208,\n",
       " 'coquille': 1431,\n",
       " 'suwanee': 6618,\n",
       " 'west warwick': 7278,\n",
       " 'white plains': 7340,\n",
       " 'orlando': 4959,\n",
       " 'merrillville': 4119,\n",
       " 'glenwood springs': 2564,\n",
       " 'downers grove': 1774,\n",
       " 'livermore': 3733,\n",
       " 'wilton': 7414,\n",
       " 'park ridge': 5079,\n",
       " 'olathe': 4891,\n",
       " 'kent': 3339,\n",
       " 'algona': 135,\n",
       " 'norwalk': 4802,\n",
       " 'gainesville': 2436,\n",
       " 'north chicago': 4719,\n",
       " 'pineville': 5240,\n",
       " 'milford': 4185,\n",
       " 'manchester city': 3902,\n",
       " 'pennington': 5156,\n",
       " 'glendora': 2547,\n",
       " 'evansville': 2110,\n",
       " 'nashville': 4508,\n",
       " 'kingsport': 3393,\n",
       " 'st. petersburg': 6424,\n",
       " 'marysville': 3983,\n",
       " 'edsion': 1958,\n",
       " 'clinton': 1320,\n",
       " 'fresh meadows': 2394,\n",
       " 'malibu': 3882,\n",
       " 'ionia': 3155,\n",
       " 'battle creek': 457,\n",
       " 'guilford': 2708,\n",
       " 'decatur': 1638,\n",
       " 'pomona': 5325,\n",
       " 'akron': 101,\n",
       " 'st. charles': 6391,\n",
       " 'covina': 1491,\n",
       " 'northfield': 4785,\n",
       " 'south brunswick': 6245,\n",
       " 'santa ana': 5951,\n",
       " 'la palma': 3448,\n",
       " 'miami lakes': 4140,\n",
       " 'dublin': 1787,\n",
       " 'lodi': 3746,\n",
       " 'hialeah': 2915,\n",
       " 'doraville': 1760,\n",
       " 'walnut creek': 7054,\n",
       " 'rye': 5772,\n",
       " 'pullman': 5436,\n",
       " 'bridgeport': 829,\n",
       " 'lacey': 3461,\n",
       " 'federal way': 2202,\n",
       " 'stanton': 6445,\n",
       " '800 n. state college blvd.': 49,\n",
       " 'sacramento': 5789,\n",
       " 'cullowhee': 1560,\n",
       " 'danvers': 1606,\n",
       " 'hopewell junction': 3035,\n",
       " 'naples': 4502,\n",
       " 'rancho cucamonga': 5471,\n",
       " 'berkeley': 576,\n",
       " 'medley': 4081,\n",
       " 'kahului': 3281,\n",
       " 'elizabeth': 1990,\n",
       " 'ballston lake': 399,\n",
       " 'woburn': 7469,\n",
       " 'carlsbad': 1038,\n",
       " 'saint rose': 5823,\n",
       " 'needham': 4530,\n",
       " 'north providence': 4759,\n",
       " 'north hills': 4735,\n",
       " 'johnstown': 3257,\n",
       " 'fountain valley': 2345,\n",
       " 'cape girardeau': 1021,\n",
       " 'the woodlands': 6723,\n",
       " 'mccook': 4038,\n",
       " 'santa clarita': 5957,\n",
       " 'boise': 719,\n",
       " 'tustin': 6867,\n",
       " 'sunrise': 6606,\n",
       " 'lombard': 3754,\n",
       " 'mansfield': 3922,\n",
       " 'westport': 7313,\n",
       " 'chaddsford': 1144,\n",
       " 'east greenbush': 1865,\n",
       " 'bushnell': 939,\n",
       " 'fontana': 2267,\n",
       " 'west hartford': 7223,\n",
       " 'burlingame': 924,\n",
       " 'normal': 4691,\n",
       " 'sleepy hollow': 6190,\n",
       " 'san juan capistrano': 5898,\n",
       " 'towson': 6816,\n",
       " 'hunt valley': 3086,\n",
       " 'clive': 1324,\n",
       " 'rutland': 5771,\n",
       " 'holmdel': 3008,\n",
       " 'rolling meadows': 5684,\n",
       " 'smithfield': 6194,\n",
       " 'bartlett': 436,\n",
       " 'la crosse': 3437,\n",
       " 'laplata': 3561,\n",
       " 'providence': 5425,\n",
       " 'caribou': 1032,\n",
       " 'college park': 1359,\n",
       " 'bedford': 502,\n",
       " 'port charlotte': 5344,\n",
       " 'cabin john': 949,\n",
       " 'la jolla': 3443,\n",
       " 'prescott': 5405,\n",
       " 'kenner': 3330,\n",
       " 'sherman': 6127,\n",
       " 'fishers': 2224,\n",
       " 'santa barbara': 5953,\n",
       " 'st. louis': 6409,\n",
       " 'clawson': 1291,\n",
       " 'edinburg': 1951,\n",
       " 'monterey': 4307,\n",
       " 'grand rapids': 2616,\n",
       " 'whitestown': 7351,\n",
       " 'exton': 2125,\n",
       " 'lewis center': 3662,\n",
       " 'hattiesburg': 2829,\n",
       " 'hayward': 2857,\n",
       " 'robbinsville': 5636,\n",
       " 'florence': 2245,\n",
       " 'duarte': 1786,\n",
       " 'wallingford': 7048,\n",
       " 'whitsett': 7359,\n",
       " 'elk grove village': 1995,\n",
       " 'wayzata': 7154,\n",
       " 'griffin': 2686,\n",
       " 'walnut': 7052,\n",
       " 'calabasas': 957,\n",
       " 'memphis': 4091,\n",
       " 'randolph': 5481,\n",
       " 'tuscaloosa': 6861,\n",
       " 'west covina': 7210,\n",
       " 'maite': 3878,\n",
       " 'northbrook': 4784,\n",
       " 'boca raton': 710,\n",
       " 'hackensack': 2726,\n",
       " 'waltham': 7060,\n",
       " 'melbourne': 4084,\n",
       " 'meadville': 4069,\n",
       " 'overland park': 4992,\n",
       " 'hollywood': 3006,\n",
       " 'fort worth': 2328,\n",
       " 'riverwoods': 5630,\n",
       " 'shakopee': 6096,\n",
       " 'mason': 3986,\n",
       " 'westlake village': 7302,\n",
       " 'monroeville': 4295,\n",
       " 'west des moines': 7213,\n",
       " 'chino hills': 1242,\n",
       " 'brainerd': 786,\n",
       " 'vernon hills': 6976,\n",
       " 'goshen': 2589,\n",
       " 'chino': 1240,\n",
       " 'storrs': 6501,\n",
       " 'gallup': 2455,\n",
       " 'kansas city': 3295,\n",
       " 'dunn loring': 1815,\n",
       " 'salem': 5826,\n",
       " 'north charleston': 4716,\n",
       " 'riviera beach': 5631,\n",
       " 'baytown': 482,\n",
       " 'chesterfield': 1213,\n",
       " 'neenah': 4534,\n",
       " 'miami beach': 4136,\n",
       " 'elyria': 2040,\n",
       " 'millersville': 4204,\n",
       " 'bridgewater': 833,\n",
       " 'pittsburg': 5257,\n",
       " 'manalapan': 3891,\n",
       " 'greeley': 2649,\n",
       " 'ridge': 5590,\n",
       " 'wise': 7461,\n",
       " 'tonawanda': 6794,\n",
       " 'niles': 4667,\n",
       " 'brookings': 858,\n",
       " 'alameda': 104,\n",
       " 'matawan': 3997,\n",
       " 'coraopolis': 1438,\n",
       " 'cincinnati': 1257,\n",
       " 'mississippi state': 4253,\n",
       " 'corona': 1452,\n",
       " 'golden': 2572,\n",
       " 'st. thomas': 6434,\n",
       " 'charlottesville': 1179,\n",
       " 'tulsa': 6853,\n",
       " 'ruston': 5765,\n",
       " 'windsor': 7426,\n",
       " 'chesterbrook': 1212,\n",
       " 'mandeville': 3905,\n",
       " 'saint joseph': 5813,\n",
       " 'chattanooga': 1187,\n",
       " 'spring': 6351,\n",
       " 'frostburg': 2402,\n",
       " 'redlands': 5516,\n",
       " 'malone': 3883,\n",
       " 'murfreesboro': 4455,\n",
       " 'medway': 4082,\n",
       " 'east peoria': 1896,\n",
       " 'center': 1120,\n",
       " 'lake city': 3492,\n",
       " 'new kensington': 4586,\n",
       " 'city of industry': 1269,\n",
       " 'blue island': 691,\n",
       " 'burr ridge': 935,\n",
       " 'elkton': 2003,\n",
       " 'dorchester': 1761,\n",
       " 'portsmouth': 5378,\n",
       " 'eagan': 1835,\n",
       " 'irwindale': 3171,\n",
       " 'san francsico': 5882,\n",
       " 'plainsboro': 5274,\n",
       " 'rose hill': 5698,\n",
       " 'monrovia': 4296,\n",
       " 'highland village': 2944,\n",
       " 'bedminster': 507,\n",
       " \"o'fallon\": 4822,\n",
       " 'staten island': 6454,\n",
       " 'hermiston': 2899,\n",
       " 'lyndhurst': 3841,\n",
       " 'spring hill': 6356,\n",
       " 'kapolei': 3297,\n",
       " 'fort smith': 2317,\n",
       " 'hilliard': 2953,\n",
       " 'keithville': 3314,\n",
       " 'mission': 4249,\n",
       " 'lincoln': 3689,\n",
       " 'west valley city': 7276,\n",
       " 'wixom': 7466,\n",
       " 'san dimas': 5867,\n",
       " 'miami gardens': 4138,\n",
       " 'odessa': 4871,\n",
       " 'ronkonkoma': 5694,\n",
       " 'temple terrace': 6702,\n",
       " 'noblesville': 4674,\n",
       " 'garfield': 2469,\n",
       " 'farmville': 2195,\n",
       " 'flemington': 2238,\n",
       " 'methuen': 4128,\n",
       " 'goleta': 2578,\n",
       " 'plymouth': 5309,\n",
       " 'poughkeepsie': 5389,\n",
       " 'smithtown': 6195,\n",
       " 'tyler': 6877,\n",
       " 'montebello': 4305,\n",
       " 'vancouver': 6958,\n",
       " 'cumberland': 1563,\n",
       " 'elgin': 1987,\n",
       " 'wyomissing': 7530,\n",
       " 'andover': 212,\n",
       " 'nutley': 4816,\n",
       " 'ithaca': 3187,\n",
       " 'west lafayette': 7238,\n",
       " 'anderson': 211,\n",
       " 'humble': 3082,\n",
       " 'carnegie': 1046,\n",
       " 'santa cruz': 5959,\n",
       " 'tinton falls': 6768,\n",
       " 'sandy': 5937,\n",
       " 'chatsworth': 1186,\n",
       " 'canton': 1010,\n",
       " 'coupeville': 1486,\n",
       " 'bakersfield': 388,\n",
       " 'culpeper': 1561,\n",
       " 'jamaica plain': 3207,\n",
       " 'harrison': 2800,\n",
       " 'ocean': 4855,\n",
       " 'trenton': 6827,\n",
       " 'auburn': 335,\n",
       " 'east hanover': 1869,\n",
       " 'highland heights': 2937,\n",
       " 'addison': 78,\n",
       " 'willow grove': 7399,\n",
       " 'stafford': 6439,\n",
       " 'mount airy': 4381,\n",
       " 'los altos': 3783,\n",
       " 'artesia': 285,\n",
       " 'maspeth': 3989,\n",
       " 'hoffman estates': 2987,\n",
       " 'tacoma': 6647,\n",
       " 'snoqualmie': 6202,\n",
       " 'temecula': 6694,\n",
       " 'oceanport': 4863,\n",
       " 'new richmond': 4601,\n",
       " 'watertown': 7126,\n",
       " 'lewisville': 3669,\n",
       " 'des moines': 1702,\n",
       " 'harbor': 2782,\n",
       " 'winston-salem': 7449,\n",
       " 'gastonia': 2485,\n",
       " 'girard': 2520,\n",
       " 'pittsfield': 5261,\n",
       " 'watkinsville': 7132,\n",
       " 'west windsor': 7279,\n",
       " 'grapevine': 2635,\n",
       " 'lebanon': 3619,\n",
       " 'los gatos': 3795,\n",
       " 'wallington': 7049,\n",
       " 'pine bluff': 5224,\n",
       " 'manchester': 3900,\n",
       " 'wayne': 7150,\n",
       " 'reading': 5497,\n",
       " 'quincy': 5453,\n",
       " 'midland': 4170,\n",
       " 'bell gardens': 524,\n",
       " 'coatesville': 1335,\n",
       " 'holly springs': 3005,\n",
       " 'ridge spring': 5591,\n",
       " 'hanover park': 2775,\n",
       " 'glassboro': 2527,\n",
       " 'hopkinton': 3040,\n",
       " 'buda': 900,\n",
       " 'north bend': 4705,\n",
       " 'alcorn state': 123,\n",
       " 'brookfield': 856,\n",
       " 'weston': 7310,\n",
       " 'round rock': 5736,\n",
       " 'woodstock': 7503,\n",
       " 'beverly hills': 612,\n",
       " 'saipan': 5825,\n",
       " 'uniontown': 6895,\n",
       " 'solon': 6216,\n",
       " 'center valley': 1125,\n",
       " 'lake mary': 3507,\n",
       " 'visalia': 7008,\n",
       " 'naugatuck': 4519,\n",
       " 'independence': 3126,\n",
       " 'joplin': 3266,\n",
       " 'fayetteville': 2198,\n",
       " 'gibbsboro': 2504,\n",
       " 'brea': 805,\n",
       " 'twinsburg': 6873,\n",
       " 'garden grove': 2464,\n",
       " 'avenel': 359,\n",
       " 'kalamazoo': 3286,\n",
       " 'worthington': 7519,\n",
       " 'rio grande city': 5608,\n",
       " 'little canada': 3719,\n",
       " 'glenside': 2559,\n",
       " 'athens': 316,\n",
       " 'platteville': 5290,\n",
       " 'thousand oaks': 6742,\n",
       " 'lubbock': 3817,\n",
       " 'seaford': 6039,\n",
       " 'hamilton': 2752,\n",
       " 'elkridge': 2002,\n",
       " '12605 e 16th avenue': 15,\n",
       " 'frederick': 2371,\n",
       " 'loveland': 3807,\n",
       " 'north bergen': 4706,\n",
       " 'largo': 3566,\n",
       " 'west long branch': 7243,\n",
       " 'napa': 4500,\n",
       " 'metairie': 4127,\n",
       " 'new albany': 4549,\n",
       " 'mckinney': 4051,\n",
       " 'greenbelt': 2659,\n",
       " 'costa mesa': 1468,\n",
       " 'greenwich': 2676,\n",
       " 'oak brook': 4824,\n",
       " 'emeryville': 2042,\n",
       " 'short hills': 6140,\n",
       " 'green bay': 2650,\n",
       " 'mossville': 4373,\n",
       " 'waukesha': 7139,\n",
       " 'east hartford': 1870,\n",
       " 'valley stream': 6939,\n",
       " 'norridge': 4693,\n",
       " 'sugarland': 6536,\n",
       " 'el paso': 1976,\n",
       " 'munster': 4452,\n",
       " 'davis junction': 1621,\n",
       " 'carmel': 1042,\n",
       " 'long beach': 3763,\n",
       " 'mayfield village': 4019,\n",
       " 'lutz': 3835,\n",
       " 'broomall': 876,\n",
       " 'buena park': 904,\n",
       " 'farmington': 2189,\n",
       " 'seminole': 6073,\n",
       " 'forestville': 2281,\n",
       " 'issaquah': 3185,\n",
       " 'anchorage': 209,\n",
       " 'jersey  city': 3232,\n",
       " 'roswell': 5729,\n",
       " 'pinetops': 5238,\n",
       " 'iriving': 3160,\n",
       " 'palmdale': 5042,\n",
       " 'ft. lauderdale': 2412,\n",
       " 'north reading': 4762,\n",
       " 'west henrietta': 7228,\n",
       " 'st paul': 6380,\n",
       " 'newburyport': 4638,\n",
       " 'armonk': 277,\n",
       " 'lakeland': 3522,\n",
       " 'floral park': 2243,\n",
       " 'oxford': 5003,\n",
       " 'johnston': 3256,\n",
       " 'lindenhurst': 3701,\n",
       " 'reynoldsburg': 5558,\n",
       " 'collierville': 1366,\n",
       " 'freeport': 2384,\n",
       " 'lauderdale by the sea': 3582,\n",
       " 'everett': 2111,\n",
       " 'coppel': 1426,\n",
       " 'springfield gardens': 6365,\n",
       " 'bear': 487,\n",
       " 'west hollywood': 7232,\n",
       " 'loves park': 3808,\n",
       " 'venice': 6968,\n",
       " 'spring house': 6358,\n",
       " 'edina': 1949,\n",
       " 'conway': 1409,\n",
       " 'clakamas': 1272,\n",
       " 'san luis': 5903,\n",
       " 'ontario': 4930,\n",
       " 'south san francisco': 6304,\n",
       " 'winslow': 7444,\n",
       " 'tecumseh': 6688,\n",
       " 'raritan': 5487,\n",
       " 'racine': 5457,\n",
       " 'carmichaels': 1045,\n",
       " 'warwick': 7093,\n",
       " 'hadley': 2728,\n",
       " 'midvale': 4174,\n",
       " 'geneva': 2495,\n",
       " 'perris': 5164,\n",
       " 'hancock': 2770,\n",
       " 'north hollywood': 4736,\n",
       " 'lenexa': 3646,\n",
       " 'la porte': 3450,\n",
       " 'cranford': 1503,\n",
       " 'lighthouse point': 3681,\n",
       " 'pompano beach': 5327,\n",
       " 'murray hill': 4460,\n",
       " 'buffalo grove': 907,\n",
       " 'piscatway': 5252,\n",
       " 'cibolo': 1254,\n",
       " 'waco': 7019,\n",
       " 'grass lake': 2636,\n",
       " 'ruth': 5766,\n",
       " 'fairbanks': 2132,\n",
       " 'euless': 2101,\n",
       " 'fargo': 2179,\n",
       " 'north liberty': 4743,\n",
       " 'bentonville': 572,\n",
       " 'davis': 1620,\n",
       " 'toms river': 6793,\n",
       " 'bagley': 383,\n",
       " 'mentor': 4102,\n",
       " 'palos hills': 5053,\n",
       " 'aloha': 152,\n",
       " 'tomball': 6791,\n",
       " 'amana': 185,\n",
       " \"st. mary's\": 6412,\n",
       " 'ypsilanti': 7564,\n",
       " 'islandia': 3178,\n",
       " 'indiantown': 3141,\n",
       " 'longview': 3773,\n",
       " 'rivedale': 5616,\n",
       " 'manassas city': 3894,\n",
       " 'paulden': 5114,\n",
       " 'galloway': 2454,\n",
       " 'florham park': 2247,\n",
       " 'marina del rey': 3954,\n",
       " 'riverdale': 5623,\n",
       " 'indiana': 3136,\n",
       " 'defuniak springs': 1655,\n",
       " 'bemidji': 557,\n",
       " 'killeen': 3373,\n",
       " 'western springs': 7291,\n",
       " 'slippery rock': 6192,\n",
       " 'san gabriel': 5889,\n",
       " 'glens falls': 2558,\n",
       " 'encinitas': 2052,\n",
       " 'charles town': 1171,\n",
       " 'dover': 1769,\n",
       " 'powell': 5397,\n",
       " 'farmers branch': 2183,\n",
       " 'kearney': 3307,\n",
       " 'concord': 1395,\n",
       " 'laguna niguel': 3480,\n",
       " 'des plaines': 1704,\n",
       " 'chambersburg': 1149,\n",
       " 'marietta': 3950,\n",
       " 'oakbrook terrace': 4837,\n",
       " 'rensselaer': 5542,\n",
       " 'kyle': 3429,\n",
       " 'pheonix': 5189,\n",
       " 'olivette': 4913,\n",
       " 'basalt': 439,\n",
       " 'hartford': 2809,\n",
       " 'gilbert': 2513,\n",
       " 'port washington': 5366,\n",
       " 'muscatine': 4465,\n",
       " 'peasanton': 5135,\n",
       " 'north las vegas': 4741,\n",
       " ...}"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "keys= list(dff_final.JOB_CITY)\n",
    "values =list(df_final.JOB_CITY)\n",
    "print(len(keys))\n",
    "print(len(values))\n",
    "\n",
    "Job_city_dict={}\n",
    "\n",
    "for i in range(len(keys)):\n",
    "    Job_city_dict[keys[i]] = values[i]\n",
    "Job_city_dict\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad742003",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d294f8e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16ca8f28",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "81ea2729",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(514964, 13)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = df_final.drop('CASE_STATUS', axis=1)\n",
    "y = df_final.CASE_STATUS\n",
    "df_final.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "13ea8a33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>JOB_CITY</th>\n",
       "      <th>JOB_STATE</th>\n",
       "      <th>REQD_EDUCATION</th>\n",
       "      <th>TRAINING_REQD</th>\n",
       "      <th>CITIZENSHIP</th>\n",
       "      <th>ADMISSION_TYPE</th>\n",
       "      <th>WORKER_EDUCATION</th>\n",
       "      <th>OCCUPATION</th>\n",
       "      <th>NEW_RELATED_MAJOR</th>\n",
       "      <th>NEW_WORKER_MAJOR</th>\n",
       "      <th>NEW_EMPLOYER_NAME</th>\n",
       "      <th>WAGE_OFFERED</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4921</td>\n",
       "      <td>35</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3173</td>\n",
       "      <td>37</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4405</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>165</td>\n",
       "      <td>29</td>\n",
       "      <td>4</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2264</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>14</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4056</td>\n",
       "      <td>52</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535053</th>\n",
       "      <td>6073</td>\n",
       "      <td>50</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>34</td>\n",
       "      <td>18</td>\n",
       "      <td>5</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535054</th>\n",
       "      <td>4615</td>\n",
       "      <td>40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535055</th>\n",
       "      <td>5955</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535063</th>\n",
       "      <td>3165</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535083</th>\n",
       "      <td>5280</td>\n",
       "      <td>50</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>514964 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        JOB_CITY  JOB_STATE  REQD_EDUCATION  TRAINING_REQD  CITIZENSHIP  \\\n",
       "0           4921         35               4              0           80   \n",
       "1           3173         37               4              0           80   \n",
       "2           4405          4               4              0          165   \n",
       "3           2264          4               4              0           14   \n",
       "4           4056         52               1              0           80   \n",
       "...          ...        ...             ...            ...          ...   \n",
       "535053      6073         50               5              0           34   \n",
       "535054      4615         40               1              0           80   \n",
       "535055      5955          4               4              0           80   \n",
       "535063      3165          4               1              0           80   \n",
       "535083      5280         50               4              0           80   \n",
       "\n",
       "        ADMISSION_TYPE  WORKER_EDUCATION  OCCUPATION  NEW_RELATED_MAJOR  \\\n",
       "0                   18                 1          12                  1   \n",
       "1                   18                 4          12                  1   \n",
       "2                   29                 4          12                  1   \n",
       "3                   18                 4           1                  1   \n",
       "4                   18                 4          12                  1   \n",
       "...                ...               ...         ...                ...   \n",
       "535053              18                 5          11                  1   \n",
       "535054              18                 1          12                  0   \n",
       "535055              18                 4           1                  0   \n",
       "535063              18                 1          12                  1   \n",
       "535083              18                 1          12                  1   \n",
       "\n",
       "        NEW_WORKER_MAJOR  NEW_EMPLOYER_NAME  WAGE_OFFERED  \n",
       "0                      1                  0             5  \n",
       "1                      0                  0             5  \n",
       "2                      1                  0             5  \n",
       "3                      1                  2             5  \n",
       "4                      1                  0             5  \n",
       "...                  ...                ...           ...  \n",
       "535053                 1                  0             1  \n",
       "535054                 1                  0             5  \n",
       "535055                 1                  2             3  \n",
       "535063                 1                  0             2  \n",
       "535083                 1                  0             5  \n",
       "\n",
       "[514964 rows x 12 columns]"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "9ab0e512",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(360474, 12) (360474,)\n",
      "(154490, 12) (154490,)\n"
     ]
    }
   ],
   "source": [
    "seed = 10\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)\n",
    "print(x_train.shape,y_train.shape)\n",
    "print(x_test.shape,y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "5b31e501",
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import SMOTE\n",
    "sm = SMOTE(random_state=12, sampling_strategy=1)\n",
    "x_train_res, y_train_res = sm.fit_resample(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "abc27501",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>JOB_CITY</th>\n",
       "      <th>JOB_STATE</th>\n",
       "      <th>REQD_EDUCATION</th>\n",
       "      <th>TRAINING_REQD</th>\n",
       "      <th>CITIZENSHIP</th>\n",
       "      <th>ADMISSION_TYPE</th>\n",
       "      <th>WORKER_EDUCATION</th>\n",
       "      <th>OCCUPATION</th>\n",
       "      <th>NEW_RELATED_MAJOR</th>\n",
       "      <th>NEW_WORKER_MAJOR</th>\n",
       "      <th>NEW_EMPLOYER_NAME</th>\n",
       "      <th>WAGE_OFFERED</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>4921</td>\n",
       "      <td>35</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>3173</td>\n",
       "      <td>37</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>4405</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>165</td>\n",
       "      <td>29</td>\n",
       "      <td>4</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>2264</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>14</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>4056</td>\n",
       "      <td>52</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535053</th>\n",
       "      <td>1</td>\n",
       "      <td>6073</td>\n",
       "      <td>50</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>34</td>\n",
       "      <td>18</td>\n",
       "      <td>5</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535054</th>\n",
       "      <td>1</td>\n",
       "      <td>4615</td>\n",
       "      <td>40</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535055</th>\n",
       "      <td>1</td>\n",
       "      <td>5955</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535063</th>\n",
       "      <td>0</td>\n",
       "      <td>3165</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>535083</th>\n",
       "      <td>0</td>\n",
       "      <td>5280</td>\n",
       "      <td>50</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>80</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>514964 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        CASE_STATUS  JOB_CITY  JOB_STATE  REQD_EDUCATION  TRAINING_REQD  \\\n",
       "0                 1      4921         35               4              0   \n",
       "1                 1      3173         37               4              0   \n",
       "2                 1      4405          4               4              0   \n",
       "3                 1      2264          4               4              0   \n",
       "4                 1      4056         52               1              0   \n",
       "...             ...       ...        ...             ...            ...   \n",
       "535053            1      6073         50               5              0   \n",
       "535054            1      4615         40               1              0   \n",
       "535055            1      5955          4               4              0   \n",
       "535063            0      3165          4               1              0   \n",
       "535083            0      5280         50               4              0   \n",
       "\n",
       "        CITIZENSHIP  ADMISSION_TYPE  WORKER_EDUCATION  OCCUPATION  \\\n",
       "0                80              18                 1          12   \n",
       "1                80              18                 4          12   \n",
       "2               165              29                 4          12   \n",
       "3                14              18                 4           1   \n",
       "4                80              18                 4          12   \n",
       "...             ...             ...               ...         ...   \n",
       "535053           34              18                 5          11   \n",
       "535054           80              18                 1          12   \n",
       "535055           80              18                 4           1   \n",
       "535063           80              18                 1          12   \n",
       "535083           80              18                 1          12   \n",
       "\n",
       "        NEW_RELATED_MAJOR  NEW_WORKER_MAJOR  NEW_EMPLOYER_NAME  WAGE_OFFERED  \n",
       "0                       1                 1                  0             5  \n",
       "1                       1                 0                  0             5  \n",
       "2                       1                 1                  0             5  \n",
       "3                       1                 1                  2             5  \n",
       "4                       1                 1                  0             5  \n",
       "...                   ...               ...                ...           ...  \n",
       "535053                  1                 1                  0             1  \n",
       "535054                  0                 1                  0             5  \n",
       "535055                  0                 1                  2             3  \n",
       "535063                  1                 1                  0             2  \n",
       "535083                  1                 1                  0             5  \n",
       "\n",
       "[514964 rows x 13 columns]"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_final"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "5a3ffa00",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{1: 1, 0: 0},\n",
       " {'omaha': 4921,\n",
       "  'iselin': 3173,\n",
       "  'mountain view': 4405,\n",
       "  'folsom': 2264,\n",
       "  'mclean': 4056,\n",
       "  'san francisco': 5874,\n",
       "  'porterville': 5374,\n",
       "  'san diego': 5863,\n",
       "  'falls church': 2168,\n",
       "  'jersey city': 3234,\n",
       "  'new york': 4615,\n",
       "  'hillsboro': 2955,\n",
       "  'santa maria': 5965,\n",
       "  'salt lake city': 5837,\n",
       "  'savannah': 5996,\n",
       "  'fort wayne': 2325,\n",
       "  'west conshohocken': 7209,\n",
       "  'milpitas': 4219,\n",
       "  'east hampton': 1868,\n",
       "  'san jose': 5894,\n",
       "  'muncie': 4449,\n",
       "  'northville': 4796,\n",
       "  'cambridge': 985,\n",
       "  'east moline': 1886,\n",
       "  'fort lauderdale': 2298,\n",
       "  'rosemead': 5707,\n",
       "  'southborough': 6317,\n",
       "  'college station': 1362,\n",
       "  'pittsburgh': 5258,\n",
       "  'bothell': 755,\n",
       "  'philadelphia': 5191,\n",
       "  'las vegas': 3574,\n",
       "  'cranbury': 1500,\n",
       "  'oklahoma city': 4888,\n",
       "  'piscataway': 5247,\n",
       "  'north wales': 4776,\n",
       "  'elkhart': 1998,\n",
       "  'iowa city': 3157,\n",
       "  'hanover': 2774,\n",
       "  'houston': 3063,\n",
       "  'westborough': 7286,\n",
       "  'birmingham': 644,\n",
       "  'danbury': 1602,\n",
       "  'washington': 7099,\n",
       "  'southlake': 6324,\n",
       "  'cleveland': 1310,\n",
       "  'montgomery': 4313,\n",
       "  'gaithersburg': 2442,\n",
       "  'apopka': 243,\n",
       "  'gallina': 2452,\n",
       "  'tumon': 6854,\n",
       "  'naperville': 4501,\n",
       "  'carol stream': 1050,\n",
       "  'schaumburg': 6008,\n",
       "  'meriden': 4112,\n",
       "  'fremont': 2387,\n",
       "  'annandale': 224,\n",
       "  'charlotte': 1176,\n",
       "  'fort collins': 2293,\n",
       "  'verona': 6979,\n",
       "  'denver': 1690,\n",
       "  'atlanta': 320,\n",
       "  'monmouth jct': 4286,\n",
       "  'sheffield': 6111,\n",
       "  'brooklyn': 862,\n",
       "  'austin': 350,\n",
       "  'san ramon': 5917,\n",
       "  'columbus': 1383,\n",
       "  'joliet': 3259,\n",
       "  'colorado springs': 1377,\n",
       "  'miami': 4135,\n",
       "  'chicago': 1225,\n",
       "  'troy': 6837,\n",
       "  'teaneck': 6686,\n",
       "  'cedar park': 1110,\n",
       "  'boulder': 758,\n",
       "  'monmouth junction': 4288,\n",
       "  'coral gables': 1432,\n",
       "  'tampa': 6662,\n",
       "  'dayton': 1625,\n",
       "  'norman': 4692,\n",
       "  'ellicott city': 2011,\n",
       "  'jamaica': 3204,\n",
       "  'winona': 7442,\n",
       "  'herndon': 2904,\n",
       "  'chantilly': 1160,\n",
       "  'dallas': 1592,\n",
       "  'titusville': 6774,\n",
       "  'shiprock': 6132,\n",
       "  'edison': 1952,\n",
       "  'san bernardino': 5858,\n",
       "  'sunnyvale': 6603,\n",
       "  'louisville': 3805,\n",
       "  'durham': 1820,\n",
       "  'burke': 922,\n",
       "  'crownpoint': 1544,\n",
       "  'bellevue': 539,\n",
       "  'san antonio': 5852,\n",
       "  'fairfax': 2137,\n",
       "  'little rock': 3726,\n",
       "  'castle rock': 1084,\n",
       "  'norwood': 4807,\n",
       "  'novato': 4812,\n",
       "  'boston': 751,\n",
       "  'katy': 3301,\n",
       "  'saginaw': 5799,\n",
       "  'boxborough': 769,\n",
       "  'doral': 1757,\n",
       "  'union city': 6890,\n",
       "  'rochester': 5643,\n",
       "  'pleasant prairie': 5299,\n",
       "  'aguadilla': 92,\n",
       "  'westchester': 7290,\n",
       "  'columbia': 1380,\n",
       "  'rosemont': 5709,\n",
       "  'canoga park': 1006,\n",
       "  'wainscott': 7031,\n",
       "  'fort lee': 2299,\n",
       "  'aliso viejo': 140,\n",
       "  'nashua': 4507,\n",
       "  'petersburg': 5179,\n",
       "  'apex': 242,\n",
       "  'san pedro': 5915,\n",
       "  'culver city': 1562,\n",
       "  'plano': 5280,\n",
       "  'manteca': 3923,\n",
       "  'escalon': 2080,\n",
       "  'irving': 3166,\n",
       "  'santa clara': 5955,\n",
       "  'hauppauge': 2833,\n",
       "  'los angeles': 3785,\n",
       "  'sterling': 6469,\n",
       "  'germantown': 2499,\n",
       "  'portland': 5375,\n",
       "  'bethesda': 603,\n",
       "  'seattle': 6052,\n",
       "  'menlo park': 4098,\n",
       "  'somers': 6219,\n",
       "  'greenwood village': 2678,\n",
       "  'provo': 5429,\n",
       "  'lauderhill': 3585,\n",
       "  'north brunswick': 4712,\n",
       "  'tallahassee': 6656,\n",
       "  'framingham': 2351,\n",
       "  'orangeburg': 4943,\n",
       "  'tempe': 6696,\n",
       "  'worcester': 7515,\n",
       "  'arlington heights': 273,\n",
       "  'el dorado hills': 1974,\n",
       "  'palo alto': 5049,\n",
       "  'winesburg': 7431,\n",
       "  'milwaukee': 4222,\n",
       "  'minnetonka': 4237,\n",
       "  'harbor beach': 2783,\n",
       "  'springfield': 6364,\n",
       "  'beltsville': 553,\n",
       "  'south windsor': 6313,\n",
       "  'symmes township': 6641,\n",
       "  'stamford': 6441,\n",
       "  'redmond': 5519,\n",
       "  'melville': 4088,\n",
       "  'peoria': 5161,\n",
       "  'palm beach gardens': 5036,\n",
       "  'high point': 2931,\n",
       "  'south burlington': 6246,\n",
       "  'rockville': 5664,\n",
       "  'saint louis': 5815,\n",
       "  'redwood shores': 5527,\n",
       "  'kirkland': 3403,\n",
       "  'baltimore': 403,\n",
       "  'flourtown': 2253,\n",
       "  'baton rouge': 455,\n",
       "  'acton': 71,\n",
       "  'alpharetta': 158,\n",
       "  'pasadena': 5099,\n",
       "  'la mirada': 3447,\n",
       "  'torrance': 6802,\n",
       "  'stillwater': 6482,\n",
       "  'norcross': 4684,\n",
       "  'glastonbury': 2529,\n",
       "  'hermosa beach': 2901,\n",
       "  'monee': 4282,\n",
       "  'ashburn': 291,\n",
       "  'west point': 7262,\n",
       "  'juno beach': 3275,\n",
       "  'rowland heights': 5740,\n",
       "  'moon township': 4324,\n",
       "  'deerfield': 1651,\n",
       "  'vernon': 6975,\n",
       "  'newark': 4628,\n",
       "  'centreville': 1138,\n",
       "  'cedar rapids': 1111,\n",
       "  'duluth': 1798,\n",
       "  'littleton': 3732,\n",
       "  'lake forest': 3497,\n",
       "  'lorton': 3780,\n",
       "  'moonachie': 4329,\n",
       "  'fairview': 2150,\n",
       "  'beaverton': 499,\n",
       "  'enola': 2068,\n",
       "  'palo alo': 5048,\n",
       "  'darien': 1611,\n",
       "  'richardson': 5569,\n",
       "  'madison': 3864,\n",
       "  'waterford': 7122,\n",
       "  'ann arbor': 221,\n",
       "  'pleasanton': 5301,\n",
       "  'clovis': 1328,\n",
       "  'moline': 4275,\n",
       "  'north bay village': 4703,\n",
       "  'orange': 4938,\n",
       "  'jupiter': 3276,\n",
       "  'southfield': 6320,\n",
       "  'morristown': 4359,\n",
       "  'oakland': 4842,\n",
       "  'libertyville': 3678,\n",
       "  'fresno': 2395,\n",
       "  'la vista': 3459,\n",
       "  'avon': 363,\n",
       "  'bala cynwyd': 391,\n",
       "  'reston': 5552,\n",
       "  'albany': 112,\n",
       "  'northridge': 4793,\n",
       "  'belleville': 538,\n",
       "  'ramsey': 5467,\n",
       "  'burlington': 925,\n",
       "  'farmington hills': 2191,\n",
       "  'ossining': 4980,\n",
       "  'encino': 2053,\n",
       "  'itasca': 3186,\n",
       "  'siloam springs': 6154,\n",
       "  'jacksonville': 3200,\n",
       "  'richmond': 5580,\n",
       "  'mettawa': 4130,\n",
       "  'bronx': 853,\n",
       "  'glen allen': 2530,\n",
       "  'wilmington': 7409,\n",
       "  'charleston': 1172,\n",
       "  'foster city': 2339,\n",
       "  'gardena': 2465,\n",
       "  'newton': 4650,\n",
       "  'king of prussia': 3381,\n",
       "  'carpentersville': 1053,\n",
       "  'piedmont': 5210,\n",
       "  'buffalo': 906,\n",
       "  'lexington': 3670,\n",
       "  'radnor': 5459,\n",
       "  'lantana': 3558,\n",
       "  'westminster': 7305,\n",
       "  'carrollton': 1061,\n",
       "  'san mateo': 5911,\n",
       "  'campbell': 994,\n",
       "  'englewood': 2057,\n",
       "  'glendale': 2544,\n",
       "  'arcadia': 255,\n",
       "  'watsonville': 7134,\n",
       "  'broomfield': 877,\n",
       "  'waxhaw': 7147,\n",
       "  'raleigh': 5463,\n",
       "  'east brunswick': 1852,\n",
       "  'cherry hill': 1202,\n",
       "  'irvine': 3165,\n",
       "  'st. martinville': 6411,\n",
       "  'sugar land': 6533,\n",
       "  'douglas': 1766,\n",
       "  'carson': 1065,\n",
       "  'rancho palos verdes': 5477,\n",
       "  'clearwater': 1304,\n",
       "  'brooklyn park': 867,\n",
       "  'signal hill': 6150,\n",
       "  'seaside': 6047,\n",
       "  'san bruno': 5859,\n",
       "  'braintree': 787,\n",
       "  'el segundo': 1979,\n",
       "  'manassas': 3893,\n",
       "  'alhambra': 137,\n",
       "  'mendota heights': 4096,\n",
       "  'east wareham': 1917,\n",
       "  'fullerton': 2422,\n",
       "  'brisbane': 841,\n",
       "  'warren': 7080,\n",
       "  'whittier': 7361,\n",
       "  'lehi': 3632,\n",
       "  'south plainfield': 6289,\n",
       "  'san carlos': 5860,\n",
       "  'carrington': 1056,\n",
       "  'hagerstown': 2732,\n",
       "  'duncan': 1804,\n",
       "  'newport beach': 4646,\n",
       "  'bensenville': 565,\n",
       "  'suite: 202': 6571,\n",
       "  'st. clair shores': 6392,\n",
       "  'york': 7556,\n",
       "  'st. paul': 6416,\n",
       "  'minot': 4241,\n",
       "  'chicopee': 1234,\n",
       "  'van nuys': 6953,\n",
       "  'brentwood': 813,\n",
       "  'elmhurst': 2020,\n",
       "  'somerset': 6222,\n",
       "  'syracuse': 6644,\n",
       "  'amelia': 189,\n",
       "  'pennsauken': 5158,\n",
       "  'escondido': 2081,\n",
       "  'keasbey': 3310,\n",
       "  'shattuck': 6103,\n",
       "  'huntersville': 3087,\n",
       "  'rocklin': 5661,\n",
       "  'north kansas city': 4739,\n",
       "  'chelmsford': 1193,\n",
       "  'cary': 1072,\n",
       "  'east windsor': 1920,\n",
       "  'fort myers': 2311,\n",
       "  'forest hills': 2277,\n",
       "  'pawcatuck': 5118,\n",
       "  'wichita': 7362,\n",
       "  'annapolis': 227,\n",
       "  'shreveport': 6142,\n",
       "  'sturgeon bay': 6521,\n",
       "  'cupertino': 1570,\n",
       "  'norristown': 4694,\n",
       "  'wakefield': 7035,\n",
       "  'delray beach': 1678,\n",
       "  'leawood': 3618,\n",
       "  'detroit': 1713,\n",
       "  'owings mills': 5001,\n",
       "  'waller': 7047,\n",
       "  'tamuning': 6664,\n",
       "  'morgantown': 4349,\n",
       "  'lawrence': 3600,\n",
       "  'bloomfield': 675,\n",
       "  'morganton': 4348,\n",
       "  'pinon': 5242,\n",
       "  'bethlehem': 605,\n",
       "  'wexford': 7320,\n",
       "  'richardson,': 5570,\n",
       "  'college point': 1361,\n",
       "  'plantation': 5282,\n",
       "  'brevard': 815,\n",
       "  'huntington': 3090,\n",
       "  'mechanicsburg': 4073,\n",
       "  'livonia': 3736,\n",
       "  'delano': 1667,\n",
       "  'kealakekua': 3305,\n",
       "  'latrobe': 3581,\n",
       "  'redwood city': 5523,\n",
       "  'livingston': 3735,\n",
       "  'minneapolis': 4234,\n",
       "  'chicago heights': 1226,\n",
       "  'lambertville': 3534,\n",
       "  'scottsdale': 6029,\n",
       "  'pecos': 5137,\n",
       "  'lagrange': 3476,\n",
       "  'indianapolis': 3137,\n",
       "  'port wentworth': 5367,\n",
       "  'georgetown': 2496,\n",
       "  'espanola': 2083,\n",
       "  'panama city': 5061,\n",
       "  'wellesley': 7170,\n",
       "  'pharr': 5187,\n",
       "  'hodgkins': 2985,\n",
       "  'ny': 4817,\n",
       "  'marlborough': 3961,\n",
       "  'dearborn': 1635,\n",
       "  'san clemente': 5861,\n",
       "  'malta': 3884,\n",
       "  'chesapeake': 1206,\n",
       "  'alcoa center': 122,\n",
       "  'solana beach': 6214,\n",
       "  'lawrenceville': 3604,\n",
       "  'long island city': 3768,\n",
       "  'fowlerville': 2347,\n",
       "  'greenville': 2674,\n",
       "  'franklin': 2358,\n",
       "  'lanham': 3553,\n",
       "  'camarillo': 980,\n",
       "  'draper': 1780,\n",
       "  'blue bell': 689,\n",
       "  'west palm beach': 7258,\n",
       "  'kingwood': 3397,\n",
       "  'valencia': 6929,\n",
       "  \"downer's grove\": 1773,\n",
       "  'rockleigh': 5660,\n",
       "  'lincolnshire': 3694,\n",
       "  'erlanger': 2078,\n",
       "  'santa monica': 5966,\n",
       "  'woodbridge': 7480,\n",
       "  'new bern': 4553,\n",
       "  'alexandria': 131,\n",
       "  'virginia beach': 7005,\n",
       "  'topeka': 6799,\n",
       "  'johns creek': 3249,\n",
       "  'north franklin': 4731,\n",
       "  'greensboro': 2671,\n",
       "  'flushing': 2259,\n",
       "  'sayreville': 6001,\n",
       "  'gowanda': 2592,\n",
       "  'highland hills': 2938,\n",
       "  'ellisville': 2013,\n",
       "  'dulles': 1797,\n",
       "  'sugar mountain': 6534,\n",
       "  'frisco': 2400,\n",
       "  'south riding': 6299,\n",
       "  'big stone gap': 625,\n",
       "  'spartanburg': 6336,\n",
       "  'eden prairie': 1940,\n",
       "  'skokie': 6183,\n",
       "  'kensington': 3338,\n",
       "  'evanston': 2109,\n",
       "  'fairfield': 2140,\n",
       "  'west valley': 7275,\n",
       "  'foothill ranch': 2268,\n",
       "  'plainfield': 5272,\n",
       "  'tucson': 6847,\n",
       "  'knoxville': 3418,\n",
       "  'langhorne': 3551,\n",
       "  'henderson': 2887,\n",
       "  'yuma': 7569,\n",
       "  'orem': 4954,\n",
       "  'fredericksburg': 2374,\n",
       "  'fridley': 2397,\n",
       "  'coppell': 1427,\n",
       "  'sidney': 6147,\n",
       "  'canonsburg': 1008,\n",
       "  'del mar': 1659,\n",
       "  'phoenix': 5201,\n",
       "  'woonsocket': 7513,\n",
       "  'secaucus': 6057,\n",
       "  'malvern': 3885,\n",
       "  'converse': 1408,\n",
       "  'moorestown': 4334,\n",
       "  'statesboro': 6455,\n",
       "  'bloomington': 681,\n",
       "  'richfield': 5574,\n",
       "  'rancho cordova': 5470,\n",
       "  'anaheim': 205,\n",
       "  'paramus': 5072,\n",
       "  'fond du lac': 2266,\n",
       "  'mcallen': 4030,\n",
       "  'progreso': 5419,\n",
       "  'huntington beach': 3091,\n",
       "  'poplar bluff': 5340,\n",
       "  'shrewsbury': 6144,\n",
       "  'honolulu': 3024,\n",
       "  'vienna': 6990,\n",
       "  'durant': 1819,\n",
       "  'rancho dominguez': 5474,\n",
       "  'albuquerque': 118,\n",
       "  'riverside': 5626,\n",
       "  'gillette': 2517,\n",
       "  'east lansing': 1877,\n",
       "  'metuchen': 4132,\n",
       "  'auburn hills': 338,\n",
       "  'middletown': 4165,\n",
       "  'claxton': 1292,\n",
       "  'parlin': 5091,\n",
       "  'channelview': 1159,\n",
       "  'chandler': 1155,\n",
       "  'modesto': 4267,\n",
       "  'englewood cliffs': 2060,\n",
       "  'montvale': 4320,\n",
       "  'belmont': 549,\n",
       "  'silver spring': 6158,\n",
       "  'newtown': 4655,\n",
       "  'novi': 4814,\n",
       "  'burbank': 917,\n",
       "  'west chester': 7205,\n",
       "  'woodcliff lake': 7483,\n",
       "  'allentown': 147,\n",
       "  'research triangle park': 5549,\n",
       "  'lake worth': 3519,\n",
       "  'morrisville': 4360,\n",
       "  'newtown square': 4656,\n",
       "  'basking ridge': 443,\n",
       "  'ada': 73,\n",
       "  'danville': 1607,\n",
       "  'longmont': 3772,\n",
       "  'hidalgo': 2924,\n",
       "  'catonsville': 1094,\n",
       "  'arlington': 272,\n",
       "  'hudson': 3075,\n",
       "  'aurora': 346,\n",
       "  'natick': 4514,\n",
       "  'norfolk': 4688,\n",
       "  'cheshire': 1209,\n",
       "  'jasper': 3214,\n",
       "  'princeton': 5416,\n",
       "  'parsippany': 5096,\n",
       "  'eunice': 2102,\n",
       "  'champaign': 1151,\n",
       "  'beeville': 510,\n",
       "  'lansing': 3557,\n",
       "  'cusseta': 1574,\n",
       "  'jamesburg': 3208,\n",
       "  'university park': 6905,\n",
       "  'west columbia': 7208,\n",
       "  'coquille': 1431,\n",
       "  'suwanee': 6618,\n",
       "  'west warwick': 7278,\n",
       "  'white plains': 7340,\n",
       "  'orlando': 4959,\n",
       "  'merrillville': 4119,\n",
       "  'glenwood springs': 2564,\n",
       "  'downers grove': 1774,\n",
       "  'livermore': 3733,\n",
       "  'wilton': 7414,\n",
       "  'park ridge': 5079,\n",
       "  'olathe': 4891,\n",
       "  'kent': 3339,\n",
       "  'algona': 135,\n",
       "  'norwalk': 4802,\n",
       "  'gainesville': 2436,\n",
       "  'north chicago': 4719,\n",
       "  'pineville': 5240,\n",
       "  'milford': 4185,\n",
       "  'manchester city': 3902,\n",
       "  'pennington': 5156,\n",
       "  'glendora': 2547,\n",
       "  'evansville': 2110,\n",
       "  'nashville': 4508,\n",
       "  'kingsport': 3393,\n",
       "  'st. petersburg': 6424,\n",
       "  'marysville': 3983,\n",
       "  'edsion': 1958,\n",
       "  'clinton': 1320,\n",
       "  'fresh meadows': 2394,\n",
       "  'malibu': 3882,\n",
       "  'ionia': 3155,\n",
       "  'battle creek': 457,\n",
       "  'guilford': 2708,\n",
       "  'decatur': 1638,\n",
       "  'pomona': 5325,\n",
       "  'akron': 101,\n",
       "  'st. charles': 6391,\n",
       "  'covina': 1491,\n",
       "  'northfield': 4785,\n",
       "  'south brunswick': 6245,\n",
       "  'santa ana': 5951,\n",
       "  'la palma': 3448,\n",
       "  'miami lakes': 4140,\n",
       "  'dublin': 1787,\n",
       "  'lodi': 3746,\n",
       "  'hialeah': 2915,\n",
       "  'doraville': 1760,\n",
       "  'walnut creek': 7054,\n",
       "  'rye': 5772,\n",
       "  'pullman': 5436,\n",
       "  'bridgeport': 829,\n",
       "  'lacey': 3461,\n",
       "  'federal way': 2202,\n",
       "  'stanton': 6445,\n",
       "  '800 n. state college blvd.': 49,\n",
       "  'sacramento': 5789,\n",
       "  'cullowhee': 1560,\n",
       "  'danvers': 1606,\n",
       "  'hopewell junction': 3035,\n",
       "  'naples': 4502,\n",
       "  'rancho cucamonga': 5471,\n",
       "  'berkeley': 576,\n",
       "  'medley': 4081,\n",
       "  'kahului': 3281,\n",
       "  'elizabeth': 1990,\n",
       "  'ballston lake': 399,\n",
       "  'woburn': 7469,\n",
       "  'carlsbad': 1038,\n",
       "  'saint rose': 5823,\n",
       "  'needham': 4530,\n",
       "  'north providence': 4759,\n",
       "  'north hills': 4735,\n",
       "  'johnstown': 3257,\n",
       "  'fountain valley': 2345,\n",
       "  'cape girardeau': 1021,\n",
       "  'the woodlands': 6723,\n",
       "  'mccook': 4038,\n",
       "  'santa clarita': 5957,\n",
       "  'boise': 719,\n",
       "  'tustin': 6867,\n",
       "  'sunrise': 6606,\n",
       "  'lombard': 3754,\n",
       "  'mansfield': 3922,\n",
       "  'westport': 7313,\n",
       "  'chaddsford': 1144,\n",
       "  'east greenbush': 1865,\n",
       "  'bushnell': 939,\n",
       "  'fontana': 2267,\n",
       "  'west hartford': 7223,\n",
       "  'burlingame': 924,\n",
       "  'normal': 4691,\n",
       "  'sleepy hollow': 6190,\n",
       "  'san juan capistrano': 5898,\n",
       "  'towson': 6816,\n",
       "  'hunt valley': 3086,\n",
       "  'clive': 1324,\n",
       "  'rutland': 5771,\n",
       "  'holmdel': 3008,\n",
       "  'rolling meadows': 5684,\n",
       "  'smithfield': 6194,\n",
       "  'bartlett': 436,\n",
       "  'la crosse': 3437,\n",
       "  'laplata': 3561,\n",
       "  'providence': 5425,\n",
       "  'caribou': 1032,\n",
       "  'college park': 1359,\n",
       "  'bedford': 502,\n",
       "  'port charlotte': 5344,\n",
       "  'cabin john': 949,\n",
       "  'la jolla': 3443,\n",
       "  'prescott': 5405,\n",
       "  'kenner': 3330,\n",
       "  'sherman': 6127,\n",
       "  'fishers': 2224,\n",
       "  'santa barbara': 5953,\n",
       "  'st. louis': 6409,\n",
       "  'clawson': 1291,\n",
       "  'edinburg': 1951,\n",
       "  'monterey': 4307,\n",
       "  'grand rapids': 2616,\n",
       "  'whitestown': 7351,\n",
       "  'exton': 2125,\n",
       "  'lewis center': 3662,\n",
       "  'hattiesburg': 2829,\n",
       "  'hayward': 2857,\n",
       "  'robbinsville': 5636,\n",
       "  'florence': 2245,\n",
       "  'duarte': 1786,\n",
       "  'wallingford': 7048,\n",
       "  'whitsett': 7359,\n",
       "  'elk grove village': 1995,\n",
       "  'wayzata': 7154,\n",
       "  'griffin': 2686,\n",
       "  'walnut': 7052,\n",
       "  'calabasas': 957,\n",
       "  'memphis': 4091,\n",
       "  'randolph': 5481,\n",
       "  'tuscaloosa': 6861,\n",
       "  'west covina': 7210,\n",
       "  'maite': 3878,\n",
       "  'northbrook': 4784,\n",
       "  'boca raton': 710,\n",
       "  'hackensack': 2726,\n",
       "  'waltham': 7060,\n",
       "  'melbourne': 4084,\n",
       "  'meadville': 4069,\n",
       "  'overland park': 4992,\n",
       "  'hollywood': 3006,\n",
       "  'fort worth': 2328,\n",
       "  'riverwoods': 5630,\n",
       "  'shakopee': 6096,\n",
       "  'mason': 3986,\n",
       "  'westlake village': 7302,\n",
       "  'monroeville': 4295,\n",
       "  'west des moines': 7213,\n",
       "  'chino hills': 1242,\n",
       "  'brainerd': 786,\n",
       "  'vernon hills': 6976,\n",
       "  'goshen': 2589,\n",
       "  'chino': 1240,\n",
       "  'storrs': 6501,\n",
       "  'gallup': 2455,\n",
       "  'kansas city': 3295,\n",
       "  'dunn loring': 1815,\n",
       "  'salem': 5826,\n",
       "  'north charleston': 4716,\n",
       "  'riviera beach': 5631,\n",
       "  'baytown': 482,\n",
       "  'chesterfield': 1213,\n",
       "  'neenah': 4534,\n",
       "  'miami beach': 4136,\n",
       "  'elyria': 2040,\n",
       "  'millersville': 4204,\n",
       "  'bridgewater': 833,\n",
       "  'pittsburg': 5257,\n",
       "  'manalapan': 3891,\n",
       "  'greeley': 2649,\n",
       "  'ridge': 5590,\n",
       "  'wise': 7461,\n",
       "  'tonawanda': 6794,\n",
       "  'niles': 4667,\n",
       "  'brookings': 858,\n",
       "  'alameda': 104,\n",
       "  'matawan': 3997,\n",
       "  'coraopolis': 1438,\n",
       "  'cincinnati': 1257,\n",
       "  'mississippi state': 4253,\n",
       "  'corona': 1452,\n",
       "  'golden': 2572,\n",
       "  'st. thomas': 6434,\n",
       "  'charlottesville': 1179,\n",
       "  'tulsa': 6853,\n",
       "  'ruston': 5765,\n",
       "  'windsor': 7426,\n",
       "  'chesterbrook': 1212,\n",
       "  'mandeville': 3905,\n",
       "  'saint joseph': 5813,\n",
       "  'chattanooga': 1187,\n",
       "  'spring': 6351,\n",
       "  'frostburg': 2402,\n",
       "  'redlands': 5516,\n",
       "  'malone': 3883,\n",
       "  'murfreesboro': 4455,\n",
       "  'medway': 4082,\n",
       "  'east peoria': 1896,\n",
       "  'center': 1120,\n",
       "  'lake city': 3492,\n",
       "  'new kensington': 4586,\n",
       "  'city of industry': 1269,\n",
       "  'blue island': 691,\n",
       "  'burr ridge': 935,\n",
       "  'elkton': 2003,\n",
       "  'dorchester': 1761,\n",
       "  'portsmouth': 5378,\n",
       "  'eagan': 1835,\n",
       "  'irwindale': 3171,\n",
       "  'san francsico': 5882,\n",
       "  'plainsboro': 5274,\n",
       "  'rose hill': 5698,\n",
       "  'monrovia': 4296,\n",
       "  'highland village': 2944,\n",
       "  'bedminster': 507,\n",
       "  \"o'fallon\": 4822,\n",
       "  'staten island': 6454,\n",
       "  'hermiston': 2899,\n",
       "  'lyndhurst': 3841,\n",
       "  'spring hill': 6356,\n",
       "  'kapolei': 3297,\n",
       "  'fort smith': 2317,\n",
       "  'hilliard': 2953,\n",
       "  'keithville': 3314,\n",
       "  'mission': 4249,\n",
       "  'lincoln': 3689,\n",
       "  'west valley city': 7276,\n",
       "  'wixom': 7466,\n",
       "  'san dimas': 5867,\n",
       "  'miami gardens': 4138,\n",
       "  'odessa': 4871,\n",
       "  'ronkonkoma': 5694,\n",
       "  'temple terrace': 6702,\n",
       "  'noblesville': 4674,\n",
       "  'garfield': 2469,\n",
       "  'farmville': 2195,\n",
       "  'flemington': 2238,\n",
       "  'methuen': 4128,\n",
       "  'goleta': 2578,\n",
       "  'plymouth': 5309,\n",
       "  'poughkeepsie': 5389,\n",
       "  'smithtown': 6195,\n",
       "  'tyler': 6877,\n",
       "  'montebello': 4305,\n",
       "  'vancouver': 6958,\n",
       "  'cumberland': 1563,\n",
       "  'elgin': 1987,\n",
       "  'wyomissing': 7530,\n",
       "  'andover': 212,\n",
       "  'nutley': 4816,\n",
       "  'ithaca': 3187,\n",
       "  'west lafayette': 7238,\n",
       "  'anderson': 211,\n",
       "  'humble': 3082,\n",
       "  'carnegie': 1046,\n",
       "  'santa cruz': 5959,\n",
       "  'tinton falls': 6768,\n",
       "  'sandy': 5937,\n",
       "  'chatsworth': 1186,\n",
       "  'canton': 1010,\n",
       "  'coupeville': 1486,\n",
       "  'bakersfield': 388,\n",
       "  'culpeper': 1561,\n",
       "  'jamaica plain': 3207,\n",
       "  'harrison': 2800,\n",
       "  'ocean': 4855,\n",
       "  'trenton': 6827,\n",
       "  'auburn': 335,\n",
       "  'east hanover': 1869,\n",
       "  'highland heights': 2937,\n",
       "  'addison': 78,\n",
       "  'willow grove': 7399,\n",
       "  'stafford': 6439,\n",
       "  'mount airy': 4381,\n",
       "  'los altos': 3783,\n",
       "  'artesia': 285,\n",
       "  'maspeth': 3989,\n",
       "  'hoffman estates': 2987,\n",
       "  'tacoma': 6647,\n",
       "  'snoqualmie': 6202,\n",
       "  'temecula': 6694,\n",
       "  'oceanport': 4863,\n",
       "  'new richmond': 4601,\n",
       "  'watertown': 7126,\n",
       "  'lewisville': 3669,\n",
       "  'des moines': 1702,\n",
       "  'harbor': 2782,\n",
       "  'winston-salem': 7449,\n",
       "  'gastonia': 2485,\n",
       "  'girard': 2520,\n",
       "  'pittsfield': 5261,\n",
       "  'watkinsville': 7132,\n",
       "  'west windsor': 7279,\n",
       "  'grapevine': 2635,\n",
       "  'lebanon': 3619,\n",
       "  'los gatos': 3795,\n",
       "  'wallington': 7049,\n",
       "  'pine bluff': 5224,\n",
       "  'manchester': 3900,\n",
       "  'wayne': 7150,\n",
       "  'reading': 5497,\n",
       "  'quincy': 5453,\n",
       "  'midland': 4170,\n",
       "  'bell gardens': 524,\n",
       "  'coatesville': 1335,\n",
       "  'holly springs': 3005,\n",
       "  'ridge spring': 5591,\n",
       "  'hanover park': 2775,\n",
       "  'glassboro': 2527,\n",
       "  'hopkinton': 3040,\n",
       "  'buda': 900,\n",
       "  'north bend': 4705,\n",
       "  'alcorn state': 123,\n",
       "  'brookfield': 856,\n",
       "  'weston': 7310,\n",
       "  'round rock': 5736,\n",
       "  'woodstock': 7503,\n",
       "  'beverly hills': 612,\n",
       "  'saipan': 5825,\n",
       "  'uniontown': 6895,\n",
       "  'solon': 6216,\n",
       "  'center valley': 1125,\n",
       "  'lake mary': 3507,\n",
       "  'visalia': 7008,\n",
       "  'naugatuck': 4519,\n",
       "  'independence': 3126,\n",
       "  'joplin': 3266,\n",
       "  'fayetteville': 2198,\n",
       "  'gibbsboro': 2504,\n",
       "  'brea': 805,\n",
       "  'twinsburg': 6873,\n",
       "  'garden grove': 2464,\n",
       "  'avenel': 359,\n",
       "  'kalamazoo': 3286,\n",
       "  'worthington': 7519,\n",
       "  'rio grande city': 5608,\n",
       "  'little canada': 3719,\n",
       "  'glenside': 2559,\n",
       "  'athens': 316,\n",
       "  'platteville': 5290,\n",
       "  'thousand oaks': 6742,\n",
       "  'lubbock': 3817,\n",
       "  'seaford': 6039,\n",
       "  'hamilton': 2752,\n",
       "  'elkridge': 2002,\n",
       "  '12605 e 16th avenue': 15,\n",
       "  'frederick': 2371,\n",
       "  'loveland': 3807,\n",
       "  'north bergen': 4706,\n",
       "  'largo': 3566,\n",
       "  'west long branch': 7243,\n",
       "  'napa': 4500,\n",
       "  'metairie': 4127,\n",
       "  'new albany': 4549,\n",
       "  'mckinney': 4051,\n",
       "  'greenbelt': 2659,\n",
       "  'costa mesa': 1468,\n",
       "  'greenwich': 2676,\n",
       "  'oak brook': 4824,\n",
       "  'emeryville': 2042,\n",
       "  'short hills': 6140,\n",
       "  'green bay': 2650,\n",
       "  'mossville': 4373,\n",
       "  'waukesha': 7139,\n",
       "  'east hartford': 1870,\n",
       "  'valley stream': 6939,\n",
       "  'norridge': 4693,\n",
       "  'sugarland': 6536,\n",
       "  'el paso': 1976,\n",
       "  'munster': 4452,\n",
       "  'davis junction': 1621,\n",
       "  'carmel': 1042,\n",
       "  'long beach': 3763,\n",
       "  'mayfield village': 4019,\n",
       "  'lutz': 3835,\n",
       "  'broomall': 876,\n",
       "  'buena park': 904,\n",
       "  'farmington': 2189,\n",
       "  'seminole': 6073,\n",
       "  'forestville': 2281,\n",
       "  'issaquah': 3185,\n",
       "  'anchorage': 209,\n",
       "  'jersey  city': 3232,\n",
       "  'roswell': 5729,\n",
       "  'pinetops': 5238,\n",
       "  'iriving': 3160,\n",
       "  'palmdale': 5042,\n",
       "  'ft. lauderdale': 2412,\n",
       "  'north reading': 4762,\n",
       "  'west henrietta': 7228,\n",
       "  'st paul': 6380,\n",
       "  'newburyport': 4638,\n",
       "  'armonk': 277,\n",
       "  'lakeland': 3522,\n",
       "  'floral park': 2243,\n",
       "  'oxford': 5003,\n",
       "  'johnston': 3256,\n",
       "  'lindenhurst': 3701,\n",
       "  'reynoldsburg': 5558,\n",
       "  'collierville': 1366,\n",
       "  'freeport': 2384,\n",
       "  'lauderdale by the sea': 3582,\n",
       "  'everett': 2111,\n",
       "  'coppel': 1426,\n",
       "  'springfield gardens': 6365,\n",
       "  'bear': 487,\n",
       "  'west hollywood': 7232,\n",
       "  'loves park': 3808,\n",
       "  'venice': 6968,\n",
       "  'spring house': 6358,\n",
       "  'edina': 1949,\n",
       "  'conway': 1409,\n",
       "  'clakamas': 1272,\n",
       "  'san luis': 5903,\n",
       "  'ontario': 4930,\n",
       "  'south san francisco': 6304,\n",
       "  'winslow': 7444,\n",
       "  'tecumseh': 6688,\n",
       "  'raritan': 5487,\n",
       "  'racine': 5457,\n",
       "  'carmichaels': 1045,\n",
       "  'warwick': 7093,\n",
       "  'hadley': 2728,\n",
       "  'midvale': 4174,\n",
       "  'geneva': 2495,\n",
       "  'perris': 5164,\n",
       "  'hancock': 2770,\n",
       "  'north hollywood': 4736,\n",
       "  'lenexa': 3646,\n",
       "  'la porte': 3450,\n",
       "  'cranford': 1503,\n",
       "  'lighthouse point': 3681,\n",
       "  'pompano beach': 5327,\n",
       "  'murray hill': 4460,\n",
       "  'buffalo grove': 907,\n",
       "  'piscatway': 5252,\n",
       "  'cibolo': 1254,\n",
       "  'waco': 7019,\n",
       "  'grass lake': 2636,\n",
       "  'ruth': 5766,\n",
       "  'fairbanks': 2132,\n",
       "  'euless': 2101,\n",
       "  'fargo': 2179,\n",
       "  'north liberty': 4743,\n",
       "  'bentonville': 572,\n",
       "  'davis': 1620,\n",
       "  'toms river': 6793,\n",
       "  'bagley': 383,\n",
       "  'mentor': 4102,\n",
       "  'palos hills': 5053,\n",
       "  'aloha': 152,\n",
       "  'tomball': 6791,\n",
       "  'amana': 185,\n",
       "  \"st. mary's\": 6412,\n",
       "  'ypsilanti': 7564,\n",
       "  'islandia': 3178,\n",
       "  'indiantown': 3141,\n",
       "  'longview': 3773,\n",
       "  'rivedale': 5616,\n",
       "  'manassas city': 3894,\n",
       "  'paulden': 5114,\n",
       "  'galloway': 2454,\n",
       "  'florham park': 2247,\n",
       "  'marina del rey': 3954,\n",
       "  'riverdale': 5623,\n",
       "  'indiana': 3136,\n",
       "  'defuniak springs': 1655,\n",
       "  'bemidji': 557,\n",
       "  'killeen': 3373,\n",
       "  'western springs': 7291,\n",
       "  'slippery rock': 6192,\n",
       "  'san gabriel': 5889,\n",
       "  'glens falls': 2558,\n",
       "  'encinitas': 2052,\n",
       "  'charles town': 1171,\n",
       "  'dover': 1769,\n",
       "  'powell': 5397,\n",
       "  'farmers branch': 2183,\n",
       "  'kearney': 3307,\n",
       "  'concord': 1395,\n",
       "  'laguna niguel': 3480,\n",
       "  'des plaines': 1704,\n",
       "  'chambersburg': 1149,\n",
       "  'marietta': 3950,\n",
       "  'oakbrook terrace': 4837,\n",
       "  'rensselaer': 5542,\n",
       "  'kyle': 3429,\n",
       "  'pheonix': 5189,\n",
       "  'olivette': 4913,\n",
       "  'basalt': 439,\n",
       "  'hartford': 2809,\n",
       "  'gilbert': 2513,\n",
       "  'port washington': 5366,\n",
       "  'muscatine': 4465,\n",
       "  'peasanton': 5135,\n",
       "  'north las vegas': 4741,\n",
       "  ...},\n",
       " {'NE': 35,\n",
       "  'NJ': 37,\n",
       "  'CA': 4,\n",
       "  'VA': 52,\n",
       "  'NY': 40,\n",
       "  'OR': 43,\n",
       "  'UT': 51,\n",
       "  'GA': 11,\n",
       "  'IN': 18,\n",
       "  'PA': 44,\n",
       "  'MI': 27,\n",
       "  'MA': 22,\n",
       "  'IL': 17,\n",
       "  'FL': 10,\n",
       "  'TX': 50,\n",
       "  'WA': 55,\n",
       "  'NV': 39,\n",
       "  'OK': 42,\n",
       "  'IA': 15,\n",
       "  'MD': 24,\n",
       "  'AL': 1,\n",
       "  'CT': 6,\n",
       "  'DC': 7,\n",
       "  'OH': 41,\n",
       "  'NM': 38,\n",
       "  'GUAM': 13,\n",
       "  'NC': 33,\n",
       "  'CO': 5,\n",
       "  'WI': 56,\n",
       "  'MN': 28,\n",
       "  'KY': 20,\n",
       "  'AR': 2,\n",
       "  'PR': 45,\n",
       "  'NH': 36,\n",
       "  'SC': 47,\n",
       "  'AZ': 3,\n",
       "  'VT': 54,\n",
       "  'MO': 29,\n",
       "  'LA': 21,\n",
       "  'DE': 8,\n",
       "  'ND': 34,\n",
       "  'TN': 49,\n",
       "  'KS': 19,\n",
       "  'WV': 57,\n",
       "  'HI': 14,\n",
       "  'MS': 31,\n",
       "  'RI': 46,\n",
       "  'WY': 58,\n",
       "  'ID': 16,\n",
       "  'ME': 25,\n",
       "  'SD': 48,\n",
       "  'VI': 53,\n",
       "  'MP': 30,\n",
       "  'AK': 0,\n",
       "  'MT': 32,\n",
       "  'MARSHALL ISLANDS': 23,\n",
       "  'FEDERATED STATES OF MICRONESIA': 9,\n",
       "  'GU': 12,\n",
       "  'MH': 26},\n",
       " {\"Master's\": 4,\n",
       "  \"Bachelor's\": 1,\n",
       "  'Doctorate': 2,\n",
       "  'Other': 6,\n",
       "  'None': 5,\n",
       "  'High School': 3,\n",
       "  \"Associate's\": 0},\n",
       " {'N': 0, 'Y': 1},\n",
       " {'INDIA': 80,\n",
       "  'SOUTH KOREA': 165,\n",
       "  'BANGLADESH': 14,\n",
       "  'PHILIPPINES': 143,\n",
       "  'CHINA': 39,\n",
       "  'TAIWAN': 179,\n",
       "  'UNITED KINGDOM': 192,\n",
       "  'JORDAN': 90,\n",
       "  'SAUDI ARABIA': 154,\n",
       "  'ISRAEL': 85,\n",
       "  'COLOMBIA': 40,\n",
       "  'CANADA': 34,\n",
       "  'SPAIN': 168,\n",
       "  'NEPAL': 127,\n",
       "  'ITALY': 86,\n",
       "  'SRI LANKA': 169,\n",
       "  'PAKISTAN': 137,\n",
       "  'FRANCE': 62,\n",
       "  'GHANA': 67,\n",
       "  'EL SALVADOR': 55,\n",
       "  'IRAN': 82,\n",
       "  'NETHERLANDS': 128,\n",
       "  'POLAND': 144,\n",
       "  'TURKEY': 186,\n",
       "  'SENEGAL': 155,\n",
       "  'MEXICO': 117,\n",
       "  'BRAZIL': 25,\n",
       "  'MALAYSIA': 110,\n",
       "  'LEBANON': 99,\n",
       "  'NICARAGUA': 131,\n",
       "  'ECUADOR': 53,\n",
       "  'ROMANIA': 148,\n",
       "  'CZECH REPUBLIC': 47,\n",
       "  'HONG KONG': 77,\n",
       "  'SOUTH AFRICA': 164,\n",
       "  'PERU': 142,\n",
       "  'VENEZUELA': 197,\n",
       "  'VIETNAM': 198,\n",
       "  'NIGERIA': 133,\n",
       "  'JAPAN': 89,\n",
       "  'HUNGARY': 78,\n",
       "  'NEW ZEALAND': 130,\n",
       "  'INDONESIA': 81,\n",
       "  'RUSSIA': 149,\n",
       "  'HONDURAS': 76,\n",
       "  'GUATEMALA': 71,\n",
       "  'AUSTRALIA': 9,\n",
       "  'UKRAINE': 190,\n",
       "  'KENYA': 92,\n",
       "  'ARGENTINA': 6,\n",
       "  'UNITED STATES OF AMERICA': 193,\n",
       "  'URUGUAY': 194,\n",
       "  'HAITI': 75,\n",
       "  'FINLAND': 61,\n",
       "  'DOMINICAN REPUBLIC': 52,\n",
       "  'IRELAND': 84,\n",
       "  'GUYANA': 74,\n",
       "  'ETHIOPIA': 59,\n",
       "  'DENMARK': 49,\n",
       "  'AUSTRIA': 10,\n",
       "  'GERMANY': 66,\n",
       "  'JAMAICA': 88,\n",
       "  'PORTUGAL': 145,\n",
       "  'SWEDEN': 176,\n",
       "  'GREECE': 69,\n",
       "  'ZIMBABWE': 202,\n",
       "  'PALESTINE': 138,\n",
       "  'SERBIA': 156,\n",
       "  'SYRIA': 178,\n",
       "  'SINGAPORE': 160,\n",
       "  'TRINIDAD AND TOBAGO': 184,\n",
       "  'GRENADA': 70,\n",
       "  'MONGOLIA': 121,\n",
       "  'THAILAND': 182,\n",
       "  'BELARUS': 16,\n",
       "  'BOLIVIA': 22,\n",
       "  'BELGIUM': 17,\n",
       "  'EGYPT': 54,\n",
       "  'IRAQ': 83,\n",
       "  'TAJIKISTAN': 180,\n",
       "  'SLOVAKIA': 161,\n",
       "  'MACEDONIA': 107,\n",
       "  'CHILE': 38,\n",
       "  'UZBEKISTAN': 195,\n",
       "  'ALGERIA': 2,\n",
       "  'TUNISIA': 185,\n",
       "  'MOROCCO': 124,\n",
       "  'ZAMBIA': 201,\n",
       "  'CYPRUS': 46,\n",
       "  'LITHUANIA': 104,\n",
       "  'CAMEROON': 33,\n",
       "  'SWITZERLAND': 177,\n",
       "  'KOSOVO': 94,\n",
       "  'BULGARIA': 28,\n",
       "  'SAINT VINCENT AND THE GRENADINES': 151,\n",
       "  'TANZANIA': 181,\n",
       "  'ANGOLA': 4,\n",
       "  'NORWAY': 135,\n",
       "  'AZERBAIJAN': 11,\n",
       "  'PANAMA': 140,\n",
       "  'COSTA RICA': 42,\n",
       "  'MALAWI': 109,\n",
       "  'YEMEN': 199,\n",
       "  'ST KITTS AND NEVIS': 170,\n",
       "  'SERBIA AND MONTENEGRO': 157,\n",
       "  'BURMA (MYANMAR)': 30,\n",
       "  'ANTIGUA AND BARBUDA': 5,\n",
       "  'ST LUCIA': 171,\n",
       "  'LATVIA': 98,\n",
       "  'TURKMENISTAN': 187,\n",
       "  'GEORGIA': 65,\n",
       "  'SOMALIA': 163,\n",
       "  'KYRGYZSTAN': 96,\n",
       "  'MACAU': 106,\n",
       "  'MALI': 112,\n",
       "  'BENIN': 19,\n",
       "  'ICELAND': 79,\n",
       "  'ARMENIA': 7,\n",
       "  'ALBANIA': 1,\n",
       "  'CROATIA': 44,\n",
       "  'RWANDA': 150,\n",
       "  'BAHRAIN': 13,\n",
       "  'BOSNIA AND HERZEGOVINA': 23,\n",
       "  'BELIZE': 18,\n",
       "  'LIBYA': 102,\n",
       "  'ESTONIA': 58,\n",
       "  'GAMBIA': 64,\n",
       "  'NIGER': 132,\n",
       "  'BURKINA FASO': 29,\n",
       "  'DOMINICA': 51,\n",
       "  'EQUATORIAL GUINEA': 56,\n",
       "  'SLOVENIA': 162,\n",
       "  'YUGOSLAVIA': 200,\n",
       "  'ERITREA': 57,\n",
       "  'GUINEA-BISSAU': 73,\n",
       "  'ST VINCENT': 172,\n",
       "  'BAHAMAS': 12,\n",
       "  'KAZAKHSTAN': 91,\n",
       "  'LUXEMBOURG': 105,\n",
       "  'PARAGUAY': 141,\n",
       "  'MOLDOVA': 119,\n",
       "  'KUWAIT': 95,\n",
       "  'TOGO': 183,\n",
       "  'SIERRA LEONE': 159,\n",
       "  'SUDAN': 173,\n",
       "  'BOTSWANA': 24,\n",
       "  'DEMOCRATIC REPUBLIC OF CONGO': 48,\n",
       "  'CAMBODIA': 32,\n",
       "  'MONTENEGRO': 122,\n",
       "  'KIRIBATI': 93,\n",
       "  'BHUTAN': 21,\n",
       "  'AFGHANISTAN': 0,\n",
       "  'MAURITIUS': 116,\n",
       "  'UGANDA': 189,\n",
       "  'IVORY COAST': 87,\n",
       "  'SEYCHELLES': 158,\n",
       "  'UNITED ARAB EMIRATES': 191,\n",
       "  'BERMUDA': 20,\n",
       "  'BARBADOS': 15,\n",
       "  'SURINAME': 174,\n",
       "  'PALESTINIAN TERRITORIES': 139,\n",
       "  'NETHERLANDS ANTILLES': 129,\n",
       "  'NAMIBIA': 126,\n",
       "  \"COTE d'IVOIRE\": 43,\n",
       "  'BURUNDI': 31,\n",
       "  'GABON': 63,\n",
       "  'SOUTH SUDAN': 166,\n",
       "  'MAURITANIA': 115,\n",
       "  'MARSHALL ISLANDS': 114,\n",
       "  'BRITISH VIRGIN ISLANDS': 26,\n",
       "  'FIJI': 60,\n",
       "  'MALDIVES': 111,\n",
       "  'REPUBLIC OF CONGO': 147,\n",
       "  'MADAGASCAR': 108,\n",
       "  'LAOS': 97,\n",
       "  'LIECHTENSTEIN': 103,\n",
       "  'GUINEA': 72,\n",
       "  'SAO TOME AND PRINCIPE': 153,\n",
       "  'CUBA': 45,\n",
       "  'SAMOA': 152,\n",
       "  'SWAZILAND': 175,\n",
       "  'MALTA': 113,\n",
       "  'ANDORRA': 3,\n",
       "  'SOVIET UNION': 167,\n",
       "  'LESOTHO': 100,\n",
       "  'NORTH KOREA': 134,\n",
       "  'CAYMAN ISLANDS': 36,\n",
       "  'CHAD': 37,\n",
       "  'COMOROS': 41,\n",
       "  'BRUNEI': 27,\n",
       "  'TURKS AND CAICOS ISLANDS': 188,\n",
       "  'CAPE VERDE': 35,\n",
       "  'LIBERIA': 101,\n",
       "  'VANUATU': 196,\n",
       "  'OMAN': 136,\n",
       "  'ARUBA': 8,\n",
       "  'MOZAMBIQUE': 125,\n",
       "  'DEPENDANT TERRITORY': 50,\n",
       "  'MONACO': 120,\n",
       "  'MONTSERRAT': 123,\n",
       "  'MICRONESIA': 118,\n",
       "  'GIBRALTAR': 68,\n",
       "  'QATAR': 146},\n",
       " {'H-1B': 18,\n",
       "  'L-1': 29,\n",
       "  'J-1': 25,\n",
       "  'F-1': 12,\n",
       "  'Parolee': 42,\n",
       "  'TN': 50,\n",
       "  'Not in USA': 34,\n",
       "  'B-2': 3,\n",
       "  'A1/A2': 1,\n",
       "  'EWI': 11,\n",
       "  'E-2': 9,\n",
       "  'C-1': 4,\n",
       "  'B-1': 2,\n",
       "  'F-2': 13,\n",
       "  'H-4': 23,\n",
       "  'E-3': 10,\n",
       "  'TPS': 51,\n",
       "  'H-2B': 21,\n",
       "  'R-1': 44,\n",
       "  'O-1': 35,\n",
       "  'E-1': 8,\n",
       "  'L-2': 30,\n",
       "  'H-1B1': 19,\n",
       "  'A-3': 0,\n",
       "  'H-1A': 17,\n",
       "  'H-3': 22,\n",
       "  'G-4': 15,\n",
       "  'J-2': 26,\n",
       "  'P-1': 38,\n",
       "  'G-5': 16,\n",
       "  'P-3': 40,\n",
       "  'M-1': 31,\n",
       "  'TD': 49,\n",
       "  'I': 24,\n",
       "  'VWT': 56,\n",
       "  'N': 33,\n",
       "  'VWB': 55,\n",
       "  'G-1': 14,\n",
       "  'R-2': 45,\n",
       "  'O-2': 36,\n",
       "  'H-2A': 20,\n",
       "  'C-3': 5,\n",
       "  'Q': 43,\n",
       "  'V-2': 54,\n",
       "  'P-4': 41,\n",
       "  'K-1': 27,\n",
       "  'D-1': 6,\n",
       "  'O-3': 37,\n",
       "  'T-1': 47,\n",
       "  'M-2': 32,\n",
       "  'P-2': 39,\n",
       "  'V-1': 53,\n",
       "  'U-1': 52,\n",
       "  'D-2': 7,\n",
       "  'S-6': 46,\n",
       "  'T-2': 48,\n",
       "  'K-4': 28},\n",
       " {\"Bachelor's\": 1,\n",
       "  \"Master's\": 4,\n",
       "  'Doctorate': 2,\n",
       "  'Other': 6,\n",
       "  \"Associate's\": 0,\n",
       "  'None': 5,\n",
       "  'High School': 3},\n",
       " {'computer occupations': 12,\n",
       "  'Architecture & Engineering': 1,\n",
       "  'Others': 11,\n",
       "  'Medical Occupations': 10,\n",
       "  'Education Occupations': 3,\n",
       "  'Management Occupation': 7,\n",
       "  'Financial Occupation': 4,\n",
       "  'Advance Sciences': 0,\n",
       "  'Food Occupation': 5,\n",
       "  'Business Occupation': 2,\n",
       "  'Law Occupation': 6,\n",
       "  'Mathematical Occupations': 9,\n",
       "  'Marketing Occupation': 8},\n",
       " {'STEM Major': 1, 'NON-STEM Major': 0},\n",
       " {'STEM Major': 1, 'NON-STEM Major': 0},\n",
       " {'Other Employer': 0, 'Top 10 Employer': 2, 'TOP 10-2O Employer': 1},\n",
       " {'Between 60000 -100000': 5,\n",
       "  'Between 100000 -1500000': 2,\n",
       "  'Between 30000 -60000': 4,\n",
       "  'Below 30000': 1,\n",
       "  'Between 150000 -2000000': 3,\n",
       "  'Above 200000': 0}]"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#making dictionary to map the label encoder from call back later\n",
    "cols =list(df_final.columns)\n",
    "cols\n",
    "\n",
    "dict_list=[]\n",
    "for i in range(len(cols)):\n",
    "    keys= list(dff_final[cols[i]])\n",
    "    values =list(df_final[cols[i]])\n",
    "    #print(keys[2])\n",
    "    #print(values[2])\n",
    "    #print(len(keys))\n",
    "    #print(len(values))\n",
    "    \n",
    "    dict_name = cols[i]+'_'+'dict'\n",
    "    #print(dict_name)\n",
    "    \n",
    "    dict_list.append(dict_name)\n",
    "    dict_list[i]={}\n",
    "    \n",
    "    for a in range(len(keys)):\n",
    "        dict_list[i][keys[a]] = values[a]\n",
    "dict_list\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e09ffab4",
   "metadata": {},
   "source": [
    "## Best model is Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "69c36bae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "JOB_CITY             0.201216\n",
       "OCCUPATION           0.178009\n",
       "CITIZENSHIP          0.116924\n",
       "WAGE_OFFERED         0.110781\n",
       "JOB_STATE            0.100977\n",
       "REQD_EDUCATION       0.080858\n",
       "ADMISSION_TYPE       0.079428\n",
       "WORKER_EDUCATION     0.065121\n",
       "NEW_WORKER_MAJOR     0.028658\n",
       "NEW_RELATED_MAJOR    0.017706\n",
       "NEW_EMPLOYER_NAME    0.017459\n",
       "TRAINING_REQD        0.002863\n",
       "dtype: float64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "final_forest = RandomForestClassifier(n_estimators = 100, max_depth = 35, max_features = 4, random_state = 123, n_jobs = -1)\n",
    "final_forest.fit(x_train_res, y_train_res)\n",
    "\n",
    "feature_scores = pd.Series(final_forest.feature_importances_, index=x_train_res.columns).sort_values(ascending=False)\n",
    "\n",
    "feature_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "3e8df882",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "marker": {
          "color": "blue"
         },
         "orientation": "v",
         "type": "bar",
         "x": [
          "JOB_CITY",
          "OCCUPATION",
          "CITIZENSHIP",
          "WAGE_OFFERED",
          "JOB_STATE",
          "REQD_EDUCATION",
          "ADMISSION_TYPE",
          "WORKER_EDUCATION",
          "NEW_WORKER_MAJOR",
          "NEW_RELATED_MAJOR",
          "NEW_EMPLOYER_NAME",
          "TRAINING_REQD"
         ],
         "y": [
          0.2012156666249775,
          0.17800942537340614,
          0.11692399173703326,
          0.11078148905839583,
          0.10097700853419835,
          0.08085812370899721,
          0.07942848423133676,
          0.06512059594860747,
          0.028657512225159686,
          0.017705778798244365,
          0.017458704627126592,
          0.002863219132516811
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Visualization of Feature Importance Score "
        },
        "xaxis": {
         "title": {
          "text": "Features"
         }
        },
        "yaxis": {
         "title": {
          "text": "Feature importance score"
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"adc4a936-5165-4de1-b099-538b5cbcff0b\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"adc4a936-5165-4de1-b099-538b5cbcff0b\")) {                    Plotly.newPlot(                        \"adc4a936-5165-4de1-b099-538b5cbcff0b\",                        [{\"marker\":{\"color\":\"blue\"},\"orientation\":\"v\",\"x\":[\"JOB_CITY\",\"OCCUPATION\",\"CITIZENSHIP\",\"WAGE_OFFERED\",\"JOB_STATE\",\"REQD_EDUCATION\",\"ADMISSION_TYPE\",\"WORKER_EDUCATION\",\"NEW_WORKER_MAJOR\",\"NEW_RELATED_MAJOR\",\"NEW_EMPLOYER_NAME\",\"TRAINING_REQD\"],\"y\":[0.2012156666249775,0.17800942537340614,0.11692399173703326,0.11078148905839583,0.10097700853419835,0.08085812370899721,0.07942848423133676,0.06512059594860747,0.028657512225159686,0.017705778798244365,0.017458704627126592,0.002863219132516811],\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}},\"title\":{\"text\":\"Visualization of Feature Importance Score \"},\"yaxis\":{\"title\":{\"text\":\"Feature importance score\"}},\"xaxis\":{\"title\":{\"text\":\"Features\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('adc4a936-5165-4de1-b099-538b5cbcff0b');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig  = go.Figure([go.Bar(y = feature_scores, x=feature_scores.index, marker_color = 'blue',orientation='v')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "fig.update_layout(title = 'Visualization of Feature Importance Score ',\n",
    "                 yaxis_title = 'Feature importance score',\n",
    "                 xaxis_title = 'Features')\n",
    "                 #barmode = 'group')\n",
    "\n",
    "fig.show()\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "514cd499",
   "metadata": {},
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "final_forest = RandomForestClassifier(n_estimators = 50, max_depth = 30, max_features = 4, random_state = 123, n_jobs = -1)\n",
    "final_forest.fit(x_train_res, y_train_res)\n",
    "importances = final_forest.feature_importances_\n",
    "std = np.std([tree.feature_importances_ for tree in final_forest.estimators_],\n",
    "             axis=0)\n",
    "indices = np.argsort(importances)[::-1]\n",
    "\n",
    "# Plot the feature importances of the forest\n",
    "plt.figure(figsize=(10, 8))\n",
    "plt.title(\"Feature importances\")\n",
    "plt.bar(range(x_train_res.shape[1]), importances[indices],\n",
    "       color=\"g\", yerr=std[indices], align=\"center\")\n",
    "plt.xticks(range(x_train_res.shape[1]), x_train_res.columns, rotation = 90)\n",
    "plt.xlim([-1, x_train_res.shape[1]])\n",
    "plt.show()                                             "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff2a3209",
   "metadata": {},
   "source": [
    "### saving best model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "5320950d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#best_model = final_forest\n",
    "path ='/Users/ujjwaloli/Desktop/Capstone Project/'\n",
    "\n",
    "\n",
    "\n",
    "pickle_out = open(path + \"best_model_file2\",\"wb\")\n",
    "pickle.dump(final_forest, pickle_out)\n",
    "pickle_out.close()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "050f59ed",
   "metadata": {},
   "source": [
    "### reading best model file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "74ff59e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#pickle_in = open(path+\"best_model_file\",'rb')\n",
    "pickle_in = open(\"best_model_file2\",'rb')\n",
    "model = pickle.load(pickle_in)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "f166b5cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[JOB_CITY             2191\n",
       " JOB_STATE              27\n",
       " REQD_EDUCATION          4\n",
       " CITIZENSHIP            80\n",
       " ADMISSION_TYPE         18\n",
       " WORKER_EDUCATION        4\n",
       " time_taken              4\n",
       " OCCUPATION             12\n",
       " NEW_RELATED_MAJOR       1\n",
       " NEW_WORKER_MAJOR        1\n",
       " NEW_EMPLOYER_NAME       0\n",
       " WAGE_OFFERED            5\n",
       " Name: 429731, dtype: int64]"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test.iloc[0]\n",
    "[x_test.iloc[0]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "90b8a76a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>JOB_CITY</th>\n",
       "      <th>JOB_STATE</th>\n",
       "      <th>REQD_EDUCATION</th>\n",
       "      <th>CITIZENSHIP</th>\n",
       "      <th>ADMISSION_TYPE</th>\n",
       "      <th>WORKER_EDUCATION</th>\n",
       "      <th>time_taken</th>\n",
       "      <th>OCCUPATION</th>\n",
       "      <th>NEW_RELATED_MAJOR</th>\n",
       "      <th>NEW_WORKER_MAJOR</th>\n",
       "      <th>NEW_EMPLOYER_NAME</th>\n",
       "      <th>WAGE_OFFERED</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>436527</th>\n",
       "      <td>1</td>\n",
       "      <td>seattle</td>\n",
       "      <td>WA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>L-1</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Top 10 Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>418820</th>\n",
       "      <td>1</td>\n",
       "      <td>santa clara</td>\n",
       "      <td>CA</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 9 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 150000 -2000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44619</th>\n",
       "      <td>1</td>\n",
       "      <td>parlin</td>\n",
       "      <td>NJ</td>\n",
       "      <td>Master's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 24 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>389229</th>\n",
       "      <td>1</td>\n",
       "      <td>wayne</td>\n",
       "      <td>PA</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>235026</th>\n",
       "      <td>1</td>\n",
       "      <td>new york</td>\n",
       "      <td>NY</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>SOUTH KOREA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>Others</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 30000 -60000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>291552</th>\n",
       "      <td>1</td>\n",
       "      <td>college station</td>\n",
       "      <td>TX</td>\n",
       "      <td>Master's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 3 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Top 10 Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>66310</th>\n",
       "      <td>1</td>\n",
       "      <td>redmond</td>\n",
       "      <td>WA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 60000 -100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>130161</th>\n",
       "      <td>1</td>\n",
       "      <td>paramus</td>\n",
       "      <td>NJ</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>GERMANY</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Bachelor's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>Others</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 150000 -2000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>389072</th>\n",
       "      <td>1</td>\n",
       "      <td>mountain view</td>\n",
       "      <td>CA</td>\n",
       "      <td>Master's</td>\n",
       "      <td>INDIA</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>computer occupations</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 100000 -1500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>124747</th>\n",
       "      <td>1</td>\n",
       "      <td>wausau</td>\n",
       "      <td>WI</td>\n",
       "      <td>Master's</td>\n",
       "      <td>TAIWAN</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>Master's</td>\n",
       "      <td>at least 6 months</td>\n",
       "      <td>Financial Occupation</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>NON-STEM Major</td>\n",
       "      <td>Other Employer</td>\n",
       "      <td>Between 30000 -60000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150000 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        CASE_STATUS         JOB_CITY JOB_STATE REQD_EDUCATION  CITIZENSHIP  \\\n",
       "436527            1          seattle        WA       Master's        INDIA   \n",
       "418820            1      santa clara        CA     Bachelor's        INDIA   \n",
       "44619             1           parlin        NJ       Master's        INDIA   \n",
       "389229            1            wayne        PA     Bachelor's        INDIA   \n",
       "235026            1         new york        NY     Bachelor's  SOUTH KOREA   \n",
       "...             ...              ...       ...            ...          ...   \n",
       "291552            1  college station        TX       Master's        INDIA   \n",
       "66310             1          redmond        WA       Master's        INDIA   \n",
       "130161            1          paramus        NJ     Bachelor's      GERMANY   \n",
       "389072            1    mountain view        CA       Master's        INDIA   \n",
       "124747            1           wausau        WI       Master's       TAIWAN   \n",
       "\n",
       "       ADMISSION_TYPE WORKER_EDUCATION          time_taken  \\\n",
       "436527            L-1       Bachelor's   at least 3 months   \n",
       "418820           H-1B       Bachelor's   at least 9 months   \n",
       "44619            H-1B         Master's  at least 24 months   \n",
       "389229           H-1B         Master's   at least 6 months   \n",
       "235026           H-1B       Bachelor's   at least 3 months   \n",
       "...               ...              ...                 ...   \n",
       "291552           H-1B         Master's   at least 3 months   \n",
       "66310            H-1B       Bachelor's   at least 6 months   \n",
       "130161           H-1B       Bachelor's   at least 6 months   \n",
       "389072           H-1B         Master's   at least 6 months   \n",
       "124747           H-1B         Master's   at least 6 months   \n",
       "\n",
       "                  OCCUPATION NEW_RELATED_MAJOR NEW_WORKER_MAJOR  \\\n",
       "436527  computer occupations        STEM Major       STEM Major   \n",
       "418820  computer occupations        STEM Major       STEM Major   \n",
       "44619   computer occupations        STEM Major       STEM Major   \n",
       "389229  computer occupations        STEM Major   NON-STEM Major   \n",
       "235026                Others    NON-STEM Major   NON-STEM Major   \n",
       "...                      ...               ...              ...   \n",
       "291552  computer occupations        STEM Major       STEM Major   \n",
       "66310   computer occupations        STEM Major       STEM Major   \n",
       "130161                Others    NON-STEM Major   NON-STEM Major   \n",
       "389072  computer occupations        STEM Major       STEM Major   \n",
       "124747  Financial Occupation    NON-STEM Major   NON-STEM Major   \n",
       "\n",
       "       NEW_EMPLOYER_NAME             WAGE_OFFERED  \n",
       "436527   Top 10 Employer    Between 60000 -100000  \n",
       "418820    Other Employer  Between 150000 -2000000  \n",
       "44619     Other Employer    Between 60000 -100000  \n",
       "389229    Other Employer    Between 60000 -100000  \n",
       "235026    Other Employer     Between 30000 -60000  \n",
       "...                  ...                      ...  \n",
       "291552   Top 10 Employer    Between 60000 -100000  \n",
       "66310     Other Employer    Between 60000 -100000  \n",
       "130161    Other Employer  Between 150000 -2000000  \n",
       "389072    Other Employer  Between 100000 -1500000  \n",
       "124747    Other Employer     Between 30000 -60000  \n",
       "\n",
       "[150000 rows x 13 columns]"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_dff"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e665c6d",
   "metadata": {},
   "source": [
    "# Dashboard starts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "86c8632f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [10/Feb/2022 16:21:27] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [10/Feb/2022 16:21:32] \"GET /_dash-layout HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [10/Feb/2022 16:21:32] \"GET /_dash-dependencies HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [10/Feb/2022 16:21:33] \"GET /_favicon.ico?v=2.1.0 HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "#Navigation bar\n",
    "\n",
    "app = dash.Dash(external_stylesheets=[ dbc.themes.FLATLY],)\n",
    "\n",
    "DOL_Logo =\"https://twitter.com/USDOL/photo\" #reference: https://twitter.com/USDOL\n",
    "\n",
    "navbar = dbc.Navbar(id='navbar', children =[\n",
    "    \n",
    "    html.A(\n",
    "    dbc.Row([\n",
    "        dbc.Col(html.Img(src = DOL_Logo, height= \"70px\")),\n",
    "        dbc.Col(\n",
    "            dbc.NavbarBrand(\"Perm Cases Tracker\", style ={'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'})\n",
    "        \n",
    "        )\n",
    "        \n",
    "        ], align =\"center\"), #aligns title to center\n",
    "        #no_gutters=True),\n",
    "    href ='/'\n",
    "    ),\n",
    "    dbc.Button(id ='button', children = 'Clicke Me!', color ='primary', className ='ml-auto',href='/')\n",
    "    \n",
    "    \n",
    "    ])\n",
    "    \n",
    "    \n",
    "    \n",
    "app.layout =html.Div(id ='parent', children=[navbar])\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run_server()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ff82179",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "from dash import Dash, dcc, html\n",
    "from dash.dependencies import Input, Output, State\n",
    "import dash_design_kit as ddk\n",
    "\n",
    "external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']\n",
    "\n",
    "app = Dash(__name__, external_stylesheets=external_stylesheets)\n",
    "\n",
    "app.layout = html.Div([\n",
    "    dcc.Input(id='input-1-state', type='text', value='Montréal'),\n",
    "    dcc.Input(id='input-2-state', type='text', value='Canada'),\n",
    "    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),\n",
    "    html.Div(id='output-state')\n",
    "])\n",
    "\n",
    "\n",
    "@app.callback(Output('output-state', 'children'),\n",
    "              Input('submit-button-state', 'n_clicks'),\n",
    "              State('input-1-state', 'value'),\n",
    "              State('input-2-state', 'value'))\n",
    "def update_output(n_clicks, input1, input2):\n",
    "    return u'''\n",
    "        The Button has been pressed {} times,\n",
    "        Input 1 is \"{}\",\n",
    "        and Input 2 is \"{}\"\n",
    "    '''.format(n_clicks, input1, input2)\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run_server(debug=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0f9c961",
   "metadata": {},
   "source": [
    "def encode_test_series(test_series,final_dff):\n",
    "\n",
    "    final_dfff = final_dff.append(test_series, ignore_index = True)\n",
    "    \n",
    "    print(final_dff.tail())\n",
    "    \n",
    "    print(\"*********************\")\n",
    "    print(final_dfff.tail())\n",
    "    \n",
    "    final_dfff['JOB_CITY'] = final_dfff['JOB_CITY'].str.lower()\n",
    "    \n",
    "    categorical_col=['JOB_CITY', 'JOB_STATE','REQD_EDUCATION', 'CITIZENSHIP', 'ADMISSION_TYPE', 'WORKER_EDUCATION','time_taken', 'OCCUPATION', 'NEW_RELATED_MAJOR', 'NEW_WORKER_MAJOR', 'NEW_EMPLOYER_NAME', 'WAGE_OFFERED']\n",
    "    dummy_df = pd.get_dummies(final_dfff[categorical_col])\n",
    "    final_dfff =pd.concat([final_dfff,dummy_df],axis=1)\n",
    "    final_dfff =final_dfff.drop(categorical_col,axis=1)\n",
    "\n",
    "\n",
    "    test_series_encode = final_dfff.tail(1)\n",
    "    \n",
    "    test_series_encode.drop('CASE_STATUS',  axis=1, inplace=True)\n",
    "\n",
    "    #len_val = len(final_dfff)\n",
    "\n",
    "    #finale_dff = final_dfff.iloc[:len_val-1]\n",
    "\n",
    "    x = final_dff.drop('CASE_STATUS', axis=1)\n",
    "    y = final_dff.CASE_STATUS\n",
    "    \n",
    "    \n",
    "    print(\"Printing x and y\")\n",
    "    #print(x)\n",
    "    #print(y)\n",
    "    \n",
    "    \n",
    "    \n",
    "    seed = 10\n",
    "    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)\n",
    "    print(x_train.shape,y_train.shape)\n",
    "    print(x_test.shape,y_test.shape)\n",
    "    \n",
    "    \n",
    "    print(y_train.value_counts())\n",
    "    print(y_test.value_counts())\n",
    "    \n",
    "    from imblearn.over_sampling import SMOTE\n",
    "    sm = SMOTE(random_state=12, sampling_strategy=1)\n",
    "    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)\n",
    "\n",
    "\n",
    "    return test_series_encode,x_train_res, y_train_res,x_test, y_test\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "1c7b4514",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        CASE_STATUS         JOB_CITY JOB_STATE REQD_EDUCATION CITIZENSHIP  \\\n",
      "291552            1  college station        TX       Master's       INDIA   \n",
      "66310             1          redmond        WA       Master's       INDIA   \n",
      "130161            1          paramus        NJ     Bachelor's     GERMANY   \n",
      "389072            1    mountain view        CA       Master's       INDIA   \n",
      "124747            1           wausau        WI       Master's      TAIWAN   \n",
      "\n",
      "       ADMISSION_TYPE WORKER_EDUCATION         time_taken  \\\n",
      "291552           H-1B         Master's  at least 3 months   \n",
      "66310            H-1B       Bachelor's  at least 6 months   \n",
      "130161           H-1B       Bachelor's  at least 6 months   \n",
      "389072           H-1B         Master's  at least 6 months   \n",
      "124747           H-1B         Master's  at least 6 months   \n",
      "\n",
      "                  OCCUPATION NEW_RELATED_MAJOR NEW_WORKER_MAJOR  \\\n",
      "291552  computer occupations        STEM Major       STEM Major   \n",
      "66310   computer occupations        STEM Major       STEM Major   \n",
      "130161                Others    NON-STEM Major   NON-STEM Major   \n",
      "389072  computer occupations        STEM Major       STEM Major   \n",
      "124747  Financial Occupation    NON-STEM Major   NON-STEM Major   \n",
      "\n",
      "       NEW_EMPLOYER_NAME             WAGE_OFFERED  \n",
      "291552   Top 10 Employer    Between 60000 -100000  \n",
      "66310     Other Employer    Between 60000 -100000  \n",
      "130161    Other Employer  Between 150000 -2000000  \n",
      "389072    Other Employer  Between 100000 -1500000  \n",
      "124747    Other Employer     Between 30000 -60000  \n",
      "*********************\n",
      "       CASE_STATUS       JOB_CITY JOB_STATE REQD_EDUCATION CITIZENSHIP  \\\n",
      "149996           1        redmond        WA       Master's       INDIA   \n",
      "149997           1        paramus        NJ     Bachelor's     GERMANY   \n",
      "149998           1  mountain view        CA       Master's       INDIA   \n",
      "149999           1         wausau        WI       Master's      TAIWAN   \n",
      "150000                   new york        NY    High School       INDIA   \n",
      "\n",
      "       ADMISSION_TYPE WORKER_EDUCATION          time_taken  \\\n",
      "149996           H-1B       Bachelor's   at least 6 months   \n",
      "149997           H-1B       Bachelor's   at least 6 months   \n",
      "149998           H-1B         Master's   at least 6 months   \n",
      "149999           H-1B         Master's   at least 6 months   \n",
      "150000           H-1B      High School  at least 15 months   \n",
      "\n",
      "                  OCCUPATION NEW_RELATED_MAJOR NEW_WORKER_MAJOR  \\\n",
      "149996  computer occupations        STEM Major       STEM Major   \n",
      "149997                Others    NON-STEM Major   NON-STEM Major   \n",
      "149998  computer occupations        STEM Major       STEM Major   \n",
      "149999  Financial Occupation    NON-STEM Major   NON-STEM Major   \n",
      "150000  computer occupations        STEM Major       STEM Major   \n",
      "\n",
      "       NEW_EMPLOYER_NAME             WAGE_OFFERED  \n",
      "149996    Other Employer    Between 60000 -100000  \n",
      "149997    Other Employer  Between 150000 -2000000  \n",
      "149998    Other Employer  Between 100000 -1500000  \n",
      "149999    Other Employer     Between 30000 -60000  \n",
      "150000    Other Employer              Below 30000  \n",
      "Printing x and y\n",
      "(105000, 12) (105000,)\n",
      "(45000, 12) (45000,)\n",
      "1    99091\n",
      "0     5909\n",
      "Name: CASE_STATUS, dtype: int64\n",
      "1    42460\n",
      "0     2540\n",
      "Name: CASE_STATUS, dtype: int64\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Unable to convert array of bytes/strings into decimal numbers with dtype='numeric'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[1;32m    786\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 787\u001b[0;31m                 \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat64\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    788\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mValueError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: could not convert string to float: 'new york'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_49744/3546004773.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 16\u001b[0;31m \u001b[0mtest_series_encode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mx_train_res\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train_res\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mx_test\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_test\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mencode_test_series\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest_series\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mfinal_dff\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     17\u001b[0m \u001b[0;31m#print(test_series_encode)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_49744/573384284.py\u001b[0m in \u001b[0;36mencode_test_series\u001b[0;34m(test_series, final_dff)\u001b[0m\n\u001b[1;32m     45\u001b[0m     \u001b[0;32mfrom\u001b[0m \u001b[0mimblearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mover_sampling\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mSMOTE\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     46\u001b[0m     \u001b[0msm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mSMOTE\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m12\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msampling_strategy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 47\u001b[0;31m     \u001b[0mx_train_res\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train_res\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msm\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit_resample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     48\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     49\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/imblearn/base.py\u001b[0m in \u001b[0;36mfit_resample\u001b[0;34m(self, X, y)\u001b[0m\n\u001b[1;32m     75\u001b[0m         \u001b[0mcheck_classification_targets\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     76\u001b[0m         \u001b[0marrays_transformer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mArraysTransformer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 77\u001b[0;31m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbinarize_y\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_X_y\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     78\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     79\u001b[0m         self.sampling_strategy_ = check_sampling_strategy(\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/imblearn/base.py\u001b[0m in \u001b[0;36m_check_X_y\u001b[0;34m(self, X, y, accept_sparse)\u001b[0m\n\u001b[1;32m    130\u001b[0m             \u001b[0maccept_sparse\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m\"csr\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"csc\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    131\u001b[0m         \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbinarize_y\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_target_type\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mindicate_one_vs_all\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 132\u001b[0;31m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_validate_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreset\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0maccept_sparse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    133\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbinarize_y\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    134\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    579\u001b[0m                 \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    580\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 581\u001b[0;31m                 \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    582\u001b[0m             \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    583\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[1;32m    962\u001b[0m         \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"y cannot be None\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    963\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 964\u001b[0;31m     X = check_array(\n\u001b[0m\u001b[1;32m    965\u001b[0m         \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    966\u001b[0m         \u001b[0maccept_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0maccept_sparse\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[1;32m    787\u001b[0m                 \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat64\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    788\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mValueError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 789\u001b[0;31m                 raise ValueError(\n\u001b[0m\u001b[1;32m    790\u001b[0m                     \u001b[0;34m\"Unable to convert array of bytes/strings \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    791\u001b[0m                     \u001b[0;34m\"into decimal numbers with dtype='numeric'\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: Unable to convert array of bytes/strings into decimal numbers with dtype='numeric'"
     ]
    }
   ],
   "source": [
    "test_series =pd.Series({'CASE_STATUS':'',\n",
    "                        'JOB_CITY':str('new york'),\n",
    "                       'JOB_STATE':'NY',\n",
    "                       'REQD_EDUCATION': 'High School',\n",
    "                       'CITIZENSHIP':'INDIA',\n",
    "                       'ADMISSION_TYPE':'H-1B',\n",
    "                       'WORKER_EDUCATION':'High School',\n",
    "                       'time_taken': 'at least 15 months',\n",
    "                       'OCCUPATION': 'computer occupations',\n",
    "                       'NEW_RELATED_MAJOR': 'STEM Major',\n",
    "                       'NEW_WORKER_MAJOR': 'STEM Major',\n",
    "                       'NEW_EMPLOYER_NAME':'Other Employer',\n",
    "                       'WAGE_OFFERED': 'Below 30000'})\n",
    "        \n",
    "\n",
    "test_series_encode,x_train_res, y_train_res,x_test, y_test = encode_test_series(test_series,final_dff)\n",
    "#print(test_series_encode)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac57e5d4",
   "metadata": {},
   "source": [
    "# working dashboard 2\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a47239c",
   "metadata": {},
   "source": [
    "# working dashboard 2 finish"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13988962",
   "metadata": {},
   "source": [
    "## working dashboard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "7014ae9a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "212316\n",
      "180389\n",
      "94755\n",
      "20665\n",
      "10425\n",
      "8195\n",
      "2715\n",
      "2705\n",
      "2395\n",
      "192\n",
      "13\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:15] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/deps/polyfill@7.v2_1_0m1643989747.12.1.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/deps/react@16.v2_1_0m1643989747.14.0.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/deps/react-dom@16.v2_1_0m1643989747.14.0.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/deps/prop-types@15.v2_1_0m1643989747.7.2.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/dcc/dash_core_components-shared.v2_1_0m1643989747.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash_bootstrap_components/_components/dash_bootstrap_components.v1_0_3m1644272688.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/dcc/dash_core_components.v2_1_0m1643989747.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/dash-renderer/build/dash_renderer.v2_1_0m1643989747.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/dash_table/bundle.v5_1_0m1643989747.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:16] \"GET /_dash-component-suites/dash/html/dash_html_components.v2_0_1m1643989747.min.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:20] \"GET /_dash-dependencies HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:23] \"GET /_favicon.ico?v=2.1.0 HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:23] \"GET /_dash-layout HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:25] \"GET /_dash-component-suites/dash/dcc/async-markdown.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:25] \"GET /_dash-component-suites/dash/dcc/async-graph.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:25] \"GET /_dash-component-suites/dash/dcc/async-dropdown.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:25] \"GET /_dash-component-suites/dash/dcc/async-plotlyjs.js HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:27] \"POST /_dash-update-component HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:27] \"POST /_dash-update-component HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:27] \"POST /_dash-update-component HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:27] \"POST /_dash-update-component HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [14/Apr/2022 23:47:27] \"POST /_dash-update-component HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Country name is  INDIA\n",
      "*****************############################\n",
      "CertifiedJob_city is\n",
      " new york\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [14/Apr/2022 23:47:27] \"GET /_dash-component-suites/dash/dcc/async-highlight.js HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "*****************############################\n",
      "Job_city is new york\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 496, in show_answer\n",
      "    value_2 = dict_list[2][str(job_state)]\n",
      "KeyError: 'NEW YO'\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 496, in show_answer\n",
      "    value_2 = dict_list[2][str(job_state)]\n",
      "KeyError: 'NEW YO'\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 496, in show_answer\n",
      "    value_2 = dict_list[2][str(job_state)]\n",
      "KeyError: 'NEW YO'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [14/Apr/2022 23:47:56] \"POST /_dash-update-component HTTP/1.1\" 500 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "*****************############################\n",
      "Job_city is new york\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 496, in show_answer\n",
      "    value_2 = dict_list[2][str(job_state)]\n",
      "KeyError: 'NEW YO'\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 496, in show_answer\n",
      "    value_2 = dict_list[2][str(job_state)]\n",
      "KeyError: 'NEW YO'\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 496, in show_answer\n",
      "    value_2 = dict_list[2][str(job_state)]\n",
      "KeyError: 'NEW YO'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [14/Apr/2022 23:48:01] \"POST /_dash-update-component HTTP/1.1\" 500 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "*****************############################\n",
      "Job_city is new york\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 498, in show_answer\n",
      "    value_4=dict_list[4][str(citizenship_type)]\n",
      "KeyError: 'INDIA'\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 498, in show_answer\n",
      "    value_4=dict_list[4][str(citizenship_type)]\n",
      "KeyError: 'INDIA'\n",
      "Exception on /_dash-update-component [POST]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 2447, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1952, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1821, in handle_user_exception\n",
      "    reraise(exc_type, exc_value, tb)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/_compat.py\", line 39, in reraise\n",
      "    raise value\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1950, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\", line 1936, in dispatch_request\n",
      "    return self.view_functions[rule.endpoint](**req.view_args)\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\", line 1344, in dispatch\n",
      "    response.set_data(func(*args, outputs_list=outputs_list))\n",
      "  File \"/Users/ujjwaloli/opt/anaconda3/lib/python3.9/site-packages/dash/_callback.py\", line 151, in add_context\n",
      "    output_value = func(*func_args, **func_kwargs)  # %% callback invoked %%\n",
      "  File \"/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3620009355.py\", line 498, in show_answer\n",
      "    value_4=dict_list[4][str(citizenship_type)]\n",
      "KeyError: 'INDIA'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [14/Apr/2022 23:48:27] \"POST /_dash-update-component HTTP/1.1\" 500 -\n"
     ]
    }
   ],
   "source": [
    "\n",
    "#Navigation bar\n",
    "\n",
    "app = dash.Dash(external_stylesheets=[ dbc.themes.FLATLY],)\n",
    "\n",
    "#\"https://en.wikipedia.org/wiki/United_States_Department_of_Labor#/media/File:Seal_of_the_United_States_Department_of_Labor.svg\" #reference: https://twitter.com/USDOL\n",
    "\n",
    "DOL_Logo = mpimg.imread('/Users/ujjwaloli/Desktop/Capstone Project/DashboardLogo.png')\n",
    "\n",
    "\n",
    "total = len(df.index)\n",
    "certified = df[df['CASE_STATUS']=='Certified']['CASE_STATUS'].count()\n",
    "denied = df[df['CASE_STATUS']=='Denied']['CASE_STATUS'].count()\n",
    "certified_perc = round(certified/total *100,2)\n",
    "denied_perc = round(denied/total *100,2)\n",
    "\n",
    "def line_plot(df):\n",
    "    fig = go.Figure(data = [go.Scatter(y =cases_status_df['Approved_Case'], x=cases_status_df['Year'], line=dict(color ='firebrick', width =4),text = cases_status_df['Approved_Case'] , name ='Cases Approved'),\n",
    "                           go.Scatter(y =cases_status_df['Denied_Case'], x=cases_status_df['Year'], line=dict(color ='blue', width =4),text = cases_status_df['Denied_Case'] , name ='Cases Denied')])\n",
    "    \n",
    "    \n",
    "    fig.update_layout(title='Analysis of Approved cases and Denied Cases over the years',\n",
    "                         xaxis_title='Year',\n",
    "                         yaxis_title='No of cases',\n",
    "                         margin=dict(l =4,r=4,t=30,b=4))\n",
    "    return fig\n",
    "\n",
    "\n",
    "def lolipop_plot(decesion_df_total):\n",
    "    fig1 = go.Figure()\n",
    "    # Draw points\n",
    "    fig1.add_trace(go.Scatter(x = decesion_df_total[\"Total decided cases\"],\n",
    "                              y = decesion_df_total[\"Time Taken\"],\n",
    "                              mode = 'markers',\n",
    "                              marker_color ='darkblue',\n",
    "                              marker_size  = 10))\n",
    "    # Draw lines\n",
    "    for i in range(0, len(decesion_df_total)):\n",
    "        print(decesion_df_total[\"Total decided cases\"][i])\n",
    "        fig1.add_shape(type='line',x0 = 0, y0 = i,\n",
    "                                   x1 = decesion_df_total[\"Total decided cases\"][i],\n",
    "                                  y1 = i,\n",
    "                                  line=dict(color='crimson', width = 3))\n",
    "    # Set title\n",
    "    fig1.update_layout(title_text = \n",
    "                       \"Analysis of time taken for cases to decided\",\n",
    "                       title_font_size = 20)\n",
    "    # Set x-axes range\n",
    "    fig1.update_xaxes(title = 'Number of decided applications' , range=[0, 220000])\n",
    "    fig1.update_yaxes(title = 'Time taken')\n",
    "    \n",
    "    return fig1\n",
    "\n",
    "\n",
    "def country_bar_plot(top_15_countries):\n",
    "    fig  = go.Figure([go.Bar(x = top_15_countries['COUNTRY'],y=top_15_countries['Total applications'], marker_color = 'indianred')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 15 countries with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of perm applications',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "\n",
    "    return fig\n",
    "\n",
    "def visa_bar_plot(top_10_visa):\n",
    "    \n",
    "    fig  = go.Figure([go.Bar(x = top_10_visa['CLASS_OF_ADMISSION'],y=top_10_visa['Total applications'], marker_color = 'blue')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 10 class of admissions with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of class of admissions',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "        \n",
    "    return fig\n",
    "\n",
    "def top_employer(ax):\n",
    "    \n",
    "    #print(ax['Total_count'])\n",
    "\n",
    "    fig  = go.Figure([go.Bar(x = ax['Total_count'], y =ax['Employer Name'], marker_color = 'green',orientation='h')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top sponsoring employers ',\n",
    "                     yaxis_title = 'PERM applications',\n",
    "                     xaxis_title = 'Name of employers')\n",
    "                     #barmode = 'group'\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "def top_employer_by_year(emp_year):\n",
    "    fig = px.histogram(emp_year, x=\"Year\", y=\"CASE_STATUS\",\n",
    "             color='EMPLOYER_NAME', barmode='group',\n",
    "             #histfunc='avg',\n",
    "             height=500,title=\"Top Employers(sponsors) over the years\")\n",
    "\n",
    "    return fig\n",
    "\n",
    "def data_for_cases(header, total_cases, percent):\n",
    "    card_content =[\n",
    "        dbc.CardHeader(header),\n",
    "\n",
    "        dbc.CardBody(\n",
    "            [\n",
    "             dcc.Markdown(dangerously_allow_html =True,\n",
    "                     children = [\"{0}<br><sub>+{1}</sub></br>\".format(total_cases,percent)]\n",
    "             )\n",
    "            ]\n",
    "        )\n",
    "\n",
    "\n",
    "    ]\n",
    "    return card_content\n",
    "\n",
    "body_app =dbc.Container([\n",
    "    \n",
    "    dbc.Row(html.Marquee(\"India is the top conuntry in terms of certified cases\"), style ={'color':'green'}),\n",
    "    dbc.Row([\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Certified\",f'{certified:,}', f'{certified_perc:,}'), color='success', style={'text-align':'center'},inverse = True)),\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Denied\",f'{denied:,}', f'{denied_perc:,}'), color='danger', style={'text-align':'center'},inverse = True)),\n",
    "        #dbc.Col(dbc.Card(card_content, color='secondary', outline=True))\n",
    "        \n",
    "        \n",
    "    ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H5('Analysis & Visualizations'), style={'textAlign':'center','fontWeight':'bold','family':'georgia'})]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    #dbc.Row(dcc.Graph(id='line_plot', figure = line_plot(cases_status_df)), style ={'height':'450px'}),\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='line_plot', figure = line_plot(cases_status_df)), style ={'height':'450px'}),\n",
    "             #dbc.Col(html.Div(), style={'height':'450px'})\n",
    "            dbc.Col(dcc.Graph(id='lolipop-plot', figure = lolipop_plot(decesion_df_total)), style ={'height':'450px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Case status analysis by country with -- %'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='country_bar_plot', figure = country_bar_plot(top_15_countries)), style ={'height':'450px'}),\n",
    "             #dbc.Col(html.Div(), style={'height':'450px'})\n",
    "            dbc.Col([html.Div(id='dropdown_div', children =[\n",
    "                    dcc.Dropdown(id = 'country-dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in top_15_countries_perc['COUNTRY'].unique()],\n",
    "                        value = 'INDIA',\n",
    "                        placeholder='Select the country')],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='pie-plot')],style ={'height':'450px', 'width':'300px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Case status analysis by visa type with -- %'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='visa_bar_plot', figure = visa_bar_plot(top_10_visa)), style ={'height':'450px'}),\n",
    "             dbc.Col([html.Div(id='dropdown_div2', children =[\n",
    "                    dcc.Dropdown(id = 'visa-dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in top_10_visa_perc['CLASS_OF_ADMISSION'].unique()],\n",
    "                        value = 'H1B',\n",
    "                        placeholder='Select the visa type')],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='visa-pie-plot')],style ={'height':'450px','width':'300px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer', figure = top_employer(ax)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer_by_year', figure = top_employer_by_year(emp_year)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4(\"Case status analysis by education level and wage level\"))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col([html.Div(id='education_div', children =[\n",
    "                    dcc.RadioItems(id = 'certification_edu_ratio', #this is an input \n",
    "                        options = [{'label': 'Certified', 'value': 'Certified'},\n",
    "                                   {'label': 'Denied', 'value': 'Denied'}],\n",
    "                        value = 'Certified'),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='education_bar_plot')],style ={'height':'450px','width':'600px'})]),\n",
    "             \n",
    "             dbc.Col([html.Div(id='wages_div', children =[\n",
    "                    dcc.RadioItems(id = 'certification_wage_radio', #this is an input \n",
    "                        options = [{'label': 'Certified', 'value': 'Certified'},\n",
    "                                   {'label': 'Denied', 'value': 'Denied'}],\n",
    "                        value = 'Certified'),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='wage_bar_plot')],style ={'height':'450px','width':'600px'})])\n",
    "             \n",
    "             \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Prediction based on model'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(dcc.Markdown('''\n",
    "    #### Note: For your employer check following condition:\n",
    "    \n",
    "    \n",
    "    '1.If your employer are:    *COGNIZANT TECHNOLOGY SOLUTIONS US CORPORATION, MICROSOFT CORPORATION, INTEL CORPORATION,', \n",
    "                     AMAZON CORPORATE LLC, GOOGLE LLC, FACEBOOK INC., APPLE INC., INFOSYS LTD.,\n",
    "                'GOOGLE LLC, AMAZON.COM SERVICES, INC.* then select' ,  **Top 10 Employer**\n",
    "                \n",
    "                \n",
    "    2.If your employer are:  *CISCO SYSTEMS, INC.,\n",
    "                             'ORACLE AMERICA, INC.,\n",
    "                             'HCL AMERICA INC.,\n",
    "                            ' TATA CONSULTANCY SERVICES LIMITED,\n",
    "                            ' DELOITTE CONSULTING LLP,\n",
    "                             'QUALCOMM TECHNOLOGIES INC.,\n",
    "                            ' ERNST & YOUNG U.S. LLP,\n",
    "                             'JP MORGAN CHASE & CO,\n",
    "                             'WIPRO LIMITED,\n",
    "                             'Defender Services, Inc.* then select **Top 20 Employer**\n",
    "                             \n",
    "                             \n",
    "    3.If your employer are  *SALESFORCE.COM,\n",
    "                            ' HOUSE OF RAEFORD FARMS, INC.,\n",
    "                             'LINKEDIN CORPORATION,'\n",
    "                            ' IBM CORPORATION,'\n",
    "                            ' VMWARE, INC.,'\n",
    "                            ' KFORCE INC.'\n",
    "                            ' WAYNE FARMS LLC,'\n",
    "                            ' CAPGEMINI AMERICA, INC.,\n",
    "                             'YAHOO! INC.,\n",
    "                             'CAPITAL ONE SERVICES, LLC* then select **Top 30 Employer**\n",
    "                             \n",
    "                             \n",
    "    4.If your employer are not in any of above list select **Other Employer**'''\n",
    "                    \n",
    "                              \n",
    "                             ))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    ## try again\n",
    "    dbc.Row([dbc.Col([\n",
    "        dbc.Row([html.Div(html.H6('Select CITY where job is located'))]),\n",
    "        dbc.Row([html.Div(id='job_city', children =[\n",
    "                    dcc.Dropdown(id = 'job_city_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['JOB_CITY'].unique()],\n",
    "                        value = 'new york',\n",
    "                        placeholder='Select CITY where job is located')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                            ]),\n",
    "                      \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select STATE where job is located'))]),\n",
    "            dbc.Row([html.Div(id='job_state', children =[\n",
    "                    dcc.Dropdown(id = 'job_state_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['JOB_STATE'].unique()],\n",
    "                        value = 'NEW YO',\n",
    "                        placeholder='Select STATE where job is located ')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                            ]),\n",
    "                \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select education requirement needed for the job'))]),\n",
    "        dbc.Row([html.Div(id='required_education', children =[\n",
    "                    dcc.Dropdown(id = 'recquired_education_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['REQD_EDUCATION'].unique()],\n",
    "                        value = 'High School',\n",
    "                        placeholder='Select Required Education type for the job as input')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                        ]),\n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "     dbc.Row([dbc.Col([\n",
    "         dbc.Row([html.Div(html.H6('Select Country where you belong to!!'))]),\n",
    "        dbc.Row([html.Div(id='citizenship_type', children =[\n",
    "                    dcc.Dropdown(id = 'citizenship_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['CITIZENSHIP'].unique()],\n",
    "                        value = 'INDIA',\n",
    "                        placeholder='Select which country are you from?')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                         ]),\n",
    "                \n",
    "                      \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select which visa/admission you have'))]),\n",
    "        dbc.Row([html.Div(id='admission_type', children =[\n",
    "                    dcc.Dropdown(id = 'admission_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['ADMISSION_TYPE'].unique()],\n",
    "                        value = 'H1B',\n",
    "                        placeholder='Select your visa type as input')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                          ]),\n",
    "                \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select your highest education level'))]),\n",
    "        dbc.Row([html.Div(id='worker_education', children =[\n",
    "                    dcc.Dropdown(id = 'worker_education_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['WORKER_EDUCATION'].unique()],\n",
    "                        value = 'High School',\n",
    "                        placeholder='Select employee education')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                        ]),\n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col([\n",
    "        dbc.Row([html.Div(html.H6('Select if the job requires training or not'))]),\n",
    "        dbc.Row([html.Div(id='time_taken', children =[\n",
    "                    dcc.Dropdown(id = 'time_taken_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['TRAINING_REQD'].unique()],\n",
    "                        value = 'N',\n",
    "                        placeholder='Select Y for yes and N for no')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                        ]),\n",
    "                      \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select occupation that job belongs to'))]),\n",
    "        dbc.Row([html.Div(id='occupation', children =[\n",
    "                    dcc.Dropdown(id = 'occupation_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['OCCUPATION'].unique()],\n",
    "                        value = 'computer occupations',\n",
    "                        placeholder='Select job from dropdown')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                            ]),\n",
    "                \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select if the work major requires STEM or NON-STEM'))]),\n",
    "        dbc.Row([html.Div(id='related_major', children =[\n",
    "                    dcc.Dropdown(id = 'related_major_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['NEW_RELATED_MAJOR'].unique()],\n",
    "                        value = 'STEM Major',\n",
    "                        placeholder='Select if your major is STEM ')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                            ]),\n",
    "             \n",
    "        \n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    \n",
    "    dbc.Row([ dbc.Col([\n",
    "        dbc.Row([html.Div(html.H6('Select if your major is STEM or NON-STEM'))]),\n",
    "        dbc.Row([html.Div(id='worker_major', children =[\n",
    "                    dcc.Dropdown(id = 'worker_major_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['NEW_WORKER_MAJOR'].unique()],\n",
    "                        value = 'STEM Major',\n",
    "                        placeholder='Select if your major is STEM')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                        ]),\n",
    "                      \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select if your employer is Top-10, Top20 ,Top30 or Other. Refer to above!!'))]),\n",
    "        dbc.Row([html.Div(id='employer_name', children =[\n",
    "                    dcc.Dropdown(id = 'employer_name_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['NEW_EMPLOYER_NAME'].unique()],\n",
    "                        value = 'Top 10 Employer',\n",
    "                        placeholder='Select your employer')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                        ]),\n",
    "                \n",
    "            dbc.Col([\n",
    "                dbc.Row([html.Div(html.H6('Select the wage offered category'))]),\n",
    "        dbc.Row([html.Div(id='wage_offered', children =[\n",
    "                    dcc.Dropdown(id = 'wage_offered_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['WAGE_OFFERED'].unique()],\n",
    "                        value = 'Below 30000',\n",
    "                        placeholder='Select your wage category')],style = {'width':'100%', 'display':'inline-block'})])\n",
    "                        ]),\n",
    "             \n",
    "        \n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    ## try finish\n",
    "    \n",
    "   \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([\n",
    "        dbc.Col(dbc.Button(id='generate_ans', children ='Generate Prediction', color ='dark', n_clicks =0),\n",
    "                    width={'size':15, 'offset':3}),\n",
    "        \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([\n",
    "        dbc.Col(\n",
    "            dcc.Loading(\n",
    "                id='Load-ans',\n",
    "                type='default',\n",
    "                children =html.Div(id='div_answer', style={'textAlign':'center', 'color':'black','height':'50px', 'fontWeight':'bold'})\n",
    "            )\n",
    "        ),\n",
    "        \n",
    "        \n",
    "    ])\n",
    "    \n",
    "    \n",
    "],fluid=True)\n",
    "navbar = dbc.Navbar(id='navbar', children =[\n",
    "    \n",
    "    html.A(\n",
    "    dbc.Row([\n",
    "        dbc.Col(html.Img(src = DOL_Logo, height= \"70px\")),\n",
    "        dbc.Col(\n",
    "            dbc.NavbarBrand(\"Perm Cases Tracker\", style ={'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'})\n",
    "        \n",
    "        )\n",
    "        \n",
    "        ], align =\"center\"), #aligns title to center\n",
    "        #no_gutters=True),\n",
    "    href ='/'\n",
    "    ),\n",
    "    dbc.Button(id ='button', children = 'Clicke Me!', color ='primary', className ='ml-auto',href='/')\n",
    "    \n",
    "    \n",
    "    ])\n",
    "    \n",
    "    \n",
    "    \n",
    "app.layout =html.Div(id ='parent', children=[navbar,body_app])\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id ='div_answer',component_property = 'children'),\n",
    "             [Input(component_id='generate_ans',component_property='n_clicks')],\n",
    "             [State(component_id='job_city_dropdown',component_property='value'),\n",
    "              State('job_state_dropdown','value'),\n",
    "              State('recquired_education_type_dropdown','value'),\n",
    "              State('citizenship_type_dropdown','value'),\n",
    "              State('admission_type_dropdown','value'),\n",
    "              State('worker_education_type_dropdown','value'),\n",
    "              State('time_taken_dropdown','value'),\n",
    "              State('occupation_dropdown','value'),\n",
    "              State('related_major_dropdown','value'),\n",
    "              State('worker_major_type_dropdown','value'),\n",
    "              State('employer_name_dropdown','value'),\n",
    "              State('wage_offered_dropdown','value')])\n",
    "\n",
    "def show_answer(clicks,job_city,job_state,required_education,citizenship_type,admission_type,worker_education,\n",
    "                training_reqd, occupation,related_major, worker_major, employer_name, wage_offered):\n",
    "    \n",
    "    print(\"*****************############################\")\n",
    "    print(\"Job_city is\",job_city)\n",
    "                \n",
    "    if clicks>0:\n",
    "        value_1 = dict_list[1][str(job_city)]\n",
    "        value_2 = dict_list[2][str(job_state)]\n",
    "        value_3=dict_list[3][str(required_education)]\n",
    "        value_4=dict_list[4][str(citizenship_type)]\n",
    "        value_5= dict_list[5][str(admission_type)]\n",
    "        value_6= dict_list[6][str(worker_education)]\n",
    "        value_7= dict_list[7][str(training_reqd)]\n",
    "        value_8= dict_list[8][str(occupation)]\n",
    "        value_9= dict_list[9][str(related_major)]\n",
    "        value_10= dict_list[10][str(worker_major)]\n",
    "        value_11= dict_list[11][str(employer_name)]\n",
    "        value_12= dict_list[12][str(wage_offered)]\n",
    "        \n",
    "        test_series =pd.Series({'JOB_CITY':value_1,\n",
    "                       'JOB_STATE':value_2,\n",
    "                       'REQD_EDUCATION':value_3,\n",
    "                       'CITIZENSHIP':value_4,\n",
    "                       'ADMISSION_TYPE':value_5,\n",
    "                       'WORKER_EDUCATION':value_6,\n",
    "                       'TRAINING_REQD':value_7,\n",
    "                       'OCCUPATION':value_8,\n",
    "                       'NEW_RELATED_MAJOR':value_9,\n",
    "                       'NEW_WORKER_MAJOR':value_10,\n",
    "                       'NEW_EMPLOYER_NAME':value_11,\n",
    "                       'WAGE_OFFERED':value_12})\n",
    "        \n",
    "        pickle_in = open(\"best_model_file\",'rb')\n",
    "        model = pickle.load(pickle_in)\n",
    "        \n",
    "        prediction_model = model.predict([test_series])\n",
    "        \n",
    "        print(prediction_model)\n",
    "        \n",
    "        \n",
    "        for i in range(len(prediction_model)):\n",
    "            if prediction_model[i] == 1:\n",
    "              ans = \"Congratulations!! based on the criterias you entered, there is likelihood that your case will be Certified\"\n",
    "            else:\n",
    "              ans = \"Unfortunately based on the criterias you entered, there is likelihood that your case will be Denied\"\n",
    "            \n",
    "        print(ans)\n",
    "        return ans\n",
    "    \n",
    "    else:\n",
    "        return\"\"\n",
    "\n",
    "    \n",
    "@app.callback(Output(component_id='education_bar_plot', component_property ='figure'),\n",
    "                [Input(component_id='certification_edu_ratio', component_property ='value')])\n",
    "\n",
    "def generate_education_bar(status):\n",
    "    \n",
    "    colors = ['blueviolet','blue','lightskyblue','lightsteelblue','mediumblue','aqua','midnightblue']\n",
    "    \n",
    "    if status =='Certified':\n",
    "        \n",
    "        dff_education_approved = dff_education_total.sort_values(by ='Approved_Ratio', ascending =False)\n",
    "\n",
    "\n",
    "        fig  = go.Figure([go.Bar(x = dff_education_approved['REQD_EDUCATION'],y=dff_education_approved['Approved_Ratio'], marker_color =colors)])\n",
    "                         #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "        fig.update_layout(title = 'Certified cases ratio by education level',\n",
    "                         xaxis_title = 'Education level',\n",
    "                         yaxis_title = 'Approved ratio')\n",
    "                         #barmode = 'group')\n",
    "\n",
    "        return fig\n",
    "    \n",
    "    elif status =='Denied':\n",
    "        dff_education_denied = dff_education_total.sort_values(by ='Denied_Ratio', ascending =False)\n",
    "\n",
    "\n",
    "        fig  = go.Figure([go.Bar(x = dff_education_denied['REQD_EDUCATION'],y=dff_education_denied['Denied_Ratio'], marker_color = colors)])\n",
    "                         #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "        fig.update_layout(title = 'Denied cases ratio by education level',\n",
    "                         xaxis_title = 'Education level',\n",
    "                         yaxis_title = 'Denied ratio')\n",
    "                         #barmode = 'group')\n",
    "\n",
    "        return fig\n",
    "\n",
    "@app.callback(Output(component_id='wage_bar_plot', component_property ='figure'),\n",
    "                [Input(component_id='certification_wage_radio', component_property ='value')])\n",
    "\n",
    "def generate_wage_plot(status_wage):\n",
    "    \n",
    "    colors = ['blueviolet','blue','lightskyblue','lightsteelblue','mediumblue','aqua','midnightblue']\n",
    "    \n",
    "    print(status_wage)\n",
    "    if status_wage == 'Certified':\n",
    "        \n",
    "        dff_wages_approved = dff_wages_total.sort_values(by ='Approved_Ratio', ascending =False)\n",
    "        #dff_wages_approved\n",
    "\n",
    "        fig  = go.Figure([go.Bar(x = dff_wages_approved['WAGE_OFFERED'],y=dff_wages_approved['Approved_Ratio'], marker_color = colors)])\n",
    "                         #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "        fig.update_layout(title = 'Approved cases ratio by wage category',\n",
    "                         xaxis_title = 'Wages cateogry',\n",
    "                         yaxis_title = 'Approved ratio')\n",
    "                         #barmode = 'group')\n",
    "\n",
    "        return fig\n",
    "    \n",
    "    elif status_wage== 'Denied':\n",
    "        \n",
    "        dff_wages_denied = dff_wages_total.sort_values(by ='Denied_Ratio', ascending =False)\n",
    "        \n",
    "        fig  = go.Figure([go.Bar(x = dff_wages_denied['WAGE_OFFERED'],y=dff_wages_denied['Denied_Ratio'], marker_color = colors)])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "        fig.update_layout(title = 'Denied cases ratio by wage category',\n",
    "                         xaxis_title = 'Wages cateogry',\n",
    "                         yaxis_title = 'Denied ratio')\n",
    "                         #barmode = 'group')\n",
    "\n",
    "        return fig\n",
    "        \n",
    "\n",
    "        \n",
    "        \n",
    "    \n",
    "\n",
    "@app.callback(Output(component_id='pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='country-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_pie(country):\n",
    "    \n",
    "    print(\"Country name is \", country)\n",
    "    \n",
    "    df_final = top_15_countries_perc.loc[top_15_countries_perc['COUNTRY']=='{}'.format(country)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id='visa-pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='visa-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_visa_pie(visa_dropdown):\n",
    "    \n",
    "    df_final = top_10_visa_perc.loc[top_10_visa_perc['CLASS_OF_ADMISSION']=='{}'.format(visa_dropdown)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "    \n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run_server()\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "bb2fea50",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'time_taken'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   3360\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3361\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3362\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/pandas/_libs/index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/pandas/_libs/index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas/_libs/hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32mpandas/_libs/hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'time_taken'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_15433/3473736198.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m    227\u001b[0m     dbc.Row([dbc.Col([html.Div(id='time_taken', children =[\n\u001b[1;32m    228\u001b[0m                     dcc.Dropdown(id = 'time_taken_dropdown', #this is an input \n\u001b[0;32m--> 229\u001b[0;31m                         \u001b[0moptions\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m'label'\u001b[0m \u001b[0;34m:\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'value'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0mi\u001b[0m \u001b[0;34m}\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mdff_final\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'time_taken'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munique\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    230\u001b[0m                         \u001b[0mvalue\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'at least 6 months'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    231\u001b[0m                         placeholder='Select how long have you been waiting for?')],style = {'width':'100%', 'display':'inline-block'})]),\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   3456\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3457\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3458\u001b[0;31m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3459\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3460\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   3361\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3362\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3363\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3364\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3365\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mis_scalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0misna\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhasnans\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 'time_taken'"
     ]
    }
   ],
   "source": [
    "\n",
    "#Navigation bar\n",
    "\n",
    "app = dash.Dash(external_stylesheets=[ dbc.themes.FLATLY],)\n",
    "\n",
    "#\"https://en.wikipedia.org/wiki/United_States_Department_of_Labor#/media/File:Seal_of_the_United_States_Department_of_Labor.svg\" #reference: https://twitter.com/USDOL\n",
    "\n",
    "DOL_Logo = mpimg.imread('/Users/ujjwaloli/Desktop/Capstone Project/DashboardLogo.png')\n",
    "\n",
    "\n",
    "total = len(df.index)\n",
    "certified = df[df['CASE_STATUS']=='Certified']['CASE_STATUS'].count()\n",
    "denied = df[df['CASE_STATUS']=='Denied']['CASE_STATUS'].count()\n",
    "certified_perc = round(certified/total *100,2)\n",
    "denied_perc = round(denied/total *100,2)\n",
    "\n",
    "def line_plot(df):\n",
    "    fig = go.Figure(data = [go.Scatter(y =cases_status_df['Approved_Case'], x=cases_status_df['Year'], line=dict(color ='firebrick', width =4),text = cases_status_df['Approved_Case'] , name ='Cases Approved'),\n",
    "                           go.Scatter(y =cases_status_df['Denied_Case'], x=cases_status_df['Year'], line=dict(color ='blue', width =4),text = cases_status_df['Denied_Case'] , name ='Cases Denied')])\n",
    "    \n",
    "    \n",
    "    fig.update_layout(title='Analysis of Approved cases and Denied Cases over the years',\n",
    "                         xaxis_title='Year',\n",
    "                         yaxis_title='No of cases',\n",
    "                         margin=dict(l =4,r=4,t=30,b=4))\n",
    "    return fig\n",
    "\n",
    "def country_bar_plot(top_15_countries):\n",
    "    fig  = go.Figure([go.Bar(x = top_15_countries['COUNTRY'],y=top_15_countries['Total applications'], marker_color = 'indianred')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 15 countries with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of perm applications',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "\n",
    "    return fig\n",
    "\n",
    "def visa_bar_plot(top_10_visa):\n",
    "    \n",
    "    fig  = go.Figure([go.Bar(x = top_10_visa['CLASS_OF_ADMISSION'],y=top_10_visa['Total applications'], marker_color = 'blue')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 10 class of admissions with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of class of admissions',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "        \n",
    "    return fig\n",
    "\n",
    "def top_employer(ax):\n",
    "    \n",
    "    #print(ax['Total_count'])\n",
    "\n",
    "    fig  = go.Figure([go.Bar(x = ax['Total_count'], y =ax['Employer Name'], marker_color = 'green',orientation='h')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top sponsoring employers ',\n",
    "                     yaxis_title = 'PERM applications',\n",
    "                     xaxis_title = 'Name of employers')\n",
    "                     #barmode = 'group'\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "def top_employer_by_year(emp_year):\n",
    "    fig = px.histogram(emp_year, x=\"Year\", y=\"CASE_STATUS\",\n",
    "             color='EMPLOYER_NAME', barmode='group',\n",
    "             #histfunc='avg',\n",
    "             height=500,title=\"Top Employers(sponsors) over the years\")\n",
    "\n",
    "    return fig\n",
    "\n",
    "def data_for_cases(header, total_cases, percent):\n",
    "    card_content =[\n",
    "        dbc.CardHeader(header),\n",
    "\n",
    "        dbc.CardBody(\n",
    "            [\n",
    "             dcc.Markdown(dangerously_allow_html =True,\n",
    "                     children = [\"{0}<br><sub>+{1}</sub></br>\".format(total_cases,percent)]\n",
    "             )\n",
    "            ]\n",
    "        )\n",
    "\n",
    "\n",
    "    ]\n",
    "    return card_content\n",
    "\n",
    "body_app =dbc.Container([\n",
    "    \n",
    "    dbc.Row(html.Marquee(\"India is the top conuntry in terms of certified cases\"), style ={'color':'green'}),\n",
    "    dbc.Row([\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Certified\",f'{certified:,}', f'{certified_perc:,}'), color='success', style={'text-align':'center'},inverse = True)),\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Denied\",f'{denied:,}', f'{denied_perc:,}'), color='danger', style={'text-align':'center'},inverse = True)),\n",
    "        #dbc.Col(dbc.Card(card_content, color='secondary', outline=True))\n",
    "        \n",
    "        \n",
    "    ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H5('Analysis & Visualizations'), style={'textAlign':'center','fontWeight':'bold','family':'georgia'})]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='line_plot', figure = line_plot(cases_status_df)), style ={'height':'450px'}), \n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Case status analysis by country with -- %'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='country_bar_plot', figure = country_bar_plot(top_15_countries)), style ={'height':'450px'}),\n",
    "             #dbc.Col(html.Div(), style={'height':'450px'})\n",
    "            dbc.Col([html.Div(id='dropdown_div', children =[\n",
    "                    dcc.Dropdown(id = 'country-dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in top_15_countries_perc['COUNTRY'].unique()],\n",
    "                        value = 'INDIA',\n",
    "                        placeholder='Select the country')],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='pie-plot')],style ={'height':'450px', 'width':'300px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Case status analysis by visa type with -- %'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='visa_bar_plot', figure = visa_bar_plot(top_10_visa)), style ={'height':'450px'}),\n",
    "             dbc.Col([html.Div(id='dropdown_div2', children =[\n",
    "                    dcc.Dropdown(id = 'visa-dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in top_10_visa_perc['CLASS_OF_ADMISSION'].unique()],\n",
    "                        value = 'H1B',\n",
    "                        placeholder='Select the visa type')],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='visa-pie-plot')],style ={'height':'450px','width':'300px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer', figure = top_employer(ax)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer_by_year', figure = top_employer_by_year(emp_year)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Prediction based on model'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    ## try again\n",
    "    dbc.Row([dbc.Col([html.Div(id='job_city', children =[\n",
    "                    dcc.Dropdown(id = 'job_city_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['JOB_CITY'].unique()],\n",
    "                        value = 'new york',\n",
    "                        placeholder='Select CITY where job is located')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                      \n",
    "            dbc.Col([html.Div(id='job_state', children =[\n",
    "                    dcc.Dropdown(id = 'job_state_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['JOB_STATE'].unique()],\n",
    "                        value = 'NEW YO',\n",
    "                        placeholder='Select STATE where job is located ')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                \n",
    "            dbc.Col([html.Div(id='required_education', children =[\n",
    "                    dcc.Dropdown(id = 'recquired_education_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['REQD_EDUCATION'].unique()],\n",
    "                        value = 'High School',\n",
    "                        placeholder='Select Required Education type for the job as input')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "     dbc.Row([dbc.Col([html.Div(id='citizenship_type', children =[\n",
    "                    dcc.Dropdown(id = 'citizenship_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['CITIZENSHIP'].unique()],\n",
    "                        value = 'INDIA',\n",
    "                        placeholder='Select which country are you from?')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                \n",
    "                      \n",
    "            dbc.Col([html.Div(id='admission_type', children =[\n",
    "                    dcc.Dropdown(id = 'admission_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['ADMISSION_TYPE'].unique()],\n",
    "                        value = 'H1B',\n",
    "                        placeholder='Select your visa type as input')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                \n",
    "            dbc.Col([html.Div(id='worker_education', children =[\n",
    "                    dcc.Dropdown(id = 'worker_education_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['WORKER_EDUCATION'].unique()],\n",
    "                        value = 'High School',\n",
    "                        placeholder='Select employee education')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col([html.Div(id='time_taken', children =[\n",
    "                    dcc.Dropdown(id = 'time_taken_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['time_taken'].unique()],\n",
    "                        value = 'at least 6 months',\n",
    "                        placeholder='Select how long have you been waiting for?')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                      \n",
    "            dbc.Col([html.Div(id='occupation', children =[\n",
    "                    dcc.Dropdown(id = 'occupation_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['OCCUPATION'].unique()],\n",
    "                        value = 'computer occupations',\n",
    "                        placeholder='Select job from dropdown')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                \n",
    "            dbc.Col([html.Div(id='related_major', children =[\n",
    "                    dcc.Dropdown(id = 'related_major_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['NEW_RELATED_MAJOR'].unique()],\n",
    "                        value = 'STEM Major',\n",
    "                        placeholder='Select if your major is STEM ')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "             \n",
    "        \n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    \n",
    "    dbc.Row([ dbc.Col([html.Div(id='worker_major', children =[\n",
    "                    dcc.Dropdown(id = 'worker_major_type_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['NEW_WORKER_MAJOR'].unique()],\n",
    "                        value = 'STEM Major',\n",
    "                        placeholder='Select if your major is STEM')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                      \n",
    "            dbc.Col([html.Div(id='employer_name', children =[\n",
    "                    dcc.Dropdown(id = 'employer_name_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['NEW_EMPLOYER_NAME'].unique()],\n",
    "                        value = 'Top 10 Employer',\n",
    "                        placeholder='Select your employer')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "                \n",
    "            dbc.Col([html.Div(id='wage_offered', children =[\n",
    "                    dcc.Dropdown(id = 'wage_offered_dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in dff_final['WAGE_OFFERED'].unique()],\n",
    "                        value = 'Below 30000',\n",
    "                        placeholder='Select your wage category')],style = {'width':'100%', 'display':'inline-block'})]),\n",
    "             \n",
    "        \n",
    "            \n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    ## try finish\n",
    "    \n",
    "   \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([\n",
    "        dbc.Col(dbc.Button(id='generate_ans', children ='Generate Answer', color ='dark', n_clicks =0),\n",
    "                    width={'size':15, 'offset':3}),\n",
    "        \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([\n",
    "        dbc.Col(\n",
    "            dcc.Loading(\n",
    "                id='Load-ans',\n",
    "                type='default',\n",
    "                children =html.Div(id='div_answer', style={'textAlign':'center', 'color':'black','height':'50px', 'fontWeight':'bold'})\n",
    "            )\n",
    "        ),\n",
    "        \n",
    "        \n",
    "    ])\n",
    "    \n",
    "    \n",
    "],fluid=True)\n",
    "navbar = dbc.Navbar(id='navbar', children =[\n",
    "    \n",
    "    html.A(\n",
    "    dbc.Row([\n",
    "        dbc.Col(html.Img(src = DOL_Logo, height= \"70px\")),\n",
    "        dbc.Col(\n",
    "            dbc.NavbarBrand(\"Perm Cases Tracker\", style ={'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'})\n",
    "        \n",
    "        )\n",
    "        \n",
    "        ], align =\"center\"), #aligns title to center\n",
    "        #no_gutters=True),\n",
    "    href ='/'\n",
    "    ),\n",
    "    dbc.Button(id ='button', children = 'Clicke Me!', color ='primary', className ='ml-auto',href='/')\n",
    "    \n",
    "    \n",
    "    ])\n",
    "    \n",
    "    \n",
    "    \n",
    "app.layout =html.Div(id ='parent', children=[navbar,body_app])\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id ='div_answer',component_property = 'children'),\n",
    "             [Input(component_id='generate_ans',component_property='n_clicks')],\n",
    "             [State(component_id='job_city_dropdown',component_property='value'),\n",
    "              State('job_state_dropdown','value'),\n",
    "              State('recquired_education_type_dropdown','value'),\n",
    "              State('citizenship_type_dropdown','value'),\n",
    "              State('admission_type_dropdown','value'),\n",
    "              State('worker_education_type_dropdown','value'),\n",
    "              State('time_taken_dropdown','value'),\n",
    "              State('occupation_dropdown','value'),\n",
    "              State('related_major_dropdown','value'),\n",
    "              State('worker_major_type_dropdown','value'),\n",
    "              State('employer_name_dropdown','value'),\n",
    "              State('wage_offered_dropdown','value')])\n",
    "\n",
    "def show_answer(clicks,job_city,job_state,required_education,citizenship_type,admission_type,worker_education,\n",
    "                time_taken, occupation,related_major, worker_major, employer_name, wage_offered):\n",
    "    \n",
    "    print(\"*****************############################\")\n",
    "    print(\"Job_city is\",job_city)\n",
    "                \n",
    "    if clicks>0:\n",
    "        value_1 = dict_list[1][str(job_city)]\n",
    "        value_2 = dict_list[2][str(job_state)]\n",
    "        value_3=dict_list[3][str(required_education)]\n",
    "        value_4=dict_list[4][str(citizenship_type)]\n",
    "        value_5= dict_list[5][str(admission_type)]\n",
    "        value_6= dict_list[6][str(worker_education)]\n",
    "        value_7= dict_list[7][str(time_taken)]\n",
    "        value_8= dict_list[8][str(occupation)]\n",
    "        value_9= dict_list[9][str(related_major)]\n",
    "        value_10= dict_list[10][str(worker_major)]\n",
    "        value_11= dict_list[11][str(employer_name)]\n",
    "        value_12= dict_list[12][str(wage_offered)]\n",
    "        \n",
    "        test_series =pd.Series({'JOB_CITY':value_1,\n",
    "                       'JOB_STATE':value_2,\n",
    "                       'REQD_EDUCATION':value_3,\n",
    "                       'CITIZENSHIP':value_4,\n",
    "                       'ADMISSION_TYPE':value_5,\n",
    "                       'WORKER_EDUCATION':value_6,\n",
    "                       'time_taken':value_7,\n",
    "                       'OCCUPATION':value_8,\n",
    "                       'NEW_RELATED_MAJOR':value_9,\n",
    "                       'NEW_WORKER_MAJOR':value_10,\n",
    "                       'NEW_EMPLOYER_NAME':value_11,\n",
    "                       'WAGE_OFFERED':value_12})\n",
    "        \n",
    "        pickle_in = open(\"best_model_file\",'rb')\n",
    "        model = pickle.load(pickle_in)\n",
    "        \n",
    "        prediction_model = model.predict([test_series])\n",
    "        \n",
    "        print(prediction_model)\n",
    "        \n",
    "        \n",
    "        for i in range(len(prediction_model)):\n",
    "            if prediction_model[i] == 1:\n",
    "              ans = \"Certified\"\n",
    "            else:\n",
    "              ans = \"Denied\"\n",
    "            \n",
    "        print(ans)\n",
    "        return ans\n",
    "    \n",
    "    else:\n",
    "        return\"\"\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id='pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='country-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_pie(country):\n",
    "    \n",
    "    print(\"Country name is \", country)\n",
    "    \n",
    "    df_final = top_15_countries_perc.loc[top_15_countries_perc['COUNTRY']=='{}'.format(country)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id='visa-pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='visa-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_visa_pie(visa_dropdown):\n",
    "    \n",
    "    df_final = top_10_visa_perc.loc[top_10_visa_perc['CLASS_OF_ADMISSION']=='{}'.format(visa_dropdown)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "    \n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run_server(debug = False)\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dbf28982",
   "metadata": {},
   "source": [
    "## working dashboard finished"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "433b430a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      "Dash is running on http://127.0.0.1:8050/\n",
      "\n",
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "ename": "OSError",
     "evalue": "[Errno 9] Bad file descriptor",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mOSError\u001b[0m                                   Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/w7/tznrgnyd2xz88fgkw3yq2zwh0000gn/T/ipykernel_23195/393863903.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m    216\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    217\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"__main__\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 218\u001b[0;31m     \u001b[0mapp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun_server\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    219\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    220\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/dash/dash.py\u001b[0m in \u001b[0;36mrun_server\u001b[0;34m(self, host, port, proxy, debug, dev_tools_ui, dev_tools_props_check, dev_tools_serve_dev_bundles, dev_tools_hot_reload, dev_tools_hot_reload_interval, dev_tools_hot_reload_watch_interval, dev_tools_hot_reload_max_retry, dev_tools_silence_routes_logging, dev_tools_prune_errors, **flask_run_options)\u001b[0m\n\u001b[1;32m   2045\u001b[0m                     \u001b[0mextra_files\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2046\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2047\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mserver\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhost\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mhost\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mport\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mport\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdebug\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdebug\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mflask_run_options\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/flask/app.py\u001b[0m in \u001b[0;36mrun\u001b[0;34m(self, host, port, debug, load_dotenv, **options)\u001b[0m\n\u001b[1;32m    988\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    989\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 990\u001b[0;31m             \u001b[0mrun_simple\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhost\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mport\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0moptions\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    991\u001b[0m         \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    992\u001b[0m             \u001b[0;31m# reset the first request information if the development server\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/werkzeug/serving.py\u001b[0m in \u001b[0;36mrun_simple\u001b[0;34m(hostname, port, application, use_reloader, use_debugger, use_evalex, extra_files, exclude_patterns, reloader_interval, reloader_type, threaded, processes, request_handler, static_files, passthrough_errors, ssl_context)\u001b[0m\n\u001b[1;32m   1008\u001b[0m         )\n\u001b[1;32m   1009\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1010\u001b[0;31m         \u001b[0minner\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1011\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1012\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/werkzeug/serving.py\u001b[0m in \u001b[0;36minner\u001b[0;34m()\u001b[0m\n\u001b[1;32m    948\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mLookupError\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    949\u001b[0m             \u001b[0mfd\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 950\u001b[0;31m         srv = make_server(\n\u001b[0m\u001b[1;32m    951\u001b[0m             \u001b[0mhostname\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    952\u001b[0m             \u001b[0mport\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/werkzeug/serving.py\u001b[0m in \u001b[0;36mmake_server\u001b[0;34m(host, port, app, threaded, processes, request_handler, passthrough_errors, ssl_context, fd)\u001b[0m\n\u001b[1;32m    780\u001b[0m         \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"cannot have a multithreaded and multi process server.\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    781\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mthreaded\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 782\u001b[0;31m         return ThreadedWSGIServer(\n\u001b[0m\u001b[1;32m    783\u001b[0m             \u001b[0mhost\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mport\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mapp\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrequest_handler\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpassthrough_errors\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mssl_context\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfd\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mfd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    784\u001b[0m         )\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/site-packages/werkzeug/serving.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, host, port, app, handler, passthrough_errors, ssl_context, fd)\u001b[0m\n\u001b[1;32m    674\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    675\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mfd\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 676\u001b[0;31m             \u001b[0mreal_sock\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msocket\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfromfd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfd\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0maddress_family\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msocket\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSOCK_STREAM\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    677\u001b[0m             \u001b[0mport\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    678\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/socket.py\u001b[0m in \u001b[0;36mfromfd\u001b[0;34m(fd, family, type, proto)\u001b[0m\n\u001b[1;32m    542\u001b[0m     \u001b[0mdescriptor\u001b[0m\u001b[0;34m.\u001b[0m  \u001b[0mThe\u001b[0m \u001b[0mremaining\u001b[0m \u001b[0marguments\u001b[0m \u001b[0mare\u001b[0m \u001b[0mthe\u001b[0m \u001b[0msame\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0msocket\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    543\u001b[0m     \"\"\"\n\u001b[0;32m--> 544\u001b[0;31m     \u001b[0mnfd\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdup\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfd\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    545\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0msocket\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfamily\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mproto\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnfd\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    546\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mOSError\u001b[0m: [Errno 9] Bad file descriptor"
     ]
    }
   ],
   "source": [
    "\n",
    "#Navigation bar\n",
    "\n",
    "app = dash.Dash(external_stylesheets=[ dbc.themes.FLATLY],)\n",
    "\n",
    "#\"https://en.wikipedia.org/wiki/United_States_Department_of_Labor#/media/File:Seal_of_the_United_States_Department_of_Labor.svg\" #reference: https://twitter.com/USDOL\n",
    "\n",
    "DOL_Logo = mpimg.imread('/Users/ujjwaloli/Desktop/Capstone Project/DashboardLogo.png')\n",
    "\n",
    "\n",
    "total = len(df.index)\n",
    "certified = df[df['CASE_STATUS']=='Certified']['CASE_STATUS'].count()\n",
    "denied = df[df['CASE_STATUS']=='Denied']['CASE_STATUS'].count()\n",
    "certified_perc = round(certified/total *100,2)\n",
    "denied_perc = round(denied/total *100,2)\n",
    "\n",
    "def line_plot(df):\n",
    "    fig = go.Figure(data = [go.Scatter(y =cases_status_df['Approved_Case'], x=cases_status_df['Year'], line=dict(color ='firebrick', width =4),text = cases_status_df['Approved_Case'] , name ='Cases Approved'),\n",
    "                           go.Scatter(y =cases_status_df['Denied_Case'], x=cases_status_df['Year'], line=dict(color ='blue', width =4),text = cases_status_df['Denied_Case'] , name ='Cases Denied')])\n",
    "    \n",
    "    \n",
    "    fig.update_layout(title='Analysis of Approved cases and Denied Cases over the years',\n",
    "                         xaxis_title='Year',\n",
    "                         yaxis_title='No of cases',\n",
    "                         margin=dict(l =4,r=4,t=30,b=4))\n",
    "    return fig\n",
    "\n",
    "def country_bar_plot(top_15_countries):\n",
    "    fig  = go.Figure([go.Bar(x = top_15_countries['COUNTRY'],y=top_15_countries['Total applications'], marker_color = 'indianred')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 15 countries with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of perm applications',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "\n",
    "    return fig\n",
    "\n",
    "def visa_bar_plot(top_10_visa):\n",
    "    \n",
    "    fig  = go.Figure([go.Bar(x = top_10_visa['CLASS_OF_ADMISSION'],y=top_10_visa['Total applications'], marker_color = 'blue')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 10 class of admissions with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of class of admissions',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "        \n",
    "    return fig\n",
    "\n",
    "def top_employer(ax):\n",
    "    \n",
    "    #print(ax['Total_count'])\n",
    "\n",
    "    fig  = go.Figure([go.Bar(x = ax['Total_count'], y =ax['Employer Name'], marker_color = 'green',orientation='h')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top sponsoring employers ',\n",
    "                     yaxis_title = 'PERM applications',\n",
    "                     xaxis_title = 'Name of employers')\n",
    "                     #barmode = 'group'\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "def top_employer_by_year(emp_year):\n",
    "    fig = px.histogram(emp_year, x=\"Year\", y=\"CASE_STATUS\",\n",
    "             color='EMPLOYER_NAME', barmode='group',\n",
    "             #histfunc='avg',\n",
    "             height=500,title=\"Top Employers(sponsors) over the years\")\n",
    "\n",
    "    return fig\n",
    "\n",
    "def data_for_cases(header, total_cases, percent):\n",
    "    card_content =[\n",
    "        dbc.CardHeader(header),\n",
    "\n",
    "        dbc.CardBody(\n",
    "            [\n",
    "             dcc.Markdown(dangerously_allow_html =True,\n",
    "                     children = [\"{0}<br><sub>+{1}</sub></br>\".format(total_cases,percent)]\n",
    "             )\n",
    "            ]\n",
    "        )\n",
    "\n",
    "\n",
    "    ]\n",
    "    return card_content\n",
    "\n",
    "body_app =dbc.Container([\n",
    "    \n",
    "    dbc.Row(html.Marquee(\"India is the top conuntry in terms of certified cases\"), style ={'color':'green'}),\n",
    "    dbc.Row([\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Certified\",f'{certified:,}', f'{certified_perc:,}'), color='success', style={'text-align':'center'},inverse = True)),\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Denied\",f'{denied:,}', f'{denied_perc:,}'), color='danger', style={'text-align':'center'},inverse = True)),\n",
    "        #dbc.Col(dbc.Card(card_content, color='secondary', outline=True))\n",
    "        \n",
    "        \n",
    "    ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H5('Analysis & Visualizations'), style={'textAlign':'center','fontWeight':'bold','family':'georgia'})]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='line_plot', figure = line_plot(cases_status_df)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Case status analysis by country with -- %'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='country_bar_plot', figure = country_bar_plot(top_15_countries)), style ={'height':'450px'}),\n",
    "             #dbc.Col(html.Div(), style={'height':'450px'})\n",
    "            dbc.Col([html.Div(id='dropdown_div', children =[\n",
    "                    dcc.Dropdown(id = 'country-dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in top_15_countries_perc['COUNTRY'].unique()],\n",
    "                        value = 'INDIA',\n",
    "                        placeholder='Select the country')],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='pie-plot')],style ={'height':'450px', 'width':'300px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H4('Case status analysis by visa type with -- %'))]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='visa_bar_plot', figure = visa_bar_plot(top_10_visa)), style ={'height':'450px'}),\n",
    "             dbc.Col([html.Div(id='dropdown_div2', children =[\n",
    "                    dcc.Dropdown(id = 'visa-dropdown', #this is an input \n",
    "                        options = [{'label' :i, 'value':i } for i in top_10_visa_perc['CLASS_OF_ADMISSION'].unique()],\n",
    "                        value = 'H1B',\n",
    "                        placeholder='Select the visa type')],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                         \n",
    "                     #html.Div(id ='pie-chart', children=[ #this is an output\n",
    "                     dcc.Graph(id ='visa-pie-plot')],style ={'height':'450px','width':'300px'})\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer', figure = top_employer(ax)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer_by_year', figure = top_employer_by_year(emp_year)), style ={'height':'450px'})\n",
    "    \n",
    "    \n",
    "],fluid=True)\n",
    "navbar = dbc.Navbar(id='navbar', children =[\n",
    "    \n",
    "    html.A(\n",
    "    dbc.Row([\n",
    "        dbc.Col(html.Img(src = DOL_Logo, height= \"70px\")),\n",
    "        dbc.Col(\n",
    "            dbc.NavbarBrand(\"Perm Cases Tracker\", style ={'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'})\n",
    "        \n",
    "        )\n",
    "        \n",
    "        ], align =\"center\"), #aligns title to center\n",
    "        #no_gutters=True),\n",
    "    href ='/'\n",
    "    ),\n",
    "    dbc.Button(id ='button', children = 'Clicke Me!', color ='primary', className ='ml-auto',href='/')\n",
    "    \n",
    "    \n",
    "    ])\n",
    "    \n",
    "    \n",
    "    \n",
    "app.layout =html.Div(id ='parent', children=[navbar,body_app])\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id='pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='country-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_pie(country):\n",
    "    \n",
    "    df_final = top_15_countries_perc.loc[top_15_countries_perc['COUNTRY']=='{}'.format(country)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id='visa-pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='visa-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_visa_pie(visa_dropdown):\n",
    "    \n",
    "    df_final = top_10_visa_perc.loc[top_10_visa_perc['CLASS_OF_ADMISSION']=='{}'.format(visa_dropdown)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "    \n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run_server()\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "55d2d3f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   CLASS_OF_ADMISSION       variable  value\n",
      "0                H-1B  Approved Case  93.65\n",
      "10               H-1B    Denied Case   6.35\n"
     ]
    },
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "hole": 0.3,
         "labels": [
          "Approved Case",
          "Denied Case"
         ],
         "type": "pie",
         "values": [
          93.65,
          6.35
         ]
        }
       ],
       "layout": {
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        }
       }
      },
      "text/html": [
       "<div>                            <div id=\"42b5e289-bbbb-4051-a580-72bb5c587a2e\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"42b5e289-bbbb-4051-a580-72bb5c587a2e\")) {                    Plotly.newPlot(                        \"42b5e289-bbbb-4051-a580-72bb5c587a2e\",                        [{\"hole\":0.3,\"labels\":[\"Approved Case\",\"Denied Case\"],\"values\":[93.65,6.35],\"type\":\"pie\"}],                        {\"template\":{\"data\":{\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"white\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"gridwidth\":2,\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"bgcolor\":\"#E5ECF6\",\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"zerolinewidth\":2}}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('42b5e289-bbbb-4051-a580-72bb5c587a2e');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def generate_visa_pie(visa_dropdown):\n",
    "    \n",
    "    df_final = top_10_visa_perc.loc[top_10_visa_perc['CLASS_OF_ADMISSION']=='{}'.format(visa_dropdown)]\n",
    "    \n",
    "    print(df_final)\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "\n",
    "generate_visa_pie('H-1B')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84167bd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#Navigation bar\n",
    "\n",
    "app = dash.Dash(external_stylesheets=[ dbc.themes.FLATLY],)\n",
    "\n",
    "#\"https://en.wikipedia.org/wiki/United_States_Department_of_Labor#/media/File:Seal_of_the_United_States_Department_of_Labor.svg\" #reference: https://twitter.com/USDOL\n",
    "\n",
    "DOL_Logo = mpimg.imread('/Users/ujjwaloli/Desktop/Capstone Project/DashboardLogo.png')\n",
    "\n",
    "\n",
    "total = len(df.index)\n",
    "certified = df[df['CASE_STATUS']=='Certified']['CASE_STATUS'].count()\n",
    "denied = df[df['CASE_STATUS']=='Denied']['CASE_STATUS'].count()\n",
    "certified_perc = round(certified/total *100,2)\n",
    "denied_perc = round(denied/total *100,2)\n",
    "\n",
    "def line_plot(df):\n",
    "    fig = go.Figure(data = [go.Scatter(y =cases_status_df['Approved_Case'], x=cases_status_df['Year'], line=dict(color ='firebrick', width =4),text = cases_status_df['Approved_Case'] , name ='Cases Approved'),\n",
    "                           go.Scatter(y =cases_status_df['Denied_Case'], x=cases_status_df['Year'], line=dict(color ='blue', width =4),text = cases_status_df['Denied_Case'] , name ='Cases Denied')])\n",
    "    \n",
    "    \n",
    "    fig.update_layout(title='Analysis of Approved cases and Denied Cases over the years',\n",
    "                         xaxis_title='Year',\n",
    "                         yaxis_title='No of cases',\n",
    "                         margin=dict(l =4,r=4,t=30,b=4))\n",
    "    return fig\n",
    "\n",
    "def country_bar_plot(top_15_countries):\n",
    "    fig  = go.Figure([go.Bar(x = top_15_countries['COUNTRY'],y=top_15_countries['Total applications'], marker_color = 'indianred')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 15 countries with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of perm applications',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "\n",
    "    return fig\n",
    "\n",
    "def visa_bar_plot(top_10_visa):\n",
    "    \n",
    "    fig  = go.Figure([go.Bar(x = top_10_visa['CLASS_OF_ADMISSION'],y=top_10_visa['Total applications'], marker_color = 'blue')])\n",
    "                 #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top 10 class of admissions with high perm applications',\n",
    "                     xaxis_title = 'Countries',\n",
    "                     yaxis_title = 'Number of class of admissions',\n",
    "                     margin = dict(l =4,r=4,t=30,b=4))\n",
    "                     #barmode = 'group')\n",
    "        \n",
    "    return fig\n",
    "\n",
    "def top_employer(ax):\n",
    "    \n",
    "    #print(ax['Total_count'])\n",
    "\n",
    "    fig  = go.Figure([go.Bar(x = ax['Total_count'], y =ax['Employer Name'], marker_color = 'green',orientation='h')])\n",
    "                     #go.Bar(x = df[],y=df[], marker_color = 'blue', name ='')]) #having two go.Bar will have two bar plots \n",
    "\n",
    "\n",
    "    fig.update_layout(title = 'Top sponsoring employers ',\n",
    "                     yaxis_title = 'PERM applications',\n",
    "                     xaxis_title = 'Name of employers')\n",
    "                     #barmode = 'group'\n",
    "        \n",
    "    return fig\n",
    "\n",
    "\n",
    "def top_employer_by_year(emp_year):\n",
    "    fig = px.histogram(emp_year, x=\"Year\", y=\"CASE_STATUS\",\n",
    "             color='EMPLOYER_NAME', barmode='group',\n",
    "             #histfunc='avg',\n",
    "             height=500,title=\"Top Employers(sponsors) over the years\")\n",
    "\n",
    "    return fig\n",
    "\n",
    "def data_for_cases(header, total_cases, percent):\n",
    "    card_content =[\n",
    "        dbc.CardHeader(header),\n",
    "\n",
    "        dbc.CardBody(\n",
    "            [\n",
    "             dcc.Markdown(dangerously_allow_html =True,\n",
    "                     children = [\"{0}<br><sub>+{1}</sub></br>\".format(total_cases,percent)]\n",
    "             )\n",
    "            ]\n",
    "        )\n",
    "\n",
    "\n",
    "    ]\n",
    "    return card_content\n",
    "\n",
    "body_app =dbc.Container([\n",
    "    \n",
    "    dbc.Row(html.Marquee(\"India is the top conuntry in terms of certified cases\"), style ={'color':'green'}),\n",
    "    dbc.Row([\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Certified\",f'{certified:,}', f'{certified_perc:,}'), color='success', style={'text-align':'center'},inverse = True)),\n",
    "        dbc.Col(dbc.Card(data_for_cases(\"Denied\",f'{denied:,}', f'{denied_perc:,}'), color='danger', style={'text-align':'center'},inverse = True)),\n",
    "        #dbc.Col(dbc.Card(card_content, color='secondary', outline=True))\n",
    "        \n",
    "        \n",
    "    ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([html.Div(html.H5('Analysis & Visualizations'), style={'textAlign':'center','fontWeight':'bold','family':'georgia'})]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='line_plot', figure = line_plot(cases_status_df)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='country_bar_plot', figure = country_bar_plot(top_15_countries)), style ={'height':'450px'}),\n",
    "             #dbc.Col(html.Div(), style={'height':'450px'})\n",
    "            dbc.Col([html.Div(id='dropdown_div', children =[\n",
    "                dcc.Dropdown(id = 'country-dropdown', #this is an input \n",
    "                        options = [{'label':i,'value':i } for i in top_15_countries_perc['COUNTRY'].unique()],\n",
    "                        value = 'INDIA',\n",
    "                        placeholder='Select the country')\n",
    "            ],style = {'width':'100%', 'display':'inline-block'}),\n",
    "                     html.Div(dcc.Graph(id ='pie-plot'),style ={'height':'450px'})\n",
    "                    \n",
    "                    \n",
    "                    ])\n",
    "                    \n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row([dbc.Col(dcc.Graph(id='visa_bar_plot', figure = visa_bar_plot(top_10_visa)), style ={'height':'450px'}),\n",
    "             dbc.Col(html.Div(), style={'height':'450px'})\n",
    "            ]),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer', figure = top_employer(ax)), style ={'height':'450px'}),\n",
    "    \n",
    "    html.Br(),\n",
    "    \n",
    "    dbc.Row(dcc.Graph(id='top_employer_by_year', figure = top_employer_by_year(emp_year)), style ={'height':'450px'})\n",
    "    \n",
    "    \n",
    "],fluid=True)\n",
    "navbar = dbc.Navbar(id='navbar', children =[\n",
    "    \n",
    "    html.A(\n",
    "    dbc.Row([\n",
    "        dbc.Col(html.Img(src = DOL_Logo, height= \"70px\")),\n",
    "        dbc.Col(\n",
    "            dbc.NavbarBrand(\"Perm Cases Tracker\", style ={'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'})\n",
    "        \n",
    "        )\n",
    "        \n",
    "        ], align =\"center\"), #aligns title to center\n",
    "        #no_gutters=True),\n",
    "    href ='/'\n",
    "    ),\n",
    "    dbc.Button(id ='button', children = 'Clicke Me!', color ='primary', className ='ml-auto',href='/')\n",
    "    \n",
    "    \n",
    "    ])\n",
    "    \n",
    "    \n",
    "    \n",
    "app.layout =html.Div(id ='parent', children=[navbar,body_app])\n",
    "\n",
    "\n",
    "@app.callback(Output(component_id='pie-plot', component_property ='figure'),\n",
    "                [Input(component_id='country-dropdown', component_property ='value')])\n",
    "\n",
    "def generate_pie(country):\n",
    "    \n",
    "    df_final = top_15_countries_perc.loc[top_15_countries_perc['COUNTRY']=='{}'.format(country)]\n",
    "    \n",
    "    fig = go.Figure(data=[go.Pie(labels=df_final['variable'], values=df_final['value'], hole=.3)])\n",
    "        \n",
    "    return fig\n",
    "    \n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run_server()\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b79ab0cd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de21869c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
