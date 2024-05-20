import streamlit as st
from streamlit_tags import st_tags
import psycopg2
import pandas as pd
import time
from openai import OpenAI
import os

#os.environ["OPENAI_API_KEY"] ='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'  # Replace with your actual OpenAI API key
MODEL="gpt-4o"

# Function to reset session state
def reset_state():
    st.session_state.update({
        'step': 1,
        'age': 0,
        'gender': "Male",
        'comorbidities': [],
        'medications': [],
        'side_effects': [],
        'new_prescribed_medications': [],
        'search_results': pd.DataFrame(),
        'selected_drug': None,
        'info_types': [],
        'DUR_check': None
    })

# Initialize session state for user inputs if they don't already exist
if 'step' not in st.session_state:
    reset_state()

st.set_page_config(page_title="SMART DUR CDSS", page_icon="📝")
st.title("SMART_DUR CDSS📝")
st.markdown("""
    안녕하세요, 저는 강민규 교수가 만든 "SMART Drug Utilization Review (DUR) CDSS"입니다. 
    
    더 많은 정보를 원하시면 irreversibly@gmail.com으로 이메일을 보내주세요.
    이 GPT는 의사나 약사와 상담하기 전에 정보를 제공하는 것을 목적으로 합니다. 
    정확한 정보는 의사나 약사에게 문의하세요.
    
    보안과 개인정보 보호와 관련하여, 사용자의 개인 데이터는 즉시 쿼리 범위를 넘어 저장되거나 사용되지 않습니다.
""")

# PostgreSQL connection details
DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'dbname': 'HIRA_DB',
    'host': '222.116.163.76',
    'port': '5432'
}

def get_drugs_containing_keyword(keyword):
    schema_name = 'hira_02'
    table_name = 'hiradb_202307'

    # Establish the database connection
    conn = psycopg2.connect(**DB_CONFIG)
    query = f'SELECT 한글상품명, 품목기준코드, "제품코드(개정후)", "일반명코드(성분명코드)", atc코드 FROM {schema_name}.{table_name} WHERE 한글상품명 LIKE %s'
    
    # Use pandas to read the SQL query into a DataFrame
    df_druglist = pd.read_sql_query(query, conn, params=('%' + keyword + '%',))
    conn.close()

    # Remove duplicates
    return df_druglist.drop_duplicates()

def extract_drug_pa_labels(keywords):
    schema_name = 'drug_label_llm'
    table_name = "drug_pa_only"
    combined_df = pd.DataFrame()

    conn = psycopg2.connect(**DB_CONFIG)
    for keyword in keywords:
        query = f"SELECT drug_item_cd, drug_article_tp, p_text_pi, p_text_with FROM {schema_name}.\"{table_name}\" WHERE drug_item_cd LIKE %s"
        df_label = pd.read_sql_query(query, conn, params=(f'{keyword}%',))
        combined_df = pd.concat([combined_df, df_label], ignore_index=True)
    conn.close()

    return combined_df

def extract_drug_in_labels(keywords):
    schema_name = 'drug_label_llm'
    table_name = "drug_in_only"
    combined_df = pd.DataFrame()

    conn = psycopg2.connect(**DB_CONFIG)
    for keyword in keywords:
        query = f"SELECT p_text_with FROM {schema_name}.\"{table_name}\" WHERE drug_item_cd LIKE %s"
        df_label = pd.read_sql_query(query, conn, params=(f'{keyword}%',))
        combined_df = pd.concat([combined_df, df_label], ignore_index=True)
    conn.close()

    return combined_df

def extract_general_information_labels(selected_drugs_info):
    schema_name = 'drug_label_llm'
    table_name = "drug_total_label"
    combined_df = pd.DataFrame()

    df_atc = get_atc(selected_drugs_info['atc'].unique())
    df_atc.set_index('atc_code', inplace=True)

    conn = psycopg2.connect(**DB_CONFIG)
    for index, row in selected_drugs_info.iterrows():
        keyword = row['code']
        query = f"SELECT * FROM {schema_name}.\"{table_name}\" WHERE item_seq LIKE %s"
        df_label = pd.read_sql_query(query, conn, params=(f'{keyword}%',))
        if not df_label.empty:
            atc_code = row['atc']
            df_label['atc_code'] = atc_code
            df_label['atc_name_english'] = df_atc.at[atc_code, 'atc_name'] if atc_code in df_atc.index else None
            df_label['atc_name_korean'] = df_atc.at[atc_code, 'atc_korean_name'] if atc_code in df_atc.index else None
            combined_df = pd.concat([combined_df, df_label], ignore_index=True)
    conn.close()

    return combined_df

def extract_main_ingredients(keywords):
    schema_name = 'drug_label_llm'
    table_name = "drug_main_item_long" 
    combined_df = pd.DataFrame()

    conn = psycopg2.connect(**DB_CONFIG)
    for keyword in keywords:
        query = f"SELECT item_name FROM {schema_name}.\"{table_name}\" WHERE \"﻿item_seq\" LIKE %s"
        df_label = pd.read_sql_query(query, conn, params=(f'{keyword}%',))
        combined_df = pd.concat([combined_df, df_label], ignore_index=True)
    conn.close()

    return combined_df

def extract_additives(keywords):
    schema_name = 'drug_label_llm'
    table_name = "drug_ingr_long"
    combined_df = pd.DataFrame()

    conn = psycopg2.connect(**DB_CONFIG)
    for keyword in keywords:
        query = f"SELECT item_name FROM {schema_name}.\"{table_name}\" WHERE \"﻿item_seq\" LIKE %s"
        df_label = pd.read_sql_query(query, conn, params=(f'{keyword}%',))
        combined_df = pd.concat([combined_df, df_label], ignore_index=True)
    conn.close()

    return combined_df

def get_atc(keywords):
    schema_name = 'drug_label_llm'
    table_name = "atc_list"
    combined_df = pd.DataFrame()

    conn = psycopg2.connect(**DB_CONFIG)
    for keyword in keywords:
        query = f"SELECT * FROM {schema_name}.\"{table_name}\" WHERE \"atc_code\" = %s"
        df_label = pd.read_sql_query(query, conn, params=(keyword,))
        combined_df = pd.concat([combined_df, df_label], ignore_index=True)
    conn.close()

    return combined_df

def generate_instructions_from_df(drug_df, df_info):
    all_instructions = ""

    # Iterate over the rows of the DataFrame
    for index, row in df_info.iterrows():
        print(index,row)
        instruction_introduction = f'''
=============================================================
drug brand name : {row['item_name']}
main ingredient : {row['main_item_ingr']}
'''
        temp_item_seq = row['item_seq']
        drug_pa_information = ""


        for _, drug_row in drug_df.iterrows():
            if drug_row['drug_item_cd'] == temp_item_seq:
                drug_pa_information_temp = f'''
information : {drug_row['drug_article_tp']} - {drug_row['p_text_pi']}
--------------------------------------------------------
{drug_row['p_text_with']}
--------------------------------------------------------
'''
                drug_pa_information += drug_pa_information_temp

        instructions = instruction_introduction + drug_pa_information
        all_instructions += instructions

    return all_instructions.strip()


def get_patient_information():
    return f"Age: {st.session_state.age}, Gender: {st.session_state.gender}, Comorbidities: {', '.join(st.session_state.comorbidities)}, Medications: {', '.join(st.session_state.medications)}, Side Effects: {', '.join(st.session_state.side_effects)}"

# Sidebar for OpenAI API Key and displaying patient information
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if st.button("Save OpenAI API Key"):
        st.session_state["OPENAI_API_KEY"] = openai_api_key

    if "OPENAI_API_KEY" in st.session_state:
        st.header("Submitted Patient Information")

        st.markdown("""
        <style>
        .sidebar-content {
            background-color: #f9f9f9;
            padding: 5px;
            border-radius: 5px;
            font-size: 0.85em;
        }
        .sidebar-content h4 {
            margin-bottom: 5px;
            font-size: 1em;
        }
        .sidebar-content p {
            margin: 0;
            font-size: 0.85em;
        }
        .sidebar-section {
            margin-bottom: 10px;
        }
        .inline-section {
            display: flex;
            justify-content: space-between;
        }
        .inline-section div {
            flex: 1;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-section inline-section">
                <div>
                    <h4>Age:</h4>
                    <p>{}</p>
                </div>
                <div>
                    <h4>Gender:</h4>
                    <p>{}</p>
                </div>
            </div>
            <div class="sidebar-section">
                <h4>Comorbidities:</h4>
                <p>{}</p>
            </div>
            <div class="sidebar-section">
                <h4>Medications:</h4>
                <p>{}</p>
            </div>
            <div class="sidebar-section">
                <h4>Side Effects:</h4>
                <p>{}</p>
            </div>
        </div>
        """.format(
            st.session_state.age,
            st.session_state.gender,
            ', '.join(st.session_state.comorbidities) or 'None',
            ', '.join(st.session_state.medications) or 'None',
            ', '.join(st.session_state.side_effects) or 'None'
        ), unsafe_allow_html=True)

        # Newly Prescribed Medications
        st.header("Newly Prescribed Medications")
        drug_names = [med['name'] for med in st.session_state['new_prescribed_medications']]
        selected_drug_name = st.radio("Select a drug to view information", drug_names)
        st.session_state.selected_drug = next((med for med in st.session_state['new_prescribed_medications'] if med['name'] == selected_drug_name), None)

        # Information Type Selection
        info_types = ["Total", "Main Ingredients", "Additives", "Contraindications", "General Information", "Interactions"]
        st.session_state.info_types = st.multiselect("Select information type(s)", info_types)

        if st.button("View Medication Information"):
            st.session_state.step = 3
            st.experimental_rerun()

        # DUR Checks
        st.header("SMART DUR CDSS")
        dur_options = ["General DUR check", "Allergy DUR check", "Side Effect DUR check", "Contraindication DUR check", "Interaction DUR check"]
        st.session_state.DUR_check = st.radio("Select a DUR check", dur_options)
        if st.button("DUR Check"):
            st.session_state.step = 4
            st.experimental_rerun()

# Main content for inputting patient information
if "OPENAI_API_KEY" in st.session_state:
    if st.session_state.step == 1:
        st.chat_message("assistant").write("복약지도를 원하는 환자의 임상정보 입력해주세요")

        with st.form("patient_info"):
            col1, col2 = st.columns(2)
            age = st.number_input("Enter age", min_value=0, max_value=150, step=1, value=st.session_state.age, key='age_input')
            gender = st.selectbox("Select gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.gender))

            comorbidities = st_tags(
                key='comorbidities_input',
                label='Comorbidities:',
                text='Press enter to add more',
                value=st.session_state['comorbidities']
            )

            medications = st_tags(
                key='medications_input',
                label='Medications:',
                text='Press enter to add more',
                value=st.session_state['medications']
            )

            side_effects = st_tags(
                key='side_effects_input',
                label='Side Effects:',
                text='Press enter to add more',
                value=st.session_state['side_effects']
            )

            if st.form_submit_button("Submit Information"):
                st.session_state.update({
                    'age': age,
                    'gender': gender,
                    'comorbidities': comorbidities,
                    'medications': medications,
                    'side_effects': side_effects,
                    'step': 2
                })
                st.success("Information submitted successfully!")
                st.experimental_rerun()

    elif st.session_state.step == 2:
        st.chat_message("assistant").write("이번에 환자가 투약받은 약제를 선택해주세요")

        keyword = st.text_input("Enter the name of the medication:", key='keyword_input')
        if st.button("Search Medication") and keyword:
            st.session_state['search_results'] = get_drugs_containing_keyword(keyword)

        if not st.session_state['search_results'].empty:
            st.write("Search Results:")
            st.dataframe(st.session_state['search_results'])

            row_number = st.number_input("Enter the row number of the medication to add:", min_value=0, max_value=len(st.session_state['search_results']) - 1, step=1)
            
            if st.button("Add Medication"):
                selected_medication = st.session_state['search_results'].iloc[row_number]
                medication_info = {
                    "name": selected_medication['한글상품명'],
                    "code": selected_medication['품목기준코드'],
                    "atc": selected_medication['atc코드']
                }
                st.session_state['new_prescribed_medications'].append(medication_info)
                st.success(f"Medication '{medication_info['name']}' added successfully!")
                st.experimental_rerun()

        if st.button("Submit Medications"):
            st.success("Newly prescribed medications submitted successfully!")
            st.session_state.step = 3
            st.experimental_rerun()

    elif st.session_state.step == 3:
        if st.session_state.selected_drug and st.session_state.info_types:
            selected_drug_info = st.session_state.selected_drug
            drug_code = selected_drug_info['code']

            st.write(f"### {selected_drug_info['name']} Information")
            
            if "Total" in st.session_state.info_types:
                df_info = extract_general_information_labels(pd.DataFrame([selected_drug_info]))
                if not df_info.empty:
                    st.markdown("### General Information")
                    for idx, row in df_info.iterrows():
                        st.markdown(f"- **{row['atc_name_english'] or ''}** ({row['atc_name_korean'] or ''})")
                        for col in df_info.columns:
                            if col not in ['atc_code', 'atc_name_english', 'atc_name_korean']:
                                st.markdown(f"  - {col}: {row[col]}")
                else:
                    st.write("No general information found.")
            if "Main Ingredients" in st.session_state.info_types:
                df_main_ingr = extract_main_ingredients([drug_code])
                if not df_main_ingr.empty:
                    st.markdown("### Main Ingredients")
                    for item in df_main_ingr['item_name']:
                        st.markdown(f"- {item}")
                else:
                    st.write("No main ingredients found.")
            if "Additives" in st.session_state.info_types:
                df_ingr = extract_additives([drug_code])
                if not df_ingr.empty:
                    st.markdown("### Additives")
                    st.markdown(", ".join(df_ingr['item_name']))
                else:
                    st.write("No additives found.")
            if "Contraindications" in st.session_state.info_types:
                df_pa_labels = extract_drug_pa_labels([drug_code])
                if not df_pa_labels.empty:
                    st.markdown("### Contraindications")
                    grouped = df_pa_labels.groupby('drug_article_tp')
                    for article_type, group in grouped:
                        st.markdown(f"#### {article_type}")
                        for text in group['p_text_with']:
                            st.markdown(f"- {text}")
                else:
                    st.write("No contraindications found.")
            if "General Information" in st.session_state.info_types:
                df_info = extract_general_information_labels(pd.DataFrame([selected_drug_info]))
                if not df_info.empty:
                    st.markdown("### General Information")
                    for idx, row in df_info.iterrows():
                        st.markdown(f"- **{row['atc_name_english'] or ''}** ({row['atc_name_korean'] or ''})")
                        for col in df_info.columns:
                            if col not in ['atc_code', 'atc_name_english', 'atc_name_korean']:
                                st.markdown(f"  - {col}: {row[col]}")
                else:
                    st.write("No general information found.")
            if "Interactions" in st.session_state.info_types:
                df_in_labels = extract_drug_in_labels([drug_code])
                if not df_in_labels.empty:
                    st.markdown("### Interactions")
                    for interaction in df_in_labels['p_text_with']:
                        st.markdown(interaction.replace('\n', '<br>'), unsafe_allow_html=True)
                else:
                    st.write("No interactions found.")

    elif st.session_state.step == 4:
        if st.session_state.DUR_check:
            st.write(f"### {st.session_state.DUR_check}")

            if st.session_state.DUR_check == "General DUR check":
                if st.session_state.selected_drug:
                    df_info = extract_general_information_labels(pd.DataFrame([st.session_state.selected_drug]))
                    #df_in_labels = extract_drug_in_labels([st.session_state.selected_drug['code']])
                    df_pa_labels = extract_drug_pa_labels(st.session_state.selected_drug['code'])
                    patient_information = get_patient_information()

                    ## revised version of GPT4

                    instruction_template_introduction = '''
                    You are a medical assistant chatbot specialized in medication guidance.
                    Your goal is to generate a personalized medication guide for the patient, focusing on crucial precautions and potential side effects of their newly prescribed medications, especially considering the patient's age and comorbidities.

                    Patient's Information:
                    -----------------------------------
                    {patient_information}

                    Prescription Drug Information and Specific Precautions:
                    [Detailed Drug Information Here]
                    '''

                    instruction_template_interaction_tail = '''
                    Guidelines for Medication Guide Creation:
                    - Output and summary should be very concise, mentioning only what is relevant to the given patient information. They should be written in bullet points, not large paragraphs.- Imagine you're a doctor or pharmacist who wants to give patients information about what to watch out for to avoid adverse drug reactions. Create a medication guide as if you were talking to a patient.
                    - Unless it's absolutely contraindicated, I think it's more important to be clear with patients about side effects or symptoms associated with exacerbation of a disease that may be caused by a medication, rather than using a dosing caution or contraindication, and to ask them if they've experienced those symptoms after a few days of taking the medication, and to discontinue the medication quickly if they do experience side effects or exacerbation of an underlying condition. So only tell them not to take it if it's absolutely contraindicated, and if it's not absolutely contraindicated, tone it down and explain the possible side effects and diseases that may occur. 
                    - Start with a personalized greeting using the patient's name.
                    - For each prescribed medication, list the brand name, main ingredients, and specific precautions.
                    - Prioritize information relevant to the patient's age and comorbidities, especially considering the elderly.
                    - Clearly explain potential side effects, making them understandable to non-medical users.
                    - Highlight any contraindications or precautions relevant to the existing medications.
                    - At the end, emphasize the signs and symptoms of possible side effects rather than advising against taking the medication. Encourage the patient to report these symptoms and consult the chatbot for guidance.
                    - Use simple visual aids or icons where possible for clarity.
                    '''

                    #st.markdown(df_info)
                    #st.markdown(df_in_labels)
                    #st.markdown(patient_information)

                    #st.markdown(generate_instructions_from_df(df_in_labels, df_info))

                    #introduction_template_reconciliation = instruction_template_introduction.format(patient_information=patient_information) + generate_instructions_from_df(df_in_labels, df_info) + instruction_template_tail
                    intrudction_template_interaction = instruction_template_introduction.format(patient_information = patient_information) + generate_instructions_from_df (df_pa_labels, df_info) + instruction_template_interaction_tail

                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", openai_api_key))

                    completion = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", 
                             "content": intrudction_template_interaction}, # <-- This is the system message that provides context to the model
                        ]
                        )
                                     
                    # Get the content of the guidance message
                    guidance = completion.choices[0].message.content
                    st.markdown(guidance)

            elif st.session_state.DUR_check == "Allergy DUR check":
                st.write("Performing Allergy DUR check...")
            elif st.session_state.DUR_check == "Side Effect DUR check":
                st.write("Performing Side Effect DUR check...")
            elif st.session_state.DUR_check == "Contraindication DUR check":
                st.write("Performing Contraindication DUR check...")
            elif st.session_state.DUR_check == "Interaction DUR check":
                st.write("Performing Interaction DUR check...")

