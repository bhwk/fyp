from llama_index.llms.ollama import Ollama


llm = Ollama(
    model="mistral-nemo",
    temperature=0.4,
    request_timeout=500,
    context_window=16000,
    additional_kwargs={"top_k": 10},
)

history = ""
with open("./combined-record.txt") as f:
    history = f.read()

print("Generating patient summary.")
generated_summary = llm.complete(
    f"""
    SYSTEM PROMPT:
    You are a highly capable summarization assistant. Your goal is to extract and summarize key medical information from detailed health records in a concise and structured format. Focus on identifying:

    Patient Demographics: Include name, gender, date of birth, marital status, and deceased status.
    Vital Observations: Summarize relevant metrics like height, weight, BMI, blood pressure, glucose, cholesterol, etc., emphasizing significant trends or noteworthy values.
    Medical Conditions: List significant diagnoses along with recorded dates.
    Procedures: Highlight completed procedures.
    Medications: Categorize medications into active and stopped, noting their names and dosages.
    Allergies: Summarize reported allergies, if any.
    Smoking Status: Include tobacco smoking status.

    Input Example:
    Name: John Doe
    Gender: Male
    Born: 1950-01-01
    Marital Status: M
    Deceased: True
    ... [additional details as seen in the record above].

    Output Format:
    Use the following structured format for your response:
    ### Summary  
    **Demographics:**  
    - Name: [Name]  
    - Gender: [Gender]  
    - DOB: [Date of Birth]  
    - Marital Status: [Marital Status]  
    - Deceased: [Yes/No]  

    **Observations:**  
    - Height: [Value] cm  
    - Weight: [Value] kg  
    - BMI: [Value] kg/m²  
    - Blood Pressure: Systolic [Value] mmHg / Diastolic [Value] mmHg  
    - Glucose: [Value] mg/dL  
    - Hemoglobin A1c: [Value] %  
    - Cholesterol: Total [Value] mg/dL, LDL [Value] mg/dL, HDL [Value] mg/dL  
    - [Add other metrics as needed.]  

    **Conditions:**  
    - [Condition Name]: [Date Recorded]  
    - ...  

    **Procedures:**  
    - [Procedure Name(s)]  

    **Medications:**  
    - **Active:**  
    - [Medication Name, Dosage]  
    - **Stopped:**  
    - [Medication Name, Dosage]  

    **Allergies:**  
    - [List of allergies or "None"]  

    **Smoking Status:**  
    - [Never smoker/Current smoker/Former smoker]  
    
    ---
    [PATIENT HISTORY]
        {history}
    [/PATIENT HISTORY]
        """
)

print("Generating synthetic report")

synth_report = llm.complete(
    f"""
    SYSTEM PROMPT:
    You are an advanced language model tasked with summarizing patient data into a concise and structured medical report. Follow these instructions precisely:

    Anonymization: The patient’s name and any identifying information must be removed or replaced with placeholders (e.g., "[Anonymized]").

    Structure: Organize the summary into the following sections:
        Patient Summary
        Clinical Observations
        Relevant Medical History
        Recent Procedures
        Medications
        Allergies

    Formatting Rules:
        Use bullet points for observations and history to enhance readability.
        Replace values with rounded values for lab results and vital signs. Use approximate ranges if values fluctuate.
        Replace exact dates with the patient’s life phase (e.g., "Young Adulthood" or "Late Adulthood").
        Avoid including exact locations or unnecessary identifiers.

    Clinical Data Requirements:
        Summarize relevant vitals (e.g., blood pressure, BMI, glucose levels) with appropriate medical context.
        Round all values in the summary.
        Ensure that all tests present in the original summary are present in the anonymized summary.

    ## Tone and Clarity:
        Use formal and professional language. Avoid abbreviations unless they are common medical terms (e.g., "BP" for blood pressure).
        Write in full sentences, ensuring clarity for medical professionals reviewing the report.

    ---
    [REPORT SUMMARY]
            {generated_summary}
    [/REPORT SUMMARY]
        """
)

with open("./few_shot_result.txt", "w") as out:
    out.write(
        f"""Patient summary:\n{generated_summary}\n\nSynth report:\n{synth_report}"""
    )
