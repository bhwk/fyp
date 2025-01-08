from llama_index.llms.ollama import Ollama


llm = Ollama(
    model="mistral-nemo",
    temperature=0.4,
    request_timeout=1000,
    context_window=16000,
    additional_kwargs={"top_k": 10},
)

history = ""
with open("./combined-record.txt") as f:
    history = f.read()

print("Generating patient summary.")
generated_summary = llm.complete(
    f"""SYSTEM PROMPT:
    You are a highly skilled medical assistant trained to generate detailed and structured patient summaries based on provided clinical data. Ensure your summaries are professional, precise, and adhere to the following guidelines:

    Include Key Identifiers: Use the patient's full name, age, and any other provided demographic information in the summary. Ensure accurate representation of their identity.
    ## Organizational Structure: Break the summary into clear sections with headers:
        Patient Information
        Clinical Observations
        Relevant Medical History
        Recent Procedures
        Medications
        Allergies

    ## Data Reporting:
        Include exact numerical values for vital signs, lab results, and other quantitative data (e.g., height, weight, BMI, blood pressure, glucose levels).
        Use precise dates for events, diagnoses, or procedures where applicable.
        Report medical conditions and history with specific terminology.
        Summarize readings into a range of values that include the maximum and minimum values for each result and observation. (min value - max value)
        Provide the range of values that occur for each reading.

    ## Tone and Clarity:
        Use formal and professional language. Avoid abbreviations unless they are common medical terms (e.g., "BP" for blood pressure).
        Write in full sentences, ensuring clarity for medical professionals reviewing the report.
    
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

with open("./zero_shot_result.txt", "w") as out:
    out.write(
        f"""Patient summary:\n{generated_summary}\n\nSynth report:\n{synth_report}"""
    )
