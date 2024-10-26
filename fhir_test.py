import asyncio
import aiofiles
import pathlib
import os
import json
from collections import deque
import time
from datetime import datetime

FLAT_FILE_PATH = "./temp/flat"
BUNDLE_DIR = pathlib.Path("./fhir/")


async def load_bundle(bundle_path):
    async with aiofiles.open(bundle_path, mode="r") as f:
        content = await f.read()
        return json.loads(content)


async def process_bundle(bundle_path):
    file_name = bundle_path.stem
    bundle = await load_bundle(bundle_path)
    patient_data, observations, conditions, medications, procedures, allergies = (
        extract_patient_data(bundle)
    )

    patient_text = f"Patient {patient_data["name"][0]["given"][0]} \
        {patient_data["name"][0]["family"]}\
        gender {patient_data["gender"]}\
        born {patient_data["birthDate"]}\
        {patient_data["maritalStatus"]["text"]} " + (
        f"deceased {patient_data.get("deceasedDateTime", "")}"
        if "deceasedDateTime" in patient_data or "deceasedBoolean" in patient_data
        else "alive"
    )
    observation_text = " ".join(
        [extract_observation_value(obs) for obs in observations]
    )
    condition_texts = " ".join(
        [cond.get("code", {}).get("text", "") for cond in conditions]
    )
    medication_texts = " ".join([extract_medication_value(med) for med in medications])

    allergy_texts = " ".join(
        [extract_allergy_information(allergy) for allergy in allergies]
    )

    metadata = {
        "name": f"{patient_data["name"][0]["given"][0]} {patient_data["name"][0]["family"]}",
        "birth_date": patient_data["birthDate"],
        "conditions": [cond.get("code", {}).get("text", "") for cond in conditions],
        "observations": [extract_observation_value(obs) for obs in observations],
        "medications": [extract_medication_value(med) for med in medications],
        "procedures": [extract_procedure_value(procedure) for procedure in procedures],
        "allergies": [extract_allergy_information(allergy) for allergy in allergies],
    }

    # procedure_texts = " ".join(
    #    [extract_procedure_value(procedure) for procedure in procedures]
    # )

    combined_texts = f"{patient_text}\nObservations: {observation_text}\nConditions: {condition_texts}\nMedications: {medication_texts}\nAllergies: {allergy_texts}"

    patient_object = {
        "id": patient_data["id"],
        "text": combined_texts,
        "metadata": metadata,
    }

    async with aiofiles.open(
        f"{FLAT_FILE_PATH}/{file_name}_processed.json", "w"
    ) as out:
        await out.write(json.dumps(patient_object, indent=4))


def extract_allergy_information(allergy):
    return allergy["code"]["text"]


def extract_medication_value(medication):
    medication_name = medication.get("medicationCodeableConcept", {}).get("text", "")
    status = medication.get("status", "")

    return f"{medication_name} {status}"


def extract_procedure_value(procedure):
    name = procedure["code"]["text"]
    status = procedure["status"]
    period = ""
    if "performedPeriod" in procedure:
        start_date = datetime.fromisoformat(
            procedure["performedPeriod"]["start"]
        ).strftime("%Y:%m:%d")
        end_date = datetime.fromisoformat(procedure["performedPeriod"]["end"]).strftime(
            "%Y:%m:%d"
        )

        period = f"{start_date}-{end_date}"

    return f"{name} {status} {period}"


def extract_observation_value(observation):
    """Extract the main value from an Observation based on available fields."""
    if "valueQuantity" in observation:
        # Extract numerical value with unit
        quantity = observation["valueQuantity"]
        text = observation["code"]["text"]
        date_time = datetime.fromisoformat(observation["issued"]).strftime("%Y:%m:%d")

        return (
            f"{text} {quantity['value']} {quantity['unit']} {date_time}"
            if "unit" in quantity
            else f"{text} {quantity["value"]}"
        )

    elif "valueCodeableConcept" in observation:
        # Extract coded value (e.g., LOINC code description)
        return observation["valueCodeableConcept"]["text"]

    elif "valueBoolean" in observation:
        # Extract boolean value
        return "True" if observation["valueBoolean"] else "False"

    elif "valueString" in observation:
        # Extract plain text value
        return observation["valueString"]

    else:
        # If no recognizable value field is present
        return "None"


def extract_patient_data(fhir_data):
    patient_resource = fhir_data["entry"][0]["resource"]
    observations = [
        entry["resource"]
        for entry in fhir_data["entry"]
        if entry["resource"]["resourceType"] == "Observation"
    ]
    conditions = [
        entry["resource"]
        for entry in fhir_data["entry"]
        if entry["resource"]["resourceType"] == "Condition"
    ]
    medications = [
        entry["resource"]
        for entry in fhir_data["entry"]
        if entry["resource"]["resourceType"] == "MedicationRequest"
    ]
    procedures = [
        entry["resource"]
        for entry in fhir_data["entry"]
        if entry["resource"]["resourceType"] == "Procedure"
    ]
    allergies = [
        entry["resource"]
        for entry in fhir_data["entry"]
        if entry["resource"]["resourceType"] == "AllergyIntolerance"
    ]
    return (
        patient_resource,
        observations,
        conditions,
        medications,
        procedures,
        allergies,
    )


async def process_batch(batch):
    tasks = [process_bundle(file) for file in batch if file.is_file()]
    return await asyncio.gather(*tasks)


async def load_and_process_bundles(dir_path: pathlib.Path, batch_size=100):
    filenames = [file for file in dir_path.iterdir()]
    results = []
    start_time = time.time()

    file_queue = deque(filenames)

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / len(results)) * len(filenames)
        remaining_time = estimated_total_time - elapsed_time
        print(
            f"Processed {len(results)}/{len(filenames)} "
            f"({len(results)/len(filenames)*100:.2f}%) "
            f"| Elapsed: {elapsed_time:.2f}s "
            f"| Remaining: {remaining_time:.2f}s "
        )

        await asyncio.sleep(0.01)


async def main():
    if not os.path.exists(FLAT_FILE_PATH):
        os.makedirs(FLAT_FILE_PATH)
    await load_and_process_bundles(BUNDLE_DIR)


if __name__ == "__main__":
    asyncio.run(main())
