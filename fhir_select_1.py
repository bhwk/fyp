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

    if not os.path.exists(f"{FLAT_FILE_PATH}/{file_name}"):
        os.makedirs(f"{FLAT_FILE_PATH}/{file_name}")

    bundle = await load_bundle(bundle_path)
    patient_data, observations, conditions, medications, procedures, allergies = (
        extract_patient_data(bundle)
    )

    patient_object = dict()
    patient_object["patient_info"] = dict()

    # add patient details
    patient_object["patient_info"]["name"] = (
        f"{patient_data["name"][0]["given"][0]} {patient_data["name"][0]["family"]}"
    )
    patient_object["patient_info"]["gender"] = patient_data["gender"]
    patient_object["patient_info"]["birthDate"] = patient_data["birthDate"]
    patient_object["patient_info"]["maritalStatus"] = patient_data["maritalStatus"][
        "text"
    ]
    if "deceasedDateTime" or "deceasedBoolean" in patient_data:
        patient_object["patient_info"]["deceased"] = True
    else:
        patient_object["patient_info"]["deceased"] = False

    patient_object["patient_info"]["conditions"] = list(dict())
    patient_object["patient_info"]["medications"] = list(dict())
    patient_object["patient_info"]["allergies"] = list()

    for observation in observations:
        effective_date, value = extract_observation_value(observation)
        if effective_date not in patient_object:
            patient_object[effective_date] = dict()
            patient_object[effective_date]["observations"] = list()
        patient_object[effective_date]["observations"].append(value)

    for condition in conditions:
        value = extract_condition(condition)
        patient_object["patient_info"]["conditions"].append(value)

    for medication in medications:
        value = extract_medication_value(medication)
        patient_object["patient_info"]["medications"].append(value)
    for allergy in allergies:
        name = extract_allergy_information(allergy)
        patient_object["patient_info"]["allergies"].append(name)

    for procedure in procedures:
        date, value = extract_procedure_value(procedure)
        if not date:
            continue
        if date not in patient_object:
            patient_object[date] = dict()
            patient_object[date]["procedures"] = list()
        patient_object[date]["procedures"].append(value)

    # Write out the combined patient info_file
    async with aiofiles.open(
        f"{FLAT_FILE_PATH}/{file_name}/{file_name}_patient_info.txt", "w"
    ) as out:
        patient_info = patient_object["patient_info"]

        patient = f"Name: {patient_info["name"]} Gender: {patient_info["gender"]} Born: {patient_info["birthDate"]} MaritalStatus: {patient_info["maritalStatus"]} Deceased: {patient_info["deceased"]}"

        combined_conditions = " ".join(patient_info["conditions"])
        combined_medications = " ".join(patient_info["medications"])
        combined_allergy = " ".join(patient_info["allergies"])

        await out.write(
            f"{patient}\nConditions: {combined_conditions}\nMedications: {combined_medications}\nAllergies: {combined_allergy}"
        )

    for key in patient_object.keys():
        # the rest of the keys are dates, so we skip patient_info key
        if key == "patient_info":
            continue

        observations = patient_object[key].get("observations", "")
        procedures = patient_object[key].get("procedures", "")

        combined_observations = " ".join(observations)
        combined_procedures = " ".join(procedures)

        async with aiofiles.open(
            f"{FLAT_FILE_PATH}/{file_name}/{file_name}_{key}.txt", "w"
        ) as out:
            patient_info = patient_object["patient_info"]
            patient = f"Name: {patient_info["name"]} Gender: {patient_info["gender"]} Born: {patient_info["birthDate"]} MaritalStatus: {patient_info["maritalStatus"]} Deceased: {patient_info["deceased"]}"

            await out.write(
                f"{patient}\nObservations: {combined_observations}\nProcedures: {combined_procedures}"
            )


def extract_allergy_information(allergy):
    return allergy["code"]["text"]


def extract_medication_value(medication):
    medication_name = medication.get("medicationCodeableConcept", {}).get("text", "")
    status = medication.get("status", "")
    return f"{status} {medication_name}"


def extract_condition(entry):
    name = entry["code"]["text"]
    recorded_date = datetime.fromisoformat(entry["recordedDate"]).strftime("%Y/%m/%d")
    return f"{name} recorded {recorded_date}"


def extract_procedure_value(procedure):
    name = procedure["code"]["text"]
    status = procedure["status"]
    if "performedPeriod" in procedure:
        start_date = datetime.fromisoformat(
            procedure["performedPeriod"]["start"]
        ).strftime("%Y/%m/%d")
        end_date = datetime.fromisoformat(procedure["performedPeriod"]["end"]).strftime(
            "%Y/%m/%d"
        )

        if status == "completed":
            return (end_date.replace("/", "-"), f" {status} {name}")
        else:
            return (start_date.replace("/", "-"), f" {status} {name}")
    else:
        return ("", f"{status} {name}")


def extract_observation_value(entry):
    """Extract the main value from an Observation based on available fields."""
    code = entry["code"]["coding"][0]["display"]
    effective_date = datetime.fromisoformat(entry["effectiveDateTime"]).strftime(
        "%d/%m/%Y"
    )
    # issued = datetime.fromisoformat(entry["issued"]).strftime("%d/%m/%Y %H:%M:%S")
    if "valueQuantity" in entry:
        value = format(
            f"{entry["valueQuantity"]["value"]:.2f} {entry["valueQuantity"]["unit"]}"
        )
    elif "valueCodeableConcept" in entry:
        # Extract coded value (e.g., LOINC code description)
        value = entry["valueCodeableConcept"]["text"]
    elif "valueBoolean" in entry:
        value = "True" if entry["valueBoolean"] else "False"
    elif "valueString" in entry:
        value = entry["valueString"]

    elif "component" in entry:
        # have to extract components from the observation
        component_list = []
        for component in entry["component"]:
            comp_code = component["code"]["coding"][0]["display"]
            if "valueQuantity" in component:
                comp_value = format(
                    f"{component["valueQuantity"]["value"]:.2f} {component["valueQuantity"]["unit"]}"
                )
            elif "valueCodeableConcept" in component:
                # Extract coded value (e.g., LOINC code description)
                comp_value = component["valueCodeableConcept"]["text"]
            elif "valueBoolean" in component:
                comp_value = "True" if component["valueBoolean"] else "False"
            elif "valueString" in component:
                comp_value = component["valueString"]
            else:
                comp_value = "None"
            component_list.append(
                format(f"Component is {comp_code}. Value is {comp_value}")
            )
        components = " ".join(component_list)
        return (
            effective_date.replace("/", "-"),
            f"Code is {code}. {components}.",
        )

    else:
        value = "None"
    return (
        effective_date.replace("/", "-"),
        f"Code is {code}. Value is {value}.",
    )


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
