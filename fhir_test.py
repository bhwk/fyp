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
    patient = find_patient(bundle)

    for i, entry in enumerate(bundle["entry"]):
        data = filter_entry(entry)
        if not os.path.exists(f"{FLAT_FILE_PATH}/{file_name}"):
            os.makedirs(f"{FLAT_FILE_PATH}/{file_name}")
        async with aiofiles.open(
            f"{FLAT_FILE_PATH}/{file_name}/{file_name}_{i}.txt", "w"
        ) as out:
            await out.write(
                f"Patient name is {patient["PatientFirstName"]} {patient['PatientFamilyName']}\n{data}"
            )


def filter_entry(entry):
    if entry["resource"]["resourceType"] == "Observation":
        return extract_observation(entry["resource"])


def extract_observation(entry):
    # Extract necessary fields from observation resource
    status = entry["status"]
    category = entry["category"][0]["coding"][0]["display"]
    code = entry["code"]["coding"][0]["display"]
    effective_time = datetime.fromisoformat(entry["effectiveDateTime"]).strftime(
        "%d/%m/%Y %H:%M:%S"
    )
    issued = datetime.fromisoformat(entry["issued"]).strftime("%d/%m/%Y %H:%M:%S")
    value = (
        format(
            f"{entry["valueQuantity"]["value"]:.2f} {entry["valueQuantity"]["unit"]}"
        )
        if "valueQuantity" in entry
        else "None"
    )
    return f"Entry is type {entry["resourceType"]}. Status is {status}. Category is {category}. \
    Code is {code}. This entry was effective on {effective_time}. This entry was issued {issued}. \
    Value quantity for entry is {value}"


def filter_patient(entry):
    return entry["resource"]["resourceType"] == "Patient"


def find_patient(bundle):
    patients = list(filter(filter_patient, bundle["entry"]))
    if len(patients) < 1:
        raise Exception("No patient found")
    else:
        patient = patients[0]["resource"]
        patient_id = patient["id"]
        patient_given_name = patient["name"][0]["given"][0]
        patient_family_name = patient["name"][0]["family"]
        return {
            "PatientFirstName": patient_given_name,
            "PatientFamilyName": patient_family_name,
            "PatientID": patient_id,
        }


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
