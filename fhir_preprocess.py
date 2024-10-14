import asyncio
import aiofiles
import pathlib
import os
import json
from collections import deque
import time


FLAT_FILE_PATH = "./temp/flat"
BUNDLE_DIR = pathlib.Path("./fhir/")


def flatten_fhir(json):
    out = {}

    def flatten(json_to_flat, name=""):
        # If the Nested key-value
        # pair is of dict type
        if type(json_to_flat) is dict:
            for sub_attrib in json_to_flat:
                flatten(json_to_flat[sub_attrib], name + sub_attrib + " ")

        # If the Nested key-value
        # pair is of list type
        elif type(json_to_flat) is list:
            i = 0
            for sub_json in json_to_flat:
                flatten(sub_json, name + str(i) + " ")
                i += 1
        else:
            out[name[:-1]] = json_to_flat

    flatten(json)
    return out


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


def flat_to_string(flattened_json):
    output = ""
    for entry in flattened_json:
        output += f"{entry} IS {flattened_json[entry]}. "
    return output


async def load_bundle(bundle_path):
    async with aiofiles.open(bundle_path, mode="r") as f:
        content = await f.read()
        return json.loads(content)


async def process_bundle(bundle_path):
    file_name = bundle_path.stem
    bundle = await load_bundle(bundle_path)
    patient = find_patient(bundle)
    flat_patient = flatten_fhir(patient)

    for i, entry in enumerate(bundle["entry"]):
        flat_entry = flatten_fhir(entry["resource"])
        if not os.path.exists(f"{FLAT_FILE_PATH}/{file_name}"):
            os.makedirs(f"{FLAT_FILE_PATH}/{file_name}")
        async with aiofiles.open(
            f"{FLAT_FILE_PATH}/{file_name}/{file_name}_{i}.txt", "w"
        ) as out:
            await out.write(
                f"{flat_to_string(flat_patient)}\n{flat_to_string(flat_entry)}"
            )
    return True


async def process_batch(batch):
    tasks = [process_bundle(file) for file in batch if file.is_file()]
    return await asyncio.gather(*tasks)


async def load_and_flatten_bundles(dir_path: pathlib.Path, batch_size=100):
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
    await load_and_flatten_bundles(BUNDLE_DIR)


if __name__ == "__main__":
    asyncio.run(main())
