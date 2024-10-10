from typing import Any
import pathlib
import glob
import ray
import re
import os
import json


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
        output += f"{entry} {flattened_json[entry]}. "
    return output


def flatten_bundle(bundle_path):
    file_name = bundle_path.stem
    with open(bundle_path) as raw:
        bundle = json.load(raw)

        patient = find_patient(bundle)
        flat_patient = flatten_fhir(patient)

        for i, entry in enumerate(bundle["entry"]):
            flat_entry = flatten_fhir(entry["resource"])
            with open(f"{FLAT_FILE_PATH}/{file_name}_{i}.txt", "w") as out:
                out.write(
                    f"{flat_to_string(flat_patient)}\n{flat_to_string(flat_entry)}"
                )


if not os.path.exists(FLAT_FILE_PATH):
    os.mkdir(FLAT_FILE_PATH)
for file in BUNDLE_DIR.glob("*"):
    flatten_bundle(file)
