from pprint import pprint
from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.patient import Patient
import pathlib


def parse_bundle_entries(bundle: Bundle, patients: list):
    for entry in bundle.entry:
        if isinstance(entry.resource, Patient):
            patient = entry.resource
            patient_data = {
                "id": patient.id,
                "name": patient.name[0].given[0] if patient.name else None,
                "family": patient.name[0].family if patient.name else None,
                "gender": patient.gender,
                "birthDate": patient.birthDate.isoformat()
                if patient.birthDate
                else None,
                "address": patient.address[0].text if patient.address else None,
                "phone": patient.telecom[0].value if patient.telecom else None,
            }
            patients.append(patient_data)


patients = []


files = pathlib.Path("fhir").glob("*")
for file in files:
    print(f"Parsing {file}...")
    bundle = Bundle.parse_file(file)
    parse_bundle_entries(bundle, patients)

pprint(patients)
