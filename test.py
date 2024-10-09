from typing import Any
from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.patient import Patient
import pathlib
import ray


def parse_bundle_entries(path: dict[str, Any]):
    file = path["path"]
    bundle = Bundle.parse_file(file)

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
                "address": {
                    "line": patient.address[0].line[0],
                    "city": patient.address[0].city,
                    "state": patient.address[0].state,
                    "postalCode": patient.address[0].postalCode,
                    "country": patient.address[0].country,
                }
                if patient.address
                else None,
                "phone": patient.telecom[0].value if patient.telecom else None,
            }
            return patient_data


def create_ds():
    # create our bundle ds
    BUNDLE_DIR = pathlib.Path("./fhir/")
    ds = ray.data.from_items([{"path": path} for path in BUNDLE_DIR.rglob("*")])

    # build patient dataset
    patient_ds = ds.map(parse_bundle_entries)
    patient_ds.write_parquet("./patient_ds/")


create_ds()
