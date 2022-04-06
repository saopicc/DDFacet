#!/usr/bin/env python
from ast import Import
import sys
import os.path

SCHEMA_FILENAME = "ddfacet_stimela_inputs_schema.yaml"
TWEAKS_FILENAME = "ddfacet_stimela_inputs_tweaks.yaml"

try:
    from scabha.cargo import Parameter
    from omegaconf import OmegaConf
except ImportError as exc:
    print(f"""
    This script is for developers only!

    This will regenerate the {SCHEMA_FILENAME} schema file for DDFacet, based on DefaultParset.cfg.
    DDFacet devs are meant to re-run this script as a one-off operation when the default parset changes, 
    and then commit the updated schema file.

    To generate the schema, the scabha2 package is required. Please make a virtualenv where both 
    DDFacet and scabha2 (https://github.com/caracal-pipeline/scabha2) are installed, and run this script 
    within that virtualenv. Currently the import of scabha is failing.
    """)
    sys.exit(1)


from DDFacet.Parset import ReadCFG
from dataclasses import make_dataclass
from typing import Dict

def generate_schema(filename, output_name):

    print(f"Reading {filename}")

    parset = ReadCFG.Parset(filename)

    default_values = parset.value_dict
    all_attrs = parset.attr_dict

    base_config = OmegaConf.create()

    for section in parset.sections:
        section_config = OmegaConf.create()
        for name, value in default_values[section].items():
            if not name.startswith("_"):
                attr = OmegaConf.create(all_attrs[section][name])
                if not getattr(attr, 'nocmdline', None):
                    doc = attr.doc.replace("\n", " ")
                    if value not in (None, ''):
                        doc += f" (Default: {value})"
                    param = dict(
                        info=doc, 
                        required=False)   # no default -- let default parset supply it
#                        default=value, required=False)
                    if hasattr(attr, 'type'):
                        param['dtype'] = attr.type.__name__
                    section_config[name] = param

        # insert into main config
        base_config[section] = section_config

    # # tweak the schemas -- commented out since we don't need it for now
    tweaks = OmegaConf.load(TWEAKS_FILENAME)
    base_config = OmegaConf.merge(base_config, tweaks)    

    OmegaConf.save(base_config, output_name)

    print(f"Saved schema to {output_name}, loading back")

    # read config as structured schema
    structured = make_dataclass("DDFConfig",
        [(name.replace("-", "_"), Dict[str, Parameter]) for name in parset.sections]
    )
    structured = OmegaConf.create(structured)
    new_config = OmegaConf.load(output_name)
    new_config = OmegaConf.merge(structured, new_config)

    OmegaConf.save(new_config, sys.stdout)

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else f"{os.path.dirname(ReadCFG.__file__)}/DefaultParset.cfg"
    output_name = sys.argv[2] if len(sys.argv) > 2 else SCHEMA_FILENAME

    generate_schema(filename, output_name)

    print(f"Successfully regenerated {output_name} from {filename}")
