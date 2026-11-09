"""Freeze a compiled full split and prepared case, with source and binary identity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def freeze(build,template,output):
    build,template,output=map(Path,(build,template,output))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    binary=build/'source/jorek_model303'
    if not binary.is_file():raise FileNotFoundError(binary)
    protocol=json.loads((build/'protocol.json').read_text())
    protocol['supported_schemes']=['euler','cnab2','sbdf2']
    protocol['scheme']='native temporal/constraint solve; frozen L implicit; centered nonlinear volume forcing explicit'
    protocol['source_sha256']={name:sha(build/'source'/name) for name in
                              ('models/model303/mod_elt_matrix_fft.f90','jorek2_main.f90')}
    protocol['binary_sha256']=sha(binary)
    protocol['patched_sha256']=protocol['source_sha256']['models/model303/mod_elt_matrix_fft.f90']
    shutil.copytree(template,output)
    (output/'jorek.exe').chmod(0o755);shutil.copyfile(binary,output/'jorek.exe')
    (output/'full_split_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    manifest=json.loads((output/'case.json').read_text())
    manifest['snapshot_hashes']['jorek.exe']=sha(binary)
    manifest['snapshot_hashes']['full_split_protocol.json']=sha(output/'full_split_protocol.json')
    manifest['executable_sha256']=sha(binary)
    manifest['files']['jorek.exe']={'source':str(binary.resolve()),'sha256':sha(binary)}
    manifest['full_split_protocol']='full_split_protocol.json'
    (output/'case.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();freeze(a.build,a.template,a.output)
