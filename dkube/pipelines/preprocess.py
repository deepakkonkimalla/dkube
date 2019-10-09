from kfp.components._yaml_utils import load_yaml
from kfp.components._yaml_utils import dump_yaml
from kfp import components

SPEC = \
"""
name: dkube-preprocess
description: |
    Component which can be used to perform data preprocessing on Dkube platform.
    Dkube preprocess provides,
    * Ability to orchestrate and run custom containers.
    * Renders utilization graphs for CPU, Memory.
    * Tags to group related preprocessing jobs.
metadata:
  annotations: {platform: 'Dkube'}
  labels: {platform: 'Dkube', wfid: '{{workflow.uid}}', runid: '{{pod.name}}', stage: 'preprocess'}
inputs:
  - {name: auth_token,      type: String,   optional: false,
     description: 'Required. Dkube authentication token.'}
  - {name: name,            type: String,   optional: false,
     description: 'Required. Name with which a dataset in dkube will be created for the output of a datajob'}
  - {name: container,       type: Dict,     optional: false,
     description: 'Required. Container to use for preprocessing. Format: {"image":<url>, "username":<>, "password":<>}'}
  - {name: program,         type: String,   optional: true,     default: '',
     description: 'Optional. Program imported in Dkube to be run inside container. If not specified container should have entrypoint.'}
  - {name: run_script,      type: String,   optional: true,     default: '',
     description: 'Optional. Script to run the program. If not specified container should have entrypoint.'}
  - {name: datasets,        type: List,     optional: true,     default: '[]',
     description: 'Optional. List of input datasets required for preprocessing. These datasets must be created in Dkube.'}
  - {name: config,          type: String,   optional: true,      default: '',
    description: 'Optional. HP file or configuration data required for training program.
                  Supported inputs - 
                  d3s://<path> - Path to a file in dkube storage.
                  <string> - Inline data'} 
  - {name: envs,            type: List,     optional: true,     default: '[]',
     description: 'Optional. Environments for preprocess program. Exact key value will be made available for the container'}
  - {name: access_url,      type: String,   optional: true,     default: '',
     description: 'Optional. URL at which dkube is accessible, copy paste from the browser of this window. Required for cloud deployments.'}
outputs:
  - {name: rundetails,      description: 'Details of the dkube run'}
  - {name: artifact,        description: 'Identifier in Dkube storage where artifact generated by this component are stored.'}
implementation:
  container:
    image: ocdr/dkubepl:1.4.0
    command: ['dkubepl']
    args: [
      preprocess,
      --accessurl, {inputValue: access_url},
      --token, {inputValue: auth_token},
      --target, {inputValue: name},
      --container, {inputValue: container},
      --script, {inputValue: run_script},
      --program, {inputValue: program},
      --datasets, {inputValue: datasets},
      --config, {inputValue: config},
      --envs, {inputValue: envs},
      --runid, '{{pod.name}}'
    ]
    fileOutputs:
      rundetails:   /tmp/rundetails
      artifact:     /tmp/artifact

"""

def DkubePreprocessOp(name='dkube-preprocess'):
    cdict = load_yaml(SPEC)
    cdict['name'] = name
    cyaml = dump_yaml(cdict)

    return components.load_component_from_text(cyaml)
