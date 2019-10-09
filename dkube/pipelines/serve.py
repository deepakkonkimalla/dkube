from kfp.components._yaml_utils import load_yaml
from kfp.components._yaml_utils import dump_yaml
from kfp import components

SPEC = \
"""
name: dkube-serving
description: |
    Component which can be used to deploy a trained model on Dkube platform.
    Dkube serving provides,
    * Option to deploy with CPU/GPU.
    * A web server in the front and all the required infra to access the server.
    * Deployed as microserice. Serving URL is provided for any other application logic to consume the model.
    * Attempts to decode and present some abstract information about the model.
metadata:
  annotations: {platform: 'Dkube'}
  labels: {platform: 'Dkube', wfid: '{{workflow.uid}}', runid: '{{pod.name}}', stage: 'serving'}
inputs:
  - {name: auth_token,      type: String,   optional: false,
     description: 'Required. Dkube authentication token.'}
  - {name: model,           type: String,   optional: false,
     description: 'Required. Trained model in Dkube which is to be deployed for serving.'}
  - {name: device,          type: String,   optional: true,     default: 'cpu',
     description: 'Optional. Device to use for serving - allowed values, gpu/cpu/auto.'}
  - {name: access_url,      type: String,   optional: true,     default: '',
     description: 'Optional. URL at which dkube is accessible, copy paste from the browser of this window. Required for cloud deployments.'}
outputs:
  - {name: rundetails,       description: 'Details of the dkube run'}
  - {name: servingurl,       description: 'URL at which the serving web server is accessible.'}
implementation:
  container:
    image: ocdr/dkubepl:1.4.0
    command: ['dkubepl']
    args: [
      serving,
      --accessurl, {inputValue: access_url},
      --token, {inputValue: auth_token},
      --model, {inputValue: model},
      --device, {inputValue: device},
      --runid, '{{pod.name}}'
    ]
    fileOutputs:
      rundetails: /tmp/rundetails
      servingurl: /tmp/servingurl
"""

def DkubeServeOp(name='dkube-serving'):
    cdict = load_yaml(SPEC)
    cdict['name'] = name
    cyaml = dump_yaml(cdict)

    return components.load_component_from_text(cyaml)
