import sys

#import logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logger = logging.getLogger(__name__)

from .lib import logger

from .env import *
from .schema import *
from ._types import *
from ._helpers import *

def get_trained_model(name:str, environ=Environment().internal):
    for x in range(1, 10):
        job = get_training_job(environ, name)
        model = job.params.value.details.model

        if model != "": return model

        time.sleep(x)

    DkubeInternalErrorException("trained model not found for job {} for user {}".format(environ.user, name))

def training_job(name:str, environ=Environment().internal):
    return get_training_job(environ, name)
    
def launch_serving_job(name:str, autogenerate=False,
                       environ=Environment().internal,
                       tags=[],
                       container:ContainerImageDetails=ContainerImage.DKUBE_SERVING_CPU,
                       workspace:str='',
                       script:str='',
                       models:list=[],
                       datasets:list=[],
                       envs:dict={},
                       ngpus:int=0):
    assert name != '', "name cannot be empty, with name-auto-generate=True, name will be used as prefix"

    if autogenerate==True:
        name = "{}-{}".format(name, generate_version())

    job = DkubeJob()
    job.name = name

    executor = container.value.executor
    if executor == ExecutorType.Dkube.value:
        assert len(models) != 1, "A model has to be provided for serving"
        launch_dkube_serving_job(name, environ, container, models[0])
    elif executor == ExecutorType.Custom.value:
        launch_custom_serving_job(name, **kwargs)
    else:
        raise NotImplementedError

def launch_custom_serving_job(name:str, env:Environment,
                              tags=[],
                              container:ContainerImageDetails=ContainerImage.DKUBE_SERVING_CPU,
                              workspace:str='',
                              script:str='',
                              models:list=[],
                              datasets:list=[],
                              envs:dict={},
                              ngpus:int=0):
    pass
     
def launch_dkube_serving_job(name:str, env:Environment, container:ContainerImageDetails, model:str):
    #get the model details
    model = get_model(env, model)
    #check if model is deployable
    servable = False
    if model.details.format.value.format == "tensorpb":
        servable = model.details.format.value.servable
    if servable == False:
        raise DkubeModelNotServableException

    #check if model can be deployed for cpu based serving or needs a gpu
    device = model.details.format.value.devices.cpu or model.details.format.value.devices.gpu or "cpu"

    #create the dkube serving job and wait for job to be running
    job = DkubeJob()
    job.name = name
    job.params = JobParams.Serving

    serving = job.params.value
    
    serving.input.device = ServingDevice.from_str(device)
    serving.input.model  = model
    serving.input.owner  = env.user

    create_serving_job(env, job)
    logger.info("User {} Job {} created successfully \n".format(environ.user, job.name))

    state, reason = wait_for_serving_job(environ, name)
    logger.info("User {} Job {} finished with status {}, reason {} \n".format(environ.user, job.name, state, reason))



def launch_training_job(name:str, autogenerate=False,
                        environ=Environment().internal,
                        tags=[],
                        container:ContainerImageDetails=ContainerImage.DKUBE_DS_TF_CPU_1_14,
                        workspace:str='',
                        script:str='',
                        models:list=[],
                        datasets:list=[],
                        template='',
                        hptuning:str='',
                        steps:int=100,
                        batchsize:int=100,
                        epochs:int=1,
                        customkv:dict={},
                        ngpus:int=0,
                        distributeopts:DistributeOpts=DistributeOpts()):
    assert name != '', "name cannot be empty, with name-auto-generate=True, name will be used as prefix"

    if autogenerate==True:
        name = "{}-{}".format(name, generate_version())

    job = DkubeJob()
    job.name = name
    job.params = JobParams.TRAINING
 
    training = job.params.value

    executor = container.value.executor
    training.input.executor.from_str(executor)

    if executor == ExecutorType.Dkube.value:
        training.input.executor.value.framework.value.version = container.value.tag

    if workspace != '':
        training.input.workspace.data.name = '{}:{}'.format(environ.user,workspace)

    training.input.workspace.script = script

    #models = [{'name': '{}:{}'.format(environ.user, model) for model in models]
    #training.input.models = models

    ipdatasets = []
    for dataset in datasets:
        ipdatasets.append({'name':'{}:{}'.format(environ.user, dataset['name']), 'mountpath': dataset['mountpath']})
    training.input.datasets = ipdatasets

    outputs = [{'name': '{}:{}'.format(environ.user, models[0]), 'mountpath': '/opt/dkube/output'}]
    training.input.outputs = outputs

    training.input.tags = tags

    if hptuning != '':
        training.input.hptuning.name = 'hptuning.json'
        training.input.hptuning.body = hptuning

    training.input.hparams.epochs = epochs
    training.input.hparams.steps  = steps
    training.input.hparams.batchsize = batchsize
    if customkv != {}:
        training.input.hparams.customkv  = [customkv]

    training.input.ngpus = ngpus

    training.input.nworkers = distributeopts.workers
    training.input.gpus_override = distributeopts.strategy == DistributionStrategy.DISTRIBUTION_STRATEGY_AUTO

    training.run.template = template
    training.run.group = 'default'

    create_training_job(environ, job)
    logger.info("User {} Job {} created successfully \n".format(environ.user, job.name))

    state, reason = wait_for_training_job(environ, name)
    logger.info("User {} Job {} finished with status {}, reason {} \n".format(environ.user, name, state, reason))



def export_model(fspath:str, name:str, autogenerate=False,
                 environ=Environment().internal, 
                 framework:Framework=Framework.Tensorflow):

    assert fspath != '', "file system path cannot be empty"
    assert name != '', "name cannot be empty, with name-auto-generate=True, name will be used as prefix"

    if autogenerate==True:
        version = ""
        if framework == Framework.Tensorflow:
            #extract version from fspath
            version = get_tfmodel_version(fspath)
        if version == "":
            version = generate_version()
        name = "{}-{}".format(name, version)

    #upload data to dkube storage
    upload_to_dkube(environ, fspath, name)

    logger.info("data uploaded to dkube storage sucessfully")

    #create a model in dkube database
    model = Model()
    model.input.name    = name
    model.input.url     = "/model/{}".format(name)
    model.input.source  = DatumSource.dkube
    model.input.tags    = []
    model.input.remote  = False

    create_model(environ, model) 

    logger.debug("model created successfully in dkube")

