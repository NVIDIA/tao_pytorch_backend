# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API modules defining schemas and endpoints"""

import sys
import json
import requests
import shutil
import uuid

import pycuda.autoinit

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from nvidia_tao_core.api_utils import (
    microservice_utils,
    module_utils,
    ngc_utils,
    process_queue,
    json_schema_validation,
    dataclass2json_converter as dataclasses_utils
)
from flask import Flask, jsonify, make_response, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from marshmallow import Schema, fields, exceptions, validate
from marshmallow_enum import EnumField, Enum
from threading import Thread
from werkzeug.exceptions import HTTPException


flask_plugin = FlaskPlugin()
marshmallow_plugin = MarshmallowPlugin()

#
# Start processing thread
#
processing_thread = Thread(target=process_queue.process_queue)
processing_thread.daemon = True
processing_thread.start()

import json

#
# Utils
#
def sys_int_format():
    """Get integer format based on system."""
    if sys.maxsize > 2**31 - 1:
        return "int64"
    return "int32"


def is_pvc_space_free(threshold_bytes):
    """Check if pvc has required free space"""
    _, _, free_space = shutil.disk_usage('/')
    return free_space > threshold_bytes, free_space


def disk_space_check(f):
    """Decorator to check disk space for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = kwargs.get('user_id', '')
        threshold_bytes = 100 * 1024 * 1024

        pvc_free_space, pvc_free_bytes = is_pvc_space_free(threshold_bytes)
        msg = f"PVC free space remaining is {pvc_free_bytes} bytes which is less than {threshold_bytes} bytes"
        if not pvc_free_space:
            return make_response(jsonify({'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'}), 500)

        return f(*args, **kwargs)

    return decorated_function


def is_valid_uuid(uuid_to_test, version=4):
    """Check if uuid_to_test is a valid UUID"""
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


#
# Create an APISpec
#
spec = APISpec(
    title='NVIDIA TAO DNN API',
    version='v5.3.0',
    openapi_version='3.0.3',
    info={"description": 'NVIDIA TAO DNN API document'},
    tags=[
        {"name": 'NEURAL NETWORKS', "description": 'Endpoints related to Neural Network Architectures'},
        {"name": 'NVCF', "description": 'Endpoints related to NVIDIA Cloud Functions'},
        {"name": "nSpectId", "description": "NSPECT-76DN-OP7I", "externalDocs": {"url": "https://nspect.nvidia.com/review?id=NSPECT-76DN-OP7I"}}
    ],
    plugins=[flask_plugin, marshmallow_plugin],
)


spec.components.header("X-RateLimit-Limit", {
    "description": "The number of allowed requests in the current period",
    "schema": {
        "type": "integer",
        "format": sys_int_format(),
        "minimum": -sys.maxsize - 1,
        "maximum": sys.maxsize,
    }
})
spec.components.security_scheme("api-key", {"type": "apiKey", "in": "header", "name": "ngc_key"})



#
# Enum stuff for APISpecs
#
def enum_to_properties(self, field, **kwargs):
    """
    Add an OpenAPI extension for marshmallow_enum.EnumField instances
    """
    if isinstance(field, EnumField):
        return {'type': 'string', 'enum': [m.name for m in field.enum]}
    return {}


marshmallow_plugin.converter.add_attribute_function(enum_to_properties)


#
# Global schemas and enums
#
class ErrorRspSchema(Schema):
    """Class defining error response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    error = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    error_code = fields.Int(validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize), format=sys_int_format())


class JobStatusEnum(Enum):
    """Class defining job status enum"""

    Done = 'Done'
    Processing = 'Processing'
    Error = 'Error'
    Pending = 'Pending'
    Canceled = 'Canceled'


class ActionsEnum(Enum):
    """Class defining Actions enum"""

    train = 'train'
    inference = 'inference'
    export = 'export'
    evaluate = 'evaluate'
    distill = 'distill'
    convert = 'convert'


class AllowedDockerEnvVariables(Enum):
    """Allowed docker environment variables while launching DNN containers"""

    WANDB_API_KEY = "WANDB_API_KEY"
    WANDB_BASE_URL = "WANDB_BASE_URL"
    WANDB_USERNAME = "WANDB_USERNAME"
    WANDB_ENTITY = "WANDB_ENTITY"
    WANDB_PROJECT = "WANDB_PROJECT"
    CLEARML_WEB_HOST = "CLEARML_WEB_HOST"
    CLEARML_API_HOST = "CLEARML_API_HOST"
    CLEARML_FILES_HOST = "CLEARML_FILES_HOST"
    CLEARML_API_ACCESS_KEY = "CLEARML_API_ACCESS_KEY"
    CLEARML_API_SECRET_KEY = "CLEARML_API_SECRET_KEY"


class MLOpsEnum(Enum):
    """Class defining MLOps enum"""

    ClearML = 'ClearML'
    WeightAndBiases = 'WeightAndBiases'


#
# Flask app
#
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10000/hour"],
    headers_enabled=True,
    storage_uri="memory://",
)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


#
# NETWORKS API
#
class GetNetworksRspSchema(Schema):
    """Class defining Get Networks Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    networks = fields.List(fields.Str(format="uuid", validate=validate.Length(max=36)), allow_none=False, validate=validate.Length(max=sys.maxsize))


class GetActionsRspSchema(Schema):
    """Class defining Get Actions Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    actions = fields.List(EnumField(ActionsEnum), allow_none=False, validate=validate.Length(max=sys.maxsize))


class GetPtmsRspSchema(Schema):
    """Class defining Get PTMs Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    ptms = fields.List(fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048)), allow_none=False, validate=validate.Length(max=sys.maxsize))


class GetJobsRspSchema(Schema):
    """Class defining Get Jobs Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    jobs = fields.List(fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True))


class GetJobStatusRspSchema(Schema):
    """Class defining Get Job Status Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    status = EnumField(JobStatusEnum)


class StorageSchema(Schema):
    """Class defining Storage schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    uri = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    path = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)


class PtmReqSchema(Schema):
    """Class defining PTM request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class PtmSchema(Schema):
    """Class defining PTM schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    uri = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)


class CallbackSchema(Schema):
    """Class defining Callback schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    uri = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)


class PostActionReqSchema(Schema):
    """Class defining Post Action Request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    specs = fields.Raw()
    storage = fields.Raw()
    ptm = fields.Nested(PtmSchema, allow_none=True)
    callback = fields.Nested(CallbackSchema, allow_none=True)


class PostActionRspSchema(Schema):
    """Class defining Post Action Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class NVCFReqSchema(Schema):
    """Class defining NVCF request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    api_endpoint = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    neural_network_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    action_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    storage = fields.Raw()
    ptm = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    callback = fields.Nested(CallbackSchema, allow_none=True)
    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)

    telemetry_opt_out = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    use_ngc_staging = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    nvcf_helm = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    tao_api_ui_cookie = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    tao_api_admin_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    tao_api_base_url = fields.URL(validate=fields.validate.Length(max=2048))
    tao_api_status_callback_url = fields.URL(validate=fields.validate.Length(max=2048))
    automl_experiment_number = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    hosted_service_interaction = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    docker_env_vars = fields.Dict(keys=EnumField(AllowedDockerEnvVariables), values=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True))

@app.route('/api/v1/neural_networks', methods=['GET'])
@disk_space_check
def get_networks():
    """List supported neutral networks.
    ---
    get:
      tags:
      - NEURAL NETWORKS
      summary: List supported neural networks
      description: Returns the supported neural networks
      responses:
        200:
          description: Retuned the list of supported neural networks
          content:
            application/json:
              schema: GetNetworksRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    data = { 'networks': list(module_utils.get_entry_points()) }
    schema = GetNetworksRspSchema()
    schema_dict = schema.dump(schema.load(data))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/neural_networks/<neural_network_name>/actions', methods=['GET'])
@disk_space_check
def get_actions(neural_network_name):
    """List supported actions for a given neutral network.
    ---
    get:
      tags:
      - NEURAL NETWORKS
      summary: List supported actions for a given neural network
      description: Returns the supported actions for a given neural network
      parameters:
      - name: neural_network_name
        in: path
        description: Neural Network Name
        schema:
          type: string
          enum: [ "action_recognition", "bevfusion", "centerpose", "classification_pyt", "deformable_detr", "dino", "grounding_dino", "mal", "mask2former", "mask_grounding_dino", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "visual_changenet" ]
      responses:
        200:
          description: Retuned the list of supported actions for a given neural network
          content:
            application/json:
              schema: GetActionsRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    _, actions = module_utils.get_neural_network_actions(neural_network_name)

    if not actions:
        metadata = {"error": "Invalid Neural Network Name", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        
    data = { 'actions': list(actions.keys())}
    print("app.py :: get_actions :: data :: ", data)
    schema = GetActionsRspSchema()
    schema_dict = schema.dump(schema.load(data))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/neural_networks/<neural_network_name>/pretrained_models', methods=['POST'])
@disk_space_check
def list_ptms(neural_network_name):
    """List supported pretrained models for a given neutral network.
    ---
    post:
      tags:
      - NEURAL NETWORKS
      summary: List supported pretrained models for a given neural network
      description: Returns the supported pretrained models for a given neural network
      parameters:
      - name: neural_network_name
        in: path
        description: Neural Network Name
        schema:
          type: string
          enum: [ "action_recognition", "bevfusion", "centerpose", "classification_pyt", "deformable_detr", "dino", "grounding_dino", "mal", "mask2former", "mask_grounding_dino", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "visual_changenet" ]
      security:
        - api-key: []
      requestBody:
        content:
          application/json:
            schema: PtmReqSchema
        description: Login request with NGC key
        required: true
      responses:
        200:
          description: Retuned the list of supported pretrained models for a given neural network
          content:
            application/json:
              schema: GetPtmsRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = PtmReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    key = request_dict.get('ngc_key', 'invalid_key')

    try:
        ngc_token = ngc_utils.get_ngc_token(key)
    except Exception as err:
        metadata = {"error": "Unauthorized: " + err, "error_code": 7}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 401)

    try:
        data = { 'ptms': list(ngc_utils.get_model_info(ngc_token, neural_network_name)) }
        schema = GetPtmsRspSchema()
        schema_dict = schema.dump(schema.load(data))
        return make_response(jsonify(schema_dict), 200)
    except Exception as err:
        metadata = {"error": f"{err}", "error_code": 8}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)


@app.route('/api/v1/neural_networks/<neural_network_name>/actions/<action_name>:schema', methods=['GET'])
@disk_space_check
def get_schema(neural_network_name, action_name):
    """Get action specs schema.
    ---
    get:
      tags:
      - NEURAL NETWORKS
      summary: Get action specs schema
      description: Returns the schema for a given neural network action specs
      parameters:
      - name: neural_network_name
        in: path
        description: Neural Network Name
        schema:
          type: string
          enum: [ "action_recognition", "bevfusion", "centerpose", "classification_pyt", "deformable_detr", "dino", "grounding_dino", "mal", "mask2former", "mask_grounding_dino", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "visual_changenet" ]
      - name: action_name
        in: path
        description: Action Name
        schema:
          type: string
          enum: [ "convert", "distill", "evaluate", "export", "inference", "train" ]
      responses:
        200:
          description: Retuned the json-schema object for a given neural network action specs
          content:
            application/json:
              schema:
                type: object
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    ep_mappings = module_utils.get_entry_point_module_mapping()
    if not neural_network_name or neural_network_name not in ep_mappings.keys():
        metadata = {"error": "Invalid Neural Network Name", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    
    _, actions = module_utils.get_neural_network_actions(neural_network_name)
    if action_name not in actions:
        metadata = {"error": "Invalid Action Name", "error_code": 2}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    
    try:
        imported_module = dataclasses_utils.import_module_from_path(f"nvidia_tao_core.config.{neural_network_name}.default_config")
        if neural_network_name == "bevfusion" and action_name == "dataset_convert":
            expConfig = imported_module.BEVFusionDataConvertExpConfig()
        else:
            expConfig = imported_module.ExperimentConfig()
        json_with_meta_config = dataclasses_utils.dataclass_to_json(expConfig)
        json_schema = dataclasses_utils.create_json_schema(json_with_meta_config)
        print("json_schema :: ", json_schema)
        return make_response(json_schema, 200)
    except Exception as err:
        metadata = {"error": f"{err}", "error_code": 2}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)


@app.route('/api/v1/neural_networks/<neural_network_name>/actions/<action_name>', methods=['POST'])
@disk_space_check
def post_action(neural_network_name, action_name):
    """Run an action.
    ---
    post:
      tags:
      - NEURAL NETWORKS
      summary: Run an action
      description: Returns the job ID for requested action
      parameters:
      - name: neural_network_name
        in: path
        description: Neural Network Name
        schema:
          type: string
          enum: [ "action_recognition", "bevfusion", "centerpose", "classification_pyt", "deformable_detr", "dino", "grounding_dino", "mal", "mask2former", "mask_grounding_dino", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "visual_changenet" ]
      - name: action_name
        in: path
        description: Action Name
        schema:
          type: string
          enum: [ "convert", "distill", "evaluate", "export", "inference", "train" ]
      requestBody:
        content:
          application/json:
            schema: PostActionReqSchema
        description: Run action
        required: true
      responses:
        200:
          description: Retuned the job ID for requested action
          content:
            application/json:
              schema: PostActionRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Module and action validations
    module, actions = module_utils.get_neural_network_actions(neural_network_name)
    if not module:
        metadata = {"error": "Invalid Neural Network Name", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    if action_name not in actions:
        metadata = {"error": "Invalid Action Name", "error_code": 2}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    # Obtaining the JSON schema
    data = request.get_json(force=True)
    request_json_schema = data["specs"]

    # Removing None and empty string values from the JSON schema
    request_json_schema = dataclasses_utils.remove_none_empty_fields(request_json_schema)
    
    if request_json_schema is None:
        metadata = {"error": "No JSON data provided", "error_code": 3}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400) 
    
    try:
        # Fetching the JSON schema
        imported_module = dataclasses_utils.import_module_from_path(f"nvidia_tao_core.config.{neural_network_name}.default_config")
        if neural_network_name == "bevfusion" and action_name == "dataset_convert":
            expConfig = imported_module.BEVFusionDataConvertExpConfig()
        else:
            expConfig = imported_module.ExperimentConfig()
        json_with_meta_config = dataclasses_utils.dataclass_to_json(expConfig)
        json_schema = dataclasses_utils.create_json_schema(json_with_meta_config)
        
        validation_status = None
        print("json_schema :: ")
        print(json.dumps(json_schema, indent=4))

        print("Before validation :: ")
        # Validating the JSON schema
        validation_status = json_schema_validation.validate_jsonschema(request_json_schema, json_schema["properties"])
        print("After validation :: ")
        print("validation_status :: ", validation_status)
        if validation_status:
            metadata = {"error": validation_status, "error_code": 4}
            print(metadata)
            schema = ErrorRspSchema()
            print(schema)
            return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    except Exception as err:
        metadata = {"error": "Unexpected error encountered while processing the JSON schema", "error_code": 4}
        import traceback
        print(traceback.format_exc())
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    
    # Creating a Job ID for a valid request
    job_id = data.get("job_id")
    if not job_id:
        job_id = str(uuid.uuid4())

    # Adding the request to a queue
    process_queue.queue.append({'job_id': job_id, 'data': data, 'neural_network_name': neural_network_name, 'action': action_name, 'status': 'Pending'})

    # Returning the response
    data = { 'job_id': job_id }
    schema = PostActionRspSchema()
    schema_dict = schema.dump(schema.load(data))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/neural_networks/<neural_network_name>/actions/<action_name>:ids', methods=['GET'])
@disk_space_check
def get_jobs(neural_network_name, action_name):
    """List jobs for a given action.
    ---
    get:
      tags:
      - NEURAL NETWORKS
      summary: List jobs for a given action
      description: Returns the list of job IDs for a given action
      parameters:
      - name: neural_network_name
        in: path
        description: Neural Network Name
        schema:
          type: string
          enum: [ "action_recognition", "bevfusion", "centerpose", "classification_pyt", "deformable_detr", "dino", "grounding_dino", "mal", "mask2former", "mask_grounding_dino", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "visual_changenet" ]
      - name: action_name
        in: path
        description: Action Name
        schema:
          type: string
          enum: [ "convert", "distill", "evaluate", "export", "inference", "train" ]
      responses:
        200:
          description: Retuned the list of job IDs for a given network action
          content:
            application/json:
              schema: GetJobsRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    data = { 'jobs': ['f51193dc-0606-48dd-a5a9-7ab1949bfd4c'] }
    schema = GetJobsRspSchema()
    schema_dict = schema.dump(schema.load(data))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/neural_networks/<neural_network_name>/actions/<action_name>/<job_id>', methods=['GET'])
@disk_space_check
def get_job_status(neural_network_name, action_name, job_id):
    """Get job status.
    ---
    get:
      tags:
      - NEURAL NETWORKS
      summary: Get job status
      description: Returns the job status
      parameters:
      - name: neural_network_name
        in: path
        description: Neural Network Name
        schema:
          type: string
          enum: [ "action_recognition", "bevfusion", "centerpose", "classification_pyt", "deformable_detr", "dino", "grounding_dino", "mal", "mask2former", "mask_grounding_dino", "ml_recog", "ocdnet", "ocrnet", "optical_inspection", "pointpillars", "pose_classification", "re_identification", "segformer", "visual_changenet" ]
      - name: action_name
        in: path
        description: Action Name
        schema:
          type: string
          enum: [ "convert", "distill", "evaluate", "export", "inference", "train" ]
      - name: job_id
        in: path
        description: Job ID
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Retuned the job status
          content:
            application/json:
              schema: GetJobStatusRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job ID not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    if not is_valid_uuid(job_id):
        metadata = {"error": "Invalid Job ID Format", "error_code": 5}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    
    job_details = None
    
    # Checking main queue for job status
    for job in process_queue.queue:
        if job['job_id'] == job_id:
            job_details = job
    
    # Checking processing jobs for job status
    if not job_details:
        for job in process_queue.processing_jobs:
            if job['job_id'] == job_id:
                job_details = job
    
    # Checking completed jobs for job status
    if not job_details:
        for job in process_queue.completed_jobs:
            if job['job_id'] == job_id:
                job_details = job

    # If no job is found
    if not job_details:
        metadata = {"error": "Job ID Not Present", "error_code": 6}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
        
    # Module and action validations
    module, actions = job_details['neural_network_name'], job_details['action']
    if not module:
        metadata = {"error": "Invalid Neural Network Name", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    if action_name not in actions:
        metadata = {"error": "Invalid Action Name", "error_code": 2}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
                
    data = { 'status': job['status'] }
    schema = GetJobStatusRspSchema()
    schema_dict = schema.dump(schema.load(data))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/nvcf', methods=['POST'])
@disk_space_check
def post_nvcf_action():
    """Run an action on NVCF.
    ---
    post:
      tags:
      - NVCF
      summary: Run an action on NVCF
      description: Executes the supported action on NVCF
      requestBody:
        content:
          application/json:
            schema: NVCFReqSchema
        description: Run an action on NVCF
        required: true
      responses:
        200:
          description: Executes the supported action on NVCF
          content:
            application/json:
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = NVCFReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    try:
        resp = microservice_utils.invoke_microservices(request_dict)
        return make_response(jsonify(resp), 200)
    except Exception as err:
        import traceback
        print(traceback.format_exc())
        metadata = {"error": str(err), "error_code": 9}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

#
# HEALTH API
#
@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """api health endpoint"""
    return make_response(jsonify(['liveness', 'readiness']))


@app.route('/api/v1/health/liveness', methods=['GET'])
@disk_space_check
def liveness():
    """api liveness endpoint"""
    live_state = True
    if live_state:
        return make_response(jsonify("OK"), 200)
    return make_response(jsonify("Error"), 400)


@app.route('/api/v1/health/readiness', methods=['GET'])
@disk_space_check
def readiness():
    """api readiness endpoint"""
    ready_state = True
    if ready_state:
        return make_response(jsonify("OK"), 200)
    return make_response(jsonify("Error"), 400)


#
# BASIC API
#
@app.route('/', methods=['GET'])
def root():
    """api root endpoint"""
    return make_response(jsonify(['api', 'openapi.yaml', 'openapi.json', 'redoc', 'swagger']))


@app.route('/api', methods=['GET'])
def version_list():
    """version list endpoint"""
    return make_response(jsonify(['v1']))


@app.route('/api/v1', methods=['GET'])
def version_v1():
    """version endpoint"""
    return make_response(jsonify(['neural_networks']))


@app.route('/openapi.yaml', methods=['GET'])
def openapi_yaml():
    """openapi_yaml endpoint"""
    r = make_response(spec.to_yaml())
    r.mimetype = 'text/x-yaml'
    return r


@app.route('/openapi.json', methods=['GET'])
def openapi_json():
    """openapi_json endpoint"""
    r = make_response(jsonify(spec.to_dict()))
    r.mimetype = 'application/json'
    return r


@app.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('redoc.html')


@app.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    return render_template('swagger.html')


#
# End of APIs
#


with app.test_request_context():
    spec.path(view=get_networks)
    spec.path(view=get_actions)
    spec.path(view=list_ptms)
    spec.path(view=get_schema)
    spec.path(view=post_action)
    spec.path(view=get_jobs)
    spec.path(view=get_job_status)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008, debug=True)

