import requests
import json

def invoke_microservices(request_dict):
    url = f"http://localhost:8000/api/v1"
    api_endpoint = request_dict.get('api_endpoint', None)
    neural_network_name = request_dict.get('neural_network_name', None)
    ngc_api_key = request_dict.get('ngc_api_key', None)
    action_name = request_dict.get('action_name', None)
    storage = request_dict.get('storage', None)
    specs = request_dict.get('specs', None)
    job_id = request_dict.get('job_id', None)
    
    telemetry_opt_out = request_dict.get('telemetry_opt_out', "no")
    use_ngc_staging = request_dict.get('use_ngc_staging', "True")
    tao_api_ui_cookie = request_dict.get('tao_api_ui_cookie', "")
    tao_api_admin_key = request_dict.get('tao_api_admin_key', "")
    tao_api_base_url = request_dict.get('tao_api_base_url', "https://nvidia.com")
    tao_api_status_callback_url = request_dict.get('tao_api_status_callback_url', "https://nvidia.com")
    automl_experiment_number = request_dict.get('automl_experiment_number', "")
    hosted_service_interaction = request_dict.get('hosted_service_interaction', "")


    if api_endpoint == "get_networks":
        response = requests.get(f"{url}/neural_networks")
    elif api_endpoint == "get_actions":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions")
    elif api_endpoint == "list_ptms":
        req_obj = {"ngc_api_key": ngc_api_key}
        response = requests.post(f"{url}/neural_networks/{neural_network_name}/pretrained_models", req_obj)
    elif api_endpoint == "get_schema":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}:schema")
    elif api_endpoint == "post_action":
        req_obj = {"specs": specs,
                   "cloud_metadata": storage,
                   "ngc_api_key": ngc_api_key,
                   "job_id": job_id,
                   "telemetry_opt_out": telemetry_opt_out,
                   "use_ngc_staging": use_ngc_staging,
                   "tao_api_ui_cookie": tao_api_ui_cookie,
                   "tao_api_admin_key": tao_api_admin_key,
                   "tao_api_base_url": tao_api_base_url,
                   "tao_api_status_callback_url": tao_api_status_callback_url,
                   "automl_experiment_number": automl_experiment_number,
                   "hosted_service_interaction": hosted_service_interaction,
                   }
        response = requests.post(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}", data=json.dumps(req_obj))
    elif api_endpoint == "get_jobs":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}:ids")
    elif api_endpoint == "get_job_status":
        response = requests.get(f"{url}/neural_networks/{neural_network_name}/actions/{action_name}/{job_id}")
    
    if response and response.status_code in (200, 201):
        return response.json()
    else:
        raise ValueError(f"{response.json()['error_desc']}" if response.json().get('error_desc') else f"Failed to get execute (Status Code: {response.status_code} : {response.json()})")
