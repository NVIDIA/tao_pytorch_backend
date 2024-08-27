import ast
import json
import requests

MODEL_CACHE = None

def get_ngc_token(ngc_api_key: str = "", org: str = "nvidia", team: str = "tao"):
    """Authenticate to NGC"""
    url = "https://authn.nvidia.com/token"
    params = {"service": "ngc", "scope": "group/ngc"}
    if org:
        params["scope"] = f"group/ngc:{org}"
        if team:
            params["scope"] += f"&group/ngc:{org}/{team}"
    headers = {"Accept": "application/json"}
    auth = ("$oauthtoken", ngc_api_key)
    response = requests.get(url, headers=headers, auth=auth, params=params)
    if response.status_code != 200:
        raise ValueError(f"Credentials error: Invalid NGC_API_KEY")
    return response.json()["token"]


def get_model_metadata_from_ngc(ngc_token: str, org: str, team: str, model_name: str, model_version: str, file: str = ""):
    """Get model info from NGC"""
    if team:
        url = f"https://api.ngc.nvidia.com/v2/org/{org}/team/{team}/models/{model_name}/versions/{model_version}"
    else:
        url = f"https://api.ngc.nvidia.com/v2/org/{org}/models/{model_name}/versions/{model_version}"
    if file:
        url = f"{url}/files/{file}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ngc_token}"}
    response = requests.get(url, headers=headers, params={"page-size": 1000})
    if response.status_code != 200:
        raise ValueError(f"Failed to get model info for {model_name}:{model_version} ({response.status_code} {response.reason})")
    return response.json()


def get_model_info_from_ngc(ngc_token: str, org: str, team: str):
    """Get model info from NGC"""
    # Create the query to filter models and the required return fields
    url = "https://api.ngc.nvidia.com/v2/search/resources/MODEL"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ngc_token}"}
    query = f"resourceId:{org}/{team + '/' if team else ''}*"
    params = {"q": json.dumps({"query": query})}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to get model info ({response.status_code} {response.reason})")

    # Obtaining the list of models
    model_info = {}
    for page_number in range(response.json()["resultPageTotal"]):
        params = {"q": json.dumps({"fields": ["resourceId", "name", "displayName", "orgName", "teamName"], "page": page_number, "query": query})}
        response = requests.get(url, headers=headers, params=params)
        results = response.json()["results"]
    
        # Iterate through the list of models
        for model_list in results:
            for model in model_list["resources"]:
                try:
                    model_metadata = get_model_metadata_from_ngc(ngc_token, model["orgName"], model.get("teamName", ""), model["name"], "")
                    if "modelVersions" in model_metadata:
                        for model_version in model_metadata["modelVersions"]:
                            if "customMetrics" in model_version:
                                ngc_path = f'ngc://{model["resourceId"]}:{model_version["versionId"]}'
                                for customMetrics in model_version["customMetrics"]:
                                    endpoints = []
                                    for key_value in customMetrics.get("attributes", []):
                                        if key_value["key"] == "endpoints":
                                            try:
                                                endpoints = ast.literal_eval(key_value["value"])
                                            except (SyntaxError, ValueError):
                                                print(f"{key_value} not loadable by `ast.literal_eval`.")
                                    for endpoint in endpoints:
                                        if endpoint in model_info:
                                            model_info[endpoint].append(ngc_path)
                                        else:
                                            model_info[endpoint] = [ngc_path]
                except ValueError as e:
                    print(e)

    # Returning the list of models
    return model_info


def get_model_info(ngc_token: str, endpoint: str, org: str = "nvidia", team: str = "tao"):
    """Get model info"""
    global MODEL_CACHE
    if not MODEL_CACHE:
        MODEL_CACHE = get_model_info_from_ngc(ngc_token, org, team)

    if endpoint in MODEL_CACHE:
        return MODEL_CACHE[endpoint]
    else:
        raise ValueError(f"No PTM found for the neural network name {endpoint}")
