import os
import threading
import traceback
import time
import yaml
from nvidia_tao_pytorch.api.api_utils import module_utils
from nvidia_tao_core.cloud_handlers.utils import download_files_from_spec, get_results_cloud_data, monitor_and_upload
from nvidia_tao_pytorch.core.entrypoint import launch
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

# Initialize empty queue, processing jobs, and completed jobs lists
queue = []
processing_jobs = []
completed_jobs = []


# Function to process jobs from the queue
def process_queue():
    while True:
        if queue:
            job = queue.pop(0)  # Take the first job from the queue
            processing_jobs.append(job)  # Add job to processing list
            job['status'] = 'Processing'
            cloud_storage = None
            is_completed = None
            exit_event = None
            upload_thread = None
            
            try: 
                telemetry_opt_out = job["data"].get('telemetry_opt_out', "no")
                use_ngc_staging = job["data"].get('use_ngc_staging', "False")
                tao_api_ui_cookie = job["data"].get('tao_api_ui_cookie', "")
                tao_api_admin_key = job["data"].get('tao_api_admin_key', "")
                tao_api_base_url = job["data"].get('tao_api_base_url', "")
                tao_api_status_callback_url = job["data"].get('tao_api_status_callback_url', "")
                automl_experiment_number = job["data"].get('automl_experiment_number', "")
                hosted_service_interaction = job["data"].get('hosted_service_interaction', "")

                os.environ["CLOUD_BASED"] = hosted_service_interaction
                os.environ["TELEMETRY_OPT_OUT"] = telemetry_opt_out
                os.environ["TAO_ADMIN_KEY"] = tao_api_admin_key
                os.environ["TAO_API_SERVER"] = tao_api_base_url
                os.environ["TAO_LOGGING_SERVER_URL"] = tao_api_status_callback_url
                os.environ["AUTOML_EXPERIMENT_NUMBER"] = automl_experiment_number
                os.environ["JOB_ID"] = job["job_id"]

                # Obtaining the CloudStroage instance to save the results
                cloud_storage, specs = get_results_cloud_data(job["data"].get("cloud_metadata"), job["data"]["specs"], f'/results/{job["job_id"]}')
                os.makedirs(specs["results_dir"], exist_ok=True)

                # Download cloud files
                download_files_from_spec(cloud_data=job["data"].get("cloud_metadata"),
                                         data=specs,
                                         job_id=job["job_id"],
                                         network_arch=job["neural_network_name"],
                                         ngc_api_key=job["data"].get("ngc_api_key"),
                                         tao_api_ui_cookie=tao_api_ui_cookie,
                                         use_ngc_staging=use_ngc_staging,
                                        )

                # Starting the thread to update the results to the cloud
                if cloud_storage:
                    exit_event = threading.Event()
                    upload_thread = threading.Thread(target=monitor_and_upload, args=(specs["results_dir"], cloud_storage, exit_event), daemon=True)
                    upload_thread.start()

                # Creating the Spec
                with open(f'{specs["results_dir"]}/spec.yaml', 'w+') as yaml_file:
                    yaml.dump(specs, yaml_file, default_flow_style=False)

                # Creating the request object and launching the action
                args = {"subtask": job["action"],
                        "experiment_spec_file": f'{specs["results_dir"]}/spec.yaml', 
                        "results_dir": specs["results_dir"],
                        }
                _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                is_completed = launch(args, "", actions, job["neural_network_name"])

            except Exception:
                print(traceback.format_exc())
                job['status'] = 'Error'
            finally:
                # Stopping the threads
                if cloud_storage:
                    if exit_event is not None:
                        exit_event.set()
                    if upload_thread is not None:
                        upload_thread.join()

            # Updating the job status
            try:
                if is_completed:
                    job['status'] = 'Done'
                    status_logging.get_status_logger().write(
                        message=f"{job['action']} action completed successfully for {job['neural_network_name']}",
                        status_level=status_logging.Status.SUCCESS
                    )
                else:
                    job['status'] = 'Error'
                    status_logging.get_status_logger().write(
                        message=f"{job['action']} action failed for {job['neural_network_name']}",
                        status_level=status_logging.Status.FAILURE
                    )
            except Exception:
                print(traceback.format_exc())
                job['status'] = 'Error'

            processing_jobs.remove(job)  # Remove job from processing list
            completed_jobs.append(job) # Add job to completed list
        time.sleep(1)  # Check queue every second
