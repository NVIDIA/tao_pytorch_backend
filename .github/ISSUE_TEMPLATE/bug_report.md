---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Describe the bug**

A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**

Please list *minimal* steps or code snippet for us to be able to reproduce the bug.

A  helpful guide on on how to craft a minimal bug report  http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports. 


**Expected behavior**

A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**

 - Environment location: [Bare-metal, Docker, Cloud(specify cloud provider - AWS, Azure, GCP, Collab)]
 - Method of TAO Toolkit Installation install: [docker container, launcher, pip install or from source]. Please specify exact commands you used to install.
 - If method of install is [Docker], provide `docker pull` & `docker run` commands used
 - If method of install in [Launcher], provide the output of the `tao info --verbose` command and `pip show nvidia-tao` command.

**Environment details**

If NVIDIA docker image is used you don't need to specify these.
Otherwise, please provide:
- OS version
- TensorFlow version
- Python version
- CUDA version
- CUDNN version
- DALI version
- GPU Driver

**Additional context**

Add any other context about the problem here.
Example: GPU model